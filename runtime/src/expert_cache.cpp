#include "expert/runtime/expert_cache.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace expert::runtime {
namespace {

constexpr std::size_t kStateCount = 6;

bool valid_budget(const TierBudget& budget) noexcept {
  return budget.capacity_bytes != 0 && budget.high_watermark_bytes != 0 &&
         budget.low_watermark_bytes <= budget.high_watermark_bytes &&
         budget.high_watermark_bytes <= budget.capacity_bytes;
}

bool same_record(const PayloadRecord& left, const PayloadRecord& right) {
  return left.path == right.path && left.record_offset == right.record_offset &&
         left.stored_bytes == right.stored_bytes &&
         left.decoded_bytes == right.decoded_bytes &&
         left.header_bytes == right.header_bytes &&
         left.alignment == right.alignment &&
         constant_time_equal(left.payload_sha256, right.payload_sha256);
}

}  // namespace

struct ExpertCacheCore final : public std::enable_shared_from_this<ExpertCacheCore> {
  struct Waiter final {
    explicit Waiter(std::uint64_t waiter_id) : id(waiter_id) {}
    std::uint64_t id{};
    std::promise<AcquireResult> promise;
  };

  struct Entry final {
    ExpertKey key;
    PayloadRecord record;
    CacheState state{CacheState::absent};
    std::map<std::uint64_t, std::shared_ptr<Waiter>> waiters;
    std::shared_ptr<FixedBufferPool::Lease> host;
    std::shared_ptr<std::vector<std::byte>> host_copy;
    std::shared_ptr<IDeviceAllocation> device;
    std::uint64_t ram_reserved{};
    std::uint64_t vram_reserved{};
    std::uint64_t references{};
    std::uint64_t last_access{};
    OperationId io_operation{};
    OperationId upload_operation{};
    bool abandon{};
  };

  enum class TaskKind { none, read, upload };
  struct Task final {
    TaskKind kind{TaskKind::none};
    std::shared_ptr<Entry> entry;
  };

  ExpertCacheCore(ExpertCacheConfig cache_config,
                  std::shared_ptr<IAsyncStorage> async_storage,
                  std::shared_ptr<IDeviceUploader> device_uploader,
                  std::shared_ptr<FixedBufferPool> buffer_pool)
      : config(cache_config),
        storage(std::move(async_storage)),
        uploader(std::move(device_uploader)),
        buffers(std::move(buffer_pool)) {
    if (!valid_budget(config.ram) || !valid_budget(config.vram) || !storage ||
        !uploader || !buffers || buffers->alignment() < kExpertPackAlignment) {
      throw std::invalid_argument("invalid expert cache configuration");
    }
  }

  void transition_locked(Entry& entry, CacheState next) {
    if (!valid_cache_transition(entry.state, next)) {
      throw std::logic_error("invalid expert cache state transition");
    }
    const auto index = static_cast<std::size_t>(entry.state) * kStateCount +
                       static_cast<std::size_t>(next);
    Telemetry::add(metrics.state_transitions_[index]);
    entry.state = next;
  }

  void update_usage_locked() noexcept {
    Telemetry::set(metrics.ram_bytes_, ram_bytes);
    Telemetry::set(metrics.vram_bytes_, vram_bytes);
    const auto staging = buffers->bytes_in_use();
    Telemetry::set(metrics.staging_bytes_, staging);
    Telemetry::maximize(metrics.ram_high_water_, ram_bytes);
    Telemetry::maximize(metrics.vram_high_water_, vram_bytes);
    Telemetry::maximize(metrics.staging_high_water_, staging);
  }

  void release_host_locked(Entry& entry) noexcept {
    entry.host.reset();
    entry.host_copy.reset();
    if (entry.ram_reserved != 0) {
      ram_bytes -= entry.ram_reserved;
      entry.ram_reserved = 0;
    }
  }

  void release_device_locked(Entry& entry) noexcept {
    entry.device.reset();
    if (entry.vram_reserved != 0) {
      vram_bytes -= entry.vram_reserved;
      entry.vram_reserved = 0;
    }
  }

  void satisfy_ready_waiters_locked(const std::shared_ptr<Entry>& entry) {
    auto weak = weak_from_this();
    for (auto& [id, waiter] : entry->waiters) {
      (void)id;
      ++entry->references;
      auto release = [weak, key = entry->key]() noexcept {
        if (auto core = weak.lock()) {
          core->release_reference(key);
        }
      };
      waiter->promise.set_value(
          {Status::success(), ExpertLease(entry->device, std::move(release))});
    }
    entry->waiters.clear();
  }

  void fail_waiters_locked(Entry& entry, const Status& status) {
    for (auto& [id, waiter] : entry.waiters) {
      (void)id;
      waiter->promise.set_value({status, {}});
    }
    entry.waiters.clear();
  }

  void fail_entry_locked(Entry& entry, Status status) {
    if (entry.state != CacheState::failed) {
      transition_locked(entry, CacheState::failed);
    }
    fail_waiters_locked(entry, status);
    release_host_locked(entry);
    release_device_locked(entry);
    entry.io_operation = 0;
    entry.upload_operation = 0;
    update_usage_locked();
  }

  bool evict_one_locked(bool need_ram, bool need_vram,
                        const ExpertKey* excluded) {
    std::shared_ptr<Entry> candidate;
    for (const auto& [key, entry] : entries) {
      if ((excluded != nullptr && key == *excluded) || entry->references != 0 ||
          !entry->waiters.empty() ||
          entry->state == CacheState::ssd_loading ||
          entry->state == CacheState::gpu_uploading ||
          entry->state == CacheState::failed) {
        continue;
      }
      const bool useful = (need_ram && (entry->host || entry->host_copy)) ||
                          (need_vram && entry->device);
      if (!useful) {
        continue;
      }
      if (!candidate || entry->last_access < candidate->last_access ||
          (entry->last_access == candidate->last_access && key < candidate->key)) {
        candidate = entry;
      }
    }
    if (!candidate) {
      return false;
    }

    if (need_ram && (candidate->host || candidate->host_copy)) {
      release_host_locked(*candidate);
    }
    if (need_vram && candidate->device) {
      release_device_locked(*candidate);
      if (candidate->state == CacheState::vram_ready) {
        transition_locked(*candidate,
                          (candidate->host || candidate->host_copy)
                              ? CacheState::ram_ready
                              : CacheState::absent);
      }
    }
    if (candidate->state == CacheState::ram_ready && !candidate->host &&
        !candidate->host_copy) {
      transition_locked(*candidate, CacheState::absent);
    }
    Telemetry::add(metrics.eviction_count_);
    update_usage_locked();
    return true;
  }

  bool make_capacity_locked(std::uint64_t ram_need, std::uint64_t vram_need,
                            const ExpertKey& key) {
    if (ram_need > config.ram.high_watermark_bytes ||
        vram_need > config.vram.high_watermark_bytes) {
      return false;
    }
    bool pressured = ram_bytes + ram_need > config.ram.high_watermark_bytes ||
                     vram_bytes + vram_need > config.vram.high_watermark_bytes;
    while (ram_bytes + ram_need > config.ram.high_watermark_bytes ||
           vram_bytes + vram_need > config.vram.high_watermark_bytes) {
      const bool need_ram = ram_bytes + ram_need > config.ram.low_watermark_bytes;
      const bool need_vram = vram_bytes + vram_need > config.vram.low_watermark_bytes;
      if (!evict_one_locked(need_ram, need_vram, &key)) {
        Telemetry::add(metrics.stalled_by_budget_);
        return false;
      }
    }
    if (pressured) {
      // Continue evicting toward low watermarks when safe, creating headroom
      // for a burst instead of oscillating at the high mark.
      while ((ram_bytes > config.ram.low_watermark_bytes ||
              vram_bytes > config.vram.low_watermark_bytes) &&
             evict_one_locked(ram_bytes > config.ram.low_watermark_bytes,
                              vram_bytes > config.vram.low_watermark_bytes,
                              &key)) {
      }
    }
    return ram_bytes + ram_need <= config.ram.high_watermark_bytes &&
           vram_bytes + vram_need <= config.vram.high_watermark_bytes;
  }

  Task next_task_locked() {
    for (const auto& [key, entry] : entries) {
      (void)key;
      if (entry->waiters.empty() || entry->abandon) {
        continue;
      }
      if (entry->state == CacheState::ram_ready &&
          (entry->host || entry->host_copy)) {
        transition_locked(*entry, CacheState::gpu_uploading);
        Telemetry::add(metrics.upload_started_);
        return {TaskKind::upload, entry};
      }
      if (entry->state != CacheState::absent) {
        continue;
      }
      if (entry->record.stored_bytes > buffers->slot_bytes() ||
          entry->record.stored_bytes > std::numeric_limits<std::size_t>::max()) {
        fail_entry_locked(*entry,
                          Status(ErrorCode::backpressure,
                                 "expert record exceeds fixed staging slot"));
        continue;
      }
      if (!make_capacity_locked(entry->record.stored_bytes,
                                entry->record.stored_bytes, entry->key)) {
        continue;
      }
      auto host = buffers->try_acquire(
          static_cast<std::size_t>(entry->record.stored_bytes));
      if (!host) {
        Telemetry::add(metrics.stalled_by_budget_);
        continue;
      }
      entry->host = std::move(host);
      entry->ram_reserved = entry->record.stored_bytes;
      entry->vram_reserved = entry->record.stored_bytes;
      ram_bytes += entry->ram_reserved;
      vram_bytes += entry->vram_reserved;
      transition_locked(*entry, CacheState::ssd_loading);
      Telemetry::add(metrics.load_started_);
      Telemetry::add(metrics.requested_bytes_, entry->record.stored_bytes);
      update_usage_locked();
      return {TaskKind::read, entry};
    }
    return {};
  }

  void drive() {
    for (;;) {
      Task task;
      {
        std::lock_guard lock(mutex);
        if (shutting_down) {
          return;
        }
        task = next_task_locked();
      }
      if (task.kind == TaskKind::none) {
        return;
      }
      if (task.kind == TaskKind::read) {
        start_read(std::move(task.entry));
      } else {
        start_upload(std::move(task.entry));
      }
    }
  }

  void start_read(const std::shared_ptr<Entry>& entry) {
    auto weak = weak_from_this();
    ReadRequest request{entry->record, entry->host->buffer(), true};
    const auto operation = storage->read(
        std::move(request), [weak, key = entry->key](ReadResult result) mutable {
          if (auto core = weak.lock()) {
            core->read_complete(key, std::move(result));
          }
        });
    bool cancel_now = false;
    {
      std::lock_guard lock(mutex);
      if (entry->state == CacheState::ssd_loading) {
        entry->io_operation = operation;
        cancel_now = entry->abandon;
      }
    }
    if (cancel_now) {
      storage->cancel(operation);
    }
  }

  void start_upload(const std::shared_ptr<Entry>& entry) {
    const auto count = static_cast<std::size_t>(entry->record.stored_bytes);
    std::span<const std::byte> bytes;
    if (entry->host) {
      const auto buffer = entry->host->buffer();
      bytes = std::span<const std::byte>(buffer.data, count);
    } else if (entry->host_copy) {
      bytes = std::span<const std::byte>(entry->host_copy->data(), count);
    } else {
      {
        std::lock_guard lock(mutex);
        fail_entry_locked(*entry,
                          Status(ErrorCode::internal,
                                 "RAM-ready expert has no host bytes"));
      }
      drive();
      return;
    }
    const auto validated = validate_expert_record(bytes, entry->key, entry->record);
    if (!validated.status.ok()) {
      {
        std::lock_guard lock(mutex);
        Telemetry::add(metrics.checksum_errors_);
        fail_entry_locked(*entry, validated.status);
      }
      drive();
      return;
    }

    if (config.retain_host_copy && !entry->host_copy) {
      try {
        entry->host_copy = std::make_shared<std::vector<std::byte>>(count);
        std::memcpy(entry->host_copy->data(), bytes.data(), count);
      } catch (const std::bad_alloc&) {
        {
          std::lock_guard lock(mutex);
          fail_entry_locked(*entry,
                            Status(ErrorCode::backpressure,
                                   "cannot allocate pageable RAM cache copy"));
        }
        drive();
        return;
      }
    }

    auto weak = weak_from_this();
    UploadRequest request{entry->key, validated.record.sections, bytes};
    const auto operation = uploader->upload(
        request, [weak, key = entry->key](UploadResult result) mutable {
          if (auto core = weak.lock()) {
            core->upload_complete(key, std::move(result));
          }
        });
    bool cancel_now = false;
    {
      std::lock_guard lock(mutex);
      if (entry->state == CacheState::gpu_uploading) {
        entry->upload_operation = operation;
        cancel_now = entry->abandon;
      }
    }
    if (cancel_now) {
      uploader->cancel(operation);
    }
  }

  void read_complete(const ExpertKey& key, ReadResult result) {
    {
      std::lock_guard lock(mutex);
      const auto iterator = entries.find(key);
      if (iterator == entries.end() ||
          iterator->second->state != CacheState::ssd_loading) {
        return;
      }
      auto& entry = *iterator->second;
      entry.io_operation = 0;
      Telemetry::add(metrics.read_bytes_, result.read_bytes);
      if (entry.abandon || entry.waiters.empty()) {
        transition_locked(entry, CacheState::absent);
        release_host_locked(entry);
        release_device_locked(entry);
        update_usage_locked();
      } else if (!result.status.ok()) {
        if (result.status.code() == ErrorCode::short_read) {
          Telemetry::add(metrics.short_read_errors_);
        } else if (result.status.code() != ErrorCode::cancelled) {
          Telemetry::add(metrics.io_errors_);
        }
        fail_entry_locked(entry, result.status);
      } else if (result.requested_bytes != entry.record.stored_bytes ||
                 result.read_bytes != entry.record.stored_bytes) {
        Telemetry::add(metrics.short_read_errors_);
        fail_entry_locked(
            entry, Status(ErrorCode::short_read,
                          "storage completed without the complete expert record"));
      } else {
        transition_locked(entry, CacheState::ram_ready);
        Telemetry::add(metrics.load_completed_);
        Telemetry::add(metrics.useful_bytes_, entry.record.stored_bytes);
      }
    }
    drive();
  }

  void upload_complete(const ExpertKey& key, UploadResult result) {
    {
      std::lock_guard lock(mutex);
      const auto iterator = entries.find(key);
      if (iterator == entries.end() ||
          iterator->second->state != CacheState::gpu_uploading) {
        return;
      }
      auto& entry = *iterator->second;
      entry.upload_operation = 0;
      if (entry.abandon || entry.waiters.empty()) {
        transition_locked(entry, CacheState::ram_ready);
        release_host_locked(entry);
        release_device_locked(entry);
        transition_locked(entry, CacheState::absent);
        update_usage_locked();
      } else if (!result.status.ok() || !result.allocation ||
                 result.uploaded_bytes == 0 ||
                 result.uploaded_bytes != result.allocation->bytes() ||
                 result.allocation->bytes() > entry.vram_reserved) {
        Telemetry::add(metrics.upload_errors_);
        const auto status = result.status.ok()
                                ? Status(ErrorCode::upload_failed,
                                         "device upload returned an invalid allocation")
                                : result.status;
        fail_entry_locked(entry, status);
      } else {
        if (result.allocation->bytes() < entry.vram_reserved) {
          const auto released = entry.vram_reserved - result.allocation->bytes();
          entry.vram_reserved -= released;
          vram_bytes -= released;
        }
        entry.device = std::move(result.allocation);
        transition_locked(entry, CacheState::vram_ready);
        entry.last_access = ++access_clock;
        Telemetry::add(metrics.upload_completed_);
        Telemetry::add(metrics.uploaded_bytes_, result.uploaded_bytes);
        if (!config.retain_host_copy) {
          release_host_locked(entry);
        } else {
          // The long-lived RAM tier is pageable. Pinned buffers remain a
          // bounded staging resource and return to the fixed pool after H2D.
          entry.host.reset();
        }
        update_usage_locked();
        satisfy_ready_waiters_locked(iterator->second);
      }
    }
    drive();
  }

  AcquireHandle acquire(const ExpertKey& key, const PayloadRecord& record) {
    auto waiter = std::make_shared<Waiter>(next_waiter.fetch_add(1));
    auto future = waiter->promise.get_future();
    bool should_drive = false;
    {
      std::lock_guard lock(mutex);
      if (shutting_down) {
        waiter->promise.set_value(
            {Status(ErrorCode::cancelled, "expert cache is shutting down"), {}});
      } else {
        auto [iterator, inserted] = entries.try_emplace(key);
        if (inserted) {
          iterator->second = std::make_shared<Entry>();
          iterator->second->key = key;
          iterator->second->record = record;
        }
        auto& entry = *iterator->second;
        if (!same_record(entry.record, record)) {
          waiter->promise.set_value(
              {Status(ErrorCode::invalid_argument,
                      "same expert key references different immutable records"),
               {}});
        } else if (entry.state == CacheState::failed) {
          waiter->promise.set_value(
              {Status(ErrorCode::checksum_mismatch,
                      "expert is quarantined after a failed load"),
               {}});
        } else if (entry.state == CacheState::vram_ready && entry.device) {
          ++entry.references;
          entry.last_access = ++access_clock;
          auto weak = weak_from_this();
          waiter->promise.set_value(
              {Status::success(),
               ExpertLease(entry.device, [weak, key]() noexcept {
                 if (auto core = weak.lock()) {
                   core->release_reference(key);
                 }
               })});
        } else {
          if (!entry.waiters.empty() || entry.state != CacheState::absent) {
            Telemetry::add(metrics.load_deduplicated_);
          }
          entry.waiters.emplace(waiter->id, waiter);
          entry.abandon = false;
          should_drive = true;
        }
      }
    }
    auto weak = weak_from_this();
    AcquireHandle handle(std::move(future), [weak, key, id = waiter->id]() noexcept {
      if (auto core = weak.lock()) {
        core->cancel_waiter(key, id);
      }
    });
    if (should_drive) {
      drive();
    }
    return handle;
  }

  void cancel_waiter(const ExpertKey& key, std::uint64_t waiter_id) noexcept {
    OperationId io = 0;
    OperationId upload = 0;
    {
      std::lock_guard lock(mutex);
      const auto iterator = entries.find(key);
      if (iterator == entries.end()) {
        return;
      }
      auto& entry = *iterator->second;
      const auto waiter = entry.waiters.find(waiter_id);
      if (waiter == entry.waiters.end()) {
        return;
      }
      waiter->second->promise.set_value(
          {Status(ErrorCode::cancelled, "expert acquisition cancelled"), {}});
      entry.waiters.erase(waiter);
      Telemetry::add(metrics.cancellation_count_);
      if (entry.waiters.empty()) {
        entry.abandon = true;
        io = entry.io_operation;
        upload = entry.upload_operation;
      }
    }
    if (io != 0) {
      storage->cancel(io);
    }
    if (upload != 0) {
      uploader->cancel(upload);
    }
    drive();
  }

  void release_reference(const ExpertKey& key) noexcept {
    {
      std::lock_guard lock(mutex);
      const auto iterator = entries.find(key);
      if (iterator == entries.end() || iterator->second->references == 0) {
        return;
      }
      --iterator->second->references;
      iterator->second->last_access = ++access_clock;
    }
    drive();
  }

  std::uint64_t trim() {
    std::lock_guard lock(mutex);
    const auto before = ram_bytes + vram_bytes;
    bool progress = true;
    while (progress) {
      progress = evict_one_locked(true, true, nullptr);
    }
    return before - (ram_bytes + vram_bytes);
  }

  void shutdown() noexcept {
    std::vector<OperationId> reads;
    std::vector<OperationId> uploads;
    {
      std::lock_guard lock(mutex);
      if (shutting_down) {
        return;
      }
      shutting_down = true;
      for (auto& [key, entry] : entries) {
        (void)key;
        fail_waiters_locked(
            *entry, Status(ErrorCode::cancelled, "expert cache shutdown"));
        if (entry->io_operation != 0) {
          reads.push_back(entry->io_operation);
        }
        if (entry->upload_operation != 0) {
          uploads.push_back(entry->upload_operation);
        }
      }
    }
    for (auto operation : reads) {
      storage->cancel(operation);
    }
    for (auto operation : uploads) {
      uploader->cancel(operation);
    }
  }

  ExpertCacheConfig config;
  std::shared_ptr<IAsyncStorage> storage;
  std::shared_ptr<IDeviceUploader> uploader;
  std::shared_ptr<FixedBufferPool> buffers;
  mutable std::mutex mutex;
  std::map<ExpertKey, std::shared_ptr<Entry>> entries;
  Telemetry metrics;
  std::atomic<std::uint64_t> next_waiter{1};
  std::uint64_t access_clock{};
  std::uint64_t ram_bytes{};
  std::uint64_t vram_bytes{};
  bool shutting_down{};
};

ExpertLease::ExpertLease(std::shared_ptr<IDeviceAllocation> allocation,
                         std::function<void()> release) noexcept
    : allocation_(std::move(allocation)), release_(std::move(release)) {}

ExpertLease::ExpertLease(ExpertLease&& other) noexcept
    : allocation_(std::move(other.allocation_)),
      release_(std::move(other.release_)) {}

ExpertLease& ExpertLease::operator=(ExpertLease&& other) noexcept {
  if (this != &other) {
    reset();
    allocation_ = std::move(other.allocation_);
    release_ = std::move(other.release_);
  }
  return *this;
}

ExpertLease::~ExpertLease() { reset(); }

ExpertLease::operator bool() const noexcept { return allocation_ != nullptr; }

const IDeviceAllocation* ExpertLease::get() const noexcept {
  return allocation_.get();
}

std::shared_ptr<const IDeviceAllocation> ExpertLease::share() const noexcept {
  return allocation_;
}

void ExpertLease::reset() noexcept {
  allocation_.reset();
  if (release_) {
    auto release = std::move(release_);
    release();
  }
}

AcquireHandle::AcquireHandle(std::future<AcquireResult> future,
                             std::function<void()> cancel) noexcept
    : future_(std::move(future)), cancel_(std::move(cancel)) {}

void AcquireHandle::cancel() noexcept {
  if (cancel_) {
    auto cancel = std::move(cancel_);
    cancel();
  }
}

ExpertCache::ExpertCache(ExpertCacheConfig config,
                         std::shared_ptr<IAsyncStorage> storage,
                         std::shared_ptr<IDeviceUploader> uploader,
                         std::shared_ptr<FixedBufferPool> buffers)
    : core_(std::make_shared<ExpertCacheCore>(config, std::move(storage),
                                               std::move(uploader),
                                               std::move(buffers))) {}

ExpertCache::~ExpertCache() {
  if (core_) {
    core_->shutdown();
  }
}

AcquireHandle ExpertCache::acquire(const ExpertKey& key,
                                   const PayloadRecord& record) {
  return core_->acquire(key, record);
}

std::optional<CacheEntrySnapshot> ExpertCache::inspect(
    const ExpertKey& key) const {
  std::lock_guard lock(core_->mutex);
  const auto iterator = core_->entries.find(key);
  if (iterator == core_->entries.end()) {
    return std::nullopt;
  }
  const auto& entry = *iterator->second;
  return CacheEntrySnapshot{entry.state, entry.references, entry.waiters.size(),
                            entry.host != nullptr || entry.host_copy != nullptr,
                            entry.device != nullptr};
}

TelemetrySnapshot ExpertCache::telemetry() const noexcept {
  return core_->metrics.snapshot();
}

std::uint64_t ExpertCache::trim() { return core_->trim(); }

}  // namespace expert::runtime
