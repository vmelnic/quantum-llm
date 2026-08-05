#include "expert/runtime/expert_cache.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace expert::runtime {
namespace {

constexpr std::size_t kStateCount = 6;
constexpr std::uint64_t kRoutingScoreScale = 1ULL << 20U;

bool valid_budget(const TierBudget& budget) noexcept {
  return budget.capacity_bytes != 0 && budget.high_watermark_bytes != 0 &&
         budget.low_watermark_bytes <= budget.high_watermark_bytes &&
         budget.high_watermark_bytes <= budget.capacity_bytes;
}

bool same_record(const PayloadRecord& left, const PayloadRecord& right) {
  return left.path == right.path && left.extents == right.extents &&
         left.record_offset == right.record_offset &&
         left.stored_bytes == right.stored_bytes &&
         left.decoded_bytes == right.decoded_bytes &&
         left.device_bytes == right.device_bytes &&
         left.source_abi == right.source_abi &&
         left.header_bytes == right.header_bytes &&
         left.alignment == right.alignment &&
         constant_time_equal(left.payload_sha256, right.payload_sha256);
}

std::uint64_t device_bytes(const PayloadRecord& record) noexcept {
  return record.device_bytes == 0U ? record.stored_bytes : record.device_bytes;
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
    std::optional<ExpertSections> validated_sections;
    DeepSeekCompactSections validated_compact;
    std::uint64_t ram_reserved{};
    std::uint64_t vram_reserved{};
    std::uint64_t references{};
    std::uint64_t last_access{};
    std::uint32_t frequency{};
    std::uint64_t routing_score_mass_q20{};
    std::uint32_t routing_score_peak_q20{};
    bool vram_resident{true};
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
                  std::shared_ptr<FixedBufferPool> buffer_pool,
                  std::shared_ptr<IDeviceResidencyDirectory> device_directory)
      : config(cache_config),
        storage(std::move(async_storage)),
        uploader(std::move(device_uploader)),
        buffers(std::move(buffer_pool)),
        directory(std::move(device_directory)),
        ram_partition_bytes(config.placement.layer_partition_count),
        vram_partition_bytes(config.placement.layer_partition_count) {
    if (!valid_budget(config.ram) || !valid_budget(config.vram) || !storage ||
        !uploader || !buffers || buffers->alignment() < kExpertPackAlignment ||
        config.placement.layer_partition_count == 0 ||
        config.placement.layers_per_partition == 0 ||
        config.placement.ram_shared_burst_bytes >=
            config.ram.high_watermark_bytes ||
        config.placement.vram_shared_burst_bytes >=
            config.vram.high_watermark_bytes ||
        config.placement.vram_transient_bytes >=
            config.vram.high_watermark_bytes) {
      throw std::invalid_argument("invalid expert cache configuration");
    }
  }

  [[nodiscard]] std::size_t partition_for(const ExpertKey& key) const noexcept {
    const auto partition =
        key.layer / config.placement.layers_per_partition;
    return std::min<std::size_t>(partition,
                                 ram_partition_bytes.size() - 1U);
  }

  [[nodiscard]] std::uint64_t partition_quota(
      const TierBudget& tier, std::uint64_t shared_burst) const noexcept {
    return (tier.high_watermark_bytes - shared_burst) /
           config.placement.layer_partition_count;
  }

  void reserve_ram_locked(Entry& entry, std::uint64_t bytes) noexcept {
    entry.ram_reserved += bytes;
    ram_bytes += bytes;
    ram_partition_bytes[partition_for(entry.key)] += bytes;
  }

  void reserve_vram_locked(Entry& entry, std::uint64_t bytes,
                           bool resident) noexcept {
    entry.vram_resident = resident;
    entry.vram_reserved += bytes;
    vram_bytes += bytes;
    if (resident) {
      vram_resident_bytes += bytes;
      vram_partition_bytes[partition_for(entry.key)] += bytes;
    } else {
      vram_transient_bytes += bytes;
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

  void touch_locked(Entry& entry) noexcept {
    entry.last_access = ++access_clock;
    if (entry.frequency != std::numeric_limits<std::uint32_t>::max()) {
      ++entry.frequency;
    }
    // Bounded, deterministic aging lets placement follow a changed workload
    // without scanning the cache in the decode hot path.
    if ((access_clock & 0xffffU) == 0U) {
      for (auto& [key, candidate] : entries) {
        (void)key;
        candidate->frequency = (candidate->frequency + 1U) / 2U;
        candidate->routing_score_mass_q20 =
            (candidate->routing_score_mass_q20 + 1U) / 2U;
        candidate->routing_score_peak_q20 =
            (candidate->routing_score_peak_q20 + 1U) / 2U;
      }
    }
  }

  void add_routing_score_locked(Entry& entry, double sum,
                                double maximum) noexcept {
    if (!std::isfinite(sum) || !std::isfinite(maximum) || sum <= 0.0 ||
        maximum < 0.0) {
      return;
    }
    const auto scaled_sum = static_cast<std::uint64_t>(std::min(
        std::ceil(sum * static_cast<double>(kRoutingScoreScale)),
        static_cast<double>(std::numeric_limits<std::uint64_t>::max())));
    entry.routing_score_mass_q20 =
        scaled_sum > std::numeric_limits<std::uint64_t>::max() -
                         entry.routing_score_mass_q20
            ? std::numeric_limits<std::uint64_t>::max()
            : entry.routing_score_mass_q20 + scaled_sum;
    const auto scaled_max = static_cast<std::uint32_t>(std::min(
        std::ceil(std::min(maximum, 1.0) *
                  static_cast<double>(kRoutingScoreScale)),
        static_cast<double>(std::numeric_limits<std::uint32_t>::max())));
    entry.routing_score_peak_q20 =
        std::max(entry.routing_score_peak_q20, scaled_max);
  }

  [[nodiscard]] std::uint64_t temperature_locked(
      const Entry& entry) const noexcept {
    const auto frequency_heat =
        static_cast<std::uint64_t>(entry.frequency) * kRoutingScoreScale;
    const auto peak_heat = entry.routing_score_peak_q20 / 4U;
    const auto score_heat =
        peak_heat > std::numeric_limits<std::uint64_t>::max() -
                        entry.routing_score_mass_q20
            ? std::numeric_limits<std::uint64_t>::max()
            : entry.routing_score_mass_q20 + peak_heat;
    return score_heat > std::numeric_limits<std::uint64_t>::max() -
                            frequency_heat
               ? std::numeric_limits<std::uint64_t>::max()
               : frequency_heat + score_heat;
  }

  void update_usage_locked() noexcept {
    Telemetry::set(metrics.ram_bytes_, ram_bytes);
    Telemetry::set(metrics.vram_bytes_, vram_bytes);
    Telemetry::set(metrics.vram_resident_bytes_, vram_resident_bytes);
    Telemetry::set(metrics.vram_transient_bytes_, vram_transient_bytes);
    const auto staging = buffers->bytes_in_use();
    Telemetry::set(metrics.staging_bytes_, staging);
    Telemetry::maximize(metrics.ram_high_water_, ram_bytes);
    Telemetry::maximize(metrics.vram_high_water_, vram_bytes);
    Telemetry::maximize(metrics.vram_resident_high_water_,
                        vram_resident_bytes);
    Telemetry::maximize(metrics.vram_transient_high_water_,
                        vram_transient_bytes);
    Telemetry::maximize(metrics.staging_high_water_, staging);
  }

  void release_host_locked(Entry& entry) noexcept {
    entry.host.reset();
    entry.host_copy.reset();
    entry.validated_sections.reset();
    if (entry.ram_reserved != 0) {
      ram_partition_bytes[partition_for(entry.key)] -= entry.ram_reserved;
      ram_bytes -= entry.ram_reserved;
      entry.ram_reserved = 0;
    }
  }

  void release_device_locked(Entry& entry) noexcept {
    if (entry.device && directory) {
      directory->retire(entry.key);
    }
    entry.device.reset();
    if (entry.vram_reserved != 0) {
      if (entry.vram_resident) {
        vram_resident_bytes -= entry.vram_reserved;
        vram_partition_bytes[partition_for(entry.key)] -= entry.vram_reserved;
      } else {
        vram_transient_bytes -= entry.vram_reserved;
      }
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
                        const ExpertKey* excluded,
                        std::optional<std::size_t> required_partition = {},
                        bool over_quota_only = false,
                        std::optional<bool> required_vram_class = {},
                        std::optional<std::uint64_t>
                            maximum_temperature_exclusive = {}) {
    std::shared_ptr<Entry> candidate;
    std::uint64_t candidate_excess{};
    std::uint64_t candidate_temperature{};
    const auto ram_quota = partition_quota(
        config.ram, config.placement.ram_shared_burst_bytes);
    const auto vram_quota = partition_quota(
        config.vram, config.placement.vram_shared_burst_bytes);
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
      if (need_vram && required_vram_class &&
          (!entry->device ||
           entry->vram_resident != *required_vram_class)) {
        continue;
      }
      const auto entry_temperature = temperature_locked(*entry);
      if (maximum_temperature_exclusive &&
          entry_temperature >= *maximum_temperature_exclusive) {
        continue;
      }
      const auto partition = partition_for(key);
      if (required_partition && partition != *required_partition) {
        continue;
      }
      const auto ram_excess =
          need_ram && ram_partition_bytes[partition] > ram_quota
              ? ram_partition_bytes[partition] - ram_quota
              : 0;
      const auto vram_excess =
          need_vram && vram_partition_bytes[partition] > vram_quota
              ? vram_partition_bytes[partition] - vram_quota
              : 0;
      const auto excess = std::max(ram_excess, vram_excess);
      if (over_quota_only && excess == 0) {
        continue;
      }
      if (!candidate || entry_temperature < candidate_temperature ||
          (entry_temperature == candidate_temperature &&
           (excess > candidate_excess ||
            (excess == candidate_excess &&
             (entry->last_access < candidate->last_access ||
              (entry->last_access == candidate->last_access &&
               key < candidate->key)))))) {
        candidate = entry;
        candidate_excess = excess;
        candidate_temperature = entry_temperature;
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
    if (required_partition) {
      Telemetry::add(metrics.same_partition_evictions_);
    } else if (candidate_excess != 0) {
      Telemetry::add(metrics.over_quota_evictions_);
    }
    update_usage_locked();
    return true;
  }

  bool make_partition_capacity_locked(std::uint64_t ram_need,
                                      const ExpertKey& key) {
    if (config.placement.layer_partition_count == 1) {
      return true;
    }
    const auto partition = partition_for(key);
    const auto ram_limit =
        partition_quota(config.ram,
                        config.placement.ram_shared_burst_bytes) +
        config.placement.ram_shared_burst_bytes;
    if (ram_need > ram_limit) {
      return false;
    }
    while (ram_partition_bytes[partition] + ram_need > ram_limit) {
      const bool need_partition_ram =
          ram_partition_bytes[partition] + ram_need > ram_limit;
      if (!evict_one_locked(need_partition_ram, false, &key,
                            partition)) {
        Telemetry::add(metrics.stalled_by_budget_);
        return false;
      }
    }
    return true;
  }

  bool make_vram_class_capacity_locked(std::uint64_t vram_need,
                                       const ExpertKey& key,
                                       bool resident,
                                       std::uint64_t admission_temperature) {
    if (vram_need == 0) return true;
    const auto transient_limit =
        config.placement.vram_transient_bytes;
    const auto admission_limit =
        resident && transient_limit != 0
            ? std::optional<std::uint64_t>(admission_temperature)
            : std::nullopt;
    const auto limit = resident
                           ? config.vram.high_watermark_bytes - transient_limit
                           : transient_limit;
    if (vram_need > limit) return false;
    auto& usage = resident ? vram_resident_bytes : vram_transient_bytes;
    while (usage + vram_need > limit) {
      bool evicted = false;
      if (resident && config.placement.layer_partition_count > 1) {
        const auto partition = partition_for(key);
        const auto quota = partition_quota(
            config.vram, config.placement.vram_shared_burst_bytes);
        if (vram_partition_bytes[partition] + vram_need > quota) {
          evicted = evict_one_locked(false, true, &key, partition, false,
                                     true, admission_limit);
        }
      }
      if (!evicted &&
          !evict_one_locked(false, true, &key, {}, false, resident,
                            admission_limit)) {
        Telemetry::add(metrics.stalled_by_budget_);
        return false;
      }
    }
    return true;
  }

  void promote_vram_locked(Entry& entry) {
    if (!entry.device || entry.vram_resident || entry.vram_reserved == 0 ||
        config.placement.vram_transient_bytes == 0 ||
        entry.frequency < 2 ||
        !make_vram_class_capacity_locked(entry.vram_reserved, entry.key,
                                         true, temperature_locked(entry))) {
      return;
    }
    vram_transient_bytes -= entry.vram_reserved;
    vram_resident_bytes += entry.vram_reserved;
    vram_partition_bytes[partition_for(entry.key)] += entry.vram_reserved;
    entry.vram_resident = true;
    update_usage_locked();
  }

  bool make_capacity_locked(std::uint64_t ram_need, std::uint64_t vram_need,
                            const ExpertKey& key, bool resident,
                            std::uint64_t admission_temperature) {
    if (ram_need > config.ram.high_watermark_bytes ||
        vram_need > config.vram.high_watermark_bytes) {
      return false;
    }
    if (!make_partition_capacity_locked(ram_need, key) ||
        !make_vram_class_capacity_locked(vram_need, key, resident,
                                         admission_temperature)) {
      return false;
    }
    const bool ram_pressured =
        ram_bytes + ram_need > config.ram.high_watermark_bytes;
    while (ram_bytes + ram_need > config.ram.high_watermark_bytes) {
      if (!evict_one_locked(true, false, &key, {}, true) &&
          !evict_one_locked(true, false, &key)) {
        Telemetry::add(metrics.stalled_by_budget_);
        return false;
      }
    }
    if (ram_pressured && config.placement.layer_partition_count == 1) {
      // Continue evicting toward low watermarks when safe, creating headroom
      // for a burst instead of oscillating at the high mark.
      while (ram_bytes > config.ram.low_watermark_bytes &&
             evict_one_locked(true, false, &key)) {
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
        if (entry->vram_reserved == 0) {
          bool resident =
              config.placement.vram_transient_bytes == 0 ||
              entry->frequency >= 2;
          const auto vram_need = device_bytes(entry->record);
          if (!make_capacity_locked(0, vram_need,
                                    entry->key, resident,
                                    temperature_locked(*entry))) {
            if (!resident ||
                !make_capacity_locked(0, vram_need,
                                      entry->key, false,
                                      temperature_locked(*entry))) {
              continue;
            }
            resident = false;
          }
          reserve_vram_locked(*entry, vram_need, resident);
          update_usage_locked();
        }
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
      bool resident =
          config.placement.vram_transient_bytes == 0 ||
          entry->frequency >= 2;
      const auto vram_need = device_bytes(entry->record);
      if (!make_capacity_locked(entry->record.stored_bytes,
                                vram_need, entry->key,
                                resident, temperature_locked(*entry))) {
        if (!resident ||
            !make_capacity_locked(entry->record.stored_bytes,
                                  vram_need, entry->key,
                                  false, temperature_locked(*entry))) {
          continue;
        }
        resident = false;
      }
      auto host = buffers->try_acquire(
          static_cast<std::size_t>(entry->record.stored_bytes));
      if (!host) {
        Telemetry::add(metrics.stalled_by_budget_);
        continue;
      }
      entry->host = std::move(host);
      reserve_ram_locked(*entry, entry->record.stored_bytes);
      reserve_vram_locked(*entry, vram_need, resident);
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
    ExpertSections sections;
    DeepSeekCompactSections compact;
    if (entry->validated_sections) {
      sections = *entry->validated_sections;
      compact = entry->validated_compact;
      Telemetry::add(metrics.validated_ram_reuses_);
    } else {
      const auto validated =
          validate_expert_admission(bytes, entry->key, entry->record,
                                    !config.trusted_immutable_source);
      Telemetry::add(metrics.record_validations_);
      if (!validated.status.ok()) {
        {
          std::lock_guard lock(mutex);
          Telemetry::add(metrics.checksum_errors_);
          fail_entry_locked(*entry, validated.status);
        }
        drive();
        return;
      }
      sections = validated.target;
      compact = validated.compact;
      entry->validated_sections = sections;
      entry->validated_compact = compact;
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
    UploadRequest request{entry->key, entry->record.source_abi, sections, bytes,
                          compact};
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
          if (entry.vram_resident) {
            vram_resident_bytes -= released;
            vram_partition_bytes[partition_for(entry.key)] -= released;
          } else {
            vram_transient_bytes -= released;
          }
        }
        if (directory) {
          const auto published = directory->publish(entry.key, result.allocation);
          if (!published.ok()) {
            Telemetry::add(metrics.upload_errors_);
            fail_entry_locked(entry, published);
          }
        }
        if (entry.state != CacheState::failed) {
          entry.device = std::move(result.allocation);
          transition_locked(entry, CacheState::vram_ready);
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
        touch_locked(entry);
        promote_vram_locked(entry);
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
          Telemetry::add(metrics.acquire_vram_hits_);
          ++entry.references;
          auto weak = weak_from_this();
          waiter->promise.set_value(
              {Status::success(),
               ExpertLease(entry.device, [weak, key]() noexcept {
                 if (auto core = weak.lock()) {
                   core->release_reference(key);
                 }
               })});
        } else {
          if (entry.state == CacheState::ram_ready ||
              entry.state == CacheState::gpu_uploading) {
            Telemetry::add(metrics.acquire_ram_hits_);
          } else if (entry.state == CacheState::absent ||
                     entry.state == CacheState::ssd_loading) {
            Telemetry::add(metrics.acquire_ssd_misses_);
          }
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

  std::optional<HostExpertLease> try_acquire_host(
      const ExpertKey& key, const PayloadRecord& record,
      bool record_access) {
    std::lock_guard lock(mutex);
    if (shutting_down) return std::nullopt;
    const auto iterator = entries.find(key);
    if (iterator == entries.end()) return std::nullopt;
    auto& entry = *iterator->second;
    if (record.source_abi != kExpertSourceAbiExpertPackV1 ||
        !same_record(entry.record, record) || !entry.host_copy ||
        !entry.validated_sections ||
        (entry.state != CacheState::ram_ready &&
         entry.state != CacheState::vram_ready)) {
      return std::nullopt;
    }
    if (record_access) touch_locked(entry);
    ++entry.references;
    Telemetry::add(metrics.acquire_ram_hits_);
    auto weak = weak_from_this();
    return HostExpertLease(
        entry.host_copy, *entry.validated_sections,
        [weak, key]() noexcept {
          if (auto core = weak.lock()) core->release_reference(key);
        });
  }

  bool record_access(const ExpertKey& key, std::uint32_t count) {
    if (count == 0) return false;
    std::lock_guard lock(mutex);
    if (shutting_down) return false;
    const auto iterator = entries.find(key);
    if (iterator == entries.end()) return false;
    for (std::uint32_t access = 0; access < count; ++access) {
      touch_locked(*iterator->second);
    }
    return true;
  }

  std::uint64_t record_accesses(std::span<const ExpertAccess> accesses) {
    std::lock_guard lock(mutex);
    if (shutting_down) return 0;
    std::uint64_t recorded = 0;
    for (const auto& access : accesses) {
      if (access.count == 0) continue;
      const auto iterator = entries.find(access.key);
      if (iterator == entries.end()) continue;
      for (std::uint32_t item = 0; item < access.count; ++item) {
        touch_locked(*iterator->second);
      }
      add_routing_score_locked(*iterator->second, access.routing_score_sum,
                               access.routing_score_max);
      recorded += access.count;
    }
    return recorded;
  }

  bool vram_admission_would_improve(const ExpertKey& key,
                                    const PayloadRecord& record) {
    std::lock_guard lock(mutex);
    if (shutting_down) return false;
    const auto iterator = entries.find(key);
    if (iterator == entries.end()) return false;
    const auto& entry = *iterator->second;
    if (!same_record(entry.record, record) || entry.device ||
        (!entry.host && !entry.host_copy) ||
        entry.state != CacheState::ram_ready) {
      return false;
    }
    const auto need = device_bytes(entry.record);
    if (vram_bytes + need <= config.vram.high_watermark_bytes) return true;
    const auto deficit =
        vram_bytes + need - config.vram.high_watermark_bytes;
    const auto entry_temperature = temperature_locked(entry);
    std::uint64_t colder_bytes = 0;
    for (const auto& [candidate_key, candidate] : entries) {
      if (candidate_key == key || !candidate->device ||
          candidate->references != 0 || !candidate->waiters.empty() ||
          temperature_locked(*candidate) >= entry_temperature ||
          candidate->state == CacheState::gpu_uploading ||
          candidate->state == CacheState::failed) {
        continue;
      }
      colder_bytes += candidate->vram_reserved;
      if (colder_bytes >= deficit) return true;
    }
    return false;
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

  CacheUsage trim_to(std::uint64_t ram_target,
                     std::uint64_t vram_target) {
    std::lock_guard lock(mutex);
    bool progress = true;
    while (progress && (ram_bytes > ram_target || vram_bytes > vram_target)) {
      progress = evict_one_locked(ram_bytes > ram_target,
                                  vram_bytes > vram_target, nullptr);
    }
    return {ram_bytes, vram_bytes};
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
  std::shared_ptr<IDeviceResidencyDirectory> directory;
  mutable std::mutex mutex;
  std::map<ExpertKey, std::shared_ptr<Entry>> entries;
  Telemetry metrics;
  std::atomic<std::uint64_t> next_waiter{1};
  std::uint64_t access_clock{};
  std::uint64_t ram_bytes{};
  std::uint64_t vram_bytes{};
  std::uint64_t vram_resident_bytes{};
  std::uint64_t vram_transient_bytes{};
  std::vector<std::uint64_t> ram_partition_bytes;
  std::vector<std::uint64_t> vram_partition_bytes;
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

HostExpertLease::HostExpertLease(
    std::shared_ptr<const std::vector<std::byte>> bytes,
    ExpertSections sections, std::function<void()> release) noexcept
    : bytes_(std::move(bytes)), sections_(sections),
      release_(std::move(release)) {}

HostExpertLease::HostExpertLease(HostExpertLease&& other) noexcept
    : bytes_(std::move(other.bytes_)), sections_(other.sections_),
      release_(std::move(other.release_)) {}

HostExpertLease& HostExpertLease::operator=(HostExpertLease&& other) noexcept {
  if (this != &other) {
    reset();
    bytes_ = std::move(other.bytes_);
    sections_ = other.sections_;
    release_ = std::move(other.release_);
  }
  return *this;
}

HostExpertLease::~HostExpertLease() { reset(); }

HostExpertLease::operator bool() const noexcept { return bytes_ != nullptr; }

std::span<const std::byte> HostExpertLease::bytes() const noexcept {
  if (!bytes_) return {};
  return *bytes_;
}

const ExpertSections& HostExpertLease::sections() const noexcept {
  return sections_;
}

void HostExpertLease::reset() noexcept {
  bytes_.reset();
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
                         std::shared_ptr<FixedBufferPool> buffers,
                         std::shared_ptr<IDeviceResidencyDirectory> directory)
    : core_(std::make_shared<ExpertCacheCore>(config, std::move(storage),
                                               std::move(uploader),
                                               std::move(buffers),
                                               std::move(directory))) {}

ExpertCache::~ExpertCache() {
  if (core_) {
    core_->shutdown();
  }
}

AcquireHandle ExpertCache::acquire(const ExpertKey& key,
                                   const PayloadRecord& record) {
  return core_->acquire(key, record);
}

std::optional<HostExpertLease> ExpertCache::try_acquire_host(
    const ExpertKey& key, const PayloadRecord& record, bool record_access) {
  return core_->try_acquire_host(key, record, record_access);
}

bool ExpertCache::record_access(const ExpertKey& key, std::uint32_t count) {
  return core_->record_access(key, count);
}

std::uint64_t ExpertCache::record_accesses(
    std::span<const ExpertAccess> accesses) {
  return core_->record_accesses(accesses);
}

bool ExpertCache::vram_admission_would_improve(
    const ExpertKey& key, const PayloadRecord& record) {
  return core_->vram_admission_would_improve(key, record);
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
                            entry.device != nullptr, entry.frequency,
                            entry.routing_score_mass_q20,
                            entry.routing_score_peak_q20,
                            core_->temperature_locked(entry)};
}

TelemetrySnapshot ExpertCache::telemetry() const noexcept {
  return core_->metrics.snapshot();
}

CacheUsage ExpertCache::usage() const noexcept {
  const auto snapshot = core_->metrics.snapshot();
  return {snapshot.ram_bytes, snapshot.vram_bytes};
}

std::uint64_t ExpertCache::trim() {
  const auto before = usage();
  const auto after = core_->trim_to(0U, 0U);
  return before.ram_bytes - after.ram_bytes +
         before.vram_bytes - after.vram_bytes;
}

CacheUsage ExpertCache::trim_to(std::uint64_t ram_target_bytes,
                                std::uint64_t vram_target_bytes) {
  return core_->trim_to(ram_target_bytes, vram_target_bytes);
}

}  // namespace expert::runtime
