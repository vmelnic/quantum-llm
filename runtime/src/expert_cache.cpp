#include "expert/runtime/expert_cache.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
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
         left.hidden == right.hidden &&
         left.intermediate == right.intermediate &&
         left.quant_block_size == right.quant_block_size &&
         left.source_abi == right.source_abi &&
         left.record_abi == right.record_abi &&
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
    Waiter(std::uint64_t waiter_id, ExpertAcquireOptions acquire_options)
        : id(waiter_id), options(acquire_options),
          created_at(std::chrono::steady_clock::now()) {}
    std::uint64_t id{};
    ExpertAcquireOptions options;
    std::chrono::steady_clock::time_point created_at;
    std::promise<AcquireResult> promise;
  };

  struct HostWaiter final {
    HostWaiter(std::uint64_t waiter_id, HostPreloadOptions preload_options)
        : id(waiter_id), options(preload_options),
          created_at(std::chrono::steady_clock::now()) {}
    std::uint64_t id{};
    HostPreloadOptions options;
    std::chrono::steady_clock::time_point created_at;
    std::promise<HostPreloadResult> promise;
  };

  struct Entry final {
    ExpertKey key;
    PayloadRecord record;
    CacheState state{CacheState::absent};
    std::map<std::uint64_t, std::shared_ptr<Waiter>> waiters;
    std::map<std::uint64_t, std::shared_ptr<HostWaiter>> host_waiters;
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
    std::uint32_t successful_reads{};
    ExpertRequestPriority staging_priority{ExpertRequestPriority::demand};
    ExpertRequestPriority upload_priority{ExpertRequestPriority::demand};
    bool ram_protected{};
    bool ram_protected_requested{};
    bool ram_protection_persistent{};
    bool vram_resident{true};
    bool vram_resident_requested{};
    bool force_host_retention{};
    bool host_preloaded{};
    bool preparing_host{};
    OperationId io_operation{};
    OperationId upload_operation{};
    std::chrono::steady_clock::time_point load_started_at{};
    std::chrono::steady_clock::time_point upload_started_at{};
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
            config.vram.high_watermark_bytes ||
        config.placement.ram_protected_bytes >=
            config.ram.high_watermark_bytes) {
      throw std::invalid_argument("invalid expert cache configuration");
    }
    host_worker = std::thread([this] { host_worker_loop(); });
  }

  [[nodiscard]] std::unique_lock<std::mutex> acquire_lock() const {
    const auto started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex);
    const auto waited = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    Telemetry::add(metrics.mutex_acquisitions_);
    Telemetry::add(metrics.mutex_wait_ns_, waited);
    Telemetry::maximize(metrics.mutex_wait_max_ns_, waited);
    return lock;
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

  [[nodiscard]] static BufferPoolClass buffer_class(
      ExpertRequestPriority priority) noexcept {
    return priority == ExpertRequestPriority::demand
               ? BufferPoolClass::demand
               : BufferPoolClass::background;
  }

  [[nodiscard]] static bool higher_priority(
      ExpertRequestPriority left,
      ExpertRequestPriority right) noexcept {
    return static_cast<std::uint8_t>(left) >
           static_cast<std::uint8_t>(right);
  }

  [[nodiscard]] static std::size_t priority_index(
      ExpertRequestPriority priority) noexcept {
    return static_cast<std::size_t>(priority);
  }

  template <typename WaiterType>
  void record_waiter_completion(const WaiterType& waiter) noexcept {
    const auto index = priority_index(waiter.options.priority);
    Telemetry::add(metrics.completed_waiters_by_priority_[index]);
    Telemetry::add(
        metrics.waiter_wait_ns_by_priority_[index],
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - waiter.created_at)
                .count()));
  }

  template <typename WaiterType>
  void record_waiter_failure(const WaiterType& waiter) noexcept {
    Telemetry::add(metrics.failed_waiters_by_priority_[priority_index(
        waiter.options.priority)]);
  }

  [[nodiscard]] ExpertRequestPriority effective_priority_locked(
      const Entry& entry) const noexcept {
    auto result = ExpertRequestPriority::warm;
    for (const auto& [id, waiter] : entry.waiters) {
      (void)id;
      if (higher_priority(waiter->options.priority, result))
        result = waiter->options.priority;
    }
    for (const auto& [id, waiter] : entry.host_waiters) {
      (void)id;
      if (higher_priority(waiter->options.priority, result))
        result = waiter->options.priority;
    }
    return result;
  }

  [[nodiscard]] bool has_waiters_locked(const Entry& entry) const noexcept {
    return !entry.waiters.empty() || !entry.host_waiters.empty();
  }

  void refresh_task_indexes_locked(const Entry& entry) {
    for (auto& keys : read_task_keys) keys.erase(entry.key);
    for (auto& keys : upload_task_keys) keys.erase(entry.key);
    if (entry.abandon || !has_waiters_locked(entry)) return;
    const auto priority = priority_index(effective_priority_locked(entry));
    if (entry.state == CacheState::absent) {
      read_task_keys[priority].insert(entry.key);
    } else if (entry.state == CacheState::ram_ready &&
               !entry.waiters.empty() &&
               (entry.host || entry.host_copy)) {
      upload_task_keys[priority].insert(entry.key);
    }
  }

  void refresh_indexes_locked(const Entry& entry) {
    refresh_task_indexes_locked(entry);
  }

  [[nodiscard]] bool device_allows_host_retention_locked(
      const Entry& entry) const noexcept {
    return std::any_of(entry.waiters.begin(), entry.waiters.end(),
                       [](const auto& item) {
                         return item.second->options.allow_host_retention;
                       });
  }

  void reserve_ram_locked(Entry& entry, std::uint64_t bytes,
                          bool protected_class) noexcept {
    entry.ram_protected = protected_class;
    entry.ram_reserved += bytes;
    ram_bytes += bytes;
    if (protected_class)
      ram_protected_bytes += bytes;
    else
      ram_probationary_bytes += bytes;
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
    refresh_indexes_locked(entry);
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
    Telemetry::set(metrics.ram_probationary_bytes_, ram_probationary_bytes);
    Telemetry::set(metrics.ram_protected_bytes_, ram_protected_bytes);
    Telemetry::set(metrics.vram_bytes_, vram_bytes);
    Telemetry::set(metrics.vram_resident_bytes_, vram_resident_bytes);
    Telemetry::set(metrics.vram_transient_bytes_, vram_transient_bytes);
    const auto staging = buffers->bytes_in_use();
    Telemetry::set(metrics.staging_bytes_, staging);
    Telemetry::maximize(metrics.ram_high_water_, ram_bytes);
    Telemetry::maximize(metrics.ram_probationary_high_water_,
                        ram_probationary_bytes);
    Telemetry::maximize(metrics.ram_protected_high_water_,
                        ram_protected_bytes);
    Telemetry::maximize(metrics.vram_high_water_, vram_bytes);
    Telemetry::maximize(metrics.vram_resident_high_water_,
                        vram_resident_bytes);
    Telemetry::maximize(metrics.vram_transient_high_water_,
                        vram_transient_bytes);
    Telemetry::maximize(metrics.staging_high_water_, staging);
  }

  void release_staging_locked(Entry& entry) noexcept {
    entry.host.reset();
  }

  void release_host_locked(Entry& entry) noexcept {
    if (entry.host_preloaded && entry.host_copy) {
      Telemetry::add(metrics.preloaded_host_wasted_);
      Telemetry::add(metrics.preloaded_host_wasted_bytes_,
                     entry.ram_reserved);
      entry.host_preloaded = false;
    }
    release_staging_locked(entry);
    entry.host_copy.reset();
    entry.validated_sections.reset();
    if (entry.ram_reserved != 0) {
      ram_partition_bytes[partition_for(entry.key)] -= entry.ram_reserved;
      ram_bytes -= entry.ram_reserved;
      if (entry.ram_protected)
        ram_protected_bytes -= entry.ram_reserved;
      else
        ram_probationary_bytes -= entry.ram_reserved;
      entry.ram_reserved = 0;
    }
    entry.ram_protected = false;
    if (!entry.ram_protection_persistent)
      entry.ram_protected_requested = false;
  }

  void release_device_locked(Entry& entry,
                             bool directory_entry_retired = false) noexcept {
    if (entry.device && directory && !directory_entry_retired) {
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

  void add_reference_locked(Entry& entry) noexcept {
    if (entry.references++ != 0 || !entry.device) return;
    ++vram_referenced_entries;
    vram_referenced_bytes += entry.vram_reserved;
    if (!entry.vram_resident)
      vram_transient_referenced_bytes += entry.vram_reserved;
    Telemetry::set(metrics.vram_referenced_entries_,
                   vram_referenced_entries);
    Telemetry::set(metrics.vram_referenced_bytes_, vram_referenced_bytes);
    Telemetry::set(metrics.vram_transient_referenced_bytes_,
                   vram_transient_referenced_bytes);
  }

  void remove_reference_locked(Entry& entry) noexcept {
    if (entry.references == 0 || --entry.references != 0 || !entry.device)
      return;
    --vram_referenced_entries;
    vram_referenced_bytes -= entry.vram_reserved;
    if (!entry.vram_resident)
      vram_transient_referenced_bytes -= entry.vram_reserved;
    Telemetry::set(metrics.vram_referenced_entries_,
                   vram_referenced_entries);
    Telemetry::set(metrics.vram_referenced_bytes_, vram_referenced_bytes);
    Telemetry::set(metrics.vram_transient_referenced_bytes_,
                   vram_transient_referenced_bytes);
  }

  void satisfy_ready_waiters_locked(const std::shared_ptr<Entry>& entry) {
    auto weak = weak_from_this();
    for (auto& [id, waiter] : entry->waiters) {
      (void)id;
      record_waiter_completion(*waiter);
      add_reference_locked(*entry);
      auto release = [weak, key = entry->key]() noexcept {
        if (auto core = weak.lock()) {
          core->release_reference(key);
        }
      };
      waiter->promise.set_value(
          {Status::success(), ExpertLease(entry->device, std::move(release))});
    }
    entry->waiters.clear();
    refresh_task_indexes_locked(*entry);
  }

  void satisfy_host_waiters_locked(Entry& entry) {
    if (!entry.host_copy || !entry.validated_sections) return;
    auto weak = weak_from_this();
    for (auto& [id, waiter] : entry.host_waiters) {
      (void)id;
      record_waiter_completion(*waiter);
      HostExpertLease lease;
      if (waiter->options.acquire_lease) {
        add_reference_locked(entry);
        lease = HostExpertLease(
            entry.host_copy, *entry.validated_sections,
            entry.validated_compact, entry.record.source_abi,
            [weak, key = entry.key]() noexcept {
              if (auto core = weak.lock()) core->release_reference(key);
            });
      }
      waiter->promise.set_value(
          {Status::success(), true, std::move(lease)});
      Telemetry::add(metrics.host_preloads_completed_);
    }
    entry.host_preloaded = true;
    entry.host_waiters.clear();
    refresh_task_indexes_locked(entry);
  }

  void fail_waiters_locked(Entry& entry, const Status& status) {
    for (auto& [id, waiter] : entry.waiters) {
      (void)id;
      record_waiter_completion(*waiter);
      record_waiter_failure(*waiter);
      waiter->promise.set_value({status, {}});
    }
    entry.waiters.clear();
    for (auto& [id, waiter] : entry.host_waiters) {
      (void)id;
      record_waiter_completion(*waiter);
      record_waiter_failure(*waiter);
      waiter->promise.set_value({status, false});
    }
    entry.host_waiters.clear();
    refresh_task_indexes_locked(entry);
  }

  bool reject_blocked_device_waiters_locked(Entry& entry) {
    bool rejected = false;
    for (auto iterator = entry.waiters.begin();
         iterator != entry.waiters.end();) {
      const auto& waiter = iterator->second;
      if (!waiter->options.fail_fast_on_device_admission) {
        ++iterator;
        continue;
      }
      record_waiter_completion(*waiter);
      record_waiter_failure(*waiter);
      waiter->promise.set_value(
          {Status(ErrorCode::backpressure,
                  "device admission is blocked by the current placement"),
           {}});
      iterator = entry.waiters.erase(iterator);
      Telemetry::add(metrics.device_admission_rejections_);
      rejected = true;
    }
    if (rejected) refresh_task_indexes_locked(entry);
    return rejected;
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
                            maximum_temperature_exclusive = {},
                        std::optional<bool> required_ram_class = {}) {
    const auto ram_quota = partition_quota(
        config.ram, config.placement.ram_shared_burst_bytes);
    const auto vram_quota = partition_quota(
        config.vram, config.placement.vram_shared_burst_bytes);
    std::set<ExpertKey> busy_retirements;
    for (;;) {
      Entry* candidate{};
      std::uint64_t candidate_excess{};
      std::uint64_t candidate_temperature{};
      std::uint64_t scanned_candidates{};
      const auto scan_started = std::chrono::steady_clock::now();
      for (const auto& [key, stored_entry] : entries) {
        auto* entry = stored_entry.get();
        ++scanned_candidates;
        if ((excluded != nullptr && key == *excluded) ||
            busy_retirements.contains(key) || entry->references != 0 ||
            has_waiters_locked(*entry) || entry->preparing_host ||
            entry->state == CacheState::ssd_loading ||
            entry->state == CacheState::gpu_uploading ||
            entry->state == CacheState::failed) {
          continue;
        }
        if (need_vram && entry->device && directory &&
            directory->route_pinned(key)) {
          continue;
        }
        const bool useful = (need_ram && (entry->host || entry->host_copy)) ||
                            (need_vram && entry->device);
        if (!useful) continue;
        if (need_vram && required_vram_class &&
            (!entry->device ||
             entry->vram_resident != *required_vram_class)) {
          continue;
        }
        if (need_ram && required_ram_class &&
            ((!entry->host && !entry->host_copy) ||
             entry->ram_protected != *required_ram_class)) {
          continue;
        }
        const auto entry_temperature = temperature_locked(*entry);
        if (maximum_temperature_exclusive &&
            entry_temperature >= *maximum_temperature_exclusive) {
          continue;
        }
        const auto partition = partition_for(key);
        if (required_partition && partition != *required_partition) continue;
        const auto ram_excess =
            need_ram && ram_partition_bytes[partition] > ram_quota
                ? ram_partition_bytes[partition] - ram_quota
                : 0;
        const auto vram_excess =
            need_vram && vram_partition_bytes[partition] > vram_quota
                ? vram_partition_bytes[partition] - vram_quota
                : 0;
        const auto excess = std::max(ram_excess, vram_excess);
        if (over_quota_only && excess == 0) continue;
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
      Telemetry::add(metrics.eviction_scan_calls_);
      Telemetry::add(metrics.eviction_scan_candidates_, scanned_candidates);
      Telemetry::add(
          metrics.eviction_scan_ns_,
          static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - scan_started)
                  .count()));
      if (!candidate) return false;

      // Do not partially evict RAM before a device retirement is known to be
      // safe. A failed asynchronous pin retirement rejects only this victim;
      // continue scanning instead of stalling the complete admission round.
      if (need_vram && candidate->device && directory &&
          !directory->try_retire(candidate->key)) {
        Telemetry::add(metrics.eviction_retire_retries_);
        busy_retirements.insert(candidate->key);
        continue;
      }
      const auto evicted_ram_bytes =
          need_ram && (candidate->host || candidate->host_copy)
              ? candidate->ram_reserved
              : 0U;
      const auto evicted_vram_bytes =
          need_vram && candidate->device ? candidate->vram_reserved : 0U;
      const auto ram_class = candidate->ram_protected ? 1U : 0U;
      const auto vram_class = candidate->vram_resident ? 1U : 0U;
      if (need_ram && (candidate->host || candidate->host_copy))
        release_host_locked(*candidate);
      if (need_vram && candidate->device) {
        release_device_locked(*candidate, /*directory_entry_retired=*/true);
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
      if (evicted_ram_bytes != 0) {
        Telemetry::add(metrics.ram_evictions_by_class_[ram_class]);
        Telemetry::add(metrics.ram_evicted_bytes_by_class_[ram_class],
                       evicted_ram_bytes);
      }
      if (evicted_vram_bytes != 0) {
        Telemetry::add(metrics.vram_evictions_by_class_[vram_class]);
        Telemetry::add(metrics.vram_evicted_bytes_by_class_[vram_class],
                       evicted_vram_bytes);
      }
      if (required_partition)
        Telemetry::add(metrics.same_partition_evictions_);
      else if (candidate_excess != 0)
        Telemetry::add(metrics.over_quota_evictions_);
      update_usage_locked();
      return true;
    }
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

  bool make_ram_class_capacity_locked(std::uint64_t ram_need,
                                      const ExpertKey& key,
                                      bool protected_class) {
    if (ram_need == 0 || config.placement.ram_protected_bytes == 0)
      return true;
    const auto limit = protected_class
                           ? config.placement.ram_protected_bytes
                           : config.ram.high_watermark_bytes -
                                 config.placement.ram_protected_bytes;
    if (ram_need > limit) return false;
    auto& usage = protected_class ? ram_protected_bytes
                                  : ram_probationary_bytes;
    while (usage + ram_need > limit) {
      if (!evict_one_locked(true, false, &key, {}, false, {}, {},
                            protected_class)) {
        Telemetry::add(metrics.stalled_by_budget_);
        return false;
      }
    }
    return true;
  }

  void promote_ram_locked(Entry& entry, bool force = false) {
    if (!entry.host_copy || entry.ram_protected || entry.ram_reserved == 0 ||
        config.placement.ram_protected_bytes == 0 ||
        !force) {
      return;
    }
    if (!make_ram_class_capacity_locked(entry.ram_reserved, entry.key, true)) {
      Telemetry::add(metrics.ram_promotion_failures_);
      return;
    }
    ram_probationary_bytes -= entry.ram_reserved;
    ram_protected_bytes += entry.ram_reserved;
    entry.ram_protected = true;
    entry.ram_protected_requested = true;
    Telemetry::add(metrics.ram_promotions_);
    update_usage_locked();
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

  void promote_vram_locked(Entry& entry, bool force = false) {
    if (!entry.device || entry.vram_resident || entry.vram_reserved == 0 ||
        config.placement.vram_transient_bytes == 0 ||
        (!force && entry.frequency < 2)) {
      return;
    }
    if (!make_vram_class_capacity_locked(entry.vram_reserved, entry.key,
                                         true, temperature_locked(entry))) {
      Telemetry::add(metrics.vram_promotion_failures_);
      return;
    }
    if (entry.references != 0)
      vram_transient_referenced_bytes -= entry.vram_reserved;
    vram_transient_bytes -= entry.vram_reserved;
    vram_resident_bytes += entry.vram_reserved;
    vram_partition_bytes[partition_for(entry.key)] += entry.vram_reserved;
    entry.vram_resident = true;
    entry.vram_resident_requested = true;
    Telemetry::set(metrics.vram_transient_referenced_bytes_,
                   vram_transient_referenced_bytes);
    Telemetry::add(metrics.vram_promotions_);
    update_usage_locked();
  }

  bool make_capacity_locked(std::uint64_t ram_need, std::uint64_t vram_need,
                            const ExpertKey& key, bool ram_protected,
                            bool resident,
                            std::uint64_t admission_temperature) {
    if (ram_need > config.ram.high_watermark_bytes ||
        vram_need > config.vram.high_watermark_bytes) {
      return false;
    }
    if (!make_partition_capacity_locked(ram_need, key) ||
        !make_ram_class_capacity_locked(ram_need, key, ram_protected) ||
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
    if (ram_pressured && config.placement.layer_partition_count == 1 &&
        config.placement.ram_protected_bytes == 0) {
      // Continue evicting toward low watermarks when safe, creating headroom
      // for a burst instead of oscillating at the high mark.
      while (ram_bytes > config.ram.low_watermark_bytes &&
             evict_one_locked(true, false, &key)) {
      }
    }
    return ram_bytes + ram_need <= config.ram.high_watermark_bytes &&
           vram_bytes + vram_need <= config.vram.high_watermark_bytes;
  }

  bool make_device_capacity_locked(Entry& entry, std::uint64_t ram_need,
                                   bool ram_protected,
                                   bool& resident) {
    const auto vram_need = device_bytes(entry.record);
    resident = config.placement.vram_transient_bytes == 0 ||
               entry.vram_resident_requested || entry.frequency >= 2;
    bool admitted = make_capacity_locked(
        ram_need, vram_need, entry.key, ram_protected, resident,
        temperature_locked(entry));
    if (!admitted && resident) {
      admitted = make_capacity_locked(
          ram_need, vram_need, entry.key, ram_protected, false,
          temperature_locked(entry));
      resident = false;
    } else if (!admitted && !resident) {
      const auto resident_limit = config.vram.high_watermark_bytes -
                                  config.placement.vram_transient_bytes;
      if (vram_resident_bytes + vram_need <= resident_limit) {
        admitted = make_capacity_locked(
            ram_need, vram_need, entry.key, ram_protected, true,
            temperature_locked(entry));
        resident = admitted;
      }
    }
    return admitted;
  }

  Task next_task_locked() {
    const auto selection_started = std::chrono::steady_clock::now();
    std::uint64_t selection_candidates = 0U;
    const auto finish_selection = [&](Task task) {
      Telemetry::add(metrics.task_selection_calls_);
      Telemetry::add(metrics.task_selection_candidates_,
                     selection_candidates);
      Telemetry::add(
          metrics.task_selection_ns_,
          static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - selection_started)
                  .count()));
      return task;
    };
    constexpr std::array priorities{
        ExpertRequestPriority::demand,
        ExpertRequestPriority::prefetch,
        ExpertRequestPriority::warm,
    };
    for (const auto priority : priorities) {
      const auto slot = priority_index(priority);
      // Upload RAM-ready demand before consuming another staging slot.
      for (auto key_iterator = upload_task_keys[slot].begin();
           key_iterator != upload_task_keys[slot].end();) {
        const auto key = *key_iterator++;
        const auto entry_iterator = entries.find(key);
        if (entry_iterator == entries.end())
          throw std::logic_error("upload task index references a missing entry");
        const auto& entry = entry_iterator->second;
        ++selection_candidates;
        if (entry->waiters.empty() || entry->abandon ||
            effective_priority_locked(*entry) != priority ||
            entry->state != CacheState::ram_ready ||
            (!entry->host && !entry->host_copy)) {
          refresh_task_indexes_locked(*entry);
          continue;
        }
        if (entry->vram_reserved == 0) {
          bool resident{};
          if (!make_device_capacity_locked(*entry, 0U, false, resident)) {
            reject_blocked_device_waiters_locked(*entry);
            continue;
          }
          reserve_vram_locked(*entry, device_bytes(entry->record), resident);
          update_usage_locked();
        }
        transition_locked(*entry, CacheState::gpu_uploading);
        entry->upload_priority = priority;
        entry->upload_started_at = std::chrono::steady_clock::now();
        Telemetry::add(metrics.upload_started_);
        Telemetry::add(
            metrics.uploads_started_by_priority_[priority_index(priority)]);
        return finish_selection({TaskKind::upload, entry});
      }
      for (auto key_iterator = read_task_keys[slot].begin();
           key_iterator != read_task_keys[slot].end();) {
        const auto key = *key_iterator++;
        const auto entry_iterator = entries.find(key);
        if (entry_iterator == entries.end())
          throw std::logic_error("read task index references a missing entry");
        const auto& entry = entry_iterator->second;
        ++selection_candidates;
        if (!has_waiters_locked(*entry) || entry->abandon ||
            effective_priority_locked(*entry) != priority ||
            entry->state != CacheState::absent) {
          refresh_task_indexes_locked(*entry);
          continue;
        }
        if (entry->record.stored_bytes > buffers->slot_bytes() ||
            entry->record.stored_bytes >
                std::numeric_limits<std::size_t>::max()) {
          fail_entry_locked(*entry,
                            Status(ErrorCode::backpressure,
                                   "expert record exceeds fixed staging slot"));
          continue;
        }
        if (!entry->waiters.empty() &&
            device_bytes(entry->record) > config.vram.high_watermark_bytes) {
          // A host-only preload is valid without any device capacity, but a
          // device waiter whose immutable allocation can never fit must not
          // spend storage bandwidth before it is cancelled or rejected.
          continue;
        }
        const bool protected_ram = entry->ram_protected_requested;
        bool resident{};
        const bool device_demand = !entry->waiters.empty();
        const bool admitted = device_demand
            ? make_device_capacity_locked(
                  *entry, entry->record.stored_bytes, protected_ram, resident)
            : make_capacity_locked(entry->record.stored_bytes, 0U,
                                   entry->key, protected_ram, false,
                                   temperature_locked(*entry));
        if (!admitted) {
          if (device_demand) reject_blocked_device_waiters_locked(*entry);
          continue;
        }
        auto host = buffers->try_acquire(
            static_cast<std::size_t>(entry->record.stored_bytes),
            buffer_class(priority));
        if (!host) {
          Telemetry::add(metrics.stalled_by_budget_);
          Telemetry::add(
              metrics.staging_stalls_by_priority_[priority_index(priority)]);
          continue;
        }
        entry->host = std::move(host);
        entry->staging_priority = priority;
        reserve_ram_locked(*entry, entry->record.stored_bytes,
                           protected_ram);
        if (device_demand) {
          reserve_vram_locked(*entry, device_bytes(entry->record), resident);
        }
        if (entry->successful_reads != 0) {
          Telemetry::add(metrics.reload_count_);
          Telemetry::add(metrics.reread_bytes_, entry->record.stored_bytes);
        }
        transition_locked(*entry, CacheState::ssd_loading);
        entry->load_started_at = std::chrono::steady_clock::now();
        Telemetry::add(metrics.load_started_);
        Telemetry::add(
            metrics.reads_started_by_priority_[priority_index(priority)]);
        Telemetry::add(metrics.requested_bytes_, entry->record.stored_bytes);
        update_usage_locked();
        return finish_selection({TaskKind::read, entry});
      }
    }
    return finish_selection({});
  }

  void drive() {
    for (;;) {
      Task task;
      {
        auto lock = acquire_lock();
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
        auto lock = acquire_lock();
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
        auto lock = acquire_lock();
        fail_entry_locked(*entry,
                          Status(ErrorCode::internal,
                                 "RAM-ready expert has no host bytes"));
      }
      drive();
      return;
    }
    if (!entry->validated_sections) {
      {
        auto lock = acquire_lock();
        fail_entry_locked(*entry,
                          Status(ErrorCode::internal,
                                 "RAM-ready expert was not host-validated"));
      }
      drive();
      return;
    }
    const auto sections = *entry->validated_sections;
    const auto compact = entry->validated_compact;
    if (!entry->host && entry->host_copy) {
      Telemetry::add(metrics.validated_ram_reuses_);
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
        auto lock = acquire_lock();
      if (entry->state == CacheState::gpu_uploading) {
        entry->upload_operation = operation;
        cancel_now = entry->abandon;
      }
    }
    if (cancel_now) {
      uploader->cancel(operation);
    }
  }

  void prepare_host_entry(const std::shared_ptr<Entry>& entry) {
    const auto count = static_cast<std::size_t>(entry->record.stored_bytes);
    const auto staging = entry->host ? entry->host->buffer() : MutableBuffer{};
    if (!staging.data || staging.capacity < count) {
        auto lock = acquire_lock();
      if (entry->state == CacheState::ssd_loading) {
        entry->preparing_host = false;
        fail_entry_locked(*entry,
                          Status(ErrorCode::internal,
                                 "host preparation lost its staging record"));
      }
      return;
    }
    const std::span<const std::byte> bytes(staging.data, count);
    const auto validation_started = std::chrono::steady_clock::now();
    const auto validated = validate_expert_admission(
        bytes, entry->key, entry->record, !config.trusted_immutable_source);
    const auto validation_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - validation_started)
            .count());
    Telemetry::add(metrics.record_validations_);
    Telemetry::add(metrics.host_validation_ns_, validation_ns);
    Telemetry::add(
        metrics.host_validation_ns_by_priority_[priority_index(
            entry->staging_priority)],
        validation_ns);
    if (!validated.status.ok()) {
      Telemetry::add(metrics.host_validation_failures_);
      Telemetry::add(metrics.validation_errors_by_priority_[priority_index(
          entry->staging_priority)]);
        auto lock = acquire_lock();
      if (entry->state == CacheState::ssd_loading) {
        entry->preparing_host = false;
        Telemetry::add(metrics.checksum_errors_);
        fail_entry_locked(*entry, validated.status);
      }
      return;
    }

    std::shared_ptr<std::vector<std::byte>> retained;
    for (;;) {
      bool need_copy = false;
      {
        auto lock = acquire_lock();
        if (entry->state != CacheState::ssd_loading || shutting_down) return;
        need_copy = config.retain_host_copy &&
            (entry->force_host_retention ||
             (device_allows_host_retention_locked(*entry) &&
              entry->frequency >=
                  config.ram_retention_minimum_frequency));
      }
      if (need_copy && !retained) {
        const auto copy_started = std::chrono::steady_clock::now();
        try {
          retained = std::make_shared<std::vector<std::byte>>(count);
          std::memcpy(retained->data(), bytes.data(), count);
          Telemetry::add(metrics.host_copy_bytes_, count);
          Telemetry::add(metrics.host_copy_bytes_by_priority_[priority_index(
                             entry->staging_priority)],
                         count);
          const auto copy_ns = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - copy_started)
                  .count());
          Telemetry::add(
              metrics.ram_retention_copy_ns_, copy_ns);
          Telemetry::add(
              metrics.ram_retention_copy_ns_by_priority_[priority_index(
                  entry->staging_priority)],
              copy_ns);
        } catch (const std::bad_alloc&) {
        auto lock = acquire_lock();
          if (entry->state == CacheState::ssd_loading) {
            entry->preparing_host = false;
            fail_entry_locked(
                *entry, Status(ErrorCode::backpressure,
                               "cannot allocate pageable RAM cache copy"));
          }
          return;
        }
      }

      auto lock = acquire_lock();
      if (entry->state != CacheState::ssd_loading || shutting_down) return;
      const bool copy_now_required = config.retain_host_copy &&
          (entry->force_host_retention ||
           (device_allows_host_retention_locked(*entry) &&
            entry->frequency >= config.ram_retention_minimum_frequency));
      if (copy_now_required && !retained) {
        lock.unlock();
        continue;
      }
      entry->preparing_host = false;
      if (entry->abandon || !has_waiters_locked(*entry)) {
        transition_locked(*entry, CacheState::absent);
        release_host_locked(*entry);
        release_device_locked(*entry);
        update_usage_locked();
        break;
      }
      entry->validated_sections = validated.target;
      entry->validated_compact = validated.compact;
      entry->host_copy = std::move(retained);
      ++entry->successful_reads;
      transition_locked(*entry, CacheState::ram_ready);
      Telemetry::add(metrics.load_completed_);
      Telemetry::add(metrics.useful_bytes_, entry->record.stored_bytes);
      if (entry->ram_protected_requested) promote_ram_locked(*entry, true);
      satisfy_host_waiters_locked(*entry);
      const bool joined_demand = std::any_of(
          entry->waiters.begin(), entry->waiters.end(), [](const auto& item) {
            return item.second->options.priority ==
                   ExpertRequestPriority::demand;
          });
      if (entry->host_preloaded && joined_demand) {
        Telemetry::add(metrics.preloaded_host_useful_);
        Telemetry::add(metrics.preloaded_host_useful_bytes_,
                       entry->ram_reserved);
        entry->host_preloaded = false;
      }
      if (entry->waiters.empty()) {
        release_staging_locked(*entry);
        if (!entry->host_copy) {
          release_host_locked(*entry);
          transition_locked(*entry, CacheState::absent);
        }
      }
      update_usage_locked();
      break;
    }
    drive();
  }

  void host_worker_loop() {
    for (;;) {
      std::shared_ptr<Entry> entry;
      {
      auto lock = acquire_lock();
        host_condition.wait(lock, [&] {
          return host_worker_stop || !host_prepare_queue.empty();
        });
        if (host_worker_stop && host_prepare_queue.empty()) return;
        entry = std::move(host_prepare_queue.front());
        host_prepare_queue.pop_front();
      }
      prepare_host_entry(entry);
    }
  }

  void read_complete(const ExpertKey& key, ReadResult result) {
    bool queued_prepare = false;
    {
      auto lock = acquire_lock();
      const auto iterator = entries.find(key);
      if (iterator == entries.end() ||
          iterator->second->state != CacheState::ssd_loading) {
        return;
      }
      auto& entry = *iterator->second;
      const auto completion_priority = priority_index(entry.staging_priority);
      Telemetry::add(
          metrics.reads_completed_by_priority_[completion_priority]);
      if (entry.load_started_at != std::chrono::steady_clock::time_point{}) {
        const auto waited = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - entry.load_started_at)
                .count());
        Telemetry::add(
            metrics.storage_wait_ns_, waited);
        Telemetry::add(
            metrics.storage_wait_ns_by_priority_[completion_priority], waited);
        entry.load_started_at = {};
      }
      entry.io_operation = 0;
      Telemetry::add(metrics.read_bytes_, result.read_bytes);
      Telemetry::add(
          metrics.read_bytes_by_priority_[completion_priority],
          result.read_bytes);
      if (entry.abandon || !has_waiters_locked(entry)) {
        transition_locked(entry, CacheState::absent);
        release_host_locked(entry);
        release_device_locked(entry);
        update_usage_locked();
      } else if (result.status.code() == ErrorCode::cancelled) {
        transition_locked(entry, CacheState::absent);
        release_host_locked(entry);
        release_device_locked(entry);
        entry.abandon = false;
        update_usage_locked();
      } else if (!result.status.ok()) {
        Telemetry::add(metrics.io_errors_by_priority_[completion_priority]);
        if (result.status.code() == ErrorCode::short_read) {
          Telemetry::add(metrics.short_read_errors_);
        } else if (result.status.code() != ErrorCode::cancelled) {
          Telemetry::add(metrics.io_errors_);
        }
        fail_entry_locked(entry, result.status);
      } else if (result.requested_bytes != entry.record.stored_bytes ||
                 result.read_bytes != entry.record.stored_bytes) {
        Telemetry::add(metrics.io_errors_by_priority_[completion_priority]);
        Telemetry::add(metrics.short_read_errors_);
        fail_entry_locked(
            entry, Status(ErrorCode::short_read,
                          "storage completed without the complete expert record"));
      } else {
        entry.preparing_host = true;
        host_prepare_queue.push_back(iterator->second);
        queued_prepare = true;
      }
    }
    if (queued_prepare) host_condition.notify_one();
    drive();
  }

  void upload_complete(const ExpertKey& key, UploadResult result) {
    {
      auto lock = acquire_lock();
      const auto iterator = entries.find(key);
      if (iterator == entries.end() ||
          iterator->second->state != CacheState::gpu_uploading) {
        return;
      }
      auto& entry = *iterator->second;
      if (entry.upload_started_at != std::chrono::steady_clock::time_point{}) {
        const auto waited = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - entry.upload_started_at)
                .count());
        Telemetry::add(
            metrics.upload_wait_ns_, waited);
        Telemetry::add(metrics.upload_wait_ns_by_priority_[priority_index(
                           entry.upload_priority)],
                       waited);
        entry.upload_started_at = {};
      }
      entry.upload_operation = 0;
      if (entry.abandon || entry.waiters.empty()) {
        transition_locked(entry, CacheState::ram_ready);
        release_device_locked(entry);
        release_staging_locked(entry);
        if (!entry.host_copy) {
          release_host_locked(entry);
          transition_locked(entry, CacheState::absent);
        }
        update_usage_locked();
      } else if (!result.status.ok() || !result.allocation ||
                 result.uploaded_bytes == 0 ||
                 result.uploaded_bytes != result.allocation->bytes() ||
                 result.allocation->bytes() > entry.vram_reserved) {
        Telemetry::add(metrics.upload_errors_);
        Telemetry::add(metrics.upload_errors_by_priority_[priority_index(
            entry.upload_priority)]);
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
            Telemetry::add(metrics.upload_errors_by_priority_[priority_index(
                entry.upload_priority)]);
            fail_entry_locked(entry, published);
          }
        }
        if (entry.state != CacheState::failed) {
          entry.device = std::move(result.allocation);
          transition_locked(entry, CacheState::vram_ready);
          Telemetry::add(metrics.upload_completed_);
          Telemetry::add(metrics.uploads_completed_by_priority_[priority_index(
              entry.upload_priority)]);
          Telemetry::add(metrics.uploaded_bytes_, result.uploaded_bytes);
          Telemetry::add(metrics.uploaded_bytes_by_priority_[priority_index(
                             entry.upload_priority)],
                         result.uploaded_bytes);
          if (!config.retain_host_copy || !entry.host_copy) {
            // Either the RAM tier is disabled or the record stayed
            // pack-resident (below the retention frequency): no pageable copy
            // exists, so drop the staging lease together with its RAM
            // reservation instead of charging bytes nothing backs.
            release_host_locked(entry);
          } else {
            // The long-lived RAM tier is pageable. Pinned buffers remain a
            // bounded staging resource and return to the fixed pool after H2D.
            release_staging_locked(entry);
          }
          update_usage_locked();
          satisfy_ready_waiters_locked(iterator->second);
        }
      }
    }
    drive();
  }

  AcquireHandle acquire(const ExpertKey& key, const PayloadRecord& record,
                        ExpertAcquireOptions options) {
    auto waiter =
        std::make_shared<Waiter>(next_waiter.fetch_add(1), options);
    auto future = waiter->promise.get_future();
    bool should_drive = false;
    {
      auto lock = acquire_lock();
      Telemetry::add(
          metrics.device_requests_[priority_index(options.priority)]);
      if (shutting_down) {
        record_waiter_completion(*waiter);
        Telemetry::add(metrics.cancellations_by_priority_[priority_index(
            options.priority)]);
        Telemetry::add(metrics.cancellation_count_);
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
        const auto previous_priority = has_waiters_locked(entry)
                                           ? effective_priority_locked(entry)
                                           : options.priority;
        if (options.record_access) {
          touch_locked(entry);
          promote_ram_locked(entry);
          promote_vram_locked(entry);
        }
        if (options.protect_vram) {
          entry.vram_resident_requested = true;
          promote_vram_locked(entry, true);
        }
        if (!same_record(entry.record, record)) {
          record_waiter_completion(*waiter);
          record_waiter_failure(*waiter);
          waiter->promise.set_value(
              {Status(ErrorCode::invalid_argument,
                      "same expert key references different immutable records"),
               {}});
        } else if (entry.state == CacheState::failed) {
          record_waiter_completion(*waiter);
          record_waiter_failure(*waiter);
          waiter->promise.set_value(
              {Status(ErrorCode::checksum_mismatch,
                      "expert is quarantined after a failed load"),
               {}});
        } else if (entry.state == CacheState::vram_ready && entry.device) {
          Telemetry::add(metrics.acquire_vram_hits_);
          Telemetry::add(metrics.vram_hits_by_priority_[priority_index(
              options.priority)]);
          record_waiter_completion(*waiter);
          add_reference_locked(entry);
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
            Telemetry::add(metrics.ram_hits_by_priority_[priority_index(
                options.priority)]);
            if (options.priority == ExpertRequestPriority::demand &&
                entry.host_preloaded) {
              Telemetry::add(metrics.preloaded_host_useful_);
              Telemetry::add(metrics.preloaded_host_useful_bytes_,
                             entry.ram_reserved);
              entry.host_preloaded = false;
            }
          } else if (entry.state == CacheState::absent ||
                     entry.state == CacheState::ssd_loading) {
            Telemetry::add(metrics.acquire_ssd_misses_);
            Telemetry::add(metrics.ssd_misses_by_priority_[priority_index(
                options.priority)]);
          }
          if (!entry.waiters.empty() || entry.state != CacheState::absent) {
            Telemetry::add(metrics.load_deduplicated_);
          }
          entry.waiters.emplace(waiter->id, waiter);
          if (higher_priority(options.priority, previous_priority))
            Telemetry::add(metrics.priority_upgrades_);
          entry.abandon = false;
          refresh_task_indexes_locked(entry);
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

  HostPreloadHandle preload_host(const ExpertKey& key,
                                 const PayloadRecord& record,
                                 HostPreloadOptions options) {
    auto waiter =
        std::make_shared<HostWaiter>(next_waiter.fetch_add(1), options);
    auto future = waiter->promise.get_future();
    bool should_drive = false;
    {
      auto lock = acquire_lock();
      Telemetry::add(metrics.host_preloads_requested_);
      Telemetry::add(
          metrics.host_requests_[priority_index(options.priority)]);
      if (shutting_down) {
        record_waiter_completion(*waiter);
        Telemetry::add(metrics.cancellations_by_priority_[priority_index(
            options.priority)]);
        Telemetry::add(metrics.cancellation_count_);
        waiter->promise.set_value(
            {Status(ErrorCode::cancelled, "expert cache is shutting down"),
             false});
      } else {
        auto [iterator, inserted] = entries.try_emplace(key);
        if (inserted) {
          iterator->second = std::make_shared<Entry>();
          iterator->second->key = key;
          iterator->second->record = record;
        }
        auto& entry = *iterator->second;
        const auto previous_priority = has_waiters_locked(entry)
                                           ? effective_priority_locked(entry)
                                           : options.priority;
        if (!same_record(entry.record, record)) {
          record_waiter_completion(*waiter);
          record_waiter_failure(*waiter);
          waiter->promise.set_value(
              {Status(ErrorCode::invalid_argument,
                      "same expert key references different immutable records"),
               false});
        } else if (entry.state == CacheState::failed) {
          record_waiter_completion(*waiter);
          record_waiter_failure(*waiter);
          waiter->promise.set_value(
              {Status(ErrorCode::checksum_mismatch,
                      "expert is quarantined after a failed load"),
               false});
        } else if (entry.host_copy && entry.validated_sections &&
                   (entry.state == CacheState::ram_ready ||
                    entry.state == CacheState::vram_ready)) {
          Telemetry::add(metrics.ram_hits_by_priority_[priority_index(
              options.priority)]);
          if (options.protect_ram) {
            entry.ram_protected_requested = true;
            entry.ram_protection_persistent = true;
            promote_ram_locked(entry, true);
          }
          HostExpertLease lease;
          if (options.acquire_lease) {
            add_reference_locked(entry);
            auto weak = weak_from_this();
            lease = HostExpertLease(
                entry.host_copy, *entry.validated_sections,
                entry.validated_compact, entry.record.source_abi,
                [weak, key = entry.key]() noexcept {
                  if (auto core = weak.lock()) core->release_reference(key);
                });
          }
          waiter->promise.set_value(
              {Status::success(), true, std::move(lease)});
          record_waiter_completion(*waiter);
          Telemetry::add(metrics.host_preloads_completed_);
        } else if (entry.state == CacheState::ram_ready ||
                   entry.state == CacheState::vram_ready ||
                   entry.state == CacheState::gpu_uploading) {
          Telemetry::add(metrics.host_lookup_misses_by_priority_[
              priority_index(options.priority)]);
          record_waiter_completion(*waiter);
          record_waiter_failure(*waiter);
          waiter->promise.set_value(
              {Status(ErrorCode::backpressure,
                      "resident expert has no pageable host copy"),
               false});
        } else {
          Telemetry::add(metrics.ssd_misses_by_priority_[priority_index(
              options.priority)]);
          entry.force_host_retention = true;
          entry.ram_protected_requested =
              entry.ram_protected_requested || options.protect_ram;
          entry.ram_protection_persistent =
              entry.ram_protection_persistent || options.protect_ram;
          entry.host_waiters.emplace(waiter->id, waiter);
          if (higher_priority(options.priority, previous_priority))
            Telemetry::add(metrics.priority_upgrades_);
          entry.abandon = false;
          refresh_task_indexes_locked(entry);
          should_drive = true;
        }
      }
    }
    auto weak = weak_from_this();
    HostPreloadHandle handle(
        std::move(future), [weak, key, id = waiter->id]() noexcept {
          if (auto core = weak.lock()) core->cancel_host_waiter(key, id);
        });
    if (should_drive) drive();
    return handle;
  }

  std::optional<HostExpertLease> try_acquire_host(
      const ExpertKey& key, const PayloadRecord& record,
      bool record_access, ExpertRequestPriority priority) {
      auto lock = acquire_lock();
    const auto priority_slot = priority_index(priority);
    Telemetry::add(metrics.host_requests_[priority_slot]);
    if (shutting_down) return std::nullopt;
    const auto iterator = entries.find(key);
    if (iterator == entries.end()) {
      Telemetry::add(metrics.host_lookup_misses_by_priority_[priority_slot]);
      Telemetry::add(metrics.ssd_misses_by_priority_[priority_slot]);
      return std::nullopt;
    }
    auto& entry = *iterator->second;
    if ((record.source_abi != kExpertSourceAbiExpertPackV1 &&
         record.source_abi != kExpertSourceAbiDeepSeekCompactV1) ||
        !same_record(entry.record, record) || !entry.host_copy ||
        !entry.validated_sections ||
        (entry.state != CacheState::ram_ready &&
         entry.state != CacheState::vram_ready)) {
      Telemetry::add(metrics.host_lookup_misses_by_priority_[priority_slot]);
      if (entry.state == CacheState::absent ||
          entry.state == CacheState::ssd_loading) {
        Telemetry::add(metrics.ssd_misses_by_priority_[priority_slot]);
      }
      return std::nullopt;
    }
    if (record_access) {
      touch_locked(entry);
      promote_ram_locked(entry);
      promote_vram_locked(entry);
    }
    add_reference_locked(entry);
    Telemetry::add(metrics.acquire_ram_hits_);
    Telemetry::add(metrics.ram_hits_by_priority_[priority_slot]);
    auto weak = weak_from_this();
    return HostExpertLease(
        entry.host_copy, *entry.validated_sections, entry.validated_compact,
        record.source_abi,
        [weak, key]() noexcept {
          if (auto core = weak.lock()) core->release_reference(key);
        });
  }

  bool record_access(const ExpertKey& key, std::uint32_t count) {
    if (count == 0) return false;
    auto lock = acquire_lock();
    if (shutting_down) return false;
    const auto iterator = entries.find(key);
    if (iterator == entries.end()) return false;
    for (std::uint32_t access = 0; access < count; ++access) {
      touch_locked(*iterator->second);
    }
    promote_ram_locked(*iterator->second);
    promote_vram_locked(*iterator->second);
    return true;
  }

  std::uint64_t record_accesses(std::span<const ExpertAccess> accesses) {
    auto lock = acquire_lock();
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
      promote_ram_locked(*iterator->second);
      promote_vram_locked(*iterator->second);
      recorded += access.count;
    }
    return recorded;
  }

  bool protect(const ExpertKey& key, bool ram, bool vram) {
    auto lock = acquire_lock();
    if (shutting_down) return false;
    const auto iterator = entries.find(key);
    if (iterator == entries.end()) return false;
    auto& entry = *iterator->second;
    if (ram) {
      entry.ram_protected_requested = true;
      promote_ram_locked(entry, true);
    }
    if (vram) {
      entry.vram_resident_requested = true;
      promote_vram_locked(entry, true);
    }
    return (!ram || !entry.host_copy || entry.ram_protected ||
            config.placement.ram_protected_bytes == 0) &&
           (!vram || !entry.device || entry.vram_resident ||
            config.placement.vram_transient_bytes == 0);
  }

  bool vram_admission_would_improve(const ExpertKey& key,
                                    const PayloadRecord& record) {
    auto lock = acquire_lock();
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
    const bool resident = config.placement.vram_transient_bytes == 0 ||
                          entry.vram_resident_requested ||
                          entry.frequency >= 2;
    const auto limit = resident
                           ? config.vram.high_watermark_bytes -
                                 config.placement.vram_transient_bytes
                           : config.placement.vram_transient_bytes;
    const auto usage = resident ? vram_resident_bytes : vram_transient_bytes;
    if (usage + need <= limit) return true;
    const auto deficit = usage + need - limit;
    const auto entry_temperature = temperature_locked(entry);
    std::uint64_t colder_bytes = 0;
    std::uint64_t scanned_candidates = 0;
    const auto scan_started = std::chrono::steady_clock::now();
    const auto record_scan = [&] {
      Telemetry::add(metrics.vram_admission_scan_calls_);
      Telemetry::add(metrics.vram_admission_scan_candidates_,
                     scanned_candidates);
      Telemetry::add(
          metrics.vram_admission_scan_ns_,
          static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - scan_started)
                  .count()));
    };
    for (const auto& [candidate_key, stored_candidate] : entries) {
      const auto* candidate = stored_candidate.get();
      ++scanned_candidates;
      if (candidate_key == key || !candidate->device ||
          candidate->vram_resident != resident ||
          candidate->references != 0 || has_waiters_locked(*candidate) ||
          (directory && directory->route_pinned(candidate_key)) ||
          temperature_locked(*candidate) >= entry_temperature ||
          candidate->state == CacheState::gpu_uploading ||
          candidate->state == CacheState::failed) {
        continue;
      }
      colder_bytes += candidate->vram_reserved;
      if (colder_bytes >= deficit) {
        record_scan();
        return true;
      }
    }
    record_scan();
    return false;
  }

  std::uint64_t vram_stale_resident_bytes(std::uint64_t minimum_age) const {
    auto lock = acquire_lock();
    if (shutting_down) return 0;
    std::uint64_t stale_bytes = 0;
    for (const auto& [key, entry] : entries) {
      (void)key;
      if (!entry->device || entry->references != 0 ||
          !entry->waiters.empty() ||
          entry->state == CacheState::gpu_uploading ||
          entry->state == CacheState::failed ||
          access_clock - entry->last_access < minimum_age) {
        continue;
      }
      stale_bytes =
          entry->vram_reserved >
                  std::numeric_limits<std::uint64_t>::max() - stale_bytes
              ? std::numeric_limits<std::uint64_t>::max()
              : stale_bytes + entry->vram_reserved;
    }
    return stale_bytes;
  }

  void cancel_waiter(const ExpertKey& key, std::uint64_t waiter_id) noexcept {
    OperationId io = 0;
    OperationId upload = 0;
    {
      auto lock = acquire_lock();
      const auto iterator = entries.find(key);
      if (iterator == entries.end()) {
        return;
      }
      auto& entry = *iterator->second;
      const auto waiter = entry.waiters.find(waiter_id);
      if (waiter == entry.waiters.end()) {
        return;
      }
      record_waiter_completion(*waiter->second);
      waiter->second->promise.set_value(
          {Status(ErrorCode::cancelled, "expert acquisition cancelled"), {}});
      Telemetry::add(metrics.cancellations_by_priority_[priority_index(
          waiter->second->options.priority)]);
      entry.waiters.erase(waiter);
      Telemetry::add(metrics.cancellation_count_);
      if (!has_waiters_locked(entry)) {
        entry.abandon = true;
        io = entry.io_operation;
        upload = entry.upload_operation;
        if (entry.state == CacheState::ram_ready && !entry.preparing_host) {
          release_staging_locked(entry);
          if (!entry.host_copy) {
            release_host_locked(entry);
            transition_locked(entry, CacheState::absent);
          }
          entry.abandon = false;
          update_usage_locked();
        }
      }
      refresh_task_indexes_locked(entry);
    }
    if (io != 0) {
      storage->cancel(io);
    }
    if (upload != 0) {
      uploader->cancel(upload);
    }
    drive();
  }

  void cancel_host_waiter(const ExpertKey& key,
                          std::uint64_t waiter_id) noexcept {
    OperationId io = 0;
    OperationId upload = 0;
    {
      auto lock = acquire_lock();
      const auto iterator = entries.find(key);
      if (iterator == entries.end()) return;
      auto& entry = *iterator->second;
      const auto waiter = entry.host_waiters.find(waiter_id);
      if (waiter == entry.host_waiters.end()) return;
      record_waiter_completion(*waiter->second);
      waiter->second->promise.set_value(
          {Status(ErrorCode::cancelled, "host preload cancelled"), false});
      Telemetry::add(metrics.cancellations_by_priority_[priority_index(
          waiter->second->options.priority)]);
      entry.host_waiters.erase(waiter);
      Telemetry::add(metrics.cancellation_count_);
      if (!has_waiters_locked(entry)) {
        entry.abandon = true;
        io = entry.io_operation;
        upload = entry.upload_operation;
        if (entry.state == CacheState::ram_ready && !entry.preparing_host) {
          release_staging_locked(entry);
          if (!entry.host_copy) {
            release_host_locked(entry);
            transition_locked(entry, CacheState::absent);
          }
          entry.abandon = false;
          update_usage_locked();
        }
      }
      refresh_task_indexes_locked(entry);
    }
    if (io != 0) storage->cancel(io);
    if (upload != 0) uploader->cancel(upload);
    drive();
  }

  void release_reference(const ExpertKey& key) noexcept {
    {
      auto lock = acquire_lock();
      const auto iterator = entries.find(key);
      if (iterator == entries.end() || iterator->second->references == 0) {
        return;
      }
      remove_reference_locked(*iterator->second);
      iterator->second->last_access = ++access_clock;
    }
    drive();
  }

  CacheUsage trim_to(std::uint64_t ram_target,
                     std::uint64_t vram_target) {
    auto lock = acquire_lock();
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
      auto lock = acquire_lock();
      if (shutting_down) {
        return;
      }
      shutting_down = true;
      host_worker_stop = true;
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
    host_condition.notify_all();
    if (host_worker.joinable()) host_worker.join();
  }

  ExpertCacheConfig config;
  std::shared_ptr<IAsyncStorage> storage;
  std::shared_ptr<IDeviceUploader> uploader;
  std::shared_ptr<FixedBufferPool> buffers;
  std::shared_ptr<IDeviceResidencyDirectory> directory;
  mutable std::mutex mutex;
  std::map<ExpertKey, std::shared_ptr<Entry>> entries;
  std::array<std::set<ExpertKey>, kExpertPriorityCount> read_task_keys;
  std::array<std::set<ExpertKey>, kExpertPriorityCount> upload_task_keys;
  mutable Telemetry metrics;
  std::atomic<std::uint64_t> next_waiter{1};
  std::uint64_t access_clock{};
  std::uint64_t ram_bytes{};
  std::uint64_t ram_probationary_bytes{};
  std::uint64_t ram_protected_bytes{};
  std::uint64_t vram_bytes{};
  std::uint64_t vram_resident_bytes{};
  std::uint64_t vram_transient_bytes{};
  std::uint64_t vram_referenced_entries{};
  std::uint64_t vram_referenced_bytes{};
  std::uint64_t vram_transient_referenced_bytes{};
  std::vector<std::uint64_t> ram_partition_bytes;
  std::vector<std::uint64_t> vram_partition_bytes;
  std::condition_variable host_condition;
  std::deque<std::shared_ptr<Entry>> host_prepare_queue;
  std::thread host_worker;
  bool host_worker_stop{};
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
    ExpertSections sections, DeepSeekCompactSections compact,
    std::uint32_t source_abi, std::function<void()> release) noexcept
    : bytes_(std::move(bytes)), sections_(sections), compact_(compact),
      source_abi_(source_abi),
      release_(std::move(release)) {}

HostExpertLease::HostExpertLease(HostExpertLease&& other) noexcept
    : bytes_(std::move(other.bytes_)), sections_(other.sections_),
      compact_(other.compact_), source_abi_(other.source_abi_),
      release_(std::move(other.release_)) {}

HostExpertLease& HostExpertLease::operator=(HostExpertLease&& other) noexcept {
  if (this != &other) {
    reset();
    bytes_ = std::move(other.bytes_);
    sections_ = other.sections_;
    compact_ = other.compact_;
    source_abi_ = other.source_abi_;
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

const DeepSeekCompactSections& HostExpertLease::compact_sections()
    const noexcept {
  return compact_;
}

std::uint32_t HostExpertLease::source_abi() const noexcept {
  return source_abi_;
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

HostPreloadHandle::HostPreloadHandle(
    std::future<HostPreloadResult> future,
    std::function<void()> cancel) noexcept
    : future_(std::move(future)), cancel_(std::move(cancel)) {}

HostPreloadHandle::~HostPreloadHandle() { cancel(); }

void HostPreloadHandle::cancel() noexcept {
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
  return core_->acquire(key, record, {});
}

AcquireHandle ExpertCache::acquire(const ExpertKey& key,
                                   const PayloadRecord& record,
                                   ExpertAcquireOptions options) {
  return core_->acquire(key, record, options);
}

HostPreloadHandle ExpertCache::preload_host(
    const ExpertKey& key, const PayloadRecord& record,
    HostPreloadOptions options) {
  return core_->preload_host(key, record, options);
}

std::optional<HostExpertLease> ExpertCache::try_acquire_host(
    const ExpertKey& key, const PayloadRecord& record, bool record_access,
    ExpertRequestPriority priority) {
  return core_->try_acquire_host(key, record, record_access, priority);
}

bool ExpertCache::record_access(const ExpertKey& key, std::uint32_t count) {
  return core_->record_access(key, count);
}

std::uint64_t ExpertCache::record_accesses(
    std::span<const ExpertAccess> accesses) {
  return core_->record_accesses(accesses);
}

bool ExpertCache::protect(const ExpertKey& key, bool ram, bool vram) {
  return core_->protect(key, ram, vram);
}

bool ExpertCache::vram_admission_would_improve(
    const ExpertKey& key, const PayloadRecord& record) {
  return core_->vram_admission_would_improve(key, record);
}

std::uint64_t ExpertCache::vram_stale_resident_bytes(
    std::uint64_t minimum_age) const {
  return core_->vram_stale_resident_bytes(minimum_age);
}

std::optional<CacheEntrySnapshot> ExpertCache::inspect(
    const ExpertKey& key) const {
  auto lock = core_->acquire_lock();
  const auto iterator = core_->entries.find(key);
  if (iterator == core_->entries.end()) {
    return std::nullopt;
  }
  const auto& entry = *iterator->second;
  return CacheEntrySnapshot{entry.state, entry.references,
                            entry.waiters.size() + entry.host_waiters.size(),
                            entry.host != nullptr || entry.host_copy != nullptr,
                            entry.device != nullptr, entry.ram_protected,
                            entry.vram_resident, entry.frequency,
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
