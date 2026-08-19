#include "expert/runtime/telemetry.hpp"

#include <algorithm>

namespace expert::runtime {

void Telemetry::add(std::atomic<std::uint64_t>& target,
                    std::uint64_t value) noexcept {
  target.fetch_add(value, std::memory_order_relaxed);
}

void Telemetry::set(std::atomic<std::uint64_t>& target,
                    std::uint64_t value) noexcept {
  target.store(value, std::memory_order_relaxed);
}

void Telemetry::maximize(std::atomic<std::uint64_t>& target,
                         std::uint64_t value) noexcept {
  auto previous = target.load(std::memory_order_relaxed);
  while (previous < value &&
         !target.compare_exchange_weak(previous, value,
                                       std::memory_order_relaxed)) {
  }
}

TelemetrySnapshot Telemetry::snapshot() const noexcept {
  TelemetrySnapshot result;
  for (std::size_t priority = 0; priority < kExpertPriorityCount; ++priority) {
#define EXPERT_PRIORITY_SNAPSHOT(field) \
    result.field[priority] = field##_[priority].load(std::memory_order_relaxed)
    EXPERT_PRIORITY_SNAPSHOT(device_requests);
    EXPERT_PRIORITY_SNAPSHOT(host_requests);
    EXPERT_PRIORITY_SNAPSHOT(vram_hits_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(ram_hits_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(ssd_misses_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(host_lookup_misses_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(reads_started_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(reads_completed_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(read_bytes_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(storage_wait_ns_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(host_validation_ns_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(host_copy_bytes_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(ram_retention_copy_ns_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(uploads_started_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(uploads_completed_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(uploaded_bytes_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(upload_wait_ns_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(completed_waiters_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(waiter_wait_ns_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(cancellations_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(failed_waiters_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(io_errors_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(validation_errors_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(upload_errors_by_priority);
    EXPERT_PRIORITY_SNAPSHOT(staging_stalls_by_priority);
#undef EXPERT_PRIORITY_SNAPSHOT
  }
  result.acquire_vram_hits = acquire_vram_hits_.load(std::memory_order_relaxed);
  result.acquire_ram_hits = acquire_ram_hits_.load(std::memory_order_relaxed);
  result.acquire_ssd_misses = acquire_ssd_misses_.load(std::memory_order_relaxed);
#define EXPERT_SNAPSHOT(field) \
  result.field = field##_.load(std::memory_order_relaxed)
  EXPERT_SNAPSHOT(load_started);
  EXPERT_SNAPSHOT(load_deduplicated);
  EXPERT_SNAPSHOT(load_completed);
  EXPERT_SNAPSHOT(upload_started);
  EXPERT_SNAPSHOT(upload_completed);
  EXPERT_SNAPSHOT(record_validations);
  EXPERT_SNAPSHOT(validated_ram_reuses);
  EXPERT_SNAPSHOT(requested_bytes);
  EXPERT_SNAPSHOT(useful_bytes);
  EXPERT_SNAPSHOT(read_bytes);
  EXPERT_SNAPSHOT(reload_count);
  EXPERT_SNAPSHOT(reread_bytes);
  EXPERT_SNAPSHOT(host_preloads_requested);
  EXPERT_SNAPSHOT(host_preloads_completed);
  EXPERT_SNAPSHOT(priority_upgrades);
  EXPERT_SNAPSHOT(host_validation_ns);
  EXPERT_SNAPSHOT(host_validation_failures);
  EXPERT_SNAPSHOT(host_copy_bytes);
  EXPERT_SNAPSHOT(preloaded_host_useful);
  EXPERT_SNAPSHOT(preloaded_host_useful_bytes);
  EXPERT_SNAPSHOT(preloaded_host_wasted);
  EXPERT_SNAPSHOT(preloaded_host_wasted_bytes);
  EXPERT_SNAPSHOT(uploaded_bytes);
  EXPERT_SNAPSHOT(storage_wait_ns);
  EXPERT_SNAPSHOT(ram_retention_copy_ns);
  EXPERT_SNAPSHOT(upload_wait_ns);
  EXPERT_SNAPSHOT(ram_bytes);
  EXPERT_SNAPSHOT(ram_high_water);
  EXPERT_SNAPSHOT(ram_probationary_bytes);
  EXPERT_SNAPSHOT(ram_protected_bytes);
  EXPERT_SNAPSHOT(ram_probationary_high_water);
  EXPERT_SNAPSHOT(ram_protected_high_water);
  EXPERT_SNAPSHOT(vram_bytes);
  EXPERT_SNAPSHOT(vram_high_water);
  EXPERT_SNAPSHOT(vram_resident_bytes);
  EXPERT_SNAPSHOT(vram_transient_bytes);
  EXPERT_SNAPSHOT(vram_resident_high_water);
  EXPERT_SNAPSHOT(vram_transient_high_water);
  EXPERT_SNAPSHOT(vram_referenced_entries);
  EXPERT_SNAPSHOT(vram_referenced_bytes);
  EXPERT_SNAPSHOT(vram_transient_referenced_bytes);
  EXPERT_SNAPSHOT(ram_promotions);
  EXPERT_SNAPSHOT(vram_promotions);
  EXPERT_SNAPSHOT(ram_promotion_failures);
  EXPERT_SNAPSHOT(vram_promotion_failures);
  EXPERT_SNAPSHOT(staging_bytes);
  EXPERT_SNAPSHOT(staging_high_water);
  EXPERT_SNAPSHOT(eviction_count);
  EXPERT_SNAPSHOT(eviction_scan_calls);
  EXPERT_SNAPSHOT(eviction_scan_candidates);
  EXPERT_SNAPSHOT(eviction_scan_ns);
  EXPERT_SNAPSHOT(eviction_retire_retries);
  EXPERT_SNAPSHOT(vram_admission_scan_calls);
  EXPERT_SNAPSHOT(vram_admission_scan_candidates);
  EXPERT_SNAPSHOT(vram_admission_scan_ns);
  EXPERT_SNAPSHOT(task_selection_calls);
  EXPERT_SNAPSHOT(task_selection_candidates);
  EXPERT_SNAPSHOT(task_selection_ns);
  EXPERT_SNAPSHOT(mutex_acquisitions);
  EXPERT_SNAPSHOT(mutex_wait_ns);
  EXPERT_SNAPSHOT(mutex_wait_max_ns);
  EXPERT_SNAPSHOT(same_partition_evictions);
  EXPERT_SNAPSHOT(over_quota_evictions);
  EXPERT_SNAPSHOT(stalled_by_budget);
  EXPERT_SNAPSHOT(device_admission_rejections);
  EXPERT_SNAPSHOT(cancellation_count);
  EXPERT_SNAPSHOT(short_read_errors);
  EXPERT_SNAPSHOT(checksum_errors);
  EXPERT_SNAPSHOT(io_errors);
  EXPERT_SNAPSHOT(upload_errors);
#undef EXPERT_SNAPSHOT
  for (std::size_t class_index = 0; class_index < 2U; ++class_index) {
    result.ram_evictions_by_class[class_index] =
        ram_evictions_by_class_[class_index].load(std::memory_order_relaxed);
    result.ram_evicted_bytes_by_class[class_index] =
        ram_evicted_bytes_by_class_[class_index].load(
            std::memory_order_relaxed);
    result.vram_evictions_by_class[class_index] =
        vram_evictions_by_class_[class_index].load(std::memory_order_relaxed);
    result.vram_evicted_bytes_by_class[class_index] =
        vram_evicted_bytes_by_class_[class_index].load(
            std::memory_order_relaxed);
  }
  for (std::size_t index = 0; index < result.state_transitions.size(); ++index) {
    result.state_transitions[index] =
        state_transitions_[index].load(std::memory_order_relaxed);
  }
  return result;
}

}  // namespace expert::runtime
