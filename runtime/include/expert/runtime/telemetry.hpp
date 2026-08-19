#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace expert::runtime {

inline constexpr std::size_t kExpertPriorityCount = 3U;

struct ExpertCacheCore;

struct TelemetrySnapshot final {
  // Priority order is warm, prefetch, demand.
  std::array<std::uint64_t, kExpertPriorityCount> device_requests{};
  std::array<std::uint64_t, kExpertPriorityCount> host_requests{};
  std::array<std::uint64_t, kExpertPriorityCount> vram_hits_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> ram_hits_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> ssd_misses_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount>
      host_lookup_misses_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> reads_started_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> reads_completed_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> read_bytes_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> storage_wait_ns_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount>
      host_validation_ns_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount>
      host_copy_bytes_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount>
      ram_retention_copy_ns_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> uploads_started_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> uploads_completed_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> uploaded_bytes_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> upload_wait_ns_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> completed_waiters_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> waiter_wait_ns_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> cancellations_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount>
      failed_waiters_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> io_errors_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount>
      validation_errors_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> upload_errors_by_priority{};
  std::array<std::uint64_t, kExpertPriorityCount> staging_stalls_by_priority{};
  std::uint64_t acquire_vram_hits{};
  std::uint64_t acquire_ram_hits{};
  std::uint64_t acquire_ssd_misses{};
  std::uint64_t load_started{};
  std::uint64_t load_deduplicated{};
  std::uint64_t load_completed{};
  std::uint64_t upload_started{};
  std::uint64_t upload_completed{};
  std::uint64_t record_validations{};
  std::uint64_t validated_ram_reuses{};
  std::uint64_t requested_bytes{};
  std::uint64_t useful_bytes{};
  std::uint64_t read_bytes{};
  std::uint64_t reload_count{};
  std::uint64_t reread_bytes{};
  std::uint64_t host_preloads_requested{};
  std::uint64_t host_preloads_completed{};
  std::uint64_t priority_upgrades{};
  std::uint64_t host_validation_ns{};
  std::uint64_t host_validation_failures{};
  std::uint64_t host_copy_bytes{};
  std::uint64_t preloaded_host_useful{};
  std::uint64_t preloaded_host_useful_bytes{};
  std::uint64_t preloaded_host_wasted{};
  std::uint64_t preloaded_host_wasted_bytes{};
  std::uint64_t uploaded_bytes{};
  std::uint64_t storage_wait_ns{};
  std::uint64_t ram_retention_copy_ns{};
  std::uint64_t upload_wait_ns{};
  std::uint64_t ram_bytes{};
  std::uint64_t ram_high_water{};
  std::uint64_t ram_probationary_bytes{};
  std::uint64_t ram_protected_bytes{};
  std::uint64_t ram_probationary_high_water{};
  std::uint64_t ram_protected_high_water{};
  std::uint64_t vram_bytes{};
  std::uint64_t vram_high_water{};
  std::uint64_t vram_resident_bytes{};
  std::uint64_t vram_transient_bytes{};
  std::uint64_t vram_resident_high_water{};
  std::uint64_t vram_transient_high_water{};
  std::uint64_t vram_referenced_entries{};
  std::uint64_t vram_referenced_bytes{};
  std::uint64_t vram_transient_referenced_bytes{};
  std::uint64_t ram_promotions{};
  std::uint64_t vram_promotions{};
  std::uint64_t ram_promotion_failures{};
  std::uint64_t vram_promotion_failures{};
  std::array<std::uint64_t, 2> ram_evictions_by_class{};
  std::array<std::uint64_t, 2> ram_evicted_bytes_by_class{};
  std::array<std::uint64_t, 2> vram_evictions_by_class{};
  std::array<std::uint64_t, 2> vram_evicted_bytes_by_class{};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_high_water{};
  std::uint64_t eviction_count{};
  std::uint64_t eviction_scan_calls{};
  std::uint64_t eviction_scan_candidates{};
  std::uint64_t eviction_scan_ns{};
  std::uint64_t eviction_retire_retries{};
  std::uint64_t vram_admission_scan_calls{};
  std::uint64_t vram_admission_scan_candidates{};
  std::uint64_t vram_admission_scan_ns{};
  std::uint64_t task_selection_calls{};
  std::uint64_t task_selection_candidates{};
  std::uint64_t task_selection_ns{};
  std::uint64_t mutex_acquisitions{};
  std::uint64_t mutex_wait_ns{};
  std::uint64_t mutex_wait_max_ns{};
  std::uint64_t same_partition_evictions{};
  std::uint64_t over_quota_evictions{};
  std::uint64_t stalled_by_budget{};
  std::uint64_t device_admission_rejections{};
  std::uint64_t cancellation_count{};
  std::uint64_t short_read_errors{};
  std::uint64_t checksum_errors{};
  std::uint64_t io_errors{};
  std::uint64_t upload_errors{};
  std::array<std::uint64_t, 36> state_transitions{};
};

class Telemetry final {
 public:
  [[nodiscard]] TelemetrySnapshot snapshot() const noexcept;

 private:
  friend struct ExpertCacheCore;

  static void add(std::atomic<std::uint64_t>& target,
                  std::uint64_t value = 1) noexcept;
  static void set(std::atomic<std::uint64_t>& target,
                  std::uint64_t value) noexcept;
  static void maximize(std::atomic<std::uint64_t>& target,
                       std::uint64_t value) noexcept;

  std::atomic<std::uint64_t> load_started_{0};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      device_requests_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount> host_requests_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      vram_hits_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      ram_hits_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      ssd_misses_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      host_lookup_misses_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      reads_started_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      reads_completed_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      read_bytes_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      storage_wait_ns_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      host_validation_ns_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      host_copy_bytes_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      ram_retention_copy_ns_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      uploads_started_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      uploads_completed_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      uploaded_bytes_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      upload_wait_ns_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      completed_waiters_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      waiter_wait_ns_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      cancellations_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      failed_waiters_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      io_errors_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      validation_errors_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      upload_errors_by_priority_{};
  std::array<std::atomic<std::uint64_t>, kExpertPriorityCount>
      staging_stalls_by_priority_{};
  std::atomic<std::uint64_t> acquire_vram_hits_{0};
  std::atomic<std::uint64_t> acquire_ram_hits_{0};
  std::atomic<std::uint64_t> acquire_ssd_misses_{0};
  std::atomic<std::uint64_t> load_deduplicated_{0};
  std::atomic<std::uint64_t> load_completed_{0};
  std::atomic<std::uint64_t> upload_started_{0};
  std::atomic<std::uint64_t> upload_completed_{0};
  std::atomic<std::uint64_t> record_validations_{0};
  std::atomic<std::uint64_t> validated_ram_reuses_{0};
  std::atomic<std::uint64_t> requested_bytes_{0};
  std::atomic<std::uint64_t> useful_bytes_{0};
  std::atomic<std::uint64_t> read_bytes_{0};
  std::atomic<std::uint64_t> reload_count_{0};
  std::atomic<std::uint64_t> reread_bytes_{0};
  std::atomic<std::uint64_t> host_preloads_requested_{0};
  std::atomic<std::uint64_t> host_preloads_completed_{0};
  std::atomic<std::uint64_t> priority_upgrades_{0};
  std::atomic<std::uint64_t> host_validation_ns_{0};
  std::atomic<std::uint64_t> host_validation_failures_{0};
  std::atomic<std::uint64_t> host_copy_bytes_{0};
  std::atomic<std::uint64_t> preloaded_host_useful_{0};
  std::atomic<std::uint64_t> preloaded_host_useful_bytes_{0};
  std::atomic<std::uint64_t> preloaded_host_wasted_{0};
  std::atomic<std::uint64_t> preloaded_host_wasted_bytes_{0};
  std::atomic<std::uint64_t> uploaded_bytes_{0};
  std::atomic<std::uint64_t> storage_wait_ns_{0};
  std::atomic<std::uint64_t> ram_retention_copy_ns_{0};
  std::atomic<std::uint64_t> upload_wait_ns_{0};
  std::atomic<std::uint64_t> ram_bytes_{0};
  std::atomic<std::uint64_t> ram_high_water_{0};
  std::atomic<std::uint64_t> ram_probationary_bytes_{0};
  std::atomic<std::uint64_t> ram_protected_bytes_{0};
  std::atomic<std::uint64_t> ram_probationary_high_water_{0};
  std::atomic<std::uint64_t> ram_protected_high_water_{0};
  std::atomic<std::uint64_t> vram_bytes_{0};
  std::atomic<std::uint64_t> vram_high_water_{0};
  std::atomic<std::uint64_t> vram_resident_bytes_{0};
  std::atomic<std::uint64_t> vram_transient_bytes_{0};
  std::atomic<std::uint64_t> vram_resident_high_water_{0};
  std::atomic<std::uint64_t> vram_transient_high_water_{0};
  std::atomic<std::uint64_t> vram_referenced_entries_{0};
  std::atomic<std::uint64_t> vram_referenced_bytes_{0};
  std::atomic<std::uint64_t> vram_transient_referenced_bytes_{0};
  std::atomic<std::uint64_t> ram_promotions_{0};
  std::atomic<std::uint64_t> vram_promotions_{0};
  std::atomic<std::uint64_t> ram_promotion_failures_{0};
  std::atomic<std::uint64_t> vram_promotion_failures_{0};
  std::array<std::atomic<std::uint64_t>, 2> ram_evictions_by_class_{};
  std::array<std::atomic<std::uint64_t>, 2> ram_evicted_bytes_by_class_{};
  std::array<std::atomic<std::uint64_t>, 2> vram_evictions_by_class_{};
  std::array<std::atomic<std::uint64_t>, 2> vram_evicted_bytes_by_class_{};
  std::atomic<std::uint64_t> staging_bytes_{0};
  std::atomic<std::uint64_t> staging_high_water_{0};
  std::atomic<std::uint64_t> eviction_count_{0};
  std::atomic<std::uint64_t> eviction_scan_calls_{0};
  std::atomic<std::uint64_t> eviction_scan_candidates_{0};
  std::atomic<std::uint64_t> eviction_scan_ns_{0};
  std::atomic<std::uint64_t> eviction_retire_retries_{0};
  std::atomic<std::uint64_t> vram_admission_scan_calls_{0};
  std::atomic<std::uint64_t> vram_admission_scan_candidates_{0};
  std::atomic<std::uint64_t> vram_admission_scan_ns_{0};
  std::atomic<std::uint64_t> task_selection_calls_{0};
  std::atomic<std::uint64_t> task_selection_candidates_{0};
  std::atomic<std::uint64_t> task_selection_ns_{0};
  std::atomic<std::uint64_t> mutex_acquisitions_{0};
  std::atomic<std::uint64_t> mutex_wait_ns_{0};
  std::atomic<std::uint64_t> mutex_wait_max_ns_{0};
  std::atomic<std::uint64_t> same_partition_evictions_{0};
  std::atomic<std::uint64_t> over_quota_evictions_{0};
  std::atomic<std::uint64_t> stalled_by_budget_{0};
  std::atomic<std::uint64_t> device_admission_rejections_{0};
  std::atomic<std::uint64_t> cancellation_count_{0};
  std::atomic<std::uint64_t> short_read_errors_{0};
  std::atomic<std::uint64_t> checksum_errors_{0};
  std::atomic<std::uint64_t> io_errors_{0};
  std::atomic<std::uint64_t> upload_errors_{0};
  std::array<std::atomic<std::uint64_t>, 36> state_transitions_{};
};

}  // namespace expert::runtime
