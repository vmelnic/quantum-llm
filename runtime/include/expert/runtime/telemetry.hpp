#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace expert::runtime {

struct ExpertCacheCore;

struct TelemetrySnapshot final {
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
  std::uint64_t uploaded_bytes{};
  std::uint64_t ram_bytes{};
  std::uint64_t ram_high_water{};
  std::uint64_t vram_bytes{};
  std::uint64_t vram_high_water{};
  std::uint64_t vram_resident_bytes{};
  std::uint64_t vram_transient_bytes{};
  std::uint64_t vram_resident_high_water{};
  std::uint64_t vram_transient_high_water{};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_high_water{};
  std::uint64_t eviction_count{};
  std::uint64_t same_partition_evictions{};
  std::uint64_t over_quota_evictions{};
  std::uint64_t stalled_by_budget{};
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
  std::atomic<std::uint64_t> uploaded_bytes_{0};
  std::atomic<std::uint64_t> ram_bytes_{0};
  std::atomic<std::uint64_t> ram_high_water_{0};
  std::atomic<std::uint64_t> vram_bytes_{0};
  std::atomic<std::uint64_t> vram_high_water_{0};
  std::atomic<std::uint64_t> vram_resident_bytes_{0};
  std::atomic<std::uint64_t> vram_transient_bytes_{0};
  std::atomic<std::uint64_t> vram_resident_high_water_{0};
  std::atomic<std::uint64_t> vram_transient_high_water_{0};
  std::atomic<std::uint64_t> staging_bytes_{0};
  std::atomic<std::uint64_t> staging_high_water_{0};
  std::atomic<std::uint64_t> eviction_count_{0};
  std::atomic<std::uint64_t> same_partition_evictions_{0};
  std::atomic<std::uint64_t> over_quota_evictions_{0};
  std::atomic<std::uint64_t> stalled_by_budget_{0};
  std::atomic<std::uint64_t> cancellation_count_{0};
  std::atomic<std::uint64_t> short_read_errors_{0};
  std::atomic<std::uint64_t> checksum_errors_{0};
  std::atomic<std::uint64_t> io_errors_{0};
  std::atomic<std::uint64_t> upload_errors_{0};
  std::array<std::atomic<std::uint64_t>, 36> state_transitions_{};
};

}  // namespace expert::runtime
