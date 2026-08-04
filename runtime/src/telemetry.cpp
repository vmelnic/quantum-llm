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
  EXPERT_SNAPSHOT(requested_bytes);
  EXPERT_SNAPSHOT(useful_bytes);
  EXPERT_SNAPSHOT(read_bytes);
  EXPERT_SNAPSHOT(uploaded_bytes);
  EXPERT_SNAPSHOT(ram_bytes);
  EXPERT_SNAPSHOT(ram_high_water);
  EXPERT_SNAPSHOT(vram_bytes);
  EXPERT_SNAPSHOT(vram_high_water);
  EXPERT_SNAPSHOT(staging_bytes);
  EXPERT_SNAPSHOT(staging_high_water);
  EXPERT_SNAPSHOT(eviction_count);
  EXPERT_SNAPSHOT(stalled_by_budget);
  EXPERT_SNAPSHOT(cancellation_count);
  EXPERT_SNAPSHOT(short_read_errors);
  EXPERT_SNAPSHOT(checksum_errors);
  EXPERT_SNAPSHOT(io_errors);
  EXPERT_SNAPSHOT(upload_errors);
#undef EXPERT_SNAPSHOT
  for (std::size_t index = 0; index < result.state_transitions.size(); ++index) {
    result.state_transitions[index] =
        state_transitions_[index].load(std::memory_order_relaxed);
  }
  return result;
}

}  // namespace expert::runtime
