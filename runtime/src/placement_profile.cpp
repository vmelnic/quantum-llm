#include "expert/runtime/placement_profile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace expert::runtime {
namespace {

struct DomainPlan final {
  Status status;
  std::uint64_t cache_bytes{};
  std::uint64_t slots{};
};

[[nodiscard]] bool add_overflows(std::uint64_t left,
                                 std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left;
}

[[nodiscard]] DomainPlan solve_domain(const PlacementDomainInput& input,
                                      const char* name) noexcept {
  if (input.usable_bytes == 0U || input.expert_record_bytes == 0U ||
      input.emergency_reserve_bytes >= input.usable_bytes ||
      add_overflows(input.fixed_bytes, input.emergency_reserve_bytes) ||
      input.fixed_bytes + input.emergency_reserve_bytes > input.usable_bytes) {
    return {{ErrorCode::invalid_argument,
             std::string("invalid ") + name + " placement envelope"},
            0U, 0U};
  }
  const auto available = input.usable_bytes - input.fixed_bytes -
                         input.emergency_reserve_bytes;
  auto slots = available / input.expert_record_bytes;
  if (input.maximum_expert_slots != 0U)
    slots = std::min(slots, input.maximum_expert_slots);
  if (slots < input.minimum_expert_slots) {
    return {{ErrorCode::backpressure,
             std::string(name) +
                 " placement cannot satisfy minimum expert slots"},
            0U, slots};
  }
  return {Status::success(), slots * input.expert_record_bytes, slots};
}

}  // namespace

PlacementProfilePlan solve_placement_profile(
    const PlacementProfileInput& input) noexcept {
  PlacementProfilePlan result;
  if (!std::isfinite(input.costs.cpu_ns_per_selection) ||
      input.costs.cpu_ns_per_selection <= 0.0 ||
      !std::isfinite(input.costs.gpu_ns_per_selection) ||
      input.costs.gpu_ns_per_selection <= 0.0 ||
      !std::isfinite(input.costs.h2d_bytes_per_second) ||
      input.costs.h2d_bytes_per_second <= 0.0 ||
      input.costs.cpu_samples == 0U || input.costs.gpu_samples == 0U ||
      input.costs.h2d_sample_bytes == 0U ||
      !std::isfinite(input.observation_ewma_alpha) ||
      input.observation_ewma_alpha <= 0.0 ||
      input.observation_ewma_alpha > 1.0 ||
      input.maximum_dispatch_candidates == 0U ||
      input.maximum_dispatch_trace_decisions == 0U) {
    result.status = {ErrorCode::invalid_argument,
                     "placement profile requires measured finite costs"};
    return result;
  }
  const auto host = solve_domain(input.host, "host");
  if (!host.status.ok()) {
    result.status = host.status;
    return result;
  }
  const auto device = solve_domain(input.device, "device");
  if (!device.status.ok()) {
    result.status = device.status;
    return result;
  }
  result.dispatch =
      {input.costs.cpu_ns_per_selection,
       input.costs.gpu_ns_per_selection,
       input.costs.h2d_bytes_per_second,
       input.observation_ewma_alpha,
       input.maximum_dispatch_candidates,
       input.maximum_dispatch_trace_decisions};
  result.governor =
      {input.host.usable_bytes, input.device.usable_bytes,
       input.host.emergency_reserve_bytes,
       input.device.emergency_reserve_bytes};
  result.host_cache_bytes = host.cache_bytes;
  result.device_cache_bytes = device.cache_bytes;
  result.host_expert_slots = host.slots;
  result.device_expert_slots = device.slots;
  result.status = Status::success();
  return result;
}

}  // namespace expert::runtime
