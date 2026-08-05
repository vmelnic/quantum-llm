#pragma once

#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/resource_governor.hpp"

#include <cstddef>
#include <cstdint>

namespace expert::runtime {

struct PlacementCostProfile final {
  double cpu_ns_per_selection{};
  double gpu_ns_per_selection{};
  double h2d_bytes_per_second{};
  std::uint64_t cpu_samples{};
  std::uint64_t gpu_samples{};
  std::uint64_t h2d_sample_bytes{};
};

struct PlacementDomainInput final {
  // Operator/probe-approved bytes available to this runtime after external OS,
  // display, driver, and co-tenant needs. This is never inferred from model
  // parameter count.
  std::uint64_t usable_bytes{};
  // Dense weights, request/KV state, workspaces, and staging that the governor
  // must reserve independently of the trimmable expert cache.
  std::uint64_t fixed_bytes{};
  std::uint64_t emergency_reserve_bytes{};
  std::uint64_t expert_record_bytes{};
  std::uint64_t minimum_expert_slots{};
  // Zero means all slots that fit; otherwise this is a hard operator cap.
  std::uint64_t maximum_expert_slots{};
};

struct PlacementProfileInput final {
  PlacementCostProfile costs;
  PlacementDomainInput host;
  PlacementDomainInput device;
  double observation_ewma_alpha{0.125};
  std::size_t maximum_dispatch_candidates{4096U};
  std::size_t maximum_dispatch_trace_decisions{256U};
};

struct PlacementProfilePlan final {
  Status status;
  HybridDispatchConfig dispatch;
  MemoryGovernorConfig governor;
  std::uint64_t host_cache_bytes{};
  std::uint64_t device_cache_bytes{};
  std::uint64_t host_expert_slots{};
  std::uint64_t device_expert_slots{};
};

// Converts measured costs and explicit memory envelopes into the two runtime
// authorities. It rejects missing measurements and under-provisioned minima;
// it never overcommits or silently reduces a required slot count.
[[nodiscard]] PlacementProfilePlan solve_placement_profile(
    const PlacementProfileInput& input) noexcept;

}  // namespace expert::runtime
