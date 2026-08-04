#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace expert::runtime {

enum class HybridExecutor : std::uint8_t {
  gpu_resident,
  gpu_upload,
  cpu_local,
};

enum class HybridDispatchReason : std::uint8_t {
  resident_gpu,
  cpu_only,
  gpu_only,
  cpu_lower_critical_path,
  gpu_lower_critical_path,
  cpu_stable_tie,
};

struct HybridDispatchConfig final {
  double initial_cpu_ns_per_selection{112'652.0};
  double initial_gpu_ns_per_selection{6'954.0};
  double initial_h2d_bytes_per_second{8.0 * 1024.0 * 1024.0 * 1024.0};
  double observation_ewma_alpha{0.125};
  std::size_t maximum_candidates{4096};
  std::size_t maximum_trace_decisions{256};
};

struct HybridDispatchCandidate final {
  std::uint32_t expert{};
  std::uint32_t selections{};
  std::uint64_t record_bytes{};
  bool gpu_resident{};
  bool cpu_available{};
  bool gpu_available{};
};

struct HybridDispatchDecision final {
  std::uint32_t expert{};
  HybridExecutor executor{HybridExecutor::cpu_local};
  HybridDispatchReason reason{HybridDispatchReason::cpu_only};
  std::uint64_t cpu_choice_critical_ns{};
  std::uint64_t gpu_choice_critical_ns{};
};

struct HybridDispatchPlan final {
  Status status;
  std::vector<HybridDispatchDecision> decisions;
  std::uint64_t projected_cpu_ns{};
  std::uint64_t projected_gpu_ns{};
  std::uint64_t projected_h2d_ns{};
  std::uint64_t projected_critical_ns{};
};

struct HybridDispatchTelemetry final {
  std::uint64_t plans{};
  std::uint64_t candidates{};
  std::uint64_t resident_gpu{};
  std::uint64_t cpu_only{};
  std::uint64_t gpu_only{};
  std::uint64_t cpu_cost_wins{};
  std::uint64_t gpu_cost_wins{};
  std::uint64_t stable_ties{};
  std::uint64_t rejected_plans{};
  double cpu_ns_per_selection{};
  double gpu_ns_per_selection{};
  double h2d_bytes_per_second{};
};

// Greedy, deterministic critical-path planner for one routed MoE layer.
// Resident experts never leave the GPU. Flexible RAM-resident misses are
// ordered by descending CPU work and assigned to the lane that minimizes the
// projected CPU/GPU/H2D critical path. All observations and trace storage are
// explicitly bounded.
class HybridDispatchPlanner final {
 public:
  explicit HybridDispatchPlanner(HybridDispatchConfig config = {});

  void observe_cpu(std::uint64_t elapsed_ns,
                   std::uint64_t selections) noexcept;
  void observe_gpu(std::uint64_t elapsed_ns,
                   std::uint64_t selections) noexcept;
  void observe_h2d(std::uint64_t elapsed_ns, std::uint64_t bytes) noexcept;

  [[nodiscard]] HybridDispatchPlan plan(
      std::span<const HybridDispatchCandidate> candidates);
  [[nodiscard]] HybridDispatchTelemetry telemetry() const noexcept;
  [[nodiscard]] std::vector<HybridDispatchDecision> trace() const;

 private:
  HybridDispatchConfig config_;
  HybridDispatchTelemetry telemetry_;
  std::vector<HybridDispatchDecision> trace_;
  std::size_t next_trace_slot_{};
};

}  // namespace expert::runtime
