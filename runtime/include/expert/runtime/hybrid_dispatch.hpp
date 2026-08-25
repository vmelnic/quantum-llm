#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
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
  gpu_stable_tie,
  cpu_calibration,
  gpu_cache_warm,
};

struct HybridDispatchConfig final {
  double initial_cpu_ns_per_selection{112'652.0};
  double initial_gpu_ns_per_selection{6'954.0};
  double initial_h2d_bytes_per_second{8.0 * 1024.0 * 1024.0 * 1024.0};
  double observation_ewma_alpha{0.125};
  std::size_t maximum_candidates{4096};
  std::size_t maximum_trace_decisions{256};
  // Do not send flexible misses to the CPU until the running service has
  // measured its actual upload path. Artifact defaults remain telemetry
  // seeds, not evidence that a host lane will shorten the critical path.
  bool require_live_h2d_before_cpu{};
  // Once H2D has been measured, execute one RAM-ready expert on the CPU to
  // calibrate that lane before making adaptive split decisions.
  bool bootstrap_cpu_probe{};
  // Warming a reusable expert is preferable to a CPU tie because the upload
  // benefits later routes. The legacy stable CPU tie remains the default.
  bool prefer_gpu_on_tie{};
  // Preserve at least this many non-resident GPU admissions per plan when
  // flexible candidates exist. Candidates are selected by placement heat.
  std::size_t minimum_gpu_uploads{};
};

struct HybridDispatchCandidate final {
  std::uint32_t expert{};
  std::uint32_t selections{};
  std::uint64_t record_bytes{};
  bool gpu_resident{};
  bool cpu_available{};
  bool gpu_available{};
  std::uint64_t placement_temperature{};
  std::uint64_t last_access{};
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
  std::uint64_t cpu_calibrations{};
  std::uint64_t gpu_cache_warms{};
  std::uint64_t rejected_plans{};
  std::uint64_t cpu_observations{};
  std::uint64_t gpu_observations{};
  std::uint64_t h2d_observations{};
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
  mutable std::mutex mutex_;
  HybridDispatchTelemetry telemetry_;
  std::vector<HybridDispatchDecision> trace_;
  std::size_t next_trace_slot_{};
};

}  // namespace expert::runtime
