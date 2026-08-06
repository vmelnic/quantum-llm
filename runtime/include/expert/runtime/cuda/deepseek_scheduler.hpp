#pragma once

#include "expert/runtime/cuda/deepseek_decode.hpp"
#include "expert/runtime/deepseek_catalog.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/route_census.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace expert::runtime::cuda {

enum class DeepSeekScheduledState : std::uint8_t {
  runnable,
  waiting_for_experts,
  complete,
  failed,
  cancelled,
};

struct DeepSeekDecodeSchedulerConfig final {
  std::uint64_t model_id{};
  std::size_t maximum_requests{};
  std::size_t maximum_inflight_acquires{};
  std::size_t maximum_layer_advances_per_poll{};
  std::size_t maximum_inflight_prefetch{1U};
  std::size_t transition_predictions_per_layer{1U};
  // Retain the most recent six routed experts for every layer and controller.
  // This deterministic working set is bounded by 6 * layer count.
  bool retain_previous_route{};
};

struct DeepSeekScheduledRequestSnapshot final {
  DeepSeekScheduledState state{DeepSeekScheduledState::runnable};
  Status status;
  std::uint32_t layer{};
  std::size_t queued_experts{};
  std::size_t inflight_acquires{};
  std::size_t held_leases{};
  std::size_t held_host_leases{};
};

struct DeepSeekDecodeSchedulerSnapshot final {
  std::size_t requests{};
  std::size_t runnable_requests{};
  std::size_t waiting_requests{};
  std::size_t cuda_pending_requests{};
  std::size_t inflight_acquires{};
  std::size_t retained_working_set_experts{};
  std::uint64_t submitted_requests{};
  std::uint64_t rejected_requests{};
  std::uint64_t layer_advances{};
  std::uint64_t cuda_pending_polls{};
  std::uint64_t cuda_waits{};
  std::uint64_t cuda_wait_ns{};
  std::uint64_t expert_suspensions{};
  std::uint64_t acquires_started{};
  std::uint64_t acquires_completed{};
  std::uint64_t host_resolves{};
  std::uint64_t cpu_placements{};
  std::uint64_t hybrid_layers{};
  std::uint64_t route_observations{};
  std::uint64_t prefetch_predictions{};
  std::uint64_t prefetch_scheduled{};
  std::uint64_t prefetch_completed{};
  std::uint64_t prefetch_useful{};
  std::uint64_t prefetch_late{};
  std::uint64_t prefetch_incorrect{};
  std::uint64_t prefetch_cancelled{};
  std::uint64_t prefetch_evicted_before_use{};
  std::uint64_t controller_advance_ns{};
  std::uint64_t expert_wait_ns{};
  std::uint64_t poll_ns{};
  std::uint64_t completed_requests{};
  std::uint64_t failed_requests{};
  std::uint64_t cancelled_requests{};
};

struct DeepSeekHybridSchedulerDependencies final {
  std::shared_ptr<cpu::DeepSeekPackedExecutor> cpu_executor;
  std::shared_ptr<HybridDispatchPlanner> planner;
  std::shared_ptr<RouteCensus> route_census;
};

// Single-owner, non-blocking outer loop for DeepSeek decode controllers.
// poll() never waits for storage or CUDA admission. It advances runnable
// requests fairly, starts only the configured number of cache acquisitions,
// and retains every resulting lease until the suspended layer has resumed.
class DeepSeekDecodeScheduler final {
 public:
  DeepSeekDecodeScheduler(DeepSeekDecodeSchedulerConfig config,
                          ExpertCache& cache,
                          const DeepSeekExpertCatalog& catalog,
                          DeepSeekHybridSchedulerDependencies hybrid = {});
  ~DeepSeekDecodeScheduler();
  DeepSeekDecodeScheduler(const DeepSeekDecodeScheduler&) = delete;
  DeepSeekDecodeScheduler& operator=(const DeepSeekDecodeScheduler&) = delete;

  [[nodiscard]] Status submit(
      std::uint64_t request_id,
      std::shared_ptr<DeepSeekDecodeController> controller,
      const DeepSeekDecodeBegin& begin);
  [[nodiscard]] Status submit_verify(
      std::uint64_t request_id,
      std::shared_ptr<DeepSeekDecodeController> controller,
      const DeepSeekVerifyBegin& begin);
  [[nodiscard]] Status poll();
  // Called by a blocking service loop only after poll() has submitted all
  // currently runnable work. Waits for one request-private CUDA event so the
  // control thread does not busy-spin while other request streams remain free.
  [[nodiscard]] Status wait_for_cuda_progress();
  [[nodiscard]] Status cancel(std::uint64_t request_id) noexcept;
  [[nodiscard]] Status retire(std::uint64_t request_id);
  [[nodiscard]] std::optional<DeepSeekScheduledRequestSnapshot> inspect(
      std::uint64_t request_id) const;
  [[nodiscard]] DeepSeekDecodeSchedulerSnapshot snapshot() const noexcept;

 private:
  struct Core;
  std::unique_ptr<Core> core_;
};

}  // namespace expert::runtime::cuda
