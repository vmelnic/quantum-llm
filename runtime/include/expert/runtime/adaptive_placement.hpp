#pragma once

#include "expert/runtime/expert_cache.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>

namespace expert::runtime {

struct AdaptivePlacementConfig final {
  bool enable_prefetch{true};
  double conservative_h2d_bytes_per_second{8.0 * 1024.0 * 1024.0 * 1024.0};
  double admission_margin{1.25};
  double initial_cpu_ns_per_selection{100'000.0};
  double cpu_cost_ewma_alpha{0.125};
  double routing_score_reuse_weight{1.0};
  std::uint32_t maximum_inflight_promotions{1};
  std::uint32_t maximum_candidates{4096};
  std::uint32_t minimum_recent_observations{2};
  std::uint64_t candidate_ttl_epochs{192};
};

struct AdaptivePlacementTelemetry final {
  std::uint64_t considered{};
  std::uint64_t admission_rejected{};
  std::uint64_t scheduled{};
  std::uint64_t completed{};
  std::uint64_t failed{};
  std::uint64_t scheduled_bytes{};
  std::uint64_t useful_prefetches{};
  std::uint64_t useful_prefetch_bytes{};
  std::uint64_t wasted_prefetches{};
  std::uint64_t wasted_prefetch_bytes{};
  std::uint64_t stale_cancellations{};
  std::uint64_t candidate_evictions{};
  std::uint64_t credit_rejections{};
};

// Cost-driven RAM -> VRAM placement. The current expert invocation is never
// blocked: callers execute it on CPU, then offer its accumulated reuse debt to
// this planner for a future asynchronous promotion.
class AdaptivePlacementPlanner final {
 public:
  AdaptivePlacementPlanner(ExpertCache& cache,
                           AdaptivePlacementConfig config = {});
  ~AdaptivePlacementPlanner();
  AdaptivePlacementPlanner(const AdaptivePlacementPlanner&) = delete;
  AdaptivePlacementPlanner& operator=(const AdaptivePlacementPlanner&) = delete;

  void observe_cpu_batch(std::uint64_t elapsed_ns,
                         std::uint64_t selections) noexcept;
  // Advances the bounded prediction epoch and attributes completed/pending
  // promotions that are selected by the exact router.
  void observe_routes(std::span<const ExpertKey> gpu_resident,
                      std::span<const ExpertKey> missing);
  void consider(const ExpertKey& key, const PayloadRecord& record,
                std::uint32_t selections, double routing_score_sum = 0.0);
  void poll();
  [[nodiscard]] Status drain(std::chrono::milliseconds timeout);
  // Completes every profitable queued replacement and freezes admission.
  // Route/access feedback may continue, but a decode epoch cannot mutate VRAM
  // placement until resume() is called at a safe request boundary.
  [[nodiscard]] Status quiesce(std::chrono::milliseconds timeout);
  void resume() noexcept;
  [[nodiscard]] bool frozen() const noexcept;
  [[nodiscard]] AdaptivePlacementTelemetry telemetry() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace expert::runtime
