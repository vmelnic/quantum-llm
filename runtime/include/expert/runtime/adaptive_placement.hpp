#pragma once

#include "expert/runtime/expert_cache.hpp"

#include <chrono>
#include <cstdint>
#include <memory>

namespace expert::runtime {

struct AdaptivePlacementConfig final {
  double conservative_h2d_bytes_per_second{8.0 * 1024.0 * 1024.0 * 1024.0};
  double admission_margin{1.25};
  double initial_cpu_ns_per_selection{100'000.0};
  double cpu_cost_ewma_alpha{0.125};
  std::uint32_t maximum_inflight_promotions{1};
};

struct AdaptivePlacementTelemetry final {
  std::uint64_t considered{};
  std::uint64_t admission_rejected{};
  std::uint64_t scheduled{};
  std::uint64_t completed{};
  std::uint64_t failed{};
  std::uint64_t scheduled_bytes{};
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
  void consider(const ExpertKey& key, const PayloadRecord& record,
                std::uint32_t selections);
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
