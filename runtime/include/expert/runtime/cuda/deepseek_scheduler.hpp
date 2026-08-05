#pragma once

#include "expert/runtime/cuda/deepseek_decode.hpp"
#include "expert/runtime/deepseek_catalog.hpp"
#include "expert/runtime/expert_cache.hpp"

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
};

struct DeepSeekScheduledRequestSnapshot final {
  DeepSeekScheduledState state{DeepSeekScheduledState::runnable};
  Status status;
  std::uint32_t layer{};
  std::size_t queued_experts{};
  std::size_t inflight_acquires{};
  std::size_t held_leases{};
};

struct DeepSeekDecodeSchedulerSnapshot final {
  std::size_t requests{};
  std::size_t runnable_requests{};
  std::size_t waiting_requests{};
  std::size_t inflight_acquires{};
  std::uint64_t submitted_requests{};
  std::uint64_t rejected_requests{};
  std::uint64_t layer_advances{};
  std::uint64_t expert_suspensions{};
  std::uint64_t acquires_started{};
  std::uint64_t acquires_completed{};
  std::uint64_t completed_requests{};
  std::uint64_t failed_requests{};
  std::uint64_t cancelled_requests{};
};

// Single-owner, non-blocking outer loop for DeepSeek decode controllers.
// poll() never waits for storage or CUDA admission. It advances runnable
// requests fairly, starts only the configured number of cache acquisitions,
// and retains every resulting lease until the suspended layer has resumed.
class DeepSeekDecodeScheduler final {
 public:
  DeepSeekDecodeScheduler(DeepSeekDecodeSchedulerConfig config,
                          ExpertCache& cache,
                          const DeepSeekExpertCatalog& catalog);
  ~DeepSeekDecodeScheduler();
  DeepSeekDecodeScheduler(const DeepSeekDecodeScheduler&) = delete;
  DeepSeekDecodeScheduler& operator=(const DeepSeekDecodeScheduler&) = delete;

  [[nodiscard]] Status submit(
      std::uint64_t request_id,
      std::shared_ptr<DeepSeekDecodeController> controller,
      const DeepSeekDecodeBegin& begin);
  [[nodiscard]] Status poll();
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
