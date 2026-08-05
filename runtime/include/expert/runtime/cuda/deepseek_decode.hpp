#pragma once

#include "expert/runtime/cuda/deepseek_request.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace expert::runtime::cuda {

struct DeepSeekDecodeRope final {
  const float* base_cosine{};
  const float* base_sine{};
  const float* compressed_cosine{};
  const float* compressed_sine{};
  const float* ratio_four_group_cosine{};
  const float* ratio_four_group_sine{};
  const float* ratio_128_group_cosine{};
  const float* ratio_128_group_sine{};
};

struct DeepSeekDecodeBegin final {
  const float* input_streams{};  // device [4, 4096]
  DeepSeekDecodeRope rope;
  std::uint32_t position{};
  std::uint32_t token_id{};
  // Half-open range. Full-model decode uses [0, 43); a strict subrange is
  // useful for pipeline ownership and independently qualified slices.
  std::uint32_t first_layer{};
  std::uint32_t layer_limit{kDeepSeekLayers};
};

enum class DeepSeekDecodeProgress : std::uint8_t {
  layer_complete,
  needs_experts,
  token_complete,
};

struct DeepSeekDecodeAdvanceResult final {
  Status status;
  DeepSeekDecodeProgress progress{DeepSeekDecodeProgress::layer_complete};
  std::uint32_t layer{};
  std::vector<std::uint32_t> missing_experts;
  // Ready entries are device-pinned across a cold-plan suspension. The
  // scheduler must convert them to cache leases before loading misses so an
  // eviction cannot wait on the controller's own directory pin.
  std::vector<std::uint32_t> ready_experts;
  // Exact routed selection excluding the invariant shared expert.
  std::vector<std::uint32_t> routed_experts;
};

struct DeepSeekRouteTraceEntry final {
  std::uint32_t layer{};
  // The shared expert is invariant and intentionally excluded.
  std::array<std::uint32_t, 6U> routed_experts{};
};

struct DeepSeekDecodeControllerResult;

// Executes one layer per advance() call. A cache miss returns control without
// recomputing attention/router; after the caller publishes the missing experts,
// advance() replans and resumes the same FFN. Independent directory pin tokens
// keep concurrent requests safe while they wait.
class DeepSeekDecodeController final {
 public:
  ~DeepSeekDecodeController();
  DeepSeekDecodeController(const DeepSeekDecodeController&) = delete;
  DeepSeekDecodeController& operator=(const DeepSeekDecodeController&) = delete;

  [[nodiscard]] Status begin(const DeepSeekDecodeBegin& launch) noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult advance() noexcept;
  [[nodiscard]] Status cancel() noexcept;

  [[nodiscard]] const float* output_streams() const noexcept {
    return complete_ ? request_->streams_a_ : nullptr;
  }
  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] std::uint32_t current_layer() const noexcept {
    return current_layer_;
  }
  [[nodiscard]] std::span<const DeepSeekRouteTraceEntry> route_trace()
      const noexcept {
    return route_trace_;
  }

 private:
  friend DeepSeekDecodeControllerResult create_deepseek_decode_controller(
      std::shared_ptr<DeepSeekRequestState>,
      std::shared_ptr<CudaExpertDirectory>, void*) noexcept;
  DeepSeekDecodeController(std::shared_ptr<DeepSeekRequestState> request,
                           std::shared_ptr<CudaExpertDirectory> directory,
                           void* stream) noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult fail(Status status) noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult plan_and_execute() noexcept;

  std::shared_ptr<DeepSeekRequestState> request_;
  std::shared_ptr<CudaExpertDirectory> directory_;
  void* stream_{};
  DeepSeekDecodeRope rope_{};
  std::uint32_t position_{};
  std::uint32_t token_id_{};
  std::uint32_t current_layer_{};
  std::uint32_t layer_limit_{};
  std::uint64_t pin_id_{};
  std::vector<DeepSeekRouteTraceEntry> route_trace_;
  bool active_{};
  bool waiting_for_experts_{};
  bool complete_{};
};

struct DeepSeekDecodeControllerResult final {
  Status status;
  std::shared_ptr<DeepSeekDecodeController> controller;
};

[[nodiscard]] DeepSeekDecodeControllerResult create_deepseek_decode_controller(
    std::shared_ptr<DeepSeekRequestState> request,
    std::shared_ptr<CudaExpertDirectory> directory,
    void* stream = nullptr) noexcept;

}  // namespace expert::runtime::cuda
