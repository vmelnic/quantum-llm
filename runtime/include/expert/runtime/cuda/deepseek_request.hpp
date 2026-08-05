#pragma once

#include "expert/runtime/cuda/deepseek_attention.hpp"
#include "expert/runtime/cuda/deepseek_ffn.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

inline constexpr std::uint32_t kDeepSeekLayers = 43U;

struct DeepSeekRequestConfig final {
  std::uint32_t max_context_tokens{};
  // Hard per-request CUDA state limit. Creation fails before allocation when
  // the complete 43-layer state does not fit.
  std::uint64_t device_state_budget_bytes{};
};

struct DeepSeekRequestStateSize final {
  Status status;
  std::uint64_t attention_bytes{};
  std::uint64_t ffn_bytes{};
  std::uint64_t stream_bytes{};
  std::uint64_t total_bytes{};
};

[[nodiscard]] DeepSeekRequestStateSize deepseek_request_state_size(
    std::uint32_t max_context_tokens) noexcept;

struct DeepSeekLayerStateView final {
  const DeepSeekAttentionBinding* attention_weights{};
  DeepSeekAttentionState* attention_state{};
  const DeepSeekFfnBinding* ffn_weights{};
  DeepSeekFfnState* ffn_state{};
  std::uint32_t compress_ratio{};
};

struct DeepSeekRequestStateResult;

// Owns all mutable CUDA state for one request across the 43-layer schedule.
// It also retains the immutable resident model, so every binding remains valid
// for the complete request lifetime.
class DeepSeekRequestState final {
 public:
  ~DeepSeekRequestState();
  DeepSeekRequestState(const DeepSeekRequestState&) = delete;
  DeepSeekRequestState& operator=(const DeepSeekRequestState&) = delete;

  [[nodiscard]] std::uint32_t max_context_tokens() const noexcept {
    return max_context_tokens_;
  }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] static constexpr std::uint32_t layer_count() noexcept {
    return kDeepSeekLayers;
  }
  [[nodiscard]] DeepSeekLayerStateView layer(
      std::uint32_t index) const noexcept;

 private:
  friend DeepSeekRequestStateResult create_deepseek_request_state(
      std::shared_ptr<const DeepSeekResidentModelState>,
      const DeepSeekRequestConfig&) noexcept;
  friend class DeepSeekDecodeController;
  DeepSeekRequestState() = default;

  std::shared_ptr<const DeepSeekResidentModelState> model_;
  std::array<DeepSeekAttentionBinding, kDeepSeekLayers> attention_weights_{};
  std::array<DeepSeekFfnBinding, kDeepSeekLayers> ffn_weights_{};
  std::array<std::shared_ptr<DeepSeekAttentionState>, kDeepSeekLayers>
      attention_states_{};
  std::array<std::shared_ptr<DeepSeekFfnState>, kDeepSeekLayers> ffn_states_{};
  std::uint32_t max_context_tokens_{};
  std::uint64_t bytes_{};
  float* stream_allocation_{};
  float* streams_a_{};
  float* streams_b_{};
};

struct DeepSeekRequestStateResult final {
  Status status;
  std::shared_ptr<DeepSeekRequestState> state;
};

[[nodiscard]] DeepSeekRequestStateResult create_deepseek_request_state(
    std::shared_ptr<const DeepSeekResidentModelState> model,
    const DeepSeekRequestConfig& config) noexcept;

}  // namespace expert::runtime::cuda
