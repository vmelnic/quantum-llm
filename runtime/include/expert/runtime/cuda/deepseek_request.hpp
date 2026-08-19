#pragma once

#include "expert/runtime/cuda/deepseek_attention.hpp"
#include "expert/runtime/cuda/deepseek_ffn.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace expert::runtime::cuda {

struct DeepSeekRequestConfig final {
  std::uint32_t max_context_tokens{};
  // Hard per-request CUDA state limit. Creation fails before allocation when
  // the artifact-declared layer program does not fit.
  std::uint64_t device_state_budget_bytes{};
  std::span<const std::uint32_t> compression_ratios;
  std::uint32_t hash_router_layers{};
};

struct DeepSeekRequestStateSize final {
  Status status;
  std::uint64_t attention_bytes{};
  std::uint64_t ffn_bytes{};
  std::uint64_t io_bytes{};
  std::uint64_t stream_bytes{};
  std::uint64_t total_bytes{};
};

[[nodiscard]] DeepSeekRequestStateSize deepseek_request_state_size(
    std::uint32_t max_context_tokens,
    std::span<const std::uint32_t> compression_ratios) noexcept;

struct DeepSeekLayerStateView final {
  const DeepSeekAttentionBinding* attention_weights{};
  DeepSeekAttentionState* attention_state{};
  const DeepSeekFfnBinding* ffn_weights{};
  DeepSeekFfnState* ffn_state{};
  std::uint32_t compress_ratio{};
};

struct DeepSeekRequestStateResult;
struct DeepSeekVerifyStateResult;

// Owns all mutable CUDA state for one artifact-declared layer schedule.
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
  [[nodiscard]] std::uint32_t layer_count() const noexcept {
    return static_cast<std::uint32_t>(attention_weights_.size());
  }
  [[nodiscard]] std::span<const std::uint32_t> compression_ratios()
      const noexcept { return compression_ratios_; }
  [[nodiscard]] DeepSeekLayerStateView layer(
      std::uint32_t index) const noexcept;
  [[nodiscard]] Status embed(std::uint32_t token,
                             void* stream = nullptr) noexcept;
  [[nodiscard]] Status project_logits(void* stream = nullptr) noexcept;
  [[nodiscard]] const float* current_streams() const noexcept {
    return streams_a_;
  }
  // Provider-facing execution buffers. Their roles are defined by the
  // artifact operation ABI: embedding/routed FFN produce the primary stream
  // set, while attention produces the intermediate stream set consumed by
  // the router and routed FFN. Exposing the buffers keeps operation dispatch
  // outside the request-state implementation without exposing ownership.
  [[nodiscard]] float* primary_streams() noexcept { return streams_a_; }
  [[nodiscard]] const float* primary_streams() const noexcept {
    return streams_a_;
  }
  [[nodiscard]] float* attention_streams() noexcept { return streams_b_; }
  [[nodiscard]] const float* attention_streams() const noexcept {
    return streams_b_;
  }
  [[nodiscard]] const float* logits() const noexcept {
    return io_state_ ? io_state_->logits() : nullptr;
  }
  [[nodiscard]] const std::uint32_t* sampled_token() const noexcept {
    return io_state_ ? io_state_->sampled_token() : nullptr;
  }

 private:
  friend DeepSeekRequestStateResult create_deepseek_request_state(
      std::shared_ptr<const DeepSeekResidentModelState>,
      const DeepSeekRequestConfig&) noexcept;
  friend class DeepSeekDecodeController;
  friend class DeepSeekVerifyState;
  DeepSeekRequestState() = default;

  std::shared_ptr<const DeepSeekResidentModelState> model_;
  std::vector<DeepSeekAttentionBinding> attention_weights_;
  std::vector<DeepSeekFfnBinding> ffn_weights_;
  std::vector<std::shared_ptr<DeepSeekAttentionState>> attention_states_;
  std::vector<std::shared_ptr<DeepSeekFfnState>> ffn_states_;
  std::vector<std::uint32_t> compression_ratios_;
  DeepSeekIoBinding io_weights_{};
  std::shared_ptr<DeepSeekIoState> io_state_;
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
