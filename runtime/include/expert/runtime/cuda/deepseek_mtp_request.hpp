#pragma once

#include "expert/runtime/cuda/deepseek_attention.hpp"
#include "expert/runtime/cuda/deepseek_ffn.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"

#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

struct DeepSeekMtpRequestConfig final {
  std::uint32_t max_context_tokens{};
  std::uint64_t device_state_budget_bytes{};
};

struct DeepSeekMtpRequestStateSize final {
  Status status;
  std::uint64_t attention_bytes{};
  std::uint64_t ffn_bytes{};
  std::uint64_t glue_bytes{};
  std::uint64_t workspace_bytes{};
  std::uint64_t total_bytes{};
};

[[nodiscard]] DeepSeekMtpRequestStateSize deepseek_mtp_request_state_size(
    std::uint32_t max_context_tokens) noexcept;

struct DeepSeekMtpRequestStateResult;

// Request-owned state for one native V4 MTP layer. prepare() advances the
// causal MTP attention and publishes a learned top-6 route. The control plane
// resolves/pins that route through the normal expert cache, then complete()
// executes the FFN and shared vocabulary head. No draft is client-visible here.
class DeepSeekMtpRequestState final {
 public:
  ~DeepSeekMtpRequestState();
  DeepSeekMtpRequestState(const DeepSeekMtpRequestState&) = delete;
  DeepSeekMtpRequestState& operator=(const DeepSeekMtpRequestState&) = delete;

  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint32_t max_context_tokens() const noexcept {
    return max_context_tokens_;
  }
  [[nodiscard]] Status prepare(
      std::uint32_t next_token, const float* target_streams,
      std::uint32_t position, const float* cosine, const float* sine,
      void* stream = nullptr) noexcept;
  [[nodiscard]] Status complete(
      const DeviceExpertEntry* directory_entries,
      std::uint32_t experts_per_layer = 257U,
      void* stream = nullptr) noexcept;
  // Drops an unused route/draft while retaining the already-valid causal MTP
  // attention write for this position (used during prefill and rejection).
  [[nodiscard]] Status abandon_draft() noexcept;

  [[nodiscard]] const std::uint32_t* expert_indices() const noexcept {
    return ffn_state_ ? ffn_state_->expert_indices() : nullptr;
  }
  [[nodiscard]] const float* routing_weights() const noexcept {
    return ffn_state_ ? ffn_state_->routing_weights() : nullptr;
  }
  [[nodiscard]] static constexpr std::uint32_t selection_count() noexcept {
    return DeepSeekFfnState::selection_count();
  }
  [[nodiscard]] const std::uint32_t* draft_token() const noexcept {
    return glue_state_ ? glue_state_->head_state().sampled_token() : nullptr;
  }
  [[nodiscard]] const float* logits() const noexcept {
    return glue_state_ ? glue_state_->head_state().logits() : nullptr;
  }
  [[nodiscard]] const float* mixed_streams() const noexcept {
    return glue_state_ ? glue_state_->mixed_streams() : nullptr;
  }
  [[nodiscard]] const float* attention_streams() const noexcept {
    return attention_streams_;
  }
  [[nodiscard]] const float* block_streams() const noexcept {
    return block_streams_;
  }
  [[nodiscard]] const float* normalized_output() const noexcept {
    return glue_state_ ? glue_state_->head_state().normalized() : nullptr;
  }
  [[nodiscard]] bool prepared() const noexcept { return prepared_; }
  [[nodiscard]] bool complete() const noexcept { return complete_; }

 private:
  friend DeepSeekMtpRequestStateResult create_deepseek_mtp_request_state(
      std::shared_ptr<const DeepSeekResidentModelState>,
      std::shared_ptr<const DeepSeekResidentTensorState>,
      const DeepSeekMtpRequestConfig&) noexcept;
  DeepSeekMtpRequestState() = default;

  std::shared_ptr<const DeepSeekResidentModelState> target_model_;
  std::shared_ptr<const DeepSeekResidentTensorState> mtp_model_;
  DeepSeekIoBinding target_io_;
  DeepSeekMtpGlueBinding glue_weights_;
  DeepSeekAttentionBinding attention_weights_;
  DeepSeekFfnBinding ffn_weights_;
  std::shared_ptr<DeepSeekMtpGlueState> glue_state_;
  std::shared_ptr<DeepSeekAttentionState> attention_state_;
  std::shared_ptr<DeepSeekFfnState> ffn_state_;
  float* workspace_allocation_{};
  float* embedding_{};
  float* attention_streams_{};
  float* block_streams_{};
  std::uint32_t max_context_tokens_{};
  std::uint64_t bytes_{};
  bool prepared_{};
  bool complete_{};
};

struct DeepSeekMtpRequestStateResult final {
  Status status;
  std::shared_ptr<DeepSeekMtpRequestState> state;
};

[[nodiscard]] DeepSeekMtpRequestStateResult create_deepseek_mtp_request_state(
    std::shared_ptr<const DeepSeekResidentModelState> target_model,
    std::shared_ptr<const DeepSeekResidentTensorState> mtp_model,
    const DeepSeekMtpRequestConfig& config) noexcept;

}  // namespace expert::runtime::cuda
