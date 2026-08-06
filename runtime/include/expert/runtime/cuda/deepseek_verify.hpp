#pragma once

#include "expert/runtime/cuda/deepseek_request.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

struct DeepSeekVerifyStateSize final {
  Status status;
  std::uint64_t secondary_ffn_bytes{};
  std::uint64_t pair_ffn_bytes{};
  std::uint64_t secondary_io_bytes{};
  std::uint64_t secondary_stream_bytes{};
  std::uint64_t rollback_bytes{};
  std::uint64_t total_bytes{};
};

[[nodiscard]] DeepSeekVerifyStateSize deepseek_verify_state_size() noexcept;

struct DeepSeekVerifyLayerStateView final {
  const DeepSeekAttentionBinding* attention_weights{};
  DeepSeekAttentionState* attention_state{};
  const DeepSeekFfnBinding* ffn_weights{};
  std::array<DeepSeekFfnState*, 2U> ffn_states{};
  std::uint32_t compress_ratio{};
  void* rollback_checkpoint{};
};

struct DeepSeekVerifyStateResult;

// Bounded request-owned state for a two-row target verification transaction.
// Row zero reuses the ordinary request state and is always committed. Row one
// owns only the extra FFN/stream/head workspace required while it is
// speculative; both rows share the causal attention cache.
class DeepSeekVerifyState final {
 public:
  ~DeepSeekVerifyState();
  DeepSeekVerifyState(const DeepSeekVerifyState&) = delete;
  DeepSeekVerifyState& operator=(const DeepSeekVerifyState&) = delete;

  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::shared_ptr<DeepSeekRequestState> primary_request()
      const noexcept {
    return request_;
  }
  [[nodiscard]] DeepSeekVerifyLayerStateView layer(
      std::uint32_t index) const noexcept;
  [[nodiscard]] DeepSeekFfnPairWorkspace* ffn_workspace() const noexcept {
    return ffn_workspace_.get();
  }
  [[nodiscard]] const float* primary_streams() const noexcept;
  [[nodiscard]] const float* speculative_streams() const noexcept {
    return speculative_streams_a_;
  }
  [[nodiscard]] float* speculative_attention_output() const noexcept {
    return speculative_streams_b_;
  }
  [[nodiscard]] Status embed_speculative(std::uint32_t token,
                                         void* stream) noexcept;
  [[nodiscard]] Status begin_transaction(
      std::uint32_t speculative_position) noexcept;
  [[nodiscard]] Status checkpoint_layer(std::uint32_t layer,
                                        void* stream) noexcept;
  [[nodiscard]] Status project_pair_logits(void* stream) noexcept;
  [[nodiscard]] const std::uint32_t* primary_sampled_token() const noexcept;
  [[nodiscard]] const std::uint32_t* bonus_sampled_token() const noexcept;

  // Accept copies the speculative final streams into the ordinary request.
  // Reject restores only recurrent compression state; explicit KV/CSA slots
  // remain logically outside the committed position and are overwritten on
  // replay. abort_transaction() is the failure/cancellation equivalent.
  [[nodiscard]] Status finish_transaction(bool accept,
                                          void* stream) noexcept;
  [[nodiscard]] Status abort_transaction(void* stream) noexcept;

 private:
  friend DeepSeekVerifyStateResult create_deepseek_verify_state(
      std::shared_ptr<DeepSeekRequestState>, std::uint64_t) noexcept;
  friend class DeepSeekDecodeController;
  DeepSeekVerifyState() = default;
  [[nodiscard]] Status restore_checkpoints(void* stream) noexcept;

  std::shared_ptr<DeepSeekRequestState> request_;
  std::array<std::shared_ptr<DeepSeekFfnState>, kDeepSeekLayers>
      secondary_ffn_states_{};
  std::shared_ptr<DeepSeekFfnPairWorkspace> ffn_workspace_;
  std::shared_ptr<DeepSeekIoState> secondary_io_state_;
  void* stream_allocation_{};
  float* speculative_streams_a_{};
  float* speculative_streams_b_{};
  void* rollback_allocation_{};
  std::array<void*, kDeepSeekLayers> rollback_checkpoints_{};
  std::array<bool, kDeepSeekLayers> checkpointed_{};
  std::uint64_t bytes_{};
  std::uint32_t speculative_position_{};
  bool active_{};
  bool recurrent_boundary_{};
};

struct DeepSeekVerifyStateResult final {
  Status status;
  std::shared_ptr<DeepSeekVerifyState> state;
};

[[nodiscard]] DeepSeekVerifyStateResult create_deepseek_verify_state(
    std::shared_ptr<DeepSeekRequestState> request,
    std::uint64_t device_state_budget_bytes) noexcept;

}  // namespace expert::runtime::cuda
