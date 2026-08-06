#pragma once

#include "expert/runtime/cuda/deepseek_dense.hpp"
#include "expert/runtime/cuda/deepseek_io.hpp"

#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

inline constexpr std::uint32_t kDeepSeekMtpStreams = 4U;

struct DeepSeekMtpGlueBinding final {
  Int8Matrix e_projection;
  Int8Matrix h_projection;
  const std::uint16_t* embedding_norm{};
  const std::uint16_t* hidden_norm{};
  const std::uint16_t* output_norm{};
  const float* head_function{};
  const float* head_base{};
  const float* head_scale{};
};

struct DeepSeekMtpGlueStateResult;

class DeepSeekMtpGlueState final {
 public:
  ~DeepSeekMtpGlueState();
  DeepSeekMtpGlueState(const DeepSeekMtpGlueState&) = delete;
  DeepSeekMtpGlueState& operator=(const DeepSeekMtpGlueState&) = delete;

  [[nodiscard]] std::uint64_t bytes() const noexcept;
  [[nodiscard]] const float* normalized_token() const noexcept {
    return normalized_token_;
  }
  [[nodiscard]] const float* normalized_streams() const noexcept {
    return normalized_streams_;
  }
  [[nodiscard]] const float* mixed_streams() const noexcept { return mixed_; }
  [[nodiscard]] const DeepSeekIoState& head_state() const noexcept {
    return *head_state_;
  }

 private:
  friend DeepSeekMtpGlueStateResult create_deepseek_mtp_glue_state() noexcept;
  friend Status deepseek_mtp_mix(const DeepSeekMtpGlueBinding&, const float*,
                                 const float*, DeepSeekMtpGlueState&, float,
                                 void*) noexcept;
  friend Status deepseek_mtp_collapse(const DeepSeekMtpGlueBinding&,
                                      const float*, DeepSeekMtpGlueState&,
                                      float, void*) noexcept;
  DeepSeekMtpGlueState(void* allocation, std::uint64_t allocation_bytes,
                       std::shared_ptr<DeepSeekIoState> head_state) noexcept;

  void* allocation_{};
  std::uint64_t allocation_bytes_{};
  float* normalized_token_{};
  float* normalized_streams_{};
  float* token_projection_{};
  float* mixed_{};
  std::shared_ptr<DeepSeekIoState> head_state_;
};

struct DeepSeekMtpGlueStateResult final {
  Status status;
  std::shared_ptr<DeepSeekMtpGlueState> state;
};

[[nodiscard]] DeepSeekMtpGlueStateResult
create_deepseek_mtp_glue_state() noexcept;

// Produces h_proj(hnorm(previous_streams)) + e_proj(enorm(embedding)).
[[nodiscard]] Status deepseek_mtp_mix(
    const DeepSeekMtpGlueBinding& weights, const float* embedding,
    const float* previous_streams, DeepSeekMtpGlueState& state,
    float epsilon, void* stream) noexcept;

// Applies the MTP-specific HC head and final norm to the post-decoder streams.
// The boundary smoke may pass state.mixed_streams() before the decoder exists.
[[nodiscard]] Status deepseek_mtp_collapse(
    const DeepSeekMtpGlueBinding& weights, const float* decoder_streams,
    DeepSeekMtpGlueState& state, float epsilon, void* stream) noexcept;

}  // namespace expert::runtime::cuda
