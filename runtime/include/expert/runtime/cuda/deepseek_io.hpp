#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

inline constexpr std::uint32_t kDeepSeekHidden = 4096U;
inline constexpr std::uint32_t kDeepSeekVocab = 129280U;

struct DeepSeekIoBinding final {
  const std::uint16_t* embedding{};
  const std::uint16_t* final_norm{};
  const std::uint16_t* head{};
  const float* head_function{};
  const float* head_base{};
  const float* head_scale{};
};

struct DeepSeekIoStateResult;

class DeepSeekIoState final {
 public:
  ~DeepSeekIoState();
  DeepSeekIoState(const DeepSeekIoState&) = delete;
  DeepSeekIoState& operator=(const DeepSeekIoState&) = delete;

  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const std::uint32_t* sampled_token() const noexcept {
    return sampled_token_;
  }
  [[nodiscard]] const float* head_gates() const noexcept { return head_pre_; }
  [[nodiscard]] const float* collapsed() const noexcept { return collapsed_; }
  [[nodiscard]] const float* normalized() const noexcept { return normalized_; }

 private:
  friend DeepSeekIoStateResult create_deepseek_io_state() noexcept;
  friend Status deepseek_embed(const DeepSeekIoBinding&, std::uint32_t,
                               float*, void*) noexcept;
  friend Status deepseek_head(const DeepSeekIoBinding&, const float*,
                              DeepSeekIoState&, float, void*) noexcept;
  friend Status deepseek_hc_head(const DeepSeekIoBinding&, const float*,
                                 DeepSeekIoState&, float, void*) noexcept;
  DeepSeekIoState(void* allocation, std::uint64_t bytes) noexcept;
  void map(void* allocation) noexcept;

  void* allocation_{};
  std::uint64_t bytes_{};
  float* normalized_streams_{};
  float* head_mixes_{};
  float* head_pre_{};
  float* collapsed_{};
  float* normalized_{};
  float* logits_{};
  std::uint32_t* sampled_token_{};
};

struct DeepSeekIoStateResult final {
  Status status;
  std::shared_ptr<DeepSeekIoState> state;
};

[[nodiscard]] std::uint64_t deepseek_io_state_size() noexcept;
[[nodiscard]] DeepSeekIoStateResult create_deepseek_io_state() noexcept;

// Expands one checkpoint BF16 embedding row into the four FP32 runtime streams.
[[nodiscard]] Status deepseek_embed(
    const DeepSeekIoBinding& weights, std::uint32_t token, float* streams,
    void* stream) noexcept;

// Applies the checkpoint HC head, final RMSNorm and untied BF16 output head,
// then writes both complete logits and greedy argmax into request-owned state.
[[nodiscard]] Status deepseek_head(
    const DeepSeekIoBinding& weights, const float* streams,
    DeepSeekIoState& state, float epsilon, void* stream) noexcept;

// Applies only the hyper-head collapse and final RMSNorm. MTP reuses this
// boundary before the shared vocabulary head, avoiding a duplicate kernel ABI.
[[nodiscard]] Status deepseek_hc_head(
    const DeepSeekIoBinding& weights, const float* streams,
    DeepSeekIoState& state, float epsilon, void* stream) noexcept;

}  // namespace expert::runtime::cuda
