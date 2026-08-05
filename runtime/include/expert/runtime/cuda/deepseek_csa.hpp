#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

class DeepSeekCompressorState final {
 public:
  DeepSeekCompressorState(float* values, float* scores, std::uint32_t ratio,
                          std::uint32_t projected_width) noexcept;
  ~DeepSeekCompressorState();
  DeepSeekCompressorState(const DeepSeekCompressorState&) = delete;
  DeepSeekCompressorState& operator=(const DeepSeekCompressorState&) = delete;

  [[nodiscard]] float* values() const noexcept { return values_; }
  [[nodiscard]] float* scores() const noexcept { return scores_; }
  [[nodiscard]] std::uint32_t ratio() const noexcept { return ratio_; }
  [[nodiscard]] std::uint32_t projected_width() const noexcept {
    return projected_width_;
  }
  [[nodiscard]] std::uint64_t bytes() const noexcept;
  [[nodiscard]] Status reset(void* stream) noexcept;

 private:
  float* values_{};
  float* scores_{};
  std::uint32_t ratio_{};
  std::uint32_t projected_width_{};
};

struct DeepSeekCompressorStateResult final {
  Status status;
  std::shared_ptr<DeepSeekCompressorState> state;
};

[[nodiscard]] DeepSeekCompressorStateResult create_deepseek_compressor_state(
    std::uint32_t ratio) noexcept;

// Updates persistent decode state and emits a normalized 512-vector exactly
// when a compression group closes. Ratio four uses the checkpoint's overlap
// geometry; ratio 128 uses ordinary 128-token gated pooling.
[[nodiscard]] Status deepseek_compressor_decode(
    DeepSeekCompressorState& state, const float* projected_values,
    const float* projected_scores, const float* ape,
    const std::uint16_t* norm_weight, float* pooled_workspace,
    float* normalized_output, std::uint32_t position, float epsilon,
    void* stream) noexcept;

}  // namespace expert::runtime::cuda
