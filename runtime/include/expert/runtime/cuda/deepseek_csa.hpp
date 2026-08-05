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

// Publishes one normalized compressed KV vector in the checkpoint's BF16 cache
// ABI. RoPE applies to the last 64 dimensions; the first 448 receive the
// block-64 power-of-two FP8 quantize/dequantize simulation used during QAT.
[[nodiscard]] Status deepseek_compressed_kv_publish(
    const float* normalized, const float* cosine, const float* sine,
    std::uint16_t* cache, std::uint32_t slot, void* stream) noexcept;

// Online-softmax sparse decode. The learned sink contributes to the
// denominator with a zero value vector. Indices may contain -1 sentinels.
[[nodiscard]] Status deepseek_sparse_attention_decode(
    const std::uint16_t* query, const std::uint16_t* kv_cache,
    const std::int32_t* indices, std::uint32_t selected,
    const float* attention_sink, std::uint16_t* output,
    std::uint32_t heads, void* stream) noexcept;

}  // namespace expert::runtime::cuda
