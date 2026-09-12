#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>

namespace expert::runtime::cuda {

// One BF16 GQA segment evaluated by the vendored FlashAttention SM80 kernel.
// Q is [rows, query_heads, head_dim], K/V are
// [kv_heads, tokens, head_dim]. FP32 output and LSE are head-major so callers
// can merge independently decoded segments without an intermediate rounding.
struct FlashGqaSegmentLaunch final {
  const void* queries{};
  const void* keys{};
  const void* values{};
  float* output{};  // [query_heads, rows, head_dim]
  float* lse{};     // [query_heads, rows]
  std::uint32_t rows{};
  std::uint32_t tokens{};
  std::uint32_t query_heads{};
  std::uint32_t kv_heads{};
  std::uint32_t head_dim{};
  bool causal{};
  void* stream{};
};

[[nodiscard]] Status flash_gqa_segment_bf16(
    const FlashGqaSegmentLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
