#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>

namespace expert::runtime::cuda {

struct Int8Matrix final {
  const std::int8_t* weights{};
  const float* scales{};
  std::uint32_t rows{};
  std::uint32_t columns{};
};

[[nodiscard]] Status embedding(const Int8Matrix& matrix, std::uint32_t token,
                               float* output, void* stream) noexcept;
[[nodiscard]] Status gemv(const Int8Matrix& matrix, const float* input,
                          float* output, void* stream) noexcept;
[[nodiscard]] Status gemv_f32(const float* matrix, std::uint32_t rows,
                              std::uint32_t columns, const float* input,
                              float* output, void* stream) noexcept;
[[nodiscard]] Status rms_norm(const float* input, const float* weight,
                              float* output, std::uint32_t elements,
                              float epsilon, void* stream) noexcept;
[[nodiscard]] Status add_in_place(float* destination, const float* source,
                                  std::uint32_t elements, void* stream) noexcept;

// Normalizes Q/K, applies HF OLMoE rotary embedding, and writes K/V for the
// current position into persistent caches [context, heads, head_dim].
[[nodiscard]] Status qkv_rope_cache(
    float* query, float* key, const float* value, const float* q_norm_weight,
    const float* k_norm_weight, float* key_cache, float* value_cache,
    std::uint32_t position, std::uint32_t heads, std::uint32_t head_dim,
    float epsilon, float rope_theta, void* stream) noexcept;

[[nodiscard]] Status attention_decode(
    const float* query, const float* key_cache, const float* value_cache,
    float* output, std::uint32_t context_tokens, std::uint32_t heads,
    std::uint32_t head_dim, void* stream) noexcept;

[[nodiscard]] Status router_topk(
    const float* input, const float* router_weights, std::uint32_t hidden,
    std::uint32_t experts, std::uint32_t top_k, float* logits,
    float* topk_scores, std::uint32_t* topk_indices, void* stream) noexcept;

[[nodiscard]] Status argmax(const float* values, std::uint32_t count,
                            std::uint32_t* output, void* stream) noexcept;

}  // namespace expert::runtime::cuda
