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
// Rows are split evenly across groups. Each row consumes the corresponding
// group activation from row-major [groups, columns] input in one launch.
[[nodiscard]] Status gemv_grouped_inputs(
    const Int8Matrix& matrix, const float* input, float* output,
    std::uint32_t groups, void* stream) noexcept;
[[nodiscard]] Status gemv_grouped_inputs_batch_weight_reuse(
    const Int8Matrix& matrix, const float* input, float* output,
    std::uint32_t groups, std::uint32_t batch, void* stream) noexcept;
// Inputs and outputs are row-major [batch, columns] and [batch, rows]. Blocks
// for the same matrix row are adjacent so concurrent requests reuse weights.
[[nodiscard]] Status gemv_batch(const Int8Matrix& matrix, const float* input,
                                float* output, std::uint32_t batch,
                                void* stream) noexcept;
// Large-output projection path (for example lm_head). One warp keeps several
// request accumulators and reads each weight row once. Bounded to batch <= 8 to
// avoid the register-pressure regression observed on smaller dense matrices.
[[nodiscard]] Status gemv_batch_weight_reuse(
    const Int8Matrix& matrix, const float* input, float* output,
    std::uint32_t batch, void* stream) noexcept;
[[nodiscard]] Status gemv_f32(const float* matrix, std::uint32_t rows,
                              std::uint32_t columns, const float* input,
                              float* output, void* stream) noexcept;
[[nodiscard]] Status gemv_f32_batch(
    const float* matrix, std::uint32_t rows, std::uint32_t columns,
    const float* input, float* output, std::uint32_t batch,
    void* stream) noexcept;
[[nodiscard]] Status gemv_f32_batch_weight_reuse(
    const float* matrix, std::uint32_t rows, std::uint32_t columns,
    const float* input, float* output, std::uint32_t batch,
    void* stream) noexcept;
[[nodiscard]] Status gemv_bf16(const std::uint16_t* matrix,
                               std::uint32_t rows, std::uint32_t columns,
                               const float* input, float* output,
                               void* stream) noexcept;
[[nodiscard]] Status gemv_bf16_batch(
    const std::uint16_t* matrix, std::uint32_t rows, std::uint32_t columns,
    const float* input, float* output, std::uint32_t batch,
    void* stream) noexcept;
[[nodiscard]] Status rms_norm(const float* input, const float* weight,
                              float* output, std::uint32_t elements,
                              float epsilon, void* stream) noexcept;
[[nodiscard]] Status rms_norm_bf16_weight(
    const float* input, const std::uint16_t* weight, float* output,
    std::uint32_t elements, float epsilon, void* stream) noexcept;
// Input/output are row-major [rows, elements]. Every row has an independent
// RMS statistic and shares the same BF16 norm weight.
[[nodiscard]] Status rms_norm_bf16_weight_batch(
    const float* input, const std::uint16_t* weight, float* output,
    std::uint32_t rows, std::uint32_t elements, float epsilon,
    void* stream) noexcept;
// Qwen3-Next stores zero-centered RMSNorm weights and applies (1 + weight).
[[nodiscard]] Status qwen3_next_rms_norm(
    const float* input, const float* weight, float* output,
    std::uint32_t elements, float epsilon, void* stream) noexcept;
[[nodiscard]] Status add_in_place(float* destination, const float* source,
                                  std::uint32_t elements, void* stream) noexcept;

[[nodiscard]] Status silu_product(const float* gate, const float* up,
                                  float* output, std::uint32_t elements,
                                  void* stream) noexcept;
[[nodiscard]] Status sigmoid_scale_in_place(float* values,
                                            const float* gate,
                                            std::uint32_t elements,
                                            void* stream) noexcept;

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

[[nodiscard]] Status router_topk_normalized(
    const float* input, const float* router_weights, std::uint32_t hidden,
    std::uint32_t experts, std::uint32_t top_k, float* logits,
    float* topk_scores, std::uint32_t* topk_indices, void* stream) noexcept;
[[nodiscard]] Status router_topk_normalized_batch(
    const float* input, const float* router_weights, std::uint32_t rows,
    std::uint32_t hidden, std::uint32_t experts, std::uint32_t top_k,
    float* logits, float* topk_scores, std::uint32_t* topk_indices,
    void* stream) noexcept;

// DeepSeek-V4 sqrt(softplus) routing. Hash layers select through the immutable
// token table; learned layers select by score+bias while weighting by the
// unbiased score. Both normalize the selected weights before route scaling.
[[nodiscard]] Status deepseek_router_hash(
    const float* input, const std::uint16_t* router_weights,
    const std::int64_t* token_experts, std::uint32_t token_id,
    float* logits, float* topk_scores, std::uint32_t* topk_indices,
    float route_scale, void* stream) noexcept;
[[nodiscard]] Status deepseek_router_learned(
    const float* input, const std::uint16_t* router_weights,
    const float* selection_bias, float* logits, float* topk_scores,
    std::uint32_t* topk_indices, float route_scale, void* stream) noexcept;
[[nodiscard]] Status deepseek_router_hash_batch(
    const float* input, const std::uint16_t* router_weights,
    const std::int64_t* token_experts, std::uint32_t token_zero,
    std::uint32_t token_one, float* logits, float* topk_scores,
    std::uint32_t* topk_indices, float route_scale, void* stream) noexcept;
[[nodiscard]] Status deepseek_router_learned_batch(
    const float* input, const std::uint16_t* router_weights,
    const float* selection_bias, float* logits, float* topk_scores,
    std::uint32_t* topk_indices, float route_scale, void* stream) noexcept;

// Qwen3-Next full attention. q_and_gate is laid out per query head as
// [query(head_dim), output_gate(head_dim)]. K/V caches retain only KV heads.
[[nodiscard]] Status qwen3_next_qkv_rope_cache(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, float* key_cache,
    float* value_cache, std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* stream) noexcept;

[[nodiscard]] Status qwen3_next_attention_decode(
    const float* q_and_gate, const float* key_cache,
    const float* value_cache, float* output, std::uint32_t context_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* stream) noexcept;

// Paged FP16 KV variant. Every page is one allocation containing K then V for
// every full-attention layer. The page table contains device page bases for a
// single request slot. Attention uses online softmax and has constant shared
// memory with respect to context length.
[[nodiscard]] Status qwen3_next_qkv_rope_cache_paged_fp16(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* stream) noexcept;

[[nodiscard]] Status qwen3_next_attention_decode_paged_fp16(
    const float* q_and_gate, const void* const* page_table,
    float* output, std::uint32_t context_tokens,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* stream) noexcept;

struct Qwen3NextDeltaLaunch final {
  const float* projected_qkvz{};  // [2*key_dim + 2*value_dim]
  const float* projected_ba{};    // [2*value_heads]
  const float* conv_weights{};    // [2*key_dim + value_dim, 1, kernel]
  const float* dt_bias{};         // [value_heads]
  const float* a_log{};           // [value_heads]
  const float* norm_weight{};     // [value_head_dim]
  float* conv_state{};            // [conv_dim, kernel]
  float* recurrent_state{};       // [value_heads, key_head_dim, value_head_dim]
  float* conv_output{};           // [conv_dim] workspace
  float* output{};                // [value_dim]
  std::uint32_t key_heads{};
  std::uint32_t value_heads{};
  std::uint32_t key_head_dim{};
  std::uint32_t value_head_dim{};
  std::uint32_t conv_kernel{};
  float epsilon{};
  void* stream{};
};

// Exact one-token recurrent Gated DeltaNet update, including causal depthwise
// convolution, q/k L2 normalization, recurrent state and gated RMSNorm.
[[nodiscard]] Status qwen3_next_delta_decode(
    const Qwen3NextDeltaLaunch& launch) noexcept;

[[nodiscard]] Status argmax(const float* values, std::uint32_t count,
                            std::uint32_t* output, void* stream) noexcept;
[[nodiscard]] Status argmax_batch(const float* values, std::uint32_t count,
                                  std::uint32_t batch,
                                  std::uint32_t* output,
                                  void* stream) noexcept;

}  // namespace expert::runtime::cuda
