#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>

namespace expert::runtime::cuda {

// The exact staged-FP16 attention path is GEMM-shaped by total query heads,
// not by warps per KV head. Keep this contract shared by descriptor admission,
// QKV preparation, and attention execution so a supported GQA geometry cannot
// pass startup and then fail on its first request.
inline constexpr std::uint32_t kMaximumExactFp16GroupedQueryHeads = 16U;

struct Int8Matrix final {
  const std::int8_t* weights{};
  const float* scales{};
  std::uint32_t rows{};
  std::uint32_t columns{};
};

// Dense Expert Pack quant ABI 3. Rows retain their logical column count while
// storage pads every row to a complete block of 32 nibbles. One UE8M0 scale
// byte belongs to each padded block.
struct Fp4Block32Matrix final {
  const std::uint8_t* weights{};
  const std::uint8_t* scales{};
  std::uint32_t rows{};
  std::uint32_t columns{};
  std::uint32_t padded_columns{};
};

[[nodiscard]] Status fp4_embedding(const Fp4Block32Matrix& matrix,
                                   std::uint32_t token, float* output,
                                   void* stream) noexcept;
[[nodiscard]] Status fp4_embedding_batch(
    const Fp4Block32Matrix& matrix, const std::uint32_t* tokens,
    float* output, std::uint32_t batch, void* stream) noexcept;
// Quantizes row-major FP32 activations once so several projections sharing an
// input can execute direct packed-FP4 dp4a GEMVs without redundant work.
[[nodiscard]] Status quantize_q8_batch(
    const float* input, std::int8_t* output, float* scales,
    std::uint32_t rows, std::uint32_t columns,
    std::uint32_t padded_columns, void* stream) noexcept;
[[nodiscard]] Status fp4_gemv_q8_batch(
    const Fp4Block32Matrix& matrix, const std::int8_t* input,
    const float* input_scales, float* output, std::uint32_t batch,
    void* stream) noexcept;
// One warp owns one output row and accumulates up to eight activation rows
// while reading every packed FP4 weight exactly once.
[[nodiscard]] Status fp4_gemv_q8_batch_weight_reuse(
    const Fp4Block32Matrix& matrix, const std::int8_t* input,
    const float* input_scales, float* output, std::uint32_t batch,
    void* stream) noexcept;
// SM80 Tensor Core path for causal prefill. A CTA retains four 128x32 output
// tiles and shares every decoded FP4 weight block across as many as 512
// activation rows.
[[nodiscard]] Status fp4_gemm_q8_block32(
    const Fp4Block32Matrix& matrix, const std::int8_t* input,
    const float* input_scales, float* output, std::uint32_t batch,
    void* stream) noexcept;

// Layer-major prefill may decode an FP4 matrix once and reuse the exact BF16
// operand across every prompt chunk for that artifact operation. Workspace
// capacities are explicit; the runtime owns placement and lifetime.
[[nodiscard]] Status fp4_decode_matrix_bf16(
    const Fp4Block32Matrix& matrix, void* decoded_weights,
    std::size_t decoded_weight_bytes, void* stream) noexcept;
[[nodiscard]] Status bf16_gemm_q8_block32(
    const Fp4Block32Matrix& matrix, const void* decoded_weights,
    std::size_t decoded_weight_bytes, const std::int8_t* input,
    const float* input_scales, void* decoded_input,
    std::size_t decoded_input_bytes, float* output, std::uint32_t batch,
    void* stream) noexcept;

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
// Large-output projection path (for example lm_head). One warp keeps up to
// eight request accumulators and reads each weight row once. Larger batches are
// split into contiguous eight-row tiles by the launcher.
[[nodiscard]] Status gemv_batch_weight_reuse(
    const Int8Matrix& matrix, const float* input, float* output,
    std::uint32_t batch, void* stream) noexcept;
// Exact dequantization boundary for large causal-prefill tiles. The immutable
// INT8 rows are expanded once into caller-owned FP32 scratch, then one SGEMM
// consumes the complete activation tile. This avoids rescanning dense weights
// once per small GEMV group while preserving the artifact's row scales.
[[nodiscard]] Status int8_gemm_f32_batch(
    const Int8Matrix& matrix, const float* input, float* output,
    std::uint32_t batch, float* decoded_matrix,
    std::uint64_t decoded_matrix_values, void* stream) noexcept;
[[nodiscard]] Status int8_grouped_gemm_f32_batch(
    const Int8Matrix& matrix, const float* input, float* output,
    std::uint32_t groups, std::uint32_t batch, float* decoded_matrix,
    std::uint64_t decoded_matrix_values, void* stream) noexcept;
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
[[nodiscard]] Status bf16_gemm_f32_batch(
    const std::uint16_t* matrix, std::uint32_t rows, std::uint32_t columns,
    const float* input, float* output, std::uint32_t batch,
    float* decoded_matrix, std::uint64_t decoded_matrix_values,
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
[[nodiscard]] Status zero_centered_rms_norm_batch(
    const float* input, const float* weight, float* output,
    std::uint32_t rows, std::uint32_t elements, float epsilon,
    void* stream) noexcept;
[[nodiscard]] Status rms_norm_batch(
    const float* input, const float* weight, float* output,
    std::uint32_t rows, std::uint32_t elements, float epsilon,
    void* stream) noexcept;
[[nodiscard]] Status weightless_rms_norm_batch(
    const float* input, float* output, std::uint32_t rows,
    std::uint32_t elements, float epsilon, void* stream) noexcept;
[[nodiscard]] Status add_in_place(float* destination, const float* source,
                                  std::uint32_t elements, void* stream) noexcept;
[[nodiscard]] Status add_bias_in_place(float* destination, const float* bias,
                                       std::uint32_t rows,
                                       std::uint32_t columns,
                                       void* stream) noexcept;
[[nodiscard]] Status layer_norm_batch(
    const float* input, const float* weight, const float* bias, float* output,
    std::uint32_t rows, std::uint32_t elements, float epsilon,
    void* stream) noexcept;
[[nodiscard]] Status gelu_tanh_in_place(float* values,
                                        std::uint32_t elements,
                                        void* stream) noexcept;
[[nodiscard]] Status gelu_exact_in_place(float* values,
                                         std::uint32_t elements,
                                         void* stream) noexcept;

// Artifact-declared vision composite primitives. Patch rows follow the
// processor's spatial-merge-major order. Segment bounds keep attention exact
// and independent between multiple images without a quadratic mask tensor.
[[nodiscard]] Status add_fp4_position_interpolation(
    float* hidden, const Fp4Block32Matrix& positions,
    const std::uint32_t* interpolation_indices,
    const float* interpolation_weights, std::uint32_t rows,
    void* stream) noexcept;
[[nodiscard]] Status vision_qkv_rope_in_place(
    float* qkv, const std::uint32_t* row_positions,
    std::uint32_t rows, std::uint32_t heads, std::uint32_t head_dim,
    float rope_theta, void* stream) noexcept;
[[nodiscard]] Status vision_segment_attention(
    const float* qkv, const std::uint32_t* segment_first,
    const std::uint32_t* segment_last, float* output, std::uint32_t rows,
    std::uint32_t heads, std::uint32_t head_dim,
    void* stream) noexcept;

// Standard Q/K RMSNorm plus Qwen-style three-axis interleaved mRoPE.
[[nodiscard]] Status gated_gqa_qk_norm_mrope_batch(
    float* q_and_gate, float* key, const float* q_norm_weight,
    const float* k_norm_weight, const std::uint32_t* positions_thw,
    std::uint32_t rows, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t mrope_section_0,
    std::uint32_t mrope_section_1, std::uint32_t mrope_section_2,
    float epsilon, float rope_theta, void* stream) noexcept;
[[nodiscard]] Status store_gqa_kv_paged_fp4_batch(
    const float* key, const float* value, const void* const* page_table,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t first_cache_position, std::uint32_t rows,
    std::uint32_t kv_heads, std::uint32_t head_dim, void* stream) noexcept;
[[nodiscard]] Status store_gqa_kv_paged_fp8_batch(
    const float* key, const float* value, const void* const* page_table,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t first_cache_position, std::uint32_t rows,
    std::uint32_t kv_heads, std::uint32_t head_dim, void* stream) noexcept;
[[nodiscard]] Status store_gqa_kv_fp16_batch(
    const float* key, const float* value, void* fp16_keys,
    void* fp16_values, std::uint32_t rows, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* stream) noexcept;

[[nodiscard]] Status silu_product(const float* gate, const float* up,
                                  float* output, std::uint32_t elements,
                                  void* stream) noexcept;
[[nodiscard]] Status relu2_in_place(float* values, std::uint32_t elements,
                                    void* stream) noexcept;
[[nodiscard]] Status deepseek_swiglu_product(
    const float* gate, const float* up, float* output,
    std::uint32_t elements, float limit, bool bf16_output,
    void* stream) noexcept;
[[nodiscard]] Status sigmoid_scale_in_place(float* values,
                                            const float* gate,
                                            std::uint32_t elements,
                                            void* stream) noexcept;
[[nodiscard]] Status sigmoid_product_in_place(float* values,
                                              const float* gate,
                                              std::uint32_t elements,
                                              void* stream) noexcept;
[[nodiscard]] Status scaled_tanh_in_place(float* values,
                                          std::uint32_t elements,
                                          float multiplier, float softcap,
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

// Sigmoid router with selection-only expert bias. Returned weights are the
// unbiased sigmoid scores normalized over the selected experts, matching the
// provider-neutral router.sigmoid-bias.topk.v1 contract.
[[nodiscard]] Status sigmoid_bias_router_topk_batch(
    const float* input, const float* router_weights,
    const float* expert_bias, std::uint32_t rows, std::uint32_t hidden,
    std::uint32_t experts, std::uint32_t top_k, float normalization_epsilon,
    float routed_scaling_factor, float* logits, float* topk_scores,
    std::uint32_t* topk_indices, void* stream) noexcept;

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
[[nodiscard]] Status deepseek_router_hash_rows(
    const float* input, const std::uint16_t* router_weights,
    const std::int64_t* token_experts, const std::uint32_t* token_ids,
    std::uint32_t rows, float* logits, float* topk_scores,
    std::uint32_t* topk_indices, float route_scale, void* stream) noexcept;
[[nodiscard]] Status deepseek_router_learned_rows(
    const float* input, const std::uint16_t* router_weights,
    const float* selection_bias, std::uint32_t rows, float* logits,
    float* topk_scores, std::uint32_t* topk_indices, float route_scale,
    void* stream) noexcept;

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

// Standard (non-zero-centered) per-head Q/K RMSNorm, full rotary embedding,
// and GQA cache/update used by block.full-attention.gqa.qk-norm.v1.
[[nodiscard]] Status gqa_qkv_rope_cache(
    float* query, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, float* key_cache,
    float* value_cache, std::uint32_t position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* stream) noexcept;

[[nodiscard]] Status gqa_attention_decode(
    const float* query, const float* key_cache, const float* value_cache,
    float* output, std::uint32_t context_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* stream) noexcept;

struct CausalShortConvLaunch final {
  const float* projected_bcx{};  // [rows, 3 * hidden]: B, C, x
  const float* weights{};        // [hidden, kernel]
  float* state{};                // [rows, hidden, kernel]
  float* output{};               // [rows, hidden]
  std::uint32_t rows{};
  std::uint32_t hidden{};
  std::uint32_t kernel{};
  void* stream{};
};

// Exact one-token depthwise causal update for
// block.causal-short-conv.gated.v1: conv(B*x) * C.
[[nodiscard]] Status causal_short_conv_decode(
    const CausalShortConvLaunch& launch) noexcept;

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

// Output-gated GQA with a compact FP4-E2M1/UE8M0 block-32 paged cache. Each
// page stores all full-attention layers as K records followed by V records;
// every [token, kv-head] record contains packed nibbles then scale bytes.
[[nodiscard]] Status gated_gqa_qkv_rope_cache_paged_fp4(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* stream) noexcept;

// Variant for shifted causal streams such as MTP. The physical cache index
// is stream-local while rotary_position remains the target-model position.
[[nodiscard]] Status gated_gqa_qkv_rope_cache_paged_fp4_at(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t cache_position, std::uint32_t rotary_position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* stream) noexcept;

// Batched form for one contiguous causal stream. Page addresses come from the
// request page table, so a chunk may cross physical KV page boundaries.
[[nodiscard]] Status gated_gqa_qkv_rope_cache_paged_fp4_batch(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t first_rotary_position, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* stream) noexcept;

// Output-gated GQA with E4M3 K/V values and one FP16 dynamic scale per
// [token, kv-head, K-or-V] record. The cache remains paged and token-major;
// attention dequantizes to BF16 and accumulates Tensor Core products in FP32.
[[nodiscard]] Status gated_gqa_qkv_rope_cache_paged_fp8_at(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t cache_position, std::uint32_t rotary_position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* stream) noexcept;

[[nodiscard]] Status gated_gqa_qkv_rope_cache_paged_fp8_batch(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t first_rotary_position, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* stream) noexcept;

struct PagedFp4GatedGqaAttentionLaunch final {
  const float* q_and_gate{};
  const void* const* page_table{};
  float* output{};
  float* partial_maxima{};   // [maximum_splits, query_heads]
  float* partial_sums{};     // [maximum_splits, query_heads]
  float* partial_outputs{};  // [maximum_splits, query_heads, head_dim]
  std::uint32_t context_tokens{};
  std::uint32_t full_attention_layer{};
  std::uint32_t page_tokens{};
  std::uint32_t query_heads{};
  std::uint32_t kv_heads{};
  std::uint32_t head_dim{};
  std::uint32_t split_tokens{};
  std::uint32_t maximum_splits{};
  void* stream{};
};

// Split-K Flash-Decoding: one producer block owns a KV-head/context slice and
// reuses each decoded K/V value across every grouped query head. A second
// kernel combines partial online-softmax states without approximation.
[[nodiscard]] Status gated_gqa_attention_decode_paged_fp4(
    const PagedFp4GatedGqaAttentionLaunch& launch) noexcept;
// Tensor Core Flash-Decoding variant. GQA heads sharing one KV head are
// evaluated as matrix rows while retaining exact context and top-k behavior.
[[nodiscard]] Status gated_gqa_attention_decode_paged_fp4_tensor_core(
    const PagedFp4GatedGqaAttentionLaunch& launch) noexcept;

using PagedFp8GatedGqaAttentionLaunch = PagedFp4GatedGqaAttentionLaunch;

[[nodiscard]] Status gated_gqa_attention_decode_paged_fp8_tensor_core(
    const PagedFp8GatedGqaAttentionLaunch& launch) noexcept;

struct PagedFp4GatedGqaPrefillLaunch final {
  const float* q_and_gate{};  // [rows, 2 * query_heads * head_dim]
  const void* const* page_table{};
  float* output{};            // [rows, query_heads * head_dim]
  float* partial_maxima{};    // [rows, maximum_splits, query_heads]
  float* partial_sums{};      // [rows, maximum_splits, query_heads]
  float* partial_outputs{};   // [rows, maximum_splits, query_heads, head_dim]
  std::uint32_t first_context_tokens{};
  std::uint32_t rows{};
  std::uint32_t full_attention_layer{};
  std::uint32_t page_tokens{};
  std::uint32_t query_heads{};
  std::uint32_t kv_heads{};
  std::uint32_t head_dim{};
  std::uint32_t split_tokens{};
  std::uint32_t maximum_splits{};
  void* stream{};
};

// Scratch for exact staged prefill attention. Q/K/V and probabilities are
// BF16, scores and online-softmax state are FP32. Sizes are explicit so a
// provider can budget the temporary tier without hidden CUDA allocations.
struct PagedFp4GatedGqaStagedPrefillWorkspace final {
  void* queries{};
  std::size_t query_bytes{};
  void* keys{};
  std::size_t key_bytes{};
  void* values{};
  std::size_t value_bytes{};
  float* scores{};
  std::size_t score_bytes{};
  void* probabilities{};
  std::size_t probability_bytes{};
  float* accumulator{};
  std::size_t accumulator_bytes{};
  float* maxima{};
  std::size_t maxima_bytes{};
  float* sums{};
  std::size_t sum_bytes{};
  std::uint32_t split_tokens{};
};

using PagedFp8GatedGqaPrefillLaunch = PagedFp4GatedGqaPrefillLaunch;
using PagedFp8GatedGqaStagedPrefillWorkspace =
    PagedFp4GatedGqaStagedPrefillWorkspace;

// Exact causal microbatch attention. Positions times grouped query heads must
// fit one 16-row WMMA tile so packed K/V is shared across both dimensions.
[[nodiscard]] Status gated_gqa_attention_microbatch_paged_fp4_tensor_core(
    const PagedFp4GatedGqaPrefillLaunch& launch) noexcept;

[[nodiscard]] Status gated_gqa_attention_microbatch_paged_fp8_tensor_core(
    const PagedFp8GatedGqaPrefillLaunch& launch) noexcept;

// Exact causal attention for a contiguous prefill chunk. One block shares
// every decoded K/V record between up to eight adjacent query rows.
[[nodiscard]] Status gated_gqa_attention_prefill_paged_fp4(
    const PagedFp4GatedGqaPrefillLaunch& launch) noexcept;

// Exact online-softmax prefill backed by BF16 Tensor Core GEMMs. Packed K/V
// is decoded once per split and shared by every query row/head in its GQA
// group; no score, token, or top-k approximation is applied.
[[nodiscard]] Status gated_gqa_attention_staged_prefill_paged_fp4(
    const PagedFp4GatedGqaPrefillLaunch& launch,
    const PagedFp4GatedGqaStagedPrefillWorkspace& workspace) noexcept;

[[nodiscard]] Status gated_gqa_attention_staged_prefill_paged_fp8(
    const PagedFp8GatedGqaPrefillLaunch& launch,
    const PagedFp8GatedGqaStagedPrefillWorkspace& workspace) noexcept;

// Output-gated GQA backed by an authoritative FP16 host cache. K/V remains
// token-major in pinned host memory; the staged provider copies one bounded
// token tile at a time and transposes it for the batched Tensor Core GEMMs.
// retained_first_token may select an exact suffix for draft-only scoring. A
// zero value evaluates the complete causal prefix.
[[nodiscard]] Status gated_gqa_qkv_rope_fp16_batch(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight,
    void* fp16_keys, void* fp16_values, std::uint32_t first_rotary_position,
    std::uint32_t rows, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* stream) noexcept;
[[nodiscard]] Status standard_gqa_qkv_rope_fp16_batch(
    float* query, float* key, const float* value,
    void* fp16_keys, void* fp16_values, std::uint32_t first_rotary_position,
    std::uint32_t rows, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, void* stream) noexcept;
// Weightless per-head Q/K RMSNorm with an artifact-declared query scale.
// RoPE is optional so the same operation ABI covers positional and NoPE
// attention layers without a model-family branch.
[[nodiscard]] Status normalized_gqa_qkv_fp16_batch(
    float* query, float* key, const float* value,
    void* fp16_keys, void* fp16_values,
    std::uint32_t first_rotary_position, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float query_scale, float rope_theta, bool apply_rope,
    void* stream) noexcept;

// Publishes exact projected K/V rows without a positional transform. Query
// rows remain in FP32 for the attention provider. This is a distinct ABI from
// the RoPE path because position encoding is model mathematics, not a launch
// policy that the runtime may infer.
[[nodiscard]] Status standard_gqa_kv_fp16_batch(
    const float* key, const float* value,
    void* fp16_keys, void* fp16_values, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* stream) noexcept;

// Encodes already RoPE-transformed FP16 K/V rows into the compact paged
// FP4-E2M1/UE8M0 cache. This lets a lossless authoritative cache and an
// approximate full-context draft cache be populated by the same target pass.
[[nodiscard]] Status pack_gqa_kv_fp16_to_paged_fp4(
    const void* fp16_keys, const void* fp16_values,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t rows, std::uint32_t kv_heads, std::uint32_t head_dim,
    void* stream) noexcept;

// Lossless FP16 page publication. Source rows have already passed the exact
// Q/K normalization and rotary transform; no quantization is performed.
[[nodiscard]] Status store_gqa_kv_fp16_to_paged(
    const void* fp16_keys, const void* fp16_values,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t rows, std::uint32_t kv_heads, std::uint32_t head_dim,
    void* stream) noexcept;

struct HostFp16GatedGqaAttentionLaunch final {
  const float* q_and_gate{};
  const void* host_keys{};       // [cache_capacity, kv_heads, head_dim]
  const void* host_values{};     // [cache_capacity, kv_heads, head_dim]
  float* output{};
  std::uint32_t cache_capacity{};
  std::uint32_t first_context_tokens{};
  std::uint32_t retained_first_token{};
  std::uint32_t rows{};
  std::uint32_t query_heads{};
  std::uint32_t kv_heads{};
  std::uint32_t head_dim{};
  void* stream{};
  // Optional request-owned paged host source. When present, these replace
  // host_keys/host_values and let exact FP16 KV grow without relocating a
  // contiguous cache or tying it to an execution slot.
  const void* const* host_key_pages{};
  const void* const* host_value_pages{};
  std::uint32_t host_page_tokens{};
  std::uint32_t host_page_count{};
  bool output_gated{true};
};

struct DeviceFp16GatedGqaAttentionLaunch final {
  const float* q_and_gate{};
  const void* device_keys{};     // [cache_capacity, kv_heads, head_dim]
  const void* device_values{};   // [cache_capacity, kv_heads, head_dim]
  float* output{};
  std::uint32_t cache_capacity{};
  std::uint32_t first_context_tokens{};
  std::uint32_t retained_first_token{};
  std::uint32_t rows{};
  std::uint32_t query_heads{};
  std::uint32_t kv_heads{};
  std::uint32_t head_dim{};
  void* stream{};
  // Optional device-resident table of device page bases. When present, it
  // replaces the contiguous pointers and keeps exact FP16 KV progressively
  // allocated without copying page contents through the host.
  const void* const* device_pages{};
  std::uint32_t device_page_layer{};
  std::uint32_t device_page_tokens{};
  std::uint32_t device_page_count{};
  bool output_gated{true};
};

struct HostFp16GatedGqaAttentionWorkspace final {
  void* queries{};
  std::size_t query_bytes{};
  void* raw_keys{};
  std::size_t raw_key_bytes{};
  void* raw_values{};
  std::size_t raw_value_bytes{};
  void* keys{};
  std::size_t key_bytes{};
  void* values{};
  std::size_t value_bytes{};
  float* scores{};
  std::size_t score_bytes{};
  void* probabilities{};
  std::size_t probability_bytes{};
  float* accumulator{};
  std::size_t accumulator_bytes{};
  float* maxima{};
  std::size_t maxima_bytes{};
  float* sums{};
  std::size_t sum_bytes{};
  std::uint32_t split_tokens{};
};

[[nodiscard]] Status gated_gqa_attention_staged_host_fp16(
    const HostFp16GatedGqaAttentionLaunch& launch,
    const HostFp16GatedGqaAttentionWorkspace& workspace) noexcept;

// Device-source variant used while a layer-major prefill owns one complete
// layer cache in VRAM. It preserves exact FP16 KV while avoiding repeated
// host-to-device reads of the growing prefix.
[[nodiscard]] Status gated_gqa_attention_staged_device_fp16(
    const DeviceFp16GatedGqaAttentionLaunch& launch,
    const HostFp16GatedGqaAttentionWorkspace& workspace) noexcept;

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

struct SplitGatedDeltaLaunch final {
  const float* projected_qkv{};  // [2*key_dim + value_dim]: Q, K, V
  const float* projected_z{};    // [value_heads, value_head_dim]
  const float* projected_b{};    // [value_heads]
  const float* projected_a{};    // [value_heads]
  const float* conv_weights{};   // [2*key_dim + value_dim, kernel]
  const float* dt_bias{};        // [value_heads]
  const float* a_log{};          // [value_heads]
  const float* norm_weight{};    // [value_head_dim]
  float* conv_state{};           // [conv_dim, kernel]
  float* recurrent_state{};      // [value_heads, key_head_dim, value_head_dim]
  float* conv_output{};          // [conv_dim] workspace
  float* output{};               // [value_dim]
  std::uint32_t key_heads{};
  std::uint32_t value_heads{};
  std::uint32_t key_head_dim{};
  std::uint32_t value_head_dim{};
  std::uint32_t conv_kernel{};
  float epsilon{};
  void* stream{};
};

// Recurrent Gated DeltaNet with independently projected QKV, Z, beta and
// decay inputs. This capability is geometry-driven and is not tied to a model
// family or tensor path.
[[nodiscard]] Status split_gated_delta_decode(
    const SplitGatedDeltaLaunch& launch) noexcept;

struct SplitGatedDeltaPrefillLaunch final {
  const float* projected_qkv{};  // [rows, 2*key_dim + value_dim]
  const float* projected_z{};    // [rows, value_heads, value_head_dim]
  const float* projected_b{};    // [rows, value_heads]
  const float* projected_a{};    // [rows, value_heads]
  const float* conv_weights{};
  const float* dt_bias{};
  const float* a_log{};
  const float* norm_weight{};
  float* conv_state{};
  float* recurrent_state{};
  float* conv_output{};          // [rows, conv_dim]
  float* output{};               // [rows, value_dim]
  std::uint32_t rows{};
  std::uint32_t key_heads{};
  std::uint32_t value_heads{};
  std::uint32_t key_head_dim{};
  std::uint32_t value_head_dim{};
  std::uint32_t conv_kernel{};
  float epsilon{};
  void* stream{};
};

// Advances one request's causal convolution and recurrent state for a whole
// chunk in two launches instead of invoking the decode kernels per token.
[[nodiscard]] Status split_gated_delta_prefill(
    const SplitGatedDeltaPrefillLaunch& launch) noexcept;

struct Mamba2BatchLaunch final {
  const float* projected{};       // [rows, intermediate + conv + heads]
  const float* conv_weights{};    // [conv_size, 1, conv_kernel]
  const float* conv_bias{};       // [conv_size]
  const float* dt_bias{};         // [heads]
  const float* a_log{};           // [heads]
  const float* skip{};            // [heads]
  const float* norm_weight{};     // [intermediate]
  float* conv_state{};            // [conv_size, conv_kernel]
  float* recurrent_state{};       // [heads, head_dim, state_size]
  float* conv_output{};           // [rows, conv_size]
  float* output{};                // [rows, intermediate]
  std::uint32_t rows{};
  std::uint32_t heads{};
  std::uint32_t head_dim{};
  std::uint32_t state_size{};
  std::uint32_t groups{};
  std::uint32_t conv_kernel{};
  float epsilon{};
  float time_step_min{};
  void* stream{};
};

[[nodiscard]] Status mamba2_forward(
    const Mamba2BatchLaunch& launch) noexcept;

[[nodiscard]] Status argmax(const float* values, std::uint32_t count,
                            std::uint32_t* output, void* stream) noexcept;
[[nodiscard]] Status argmax_batch(const float* values, std::uint32_t count,
                                  std::uint32_t batch,
                                  std::uint32_t* output,
                                  void* stream) noexcept;

// Deterministic descending top-k over one logit row. NaNs are excluded and
// equal logits are ordered by the lower token id, matching host sampling.
[[nodiscard]] Status topk_logits(const float* values, std::uint32_t count,
                                 std::uint32_t top_k,
                                 float* output_values,
                                 std::uint32_t* output_indices,
                                 void* stream) noexcept;

}  // namespace expert::runtime::cuda
