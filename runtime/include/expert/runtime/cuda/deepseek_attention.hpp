#pragma once

#include "expert/runtime/cuda/deepseek_csa.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"

#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

struct DeepSeekAttentionStateResult;

class DeepSeekAttentionState final {
 public:
  ~DeepSeekAttentionState();
  DeepSeekAttentionState(const DeepSeekAttentionState&) = delete;
  DeepSeekAttentionState& operator=(const DeepSeekAttentionState&) = delete;

  [[nodiscard]] std::uint32_t compress_ratio() const noexcept { return ratio_; }
  [[nodiscard]] std::uint32_t max_context_tokens() const noexcept {
    return max_context_;
  }
  [[nodiscard]] std::uint64_t bytes() const noexcept;

 private:
  friend DeepSeekAttentionStateResult create_deepseek_attention_state(
      std::uint32_t, std::uint32_t) noexcept;
  friend Status deepseek_attention_decode(const struct DeepSeekAttentionLaunch&) noexcept;
  DeepSeekAttentionState(void* allocation, std::uint64_t allocation_bytes,
                         std::uint32_t ratio, std::uint32_t max_context,
                         std::uint32_t max_compressed) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t allocation_bytes_{};
  std::uint32_t ratio_{};
  std::uint32_t max_context_{};
  std::uint32_t max_compressed_{};
  std::shared_ptr<DeepSeekCompressorState> compressor_;
  std::shared_ptr<DeepSeekCompressorState> index_compressor_;

  std::uint16_t* kv_cache_{};
  std::uint16_t* index_cache_{};
  std::uint16_t* query_bf16_{};
  std::uint16_t* attention_bf16_{};
  std::uint16_t* index_query_bf16_{};
  std::int32_t* indices_{};
  float *hca_normalized_{}, *hca_mixes_{}, *collapsed_{}, *attention_input_{};
  float *pre_{}, *post_{}, *comb_{};
  float *query_rank_{}, *query_norm_{}, *query_{}, *kv_{}, *kv_norm_{};
  float *attention_output_{}, *group_output_{}, *sublayer_{};
  float *compress_values_{}, *compress_scores_{}, *compress_pooled_{};
  float* compress_output_{};
  float *index_query_{}, *index_head_weights_{}, *index_scores_{};
  float *index_compress_values_{}, *index_compress_scores_{};
  float *index_compress_pooled_{}, *index_compress_output_{};
};

struct DeepSeekAttentionStateResult final {
  Status status;
  std::shared_ptr<DeepSeekAttentionState> state;
};

// Allocates all per-request CSA cache/workspace up front. No allocation occurs
// in deepseek_attention_decode.
[[nodiscard]] DeepSeekAttentionStateResult create_deepseek_attention_state(
    std::uint32_t compress_ratio, std::uint32_t max_context_tokens) noexcept;

struct DeepSeekAttentionLaunch final {
  const DeepSeekAttentionBinding* weights{};
  DeepSeekAttentionState* state{};
  const float* streams{};       // [4, 4096]
  float* updated_streams{};     // [4, 4096]
  const float* cosine{};        // [32], current position
  const float* sine{};          // [32], current position
  const float* compressed_cosine{};  // [32], group start; required on emit
  const float* compressed_sine{};    // [32], group start; required on emit
  std::uint32_t position{};
  float epsilon{1e-6F};
  std::uint32_t sinkhorn_iterations{20U};
  void* stream{};
};

// One-token attention sublayer: HCA pre, normalization, Q/KV + CSA/index,
// sparse attention, grouped output projection, and HCA post.
[[nodiscard]] Status deepseek_attention_decode(
    const DeepSeekAttentionLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
