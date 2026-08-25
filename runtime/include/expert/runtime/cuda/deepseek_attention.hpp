#pragma once

#include "expert/runtime/cuda/deepseek_csa.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"

#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

struct DeepSeekAttentionStateResult;
struct DeepSeekAttentionStateSize;
struct DeepSeekAttentionPairWorkspaceResult;
struct DeepSeekAttentionPairWorkspaceSize;
struct DeepSeekAttentionBatchWorkspaceResult;
struct DeepSeekAttentionBatchWorkspaceSize;

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
  // Ratio-four compression advances an overlapping recurrent window whenever
  // a group closes. A speculative row that closes such a group cannot be
  // rolled back by overwriting only its explicit KV slot. These helpers expose
  // the minimal device checkpoint required for that boundary; other ratios
  // return zero and remain position-logical.
  [[nodiscard]] std::uint64_t speculative_checkpoint_bytes() const noexcept;
  [[nodiscard]] Status checkpoint_speculative_state(
      void* destination, void* stream) const noexcept;
  [[nodiscard]] Status restore_speculative_state(
      const void* source, void* stream) noexcept;
  // Retained-session checkpoints live in pinned host RAM. Explicit KV slots
  // are position-addressed and are overwritten by replay; only the ratio-four
  // recurrent compressor boundary must be copied to make rewind exact.
  [[nodiscard]] std::uint64_t retention_checkpoint_bytes() const noexcept {
    return speculative_checkpoint_bytes();
  }
  [[nodiscard]] Status checkpoint_retention_state(
      void* host_destination, void* stream) const noexcept;
  [[nodiscard]] Status restore_retention_state(
      const void* host_source, void* stream) noexcept;

 private:
  friend DeepSeekAttentionStateResult create_deepseek_attention_state(
      std::uint32_t, std::uint32_t) noexcept;
  friend DeepSeekAttentionStateSize deepseek_attention_state_size(
      std::uint32_t, std::uint32_t) noexcept;
  friend Status deepseek_attention_decode(const struct DeepSeekAttentionLaunch&) noexcept;
  friend Status deepseek_attention_decode_pair(
      const struct DeepSeekAttentionPairLaunch&) noexcept;
  friend Status deepseek_attention_decode_batch(
      const struct DeepSeekAttentionBatchLaunch&) noexcept;
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

// Request-private transient storage for two causally adjacent target rows.
// Persistent KV, CSA, and compressor state remains owned by the per-layer
// DeepSeekAttentionState; this allocation is reused across all 43 layers.
class DeepSeekAttentionPairWorkspace final {
 public:
  ~DeepSeekAttentionPairWorkspace();
  DeepSeekAttentionPairWorkspace(const DeepSeekAttentionPairWorkspace&) = delete;
  DeepSeekAttentionPairWorkspace& operator=(
      const DeepSeekAttentionPairWorkspace&) = delete;

  [[nodiscard]] std::uint64_t bytes() const noexcept {
    return allocation_bytes_;
  }
  [[nodiscard]] std::uint32_t max_context_tokens() const noexcept {
    return max_context_;
  }

 private:
  friend DeepSeekAttentionPairWorkspaceResult
  create_deepseek_attention_pair_workspace(std::uint32_t) noexcept;
  friend DeepSeekAttentionPairWorkspaceSize
  deepseek_attention_pair_workspace_size(std::uint32_t) noexcept;
  friend Status deepseek_attention_decode_pair(
      const struct DeepSeekAttentionPairLaunch&) noexcept;
  DeepSeekAttentionPairWorkspace(void* allocation,
                                 std::uint64_t allocation_bytes,
                                 std::uint32_t max_context) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t allocation_bytes_{};
  std::uint32_t max_context_{};
  std::uint32_t selected_per_row_{};
  std::uint32_t index_scores_per_row_{};

  std::uint16_t *query_bf16_{}, *attention_bf16_{}, *index_query_bf16_{};
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

// Request-private transient storage for a bounded causal prefill tile. Dense
// projections operate on all rows; publication, compression and sparse
// attention still advance strictly in row order.
class DeepSeekAttentionBatchWorkspace final {
 public:
  ~DeepSeekAttentionBatchWorkspace();
  DeepSeekAttentionBatchWorkspace(const DeepSeekAttentionBatchWorkspace&) =
      delete;
  DeepSeekAttentionBatchWorkspace& operator=(
      const DeepSeekAttentionBatchWorkspace&) = delete;
  [[nodiscard]] std::uint64_t bytes() const noexcept {
    return allocation_bytes_;
  }
  [[nodiscard]] std::uint32_t maximum_rows() const noexcept {
    return maximum_rows_;
  }
  [[nodiscard]] std::uint32_t max_context_tokens() const noexcept {
    return max_context_;
  }

 private:
  friend DeepSeekAttentionBatchWorkspaceResult
  create_deepseek_attention_batch_workspace(std::uint32_t,
                                             std::uint32_t) noexcept;
  friend DeepSeekAttentionBatchWorkspaceSize
  deepseek_attention_batch_workspace_size(std::uint32_t,
                                           std::uint32_t) noexcept;
  friend Status deepseek_attention_decode_batch(
      const struct DeepSeekAttentionBatchLaunch&) noexcept;
  DeepSeekAttentionBatchWorkspace(void* allocation,
                                  std::uint64_t allocation_bytes,
                                  std::uint32_t max_context,
                                  std::uint32_t maximum_rows) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t allocation_bytes_{};
  std::uint32_t max_context_{};
  std::uint32_t maximum_rows_{};
  std::uint32_t selected_per_row_{};
  std::uint32_t index_scores_per_row_{};
  std::uint16_t *query_bf16_{}, *attention_bf16_{}, *index_query_bf16_{};
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
  // Reused serially for the largest artifact-declared dense projection.
  float* decoded_dense_matrix_{};
};

struct DeepSeekAttentionBatchWorkspaceResult final {
  Status status;
  std::shared_ptr<DeepSeekAttentionBatchWorkspace> workspace;
};

struct DeepSeekAttentionBatchWorkspaceSize final {
  Status status;
  std::uint64_t bytes{};
};

[[nodiscard]] DeepSeekAttentionBatchWorkspaceSize
deepseek_attention_batch_workspace_size(
    std::uint32_t max_context_tokens,
    std::uint32_t maximum_rows) noexcept;
[[nodiscard]] DeepSeekAttentionBatchWorkspaceResult
create_deepseek_attention_batch_workspace(
    std::uint32_t max_context_tokens,
    std::uint32_t maximum_rows) noexcept;

struct DeepSeekAttentionPairWorkspaceResult final {
  Status status;
  std::shared_ptr<DeepSeekAttentionPairWorkspace> workspace;
};

struct DeepSeekAttentionPairWorkspaceSize final {
  Status status;
  std::uint64_t bytes{};
};

[[nodiscard]] DeepSeekAttentionPairWorkspaceSize
deepseek_attention_pair_workspace_size(
    std::uint32_t max_context_tokens) noexcept;

[[nodiscard]] DeepSeekAttentionPairWorkspaceResult
create_deepseek_attention_pair_workspace(
    std::uint32_t max_context_tokens) noexcept;

struct DeepSeekAttentionStateResult final {
  Status status;
  std::shared_ptr<DeepSeekAttentionState> state;
};

struct DeepSeekAttentionStateSize final {
  Status status;
  std::uint64_t bytes{};
};

// Computes the exact CUDA allocation footprint before any allocation occurs.
[[nodiscard]] DeepSeekAttentionStateSize deepseek_attention_state_size(
    std::uint32_t compress_ratio,
    std::uint32_t max_context_tokens) noexcept;

// Minimal rollback storage for a speculative row at this compression ratio.
// Only ratio four has recurrent overlap state that cannot be recovered by
// replaying the same explicit position.
[[nodiscard]] std::uint64_t deepseek_attention_speculative_checkpoint_size(
    std::uint32_t compress_ratio) noexcept;

// Allocates all per-request sliding-window/CSA cache and workspace up front.
// Ratio zero is the checkpoint's pure sliding-window mode. No allocation occurs
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
  struct ProfileEvents {
    void* hca_pre_norm_stop{};
    void* projection_stop{};
    void* sparse_attention_stop{};
    void* output_projection_stop{};
  } const* profile_events{};
};

// One-token attention sublayer: HCA pre, normalization, Q/KV + CSA/index,
// sparse attention, grouped output projection, and HCA post.
[[nodiscard]] Status deepseek_attention_decode(
    const DeepSeekAttentionLaunch& launch) noexcept;

struct DeepSeekAttentionPairLaunch final {
  const DeepSeekAttentionBinding* weights{};
  DeepSeekAttentionState* state{};
  DeepSeekAttentionPairWorkspace* workspace{};
  const float* streams[2]{};          // each [4, 4096]
  float* updated_streams[2]{};        // each [4, 4096]
  const float* cosine[2]{};
  const float* sine[2]{};
  const float* compressed_cosine[2]{};
  const float* compressed_sine[2]{};
  std::uint32_t positions[2]{};
  // Optional destination for the minimal ratio-four recurrent checkpoint.
  // It is written after row zero and before any row-one state mutation.
  void* speculative_checkpoint{};
  float epsilon{1e-6F};
  std::uint32_t sinkhorn_iterations{20U};
  void* stream{};
};

// Two-token target attention with shared dense-weight reads. Projection work
// is batched, while cache publication and sparse attention remain strictly
// row-zero-before-row-one so ring-buffer and compressor semantics are exact.
[[nodiscard]] Status deepseek_attention_decode_pair(
    const DeepSeekAttentionPairLaunch& launch) noexcept;

struct DeepSeekAttentionBatchRow final {
  const float* cosine{};
  const float* sine{};
  const float* compressed_cosine{};
  const float* compressed_sine{};
  std::uint32_t position{};
};

struct DeepSeekAttentionBatchLaunch final {
  const DeepSeekAttentionBinding* weights{};
  DeepSeekAttentionState* state{};
  DeepSeekAttentionBatchWorkspace* workspace{};
  const float* streams{};       // [rows, 4, 4096]
  float* updated_streams{};     // [rows, 4, 4096]
  const DeepSeekAttentionBatchRow* row_state{};  // host [rows]
  std::uint32_t rows{};
  float epsilon{1e-6F};
  std::uint32_t sinkhorn_iterations{20U};
  void* stream{};
  struct ProfileEvents {
    void* hca_pre_norm_stop{};
    void* projection_stop{};
    void* causal_attention_stop{};
    void* output_projection_stop{};
  } const* profile_events{};
};

[[nodiscard]] Status deepseek_attention_decode_batch(
    const DeepSeekAttentionBatchLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
