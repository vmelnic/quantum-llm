#pragma once

#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cpu/deepseek_packed_executor.hpp"

#include <cstdint>
#include <array>
#include <memory>
#include <span>

namespace expert::runtime::cuda {

struct DeepSeekFfnStateResult;
struct DeepSeekFfnHybridWorkspaceResult;
struct DeepSeekFfnPairWorkspaceResult;
struct DeepSeekFfnBatchWorkspaceResult;
struct DeepSeekRoutePredictionStateResult;
struct DeepSeekFfnHybridPairExecuteLaunch;

[[nodiscard]] Status deepseek_ffn_execute_pair_hybrid(
    const DeepSeekFfnHybridPairExecuteLaunch& launch) noexcept;

class DeepSeekFfnState final {
 public:
  ~DeepSeekFfnState();
  DeepSeekFfnState(const DeepSeekFfnState&) = delete;
  DeepSeekFfnState& operator=(const DeepSeekFfnState&) = delete;

  [[nodiscard]] std::uint32_t layer() const noexcept { return layer_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  // Six routed experts followed by the always-resident shared expert (256).
  // The control plane pins all seven before calling deepseek_ffn_execute.
  [[nodiscard]] const std::uint32_t* expert_indices() const noexcept {
    return expert_indices_;
  }
  [[nodiscard]] const float* routing_weights() const noexcept {
    return routing_weights_;
  }
  [[nodiscard]] const float* normalized_input() const noexcept {
    return ffn_input_;
  }
  [[nodiscard]] const float* routed_selection_outputs() const noexcept {
    return routed_selection_outputs_;
  }
  [[nodiscard]] static constexpr std::uint32_t selection_count() noexcept {
    return 7U;
  }

 private:
  friend DeepSeekFfnStateResult create_deepseek_ffn_state(
      std::uint32_t) noexcept;
  friend std::uint64_t deepseek_ffn_state_size() noexcept;
  friend Status deepseek_ffn_route(const struct DeepSeekFfnRouteLaunch&) noexcept;
  friend Status deepseek_ffn_execute(
      const struct DeepSeekFfnExecuteLaunch&) noexcept;
  friend Status deepseek_ffn_execute_selections(
      const struct DeepSeekFfnSelectionExecuteLaunch&) noexcept;
  friend Status deepseek_ffn_import_selection_output(
      const struct DeepSeekFfnSelectionImportLaunch&) noexcept;
  friend Status deepseek_ffn_finalize(
      const struct DeepSeekFfnFinalizeLaunch&) noexcept;
  friend Status deepseek_ffn_execute_hybrid(
      const struct DeepSeekFfnHybridExecuteLaunch&) noexcept;
  friend Status deepseek_ffn_execute_pair_hybrid(
      const struct DeepSeekFfnHybridPairExecuteLaunch&) noexcept;
  friend Status deepseek_ffn_gather_pair_routes(
      const struct DeepSeekFfnPairRouteGather&) noexcept;
  friend Status deepseek_ffn_route_pair(
      const struct DeepSeekFfnPairRouteLaunch&) noexcept;
  friend Status deepseek_ffn_execute_pair(
      const struct DeepSeekFfnPairExecuteLaunch&) noexcept;
  DeepSeekFfnState(void* allocation, std::uint64_t bytes,
                   std::uint32_t layer) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t bytes_{};
  std::uint32_t layer_{};
  float *hca_normalized_{}, *hca_mixes_{}, *collapsed_{}, *ffn_input_{};
  float *pre_{}, *post_{}, *comb_{}, *router_logits_{}, *routing_weights_{};
  std::uint32_t* expert_indices_{};
  float *routed_intermediate_{}, *routed_selection_outputs_{},
      *routed_output_{};
  std::int8_t *routed_q_input_{}, *routed_q_intermediate_{};
  float *routed_q_input_scales_{}, *routed_q_intermediate_scales_{};
  std::uint8_t* routed_selection_mask_{};
  float *shared_intermediate_{}, *shared_output_{};
};

struct DeepSeekFfnStateResult final {
  Status status;
  std::shared_ptr<DeepSeekFfnState> state;
};

[[nodiscard]] std::uint64_t deepseek_ffn_state_size() noexcept;

[[nodiscard]] DeepSeekFfnStateResult create_deepseek_ffn_state(
    std::uint32_t layer) noexcept;

// Request-private scratch for an advisory cross-layer route prediction. The
// exact router never consumes these buffers: the provider may use the copied
// indices only to begin page movement before the authoritative next-layer
// route is available.
class DeepSeekRoutePredictionState final {
 public:
  ~DeepSeekRoutePredictionState();
  DeepSeekRoutePredictionState(const DeepSeekRoutePredictionState&) = delete;
  DeepSeekRoutePredictionState& operator=(
      const DeepSeekRoutePredictionState&) = delete;

  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] const std::uint32_t* expert_indices() const noexcept {
    return expert_indices_;
  }
  [[nodiscard]] static constexpr std::uint32_t selection_count() noexcept {
    return 6U;
  }

 private:
  friend std::uint64_t deepseek_route_prediction_state_size() noexcept;
  friend DeepSeekRoutePredictionStateResult
  create_deepseek_route_prediction_state() noexcept;
  friend Status deepseek_predict_route(
      const struct DeepSeekRoutePredictionLaunch&) noexcept;
  DeepSeekRoutePredictionState(void* allocation,
                               std::uint64_t bytes) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t bytes_{};
  float* router_logits_{};
  float* routing_weights_{};
  std::uint32_t* expert_indices_{};
};

struct DeepSeekRoutePredictionStateResult final {
  Status status;
  std::shared_ptr<DeepSeekRoutePredictionState> state;
};

[[nodiscard]] std::uint64_t deepseek_route_prediction_state_size() noexcept;

[[nodiscard]] DeepSeekRoutePredictionStateResult
create_deepseek_route_prediction_state() noexcept;

struct DeepSeekRoutePredictionLaunch final {
  const DeepSeekFfnBinding* target_weights{};
  DeepSeekRoutePredictionState* state{};
  // Normalized gate input from the preceding layer. This is advisory only;
  // the target layer recomputes and executes its authoritative route.
  const float* preceding_gate_input{};
  std::uint32_t token_id{};
  void* stream{};
};

[[nodiscard]] Status deepseek_predict_route(
    const DeepSeekRoutePredictionLaunch& launch) noexcept;

struct DeepSeekFfnRouteLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* state{};
  const float* streams{};  // [4, 4096], attention-updated streams
  std::uint32_t token_id{};
  float epsilon{1e-6F};
  std::uint32_t sinkhorn_iterations{20U};
  void* stream{};
};

// Produces device-resident routing weights/indices. The call remains
// asynchronous; directory planning is the synchronization/control boundary.
[[nodiscard]] Status deepseek_ffn_route(
    const DeepSeekFfnRouteLaunch& launch) noexcept;

struct DeepSeekFfnExecutionTiming final {
  float routed_gpu_ms{};
  std::uint32_t routed_selections{};
};

struct DeepSeekFfnExecuteLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* state{};
  const DeviceExpertEntry* directory_entries{};
  const float* streams{};   // same streams passed to route
  float* updated_streams{}; // [4, 4096]
  std::uint32_t experts_per_layer{257U};
  void* stream{};
  DeepSeekFfnExecutionTiming* timing{};
  struct ProfileEvents {
    void* routed_stop{};
    void* aggregate_stop{};
    void* shared_stop{};
    void* merge_stop{};
  } const* profile_events{};
};

// Requires an active directory pin covering state.expert_indices()[0..7).
// Executes routed top-6 plus shared expert, then applies FFN HCA post.
[[nodiscard]] Status deepseek_ffn_execute(
    const DeepSeekFfnExecuteLaunch& launch) noexcept;

// Executes only the selected exact top-k slots and writes their independent
// outputs into the ordinary request workspace. This allows ready expert pages
// to execute while other exact selections are still being supplied. Calls may
// arrive in any grouping, but every top-k slot must execute exactly once before
// deepseek_ffn_finalize(). The final aggregation order remains unchanged.
struct DeepSeekFfnSelectionExecuteLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* state{};
  const DeviceExpertEntry* directory_entries{};
  // Bit i selects routed top-k slot i. Bits outside selection_count() are
  // rejected.
  std::uint64_t selection_mask{};
  std::uint32_t experts_per_layer{257U};
  void* stream{};
};

[[nodiscard]] Status deepseek_ffn_execute_selections(
    const DeepSeekFfnSelectionExecuteLaunch& launch) noexcept;

// Imports one exact remotely executed selection into its ordinary top-k slot.
// The host buffer contains only the expert output activation; route weights
// remain device-resident and are applied later by deepseek_ffn_finalize().
struct DeepSeekFfnSelectionImportLaunch final {
  DeepSeekFfnState* state{};
  std::uint32_t selection_index{};
  const float* host_output{};
  std::uint64_t host_output_bytes{};
  void* stream{};
};

[[nodiscard]] Status deepseek_ffn_import_selection_output(
    const DeepSeekFfnSelectionImportLaunch& launch) noexcept;

struct DeepSeekFfnFinalizeLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* state{};
  const DeviceExpertEntry* directory_entries{};
  const float* streams{};
  float* updated_streams{};
  std::uint32_t experts_per_layer{257U};
  void* stream{};
  const DeepSeekFfnExecuteLaunch::ProfileEvents* profile_events{};
};

// Aggregates the already-computed selection outputs in stable top-k order,
// executes the always-resident shared expert, and applies the HCA post block.
[[nodiscard]] Status deepseek_ffn_finalize(
    const DeepSeekFfnFinalizeLaunch& launch) noexcept;

// Request-private two-row workspace. HCA/router state and compute-ready arrays
// are produced directly here so dense and expert kernels can read each weight
// set once for both verifier rows. The two ordinary FFN states retain only
// their bounded ownership/identity role in this path.
class DeepSeekFfnPairWorkspace final {
 public:
  ~DeepSeekFfnPairWorkspace();
  DeepSeekFfnPairWorkspace(const DeepSeekFfnPairWorkspace&) = delete;
  DeepSeekFfnPairWorkspace& operator=(const DeepSeekFfnPairWorkspace&) = delete;
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] const std::uint32_t* expert_indices() const noexcept {
    return expert_indices_;
  }
  [[nodiscard]] const float* normalized_input() const noexcept {
    return ffn_input_;
  }
  [[nodiscard]] static constexpr std::uint32_t selection_count() noexcept {
    return 14U;
  }

 private:
  friend DeepSeekFfnPairWorkspaceResult
  create_deepseek_ffn_pair_workspace() noexcept;
  friend std::uint64_t deepseek_ffn_pair_workspace_size() noexcept;
  friend Status deepseek_ffn_gather_pair_routes(
      const struct DeepSeekFfnPairRouteGather&) noexcept;
  friend Status deepseek_ffn_route_pair(
      const struct DeepSeekFfnPairRouteLaunch&) noexcept;
  friend Status deepseek_ffn_execute_pair(
      const struct DeepSeekFfnPairExecuteLaunch&) noexcept;
  friend Status deepseek_ffn_import_pair_selection_output(
      const struct DeepSeekFfnPairSelectionImportLaunch&) noexcept;
  friend Status deepseek_ffn_finalize_pair(
      const struct DeepSeekFfnPairExecuteLaunch&) noexcept;
  friend Status deepseek_ffn_execute_pair_hybrid(
      const DeepSeekFfnHybridPairExecuteLaunch&) noexcept;
  DeepSeekFfnPairWorkspace(void* allocation, std::uint64_t bytes) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t bytes_{};
  float *hca_normalized_{}, *hca_mixes_{}, *collapsed_{}, *pre_{}, *post_{},
      *comb_{}, *ffn_input_{}, *router_logits_{}, *routing_weights_{},
      *routed_intermediate_{},
      *routed_selection_outputs_{}, *routed_output_{},
      *shared_intermediate_{}, *shared_output_{};
  std::uint32_t *expert_indices_{}, *routed_indices_{}, *shared_indices_{};
  float* shared_weights_{};
  std::int8_t *routed_q_input_{}, *routed_q_intermediate_{};
  float *routed_q_input_scales_{}, *routed_q_intermediate_scales_{};
};

class CudaCompactExpertAllocation;

// Transient device tile used by exact causal prefill. Routing metadata and
// independent selection outputs remain in stable row/top-k order; expert pages
// are executed separately in expert-major batches.
class DeepSeekFfnBatchWorkspace final {
 public:
  ~DeepSeekFfnBatchWorkspace();
  DeepSeekFfnBatchWorkspace(const DeepSeekFfnBatchWorkspace&) = delete;
  DeepSeekFfnBatchWorkspace& operator=(const DeepSeekFfnBatchWorkspace&) =
      delete;
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint32_t maximum_rows() const noexcept {
    return maximum_rows_;
  }
  [[nodiscard]] const float* normalized_input() const noexcept {
    return ffn_input_;
  }
  [[nodiscard]] float* normalized_input() noexcept { return ffn_input_; }
  [[nodiscard]] const float* routing_weights() const noexcept {
    return routing_weights_;
  }
  [[nodiscard]] float* routing_weights() noexcept { return routing_weights_; }
  [[nodiscard]] const std::uint32_t* expert_indices() const noexcept {
    return expert_indices_;
  }
  [[nodiscard]] std::uint32_t* expert_indices() noexcept {
    return expert_indices_;
  }
  [[nodiscard]] const float* post_control() const noexcept { return post_; }
  [[nodiscard]] float* post_control() noexcept { return post_; }
  [[nodiscard]] const float* combination_control() const noexcept {
    return comb_;
  }
  [[nodiscard]] float* combination_control() noexcept { return comb_; }
  [[nodiscard]] float* selection_outputs() noexcept {
    return routed_selection_outputs_;
  }
  [[nodiscard]] const float* selection_outputs() const noexcept {
    return routed_selection_outputs_;
  }
  [[nodiscard]] float* expert_inputs() noexcept { return direct_inputs_; }
  [[nodiscard]] float* expert_outputs() noexcept { return direct_outputs_; }

 private:
  friend DeepSeekFfnBatchWorkspaceResult
  create_deepseek_ffn_batch_workspace(std::uint32_t) noexcept;
  friend std::uint64_t deepseek_ffn_batch_workspace_size(
      std::uint32_t) noexcept;
  friend Status deepseek_ffn_route_batch(
      const struct DeepSeekFfnBatchRouteLaunch&) noexcept;
  friend Status deepseek_ffn_execute_packed_batch(
      const struct DeepSeekFfnPackedBatchLaunch&) noexcept;
  friend Status deepseek_ffn_finalize_batch(
      const struct DeepSeekFfnBatchFinalizeLaunch&) noexcept;
  DeepSeekFfnBatchWorkspace(void* allocation, std::uint64_t bytes,
                            std::uint32_t maximum_rows) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t bytes_{};
  std::uint32_t maximum_rows_{};
  float *hca_normalized_{}, *hca_mixes_{}, *collapsed_{}, *pre_{}, *post_{},
      *comb_{}, *ffn_input_{}, *router_logits_{}, *routing_weights_{},
      *routed_selection_outputs_{}, *routed_output_{}, *shared_intermediate_{},
      *shared_output_{}, *direct_inputs_{}, *direct_gate_{}, *direct_up_{},
      *direct_intermediate_{}, *direct_outputs_{};
  std::uint32_t *expert_indices_{}, *shared_indices_{};
  std::int8_t *shared_q_input_{}, *shared_q_intermediate_{},
      *direct_q_input_{}, *direct_q_intermediate_{};
  float *shared_q_input_scales_{}, *shared_q_intermediate_scales_{},
      *direct_q_input_scales_{}, *direct_q_intermediate_scales_{};
};

struct DeepSeekFfnBatchWorkspaceResult final {
  Status status;
  std::shared_ptr<DeepSeekFfnBatchWorkspace> workspace;
};

[[nodiscard]] std::uint64_t deepseek_ffn_batch_workspace_size(
    std::uint32_t maximum_rows) noexcept;
[[nodiscard]] DeepSeekFfnBatchWorkspaceResult
create_deepseek_ffn_batch_workspace(std::uint32_t maximum_rows) noexcept;

struct DeepSeekFfnBatchRouteLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* identity_state{};
  DeepSeekFfnBatchWorkspace* workspace{};
  const float* streams{};              // [rows, 4, 4096]
  const std::uint32_t* token_ids{};     // host [rows]
  std::uint32_t rows{};
  float epsilon{1e-6F};
  std::uint32_t sinkhorn_iterations{20U};
  void* stream{};
};

[[nodiscard]] Status deepseek_ffn_route_batch(
    const DeepSeekFfnBatchRouteLaunch& launch) noexcept;

struct DeepSeekFfnPackedBatchLaunch final {
  const CudaCompactExpertAllocation* expert{};
  DeepSeekFfnBatchWorkspace* workspace{};
  const float* input{};   // device [rows, 4096]
  float* output{};        // device [rows, 4096]
  std::uint32_t rows{};
  float swiglu_limit{10.0F};
  bool bf16_intermediate{true};
  void* stream{};
};

[[nodiscard]] Status deepseek_ffn_execute_packed_batch(
    const DeepSeekFfnPackedBatchLaunch& launch) noexcept;

struct DeepSeekFfnBatchFinalizeLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* identity_state{};
  DeepSeekFfnBatchWorkspace* workspace{};
  const DeviceExpertEntry* directory_entries{};
  const float* streams{};              // [rows, 4, 4096]
  float* updated_streams{};            // [rows, 4, 4096]
  std::uint32_t rows{};
  std::uint32_t experts_per_layer{257U};
  void* stream{};
};

[[nodiscard]] Status deepseek_ffn_finalize_batch(
    const DeepSeekFfnBatchFinalizeLaunch& launch) noexcept;

struct DeepSeekFfnPairWorkspaceResult final {
  Status status;
  std::shared_ptr<DeepSeekFfnPairWorkspace> workspace;
};

[[nodiscard]] std::uint64_t deepseek_ffn_pair_workspace_size() noexcept;
[[nodiscard]] DeepSeekFfnPairWorkspaceResult
create_deepseek_ffn_pair_workspace() noexcept;

struct DeepSeekFfnPairRouteLaunch final {
  const DeepSeekFfnBinding* weights{};
  std::array<DeepSeekFfnState*, 2U> states{};
  DeepSeekFfnPairWorkspace* workspace{};
  std::array<const float*, 2U> streams{};
  std::array<std::uint32_t, 2U> token_ids{};
  float epsilon{1e-6F};
  std::uint32_t sinkhorn_iterations{20U};
  void* stream{};
};

// Computes both exact routes with shared HCA/router weight reads and publishes
// the 14-entry directory layout directly in the pair workspace.
[[nodiscard]] Status deepseek_ffn_route_pair(
    const DeepSeekFfnPairRouteLaunch& launch) noexcept;

struct DeepSeekFfnPairRouteGather final {
  std::array<const DeepSeekFfnState*, 2U> states{};
  DeepSeekFfnPairWorkspace* workspace{};
  void* stream{};
};

// Gathers both exact route selections into one stable device span suitable for
// a single directory pin transaction. It does not synchronize the stream.
[[nodiscard]] Status deepseek_ffn_gather_pair_routes(
    const DeepSeekFfnPairRouteGather& launch) noexcept;

struct DeepSeekFfnPairExecuteLaunch final {
  const DeepSeekFfnBinding* weights{};
  std::array<DeepSeekFfnState*, 2U> states{};
  DeepSeekFfnPairWorkspace* workspace{};
  const DeviceExpertEntry* directory_entries{};
  std::array<const float*, 2U> streams{};
  std::array<float*, 2U> updated_streams{};
  std::uint32_t experts_per_layer{257U};
  void* stream{};
};

// Requires one active directory pin covering workspace.expert_indices()[0..14).
// Routed and shared experts execute as two-row batches; HCA post remains
// causal-row local because its mutable stream state is request-owned.
[[nodiscard]] Status deepseek_ffn_execute_pair(
    const DeepSeekFfnPairExecuteLaunch& launch) noexcept;

// Imports one exact externally executed pair selection. Flat selection slots
// are row-major: [row0 top-k, row1 top-k]. Route weights remain on the primary
// device and are applied by deepseek_ffn_finalize_pair().
struct DeepSeekFfnPairSelectionImportLaunch final {
  DeepSeekFfnPairWorkspace* workspace{};
  std::uint32_t selection_index{};
  const float* host_output{};
  std::uint64_t host_output_bytes{};
  void* stream{};
};

[[nodiscard]] Status deepseek_ffn_import_pair_selection_output(
    const DeepSeekFfnPairSelectionImportLaunch& launch) noexcept;

// Completes exact stable aggregation, the always-resident shared expert and
// HCA post after all 12 routed pair outputs have been imported.
[[nodiscard]] Status deepseek_ffn_finalize_pair(
    const DeepSeekFfnPairExecuteLaunch& launch) noexcept;

class DeepSeekFfnHybridWorkspace final {
 public:
  ~DeepSeekFfnHybridWorkspace();
  DeepSeekFfnHybridWorkspace(const DeepSeekFfnHybridWorkspace&) = delete;
  DeepSeekFfnHybridWorkspace& operator=(
      const DeepSeekFfnHybridWorkspace&) = delete;
  [[nodiscard]] std::uint64_t device_bytes() const noexcept {
    return device_bytes_;
  }
  [[nodiscard]] std::uint64_t pinned_host_bytes() const noexcept {
    return host_bytes_;
  }

 private:
  friend DeepSeekFfnHybridWorkspaceResult
  create_deepseek_ffn_hybrid_workspace() noexcept;
  friend Status deepseek_ffn_execute_hybrid(
      const struct DeepSeekFfnHybridExecuteLaunch&) noexcept;
  friend Status deepseek_ffn_execute_pair_hybrid(
      const DeepSeekFfnHybridPairExecuteLaunch&) noexcept;
  DeepSeekFfnHybridWorkspace(void* device_allocation,
                             std::uint64_t device_bytes,
                             void* host_allocation,
                             std::uint64_t host_bytes,
                             void* input_ready_event) noexcept;
  void map() noexcept;

  void* device_allocation_{};
  void* host_allocation_{};
  std::uint64_t device_bytes_{};
  std::uint64_t host_bytes_{};
  std::uint8_t* selection_mask_{};
  std::uint32_t* alternate_slot_by_selection_{};
  float* alternate_outputs_{};
  float* host_input_{};
  float* host_outputs_{};
  void* input_ready_event_{};
};

struct DeepSeekFfnHybridWorkspaceResult final {
  Status status;
  std::shared_ptr<DeepSeekFfnHybridWorkspace> workspace;
};

[[nodiscard]] DeepSeekFfnHybridWorkspaceResult
create_deepseek_ffn_hybrid_workspace() noexcept;

struct DeepSeekFfnHybridExecuteLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* state{};
  const DeviceExpertEntry* directory_entries{};
  const float* streams{};
  float* updated_streams{};
  DeepSeekFfnHybridWorkspace* workspace{};
  cpu::DeepSeekPackedExecutor* cpu_executor{};
  std::span<const cpu::DeepSeekPackedWorkGroup> cpu_groups;
  std::uint32_t experts_per_layer{257U};
  void* stream{};
  const DeepSeekFfnExecuteLaunch::ProfileEvents* profile_events{};
};

// Executes the unmasked selections on CUDA while the designated compact RAM
// selections run on the persistent CPU pool. CPU outputs return as a compact
// array and are merged in stable top-k order before the shared expert/HCA post.
[[nodiscard]] Status deepseek_ffn_execute_hybrid(
    const DeepSeekFfnHybridExecuteLaunch& launch) noexcept;

struct DeepSeekFfnHybridPairExecuteLaunch final {
  const DeepSeekFfnBinding* weights{};
  std::array<DeepSeekFfnState*, 2U> states{};
  DeepSeekFfnPairWorkspace* pair_workspace{};
  const DeviceExpertEntry* directory_entries{};
  std::array<const float*, 2U> streams{};
  std::array<float*, 2U> updated_streams{};
  DeepSeekFfnHybridWorkspace* hybrid_workspace{};
  cpu::DeepSeekPackedExecutor* cpu_executor{};
  std::span<const cpu::DeepSeekPackedWorkGroup> cpu_groups;
  std::uint32_t experts_per_layer{257U};
  void* stream{};
};

// Exact two-row counterpart used by MTP verification. Global selection
// indices are row-major in [0, 12); GPU and CPU partial outputs are merged
// against the original per-row route weights without changing top-k.
[[nodiscard]] Status deepseek_ffn_execute_pair_hybrid(
    const DeepSeekFfnHybridPairExecuteLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
