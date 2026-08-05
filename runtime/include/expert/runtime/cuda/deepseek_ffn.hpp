#pragma once

#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cpu/deepseek_packed_executor.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace expert::runtime::cuda {

struct DeepSeekFfnStateResult;
struct DeepSeekFfnHybridWorkspaceResult;

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
  friend Status deepseek_ffn_execute_hybrid(
      const struct DeepSeekFfnHybridExecuteLaunch&) noexcept;
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
  float *shared_intermediate_{}, *shared_output_{};
};

struct DeepSeekFfnStateResult final {
  Status status;
  std::shared_ptr<DeepSeekFfnState> state;
};

[[nodiscard]] std::uint64_t deepseek_ffn_state_size() noexcept;

[[nodiscard]] DeepSeekFfnStateResult create_deepseek_ffn_state(
    std::uint32_t layer) noexcept;

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
};

// Requires an active directory pin covering state.expert_indices()[0..7).
// Executes routed top-6 plus shared expert, then applies FFN HCA post.
[[nodiscard]] Status deepseek_ffn_execute(
    const DeepSeekFfnExecuteLaunch& launch) noexcept;

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
  DeepSeekFfnHybridWorkspace(void* device_allocation,
                             std::uint64_t device_bytes,
                             void* host_allocation,
                             std::uint64_t host_bytes) noexcept;
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
};

// Executes the unmasked selections on CUDA while the designated compact RAM
// selections run on the persistent CPU pool. CPU outputs return as a compact
// array and are merged in stable top-k order before the shared expert/HCA post.
[[nodiscard]] Status deepseek_ffn_execute_hybrid(
    const DeepSeekFfnHybridExecuteLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
