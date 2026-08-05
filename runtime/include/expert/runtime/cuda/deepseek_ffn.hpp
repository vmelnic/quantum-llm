#pragma once

#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"

#include <cstdint>
#include <memory>

namespace expert::runtime::cuda {

struct DeepSeekFfnStateResult;

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
  [[nodiscard]] static constexpr std::uint32_t selection_count() noexcept {
    return 7U;
  }

 private:
  friend DeepSeekFfnStateResult create_deepseek_ffn_state(
      std::uint32_t) noexcept;
  friend Status deepseek_ffn_route(const struct DeepSeekFfnRouteLaunch&) noexcept;
  friend Status deepseek_ffn_execute(
      const struct DeepSeekFfnExecuteLaunch&) noexcept;
  DeepSeekFfnState(void* allocation, std::uint64_t bytes,
                   std::uint32_t layer) noexcept;
  void map(void* base) noexcept;

  void* allocation_{};
  std::uint64_t bytes_{};
  std::uint32_t layer_{};
  float *hca_normalized_{}, *hca_mixes_{}, *collapsed_{}, *ffn_input_{};
  float *pre_{}, *post_{}, *comb_{}, *router_logits_{}, *routing_weights_{};
  std::uint32_t* expert_indices_{};
  float *routed_intermediate_{}, *routed_output_{};
  float *shared_intermediate_{}, *shared_output_{};
};

struct DeepSeekFfnStateResult final {
  Status status;
  std::shared_ptr<DeepSeekFfnState> state;
};

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

struct DeepSeekFfnExecuteLaunch final {
  const DeepSeekFfnBinding* weights{};
  DeepSeekFfnState* state{};
  const DeviceExpertEntry* directory_entries{};
  const float* streams{};   // same streams passed to route
  float* updated_streams{}; // [4, 4096]
  std::uint32_t experts_per_layer{257U};
  void* stream{};
};

// Requires an active directory pin covering state.expert_indices()[0..7).
// Executes routed top-6 plus shared expert, then applies FFN HCA post.
[[nodiscard]] Status deepseek_ffn_execute(
    const DeepSeekFfnExecuteLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
