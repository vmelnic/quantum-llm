#include "expert/runtime/cuda/deepseek_ffn.hpp"

#include "expert/runtime/cuda/deepseek_hca.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kIntermediate = 2048U;
constexpr std::uint32_t kTopK = 6U;
constexpr std::uint32_t kSharedExpert = 256U;
constexpr std::size_t kAlignment = 256U;

Status failure(cudaError_t error, const char* operation) noexcept {
  return {ErrorCode::internal,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

std::size_t align_up(std::size_t value) noexcept {
  return (value + kAlignment - 1U) & ~(kAlignment - 1U);
}

struct Arena final {
  std::byte* base{};
  std::size_t cursor{};

  template <typename T>
  T* take(std::size_t count) noexcept {
    cursor = align_up(cursor);
    auto* result = reinterpret_cast<T*>(base ? base + cursor : nullptr);
    cursor += count * sizeof(T);
    return result;
  }
};

Status check_binding(const DeepSeekFfnBinding& weights,
                     const DeepSeekFfnState& state) noexcept {
  if (weights.layer != state.layer() || !weights.ffn_norm ||
      !weights.router_weight || !weights.hca_function || !weights.hca_base ||
      !weights.hca_scale ||
      (weights.hash_router && (!weights.token_experts || weights.router_bias)) ||
      (!weights.hash_router && (!weights.router_bias || weights.token_experts)))
    return {ErrorCode::invalid_argument, "incomplete DeepSeek FFN binding"};
  return Status::success();
}

}  // namespace

DeepSeekFfnState::DeepSeekFfnState(void* allocation, std::uint64_t bytes,
                                   std::uint32_t layer) noexcept
    : allocation_(allocation), bytes_(bytes), layer_(layer) {}

DeepSeekFfnState::~DeepSeekFfnState() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

void DeepSeekFfnState::map(void* raw_base) noexcept {
  Arena arena{static_cast<std::byte*>(raw_base)};
  hca_normalized_ = arena.take<float>(4U * kHidden);
  hca_mixes_ = arena.take<float>(24U);
  collapsed_ = arena.take<float>(kHidden);
  ffn_input_ = arena.take<float>(kHidden);
  pre_ = arena.take<float>(4U);
  post_ = arena.take<float>(4U);
  comb_ = arena.take<float>(16U);
  router_logits_ = arena.take<float>(256U);
  routing_weights_ = arena.take<float>(7U);
  expert_indices_ = arena.take<std::uint32_t>(7U);
  routed_intermediate_ = arena.take<float>(kTopK * kIntermediate);
  routed_output_ = arena.take<float>(kHidden);
  shared_intermediate_ = arena.take<float>(kIntermediate);
  shared_output_ = arena.take<float>(kHidden);
  bytes_ = align_up(arena.cursor);
}

DeepSeekFfnStateResult create_deepseek_ffn_state(
    std::uint32_t layer) noexcept {
  if (layer >= 43U)
    return {{ErrorCode::invalid_argument, "invalid DeepSeek FFN state layer"},
            {}};
  DeepSeekFfnState sizing(nullptr, 0U, layer);
  sizing.map(nullptr);
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, sizing.bytes());
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek FFN state allocation"), {}};
  auto state = std::shared_ptr<DeepSeekFfnState>(
      new DeepSeekFfnState(allocation, sizing.bytes(), layer));
  state->map(allocation);
  error = cudaMemset(allocation, 0, state->bytes());
  const std::uint32_t shared = kSharedExpert;
  const float shared_weight = 1.0F;
  if (error == cudaSuccess)
    error = cudaMemcpy(state->expert_indices_ + kTopK, &shared,
                       sizeof(shared), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(state->routing_weights_ + kTopK, &shared_weight,
                       sizeof(shared_weight), cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek FFN state initialization"), {}};
  return {Status::success(), std::move(state)};
}

Status deepseek_ffn_route(const DeepSeekFfnRouteLaunch& launch) noexcept {
  if (!launch.weights || !launch.state || !launch.streams ||
      launch.epsilon <= 0.0F || launch.sinkhorn_iterations == 0U)
    return {ErrorCode::invalid_argument, "invalid DeepSeek FFN route launch"};
  auto status = check_binding(*launch.weights, *launch.state);
  if (!status.ok()) return status;
  auto& state = *launch.state;
  const auto& weights = *launch.weights;
  status = deepseek_hca_pre(
      {weights.hca_function, weights.hca_base, weights.hca_scale, kHidden},
      launch.streams, state.collapsed_, state.pre_, state.post_, state.comb_,
      {state.hca_normalized_, state.hca_mixes_}, launch.epsilon,
      launch.sinkhorn_iterations, launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight(state.collapsed_, weights.ffn_norm,
                                state.ffn_input_, kHidden, launch.epsilon,
                                launch.stream);
  if (!status.ok()) return status;
  if (weights.hash_router) {
    return deepseek_router_hash(
        state.ffn_input_, weights.router_weight, weights.token_experts,
        launch.token_id, state.router_logits_, state.routing_weights_,
        state.expert_indices_, 1.5F, launch.stream);
  }
  return deepseek_router_learned(
      state.ffn_input_, weights.router_weight, weights.router_bias,
      state.router_logits_, state.routing_weights_, state.expert_indices_,
      1.5F, launch.stream);
}

Status deepseek_ffn_execute(const DeepSeekFfnExecuteLaunch& launch) noexcept {
  if (!launch.weights || !launch.state || !launch.directory_entries ||
      !launch.streams || !launch.updated_streams ||
      launch.experts_per_layer != 257U)
    return {ErrorCode::invalid_argument, "invalid DeepSeek FFN execute launch"};
  auto status = check_binding(*launch.weights, *launch.state);
  if (!status.ok()) return status;
  auto& state = *launch.state;
  const auto layer = launch.weights->layer;
  status = launch_moe_single_token({
      state.ffn_input_, nullptr, nullptr, nullptr, nullptr,
      state.routing_weights_, state.expert_indices_,
      state.routed_intermediate_, state.routed_output_, kHidden,
      kIntermediate, kTopK, launch.experts_per_layer, launch.stream,
      launch.directory_entries, layer, 10.0F});
  if (!status.ok()) return status;
  status = launch_moe_single_token({
      state.ffn_input_, nullptr, nullptr, nullptr, nullptr,
      state.routing_weights_ + kTopK, state.expert_indices_ + kTopK,
      state.shared_intermediate_, state.shared_output_, kHidden,
      kIntermediate, 1U, launch.experts_per_layer, launch.stream,
      launch.directory_entries, layer, 10.0F});
  if (!status.ok()) return status;
  status = add_in_place(state.routed_output_, state.shared_output_, kHidden,
                        launch.stream);
  if (!status.ok()) return status;
  return deepseek_hca_post(state.routed_output_, launch.streams, state.post_,
                           state.comb_, launch.updated_streams, kHidden,
                           launch.stream);
}

}  // namespace expert::runtime::cuda
