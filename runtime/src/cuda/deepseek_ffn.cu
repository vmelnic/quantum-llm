#include "expert/runtime/cuda/deepseek_ffn.hpp"

#include "expert/runtime/cuda/deepseek_hca.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
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

Status record_profile_event(void* event, cudaStream_t stream,
                            const char* operation) noexcept {
  if (!event) return Status::success();
  const auto error = cudaEventRecord(static_cast<cudaEvent_t>(event), stream);
  return error == cudaSuccess ? Status::success() : failure(error, operation);
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

DeepSeekFfnPairWorkspace::DeepSeekFfnPairWorkspace(
    void* allocation, std::uint64_t bytes) noexcept
    : allocation_(allocation), bytes_(bytes) {
  map(allocation);
}

DeepSeekFfnPairWorkspace::~DeepSeekFfnPairWorkspace() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

void DeepSeekFfnPairWorkspace::map(void* base) noexcept {
  Arena arena{static_cast<std::byte*>(base)};
  ffn_input_ = arena.take<float>(2U * kHidden);
  routing_weights_ = arena.take<float>(2U * kTopK);
  expert_indices_ = arena.take<std::uint32_t>(2U * (kTopK + 1U));
  routed_indices_ = arena.take<std::uint32_t>(2U * kTopK);
  shared_weights_ = arena.take<float>(2U);
  shared_indices_ = arena.take<std::uint32_t>(2U);
  routed_intermediate_ = arena.take<float>(2U * kTopK * kIntermediate);
  routed_selection_outputs_ = arena.take<float>(2U * kTopK * kHidden);
  routed_output_ = arena.take<float>(2U * kHidden);
  routed_q_input_ = arena.take<std::int8_t>(2U * kHidden);
  routed_q_input_scales_ = arena.take<float>(2U);
  routed_q_intermediate_ =
      arena.take<std::int8_t>(2U * kTopK * kIntermediate);
  routed_q_intermediate_scales_ = arena.take<float>(2U * kTopK);
  shared_intermediate_ = arena.take<float>(2U * kIntermediate);
  shared_output_ = arena.take<float>(2U * kHidden);
  bytes_ = align_up(arena.cursor);
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
  routed_selection_outputs_ = arena.take<float>(kTopK * kHidden);
  routed_output_ = arena.take<float>(kHidden);
  routed_q_input_ = arena.take<std::int8_t>(kHidden);
  routed_q_input_scales_ = arena.take<float>(1U);
  routed_q_intermediate_ =
      arena.take<std::int8_t>(kTopK * kIntermediate);
  routed_q_intermediate_scales_ = arena.take<float>(kTopK);
  shared_intermediate_ = arena.take<float>(kIntermediate);
  shared_output_ = arena.take<float>(kHidden);
  bytes_ = align_up(arena.cursor);
}

DeepSeekFfnHybridWorkspace::DeepSeekFfnHybridWorkspace(
    void* device_allocation, std::uint64_t device_bytes,
    void* host_allocation, std::uint64_t host_bytes) noexcept
    : device_allocation_(device_allocation), host_allocation_(host_allocation),
      device_bytes_(device_bytes), host_bytes_(host_bytes) {
  map();
}

DeepSeekFfnHybridWorkspace::~DeepSeekFfnHybridWorkspace() {
  if (host_allocation_) static_cast<void>(cudaFreeHost(host_allocation_));
  if (device_allocation_) static_cast<void>(cudaFree(device_allocation_));
}

void DeepSeekFfnHybridWorkspace::map() noexcept {
  Arena device{static_cast<std::byte*>(device_allocation_)};
  selection_mask_ = device.take<std::uint8_t>(kTopK);
  alternate_slot_by_selection_ = device.take<std::uint32_t>(kTopK);
  alternate_outputs_ = device.take<float>(kTopK * kHidden);
  device_bytes_ = align_up(device.cursor);
  auto* host = static_cast<float*>(host_allocation_);
  host_input_ = host;
  host_outputs_ = host ? host + kHidden : nullptr;
  host_bytes_ = (kHidden + kTopK * kHidden) * sizeof(float);
}

DeepSeekFfnHybridWorkspaceResult create_deepseek_ffn_hybrid_workspace()
    noexcept {
  DeepSeekFfnHybridWorkspace sizing(nullptr, 0U, nullptr, 0U);
  void* device = nullptr;
  auto error = cudaMalloc(&device, sizing.device_bytes());
  if (error != cudaSuccess)
    return {failure(error, "allocate DeepSeek hybrid device workspace"), {}};
  void* host = nullptr;
  error = cudaHostAlloc(&host, sizing.pinned_host_bytes(),
                        cudaHostAllocPortable);
  if (error != cudaSuccess) {
    static_cast<void>(cudaFree(device));
    return {failure(error, "allocate DeepSeek hybrid pinned workspace"), {}};
  }
  return {Status::success(), std::shared_ptr<DeepSeekFfnHybridWorkspace>(
      new DeepSeekFfnHybridWorkspace(device, sizing.device_bytes(), host,
                                     sizing.pinned_host_bytes()))};
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

std::uint64_t deepseek_ffn_state_size() noexcept {
  DeepSeekFfnState sizing(nullptr, 0U, 0U);
  sizing.map(nullptr);
  return sizing.bytes();
}

std::uint64_t deepseek_ffn_pair_workspace_size() noexcept {
  DeepSeekFfnPairWorkspace sizing(nullptr, 0U);
  return sizing.bytes();
}

DeepSeekFfnPairWorkspaceResult create_deepseek_ffn_pair_workspace() noexcept {
  const auto bytes = deepseek_ffn_pair_workspace_size();
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, bytes);
  if (error != cudaSuccess)
    return {failure(error, "allocate DeepSeek pair FFN workspace"), {}};
  auto workspace = std::shared_ptr<DeepSeekFfnPairWorkspace>(
      new DeepSeekFfnPairWorkspace(allocation, bytes));
  error = cudaMemset(allocation, 0, bytes);
  if (error != cudaSuccess)
    return {failure(error, "reset DeepSeek pair FFN workspace"), {}};
  return {Status::success(), std::move(workspace)};
}

Status deepseek_ffn_gather_pair_routes(
    const DeepSeekFfnPairRouteGather& launch) noexcept {
  if (!launch.workspace || !launch.states[0] || !launch.states[1] ||
      launch.states[0]->layer() != launch.states[1]->layer())
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek pair route gather"};
  auto& workspace = *launch.workspace;
  auto stream = static_cast<cudaStream_t>(launch.stream);
  for (std::uint32_t row = 0U; row < 2U; ++row) {
    const auto& state = *launch.states[row];
    auto error = cudaMemcpyAsync(
        workspace.ffn_input_ + static_cast<std::size_t>(row) * kHidden,
        state.ffn_input_, kHidden * sizeof(float), cudaMemcpyDeviceToDevice,
        stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(
          workspace.routing_weights_ + static_cast<std::size_t>(row) * kTopK,
          state.routing_weights_, kTopK * sizeof(float),
          cudaMemcpyDeviceToDevice, stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(
          workspace.expert_indices_ +
              static_cast<std::size_t>(row) * (kTopK + 1U),
          state.expert_indices_, (kTopK + 1U) * sizeof(std::uint32_t),
          cudaMemcpyDeviceToDevice, stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(
          workspace.routed_indices_ + static_cast<std::size_t>(row) * kTopK,
          state.expert_indices_, kTopK * sizeof(std::uint32_t),
          cudaMemcpyDeviceToDevice, stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(workspace.shared_weights_ + row,
                              state.routing_weights_ + kTopK, sizeof(float),
                              cudaMemcpyDeviceToDevice, stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(workspace.shared_indices_ + row,
                              state.expert_indices_ + kTopK,
                              sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
                              stream);
    if (error != cudaSuccess)
      return failure(error, "gather DeepSeek pair route state");
  }
  return Status::success();
}

Status deepseek_ffn_execute_pair(
    const DeepSeekFfnPairExecuteLaunch& launch) noexcept {
  if (!launch.weights || !launch.states[0] || !launch.states[1] ||
      !launch.workspace || !launch.directory_entries || !launch.streams[0] ||
      !launch.streams[1] || !launch.updated_streams[0] ||
      !launch.updated_streams[1] || launch.experts_per_layer != 257U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek pair FFN execute launch"};
  for (auto* state : launch.states) {
    const auto checked = check_binding(*launch.weights, *state);
    if (!checked.ok()) return checked;
  }
  auto& workspace = *launch.workspace;
  const auto layer = launch.weights->layer;
  auto status = launch_moe_selection_batch({
      workspace.ffn_input_, workspace.routing_weights_,
      workspace.routed_indices_, nullptr, workspace.routed_intermediate_,
      workspace.routed_selection_outputs_, workspace.routed_q_input_,
      workspace.routed_q_input_scales_, workspace.routed_q_intermediate_,
      workspace.routed_q_intermediate_scales_, 2U, kHidden, kIntermediate,
      kTopK, launch.experts_per_layer, launch.stream,
      launch.directory_entries, layer, 10.0F, true, true});
  if (!status.ok()) return status;
  status = launch_moe_aggregate({
      workspace.routed_selection_outputs_, nullptr, nullptr, nullptr,
      workspace.routing_weights_, workspace.routed_output_, 0U, 2U, kHidden,
      kTopK, launch.stream});
  if (!status.ok()) return status;
  status = launch_moe_selection_batch({
      workspace.ffn_input_, workspace.shared_weights_,
      workspace.shared_indices_, nullptr, workspace.shared_intermediate_,
      workspace.shared_output_, workspace.routed_q_input_,
      workspace.routed_q_input_scales_, workspace.routed_q_intermediate_,
      workspace.routed_q_intermediate_scales_, 2U, kHidden, kIntermediate, 1U,
      launch.experts_per_layer, launch.stream, launch.directory_entries, layer,
      10.0F, true, true});
  if (!status.ok()) return status;
  status = add_in_place(workspace.routed_output_, workspace.shared_output_,
                        2U * kHidden, launch.stream);
  if (!status.ok()) return status;
  for (std::uint32_t row = 0U; row < 2U; ++row) {
    auto& state = *launch.states[row];
    status = deepseek_hca_post(
        workspace.routed_output_ + static_cast<std::size_t>(row) * kHidden,
        launch.streams[row], state.post_, state.comb_,
        launch.updated_streams[row], kHidden, launch.stream);
    if (!status.ok()) return status;
  }
  return Status::success();
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
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  cudaEvent_t routed_start{}, routed_stop{};
  if (launch.timing) {
    auto error = cudaEventCreate(&routed_start);
    if (error == cudaSuccess) error = cudaEventCreate(&routed_stop);
    if (error == cudaSuccess)
      error = cudaEventRecord(routed_start,
                              static_cast<cudaStream_t>(launch.stream));
    if (error != cudaSuccess) {
      if (routed_stop) static_cast<void>(cudaEventDestroy(routed_stop));
      if (routed_start) static_cast<void>(cudaEventDestroy(routed_start));
      return failure(error, "start DeepSeek routed timing");
    }
  }
  status = launch_moe_selection_batch({
      state.ffn_input_, state.routing_weights_, state.expert_indices_, nullptr,
      state.routed_intermediate_, state.routed_selection_outputs_,
      state.routed_q_input_, state.routed_q_input_scales_,
      state.routed_q_intermediate_, state.routed_q_intermediate_scales_,
      1U, kHidden, kIntermediate, kTopK, launch.experts_per_layer, launch.stream,
      launch.directory_entries, layer, 10.0F, true, true});
  if (!status.ok()) {
    if (routed_stop) static_cast<void>(cudaEventDestroy(routed_stop));
    if (routed_start) static_cast<void>(cudaEventDestroy(routed_start));
    return status;
  }
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->routed_stop : nullptr,
      stream, "record DeepSeek routed FFN stop");
  if (!status.ok()) return status;
  if (launch.timing) {
    auto error = cudaEventRecord(routed_stop,
                                 static_cast<cudaStream_t>(launch.stream));
    if (error == cudaSuccess) error = cudaEventSynchronize(routed_stop);
    if (error == cudaSuccess)
      error = cudaEventElapsedTime(&launch.timing->routed_gpu_ms,
                                   routed_start, routed_stop);
    static_cast<void>(cudaEventDestroy(routed_stop));
    static_cast<void>(cudaEventDestroy(routed_start));
    if (error != cudaSuccess)
      return failure(error, "finish DeepSeek routed timing");
    launch.timing->routed_selections = kTopK;
  }
  status = launch_moe_aggregate({
      state.routed_selection_outputs_, nullptr, nullptr, nullptr,
      state.routing_weights_, state.routed_output_, 0U, 1U, kHidden, kTopK,
      launch.stream});
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->aggregate_stop : nullptr,
      stream, "record DeepSeek routed aggregate stop");
  if (!status.ok()) return status;
  status = launch_moe_single_token({
      state.ffn_input_, nullptr, nullptr, nullptr, nullptr,
      state.routing_weights_ + kTopK, state.expert_indices_ + kTopK,
      state.shared_intermediate_, state.shared_output_, kHidden,
      kIntermediate, 1U, launch.experts_per_layer, launch.stream,
      launch.directory_entries, layer, 10.0F, true});
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->shared_stop : nullptr,
      stream, "record DeepSeek shared FFN stop");
  if (!status.ok()) return status;
  status = add_in_place(state.routed_output_, state.shared_output_, kHidden,
                        launch.stream);
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->merge_stop : nullptr,
      stream, "record DeepSeek FFN merge stop");
  if (!status.ok()) return status;
  return deepseek_hca_post(state.routed_output_, launch.streams, state.post_,
                           state.comb_, launch.updated_streams, kHidden,
                           launch.stream);
}

Status deepseek_ffn_execute_hybrid(
    const DeepSeekFfnHybridExecuteLaunch& launch) noexcept {
  if (!launch.weights || !launch.state || !launch.directory_entries ||
      !launch.streams || !launch.updated_streams || !launch.workspace ||
      !launch.cpu_executor ||
      launch.cpu_groups.empty() || launch.experts_per_layer != 257U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek hybrid FFN execute launch"};
  auto status = check_binding(*launch.weights, *launch.state);
  if (!status.ok()) return status;
  std::array<std::uint8_t, kTopK> primary_mask{};
  primary_mask.fill(1U);
  std::array<std::uint32_t, kTopK> alternate_slots{};
  alternate_slots.fill(0U);
  std::array<bool, kTopK> claimed{};
  std::uint32_t alternate_count = 0U;
  for (const auto& group : launch.cpu_groups) {
    if (group.selections.size() != group.output_slots.size())
      return {ErrorCode::invalid_argument,
              "DeepSeek hybrid CPU selection mapping mismatch"};
    for (std::size_t index = 0U; index < group.selections.size(); ++index) {
      const auto selection = group.selections[index];
      const auto output_slot = group.output_slots[index];
      if (selection >= kTopK || output_slot >= kTopK || claimed[selection])
        return {ErrorCode::invalid_argument,
                "invalid or duplicate DeepSeek hybrid CPU selection"};
      claimed[selection] = true;
      primary_mask[selection] = 0U;
      alternate_slots[selection] = output_slot;
      alternate_count = std::max(alternate_count, output_slot + 1U);
    }
  }
  const auto cpu_selection_count = static_cast<std::uint32_t>(
      std::count(claimed.begin(), claimed.end(), true));
  if (cpu_selection_count == 0U || alternate_count != cpu_selection_count)
    return {ErrorCode::invalid_argument,
            "DeepSeek hybrid CPU outputs must use dense compact slots"};

  auto& state = *launch.state;
  auto& workspace = *launch.workspace;
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  auto error = cudaMemcpyAsync(workspace.host_input_, state.ffn_input_,
                               kHidden * sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess)
    return failure(error, "stage DeepSeek hybrid CPU input");
  error = cudaMemcpyAsync(workspace.selection_mask_, primary_mask.data(),
                          primary_mask.size(), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(workspace.alternate_slot_by_selection_,
                            alternate_slots.data(),
                            alternate_slots.size() * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice, stream);
  }
  if (error != cudaSuccess)
    return failure(error, "stage DeepSeek hybrid selection map");

  const auto layer = launch.weights->layer;
  status = launch_moe_selection_batch({
      state.ffn_input_, state.routing_weights_, state.expert_indices_,
      workspace.selection_mask_, state.routed_intermediate_,
      state.routed_selection_outputs_, state.routed_q_input_,
      state.routed_q_input_scales_, state.routed_q_intermediate_,
      state.routed_q_intermediate_scales_, 1U, kHidden, kIntermediate, kTopK,
      launch.experts_per_layer, launch.stream, launch.directory_entries, layer,
      10.0F, true, true});
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->routed_stop : nullptr,
      stream, "record DeepSeek hybrid routed FFN stop");
  if (!status.ok()) return status;

  status = launch.cpu_executor->execute(
      launch.cpu_groups,
      std::span<const float>(workspace.host_input_, kHidden), 1U, kTopK,
      std::span<float>(workspace.host_outputs_,
                       static_cast<std::size_t>(alternate_count) * kHidden));
  if (!status.ok()) {
    static_cast<void>(cudaStreamSynchronize(stream));
    return status;
  }
  error = cudaMemcpyAsync(workspace.alternate_outputs_,
                          workspace.host_outputs_,
                          static_cast<std::size_t>(alternate_count) * kHidden *
                              sizeof(float),
                          cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess)
    return failure(error, "upload DeepSeek hybrid CPU outputs");
  status = launch_moe_aggregate({
      state.routed_selection_outputs_, workspace.alternate_outputs_,
      workspace.selection_mask_, workspace.alternate_slot_by_selection_,
      state.routing_weights_, state.routed_output_, alternate_count, 1U,
      kHidden, kTopK, launch.stream});
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->aggregate_stop : nullptr,
      stream, "record DeepSeek hybrid aggregate stop");
  if (!status.ok()) return status;
  status = launch_moe_single_token({
      state.ffn_input_, nullptr, nullptr, nullptr, nullptr,
      state.routing_weights_ + kTopK, state.expert_indices_ + kTopK,
      state.shared_intermediate_, state.shared_output_, kHidden,
      kIntermediate, 1U, launch.experts_per_layer, launch.stream,
      launch.directory_entries, layer, 10.0F, true});
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->shared_stop : nullptr,
      stream, "record DeepSeek hybrid shared FFN stop");
  if (!status.ok()) return status;
  status = add_in_place(state.routed_output_, state.shared_output_, kHidden,
                        launch.stream);
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->merge_stop : nullptr,
      stream, "record DeepSeek hybrid FFN merge stop");
  if (!status.ok()) return status;
  return deepseek_hca_post(state.routed_output_, launch.streams, state.post_,
                           state.comb_, launch.updated_streams, kHidden,
                           launch.stream);
}

}  // namespace expert::runtime::cuda
