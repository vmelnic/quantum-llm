#include "expert/runtime/cuda/deepseek_ffn.hpp"

#include "expert/runtime/cuda/deepseek_hca.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
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

__global__ void pair_route_layout_kernel(
    const std::uint32_t* routed_indices, std::uint32_t* expert_indices,
    float* shared_weights, std::uint32_t* shared_indices) {
  const auto index = static_cast<std::uint32_t>(threadIdx.x);
  if (index < 12U) {
    const auto row = index / kTopK;
    const auto slot = index % kTopK;
    expert_indices[row * (kTopK + 1U) + slot] = routed_indices[index];
  }
  if (index < 2U) {
    expert_indices[index * (kTopK + 1U) + kTopK] = kSharedExpert;
    shared_weights[index] = 1.0F;
    shared_indices[index] = kSharedExpert;
  }
}

__global__ void batch_shared_layout_kernel(std::uint32_t* shared_indices,
                                           std::uint32_t rows) {
  const auto row = static_cast<std::uint32_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) shared_indices[row] = kSharedExpert;
}

__global__ void routed_selection_mask_kernel(std::uint8_t* mask,
                                             std::uint64_t bits) {
  const auto slot = static_cast<std::uint32_t>(threadIdx.x);
  if (slot < kTopK)
    mask[slot] = static_cast<std::uint8_t>((bits >> slot) & 1U);
}

}  // namespace

DeepSeekFfnState::DeepSeekFfnState(void* allocation, std::uint64_t bytes,
                                   std::uint32_t layer) noexcept
    : allocation_(allocation), bytes_(bytes), layer_(layer) {}

DeepSeekFfnState::~DeepSeekFfnState() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

DeepSeekRoutePredictionState::DeepSeekRoutePredictionState(
    void* allocation, std::uint64_t bytes) noexcept
    : allocation_(allocation), bytes_(bytes) {
  map(allocation);
}

DeepSeekRoutePredictionState::~DeepSeekRoutePredictionState() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

void DeepSeekRoutePredictionState::map(void* raw_base) noexcept {
  Arena arena{static_cast<std::byte*>(raw_base)};
  router_logits_ = arena.take<float>(256U);
  routing_weights_ = arena.take<float>(kTopK);
  expert_indices_ = arena.take<std::uint32_t>(kTopK);
  bytes_ = align_up(arena.cursor);
}

DeepSeekFfnPairWorkspace::DeepSeekFfnPairWorkspace(
    void* allocation, std::uint64_t bytes) noexcept
    : allocation_(allocation), bytes_(bytes) {
  map(allocation);
}

DeepSeekFfnPairWorkspace::~DeepSeekFfnPairWorkspace() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

DeepSeekFfnBatchWorkspace::DeepSeekFfnBatchWorkspace(
    void* allocation, std::uint64_t bytes,
    std::uint32_t maximum_rows) noexcept
    : allocation_(allocation), bytes_(bytes), maximum_rows_(maximum_rows) {
  map(allocation);
}

DeepSeekFfnBatchWorkspace::~DeepSeekFfnBatchWorkspace() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

void DeepSeekFfnPairWorkspace::map(void* base) noexcept {
  Arena arena{static_cast<std::byte*>(base)};
  hca_normalized_ = arena.take<float>(2U * 4U * kHidden);
  hca_mixes_ = arena.take<float>(2U * 24U);
  collapsed_ = arena.take<float>(2U * kHidden);
  pre_ = arena.take<float>(2U * 4U);
  post_ = arena.take<float>(2U * 4U);
  comb_ = arena.take<float>(2U * 16U);
  ffn_input_ = arena.take<float>(2U * kHidden);
  router_logits_ = arena.take<float>(2U * 256U);
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

void DeepSeekFfnBatchWorkspace::map(void* base) noexcept {
  Arena arena{static_cast<std::byte*>(base)};
  const auto rows = static_cast<std::size_t>(maximum_rows_);
  hca_normalized_ = arena.take<float>(rows * 4U * kHidden);
  hca_mixes_ = arena.take<float>(rows * 24U);
  collapsed_ = arena.take<float>(rows * kHidden);
  pre_ = arena.take<float>(rows * 4U);
  post_ = arena.take<float>(rows * 4U);
  comb_ = arena.take<float>(rows * 16U);
  ffn_input_ = arena.take<float>(rows * kHidden);
  router_logits_ = arena.take<float>(rows * 256U);
  routing_weights_ = arena.take<float>(rows * kTopK);
  expert_indices_ = arena.take<std::uint32_t>(rows * kTopK);
  routed_selection_outputs_ =
      arena.take<float>(rows * kTopK * kHidden);
  routed_output_ = arena.take<float>(rows * kHidden);
  shared_indices_ = arena.take<std::uint32_t>(rows);
  shared_intermediate_ = arena.take<float>(rows * kIntermediate);
  shared_output_ = arena.take<float>(rows * kHidden);
  shared_q_input_ = arena.take<std::int8_t>(rows * kHidden);
  shared_q_input_scales_ = arena.take<float>(rows);
  shared_q_intermediate_ = arena.take<std::int8_t>(rows * kIntermediate);
  shared_q_intermediate_scales_ = arena.take<float>(rows);
  direct_inputs_ = arena.take<float>(rows * kHidden);
  direct_gate_ = arena.take<float>(rows * kIntermediate);
  direct_up_ = arena.take<float>(rows * kIntermediate);
  direct_intermediate_ = arena.take<float>(rows * kIntermediate);
  direct_outputs_ = arena.take<float>(rows * kHidden);
  direct_q_input_ = arena.take<std::int8_t>(rows * kHidden);
  direct_q_input_scales_ = arena.take<float>(rows);
  direct_q_intermediate_ = arena.take<std::int8_t>(rows * kIntermediate);
  direct_q_intermediate_scales_ = arena.take<float>(rows);
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
  routed_selection_mask_ = arena.take<std::uint8_t>(kTopK);
  shared_intermediate_ = arena.take<float>(kIntermediate);
  shared_output_ = arena.take<float>(kHidden);
  bytes_ = align_up(arena.cursor);
}

DeepSeekFfnHybridWorkspace::DeepSeekFfnHybridWorkspace(
    void* device_allocation, std::uint64_t device_bytes,
    void* host_allocation, std::uint64_t host_bytes,
    void* input_ready_event) noexcept
    : device_allocation_(device_allocation), host_allocation_(host_allocation),
      device_bytes_(device_bytes), host_bytes_(host_bytes),
      input_ready_event_(input_ready_event) {
  map();
}

DeepSeekFfnHybridWorkspace::~DeepSeekFfnHybridWorkspace() {
  if (input_ready_event_)
    static_cast<void>(cudaEventDestroy(
        static_cast<cudaEvent_t>(input_ready_event_)));
  if (host_allocation_) static_cast<void>(cudaFreeHost(host_allocation_));
  if (device_allocation_) static_cast<void>(cudaFree(device_allocation_));
}

void DeepSeekFfnHybridWorkspace::map() noexcept {
  Arena device{static_cast<std::byte*>(device_allocation_)};
  selection_mask_ = device.take<std::uint8_t>(2U * kTopK);
  alternate_slot_by_selection_ = device.take<std::uint32_t>(2U * kTopK);
  alternate_outputs_ = device.take<float>(2U * kTopK * kHidden);
  device_bytes_ = align_up(device.cursor);
  auto* host = static_cast<float*>(host_allocation_);
  host_input_ = host;
  host_outputs_ = host ? host + 2U * kHidden : nullptr;
  host_bytes_ = (2U * kHidden + 2U * kTopK * kHidden) * sizeof(float);
}

DeepSeekFfnHybridWorkspaceResult create_deepseek_ffn_hybrid_workspace()
    noexcept {
  DeepSeekFfnHybridWorkspace sizing(nullptr, 0U, nullptr, 0U, nullptr);
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
  cudaEvent_t input_ready = nullptr;
  error = cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming);
  if (error != cudaSuccess) {
    static_cast<void>(cudaFreeHost(host));
    static_cast<void>(cudaFree(device));
    return {failure(error, "create DeepSeek hybrid input event"), {}};
  }
  return {Status::success(), std::shared_ptr<DeepSeekFfnHybridWorkspace>(
      new DeepSeekFfnHybridWorkspace(device, sizing.device_bytes(), host,
                                     sizing.pinned_host_bytes(),
                                     input_ready))};
}

DeepSeekFfnStateResult create_deepseek_ffn_state(
    std::uint32_t layer) noexcept {
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

std::uint64_t deepseek_route_prediction_state_size() noexcept {
  DeepSeekRoutePredictionState sizing(nullptr, 0U);
  return sizing.bytes();
}

DeepSeekRoutePredictionStateResult
create_deepseek_route_prediction_state() noexcept {
  const auto bytes = deepseek_route_prediction_state_size();
  void* allocation = nullptr;
  const auto error = cudaMalloc(&allocation, bytes);
  if (error != cudaSuccess)
    return {failure(error, "allocate DeepSeek route prediction state"), {}};
  auto state = std::shared_ptr<DeepSeekRoutePredictionState>(
      new DeepSeekRoutePredictionState(allocation, bytes));
  return {Status::success(), std::move(state)};
}

Status deepseek_predict_route(
    const DeepSeekRoutePredictionLaunch& launch) noexcept {
  if (!launch.target_weights || !launch.state ||
      !launch.preceding_gate_input)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek route prediction launch"};
  const auto& weights = *launch.target_weights;
  auto& state = *launch.state;
  if (!weights.router_weight ||
      (weights.hash_router && (!weights.token_experts || weights.router_bias)) ||
      (!weights.hash_router && (!weights.router_bias || weights.token_experts)))
    return {ErrorCode::invalid_argument,
            "incomplete DeepSeek prediction router binding"};
  if (weights.hash_router) {
    return deepseek_router_hash(
        launch.preceding_gate_input, weights.router_weight,
        weights.token_experts, launch.token_id, state.router_logits_,
        state.routing_weights_, state.expert_indices_, 1.5F, launch.stream);
  }
  return deepseek_router_learned(
      launch.preceding_gate_input, weights.router_weight, weights.router_bias,
      state.router_logits_, state.routing_weights_, state.expert_indices_,
      1.5F, launch.stream);
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

std::uint64_t deepseek_ffn_batch_workspace_size(
    std::uint32_t maximum_rows) noexcept {
  if (maximum_rows == 0U ||
      maximum_rows > kDeepSeekMaximumSequenceRows)
    return 0U;
  DeepSeekFfnBatchWorkspace sizing(nullptr, 0U, maximum_rows);
  return sizing.bytes();
}

DeepSeekFfnBatchWorkspaceResult create_deepseek_ffn_batch_workspace(
    std::uint32_t maximum_rows) noexcept {
  const auto bytes = deepseek_ffn_batch_workspace_size(maximum_rows);
  if (bytes == 0U)
    return {{ErrorCode::invalid_argument,
             "invalid DeepSeek batch FFN workspace geometry"}, {}};
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, bytes);
  if (error != cudaSuccess)
    return {failure(error, "allocate DeepSeek batch FFN workspace"), {}};
  auto workspace = std::shared_ptr<DeepSeekFfnBatchWorkspace>(
      new DeepSeekFfnBatchWorkspace(allocation, bytes, maximum_rows));
  error = cudaMemset(allocation, 0, bytes);
  if (error != cudaSuccess)
    return {failure(error, "reset DeepSeek batch FFN workspace"), {}};
  return {Status::success(), std::move(workspace)};
}

Status deepseek_ffn_route_batch(
    const DeepSeekFfnBatchRouteLaunch& launch) noexcept {
  if (!launch.weights || !launch.identity_state || !launch.workspace ||
      !launch.streams || !launch.token_ids || launch.rows == 0U ||
      launch.rows > launch.workspace->maximum_rows() ||
      launch.epsilon <= 0.0F || launch.sinkhorn_iterations == 0U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek batch FFN route launch"};
  auto status = check_binding(*launch.weights, *launch.identity_state);
  if (!status.ok()) return status;
  auto& workspace = *launch.workspace;
  const auto& weights = *launch.weights;
  status = deepseek_hca_pre_batch(
      {weights.hca_function, weights.hca_base, weights.hca_scale, kHidden},
      launch.streams, launch.rows, workspace.collapsed_, workspace.pre_,
      workspace.post_, workspace.comb_,
      {workspace.hca_normalized_, workspace.hca_mixes_}, launch.epsilon,
      launch.sinkhorn_iterations, launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight_batch(
      workspace.collapsed_, weights.ffn_norm, workspace.ffn_input_,
      launch.rows, kHidden, launch.epsilon, launch.stream);
  if (!status.ok()) return status;
  if (weights.hash_router) {
    return deepseek_router_hash_rows(
        workspace.ffn_input_, weights.router_weight, weights.token_experts,
        launch.token_ids, launch.rows, workspace.router_logits_,
        workspace.routing_weights_, workspace.expert_indices_, 1.5F,
        launch.stream);
  }
  return deepseek_router_learned_rows(
      workspace.ffn_input_, weights.router_weight, weights.router_bias,
      launch.rows, workspace.router_logits_, workspace.routing_weights_,
      workspace.expert_indices_, 1.5F, launch.stream);
}

Status deepseek_ffn_execute_packed_batch(
    const DeepSeekFfnPackedBatchLaunch& launch) noexcept {
  if (!launch.expert || !launch.workspace || !launch.input || !launch.output ||
      launch.rows == 0U || launch.rows > launch.workspace->maximum_rows() ||
      launch.swiglu_limit < 0.0F)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek packed FFN batch launch"};
  const auto& sections = launch.expert->sections();
  const auto* base = launch.expert->base();
  constexpr std::uint64_t kMatrixValues =
      static_cast<std::uint64_t>(kHidden) * kIntermediate;
  constexpr std::uint64_t kWeightBytes = kMatrixValues / 2U;
  constexpr std::uint64_t kScaleBytes = kMatrixValues / 32U;
  if (!base || sections.w1_weight_bytes != kWeightBytes ||
      sections.w3_weight_bytes != kWeightBytes ||
      sections.w2_weight_bytes != kWeightBytes ||
      sections.w1_scale_bytes != kScaleBytes ||
      sections.w3_scale_bytes != kScaleBytes ||
      sections.w2_scale_bytes != kScaleBytes)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek packed FFN batch expert"};
  auto& workspace = *launch.workspace;
  auto status = quantize_q8_batch(
      launch.input, workspace.direct_q_input_,
      workspace.direct_q_input_scales_, launch.rows, kHidden, kHidden,
      launch.stream);
  if (!status.ok()) return status;
  const Fp4Block32Matrix w1{
      base + sections.w1_weight_offset,
      base + sections.w1_scale_offset, kIntermediate, kHidden, kHidden};
  const Fp4Block32Matrix w3{
      base + sections.w3_weight_offset,
      base + sections.w3_scale_offset, kIntermediate, kHidden, kHidden};
  const Fp4Block32Matrix w2{
      base + sections.w2_weight_offset,
      base + sections.w2_scale_offset, kHidden, kIntermediate, kIntermediate};
  status = fp4_gemm_q8_block32(
      w1, workspace.direct_q_input_, workspace.direct_q_input_scales_,
      workspace.direct_gate_, launch.rows, launch.stream);
  if (!status.ok()) return status;
  status = fp4_gemm_q8_block32(
      w3, workspace.direct_q_input_, workspace.direct_q_input_scales_,
      workspace.direct_up_, launch.rows, launch.stream);
  if (!status.ok()) return status;
  const auto intermediate_values = launch.rows * kIntermediate;
  status = deepseek_swiglu_product(
      workspace.direct_gate_, workspace.direct_up_,
      workspace.direct_intermediate_, intermediate_values,
      launch.swiglu_limit, launch.bf16_intermediate, launch.stream);
  if (!status.ok()) return status;
  status = quantize_q8_batch(
      workspace.direct_intermediate_, workspace.direct_q_intermediate_,
      workspace.direct_q_intermediate_scales_, launch.rows, kIntermediate,
      kIntermediate, launch.stream);
  if (!status.ok()) return status;
  return fp4_gemm_q8_block32(
      w2, workspace.direct_q_intermediate_,
      workspace.direct_q_intermediate_scales_, launch.output, launch.rows,
      launch.stream);
}

Status deepseek_ffn_finalize_batch(
    const DeepSeekFfnBatchFinalizeLaunch& launch) noexcept {
  if (!launch.weights || !launch.identity_state || !launch.workspace ||
      !launch.directory_entries || !launch.streams ||
      !launch.updated_streams || launch.rows == 0U ||
      launch.rows > launch.workspace->maximum_rows() ||
      launch.experts_per_layer != 257U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek batch FFN finalize launch"};
  auto status = check_binding(*launch.weights, *launch.identity_state);
  if (!status.ok()) return status;
  auto& workspace = *launch.workspace;
  status = launch_moe_aggregate({
      workspace.routed_selection_outputs_, nullptr, nullptr, nullptr,
      workspace.routing_weights_, workspace.routed_output_, 0U, launch.rows,
      kHidden, kTopK, launch.stream});
  if (!status.ok()) return status;
  batch_shared_layout_kernel<<<(launch.rows + 255U) / 256U, 256U, 0,
                               static_cast<cudaStream_t>(launch.stream)>>>(
      workspace.shared_indices_, launch.rows);
  auto error = cudaPeekAtLastError();
  if (error != cudaSuccess)
    return failure(error, "prepare DeepSeek batch shared route");
  status = launch_moe_selection_batch({
      workspace.ffn_input_, nullptr, workspace.shared_indices_, nullptr,
      workspace.shared_intermediate_, workspace.shared_output_,
      workspace.shared_q_input_, workspace.shared_q_input_scales_,
      workspace.shared_q_intermediate_,
      workspace.shared_q_intermediate_scales_, launch.rows, kHidden,
      kIntermediate, 1U, launch.experts_per_layer, launch.stream,
      launch.directory_entries, launch.weights->layer, 10.0F, true, true});
  if (!status.ok()) return status;
  status = add_in_place(workspace.routed_output_, workspace.shared_output_,
                        launch.rows * kHidden, launch.stream);
  if (!status.ok()) return status;
  return deepseek_hca_post_batch(
      workspace.routed_output_, launch.streams, workspace.post_,
      workspace.comb_, launch.updated_streams, launch.rows, kHidden,
      launch.stream);
}

Status deepseek_ffn_route_pair(
    const DeepSeekFfnPairRouteLaunch& launch) noexcept {
  if (!launch.weights || !launch.states[0] || !launch.states[1] ||
      !launch.workspace || !launch.streams[0] || !launch.streams[1] ||
      launch.epsilon <= 0.0F || launch.sinkhorn_iterations == 0U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek pair FFN route launch"};
  for (auto* state : launch.states) {
    const auto checked = check_binding(*launch.weights, *state);
    if (!checked.ok()) return checked;
  }
  auto& workspace = *launch.workspace;
  const auto& weights = *launch.weights;
  auto status = deepseek_hca_pre_pair(
      {weights.hca_function, weights.hca_base, weights.hca_scale, kHidden},
      launch.streams, workspace.collapsed_, workspace.pre_, workspace.post_,
      workspace.comb_, {workspace.hca_normalized_, workspace.hca_mixes_},
      launch.epsilon, launch.sinkhorn_iterations, launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight_batch(
      workspace.collapsed_, weights.ffn_norm, workspace.ffn_input_, 2U,
      kHidden, launch.epsilon, launch.stream);
  if (!status.ok()) return status;
  if (weights.hash_router) {
    status = deepseek_router_hash_batch(
        workspace.ffn_input_, weights.router_weight, weights.token_experts,
        launch.token_ids[0], launch.token_ids[1], workspace.router_logits_,
        workspace.routing_weights_, workspace.routed_indices_, 1.5F,
        launch.stream);
  } else {
    status = deepseek_router_learned_batch(
        workspace.ffn_input_, weights.router_weight, weights.router_bias,
        workspace.router_logits_, workspace.routing_weights_,
        workspace.routed_indices_, 1.5F, launch.stream);
  }
  if (!status.ok()) return status;
  pair_route_layout_kernel<<<1U, 32U, 0,
                             static_cast<cudaStream_t>(launch.stream)>>>(
      workspace.routed_indices_, workspace.expert_indices_,
      workspace.shared_weights_, workspace.shared_indices_);
  const auto error = cudaPeekAtLastError();
  return error == cudaSuccess
      ? Status::success()
      : failure(error, "publish DeepSeek pair route layout");
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
    status = deepseek_hca_post(
        workspace.routed_output_ + static_cast<std::size_t>(row) * kHidden,
        launch.streams[row], workspace.post_ + row * 4U,
        workspace.comb_ + row * 16U,
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

Status deepseek_ffn_execute_selections(
    const DeepSeekFfnSelectionExecuteLaunch& launch) noexcept {
  constexpr auto kAllSelections = (std::uint64_t{1U} << kTopK) - 1U;
  if (!launch.weights || !launch.state || !launch.directory_entries ||
      launch.selection_mask == 0U ||
      (launch.selection_mask & ~kAllSelections) != 0U ||
      launch.experts_per_layer != 257U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek selection FFN execute launch"};
  auto status = check_binding(*launch.weights, *launch.state);
  if (!status.ok()) return status;
  auto& state = *launch.state;
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  routed_selection_mask_kernel<<<1U, 32U, 0, stream>>>(
      state.routed_selection_mask_, launch.selection_mask);
  const auto mask_error = cudaPeekAtLastError();
  if (mask_error != cudaSuccess)
    return failure(mask_error, "publish DeepSeek routed selection mask");
  return launch_moe_selection_batch({
      state.ffn_input_, state.routing_weights_, state.expert_indices_,
      state.routed_selection_mask_, state.routed_intermediate_,
      state.routed_selection_outputs_, state.routed_q_input_,
      state.routed_q_input_scales_, state.routed_q_intermediate_,
      state.routed_q_intermediate_scales_, 1U, kHidden, kIntermediate, kTopK,
      launch.experts_per_layer, launch.stream, launch.directory_entries,
      launch.weights->layer, 10.0F, true, true});
}

Status deepseek_ffn_import_selection_output(
    const DeepSeekFfnSelectionImportLaunch& launch) noexcept {
  if (!launch.state || !launch.host_output ||
      launch.selection_index >= kTopK ||
      launch.host_output_bytes != kHidden * sizeof(float))
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek remote selection output"};
  auto& state = *launch.state;
  const auto error = cudaMemcpyAsync(
      state.routed_selection_outputs_ +
          static_cast<std::size_t>(launch.selection_index) * kHidden,
      launch.host_output, launch.host_output_bytes, cudaMemcpyHostToDevice,
      static_cast<cudaStream_t>(launch.stream));
  return error == cudaSuccess
             ? Status::success()
             : failure(error, "import DeepSeek remote selection output");
}

Status deepseek_ffn_finalize(const DeepSeekFfnFinalizeLaunch& launch) noexcept {
  if (!launch.weights || !launch.state || !launch.directory_entries ||
      !launch.streams || !launch.updated_streams ||
      launch.experts_per_layer != 257U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek FFN finalize launch"};
  auto status = check_binding(*launch.weights, *launch.state);
  if (!status.ok()) return status;
  auto& state = *launch.state;
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  status = launch_moe_aggregate({
      state.routed_selection_outputs_, nullptr, nullptr, nullptr,
      state.routing_weights_, state.routed_output_, 0U, 1U, kHidden, kTopK,
      launch.stream});
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->aggregate_stop : nullptr,
      stream, "record DeepSeek incremental aggregate stop");
  if (!status.ok()) return status;
  status = launch_moe_single_token({
      state.ffn_input_, nullptr, nullptr, nullptr, nullptr,
      state.routing_weights_ + kTopK, state.expert_indices_ + kTopK,
      state.shared_intermediate_, state.shared_output_, kHidden,
      kIntermediate, 1U, launch.experts_per_layer, launch.stream,
      launch.directory_entries, launch.weights->layer, 10.0F, true});
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->shared_stop : nullptr,
      stream, "record DeepSeek incremental shared FFN stop");
  if (!status.ok()) return status;
  status = add_in_place(state.routed_output_, state.shared_output_, kHidden,
                        launch.stream);
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->merge_stop : nullptr,
      stream, "record DeepSeek incremental merge stop");
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
  const auto input_ready =
      static_cast<cudaEvent_t>(workspace.input_ready_event_);
  auto error = cudaMemcpyAsync(workspace.host_input_, state.ffn_input_,
                               kHidden * sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) error = cudaEventRecord(input_ready, stream);
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

  // The event is recorded immediately after the activation D2H. CUDA work
  // for the resident selections is already queued behind it, so waiting for
  // only this event starts the CPU lane while the GPU lane is executing.
  error = cudaEventSynchronize(input_ready);
  if (error != cudaSuccess)
    return failure(error, "wait for DeepSeek hybrid CPU input");
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
  if (error != cudaSuccess)
    return failure(error, "upload DeepSeek hybrid CPU outputs");
  // The compact result upload and every dependent CUDA operation use the same
  // stream. Do not block the host here; the next hybrid invocation's input
  // event naturally waits for this layer before reusing the pinned buffers.
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

Status deepseek_ffn_execute_pair_hybrid(
    const DeepSeekFfnHybridPairExecuteLaunch& launch) noexcept {
  constexpr std::uint32_t kRows = 2U;
  constexpr std::uint32_t kSelections = kRows * kTopK;
  if (!launch.weights || !launch.states[0] || !launch.states[1] ||
      !launch.pair_workspace || !launch.directory_entries ||
      !launch.streams[0] || !launch.streams[1] ||
      !launch.updated_streams[0] || !launch.updated_streams[1] ||
      !launch.hybrid_workspace || !launch.cpu_executor ||
      launch.cpu_groups.empty() || launch.experts_per_layer != 257U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek hybrid pair FFN execute launch"};
  for (auto* state : launch.states) {
    const auto checked = check_binding(*launch.weights, *state);
    if (!checked.ok()) return checked;
  }

  std::array<std::uint8_t, kSelections> primary_mask{};
  primary_mask.fill(1U);
  std::array<std::uint32_t, kSelections> alternate_slots{};
  std::array<bool, kSelections> claimed{};
  std::uint32_t alternate_count = 0U;
  for (const auto& group : launch.cpu_groups) {
    if (group.selections.size() != group.output_slots.size())
      return {ErrorCode::invalid_argument,
              "DeepSeek hybrid pair CPU selection mapping mismatch"};
    for (std::size_t index = 0U; index < group.selections.size(); ++index) {
      const auto selection = group.selections[index];
      const auto output_slot = group.output_slots[index];
      if (selection >= kSelections || output_slot >= kSelections ||
          claimed[selection])
        return {ErrorCode::invalid_argument,
                "invalid or duplicate DeepSeek hybrid pair selection"};
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
            "DeepSeek hybrid pair CPU outputs must use compact slots"};

  auto& pair = *launch.pair_workspace;
  auto& hybrid = *launch.hybrid_workspace;
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  const auto input_ready =
      static_cast<cudaEvent_t>(hybrid.input_ready_event_);
  auto error = cudaMemcpyAsync(hybrid.host_input_, pair.ffn_input_,
                               kRows * kHidden * sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) error = cudaEventRecord(input_ready, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(hybrid.selection_mask_, primary_mask.data(),
                            primary_mask.size(), cudaMemcpyHostToDevice,
                            stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(hybrid.alternate_slot_by_selection_,
                            alternate_slots.data(),
                            alternate_slots.size() * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    return failure(error, "stage DeepSeek hybrid pair inputs");

  auto status = launch_moe_selection_batch({
      pair.ffn_input_, pair.routing_weights_, pair.routed_indices_,
      hybrid.selection_mask_, pair.routed_intermediate_,
      pair.routed_selection_outputs_, pair.routed_q_input_,
      pair.routed_q_input_scales_, pair.routed_q_intermediate_,
      pair.routed_q_intermediate_scales_, kRows, kHidden, kIntermediate,
      kTopK, launch.experts_per_layer, launch.stream,
      launch.directory_entries, launch.weights->layer, 10.0F, true, true});
  if (!status.ok()) return status;

  error = cudaEventSynchronize(input_ready);
  if (error != cudaSuccess)
    return failure(error, "wait for DeepSeek hybrid pair CPU input");
  status = launch.cpu_executor->execute(
      launch.cpu_groups,
      std::span<const float>(hybrid.host_input_, kRows * kHidden), kRows,
      kTopK,
      std::span<float>(hybrid.host_outputs_,
                       static_cast<std::size_t>(alternate_count) * kHidden));
  if (!status.ok()) {
    static_cast<void>(cudaStreamSynchronize(stream));
    return status;
  }
  error = cudaMemcpyAsync(
      hybrid.alternate_outputs_, hybrid.host_outputs_,
      static_cast<std::size_t>(alternate_count) * kHidden * sizeof(float),
      cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    return failure(error, "upload DeepSeek hybrid pair CPU outputs");

  status = launch_moe_aggregate({
      pair.routed_selection_outputs_, hybrid.alternate_outputs_,
      hybrid.selection_mask_, hybrid.alternate_slot_by_selection_,
      pair.routing_weights_, pair.routed_output_, alternate_count, kRows,
      kHidden, kTopK, launch.stream});
  if (!status.ok()) return status;
  status = launch_moe_selection_batch({
      pair.ffn_input_, pair.shared_weights_, pair.shared_indices_, nullptr,
      pair.shared_intermediate_, pair.shared_output_, pair.routed_q_input_,
      pair.routed_q_input_scales_, pair.routed_q_intermediate_,
      pair.routed_q_intermediate_scales_, kRows, kHidden, kIntermediate, 1U,
      launch.experts_per_layer, launch.stream, launch.directory_entries,
      launch.weights->layer, 10.0F, true, true});
  if (!status.ok()) return status;
  status = add_in_place(pair.routed_output_, pair.shared_output_,
                        kRows * kHidden, launch.stream);
  if (!status.ok()) return status;
  for (std::uint32_t row = 0U; row < kRows; ++row) {
    status = deepseek_hca_post(
        pair.routed_output_ + static_cast<std::size_t>(row) * kHidden,
        launch.streams[row], pair.post_ + row * 4U,
        pair.comb_ + row * 16U, launch.updated_streams[row], kHidden,
        launch.stream);
    if (!status.ok()) return status;
  }
  return Status::success();
}

}  // namespace expert::runtime::cuda
