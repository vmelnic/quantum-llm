#include "expert/runtime/cuda/deepseek_attention.hpp"

#include "expert/runtime/cuda/deepseek_hca.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kHeads = 64U;
constexpr std::uint32_t kHeadDim = 512U;
constexpr std::uint32_t kWindow = 128U;
constexpr std::uint32_t kIndexTopK = 512U;
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

std::size_t align_up(std::size_t value) {
  return (value + kAlignment - 1U) & ~(kAlignment - 1U);
}

std::uint64_t compressor_state_bytes(std::uint32_t ratio,
                                     std::uint32_t head_dim) {
  if (ratio == 0U) return 0U;
  const auto rows = ratio == 4U ? 8U : ratio;
  const auto width = ratio == 4U ? 2U * head_dim : head_dim;
  return 2ULL * rows * width * sizeof(float);
}

struct Arena final {
  std::byte* base{};
  std::size_t cursor{};

  template <typename T>
  T* take(std::size_t count) {
    cursor = align_up(cursor);
    auto* result = reinterpret_cast<T*>(base ? base + cursor : nullptr);
    cursor += count * sizeof(T);
    return result;
  }
};

}  // namespace

void DeepSeekAttentionState::map(void* raw_base) noexcept {
  auto* base = static_cast<std::byte*>(raw_base);
  Arena arena{base};
  const auto compressed = max_compressed_;
  const auto ratio = ratio_;
  const auto compressor_width = ratio == 4U ? 1024U :
                                ratio == 128U ? 512U : 0U;
  const auto selected = kWindow +
      (ratio == 4U ? std::min(kIndexTopK, compressed) : compressed);
  kv_cache_ = arena.take<std::uint16_t>(
      static_cast<std::size_t>(kWindow + compressed) * kHeadDim);
  index_cache_ = ratio == 4U
      ? arena.take<std::uint16_t>(static_cast<std::size_t>(compressed) * 128U)
      : nullptr;
  query_bf16_ = arena.take<std::uint16_t>(kHeads * kHeadDim);
  attention_bf16_ = arena.take<std::uint16_t>(kHeads * kHeadDim);
  index_query_bf16_ = ratio == 4U
      ? arena.take<std::uint16_t>(kHeads * 128U) : nullptr;
  indices_ = arena.take<std::int32_t>(selected);
  hca_normalized_ = arena.take<float>(4U * kHidden);
  hca_mixes_ = arena.take<float>(24U);
  collapsed_ = arena.take<float>(kHidden);
  attention_input_ = arena.take<float>(kHidden);
  pre_ = arena.take<float>(4U);
  post_ = arena.take<float>(4U);
  comb_ = arena.take<float>(16U);
  query_rank_ = arena.take<float>(1024U);
  query_norm_ = arena.take<float>(1024U);
  query_ = arena.take<float>(kHeads * kHeadDim);
  kv_ = arena.take<float>(kHeadDim);
  kv_norm_ = arena.take<float>(kHeadDim);
  attention_output_ = arena.take<float>(kHeads * kHeadDim);
  group_output_ = arena.take<float>(8192U);
  sublayer_ = arena.take<float>(kHidden);
  compress_values_ = ratio != 0U ? arena.take<float>(compressor_width) : nullptr;
  compress_scores_ = ratio != 0U ? arena.take<float>(compressor_width) : nullptr;
  compress_pooled_ = ratio != 0U ? arena.take<float>(kHeadDim) : nullptr;
  compress_output_ = ratio != 0U ? arena.take<float>(kHeadDim) : nullptr;
  if (ratio == 4U) {
    index_query_ = arena.take<float>(kHeads * 128U);
    index_head_weights_ = arena.take<float>(kHeads);
    index_scores_ = arena.take<float>(compressed);
    index_compress_values_ = arena.take<float>(256U);
    index_compress_scores_ = arena.take<float>(256U);
    index_compress_pooled_ = arena.take<float>(128U);
    index_compress_output_ = arena.take<float>(128U);
  }
  allocation_bytes_ = align_up(arena.cursor);
}

void DeepSeekAttentionPairWorkspace::map(void* raw_base) noexcept {
  Arena arena{static_cast<std::byte*>(raw_base)};
  constexpr std::size_t rows = 2U;
  selected_per_row_ = kWindow +
      std::max(kIndexTopK, max_context_ / 128U);
  index_scores_per_row_ = max_context_ / 4U;
  query_bf16_ = arena.take<std::uint16_t>(rows * kHeads * kHeadDim);
  attention_bf16_ = arena.take<std::uint16_t>(rows * kHeads * kHeadDim);
  index_query_bf16_ = arena.take<std::uint16_t>(rows * kHeads * 128U);
  indices_ = arena.take<std::int32_t>(rows * selected_per_row_);
  hca_normalized_ = arena.take<float>(rows * 4U * kHidden);
  hca_mixes_ = arena.take<float>(rows * 24U);
  collapsed_ = arena.take<float>(rows * kHidden);
  attention_input_ = arena.take<float>(rows * kHidden);
  pre_ = arena.take<float>(rows * 4U);
  post_ = arena.take<float>(rows * 4U);
  comb_ = arena.take<float>(rows * 16U);
  query_rank_ = arena.take<float>(rows * 1024U);
  query_norm_ = arena.take<float>(rows * 1024U);
  query_ = arena.take<float>(rows * kHeads * kHeadDim);
  kv_ = arena.take<float>(rows * kHeadDim);
  kv_norm_ = arena.take<float>(rows * kHeadDim);
  attention_output_ = arena.take<float>(rows * kHeads * kHeadDim);
  group_output_ = arena.take<float>(rows * 8192U);
  sublayer_ = arena.take<float>(rows * kHidden);
  compress_values_ = arena.take<float>(rows * 1024U);
  compress_scores_ = arena.take<float>(rows * 1024U);
  compress_pooled_ = arena.take<float>(rows * kHeadDim);
  compress_output_ = arena.take<float>(rows * kHeadDim);
  index_query_ = arena.take<float>(rows * kHeads * 128U);
  index_head_weights_ = arena.take<float>(rows * kHeads);
  index_scores_ = arena.take<float>(rows * index_scores_per_row_);
  index_compress_values_ = arena.take<float>(rows * 256U);
  index_compress_scores_ = arena.take<float>(rows * 256U);
  index_compress_pooled_ = arena.take<float>(rows * 128U);
  index_compress_output_ = arena.take<float>(rows * 128U);
  allocation_bytes_ = align_up(arena.cursor);
}

namespace {

__global__ void query_prepare_kernel(
    const float* input, const float* cosine, const float* sine,
    __nv_bfloat16* output, float epsilon) {
  __shared__ float inverse;
  __shared__ float values[kHeadDim];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto offset = static_cast<std::size_t>(head) * kHeadDim;
  values[dimension] = input[offset + dimension];
  __syncthreads();
  if (dimension == 0U) {
    float squares = 0.0F;
    for (unsigned index = 0; index < kHeadDim; ++index)
      squares += values[index] * values[index];
    inverse = rsqrtf(squares / static_cast<float>(kHeadDim) + epsilon);
  }
  __syncthreads();
  float value = values[dimension] * inverse;
  if (dimension >= 448U) {
    const auto local = dimension - 448U;
    const auto pair = local / 2U;
    const auto left = values[448U + pair * 2U] * inverse;
    const auto right = values[449U + pair * 2U] * inverse;
    value = local % 2U == 0U
        ? left * cosine[pair] - right * sine[pair]
        : right * cosine[pair] + left * sine[pair];
  }
  output[offset + dimension] = __float2bfloat16_rn(value);
}

__global__ void query_prepare_pair_kernel(
    const float* input, const float* cosine_zero, const float* sine_zero,
    const float* cosine_one, const float* sine_one,
    __nv_bfloat16* output, float epsilon) {
  __shared__ float inverse;
  __shared__ float values[kHeadDim];
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto offset =
      (static_cast<std::size_t>(row) * kHeads + head) * kHeadDim;
  values[dimension] = input[offset + dimension];
  __syncthreads();
  if (dimension == 0U) {
    float squares = 0.0F;
    for (unsigned index = 0; index < kHeadDim; ++index)
      squares += values[index] * values[index];
    inverse = rsqrtf(squares / static_cast<float>(kHeadDim) + epsilon);
  }
  __syncthreads();
  float value = values[dimension] * inverse;
  if (dimension >= 448U) {
    const auto* cosine = row == 0U ? cosine_zero : cosine_one;
    const auto* sine = row == 0U ? sine_zero : sine_one;
    const auto local = dimension - 448U;
    const auto pair = local / 2U;
    const auto left = values[448U + pair * 2U] * inverse;
    const auto right = values[449U + pair * 2U] * inverse;
    value = local % 2U == 0U
        ? left * cosine[pair] - right * sine[pair]
        : right * cosine[pair] + left * sine[pair];
  }
  output[offset + dimension] = __float2bfloat16_rn(value);
}

__global__ void attention_output_kernel(
    const __nv_bfloat16* input, const float* cosine, const float* sine,
    float* output) {
  __shared__ float rope[64];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto offset = static_cast<std::size_t>(head) * kHeadDim;
  if (dimension >= 448U)
    rope[dimension - 448U] = __bfloat162float(input[offset + dimension]);
  __syncthreads();
  float value = __bfloat162float(input[offset + dimension]);
  if (dimension >= 448U) {
    const auto local = dimension - 448U;
    const auto pair = local / 2U;
    const float left = rope[pair * 2U];
    const float right = rope[pair * 2U + 1U];
    value = local % 2U == 0U
        ? left * cosine[pair] + right * sine[pair]
        : right * cosine[pair] - left * sine[pair];
  }
  output[offset + dimension] = value;
}

__global__ void attention_output_pair_kernel(
    const __nv_bfloat16* input, const float* cosine_zero,
    const float* sine_zero, const float* cosine_one, const float* sine_one,
    float* output) {
  __shared__ float rope[64];
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto offset =
      (static_cast<std::size_t>(row) * kHeads + head) * kHeadDim;
  if (dimension >= 448U)
    rope[dimension - 448U] = __bfloat162float(input[offset + dimension]);
  __syncthreads();
  float value = __bfloat162float(input[offset + dimension]);
  if (dimension >= 448U) {
    const auto* cosine = row == 0U ? cosine_zero : cosine_one;
    const auto* sine = row == 0U ? sine_zero : sine_one;
    const auto local = dimension - 448U;
    const auto pair = local / 2U;
    const float left = rope[pair * 2U];
    const float right = rope[pair * 2U + 1U];
    value = local % 2U == 0U
        ? left * cosine[pair] + right * sine[pair]
        : right * cosine[pair] - left * sine[pair];
  }
  output[offset + dimension] = value;
}

__global__ void window_indices_kernel(std::uint32_t position,
                                      std::uint32_t count,
                                      std::int32_t* output) {
  const auto index = static_cast<std::uint32_t>(threadIdx.x);
  if (index >= count) return;
  if (position < kWindow)
    output[index] = static_cast<std::int32_t>(index);
  else {
    const auto newest = position % kWindow;
    output[index] = static_cast<std::int32_t>((newest + 1U + index) % kWindow);
  }
}

__global__ void compressed_indices_kernel(std::uint32_t count,
                                          std::int32_t* output) {
  const auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
  if (index < count) output[index] = static_cast<std::int32_t>(kWindow + index);
}

Status check_binding(const DeepSeekAttentionBinding& weights,
                     const DeepSeekAttentionState& state) {
  if (weights.compress_ratio != state.compress_ratio() ||
      !weights.wq_a.weights || !weights.wq_b.weights || !weights.wkv.weights ||
      !weights.wo_a.weights || !weights.wo_b.weights ||
      !weights.attention_norm || !weights.query_norm || !weights.kv_norm ||
      !weights.attention_sink || !weights.hca_function || !weights.hca_base ||
      !weights.hca_scale)
    return {ErrorCode::invalid_argument, "incomplete DeepSeek attention binding"};
  if (state.compress_ratio() != 0U &&
      (!weights.compressor_wkv || !weights.compressor_wgate ||
       !weights.compressor_ape || !weights.compressor_norm))
    return {ErrorCode::invalid_argument,
            "incomplete DeepSeek compressed attention binding"};
  if (state.compress_ratio() == 4U &&
      (!weights.index_wq_b.weights || !weights.index_weights ||
       !weights.index_compressor_wkv || !weights.index_compressor_wgate ||
       !weights.index_compressor_ape || !weights.index_compressor_norm))
    return {ErrorCode::invalid_argument, "incomplete DeepSeek index binding"};
  return Status::success();
}

}  // namespace

DeepSeekAttentionPairWorkspace::DeepSeekAttentionPairWorkspace(
    void* allocation, std::uint64_t allocation_bytes,
    std::uint32_t max_context) noexcept
    : allocation_(allocation), allocation_bytes_(allocation_bytes),
      max_context_(max_context) {}

DeepSeekAttentionPairWorkspace::~DeepSeekAttentionPairWorkspace() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

DeepSeekAttentionPairWorkspaceSize deepseek_attention_pair_workspace_size(
    std::uint32_t max_context) noexcept {
  if (max_context == 0U || max_context > 1'048'576U)
    return {{ErrorCode::invalid_argument,
             "unsupported DeepSeek attention pair context"}, 0U};
  DeepSeekAttentionPairWorkspace sizing(nullptr, 0U, max_context);
  sizing.map(nullptr);
  return {Status::success(), sizing.allocation_bytes_};
}

DeepSeekAttentionPairWorkspaceResult create_deepseek_attention_pair_workspace(
    std::uint32_t max_context) noexcept {
  const auto size = deepseek_attention_pair_workspace_size(max_context);
  if (!size.status.ok()) return {size.status, {}};
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, size.bytes);
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek attention pair workspace allocation"), {}};
  auto workspace = std::shared_ptr<DeepSeekAttentionPairWorkspace>(
      new DeepSeekAttentionPairWorkspace(allocation, size.bytes, max_context));
  workspace->map(allocation);
  error = cudaMemset(allocation, 0, size.bytes);
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek attention pair workspace reset"), {}};
  return {Status::success(), std::move(workspace)};
}

DeepSeekAttentionState::DeepSeekAttentionState(
    void* allocation, std::uint64_t allocation_bytes, std::uint32_t ratio,
    std::uint32_t max_context, std::uint32_t max_compressed) noexcept
    : allocation_(allocation), allocation_bytes_(allocation_bytes), ratio_(ratio),
      max_context_(max_context), max_compressed_(max_compressed) {}

DeepSeekAttentionState::~DeepSeekAttentionState() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

std::uint64_t DeepSeekAttentionState::bytes() const noexcept {
  return allocation_bytes_ + (compressor_ ? compressor_->bytes() : 0U) +
         (index_compressor_ ? index_compressor_->bytes() : 0U);
}

std::uint64_t DeepSeekAttentionState::speculative_checkpoint_bytes()
    const noexcept {
  if (ratio_ != 4U || !compressor_ || !index_compressor_) return 0U;
  return compressor_->bytes() + index_compressor_->bytes();
}

Status DeepSeekAttentionState::checkpoint_speculative_state(
    void* destination, void* stream) const noexcept {
  const auto bytes = speculative_checkpoint_bytes();
  if (bytes == 0U) return Status::success();
  if (!destination)
    return {ErrorCode::invalid_argument,
            "missing DeepSeek speculative checkpoint destination"};
  auto* cursor = static_cast<std::byte*>(destination);
  const auto copy = [&](const DeepSeekCompressorState& state) -> Status {
    const auto half = state.bytes() / 2U;
    auto error = cudaMemcpyAsync(cursor, state.values(), half,
                                 cudaMemcpyDeviceToDevice,
                                 static_cast<cudaStream_t>(stream));
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(cursor + half, state.scores(), half,
                              cudaMemcpyDeviceToDevice,
                              static_cast<cudaStream_t>(stream));
    if (error != cudaSuccess)
      return failure(error, "checkpoint DeepSeek compressor state");
    cursor += state.bytes();
    return Status::success();
  };
  auto status = copy(*compressor_);
  return status.ok() ? copy(*index_compressor_) : status;
}

Status DeepSeekAttentionState::restore_speculative_state(
    const void* source, void* stream) noexcept {
  const auto bytes = speculative_checkpoint_bytes();
  if (bytes == 0U) return Status::success();
  if (!source)
    return {ErrorCode::invalid_argument,
            "missing DeepSeek speculative checkpoint source"};
  auto* cursor = static_cast<const std::byte*>(source);
  const auto copy = [&](DeepSeekCompressorState& state) -> Status {
    const auto half = state.bytes() / 2U;
    auto error = cudaMemcpyAsync(state.values(), cursor, half,
                                 cudaMemcpyDeviceToDevice,
                                 static_cast<cudaStream_t>(stream));
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(state.scores(), cursor + half, half,
                              cudaMemcpyDeviceToDevice,
                              static_cast<cudaStream_t>(stream));
    if (error != cudaSuccess)
      return failure(error, "restore DeepSeek compressor state");
    cursor += state.bytes();
    return Status::success();
  };
  auto status = copy(*compressor_);
  return status.ok() ? copy(*index_compressor_) : status;
}

Status DeepSeekAttentionState::checkpoint_retention_state(
    void* host_destination, void* stream) const noexcept {
  const auto bytes = retention_checkpoint_bytes();
  if (bytes == 0U) return Status::success();
  if (!host_destination)
    return {ErrorCode::invalid_argument,
            "missing DeepSeek retention checkpoint destination"};
  auto* cursor = static_cast<std::byte*>(host_destination);
  const auto copy = [&](const DeepSeekCompressorState& state) -> Status {
    const auto half = state.bytes() / 2U;
    auto error = cudaMemcpyAsync(cursor, state.values(), half,
                                 cudaMemcpyDeviceToHost,
                                 static_cast<cudaStream_t>(stream));
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(cursor + half, state.scores(), half,
                              cudaMemcpyDeviceToHost,
                              static_cast<cudaStream_t>(stream));
    if (error != cudaSuccess)
      return failure(error, "checkpoint retained DeepSeek compressor state");
    cursor += state.bytes();
    return Status::success();
  };
  auto status = copy(*compressor_);
  return status.ok() ? copy(*index_compressor_) : status;
}

Status DeepSeekAttentionState::restore_retention_state(
    const void* host_source, void* stream) noexcept {
  const auto bytes = retention_checkpoint_bytes();
  if (bytes == 0U) return Status::success();
  if (!host_source)
    return {ErrorCode::invalid_argument,
            "missing DeepSeek retention checkpoint source"};
  auto* cursor = static_cast<const std::byte*>(host_source);
  const auto copy = [&](DeepSeekCompressorState& state) -> Status {
    const auto half = state.bytes() / 2U;
    auto error = cudaMemcpyAsync(state.values(), cursor, half,
                                 cudaMemcpyHostToDevice,
                                 static_cast<cudaStream_t>(stream));
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(state.scores(), cursor + half, half,
                              cudaMemcpyHostToDevice,
                              static_cast<cudaStream_t>(stream));
    if (error != cudaSuccess)
      return failure(error, "restore retained DeepSeek compressor state");
    cursor += state.bytes();
    return Status::success();
  };
  auto status = copy(*compressor_);
  return status.ok() ? copy(*index_compressor_) : status;
}

DeepSeekAttentionStateResult create_deepseek_attention_state(
    std::uint32_t ratio, std::uint32_t max_context) noexcept {
  const auto size = deepseek_attention_state_size(ratio, max_context);
  if (!size.status.ok()) return {size.status, {}};
  const auto compressed = ratio == 0U ? 0U : max_context / ratio;
  DeepSeekAttentionState sizing(nullptr, 0U, ratio, max_context, compressed);
  sizing.map(nullptr);
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, sizing.allocation_bytes_);
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek attention state allocation"), {}};
  auto state = std::shared_ptr<DeepSeekAttentionState>(
      new DeepSeekAttentionState(allocation, 0U, ratio, max_context,
                                 compressed));
  state->map(allocation);
  error = cudaMemset(allocation, 0, state->allocation_bytes_);
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek attention state reset"), {}};
  if (ratio != 0U) {
    auto compressor = create_deepseek_compressor_state(ratio, 512U);
    if (!compressor.status.ok()) return {compressor.status, {}};
    state->compressor_ = std::move(compressor.state);
  }
  if (ratio == 4U) {
    auto index = create_deepseek_compressor_state(4U, 128U);
    if (!index.status.ok()) return {index.status, {}};
    state->index_compressor_ = std::move(index.state);
  }
  return {Status::success(), std::move(state)};
}

DeepSeekAttentionStateSize deepseek_attention_state_size(
    std::uint32_t ratio, std::uint32_t max_context) noexcept {
  if ((ratio != 0U && ratio != 4U && ratio != 128U) || max_context == 0U ||
      max_context > 1'048'576U)
    return {{ErrorCode::invalid_argument,
             "unsupported DeepSeek attention state geometry"}, 0U};
  const auto compressed = ratio == 0U ? 0U : max_context / ratio;
  DeepSeekAttentionState sizing(nullptr, 0U, ratio, max_context, compressed);
  sizing.map(nullptr);
  const auto compressor_bytes = compressor_state_bytes(ratio, kHeadDim) +
      (ratio == 4U ? compressor_state_bytes(4U, 128U) : 0U);
  return {Status::success(), sizing.allocation_bytes_ + compressor_bytes};
}

std::uint64_t deepseek_attention_speculative_checkpoint_size(
    std::uint32_t ratio) noexcept {
  if (ratio != 4U) return 0U;
  return compressor_state_bytes(4U, 512U) +
         compressor_state_bytes(4U, 128U);
}

Status deepseek_attention_decode(const DeepSeekAttentionLaunch& launch) noexcept {
  if (!launch.weights || !launch.state || !launch.streams ||
      !launch.updated_streams || !launch.cosine || !launch.sine ||
      launch.epsilon <= 0.0F || launch.sinkhorn_iterations == 0U ||
      launch.position >= launch.state->max_context_tokens())
    return {ErrorCode::invalid_argument, "invalid DeepSeek attention launch"};
  auto status = check_binding(*launch.weights, *launch.state);
  if (!status.ok()) return status;
  auto& state = *launch.state;
  const auto& weights = *launch.weights;
  const bool emits = state.ratio_ != 0U &&
                     (launch.position + 1U) % state.ratio_ == 0U;
  if (emits && (!launch.compressed_cosine || !launch.compressed_sine))
    return {ErrorCode::invalid_argument, "missing compressed-position RoPE"};
  auto raw_stream = static_cast<cudaStream_t>(launch.stream);

  status = deepseek_hca_pre(
      {weights.hca_function, weights.hca_base, weights.hca_scale, kHidden},
      launch.streams, state.collapsed_, state.pre_, state.post_, state.comb_,
      {state.hca_normalized_, state.hca_mixes_}, launch.epsilon,
      launch.sinkhorn_iterations, launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight(state.collapsed_, weights.attention_norm,
                                state.attention_input_, kHidden,
                                launch.epsilon, launch.stream);
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->hca_pre_norm_stop : nullptr,
      raw_stream, "record DeepSeek HCA pre/norm stop");
  if (!status.ok()) return status;
  status = gemv(weights.wq_a, state.attention_input_, state.query_rank_,
                launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight(state.query_rank_, weights.query_norm,
                                state.query_norm_, 1024U, launch.epsilon,
                                launch.stream);
  if (!status.ok()) return status;
  status = gemv(weights.wq_b, state.query_norm_, state.query_, launch.stream);
  if (!status.ok()) return status;
  query_prepare_kernel<<<kHeads, kHeadDim, 0, raw_stream>>>(
      state.query_, launch.cosine, launch.sine,
      reinterpret_cast<__nv_bfloat16*>(state.query_bf16_), launch.epsilon);

  status = gemv(weights.wkv, state.attention_input_, state.kv_, launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight(state.kv_, weights.kv_norm, state.kv_norm_,
                                kHeadDim, launch.epsilon, launch.stream);
  if (!status.ok()) return status;
  status = deepseek_compressed_kv_publish(
      state.kv_norm_, launch.cosine, launch.sine, state.kv_cache_,
      launch.position % kWindow, launch.stream);
  if (!status.ok()) return status;

  std::uint32_t compressed_count = 0U;
  if (state.ratio_ != 0U) {
    const auto width = state.ratio_ == 4U ? 1024U : 512U;
    status = gemv_bf16(weights.compressor_wkv, width, kHidden,
                       state.attention_input_, state.compress_values_,
                       launch.stream);
    if (!status.ok()) return status;
    status = gemv_bf16(weights.compressor_wgate, width, kHidden,
                       state.attention_input_, state.compress_scores_,
                       launch.stream);
    if (!status.ok()) return status;
    status = deepseek_compressor_decode(
        *state.compressor_, state.compress_values_, state.compress_scores_,
        weights.compressor_ape, weights.compressor_norm,
        state.compress_pooled_, state.compress_output_, launch.position,
        launch.epsilon, launch.stream);
    if (!status.ok()) return status;
    compressed_count = (launch.position + 1U) / state.ratio_;
    if (emits) {
      status = deepseek_compressed_kv_publish(
          state.compress_output_, launch.compressed_cosine,
          launch.compressed_sine, state.kv_cache_,
          kWindow + compressed_count - 1U, launch.stream);
      if (!status.ok()) return status;
    }
  }

  const auto window_count = std::min(launch.position + 1U, kWindow);
  window_indices_kernel<<<1, kWindow, 0, raw_stream>>>(
      launch.position, window_count, state.indices_);
  std::uint32_t compressed_selected = compressed_count;
  if (state.ratio_ == 4U) {
    status = gemv(weights.index_wq_b, state.query_norm_, state.index_query_,
                  launch.stream);
    if (!status.ok()) return status;
    status = deepseek_index_prepare(state.index_query_, launch.cosine,
                                    launch.sine, state.index_query_bf16_,
                                    kHeads, launch.stream);
    if (!status.ok()) return status;
    status = gemv_bf16(weights.index_compressor_wkv, 256U, kHidden,
                       state.attention_input_, state.index_compress_values_,
                       launch.stream);
    if (!status.ok()) return status;
    status = gemv_bf16(weights.index_compressor_wgate, 256U, kHidden,
                       state.attention_input_, state.index_compress_scores_,
                       launch.stream);
    if (!status.ok()) return status;
    status = deepseek_compressor_decode(
        *state.index_compressor_, state.index_compress_values_,
        state.index_compress_scores_, weights.index_compressor_ape,
        weights.index_compressor_norm, state.index_compress_pooled_,
        state.index_compress_output_, launch.position, launch.epsilon,
        launch.stream);
    if (!status.ok()) return status;
    if (emits) {
      status = deepseek_index_prepare(
          state.index_compress_output_, launch.compressed_cosine,
          launch.compressed_sine,
          state.index_cache_ + static_cast<std::size_t>(compressed_count - 1U) * 128U,
          1U, launch.stream);
      if (!status.ok()) return status;
    }
    compressed_selected = std::min(kIndexTopK, compressed_count);
    if (compressed_selected != 0U) {
      status = gemv_bf16(weights.index_weights, kHeads, kHidden,
                         state.attention_input_, state.index_head_weights_,
                         launch.stream);
      if (!status.ok()) return status;
      status = deepseek_index_topk(
          state.index_query_bf16_, state.index_cache_,
          state.index_head_weights_, compressed_count, compressed_selected,
          state.index_scores_, state.indices_ + window_count, launch.stream,
          kWindow);
      if (!status.ok()) return status;
    }
  } else if (compressed_count != 0U) {
    compressed_indices_kernel<<<(compressed_count + 255U) / 256U, 256U, 0,
                                  raw_stream>>>(
        compressed_count, state.indices_ + window_count);
  }
  auto error = cudaPeekAtLastError();
  if (error != cudaSuccess) return failure(error, "DeepSeek attention preparation");
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->projection_stop : nullptr,
      raw_stream, "record DeepSeek projection stop");
  if (!status.ok()) return status;

  status = deepseek_sparse_attention_decode(
      state.query_bf16_, state.kv_cache_, state.indices_,
      window_count + compressed_selected, weights.attention_sink,
      state.attention_bf16_, kHeads, launch.stream);
  if (!status.ok()) return status;
  attention_output_kernel<<<kHeads, kHeadDim, 0, raw_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(state.attention_bf16_),
      launch.cosine, launch.sine, state.attention_output_);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess)
    return failure(error, "DeepSeek attention output preparation");
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->sparse_attention_stop
                            : nullptr,
      raw_stream, "record DeepSeek sparse attention stop");
  if (!status.ok()) return status;
  status = gemv_grouped_inputs(weights.wo_a, state.attention_output_,
                               state.group_output_, 8U, launch.stream);
  if (!status.ok()) return status;
  status = gemv(weights.wo_b, state.group_output_, state.sublayer_,
                launch.stream);
  if (!status.ok()) return status;
  status = record_profile_event(
      launch.profile_events ? launch.profile_events->output_projection_stop
                            : nullptr,
      raw_stream, "record DeepSeek attention output projection stop");
  if (!status.ok()) return status;
  return deepseek_hca_post(state.sublayer_, launch.streams, state.post_,
                           state.comb_, launch.updated_streams, kHidden,
                           launch.stream);
}

Status deepseek_attention_decode_pair(
    const DeepSeekAttentionPairLaunch& launch) noexcept {
  if (!launch.weights || !launch.state || !launch.workspace ||
      !launch.streams[0] || !launch.streams[1] ||
      !launch.updated_streams[0] || !launch.updated_streams[1] ||
      !launch.cosine[0] || !launch.cosine[1] ||
      !launch.sine[0] || !launch.sine[1] || launch.epsilon <= 0.0F ||
      launch.sinkhorn_iterations == 0U ||
      launch.positions[1] != launch.positions[0] + 1U ||
      launch.positions[1] >= launch.state->max_context_tokens() ||
      launch.workspace->max_context_tokens() <
          launch.state->max_context_tokens())
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek attention pair launch"};
  auto status = check_binding(*launch.weights, *launch.state);
  if (!status.ok()) return status;
  auto& state = *launch.state;
  auto& workspace = *launch.workspace;
  const auto& weights = *launch.weights;
  const auto raw_stream = static_cast<cudaStream_t>(launch.stream);
  bool checkpoint_written = false;
  const auto fail_after_checkpoint = [&](Status problem) -> Status {
    if (!checkpoint_written) return problem;
    const auto restored = state.restore_speculative_state(
        launch.speculative_checkpoint, launch.stream);
    return restored.ok() ? problem : restored;
  };

  for (std::uint32_t row = 0U; row < 2U; ++row) {
    const bool emits = state.ratio_ != 0U &&
        (launch.positions[row] + 1U) % state.ratio_ == 0U;
    if (emits &&
        (!launch.compressed_cosine[row] || !launch.compressed_sine[row]))
      return {ErrorCode::invalid_argument,
              "missing DeepSeek pair compressed-position RoPE"};
  }
  const bool row_one_recurrent_boundary = state.ratio_ == 4U &&
      (launch.positions[1] + 1U) % 4U == 0U;
  if (row_one_recurrent_boundary && !launch.speculative_checkpoint)
    return {ErrorCode::invalid_argument,
            "missing DeepSeek pair speculative checkpoint"};

  // HCA control and all large dense projections consume contiguous
  // [2, columns] activations and read each weight row once.
  status = deepseek_hca_pre_pair(
      {weights.hca_function, weights.hca_base, weights.hca_scale, kHidden},
      {launch.streams[0], launch.streams[1]}, workspace.collapsed_,
      workspace.pre_, workspace.post_, workspace.comb_,
      {workspace.hca_normalized_, workspace.hca_mixes_}, launch.epsilon,
      launch.sinkhorn_iterations, launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight_batch(
      workspace.collapsed_, weights.attention_norm,
      workspace.attention_input_, 2U, kHidden, launch.epsilon,
      launch.stream);
  if (!status.ok()) return status;
  status = gemv_batch_weight_reuse(
      weights.wq_a, workspace.attention_input_, workspace.query_rank_, 2U,
      launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight_batch(
      workspace.query_rank_, weights.query_norm, workspace.query_norm_, 2U,
      1024U, launch.epsilon, launch.stream);
  if (!status.ok()) return status;
  status = gemv_batch_weight_reuse(
      weights.wq_b, workspace.query_norm_, workspace.query_, 2U,
      launch.stream);
  if (!status.ok()) return status;
  query_prepare_pair_kernel<<<dim3(kHeads, 2U), kHeadDim, 0, raw_stream>>>(
      workspace.query_, launch.cosine[0], launch.sine[0], launch.cosine[1],
      launch.sine[1],
      reinterpret_cast<__nv_bfloat16*>(workspace.query_bf16_),
      launch.epsilon);
  status = gemv_batch_weight_reuse(
      weights.wkv, workspace.attention_input_, workspace.kv_, 2U,
      launch.stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight_batch(
      workspace.kv_, weights.kv_norm, workspace.kv_norm_, 2U, kHeadDim,
      launch.epsilon, launch.stream);
  if (!status.ok()) return status;

  if (state.ratio_ != 0U) {
    const auto width = state.ratio_ == 4U ? 1024U : 512U;
    status = gemv_bf16_batch(
        weights.compressor_wkv, width, kHidden, workspace.attention_input_,
        workspace.compress_values_, 2U, launch.stream);
    if (!status.ok()) return status;
    status = gemv_bf16_batch(
        weights.compressor_wgate, width, kHidden, workspace.attention_input_,
        workspace.compress_scores_, 2U, launch.stream);
    if (!status.ok()) return status;
  }
  if (state.ratio_ == 4U) {
    status = gemv_batch_weight_reuse(
        weights.index_wq_b, workspace.query_norm_, workspace.index_query_,
        2U, launch.stream);
    if (!status.ok()) return status;
    status = gemv_bf16_batch(
        weights.index_compressor_wkv, 256U, kHidden,
        workspace.attention_input_, workspace.index_compress_values_, 2U,
        launch.stream);
    if (!status.ok()) return status;
    status = gemv_bf16_batch(
        weights.index_compressor_wgate, 256U, kHidden,
        workspace.attention_input_, workspace.index_compress_scores_, 2U,
        launch.stream);
    if (!status.ok()) return status;
    status = gemv_bf16_batch(
        weights.index_weights, kHeads, kHidden, workspace.attention_input_,
        workspace.index_head_weights_, 2U, launch.stream);
    if (!status.ok()) return status;
  }
  auto error = cudaPeekAtLastError();
  if (error != cudaSuccess)
    return failure(error, "DeepSeek pair projection preparation");

  const auto execute_causal_row = [&](std::uint32_t row) -> Status {
    const auto position = launch.positions[row];
    const bool emits = state.ratio_ != 0U &&
        (position + 1U) % state.ratio_ == 0U;
    auto row_status = deepseek_compressed_kv_publish(
        workspace.kv_norm_ + row * kHeadDim, launch.cosine[row],
        launch.sine[row], state.kv_cache_, position % kWindow,
        launch.stream);
    if (!row_status.ok()) return row_status;

    std::uint32_t compressed_count = 0U;
    if (state.ratio_ != 0U) {
      const auto width = state.ratio_ == 4U ? 1024U : 512U;
      row_status = deepseek_compressor_decode(
          *state.compressor_, workspace.compress_values_ + row * width,
          workspace.compress_scores_ + row * width, weights.compressor_ape,
          weights.compressor_norm,
          workspace.compress_pooled_ + row * kHeadDim,
          workspace.compress_output_ + row * kHeadDim, position,
          launch.epsilon, launch.stream);
      if (!row_status.ok()) return row_status;
      compressed_count = (position + 1U) / state.ratio_;
      if (emits) {
        row_status = deepseek_compressed_kv_publish(
            workspace.compress_output_ + row * kHeadDim,
            launch.compressed_cosine[row], launch.compressed_sine[row],
            state.kv_cache_, kWindow + compressed_count - 1U,
            launch.stream);
        if (!row_status.ok()) return row_status;
      }
    }

    const auto window_count = std::min(position + 1U, kWindow);
    auto* row_indices = workspace.indices_ +
        static_cast<std::size_t>(row) * workspace.selected_per_row_;
    window_indices_kernel<<<1, kWindow, 0, raw_stream>>>(
        position, window_count, row_indices);
    std::uint32_t compressed_selected = compressed_count;
    if (state.ratio_ == 4U) {
      row_status = deepseek_index_prepare(
          workspace.index_query_ + row * kHeads * 128U,
          launch.cosine[row], launch.sine[row],
          workspace.index_query_bf16_ + row * kHeads * 128U, kHeads,
          launch.stream);
      if (!row_status.ok()) return row_status;
      row_status = deepseek_compressor_decode(
          *state.index_compressor_,
          workspace.index_compress_values_ + row * 256U,
          workspace.index_compress_scores_ + row * 256U,
          weights.index_compressor_ape, weights.index_compressor_norm,
          workspace.index_compress_pooled_ + row * 128U,
          workspace.index_compress_output_ + row * 128U, position,
          launch.epsilon, launch.stream);
      if (!row_status.ok()) return row_status;
      if (emits) {
        row_status = deepseek_index_prepare(
            workspace.index_compress_output_ + row * 128U,
            launch.compressed_cosine[row], launch.compressed_sine[row],
            state.index_cache_ +
                static_cast<std::size_t>(compressed_count - 1U) * 128U,
            1U, launch.stream);
        if (!row_status.ok()) return row_status;
      }
      compressed_selected = std::min(kIndexTopK, compressed_count);
      if (compressed_selected != 0U) {
        row_status = deepseek_index_topk(
            workspace.index_query_bf16_ + row * kHeads * 128U,
            state.index_cache_, workspace.index_head_weights_ + row * kHeads,
            compressed_count, compressed_selected,
            workspace.index_scores_ +
                static_cast<std::size_t>(row) *
                    workspace.index_scores_per_row_,
            row_indices + window_count, launch.stream, kWindow);
        if (!row_status.ok()) return row_status;
      }
    } else if (compressed_count != 0U) {
      compressed_indices_kernel<<<(compressed_count + 255U) / 256U, 256U, 0,
                                    raw_stream>>>(
          compressed_count, row_indices + window_count);
    }
    error = cudaPeekAtLastError();
    if (error != cudaSuccess)
      return failure(error, "DeepSeek pair attention preparation");
    return deepseek_sparse_attention_decode(
        workspace.query_bf16_ + row * kHeads * kHeadDim, state.kv_cache_,
        row_indices, window_count + compressed_selected,
        weights.attention_sink,
        workspace.attention_bf16_ + row * kHeads * kHeadDim, kHeads,
        launch.stream);
  };

  status = execute_causal_row(0U);
  if (!status.ok()) return status;
  if (row_one_recurrent_boundary) {
    status = state.checkpoint_speculative_state(
        launch.speculative_checkpoint, launch.stream);
    if (!status.ok()) return status;
    checkpoint_written = true;
  }
  status = execute_causal_row(1U);
  if (!status.ok()) return fail_after_checkpoint(status);

  attention_output_pair_kernel<<<dim3(kHeads, 2U), kHeadDim, 0, raw_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(workspace.attention_bf16_),
      launch.cosine[0], launch.sine[0], launch.cosine[1], launch.sine[1],
      workspace.attention_output_);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess)
    return fail_after_checkpoint(
        failure(error, "DeepSeek pair attention output preparation"));
  status = gemv_grouped_inputs_batch_weight_reuse(
      weights.wo_a, workspace.attention_output_, workspace.group_output_, 8U,
      2U, launch.stream);
  if (!status.ok()) return fail_after_checkpoint(status);
  status = gemv_batch_weight_reuse(
      weights.wo_b, workspace.group_output_, workspace.sublayer_, 2U,
      launch.stream);
  if (!status.ok()) return fail_after_checkpoint(status);
  for (std::uint32_t row = 0U; row < 2U; ++row) {
    status = deepseek_hca_post(
        workspace.sublayer_ + row * kHidden, launch.streams[row],
        workspace.post_ + row * 4U, workspace.comb_ + row * 16U,
        launch.updated_streams[row], kHidden, launch.stream);
    if (!status.ok()) return fail_after_checkpoint(status);
  }
  return Status::success();
}

}  // namespace expert::runtime::cuda
