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

std::size_t align_up(std::size_t value) {
  return (value + kAlignment - 1U) & ~(kAlignment - 1U);
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
  const auto compressor_width = ratio == 4U ? 1024U : 512U;
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
  compress_values_ = arena.take<float>(compressor_width);
  compress_scores_ = arena.take<float>(compressor_width);
  compress_pooled_ = arena.take<float>(kHeadDim);
  compress_output_ = arena.take<float>(kHeadDim);
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
      !weights.hca_scale || !weights.compressor_wkv ||
      !weights.compressor_wgate || !weights.compressor_ape ||
      !weights.compressor_norm)
    return {ErrorCode::invalid_argument, "incomplete DeepSeek attention binding"};
  if (state.compress_ratio() == 4U &&
      (!weights.index_wq_b.weights || !weights.index_weights ||
       !weights.index_compressor_wkv || !weights.index_compressor_wgate ||
       !weights.index_compressor_ape || !weights.index_compressor_norm))
    return {ErrorCode::invalid_argument, "incomplete DeepSeek index binding"};
  return Status::success();
}

}  // namespace

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

DeepSeekAttentionStateResult create_deepseek_attention_state(
    std::uint32_t ratio, std::uint32_t max_context) noexcept {
  if ((ratio != 4U && ratio != 128U) || max_context == 0U ||
      max_context > 1'048'576U)
    return {{ErrorCode::invalid_argument,
             "unsupported DeepSeek attention state geometry"}, {}};
  const auto compressed = max_context / ratio;
  DeepSeekAttentionState sizing(nullptr, 0U, ratio, max_context, compressed);
  sizing.map(nullptr);
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, sizing.allocation_bytes_);
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek attention state allocation"), {}};
  auto state = std::shared_ptr<DeepSeekAttentionState>(
      new DeepSeekAttentionState(allocation, sizing.allocation_bytes_, ratio,
                                 max_context, compressed));
  state->map(allocation);
  error = cudaMemset(allocation, 0, state->allocation_bytes_);
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek attention state reset"), {}};
  auto compressor = create_deepseek_compressor_state(ratio, 512U);
  if (!compressor.status.ok()) return {compressor.status, {}};
  state->compressor_ = std::move(compressor.state);
  if (ratio == 4U) {
    auto index = create_deepseek_compressor_state(4U, 128U);
    if (!index.status.ok()) return {index.status, {}};
    state->index_compressor_ = std::move(index.state);
  }
  return {Status::success(), std::move(state)};
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
  const bool emits = (launch.position + 1U) % state.ratio_ == 0U;
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
      weights.compressor_ape, weights.compressor_norm, state.compress_pooled_,
      state.compress_output_, launch.position, launch.epsilon, launch.stream);
  if (!status.ok()) return status;
  const auto compressed_count = (launch.position + 1U) / state.ratio_;
  if (emits) {
    status = deepseek_compressed_kv_publish(
        state.compress_output_, launch.compressed_cosine,
        launch.compressed_sine, state.kv_cache_,
        kWindow + compressed_count - 1U, launch.stream);
    if (!status.ok()) return status;
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

  status = deepseek_sparse_attention_decode(
      state.query_bf16_, state.kv_cache_, state.indices_,
      window_count + compressed_selected, weights.attention_sink,
      state.attention_bf16_, kHeads, launch.stream);
  if (!status.ok()) return status;
  attention_output_kernel<<<kHeads, kHeadDim, 0, raw_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(state.attention_bf16_),
      launch.cosine, launch.sine, state.attention_output_);
  status = gemv_grouped_inputs(weights.wo_a, state.attention_output_,
                               state.group_output_, 8U, launch.stream);
  if (!status.ok()) return status;
  status = gemv(weights.wo_b, state.group_output_, state.sublayer_,
                launch.stream);
  if (!status.ok()) return status;
  return deepseek_hca_post(state.sublayer_, launch.streams, state.post_,
                           state.comb_, launch.updated_streams, kHidden,
                           launch.stream);
}

}  // namespace expert::runtime::cuda
