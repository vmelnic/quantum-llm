#include "expert/runtime/cuda/deepseek_csa.hpp"

#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::uint32_t kHeadDim = 512U;
constexpr float kNegativeInfinity = -3.402823466e+38F;

Status failure(cudaError_t error, const char* operation) {
  return {ErrorCode::upload_failed,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

__global__ void fill_kernel(float* values, std::uint32_t count, float value) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) values[index] = value;
}

__global__ void compressor_store_kernel(
    const float* values, const float* scores, const float* ape,
    float* value_state, float* score_state, std::uint32_t state_row,
    std::uint32_t ape_row, std::uint32_t width) {
  const auto column =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (column >= width) return;
  const auto destination = static_cast<std::size_t>(state_row) * width + column;
  value_state[destination] = values[column];
  score_state[destination] =
      scores[column] + ape[static_cast<std::size_t>(ape_row) * width + column];
}

__global__ void compressor_pool_kernel(
    const float* values, const float* scores, float* output,
    std::uint32_t ratio, std::uint32_t width) {
  const auto dimension = static_cast<std::uint32_t>(blockIdx.x);
  if (dimension >= kHeadDim || threadIdx.x != 0U) return;
  const auto candidates = ratio == 4U ? 8U : ratio;
  float maximum = kNegativeInfinity;
  for (std::uint32_t row = 0; row < candidates; ++row) {
    const auto source_row = row;
    const auto source_column =
        ratio == 4U && row >= 4U ? kHeadDim + dimension : dimension;
    maximum = fmaxf(maximum,
                    scores[static_cast<std::size_t>(source_row) * width +
                           source_column]);
  }
  float denominator = 0.0F;
  float result = 0.0F;
  for (std::uint32_t row = 0; row < candidates; ++row) {
    const auto source_column =
        ratio == 4U && row >= 4U ? kHeadDim + dimension : dimension;
    const auto source = static_cast<std::size_t>(row) * width + source_column;
    const float weight = expf(scores[source] - maximum);
    denominator += weight;
    result += weight * values[source];
  }
  output[dimension] = result / denominator;
}

__global__ void compressor_overlap_advance_kernel(float* values, float* scores,
                                                   std::uint32_t width) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const auto count = 4U * width;
  if (index >= count) return;
  values[index] = values[count + index];
  scores[index] = scores[count + index];
}

__global__ void compressed_kv_publish_kernel(
    const float* normalized, const float* cosine, const float* sine,
    __nv_bfloat16* cache, std::uint32_t slot) {
  __shared__ float base[kHeadDim];
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (dimension < kHeadDim)
    base[dimension] = __bfloat162float(__float2bfloat16_rn(normalized[dimension]));
  __syncthreads();
  if (dimension < 448U) {
    const auto group = dimension / 64U;
    float maximum = 0.0F;
    for (std::uint32_t item = group * 64U; item < (group + 1U) * 64U; ++item)
      maximum = fmaxf(maximum, fabsf(base[item]));
    maximum = fmaxf(maximum, 1.0e-4F);
    const float scale = exp2f(ceilf(log2f(maximum / 448.0F)));
    const __nv_fp8_e4m3 quantized(base[dimension] / scale);
    const float restored = static_cast<float>(quantized) * scale;
    cache[static_cast<std::size_t>(slot) * kHeadDim + dimension] =
        __float2bfloat16_rn(restored);
  } else if (dimension < kHeadDim) {
    const auto local = dimension - 448U;
    const auto pair = local / 2U;
    const auto left_index = 448U + pair * 2U;
    const float left = base[left_index];
    const float right = base[left_index + 1U];
    const float rotated = local % 2U == 0U
                              ? left * cosine[pair] - right * sine[pair]
                              : right * cosine[pair] + left * sine[pair];
    cache[static_cast<std::size_t>(slot) * kHeadDim + dimension] =
        __float2bfloat16_rn(rotated);
  }
}

__device__ float block_sum(float value) {
  __shared__ float partial[kThreads];
  partial[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride; stride >>= 1U) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  return partial[0];
}

__global__ void sparse_attention_decode_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* cache,
    const std::int32_t* indices, std::uint32_t selected,
    const float* sink, __nv_bfloat16* output) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  const auto q_base = static_cast<std::size_t>(head) * kHeadDim;
  float accum0 = 0.0F;
  float accum1 = 0.0F;
  __shared__ float maximum;
  __shared__ float denominator;
  __shared__ float previous_scale;
  __shared__ float token_weight;
  if (lane == 0U) {
    maximum = sink[head];
    denominator = 1.0F;
  }
  __syncthreads();
  for (std::uint32_t item = 0; item < selected; ++item) {
    const auto slot = indices[item];
    float partial = 0.0F;
    if (slot >= 0) {
      partial += __bfloat162float(query[q_base + lane]) *
                 __bfloat162float(cache[static_cast<std::size_t>(slot) * kHeadDim + lane]);
      partial += __bfloat162float(query[q_base + lane + kThreads]) *
                 __bfloat162float(cache[static_cast<std::size_t>(slot) * kHeadDim + lane + kThreads]);
    }
    const float dot = block_sum(partial);
    if (lane == 0U) {
      if (slot >= 0) {
        const float score = dot * rsqrtf(static_cast<float>(kHeadDim));
        const float next_maximum = fmaxf(maximum, score);
        previous_scale = expf(maximum - next_maximum);
        token_weight = expf(score - next_maximum);
        denominator = denominator * previous_scale + token_weight;
        maximum = next_maximum;
      } else {
        previous_scale = 1.0F;
        token_weight = 0.0F;
      }
    }
    __syncthreads();
    if (slot >= 0) {
      const auto cache_base = static_cast<std::size_t>(slot) * kHeadDim;
      accum0 = accum0 * previous_scale +
               token_weight * __bfloat162float(cache[cache_base + lane]);
      accum1 = accum1 * previous_scale +
               token_weight * __bfloat162float(cache[cache_base + lane + kThreads]);
    }
    __syncthreads();
  }
  output[q_base + lane] = __float2bfloat16_rn(accum0 / denominator);
  output[q_base + lane + kThreads] =
      __float2bfloat16_rn(accum1 / denominator);
}

}  // namespace

DeepSeekCompressorState::DeepSeekCompressorState(
    float* values, float* scores, std::uint32_t ratio,
    std::uint32_t projected_width) noexcept
    : values_(values), scores_(scores), ratio_(ratio),
      projected_width_(projected_width) {}

DeepSeekCompressorState::~DeepSeekCompressorState() {
  if (values_) static_cast<void>(cudaFree(values_));
  if (scores_) static_cast<void>(cudaFree(scores_));
}

std::uint64_t DeepSeekCompressorState::bytes() const noexcept {
  const auto rows = ratio_ == 4U ? 8U : ratio_;
  return 2ULL * rows * projected_width_ * sizeof(float);
}

Status DeepSeekCompressorState::reset(void* stream) noexcept {
  const auto rows = ratio_ == 4U ? 8U : ratio_;
  const auto count = rows * projected_width_;
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  auto error = cudaMemsetAsync(values_, 0, count * sizeof(float), cuda_stream);
  if (error != cudaSuccess) return failure(error, "DeepSeek compressor reset values");
  fill_kernel<<<(count + kThreads - 1U) / kThreads, kThreads, 0, cuda_stream>>>(
      scores_, count, kNegativeInfinity);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek compressor reset scores");
}

DeepSeekCompressorStateResult create_deepseek_compressor_state(
    std::uint32_t ratio) noexcept {
  if (ratio != 4U && ratio != 128U)
    return {{ErrorCode::invalid_argument, "unsupported DeepSeek compressor ratio"}, {}};
  const auto width = ratio == 4U ? 1024U : 512U;
  const auto rows = ratio == 4U ? 8U : ratio;
  float* values = nullptr;
  float* scores = nullptr;
  auto error = cudaMalloc(reinterpret_cast<void**>(&values),
                          static_cast<std::size_t>(rows) * width * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void**>(&scores),
                       static_cast<std::size_t>(rows) * width * sizeof(float));
  if (error != cudaSuccess) {
    if (values) static_cast<void>(cudaFree(values));
    if (scores) static_cast<void>(cudaFree(scores));
    return {failure(error, "DeepSeek compressor state allocation"), {}};
  }
  auto state = std::make_shared<DeepSeekCompressorState>(values, scores, ratio,
                                                         width);
  const auto reset = state->reset(nullptr);
  if (!reset.ok()) return {reset, {}};
  return {Status::success(), std::move(state)};
}

Status deepseek_compressor_decode(
    DeepSeekCompressorState& state, const float* projected_values,
    const float* projected_scores, const float* ape,
    const std::uint16_t* norm_weight, float* pooled_workspace,
    float* normalized_output, std::uint32_t position, float epsilon,
    void* stream) noexcept {
  if (!projected_values || !projected_scores || !ape || !norm_weight ||
      !pooled_workspace || !normalized_output || epsilon <= 0.0F)
    return {ErrorCode::invalid_argument, "invalid DeepSeek compressor launch"};
  const auto ratio = state.ratio();
  const auto width = state.projected_width();
  if ((ratio != 4U && ratio != 128U) ||
      width != (ratio == 4U ? 1024U : 512U))
    return {ErrorCode::invalid_argument, "invalid DeepSeek compressor state"};
  const auto local = position % ratio;
  const auto state_row = ratio == 4U ? ratio + local : local;
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  compressor_store_kernel<<<(width + kThreads - 1U) / kThreads, kThreads, 0,
                             cuda_stream>>>(
      projected_values, projected_scores, ape, state.values(), state.scores(),
      state_row, local, width);
  auto error = cudaPeekAtLastError();
  if (error != cudaSuccess) return failure(error, "DeepSeek compressor store");
  if ((position + 1U) % ratio != 0U) return Status::success();
  compressor_pool_kernel<<<kHeadDim, 1, 0, cuda_stream>>>(
      state.values(), state.scores(), pooled_workspace, ratio, width);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) return failure(error, "DeepSeek compressor pool");
  if (ratio == 4U) {
    const auto count = 4U * width;
    compressor_overlap_advance_kernel<<<(count + kThreads - 1U) / kThreads,
                                         kThreads, 0, cuda_stream>>>(
        state.values(), state.scores(), width);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess)
      return failure(error, "DeepSeek compressor overlap advance");
  }
  return rms_norm_bf16_weight(pooled_workspace, norm_weight,
                              normalized_output, kHeadDim, epsilon, stream);
}

Status deepseek_compressed_kv_publish(
    const float* normalized, const float* cosine, const float* sine,
    std::uint16_t* cache, std::uint32_t slot, void* stream) noexcept {
  if (!normalized || !cosine || !sine || !cache)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek compressed KV publication"};
  compressed_kv_publish_kernel<<<1, kHeadDim, 0,
                                 static_cast<cudaStream_t>(stream)>>>(
      normalized, cosine, sine, reinterpret_cast<__nv_bfloat16*>(cache), slot);
  const auto error = cudaPeekAtLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek compressed KV publication");
}

Status deepseek_sparse_attention_decode(
    const std::uint16_t* query, const std::uint16_t* kv_cache,
    const std::int32_t* indices, std::uint32_t selected,
    const float* attention_sink, std::uint16_t* output,
    std::uint32_t heads, void* stream) noexcept {
  if (!query || !kv_cache || !indices || !selected || !attention_sink ||
      !output || !heads || heads > 64U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek sparse attention launch"};
  sparse_attention_decode_kernel<<<heads, kThreads, 0,
                                   static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),
      reinterpret_cast<const __nv_bfloat16*>(kv_cache), indices, selected,
      attention_sink, reinterpret_cast<__nv_bfloat16*>(output));
  const auto error = cudaPeekAtLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek sparse attention");
}

}  // namespace expert::runtime::cuda
