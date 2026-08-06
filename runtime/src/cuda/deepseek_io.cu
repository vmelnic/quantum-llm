#include "expert/runtime/cuda/deepseek_io.hpp"

#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint32_t kStreams = 4U;
constexpr std::uint32_t kThreads = 256U;
constexpr std::uint64_t kNormalizedStreamValues =
    static_cast<std::uint64_t>(kStreams) * kDeepSeekHidden;
constexpr std::uint64_t kStateFloatValues =
    kNormalizedStreamValues + kStreams + kStreams + kDeepSeekHidden +
    kDeepSeekHidden + kDeepSeekVocab;
constexpr std::uint64_t kStateBytes =
    kStateFloatValues * sizeof(float) + sizeof(std::uint32_t);

Status failure(cudaError_t error, const char* operation) {
  return {ErrorCode::internal,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

__device__ float bf16_round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void embedding_kernel(const std::uint16_t* embedding,
                                 std::uint32_t token, float* streams) {
  const auto dimension =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (dimension >= kDeepSeekHidden) return;
  const auto bits = embedding[static_cast<std::size_t>(token) *
                                  kDeepSeekHidden + dimension];
  const float value = __uint_as_float(static_cast<unsigned>(bits) << 16U);
  for (std::uint32_t stream = 0U; stream < kStreams; ++stream)
    streams[static_cast<std::size_t>(stream) * kDeepSeekHidden + dimension] =
        value;
}

__global__ void normalize_head_streams_kernel(
    const float* streams, float* normalized, float epsilon) {
  __shared__ float partial[kThreads];
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < kNormalizedStreamValues;
       index += blockDim.x) {
    const float value = bf16_round(streams[index]);
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride; stride >>= 1U) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse = rsqrtf(
      partial[0] / static_cast<float>(kNormalizedStreamValues) + epsilon);
  for (std::uint32_t index = threadIdx.x; index < kNormalizedStreamValues;
       index += blockDim.x)
    normalized[index] = bf16_round(streams[index]) * inverse;
}

__global__ void head_pre_kernel(const float* mixes, const float* scale,
                                const float* base, float* pre,
                                float epsilon) {
  if (threadIdx.x < kStreams) {
    const auto index = threadIdx.x;
    pre[index] = 1.0F /
                     (1.0F + expf(-(mixes[index] * scale[0] + base[index]))) +
                 epsilon;
  }
}

__global__ void head_collapse_kernel(const float* streams, const float* pre,
                                     float* collapsed) {
  const auto dimension =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (dimension >= kDeepSeekHidden) return;
  float value = 0.0F;
  for (std::uint32_t stream = 0U; stream < kStreams; ++stream)
    value += pre[stream] *
             bf16_round(streams[static_cast<std::size_t>(stream) *
                                    kDeepSeekHidden + dimension]);
  collapsed[dimension] = bf16_round(value);
}

__global__ void round_bf16_kernel(float* values, std::uint32_t count) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) values[index] = bf16_round(values[index]);
}

}  // namespace

DeepSeekIoState::DeepSeekIoState(void* allocation,
                                 std::uint64_t bytes) noexcept
    : allocation_(allocation), bytes_(bytes) {
  map(allocation);
}

DeepSeekIoState::~DeepSeekIoState() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

void DeepSeekIoState::map(void* allocation) noexcept {
  auto* cursor = static_cast<float*>(allocation);
  normalized_streams_ = cursor;
  cursor += kNormalizedStreamValues;
  head_mixes_ = cursor;
  cursor += kStreams;
  head_pre_ = cursor;
  cursor += kStreams;
  collapsed_ = cursor;
  cursor += kDeepSeekHidden;
  normalized_ = cursor;
  cursor += kDeepSeekHidden;
  logits_ = cursor;
  cursor += kDeepSeekVocab;
  sampled_token_ = reinterpret_cast<std::uint32_t*>(cursor);
}

std::uint64_t deepseek_io_state_size() noexcept { return kStateBytes; }

DeepSeekIoStateResult create_deepseek_io_state() noexcept {
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, kStateBytes);
  if (error != cudaSuccess)
    return {failure(error, "allocate DeepSeek I/O state"), {}};
  error = cudaMemset(allocation, 0, kStateBytes);
  if (error != cudaSuccess) {
    static_cast<void>(cudaFree(allocation));
    return {failure(error, "reset DeepSeek I/O state"), {}};
  }
  return {Status::success(), std::shared_ptr<DeepSeekIoState>(
                                 new DeepSeekIoState(allocation, kStateBytes))};
}

Status deepseek_embed(const DeepSeekIoBinding& weights, std::uint32_t token,
                      float* streams, void* raw_stream) noexcept {
  if (!weights.embedding || token >= kDeepSeekVocab || !streams)
    return {ErrorCode::invalid_argument, "invalid DeepSeek embedding launch"};
  embedding_kernel<<<(kDeepSeekHidden + kThreads - 1U) / kThreads, kThreads,
                       0, static_cast<cudaStream_t>(raw_stream)>>>(
      weights.embedding, token, streams);
  const auto error = cudaPeekAtLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek embedding");
}

Status deepseek_head(const DeepSeekIoBinding& weights, const float* streams,
                     DeepSeekIoState& state, float epsilon,
                     void* raw_stream) noexcept {
  if (!weights.head)
    return {ErrorCode::invalid_argument, "invalid DeepSeek head launch"};
  auto status = deepseek_hc_head(weights, streams, state, epsilon, raw_stream);
  if (!status.ok()) return status;
  status = gemv_bf16(weights.head, kDeepSeekVocab, kDeepSeekHidden,
                     state.normalized_, state.logits_, raw_stream);
  if (!status.ok()) return status;
  status = argmax(state.logits_, kDeepSeekVocab, state.sampled_token_,
                  raw_stream);
  if (!status.ok()) return status;
  const auto error = cudaPeekAtLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek output head");
}

Status deepseek_hc_head(const DeepSeekIoBinding& weights, const float* streams,
                        DeepSeekIoState& state, float epsilon,
                        void* raw_stream) noexcept {
  if (!weights.final_norm || !weights.head_function || !weights.head_base ||
      !weights.head_scale || !streams || epsilon <= 0.0F)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek hyper-head launch"};
  const auto stream = static_cast<cudaStream_t>(raw_stream);
  normalize_head_streams_kernel<<<1, kThreads, 0, stream>>>(
      streams, state.normalized_streams_, epsilon);
  auto status = gemv_f32(weights.head_function, kStreams,
                         kNormalizedStreamValues, state.normalized_streams_,
                         state.head_mixes_, raw_stream);
  if (!status.ok()) return status;
  head_pre_kernel<<<1, 32U, 0, stream>>>(
      state.head_mixes_, weights.head_scale, weights.head_base,
      state.head_pre_, epsilon);
  head_collapse_kernel<<<(kDeepSeekHidden + kThreads - 1U) / kThreads,
                          kThreads, 0, stream>>>(
      streams, state.head_pre_, state.collapsed_);
  status = rms_norm_bf16_weight(state.collapsed_, weights.final_norm,
                                state.normalized_, kDeepSeekHidden, epsilon,
                                raw_stream);
  if (!status.ok()) return status;
  round_bf16_kernel<<<(kDeepSeekHidden + kThreads - 1U) / kThreads, kThreads,
                       0, stream>>>(state.normalized_, kDeepSeekHidden);
  const auto error = cudaPeekAtLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek hyper-head");
}

}  // namespace expert::runtime::cuda
