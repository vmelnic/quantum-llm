#include "expert/runtime/cuda/deepseek_hca.hpp"

#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr float kNegativeInfinity = -3.402823466e+38F;

Status failure(cudaError_t error, const char* operation) {
  return {ErrorCode::upload_failed,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

__global__ void hca_normalize_kernel(const float* input, float* output,
                                     std::uint32_t count, float epsilon) {
  __shared__ float partial[kThreads];
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x) {
    const float value = input[index];
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse =
      rsqrtf(partial[0] / static_cast<float>(count) + epsilon);
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x) {
    output[index] = input[index] * inverse;
  }
}

__global__ void hca_normalize_pair_kernel(
    const float* input_zero, const float* input_one, float* output,
    std::uint32_t count, float epsilon) {
  __shared__ float partial[kThreads];
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  const auto* input = row == 0U ? input_zero : input_one;
  output += static_cast<std::size_t>(row) * count;
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x) {
    const float value = input[index];
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride; stride >>= 1U) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse =
      rsqrtf(partial[0] / static_cast<float>(count) + epsilon);
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x)
    output[index] = input[index] * inverse;
}

__global__ void hca_normalize_batch_kernel(
    const float* input, float* output, std::uint32_t count, float epsilon) {
  __shared__ float partial[kThreads];
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  input += static_cast<std::size_t>(row) * count;
  output += static_cast<std::size_t>(row) * count;
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x) {
    const auto value = input[index];
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride; stride >>= 1U) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const auto inverse =
      rsqrtf(partial[0] / static_cast<float>(count) + epsilon);
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x)
    output[index] = input[index] * inverse;
}

__global__ void hca_split_sinkhorn_kernel(
    const float* mixes, const float* base, const float* scale, float* pre,
    float* post, float* comb, float epsilon, std::uint32_t iterations) {
  if (threadIdx.x != 0U) return;
  for (std::uint32_t index = 0; index < 4U; ++index) {
    pre[index] = 1.0F / (1.0F + expf(-(mixes[index] * scale[0] + base[index]))) +
                 epsilon;
    post[index] =
        2.0F /
        (1.0F + expf(-(mixes[4U + index] * scale[1] + base[4U + index])));
  }
  for (std::uint32_t row = 0; row < 4U; ++row) {
    float maximum = kNegativeInfinity;
    for (std::uint32_t column = 0; column < 4U; ++column) {
      const auto index = row * 4U + column;
      const float value = mixes[8U + index] * scale[2] + base[8U + index];
      comb[index] = value;
      maximum = fmaxf(maximum, value);
    }
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < 4U; ++column) {
      const auto index = row * 4U + column;
      comb[index] = expf(comb[index] - maximum);
      sum += comb[index];
    }
    for (std::uint32_t column = 0; column < 4U; ++column) {
      const auto index = row * 4U + column;
      comb[index] = comb[index] / sum + epsilon;
    }
  }
  // Official ordering: initial column normalization, then 19 row/column pairs.
  for (std::uint32_t column = 0; column < 4U; ++column) {
    float sum = 0.0F;
    for (std::uint32_t row = 0; row < 4U; ++row) sum += comb[row * 4U + column];
    for (std::uint32_t row = 0; row < 4U; ++row)
      comb[row * 4U + column] /= sum + epsilon;
  }
  for (std::uint32_t iteration = 1U; iteration < iterations; ++iteration) {
    for (std::uint32_t row = 0; row < 4U; ++row) {
      float sum = 0.0F;
      for (std::uint32_t column = 0; column < 4U; ++column)
        sum += comb[row * 4U + column];
      for (std::uint32_t column = 0; column < 4U; ++column)
        comb[row * 4U + column] /= sum + epsilon;
    }
    for (std::uint32_t column = 0; column < 4U; ++column) {
      float sum = 0.0F;
      for (std::uint32_t row = 0; row < 4U; ++row)
        sum += comb[row * 4U + column];
      for (std::uint32_t row = 0; row < 4U; ++row)
        comb[row * 4U + column] /= sum + epsilon;
    }
  }
}

__global__ void hca_collapse_kernel(const float* streams, const float* pre,
                                    float* output, std::uint32_t hidden) {
  const auto dimension =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (dimension >= hidden) return;
  float result = 0.0F;
  for (std::uint32_t input = 0; input < 4U; ++input)
    result += pre[input] * streams[static_cast<std::size_t>(input) * hidden + dimension];
  output[dimension] = result;
}

__global__ void hca_collapse_pair_kernel(
    const float* streams_zero, const float* streams_one, const float* pre,
    float* output, std::uint32_t hidden) {
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (dimension >= hidden) return;
  const auto* streams = row == 0U ? streams_zero : streams_one;
  pre += row * 4U;
  output += static_cast<std::size_t>(row) * hidden;
  float result = 0.0F;
  for (std::uint32_t input = 0U; input < 4U; ++input)
    result += pre[input] *
        streams[static_cast<std::size_t>(input) * hidden + dimension];
  output[dimension] = result;
}

__global__ void hca_collapse_batch_kernel(
    const float* streams, const float* pre, float* output,
    std::uint32_t hidden) {
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (dimension >= hidden) return;
  streams += static_cast<std::size_t>(row) * kDeepSeekHcaStreams * hidden;
  pre += static_cast<std::size_t>(row) * kDeepSeekHcaStreams;
  output += static_cast<std::size_t>(row) * hidden;
  float result = 0.0F;
  for (std::uint32_t input = 0U; input < kDeepSeekHcaStreams; ++input)
    result += pre[input] *
              streams[static_cast<std::size_t>(input) * hidden + dimension];
  output[dimension] = result;
}

__global__ void hca_post_kernel(const float* sublayer, const float* streams,
                                const float* post, const float* comb,
                                float* updated, std::uint32_t hidden) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const auto count = 4U * hidden;
  if (index >= count) return;
  const auto output = index / hidden;
  const auto dimension = index % hidden;
  float result = post[output] * sublayer[dimension];
  for (std::uint32_t input = 0; input < 4U; ++input) {
    result += comb[input * 4U + output] *
              streams[static_cast<std::size_t>(input) * hidden + dimension];
  }
  updated[index] = result;
}

__global__ void hca_post_batch_kernel(
    const float* sublayer, const float* streams, const float* post,
    const float* comb, float* updated, std::uint32_t hidden) {
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const auto count = kDeepSeekHcaStreams * hidden;
  if (index >= count) return;
  sublayer += static_cast<std::size_t>(row) * hidden;
  streams += static_cast<std::size_t>(row) * count;
  post += static_cast<std::size_t>(row) * kDeepSeekHcaStreams;
  comb += static_cast<std::size_t>(row) * kDeepSeekHcaStreams *
          kDeepSeekHcaStreams;
  updated += static_cast<std::size_t>(row) * count;
  const auto output = index / hidden;
  const auto dimension = index % hidden;
  auto result = post[output] * sublayer[dimension];
  for (std::uint32_t input = 0U; input < kDeepSeekHcaStreams; ++input) {
    result += comb[input * kDeepSeekHcaStreams + output] *
              streams[static_cast<std::size_t>(input) * hidden + dimension];
  }
  updated[index] = result;
}

}  // namespace

DeepSeekHcaParameters::DeepSeekHcaParameters(float* function, float* base,
                                             float* scale,
                                             std::uint32_t hidden) noexcept
    : function_(function), base_(base), scale_(scale), hidden_(hidden) {}

DeepSeekHcaParameters::~DeepSeekHcaParameters() {
  if (function_) static_cast<void>(cudaFree(function_));
  if (base_) static_cast<void>(cudaFree(base_));
  if (scale_) static_cast<void>(cudaFree(scale_));
}

std::uint64_t DeepSeekHcaParameters::bytes() const noexcept {
  return (static_cast<std::uint64_t>(kDeepSeekHcaMixes) *
              kDeepSeekHcaStreams * hidden_ +
          kDeepSeekHcaMixes + 3U) *
         sizeof(float);
}

DeepSeekHcaAdmissionResult admit_deepseek_hca(
    std::span<const std::byte> function, std::span<const std::byte> base,
    std::span<const std::byte> scale, std::uint32_t hidden) noexcept {
  const auto function_bytes = static_cast<std::uint64_t>(kDeepSeekHcaMixes) *
                              kDeepSeekHcaStreams * hidden * sizeof(float);
  if (hidden == 0U || function.size() != function_bytes ||
      base.size() != kDeepSeekHcaMixes * sizeof(float) ||
      scale.size() != 3U * sizeof(float) ||
      function_bytes > std::numeric_limits<std::size_t>::max()) {
    return {{ErrorCode::invalid_argument, "invalid DeepSeek HCA geometry"}, {}};
  }
  float* device_function = nullptr;
  float* device_base = nullptr;
  float* device_scale = nullptr;
  auto error = cudaMalloc(reinterpret_cast<void**>(&device_function),
                          function.size());
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void**>(&device_base), base.size());
  if (error == cudaSuccess)
    error = cudaMalloc(reinterpret_cast<void**>(&device_scale), scale.size());
  if (error == cudaSuccess)
    error = cudaMemcpy(device_function, function.data(), function.size(),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_base, base.data(), base.size(),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(device_scale, scale.data(), scale.size(),
                       cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    if (device_function) static_cast<void>(cudaFree(device_function));
    if (device_base) static_cast<void>(cudaFree(device_base));
    if (device_scale) static_cast<void>(cudaFree(device_scale));
    return {failure(error, "DeepSeek HCA allocation/H2D"), {}};
  }
  return {Status::success(), std::make_shared<DeepSeekHcaParameters>(
                                 device_function, device_base, device_scale,
                                 hidden)};
}

Status deepseek_hca_pre(
    const DeepSeekHcaParameters& parameters, const float* streams,
    float* collapsed, float* pre, float* post, float* comb,
    const DeepSeekHcaWorkspace& workspace, float epsilon,
    std::uint32_t sinkhorn_iterations, void* stream) noexcept {
  return deepseek_hca_pre(parameters.view(), streams, collapsed, pre, post,
                          comb, workspace, epsilon, sinkhorn_iterations,
                          stream);
}

Status deepseek_hca_pre_pair(
    const DeepSeekHcaView& parameters,
    const std::array<const float*, 2U>& streams, float* collapsed,
    float* pre, float* post, float* comb,
    const DeepSeekHcaWorkspace& workspace, float epsilon,
    std::uint32_t sinkhorn_iterations, void* stream) noexcept {
  if (!parameters.function || !parameters.base || !parameters.scale ||
      !streams[0] || !streams[1] || !collapsed || !pre || !post || !comb ||
      !workspace.normalized || !workspace.mixes || parameters.hidden == 0U ||
      epsilon <= 0.0F || sinkhorn_iterations == 0U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek HCA pair launch"};
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  const auto values = kDeepSeekHcaStreams * parameters.hidden;
  hca_normalize_pair_kernel<<<2U, kThreads, 0, cuda_stream>>>(
      streams[0], streams[1], workspace.normalized, values, epsilon);
  auto status = gemv_f32_batch_weight_reuse(
      parameters.function, kDeepSeekHcaMixes, values, workspace.normalized,
      workspace.mixes, 2U, stream);
  if (!status.ok()) return status;
  for (std::uint32_t row = 0U; row < 2U; ++row)
    hca_split_sinkhorn_kernel<<<1U, 32U, 0, cuda_stream>>>(
        workspace.mixes + row * kDeepSeekHcaMixes, parameters.base,
        parameters.scale, pre + row * 4U, post + row * 4U,
        comb + row * 16U, epsilon, sinkhorn_iterations);
  hca_collapse_pair_kernel<<<
      dim3((parameters.hidden + kThreads - 1U) / kThreads, 2U), kThreads, 0,
      cuda_stream>>>(streams[0], streams[1], pre, collapsed,
                     parameters.hidden);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek HCA pair pre");
}

Status deepseek_hca_pre_batch(
    const DeepSeekHcaView& parameters, const float* streams,
    std::uint32_t rows, float* collapsed, float* pre, float* post,
    float* comb, const DeepSeekHcaWorkspace& workspace, float epsilon,
    std::uint32_t sinkhorn_iterations, void* stream) noexcept {
  if (!parameters.function || !parameters.base || !parameters.scale ||
      !streams || !rows || !collapsed || !pre || !post || !comb ||
      !workspace.normalized || !workspace.mixes || parameters.hidden == 0U ||
      epsilon <= 0.0F || sinkhorn_iterations == 0U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek HCA batch launch"};
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  const auto values = kDeepSeekHcaStreams * parameters.hidden;
  hca_normalize_batch_kernel<<<rows, kThreads, 0, cuda_stream>>>(
      streams, workspace.normalized, values, epsilon);
  auto status = gemv_f32_batch_weight_reuse(
      parameters.function, kDeepSeekHcaMixes, values, workspace.normalized,
      workspace.mixes, rows, stream);
  if (!status.ok()) return status;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    hca_split_sinkhorn_kernel<<<1U, 32U, 0, cuda_stream>>>(
        workspace.mixes + row * kDeepSeekHcaMixes, parameters.base,
        parameters.scale, pre + row * kDeepSeekHcaStreams,
        post + row * kDeepSeekHcaStreams,
        comb + row * kDeepSeekHcaStreams * kDeepSeekHcaStreams, epsilon,
        sinkhorn_iterations);
  }
  hca_collapse_batch_kernel<<<
      dim3((parameters.hidden + kThreads - 1U) / kThreads, rows), kThreads, 0,
      cuda_stream>>>(streams, pre, collapsed, parameters.hidden);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek HCA batch pre");
}

Status deepseek_hca_pre(
    const DeepSeekHcaView& parameters, const float* streams,
    float* collapsed, float* pre, float* post, float* comb,
    const DeepSeekHcaWorkspace& workspace, float epsilon,
    std::uint32_t sinkhorn_iterations, void* stream) noexcept {
  if (!parameters.function || !parameters.base || !parameters.scale ||
      !streams || !collapsed || !pre || !post || !comb ||
      !workspace.normalized || !workspace.mixes || parameters.hidden == 0U ||
      epsilon <= 0.0F || sinkhorn_iterations == 0U) {
    return {ErrorCode::invalid_argument, "invalid DeepSeek HCA pre launch"};
  }
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  const auto values = kDeepSeekHcaStreams * parameters.hidden;
  hca_normalize_kernel<<<1, kThreads, 0, cuda_stream>>>(
      streams, workspace.normalized, values, epsilon);
  auto status = gemv_f32(parameters.function, kDeepSeekHcaMixes, values,
                         workspace.normalized, workspace.mixes, stream);
  if (!status.ok()) return status;
  hca_split_sinkhorn_kernel<<<1, 32, 0, cuda_stream>>>(
      workspace.mixes, parameters.base, parameters.scale, pre, post, comb,
      epsilon, sinkhorn_iterations);
  hca_collapse_kernel<<<(parameters.hidden + kThreads - 1U) / kThreads,
                         kThreads, 0, cuda_stream>>>(
      streams, pre, collapsed, parameters.hidden);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek HCA pre");
}

Status deepseek_hca_post(const float* sublayer, const float* streams,
                         const float* post, const float* comb, float* updated,
                         std::uint32_t hidden, void* stream) noexcept {
  if (!sublayer || !streams || !post || !comb || !updated || hidden == 0U)
    return {ErrorCode::invalid_argument, "invalid DeepSeek HCA post launch"};
  const auto count = kDeepSeekHcaStreams * hidden;
  hca_post_kernel<<<(count + kThreads - 1U) / kThreads, kThreads, 0,
                    static_cast<cudaStream_t>(stream)>>>(
      sublayer, streams, post, comb, updated, hidden);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek HCA post");
}

Status deepseek_hca_post_batch(
    const float* sublayer, const float* streams, const float* post,
    const float* comb, float* updated, std::uint32_t rows,
    std::uint32_t hidden, void* stream) noexcept {
  if (!sublayer || !streams || !post || !comb || !updated || !rows || !hidden)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek HCA post batch launch"};
  const auto count = kDeepSeekHcaStreams * hidden;
  hca_post_batch_kernel<<<dim3((count + kThreads - 1U) / kThreads, rows),
                              kThreads, 0,
                              static_cast<cudaStream_t>(stream)>>>(
      sublayer, streams, post, comb, updated, hidden);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek HCA post batch");
}

}  // namespace expert::runtime::cuda
