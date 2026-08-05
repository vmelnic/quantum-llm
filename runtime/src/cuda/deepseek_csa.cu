#include "expert/runtime/cuda/deepseek_csa.hpp"

#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>

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

}  // namespace expert::runtime::cuda
