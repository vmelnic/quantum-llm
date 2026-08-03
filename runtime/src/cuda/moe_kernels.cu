#include "expert/runtime/cuda/moe_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;

__device__ float block_sum(float value) {
  __shared__ float sums[kThreads];
  sums[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return sums[0];
}

__global__ void gate_up_silu(
    const float* input, const std::int8_t* const* weights,
    const float* const* scales, float* intermediate, std::uint32_t hidden,
    std::uint32_t width, const std::uint32_t* indices) {
  const auto slot = static_cast<std::uint32_t>(blockIdx.y);
  const auto expert = indices == nullptr ? slot : indices[slot];
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= width) return;

  const auto* expert_weights = weights[expert];
  const auto* expert_scales = scales[expert];
  const auto* gate = expert_weights + static_cast<std::size_t>(row) * hidden;
  const auto* up = expert_weights +
                   static_cast<std::size_t>(width + row) * hidden;
  float gate_sum = 0.0F;
  float up_sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < hidden;
       column += blockDim.x) {
    const float activation = input[column];
    gate_sum += static_cast<float>(gate[column]) * activation;
    up_sum += static_cast<float>(up[column]) * activation;
  }
  gate_sum = block_sum(gate_sum);
  up_sum = block_sum(up_sum);
  if (threadIdx.x == 0) {
    gate_sum *= expert_scales[row];
    up_sum *= expert_scales[width + row];
    const float silu = gate_sum / (1.0F + expf(-gate_sum));
    intermediate[static_cast<std::size_t>(slot) * width + row] =
        silu * up_sum;
  }
}

__global__ void down_weighted_ordered(
    const std::int8_t* const* weights, const float* const* scales,
    const float* routing, const float* intermediate, float* output,
    std::uint32_t hidden, std::uint32_t width, std::uint32_t top_k,
    const std::uint32_t* indices) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= hidden) return;

  float ordered_total = 0.0F;
  for (std::uint32_t expert = 0; expert < top_k; ++expert) {
    const auto expert_id = indices == nullptr ? expert : indices[expert];
    const auto* matrix_row =
        weights[expert_id] + static_cast<std::size_t>(row) * width;
    const auto* activation =
        intermediate + static_cast<std::size_t>(expert) * width;
    float partial = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      partial += static_cast<float>(matrix_row[column]) * activation[column];
    }
    partial = block_sum(partial);
    if (threadIdx.x == 0) {
      ordered_total += routing[expert] * partial * scales[expert_id][row];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = ordered_total;
}

__global__ void gate_up_silu_batch(
    const float* input, const std::int8_t* const* weights,
    const float* const* scales, float* intermediate, std::uint32_t hidden,
    std::uint32_t width, std::uint32_t top_k,
    const std::uint32_t* indices) {
  const auto selection = static_cast<std::uint32_t>(blockIdx.y);
  const auto request_row = selection / top_k;
  const auto slot = selection % top_k;
  const auto expert = indices[static_cast<std::size_t>(request_row) * top_k + slot];
  const auto output_row = static_cast<std::uint32_t>(blockIdx.x);
  if (output_row >= width) return;
  const auto* expert_weights = weights[expert];
  const auto* expert_scales = scales[expert];
  const auto* gate = expert_weights + static_cast<std::size_t>(output_row) * hidden;
  const auto* up = expert_weights + static_cast<std::size_t>(width + output_row) * hidden;
  const auto* activation = input + static_cast<std::size_t>(request_row) * hidden;
  float gate_sum = 0.0F, up_sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    gate_sum += static_cast<float>(gate[column]) * activation[column];
    up_sum += static_cast<float>(up[column]) * activation[column];
  }
  gate_sum = block_sum(gate_sum);
  up_sum = block_sum(up_sum);
  if (threadIdx.x == 0) {
    gate_sum *= expert_scales[output_row];
    up_sum *= expert_scales[width + output_row];
    const auto offset = (static_cast<std::size_t>(request_row) * top_k + slot) * width + output_row;
    intermediate[offset] = (gate_sum / (1.0F + expf(-gate_sum))) * up_sum;
  }
}

__global__ void down_weighted_ordered_batch(
    const std::int8_t* const* weights, const float* const* scales,
    const float* routing, const std::uint32_t* indices,
    const float* intermediate, float* output, std::uint32_t hidden,
    std::uint32_t width, std::uint32_t top_k) {
  const auto output_row = static_cast<std::uint32_t>(blockIdx.x);
  const auto request_row = static_cast<std::uint32_t>(blockIdx.y);
  if (output_row >= hidden) return;
  float ordered_total = 0.0F;
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    const auto selection = static_cast<std::size_t>(request_row) * top_k + slot;
    const auto expert = indices[selection];
    const auto* matrix_row = weights[expert] + static_cast<std::size_t>(output_row) * width;
    const auto* activation = intermediate + selection * width;
    float partial = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < width; column += blockDim.x)
      partial += static_cast<float>(matrix_row[column]) * activation[column];
    partial = block_sum(partial);
    if (threadIdx.x == 0)
      ordered_total += routing[selection] * partial * scales[expert][output_row];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[static_cast<std::size_t>(request_row) * hidden + output_row] = ordered_total;
}

Status cuda_status(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return Status::success();
  return Status(ErrorCode::upload_failed,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

}  // namespace

Status launch_moe_single_token(const MoeLaunch& launch) noexcept {
  if (launch.input == nullptr || launch.gate_up_weights == nullptr ||
      launch.gate_up_scales == nullptr || launch.down_weights == nullptr ||
      launch.down_scales == nullptr || launch.routing_weights == nullptr ||
      launch.intermediate == nullptr || launch.output == nullptr ||
      launch.hidden_size == 0 || launch.intermediate_size == 0 ||
      launch.top_k == 0 || launch.top_k > 64 ||
      launch.expert_table_size < launch.top_k) {
    return Status(ErrorCode::invalid_argument, "invalid MoE CUDA launch");
  }
  auto stream = static_cast<cudaStream_t>(launch.stream);
  const dim3 gate_grid(launch.intermediate_size, launch.top_k);
  gate_up_silu<<<gate_grid, kThreads, 0, stream>>>(
      launch.input, launch.gate_up_weights, launch.gate_up_scales,
      launch.intermediate, launch.hidden_size, launch.intermediate_size,
      launch.expert_indices);
  auto status = cuda_status(cudaPeekAtLastError(), "gate_up_silu launch");
  if (!status.ok()) return status;

  down_weighted_ordered<<<launch.hidden_size, kThreads, 0, stream>>>(
      launch.down_weights, launch.down_scales, launch.routing_weights,
      launch.intermediate, launch.output, launch.hidden_size,
      launch.intermediate_size, launch.top_k, launch.expert_indices);
  return cuda_status(cudaPeekAtLastError(), "down_weighted_ordered launch");
}

Status launch_moe_batch(const MoeBatchLaunch& launch) noexcept {
  if (launch.input == nullptr || launch.gate_up_weights == nullptr ||
      launch.gate_up_scales == nullptr || launch.down_weights == nullptr ||
      launch.down_scales == nullptr || launch.routing_weights == nullptr ||
      launch.expert_indices == nullptr || launch.intermediate == nullptr ||
      launch.output == nullptr || launch.rows == 0 || launch.hidden_size == 0 ||
      launch.intermediate_size == 0 || launch.top_k == 0 || launch.top_k > 64 ||
      launch.expert_table_size < launch.top_k) {
    return Status(ErrorCode::invalid_argument, "invalid batched MoE CUDA launch");
  }
  auto stream = static_cast<cudaStream_t>(launch.stream);
  const dim3 gate_grid(launch.intermediate_size, launch.rows * launch.top_k);
  gate_up_silu_batch<<<gate_grid, kThreads, 0, stream>>>(
      launch.input, launch.gate_up_weights, launch.gate_up_scales,
      launch.intermediate, launch.hidden_size, launch.intermediate_size,
      launch.top_k, launch.expert_indices);
  auto status = cuda_status(cudaPeekAtLastError(), "gate_up_silu_batch launch");
  if (!status.ok()) return status;
  const dim3 down_grid(launch.hidden_size, launch.rows);
  down_weighted_ordered_batch<<<down_grid, kThreads, 0, stream>>>(
      launch.down_weights, launch.down_scales, launch.routing_weights,
      launch.expert_indices, launch.intermediate, launch.output,
      launch.hidden_size, launch.intermediate_size, launch.top_k);
  return cuda_status(cudaPeekAtLastError(), "down_weighted_ordered_batch launch");
}

}  // namespace expert::runtime::cuda
