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
    std::uint32_t width) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.y);
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
    intermediate[static_cast<std::size_t>(expert) * width + row] =
        silu * up_sum;
  }
}

__global__ void down_weighted_ordered(
    const std::int8_t* const* weights, const float* const* scales,
    const float* routing, const float* intermediate, float* output,
    std::uint32_t hidden, std::uint32_t width, std::uint32_t top_k) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= hidden) return;

  float ordered_total = 0.0F;
  for (std::uint32_t expert = 0; expert < top_k; ++expert) {
    const auto* matrix_row =
        weights[expert] + static_cast<std::size_t>(row) * width;
    const auto* activation =
        intermediate + static_cast<std::size_t>(expert) * width;
    float partial = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      partial += static_cast<float>(matrix_row[column]) * activation[column];
    }
    partial = block_sum(partial);
    if (threadIdx.x == 0) {
      ordered_total += routing[expert] * partial * scales[expert][row];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = ordered_total;
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
      launch.top_k == 0 || launch.top_k > 64) {
    return Status(ErrorCode::invalid_argument, "invalid MoE CUDA launch");
  }
  auto stream = static_cast<cudaStream_t>(launch.stream);
  const dim3 gate_grid(launch.intermediate_size, launch.top_k);
  gate_up_silu<<<gate_grid, kThreads, 0, stream>>>(
      launch.input, launch.gate_up_weights, launch.gate_up_scales,
      launch.intermediate, launch.hidden_size, launch.intermediate_size);
  auto status = cuda_status(cudaPeekAtLastError(), "gate_up_silu launch");
  if (!status.ok()) return status;

  down_weighted_ordered<<<launch.hidden_size, kThreads, 0, stream>>>(
      launch.down_weights, launch.down_scales, launch.routing_weights,
      launch.intermediate, launch.output, launch.hidden_size,
      launch.intermediate_size, launch.top_k);
  return cuda_status(cudaPeekAtLastError(), "down_weighted_ordered launch");
}

}  // namespace expert::runtime::cuda
