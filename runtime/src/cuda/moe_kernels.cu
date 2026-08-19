#include "expert/runtime/cuda/moe_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;

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
  const float result = sums[0];
  __syncthreads();
  return result;
}

__device__ float warp_sum(float value) {
  for (unsigned offset = kWarpSize / 2; offset != 0; offset >>= 1U)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__global__ void quantize_q8_vectors(const float* input, std::int8_t* output,
                                    float* scales, std::uint32_t width) {
  const auto vector = static_cast<std::uint32_t>(blockIdx.x);
  const auto* source = input + static_cast<std::size_t>(vector) * width;
  auto* target = output + static_cast<std::size_t>(vector) * width;
  __shared__ float maxima[kThreads];
  float maximum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < width;
       column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(source[column]));
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x],
                                  maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const auto scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) scales[vector] = scale;
  __syncthreads();
  for (std::uint32_t column = threadIdx.x; column < width;
       column += blockDim.x) {
    auto quantized = __float2int_rn(source[column] / scale);
    quantized = max(-127, min(127, quantized));
    target[column] = static_cast<std::int8_t>(quantized);
  }
}

__device__ std::int8_t decode_fp4_twice(std::uint8_t code) {
  const auto index = code & 0x07U;
  const auto magnitude = index <= 4U ? static_cast<int>(index)
                         : index == 5U ? 6
                         : index == 6U ? 8
                                       : 12;
  return static_cast<std::int8_t>((code & 0x08U) != 0U ? -magnitude
                                                       : magnitude);
}

__device__ float decode_ue8m0(std::uint8_t code) {
  // All source scale bytes were validated before publication. Constructing
  // IEEE-754 powers of two avoids a transcendental ldexpf in every dot group.
  return code == 0U ? __uint_as_float(0x00400000U)
                    : __uint_as_float(static_cast<unsigned>(code) << 23U);
}

__device__ int packed_fp4x4(std::uint8_t first, std::uint8_t second) {
  const auto a = static_cast<std::uint8_t>(decode_fp4_twice(first & 0x0fU));
  const auto b = static_cast<std::uint8_t>(decode_fp4_twice(first >> 4U));
  const auto c = static_cast<std::uint8_t>(decode_fp4_twice(second & 0x0fU));
  const auto d = static_cast<std::uint8_t>(decode_fp4_twice(second >> 4U));
  return static_cast<int>(a) | (static_cast<int>(b) << 8U) |
         (static_cast<int>(c) << 16U) | (static_cast<int>(d) << 24U);
}

__device__ float packed_fp4_q8_dot(const std::uint8_t* weights,
                                   const std::uint8_t* weight_scales,
                                   const std::int8_t* activation,
                                   float activation_scale,
                                   std::uint32_t row,
                                   std::uint32_t columns) {
  const auto lane = threadIdx.x % kWarpSize;
  const auto blocks = columns / 32U;
  const auto* row_weights =
      weights + static_cast<std::size_t>(row) * (columns / 2U);
  const auto* row_scales =
      weight_scales + static_cast<std::size_t>(row) * blocks;
  float total = 0.0F;
  const auto groups = columns / 4U;
  for (std::uint32_t group = lane; group < groups; group += kWarpSize) {
    const auto byte_offset = group * 2U;
    const auto packed = packed_fp4x4(row_weights[byte_offset],
                                     row_weights[byte_offset + 1U]);
    const auto q = *reinterpret_cast<const int*>(activation + group * 4U);
    const auto dot = __dp4a(packed, q, 0);
    const auto scale_code = row_scales[group / 8U];
    total += static_cast<float>(dot) * decode_ue8m0(scale_code);
  }
  return warp_sum(total) * activation_scale * 0.5F;
}

__global__ void gate_up_silu(
    const float* input, const std::int8_t* const* weights,
    const float* const* scales, const DeviceExpertEntry* directory,
    std::uint32_t directory_offset, float* intermediate,
    std::uint32_t hidden, std::uint32_t width,
    const std::uint32_t* indices, float swiglu_limit,
    bool bf16_intermediate) {
  const auto slot = static_cast<std::uint32_t>(blockIdx.y);
  const auto expert = indices == nullptr ? slot : indices[slot];
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= width) return;

  const auto* expert_weights =
      directory ? directory[directory_offset + expert].gate_up
                : weights[expert];
  const auto* expert_scales =
      directory ? directory[directory_offset + expert].gate_up_scales
                : scales[expert];
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
    if (swiglu_limit > 0.0F) {
      gate_sum = fminf(gate_sum, swiglu_limit);
      up_sum = fminf(fmaxf(up_sum, -swiglu_limit), swiglu_limit);
    }
    const float silu = gate_sum / (1.0F + expf(-gate_sum));
    float value = silu * up_sum;
    if (bf16_intermediate)
      value = __bfloat162float(__float2bfloat16_rn(value));
    intermediate[static_cast<std::size_t>(slot) * width + row] = value;
  }
}

__global__ void down_weighted_ordered(
    const std::int8_t* const* weights, const float* const* scales,
    const DeviceExpertEntry* directory, std::uint32_t directory_offset,
    const float* routing, const float* intermediate, float* output,
    std::uint32_t hidden, std::uint32_t width, std::uint32_t top_k,
    const std::uint32_t* indices) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= hidden) return;

  float ordered_total = 0.0F;
  for (std::uint32_t expert = 0; expert < top_k; ++expert) {
    const auto expert_id = indices == nullptr ? expert : indices[expert];
    const auto* expert_weights =
        directory ? directory[directory_offset + expert_id].down
                  : weights[expert_id];
    const auto* expert_scales =
        directory ? directory[directory_offset + expert_id].down_scales
                  : scales[expert_id];
    const auto* matrix_row =
        expert_weights + static_cast<std::size_t>(row) * width;
    const auto* activation =
        intermediate + static_cast<std::size_t>(expert) * width;
    float partial = 0.0F;
    for (std::uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      partial += static_cast<float>(matrix_row[column]) * activation[column];
    }
    partial = block_sum(partial);
    if (threadIdx.x == 0) {
      ordered_total += routing[expert] * partial * expert_scales[row];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[row] = ordered_total;
}

__global__ void gate_up_silu_batch(
    const float* input, const std::int8_t* const* weights,
    const float* const* scales, const DeviceExpertEntry* directory,
    std::uint32_t directory_offset, float* intermediate,
    std::uint32_t hidden, std::uint32_t width, std::uint32_t top_k,
    const std::uint32_t* indices, float swiglu_limit,
    bool bf16_intermediate) {
  const auto selection = static_cast<std::uint32_t>(blockIdx.y);
  const auto request_row = selection / top_k;
  const auto slot = selection % top_k;
  const auto expert = indices[static_cast<std::size_t>(request_row) * top_k + slot];
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto output_row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (output_row >= width) return;
  const auto* expert_weights =
      directory ? directory[directory_offset + expert].gate_up
                : weights[expert];
  const auto* expert_scales =
      directory ? directory[directory_offset + expert].gate_up_scales
                : scales[expert];
  const auto* gate = expert_weights + static_cast<std::size_t>(output_row) * hidden;
  const auto* up = expert_weights + static_cast<std::size_t>(width + output_row) * hidden;
  const auto* activation = input + static_cast<std::size_t>(request_row) * hidden;
  float gate_sum = 0.0F, up_sum = 0.0F;
  for (std::uint32_t column = lane; column < hidden; column += kWarpSize) {
    gate_sum += static_cast<float>(gate[column]) * activation[column];
    up_sum += static_cast<float>(up[column]) * activation[column];
  }
  gate_sum = warp_sum(gate_sum);
  up_sum = warp_sum(up_sum);
  if (lane == 0) {
    gate_sum *= expert_scales[output_row];
    up_sum *= expert_scales[width + output_row];
    if (swiglu_limit > 0.0F) {
      gate_sum = fminf(gate_sum, swiglu_limit);
      up_sum = fminf(fmaxf(up_sum, -swiglu_limit), swiglu_limit);
    }
    const auto offset = (static_cast<std::size_t>(request_row) * top_k + slot) * width + output_row;
    float value = (gate_sum / (1.0F + expf(-gate_sum))) * up_sum;
    if (bf16_intermediate)
      value = __bfloat162float(__float2bfloat16_rn(value));
    intermediate[offset] = value;
  }
}

__global__ void down_weighted_ordered_batch(
    const std::int8_t* const* weights, const float* const* scales,
    const DeviceExpertEntry* directory, std::uint32_t directory_offset,
    const float* routing, const std::uint32_t* indices,
    const float* intermediate, float* output, std::uint32_t hidden,
    std::uint32_t width, std::uint32_t top_k) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto output_row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const auto request_row = static_cast<std::uint32_t>(blockIdx.y);
  if (output_row >= hidden) return;
  float ordered_total = 0.0F;
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    const auto selection = static_cast<std::size_t>(request_row) * top_k + slot;
    const auto expert = indices[selection];
    const auto* expert_weights =
        directory ? directory[directory_offset + expert].down
                  : weights[expert];
    const auto* expert_scales =
        directory ? directory[directory_offset + expert].down_scales
                  : scales[expert];
    const auto* matrix_row = expert_weights + static_cast<std::size_t>(output_row) * width;
    const auto* activation = intermediate + selection * width;
    float partial = 0.0F;
    for (std::uint32_t column = lane; column < width; column += kWarpSize)
      partial += static_cast<float>(matrix_row[column]) * activation[column];
    partial = warp_sum(partial);
    if (lane == 0)
      ordered_total += routing[selection] * partial * expert_scales[output_row];
  }
  if (lane == 0)
    output[static_cast<std::size_t>(request_row) * hidden + output_row] = ordered_total;
}

__global__ void gate_up_silu_selection_batch(
    const float* input, const DeviceExpertEntry* directory,
    std::uint32_t directory_offset, float* intermediate,
    std::uint32_t hidden, std::uint32_t width, std::uint32_t top_k,
    const std::uint32_t* indices, const std::uint8_t* mask,
    float swiglu_limit, bool bf16_intermediate,
    const std::int8_t* quantized_input,
    const float* quantized_input_scales) {
  const auto selection = static_cast<std::uint32_t>(blockIdx.y);
  if (mask != nullptr && mask[selection] == 0) return;
  const auto request_row = selection / top_k;
  const auto expert = indices[selection];
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto output_row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (output_row >= width) return;
  const auto& entry = directory[directory_offset + expert];
  const auto* activation =
      input + static_cast<std::size_t>(request_row) * hidden;
  float gate_sum = 0.0F;
  float up_sum = 0.0F;
  if (entry.format == static_cast<std::uint32_t>(
                          DeviceExpertFormat::fp4_e2m1_ue8m0_block32)) {
    const auto* q = quantized_input +
                    static_cast<std::size_t>(request_row) * hidden;
    gate_sum = packed_fp4_q8_dot(entry.w1_fp4, entry.w1_ue8m0, q,
                                quantized_input_scales[request_row],
                                output_row, hidden);
    up_sum = packed_fp4_q8_dot(entry.w3_fp4, entry.w3_ue8m0, q,
                              quantized_input_scales[request_row],
                              output_row, hidden);
  } else {
    const auto* gate =
        entry.gate_up + static_cast<std::size_t>(output_row) * hidden;
    const auto* up = entry.gate_up +
                     static_cast<std::size_t>(width + output_row) * hidden;
    for (std::uint32_t column = lane; column < hidden;
         column += kWarpSize) {
      gate_sum += static_cast<float>(gate[column]) * activation[column];
      up_sum += static_cast<float>(up[column]) * activation[column];
    }
    gate_sum = warp_sum(gate_sum);
    up_sum = warp_sum(up_sum);
    gate_sum *= entry.gate_up_scales[output_row];
    up_sum *= entry.gate_up_scales[width + output_row];
  }
  if (lane == 0) {
    if (swiglu_limit > 0.0F) {
      gate_sum = fminf(gate_sum, swiglu_limit);
      up_sum = fminf(fmaxf(up_sum, -swiglu_limit), swiglu_limit);
    }
    float value = (gate_sum / (1.0F + expf(-gate_sum))) * up_sum;
    if (bf16_intermediate)
      value = __bfloat162float(__float2bfloat16_rn(value));
    intermediate[static_cast<std::size_t>(selection) * width + output_row] =
        value;
  }
}

__global__ void down_selection_batch(
    const DeviceExpertEntry* directory, std::uint32_t directory_offset,
    const std::uint32_t* indices, const std::uint8_t* mask,
    const float* intermediate, float* selection_outputs,
    std::uint32_t hidden, std::uint32_t width,
    const std::int8_t* quantized_intermediate,
    const float* quantized_intermediate_scales) {
  const auto selection = static_cast<std::uint32_t>(blockIdx.y);
  if (mask != nullptr && mask[selection] == 0) return;
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto output_row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (output_row >= hidden) return;
  const auto& entry = directory[directory_offset + indices[selection]];
  const auto* activation =
      intermediate + static_cast<std::size_t>(selection) * width;
  float partial = 0.0F;
  if (entry.format == static_cast<std::uint32_t>(
                          DeviceExpertFormat::fp4_e2m1_ue8m0_block32)) {
    const auto* q = quantized_intermediate +
                    static_cast<std::size_t>(selection) * width;
    partial = packed_fp4_q8_dot(
        entry.w2_fp4, entry.w2_ue8m0, q,
        quantized_intermediate_scales[selection], output_row, width);
  } else {
    const auto* weights =
        entry.down + static_cast<std::size_t>(output_row) * width;
    for (std::uint32_t column = lane; column < width;
         column += kWarpSize) {
      partial += static_cast<float>(weights[column]) * activation[column];
    }
    partial = warp_sum(partial);
    partial *= entry.down_scales[output_row];
  }
  if (lane == 0) {
    selection_outputs[static_cast<std::size_t>(selection) * hidden +
                      output_row] = partial;
  }
}

__global__ void aggregate_selection_outputs(
    const float* selection_outputs, const float* alternate_outputs,
    const std::uint8_t* primary_mask,
    const std::uint32_t* alternate_slot_by_selection,
    const float* routing, float* output,
    std::uint32_t hidden, std::uint32_t top_k,
    std::uint32_t value_count) {
  for (auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x +
                                               threadIdx.x);
       index < value_count; index += blockDim.x * gridDim.x) {
    const auto row = index / hidden;
    const auto column = index % hidden;
    float total = 0.0F;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      const auto selection = static_cast<std::size_t>(row) * top_k + slot;
      const bool primary = primary_mask == nullptr || primary_mask[selection];
      const auto source_selection = primary
          ? selection
          : static_cast<std::size_t>(alternate_slot_by_selection[selection]);
      const auto* source = primary ? selection_outputs : alternate_outputs;
      total += routing[selection] * source[source_selection * hidden + column];
    }
    output[index] = total;
  }
}

Status cuda_status(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return Status::success();
  return Status(ErrorCode::upload_failed,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

}  // namespace

Status launch_moe_single_token(const MoeLaunch& launch) noexcept {
  const bool pointer_table = launch.gate_up_weights != nullptr &&
                             launch.gate_up_scales != nullptr &&
                             launch.down_weights != nullptr &&
                             launch.down_scales != nullptr;
  if (launch.input == nullptr ||
      (!pointer_table && launch.directory_entries == nullptr) ||
      launch.routing_weights == nullptr ||
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
      launch.directory_entries,
      launch.directory_layer * launch.expert_table_size, launch.intermediate,
      launch.hidden_size, launch.intermediate_size, launch.expert_indices,
      launch.swiglu_limit, launch.bf16_intermediate);
  auto status = cuda_status(cudaPeekAtLastError(), "gate_up_silu launch");
  if (!status.ok()) return status;

  down_weighted_ordered<<<launch.hidden_size, kThreads, 0, stream>>>(
      launch.down_weights, launch.down_scales, launch.directory_entries,
      launch.directory_layer * launch.expert_table_size,
      launch.routing_weights, launch.intermediate, launch.output,
      launch.hidden_size, launch.intermediate_size, launch.top_k,
      launch.expert_indices);
  return cuda_status(cudaPeekAtLastError(), "down_weighted_ordered launch");
}

Status launch_moe_batch(const MoeBatchLaunch& launch) noexcept {
  const bool pointer_table = launch.gate_up_weights != nullptr &&
                             launch.gate_up_scales != nullptr &&
                             launch.down_weights != nullptr &&
                             launch.down_scales != nullptr;
  if (launch.input == nullptr ||
      (!pointer_table && launch.directory_entries == nullptr) ||
      launch.routing_weights == nullptr ||
      launch.expert_indices == nullptr || launch.intermediate == nullptr ||
      launch.output == nullptr || launch.rows == 0 || launch.hidden_size == 0 ||
      launch.intermediate_size == 0 || launch.top_k == 0 || launch.top_k > 64 ||
      launch.expert_table_size < launch.top_k) {
    return Status(ErrorCode::invalid_argument, "invalid batched MoE CUDA launch");
  }
  auto stream = static_cast<cudaStream_t>(launch.stream);
  const dim3 gate_grid(
      (launch.intermediate_size + kWarpsPerBlock - 1U) / kWarpsPerBlock,
      launch.rows * launch.top_k);
  gate_up_silu_batch<<<gate_grid, kThreads, 0, stream>>>(
      launch.input, launch.gate_up_weights, launch.gate_up_scales,
      launch.directory_entries,
      launch.directory_layer * launch.expert_table_size, launch.intermediate,
      launch.hidden_size, launch.intermediate_size, launch.top_k,
      launch.expert_indices, launch.swiglu_limit,
      launch.bf16_intermediate);
  auto status = cuda_status(cudaPeekAtLastError(), "gate_up_silu_batch launch");
  if (!status.ok()) return status;
  const dim3 down_grid(
      (launch.hidden_size + kWarpsPerBlock - 1U) / kWarpsPerBlock,
      launch.rows);
  down_weighted_ordered_batch<<<down_grid, kThreads, 0, stream>>>(
      launch.down_weights, launch.down_scales, launch.directory_entries,
      launch.directory_layer * launch.expert_table_size,
      launch.routing_weights, launch.expert_indices, launch.intermediate,
      launch.output, launch.hidden_size, launch.intermediate_size,
      launch.top_k);
  return cuda_status(cudaPeekAtLastError(), "down_weighted_ordered_batch launch");
}

Status launch_moe_selection_batch(
    const MoeSelectionBatchLaunch& launch) noexcept {
  if (launch.input == nullptr || launch.directory_entries == nullptr ||
      launch.expert_indices == nullptr || launch.intermediate == nullptr ||
      launch.selection_outputs == nullptr ||
      (launch.enable_packed_fp4 &&
       (launch.quantized_input == nullptr ||
        launch.quantized_input_scales == nullptr ||
        launch.quantized_intermediate == nullptr ||
        launch.quantized_intermediate_scales == nullptr)) || launch.rows == 0 ||
      launch.hidden_size == 0 || launch.intermediate_size == 0 ||
      launch.top_k == 0 || launch.top_k > 64 ||
      launch.expert_table_size < launch.top_k) {
    return Status(ErrorCode::invalid_argument,
                  "invalid selection MoE CUDA launch");
  }
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  const auto selections = launch.rows * launch.top_k;
  auto status = Status::success();
  if (launch.enable_packed_fp4) {
    quantize_q8_vectors<<<launch.rows, kThreads, 0, stream>>>(
        launch.input, launch.quantized_input, launch.quantized_input_scales,
        launch.hidden_size);
    status = cuda_status(cudaPeekAtLastError(),
                         "quantize routed input launch");
    if (!status.ok()) return status;
  }
  const dim3 gate_grid(
      (launch.intermediate_size + kWarpsPerBlock - 1U) / kWarpsPerBlock,
      selections);
  gate_up_silu_selection_batch<<<gate_grid, kThreads, 0, stream>>>(
      launch.input, launch.directory_entries,
      launch.directory_layer * launch.expert_table_size,
      launch.intermediate, launch.hidden_size, launch.intermediate_size,
      launch.top_k, launch.expert_indices, launch.selection_mask,
      launch.swiglu_limit, launch.bf16_intermediate, launch.quantized_input,
      launch.quantized_input_scales);
  status =
      cuda_status(cudaPeekAtLastError(), "gate_up_silu_selection launch");
  if (!status.ok()) return status;
  if (launch.enable_packed_fp4) {
    quantize_q8_vectors<<<selections, kThreads, 0, stream>>>(
        launch.intermediate, launch.quantized_intermediate,
        launch.quantized_intermediate_scales, launch.intermediate_size);
    status = cuda_status(cudaPeekAtLastError(),
                         "quantize routed intermediate launch");
    if (!status.ok()) return status;
  }
  const dim3 down_grid(
      (launch.hidden_size + kWarpsPerBlock - 1U) / kWarpsPerBlock,
      selections);
  down_selection_batch<<<down_grid, kThreads, 0, stream>>>(
      launch.directory_entries,
      launch.directory_layer * launch.expert_table_size,
      launch.expert_indices, launch.selection_mask, launch.intermediate,
      launch.selection_outputs, launch.hidden_size, launch.intermediate_size,
      launch.quantized_intermediate,
      launch.quantized_intermediate_scales);
  return cuda_status(cudaPeekAtLastError(), "down_selection launch");
}

Status launch_moe_aggregate(const MoeAggregateLaunch& launch) noexcept {
  if (launch.selection_outputs == nullptr ||
      launch.routing_weights == nullptr || launch.output == nullptr ||
      launch.rows == 0 || launch.hidden_size == 0 || launch.top_k == 0 ||
      launch.top_k > 64 ||
      (launch.primary_mask != nullptr &&
       (launch.alternate_outputs == nullptr ||
        launch.alternate_slot_by_selection == nullptr ||
        launch.alternate_output_count == 0)) ||
      (launch.primary_mask == nullptr &&
       (launch.alternate_outputs != nullptr ||
        launch.alternate_slot_by_selection != nullptr ||
        launch.alternate_output_count != 0))) {
    return Status(ErrorCode::invalid_argument,
                  "invalid MoE aggregate CUDA launch");
  }
  const auto values = launch.rows * launch.hidden_size;
  const auto blocks = std::min(256U, (values + kThreads - 1U) / kThreads);
  aggregate_selection_outputs<<<blocks, kThreads, 0,
                                static_cast<cudaStream_t>(launch.stream)>>>(
      launch.selection_outputs, launch.alternate_outputs,
      launch.primary_mask, launch.alternate_slot_by_selection,
      launch.routing_weights, launch.output,
      launch.hidden_size, launch.top_k, values);
  return cuda_status(cudaPeekAtLastError(), "aggregate_selection launch");
}

}  // namespace expert::runtime::cuda
