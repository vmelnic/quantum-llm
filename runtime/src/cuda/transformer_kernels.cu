#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned kMaximumRouterTopK = 32;
constexpr unsigned kMaximumWeightReuseBatch = 8;
constexpr float kNegativeInfinity = -3.402823466e+38F;

__device__ float warp_sum(float value) {
  for (unsigned offset = kWarpSize / 2; offset; offset >>= 1U)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__device__ float reduce_sum(float value) {
  __shared__ float shared[kThreads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__global__ void embedding_kernel(const std::int8_t* weights,
                                 const float* scales, std::uint32_t token,
                                 std::uint32_t columns, float* output) {
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) {
    output[i] = static_cast<float>(weights[static_cast<std::size_t>(token) * columns + i]) * scales[token];
  }
}

__global__ void int8_gemv_kernel(const std::int8_t* weights,
                                 const float* scales, const float* input,
                                 float* output, std::uint32_t rows,
                                 std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize) {
    partial += static_cast<float>(weight[i]) * input[i];
  }
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial * scales[row];
}

__global__ void int8_gemv_vector_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(
      blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    const auto activation =
        *reinterpret_cast<const float4*>(input + column);
    partial += static_cast<float>(packed.x) * activation.x;
    partial += static_cast<float>(packed.y) * activation.y;
    partial += static_cast<float>(packed.z) * activation.z;
    partial += static_cast<float>(packed.w) * activation.w;
  }
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial * scales[row];
}

__global__ void int8_gemv_grouped_inputs_vector_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t rows_per_group) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(
      blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  const auto group = row / rows_per_group;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(group) * columns;
  float partial = 0.0F;
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    const auto values =
        *reinterpret_cast<const float4*>(activation + column);
    partial += static_cast<float>(packed.x) * values.x;
    partial += static_cast<float>(packed.y) * values.y;
    partial += static_cast<float>(packed.z) * values.z;
    partial += static_cast<float>(packed.w) * values.w;
  }
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial * scales[row];
}

__global__ void int8_gemv_grouped_inputs_batch_reuse_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t rows_per_group, std::uint32_t groups,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto group = row / rows_per_group;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  float partial[kMaximumWeightReuseBatch]{};
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    for (std::uint32_t request = 0U; request < batch; ++request) {
      const auto* activation = input +
          (static_cast<std::size_t>(request) * groups + group) * columns;
      const auto values =
          *reinterpret_cast<const float4*>(activation + column);
      partial[request] += static_cast<float>(packed.x) * values.x;
      partial[request] += static_cast<float>(packed.y) * values.y;
      partial[request] += static_cast<float>(packed.z) * values.z;
      partial[request] += static_cast<float>(packed.w) * values.w;
    }
  }
  for (std::uint32_t request = 0U; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] =
          sum * scales[row];
  }
}

__global__ void int8_gemv_batch_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto work = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const auto row = static_cast<std::uint32_t>(work / batch);
  const auto request = static_cast<std::uint32_t>(work % batch);
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(request) * columns;
  float partial = 0.0F;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += static_cast<float>(weight[i]) * activation[i];
  partial = warp_sum(partial);
  if (lane == 0)
    output[static_cast<std::size_t>(request) * rows + row] =
        partial * scales[row];
}

// Vectorized twin of int8_gemv_batch_kernel: identical scales/groups
// semantics, but each lane consumes four weights and four activations per
// iteration through char4/float4 loads. Requires columns % 4 == 0; row and
// request strides then stay naturally aligned for both vector types.
__global__ void int8_gemv_batch_vector_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto work = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const auto row = static_cast<std::uint32_t>(work / batch);
  const auto request = static_cast<std::uint32_t>(work % batch);
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(request) * columns;
  float partial = 0.0F;
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    const auto values =
        *reinterpret_cast<const float4*>(activation + column);
    partial += static_cast<float>(packed.x) * values.x;
    partial += static_cast<float>(packed.y) * values.y;
    partial += static_cast<float>(packed.z) * values.z;
    partial += static_cast<float>(packed.w) * values.w;
  }
  partial = warp_sum(partial);
  if (lane == 0)
    output[static_cast<std::size_t>(request) * rows + row] =
        partial * scales[row];
}

__global__ void int8_gemv_batch_weight_reuse_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  float partial[kMaximumWeightReuseBatch]{};
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t column = lane; column < columns; column += kWarpSize) {
    const auto value = static_cast<float>(weight[column]);
    for (std::uint32_t request = 0; request < batch; ++request) {
      partial[request] +=
          value * input[static_cast<std::size_t>(request) * columns + column];
    }
  }
  for (std::uint32_t request = 0; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0) {
      output[static_cast<std::size_t>(request) * rows + row] =
          sum * scales[row];
    }
  }
}

__global__ void f32_gemv_kernel(const float* weights, const float* input,
                                float* output, std::uint32_t rows,
                                std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += weight[i] * input[i];
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial;
}

__global__ void f32_gemv_batch_kernel(
    const float* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto work = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const auto row = static_cast<std::uint32_t>(work / batch);
  const auto request = static_cast<std::uint32_t>(work % batch);
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(request) * columns;
  float partial = 0.0F;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += weight[i] * activation[i];
  partial = warp_sum(partial);
  if (lane == 0)
    output[static_cast<std::size_t>(request) * rows + row] = partial;
}

__global__ void f32_gemv_batch_reuse_kernel(
    const float* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  float partial[kMaximumWeightReuseBatch]{};
  for (std::uint32_t column = lane; column < columns;
       column += kWarpSize) {
    const auto value = weight[column];
    for (std::uint32_t request = 0U; request < batch; ++request)
      partial[request] += value *
          input[static_cast<std::size_t>(request) * columns + column];
  }
  for (std::uint32_t request = 0U; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] = sum;
  }
}

__global__ void bf16_gemv_kernel(const std::uint16_t* weights,
                                  const float* input, float* output,
                                  std::uint32_t rows,
                                  std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += __uint_as_float(static_cast<unsigned>(weight[i]) << 16U) * input[i];
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial;
}

__global__ void bf16_gemv_batch_reuse_kernel(
    const std::uint16_t* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  float partial[kMaximumWeightReuseBatch]{};
  for (std::uint32_t column = lane; column < columns;
       column += kWarpSize) {
    const auto value =
        __uint_as_float(static_cast<unsigned>(weight[column]) << 16U);
    for (std::uint32_t request = 0U; request < batch; ++request)
      partial[request] += value *
          input[static_cast<std::size_t>(request) * columns + column];
  }
  for (std::uint32_t request = 0U; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] = sum;
  }
}

__global__ void rms_kernel(const float* input, const float* weight,
                           float* output, std::uint32_t count, float epsilon) {
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) output[i] = input[i] * inverse * weight[i];
}

__global__ void rms_bf16_weight_kernel(const float* input,
                                        const std::uint16_t* weight,
                                        float* output, std::uint32_t count,
                                        float epsilon) {
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const float scale =
        __uint_as_float(static_cast<unsigned>(weight[i]) << 16U);
    output[i] = input[i] * inverse * scale;
  }
}

__global__ void rms_bf16_weight_batch_kernel(
    const float* input, const std::uint16_t* weight, float* output,
    std::uint32_t count, float epsilon) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  input += static_cast<std::size_t>(row) * count;
  output += static_cast<std::size_t>(row) * count;
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const float scale =
        __uint_as_float(static_cast<unsigned>(weight[i]) << 16U);
    output[i] = input[i] * inverse * scale;
  }
}

__global__ void qwen_rms_kernel(const float* input, const float* weight,
                                float* output, std::uint32_t count,
                                float epsilon) {
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    output[i] = input[i] * inverse * (1.0F + weight[i]);
}

__global__ void add_kernel(float* destination, const float* source,
                           std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) destination[i] += source[i];
}

__global__ void silu_product_kernel(const float* gate, const float* up,
                                    float* output, std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) output[i] = (gate[i] / (1.0F + expf(-gate[i]))) * up[i];
}

__global__ void sigmoid_scale_kernel(float* values, const float* gate,
                                     std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) values[i] *= 1.0F / (1.0F + expf(-gate[0]));
}

__global__ void qkv_rope_kernel(float* query, float* key, const float* value,
                                float* key_cache, float* value_cache,
                                std::uint32_t position, std::uint32_t heads,
                                std::uint32_t head_dim, float theta) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head >= heads || dimension >= head_dim) return;
  const auto base = static_cast<std::size_t>(head) * head_dim;
  __shared__ float normalized_q[128];
  __shared__ float normalized_k[128];
  normalized_q[dimension] = query[base + dimension];
  normalized_k[dimension] = key[base + dimension];
  __syncthreads();
  const auto half = head_dim / 2U;
  const auto frequency_index = dimension % half;
  const float inverse_frequency = powf(theta, -2.0F * static_cast<float>(frequency_index) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(position) * inverse_frequency;
  const float cosine = cosf(angle), sine = sinf(angle);
  const float q_rotated = dimension < half ? -normalized_q[dimension + half] : normalized_q[dimension - half];
  const float k_rotated = dimension < half ? -normalized_k[dimension + half] : normalized_k[dimension - half];
  query[base + dimension] = normalized_q[dimension] * cosine + q_rotated * sine;
  const float final_key = normalized_k[dimension] * cosine + k_rotated * sine;
  key[base + dimension] = final_key;
  const auto cache = (static_cast<std::size_t>(position) * heads + head) * head_dim + dimension;
  key_cache[cache] = final_key;
  value_cache[cache] = value[base + dimension];
}

__global__ void attention_kernel(const float* query, const float* key_cache,
                                 const float* value_cache, float* output,
                                 std::uint32_t tokens, std::uint32_t heads,
                                 std::uint32_t head_dim) {
  extern __shared__ float scores[];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto q_base = static_cast<std::size_t>(head) * head_dim;
  for (std::uint32_t token = threadIdx.x; token < tokens; token += blockDim.x) {
    const auto cache_base = (static_cast<std::size_t>(token) * heads + head) * head_dim;
    float dot = 0.0F;
    for (std::uint32_t d = 0; d < head_dim; ++d) dot += query[q_base + d] * key_cache[cache_base + d];
    scores[token] = dot * rsqrtf(static_cast<float>(head_dim));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float maximum = kNegativeInfinity;
    for (std::uint32_t token = 0; token < tokens; ++token) maximum = fmaxf(maximum, scores[token]);
    float denominator = 0.0F;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      scores[token] = expf(scores[token] - maximum);
      denominator += scores[token];
    }
    scores[tokens] = denominator;
  }
  __syncthreads();
  if (dimension < head_dim) {
    float result = 0.0F;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      const auto cache = (static_cast<std::size_t>(token) * heads + head) * head_dim + dimension;
      result += (scores[token] / scores[tokens]) * value_cache[cache];
    }
    output[q_base + dimension] = result;
  }
}

__global__ void router_logits_kernel(const float* input, const float* weights,
                                     float* logits, std::uint32_t hidden,
                                     std::uint32_t experts) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x);
  if (expert >= experts) return;
  float partial = 0.0F;
  const auto* row = weights + static_cast<std::size_t>(expert) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) partial += row[i] * input[i];
  partial = reduce_sum(partial);
  if (threadIdx.x == 0) logits[expert] = partial;
}

__global__ void router_select_kernel(const float* logits, std::uint32_t experts,
                                     std::uint32_t top_k, float* scores,
                                     std::uint32_t* indices,
                                     bool normalize_selected) {
  if (threadIdx.x != 0) return;
  if (normalize_selected) {
    float selected_maximum = kNegativeInfinity;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      float best = kNegativeInfinity;
      std::uint32_t best_index = 0;
      for (std::uint32_t i = 0; i < experts; ++i) {
        bool used = false;
        for (std::uint32_t previous = 0; previous < slot; ++previous)
          used |= indices[previous] == i;
        if (!used && logits[i] > best) {
          best = logits[i];
          best_index = i;
        }
      }
      scores[slot] = best;
      indices[slot] = best_index;
      selected_maximum = fmaxf(selected_maximum, best);
    }
    float selected_sum = 0.0F;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      scores[slot] = expf(scores[slot] - selected_maximum);
      selected_sum += scores[slot];
    }
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      scores[slot] /= selected_sum;
    return;
  }
  float maximum = kNegativeInfinity;
  for (std::uint32_t i = 0; i < experts; ++i) maximum = fmaxf(maximum, logits[i]);
  float denominator = 0.0F;
  for (std::uint32_t i = 0; i < experts; ++i) denominator += expf(logits[i] - maximum);
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    float best = -1.0F;
    std::uint32_t best_index = 0;
    for (std::uint32_t i = 0; i < experts; ++i) {
      bool used = false;
      for (std::uint32_t previous = 0; previous < slot; ++previous) used |= indices[previous] == i;
      const float probability = expf(logits[i] - maximum) / denominator;
      if (!used && probability > best) { best = probability; best_index = i; }
    }
    scores[slot] = best;
    indices[slot] = best_index;
  }
}

__global__ void router_select_batch_kernel(
    const float* logits, std::uint32_t experts, std::uint32_t top_k,
    float* scores, std::uint32_t* indices) {
  const auto request = static_cast<std::uint32_t>(blockIdx.x);
  logits += static_cast<std::size_t>(request) * experts;
  scores += static_cast<std::size_t>(request) * top_k;
  indices += static_cast<std::size_t>(request) * top_k;
  __shared__ float candidate_values[kThreads];
  __shared__ std::uint32_t candidate_indices[kThreads];
  __shared__ float selected_values[kMaximumRouterTopK];
  __shared__ std::uint32_t selected_indices[kMaximumRouterTopK];
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    float best = kNegativeInfinity;
    std::uint32_t best_index = 0xffffffffU;
    for (std::uint32_t i = threadIdx.x; i < experts; i += blockDim.x) {
      bool used = false;
      for (std::uint32_t previous = 0; previous < slot; ++previous)
        used |= selected_indices[previous] == i;
      if (!used && (logits[i] > best ||
                    (logits[i] == best && i < best_index))) {
        best = logits[i];
        best_index = i;
      }
    }
    candidate_values[threadIdx.x] = best;
    candidate_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
      if (threadIdx.x < stride) {
        const auto other_value = candidate_values[threadIdx.x + stride];
        const auto other_index = candidate_indices[threadIdx.x + stride];
        if (other_value > candidate_values[threadIdx.x] ||
            (other_value == candidate_values[threadIdx.x] &&
             other_index < candidate_indices[threadIdx.x])) {
          candidate_values[threadIdx.x] = other_value;
          candidate_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected_values[slot] = candidate_values[0];
      selected_indices[slot] = candidate_indices[0];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float selected_maximum = kNegativeInfinity;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      selected_maximum = fmaxf(selected_maximum, selected_values[slot]);
    float selected_sum = 0.0F;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      scores[slot] = expf(selected_values[slot] - selected_maximum);
      indices[slot] = selected_indices[slot];
      selected_sum += scores[slot];
    }
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      scores[slot] /= selected_sum;
  }
}

__device__ float deepseek_route_score(float logit) {
  const float softplus = logit > 20.0F ? logit : log1pf(expf(logit));
  return sqrtf(softplus);
}

__global__ void deepseek_hash_router_kernel(
    const float* logits, const std::int64_t* token_experts,
    std::uint32_t token_id, float route_scale, float* scores,
    std::uint32_t* indices) {
  const auto slot = static_cast<std::uint32_t>(threadIdx.x);
  __shared__ float selected[6];
  if (slot < 6U) {
    const auto expert = token_experts[
        static_cast<std::size_t>(token_id) * 6U + slot];
    if (expert < 0 || expert >= 256) {
      indices[slot] = 0xffffffffU;
      selected[slot] = 0.0F;
    } else {
      indices[slot] = static_cast<std::uint32_t>(expert);
      selected[slot] = deepseek_route_score(logits[expert]);
    }
  }
  __syncthreads();
  if (slot == 0U) {
    float total = 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      total += selected[index];
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      scores[index] = selected[index] * multiplier;
  }
}

__global__ void deepseek_learned_router_kernel(
    const float* logits, const float* bias, float route_scale,
    float* scores, std::uint32_t* indices) {
  __shared__ float candidates[kThreads];
  __shared__ std::uint32_t candidate_indices[kThreads];
  __shared__ float selected_scores[6];
  __shared__ std::uint32_t selected_indices[6];
  const auto expert = static_cast<std::uint32_t>(threadIdx.x);
  const float unbiased = deepseek_route_score(logits[expert]);
  for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
    bool used = false;
    for (std::uint32_t previous = 0U; previous < slot; ++previous)
      used |= selected_indices[previous] == expert;
    candidates[expert] = used ? kNegativeInfinity : unbiased + bias[expert];
    candidate_indices[expert] = expert;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride; stride >>= 1U) {
      if (expert < stride) {
        const auto other_value = candidates[expert + stride];
        const auto other_index = candidate_indices[expert + stride];
        if (other_value > candidates[expert] ||
            (other_value == candidates[expert] &&
             other_index < candidate_indices[expert])) {
          candidates[expert] = other_value;
          candidate_indices[expert] = other_index;
        }
      }
      __syncthreads();
    }
    if (expert == 0U) {
      selected_indices[slot] = candidate_indices[0];
      selected_scores[slot] =
          deepseek_route_score(logits[selected_indices[slot]]);
    }
    __syncthreads();
  }
  if (expert == 0U) {
    float total = 0.0F;
    for (const float value : selected_scores) total += value;
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
      indices[slot] = selected_indices[slot];
      scores[slot] = selected_scores[slot] * multiplier;
    }
  }
}

__global__ void deepseek_hash_router_batch_kernel(
    const float* logits, const std::int64_t* token_experts,
    std::uint32_t token_zero, std::uint32_t token_one, float route_scale,
    float* scores, std::uint32_t* indices) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  const auto slot = static_cast<std::uint32_t>(threadIdx.x);
  const auto token_id = row == 0U ? token_zero : token_one;
  logits += static_cast<std::size_t>(row) * 256U;
  scores += static_cast<std::size_t>(row) * 6U;
  indices += static_cast<std::size_t>(row) * 6U;
  __shared__ float selected[6];
  if (slot < 6U) {
    const auto expert = token_experts[
        static_cast<std::size_t>(token_id) * 6U + slot];
    if (expert < 0 || expert >= 256) {
      indices[slot] = 0xffffffffU;
      selected[slot] = 0.0F;
    } else {
      indices[slot] = static_cast<std::uint32_t>(expert);
      selected[slot] = deepseek_route_score(logits[expert]);
    }
  }
  __syncthreads();
  if (slot == 0U) {
    float total = 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      total += selected[index];
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      scores[index] = selected[index] * multiplier;
  }
}

__global__ void deepseek_learned_router_batch_kernel(
    const float* logits, const float* bias, float route_scale,
    float* scores, std::uint32_t* indices) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  const auto expert = static_cast<std::uint32_t>(threadIdx.x);
  logits += static_cast<std::size_t>(row) * 256U;
  scores += static_cast<std::size_t>(row) * 6U;
  indices += static_cast<std::size_t>(row) * 6U;
  __shared__ float candidates[kThreads];
  __shared__ std::uint32_t candidate_indices[kThreads];
  __shared__ float selected_scores[6];
  __shared__ std::uint32_t selected_indices[6];
  const float unbiased = deepseek_route_score(logits[expert]);
  for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
    bool used = false;
    for (std::uint32_t previous = 0U; previous < slot; ++previous)
      used |= selected_indices[previous] == expert;
    candidates[expert] = used ? kNegativeInfinity : unbiased + bias[expert];
    candidate_indices[expert] = expert;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride; stride >>= 1U) {
      if (expert < stride) {
        const auto other_value = candidates[expert + stride];
        const auto other_index = candidate_indices[expert + stride];
        if (other_value > candidates[expert] ||
            (other_value == candidates[expert] &&
             other_index < candidate_indices[expert])) {
          candidates[expert] = other_value;
          candidate_indices[expert] = other_index;
        }
      }
      __syncthreads();
    }
    if (expert == 0U) {
      selected_indices[slot] = candidate_indices[0];
      selected_scores[slot] =
          deepseek_route_score(logits[selected_indices[slot]]);
    }
    __syncthreads();
  }
  if (expert == 0U) {
    float total = 0.0F;
    for (const float value : selected_scores) total += value;
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
      indices[slot] = selected_indices[slot];
      scores[slot] = selected_scores[slot] * multiplier;
    }
  }
}

__global__ void qwen_qkv_rope_kernel(
    float* q_and_gate, float* key, const float* value,
    const float* q_weight, const float* k_weight, float* key_cache,
    float* value_cache, std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head < query_heads) {
    auto* query = q_and_gate + static_cast<std::size_t>(head) * 2U * head_dim;
    float square = dimension < head_dim ? query[dimension] * query[dimension] : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim) {
      const float normalized_value = query[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + q_weight[dimension]);
      normalized[dimension] = normalized_value;
    }
    __syncthreads();
    if (dimension < head_dim) {
      float final_query = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_query = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      query[dimension] = final_query;
    }
  }
  __syncthreads();
  if (head < kv_heads) {
    auto* key_head = key + static_cast<std::size_t>(head) * head_dim;
    float square = dimension < head_dim
                       ? key_head[dimension] * key_head[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim) {
      normalized[dimension] = key_head[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + k_weight[dimension]);
    }
    __syncthreads();
    if (dimension < head_dim) {
      float final_key = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_key = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      key_head[dimension] = final_key;
      const auto cache =
          (static_cast<std::size_t>(position) * kv_heads + head) * head_dim +
          dimension;
      key_cache[cache] = final_key;
      value_cache[cache] =
          value[static_cast<std::size_t>(head) * head_dim + dimension];
    }
  }
}

__global__ void qwen_attention_kernel(
    const float* q_and_gate, const float* key_cache,
    const float* value_cache, float* output, std::uint32_t tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  __shared__ float online_maximum;
  __shared__ float online_denominator;
  __shared__ float previous_scale;
  __shared__ float token_scale;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto kv_head = query_head / (query_heads / kv_heads);
  const auto query = q_and_gate +
      static_cast<std::size_t>(query_head) * 2U * head_dim;
  if (threadIdx.x == 0) {
    online_maximum = kNegativeInfinity;
    online_denominator = 0.0F;
  }
  __syncthreads();
  float result = 0.0F;
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto cache =
        (static_cast<std::size_t>(token) * kv_heads + kv_head) * head_dim;
    float dot = threadIdx.x < head_dim
                    ? query[threadIdx.x] * key_cache[cache + threadIdx.x]
                    : 0.0F;
    dot = reduce_sum(dot) * rsqrtf(static_cast<float>(head_dim));
    if (threadIdx.x == 0) {
      const float next_maximum = fmaxf(online_maximum, dot);
      previous_scale = expf(online_maximum - next_maximum);
      token_scale = expf(dot - next_maximum);
      online_denominator = online_denominator * previous_scale + token_scale;
      online_maximum = next_maximum;
    }
    __syncthreads();
    if (threadIdx.x < head_dim)
      result = result * previous_scale + token_scale *
          value_cache[cache + threadIdx.x];
    __syncthreads();
  }
  if (threadIdx.x < head_dim) {
    const float gate = query[head_dim + threadIdx.x];
    output[static_cast<std::size_t>(query_head) * head_dim + threadIdx.x] =
        (result / online_denominator) / (1.0F + expf(-gate));
  }
}

__global__ void qwen_qkv_rope_paged_fp16_kernel(
    float* q_and_gate, float* key, const float* value,
    const float* q_weight, const float* k_weight, __half* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head < query_heads) {
    auto* query = q_and_gate + static_cast<std::size_t>(head) * 2U * head_dim;
    float square = dimension < head_dim ? query[dimension] * query[dimension] : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = query[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + q_weight[dimension]);
    __syncthreads();
    if (dimension < head_dim) {
      float final_query = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_query = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      query[dimension] = final_query;
    }
  }
  __syncthreads();
  if (head < kv_heads) {
    auto* key_head = key + static_cast<std::size_t>(head) * head_dim;
    float square = dimension < head_dim
                       ? key_head[dimension] * key_head[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = key_head[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + k_weight[dimension]);
    __syncthreads();
    if (dimension < head_dim) {
      float final_key = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_key = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      key_head[dimension] = final_key;
      const auto page_elements = static_cast<std::size_t>(page_tokens) *
                                 kv_heads * head_dim;
      auto* key_page = page + static_cast<std::size_t>(full_attention_layer) *
                                  2U * page_elements;
      auto* value_page = key_page + page_elements;
      const auto cache =
          (static_cast<std::size_t>(position % page_tokens) * kv_heads + head) *
              head_dim + dimension;
      key_page[cache] = __float2half_rn(final_key);
      value_page[cache] = __float2half_rn(
          value[static_cast<std::size_t>(head) * head_dim + dimension]);
    }
  }
}

__global__ void qwen_attention_paged_fp16_kernel(
    const float* q_and_gate, const void* const* page_table, float* output,
    std::uint32_t tokens, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
  __shared__ float online_maximum;
  __shared__ float online_denominator;
  __shared__ float previous_scale;
  __shared__ float token_scale;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto kv_head = query_head / (query_heads / kv_heads);
  const auto* query = q_and_gate +
      static_cast<std::size_t>(query_head) * 2U * head_dim;
  const auto page_elements = static_cast<std::size_t>(page_tokens) *
                             kv_heads * head_dim;
  if (threadIdx.x == 0) {
    online_maximum = kNegativeInfinity;
    online_denominator = 0.0F;
  }
  __syncthreads();
  float result = 0.0F;
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto* page = static_cast<const __half*>(page_table[token / page_tokens]);
    const auto* key_page = page +
        static_cast<std::size_t>(full_attention_layer) * 2U * page_elements;
    const auto* value_page = key_page + page_elements;
    const auto cache =
        (static_cast<std::size_t>(token % page_tokens) * kv_heads + kv_head) *
            head_dim + threadIdx.x;
    float dot = threadIdx.x < head_dim
                    ? query[threadIdx.x] * __half2float(key_page[cache])
                    : 0.0F;
    dot = reduce_sum(dot) * rsqrtf(static_cast<float>(head_dim));
    if (threadIdx.x == 0) {
      const float next_maximum = fmaxf(online_maximum, dot);
      previous_scale = expf(online_maximum - next_maximum);
      token_scale = expf(dot - next_maximum);
      online_denominator = online_denominator * previous_scale + token_scale;
      online_maximum = next_maximum;
    }
    __syncthreads();
    if (threadIdx.x < head_dim)
      result = result * previous_scale + token_scale *
          __half2float(value_page[cache]);
    __syncthreads();
  }
  if (threadIdx.x < head_dim) {
    const float gate = query[head_dim + threadIdx.x];
    output[static_cast<std::size_t>(query_head) * head_dim + threadIdx.x] =
        (result / online_denominator) / (1.0F + expf(-gate));
  }
}

__global__ void qwen_delta_conv_kernel(
    const float* projected, const float* weights, float* state,
    float* output, std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t key_dim, std::uint32_t value_dim,
    std::uint32_t key_head_dim, std::uint32_t value_head_dim,
    std::uint32_t kernel) {
  const auto channel =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const auto conv_dim = 2U * key_dim + value_dim;
  if (channel >= conv_dim) return;
  const auto ratio = value_heads / key_heads;
  const auto group_width =
      2U * key_head_dim + 2U * ratio * value_head_dim;
  std::uint32_t source = 0;
  if (channel < key_dim) {
    const auto group = channel / key_head_dim;
    source = group * group_width + channel % key_head_dim;
  } else if (channel < 2U * key_dim) {
    const auto local = channel - key_dim;
    const auto group = local / key_head_dim;
    source = group * group_width + key_head_dim + local % key_head_dim;
  } else {
    const auto local = channel - 2U * key_dim;
    const auto value_head = local / value_head_dim;
    const auto group = value_head / ratio;
    const auto replica = value_head % ratio;
    source = group * group_width + 2U * key_head_dim +
             replica * value_head_dim + local % value_head_dim;
  }
  auto* channel_state = state + static_cast<std::size_t>(channel) * kernel;
  for (std::uint32_t i = 1; i < kernel; ++i)
    channel_state[i - 1] = channel_state[i];
  channel_state[kernel - 1] = projected[source];
  const auto* channel_weights =
      weights + static_cast<std::size_t>(channel) * kernel;
  float sum = 0.0F;
  for (std::uint32_t i = 0; i < kernel; ++i)
    sum += channel_state[i] * channel_weights[i];
  output[channel] = sum / (1.0F + expf(-sum));
}

__global__ void qwen_delta_recurrent_kernel(
    const float* projected_qkvz, const float* projected_ba,
    const float* conv, const float* dt_bias, const float* a_log,
    const float* norm_weight, float* recurrent, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t key_head_dim, std::uint32_t value_head_dim,
    float epsilon) {
  __shared__ float query[256];
  __shared__ float key[256];
  __shared__ float per_dimension[256];
  const auto value_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto d = static_cast<std::uint32_t>(threadIdx.x);
  const auto ratio = value_heads / key_heads;
  const auto key_head = value_head / ratio;
  const auto key_dim = key_heads * key_head_dim;
  if (d < key_head_dim) {
    query[d] = conv[static_cast<std::size_t>(key_head) * key_head_dim + d];
    key[d] = conv[key_dim + static_cast<std::size_t>(key_head) * key_head_dim + d];
  }
  __syncthreads();
  float q_square = d < key_head_dim ? query[d] * query[d] : 0.0F;
  float k_square = d < key_head_dim ? key[d] * key[d] : 0.0F;
  q_square = reduce_sum(q_square);
  k_square = reduce_sum(k_square);
  if (d < key_head_dim) {
    query[d] *= rsqrtf(q_square + 1.0e-6F) *
                rsqrtf(static_cast<float>(key_head_dim));
    key[d] *= rsqrtf(k_square + 1.0e-6F);
  }
  __syncthreads();
  const auto group = value_head / ratio;
  const auto replica = value_head % ratio;
  const auto ba_stride = 2U * ratio;
  const float beta = 1.0F /
      (1.0F + expf(-projected_ba[group * ba_stride + replica]));
  const float a = projected_ba[group * ba_stride + ratio + replica];
  const float softplus = log1pf(expf(-fabsf(a + dt_bias[value_head]))) +
                         fmaxf(a + dt_bias[value_head], 0.0F);
  const float decay = expf(-expf(a_log[value_head]) * softplus);
  float core = 0.0F;
  if (d < value_head_dim) {
    auto* column = recurrent +
        static_cast<std::size_t>(value_head) * key_head_dim * value_head_dim + d;
    float memory = 0.0F;
    for (std::uint32_t i = 0; i < key_head_dim; ++i) {
      column[static_cast<std::size_t>(i) * value_head_dim] *= decay;
      memory += column[static_cast<std::size_t>(i) * value_head_dim] * key[i];
    }
    const float value = conv[2U * key_dim +
                             static_cast<std::size_t>(value_head) * value_head_dim + d];
    const float delta = (value - memory) * beta;
    for (std::uint32_t i = 0; i < key_head_dim; ++i) {
      auto& state_value = column[static_cast<std::size_t>(i) * value_head_dim];
      state_value += key[i] * delta;
      core += state_value * query[i];
    }
    per_dimension[d] = core * core;
  } else {
    per_dimension[d] = 0.0F;
  }
  __syncthreads();
  float square = reduce_sum(per_dimension[d]);
  if (d < value_head_dim) {
    const auto group_width =
        2U * key_head_dim + 2U * ratio * value_head_dim;
    const auto z_index = group * group_width + 2U * key_head_dim +
                         ratio * value_head_dim + replica * value_head_dim + d;
    const float z = projected_qkvz[z_index];
    const float gated = z / (1.0F + expf(-z));
    output[static_cast<std::size_t>(value_head) * value_head_dim + d] =
        core * rsqrtf(square / static_cast<float>(value_head_dim) + epsilon) *
        norm_weight[d] * gated;
  }
}

__global__ void argmax_batch_kernel(const float* values, std::uint32_t count,
                                    std::uint32_t* output) {
  const auto request = static_cast<std::uint32_t>(blockIdx.x);
  values += static_cast<std::size_t>(request) * count;
  __shared__ float best_values[kThreads];
  __shared__ std::uint32_t best_indices[kThreads];
  float best = kNegativeInfinity;
  std::uint32_t index = 0xffffffffU;
  for (std::uint32_t item = threadIdx.x; item < count;
       item += blockDim.x) {
    const auto value = values[item];
    if (value > best || (value == best && item < index)) {
      best = value;
      index = item;
    }
  }
  best_values[threadIdx.x] = best;
  best_indices[threadIdx.x] = index;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const auto other_value = best_values[threadIdx.x + stride];
      const auto other_index = best_indices[threadIdx.x + stride];
      if (other_value > best_values[threadIdx.x] ||
          (other_value == best_values[threadIdx.x] &&
           other_index < best_indices[threadIdx.x])) {
        best_values[threadIdx.x] = other_value;
        best_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[request] = best_indices[0];
}

Status checked(cudaError_t error, const char* name) {
  return error == cudaSuccess ? Status::success() : Status(ErrorCode::upload_failed, std::string(name) + ": " + cudaGetErrorString(error));
}

}  // namespace

Status embedding(const Int8Matrix& m, std::uint32_t token, float* output, void* raw) noexcept {
  if (!m.weights || !m.scales || !output || token >= m.rows || !m.columns) return Status(ErrorCode::invalid_argument, "invalid embedding");
  embedding_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(m.weights, m.scales, token, m.columns, output);
  return checked(cudaPeekAtLastError(), "embedding");
}
Status gemv(const Int8Matrix& m, const float* input, float* output, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows || !m.columns) return Status(ErrorCode::invalid_argument, "invalid gemv");
  const auto blocks = (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (m.columns % 4U == 0U) {
    int8_gemv_vector_kernel<<<blocks, kThreads, 0,
                              static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns);
  } else {
    int8_gemv_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns);
  }
  return checked(cudaPeekAtLastError(), "int8 gemv");
}
Status gemv_grouped_inputs(const Int8Matrix& m, const float* input,
                           float* output, std::uint32_t groups,
                           void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !groups || m.rows % groups != 0U ||
      m.columns % 4U != 0U)
    return Status(ErrorCode::invalid_argument,
                  "invalid grouped-input gemv");
  const auto blocks = (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  int8_gemv_grouped_inputs_vector_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, output, m.rows, m.columns,
      m.rows / groups);
  return checked(cudaPeekAtLastError(), "grouped-input int8 gemv");
}
Status gemv_grouped_inputs_batch_weight_reuse(
    const Int8Matrix& m, const float* input, float* output,
    std::uint32_t groups, std::uint32_t batch, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !groups || m.rows % groups != 0U ||
      m.columns % 4U != 0U || !batch || batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid grouped-input weight-reuse batched gemv");
  const auto blocks = (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  int8_gemv_grouped_inputs_batch_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, output, m.rows, m.columns,
      m.rows / groups, groups, batch);
  return checked(cudaPeekAtLastError(),
                 "grouped-input weight-reuse int8 batched gemv");
}
Status gemv_batch(const Int8Matrix& m, const float* input, float* output,
                  std::uint32_t batch, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !batch)
    return Status(ErrorCode::invalid_argument, "invalid batched gemv");
  const auto blocks = static_cast<std::uint64_t>(m.rows) * batch;
  if (blocks > 0xffffffffULL)
    return Status(ErrorCode::invalid_argument, "batched gemv grid too large");
  const auto grid = (blocks + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (m.columns % 4U == 0U) {
    int8_gemv_batch_vector_kernel<<<static_cast<unsigned>(grid), kThreads, 0,
                                    static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns, batch);
  } else {
    int8_gemv_batch_kernel<<<static_cast<unsigned>(grid), kThreads, 0,
                             static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns, batch);
  }
  return checked(cudaPeekAtLastError(), "int8 batched gemv");
}
Status gemv_batch_weight_reuse(const Int8Matrix& m, const float* input,
                               float* output, std::uint32_t batch,
                               void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !batch || batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched gemv");
  const auto blocks =
      (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  int8_gemv_batch_weight_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, output, m.rows, m.columns, batch);
  return checked(cudaPeekAtLastError(), "weight-reuse int8 batched gemv");
}
Status gemv_f32(const float* matrix, std::uint32_t rows, std::uint32_t columns, const float* input, float* output, void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns) return Status(ErrorCode::invalid_argument, "invalid f32 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  f32_gemv_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(matrix, input, output, rows, columns);
  return checked(cudaPeekAtLastError(), "f32 gemv");
}
Status gemv_f32_batch(const float* matrix, std::uint32_t rows,
                      std::uint32_t columns, const float* input, float* output,
                      std::uint32_t batch, void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch)
    return Status(ErrorCode::invalid_argument, "invalid batched f32 gemv");
  const auto blocks = static_cast<std::uint64_t>(rows) * batch;
  if (blocks > 0xffffffffULL)
    return Status(ErrorCode::invalid_argument,
                  "batched f32 gemv grid too large");
  const auto grid = (blocks + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  f32_gemv_batch_kernel<<<static_cast<unsigned>(grid), kThreads, 0,
                          static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns, batch);
  return checked(cudaPeekAtLastError(), "f32 batched gemv");
}
Status gemv_f32_batch_weight_reuse(
    const float* matrix, std::uint32_t rows, std::uint32_t columns,
    const float* input, float* output, std::uint32_t batch,
    void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch ||
      batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched f32 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  f32_gemv_batch_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns, batch);
  return checked(cudaPeekAtLastError(),
                 "weight-reuse f32 batched gemv");
}
Status gemv_bf16(const std::uint16_t* matrix, std::uint32_t rows,
                  std::uint32_t columns, const float* input, float* output,
                  void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns)
    return Status(ErrorCode::invalid_argument, "invalid bf16 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  bf16_gemv_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns);
  return checked(cudaPeekAtLastError(), "bf16 gemv");
}
Status gemv_bf16_batch(const std::uint16_t* matrix, std::uint32_t rows,
                       std::uint32_t columns, const float* input,
                       float* output, std::uint32_t batch,
                       void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch ||
      batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched bf16 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  bf16_gemv_batch_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns, batch);
  return checked(cudaPeekAtLastError(), "weight-reuse bf16 batched gemv");
}
Status rms_norm(const float* input, const float* weight, float* output, std::uint32_t elements, float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !elements || epsilon <= 0) return Status(ErrorCode::invalid_argument, "invalid rms norm");
  rms_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "rms norm");
}
Status rms_norm_bf16_weight(const float* input, const std::uint16_t* weight,
                            float* output, std::uint32_t elements,
                            float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !elements || epsilon <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid bf16-weight rms norm");
  rms_bf16_weight_kernel<<<1, kThreads, 0,
                           static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "bf16-weight rms norm");
}
Status rms_norm_bf16_weight_batch(
    const float* input, const std::uint16_t* weight, float* output,
    std::uint32_t rows, std::uint32_t elements, float epsilon,
    void* raw) noexcept {
  if (!input || !weight || !output || !rows || !elements || epsilon <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid batched bf16-weight rms norm");
  rms_bf16_weight_batch_kernel<<<rows, kThreads, 0,
                                 static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "batched bf16-weight rms norm");
}
Status qwen3_next_rms_norm(const float* input, const float* weight,
                           float* output, std::uint32_t elements,
                           float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !elements || epsilon <= 0)
    return Status(ErrorCode::invalid_argument, "invalid Qwen3-Next rms norm");
  qwen_rms_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "Qwen3-Next rms norm");
}
Status add_in_place(float* destination, const float* source, std::uint32_t elements, void* raw) noexcept {
  if (!destination || !source || !elements) return Status(ErrorCode::invalid_argument, "invalid add");
  add_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(destination, source, elements);
  return checked(cudaPeekAtLastError(), "add");
}
Status silu_product(const float* gate, const float* up, float* output,
                    std::uint32_t elements, void* raw) noexcept {
  if (!gate || !up || !output || !elements)
    return Status(ErrorCode::invalid_argument, "invalid silu product");
  silu_product_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                        static_cast<cudaStream_t>(raw)>>>(gate, up, output,
                                                          elements);
  return checked(cudaPeekAtLastError(), "silu product");
}
Status sigmoid_scale_in_place(float* values, const float* gate,
                              std::uint32_t elements, void* raw) noexcept {
  if (!values || !gate || !elements)
    return Status(ErrorCode::invalid_argument, "invalid sigmoid scale");
  sigmoid_scale_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                         static_cast<cudaStream_t>(raw)>>>(values, gate,
                                                           elements);
  return checked(cudaPeekAtLastError(), "sigmoid scale");
}
Status qkv_rope_cache(float* q, float* k, const float* v, const float* qw, const float* kw, float* kc, float* vc, std::uint32_t pos, std::uint32_t heads, std::uint32_t dim, float eps, float theta, void* raw) noexcept {
  if (!q || !k || !v || !qw || !kw || !kc || !vc || !heads || dim != 128 || eps <= 0 || theta <= 0) return Status(ErrorCode::invalid_argument, "invalid qkv rope");
  auto stream = static_cast<cudaStream_t>(raw);
  const auto elements = heads * dim;
  rms_kernel<<<1, kThreads, 0, stream>>>(q, qw, q, elements, eps);
  if (auto status = checked(cudaPeekAtLastError(), "q rms norm"); !status.ok()) return status;
  rms_kernel<<<1, kThreads, 0, stream>>>(k, kw, k, elements, eps);
  if (auto status = checked(cudaPeekAtLastError(), "k rms norm"); !status.ok()) return status;
  qkv_rope_kernel<<<heads, 128, 0, stream>>>(q, k, v, kc, vc, pos, heads, dim, theta);
  return checked(cudaPeekAtLastError(), "qkv rope");
}
Status attention_decode(const float* q, const float* kc, const float* vc, float* output, std::uint32_t tokens, std::uint32_t heads, std::uint32_t dim, void* raw) noexcept {
  if (!q || !kc || !vc || !output || !tokens || !heads || dim > kThreads) return Status(ErrorCode::invalid_argument, "invalid attention");
  attention_kernel<<<heads, kThreads, (static_cast<std::size_t>(tokens) + 1) * sizeof(float), static_cast<cudaStream_t>(raw)>>>(q, kc, vc, output, tokens, heads, dim);
  return checked(cudaPeekAtLastError(), "attention");
}
Status router_topk(const float* input, const float* weights, std::uint32_t hidden, std::uint32_t experts, std::uint32_t top_k, float* logits, float* scores, std::uint32_t* indices, void* raw) noexcept {
  if (!input || !weights || !logits || !scores || !indices || !hidden || !experts || !top_k || top_k > experts) return Status(ErrorCode::invalid_argument, "invalid router");
  auto stream = static_cast<cudaStream_t>(raw);
  router_logits_kernel<<<experts, kThreads, 0, stream>>>(input, weights, logits, hidden, experts);
  if (auto status = checked(cudaPeekAtLastError(), "router logits"); !status.ok()) return status;
  router_select_kernel<<<1, 1, 0, stream>>>(logits, experts, top_k, scores,
                                             indices, false);
  return checked(cudaPeekAtLastError(), "router select");
}
Status router_topk_normalized(const float* input, const float* weights,
                              std::uint32_t hidden, std::uint32_t experts,
                              std::uint32_t top_k, float* logits,
                              float* scores, std::uint32_t* indices,
                              void* raw) noexcept {
  if (!input || !weights || !logits || !scores || !indices || !hidden ||
      !experts || !top_k || top_k > experts)
    return Status(ErrorCode::invalid_argument, "invalid normalized router");
  auto stream = static_cast<cudaStream_t>(raw);
  router_logits_kernel<<<experts, kThreads, 0, stream>>>(
      input, weights, logits, hidden, experts);
  if (auto status = checked(cudaPeekAtLastError(), "router logits");
      !status.ok())
    return status;
  router_select_kernel<<<1, 1, 0, stream>>>(logits, experts, top_k, scores,
                                             indices, true);
  return checked(cudaPeekAtLastError(), "normalized router select");
}
Status router_topk_normalized_batch(
    const float* input, const float* weights, std::uint32_t rows,
    std::uint32_t hidden, std::uint32_t experts, std::uint32_t top_k,
    float* logits, float* scores, std::uint32_t* indices,
    void* raw) noexcept {
  if (!input || !weights || !logits || !scores || !indices || !rows ||
      !hidden || !experts || !top_k || top_k > experts ||
      top_k > kMaximumRouterTopK)
    return Status(ErrorCode::invalid_argument,
                  "invalid normalized batched router");
  auto stream = static_cast<cudaStream_t>(raw);
  auto status = gemv_f32_batch(weights, experts, hidden, input, logits, rows,
                               raw);
  if (!status.ok()) return status;
  router_select_batch_kernel<<<rows, kThreads, 0, stream>>>(
      logits, experts, top_k, scores, indices);
  return checked(cudaPeekAtLastError(), "normalized batched router select");
}
Status deepseek_router_hash(
    const float* input, const std::uint16_t* weights,
    const std::int64_t* token_experts, std::uint32_t token_id,
    float* logits, float* scores, std::uint32_t* indices,
    float route_scale, void* raw) noexcept {
  if (!input || !weights || !token_experts || !logits || !scores ||
      !indices || token_id >= 129280U || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek hash router");
  auto status = gemv_bf16(weights, 256U, 4096U, input, logits, raw);
  if (!status.ok()) return status;
  deepseek_hash_router_kernel<<<1, 32, 0,
                                static_cast<cudaStream_t>(raw)>>>(
      logits, token_experts, token_id, route_scale, scores, indices);
  return checked(cudaPeekAtLastError(), "DeepSeek hash router select");
}
Status deepseek_router_learned(
    const float* input, const std::uint16_t* weights,
    const float* selection_bias, float* logits, float* scores,
    std::uint32_t* indices, float route_scale, void* raw) noexcept {
  if (!input || !weights || !selection_bias || !logits || !scores ||
      !indices || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek learned router");
  auto status = gemv_bf16(weights, 256U, 4096U, input, logits, raw);
  if (!status.ok()) return status;
  deepseek_learned_router_kernel<<<1, kThreads, 0,
                                   static_cast<cudaStream_t>(raw)>>>(
      logits, selection_bias, route_scale, scores, indices);
  return checked(cudaPeekAtLastError(), "DeepSeek learned router select");
}
Status deepseek_router_hash_batch(
    const float* input, const std::uint16_t* weights,
    const std::int64_t* token_experts, std::uint32_t token_zero,
    std::uint32_t token_one, float* logits, float* scores,
    std::uint32_t* indices, float route_scale, void* raw) noexcept {
  if (!input || !weights || !token_experts || !logits || !scores ||
      !indices || token_zero >= 129280U || token_one >= 129280U ||
      !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek batched hash router");
  auto status = gemv_bf16_batch(weights, 256U, 4096U, input, logits, 2U, raw);
  if (!status.ok()) return status;
  deepseek_hash_router_batch_kernel<<<2U, 32U, 0,
                                      static_cast<cudaStream_t>(raw)>>>(
      logits, token_experts, token_zero, token_one, route_scale, scores,
      indices);
  return checked(cudaPeekAtLastError(),
                 "DeepSeek batched hash router select");
}
Status deepseek_router_learned_batch(
    const float* input, const std::uint16_t* weights,
    const float* selection_bias, float* logits, float* scores,
    std::uint32_t* indices, float route_scale, void* raw) noexcept {
  if (!input || !weights || !selection_bias || !logits || !scores ||
      !indices || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek batched learned router");
  auto status = gemv_bf16_batch(weights, 256U, 4096U, input, logits, 2U, raw);
  if (!status.ok()) return status;
  deepseek_learned_router_batch_kernel<<<
      2U, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      logits, selection_bias, route_scale, scores, indices);
  return checked(cudaPeekAtLastError(),
                 "DeepSeek batched learned router select");
}
Status qwen3_next_qkv_rope_cache(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, float* key_cache,
    float* value_cache, std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* raw) noexcept {
  if (!q_and_gate || !key || !value || !q_norm_weight || !k_norm_weight ||
      !key_cache || !value_cache || !query_heads || !kv_heads ||
      query_heads % kv_heads || !head_dim || head_dim > kThreads ||
      !rotary_dim || rotary_dim > head_dim || rotary_dim % 2U ||
      epsilon <= 0 || rope_theta <= 0)
    return Status(ErrorCode::invalid_argument, "invalid Qwen3-Next qkv rope");
  qwen_qkv_rope_kernel<<<query_heads, kThreads, 0,
                         static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight, key_cache,
      value_cache, position, query_heads, kv_heads, head_dim, rotary_dim,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(), "Qwen3-Next qkv rope");
}
Status qwen3_next_attention_decode(
    const float* q_and_gate, const float* key_cache,
    const float* value_cache, float* output, std::uint32_t context_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* raw) noexcept {
  if (!q_and_gate || !key_cache || !value_cache || !output ||
      !context_tokens || !query_heads || !kv_heads || query_heads % kv_heads ||
      !head_dim || head_dim > kThreads)
    return Status(ErrorCode::invalid_argument,
                  "invalid Qwen3-Next attention decode");
  qwen_attention_kernel<<<query_heads, kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key_cache, value_cache, output, context_tokens, query_heads,
      kv_heads, head_dim);
  return checked(cudaPeekAtLastError(), "Qwen3-Next attention");
}
Status qwen3_next_qkv_rope_cache_paged_fp16(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* raw) noexcept {
  if (!q_and_gate || !key || !value || !q_norm_weight || !k_norm_weight ||
      !page || !page_tokens || !query_heads || !kv_heads ||
      query_heads % kv_heads || !head_dim || head_dim > kThreads ||
      !rotary_dim || rotary_dim > head_dim || rotary_dim % 2U ||
      epsilon <= 0 || rope_theta <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid paged Qwen3-Next qkv rope");
  qwen_qkv_rope_paged_fp16_kernel<<<query_heads, kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight,
      static_cast<__half*>(page), full_attention_layer, page_tokens, position,
      query_heads, kv_heads, head_dim, rotary_dim, epsilon, rope_theta);
  return checked(cudaPeekAtLastError(), "paged Qwen3-Next qkv rope");
}
Status qwen3_next_attention_decode_paged_fp16(
    const float* q_and_gate, const void* const* page_table,
    float* output, std::uint32_t context_tokens,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* raw) noexcept {
  if (!q_and_gate || !page_table || !output || !context_tokens ||
      !page_tokens || !query_heads || !kv_heads || query_heads % kv_heads ||
      !head_dim || head_dim > kThreads)
    return Status(ErrorCode::invalid_argument,
                  "invalid paged Qwen3-Next attention decode");
  qwen_attention_paged_fp16_kernel<<<query_heads, kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, page_table, output, context_tokens, full_attention_layer,
      page_tokens, query_heads, kv_heads, head_dim);
  return checked(cudaPeekAtLastError(), "paged Qwen3-Next attention");
}
Status qwen3_next_delta_decode(const Qwen3NextDeltaLaunch& launch) noexcept {
  if (!launch.projected_qkvz || !launch.projected_ba ||
      !launch.conv_weights || !launch.dt_bias || !launch.a_log ||
      !launch.norm_weight || !launch.conv_state || !launch.recurrent_state ||
      !launch.conv_output || !launch.output || !launch.key_heads ||
      !launch.value_heads || launch.value_heads % launch.key_heads ||
      !launch.key_head_dim || launch.key_head_dim > kThreads ||
      !launch.value_head_dim || launch.value_head_dim > kThreads ||
      !launch.conv_kernel || launch.conv_kernel > 16U || launch.epsilon <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid Qwen3-Next delta launch");
  const auto key_dim = launch.key_heads * launch.key_head_dim;
  const auto value_dim = launch.value_heads * launch.value_head_dim;
  const auto conv_dim = 2U * key_dim + value_dim;
  auto stream = static_cast<cudaStream_t>(launch.stream);
  qwen_delta_conv_kernel<<<(conv_dim + kThreads - 1U) / kThreads, kThreads, 0,
                            stream>>>(
      launch.projected_qkvz, launch.conv_weights, launch.conv_state,
      launch.conv_output, launch.key_heads, launch.value_heads, key_dim,
      value_dim, launch.key_head_dim, launch.value_head_dim,
      launch.conv_kernel);
  if (auto status = checked(cudaPeekAtLastError(), "Qwen3-Next delta conv");
      !status.ok())
    return status;
  qwen_delta_recurrent_kernel<<<launch.value_heads, kThreads, 0, stream>>>(
      launch.projected_qkvz, launch.projected_ba, launch.conv_output,
      launch.dt_bias, launch.a_log, launch.norm_weight,
      launch.recurrent_state, launch.output, launch.key_heads,
      launch.value_heads, launch.key_head_dim, launch.value_head_dim,
      launch.epsilon);
  return checked(cudaPeekAtLastError(), "Qwen3-Next delta recurrent");
}
Status argmax(const float* values, std::uint32_t count, std::uint32_t* output, void* raw) noexcept {
  if (!values || !count || !output) return Status(ErrorCode::invalid_argument, "invalid argmax");
  argmax_batch_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(values, count, output);
  return checked(cudaPeekAtLastError(), "argmax");
}
Status argmax_batch(const float* values, std::uint32_t count,
                    std::uint32_t batch, std::uint32_t* output,
                    void* raw) noexcept {
  if (!values || !count || !batch || !output)
    return Status(ErrorCode::invalid_argument, "invalid batched argmax");
  argmax_batch_kernel<<<batch, kThreads, 0,
                        static_cast<cudaStream_t>(raw)>>>(values, count,
                                                          output);
  return checked(cudaPeekAtLastError(), "batched argmax");
}

}  // namespace expert::runtime::cuda
