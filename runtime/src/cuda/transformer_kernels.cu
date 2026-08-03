#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr float kNegativeInfinity = -3.402823466e+38F;

__device__ float reduce_sum(float value) {
  __shared__ float shared[kThreads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  return shared[0];
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
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) {
    partial += static_cast<float>(weight[i]) * input[i];
  }
  partial = reduce_sum(partial);
  if (threadIdx.x == 0) output[row] = partial * scales[row];
}

__global__ void f32_gemv_kernel(const float* weights, const float* input,
                                float* output, std::uint32_t rows,
                                std::uint32_t columns) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) partial += weight[i] * input[i];
  partial = reduce_sum(partial);
  if (threadIdx.x == 0) output[row] = partial;
}

__global__ void rms_kernel(const float* input, const float* weight,
                           float* output, std::uint32_t count, float epsilon) {
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) output[i] = input[i] * inverse * weight[i];
}

__global__ void add_kernel(float* destination, const float* source,
                           std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) destination[i] += source[i];
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
                                     std::uint32_t* indices) {
  if (threadIdx.x != 0) return;
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

__global__ void argmax_kernel(const float* values, std::uint32_t count,
                              std::uint32_t* output) {
  if (threadIdx.x != 0) return;
  float best = kNegativeInfinity;
  std::uint32_t index = 0;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (values[i] > best) { best = values[i]; index = i; }
  }
  *output = index;
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
  int8_gemv_kernel<<<m.rows, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(m.weights, m.scales, input, output, m.rows, m.columns);
  return checked(cudaPeekAtLastError(), "int8 gemv");
}
Status gemv_f32(const float* matrix, std::uint32_t rows, std::uint32_t columns, const float* input, float* output, void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns) return Status(ErrorCode::invalid_argument, "invalid f32 gemv");
  f32_gemv_kernel<<<rows, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(matrix, input, output, rows, columns);
  return checked(cudaPeekAtLastError(), "f32 gemv");
}
Status rms_norm(const float* input, const float* weight, float* output, std::uint32_t elements, float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !elements || epsilon <= 0) return Status(ErrorCode::invalid_argument, "invalid rms norm");
  rms_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "rms norm");
}
Status add_in_place(float* destination, const float* source, std::uint32_t elements, void* raw) noexcept {
  if (!destination || !source || !elements) return Status(ErrorCode::invalid_argument, "invalid add");
  add_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(destination, source, elements);
  return checked(cudaPeekAtLastError(), "add");
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
  router_select_kernel<<<1, 1, 0, stream>>>(logits, experts, top_k, scores, indices);
  return checked(cudaPeekAtLastError(), "router select");
}
Status argmax(const float* values, std::uint32_t count, std::uint32_t* output, void* raw) noexcept {
  if (!values || !count || !output) return Status(ErrorCode::invalid_argument, "invalid argmax");
  argmax_kernel<<<1, 1, 0, static_cast<cudaStream_t>(raw)>>>(values, count, output);
  return checked(cudaPeekAtLastError(), "argmax");
}

}  // namespace expert::runtime::cuda
