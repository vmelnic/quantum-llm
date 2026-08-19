#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

void status_check(const expert::runtime::Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
               "cudaMalloc smoke buffer");
  }
  ~DeviceBuffer() { static_cast<void>(cudaFree(pointer_)); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const noexcept { return pointer_; }
  void upload(const std::vector<T>& values) {
    if (values.size() != count_) throw std::runtime_error("upload size mismatch");
    cuda_check(cudaMemcpy(pointer_, values.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upload smoke buffer");
  }
  std::vector<T> download() const {
    std::vector<T> result(count_);
    cuda_check(cudaMemcpy(result.data(), pointer_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "download smoke buffer");
    return result;
  }
  void zero() {
    cuda_check(cudaMemset(pointer_, 0, count_ * sizeof(T)), "zero smoke buffer");
  }

 private:
  T* pointer_{};
  std::size_t count_{};
};

float pattern(std::size_t index, float scale = 0.08F) {
  return std::sin(static_cast<float>(index) * 0.173F) * scale;
}

float silu(float value) { return value / (1.0F + std::exp(-value)); }

double maximum_error(const std::vector<float>& actual,
                     const std::vector<float>& expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("comparison size mismatch");
  double result = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i)
    result = std::max(result, std::abs(static_cast<double>(actual[i]) - expected[i]));
  return result;
}

double check_rms() {
  constexpr std::uint32_t count = 2048;
  std::vector<float> input(count), weight(count), expected(count);
  double square = 0.0;
  for (std::uint32_t i = 0; i < count; ++i) {
    input[i] = pattern(i);
    weight[i] = pattern(i + 31, 0.02F);
    square += static_cast<double>(input[i]) * input[i];
  }
  const auto inverse = 1.0 / std::sqrt(square / count + 1.0e-6);
  for (std::uint32_t i = 0; i < count; ++i)
    expected[i] = static_cast<float>(input[i] * inverse * (1.0 + weight[i]));
  DeviceBuffer<float> d_input(count), d_weight(count), d_output(count);
  d_input.upload(input); d_weight.upload(weight);
  status_check(expert::runtime::cuda::qwen3_next_rms_norm(
      d_input.get(), d_weight.get(), d_output.get(), count, 1.0e-6F, nullptr));
  cuda_check(cudaDeviceSynchronize(), "rms synchronize");
  return maximum_error(d_output.download(), expected);
}

double check_batched_gemv() {
  constexpr std::uint32_t rows = 7, columns = 13, batch = 3;
  std::vector<std::int8_t> weights(rows * columns);
  std::vector<float> scales(rows), input(batch * columns), expected(batch * rows);
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = static_cast<std::int8_t>(static_cast<int>(i % 11) - 5);
  for (std::uint32_t row = 0; row < rows; ++row)
    scales[row] = 0.01F * static_cast<float>(row + 1U);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = pattern(i, 0.3F);
  for (std::uint32_t request = 0; request < batch; ++request) {
    for (std::uint32_t row = 0; row < rows; ++row) {
      float sum = 0.0F;
      for (std::uint32_t column = 0; column < columns; ++column)
        sum += static_cast<float>(weights[row * columns + column]) *
               input[request * columns + column];
      expected[request * rows + row] = sum * scales[row];
    }
  }
  DeviceBuffer<std::int8_t> d_weights(weights.size());
  DeviceBuffer<float> d_scales(scales.size()), d_input(input.size()),
      d_output(expected.size());
  d_weights.upload(weights); d_scales.upload(scales); d_input.upload(input);
  status_check(expert::runtime::cuda::gemv_batch(
      {d_weights.get(), d_scales.get(), rows, columns}, d_input.get(),
      d_output.get(), batch, nullptr));
  cuda_check(cudaDeviceSynchronize(), "batched gemv synchronize");
  return maximum_error(d_output.download(), expected);
}

double check_batched_router() {
  constexpr std::uint32_t rows = 3, hidden = 5, experts = 8, top_k = 3;
  std::vector<float> input(rows * hidden), weights(experts * hidden);
  for (std::size_t i = 0; i < input.size(); ++i) input[i] = pattern(i, 0.4F);
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = pattern(i + 19, 0.6F);
  DeviceBuffer<float> d_input(input.size()), d_weights(weights.size()),
      d_logits(rows * experts), d_scores(rows * top_k);
  DeviceBuffer<std::uint32_t> d_indices(rows * top_k);
  d_input.upload(input); d_weights.upload(weights);
  status_check(expert::runtime::cuda::router_topk_normalized_batch(
      d_input.get(), d_weights.get(), rows, hidden, experts, top_k,
      d_logits.get(), d_scores.get(), d_indices.get(), nullptr));
  cuda_check(cudaDeviceSynchronize(), "batched router synchronize");
  const auto scores = d_scores.download();
  const auto indices = d_indices.download();
  double error = 0.0;
  for (std::uint32_t request = 0; request < rows; ++request) {
    std::vector<float> logits(experts), probabilities(experts);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      for (std::uint32_t column = 0; column < hidden; ++column)
        logits[expert] += weights[expert * hidden + column] *
                          input[request * hidden + column];
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    float denominator = 0.0F;
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      denominator += probabilities[expert] = std::exp(logits[expert] - maximum);
    for (auto& value : probabilities) value /= denominator;
    std::vector<std::uint32_t> order(experts);
    for (std::uint32_t i = 0; i < experts; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                      [&](auto left, auto right) {
                        return probabilities[left] > probabilities[right];
                      });
    float selected_sum = 0.0F;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      selected_sum += probabilities[order[slot]];
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      const auto offset = request * top_k + slot;
      if (indices[offset] != order[slot]) return 1.0e9;
      error = std::max(error, std::abs(static_cast<double>(scores[offset]) -
          probabilities[order[slot]] / selected_sum));
    }
  }
  return error;
}

double check_sigmoid_bias_router() {
  constexpr std::uint32_t rows = 2, hidden = 5, experts = 8, top_k = 3;
  constexpr float epsilon = 1.0e-6F, route_scale = 1.25F;
  std::vector<float> input(rows * hidden), weights(experts * hidden),
      bias(experts);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = pattern(i + 211, 0.4F);
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = pattern(i + 307, 0.6F);
  for (std::uint32_t expert = 0; expert < experts; ++expert)
    bias[expert] = (expert == 7U ? 0.7F : -0.03F * expert);
  DeviceBuffer<float> d_input(input.size()), d_weights(weights.size()),
      d_bias(bias.size()), d_logits(rows * experts),
      d_scores(rows * top_k);
  DeviceBuffer<std::uint32_t> d_indices(rows * top_k);
  d_input.upload(input);
  d_weights.upload(weights);
  d_bias.upload(bias);
  status_check(expert::runtime::cuda::sigmoid_bias_router_topk_batch(
      d_input.get(), d_weights.get(), d_bias.get(), rows, hidden, experts,
      top_k, epsilon, route_scale, d_logits.get(), d_scores.get(),
      d_indices.get(), nullptr));
  cuda_check(cudaDeviceSynchronize(), "sigmoid-bias router synchronize");
  const auto scores = d_scores.download();
  const auto indices = d_indices.download();
  double error = 0.0;
  for (std::uint32_t row = 0; row < rows; ++row) {
    std::vector<float> unbiased(experts), selection(experts);
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      float logit = 0.0F;
      for (std::uint32_t column = 0; column < hidden; ++column)
        logit += weights[expert * hidden + column] *
                 input[row * hidden + column];
      unbiased[expert] = 1.0F / (1.0F + std::exp(-logit));
      selection[expert] = unbiased[expert] + bias[expert];
    }
    std::vector<std::uint32_t> order(experts);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      order[expert] = expert;
    std::partial_sort(
        order.begin(), order.begin() + top_k, order.end(),
        [&](auto left, auto right) {
          return selection[left] != selection[right]
                     ? selection[left] > selection[right]
                     : left < right;
        });
    float denominator = epsilon;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      denominator += unbiased[order[slot]];
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      const auto offset = row * top_k + slot;
      if (indices[offset] != order[slot]) return 1.0e9;
      const auto expected = unbiased[order[slot]] * route_scale / denominator;
      error = std::max(
          error, std::abs(static_cast<double>(scores[offset]) - expected));
    }
  }
  return error;
}

double check_causal_short_conv() {
  constexpr std::uint32_t rows = 2, hidden = 7, kernel = 3;
  std::vector<float> first(rows * 3U * hidden), second(first.size()),
      weights(hidden * kernel), expected_state(rows * hidden * kernel),
      expected_output(rows * hidden);
  for (std::size_t i = 0; i < first.size(); ++i) {
    first[i] = pattern(i + 401, 0.5F);
    second[i] = pattern(i + 503, 0.4F);
  }
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = pattern(i + 607, 0.3F);
  const auto update = [&](const std::vector<float>& projected) {
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto projected_base = static_cast<std::size_t>(row) * 3U * hidden;
      for (std::uint32_t channel = 0; channel < hidden; ++channel) {
        auto* state = expected_state.data() +
            (static_cast<std::size_t>(row) * hidden + channel) * kernel;
        for (std::uint32_t index = 1; index < kernel; ++index)
          state[index - 1U] = state[index];
        state[kernel - 1U] = projected[projected_base + channel] *
            projected[projected_base + 2U * hidden + channel];
        float convolution = 0.0F;
        for (std::uint32_t index = 0; index < kernel; ++index)
          convolution += state[index] * weights[channel * kernel + index];
        expected_output[row * hidden + channel] =
            projected[projected_base + hidden + channel] * convolution;
      }
    }
  };
  DeviceBuffer<float> d_projected(first.size()), d_weights(weights.size()),
      d_state(expected_state.size()), d_output(expected_output.size());
  d_weights.upload(weights);
  d_state.zero();
  d_projected.upload(first);
  status_check(expert::runtime::cuda::causal_short_conv_decode({
      d_projected.get(), d_weights.get(), d_state.get(), d_output.get(),
      rows, hidden, kernel, nullptr}));
  update(first);
  d_projected.upload(second);
  status_check(expert::runtime::cuda::causal_short_conv_decode({
      d_projected.get(), d_weights.get(), d_state.get(), d_output.get(),
      rows, hidden, kernel, nullptr}));
  update(second);
  cuda_check(cudaDeviceSynchronize(), "causal short-conv synchronize");
  return std::max(maximum_error(d_state.download(), expected_state),
                  maximum_error(d_output.download(), expected_output));
}

double check_standard_gqa_attention() {
  constexpr std::uint32_t query_heads = 4, kv_heads = 2, dim = 64;
  std::vector<float> query(query_heads * dim), key(kv_heads * dim),
      value(kv_heads * dim), q_weight(dim), k_weight(dim),
      expected_query(query.size()), expected_key(key.size()),
      expected_output(query.size());
  for (std::size_t i = 0; i < query.size(); ++i)
    query[i] = pattern(i + 701, 0.5F);
  for (std::size_t i = 0; i < key.size(); ++i) {
    key[i] = pattern(i + 809, 0.4F);
    value[i] = pattern(i + 907, 0.6F);
  }
  for (std::uint32_t dimension = 0; dimension < dim; ++dimension) {
    q_weight[dimension] = 1.0F + pattern(dimension + 1009, 0.03F);
    k_weight[dimension] = 1.0F + pattern(dimension + 1103, 0.03F);
  }
  for (std::uint32_t head = 0; head < query_heads; ++head) {
    double square = 0.0;
    for (std::uint32_t dimension = 0; dimension < dim; ++dimension) {
      const auto value_at = query[head * dim + dimension];
      square += static_cast<double>(value_at) * value_at;
    }
    const auto inverse = 1.0 / std::sqrt(square / dim + 1.0e-5);
    const auto kv_head = head / (query_heads / kv_heads);
    for (std::uint32_t dimension = 0; dimension < dim; ++dimension) {
      expected_query[head * dim + dimension] = static_cast<float>(
          query[head * dim + dimension] * inverse * q_weight[dimension]);
      expected_output[head * dim + dimension] =
          value[kv_head * dim + dimension];
    }
  }
  for (std::uint32_t head = 0; head < kv_heads; ++head) {
    double square = 0.0;
    for (std::uint32_t dimension = 0; dimension < dim; ++dimension) {
      const auto value_at = key[head * dim + dimension];
      square += static_cast<double>(value_at) * value_at;
    }
    const auto inverse = 1.0 / std::sqrt(square / dim + 1.0e-5);
    for (std::uint32_t dimension = 0; dimension < dim; ++dimension)
      expected_key[head * dim + dimension] = static_cast<float>(
          key[head * dim + dimension] * inverse * k_weight[dimension]);
  }
  DeviceBuffer<float> d_query(query.size()), d_key(key.size()),
      d_value(value.size()), d_q_weight(q_weight.size()),
      d_k_weight(k_weight.size()), d_key_cache(key.size()),
      d_value_cache(value.size()), d_output(expected_output.size());
  d_query.upload(query);
  d_key.upload(key);
  d_value.upload(value);
  d_q_weight.upload(q_weight);
  d_k_weight.upload(k_weight);
  status_check(expert::runtime::cuda::gqa_qkv_rope_cache(
      d_query.get(), d_key.get(), d_value.get(), d_q_weight.get(),
      d_k_weight.get(), d_key_cache.get(), d_value_cache.get(), 0U,
      query_heads, kv_heads, dim, dim, 1.0e-5F, 1'000'000.0F, nullptr));
  status_check(expert::runtime::cuda::gqa_attention_decode(
      d_query.get(), d_key_cache.get(), d_value_cache.get(), d_output.get(),
      1U, query_heads, kv_heads, dim, nullptr));
  cuda_check(cudaDeviceSynchronize(), "standard GQA synchronize");
  return std::max({maximum_error(d_query.download(), expected_query),
                   maximum_error(d_key.download(), expected_key),
                   maximum_error(d_output.download(), expected_output)});
}

double check_compact_moe_aggregation() {
  constexpr std::uint32_t rows = 2, hidden = 5, top_k = 3;
  constexpr std::uint32_t selections = rows * top_k;
  std::vector<float> primary(selections * hidden), routing{
      0.5F, 0.3F, 0.2F, 0.1F, 0.6F, 0.3F};
  for (std::size_t index = 0; index < primary.size(); ++index)
    primary[index] = pattern(index + 101, 0.7F);
  const std::vector<std::uint8_t> mask{1, 0, 1, 1, 0, 1};
  const std::vector<std::uint32_t> compact_slot{0, 0, 0, 0, 1, 0};
  std::vector<float> compact(2U * hidden);
  for (std::uint32_t column = 0; column < hidden; ++column) {
    compact[column] = primary[hidden + column];
    compact[hidden + column] = primary[4U * hidden + column];
  }
  std::vector<float> expected(rows * hidden);
  for (std::uint32_t row = 0; row < rows; ++row)
    for (std::uint32_t column = 0; column < hidden; ++column)
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        const auto selection = row * top_k + slot;
        expected[row * hidden + column] +=
            routing[selection] * primary[selection * hidden + column];
      }
  DeviceBuffer<float> d_primary(primary.size()), d_compact(compact.size()),
      d_routing(routing.size()), d_output(expected.size());
  DeviceBuffer<std::uint8_t> d_mask(mask.size());
  DeviceBuffer<std::uint32_t> d_compact_slot(compact_slot.size());
  d_primary.upload(primary);
  d_compact.upload(compact);
  d_routing.upload(routing);
  d_mask.upload(mask);
  d_compact_slot.upload(compact_slot);
  status_check(expert::runtime::cuda::launch_moe_aggregate({
      d_primary.get(), d_compact.get(), d_mask.get(), d_compact_slot.get(),
      d_routing.get(), d_output.get(), 2, rows, hidden, top_k, nullptr}));
  cuda_check(cudaDeviceSynchronize(), "compact aggregation synchronize");
  return maximum_error(d_output.download(), expected);
}

double check_full_attention() {
  constexpr std::uint32_t query_heads = 16, kv_heads = 2, dim = 256;
  constexpr std::uint32_t tokens = 3, page_tokens = 2;
  std::vector<float> q(static_cast<std::size_t>(query_heads) * 2U * dim);
  std::vector<float> k(kv_heads * dim), v(kv_heads * dim), norm(dim);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = pattern(i, 0.11F);
  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = pattern(i + 17, 0.09F);
    v[i] = pattern(i + 43, 0.13F);
  }
  for (std::size_t i = 0; i < norm.size(); ++i) norm[i] = pattern(i + 9, 0.01F);
  std::vector<float> expected(query_heads * dim);
  for (std::uint32_t head = 0; head < query_heads; ++head) {
    const auto kv = head / (query_heads / kv_heads);
    for (std::uint32_t d = 0; d < dim; ++d) {
      const auto gate = q[static_cast<std::size_t>(head) * 2U * dim + dim + d];
      expected[static_cast<std::size_t>(head) * dim + d] =
          v[static_cast<std::size_t>(kv) * dim + d] /
          (1.0F + std::exp(-gate));
    }
  }
  DeviceBuffer<float> d_q(q.size()), d_k(k.size()), d_v(v.size()), d_norm(norm.size());
  DeviceBuffer<float> d_k_cache(tokens * k.size()), d_v_cache(tokens * v.size()),
      d_output(expected.size()), d_paged_output(expected.size());
  d_q.upload(q); d_k.upload(k); d_v.upload(v); d_norm.upload(norm);
  status_check(expert::runtime::cuda::qwen3_next_qkv_rope_cache(
      d_q.get(), d_k.get(), d_v.get(), d_norm.get(), d_norm.get(),
      d_k_cache.get(), d_v_cache.get(), 0, query_heads, kv_heads, dim, 64,
      1.0e-6F, 10000000.0F, nullptr));
  status_check(expert::runtime::cuda::qwen3_next_attention_decode(
      d_q.get(), d_k_cache.get(), d_v_cache.get(), d_output.get(), 1,
      query_heads, kv_heads, dim, nullptr));
  cuda_check(cudaDeviceSynchronize(), "attention synchronize");
  const auto one_token_error = maximum_error(d_output.download(), expected);

  const auto page_elements = static_cast<std::size_t>(page_tokens) * kv_heads * dim;
  DeviceBuffer<std::uint16_t> d_page0(2U * page_elements),
      d_page1(2U * page_elements);
  DeviceBuffer<void*> d_page_table(2);
  d_page_table.upload({d_page0.get(), d_page1.get()});
  for (std::uint32_t position = 0; position < tokens; ++position) {
    d_q.upload(q); d_k.upload(k); d_v.upload(v);
    status_check(expert::runtime::cuda::qwen3_next_qkv_rope_cache(
        d_q.get(), d_k.get(), d_v.get(), d_norm.get(), d_norm.get(),
        d_k_cache.get(), d_v_cache.get(), position, query_heads, kv_heads, dim,
        64, 1.0e-6F, 10000000.0F, nullptr));
    d_q.upload(q); d_k.upload(k); d_v.upload(v);
    status_check(
        expert::runtime::cuda::qwen3_next_qkv_rope_cache_paged_fp16(
            d_q.get(), d_k.get(), d_v.get(), d_norm.get(), d_norm.get(),
            position < page_tokens ? static_cast<void*>(d_page0.get())
                                   : static_cast<void*>(d_page1.get()),
            0, page_tokens, position, query_heads, kv_heads, dim, 64,
            1.0e-6F, 10000000.0F, nullptr));
  }
  status_check(expert::runtime::cuda::qwen3_next_attention_decode_paged_fp16(
      d_q.get(), reinterpret_cast<const void* const*>(d_page_table.get()),
      d_paged_output.get(), tokens, 0, page_tokens, query_heads, kv_heads, dim,
      nullptr));
  d_q.upload(q); d_k.upload(k); d_v.upload(v);
  status_check(expert::runtime::cuda::qwen3_next_qkv_rope_cache(
      d_q.get(), d_k.get(), d_v.get(), d_norm.get(), d_norm.get(),
      d_k_cache.get(), d_v_cache.get(), tokens - 1U, query_heads, kv_heads, dim,
      64, 1.0e-6F, 10000000.0F, nullptr));
  status_check(expert::runtime::cuda::qwen3_next_attention_decode(
      d_q.get(), d_k_cache.get(), d_v_cache.get(), d_output.get(), tokens,
      query_heads, kv_heads, dim, nullptr));
  cuda_check(cudaDeviceSynchronize(), "paged attention synchronize");
  return std::max(one_token_error,
                  maximum_error(d_paged_output.download(), d_output.download()));
}

struct Fp4AttentionErrors final {
  double kernel{};
  double quantization{};
};

float decode_fp4_value(const std::uint8_t* record,
                       std::uint32_t dimension,
                       std::uint32_t head_dim) {
  constexpr float values[8]{0.0F, 0.5F, 1.0F, 1.5F,
                            2.0F, 3.0F, 4.0F, 6.0F};
  const auto packed = record[dimension / 2U];
  const auto code = static_cast<std::uint8_t>(
      (dimension & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
  const auto scale = std::ldexp(
      1.0F,
      static_cast<int>(record[head_dim / 2U + dimension / 32U]) - 127);
  const auto magnitude = values[code & 0x07U] * scale;
  return (code & 0x08U) != 0U ? -magnitude : magnitude;
}

Fp4AttentionErrors check_paged_fp4_attention() {
  constexpr std::uint32_t query_heads = 6U, kv_heads = 1U, dim = 256U;
  constexpr std::uint32_t tokens = 5U, page_tokens = 2U;
  constexpr std::uint32_t full_layers = 2U, full_layer = 1U;
  constexpr std::uint32_t split_tokens = 2U, maximum_splits = 3U;
  constexpr std::uint32_t record_bytes = dim / 2U + dim / 32U;
  constexpr std::size_t records_per_kind = page_tokens * kv_heads;
  constexpr std::size_t page_bytes =
      full_layers * 2U * records_per_kind * record_bytes;

  std::vector<float> norm(dim);
  for (std::size_t index = 0U; index < norm.size(); ++index)
    norm[index] = pattern(index + 2201U, 0.02F);
  DeviceBuffer<float> d_q(static_cast<std::size_t>(query_heads) * 2U * dim),
      d_k(static_cast<std::size_t>(kv_heads) * dim),
      d_v(static_cast<std::size_t>(kv_heads) * dim), d_norm(norm.size()),
      d_output(static_cast<std::size_t>(query_heads) * dim),
      d_partial_maxima(maximum_splits * query_heads),
      d_partial_sums(maximum_splits * query_heads),
      d_partial_outputs(static_cast<std::size_t>(maximum_splits) *
                        query_heads * dim);
  DeviceBuffer<std::uint8_t> d_page0(page_bytes), d_page1(page_bytes),
      d_page2(page_bytes);
  DeviceBuffer<void*> d_page_table(3U);
  d_norm.upload(norm);
  d_page0.zero();
  d_page1.zero();
  d_page2.zero();
  d_page_table.upload({d_page0.get(), d_page1.get(), d_page2.get()});

  std::vector<std::vector<float>> fp32_keys;
  std::vector<std::vector<float>> fp32_values;
  std::vector<float> final_query;
  for (std::uint32_t position = 0U; position < tokens; ++position) {
    std::vector<float> query(
        static_cast<std::size_t>(query_heads) * 2U * dim);
    std::vector<float> key(static_cast<std::size_t>(kv_heads) * dim);
    std::vector<float> value(static_cast<std::size_t>(kv_heads) * dim);
    for (std::size_t index = 0U; index < query.size(); ++index)
      query[index] = pattern(index + 2309U + 97U * position, 0.21F);
    for (std::size_t index = 0U; index < key.size(); ++index) {
      key[index] = pattern(index + 2609U + 131U * position, 0.27F);
      value[index] = pattern(index + 2903U + 173U * position, 0.43F);
    }
    d_q.upload(query);
    d_k.upload(key);
    d_v.upload(value);
    void* page = position < 2U
                     ? static_cast<void*>(d_page0.get())
                     : position < 4U ? static_cast<void*>(d_page1.get())
                                     : static_cast<void*>(d_page2.get());
    status_check(
        expert::runtime::cuda::gated_gqa_qkv_rope_cache_paged_fp4(
            d_q.get(), d_k.get(), d_v.get(), d_norm.get(), d_norm.get(), page,
            full_layer, page_tokens, position, query_heads, kv_heads, dim, 64U,
            1.0e-6F, 10000000.0F, nullptr));
    fp32_keys.push_back(d_k.download());
    fp32_values.push_back(std::move(value));
    if (position + 1U == tokens) final_query = d_q.download();
  }
  status_check(expert::runtime::cuda::gated_gqa_attention_decode_paged_fp4({
      d_q.get(), reinterpret_cast<const void* const*>(d_page_table.get()),
      d_output.get(), d_partial_maxima.get(), d_partial_sums.get(),
      d_partial_outputs.get(), tokens, full_layer, page_tokens, query_heads,
      kv_heads, dim, split_tokens, maximum_splits, nullptr}));
  cuda_check(cudaDeviceSynchronize(), "paged FP4 attention synchronize");

  const std::vector<std::vector<std::uint8_t>> pages{
      d_page0.download(), d_page1.download(), d_page2.download()};
  std::vector<std::vector<float>> decoded_keys(
      tokens, std::vector<float>(dim));
  std::vector<std::vector<float>> decoded_values(
      tokens, std::vector<float>(dim));
  for (std::uint32_t token = 0U; token < tokens; ++token) {
    const auto& page = pages[token / page_tokens];
    const auto layer_base = static_cast<std::size_t>(full_layer) * 2U *
                            records_per_kind * record_bytes;
    const auto record = static_cast<std::size_t>(token % page_tokens) *
                        kv_heads;
    const auto* key_record =
        page.data() + layer_base + record * record_bytes;
    const auto* value_record =
        page.data() + layer_base + records_per_kind * record_bytes +
        record * record_bytes;
    for (std::uint32_t dimension = 0U; dimension < dim; ++dimension) {
      decoded_keys[token][dimension] =
          decode_fp4_value(key_record, dimension, dim);
      decoded_values[token][dimension] =
          decode_fp4_value(value_record, dimension, dim);
    }
  }

  const auto reference = [&](const std::vector<std::vector<float>>& keys,
                             const std::vector<std::vector<float>>& values) {
    std::vector<float> output(static_cast<std::size_t>(query_heads) * dim);
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto* query = final_query.data() +
                          static_cast<std::size_t>(head) * 2U * dim;
      std::vector<double> scores(tokens);
      double maximum = -1.0e300;
      for (std::uint32_t token = 0U; token < tokens; ++token) {
        double dot = 0.0;
        for (std::uint32_t dimension = 0U; dimension < dim; ++dimension)
          dot += static_cast<double>(query[dimension]) *
                 keys[token][dimension];
        scores[token] = dot / std::sqrt(static_cast<double>(dim));
        maximum = std::max(maximum, scores[token]);
      }
      double denominator = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (std::uint32_t dimension = 0U; dimension < dim; ++dimension) {
        double sum = 0.0;
        for (std::uint32_t token = 0U; token < tokens; ++token)
          sum += scores[token] * values[token][dimension];
        const auto gate = 1.0 / (1.0 + std::exp(-query[dim + dimension]));
        output[static_cast<std::size_t>(head) * dim + dimension] =
            static_cast<float>((sum / denominator) * gate);
      }
    }
    return output;
  };
  const auto decoded_reference = reference(decoded_keys, decoded_values);
  const auto fp32_reference = reference(fp32_keys, fp32_values);
  const auto actual = d_output.download();
  return {maximum_error(actual, decoded_reference),
          maximum_error(actual, fp32_reference)};
}

double check_delta() {
  constexpr std::uint32_t key_heads = 16, value_heads = 32;
  constexpr std::uint32_t key_dim = 128, value_dim = 128, kernel = 4;
  constexpr std::uint32_t key_total = key_heads * key_dim;
  constexpr std::uint32_t value_total = value_heads * value_dim;
  constexpr std::uint32_t conv_total = 2U * key_total + value_total;
  constexpr std::uint32_t projected_total = 2U * key_total + 2U * value_total;
  std::vector<float> projected(projected_total), ba(2U * value_heads),
      conv_weights(static_cast<std::size_t>(conv_total) * kernel),
      dt(value_heads), a_log(value_heads), norm(value_dim);
  for (std::size_t i = 0; i < projected.size(); ++i) projected[i] = pattern(i, 0.07F);
  for (std::size_t i = 0; i < ba.size(); ++i) ba[i] = pattern(i + 71, 0.09F);
  for (std::size_t i = 0; i < conv_weights.size(); ++i)
    conv_weights[i] = pattern(i + 103, 0.04F);
  for (std::uint32_t i = 0; i < value_heads; ++i) {
    dt[i] = pattern(i + 5, 0.03F);
    a_log[i] = -1.5F + 0.01F * static_cast<float>(i);
  }
  for (std::uint32_t i = 0; i < value_dim; ++i) norm[i] = 1.0F + pattern(i, 0.02F);

  const auto ratio = value_heads / key_heads;
  const auto group_width = 2U * key_dim + 2U * ratio * value_dim;
  std::vector<float> conv(conv_total);
  for (std::uint32_t channel = 0; channel < conv_total; ++channel) {
    std::uint32_t source = 0;
    if (channel < key_total) {
      const auto group = channel / key_dim;
      source = group * group_width + channel % key_dim;
    } else if (channel < 2U * key_total) {
      const auto local = channel - key_total;
      const auto group = local / key_dim;
      source = group * group_width + key_dim + local % key_dim;
    } else {
      const auto local = channel - 2U * key_total;
      const auto head = local / value_dim;
      source = (head / ratio) * group_width + 2U * key_dim +
               (head % ratio) * value_dim + local % value_dim;
    }
    conv[channel] = silu(projected[source] *
                         conv_weights[static_cast<std::size_t>(channel) * kernel + kernel - 1]);
  }
  std::vector<float> expected(value_total);
  for (std::uint32_t head = 0; head < value_heads; ++head) {
    const auto key_head = head / ratio;
    double q_square = 1.0e-6, k_square = 1.0e-6;
    for (std::uint32_t i = 0; i < key_dim; ++i) {
      const auto qv = conv[static_cast<std::size_t>(key_head) * key_dim + i];
      const auto kv = conv[key_total + static_cast<std::size_t>(key_head) * key_dim + i];
      q_square += static_cast<double>(qv) * qv;
      k_square += static_cast<double>(kv) * kv;
    }
    const auto q_inverse = 1.0 / std::sqrt(q_square * key_dim);
    const auto k_inverse = 1.0 / std::sqrt(k_square);
    double qk = 0.0;
    for (std::uint32_t i = 0; i < key_dim; ++i)
      qk += conv[static_cast<std::size_t>(key_head) * key_dim + i] * q_inverse *
            conv[key_total + static_cast<std::size_t>(key_head) * key_dim + i] * k_inverse;
    const auto group = head / ratio, replica = head % ratio;
    const auto beta = 1.0F /
        (1.0F + std::exp(-ba[group * 2U * ratio + replica]));
    std::vector<float> core(value_dim);
    double square = 0.0;
    for (std::uint32_t d = 0; d < value_dim; ++d) {
      core[d] = static_cast<float>(qk) * beta *
                conv[2U * key_total + static_cast<std::size_t>(head) * value_dim + d];
      square += static_cast<double>(core[d]) * core[d];
    }
    const auto inverse = 1.0 / std::sqrt(square / value_dim + 1.0e-6);
    for (std::uint32_t d = 0; d < value_dim; ++d) {
      const auto z = projected[group * group_width + 2U * key_dim +
                               ratio * value_dim + replica * value_dim + d];
      expected[static_cast<std::size_t>(head) * value_dim + d] =
          static_cast<float>(core[d] * inverse * norm[d] * silu(z));
    }
  }

  DeviceBuffer<float> d_projected(projected.size()), d_ba(ba.size()),
      d_weights(conv_weights.size()), d_dt(dt.size()), d_a(a_log.size()),
      d_norm(norm.size()), d_conv_state(static_cast<std::size_t>(conv_total) * kernel),
      d_recurrent(static_cast<std::size_t>(value_heads) * key_dim * value_dim),
      d_conv(conv_total), d_output(value_total);
  d_projected.upload(projected); d_ba.upload(ba); d_weights.upload(conv_weights);
  d_dt.upload(dt); d_a.upload(a_log); d_norm.upload(norm);
  d_conv_state.zero(); d_recurrent.zero();
  status_check(expert::runtime::cuda::qwen3_next_delta_decode({
      d_projected.get(), d_ba.get(), d_weights.get(), d_dt.get(), d_a.get(),
      d_norm.get(), d_conv_state.get(), d_recurrent.get(), d_conv.get(),
      d_output.get(), key_heads, value_heads, key_dim, value_dim, kernel,
      1.0e-6F, nullptr}));
  cuda_check(cudaDeviceSynchronize(), "delta synchronize");
  return maximum_error(d_output.download(), expected);
}

double check_split_delta() {
  constexpr std::uint32_t key_heads = 2, value_heads = 4;
  constexpr std::uint32_t key_dim = 16, value_dim = 16, kernel = 4;
  constexpr std::uint32_t key_total = key_heads * key_dim;
  constexpr std::uint32_t value_total = value_heads * value_dim;
  constexpr std::uint32_t conv_total = 2U * key_total + value_total;
  std::vector<float> first_qkv(conv_total), second_qkv(conv_total),
      first_z(value_total), second_z(value_total), first_b(value_heads),
      second_b(value_heads), first_a(value_heads), second_a(value_heads),
      conv_weights(static_cast<std::size_t>(conv_total) * kernel),
      dt(value_heads), a_log(value_heads), norm(value_dim);
  for (std::size_t index = 0; index < first_qkv.size(); ++index) {
    first_qkv[index] = pattern(index + 1201, 0.09F);
    second_qkv[index] = pattern(index + 1301, 0.08F);
  }
  for (std::size_t index = 0; index < first_z.size(); ++index) {
    first_z[index] = pattern(index + 1409, 0.11F);
    second_z[index] = pattern(index + 1511, 0.10F);
  }
  for (std::uint32_t head = 0; head < value_heads; ++head) {
    first_b[head] = pattern(head + 1601, 0.2F);
    second_b[head] = pattern(head + 1613, 0.2F);
    first_a[head] = pattern(head + 1709, 0.3F);
    second_a[head] = pattern(head + 1723, 0.3F);
    dt[head] = pattern(head + 1801, 0.1F);
    a_log[head] = -1.2F + 0.07F * static_cast<float>(head);
  }
  for (std::size_t index = 0; index < conv_weights.size(); ++index)
    conv_weights[index] = pattern(index + 1901, 0.05F);
  for (std::uint32_t dimension = 0; dimension < value_dim; ++dimension)
    norm[dimension] = 1.0F + pattern(dimension + 2003, 0.03F);

  std::vector<float> expected_conv_state(
      static_cast<std::size_t>(conv_total) * kernel);
  std::vector<float> expected_recurrent(
      static_cast<std::size_t>(value_heads) * key_dim * value_dim);
  std::vector<float> expected_output(value_total);
  const auto reference_step = [&](const std::vector<float>& qkv,
                                  const std::vector<float>& z,
                                  const std::vector<float>& b,
                                  const std::vector<float>& a) {
    std::vector<float> conv(conv_total);
    for (std::uint32_t channel = 0; channel < conv_total; ++channel) {
      auto* state = expected_conv_state.data() +
          static_cast<std::size_t>(channel) * kernel;
      for (std::uint32_t index = 1U; index < kernel; ++index)
        state[index - 1U] = state[index];
      state[kernel - 1U] = qkv[channel];
      float sum = 0.0F;
      for (std::uint32_t index = 0U; index < kernel; ++index)
        sum += state[index] *
               conv_weights[static_cast<std::size_t>(channel) * kernel +
                            index];
      conv[channel] = silu(sum);
    }
    const auto ratio = value_heads / key_heads;
    for (std::uint32_t head = 0; head < value_heads; ++head) {
      const auto key_head = head / ratio;
      std::vector<float> query(key_dim), key(key_dim);
      double q_square = 1.0e-6;
      double k_square = 1.0e-6;
      for (std::uint32_t dimension = 0; dimension < key_dim; ++dimension) {
        query[dimension] =
            conv[static_cast<std::size_t>(key_head) * key_dim + dimension];
        key[dimension] =
            conv[key_total +
                 static_cast<std::size_t>(key_head) * key_dim + dimension];
        q_square += static_cast<double>(query[dimension]) * query[dimension];
        k_square += static_cast<double>(key[dimension]) * key[dimension];
      }
      for (std::uint32_t dimension = 0; dimension < key_dim; ++dimension) {
        query[dimension] = static_cast<float>(
            query[dimension] / std::sqrt(q_square * key_dim));
        key[dimension] =
            static_cast<float>(key[dimension] / std::sqrt(k_square));
      }
      const auto beta = 1.0F / (1.0F + std::exp(-b[head]));
      const auto softplus =
          std::log1p(std::exp(-std::abs(a[head] + dt[head]))) +
          std::max(a[head] + dt[head], 0.0F);
      const auto decay = std::exp(-std::exp(a_log[head]) * softplus);
      std::vector<float> core(value_dim);
      double square = 0.0;
      for (std::uint32_t dimension = 0; dimension < value_dim; ++dimension) {
        auto* column = expected_recurrent.data() +
            static_cast<std::size_t>(head) * key_dim * value_dim + dimension;
        float memory = 0.0F;
        for (std::uint32_t index = 0; index < key_dim; ++index) {
          auto& state = column[static_cast<std::size_t>(index) * value_dim];
          state *= decay;
          memory += state * key[index];
        }
        const auto value =
            conv[2U * key_total +
                 static_cast<std::size_t>(head) * value_dim + dimension];
        const auto delta = (value - memory) * beta;
        for (std::uint32_t index = 0; index < key_dim; ++index) {
          auto& state = column[static_cast<std::size_t>(index) * value_dim];
          state += key[index] * delta;
          core[dimension] += state * query[index];
        }
        square += static_cast<double>(core[dimension]) * core[dimension];
      }
      const auto inverse = 1.0 / std::sqrt(square / value_dim + 1.0e-6);
      for (std::uint32_t dimension = 0; dimension < value_dim; ++dimension)
        expected_output[static_cast<std::size_t>(head) * value_dim +
                        dimension] =
            static_cast<float>(core[dimension] * inverse * norm[dimension] *
                               silu(z[static_cast<std::size_t>(head) *
                                          value_dim +
                                      dimension]));
    }
  };
  reference_step(first_qkv, first_z, first_b, first_a);
  reference_step(second_qkv, second_z, second_b, second_a);

  DeviceBuffer<float> d_qkv(conv_total), d_z(value_total), d_b(value_heads),
      d_a(value_heads), d_weights(conv_weights.size()), d_dt(dt.size()),
      d_a_log(a_log.size()), d_norm(norm.size()),
      d_conv_state(expected_conv_state.size()),
      d_recurrent(expected_recurrent.size()), d_conv(conv_total),
      d_output(value_total);
  d_weights.upload(conv_weights);
  d_dt.upload(dt);
  d_a_log.upload(a_log);
  d_norm.upload(norm);
  d_conv_state.zero();
  d_recurrent.zero();
  const auto run = [&](const std::vector<float>& qkv,
                       const std::vector<float>& z,
                       const std::vector<float>& b,
                       const std::vector<float>& a) {
    d_qkv.upload(qkv);
    d_z.upload(z);
    d_b.upload(b);
    d_a.upload(a);
    status_check(expert::runtime::cuda::split_gated_delta_decode({
        d_qkv.get(), d_z.get(), d_b.get(), d_a.get(), d_weights.get(),
        d_dt.get(), d_a_log.get(), d_norm.get(), d_conv_state.get(),
        d_recurrent.get(), d_conv.get(), d_output.get(), key_heads,
        value_heads, key_dim, value_dim, kernel, 1.0e-6F, nullptr}));
  };
  run(first_qkv, first_z, first_b, first_a);
  run(second_qkv, second_z, second_b, second_a);
  cuda_check(cudaDeviceSynchronize(), "split delta synchronize");
  return std::max({maximum_error(d_output.download(), expected_output),
                   maximum_error(d_conv_state.download(),
                                 expected_conv_state),
                   maximum_error(d_recurrent.download(),
                                 expected_recurrent)});
}

}  // namespace

int main() {
  try {
    const auto rms = check_rms();
    const auto batched_gemv = check_batched_gemv();
    const auto batched_router = check_batched_router();
    const auto sigmoid_bias_router = check_sigmoid_bias_router();
    const auto short_conv = check_causal_short_conv();
    const auto standard_gqa = check_standard_gqa_attention();
    const auto compact_aggregate = check_compact_moe_aggregation();
    const auto attention = check_full_attention();
    const auto fp4_attention = check_paged_fp4_attention();
    const auto delta = check_delta();
    const auto split_delta = check_split_delta();
    const bool valid = rms < 2.0e-6 && batched_gemv < 2.0e-6 &&
                       batched_router < 2.0e-6 &&
                       sigmoid_bias_router < 2.0e-6 &&
                       short_conv < 2.0e-6 && standard_gqa < 2.0e-5 &&
                       compact_aggregate < 2.0e-6 && attention < 2.0e-4 &&
                       // The official execution dtype is BF16. This bound
                       // isolates kernel arithmetic from the separately
                       // reported FP4 KV quantization error.
                       fp4_attention.kernel < 3.0e-4 &&
                       fp4_attention.quantization < 4.0e-2 &&
                       delta < 2.0e-5 && split_delta < 2.0e-5;
    std::cout << "{\"valid\":" << (valid ? "true" : "false")
              << ",\"qwen_rms_max_abs\":" << rms
              << ",\"batched_gemv_max_abs\":" << batched_gemv
              << ",\"batched_router_max_abs\":" << batched_router
              << ",\"sigmoid_bias_router_max_abs\":"
              << sigmoid_bias_router
              << ",\"short_conv_max_abs\":" << short_conv
              << ",\"standard_gqa_max_abs\":" << standard_gqa
              << ",\"compact_aggregate_max_abs\":" << compact_aggregate
              << ",\"full_attention_max_abs\":" << attention
              << ",\"fp4_attention_kernel_max_abs\":"
              << fp4_attention.kernel
              << ",\"fp4_attention_quantization_max_abs\":"
              << fp4_attention.quantization
              << ",\"delta_max_abs\":" << delta
              << ",\"split_delta_max_abs\":" << split_delta << "}\n";
    return valid ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3-Next CUDA smoke: " << error.what() << '\n';
    return 1;
  }
}
