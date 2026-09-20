#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          count * sizeof(T)),
               "allocate dense FP4 smoke buffer");
  }
  ~DeviceBuffer() { static_cast<void>(cudaFree(pointer_)); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] T* get() const noexcept { return pointer_; }
  void upload(const std::vector<T>& values) {
    if (values.size() != count_)
      throw std::runtime_error("dense FP4 smoke upload size mismatch");
    cuda_check(cudaMemcpy(pointer_, values.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upload dense FP4 smoke buffer");
  }
  [[nodiscard]] std::vector<T> download() const {
    std::vector<T> result(count_);
    cuda_check(cudaMemcpy(result.data(), pointer_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "download dense FP4 smoke buffer");
    return result;
  }

 private:
  T* pointer_{};
  std::size_t count_{};
};

template <typename T>
class PinnedBuffer final {
 public:
  explicit PinnedBuffer(std::size_t count) : count_(count) {
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&pointer_),
                             count * sizeof(T), cudaHostAllocPortable),
               "allocate pinned dense FP4 smoke buffer");
  }
  ~PinnedBuffer() { static_cast<void>(cudaFreeHost(pointer_)); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  [[nodiscard]] T* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T* pointer_{};
  std::size_t count_{};
};

float decode_fp4(std::uint8_t code) {
  constexpr float levels[]{0.0F, 0.5F, 1.0F, 1.5F,
                           2.0F, 3.0F, 4.0F, 6.0F};
  const auto value = levels[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float scale(std::uint8_t code) {
  return std::ldexp(1.0F, static_cast<int>(code) - 127);
}

std::uint16_t binary16_bits(float value) {
  const auto encoded = __float2half_rn(value);
  std::uint16_t bits{};
  std::memcpy(&bits, &encoded, sizeof(bits));
  return bits;
}

float binary16_value(std::uint16_t bits) {
  __half encoded{};
  std::memcpy(&encoded, &bits, sizeof(bits));
  return __half2float(encoded);
}

float rounded_bf16(float value) {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) != 0x7f800000U)
    bits += 0x7fffU + ((bits >> 16U) & 1U);
  bits &= 0xffff0000U;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint8_t host_fp4_scale_code(float maximum) {
  if (maximum == 0.0F) return 127U;
  const auto exponent = static_cast<int>(
      std::ceil(std::log2(maximum / 6.0F)));
  return static_cast<std::uint8_t>(
      std::max(1, std::min(254, exponent + 127)));
}

std::uint8_t host_encode_fp4(float value, float block_scale) {
  const auto magnitude = std::abs(value / block_scale);
  std::uint8_t code{};
  if (magnitude <= 0.25F) code = 0U;
  else if (magnitude < 0.75F) code = 1U;
  else if (magnitude <= 1.25F) code = 2U;
  else if (magnitude < 1.75F) code = 3U;
  else if (magnitude <= 2.5F) code = 4U;
  else if (magnitude < 3.5F) code = 5U;
  else if (magnitude <= 5.0F) code = 6U;
  else code = 7U;
  return static_cast<std::uint8_t>(code | (value < 0.0F ? 8U : 0U));
}

double numerical_check() {
  constexpr std::uint32_t rows = 67U;
  constexpr std::uint32_t columns = 100U;
  constexpr std::uint32_t padded = 128U;
  constexpr std::uint32_t batch = 129U;
  std::vector<std::uint8_t> packed(
      static_cast<std::size_t>(rows) * padded / 2U);
  std::vector<std::uint8_t> scales(
      static_cast<std::size_t>(rows) * padded / 32U);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t block = 0; block < padded / 32U; ++block)
      scales[static_cast<std::size_t>(row) * (padded / 32U) + block] =
          static_cast<std::uint8_t>(124U + (row + block) % 7U);
    for (std::uint32_t pair = 0; pair < padded / 2U; ++pair) {
      const auto low = static_cast<std::uint8_t>((row + 3U * pair) % 16U);
      const auto high =
          static_cast<std::uint8_t>((5U * row + pair + 1U) % 16U);
      packed[static_cast<std::size_t>(row) * (padded / 2U) + pair] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  std::vector<float> input(static_cast<std::size_t>(batch) * columns);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = std::sin(static_cast<float>(index) * 0.071F) * 0.7F;

  std::vector<float> expected(static_cast<std::size_t>(batch) * rows);
  for (std::uint32_t request = 0; request < batch; ++request) {
    float maximum = 0.0F;
    for (std::uint32_t column = 0; column < columns; ++column)
      maximum = std::max(
          maximum,
          std::abs(input[static_cast<std::size_t>(request) * columns + column]));
    const auto input_scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
    std::vector<std::int8_t> q8(padded);
    for (std::uint32_t column = 0; column < columns; ++column) {
      auto value = static_cast<int>(std::nearbyint(
          input[static_cast<std::size_t>(request) * columns + column] /
          input_scale));
      value = std::max(-127, std::min(127, value));
      q8[column] = static_cast<std::int8_t>(value);
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
      float sum = 0.0F;
      for (std::uint32_t column = 0; column < padded; ++column) {
        const auto byte = packed[static_cast<std::size_t>(row) *
                                     (padded / 2U) +
                                 column / 2U];
        const auto code = static_cast<std::uint8_t>(
            (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
        const auto scale_code = scales[static_cast<std::size_t>(row) *
                                                (padded / 32U) +
                                            column / 32U];
        sum += static_cast<float>(q8[column]) * decode_fp4(code) *
               scale(scale_code) * input_scale;
      }
      expected[static_cast<std::size_t>(request) * rows + row] = sum;
    }
  }

  DeviceBuffer<std::uint8_t> device_weights(packed.size());
  DeviceBuffer<std::uint8_t> device_scales(scales.size());
  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::int8_t> device_q8(
      static_cast<std::size_t>(batch) * padded);
  DeviceBuffer<float> device_q8_scales(batch);
  DeviceBuffer<float> device_output(expected.size());
  DeviceBuffer<std::uint16_t> decoded_weights(
      static_cast<std::size_t>(rows) * padded);
  DeviceBuffer<std::uint16_t> decoded_input(
      static_cast<std::size_t>(batch) * padded);
  device_weights.upload(packed);
  device_scales.upload(scales);
  device_input.upload(input);
  status_check(expert::runtime::cuda::quantize_q8_batch(
      device_input.get(), device_q8.get(), device_q8_scales.get(), batch,
      columns, padded, nullptr));
  status_check(expert::runtime::cuda::fp4_gemv_q8_batch(
      {device_weights.get(), device_scales.get(), rows, columns, padded},
      device_q8.get(), device_q8_scales.get(), device_output.get(), batch,
      nullptr));
  cuda_check(cudaDeviceSynchronize(), "synchronize dense FP4 numerical smoke");
  const auto actual = device_output.download();
  double maximum_error = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(actual[index]) - expected[index]));
  status_check(expert::runtime::cuda::fp4_gemv_q8_batch_weight_reuse(
      {device_weights.get(), device_scales.get(), rows, columns, padded},
      device_q8.get(), device_q8_scales.get(), device_output.get(), 8U,
      nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize dense FP4 weight-reuse smoke");
  const auto reused = device_output.download();
  for (std::size_t index = 0; index < reused.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(reused[index]) - expected[index]));
  status_check(expert::runtime::cuda::fp4_gemv_q8_batch_weight_reuse(
      {device_weights.get(), device_scales.get(), rows, columns, padded},
      device_q8.get(), device_q8_scales.get(), device_output.get(), 5U,
      nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize dense FP4 batch-5 tensor-core smoke");
  const auto batch_five = device_output.download();
  for (std::size_t index = 0; index < 5U * rows; ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(batch_five[index]) - expected[index]));
  status_check(expert::runtime::cuda::fp4_gemv_q8_batch_weight_reuse(
      {device_weights.get(), device_scales.get(), rows, columns, padded},
      device_q8.get(), device_q8_scales.get(), device_output.get(), 2U,
      nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize dense FP4 batch-2 tensor-core smoke");
  const auto batch_two = device_output.download();
  for (std::size_t index = 0; index < 2U * rows; ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(batch_two[index]) - expected[index]));
  status_check(expert::runtime::cuda::fp4_gemm_q8_block32(
      {device_weights.get(), device_scales.get(), rows, columns, padded},
      device_q8.get(), device_q8_scales.get(), device_output.get(), batch,
      nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize dense FP4 tensor-core smoke");
  const auto tiled = device_output.download();
  for (std::size_t index = 0; index < tiled.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(tiled[index]) - expected[index]));
  const expert::runtime::cuda::Fp4Block32Matrix matrix{
      device_weights.get(), device_scales.get(), rows, columns, padded};
  status_check(expert::runtime::cuda::fp4_decode_matrix_bf16(
      matrix, decoded_weights.get(),
      static_cast<std::size_t>(rows) * padded * sizeof(std::uint16_t),
      nullptr));
  status_check(expert::runtime::cuda::bf16_gemm_q8_block32(
      matrix, decoded_weights.get(),
      static_cast<std::size_t>(rows) * padded * sizeof(std::uint16_t),
      device_q8.get(), device_q8_scales.get(), decoded_input.get(),
      static_cast<std::size_t>(batch) * padded * sizeof(std::uint16_t),
      device_output.get(), batch, nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize staged BF16 Q8 smoke");
  const auto staged = device_output.download();
  for (std::size_t index = 0; index < staged.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(staged[index]) - expected[index]));

  DeviceBuffer<float> embedding(columns);
  status_check(expert::runtime::cuda::fp4_embedding(
      {device_weights.get(), device_scales.get(), rows, columns, padded}, 17U,
      embedding.get(), nullptr));
  cuda_check(cudaDeviceSynchronize(), "synchronize dense FP4 embedding smoke");
  const auto actual_embedding = embedding.download();
  for (std::uint32_t column = 0; column < columns; ++column) {
    const auto byte = packed[17U * (padded / 2U) + column / 2U];
    const auto code = static_cast<std::uint8_t>(
        (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
    const auto expected_value =
        decode_fp4(code) * scale(scales[17U * (padded / 32U) + column / 32U]);
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(actual_embedding[column]) -
                 expected_value));
  }
  return maximum_error;
}

double delta_prefill_check() {
  constexpr std::uint32_t rows = 7U;
  constexpr std::uint32_t key_heads = 2U;
  constexpr std::uint32_t value_heads = 4U;
  constexpr std::uint32_t key_head_dim = 32U;
  constexpr std::uint32_t value_head_dim = 32U;
  constexpr std::uint32_t conv_kernel = 4U;
  constexpr float epsilon = 1.0e-6F;
  constexpr std::uint32_t key_dim = key_heads * key_head_dim;
  constexpr std::uint32_t value_dim = value_heads * value_head_dim;
  constexpr std::uint32_t conv_dim = 2U * key_dim + value_dim;
  constexpr std::uint32_t recurrent_values =
      value_heads * key_head_dim * value_head_dim;

  const auto signal = [](std::size_t count, float frequency, float scale) {
    std::vector<float> result(count);
    for (std::size_t index = 0U; index < count; ++index)
      result[index] = std::sin(static_cast<float>(index + 1U) * frequency) *
                      scale;
    return result;
  };
  const auto qkv = signal(static_cast<std::size_t>(rows) * conv_dim, 0.017F,
                          0.35F);
  const auto z = signal(static_cast<std::size_t>(rows) * value_dim, 0.023F,
                        0.25F);
  const auto b = signal(static_cast<std::size_t>(rows) * value_heads, 0.11F,
                        0.4F);
  const auto a = signal(static_cast<std::size_t>(rows) * value_heads, 0.07F,
                        0.3F);
  const auto weights = signal(static_cast<std::size_t>(conv_dim) * conv_kernel,
                              0.019F, 0.2F);
  std::vector<float> time_bias(value_heads);
  std::vector<float> decay_log(value_heads);
  for (std::uint32_t head = 0U; head < value_heads; ++head) {
    time_bias[head] = -0.4F + 0.1F * static_cast<float>(head);
    decay_log[head] = std::log(0.3F + 0.1F * static_cast<float>(head));
  }
  std::vector<float> norm(value_head_dim);
  for (std::uint32_t dimension = 0U; dimension < value_head_dim; ++dimension)
    norm[dimension] = 0.8F + 0.01F * static_cast<float>(dimension);
  const std::vector<float> zero_conv(
      static_cast<std::size_t>(conv_dim) * conv_kernel, 0.0F);
  const std::vector<float> zero_recurrent(recurrent_values, 0.0F);

  DeviceBuffer<float> device_qkv(qkv.size());
  DeviceBuffer<float> device_z(z.size());
  DeviceBuffer<float> device_b(b.size());
  DeviceBuffer<float> device_a(a.size());
  DeviceBuffer<float> device_weights(weights.size());
  DeviceBuffer<float> device_time_bias(time_bias.size());
  DeviceBuffer<float> device_decay_log(decay_log.size());
  DeviceBuffer<float> device_norm(norm.size());
  device_qkv.upload(qkv);
  device_z.upload(z);
  device_b.upload(b);
  device_a.upload(a);
  device_weights.upload(weights);
  device_time_bias.upload(time_bias);
  device_decay_log.upload(decay_log);
  device_norm.upload(norm);

  DeviceBuffer<float> scalar_conv_state(zero_conv.size());
  DeviceBuffer<float> scalar_recurrent(zero_recurrent.size());
  DeviceBuffer<float> scalar_conv_workspace(conv_dim);
  DeviceBuffer<float> scalar_output(static_cast<std::size_t>(rows) * value_dim);
  scalar_conv_state.upload(zero_conv);
  scalar_recurrent.upload(zero_recurrent);
  std::vector<float> scalar_conv_checkpoints(
      static_cast<std::size_t>(rows) * zero_conv.size());
  std::vector<float> scalar_recurrent_checkpoints(
      static_cast<std::size_t>(rows) * zero_recurrent.size());
  for (std::uint32_t row = 0U; row < rows; ++row) {
    status_check(expert::runtime::cuda::split_gated_delta_decode({
        device_qkv.get() + static_cast<std::size_t>(row) * conv_dim,
        device_z.get() + static_cast<std::size_t>(row) * value_dim,
        device_b.get() + static_cast<std::size_t>(row) * value_heads,
        device_a.get() + static_cast<std::size_t>(row) * value_heads,
        device_weights.get(), device_time_bias.get(), device_decay_log.get(),
        device_norm.get(), scalar_conv_state.get(), scalar_recurrent.get(),
        scalar_conv_workspace.get(),
        scalar_output.get() + static_cast<std::size_t>(row) * value_dim,
        key_heads, value_heads, key_head_dim, value_head_dim, conv_kernel,
        epsilon,
        expert::runtime::cuda::GatedDeltaOutputActivation::sigmoid,
        nullptr}));
    const auto conv_snapshot = scalar_conv_state.download();
    const auto recurrent_snapshot = scalar_recurrent.download();
    std::copy(conv_snapshot.begin(), conv_snapshot.end(),
              scalar_conv_checkpoints.begin() +
                  static_cast<std::size_t>(row) * zero_conv.size());
    std::copy(recurrent_snapshot.begin(), recurrent_snapshot.end(),
              scalar_recurrent_checkpoints.begin() +
                  static_cast<std::size_t>(row) * zero_recurrent.size());
  }

  DeviceBuffer<float> batch_conv_state(zero_conv.size());
  DeviceBuffer<float> batch_recurrent(zero_recurrent.size());
  DeviceBuffer<float> batch_conv_workspace(
      static_cast<std::size_t>(rows) * conv_dim);
  DeviceBuffer<float> batch_output(static_cast<std::size_t>(rows) * value_dim);
  DeviceBuffer<float> batch_recurrent_workspace(
      recurrent_values + 2U * static_cast<std::size_t>(rows) * value_heads);
  DeviceBuffer<float> batch_conv_checkpoints(
      static_cast<std::size_t>(rows) * zero_conv.size());
  DeviceBuffer<float> batch_recurrent_checkpoints(
      static_cast<std::size_t>(rows) * zero_recurrent.size());
  batch_conv_state.upload(zero_conv);
  batch_recurrent.upload(zero_recurrent);
  status_check(expert::runtime::cuda::split_gated_delta_prefill({
      device_qkv.get(), device_z.get(), device_b.get(), device_a.get(),
      device_weights.get(), device_time_bias.get(), device_decay_log.get(),
      device_norm.get(), batch_conv_state.get(), batch_recurrent.get(),
      batch_conv_workspace.get(), batch_output.get(),
      batch_recurrent_workspace.get(),
      (recurrent_values +
       2U * static_cast<std::size_t>(rows) * value_heads) * sizeof(float),
      batch_conv_checkpoints.get(), batch_recurrent_checkpoints.get(),
      rows, key_heads, value_heads, key_head_dim, value_head_dim, conv_kernel,
      epsilon,
      expert::runtime::cuda::GatedDeltaOutputActivation::sigmoid,
      nullptr}));
  cuda_check(cudaDeviceSynchronize(), "synchronize gated-delta prefill smoke");

  double maximum_error = 0.0;
  const auto compare = [&](const auto& left, const auto& right) {
    if (left.size() != right.size())
      throw std::runtime_error("gated-delta comparison size mismatch");
    for (std::size_t index = 0U; index < left.size(); ++index)
      maximum_error = std::max(
          maximum_error,
          std::abs(static_cast<double>(left[index]) - right[index]));
  };
  compare(scalar_output.download(), batch_output.download());
  compare(scalar_conv_state.download(), batch_conv_state.download());
  compare(scalar_recurrent.download(), batch_recurrent.download());
  compare(scalar_conv_checkpoints, batch_conv_checkpoints.download());
  DeviceBuffer<float> restored_recurrent_checkpoint(zero_recurrent.size());
  for (std::uint32_t row = 0U; row < rows; ++row) {
    status_check(expert::runtime::cuda::
                     restore_split_gated_delta_recurrent_checkpoint(
                         batch_recurrent_checkpoints.get() +
                             static_cast<std::size_t>(row) * recurrent_values,
                         restored_recurrent_checkpoint.get(), value_heads,
                         key_head_dim, value_head_dim, nullptr));
    compare(std::span<const float>(scalar_recurrent_checkpoints)
                .subspan(static_cast<std::size_t>(row) * recurrent_values,
                         recurrent_values),
            restored_recurrent_checkpoint.download());
  }
  return maximum_error;
}

double attention_prefill_check() {
  constexpr std::uint32_t rows = 11U;
  constexpr std::uint32_t query_heads = 4U;
  constexpr std::uint32_t kv_heads = 2U;
  constexpr std::uint32_t head_dim = 32U;
  constexpr std::uint32_t rotary_dim = 32U;
  constexpr std::uint32_t page_tokens = 8U;
  constexpr std::uint32_t split_tokens = 4U;
  constexpr std::uint32_t maximum_splits = 3U;
  constexpr std::uint32_t query_width = 2U * query_heads * head_dim;
  constexpr std::uint32_t kv_width = kv_heads * head_dim;
  constexpr std::uint32_t attention_width = query_heads * head_dim;
  constexpr std::uint32_t record_bytes = head_dim / 2U + head_dim / 32U;
  constexpr std::uint32_t page_bytes =
      2U * page_tokens * kv_heads * record_bytes;
  constexpr float epsilon = 1.0e-6F;
  constexpr float theta = 10000.0F;

  std::vector<float> query(static_cast<std::size_t>(rows) * query_width);
  std::vector<float> key(static_cast<std::size_t>(rows) * kv_width);
  std::vector<float> value(static_cast<std::size_t>(rows) * kv_width);
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 1U) * 0.013F) * 0.3F;
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::cos(static_cast<float>(index + 1U) * 0.017F) * 0.25F;
    value[index] = std::sin(static_cast<float>(index + 1U) * 0.021F) * 0.2F;
  }
  std::vector<float> query_norm(head_dim);
  std::vector<float> key_norm(head_dim);
  for (std::uint32_t index = 0U; index < head_dim; ++index) {
    query_norm[index] = 0.01F * std::sin(static_cast<float>(index));
    key_norm[index] = 0.01F * std::cos(static_cast<float>(index));
  }
  const std::vector<std::uint8_t> zero_page(page_bytes, 0U);

  DeviceBuffer<float> device_query_norm(head_dim);
  DeviceBuffer<float> device_key_norm(head_dim);
  DeviceBuffer<float> device_value(value.size());
  device_query_norm.upload(query_norm);
  device_key_norm.upload(key_norm);
  device_value.upload(value);

  DeviceBuffer<float> scalar_query(query.size());
  DeviceBuffer<float> scalar_key(key.size());
  DeviceBuffer<float> scalar_output(
      static_cast<std::size_t>(rows) * attention_width);
  DeviceBuffer<std::uint8_t> scalar_page_zero(page_bytes);
  DeviceBuffer<std::uint8_t> scalar_page_one(page_bytes);
  DeviceBuffer<void*> scalar_table(2U);
  scalar_query.upload(query);
  scalar_key.upload(key);
  scalar_page_zero.upload(zero_page);
  scalar_page_one.upload(zero_page);
  scalar_table.upload(
      std::vector<void*>{scalar_page_zero.get(), scalar_page_one.get()});
  DeviceBuffer<float> scalar_maxima(maximum_splits * query_heads);
  DeviceBuffer<float> scalar_sums(maximum_splits * query_heads);
  DeviceBuffer<float> scalar_partials(
      maximum_splits * query_heads * head_dim);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    auto* page = row < page_tokens ? scalar_page_zero.get()
                                   : scalar_page_one.get();
    status_check(
        expert::runtime::cuda::gated_gqa_qkv_rope_cache_paged_fp4_at(
            scalar_query.get() + static_cast<std::size_t>(row) * query_width,
            scalar_key.get() + static_cast<std::size_t>(row) * kv_width,
            device_value.get() + static_cast<std::size_t>(row) * kv_width,
            device_query_norm.get(), device_key_norm.get(), page, 0U,
            page_tokens, row, row, query_heads, kv_heads, head_dim, rotary_dim,
            epsilon, theta, nullptr));
    status_check(
        expert::runtime::cuda::gated_gqa_attention_decode_paged_fp4({
            scalar_query.get() + static_cast<std::size_t>(row) * query_width,
            reinterpret_cast<const void* const*>(scalar_table.get()),
            scalar_output.get() + static_cast<std::size_t>(row) *
                                      attention_width,
            scalar_maxima.get(), scalar_sums.get(), scalar_partials.get(),
            row + 1U, 0U, page_tokens, query_heads, kv_heads, head_dim,
            split_tokens, maximum_splits, nullptr}));
  }

  DeviceBuffer<float> batch_query(query.size());
  DeviceBuffer<float> batch_key(key.size());
  DeviceBuffer<float> batch_output(
      static_cast<std::size_t>(rows) * attention_width);
  DeviceBuffer<std::uint8_t> batch_page_zero(page_bytes);
  DeviceBuffer<std::uint8_t> batch_page_one(page_bytes);
  DeviceBuffer<void*> batch_table(2U);
  batch_query.upload(query);
  batch_key.upload(key);
  batch_page_zero.upload(zero_page);
  batch_page_one.upload(zero_page);
  batch_table.upload(
      std::vector<void*>{batch_page_zero.get(), batch_page_one.get()});
  DeviceBuffer<float> batch_maxima(
      static_cast<std::size_t>(rows) * maximum_splits * query_heads);
  DeviceBuffer<float> batch_sums(
      static_cast<std::size_t>(rows) * maximum_splits * query_heads);
  DeviceBuffer<float> batch_partials(
      static_cast<std::size_t>(rows) * maximum_splits * query_heads *
      head_dim);
  DeviceBuffer<float> staged_output(
      static_cast<std::size_t>(rows) * attention_width);
  const auto staged_query_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  const auto staged_kv_values =
      static_cast<std::size_t>(kv_heads) * split_tokens * head_dim;
  const auto staged_score_values =
      static_cast<std::size_t>(rows) * query_heads * split_tokens;
  DeviceBuffer<std::uint16_t> staged_queries(staged_query_values);
  DeviceBuffer<std::uint16_t> staged_keys(staged_kv_values);
  DeviceBuffer<std::uint16_t> staged_values(staged_kv_values);
  DeviceBuffer<float> staged_scores(staged_score_values);
  DeviceBuffer<std::uint16_t> staged_probabilities(staged_score_values);
  DeviceBuffer<float> staged_accumulator(staged_query_values);
  DeviceBuffer<float> staged_maxima(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> staged_sums(
      static_cast<std::size_t>(rows) * query_heads);
  status_check(
      expert::runtime::cuda::gated_gqa_qkv_rope_cache_paged_fp4_batch(
          batch_query.get(), batch_key.get(), device_value.get(),
          device_query_norm.get(), device_key_norm.get(),
          reinterpret_cast<const void* const*>(batch_table.get()), 0U,
          page_tokens, 0U, 0U, rows, query_heads, kv_heads, head_dim,
          rotary_dim, epsilon, theta, nullptr));
  const expert::runtime::cuda::PagedFp4GatedGqaPrefillLaunch batch_launch{
      batch_query.get(),
      reinterpret_cast<const void* const*>(batch_table.get()),
      batch_output.get(), batch_maxima.get(), batch_sums.get(),
      batch_partials.get(), 1U, rows, 0U, page_tokens, query_heads, kv_heads,
      head_dim, split_tokens, maximum_splits, nullptr};
  status_check(
      expert::runtime::cuda::gated_gqa_attention_prefill_paged_fp4(
          batch_launch));
  auto staged_launch = batch_launch;
  staged_launch.output = staged_output.get();
  status_check(
      expert::runtime::cuda::gated_gqa_attention_staged_prefill_paged_fp4(
          staged_launch,
          {staged_queries.get(),
           staged_query_values * sizeof(std::uint16_t),
           staged_keys.get(),
           staged_kv_values * sizeof(std::uint16_t),
           staged_values.get(),
           staged_kv_values * sizeof(std::uint16_t),
           staged_scores.get(),
           staged_score_values * sizeof(float),
           staged_probabilities.get(),
           staged_score_values * sizeof(std::uint16_t),
           staged_accumulator.get(),
           staged_query_values * sizeof(float),
           staged_maxima.get(),
           static_cast<std::size_t>(rows) * query_heads * sizeof(float),
           staged_sums.get(),
           static_cast<std::size_t>(rows) * query_heads * sizeof(float),
           split_tokens}));
  cuda_check(cudaDeviceSynchronize(), "synchronize attention prefill smoke");

  double maximum_error = 0.0;
  const auto compare = [&](const auto& left, const auto& right) {
    if (left.size() != right.size())
      throw std::runtime_error("attention comparison size mismatch");
    for (std::size_t index = 0U; index < left.size(); ++index)
      maximum_error = std::max(
          maximum_error,
          std::abs(static_cast<double>(left[index]) - right[index]));
  };
  compare(scalar_query.download(), batch_query.download());
  compare(scalar_key.download(), batch_key.download());
  compare(scalar_output.download(), batch_output.download());
  compare(scalar_output.download(), staged_output.download());
  if (scalar_page_zero.download() != batch_page_zero.download() ||
      scalar_page_one.download() != batch_page_one.download())
    maximum_error = std::max(maximum_error, 1.0);
  return maximum_error;
}

struct Fp8KvAttentionCheck final {
  double query_maximum_absolute_difference{};
  double implementation_maximum_absolute_difference{};
  double output_maximum_absolute_difference{};
  double output_mean_absolute_difference{};
  double output_cosine_similarity{};
};

Fp8KvAttentionCheck fp8_kv_attention_check() {
  constexpr std::uint32_t rows = 11U;
  constexpr std::uint32_t query_heads = 8U;
  constexpr std::uint32_t kv_heads = 1U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t rotary_dim = 256U;
  constexpr std::uint32_t page_tokens = 8U;
  constexpr std::uint32_t split_tokens = 8U;
  constexpr std::uint32_t query_width = 2U * query_heads * head_dim;
  constexpr std::uint32_t kv_width = kv_heads * head_dim;
  constexpr std::uint32_t attention_width = query_heads * head_dim;
  constexpr std::uint32_t fp8_record_bytes =
      head_dim + sizeof(std::uint16_t);
  constexpr std::uint32_t page_bytes =
      2U * page_tokens * kv_heads * fp8_record_bytes;
  constexpr std::size_t query_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  constexpr std::size_t raw_kv_values =
      static_cast<std::size_t>(rows) * kv_heads * head_dim;
  constexpr std::size_t staged_kv_values =
      static_cast<std::size_t>(split_tokens) * kv_heads * head_dim;
  constexpr std::size_t score_values =
      static_cast<std::size_t>(rows) * query_heads * split_tokens;
  constexpr std::size_t state_values =
      static_cast<std::size_t>(rows) * query_heads;

  std::vector<float> query(static_cast<std::size_t>(rows) * query_width);
  std::vector<float> key(static_cast<std::size_t>(rows) * kv_width);
  std::vector<float> value(static_cast<std::size_t>(rows) * kv_width);
  std::vector<float> query_norm(head_dim);
  std::vector<float> key_norm(head_dim);
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 3U) * 0.011F) *
                   (0.15F + static_cast<float>(index % 17U) * 0.005F);
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::cos(static_cast<float>(index + 5U) * 0.013F) * 0.4F;
    value[index] =
        std::sin(static_cast<float>(index + 7U) * 0.017F) * 0.35F;
  }
  for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
    query_norm[dimension] =
        0.01F * std::sin(static_cast<float>(dimension) * 0.1F);
    key_norm[dimension] =
        0.01F * std::cos(static_cast<float>(dimension) * 0.1F);
  }

  DeviceBuffer<float> device_query_norm(head_dim);
  DeviceBuffer<float> device_key_norm(head_dim);
  DeviceBuffer<float> device_value(value.size());
  device_query_norm.upload(query_norm);
  device_key_norm.upload(key_norm);
  device_value.upload(value);

  DeviceBuffer<float> fp16_query(query.size());
  DeviceBuffer<float> fp16_key(key.size());
  DeviceBuffer<std::uint16_t> fp16_keys(raw_kv_values);
  DeviceBuffer<std::uint16_t> fp16_values(raw_kv_values);
  DeviceBuffer<float> fp16_output(
      static_cast<std::size_t>(rows) * attention_width);
  fp16_query.upload(query);
  fp16_key.upload(key);
  status_check(expert::runtime::cuda::gated_gqa_qkv_rope_fp16_batch(
      fp16_query.get(), fp16_key.get(), device_value.get(),
      device_query_norm.get(), device_key_norm.get(), fp16_keys.get(),
      fp16_values.get(), 0U, rows, query_heads, kv_heads, head_dim,
      rotary_dim, 1.0e-6F, 10000.0F, nullptr));

  DeviceBuffer<std::uint16_t> fp16_staged_queries(query_values);
  DeviceBuffer<std::uint16_t> fp16_raw_keys(staged_kv_values);
  DeviceBuffer<std::uint16_t> fp16_raw_values(staged_kv_values);
  DeviceBuffer<std::uint16_t> fp16_staged_keys(staged_kv_values);
  DeviceBuffer<std::uint16_t> fp16_staged_values(staged_kv_values);
  DeviceBuffer<float> fp16_scores(score_values);
  DeviceBuffer<std::uint16_t> fp16_probabilities(score_values);
  DeviceBuffer<float> fp16_accumulator(query_values);
  DeviceBuffer<float> fp16_maxima(state_values);
  DeviceBuffer<float> fp16_sums(state_values);
  status_check(expert::runtime::cuda::gated_gqa_attention_staged_device_fp16(
      {fp16_query.get(), fp16_keys.get(), fp16_values.get(),
       fp16_output.get(), rows, 1U, 0U, rows, query_heads, kv_heads,
       head_dim, nullptr},
      {fp16_staged_queries.get(), query_values * sizeof(std::uint16_t),
       fp16_raw_keys.get(), staged_kv_values * sizeof(std::uint16_t),
       fp16_raw_values.get(), staged_kv_values * sizeof(std::uint16_t),
       fp16_staged_keys.get(), staged_kv_values * sizeof(std::uint16_t),
       fp16_staged_values.get(), staged_kv_values * sizeof(std::uint16_t),
       fp16_scores.get(), score_values * sizeof(float),
       fp16_probabilities.get(), score_values * sizeof(std::uint16_t),
       fp16_accumulator.get(), query_values * sizeof(float),
       fp16_maxima.get(), state_values * sizeof(float), fp16_sums.get(),
       state_values * sizeof(float), split_tokens}));

  DeviceBuffer<float> fp8_query(query.size());
  DeviceBuffer<float> fp8_key(key.size());
  DeviceBuffer<std::uint8_t> fp8_page_zero(page_bytes);
  DeviceBuffer<std::uint8_t> fp8_page_one(page_bytes);
  DeviceBuffer<void*> fp8_page_table(2U);
  DeviceBuffer<float> fp8_output(
      static_cast<std::size_t>(rows) * attention_width);
  fp8_query.upload(query);
  fp8_key.upload(key);
  fp8_page_table.upload(
      std::vector<void*>{fp8_page_zero.get(), fp8_page_one.get()});
  status_check(
      expert::runtime::cuda::gated_gqa_qkv_rope_cache_paged_fp8_batch(
          fp8_query.get(), fp8_key.get(), device_value.get(),
          device_query_norm.get(), device_key_norm.get(),
          reinterpret_cast<const void* const*>(fp8_page_table.get()), 0U,
          page_tokens, 0U, 0U, rows, query_heads, kv_heads, head_dim,
          rotary_dim, 1.0e-6F, 10000.0F, nullptr));

  DeviceBuffer<std::uint16_t> fp8_staged_queries(query_values);
  DeviceBuffer<std::uint16_t> fp8_staged_keys(staged_kv_values);
  DeviceBuffer<std::uint16_t> fp8_staged_values(staged_kv_values);
  DeviceBuffer<float> fp8_scores(score_values);
  DeviceBuffer<std::uint16_t> fp8_probabilities(score_values);
  DeviceBuffer<float> fp8_accumulator(query_values);
  DeviceBuffer<float> fp8_maxima(state_values);
  DeviceBuffer<float> fp8_sums(state_values);
  const expert::runtime::cuda::PagedFp8GatedGqaPrefillLaunch fp8_launch{
      fp8_query.get(),
      reinterpret_cast<const void* const*>(fp8_page_table.get()),
      fp8_output.get(), nullptr, nullptr, nullptr, 1U, rows, 0U,
      page_tokens, query_heads, kv_heads, head_dim, split_tokens, 2U,
      nullptr};
  status_check(
      expert::runtime::cuda::gated_gqa_attention_staged_prefill_paged_fp8(
          fp8_launch,
          {fp8_staged_queries.get(), query_values * sizeof(std::uint16_t),
           fp8_staged_keys.get(), staged_kv_values * sizeof(std::uint16_t),
           fp8_staged_values.get(), staged_kv_values * sizeof(std::uint16_t),
           fp8_scores.get(), score_values * sizeof(float),
           fp8_probabilities.get(), score_values * sizeof(std::uint16_t),
           fp8_accumulator.get(), query_values * sizeof(float),
           fp8_maxima.get(), state_values * sizeof(float), fp8_sums.get(),
           state_values * sizeof(float), split_tokens}));
  DeviceBuffer<float> fp8_decode_output(attention_width);
  DeviceBuffer<float> fp8_decode_maxima(2U * query_heads);
  DeviceBuffer<float> fp8_decode_sums(2U * query_heads);
  DeviceBuffer<float> fp8_decode_partials(
      2U * query_heads * head_dim);
  status_check(
      expert::runtime::cuda::gated_gqa_attention_decode_paged_fp8_tensor_core({
          fp8_query.get() +
              static_cast<std::size_t>(rows - 1U) * query_width,
          reinterpret_cast<const void* const*>(fp8_page_table.get()),
          fp8_decode_output.get(), fp8_decode_maxima.get(),
          fp8_decode_sums.get(), fp8_decode_partials.get(), rows, 0U,
          page_tokens, query_heads, kv_heads, head_dim, split_tokens, 2U,
          nullptr}));
  cuda_check(cudaDeviceSynchronize(), "synchronize FP8 KV attention check");

  const auto reference_query = fp16_query.download();
  const auto candidate_query = fp8_query.download();
  const auto reference_output = fp16_output.download();
  const auto candidate_output = fp8_output.download();
  const auto decode_output = fp8_decode_output.download();
  Fp8KvAttentionCheck result;
  double dot{};
  double reference_square{};
  double candidate_square{};
  for (std::size_t index = 0U; index < reference_query.size(); ++index)
    result.query_maximum_absolute_difference = std::max(
        result.query_maximum_absolute_difference,
        std::abs(static_cast<double>(reference_query[index]) -
                 candidate_query[index]));
  for (std::size_t index = 0U; index < reference_output.size(); ++index) {
    const auto reference = static_cast<double>(reference_output[index]);
    const auto candidate = static_cast<double>(candidate_output[index]);
    const auto difference = std::abs(reference - candidate);
    result.output_maximum_absolute_difference =
        std::max(result.output_maximum_absolute_difference, difference);
    result.output_mean_absolute_difference += difference;
    dot += reference * candidate;
    reference_square += reference * reference;
    candidate_square += candidate * candidate;
  }
  const auto final_output_offset =
      static_cast<std::size_t>(rows - 1U) * attention_width;
  for (std::size_t index = 0U; index < decode_output.size(); ++index)
    result.implementation_maximum_absolute_difference = std::max(
        result.implementation_maximum_absolute_difference,
        std::abs(static_cast<double>(
            candidate_output[final_output_offset + index] -
            decode_output[index])));
  result.output_mean_absolute_difference /= reference_output.size();
  result.output_cosine_similarity =
      dot / std::sqrt(reference_square * candidate_square);
  return result;
}

struct Q8KvAttentionCheck final {
  double layout_maximum_byte_difference{};
  double independent_oracle_maximum_absolute_difference{};
  double five_query_maximum_absolute_difference{};
};

double q8_five_query_microbatch_check() {
  constexpr std::uint32_t context_tokens = 16U;
  constexpr std::uint32_t first_context_tokens = 12U;
  constexpr std::uint32_t rows = 5U;
  constexpr std::uint32_t query_heads = 6U;
  constexpr std::uint32_t kv_heads = 1U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = context_tokens;
  constexpr std::uint32_t record_bytes =
      head_dim + sizeof(std::uint16_t);
  constexpr std::uint32_t page_bytes =
      2U * page_tokens * kv_heads * record_bytes;
  constexpr std::uint32_t query_width =
      2U * query_heads * head_dim;

  std::vector<float> key(
      static_cast<std::size_t>(context_tokens) * head_dim);
  std::vector<float> value(key.size());
  std::vector<float> query(static_cast<std::size_t>(rows) * query_width);
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::sin(static_cast<float>(index + 31U) * 0.013F) *
                 (0.19F + static_cast<float>(index % 11U) * 0.009F);
    value[index] = std::cos(static_cast<float>(index + 23U) * 0.017F) *
                   (0.17F + static_cast<float>(index % 9U) * 0.011F);
  }
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 7U) * 0.005F) *
                   0.37F;

  // Independent Q8 encoder/dequantizer used only by the CPU oracle.
  const auto decode = [](const std::vector<float>& source) {
    std::vector<float> result(source.size());
    for (std::uint32_t token = 0U; token < context_tokens; ++token) {
      float maximum{};
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
        maximum = std::max(
            maximum,
            std::abs(source[static_cast<std::size_t>(token) * head_dim +
                            dimension]));
      const auto scale = binary16_value(binary16_bits(
          std::max(maximum / 127.0F, std::ldexp(1.0F, -24))));
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto index =
            static_cast<std::size_t>(token) * head_dim + dimension;
        const auto encoded = std::max(
            -127, std::min(127, static_cast<int>(
                std::nearbyint(source[index] / scale))));
        result[index] = static_cast<float>(encoded) * scale;
      }
    }
    return result;
  };
  const auto decoded_key = decode(key);
  const auto decoded_value = decode(value);

  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<std::uint8_t> device_page(page_bytes);
  DeviceBuffer<void*> device_page_table(1U);
  DeviceBuffer<float> device_output(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  DeviceBuffer<float> partial_maxima(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> partial_sums(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> partial_outputs(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  device_key.upload(key);
  device_value.upload(value);
  device_query.upload(query);
  cuda_check(cudaMemset(device_page.get(), 0, page_bytes),
             "initialize five-query Q8 page");
  device_page_table.upload(std::vector<void*>{device_page.get()});
  status_check(expert::runtime::cuda::store_gqa_kv_paged_q8_batch(
      device_key.get(), device_value.get(),
      reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
      page_tokens, 0U, context_tokens, kv_heads, head_dim, nullptr));
  status_check(
      expert::runtime::cuda::gated_gqa_attention_microbatch_paged_q8_tensor_core(
          {device_query.get(),
           reinterpret_cast<const void* const*>(device_page_table.get()),
           device_output.get(), partial_maxima.get(), partial_sums.get(),
           partial_outputs.get(), first_context_tokens, rows, 0U,
           page_tokens, query_heads, kv_heads, head_dim, context_tokens, 1U,
           nullptr}));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize five-query Q8 microbatch check");

  std::vector<float> expected(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const auto visible_tokens = first_context_tokens + row;
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto* head_query = query.data() +
          (static_cast<std::size_t>(row) * query_heads + head) *
              2U * head_dim;
      std::vector<float> scores(visible_tokens);
      auto maximum = -std::numeric_limits<float>::infinity();
      for (std::uint32_t token = 0U; token < visible_tokens; ++token) {
        float dot{};
        for (std::uint32_t dimension = 0U; dimension < head_dim;
             ++dimension)
          dot += rounded_bf16(head_query[dimension]) *
                 rounded_bf16(decoded_key[
                     static_cast<std::size_t>(token) * head_dim +
                     dimension]);
        scores[token] = dot / std::sqrt(static_cast<float>(head_dim));
        maximum = std::max(maximum, scores[token]);
      }
      std::vector<float> probabilities(visible_tokens);
      float denominator{};
      for (std::uint32_t token = 0U; token < visible_tokens; ++token) {
        probabilities[token] =
            rounded_bf16(std::exp(scores[token] - maximum));
        denominator += probabilities[token];
      }
      for (std::uint32_t dimension = 0U; dimension < head_dim;
           ++dimension) {
        float numerator{};
        for (std::uint32_t token = 0U; token < visible_tokens; ++token)
          numerator += probabilities[token] *
              rounded_bf16(decoded_value[
                  static_cast<std::size_t>(token) * head_dim + dimension]);
        const auto gate = head_query[head_dim + dimension];
        expected[(static_cast<std::size_t>(row) * query_heads + head) *
                     head_dim +
                 dimension] =
            (numerator / denominator) / (1.0F + std::exp(-gate));
      }
    }
  }
  const auto actual = device_output.download();
  double maximum_difference{};
  for (std::size_t index = 0U; index < actual.size(); ++index)
    maximum_difference = std::max(
        maximum_difference,
        std::abs(static_cast<double>(actual[index]) - expected[index]));
  return maximum_difference;
}

Q8KvAttentionCheck q8_kv_attention_check() {
  constexpr std::uint32_t rows = 8U;
  constexpr std::uint32_t query_heads = 8U;
  constexpr std::uint32_t kv_heads = 1U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = rows;
  constexpr std::uint32_t record_bytes =
      head_dim + sizeof(std::uint16_t);
  constexpr std::uint32_t page_bytes =
      2U * page_tokens * kv_heads * record_bytes;
  constexpr std::uint32_t query_width = 2U * query_heads * head_dim;

  std::vector<float> key(static_cast<std::size_t>(rows) * head_dim);
  std::vector<float> value(key.size());
  std::vector<float> query(query_width);
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::sin(static_cast<float>(index + 11U) * 0.019F) *
                 (0.2F + static_cast<float>(index % 13U) * 0.017F);
    value[index] = std::cos(static_cast<float>(index + 17U) * 0.023F) *
                   (0.15F + static_cast<float>(index % 7U) * 0.013F);
  }
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 5U) * 0.007F) * 0.4F;

  // Independent record encoder. It deliberately does not call or share a
  // scale/layout helper with the CUDA implementation under test.
  std::vector<std::uint8_t> expected_page(page_bytes);
  std::vector<float> decoded_key(key.size());
  std::vector<float> decoded_value(value.size());
  const auto encode_kind = [&](const std::vector<float>& source,
                               std::vector<float>& decoded,
                               std::size_t kind_offset) {
    for (std::uint32_t row = 0U; row < rows; ++row) {
      float maximum{};
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
        maximum = std::max(
            maximum,
            std::abs(source[static_cast<std::size_t>(row) * head_dim +
                            dimension]));
      const auto scale_bits = binary16_bits(
          std::max(maximum / 127.0F, std::ldexp(1.0F, -24)));
      const auto record_scale = binary16_value(scale_bits);
      auto* record = expected_page.data() + kind_offset +
                     static_cast<std::size_t>(row) * record_bytes;
      std::memcpy(record + head_dim, &scale_bits, sizeof(scale_bits));
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto index =
            static_cast<std::size_t>(row) * head_dim + dimension;
        const auto quantized = std::max(
            -127, std::min(127, static_cast<int>(
                std::nearbyint(source[index] / record_scale))));
        const auto signed_value = static_cast<std::int8_t>(quantized);
        std::memcpy(record + dimension, &signed_value, sizeof(signed_value));
        decoded[index] = static_cast<float>(signed_value) * record_scale;
      }
    }
  };
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  encode_kind(key, decoded_key, 0U);
  encode_kind(value, decoded_value, records_per_kind * record_bytes);

  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<std::uint8_t> device_page(page_bytes);
  DeviceBuffer<void*> device_page_table(1U);
  DeviceBuffer<float> device_output(
      static_cast<std::size_t>(query_heads) * head_dim);
  DeviceBuffer<float> partial_maxima(query_heads);
  DeviceBuffer<float> partial_sums(query_heads);
  DeviceBuffer<float> partial_outputs(
      static_cast<std::size_t>(query_heads) * head_dim);
  device_key.upload(key);
  device_value.upload(value);
  device_query.upload(query);
  cuda_check(cudaMemset(device_page.get(), 0, page_bytes),
             "initialize Q8 numerical page");
  device_page_table.upload(std::vector<void*>{device_page.get()});
  status_check(expert::runtime::cuda::store_gqa_kv_paged_q8_batch(
      device_key.get(), device_value.get(),
      reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
      page_tokens, 0U, rows, kv_heads, head_dim, nullptr));
  status_check(
      expert::runtime::cuda::gated_gqa_attention_decode_paged_q8_tensor_core({
          device_query.get(),
          reinterpret_cast<const void* const*>(device_page_table.get()),
          device_output.get(), partial_maxima.get(), partial_sums.get(),
          partial_outputs.get(), rows, 0U, page_tokens, query_heads, kv_heads,
          head_dim, rows, 1U, nullptr}));
  cuda_check(cudaDeviceSynchronize(), "synchronize Q8 KV attention check");

  Q8KvAttentionCheck result;
  const auto actual_page = device_page.download();
  for (std::size_t index = 0U; index < expected_page.size(); ++index)
    result.layout_maximum_byte_difference = std::max(
        result.layout_maximum_byte_difference,
        static_cast<double>(std::abs(static_cast<int>(actual_page[index]) -
                                     expected_page[index])));

  std::vector<float> expected_output(
      static_cast<std::size_t>(query_heads) * head_dim);
  for (std::uint32_t head = 0U; head < query_heads; ++head) {
    const auto* head_query = query.data() +
        static_cast<std::size_t>(head) * 2U * head_dim;
    std::array<float, rows> scores{};
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t row = 0U; row < rows; ++row) {
      float dot{};
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
        dot += rounded_bf16(head_query[dimension]) *
               rounded_bf16(decoded_key[
                   static_cast<std::size_t>(row) * head_dim + dimension]);
      scores[row] = dot / std::sqrt(static_cast<float>(head_dim));
      maximum = std::max(maximum, scores[row]);
    }
    std::array<float, rows> probabilities{};
    float denominator{};
    for (std::uint32_t row = 0U; row < rows; ++row) {
      probabilities[row] = rounded_bf16(std::exp(scores[row] - maximum));
      denominator += probabilities[row];
    }
    for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
      float numerator{};
      for (std::uint32_t row = 0U; row < rows; ++row)
        numerator += probabilities[row] * rounded_bf16(decoded_value[
            static_cast<std::size_t>(row) * head_dim + dimension]);
      const auto gate = head_query[head_dim + dimension];
      expected_output[static_cast<std::size_t>(head) * head_dim + dimension] =
          (numerator / denominator) / (1.0F + std::exp(-gate));
    }
  }
  const auto actual_output = device_output.download();
  for (std::size_t index = 0U; index < actual_output.size(); ++index)
    result.independent_oracle_maximum_absolute_difference = std::max(
        result.independent_oracle_maximum_absolute_difference,
        std::abs(static_cast<double>(actual_output[index]) -
                 expected_output[index]));
  result.five_query_maximum_absolute_difference =
      q8_five_query_microbatch_check();
  return result;
}

struct Fp4KeyOutlier1Check final {
  double layout_maximum_byte_difference{};
  double implementation_maximum_absolute_difference{};
  double prefill_implementation_maximum_absolute_difference{};
  double candidate_fp16_maximum_absolute_difference{};
  double ordinary_fp4_fp16_maximum_absolute_difference{};
};

Fp4KeyOutlier1Check fp4_key_outlier1_check() {
  constexpr std::uint32_t rows = 8U;
  constexpr std::uint32_t query_heads = 8U;
  constexpr std::uint32_t kv_heads = 1U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = rows;
  constexpr std::uint32_t blocks = head_dim / 32U;
  constexpr std::uint32_t fp4_record_bytes =
      head_dim / 2U + blocks;
  constexpr std::uint32_t key_record_bytes =
      fp4_record_bytes + blocks * sizeof(std::uint32_t);
  constexpr std::uint32_t page_bytes =
      page_tokens * kv_heads * (key_record_bytes + fp4_record_bytes);
  constexpr std::uint32_t query_width = 2U * query_heads * head_dim;
  constexpr std::uint32_t kv_width = kv_heads * head_dim;
  constexpr std::size_t kv_values =
      static_cast<std::size_t>(rows) * kv_width;

  std::vector<float> key(kv_values);
  std::vector<float> value(kv_values);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
      const auto index = static_cast<std::size_t>(row) * head_dim + dimension;
      key[index] = std::sin(static_cast<float>(index + 5U) * 0.037F) *
                   (0.2F + 0.01F * static_cast<float>(dimension % 11U));
      value[index] = std::cos(static_cast<float>(index + 7U) * 0.029F) *
                     (0.15F + 0.01F * static_cast<float>(dimension % 13U));
    }
    for (std::uint32_t block = 0U; block < blocks; ++block) {
      const auto outlier = (row * 7U + block * 11U + 3U) % 32U;
      key[static_cast<std::size_t>(row) * head_dim + block * 32U + outlier] =
          ((row + block) & 1U ? -1.0F : 1.0F) *
          (3.0F + 0.25F * static_cast<float>(block));
    }
  }
  std::vector<float> query(static_cast<std::size_t>(rows) * query_width);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto base =
            (static_cast<std::size_t>(row) * query_heads + head) * 2U *
            head_dim;
        query[base + dimension] =
            std::sin(static_cast<float>(
                         (row * query_heads + head) * head_dim + dimension +
                         1U) *
                     0.007F) *
            0.2F;
        query[base + head_dim + dimension] =
            std::cos(static_cast<float>(row + head + dimension + 1U) *
                     0.011F) *
            0.3F;
      }
    }
  }

  std::vector<std::uint8_t> expected_page(page_bytes, 0U);
  std::vector<float> decoded_candidate_keys(kv_values);
  std::vector<float> decoded_candidate_values(kv_values);
  std::vector<float> decoded_fp4_keys(kv_values);
  std::vector<float> decoded_fp4_values(kv_values);
  auto* expected_keys = expected_page.data();
  auto* expected_values =
      expected_keys + page_tokens * kv_heads * key_record_bytes;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    auto* key_record = expected_keys +
        static_cast<std::size_t>(row) * key_record_bytes;
    auto* value_record = expected_values +
        static_cast<std::size_t>(row) * fp4_record_bytes;
    for (std::uint32_t block = 0U; block < blocks; ++block) {
      const auto first = block * 32U;
      std::uint32_t outlier{};
      float outlier_magnitude{};
      float value_maximum{};
      for (std::uint32_t offset = 0U; offset < 32U; ++offset) {
        const auto index = static_cast<std::size_t>(row) * head_dim +
                           first + offset;
        const auto magnitude = std::abs(key[index]);
        if (magnitude > outlier_magnitude) {
          outlier_magnitude = magnitude;
          outlier = offset;
        }
        value_maximum = std::max(value_maximum, std::abs(value[index]));
      }
      float retained_key_maximum{};
      for (std::uint32_t offset = 0U; offset < 32U; ++offset)
        if (offset != outlier)
          retained_key_maximum = std::max(
              retained_key_maximum,
              std::abs(key[static_cast<std::size_t>(row) * head_dim +
                           first + offset]));
      const auto candidate_key_scale_code =
          host_fp4_scale_code(retained_key_maximum);
      const auto ordinary_key_scale_code =
          host_fp4_scale_code(outlier_magnitude);
      const auto value_scale_code = host_fp4_scale_code(value_maximum);
      const auto candidate_key_scale = scale(candidate_key_scale_code);
      const auto ordinary_key_scale = scale(ordinary_key_scale_code);
      const auto value_scale = scale(value_scale_code);
      key_record[head_dim / 2U + block] = candidate_key_scale_code;
      value_record[head_dim / 2U + block] = value_scale_code;
      auto* correction = key_record + fp4_record_bytes +
                         block * sizeof(std::uint32_t);
      correction[0] = static_cast<std::uint8_t>(outlier);
      correction[1] = 0U;
      const auto outlier_bits = binary16_bits(
          key[static_cast<std::size_t>(row) * head_dim + first + outlier]);
      std::memcpy(correction + 2U, &outlier_bits, sizeof(outlier_bits));
      for (std::uint32_t offset = 0U; offset < 32U; offset += 2U) {
        const auto first_index =
            static_cast<std::size_t>(row) * head_dim + first + offset;
        const auto key_low =
            host_encode_fp4(key[first_index], candidate_key_scale);
        const auto key_high =
            host_encode_fp4(key[first_index + 1U], candidate_key_scale);
        const auto value_low =
            host_encode_fp4(value[first_index], value_scale);
        const auto value_high =
            host_encode_fp4(value[first_index + 1U], value_scale);
        key_record[(first + offset) / 2U] =
            static_cast<std::uint8_t>(key_low | (key_high << 4U));
        value_record[(first + offset) / 2U] =
            static_cast<std::uint8_t>(value_low | (value_high << 4U));
      }
      for (std::uint32_t offset = 0U; offset < 32U; ++offset) {
        const auto index = static_cast<std::size_t>(row) * head_dim +
                           first + offset;
        const auto packed_key = key_record[(first + offset) / 2U];
        const auto key_code = static_cast<std::uint8_t>(
            (offset & 1U) == 0U ? packed_key & 0x0fU : packed_key >> 4U);
        const auto packed_value = value_record[(first + offset) / 2U];
        const auto value_code = static_cast<std::uint8_t>(
            (offset & 1U) == 0U ? packed_value & 0x0fU
                                : packed_value >> 4U);
        decoded_candidate_keys[index] =
            offset == outlier ? binary16_value(outlier_bits)
                              : decode_fp4(key_code) * candidate_key_scale;
        decoded_candidate_values[index] =
            decode_fp4(value_code) * value_scale;
        decoded_fp4_keys[index] =
            decode_fp4(host_encode_fp4(key[index], ordinary_key_scale)) *
            ordinary_key_scale;
        decoded_fp4_values[index] = decoded_candidate_values[index];
      }
    }
  }

  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_output(
      static_cast<std::size_t>(query_heads) * head_dim);
  DeviceBuffer<float> device_prefill_output(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  DeviceBuffer<std::uint8_t> device_page(page_bytes);
  DeviceBuffer<void*> device_page_table(1U);
  DeviceBuffer<float> partial_maxima(query_heads);
  DeviceBuffer<float> partial_sums(query_heads);
  DeviceBuffer<float> partial_outputs(
      static_cast<std::size_t>(query_heads) * head_dim);
  constexpr std::uint32_t staged_split_tokens = 3U;
  const auto staged_query_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  const auto staged_kv_values = static_cast<std::size_t>(kv_heads) *
                                staged_split_tokens * head_dim;
  const auto staged_score_values =
      static_cast<std::size_t>(rows) * query_heads * staged_split_tokens;
  DeviceBuffer<std::uint16_t> staged_queries(staged_query_values);
  DeviceBuffer<std::uint16_t> staged_keys(staged_kv_values);
  DeviceBuffer<std::uint16_t> staged_values(staged_kv_values);
  DeviceBuffer<float> staged_scores(staged_score_values);
  DeviceBuffer<std::uint16_t> staged_probabilities(staged_score_values);
  DeviceBuffer<float> staged_accumulator(staged_query_values);
  DeviceBuffer<float> staged_maxima(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> staged_sums(
      static_cast<std::size_t>(rows) * query_heads);
  device_key.upload(key);
  device_value.upload(value);
  device_query.upload(query);
  cuda_check(cudaMemset(device_page.get(), 0, page_bytes),
             "initialize FP4 key-outlier-1 numerical page");
  device_page_table.upload(std::vector<void*>{device_page.get()});
  status_check(expert::runtime::cuda::store_gqa_kv_paged_fp4_key_outlier1_batch(
      device_key.get(), device_value.get(),
      reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
      page_tokens, 0U, rows, kv_heads, head_dim, nullptr));
  status_check(expert::runtime::cuda::
                   gated_gqa_attention_decode_paged_fp4_key_outlier1_tensor_core(
                       {device_query.get() +
                            static_cast<std::size_t>(rows - 1U) * query_width,
                        reinterpret_cast<const void* const*>(
                            device_page_table.get()),
                        device_output.get(), partial_maxima.get(),
                        partial_sums.get(), partial_outputs.get(), rows, 0U,
                        page_tokens, query_heads, kv_heads, head_dim, rows,
                        1U, nullptr}));
  status_check(expert::runtime::cuda::
                   gated_gqa_attention_staged_prefill_paged_fp4_key_outlier1(
                       {device_query.get(),
                        reinterpret_cast<const void* const*>(
                            device_page_table.get()),
                        device_prefill_output.get(), nullptr, nullptr,
                        nullptr, 1U, rows, 0U, page_tokens, query_heads,
                        kv_heads, head_dim, rows, 1U, nullptr},
                       {staged_queries.get(),
                        staged_query_values * sizeof(std::uint16_t),
                        staged_keys.get(),
                        staged_kv_values * sizeof(std::uint16_t),
                        staged_values.get(),
                        staged_kv_values * sizeof(std::uint16_t),
                        staged_scores.get(),
                        staged_score_values * sizeof(float),
                        staged_probabilities.get(),
                        staged_score_values * sizeof(std::uint16_t),
                        staged_accumulator.get(),
                        staged_query_values * sizeof(float),
                        staged_maxima.get(),
                        static_cast<std::size_t>(rows) * query_heads *
                            sizeof(float),
                        staged_sums.get(),
                        static_cast<std::size_t>(rows) * query_heads *
                            sizeof(float),
                        staged_split_tokens}));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize FP4 key-outlier-1 numerical check");

  const auto actual_page = device_page.download();
  Fp4KeyOutlier1Check result;
  for (std::size_t index = 0U; index < expected_page.size(); ++index)
    result.layout_maximum_byte_difference = std::max(
        result.layout_maximum_byte_difference,
        static_cast<double>(std::abs(static_cast<int>(actual_page[index]) -
                                     expected_page[index])));

  const auto attention = [&](const std::vector<float>& keys,
                             const std::vector<float>& values,
                             std::uint32_t query_row,
                             std::uint32_t context_rows) {
    std::vector<float> output(
        static_cast<std::size_t>(query_heads) * head_dim);
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto query_base =
          (static_cast<std::size_t>(query_row) * query_heads + head) * 2U *
          head_dim;
      std::array<float, rows> scores{};
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::uint32_t row = 0U; row < context_rows; ++row) {
        float score{};
        for (std::uint32_t dimension = 0U; dimension < head_dim;
             ++dimension)
          score += rounded_bf16(query[query_base + dimension]) *
                   rounded_bf16(keys[static_cast<std::size_t>(row) *
                                         head_dim + dimension]);
        scores[row] = score / std::sqrt(static_cast<float>(head_dim));
        maximum = std::max(maximum, scores[row]);
      }
      std::array<float, rows> probabilities{};
      float sum{};
      for (std::uint32_t row = 0U; row < context_rows; ++row) {
        probabilities[row] = rounded_bf16(std::exp(scores[row] - maximum));
        sum += probabilities[row];
      }
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        float accumulated{};
        for (std::uint32_t row = 0U; row < context_rows; ++row)
          accumulated += probabilities[row] * rounded_bf16(
              values[static_cast<std::size_t>(row) * head_dim + dimension]);
        const auto gate = query[query_base + head_dim + dimension];
        output[static_cast<std::size_t>(head) * head_dim + dimension] =
            (accumulated / sum) / (1.0F + std::exp(-gate));
      }
    }
    return output;
  };
  std::vector<float> fp16_keys(key.size());
  std::vector<float> fp16_values(value.size());
  for (std::size_t index = 0U; index < key.size(); ++index) {
    fp16_keys[index] = binary16_value(binary16_bits(key[index]));
    fp16_values[index] = binary16_value(binary16_bits(value[index]));
  }
  const auto candidate_oracle =
      attention(decoded_candidate_keys, decoded_candidate_values,
                rows - 1U, rows);
  const auto fp16_oracle =
      attention(fp16_keys, fp16_values, rows - 1U, rows);
  const auto ordinary_fp4_oracle = attention(decoded_fp4_keys,
                                              decoded_fp4_values,
                                              rows - 1U, rows);
  const auto actual_output = device_output.download();
  for (std::size_t index = 0U; index < actual_output.size(); ++index) {
    result.implementation_maximum_absolute_difference = std::max(
        result.implementation_maximum_absolute_difference,
        std::abs(static_cast<double>(actual_output[index]) -
                 candidate_oracle[index]));
    result.candidate_fp16_maximum_absolute_difference = std::max(
        result.candidate_fp16_maximum_absolute_difference,
        std::abs(static_cast<double>(candidate_oracle[index]) -
                 fp16_oracle[index]));
    result.ordinary_fp4_fp16_maximum_absolute_difference = std::max(
        result.ordinary_fp4_fp16_maximum_absolute_difference,
        std::abs(static_cast<double>(ordinary_fp4_oracle[index]) -
                 fp16_oracle[index]));
  }
  const auto actual_prefill = device_prefill_output.download();
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const auto oracle = attention(decoded_candidate_keys,
                                  decoded_candidate_values, row, row + 1U);
    for (std::size_t index = 0U; index < oracle.size(); ++index) {
      const auto actual_index =
          static_cast<std::size_t>(row) * oracle.size() + index;
      result.prefill_implementation_maximum_absolute_difference = std::max(
          result.prefill_implementation_maximum_absolute_difference,
          std::abs(static_cast<double>(actual_prefill[actual_index]) -
                   oracle[index]));
    }
  }
  return result;
}

struct Q4BfpKeyOutlier1Check final {
  double layout_maximum_byte_difference{};
  double query_maximum_byte_difference{};
  double query_scale_maximum_absolute_difference{};
  double independent_oracle_maximum_absolute_difference{};
};

std::uint8_t host_q4_bfp_exponent(float desired_scale, float base) {
  const auto ratio = desired_scale / base;
  if (ratio <= 1.0F) return 0U;
  return static_cast<std::uint8_t>(
      std::max(0, std::min(4, static_cast<int>(std::ceil(std::log2(ratio))))));
}

std::uint8_t host_signed_q4(float value, float scale) {
  const auto quantized = std::max(
      -7, std::min(7, static_cast<int>(std::nearbyint(value / scale))));
  return static_cast<std::uint8_t>(quantized & 0x0f);
}

std::int32_t host_signed_q4_value(std::uint8_t nibble) {
  return nibble >= 8U ? static_cast<std::int32_t>(nibble) - 16
                      : static_cast<std::int32_t>(nibble);
}

std::uint8_t host_q5_bfp_exponent(float desired_scale, float base) {
  const auto ratio = desired_scale / base;
  if (ratio <= 1.0F) return 0U;
  return static_cast<std::uint8_t>(
      std::max(0, std::min(3, static_cast<int>(std::ceil(std::log2(ratio))))));
}

std::uint8_t host_signed_q5(float value, float scale) {
  const auto quantized = std::max(
      -15, std::min(15, static_cast<int>(std::nearbyint(value / scale))));
  return static_cast<std::uint8_t>(quantized & 0x1f);
}

std::int32_t host_signed_q5_value(std::uint8_t code) {
  return code >= 16U ? static_cast<std::int32_t>(code) - 32
                     : static_cast<std::int32_t>(code);
}

template <bool PreserveKeyOutlier, bool PerHead = false>
Q4BfpKeyOutlier1Check q4_bfp_check_impl() {
  static_assert(!(PreserveKeyOutlier && PerHead));
  constexpr std::uint32_t stored_tokens = 8U;
  constexpr std::uint32_t rows = 5U;
  constexpr std::uint32_t first_context_tokens = 4U;
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 4U;
  constexpr std::uint32_t grouped_heads = query_heads / kv_heads;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = 32U;
  constexpr std::uint32_t blocks = head_dim / 32U;
  constexpr std::uint32_t exponent_bytes =
      PerHead ? 0U : head_dim / 64U;
  constexpr std::uint32_t key_correction_bytes = PreserveKeyOutlier
      ? blocks * sizeof(std::uint32_t)
      : 0U;
  constexpr std::uint32_t key_record_bytes =
      head_dim / 2U + exponent_bytes + sizeof(std::uint16_t) +
      key_correction_bytes;
  constexpr std::uint32_t value_record_bytes =
      head_dim / 2U + exponent_bytes + sizeof(std::uint16_t);
  constexpr std::size_t records =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  constexpr std::size_t page_bytes =
      records * (key_record_bytes + value_record_bytes);
  constexpr std::size_t key_code_bytes = records * (head_dim / 2U);
  constexpr std::size_t key_exponent_bytes = records * exponent_bytes;
  constexpr std::size_t key_base_bytes =
      records * sizeof(std::uint16_t);
  constexpr std::size_t correction_bytes =
      records * key_correction_bytes;
  constexpr std::size_t value_code_bytes = records * (head_dim / 2U);
  constexpr std::size_t value_exponent_bytes = records * exponent_bytes;
  constexpr std::size_t key_exponent_offset = key_code_bytes;
  constexpr std::size_t key_base_offset =
      key_exponent_offset + key_exponent_bytes;
  constexpr std::size_t correction_offset =
      key_base_offset + key_base_bytes;
  constexpr std::size_t value_code_offset =
      correction_offset + correction_bytes;
  constexpr std::size_t value_exponent_offset =
      value_code_offset + value_code_bytes;
  constexpr std::size_t value_base_offset =
      value_exponent_offset + value_exponent_bytes;

  const auto kv_values = static_cast<std::size_t>(stored_tokens) *
                         kv_heads * head_dim;
  std::vector<float> key(kv_values);
  std::vector<float> value(kv_values);
  for (std::uint32_t token = 0U; token < stored_tokens; ++token) {
    for (std::uint32_t head = 0U; head < kv_heads; ++head) {
      const auto record = static_cast<std::size_t>(token) * kv_heads + head;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto index = record * head_dim + dimension;
        key[index] =
            std::sin(static_cast<float>(index + 11U) * 0.013F) *
            (0.08F + 0.006F * static_cast<float>(dimension % 17U));
        value[index] =
            std::cos(static_cast<float>(index + 19U) * 0.009F) *
            (0.07F + 0.005F * static_cast<float>(dimension % 23U));
      }
      for (std::uint32_t block = 0U; block < blocks; ++block) {
        const auto outlier =
            (token * 5U + head * 7U + block * 11U + 3U) % 32U;
        key[record * head_dim + block * 32U + outlier] =
            ((token + head + block) & 1U ? -1.0F : 1.0F) *
            (1.75F + 0.125F * static_cast<float>(block + head));
      }
    }
  }

  const auto query_values = static_cast<std::size_t>(rows) * query_heads *
                            2U * head_dim;
  std::vector<float> query(query_values);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto base =
          (static_cast<std::size_t>(row) * query_heads + head) * 2U *
          head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        query[base + dimension] =
            std::sin(static_cast<float>(base + dimension + 1U) * 0.005F) *
            (0.11F + 0.002F * static_cast<float>(dimension % 13U));
        query[base + head_dim + dimension] =
            std::cos(static_cast<float>(row + head + dimension + 1U) *
                     0.017F) *
            0.4F;
      }
    }
  }

  // This encoder is deliberately host-only and shares no decoder, scale
  // helper, or layout helper with the CUDA implementation under test.
  std::vector<std::uint8_t> expected_page(page_bytes, 0U);
  for (std::uint32_t token = 0U; token < stored_tokens; ++token) {
    for (std::uint32_t head = 0U; head < kv_heads; ++head) {
      const auto source_record =
          static_cast<std::size_t>(token) * kv_heads + head;
      const auto plane_record =
          static_cast<std::size_t>(head) * page_tokens + token;
      std::array<float, blocks> key_desired{};
      std::array<float, blocks> value_desired{};
      std::array<std::uint32_t, blocks> outliers{};
      float key_maximum_scale{};
      float value_maximum_scale{};
      for (std::uint32_t block = 0U; block < blocks; ++block) {
        float outlier_magnitude{};
        float value_maximum{};
        for (std::uint32_t offset = 0U; offset < 32U; ++offset) {
          const auto index = source_record * head_dim + block * 32U + offset;
          const auto magnitude = std::abs(key[index]);
          if (magnitude > outlier_magnitude) {
            outlier_magnitude = magnitude;
            outliers[block] = offset;
          }
          value_maximum = std::max(value_maximum, std::abs(value[index]));
        }
        float retained_maximum{};
        for (std::uint32_t offset = 0U; offset < 32U; ++offset) {
          if (offset == outliers[block]) continue;
          const auto index = source_record * head_dim + block * 32U + offset;
          retained_maximum =
              std::max(retained_maximum, std::abs(key[index]));
        }
        key_desired[block] =
            (PreserveKeyOutlier ? retained_maximum : outlier_magnitude) /
            7.0F;
        value_desired[block] = value_maximum / 7.0F;
        key_maximum_scale =
            std::max(key_maximum_scale, key_desired[block]);
        value_maximum_scale =
            std::max(value_maximum_scale, value_desired[block]);
      }
      const auto key_base = binary16_value(binary16_bits(std::max(
          PerHead ? key_maximum_scale : key_maximum_scale / 16.0F,
          std::ldexp(1.0F, -24))));
      const auto value_base = binary16_value(binary16_bits(std::max(
          PerHead ? value_maximum_scale : value_maximum_scale / 16.0F,
          std::ldexp(1.0F, -24))));
      const auto key_base_bits = binary16_bits(key_base);
      const auto value_base_bits = binary16_bits(value_base);
      std::memcpy(expected_page.data() + key_base_offset +
                      plane_record * sizeof(std::uint16_t),
                  &key_base_bits, sizeof(key_base_bits));
      std::memcpy(expected_page.data() + value_base_offset +
                      plane_record * sizeof(std::uint16_t),
                  &value_base_bits, sizeof(value_base_bits));

      std::array<std::uint8_t, blocks> key_exponents{};
      std::array<std::uint8_t, blocks> value_exponents{};
      for (std::uint32_t block = 0U; block < blocks; ++block) {
        if constexpr (!PerHead) {
          key_exponents[block] =
              host_q4_bfp_exponent(key_desired[block], key_base);
          value_exponents[block] =
              host_q4_bfp_exponent(value_desired[block], value_base);
        }
        if constexpr (PreserveKeyOutlier) {
          auto* correction = expected_page.data() + correction_offset +
              plane_record * key_correction_bytes +
              block * sizeof(std::uint32_t);
          correction[0] = static_cast<std::uint8_t>(outliers[block]);
          correction[1] = 0U;
          const auto outlier_bits = binary16_bits(
              key[source_record * head_dim + block * 32U + outliers[block]]);
          std::memcpy(correction + 2U, &outlier_bits, sizeof(outlier_bits));
        }
      }
      if constexpr (!PerHead) {
        for (std::uint32_t block = 0U; block < blocks; block += 2U) {
          expected_page[key_exponent_offset +
                        plane_record * exponent_bytes + block / 2U] =
              static_cast<std::uint8_t>(key_exponents[block] |
                                        (key_exponents[block + 1U] << 4U));
          expected_page[value_exponent_offset +
                        plane_record * exponent_bytes + block / 2U] =
              static_cast<std::uint8_t>(value_exponents[block] |
                                        (value_exponents[block + 1U] << 4U));
        }
      }
      for (std::uint32_t dimension = 0U; dimension < head_dim;
           dimension += 2U) {
        const auto block = dimension / 32U;
        const auto key_scale =
            std::ldexp(key_base, key_exponents[block]);
        const auto value_scale =
            std::ldexp(value_base, value_exponents[block]);
        const auto first_index = source_record * head_dim + dimension;
        const auto key_low =
            PreserveKeyOutlier && dimension % 32U == outliers[block]
                ? 0U
                : host_signed_q4(key[first_index], key_scale);
        const auto key_high =
            PreserveKeyOutlier &&
                    (dimension + 1U) % 32U == outliers[block]
                ? 0U
                : host_signed_q4(key[first_index + 1U], key_scale);
        const auto value_low = host_signed_q4(value[first_index], value_scale);
        const auto value_high =
            host_signed_q4(value[first_index + 1U], value_scale);
        expected_page[plane_record * (head_dim / 2U) + dimension / 2U] =
            static_cast<std::uint8_t>(key_low | (key_high << 4U));
        expected_page[value_code_offset +
                      plane_record * (head_dim / 2U) + dimension / 2U] =
            static_cast<std::uint8_t>(value_low | (value_high << 4U));
      }
    }
  }

  std::vector<std::int8_t> expected_q8(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  std::vector<float> expected_query_scales(
      static_cast<std::size_t>(rows) * query_heads);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto source =
          (static_cast<std::size_t>(row) * query_heads + head) * 2U *
          head_dim;
      const auto target =
          (static_cast<std::size_t>(row) * query_heads + head) * head_dim;
      float maximum{};
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
        maximum = std::max(maximum,
                           std::abs(query[source + dimension]));
      const auto query_scale =
          std::max(maximum / 127.0F, std::ldexp(1.0F, -24));
      expected_query_scales[static_cast<std::size_t>(row) * query_heads +
                            head] = query_scale;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto quantized = std::max(
            -127, std::min(127, static_cast<int>(std::nearbyint(
                               query[source + dimension] / query_scale))));
        expected_q8[target + dimension] =
            static_cast<std::int8_t>(quantized);
      }
    }
  }

  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<std::uint8_t> device_page(page_bytes);
  DeviceBuffer<void*> device_page_table(1U);
  DeviceBuffer<std::int8_t> device_q8(expected_q8.size());
  DeviceBuffer<float> device_query_scales(expected_query_scales.size());
  DeviceBuffer<float> device_output(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  DeviceBuffer<float> partial_maxima(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> partial_sums(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> partial_outputs(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  device_key.upload(key);
  device_value.upload(value);
  device_query.upload(query);
  cuda_check(cudaMemset(device_page.get(), 0, page_bytes),
             "initialize Q4 BFP numerical page");
  device_page_table.upload(std::vector<void*>{device_page.get()});
  if constexpr (PerHead)
    status_check(expert::runtime::cuda::store_gqa_kv_paged_q4_per_head_batch(
        device_key.get(), device_value.get(),
        reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
        page_tokens, 0U, stored_tokens, kv_heads, head_dim, nullptr));
  else if constexpr (PreserveKeyOutlier)
    status_check(
        expert::runtime::cuda::store_gqa_kv_paged_q4_bfp_key_outlier1_batch(
            device_key.get(), device_value.get(),
            reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
            page_tokens, 0U, stored_tokens, kv_heads, head_dim, nullptr));
  else
    status_check(expert::runtime::cuda::store_gqa_kv_paged_q4_bfp_batch(
        device_key.get(), device_value.get(),
        reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
        page_tokens, 0U, stored_tokens, kv_heads, head_dim, nullptr));
  status_check(expert::runtime::cuda::quantize_gqa_queries_q8(
      device_query.get(), device_q8.get(), device_query_scales.get(), rows,
      query_heads, head_dim, nullptr));
  const expert::runtime::cuda::PagedQ4BfpGatedGqaAttentionLaunch launch{
      device_query.get(), device_q8.get(), device_query_scales.get(),
      reinterpret_cast<const void* const*>(device_page_table.get()),
      device_output.get(), partial_maxima.get(), partial_sums.get(),
      partial_outputs.get(), first_context_tokens, rows, 0U, page_tokens,
      query_heads, kv_heads, head_dim, page_tokens, 1U, nullptr};
  if constexpr (PerHead)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_microbatch_paged_q4_per_head_tensor_core(
                         launch));
  else if constexpr (PreserveKeyOutlier)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_microbatch_paged_q4_bfp_key_outlier1_tensor_core(
                         launch));
  else
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_microbatch_paged_q4_bfp_tensor_core(
                         launch));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize Q4 BFP five-query numerical check");

  Q4BfpKeyOutlier1Check result;
  const auto actual_page = device_page.download();
  for (std::size_t index = 0U; index < expected_page.size(); ++index)
    result.layout_maximum_byte_difference = std::max(
        result.layout_maximum_byte_difference,
        static_cast<double>(std::abs(static_cast<int>(actual_page[index]) -
                                     expected_page[index])));
  const auto actual_q8 = device_q8.download();
  for (std::size_t index = 0U; index < expected_q8.size(); ++index)
    result.query_maximum_byte_difference = std::max(
        result.query_maximum_byte_difference,
        static_cast<double>(std::abs(static_cast<int>(actual_q8[index]) -
                                     expected_q8[index])));
  const auto actual_query_scales = device_query_scales.download();
  for (std::size_t index = 0U; index < expected_query_scales.size(); ++index)
    result.query_scale_maximum_absolute_difference = std::max(
        result.query_scale_maximum_absolute_difference,
        std::abs(static_cast<double>(actual_query_scales[index]) -
                 expected_query_scales[index]));

  const auto packed_nibble = [](const std::uint8_t* codes,
                                std::size_t record,
                                std::uint32_t dimension) {
    const auto packed =
        codes[record * (head_dim / 2U) + dimension / 2U];
    return static_cast<std::uint8_t>(
        (dimension & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
  };
  const auto exponent = [](const std::uint8_t* exponents,
                           std::size_t record, std::uint32_t block) {
    if constexpr (PerHead) return static_cast<std::uint8_t>(0U);
    const auto packed =
        exponents[record * exponent_bytes + block / 2U];
    return static_cast<std::uint8_t>(
        (block & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
  };
  const auto* key_codes = expected_page.data();
  const auto* key_exponents = expected_page.data() + key_exponent_offset;
  const auto* key_bases = expected_page.data() + key_base_offset;
  const auto* corrections = expected_page.data() + correction_offset;
  const auto* value_codes = expected_page.data() + value_code_offset;
  const auto* value_exponents =
      expected_page.data() + value_exponent_offset;
  const auto* value_bases = expected_page.data() + value_base_offset;
  const auto half_at = [](const std::uint8_t* bytes, std::size_t offset) {
    std::uint16_t bits{};
    std::memcpy(&bits, bytes + offset, sizeof(bits));
    return binary16_value(bits);
  };
  const auto actual_output = device_output.download();
  const auto score_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const auto context_tokens = first_context_tokens + row;
    for (std::uint32_t query_head = 0U; query_head < query_heads;
         ++query_head) {
      const auto kv_head = query_head / grouped_heads;
      const auto query_record =
          static_cast<std::size_t>(row) * query_heads + query_head;
      std::vector<float> scores(context_tokens);
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::uint32_t token = 0U; token < context_tokens; ++token) {
        const auto record =
            static_cast<std::size_t>(kv_head) * page_tokens + token;
        const auto key_base =
            half_at(key_bases, record * sizeof(std::uint16_t));
        std::int32_t dot{};
        float correction_dot{};
        for (std::uint32_t dimension = 0U; dimension < head_dim;
             ++dimension) {
          const auto block = dimension / 32U;
          const auto expanded =
              host_signed_q4_value(
                  packed_nibble(key_codes, record, dimension)) *
              (1 << exponent(key_exponents, record, block));
          dot += static_cast<std::int32_t>(
                     expected_q8[query_record * head_dim + dimension]) *
                 expanded;
        }
        if constexpr (PreserveKeyOutlier) {
          for (std::uint32_t block = 0U; block < blocks; ++block) {
            const auto* correction = corrections +
                record * key_correction_bytes +
                block * sizeof(std::uint32_t);
            const auto dimension = block * 32U + correction[0];
            correction_dot +=
                static_cast<float>(
                    expected_q8[query_record * head_dim + dimension]) *
                expected_query_scales[query_record] *
                half_at(correction, 2U);
          }
        }
        scores[token] =
            (static_cast<float>(dot) * expected_query_scales[query_record] *
                 key_base +
             correction_dot) *
            score_scale;
        maximum = std::max(maximum, scores[token]);
      }
      std::vector<float> probabilities(context_tokens);
      float denominator{};
      float weighted_maximum{};
      for (std::uint32_t token = 0U; token < context_tokens; ++token) {
        probabilities[token] = std::exp(scores[token] - maximum);
        denominator += probabilities[token];
        const auto record =
            static_cast<std::size_t>(kv_head) * page_tokens + token;
        const auto value_base =
            half_at(value_bases, record * sizeof(std::uint16_t));
        weighted_maximum = std::max(
            weighted_maximum, probabilities[token] * value_base);
      }
      const auto probability_scale =
          std::max(weighted_maximum / 127.0F, std::ldexp(1.0F, -24));
      std::vector<std::int8_t> quantized_probabilities(context_tokens);
      for (std::uint32_t token = 0U; token < context_tokens; ++token) {
        const auto record =
            static_cast<std::size_t>(kv_head) * page_tokens + token;
        const auto value_base =
            half_at(value_bases, record * sizeof(std::uint16_t));
        const auto quantized = std::max(
            0, std::min(127, static_cast<int>(std::nearbyint(
                                probabilities[token] * value_base /
                                probability_scale))));
        quantized_probabilities[token] =
            static_cast<std::int8_t>(quantized);
      }
      const auto gate_base = query_record * 2U * head_dim + head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim;
           ++dimension) {
        std::int32_t product{};
        for (std::uint32_t token = 0U; token < context_tokens; ++token) {
          const auto record =
              static_cast<std::size_t>(kv_head) * page_tokens + token;
          const auto block = dimension / 32U;
          const auto expanded =
              host_signed_q4_value(
                  packed_nibble(value_codes, record, dimension)) *
              (1 << exponent(value_exponents, record, block));
          product += static_cast<std::int32_t>(
                         quantized_probabilities[token]) *
                     expanded;
        }
        const auto expected =
            (static_cast<float>(product) * probability_scale / denominator) /
            (1.0F + std::exp(-query[gate_base + dimension]));
        const auto output_index = query_record * head_dim + dimension;
        result.independent_oracle_maximum_absolute_difference = std::max(
            result.independent_oracle_maximum_absolute_difference,
            std::abs(static_cast<double>(actual_output[output_index]) -
                     expected));
      }
    }
  }
  return result;
}

Q4BfpKeyOutlier1Check q4_bfp_key_outlier1_check() {
  return q4_bfp_check_impl<true>();
}

Q4BfpKeyOutlier1Check q4_bfp_check() {
  return q4_bfp_check_impl<false>();
}

Q4BfpKeyOutlier1Check q4_per_head_check() {
  return q4_bfp_check_impl<false, true>();
}

struct Q5Q4BfpCheck final {
  double layout_maximum_byte_difference{};
  double query_maximum_byte_difference{};
  double query_scale_maximum_absolute_difference{};
  double independent_oracle_maximum_absolute_difference{};
};

Q5Q4BfpCheck q5_q4_bfp_check() {
  constexpr std::uint32_t stored_tokens = 8U;
  constexpr std::uint32_t rows = 5U;
  constexpr std::uint32_t first_context_tokens = 4U;
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 4U;
  constexpr std::uint32_t grouped_heads = query_heads / kv_heads;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = 32U;
  constexpr std::uint32_t blocks = head_dim / 32U;
  constexpr std::uint32_t exponent_bytes = head_dim / 64U;
  constexpr std::uint32_t key_code_bytes = head_dim * 5U / 8U;
  constexpr std::uint32_t value_code_bytes = head_dim / 2U;
  constexpr std::uint32_t key_record_bytes =
      key_code_bytes + exponent_bytes + sizeof(std::uint16_t);
  constexpr std::uint32_t value_record_bytes =
      value_code_bytes + exponent_bytes + sizeof(std::uint16_t);
  constexpr std::size_t records =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  constexpr std::size_t page_bytes =
      records * (key_record_bytes + value_record_bytes);
  constexpr std::size_t key_exponent_offset = records * key_code_bytes;
  constexpr std::size_t key_base_offset =
      key_exponent_offset + records * exponent_bytes;
  constexpr std::size_t value_code_offset =
      key_base_offset + records * sizeof(std::uint16_t);
  constexpr std::size_t value_exponent_offset =
      value_code_offset + records * value_code_bytes;
  constexpr std::size_t value_base_offset =
      value_exponent_offset + records * exponent_bytes;

  const auto kv_values = static_cast<std::size_t>(stored_tokens) *
                         kv_heads * head_dim;
  std::vector<float> key(kv_values);
  std::vector<float> value(kv_values);
  for (std::uint32_t token = 0U; token < stored_tokens; ++token) {
    for (std::uint32_t head = 0U; head < kv_heads; ++head) {
      const auto record = static_cast<std::size_t>(token) * kv_heads + head;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto index = record * head_dim + dimension;
        key[index] =
            std::sin(static_cast<float>(index + 11U) * 0.013F) *
            (0.08F + 0.006F * static_cast<float>(dimension % 17U));
        value[index] =
            std::cos(static_cast<float>(index + 19U) * 0.009F) *
            (0.07F + 0.005F * static_cast<float>(dimension % 23U));
      }
      for (std::uint32_t block = 0U; block < blocks; ++block) {
        const auto outlier =
            (token * 5U + head * 7U + block * 11U + 3U) % 32U;
        key[record * head_dim + block * 32U + outlier] =
            ((token + head + block) & 1U ? -1.0F : 1.0F) *
            (1.75F + 0.125F * static_cast<float>(block + head));
      }
    }
  }

  const auto query_values = static_cast<std::size_t>(rows) * query_heads *
                            2U * head_dim;
  std::vector<float> query(query_values);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto base =
          (static_cast<std::size_t>(row) * query_heads + head) * 2U *
          head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        query[base + dimension] =
            std::sin(static_cast<float>(base + dimension + 1U) * 0.005F) *
            (0.11F + 0.002F * static_cast<float>(dimension % 13U));
        query[base + head_dim + dimension] =
            std::cos(static_cast<float>(row + head + dimension + 1U) *
                     0.017F) *
            0.4F;
      }
    }
  }

  // This host encoder intentionally shares no codec or layout helper with
  // the CUDA implementation under test.
  std::vector<std::uint8_t> expected_page(page_bytes, 0U);
  for (std::uint32_t token = 0U; token < stored_tokens; ++token) {
    for (std::uint32_t head = 0U; head < kv_heads; ++head) {
      const auto source_record =
          static_cast<std::size_t>(token) * kv_heads + head;
      const auto plane_record =
          static_cast<std::size_t>(head) * page_tokens + token;
      std::array<float, blocks> key_desired{};
      std::array<float, blocks> value_desired{};
      float key_maximum_scale{};
      float value_maximum_scale{};
      for (std::uint32_t block = 0U; block < blocks; ++block) {
        float key_maximum{};
        float value_maximum{};
        for (std::uint32_t offset = 0U; offset < 32U; ++offset) {
          const auto index = source_record * head_dim + block * 32U + offset;
          key_maximum = std::max(key_maximum, std::abs(key[index]));
          value_maximum = std::max(value_maximum, std::abs(value[index]));
        }
        key_desired[block] = key_maximum / 15.0F;
        value_desired[block] = value_maximum / 7.0F;
        key_maximum_scale =
            std::max(key_maximum_scale, key_desired[block]);
        value_maximum_scale =
            std::max(value_maximum_scale, value_desired[block]);
      }
      const auto key_base = binary16_value(binary16_bits(
          std::max(key_maximum_scale / 8.0F, std::ldexp(1.0F, -24))));
      const auto value_base = binary16_value(binary16_bits(
          std::max(value_maximum_scale / 16.0F, std::ldexp(1.0F, -24))));
      const auto key_base_bits = binary16_bits(key_base);
      const auto value_base_bits = binary16_bits(value_base);
      std::memcpy(expected_page.data() + key_base_offset +
                      plane_record * sizeof(std::uint16_t),
                  &key_base_bits, sizeof(key_base_bits));
      std::memcpy(expected_page.data() + value_base_offset +
                      plane_record * sizeof(std::uint16_t),
                  &value_base_bits, sizeof(value_base_bits));

      std::array<std::uint8_t, blocks> key_exponents{};
      std::array<std::uint8_t, blocks> value_exponents{};
      for (std::uint32_t block = 0U; block < blocks; ++block) {
        key_exponents[block] =
            host_q5_bfp_exponent(key_desired[block], key_base);
        value_exponents[block] =
            host_q4_bfp_exponent(value_desired[block], value_base);
      }
      for (std::uint32_t block = 0U; block < blocks; block += 2U) {
        expected_page[key_exponent_offset +
                      plane_record * exponent_bytes + block / 2U] =
            static_cast<std::uint8_t>(key_exponents[block] |
                                      (key_exponents[block + 1U] << 4U));
        expected_page[value_exponent_offset +
                      plane_record * exponent_bytes + block / 2U] =
            static_cast<std::uint8_t>(value_exponents[block] |
                                      (value_exponents[block + 1U] << 4U));
      }
      for (std::uint32_t group = 0U; group < head_dim / 8U; ++group) {
        const auto first = group * 8U;
        const auto block = first / 32U;
        const auto key_scale = std::ldexp(key_base, key_exponents[block]);
        const auto value_scale =
            std::ldexp(value_base, value_exponents[block]);
        std::uint32_t packed_key_magnitudes{};
        std::uint8_t packed_key_signs{};
        std::uint32_t packed_value{};
        for (std::uint32_t index = 0U; index < 8U; ++index) {
          const auto source = source_record * head_dim + first + index;
          const auto key_code = host_signed_q5_value(
              host_signed_q5(key[source], key_scale));
          packed_key_magnitudes |=
              static_cast<std::uint32_t>(std::abs(key_code)) <<
              (4U * index);
          if (key_code < 0)
            packed_key_signs |= static_cast<std::uint8_t>(1U << index);
          packed_value |= static_cast<std::uint32_t>(
                              host_signed_q4(value[source], value_scale))
                          << (4U * index);
        }
        std::memcpy(expected_page.data() +
                        plane_record * (head_dim / 2U) +
                        group * sizeof(std::uint32_t),
                    &packed_key_magnitudes,
                    sizeof(packed_key_magnitudes));
        expected_page[records * (head_dim / 2U) +
                      plane_record * (head_dim / 8U) + group] =
            packed_key_signs;
        std::memcpy(expected_page.data() + value_code_offset +
                        plane_record * value_code_bytes +
                        group * sizeof(std::uint32_t),
                    &packed_value, sizeof(packed_value));
      }
    }
  }

  std::vector<std::int8_t> expected_q8(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  std::vector<float> expected_query_scales(
      static_cast<std::size_t>(rows) * query_heads);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto source =
          (static_cast<std::size_t>(row) * query_heads + head) * 2U *
          head_dim;
      const auto target =
          (static_cast<std::size_t>(row) * query_heads + head) * head_dim;
      float maximum{};
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
        maximum = std::max(maximum, std::abs(query[source + dimension]));
      const auto scale =
          std::max(maximum / 127.0F, std::ldexp(1.0F, -24));
      expected_query_scales[static_cast<std::size_t>(row) * query_heads +
                            head] = scale;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto quantized = std::max(
            -127, std::min(127, static_cast<int>(std::nearbyint(
                               query[source + dimension] / scale))));
        expected_q8[target + dimension] =
            static_cast<std::int8_t>(quantized);
      }
    }
  }

  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<std::uint8_t> device_page(page_bytes);
  DeviceBuffer<void*> device_page_table(1U);
  DeviceBuffer<std::int8_t> device_q8(expected_q8.size());
  DeviceBuffer<float> device_query_scales(expected_query_scales.size());
  DeviceBuffer<float> device_output(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  DeviceBuffer<float> partial_maxima(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> partial_sums(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> partial_outputs(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  device_key.upload(key);
  device_value.upload(value);
  device_query.upload(query);
  cuda_check(cudaMemset(device_page.get(), 0, page_bytes),
             "initialize Q5-K/Q4-V BFP numerical page");
  device_page_table.upload(std::vector<void*>{device_page.get()});
  status_check(expert::runtime::cuda::store_gqa_kv_paged_q5_q4_bfp_batch(
      device_key.get(), device_value.get(),
      reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
      page_tokens, 0U, stored_tokens, kv_heads, head_dim, nullptr));
  status_check(expert::runtime::cuda::quantize_gqa_queries_q8(
      device_query.get(), device_q8.get(), device_query_scales.get(), rows,
      query_heads, head_dim, nullptr));
  status_check(expert::runtime::cuda::
                   gated_gqa_attention_microbatch_paged_q5_q4_bfp_tensor_core(
                       {device_query.get(), device_q8.get(),
                        device_query_scales.get(),
                        reinterpret_cast<const void* const*>(
                            device_page_table.get()),
                        device_output.get(), partial_maxima.get(),
                        partial_sums.get(), partial_outputs.get(),
                        first_context_tokens, rows, 0U, page_tokens,
                        query_heads, kv_heads, head_dim, page_tokens, 1U,
                        nullptr}));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize Q5-K/Q4-V BFP five-query numerical check");

  Q5Q4BfpCheck result;
  const auto actual_page = device_page.download();
  for (std::size_t index = 0U; index < expected_page.size(); ++index)
    result.layout_maximum_byte_difference = std::max(
        result.layout_maximum_byte_difference,
        static_cast<double>(std::abs(static_cast<int>(actual_page[index]) -
                                     expected_page[index])));
  const auto actual_q8 = device_q8.download();
  for (std::size_t index = 0U; index < expected_q8.size(); ++index)
    result.query_maximum_byte_difference = std::max(
        result.query_maximum_byte_difference,
        static_cast<double>(std::abs(static_cast<int>(actual_q8[index]) -
                                     expected_q8[index])));
  const auto actual_query_scales = device_query_scales.download();
  for (std::size_t index = 0U; index < expected_query_scales.size(); ++index)
    result.query_scale_maximum_absolute_difference = std::max(
        result.query_scale_maximum_absolute_difference,
        std::abs(static_cast<double>(actual_query_scales[index]) -
                 expected_query_scales[index]));

  const auto exponent = [](const std::uint8_t* exponents,
                           std::size_t record, std::uint32_t block) {
    const auto packed = exponents[record * exponent_bytes + block / 2U];
    return static_cast<std::uint8_t>(
        (block & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
  };
  const auto q5_value = [](const std::uint8_t* codes, std::size_t record,
                           std::uint32_t dimension) {
    const auto group = dimension / 8U;
    const auto index = dimension % 8U;
    const auto packed = codes[record * (head_dim / 2U) + dimension / 2U];
    const auto magnitude = static_cast<std::int32_t>(
        (dimension & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
    const auto signs = codes[records * (head_dim / 2U) +
                             record * (head_dim / 8U) + group];
    return (signs & (1U << index)) != 0U ? -magnitude : magnitude;
  };
  const auto q4_code = [](const std::uint8_t* codes, std::size_t record,
                          std::uint32_t dimension) {
    const auto packed =
        codes[record * value_code_bytes + dimension / 2U];
    return static_cast<std::uint8_t>(
        (dimension & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
  };
  const auto half_at = [](const std::uint8_t* bytes, std::size_t offset) {
    std::uint16_t bits{};
    std::memcpy(&bits, bytes + offset, sizeof(bits));
    return binary16_value(bits);
  };
  const auto* key_codes = expected_page.data();
  const auto* key_exponents = expected_page.data() + key_exponent_offset;
  const auto* key_bases = expected_page.data() + key_base_offset;
  const auto* value_codes = expected_page.data() + value_code_offset;
  const auto* value_exponents =
      expected_page.data() + value_exponent_offset;
  const auto* value_bases = expected_page.data() + value_base_offset;
  const auto actual_output = device_output.download();
  const auto score_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const auto context_tokens = first_context_tokens + row;
    for (std::uint32_t query_head = 0U; query_head < query_heads;
         ++query_head) {
      const auto kv_head = query_head / grouped_heads;
      const auto query_record =
          static_cast<std::size_t>(row) * query_heads + query_head;
      std::vector<float> scores(context_tokens);
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::uint32_t token = 0U; token < context_tokens; ++token) {
        const auto record =
            static_cast<std::size_t>(kv_head) * page_tokens + token;
        const auto key_base =
            half_at(key_bases, record * sizeof(std::uint16_t));
        std::int32_t dot{};
        for (std::uint32_t dimension = 0U; dimension < head_dim;
             ++dimension) {
          const auto block = dimension / 32U;
          const auto expanded = q5_value(key_codes, record, dimension) *
              (1 << exponent(key_exponents, record, block));
          dot += static_cast<std::int32_t>(
                     expected_q8[query_record * head_dim + dimension]) *
                 expanded;
        }
        scores[token] =
            static_cast<float>(dot) * expected_query_scales[query_record] *
            key_base * score_scale;
        maximum = std::max(maximum, scores[token]);
      }
      std::vector<float> probabilities(context_tokens);
      float denominator{};
      float weighted_maximum{};
      for (std::uint32_t token = 0U; token < context_tokens; ++token) {
        probabilities[token] = std::exp(scores[token] - maximum);
        denominator += probabilities[token];
        const auto record =
            static_cast<std::size_t>(kv_head) * page_tokens + token;
        const auto value_base =
            half_at(value_bases, record * sizeof(std::uint16_t));
        weighted_maximum = std::max(
            weighted_maximum, probabilities[token] * value_base);
      }
      const auto probability_scale =
          std::max(weighted_maximum / 127.0F, std::ldexp(1.0F, -24));
      std::vector<std::int8_t> quantized_probabilities(context_tokens);
      for (std::uint32_t token = 0U; token < context_tokens; ++token) {
        const auto record =
            static_cast<std::size_t>(kv_head) * page_tokens + token;
        const auto value_base =
            half_at(value_bases, record * sizeof(std::uint16_t));
        quantized_probabilities[token] = static_cast<std::int8_t>(std::max(
            0, std::min(127, static_cast<int>(std::nearbyint(
                                probabilities[token] * value_base /
                                probability_scale)))));
      }
      const auto gate_base = query_record * 2U * head_dim + head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim;
           ++dimension) {
        std::int32_t product{};
        for (std::uint32_t token = 0U; token < context_tokens; ++token) {
          const auto record =
              static_cast<std::size_t>(kv_head) * page_tokens + token;
          const auto block = dimension / 32U;
          const auto expanded = host_signed_q4_value(
              q4_code(value_codes, record, dimension)) *
              (1 << exponent(value_exponents, record, block));
          product += static_cast<std::int32_t>(
                         quantized_probabilities[token]) *
                     expanded;
        }
        const auto expected =
            (static_cast<float>(product) * probability_scale / denominator) /
            (1.0F + std::exp(-query[gate_base + dimension]));
        const auto output_index = query_record * head_dim + dimension;
        result.independent_oracle_maximum_absolute_difference = std::max(
            result.independent_oracle_maximum_absolute_difference,
            std::abs(static_cast<double>(actual_output[output_index]) -
                     expected));
      }
    }
  }
  return result;
}

struct Q4BfpAttentionProfile final {
  std::uint32_t context_tokens{};
  double milliseconds{};
  double sixteen_layer_milliseconds{};
  double kv_gb_per_second{};
};

enum class PackedBfpProfileEncoding : std::uint8_t {
  q4_key_outlier1,
  q4,
  q4_per_head,
  q5_q4,
};

Q4BfpAttentionProfile packed_bfp_attention_profile(
    std::uint32_t context_tokens, PackedBfpProfileEncoding encoding) {
  constexpr std::uint32_t rows = 5U;
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 4U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = 256U;
  if (context_tokens < rows || context_tokens % page_tokens)
    throw std::runtime_error("invalid Q4 BFP profile context");
  const auto split_tokens = context_tokens <= 65536U ? 512U : 1024U;
  const auto maximum_splits =
      (context_tokens + split_tokens - 1U) / split_tokens;
  const auto pages = context_tokens / page_tokens;
  const auto per_head =
      encoding == PackedBfpProfileEncoding::q4_per_head;
  const auto key_record_bytes =
      per_head ? 130U
               : encoding == PackedBfpProfileEncoding::q4 ? 134U : 166U;
  const auto value_record_bytes = per_head ? 130U : 134U;
  const auto page_bytes = static_cast<std::size_t>(page_tokens) * kv_heads *
                          (key_record_bytes + value_record_bytes);
  const auto payload_bytes = static_cast<std::size_t>(pages) * page_bytes;
  const auto query_values = static_cast<std::size_t>(rows) * query_heads *
                            2U * head_dim;
  const auto output_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  const auto partials = static_cast<std::size_t>(rows) * maximum_splits *
                        query_heads;

  DeviceBuffer<float> query(query_values);
  DeviceBuffer<std::int8_t> q8(
      static_cast<std::size_t>(rows) * query_heads * head_dim);
  DeviceBuffer<float> query_scales(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> output(output_values);
  DeviceBuffer<float> partial_maxima(partials);
  DeviceBuffer<float> partial_sums(partials);
  DeviceBuffer<float> partial_outputs(partials * head_dim);
  DeviceBuffer<std::uint8_t> page_storage(payload_bytes);
  DeviceBuffer<void*> page_table(pages);
  std::vector<float> host_query(query_values);
  for (std::size_t index = 0U; index < host_query.size(); ++index)
    host_query[index] =
        std::sin(static_cast<float>(index + 1U) * 0.001F) * 0.1F;
  query.upload(host_query);
  cuda_check(cudaMemset(page_storage.get(), 0, payload_bytes),
             "initialize Q4 BFP profile pages");
  std::vector<void*> host_pages(pages);
  for (std::uint32_t page = 0U; page < pages; ++page)
    host_pages[page] = page_storage.get() +
                       static_cast<std::size_t>(page) * page_bytes;
  page_table.upload(host_pages);
  status_check(expert::runtime::cuda::quantize_gqa_queries_q8(
      query.get(), q8.get(), query_scales.get(), rows, query_heads,
      head_dim, nullptr));
  const expert::runtime::cuda::PagedQ4BfpGatedGqaAttentionLaunch launch{
      query.get(), q8.get(), query_scales.get(),
      reinterpret_cast<const void* const*>(page_table.get()), output.get(),
      partial_maxima.get(), partial_sums.get(), partial_outputs.get(),
      context_tokens - rows + 1U, rows, 0U, page_tokens, query_heads,
      kv_heads, head_dim, split_tokens, maximum_splits, nullptr};
  const auto run_attention = [&]() {
    if (encoding == PackedBfpProfileEncoding::q5_q4)
      status_check(expert::runtime::cuda::
                       gated_gqa_attention_microbatch_paged_q5_q4_bfp_tensor_core(
                           launch));
    else if (encoding == PackedBfpProfileEncoding::q4_per_head)
      status_check(expert::runtime::cuda::
                       gated_gqa_attention_microbatch_paged_q4_per_head_tensor_core(
                           launch));
    else if (encoding == PackedBfpProfileEncoding::q4)
      status_check(expert::runtime::cuda::
                       gated_gqa_attention_microbatch_paged_q4_bfp_tensor_core(
                           launch));
    else
      status_check(expert::runtime::cuda::
                       gated_gqa_attention_microbatch_paged_q4_bfp_key_outlier1_tensor_core(
                           launch));
  };
  for (unsigned warmup = 0U; warmup < 2U; ++warmup) {
    run_attention();
  }
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "create Q4 BFP profile start event");
  cuda_check(cudaEventCreate(&stop), "create Q4 BFP profile stop event");
  cuda_check(cudaEventRecord(start), "record Q4 BFP profile start");
  constexpr unsigned iterations = 8U;
  for (unsigned iteration = 0U; iteration < iterations; ++iteration)
    run_attention();
  cuda_check(cudaEventRecord(stop), "record Q4 BFP profile stop");
  cuda_check(cudaEventSynchronize(stop), "synchronize Q4 BFP profile stop");
  float total_milliseconds{};
  cuda_check(cudaEventElapsedTime(&total_milliseconds, start, stop),
             "measure Q4 BFP profile elapsed time");
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  const auto milliseconds =
      static_cast<double>(total_milliseconds) / iterations;
  return {context_tokens, milliseconds, 16.0 * milliseconds,
          static_cast<double>(payload_bytes) / (milliseconds * 1.0e6)};
}

Q4BfpAttentionProfile q4_bfp_attention_profile(
    std::uint32_t context_tokens) {
  return packed_bfp_attention_profile(
      context_tokens, PackedBfpProfileEncoding::q4_key_outlier1);
}

Q4BfpAttentionProfile q4_bfp_plain_attention_profile(
    std::uint32_t context_tokens) {
  return packed_bfp_attention_profile(
      context_tokens, PackedBfpProfileEncoding::q4);
}

Q4BfpAttentionProfile q4_per_head_attention_profile(
    std::uint32_t context_tokens) {
  return packed_bfp_attention_profile(
      context_tokens, PackedBfpProfileEncoding::q4_per_head);
}

Q4BfpAttentionProfile q5_q4_bfp_attention_profile(
    std::uint32_t context_tokens) {
  return packed_bfp_attention_profile(
      context_tokens, PackedBfpProfileEncoding::q5_q4);
}

double bandwidth_check() {
  constexpr std::uint32_t rows = 17408U;
  constexpr std::uint32_t columns = 5120U;
  constexpr std::uint32_t iterations = 12U;
  const auto weight_bytes = static_cast<std::size_t>(rows) * columns / 2U;
  const auto scale_bytes = static_cast<std::size_t>(rows) * columns / 32U;
  std::vector<std::uint8_t> weights(weight_bytes, 0x21U);
  std::vector<std::uint8_t> scales(scale_bytes, 127U);
  std::vector<float> input(columns);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.1F;
  DeviceBuffer<std::uint8_t> device_weights(weight_bytes);
  DeviceBuffer<std::uint8_t> device_scales(scale_bytes);
  DeviceBuffer<float> device_input(columns);
  DeviceBuffer<std::int8_t> device_q8(columns);
  DeviceBuffer<float> device_q8_scale(1U);
  DeviceBuffer<float> device_output(rows);
  device_weights.upload(weights);
  device_scales.upload(scales);
  device_input.upload(input);
  status_check(expert::runtime::cuda::quantize_q8_batch(
      device_input.get(), device_q8.get(), device_q8_scale.get(), 1U,
      columns, columns, nullptr));
  for (unsigned warmup = 0; warmup < 3U; ++warmup)
    status_check(expert::runtime::cuda::fp4_gemv_q8_batch(
        {device_weights.get(), device_scales.get(), rows, columns, columns},
        device_q8.get(), device_q8_scale.get(), device_output.get(), 1U,
        nullptr));
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "create dense FP4 start event");
  cuda_check(cudaEventCreate(&stop), "create dense FP4 stop event");
  cuda_check(cudaEventRecord(start), "record dense FP4 start");
  for (unsigned iteration = 0; iteration < iterations; ++iteration)
    status_check(expert::runtime::cuda::fp4_gemv_q8_batch(
        {device_weights.get(), device_scales.get(), rows, columns, columns},
        device_q8.get(), device_q8_scale.get(), device_output.get(), 1U,
        nullptr));
  cuda_check(cudaEventRecord(stop), "record dense FP4 stop");
  cuda_check(cudaEventSynchronize(stop), "synchronize dense FP4 stop");
  float milliseconds{};
  cuda_check(cudaEventElapsedTime(&milliseconds, start, stop),
             "measure dense FP4 elapsed time");
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  const auto bytes = static_cast<double>(weight_bytes + scale_bytes) *
                     iterations;
  return bytes / (static_cast<double>(milliseconds) * 1.0e6);
}

struct PrefillGemmProfile final {
  double current_effective_weight_gb_per_second{};
  double current_milliseconds{};
  double staged_decode_milliseconds{};
  double staged_gemm_milliseconds{};
};

PrefillGemmProfile batch_bandwidth_check() {
  constexpr std::uint32_t rows = 17408U;
  constexpr std::uint32_t columns = 5120U;
  constexpr std::uint32_t batch = 512U;
  constexpr std::uint32_t iterations = 12U;
  const auto weight_bytes = static_cast<std::size_t>(rows) * columns / 2U;
  const auto scale_bytes = static_cast<std::size_t>(rows) * columns / 32U;
  std::vector<std::uint8_t> weights(weight_bytes, 0x21U);
  std::vector<std::uint8_t> scales(scale_bytes, 127U);
  std::vector<float> input(static_cast<std::size_t>(batch) * columns);
  for (std::size_t index = 0U; index < input.size(); ++index)
    input[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.1F;
  DeviceBuffer<std::uint8_t> device_weights(weight_bytes);
  DeviceBuffer<std::uint8_t> device_scales(scale_bytes);
  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::int8_t> device_q8(
      static_cast<std::size_t>(batch) * columns);
  DeviceBuffer<float> device_q8_scales(batch);
  DeviceBuffer<float> device_output(static_cast<std::size_t>(batch) * rows);
  DeviceBuffer<std::uint16_t> decoded_weights(
      static_cast<std::size_t>(rows) * columns);
  DeviceBuffer<std::uint16_t> decoded_input(
      static_cast<std::size_t>(batch) * columns);
  device_weights.upload(weights);
  device_scales.upload(scales);
  device_input.upload(input);
  status_check(expert::runtime::cuda::quantize_q8_batch(
      device_input.get(), device_q8.get(), device_q8_scales.get(), batch,
      columns, columns, nullptr));
  for (unsigned warmup = 0U; warmup < 3U; ++warmup)
    status_check(expert::runtime::cuda::fp4_gemm_q8_block32(
        {device_weights.get(), device_scales.get(), rows, columns, columns},
        device_q8.get(), device_q8_scales.get(), device_output.get(), batch,
        nullptr));
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "create batched FP4 start event");
  cuda_check(cudaEventCreate(&stop), "create batched FP4 stop event");
  cuda_check(cudaEventRecord(start), "record batched FP4 start");
  for (unsigned iteration = 0U; iteration < iterations; ++iteration)
    status_check(expert::runtime::cuda::fp4_gemm_q8_block32(
        {device_weights.get(), device_scales.get(), rows, columns, columns},
        device_q8.get(), device_q8_scales.get(), device_output.get(), batch,
        nullptr));
  cuda_check(cudaEventRecord(stop), "record batched FP4 stop");
  cuda_check(cudaEventSynchronize(stop), "synchronize batched FP4 stop");
  float milliseconds{};
  cuda_check(cudaEventElapsedTime(&milliseconds, start, stop),
             "measure batched FP4 elapsed time");
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  const auto bytes = static_cast<double>(weight_bytes + scale_bytes) *
                     iterations;
  const auto current_milliseconds =
      static_cast<double>(milliseconds) / iterations;

  const expert::runtime::cuda::Fp4Block32Matrix matrix{
      device_weights.get(), device_scales.get(), rows, columns, columns};
  cudaEvent_t decode_start{}, decode_stop{};
  cuda_check(cudaEventCreate(&decode_start),
             "create staged decode start event");
  cuda_check(cudaEventCreate(&decode_stop),
             "create staged decode stop event");
  cuda_check(cudaEventRecord(decode_start),
             "record staged decode start");
  status_check(expert::runtime::cuda::fp4_decode_matrix_bf16(
      matrix, decoded_weights.get(),
      static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t),
      nullptr));
  cuda_check(cudaEventRecord(decode_stop), "record staged decode stop");
  cuda_check(cudaEventSynchronize(decode_stop),
             "synchronize staged decode stop");
  float decode_milliseconds{};
  cuda_check(cudaEventElapsedTime(&decode_milliseconds, decode_start,
                                  decode_stop),
             "measure staged decode elapsed time");
  static_cast<void>(cudaEventDestroy(decode_start));
  static_cast<void>(cudaEventDestroy(decode_stop));
  for (unsigned warmup = 0U; warmup < 3U; ++warmup)
    status_check(expert::runtime::cuda::bf16_gemm_q8_block32(
        matrix, decoded_weights.get(),
        static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t),
        device_q8.get(), device_q8_scales.get(), decoded_input.get(),
        static_cast<std::size_t>(batch) * columns * sizeof(std::uint16_t),
        device_output.get(), batch, nullptr));
  cudaEvent_t staged_start{}, staged_stop{};
  cuda_check(cudaEventCreate(&staged_start),
             "create staged GEMM start event");
  cuda_check(cudaEventCreate(&staged_stop),
             "create staged GEMM stop event");
  cuda_check(cudaEventRecord(staged_start), "record staged GEMM start");
  for (unsigned iteration = 0U; iteration < iterations; ++iteration)
    status_check(expert::runtime::cuda::bf16_gemm_q8_block32(
        matrix, decoded_weights.get(),
        static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t),
        device_q8.get(), device_q8_scales.get(), decoded_input.get(),
        static_cast<std::size_t>(batch) * columns * sizeof(std::uint16_t),
        device_output.get(), batch, nullptr));
  cuda_check(cudaEventRecord(staged_stop), "record staged GEMM stop");
  cuda_check(cudaEventSynchronize(staged_stop),
             "synchronize staged GEMM stop");
  float staged_total_milliseconds{};
  cuda_check(cudaEventElapsedTime(&staged_total_milliseconds, staged_start,
                                  staged_stop),
             "measure staged GEMM elapsed time");
  static_cast<void>(cudaEventDestroy(staged_start));
  static_cast<void>(cudaEventDestroy(staged_stop));
  return {bytes / (static_cast<double>(milliseconds) * 1.0e6),
          current_milliseconds, static_cast<double>(decode_milliseconds),
          static_cast<double>(staged_total_milliseconds) / iterations};
}

struct DecodeBatchBandwidth {
  double selected_gb_per_second{};
  double tensor_core_gb_per_second{};
};

DecodeBatchBandwidth decode_batch_bandwidth_check(std::uint32_t batch = 2U) {
  constexpr std::uint32_t rows = 17408U;
  constexpr std::uint32_t columns = 5120U;
  constexpr std::uint32_t iterations = 12U;
  const auto weight_bytes = static_cast<std::size_t>(rows) * columns / 2U;
  const auto scale_bytes = static_cast<std::size_t>(rows) * columns / 32U;
  std::vector<std::uint8_t> weights(weight_bytes, 0x21U);
  std::vector<std::uint8_t> scales(scale_bytes, 127U);
  std::vector<float> input(static_cast<std::size_t>(batch) * columns);
  for (std::size_t index = 0U; index < input.size(); ++index)
    input[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.1F;
  DeviceBuffer<std::uint8_t> device_weights(weight_bytes);
  DeviceBuffer<std::uint8_t> device_scales(scale_bytes);
  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::int8_t> device_q8(
      static_cast<std::size_t>(batch) * columns);
  DeviceBuffer<float> device_q8_scales(batch);
  DeviceBuffer<float> device_output(static_cast<std::size_t>(batch) * rows);
  device_weights.upload(weights);
  device_scales.upload(scales);
  device_input.upload(input);
  status_check(expert::runtime::cuda::quantize_q8_batch(
      device_input.get(), device_q8.get(), device_q8_scales.get(), batch,
      columns, columns, nullptr));
  const expert::runtime::cuda::Fp4Block32Matrix matrix{
      device_weights.get(), device_scales.get(), rows, columns, columns};

  const auto measure = [&](std::uint32_t implementation) {
    for (unsigned warmup = 0U; warmup < 3U; ++warmup) {
      if (implementation == 1U)
        status_check(expert::runtime::cuda::fp4_gemm_q8_block32(
            matrix, device_q8.get(), device_q8_scales.get(),
            device_output.get(), batch, nullptr));
      else
        status_check(
            expert::runtime::cuda::fp4_gemv_q8_batch_weight_reuse(
                matrix, device_q8.get(), device_q8_scales.get(),
                device_output.get(), batch, nullptr));
    }
    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "create decode batch start event");
    cuda_check(cudaEventCreate(&stop), "create decode batch stop event");
    cuda_check(cudaEventRecord(start), "record decode batch start");
    for (unsigned iteration = 0U; iteration < iterations; ++iteration) {
      if (implementation == 1U)
        status_check(expert::runtime::cuda::fp4_gemm_q8_block32(
            matrix, device_q8.get(), device_q8_scales.get(),
            device_output.get(), batch, nullptr));
      else
        status_check(
            expert::runtime::cuda::fp4_gemv_q8_batch_weight_reuse(
                matrix, device_q8.get(), device_q8_scales.get(),
                device_output.get(), batch, nullptr));
    }
    cuda_check(cudaEventRecord(stop), "record decode batch stop");
    cuda_check(cudaEventSynchronize(stop), "synchronize decode batch stop");
    float milliseconds{};
    cuda_check(cudaEventElapsedTime(&milliseconds, start, stop),
               "measure decode batch elapsed time");
    static_cast<void>(cudaEventDestroy(start));
    static_cast<void>(cudaEventDestroy(stop));
    const auto bytes = static_cast<double>(weight_bytes + scale_bytes) *
                       iterations;
    return bytes / (static_cast<double>(milliseconds) * 1.0e6);
  };
  return {measure(0U), measure(1U)};
}

double wide_decode_batch_bandwidth_check(std::uint32_t batch = 2U) {
  constexpr std::uint32_t rows = 5120U;
  constexpr std::uint32_t columns = 17408U;
  constexpr std::uint32_t iterations = 12U;
  const auto weight_bytes = static_cast<std::size_t>(rows) * columns / 2U;
  const auto scale_bytes = static_cast<std::size_t>(rows) * columns / 32U;
  std::vector<std::uint8_t> weights(weight_bytes, 0x21U);
  std::vector<std::uint8_t> scales(scale_bytes, 127U);
  std::vector<float> input(static_cast<std::size_t>(batch) * columns);
  for (std::size_t index = 0U; index < input.size(); ++index)
    input[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.1F;
  DeviceBuffer<std::uint8_t> device_weights(weight_bytes);
  DeviceBuffer<std::uint8_t> device_scales(scale_bytes);
  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<std::int8_t> device_q8(
      static_cast<std::size_t>(batch) * columns);
  DeviceBuffer<float> device_q8_scales(batch);
  DeviceBuffer<float> device_output(static_cast<std::size_t>(batch) * rows);
  device_weights.upload(weights);
  device_scales.upload(scales);
  device_input.upload(input);
  status_check(expert::runtime::cuda::quantize_q8_batch(
      device_input.get(), device_q8.get(), device_q8_scales.get(), batch,
      columns, columns, nullptr));
  const expert::runtime::cuda::Fp4Block32Matrix matrix{
      device_weights.get(), device_scales.get(), rows, columns, columns};
  for (unsigned warmup = 0U; warmup < 3U; ++warmup)
    status_check(expert::runtime::cuda::fp4_gemv_q8_batch_weight_reuse(
        matrix, device_q8.get(), device_q8_scales.get(), device_output.get(),
        batch, nullptr));
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start),
             "create wide decode batch start event");
  cuda_check(cudaEventCreate(&stop),
             "create wide decode batch stop event");
  cuda_check(cudaEventRecord(start), "record wide decode batch start");
  for (unsigned iteration = 0U; iteration < iterations; ++iteration)
    status_check(expert::runtime::cuda::fp4_gemv_q8_batch_weight_reuse(
        matrix, device_q8.get(), device_q8_scales.get(), device_output.get(),
        batch, nullptr));
  cuda_check(cudaEventRecord(stop), "record wide decode batch stop");
  cuda_check(cudaEventSynchronize(stop),
             "synchronize wide decode batch stop");
  float milliseconds{};
  cuda_check(cudaEventElapsedTime(&milliseconds, start, stop),
             "measure wide decode batch elapsed time");
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  const auto bytes = static_cast<double>(weight_bytes + scale_bytes) *
                     iterations;
  return bytes / (static_cast<double>(milliseconds) * 1.0e6);
}

double attention_prefill_4096_milliseconds() {
  constexpr std::uint32_t rows = 64U;
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 4U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = 256U;
  constexpr std::uint32_t context_tokens = 4096U;
  constexpr std::uint32_t split_tokens = 512U;
  constexpr std::uint32_t maximum_splits = 8U;
  constexpr std::uint32_t iterations = 4U;
  constexpr std::uint32_t pages = context_tokens / page_tokens;
  constexpr std::uint32_t record_bytes = head_dim / 2U + head_dim / 32U;
  constexpr std::uint32_t page_bytes =
      2U * page_tokens * kv_heads * record_bytes;
  const auto query_values =
      static_cast<std::size_t>(rows) * query_heads * 2U * head_dim;
  const auto output_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  const auto partial_count = static_cast<std::size_t>(rows) *
                             maximum_splits * query_heads;
  DeviceBuffer<float> query(query_values);
  DeviceBuffer<float> output(output_values);
  DeviceBuffer<float> maxima(partial_count);
  DeviceBuffer<float> sums(partial_count);
  DeviceBuffer<float> partials(partial_count * head_dim);
  DeviceBuffer<std::uint8_t> page_storage(
      static_cast<std::size_t>(pages) * page_bytes);
  DeviceBuffer<void*> page_table(pages);
  std::vector<float> host_query(query_values);
  for (std::size_t index = 0U; index < host_query.size(); ++index)
    host_query[index] =
        std::sin(static_cast<float>(index + 1U) * 0.001F) * 0.1F;
  query.upload(host_query);
  page_storage.upload(std::vector<std::uint8_t>(
      static_cast<std::size_t>(pages) * page_bytes, 0U));
  std::vector<void*> host_pages(pages);
  for (std::uint32_t page = 0U; page < pages; ++page)
    host_pages[page] = page_storage.get() +
                       static_cast<std::size_t>(page) * page_bytes;
  page_table.upload(host_pages);
  const expert::runtime::cuda::PagedFp4GatedGqaPrefillLaunch launch{
      query.get(), reinterpret_cast<const void* const*>(page_table.get()),
      output.get(), maxima.get(), sums.get(), partials.get(),
      context_tokens - rows + 1U, rows, 0U, page_tokens, query_heads,
      kv_heads, head_dim, split_tokens, maximum_splits, nullptr};
  for (unsigned warmup = 0U; warmup < 2U; ++warmup)
    status_check(
        expert::runtime::cuda::gated_gqa_attention_prefill_paged_fp4(launch));
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "create attention start event");
  cuda_check(cudaEventCreate(&stop), "create attention stop event");
  cuda_check(cudaEventRecord(start), "record attention start");
  for (unsigned iteration = 0U; iteration < iterations; ++iteration)
    status_check(
        expert::runtime::cuda::gated_gqa_attention_prefill_paged_fp4(launch));
  cuda_check(cudaEventRecord(stop), "record attention stop");
  cuda_check(cudaEventSynchronize(stop), "synchronize attention stop");
  float milliseconds{};
  cuda_check(cudaEventElapsedTime(&milliseconds, start, stop),
             "measure attention elapsed time");
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  return static_cast<double>(milliseconds) / iterations;
}

struct LongContextAttentionProfile final {
  struct Decode final {
    std::uint32_t split_tokens{};
    double milliseconds{};
    double kv_gb_per_second{};
    double maximum_absolute_difference{};
  };

  double prefill_milliseconds{};
  double prefill_maximum_absolute_difference{};
  double decode_milliseconds{};
  double decode_kv_gb_per_second{};
  double tensor_core_decode_milliseconds{};
  double tensor_core_decode_kv_gb_per_second{};
  double tensor_core_maximum_absolute_difference{};
  double two_position_prefill_milliseconds{};
  double two_position_microbatch_milliseconds{};
  double two_position_maximum_absolute_difference{};
  std::vector<Decode> decode_profiles;
};

LongContextAttentionProfile attention_262144_profile() {
  constexpr std::uint32_t rows = 512U;
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 4U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = 256U;
  constexpr std::uint32_t context_tokens = 262144U;
  constexpr std::uint32_t prefill_split_tokens = 4096U;
  constexpr std::uint32_t prefill_maximum_splits = 64U;
  constexpr std::uint32_t staged_prefill_split_tokens = 8192U;
  constexpr std::array<std::uint32_t, 6U> decode_split_tokens{
      256U, 512U, 1024U, 2048U, 4096U, 8192U};
  constexpr std::uint32_t decode_maximum_splits = 1024U;
  constexpr std::uint32_t record_bytes = head_dim / 2U + head_dim / 32U;
  constexpr std::uint32_t page_bytes =
      2U * page_tokens * kv_heads * record_bytes;
  constexpr std::uint32_t pages = context_tokens / page_tokens;
  const auto query_values =
      static_cast<std::size_t>(rows) * query_heads * 2U * head_dim;
  const auto output_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  const auto prefill_partial_count = static_cast<std::size_t>(rows) *
                                     prefill_maximum_splits * query_heads;

  DeviceBuffer<float> query(query_values);
  DeviceBuffer<float> output(output_values);
  DeviceBuffer<float> prefill_maxima(prefill_partial_count);
  DeviceBuffer<float> prefill_sums(prefill_partial_count);
  DeviceBuffer<float> prefill_partials(prefill_partial_count * head_dim);
  DeviceBuffer<float> prefill_reference_output(output_values);
  const auto staged_query_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  const auto staged_kv_values = static_cast<std::size_t>(kv_heads) *
                                staged_prefill_split_tokens * head_dim;
  const auto staged_score_values = static_cast<std::size_t>(rows) *
                                   query_heads *
                                   staged_prefill_split_tokens;
  DeviceBuffer<std::uint16_t> staged_queries(staged_query_values);
  DeviceBuffer<std::uint16_t> staged_keys(staged_kv_values);
  DeviceBuffer<std::uint16_t> staged_values(staged_kv_values);
  DeviceBuffer<float> staged_scores(staged_score_values);
  DeviceBuffer<std::uint16_t> staged_probabilities(staged_score_values);
  DeviceBuffer<float> staged_accumulator(staged_query_values);
  DeviceBuffer<float> staged_maxima(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> staged_sums(
      static_cast<std::size_t>(rows) * query_heads);
  DeviceBuffer<float> decode_maxima(decode_maximum_splits * query_heads);
  DeviceBuffer<float> decode_sums(decode_maximum_splits * query_heads);
  DeviceBuffer<float> decode_partials(
      decode_maximum_splits * query_heads * head_dim);
  const auto kv_payload_bytes =
      static_cast<std::size_t>(pages) * page_bytes;
  DeviceBuffer<std::uint8_t> page_storage(kv_payload_bytes);
  DeviceBuffer<void*> page_table(pages);

  std::vector<float> host_query(query_values);
  for (std::size_t index = 0U; index < host_query.size(); ++index)
    host_query[index] =
        std::sin(static_cast<float>(index + 1U) * 0.001F) * 0.1F;
  query.upload(host_query);
  // Exercise the same FP4 decode and online-softmax path as a populated KV
  // cache. Packed values alternate signs while scale exponents vary by
  // token, head, block, and K/V kind; all generated values remain finite.
  std::vector<std::uint8_t> host_page_storage(kv_payload_bytes, 0x19U);
  constexpr std::uint32_t packed_bytes = head_dim / 2U;
  constexpr std::uint32_t scale_bytes = head_dim / 32U;
  for (std::uint32_t page = 0U; page < pages; ++page) {
    auto* page_base = host_page_storage.data() +
                      static_cast<std::size_t>(page) * page_bytes;
    for (std::uint32_t kind = 0U; kind < 2U; ++kind) {
      auto* kind_base = page_base +
                        static_cast<std::size_t>(kind) * page_tokens *
                            kv_heads * record_bytes;
      for (std::uint32_t token = 0U; token < page_tokens; ++token) {
        const auto global_token = page * page_tokens + token;
        for (std::uint32_t head = 0U; head < kv_heads; ++head) {
          auto* record = kind_base +
                         (static_cast<std::size_t>(token) * kv_heads + head) *
                             record_bytes;
          for (std::uint32_t block = 0U; block < scale_bytes; ++block)
            record[packed_bytes + block] = static_cast<std::uint8_t>(
                121U + (global_token + head + block + kind) % 7U);
        }
      }
    }
  }
  page_storage.upload(host_page_storage);
  std::vector<void*> host_pages(pages);
  for (std::uint32_t page = 0U; page < pages; ++page)
    host_pages[page] = page_storage.get() +
                       static_cast<std::size_t>(page) * page_bytes;
  page_table.upload(host_pages);

  const expert::runtime::cuda::PagedFp4GatedGqaPrefillLaunch prefill{
      query.get(), reinterpret_cast<const void* const*>(page_table.get()),
      output.get(), prefill_maxima.get(), prefill_sums.get(),
      prefill_partials.get(), context_tokens - rows + 1U, rows, 0U,
      page_tokens, query_heads, kv_heads, head_dim, prefill_split_tokens,
      prefill_maximum_splits, nullptr};
  status_check(
      expert::runtime::cuda::gated_gqa_attention_prefill_paged_fp4(prefill));
  cuda_check(cudaMemcpy(prefill_reference_output.get(), output.get(),
                        output_values * sizeof(float),
                        cudaMemcpyDeviceToDevice),
             "copy long-context prefill reference");
  const expert::runtime::cuda::PagedFp4GatedGqaStagedPrefillWorkspace
      staged_workspace{
          staged_queries.get(),
          staged_query_values * sizeof(std::uint16_t),
          staged_keys.get(),
          staged_kv_values * sizeof(std::uint16_t),
          staged_values.get(),
          staged_kv_values * sizeof(std::uint16_t),
          staged_scores.get(),
          staged_score_values * sizeof(float),
          staged_probabilities.get(),
          staged_score_values * sizeof(std::uint16_t),
          staged_accumulator.get(),
          staged_query_values * sizeof(float),
          staged_maxima.get(),
          static_cast<std::size_t>(rows) * query_heads * sizeof(float),
          staged_sums.get(),
          static_cast<std::size_t>(rows) * query_heads * sizeof(float),
          staged_prefill_split_tokens};
  status_check(
      expert::runtime::cuda::gated_gqa_attention_staged_prefill_paged_fp4(
          prefill, staged_workspace));
  cudaEvent_t prefill_start{}, prefill_stop{};
  cuda_check(cudaEventCreate(&prefill_start),
             "create long-context prefill start event");
  cuda_check(cudaEventCreate(&prefill_stop),
             "create long-context prefill stop event");
  cuda_check(cudaEventRecord(prefill_start),
             "record long-context prefill start");
  status_check(
      expert::runtime::cuda::gated_gqa_attention_staged_prefill_paged_fp4(
          prefill, staged_workspace));
  cuda_check(cudaEventRecord(prefill_stop),
             "record long-context prefill stop");
  cuda_check(cudaEventSynchronize(prefill_stop),
             "synchronize long-context prefill stop");
  float prefill_milliseconds{};
  cuda_check(cudaEventElapsedTime(&prefill_milliseconds, prefill_start,
                                  prefill_stop),
             "measure long-context prefill elapsed time");
  static_cast<void>(cudaEventDestroy(prefill_start));
  static_cast<void>(cudaEventDestroy(prefill_stop));
  const auto prefill_reference = prefill_reference_output.download();
  const auto prefill_staged = output.download();
  double prefill_maximum_absolute_difference{};
  for (std::size_t index = 0U; index < output_values; ++index)
    prefill_maximum_absolute_difference = std::max(
        prefill_maximum_absolute_difference,
        static_cast<double>(
            std::abs(prefill_reference[index] - prefill_staged[index])));

  constexpr unsigned microbatch_iterations = 8U;
  const expert::runtime::cuda::PagedFp4GatedGqaPrefillLaunch two_prefill{
      query.get(), reinterpret_cast<const void* const*>(page_table.get()),
      output.get(), prefill_maxima.get(), prefill_sums.get(),
      prefill_partials.get(), context_tokens - 1U, 2U, 0U, page_tokens,
      query_heads, kv_heads, head_dim, prefill_split_tokens,
      prefill_maximum_splits, nullptr};
  for (unsigned warmup = 0U; warmup < 2U; ++warmup)
    status_check(
        expert::runtime::cuda::gated_gqa_attention_prefill_paged_fp4(
            two_prefill));
  cudaEvent_t two_prefill_start{}, two_prefill_stop{};
  cuda_check(cudaEventCreate(&two_prefill_start),
             "create two-position prefill start event");
  cuda_check(cudaEventCreate(&two_prefill_stop),
             "create two-position prefill stop event");
  cuda_check(cudaEventRecord(two_prefill_start),
             "record two-position prefill start");
  for (unsigned iteration = 0U; iteration < microbatch_iterations;
       ++iteration)
    status_check(
        expert::runtime::cuda::gated_gqa_attention_prefill_paged_fp4(
            two_prefill));
  cuda_check(cudaEventRecord(two_prefill_stop),
             "record two-position prefill stop");
  cuda_check(cudaEventSynchronize(two_prefill_stop),
             "synchronize two-position prefill stop");
  float two_prefill_total_milliseconds{};
  cuda_check(cudaEventElapsedTime(&two_prefill_total_milliseconds,
                                  two_prefill_start, two_prefill_stop),
             "measure two-position prefill elapsed time");
  static_cast<void>(cudaEventDestroy(two_prefill_start));
  static_cast<void>(cudaEventDestroy(two_prefill_stop));
  const auto two_prefill_milliseconds =
      static_cast<double>(two_prefill_total_milliseconds) /
      microbatch_iterations;
  const auto two_prefill_output = output.download();

  const expert::runtime::cuda::PagedFp4GatedGqaPrefillLaunch two_microbatch{
      query.get(), reinterpret_cast<const void* const*>(page_table.get()),
      output.get(), decode_maxima.get(), decode_sums.get(),
      decode_partials.get(), context_tokens - 1U, 2U, 0U, page_tokens,
      query_heads, kv_heads, head_dim, 512U, 512U, nullptr};
  for (unsigned warmup = 0U; warmup < 2U; ++warmup)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_microbatch_paged_fp4_tensor_core(
                         two_microbatch));
  cudaEvent_t two_microbatch_start{}, two_microbatch_stop{};
  cuda_check(cudaEventCreate(&two_microbatch_start),
             "create fused two-position attention start event");
  cuda_check(cudaEventCreate(&two_microbatch_stop),
             "create fused two-position attention stop event");
  cuda_check(cudaEventRecord(two_microbatch_start),
             "record fused two-position attention start");
  for (unsigned iteration = 0U; iteration < microbatch_iterations;
       ++iteration)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_microbatch_paged_fp4_tensor_core(
                         two_microbatch));
  cuda_check(cudaEventRecord(two_microbatch_stop),
             "record fused two-position attention stop");
  cuda_check(cudaEventSynchronize(two_microbatch_stop),
             "synchronize fused two-position attention stop");
  float two_microbatch_total_milliseconds{};
  cuda_check(cudaEventElapsedTime(&two_microbatch_total_milliseconds,
                                  two_microbatch_start,
                                  two_microbatch_stop),
             "measure fused two-position attention elapsed time");
  static_cast<void>(cudaEventDestroy(two_microbatch_start));
  static_cast<void>(cudaEventDestroy(two_microbatch_stop));
  const auto two_microbatch_milliseconds =
      static_cast<double>(two_microbatch_total_milliseconds) /
      microbatch_iterations;
  const auto two_microbatch_output = output.download();
  double two_position_maximum_absolute_difference{};
  for (std::size_t index = 0U;
       index < 2U * static_cast<std::size_t>(query_heads) * head_dim; ++index)
    two_position_maximum_absolute_difference = std::max(
        two_position_maximum_absolute_difference,
        static_cast<double>(std::abs(two_microbatch_output[index] -
                                     two_prefill_output[index])));

  constexpr unsigned decode_iterations = 8U;
  std::vector<LongContextAttentionProfile::Decode> decode_profiles;
  std::vector<float> reference_output;
  std::vector<float> production_output;
  for (const auto split_tokens : decode_split_tokens) {
    const expert::runtime::cuda::PagedFp4GatedGqaAttentionLaunch decode{
        query.get(), reinterpret_cast<const void* const*>(page_table.get()),
        output.get(), decode_maxima.get(), decode_sums.get(),
        decode_partials.get(), context_tokens, 0U, page_tokens, query_heads,
        kv_heads, head_dim, split_tokens, decode_maximum_splits, nullptr};
    for (unsigned warmup = 0U; warmup < 2U; ++warmup)
      status_check(
          expert::runtime::cuda::gated_gqa_attention_decode_paged_fp4(
              decode));
    cudaEvent_t decode_start{}, decode_stop{};
    cuda_check(cudaEventCreate(&decode_start),
               "create long-context decode start event");
    cuda_check(cudaEventCreate(&decode_stop),
               "create long-context decode stop event");
    cuda_check(cudaEventRecord(decode_start),
               "record long-context decode start");
    for (unsigned iteration = 0U; iteration < decode_iterations; ++iteration)
      status_check(
          expert::runtime::cuda::gated_gqa_attention_decode_paged_fp4(
              decode));
    cuda_check(cudaEventRecord(decode_stop),
               "record long-context decode stop");
    cuda_check(cudaEventSynchronize(decode_stop),
               "synchronize long-context decode stop");
    float decode_total_milliseconds{};
    cuda_check(cudaEventElapsedTime(&decode_total_milliseconds, decode_start,
                                    decode_stop),
               "measure long-context decode elapsed time");
    static_cast<void>(cudaEventDestroy(decode_start));
    static_cast<void>(cudaEventDestroy(decode_stop));
    const auto milliseconds =
        static_cast<double>(decode_total_milliseconds) / decode_iterations;
    const auto host_output = output.download();
    if (reference_output.empty()) reference_output = host_output;
    if (split_tokens == 512U) production_output = host_output;
    double maximum_absolute_difference{};
    for (std::size_t index = 0U;
         index < static_cast<std::size_t>(query_heads) * head_dim; ++index)
      maximum_absolute_difference = std::max(
          maximum_absolute_difference,
          static_cast<double>(
              std::abs(host_output[index] - reference_output[index])));
    decode_profiles.push_back(
        {split_tokens, milliseconds,
         static_cast<double>(kv_payload_bytes) / (milliseconds * 1.0e6),
         maximum_absolute_difference});
  }
  const auto production = std::find_if(
      decode_profiles.begin(), decode_profiles.end(),
      [](const auto& profile) { return profile.split_tokens == 512U; });
  if (production == decode_profiles.end())
    throw std::runtime_error("missing production attention profile");
  if (production_output.empty())
    throw std::runtime_error("missing production attention output");
  const expert::runtime::cuda::PagedFp4GatedGqaAttentionLaunch tensor_core{
      query.get(), reinterpret_cast<const void* const*>(page_table.get()),
      output.get(), decode_maxima.get(), decode_sums.get(),
      decode_partials.get(), context_tokens, 0U, page_tokens, query_heads,
      kv_heads, head_dim, 512U, decode_maximum_splits, nullptr};
  for (unsigned warmup = 0U; warmup < 2U; ++warmup)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_decode_paged_fp4_tensor_core(
                         tensor_core));
  cudaEvent_t tensor_start{}, tensor_stop{};
  cuda_check(cudaEventCreate(&tensor_start),
             "create Tensor Core decode start event");
  cuda_check(cudaEventCreate(&tensor_stop),
             "create Tensor Core decode stop event");
  cuda_check(cudaEventRecord(tensor_start),
             "record Tensor Core decode start");
  for (unsigned iteration = 0U; iteration < decode_iterations; ++iteration)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_decode_paged_fp4_tensor_core(
                         tensor_core));
  cuda_check(cudaEventRecord(tensor_stop),
             "record Tensor Core decode stop");
  cuda_check(cudaEventSynchronize(tensor_stop),
             "synchronize Tensor Core decode stop");
  float tensor_total_milliseconds{};
  cuda_check(cudaEventElapsedTime(&tensor_total_milliseconds, tensor_start,
                                  tensor_stop),
             "measure Tensor Core decode elapsed time");
  static_cast<void>(cudaEventDestroy(tensor_start));
  static_cast<void>(cudaEventDestroy(tensor_stop));
  const auto tensor_milliseconds =
      static_cast<double>(tensor_total_milliseconds) / decode_iterations;
  const auto tensor_output = output.download();
  double tensor_maximum_absolute_difference{};
  for (std::size_t index = 0U;
       index < static_cast<std::size_t>(query_heads) * head_dim; ++index)
    tensor_maximum_absolute_difference = std::max(
        tensor_maximum_absolute_difference,
        static_cast<double>(
            std::abs(tensor_output[index] - production_output[index])));
  return {static_cast<double>(prefill_milliseconds),
          prefill_maximum_absolute_difference,
          production->milliseconds, production->kv_gb_per_second,
          tensor_milliseconds,
          static_cast<double>(kv_payload_bytes) /
              (tensor_milliseconds * 1.0e6),
          tensor_maximum_absolute_difference,
          two_prefill_milliseconds,
          two_microbatch_milliseconds,
          two_position_maximum_absolute_difference,
          std::move(decode_profiles)};
}

struct Fp4KeyOutlier1AttentionProfile final {
  double milliseconds{};
  double kv_gb_per_second{};
  double sixteen_layer_milliseconds{};
};

Fp4KeyOutlier1AttentionProfile
fp4_key_outlier1_attention_262144_profile() {
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 4U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::uint32_t page_tokens = 256U;
  constexpr std::uint32_t context_tokens = 262144U;
  constexpr std::uint32_t split_tokens = 512U;
  constexpr std::uint32_t maximum_splits = 512U;
  constexpr std::uint32_t fp4_record_bytes =
      head_dim / 2U + head_dim / 32U;
  constexpr std::uint32_t key_correction_bytes =
      head_dim / 32U * sizeof(std::uint32_t);
  constexpr std::uint32_t key_record_bytes =
      fp4_record_bytes + key_correction_bytes;
  constexpr std::uint32_t page_bytes =
      page_tokens * kv_heads * (key_record_bytes + fp4_record_bytes);
  constexpr std::uint32_t pages = context_tokens / page_tokens;
  constexpr std::size_t kv_payload_bytes =
      static_cast<std::size_t>(pages) * page_bytes;

  DeviceBuffer<float> query(
      static_cast<std::size_t>(query_heads) * 2U * head_dim);
  DeviceBuffer<float> output(
      static_cast<std::size_t>(query_heads) * head_dim);
  DeviceBuffer<float> partial_maxima(maximum_splits * query_heads);
  DeviceBuffer<float> partial_sums(maximum_splits * query_heads);
  DeviceBuffer<float> partial_outputs(
      static_cast<std::size_t>(maximum_splits) * query_heads * head_dim);
  DeviceBuffer<std::uint8_t> page_storage(kv_payload_bytes);
  DeviceBuffer<void*> page_table(pages);

  std::vector<float> host_query(
      static_cast<std::size_t>(query_heads) * 2U * head_dim);
  for (std::size_t index = 0U; index < host_query.size(); ++index)
    host_query[index] =
        std::sin(static_cast<float>(index + 1U) * 0.001F) * 0.1F;
  query.upload(host_query);
  cuda_check(cudaMemset(page_storage.get(), 0, kv_payload_bytes),
             "initialize FP4 key-outlier-1 long-context pages");
  std::vector<void*> host_pages(pages);
  for (std::uint32_t page = 0U; page < pages; ++page)
    host_pages[page] = page_storage.get() +
                       static_cast<std::size_t>(page) * page_bytes;
  page_table.upload(host_pages);

  const expert::runtime::cuda::PagedFp4KeyOutlier1GatedGqaAttentionLaunch
      launch{query.get(),
             reinterpret_cast<const void* const*>(page_table.get()),
             output.get(),
             partial_maxima.get(),
             partial_sums.get(),
             partial_outputs.get(),
             context_tokens,
             0U,
             page_tokens,
             query_heads,
             kv_heads,
             head_dim,
             split_tokens,
             maximum_splits,
             nullptr};
  for (unsigned warmup = 0U; warmup < 2U; ++warmup)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_decode_paged_fp4_key_outlier1_tensor_core(
                         launch));
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start),
             "create FP4 key-outlier-1 attention start event");
  cuda_check(cudaEventCreate(&stop),
             "create FP4 key-outlier-1 attention stop event");
  cuda_check(cudaEventRecord(start),
             "record FP4 key-outlier-1 attention start");
  constexpr unsigned iterations = 8U;
  for (unsigned iteration = 0U; iteration < iterations; ++iteration)
    status_check(expert::runtime::cuda::
                     gated_gqa_attention_decode_paged_fp4_key_outlier1_tensor_core(
                         launch));
  cuda_check(cudaEventRecord(stop),
             "record FP4 key-outlier-1 attention stop");
  cuda_check(cudaEventSynchronize(stop),
             "synchronize FP4 key-outlier-1 attention stop");
  float total_milliseconds{};
  cuda_check(cudaEventElapsedTime(&total_milliseconds, start, stop),
             "measure FP4 key-outlier-1 attention elapsed time");
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  const auto milliseconds =
      static_cast<double>(total_milliseconds) / iterations;
  return {milliseconds,
          static_cast<double>(kv_payload_bytes) / (milliseconds * 1.0e6),
          16.0 * milliseconds};
}

double standard_gqa_ratio16_check() {
  constexpr std::uint32_t rows = 2U;
  constexpr std::uint32_t query_heads = 32U;
  constexpr std::uint32_t kv_heads = 2U;
  constexpr std::uint32_t head_dim = 32U;
  constexpr std::uint32_t rotary_dim = 32U;
  constexpr float theta = 10000.0F;
  const auto query_values = static_cast<std::size_t>(rows) * query_heads *
                            head_dim;
  const auto kv_values = static_cast<std::size_t>(rows) * kv_heads * head_dim;
  std::vector<float> query(query_values);
  std::vector<float> key(kv_values);
  std::vector<float> value(kv_values);
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 1U) * 0.013F);
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::cos(static_cast<float>(index + 3U) * 0.017F);
    value[index] = std::sin(static_cast<float>(index + 5U) * 0.019F);
  }
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<std::uint16_t> fp16_keys(kv_values);
  DeviceBuffer<std::uint16_t> fp16_values(kv_values);
  device_query.upload(query);
  device_key.upload(key);
  device_value.upload(value);
  status_check(expert::runtime::cuda::standard_gqa_qkv_rope_fp16_batch(
      device_query.get(), device_key.get(), device_value.get(),
      fp16_keys.get(), fp16_values.get(), 7U, rows, query_heads, kv_heads,
      head_dim, rotary_dim, theta, nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize ratio-16 standard GQA smoke");
  const auto actual_query = device_query.download();
  const auto actual_key = device_key.download();
  const auto actual_fp16_keys = fp16_keys.download();
  const auto actual_fp16_values = fp16_values.download();
  double maximum_error{};
  const auto expected_rope = [&](const float* source, std::uint32_t dimension,
                                 std::uint32_t position) {
    const auto half = rotary_dim / 2U;
    const auto pair = dimension % half;
    const auto angle = static_cast<float>(position) *
        std::pow(theta, -2.0F * static_cast<float>(pair) /
                            static_cast<float>(rotary_dim));
    const auto other = dimension < half ? -source[dimension + half]
                                        : source[dimension - half];
    return source[dimension] * std::cos(angle) + other * std::sin(angle);
  };
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto base = (static_cast<std::size_t>(row) * query_heads + head) *
                        head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(
                actual_query[base + dimension] -
                expected_rope(query.data() + base, dimension, 7U + row))));
    }
    for (std::uint32_t head = 0U; head < kv_heads; ++head) {
      const auto base = (static_cast<std::size_t>(row) * kv_heads + head) *
                        head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto expected = expected_rope(
            key.data() + base, dimension, 7U + row);
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(actual_key[base + dimension] -
                                         expected)));
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(
                __half2float(*reinterpret_cast<const __half*>(
                    &actual_fp16_keys[base + dimension])) - expected)));
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(
                __half2float(*reinterpret_cast<const __half*>(
                    &actual_fp16_values[base + dimension])) -
                value[base + dimension])));
      }
    }
  }
  return maximum_error;
}

double gated_gqa_ratio12_check() {
  constexpr std::uint32_t rows = 2U;
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 2U;
  constexpr std::uint32_t head_dim = 32U;
  constexpr std::uint32_t rotary_dim = 32U;
  constexpr float epsilon = 1.0e-6F;
  constexpr float theta = 10000.0F;
  const auto query_values = static_cast<std::size_t>(rows) * 2U *
                            query_heads * head_dim;
  const auto kv_values = static_cast<std::size_t>(rows) * kv_heads * head_dim;
  std::vector<float> query(query_values);
  std::vector<float> key(kv_values);
  std::vector<float> value(kv_values);
  std::vector<float> query_norm(head_dim);
  std::vector<float> key_norm(head_dim);
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 1U) * 0.013F);
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::cos(static_cast<float>(index + 3U) * 0.017F);
    value[index] = std::sin(static_cast<float>(index + 5U) * 0.019F);
  }
  for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
    query_norm[dimension] =
        0.01F * std::sin(static_cast<float>(dimension) * 0.1F);
    key_norm[dimension] =
        0.01F * std::cos(static_cast<float>(dimension) * 0.1F);
  }
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<float> device_query_norm(query_norm.size());
  DeviceBuffer<float> device_key_norm(key_norm.size());
  DeviceBuffer<std::uint16_t> fp16_keys(kv_values);
  DeviceBuffer<std::uint16_t> fp16_values(kv_values);
  device_query.upload(query);
  device_key.upload(key);
  device_value.upload(value);
  device_query_norm.upload(query_norm);
  device_key_norm.upload(key_norm);
  status_check(expert::runtime::cuda::gated_gqa_qkv_rope_fp16_batch(
      device_query.get(), device_key.get(), device_value.get(),
      device_query_norm.get(), device_key_norm.get(), fp16_keys.get(),
      fp16_values.get(), 7U, rows, query_heads, kv_heads, head_dim,
      rotary_dim, epsilon, theta, nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize ratio-12 gated GQA smoke");
  const auto actual_query = device_query.download();
  const auto actual_key = device_key.download();
  const auto actual_fp16_keys = fp16_keys.download();
  const auto actual_fp16_values = fp16_values.download();
  const auto expected_rope = [&](const std::vector<float>& normalized,
                                 std::uint32_t dimension,
                                 std::uint32_t position) {
    const auto half = rotary_dim / 2U;
    const auto pair = dimension % half;
    const auto angle = static_cast<float>(position) *
        std::pow(theta, -2.0F * static_cast<float>(pair) /
                            static_cast<float>(rotary_dim));
    const auto other = dimension < half ? -normalized[dimension + half]
                                        : normalized[dimension - half];
    return normalized[dimension] * std::cos(angle) +
           other * std::sin(angle);
  };
  double maximum_error{};
  const auto normalized_head = [&](const float* source,
                                   const std::vector<float>& weight) {
    float square{};
    for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
      square += source[dimension] * source[dimension];
    const auto scale =
        1.0F / std::sqrt(square / static_cast<float>(head_dim) + epsilon);
    std::vector<float> result(head_dim);
    for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension)
      result[dimension] =
          source[dimension] * scale * (1.0F + weight[dimension]);
    return result;
  };
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto base = (static_cast<std::size_t>(row) * query_heads + head) *
                        2U * head_dim;
      const auto normalized = normalized_head(query.data() + base,
                                              query_norm);
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto expected = expected_rope(normalized, dimension, 7U + row);
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(
                std::abs(actual_query[base + dimension] - expected)));
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(
                actual_query[base + head_dim + dimension] -
                query[base + head_dim + dimension])));
      }
    }
    for (std::uint32_t head = 0U; head < kv_heads; ++head) {
      const auto base = (static_cast<std::size_t>(row) * kv_heads + head) *
                        head_dim;
      const auto normalized = normalized_head(key.data() + base, key_norm);
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto expected = expected_rope(normalized, dimension, 7U + row);
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(actual_key[base + dimension] -
                                         expected)));
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(
                __half2float(*reinterpret_cast<const __half*>(
                    &actual_fp16_keys[base + dimension])) - expected)));
        maximum_error = std::max(
            maximum_error,
            static_cast<double>(std::abs(
                __half2float(*reinterpret_cast<const __half*>(
                    &actual_fp16_values[base + dimension])) -
                value[base + dimension])));
      }
    }
  }
  return maximum_error;
}

double standard_gqa_no_position_check() {
  constexpr std::uint32_t rows = 3U;
  constexpr std::uint32_t query_heads = 32U;
  constexpr std::uint32_t kv_heads = 2U;
  constexpr std::uint32_t head_dim = 32U;
  const auto query_values = static_cast<std::size_t>(rows) * query_heads *
                            head_dim;
  const auto kv_values = static_cast<std::size_t>(rows) * kv_heads * head_dim;
  std::vector<float> query(query_values);
  std::vector<float> key(kv_values);
  std::vector<float> value(kv_values);
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 1U) * 0.011F);
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::cos(static_cast<float>(index + 3U) * 0.023F);
    value[index] = std::sin(static_cast<float>(index + 5U) * 0.029F);
  }
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<std::uint16_t> fp16_keys(kv_values);
  DeviceBuffer<std::uint16_t> fp16_values(kv_values);
  device_query.upload(query);
  device_key.upload(key);
  device_value.upload(value);
  status_check(expert::runtime::cuda::standard_gqa_kv_fp16_batch(
      device_key.get(), device_value.get(),
      fp16_keys.get(), fp16_values.get(), rows, query_heads, kv_heads,
      head_dim, nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize position-free standard GQA smoke");
  const auto actual_query = device_query.download();
  const auto actual_key = device_key.download();
  const auto actual_fp16_keys = fp16_keys.download();
  const auto actual_fp16_values = fp16_values.download();
  double maximum_error{};
  for (std::size_t index = 0U; index < query.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        static_cast<double>(std::abs(actual_query[index] - query[index])));
  for (std::size_t index = 0U; index < key.size(); ++index) {
    maximum_error = std::max(
        maximum_error,
        static_cast<double>(std::abs(actual_key[index] - key[index])));
    maximum_error = std::max(
        maximum_error,
        static_cast<double>(std::abs(
            __half2float(*reinterpret_cast<const __half*>(
                &actual_fp16_keys[index])) - key[index])));
    maximum_error = std::max(
        maximum_error,
        static_cast<double>(std::abs(
            __half2float(*reinterpret_cast<const __half*>(
                &actual_fp16_values[index])) - value[index])));
  }
  return maximum_error;
}

struct DenseFp4Abi2KernelErrors final {
  double weightless_rms{};
  double sigmoid_product{};
  double scaled_tanh{};
  double normalized_gqa{};
};

DenseFp4Abi2KernelErrors dense_fp4_abi2_kernel_check() {
  constexpr std::uint32_t rows = 2U;
  constexpr std::uint32_t elements = 32U;
  constexpr float epsilon = 1.0e-5F;
  constexpr float multiplier = 0.19611613513818404F;
  constexpr float softcap = 20.0F;
  std::vector<float> input(static_cast<std::size_t>(rows) * elements);
  std::vector<float> gate(input.size());
  for (std::size_t index = 0U; index < input.size(); ++index) {
    input[index] = std::sin(static_cast<float>(index + 1U) * 0.071F) * 3.0F;
    gate[index] = std::cos(static_cast<float>(index + 3U) * 0.053F);
  }

  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<float> device_output(input.size());
  DeviceBuffer<float> device_gate(gate.size());
  device_input.upload(input);
  device_output.upload(input);
  device_gate.upload(gate);
  status_check(expert::runtime::cuda::weightless_rms_norm_batch(
      device_input.get(), device_output.get(), rows, elements, epsilon,
      nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize ABI-2 weightless RMS norm smoke");
  const auto normalized = device_output.download();

  DenseFp4Abi2KernelErrors result;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    float square{};
    for (std::uint32_t column = 0U; column < elements; ++column) {
      const auto value = input[static_cast<std::size_t>(row) * elements +
                               column];
      square += value * value;
    }
    const auto inverse = 1.0F /
        std::sqrt(square / static_cast<float>(elements) + epsilon);
    for (std::uint32_t column = 0U; column < elements; ++column) {
      const auto index = static_cast<std::size_t>(row) * elements + column;
      result.weightless_rms = std::max(
          result.weightless_rms,
          std::abs(static_cast<double>(normalized[index] -
                                       input[index] * inverse)));
    }
  }

  device_output.upload(input);
  status_check(expert::runtime::cuda::sigmoid_product_in_place(
      device_output.get(), device_gate.get(),
      static_cast<std::uint32_t>(input.size()), nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize ABI-2 sigmoid product smoke");
  const auto sigmoid = device_output.download();
  for (std::size_t index = 0U; index < input.size(); ++index) {
    const auto expected = input[index] /
        (1.0F + std::exp(-gate[index]));
    result.sigmoid_product = std::max(
        result.sigmoid_product,
        std::abs(static_cast<double>(sigmoid[index] - expected)));
  }

  device_output.upload(input);
  status_check(expert::runtime::cuda::scaled_tanh_in_place(
      device_output.get(), static_cast<std::uint32_t>(input.size()),
      multiplier, softcap, nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize ABI-2 scaled tanh smoke");
  const auto tanh_output = device_output.download();
  for (std::size_t index = 0U; index < input.size(); ++index) {
    const auto expected = softcap *
        std::tanh(input[index] * multiplier / softcap);
    result.scaled_tanh = std::max(
        result.scaled_tanh,
        std::abs(static_cast<double>(tanh_output[index] - expected)));
  }

  constexpr std::uint32_t query_heads = 4U;
  constexpr std::uint32_t kv_heads = 2U;
  constexpr std::uint32_t head_dim = 32U;
  constexpr std::uint32_t rotary_dim = 32U;
  constexpr std::uint32_t first_position = 11U;
  constexpr float query_scale = 3.87F;
  constexpr float rope_theta = 500000.0F;
  const auto query_values = static_cast<std::size_t>(rows) * query_heads *
                            head_dim;
  const auto kv_values = static_cast<std::size_t>(rows) * kv_heads * head_dim;
  std::vector<float> query(query_values);
  std::vector<float> key(kv_values);
  std::vector<float> value(kv_values);
  for (std::size_t index = 0U; index < query.size(); ++index)
    query[index] = std::sin(static_cast<float>(index + 7U) * 0.019F);
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = std::cos(static_cast<float>(index + 5U) * 0.023F);
    value[index] = std::sin(static_cast<float>(index + 9U) * 0.029F);
  }
  DeviceBuffer<float> device_query(query.size());
  DeviceBuffer<float> device_key(key.size());
  DeviceBuffer<float> device_value(value.size());
  DeviceBuffer<std::uint16_t> fp16_keys(kv_values);
  DeviceBuffer<std::uint16_t> fp16_values(kv_values);
  device_query.upload(query);
  device_key.upload(key);
  device_value.upload(value);
  status_check(expert::runtime::cuda::normalized_gqa_qkv_fp16_batch(
      device_query.get(), device_key.get(), device_value.get(),
      fp16_keys.get(), fp16_values.get(), first_position, rows, query_heads,
      kv_heads, head_dim, rotary_dim, epsilon, query_scale, rope_theta, true,
      nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize ABI-2 normalized GQA smoke");
  const auto actual_query = device_query.download();
  const auto actual_key = device_key.download();
  const auto actual_fp16_keys = fp16_keys.download();
  const auto actual_fp16_values = fp16_values.download();
  const auto expected_head = [&](const float* source,
                                 std::uint32_t dimension,
                                 std::uint32_t position,
                                 float scale) {
    float square{};
    for (std::uint32_t index = 0U; index < head_dim; ++index)
      square += source[index] * source[index];
    const auto inverse = 1.0F /
        std::sqrt(square / static_cast<float>(head_dim) + epsilon);
    const auto half = rotary_dim / 2U;
    const auto pair = dimension % half;
    const auto angle = static_cast<float>(position) *
        std::pow(rope_theta, -2.0F * static_cast<float>(pair) /
                                 static_cast<float>(rotary_dim));
    const auto current = source[dimension] * inverse * scale;
    const auto other_index = dimension < half
        ? dimension + half
        : dimension - half;
    auto other = source[other_index] * inverse * scale;
    if (dimension < half) other = -other;
    return current * std::cos(angle) + other * std::sin(angle);
  };
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t head = 0U; head < query_heads; ++head) {
      const auto base = (static_cast<std::size_t>(row) * query_heads + head) *
                        head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto expected = expected_head(
            query.data() + base, dimension, first_position + row,
            query_scale);
        result.normalized_gqa = std::max(
            result.normalized_gqa,
            std::abs(static_cast<double>(actual_query[base + dimension] -
                                         expected)));
      }
    }
    for (std::uint32_t head = 0U; head < kv_heads; ++head) {
      const auto base = (static_cast<std::size_t>(row) * kv_heads + head) *
                        head_dim;
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto index = base + dimension;
        const auto expected = expected_head(
            key.data() + base, dimension, first_position + row, 1.0F);
        result.normalized_gqa = std::max(
            result.normalized_gqa,
            std::abs(static_cast<double>(actual_key[index] - expected)));
        result.normalized_gqa = std::max(
            result.normalized_gqa,
            std::abs(static_cast<double>(
                __half2float(*reinterpret_cast<const __half*>(
                    &actual_fp16_keys[index])) - expected)));
        result.normalized_gqa = std::max(
            result.normalized_gqa,
            std::abs(static_cast<double>(
                __half2float(*reinterpret_cast<const __half*>(
                    &actual_fp16_values[index])) - value[index])));
      }
    }
  }
  return result;
}

double host_fp16_attention_check() {
  constexpr std::uint32_t rows = 3U;
  constexpr std::uint32_t query_heads = 32U;
  constexpr std::uint32_t kv_heads = 2U;
  constexpr std::uint32_t head_dim = 32U;
  constexpr std::uint32_t cache_capacity = 8U;
  constexpr std::uint32_t page_tokens = 4U;
  constexpr std::uint32_t page_count = cache_capacity / page_tokens;
  constexpr std::uint32_t first_context_tokens = 5U;
  constexpr std::uint32_t split_tokens = 4U;
  constexpr std::uint32_t grouped_heads = query_heads / kv_heads;
  constexpr std::uint32_t matrix_rows = rows * grouped_heads;
  constexpr std::size_t query_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  constexpr std::size_t cache_values =
      static_cast<std::size_t>(cache_capacity) * kv_heads * head_dim;
  constexpr std::size_t staged_values =
      static_cast<std::size_t>(split_tokens) * kv_heads * head_dim;
  constexpr std::size_t score_values =
      static_cast<std::size_t>(kv_heads) * matrix_rows * split_tokens;

  std::vector<float> query_gate(query_values * 2U);
  for (std::size_t index = 0U; index < query_gate.size(); ++index)
    query_gate[index] = std::sin(static_cast<float>(index + 1U) * 0.013F) *
                        0.35F;
  PinnedBuffer<__half> host_keys(cache_values);
  PinnedBuffer<__half> host_values(cache_values);
  for (std::size_t index = 0U; index < cache_values; ++index) {
    host_keys.get()[index] =
        __float2half(std::cos(static_cast<float>(index + 3U) * 0.017F) *
                     0.4F);
    host_values.get()[index] =
        __float2half(std::sin(static_cast<float>(index + 5U) * 0.019F) *
                     0.3F);
  }

  DeviceBuffer<float> device_query(query_gate.size());
  DeviceBuffer<float> output(query_values);
  DeviceBuffer<std::uint16_t> device_cache_keys(cache_values);
  DeviceBuffer<std::uint16_t> device_cache_values(cache_values);
  const auto page_values = static_cast<std::size_t>(page_tokens) * kv_heads *
                           head_dim;
  DeviceBuffer<std::uint16_t> device_page_zero(2U * page_values);
  DeviceBuffer<std::uint16_t> device_page_one(2U * page_values);
  DeviceBuffer<void*> device_page_table(page_count);
  DeviceBuffer<std::uint16_t> queries(query_values);
  DeviceBuffer<std::uint16_t> raw_keys(staged_values);
  DeviceBuffer<std::uint16_t> raw_values(staged_values);
  DeviceBuffer<std::uint16_t> keys(staged_values);
  DeviceBuffer<std::uint16_t> values(staged_values);
  DeviceBuffer<float> scores(score_values);
  DeviceBuffer<std::uint16_t> probabilities(score_values);
  DeviceBuffer<float> accumulator(query_values);
  DeviceBuffer<float> maxima(static_cast<std::size_t>(kv_heads) * matrix_rows);
  DeviceBuffer<float> sums(static_cast<std::size_t>(kv_heads) * matrix_rows);
  device_query.upload(query_gate);
  cuda_check(cudaMemcpy(device_cache_keys.get(), host_keys.get(),
                        cache_values * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice),
             "upload staged device FP16 keys");
  cuda_check(cudaMemcpy(device_cache_values.get(), host_values.get(),
                        cache_values * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice),
             "upload staged device FP16 values");
  std::array<void*, page_count> device_pages{
      device_page_zero.get(), device_page_one.get()};
  device_page_table.upload(
      std::vector<void*>(device_pages.begin(), device_pages.end()));
  status_check(expert::runtime::cuda::store_gqa_kv_fp16_to_paged(
      device_cache_keys.get(), device_cache_values.get(),
      reinterpret_cast<const void* const*>(device_page_table.get()), 0U,
      page_tokens, 0U, cache_capacity, kv_heads, head_dim, nullptr));
  status_check(expert::runtime::cuda::gated_gqa_attention_staged_host_fp16(
      {device_query.get(), host_keys.get(), host_values.get(), output.get(),
       cache_capacity, first_context_tokens, 0U, rows, query_heads, kv_heads,
       head_dim, nullptr},
      {queries.get(), query_values * sizeof(std::uint16_t), raw_keys.get(),
       staged_values * sizeof(std::uint16_t), raw_values.get(),
       staged_values * sizeof(std::uint16_t), keys.get(),
       staged_values * sizeof(std::uint16_t), values.get(),
       staged_values * sizeof(std::uint16_t), scores.get(),
       score_values * sizeof(float), probabilities.get(),
       score_values * sizeof(std::uint16_t), accumulator.get(),
       query_values * sizeof(float), maxima.get(),
       static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
       sums.get(),
       static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
       split_tokens}));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize staged host FP16 attention smoke");
  const auto actual = output.download();
  status_check(expert::runtime::cuda::gated_gqa_attention_staged_device_fp16(
      {device_query.get(), device_cache_keys.get(), device_cache_values.get(),
       output.get(), cache_capacity, first_context_tokens, 0U, rows,
       query_heads, kv_heads, head_dim, nullptr},
      {queries.get(), query_values * sizeof(std::uint16_t), raw_keys.get(),
       staged_values * sizeof(std::uint16_t), raw_values.get(),
       staged_values * sizeof(std::uint16_t), keys.get(),
       staged_values * sizeof(std::uint16_t), values.get(),
       staged_values * sizeof(std::uint16_t), scores.get(),
       score_values * sizeof(float), probabilities.get(),
       score_values * sizeof(std::uint16_t), accumulator.get(),
       query_values * sizeof(float), maxima.get(),
       static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
       sums.get(),
       static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
       split_tokens}));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize staged device FP16 attention smoke");
  const auto device_actual = output.download();
  status_check(expert::runtime::cuda::gated_gqa_attention_staged_device_fp16(
      {device_query.get(), nullptr, nullptr, output.get(), cache_capacity,
       first_context_tokens, 0U, rows, query_heads, kv_heads, head_dim,
       nullptr,
       reinterpret_cast<const void* const*>(device_page_table.get()),
       0U, page_tokens, page_count},
      {queries.get(), query_values * sizeof(std::uint16_t), raw_keys.get(),
       staged_values * sizeof(std::uint16_t), raw_values.get(),
       staged_values * sizeof(std::uint16_t), keys.get(),
       staged_values * sizeof(std::uint16_t), values.get(),
       staged_values * sizeof(std::uint16_t), scores.get(),
       score_values * sizeof(float), probabilities.get(),
       score_values * sizeof(std::uint16_t), accumulator.get(),
       query_values * sizeof(float), maxima.get(),
       static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
       sums.get(),
       static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
       split_tokens}));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize staged paged-device FP16 attention smoke");
  const auto paged_device_actual = output.download();

  std::vector<float> expected(query_values);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const auto context = first_context_tokens + row;
    for (std::uint32_t query_head = 0U; query_head < query_heads;
         ++query_head) {
      const auto kv_head = query_head / grouped_heads;
      std::vector<float> accumulated(head_dim, 0.0F);
      float maximum = -std::numeric_limits<float>::infinity();
      float denominator = 0.0F;
      for (std::uint32_t first = 0U; first < context;
           first += split_tokens) {
        const auto count = std::min(split_tokens, context - first);
        std::vector<float> tile_scores(count);
        float tile_maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0U; token < count; ++token) {
          float dot{};
          for (std::uint32_t dimension = 0U; dimension < head_dim;
               ++dimension) {
            const auto query_index =
                (static_cast<std::size_t>(row) * query_heads + query_head) *
                    2U * head_dim +
                dimension;
            const auto key_index =
                (static_cast<std::size_t>(first + token) * kv_heads +
                 kv_head) *
                    head_dim +
                dimension;
            dot += __half2float(__float2half(query_gate[query_index])) *
                   __half2float(host_keys.get()[key_index]);
          }
          tile_scores[token] = dot / std::sqrt(static_cast<float>(head_dim));
          tile_maximum = std::max(tile_maximum, tile_scores[token]);
        }
        const auto next_maximum = std::max(maximum, tile_maximum);
        const auto previous_scale = std::isfinite(maximum)
                                        ? std::exp(maximum - next_maximum)
                                        : 0.0F;
        for (auto& value : accumulated) value *= previous_scale;
        denominator *= previous_scale;
        for (std::uint32_t token = 0U; token < count; ++token) {
          const auto probability = __half2float(__float2half(
              std::exp(tile_scores[token] - next_maximum)));
          denominator += probability;
          for (std::uint32_t dimension = 0U; dimension < head_dim;
               ++dimension) {
            const auto value_index =
                (static_cast<std::size_t>(first + token) * kv_heads +
                 kv_head) *
                    head_dim +
                dimension;
            accumulated[dimension] +=
                probability * __half2float(host_values.get()[value_index]);
          }
        }
        maximum = next_maximum;
      }
      for (std::uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
        const auto query_base =
            (static_cast<std::size_t>(row) * query_heads + query_head) *
            2U * head_dim;
        const auto output_index =
            (static_cast<std::size_t>(row) * query_heads + query_head) *
                head_dim +
            dimension;
        expected[output_index] =
            (accumulated[dimension] / denominator) /
            (1.0F + std::exp(-query_gate[query_base + head_dim + dimension]));
      }
    }
  }
  double maximum_error{};
  for (std::size_t index = 0U; index < actual.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(actual[index] - expected[index])));
  for (std::size_t index = 0U; index < actual.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(device_actual[index] - actual[index])));
  for (std::size_t index = 0U; index < actual.size(); ++index)
    maximum_error = std::max(
        maximum_error,
        std::abs(static_cast<double>(paged_device_actual[index] -
                                    actual[index])));
  return maximum_error;
}

struct ResidentFp16AttentionProfile final {
  struct Split final {
    std::uint32_t split_tokens{};
    double milliseconds{};
    double raw_kv_gb_per_second{};
  };
  double best_milliseconds{};
  double best_raw_kv_gb_per_second{};
  std::uint32_t best_split_tokens{};
  std::vector<Split> splits;
};

ResidentFp16AttentionProfile resident_fp16_attention_262144_profile() {
  constexpr std::uint32_t context_tokens = 262'144U;
  constexpr std::uint32_t rows = 1U;
  constexpr std::uint32_t query_heads = 24U;
  constexpr std::uint32_t kv_heads = 4U;
  constexpr std::uint32_t head_dim = 256U;
  constexpr std::array<std::uint32_t, 6U> split_options{
      2'048U, 4'096U, 8'192U, 16'384U, 32'768U, 65'536U};
  constexpr std::uint32_t maximum_split = split_options.back();
  constexpr std::uint32_t grouped_heads = query_heads / kv_heads;
  constexpr std::uint32_t matrix_rows = rows * grouped_heads;
  constexpr std::size_t query_values =
      static_cast<std::size_t>(rows) * query_heads * head_dim;
  constexpr std::size_t cache_values =
      static_cast<std::size_t>(context_tokens) * kv_heads * head_dim;
  constexpr std::size_t staged_values =
      static_cast<std::size_t>(maximum_split) * kv_heads * head_dim;
  constexpr std::size_t score_values =
      static_cast<std::size_t>(kv_heads) * matrix_rows * maximum_split;
  constexpr std::uint64_t raw_kv_bytes =
      2ULL * cache_values * sizeof(std::uint16_t);
  constexpr unsigned iterations = 4U;

  std::vector<float> query_gate(query_values * 2U);
  for (std::size_t index = 0U; index < query_gate.size(); ++index)
    query_gate[index] =
        std::sin(static_cast<float>(index + 1U) * 0.0013F) * 0.2F;
  DeviceBuffer<float> device_query(query_gate.size());
  DeviceBuffer<float> output(query_values);
  DeviceBuffer<std::uint16_t> cache_keys(cache_values);
  DeviceBuffer<std::uint16_t> cache_values_buffer(cache_values);
  DeviceBuffer<std::uint16_t> queries(query_values);
  DeviceBuffer<std::uint16_t> keys(staged_values);
  DeviceBuffer<std::uint16_t> values(staged_values);
  DeviceBuffer<float> scores(score_values);
  DeviceBuffer<std::uint16_t> probabilities(score_values);
  DeviceBuffer<float> accumulator(query_values);
  DeviceBuffer<float> maxima(static_cast<std::size_t>(kv_heads) * matrix_rows);
  DeviceBuffer<float> sums(static_cast<std::size_t>(kv_heads) * matrix_rows);
  device_query.upload(query_gate);
  cuda_check(cudaMemset(cache_keys.get(), 0,
                        cache_values * sizeof(std::uint16_t)),
             "zero resident FP16 keys");
  cuda_check(cudaMemset(cache_values_buffer.get(), 0,
                        cache_values * sizeof(std::uint16_t)),
             "zero resident FP16 values");

  ResidentFp16AttentionProfile result;
  result.best_milliseconds = std::numeric_limits<double>::infinity();
  for (const auto split_tokens : split_options) {
    const expert::runtime::cuda::HostFp16GatedGqaAttentionWorkspace workspace{
        queries.get(), query_values * sizeof(std::uint16_t), nullptr, 0U,
        nullptr, 0U, keys.get(), staged_values * sizeof(std::uint16_t),
        values.get(), staged_values * sizeof(std::uint16_t), scores.get(),
        score_values * sizeof(float), probabilities.get(),
        score_values * sizeof(std::uint16_t), accumulator.get(),
        query_values * sizeof(float), maxima.get(),
        static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
        sums.get(),
        static_cast<std::size_t>(kv_heads) * matrix_rows * sizeof(float),
        split_tokens};
    const expert::runtime::cuda::DeviceFp16GatedGqaAttentionLaunch launch{
        device_query.get(), cache_keys.get(), cache_values_buffer.get(),
        output.get(), context_tokens, context_tokens, 0U, rows, query_heads,
        kv_heads, head_dim, nullptr};
    status_check(
        expert::runtime::cuda::gated_gqa_attention_staged_device_fp16(
            launch, workspace));
    cuda_check(cudaDeviceSynchronize(),
               "warm resident FP16 attention profile");
    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start),
               "create resident FP16 attention start");
    cuda_check(cudaEventCreate(&stop),
               "create resident FP16 attention stop");
    cuda_check(cudaEventRecord(start),
               "record resident FP16 attention start");
    for (unsigned iteration = 0U; iteration < iterations; ++iteration)
      status_check(
          expert::runtime::cuda::gated_gqa_attention_staged_device_fp16(
              launch, workspace));
    cuda_check(cudaEventRecord(stop),
               "record resident FP16 attention stop");
    cuda_check(cudaEventSynchronize(stop),
               "synchronize resident FP16 attention stop");
    float total_milliseconds{};
    cuda_check(cudaEventElapsedTime(&total_milliseconds, start, stop),
               "measure resident FP16 attention");
    static_cast<void>(cudaEventDestroy(start));
    static_cast<void>(cudaEventDestroy(stop));
    const auto milliseconds =
        static_cast<double>(total_milliseconds) / iterations;
    const auto gb_per_second =
        static_cast<double>(raw_kv_bytes) / (milliseconds * 1.0e6);
    result.splits.push_back({split_tokens, milliseconds, gb_per_second});
    if (milliseconds < result.best_milliseconds) {
      result.best_milliseconds = milliseconds;
      result.best_raw_kv_gb_per_second = gb_per_second;
      result.best_split_tokens = split_tokens;
    }
  }
  return result;
}

struct TopKLogitsProfile final {
  bool pass{};
  double milliseconds{};
};

TopKLogitsProfile topk_logits_check() {
  const std::vector<float> logits{
      1.0F, 7.0F, std::numeric_limits<float>::quiet_NaN(), 7.0F,
      -2.0F, 4.5F, 4.5F, 9.0F, 0.0F};
  DeviceBuffer<float> device_logits(logits.size());
  DeviceBuffer<float> device_values(5U);
  DeviceBuffer<std::uint32_t> device_indices(5U);
  const auto workspace_items =
      expert::runtime::cuda::topk_logits_workspace_items(logits.size());
  DeviceBuffer<float> workspace_values(workspace_items);
  DeviceBuffer<std::uint32_t> workspace_indices(workspace_items);
  device_logits.upload(logits);
  status_check(expert::runtime::cuda::topk_logits(
      device_logits.get(), static_cast<std::uint32_t>(logits.size()), 5U,
      device_values.get(), device_indices.get(),
      {workspace_values.get(), workspace_items * sizeof(float),
       workspace_indices.get(), workspace_items * sizeof(std::uint32_t)},
      nullptr));
  cuda_check(cudaDeviceSynchronize(), "synchronize top-k logits smoke");
  const auto values = device_values.download();
  const auto indices = device_indices.download();
  const auto small_pass =
      indices == std::vector<std::uint32_t>({7U, 1U, 3U, 5U, 6U}) &&
      values == std::vector<float>({9.0F, 7.0F, 7.0F, 4.5F, 4.5F});

  constexpr std::uint32_t vocabulary = 248320U;
  constexpr std::uint32_t top_k = 20U;
  constexpr std::uint32_t iterations = 32U;
  std::vector<float> large_logits(vocabulary);
  for (std::uint32_t index = 0U; index < vocabulary; ++index)
    large_logits[index] =
        std::sin(static_cast<float>(index) * 0.0137F) +
        0.37F * std::cos(static_cast<float>(index) * 0.0071F);
  large_logits[19U] = 7.0F;
  large_logits[127U] = 7.0F;
  large_logits[1021U] = std::numeric_limits<float>::quiet_NaN();
  std::vector<std::uint32_t> expected(vocabulary);
  for (std::uint32_t index = 0U; index < vocabulary; ++index)
    expected[index] = index;
  std::stable_sort(expected.begin(), expected.end(), [&](auto left, auto right) {
    const auto left_value = large_logits[left];
    const auto right_value = large_logits[right];
    if (std::isnan(left_value)) return false;
    if (std::isnan(right_value)) return true;
    if (left_value != right_value) return left_value > right_value;
    return left < right;
  });
  expected.resize(top_k);

  DeviceBuffer<float> large_device_logits(vocabulary);
  DeviceBuffer<float> large_device_values(top_k);
  DeviceBuffer<std::uint32_t> large_device_indices(top_k);
  const auto large_workspace_items =
      expert::runtime::cuda::topk_logits_workspace_items(vocabulary);
  DeviceBuffer<float> large_workspace_values(large_workspace_items);
  DeviceBuffer<std::uint32_t> large_workspace_indices(large_workspace_items);
  large_device_logits.upload(large_logits);
  const expert::runtime::cuda::TopKLogitsWorkspace large_workspace{
      large_workspace_values.get(), large_workspace_items * sizeof(float),
      large_workspace_indices.get(),
      large_workspace_items * sizeof(std::uint32_t)};
  for (std::uint32_t warmup = 0U; warmup < 3U; ++warmup)
    status_check(expert::runtime::cuda::topk_logits(
        large_device_logits.get(), vocabulary, top_k,
        large_device_values.get(), large_device_indices.get(),
        large_workspace, nullptr));
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "create top-k start event");
  cuda_check(cudaEventCreate(&stop), "create top-k stop event");
  cuda_check(cudaEventRecord(start), "record top-k start");
  for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration)
    status_check(expert::runtime::cuda::topk_logits(
        large_device_logits.get(), vocabulary, top_k,
        large_device_values.get(), large_device_indices.get(),
        large_workspace, nullptr));
  cuda_check(cudaEventRecord(stop), "record top-k stop");
  cuda_check(cudaEventSynchronize(stop), "synchronize top-k stop");
  float total_milliseconds{};
  cuda_check(cudaEventElapsedTime(&total_milliseconds, start, stop),
             "measure top-k elapsed time");
  static_cast<void>(cudaEventDestroy(start));
  static_cast<void>(cudaEventDestroy(stop));
  return {small_pass && large_device_indices.download() == expected,
          static_cast<double>(total_milliseconds) / iterations};
}

bool presence_penalty_check() {
  const std::vector<float> logits{3.0F, 2.0F, 1.0F, -1.0F};
  const std::vector<std::uint8_t> emitted{0U, 1U, 1U, 0U};
  DeviceBuffer<float> device_logits(logits.size());
  DeviceBuffer<std::uint8_t> device_emitted(emitted.size());
  device_logits.upload(logits);
  device_emitted.upload(emitted);
  status_check(expert::runtime::cuda::apply_presence_penalty(
      device_logits.get(), device_emitted.get(),
      static_cast<std::uint32_t>(logits.size()), 1.5F, nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize presence-penalty smoke");
  return device_logits.download() ==
         std::vector<float>({3.0F, 0.5F, -0.5F, -1.0F});
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) ==
                         "--q4-bfp-batch-5-profile") {
      const auto numerical = q4_bfp_key_outlier1_check();
      const std::array profiles{
          q4_bfp_attention_profile(32768U),
          q4_bfp_attention_profile(131072U)};
      const auto pass =
          numerical.layout_maximum_byte_difference == 0.0 &&
          numerical.query_maximum_byte_difference == 0.0 &&
          numerical.query_scale_maximum_absolute_difference < 1.0e-8 &&
          numerical.independent_oracle_maximum_absolute_difference < 2.0e-4 &&
          profiles[0].sixteen_layer_milliseconds <= 11.45 &&
          profiles[1].sixteen_layer_milliseconds <= 23.21;
      std::cout
          << "{\"pass\":" << (pass ? "true" : "false")
          << ",\"layout_maximum_byte_difference\":"
          << numerical.layout_maximum_byte_difference
          << ",\"query_maximum_byte_difference\":"
          << numerical.query_maximum_byte_difference
          << ",\"query_scale_maximum_absolute_difference\":"
          << numerical.query_scale_maximum_absolute_difference
          << ",\"independent_oracle_maximum_absolute_difference\":"
          << numerical.independent_oracle_maximum_absolute_difference
          << ",\"profiles\":[";
      for (std::size_t index = 0U; index < profiles.size(); ++index) {
        if (index != 0U) std::cout << ',';
        const auto& profile = profiles[index];
        std::cout << "{\"context_tokens\":" << profile.context_tokens
                  << ",\"milliseconds\":" << profile.milliseconds
                  << ",\"sixteen_layer_milliseconds\":"
                  << profile.sixteen_layer_milliseconds
                  << ",\"kv_gb_per_second\":"
                  << profile.kv_gb_per_second << '}';
      }
      std::cout << "]}\n";
      return pass ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) ==
                         "--q5-q4-bfp-batch-5-profile") {
      const auto numerical = q5_q4_bfp_check();
      const std::array profiles{
          q5_q4_bfp_attention_profile(32768U),
          q5_q4_bfp_attention_profile(131072U)};
      const auto pass =
          numerical.layout_maximum_byte_difference == 0.0 &&
          numerical.query_maximum_byte_difference == 0.0 &&
          numerical.query_scale_maximum_absolute_difference < 1.0e-8 &&
          numerical.independent_oracle_maximum_absolute_difference < 2.0e-4 &&
          std::isfinite(profiles[0].sixteen_layer_milliseconds) &&
          profiles[0].sixteen_layer_milliseconds > 0.0 &&
          profiles[1].sixteen_layer_milliseconds <= 16.7527;
      std::cout
          << "{\"pass\":" << (pass ? "true" : "false")
          << ",\"layout_maximum_byte_difference\":"
          << numerical.layout_maximum_byte_difference
          << ",\"query_maximum_byte_difference\":"
          << numerical.query_maximum_byte_difference
          << ",\"query_scale_maximum_absolute_difference\":"
          << numerical.query_scale_maximum_absolute_difference
          << ",\"independent_oracle_maximum_absolute_difference\":"
          << numerical.independent_oracle_maximum_absolute_difference
          << ",\"profiles\":[";
      for (std::size_t index = 0U; index < profiles.size(); ++index) {
        if (index != 0U) std::cout << ',';
        const auto& profile = profiles[index];
        std::cout << "{\"context_tokens\":" << profile.context_tokens
                  << ",\"milliseconds\":" << profile.milliseconds
                  << ",\"sixteen_layer_milliseconds\":"
                  << profile.sixteen_layer_milliseconds
                  << ",\"kv_gb_per_second\":"
                  << profile.kv_gb_per_second << '}';
      }
      std::cout << "]}\n";
      return pass ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) ==
                         "--q4-bfp-plain-batch-5-profile") {
      const auto numerical = q4_bfp_check();
      const std::array profiles{
          q4_bfp_plain_attention_profile(32768U),
          q4_bfp_plain_attention_profile(131072U)};
      const auto pass =
          numerical.layout_maximum_byte_difference == 0.0 &&
          numerical.query_maximum_byte_difference == 0.0 &&
          numerical.query_scale_maximum_absolute_difference < 1.0e-8 &&
          numerical.independent_oracle_maximum_absolute_difference < 2.0e-4 &&
          std::isfinite(profiles[0].sixteen_layer_milliseconds) &&
          profiles[0].sixteen_layer_milliseconds > 0.0 &&
          profiles[1].sixteen_layer_milliseconds <= 16.7527;
      std::cout
          << "{\"pass\":" << (pass ? "true" : "false")
          << ",\"layout_maximum_byte_difference\":"
          << numerical.layout_maximum_byte_difference
          << ",\"query_maximum_byte_difference\":"
          << numerical.query_maximum_byte_difference
          << ",\"query_scale_maximum_absolute_difference\":"
          << numerical.query_scale_maximum_absolute_difference
          << ",\"independent_oracle_maximum_absolute_difference\":"
          << numerical.independent_oracle_maximum_absolute_difference
          << ",\"profiles\":[";
      for (std::size_t index = 0U; index < profiles.size(); ++index) {
        if (index != 0U) std::cout << ',';
        const auto& profile = profiles[index];
        std::cout << "{\"context_tokens\":" << profile.context_tokens
                  << ",\"milliseconds\":" << profile.milliseconds
                  << ",\"sixteen_layer_milliseconds\":"
                  << profile.sixteen_layer_milliseconds
                  << ",\"kv_gb_per_second\":"
                  << profile.kv_gb_per_second << '}';
      }
      std::cout << "]}\n";
      return pass ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) ==
                         "--q4-per-head-batch-5-profile") {
      const auto numerical = q4_per_head_check();
      const std::array profiles{
          q4_per_head_attention_profile(32768U),
          q4_per_head_attention_profile(131072U)};
      const auto pass =
          numerical.layout_maximum_byte_difference == 0.0 &&
          numerical.query_maximum_byte_difference == 0.0 &&
          numerical.query_scale_maximum_absolute_difference < 1.0e-8 &&
          numerical.independent_oracle_maximum_absolute_difference < 2.0e-4 &&
          std::isfinite(profiles[0].sixteen_layer_milliseconds) &&
          profiles[0].sixteen_layer_milliseconds > 0.0 &&
          profiles[1].sixteen_layer_milliseconds <= 16.7527;
      std::cout
          << "{\"pass\":" << (pass ? "true" : "false")
          << ",\"layout_maximum_byte_difference\":"
          << numerical.layout_maximum_byte_difference
          << ",\"query_maximum_byte_difference\":"
          << numerical.query_maximum_byte_difference
          << ",\"query_scale_maximum_absolute_difference\":"
          << numerical.query_scale_maximum_absolute_difference
          << ",\"independent_oracle_maximum_absolute_difference\":"
          << numerical.independent_oracle_maximum_absolute_difference
          << ",\"profiles\":[";
      for (std::size_t index = 0U; index < profiles.size(); ++index) {
        if (index != 0U) std::cout << ',';
        const auto& profile = profiles[index];
        std::cout << "{\"context_tokens\":" << profile.context_tokens
                  << ",\"milliseconds\":" << profile.milliseconds
                  << ",\"sixteen_layer_milliseconds\":"
                  << profile.sixteen_layer_milliseconds
                  << ",\"kv_gb_per_second\":"
                  << profile.kv_gb_per_second << '}';
      }
      std::cout << "]}\n";
      return pass ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) ==
                         "--decode-batch-5-profile") {
      const auto maximum_absolute_error = numerical_check();
      const auto narrow = decode_batch_bandwidth_check(5U);
      const auto wide = wide_decode_batch_bandwidth_check(5U);
      std::cout << "{\"batch\":5"
                << ",\"maximum_absolute_error\":"
                << maximum_absolute_error
                << ",\"narrow_weight_reuse_gb_per_second\":"
                << narrow.selected_gb_per_second
                << ",\"narrow_tensor_core_gb_per_second\":"
                << narrow.tensor_core_gb_per_second
                << ",\"wide_weight_reuse_gb_per_second\":" << wide
                << "}\n";
      return maximum_absolute_error < 2.0e-4 ? 0 : 1;
    }
    const auto error = numerical_check();
    const auto topk = topk_logits_check();
    const auto presence_penalty_pass = presence_penalty_check();
    const auto delta_error = delta_prefill_check();
    const auto attention_error = attention_prefill_check();
    const auto fp8_kv_attention = fp8_kv_attention_check();
    const auto q8_kv_attention = q8_kv_attention_check();
    const auto fp4_key_outlier1 = fp4_key_outlier1_check();
    const auto q4_bfp_key_outlier1 = q4_bfp_key_outlier1_check();
    const auto q5_q4_bfp = q5_q4_bfp_check();
    const auto standard_gqa_ratio16_error = standard_gqa_ratio16_check();
    const auto gated_gqa_ratio12_error = gated_gqa_ratio12_check();
    const auto standard_gqa_no_position_error =
        standard_gqa_no_position_check();
    const auto abi2_errors = dense_fp4_abi2_kernel_check();
    const auto host_fp16_attention_error = host_fp16_attention_check();
    const auto resident_fp16_attention =
        resident_fp16_attention_262144_profile();
    const auto bandwidth = bandwidth_check();
    const auto batch_bandwidth = batch_bandwidth_check();
    const auto decode_batch_bandwidth = decode_batch_bandwidth_check();
    const auto wide_decode_batch_bandwidth =
        wide_decode_batch_bandwidth_check();
    const auto attention_milliseconds =
        attention_prefill_4096_milliseconds();
    const auto long_context_attention = attention_262144_profile();
    const auto fp4_key_outlier1_attention =
        fp4_key_outlier1_attention_262144_profile();
    // The official model executes attention operands in BF16. Keep the
    // scalar-FP32 comparison as a reported drift measurement, but use the
    // same strict 2e-4 bound as the FP4/BF16 dense execution contract.
    const auto attention_profiles_pass = std::all_of(
        long_context_attention.decode_profiles.begin(),
        long_context_attention.decode_profiles.end(), [](const auto& item) {
          return std::isfinite(item.milliseconds) &&
                 item.milliseconds > 0.0 &&
                 std::isfinite(item.kv_gb_per_second) &&
                 item.kv_gb_per_second > 0.0 &&
                 item.maximum_absolute_difference < 2.0e-4;
        });
    const bool pass = topk.pass && topk.milliseconds <= 7.09 &&
                      presence_penalty_pass &&
                      error < 2.0e-4 &&
                      delta_error < 2.0e-5 &&
                      attention_error < 2.0e-4 &&
                      fp8_kv_attention.
                              query_maximum_absolute_difference <
                          1.0e-6 &&
                      fp8_kv_attention.
                              implementation_maximum_absolute_difference <
                          2.0e-4 &&
                      q8_kv_attention.layout_maximum_byte_difference == 0.0 &&
                      q8_kv_attention.
                              independent_oracle_maximum_absolute_difference <
                          2.0e-3 &&
                      q8_kv_attention.
                              five_query_maximum_absolute_difference <
                          2.0e-3 &&
                      fp4_key_outlier1.layout_maximum_byte_difference ==
                          0.0 &&
                      fp4_key_outlier1.
                              implementation_maximum_absolute_difference <
                          2.0e-3 &&
                      fp4_key_outlier1.
                              prefill_implementation_maximum_absolute_difference <
                          2.0e-3 &&
                      fp4_key_outlier1.
                              candidate_fp16_maximum_absolute_difference <
                          fp4_key_outlier1.
                              ordinary_fp4_fp16_maximum_absolute_difference &&
                      q4_bfp_key_outlier1.
                              layout_maximum_byte_difference == 0.0 &&
                      q4_bfp_key_outlier1.
                              query_maximum_byte_difference == 0.0 &&
                      q4_bfp_key_outlier1.
                              query_scale_maximum_absolute_difference <
                          1.0e-8 &&
                      q4_bfp_key_outlier1.
                              independent_oracle_maximum_absolute_difference <
                          2.0e-4 &&
                      q5_q4_bfp.layout_maximum_byte_difference == 0.0 &&
                      q5_q4_bfp.query_maximum_byte_difference == 0.0 &&
                      q5_q4_bfp.
                              query_scale_maximum_absolute_difference <
                          1.0e-8 &&
                      q5_q4_bfp.
                              independent_oracle_maximum_absolute_difference <
                          2.0e-4 &&
                      std::isfinite(fp8_kv_attention.
                                        output_maximum_absolute_difference) &&
                      std::isfinite(fp8_kv_attention.
                                        output_mean_absolute_difference) &&
                      std::isfinite(
                          fp8_kv_attention.output_cosine_similarity) &&
                      standard_gqa_ratio16_error < 5.0e-4 &&
                      gated_gqa_ratio12_error < 5.0e-4 &&
                      standard_gqa_no_position_error < 5.0e-4 &&
                      abi2_errors.weightless_rms < 2.0e-5 &&
                      abi2_errors.sigmoid_product < 2.0e-6 &&
                      abi2_errors.scaled_tanh < 2.0e-6 &&
                      abi2_errors.normalized_gqa < 5.0e-4 &&
                      host_fp16_attention_error < 5.0e-4 &&
                      std::isfinite(
                          resident_fp16_attention.best_milliseconds) &&
                      resident_fp16_attention.best_milliseconds > 0.0 &&
                      std::isfinite(bandwidth) && bandwidth > 0.0 &&
                      std::isfinite(batch_bandwidth.
                                        current_effective_weight_gb_per_second) &&
                      batch_bandwidth.current_effective_weight_gb_per_second >
                          0.0 &&
                      std::isfinite(batch_bandwidth.current_milliseconds) &&
                      batch_bandwidth.current_milliseconds > 0.0 &&
                      std::isfinite(
                          batch_bandwidth.staged_decode_milliseconds) &&
                      batch_bandwidth.staged_decode_milliseconds > 0.0 &&
                      std::isfinite(
                          batch_bandwidth.staged_gemm_milliseconds) &&
                      batch_bandwidth.staged_gemm_milliseconds > 0.0 &&
                      std::isfinite(
                          decode_batch_bandwidth.selected_gb_per_second) &&
                      decode_batch_bandwidth.selected_gb_per_second >
                          0.0 &&
                      std::isfinite(
                          decode_batch_bandwidth.tensor_core_gb_per_second) &&
                      decode_batch_bandwidth.tensor_core_gb_per_second >
                          0.0 &&
                      std::isfinite(wide_decode_batch_bandwidth) &&
                      wide_decode_batch_bandwidth > 0.0 &&
                      std::isfinite(attention_milliseconds) &&
                      attention_milliseconds > 0.0 &&
                      std::isfinite(
                          long_context_attention.prefill_milliseconds) &&
                      long_context_attention.prefill_milliseconds > 0.0 &&
                      long_context_attention.
                              prefill_maximum_absolute_difference <
                          2.0e-4 &&
                      std::isfinite(
                          long_context_attention.decode_milliseconds) &&
                      long_context_attention.decode_milliseconds > 0.0 &&
                      std::isfinite(
                          long_context_attention.decode_kv_gb_per_second) &&
                      long_context_attention.decode_kv_gb_per_second > 0.0 &&
                      std::isfinite(long_context_attention.
                                        tensor_core_decode_milliseconds) &&
                      long_context_attention.tensor_core_decode_milliseconds >
                          0.0 &&
                      std::isfinite(long_context_attention.
                                        tensor_core_decode_kv_gb_per_second) &&
                      long_context_attention.
                              tensor_core_decode_kv_gb_per_second >
                          0.0 &&
                      long_context_attention.
                              tensor_core_maximum_absolute_difference <
                          2.0e-4 &&
                      std::isfinite(long_context_attention.
                                        two_position_prefill_milliseconds) &&
                      long_context_attention.two_position_prefill_milliseconds >
                          0.0 &&
                      std::isfinite(long_context_attention.
                                        two_position_microbatch_milliseconds) &&
                      long_context_attention.
                              two_position_microbatch_milliseconds >
                          0.0 &&
                      long_context_attention.
                              two_position_maximum_absolute_difference <
                          2.0e-4 &&
                      std::isfinite(fp4_key_outlier1_attention.
                                        sixteen_layer_milliseconds) &&
                      fp4_key_outlier1_attention.
                              sixteen_layer_milliseconds <=
                          32.5 &&
                      attention_profiles_pass;
    std::cout << "{\"pass\":" << (pass ? "true" : "false")
              << ",\"topk_logits_pass\":"
              << (topk.pass ? "true" : "false")
              << ",\"topk_logits_248320_milliseconds\":"
              << topk.milliseconds
              << ",\"presence_penalty_pass\":"
              << (presence_penalty_pass ? "true" : "false")
              << ",\"maximum_absolute_error\":" << error
              << ",\"delta_prefill_maximum_absolute_error\":"
              << delta_error
              << ",\"attention_prefill_maximum_absolute_error\":"
              << attention_error
              << ",\"fp8_kv_query_maximum_absolute_difference\":"
              << fp8_kv_attention.query_maximum_absolute_difference
              << ",\"fp8_kv_implementation_maximum_absolute_difference\":"
              << fp8_kv_attention.
                     implementation_maximum_absolute_difference
              << ",\"fp8_kv_output_maximum_absolute_difference\":"
              << fp8_kv_attention.output_maximum_absolute_difference
              << ",\"fp8_kv_output_mean_absolute_difference\":"
              << fp8_kv_attention.output_mean_absolute_difference
              << ",\"fp8_kv_output_cosine_similarity\":"
              << fp8_kv_attention.output_cosine_similarity
              << ",\"q8_kv_layout_maximum_byte_difference\":"
              << q8_kv_attention.layout_maximum_byte_difference
              << ",\"q8_kv_independent_oracle_maximum_absolute_difference\":"
              << q8_kv_attention.
                     independent_oracle_maximum_absolute_difference
              << ",\"q8_kv_five_query_maximum_absolute_difference\":"
              << q8_kv_attention.
                     five_query_maximum_absolute_difference
              << ",\"fp4_key_outlier1_layout_maximum_byte_difference\":"
              << fp4_key_outlier1.layout_maximum_byte_difference
              << ",\"fp4_key_outlier1_implementation_maximum_absolute_difference\":"
              << fp4_key_outlier1.
                     implementation_maximum_absolute_difference
              << ",\"fp4_key_outlier1_prefill_implementation_maximum_absolute_difference\":"
              << fp4_key_outlier1.
                     prefill_implementation_maximum_absolute_difference
              << ",\"fp4_key_outlier1_fp16_maximum_absolute_difference\":"
              << fp4_key_outlier1.
                     candidate_fp16_maximum_absolute_difference
              << ",\"ordinary_fp4_fp16_maximum_absolute_difference\":"
              << fp4_key_outlier1.
                     ordinary_fp4_fp16_maximum_absolute_difference
              << ",\"q4_bfp_key_outlier1_layout_maximum_byte_difference\":"
              << q4_bfp_key_outlier1.layout_maximum_byte_difference
              << ",\"q4_bfp_key_outlier1_query_maximum_byte_difference\":"
              << q4_bfp_key_outlier1.query_maximum_byte_difference
              << ",\"q4_bfp_key_outlier1_query_scale_maximum_absolute_difference\":"
              << q4_bfp_key_outlier1.
                     query_scale_maximum_absolute_difference
              << ",\"q4_bfp_key_outlier1_independent_oracle_maximum_absolute_difference\":"
              << q4_bfp_key_outlier1.
                     independent_oracle_maximum_absolute_difference
              << ",\"q5_q4_bfp_layout_maximum_byte_difference\":"
              << q5_q4_bfp.layout_maximum_byte_difference
              << ",\"q5_q4_bfp_query_maximum_byte_difference\":"
              << q5_q4_bfp.query_maximum_byte_difference
              << ",\"q5_q4_bfp_query_scale_maximum_absolute_difference\":"
              << q5_q4_bfp.query_scale_maximum_absolute_difference
              << ",\"q5_q4_bfp_independent_oracle_maximum_absolute_difference\":"
              << q5_q4_bfp.independent_oracle_maximum_absolute_difference
              << ",\"standard_gqa_ratio16_maximum_absolute_error\":"
              << standard_gqa_ratio16_error
              << ",\"gated_gqa_ratio12_maximum_absolute_error\":"
              << gated_gqa_ratio12_error
              << ",\"standard_gqa_no_position_maximum_absolute_error\":"
              << standard_gqa_no_position_error
              << ",\"abi2_weightless_rms_maximum_absolute_error\":"
              << abi2_errors.weightless_rms
              << ",\"abi2_sigmoid_product_maximum_absolute_error\":"
              << abi2_errors.sigmoid_product
              << ",\"abi2_scaled_tanh_maximum_absolute_error\":"
              << abi2_errors.scaled_tanh
              << ",\"abi2_normalized_gqa_maximum_absolute_error\":"
              << abi2_errors.normalized_gqa
              << ",\"host_fp16_attention_maximum_absolute_error\":"
              << host_fp16_attention_error
              << ",\"resident_fp16_attention_262144_best_milliseconds\":"
              << resident_fp16_attention.best_milliseconds
              << ",\"resident_fp16_attention_262144_best_raw_kv_gb_per_second\":"
              << resident_fp16_attention.best_raw_kv_gb_per_second
              << ",\"resident_fp16_attention_262144_best_split_tokens\":"
              << resident_fp16_attention.best_split_tokens
              << ",\"effective_weight_gb_per_second\":" << bandwidth
              << ",\"effective_batch_weight_gb_per_second\":"
              << batch_bandwidth.current_effective_weight_gb_per_second
              << ",\"prefill_gemm_current_milliseconds\":"
              << batch_bandwidth.current_milliseconds
              << ",\"prefill_gemm_staged_decode_milliseconds\":"
              << batch_bandwidth.staged_decode_milliseconds
              << ",\"prefill_gemm_staged_milliseconds\":"
              << batch_bandwidth.staged_gemm_milliseconds
              << ",\"decode_batch_selected_gb_per_second\":"
              << decode_batch_bandwidth.selected_gb_per_second
              << ",\"decode_batch_tensor_core_gb_per_second\":"
              << decode_batch_bandwidth.tensor_core_gb_per_second
              << ",\"wide_decode_batch_selected_gb_per_second\":"
              << wide_decode_batch_bandwidth
              << ",\"attention_prefill_4096_milliseconds\":"
              << attention_milliseconds
              << ",\"attention_prefill_262144_milliseconds\":"
              << long_context_attention.prefill_milliseconds
              << ",\"attention_prefill_262144_maximum_absolute_difference\":"
              << long_context_attention.
                     prefill_maximum_absolute_difference
              << ",\"attention_decode_262144_milliseconds\":"
              << long_context_attention.decode_milliseconds
              << ",\"attention_decode_262144_kv_gb_per_second\":"
              << long_context_attention.decode_kv_gb_per_second
              << ",\"attention_tensor_core_decode_262144_milliseconds\":"
              << long_context_attention.tensor_core_decode_milliseconds
              << ",\"attention_tensor_core_decode_262144_kv_gb_per_second\":"
              << long_context_attention.tensor_core_decode_kv_gb_per_second
              << ",\"attention_tensor_core_maximum_absolute_difference\":"
              << long_context_attention.
                     tensor_core_maximum_absolute_difference
              << ",\"attention_two_position_prefill_262144_milliseconds\":"
              << long_context_attention.two_position_prefill_milliseconds
              << ",\"attention_two_position_microbatch_262144_milliseconds\":"
              << long_context_attention.two_position_microbatch_milliseconds
              << ",\"attention_two_position_maximum_absolute_difference\":"
              << long_context_attention.
                     two_position_maximum_absolute_difference
              << ",\"attention_fp4_key_outlier1_262144_milliseconds\":"
              << fp4_key_outlier1_attention.milliseconds
              << ",\"attention_fp4_key_outlier1_262144_kv_gb_per_second\":"
              << fp4_key_outlier1_attention.kv_gb_per_second
              << ",\"attention_fp4_key_outlier1_262144_sixteen_layer_milliseconds\":"
              << fp4_key_outlier1_attention.sixteen_layer_milliseconds
              << ",\"attention_decode_262144_profiles\":[";
    for (std::size_t index = 0U;
         index < long_context_attention.decode_profiles.size(); ++index) {
      if (index != 0U) std::cout << ',';
      const auto& profile = long_context_attention.decode_profiles[index];
      std::cout << "{\"split_tokens\":" << profile.split_tokens
                << ",\"milliseconds\":" << profile.milliseconds
                << ",\"kv_gb_per_second\":" << profile.kv_gb_per_second
                << ",\"maximum_absolute_difference\":"
                << profile.maximum_absolute_difference << '}';
    }
    std::cout << "],\"resident_fp16_attention_262144_profiles\":[";
    for (std::size_t index = 0U;
         index < resident_fp16_attention.splits.size(); ++index) {
      if (index != 0U) std::cout << ',';
      const auto& profile = resident_fp16_attention.splits[index];
      std::cout << "{\"split_tokens\":" << profile.split_tokens
                << ",\"milliseconds\":" << profile.milliseconds
                << ",\"raw_kv_gb_per_second\":"
                << profile.raw_kv_gb_per_second << '}';
    }
    std::cout << "]}\n";
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
