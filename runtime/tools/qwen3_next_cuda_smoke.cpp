#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
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

double check_full_attention() {
  constexpr std::uint32_t query_heads = 16, kv_heads = 2, dim = 256;
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
  DeviceBuffer<float> d_k_cache(k.size()), d_v_cache(v.size()), d_output(expected.size());
  d_q.upload(q); d_k.upload(k); d_v.upload(v); d_norm.upload(norm);
  status_check(expert::runtime::cuda::qwen3_next_qkv_rope_cache(
      d_q.get(), d_k.get(), d_v.get(), d_norm.get(), d_norm.get(),
      d_k_cache.get(), d_v_cache.get(), 0, query_heads, kv_heads, dim, 64,
      1.0e-6F, 10000000.0F, nullptr));
  status_check(expert::runtime::cuda::qwen3_next_attention_decode(
      d_q.get(), d_k_cache.get(), d_v_cache.get(), d_output.get(), 1,
      query_heads, kv_heads, dim, nullptr));
  cuda_check(cudaDeviceSynchronize(), "attention synchronize");
  return maximum_error(d_output.download(), expected);
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

}  // namespace

int main() {
  try {
    const auto rms = check_rms();
    const auto attention = check_full_attention();
    const auto delta = check_delta();
    const bool valid = rms < 2.0e-6 && attention < 2.0e-6 && delta < 2.0e-5;
    std::cout << "{\"valid\":" << (valid ? "true" : "false")
              << ",\"qwen_rms_max_abs\":" << rms
              << ",\"full_attention_max_abs\":" << attention
              << ",\"delta_max_abs\":" << delta << "}\n";
    return valid ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3-Next CUDA smoke: " << error.what() << '\n';
    return 1;
  }
}
