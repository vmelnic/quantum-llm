#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kRows = 7U;
constexpr std::uint32_t kColumns = 33U;
constexpr std::uint32_t kPaddedColumns = 64U;
constexpr std::uint32_t kBatch = 5U;

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
               "allocate MXFP6 smoke buffer");
  }
  ~DeviceBuffer() { static_cast<void>(cudaFree(pointer_)); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() const noexcept { return pointer_; }

  void upload(const std::vector<T>& values) {
    if (values.size() != count_)
      throw std::runtime_error("MXFP6 smoke upload size mismatch");
    cuda_check(cudaMemcpy(pointer_, values.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upload MXFP6 smoke buffer");
  }

  [[nodiscard]] std::vector<T> download() const {
    std::vector<T> values(count_);
    cuda_check(cudaMemcpy(values.data(), pointer_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "download MXFP6 smoke buffer");
    return values;
  }

 private:
  T* pointer_{};
  std::size_t count_{};
};

float decode_e3m2(std::uint8_t code) {
  const auto magnitude = code & 0x1fU;
  const auto exponent = magnitude >> 2U;
  const auto mantissa = magnitude & 0x03U;
  const auto value = exponent == 0U
      ? static_cast<float>(mantissa) / 16.0F
      : std::ldexp(1.0F + static_cast<float>(mantissa) / 4.0F,
                   static_cast<int>(exponent) - 3);
  return (code & 0x20U) != 0U ? -value : value;
}

void pack_code(std::vector<std::uint8_t>& payload, std::uint32_t row,
               std::uint32_t column, std::uint8_t code) {
  const auto group = static_cast<std::size_t>(row) *
                         (kPaddedColumns / 4U) +
                     column / 4U;
  const auto shift = 6U * (column & 3U);
  const auto base = group * 3U;
  auto word = static_cast<std::uint32_t>(payload[base]) |
              (static_cast<std::uint32_t>(payload[base + 1U]) << 8U) |
              (static_cast<std::uint32_t>(payload[base + 2U]) << 16U);
  word |= static_cast<std::uint32_t>(code) << shift;
  payload[base] = static_cast<std::uint8_t>(word);
  payload[base + 1U] = static_cast<std::uint8_t>(word >> 8U);
  payload[base + 2U] = static_cast<std::uint8_t>(word >> 16U);
}

float maximum_error(const std::vector<float>& actual,
                    const std::vector<float>& expected) {
  if (actual.size() != expected.size())
    throw std::runtime_error("MXFP6 smoke output size mismatch");
  float result{};
  for (std::size_t index = 0U; index < actual.size(); ++index)
    result = std::max(result, std::abs(actual[index] - expected[index]));
  return result;
}

}  // namespace

int main() {
  try {
    std::vector<std::uint8_t> weights(
        static_cast<std::size_t>(kRows) * kPaddedColumns * 3U / 4U);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(kRows) * kPaddedColumns / 32U);
    std::vector<float> decoded(
        static_cast<std::size_t>(kRows) * kPaddedColumns);
    for (std::uint32_t row = 0U; row < kRows; ++row) {
      for (std::uint32_t block = 0U; block < kPaddedColumns / 32U; ++block) {
        const auto scale_code = static_cast<std::uint8_t>(126U +
                                                          ((row + block) & 1U));
        scales[static_cast<std::size_t>(row) * 2U + block] = scale_code;
        const auto scale = std::ldexp(1.0F,
                                      static_cast<int>(scale_code) - 127);
        for (std::uint32_t local = 0U; local < 32U; ++local) {
          const auto column = block * 32U + local;
          auto code = static_cast<std::uint8_t>(
              (row * 11U + column * 7U) & 0x1fU);
          if ((row + column) % 3U == 0U) code |= 0x20U;
          pack_code(weights, row, column, code);
          decoded[static_cast<std::size_t>(row) * kPaddedColumns + column] =
              decode_e3m2(code) * scale;
        }
      }
    }

    std::vector<std::int8_t> input(
        static_cast<std::size_t>(kBatch) * kPaddedColumns);
    std::vector<float> input_scales(kBatch);
    for (std::uint32_t request = 0U; request < kBatch; ++request) {
      input_scales[request] = 0.03125F * static_cast<float>(request + 1U);
      for (std::uint32_t column = 0U; column < kColumns; ++column)
        input[static_cast<std::size_t>(request) * kPaddedColumns + column] =
            static_cast<std::int8_t>(
                static_cast<int>((request * 19U + column * 13U) % 255U) -
                127);
    }
    std::vector<float> expected_gemv(
        static_cast<std::size_t>(kBatch) * kRows);
    for (std::uint32_t request = 0U; request < kBatch; ++request)
      for (std::uint32_t row = 0U; row < kRows; ++row) {
        float total{};
        for (std::uint32_t column = 0U; column < kColumns; ++column)
          total += decoded[static_cast<std::size_t>(row) * kPaddedColumns +
                           column] *
                   static_cast<float>(
                       input[static_cast<std::size_t>(request) *
                                 kPaddedColumns +
                             column]) *
                   input_scales[request];
        expected_gemv[static_cast<std::size_t>(request) * kRows + row] =
            total;
      }

    DeviceBuffer<std::uint8_t> device_weights(weights.size());
    DeviceBuffer<std::uint8_t> device_scales(scales.size());
    DeviceBuffer<std::int8_t> device_input(input.size());
    DeviceBuffer<float> device_input_scales(input_scales.size());
    DeviceBuffer<float> device_gemv(expected_gemv.size());
    device_weights.upload(weights);
    device_scales.upload(scales);
    device_input.upload(input);
    device_input_scales.upload(input_scales);
    const expert::runtime::cuda::Mxfp6E3m2Block32Matrix matrix{
        device_weights.get(), device_scales.get(), kRows, kColumns,
        kPaddedColumns};
    status_check(expert::runtime::cuda::mxfp6_gemv_q8_batch_weight_reuse(
        matrix, device_input.get(), device_input_scales.get(),
        device_gemv.get(), kBatch, nullptr));

    std::vector<std::uint32_t> tokens{0U, 3U, 6U};
    std::vector<float> expected_embedding(tokens.size() * kColumns);
    for (std::size_t request = 0U; request < tokens.size(); ++request)
      std::copy_n(decoded.data() +
                      static_cast<std::size_t>(tokens[request]) *
                          kPaddedColumns,
                  kColumns,
                  expected_embedding.data() + request * kColumns);
    DeviceBuffer<std::uint32_t> device_tokens(tokens.size());
    DeviceBuffer<float> device_embedding(expected_embedding.size());
    device_tokens.upload(tokens);
    status_check(expert::runtime::cuda::mxfp6_embedding_batch(
        matrix, device_tokens.get(), device_embedding.get(),
        static_cast<std::uint32_t>(tokens.size()), nullptr));
    cuda_check(cudaDeviceSynchronize(), "synchronize MXFP6 smoke");

    const auto gemv_error = maximum_error(device_gemv.download(),
                                          expected_gemv);
    const auto embedding_error = maximum_error(device_embedding.download(),
                                               expected_embedding);
    if (gemv_error > 2.0e-3F || embedding_error != 0.0F)
      throw std::runtime_error("MXFP6 CUDA numerical gate failed");
    std::cout << "{\"mxfp6_gemv_max_abs\":" << gemv_error
              << ",\"mxfp6_embedding_max_abs\":" << embedding_error
              << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
