#include "expert/runtime/cuda/moe_kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using expert::runtime::cuda::DeviceExpertEntry;
using expert::runtime::cuda::DeviceExpertFormat;
using expert::runtime::cuda::DeviceExpertState;
using expert::runtime::cuda::MoeGroupedSelectionWork;

constexpr std::uint32_t kExperts = 3U;
constexpr std::uint32_t kRows = 8U;
constexpr std::uint32_t kTopK = 2U;
constexpr std::uint32_t kHidden = 64U;
constexpr std::uint32_t kWidth = 32U;

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
                          count_ * sizeof(T)),
               "allocate grouped FP4 smoke buffer");
  }
  ~DeviceBuffer() { static_cast<void>(cudaFree(pointer_)); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const noexcept { return pointer_; }
  void upload(const std::vector<T>& values) {
    if (values.size() != count_)
      throw std::runtime_error("grouped FP4 smoke upload size mismatch");
    cuda_check(cudaMemcpy(pointer_, values.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upload grouped FP4 smoke buffer");
  }
  std::vector<T> download() const {
    std::vector<T> result(count_);
    cuda_check(cudaMemcpy(result.data(), pointer_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "download grouped FP4 smoke buffer");
    return result;
  }

 private:
  T* pointer_{};
  std::size_t count_{};
};

float decode_fp4(std::uint8_t code) {
  constexpr float values[]{0.0F, 0.5F, 1.0F, 1.5F,
                           2.0F, 3.0F, 4.0F, 6.0F};
  const auto value = values[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

float decode_scale(std::uint8_t code) {
  return code == 0U ? std::ldexp(1.0F, -127)
                    : std::ldexp(1.0F, static_cast<int>(code) - 127);
}

std::int8_t quantize(float value, float scale) {
  auto result = static_cast<int>(std::nearbyint(value / scale));
  result = std::clamp(result, -127, 127);
  return static_cast<std::int8_t>(result);
}

std::vector<std::int8_t> quantize_rows(const std::vector<float>& values,
                                       std::uint32_t rows,
                                       std::uint32_t columns,
                                       std::vector<float>& scales) {
  std::vector<std::int8_t> result(values.size());
  scales.resize(rows);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    float maximum{};
    for (std::uint32_t column = 0U; column < columns; ++column)
      maximum = std::max(
          maximum,
          std::abs(values[static_cast<std::size_t>(row) * columns + column]));
    scales[row] = maximum > 0.0F ? maximum / 127.0F : 1.0F;
    for (std::uint32_t column = 0U; column < columns; ++column)
      result[static_cast<std::size_t>(row) * columns + column] = quantize(
          values[static_cast<std::size_t>(row) * columns + column], scales[row]);
  }
  return result;
}

float dot(const std::vector<std::uint8_t>& packed,
          const std::vector<std::uint8_t>& scales,
          std::size_t packed_row, std::size_t scale_row,
          const std::int8_t* activation, float activation_scale,
          std::uint32_t columns) {
  float total{};
  for (std::uint32_t column = 0U; column < columns; ++column) {
    const auto byte = packed[packed_row + column / 2U];
    const auto code = static_cast<std::uint8_t>(
        (column & 1U) == 0U ? byte & 15U : byte >> 4U);
    total += decode_fp4(code) *
             decode_scale(scales[scale_row + column / 32U]) *
             static_cast<float>(activation[column]);
  }
  return total * activation_scale;
}

}  // namespace

int main() {
  try {
    constexpr auto selections = kRows * kTopK;
    constexpr auto gate_packed_per_expert =
        static_cast<std::size_t>(kWidth) * kHidden / 2U;
    constexpr auto gate_scales_per_expert =
        static_cast<std::size_t>(kWidth) * kHidden / 32U;
    constexpr auto down_packed_per_expert =
        static_cast<std::size_t>(kHidden) * kWidth / 2U;
    constexpr auto down_scales_per_expert =
        static_cast<std::size_t>(kHidden) * kWidth / 32U;

    std::vector<std::uint8_t> w1(kExperts * gate_packed_per_expert);
    std::vector<std::uint8_t> w3(kExperts * gate_packed_per_expert);
    std::vector<std::uint8_t> w2(kExperts * down_packed_per_expert);
    std::vector<std::uint8_t> s1(kExperts * gate_scales_per_expert);
    std::vector<std::uint8_t> s3(kExperts * gate_scales_per_expert);
    std::vector<std::uint8_t> s2(kExperts * down_scales_per_expert);
    for (std::size_t index = 0U; index < w1.size(); ++index) {
      w1[index] = static_cast<std::uint8_t>(
          ((index * 3U + 1U) & 15U) | (((index * 5U + 2U) & 15U) << 4U));
      w3[index] = static_cast<std::uint8_t>(
          ((index * 7U + 3U) & 15U) | (((index * 11U + 4U) & 15U) << 4U));
    }
    for (std::size_t index = 0U; index < w2.size(); ++index)
      w2[index] = static_cast<std::uint8_t>(
          ((index * 13U + 5U) & 15U) | (((index * 17U + 6U) & 15U) << 4U));
    for (std::size_t index = 0U; index < s1.size(); ++index) {
      s1[index] = static_cast<std::uint8_t>(123U + index % 5U);
      s3[index] = static_cast<std::uint8_t>(122U + index % 6U);
    }
    for (std::size_t index = 0U; index < s2.size(); ++index)
      s2[index] = static_cast<std::uint8_t>(123U + index % 5U);

    std::vector<float> input(static_cast<std::size_t>(kRows) * kHidden);
    for (std::size_t index = 0U; index < input.size(); ++index)
      input[index] = std::sin(static_cast<float>(index) * 0.037F) * 0.75F;
    const std::vector<std::uint32_t> indices{
        0U, 1U, 0U, 2U, 1U, 0U, 2U, 0U,
        0U, 1U, 1U, 2U, 2U, 0U, 0U, 1U};
    std::vector<float> routing(selections, 0.5F);

    std::vector<float> input_scales;
    const auto q8_input = quantize_rows(input, kRows, kHidden, input_scales);
    std::vector<float> intermediate(
        static_cast<std::size_t>(selections) * kWidth);
    for (std::uint32_t selection = 0U; selection < selections; ++selection) {
      const auto expert = indices[selection];
      const auto row = selection / kTopK;
      const auto packed_base = expert * gate_packed_per_expert;
      const auto scale_base = expert * gate_scales_per_expert;
      for (std::uint32_t output = 0U; output < kWidth; ++output) {
        const auto gate = dot(
            w1, s1, packed_base + static_cast<std::size_t>(output) * kHidden / 2U,
            scale_base + static_cast<std::size_t>(output) * kHidden / 32U,
            q8_input.data() + static_cast<std::size_t>(row) * kHidden,
            input_scales[row], kHidden);
        const auto up = dot(
            w3, s3, packed_base + static_cast<std::size_t>(output) * kHidden / 2U,
            scale_base + static_cast<std::size_t>(output) * kHidden / 32U,
            q8_input.data() + static_cast<std::size_t>(row) * kHidden,
            input_scales[row], kHidden);
        intermediate[static_cast<std::size_t>(selection) * kWidth + output] =
            (gate / (1.0F + std::exp(-gate))) * up;
      }
    }
    std::vector<float> intermediate_scales;
    const auto q8_intermediate = quantize_rows(
        intermediate, selections, kWidth, intermediate_scales);
    std::vector<float> expected(
        static_cast<std::size_t>(selections) * kHidden);
    for (std::uint32_t selection = 0U; selection < selections; ++selection) {
      const auto expert = indices[selection];
      const auto packed_base = expert * down_packed_per_expert;
      const auto scale_base = expert * down_scales_per_expert;
      for (std::uint32_t output = 0U; output < kHidden; ++output)
        expected[static_cast<std::size_t>(selection) * kHidden + output] = dot(
            w2, s2, packed_base + static_cast<std::size_t>(output) * kWidth / 2U,
            scale_base + static_cast<std::size_t>(output) * kWidth / 32U,
            q8_intermediate.data() +
                static_cast<std::size_t>(selection) * kWidth,
            intermediate_scales[selection], kWidth);
    }

    std::vector<MoeGroupedSelectionWork> work;
    std::map<std::uint32_t, std::vector<std::uint32_t>> by_expert;
    for (std::uint32_t selection = 0U; selection < selections; ++selection)
      by_expert[indices[selection]].push_back(selection);
    for (const auto& [expert, selected] : by_expert)
      for (std::size_t first = 0U; first < selected.size();
           first += expert::runtime::cuda::kMoeGroupedSelectionWidth) {
        MoeGroupedSelectionWork item{};
        item.expert = expert;
        item.count = static_cast<std::uint32_t>(std::min<std::size_t>(
            expert::runtime::cuda::kMoeGroupedSelectionWidth,
            selected.size() - first));
        std::copy_n(selected.data() + first, item.count, item.selections);
        work.push_back(item);
      }

    DeviceBuffer<std::uint8_t> d_w1(w1.size()), d_w3(w3.size()), d_w2(w2.size());
    DeviceBuffer<std::uint8_t> d_s1(s1.size()), d_s3(s3.size()), d_s2(s2.size());
    DeviceBuffer<float> d_input(input.size()), d_routing(routing.size());
    DeviceBuffer<std::uint32_t> d_indices(indices.size());
    DeviceBuffer<float> d_intermediate(intermediate.size());
    DeviceBuffer<float> d_output(expected.size());
    DeviceBuffer<std::int8_t> d_q8_input(q8_input.size());
    DeviceBuffer<float> d_q8_input_scales(input_scales.size());
    DeviceBuffer<std::int8_t> d_q8_intermediate(q8_intermediate.size());
    DeviceBuffer<float> d_q8_intermediate_scales(intermediate_scales.size());
    DeviceBuffer<MoeGroupedSelectionWork> d_work(work.size());
    DeviceBuffer<DeviceExpertEntry> d_directory(kExperts);
    d_w1.upload(w1); d_w3.upload(w3); d_w2.upload(w2);
    d_s1.upload(s1); d_s3.upload(s3); d_s2.upload(s2);
    d_input.upload(input); d_routing.upload(routing); d_indices.upload(indices);
    d_work.upload(work);
    std::vector<DeviceExpertEntry> directory(kExperts);
    for (std::uint32_t expert = 0U; expert < kExperts; ++expert) {
      auto& entry = directory[expert];
      entry.w1_fp4 = d_w1.get() + expert * gate_packed_per_expert;
      entry.w1_ue8m0 = d_s1.get() + expert * gate_scales_per_expert;
      entry.w3_fp4 = d_w3.get() + expert * gate_packed_per_expert;
      entry.w3_ue8m0 = d_s3.get() + expert * gate_scales_per_expert;
      entry.w2_fp4 = d_w2.get() + expert * down_packed_per_expert;
      entry.w2_ue8m0 = d_s2.get() + expert * down_scales_per_expert;
      entry.format = static_cast<std::uint32_t>(
          DeviceExpertFormat::fp4_e2m1_ue8m0_block32);
      entry.state = static_cast<std::uint32_t>(DeviceExpertState::ready);
    }
    d_directory.upload(directory);

    status_check(expert::runtime::cuda::launch_moe_selection_batch({
        d_input.get(), d_routing.get(), d_indices.get(), nullptr,
        d_intermediate.get(), d_output.get(), d_q8_input.get(),
        d_q8_input_scales.get(), d_q8_intermediate.get(),
        d_q8_intermediate_scales.get(), kRows, kHidden, kWidth, kTopK,
        kExperts, nullptr, d_directory.get(), 0U, 0.0F, false, true, false,
        nullptr, nullptr, nullptr, d_work.get(),
        static_cast<std::uint32_t>(work.size()), true}));
    cuda_check(cudaDeviceSynchronize(), "synchronize grouped FP4 smoke");
    const auto actual = d_output.download();
    double maximum_absolute_error{};
    double dot_product{}, expected_square{}, actual_square{};
    for (std::size_t index = 0U; index < actual.size(); ++index) {
      maximum_absolute_error = std::max(
          maximum_absolute_error,
          std::abs(static_cast<double>(actual[index]) - expected[index]));
      dot_product += static_cast<double>(actual[index]) * expected[index];
      expected_square += static_cast<double>(expected[index]) * expected[index];
      actual_square += static_cast<double>(actual[index]) * actual[index];
    }
    const auto cosine = dot_product / std::sqrt(expected_square * actual_square);
    const auto pass = maximum_absolute_error <= 1.0e-3 && cosine >= 0.999999;
    std::cout << "{\"pass\":" << (pass ? "true" : "false")
              << ",\"work_items\":" << work.size()
              << ",\"selections\":" << selections
              << ",\"maximum_absolute_error\":" << maximum_absolute_error
              << ",\"cosine\":" << cosine << "}\n";
    return pass ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "grouped FP4 MoE CUDA smoke: " << error.what() << '\n';
    return 1;
  }
}
