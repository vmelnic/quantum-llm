#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/sha256.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kHidden = 16U;
constexpr std::uint32_t kIntermediate = 16U;

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
               "allocate native NVFP4 smoke buffer");
  }
  ~DeviceBuffer() { static_cast<void>(cudaFree(pointer_)); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() const noexcept { return pointer_; }

  void upload(const std::vector<T>& values) {
    if (values.size() != count_)
      throw std::runtime_error("native NVFP4 smoke upload size mismatch");
    cuda_check(cudaMemcpy(pointer_, values.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice),
               "upload native NVFP4 smoke buffer");
  }

  [[nodiscard]] std::vector<T> download() const {
    std::vector<T> values(count_);
    cuda_check(cudaMemcpy(values.data(), pointer_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "download native NVFP4 smoke buffer");
    return values;
  }

 private:
  T* pointer_{};
  std::size_t count_{};
};

float decode_e2m1(std::uint8_t code) {
  constexpr float values[8]{0.0F, 0.5F, 1.0F, 1.5F,
                            2.0F, 3.0F, 4.0F, 6.0F};
  const auto value = values[code & 0x07U];
  return (code & 0x08U) != 0U ? -value : value;
}

float decode_e4m3fn(std::uint8_t code) {
  const auto negative = (code & 0x80U) != 0U;
  const auto exponent = (code >> 3U) & 0x0fU;
  const auto mantissa = code & 0x07U;
  if (exponent == 0x0fU && mantissa == 0x07U)
    return std::numeric_limits<float>::quiet_NaN();
  const auto value = exponent == 0U
      ? std::ldexp(static_cast<float>(mantissa) / 8.0F, -6)
      : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                   static_cast<int>(exponent) - 7);
  return negative ? -value : value;
}

float round_e4m3fn(float value) {
  float best{};
  float best_error = std::numeric_limits<float>::infinity();
  unsigned best_code{};
  for (unsigned code = 0U; code < 256U; ++code) {
    const auto candidate = decode_e4m3fn(static_cast<std::uint8_t>(code));
    if (!std::isfinite(candidate)) continue;
    const auto error = std::abs(value - candidate);
    if (error < best_error ||
        (error == best_error && (code & 1U) == 0U &&
         (best_code & 1U) != 0U)) {
      best = candidate;
      best_error = error;
      best_code = code;
    }
  }
  return best;
}

float round_e2m1(float value) {
  constexpr float values[8]{0.0F, 0.5F, 1.0F, 1.5F,
                            2.0F, 3.0F, 4.0F, 6.0F};
  const auto sign = value < 0.0F ? -1.0F : 1.0F;
  const auto magnitude = std::abs(value);
  float best{};
  float best_error = std::numeric_limits<float>::infinity();
  std::size_t best_code{};
  for (std::size_t code = 0U; code < std::size(values); ++code) {
    const auto candidate = values[code];
    const auto error = std::abs(magnitude - candidate);
    if (error < best_error ||
        (error == best_error && (code & 1U) == 0U &&
         (best_code & 1U) != 0U)) {
      best = candidate;
      best_error = error;
      best_code = code;
    }
  }
  return sign * best;
}

float round_bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  bits &= 0xffff0000U;
  return std::bit_cast<float>(bits);
}

std::vector<float> quantize_dequantize(const std::vector<float>& input,
                                       float global_scale) {
  std::vector<float> output(input.size());
  for (std::size_t base = 0U; base < input.size(); base += 16U) {
    float maximum{};
    for (std::size_t index = base; index < base + 16U; ++index)
      maximum = std::max(maximum, std::abs(input[index]));
    const auto local = round_e4m3fn(
        std::min(448.0F, global_scale * maximum / 6.0F));
    for (std::size_t index = base; index < base + 16U; ++index) {
      const auto scaled = local == 0.0F
          ? 0.0F
          : std::clamp(input[index] * global_scale / local, -6.0F, 6.0F);
      output[index] = round_e2m1(scaled) * local / global_scale;
    }
  }
  return output;
}

float matrix_value(const std::vector<std::uint8_t>& packed,
                   const std::vector<std::uint8_t>& sidecar,
                   std::uint32_t row, std::uint32_t column,
                   std::uint32_t rows, std::uint32_t columns) {
  const auto byte = packed[static_cast<std::size_t>(row) * columns / 2U +
                           column / 2U];
  const auto code = static_cast<std::uint8_t>(
      (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
  float global_divisor{};
  const auto local_scales = static_cast<std::size_t>(rows) * columns / 16U;
  std::memcpy(&global_divisor, sidecar.data() + local_scales,
              sizeof(global_divisor));
  return decode_e2m1(code) *
         decode_e4m3fn(sidecar[static_cast<std::size_t>(row) *
                               columns / 16U + column / 16U]) /
         global_divisor;
}

std::vector<std::uint8_t> make_weights(unsigned seed) {
  std::vector<std::uint8_t> packed(kHidden * kHidden / 2U);
  for (std::uint32_t row = 0U; row < kHidden; ++row) {
    for (std::uint32_t pair = 0U; pair < kHidden / 2U; ++pair) {
      const auto low = static_cast<std::uint8_t>(
          1U + (seed + row + 3U * pair) % 7U);
      const auto high = static_cast<std::uint8_t>(
          (1U + (seed + 2U * row + pair) % 7U) |
          (((seed + row + pair) & 1U) != 0U ? 8U : 0U));
      packed[static_cast<std::size_t>(row) * kHidden / 2U + pair] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  return packed;
}

std::vector<std::uint8_t> make_sidecar(float weight_global_divisor,
                                       float input_global_divisor) {
  std::vector<std::uint8_t> sidecar(kHidden + 2U * sizeof(float), 0x38U);
  std::memcpy(sidecar.data() + kHidden, &weight_global_divisor,
              sizeof(weight_global_divisor));
  std::memcpy(sidecar.data() + kHidden + sizeof(float),
              &input_global_divisor, sizeof(input_global_divisor));
  return sidecar;
}

std::vector<float> reference(const std::vector<float>& input,
                             const std::vector<std::uint8_t>& gate_weights,
                             const std::vector<std::uint8_t>& gate_sidecar,
                             const std::vector<std::uint8_t>& up_weights,
                             const std::vector<std::uint8_t>& up_sidecar,
                             const std::vector<std::uint8_t>& down_weights,
                             const std::vector<std::uint8_t>& down_sidecar,
                             std::uint32_t hidden = kHidden,
                             std::uint32_t intermediate = kIntermediate) {
  const auto gate_local = static_cast<std::size_t>(intermediate) * hidden / 16U;
  const auto down_local = static_cast<std::size_t>(hidden) * intermediate / 16U;
  float gate_input_divisor{};
  float up_input_divisor{};
  float down_input_divisor{};
  const auto input_divisor = [](const std::vector<std::uint8_t>& sidecar,
                                std::size_t local) {
    if (sidecar.size() == local + 2U * sizeof(float)) {
      float result{};
      std::memcpy(&result, sidecar.data() + local + sizeof(float),
                  sizeof(result));
      return result;
    }
    if (sidecar.size() == local + sizeof(float) + sizeof(std::uint16_t)) {
      std::uint16_t bits{};
      std::memcpy(&bits, sidecar.data() + local + sizeof(float),
                  sizeof(bits));
      return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    throw std::runtime_error("native NVFP4 input divisor encoding is invalid");
  };
  gate_input_divisor = input_divisor(gate_sidecar, gate_local);
  up_input_divisor = input_divisor(up_sidecar, gate_local);
  down_input_divisor = input_divisor(down_sidecar, down_local);
  const auto gate_input =
      quantize_dequantize(input, gate_input_divisor);
  const auto up_input =
      quantize_dequantize(input, up_input_divisor);
  std::vector<float> intermediate_values(intermediate);
  for (std::uint32_t row = 0U; row < intermediate; ++row) {
    float gate{};
    float up{};
    for (std::uint32_t column = 0U; column < hidden; ++column) {
      gate += matrix_value(gate_weights, gate_sidecar, row, column,
                           intermediate, hidden) *
              gate_input[column];
      up += matrix_value(up_weights, up_sidecar, row, column,
                         intermediate, hidden) *
            up_input[column];
    }
    gate = round_bf16(gate);
    up = round_bf16(up);
    intermediate_values[row] =
        round_bf16((gate / (1.0F + std::exp(-gate))) * up);
  }
  const auto down_input =
      quantize_dequantize(intermediate_values, down_input_divisor);
  std::vector<float> output(hidden);
  for (std::uint32_t row = 0U; row < hidden; ++row) {
    float value{};
    for (std::uint32_t column = 0U; column < intermediate; ++column)
      value += matrix_value(down_weights, down_sidecar, row, column,
                            hidden, intermediate) *
               down_input[column];
    output[row] = round_bf16(value);
  }
  return output;
}

template <typename T>
T read_scalar(const std::vector<std::byte>& bytes, std::size_t offset) {
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
    throw std::runtime_error("native NVFP4 record header is truncated");
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

std::vector<std::byte> read_range(const std::filesystem::path& path,
                                  std::uint64_t offset,
                                  std::uint64_t byte_count) {
  if (!byte_count || byte_count > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("native NVFP4 record range is invalid");
  std::ifstream input(path, std::ios::binary);
  if (!input || offset >
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamoff>::max()))
    throw std::runtime_error("native NVFP4 pack cannot be opened");
  input.seekg(static_cast<std::streamoff>(offset));
  std::vector<std::byte> result(static_cast<std::size_t>(byte_count));
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size()));
  if (!input || static_cast<std::size_t>(input.gcount()) != result.size())
    throw std::runtime_error("native NVFP4 pack range is truncated");
  return result;
}

std::vector<float> read_f32(const std::filesystem::path& path,
                            std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  std::vector<float> result(count);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size() * sizeof(float)));
  if (!input || static_cast<std::size_t>(input.gcount()) !=
                    result.size() * sizeof(float) ||
      input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("native NVFP4 input fixture has the wrong size");
  return result;
}

struct Comparison final {
  double maximum_absolute_error{};
  double maximum_relative_error{};
  double relative_l2{};
  double cosine{};
};

Comparison compare(const std::vector<float>& actual,
                   const std::vector<float>& expected) {
  if (actual.size() != expected.size() || actual.empty())
    throw std::runtime_error("native NVFP4 comparison shape mismatch");
  Comparison result;
  double dot{};
  double expected_norm{};
  double actual_norm{};
  double error_norm{};
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    const auto delta = static_cast<double>(actual[index]) - expected[index];
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, std::abs(delta));
    result.maximum_relative_error = std::max(
        result.maximum_relative_error,
        std::abs(delta) /
            std::max(1.0e-5, std::abs(static_cast<double>(expected[index]))));
    dot += static_cast<double>(actual[index]) * expected[index];
    actual_norm += static_cast<double>(actual[index]) * actual[index];
    expected_norm += static_cast<double>(expected[index]) * expected[index];
    error_norm += delta * delta;
  }
  if (expected_norm == 0.0) {
    result.cosine = actual_norm == 0.0 ? 1.0 : 0.0;
    result.relative_l2 = actual_norm == 0.0
                             ? 0.0
                             : std::numeric_limits<double>::infinity();
  } else {
    result.cosine = dot / std::sqrt(actual_norm * expected_norm);
    result.relative_l2 = std::sqrt(error_norm / expected_norm);
  }
  return result;
}

int run_artifact_gate(const std::filesystem::path& pack,
                      std::uint64_t offset, std::uint64_t stored_bytes,
                      std::uint32_t layer, std::uint32_t expert,
                      const std::filesystem::path& input_path) {
  using namespace expert::runtime;
  auto bytes = read_range(pack, offset, stored_bytes);
  const auto hidden = read_scalar<std::uint32_t>(bytes, 28U);
  const auto intermediate = read_scalar<std::uint32_t>(bytes, 32U);
  PayloadRecord expected;
  expected.stored_bytes = stored_bytes;
  expected.decoded_bytes = 3ULL * hidden * intermediate * sizeof(float);
  expected.device_bytes = read_scalar<std::uint64_t>(bytes, 60U) +
                          read_scalar<std::uint64_t>(bytes, 76U) +
                          read_scalar<std::uint64_t>(bytes, 92U) +
                          read_scalar<std::uint64_t>(bytes, 108U);
  expected.hidden = hidden;
  expected.intermediate = intermediate;
  expected.quant_block_size = kExpertNvfp4BlockSize;
  expected.source_abi = kExpertSourceAbiExpertPackV1;
  expected.record_abi = kExpertRecordAbiNvfp4Block16W4A4;
  expected.header_bytes = kExpertHeaderBytes;
  expected.alignment = kExpertPackAlignment;
  std::copy_n(bytes.data() + 116U, expected.payload_sha256.size(),
              expected.payload_sha256.begin());
  const ExpertKey key{1U, layer, expert,
                      kExpertEncodingAbiNvfp4Block16W4A4};
  const auto validated = validate_expert_record(bytes, key, expected);
  status_check(validated.status);
  const auto& sections = validated.record.sections;
  const auto matrix_bytes = static_cast<std::size_t>(hidden) * intermediate / 2U;
  const auto matrix_scales = static_cast<std::size_t>(hidden) * intermediate /
                                 kExpertNvfp4BlockSize +
                             2U * sizeof(float);
  const auto copy_section = [&](std::uint64_t section_offset,
                                std::size_t count) {
    const auto* first = reinterpret_cast<const std::uint8_t*>(
        bytes.data() + section_offset);
    return std::vector<std::uint8_t>(first, first + count);
  };
  const auto gate_weights = copy_section(sections.gate_up_q_offset,
                                         matrix_bytes);
  const auto up_weights = copy_section(sections.gate_up_q_offset + matrix_bytes,
                                       matrix_bytes);
  const auto gate_sidecar = copy_section(sections.gate_up_scale_offset,
                                         matrix_scales);
  const auto up_sidecar = copy_section(
      sections.gate_up_scale_offset + matrix_scales, matrix_scales);
  const auto down_weights = copy_section(sections.down_q_offset, matrix_bytes);
  const auto down_sidecar = copy_section(sections.down_scale_offset,
                                         matrix_scales);
  const auto input = read_f32(input_path, hidden);
  const auto expected_output = reference(
      input, gate_weights, gate_sidecar, up_weights, up_sidecar,
      down_weights, down_sidecar, hidden, intermediate);

  cuda::CudaExpertUploader uploader;
  std::promise<UploadResult> promise;
  auto future = promise.get_future();
  uploader.upload(
      {key, kExpertSourceAbiExpertPackV1,
       kExpertRecordAbiNvfp4Block16W4A4, sections, bytes, {}},
      [&promise](UploadResult result) mutable {
        promise.set_value(std::move(result));
      });
  auto uploaded = future.get();
  status_check(uploaded.status);
  cuda::CudaExpertDirectory directory(1U, key.encoding_abi, layer + 1U,
                                      expert + 1U, 1U);
  status_check(directory.publish(key, uploaded.allocation));

  DeviceBuffer<float> device_input(hidden);
  DeviceBuffer<float> device_intermediate(intermediate);
  DeviceBuffer<float> device_selection_output(hidden);
  DeviceBuffer<float> device_gate_input(hidden);
  DeviceBuffer<float> device_up_input(hidden);
  DeviceBuffer<float> device_down_input(intermediate);
  DeviceBuffer<std::uint32_t> device_indices(1U);
  device_input.upload(input);
  device_indices.upload({expert});
  status_check(cuda::launch_moe_selection_batch({
      device_input.get(), nullptr, device_indices.get(), nullptr,
      device_intermediate.get(), device_selection_output.get(), nullptr,
      nullptr, nullptr, nullptr, 1U, hidden, intermediate, 1U, expert + 1U,
      nullptr, directory.device_entries(), layer, 0.0F, true, false, true,
      device_gate_input.get(), device_up_input.get(),
      device_down_input.get()}));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize native NVFP4 artifact gate");
  const auto actual = device_selection_output.download();
  const auto metrics = compare(actual, expected_output);
  const auto pass = metrics.relative_l2 < 0.01 && metrics.cosine > 0.9999;
  std::cout << "{\"mode\":\"artifact\",\"pass\":"
            << (pass ? "true" : "false")
            << ",\"maximum_absolute_error\":"
            << metrics.maximum_absolute_error
            << ",\"maximum_relative_error\":"
            << metrics.maximum_relative_error
            << ",\"relative_l2\":" << metrics.relative_l2
            << ",\"cosine\":" << metrics.cosine << "}\n";
  return pass ? 0 : 2;
}

struct DenseRecord final {
  std::uint32_t rows{};
  std::uint32_t columns{};
  std::vector<std::uint8_t> weights;
  std::vector<std::uint8_t> sidecar;
};

DenseRecord read_dense_record(const std::filesystem::path& pack,
                              std::uint64_t offset,
                              std::uint64_t stored_bytes) {
  auto bytes = read_range(pack, offset, stored_bytes);
  constexpr std::array<std::byte, 8U> magic{
      std::byte{'E'}, std::byte{'P'}, std::byte{'D'}, std::byte{'E'},
      std::byte{'N'}, std::byte{'S'}, std::byte{'0'}, std::byte{'1'}};
  const auto version = read_scalar<std::uint16_t>(bytes, 8U);
  const auto header_bytes = read_scalar<std::uint16_t>(bytes, 10U);
  const auto flags = read_scalar<std::uint32_t>(bytes, 12U);
  const auto quant_abi = read_scalar<std::uint32_t>(bytes, 16U);
  const auto rank = read_scalar<std::uint32_t>(bytes, 20U);
  const auto rows = read_scalar<std::uint32_t>(bytes, 24U);
  const auto columns = read_scalar<std::uint32_t>(bytes, 28U);
  const auto record_bytes = read_scalar<std::uint64_t>(bytes, 44U);
  const auto data_offset = read_scalar<std::uint64_t>(bytes, 52U);
  const auto data_bytes = read_scalar<std::uint64_t>(bytes, 60U);
  const auto scale_offset = read_scalar<std::uint64_t>(bytes, 68U);
  const auto scale_bytes = read_scalar<std::uint64_t>(bytes, 76U);
  const auto local_scales = static_cast<std::uint64_t>(rows) * columns /
                            expert::runtime::kExpertNvfp4BlockSize;
  if (!std::equal(magic.begin(), magic.end(), bytes.begin()) || version != 1U ||
      header_bytes != expert::runtime::kExpertHeaderBytes || flags != 5U ||
      quant_abi != expert::runtime::kExpertRecordAbiNvfp4Block16W4A4 ||
      rank != 2U || !rows || !columns || columns % 16U ||
      record_bytes != stored_bytes || data_offset != header_bytes ||
      data_bytes != static_cast<std::uint64_t>(rows) * columns / 2U ||
      scale_bytes != local_scales + sizeof(float) + sizeof(std::uint16_t) ||
      data_offset > stored_bytes || data_bytes > stored_bytes - data_offset ||
      scale_offset > stored_bytes || scale_bytes > stored_bytes - scale_offset)
    throw std::runtime_error("native NVFP4 dense record header is invalid");
  expert::runtime::Sha256Digest expected{};
  std::copy_n(bytes.data() + 116U, expected.size(), expected.begin());
  if (!expert::runtime::constant_time_equal(
          expert::runtime::sha256(
              std::span<const std::byte>(bytes).subspan(header_bytes)),
          expected))
    throw std::runtime_error("native NVFP4 dense payload hash mismatch");
  const auto copy = [&](std::uint64_t first, std::uint64_t count) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(
        bytes.data() + first);
    return std::vector<std::uint8_t>(begin, begin + count);
  };
  return {rows, columns, copy(data_offset, data_bytes),
          copy(scale_offset, scale_bytes)};
}

struct DeviceDenseRecord final {
  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint8_t> sidecar;
  float weight_scale{};
  float input_scale{};
  std::uint32_t rows{};
  std::uint32_t columns{};

  explicit DeviceDenseRecord(const DenseRecord& source)
      : weights(source.weights.size()), sidecar(source.sidecar.size()),
        rows(source.rows), columns(source.columns) {
    weights.upload(source.weights);
    sidecar.upload(source.sidecar);
    const auto local = static_cast<std::size_t>(rows) * columns / 16U;
    float weight_divisor{};
    std::memcpy(&weight_divisor, source.sidecar.data() + local,
                sizeof(weight_divisor));
    std::uint16_t input_bits{};
    std::memcpy(&input_bits,
                source.sidecar.data() + local + sizeof(float),
                sizeof(input_bits));
    const auto input_divisor = std::bit_cast<float>(
        static_cast<std::uint32_t>(input_bits) << 16U);
    if (!(weight_divisor > 0.0F) || !(input_divisor > 0.0F))
      throw std::runtime_error("native NVFP4 dense divisor is invalid");
    weight_scale = 1.0F / weight_divisor;
    input_scale = input_divisor;
  }

  [[nodiscard]] expert::runtime::cuda::Nvfp4Block16Matrix view() const {
    return {weights.get(), sidecar.get(), weight_scale, input_scale,
            rows, columns};
  }
};

int run_shared_gate(const std::filesystem::path& pack,
                    std::uint64_t gate_offset,
                    std::uint64_t up_offset,
                    std::uint64_t down_offset,
                    std::uint64_t stored_bytes,
                    const std::filesystem::path& input_path) {
  const auto gate = read_dense_record(pack, gate_offset, stored_bytes);
  const auto up = read_dense_record(pack, up_offset, stored_bytes);
  const auto down = read_dense_record(pack, down_offset, stored_bytes);
  if (gate.rows != up.rows || gate.columns != up.columns ||
      down.rows != gate.columns || down.columns != gate.rows)
    throw std::runtime_error("native NVFP4 shared MLP geometry is invalid");
  const auto input = read_f32(input_path, gate.columns);
  const auto expected = reference(
      input, gate.weights, gate.sidecar, up.weights, up.sidecar,
      down.weights, down.sidecar, gate.columns, gate.rows);
  DeviceDenseRecord device_gate(gate);
  DeviceDenseRecord device_up(up);
  DeviceDenseRecord device_down(down);
  DeviceBuffer<float> device_input(input.size());
  DeviceBuffer<float> device_gate_output(gate.rows);
  DeviceBuffer<float> device_up_output(up.rows);
  DeviceBuffer<float> device_intermediate(gate.rows);
  DeviceBuffer<float> device_output(down.rows);
  DeviceBuffer<float> device_quantized(
      std::max<std::uint32_t>(gate.columns, gate.rows));
  device_input.upload(input);
  status_check(expert::runtime::cuda::nvfp4_gemv_f32_batch(
      device_gate.view(), device_input.get(), device_gate_output.get(),
      device_quantized.get(), 1U, nullptr));
  status_check(expert::runtime::cuda::nvfp4_gemv_f32_batch(
      device_up.view(), device_input.get(), device_up_output.get(),
      device_quantized.get(), 1U, nullptr));
  status_check(expert::runtime::cuda::silu_product(
      device_gate_output.get(), device_up_output.get(),
      device_intermediate.get(), gate.rows, nullptr));
  status_check(expert::runtime::cuda::round_bf16_in_place(
      device_intermediate.get(), gate.rows, nullptr));
  status_check(expert::runtime::cuda::nvfp4_gemv_f32_batch(
      device_down.view(), device_intermediate.get(), device_output.get(),
      device_quantized.get(), 1U, nullptr));
  cuda_check(cudaDeviceSynchronize(),
             "synchronize native NVFP4 shared gate");
  const auto metrics = compare(device_output.download(), expected);
  const auto pass = metrics.relative_l2 < 0.01 && metrics.cosine > 0.9999;
  std::cout << "{\"mode\":\"shared\",\"pass\":"
            << (pass ? "true" : "false")
            << ",\"maximum_absolute_error\":"
            << metrics.maximum_absolute_error
            << ",\"maximum_relative_error\":"
            << metrics.maximum_relative_error
            << ",\"relative_l2\":" << metrics.relative_l2
            << ",\"cosine\":" << metrics.cosine << "}\n";
  return pass ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 8 && std::string_view(argv[1]) == "--shared") {
      return run_shared_gate(
          argv[2], std::stoull(argv[3]), std::stoull(argv[4]),
          std::stoull(argv[5]), std::stoull(argv[6]), argv[7]);
    }
    if (argc == 7) {
      return run_artifact_gate(
          argv[1], std::stoull(argv[2]), std::stoull(argv[3]),
          static_cast<std::uint32_t>(std::stoul(argv[4])),
          static_cast<std::uint32_t>(std::stoul(argv[5])), argv[6]);
    }
    if (argc != 1)
      throw std::runtime_error(
          "usage: expert-native-nvfp4-cuda-smoke "
          "[pack offset bytes layer expert input-f32 | "
          "--shared pack gate-offset up-offset down-offset bytes input-f32]");
    const auto gate_weights = make_weights(1U);
    const auto up_weights = make_weights(3U);
    const auto down_weights = make_weights(5U);
    const auto gate_sidecar = make_sidecar(0.25F, 16.0F);
    const auto up_sidecar = make_sidecar(0.5F, 8.0F);
    const auto down_sidecar = make_sidecar(0.125F, 32.0F);
    std::vector<float> input(kHidden);
    for (std::uint32_t index = 0U; index < kHidden; ++index)
      input[index] = static_cast<float>(static_cast<int>(index % 7U) - 3) /
                     16.0F;
    const auto expected = reference(input, gate_weights, gate_sidecar,
                                    up_weights, up_sidecar, down_weights,
                                    down_sidecar);

    DeviceBuffer<std::uint8_t> device_gate_weights(gate_weights.size());
    DeviceBuffer<std::uint8_t> device_up_weights(up_weights.size());
    DeviceBuffer<std::uint8_t> device_down_weights(down_weights.size());
    DeviceBuffer<std::uint8_t> device_gate_sidecar(gate_sidecar.size());
    DeviceBuffer<std::uint8_t> device_up_sidecar(up_sidecar.size());
    DeviceBuffer<std::uint8_t> device_down_sidecar(down_sidecar.size());
    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_intermediate(kIntermediate);
    DeviceBuffer<float> device_selection_output(kHidden);
    DeviceBuffer<float> device_gate_input(kHidden);
    DeviceBuffer<float> device_up_input(kHidden);
    DeviceBuffer<float> device_down_input(kIntermediate);
    DeviceBuffer<std::uint32_t> device_indices(1U);
    DeviceBuffer<expert::runtime::cuda::DeviceExpertEntry> device_directory(1U);
    device_gate_weights.upload(gate_weights);
    device_up_weights.upload(up_weights);
    device_down_weights.upload(down_weights);
    device_gate_sidecar.upload(gate_sidecar);
    device_up_sidecar.upload(up_sidecar);
    device_down_sidecar.upload(down_sidecar);
    device_input.upload(input);
    device_indices.upload({0U});
    expert::runtime::cuda::DeviceExpertEntry entry{};
    entry.w1_fp4 = device_gate_weights.get();
    entry.w1_ue8m0 = device_gate_sidecar.get();
    entry.w3_fp4 = device_up_weights.get();
    entry.w3_ue8m0 = device_up_sidecar.get();
    entry.w2_fp4 = device_down_weights.get();
    entry.w2_ue8m0 = device_down_sidecar.get();
    entry.format = static_cast<std::uint32_t>(
        expert::runtime::cuda::DeviceExpertFormat::
            nvfp4_e2m1_e4m3fn_block16_w4a4);
    entry.state = static_cast<std::uint32_t>(
        expert::runtime::cuda::DeviceExpertState::ready);
    device_directory.upload({entry});

    status_check(expert::runtime::cuda::launch_moe_selection_batch({
        device_input.get(), nullptr, device_indices.get(), nullptr,
        device_intermediate.get(), device_selection_output.get(), nullptr,
        nullptr, nullptr, nullptr, 1U, kHidden, kIntermediate, 1U, 1U,
        nullptr, device_directory.get(), 0U, 0.0F, true, false, true,
        device_gate_input.get(), device_up_input.get(),
        device_down_input.get()}));
    cuda_check(cudaDeviceSynchronize(), "synchronize native NVFP4 smoke");
    const auto actual = device_selection_output.download();
    const auto metrics = compare(actual, expected);
    const auto pass = metrics.maximum_relative_error < 0.01 &&
                      metrics.cosine > 0.99999;
    std::cout << "{\"pass\":" << (pass ? "true" : "false")
              << ",\"maximum_absolute_error\":"
              << metrics.maximum_absolute_error
              << ",\"maximum_relative_error\":"
              << metrics.maximum_relative_error
              << ",\"relative_l2\":" << metrics.relative_l2
              << ",\"cosine\":" << metrics.cosine << "}\n";
    return pass ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "native NVFP4 CUDA smoke: " << error.what() << '\n';
    return 1;
  }
}
