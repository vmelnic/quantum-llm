#include "expert/runtime/cuda/active_expert_device_executor.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace er = expert::runtime;
namespace ec = expert::runtime::cuda;

namespace {

constexpr std::string_view kInputAbi = "expert.swiglu.input.f32.host.v1";
constexpr std::string_view kOutputAbi = "expert.swiglu.output.f32.host.v1";

void require(bool condition, std::string message) {
  if (!condition) throw std::runtime_error(std::move(message));
}

void cuda_check(cudaError_t error, std::string_view operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

template <typename T>
T read_le(const std::byte* bytes) {
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned value{};
  for (std::size_t index = 0U; index < sizeof(T); ++index)
    value |= static_cast<Unsigned>(std::to_integer<unsigned>(bytes[index]))
             << static_cast<unsigned>(8U * index);
  return static_cast<T>(value);
}

std::vector<std::byte> read_record(const er::PayloadRecord& record) {
  require(record.stored_bytes != 0U &&
              record.stored_bytes <=
                  std::numeric_limits<std::size_t>::max(),
          "expert record size is invalid");
  std::vector<std::byte> result(
      static_cast<std::size_t>(record.stored_bytes));
  const auto read_extent = [&](const std::filesystem::path& path,
                               std::uint64_t source_offset,
                               std::uint64_t destination_offset,
                               std::uint64_t bytes) {
    require(destination_offset <= result.size() &&
                bytes <= result.size() - destination_offset,
            "expert extent exceeds its record");
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open " + path.string());
    input.seekg(static_cast<std::streamoff>(source_offset));
    require(static_cast<bool>(input), "cannot seek in " + path.string());
    input.read(reinterpret_cast<char*>(result.data() + destination_offset),
               static_cast<std::streamsize>(bytes));
    require(input.gcount() == static_cast<std::streamsize>(bytes),
            "short expert read from " + path.string());
  };
  if (record.extents.empty()) {
    read_extent(record.path, record.record_offset, 0U, record.stored_bytes);
  } else {
    for (const auto& extent : record.extents)
      read_extent(extent.path, extent.source_offset,
                  extent.destination_offset, extent.bytes);
  }
  return result;
}

struct IndependentSections final {
  std::uint64_t gate_weight{};
  std::uint64_t gate_scale{};
  std::uint64_t up_weight{};
  std::uint64_t up_scale{};
  std::uint64_t down_weight{};
  std::uint64_t down_scale{};
};

IndependentSections parse_standard_fp4_record(
    std::span<const std::byte> record, std::uint32_t layer,
    std::uint32_t expert, std::uint32_t hidden,
    std::uint32_t intermediate) {
  constexpr std::array<std::byte, 8U> magic{
      std::byte{'E'}, std::byte{'P'}, std::byte{'E'}, std::byte{'X'},
      std::byte{'P'}, std::byte{'R'}, std::byte{'0'}, std::byte{'1'}};
  require(record.size() >= 256U &&
              std::equal(magic.begin(), magic.end(), record.begin()),
          "independent oracle rejected expert record magic");
  require(read_le<std::uint16_t>(record.data() + 8U) == 1U &&
              read_le<std::uint16_t>(record.data() + 10U) == 256U &&
              read_le<std::uint32_t>(record.data() + 16U) == 3U &&
              read_le<std::int32_t>(record.data() + 20U) ==
                  static_cast<std::int32_t>(layer) &&
              read_le<std::int32_t>(record.data() + 24U) ==
                  static_cast<std::int32_t>(expert) &&
              read_le<std::uint32_t>(record.data() + 28U) == hidden &&
              read_le<std::uint32_t>(record.data() + 32U) == intermediate &&
              read_le<std::uint64_t>(record.data() + 44U) == record.size(),
          "independent oracle rejected expert header geometry");
  const auto matrix_values =
      static_cast<std::uint64_t>(hidden) * intermediate;
  const auto weight_bytes = matrix_values / 2U;
  const auto scale_bytes = matrix_values / 32U;
  const auto gate_up_weight = read_le<std::uint64_t>(record.data() + 52U);
  const auto gate_up_weight_bytes =
      read_le<std::uint64_t>(record.data() + 60U);
  const auto gate_up_scale = read_le<std::uint64_t>(record.data() + 68U);
  const auto gate_up_scale_bytes =
      read_le<std::uint64_t>(record.data() + 76U);
  const auto down_weight = read_le<std::uint64_t>(record.data() + 84U);
  const auto down_weight_bytes =
      read_le<std::uint64_t>(record.data() + 92U);
  const auto down_scale = read_le<std::uint64_t>(record.data() + 100U);
  const auto down_scale_bytes =
      read_le<std::uint64_t>(record.data() + 108U);
  require(gate_up_weight_bytes == 2U * weight_bytes &&
              gate_up_scale_bytes == 2U * scale_bytes &&
              down_weight_bytes == weight_bytes &&
              down_scale_bytes == scale_bytes,
          "independent oracle rejected FP4 section sizes");
  const auto in_range = [&](std::uint64_t offset, std::uint64_t bytes) {
    return offset <= record.size() && bytes <= record.size() - offset;
  };
  require(in_range(gate_up_weight, 2U * weight_bytes) &&
              in_range(gate_up_scale, 2U * scale_bytes) &&
              in_range(down_weight, weight_bytes) &&
              in_range(down_scale, scale_bytes),
          "independent oracle rejected FP4 section bounds");
  return {gate_up_weight, gate_up_scale,
          gate_up_weight + weight_bytes, gate_up_scale + scale_bytes,
          down_weight, down_scale};
}

float decode_scale(std::uint8_t code) {
  return std::ldexp(1.0F, code == 0U ? -127 : static_cast<int>(code) - 127);
}

int decode_weight_twice(std::uint8_t code) {
  constexpr std::array<int, 16U> values{
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return values[code & 0x0fU];
}

float quantize_q8(std::span<const float> input,
                  std::span<std::int8_t> output) {
  require(input.size() == output.size(), "Q8 oracle shape mismatch");
  float maximum{};
  for (const auto value : input) maximum = std::max(maximum, std::abs(value));
  const auto scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
  for (std::size_t index = 0U; index < input.size(); ++index) {
    const auto rounded =
        static_cast<int>(std::nearbyint(input[index] / scale));
    output[index] =
        static_cast<std::int8_t>(std::clamp(rounded, -127, 127));
  }
  return scale;
}

float dot(const std::byte* weights, const std::byte* scales,
          std::span<const std::int8_t> input, float input_scale,
          std::uint32_t row) {
  const auto columns = static_cast<std::uint32_t>(input.size());
  const auto* packed = weights + static_cast<std::size_t>(row) * columns / 2U;
  const auto* row_scales = scales +
      static_cast<std::size_t>(row) * (columns / 32U);
  float total{};
  for (std::uint32_t block = 0U; block < columns / 32U; ++block) {
    int block_total{};
    for (std::uint32_t column = 0U; column < 32U; ++column) {
      const auto byte = std::to_integer<std::uint8_t>(
          packed[static_cast<std::size_t>(block) * 16U + column / 2U]);
      const auto code = static_cast<std::uint8_t>(
          (column & 1U) != 0U ? byte >> 4U : byte & 0x0fU);
      block_total += decode_weight_twice(code) *
                     static_cast<int>(input[block * 32U + column]);
    }
    total += static_cast<float>(block_total) *
             decode_scale(std::to_integer<std::uint8_t>(row_scales[block]));
  }
  return total * input_scale * 0.5F;
}

std::vector<float> independent_oracle(
    std::span<const std::byte> record, const IndependentSections& sections,
    std::span<const float> input, std::uint32_t intermediate) {
  std::vector<std::int8_t> quantized_input(input.size());
  const auto input_scale = quantize_q8(input, quantized_input);
  std::vector<float> activated(intermediate);
  for (std::uint32_t row = 0U; row < intermediate; ++row) {
    const auto gate = dot(record.data() + sections.gate_weight,
                          record.data() + sections.gate_scale,
                          quantized_input, input_scale, row);
    const auto up = dot(record.data() + sections.up_weight,
                        record.data() + sections.up_scale,
                        quantized_input, input_scale, row);
    activated[row] = (gate / (1.0F + std::exp(-gate))) * up;
  }
  std::vector<std::int8_t> quantized_intermediate(intermediate);
  const auto intermediate_scale =
      quantize_q8(activated, quantized_intermediate);
  std::vector<float> output(input.size());
  for (std::uint32_t row = 0U; row < output.size(); ++row)
    output[row] = dot(record.data() + sections.down_weight,
                      record.data() + sections.down_scale,
                      quantized_intermediate, intermediate_scale, row);
  return output;
}

struct PinnedInput final {
  explicit PinnedInput(std::size_t values) {
    cuda_check(cudaHostAlloc(&raw, values * sizeof(float),
                             cudaHostAllocPortable),
               "allocate independent oracle input");
    owner = std::shared_ptr<const void>(raw, [](const void* pointer) {
      static_cast<void>(cudaFreeHost(const_cast<void*>(pointer)));
    });
  }
  float* data() const noexcept { return static_cast<float*>(raw); }
  void* raw{};
  std::shared_ptr<const void> owner;
};

struct ErrorMetrics final {
  double maximum_absolute{};
  double relative_l2{};
  double reference_l2{};
};

ErrorMetrics compare(std::span<const float> actual,
                     std::span<const float> expected) {
  require(actual.size() == expected.size(), "oracle result shape mismatch");
  double error_squared{};
  double reference_squared{};
  double maximum{};
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    const auto difference =
        static_cast<double>(actual[index]) - expected[index];
    error_squared += difference * difference;
    reference_squared +=
        static_cast<double>(expected[index]) * expected[index];
    maximum = std::max(maximum, std::abs(difference));
  }
  return {maximum,
          std::sqrt(error_squared /
                    std::max(reference_squared,
                             std::numeric_limits<double>::min())),
          std::sqrt(reference_squared)};
}

std::string hex_digest(const er::Sha256Digest& digest) {
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(2U * digest.size());
  for (const auto value : digest) {
    const auto byte = std::to_integer<unsigned>(value);
    result.push_back(alphabet[byte >> 4U]);
    result.push_back(alphabet[byte & 0x0fU]);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2,
            "usage: expert-pascal-active-expert-gate <artifact-root>");
    er::ModelArtifact artifact;
    const auto loaded = er::ModelArtifact::load(argv[1], artifact);
    require(loaded.ok(), std::string(loaded.message()));
    require(artifact.model().routed_components.size() == 1U,
            "gate requires exactly one routed component");
    const auto component = artifact.model().routed_components.front();
    require(component.source_abi == er::kExpertSourceAbiExpertPackV1 &&
                component.encoding_abi == er::kExpertEncodingAbiFp4Block32 &&
                component.encoding == "fp4.e2m1.ue8m0.block32" &&
                (component.execution_capability ==
                     "moe.swiglu.routed.merge-shared.v1" ||
                 component.execution_capability ==
                     "moe.swiglu.routed.merge-shared.no-residual.v1"),
            "artifact routed component is not standard FP4 SwiGLU");
    const auto* artifact_component = artifact.find_component(component.name);
    require(artifact_component != nullptr,
            "artifact routed catalog is absent");
    cuda_check(cudaSetDevice(0), "select primary CUDA device");
    auto devices = ec::discover_pascal_active_expert_devices();
    require(devices.size() >= 2U,
            "gate requires two secondary Pascal CUDA devices");
    devices.resize(2U);

    ec::ActiveExpertDeviceExecutorConfig config;
    config.model_content_hash = artifact.model().content_hash;
    config.component = component;
    config.device_ordinals = devices;
    config.device_cache_bytes_per_device = 512ULL << 20U;
    config.device_reserve_bytes_per_device = 1ULL << 30U;
    config.host_cache_bytes_total = 1ULL << 30U;
    config.staging_slots_per_device =
        std::max<std::uint32_t>(8U, component.route_width);
    config.input_abi = std::string(kInputAbi);
    config.output_abi = std::string(kOutputAbi);
    config.activation_clamp = 0.0F;
    config.round_intermediate_to_bf16 = false;
    auto created = ec::create_active_expert_device_executor(
        std::move(config), artifact_component->catalog,
        std::make_shared<er::WindowsIocpStorage>(2U));
    require(created.status.ok() && created.executor,
            created.status.ok() ? "active executor is absent"
                                : std::string(created.status.message()));

    PinnedInput input(component.hidden_size);
    for (std::uint32_t index = 0U; index < component.hidden_size; ++index) {
      const auto phase = static_cast<float>(index % 257U) * 0.03125F;
      input.data()[index] =
          std::sin(phase) * 0.75F + std::cos(phase * 0.375F) * 0.25F;
    }
    const auto hidden_bytes =
        static_cast<std::uint64_t>(component.hidden_size) * sizeof(float);
    std::vector<er::ActiveExpertExecutionHandle> handles;
    for (std::uint32_t selection = 0U; selection < 2U; ++selection) {
      er::ActiveExpertExecutionRequest request;
      request.identity = {
          artifact.model().content_hash,
          {component.namespace_id, 0U, selection, component.encoding_abi},
          component.execution_capability, component.execution_abi,
          component.source_abi};
      request.invocation.request_id = 1U;
      request.invocation.invocation_id = selection + 1U;
      request.invocation.selection_index = selection;
      request.invocation.route_width = component.route_width;
      request.invocation.input = {
          std::string(kInputAbi), "host.pinned", input.owner,
          reinterpret_cast<const std::byte*>(input.data()), hidden_bytes};
      request.invocation.output_abi = kOutputAbi;
      request.invocation.output_bytes = hidden_bytes;
      handles.push_back(created.executor->execute(std::move(request)));
      require(handles.back().valid(), "active executor rejected request");
    }

    std::array<ErrorMetrics, 2U> metrics{};
    std::uint64_t result_weight_bytes{};
    std::uint64_t result_storage_bytes{};
    std::uint64_t result_ram_bytes{};
    std::uint64_t result_vram_bytes{};
    for (std::uint32_t selection = 0U; selection < 2U; ++selection) {
      std::optional<er::ActiveExpertExecutionResult> result;
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(60);
      while (!result && std::chrono::steady_clock::now() < deadline) {
        result = handles[selection].poll();
        if (!result) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      require(result.has_value(), "active expert gate timed out");
      require(result->status.ok() && result->output.valid() &&
                  result->output.abi == kOutputAbi &&
                  result->output.bytes == hidden_bytes &&
                  result->evidence.weight_transport_bytes == 0U,
              result->status.ok() ? "active expert result is invalid"
                                  : std::string(result->status.message()));
      result_weight_bytes += result->evidence.owner_weight_read_bytes;
      result_storage_bytes += result->evidence.owner_storage_read_bytes;
      result_ram_bytes += result->evidence.owner_ram_read_bytes;
      result_vram_bytes += result->evidence.owner_vram_read_bytes;
      const auto* record = artifact_component->catalog.find(0U, selection);
      require(record != nullptr, "oracle expert record is absent");
      const auto bytes = read_record(*record);
      const auto sections = parse_standard_fp4_record(
          bytes, 0U, selection, component.hidden_size,
          component.intermediate_size);
      const auto expected = independent_oracle(
          bytes, sections,
          std::span<const float>(input.data(), component.hidden_size),
          component.intermediate_size);
      metrics[selection] = compare(
          std::span<const float>(
              reinterpret_cast<const float*>(result->output.data),
              component.hidden_size),
          expected);
    }

    const auto telemetry = created.executor->telemetry();
    const auto maximum_absolute =
        std::max(metrics[0].maximum_absolute,
                 metrics[1].maximum_absolute);
    const auto maximum_relative_l2 =
        std::max(metrics[0].relative_l2, metrics[1].relative_l2);
    const auto minimum_reference_l2 =
        std::min(metrics[0].reference_l2, metrics[1].reference_l2);
    const auto* representative = artifact_component->catalog.find(0U, 0U);
    require(representative != nullptr, "representative expert is absent");
    const auto device_record_bytes = representative->device_bytes == 0U
                                         ? representative->stored_bytes
                                         : representative->device_bytes;
    const bool numeric_pass = minimum_reference_l2 > 1.0e-6 &&
                              maximum_absolute <= 0.05 &&
                              maximum_relative_l2 <= 2.0e-5;
    const bool evidence_pass = telemetry.requests == 2U &&
                               telemetry.completed == 2U &&
                               telemetry.failed == 0U &&
                               telemetry.owner_weight_read_bytes ==
                                   2U * device_record_bytes &&
                               telemetry.owner_storage_read_bytes ==
                                   2U * representative->stored_bytes &&
                               result_weight_bytes ==
                                   2U * device_record_bytes &&
                               result_storage_bytes ==
                                   2U * representative->stored_bytes &&
                               result_ram_bytes == 0U &&
                               result_vram_bytes == 0U &&
                               telemetry.weight_transport_bytes == 0U;
    std::cout << std::fixed << std::setprecision(8)
              << "{\"schema_version\":1,\"model_hash\":\""
              << hex_digest(artifact.model().content_hash)
              << "\",\"layers\":" << component.layer_count
              << ",\"experts_per_layer\":"
              << component.experts_per_layer << ",\"top_k\":"
              << component.route_width << ",\"hidden\":"
              << component.hidden_size << ",\"intermediate\":"
              << component.intermediate_size << ",\"devices\":["
              << devices[0] << ',' << devices[1]
              << "],\"maximum_absolute_error\":" << maximum_absolute
              << ",\"maximum_relative_l2\":" << maximum_relative_l2
              << ",\"minimum_reference_l2\":" << minimum_reference_l2
              << ",\"requests\":" << telemetry.requests
              << ",\"completed\":" << telemetry.completed
              << ",\"failed\":" << telemetry.failed
              << ",\"owner_weight_read_bytes\":"
              << telemetry.owner_weight_read_bytes
              << ",\"storage_read_bytes\":"
              << telemetry.owner_storage_read_bytes
              << ",\"weight_transport_bytes\":"
              << telemetry.weight_transport_bytes
              << ",\"numeric_pass\":"
              << (numeric_pass ? "true" : "false")
              << ",\"evidence_pass\":"
              << (evidence_pass ? "true" : "false")
              << ",\"admission_pass\":"
              << (numeric_pass && evidence_pass ? "true" : "false")
              << "}\n";
    return numeric_pass && evidence_pass ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "{\"schema_version\":1,\"admission_pass\":false,"
                 "\"error\":"
              << std::quoted(std::string(error.what())) << "}\n";
    return 1;
  }
}
