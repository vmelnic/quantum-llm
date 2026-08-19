#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/deepseek_expert.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/sha256.hpp"
#if defined(_WIN32)
#include "expert/runtime/windows_iocp_storage.hpp"
#endif

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace er = expert::runtime;

namespace {

constexpr std::uint64_t kHotBytes = 25'198'592U;

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::uint8_t hex_nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::runtime_error("invalid SHA-256 hex digit");
}

er::Sha256Digest parse_digest(const std::string& value) {
  if (value.size() != 64U) throw std::runtime_error("invalid SHA-256 length");
  er::Sha256Digest result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::byte>(
        (hex_nibble(value[index * 2U]) << 4U) |
        hex_nibble(value[index * 2U + 1U]));
  }
  return result;
}

std::vector<er::PayloadExtent> read_extents(
    const std::filesystem::path& descriptor_path,
    const std::filesystem::path& source_root) {
  std::ifstream input(descriptor_path);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid compact extent descriptor header");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    const auto first = line.find('\t');
    const auto second = line.find('\t', first + 1U);
    const auto third = line.find('\t', second + 1U);
    require(first != std::string::npos && second != std::string::npos &&
                third != std::string::npos &&
                line.find('\t', third + 1U) == std::string::npos,
            "invalid compact extent descriptor row");
    const auto destination = std::stoull(line.substr(0U, first));
    const auto bytes = std::stoull(line.substr(first + 1U, second - first - 1U));
    const auto source = std::stoull(line.substr(second + 1U, third - second - 1U));
    const std::filesystem::path shard = line.substr(third + 1U);
    require(!shard.empty() && !shard.is_absolute(),
            "extent shard must be relative to checkpoint root");
    for (const auto& component : shard) {
      require(component != "..", "extent shard may not escape checkpoint root");
    }
    result.push_back({source_root / shard, source, destination, bytes});
  }
  require(input.eof() && result.size() == 6U,
          "DeepSeek routed expert must contain exactly six extents");
  return result;
}

std::string hex_digest(const er::Sha256Digest& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : digest) {
    output << std::setw(2) << std::to_integer<unsigned>(value);
  }
  return output.str();
}

std::vector<float> cpu_expert(const std::vector<std::byte>& slot,
                              const er::DeepSeekSm86HotLayout& layout,
                              const er::DeepSeekExpertGeometry& geometry,
                              const std::vector<float>& input) {
  const auto* gate = reinterpret_cast<const std::int8_t*>(
      slot.data() + layout.gate_up_q.offset);
  const auto* scales = reinterpret_cast<const float*>(
      slot.data() + layout.gate_up_scales.offset);
  const auto* down = reinterpret_cast<const std::int8_t*>(
      slot.data() + layout.down_q.offset);
  const auto* down_scales = reinterpret_cast<const float*>(
      slot.data() + layout.down_scales.offset);
  std::vector<float> intermediate(geometry.intermediate);
  for (std::uint32_t row = 0; row < geometry.intermediate; ++row) {
    double gate_sum = 0.0;
    double up_sum = 0.0;
    for (std::uint32_t column = 0; column < geometry.hidden; ++column) {
      gate_sum += gate[static_cast<std::size_t>(row) * geometry.hidden + column] *
                  input[column];
      up_sum += gate[(static_cast<std::size_t>(geometry.intermediate + row) *
                      geometry.hidden) + column] *
                input[column];
    }
    const auto gate_value = static_cast<float>(gate_sum) * scales[row];
    const auto up_value = static_cast<float>(up_sum) *
                          scales[geometry.intermediate + row];
    intermediate[row] =
        (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
  }
  std::vector<float> output(geometry.hidden);
  for (std::uint32_t row = 0; row < geometry.hidden; ++row) {
    double sum = 0.0;
    for (std::uint32_t column = 0; column < geometry.intermediate; ++column) {
      sum += down[static_cast<std::size_t>(row) * geometry.intermediate + column] *
             intermediate[column];
    }
    output[row] = static_cast<float>(sum) * down_scales[row];
  }
  return output;
}

struct DeviceBuffers final {
  float* input{};
  float* intermediate{};
  float* output{};
  float* routing{};
  const std::int8_t** gate_table{};
  const std::int8_t** down_table{};
  const float** gate_scale_table{};
  const float** down_scale_table{};

  ~DeviceBuffers() {
    if (input) static_cast<void>(cudaFree(input));
    if (intermediate) static_cast<void>(cudaFree(intermediate));
    if (output) static_cast<void>(cudaFree(output));
    if (routing) static_cast<void>(cudaFree(routing));
    if (gate_table) static_cast<void>(cudaFree(gate_table));
    if (down_table) static_cast<void>(cudaFree(down_table));
    if (gate_scale_table) static_cast<void>(cudaFree(gate_scale_table));
    if (down_scale_table) static_cast<void>(cudaFree(down_scale_table));
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
#if !defined(_WIN32)
    (void)argc;
    (void)argv;
    throw std::runtime_error("DeepSeek IOCP admission smoke requires Windows");
#else
    if (argc != 6) {
      std::cerr << "usage: expert-deepseek-admission-smoke <descriptor-bundle> "
                   "<checkpoint-root> <expected-hot-sha256> "
                   "<source-sha256> <routed|shared>\n";
      return 64;
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path source_root = argv[2];
    const std::string expected_hot_hash = argv[3];
    const auto compact_hash = parse_digest(argv[4]);
    const std::string source_kind = argv[5];
    require(source_kind == "routed" || source_kind == "shared",
            "source kind must be routed or shared");
    const bool shared = source_kind == "shared";
    const std::uint64_t source_bytes = shared ? 25'167'360U : 13'369'344U;
    const std::uint32_t source_abi =
        shared ? er::kExpertSourceAbiDeepSeekFp8Block128V1
               : er::kExpertSourceAbiDeepSeekCompactV1;
    const auto geometry = er::DeepSeekExpertGeometry::v4_flash();
    const auto layout = er::make_deepseek_sm86_hot_layout(geometry);
    require(layout.slot_bytes == kHotBytes, "unexpected DeepSeek hot geometry");

    er::PayloadRecord record;
    record.extents = read_extents(root / "extents.tsv", source_root);
    record.record_offset = 0U;
    record.stored_bytes = source_bytes;
    record.decoded_bytes = 3ULL * 4096U * 2048U * sizeof(float);
    record.device_bytes = kHotBytes;
    record.source_abi = source_abi;
    record.header_bytes = 0U;
    record.alignment = er::kExpertPackAlignment;
    record.payload_sha256 = compact_hash;
    const er::ExpertKey key{17U, 0U, shared ? 256U : 0U,
                            er::kExpertQuantAbiDeepSeekSm86};

    auto iocp = std::make_shared<er::WindowsIocpStorage>(1U);
    auto storage = std::make_shared<er::ExtentGatherStorage>(iocp);
    auto uploader = std::make_shared<er::cuda::CudaExpertUploader>();
    auto allocator = std::make_shared<er::CudaPinnedAllocator>();
    auto buffers = std::make_shared<er::FixedBufferPool>(
        1U, source_bytes, er::kExpertPackAlignment, allocator);
    auto directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        key.model_id, key.encoding_abi, 1U, 257U, 8U);
    er::ExpertCacheConfig config;
    config.ram = {source_bytes * 2U, source_bytes * 2U, source_bytes};
    config.vram = {kHotBytes * 2U, kHotBytes * 2U, kHotBytes};
    config.retain_host_copy = false;
    er::ExpertCache cache(config, storage, uploader, buffers, directory);

    const auto cache_start = std::chrono::steady_clock::now();
    auto first_handle = cache.acquire(key, record);
    auto second_handle = cache.acquire(key, record);
    auto first = first_handle.get();
    auto second = second_handle.get();
    const auto cache_stop = std::chrono::steady_clock::now();
    require(first.status.ok(), std::string(first.status.message()));
    require(second.status.ok(), std::string(second.status.message()));
    require(first.lease.get() == second.lease.get(),
            "concurrent cache acquires did not deduplicate admission");
    const auto* allocation = dynamic_cast<const er::cuda::CudaExpertAllocation*>(
        first.lease.get());
    require(allocation != nullptr && allocation->bytes() == kHotBytes,
            "cache did not return a DeepSeek CUDA allocation");

    std::uint32_t* selected = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&selected), sizeof(std::uint32_t)),
          "cudaMalloc selected expert");
    const std::uint32_t selected_host = key.expert;
    check(cudaMemcpy(selected, &selected_host, sizeof(selected_host),
                     cudaMemcpyHostToDevice),
          "copy selected expert");
    const auto plan = directory->pin_or_collect_misses(0U, selected, 1U, nullptr);
    check(cudaFree(selected), "cudaFree selected expert");
    require(plan.status.ok() && plan.missing_experts.empty() &&
                plan.unique_experts == 1U,
            "published expert was not visible in CUDA directory");
    const auto release_status = directory->release_pins(plan.pin_id, nullptr);
    require(release_status.ok(), std::string(release_status.message()));

    std::vector<std::byte> result(layout.slot_bytes);
    check(cudaMemcpy(result.data() + layout.gate_up_q.offset,
                     allocation->gate_up(), layout.gate_up_q.bytes,
                     cudaMemcpyDeviceToHost),
          "copy admitted gate/up");
    check(cudaMemcpy(result.data() + layout.gate_up_scales.offset,
                     allocation->gate_up_scales(),
                     layout.gate_up_scales.bytes, cudaMemcpyDeviceToHost),
          "copy admitted gate/up scales");
    check(cudaMemcpy(result.data() + layout.down_q.offset, allocation->down(),
                     layout.down_q.bytes, cudaMemcpyDeviceToHost),
          "copy admitted down");
    check(cudaMemcpy(result.data() + layout.down_scales.offset,
                     allocation->down_scales(), layout.down_scales.bytes,
                     cudaMemcpyDeviceToHost),
          "copy admitted down scales");
    const auto actual_hash = hex_digest(er::sha256(result));
    require(actual_hash == expected_hot_hash,
            "cache CUDA hot-slot hash mismatch: " + actual_hash);

    std::vector<float> input(geometry.hidden);
    for (std::uint32_t index = 0; index < geometry.hidden; ++index) {
      input[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.25F;
    }
    const auto reference = cpu_expert(result, layout, geometry, input);
    DeviceBuffers device;
    check(cudaMalloc(reinterpret_cast<void**>(&device.input),
                     input.size() * sizeof(float)),
          "cudaMalloc input");
    check(cudaMalloc(reinterpret_cast<void**>(&device.intermediate),
                     geometry.intermediate * sizeof(float)),
          "cudaMalloc intermediate");
    check(cudaMalloc(reinterpret_cast<void**>(&device.output),
                     geometry.hidden * sizeof(float)),
          "cudaMalloc output");
    check(cudaMalloc(reinterpret_cast<void**>(&device.routing), sizeof(float)),
          "cudaMalloc routing");
    check(cudaMalloc(reinterpret_cast<void**>(&device.gate_table), sizeof(void*)),
          "cudaMalloc gate table");
    check(cudaMalloc(reinterpret_cast<void**>(&device.down_table), sizeof(void*)),
          "cudaMalloc down table");
    check(cudaMalloc(reinterpret_cast<void**>(&device.gate_scale_table),
                     sizeof(void*)),
          "cudaMalloc gate scale table");
    check(cudaMalloc(reinterpret_cast<void**>(&device.down_scale_table),
                     sizeof(void*)),
          "cudaMalloc down scale table");
    const auto* gate = allocation->gate_up();
    const auto* down = allocation->down();
    const auto* gate_scales = allocation->gate_up_scales();
    const auto* down_scales = allocation->down_scales();
    const float routing = 1.0F;
    check(cudaMemcpy(device.input, input.data(), input.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "copy input");
    check(cudaMemcpy(device.routing, &routing, sizeof(float),
                     cudaMemcpyHostToDevice),
          "copy routing");
    check(cudaMemcpy(device.gate_table, &gate, sizeof(void*),
                     cudaMemcpyHostToDevice),
          "copy gate table");
    check(cudaMemcpy(device.down_table, &down, sizeof(void*),
                     cudaMemcpyHostToDevice),
          "copy down table");
    check(cudaMemcpy(device.gate_scale_table, &gate_scales, sizeof(void*),
                     cudaMemcpyHostToDevice),
          "copy gate scales");
    check(cudaMemcpy(device.down_scale_table, &down_scales, sizeof(void*),
                     cudaMemcpyHostToDevice),
          "copy down scales");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    check(cudaEventRecord(start), "record GEMM start");
    const er::cuda::MoeLaunch launch{
        device.input, device.gate_table, device.gate_scale_table,
        device.down_table, device.down_scale_table, device.routing, nullptr,
        device.intermediate, device.output, geometry.hidden,
        geometry.intermediate, 1U, 1U, nullptr, nullptr, 0U};
    const auto moe_status = er::cuda::launch_moe_single_token(launch);
    require(moe_status.ok(), std::string(moe_status.message()));
    check(cudaEventRecord(stop), "record GEMM stop");
    check(cudaEventSynchronize(stop), "synchronize GEMM");
    float gemm_ms = 0.0F;
    check(cudaEventElapsedTime(&gemm_ms, start, stop), "GEMM elapsed");
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    std::vector<float> gpu_output(geometry.hidden);
    check(cudaMemcpy(gpu_output.data(), device.output,
                     gpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy GEMM output");
    double squared = 0.0;
    float max_error = 0.0F;
    for (std::size_t index = 0; index < gpu_output.size(); ++index) {
      const auto error = std::abs(gpu_output[index] - reference[index]);
      max_error = std::max(max_error, error);
      squared += static_cast<double>(error) * error;
    }
    const auto rmse = std::sqrt(squared / gpu_output.size());
    const auto telemetry = cache.telemetry();
    require(telemetry.load_started == 1U &&
                telemetry.load_deduplicated == 1U &&
                telemetry.load_completed == 1U &&
                telemetry.upload_started == 1U &&
                telemetry.upload_completed == 1U &&
                telemetry.read_bytes == source_bytes &&
                telemetry.uploaded_bytes == kHotBytes,
            "cache lifecycle telemetry violated single-flight admission");
    const auto cache_ms = std::chrono::duration<double, std::milli>(
                              cache_stop - cache_start)
                              .count();

    first.lease = {};
    second.lease = {};
    const auto trimmed = cache.trim();
    require(trimmed == kHotBytes && !cache.inspect(key)->has_device_copy,
            "cache trim did not release admitted DeepSeek expert");
    const auto source_abi_name =
        shared ? er::kDeepSeekFp8Block128Abi : er::kDeepSeekCompactAbi;
    std::cout << "{\"ok\":true,\"source_abi\":\""
              << source_abi_name << "\",\"hot_abi\":\""
              << er::kDeepSeekSm86HotAbi << "\",\"source_bytes\":"
              << source_bytes << ",\"hot_bytes\":" << layout.slot_bytes
              << ",\"sha256\":\"" << actual_hash
              << "\",\"cache_end_to_end_ms\":" << cache_ms
              << ",\"single_flight_loads\":" << telemetry.load_started
              << ",\"deduplicated_acquires\":"
              << telemetry.load_deduplicated
              << ",\"expert_gemm_ms\":" << gemm_ms
              << ",\"output_rmse\":" << rmse
              << ",\"output_max_abs_error\":" << max_error
              << ",\"trimmed_bytes\":" << trimmed << "}\n";
    return 0;
#endif
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-admission-smoke: " << error.what() << '\n';
    return 1;
  }
}
