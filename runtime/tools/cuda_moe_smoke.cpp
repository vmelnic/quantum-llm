#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
T read_le(const std::byte* bytes) {
  T value{};
  std::memcpy(&value, bytes, sizeof(T));
  return value;
}

void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

template <typename T>
void device_allocate(T*& pointer, std::size_t bytes, const char* operation) {
  void* raw = nullptr;
  cuda_check(cudaMalloc(&raw, bytes), operation);
  pointer = reinterpret_cast<T*>(raw);
}

struct HostRecord {
  std::vector<std::byte> bytes;
  expert::runtime::ValidatedExpertRecord validated;
  expert::runtime::PayloadRecord payload;
};

HostRecord load_record(std::ifstream& stream, const std::filesystem::path& path,
                       std::uint64_t offset, std::uint32_t expert_id) {
  std::array<std::byte, 148> header{};
  stream.seekg(static_cast<std::streamoff>(offset));
  stream.read(reinterpret_cast<char*>(header.data()), header.size());
  if (stream.gcount() != static_cast<std::streamsize>(header.size())) {
    throw std::runtime_error("short expert header");
  }
  const auto stored = read_le<std::uint64_t>(header.data() + 44);
  if (stored == 0 || stored > 64ULL * 1024 * 1024) {
    throw std::runtime_error("unsafe expert record size");
  }
  HostRecord result;
  result.bytes.resize(static_cast<std::size_t>(stored));
  stream.seekg(static_cast<std::streamoff>(offset));
  stream.read(reinterpret_cast<char*>(result.bytes.data()),
              static_cast<std::streamsize>(result.bytes.size()));
  if (stream.gcount() != static_cast<std::streamsize>(result.bytes.size())) {
    throw std::runtime_error("short expert record");
  }
  expert::runtime::PayloadRecord expected;
  expected.path = path;
  expected.record_offset = offset;
  expected.stored_bytes = stored;
  std::copy_n(header.data() + 116, expected.payload_sha256.size(),
              expected.payload_sha256.begin());
  const expert::runtime::ExpertKey key{0, 0, expert_id, 1};
  auto validation = expert::runtime::validate_expert_record(result.bytes, key, expected);
  if (!validation.status.ok()) {
    throw std::runtime_error(std::string(validation.status.message()));
  }
  result.validated = validation.record;
  result.payload = expected;
  return result;
}

template <typename T>
const T* section(const HostRecord& record, std::uint64_t offset) {
  return reinterpret_cast<const T*>(record.bytes.data() + offset);
}

std::vector<float> cpu_reference(const std::vector<HostRecord>& records,
                                 const std::vector<float>& input,
                                 const std::vector<float>& routing) {
  const auto hidden = records.front().validated.sections.hidden;
  const auto width = records.front().validated.sections.intermediate;
  std::vector<float> output(hidden, 0.0F);
  std::vector<float> intermediate(width);
  for (std::size_t expert = 0; expert < records.size(); ++expert) {
    const auto& record = records[expert];
    const auto& s = record.validated.sections;
    const auto* gate_up = section<std::int8_t>(record, s.gate_up_q_offset);
    const auto* gate_scales = section<float>(record, s.gate_up_scale_offset);
    const auto* down = section<std::int8_t>(record, s.down_q_offset);
    const auto* down_scales = section<float>(record, s.down_scale_offset);
    for (std::uint32_t row = 0; row < width; ++row) {
      double gate = 0.0;
      double up = 0.0;
      for (std::uint32_t column = 0; column < hidden; ++column) {
        gate += static_cast<float>(gate_up[static_cast<std::size_t>(row) * hidden + column]) * input[column];
        up += static_cast<float>(gate_up[static_cast<std::size_t>(width + row) * hidden + column]) * input[column];
      }
      const auto gate_f = static_cast<float>(gate) * gate_scales[row];
      const auto up_f = static_cast<float>(up) * gate_scales[width + row];
      intermediate[row] = (gate_f / (1.0F + std::exp(-gate_f))) * up_f;
    }
    for (std::uint32_t row = 0; row < hidden; ++row) {
      double sum = 0.0;
      for (std::uint32_t column = 0; column < width; ++column) {
        sum += static_cast<float>(down[static_cast<std::size_t>(row) * width + column]) * intermediate[column];
      }
      output[row] += routing[expert] * static_cast<float>(sum) * down_scales[row];
    }
  }
  return output;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: expert-cuda-moe-smoke <experts-000.qpack> [top-k]\n";
      return 64;
    }
    const std::filesystem::path path = argv[1];
    const auto top_k = argc == 3 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 8U;
    if (top_k == 0 || top_k > 64) throw std::runtime_error("top-k out of range");
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open expert pack");

    std::vector<HostRecord> records;
    std::uint64_t offset = 0;
    for (std::uint32_t expert = 0; expert < top_k; ++expert) {
      records.push_back(load_record(stream, path, offset, expert));
      offset += records.back().bytes.size();
    }
    const auto hidden = records.front().validated.sections.hidden;
    const auto width = records.front().validated.sections.intermediate;
    std::vector<float> input(hidden);
    for (std::uint32_t i = 0; i < hidden; ++i) input[i] = std::sin(static_cast<float>(i) * 0.013F) * 0.25F;
    std::vector<float> routing(top_k);
    float routing_sum = 0.0F;
    for (std::uint32_t i = 0; i < top_k; ++i) {
      routing[i] = static_cast<float>(top_k - i);
      routing_sum += routing[i];
    }
    for (auto& value : routing) value /= routing_sum;
    const auto reference = cpu_reference(records, input, routing);

    float* d_input{};
    float* d_intermediate{};
    float* d_output{};
    float* d_selection_output{};
    float* d_routing{};
    std::uint32_t* d_indices{};
    device_allocate(d_input, input.size() * sizeof(float), "cudaMalloc input");
    device_allocate(d_intermediate, static_cast<std::size_t>(top_k) * width * sizeof(float), "cudaMalloc intermediate");
    device_allocate(d_output, hidden * sizeof(float), "cudaMalloc output");
    device_allocate(d_selection_output,
                    static_cast<std::size_t>(top_k) * hidden * sizeof(float),
                    "cudaMalloc selection output");
    device_allocate(d_routing, routing.size() * sizeof(float), "cudaMalloc routing");
    device_allocate(d_indices, top_k * sizeof(std::uint32_t),
                    "cudaMalloc expert indices");
    cuda_check(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input");
    cuda_check(cudaMemcpy(d_routing, routing.data(), routing.size() * sizeof(float), cudaMemcpyHostToDevice), "copy routing");
    std::vector<std::uint32_t> indices(top_k);
    std::iota(indices.begin(), indices.end(), 0U);
    cuda_check(cudaMemcpy(d_indices, indices.data(), top_k * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "copy expert indices");
    const auto slot_bytes = static_cast<std::size_t>(
        std::max_element(records.begin(), records.end(), [](const auto& left, const auto& right) {
          return left.payload.stored_bytes < right.payload.stored_bytes;
        })->payload.stored_bytes);
    const auto tier_bytes = static_cast<std::uint64_t>(slot_bytes) * top_k;
    auto storage = std::make_shared<expert::runtime::WindowsIocpStorage>(2);
    auto uploader = std::make_shared<expert::runtime::cuda::CudaExpertUploader>();
    auto directory =
        std::make_shared<expert::runtime::cuda::CudaExpertDirectory>(
            0, 1, 1, top_k, top_k);
    auto buffers = std::make_shared<expert::runtime::FixedBufferPool>(
        2, slot_bytes, expert::runtime::kExpertPackAlignment,
        std::make_shared<expert::runtime::CudaPinnedAllocator>());
    expert::runtime::ExpertCache cache(
        {{tier_bytes, tier_bytes, tier_bytes},
         {tier_bytes, tier_bytes, tier_bytes}, true},
        storage, uploader, buffers, directory);
    std::vector<expert::runtime::AcquireHandle> handles;
    handles.reserve(top_k);
    for (std::uint32_t i = 0; i < top_k; ++i) {
      handles.push_back(cache.acquire({0, 0, i, 1}, records[i].payload));
    }
    std::vector<expert::runtime::ExpertLease> leases;
    leases.reserve(top_k);
    for (std::uint32_t i = 0; i < top_k; ++i) {
      auto acquired = handles[i].get();
      if (!acquired.status.ok()) {
        throw std::runtime_error(std::string("cache acquire: ") +
                                 std::string(acquired.status.message()));
      }
      leases.push_back(std::move(acquired.lease));
    }
    const auto plan = directory->pin_or_collect_misses(
        0, d_indices, top_k, nullptr);
    if (!plan.status.ok() || !plan.missing_experts.empty())
      throw std::runtime_error("CUDA directory did not publish acquired experts");
    const auto concurrent_plan = directory->pin_or_collect_misses(
        0, d_indices, top_k, nullptr);
    if (!concurrent_plan.status.ok() ||
        !concurrent_plan.missing_experts.empty() ||
        concurrent_plan.pin_id == plan.pin_id)
      throw std::runtime_error("CUDA directory rejected concurrent route pins");
    expert::runtime::cuda::MoeLaunch launch{
        d_input, nullptr, nullptr, nullptr, nullptr, d_routing, d_indices,
        d_intermediate, d_output, hidden, width, top_k, top_k, nullptr,
        directory->device_entries(), 0};
    const auto status = expert::runtime::cuda::launch_moe_single_token(launch);
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    cuda_check(cudaDeviceSynchronize(), "MoE synchronize");
    auto split_status = expert::runtime::cuda::launch_moe_selection_batch({
        d_input, d_routing, d_indices, nullptr, d_intermediate,
        d_selection_output, 1, hidden, width, top_k, top_k, nullptr,
        directory->device_entries(), 0});
    if (!split_status.ok())
      throw std::runtime_error(std::string(split_status.message()));
    split_status = expert::runtime::cuda::launch_moe_aggregate({
        d_selection_output, nullptr, nullptr, nullptr, d_routing, d_output, 0,
        1, hidden, top_k, nullptr});
    if (!split_status.ok())
      throw std::runtime_error(std::string(split_status.message()));
    cuda_check(cudaDeviceSynchronize(), "split MoE synchronize");
    constexpr int kWarmup = 10;
    constexpr int kIterations = 100;
    for (int iteration = 0; iteration < kWarmup; ++iteration) {
      const auto warmup_status = expert::runtime::cuda::launch_moe_single_token(launch);
      if (!warmup_status.ok()) throw std::runtime_error(std::string(warmup_status.message()));
    }
    cudaEvent_t start{};
    cudaEvent_t stop{};
    cuda_check(cudaEventCreate(&start), "create start event");
    cuda_check(cudaEventCreate(&stop), "create stop event");
    cuda_check(cudaEventRecord(start), "record start event");
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      const auto timed_status = expert::runtime::cuda::launch_moe_single_token(launch);
      if (!timed_status.ok()) throw std::runtime_error(std::string(timed_status.message()));
    }
    cuda_check(cudaEventRecord(stop), "record stop event");
    cuda_check(cudaEventSynchronize(stop), "synchronize stop event");
    float elapsed_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "measure events");
    const auto kernel_ms = elapsed_ms / static_cast<float>(kIterations);
    cuda_check(cudaEventDestroy(start), "destroy start event");
    cuda_check(cudaEventDestroy(stop), "destroy stop event");
    std::vector<float> actual(hidden);
    cuda_check(cudaMemcpy(actual.data(), d_output, hidden * sizeof(float), cudaMemcpyDeviceToHost), "copy output");
    double max_abs = 0.0;
    double max_rel = 0.0;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (std::uint32_t i = 0; i < hidden; ++i) {
      const double delta = std::abs(static_cast<double>(actual[i]) - reference[i]);
      max_abs = std::max(max_abs, delta);
      max_rel = std::max(max_rel, delta / std::max(1e-6, std::abs(static_cast<double>(reference[i]))));
      dot += static_cast<double>(actual[i]) * reference[i];
      norm_a += static_cast<double>(actual[i]) * actual[i];
      norm_b += static_cast<double>(reference[i]) * reference[i];
    }
    const double cosine = dot / std::sqrt(norm_a * norm_b);
    auto release_status = directory->release_pins(concurrent_plan.pin_id,
                                                  nullptr);
    if (!release_status.ok())
      throw std::runtime_error(std::string(release_status.message()));
    release_status = directory->release_pins(plan.pin_id, nullptr);
    if (!release_status.ok())
      throw std::runtime_error(std::string(release_status.message()));
    const auto cache_metrics = cache.telemetry();
    std::cout << "{\"valid\":" << (max_rel < 0.01 && cosine > 0.999999 ? "true" : "false")
              << ",\"top_k\":" << top_k << ",\"hidden\":" << hidden
              << ",\"intermediate\":" << width << ",\"max_abs\":" << max_abs
              << ",\"max_rel\":" << max_rel << ",\"cosine\":" << cosine
              << ",\"kernel_ms_per_layer\":" << kernel_ms
              << ",\"cache_read_bytes\":" << cache_metrics.read_bytes
              << ",\"cache_uploaded_bytes\":" << cache_metrics.uploaded_bytes
              << ",\"cache_loads\":" << cache_metrics.load_completed
              << ",\"cache_ram_high_water\":" << cache_metrics.ram_high_water
              << ",\"staging_high_water\":" << cache_metrics.staging_high_water
              << ",\"moe_only_16_layer_ceiling_tps\":" << (1000.0F / (16.0F * kernel_ms))
              << "}\n";
    return max_rel < 0.01 && cosine > 0.999999 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "cuda MoE smoke: " << error.what() << '\n';
    return 1;
  }
}
