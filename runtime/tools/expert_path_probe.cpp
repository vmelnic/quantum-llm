#include "expert/runtime/expert_record.hpp"

#include <cuda_runtime_api.h>
#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

template <typename T>
T read_le(const std::byte* source) {
  T value{};
  std::memcpy(&value, source, sizeof(value));
  return value;
}

void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

struct Record final {
  std::vector<std::byte> bytes;
  expert::runtime::ExpertSections sections;
};

std::vector<Record> load_records(const std::filesystem::path& path,
                                 std::uint32_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open expert pack");
  std::vector<Record> records;
  std::uint64_t offset = 0;
  for (std::uint32_t id = 0; id < count; ++id) {
    std::array<std::byte, 148> header{};
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char*>(header.data()), header.size());
    if (input.gcount() != static_cast<std::streamsize>(header.size())) break;
    const auto stored = read_le<std::uint64_t>(header.data() + 44);
    if (stored < header.size() || stored > 64ULL * 1024 * 1024)
      throw std::runtime_error("invalid expert record size");
    Record record;
    record.bytes.resize(static_cast<std::size_t>(stored));
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char*>(record.bytes.data()),
               static_cast<std::streamsize>(record.bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(record.bytes.size()))
      throw std::runtime_error("short expert record");
    expert::runtime::PayloadRecord expected;
    expected.path = path;
    expected.record_offset = offset;
    expected.stored_bytes = stored;
    std::copy_n(header.data() + 116, expected.payload_sha256.size(),
                expected.payload_sha256.begin());
    const auto result = expert::runtime::validate_expert_record(
        record.bytes, {0, 0, id, 1}, expected);
    if (!result.status.ok())
      throw std::runtime_error(std::string(result.status.message()));
    record.sections = result.record.sections;
    records.push_back(std::move(record));
    offset += stored;
  }
  if (records.empty()) throw std::runtime_error("pack has no complete records");
  return records;
}

float horizontal_sum(__m256 value) {
  const auto low = _mm256_castps256_ps128(value);
  const auto high = _mm256_extractf128_ps(value, 1);
  auto sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

__m256 load_i8x8(const std::int8_t* source) {
  const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source));
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
}

template <typename T>
const T* section(const Record& record, std::uint64_t offset) {
  return reinterpret_cast<const T*>(record.bytes.data() + offset);
}

void run_expert_avx2(const Record& record, std::span<const float> inputs,
                     std::uint32_t rows, std::span<float> intermediate,
                     std::span<float> outputs) {
  const auto& s = record.sections;
  const auto hidden = s.hidden;
  const auto width = s.intermediate;
  const auto* gate_up = section<std::int8_t>(record, s.gate_up_q_offset);
  const auto* gate_scales = section<float>(record, s.gate_up_scale_offset);
  const auto* down = section<std::int8_t>(record, s.down_q_offset);
  const auto* down_scales = section<float>(record, s.down_scale_offset);
  std::vector<__m256> gate_sum(rows);
  std::vector<__m256> up_sum(rows);
  for (std::uint32_t output = 0; output < width; ++output) {
    std::fill(gate_sum.begin(), gate_sum.end(), _mm256_setzero_ps());
    std::fill(up_sum.begin(), up_sum.end(), _mm256_setzero_ps());
    const auto* gate = gate_up + static_cast<std::size_t>(output) * hidden;
    const auto* up = gate_up + static_cast<std::size_t>(width + output) * hidden;
    for (std::uint32_t column = 0; column < hidden; column += 8) {
      const auto gate_weight = load_i8x8(gate + column);
      const auto up_weight = load_i8x8(up + column);
      for (std::uint32_t row = 0; row < rows; ++row) {
        const auto activation = _mm256_loadu_ps(
            inputs.data() + static_cast<std::size_t>(row) * hidden + column);
        gate_sum[row] = _mm256_fmadd_ps(gate_weight, activation, gate_sum[row]);
        up_sum[row] = _mm256_fmadd_ps(up_weight, activation, up_sum[row]);
      }
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto gate_value = horizontal_sum(gate_sum[row]) * gate_scales[output];
      const auto up_value = horizontal_sum(up_sum[row]) * gate_scales[width + output];
      intermediate[static_cast<std::size_t>(row) * width + output] =
          (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
    }
  }
  std::vector<__m256> down_sum(rows);
  for (std::uint32_t output = 0; output < hidden; ++output) {
    std::fill(down_sum.begin(), down_sum.end(), _mm256_setzero_ps());
    const auto* weights = down + static_cast<std::size_t>(output) * width;
    for (std::uint32_t column = 0; column < width; column += 8) {
      const auto weight = load_i8x8(weights + column);
      for (std::uint32_t row = 0; row < rows; ++row) {
        const auto activation = _mm256_loadu_ps(
            intermediate.data() + static_cast<std::size_t>(row) * width + column);
        down_sum[row] = _mm256_fmadd_ps(weight, activation, down_sum[row]);
      }
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
      outputs[static_cast<std::size_t>(row) * hidden + output] =
          horizontal_sum(down_sum[row]) * down_scales[output];
    }
  }
}

std::vector<float> run_expert_reference(const Record& record,
                                        std::span<const float> input) {
  const auto& s = record.sections;
  const auto* gate_up = section<std::int8_t>(record, s.gate_up_q_offset);
  const auto* gate_scales = section<float>(record, s.gate_up_scale_offset);
  const auto* down = section<std::int8_t>(record, s.down_q_offset);
  const auto* down_scales = section<float>(record, s.down_scale_offset);
  std::vector<float> intermediate(s.intermediate);
  std::vector<float> output(s.hidden);
  for (std::uint32_t row = 0; row < s.intermediate; ++row) {
    double gate = 0.0;
    double up = 0.0;
    for (std::uint32_t column = 0; column < s.hidden; ++column) {
      gate += static_cast<float>(
                  gate_up[static_cast<std::size_t>(row) * s.hidden + column]) *
              input[column];
      up += static_cast<float>(
                gate_up[(static_cast<std::size_t>(s.intermediate) + row) *
                            s.hidden +
                        column]) *
            input[column];
    }
    const auto gate_value = static_cast<float>(gate) * gate_scales[row];
    const auto up_value =
        static_cast<float>(up) * gate_scales[s.intermediate + row];
    intermediate[row] =
        (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
  }
  for (std::uint32_t row = 0; row < s.hidden; ++row) {
    double sum = 0.0;
    for (std::uint32_t column = 0; column < s.intermediate; ++column) {
      sum += static_cast<float>(
                 down[static_cast<std::size_t>(row) * s.intermediate + column]) *
             intermediate[column];
    }
    output[row] = static_cast<float>(sum) * down_scales[row];
  }
  return output;
}

double seconds_since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 5) {
      std::cerr << "usage: expert-path-probe <experts.qpack> [records=64] "
                   "[rows=4] [threads=6]\n";
      return 64;
    }
    const auto record_count = argc > 2 ? std::stoul(argv[2]) : 64UL;
    const auto rows = argc > 3 ? std::stoul(argv[3]) : 4UL;
    const auto threads = argc > 4 ? std::stoul(argv[4]) : 6UL;
    if (record_count == 0 || rows == 0 || rows > 32 || threads == 0)
      throw std::runtime_error("invalid probe dimensions");
    auto records = load_records(argv[1], static_cast<std::uint32_t>(record_count));
    const auto hidden = records.front().sections.hidden;
    const auto width = records.front().sections.intermediate;
    const auto slab_bytes = std::accumulate(
        records.begin(), records.end(), std::size_t{0},
        [](std::size_t total, const Record& record) {
          return total + record.bytes.size();
        });
    std::vector<float> inputs(rows * hidden);
    for (std::size_t i = 0; i < inputs.size(); ++i)
      inputs[i] = std::sin(static_cast<float>(i) * 0.013F) * 0.25F;

    void* pinned{};
    void* device{};
    cuda_check(cudaHostAlloc(&pinned, slab_bytes, cudaHostAllocDefault),
               "cudaHostAlloc slab");
    cuda_check(cudaMalloc(&device, slab_bytes), "cudaMalloc slab");
    auto pack_started = Clock::now();
    auto* cursor = static_cast<std::byte*>(pinned);
    for (const auto& record : records) {
      std::memcpy(cursor, record.bytes.data(), record.bytes.size());
      cursor += record.bytes.size();
    }
    const auto pack_seconds = seconds_since(pack_started);
    cuda_check(cudaMemcpy(device, pinned, slab_bytes, cudaMemcpyHostToDevice),
               "warm slab H2D");
    constexpr std::uint32_t h2d_iterations = 5;
    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    for (std::uint32_t i = 0; i < h2d_iterations; ++i)
      cuda_check(cudaMemcpyAsync(device, pinned, slab_bytes,
                                 cudaMemcpyHostToDevice), "slab H2D");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float h2d_ms{};
    cuda_check(cudaEventElapsedTime(&h2d_ms, start, stop), "cudaEventElapsedTime");

    std::vector<std::vector<float>> intermediates(
        records.size(), std::vector<float>(rows * width));
    std::vector<std::vector<float>> outputs(
        records.size(), std::vector<float>(rows * hidden));
    run_expert_avx2(records.front(), inputs, static_cast<std::uint32_t>(rows),
                    intermediates.front(), outputs.front());
    const auto reference = run_expert_reference(
        records.front(), std::span<const float>(inputs.data(), hidden));
    double max_abs = 0.0;
    double dot = 0.0;
    double norm_actual = 0.0;
    double norm_reference = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
      const auto actual = outputs.front()[i];
      max_abs = std::max(max_abs, std::abs(static_cast<double>(actual) -
                                           reference[i]));
      dot += static_cast<double>(actual) * reference[i];
      norm_actual += static_cast<double>(actual) * actual;
      norm_reference += static_cast<double>(reference[i]) * reference[i];
    }
    const auto cosine =
        dot / std::sqrt(std::max(1e-30, norm_actual * norm_reference));
    std::atomic<std::size_t> next{0};
    const auto cpu_started = Clock::now();
    std::vector<std::thread> workers;
    for (std::size_t thread = 0; thread < threads; ++thread) {
      workers.emplace_back([&] {
        for (;;) {
          const auto index = next.fetch_add(1);
          if (index >= records.size()) return;
          run_expert_avx2(records[index], inputs, static_cast<std::uint32_t>(rows),
                          intermediates[index], outputs[index]);
        }
      });
    }
    for (auto& worker : workers) worker.join();
    const auto cpu_seconds = seconds_since(cpu_started);
    const auto gib = 1024.0 * 1024.0 * 1024.0;
    const auto h2d_seconds = static_cast<double>(h2d_ms) / 1000.0;
    std::cout << "{\"records\":" << records.size()
              << ",\"rows_per_expert\":" << rows
              << ",\"threads\":" << threads
              << ",\"record_bytes\":" << records.front().bytes.size()
              << ",\"slab_bytes\":" << slab_bytes
              << ",\"host_pack_seconds\":" << pack_seconds
              << ",\"host_pack_gib_per_second\":"
              << (slab_bytes / gib / pack_seconds)
              << ",\"pinned_h2d_seconds_per_slab\":"
              << (h2d_seconds / h2d_iterations)
              << ",\"pinned_h2d_gib_per_second\":"
              << (slab_bytes * h2d_iterations / gib / h2d_seconds)
              << ",\"cpu_seconds\":" << cpu_seconds
              << ",\"cpu_experts_per_second\":"
              << (records.size() / cpu_seconds)
              << ",\"cpu_effective_weight_gib_per_second\":"
              << (slab_bytes / gib / cpu_seconds)
              << ",\"cpu_reference_max_abs\":" << max_abs
              << ",\"cpu_reference_cosine\":" << cosine
              << ",\"checksum\":"
              << std::accumulate(outputs.front().begin(), outputs.front().end(), 0.0)
              << "}\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device);
    cudaFreeHost(pinned);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
