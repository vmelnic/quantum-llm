#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kTileDimensions = 16U;
constexpr std::uint32_t kQueriesPerKvHead = 6U;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

float half_to_float(std::uint16_t value) {
  const auto packed = _mm_cvtsi32_si128(value);
  const auto converted = _mm256_cvtph_ps(packed);
  return _mm_cvtss_f32(_mm256_castps256_ps128(converted));
}

void value_task(const std::uint16_t* values, const float* probabilities,
                std::uint32_t tokens, std::uint32_t head_dim,
                float* output) {
  for (std::uint32_t dimension = 0U; dimension < head_dim;
       dimension += kTileDimensions) {
    const auto tile = dimension / kTileDimensions;
    const auto* tile_values =
        values + static_cast<std::size_t>(tile) * tokens * kTileDimensions;
    __m256 a00 = _mm256_setzero_ps();
    __m256 a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps();
    __m256 a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps();
    __m256 a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps();
    __m256 a31 = _mm256_setzero_ps();
    __m256 a40 = _mm256_setzero_ps();
    __m256 a41 = _mm256_setzero_ps();
    __m256 a50 = _mm256_setzero_ps();
    __m256 a51 = _mm256_setzero_ps();
    for (std::uint32_t token = 0U; token < tokens; ++token) {
      const auto* source = tile_values +
                           static_cast<std::size_t>(token) * kTileDimensions;
      const auto v0 = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(source)));
      const auto v1 = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + 8U)));
      const auto* p = probabilities +
                      static_cast<std::size_t>(token) * kQueriesPerKvHead;
      auto probability = _mm256_broadcast_ss(p);
      a00 = _mm256_fmadd_ps(v0, probability, a00);
      a01 = _mm256_fmadd_ps(v1, probability, a01);
      probability = _mm256_broadcast_ss(p + 1U);
      a10 = _mm256_fmadd_ps(v0, probability, a10);
      a11 = _mm256_fmadd_ps(v1, probability, a11);
      probability = _mm256_broadcast_ss(p + 2U);
      a20 = _mm256_fmadd_ps(v0, probability, a20);
      a21 = _mm256_fmadd_ps(v1, probability, a21);
      probability = _mm256_broadcast_ss(p + 3U);
      a30 = _mm256_fmadd_ps(v0, probability, a30);
      a31 = _mm256_fmadd_ps(v1, probability, a31);
      probability = _mm256_broadcast_ss(p + 4U);
      a40 = _mm256_fmadd_ps(v0, probability, a40);
      a41 = _mm256_fmadd_ps(v1, probability, a41);
      probability = _mm256_broadcast_ss(p + 5U);
      a50 = _mm256_fmadd_ps(v0, probability, a50);
      a51 = _mm256_fmadd_ps(v1, probability, a51);
    }
    const auto store = [output, dimension](std::uint32_t query, __m256 low,
                                            __m256 high) {
      auto* destination = output +
                          static_cast<std::size_t>(query) * 256U + dimension;
      _mm256_storeu_ps(destination, low);
      _mm256_storeu_ps(destination + 8U, high);
    };
    store(0U, a00, a01);
    store(1U, a10, a11);
    store(2U, a20, a21);
    store(3U, a30, a31);
    store(4U, a40, a41);
    store(5U, a50, a51);
  }
}

void execute(const std::vector<std::uint16_t>& values,
             const std::vector<float>& probabilities, std::uint32_t tasks,
             std::uint32_t tokens, std::uint32_t head_dim,
             std::uint32_t threads, std::vector<float>& outputs) {
  const auto values_per_task = static_cast<std::size_t>(tokens) * head_dim;
  const auto probabilities_per_task =
      static_cast<std::size_t>(tokens) * kQueriesPerKvHead;
  const auto output_per_task =
      static_cast<std::size_t>(kQueriesPerKvHead) * head_dim;
  std::atomic<std::uint32_t> next{};
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (std::uint32_t worker = 0U; worker < threads; ++worker) {
    workers.emplace_back([&] {
      while (true) {
        const auto task = next.fetch_add(1U, std::memory_order_relaxed);
        if (task >= tasks) break;
        value_task(values.data() + task * values_per_task,
                   probabilities.data() + task * probabilities_per_task,
                   tokens, head_dim,
                   outputs.data() + task * output_per_task);
      }
    });
  }
  for (auto& worker : workers) worker.join();
}

double numerical_check() {
  constexpr std::uint32_t tokens = 31U;
  constexpr std::uint32_t dimensions = 256U;
  std::vector<std::uint16_t> values(
      static_cast<std::size_t>(tokens) * dimensions);
  for (std::uint32_t dimension = 0U; dimension < dimensions;
       dimension += kTileDimensions) {
    const auto tile = dimension / kTileDimensions;
    for (std::uint32_t token = 0U; token < tokens; ++token) {
      for (std::uint32_t lane = 0U; lane < kTileDimensions; ++lane) {
        const auto sign = ((token + dimension + lane) & 1U) << 15U;
        const auto mantissa = (token * 17U + dimension + lane) & 0x01ffU;
        values[(static_cast<std::size_t>(tile) * tokens + token) *
                   kTileDimensions +
               lane] = static_cast<std::uint16_t>(sign | 0x3800U | mantissa);
      }
    }
  }
  std::vector<float> probabilities(
      static_cast<std::size_t>(tokens) * kQueriesPerKvHead);
  for (std::uint32_t token = 0U; token < tokens; ++token)
    for (std::uint32_t query = 0U; query < kQueriesPerKvHead; ++query)
      probabilities[static_cast<std::size_t>(token) * kQueriesPerKvHead +
                    query] =
          static_cast<float>((token + 1U) * (query + 1U)) * 0.00003125F;
  std::vector<float> actual(kQueriesPerKvHead * dimensions);
  value_task(values.data(), probabilities.data(), tokens, dimensions,
             actual.data());
  double maximum_error{};
  for (std::uint32_t query = 0U; query < kQueriesPerKvHead; ++query) {
    for (std::uint32_t dimension = 0U; dimension < dimensions; ++dimension) {
      const auto tile = dimension / kTileDimensions;
      const auto lane = dimension % kTileDimensions;
      float expected{};
      for (std::uint32_t token = 0U; token < tokens; ++token) {
        const auto half = values[(static_cast<std::size_t>(tile) * tokens +
                                  token) *
                                     kTileDimensions +
                                 lane];
        expected = std::fma(
            half_to_float(half),
            probabilities[static_cast<std::size_t>(token) *
                              kQueriesPerKvHead +
                          query],
            expected);
      }
      maximum_error = std::max(
          maximum_error,
          std::abs(static_cast<double>(
              actual[static_cast<std::size_t>(query) * dimensions +
                     dimension] -
              expected)));
    }
  }
  return maximum_error;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 6 || argc > 8) {
      std::cerr << "usage: expert-fp16-cpu-value-profile <layers> <kv-heads> "
                   "<head-dim> <queries-per-kv-head> <tokens-per-layer> "
                   "[iterations] [threads]\n";
      return 64;
    }
    const auto layers = static_cast<std::uint32_t>(std::stoul(argv[1]));
    const auto kv_heads = static_cast<std::uint32_t>(std::stoul(argv[2]));
    const auto head_dim = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const auto queries = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto tokens = static_cast<std::uint32_t>(std::stoul(argv[5]));
    const auto iterations = argc >= 7
                                ? static_cast<std::uint32_t>(std::stoul(argv[6]))
                                : 3U;
    const auto threads = argc == 8
                             ? static_cast<std::uint32_t>(std::stoul(argv[7]))
                             : std::max(1U, std::thread::hardware_concurrency());
    require(layers != 0U && kv_heads != 0U && head_dim == 256U &&
                queries == kQueriesPerKvHead && tokens != 0U &&
                iterations != 0U && threads != 0U && threads <= 64U,
            "unsupported FP16 CPU value profile geometry");
    const auto tasks = layers * kv_heads;
    const auto value_count = static_cast<std::size_t>(tasks) * tokens * head_dim;
    const auto probability_count =
        static_cast<std::size_t>(tasks) * tokens * queries;
    std::vector<std::uint16_t> values(value_count, 0x3c00U);
    std::vector<float> probabilities(
        probability_count, 1.0F / static_cast<float>(tokens));
    std::vector<float> outputs(
        static_cast<std::size_t>(tasks) * queries * head_dim);
    const auto error = numerical_check();
    require(error == 0.0, "FP16 CPU value kernel failed its numerical gate");
    execute(values, probabilities, tasks, tokens, head_dim, threads, outputs);
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration)
      execute(values, probabilities, tasks, tokens, head_dim, threads, outputs);
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const auto seconds_per_iteration = seconds / iterations;
    const auto value_bytes = value_count * sizeof(std::uint16_t);
    const auto probability_bytes = probability_count * sizeof(float);
    double checksum{};
    for (std::size_t index = 0U; index < outputs.size(); ++index)
      checksum += outputs[index] * static_cast<double>((index % 97U) + 1U);
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"encoding\":\"IEEE-FP16\""
              << ",\"layers\":" << layers << ",\"kv_heads\":" << kv_heads
              << ",\"head_dim\":" << head_dim
              << ",\"queries_per_kv_head\":" << queries
              << ",\"tokens_per_layer\":" << tokens
              << ",\"threads\":" << threads
              << ",\"iterations\":" << iterations
              << ",\"value_bytes_per_target_call\":" << value_bytes
              << ",\"probability_bytes_per_target_call\":"
              << probability_bytes << ",\"milliseconds_per_target_call\":"
              << seconds_per_iteration * 1000.0
              << ",\"value_gb_per_second\":"
              << static_cast<double>(value_bytes) / seconds_per_iteration /
                     1.0e9
              << ",\"total_input_gb_per_second\":"
              << static_cast<double>(value_bytes + probability_bytes) /
                     seconds_per_iteration / 1.0e9
              << ",\"maximum_absolute_error\":" << error
              << ",\"output_checksum\":" << checksum << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
