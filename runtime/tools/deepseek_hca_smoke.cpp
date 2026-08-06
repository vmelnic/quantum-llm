#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_hca.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace er = expert::runtime;

namespace {

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kStreams = 4U;
constexpr std::size_t kFunctionValues = 24U * kStreams * kHidden;
constexpr std::size_t kBaseValues = 24U;
constexpr std::size_t kScaleValues = 3U;
constexpr std::size_t kSourceBytes =
    (kFunctionValues + kBaseValues + kScaleValues) * sizeof(float);

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

std::filesystem::path safe_relative(const std::string& text) {
  const std::filesystem::path path = text;
  require(!path.empty() && !path.is_absolute(), "source shard must be relative");
  for (const auto& part : path) require(part != "..", "source shard escapes root");
  return path;
}

std::vector<er::PayloadExtent> read_extents(
    const std::filesystem::path& descriptor,
    const std::filesystem::path& source_root) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid HCA extent descriptor");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    const auto a = line.find('\t');
    const auto b = line.find('\t', a + 1U);
    const auto c = line.find('\t', b + 1U);
    require(a != std::string::npos && b != std::string::npos &&
                c != std::string::npos &&
                line.find('\t', c + 1U) == std::string::npos,
            "invalid HCA extent row");
    result.push_back({source_root / safe_relative(line.substr(c + 1U)),
                      std::stoull(line.substr(b + 1U, c - b - 1U)),
                      std::stoull(line.substr(0U, a)),
                      std::stoull(line.substr(a + 1U, b - a - 1U))});
  }
  require(input.eof() && result.size() == 3U,
          "HCA slice requires fn, base and scale extents");
  return result;
}

std::string hex_digest(const er::Sha256Digest& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : digest)
    output << std::setw(2) << std::to_integer<unsigned>(value);
  return output.str();
}

struct ErrorStats final {
  double squared{};
  float maximum{};
  std::size_t count{};

  void add(const float* actual, const float* expected, std::size_t values) {
    for (std::size_t index = 0; index < values; ++index) {
      const float error = std::abs(actual[index] - expected[index]);
      maximum = std::max(maximum, error);
      squared += static_cast<double>(error) * error;
    }
    count += values;
  }
  [[nodiscard]] double rmse() const { return std::sqrt(squared / count); }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: expert-deepseek-hca-smoke <bundle> <checkpoint> "
                   "<source-sha256>\n";
      return 64;
    }
    const std::filesystem::path bundle = argv[1];
    const std::filesystem::path source_root = argv[2];
    const std::string source_hash = argv[3];
    auto pool = std::make_shared<er::FixedBufferPool>(
        1U, kSourceBytes, 4096U, std::make_shared<er::CudaPinnedAllocator>());
    auto lease = pool->try_acquire(kSourceBytes);
    require(lease != nullptr, "could not acquire HCA staging slot");
    er::PayloadRecord record;
    record.extents = read_extents(bundle / "extents.tsv", source_root);
    record.stored_bytes = kSourceBytes;
    auto iocp = std::make_shared<er::WindowsIocpStorage>(1U);
    er::ExtentGatherStorage storage(iocp);
    std::promise<er::ReadResult> promise;
    auto future = promise.get_future();
    static_cast<void>(storage.read(
        {record, lease->buffer(), true},
        [&promise](er::ReadResult result) { promise.set_value(std::move(result)); }));
    const auto read = future.get();
    require(read.status.ok() && read.read_bytes == kSourceBytes,
            std::string(read.status.message()));
    const auto complete =
        std::span<const std::byte>(lease->buffer().data, kSourceBytes);
    require(hex_digest(er::sha256(complete)) == source_hash,
            "HCA source payload hash mismatch");

    constexpr std::size_t function_bytes = kFunctionValues * sizeof(float);
    constexpr std::size_t base_bytes = kBaseValues * sizeof(float);
    const auto admission_started = std::chrono::steady_clock::now();
    auto admitted = er::cuda::admit_deepseek_hca(
        complete.first(function_bytes),
        complete.subspan(function_bytes, base_bytes),
        complete.subspan(function_bytes + base_bytes,
                         kScaleValues * sizeof(float)),
        kHidden);
    require(admitted.status.ok() && admitted.parameters,
            std::string(admitted.status.message()));
    const auto admission_stopped = std::chrono::steady_clock::now();

    std::ifstream oracle_file(bundle / "oracle.f32", std::ios::binary | std::ios::ate);
    require(static_cast<bool>(oracle_file), "missing HCA oracle");
    const auto oracle_bytes = static_cast<std::size_t>(oracle_file.tellg());
    require(oracle_bytes % sizeof(float) == 0U, "invalid HCA oracle length");
    oracle_file.seekg(0);
    std::vector<float> oracle(oracle_bytes / sizeof(float));
    oracle_file.read(reinterpret_cast<char*>(oracle.data()),
                     static_cast<std::streamsize>(oracle_bytes));
    require(static_cast<bool>(oracle_file), "truncated HCA oracle");
    constexpr std::size_t expected_oracle_values =
        kStreams * kHidden + kHidden + 4U + 4U + 16U + kHidden +
        kStreams * kHidden;
    require(oracle.size() == expected_oracle_values,
            "unexpected HCA oracle geometry");
    std::size_t offset = 0U;
    const float* host_streams = oracle.data() + offset;
    offset += kStreams * kHidden;
    const float* host_sublayer = oracle.data() + offset;
    offset += kHidden;
    const float* expected_pre = oracle.data() + offset;
    offset += 4U;
    const float* expected_post = oracle.data() + offset;
    offset += 4U;
    const float* expected_comb = oracle.data() + offset;
    offset += 16U;
    const float* expected_collapsed = oracle.data() + offset;
    offset += kHidden;
    const float* expected_updated = oracle.data() + offset;

    float *streams = nullptr, *sublayer = nullptr, *collapsed = nullptr;
    float *pre = nullptr, *post = nullptr, *comb = nullptr, *updated = nullptr;
    float *normalized = nullptr, *mixes = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&streams),
                     kStreams * kHidden * sizeof(float)), "allocate HCA streams");
    check(cudaMalloc(reinterpret_cast<void**>(&sublayer), kHidden * sizeof(float)),
          "allocate HCA sublayer");
    check(cudaMalloc(reinterpret_cast<void**>(&collapsed), kHidden * sizeof(float)),
          "allocate HCA collapsed");
    check(cudaMalloc(reinterpret_cast<void**>(&pre), 4U * sizeof(float)),
          "allocate HCA pre");
    check(cudaMalloc(reinterpret_cast<void**>(&post), 4U * sizeof(float)),
          "allocate HCA post");
    check(cudaMalloc(reinterpret_cast<void**>(&comb), 16U * sizeof(float)),
          "allocate HCA comb");
    check(cudaMalloc(reinterpret_cast<void**>(&updated),
                     kStreams * kHidden * sizeof(float)), "allocate HCA updated");
    check(cudaMalloc(reinterpret_cast<void**>(&normalized),
                     kStreams * kHidden * sizeof(float)), "allocate HCA normalized");
    check(cudaMalloc(reinterpret_cast<void**>(&mixes), 24U * sizeof(float)),
          "allocate HCA mixes");
    check(cudaMemcpy(streams, host_streams, kStreams * kHidden * sizeof(float),
                     cudaMemcpyHostToDevice), "copy HCA streams");
    check(cudaMemcpy(sublayer, host_sublayer, kHidden * sizeof(float),
                     cudaMemcpyHostToDevice), "copy HCA sublayer");
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create HCA start event");
    check(cudaEventCreate(&stop), "create HCA stop event");
    constexpr std::uint32_t kBenchmarkIterations = 100U;
    auto status = er::cuda::deepseek_hca_pre(
        *admitted.parameters, streams, collapsed, pre, post, comb,
        {normalized, mixes}, 1e-6F, 20U, nullptr);
    require(status.ok(), std::string(status.message()));
    status = er::cuda::deepseek_hca_post(sublayer, streams, post, comb, updated,
                                         kHidden, nullptr);
    require(status.ok(), std::string(status.message()));
    check(cudaDeviceSynchronize(), "warm HCA path");
    check(cudaEventRecord(start), "record HCA start");
    for (std::uint32_t iteration = 0U; iteration < kBenchmarkIterations;
         ++iteration) {
      status = er::cuda::deepseek_hca_pre(
          *admitted.parameters, streams, collapsed, pre, post, comb,
          {normalized, mixes}, 1e-6F, 20U, nullptr);
      require(status.ok(), std::string(status.message()));
      status = er::cuda::deepseek_hca_post(
          sublayer, streams, post, comb, updated, kHidden, nullptr);
      require(status.ok(), std::string(status.message()));
    }
    check(cudaEventRecord(stop), "record HCA stop");
    check(cudaEventSynchronize(stop), "synchronize HCA");
    float execution_ms = 0.0F;
    check(cudaEventElapsedTime(&execution_ms, start, stop), "measure HCA");
    execution_ms /= static_cast<float>(kBenchmarkIterations);
    std::vector<float> actual_pre(4), actual_post(4), actual_comb(16);
    std::vector<float> actual_collapsed(kHidden), actual_updated(kStreams * kHidden);
    check(cudaMemcpy(actual_pre.data(), pre, 4U * sizeof(float), cudaMemcpyDeviceToHost),
          "copy HCA pre");
    check(cudaMemcpy(actual_post.data(), post, 4U * sizeof(float), cudaMemcpyDeviceToHost),
          "copy HCA post");
    check(cudaMemcpy(actual_comb.data(), comb, 16U * sizeof(float), cudaMemcpyDeviceToHost),
          "copy HCA comb");
    check(cudaMemcpy(actual_collapsed.data(), collapsed, kHidden * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy HCA collapsed");
    check(cudaMemcpy(actual_updated.data(), updated, kStreams * kHidden * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy HCA updated");
    ErrorStats control_error;
    control_error.add(actual_pre.data(), expected_pre, 4U);
    control_error.add(actual_post.data(), expected_post, 4U);
    control_error.add(actual_comb.data(), expected_comb, 16U);
    ErrorStats output_error;
    output_error.add(actual_collapsed.data(), expected_collapsed, kHidden);
    output_error.add(actual_updated.data(), expected_updated, kStreams * kHidden);
    require(control_error.maximum < 2e-4F && output_error.maximum < 2e-4F,
            "HCA CUDA output exceeds FP32 oracle tolerance");
    const auto admission_ms = std::chrono::duration<double, std::milli>(
                                  admission_stopped - admission_started).count();
    std::cout << "{\"ok\":true,\"source_abi\":\"deepseek-hca-f32-v1\""
              << ",\"hidden_size\":" << kHidden
              << ",\"hc_mult\":" << kStreams
              << ",\"source_bytes\":" << kSourceBytes
              << ",\"device_bytes\":" << admitted.parameters->bytes()
              << ",\"admission_ms\":" << admission_ms
              << ",\"benchmark_iterations\":" << kBenchmarkIterations
              << ",\"execution_ms\":" << execution_ms
              << ",\"control_rmse\":" << control_error.rmse()
              << ",\"control_max_abs_error\":" << control_error.maximum
              << ",\"output_rmse\":" << output_error.rmse()
              << ",\"output_max_abs_error\":" << output_error.maximum << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-hca-smoke: " << error.what() << '\n';
    return 1;
  }
}
