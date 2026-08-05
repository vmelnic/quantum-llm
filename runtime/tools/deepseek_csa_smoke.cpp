#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_csa.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
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
constexpr std::uint32_t kHeadDim = 512U;
constexpr std::size_t kNormBytes = kHeadDim * sizeof(std::uint16_t);

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::filesystem::path relative(const std::string& text) {
  const std::filesystem::path path = text;
  require(!path.empty() && !path.is_absolute(), "source path must be relative");
  for (const auto& part : path) require(part != "..", "source path escapes root");
  return path;
}
std::vector<er::PayloadExtent> extents(const std::filesystem::path& descriptor,
                                       const std::filesystem::path& source) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1", "invalid CSA descriptor");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    std::vector<std::string> item;
    std::size_t start = 0U;
    do {
      const auto separator = line.find('\t', start);
      item.push_back(line.substr(start, separator - start));
      if (separator == std::string::npos) break;
      start = separator + 1U;
    } while (true);
    require(item.size() == 4U, "invalid CSA extent row");
    result.push_back({source / relative(item[3]), std::stoull(item[2]),
                      std::stoull(item[0]), std::stoull(item[1])});
  }
  require(input.eof() && result.size() == 4U,
          "CSA slice requires wkv, wgate, ape and norm extents");
  return result;
}
std::string hex(const er::Sha256Digest& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : digest)
    output << std::setw(2) << std::to_integer<unsigned>(value);
  return output.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: expert-deepseek-csa-smoke <bundle> <checkpoint> "
                   "<source-sha256> <layer> <ratio>\n";
      return 64;
    }
    const auto layer = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto ratio = static_cast<std::uint32_t>(std::stoul(argv[5]));
    require(ratio == 4U || ratio == 128U, "unsupported CSA ratio");
    const auto width = ratio == 4U ? 1024U : 512U;
    const auto matrix_bytes =
        static_cast<std::size_t>(width) * kHidden * sizeof(std::uint16_t);
    const auto ape_bytes = static_cast<std::size_t>(ratio) * width * sizeof(float);
    const auto source_bytes = 2U * matrix_bytes + ape_bytes + kNormBytes;
    auto pool = std::make_shared<er::FixedBufferPool>(
        1U, source_bytes, 4096U, std::make_shared<er::CudaPinnedAllocator>());
    auto lease = pool->try_acquire(source_bytes);
    require(lease != nullptr, "could not acquire CSA staging slot");
    er::PayloadRecord record;
    record.extents = extents(std::filesystem::path(argv[1]) / "extents.tsv", argv[2]);
    record.stored_bytes = source_bytes;
    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    er::ExtentGatherStorage storage(iocp);
    std::promise<er::ReadResult> promise;
    auto future = promise.get_future();
    static_cast<void>(storage.read(
        {record, lease->buffer(), true},
        [&promise](er::ReadResult result) { promise.set_value(std::move(result)); }));
    const auto read = future.get();
    require(read.status.ok() && read.read_bytes == source_bytes,
            std::string(read.status.message()));
    const auto source = std::span<const std::byte>(lease->buffer().data, source_bytes);
    require(hex(er::sha256(source)) == argv[3], "CSA source hash mismatch");

    std::ifstream oracle_file(std::filesystem::path(argv[1]) / "oracle.f32",
                              std::ios::binary | std::ios::ate);
    require(static_cast<bool>(oracle_file), "missing CSA oracle");
    const auto oracle_bytes = static_cast<std::size_t>(oracle_file.tellg());
    require(oracle_bytes ==
                (static_cast<std::size_t>(ratio) * kHidden + kHeadDim) * sizeof(float),
            "invalid CSA oracle geometry");
    oracle_file.seekg(0);
    std::vector<float> oracle(oracle_bytes / sizeof(float));
    oracle_file.read(reinterpret_cast<char*>(oracle.data()),
                     static_cast<std::streamsize>(oracle_bytes));
    require(static_cast<bool>(oracle_file), "truncated CSA oracle");

    std::uint16_t *wkv = nullptr, *wgate = nullptr, *norm = nullptr;
    float *ape = nullptr, *input = nullptr, *projected_values = nullptr;
    float *projected_scores = nullptr, *pooled = nullptr, *output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&wkv), matrix_bytes), "allocate CSA wkv");
    check(cudaMalloc(reinterpret_cast<void**>(&wgate), matrix_bytes), "allocate CSA wgate");
    check(cudaMalloc(reinterpret_cast<void**>(&ape), ape_bytes), "allocate CSA ape");
    check(cudaMalloc(reinterpret_cast<void**>(&norm), kNormBytes), "allocate CSA norm");
    check(cudaMalloc(reinterpret_cast<void**>(&input), kHidden * sizeof(float)),
          "allocate CSA input");
    check(cudaMalloc(reinterpret_cast<void**>(&projected_values), width * sizeof(float)),
          "allocate CSA projected values");
    check(cudaMalloc(reinterpret_cast<void**>(&projected_scores), width * sizeof(float)),
          "allocate CSA projected scores");
    check(cudaMalloc(reinterpret_cast<void**>(&pooled), kHeadDim * sizeof(float)),
          "allocate CSA pooled");
    check(cudaMalloc(reinterpret_cast<void**>(&output), kHeadDim * sizeof(float)),
          "allocate CSA output");
    check(cudaMemcpy(wkv, source.data(), matrix_bytes, cudaMemcpyHostToDevice),
          "copy CSA wkv");
    check(cudaMemcpy(wgate, source.data() + matrix_bytes, matrix_bytes,
                     cudaMemcpyHostToDevice), "copy CSA wgate");
    check(cudaMemcpy(ape, source.data() + 2U * matrix_bytes, ape_bytes,
                     cudaMemcpyHostToDevice), "copy CSA ape");
    check(cudaMemcpy(norm, source.data() + 2U * matrix_bytes + ape_bytes,
                     kNormBytes, cudaMemcpyHostToDevice), "copy CSA norm");
    auto state = er::cuda::create_deepseek_compressor_state(ratio);
    require(state.status.ok() && state.state, std::string(state.status.message()));
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create CSA start event");
    check(cudaEventCreate(&stop), "create CSA stop event");
    check(cudaEventRecord(start), "record CSA start");
    for (std::uint32_t position = 0; position < ratio; ++position) {
      check(cudaMemcpy(input, oracle.data() + static_cast<std::size_t>(position) * kHidden,
                       kHidden * sizeof(float), cudaMemcpyHostToDevice), "copy CSA input");
      auto status = er::cuda::gemv_bf16(wkv, width, kHidden, input,
                                        projected_values, nullptr);
      require(status.ok(), std::string(status.message()));
      status = er::cuda::gemv_bf16(wgate, width, kHidden, input,
                                   projected_scores, nullptr);
      require(status.ok(), std::string(status.message()));
      status = er::cuda::deepseek_compressor_decode(
          *state.state, projected_values, projected_scores, ape, norm, pooled,
          output, position, 1e-6F, nullptr);
      require(status.ok(), std::string(status.message()));
    }
    check(cudaEventRecord(stop), "record CSA stop");
    check(cudaEventSynchronize(stop), "synchronize CSA");
    float execution_ms = 0.0F;
    check(cudaEventElapsedTime(&execution_ms, start, stop), "measure CSA");
    std::vector<float> actual(kHeadDim);
    check(cudaMemcpy(actual.data(), output, kHeadDim * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy CSA output");
    const auto* expected = oracle.data() + static_cast<std::size_t>(ratio) * kHidden;
    double squared = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const float error = std::abs(actual[index] - expected[index]);
      maximum = std::max(maximum, error);
      squared += static_cast<double>(error) * error;
    }
    require(maximum < 5e-4F, "CSA output exceeds FP32 oracle tolerance");
    std::cout << "{\"ok\":true,\"layer\":" << layer
              << ",\"compress_ratio\":" << ratio
              << ",\"source_bytes\":" << source_bytes
              << ",\"state_bytes\":" << state.state->bytes()
              << ",\"group_ms\":" << execution_ms
              << ",\"output_rmse\":" << std::sqrt(squared / actual.size())
              << ",\"output_max_abs_error\":" << maximum << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-csa-smoke: " << error.what() << '\n';
    return 1;
  }
}
