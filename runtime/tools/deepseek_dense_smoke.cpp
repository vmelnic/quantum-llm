#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_dense.hpp"
#include "expert/runtime/deepseek_expert.hpp"
#include "expert/runtime/expert_record.hpp"
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

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
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
          "invalid dense extent descriptor");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    const auto a = line.find('\t');
    const auto b = line.find('\t', a + 1U);
    const auto c = line.find('\t', b + 1U);
    require(a != std::string::npos && b != std::string::npos &&
                c != std::string::npos &&
                line.find('\t', c + 1U) == std::string::npos,
            "invalid dense extent row");
    result.push_back({source_root / safe_relative(line.substr(c + 1U)),
                      std::stoull(line.substr(b + 1U, c - b - 1U)),
                      std::stoull(line.substr(0U, a)),
                      std::stoull(line.substr(a + 1U, b - a - 1U))});
  }
  require(input.eof() && result.size() == 2U,
          "dense matrix requires weight and scale extents");
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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      std::cerr << "usage: expert-deepseek-dense-smoke <bundle> <checkpoint> "
                   "<rows> <columns> <source-sha256> <candidate-sha256>\n";
      return 64;
    }
    const std::filesystem::path bundle = argv[1];
    const std::filesystem::path source_root = argv[2];
    const auto rows = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const auto columns = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const std::string expected_source_hash = argv[5];
    const std::string expected_candidate_hash = argv[6];
    const std::uint64_t weight_bytes =
        static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_bytes =
        static_cast<std::uint64_t>(rows / 128U) * (columns / 128U);
    const auto source_bytes = weight_bytes + scale_bytes;

    auto pool = std::make_shared<er::FixedBufferPool>(
        1U, source_bytes, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    auto lease = pool->try_acquire(source_bytes);
    require(lease != nullptr, "could not acquire dense staging slot");
    er::PayloadRecord record;
    record.extents = read_extents(bundle / "extents.tsv", source_root);
    record.stored_bytes = source_bytes;
    auto iocp = std::make_shared<er::WindowsIocpStorage>(1U);
    er::ExtentGatherStorage storage(iocp);
    std::promise<er::ReadResult> promise;
    auto future = promise.get_future();
    static_cast<void>(storage.read(
        {record, lease->buffer(), true},
        [&promise](er::ReadResult result) {
          promise.set_value(std::move(result));
        }));
    const auto read = future.get();
    require(read.status.ok() && read.read_bytes == source_bytes,
            std::string(read.status.message()));
    const auto complete = std::span<const std::byte>(lease->buffer().data,
                                                     source_bytes);
    require(hex_digest(er::sha256(complete)) == expected_source_hash,
            "dense source payload hash mismatch");

    const auto admission_started = std::chrono::steady_clock::now();
    auto admitted = er::cuda::admit_deepseek_dense_matrix(
        complete.first(weight_bytes), complete.subspan(weight_bytes, scale_bytes),
        rows, columns);
    const auto admission_stopped = std::chrono::steady_clock::now();
    require(admitted.status.ok() && admitted.matrix,
            std::string(admitted.status.message()));
    const auto matrix = admitted.matrix->view();
    std::vector<std::byte> candidate(admitted.matrix->bytes());
    check(cudaMemcpy(candidate.data(), matrix.weights, weight_bytes,
                     cudaMemcpyDeviceToHost),
          "copy dense INT8 weights");
    check(cudaMemcpy(candidate.data() + weight_bytes, matrix.scales,
                     static_cast<std::size_t>(rows) * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy dense row scales");
    const auto candidate_hash = hex_digest(er::sha256(candidate));
    require(candidate_hash == expected_candidate_hash,
            "dense candidate hash mismatch: " + candidate_hash);

    std::vector<float> input(columns);
    for (std::uint32_t index = 0U; index < columns; ++index) {
      input[index] = std::sin(static_cast<float>(index) * 0.017F) * 0.2F;
    }
    const auto* host_weights =
        reinterpret_cast<const std::int8_t*>(candidate.data());
    const auto* host_scales = reinterpret_cast<const float*>(
        candidate.data() + weight_bytes);
    std::vector<float> reference(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
      double sum = 0.0;
      for (std::uint32_t column = 0U; column < columns; ++column) {
        sum += host_weights[static_cast<std::size_t>(row) * columns + column] *
               input[column];
      }
      reference[row] = static_cast<float>(sum) * host_scales[row];
    }
    float* device_input = nullptr;
    float* device_output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_input),
                     input.size() * sizeof(float)),
          "cudaMalloc dense input");
    check(cudaMalloc(reinterpret_cast<void**>(&device_output),
                     reference.size() * sizeof(float)),
          "cudaMalloc dense output");
    check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "copy dense input");
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start), "create dense start event");
    check(cudaEventCreate(&stop), "create dense stop event");
    check(cudaEventRecord(start), "record dense start");
    const auto gemv_status =
        er::cuda::gemv(matrix, device_input, device_output, nullptr);
    require(gemv_status.ok(), std::string(gemv_status.message()));
    check(cudaEventRecord(stop), "record dense stop");
    check(cudaEventSynchronize(stop), "synchronize dense GEMV");
    float gemv_ms = 0.0F;
    check(cudaEventElapsedTime(&gemv_ms, start, stop), "dense GEMV elapsed");
    std::vector<float> output(rows);
    check(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy dense output");
    check(cudaEventDestroy(start), "destroy dense start event");
    check(cudaEventDestroy(stop), "destroy dense stop event");
    check(cudaFree(device_input), "free dense input");
    check(cudaFree(device_output), "free dense output");
    double squared = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < output.size(); ++index) {
      const auto error = std::abs(output[index] - reference[index]);
      maximum = std::max(maximum, error);
      squared += static_cast<double>(error) * error;
    }
    const auto admission_ms = std::chrono::duration<double, std::milli>(
                                  admission_stopped - admission_started)
                                  .count();
    std::cout << "{\"ok\":true,\"source_abi\":\""
              << er::kDeepSeekFp8Block128Abi << "\",\"target_abi\":\""
              << er::kDeepSeekSm86DenseAbi << "\",\"rows\":" << rows
              << ",\"columns\":" << columns
              << ",\"source_bytes\":" << source_bytes
              << ",\"device_bytes\":" << admitted.matrix->bytes()
              << ",\"sha256\":\"" << candidate_hash
              << "\",\"admission_ms\":" << admission_ms
              << ",\"gemv_ms\":" << gemv_ms
              << ",\"output_rmse\":"
              << std::sqrt(squared / output.size())
              << ",\"output_max_abs_error\":" << maximum << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-dense-smoke: " << error.what() << '\n';
    return 1;
  }
}
