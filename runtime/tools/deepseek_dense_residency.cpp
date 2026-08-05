#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_dense.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
  require(!path.empty() && !path.is_absolute(), "manifest path must be relative");
  for (const auto& part : path) require(part != "..", "manifest path escapes root");
  return path;
}

std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> result;
  std::size_t start = 0U;
  while (true) {
    const auto separator = line.find('\t', start);
    if (separator == std::string::npos) {
      result.push_back(line.substr(start));
      return result;
    }
    result.push_back(line.substr(start, separator - start));
    start = separator + 1U;
  }
}

std::uint8_t hex_nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::runtime_error("invalid SHA-256 hex digit");
}

er::Sha256Digest digest(const std::string& text) {
  require(text.size() == 64U, "invalid SHA-256 length");
  er::Sha256Digest result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] = static_cast<std::byte>(
        (hex_nibble(text[index * 2U]) << 4U) |
        hex_nibble(text[index * 2U + 1U]));
  }
  return result;
}

std::vector<er::PayloadExtent> extents(
    const std::filesystem::path& descriptor,
    const std::filesystem::path& source_root) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid dense extent header");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 4U, "invalid dense extent row");
    result.push_back({source_root / safe_relative(item[3]),
                      std::stoull(item[2]), std::stoull(item[0]),
                      std::stoull(item[1])});
  }
  require(input.eof() && result.size() == 2U,
          "dense matrix must have two extents");
  return result;
}

std::vector<er::cuda::DeepSeekDenseSpec> load_specs(
    const std::filesystem::path& root,
    const std::filesystem::path& source_root, std::uint64_t& source_bytes,
    std::uint64_t& device_bytes, std::uint64_t& maximum_source) {
  std::ifstream input(root / "dense-set.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-dense-residency-v1",
          "invalid dense residency header");
  std::vector<er::cuda::DeepSeekDenseSpec> result;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 7U, "invalid dense residency row");
    const auto rows = static_cast<std::uint32_t>(std::stoul(item[1]));
    const auto columns = static_cast<std::uint32_t>(std::stoul(item[2]));
    const auto stored = std::stoull(item[3]);
    const auto device = std::stoull(item[4]);
    require(stored == static_cast<std::uint64_t>(rows) * columns +
                          static_cast<std::uint64_t>(rows / 128U) *
                              (columns / 128U) &&
                device == static_cast<std::uint64_t>(rows) * columns +
                              static_cast<std::uint64_t>(rows) * sizeof(float),
            "dense residency byte claims do not match geometry");
    er::PayloadRecord record;
    record.extents = extents(root / safe_relative(item[6]), source_root);
    record.stored_bytes = stored;
    record.device_bytes = device;
    record.source_abi = er::kExpertSourceAbiDeepSeekFp8Block128V1;
    record.header_bytes = 0U;
    record.alignment = er::kExpertPackAlignment;
    record.payload_sha256 = digest(item[5]);
    result.push_back({item[0], std::move(record), rows, columns});
    source_bytes += stored;
    device_bytes += device;
    maximum_source = std::max(maximum_source, stored);
  }
  require(input.eof() && result.size() == 236U,
          "dense residency set must contain 236 matrices");
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: expert-deepseek-dense-residency "
                   "<descriptor-root> <checkpoint-root>\n";
      return 64;
    }
    std::uint64_t source_bytes = 0U;
    std::uint64_t device_bytes = 0U;
    std::uint64_t maximum_source = 0U;
    const auto specs = load_specs(argv[1], argv[2], source_bytes, device_bytes,
                                  maximum_source);
    std::size_t free_before = 0U;
    std::size_t total = 0U;
    check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo before dense");
    require(free_before >= device_bytes + 512ULL * 1024U * 1024U,
            "insufficient VRAM headroom for dense residency set");
    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    er::ExtentGatherStorage storage(iocp);
    er::FixedBufferPool buffers(
        1U, maximum_source, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    er::cuda::DeepSeekDenseSet resident;
    const auto started = std::chrono::steady_clock::now();
    const auto status = er::cuda::DeepSeekDenseSet::load(
        storage, buffers, specs, resident);
    const auto stopped = std::chrono::steady_clock::now();
    require(status.ok(), std::string(status.message()));
    require(resident.size() == specs.size() && resident.bytes() == device_bytes,
            "dense residency set is incomplete");
    for (const auto& spec : specs) {
      require(resident.find(spec.name) != nullptr,
              "dense residency lookup missed a declared matrix");
    }
    std::size_t free_resident = 0U;
    check(cudaMemGetInfo(&free_resident, &total),
          "cudaMemGetInfo dense resident");
    resident.clear();
    const auto elapsed = std::chrono::duration<double, std::milli>(
                             stopped - started)
                             .count();
    std::cout << "{\"ok\":true,\"matrices\":" << specs.size()
              << ",\"source_bytes\":" << source_bytes
              << ",\"resident_bytes\":" << device_bytes
              << ",\"maximum_staging_bytes\":" << maximum_source
              << ",\"startup_ms\":" << elapsed
              << ",\"cuda_free_before\":" << free_before
              << ",\"cuda_free_resident\":" << free_resident << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-dense-residency: " << error.what() << '\n';
    return 1;
  }
}
