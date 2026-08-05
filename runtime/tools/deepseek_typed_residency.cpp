#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_typed.hpp"
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
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}
std::filesystem::path relative(const std::string& text) {
  const std::filesystem::path path = text;
  require(!path.empty() && !path.is_absolute(), "manifest path must be relative");
  for (const auto& part : path) require(part != "..", "manifest path escapes root");
  return path;
}
std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> result;
  std::size_t start = 0U;
  do {
    const auto separator = line.find('\t', start);
    result.push_back(line.substr(start, separator - start));
    if (separator == std::string::npos) break;
    start = separator + 1U;
  } while (true);
  return result;
}
std::uint8_t nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::runtime_error("invalid SHA-256 hex digit");
}
er::Sha256Digest digest(const std::string& text) {
  require(text.size() == 64U, "invalid SHA-256 length");
  er::Sha256Digest result{};
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = static_cast<std::byte>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  return result;
}
er::PayloadExtent extent(const std::filesystem::path& descriptor,
                         const std::filesystem::path& source_root) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid typed extent header");
  require(static_cast<bool>(std::getline(input, line)), "missing typed extent");
  const auto item = fields(line);
  require(item.size() == 4U && !std::getline(input, line),
          "typed tensor requires one extent");
  return {source_root / relative(item[3]), std::stoull(item[2]),
          std::stoull(item[0]), std::stoull(item[1])};
}
er::cuda::DeepSeekDtype dtype(const std::string& text) {
  if (text == "BF16") return er::cuda::DeepSeekDtype::bf16;
  if (text == "F32") return er::cuda::DeepSeekDtype::f32;
  if (text == "I64") return er::cuda::DeepSeekDtype::i64;
  throw std::runtime_error("unsupported typed residency dtype");
}
std::vector<er::cuda::DeepSeekTypedSpec> specs(
    const std::filesystem::path& root,
    const std::filesystem::path& source_root, std::uint64_t& bytes) {
  std::ifstream input(root / "typed-set.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-typed-residency-v1",
          "invalid typed residency header");
  std::vector<er::cuda::DeepSeekTypedSpec> result;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 6U, "invalid typed residency row");
    er::PayloadRecord record;
    record.extents.push_back(extent(root / relative(item[5]), source_root));
    record.stored_bytes = std::stoull(item[3]);
    record.payload_sha256 = digest(item[4]);
    result.push_back({item[0], std::move(record), dtype(item[1])});
    bytes += std::stoull(item[3]);
  }
  require(input.eof() && result.size() == 834U,
          "typed residency set must contain 834 tensors");
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: expert-deepseek-typed-residency "
                   "<descriptor-root> <checkpoint-root>\n";
      return 64;
    }
    std::uint64_t source_bytes = 0U;
    const auto declarations = specs(argv[1], argv[2], source_bytes);
    require(source_bytes == 2'988'256'348ULL,
            "typed residency source byte contract changed");
    std::size_t free_before = 0U, total = 0U;
    check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo before typed");
    require(free_before >= source_bytes + 512ULL * 1024U * 1024U,
            "insufficient VRAM headroom for typed residency set");
    constexpr std::size_t staging_bytes = 64U * 1024U * 1024U;
    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    er::ExtentGatherStorage storage(iocp);
    er::FixedBufferPool buffers(
        1U, staging_bytes, 4096U, std::make_shared<er::CudaPinnedAllocator>());
    er::cuda::DeepSeekTypedSet resident;
    const auto started = std::chrono::steady_clock::now();
    const auto status = er::cuda::DeepSeekTypedSet::load(
        storage, buffers, declarations, resident);
    const auto stopped = std::chrono::steady_clock::now();
    require(status.ok(), std::string(status.message()));
    require(resident.size() == 834U && resident.bytes() == source_bytes,
            "typed residency set is incomplete");
    for (const auto& declaration : declarations)
      require(resident.find(declaration.name) != nullptr,
              "typed residency lookup missed a declared tensor");
    std::size_t free_resident = 0U;
    check(cudaMemGetInfo(&free_resident, &total), "cudaMemGetInfo typed resident");
    resident.clear();
    const auto elapsed = std::chrono::duration<double, std::milli>(
                             stopped - started).count();
    std::cout << "{\"ok\":true,\"tensors\":" << declarations.size()
              << ",\"source_bytes\":" << source_bytes
              << ",\"resident_bytes\":" << source_bytes
              << ",\"staging_bytes\":" << staging_bytes
              << ",\"startup_ms\":" << elapsed
              << ",\"cuda_free_before\":" << free_before
              << ",\"cuda_free_resident\":" << free_resident << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-typed-residency: " << error.what() << '\n';
    return 1;
  }
}
