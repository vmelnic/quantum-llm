#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
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
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}
std::filesystem::path relative(const std::string& text) {
  const std::filesystem::path path = text;
  require(!path.empty() && !path.is_absolute(), "descriptor path must be relative");
  for (const auto& part : path) require(part != "..", "descriptor path escapes root");
  return path;
}
std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> result;
  std::size_t start = 0U;
  while (true) {
    const auto separator = line.find('\t', start);
    result.push_back(line.substr(start, separator - start));
    if (separator == std::string::npos) return result;
    start = separator + 1U;
  }
}
std::uint8_t nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::runtime_error("invalid SHA-256 digit");
}
er::Sha256Digest digest(const std::string& text) {
  require(text.size() == 64U, "invalid SHA-256 length");
  er::Sha256Digest result{};
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = static_cast<std::byte>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  return result;
}
std::vector<er::PayloadExtent> extents(
    const std::filesystem::path& descriptor,
    const std::filesystem::path& source_root, std::size_t expected) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid model extent header");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 4U, "invalid model extent row");
    result.push_back({source_root / relative(item[3]), std::stoull(item[2]),
                      std::stoull(item[0]), std::stoull(item[1])});
  }
  require(input.eof() && result.size() == expected,
          "model tensor has the wrong extent count");
  return result;
}
er::cuda::DeepSeekDtype dtype(const std::string& text) {
  if (text == "BF16") return er::cuda::DeepSeekDtype::bf16;
  if (text == "F32") return er::cuda::DeepSeekDtype::f32;
  if (text == "I64") return er::cuda::DeepSeekDtype::i64;
  throw std::runtime_error("unsupported model tensor dtype");
}

std::vector<er::cuda::DeepSeekDenseSpec> dense_specs(
    const std::filesystem::path& root,
    const std::filesystem::path& source_root, std::uint64_t& source_bytes,
    std::uint64_t& device_bytes, std::uint64_t& maximum_source) {
  std::ifstream input(root / "dense-set.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-dense-residency-v1",
          "invalid dense model header");
  std::vector<er::cuda::DeepSeekDenseSpec> result;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 7U, "invalid dense model row");
    er::PayloadRecord record;
    record.extents = extents(root / relative(item[6]), source_root, 2U);
    record.stored_bytes = std::stoull(item[3]);
    record.device_bytes = std::stoull(item[4]);
    record.source_abi = er::kExpertSourceAbiDeepSeekFp8Block128V1;
    record.alignment = er::kExpertPackAlignment;
    record.payload_sha256 = digest(item[5]);
    const auto rows = static_cast<std::uint32_t>(std::stoul(item[1]));
    const auto columns = static_cast<std::uint32_t>(std::stoul(item[2]));
    result.push_back({item[0], std::move(record), rows, columns});
    source_bytes += std::stoull(item[3]);
    device_bytes += std::stoull(item[4]);
    maximum_source = std::max(maximum_source, std::stoull(item[3]));
  }
  require(input.eof() && result.size() == 236U,
          "model dense set is incomplete");
  return result;
}

std::vector<er::cuda::DeepSeekTypedSpec> typed_specs(
    const std::filesystem::path& root,
    const std::filesystem::path& source_root, std::uint64_t& source_bytes) {
  std::ifstream input(root / "typed-set.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-typed-residency-v1",
          "invalid typed model header");
  std::vector<er::cuda::DeepSeekTypedSpec> result;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 6U, "invalid typed model row");
    er::PayloadRecord record;
    record.extents = extents(root / relative(item[5]), source_root, 1U);
    record.stored_bytes = std::stoull(item[3]);
    record.payload_sha256 = digest(item[4]);
    result.push_back({item[0], std::move(record), dtype(item[1])});
    source_bytes += std::stoull(item[3]);
  }
  require(input.eof() && result.size() == 834U,
          "model typed set is incomplete");
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: expert-deepseek-model-residency "
                   "<dense-bundle> <typed-bundle> <checkpoint>\n";
      return 64;
    }
    const std::filesystem::path source = argv[3];
    std::uint64_t dense_source = 0U, dense_device = 0U, maximum_dense = 0U;
    std::uint64_t typed_source = 0U;
    const auto dense = dense_specs(argv[1], source, dense_source, dense_device,
                                   maximum_dense);
    const auto typed = typed_specs(argv[2], source, typed_source);
    constexpr std::uint64_t typed_staging = 64ULL * 1024U * 1024U;
    const auto staging = std::max(maximum_dense, typed_staging);
    const auto resident_bytes = dense_device + typed_source;
    std::size_t free_before = 0U, total = 0U;
    check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo before model");
    require(free_before >= resident_bytes + 512ULL * 1024U * 1024U,
            "insufficient VRAM headroom for transactional model state");

    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    er::ExtentGatherStorage storage(iocp);
    er::FixedBufferPool buffers(1U, staging, er::kExpertPackAlignment,
                                std::make_shared<er::CudaPinnedAllocator>());
    er::cuda::DeepSeekResidentModelState model;
    const auto started = std::chrono::steady_clock::now();
    const auto status = er::cuda::DeepSeekResidentModelState::load(
        storage, buffers, dense, typed, model);
    const auto stopped = std::chrono::steady_clock::now();
    require(status.ok(), std::string(status.message()));
    require(model.dense_size() == 236U && model.typed_size() == 834U &&
                model.bytes() == resident_bytes,
            "published model state has inconsistent ownership");
    er::cuda::DeepSeekAttentionBinding ratio_four, ratio_128;
    auto bind = model.bind_attention(2U, 4U, ratio_four);
    require(bind.ok(), std::string(bind.message()));
    bind = model.bind_attention(3U, 128U, ratio_128);
    require(bind.ok(), std::string(bind.message()));
    std::size_t free_resident = 0U;
    check(cudaMemGetInfo(&free_resident, &total),
          "cudaMemGetInfo resident model");
    const auto startup_ms = std::chrono::duration<double, std::milli>(
                                stopped - started).count();
    std::cout << "{\"ok\":true,\"dense_tensors\":" << model.dense_size()
              << ",\"typed_tensors\":" << model.typed_size()
              << ",\"source_bytes\":" << dense_source + typed_source
              << ",\"resident_bytes\":" << model.bytes()
              << ",\"staging_bytes\":" << staging
              << ",\"startup_ms\":" << startup_ms
              << ",\"ratio4_bound\":true,\"ratio128_bound\":true"
              << ",\"cuda_free_before\":" << free_before
              << ",\"cuda_free_resident\":" << free_resident << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-model-residency: " << error.what() << '\n';
    return 1;
  }
}
