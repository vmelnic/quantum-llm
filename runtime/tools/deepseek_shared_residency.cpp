#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/resident_expert_set.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

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

constexpr std::uint64_t kSourceBytes = 25'167'360U;
constexpr std::uint64_t kHotBytes = 25'198'592U;
constexpr std::uint32_t kLayers = 43U;
constexpr std::uint32_t kSharedExpert = 256U;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

std::uint8_t hex_nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::runtime_error("invalid SHA-256 hex digit");
}

er::Sha256Digest parse_digest(const std::string& value) {
  require(value.size() == 64U, "invalid SHA-256 length");
  er::Sha256Digest result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = static_cast<std::byte>(
        (hex_nibble(value[index * 2U]) << 4U) |
        hex_nibble(value[index * 2U + 1U]));
  }
  return result;
}

std::filesystem::path safe_relative(const std::string& text) {
  const std::filesystem::path path = text;
  require(!path.empty() && !path.is_absolute(),
          "manifest paths must be relative");
  for (const auto& component : path) {
    require(component != "..", "manifest path escapes its root");
  }
  return path;
}

std::vector<er::PayloadExtent> read_extents(
    const std::filesystem::path& descriptor,
    const std::filesystem::path& source_root) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid extent descriptor header");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    const auto first = line.find('\t');
    const auto second = line.find('\t', first + 1U);
    const auto third = line.find('\t', second + 1U);
    require(first != std::string::npos && second != std::string::npos &&
                third != std::string::npos &&
                line.find('\t', third + 1U) == std::string::npos,
            "invalid extent descriptor row");
    const auto destination = std::stoull(line.substr(0U, first));
    const auto bytes = std::stoull(line.substr(first + 1U, second - first - 1U));
    const auto source = std::stoull(line.substr(second + 1U, third - second - 1U));
    result.push_back({source_root / safe_relative(line.substr(third + 1U)),
                      source, destination, bytes});
  }
  require(input.eof() && result.size() == 6U,
          "shared expert must contain exactly six source extents");
  return result;
}

std::vector<er::ResidentExpertSpec> read_set(
    const std::filesystem::path& root,
    const std::filesystem::path& source_root) {
  std::ifstream input(root / "shared-set.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-shared-residency-v1",
          "invalid shared residency header");
  std::vector<er::ResidentExpertSpec> result;
  while (std::getline(input, line)) {
    const auto first = line.find('\t');
    const auto second = line.find('\t', first + 1U);
    const auto third = line.find('\t', second + 1U);
    require(first != std::string::npos && second != std::string::npos &&
                third != std::string::npos &&
                line.find('\t', third + 1U) == std::string::npos,
            "invalid shared residency row");
    const auto layer = static_cast<std::uint32_t>(
        std::stoul(line.substr(0U, first)));
    const auto bytes = std::stoull(line.substr(first + 1U, second - first - 1U));
    require(layer == result.size() && layer < kLayers && bytes == kSourceBytes,
            "shared residency layers or byte geometry are not canonical");
    er::PayloadRecord record;
    record.extents = read_extents(
        root / safe_relative(line.substr(third + 1U)), source_root);
    record.stored_bytes = bytes;
    record.decoded_bytes = 3ULL * 4096U * 2048U * sizeof(float);
    record.device_bytes = kHotBytes;
    record.source_abi = er::kExpertSourceAbiDeepSeekFp8Block128V1;
    record.header_bytes = 0U;
    record.alignment = er::kExpertPackAlignment;
    record.payload_sha256 = parse_digest(
        line.substr(second + 1U, third - second - 1U));
    result.push_back({{17U, layer, kSharedExpert,
                       er::kExpertQuantAbiDeepSeekSm86},
                      std::move(record)});
  }
  require(input.eof() && result.size() == kLayers,
          "shared residency set must contain exactly 43 layers");
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: expert-deepseek-shared-residency "
                   "<descriptor-root> <checkpoint-root>\n";
      return 64;
    }
    const auto specs = read_set(argv[1], argv[2]);
    constexpr std::uint64_t kTotalHotBytes = kHotBytes * kLayers;
    constexpr std::uint64_t kTotalSourceBytes = kSourceBytes * kLayers;
    std::size_t free_before = 0U;
    std::size_t total_device = 0U;
    check(cudaMemGetInfo(&free_before, &total_device), "cudaMemGetInfo before");
    require(free_before >= kTotalHotBytes + 512ULL * 1024U * 1024U,
            "insufficient VRAM headroom for resident shared set");

    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    auto storage = std::make_shared<er::ExtentGatherStorage>(iocp);
    auto uploader = std::make_shared<er::cuda::CudaExpertUploader>();
    auto buffers = std::make_shared<er::FixedBufferPool>(
        1U, kSourceBytes, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    auto directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        17U, er::kExpertQuantAbiDeepSeekSm86, kLayers, 257U, 8U);
    er::ExpertCacheConfig config;
    config.ram = {kSourceBytes * 2U, kSourceBytes * 2U, kSourceBytes};
    config.vram = {kTotalHotBytes, kTotalHotBytes, kTotalHotBytes};
    config.retain_host_copy = false;
    er::ExpertCache cache(config, storage, uploader, buffers, directory);

    er::ResidentExpertSet resident;
    const auto started = std::chrono::steady_clock::now();
    const auto status = er::ResidentExpertSet::load(cache, specs, resident);
    const auto stopped = std::chrono::steady_clock::now();
    require(status.ok(), std::string(status.message()));
    require(resident.size() == kLayers && resident.bytes() == kTotalHotBytes,
            "resident set published an incomplete byte count");

    std::uint32_t* selected = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&selected), sizeof(std::uint32_t)),
          "cudaMalloc shared selection");
    const std::uint32_t shared = kSharedExpert;
    check(cudaMemcpy(selected, &shared, sizeof(shared), cudaMemcpyHostToDevice),
          "copy shared selection");
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
      const auto plan =
          directory->pin_or_collect_misses(layer, selected, 1U, nullptr);
      require(plan.status.ok() && plan.missing_experts.empty() &&
                  plan.unique_experts == 1U,
              "resident shared expert is absent from CUDA directory");
      const auto release = directory->release_pins(nullptr);
      require(release.ok(), std::string(release.message()));
    }
    check(cudaFree(selected), "cudaFree shared selection");
    const auto telemetry = cache.telemetry();
    require(telemetry.load_started == kLayers &&
                telemetry.load_completed == kLayers &&
                telemetry.upload_started == kLayers &&
                telemetry.upload_completed == kLayers &&
                telemetry.read_bytes == kTotalSourceBytes &&
                telemetry.uploaded_bytes == kTotalHotBytes &&
                telemetry.vram_bytes == kTotalHotBytes,
            "shared residency telemetry violated the startup contract");
    std::size_t free_resident = 0U;
    check(cudaMemGetInfo(&free_resident, &total_device),
          "cudaMemGetInfo resident");

    resident.clear();
    const auto trimmed = cache.trim();
    require(trimmed == kTotalHotBytes && cache.telemetry().vram_bytes == 0U,
            "shared residency teardown did not release every logical slot");
    const auto elapsed = std::chrono::duration<double, std::milli>(
                             stopped - started)
                             .count();
    std::cout << "{\"ok\":true,\"layers\":" << kLayers
              << ",\"source_bytes\":" << kTotalSourceBytes
              << ",\"resident_bytes\":" << kTotalHotBytes
              << ",\"startup_ms\":" << elapsed
              << ",\"loads\":" << telemetry.load_started
              << ",\"uploads\":" << telemetry.upload_started
              << ",\"cuda_free_before\":" << free_before
              << ",\"cuda_free_resident\":" << free_resident
              << ",\"trimmed_bytes\":" << trimmed << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-shared-residency: " << error.what() << '\n';
    return 1;
  }
}
