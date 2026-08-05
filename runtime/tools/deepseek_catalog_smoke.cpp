#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/deepseek_catalog.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <filesystem>
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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: expert-deepseek-catalog-smoke "
                   "<catalog> <checkpoint>\n";
      return 64;
    }
    er::DeepSeekExpertCatalog catalog;
    const auto status = er::DeepSeekExpertCatalog::load(
        argv[1], argv[2], catalog);
    require(status.ok(), std::string(status.message()));
    require(catalog.size() == 43U * 256U &&
                catalog.source_bytes() == 147'169'738'752ULL,
            "DeepSeek routed catalog has the wrong aggregate geometry");

    constexpr std::uint64_t model_id = 17U;
    const std::array<er::ExpertKey, 2> keys{{
        {model_id, 0U, 0U, er::kExpertQuantAbiDeepSeekSm86},
        {model_id, 42U, 255U, er::kExpertQuantAbiDeepSeekSm86},
    }};
    auto io = std::make_shared<er::WindowsIocpStorage>(2U);
    auto storage = std::make_shared<er::ExtentGatherStorage>(io);
    auto uploader = std::make_shared<er::cuda::CudaExpertUploader>();
    auto buffers = std::make_shared<er::FixedBufferPool>(
        1U, 13'369'344U, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    auto directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        model_id, er::kExpertQuantAbiDeepSeekSm86, 43U, 257U, 2U);
    er::ExpertCacheConfig config;
    config.ram = {2ULL * 13'369'344U, 2ULL * 13'369'344U, 13'369'344U};
    config.vram = {2ULL * 25'198'592U, 2ULL * 25'198'592U, 25'198'592U};
    config.retain_host_copy = false;
    er::ExpertCache cache(config, storage, uploader, buffers, directory);
    std::vector<er::ExpertLease> leases;
    for (const auto& key : keys) {
      const auto* record = catalog.find(key.layer, key.expert);
      require(record != nullptr, "DeepSeek catalog lookup failed");
      auto acquired = cache.acquire(key, *record).get();
      require(acquired.status.ok() && acquired.lease,
              std::string(acquired.status.message()));
      leases.push_back(std::move(acquired.lease));
    }

    std::uint32_t* selected = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&selected),
                     2U * sizeof(std::uint32_t)),
          "allocate catalog selections");
    for (const auto& key : keys) {
      check(cudaMemcpy(selected, &key.expert, sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice),
            "copy catalog selection");
      const auto plan = directory->pin_or_collect_misses(
          key.layer, selected, 1U, nullptr);
      require(plan.status.ok() && plan.missing_experts.empty() &&
                  plan.pin_id != 0U,
              "catalog acquisition did not publish the expert");
      const auto release = directory->release_pins(plan.pin_id, nullptr);
      require(release.ok(), std::string(release.message()));
    }
    check(cudaFree(selected), "release catalog selections");
    const auto telemetry = cache.telemetry();
    std::cout << "{\"ok\":true,\"experts\":" << catalog.size()
              << ",\"source_bytes\":" << catalog.source_bytes()
              << ",\"sample_loads\":" << telemetry.load_completed
              << ",\"sample_read_bytes\":" << telemetry.read_bytes
              << ",\"sample_uploaded_bytes\":"
              << telemetry.uploaded_bytes << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-catalog-smoke: " << error.what() << '\n';
    return 1;
  }
}
