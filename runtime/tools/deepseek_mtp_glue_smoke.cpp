#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/deepseek_mtp.hpp"
#include "expert/runtime/deepseek_artifacts.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
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

constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kStreams = 4U;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

std::vector<float> read_f32(const std::filesystem::path& path,
                            std::size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(input) &&
              static_cast<std::size_t>(input.tellg()) == count * sizeof(float),
          "invalid MTP oracle file: " + path.filename().string());
  input.seekg(0);
  std::vector<float> result(count);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(count * sizeof(float)));
  require(static_cast<bool>(input), "truncated MTP oracle file");
  return result;
}

struct ErrorStats final {
  double squared{};
  float maximum{};
  std::size_t count{};

  void add(const std::vector<float>& actual,
           const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "MTP comparison size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const auto error = std::abs(actual[index] - expected[index]);
      squared += static_cast<double>(error) * error;
      maximum = std::max(maximum, error);
    }
    count += actual.size();
  }
  [[nodiscard]] double rmse() const {
    return std::sqrt(squared / static_cast<double>(count));
  }
};

std::vector<float> copy_device(const float* source, std::size_t count,
                               const char* operation) {
  std::vector<float> result(count);
  check(cudaMemcpy(result.data(), source, count * sizeof(float),
                   cudaMemcpyDeviceToHost), operation);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: expert-deepseek-mtp-glue-smoke <oracle> "
                   "<mtp-set> <checkpoint>\n";
      return 64;
    }
    const std::filesystem::path oracle = argv[1];
    const std::filesystem::path mtp_root = argv[2];
    const std::filesystem::path checkpoint = argv[3];
    auto loaded = er::load_deepseek_tensor_artifacts(
        mtp_root / "dense", mtp_root / "typed-residency", checkpoint, 7U,
        19U);
    require(loaded.status.ok(), std::string(loaded.status.message()));
    auto shared = er::load_deepseek_shared_artifacts(
        mtp_root / "shared", checkpoint, 1U, 18U);
    require(shared.status.ok() && shared.shared.size() == 1U,
            std::string(shared.status.message()));
    const auto dense_device_bytes = loaded.artifacts.dense_device_bytes;
    const auto staging_bytes = std::max<std::uint64_t>(
        loaded.artifacts.maximum_source_record_bytes, 64ULL << 20U);
    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    er::ExtentGatherStorage storage(iocp);
    er::FixedBufferPool buffers(1U, staging_bytes, er::kExpertPackAlignment,
                                std::make_shared<er::CudaPinnedAllocator>());
    er::cuda::DeepSeekResidentTensorState resources;
    auto resource_status = er::cuda::DeepSeekResidentTensorState::load(
        storage, buffers, loaded.artifacts.dense, loaded.artifacts.typed,
        resources);
    require(resource_status.ok(), std::string(resource_status.message()));
    require(resources.dense_size() == 7U && resources.typed_size() == 19U,
            "incomplete MTP tensor namespace");
    er::cuda::DeepSeekMtpGlueBinding binding;
    resource_status = resources.bind_mtp_glue("mtp.0", binding);
    require(resource_status.ok(), std::string(resource_status.message()));
    er::cuda::DeepSeekAttentionBinding attention;
    resource_status = resources.bind_attention("mtp.0", 0U, 0U, attention);
    require(resource_status.ok(), std::string(resource_status.message()));
    er::cuda::DeepSeekFfnBinding ffn;
    resource_status = resources.bind_ffn(
        "mtp.0", 0U, er::cuda::DeepSeekRouterKind::learned, ffn);
    require(resource_status.ok() && !ffn.hash_router && ffn.router_bias,
            "invalid learned MTP FFN binding");

    const auto embedding = read_f32(oracle / "embedding.f32", kHidden);
    const auto previous = read_f32(oracle / "previous-streams.f32",
                                   kStreams * kHidden);
    float* device_embedding = nullptr;
    float* device_previous = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_embedding),
                     embedding.size() * sizeof(float)),
          "allocate MTP embedding");
    check(cudaMalloc(reinterpret_cast<void**>(&device_previous),
                     previous.size() * sizeof(float)),
          "allocate MTP previous streams");
    check(cudaMemcpy(device_embedding, embedding.data(),
                     embedding.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload MTP embedding");
    check(cudaMemcpy(device_previous, previous.data(),
                     previous.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload MTP previous streams");

    auto state = er::cuda::create_deepseek_mtp_glue_state();
    require(state.status.ok() && state.state,
            std::string(state.status.message()));
    auto status = er::cuda::deepseek_mtp_mix(
        binding, device_embedding, device_previous, *state.state, 1e-6F,
        nullptr);
    require(status.ok(), std::string(status.message()));
    status = er::cuda::deepseek_mtp_collapse(
        binding, state.state->mixed_streams(), *state.state, 1e-6F, nullptr);
    require(status.ok(), std::string(status.message()));
    check(cudaDeviceSynchronize(), "synchronize MTP glue");

    cudaEvent_t started{}, stopped{};
    check(cudaEventCreate(&started), "create MTP start event");
    check(cudaEventCreate(&stopped), "create MTP stop event");
    constexpr std::uint32_t kIterations = 100U;
    check(cudaEventRecord(started), "record MTP mix start");
    for (std::uint32_t iteration = 0U; iteration < kIterations; ++iteration) {
      status = er::cuda::deepseek_mtp_mix(
          binding, device_embedding, device_previous, *state.state, 1e-6F,
          nullptr);
      require(status.ok(), std::string(status.message()));
    }
    check(cudaEventRecord(stopped), "record MTP mix stop");
    check(cudaEventSynchronize(stopped), "synchronize MTP mix timing");
    float mix_ms = 0.0F;
    check(cudaEventElapsedTime(&mix_ms, started, stopped),
          "measure MTP mix");
    mix_ms /= static_cast<float>(kIterations);
    check(cudaEventRecord(started), "record MTP collapse start");
    for (std::uint32_t iteration = 0U; iteration < kIterations; ++iteration) {
      status = er::cuda::deepseek_mtp_collapse(
          binding, state.state->mixed_streams(), *state.state, 1e-6F, nullptr);
      require(status.ok(), std::string(status.message()));
    }
    check(cudaEventRecord(stopped), "record MTP collapse stop");
    check(cudaEventSynchronize(stopped),
          "synchronize MTP collapse timing");
    float collapse_ms = 0.0F;
    check(cudaEventElapsedTime(&collapse_ms, started, stopped),
          "measure MTP collapse");
    collapse_ms /= static_cast<float>(kIterations);
    check(cudaEventDestroy(started), "destroy MTP start event");
    check(cudaEventDestroy(stopped), "destroy MTP stop event");

    ErrorStats input_stats;
    input_stats.add(
        copy_device(state.state->normalized_token(), kHidden,
                    "copy MTP normalized token"),
        read_f32(oracle / "normalized-token.f32", kHidden));
    input_stats.add(
        copy_device(state.state->normalized_streams(), kStreams * kHidden,
                    "copy MTP normalized streams"),
        read_f32(oracle / "normalized-streams.f32", kStreams * kHidden));
    input_stats.add(
        copy_device(state.state->mixed_streams(), kStreams * kHidden,
                    "copy MTP mixed streams"),
        read_f32(oracle / "mixed-streams.f32", kStreams * kHidden));

    ErrorStats output_stats;
    const auto& head = state.state->head_state();
    output_stats.add(copy_device(head.head_gates(), kStreams,
                                 "copy MTP head gates"),
                     read_f32(oracle / "hc-head-gates.f32", kStreams));
    output_stats.add(copy_device(head.collapsed(), kHidden,
                                 "copy MTP collapsed output"),
                     read_f32(oracle / "collapsed.f32", kHidden));
    output_stats.add(copy_device(head.normalized(), kHidden,
                                 "copy MTP normalized output"),
                     read_f32(oracle / "normalized-output.f32", kHidden));
    require(input_stats.maximum < 2e-4F,
            "MTP input boundary exceeds oracle tolerance");
    require(output_stats.maximum < 2e-4F,
            "MTP output boundary exceeds oracle tolerance");

    check(cudaFree(device_embedding), "free MTP embedding");
    check(cudaFree(device_previous), "free MTP previous streams");
    std::cout << "{\"ok\":true,\"state_bytes\":" << state.state->bytes()
              << ",\"resident_tensor_bytes\":" << resources.bytes()
              << ",\"projection_device_bytes\":"
              << dense_device_bytes
              << ",\"mix_ms\":" << mix_ms
              << ",\"collapse_ms\":" << collapse_ms
              << ",\"input_rmse\":" << input_stats.rmse()
              << ",\"input_max_abs_error\":" << input_stats.maximum
              << ",\"output_rmse\":" << output_stats.rmse()
              << ",\"output_max_abs_error\":" << output_stats.maximum
              << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-mtp-glue-smoke: " << error.what() << '\n';
    return 1;
  }
}
