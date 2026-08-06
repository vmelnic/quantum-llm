#include "expert/runtime/cuda/deepseek_dense.hpp"
#include "expert/runtime/cuda/deepseek_mtp.hpp"

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
constexpr std::uint64_t kProjectionWeightBytes =
    static_cast<std::uint64_t>(kHidden) * kHidden;
constexpr std::uint64_t kProjectionScaleBytes =
    static_cast<std::uint64_t>(kHidden / 128U) * (kHidden / 128U);

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
  for (const auto& part : path)
    require(part != "..", "source shard escapes checkpoint root");
  return path;
}

std::vector<std::byte> read_extents(const std::filesystem::path& descriptor,
                                    const std::filesystem::path& source_root,
                                    std::size_t expected_extents) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid MTP extent descriptor");
  struct Extent final {
    std::uint64_t destination{};
    std::uint64_t bytes{};
    std::uint64_t source{};
    std::filesystem::path shard;
  };
  std::vector<Extent> extents;
  std::uint64_t total = 0U;
  while (std::getline(input, line)) {
    const auto a = line.find('\t');
    const auto b = line.find('\t', a + 1U);
    const auto c = line.find('\t', b + 1U);
    require(a != std::string::npos && b != std::string::npos &&
                c != std::string::npos &&
                line.find('\t', c + 1U) == std::string::npos,
            "invalid MTP extent row");
    Extent extent;
    extent.destination = std::stoull(line.substr(0U, a));
    extent.bytes = std::stoull(line.substr(a + 1U, b - a - 1U));
    extent.source = std::stoull(line.substr(b + 1U, c - b - 1U));
    extent.shard = source_root / safe_relative(line.substr(c + 1U));
    require(extent.destination == total, "MTP extents are not contiguous");
    total += extent.bytes;
    extents.push_back(std::move(extent));
  }
  require(input.eof() && extents.size() == expected_extents,
          "unexpected MTP extent count");
  std::vector<std::byte> result(static_cast<std::size_t>(total));
  for (const auto& extent : extents) {
    std::ifstream source(extent.shard, std::ios::binary);
    require(static_cast<bool>(source), "missing MTP source shard");
    source.seekg(static_cast<std::streamoff>(extent.source));
    source.read(reinterpret_cast<char*>(result.data() + extent.destination),
                static_cast<std::streamsize>(extent.bytes));
    require(static_cast<bool>(source), "truncated MTP source extent");
  }
  return result;
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
    if (argc != 3) {
      std::cerr << "usage: expert-deepseek-mtp-glue-smoke <oracle> "
                   "<checkpoint>\n";
      return 64;
    }
    const std::filesystem::path oracle = argv[1];
    const std::filesystem::path checkpoint = argv[2];
    auto typed = read_extents(oracle / "resources" / "typed" / "extents.tsv",
                              checkpoint, 6U);
    auto e_source = read_extents(
        oracle / "resources" / "e_proj" / "extents.tsv", checkpoint, 2U);
    auto h_source = read_extents(
        oracle / "resources" / "h_proj" / "extents.tsv", checkpoint, 2U);
    require(e_source.size() == kProjectionWeightBytes + kProjectionScaleBytes &&
                h_source.size() == e_source.size(),
            "invalid MTP projection payload geometry");

    auto e_admitted = er::cuda::admit_deepseek_dense_matrix(
        std::span<const std::byte>(e_source).first(kProjectionWeightBytes),
        std::span<const std::byte>(e_source).subspan(kProjectionWeightBytes),
        kHidden, kHidden);
    auto h_admitted = er::cuda::admit_deepseek_dense_matrix(
        std::span<const std::byte>(h_source).first(kProjectionWeightBytes),
        std::span<const std::byte>(h_source).subspan(kProjectionWeightBytes),
        kHidden, kHidden);
    require(e_admitted.status.ok() && e_admitted.matrix &&
                h_admitted.status.ok() && h_admitted.matrix,
            "MTP projection admission failed");

    constexpr std::size_t kEnormOffset = 0U;
    constexpr std::size_t kHnormOffset = kEnormOffset + kHidden * 2U;
    constexpr std::size_t kHeadFunctionOffset = kHnormOffset + kHidden * 2U;
    constexpr std::size_t kHeadBaseOffset =
        kHeadFunctionOffset + 4U * kStreams * kHidden * sizeof(float);
    constexpr std::size_t kHeadScaleOffset =
        kHeadBaseOffset + kStreams * sizeof(float);
    constexpr std::size_t kOutputNormOffset = kHeadScaleOffset + sizeof(float);
    constexpr std::size_t kTypedBytes = kOutputNormOffset + kHidden * 2U;
    require(typed.size() == kTypedBytes, "invalid MTP typed payload geometry");
    std::byte* device_typed = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_typed), typed.size()),
          "allocate MTP typed resources");
    check(cudaMemcpy(device_typed, typed.data(), typed.size(),
                     cudaMemcpyHostToDevice),
          "upload MTP typed resources");

    er::cuda::DeepSeekMtpGlueBinding binding;
    binding.e_projection = e_admitted.matrix->view();
    binding.h_projection = h_admitted.matrix->view();
    binding.embedding_norm = reinterpret_cast<const std::uint16_t*>(
        device_typed + kEnormOffset);
    binding.hidden_norm = reinterpret_cast<const std::uint16_t*>(
        device_typed + kHnormOffset);
    binding.head_function = reinterpret_cast<const float*>(
        device_typed + kHeadFunctionOffset);
    binding.head_base = reinterpret_cast<const float*>(
        device_typed + kHeadBaseOffset);
    binding.head_scale = reinterpret_cast<const float*>(
        device_typed + kHeadScaleOffset);
    binding.output_norm = reinterpret_cast<const std::uint16_t*>(
        device_typed + kOutputNormOffset);

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
    check(cudaFree(device_typed), "free MTP typed resources");
    std::cout << "{\"ok\":true,\"state_bytes\":" << state.state->bytes()
              << ",\"projection_device_bytes\":"
              << (e_admitted.matrix->bytes() + h_admitted.matrix->bytes())
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
