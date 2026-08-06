#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_attention.hpp"
#include "expert/runtime/cuda/deepseek_ffn.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/deepseek_mtp.hpp"
#include "expert/runtime/cuda/deepseek_mtp_request.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/deepseek_artifacts.hpp"
#include "expert/runtime/deepseek_catalog.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/resident_expert_set.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
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

constexpr std::uint32_t kPositions = 4U;
constexpr std::uint32_t kStreams = 4U;
constexpr std::uint32_t kHidden = 4096U;
constexpr std::size_t kStreamValues =
    static_cast<std::size_t>(kStreams) * kHidden;
constexpr std::uint64_t kMtpModelId = 18U;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

template <typename T>
std::vector<T> read_array(const std::filesystem::path& path,
                          std::size_t count) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(input) && input.tellg() >= 0 &&
              static_cast<std::size_t>(input.tellg()) == count * sizeof(T),
          "invalid MTP block oracle file: " + path.filename().string());
  input.seekg(0);
  std::vector<T> result(count);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  require(static_cast<bool>(input),
          "truncated MTP block oracle file: " + path.filename().string());
  return result;
}

struct ErrorStats final {
  double squared{};
  float maximum{};
  std::size_t count{};

  void add(const float* actual, const float* expected, std::size_t values) {
    for (std::size_t index = 0; index < values; ++index) {
      const auto error = std::abs(actual[index] - expected[index]);
      squared += static_cast<double>(error) * error;
      maximum = std::max(maximum, error);
    }
    count += values;
  }

  [[nodiscard]] double rmse() const {
    return count ? std::sqrt(squared / static_cast<double>(count)) : 0.0;
  }
};

float top_margin(const std::vector<float>& logits, std::uint32_t selected) {
  require(selected < logits.size(), "sampled token is outside vocabulary");
  float runner_up = -INFINITY;
  for (std::uint32_t token = 0U; token < logits.size(); ++token)
    if (token != selected) runner_up = std::max(runner_up, logits[token]);
  return logits[selected] - runner_up;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: expert-deepseek-mtp-block-smoke <oracle> "
                   "<target-bundle> <mtp-set> <mtp-pack> <checkpoint>\n";
      return 64;
    }
    const std::filesystem::path oracle = argv[1];
    const std::filesystem::path target_root = argv[2];
    const std::filesystem::path mtp_root = argv[3];
    const std::filesystem::path mtp_pack = argv[4];
    const std::filesystem::path checkpoint = argv[5];

    auto target_artifacts = er::load_deepseek_model_artifacts(
        target_root / "dense", target_root / "typed",
        target_root / "shared", checkpoint);
    require(target_artifacts.status.ok(),
            std::string(target_artifacts.status.message()));
    auto mtp_artifacts = er::load_deepseek_tensor_artifacts(
        mtp_root / "dense", mtp_root / "typed-residency", checkpoint,
        7U, 19U);
    require(mtp_artifacts.status.ok(),
            std::string(mtp_artifacts.status.message()));
    auto mtp_shared = er::load_deepseek_shared_artifacts(
        mtp_root / "shared", checkpoint, 1U, kMtpModelId);
    require(mtp_shared.status.ok(), std::string(mtp_shared.status.message()));
    er::DeepSeekExpertCatalog mtp_catalog;
    auto status = er::DeepSeekExpertCatalog::load_namespace(
        mtp_pack, checkpoint, 1U, mtp_catalog);
    require(status.ok(), std::string(status.message()));

    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    auto storage = std::make_shared<er::ExtentGatherStorage>(iocp);
    const auto staging_bytes = std::max<std::uint64_t>(
        64ULL << 20U,
        std::max(target_artifacts.artifacts.maximum_source_record_bytes,
                 mtp_artifacts.artifacts.maximum_source_record_bytes));
    auto buffers = std::make_shared<er::FixedBufferPool>(
        2U, staging_bytes, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    auto target = std::make_shared<er::cuda::DeepSeekResidentModelState>();
    status = er::cuda::DeepSeekResidentModelState::load(
        *storage, *buffers, target_artifacts.artifacts.dense,
        target_artifacts.artifacts.typed, *target);
    require(status.ok(), std::string(status.message()));
    auto mtp = std::make_shared<er::cuda::DeepSeekResidentTensorState>();
    status = er::cuda::DeepSeekResidentTensorState::load(
        *storage, *buffers, mtp_artifacts.artifacts.dense,
        mtp_artifacts.artifacts.typed, *mtp);
    require(status.ok(), std::string(status.message()));

    auto directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        kMtpModelId, er::kExpertQuantAbiDeepSeekSm86, 1U, 257U, 16U);
    auto uploader = std::make_shared<er::cuda::CudaExpertUploader>(
        er::cuda::CudaExpertUploaderOptions{0U, true, 0U, true});
    er::ExpertCacheConfig cache_config;
    cache_config.ram = {512ULL << 20U, 512ULL << 20U, 448ULL << 20U};
    cache_config.vram = {512ULL << 20U, 512ULL << 20U, 448ULL << 20U};
    cache_config.retain_host_copy = false;
    cache_config.trusted_immutable_source = true;
    er::ExpertCache cache(cache_config, storage, uploader, buffers, directory);
    er::ResidentExpertSet shared;
    status = er::ResidentExpertSet::load(cache, mtp_shared.shared, shared);
    require(status.ok(), std::string(status.message()));

    const auto previous = read_array<float>(
        oracle / "previous-streams.f32", kPositions * kStreamValues);
    const auto tokens = read_array<std::uint32_t>(
        oracle / "tokens.u32", kPositions);
    const auto expected_mixed = read_array<float>(
        oracle / "mixed-streams.f32", kPositions * kStreamValues);
    const auto expected_attention = read_array<float>(
        oracle / "attention-output.f32", kPositions * kStreamValues);
    const auto cosines = read_array<float>(
        oracle / "cosine.f32", kPositions * 32U);
    const auto sines = read_array<float>(
        oracle / "sine.f32", kPositions * 32U);
    const auto expected_indices = read_array<std::uint32_t>(
        oracle / "router-indices.i32", kPositions * 6U);
    const auto expected_scores = read_array<float>(
        oracle / "router-scores.f32", kPositions * 6U);
    const auto expected_block = read_array<float>(
        oracle / "block-output.f32", kStreamValues);
    const auto expected_normalized = read_array<float>(
        oracle / "normalized-output.f32", kHidden);
    const auto expected_logits = read_array<float>(
        oracle / "logits.f32", er::cuda::kDeepSeekVocab);
    const auto expected_sample = read_array<std::uint32_t>(
        oracle / "sampled.u32", 1U).front();

    float* device_previous = nullptr;
    float *device_attention = nullptr, *device_cosines = nullptr;
    float* device_sines = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_previous),
                     previous.size() * sizeof(float)),
          "allocate MTP previous streams");
    check(cudaMalloc(reinterpret_cast<void**>(&device_attention),
                     expected_attention.size() * sizeof(float)),
          "allocate MTP attention outputs");
    check(cudaMalloc(reinterpret_cast<void**>(&device_cosines),
                     cosines.size() * sizeof(float)),
          "allocate MTP cosines");
    check(cudaMalloc(reinterpret_cast<void**>(&device_sines),
                     sines.size() * sizeof(float)),
          "allocate MTP sines");
    check(cudaMemcpy(device_previous, previous.data(),
                     previous.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload MTP previous streams");
    check(cudaMemcpy(device_cosines, cosines.data(),
                     cosines.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload MTP cosines");
    check(cudaMemcpy(device_sines, sines.data(),
                     sines.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload MTP sines");

    const auto request_size = er::cuda::deepseek_mtp_request_state_size(4U);
    require(request_size.status.ok(),
            std::string(request_size.status.message()));
    auto request = er::cuda::create_deepseek_mtp_request_state(
        target, mtp, {4U, request_size.total_bytes});
    require(request.status.ok() && request.state,
            std::string(request.status.message()));

    std::vector<float> actual_mixed(expected_mixed.size());
    std::vector<std::uint32_t> actual_indices(expected_indices.size());
    std::vector<float> actual_scores(expected_scores.size());
    for (std::uint32_t position = 0U; position < kPositions; ++position) {
      status = request.state->prepare(
          tokens[position], device_previous + position * kStreamValues,
          position, device_cosines + position * 32U,
          device_sines + position * 32U, nullptr);
      require(status.ok(), std::string(status.message()));
      check(cudaMemcpy(actual_mixed.data() + position * kStreamValues,
                       request.state->mixed_streams(),
                       kStreamValues * sizeof(float), cudaMemcpyDeviceToHost),
            "copy MTP mixed streams");
      check(cudaMemcpy(device_attention + position * kStreamValues,
                       request.state->attention_streams(),
                       kStreamValues * sizeof(float), cudaMemcpyDeviceToDevice),
            "retain MTP attention output");
      check(cudaMemcpy(actual_indices.data() + position * 6U,
                       request.state->expert_indices(),
                       6U * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
            "copy MTP router indices");
      check(cudaMemcpy(actual_scores.data() + position * 6U,
                       request.state->routing_weights(),
                       6U * sizeof(float), cudaMemcpyDeviceToHost),
            "copy MTP router scores");
      if (position + 1U != kPositions) {
        status = request.state->abandon_draft();
        require(status.ok(), std::string(status.message()));
      }
    }

    std::vector<er::ExpertLease> routed_leases;
    routed_leases.reserve(6U);
    for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
      const auto expert = actual_indices[3U * 6U + slot];
      const auto* record = mtp_catalog.find(0U, expert);
      require(record != nullptr, "MTP routed expert is absent from catalog");
      auto acquired = cache.acquire(
          {kMtpModelId, 0U, expert, er::kExpertQuantAbiDeepSeekSm86},
          *record).get();
      require(acquired.status.ok() && acquired.lease,
              std::string(acquired.status.message()));
      routed_leases.push_back(std::move(acquired.lease));
    }
    const auto pin = directory->pin_or_collect_misses(
        0U, request.state->expert_indices(),
        request.state->selection_count(), nullptr);
    require(pin.status.ok() && pin.missing_experts.empty() && pin.pin_id != 0U,
            "MTP FFN dependencies are not resident");
    status = request.state->complete(directory->device_entries(), 257U,
                                     nullptr);
    require(status.ok(), std::string(status.message()));
    check(cudaDeviceSynchronize(), "synchronize complete MTP block");

    std::vector<float> actual_attention(expected_attention.size());
    std::vector<float> actual_block(kStreamValues);
    std::vector<float> actual_normalized(kHidden);
    std::vector<float> actual_logits(er::cuda::kDeepSeekVocab);
    std::uint32_t actual_sample{};
    check(cudaMemcpy(actual_attention.data(), device_attention,
                     actual_attention.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy MTP attention outputs");
    check(cudaMemcpy(actual_block.data(), request.state->block_streams(),
                     actual_block.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy MTP block output");
    check(cudaMemcpy(actual_normalized.data(),
                     request.state->normalized_output(),
                     actual_normalized.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy MTP normalized output");
    check(cudaMemcpy(actual_logits.data(),
                     request.state->logits(),
                     actual_logits.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy MTP logits");
    check(cudaMemcpy(&actual_sample,
                     request.state->draft_token(),
                     sizeof(actual_sample), cudaMemcpyDeviceToHost),
          "copy MTP sampled token");

    ErrorStats mixed_stats, attention_stats, score_stats, block_stats;
    ErrorStats normalized_stats, logits_stats;
    mixed_stats.add(actual_mixed.data(), expected_mixed.data(),
                    expected_mixed.size());
    attention_stats.add(actual_attention.data(), expected_attention.data(),
                        expected_attention.size());
    score_stats.add(actual_scores.data(), expected_scores.data(),
                    expected_scores.size());
    block_stats.add(actual_block.data(), expected_block.data(),
                    expected_block.size());
    normalized_stats.add(actual_normalized.data(), expected_normalized.data(),
                         expected_normalized.size());
    logits_stats.add(actual_logits.data(), expected_logits.data(),
                     expected_logits.size());
    const auto actual_margin = top_margin(actual_logits, actual_sample);
    const auto expected_margin = top_margin(expected_logits, expected_sample);
    std::cerr << "MTP oracle diagnostics: mixed_rmse=" << mixed_stats.rmse()
              << " mixed_max=" << mixed_stats.maximum
              << " attention_rmse=" << attention_stats.rmse()
              << " attention_max=" << attention_stats.maximum
              << " router_rmse=" << score_stats.rmse()
              << " router_max=" << score_stats.maximum
              << " block_rmse=" << block_stats.rmse()
              << " block_max=" << block_stats.maximum
              << " normalized_rmse=" << normalized_stats.rmse()
              << " normalized_max=" << normalized_stats.maximum
              << " logits_rmse=" << logits_stats.rmse()
              << " logits_max=" << logits_stats.maximum
              << " sample=" << actual_sample
              << " expected_sample=" << expected_sample
              << " actual_margin=" << actual_margin
              << " expected_margin=" << expected_margin << '\n';
    require(actual_indices == expected_indices,
            "MTP learned router selected different experts");
    require(mixed_stats.maximum < 1e-5F,
            "MTP mix exceeds oracle tolerance");
    require(attention_stats.maximum <= 1.0F / 512.0F,
            "MTP attention exceeds oracle tolerance");
    require(score_stats.maximum <= 1.0F / 1024.0F,
            "MTP router weights exceed oracle tolerance");
    require(block_stats.rmse() <= 1.0 / 128.0 &&
                block_stats.maximum <= 1.0F / 32.0F,
            "MTP block output exceeds oracle tolerance");
    require(normalized_stats.maximum <= 1.0F / 32.0F,
            "MTP normalized output exceeds oracle tolerance");
    require(logits_stats.rmse() <= 1.0 / 16.0,
            "MTP vocabulary logits exceed oracle tolerance");
    require(actual_sample == expected_sample,
            "MTP vocabulary head selected a different draft token");
    require(std::min(actual_margin, expected_margin) >
                4.0F * static_cast<float>(logits_stats.rmse()),
            "MTP draft margin is not robust to observed logit error");

    const auto release = directory->release_pins(pin.pin_id, nullptr);
    require(release.ok(), std::string(release.message()));
    std::cout << "{\"ok\":true,\"draft_token\":" << actual_sample
              << ",\"mixed_rmse\":" << mixed_stats.rmse()
              << ",\"attention_rmse\":" << attention_stats.rmse()
              << ",\"router_rmse\":" << score_stats.rmse()
              << ",\"block_rmse\":" << block_stats.rmse()
              << ",\"block_max_abs_error\":" << block_stats.maximum
              << ",\"normalized_rmse\":" << normalized_stats.rmse()
              << ",\"logits_rmse\":" << logits_stats.rmse()
              << ",\"logits_max_abs_error\":" << logits_stats.maximum
              << ",\"actual_draft_margin\":" << actual_margin
              << ",\"expected_draft_margin\":" << expected_margin
              << ",\"resident_mtp_tensor_bytes\":" << mtp->bytes()
              << ",\"resident_mtp_shared_bytes\":" << shared.bytes()
              << ",\"mtp_request_bytes\":" << request.state->bytes()
              << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-mtp-block-smoke: " << error.what() << '\n';
    return 1;
  }
}
