#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_attention.hpp"
#include "expert/runtime/cuda/deepseek_decode.hpp"
#include "expert/runtime/cuda/deepseek_scheduler.hpp"
#include "expert/runtime/cuda/deepseek_ffn.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/deepseek_request.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cpu/deepseek_packed_executor.hpp"
#include "expert/runtime/deepseek_artifacts.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/deepseek_catalog.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/placement_profile.hpp"
#include "expert/runtime/resident_expert_set.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
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
std::vector<er::ResidentExpertSpec> ffn_specs(
    const std::filesystem::path& oracle_root,
    const std::filesystem::path& source_root, std::uint64_t& source_bytes,
    std::uint32_t& layer) {
  std::ifstream input(oracle_root / "ffn" / "ffn-set.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-ffn-route-set-v2",
          "invalid FFN route-set header");
  std::vector<er::ResidentExpertSpec> result;
  layer = 43U;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 6U, "invalid FFN route-set row");
    const auto item_layer = static_cast<std::uint32_t>(std::stoul(item[0]));
    require(item_layer < 43U && (layer == 43U || layer == item_layer),
            "FFN route set spans incompatible layers");
    layer = item_layer;
    const auto expert = static_cast<std::uint32_t>(std::stoul(item[1]));
    const bool shared = item[2] == "shared";
    require((shared && expert == 256U) ||
                (!shared && item[2] == "routed" && expert < 256U),
            "invalid FFN route-set expert kind");
    const auto bytes = std::stoull(item[3]);
    require(bytes == (shared ? 25'167'360ULL : 13'369'344ULL),
            "invalid FFN source byte geometry");
    er::PayloadRecord record;
    record.extents = extents(
        oracle_root / relative(item[5]), source_root, 6U);
    record.stored_bytes = bytes;
    record.decoded_bytes = 3ULL * 4096U * 2048U * sizeof(float);
    record.device_bytes = 25'198'592ULL;
    record.source_abi = shared
        ? er::kExpertSourceAbiDeepSeekFp8Block128V1
        : er::kExpertSourceAbiDeepSeekCompactV1;
    record.alignment = er::kExpertPackAlignment;
    record.header_bytes = 0U;
    record.payload_sha256 = digest(item[4]);
    result.push_back({{17U, layer, expert, er::kExpertQuantAbiDeepSeekSm86},
                      std::move(record)});
    source_bytes += bytes;
  }
  require(input.eof() && result.size() == 7U &&
              result.back().key.expert == 256U,
          "FFN route set must contain six routed and one shared expert");
  return result;
}

std::vector<float> floats(const std::filesystem::path& path,
                          std::size_t expected) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(input) &&
              static_cast<std::size_t>(input.tellg()) == expected * sizeof(float),
          "invalid attention oracle file");
  input.seekg(0);
  std::vector<float> result(expected);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(expected * sizeof(float)));
  require(static_cast<bool>(input), "truncated attention oracle file");
  return result;
}

std::vector<std::uint32_t> integers(const std::filesystem::path& path,
                                    std::size_t expected) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  require(static_cast<bool>(input) &&
              static_cast<std::size_t>(input.tellg()) ==
                  expected * sizeof(std::uint32_t),
          "invalid attention oracle integer file");
  input.seekg(0);
  std::vector<std::uint32_t> result(expected);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(expected * sizeof(std::uint32_t)));
  require(static_cast<bool>(input), "truncated attention oracle integer file");
  return result;
}

std::uint32_t compression_ratio(std::uint32_t layer) {
  if (layer == 0U || layer == 1U || layer == 42U) return 0U;
  return layer % 2U == 0U ? 4U : 128U;
}

std::vector<std::uint32_t> prompt_tokens(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  std::vector<std::uint32_t> result;
  std::uint64_t token = 0U;
  while (input >> token) {
    require(token < 129280U, "prompt token is outside vocabulary");
    result.push_back(static_cast<std::uint32_t>(token));
  }
  require(input.eof() && !result.empty(), "prompt token file is invalid");
  return result;
}

std::array<float, 32U> rope_values(std::uint32_t position, bool sine,
                                   bool compressed) {
  std::array<float, 32U> result{};
  constexpr double dimension = 64.0;
  constexpr double factor = 16.0;
  constexpr double original_context = 65536.0;
  constexpr double beta_fast = 32.0;
  constexpr double beta_slow = 1.0;
  const double base = compressed ? 160000.0 : 10000.0;
  const auto correction = [&](double rotations) {
    return dimension * std::log(original_context /
                                (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(base));
  };
  const double low = std::clamp(std::floor(correction(beta_fast)), 0.0, 31.0);
  const double high = std::clamp(std::ceil(correction(beta_slow)), 0.0, 31.0);
  for (std::size_t index = 0U; index < result.size(); ++index) {
    double frequency = std::pow(base, -(2.0 * index) / dimension);
    if (compressed) {
      const double ramp = std::clamp(
          (static_cast<double>(index) - low) / std::max(high - low, 1e-3),
          0.0, 1.0);
      const double smooth = 1.0 - ramp;
      frequency = frequency / factor * (1.0 - smooth) + frequency * smooth;
    }
    const double angle = static_cast<double>(position) * frequency;
    result[index] = static_cast<float>(sine ? std::sin(angle) : std::cos(angle));
  }
  return result;
}

er::cuda::DeepSeekDecodeRope upload_rope(std::uint32_t position,
                                         float* device) {
  std::array<std::array<float, 32U>, 8U> values{};
  values[0] = rope_values(position, false, false);
  values[1] = rope_values(position, true, false);
  values[2] = rope_values(position, false, true);
  values[3] = rope_values(position, true, true);
  const auto ratio_four_start = position + 1U >= 4U
                                    ? position + 1U - 4U
                                    : 0U;
  const auto ratio_128_start = position + 1U >= 128U
                                  ? position + 1U - 128U
                                  : 0U;
  values[4] = rope_values(ratio_four_start, false, true);
  values[5] = rope_values(ratio_four_start, true, true);
  values[6] = rope_values(ratio_128_start, false, true);
  values[7] = rope_values(ratio_128_start, true, true);
  check(cudaMemcpy(device, values.data(), sizeof(values),
                   cudaMemcpyHostToDevice), "upload DeepSeek RoPE");
  return {device + 0U * 32U, device + 1U * 32U,
          device + 2U * 32U, device + 3U * 32U,
          device + 4U * 32U, device + 5U * 32U,
          device + 6U * 32U, device + 7U * 32U};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 8 || argc > 12) {
      std::cerr << "usage: expert-deepseek-model-residency "
                   "<dense-bundle> <typed-bundle> <checkpoint> "
                   "<attention-oracle> <routed-catalog> <io-oracle> "
                   "<shared-set> [prompt-token-file] [max-new-tokens] "
                   "[host-cache-gib] [compact-vram-cache-gib]\n";
      return 64;
    }
    const std::filesystem::path source = argv[3];
    const auto prompt = argc >= 9
                            ? std::optional(prompt_tokens(argv[8]))
                            : std::nullopt;
    const auto max_new_tokens = argc >= 10
                                    ? static_cast<std::uint32_t>(
                                          std::stoul(argv[9]))
                                    : 1U;
    require(max_new_tokens >= 1U && max_new_tokens <= 16U,
            "max-new-tokens must be in [1,16]");
    const auto host_cache_gib =
        argc >= 11 ? static_cast<std::uint64_t>(std::stoull(argv[10])) : 0U;
    require(host_cache_gib <= 48U, "host-cache-gib must be in [0,48]");
    const auto compact_cache_gib =
        argc == 12 ? static_cast<std::uint64_t>(std::stoull(argv[11])) : 0U;
    require(compact_cache_gib <= 10U,
            "compact-vram-cache-gib must be in [0,10]");
    static_cast<void>(compact_cache_gib);  // direct FP4 is the compact tier
    er::DeepSeekExpertCatalog routed_catalog;
    const auto catalog_status = er::DeepSeekExpertCatalog::load(
        argv[5], source, routed_catalog);
    require(catalog_status.ok(), std::string(catalog_status.message()));
    auto loaded_artifacts = er::load_deepseek_model_artifacts(
        argv[1], argv[2], argv[7], source);
    require(loaded_artifacts.status.ok(),
            std::string(loaded_artifacts.status.message()));
    auto artifacts = std::move(loaded_artifacts.artifacts);
    const auto dense_source = artifacts.dense_source_bytes;
    const auto dense_device = artifacts.dense_device_bytes;
    const auto maximum_dense = artifacts.maximum_source_record_bytes;
    const auto typed_source = artifacts.typed_source_bytes;
    auto dense = std::move(artifacts.dense);
    auto typed = std::move(artifacts.typed);
    auto all_shared = std::move(artifacts.shared);
    std::uint64_t ffn_source = 0U;
    std::uint32_t oracle_layer = 43U;
    const auto ffn = ffn_specs(argv[4], source, ffn_source, oracle_layer);
    const auto oracle_ratio = compression_ratio(oracle_layer);
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
    auto model = std::make_shared<er::cuda::DeepSeekResidentModelState>();
    const auto started = std::chrono::steady_clock::now();
    const auto status = er::cuda::DeepSeekResidentModelState::load(
        storage, buffers, dense, typed, *model);
    const auto stopped = std::chrono::steady_clock::now();
    require(status.ok(), std::string(status.message()));
    require(model->dense_size() == 236U && model->typed_size() == 834U &&
                model->bytes() == resident_bytes,
            "published model state has inconsistent ownership");
    const auto request_size = er::cuda::deepseek_request_state_size(4096U);
    require(request_size.status.ok() && request_size.total_bytes > 1U,
            std::string(request_size.status.message()));
    const auto rejected_request = er::cuda::create_deepseek_request_state(
        model, {4096U, request_size.total_bytes - 1U});
    require(!rejected_request.status.ok() && !rejected_request.state &&
                rejected_request.status.code() == er::ErrorCode::backpressure,
            "request state did not reject an insufficient preflight budget");
    auto request = er::cuda::create_deepseek_request_state(
        model, {4096U, request_size.total_bytes});
    require(request.status.ok() && request.state &&
                request.state->bytes() == request_size.total_bytes,
            std::string(request.status.message()));
    const auto io_input = integers(
        std::filesystem::path(argv[6]) / "input.u32", 1U);
    const auto expected_io_streams = floats(
        std::filesystem::path(argv[6]) / "streams.f32", 4U * 4096U);
    const auto expected_logits = floats(
        std::filesystem::path(argv[6]) / "logits.f32", 129280U);
    const auto expected_sampled = integers(
        std::filesystem::path(argv[6]) / "sampled.u32", 1U);
    auto io_status = request.state->embed(io_input[0]);
    require(io_status.ok(), std::string(io_status.message()));
    check(cudaDeviceSynchronize(), "synchronize DeepSeek embedding");
    std::vector<float> actual_io_streams(4U * 4096U);
    check(cudaMemcpy(actual_io_streams.data(), request.state->current_streams(),
                     actual_io_streams.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy DeepSeek embedding");
    float embedding_maximum = 0.0F;
    for (std::size_t index = 0U; index < actual_io_streams.size(); ++index)
      embedding_maximum = std::max(
          embedding_maximum,
          std::abs(actual_io_streams[index] - expected_io_streams[index]));
    require(embedding_maximum == 0.0F,
            "DeepSeek embedding differs from the BF16 source row");
    io_status = request.state->project_logits();
    require(io_status.ok(), std::string(io_status.message()));
    check(cudaDeviceSynchronize(), "synchronize DeepSeek output head");
    std::vector<float> actual_logits(129280U);
    std::uint32_t actual_sampled = 0U;
    check(cudaMemcpy(actual_logits.data(), request.state->logits(),
                     actual_logits.size() * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy DeepSeek logits");
    check(cudaMemcpy(&actual_sampled, request.state->sampled_token(),
                     sizeof(actual_sampled), cudaMemcpyDeviceToHost),
          "copy DeepSeek sampled token");
    double logits_squared = 0.0;
    float logits_maximum = 0.0F;
    for (std::size_t index = 0U; index < actual_logits.size(); ++index) {
      const auto error = std::abs(actual_logits[index] - expected_logits[index]);
      logits_maximum = std::max(logits_maximum, error);
      logits_squared += static_cast<double>(error) * error;
    }
    require(actual_sampled == expected_sampled[0] && logits_maximum < 0.1F,
            "DeepSeek output head differs from the independent oracle");
    std::uint32_t ratio_zero_layers = 0U, ratio_four_layers = 0U;
    std::uint32_t ratio_128_layers = 0U;
    for (std::uint32_t layer = 0U;
         layer < er::cuda::DeepSeekRequestState::layer_count(); ++layer) {
      const auto view = request.state->layer(layer);
      require(view.attention_weights && view.attention_state &&
                  view.ffn_weights && view.ffn_state &&
                  view.attention_weights->layer == layer &&
                  view.ffn_weights->layer == layer &&
                  view.ffn_state->layer() == layer &&
                  view.attention_state->compress_ratio() == view.compress_ratio,
              "incomplete 43-layer request state publication");
      if (view.compress_ratio == 0U) ++ratio_zero_layers;
      if (view.compress_ratio == 4U) ++ratio_four_layers;
      if (view.compress_ratio == 128U) ++ratio_128_layers;
    }
    require(ratio_zero_layers == 3U && ratio_four_layers == 20U &&
                ratio_128_layers == 20U,
            "request state has the wrong compression schedule");
    auto expert_storage = std::make_shared<er::ExtentGatherStorage>(iocp);
    auto expert_uploader = std::make_shared<er::cuda::CudaExpertUploader>(
        er::cuda::CudaExpertUploaderOptions{0U, false, 0U, true});
    auto expert_buffers = std::make_shared<er::FixedBufferPool>(
        1U, 25'167'360U, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    auto expert_directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        17U, er::kExpertQuantAbiDeepSeekSm86, 43U, 257U, 8U);
    er::ExpertCacheConfig expert_config;
    expert_config.ram = {128ULL << 20U, 128ULL << 20U, 64ULL << 20U};
    expert_config.vram = {7ULL * 25'198'592U, 7ULL * 25'198'592U,
                          25'198'592U};
    expert_config.retain_host_copy = true;
    er::ExpertCache expert_cache(expert_config, expert_storage,
                                 expert_uploader, expert_buffers,
                                 expert_directory);
    er::ResidentExpertSet ffn_resident;
    const auto ffn_load_started = std::chrono::steady_clock::now();
    const auto ffn_load_status = er::ResidentExpertSet::load(
        expert_cache, ffn, ffn_resident);
    const auto ffn_load_stopped = std::chrono::steady_clock::now();
    require(ffn_load_status.ok() && ffn_resident.size() == 7U &&
                ffn_resident.bytes() ==
                    6ULL * 13'369'344U + 25'198'592U,
            std::string(ffn_load_status.message()));
    er::cuda::DeepSeekAttentionBinding sliding_window, ratio_four, ratio_128;
    auto bind = model->bind_attention(0U, 0U, sliding_window);
    require(bind.ok(), std::string(bind.message()));
    bind = model->bind_attention(2U, 4U, ratio_four);
    require(bind.ok(), std::string(bind.message()));
    bind = model->bind_attention(3U, 128U, ratio_128);
    require(bind.ok(), std::string(bind.message()));
    er::cuda::DeepSeekFfnBinding hash_ffn, learned_ffn;
    bind = model->bind_ffn(2U, hash_ffn);
    require(bind.ok() && hash_ffn.hash_router && hash_ffn.token_experts &&
                !hash_ffn.router_bias,
            "invalid hash FFN binding");
    bind = model->bind_ffn(3U, learned_ffn);
    require(bind.ok() && !learned_ffn.hash_router && learned_ffn.router_bias &&
                !learned_ffn.token_experts,
            "invalid learned FFN binding");
    er::cuda::DeepSeekAttentionBinding oracle_attention;
    bind = model->bind_attention(oracle_layer, oracle_ratio, oracle_attention);
    require(bind.ok(), std::string(bind.message()));
    er::cuda::DeepSeekFfnBinding oracle_ffn;
    bind = model->bind_ffn(oracle_layer, oracle_ffn);
    require(bind.ok() && oracle_ffn.hash_router,
            "attention oracle requires a hash-routed FFN layer");
    constexpr std::size_t token_stream_values = 4U * 4096U;
    constexpr std::size_t decode_tokens = 4U;
    constexpr std::size_t stream_values = decode_tokens * token_stream_values;
    const auto host_streams = floats(std::filesystem::path(argv[4]) / "streams.f32",
                                     stream_values);
    const auto expected_output = floats(
        std::filesystem::path(argv[4]) / "output.f32", stream_values);
    const auto host_cosine = floats(
        std::filesystem::path(argv[4]) / "cosine.f32", decode_tokens * 32U);
    const auto host_sine = floats(
        std::filesystem::path(argv[4]) / "sine.f32", decode_tokens * 32U);
    constexpr std::size_t router_values = decode_tokens * 6U;
    const auto expected_router_scores = floats(
        std::filesystem::path(argv[4]) / "router-scores.f32", router_values);
    const auto expected_router_indices = integers(
        std::filesystem::path(argv[4]) / "router-indices.i32", router_values);
    const auto expected_block_output = floats(
        std::filesystem::path(argv[4]) / "block-output.f32",
        token_stream_values);
    float *device_streams = nullptr, *device_output = nullptr;
    float *device_cosine = nullptr, *device_sine = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_streams),
                     stream_values * sizeof(float)),
          "allocate attention streams");
    check(cudaMalloc(reinterpret_cast<void**>(&device_output),
                     stream_values * sizeof(float)),
          "allocate attention output");
    check(cudaMalloc(reinterpret_cast<void**>(&device_cosine),
                     decode_tokens * 32U * sizeof(float)),
          "allocate attention cosine");
    check(cudaMalloc(reinterpret_cast<void**>(&device_sine),
                     decode_tokens * 32U * sizeof(float)),
          "allocate attention sine");
    check(cudaMemcpy(device_streams, host_streams.data(),
                     stream_values * sizeof(float), cudaMemcpyHostToDevice),
          "copy attention streams");
    check(cudaMemcpy(device_cosine, host_cosine.data(),
                     decode_tokens * 32U * sizeof(float), cudaMemcpyHostToDevice),
          "copy attention cosine");
    check(cudaMemcpy(device_sine, host_sine.data(),
                     decode_tokens * 32U * sizeof(float), cudaMemcpyHostToDevice),
          "copy attention sine");
    auto attention_state = er::cuda::create_deepseek_attention_state(
        oracle_ratio, 4096U);
    require(attention_state.status.ok() && attention_state.state,
            std::string(attention_state.status.message()));
    cudaEvent_t attention_start{}, attention_stop{};
    check(cudaEventCreate(&attention_start), "create attention start event");
    check(cudaEventCreate(&attention_stop), "create attention stop event");
    check(cudaEventRecord(attention_start), "record attention start");
    std::vector<float> actual_output(stream_values);
    for (std::uint32_t position = 0U; position < decode_tokens; ++position) {
      const auto attention_status = er::cuda::deepseek_attention_decode({
          &oracle_attention, attention_state.state.get(),
          device_streams + position * token_stream_values,
          device_output + position * token_stream_values,
          device_cosine + position * 32U, device_sine + position * 32U,
          device_cosine, device_sine, position, 1e-6F, 20U, nullptr});
      require(attention_status.ok(), std::string(attention_status.message()));
    }
    check(cudaEventRecord(attention_stop), "record attention stop");
    check(cudaEventSynchronize(attention_stop), "synchronize attention");
    float attention_ms = 0.0F;
    check(cudaEventElapsedTime(&attention_ms, attention_start, attention_stop),
          "measure attention");
    check(cudaMemcpy(actual_output.data(), device_output,
                     stream_values * sizeof(float), cudaMemcpyDeviceToHost),
          "copy attention output");
    auto ffn_state = er::cuda::create_deepseek_ffn_state(oracle_layer);
    require(ffn_state.status.ok() && ffn_state.state,
            std::string(ffn_state.status.message()));
    cudaEvent_t router_start{}, router_stop{};
    check(cudaEventCreate(&router_start), "create router start event");
    check(cudaEventCreate(&router_stop), "create router stop event");
    check(cudaEventRecord(router_start), "record router start");
    for (std::uint32_t position = 0U; position < decode_tokens; ++position) {
      const auto route_status = er::cuda::deepseek_ffn_route({
          &oracle_ffn, ffn_state.state.get(),
          device_output + position * token_stream_values, position,
          1e-6F, 20U, nullptr});
      require(route_status.ok(), std::string(route_status.message()));
    }
    check(cudaEventRecord(router_stop), "record router stop");
    check(cudaEventSynchronize(router_stop), "synchronize router");
    float router_ms = 0.0F;
    check(cudaEventElapsedTime(&router_ms, router_start, router_stop),
          "measure router");
    std::vector<float> actual_router_scores(router_values);
    std::vector<std::uint32_t> actual_router_indices(router_values);
    for (std::uint32_t position = 0U; position < decode_tokens; ++position) {
      const auto route_status = er::cuda::deepseek_ffn_route({
          &oracle_ffn, ffn_state.state.get(),
          device_output + position * token_stream_values, position,
          1e-6F, 20U, nullptr});
      require(route_status.ok(), std::string(route_status.message()));
      check(cudaMemcpy(actual_router_scores.data() + position * 6U,
                       ffn_state.state->routing_weights(), 6U * sizeof(float),
                       cudaMemcpyDeviceToHost), "copy router scores");
      check(cudaMemcpy(actual_router_indices.data() + position * 6U,
                       ffn_state.state->expert_indices(),
                       6U * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
            "copy router indices");
    }
    float router_maximum = 0.0F;
    double router_squared = 0.0;
    for (std::size_t index = 0U; index < router_values; ++index) {
      require(actual_router_indices[index] == expected_router_indices[index],
              "hash router selected the wrong expert");
      const auto error = std::abs(
          actual_router_scores[index] - expected_router_scores[index]);
      router_maximum = std::max(router_maximum, error);
      router_squared += static_cast<double>(error) * error;
    }
    require(router_maximum < 1e-3F,
            "hash router weights exceed oracle tolerance");
    const auto final_route = er::cuda::deepseek_ffn_route({
        &oracle_ffn, ffn_state.state.get(),
        device_output + 3U * token_stream_values, 3U, 1e-6F, 20U, nullptr});
    require(final_route.ok(), std::string(final_route.message()));
    const auto directory_plan = expert_directory->pin_or_collect_misses(
        oracle_layer, ffn_state.state->expert_indices(),
        ffn_state.state->selection_count(), nullptr);
    require(directory_plan.status.ok() &&
                directory_plan.missing_experts.empty() &&
                directory_plan.unique_experts == 7U,
            "FFN route dependencies are not resident");
    const auto concurrent_directory_plan =
        expert_directory->pin_or_collect_misses(
            oracle_layer, ffn_state.state->expert_indices(),
            ffn_state.state->selection_count(), nullptr);
    require(concurrent_directory_plan.status.ok() &&
                concurrent_directory_plan.missing_experts.empty() &&
                concurrent_directory_plan.pin_id != directory_plan.pin_id,
            "CUDA directory did not retain concurrent request pins");
    float* device_block_output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_block_output),
                     token_stream_values * sizeof(float)),
          "allocate block output");
    cudaEvent_t ffn_start{}, ffn_stop{};
    check(cudaEventCreate(&ffn_start), "create FFN start event");
    check(cudaEventCreate(&ffn_stop), "create FFN stop event");
    check(cudaEventRecord(ffn_start), "record FFN start");
    er::cuda::DeepSeekFfnExecutionTiming ffn_timing;
    const auto ffn_status = er::cuda::deepseek_ffn_execute({
        &oracle_ffn, ffn_state.state.get(), expert_directory->device_entries(),
        device_output + 3U * token_stream_values, device_block_output,
        257U, nullptr, &ffn_timing});
    require(ffn_status.ok(), std::string(ffn_status.message()));
    check(cudaEventRecord(ffn_stop), "record FFN stop");
    check(cudaEventSynchronize(ffn_stop), "synchronize FFN");
    float ffn_ms = 0.0F;
    check(cudaEventElapsedTime(&ffn_ms, ffn_start, ffn_stop), "measure FFN");
    std::array<std::uint32_t, 6U> cpu_route{};
    check(cudaMemcpy(cpu_route.data(), ffn_state.state->expert_indices(),
                     cpu_route.size() * sizeof(std::uint32_t),
                     cudaMemcpyDeviceToHost),
          "copy routed experts for CPU oracle");
    std::vector<er::HostExpertLease> cpu_leases;
    cpu_leases.reserve(cpu_route.size());
    for (const auto expert : cpu_route) {
      const auto spec = std::find_if(
          ffn.begin(), ffn.begin() + 6U, [&](const auto& value) {
            return value.key.expert == expert;
          });
      require(spec != ffn.begin() + 6U,
              "CPU oracle expert is absent from the resident route");
      auto lease = expert_cache.try_acquire_host(spec->key, spec->record,
                                                 false);
      require(lease.has_value(),
              "CPU oracle expert has no retained compact RAM payload");
      cpu_leases.push_back(std::move(*lease));
    }
    std::vector<float> cpu_input(4096U),
        gpu_selection_output(6U * 4096U);
    check(cudaMemcpy(cpu_input.data(), ffn_state.state->normalized_input(),
                     cpu_input.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy FFN input for CPU oracle");
    check(cudaMemcpy(gpu_selection_output.data(),
                     ffn_state.state->routed_selection_outputs(),
                     gpu_selection_output.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy GPU expert output for CPU oracle");
    er::cpu::DeepSeekPackedExecutor cpu_executor({
        std::max(1U, std::thread::hardware_concurrency()),
        8U, 8U, 10.0F, true, true});
    std::vector<er::cpu::DeepSeekPackedWorkGroup> cpu_groups;
    cpu_groups.reserve(cpu_route.size());
    for (std::uint32_t slot = 0U; slot < cpu_route.size(); ++slot) {
      cpu_groups.push_back({cpu_leases[slot].bytes(),
                            cpu_leases[slot].compact_sections(),
                            4096U, 2048U, {slot}, {slot}});
    }
    std::vector<float> cpu_selection_output(6U * 4096U);
    const auto cpu_started = std::chrono::steady_clock::now();
    const auto cpu_status = cpu_executor.execute(
        cpu_groups, cpu_input, 1U, 6U,
        cpu_selection_output);
    const auto cpu_stopped = std::chrono::steady_clock::now();
    require(cpu_status.ok(), std::string(cpu_status.message()));
    float cpu_expert_maximum = 0.0F;
    float cpu_expert_reference_maximum = 0.0F;
    double cpu_expert_squared = 0.0;
    for (std::size_t index = 0U; index < cpu_selection_output.size(); ++index) {
      const auto error = std::abs(cpu_selection_output[index] -
                                  gpu_selection_output[index]);
      cpu_expert_maximum = std::max(cpu_expert_maximum, error);
      cpu_expert_reference_maximum = std::max(
          cpu_expert_reference_maximum, std::abs(gpu_selection_output[index]));
      cpu_expert_squared += static_cast<double>(error) * error;
    }
    const auto cpu_expert_rmse =
        std::sqrt(cpu_expert_squared / cpu_selection_output.size());
    const auto cpu_expert_relative_maximum =
        cpu_expert_reference_maximum > 0.0F
            ? cpu_expert_maximum / cpu_expert_reference_maximum
            : cpu_expert_maximum;
    require(cpu_expert_relative_maximum < 1e-3F,
            "all-core CPU expert exceeds direct-FP4 CUDA tolerance; max=" +
                std::to_string(cpu_expert_maximum) + ", rmse=" +
                std::to_string(cpu_expert_rmse) + ", reference_max=" +
                std::to_string(cpu_expert_reference_maximum) +
                ", relative_max=" +
                std::to_string(cpu_expert_relative_maximum));
    const auto cpu_metrics = cpu_executor.telemetry();
    require(cpu_metrics.workers_used_last == cpu_metrics.maximum_threads,
            "real DeepSeek CPU expert did not engage every logical worker");
    require(ffn_timing.routed_selections == 6U &&
                ffn_timing.routed_gpu_ms > 0.0F,
            "DeepSeek routed GPU timing was not measured");

    void* h2d_host = nullptr;
    void* h2d_device = nullptr;
    const auto h2d_record_bytes = cpu_leases.front().bytes().size();
    check(cudaHostAlloc(&h2d_host, h2d_record_bytes, cudaHostAllocPortable),
          "allocate DeepSeek H2D profile host buffer");
    check(cudaMalloc(&h2d_device, h2d_record_bytes),
          "allocate DeepSeek H2D profile device buffer");
    std::memcpy(h2d_host, cpu_leases.front().bytes().data(), h2d_record_bytes);
    check(cudaMemcpy(h2d_device, h2d_host, h2d_record_bytes,
                     cudaMemcpyHostToDevice),
          "warm DeepSeek H2D profile transfer");
    constexpr std::uint32_t h2d_iterations = 8U;
    check(cudaEventRecord(ffn_start), "record H2D profile start");
    for (std::uint32_t iteration = 0U; iteration < h2d_iterations;
         ++iteration) {
      check(cudaMemcpyAsync(h2d_device, h2d_host, h2d_record_bytes,
                            cudaMemcpyHostToDevice),
            "run DeepSeek H2D profile transfer");
    }
    check(cudaEventRecord(ffn_stop), "record H2D profile stop");
    check(cudaEventSynchronize(ffn_stop), "synchronize H2D profile");
    float h2d_profile_ms = 0.0F;
    check(cudaEventElapsedTime(&h2d_profile_ms, ffn_start, ffn_stop),
          "measure DeepSeek H2D profile");
    require(h2d_profile_ms > 0.0F, "DeepSeek H2D profile measured zero time");
    const auto h2d_profile_bytes =
        static_cast<std::uint64_t>(h2d_record_bytes) * h2d_iterations;
    const auto h2d_profile_bytes_per_second =
        static_cast<double>(h2d_profile_bytes) * 1000.0 /
        h2d_profile_ms;
    check(cudaFree(h2d_device), "free DeepSeek H2D profile device buffer");
    check(cudaFreeHost(h2d_host), "free DeepSeek H2D profile host buffer");
    float* device_hybrid_output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_hybrid_output),
                     token_stream_values * sizeof(float)),
          "allocate DeepSeek hybrid block output");
    const er::cpu::DeepSeekPackedWorkGroup hybrid_cpu_group{
        cpu_leases[0].bytes(), cpu_leases[0].compact_sections(),
        4096U, 2048U, {0U}, {0U}};
    auto hybrid_workspace = er::cuda::create_deepseek_ffn_hybrid_workspace();
    require(hybrid_workspace.status.ok() && hybrid_workspace.workspace,
            std::string(hybrid_workspace.status.message()));
    const auto hybrid_started = std::chrono::steady_clock::now();
    const auto hybrid_status = er::cuda::deepseek_ffn_execute_hybrid({
        &oracle_ffn, ffn_state.state.get(), expert_directory->device_entries(),
        device_output + 3U * token_stream_values, device_hybrid_output,
        hybrid_workspace.workspace.get(), &cpu_executor,
        std::span(&hybrid_cpu_group, 1U), 257U, nullptr});
    require(hybrid_status.ok(), std::string(hybrid_status.message()));
    check(cudaDeviceSynchronize(), "synchronize DeepSeek hybrid FFN");
    const auto hybrid_stopped = std::chrono::steady_clock::now();
    std::vector<float> hybrid_block_output(token_stream_values),
        hybrid_reference_block(token_stream_values);
    check(cudaMemcpy(hybrid_block_output.data(), device_hybrid_output,
                     hybrid_block_output.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy DeepSeek hybrid block output");
    check(cudaMemcpy(hybrid_reference_block.data(), device_block_output,
                     hybrid_reference_block.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "copy DeepSeek GPU block reference");
    float hybrid_block_maximum = 0.0F;
    double hybrid_block_squared = 0.0;
    for (std::size_t index = 0U; index < hybrid_block_output.size(); ++index) {
      const auto error = std::abs(hybrid_block_output[index] -
                                  hybrid_reference_block[index]);
      hybrid_block_maximum = std::max(hybrid_block_maximum, error);
      hybrid_block_squared += static_cast<double>(error) * error;
    }
    require(hybrid_block_maximum == 0.0F,
            "hybrid CPU/CUDA FFN changed the full block output");
    auto release = expert_directory->release_pins(
        concurrent_directory_plan.pin_id, nullptr);
    require(release.ok(), std::string(release.message()));
    release = expert_directory->release_pins(directory_plan.pin_id, nullptr);
    require(release.ok(), std::string(release.message()));
    std::vector<float> actual_block_output(token_stream_values);
    check(cudaMemcpy(actual_block_output.data(), device_block_output,
                     token_stream_values * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy block output");
    double block_squared = 0.0;
    float block_maximum = 0.0F;
    for (std::size_t index = 0U; index < token_stream_values; ++index) {
      require(std::isfinite(actual_block_output[index]),
              "transformer block output is non-finite");
      const auto error = std::abs(
          actual_block_output[index] - expected_block_output[index]);
      block_maximum = std::max(block_maximum, error);
      block_squared += static_cast<double>(error) * error;
    }
    require(block_maximum < 1e-2F,
            "transformer block output exceeds oracle tolerance; max=" +
                std::to_string(block_maximum));

    const auto request_layer = request.state->layer(oracle_layer);
    for (std::uint32_t position = 0U; position + 1U < decode_tokens;
         ++position) {
      const auto attention_status = er::cuda::deepseek_attention_decode({
          request_layer.attention_weights, request_layer.attention_state,
          device_streams + position * token_stream_values,
          device_output + position * token_stream_values,
          device_cosine + position * 32U, device_sine + position * 32U,
          nullptr, nullptr, position, 1e-6F, 20U, nullptr});
      require(attention_status.ok(), std::string(attention_status.message()));
    }
    auto controller = er::cuda::create_deepseek_decode_controller(
        request.state, expert_directory, nullptr);
    require(controller.status.ok() && controller.controller,
            std::string(controller.status.message()));
    const auto begin = controller.controller->begin({
        device_streams + 3U * token_stream_values,
        {device_cosine + 3U * 32U, device_sine + 3U * 32U,
         device_cosine + 3U * 32U, device_sine + 3U * 32U,
         device_cosine, device_sine, device_cosine, device_sine},
        3U, 3U, oracle_layer, oracle_layer + 1U});
    require(begin.ok(), std::string(begin.message()));
    const auto advanced = controller.controller->advance();
    require(advanced.status.ok() &&
                advanced.progress ==
                    er::cuda::DeepSeekDecodeProgress::token_complete &&
                advanced.missing_experts.empty(),
            std::string(advanced.status.message()));
    std::vector<float> controller_output(token_stream_values);
    check(cudaMemcpy(controller_output.data(),
                     controller.controller->output_streams(),
                     token_stream_values * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy controller output");
    float controller_maximum = 0.0F;
    for (std::size_t index = 0U; index < token_stream_values; ++index) {
      controller_maximum = std::max(
          controller_maximum,
          std::abs(controller_output[index] - expected_block_output[index]));
    }
    require(controller_maximum < 1e-2F,
            "decode controller exceeds block oracle tolerance");

    auto resumed_request = er::cuda::create_deepseek_request_state(
        model, {4096U, request_size.total_bytes});
    require(resumed_request.status.ok() && resumed_request.state,
            std::string(resumed_request.status.message()));
    const auto resumed_layer = resumed_request.state->layer(oracle_layer);
    for (std::uint32_t position = 0U; position + 1U < decode_tokens;
         ++position) {
      const auto attention_status = er::cuda::deepseek_attention_decode({
          resumed_layer.attention_weights, resumed_layer.attention_state,
          device_streams + position * token_stream_values,
          device_output + position * token_stream_values,
          device_cosine + position * 32U, device_sine + position * 32U,
          nullptr, nullptr, position, 1e-6F, 20U, nullptr});
      require(attention_status.ok(), std::string(attention_status.message()));
    }
    auto resumed_directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        17U, er::kExpertQuantAbiDeepSeekSm86, 43U, 257U, 8U);
    auto resumed_buffers = std::make_shared<er::FixedBufferPool>(
        1U, 25'167'360U, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    er::ExpertCache resumed_cache(
        expert_config, expert_storage, expert_uploader, resumed_buffers,
        resumed_directory);
    {
      er::ResidentExpertSet cpu_seed;
      const auto seeded = er::ResidentExpertSet::load(
          resumed_cache,
          std::span<const er::ResidentExpertSpec>(ffn.data(), 1U), cpu_seed);
      require(seeded.ok() && cpu_seed.size() == 1U,
              std::string(seeded.message()));
    }
    const auto trimmed_seed = resumed_cache.trim_to(
        128ULL << 20U, 0U);
    const auto seeded_snapshot = resumed_cache.inspect(ffn.front().key);
    require(seeded_snapshot && seeded_snapshot->has_host_copy &&
                !seeded_snapshot->has_device_copy &&
                trimmed_seed.vram_bytes == 0U,
            "DeepSeek CPU seed did not remain host-only after device trim");
    auto resumed_controller = er::cuda::create_deepseek_decode_controller(
        resumed_request.state, resumed_directory, nullptr);
    require(resumed_controller.status.ok() && resumed_controller.controller,
            std::string(resumed_controller.status.message()));
    er::ResidentExpertSet resumed_shared;
    const auto resumed_load = er::ResidentExpertSet::load(
        resumed_cache,
        std::span<const er::ResidentExpertSpec>(ffn.data() + 6U, 1U),
        resumed_shared);
    require(resumed_load.ok() && resumed_shared.size() == 1U,
            std::string(resumed_load.message()));
    auto scheduler_cpu = std::make_shared<er::cpu::DeepSeekPackedExecutor>(
        er::cpu::DeepSeekPackedExecutorConfig{
            std::max(1U, std::thread::hardware_concurrency()),
            8U, 8U, 10.0F, true, true});
    auto scheduler_planner = std::make_shared<er::HybridDispatchPlanner>(
        er::HybridDispatchConfig{1.0, 1.0, 1.0, 0.125, 6U, 32U});
    auto scheduler_census = std::make_shared<er::RouteCensus>(
        er::RouteCensusConfig{17U, ffn.front().record.payload_sha256,
                              er::kExpertQuantAbiDeepSeekSm86, 43U, 256U, 6U,
                              4096U});
    er::cuda::DeepSeekDecodeScheduler decode_scheduler(
        {17U, 2U, 2U, 1U}, resumed_cache, routed_catalog,
        {scheduler_cpu, scheduler_planner, scheduler_census});
    const auto submitted = decode_scheduler.submit(
        1U, resumed_controller.controller,
        {device_streams + 3U * token_stream_values,
         {device_cosine + 3U * 32U, device_sine + 3U * 32U,
          device_cosine + 3U * 32U, device_sine + 3U * 32U,
          device_cosine, device_sine, device_cosine, device_sine},
         3U, 3U, oracle_layer, oracle_layer + 1U});
    require(submitted.ok(), std::string(submitted.message()));
    std::size_t scheduler_peak_acquires = 0U;
    std::size_t scheduler_peak_leases = 0U;
    std::size_t scheduler_peak_host_leases = 0U;
    const auto scheduler_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (;;) {
      const auto polled = decode_scheduler.poll();
      require(polled.ok(), std::string(polled.message()));
      const auto scheduled = decode_scheduler.inspect(1U);
      require(scheduled.has_value(), "scheduled DeepSeek request disappeared");
      const auto scheduler_state = decode_scheduler.snapshot();
      scheduler_peak_acquires = std::max(
          scheduler_peak_acquires, scheduler_state.inflight_acquires);
      scheduler_peak_leases = std::max(
          scheduler_peak_leases, scheduled->held_leases);
      scheduler_peak_host_leases = std::max(
          scheduler_peak_host_leases, scheduled->held_host_leases);
      if (scheduled->state == er::cuda::DeepSeekScheduledState::complete)
        break;
      require(scheduled->state != er::cuda::DeepSeekScheduledState::failed &&
                  scheduled->state !=
                      er::cuda::DeepSeekScheduledState::cancelled,
              std::string(scheduled->status.message()));
      require(std::chrono::steady_clock::now() < scheduler_deadline,
              "DeepSeek async scheduler timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto scheduler_state = decode_scheduler.snapshot();
    require(scheduler_state.completed_requests == 1U &&
                scheduler_state.expert_suspensions == 1U &&
                scheduler_state.acquires_started == 5U &&
                scheduler_state.acquires_completed == 5U &&
                scheduler_state.host_resolves == 1U &&
                scheduler_state.cpu_placements == 1U &&
                scheduler_state.hybrid_layers == 1U &&
                scheduler_state.route_observations == 1U &&
                scheduler_census->snapshot().completed_routes == 1U &&
                scheduler_census->snapshot().total_selections == 6U &&
                scheduler_peak_acquires == 2U &&
                scheduler_peak_leases > 0U &&
                scheduler_peak_leases <= 5U &&
                scheduler_peak_host_leases == 1U,
            "DeepSeek async scheduler accounting is inconsistent");
    std::vector<float> resumed_output(token_stream_values);
    check(cudaMemcpy(resumed_output.data(),
                     resumed_controller.controller->output_streams(),
                     token_stream_values * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy resumed controller output");
    float resumed_maximum = 0.0F;
    for (std::size_t index = 0U; index < token_stream_values; ++index) {
      resumed_maximum = std::max(
          resumed_maximum,
          std::abs(resumed_output[index] - expected_block_output[index]));
    }
    require(resumed_maximum < 1e-2F,
            "resumed decode controller exceeds block oracle tolerance");

    constexpr std::uint64_t shared_hot_bytes = 43ULL * 25'198'592U;
    constexpr std::uint64_t routed_cache_slots = 258U;
    constexpr std::uint64_t routed_cache_bytes =
        routed_cache_slots * 25'198'592U;
    constexpr std::uint64_t compact_cache_bytes = 0U;
    auto full_directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        17U, er::kExpertQuantAbiDeepSeekSm86, 43U, 257U, 64U);
    auto full_buffers = std::make_shared<er::FixedBufferPool>(
        4U, 25'167'360U, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    er::ExpertCacheConfig full_config;
    const auto host_cache_bytes = host_cache_gib << 30U;
    const auto full_ram_bytes =
        std::max<std::uint64_t>(4ULL * 25'167'360U, host_cache_bytes);
    const auto full_ram_low = host_cache_bytes == 0U
                                  ? 25'167'360U
                                  : host_cache_bytes * 7U / 8U;
    full_config.ram = {full_ram_bytes, full_ram_bytes, full_ram_low};
    full_config.vram = {shared_hot_bytes + routed_cache_bytes,
                        shared_hot_bytes + routed_cache_bytes,
                        25'198'592U};
    full_config.retain_host_copy = host_cache_bytes != 0U;
    // The complete catalog was SHA-256 authenticated when it was generated
    // from this content-addressed checkpoint snapshot.
    full_config.trusted_immutable_source = true;
    auto full_uploader = std::make_shared<er::cuda::CudaExpertUploader>(
        er::cuda::CudaExpertUploaderOptions{
            routed_cache_bytes, true, 0U, true});
    er::ExpertCache full_cache(full_config, expert_storage, full_uploader,
                               full_buffers, full_directory);
    er::ResidentExpertSet full_shared;
    auto full_status = er::ResidentExpertSet::load(
        full_cache, all_shared, full_shared);
    require(full_status.ok() && full_shared.size() == 43U,
            std::string(full_status.message()));
    auto full_request = er::cuda::create_deepseek_request_state(
        model, {4096U, request_size.total_bytes});
    require(full_request.status.ok() && full_request.state,
            std::string(full_request.status.message()));
    auto full_controller = er::cuda::create_deepseek_decode_controller(
        full_request.state, full_directory, nullptr);
    require(full_controller.status.ok() && full_controller.controller,
            std::string(full_controller.status.message()));
    er::cuda::DeepSeekDecodeScheduler full_scheduler(
        {17U, 1U, 6U, 1U, true}, full_cache, routed_catalog);
    float* full_rope = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&full_rope),
                     8U * 32U * sizeof(float)),
          "allocate full-model RoPE");
    const std::vector<std::uint32_t> full_inputs =
        prompt ? *prompt : std::vector<std::uint32_t>{42U};
    require(full_inputs.size() <= 4096U,
            "DeepSeek prompt exceeds request context");
    using RouteSet = std::array<std::uint32_t, 6U>;
    std::vector<std::array<RouteSet, 43U>> full_route_trace(
        full_inputs.size() + max_new_tokens - 1U);
    std::vector<std::array<bool, 43U>> full_route_seen(
        full_route_trace.size());
    float* prefill_streams = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&prefill_streams),
                     full_inputs.size() * token_stream_values * sizeof(float)),
          "allocate layer-major prefill streams");
    const auto full_started = std::chrono::steady_clock::now();
    const auto full_deadline = full_started + std::chrono::minutes(5);
    for (std::uint32_t position = 0U; position < full_inputs.size(); ++position) {
      full_status = full_request.state->embed(full_inputs[position]);
      require(full_status.ok(), std::string(full_status.message()));
      check(cudaMemcpy(prefill_streams + position * token_stream_values,
                       full_request.state->current_streams(),
                       token_stream_values * sizeof(float),
                       cudaMemcpyDeviceToDevice),
            "stage DeepSeek prefill embedding");
    }
    for (std::uint32_t layer = 0U; layer < 43U; ++layer) {
      for (std::uint32_t position = 0U; position < full_inputs.size();
           ++position) {
        const auto request_id =
            2U + static_cast<std::uint64_t>(layer) * full_inputs.size() +
            position;
        full_status = full_scheduler.submit(
            request_id, full_controller.controller,
            {prefill_streams + position * token_stream_values,
             upload_rope(position, full_rope), position, full_inputs[position],
             layer, layer + 1U});
        require(full_status.ok(), std::string(full_status.message()));
        for (;;) {
          full_status = full_scheduler.poll();
          require(full_status.ok(), std::string(full_status.message()));
          const auto scheduled = full_scheduler.inspect(request_id);
          require(scheduled.has_value(), "full DeepSeek request disappeared");
          if (scheduled->state == er::cuda::DeepSeekScheduledState::complete)
            break;
          require(scheduled->state != er::cuda::DeepSeekScheduledState::failed &&
                      scheduled->state !=
                          er::cuda::DeepSeekScheduledState::cancelled,
                  std::string(scheduled->status.message()));
          require(std::chrono::steady_clock::now() < full_deadline,
                  "full DeepSeek prompt timed out");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        check(cudaMemcpy(prefill_streams + position * token_stream_values,
                         full_controller.controller->output_streams(),
                         token_stream_values * sizeof(float),
                         cudaMemcpyDeviceToDevice),
              "commit DeepSeek prefill layer output");
        const auto trace = full_controller.controller->route_trace();
        require(trace.size() == 1U && trace.front().layer == layer,
                "layer-major DeepSeek route trace is incomplete");
        full_route_trace[position][layer] = trace.front().routed_experts;
        full_route_seen[position][layer] = true;
        full_status = full_scheduler.retire(request_id);
        require(full_status.ok(), std::string(full_status.message()));
      }
    }
    full_status = full_request.state->project_logits();
    require(full_status.ok(), std::string(full_status.message()));
    check(cudaDeviceSynchronize(), "synchronize full DeepSeek token");
    std::uint32_t full_sampled_token = 0U;
    check(cudaMemcpy(&full_sampled_token, full_request.state->sampled_token(),
                     sizeof(full_sampled_token), cudaMemcpyDeviceToHost),
          "copy full DeepSeek sampled token");
    require(full_sampled_token < 129280U,
            "full DeepSeek token is outside vocabulary");
    const auto full_token_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - full_started).count();
    std::vector<std::uint32_t> generated_tokens{full_sampled_token};
    const auto prefill_cache_state = full_cache.telemetry();
    const auto prefill_scheduler_state = full_scheduler.snapshot();
    const auto prefill_upload_state = full_uploader->telemetry();
    const auto decode_started = std::chrono::steady_clock::now();
    for (std::uint32_t generated = 1U; generated < max_new_tokens;
         ++generated) {
      const auto position = static_cast<std::uint32_t>(full_inputs.size()) +
                            generated - 1U;
      const auto token = generated_tokens.back();
      full_status = full_request.state->embed(token);
      require(full_status.ok(), std::string(full_status.message()));
      const auto request_id =
          2U + 43ULL * full_inputs.size() + generated;
      full_status = full_scheduler.submit(
          request_id, full_controller.controller,
          {full_request.state->current_streams(),
           upload_rope(position, full_rope), position, token, 0U, 43U});
      require(full_status.ok(), std::string(full_status.message()));
      for (;;) {
        full_status = full_scheduler.poll();
        require(full_status.ok(), std::string(full_status.message()));
        const auto scheduled = full_scheduler.inspect(request_id);
        require(scheduled.has_value(), "generated DeepSeek request disappeared");
        if (scheduled->state == er::cuda::DeepSeekScheduledState::complete)
          break;
        require(scheduled->state != er::cuda::DeepSeekScheduledState::failed &&
                    scheduled->state !=
                        er::cuda::DeepSeekScheduledState::cancelled,
                std::string(scheduled->status.message()));
        require(std::chrono::steady_clock::now() < full_deadline,
                "DeepSeek generation timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      const auto trace = full_controller.controller->route_trace();
      require(trace.size() == 43U,
              "generated DeepSeek route trace is incomplete");
      for (const auto& item : trace) {
        require(item.layer < 43U,
                "generated DeepSeek route trace layer is invalid");
        full_route_trace[position][item.layer] = item.routed_experts;
        full_route_seen[position][item.layer] = true;
      }
      full_status = full_scheduler.retire(request_id);
      require(full_status.ok(), std::string(full_status.message()));
      full_status = full_request.state->project_logits();
      require(full_status.ok(), std::string(full_status.message()));
      check(cudaDeviceSynchronize(), "synchronize generated DeepSeek token");
      std::uint32_t sampled = 0U;
      check(cudaMemcpy(&sampled, full_request.state->sampled_token(),
                       sizeof(sampled), cudaMemcpyDeviceToHost),
            "copy generated DeepSeek token");
      require(sampled < 129280U,
              "generated DeepSeek token is outside vocabulary");
      generated_tokens.push_back(sampled);
    }
    const auto decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - decode_started).count();
    const auto full_scheduler_state = full_scheduler.snapshot();
    const auto full_cache_state = full_cache.telemetry();
    const auto full_upload_state = full_uploader->telemetry();
    const auto first_decode_route = full_inputs.size() - 1U;
    const auto decode_route_steps = generated_tokens.size();
    std::uint64_t route_overlap_experts = 0U;
    std::uint64_t route_overlap_possible = 0U;
    std::uint64_t route_identical_layers = 0U;
    for (std::size_t step = 0U; step < decode_route_steps; ++step) {
      for (std::size_t layer = 0U; layer < 43U; ++layer) {
        require(full_route_seen[first_decode_route + step][layer],
                "DeepSeek decode route trace has a gap");
      }
      if (step == 0U) continue;
      for (std::size_t layer = 0U; layer < 43U; ++layer) {
        const auto& previous =
            full_route_trace[first_decode_route + step - 1U][layer];
        const auto& current =
            full_route_trace[first_decode_route + step][layer];
        std::uint32_t overlap = 0U;
        for (const auto expert : current) {
          if (std::find(previous.begin(), previous.end(), expert) !=
              previous.end()) {
            ++overlap;
          }
        }
        route_overlap_experts += overlap;
        route_overlap_possible += current.size();
        if (overlap == current.size()) ++route_identical_layers;
      }
    }
    require(full_scheduler_state.completed_requests ==
                43U * full_inputs.size() + max_new_tokens - 1U &&
                full_scheduler_state.layer_advances >=
                    43U * (full_inputs.size() + max_new_tokens - 1U),
            "full DeepSeek scheduler did not complete every layer");
    double squared = 0.0;
    float maximum = 0.0F;
    std::array<float, decode_tokens> token_maximum{};
    for (std::size_t index = 0; index < stream_values; ++index) {
      require(std::isfinite(actual_output[index]), "attention output is non-finite");
      const float error = std::abs(actual_output[index] - expected_output[index]);
      maximum = std::max(maximum, error);
      token_maximum[index / token_stream_values] = std::max(
          token_maximum[index / token_stream_values], error);
      squared += static_cast<double>(error) * error;
    }
    require(maximum < 1e-3F,
            "complete attention output exceeds oracle tolerance; max=" +
                std::to_string(maximum) + ", tokens=" +
                std::to_string(token_maximum[0]) + "," +
                std::to_string(token_maximum[1]) + "," +
                std::to_string(token_maximum[2]) + "," +
                std::to_string(token_maximum[3]));
    std::size_t free_resident = 0U;
    check(cudaMemGetInfo(&free_resident, &total),
          "cudaMemGetInfo resident model");
    MEMORYSTATUSEX memory_status{};
    memory_status.dwLength = sizeof(memory_status);
    require(GlobalMemoryStatusEx(&memory_status) != 0,
            "GlobalMemoryStatusEx failed during placement profile");
    const auto cpu_ns_per_selection =
        std::chrono::duration<double, std::nano>(cpu_stopped - cpu_started)
            .count() /
        cpu_route.size();
    const auto gpu_ns_per_selection =
        static_cast<double>(ffn_timing.routed_gpu_ms) * 1.0e6 /
        ffn_timing.routed_selections;
    const auto host_committed =
        memory_status.ullTotalPhys - memory_status.ullAvailPhys;
    const auto host_fixed =
        host_committed >= full_cache_state.ram_bytes
            ? host_committed - full_cache_state.ram_bytes
            : 0U;
    const auto device_committed = total - free_resident;
    const auto device_fixed =
        device_committed >= full_cache_state.vram_bytes
            ? device_committed - full_cache_state.vram_bytes
            : 0U;
    constexpr std::uint64_t host_emergency = 4ULL << 30U;
    constexpr std::uint64_t device_emergency = 1ULL << 30U;
    const er::PlacementProfileInput measured_profile{
        {cpu_ns_per_selection, gpu_ns_per_selection,
         h2d_profile_bytes_per_second, cpu_route.size(),
         ffn_timing.routed_selections, h2d_profile_bytes},
        {memory_status.ullTotalPhys, host_fixed, host_emergency,
         ffn.front().record.stored_bytes, 6U, 11'008U},
        {total, device_fixed, device_emergency,
         ffn.front().record.stored_bytes, 7U, 11'008U},
        0.125, 256U, 256U};
    const auto placement_plan =
        er::solve_placement_profile(measured_profile);
    require(placement_plan.status.ok() &&
                placement_plan.host_expert_slots >= 6U &&
                placement_plan.device_expert_slots >= 7U,
            std::string(placement_plan.status.message()));
    er::HybridDispatchPlanner measured_planner(placement_plan.dispatch);
    std::vector<er::HybridDispatchCandidate> measured_candidates;
    measured_candidates.reserve(cpu_route.size());
    for (const auto expert : cpu_route) {
      measured_candidates.push_back(
          {expert, 1U, ffn.front().record.stored_bytes,
           false, true, true});
    }
    const auto measured_dispatch = measured_planner.plan(measured_candidates);
    require(measured_dispatch.status.ok() &&
                measured_dispatch.decisions.size() == cpu_route.size(),
            std::string(measured_dispatch.status.message()));
    const auto measured_cpu_decisions = static_cast<std::uint64_t>(
        std::count_if(measured_dispatch.decisions.begin(),
                      measured_dispatch.decisions.end(), [](const auto& item) {
                        return item.executor == er::HybridExecutor::cpu_local;
                      }));
    const auto measured_gpu_decisions =
        measured_dispatch.decisions.size() - measured_cpu_decisions;
    const auto startup_ms = std::chrono::duration<double, std::milli>(
                                stopped - started).count();
    std::cout << "{\"ok\":true,\"dense_tensors\":" << model->dense_size()
              << ",\"typed_tensors\":" << model->typed_size()
              << ",\"source_bytes\":" << dense_source + typed_source
              << ",\"resident_bytes\":" << model->bytes()
              << ",\"staging_bytes\":" << staging
              << ",\"startup_ms\":" << startup_ms
              << ",\"oracle_layer\":" << oracle_layer
              << ",\"compress_ratio\":" << oracle_ratio
              << ",\"ratio0_bound\":true,\"ratio4_bound\":true"
              << ",\"ratio128_bound\":true"
              << ",\"hash_ffn_bound\":true,\"learned_ffn_bound\":true"
              << ",\"request_layers\":" << request.state->layer_count()
              << ",\"request_4096_bytes\":" << request.state->bytes()
              << ",\"request_attention_bytes\":"
              << request_size.attention_bytes
              << ",\"request_ffn_bytes\":" << request_size.ffn_bytes
              << ",\"request_io_bytes\":" << request_size.io_bytes
              << ",\"request_stream_bytes\":" << request_size.stream_bytes
              << ",\"embedding_max_abs_error\":" << embedding_maximum
              << ",\"head_rmse\":"
              << std::sqrt(logits_squared / actual_logits.size())
              << ",\"head_max_abs_error\":" << logits_maximum
              << ",\"head_sampled_token\":" << actual_sampled
              << ",\"request_state_bytes\":" << attention_state.state->bytes()
              << ",\"decode_tokens\":" << decode_tokens
              << ",\"attention_ms\":" << attention_ms
              << ",\"attention_ms_per_token\":"
              << attention_ms / decode_tokens
              << ",\"attention_rmse\":"
              << std::sqrt(squared / stream_values)
              << ",\"attention_max_abs_error\":" << maximum
              << ",\"ffn_state_bytes\":" << ffn_state.state->bytes()
              << ",\"router_ms_per_token\":" << router_ms / decode_tokens
              << ",\"router_rmse\":"
              << std::sqrt(router_squared / router_values)
              << ",\"router_max_abs_error\":" << router_maximum
              << ",\"ffn_experts\":" << ffn_resident.size()
              << ",\"ffn_source_bytes\":" << ffn_source
              << ",\"ffn_resident_bytes\":" << ffn_resident.bytes()
              << ",\"ffn_load_ms\":"
              << std::chrono::duration<double, std::milli>(
                     ffn_load_stopped - ffn_load_started).count()
              << ",\"ffn_execute_ms\":" << ffn_ms
              << ",\"cpu_expert_count\":" << cpu_route.size()
              << ",\"cpu_expert_threads\":"
              << cpu_metrics.workers_used_last
              << ",\"cpu_expert_ms\":"
              << std::chrono::duration<double, std::milli>(
                     cpu_stopped - cpu_started).count()
              << ",\"cpu_expert_rmse\":"
              << cpu_expert_rmse
              << ",\"cpu_expert_max_abs_error\":" << cpu_expert_maximum
              << ",\"cpu_expert_max_relative_error\":"
              << cpu_expert_relative_maximum
              << ",\"hybrid_cpu_experts\":1"
              << ",\"hybrid_device_workspace_bytes\":"
              << hybrid_workspace.workspace->device_bytes()
              << ",\"hybrid_pinned_workspace_bytes\":"
              << hybrid_workspace.workspace->pinned_host_bytes()
              << ",\"hybrid_ffn_ms\":"
              << std::chrono::duration<double, std::milli>(
                     hybrid_stopped - hybrid_started).count()
              << ",\"hybrid_block_rmse\":"
              << std::sqrt(hybrid_block_squared /
                           hybrid_block_output.size())
              << ",\"hybrid_block_max_abs_error\":"
              << hybrid_block_maximum
              << ",\"block_rmse\":"
              << std::sqrt(block_squared / token_stream_values)
              << ",\"block_max_abs_error\":" << block_maximum
              << ",\"controller_max_abs_error\":" << controller_maximum
              << ",\"scheduler_cold_misses\":"
              << scheduler_state.acquires_started
              << ",\"scheduler_peak_acquires\":"
              << scheduler_peak_acquires
              << ",\"scheduler_peak_leases\":" << scheduler_peak_leases
              << ",\"scheduler_peak_host_leases\":"
              << scheduler_peak_host_leases
              << ",\"scheduler_cpu_placements\":"
              << scheduler_state.cpu_placements
              << ",\"scheduler_hybrid_layers\":"
              << scheduler_state.hybrid_layers
              << ",\"scheduler_route_observations\":"
              << scheduler_state.route_observations
              << ",\"profile_cpu_ns_per_selection\":"
              << cpu_ns_per_selection
              << ",\"profile_gpu_ns_per_selection\":"
              << gpu_ns_per_selection
              << ",\"profile_h2d_bytes_per_second\":"
              << h2d_profile_bytes_per_second
              << ",\"profile_h2d_sample_bytes\":" << h2d_profile_bytes
              << ",\"profile_host_cache_bytes\":"
              << placement_plan.host_cache_bytes
              << ",\"profile_device_cache_bytes\":"
              << placement_plan.device_cache_bytes
              << ",\"profile_host_expert_slots\":"
              << placement_plan.host_expert_slots
              << ",\"profile_device_expert_slots\":"
              << placement_plan.device_expert_slots
              << ",\"profile_dispatch_cpu_decisions\":"
              << measured_cpu_decisions
              << ",\"profile_dispatch_gpu_decisions\":"
              << measured_gpu_decisions
              << ",\"profile_dispatch_projected_ns\":"
              << measured_dispatch.projected_critical_ns
              << ",\"controller_resume_max_abs_error\":" << resumed_maximum
              << ",\"full_token_input\":" << full_inputs.front()
              << ",\"full_token_output\":" << full_sampled_token
              << ",\"full_token_ms\":" << full_token_ms
              << ",\"routed_cache_slots\":" << routed_cache_slots
              << ",\"host_cache_bytes\":" << host_cache_bytes
              << ",\"compact_vram_cache_bytes\":" << compact_cache_bytes
              << ",\"generated_tokens\":" << generated_tokens.size()
              << ",\"decode_generated_ms\":" << decode_ms
              << ",\"decode_ms_per_token\":"
              << (generated_tokens.size() > 1U
                      ? decode_ms / (generated_tokens.size() - 1U)
                      : 0.0)
              << ",\"decode_tokens_per_second\":"
              << (generated_tokens.size() > 1U && decode_ms > 0.0
                      ? 1000.0 * (generated_tokens.size() - 1U) / decode_ms
                      : 0.0)
              << ",\"decode_route_overlap_experts\":"
              << route_overlap_experts
              << ",\"decode_route_overlap_possible\":"
              << route_overlap_possible
              << ",\"decode_route_overlap_ratio\":"
              << (route_overlap_possible != 0U
                      ? static_cast<double>(route_overlap_experts) /
                            static_cast<double>(route_overlap_possible)
                      : 0.0)
              << ",\"decode_route_identical_layers\":"
              << route_identical_layers
              << ",\"generated_token_ids\":[";
    for (std::size_t index = 0U; index < generated_tokens.size(); ++index) {
      if (index != 0U) std::cout << ',';
      std::cout << generated_tokens[index];
    }
    std::cout << "]"
              << ",\"decode_route_expert_ids\":[";
    for (std::size_t step = 0U; step < decode_route_steps; ++step) {
      if (step != 0U) std::cout << ',';
      std::cout << '[';
      for (std::size_t layer = 0U; layer < 43U; ++layer) {
        if (layer != 0U) std::cout << ',';
        std::cout << '[';
        const auto& route =
            full_route_trace[first_decode_route + step][layer];
        for (std::size_t rank = 0U; rank < route.size(); ++rank) {
          if (rank != 0U) std::cout << ',';
          std::cout << route[rank];
        }
        std::cout << ']';
      }
      std::cout << ']';
    }
    std::cout << ']'
              << ",\"prompt_tokens\":" << full_inputs.size()
              << ",\"prompt_token_ids\":[";
    for (std::size_t index = 0U; index < full_inputs.size(); ++index) {
      if (index != 0U) std::cout << ',';
      std::cout << full_inputs[index];
    }
    std::cout << "]"
              << ",\"full_token_layer_advances\":"
              << full_scheduler_state.layer_advances
              << ",\"full_token_cold_acquires\":"
              << full_scheduler_state.acquires_started
              << ",\"retained_working_set_experts\":"
              << full_scheduler_state.retained_working_set_experts
              << ",\"full_cache_read_bytes\":"
              << full_cache_state.read_bytes
              << ",\"decode_cache_read_bytes\":"
              << (full_cache_state.read_bytes - prefill_cache_state.read_bytes)
              << ",\"decode_cache_ram_hits\":"
              << (full_cache_state.acquire_ram_hits -
                  prefill_cache_state.acquire_ram_hits)
              << ",\"decode_cold_acquires\":"
              << (full_scheduler_state.acquires_started -
                  prefill_scheduler_state.acquires_started)
              << ",\"decode_uploaded_bytes\":"
              << (full_cache_state.uploaded_bytes -
                  prefill_cache_state.uploaded_bytes)
              << ",\"full_cache_ram_high_water\":"
              << full_cache_state.ram_high_water
              << ",\"full_cache_uploaded_bytes\":"
              << full_cache_state.uploaded_bytes
              << ",\"full_cache_vram_high_water\":"
              << full_cache_state.vram_high_water
              << ",\"full_upload_device_allocations\":"
              << full_upload_state.device_allocations
              << ",\"full_upload_recycled_acquires\":"
              << full_upload_state.recycled_acquires
              << ",\"full_upload_recycled_releases\":"
              << full_upload_state.recycled_releases
              << ",\"full_upload_recycled_bytes\":"
              << full_upload_state.recycled_bytes
              << ",\"full_upload_device_bytes_high_water\":"
              << full_upload_state.device_bytes_high_water
              << ",\"full_upload_staging_allocations\":"
              << full_upload_state.staging_allocations
              << ",\"compact_cache_hits\":"
              << full_upload_state.compact_cache_hits
              << ",\"compact_cache_misses\":"
              << full_upload_state.compact_cache_misses
              << ",\"compact_cache_evictions\":"
              << full_upload_state.compact_cache_evictions
              << ",\"compact_h2d_bytes\":"
              << full_upload_state.compact_h2d_bytes
              << ",\"compact_cache_high_water\":"
              << full_upload_state.compact_cache_high_water
              << ",\"decode_compact_cache_hits\":"
              << (full_upload_state.compact_cache_hits -
                  prefill_upload_state.compact_cache_hits)
              << ",\"decode_compact_cache_misses\":"
              << (full_upload_state.compact_cache_misses -
                  prefill_upload_state.compact_cache_misses)
              << ",\"cuda_free_before\":" << free_before
              << ",\"cuda_free_resident\":" << free_resident << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-model-residency: " << error.what() << '\n';
    return 1;
  }
}
