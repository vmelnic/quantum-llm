#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_attention.hpp"
#include "expert/runtime/cuda/deepseek_decode.hpp"
#include "expert/runtime/cuda/deepseek_ffn.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/deepseek_request.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/resident_expert_set.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: expert-deepseek-model-residency "
                   "<dense-bundle> <typed-bundle> <checkpoint> <attention-oracle>\n";
      return 64;
    }
    const std::filesystem::path source = argv[3];
    std::uint64_t dense_source = 0U, dense_device = 0U, maximum_dense = 0U;
    std::uint64_t typed_source = 0U;
    const auto dense = dense_specs(argv[1], source, dense_source, dense_device,
                                   maximum_dense);
    const auto typed = typed_specs(argv[2], source, typed_source);
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
    auto expert_uploader = std::make_shared<er::cuda::CudaExpertUploader>();
    auto expert_buffers = std::make_shared<er::FixedBufferPool>(
        1U, 25'167'360U, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    auto expert_directory = std::make_shared<er::cuda::CudaExpertDirectory>(
        17U, er::kExpertQuantAbiDeepSeekSm86, 43U, 257U, 8U);
    er::ExpertCacheConfig expert_config;
    expert_config.ram = {50'334'720U, 50'334'720U, 25'167'360U};
    expert_config.vram = {7ULL * 25'198'592U, 7ULL * 25'198'592U,
                          25'198'592U};
    expert_config.retain_host_copy = false;
    er::ExpertCache expert_cache(expert_config, expert_storage,
                                 expert_uploader, expert_buffers,
                                 expert_directory);
    er::ResidentExpertSet ffn_resident;
    const auto ffn_load_started = std::chrono::steady_clock::now();
    const auto ffn_load_status = er::ResidentExpertSet::load(
        expert_cache, ffn, ffn_resident);
    const auto ffn_load_stopped = std::chrono::steady_clock::now();
    require(ffn_load_status.ok() && ffn_resident.size() == 7U &&
                ffn_resident.bytes() == 7ULL * 25'198'592U,
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
    const auto ffn_status = er::cuda::deepseek_ffn_execute({
        &oracle_ffn, ffn_state.state.get(), expert_directory->device_entries(),
        device_output + 3U * token_stream_values, device_block_output,
        257U, nullptr});
    require(ffn_status.ok(), std::string(ffn_status.message()));
    check(cudaEventRecord(ffn_stop), "record FFN stop");
    check(cudaEventSynchronize(ffn_stop), "synchronize FFN");
    float ffn_ms = 0.0F;
    check(cudaEventElapsedTime(&ffn_ms, ffn_start, ffn_stop), "measure FFN");
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
    auto resumed_controller = er::cuda::create_deepseek_decode_controller(
        resumed_request.state, resumed_directory, nullptr);
    require(resumed_controller.status.ok() && resumed_controller.controller,
            std::string(resumed_controller.status.message()));
    const auto resumed_begin = resumed_controller.controller->begin({
        device_streams + 3U * token_stream_values,
        {device_cosine + 3U * 32U, device_sine + 3U * 32U,
         device_cosine + 3U * 32U, device_sine + 3U * 32U,
         device_cosine, device_sine, device_cosine, device_sine},
        3U, 3U, oracle_layer, oracle_layer + 1U});
    require(resumed_begin.ok(), std::string(resumed_begin.message()));
    auto resumed_advance = resumed_controller.controller->advance();
    require(resumed_advance.status.ok() &&
                resumed_advance.progress ==
                    er::cuda::DeepSeekDecodeProgress::needs_experts &&
                resumed_advance.missing_experts.size() == 7U,
            "decode controller did not suspend on the cold route");
    er::ResidentExpertSet resumed_resident;
    const auto resumed_load = er::ResidentExpertSet::load(
        resumed_cache, ffn, resumed_resident);
    require(resumed_load.ok() && resumed_resident.size() == 7U,
            std::string(resumed_load.message()));
    resumed_advance = resumed_controller.controller->advance();
    require(resumed_advance.status.ok() &&
                resumed_advance.progress ==
                    er::cuda::DeepSeekDecodeProgress::token_complete,
            std::string(resumed_advance.status.message()));
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
              << ",\"request_stream_bytes\":" << request_size.stream_bytes
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
              << ",\"block_rmse\":"
              << std::sqrt(block_squared / token_stream_values)
              << ",\"block_max_abs_error\":" << block_maximum
              << ",\"controller_max_abs_error\":" << controller_maximum
              << ",\"controller_cold_misses\":7"
              << ",\"controller_resume_max_abs_error\":" << resumed_maximum
              << ",\"cuda_free_before\":" << free_before
              << ",\"cuda_free_resident\":" << free_resident << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-model-residency: " << error.what() << '\n';
    return 1;
  }
}
