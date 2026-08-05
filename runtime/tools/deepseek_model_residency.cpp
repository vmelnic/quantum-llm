#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_attention.hpp"
#include "expert/runtime/cuda/deepseek_ffn.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/gather_storage.hpp"
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
    er::cuda::DeepSeekFfnBinding hash_ffn, learned_ffn;
    bind = model.bind_ffn(2U, hash_ffn);
    require(bind.ok() && hash_ffn.hash_router && hash_ffn.token_experts &&
                !hash_ffn.router_bias,
            "invalid hash FFN binding");
    bind = model.bind_ffn(3U, learned_ffn);
    require(bind.ok() && !learned_ffn.hash_router && learned_ffn.router_bias &&
                !learned_ffn.token_experts,
            "invalid learned FFN binding");
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
    auto attention_state = er::cuda::create_deepseek_attention_state(4U, 4096U);
    require(attention_state.status.ok() && attention_state.state,
            std::string(attention_state.status.message()));
    cudaEvent_t attention_start{}, attention_stop{};
    check(cudaEventCreate(&attention_start), "create attention start event");
    check(cudaEventCreate(&attention_stop), "create attention stop event");
    check(cudaEventRecord(attention_start), "record attention start");
    std::vector<float> actual_output(stream_values);
    for (std::uint32_t position = 0U; position < decode_tokens; ++position) {
      const auto attention_status = er::cuda::deepseek_attention_decode({
          &ratio_four, attention_state.state.get(),
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
    auto ffn_state = er::cuda::create_deepseek_ffn_state(2U);
    require(ffn_state.status.ok() && ffn_state.state,
            std::string(ffn_state.status.message()));
    cudaEvent_t router_start{}, router_stop{};
    check(cudaEventCreate(&router_start), "create router start event");
    check(cudaEventCreate(&router_stop), "create router stop event");
    check(cudaEventRecord(router_start), "record router start");
    for (std::uint32_t position = 0U; position < decode_tokens; ++position) {
      const auto route_status = er::cuda::deepseek_ffn_route({
          &hash_ffn, ffn_state.state.get(),
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
          &hash_ffn, ffn_state.state.get(),
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
    std::cout << "{\"ok\":true,\"dense_tensors\":" << model.dense_size()
              << ",\"typed_tensors\":" << model.typed_size()
              << ",\"source_bytes\":" << dense_source + typed_source
              << ",\"resident_bytes\":" << model.bytes()
              << ",\"staging_bytes\":" << staging
              << ",\"startup_ms\":" << startup_ms
              << ",\"ratio4_bound\":true,\"ratio128_bound\":true"
              << ",\"hash_ffn_bound\":true,\"learned_ffn_bound\":true"
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
              << ",\"cuda_free_before\":" << free_before
              << ",\"cuda_free_resident\":" << free_resident << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-model-residency: " << error.what() << '\n';
    return 1;
  }
}
