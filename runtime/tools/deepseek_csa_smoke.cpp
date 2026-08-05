#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_csa.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace er = expert::runtime;

namespace {
constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kHeadDim = 512U;
constexpr std::size_t kNormBytes = kHeadDim * sizeof(std::uint16_t);

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
std::filesystem::path relative(const std::string& text) {
  const std::filesystem::path path = text;
  require(!path.empty() && !path.is_absolute(), "source path must be relative");
  for (const auto& part : path) require(part != "..", "source path escapes root");
  return path;
}
std::vector<er::PayloadExtent> extents(const std::filesystem::path& descriptor,
                                       const std::filesystem::path& source) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1", "invalid CSA descriptor");
  std::vector<er::PayloadExtent> result;
  while (std::getline(input, line)) {
    std::vector<std::string> item;
    std::size_t start = 0U;
    do {
      const auto separator = line.find('\t', start);
      item.push_back(line.substr(start, separator - start));
      if (separator == std::string::npos) break;
      start = separator + 1U;
    } while (true);
    require(item.size() == 4U, "invalid CSA extent row");
    result.push_back({source / relative(item[3]), std::stoull(item[2]),
                      std::stoull(item[0]), std::stoull(item[1])});
  }
  require(input.eof() && (result.size() == 5U || result.size() == 10U),
          "CSA slice requires five base extents and optional indexer extents");
  return result;
}
std::string hex(const er::Sha256Digest& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : digest)
    output << std::setw(2) << std::to_integer<unsigned>(value);
  return output.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: expert-deepseek-csa-smoke <bundle> <checkpoint> "
                   "<source-sha256> <layer> <ratio>\n";
      return 64;
    }
    const auto layer = static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto ratio = static_cast<std::uint32_t>(std::stoul(argv[5]));
    require(ratio == 4U || ratio == 128U, "unsupported CSA ratio");
    const auto width = ratio == 4U ? 1024U : 512U;
    const auto matrix_bytes =
        static_cast<std::size_t>(width) * kHidden * sizeof(std::uint16_t);
    const auto ape_bytes = static_cast<std::size_t>(ratio) * width * sizeof(float);
    constexpr std::size_t sink_bytes = 64U * sizeof(float);
    constexpr std::size_t index_weight_bytes = 64U * kHidden * sizeof(std::uint16_t);
    constexpr std::size_t index_extra_bytes = index_weight_bytes +
        2U * 256U * kHidden * sizeof(std::uint16_t) +
        4U * 256U * sizeof(float) + 128U * sizeof(std::uint16_t);
    const auto source_bytes =
        2U * matrix_bytes + ape_bytes + kNormBytes + sink_bytes +
        (ratio == 4U ? index_extra_bytes : 0U);
    auto pool = std::make_shared<er::FixedBufferPool>(
        1U, source_bytes, 4096U, std::make_shared<er::CudaPinnedAllocator>());
    auto lease = pool->try_acquire(source_bytes);
    require(lease != nullptr, "could not acquire CSA staging slot");
    er::PayloadRecord record;
    record.extents = extents(std::filesystem::path(argv[1]) / "extents.tsv", argv[2]);
    record.stored_bytes = source_bytes;
    auto iocp = std::make_shared<er::WindowsIocpStorage>(2U);
    er::ExtentGatherStorage storage(iocp);
    std::promise<er::ReadResult> promise;
    auto future = promise.get_future();
    static_cast<void>(storage.read(
        {record, lease->buffer(), true},
        [&promise](er::ReadResult result) { promise.set_value(std::move(result)); }));
    const auto read = future.get();
    require(read.status.ok() && read.read_bytes == source_bytes,
            std::string(read.status.message()));
    const auto source = std::span<const std::byte>(lease->buffer().data, source_bytes);
    require(hex(er::sha256(source)) == argv[3], "CSA source hash mismatch");

    std::ifstream oracle_file(std::filesystem::path(argv[1]) / "oracle.f32",
                              std::ios::binary | std::ios::ate);
    require(static_cast<bool>(oracle_file), "missing CSA oracle");
    const auto oracle_bytes = static_cast<std::size_t>(oracle_file.tellg());
    require(oracle_bytes ==
                (static_cast<std::size_t>(ratio) * kHidden + kHeadDim + 64U) * sizeof(float),
            "invalid CSA oracle geometry");
    oracle_file.seekg(0);
    std::vector<float> oracle(oracle_bytes / sizeof(float));
    oracle_file.read(reinterpret_cast<char*>(oracle.data()),
                     static_cast<std::streamsize>(oracle_bytes));
    require(static_cast<bool>(oracle_file), "truncated CSA oracle");
    std::ifstream cache_file(std::filesystem::path(argv[1]) / "cache.bf16",
                             std::ios::binary | std::ios::ate);
    require(static_cast<bool>(cache_file) &&
                static_cast<std::size_t>(cache_file.tellg()) == kNormBytes,
            "invalid CSA cache oracle");
    cache_file.seekg(0);
    std::vector<std::uint16_t> expected_cache(kHeadDim);
    cache_file.read(reinterpret_cast<char*>(expected_cache.data()), kNormBytes);
    require(static_cast<bool>(cache_file), "truncated CSA cache oracle");
    auto read_fixture = [&](const char* name, std::size_t bytes) {
      std::ifstream file(std::filesystem::path(argv[1]) / name,
                         std::ios::binary | std::ios::ate);
      require(static_cast<bool>(file) &&
                  static_cast<std::size_t>(file.tellg()) == bytes,
              std::string("invalid CSA fixture: ") + name);
      file.seekg(0);
      std::vector<std::byte> result(bytes);
      file.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(bytes));
      require(static_cast<bool>(file), std::string("truncated CSA fixture: ") + name);
      return result;
    };
    const auto sparse_q = read_fixture("sparse-q.bf16", 64U * kNormBytes);
    const auto sparse_cache = read_fixture("sparse-cache.bf16", 6U * kNormBytes);
    const auto sparse_indices = read_fixture("sparse-indices.i32", 4U * sizeof(std::int32_t));
    const auto sparse_expected = read_fixture("sparse-output.bf16", 64U * kNormBytes);
    std::vector<std::byte> index_q_input, index_cache_input;
    std::vector<std::byte> index_q_expected, index_cache_expected;
    std::vector<std::byte> index_scores_expected, index_topk_expected;
    std::vector<std::byte> index_compressor_expected;
    if (ratio == 4U) {
      index_compressor_expected = read_fixture(
          "index-compressor-output.f32", 128U * sizeof(float));
      index_q_input = read_fixture("index-q-input.f32", 64U * 128U * sizeof(float));
      index_cache_input = read_fixture("index-cache-input.f32", 6U * 128U * sizeof(float));
      index_q_expected = read_fixture("index-q.bf16", 64U * 128U * sizeof(std::uint16_t));
      index_cache_expected = read_fixture("index-cache.bf16", 6U * 128U * sizeof(std::uint16_t));
      index_scores_expected = read_fixture("index-scores.f32", 6U * sizeof(float));
      index_topk_expected = read_fixture("index-topk.i32", 3U * sizeof(std::int32_t));
    }

    std::uint16_t *wkv = nullptr, *wgate = nullptr, *norm = nullptr;
    float *ape = nullptr, *input = nullptr, *projected_values = nullptr;
    float *projected_scores = nullptr, *pooled = nullptr, *output = nullptr;
    float *cosine = nullptr, *sine = nullptr;
    float* sink = nullptr;
    std::uint16_t* cache = nullptr;
    std::uint16_t *device_sparse_q = nullptr, *device_sparse_cache = nullptr;
    std::uint16_t* device_sparse_output = nullptr;
    std::int32_t* device_sparse_indices = nullptr;
    std::uint16_t* index_weight = nullptr;
    std::uint16_t *index_wkv = nullptr, *index_wgate = nullptr;
    std::uint16_t* index_norm = nullptr;
    float* index_ape = nullptr;
    float *index_projected_values = nullptr, *index_projected_scores = nullptr;
    float *index_pooled = nullptr, *index_compressor_output = nullptr;
    float *device_index_q_input = nullptr, *device_index_cache_input = nullptr;
    std::uint16_t *device_index_q = nullptr, *device_index_cache = nullptr;
    float *device_index_head_weights = nullptr, *device_index_scores = nullptr;
    std::int32_t* device_index_topk = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&wkv), matrix_bytes), "allocate CSA wkv");
    check(cudaMalloc(reinterpret_cast<void**>(&wgate), matrix_bytes), "allocate CSA wgate");
    check(cudaMalloc(reinterpret_cast<void**>(&ape), ape_bytes), "allocate CSA ape");
    check(cudaMalloc(reinterpret_cast<void**>(&norm), kNormBytes), "allocate CSA norm");
    check(cudaMalloc(reinterpret_cast<void**>(&input), kHidden * sizeof(float)),
          "allocate CSA input");
    check(cudaMalloc(reinterpret_cast<void**>(&projected_values), width * sizeof(float)),
          "allocate CSA projected values");
    check(cudaMalloc(reinterpret_cast<void**>(&projected_scores), width * sizeof(float)),
          "allocate CSA projected scores");
    check(cudaMalloc(reinterpret_cast<void**>(&pooled), kHeadDim * sizeof(float)),
          "allocate CSA pooled");
    check(cudaMalloc(reinterpret_cast<void**>(&output), kHeadDim * sizeof(float)),
          "allocate CSA output");
    check(cudaMalloc(reinterpret_cast<void**>(&cosine), 32U * sizeof(float)),
          "allocate CSA cosine");
    check(cudaMalloc(reinterpret_cast<void**>(&sine), 32U * sizeof(float)),
          "allocate CSA sine");
    check(cudaMalloc(reinterpret_cast<void**>(&cache), kNormBytes),
          "allocate CSA cache");
    check(cudaMalloc(reinterpret_cast<void**>(&sink), sink_bytes),
          "allocate CSA sink");
    check(cudaMalloc(reinterpret_cast<void**>(&device_sparse_q), sparse_q.size()),
          "allocate sparse query");
    check(cudaMalloc(reinterpret_cast<void**>(&device_sparse_cache), sparse_cache.size()),
          "allocate sparse cache");
    check(cudaMalloc(reinterpret_cast<void**>(&device_sparse_indices), sparse_indices.size()),
          "allocate sparse indices");
    check(cudaMalloc(reinterpret_cast<void**>(&device_sparse_output), sparse_expected.size()),
          "allocate sparse output");
    if (ratio == 4U) {
      check(cudaMalloc(reinterpret_cast<void**>(&index_weight), index_weight_bytes),
            "allocate index weights");
      check(cudaMalloc(reinterpret_cast<void**>(&device_index_q_input), index_q_input.size()),
            "allocate index query input");
      check(cudaMalloc(reinterpret_cast<void**>(&device_index_cache_input), index_cache_input.size()),
            "allocate index cache input");
      check(cudaMalloc(reinterpret_cast<void**>(&device_index_q), index_q_expected.size()),
            "allocate index query");
      check(cudaMalloc(reinterpret_cast<void**>(&device_index_cache), index_cache_expected.size()),
            "allocate index cache");
      check(cudaMalloc(reinterpret_cast<void**>(&device_index_head_weights), 64U * sizeof(float)),
            "allocate index head weights");
      check(cudaMalloc(reinterpret_cast<void**>(&device_index_scores), index_scores_expected.size()),
            "allocate index scores");
      check(cudaMalloc(reinterpret_cast<void**>(&device_index_topk), index_topk_expected.size()),
            "allocate index top-k");
      constexpr std::size_t index_matrix_bytes =
          256U * kHidden * sizeof(std::uint16_t);
      check(cudaMalloc(reinterpret_cast<void**>(&index_wkv), index_matrix_bytes),
            "allocate index compressor wkv");
      check(cudaMalloc(reinterpret_cast<void**>(&index_wgate), index_matrix_bytes),
            "allocate index compressor wgate");
      check(cudaMalloc(reinterpret_cast<void**>(&index_ape),
                       4U * 256U * sizeof(float)),
            "allocate index compressor ape");
      check(cudaMalloc(reinterpret_cast<void**>(&index_norm),
                       128U * sizeof(std::uint16_t)),
            "allocate index compressor norm");
      check(cudaMalloc(reinterpret_cast<void**>(&index_projected_values),
                       256U * sizeof(float)),
            "allocate index projected values");
      check(cudaMalloc(reinterpret_cast<void**>(&index_projected_scores),
                       256U * sizeof(float)),
            "allocate index projected scores");
      check(cudaMalloc(reinterpret_cast<void**>(&index_pooled),
                       128U * sizeof(float)),
            "allocate index pooled output");
      check(cudaMalloc(reinterpret_cast<void**>(&index_compressor_output),
                       128U * sizeof(float)),
            "allocate index compressor output");
    }
    check(cudaMemcpy(wkv, source.data(), matrix_bytes, cudaMemcpyHostToDevice),
          "copy CSA wkv");
    check(cudaMemcpy(wgate, source.data() + matrix_bytes, matrix_bytes,
                     cudaMemcpyHostToDevice), "copy CSA wgate");
    check(cudaMemcpy(ape, source.data() + 2U * matrix_bytes, ape_bytes,
                     cudaMemcpyHostToDevice), "copy CSA ape");
    check(cudaMemcpy(norm, source.data() + 2U * matrix_bytes + ape_bytes,
                     kNormBytes, cudaMemcpyHostToDevice), "copy CSA norm");
    check(cudaMemcpy(sink,
                     source.data() + 2U * matrix_bytes + ape_bytes + kNormBytes,
                     sink_bytes, cudaMemcpyHostToDevice), "copy CSA sink");
    if (ratio == 4U) {
      const auto index_offset = 2U * matrix_bytes + ape_bytes + kNormBytes + sink_bytes;
      check(cudaMemcpy(index_weight, source.data() + index_offset,
                       index_weight_bytes, cudaMemcpyHostToDevice),
            "copy index weights");
      constexpr std::size_t index_matrix_bytes =
          256U * kHidden * sizeof(std::uint16_t);
      auto offset = index_offset + index_weight_bytes;
      check(cudaMemcpy(index_wkv, source.data() + offset, index_matrix_bytes,
                       cudaMemcpyHostToDevice),
            "copy index compressor wkv");
      offset += index_matrix_bytes;
      check(cudaMemcpy(index_wgate, source.data() + offset, index_matrix_bytes,
                       cudaMemcpyHostToDevice),
            "copy index compressor wgate");
      offset += index_matrix_bytes;
      check(cudaMemcpy(index_ape, source.data() + offset,
                       4U * 256U * sizeof(float), cudaMemcpyHostToDevice),
            "copy index compressor ape");
      offset += 4U * 256U * sizeof(float);
      check(cudaMemcpy(index_norm, source.data() + offset,
                       128U * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
            "copy index compressor norm");
    }
    const auto control_offset = static_cast<std::size_t>(ratio) * kHidden + kHeadDim;
    check(cudaMemcpy(cosine, oracle.data() + control_offset, 32U * sizeof(float),
                     cudaMemcpyHostToDevice), "copy CSA cosine");
    check(cudaMemcpy(sine, oracle.data() + control_offset + 32U,
                     32U * sizeof(float), cudaMemcpyHostToDevice), "copy CSA sine");
    auto state = er::cuda::create_deepseek_compressor_state(ratio);
    require(state.status.ok() && state.state, std::string(state.status.message()));
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "create CSA start event");
    check(cudaEventCreate(&stop), "create CSA stop event");
    check(cudaEventRecord(start), "record CSA start");
    for (std::uint32_t position = 0; position < ratio; ++position) {
      check(cudaMemcpy(input, oracle.data() + static_cast<std::size_t>(position) * kHidden,
                       kHidden * sizeof(float), cudaMemcpyHostToDevice), "copy CSA input");
      auto status = er::cuda::gemv_bf16(wkv, width, kHidden, input,
                                        projected_values, nullptr);
      require(status.ok(), std::string(status.message()));
      status = er::cuda::gemv_bf16(wgate, width, kHidden, input,
                                   projected_scores, nullptr);
      require(status.ok(), std::string(status.message()));
      status = er::cuda::deepseek_compressor_decode(
          *state.state, projected_values, projected_scores, ape, norm, pooled,
          output, position, 1e-6F, nullptr);
      require(status.ok(), std::string(status.message()));
    }
    auto publish_status = er::cuda::deepseek_compressed_kv_publish(
        output, cosine, sine, cache, 0U, nullptr);
    require(publish_status.ok(), std::string(publish_status.message()));
    check(cudaEventRecord(stop), "record CSA stop");
    check(cudaEventSynchronize(stop), "synchronize CSA");
    float execution_ms = 0.0F;
    check(cudaEventElapsedTime(&execution_ms, start, stop), "measure CSA");
    std::vector<float> actual(kHeadDim);
    check(cudaMemcpy(actual.data(), output, kHeadDim * sizeof(float),
                     cudaMemcpyDeviceToHost), "copy CSA output");
    const auto* expected = oracle.data() + static_cast<std::size_t>(ratio) * kHidden;
    double squared = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const float error = std::abs(actual[index] - expected[index]);
      maximum = std::max(maximum, error);
      squared += static_cast<double>(error) * error;
    }
    require(maximum < 5e-4F, "CSA output exceeds FP32 oracle tolerance");
    std::vector<std::uint16_t> actual_cache(kHeadDim);
    check(cudaMemcpy(actual_cache.data(), cache, kNormBytes, cudaMemcpyDeviceToHost),
          "copy CSA cache");
    std::size_t cache_mismatches = 0U;
    for (std::size_t index = 0; index < actual_cache.size(); ++index)
      cache_mismatches += actual_cache[index] != expected_cache[index];
    require(cache_mismatches == 0U, "CSA BF16 cache differs from oracle");
    check(cudaMemcpy(device_sparse_q, sparse_q.data(), sparse_q.size(),
                     cudaMemcpyHostToDevice), "copy sparse query");
    check(cudaMemcpy(device_sparse_cache, sparse_cache.data(), sparse_cache.size(),
                     cudaMemcpyHostToDevice), "copy sparse cache fixture");
    check(cudaMemcpy(device_sparse_indices, sparse_indices.data(), sparse_indices.size(),
                     cudaMemcpyHostToDevice), "copy sparse indices");
    cudaEvent_t sparse_start{}, sparse_stop{};
    check(cudaEventCreate(&sparse_start), "create sparse start event");
    check(cudaEventCreate(&sparse_stop), "create sparse stop event");
    check(cudaEventRecord(sparse_start), "record sparse start");
    const auto sparse_status = er::cuda::deepseek_sparse_attention_decode(
        device_sparse_q, device_sparse_cache, device_sparse_indices, 4U, sink,
        device_sparse_output, 64U, nullptr);
    require(sparse_status.ok(), std::string(sparse_status.message()));
    check(cudaEventRecord(sparse_stop), "record sparse stop");
    check(cudaEventSynchronize(sparse_stop), "synchronize sparse attention");
    float sparse_ms = 0.0F;
    check(cudaEventElapsedTime(&sparse_ms, sparse_start, sparse_stop),
          "measure sparse attention");
    std::vector<std::byte> sparse_actual(sparse_expected.size());
    check(cudaMemcpy(sparse_actual.data(), device_sparse_output, sparse_actual.size(),
                     cudaMemcpyDeviceToHost), "copy sparse output");
    const auto* actual_words =
        reinterpret_cast<const std::uint16_t*>(sparse_actual.data());
    const auto* expected_words =
        reinterpret_cast<const std::uint16_t*>(sparse_expected.data());
    std::size_t sparse_mismatches = 0U;
    float sparse_maximum = 0.0F;
    for (std::size_t index = 0; index < sparse_actual.size() / 2U; ++index) {
      sparse_mismatches += actual_words[index] != expected_words[index];
      const auto actual_bits = static_cast<std::uint32_t>(actual_words[index]) << 16U;
      const auto expected_bits = static_cast<std::uint32_t>(expected_words[index]) << 16U;
      float actual_value = 0.0F, expected_value = 0.0F;
      std::memcpy(&actual_value, &actual_bits, sizeof(float));
      std::memcpy(&expected_value, &expected_bits, sizeof(float));
      sparse_maximum = std::max(sparse_maximum,
                                std::abs(actual_value - expected_value));
    }
    require(sparse_maximum < 2e-3F,
            "sparse attention output exceeds BF16 oracle tolerance");
    std::size_t index_prepare_mismatches = 0U;
    float index_score_maximum = 0.0F;
    float index_compressor_maximum = 0.0F;
    bool index_topk_equal = true;
    float index_ms = 0.0F;
    if (ratio == 4U) {
      check(cudaMemcpy(device_index_q_input, index_q_input.data(), index_q_input.size(),
                       cudaMemcpyHostToDevice), "copy index query input");
      check(cudaMemcpy(device_index_cache_input, index_cache_input.data(),
                       index_cache_input.size(), cudaMemcpyHostToDevice),
            "copy index cache input");
      auto index_state = er::cuda::create_deepseek_compressor_state(4U, 128U);
      require(index_state.status.ok() && index_state.state,
              std::string(index_state.status.message()));
      for (std::uint32_t position = 0; position < 4U; ++position) {
        check(cudaMemcpy(input,
                         oracle.data() + static_cast<std::size_t>(position) * kHidden,
                         kHidden * sizeof(float), cudaMemcpyHostToDevice),
              "copy index compressor input");
        auto status = er::cuda::gemv_bf16(
            index_wkv, 256U, kHidden, input, index_projected_values, nullptr);
        require(status.ok(), std::string(status.message()));
        status = er::cuda::gemv_bf16(
            index_wgate, 256U, kHidden, input, index_projected_scores, nullptr);
        require(status.ok(), std::string(status.message()));
        status = er::cuda::deepseek_compressor_decode(
            *index_state.state, index_projected_values, index_projected_scores,
            index_ape, index_norm, index_pooled, index_compressor_output,
            position, 1e-6F, nullptr);
        require(status.ok(), std::string(status.message()));
      }
      check(cudaMemcpy(device_index_cache_input, index_compressor_output,
                       128U * sizeof(float), cudaMemcpyDeviceToDevice),
            "publish index compressor output");
      auto status = er::cuda::deepseek_index_prepare(
          device_index_q_input, cosine, sine, device_index_q, 64U, nullptr);
      require(status.ok(), std::string(status.message()));
      status = er::cuda::deepseek_index_prepare(
          device_index_cache_input, cosine, sine, device_index_cache, 6U, nullptr);
      require(status.ok(), std::string(status.message()));
      status = er::cuda::gemv_bf16(index_weight, 64U, kHidden, input,
                                   device_index_head_weights, nullptr);
      require(status.ok(), std::string(status.message()));
      cudaEvent_t index_start{}, index_stop{};
      check(cudaEventCreate(&index_start), "create index start event");
      check(cudaEventCreate(&index_stop), "create index stop event");
      check(cudaEventRecord(index_start), "record index start");
      status = er::cuda::deepseek_index_topk(
          device_index_q, device_index_cache, device_index_head_weights, 6U,
          3U, device_index_scores, device_index_topk, nullptr);
      require(status.ok(), std::string(status.message()));
      check(cudaEventRecord(index_stop), "record index stop");
      check(cudaEventSynchronize(index_stop), "synchronize index selection");
      check(cudaEventElapsedTime(&index_ms, index_start, index_stop),
            "measure index selection");
      std::vector<std::byte> actual_q(index_q_expected.size());
      std::vector<std::byte> actual_index_cache(index_cache_expected.size());
      check(cudaMemcpy(actual_q.data(), device_index_q, actual_q.size(),
                       cudaMemcpyDeviceToHost), "copy index query");
      check(cudaMemcpy(actual_index_cache.data(), device_index_cache,
                       actual_index_cache.size(), cudaMemcpyDeviceToHost),
            "copy index cache");
      std::vector<float> actual_index_compressor(128U);
      check(cudaMemcpy(actual_index_compressor.data(), index_compressor_output,
                       128U * sizeof(float), cudaMemcpyDeviceToHost),
            "copy index compressor output");
      const auto* expected_index_compressor =
          reinterpret_cast<const float*>(index_compressor_expected.data());
      for (std::size_t index = 0; index < 128U; ++index)
        index_compressor_maximum = std::max(
            index_compressor_maximum,
            std::abs(actual_index_compressor[index] -
                     expected_index_compressor[index]));
      for (std::size_t index = 0; index < actual_q.size(); ++index)
        index_prepare_mismatches += actual_q[index] != index_q_expected[index];
      for (std::size_t index = 0; index < actual_index_cache.size(); ++index)
        index_prepare_mismatches +=
            actual_index_cache[index] != index_cache_expected[index];
      std::vector<float> actual_scores(6U);
      std::vector<std::int32_t> actual_topk(3U);
      check(cudaMemcpy(actual_scores.data(), device_index_scores,
                       6U * sizeof(float), cudaMemcpyDeviceToHost),
            "copy index scores");
      check(cudaMemcpy(actual_topk.data(), device_index_topk,
                       3U * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
            "copy index top-k");
      const auto* expected_scores =
          reinterpret_cast<const float*>(index_scores_expected.data());
      const auto* expected_topk =
          reinterpret_cast<const std::int32_t*>(index_topk_expected.data());
      for (std::size_t index = 0; index < 6U; ++index)
        index_score_maximum = std::max(
            index_score_maximum, std::abs(actual_scores[index] - expected_scores[index]));
      for (std::size_t index = 0; index < 3U; ++index)
        index_topk_equal &= actual_topk[index] == expected_topk[index];
      require(index_compressor_maximum < 5e-4F,
              "index compressor differs from F32 oracle");
      require(index_prepare_mismatches == 0U,
              "index preparation differs from BF16 oracle");
      require(index_score_maximum < 1e-5F && index_topk_equal,
              "index selection differs from oracle");
    }
    std::cout << "{\"ok\":true,\"layer\":" << layer
              << ",\"compress_ratio\":" << ratio
              << ",\"source_bytes\":" << source_bytes
              << ",\"state_bytes\":" << state.state->bytes()
              << ",\"group_ms\":" << execution_ms
              << ",\"output_rmse\":" << std::sqrt(squared / actual.size())
              << ",\"output_max_abs_error\":" << maximum
              << ",\"cache_bf16_mismatches\":" << cache_mismatches
              << ",\"sparse_attention_ms\":" << sparse_ms
              << ",\"sparse_bf16_mismatches\":" << sparse_mismatches
              << ",\"sparse_max_abs_error\":" << sparse_maximum
              << ",\"index_prepare_mismatches\":" << index_prepare_mismatches
              << ",\"index_compressor_max_abs_error\":"
              << index_compressor_maximum
              << ",\"index_score_max_abs_error\":" << index_score_maximum
              << ",\"index_topk_equal\":" << (index_topk_equal ? "true" : "false")
              << ",\"index_ms\":" << index_ms << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-csa-smoke: " << error.what() << '\n';
    return 1;
  }
}
