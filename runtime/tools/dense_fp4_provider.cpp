#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/active_expert_device_executor.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/cpu/fp4_host_executor.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/model_tensor_store.hpp"
#include "expert/runtime/program_executor.hpp"
#include "expert/runtime/routed_expert_runtime.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/speculative_sampling.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"
#include "expert/runtime/worker_provider.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace er = expert::runtime;
namespace ec = expert::runtime::cuda;

constexpr std::string_view kHiddenAbi = "batch.hidden.f32.cuda.v1";
constexpr std::string_view kHyperAbi = "batch.hyper-hidden.f32.cuda.v1";
constexpr std::string_view kInjectionAbi =
    "batch.hyper-injection.f32.cuda.v1";
constexpr std::string_view kRouteIndexAbi =
    "batch.route-index.u32.cuda.v1";
constexpr std::string_view kRouteWeightAbi =
    "batch.route-weight.f32.cuda.v1";
constexpr std::string_view kTokenAbi = "batch.token-id.u32.host.v1";
constexpr std::string_view kPositionAbi = "batch.position.u32.host.v1";
constexpr std::string_view kMultimodalAbi =
    "request.multimodal.fp32.host.v1";
constexpr std::string_view kActiveExpertInputAbi =
    "expert.swiglu.input.f32.host.v1";
constexpr std::string_view kActiveExpertOutputAbi =
    "expert.swiglu.output.f32.host.v1";
// The artifact geometry and selected KV format determine whether the bounded
// paged FlashAttention path can use a wider prefill microbatch. Stack-backed
// protocol arrays use the maximum; device allocations and execution use the
// per-provider value. No model-family branch selects the width.
constexpr std::uint32_t kDefaultWorkspaceRows = 512U;
constexpr std::uint32_t kMaximumWorkspaceRows = 1024U;
constexpr std::uint32_t kMaximumVisionPatches = 4096U;
constexpr std::uint32_t kMaximumGpuSamplingTopK = 64U;
constexpr std::uint32_t kMaximumExactDecodeRows = 5U;
constexpr std::uint32_t kAttentionSplitTokens = 512U;
constexpr std::uint32_t kStagedPrefillSplitTokens = 8192U;

constexpr std::uint32_t packed_attention_split_tokens(
    std::uint32_t context_tokens) noexcept {
  return context_tokens <= 65536U ? 512U
       : context_tokens <= 131077U ? 1024U
                                   : 2048U;
}

enum class TargetKvEncoding : std::uint8_t {
  artifact_native,
  fp8_e4m3_per_head,
  fp4_key_outlier1,
  q4_bfp_key_outlier1,
  q4_bfp,
  q4_per_head,
  q5_q4_bfp,
  fp16
};

enum class RecurrentCheckpointMode : std::uint8_t {
  none,
  after_first,
  every_row,
};

TargetKvEncoding target_kv_encoding(std::string_view value) {
  if (value == "artifact") return TargetKvEncoding::artifact_native;
  if (value == "fp8-e4m3-per-head")
    return TargetKvEncoding::fp8_e4m3_per_head;
  if (value == "fp4-e2m1-ue8m0-block32-key-outlier1")
    return TargetKvEncoding::fp4_key_outlier1;
  if (value == "q4-bfp16-block32-key-outlier1")
    return TargetKvEncoding::q4_bfp_key_outlier1;
  if (value == "q4-bfp16-block32")
    return TargetKvEncoding::q4_bfp;
  if (value == "q4-f16-per-head")
    return TargetKvEncoding::q4_per_head;
  if (value == "q5-q4-bfp16-block32")
    return TargetKvEncoding::q5_q4_bfp;
  if (value == "fp16") return TargetKvEncoding::fp16;
  throw std::runtime_error("unsupported target KV cache dtype");
}

constexpr std::uint64_t kDenseSnapshotMagic = 0x3150414e534d5651ULL;
constexpr std::uint32_t kDenseSnapshotVersion = 2U;
constexpr std::size_t kSnapshotChunkBytes = 4U << 20U;
constexpr std::uint32_t kMaximumSnapshotVectorItems = 4U << 20U;

std::string digest_hex(const er::Sha256Digest& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0U; index < digest.size(); ++index) {
    const auto value = std::to_integer<unsigned>(digest[index]);
    result[2U * index] = digits[value >> 4U];
    result[2U * index + 1U] = digits[value & 15U];
  }
  return result;
}

class SnapshotEncoder final {
 public:
  void u8(std::uint8_t value) {
    bytes_.push_back(static_cast<std::byte>(value));
  }
  void u32(std::uint32_t value) {
    for (unsigned shift = 0U; shift != 32U; shift += 8U)
      bytes_.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
  void i32(std::int32_t value) {
    u32(std::bit_cast<std::uint32_t>(value));
  }
  void u64(std::uint64_t value) {
    for (unsigned shift = 0U; shift != 64U; shift += 8U)
      bytes_.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
  }
  void raw(std::span<const std::byte> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  template <typename T>
  void vector(const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (values.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("snapshot vector is too large");
    u32(static_cast<std::uint32_t>(values.size()));
    raw({reinterpret_cast<const std::byte*>(values.data()),
         values.size() * sizeof(T)});
  }
  [[nodiscard]] std::span<const std::byte> view() const noexcept {
    return bytes_;
  }
  [[nodiscard]] std::vector<std::byte> finish() && {
    return std::move(bytes_);
  }

 private:
  std::vector<std::byte> bytes_;
};

class SnapshotDecoder final {
 public:
  explicit SnapshotDecoder(std::span<const std::byte> bytes) : bytes_(bytes) {}
  bool u8(std::uint8_t& value) noexcept {
    if (cursor_ == bytes_.size()) return false;
    value = std::to_integer<std::uint8_t>(bytes_[cursor_++]);
    return true;
  }
  bool u32(std::uint32_t& value) noexcept {
    if (bytes_.size() - cursor_ < 4U) return false;
    value = 0U;
    for (unsigned shift = 0U; shift != 32U; shift += 8U)
      value |= std::to_integer<std::uint32_t>(bytes_[cursor_++]) << shift;
    return true;
  }
  bool i32(std::int32_t& value) noexcept {
    std::uint32_t bits{};
    if (!u32(bits)) return false;
    value = std::bit_cast<std::int32_t>(bits);
    return true;
  }
  bool u64(std::uint64_t& value) noexcept {
    if (bytes_.size() - cursor_ < 8U) return false;
    value = 0U;
    for (unsigned shift = 0U; shift != 64U; shift += 8U)
      value |= std::to_integer<std::uint64_t>(bytes_[cursor_++]) << shift;
    return true;
  }
  bool raw(std::size_t size, std::span<const std::byte>& value) noexcept {
    if (size > bytes_.size() - cursor_) return false;
    value = bytes_.subspan(cursor_, size);
    cursor_ += size;
    return true;
  }
  template <typename T>
  bool vector(std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::uint32_t count{};
    if (!u32(count) || count > kMaximumSnapshotVectorItems ||
        static_cast<std::uint64_t>(count) * sizeof(T) >
            bytes_.size() - cursor_)
      return false;
    values.resize(count);
    if (count != 0U)
      std::memcpy(values.data(), bytes_.data() + cursor_,
                  static_cast<std::size_t>(count) * sizeof(T));
    cursor_ += static_cast<std::size_t>(count) * sizeof(T);
    return true;
  }
  [[nodiscard]] bool empty() const noexcept { return cursor_ == bytes_.size(); }

 private:
  std::span<const std::byte> bytes_;
  std::size_t cursor_{};
};

struct SnapshotChunk final {
  er::Sha256Digest digest{};
  std::uint32_t bytes{};
};

struct MtpPrediction final {
  std::uint32_t token{};
  er::SamplingDistribution distribution;
};

struct DenseSnapshotManifest final {
  std::uint64_t generation{};
  er::Sha256Digest model_hash{};
  std::uint32_t max_context{};
  std::uint32_t page_tokens{};
  std::uint64_t page_bytes{};
  std::uint64_t host_page_bytes{};
  std::uint32_t current_position{};
  std::uint32_t current_batch_first{};
  std::uint32_t current_batch_rows{};
  std::uint32_t synchronization_first{};
  std::uint32_t synchronization_rows{};
  std::uint32_t synchronization_consumed{};
  std::uint32_t mtp_length{};
  std::uint32_t synchronized_token{};
  std::uint32_t draft_token{};
  std::vector<std::uint32_t> draft_tokens;
  std::vector<std::uint32_t> draft_distribution_offsets;
  std::vector<std::uint32_t> draft_distribution_tokens;
  std::vector<double> draft_distribution_probabilities;
  std::uint32_t retention_position{};
  std::int32_t rope_delta{};
  bool draft_valid{};
  bool retention_valid{};
  bool exact_decode_enabled{};
  std::uint32_t host_kv_populated_tokens{};
  std::uint32_t logical_pages{};
  std::uint64_t page_payload_bytes{};
  std::uint64_t window_payload_bytes{};
  std::uint64_t parked_bytes{};
  std::uint64_t reported_bytes{};
  std::vector<std::uint32_t> page_indices;
  std::vector<std::uint32_t> prompt_mrope_positions;
  std::vector<float> sequence_target_hidden;
  std::vector<std::uint32_t> ple_history;
  std::vector<std::uint32_t> ple_retention_history;
  std::uint32_t host_page_count{};
  std::uint64_t logical_data_bytes{};
  std::vector<SnapshotChunk> chunks;
};

std::filesystem::path snapshot_manifest_path(
    const std::filesystem::path& root, std::uint64_t generation) {
  return root / ("manifest-" + std::to_string(generation) + ".qsc");
}

void write_atomic_file(const std::filesystem::path& destination,
                       std::span<const std::byte> bytes,
                       std::uint64_t nonce) {
  std::filesystem::create_directories(destination.parent_path());
  auto candidate = destination;
  candidate += ".partial-" + std::to_string(nonce);
  {
    std::ofstream output(candidate, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create snapshot candidate");
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) throw std::runtime_error("cannot write snapshot candidate");
  }
  std::error_code error;
  std::filesystem::rename(candidate, destination, error);
  if (error) {
    std::filesystem::remove(candidate);
    throw std::runtime_error("cannot publish snapshot candidate: " +
                             error.message());
  }
}

std::vector<std::byte> read_file(const std::filesystem::path& path,
                                 std::uint64_t maximum) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size == 0U || size > maximum ||
      size > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("snapshot file size is invalid");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open snapshot file");
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::ifstream::traits_type::eof())
    throw std::runtime_error("cannot read complete snapshot file");
  return bytes;
}

std::vector<std::byte> encode_snapshot_manifest(
    const DenseSnapshotManifest& value) {
  SnapshotEncoder output;
  output.u64(kDenseSnapshotMagic);
  output.u32(kDenseSnapshotVersion);
  output.u64(value.generation);
  output.raw(value.model_hash);
  output.u32(value.max_context);
  output.u32(value.page_tokens);
  output.u64(value.page_bytes);
  output.u64(value.host_page_bytes);
  output.u32(value.current_position);
  output.u32(value.current_batch_first);
  output.u32(value.current_batch_rows);
  output.u32(value.synchronization_first);
  output.u32(value.synchronization_rows);
  output.u32(value.synchronization_consumed);
  output.u32(value.mtp_length);
  output.u32(value.synchronized_token);
  output.u32(value.draft_token);
  output.vector(value.draft_tokens);
  output.vector(value.draft_distribution_offsets);
  output.vector(value.draft_distribution_tokens);
  output.vector(value.draft_distribution_probabilities);
  output.u32(value.retention_position);
  output.i32(value.rope_delta);
  output.u8(value.draft_valid ? 1U : 0U);
  output.u8(value.retention_valid ? 1U : 0U);
  output.u8(value.exact_decode_enabled ? 1U : 0U);
  output.u32(value.host_kv_populated_tokens);
  output.u32(value.logical_pages);
  output.u64(value.page_payload_bytes);
  output.u64(value.window_payload_bytes);
  output.u64(value.parked_bytes);
  output.u64(value.reported_bytes);
  output.vector(value.page_indices);
  output.vector(value.prompt_mrope_positions);
  output.vector(value.sequence_target_hidden);
  output.vector(value.ple_history);
  output.vector(value.ple_retention_history);
  output.u32(value.host_page_count);
  output.u64(value.logical_data_bytes);
  if (value.chunks.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("snapshot has too many chunks");
  output.u32(static_cast<std::uint32_t>(value.chunks.size()));
  for (const auto& chunk : value.chunks) {
    output.raw(chunk.digest);
    output.u32(chunk.bytes);
  }
  const auto digest = er::sha256(output.view());
  output.raw(digest);
  return std::move(output).finish();
}

DenseSnapshotManifest decode_snapshot_manifest(
    std::span<const std::byte> encoded) {
  if (encoded.size() < 32U)
    throw std::runtime_error("snapshot manifest is truncated");
  const auto authenticated = encoded.first(encoded.size() - 32U);
  const auto stored = encoded.last(32U);
  er::Sha256Digest stored_digest{};
  std::copy(stored.begin(), stored.end(), stored_digest.begin());
  if (!er::constant_time_equal(er::sha256(authenticated), stored_digest))
    throw std::runtime_error("snapshot manifest checksum mismatch");
  SnapshotDecoder input(authenticated);
  DenseSnapshotManifest value;
  std::uint64_t magic{};
  std::uint32_t version{};
  std::span<const std::byte> hash;
  std::uint8_t draft{};
  std::uint8_t retention{};
  std::uint8_t exact{};
  std::uint32_t chunks{};
  if (!input.u64(magic) || !input.u32(version) ||
      magic != kDenseSnapshotMagic || version != kDenseSnapshotVersion ||
      !input.u64(value.generation) ||
      !input.raw(value.model_hash.size(), hash) ||
      !input.u32(value.max_context) || !input.u32(value.page_tokens) ||
      !input.u64(value.page_bytes) || !input.u64(value.host_page_bytes) ||
      !input.u32(value.current_position) ||
      !input.u32(value.current_batch_first) ||
      !input.u32(value.current_batch_rows) ||
      !input.u32(value.synchronization_first) ||
      !input.u32(value.synchronization_rows) ||
      !input.u32(value.synchronization_consumed) ||
      !input.u32(value.mtp_length) ||
      !input.u32(value.synchronized_token) ||
      !input.u32(value.draft_token) ||
      !input.vector(value.draft_tokens) ||
      !input.vector(value.draft_distribution_offsets) ||
      !input.vector(value.draft_distribution_tokens) ||
      !input.vector(value.draft_distribution_probabilities) ||
      !input.u32(value.retention_position) || !input.i32(value.rope_delta) ||
      !input.u8(draft) || !input.u8(retention) || !input.u8(exact) ||
      draft > 1U || retention > 1U || exact > 1U ||
      !input.u32(value.host_kv_populated_tokens) ||
      !input.u32(value.logical_pages) ||
      !input.u64(value.page_payload_bytes) ||
      !input.u64(value.window_payload_bytes) ||
      !input.u64(value.parked_bytes) ||
      !input.u64(value.reported_bytes) ||
      !input.vector(value.page_indices) ||
      !input.vector(value.prompt_mrope_positions) ||
      !input.vector(value.sequence_target_hidden) ||
      !input.vector(value.ple_history) ||
      !input.vector(value.ple_retention_history) ||
      !input.u32(value.host_page_count) ||
      !input.u64(value.logical_data_bytes) || !input.u32(chunks) ||
      chunks > kMaximumSnapshotVectorItems)
    throw std::runtime_error("snapshot manifest layout is invalid");
  std::copy(hash.begin(), hash.end(), value.model_hash.begin());
  value.draft_valid = draft != 0U;
  value.retention_valid = retention != 0U;
  value.exact_decode_enabled = exact != 0U;
  value.chunks.resize(chunks);
  for (auto& chunk : value.chunks) {
    std::span<const std::byte> digest;
    if (!input.raw(chunk.digest.size(), digest) || !input.u32(chunk.bytes) ||
        chunk.bytes == 0U || chunk.bytes > kSnapshotChunkBytes)
      throw std::runtime_error("snapshot chunk record is invalid");
    std::copy(digest.begin(), digest.end(), chunk.digest.begin());
  }
  if (!input.empty())
    throw std::runtime_error("snapshot manifest has trailing data");
  return value;
}

class ContentAddressedSnapshotWriter final {
 public:
  ContentAddressedSnapshotWriter(std::filesystem::path root,
                                 std::uint64_t generation)
      : root_(std::move(root)), generation_(generation) {
    buffer_.reserve(kSnapshotChunkBytes);
    std::filesystem::create_directories(root_);
  }
  void append(std::span<const std::byte> bytes) {
    while (!bytes.empty()) {
      const auto count = std::min(kSnapshotChunkBytes - buffer_.size(),
                                  bytes.size());
      buffer_.insert(buffer_.end(), bytes.begin(), bytes.begin() + count);
      bytes = bytes.subspan(count);
      if (buffer_.size() == kSnapshotChunkBytes) flush();
    }
  }
  std::vector<SnapshotChunk> finish() {
    if (!buffer_.empty()) flush();
    return std::move(chunks_);
  }
  [[nodiscard]] std::uint64_t written_bytes() const noexcept {
    return written_bytes_;
  }

 private:
  void flush() {
    const auto digest = er::sha256(buffer_);
    const auto path = root_ / (digest_hex(digest) + ".blob");
    std::error_code error;
    const auto existing = std::filesystem::file_size(path, error);
    if (error || existing != buffer_.size()) {
      if (!error)
        throw std::runtime_error("snapshot blob size conflicts with digest");
      write_atomic_file(path, buffer_, generation_);
      written_bytes_ += buffer_.size();
    }
    chunks_.push_back(
        {digest, static_cast<std::uint32_t>(buffer_.size())});
    buffer_.clear();
  }
  std::filesystem::path root_;
  std::uint64_t generation_{};
  std::vector<std::byte> buffer_;
  std::vector<SnapshotChunk> chunks_;
  std::uint64_t written_bytes_{};
};

enum class Kernel : std::uint8_t {
  embedding,
  hyper_initialize,
  ple,
  hyper_read,
  hyper_inject,
  hyper_reduce,
  vision,
  full_attention,
  recurrent_attention,
  router,
  routed_moe,
  ffn,
  head,
  exact_decode,
};

enum class GpuPhase : std::uint8_t {
  embedding,
  vision,
  full_attention,
  recurrent_attention,
  router,
  routed_moe,
  ffn,
  head,
  mtp,
  count,
};

void cuda_check(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void status_check(const er::Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

std::span<const std::uint32_t> host_u32_batch(
    const er::ExecutionValue& value, std::string_view abi,
    std::string_view description) {
  if (!value.valid() || value.abi != abi || value.memory_domain != "host" ||
      value.bytes == 0U ||
      value.bytes % sizeof(std::uint32_t) != 0U ||
      value.bytes / sizeof(std::uint32_t) > kMaximumWorkspaceRows ||
      reinterpret_cast<std::uintptr_t>(value.data) %
              alignof(std::uint32_t) !=
          0U)
    throw std::runtime_error(std::string(description) + " ABI mismatch");
  return {reinterpret_cast<const std::uint32_t*>(value.data),
          static_cast<std::size_t>(value.bytes / sizeof(std::uint32_t))};
}

struct MediaImage final {
  std::uint32_t prompt_offset{};
  std::uint32_t merged_tokens{};
  std::uint32_t temporal{};
  std::uint32_t height{};
  std::uint32_t width{};
  std::uint32_t patch_offset{};
};

struct MediaPayload final {
  std::vector<std::uint32_t> positions_thw;
  std::vector<MediaImage> images;
  std::vector<float> pixels;
  std::uint32_t patch_dimension{};
  std::uint32_t patch_count{};
  std::int32_t rope_delta{};

  [[nodiscard]] bool empty() const noexcept { return images.empty(); }
};

template <typename T>
T media_field(const std::byte* data, std::size_t bytes,
              std::size_t offset) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (!data || offset > bytes || sizeof(T) > bytes - offset)
    throw std::runtime_error("multimodal packet is truncated");
  T result{};
  std::memcpy(&result, data + offset, sizeof(result));
  return result;
}

MediaPayload parse_media_payload(const er::ExecutionValue& value,
                                 std::uint32_t prompt_tokens,
                                 std::uint32_t expected_patch_dimension,
                                 std::uint32_t spatial_merge) {
  constexpr std::size_t kHeaderBytes = 40U;
  constexpr std::size_t kImageBytes = 24U;
  constexpr std::array<std::byte, 8U> kMagic{
      std::byte{'Q'}, std::byte{'L'}, std::byte{'M'}, std::byte{'M'},
      std::byte{'E'}, std::byte{'D'}, std::byte{'I'}, std::byte{'A'}};
  if (!value.valid() || value.abi != kMultimodalAbi ||
      value.memory_domain != "host" || value.bytes < kHeaderBytes ||
      value.bytes > (128U << 20U))
    throw std::runtime_error("multimodal request ABI mismatch");
  const auto* data = value.data;
  if (!std::equal(kMagic.begin(), kMagic.end(), data))
    throw std::runtime_error("multimodal request magic is invalid");
  const auto version = media_field<std::uint32_t>(data, value.bytes, 8U);
  const auto flags = media_field<std::uint32_t>(data, value.bytes, 12U);
  const auto packet_tokens =
      media_field<std::uint32_t>(data, value.bytes, 16U);
  const auto image_count =
      media_field<std::uint32_t>(data, value.bytes, 20U);
  const auto patch_dimension =
      media_field<std::uint32_t>(data, value.bytes, 24U);
  const auto patch_count =
      media_field<std::uint32_t>(data, value.bytes, 28U);
  const auto rope_delta =
      media_field<std::int32_t>(data, value.bytes, 32U);
  const auto reserved = media_field<std::uint32_t>(data, value.bytes, 36U);
  if (version != 1U || reserved != 0U)
    throw std::runtime_error("multimodal request version is unsupported");
  if (image_count == 0U) {
    if (flags != 0U || packet_tokens != 0U || patch_dimension != 0U ||
        patch_count != 0U || rope_delta != 0 || value.bytes != kHeaderBytes)
      throw std::runtime_error("empty multimodal request is malformed");
    return {};
  }
  if (flags != 3U || !prompt_tokens || packet_tokens != prompt_tokens ||
      !expected_patch_dimension || patch_dimension != expected_patch_dimension ||
      !spatial_merge || image_count > 64U || !patch_count ||
      patch_count > kMaximumVisionPatches)
    throw std::runtime_error("multimodal request geometry is invalid");
  const auto position_values = static_cast<std::uint64_t>(prompt_tokens) * 3U;
  const auto descriptor_bytes =
      static_cast<std::uint64_t>(image_count) * kImageBytes;
  const auto pixel_values = static_cast<std::uint64_t>(patch_count) *
                            patch_dimension;
  const auto expected_bytes = static_cast<std::uint64_t>(kHeaderBytes) +
      position_values * sizeof(std::uint32_t) + descriptor_bytes +
      pixel_values * sizeof(float);
  if (expected_bytes != value.bytes ||
      position_values > std::numeric_limits<std::size_t>::max() ||
      pixel_values > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("multimodal request size is inconsistent");
  MediaPayload result;
  result.patch_dimension = patch_dimension;
  result.patch_count = patch_count;
  result.rope_delta = rope_delta;
  result.positions_thw.resize(static_cast<std::size_t>(position_values));
  std::size_t cursor = kHeaderBytes;
  std::memcpy(result.positions_thw.data(), data + cursor,
              result.positions_thw.size() * sizeof(std::uint32_t));
  cursor += result.positions_thw.size() * sizeof(std::uint32_t);
  result.images.reserve(image_count);
  std::uint32_t next_patch{};
  std::uint32_t previous_prompt_end{};
  for (std::uint32_t index = 0U; index < image_count; ++index) {
    MediaImage image;
    image.prompt_offset = media_field<std::uint32_t>(data, value.bytes,
                                                     cursor + 0U);
    image.merged_tokens = media_field<std::uint32_t>(data, value.bytes,
                                                     cursor + 4U);
    image.temporal = media_field<std::uint32_t>(data, value.bytes,
                                                cursor + 8U);
    image.height = media_field<std::uint32_t>(data, value.bytes,
                                              cursor + 12U);
    image.width = media_field<std::uint32_t>(data, value.bytes,
                                             cursor + 16U);
    image.patch_offset = media_field<std::uint32_t>(data, value.bytes,
                                                    cursor + 20U);
    cursor += kImageBytes;
    const auto patches = static_cast<std::uint64_t>(image.temporal) *
                         image.height * image.width;
    const auto merge_area = static_cast<std::uint64_t>(spatial_merge) *
                            spatial_merge;
    if (!image.temporal || !image.height || !image.width ||
        image.height % spatial_merge || image.width % spatial_merge ||
        patches / merge_area != image.merged_tokens ||
        image.patch_offset != next_patch ||
        patches > patch_count - next_patch || !image.merged_tokens ||
        image.prompt_offset < previous_prompt_end ||
        image.prompt_offset > prompt_tokens ||
        image.merged_tokens > prompt_tokens - image.prompt_offset)
      throw std::runtime_error("multimodal image descriptor is invalid");
    next_patch += static_cast<std::uint32_t>(patches);
    previous_prompt_end = image.prompt_offset + image.merged_tokens;
    result.images.push_back(image);
  }
  if (next_patch != patch_count)
    throw std::runtime_error("multimodal patch coverage is incomplete");
  result.pixels.resize(static_cast<std::size_t>(pixel_values));
  std::memcpy(result.pixels.data(), data + cursor,
              result.pixels.size() * sizeof(float));
  if (std::any_of(result.pixels.begin(), result.pixels.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::runtime_error("multimodal pixels contain non-finite values");
  return result;
}

template <typename T>
T* device_allocate(std::vector<void*>& allocations, std::size_t count) {
  if (count == 0U || count > std::numeric_limits<std::size_t>::max() /
                                 sizeof(T))
    throw std::runtime_error("invalid CUDA allocation size");
  void* allocation{};
  cuda_check(cudaMalloc(&allocation, count * sizeof(T)), "cudaMalloc");
  allocations.push_back(allocation);
  return static_cast<T*>(allocation);
}

std::uint64_t checked_product(std::span<const std::uint32_t> dimensions) {
  std::uint64_t result = 1U;
  for (const auto dimension : dimensions) {
    if (!dimension || result > std::numeric_limits<std::uint64_t>::max() /
                                   dimension)
      throw std::runtime_error("invalid tensor shape product");
    result *= dimension;
  }
  return result;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               std::string_view description) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw std::runtime_error(std::string(description) + " overflows");
  return left * right;
}

std::uint32_t align32(std::uint32_t value) {
  if (value > std::numeric_limits<std::uint32_t>::max() - 31U)
    throw std::runtime_error("tensor row is too wide");
  return (value + 31U) & ~31U;
}

std::uint64_t request_parameter(const er::ProgramRequestContext& request,
                                std::string_view name) {
  const auto found = request.parameters.find(name);
  if (found == request.parameters.end())
    throw std::runtime_error("missing request parameter " +
                             std::string(name));
  return found->second;
}

std::uint32_t sample_sorted_candidates(
    std::span<const float> logits, std::span<const std::uint32_t> candidates,
    const er::ProgramRequestContext& request, std::uint32_t position) {
  const auto temperature_ppm =
      request_parameter(request, "sampling_temperature_ppm");
  const auto top_p_ppm = request_parameter(request, "sampling_top_p_ppm");
  const auto min_p_ppm = request_parameter(request, "sampling_min_p_ppm");
  const auto seed = request_parameter(request, "sampling_seed");
  try {
    const auto distribution = er::make_sampling_distribution(
        logits, candidates, static_cast<std::uint32_t>(temperature_ppm),
        static_cast<std::uint32_t>(top_p_ppm),
        static_cast<std::uint32_t>(min_p_ppm));
    return er::sample_distribution(
        distribution,
        er::counter_uniform(seed, position, 0U));
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("invalid token sampling contract: ") +
                             error.what());
  }
}

std::uint32_t sample_token(std::span<const float> logits,
                           const er::ProgramRequestContext& request,
                           std::uint32_t position) {
  const auto top_k_value = request_parameter(request, "sampling_top_k");
  if (logits.empty() || top_k_value > logits.size())
    throw std::runtime_error("invalid token sampling contract");
  const auto candidate_count = static_cast<std::size_t>(
      top_k_value == 0U ? logits.size() : top_k_value);
  std::vector<std::uint32_t> candidates(logits.size());
  std::iota(candidates.begin(), candidates.end(), 0U);
  const auto greater_logit = [&](std::uint32_t left, std::uint32_t right) {
    const auto left_value = logits[left];
    const auto right_value = logits[right];
    if (std::isnan(left_value)) return false;
    if (std::isnan(right_value)) return true;
    return left_value == right_value ? left < right : left_value > right_value;
  };
  if (candidate_count != candidates.size()) {
    std::partial_sort(candidates.begin(),
                      candidates.begin() + candidate_count,
                      candidates.end(), greater_logit);
    candidates.resize(candidate_count);
  } else {
    std::sort(candidates.begin(), candidates.end(), greater_logit);
  }
  std::vector<float> sorted_logits;
  sorted_logits.reserve(candidates.size());
  for (const auto candidate : candidates)
    sorted_logits.push_back(logits[candidate]);
  return sample_sorted_candidates(sorted_logits, candidates, request,
                                  position);
}

er::Status validate_dense_fp4_descriptor(const er::ModelDescriptor& model) {
  const auto parameter = [&](std::string_view name) -> std::uint64_t {
    const auto found = model.attributes.find(name);
    return found == model.attributes.end() ? 0U : found->second;
  };
  if (model.schema_version < 3U || model.routed_components.size() > 1U ||
      model.operation_program.empty() || !model.hidden_size ||
      !model.vocab_size || !model.max_context_tokens)
    return {er::ErrorCode::invalid_argument,
            "FP4 provider requires a schema-v3 program"};
  const auto query_heads = parameter("attention_heads");
  const auto kv_heads = parameter("kv_heads");
  const auto head_dim = parameter("head_dim");
  const auto rotary = parameter("rotary_dimension");
  const auto key_heads = parameter("linear_key_heads");
  const auto value_heads = parameter("linear_value_heads");
  const auto key_dim = parameter("linear_key_head_dim");
  const auto value_dim = parameter("linear_value_head_dim");
  const auto conv = parameter("linear_conv_kernel");
  const auto mtp = parameter("mtp_layers");
  const auto has_capability = [&](std::string_view capability) {
    return std::any_of(
        model.operation_program.begin(), model.operation_program.end(),
        [capability](const auto& operation) {
          return operation.capability == capability;
        });
  };
  const auto split_recurrent = has_capability(
      "block.recurrent-linear-attention.split-gated-delta.v1") ||
      has_capability(
          "block.recurrent-linear-attention.split-gated-delta.no-residual.v1");
  const auto qsa = has_capability(
      "block.sparse-attention.qsa.output-gated.v1");
  const auto hyper = has_capability("state.hyper-connection.initialize.v1");
  const auto ple = has_capability("embedding.ngram-ple.fp4-block32.v1") ||
                   has_capability("embedding.ngram-ple.v1");
  const auto mamba2 = has_capability("block.mamba2.ssm.v1");
  const auto standard_gqa =
      has_capability("block.full-attention.standard-gqa.v1") ||
      has_capability(
          "block.full-attention.standard-gqa-no-position.v1");
  const auto mla = has_capability(
      "block.mla.causal.latent-kv.bfloat16.v1");
  const auto normalized_gated_gqa = std::any_of(
      model.operation_program.begin(), model.operation_program.end(),
      [](const auto& operation) {
        return operation.capability ==
                   "block.full-attention.output-gated.v1" &&
               operation.abi_version >= 2U;
      });
  const auto vision = std::any_of(
      model.operation_program.begin(), model.operation_program.end(),
      [](const auto& operation) {
        return operation.capability ==
               "vision.patch-transformer-merge.fp4-block32.v1";
      });
  if ((split_recurrent && mamba2) || !query_heads || !kv_heads ||
      query_heads % kv_heads ||
      parameter("activation_bf16") > 1U ||
      parameter("dense_activation_input_bf16") > 1U ||
      (!mla && query_heads / kv_heads >
          ((standard_gqa || normalized_gated_gqa || qsa)
               ? ec::kMaximumExactFp16GroupedQueryHeads
               : 8U)) ||
      !head_dim || head_dim > 256U ||
      head_dim % 32U || !rotary || rotary > head_dim || rotary % 2U ||
      (model.exact_decode_program.has_value() && mtp != 1U) ||
      (!model.exact_decode_program.has_value() && mtp != 0U))
    return {er::ErrorCode::invalid_argument,
            "FP4 descriptor exceeds the SM86 provider geometry"};
  if (mla &&
      (!parameter("q_lora_rank") || !parameter("kv_lora_rank") ||
       !parameter("qk_nope_head_dim") ||
       !parameter("qk_rope_head_dim") || !parameter("v_head_dim") ||
       parameter("kv_lora_rank") > 256U ||
       parameter("qk_rope_head_dim") > 256U ||
       parameter("kv_lora_rank") + parameter("qk_rope_head_dim") !=
           2U * kv_heads * head_dim ||
       !parameter("rope_original_context")))
    return {er::ErrorCode::invalid_argument,
            "compressed MLA geometry exceeds the SM86 provider"};
  if (split_recurrent &&
      (!key_heads || !value_heads || value_heads % key_heads || !key_dim ||
       key_dim > 256U || !value_dim || value_dim > 256U || !conv ||
       conv > 16U || parameter("zero_centered_norm") != 1U))
    return {er::ErrorCode::invalid_argument,
            "split recurrent geometry exceeds the SM86 provider"};
  if (hyper &&
      (!parameter("hyper_connection_count") ||
       parameter("hyper_connection_count") > 16U ||
       parameter("hyper_connection_width") !=
           parameter("hyper_connection_count") * model.hidden_size ||
       !parameter("hyper_connection_lowrank")))
    return {er::ErrorCode::invalid_argument,
            "Hyper-Connection geometry exceeds the SM86 provider"};
  if (qsa &&
      (!parameter("qsa_index_heads") ||
       parameter("qsa_index_kv_heads") != 1U ||
       !parameter("qsa_index_head_dim") ||
       parameter("qsa_index_head_dim") > 256U ||
       !parameter("qsa_token_budget") ||
       !parameter("qsa_compress_ratio") ||
       parameter("qsa_token_budget") % parameter("qsa_compress_ratio")))
    return {er::ErrorCode::invalid_argument,
            "QSA geometry exceeds the SM86 provider"};
  if (ple &&
      (parameter("ple_ngram_size") < 2U ||
       parameter("ple_ngram_size") > 8U ||
       !parameter("ple_heads_per_ngram") ||
       !parameter("ple_embedding_width") ||
       parameter("ple_embedding_width") %
           ((parameter("ple_ngram_size") - 1U) *
            parameter("ple_heads_per_ngram")) ||
       parameter("ple_convolution_kernel") < 2U ||
       parameter("ple_convolution_kernel") > 16U ||
       !parameter("ple_shard_count") || !parameter("ple_rows_per_shard") ||
       parameter("ple_eos_token_id") >= model.vocab_size || !hyper))
    return {er::ErrorCode::invalid_argument,
            "PLE geometry exceeds the SM86 provider"};
  if (mamba2 &&
      (!parameter("mamba_heads") || !parameter("mamba_head_dim") ||
       !parameter("mamba_state_size") || !parameter("mamba_state_groups") ||
       !parameter("mamba_conv_size") || !parameter("mamba_conv_kernel") ||
       parameter("mamba_heads") % parameter("mamba_state_groups") ||
       parameter("mamba_heads") * parameter("mamba_head_dim") > 8192U ||
       parameter("mamba_conv_kernel") > 16U))
    return {er::ErrorCode::invalid_argument,
            "Mamba2 geometry exceeds the SM86 provider"};
  if (!model.routed_components.empty()) {
    const auto& component = model.routed_components.front();
    const auto native_nvfp4 =
        component.encoding_abi ==
        er::kExpertEncodingAbiNvfp4Block16W4A4;
    if (component.source_abi != er::kExpertSourceAbiExpertPackV1 ||
        (component.encoding_abi != er::kExpertEncodingAbiFp4Block32 &&
         !native_nvfp4) ||
        !((component.execution_capability ==
               "moe.swiglu.routed.merge-shared.v1" &&
           component.router.capability ==
               "router.linear-topk.shared-swiglu.v1") ||
          (component.execution_capability ==
               "moe.relu2.routed.merge-shared.v1" &&
           component.router.capability ==
               "router.sigmoid-bias.topk.shared-relu2.v1") ||
          (component.execution_capability ==
               "moe.swiglu.routed.merge-shared.no-residual.v1" &&
           component.router.capability ==
               "router.linear-topk.shared-swiglu.no-residual.v1") ||
          (native_nvfp4 &&
           component.execution_capability ==
               "moe.swiglu.routed.nvfp4-block16.merge-shared.v1" &&
           component.router.capability ==
               "router.softmax-topk.shared-swiglu.nvfp4-block16.v1")) ||
        component.hidden_size != model.hidden_size ||
        component.shared_experts_per_layer != 1U ||
        !component.experts_per_layer || !component.route_width ||
        component.route_width > component.experts_per_layer ||
        component.route_width > 64U || !component.intermediate_size ||
        parameter("expert_count") != component.experts_per_layer ||
        parameter("route_width") != component.route_width ||
        !parameter("shared_intermediate_size") ||
        model.exact_decode_program.has_value() || mtp != 0U)
      return {er::ErrorCode::invalid_argument,
              "routed FP4 descriptor exceeds the universal provider contract"};
  }
  if (vision) {
    const auto vision_hidden = parameter("vision_hidden_size");
    const auto vision_heads = parameter("vision_heads");
    const auto vision_merge = parameter("vision_spatial_merge_size");
    const auto positions = parameter("vision_position_embeddings");
    const auto grid = parameter("vision_grid_side");
    const auto section_zero = parameter("mrope_section_0");
    const auto section_one = parameter("mrope_section_1");
    const auto section_two = parameter("mrope_section_2");
    if (!parameter("vision_depth") || !vision_hidden || !vision_heads ||
        vision_hidden % vision_heads || vision_hidden / vision_heads > 256U ||
        (vision_hidden / vision_heads) % 4U ||
        !parameter("vision_intermediate_size") ||
        !parameter("vision_channels") ||
        !parameter("vision_patch_size") ||
        !parameter("vision_temporal_patch_size") || !vision_merge ||
        vision_merge > 4U || parameter("vision_output_size") !=
                               model.hidden_size ||
        !positions || !grid || grid * grid != positions ||
        !section_zero || !section_one || !section_two ||
        section_zero + section_one + section_two != rotary / 2U)
      return {er::ErrorCode::invalid_argument,
              "vision descriptor exceeds the SM86 provider geometry"};
  }
  return er::Status::success();
}

std::vector<er::KernelCapability> provider_capabilities() {
  const auto validator = [](const er::ModelDescriptor& model) {
    return validate_dense_fp4_descriptor(model);
  };
  return {
      {"embedding.lookup.fp4-block32.v1", 1U, 2U, validator},
      {"embedding.lookup.mxfp6-e3m2-block32.v1", 1U, 1U, validator},
      {"embedding.lookup.v1", 1U, 2U, validator},
      {"embedding.lookup.bfloat16.v1", 1U, 1U, validator},
      {"dense.activation-input.bfloat16.v1", 1U, 1U, validator},
      {"state.hyper-connection.initialize.v1", 1U, 1U, validator},
      {"embedding.ngram-ple.fp4-block32.v1", 1U, 1U, validator},
      {"embedding.ngram-ple.v1", 1U, 1U, validator},
      {"state.hyper-connection.read.v1", 1U, 1U, validator},
      {"state.hyper-connection.inject.v1", 1U, 1U, validator},
      {"state.hyper-connection.reduce.v1", 1U, 1U, validator},
      {"vision.patch-transformer-merge.fp4-block32.v1", 1U, 1U,
       validator},
      {"block.full-attention.output-gated.v1", 1U, 2U, validator},
      {"block.full-attention.standard-gqa.v1", 1U, 1U, validator},
      {"block.full-attention.standard-gqa-no-position.v1", 1U, 1U,
       validator},
      {"block.mla.causal.latent-kv.bfloat16.v1", 1U, 1U, validator},
      {"block.recurrent-linear-attention.split-gated-delta.v1", 1U, 2U,
       validator},
      {"block.recurrent-linear-attention.split-gated-delta.no-residual.v1",
       1U, 2U, validator},
      {"block.sparse-attention.qsa.output-gated.v1", 1U, 1U, validator},
      {"block.mamba2.ssm.v1", 1U, 1U, validator},
      {"router.linear-topk.shared-swiglu.v1", 1U, 1U, validator},
      {"router.linear-topk.shared-swiglu.no-residual.v1", 1U, 1U,
       validator},
      {"router.sigmoid-bias.topk.shared-relu2.v1", 1U, 1U, validator},
      {"router.softmax-topk.shared-swiglu.nvfp4-block16.v1", 1U, 1U,
       validator},
      {"moe.swiglu.routed.merge-shared.v1", 1U, 1U, validator},
      {"moe.swiglu.routed.merge-shared.no-residual.v1", 1U, 1U,
       validator},
      {"moe.relu2.routed.merge-shared.v1", 1U, 1U, validator},
      {"moe.swiglu.routed.nvfp4-block16.merge-shared.v1", 1U, 1U,
       validator},
      {"ffn.swiglu.dense.fp4-block32.v1", 1U, 2U, validator},
      {"head.rmsnorm.argmax.fp4-block32.v1", 1U, 1U, validator},
      {"head.rmsnorm.token-select.fp4-block32.v1", 1U, 2U, validator},
      {"head.rmsnorm.token-select.mxfp6-e3m2-block32.v1", 1U, 1U,
       validator},
      {"head.token-select.fp4-block32.no-norm.v1", 1U, 1U, validator},
      {"head.token-select.no-norm.v1", 1U, 1U, validator},
      {"head.rmsnorm.token-select.bfloat16.v1", 1U, 1U, validator},
      {"decode.mtp.dense-full-attention.fp4-block32.exact.v1", 1U, 1U,
       validator},
      {"decode.mtp.dense-full-attention.fp4-block32.exact.v2", 2U, 2U,
       validator},
      {"decode.mtp.dense-full-attention.fp4-mxfp6-io.exact.v3", 2U, 2U,
       validator},
  };
}

Kernel kernel_from_capability(std::string_view capability) {
  if (capability == "embedding.lookup.fp4-block32.v1" ||
      capability == "embedding.lookup.mxfp6-e3m2-block32.v1" ||
      capability == "embedding.lookup.v1" ||
      capability == "embedding.lookup.bfloat16.v1")
    return Kernel::embedding;
  if (capability == "state.hyper-connection.initialize.v1")
    return Kernel::hyper_initialize;
  if (capability == "embedding.ngram-ple.fp4-block32.v1" ||
      capability == "embedding.ngram-ple.v1")
    return Kernel::ple;
  if (capability == "state.hyper-connection.read.v1")
    return Kernel::hyper_read;
  if (capability == "state.hyper-connection.inject.v1")
    return Kernel::hyper_inject;
  if (capability == "state.hyper-connection.reduce.v1")
    return Kernel::hyper_reduce;
  if (capability == "vision.patch-transformer-merge.fp4-block32.v1")
    return Kernel::vision;
  if (capability == "block.full-attention.output-gated.v1" ||
      capability == "block.full-attention.standard-gqa.v1" ||
      capability ==
          "block.full-attention.standard-gqa-no-position.v1" ||
      capability == "block.mla.causal.latent-kv.bfloat16.v1" ||
      capability == "block.sparse-attention.qsa.output-gated.v1")
    return Kernel::full_attention;
  if (capability ==
          "block.recurrent-linear-attention.split-gated-delta.v1" ||
      capability ==
          "block.recurrent-linear-attention.split-gated-delta.no-residual.v1")
    return Kernel::recurrent_attention;
  if (capability == "block.mamba2.ssm.v1")
    return Kernel::recurrent_attention;
  if (capability == "router.linear-topk.shared-swiglu.v1" ||
      capability == "router.linear-topk.shared-swiglu.no-residual.v1" ||
      capability == "router.sigmoid-bias.topk.shared-relu2.v1")
    return Kernel::router;
  if (capability ==
      "router.softmax-topk.shared-swiglu.nvfp4-block16.v1")
    return Kernel::router;
  if (capability == "moe.swiglu.routed.merge-shared.v1" ||
      capability == "moe.swiglu.routed.merge-shared.no-residual.v1" ||
      capability == "moe.relu2.routed.merge-shared.v1")
    return Kernel::routed_moe;
  if (capability ==
      "moe.swiglu.routed.nvfp4-block16.merge-shared.v1")
    return Kernel::routed_moe;
  if (capability == "ffn.swiglu.dense.fp4-block32.v1")
    return Kernel::ffn;
  if (capability == "head.rmsnorm.argmax.fp4-block32.v1" ||
      capability == "head.rmsnorm.token-select.fp4-block32.v1" ||
      capability == "head.rmsnorm.token-select.mxfp6-e3m2-block32.v1" ||
      capability == "head.token-select.fp4-block32.no-norm.v1" ||
      capability == "head.token-select.no-norm.v1" ||
      capability == "head.rmsnorm.token-select.bfloat16.v1")
    return Kernel::head;
  if (capability ==
          "decode.mtp.dense-full-attention.fp4-block32.exact.v1" ||
      capability ==
          "decode.mtp.dense-full-attention.fp4-block32.exact.v2")
    return Kernel::exact_decode;
  if (capability ==
      "decode.mtp.dense-full-attention.fp4-mxfp6-io.exact.v3")
    return Kernel::exact_decode;
  throw std::runtime_error("unsupported dense FP4 capability");
}

struct DeviceTensor final {
  std::string name;
  std::string encoding;
  std::uint32_t quant_abi{};
  std::vector<std::uint32_t> shape;
  std::byte* allocation{};
  std::uint64_t allocation_bytes{};
  const std::uint8_t* fp4_data{};
  const std::uint8_t* fp4_scales{};
  const std::uint8_t* mxfp6_data{};
  const std::uint8_t* mxfp6_scales{};
  const std::int8_t* int8_data{};
  const float* int8_scales{};
  const std::uint16_t* bf16{};
  const float* f32{};
  float nvfp4_weight_global_scale{};
  float nvfp4_input_global_scale{};
  float* dequantized{};

  [[nodiscard]] ec::Fp4Block32Matrix matrix() const {
    if (encoding != "FP4_E2M1" ||
        quant_abi != er::kExpertQuantAbiFp4Block32 ||
        shape.size() != 2U || !fp4_data ||
        !fp4_scales)
      throw std::runtime_error(name + " is not a rank-2 FP4 matrix");
    return {fp4_data, fp4_scales, shape[0], shape[1], align32(shape[1])};
  }

  [[nodiscard]] ec::Nvfp4Block16Matrix nvfp4_matrix() const {
    if (encoding != "FP4_E2M1" ||
        quant_abi != er::kExpertRecordAbiNvfp4Block16W4A4 ||
        shape.size() != 2U || !fp4_data || !fp4_scales ||
        !(nvfp4_weight_global_scale > 0.0F) ||
        !(nvfp4_input_global_scale > 0.0F))
      throw std::runtime_error(name + " is not a native NVFP4 matrix");
    return {fp4_data, fp4_scales, nvfp4_weight_global_scale,
            nvfp4_input_global_scale, shape[0], shape[1]};
  }

  [[nodiscard]] ec::Mxfp6E3m2Block32Matrix mxfp6_matrix() const {
    if (encoding != "MXFP6_E3M2" ||
        quant_abi != er::kDenseRecordAbiMxfp6E3m2Block32 ||
        shape.size() != 2U || !mxfp6_data || !mxfp6_scales)
      throw std::runtime_error(name + " is not a rank-2 MXFP6 matrix");
    return {mxfp6_data, mxfp6_scales, shape[0], shape[1],
            align32(shape[1])};
  }

  [[nodiscard]] ec::Fp4Block32Matrix flattened_matrix() const {
    if (encoding != "FP4_E2M1" ||
        quant_abi != er::kExpertQuantAbiFp4Block32 ||
        shape.size() < 2U || !fp4_data ||
        !fp4_scales)
      throw std::runtime_error(name + " is not an FP4 tensor matrix");
    const auto columns = checked_product(std::span(shape).subspan(1U));
    if (columns > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error(name + " flattened matrix is too wide");
    return {fp4_data, fp4_scales, shape[0],
            static_cast<std::uint32_t>(columns),
            align32(static_cast<std::uint32_t>(columns))};
  }

  [[nodiscard]] ec::Int8Matrix int8_matrix() const {
    if (encoding != "I8" || quant_abi != er::kExpertQuantAbiInt8PerRow ||
        shape.size() != 2U || !int8_data || !int8_scales)
      throw std::runtime_error(name + " is not a rank-2 INT8 matrix");
    return {int8_data, int8_scales, shape[0], shape[1]};
  }

  [[nodiscard]] std::uint32_t matrix_rows() const {
    if (shape.size() != 2U ||
        (encoding != "FP4_E2M1" && encoding != "MXFP6_E3M2" &&
         encoding != "I8" &&
         encoding != "BF16"))
      throw std::runtime_error(name + " is not a quantized matrix");
    return shape[0];
  }

  [[nodiscard]] std::uint32_t matrix_columns() const {
    if (shape.size() != 2U ||
        (encoding != "FP4_E2M1" && encoding != "MXFP6_E3M2" &&
         encoding != "I8" &&
         encoding != "BF16"))
      throw std::runtime_error(name + " is not a quantized matrix");
    return shape[1];
  }
};

// Artifact-authenticated tensors whose placement contract is host.mmap.
// Large sparse lookup tables must remain mapped and be decoded only for the
// rows selected by the current token; uploading them would turn a sparse
// organ into an impossible resident allocation.
struct HostTensor final {
  std::string name;
  std::string encoding;
  std::uint32_t quant_abi{};
  std::vector<std::uint32_t> shape;
  const std::byte* data{};
  std::uint64_t data_bytes{};
  const std::byte* scales{};
  std::uint64_t scale_bytes{};
};

void decode_host_quantized_row(const HostTensor& tensor, std::uint64_t row,
                               float* output) {
  if (!output || tensor.shape.size() != 2U || row >= tensor.shape[0])
    throw std::runtime_error("invalid host quantized row lookup");
  const auto columns = tensor.shape[1];
  if (tensor.encoding == "I8") {
    if (tensor.quant_abi != er::kExpertQuantAbiInt8PerRow)
      throw std::runtime_error("invalid host INT8 row ABI");
    float scale{};
    std::memcpy(&scale, tensor.scales + row * sizeof(scale), sizeof(scale));
    const auto* values = reinterpret_cast<const std::int8_t*>(tensor.data) +
                         row * columns;
    for (std::uint32_t column = 0U; column < columns; ++column)
      output[column] = static_cast<float>(values[column]) * scale;
    return;
  }
  if (tensor.encoding != "FP4_E2M1" ||
      tensor.quant_abi != er::kExpertQuantAbiFp4Block32)
    throw std::runtime_error("unsupported host quantized row encoding");
  const auto padded = align32(columns);
  const auto* packed = reinterpret_cast<const std::uint8_t*>(tensor.data) +
                       row * padded / 2U;
  const auto* scales =
      reinterpret_cast<const std::uint8_t*>(tensor.scales) +
      row * padded / 32U;
  constexpr std::array<float, 8U> levels{
      0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  for (std::uint32_t column = 0U; column < columns; ++column) {
    const auto byte = packed[column / 2U];
    const auto code = static_cast<std::uint8_t>(
        (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
    const auto magnitude = levels[code & 0x07U];
    const auto scale = std::ldexp(
        1.0F, static_cast<int>(scales[column / 32U]) - 127);
    output[column] = (code & 0x08U) != 0U ? -magnitude * scale
                                          : magnitude * scale;
  }
}

std::int64_t host_i64(const HostTensor& tensor, std::size_t index) {
  if (tensor.encoding != "I64" ||
      index >= checked_product(tensor.shape))
    throw std::runtime_error("invalid host I64 lookup");
  std::int64_t value{};
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

// Capability-owned sparse execution organ. It consumes only the routed
// component descriptor and authenticated artifact catalog; model-family and
// architecture identifiers never participate in placement or execution.
class Fp4RoutedExperts final {
 public:
  Fp4RoutedExperts(std::shared_ptr<er::ModelArtifact> artifact,
                   std::uint64_t ram_cache_bytes,
                   std::uint64_t vram_cache_bytes,
                   std::uint32_t maximum_rows,
                   std::vector<int> active_expert_devices,
                   std::uint64_t active_expert_device_cache_bytes,
                   std::uint64_t active_expert_host_cache_bytes)
      : artifact_(std::move(artifact)) {
    if (!artifact_ || artifact_->model().routed_components.size() != 1U ||
        !ram_cache_bytes || !vram_cache_bytes || !maximum_rows)
      throw std::runtime_error("invalid routed FP4 launch contract");
    component_ = artifact_->model().routed_components.front();
    relu2_ = component_.execution_capability ==
             "moe.relu2.routed.merge-shared.v1";
    const auto swiglu =
        component_.execution_capability ==
            "moe.swiglu.routed.merge-shared.v1" ||
        component_.execution_capability ==
            "moe.swiglu.routed.merge-shared.no-residual.v1";
    native_nvfp4_ = component_.execution_capability ==
        "moe.swiglu.routed.nvfp4-block16.merge-shared.v1";
    if (component_.source_abi != er::kExpertSourceAbiExpertPackV1 ||
        (component_.encoding_abi != er::kExpertEncodingAbiFp4Block32 &&
         component_.encoding_abi !=
             er::kExpertEncodingAbiNvfp4Block16W4A4) ||
        (!relu2_ && !swiglu && !native_nvfp4_) ||
        component_.shared_experts_per_layer != 1U ||
        !component_.layer_count || !component_.experts_per_layer ||
        !component_.route_width || component_.route_width > 64U ||
        !component_.hidden_size || !component_.intermediate_size)
      throw std::runtime_error("unsupported routed FP4 component contract");
    const auto* artifact_component =
        artifact_->find_component(component_.name);
    if (!artifact_component)
      throw std::runtime_error("routed FP4 catalog is absent");
    catalog_ = &artifact_component->catalog;
    std::uint64_t maximum_record_bytes{};
    std::uint64_t total_record_bytes{};
    std::uint64_t total_device_bytes{};
    for (std::uint32_t layer = 0U; layer < component_.layer_count; ++layer) {
      for (std::uint32_t expert = 0U;
           expert < component_.experts_per_layer; ++expert) {
        const auto* record = catalog_->find(layer, expert);
        if (!record)
          throw std::runtime_error("routed FP4 catalog is incomplete");
        maximum_record_bytes =
            std::max(maximum_record_bytes, record->stored_bytes);
        if (record->stored_bytes >
            std::numeric_limits<std::uint64_t>::max() - total_record_bytes)
          throw std::runtime_error("routed FP4 catalog size overflows");
        total_record_bytes += record->stored_bytes;
        const auto record_device_bytes =
            record->device_bytes == 0U ? record->stored_bytes
                                       : record->device_bytes;
        if (record_device_bytes >
            std::numeric_limits<std::uint64_t>::max() - total_device_bytes)
          throw std::runtime_error("routed FP4 device size overflows");
        total_device_bytes += record_device_bytes;
      }
    }
    if (!maximum_record_bytes ||
        maximum_record_bytes > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("routed FP4 record geometry is invalid");
    const bool active_compatible =
        component_.encoding_abi == er::kExpertEncodingAbiFp4Block32 &&
        component_.encoding == "fp4.e2m1.ue8m0.block32" && swiglu;
    if (!active_compatible) {
      active_expert_devices.clear();
      active_expert_device_cache_bytes = 0U;
      active_expert_host_cache_bytes = 0U;
    }
    if (!active_expert_devices.empty() &&
        (!active_expert_device_cache_bytes ||
         !active_expert_host_cache_bytes ||
         active_expert_host_cache_bytes >= ram_cache_bytes))
      throw std::runtime_error(
          "secondary expert devices require bounded cache budgets");
    if (active_expert_devices.empty()) {
      active_expert_device_cache_bytes = 0U;
      active_expert_host_cache_bytes = 0U;
    }
    active_expert_host_cache_bytes_ = active_expert_host_cache_bytes;
    const auto local_ram_cache_bytes =
        ram_cache_bytes - active_expert_host_cache_bytes_;
    host_cache_capacity_bytes_ =
        std::min(local_ram_cache_bytes, total_record_bytes);
    device_pool_bytes_ = total_device_bytes;
    vram_cache_capacity_bytes_ = vram_cache_bytes;
    whole_pool_host_ = host_cache_capacity_bytes_ == total_record_bytes;

    storage_ = std::make_shared<er::WindowsIocpStorage>(4U);
    if (!active_expert_devices.empty()) {
      ec::ActiveExpertDeviceExecutorConfig executor_config;
      executor_config.model_content_hash = artifact_->model().content_hash;
      executor_config.component = component_;
      executor_config.device_ordinals = std::move(active_expert_devices);
      executor_config.device_cache_bytes_per_device =
          active_expert_device_cache_bytes;
      executor_config.device_reserve_bytes_per_device = 1ULL << 30U;
      executor_config.host_cache_bytes_total =
          active_expert_host_cache_bytes_;
      executor_config.staging_slots_per_device = std::max<std::uint32_t>(
          8U, (component_.route_width + 1U) / 2U + 2U);
      executor_config.input_abi = std::string(kActiveExpertInputAbi);
      executor_config.output_abi = std::string(kActiveExpertOutputAbi);
      executor_config.activation_clamp = 0.0F;
      executor_config.round_intermediate_to_bf16 = false;
      auto created = ec::create_active_expert_device_executor(
          std::move(executor_config), *catalog_, storage_);
      if (!created.status.ok() || !created.executor)
        throw std::runtime_error(
            created.status.ok()
                ? "secondary expert executor returned no implementation"
                : std::string(created.status.message()));
      active_device_executor_ = std::move(created.executor);
    }
    uploader_ = std::make_shared<ec::CudaExpertUploader>();
    directory_ = std::make_shared<ec::CudaExpertDirectory>(
        component_.namespace_id, component_.encoding_abi,
        component_.layer_count, component_.experts_per_layer,
        maximum_rows * component_.route_width);
    const auto staging_slots = std::max<std::size_t>(
        32U, static_cast<std::size_t>(2U * component_.route_width));
    buffers_ = std::make_shared<er::FixedBufferPool>(
        staging_slots, static_cast<std::size_t>(maximum_record_bytes),
        er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>(),
        std::min<std::size_t>(6U, staging_slots));
    const auto budget = [](std::uint64_t capacity) {
      return er::TierBudget{
          capacity, capacity,
          capacity - std::min<std::uint64_t>(capacity / 8U, 1ULL << 30U)};
    };
    const auto transient_vram =
        std::min<std::uint64_t>(vram_cache_bytes / 4U, 2ULL << 30U);
    er::ExpertCacheConfig cache_config;
    cache_config.ram = budget(host_cache_capacity_bytes_);
    cache_config.vram = budget(vram_cache_bytes);
    cache_config.retain_host_copy = true;
    // Reuse heat, not layer identity, owns the bounded cache. One global pool
    // avoids stranded quotas when route entropy differs between layers.
    cache_config.placement = {1U, component_.layer_count, 0U, 0U,
                              transient_vram, 0U};
    cache_config.trusted_immutable_source = true;
    cache_config.ram_retention_minimum_frequency = 0U;
    if (whole_pool_host_) {
#if defined(EXPERT_RUNTIME_HAS_CUDA_PINNED)
      try {
        host_arena_ = std::make_shared<er::MonotonicHostAllocator>(
            static_cast<std::size_t>(total_record_bytes),
            er::kExpertPackAlignment,
            std::make_shared<er::CudaPinnedAllocator>());
        host_bank_page_locked_ = true;
      } catch (const std::bad_alloc&) {
        host_arena_ = std::make_shared<er::MonotonicHostAllocator>(
            static_cast<std::size_t>(total_record_bytes),
            er::kExpertPackAlignment,
            std::make_shared<er::AlignedHostAllocator>());
      }
#else
      host_arena_ = std::make_shared<er::MonotonicHostAllocator>(
          static_cast<std::size_t>(total_record_bytes),
          er::kExpertPackAlignment,
          std::make_shared<er::AlignedHostAllocator>());
#endif
      cache_config.retained_host_allocator = host_arena_;
    }
    cache_ = std::make_unique<er::ExpertCache>(
        cache_config, storage_, uploader_, buffers_, directory_);
    routed_ = std::make_unique<er::RoutedExpertRuntime>(
        component_, artifact_->model().content_hash, *catalog_, *cache_);
    const auto initial_cpu_ns = static_cast<double>(maximum_record_bytes) /
        (1.5 * 1024.0 * 1024.0 * 1024.0) * 1.0e9;
    planner_ = std::make_unique<er::HybridDispatchPlanner>(
        er::HybridDispatchConfig{initial_cpu_ns, 100'000.0,
                                 8.0 * 1024.0 * 1024.0 * 1024.0,
                                 0.125, component_.experts_per_layer,
                                 256U, true, true, true, 1U});
    cpu_ = std::make_unique<er::cpu::Fp4HostExecutor>(
        er::cpu::Fp4HostExecutorConfig{
            std::clamp(std::thread::hardware_concurrency(), 1U, 64U),
            8U, 8U, 0.0F, false, true});
    const auto maximum_selections = static_cast<std::size_t>(maximum_rows) *
                                    component_.route_width;
    const auto input_bytes = static_cast<std::size_t>(maximum_rows) *
                             component_.hidden_size * sizeof(float);
    const auto output_bytes = maximum_selections * component_.hidden_size *
                              sizeof(float);
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host_input_),
                             input_bytes, cudaHostAllocPortable),
               "allocate FP4 host-lane input");
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_alternate_output_),
                   output_bytes, cudaHostAllocPortable),
               "allocate FP4 host-lane output");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_alternate_output_),
                          output_bytes),
               "allocate FP4 alternate device output");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_primary_mask_),
                          maximum_selections),
               "allocate FP4 selection mask");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_alternate_slots_),
                          maximum_selections * sizeof(std::uint32_t)),
               "allocate FP4 alternate slot map");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_execution_indices_),
                          maximum_selections * sizeof(std::uint32_t)),
               "allocate FP4 execution route");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_grouped_work_),
                          maximum_selections *
                              sizeof(ec::MoeGroupedSelectionWork)),
               "allocate grouped FP4 execution work");
    cuda_check(cudaStreamCreateWithFlags(&host_transfer_stream_,
                                         cudaStreamNonBlocking),
               "create FP4 host transfer stream");
    cuda_check(cudaStreamCreateWithFlags(&execution_stream_,
                                         cudaStreamNonBlocking),
               "create FP4 execution stream");
    cuda_check(cudaEventCreateWithFlags(&route_ready_event_,
                                        cudaEventDisableTiming),
               "create FP4 route-ready event");
    cuda_check(cudaEventCreateWithFlags(&execution_done_event_,
                                        cudaEventDisableTiming),
               "create FP4 execution-done event");
    cuda_check(cudaEventCreateWithFlags(&host_input_ready_event_,
                                        cudaEventDisableTiming),
               "create FP4 host input event");
    cuda_check(cudaEventCreateWithFlags(&host_output_ready_event_,
                                        cudaEventDisableTiming),
               "create FP4 host output event");
    cuda_check(cudaEventCreate(&gpu_started_event_),
               "create FP4 GPU start event");
    cuda_check(cudaEventCreate(&gpu_finished_event_),
               "create FP4 GPU finish event");
    if (whole_pool_host_)
      host_warm_thread_ = std::thread([this] { warm_complete_host_pool(); });
  }

  ~Fp4RoutedExperts() {
    host_warm_stop_.store(true, std::memory_order_release);
    if (host_warm_thread_.joinable()) host_warm_thread_.join();
    if (host_transfer_stream_)
      static_cast<void>(cudaStreamSynchronize(host_transfer_stream_));
    if (execution_stream_)
      static_cast<void>(cudaStreamSynchronize(execution_stream_));
    for (const auto& [key, graph] : selection_graphs_) {
      static_cast<void>(key);
      if (graph) static_cast<void>(cudaGraphExecDestroy(graph));
    }
    for (const auto& [rows, graph] : aggregate_graphs_) {
      static_cast<void>(rows);
      if (graph) static_cast<void>(cudaGraphExecDestroy(graph));
    }
    if (gpu_finished_event_)
      static_cast<void>(cudaEventDestroy(gpu_finished_event_));
    if (gpu_started_event_)
      static_cast<void>(cudaEventDestroy(gpu_started_event_));
    if (host_output_ready_event_)
      static_cast<void>(cudaEventDestroy(host_output_ready_event_));
    if (host_input_ready_event_)
      static_cast<void>(cudaEventDestroy(host_input_ready_event_));
    if (execution_done_event_)
      static_cast<void>(cudaEventDestroy(execution_done_event_));
    if (route_ready_event_)
      static_cast<void>(cudaEventDestroy(route_ready_event_));
    if (execution_stream_)
      static_cast<void>(cudaStreamDestroy(execution_stream_));
    if (host_transfer_stream_)
      static_cast<void>(cudaStreamDestroy(host_transfer_stream_));
    if (device_execution_indices_)
      static_cast<void>(cudaFree(device_execution_indices_));
    if (device_grouped_work_)
      static_cast<void>(cudaFree(device_grouped_work_));
    if (device_alternate_slots_)
      static_cast<void>(cudaFree(device_alternate_slots_));
    if (device_primary_mask_)
      static_cast<void>(cudaFree(device_primary_mask_));
    if (device_alternate_output_)
      static_cast<void>(cudaFree(device_alternate_output_));
    if (host_alternate_output_)
      static_cast<void>(cudaFreeHost(host_alternate_output_));
    if (host_input_) static_cast<void>(cudaFreeHost(host_input_));
  }

  [[nodiscard]] const er::RoutedExpertComponentDescriptor& component()
      const noexcept {
    return component_;
  }

  [[nodiscard]] std::uint64_t host_cache_capacity_bytes() const noexcept {
    return host_cache_capacity_bytes_ + active_expert_host_cache_bytes_;
  }

  [[nodiscard]] std::uint64_t device_pool_bytes() const noexcept {
    return device_pool_bytes_;
  }

  [[nodiscard]] std::uint64_t vram_cache_capacity_bytes() const noexcept {
    return vram_cache_capacity_bytes_;
  }

  void configure_vram_cache_capacity(std::uint64_t capacity) {
    if (!capacity)
      throw std::runtime_error("fitted routed VRAM cache is empty");
    const auto budget = [](std::uint64_t bytes) {
      return er::TierBudget{
          bytes, bytes,
          bytes - std::min<std::uint64_t>(bytes / 8U, 1ULL << 30U)};
    };
    const auto transient =
        std::min<std::uint64_t>(capacity / 4U, 2ULL << 30U);
    status_check(cache_->configure_vram_budget(budget(capacity), transient));
    vram_cache_capacity_bytes_ = capacity;
  }

  [[nodiscard]] bool host_bank_page_locked() const noexcept {
    return host_bank_page_locked_;
  }

  [[nodiscard]] std::uint64_t host_warm_completed() const noexcept {
    return host_warm_completed_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t host_warm_failed() const noexcept {
    return host_warm_failed_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t grouped_prefill_calls() const noexcept {
    return grouped_prefill_calls_;
  }

  [[nodiscard]] std::uint64_t grouped_prefill_work_items() const noexcept {
    return grouped_prefill_work_items_;
  }

  [[nodiscard]] std::uint64_t grouped_prefill_selections() const noexcept {
    return grouped_prefill_selections_;
  }

  void execute(std::uint32_t layer, const float* input,
               const float* routing_weights,
               const std::uint32_t* routing_indices, std::uint32_t rows,
               float* intermediate, float* selection_output,
               std::int8_t* quantized_input, float* quantized_input_scales,
               std::int8_t* quantized_intermediate,
               float* quantized_intermediate_scales,
               float* nvfp4_gate_input, float* nvfp4_up_input,
               float* nvfp4_down_input, float* output) {
    if (layer >= component_.layer_count || !input || !routing_weights ||
        !routing_indices || !rows || !intermediate || !selection_output ||
        !quantized_input || !quantized_input_scales ||
        !quantized_intermediate || !quantized_intermediate_scales ||
        (native_nvfp4_ && (!nvfp4_gate_input || !nvfp4_up_input ||
                           !nvfp4_down_input)) || !output)
      throw std::runtime_error("invalid routed FP4 execution request");
    std::lock_guard execution_lock(execution_mutex_);
    if (active_device_executor_ && rows == 1U) {
      execute_active(layer, input, routing_weights, routing_indices,
                     selection_output, output);
      return;
    }
    poll_gpu_observation();
    refresh_transfer_observation();
    const auto selection_count = rows * component_.route_width;
    cuda_check(cudaEventRecord(route_ready_event_, nullptr),
               "record routed FP4 input readiness");
    cuda_check(cudaStreamWaitEvent(execution_stream_, route_ready_event_, 0U),
               "join routed FP4 execution stream");
    cuda_check(cudaStreamWaitEvent(host_transfer_stream_, route_ready_event_,
                                   0U),
               "join routed FP4 host stream");
    auto plan = directory_->pin_or_collect_misses(
        layer, routing_indices, selection_count, execution_stream_, true);
    status_check(plan.status);

    const auto record_feedback = [&](const auto& selected) {
      std::vector<std::uint32_t> counts(component_.experts_per_layer);
      for (const auto expert : selected) {
        if (expert >= counts.size())
          throw std::runtime_error("router selected an invalid expert");
        ++counts[expert];
      }
      std::vector<er::ExpertAccess> accesses;
      accesses.reserve(selected.size());
      for (std::uint32_t expert = 0U; expert < counts.size(); ++expert) {
        if (counts[expert])
          accesses.push_back(
              {routed_->key(layer, expert), counts[expert], 0.0, 0.0});
      }
      static_cast<void>(cache_->record_accesses(accesses));
    };
    record_feedback(plan.selected_experts);

    std::uint64_t pin_id = plan.pin_id;
    std::uint64_t replacement_pin_id{};
    const auto release_pin = [&]() noexcept {
      if (pin_id) {
        static_cast<void>(
            directory_->release_pins_async(pin_id, execution_stream_));
        pin_id = 0U;
      }
      if (replacement_pin_id) {
        static_cast<void>(
            directory_->release_pins_async(replacement_pin_id,
                                           execution_stream_));
        replacement_pin_id = 0U;
      }
    };
    try {
      std::map<std::uint32_t, std::uint32_t> selection_counts;
      for (const auto expert : plan.selected_experts)
        ++selection_counts[expert];
      std::vector<er::HybridDispatchCandidate> candidates;
      candidates.reserve(selection_counts.size());
      for (const auto& [expert, count] : selection_counts) {
        const auto* record = catalog_->find(layer, expert);
        if (!record)
          throw std::runtime_error("routed FP4 record is absent");
        const auto snapshot = cache_->inspect(routed_->key(layer, expert));
        const bool resident = std::find(plan.ready_experts.begin(),
                                        plan.ready_experts.end(), expert) !=
                              plan.ready_experts.end();
        const bool host_ready = snapshot && snapshot->has_host_copy &&
            (snapshot->state == er::CacheState::ram_ready ||
             snapshot->state == er::CacheState::vram_ready);
        candidates.push_back(
            {expert, count, record->stored_bytes, resident, host_ready, true,
             snapshot ? snapshot->placement_temperature : 0U,
             snapshot ? snapshot->last_access : 0U});
      }
      const auto dispatch = planner_->plan(candidates);
      status_check(dispatch.status);

      struct HeldHost final {
        std::uint32_t expert{};
        er::HostExpertLease lease;
        std::vector<std::uint32_t> selections;
        std::vector<std::uint32_t> output_slots;
      };
      std::vector<HeldHost> host;
      std::map<std::uint32_t, bool> execute_on_host;
      for (const auto& decision : dispatch.decisions) {
        // The host executor currently implements the three-matrix SwiGLU
        // record only. ReLU2 records stay exact on the GPU paging path.
        bool selected = !relu2_ && !native_nvfp4_ &&
                        decision.executor == er::HybridExecutor::cpu_local;
        if (selected) {
          const auto* record = catalog_->find(layer, decision.expert);
          auto lease = cache_->try_acquire_host(
              routed_->key(layer, decision.expert), *record, false,
              er::ExpertRequestPriority::demand);
          if (lease)
            host.push_back({decision.expert, std::move(*lease), {}, {}});
          else
            selected = false;
        }
        execute_on_host.emplace(decision.expert, selected);
      }

      std::vector<er::AcquireHandle> handles;
      for (const auto expert : plan.missing_experts) {
        if (execute_on_host.at(expert)) continue;
        const auto* record = catalog_->find(layer, expert);
        er::ExpertAcquireOptions options;
        options.priority = er::ExpertRequestPriority::demand;
        options.record_access = false;
        handles.push_back(
            cache_->acquire(routed_->key(layer, expert), *record, options));
      }
      std::vector<er::ExpertLease> leases;
      leases.reserve(handles.size());
      for (auto& handle : handles) {
        if (handle.wait_for(std::chrono::seconds(30)) !=
            std::future_status::ready) {
          handle.cancel();
          throw std::runtime_error("routed FP4 expert acquire timed out");
        }
        auto acquired = handle.get();
        status_check(acquired.status);
        leases.push_back(std::move(acquired.lease));
      }
      refresh_transfer_observation();

      std::vector<std::uint32_t> execution_indices = plan.selected_experts;
      std::vector<std::uint8_t> primary_mask(selection_count, 1U);
      std::vector<std::uint32_t> alternate_slots(selection_count, 0U);
      std::uint32_t alternate_count{};
      std::optional<std::uint32_t> replacement;
      for (const auto& [expert, on_host] : execute_on_host)
        if (!on_host) {
          replacement = expert;
          break;
        }
      if (!host.empty() && !replacement)
        throw std::runtime_error("hybrid FP4 route has no GPU anchor");
      for (std::uint32_t selection = 0U; selection < selection_count;
           ++selection) {
        const auto expert = plan.selected_experts[selection];
        if (!execute_on_host.at(expert)) continue;
        primary_mask[selection] = 0U;
        alternate_slots[selection] = alternate_count++;
        execution_indices[selection] = *replacement;
        const auto found = std::find_if(host.begin(), host.end(),
                                        [expert](const auto& item) {
                                          return item.expert == expert;
                                        });
        if (found == host.end())
          throw std::runtime_error("host FP4 lease map is incomplete");
        found->selections.push_back(selection);
        found->output_slots.push_back(alternate_slots[selection]);
      }

      const auto grouped_prefill = rows > 8U && !native_nvfp4_;
      std::vector<ec::MoeGroupedSelectionWork> grouped_work;
      std::size_t grouped_work_count{};
      std::uint64_t grouped_selection_count{};
      if (grouped_prefill) {
        grouped_work.resize(selection_count);
        std::map<std::uint32_t, std::vector<std::uint32_t>> by_expert;
        for (std::uint32_t selection = 0U; selection < selection_count;
             ++selection) {
          if (!primary_mask[selection]) continue;
          by_expert[execution_indices[selection]].push_back(selection);
          ++grouped_selection_count;
        }
        for (const auto& [expert, selections] : by_expert) {
          for (std::size_t first = 0U; first < selections.size();
               first += ec::kMoeGroupedSelectionWidth) {
            auto& item = grouped_work.at(grouped_work_count++);
            item.expert = expert;
            item.count = static_cast<std::uint32_t>(std::min<std::size_t>(
                ec::kMoeGroupedSelectionWidth, selections.size() - first));
            std::copy_n(selections.data() + first, item.count,
                        item.selections);
          }
        }
        cuda_check(cudaMemcpyAsync(
                       device_grouped_work_, grouped_work.data(),
                       grouped_work_count * sizeof(grouped_work[0]),
                       cudaMemcpyHostToDevice, execution_stream_),
                   "stage grouped FP4 execution work");
        ++grouped_prefill_calls_;
        grouped_prefill_work_items_ += grouped_work_count;
        grouped_prefill_selections_ += grouped_selection_count;
      }

      const auto index_bytes = static_cast<std::size_t>(selection_count) *
                               sizeof(std::uint32_t);
      cuda_check(cudaMemcpyAsync(device_execution_indices_,
                                 execution_indices.data(), index_bytes,
                                 cudaMemcpyHostToDevice, execution_stream_),
                 "stage hybrid FP4 execution route");
      cuda_check(cudaMemcpyAsync(device_primary_mask_, primary_mask.data(),
                                 selection_count, cudaMemcpyHostToDevice,
                                 execution_stream_),
                 "stage hybrid FP4 selection mask");
      cuda_check(cudaMemcpyAsync(device_alternate_slots_,
                                 alternate_slots.data(), index_bytes,
                                 cudaMemcpyHostToDevice, execution_stream_),
                 "stage hybrid FP4 alternate slots");

      auto ready = directory_->pin_or_collect_misses(
          layer, device_execution_indices_, selection_count,
          execution_stream_, false);
      status_check(ready.status);
      if (!ready.missing_experts.empty() || !ready.pin_id)
        throw std::runtime_error(
            "hybrid FP4 execution route is not device-resolvable");
      replacement_pin_id = ready.pin_id;

      if (!host.empty()) {
        const auto input_bytes = static_cast<std::size_t>(rows) *
                                 component_.hidden_size * sizeof(float);
        cuda_check(cudaMemcpyAsync(host_input_, input, input_bytes,
                                   cudaMemcpyDeviceToHost,
                                   host_transfer_stream_),
                   "stage FP4 host-lane input");
        cuda_check(cudaEventRecord(host_input_ready_event_,
                                   host_transfer_stream_),
                   "record FP4 host-lane input");
      }

      // Do not overwrite timing events while their preceding asynchronous
      // sample is still in flight. Execution remains fully asynchronous; a
      // later call consumes the completed sample before arming the next one.
      const bool measure_gpu = !gpu_observation_pending_;
      if (measure_gpu)
        cuda_check(cudaEventRecord(gpu_started_event_, execution_stream_),
                   "record FP4 GPU execution start");
      launch_selection_graph(layer, input, routing_weights, intermediate,
                             selection_output, quantized_input,
                             quantized_input_scales, quantized_intermediate,
                             quantized_intermediate_scales,
                             nvfp4_gate_input, nvfp4_up_input,
                             nvfp4_down_input, rows,
                             static_cast<std::uint32_t>(grouped_work_count));
      if (measure_gpu) {
        cuda_check(cudaEventRecord(gpu_finished_event_, execution_stream_),
                   "record FP4 GPU execution finish");
        gpu_observation_pending_ = true;
        gpu_observation_selections_ = selection_count - alternate_count;
      }
      if (!host.empty()) {
        cuda_check(cudaEventSynchronize(host_input_ready_event_),
                   "wait for FP4 host-lane input");
        std::vector<er::cpu::Fp4HostWorkGroup> groups;
        groups.reserve(host.size());
        for (const auto& item : host)
          groups.push_back({item.lease.bytes(), item.lease.compact_sections(),
                            component_.hidden_size,
                            component_.intermediate_size, item.selections,
                            item.output_slots});
        const auto started = std::chrono::steady_clock::now();
        status_check(cpu_->execute(
            groups,
            std::span<const float>(host_input_,
                                   static_cast<std::size_t>(rows) *
                                       component_.hidden_size),
            rows, component_.route_width,
            std::span<float>(host_alternate_output_,
                             static_cast<std::size_t>(alternate_count) *
                                 component_.hidden_size)));
        planner_->observe_cpu(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count()),
            alternate_count);
        const auto alternate_bytes =
            static_cast<std::size_t>(alternate_count) *
            component_.hidden_size * sizeof(float);
        cuda_check(cudaMemcpyAsync(device_alternate_output_,
                                   host_alternate_output_, alternate_bytes,
                                   cudaMemcpyHostToDevice,
                                   host_transfer_stream_),
                   "stage FP4 host-lane output");
        cuda_check(cudaEventRecord(host_output_ready_event_,
                                   host_transfer_stream_),
                   "record FP4 host-lane output");
        cuda_check(cudaStreamWaitEvent(execution_stream_,
                                       host_output_ready_event_, 0U),
                   "join FP4 host-lane output");
      }

      launch_aggregate_graph(selection_output, routing_weights, output, rows);
      cuda_check(cudaEventRecord(execution_done_event_, execution_stream_),
                 "record routed FP4 completion");
      cuda_check(cudaStreamWaitEvent(nullptr, execution_done_event_, 0U),
                 "join routed FP4 completion");
      release_pin();
    } catch (...) {
      release_pin();
      throw;
    }
  }

  [[nodiscard]] er::TelemetrySnapshot telemetry() const noexcept {
    return cache_->telemetry();
  }

  [[nodiscard]] er::HybridDispatchTelemetry hybrid_telemetry() const noexcept {
    return planner_->telemetry();
  }

  [[nodiscard]] std::optional<er::ActiveExpertExecutorTelemetry>
  active_telemetry() const noexcept {
    if (!active_device_executor_) return std::nullopt;
    return active_device_executor_->telemetry();
  }

 private:
  void execute_active(std::uint32_t layer, const float* input,
                      const float* routing_weights,
                      const std::uint32_t* routing_indices,
                      float* selection_output, float* output) {
    const auto hidden_bytes =
        static_cast<std::uint64_t>(component_.hidden_size) * sizeof(float);
    const auto route_bytes = static_cast<std::size_t>(component_.route_width) *
                             sizeof(std::uint32_t);
    std::vector<std::uint32_t> selected(component_.route_width);
    cuda_check(cudaMemcpy(host_input_, input,
                          static_cast<std::size_t>(hidden_bytes),
                          cudaMemcpyDeviceToHost),
               "stage secondary expert activation");
    cuda_check(cudaMemcpy(selected.data(), routing_indices, route_bytes,
                          cudaMemcpyDeviceToHost),
               "stage secondary expert route");

    const auto request_id =
        next_active_request_id_.fetch_add(1U, std::memory_order_relaxed);
    std::vector<er::ActiveExpertExecutionHandle> handles;
    handles.reserve(component_.route_width);
    for (std::uint32_t selection = 0U;
         selection < component_.route_width; ++selection) {
      const auto expert = selected[selection];
      if (expert >= component_.experts_per_layer)
        throw std::runtime_error("secondary expert route is out of range");
      er::ActiveExpertExecutionRequest request;
      request.identity = {
          artifact_->model().content_hash, routed_->key(layer, expert),
          component_.execution_capability, component_.execution_abi,
          component_.source_abi};
      request.invocation.request_id = request_id;
      request.invocation.invocation_id = selection + 1U;
      request.invocation.selection_index = selection;
      request.invocation.route_width = component_.route_width;
      request.invocation.input = {
          std::string(kActiveExpertInputAbi), "host.pinned",
          active_lifetime_, reinterpret_cast<const std::byte*>(host_input_),
          hidden_bytes};
      request.invocation.output_abi = kActiveExpertOutputAbi;
      request.invocation.output_bytes = hidden_bytes;
      auto handle = active_device_executor_->execute(std::move(request));
      if (!handle.valid())
        throw std::runtime_error(
            "secondary expert executor rejected exact route");
      handles.push_back(std::move(handle));
    }

    std::vector<bool> completed(component_.route_width, false);
    auto remaining = component_.route_width;
    while (remaining != 0U) {
      bool progressed = false;
      for (std::uint32_t selection = 0U;
           selection < component_.route_width; ++selection) {
        if (completed[selection]) continue;
        auto result = handles[selection].poll();
        if (!result) continue;
        progressed = true;
        if (!result->status.ok())
          throw std::runtime_error(std::string(result->status.message()));
        if (result->identity.key != routed_->key(layer, selected[selection]) ||
            result->request_id != request_id ||
            result->invocation_id != selection + 1U ||
            result->selection_index != selection ||
            !result->output.valid() ||
            result->output.abi != kActiveExpertOutputAbi ||
            result->output.bytes != hidden_bytes ||
            result->evidence.weight_transport_bytes != 0U)
          throw std::runtime_error(
              "secondary expert result violates exact correlation");
        std::memcpy(host_alternate_output_ +
                        static_cast<std::size_t>(selection) *
                            component_.hidden_size,
                    result->output.data,
                    static_cast<std::size_t>(hidden_bytes));
        completed[selection] = true;
        --remaining;
      }
      if (!progressed) std::this_thread::yield();
    }

    const auto output_bytes =
        static_cast<std::size_t>(component_.route_width) * hidden_bytes;
    cuda_check(cudaMemcpyAsync(device_alternate_output_,
                               host_alternate_output_, output_bytes,
                               cudaMemcpyHostToDevice,
                               host_transfer_stream_),
               "import secondary expert outputs");
    cuda_check(cudaEventRecord(host_output_ready_event_,
                               host_transfer_stream_),
               "record secondary expert outputs");
    cuda_check(cudaStreamWaitEvent(execution_stream_, host_output_ready_event_,
                                   0U),
               "join secondary expert outputs");
    std::vector<std::uint8_t> primary_mask(component_.route_width, 0U);
    std::vector<std::uint32_t> alternate_slots(component_.route_width);
    std::iota(alternate_slots.begin(), alternate_slots.end(), 0U);
    cuda_check(cudaMemcpyAsync(device_primary_mask_, primary_mask.data(),
                               primary_mask.size(), cudaMemcpyHostToDevice,
                               execution_stream_),
               "stage secondary expert mask");
    cuda_check(cudaMemcpyAsync(device_alternate_slots_,
                               alternate_slots.data(), route_bytes,
                               cudaMemcpyHostToDevice, execution_stream_),
               "stage secondary expert slots");
    launch_aggregate_graph(selection_output, routing_weights, output, 1U);
    cuda_check(cudaEventRecord(execution_done_event_, execution_stream_),
               "record secondary expert completion");
    cuda_check(cudaStreamWaitEvent(nullptr, execution_done_event_, 0U),
               "join secondary expert completion");
  }

  void launch_selection_graph(
      std::uint32_t layer, const float* input, const float* routing_weights,
      float* intermediate, float* selection_output,
      std::int8_t* quantized_input, float* quantized_input_scales,
      std::int8_t* quantized_intermediate,
      float* quantized_intermediate_scales, float* nvfp4_gate_input,
      float* nvfp4_up_input, float* nvfp4_down_input,
      std::uint32_t rows, std::uint32_t grouped_work_items) {
    if (grouped_work_items != 0U) {
      status_check(ec::launch_moe_selection_batch({
          input, routing_weights, device_execution_indices_,
          device_primary_mask_, intermediate, selection_output,
          quantized_input, quantized_input_scales, quantized_intermediate,
          quantized_intermediate_scales, rows, component_.hidden_size,
          component_.intermediate_size, component_.route_width,
          component_.experts_per_layer, execution_stream_,
          directory_->device_entries(), layer, 0.0F, false,
          true, false, nvfp4_gate_input, nvfp4_up_input, nvfp4_down_input,
          device_grouped_work_, grouped_work_items, true}));
      return;
    }
    const auto key = (static_cast<std::uint64_t>(layer) << 32U) | rows;
    auto found = selection_graphs_.find(key);
    if (found == selection_graphs_.end()) {
      cudaGraph_t graph{};
      cudaGraphExec_t executable{};
      cuda_check(cudaStreamBeginCapture(execution_stream_,
                                        cudaStreamCaptureModeThreadLocal),
                 "begin routed FP4 selection graph");
      const auto status = ec::launch_moe_selection_batch({
          input, routing_weights, device_execution_indices_,
          device_primary_mask_, intermediate, selection_output,
          quantized_input, quantized_input_scales, quantized_intermediate,
          quantized_intermediate_scales, rows, component_.hidden_size,
          component_.intermediate_size, component_.route_width,
          component_.experts_per_layer, execution_stream_,
          directory_->device_entries(), layer, 0.0F, native_nvfp4_,
          !native_nvfp4_, native_nvfp4_, nvfp4_gate_input,
          nvfp4_up_input, nvfp4_down_input});
      const auto ended = cudaStreamEndCapture(execution_stream_, &graph);
      status_check(status);
      cuda_check(ended, "end routed FP4 selection graph");
      cuda_check(cudaGraphInstantiate(&executable, graph, 0ULL),
                 "instantiate routed FP4 selection graph");
      static_cast<void>(cudaGraphDestroy(graph));
      found = selection_graphs_.emplace(key, executable).first;
    }
    cuda_check(cudaGraphLaunch(found->second, execution_stream_),
               "launch routed FP4 selection graph");
  }

  void launch_aggregate_graph(const float* selection_output,
                              const float* routing_weights, float* output,
                              std::uint32_t rows) {
    auto found = aggregate_graphs_.find(rows);
    if (found == aggregate_graphs_.end()) {
      cudaGraph_t graph{};
      cudaGraphExec_t executable{};
      cuda_check(cudaStreamBeginCapture(execution_stream_,
                                        cudaStreamCaptureModeThreadLocal),
                 "begin routed FP4 aggregate graph");
      const auto status = ec::launch_moe_aggregate(
          {selection_output, device_alternate_output_, device_primary_mask_,
           device_alternate_slots_, routing_weights, output, 1U, rows,
           component_.hidden_size, component_.route_width,
           execution_stream_, native_nvfp4_});
      const auto ended = cudaStreamEndCapture(execution_stream_, &graph);
      status_check(status);
      cuda_check(ended, "end routed FP4 aggregate graph");
      cuda_check(cudaGraphInstantiate(&executable, graph, 0ULL),
                 "instantiate routed FP4 aggregate graph");
      static_cast<void>(cudaGraphDestroy(graph));
      found = aggregate_graphs_.emplace(rows, executable).first;
    }
    cuda_check(cudaGraphLaunch(found->second, execution_stream_),
               "launch routed FP4 aggregate graph");
  }

  void refresh_transfer_observation() noexcept {
    const auto snapshot = cache_->telemetry();
    const auto bytes = snapshot.uploaded_bytes >= observed_upload_bytes_
                           ? snapshot.uploaded_bytes - observed_upload_bytes_
                           : 0U;
    const auto elapsed = snapshot.upload_wait_ns >= observed_upload_wait_ns_
                             ? snapshot.upload_wait_ns -
                                   observed_upload_wait_ns_
                             : 0U;
    observed_upload_bytes_ = snapshot.uploaded_bytes;
    observed_upload_wait_ns_ = snapshot.upload_wait_ns;
    if (bytes && elapsed) planner_->observe_h2d(elapsed, bytes);
  }

  void poll_gpu_observation() noexcept {
    if (!gpu_observation_pending_) return;
    const auto query = cudaEventQuery(gpu_finished_event_);
    if (query == cudaErrorNotReady) return;
    if (query != cudaSuccess) {
      gpu_observation_pending_ = false;
      return;
    }
    float milliseconds = 0.0F;
    if (cudaEventElapsedTime(&milliseconds, gpu_started_event_,
                             gpu_finished_event_) == cudaSuccess &&
        milliseconds > 0.0F && gpu_observation_selections_) {
      planner_->observe_gpu(
          static_cast<std::uint64_t>(milliseconds * 1.0e6F),
          gpu_observation_selections_);
    }
    gpu_observation_pending_ = false;
  }

  void warm_complete_host_pool() noexcept {
    constexpr std::size_t kWindow = 4U;
    std::vector<er::HostPreloadHandle> pending;
    pending.reserve(kWindow);
    const auto drain_one = [&]() {
      if (pending.empty()) return;
      auto handle = std::move(pending.front());
      pending.erase(pending.begin());
      while (!host_warm_stop_.load(std::memory_order_acquire) &&
             handle.wait_for(std::chrono::milliseconds(20)) !=
                 std::future_status::ready) {
      }
      if (host_warm_stop_.load(std::memory_order_acquire)) {
        handle.cancel();
        return;
      }
      auto result = handle.get();
      if (result.status.ok() && result.retained)
        host_warm_completed_.fetch_add(1U, std::memory_order_relaxed);
      else
        host_warm_failed_.fetch_add(1U, std::memory_order_relaxed);
    };
    try {
      for (std::uint32_t layer = 0U; layer < component_.layer_count; ++layer) {
        for (std::uint32_t expert = 0U;
             expert < component_.experts_per_layer; ++expert) {
          if (host_warm_stop_.load(std::memory_order_acquire)) break;
          const auto* record = catalog_->find(layer, expert);
          if (!record) {
            host_warm_failed_.fetch_add(1U, std::memory_order_relaxed);
            continue;
          }
          pending.push_back(cache_->preload_host(
              routed_->key(layer, expert), *record,
              {er::ExpertRequestPriority::warm, false, false}));
          if (pending.size() == kWindow) drain_one();
        }
        if (host_warm_stop_.load(std::memory_order_acquire)) break;
      }
      while (!pending.empty() &&
             !host_warm_stop_.load(std::memory_order_acquire))
        drain_one();
    } catch (...) {
      host_warm_failed_.fetch_add(1U, std::memory_order_relaxed);
    }
    for (auto& handle : pending) handle.cancel();
  }

  std::shared_ptr<er::ModelArtifact> artifact_;
  er::RoutedExpertComponentDescriptor component_;
  const er::ExpertCatalog* catalog_{};
  std::shared_ptr<er::WindowsIocpStorage> storage_;
  std::shared_ptr<ec::CudaExpertUploader> uploader_;
  std::shared_ptr<ec::CudaExpertDirectory> directory_;
  std::shared_ptr<er::FixedBufferPool> buffers_;
  std::unique_ptr<er::ExpertCache> cache_;
  std::unique_ptr<er::RoutedExpertRuntime> routed_;
  std::unique_ptr<er::HybridDispatchPlanner> planner_;
  std::unique_ptr<er::cpu::Fp4HostExecutor> cpu_;
  std::shared_ptr<er::IActiveExpertExecutor> active_device_executor_;
  std::shared_ptr<const void> active_lifetime_{
      this, [](const void*) noexcept {}};
  std::atomic<std::uint64_t> next_active_request_id_{1U};
  std::shared_ptr<er::MonotonicHostAllocator> host_arena_;
  std::mutex execution_mutex_;
  cudaStream_t host_transfer_stream_{};
  cudaStream_t execution_stream_{};
  cudaEvent_t route_ready_event_{};
  cudaEvent_t execution_done_event_{};
  cudaEvent_t host_input_ready_event_{};
  cudaEvent_t host_output_ready_event_{};
  cudaEvent_t gpu_started_event_{};
  cudaEvent_t gpu_finished_event_{};
  float* host_input_{};
  float* host_alternate_output_{};
  float* device_alternate_output_{};
  std::uint8_t* device_primary_mask_{};
  std::uint32_t* device_alternate_slots_{};
  std::uint32_t* device_execution_indices_{};
  ec::MoeGroupedSelectionWork* device_grouped_work_{};
  std::uint64_t grouped_prefill_calls_{};
  std::uint64_t grouped_prefill_work_items_{};
  std::uint64_t grouped_prefill_selections_{};
  std::uint64_t observed_upload_bytes_{};
  std::uint64_t observed_upload_wait_ns_{};
  std::uint32_t gpu_observation_selections_{};
  bool gpu_observation_pending_{};
  std::map<std::uint64_t, cudaGraphExec_t> selection_graphs_;
  std::map<std::uint32_t, cudaGraphExec_t> aggregate_graphs_;
  std::thread host_warm_thread_;
  std::atomic<bool> host_warm_stop_{false};
  std::atomic<std::uint64_t> host_warm_completed_{0U};
  std::atomic<std::uint64_t> host_warm_failed_{0U};
  bool whole_pool_host_{};
  bool native_nvfp4_{};
  bool host_bank_page_locked_{};
  bool relu2_{};
  std::uint64_t host_cache_capacity_bytes_{};
  std::uint64_t active_expert_host_cache_bytes_{};
  std::uint64_t device_pool_bytes_{};
  std::uint64_t vram_cache_capacity_bytes_{};
};

struct PreparedOperation final : er::IPreparedOperation {
  Kernel kernel{};
  std::string capability;
  std::uint32_t abi_version{};
  std::uint32_t logical_operation{};
  std::uint32_t logical_layer{};
  std::uint32_t component_layer{};
  std::uint32_t full_attention_slot{};
  std::uint32_t kv_layer_slot{};
  std::uint32_t attention_window_tokens{};
  std::uint32_t recurrent_slot{};
  std::uint32_t qsa_index_slot{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t ple_slot{std::numeric_limits<std::uint32_t>::max()};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
  std::map<std::string, const DeviceTensor*, std::less<>> tensors;
  std::map<std::string, const HostTensor*, std::less<>> host_tensors;
  std::map<std::string, std::size_t, std::less<>> input_indices;
  std::vector<std::pair<std::string, std::string>> outputs;
};

class DenseFp4Provider final : public er::IOperationProvider {
 public:
  DenseFp4Provider(std::shared_ptr<er::ModelArtifact> artifact,
                   std::shared_ptr<er::MappedModelTensorStore> tensor_store,
                   std::uint32_t max_context, std::uint32_t capacity,
                   std::uint64_t ram_cache_bytes,
                   std::uint64_t vram_cache_bytes,
                   std::uint64_t kv_cache_bytes,
                   std::uint32_t kv_page_tokens,
                   std::string_view kv_cache_dtype,
                   std::string_view placement_profile,
                   std::string_view routed_vram_policy,
                   bool profile_gpu_phases,
                   std::vector<int> active_expert_devices,
                   std::uint64_t active_expert_device_cache_bytes,
                   std::uint64_t active_expert_host_cache_bytes)
      : artifact_(std::move(artifact)),
        descriptor_(artifact_ ? artifact_->model() : er::ModelDescriptor{}),
        tensor_store_(std::move(tensor_store)),
        max_context_(max_context), capacity_(capacity),
        ram_cache_bytes_(ram_cache_bytes), vram_cache_bytes_(vram_cache_bytes),
        kv_cache_bytes_(kv_cache_bytes), kv_page_tokens_(kv_page_tokens),
        target_kv_encoding_(target_kv_encoding(kv_cache_dtype)),
        capacity_placement_(placement_profile != "latency"),
        fit_routed_vram_(routed_vram_policy == "fit"),
        profile_gpu_phases_(profile_gpu_phases),
        device_lifetime_(std::make_shared<std::uint8_t>(0U)) {
    status_check(validate_dense_fp4_descriptor(descriptor_));
    if (!artifact_ || !tensor_store_ || !tensor_store_->valid() || !capacity_ ||
        !max_context_ || max_context_ > descriptor_.max_context_tokens ||
        !kv_cache_bytes_ || !kv_page_tokens_ ||
        (routed_vram_policy != "fixed" && routed_vram_policy != "fit") ||
        (fit_routed_vram_ && descriptor_.routed_components.empty()))
      throw std::runtime_error("invalid dense FP4 provider launch contract");
    hidden_size_ = descriptor_.hidden_size;
    vocabulary_size_ = descriptor_.vocab_size;
    if (const auto* trace_path =
            std::getenv("QUANTUM_LLM_SAMPLING_TRACE_FILE");
        trace_path != nullptr && *trace_path != '\0') {
      sampling_trace_stream_.open(trace_path, std::ios::out | std::ios::app);
      if (!sampling_trace_stream_)
        throw std::runtime_error("cannot open diagnostic sampling trace");
      sampling_trace_stream_ << std::setprecision(17);
    }
    const auto has_capability = [&](std::string_view capability) {
      return std::any_of(
          descriptor_.operation_program.begin(),
          descriptor_.operation_program.end(),
          [capability](const auto& operation) {
            return operation.capability == capability;
          });
    };
    split_recurrent_enabled_ =
        has_capability(
            "block.recurrent-linear-attention.split-gated-delta.v1") ||
        has_capability(
            "block.recurrent-linear-attention.split-gated-delta.no-residual.v1");
    hyper_enabled_ =
        has_capability("state.hyper-connection.initialize.v1");
    qsa_enabled_ =
        has_capability("block.sparse-attention.qsa.output-gated.v1");
    ple_enabled_ =
        has_capability("embedding.ngram-ple.fp4-block32.v1") ||
        has_capability("embedding.ngram-ple.v1");
    mamba2_enabled_ = has_capability("block.mamba2.ssm.v1");
    standard_attention_enabled_ =
        has_capability("block.full-attention.standard-gqa.v1") ||
        has_capability(
            "block.full-attention.standard-gqa-no-position.v1");
    mla_enabled_ = has_capability(
        "block.mla.causal.latent-kv.bfloat16.v1");
    relu2_router_enabled_ = has_capability(
        "router.sigmoid-bias.topk.shared-relu2.v1");
    const auto zero_centered =
        descriptor_.attributes.find("zero_centered_norm");
    zero_centered_norm_ = zero_centered != descriptor_.attributes.end() &&
                          zero_centered->second == 1U;
    const auto activation_bf16 =
        descriptor_.attributes.find("activation_bf16");
    activation_bf16_ =
        activation_bf16 != descriptor_.attributes.end() &&
        activation_bf16->second == 1U;
    const auto dense_activation_input_bf16 =
        descriptor_.attributes.find("dense_activation_input_bf16");
    dense_activation_input_bf16_ =
        dense_activation_input_bf16 != descriptor_.attributes.end() &&
        dense_activation_input_bf16->second == 1U;
    query_heads_ = attribute_u32("attention_heads");
    kv_heads_ = attribute_u32("kv_heads");
    head_dim_ = attribute_u32("head_dim");
    rotary_dimension_ = attribute_u32("rotary_dimension");
    if (mla_enabled_) {
      q_lora_rank_ = attribute_u32("q_lora_rank");
      kv_lora_rank_ = attribute_u32("kv_lora_rank");
      qk_nope_head_dim_ = attribute_u32("qk_nope_head_dim");
      qk_rope_head_dim_ = attribute_u32("qk_rope_head_dim");
      v_head_dim_ = attribute_u32("v_head_dim");
      rope_factor_ = attribute_f32("rope_factor_f32_bits");
      rope_beta_fast_ = attribute_f32("rope_beta_fast_f32_bits");
      rope_beta_slow_ = attribute_f32("rope_beta_slow_f32_bits");
      rope_original_context_ = attribute_u32("rope_original_context");
      llama4_scaling_beta_ =
          attribute_f32("llama4_scaling_beta_f32_bits");
      mla_attention_scale_ = attribute_f32("attention_scale_f32_bits");
    }
    if (hyper_enabled_) {
      hyper_count_ = attribute_u32("hyper_connection_count");
      hyper_width_ = attribute_u32("hyper_connection_width");
      hyper_lowrank_ = attribute_u32("hyper_connection_lowrank");
    }
    if (qsa_enabled_) {
      qsa_index_heads_ = attribute_u32("qsa_index_heads");
      qsa_index_head_dim_ = attribute_u32("qsa_index_head_dim");
      qsa_token_budget_ = attribute_u32("qsa_token_budget");
      qsa_compress_ratio_ = attribute_u32("qsa_compress_ratio");
    }
    if (ple_enabled_) {
      ple_ngram_size_ = attribute_u32("ple_ngram_size");
      ple_heads_per_ngram_ = attribute_u32("ple_heads_per_ngram");
      ple_embedding_width_ = attribute_u32("ple_embedding_width");
      ple_convolution_kernel_ = attribute_u32("ple_convolution_kernel");
      ple_shard_count_ = attribute_u32("ple_shard_count");
      ple_rows_per_shard_ = attribute_u32("ple_rows_per_shard");
      ple_eos_token_id_ = attribute_u32("ple_eos_token_id");
      ple_head_count_ =
          (ple_ngram_size_ - 1U) * ple_heads_per_ngram_;
      ple_head_width_ = ple_embedding_width_ / ple_head_count_;
      ple_conv_state_values_ = static_cast<std::size_t>(hyper_width_) *
          (ple_convolution_kernel_ - 1U) * ple_ngram_size_;
    }
    if (split_recurrent_enabled_) {
      key_heads_ = attribute_u32("linear_key_heads");
      value_heads_ = attribute_u32("linear_value_heads");
      key_head_dim_ = attribute_u32("linear_key_head_dim");
      value_head_dim_ = attribute_u32("linear_value_head_dim");
      conv_kernel_ = attribute_u32("linear_conv_kernel");
      recurrent_conv_values_ = static_cast<std::size_t>(
          2U * key_heads_ * key_head_dim_ + value_heads_ * value_head_dim_) *
          conv_kernel_;
      recurrent_matrix_values_ = static_cast<std::size_t>(value_heads_) *
                                 key_head_dim_ * value_head_dim_;
    }
    if (mamba2_enabled_) {
      mamba_heads_ = attribute_u32("mamba_heads");
      mamba_head_dim_ = attribute_u32("mamba_head_dim");
      mamba_state_size_ = attribute_u32("mamba_state_size");
      mamba_state_groups_ = attribute_u32("mamba_state_groups");
      mamba_conv_size_ = attribute_u32("mamba_conv_size");
      mamba_conv_kernel_ = attribute_u32("mamba_conv_kernel");
      recurrent_conv_values_ = static_cast<std::size_t>(mamba_conv_size_) *
                               mamba_conv_kernel_;
      recurrent_matrix_values_ = static_cast<std::size_t>(mamba_heads_) *
                                 mamba_head_dim_ * mamba_state_size_;
    }
    epsilon_ = attribute_f32("norm_epsilon_f32_bits");
    rope_theta_ = attribute_f32("rope_theta_f32_bits");
    mtp_layers_ = attribute_u32("mtp_layers");
    if (descriptor_.exact_decode_program) {
      const auto& exact = *descriptor_.exact_decode_program;
      exact_decode_abi_ = exact.abi_version;
      if (exact.abi_version == 1U) {
        draft_depth_ = 1U;
        draft_vocabulary_size_ = vocabulary_size_;
      } else if (exact.abi_version == 2U) {
        draft_depth_ = parameter_u32(exact.parameters, "draft_depth");
        draft_vocabulary_size_ =
            parameter_u32(exact.parameters, "draft_vocabulary_size");
        mtp_q8_kv_ =
            parameter_u32(exact.parameters, "mtp_kv_encoding") == 1U;
        if (parameter_u32(exact.parameters, "source_mtp_layers") != 1U ||
            draft_depth_ < 3U || draft_depth_ > 4U ||
            draft_vocabulary_size_ == 0U ||
            draft_vocabulary_size_ > vocabulary_size_ || !mtp_q8_kv_)
          throw std::runtime_error(
              "exact-decode ABI 2 parameters are unsupported");
      } else {
        throw std::runtime_error("unsupported exact-decode ABI");
      }
    }
    if (!descriptor_.routed_components.empty()) {
      const auto& component = descriptor_.routed_components.front();
      expert_count_ = component.experts_per_layer;
      route_width_ = component.route_width;
      expert_width_ = component.intermediate_size;
      shared_intermediate_size_ = attribute_u32("shared_intermediate_size");
    }
    vision_enabled_ = std::any_of(
        descriptor_.operation_program.begin(),
        descriptor_.operation_program.end(), [](const auto& operation) {
          return operation.capability ==
                 "vision.patch-transformer-merge.fp4-block32.v1";
        });
    if (vision_enabled_) {
      mrope_sections_ = {attribute_u32("mrope_section_0"),
                         attribute_u32("mrope_section_1"),
                         attribute_u32("mrope_section_2")};
      vision_depth_ = attribute_u32("vision_depth");
      vision_hidden_size_ = attribute_u32("vision_hidden_size");
      vision_intermediate_size_ =
          attribute_u32("vision_intermediate_size");
      vision_heads_ = attribute_u32("vision_heads");
      vision_head_dim_ = vision_hidden_size_ / vision_heads_;
      vision_position_embeddings_ =
          attribute_u32("vision_position_embeddings");
      vision_grid_side_ = attribute_u32("vision_grid_side");
      vision_channels_ = attribute_u32("vision_channels");
      vision_patch_size_ = attribute_u32("vision_patch_size");
      vision_temporal_patch_size_ =
          attribute_u32("vision_temporal_patch_size");
      vision_spatial_merge_size_ =
          attribute_u32("vision_spatial_merge_size");
      vision_output_size_ = attribute_u32("vision_output_size");
      vision_epsilon_ = attribute_f32("vision_norm_epsilon_f32_bits");
      vision_rope_theta_ = attribute_f32("vision_rope_theta_f32_bits");
      const auto patch_dimension = static_cast<std::uint64_t>(vision_channels_) *
          vision_temporal_patch_size_ * vision_patch_size_ *
          vision_patch_size_;
      if (patch_dimension > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("vision patch dimension overflows");
      vision_patch_dimension_ = static_cast<std::uint32_t>(patch_dimension);
      vision_merged_width_ = vision_hidden_size_ *
          vision_spatial_merge_size_ * vision_spatial_merge_size_;
    }
    full_attention_slots_.assign(descriptor_.layer_program.size(), kNoSlot);
    global_kv_slots_.assign(descriptor_.layer_program.size(), kNoSlot);
    window_kv_slots_.assign(descriptor_.layer_program.size(), kNoSlot);
    recurrent_slots_.assign(descriptor_.layer_program.size(), kNoSlot);
    qsa_index_slots_.assign(descriptor_.layer_program.size(), kNoSlot);
    ple_slots_.assign(descriptor_.layer_program.size(), kNoSlot);
    for (const auto& operation : descriptor_.operation_program) {
      if (operation.logical_layer == er::kModelLevelOperationLayer) continue;
      if (operation.logical_layer >= full_attention_slots_.size())
        throw std::runtime_error("operation layer exceeds the artifact topology");
      if (operation.capability == "block.full-attention.output-gated.v1" ||
          operation.capability == "block.full-attention.standard-gqa.v1" ||
          operation.capability ==
              "block.full-attention.standard-gqa-no-position.v1" ||
          operation.capability ==
              "block.mla.causal.latent-kv.bfloat16.v1" ||
          operation.capability ==
              "block.sparse-attention.qsa.output-gated.v1") {
        full_attention_slots_[operation.logical_layer] = target_full_layers_++;
        if (operation.capability ==
            "block.sparse-attention.qsa.output-gated.v1")
          qsa_index_slots_[operation.logical_layer] = qsa_layers_++;
        const auto window = operation.parameters.find(
            "attention_window_tokens");
        const auto window_tokens = window == operation.parameters.end()
            ? 0U
            : static_cast<std::uint32_t>(window->second);
        if (window != operation.parameters.end() &&
            (operation.abi_version < 2U || window->second > max_context_ ||
             window->second > std::numeric_limits<std::uint32_t>::max()))
          throw std::runtime_error("invalid artifact attention window");
        if (window_tokens != 0U) {
          if (window_tokens % kv_page_tokens_ != 0U)
            throw std::runtime_error(
                "attention window must align to KV pages");
          if (window_tokens_ != 0U && window_tokens_ != window_tokens)
            throw std::runtime_error(
                "one provider instance requires one window geometry");
          window_tokens_ = window_tokens;
          window_kv_slots_[operation.logical_layer] = window_attention_layers_++;
        } else {
          global_kv_slots_[operation.logical_layer] = global_attention_layers_++;
        }
      }
      else if (operation.capability ==
                   "block.recurrent-linear-attention.split-gated-delta.v1" ||
               operation.capability ==
                   "block.recurrent-linear-attention.split-gated-delta.no-residual.v1" ||
               operation.capability == "block.mamba2.ssm.v1")
        recurrent_slots_[operation.logical_layer] = recurrent_layers_++;
      else if (operation.capability ==
                   "embedding.ngram-ple.fp4-block32.v1" ||
               operation.capability == "embedding.ngram-ple.v1")
        ple_slots_[operation.logical_layer] = ple_layers_++;
    }
    if ((qsa_enabled_ && qsa_layers_ == 0U) ||
        (!qsa_enabled_ && qsa_layers_ != 0U) ||
        (ple_enabled_ && ple_layers_ == 0U) ||
        (!ple_enabled_ && ple_layers_ != 0U))
      throw std::runtime_error("sparse organ topology is inconsistent");
    if (target_full_layers_ != attribute_u32("full_attention_layers") ||
        target_full_layers_ == 0U ||
        target_full_layers_ + mtp_layers_ >
            std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("full-attention topology is inconsistent");
    const auto fp4_record_bytes = static_cast<std::uint64_t>(
        head_dim_ / 2U + head_dim_ / 32U);
    const auto fp4_layer_page_bytes =
        2U * static_cast<std::uint64_t>(kv_page_tokens_) * kv_heads_ *
        fp4_record_bytes;
    const auto q8_record_bytes =
        static_cast<std::uint64_t>(head_dim_) + sizeof(std::uint16_t);
    const auto q8_layer_page_bytes =
        2U * static_cast<std::uint64_t>(kv_page_tokens_) * kv_heads_ *
        q8_record_bytes;
    const auto mtp_layer_page_bytes =
        mtp_q8_kv_ ? q8_layer_page_bytes : fp4_layer_page_bytes;
    const auto maximum_pages =
        (static_cast<std::uint64_t>(max_context_) + kv_page_tokens_ - 1U) /
        kv_page_tokens_;
    if (maximum_pages > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("maximum KV page count overflows");
    maximum_pages_per_slot_ = static_cast<std::uint32_t>(maximum_pages);
    host_fp16_target_page_bytes_ = static_cast<std::uint64_t>(
        global_attention_layers_);
    for (const auto factor : std::array<std::uint64_t, 5U>{
             2U, kv_page_tokens_, kv_heads_, head_dim_,
             sizeof(std::uint16_t)})
      host_fp16_target_page_bytes_ = checked_multiply(
          host_fp16_target_page_bytes_, factor, "global FP16 KV page size");
    auto resident_fp16_target_page_bytes = host_fp16_target_page_bytes_;
    if (qsa_enabled_) {
      if (window_attention_layers_ != 0U ||
          qsa_layers_ != global_attention_layers_ ||
          host_fp16_target_page_bytes_ >
              std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error(
            "QSA requires one global index per attention layer");
      qsa_index_device_page_bytes_ = static_cast<std::uint64_t>(qsa_layers_);
      for (const auto factor : std::array<std::uint64_t, 3U>{
               kv_page_tokens_, qsa_index_head_dim_,
               sizeof(std::uint16_t)})
        qsa_index_device_page_bytes_ = checked_multiply(
            qsa_index_device_page_bytes_, factor, "QSA index page size");
      if (qsa_index_device_page_bytes_ >
          std::numeric_limits<std::uint64_t>::max() -
              resident_fp16_target_page_bytes)
        throw std::runtime_error("QSA page geometry overflows");
      resident_fp16_target_page_bytes += qsa_index_device_page_bytes_;
    }
    window_pages_per_slot_ = window_tokens_ == 0U
        ? 0U
        : window_tokens_ / kv_page_tokens_;
    window_kv_page_bytes_ = window_attention_layers_;
    for (const auto factor : std::array<std::uint64_t, 5U>{
             2U, kv_page_tokens_, kv_heads_, head_dim_,
             sizeof(std::uint16_t)})
      window_kv_page_bytes_ = checked_multiply(
          window_kv_page_bytes_, factor, "window FP16 KV page size");
    window_kv_bytes_per_slot_ = checked_multiply(
        window_kv_page_bytes_, window_pages_per_slot_,
        "window FP16 KV slot size");
    const auto global_context_bytes = checked_multiply(
        resident_fp16_target_page_bytes, maximum_pages_per_slot_,
        "global FP16 KV context size");
    const auto window_reservation_bytes = checked_multiply(
        checked_multiply(2U, capacity_, "window FP16 KV reservation"),
        window_kv_bytes_per_slot_, "window FP16 KV reservation");
    const auto page_budget_bytes = window_reservation_bytes > kv_cache_bytes_
        ? 0U
        : kv_cache_bytes_ - window_reservation_bytes;
    qsa_tiered_ = qsa_enabled_ && exact_fp16_kv() && capacity_placement_;
    device_resident_fp16_kv_ =
        exact_fp16_kv() && mtp_layers_ == 0U &&
        !qsa_tiered_ &&
        global_context_bytes <= page_budget_bytes;
    compact_flash_prefill_ =
        !exact_fp16_kv() && !qsa_enabled_ && !mla_enabled_ &&
        head_dim_ == 256U && query_heads_ != 0U && kv_heads_ != 0U &&
        query_heads_ % kv_heads_ == 0U;
    large_exact_prefill_workspace_ =
        exact_fp16_kv() && qsa_enabled_ && hyper_enabled_ && ple_enabled_ &&
        !descriptor_.routed_components.empty();
    workspace_rows_ =
        (compact_flash_prefill_ || large_exact_prefill_workspace_)
            ? kMaximumWorkspaceRows
            : kDefaultWorkspaceRows;
    if (qsa_enabled_ && !device_resident_fp16_kv_ && !qsa_tiered_)
      throw std::runtime_error(
          "latency-profile QSA requires exact FP16 KV and index to fit in VRAM");
    if (window_attention_layers_ != 0U && !device_resident_fp16_kv_)
      throw std::runtime_error(
          "windowed exact attention must fit its mixed device KV layout");
    if (mla_enabled_) {
      if (target_kv_encoding_ != TargetKvEncoding::artifact_native ||
          mtp_layers_ != 0U)
        throw std::runtime_error(
            "artifact MLA requires its declared BF16 latent KV layout");
      target_kv_page_bytes_ = static_cast<std::uint64_t>(target_full_layers_);
      for (const auto factor : std::array<std::uint64_t, 3U>{
               kv_page_tokens_, kv_lora_rank_ + qk_rope_head_dim_,
               sizeof(std::uint16_t)})
        target_kv_page_bytes_ = checked_multiply(
            target_kv_page_bytes_, factor, "MLA BF16 latent KV page size");
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ = 0U;
      kv_page_bytes_ = target_kv_page_bytes_;
    } else if (target_kv_encoding_ == TargetKvEncoding::fp8_e4m3_per_head) {
      const auto fp8_record_bytes =
          static_cast<std::uint64_t>(head_dim_) + sizeof(std::uint16_t);
      const auto fp8_layer_page_bytes =
          2U * static_cast<std::uint64_t>(kv_page_tokens_) * kv_heads_ *
          fp8_record_bytes;
      target_kv_page_bytes_ =
          static_cast<std::uint64_t>(target_full_layers_) *
          fp8_layer_page_bytes;
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ =
          static_cast<std::uint64_t>(mtp_layers_) * mtp_layer_page_bytes;
      kv_page_bytes_ = target_kv_page_bytes_ + mtp_kv_page_bytes_;
    } else if (target_kv_encoding_ == TargetKvEncoding::fp4_key_outlier1) {
      const auto key_record_bytes = fp4_record_bytes +
          static_cast<std::uint64_t>(head_dim_ / 32U) *
              sizeof(std::uint32_t);
      const auto layer_page_bytes =
          static_cast<std::uint64_t>(kv_page_tokens_) * kv_heads_ *
          (key_record_bytes + fp4_record_bytes);
      target_kv_page_bytes_ =
          static_cast<std::uint64_t>(target_full_layers_) *
          layer_page_bytes;
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ =
          static_cast<std::uint64_t>(mtp_layers_) * mtp_layer_page_bytes;
      kv_page_bytes_ = target_kv_page_bytes_ + mtp_kv_page_bytes_;
    } else if (target_kv_encoding_ ==
                   TargetKvEncoding::q4_bfp_key_outlier1 ||
               target_kv_encoding_ == TargetKvEncoding::q4_bfp ||
               target_kv_encoding_ == TargetKvEncoding::q4_per_head ||
               target_kv_encoding_ == TargetKvEncoding::q5_q4_bfp) {
      if (head_dim_ != 256U || kv_page_tokens_ % 32U)
        throw std::runtime_error(
            "packed BFP target KV requires head dimension 256 and page alignment 32");
      const auto exponent_bytes =
          static_cast<std::uint64_t>(head_dim_ / 64U);
      const auto key_record_bytes =
          target_kv_encoding_ == TargetKvEncoding::q5_q4_bfp
              ? static_cast<std::uint64_t>(head_dim_) * 5U / 8U +
                    exponent_bytes + sizeof(std::uint16_t)
          : target_kv_encoding_ == TargetKvEncoding::q4_bfp
              ? static_cast<std::uint64_t>(head_dim_ / 2U) +
                    exponent_bytes + sizeof(std::uint16_t)
          : target_kv_encoding_ == TargetKvEncoding::q4_per_head
              ? static_cast<std::uint64_t>(head_dim_ / 2U) +
                    sizeof(std::uint16_t)
              : static_cast<std::uint64_t>(head_dim_ / 2U) +
                    exponent_bytes + sizeof(std::uint16_t) +
                    static_cast<std::uint64_t>(head_dim_ / 32U) *
                        sizeof(std::uint32_t);
      const auto value_record_bytes =
          static_cast<std::uint64_t>(head_dim_ / 2U) +
          (target_kv_encoding_ == TargetKvEncoding::q4_per_head
               ? 0U
               : exponent_bytes) +
          sizeof(std::uint16_t);
      const auto layer_page_bytes =
          static_cast<std::uint64_t>(kv_page_tokens_) * kv_heads_ *
          (key_record_bytes + value_record_bytes);
      target_kv_page_bytes_ =
          static_cast<std::uint64_t>(target_full_layers_) *
          layer_page_bytes;
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ =
          static_cast<std::uint64_t>(mtp_layers_) * mtp_layer_page_bytes;
      kv_page_bytes_ = target_kv_page_bytes_ + mtp_kv_page_bytes_;
    } else if (device_resident_fp16_kv()) {
      qsa_index_page_offset_ = static_cast<std::uint32_t>(
          host_fp16_target_page_bytes_);
      target_kv_page_bytes_ = resident_fp16_target_page_bytes;
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ = 0U;
      kv_page_bytes_ = target_kv_page_bytes_;
    } else {
      const auto target_device_layers =
          host_authoritative_fp16_kv() ? 0U : target_full_layers_;
      target_kv_page_bytes_ = static_cast<std::uint64_t>(
          target_device_layers) * fp4_layer_page_bytes;
      if (qsa_tiered_) {
        qsa_index_page_offset_ = 0U;
        target_kv_page_bytes_ = qsa_index_device_page_bytes_;
      }
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ =
          static_cast<std::uint64_t>(mtp_layers_) * mtp_layer_page_bytes;
      kv_page_bytes_ = target_kv_page_bytes_ + mtp_kv_page_bytes_;
    }
    const auto maximum_service_pages = checked_multiply(
        capacity_, maximum_pages_per_slot_, "service KV page count");
    kv_page_capacity_ = kv_page_bytes_ == 0U
        ? maximum_service_pages
        : std::min<std::uint64_t>(page_budget_bytes / kv_page_bytes_,
                                  maximum_service_pages);
    if (!kv_page_capacity_)
      throw std::runtime_error("KV budget fits no physical page");
    const auto reserved_paged_device_bytes = checked_multiply(
        kv_page_bytes_, maximum_service_pages,
        "maximum non-target device KV reservation");
    if (host_authoritative_fp16_kv() && !qsa_tiered_ &&
        host_fp16_target_page_bytes_ != 0U &&
        reserved_paged_device_bytes < page_budget_bytes) {
      target_mirror_page_capacity_ = std::min<std::uint64_t>(
          maximum_service_pages,
          (page_budget_bytes - reserved_paged_device_bytes) /
              host_fp16_target_page_bytes_);
    }
    service_kv_page_bytes_ = host_authoritative_fp16_kv()
        ? host_fp16_target_page_bytes_ + target_kv_page_bytes_ +
              mtp_kv_page_bytes_
        : kv_page_bytes_;
    const auto recurrent_park_values = checked_multiply(
        recurrent_layers_, recurrent_conv_values_ + recurrent_matrix_values_,
        "recurrent park state");
    const auto ple_park_values = checked_multiply(
        ple_layers_, ple_conv_state_values_, "PLE park state");
    const auto park_state_bytes = checked_multiply(
        recurrent_park_values + ple_park_values + 3ULL * hidden_size_,
        sizeof(float), "request park state");
    const auto host_state_reservation = host_authoritative_fp16_kv()
        ? checked_multiply(service_kv_page_bytes_, maximum_pages_per_slot_,
                           "authoritative KV host reservation") +
              park_state_bytes
        : 0U;
    if (!descriptor_.routed_components.empty()) {
      if (host_state_reservation >= ram_cache_bytes_)
        throw std::runtime_error(
            "authoritative KV leaves no RAM for routed experts");
      routed_experts_ = std::make_unique<Fp4RoutedExperts>(
          artifact_, ram_cache_bytes_ - host_state_reservation,
          vram_cache_bytes_, workspace_rows_,
          std::move(active_expert_devices),
          active_expert_device_cache_bytes,
          active_expert_host_cache_bytes);
      parking_ram_capacity_bytes_ =
          ram_cache_bytes_ - routed_experts_->host_cache_capacity_bytes();
    } else {
      parking_ram_capacity_bytes_ = ram_cache_bytes_;
    }
    service_kv_page_capacity_ = host_authoritative_fp16_kv()
        ? std::min<std::uint64_t>(
              parking_ram_capacity_bytes_ / service_kv_page_bytes_,
              kv_page_capacity_)
        : kv_page_capacity_;
    if (!service_kv_page_capacity_)
      throw std::runtime_error("KV RAM budget fits no service page");
    prepared_target_.resize(descriptor_.operation_program.size());
    slot_in_use_.assign(capacity_, false);
    slot_rope_deltas_.assign(capacity_, 0);
  }

  ~DenseFp4Provider() override {
    if (qsa_host_selected_keys_)
      static_cast<void>(cudaFreeHost(qsa_host_selected_keys_));
    if (qsa_host_selected_values_)
      static_cast<void>(cudaFreeHost(qsa_host_selected_values_));
    if (parking_stream_)
      static_cast<void>(cudaStreamDestroy(parking_stream_));
    for (auto& event : gpu_event_pool_) {
      if (event.start) static_cast<void>(cudaEventDestroy(event.start));
      if (event.stop) static_cast<void>(cudaEventDestroy(event.stop));
    }
    for (auto* page : all_kv_pages_)
      if (page) static_cast<void>(cudaFree(page));
    for (auto* page : all_target_mirror_pages_)
      if (page) static_cast<void>(cudaFree(page));
    release_staged_dense_weights();
    if (logits_) static_cast<void>(cudaFree(logits_));
    for (auto* allocation : allocations_)
      if (allocation) static_cast<void>(cudaFree(allocation));
  }

  er::PrepareOperationResult prepare(
      const er::OperationPreparationContext& context) override {
    try {
      if (context.model.content_hash != descriptor_.content_hash ||
          context.compiled.logical_operation >= prepared_target_.size())
        throw std::runtime_error("operation belongs to a different artifact");
      auto prepared = std::make_shared<PreparedOperation>();
      prepared->kernel = kernel_from_capability(context.operation.capability);
      prepared->capability = context.operation.capability;
      prepared->abi_version = context.operation.abi_version;
      if (prepared->kernel == Kernel::exact_decode)
        throw std::runtime_error("exact decode is not a scalar operation");
      prepared->logical_operation = context.compiled.logical_operation;
      prepared->logical_layer = context.compiled.logical_layer;
      prepared->component_layer = context.compiled.component_layer;
      prepared->parameters = context.operation.parameters;
      if (prepared->logical_layer != er::kModelLevelOperationLayer) {
        prepared->full_attention_slot =
            full_attention_slots_.at(prepared->logical_layer);
        prepared->attention_window_tokens = [&] {
          const auto found = prepared->parameters.find(
              "attention_window_tokens");
          return found == prepared->parameters.end()
              ? 0U
              : static_cast<std::uint32_t>(found->second);
        }();
        prepared->kv_layer_slot = prepared->attention_window_tokens == 0U
            ? global_kv_slots_.at(prepared->logical_layer)
            : window_kv_slots_.at(prepared->logical_layer);
        prepared->recurrent_slot = recurrent_slots_.at(prepared->logical_layer);
        prepared->qsa_index_slot =
            qsa_index_slots_.at(prepared->logical_layer);
        prepared->ple_slot = ple_slots_.at(prepared->logical_layer);
      }
      for (const auto& binding : context.tensors) {
        if (!binding.tensor)
          throw std::runtime_error("operation tensor binding is null");
        const auto host_mapped = prepared->kernel == Kernel::ple &&
            (binding.role == "layer_multipliers" ||
             binding.role == "head_vocab_sizes" ||
             binding.role == "head_offsets" ||
             binding.role.starts_with("embedding_shard."));
        if (host_mapped)
          prepared->host_tensors.emplace(binding.role,
                                         &ensure_host_tensor(*binding.tensor));
        else
          prepared->tensors.emplace(binding.role,
                                    &ensure_tensor(*binding.tensor));
      }
      for (std::size_t index = 0U;
           index < context.compiled.input_values.size(); ++index)
        prepared->input_indices.emplace(
            context.compiled.input_values[index].port, index);
      for (const auto& binding : context.compiled.output_values) {
        const auto output = context.operation.output_bindings.find(binding.port);
        if (output == context.operation.output_bindings.end())
          throw std::runtime_error("compiled output port is absent");
        prepared->outputs.emplace_back(binding.port, output->second.abi);
      }
      validate_operation(*prepared);
      if (prepared_target_[prepared->logical_operation])
        throw std::runtime_error("operation was prepared twice");
      prepared_target_[prepared->logical_operation] = prepared;
      return {er::Status::success(), std::move(prepared)};
    } catch (const std::exception& error) {
      return {{er::ErrorCode::invalid_argument, error.what()}, {}};
    }
  }

  er::PrepareOperationResult prepare_exact_decode(
      const er::ExactDecodePreparationContext& context) override {
    try {
      const auto legacy =
          context.program.capability ==
              "decode.mtp.dense-full-attention.fp4-block32.exact.v1" &&
          context.program.abi_version == 1U &&
          context.compiled.maximum_emitted_tokens == 2U &&
          parameter_u32(context.compiled.parameters, "draft_layers") == 1U;
      const auto multi_capability =
          context.program.capability ==
              "decode.mtp.dense-full-attention.fp4-block32.exact.v2" ||
          context.program.capability ==
              "decode.mtp.dense-full-attention.fp4-mxfp6-io.exact.v3";
      const auto multi =
          multi_capability &&
          context.program.abi_version == 2U &&
          context.compiled.maximum_emitted_tokens == draft_depth_ + 1U &&
          parameter_u32(context.compiled.parameters, "source_mtp_layers") ==
              1U &&
          parameter_u32(context.compiled.parameters, "draft_depth") ==
              draft_depth_ &&
          parameter_u32(context.compiled.parameters,
                        "draft_vocabulary_size") ==
              draft_vocabulary_size_ &&
          parameter_u32(context.compiled.parameters, "mtp_kv_encoding") ==
              1U;
      if (context.model.content_hash != descriptor_.content_hash ||
          (!legacy && !multi) ||
          context.program.abi_version != exact_decode_abi_ ||
          parameter_u32(context.compiled.parameters, "embedding_first") != 1U ||
          parameter_u32(context.compiled.parameters, "post_norm") != 1U)
        throw std::runtime_error("unsupported exact-decode artifact contract");
      auto prepared = std::make_shared<PreparedOperation>();
      prepared->kernel = Kernel::exact_decode;
      prepared->capability = context.program.capability;
      prepared->abi_version = context.program.abi_version;
      prepared->parameters = context.compiled.parameters;
      for (const auto& binding : context.tensors) {
        if (!binding.tensor)
          throw std::runtime_error("exact-decode tensor binding is null");
        prepared->tensors.emplace(binding.role,
                                  &ensure_tensor(*binding.tensor));
      }
      validate_exact(*prepared);
      exact_ = prepared;
      return {er::Status::success(), std::move(prepared)};
    } catch (const std::exception& error) {
      return {{er::ErrorCode::invalid_argument, error.what()}, {}};
    }
  }

  er::CreateOperationRequestStateResult create_request_state(
      const er::ProgramRequestContext& request) override;
  er::OperationExecutionHandle execute(
      const er::IPreparedOperation& operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const er::OperationInvocation& invocation) override;
  [[nodiscard]] bool supports_program_sequence(
      const er::CompiledModelProgram& program) const noexcept override;
  er::OperationExecutionHandle execute_program_sequence(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const er::ProgramSequenceInvocation& invocation) override;
  [[nodiscard]] bool supports_request_state_retention()
      const noexcept override {
    return true;
  }
  er::Status checkpoint_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      std::uint32_t next_position) override;
  er::Status rewind_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      std::uint32_t next_position) override;
  er::Status rebind_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const er::ProgramRequestContext& request) override;
  [[nodiscard]] bool supports_request_state_parking()
      const noexcept override {
    // Parking is an effective service capability, not only an implemented
    // code path. Routed artifacts may consume the entire host-cache budget;
    // in that case no complete KV page can be parked and advertising support
    // would contradict the common worker contract.
    return service_kv_page_bytes_ != 0U &&
           parking_ram_capacity_bytes_ >= service_kv_page_bytes_;
  }
  er::RequestStateParkingResult park_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      std::uint32_t next_position) override;
  er::RequestStateParkingResult restore_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& state)
      override;
  [[nodiscard]] bool supports_request_state_persistence()
      const noexcept override {
    return supports_request_state_parking();
  }
  er::RequestStateSnapshotResult save_request_state_snapshot(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const std::filesystem::path& root,
      std::uint64_t generation) override;
  er::RequestStateSnapshotResult load_request_state_snapshot(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const std::filesystem::path& root,
      std::uint64_t generation) override;
  er::Status prune_request_state_snapshots(
      const std::filesystem::path& root,
      std::uint64_t generation) override;
  er::Status synchronize_exact_decode(
      const er::IPreparedOperation& operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const er::ExactDecodeSynchronization& synchronization) override;
  er::Status synchronize_exact_decode_batch(
      const er::IPreparedOperation& operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const er::ExactDecodeSynchronizationBatch& synchronization) override;
  er::ExactDecodeExecutionHandle execute_exact_decode(
      const er::IPreparedOperation& operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      const er::ExactDecodeInvocation& invocation) override;

  [[nodiscard]] std::uint64_t kv_page_bytes() const noexcept {
    return service_kv_page_bytes_;
  }
  [[nodiscard]] std::uint64_t kv_page_capacity() const noexcept {
    return service_kv_page_capacity_;
  }
  [[nodiscard]] std::uint64_t routed_ram_cache_bytes() const noexcept {
    return routed_experts_ ? routed_experts_->host_cache_capacity_bytes() : 0U;
  }
  [[nodiscard]] std::uint64_t parking_ram_capacity_bytes() const noexcept {
    return parking_ram_capacity_bytes_;
  }
  [[nodiscard]] std::uint32_t kv_page_tokens() const noexcept {
    return kv_page_tokens_;
  }
  [[nodiscard]] std::uint32_t prefill_batch_rows() const noexcept {
    return workspace_rows_;
  }
  [[nodiscard]] std::uint64_t effective_vram_cache_bytes() const noexcept {
    return routed_experts_ ? routed_experts_->vram_cache_capacity_bytes()
                           : vram_cache_bytes_;
  }
  er::Status finalize_startup() noexcept {
    try {
      initialize_execution();
      return er::Status::success();
    } catch (const std::exception& error) {
      return {er::ErrorCode::internal, error.what()};
    }
  }
  [[nodiscard]] bool exact_fp16_kv() const noexcept {
    return target_kv_encoding_ == TargetKvEncoding::fp16;
  }
  [[nodiscard]] bool host_authoritative_fp16_kv() const noexcept {
    return exact_fp16_kv() && !device_resident_fp16_kv_;
  }
  [[nodiscard]] bool device_resident_fp16_kv() const noexcept {
    return exact_fp16_kv() && device_resident_fp16_kv_;
  }
  [[nodiscard]] std::string_view target_kv_dtype() const noexcept {
    if (mla_enabled_) return "bf16-latent";
    if (exact_fp16_kv()) return "fp16";
    if (target_kv_encoding_ == TargetKvEncoding::fp8_e4m3_per_head)
      return "fp8-e4m3-per-head";
    if (target_kv_encoding_ == TargetKvEncoding::fp4_key_outlier1)
      return "fp4-e2m1-ue8m0-block32-key-outlier1";
    if (target_kv_encoding_ == TargetKvEncoding::q4_bfp_key_outlier1)
      return "q4-bfp16-block32-key-outlier1";
    if (target_kv_encoding_ == TargetKvEncoding::q4_bfp)
      return "q4-bfp16-block32";
    if (target_kv_encoding_ == TargetKvEncoding::q4_per_head)
      return "q4-f16-per-head";
    if (target_kv_encoding_ == TargetKvEncoding::q5_q4_bfp)
      return "q5-q4-bfp16-block32";
    return "fp4-e2m1-ue8m0-block32";
  }
  [[nodiscard]] std::map<std::string, std::uint64_t, std::less<>> telemetry()
      const {
    std::uint64_t resident_pages{};
    for (const auto& slot : slot_pages_)
      resident_pages += static_cast<std::uint64_t>(std::count_if(
          slot.begin(), slot.end(), [](const void* page) { return page; }));
    auto result = std::map<std::string, std::uint64_t, std::less<>>{
            {"resident_tensor_bytes", uploaded_tensor_bytes_},
            {"provider_activation_bf16", activation_bf16_ ? 1U : 0U},
            {"provider_dense_activation_input_bf16",
             dense_activation_input_bf16_ ? 1U : 0U},
            {"provider_program_steps", program_steps_},
            {"provider_prefill_batches", prefill_batches_},
            {"provider_prefill_tokens", prefill_tokens_},
            {"provider_exact_sync_batches", exact_sync_batches_},
            {"provider_exact_sync_tokens", exact_sync_tokens_},
            {"provider_exact_calls", exact_calls_},
            {"provider_exact_target_batches", exact_target_batches_},
            {"provider_accepted_drafts", accepted_drafts_},
            {"provider_accepted_depth_0", accepted_depth_calls_[0]},
            {"provider_accepted_depth_1", accepted_depth_calls_[1]},
            {"provider_accepted_depth_2", accepted_depth_calls_[2]},
            {"provider_accepted_depth_3", accepted_depth_calls_[3]},
            {"provider_accepted_depth_4", accepted_depth_calls_[4]},
            {"provider_speculative_recurrent_checkpoint_bytes",
             speculative_recurrent_checkpoint_bytes_},
            {"provider_speculative_recurrent_restores",
             speculative_recurrent_restores_},
            {"provider_exact_decode_abi", exact_decode_abi_},
            {"provider_mtp_draft_depth", draft_depth_},
            {"provider_mtp_draft_vocabulary_size",
             draft_vocabulary_size_},
            {"provider_mtp_q8_kv", mtp_q8_kv_ ? 1U : 0U},
            {"provider_fused_multiquery_attention_calls",
             fused_multiquery_attention_calls_},
            {"provider_program_sequence_batches", program_sequence_batches_},
            {"provider_program_sequence_tokens", program_sequence_tokens_},
            {"provider_program_sequence_tiles", program_sequence_tiles_},
            {"provider_program_sequence_device_bytes",
             program_sequence_device_bytes_},
            {"provider_program_sequence_host_bytes",
             program_sequence_host_bytes_},
            {"provider_program_sequence_tile_rows", sequence_tile_rows_},
            {"provider_workspace_rows", workspace_rows_},
            {"provider_large_exact_prefill_workspace",
             large_exact_prefill_workspace_ ? 1U : 0U},
            {"provider_workspace_preflight_free_bytes",
             workspace_preflight_free_bytes_},
            {"provider_workspace_postallocation_free_bytes",
             workspace_postallocation_free_bytes_},
            {"provider_compact_flash_prefill",
             compact_flash_prefill_ ? 1U : 0U},
            {"provider_sampling_gpu_calls", sampling_gpu_calls_},
            {"provider_sampling_host_calls", sampling_host_calls_},
            {"provider_sampling_logit_transfer_bytes",
             sampling_logit_transfer_bytes_},
            {"provider_staged_dense_weight_bytes",
             staged_dense_weights_ ? staged_dense_weight_capacity_bytes_ : 0U},
            {"provider_staged_dense_weight_capacity_bytes",
             staged_dense_weight_capacity_bytes_},
            {"provider_staged_dense_weight_decode_bytes",
             staged_dense_weight_decode_bytes_},
            {"provider_staged_dense_weight_gemm_calls",
             staged_dense_weight_gemm_calls_},
            {"provider_staged_dense_weight_low_memory_fallbacks",
             staged_dense_weight_low_memory_fallbacks_},
            {"provider_logits_workspace_rows", logits_capacity_rows_},
            {"provider_logits_workspace_bytes",
             static_cast<std::uint64_t>(logits_capacity_rows_) *
                 vocabulary_size_ * sizeof(float)},
            {"provider_gpu_measured_batches", gpu_measured_batches_},
            {"provider_gpu_embedding_ns",
             gpu_phase_ns_[static_cast<std::size_t>(GpuPhase::embedding)]},
            {"provider_gpu_vision_ns",
             gpu_phase_ns_[static_cast<std::size_t>(GpuPhase::vision)]},
            {"provider_vision_batches", vision_batches_},
            {"provider_vision_images", vision_images_},
            {"provider_vision_patches", vision_patches_},
            {"provider_gpu_full_attention_ns",
             gpu_phase_ns_[static_cast<std::size_t>(
                 GpuPhase::full_attention)]},
            {"provider_gpu_recurrent_attention_ns",
             gpu_phase_ns_[static_cast<std::size_t>(
                 GpuPhase::recurrent_attention)]},
            {"provider_gpu_router_ns",
             gpu_phase_ns_[static_cast<std::size_t>(GpuPhase::router)]},
            {"provider_gpu_routed_moe_ns",
             gpu_phase_ns_[static_cast<std::size_t>(GpuPhase::routed_moe)]},
            {"provider_gpu_ffn_ns",
             gpu_phase_ns_[static_cast<std::size_t>(GpuPhase::ffn)]},
            {"provider_gpu_head_ns",
             gpu_phase_ns_[static_cast<std::size_t>(GpuPhase::head)]},
            {"provider_gpu_mtp_ns",
             gpu_phase_ns_[static_cast<std::size_t>(GpuPhase::mtp)]},
            {"provider_device_resident_fp16_kv",
             device_resident_fp16_kv() ? 1U : 0U},
            {"provider_target_mirror_page_capacity",
             target_mirror_page_capacity_},
            {"provider_target_mirror_attention_calls",
             target_mirror_attention_calls_},
            {"provider_target_host_attention_calls",
             target_host_attention_calls_},
            {"provider_target_mirror_spills", target_mirror_spills_},
            {"provider_target_mirror_restore_bytes",
             target_mirror_restore_bytes_},
            {"provider_qsa_tiered", qsa_tiered_ ? 1U : 0U},
            {"provider_qsa_index_device_page_bytes",
             qsa_index_device_page_bytes_},
            {"provider_qsa_host_commit_bytes", qsa_host_commit_bytes_},
            {"provider_qsa_selected_host_bytes", qsa_selected_host_bytes_},
            {"provider_qsa_selected_host_tokens", qsa_selected_host_tokens_},
            {"provider_qsa_selected_host_calls", qsa_selected_host_calls_},
            {"provider_qsa_device_staged_calls", qsa_device_staged_calls_},
            {"provider_qsa_score_device_to_host_bytes",
             qsa_score_device_to_host_bytes_},
            {"provider_authoritative_fp16_kv_bytes", host_kv_bytes_},
            {"provider_parked_request_bytes", parked_request_bytes_},
            {"provider_parked_session_bytes", parked_session_bytes_},
            {"provider_request_park_calls", park_calls_},
            {"provider_request_restore_calls", restore_calls_},
            {"provider_park_device_to_host_bytes",
             park_device_to_host_bytes_},
            {"provider_restore_host_to_device_bytes",
             restore_host_to_device_bytes_},
            {"provider_target_kv_page_bytes", target_kv_page_bytes_},
            {"provider_global_attention_layers", global_attention_layers_},
            {"provider_window_attention_layers", window_attention_layers_},
            {"provider_window_kv_tokens", window_tokens_},
            {"provider_window_kv_bytes_per_slot",
             window_kv_bytes_per_slot_},
            {"provider_mtp_kv_page_bytes", mtp_kv_page_bytes_},
            {"provider_target_mirror_resident_pages",
             static_cast<std::uint64_t>(std::accumulate(
                 slot_target_mirror_pages_.begin(),
                 slot_target_mirror_pages_.end(), std::size_t{0U},
                 [](std::size_t count, const auto& pages) {
                   return count + static_cast<std::size_t>(std::count_if(
                                      pages.begin(), pages.end(),
                                      [](const void* page) { return page; }));
                 }))},
            {"provider_target_mirror_resident_bytes",
             static_cast<std::uint64_t>(std::accumulate(
                 slot_target_mirror_pages_.begin(),
                 slot_target_mirror_pages_.end(), std::size_t{0U},
                 [](std::size_t count, const auto& pages) {
                   return count + static_cast<std::size_t>(std::count_if(
                                      pages.begin(), pages.end(),
                                      [](const void* page) { return page; }));
                 })) * host_fp16_target_page_bytes_},
            {"kv_allocated_pages", resident_pages},
            {"kv_physical_pages", all_kv_pages_.size()}};
    if (routed_experts_) {
      const auto cache = routed_experts_->telemetry();
      const auto hybrid = routed_experts_->hybrid_telemetry();
      result.emplace("cache_vram_hits", cache.acquire_vram_hits);
      result.emplace("cache_ram_hits", cache.acquire_ram_hits);
      result.emplace("cache_ssd_misses", cache.acquire_ssd_misses);
      result.emplace("cache_read_bytes", cache.read_bytes);
      result.emplace("cache_uploaded_bytes", cache.uploaded_bytes);
      result.emplace("cache_storage_wait_ns", cache.storage_wait_ns);
      result.emplace("cache_upload_wait_ns", cache.upload_wait_ns);
      result.emplace("cache_evictions", cache.eviction_count);
      result.emplace("routed_host_bank_page_locked",
                     routed_experts_->host_bank_page_locked() ? 1U : 0U);
      result.emplace("routed_host_warm_completed",
                     routed_experts_->host_warm_completed());
      result.emplace("routed_host_warm_failed",
                     routed_experts_->host_warm_failed());
      result.emplace("routed_grouped_prefill_calls",
                     routed_experts_->grouped_prefill_calls());
      result.emplace("routed_grouped_prefill_work_items",
                     routed_experts_->grouped_prefill_work_items());
      result.emplace("routed_grouped_prefill_selections",
                     routed_experts_->grouped_prefill_selections());
      result.emplace("routed_cpu_decisions", hybrid.cpu_only +
                                             hybrid.cpu_cost_wins +
                                             hybrid.cpu_calibrations);
      result.emplace("routed_gpu_upload_decisions", hybrid.gpu_only +
                                                    hybrid.gpu_cost_wins +
                                                    hybrid.gpu_cache_warms +
                                                    hybrid.stable_ties);
      result.emplace("routed_cpu_observations", hybrid.cpu_observations);
      result.emplace("routed_gpu_observations", hybrid.gpu_observations);
      result.emplace("routed_h2d_observations", hybrid.h2d_observations);
      result.emplace("routed_resident_gpu_decisions", hybrid.resident_gpu);
      result.emplace("routed_cpu_ns_per_selection",
                     static_cast<std::uint64_t>(
                         std::llround(hybrid.cpu_ns_per_selection)));
      result.emplace("routed_gpu_ns_per_selection",
                     static_cast<std::uint64_t>(
                         std::llround(hybrid.gpu_ns_per_selection)));
      result.emplace("routed_h2d_bytes_per_second",
                     static_cast<std::uint64_t>(
                         std::llround(hybrid.h2d_bytes_per_second)));
      if (const auto active = routed_experts_->active_telemetry()) {
        result.emplace("active_expert_requests", active->requests);
        result.emplace("active_expert_completed", active->completed);
        result.emplace("active_expert_failed", active->failed);
        result.emplace("active_expert_cancelled", active->cancelled);
        result.emplace("active_expert_input_bytes",
                       active->activation_input_bytes);
        result.emplace("active_expert_output_bytes",
                       active->activation_output_bytes);
        result.emplace("active_expert_weight_bytes",
                       active->owner_weight_read_bytes);
        result.emplace("active_expert_storage_bytes",
                       active->owner_storage_read_bytes);
        result.emplace("active_expert_ram_bytes",
                       active->owner_ram_read_bytes);
        result.emplace("active_expert_vram_bytes",
                       active->owner_vram_read_bytes);
        result.emplace("active_expert_execution_ns",
                       active->owner_execution_ns);
      }
    }
    return result;
  }

 private:
  static constexpr std::uint32_t kNoSlot =
      std::numeric_limits<std::uint32_t>::max();

  struct PinnedHostBuffer final {
    PinnedHostBuffer() = default;
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          bytes(std::exchange(other.bytes, 0U)) {}
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
      if (this == &other) return *this;
      if (data) static_cast<void>(cudaFreeHost(data));
      data = std::exchange(other.data, nullptr);
      bytes = std::exchange(other.bytes, 0U);
      return *this;
    }
    ~PinnedHostBuffer() {
      if (data) static_cast<void>(cudaFreeHost(data));
    }
    void allocate(std::size_t size) {
      if (!size || data) throw std::runtime_error("invalid pinned allocation");
      void* allocation{};
      cuda_check(cudaHostAlloc(&allocation, size, cudaHostAllocPortable),
                 "allocate parked request blob");
      data = static_cast<std::byte*>(allocation);
      bytes = size;
    }
    std::byte* data{};
    std::size_t bytes{};
  };

  struct ParkedRequestState final {
    PinnedHostBuffer payload;
    std::vector<std::uint32_t> page_indices;
    std::uint32_t logical_pages{};
    std::uint64_t page_payload_bytes{};
    std::uint64_t window_payload_bytes{};
    std::uint64_t bytes{};
    std::uint64_t reported_bytes{};
  };

  struct HostKvPage final {
    std::uint16_t* allocation{};
  };

  struct GpuEventPair final {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    GpuPhase phase{};
  };

  class RequestState final : public er::IOperationProviderRequestState {
   public:
    RequestState(DenseFp4Provider& provider, std::uint32_t slot) noexcept
        : provider_(provider), slot_(slot) {}
    ~RequestState() override { provider_.release_request_state(*this); }
    [[nodiscard]] std::uint32_t slot() const noexcept { return slot_; }
    [[nodiscard]] bool parked() const noexcept {
      return parked_state.has_value();
    }
    std::uint32_t current_position{};
    std::uint32_t current_batch_first{};
    std::uint32_t current_batch_rows{};
    std::uint32_t synchronization_first{};
    std::uint32_t synchronization_rows{};
    std::uint32_t synchronization_consumed{};
    std::uint32_t mtp_length{};
    std::uint32_t synchronized_token{};
    std::uint32_t draft_token{};
    std::vector<MtpPrediction> draft_predictions;
    std::uint32_t retention_position{};
    std::int32_t rope_delta{};
    std::vector<std::uint32_t> prompt_mrope_positions;
    std::vector<float> sequence_target_hidden;
    std::vector<std::uint32_t> ple_history;
    std::vector<std::uint32_t> ple_retention_history;
    bool draft_valid{};
    bool retention_valid{};
    bool exact_decode_enabled{};
    std::vector<HostKvPage> host_kv_pages;
    std::uint32_t host_kv_populated_tokens{};
    std::uint64_t host_kv_bytes{};
    bool target_mirror_enabled{};
    std::optional<ParkedRequestState> parked_state;

   private:
    friend class DenseFp4Provider;
    DenseFp4Provider& provider_;
    std::uint32_t slot_{};
  };

  struct SequenceState final {
    std::shared_ptr<RequestState> request;
    er::ProgramRequestContext generation;
    std::vector<const PreparedOperation*> operations;
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> positions;
    // Allocated only when exact decode needs every target hidden row after
    // prefill. Sampling keeps the working tile exclusively on the GPU.
    std::vector<float> hidden;
    MediaPayload media;
    std::size_t tile_first{};
    std::size_t tile_rows{};
    std::size_t next_operation{};
    std::size_t next_row{};
    std::uint32_t retention_position{};
    std::atomic<bool> cancelled{};
    bool vision_prepared{};
    bool staging_disabled{};
    bool terminal{};
  };

  struct StagedDenseWeight final {
    const void* data{};
    std::size_t bytes{};
  };

  static std::uint32_t parameter_u32(
      const std::map<std::string, std::uint64_t, std::less<>>& values,
      std::string_view name) {
    const auto found = values.find(name);
    if (found == values.end() ||
        found->second > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("missing or invalid program parameter " +
                               std::string(name));
    return static_cast<std::uint32_t>(found->second);
  }

  static float parameter_f32(
      const std::map<std::string, std::uint64_t, std::less<>>& values,
      std::string_view name) {
    const auto bits = parameter_u32(values, name);
    const auto result = std::bit_cast<float>(bits);
    if (!std::isfinite(result) || !(result > 0.0F))
      throw std::runtime_error("invalid floating-point program parameter " +
                               std::string(name));
    return result;
  }

  static float parameter_f32_or(
      const std::map<std::string, std::uint64_t, std::less<>>& values,
      std::string_view name, float fallback) {
    if (!values.contains(name)) return fallback;
    return parameter_f32(values, name);
  }

  std::uint32_t attribute_u32(std::string_view name) const {
    return parameter_u32(descriptor_.attributes, name);
  }

  float attribute_f32(std::string_view name) const {
    const auto value = attribute_u32(name);
    const auto result = std::bit_cast<float>(value);
    if (!std::isfinite(result) || !(result > 0.0F))
      throw std::runtime_error("invalid floating-point model parameter " +
                               std::string(name));
    return result;
  }

  DeviceTensor& ensure_tensor(const er::ImmutableModelTensor& source) {
    const auto retained = tensors_.find(source.name);
    if (retained != tensors_.end()) return *retained->second;
    if (!source.value.valid() ||
        source.value.abi != "artifact.dense-record.v1" ||
        source.value.memory_domain != "host.mmap.readonly" ||
        source.data_offset > source.value.bytes ||
        source.data_bytes > source.value.bytes - source.data_offset ||
        source.scale_offset > source.value.bytes ||
        source.scale_bytes > source.value.bytes - source.scale_offset)
      throw std::runtime_error("invalid mapped dense tensor " + source.name);
    auto tensor = std::make_unique<DeviceTensor>();
    tensor->name = source.name;
    tensor->encoding = source.encoding;
    tensor->quant_abi = source.quant_abi;
    tensor->shape = source.shape;
    const auto scale_allocation_offset = source.encoding == "I8"
        ? (source.data_bytes + alignof(float) - 1U) &
              ~(static_cast<std::uint64_t>(alignof(float)) - 1U)
        : source.data_bytes;
    if (scale_allocation_offset < source.data_bytes ||
        source.scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                                 scale_allocation_offset)
      throw std::runtime_error("dense tensor allocation overflows");
    tensor->allocation_bytes = scale_allocation_offset + source.scale_bytes;
    if (!tensor->allocation_bytes ||
        tensor->allocation_bytes > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("dense tensor allocation is invalid");
    tensor->allocation = device_allocate<std::byte>(
        allocations_, static_cast<std::size_t>(tensor->allocation_bytes));
    const auto* host = source.value.data;
    cuda_check(cudaMemcpy(tensor->allocation, host + source.data_offset,
                          static_cast<std::size_t>(source.data_bytes),
                          cudaMemcpyHostToDevice),
               "upload dense tensor data");
    if (source.scale_bytes)
      cuda_check(cudaMemcpy(tensor->allocation + scale_allocation_offset,
                            host + source.scale_offset,
                            static_cast<std::size_t>(source.scale_bytes),
                            cudaMemcpyHostToDevice),
                 "upload dense tensor scales");
    if (source.encoding == "FP4_E2M1") {
      if ((source.quant_abi != er::kExpertQuantAbiFp4Block32 &&
           source.quant_abi != er::kExpertRecordAbiNvfp4Block16W4A4) ||
          source.shape.empty())
        throw std::runtime_error("FP4 tensor has the wrong quantization ABI");
      const auto rows = checked_product(std::span(source.shape).first(
          source.shape.size() - 1U));
      if (source.quant_abi == er::kExpertRecordAbiNvfp4Block16W4A4) {
        if (source.shape.size() != 2U || source.shape[1] % 16U)
          throw std::runtime_error("native NVFP4 matrix geometry is invalid");
        const auto local = rows * source.shape.back() / 16U;
        if (source.data_bytes != rows * source.shape.back() / 2U ||
            (source.scale_bytes != local + 2U * sizeof(float) &&
             source.scale_bytes != local + sizeof(float) +
                                       sizeof(std::uint16_t)))
          throw std::runtime_error(
              "native NVFP4 matrix storage geometry is inconsistent");
        float weight_divisor{};
        std::memcpy(&weight_divisor, host + source.scale_offset + local,
                    sizeof(weight_divisor));
        float input_divisor{};
        if (source.scale_bytes == local + 2U * sizeof(float)) {
          std::memcpy(&input_divisor,
                      host + source.scale_offset + local + sizeof(float),
                      sizeof(input_divisor));
        } else {
          std::uint16_t bits{};
          std::memcpy(&bits,
                      host + source.scale_offset + local + sizeof(float),
                      sizeof(bits));
          input_divisor = std::bit_cast<float>(
              static_cast<std::uint32_t>(bits) << 16U);
        }
        if (!std::isfinite(weight_divisor) || !(weight_divisor > 0.0F) ||
            !std::isfinite(input_divisor) || !(input_divisor > 0.0F))
          throw std::runtime_error("native NVFP4 global divisor is invalid");
        tensor->fp4_data =
            reinterpret_cast<const std::uint8_t*>(tensor->allocation);
        tensor->fp4_scales = reinterpret_cast<const std::uint8_t*>(
            tensor->allocation + scale_allocation_offset);
        tensor->nvfp4_weight_global_scale = 1.0F / weight_divisor;
        tensor->nvfp4_input_global_scale = input_divisor;
      } else {
      const auto padded = align32(source.shape.back());
      const auto expected_data = rows * padded / 2U;
      const auto expected_scales = rows * padded / 32U;
      if (source.data_bytes != expected_data ||
          source.scale_bytes != expected_scales)
        throw std::runtime_error("FP4 tensor storage geometry is inconsistent");
      tensor->fp4_data =
          reinterpret_cast<const std::uint8_t*>(tensor->allocation);
      tensor->fp4_scales = reinterpret_cast<const std::uint8_t*>(
          tensor->allocation + scale_allocation_offset);
      if (source.shape.size() >= 3U) {
        const auto row_count = checked_product(
            std::span(source.shape).first(source.shape.size() - 1U));
        const auto columns = source.shape.back();
        std::vector<float> decoded(
            static_cast<std::size_t>(row_count) * columns);
        const auto* packed = reinterpret_cast<const std::uint8_t*>(
            host + source.data_offset);
        const auto* scales = reinterpret_cast<const std::uint8_t*>(
            host + source.scale_offset);
        constexpr std::array<float, 8U> levels{
            0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
        for (std::uint64_t row = 0U; row < row_count; ++row) {
          const auto* row_data = packed + row * padded / 2U;
          const auto* row_scales = scales + row * padded / 32U;
          for (std::uint32_t column = 0U; column < columns; ++column) {
            const auto byte = row_data[column / 2U];
            const auto code = static_cast<std::uint8_t>(
                (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
            const auto magnitude = levels[code & 0x07U];
            const auto scale_code = row_scales[column / 32U];
            const auto scale = std::ldexp(
                1.0F, static_cast<int>(scale_code) - 127);
            decoded[static_cast<std::size_t>(row) * columns + column] =
                (code & 0x08U) != 0U ? -magnitude * scale
                                     : magnitude * scale;
          }
        }
        tensor->dequantized = device_allocate<float>(allocations_, decoded.size());
        cuda_check(cudaMemcpy(tensor->dequantized, decoded.data(),
                              decoded.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "upload dequantized convolution tensor");
      }
      }
    } else if (source.encoding == "MXFP6_E3M2") {
      if (source.quant_abi != er::kDenseRecordAbiMxfp6E3m2Block32 ||
          source.shape.size() != 2U)
        throw std::runtime_error("MXFP6 tensor has the wrong ABI");
      const auto rows = source.shape[0];
      const auto padded = align32(source.shape[1]);
      const auto expected_data =
          static_cast<std::uint64_t>(rows) * padded * 3U / 4U;
      const auto expected_scales =
          static_cast<std::uint64_t>(rows) * padded / 32U;
      if (source.data_bytes != expected_data ||
          source.scale_bytes != expected_scales)
        throw std::runtime_error(
            "MXFP6 tensor storage geometry is inconsistent");
      tensor->mxfp6_data =
          reinterpret_cast<const std::uint8_t*>(tensor->allocation);
      tensor->mxfp6_scales = reinterpret_cast<const std::uint8_t*>(
          tensor->allocation + scale_allocation_offset);
    } else if (source.encoding == "I8") {
      if (source.quant_abi != er::kExpertQuantAbiInt8PerRow ||
          source.shape.size() != 2U ||
          source.data_bytes != checked_product(source.shape) ||
          source.scale_bytes !=
              static_cast<std::uint64_t>(source.shape[0]) * sizeof(float))
        throw std::runtime_error("INT8 tensor storage geometry is inconsistent");
      tensor->int8_data =
          reinterpret_cast<const std::int8_t*>(tensor->allocation);
      tensor->int8_scales = reinterpret_cast<const float*>(
          tensor->allocation + scale_allocation_offset);
    } else if (source.encoding == "F32") {
      if (source.quant_abi != 0U || source.scale_bytes != 0U ||
          source.data_bytes != checked_product(source.shape) * sizeof(float))
        throw std::runtime_error("F32 tensor storage geometry is inconsistent");
      tensor->f32 = reinterpret_cast<const float*>(tensor->allocation);
    } else if (source.encoding == "BF16") {
      if (source.quant_abi != 0U || source.scale_bytes != 0U ||
          source.data_bytes != checked_product(source.shape) *
                                   sizeof(std::uint16_t))
        throw std::runtime_error("BF16 tensor storage geometry is inconsistent");
      tensor->bf16 = reinterpret_cast<const std::uint16_t*>(
          tensor->allocation);
    } else {
      throw std::runtime_error("dense FP4 provider rejects tensor encoding " +
                               source.encoding);
    }
    auto* result = tensor.get();
    if (!tensors_.emplace(source.name, std::move(tensor)).second)
      throw std::runtime_error("duplicate dense tensor upload");
    uploaded_tensor_bytes_ += source.data_bytes + source.scale_bytes;
    ++uploaded_tensor_count_;
    if (uploaded_tensor_bytes_ >= next_upload_progress_bytes_) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - tensor_upload_started_);
      std::cerr << "startup_phase=tensor_upload_progress tensors="
                << uploaded_tensor_count_ << " bytes=" << uploaded_tensor_bytes_
                << " elapsed_ms=" << elapsed.count() << '\n'
                << std::flush;
      do {
        next_upload_progress_bytes_ += 1ULL << 30U;
      } while (uploaded_tensor_bytes_ >= next_upload_progress_bytes_);
    }
    return *result;
  }

  HostTensor& ensure_host_tensor(const er::ImmutableModelTensor& source) {
    const auto retained = host_tensors_.find(source.name);
    if (retained != host_tensors_.end()) return *retained->second;
    if (!source.value.valid() ||
        source.value.abi != "artifact.dense-record.v1" ||
        source.value.memory_domain != "host.mmap.readonly" ||
        source.data_offset > source.value.bytes ||
        source.data_bytes > source.value.bytes - source.data_offset ||
        source.scale_offset > source.value.bytes ||
        source.scale_bytes > source.value.bytes - source.scale_offset)
      throw std::runtime_error("invalid host-mapped tensor " + source.name);
    if (source.encoding == "FP4_E2M1") {
      if (source.quant_abi != er::kExpertQuantAbiFp4Block32 ||
          source.shape.size() != 2U)
        throw std::runtime_error("host FP4 tensor has the wrong ABI");
      const auto rows = source.shape[0];
      const auto padded = align32(source.shape[1]);
      if (source.data_bytes != static_cast<std::uint64_t>(rows) * padded / 2U ||
          source.scale_bytes !=
              static_cast<std::uint64_t>(rows) * padded / 32U)
        throw std::runtime_error("host FP4 tensor geometry is inconsistent");
    } else if (source.encoding == "I8") {
      if (source.quant_abi != er::kExpertQuantAbiInt8PerRow ||
          source.shape.size() != 2U ||
          source.data_bytes != checked_product(source.shape) ||
          source.scale_bytes !=
              static_cast<std::uint64_t>(source.shape[0]) * sizeof(float))
        throw std::runtime_error("host INT8 tensor geometry is inconsistent");
    } else if (source.encoding == "I64") {
      if (source.quant_abi != 0U || source.scale_bytes != 0U ||
          source.data_bytes != checked_product(source.shape) *
                                   sizeof(std::int64_t))
        throw std::runtime_error("host I64 tensor geometry is inconsistent");
    } else {
      throw std::runtime_error("unsupported host-mapped tensor encoding " +
                               source.encoding);
    }
    auto tensor = std::make_unique<HostTensor>();
    tensor->name = source.name;
    tensor->encoding = source.encoding;
    tensor->quant_abi = source.quant_abi;
    tensor->shape = source.shape;
    tensor->data = source.value.data + source.data_offset;
    tensor->data_bytes = source.data_bytes;
    tensor->scales = source.value.data + source.scale_offset;
    tensor->scale_bytes = source.scale_bytes;
    auto* result = tensor.get();
    if (!host_tensors_.emplace(source.name, std::move(tensor)).second)
      throw std::runtime_error("duplicate host-mapped tensor");
    return *result;
  }

  const DeviceTensor& binding(const PreparedOperation& operation,
                              std::string_view role) const {
    const auto found = operation.tensors.find(role);
    if (found == operation.tensors.end() || found->second == nullptr)
      throw std::runtime_error("missing tensor role " + std::string(role));
    return *found->second;
  }

  const HostTensor& host_binding(const PreparedOperation& operation,
                                 std::string_view role) const {
    const auto found = operation.host_tensors.find(role);
    if (found == operation.host_tensors.end() || found->second == nullptr)
      throw std::runtime_error("missing host tensor role " +
                               std::string(role));
    return *found->second;
  }

  static void expect_shape(const DeviceTensor& tensor,
                           std::initializer_list<std::uint32_t> shape,
                           std::string_view encoding) {
    if (tensor.encoding != encoding ||
        tensor.shape != std::vector<std::uint32_t>(shape))
      throw std::runtime_error("tensor role has an incompatible shape/encoding");
  }

  static void expect_shape(const HostTensor& tensor,
                           std::initializer_list<std::uint32_t> shape,
                           std::string_view encoding) {
    if (tensor.encoding != encoding ||
        tensor.shape != std::vector<std::uint32_t>(shape))
      throw std::runtime_error(
          "host tensor role has an incompatible shape/encoding");
  }

  static bool is_quantized_matrix(const DeviceTensor& tensor) noexcept {
    return tensor.shape.size() == 2U &&
           (tensor.encoding == "FP4_E2M1" ||
            tensor.encoding == "MXFP6_E3M2" || tensor.encoding == "I8" ||
            tensor.encoding == "BF16");
  }

  static void expect_quantized_shape(
      const DeviceTensor& tensor,
      std::initializer_list<std::uint32_t> shape) {
    if (!is_quantized_matrix(tensor) ||
        tensor.shape != std::vector<std::uint32_t>(shape))
      throw std::runtime_error(
          "tensor role has an incompatible quantized shape");
  }

  static void expect_quantized_shape(
      const HostTensor& tensor,
      std::initializer_list<std::uint32_t> shape) {
    if ((tensor.encoding != "FP4_E2M1" && tensor.encoding != "I8") ||
        tensor.shape.size() != 2U ||
        tensor.shape != std::vector<std::uint32_t>(shape))
      throw std::runtime_error(
          "host tensor role has an incompatible quantized shape");
  }

  void validate_operation(PreparedOperation& operation);
  void validate_exact(PreparedOperation& operation);
  void initialize_execution();
  void allocate_workspace();
  void allocate_state();
  void release_request_state(RequestState& state) noexcept;
  void release_slot(std::uint32_t slot) noexcept;
  [[nodiscard]] er::ExecutionValue device_hidden_value(
      std::uint32_t rows) const;
  [[nodiscard]] er::ExecutionValue device_value(
      std::string_view abi, const void* pointer,
      std::uint64_t bytes) const;
  void require_device_value(const er::ExecutionValue& value,
                            std::string_view abi, const void* pointer,
                            std::uint64_t bytes) const;
  [[nodiscard]] er::OperationExecutionHandle completed_operation(
      er::OperationExecutionResult result) const;
  [[nodiscard]] er::ExactDecodeExecutionHandle completed_exact(
      er::ExactDecodeExecutionResult result) const;
  [[nodiscard]] std::optional<er::OperationExecutionResult>
  poll_program_sequence(const std::shared_ptr<SequenceState>& sequence);
  [[nodiscard]] bool stage_operation_weights(
      const PreparedOperation& operation);
  [[nodiscard]] bool ensure_staged_dense_weights();
  void release_staged_dense_weights() noexcept;
  [[nodiscard]] bool activate_staged_weights(
      const PreparedOperation& operation);
  void deactivate_staged_weights() noexcept;
  [[nodiscard]] std::uint64_t pending_sequence_kv_bytes(
      const SequenceState& sequence,
      const PreparedOperation& operation) const;
  [[nodiscard]] bool sequence_staging_has_headroom(
      SequenceState& sequence, const PreparedOperation& operation);
  void ensure_logits_capacity(std::uint32_t rows);
  void ensure_sequence_workspace();
  void release_sequence_workspace() noexcept;
  [[nodiscard]] const er::ExecutionValue& invocation_input(
      const PreparedOperation& operation,
      const er::OperationInvocation& invocation,
      std::string_view port) const;
  [[nodiscard]] std::uint32_t require_hidden_value(
      const er::ExecutionValue& value) const;
  static GpuPhase gpu_phase(Kernel kernel);
  std::size_t begin_gpu_phase(GpuPhase phase);
  void end_gpu_phase(std::size_t event);
  void collect_gpu_phases();
  void normalize_rows(const float* input, const float* weight, float* output,
                      std::uint32_t rows);
  void normalize_operation_input(const PreparedOperation& operation,
                                 const float* input, float* output,
                                 std::uint32_t rows);
  void finish_attention_block(const PreparedOperation& operation,
                              std::uint32_t rows);
  void run_embedding(const PreparedOperation& operation,
                     const std::uint32_t* tokens, std::uint32_t rows);
  void run_hyper_read(const PreparedOperation& operation,
                      std::uint32_t rows, bool reduce);
  void run_ple(const PreparedOperation& operation, RequestState& state,
               std::span<const std::uint32_t> tokens, std::uint32_t rows);
  void quantize_rows(const float* input, std::uint32_t rows,
                     std::uint32_t columns);
  void prepare_dense_activation(const float* input, std::uint32_t rows,
                                std::uint32_t columns);
  void project_quantized(const DeviceTensor& weight, const float* input,
                         float* output, std::uint32_t rows);
  void project(const DeviceTensor& weight, const float* input, float* output,
               std::uint32_t rows);
  void project_vision(const DeviceTensor& weight, const float* input,
                      float* output, std::uint32_t rows);
  void run_vision(const PreparedOperation& operation,
                  const MediaPayload& media, float* tile_hidden,
                  std::uint32_t prompt_rows, std::uint32_t tile_first,
                  std::uint32_t tile_rows, bool prepare);
  void ensure_page(std::uint32_t slot, std::uint32_t cache_position);
  void checkpoint_window_layer(std::uint32_t slot,
                               std::uint32_t window_layer);
  void checkpoint_window_state(std::uint32_t slot);
  void restore_window_checkpoint(std::uint32_t slot);
  void ensure_host_kv_page(RequestState& state, std::uint32_t cache_position);
  [[nodiscard]] bool ensure_target_mirror_page(
      RequestState& state, std::uint32_t cache_position);
  void release_target_mirror_pages(std::uint32_t slot) noexcept;
  void trim_target_mirror(RequestState& state,
                          std::uint32_t populated_tokens) noexcept;
  void restore_target_mirror(RequestState& state,
                             std::uint32_t slot) noexcept;
  void trim_host_kv(RequestState& state, std::uint32_t populated_tokens)
      noexcept;
  void release_host_kv(RequestState& state) noexcept;
  [[nodiscard]] std::pair<std::uint16_t*, std::uint16_t*> host_kv_layer_page(
      const RequestState& state, std::uint32_t full_attention_slot,
      std::uint32_t page_index) const;
  void stage_qsa_selected_host(const RequestState& state,
                               std::uint32_t full_attention_slot,
                               std::span<const std::uint32_t> selected);
  void run_full_attention(const PreparedOperation& operation,
                          RequestState& state,
                          std::span<const std::uint32_t> cache_positions,
                          std::span<const std::uint32_t> rotary_positions,
                          std::uint32_t rows,
                          std::uint32_t full_attention_slot,
                          std::span<const std::uint32_t> mrope_positions = {});
  void run_recurrent_attention(const PreparedOperation& operation,
                               std::uint32_t slot, std::uint32_t rows,
                               RecurrentCheckpointMode checkpoint_mode);
  void run_router(const PreparedOperation& operation, std::uint32_t rows);
  void run_routed_moe(const PreparedOperation& operation,
                      std::uint32_t rows);
  void run_ffn(const PreparedOperation& operation, std::uint32_t rows);
  std::vector<std::uint32_t> run_head(const PreparedOperation& operation,
                                      std::uint32_t rows,
                                      const er::ProgramRequestContext* request,
                                      RequestState* state,
                                      std::uint32_t sample_position,
                                      bool terminal_only = false);
  std::vector<std::uint32_t> run_target(
      RequestState& state, std::span<const std::uint32_t> tokens,
      std::span<const std::uint32_t> positions,
      RecurrentCheckpointMode checkpoint_mode);
  std::optional<MtpPrediction> run_mtp(
      RequestState& state, std::span<const std::uint32_t> tokens,
      const float* previous_hidden,
      std::span<const std::uint32_t> rotary_positions, bool produce_logits,
      const er::ProgramRequestContext* request = nullptr,
      const std::uint8_t* presence = nullptr,
      std::uint32_t sample_position = 0U,
      std::uint32_t random_stream = 1U);
  void extend_mtp_rollout(RequestState& state,
                          const er::ProgramRequestContext& request,
                          MtpPrediction first,
                          std::uint32_t context_limit);
  er::SamplingDistribution distribution_from_logits(
      float* row_logits, std::uint32_t vocabulary,
      const er::ProgramRequestContext& request, const std::uint8_t* presence);
  void trace_sampling(const er::ProgramRequestContext& request,
                      std::uint32_t position, std::uint32_t token,
                      std::string_view mode,
                      const er::SamplingDistribution* target,
                      const er::SamplingDistribution* draft = nullptr);
  void project_prefix(const DeviceTensor& weight, const float* input,
                      float* output, std::uint32_t matrix_rows);
  void restore_recurrent_checkpoint(std::uint32_t slot);
  void restore_speculative_recurrent_checkpoint(std::uint32_t slot,
                                                std::uint32_t row);
  float* slot_last_hidden(std::uint32_t slot) const;
  float* recurrent_conv(std::uint32_t recurrent_slot,
                        std::uint32_t request_slot) const;
  float* recurrent_matrix(std::uint32_t recurrent_slot,
                          std::uint32_t request_slot) const;

  std::shared_ptr<er::ModelArtifact> artifact_;
  er::ModelDescriptor descriptor_;
  std::shared_ptr<er::MappedModelTensorStore> tensor_store_;
  std::uint32_t max_context_{};
  std::uint32_t capacity_{};
  std::uint64_t ram_cache_bytes_{};
  std::uint64_t vram_cache_bytes_{};
  std::uint64_t parking_ram_capacity_bytes_{};
  std::uint64_t kv_cache_bytes_{};
  std::uint32_t kv_page_tokens_{};
  std::uint32_t workspace_rows_{kDefaultWorkspaceRows};
  std::uint64_t workspace_preflight_free_bytes_{};
  std::uint64_t workspace_postallocation_free_bytes_{};
  TargetKvEncoding target_kv_encoding_{TargetKvEncoding::artifact_native};
  bool capacity_placement_{};
  bool fit_routed_vram_{};
  bool device_resident_fp16_kv_{};
  bool compact_flash_prefill_{};
  bool large_exact_prefill_workspace_{};
  std::uint32_t hidden_size_{};
  std::uint32_t vocabulary_size_{};
  std::uint32_t query_heads_{};
  std::uint32_t kv_heads_{};
  std::uint32_t head_dim_{};
  std::uint32_t rotary_dimension_{};
  std::uint32_t key_heads_{};
  std::uint32_t value_heads_{};
  std::uint32_t key_head_dim_{};
  std::uint32_t value_head_dim_{};
  std::uint32_t conv_kernel_{};
  std::uint32_t mamba_heads_{};
  std::uint32_t mamba_head_dim_{};
  std::uint32_t mamba_state_size_{};
  std::uint32_t mamba_state_groups_{};
  std::uint32_t mamba_conv_size_{};
  std::uint32_t mamba_conv_kernel_{};
  std::size_t recurrent_conv_values_{};
  std::size_t recurrent_matrix_values_{};
  bool split_recurrent_enabled_{};
  bool mamba2_enabled_{};
  bool standard_attention_enabled_{};
  bool mla_enabled_{};
  bool relu2_router_enabled_{};
  bool zero_centered_norm_{};
  bool activation_bf16_{};
  bool dense_activation_input_bf16_{};
  bool hyper_enabled_{};
  bool qsa_enabled_{};
  bool qsa_tiered_{};
  bool ple_enabled_{};
  std::uint32_t q_lora_rank_{};
  std::uint32_t kv_lora_rank_{};
  std::uint32_t qk_nope_head_dim_{};
  std::uint32_t qk_rope_head_dim_{};
  std::uint32_t v_head_dim_{};
  float rope_factor_{};
  float rope_beta_fast_{};
  float rope_beta_slow_{};
  std::uint32_t rope_original_context_{};
  float llama4_scaling_beta_{};
  float mla_attention_scale_{};
  std::uint32_t hyper_count_{};
  std::uint32_t hyper_width_{};
  std::uint32_t hyper_lowrank_{};
  std::uint32_t qsa_index_heads_{};
  std::uint32_t qsa_index_head_dim_{};
  std::uint32_t qsa_token_budget_{};
  std::uint32_t qsa_compress_ratio_{};
  std::uint32_t qsa_layers_{};
  std::uint32_t qsa_index_page_offset_{};
  std::uint64_t qsa_index_device_page_bytes_{};
  std::uint32_t ple_ngram_size_{};
  std::uint32_t ple_heads_per_ngram_{};
  std::uint32_t ple_head_count_{};
  std::uint32_t ple_head_width_{};
  std::uint32_t ple_embedding_width_{};
  std::uint32_t ple_convolution_kernel_{};
  std::uint32_t ple_shard_count_{};
  std::uint32_t ple_rows_per_shard_{};
  std::uint32_t ple_eos_token_id_{};
  std::uint32_t ple_layers_{};
  std::size_t ple_conv_state_values_{};
  std::uint32_t mtp_layers_{};
  std::uint32_t exact_decode_abi_{};
  std::uint32_t draft_depth_{};
  std::uint32_t draft_vocabulary_size_{};
  bool mtp_q8_kv_{};
  std::uint32_t target_full_layers_{};
  std::uint32_t global_attention_layers_{};
  std::uint32_t window_attention_layers_{};
  std::uint32_t window_tokens_{};
  std::uint32_t window_pages_per_slot_{};
  std::uint32_t recurrent_layers_{};
  std::uint32_t intermediate_size_{};
  std::uint32_t expert_count_{};
  std::uint32_t route_width_{};
  std::uint32_t expert_width_{};
  std::uint32_t shared_intermediate_size_{};
  bool vision_enabled_{};
  std::array<std::uint32_t, 3U> mrope_sections_{};
  std::uint32_t vision_depth_{};
  std::uint32_t vision_hidden_size_{};
  std::uint32_t vision_intermediate_size_{};
  std::uint32_t vision_heads_{};
  std::uint32_t vision_head_dim_{};
  std::uint32_t vision_position_embeddings_{};
  std::uint32_t vision_grid_side_{};
  std::uint32_t vision_channels_{};
  std::uint32_t vision_patch_size_{};
  std::uint32_t vision_temporal_patch_size_{};
  std::uint32_t vision_spatial_merge_size_{};
  std::uint32_t vision_output_size_{};
  std::uint32_t vision_patch_dimension_{};
  std::uint32_t vision_merged_width_{};
  float vision_epsilon_{};
  float vision_rope_theta_{};
  float epsilon_{};
  float rope_theta_{};
  std::uint64_t kv_page_bytes_{};
  std::uint64_t target_kv_page_bytes_{};
  std::uint64_t host_fp16_target_page_bytes_{};
  std::uint64_t window_kv_page_bytes_{};
  std::uint64_t window_kv_bytes_per_slot_{};
  std::uint64_t mtp_kv_page_offset_{};
  std::uint64_t mtp_kv_page_bytes_{};
  std::uint64_t service_kv_page_bytes_{};
  std::uint32_t maximum_pages_per_slot_{};
  std::uint64_t kv_page_capacity_{};
  std::uint64_t service_kv_page_capacity_{};
  std::uint64_t target_mirror_page_capacity_{};
  std::uint64_t host_kv_bytes_{};
  std::uint64_t target_mirror_attention_calls_{};
  std::uint64_t target_host_attention_calls_{};
  std::uint64_t target_mirror_spills_{};
  std::uint64_t target_mirror_restore_bytes_{};
  std::uint64_t parked_request_bytes_{};
  std::uint64_t parked_session_bytes_{};
  std::uint64_t park_calls_{};
  std::uint64_t restore_calls_{};
  std::uint64_t park_device_to_host_bytes_{};
  std::uint64_t restore_host_to_device_bytes_{};
  std::uint64_t qsa_host_commit_bytes_{};
  std::uint64_t qsa_selected_host_bytes_{};
  std::uint64_t qsa_selected_host_tokens_{};
  std::uint64_t qsa_selected_host_calls_{};
  std::uint64_t qsa_device_staged_calls_{};
  std::uint64_t qsa_score_device_to_host_bytes_{};
  std::uint32_t staged_device_slot_{kNoSlot};
  std::uint32_t staged_device_layer_{kNoSlot};
  std::uint32_t staged_device_context_{};
  std::uint64_t uploaded_tensor_bytes_{};
  std::uint64_t uploaded_tensor_count_{};
  std::uint64_t next_upload_progress_bytes_{1ULL << 30U};
  std::chrono::steady_clock::time_point tensor_upload_started_{
      std::chrono::steady_clock::now()};
  std::uint64_t program_steps_{};
  std::uint64_t prefill_batches_{};
  std::uint64_t prefill_tokens_{};
  std::uint64_t exact_sync_batches_{};
  std::uint64_t exact_sync_tokens_{};
  std::uint64_t exact_calls_{};
  std::uint64_t exact_target_batches_{};
  std::uint64_t accepted_drafts_{};
  std::array<std::uint64_t, kMaximumExactDecodeRows> accepted_depth_calls_{};
  std::uint64_t speculative_recurrent_checkpoint_bytes_{};
  std::uint64_t speculative_recurrent_restores_{};
  std::uint64_t fused_multiquery_attention_calls_{};
  std::uint64_t program_sequence_batches_{};
  std::uint64_t program_sequence_tokens_{};
  std::uint64_t program_sequence_tiles_{};
  std::uint64_t program_sequence_device_bytes_{};
  std::uint64_t program_sequence_host_bytes_{};
  std::uint64_t sampling_gpu_calls_{};
  std::uint64_t sampling_host_calls_{};
  std::uint64_t sampling_logit_transfer_bytes_{};
  std::ofstream sampling_trace_stream_;
  std::uint64_t sampling_trace_records_{};
  std::uint64_t vision_batches_{};
  std::uint64_t vision_images_{};
  std::uint64_t vision_patches_{};
  std::array<std::uint64_t, static_cast<std::size_t>(GpuPhase::count)>
      gpu_phase_ns_{};
  std::uint64_t gpu_measured_batches_{};
  bool profile_gpu_phases_{};
  std::vector<GpuEventPair> gpu_event_pool_;
  std::size_t active_gpu_events_{};
  std::vector<std::uint32_t> full_attention_slots_;
  std::vector<std::uint32_t> global_kv_slots_;
  std::vector<std::uint32_t> window_kv_slots_;
  std::vector<std::uint32_t> recurrent_slots_;
  std::vector<std::uint32_t> qsa_index_slots_;
  std::vector<std::uint32_t> ple_slots_;
  std::vector<std::shared_ptr<PreparedOperation>> prepared_target_;
  std::shared_ptr<PreparedOperation> exact_;
  std::shared_ptr<PreparedOperation> mtp_attention_;
  std::shared_ptr<PreparedOperation> mtp_ffn_;
  std::unique_ptr<Fp4RoutedExperts> routed_experts_;
  std::map<std::string, std::unique_ptr<DeviceTensor>, std::less<>> tensors_;
  std::map<std::string, std::unique_ptr<HostTensor>, std::less<>> host_tensors_;
  std::vector<void*> allocations_;
  std::shared_ptr<const void> device_lifetime_;
  std::mutex mutex_;
  std::vector<bool> slot_in_use_;
  std::vector<std::int32_t> slot_rope_deltas_;
  bool initialized_{};
  const PreparedOperation* staged_operation_{};
  std::map<const DeviceTensor*, StagedDenseWeight> staged_weight_bindings_;
  std::map<const DeviceTensor*, StagedDenseWeight> active_staged_weights_;
  std::size_t staged_dense_weight_capacity_bytes_{};
  std::size_t staged_dense_input_capacity_bytes_{};
  std::size_t staged_int8_weight_capacity_values_{};
  std::size_t staged_score_capacity_values_{};
  std::size_t staged_probability_capacity_values_{};
  std::uint64_t staged_dense_weight_decode_bytes_{};
  std::uint64_t staged_dense_weight_gemm_calls_{};
  std::uint64_t staged_dense_weight_low_memory_fallbacks_{};
  std::uint32_t logits_capacity_rows_{};

  // Workspace and state are declared below with the execution methods.
  float* hidden_{};
  float* sequence_hidden_{};
  std::size_t sequence_hidden_values_{};
  float* sequence_hyper_{};
  std::size_t sequence_hyper_values_{};
  float* sequence_injection_{};
  std::size_t sequence_injection_values_{};
  std::uint32_t sequence_tile_rows_{};
  float* normalized_{};
  float* residual_{};
  float* hyper_{};
  float* hyper_normalized_{};
  float* hyper_lowrank_values_{};
  float* hyper_mix_{};
  float* hyper_injection_{};
  float* ple_embeddings_{};
  float* ple_key_{};
  float* ple_key_norm_{};
  float* ple_query_norm_{};
  float* ple_value_{};
  float* ple_gated_{};
  float* ple_conv_norm_{};
  float* ple_conv_output_{};
  float* qsa_projected_{};
  float* qsa_scores_{};
  std::uint32_t* qsa_selected_{};
  float* query_gate_{};
  std::int8_t* attention_q8_queries_{};
  float* attention_query_scales_{};
  float* mla_query_rank_{};
  float* mla_latent_{};
  float* mla_latent_query_{};
  float* key_{};
  float* value_{};
  float* attention_{};
  float* projected_qkv_{};
  float* projected_z_{};
  float* projected_b_{};
  float* projected_a_{};
  float* conv_output_{};
  float* delta_output_{};
  float* gate_{};
  float* up_{};
  float* intermediate_{};
  float* shared_output_{};
  float* shared_scalar_{};
  float* router_logits_{};
  float* routing_scores_{};
  std::uint32_t* routing_indices_{};
  float* moe_intermediate_{};
  float* moe_selection_output_{};
  float* moe_output_{};
  std::int8_t* moe_q8_intermediate_{};
  float* moe_q8_intermediate_scales_{};
  float* moe_nvfp4_gate_input_{};
  float* moe_nvfp4_up_input_{};
  float* moe_nvfp4_down_input_{};
  float* logits_{};
  float* vision_pixels_{};
  float* vision_hidden_{};
  float* vision_normalized_{};
  float* vision_qkv_{};
  float* vision_attention_{};
  float* vision_intermediate_{};
  float* vision_merger_{};
  float* vision_output_{};
  std::uint32_t* vision_positions_{};
  std::uint32_t* vision_segment_first_{};
  std::uint32_t* vision_segment_last_{};
  std::uint32_t* vision_interpolation_indices_{};
  float* vision_interpolation_weights_{};
  std::uint32_t* output_tokens_{};
  float* sampling_top_logits_{};
  std::uint32_t* sampling_top_tokens_{};
  float* sampling_top_partial_logits_{};
  std::uint32_t* sampling_top_partial_tokens_{};
  std::size_t sampling_top_partial_items_{};
  std::uint8_t* sampling_presence_{};
  std::uint8_t* sampling_proposal_presence_{};
  std::int8_t* q8_{};
  float* q8_scales_{};
  float* nvfp4_input_{};
  std::uint16_t* staged_dense_weights_{};
  std::uint16_t* staged_dense_input_{};
  float* staged_int8_weights_{};
  float* partial_maxima_{};
  float* partial_sums_{};
  float* partial_outputs_{};
  std::uint32_t attention_maximum_splits_{};
  std::uint16_t* staged_queries_{};
  std::uint16_t* staged_raw_keys_{};
  std::uint16_t* staged_raw_values_{};
  std::uint16_t* staged_device_keys_{};
  std::uint16_t* staged_device_values_{};
  std::uint16_t* staged_keys_{};
  std::uint16_t* staged_values_{};
  float* staged_scores_{};
  std::uint16_t* staged_probabilities_{};
  float* staged_accumulator_{};
  std::uint32_t staged_split_tokens_{};
  float* mtp_embedding_{};
  float* mtp_embedding_norm_{};
  float* mtp_hidden_norm_{};
  float* mtp_fusion_input_{};
  float* slot_target_hidden_batch_{};
  float* slot_last_hidden_{};
  float* slot_mtp_last_hidden_{};
  float* mtp_rollout_previous_hidden_{};
  std::vector<float*> recurrent_conv_state_;
  std::vector<float*> recurrent_matrix_state_;
  std::vector<float*> recurrent_conv_checkpoint_;
  std::vector<float*> recurrent_matrix_checkpoint_;
  std::vector<float*> recurrent_conv_speculative_checkpoint_;
  std::vector<float*> recurrent_matrix_speculative_checkpoint_;
  std::vector<float*> recurrent_conv_retention_checkpoint_;
  std::vector<float*> recurrent_matrix_retention_checkpoint_;
  std::vector<float*> ple_conv_state_;
  std::vector<float*> ple_conv_retention_checkpoint_;
  std::vector<float> ple_host_embeddings_;
  std::vector<float> qsa_host_scores_;
  std::vector<std::uint32_t> qsa_host_selected_;
  std::uint16_t* qsa_host_selected_keys_{};
  std::uint16_t* qsa_host_selected_values_{};
  float* slot_retention_last_hidden_{};
  void** device_page_table_{};
  void** device_mtp_page_table_{};
  void** device_target_mirror_page_table_{};
  void** device_window_page_table_{};
  std::byte* window_kv_{};
  std::byte* window_retention_kv_{};
  std::vector<std::vector<void*>> slot_pages_;
  std::vector<void*> free_kv_pages_;
  std::vector<void*> all_kv_pages_;
  std::vector<std::vector<void*>> slot_target_mirror_pages_;
  std::vector<void*> free_target_mirror_pages_;
  std::vector<void*> all_target_mirror_pages_;
  cudaStream_t parking_stream_{};
};

void DenseFp4Provider::validate_operation(PreparedOperation& operation) {
  switch (operation.kernel) {
    case Kernel::embedding:
      expect_quantized_shape(binding(operation, "weight"),
                             {vocabulary_size_, hidden_size_});
      if (operation.capability == "embedding.lookup.bfloat16.v1" &&
          binding(operation, "weight").encoding != "BF16")
        throw std::runtime_error("BF16 embedding has the wrong encoding");
      if (operation.capability ==
              "embedding.lookup.mxfp6-e3m2-block32.v1" &&
          binding(operation, "weight").quant_abi !=
              er::kDenseRecordAbiMxfp6E3m2Block32)
        throw std::runtime_error("MXFP6 embedding has the wrong ABI");
      if (operation.abi_version >= 2U) {
        if (parameter_u32(operation.parameters, "output_norm_mode") != 1U)
          throw std::runtime_error("unsupported embedding ABI 2 norm mode");
        static_cast<void>(parameter_f32(
            operation.parameters, "output_norm_epsilon_f32_bits"));
      }
      break;
    case Kernel::hyper_initialize:
      if (!hyper_enabled_ || !operation.tensors.empty() ||
          !operation.host_tensors.empty() ||
          parameter_u32(operation.parameters, "stream_count") != hyper_count_)
        throw std::runtime_error("Hyper initialize contract is invalid");
      break;
    case Kernel::hyper_read:
    case Kernel::hyper_reduce:
      if (!hyper_enabled_ ||
          parameter_u32(operation.parameters, "stream_count") != hyper_count_)
        throw std::runtime_error("Hyper read geometry is invalid");
      expect_shape(binding(operation, "norm"), {hyper_width_}, "F32");
      expect_quantized_shape(binding(operation, "mix_down"),
                             {hyper_lowrank_, hyper_width_});
      expect_quantized_shape(binding(operation, "mix_up"),
                             {hyper_width_, hyper_lowrank_});
      if (operation.kernel == Kernel::hyper_read)
        expect_quantized_shape(binding(operation, "inject"),
                               {hyper_count_, hyper_width_});
      else if (operation.tensors.size() != 3U)
        throw std::runtime_error("Hyper reduce has unexpected tensors");
      static_cast<void>(parameter_f32(
          operation.parameters, "norm_epsilon_f32_bits"));
      break;
    case Kernel::hyper_inject:
      if (!hyper_enabled_ || !operation.tensors.empty() ||
          !operation.host_tensors.empty() ||
          parameter_u32(operation.parameters, "stream_count") != hyper_count_)
        throw std::runtime_error("Hyper injection contract is invalid");
      break;
    case Kernel::ple: {
      if (!ple_enabled_ || operation.ple_slot == kNoSlot ||
          operation.host_tensors.size() != 3U + ple_shard_count_)
        throw std::runtime_error("PLE operation has no artifact placement");
      expect_quantized_shape(binding(operation, "key_projection"),
                             {hyper_width_, ple_embedding_width_});
      expect_quantized_shape(binding(operation, "value_projection"),
                             {hidden_size_, ple_embedding_width_});
      for (const auto role : {"key_norm", "query_norm", "convolution_norm"})
        expect_shape(binding(operation, role), {hyper_width_}, "F32");
      auto& convolution =
          const_cast<DeviceTensor&>(binding(operation, "convolution"));
      expect_shape(convolution,
                   {hyper_width_, 1U, ple_convolution_kernel_},
                   "FP4_E2M1");
      if (!convolution.dequantized)
        throw std::runtime_error("PLE convolution was not decoded");
      expect_shape(host_binding(operation, "layer_multipliers"),
                   {ple_ngram_size_}, "I64");
      expect_shape(host_binding(operation, "head_vocab_sizes"),
                   {ple_head_count_}, "I64");
      expect_shape(host_binding(operation, "head_offsets"),
                   {ple_head_count_}, "I64");
      for (std::uint32_t shard = 0U; shard < ple_shard_count_; ++shard)
        expect_quantized_shape(
            host_binding(operation, std::string("embedding_shard.") +
                                        std::to_string(shard)),
            {ple_rows_per_shard_, ple_head_width_});
      break;
    }
    case Kernel::vision: {
      if (!vision_enabled_ ||
          operation.logical_layer != er::kModelLevelOperationLayer)
        throw std::runtime_error("vision operation has no artifact geometry");
      expect_shape(binding(operation, "patch_projection"),
                   {vision_hidden_size_, vision_channels_,
                    vision_temporal_patch_size_, vision_patch_size_,
                    vision_patch_size_}, "FP4_E2M1");
      if (!binding(operation, "patch_projection").dequantized)
        throw std::runtime_error("vision patch projection was not decoded");
      expect_shape(binding(operation, "patch_bias"),
                   {vision_hidden_size_}, "F32");
      expect_shape(binding(operation, "position_embedding"),
                   {vision_position_embeddings_, vision_hidden_size_},
                   "FP4_E2M1");
      for (std::uint32_t layer = 0U; layer < vision_depth_; ++layer) {
        const auto role = [layer](std::string_view suffix) {
          return std::string("block.") + std::to_string(layer) + "." +
                 std::string(suffix);
        };
        for (const auto suffix : {"norm1_weight", "norm1_bias",
                                  "norm2_weight", "norm2_bias"})
          expect_shape(binding(operation, role(suffix)),
                       {vision_hidden_size_}, "F32");
        expect_shape(binding(operation, role("qkv_projection")),
                     {3U * vision_hidden_size_, vision_hidden_size_},
                     "FP4_E2M1");
        expect_shape(binding(operation, role("qkv_bias")),
                     {3U * vision_hidden_size_}, "F32");
        expect_shape(binding(operation, role("attention_projection")),
                     {vision_hidden_size_, vision_hidden_size_},
                     "FP4_E2M1");
        expect_shape(binding(operation, role("attention_bias")),
                     {vision_hidden_size_}, "F32");
        expect_shape(binding(operation, role("mlp_fc1")),
                     {vision_intermediate_size_, vision_hidden_size_},
                     "FP4_E2M1");
        expect_shape(binding(operation, role("mlp_fc1_bias")),
                     {vision_intermediate_size_}, "F32");
        expect_shape(binding(operation, role("mlp_fc2")),
                     {vision_hidden_size_, vision_intermediate_size_},
                     "FP4_E2M1");
        expect_shape(binding(operation, role("mlp_fc2_bias")),
                     {vision_hidden_size_}, "F32");
      }
      for (const auto suffix : {"merger_norm_weight", "merger_norm_bias"})
        expect_shape(binding(operation, suffix), {vision_hidden_size_}, "F32");
      expect_shape(binding(operation, "merger_fc1"),
                   {vision_merged_width_, vision_merged_width_}, "FP4_E2M1");
      expect_shape(binding(operation, "merger_fc1_bias"),
                   {vision_merged_width_}, "F32");
      expect_shape(binding(operation, "merger_fc2"),
                   {vision_output_size_, vision_merged_width_}, "FP4_E2M1");
      expect_shape(binding(operation, "merger_fc2_bias"),
                   {vision_output_size_}, "F32");
      break;
    }
    case Kernel::full_attention: {
      if (operation.full_attention_slot == kNoSlot)
        throw std::runtime_error("full attention has no artifact slot");
      if (operation.capability ==
          "block.mla.causal.latent-kv.bfloat16.v1") {
        if (!mla_enabled_)
          throw std::runtime_error("compressed MLA is not enabled");
        expect_shape(binding(operation, "input_norm"), {hidden_size_},
                     "BF16");
        expect_shape(binding(operation, "query_a"),
                     {q_lora_rank_, hidden_size_}, "BF16");
        expect_shape(binding(operation, "query_a_norm"), {q_lora_rank_},
                     "BF16");
        expect_shape(binding(operation, "query_b"),
                     {query_heads_ *
                          (qk_nope_head_dim_ + qk_rope_head_dim_),
                      q_lora_rank_}, "BF16");
        expect_shape(binding(operation, "kv_a"),
                     {kv_lora_rank_ + qk_rope_head_dim_, hidden_size_},
                     "BF16");
        expect_shape(binding(operation, "kv_a_norm"), {kv_lora_rank_},
                     "BF16");
        expect_shape(binding(operation, "kv_b"),
                     {query_heads_ * (qk_nope_head_dim_ + v_head_dim_),
                      kv_lora_rank_}, "BF16");
        expect_shape(binding(operation, "output"),
                     {hidden_size_, query_heads_ * v_head_dim_}, "BF16");
        break;
      }
      const auto qsa = operation.capability ==
          "block.sparse-attention.qsa.output-gated.v1";
      if (!qsa)
        expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
      else if (!qsa_enabled_ || operation.qsa_index_slot == kNoSlot ||
               (!device_resident_fp16_kv() && !qsa_tiered_))
        throw std::runtime_error("QSA has no exact cache placement");
      const auto separated_gate = operation.abi_version >= 2U;
      expect_quantized_shape(
          binding(operation, "query_projection"),
          {((operation.capability == "block.full-attention.standard-gqa.v1" ||
             operation.capability ==
                 "block.full-attention.standard-gqa-no-position.v1" ||
             separated_gate)
                ? query_heads_
                : 2U * query_heads_) * head_dim_,
           hidden_size_});
      expect_quantized_shape(binding(operation, "key_projection"),
                             {kv_heads_ * head_dim_, hidden_size_});
      expect_quantized_shape(binding(operation, "value_projection"),
                             {kv_heads_ * head_dim_, hidden_size_});
      expect_quantized_shape(binding(operation, "output_projection"),
                             {hidden_size_, query_heads_ * head_dim_});
      if (qsa) {
        expect_shape(binding(operation, "query_norm"), {head_dim_}, "F32");
        expect_shape(binding(operation, "key_norm"), {head_dim_}, "F32");
        expect_quantized_shape(
            binding(operation, "index_projection"),
            {(qsa_index_heads_ + 1U) * qsa_index_head_dim_, hidden_size_});
        expect_shape(binding(operation, "index_query_norm"),
                     {qsa_index_head_dim_}, "F32");
        expect_shape(binding(operation, "index_key_norm"),
                     {qsa_index_head_dim_}, "F32");
        static_cast<void>(parameter_f32(
            operation.parameters, "norm_epsilon_f32_bits"));
        static_cast<void>(parameter_f32(
            operation.parameters, "rope_theta_f32_bits"));
        if (parameter_u32(operation.parameters, "rotary_dimension") !=
            rotary_dimension_)
          throw std::runtime_error("QSA rotary geometry is invalid");
      } else if (separated_gate) {
        expect_quantized_shape(binding(operation, "gate_projection"),
                               {query_heads_ * head_dim_, hidden_size_});
        expect_shape(binding(operation, "post_norm"), {hidden_size_}, "F32");
        if (!device_resident_fp16_kv() ||
            parameter_u32(operation.parameters, "input_norm_mode") > 1U ||
            parameter_u32(operation.parameters, "qk_norm_mode") != 1U ||
            parameter_u32(operation.parameters, "post_norm_mode") > 1U ||
            parameter_u32(operation.parameters, "rotary_enabled") > 1U ||
            parameter_u32(operation.parameters, "attention_window_tokens") !=
                operation.attention_window_tokens)
          throw std::runtime_error(
              "output-gated attention ABI 2 contract is invalid");
        static_cast<void>(parameter_f32(
            operation.parameters, "input_norm_epsilon_f32_bits"));
        static_cast<void>(parameter_f32(
            operation.parameters, "qk_norm_epsilon_f32_bits"));
        static_cast<void>(parameter_f32(
            operation.parameters, "query_scale_f32_bits"));
        static_cast<void>(parameter_f32(
            operation.parameters, "post_norm_epsilon_f32_bits"));
        if (parameter_u32(operation.parameters, "rotary_enabled") != 0U)
          static_cast<void>(parameter_f32(
              operation.parameters, "rope_theta_f32_bits"));
        if (operation.kv_layer_slot == kNoSlot)
          throw std::runtime_error("attention ABI 2 has no physical KV slot");
      } else if (operation.capability !=
                     "block.full-attention.standard-gqa.v1" &&
          operation.capability !=
              "block.full-attention.standard-gqa-no-position.v1") {
        expect_shape(binding(operation, "query_norm"), {head_dim_}, "F32");
        expect_shape(binding(operation, "key_norm"), {head_dim_}, "F32");
      } else if (!device_resident_fp16_kv()) {
        throw std::runtime_error(
            "standard GQA requires device-resident exact FP16 KV");
      }
      break;
    }
    case Kernel::recurrent_attention: {
      if (operation.recurrent_slot == kNoSlot)
        throw std::runtime_error("recurrent attention has no artifact slot");
      if (operation.capability == "block.mamba2.ssm.v1") {
        const auto intermediate = mamba_heads_ * mamba_head_dim_;
        const auto projection = intermediate + mamba_conv_size_ + mamba_heads_;
        expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
        expect_quantized_shape(binding(operation, "input_projection"),
                               {projection, hidden_size_});
        auto& convolution =
            const_cast<DeviceTensor&>(binding(operation, "convolution"));
        expect_shape(convolution,
                     {mamba_conv_size_, 1U, mamba_conv_kernel_}, "FP4_E2M1");
        if (!convolution.dequantized)
          throw std::runtime_error("Mamba2 convolution was not dequantized");
        expect_shape(binding(operation, "convolution_bias"),
                     {mamba_conv_size_}, "F32");
        for (const auto role : {"time_bias", "decay_log", "skip"})
          expect_shape(binding(operation, role), {mamba_heads_}, "F32");
        expect_shape(binding(operation, "output_norm"), {intermediate},
                     "F32");
        expect_quantized_shape(binding(operation, "output_projection"),
                               {hidden_size_, intermediate});
        static_cast<void>(parameter_f32(operation.parameters,
                                        "time_step_min_f32_bits"));
        break;
      }
      const auto key_dimension = key_heads_ * key_head_dim_;
      const auto value_dimension = value_heads_ * value_head_dim_;
      const auto conv_dimension = 2U * key_dimension + value_dimension;
      if (operation.capability !=
          "block.recurrent-linear-attention.split-gated-delta.no-residual.v1")
        expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
      expect_quantized_shape(binding(operation, "qkv_projection"),
                             {conv_dimension, hidden_size_});
      expect_quantized_shape(binding(operation, "z_projection"),
                             {value_dimension, hidden_size_});
      expect_quantized_shape(binding(operation, "b_projection"),
                             {value_heads_, hidden_size_});
      expect_quantized_shape(binding(operation, "a_projection"),
                             {value_heads_, hidden_size_});
      auto& convolution = const_cast<DeviceTensor&>(
          binding(operation, "convolution"));
      expect_shape(convolution, {conv_dimension, 1U, conv_kernel_},
                   "FP4_E2M1");
      if (!convolution.dequantized)
        throw std::runtime_error(
            "recurrent convolution was not dequantized");
      expect_shape(binding(operation, "time_bias"), {value_heads_}, "F32");
      expect_shape(binding(operation, "decay_log"), {value_heads_}, "F32");
      expect_shape(binding(operation, "output_norm"), {value_head_dim_},
                   "F32");
      expect_quantized_shape(binding(operation, "output_projection"),
                             {hidden_size_, value_dimension});
      if (operation.abi_version >= 2U) {
        const auto activation = parameter_u32(
            operation.parameters, "output_gate_activation");
        if (activation < 1U || activation > 2U)
          throw std::runtime_error(
              "recurrent output-gate activation is invalid");
      }
      break;
    }
    case Kernel::router:
      if (!routed_experts_ ||
          operation.logical_layer == er::kModelLevelOperationLayer ||
          operation.component_layer >=
              routed_experts_->component().layer_count)
        throw std::runtime_error("router has no routed component layer");
      if (operation.capability ==
          "router.softmax-topk.shared-swiglu.nvfp4-block16.v1") {
        expect_shape(binding(operation, "input_norm"), {hidden_size_},
                     "BF16");
        expect_shape(binding(operation, "router_weight"),
                     {expert_count_, hidden_size_}, "BF16");
        for (const auto role : {"shared_gate_projection",
                                "shared_up_projection",
                                "shared_down_projection"}) {
          const auto& tensor = binding(operation, role);
          if (tensor.quant_abi != er::kExpertRecordAbiNvfp4Block16W4A4)
            throw std::runtime_error(
                "native NVFP4 shared projection has the wrong ABI");
        }
        expect_quantized_shape(binding(operation, "shared_gate_projection"),
                               {shared_intermediate_size_, hidden_size_});
        expect_quantized_shape(binding(operation, "shared_up_projection"),
                               {shared_intermediate_size_, hidden_size_});
        expect_quantized_shape(binding(operation, "shared_down_projection"),
                               {hidden_size_, shared_intermediate_size_});
        static_cast<void>(parameter_f32(
            operation.parameters, "route_scale_f32_bits"));
        break;
      }
      if (operation.capability !=
          "router.linear-topk.shared-swiglu.no-residual.v1")
        expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
      expect_shape(binding(operation, "router_weight"),
                   {expert_count_, hidden_size_}, "F32");
      if (operation.capability ==
          "router.sigmoid-bias.topk.shared-relu2.v1") {
        expect_shape(binding(operation, "correction_bias"), {expert_count_},
                     "F32");
        expect_quantized_shape(binding(operation, "shared_up_projection"),
                               {shared_intermediate_size_, hidden_size_});
        expect_quantized_shape(binding(operation, "shared_down_projection"),
                               {hidden_size_, shared_intermediate_size_});
        static_cast<void>(parameter_f32(operation.parameters,
                                        "scale_f32_bits"));
      } else {
        expect_quantized_shape(binding(operation, "shared_gate_projection"),
                               {shared_intermediate_size_, hidden_size_});
        expect_quantized_shape(binding(operation, "shared_up_projection"),
                               {shared_intermediate_size_, hidden_size_});
        expect_quantized_shape(binding(operation, "shared_down_projection"),
                               {hidden_size_, shared_intermediate_size_});
        expect_shape(binding(operation, "shared_router"),
                     {1U, hidden_size_}, "F32");
      }
      break;
    case Kernel::routed_moe:
      if (!routed_experts_ || !operation.tensors.empty() ||
          operation.logical_layer == er::kModelLevelOperationLayer ||
          operation.component_layer >=
              routed_experts_->component().layer_count)
        throw std::runtime_error("routed MoE has no component layer");
      break;
    case Kernel::ffn: {
      expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
      if (operation.abi_version >= 2U) {
        expect_shape(binding(operation, "post_norm"), {hidden_size_}, "F32");
        if (parameter_u32(operation.parameters, "input_norm_mode") > 1U ||
            parameter_u32(operation.parameters, "post_norm_mode") > 1U)
          throw std::runtime_error("dense FFN ABI 2 norm mode is invalid");
        static_cast<void>(parameter_f32(
            operation.parameters, "input_norm_epsilon_f32_bits"));
        static_cast<void>(parameter_f32(
            operation.parameters, "post_norm_epsilon_f32_bits"));
      }
      const auto& gate = binding(operation, "gate_projection");
      const auto& up = binding(operation, "up_projection");
      if (!is_quantized_matrix(gate) || gate.shape[1] != hidden_size_ ||
          !is_quantized_matrix(up) || up.shape != gate.shape)
        throw std::runtime_error("dense FFN gate/up geometry is invalid");
      if (!intermediate_size_)
        intermediate_size_ = gate.shape[0];
      else if (intermediate_size_ != gate.shape[0])
        throw std::runtime_error("dense FFN width changes between layers");
      expect_quantized_shape(binding(operation, "down_projection"),
                             {hidden_size_, intermediate_size_});
      break;
    }
    case Kernel::head:
      if (operation.capability ==
          "head.rmsnorm.token-select.bfloat16.v1") {
        expect_shape(binding(operation, "norm"), {hidden_size_}, "BF16");
        expect_shape(binding(operation, "weight"),
                     {vocabulary_size_, hidden_size_}, "BF16");
        break;
      }
      if (operation.capability !=
              "head.token-select.fp4-block32.no-norm.v1" &&
          operation.capability != "head.token-select.no-norm.v1")
        expect_shape(binding(operation, "norm"), {hidden_size_}, "F32");
      expect_quantized_shape(binding(operation, "weight"),
                             {vocabulary_size_, hidden_size_});
      if (operation.capability ==
              "head.rmsnorm.token-select.mxfp6-e3m2-block32.v1" &&
          binding(operation, "weight").quant_abi !=
              er::kDenseRecordAbiMxfp6E3m2Block32)
        throw std::runtime_error("MXFP6 output head has the wrong ABI");
      if (operation.abi_version >= 2U) {
        static_cast<void>(parameter_f32(
            operation.parameters, "logit_multiplier_f32_bits"));
        static_cast<void>(parameter_f32(
            operation.parameters, "logit_softcap_f32_bits"));
      }
      break;
    case Kernel::exact_decode:
      throw std::runtime_error("exact decode reached scalar validation");
  }
}

void DenseFp4Provider::validate_exact(PreparedOperation& operation) {
  if (mtp_layers_ != 1U)
    throw std::runtime_error("this exact provider implements one MTP layer");
  expect_quantized_shape(binding(operation, "token_embedding"),
                         {vocabulary_size_, hidden_size_});
  expect_quantized_shape(binding(operation, "output_head"),
                         {vocabulary_size_, hidden_size_});
  const auto mixed_mxfp6 = operation.capability ==
      "decode.mtp.dense-full-attention.fp4-mxfp6-io.exact.v3";
  const auto expected_io_abi = mixed_mxfp6
      ? er::kDenseRecordAbiMxfp6E3m2Block32
      : er::kExpertQuantAbiFp4Block32;
  if (binding(operation, "token_embedding").quant_abi != expected_io_abi ||
      binding(operation, "output_head").quant_abi != expected_io_abi)
    throw std::runtime_error("exact decode I/O quantization ABI mismatch");
  expect_shape(binding(operation, "fusion_projection"),
               {hidden_size_, 2U * hidden_size_}, "FP4_E2M1");
  for (const auto role : {"embedding_norm", "hidden_norm", "draft_norm"})
    expect_shape(binding(operation, role), {hidden_size_}, "F32");
  const auto role = [](std::string_view suffix) {
    return std::string("layer.0.") + std::string(suffix);
  };
  expect_shape(binding(operation, role("input_norm")), {hidden_size_}, "F32");
  expect_shape(binding(operation, role("query_projection")),
               {2U * query_heads_ * head_dim_, hidden_size_}, "FP4_E2M1");
  expect_shape(binding(operation, role("key_projection")),
               {kv_heads_ * head_dim_, hidden_size_}, "FP4_E2M1");
  expect_shape(binding(operation, role("value_projection")),
               {kv_heads_ * head_dim_, hidden_size_}, "FP4_E2M1");
  expect_shape(binding(operation, role("output_projection")),
               {hidden_size_, query_heads_ * head_dim_}, "FP4_E2M1");
  expect_shape(binding(operation, role("query_norm")), {head_dim_}, "F32");
  expect_shape(binding(operation, role("key_norm")), {head_dim_}, "F32");
  expect_shape(binding(operation, role("post_attention_norm")),
               {hidden_size_}, "F32");
  expect_shape(binding(operation, role("gate_projection")),
               {intermediate_size_, hidden_size_}, "FP4_E2M1");
  expect_shape(binding(operation, role("up_projection")),
               {intermediate_size_, hidden_size_}, "FP4_E2M1");
  expect_shape(binding(operation, role("down_projection")),
               {hidden_size_, intermediate_size_}, "FP4_E2M1");
  mtp_attention_ = std::make_shared<PreparedOperation>();
  mtp_attention_->kernel = Kernel::full_attention;
  mtp_attention_->full_attention_slot =
      host_authoritative_fp16_kv() ? 0U : target_full_layers_;
  for (const auto suffix : {"input_norm", "query_projection",
                            "key_projection", "value_projection",
                            "output_projection", "query_norm", "key_norm"})
    mtp_attention_->tensors.emplace(
        suffix, operation.tensors.at(role(suffix)));
  mtp_ffn_ = std::make_shared<PreparedOperation>();
  mtp_ffn_->kernel = Kernel::ffn;
  mtp_ffn_->tensors.emplace(
      "input_norm", operation.tensors.at(role("post_attention_norm")));
  for (const auto suffix : {"gate_projection", "up_projection",
                            "down_projection"})
    mtp_ffn_->tensors.emplace(suffix, operation.tensors.at(role(suffix)));
}

void DenseFp4Provider::initialize_execution() {
  if (initialized_) return;
  if (std::any_of(prepared_target_.begin(), prepared_target_.end(),
                  [](const auto& item) { return !item; }) ||
      (descriptor_.exact_decode_program.has_value() && !exact_) ||
      (!intermediate_size_ && !routed_experts_))
    throw std::runtime_error("FP4 operation program is not fully prepared");
  for (const auto& operation : prepared_target_) {
    if (operation->kernel == Kernel::embedding ||
        operation->kernel == Kernel::vision ||
        operation->kernel == Kernel::head)
      continue;
    std::set<const DeviceTensor*> unique;
    std::size_t fp4_operation_bytes{};
    std::size_t int8_matrix_bytes{};
    for (const auto& [role, tensor] : operation->tensors) {
      static_cast<void>(role);
      if (!tensor || tensor->shape.size() != 2U ||
          !unique.insert(tensor).second)
        continue;
      if (tensor->encoding == "FP4_E2M1" &&
          tensor->quant_abi == er::kExpertQuantAbiFp4Block32) {
        const auto matrix = tensor->matrix();
        const auto values =
            static_cast<std::size_t>(matrix.rows) * matrix.padded_columns;
        if (values > std::numeric_limits<std::size_t>::max() /
                         sizeof(std::uint16_t) ||
            fp4_operation_bytes >
                std::numeric_limits<std::size_t>::max() -
                    values * sizeof(std::uint16_t))
          throw std::runtime_error(
              "staged FP4 weight workspace overflows");
        fp4_operation_bytes += values * sizeof(std::uint16_t);
      } else if (tensor->encoding == "I8") {
        const auto matrix = tensor->int8_matrix();
        const auto values = static_cast<std::uint64_t>(matrix.rows) *
                            matrix.columns;
        if (values > std::numeric_limits<std::size_t>::max() /
                         sizeof(float))
          throw std::runtime_error(
              "staged INT8 weight workspace overflows");
        int8_matrix_bytes = std::max(
            int8_matrix_bytes,
            static_cast<std::size_t>(values) * sizeof(float));
      }
    }
    staged_dense_weight_capacity_bytes_ =
        std::max(staged_dense_weight_capacity_bytes_, fp4_operation_bytes);
    staged_int8_weight_capacity_values_ = std::max(
        staged_int8_weight_capacity_values_, int8_matrix_bytes / sizeof(float));
  }
  if (large_exact_prefill_workspace_) {
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
               "inspect large exact-prefill workspace headroom");
    workspace_preflight_free_bytes_ = free_bytes;
    const auto reserve = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(total_bytes) / 8U, 1ULL << 30U);
    if (static_cast<std::uint64_t>(free_bytes) <= reserve)
      throw std::runtime_error(
          "large exact-prefill workspace cannot preserve the device reserve");
  }
  const auto maximum_columns = align32(std::max(
      {2U * hidden_size_, hidden_size_, intermediate_size_,
       shared_intermediate_size_, expert_width_,
       query_heads_ * head_dim_, value_heads_ * value_head_dim_,
       hyper_width_, hyper_lowrank_, ple_embedding_width_,
       (qsa_index_heads_ + 1U) * qsa_index_head_dim_,
       mamba_heads_ * mamba_head_dim_ + mamba_conv_size_ + mamba_heads_,
       vision_patch_dimension_, vision_hidden_size_,
       vision_intermediate_size_, vision_merged_width_}));
  staged_dense_input_capacity_bytes_ =
      static_cast<std::size_t>(workspace_rows_) * maximum_columns *
      sizeof(std::uint16_t);
  allocate_workspace();
  allocate_state();
  if (large_exact_prefill_workspace_) {
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
               "verify large exact-prefill workspace reserve");
    workspace_postallocation_free_bytes_ = free_bytes;
    const auto reserve = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(total_bytes) / 8U, 1ULL << 30U);
    if (static_cast<std::uint64_t>(free_bytes) < reserve)
      throw std::runtime_error(
          "large exact-prefill workspace consumed the device reserve");
  }
  if (fit_routed_vram_ && routed_experts_) {
    if (!all_kv_pages_.empty() || !all_target_mirror_pages_.empty())
      throw std::runtime_error(
          "routed VRAM fitting must precede dynamic KV allocation");
    const auto future_kv_bytes = checked_multiply(
        kv_page_capacity_, kv_page_bytes_, "future device KV reservation");
    const auto future_mirror_bytes = checked_multiply(
        target_mirror_page_capacity_, host_fp16_target_page_bytes_,
        "future target mirror reservation");
    const auto maximum_logits_bytes = checked_multiply(
        checked_multiply(workspace_rows_, vocabulary_size_,
                         "maximum logits workspace values"),
        sizeof(float), "maximum logits workspace bytes");
    const auto current_logits_bytes = checked_multiply(
        checked_multiply(logits_capacity_rows_, vocabulary_size_,
                         "current logits workspace values"),
        sizeof(float), "current logits workspace bytes");
    const auto future_logits_bytes =
        maximum_logits_bytes - current_logits_bytes;
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
               "inspect routed VRAM fit headroom");
    const auto emergency_reserve = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(total_bytes) / 8U, 1ULL << 30U);
    auto withheld = future_kv_bytes;
    if (future_mirror_bytes >
        std::numeric_limits<std::uint64_t>::max() - withheld)
      throw std::runtime_error("routed VRAM fit reservation overflows");
    withheld += future_mirror_bytes;
    if (future_logits_bytes >
        std::numeric_limits<std::uint64_t>::max() - withheld)
      throw std::runtime_error("routed VRAM fit reservation overflows");
    withheld += future_logits_bytes;
    if (emergency_reserve >
        std::numeric_limits<std::uint64_t>::max() - withheld)
      throw std::runtime_error("routed VRAM fit reservation overflows");
    withheld += emergency_reserve;
    if (static_cast<std::uint64_t>(free_bytes) <= withheld)
      throw std::runtime_error(
          "fixed CUDA organs and dynamic state leave no routed VRAM cache");
    const auto fitted = std::min<std::uint64_t>(
        routed_experts_->device_pool_bytes(),
        static_cast<std::uint64_t>(free_bytes) - withheld);
    routed_experts_->configure_vram_cache_capacity(fitted);
  }
  initialized_ = true;
}

void DenseFp4Provider::allocate_workspace() {
  const auto query_width = 2U * query_heads_ * head_dim_;
  const auto key_value_width = kv_heads_ * head_dim_;
  const auto attention_width = query_heads_ * head_dim_;
  const auto key_dimension = key_heads_ * key_head_dim_;
  const auto value_dimension = value_heads_ * value_head_dim_;
  const auto conv_dimension = 2U * key_dimension + value_dimension;
  const auto mamba_intermediate = mamba_heads_ * mamba_head_dim_;
  const auto mamba_projection =
      mamba_intermediate + mamba_conv_size_ + mamba_heads_;
  const auto recurrent_projection =
      std::max<std::uint32_t>({1U, conv_dimension, mamba_projection});
  const auto recurrent_conv_output =
      std::max<std::uint32_t>({1U, conv_dimension, mamba_conv_size_});
  const auto recurrent_output =
      std::max<std::uint32_t>({1U, value_dimension, mamba_intermediate});
  const auto workspace_intermediate = std::max(
      {intermediate_size_, shared_intermediate_size_, expert_width_,
       mamba_intermediate, hyper_lowrank_, ple_embedding_width_});
  const auto maximum_columns = std::max(
      {2U * hidden_size_, hidden_size_, workspace_intermediate,
       attention_width,
       hyper_width_, (qsa_index_heads_ + 1U) * qsa_index_head_dim_,
       value_dimension, mamba_projection, vision_patch_dimension_,
       vision_hidden_size_,
       vision_intermediate_size_, vision_merged_width_});
  const auto padded_columns = align32(maximum_columns);
  hidden_ = device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
  const auto hidden_row_bytes =
      static_cast<std::uint64_t>(hidden_size_) * sizeof(float);
  const auto minimum_sequence_bytes =
      static_cast<std::uint64_t>(capacity_) * workspace_rows_ *
      hidden_row_bytes;
  const auto desired_sequence_bytes = std::max<std::uint64_t>(
      minimum_sequence_bytes,
      std::min<std::uint64_t>(128ULL << 20U, vram_cache_bytes_ / 64U));
  auto rows_per_slot = desired_sequence_bytes /
      (static_cast<std::uint64_t>(capacity_) * hidden_row_bytes);
  rows_per_slot = std::max<std::uint64_t>(workspace_rows_, rows_per_slot);
  rows_per_slot -= rows_per_slot % workspace_rows_;
  sequence_tile_rows_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(
      rows_per_slot, std::numeric_limits<std::uint32_t>::max()));
  sequence_hidden_values_ = static_cast<std::size_t>(capacity_) *
                            sequence_tile_rows_ * hidden_size_;
  sequence_hidden_ = device_allocate<float>(
      allocations_, sequence_hidden_values_);
  if (hyper_enabled_) {
    sequence_hyper_values_ = static_cast<std::size_t>(capacity_) *
                             sequence_tile_rows_ * hyper_width_;
    sequence_injection_values_ = static_cast<std::size_t>(capacity_) *
                                 sequence_tile_rows_ * hyper_count_;
    sequence_hyper_ = device_allocate<float>(
        allocations_, sequence_hyper_values_);
    sequence_injection_ = device_allocate<float>(
        allocations_, sequence_injection_values_);
  }
  normalized_ =
      device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
  residual_ = device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
  if (hyper_enabled_) {
    hyper_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    hyper_normalized_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    hyper_lowrank_values_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) *
                          hyper_lowrank_);
    hyper_mix_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    hyper_injection_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_count_);
  }
  if (ple_enabled_) {
    ple_embeddings_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) *
                          ple_embedding_width_);
    ple_key_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    ple_key_norm_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    ple_query_norm_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    ple_value_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hidden_size_);
    ple_gated_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    ple_conv_norm_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    ple_conv_output_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * hyper_width_);
    ple_host_embeddings_.resize(
        static_cast<std::size_t>(workspace_rows_) * ple_embedding_width_);
  }
  if (qsa_enabled_) {
    qsa_projected_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) *
                          (qsa_index_heads_ + 1U) * qsa_index_head_dim_);
    const auto maximum_blocks =
        (max_context_ + qsa_compress_ratio_ - 1U) / qsa_compress_ratio_;
    qsa_scores_ = device_allocate<float>(allocations_, maximum_blocks);
    qsa_selected_ = device_allocate<std::uint32_t>(
        allocations_, qsa_token_budget_ + qsa_compress_ratio_ - 1U);
    qsa_host_scores_.resize(maximum_blocks);
    qsa_host_selected_.reserve(
        qsa_token_budget_ + qsa_compress_ratio_ - 1U);
    if (qsa_tiered_) {
      const auto selected_values = checked_multiply(
          qsa_token_budget_ + qsa_compress_ratio_ - 1U,
          static_cast<std::uint64_t>(kv_heads_) * head_dim_,
          "tiered QSA staging values");
      const auto selected_bytes = checked_multiply(
          selected_values, sizeof(std::uint16_t),
          "tiered QSA staging bytes");
      void* keys{};
      void* values{};
      cuda_check(cudaHostAlloc(&keys, static_cast<std::size_t>(selected_bytes),
                               cudaHostAllocPortable),
                 "allocate tiered QSA key staging");
      try {
        cuda_check(cudaHostAlloc(&values,
                                 static_cast<std::size_t>(selected_bytes),
                                 cudaHostAllocPortable),
                   "allocate tiered QSA value staging");
      } catch (...) {
        static_cast<void>(cudaFreeHost(keys));
        throw;
      }
      qsa_host_selected_keys_ = static_cast<std::uint16_t*>(keys);
      qsa_host_selected_values_ = static_cast<std::uint16_t*>(values);
    }
  }
  query_gate_ =
      device_allocate<float>(allocations_, workspace_rows_ * query_width);
  if (target_kv_encoding_ == TargetKvEncoding::q4_bfp_key_outlier1 ||
      target_kv_encoding_ == TargetKvEncoding::q4_bfp ||
      target_kv_encoding_ == TargetKvEncoding::q4_per_head ||
      target_kv_encoding_ == TargetKvEncoding::q5_q4_bfp) {
    attention_q8_queries_ = device_allocate<std::int8_t>(
        allocations_, static_cast<std::size_t>(kMaximumExactDecodeRows) *
                          query_heads_ * head_dim_);
    attention_query_scales_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kMaximumExactDecodeRows) *
                          query_heads_);
  }
  if (mla_enabled_) {
    mla_query_rank_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) *
                          q_lora_rank_);
    mla_latent_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) *
                          (kv_lora_rank_ + qk_rope_head_dim_));
    mla_latent_query_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) *
                          query_heads_ * kv_lora_rank_);
  }
  key_ = device_allocate<float>(allocations_, workspace_rows_ * key_value_width);
  value_ = device_allocate<float>(allocations_, workspace_rows_ * key_value_width);
  attention_ =
      device_allocate<float>(allocations_, workspace_rows_ * attention_width);
  projected_qkv_ =
      device_allocate<float>(allocations_, workspace_rows_ * recurrent_projection);
  projected_z_ =
      device_allocate<float>(allocations_,
                             workspace_rows_ * std::max(1U, value_dimension));
  projected_b_ =
      device_allocate<float>(allocations_,
                             workspace_rows_ * std::max(1U, value_heads_));
  projected_a_ =
      device_allocate<float>(allocations_,
                             workspace_rows_ * std::max(1U, value_heads_));
  conv_output_ =
      device_allocate<float>(allocations_,
                             workspace_rows_ * recurrent_conv_output);
  delta_output_ =
      device_allocate<float>(allocations_, workspace_rows_ * recurrent_output);
  gate_ =
      device_allocate<float>(allocations_, workspace_rows_ * workspace_intermediate);
  up_ = device_allocate<float>(allocations_,
                               workspace_rows_ * workspace_intermediate);
  intermediate_ =
      device_allocate<float>(allocations_,
                             workspace_rows_ * workspace_intermediate);
  if (routed_experts_) {
    shared_output_ =
        device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
    shared_scalar_ = device_allocate<float>(allocations_, workspace_rows_);
    router_logits_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * expert_count_);
    routing_scores_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * route_width_);
    routing_indices_ = device_allocate<std::uint32_t>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * route_width_);
    moe_intermediate_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * route_width_ *
                          expert_width_);
    moe_selection_output_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * route_width_ *
                          hidden_size_);
    moe_output_ =
        device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
    moe_q8_intermediate_ = device_allocate<std::int8_t>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * route_width_ *
                          expert_width_);
    moe_q8_intermediate_scales_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(workspace_rows_) * route_width_);
    if (routed_experts_->component().encoding_abi ==
        er::kExpertEncodingAbiNvfp4Block16W4A4) {
      moe_nvfp4_gate_input_ = device_allocate<float>(
          allocations_, static_cast<std::size_t>(workspace_rows_) *
                            route_width_ * hidden_size_);
      moe_nvfp4_up_input_ = device_allocate<float>(
          allocations_, static_cast<std::size_t>(workspace_rows_) *
                            route_width_ * hidden_size_);
      moe_nvfp4_down_input_ = device_allocate<float>(
          allocations_, static_cast<std::size_t>(workspace_rows_) *
                            route_width_ * expert_width_);
    }
  }
  ensure_logits_capacity(1U);
  output_tokens_ =
      device_allocate<std::uint32_t>(allocations_, workspace_rows_);
  sampling_top_logits_ = device_allocate<float>(
      allocations_, kMaximumGpuSamplingTopK);
  sampling_top_tokens_ = device_allocate<std::uint32_t>(
      allocations_, kMaximumGpuSamplingTopK);
  sampling_top_partial_items_ =
      ec::topk_logits_workspace_items(vocabulary_size_);
  sampling_top_partial_logits_ = device_allocate<float>(
      allocations_, sampling_top_partial_items_);
  sampling_top_partial_tokens_ = device_allocate<std::uint32_t>(
      allocations_, sampling_top_partial_items_);
  sampling_presence_ = device_allocate<std::uint8_t>(
      allocations_, static_cast<std::size_t>(capacity_) * vocabulary_size_);
  sampling_proposal_presence_ =
      device_allocate<std::uint8_t>(allocations_, vocabulary_size_);
  q8_ = device_allocate<std::int8_t>(allocations_,
                                     workspace_rows_ * padded_columns);
  q8_scales_ = device_allocate<float>(allocations_, workspace_rows_);
  nvfp4_input_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(workspace_rows_) *
                        maximum_columns);
  if (vision_enabled_) {
    const auto maximum_merged_rows = kMaximumVisionPatches /
        (vision_spatial_merge_size_ * vision_spatial_merge_size_);
    vision_pixels_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kMaximumVisionPatches) *
                          vision_patch_dimension_);
    vision_hidden_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kMaximumVisionPatches) *
                          vision_hidden_size_);
    vision_normalized_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kMaximumVisionPatches) *
                          vision_hidden_size_);
    vision_qkv_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kMaximumVisionPatches) * 3U *
                          vision_hidden_size_);
    vision_attention_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kMaximumVisionPatches) *
                          vision_hidden_size_);
    vision_intermediate_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kMaximumVisionPatches) *
                          vision_intermediate_size_);
    vision_merger_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(maximum_merged_rows) *
                          vision_merged_width_);
    vision_output_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(maximum_merged_rows) *
                          vision_output_size_);
    vision_positions_ = device_allocate<std::uint32_t>(
        allocations_, 2U * kMaximumVisionPatches);
    vision_segment_first_ = device_allocate<std::uint32_t>(
        allocations_, kMaximumVisionPatches);
    vision_segment_last_ = device_allocate<std::uint32_t>(
        allocations_, kMaximumVisionPatches);
    vision_interpolation_indices_ = device_allocate<std::uint32_t>(
        allocations_, 4U * kMaximumVisionPatches);
    vision_interpolation_weights_ = device_allocate<float>(
        allocations_, 4U * kMaximumVisionPatches);
  }
  if (staged_int8_weight_capacity_values_ != 0U)
    staged_int8_weights_ = device_allocate<float>(
        allocations_, staged_int8_weight_capacity_values_);
  staged_dense_input_ = device_allocate<std::uint16_t>(
      allocations_, staged_dense_input_capacity_bytes_ /
                        sizeof(std::uint16_t));
  attention_maximum_splits_ =
      (max_context_ + kAttentionSplitTokens - 1U) / kAttentionSplitTokens;
  const auto attention_state_rows =
      std::max<std::uint64_t>(
          workspace_rows_,
          static_cast<std::uint64_t>(kMaximumExactDecodeRows) *
              attention_maximum_splits_);
  partial_maxima_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(attention_state_rows) *
                        query_heads_);
  partial_sums_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(attention_state_rows) *
                        query_heads_);
  partial_outputs_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(kMaximumExactDecodeRows) *
                        attention_maximum_splits_ * query_heads_ * head_dim_);
  staged_split_tokens_ = std::min(kStagedPrefillSplitTokens, max_context_);
  const auto staged_query_values =
      static_cast<std::size_t>(workspace_rows_) * query_heads_ * head_dim_;
  const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                staged_split_tokens_ * head_dim_;
  const auto full_score_values =
      static_cast<std::size_t>(workspace_rows_) * query_heads_ *
      staged_split_tokens_;
  staged_score_capacity_values_ = compact_flash_prefill_
      ? staged_query_values
      : full_score_values;
  // The paged FlashAttention path does not materialize probabilities, but the
  // common workspace contract requires a non-null pointer for fail-closed
  // validation. Exact-F16 and unsupported geometries retain the full arena.
  staged_probability_capacity_values_ = compact_flash_prefill_
      ? 1U
      : full_score_values;
  staged_queries_ =
      device_allocate<std::uint16_t>(allocations_, staged_query_values);
  if (exact_fp16_kv()) {
    staged_raw_keys_ =
        device_allocate<std::uint16_t>(allocations_, staged_kv_values);
    staged_raw_values_ =
        device_allocate<std::uint16_t>(allocations_, staged_kv_values);
    if (host_authoritative_fp16_kv()) {
      const auto staged_layer_values = static_cast<std::size_t>(max_context_) *
                                       kv_heads_ * head_dim_;
      staged_device_keys_ =
          device_allocate<std::uint16_t>(allocations_, staged_layer_values);
      staged_device_values_ =
          device_allocate<std::uint16_t>(allocations_, staged_layer_values);
    }
  }
  staged_keys_ =
      device_allocate<std::uint16_t>(allocations_, staged_kv_values);
  staged_values_ =
      device_allocate<std::uint16_t>(allocations_, staged_kv_values);
  staged_scores_ =
      device_allocate<float>(allocations_, staged_score_capacity_values_);
  staged_probabilities_ =
      device_allocate<std::uint16_t>(
          allocations_, staged_probability_capacity_values_);
  staged_accumulator_ =
      device_allocate<float>(allocations_, staged_query_values);
  mtp_embedding_ =
      device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
  mtp_embedding_norm_ =
      device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
  mtp_hidden_norm_ =
      device_allocate<float>(allocations_, workspace_rows_ * hidden_size_);
  mtp_fusion_input_ =
      device_allocate<float>(allocations_, workspace_rows_ * 2U * hidden_size_);
  slot_target_hidden_batch_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(capacity_) * workspace_rows_ *
                        hidden_size_);
}

void DenseFp4Provider::allocate_state() {
  cuda_check(cudaStreamCreateWithFlags(&parking_stream_, cudaStreamNonBlocking),
             "create request parking stream");
  slot_last_hidden_ =
      device_allocate<float>(allocations_, capacity_ * hidden_size_);
  slot_mtp_last_hidden_ =
      device_allocate<float>(allocations_, capacity_ * hidden_size_);
  mtp_rollout_previous_hidden_ =
      device_allocate<float>(allocations_, hidden_size_);
  slot_retention_last_hidden_ =
      device_allocate<float>(allocations_, capacity_ * hidden_size_);
  const auto conv_values = recurrent_conv_values_;
  const auto matrix_values = recurrent_matrix_values_;
  recurrent_conv_state_.resize(recurrent_layers_);
  recurrent_matrix_state_.resize(recurrent_layers_);
  recurrent_conv_checkpoint_.resize(recurrent_layers_);
  recurrent_matrix_checkpoint_.resize(recurrent_layers_);
  recurrent_conv_speculative_checkpoint_.resize(recurrent_layers_);
  recurrent_matrix_speculative_checkpoint_.resize(recurrent_layers_);
  recurrent_conv_retention_checkpoint_.resize(recurrent_layers_);
  recurrent_matrix_retention_checkpoint_.resize(recurrent_layers_);
  for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
    recurrent_conv_state_[layer] =
        device_allocate<float>(allocations_, capacity_ * conv_values);
    recurrent_matrix_state_[layer] =
        device_allocate<float>(allocations_, capacity_ * matrix_values);
    recurrent_conv_checkpoint_[layer] =
        device_allocate<float>(allocations_, capacity_ * conv_values);
    recurrent_matrix_checkpoint_[layer] =
        device_allocate<float>(allocations_, capacity_ * matrix_values);
    recurrent_conv_speculative_checkpoint_[layer] = device_allocate<float>(
        allocations_, static_cast<std::size_t>(capacity_) *
                          kMaximumExactDecodeRows * conv_values);
    recurrent_matrix_speculative_checkpoint_[layer] = device_allocate<float>(
        allocations_, static_cast<std::size_t>(capacity_) *
                          kMaximumExactDecodeRows * matrix_values);
    recurrent_conv_retention_checkpoint_[layer] =
        device_allocate<float>(allocations_, capacity_ * conv_values);
    recurrent_matrix_retention_checkpoint_[layer] =
        device_allocate<float>(allocations_, capacity_ * matrix_values);
  }
  ple_conv_state_.resize(ple_layers_);
  ple_conv_retention_checkpoint_.resize(ple_layers_);
  for (std::uint32_t layer = 0U; layer < ple_layers_; ++layer) {
    ple_conv_state_[layer] = device_allocate<float>(
        allocations_, capacity_ * ple_conv_state_values_);
    ple_conv_retention_checkpoint_[layer] = device_allocate<float>(
        allocations_, capacity_ * ple_conv_state_values_);
  }
  device_page_table_ = device_allocate<void*>(
      allocations_, static_cast<std::size_t>(capacity_) *
                        maximum_pages_per_slot_);
  cuda_check(cudaMemset(device_page_table_, 0,
                        static_cast<std::size_t>(capacity_) *
                            maximum_pages_per_slot_ * sizeof(void*)),
             "zero FP4 KV page table");
  if ((target_kv_encoding_ == TargetKvEncoding::fp8_e4m3_per_head ||
       target_kv_encoding_ == TargetKvEncoding::fp4_key_outlier1 ||
       target_kv_encoding_ == TargetKvEncoding::q4_bfp_key_outlier1 ||
       target_kv_encoding_ == TargetKvEncoding::q4_bfp ||
       target_kv_encoding_ == TargetKvEncoding::q4_per_head ||
       target_kv_encoding_ == TargetKvEncoding::q5_q4_bfp) &&
      mtp_layers_ != 0U) {
    device_mtp_page_table_ = device_allocate<void*>(
        allocations_, static_cast<std::size_t>(capacity_) *
                          maximum_pages_per_slot_);
    cuda_check(cudaMemset(device_mtp_page_table_, 0,
                          static_cast<std::size_t>(capacity_) *
                              maximum_pages_per_slot_ * sizeof(void*)),
               "zero mixed MTP KV page table");
  } else {
    device_mtp_page_table_ = device_page_table_;
  }
  if (target_mirror_page_capacity_ != 0U) {
    device_target_mirror_page_table_ = device_allocate<void*>(
        allocations_, static_cast<std::size_t>(capacity_) *
                          maximum_pages_per_slot_);
    cuda_check(cudaMemset(device_target_mirror_page_table_, 0,
                          static_cast<std::size_t>(capacity_) *
                              maximum_pages_per_slot_ * sizeof(void*)),
               "zero exact FP16 target mirror page table");
  }
  if (window_attention_layers_ != 0U) {
    if (!window_kv_bytes_per_slot_ || !window_pages_per_slot_)
      throw std::runtime_error("window KV geometry is empty");
    const auto window_bytes = static_cast<std::size_t>(capacity_) *
                              window_kv_bytes_per_slot_;
    window_kv_ = device_allocate<std::byte>(allocations_, window_bytes);
    window_retention_kv_ =
        device_allocate<std::byte>(allocations_, window_bytes);
    device_window_page_table_ = device_allocate<void*>(
        allocations_, static_cast<std::size_t>(capacity_) *
                          maximum_pages_per_slot_);
    std::vector<void*> table(static_cast<std::size_t>(capacity_) *
                             maximum_pages_per_slot_);
    for (std::uint32_t slot = 0U; slot < capacity_; ++slot) {
      auto* slot_base = window_kv_ +
          static_cast<std::size_t>(slot) * window_kv_bytes_per_slot_;
      for (std::uint32_t page = 0U; page < maximum_pages_per_slot_; ++page)
        table[static_cast<std::size_t>(slot) * maximum_pages_per_slot_ + page] =
            slot_base + static_cast<std::size_t>(page % window_pages_per_slot_) *
                            window_kv_page_bytes_;
    }
    cuda_check(cudaMemcpy(device_window_page_table_, table.data(),
                          table.size() * sizeof(table[0]),
                          cudaMemcpyHostToDevice),
               "publish cyclic window KV page table");
  } else {
    device_window_page_table_ = device_page_table_;
  }
  slot_pages_.assign(capacity_,
                     std::vector<void*>(maximum_pages_per_slot_, nullptr));
  slot_target_mirror_pages_.assign(
      capacity_, std::vector<void*>(maximum_pages_per_slot_, nullptr));
  free_target_mirror_pages_.reserve(
      static_cast<std::size_t>(target_mirror_page_capacity_));
  all_target_mirror_pages_.reserve(
      static_cast<std::size_t>(target_mirror_page_capacity_));
}

float* DenseFp4Provider::slot_last_hidden(std::uint32_t slot) const {
  return slot_last_hidden_ + static_cast<std::size_t>(slot) * hidden_size_;
}

float* DenseFp4Provider::recurrent_conv(std::uint32_t recurrent_slot,
                                        std::uint32_t request_slot) const {
  return recurrent_conv_state_.at(recurrent_slot) +
         request_slot * recurrent_conv_values_;
}

float* DenseFp4Provider::recurrent_matrix(std::uint32_t recurrent_slot,
                                          std::uint32_t request_slot) const {
  return recurrent_matrix_state_.at(recurrent_slot) +
         request_slot * recurrent_matrix_values_;
}

void DenseFp4Provider::release_request_state(RequestState& state) noexcept {
  try {
    if (!state.parked())
      release_slot(state.slot());
    std::lock_guard lock(mutex_);
    if (state.parked()) {
      const auto bytes = state.parked_state->bytes;
      const auto reported = state.parked_state->reported_bytes;
      if (bytes <= parked_request_bytes_) parked_request_bytes_ -= bytes;
      if (reported <= parked_session_bytes_)
        parked_session_bytes_ -= reported;
    }
    state.parked_state.reset();
    release_host_kv(state);
    state.slot_ = kNoSlot;
  } catch (...) {
  }
}

void DenseFp4Provider::release_slot(std::uint32_t slot) noexcept {
  try {
    std::lock_guard lock(mutex_);
    if (slot >= slot_in_use_.size() || !slot_in_use_[slot]) return;
    if (slot < slot_pages_.size()) {
      for (std::uint32_t page_index = 0U;
           page_index < slot_pages_[slot].size(); ++page_index) {
        auto*& page = slot_pages_[slot][page_index];
        if (!page) continue;
        free_kv_pages_.push_back(page);
        page = nullptr;
        void* empty{};
        static_cast<void>(cudaMemcpy(
            device_page_table_ + static_cast<std::size_t>(slot) *
                                     maximum_pages_per_slot_ +
                page_index,
            &empty, sizeof(empty), cudaMemcpyHostToDevice));
        if (device_mtp_page_table_ != device_page_table_)
          static_cast<void>(cudaMemcpy(
              device_mtp_page_table_ + static_cast<std::size_t>(slot) *
                                           maximum_pages_per_slot_ +
                  page_index,
              &empty, sizeof(empty), cudaMemcpyHostToDevice));
      }
    }
    release_target_mirror_pages(slot);
    if (slot < slot_rope_deltas_.size()) slot_rope_deltas_[slot] = 0;
    slot_in_use_[slot] = false;
  } catch (...) {
  }
}

er::CreateOperationRequestStateResult DenseFp4Provider::create_request_state(
    const er::ProgramRequestContext& request) {
  try {
    std::lock_guard lock(mutex_);
    initialize_execution();
    const auto context = request.parameters.find("reserved_context_tokens");
    if (context == request.parameters.end() || context->second == 0U ||
        context->second > max_context_)
      throw std::runtime_error("request has an invalid context reservation");
    const auto sampling_temperature =
        request.parameters.find("sampling_temperature_ppm");
    if (sampling_temperature != request.parameters.end() &&
        sampling_temperature->second != 0U) {
      const auto first = request.parameters.find(
          "sampling_first_output_position");
      const auto presence = request.parameters.find(
          "sampling_presence_penalty_biased_ppm");
      if (first == request.parameters.end() || first->second >= max_context_ ||
          presence == request.parameters.end() || presence->second > 4'000'000U)
        throw std::runtime_error("request has an invalid sampling profile");
    }
    const auto free = std::find(slot_in_use_.begin(), slot_in_use_.end(), false);
    if (free == slot_in_use_.end())
      return {{er::ErrorCode::backpressure,
               "dense FP4 provider has no free request slot"},
              {}};
    const auto slot = static_cast<std::uint32_t>(free - slot_in_use_.begin());
    *free = true;
    cuda_check(cudaMemset(
                   sampling_presence_ +
                       static_cast<std::size_t>(slot) * vocabulary_size_,
                   0, vocabulary_size_),
               "zero request sampling presence");
    const auto conv_bytes = recurrent_conv_values_ * sizeof(float);
    const auto matrix_bytes = recurrent_matrix_values_ * sizeof(float);
    for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
      cuda_check(cudaMemset(recurrent_conv(layer, slot), 0, conv_bytes),
                 "zero recurrent convolution state");
      cuda_check(cudaMemset(recurrent_matrix(layer, slot), 0, matrix_bytes),
                 "zero recurrent matrix state");
    }
    for (std::uint32_t layer = 0U; layer < ple_layers_; ++layer)
      cuda_check(cudaMemset(
                     ple_conv_state_[layer] +
                         static_cast<std::size_t>(slot) *
                             ple_conv_state_values_,
                     0, ple_conv_state_values_ * sizeof(float)),
                 "zero PLE convolution state");
    auto state = std::make_shared<RequestState>(*this, slot);
    state->target_mirror_enabled = target_mirror_page_capacity_ != 0U;
    state->ple_history.assign(
        static_cast<std::size_t>(ple_layers_) * (ple_ngram_size_ - 1U),
        ple_eos_token_id_);
    state->ple_retention_history = state->ple_history;
    return {er::Status::success(), std::move(state)};
  } catch (const std::exception& error) {
    return {{er::ErrorCode::internal, error.what()}, {}};
  }
}

er::Status DenseFp4Provider::rebind_request_state(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const er::ProgramRequestContext& request) {
  try {
    std::lock_guard lock(mutex_);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    const auto first = request.parameters.find(
        "sampling_first_output_position");
    const auto presence = request.parameters.find(
        "sampling_presence_penalty_biased_ppm");
    if (!state || state->parked() || state->slot() >= capacity_ ||
        first == request.parameters.end() || first->second >= max_context_ ||
        presence == request.parameters.end() || presence->second > 4'000'000U)
      throw std::runtime_error("sampling request rebind is invalid");
    cuda_check(cudaMemset(
                   sampling_presence_ +
                       static_cast<std::size_t>(state->slot()) *
                           vocabulary_size_,
                   0, vocabulary_size_),
               "reset request sampling presence");
    return er::Status::success();
  } catch (const std::exception& error) {
    return {er::ErrorCode::internal, error.what()};
  }
}

er::ExecutionValue DenseFp4Provider::device_hidden_value(
    std::uint32_t rows) const {
  if (!rows || rows > workspace_rows_)
    throw std::runtime_error("invalid FP4 hidden batch size");
  return {std::string(kHiddenAbi), "cuda.device", device_lifetime_,
          reinterpret_cast<const std::byte*>(hidden_),
          static_cast<std::uint64_t>(rows) * hidden_size_ * sizeof(float)};
}

er::ExecutionValue DenseFp4Provider::device_value(
    std::string_view abi, const void* pointer, std::uint64_t bytes) const {
  if (abi.empty() || !pointer || !bytes)
    throw std::runtime_error("invalid FP4 device value");
  return {std::string(abi), "cuda.device", device_lifetime_,
          reinterpret_cast<const std::byte*>(pointer), bytes};
}

void DenseFp4Provider::require_device_value(
    const er::ExecutionValue& value, std::string_view abi,
    const void* pointer, std::uint64_t bytes) const {
  if (value.abi != abi || value.memory_domain != "cuda.device" ||
      value.data != reinterpret_cast<const std::byte*>(pointer) ||
      value.bytes != bytes)
    throw std::runtime_error("FP4 intermediate value ABI mismatch");
}

er::OperationExecutionHandle DenseFp4Provider::completed_operation(
    er::OperationExecutionResult result) const {
  struct State final {
    er::OperationExecutionResult result;
    bool terminal{};
  };
  auto state = std::make_shared<State>();
  state->result = std::move(result);
  return er::OperationExecutionHandle::from_callbacks(
      [state]() -> std::optional<er::OperationExecutionResult> {
        if (state->terminal) return std::nullopt;
        state->terminal = true;
        return std::move(state->result);
      },
      [state] { state->terminal = true; });
}

er::ExactDecodeExecutionHandle DenseFp4Provider::completed_exact(
    er::ExactDecodeExecutionResult result) const {
  struct State final {
    er::ExactDecodeExecutionResult result;
    bool terminal{};
  };
  auto state = std::make_shared<State>();
  state->result = std::move(result);
  return er::ExactDecodeExecutionHandle::from_callbacks(
      [state]() -> std::optional<er::ExactDecodeExecutionResult> {
        if (state->terminal) return std::nullopt;
        state->terminal = true;
        return std::move(state->result);
      },
      [state] { state->terminal = true; });
}

const er::ExecutionValue& DenseFp4Provider::invocation_input(
    const PreparedOperation& operation,
    const er::OperationInvocation& invocation, std::string_view port) const {
  const auto found = operation.input_indices.find(port);
  if (found == operation.input_indices.end() ||
      found->second >= invocation.inputs.size())
    throw std::runtime_error("operation input port is absent");
  return invocation.inputs[found->second];
}

std::uint32_t DenseFp4Provider::require_hidden_value(
    const er::ExecutionValue& value) const {
  const auto row_bytes =
      static_cast<std::uint64_t>(hidden_size_) * sizeof(float);
  if (value.abi != kHiddenAbi || value.memory_domain != "cuda.device" ||
      value.data != reinterpret_cast<const std::byte*>(hidden_) ||
      value.bytes % row_bytes != 0U || value.bytes / row_bytes == 0U ||
      value.bytes / row_bytes > workspace_rows_)
    throw std::runtime_error("hidden-state execution ABI mismatch");
  return static_cast<std::uint32_t>(value.bytes / row_bytes);
}

GpuPhase DenseFp4Provider::gpu_phase(Kernel kernel) {
  switch (kernel) {
    case Kernel::embedding:
    case Kernel::hyper_initialize:
    case Kernel::ple:
      return GpuPhase::embedding;
    case Kernel::hyper_read:
    case Kernel::hyper_inject:
    case Kernel::hyper_reduce:
      return GpuPhase::ffn;
    case Kernel::vision:
      return GpuPhase::vision;
    case Kernel::full_attention:
      return GpuPhase::full_attention;
    case Kernel::recurrent_attention:
      return GpuPhase::recurrent_attention;
    case Kernel::router:
      return GpuPhase::router;
    case Kernel::routed_moe:
      return GpuPhase::routed_moe;
    case Kernel::ffn:
      return GpuPhase::ffn;
    case Kernel::head:
      return GpuPhase::head;
    case Kernel::exact_decode:
      return GpuPhase::mtp;
  }
  throw std::runtime_error("unknown dense FP4 GPU phase");
}

std::size_t DenseFp4Provider::begin_gpu_phase(GpuPhase phase) {
  if (!profile_gpu_phases_)
    return std::numeric_limits<std::size_t>::max();
  if (active_gpu_events_ == gpu_event_pool_.size()) {
    GpuEventPair event;
    cuda_check(cudaEventCreate(&event.start), "create GPU phase start event");
    try {
      cuda_check(cudaEventCreate(&event.stop), "create GPU phase stop event");
    } catch (...) {
      static_cast<void>(cudaEventDestroy(event.start));
      throw;
    }
    gpu_event_pool_.push_back(event);
  }
  const auto index = active_gpu_events_++;
  auto& event = gpu_event_pool_[index];
  event.phase = phase;
  cuda_check(cudaEventRecord(event.start), "record GPU phase start");
  return index;
}

void DenseFp4Provider::end_gpu_phase(std::size_t index) {
  if (!profile_gpu_phases_) return;
  if (index >= active_gpu_events_)
    throw std::runtime_error("GPU phase event index is invalid");
  cuda_check(cudaEventRecord(gpu_event_pool_[index].stop),
             "record GPU phase stop");
}

void DenseFp4Provider::collect_gpu_phases() {
  if (!profile_gpu_phases_) return;
  if (!active_gpu_events_) return;
  cuda_check(cudaEventSynchronize(gpu_event_pool_[active_gpu_events_ - 1U].stop),
             "synchronize GPU phase events");
  for (std::size_t index = 0U; index < active_gpu_events_; ++index) {
    float milliseconds{};
    cuda_check(cudaEventElapsedTime(&milliseconds,
                                    gpu_event_pool_[index].start,
                                    gpu_event_pool_[index].stop),
               "read GPU phase elapsed time");
    const auto nanoseconds = static_cast<std::uint64_t>(
        std::llround(static_cast<double>(milliseconds) * 1'000'000.0));
    gpu_phase_ns_[static_cast<std::size_t>(gpu_event_pool_[index].phase)] +=
        nanoseconds;
  }
  active_gpu_events_ = 0U;
  ++gpu_measured_batches_;
}

void DenseFp4Provider::normalize_rows(const float* input, const float* weight,
                                      float* output, std::uint32_t rows) {
  if (zero_centered_norm_)
    status_check(ec::zero_centered_rms_norm_batch(
        input, weight, output, rows, hidden_size_, epsilon_, nullptr));
  else
    status_check(ec::rms_norm_batch(input, weight, output, rows, hidden_size_,
                                    epsilon_, nullptr));
}

void DenseFp4Provider::normalize_operation_input(
    const PreparedOperation& operation, const float* input, float* output,
    std::uint32_t rows) {
  if (operation.abi_version < 2U) {
    const auto& norm = binding(operation, "input_norm");
    if (norm.encoding == "BF16")
      status_check(ec::rms_norm_bf16_weight_batch(
          input, norm.bf16, output, rows, hidden_size_, epsilon_, nullptr));
    else
      normalize_rows(input, norm.f32, output, rows);
    if (norm.encoding == "BF16")
      status_check(ec::round_bf16_in_place(
          output, static_cast<std::uint64_t>(rows) * hidden_size_, nullptr));
    return;
  }
  const auto mode = parameter_u32(operation.parameters, "input_norm_mode");
  const auto epsilon = parameter_f32(
      operation.parameters, "input_norm_epsilon_f32_bits");
  if (mode == 0U)
    status_check(ec::rms_norm_batch(
        input, binding(operation, "input_norm").f32, output, rows,
        hidden_size_, epsilon, nullptr));
  else if (mode == 1U)
    status_check(ec::zero_centered_rms_norm_batch(
        input, binding(operation, "input_norm").f32, output, rows,
        hidden_size_, epsilon, nullptr));
  else
    throw std::runtime_error("unsupported operation input norm mode");
}

void DenseFp4Provider::finish_attention_block(
    const PreparedOperation& operation, std::uint32_t rows) {
  const auto values = rows * hidden_size_;
  if (operation.abi_version < 2U) {
    status_check(ec::add_in_place(hidden_, residual_, values, nullptr,
                                  activation_bf16_));
    if (operation.capability ==
        "block.mla.causal.latent-kv.bfloat16.v1")
      status_check(ec::round_bf16_in_place(hidden_, values, nullptr));
    return;
  }
  const auto mode = parameter_u32(operation.parameters, "post_norm_mode");
  const auto epsilon = parameter_f32(
      operation.parameters, "post_norm_epsilon_f32_bits");
  if (mode == 0U)
    status_check(ec::rms_norm_batch(
        residual_, binding(operation, "post_norm").f32, normalized_, rows,
        hidden_size_, epsilon, nullptr));
  else if (mode == 1U)
    status_check(ec::zero_centered_rms_norm_batch(
        residual_, binding(operation, "post_norm").f32, normalized_, rows,
        hidden_size_, epsilon, nullptr));
  else
    throw std::runtime_error("unsupported attention post norm mode");
  status_check(ec::add_in_place(hidden_, normalized_, values, nullptr,
                                activation_bf16_));
}

void DenseFp4Provider::run_embedding(
    const PreparedOperation& operation, const std::uint32_t* tokens,
    std::uint32_t rows) {
  if (!tokens || !rows || rows > workspace_rows_)
    throw std::runtime_error("invalid embedding batch");
  cuda_check(cudaMemcpy(output_tokens_, tokens,
                        static_cast<std::size_t>(rows) * sizeof(tokens[0]),
                        cudaMemcpyHostToDevice),
             "upload embedding token batch");
  const auto& weight = binding(operation, "weight");
  if (weight.encoding == "FP4_E2M1") {
    status_check(ec::fp4_embedding_batch(
        weight.matrix(), output_tokens_, hidden_, rows, nullptr));
  } else if (weight.encoding == "MXFP6_E3M2") {
    status_check(ec::mxfp6_embedding_batch(
        weight.mxfp6_matrix(), output_tokens_, hidden_, rows, nullptr));
  } else if (weight.encoding == "I8") {
    const auto matrix = weight.int8_matrix();
    for (std::uint32_t row = 0U; row < rows; ++row)
      status_check(ec::embedding(
          matrix, tokens[row],
          hidden_ + static_cast<std::size_t>(row) * hidden_size_, nullptr));
  } else if (weight.encoding == "BF16") {
    status_check(ec::embedding_bf16_batch(
        weight.bf16, vocabulary_size_, hidden_size_, output_tokens_, hidden_,
        rows, nullptr));
  } else {
    throw std::runtime_error("unsupported embedding tensor encoding");
  }
  if (operation.abi_version >= 2U) {
    if (parameter_u32(operation.parameters, "output_norm_mode") != 1U)
      throw std::runtime_error("unsupported embedding output norm mode");
    status_check(ec::weightless_rms_norm_batch(
        hidden_, normalized_, rows, hidden_size_,
        parameter_f32(operation.parameters,
                      "output_norm_epsilon_f32_bits"),
        nullptr));
    cuda_check(cudaMemcpy(hidden_, normalized_,
                          static_cast<std::size_t>(rows) * hidden_size_ *
                              sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "apply embedding output norm");
  }
}

void DenseFp4Provider::run_hyper_read(
    const PreparedOperation& operation, std::uint32_t rows, bool reduce) {
  if (!hyper_enabled_ || !rows || rows > workspace_rows_ ||
      (reduce && operation.kernel != Kernel::hyper_reduce) ||
      (!reduce && operation.kernel != Kernel::hyper_read))
    throw std::runtime_error("invalid Hyper read batch");
  const auto epsilon = parameter_f32(
      operation.parameters, "norm_epsilon_f32_bits");
  status_check(ec::hyper_group_norm_batch(
      hyper_, binding(operation, "norm").f32, hyper_normalized_, rows,
      hidden_size_, hyper_count_, epsilon, nullptr));
  project(binding(operation, "mix_down"), hyper_normalized_,
          hyper_lowrank_values_, rows);
  status_check(ec::hyper_prepare_mix_batch(
      hyper_lowrank_values_, rows * hyper_lowrank_, hyper_count_, nullptr));
  project(binding(operation, "mix_up"), hyper_lowrank_values_, hyper_mix_,
          rows);
  if (reduce) {
    cuda_check(cudaMemset(hyper_injection_, 0,
                          static_cast<std::size_t>(rows) * hyper_count_ *
                              sizeof(float)),
               "clear unused Hyper injection output");
  } else {
    project(binding(operation, "inject"), hyper_normalized_,
            hyper_injection_, rows);
  }
  status_check(ec::hyper_finish_read_batch(
      hyper_normalized_, hyper_mix_, hidden_, hyper_injection_, rows,
      hidden_size_, hyper_count_, nullptr));
}

void DenseFp4Provider::run_ple(
    const PreparedOperation& operation, RequestState& state,
    std::span<const std::uint32_t> tokens, std::uint32_t rows) {
  if (!ple_enabled_ || operation.kernel != Kernel::ple ||
      operation.ple_slot >= ple_layers_ || rows == 0U ||
      rows > workspace_rows_ || tokens.size() != rows ||
      state.ple_history.size() !=
          static_cast<std::size_t>(ple_layers_) *
              (ple_ngram_size_ - 1U))
    throw std::runtime_error("invalid PLE batch");
  const auto& multipliers = host_binding(operation, "layer_multipliers");
  const auto& vocab_sizes = host_binding(operation, "head_vocab_sizes");
  const auto& offsets = host_binding(operation, "head_offsets");
  auto* history = state.ple_history.data() +
      static_cast<std::size_t>(operation.ple_slot) *
          (ple_ngram_size_ - 1U);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    std::array<std::uint32_t, 8U> shifted{};
    shifted[0] = tokens[row];
    for (std::uint32_t shift = 1U; shift < ple_ngram_size_; ++shift)
      shifted[shift] = history[shift - 1U];
    for (std::uint32_t ngram = 2U; ngram <= ple_ngram_size_; ++ngram) {
      std::uint64_t mixed =
          static_cast<std::uint64_t>(shifted[0]) *
          static_cast<std::uint64_t>(host_i64(multipliers, 0U));
      for (std::uint32_t position = 1U; position < ngram; ++position)
        mixed ^= static_cast<std::uint64_t>(shifted[position]) *
                 static_cast<std::uint64_t>(
                     host_i64(multipliers, position));
      const auto signed_mixed = std::bit_cast<std::int64_t>(mixed);
      const auto first_head = (ngram - 2U) * ple_heads_per_ngram_;
      for (std::uint32_t local = 0U; local < ple_heads_per_ngram_; ++local) {
        const auto head = first_head + local;
        const auto modulus = host_i64(vocab_sizes, head);
        const auto offset = host_i64(offsets, head);
        if (modulus <= 0 || offset < 0)
          throw std::runtime_error("PLE hash metadata is invalid");
        auto remainder = signed_mixed % modulus;
        if (remainder < 0) remainder += modulus;
        const auto global_row =
            static_cast<std::uint64_t>(offset + remainder);
        const auto shard = global_row / ple_rows_per_shard_;
        const auto shard_row = global_row % ple_rows_per_shard_;
        if (shard >= ple_shard_count_)
          throw std::runtime_error("PLE hash exceeds its embedding shards");
        const auto& table = host_binding(
            operation, std::string("embedding_shard.") +
                           std::to_string(shard));
        decode_host_quantized_row(
            table, shard_row,
            ple_host_embeddings_.data() +
                (static_cast<std::size_t>(row) * ple_head_count_ + head) *
                    ple_head_width_);
      }
    }
    if (tokens[row] == ple_eos_token_id_) {
      std::fill(history, history + ple_ngram_size_ - 1U,
                ple_eos_token_id_);
    } else {
      for (std::uint32_t shift = ple_ngram_size_ - 1U; shift > 1U; --shift)
        history[shift - 1U] = history[shift - 2U];
      history[0] = tokens[row];
    }
  }
  cuda_check(cudaMemcpy(
                 ple_embeddings_, ple_host_embeddings_.data(),
                 static_cast<std::size_t>(rows) * ple_embedding_width_ *
                     sizeof(float),
                 cudaMemcpyHostToDevice),
             "upload selected PLE embedding rows");
  project(binding(operation, "key_projection"), ple_embeddings_, ple_key_,
          rows);
  project(binding(operation, "value_projection"), ple_embeddings_,
          ple_value_, rows);
  status_check(ec::hyper_group_norm_batch(
      ple_key_, binding(operation, "key_norm").f32, ple_key_norm_, rows,
      hidden_size_, hyper_count_, epsilon_, nullptr));
  status_check(ec::hyper_group_norm_batch(
      hyper_, binding(operation, "query_norm").f32, ple_query_norm_, rows,
      hidden_size_, hyper_count_, epsilon_, nullptr));
  status_check(ec::ple_gate_batch(
      ple_key_norm_, ple_query_norm_, ple_value_, ple_gated_, rows,
      hidden_size_, hyper_count_, nullptr));
  status_check(ec::hyper_group_norm_batch(
      ple_gated_, binding(operation, "convolution_norm").f32,
      ple_conv_norm_, rows, hidden_size_, hyper_count_, epsilon_, nullptr));
  auto* conv_state = ple_conv_state_.at(operation.ple_slot) +
      static_cast<std::size_t>(state.slot()) * ple_conv_state_values_;
  status_check(ec::ple_dilated_conv(
      {ple_conv_norm_, binding(operation, "convolution").dequantized,
       conv_state, ple_conv_output_, rows, hyper_width_,
       ple_convolution_kernel_, ple_ngram_size_, nullptr}));
  status_check(ec::add_in_place(ple_gated_, ple_conv_output_,
                                rows * hyper_width_, nullptr,
                                activation_bf16_));
  status_check(ec::add_in_place(hyper_, ple_gated_, rows * hyper_width_,
                                nullptr, activation_bf16_));
}

void DenseFp4Provider::quantize_rows(const float* input, std::uint32_t rows,
                                     std::uint32_t columns) {
  const auto padded = align32(columns);
  if (activation_bf16_)
    status_check(ec::quantize_q8_batch_bf16(
        input, q8_, q8_scales_, rows, columns, padded, nullptr));
  else
    status_check(ec::quantize_q8_batch(
      input, q8_, q8_scales_, rows, columns, padded, nullptr));
}

void DenseFp4Provider::prepare_dense_activation(
    const float* input, std::uint32_t rows, std::uint32_t columns) {
  if (dense_activation_input_bf16_) {
    status_check(ec::convert_f32_to_bf16_batch(
        input, staged_dense_input_, staged_dense_input_capacity_bytes_, rows,
        columns, align32(columns), nullptr));
    return;
  }
  quantize_rows(input, rows, columns);
}

bool DenseFp4Provider::stage_operation_weights(
    const PreparedOperation& operation) {
  if (staged_operation_ == &operation) return staged_dense_weights_ != nullptr;
  if (!ensure_staged_dense_weights()) return false;
  staged_weight_bindings_.clear();
  std::set<const DeviceTensor*> unique;
  auto* cursor = reinterpret_cast<std::byte*>(staged_dense_weights_);
  std::size_t used{};
  for (const auto& [role, tensor] : operation.tensors) {
    static_cast<void>(role);
    if (!tensor || tensor->encoding != "FP4_E2M1" ||
        tensor->quant_abi != er::kExpertQuantAbiFp4Block32 ||
        tensor->shape.size() != 2U || !unique.insert(tensor).second)
      continue;
    const auto matrix = tensor->matrix();
    const auto bytes = static_cast<std::size_t>(matrix.rows) *
                       matrix.padded_columns * sizeof(std::uint16_t);
    if (!staged_dense_weights_ ||
        used > staged_dense_weight_capacity_bytes_ ||
        bytes > staged_dense_weight_capacity_bytes_ - used)
      throw std::runtime_error("staged dense weight workspace is exhausted");
    status_check(ec::fp4_decode_matrix_bf16(matrix, cursor, bytes, nullptr));
    staged_weight_bindings_.emplace(
        tensor, StagedDenseWeight{cursor, bytes});
    cursor += bytes;
    used += bytes;
  }
  staged_operation_ = &operation;
  staged_dense_weight_decode_bytes_ += used;
  return true;
}

bool DenseFp4Provider::ensure_staged_dense_weights() {
  if (staged_dense_weights_ || staged_dense_weight_capacity_bytes_ == 0U)
    return staged_dense_weights_ != nullptr;
  std::size_t free_bytes{};
  std::size_t total_bytes{};
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
             "inspect staged dense weight headroom");
  const auto reserve_bytes = std::min<std::uint64_t>(
      static_cast<std::uint64_t>(total_bytes) / 8U, 1ULL << 30U);
  if (staged_dense_weight_capacity_bytes_ >
          std::numeric_limits<std::uint64_t>::max() - reserve_bytes ||
      static_cast<std::uint64_t>(free_bytes) <
          reserve_bytes + staged_dense_weight_capacity_bytes_)
    return false;
  void* allocation{};
  const auto error =
      cudaMalloc(&allocation, staged_dense_weight_capacity_bytes_);
  if (error == cudaErrorMemoryAllocation) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  cuda_check(error, "allocate staged dense weight workspace");
  staged_dense_weights_ = static_cast<std::uint16_t*>(allocation);
  return true;
}

void DenseFp4Provider::release_staged_dense_weights() noexcept {
  deactivate_staged_weights();
  staged_weight_bindings_.clear();
  staged_operation_ = nullptr;
  if (staged_dense_weights_)
    static_cast<void>(cudaFree(staged_dense_weights_));
  staged_dense_weights_ = nullptr;
}

bool DenseFp4Provider::activate_staged_weights(
    const PreparedOperation& operation) {
  if (!stage_operation_weights(operation)) return false;
  active_staged_weights_ = staged_weight_bindings_;
  return true;
}

void DenseFp4Provider::deactivate_staged_weights() noexcept {
  active_staged_weights_.clear();
}

std::uint64_t DenseFp4Provider::pending_sequence_kv_bytes(
    const SequenceState& sequence,
    const PreparedOperation& operation) const {
  if (operation.kernel != Kernel::full_attention || kv_page_bytes_ == 0U)
    return 0U;
  const auto slot = sequence.request->slot();
  std::uint64_t pages{};
  auto previous = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t row = 0U; row < sequence.tile_rows; ++row) {
    const auto position = sequence.positions[sequence.tile_first + row];
    const auto page = position / kv_page_tokens_;
    if (page == previous) continue;
    previous = page;
    if (!slot_pages_.at(slot).at(page)) ++pages;
  }
  return checked_multiply(pages, kv_page_bytes_,
                          "pending sequence KV bytes");
}

bool DenseFp4Provider::sequence_staging_has_headroom(
    SequenceState& sequence, const PreparedOperation& operation) {
  if (sequence.staging_disabled) return false;
  std::size_t free_bytes{};
  std::size_t total_bytes{};
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
             "inspect sequence staging headroom");
  const auto reserve_bytes = std::min<std::uint64_t>(
      static_cast<std::uint64_t>(total_bytes) / 8U, 1ULL << 30U);
  const auto pending_kv = pending_sequence_kv_bytes(sequence, operation);
  auto required = reserve_bytes;
  if (pending_kv > std::numeric_limits<std::uint64_t>::max() - required)
    throw std::runtime_error("sequence staging headroom overflows");
  required += pending_kv;
  if (!staged_dense_weights_) {
    if (staged_dense_weight_capacity_bytes_ >
        std::numeric_limits<std::uint64_t>::max() - required)
      throw std::runtime_error("sequence staging allocation overflows");
    required += staged_dense_weight_capacity_bytes_;
  }
  if (static_cast<std::uint64_t>(free_bytes) >= required) return true;
  release_staged_dense_weights();
  sequence.staging_disabled = true;
  ++staged_dense_weight_low_memory_fallbacks_;
  return false;
}

void DenseFp4Provider::ensure_logits_capacity(std::uint32_t rows) {
  if (rows == 0U || rows > workspace_rows_)
    throw std::runtime_error("invalid logits workspace row count");
  if (rows <= logits_capacity_rows_) return;
  const auto values = checked_multiply(
      rows, vocabulary_size_, "logits workspace values");
  const auto bytes = checked_multiply(
      values, sizeof(float), "logits workspace bytes");
  if (bytes > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("logits workspace exceeds address space");
  void* replacement{};
  cuda_check(cudaMalloc(&replacement, static_cast<std::size_t>(bytes)),
             "allocate logits workspace");
  if (logits_) {
    const auto released = cudaFree(logits_);
    if (released != cudaSuccess) {
      static_cast<void>(cudaFree(replacement));
      cuda_check(released, "release previous logits workspace");
    }
  }
  logits_ = static_cast<float*>(replacement);
  logits_capacity_rows_ = rows;
}

void DenseFp4Provider::ensure_sequence_workspace() {
  if (sequence_hidden_ && (!hyper_enabled_ ||
                           (sequence_hyper_ && sequence_injection_)))
    return;
  if (!sequence_hidden_values_)
    throw std::runtime_error("sequence workspace geometry is empty");
  if (!sequence_hidden_)
    sequence_hidden_ =
        device_allocate<float>(allocations_, sequence_hidden_values_);
  if (hyper_enabled_) {
    if (!sequence_hyper_values_ || !sequence_injection_values_)
      throw std::runtime_error("Hyper sequence workspace geometry is empty");
    if (!sequence_hyper_)
      sequence_hyper_ =
          device_allocate<float>(allocations_, sequence_hyper_values_);
    if (!sequence_injection_)
      sequence_injection_ =
          device_allocate<float>(allocations_, sequence_injection_values_);
  }
}

void DenseFp4Provider::release_sequence_workspace() noexcept {
  const auto release = [this](float*& allocation) {
    if (!allocation) return;
    const auto pointer = allocation;
    if (cudaFree(pointer) != cudaSuccess) return;
    const auto found = std::find(allocations_.begin(), allocations_.end(),
                                 static_cast<void*>(pointer));
    if (found != allocations_.end()) *found = nullptr;
    allocation = nullptr;
  };
  release(sequence_injection_);
  release(sequence_hyper_);
  release(sequence_hidden_);
}

void DenseFp4Provider::project_quantized(const DeviceTensor& weight,
                                         const float* input, float* output,
                                         std::uint32_t rows) {
  if (weight.encoding == "I8") {
    const auto matrix = weight.int8_matrix();
    const auto matrix_values = static_cast<std::uint64_t>(matrix.rows) *
                               matrix.columns;
    if (rows > 8U && staged_int8_weights_ &&
        matrix_values <= staged_int8_weight_capacity_values_)
      status_check(ec::int8_gemm_f32_batch(
          matrix, input, output, rows, staged_int8_weights_,
          staged_int8_weight_capacity_values_, nullptr));
    else if (rows > 1U)
      status_check(ec::gemv_batch_weight_reuse(
          matrix, input, output, rows, nullptr));
    else
      status_check(ec::gemv(matrix, input, output, nullptr));
    return;
  }
  if (weight.encoding == "BF16") {
    if (!weight.bf16 || weight.shape.size() != 2U)
      throw std::runtime_error("invalid BF16 projection matrix");
    if (rows > 1U)
      status_check(ec::gemv_bf16_batch(
          weight.bf16, weight.shape[0], weight.shape[1], input, output,
          rows, nullptr));
    else
      status_check(ec::gemv_bf16(
          weight.bf16, weight.shape[0], weight.shape[1], input, output,
          nullptr));
    return;
  }
  if (weight.encoding == "MXFP6_E3M2") {
    if (dense_activation_input_bf16_)
      throw std::runtime_error(
          "BF16 dense activation input does not support MXFP6 matrices");
    status_check(ec::mxfp6_gemv_q8_batch_weight_reuse(
        weight.mxfp6_matrix(), q8_, q8_scales_, output, rows, nullptr));
    return;
  }
  if (weight.encoding == "FP4_E2M1" &&
      weight.quant_abi == er::kExpertRecordAbiNvfp4Block16W4A4) {
    status_check(ec::nvfp4_gemv_f32_batch(
        weight.nvfp4_matrix(), input, output, nvfp4_input_, rows, nullptr));
    return;
  }
  const auto matrix = weight.matrix();
  const auto staged = active_staged_weights_.find(&weight);
  if (dense_activation_input_bf16_) {
    if (rows > 8U && staged != active_staged_weights_.end()) {
      ++staged_dense_weight_gemm_calls_;
      status_check(ec::bf16_gemm_bf16_block32(
          matrix, staged->second.data, staged->second.bytes,
          staged_dense_input_, staged_dense_input_capacity_bytes_, output,
          rows, nullptr, activation_bf16_));
    } else {
      status_check(ec::fp4_gemm_bf16_block32(
          matrix, staged_dense_input_, staged_dense_input_capacity_bytes_,
          output, rows, nullptr, activation_bf16_));
    }
    return;
  }
  if (rows > 8U && staged != active_staged_weights_.end()) {
    ++staged_dense_weight_gemm_calls_;
    status_check(ec::bf16_gemm_q8_block32(
        matrix, staged->second.data, staged->second.bytes, q8_, q8_scales_,
        staged_dense_input_, staged_dense_input_capacity_bytes_, output, rows,
        nullptr, activation_bf16_));
  } else if (rows > 8U)
    status_check(ec::fp4_gemm_q8_block32(
        matrix, q8_, q8_scales_, output, rows, nullptr,
        activation_bf16_));
  else if (rows > 1U)
    status_check(ec::fp4_gemv_q8_batch_weight_reuse(
        matrix, q8_, q8_scales_, output, rows, nullptr,
        activation_bf16_));
  else
    status_check(ec::fp4_gemv_q8_batch(matrix, q8_, q8_scales_, output, rows,
                                       nullptr, activation_bf16_));
}

void DenseFp4Provider::project(const DeviceTensor& weight, const float* input,
                               float* output, std::uint32_t rows) {
  if (weight.encoding == "FP4_E2M1" &&
      weight.quant_abi == er::kExpertQuantAbiFp4Block32)
    prepare_dense_activation(input, rows, weight.matrix_columns());
  else if (weight.encoding == "MXFP6_E3M2")
    prepare_dense_activation(input, rows, weight.matrix_columns());
  project_quantized(weight, input, output, rows);
}

void DenseFp4Provider::project_prefix(const DeviceTensor& weight,
                                      const float* input, float* output,
                                      std::uint32_t matrix_rows) {
  if (weight.encoding == "MXFP6_E3M2") {
    if (dense_activation_input_bf16_)
      throw std::runtime_error(
          "BF16 dense activation input does not support MXFP6 matrices");
    auto matrix = weight.mxfp6_matrix();
    if (!matrix_rows || matrix_rows > matrix.rows)
      throw std::runtime_error("MXFP6 projection prefix is invalid");
    matrix.rows = matrix_rows;
    prepare_dense_activation(input, 1U, matrix.columns);
    status_check(ec::mxfp6_gemv_q8_batch_weight_reuse(
        matrix, q8_, q8_scales_, output, 1U, nullptr));
    return;
  }
  auto matrix = weight.matrix();
  if (!matrix_rows || matrix_rows > matrix.rows)
    throw std::runtime_error("FP4 projection prefix is invalid");
  matrix.rows = matrix_rows;
  prepare_dense_activation(input, 1U, matrix.columns);
  if (dense_activation_input_bf16_)
    status_check(ec::fp4_gemm_bf16_block32(
        matrix, staged_dense_input_, staged_dense_input_capacity_bytes_,
        output, 1U, nullptr, activation_bf16_));
  else
    status_check(ec::fp4_gemv_q8_batch(
        matrix, q8_, q8_scales_, output, 1U, nullptr, activation_bf16_));
}

void DenseFp4Provider::project_vision(const DeviceTensor& weight,
                                      const float* input, float* output,
                                      std::uint32_t rows) {
  if (!rows || rows > kMaximumVisionPatches)
    throw std::runtime_error("vision projection row count is invalid");
  const auto matrix = weight.matrix();
  for (std::uint32_t first = 0U; first < rows; first += workspace_rows_) {
    const auto chunk = std::min(workspace_rows_, rows - first);
    project(weight, input + static_cast<std::size_t>(first) * matrix.columns,
            output + static_cast<std::size_t>(first) * matrix.rows, chunk);
  }
}

void DenseFp4Provider::run_vision(const PreparedOperation& operation,
                                  const MediaPayload& media,
                                  float* tile_hidden,
                                  std::uint32_t prompt_rows,
                                  std::uint32_t tile_first,
                                  std::uint32_t tile_rows, bool prepare) {
  if (operation.kernel != Kernel::vision || !vision_enabled_ || media.empty())
    return;
  if (media.patch_count > kMaximumVisionPatches ||
      media.patch_dimension != vision_patch_dimension_ ||
      !tile_hidden || !prompt_rows || !tile_rows ||
      tile_first > prompt_rows || tile_rows > prompt_rows - tile_first)
    throw std::runtime_error("vision execution geometry is invalid");
  const auto merge_area = vision_spatial_merge_size_ *
                          vision_spatial_merge_size_;
  if (media.patch_count % merge_area)
    throw std::runtime_error("vision patches do not form merge groups");
  const auto merged_rows = media.patch_count / merge_area;

  if (prepare) {
  std::vector<std::uint32_t> row_positions(2U * media.patch_count);
  std::vector<std::uint32_t> segment_first(media.patch_count);
  std::vector<std::uint32_t> segment_last(media.patch_count);
  std::vector<std::uint32_t> interpolation_indices(4U * media.patch_count);
  std::vector<float> interpolation_weights(4U * media.patch_count);
  for (const auto& image : media.images) {
    const auto patches = image.temporal * image.height * image.width;
    const auto image_last = image.patch_offset + patches;
    const auto blocks_w = image.width / vision_spatial_merge_size_;
    for (std::uint32_t local = 0U; local < patches; ++local) {
      const auto patch = image.patch_offset + local;
      const auto spatial = local % (image.height * image.width);
      const auto in_column = spatial % vision_spatial_merge_size_;
      const auto in_row = (spatial / vision_spatial_merge_size_) %
                          vision_spatial_merge_size_;
      const auto block_column =
          (spatial / merge_area) % blocks_w;
      const auto block_row = spatial / (merge_area * blocks_w);
      const auto row = block_row * vision_spatial_merge_size_ + in_row;
      const auto column =
          block_column * vision_spatial_merge_size_ + in_column;
      row_positions[2U * patch] = row;
      row_positions[2U * patch + 1U] = column;
      segment_first[patch] = image.patch_offset;
      segment_last[patch] = image_last;

      const auto axis_taps = [this](std::uint32_t index,
                                    std::uint32_t size) {
        const auto coordinate = size <= 1U
            ? 0.0F
            : static_cast<float>(index) *
                  static_cast<float>(vision_grid_side_ - 1U) /
                  static_cast<float>(size - 1U);
        const auto low = static_cast<std::uint32_t>(std::floor(coordinate));
        const auto high = std::min(low + 1U, vision_grid_side_ - 1U);
        const auto high_weight = coordinate - static_cast<float>(low);
        return std::array<std::pair<std::uint32_t, float>, 2U>{
            std::pair{low, 1.0F - high_weight},
            std::pair{high, high_weight}};
      };
      const auto row_taps = axis_taps(row, image.height);
      const auto column_taps = axis_taps(column, image.width);
      std::uint32_t tap{};
      for (const auto& [source_row, row_weight] : row_taps) {
        for (const auto& [source_column, column_weight] : column_taps) {
          interpolation_indices[4U * patch + tap] =
              source_row * vision_grid_side_ + source_column;
          interpolation_weights[4U * patch + tap] =
              row_weight * column_weight;
          ++tap;
        }
      }
    }
  }

  cuda_check(cudaMemcpy(vision_pixels_, media.pixels.data(),
                        media.pixels.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "upload vision pixels");
  cuda_check(cudaMemcpy(vision_positions_, row_positions.data(),
                        row_positions.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice),
             "upload vision positions");
  cuda_check(cudaMemcpy(vision_segment_first_, segment_first.data(),
                        segment_first.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice),
             "upload vision segment starts");
  cuda_check(cudaMemcpy(vision_segment_last_, segment_last.data(),
                        segment_last.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice),
             "upload vision segment ends");
  cuda_check(cudaMemcpy(vision_interpolation_indices_,
                        interpolation_indices.data(),
                        interpolation_indices.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice),
             "upload vision interpolation indices");
  cuda_check(cudaMemcpy(vision_interpolation_weights_,
                        interpolation_weights.data(),
                        interpolation_weights.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
             "upload vision interpolation weights");

  const auto& patch = binding(operation, "patch_projection");
  for (std::uint32_t first = 0U; first < media.patch_count;
       first += workspace_rows_) {
    const auto rows = std::min(workspace_rows_, media.patch_count - first);
    status_check(ec::gemv_f32_batch(
        patch.dequantized, vision_hidden_size_, vision_patch_dimension_,
        vision_pixels_ + static_cast<std::size_t>(first) *
                             vision_patch_dimension_,
        vision_hidden_ + static_cast<std::size_t>(first) *
                             vision_hidden_size_,
        rows, nullptr));
  }
  status_check(ec::add_bias_in_place(
      vision_hidden_, binding(operation, "patch_bias").f32,
      media.patch_count, vision_hidden_size_, nullptr));
  status_check(ec::add_fp4_position_interpolation(
      vision_hidden_, binding(operation, "position_embedding").matrix(),
      vision_interpolation_indices_, vision_interpolation_weights_,
      media.patch_count, nullptr));

  const auto role = [](std::uint32_t layer, std::string_view suffix) {
    return std::string("block.") + std::to_string(layer) + "." +
           std::string(suffix);
  };
  for (std::uint32_t layer = 0U; layer < vision_depth_; ++layer) {
    status_check(ec::layer_norm_batch(
        vision_hidden_, binding(operation, role(layer, "norm1_weight")).f32,
        binding(operation, role(layer, "norm1_bias")).f32,
        vision_normalized_, media.patch_count, vision_hidden_size_,
        vision_epsilon_, nullptr));
    project_vision(binding(operation, role(layer, "qkv_projection")),
                   vision_normalized_, vision_qkv_, media.patch_count);
    status_check(ec::add_bias_in_place(
        vision_qkv_, binding(operation, role(layer, "qkv_bias")).f32,
        media.patch_count, 3U * vision_hidden_size_, nullptr));
    status_check(ec::vision_qkv_rope_in_place(
        vision_qkv_, vision_positions_, media.patch_count, vision_heads_,
        vision_head_dim_, vision_rope_theta_, nullptr));
    status_check(ec::vision_segment_attention(
        vision_qkv_, vision_segment_first_, vision_segment_last_,
        vision_attention_, media.patch_count, vision_heads_,
        vision_head_dim_, nullptr));
    project_vision(binding(operation, role(layer, "attention_projection")),
                   vision_attention_, vision_normalized_, media.patch_count);
    status_check(ec::add_bias_in_place(
        vision_normalized_,
        binding(operation, role(layer, "attention_bias")).f32,
        media.patch_count, vision_hidden_size_, nullptr));
    status_check(ec::add_in_place(
        vision_hidden_, vision_normalized_,
        media.patch_count * vision_hidden_size_, nullptr,
        activation_bf16_));

    status_check(ec::layer_norm_batch(
        vision_hidden_, binding(operation, role(layer, "norm2_weight")).f32,
        binding(operation, role(layer, "norm2_bias")).f32,
        vision_normalized_, media.patch_count, vision_hidden_size_,
        vision_epsilon_, nullptr));
    project_vision(binding(operation, role(layer, "mlp_fc1")),
                   vision_normalized_, vision_intermediate_,
                   media.patch_count);
    status_check(ec::add_bias_in_place(
        vision_intermediate_,
        binding(operation, role(layer, "mlp_fc1_bias")).f32,
        media.patch_count, vision_intermediate_size_, nullptr));
    status_check(ec::gelu_tanh_in_place(
        vision_intermediate_,
        media.patch_count * vision_intermediate_size_, nullptr));
    project_vision(binding(operation, role(layer, "mlp_fc2")),
                   vision_intermediate_, vision_normalized_,
                   media.patch_count);
    status_check(ec::add_bias_in_place(
        vision_normalized_,
        binding(operation, role(layer, "mlp_fc2_bias")).f32,
        media.patch_count, vision_hidden_size_, nullptr));
    status_check(ec::add_in_place(
        vision_hidden_, vision_normalized_,
        media.patch_count * vision_hidden_size_, nullptr,
        activation_bf16_));
  }

  status_check(ec::layer_norm_batch(
      vision_hidden_, binding(operation, "merger_norm_weight").f32,
      binding(operation, "merger_norm_bias").f32, vision_normalized_,
      media.patch_count, vision_hidden_size_, vision_epsilon_, nullptr));
  project_vision(binding(operation, "merger_fc1"), vision_normalized_,
                 vision_merger_, merged_rows);
  status_check(ec::add_bias_in_place(
      vision_merger_, binding(operation, "merger_fc1_bias").f32,
      merged_rows, vision_merged_width_, nullptr));
  status_check(ec::gelu_exact_in_place(
      vision_merger_, merged_rows * vision_merged_width_, nullptr));
  project_vision(binding(operation, "merger_fc2"), vision_merger_,
                 vision_output_, merged_rows);
  status_check(ec::add_bias_in_place(
      vision_output_, binding(operation, "merger_fc2_bias").f32,
      merged_rows, vision_output_size_, nullptr));
  ++vision_batches_;
  vision_images_ += media.images.size();
  vision_patches_ += media.patch_count;
  }

  const auto tile_last = tile_first + tile_rows;
  for (const auto& image : media.images) {
    if (image.prompt_offset + image.merged_tokens > prompt_rows)
      throw std::runtime_error("visual embedding target exceeds prompt");
    const auto image_first = image.prompt_offset;
    const auto image_last = image.prompt_offset + image.merged_tokens;
    const auto copy_first = std::max(image_first, tile_first);
    const auto copy_last = std::min(image_last, tile_last);
    if (copy_first >= copy_last) continue;
    const auto output_offset = image.patch_offset / merge_area;
    const auto image_row_offset = copy_first - image_first;
    const auto tile_row_offset = copy_first - tile_first;
    const auto copy_rows = copy_last - copy_first;
    cuda_check(cudaMemcpy(
                   tile_hidden + static_cast<std::size_t>(tile_row_offset) *
                                     hidden_size_,
                   vision_output_ +
                       static_cast<std::size_t>(output_offset +
                                                image_row_offset) *
                           hidden_size_,
                   static_cast<std::size_t>(copy_rows) * hidden_size_ *
                       sizeof(float),
                   cudaMemcpyDeviceToDevice),
               "apply visual embeddings to sequence tile");
  }
}

void DenseFp4Provider::ensure_page(std::uint32_t slot,
                                   std::uint32_t cache_position) {
  if (slot >= capacity_ || cache_position >= max_context_)
    throw std::runtime_error("KV page address exceeds its reservation");
  const auto page_index = cache_position / kv_page_tokens_;
  auto*& page = slot_pages_.at(slot).at(page_index);
  if (page) return;
  if (!free_kv_pages_.empty()) {
    page = free_kv_pages_.back();
    free_kv_pages_.pop_back();
  } else {
    if (all_kv_pages_.size() >= kv_page_capacity_)
      throw std::runtime_error("KV physical page budget is exhausted");
    cuda_check(cudaMalloc(&page, static_cast<std::size_t>(kv_page_bytes_)),
               "allocate KV page");
    all_kv_pages_.push_back(page);
  }
  cuda_check(cudaMemcpy(
                 device_page_table_ +
                     static_cast<std::size_t>(slot) * maximum_pages_per_slot_ +
                     page_index,
                 &page, sizeof(page), cudaMemcpyHostToDevice),
             "publish KV page");
  if (device_mtp_page_table_ != device_page_table_) {
    auto* mtp_page = static_cast<void*>(
        static_cast<std::byte*>(page) + mtp_kv_page_offset_);
    cuda_check(cudaMemcpy(
                   device_mtp_page_table_ +
                       static_cast<std::size_t>(slot) *
                           maximum_pages_per_slot_ +
                       page_index,
                   &mtp_page, sizeof(mtp_page), cudaMemcpyHostToDevice),
               "publish MTP KV page");
  }
}

void DenseFp4Provider::checkpoint_window_layer(
    std::uint32_t slot, std::uint32_t window_layer) {
  if (window_attention_layers_ == 0U) return;
  if (slot >= capacity_ || window_layer >= window_attention_layers_)
    throw std::runtime_error("window checkpoint layer is invalid");
  const auto layer_page_bytes =
      2U * static_cast<std::uint64_t>(kv_page_tokens_) * kv_heads_ *
      head_dim_ * sizeof(std::uint16_t);
  const auto slot_offset =
      static_cast<std::uint64_t>(slot) * window_kv_bytes_per_slot_;
  for (std::uint32_t page = 0U; page < window_pages_per_slot_; ++page) {
    const auto offset = slot_offset +
        static_cast<std::uint64_t>(page) * window_kv_page_bytes_ +
        static_cast<std::uint64_t>(window_layer) * layer_page_bytes;
    cuda_check(cudaMemcpy(window_retention_kv_ + offset, window_kv_ + offset,
                          static_cast<std::size_t>(layer_page_bytes),
                          cudaMemcpyDeviceToDevice),
               "checkpoint window-attention layer");
  }
}

void DenseFp4Provider::checkpoint_window_state(std::uint32_t slot) {
  if (window_attention_layers_ == 0U) return;
  if (slot >= capacity_)
    throw std::runtime_error("window checkpoint slot is invalid");
  const auto offset =
      static_cast<std::uint64_t>(slot) * window_kv_bytes_per_slot_;
  cuda_check(cudaMemcpy(window_retention_kv_ + offset, window_kv_ + offset,
                        static_cast<std::size_t>(window_kv_bytes_per_slot_),
                        cudaMemcpyDeviceToDevice),
             "checkpoint window-attention state");
}

void DenseFp4Provider::restore_window_checkpoint(std::uint32_t slot) {
  if (window_attention_layers_ == 0U) return;
  if (slot >= capacity_)
    throw std::runtime_error("window rewind slot is invalid");
  const auto offset =
      static_cast<std::uint64_t>(slot) * window_kv_bytes_per_slot_;
  cuda_check(cudaMemcpy(window_kv_ + offset, window_retention_kv_ + offset,
                        static_cast<std::size_t>(window_kv_bytes_per_slot_),
                        cudaMemcpyDeviceToDevice),
             "rewind window-attention state");
}

void DenseFp4Provider::ensure_host_kv_page(
    RequestState& state, std::uint32_t cache_position) {
  if (!host_authoritative_fp16_kv()) return;
  if (cache_position >= max_context_)
    throw std::runtime_error("authoritative FP16 KV position is invalid");
  const auto page_index = cache_position / kv_page_tokens_;
  while (state.host_kv_pages.size() <= page_index) {
    const auto page_bytes = host_fp16_target_page_bytes_;
    if (!page_bytes ||
        page_bytes > std::numeric_limits<std::size_t>::max() ||
        page_bytes > parking_ram_capacity_bytes_ ||
        host_kv_bytes_ + parked_request_bytes_ >
            parking_ram_capacity_bytes_ - page_bytes)
      throw std::runtime_error(
          "authoritative FP16 KV RAM capacity is exhausted");
    void* allocation{};
    cuda_check(cudaHostAlloc(&allocation, static_cast<std::size_t>(page_bytes),
                             cudaHostAllocPortable),
               "allocate progressive authoritative FP16 KV page");
    state.host_kv_pages.push_back(
        {static_cast<std::uint16_t*>(allocation)});
    state.host_kv_bytes += page_bytes;
    host_kv_bytes_ += page_bytes;
  }
}

bool DenseFp4Provider::ensure_target_mirror_page(
    RequestState& state, std::uint32_t cache_position) {
  if (!state.target_mirror_enabled || !device_target_mirror_page_table_)
    return false;
  const auto slot = state.slot();
  if (slot >= capacity_ || cache_position >= max_context_)
    throw std::runtime_error("target mirror page address is invalid");
  const auto page_index = cache_position / kv_page_tokens_;
  auto*& page = slot_target_mirror_pages_.at(slot).at(page_index);
  if (page) return true;
  if (!free_target_mirror_pages_.empty()) {
    page = free_target_mirror_pages_.back();
    free_target_mirror_pages_.pop_back();
  } else if (all_target_mirror_pages_.size() <
             target_mirror_page_capacity_) {
    const auto allocation = cudaMalloc(
        &page, static_cast<std::size_t>(host_fp16_target_page_bytes_));
    if (allocation != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      release_target_mirror_pages(slot);
      state.target_mirror_enabled = false;
      ++target_mirror_spills_;
      return false;
    }
    all_target_mirror_pages_.push_back(page);
  } else {
    release_target_mirror_pages(slot);
    state.target_mirror_enabled = false;
    ++target_mirror_spills_;
    return false;
  }
  cuda_check(cudaMemcpy(
                 device_target_mirror_page_table_ +
                     static_cast<std::size_t>(slot) *
                         maximum_pages_per_slot_ +
                     page_index,
                 &page, sizeof(page), cudaMemcpyHostToDevice),
             "publish exact FP16 target mirror page");
  return true;
}

void DenseFp4Provider::release_target_mirror_pages(
    std::uint32_t slot) noexcept {
  try {
    if (slot >= slot_target_mirror_pages_.size()) return;
    void* empty{};
    for (std::uint32_t page_index = 0U;
         page_index < maximum_pages_per_slot_; ++page_index) {
      auto*& page = slot_target_mirror_pages_[slot][page_index];
      if (!page) continue;
      free_target_mirror_pages_.push_back(page);
      page = nullptr;
      if (device_target_mirror_page_table_)
        static_cast<void>(cudaMemcpy(
            device_target_mirror_page_table_ +
                static_cast<std::size_t>(slot) * maximum_pages_per_slot_ +
                page_index,
            &empty, sizeof(empty), cudaMemcpyHostToDevice));
    }
  } catch (...) {
  }
}

void DenseFp4Provider::trim_target_mirror(
    RequestState& state, std::uint32_t populated_tokens) noexcept {
  try {
    if (!state.target_mirror_enabled ||
        state.slot() >= slot_target_mirror_pages_.size())
      return;
    const auto pages = populated_tokens == 0U
        ? 0U
        : (static_cast<std::uint64_t>(populated_tokens) + kv_page_tokens_ -
           1U) /
              kv_page_tokens_;
    void* empty{};
    auto& mirror = slot_target_mirror_pages_[state.slot()];
    for (std::uint64_t page_index = pages;
         page_index < mirror.size(); ++page_index) {
      auto*& page = mirror[static_cast<std::size_t>(page_index)];
      if (!page) continue;
      free_target_mirror_pages_.push_back(page);
      page = nullptr;
      static_cast<void>(cudaMemcpy(
          device_target_mirror_page_table_ +
              static_cast<std::size_t>(state.slot()) *
                  maximum_pages_per_slot_ +
              page_index,
          &empty, sizeof(empty), cudaMemcpyHostToDevice));
    }
  } catch (...) {
  }
}

void DenseFp4Provider::restore_target_mirror(
    RequestState& state, std::uint32_t slot) noexcept {
  state.target_mirror_enabled = target_mirror_page_capacity_ != 0U;
  if (!state.target_mirror_enabled ||
      state.host_kv_pages.size() > target_mirror_page_capacity_)
    state.target_mirror_enabled = false;
  if (!state.target_mirror_enabled) return;
  try {
    for (std::uint32_t page_index = 0U;
         page_index < state.host_kv_pages.size(); ++page_index) {
      void* page{};
      if (!free_target_mirror_pages_.empty()) {
        page = free_target_mirror_pages_.back();
        free_target_mirror_pages_.pop_back();
      } else if (all_target_mirror_pages_.size() <
                 target_mirror_page_capacity_) {
        cuda_check(cudaMalloc(
                       &page, static_cast<std::size_t>(
                                  host_fp16_target_page_bytes_)),
                   "allocate restored exact FP16 target mirror page");
        all_target_mirror_pages_.push_back(page);
      } else {
        throw std::runtime_error("target mirror capacity is exhausted");
      }
      slot_target_mirror_pages_.at(slot).at(page_index) = page;
      cuda_check(cudaMemcpy(
                     page, state.host_kv_pages[page_index].allocation,
                     static_cast<std::size_t>(
                         host_fp16_target_page_bytes_),
                     cudaMemcpyHostToDevice),
                 "restore exact FP16 target mirror page");
      cuda_check(cudaMemcpy(
                     device_target_mirror_page_table_ +
                         static_cast<std::size_t>(slot) *
                             maximum_pages_per_slot_ +
                         page_index,
                     &page, sizeof(page), cudaMemcpyHostToDevice),
                 "publish restored exact FP16 target mirror page");
      target_mirror_restore_bytes_ += host_fp16_target_page_bytes_;
    }
  } catch (...) {
    release_target_mirror_pages(slot);
    state.target_mirror_enabled = false;
  }
}

void DenseFp4Provider::release_host_kv(RequestState& state) noexcept {
  for (auto& page : state.host_kv_pages) {
    if (!page.allocation) continue;
    static_cast<void>(cudaFreeHost(page.allocation));
    page.allocation = nullptr;
  }
  if (state.host_kv_bytes <= host_kv_bytes_)
    host_kv_bytes_ -= state.host_kv_bytes;
  state.host_kv_pages.clear();
  state.host_kv_bytes = 0U;
  state.host_kv_populated_tokens = 0U;
}

void DenseFp4Provider::trim_host_kv(
    RequestState& state, std::uint32_t populated_tokens) noexcept {
  const auto pages = populated_tokens == 0U
      ? 0U
      : (static_cast<std::uint64_t>(populated_tokens) + kv_page_tokens_ - 1U) /
            kv_page_tokens_;
  const auto page_bytes = host_fp16_target_page_bytes_;
  while (state.host_kv_pages.size() > pages) {
    auto& page = state.host_kv_pages.back();
    if (page.allocation) static_cast<void>(cudaFreeHost(page.allocation));
    state.host_kv_pages.pop_back();
    if (page_bytes <= state.host_kv_bytes)
      state.host_kv_bytes -= page_bytes;
    if (page_bytes <= host_kv_bytes_) host_kv_bytes_ -= page_bytes;
  }
  state.host_kv_populated_tokens = populated_tokens;
}

std::pair<std::uint16_t*, std::uint16_t*>
DenseFp4Provider::host_kv_layer_page(
    const RequestState& state, std::uint32_t full_attention_slot,
    std::uint32_t page_index) const {
  if (full_attention_slot >= target_full_layers_ ||
      page_index >= state.host_kv_pages.size() ||
      !state.host_kv_pages[page_index].allocation)
    throw std::runtime_error("authoritative FP16 KV page is absent");
  const auto page_values = static_cast<std::size_t>(kv_page_tokens_) *
                           kv_heads_ * head_dim_;
  auto* keys = state.host_kv_pages[page_index].allocation +
               static_cast<std::size_t>(full_attention_slot) *
                   2U * page_values;
  return {keys, keys + page_values};
}

void DenseFp4Provider::stage_qsa_selected_host(
    const RequestState& state, std::uint32_t full_attention_slot,
    std::span<const std::uint32_t> selected) {
  if (!qsa_tiered_ || !qsa_host_selected_keys_ ||
      !qsa_host_selected_values_ || selected.empty() ||
      selected.size() > qsa_token_budget_ + qsa_compress_ratio_ - 1U)
    throw std::runtime_error("invalid tiered QSA selection staging");
  cuda_check(cudaStreamSynchronize(nullptr),
             "finish tiered QSA host commits");
  const auto row_values = static_cast<std::size_t>(kv_heads_) * head_dim_;
  std::size_t output_token{};
  while (output_token < selected.size()) {
    const auto first = selected[output_token];
    const auto page_index = first / kv_page_tokens_;
    const auto page_offset = first % kv_page_tokens_;
    std::size_t count = 1U;
    while (output_token + count < selected.size() &&
           selected[output_token + count] == first + count &&
           page_offset + count < kv_page_tokens_)
      ++count;
    const auto [keys, values] =
        host_kv_layer_page(state, full_attention_slot, page_index);
    const auto source_offset = static_cast<std::size_t>(page_offset) *
                               row_values;
    const auto destination_offset = output_token * row_values;
    const auto bytes = count * row_values * sizeof(std::uint16_t);
    std::memcpy(qsa_host_selected_keys_ + destination_offset,
                keys + source_offset, bytes);
    std::memcpy(qsa_host_selected_values_ + destination_offset,
                values + source_offset, bytes);
    output_token += count;
  }
  const auto values = selected.size() * row_values;
  const auto bytes = values * sizeof(std::uint16_t);
  cuda_check(cudaMemcpy(staged_raw_keys_, qsa_host_selected_keys_, bytes,
                        cudaMemcpyHostToDevice),
             "upload tiered QSA selected keys");
  cuda_check(cudaMemcpy(staged_raw_values_, qsa_host_selected_values_, bytes,
                        cudaMemcpyHostToDevice),
             "upload tiered QSA selected values");
  qsa_selected_host_bytes_ += 2U * bytes;
  qsa_selected_host_tokens_ += selected.size();
  ++qsa_selected_host_calls_;
}

void DenseFp4Provider::run_full_attention(
    const PreparedOperation& operation, RequestState& state,
    std::span<const std::uint32_t> cache_positions,
    std::span<const std::uint32_t> rotary_positions, std::uint32_t rows,
    std::uint32_t full_attention_slot,
    std::span<const std::uint32_t> mrope_positions) {
  const auto slot = state.slot();
  if (!rows || rows > workspace_rows_ || cache_positions.size() != rows ||
      rotary_positions.size() != rows ||
      (!mrope_positions.empty() && mrope_positions.size() != 3U * rows))
    throw std::runtime_error("invalid full-attention microbatch");
  for (std::uint32_t row = 1U; row < rows; ++row)
    if (cache_positions[row] != cache_positions.front() + row ||
        (mrope_positions.empty() &&
         rotary_positions[row] != rotary_positions.front() + row))
      throw std::runtime_error(
          "full-attention microbatch positions are not contiguous");
  if (operation.capability ==
      "block.mla.causal.latent-kv.bfloat16.v1") {
    if (!mla_enabled_ || !mrope_positions.empty() ||
        full_attention_slot >= target_full_layers_ ||
        cache_positions.back() >= max_context_)
      throw std::runtime_error("invalid compressed MLA microbatch");
    project(binding(operation, "query_a"), normalized_, mla_query_rank_,
            rows);
    status_check(ec::round_bf16_in_place(
        mla_query_rank_, static_cast<std::uint64_t>(rows) * q_lora_rank_,
        nullptr));
    status_check(ec::rms_norm_bf16_weight_batch(
        mla_query_rank_, binding(operation, "query_a_norm").bf16,
        mla_query_rank_, rows, q_lora_rank_, epsilon_, nullptr));
    status_check(ec::round_bf16_in_place(
        mla_query_rank_, static_cast<std::uint64_t>(rows) * q_lora_rank_,
        nullptr));
    project(binding(operation, "query_b"), mla_query_rank_, query_gate_,
            rows);
    status_check(ec::round_bf16_in_place(
        query_gate_, static_cast<std::uint64_t>(rows) * query_heads_ *
                         (qk_nope_head_dim_ + qk_rope_head_dim_), nullptr));
    project(binding(operation, "kv_a"), normalized_, mla_latent_, rows);
    status_check(ec::round_bf16_in_place(
        mla_latent_, static_cast<std::uint64_t>(rows) *
                         (kv_lora_rank_ + qk_rope_head_dim_), nullptr));
    status_check(ec::rms_norm_bf16_weight_strided_batch(
        mla_latent_, kv_lora_rank_ + qk_rope_head_dim_,
        binding(operation, "kv_a_norm").bf16, mla_latent_,
        kv_lora_rank_ + qk_rope_head_dim_, rows, kv_lora_rank_, epsilon_,
        nullptr));

    const void* const* page_table{};
    const bool authoritative = host_authoritative_fp16_kv();
    if (authoritative) {
      ensure_host_kv_page(state, cache_positions.back());
      if (!state.target_mirror_enabled)
        throw std::runtime_error(
            "compressed MLA requires a device mirror for this context");
      for (const auto position : cache_positions)
        if (!ensure_target_mirror_page(state, position))
          throw std::runtime_error(
              "compressed MLA exhausted its device mirror");
      page_table = reinterpret_cast<const void* const*>(
          device_target_mirror_page_table_ +
          static_cast<std::size_t>(slot) * maximum_pages_per_slot_);
    } else {
      for (const auto position : cache_positions) ensure_page(slot, position);
      page_table = reinterpret_cast<const void* const*>(
          device_page_table_ + static_cast<std::size_t>(slot) *
                                   maximum_pages_per_slot_);
    }
    status_check(ec::mla_causal_latent_bf16({
        query_gate_, mla_latent_, binding(operation, "kv_b").bf16,
        page_table, attention_, mla_latent_query_, partial_maxima_,
        partial_sums_, partial_outputs_, full_attention_slot, kv_page_tokens_,
        cache_positions.front(), rows, query_heads_, kv_lora_rank_,
        qk_nope_head_dim_, qk_rope_head_dim_, v_head_dim_,
        mla_attention_scale_, rope_theta_, rope_factor_, rope_beta_fast_,
        rope_beta_slow_,
        rope_original_context_, llama4_scaling_beta_, nullptr}));
    if (authoritative) {
      const auto half = static_cast<std::uint32_t>(kv_heads_ * head_dim_);
      for (const auto position : cache_positions) {
        const auto page_index = position / kv_page_tokens_;
        const auto page_offset = position % kv_page_tokens_;
        auto* page = static_cast<std::uint16_t*>(
            slot_target_mirror_pages_[slot][page_index]);
        const auto layer_base = static_cast<std::size_t>(
            full_attention_slot) * 2U * kv_page_tokens_ * half;
        const auto [host_first, host_second] = host_kv_layer_page(
            state, full_attention_slot, page_index);
        cuda_check(cudaMemcpyAsync(
            host_first + static_cast<std::size_t>(page_offset) * half,
            page + layer_base + static_cast<std::size_t>(page_offset) * half,
            half * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "commit compressed MLA latent first half");
        cuda_check(cudaMemcpyAsync(
            host_second + static_cast<std::size_t>(page_offset) * half,
            page + layer_base + static_cast<std::size_t>(kv_page_tokens_) *
                       half + static_cast<std::size_t>(page_offset) * half,
            half * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "commit compressed MLA latent second half");
      }
      state.host_kv_populated_tokens = std::max(
          state.host_kv_populated_tokens, cache_positions.front() + rows);
    }
    project(binding(operation, "output"), attention_, residual_, rows);
    status_check(ec::round_bf16_in_place(
        residual_, static_cast<std::uint64_t>(rows) * hidden_size_,
        nullptr));
    return;
  }
  const auto qsa = operation.capability ==
      "block.sparse-attention.qsa.output-gated.v1";
  const auto* projection_input = qsa ? hidden_ : normalized_;
  prepare_dense_activation(projection_input, rows, hidden_size_);
  project_quantized(binding(operation, "query_projection"), projection_input,
                    query_gate_, rows);
  project_quantized(binding(operation, "key_projection"), projection_input,
                    key_, rows);
  project_quantized(binding(operation, "value_projection"), projection_input,
                    value_, rows);
  if (qsa) {
    if ((!device_resident_fp16_kv() && !qsa_tiered_) ||
        !mrope_positions.empty() ||
        operation.qsa_index_slot >= qsa_layers_ ||
        full_attention_slot >= target_full_layers_ ||
        cache_positions.back() >= max_context_)
      throw std::runtime_error("QSA execution geometry is invalid");
    for (const auto position : cache_positions) ensure_page(slot, position);
    const auto* page_table = reinterpret_cast<const void* const*>(
        device_page_table_ + static_cast<std::size_t>(slot) *
                                 maximum_pages_per_slot_);
    status_check(ec::gated_gqa_qkv_rope_fp16_batch(
        query_gate_, key_, value_, binding(operation, "query_norm").f32,
        binding(operation, "key_norm").f32, staged_raw_keys_,
        staged_raw_values_, rotary_positions.front(), rows, query_heads_,
        kv_heads_, head_dim_, rotary_dimension_, epsilon_, rope_theta_,
        nullptr));
    bool staged_complete_layer{};
    if (device_resident_fp16_kv()) {
      status_check(ec::store_gqa_kv_fp16_to_paged(
          staged_raw_keys_, staged_raw_values_, page_table,
          full_attention_slot, kv_page_tokens_, cache_positions.front(), rows,
          kv_heads_, head_dim_, nullptr));
    } else {
      ensure_host_kv_page(state, cache_positions.back());
      const auto row_values = static_cast<std::size_t>(kv_heads_) * head_dim_;
      std::uint32_t copied{};
      while (copied < rows) {
        const auto position = cache_positions.front() + copied;
        const auto page_index = position / kv_page_tokens_;
        const auto page_offset = position % kv_page_tokens_;
        const auto count = std::min(rows - copied,
                                    kv_page_tokens_ - page_offset);
        const auto [host_keys, host_values] =
            host_kv_layer_page(state, full_attention_slot, page_index);
        const auto host_offset = static_cast<std::size_t>(page_offset) *
                                 row_values;
        const auto device_offset = static_cast<std::size_t>(copied) *
                                   row_values;
        const auto copy_bytes = static_cast<std::size_t>(count) * row_values *
                                sizeof(std::uint16_t);
        cuda_check(cudaMemcpyAsync(host_keys + host_offset,
                                   staged_raw_keys_ + device_offset,
                                   copy_bytes, cudaMemcpyDeviceToHost),
                   "commit tiered QSA keys");
        cuda_check(cudaMemcpyAsync(host_values + host_offset,
                                   staged_raw_values_ + device_offset,
                                   copy_bytes, cudaMemcpyDeviceToHost),
                   "commit tiered QSA values");
        qsa_host_commit_bytes_ += 2U * copy_bytes;
        copied += count;
      }
      state.host_kv_populated_tokens = std::max(
          state.host_kv_populated_tokens, cache_positions.front() + rows);
      staged_complete_layer =
          cache_positions.front() == 0U ||
          (staged_device_slot_ == slot &&
           staged_device_layer_ == full_attention_slot &&
           staged_device_context_ == cache_positions.front());
      if (staged_complete_layer) {
        const auto device_offset =
            static_cast<std::size_t>(cache_positions.front()) * row_values;
        const auto copy_values = static_cast<std::size_t>(rows) * row_values;
        cuda_check(cudaMemcpyAsync(staged_device_keys_ + device_offset,
                                   staged_raw_keys_,
                                   copy_values * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice),
                   "extend staged QSA keys");
        cuda_check(cudaMemcpyAsync(staged_device_values_ + device_offset,
                                   staged_raw_values_,
                                   copy_values * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice),
                   "extend staged QSA values");
        staged_device_slot_ = slot;
        staged_device_layer_ = full_attention_slot;
        staged_device_context_ = cache_positions.front() + rows;
      }
    }
    project(binding(operation, "index_projection"), projection_input,
            qsa_projected_, rows);
    status_check(ec::qsa_prepare_index(
        {qsa_projected_, binding(operation, "index_query_norm").f32,
         page_table, qsa_index_page_offset_, operation.qsa_index_slot,
         kv_page_tokens_, cache_positions.front(), rows, qsa_index_heads_,
         qsa_index_head_dim_, rotary_dimension_, epsilon_, rope_theta_,
         nullptr}));
    const auto projected_row = static_cast<std::size_t>(
        (qsa_index_heads_ + 1U) * qsa_index_head_dim_);
    const auto query_row =
        static_cast<std::size_t>(2U) * query_heads_ * head_dim_;
    for (std::uint32_t row = 0U; row < rows; ++row) {
      const auto visible = cache_positions[row] + 1U;
      const auto complete_blocks = visible / qsa_compress_ratio_;
      qsa_host_selected_.clear();
      if (complete_blocks != 0U) {
        status_check(ec::qsa_score_blocks(
            {qsa_projected_ + static_cast<std::size_t>(row) * projected_row,
             page_table, binding(operation, "index_key_norm").f32,
             qsa_scores_, qsa_index_page_offset_, operation.qsa_index_slot,
             kv_page_tokens_, visible, qsa_index_heads_, qsa_index_head_dim_,
             rotary_dimension_, qsa_compress_ratio_, epsilon_, rope_theta_,
             nullptr}));
        cuda_check(cudaMemcpy(qsa_host_scores_.data(), qsa_scores_,
                              static_cast<std::size_t>(complete_blocks) *
                                  sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy QSA block scores");
        qsa_score_device_to_host_bytes_ +=
            static_cast<std::uint64_t>(complete_blocks) * sizeof(float);
        std::vector<std::uint32_t> blocks(complete_blocks);
        std::iota(blocks.begin(), blocks.end(), 0U);
        const auto selected_blocks = std::min(
            complete_blocks, qsa_token_budget_ / qsa_compress_ratio_);
        std::partial_sort(
            blocks.begin(), blocks.begin() + selected_blocks, blocks.end(),
            [&](std::uint32_t left, std::uint32_t right) {
              if (qsa_host_scores_[left] == qsa_host_scores_[right])
                return left < right;
              return qsa_host_scores_[left] > qsa_host_scores_[right];
            });
        blocks.resize(selected_blocks);
        for (const auto block : blocks)
          for (std::uint32_t item = 0U; item < qsa_compress_ratio_; ++item)
            qsa_host_selected_.push_back(
                block * qsa_compress_ratio_ + item);
      }
      for (std::uint32_t token = complete_blocks * qsa_compress_ratio_;
           token < visible; ++token)
        qsa_host_selected_.push_back(token);
      std::sort(qsa_host_selected_.begin(), qsa_host_selected_.end());
      if (qsa_host_selected_.empty() ||
          qsa_host_selected_.size() >
              qsa_token_budget_ + qsa_compress_ratio_ - 1U)
        throw std::runtime_error("QSA selection geometry is invalid");
      cuda_check(cudaMemcpy(qsa_selected_, qsa_host_selected_.data(),
                            qsa_host_selected_.size() *
                                sizeof(qsa_host_selected_[0]),
                            cudaMemcpyHostToDevice),
                 "upload QSA selected token indices");
      auto* row_output = attention_ + static_cast<std::size_t>(row) *
          query_heads_ * head_dim_;
      const auto* row_query = query_gate_ +
          static_cast<std::size_t>(row) * query_row;
      if (device_resident_fp16_kv()) {
        status_check(ec::qsa_selected_attention(
            {row_query, page_table, qsa_selected_, row_output,
             static_cast<std::uint32_t>(qsa_host_selected_.size()),
             full_attention_slot, kv_page_tokens_, query_heads_, kv_heads_,
             head_dim_, nullptr}));
      } else if (staged_complete_layer) {
        status_check(ec::qsa_selected_contiguous_attention(
            {row_query, staged_device_keys_, staged_device_values_,
             qsa_selected_, row_output,
             static_cast<std::uint32_t>(qsa_host_selected_.size()),
             query_heads_, kv_heads_, head_dim_, nullptr}));
        ++qsa_device_staged_calls_;
      } else {
        stage_qsa_selected_host(state, full_attention_slot,
                                qsa_host_selected_);
        status_check(ec::qsa_selected_contiguous_attention(
            {row_query, staged_raw_keys_, staged_raw_values_, nullptr,
             row_output,
             static_cast<std::uint32_t>(qsa_host_selected_.size()),
             query_heads_, kv_heads_, head_dim_, nullptr}));
      }
    }
    project(binding(operation, "output_projection"), attention_, residual_,
            rows);
    return;
  }
  if (operation.abi_version >= 2U) {
    if (!device_resident_fp16_kv() || !mrope_positions.empty() ||
        operation.kv_layer_slot == kNoSlot ||
        cache_positions.back() >= max_context_)
      throw std::runtime_error(
          "normalized output-gated GQA requires device FP16 KV");
    project_quantized(binding(operation, "gate_projection"), projection_input,
                      gate_, rows);
    const auto windowed = operation.attention_window_tokens != 0U;
    auto** selected_table = windowed ? device_window_page_table_
                                     : device_page_table_;
    if (!windowed)
      for (const auto position : cache_positions) ensure_page(slot, position);
    const auto* page_table = reinterpret_cast<const void* const*>(
        selected_table + static_cast<std::size_t>(slot) *
                             maximum_pages_per_slot_);
    const auto rotary_enabled =
        parameter_u32(operation.parameters, "rotary_enabled") != 0U;
    status_check(ec::normalized_gqa_qkv_fp16_batch(
        query_gate_, key_, value_, staged_raw_keys_, staged_raw_values_,
        rotary_positions.front(), rows, query_heads_, kv_heads_, head_dim_,
        rotary_dimension_,
        parameter_f32(operation.parameters, "qk_norm_epsilon_f32_bits"),
        parameter_f32(operation.parameters, "query_scale_f32_bits"),
        parameter_f32_or(operation.parameters, "rope_theta_f32_bits", 1.0F),
        rotary_enabled, nullptr));

    const auto query_values = static_cast<std::size_t>(workspace_rows_) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = staged_score_capacity_values_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_,
        staged_probability_capacity_values_ * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(workspace_rows_) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(workspace_rows_) *
                           query_heads_ * sizeof(float),
        staged_split_tokens_};
    const auto q_row_values =
        static_cast<std::size_t>(query_heads_) * head_dim_;
    const auto kv_row_values =
        static_cast<std::size_t>(kv_heads_) * head_dim_;
    const auto execute_rows = [&](std::uint32_t first_row,
                                  std::uint32_t count) {
      status_check(ec::store_gqa_kv_fp16_to_paged(
          staged_raw_keys_ + static_cast<std::size_t>(first_row) *
                                 kv_row_values,
          staged_raw_values_ + static_cast<std::size_t>(first_row) *
                                   kv_row_values,
          page_table, operation.kv_layer_slot, kv_page_tokens_,
          cache_positions[first_row], count, kv_heads_, head_dim_, nullptr));
      ec::DeviceFp16GatedGqaAttentionLaunch launch{};
      launch.q_and_gate = query_gate_ +
          static_cast<std::size_t>(first_row) * q_row_values;
      launch.output = attention_ +
          static_cast<std::size_t>(first_row) * q_row_values;
      launch.cache_capacity = max_context_;
      launch.first_context_tokens = cache_positions[first_row] + 1U;
      launch.retained_first_token = windowed &&
              launch.first_context_tokens > operation.attention_window_tokens
          ? launch.first_context_tokens - operation.attention_window_tokens
          : 0U;
      launch.rows = count;
      launch.query_heads = query_heads_;
      launch.kv_heads = kv_heads_;
      launch.head_dim = head_dim_;
      launch.device_pages = page_table;
      launch.device_page_layer = operation.kv_layer_slot;
      launch.device_page_tokens = kv_page_tokens_;
      launch.device_page_count = maximum_pages_per_slot_;
      launch.output_gated = false;
      status_check(
          ec::gated_gqa_attention_staged_device_fp16(launch, workspace));
    };
    if (windowed) {
      // Publishing a future row can overwrite a cyclic entry still required
      // by an earlier row in the same microbatch. Execute windowed attention
      // row-by-row after the shared projection to preserve exact causality.
      for (std::uint32_t row = 0U; row < rows; ++row) execute_rows(row, 1U);
    } else {
      execute_rows(0U, rows);
    }
    status_check(ec::sigmoid_product_in_place(
        attention_, gate_, rows * query_heads_ * head_dim_, nullptr,
        activation_bf16_));
    project(binding(operation, "output_projection"), attention_, residual_,
            rows);
    return;
  }
  if (operation.capability == "block.full-attention.standard-gqa.v1" ||
      operation.capability ==
          "block.full-attention.standard-gqa-no-position.v1") {
    if (!mrope_positions.empty() || !device_resident_fp16_kv() ||
        full_attention_slot >= target_full_layers_ ||
        cache_positions.back() >= max_context_)
      throw std::runtime_error(
          "standard GQA requires device-resident exact FP16 KV");
    for (const auto position : cache_positions) ensure_page(slot, position);
    const auto* page_table = reinterpret_cast<const void* const*>(
        device_page_table_ + static_cast<std::size_t>(slot) *
                                 maximum_pages_per_slot_);
    if (operation.capability == "block.full-attention.standard-gqa.v1")
      status_check(ec::standard_gqa_qkv_rope_fp16_batch(
          query_gate_, key_, value_, staged_raw_keys_, staged_raw_values_,
          rotary_positions.front(), rows, query_heads_, kv_heads_, head_dim_,
          rotary_dimension_, rope_theta_, nullptr));
    else
      status_check(ec::standard_gqa_kv_fp16_batch(
          key_, value_, staged_raw_keys_, staged_raw_values_, rows,
          query_heads_, kv_heads_, head_dim_, nullptr));
    status_check(ec::store_gqa_kv_fp16_to_paged(
        staged_raw_keys_, staged_raw_values_, page_table, full_attention_slot,
        kv_page_tokens_, cache_positions.front(), rows, kv_heads_, head_dim_,
        nullptr));

    const auto query_values = static_cast<std::size_t>(workspace_rows_) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = staged_score_capacity_values_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_,
        staged_probability_capacity_values_ * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(workspace_rows_) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(workspace_rows_) *
                           query_heads_ * sizeof(float),
        staged_split_tokens_};
    ec::DeviceFp16GatedGqaAttentionLaunch launch{};
    launch.q_and_gate = query_gate_;
    launch.output = attention_;
    launch.cache_capacity = max_context_;
    launch.first_context_tokens = cache_positions.front() + 1U;
    launch.rows = rows;
    launch.query_heads = query_heads_;
    launch.kv_heads = kv_heads_;
    launch.head_dim = head_dim_;
    launch.device_pages = page_table;
    launch.device_page_layer = full_attention_slot;
    launch.device_page_tokens = kv_page_tokens_;
    launch.device_page_count =
        static_cast<std::uint32_t>(slot_pages_[slot].size());
    launch.output_gated = false;
    status_check(
        ec::gated_gqa_attention_staged_device_fp16(launch, workspace));
    project(binding(operation, "output_projection"), attention_, residual_,
            rows);
    return;
  }
  if (!mrope_positions.empty()) {
    if (!vision_positions_)
      throw std::runtime_error("mRoPE workspace is absent");
    cuda_check(cudaMemcpy(vision_positions_, mrope_positions.data(),
                          mrope_positions.size_bytes(),
                          cudaMemcpyHostToDevice),
               "upload three-axis rotary positions");
    status_check(ec::gated_gqa_qk_norm_mrope_batch(
        query_gate_, key_, binding(operation, "query_norm").f32,
        binding(operation, "key_norm").f32, vision_positions_, rows,
        query_heads_, kv_heads_, head_dim_, rotary_dimension_,
        mrope_sections_[0], mrope_sections_[1], mrope_sections_[2], epsilon_,
        rope_theta_, nullptr, activation_bf16_));
  }
  const auto authoritative = host_authoritative_fp16_kv() &&
                             mtp_attention_.get() != &operation;
  if (authoritative) {
    if (full_attention_slot >= target_full_layers_ ||
        cache_positions.back() >= max_context_)
      throw std::runtime_error("invalid authoritative FP16 attention state");
    ensure_host_kv_page(state, cache_positions.back());
    if (mrope_positions.empty())
      status_check(ec::gated_gqa_qkv_rope_fp16_batch(
          query_gate_, key_, value_, binding(operation, "query_norm").f32,
          binding(operation, "key_norm").f32, staged_raw_keys_,
          staged_raw_values_, rotary_positions.front(), rows, query_heads_,
          kv_heads_, head_dim_, rotary_dimension_, epsilon_, rope_theta_,
          nullptr));
    else
      status_check(ec::store_gqa_kv_fp16_batch(
          key_, value_, staged_raw_keys_, staged_raw_values_, rows, kv_heads_,
          head_dim_, nullptr));
    const auto row_values = static_cast<std::size_t>(kv_heads_) * head_dim_;
    std::uint32_t copied{};
    while (copied < rows) {
      const auto position = cache_positions.front() + copied;
      const auto page_index = position / kv_page_tokens_;
      const auto page_offset = position % kv_page_tokens_;
      const auto count = std::min(rows - copied,
                                  kv_page_tokens_ - page_offset);
      const auto [host_keys, host_values] =
          host_kv_layer_page(state, full_attention_slot, page_index);
      const auto host_offset = static_cast<std::size_t>(page_offset) *
                               row_values;
      const auto device_offset = static_cast<std::size_t>(copied) *
                                 row_values;
      const auto copy_bytes = static_cast<std::size_t>(count) * row_values *
                              sizeof(std::uint16_t);
      cuda_check(cudaMemcpyAsync(host_keys + host_offset,
                                 staged_raw_keys_ + device_offset, copy_bytes,
                                 cudaMemcpyDeviceToHost),
                 "commit progressive authoritative FP16 keys");
      cuda_check(cudaMemcpyAsync(host_values + host_offset,
                                 staged_raw_values_ + device_offset,
                                 copy_bytes, cudaMemcpyDeviceToHost),
                 "commit progressive authoritative FP16 values");
      copied += count;
    }
    state.host_kv_populated_tokens = std::max(
        state.host_kv_populated_tokens, cache_positions.front() + rows);
    bool mirrored = state.target_mirror_enabled;
    if (mirrored) {
      for (const auto position : cache_positions) {
        if (!ensure_target_mirror_page(state, position)) {
          mirrored = false;
          break;
        }
      }
    }
    const auto continues_device_layer = !mirrored &&
        (cache_positions.front() == 0U ||
         (staged_device_slot_ == slot &&
          staged_device_layer_ == full_attention_slot &&
          staged_device_context_ == cache_positions.front()));
    if (mirrored) {
      const auto* mirror_page_table = reinterpret_cast<const void* const*>(
          device_target_mirror_page_table_ +
          static_cast<std::size_t>(slot) * maximum_pages_per_slot_);
      status_check(ec::store_gqa_kv_fp16_to_paged(
          staged_raw_keys_, staged_raw_values_, mirror_page_table,
          full_attention_slot, kv_page_tokens_, cache_positions.front(), rows,
          kv_heads_, head_dim_, nullptr));
    }
    if (continues_device_layer) {
      const auto device_offset =
          static_cast<std::size_t>(cache_positions.front()) * row_values;
      const auto copy_values = static_cast<std::size_t>(rows) * row_values;
      cuda_check(cudaMemcpyAsync(staged_device_keys_ + device_offset,
                                 staged_raw_keys_,
                                 copy_values * sizeof(std::uint16_t),
                                 cudaMemcpyDeviceToDevice),
                 "extend device FP16 prefill keys");
      cuda_check(cudaMemcpyAsync(staged_device_values_ + device_offset,
                                 staged_raw_values_,
                                 copy_values * sizeof(std::uint16_t),
                                 cudaMemcpyDeviceToDevice),
                 "extend device FP16 prefill values");
      staged_device_slot_ = slot;
      staged_device_layer_ = full_attention_slot;
      staged_device_context_ = cache_positions.front() + rows;
    }
    const auto first_context_tokens = cache_positions.front() + 1U;
    const auto query_values = static_cast<std::size_t>(workspace_rows_) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = staged_score_capacity_values_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_,
        staged_probability_capacity_values_ * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(workspace_rows_) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(workspace_rows_) *
                           query_heads_ * sizeof(float),
        staged_split_tokens_};
    if (mirrored) {
      const auto* mirror_page_table = reinterpret_cast<const void* const*>(
          device_target_mirror_page_table_ +
          static_cast<std::size_t>(slot) * maximum_pages_per_slot_);
      status_check(ec::gated_gqa_attention_staged_device_fp16(
          {query_gate_, nullptr, nullptr, attention_, max_context_,
           first_context_tokens, 0U, rows, query_heads_, kv_heads_, head_dim_,
           nullptr, mirror_page_table, full_attention_slot, kv_page_tokens_,
           maximum_pages_per_slot_},
          workspace));
      ++target_mirror_attention_calls_;
    } else if (continues_device_layer) {
      status_check(ec::gated_gqa_attention_staged_device_fp16(
          {query_gate_, staged_device_keys_, staged_device_values_, attention_,
           max_context_, first_context_tokens, 0U, rows,
           query_heads_, kv_heads_, head_dim_, nullptr},
          workspace));
    } else {
      std::vector<const void*> key_pages;
      std::vector<const void*> value_pages;
      key_pages.reserve(state.host_kv_pages.size());
      value_pages.reserve(state.host_kv_pages.size());
      for (std::uint32_t page_index = 0U;
           page_index < state.host_kv_pages.size(); ++page_index) {
        const auto [keys, values] =
            host_kv_layer_page(state, full_attention_slot, page_index);
        key_pages.push_back(keys);
        value_pages.push_back(values);
      }
      status_check(ec::gated_gqa_attention_staged_host_fp16(
          {query_gate_, nullptr, nullptr, attention_,
           static_cast<std::uint32_t>(state.host_kv_pages.size()) *
               kv_page_tokens_,
           first_context_tokens,
           0U, rows, query_heads_, kv_heads_, head_dim_,
           nullptr, key_pages.data(), value_pages.data(), kv_page_tokens_,
           static_cast<std::uint32_t>(key_pages.size())},
          workspace));
      ++target_host_attention_calls_;
    }
    project(binding(operation, "output_projection"), attention_, residual_,
            rows);
    return;
  }
  if (device_resident_fp16_kv() && mtp_attention_.get() != &operation) {
    if (full_attention_slot >= target_full_layers_ ||
        cache_positions.back() >= max_context_)
      throw std::runtime_error("invalid device-resident FP16 attention state");
    for (const auto position : cache_positions) ensure_page(slot, position);
    const auto* page_table = reinterpret_cast<const void* const*>(
        device_page_table_ + static_cast<std::size_t>(slot) *
                                 maximum_pages_per_slot_);
    if (mrope_positions.empty())
      status_check(ec::gated_gqa_qkv_rope_fp16_batch(
          query_gate_, key_, value_, binding(operation, "query_norm").f32,
          binding(operation, "key_norm").f32, staged_raw_keys_,
          staged_raw_values_, rotary_positions.front(), rows, query_heads_,
          kv_heads_, head_dim_, rotary_dimension_, epsilon_, rope_theta_,
          nullptr));
    else
      status_check(ec::store_gqa_kv_fp16_batch(
          key_, value_, staged_raw_keys_, staged_raw_values_, rows, kv_heads_,
          head_dim_, nullptr));
    status_check(ec::store_gqa_kv_fp16_to_paged(
        staged_raw_keys_, staged_raw_values_, page_table,
        full_attention_slot, kv_page_tokens_, cache_positions.front(), rows,
        kv_heads_, head_dim_, nullptr));

    const auto query_values = static_cast<std::size_t>(workspace_rows_) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = staged_score_capacity_values_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_,
        staged_probability_capacity_values_ * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(workspace_rows_) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(workspace_rows_) *
                           query_heads_ * sizeof(float),
        staged_split_tokens_};
    status_check(ec::gated_gqa_attention_staged_device_fp16(
        {query_gate_, nullptr, nullptr, attention_, max_context_,
         cache_positions.front() + 1U, 0U, rows, query_heads_, kv_heads_,
         head_dim_, nullptr, page_table, full_attention_slot,
         kv_page_tokens_, static_cast<std::uint32_t>(
                              slot_pages_[slot].size())},
        workspace));
    project(binding(operation, "output_projection"), attention_, residual_,
            rows);
    return;
  }
  if (full_attention_slot >=
      (host_authoritative_fp16_kv() ? mtp_layers_
                              : target_full_layers_ + mtp_layers_))
    throw std::runtime_error("invalid paged full-attention slot");
  const auto target_fp8 =
      target_kv_encoding_ == TargetKvEncoding::fp8_e4m3_per_head;
  const auto target_fp4_key_outlier1 =
      target_kv_encoding_ == TargetKvEncoding::fp4_key_outlier1;
  const auto target_q4_bfp_outlier1 =
      target_kv_encoding_ == TargetKvEncoding::q4_bfp_key_outlier1;
  const auto target_q4_bfp =
      target_kv_encoding_ == TargetKvEncoding::q4_bfp;
  const auto target_q4_per_head =
      target_kv_encoding_ == TargetKvEncoding::q4_per_head;
  const auto target_q5_q4_bfp =
      target_kv_encoding_ == TargetKvEncoding::q5_q4_bfp;
  const auto mtp_q8 = mtp_q8_kv_ && mtp_attention_.get() == &operation;
  const auto mixed_mtp =
      (target_fp8 || target_fp4_key_outlier1 || target_q4_bfp_outlier1 ||
       target_q4_bfp || target_q4_per_head ||
       target_q5_q4_bfp || mtp_q8) &&
      full_attention_slot >= target_full_layers_;
  const auto physical_layer =
      mixed_mtp ? full_attention_slot - target_full_layers_
                : full_attention_slot;
  auto** selected_page_table = mixed_mtp ? device_mtp_page_table_
                                         : device_page_table_;
  const auto* page_table = reinterpret_cast<const void* const*>(
      selected_page_table + static_cast<std::size_t>(slot) *
                                maximum_pages_per_slot_);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    ensure_page(slot, cache_positions[row]);
  }
  if (!mrope_positions.empty()) {
    if (mtp_q8)
      status_check(ec::store_gqa_kv_paged_q8_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
    else if (target_fp8 && !mixed_mtp)
      status_check(ec::store_gqa_kv_paged_fp8_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
    else if (target_fp4_key_outlier1 && !mixed_mtp)
      status_check(ec::store_gqa_kv_paged_fp4_key_outlier1_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
    else if (target_q4_bfp_outlier1 && !mixed_mtp)
      status_check(ec::store_gqa_kv_paged_q4_bfp_key_outlier1_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
    else if (target_q4_bfp && !mixed_mtp)
      status_check(ec::store_gqa_kv_paged_q4_bfp_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
    else if (target_q4_per_head && !mixed_mtp)
      status_check(ec::store_gqa_kv_paged_q4_per_head_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
    else if (target_q5_q4_bfp && !mixed_mtp)
      status_check(ec::store_gqa_kv_paged_q5_q4_bfp_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
    else
      status_check(ec::store_gqa_kv_paged_fp4_batch(
          key_, value_, page_table, physical_layer, kv_page_tokens_,
          cache_positions.front(), rows, kv_heads_, head_dim_, nullptr));
  }
  if (rows == 1U) {
    auto* page =
        slot_pages_[slot][cache_positions.front() / kv_page_tokens_];
    if (mixed_mtp)
      page = static_cast<void*>(static_cast<std::byte*>(page) +
                                mtp_kv_page_offset_);
    if (mtp_q8) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q8_at(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            query_heads_, kv_heads_, head_dim_, rotary_dimension_, epsilon_,
            rope_theta_, nullptr, activation_bf16_));
      status_check(ec::gated_gqa_attention_decode_paged_q8_tensor_core({
          query_gate_, page_table, attention_, partial_maxima_, partial_sums_,
          partial_outputs_, cache_positions.front() + 1U, physical_layer,
          kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
          kAttentionSplitTokens, attention_maximum_splits_, nullptr,
          activation_bf16_}));
    } else if (target_fp8 && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_fp8_at(
          query_gate_, key_, value_,
          binding(operation, "query_norm").f32,
          binding(operation, "key_norm").f32, page, physical_layer,
          kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
          query_heads_, kv_heads_, head_dim_, rotary_dimension_, epsilon_,
          rope_theta_, nullptr));
      status_check(ec::gated_gqa_attention_decode_paged_fp8_tensor_core({
          query_gate_, page_table, attention_, partial_maxima_, partial_sums_,
          partial_outputs_, cache_positions.front() + 1U, physical_layer,
          kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
          kAttentionSplitTokens, attention_maximum_splits_, nullptr,
          activation_bf16_}));
    } else if (target_fp4_key_outlier1 && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(
            ec::gated_gqa_qkv_rope_cache_paged_fp4_key_outlier1_at(
                query_gate_, key_, value_,
                binding(operation, "query_norm").f32,
                binding(operation, "key_norm").f32, page, physical_layer,
                kv_page_tokens_, cache_positions.front(),
                rotary_positions.front(), query_heads_, kv_heads_, head_dim_,
                rotary_dimension_, epsilon_, rope_theta_, nullptr));
      status_check(ec::
                       gated_gqa_attention_decode_paged_fp4_key_outlier1_tensor_core(
                           {query_gate_, page_table, attention_,
                            partial_maxima_, partial_sums_, partial_outputs_,
                            cache_positions.front() + 1U, physical_layer,
                            kv_page_tokens_, query_heads_, kv_heads_,
                            head_dim_, kAttentionSplitTokens,
                            attention_maximum_splits_, nullptr,
                            activation_bf16_}));
    } else if (target_q4_bfp_outlier1 && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(
            ec::gated_gqa_qkv_rope_cache_paged_q4_bfp_key_outlier1_at(
                query_gate_, key_, value_,
                binding(operation, "query_norm").f32,
                binding(operation, "key_norm").f32, page, physical_layer,
                kv_page_tokens_, cache_positions.front(),
                rotary_positions.front(), query_heads_, kv_heads_, head_dim_,
                rotary_dimension_, epsilon_, rope_theta_, nullptr));
      status_check(ec::quantize_gqa_queries_q8(
          query_gate_, attention_q8_queries_, attention_query_scales_, 1U,
          query_heads_, head_dim_, nullptr));
      const auto context_tokens = cache_positions.front() + 1U;
      status_check(ec::
          gated_gqa_attention_microbatch_paged_q4_bfp_key_outlier1_tensor_core(
              {query_gate_, attention_q8_queries_, attention_query_scales_,
               page_table, attention_, partial_maxima_, partial_sums_,
               partial_outputs_, context_tokens, 1U, physical_layer,
               kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
               packed_attention_split_tokens(context_tokens),
               attention_maximum_splits_, nullptr, activation_bf16_}));
    } else if (target_q4_bfp && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q4_bfp_at(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            query_heads_, kv_heads_, head_dim_, rotary_dimension_, epsilon_,
            rope_theta_, nullptr));
      status_check(ec::quantize_gqa_queries_q8(
          query_gate_, attention_q8_queries_, attention_query_scales_, 1U,
          query_heads_, head_dim_, nullptr));
      const auto context_tokens = cache_positions.front() + 1U;
      status_check(ec::gated_gqa_attention_microbatch_paged_q4_bfp_tensor_core(
          {query_gate_, attention_q8_queries_, attention_query_scales_,
           page_table, attention_, partial_maxima_, partial_sums_,
           partial_outputs_, context_tokens, 1U, physical_layer,
           kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
           packed_attention_split_tokens(context_tokens),
           attention_maximum_splits_, nullptr, activation_bf16_}));
    } else if (target_q4_per_head && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q4_per_head_at(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            query_heads_, kv_heads_, head_dim_, rotary_dimension_, epsilon_,
            rope_theta_, nullptr, activation_bf16_));
      status_check(ec::quantize_gqa_queries_q8(
          query_gate_, attention_q8_queries_, attention_query_scales_, 1U,
          query_heads_, head_dim_, nullptr));
      const auto context_tokens = cache_positions.front() + 1U;
      status_check(
          ec::gated_gqa_attention_microbatch_paged_q4_per_head_tensor_core(
              {query_gate_, attention_q8_queries_, attention_query_scales_,
               page_table, attention_, partial_maxima_, partial_sums_,
               partial_outputs_, context_tokens, 1U, physical_layer,
               kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
               packed_attention_split_tokens(context_tokens),
               attention_maximum_splits_, nullptr, activation_bf16_}));
    } else if (target_q5_q4_bfp && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q5_q4_bfp_at(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            query_heads_, kv_heads_, head_dim_, rotary_dimension_, epsilon_,
            rope_theta_, nullptr));
      status_check(ec::quantize_gqa_queries_q8(
          query_gate_, attention_q8_queries_, attention_query_scales_, 1U,
          query_heads_, head_dim_, nullptr));
      const auto context_tokens = cache_positions.front() + 1U;
      status_check(ec::gated_gqa_attention_microbatch_paged_q5_q4_bfp_tensor_core(
          {query_gate_, attention_q8_queries_, attention_query_scales_,
           page_table, attention_, partial_maxima_, partial_sums_,
           partial_outputs_, context_tokens, 1U, physical_layer,
           kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
           packed_attention_split_tokens(context_tokens),
           attention_maximum_splits_, nullptr, activation_bf16_}));
    } else {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_fp4_at(
          query_gate_, key_, value_,
          binding(operation, "query_norm").f32,
          binding(operation, "key_norm").f32, page, physical_layer,
          kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
          query_heads_, kv_heads_, head_dim_, rotary_dimension_, epsilon_,
          rope_theta_, nullptr));
      status_check(ec::gated_gqa_attention_decode_paged_fp4_tensor_core({
          query_gate_, page_table, attention_, partial_maxima_, partial_sums_,
          partial_outputs_, cache_positions.front() + 1U, physical_layer,
          kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
          kAttentionSplitTokens, attention_maximum_splits_, nullptr,
          activation_bf16_}));
    }
  } else {
    if (mtp_q8) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q8_batch(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page_table, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            rows, query_heads_, kv_heads_, head_dim_, rotary_dimension_,
            epsilon_, rope_theta_, nullptr, activation_bf16_));
    } else if (target_fp8 && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_fp8_batch(
          query_gate_, key_, value_, binding(operation, "query_norm").f32,
          binding(operation, "key_norm").f32, page_table, physical_layer,
          kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
          rows, query_heads_, kv_heads_, head_dim_, rotary_dimension_,
          epsilon_, rope_theta_, nullptr));
    } else if (target_fp4_key_outlier1 && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(
            ec::gated_gqa_qkv_rope_cache_paged_fp4_key_outlier1_batch(
                query_gate_, key_, value_,
                binding(operation, "query_norm").f32,
                binding(operation, "key_norm").f32, page_table,
                physical_layer, kv_page_tokens_, cache_positions.front(),
                rotary_positions.front(), rows, query_heads_, kv_heads_,
                head_dim_, rotary_dimension_, epsilon_, rope_theta_,
                nullptr));
    } else if (target_q4_bfp_outlier1 && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(
            ec::gated_gqa_qkv_rope_cache_paged_q4_bfp_key_outlier1_batch(
                query_gate_, key_, value_,
                binding(operation, "query_norm").f32,
                binding(operation, "key_norm").f32, page_table,
                physical_layer, kv_page_tokens_, cache_positions.front(),
                rotary_positions.front(), rows, query_heads_, kv_heads_,
                head_dim_, rotary_dimension_, epsilon_, rope_theta_,
                nullptr));
    } else if (target_q4_bfp && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q4_bfp_batch(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page_table, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            rows, query_heads_, kv_heads_, head_dim_, rotary_dimension_,
            epsilon_, rope_theta_, nullptr));
    } else if (target_q4_per_head && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q4_per_head_batch(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page_table, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            rows, query_heads_, kv_heads_, head_dim_, rotary_dimension_,
            epsilon_, rope_theta_, nullptr, activation_bf16_));
    } else if (target_q5_q4_bfp && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_q5_q4_bfp_batch(
            query_gate_, key_, value_,
            binding(operation, "query_norm").f32,
            binding(operation, "key_norm").f32, page_table, physical_layer,
            kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
            rows, query_heads_, kv_heads_, head_dim_, rotary_dimension_,
            epsilon_, rope_theta_, nullptr));
    } else {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_fp4_batch(
          query_gate_, key_, value_, binding(operation, "query_norm").f32,
          binding(operation, "key_norm").f32, page_table, physical_layer,
          kv_page_tokens_, cache_positions.front(), rotary_positions.front(),
          rows, query_heads_, kv_heads_, head_dim_, rotary_dimension_,
          epsilon_, rope_theta_, nullptr));
    }
    if (rows * (query_heads_ / kv_heads_) <= 32U) {
      ++fused_multiquery_attention_calls_;
      const ec::PagedFp4GatedGqaPrefillLaunch launch{
          query_gate_, page_table, attention_, partial_maxima_, partial_sums_,
          partial_outputs_, cache_positions.front() + 1U, rows,
          physical_layer, kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
          kAttentionSplitTokens, attention_maximum_splits_, nullptr,
          activation_bf16_};
      if (mtp_q8)
        status_check(
            ec::gated_gqa_attention_microbatch_paged_q8_tensor_core(launch));
      else if (target_fp8 && !mixed_mtp)
        status_check(
            ec::gated_gqa_attention_microbatch_paged_fp8_tensor_core(launch));
      else if (target_fp4_key_outlier1 && !mixed_mtp)
        status_check(ec::
                         gated_gqa_attention_microbatch_paged_fp4_key_outlier1_tensor_core(
                             launch));
      else if (target_q4_bfp_outlier1 && !mixed_mtp) {
        status_check(ec::quantize_gqa_queries_q8(
            query_gate_, attention_q8_queries_, attention_query_scales_,
            rows, query_heads_, head_dim_, nullptr));
        const auto context_tokens = cache_positions.front() + 1U;
        status_check(ec::
            gated_gqa_attention_microbatch_paged_q4_bfp_key_outlier1_tensor_core(
                {query_gate_, attention_q8_queries_,
                 attention_query_scales_, page_table, attention_,
                 partial_maxima_, partial_sums_, partial_outputs_,
                 context_tokens, rows, physical_layer, kv_page_tokens_,
                 query_heads_, kv_heads_, head_dim_,
                 packed_attention_split_tokens(context_tokens),
                 attention_maximum_splits_, nullptr, activation_bf16_}));
      }
      else if (target_q4_bfp && !mixed_mtp) {
        status_check(ec::quantize_gqa_queries_q8(
            query_gate_, attention_q8_queries_, attention_query_scales_,
            rows, query_heads_, head_dim_, nullptr));
        const auto context_tokens = cache_positions.front() + 1U;
        status_check(ec::gated_gqa_attention_microbatch_paged_q4_bfp_tensor_core(
            {query_gate_, attention_q8_queries_, attention_query_scales_,
             page_table, attention_, partial_maxima_, partial_sums_,
             partial_outputs_, context_tokens, rows, physical_layer,
             kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
             packed_attention_split_tokens(context_tokens),
             attention_maximum_splits_, nullptr, activation_bf16_}));
      }
      else if (target_q4_per_head && !mixed_mtp) {
        status_check(ec::quantize_gqa_queries_q8(
            query_gate_, attention_q8_queries_, attention_query_scales_,
            rows, query_heads_, head_dim_, nullptr));
        const auto context_tokens = cache_positions.front() + 1U;
        status_check(
            ec::gated_gqa_attention_microbatch_paged_q4_per_head_tensor_core(
                {query_gate_, attention_q8_queries_, attention_query_scales_,
                 page_table, attention_, partial_maxima_, partial_sums_,
                 partial_outputs_, context_tokens, rows, physical_layer,
                 kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
                 packed_attention_split_tokens(context_tokens),
                 attention_maximum_splits_, nullptr, activation_bf16_}));
      }
      else if (target_q5_q4_bfp && !mixed_mtp) {
        status_check(ec::quantize_gqa_queries_q8(
            query_gate_, attention_q8_queries_, attention_query_scales_,
            rows, query_heads_, head_dim_, nullptr));
        const auto context_tokens = cache_positions.front() + 1U;
        status_check(ec::gated_gqa_attention_microbatch_paged_q5_q4_bfp_tensor_core(
            {query_gate_, attention_q8_queries_, attention_query_scales_,
             page_table, attention_, partial_maxima_, partial_sums_,
             partial_outputs_, context_tokens, rows, physical_layer,
             kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
             packed_attention_split_tokens(context_tokens),
             attention_maximum_splits_, nullptr, activation_bf16_}));
      }
      else
        status_check(
            ec::gated_gqa_attention_microbatch_paged_fp4_tensor_core(launch));
    } else {
      const ec::PagedFp4GatedGqaPrefillLaunch launch{
          query_gate_, page_table, attention_, partial_maxima_, partial_sums_,
          partial_outputs_, cache_positions.front() + 1U, rows,
          physical_layer, kv_page_tokens_, query_heads_, kv_heads_,
          head_dim_, kAttentionSplitTokens,
          attention_maximum_splits_, nullptr, activation_bf16_};
      const auto query_values =
          static_cast<std::size_t>(workspace_rows_) * query_heads_ * head_dim_;
      const auto kv_values = static_cast<std::size_t>(kv_heads_) *
                             staged_split_tokens_ * head_dim_;
      const auto score_values = staged_score_capacity_values_;
      const ec::PagedFp4GatedGqaStagedPrefillWorkspace workspace{
          staged_queries_, query_values * sizeof(std::uint16_t),
          staged_keys_, kv_values * sizeof(std::uint16_t),
          staged_values_, kv_values * sizeof(std::uint16_t),
          staged_scores_, score_values * sizeof(float),
          staged_probabilities_,
          staged_probability_capacity_values_ * sizeof(std::uint16_t),
          staged_accumulator_, query_values * sizeof(float), partial_maxima_,
          static_cast<std::size_t>(workspace_rows_) * query_heads_ *
              sizeof(float),
          partial_sums_,
          static_cast<std::size_t>(workspace_rows_) * query_heads_ *
              sizeof(float),
          staged_split_tokens_};
      if (mtp_q8)
        status_check(ec::gated_gqa_attention_staged_prefill_paged_q8(
            launch, workspace));
      else if (target_fp8 && !mixed_mtp)
        status_check(ec::gated_gqa_attention_staged_prefill_paged_fp8(
            launch, workspace));
      else if (target_fp4_key_outlier1 && !mixed_mtp)
        status_check(
            ec::gated_gqa_attention_staged_prefill_paged_fp4_key_outlier1(
                launch, workspace));
      else if (target_q4_bfp_outlier1 && !mixed_mtp)
        status_check(
            ec::gated_gqa_attention_staged_prefill_paged_q4_bfp_key_outlier1(
                launch, workspace));
      else if (target_q4_bfp && !mixed_mtp)
        status_check(ec::gated_gqa_attention_staged_prefill_paged_q4_bfp(
            launch, workspace));
      else if (target_q4_per_head && !mixed_mtp)
        status_check(
            ec::gated_gqa_attention_staged_prefill_paged_q4_per_head(
                launch, workspace));
      else if (target_q5_q4_bfp && !mixed_mtp)
        status_check(ec::gated_gqa_attention_staged_prefill_paged_q5_q4_bfp(
            launch, workspace));
      else
        status_check(ec::gated_gqa_attention_staged_prefill_paged_fp4(
            launch, workspace));
    }
  }
  project(binding(operation, "output_projection"), attention_, residual_,
          rows);
}

void DenseFp4Provider::run_recurrent_attention(
    const PreparedOperation& operation, std::uint32_t slot,
    std::uint32_t rows, RecurrentCheckpointMode checkpoint_mode) {
  if (operation.capability == "block.mamba2.ssm.v1") {
    if (checkpoint_mode != RecurrentCheckpointMode::none)
      throw std::runtime_error(
          "Mamba2 does not advertise transactional draft checkpoints");
    prepare_dense_activation(normalized_, rows, hidden_size_);
    project_quantized(binding(operation, "input_projection"), normalized_,
                      projected_qkv_, rows);
    status_check(ec::mamba2_forward({
        projected_qkv_, binding(operation, "convolution").dequantized,
        binding(operation, "convolution_bias").f32,
        binding(operation, "time_bias").f32,
        binding(operation, "decay_log").f32,
        binding(operation, "skip").f32,
        binding(operation, "output_norm").f32,
        recurrent_conv(operation.recurrent_slot, slot),
        recurrent_matrix(operation.recurrent_slot, slot), conv_output_,
        delta_output_, rows, mamba_heads_, mamba_head_dim_, mamba_state_size_,
        mamba_state_groups_, mamba_conv_kernel_, epsilon_,
        parameter_f32(operation.parameters, "time_step_min_f32_bits"),
        nullptr}));
    project(binding(operation, "output_projection"), delta_output_, residual_,
            rows);
    return;
  }
  const auto key_dimension = key_heads_ * key_head_dim_;
  const auto value_dimension = value_heads_ * value_head_dim_;
  const auto conv_dimension = 2U * key_dimension + value_dimension;
  const auto split_no_residual = operation.capability ==
      "block.recurrent-linear-attention.split-gated-delta.no-residual.v1";
  const auto* projection_input = split_no_residual ? hidden_ : normalized_;
  prepare_dense_activation(projection_input, rows, hidden_size_);
  project_quantized(binding(operation, "qkv_projection"), projection_input,
                    projected_qkv_, rows);
  project_quantized(binding(operation, "z_projection"), projection_input,
                    projected_z_, rows);
  project_quantized(binding(operation, "b_projection"), projection_input,
                    projected_b_, rows);
  project_quantized(binding(operation, "a_projection"), projection_input,
                    projected_a_, rows);
  const auto conv_state_values =
      static_cast<std::size_t>(conv_dimension) * conv_kernel_;
  const auto matrix_state_values = static_cast<std::size_t>(value_heads_) *
                                   key_head_dim_ * value_head_dim_;
  const auto output_gate_activation = operation.abi_version >= 2U
      ? static_cast<ec::GatedDeltaOutputActivation>(
            parameter_u32(operation.parameters, "output_gate_activation"))
      : ec::GatedDeltaOutputActivation::silu;
  if (rows > 1U &&
      checkpoint_mode != RecurrentCheckpointMode::after_first) {
    const auto checkpoint_every_row =
        checkpoint_mode == RecurrentCheckpointMode::every_row;
    auto* conv_checkpoints = checkpoint_every_row
        ? recurrent_conv_speculative_checkpoint_.at(
              operation.recurrent_slot) +
              static_cast<std::size_t>(slot) * kMaximumExactDecodeRows *
                  conv_state_values
        : nullptr;
    auto* matrix_checkpoints = checkpoint_every_row
        ? recurrent_matrix_speculative_checkpoint_.at(
              operation.recurrent_slot) +
              static_cast<std::size_t>(slot) * kMaximumExactDecodeRows *
                  matrix_state_values
        : nullptr;
    status_check(ec::split_gated_delta_prefill({
        projected_qkv_, projected_z_, projected_b_, projected_a_,
        binding(operation, "convolution").dequantized,
        binding(operation, "time_bias").f32,
        binding(operation, "decay_log").f32,
        binding(operation, "output_norm").f32,
        recurrent_conv(operation.recurrent_slot, slot),
        recurrent_matrix(operation.recurrent_slot, slot), conv_output_,
        delta_output_, gate_,
        static_cast<std::size_t>(workspace_rows_) *
            std::max({intermediate_size_, shared_intermediate_size_,
                      expert_width_, mamba_heads_ * mamba_head_dim_,
                      hyper_lowrank_, ple_embedding_width_}) *
            sizeof(float),
        conv_checkpoints, matrix_checkpoints,
        rows, key_heads_, value_heads_, key_head_dim_, value_head_dim_,
        conv_kernel_, epsilon_, output_gate_activation, activation_bf16_,
        nullptr}));
    project(binding(operation, "output_projection"), delta_output_, residual_,
            rows);
    return;
  }
  for (std::uint32_t row = 0U; row < rows; ++row) {
    status_check(ec::split_gated_delta_decode({
        projected_qkv_ + static_cast<std::size_t>(row) * conv_dimension,
        projected_z_ + static_cast<std::size_t>(row) * value_dimension,
        projected_b_ + static_cast<std::size_t>(row) * value_heads_,
        projected_a_ + static_cast<std::size_t>(row) * value_heads_,
        binding(operation, "convolution").dequantized,
        binding(operation, "time_bias").f32,
        binding(operation, "decay_log").f32,
        binding(operation, "output_norm").f32,
        recurrent_conv(operation.recurrent_slot, slot),
        recurrent_matrix(operation.recurrent_slot, slot),
        conv_output_ + static_cast<std::size_t>(row) * conv_dimension,
        delta_output_ + static_cast<std::size_t>(row) * value_dimension,
        key_heads_, value_heads_, key_head_dim_, value_head_dim_, conv_kernel_,
        epsilon_, output_gate_activation, activation_bf16_, nullptr}));
    if (checkpoint_mode == RecurrentCheckpointMode::after_first &&
        row == 0U) {
      auto* conv_checkpoint =
          recurrent_conv_checkpoint_.at(operation.recurrent_slot) +
          static_cast<std::size_t>(slot) * conv_state_values;
      auto* matrix_checkpoint =
          recurrent_matrix_checkpoint_.at(operation.recurrent_slot) +
          static_cast<std::size_t>(slot) * matrix_state_values;
      cuda_check(cudaMemcpy(conv_checkpoint,
                            recurrent_conv(operation.recurrent_slot, slot),
                            conv_state_values * sizeof(float),
                            cudaMemcpyDeviceToDevice),
                 "checkpoint recurrent convolution state");
      cuda_check(cudaMemcpy(matrix_checkpoint,
                            recurrent_matrix(operation.recurrent_slot, slot),
                            matrix_state_values * sizeof(float),
                            cudaMemcpyDeviceToDevice),
                 "checkpoint recurrent matrix state");
    }
  }
  project(binding(operation, "output_projection"), delta_output_, residual_,
          rows);
}

void DenseFp4Provider::run_router(const PreparedOperation& operation,
                                  std::uint32_t rows) {
  if (!routed_experts_ || !rows || rows > workspace_rows_)
    throw std::runtime_error("invalid routed FP4 router batch");
  if (operation.capability ==
      "router.softmax-topk.shared-swiglu.nvfp4-block16.v1") {
    const auto& norm = binding(operation, "input_norm");
    status_check(ec::rms_norm_bf16_weight_batch(
        hidden_, norm.bf16, normalized_, rows, hidden_size_, epsilon_,
        nullptr));
    status_check(ec::round_bf16_in_place(
        normalized_, static_cast<std::uint64_t>(rows) * hidden_size_,
        nullptr));
    project(binding(operation, "shared_gate_projection"), normalized_, gate_,
            rows);
    project(binding(operation, "shared_up_projection"), normalized_, up_,
            rows);
    status_check(ec::silu_product(
        gate_, up_, intermediate_, rows * shared_intermediate_size_,
        nullptr, activation_bf16_));
    status_check(ec::round_bf16_in_place(
        intermediate_, static_cast<std::uint64_t>(rows) *
                           shared_intermediate_size_, nullptr));
    project(binding(operation, "shared_down_projection"), intermediate_,
            shared_output_, rows);
    project(binding(operation, "router_weight"), normalized_, router_logits_,
            rows);
    status_check(ec::round_bf16_in_place(
        router_logits_, static_cast<std::uint64_t>(rows) * expert_count_,
        nullptr));
    status_check(ec::router_topk_normalized_logits_batch(
        router_logits_, rows, expert_count_, route_width_,
        parameter_f32(operation.parameters, "route_scale_f32_bits"),
        routing_scores_, routing_indices_, nullptr));
    status_check(ec::round_bf16_in_place(
        routing_scores_, static_cast<std::uint64_t>(rows) * route_width_,
        nullptr));
    return;
  }
  if (operation.capability ==
      "router.linear-topk.shared-swiglu.no-residual.v1")
    cuda_check(cudaMemcpy(normalized_, hidden_,
                          static_cast<std::size_t>(rows) * hidden_size_ *
                              sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "retain unnormalized routed input");
  else
    normalize_rows(hidden_, binding(operation, "input_norm").f32,
                   normalized_, rows);
  prepare_dense_activation(normalized_, rows, hidden_size_);
  if (operation.capability ==
      "router.sigmoid-bias.topk.shared-relu2.v1") {
    project_quantized(binding(operation, "shared_up_projection"), normalized_,
                      up_, rows);
    status_check(ec::relu2_in_place(
        up_, rows * shared_intermediate_size_, nullptr));
    project(binding(operation, "shared_down_projection"), up_, shared_output_,
            rows);
    status_check(ec::sigmoid_bias_router_topk_batch(
        normalized_, binding(operation, "router_weight").f32,
        binding(operation, "correction_bias").f32, rows, hidden_size_,
        expert_count_, route_width_, 1.0e-20F,
        parameter_f32(operation.parameters, "scale_f32_bits"), router_logits_,
        routing_scores_, routing_indices_, nullptr));
    return;
  }
  project_quantized(binding(operation, "shared_gate_projection"), normalized_,
                    gate_, rows);
  project_quantized(binding(operation, "shared_up_projection"), normalized_,
                    up_, rows);
  status_check(ec::silu_product(
      gate_, up_, intermediate_, rows * shared_intermediate_size_, nullptr,
      activation_bf16_));
  project(binding(operation, "shared_down_projection"), intermediate_,
          shared_output_, rows);
  status_check(ec::gemv_f32_batch(
      binding(operation, "shared_router").f32, 1U, hidden_size_, normalized_,
      shared_scalar_, rows, nullptr));
  for (std::uint32_t row = 0U; row < rows; ++row)
    status_check(ec::sigmoid_scale_in_place(
        shared_output_ + static_cast<std::size_t>(row) * hidden_size_,
        shared_scalar_ + row, hidden_size_, nullptr));
  status_check(ec::router_topk_normalized_batch(
      normalized_, binding(operation, "router_weight").f32, rows,
      hidden_size_, expert_count_, route_width_, router_logits_,
      routing_scores_, routing_indices_, nullptr));
}

void DenseFp4Provider::run_routed_moe(const PreparedOperation& operation,
                                      std::uint32_t rows) {
  if (!routed_experts_ || !rows || rows > workspace_rows_)
    throw std::runtime_error("invalid routed FP4 MoE batch");
  routed_experts_->execute(
      operation.component_layer, normalized_, routing_scores_,
      routing_indices_, rows, moe_intermediate_, moe_selection_output_, q8_,
      q8_scales_, moe_q8_intermediate_, moe_q8_intermediate_scales_,
      moe_nvfp4_gate_input_, moe_nvfp4_up_input_, moe_nvfp4_down_input_,
      moe_output_);
  status_check(ec::add_in_place(moe_output_, shared_output_,
                                rows * hidden_size_, nullptr,
                                activation_bf16_));
  const auto native_nvfp4 = operation.capability ==
      "moe.swiglu.routed.nvfp4-block16.merge-shared.v1";
  if (native_nvfp4)
    status_check(ec::round_bf16_in_place(
        moe_output_, static_cast<std::uint64_t>(rows) * hidden_size_,
        nullptr));
  if (operation.capability ==
      "moe.swiglu.routed.merge-shared.no-residual.v1")
    cuda_check(cudaMemcpy(hidden_, moe_output_,
                          static_cast<std::size_t>(rows) * hidden_size_ *
                              sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "commit no-residual routed output");
  else
    status_check(ec::add_in_place(hidden_, moe_output_, rows * hidden_size_,
                                  nullptr, activation_bf16_));
  if (native_nvfp4)
    status_check(ec::round_bf16_in_place(
        hidden_, static_cast<std::uint64_t>(rows) * hidden_size_, nullptr));
}

void DenseFp4Provider::run_ffn(const PreparedOperation& operation,
                               std::uint32_t rows) {
  cuda_check(cudaMemcpy(residual_, hidden_,
                        static_cast<std::size_t>(rows) * hidden_size_ *
                            sizeof(float),
                        cudaMemcpyDeviceToDevice),
             "retain FFN residual");
  normalize_operation_input(operation, hidden_, normalized_, rows);
  prepare_dense_activation(normalized_, rows, hidden_size_);
  project_quantized(binding(operation, "gate_projection"), normalized_, gate_,
                    rows);
  project_quantized(binding(operation, "up_projection"), normalized_, up_,
                    rows);
  status_check(ec::silu_product(gate_, up_, intermediate_,
                                rows * intermediate_size_, nullptr,
                                activation_bf16_));
  project(binding(operation, "down_projection"), intermediate_, hidden_, rows);
  if (operation.abi_version < 2U) {
    status_check(ec::add_in_place(hidden_, residual_, rows * hidden_size_,
                                  nullptr, activation_bf16_));
  } else {
    const auto mode = parameter_u32(operation.parameters, "post_norm_mode");
    const auto epsilon = parameter_f32(
        operation.parameters, "post_norm_epsilon_f32_bits");
    if (mode == 0U)
      status_check(ec::rms_norm_batch(
          hidden_, binding(operation, "post_norm").f32, normalized_, rows,
          hidden_size_, epsilon, nullptr));
    else if (mode == 1U)
      status_check(ec::zero_centered_rms_norm_batch(
          hidden_, binding(operation, "post_norm").f32, normalized_, rows,
          hidden_size_, epsilon, nullptr));
    else
      throw std::runtime_error("unsupported FFN post norm mode");
    status_check(ec::add_in_place(normalized_, residual_,
                                  rows * hidden_size_, nullptr,
                                  activation_bf16_));
    cuda_check(cudaMemcpy(hidden_, normalized_,
                          static_cast<std::size_t>(rows) * hidden_size_ *
                              sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "commit post-normalized FFN update");
  }
}

void DenseFp4Provider::trace_sampling(
    const er::ProgramRequestContext& request, std::uint32_t position,
    std::uint32_t token, std::string_view mode,
    const er::SamplingDistribution* target,
    const er::SamplingDistribution* draft) {
  if (!sampling_trace_stream_.is_open()) return;
  if (target != nullptr && !(target->probability(token) > 0.0))
    throw std::runtime_error(
        "diagnostic token is outside its target sampling support");
  auto& output = sampling_trace_stream_;
  output << "{\"schema\":\"sampling-decision-v1\",\"request_id\":"
         << request.request_id << ",\"position\":" << position
         << ",\"token\":" << token << ",\"mode\":\"" << mode << '"';
  const auto write_distribution = [&](std::string_view name,
                                      const er::SamplingDistribution* value) {
    output << ",\"" << name << "\":";
    if (value == nullptr) {
      output << "null";
      return;
    }
    output << '[';
    for (std::size_t index = 0U; index < value->entries.size(); ++index) {
      if (index != 0U) output << ',';
      const auto& entry = value->entries[index];
      output << '[' << entry.token << ',' << entry.probability << ']';
    }
    output << ']';
  };
  write_distribution("target", target);
  write_distribution("draft", draft);
  output << "}\n";
  if (!output)
    throw std::runtime_error("cannot write diagnostic sampling trace");
  if ((++sampling_trace_records_ & 63U) == 0U) output.flush();
}

er::SamplingDistribution DenseFp4Provider::distribution_from_logits(
    float* row_logits, std::uint32_t vocabulary,
    const er::ProgramRequestContext& request, const std::uint8_t* presence) {
  if (!row_logits || !vocabulary || vocabulary > vocabulary_size_)
    throw std::runtime_error("sampling logit row is invalid");
  const auto biased_presence = request_parameter(
      request, "sampling_presence_penalty_biased_ppm");
  if (biased_presence > 4'000'000U)
    throw std::runtime_error("sampling presence penalty is invalid");
  const auto presence_penalty =
      (static_cast<double>(biased_presence) - 2'000'000.0) / 1'000'000.0;
  if (presence_penalty != 0.0) {
    if (!presence)
      throw std::runtime_error("sampling presence state is absent");
    status_check(ec::apply_presence_penalty(
        row_logits, presence, vocabulary,
        static_cast<float>(presence_penalty), nullptr));
  }
  const auto temperature = request_parameter(
      request, "sampling_temperature_ppm");
  if (temperature == 0U) {
    status_check(ec::argmax(row_logits, vocabulary, output_tokens_, nullptr));
    std::uint32_t token{};
    cuda_check(cudaMemcpy(&token, output_tokens_, sizeof(token),
                          cudaMemcpyDeviceToHost),
               "copy exact greedy token");
    return {{{token, 1.0}}};
  }
  const auto top_k = request_parameter(request, "sampling_top_k");
  if (top_k == 0U || top_k > kMaximumGpuSamplingTopK || top_k > vocabulary)
    throw std::runtime_error(
        "sampled exact decode requires bounded nonzero top-k");
  status_check(ec::topk_logits(
      row_logits, vocabulary, static_cast<std::uint32_t>(top_k),
      sampling_top_logits_, sampling_top_tokens_,
      {sampling_top_partial_logits_,
       sampling_top_partial_items_ * sizeof(float),
       sampling_top_partial_tokens_,
       sampling_top_partial_items_ * sizeof(std::uint32_t)},
      nullptr));
  std::vector<float> host_logits(static_cast<std::size_t>(top_k));
  std::vector<std::uint32_t> host_tokens(static_cast<std::size_t>(top_k));
  cuda_check(cudaMemcpy(host_logits.data(), sampling_top_logits_,
                        host_logits.size() * sizeof(host_logits[0]),
                        cudaMemcpyDeviceToHost),
             "copy speculative top-k logits");
  cuda_check(cudaMemcpy(host_tokens.data(), sampling_top_tokens_,
                        host_tokens.size() * sizeof(host_tokens[0]),
                        cudaMemcpyDeviceToHost),
             "copy speculative top-k tokens");
  ++sampling_gpu_calls_;
  sampling_logit_transfer_bytes_ +=
      host_logits.size() * sizeof(host_logits[0]) +
      host_tokens.size() * sizeof(host_tokens[0]);
  return er::make_sampling_distribution(
      host_logits, host_tokens, static_cast<std::uint32_t>(temperature),
      static_cast<std::uint32_t>(
          request_parameter(request, "sampling_top_p_ppm")),
      static_cast<std::uint32_t>(
          request_parameter(request, "sampling_min_p_ppm")));
}

std::vector<std::uint32_t> DenseFp4Provider::run_head(
    const PreparedOperation& operation, std::uint32_t rows,
    const er::ProgramRequestContext* request, RequestState* state,
    std::uint32_t sample_position, bool terminal_only) {
  const auto head_rows = terminal_only ? 1U : rows;
  ensure_logits_capacity(head_rows);
  const auto* head_input =
      terminal_only
          ? hidden_ + static_cast<std::size_t>(rows - 1U) * hidden_size_
          : hidden_;
  const auto no_norm =
      operation.capability ==
          "head.token-select.fp4-block32.no-norm.v1" ||
      operation.capability == "head.token-select.no-norm.v1";
  if (!no_norm) {
    const auto& norm = binding(operation, "norm");
    if (norm.encoding == "BF16") {
      status_check(ec::rms_norm_bf16_weight_batch(
          head_input, norm.bf16, normalized_, head_rows, hidden_size_,
          epsilon_, nullptr));
      status_check(ec::round_bf16_in_place(
          normalized_, static_cast<std::uint64_t>(head_rows) * hidden_size_,
          nullptr));
    } else {
      normalize_rows(head_input, norm.f32, normalized_, head_rows);
    }
  }
  project(binding(operation, "weight"), no_norm ? head_input : normalized_,
          logits_, head_rows);
  if (operation.capability ==
      "head.rmsnorm.token-select.bfloat16.v1")
    status_check(ec::round_bf16_in_place(
        logits_, static_cast<std::uint64_t>(head_rows) * vocabulary_size_,
        nullptr));
  if (operation.abi_version >= 2U)
    status_check(ec::scaled_tanh_in_place(
        logits_, head_rows * vocabulary_size_,
        parameter_f32(operation.parameters, "logit_multiplier_f32_bits"),
        parameter_f32(operation.parameters, "logit_softcap_f32_bits"),
        nullptr));
  if (request != nullptr) {
    if (head_rows != 1U)
      throw std::runtime_error(
          "sampling requires a single terminal head row");
    if (state == nullptr || state->slot() >= capacity_)
      throw std::runtime_error("sampling request state is absent");
    const auto first_output = request_parameter(
        *request, "sampling_first_output_position");
    const auto output_sampling = sample_position >= first_output;
    const auto biased_presence = request_parameter(
        *request, "sampling_presence_penalty_biased_ppm");
    if (biased_presence > 4'000'000U)
      throw std::runtime_error("sampling presence penalty is invalid");
    const auto presence =
        (static_cast<double>(biased_presence) - 2'000'000.0) / 1'000'000.0;
    auto* emitted = sampling_presence_ +
        static_cast<std::size_t>(state->slot()) * vocabulary_size_;
    if (output_sampling && presence != 0.0)
      status_check(ec::apply_presence_penalty(
          logits_, emitted, vocabulary_size_, static_cast<float>(presence),
          nullptr));
    const auto temperature = request_parameter(
        *request, "sampling_temperature_ppm");
    const auto top_k = request_parameter(*request, "sampling_top_k");
    std::uint32_t selected{};
    std::optional<er::SamplingDistribution> trace_distribution;
    if (temperature == 0U) {
      status_check(ec::argmax(logits_, vocabulary_size_, output_tokens_,
                              nullptr));
      cuda_check(cudaMemcpy(&selected, output_tokens_, sizeof(selected),
                            cudaMemcpyDeviceToHost),
                 "copy dense FP4 greedy token");
      if (sampling_trace_stream_.is_open())
        trace_distribution = er::SamplingDistribution{{{selected, 1.0}}};
    } else if (top_k != 0U && top_k <= kMaximumGpuSamplingTopK) {
      status_check(ec::topk_logits(
          logits_, vocabulary_size_, static_cast<std::uint32_t>(top_k),
          sampling_top_logits_, sampling_top_tokens_,
          {sampling_top_partial_logits_,
           sampling_top_partial_items_ * sizeof(float),
           sampling_top_partial_tokens_,
           sampling_top_partial_items_ * sizeof(std::uint32_t)},
          nullptr));
      std::vector<float> host_logits(static_cast<std::size_t>(top_k));
      std::vector<std::uint32_t> host_tokens(
          static_cast<std::size_t>(top_k));
      cuda_check(cudaMemcpy(host_logits.data(), sampling_top_logits_,
                            host_logits.size() * sizeof(host_logits[0]),
                            cudaMemcpyDeviceToHost),
                 "copy dense FP4 top-k sampling logits");
      cuda_check(cudaMemcpy(host_tokens.data(), sampling_top_tokens_,
                            host_tokens.size() * sizeof(host_tokens[0]),
                            cudaMemcpyDeviceToHost),
                 "copy dense FP4 top-k sampling tokens");
      ++sampling_gpu_calls_;
      sampling_logit_transfer_bytes_ +=
          host_logits.size() * sizeof(host_logits[0]) +
          host_tokens.size() * sizeof(host_tokens[0]);
      if (sampling_trace_stream_.is_open())
        trace_distribution = er::make_sampling_distribution(
            host_logits, host_tokens, static_cast<std::uint32_t>(temperature),
            static_cast<std::uint32_t>(request_parameter(*request,
                                                         "sampling_top_p_ppm")),
            static_cast<std::uint32_t>(request_parameter(*request,
                                                         "sampling_min_p_ppm")));
      selected = sample_sorted_candidates(host_logits, host_tokens, *request,
                                           sample_position);
    } else {
      std::vector<float> host_logits(vocabulary_size_);
      cuda_check(cudaMemcpy(host_logits.data(), logits_,
                            host_logits.size() * sizeof(host_logits[0]),
                            cudaMemcpyDeviceToHost),
                 "copy dense FP4 sampling logits");
      ++sampling_host_calls_;
      sampling_logit_transfer_bytes_ +=
          host_logits.size() * sizeof(host_logits[0]);
      selected = sample_token(host_logits, *request, sample_position);
    }
    if (output_sampling)
      trace_sampling(*request, sample_position, selected, "head",
                     trace_distribution ? &*trace_distribution : nullptr);
    if (output_sampling)
      cuda_check(cudaMemset(emitted + selected, 1, 1),
                 "mark emitted sampling token");
    return {selected};
  }
  status_check(ec::argmax_batch(logits_, vocabulary_size_, head_rows,
                                output_tokens_, nullptr));
  std::vector<std::uint32_t> result(head_rows);
  cuda_check(cudaMemcpy(result.data(), output_tokens_,
                        head_rows * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost),
             "copy dense FP4 output tokens");
  return result;
}

std::vector<std::uint32_t> DenseFp4Provider::run_target(
    RequestState& state, std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> positions,
    RecurrentCheckpointMode checkpoint_mode) {
  if (tokens.empty() || tokens.size() > workspace_rows_ ||
      tokens.size() != positions.size())
    throw std::runtime_error("invalid target-model microbatch");
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  const auto slot = state.slot();
  ++exact_target_batches_;
  if (checkpoint_mode == RecurrentCheckpointMode::every_row) {
    speculative_recurrent_checkpoint_bytes_ +=
        static_cast<std::uint64_t>(rows) * recurrent_layers_ *
        (recurrent_conv_values_ + recurrent_matrix_values_) * sizeof(float);
  }
  std::vector<std::uint32_t> result;
  try {
    for (const auto& operation_pointer : prepared_target_) {
      const auto& operation = *operation_pointer;
      const auto phase_event = begin_gpu_phase(gpu_phase(operation.kernel));
      const auto stage_candidate =
          rows > 8U && operation.kernel != Kernel::embedding &&
          operation.kernel != Kernel::vision &&
          operation.kernel != Kernel::head &&
          operation.kernel != Kernel::routed_moe;
      const auto staged =
          stage_candidate && activate_staged_weights(operation);
      switch (operation.kernel) {
      case Kernel::hyper_initialize:
      case Kernel::ple:
      case Kernel::hyper_read:
      case Kernel::hyper_inject:
      case Kernel::hyper_reduce:
        throw std::runtime_error(
            "Hyper/PLE program has no exact-decode contract");
      case Kernel::embedding:
        for (std::uint32_t row = 0U; row < rows; ++row) {
          if (tokens[row] >= vocabulary_size_)
            throw std::runtime_error("target token exceeds vocabulary");
        }
        run_embedding(operation, tokens.data(), rows);
        break;
      case Kernel::vision:
        break;
      case Kernel::full_attention: {
        cuda_check(cudaMemcpy(residual_, hidden_,
                              static_cast<std::size_t>(rows) * hidden_size_ *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "retain target attention residual");
        normalize_operation_input(operation, hidden_, normalized_, rows);
        std::array<std::uint32_t, kMaximumWorkspaceRows> rotary{};
        for (std::uint32_t row = 0U; row < rows; ++row) {
          const auto adjusted = static_cast<std::int64_t>(positions[row]) +
                                slot_rope_deltas_.at(slot);
          if (adjusted < 0 ||
              adjusted > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("adjusted rotary position is invalid");
          rotary[row] = static_cast<std::uint32_t>(adjusted);
        }
        run_full_attention(operation, state, positions,
                           std::span(rotary).first(rows), rows,
                           operation.full_attention_slot);
        finish_attention_block(operation, rows);
        break;
      }
      case Kernel::recurrent_attention:
        cuda_check(cudaMemcpy(residual_, hidden_,
                              static_cast<std::size_t>(rows) * hidden_size_ *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "retain target recurrent residual");
        normalize_rows(hidden_, binding(operation, "input_norm").f32,
                       normalized_, rows);
        run_recurrent_attention(operation, slot, rows, checkpoint_mode);
        status_check(ec::add_in_place(hidden_, residual_, rows * hidden_size_,
                                      nullptr, activation_bf16_));
        break;
      case Kernel::router:
        run_router(operation, rows);
        break;
      case Kernel::routed_moe:
        run_routed_moe(operation, rows);
        break;
      case Kernel::ffn:
        run_ffn(operation, rows);
        break;
      case Kernel::head:
        result = run_head(operation, rows, nullptr, nullptr,
                          positions.back());
        break;
      case Kernel::exact_decode:
        throw std::runtime_error("exact service appeared in scalar program");
      }
      if (staged) deactivate_staged_weights();
      end_gpu_phase(phase_event);
      if (operation.kernel == Kernel::head) collect_gpu_phases();
    }
  } catch (...) {
    release_staged_dense_weights();
    throw;
  }
  release_staged_dense_weights();
  if (result.size() != rows)
    throw std::runtime_error("target program produced no token head");
  return result;
}

void DenseFp4Provider::restore_recurrent_checkpoint(std::uint32_t slot) {
  const auto conv_values = recurrent_conv_values_;
  const auto matrix_values = recurrent_matrix_values_;
  for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
    cuda_check(cudaMemcpy(
                   recurrent_conv(layer, slot),
                   recurrent_conv_checkpoint_[layer] +
                       static_cast<std::size_t>(slot) * conv_values,
                   conv_values * sizeof(float), cudaMemcpyDeviceToDevice),
               "restore recurrent convolution checkpoint");
    cuda_check(cudaMemcpy(
                   recurrent_matrix(layer, slot),
                   recurrent_matrix_checkpoint_[layer] +
                       static_cast<std::size_t>(slot) * matrix_values,
                   matrix_values * sizeof(float), cudaMemcpyDeviceToDevice),
               "restore recurrent matrix checkpoint");
  }
}

void DenseFp4Provider::restore_speculative_recurrent_checkpoint(
    std::uint32_t slot, std::uint32_t row) {
  if (slot >= capacity_ || row >= kMaximumExactDecodeRows)
    throw std::runtime_error("invalid speculative recurrent checkpoint");
  ++speculative_recurrent_restores_;
  const auto conv_values = recurrent_conv_values_;
  const auto matrix_values = recurrent_matrix_values_;
  for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
    cuda_check(cudaMemcpy(
                   recurrent_conv(layer, slot),
                   recurrent_conv_speculative_checkpoint_[layer] +
                       (static_cast<std::size_t>(slot) *
                            kMaximumExactDecodeRows +
                        row) * conv_values,
                   conv_values * sizeof(float), cudaMemcpyDeviceToDevice),
               "restore speculative recurrent convolution state");
    status_check(ec::restore_split_gated_delta_recurrent_checkpoint(
        recurrent_matrix_speculative_checkpoint_[layer] +
            (static_cast<std::size_t>(slot) * kMaximumExactDecodeRows + row) *
                matrix_values,
        recurrent_matrix(layer, slot), value_heads_, key_head_dim_,
        value_head_dim_, nullptr));
  }
}

std::optional<MtpPrediction> DenseFp4Provider::run_mtp(
    RequestState& state, std::span<const std::uint32_t> tokens,
    const float* previous_hidden,
    std::span<const std::uint32_t> rotary_positions, bool produce_logits,
    const er::ProgramRequestContext* request, const std::uint8_t* presence,
    std::uint32_t sample_position, std::uint32_t random_stream) {
  if (!exact_ || !mtp_attention_ || !mtp_ffn_ || tokens.empty() ||
      tokens.size() > workspace_rows_ || tokens.size() != rotary_positions.size() ||
      state.mtp_length + tokens.size() > max_context_)
    throw std::runtime_error("invalid MTP microbatch");
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  for (std::uint32_t row = 0U; row < rows; ++row) {
    if (tokens[row] >= vocabulary_size_ ||
        rotary_positions[row] != state.mtp_length + row + 1U)
      throw std::runtime_error("MTP token/position stream is not contiguous");
  }
  const auto phase_event = begin_gpu_phase(GpuPhase::mtp);
  cuda_check(cudaMemcpy(output_tokens_, tokens.data(),
                        rows * sizeof(tokens[0]), cudaMemcpyHostToDevice),
             "upload MTP token batch");
  const auto& mtp_token_embedding = binding(*exact_, "token_embedding");
  if (mtp_token_embedding.encoding == "MXFP6_E3M2")
    status_check(ec::mxfp6_embedding_batch(
        mtp_token_embedding.mxfp6_matrix(), output_tokens_, mtp_embedding_,
        rows, nullptr));
  else
    status_check(ec::fp4_embedding_batch(
        mtp_token_embedding.matrix(), output_tokens_, mtp_embedding_, rows,
        nullptr));
  normalize_rows(mtp_embedding_, binding(*exact_, "embedding_norm").f32,
                 mtp_embedding_norm_, rows);
  normalize_rows(previous_hidden, binding(*exact_, "hidden_norm").f32,
                 mtp_hidden_norm_, rows);
  const auto hidden_bytes =
      static_cast<std::size_t>(hidden_size_) * sizeof(float);
  const auto fusion_pitch = 2U * hidden_bytes;
  cuda_check(cudaMemcpy2D(mtp_fusion_input_, fusion_pitch,
                          mtp_embedding_norm_, hidden_bytes, hidden_bytes,
                          rows, cudaMemcpyDeviceToDevice),
             "assemble MTP normalized embedding batch");
  cuda_check(cudaMemcpy2D(mtp_fusion_input_ + hidden_size_, fusion_pitch,
                          mtp_hidden_norm_, hidden_bytes, hidden_bytes, rows,
                          cudaMemcpyDeviceToDevice),
             "assemble MTP target hidden batch");
  project(binding(*exact_, "fusion_projection"), mtp_fusion_input_, hidden_,
          rows);
  cuda_check(cudaMemcpy(residual_, hidden_,
                        static_cast<std::size_t>(rows) * hidden_size_ *
                            sizeof(float),
                        cudaMemcpyDeviceToDevice),
             "retain MTP attention residual");
  normalize_rows(hidden_, binding(*mtp_attention_, "input_norm").f32,
                 normalized_, rows);
  std::array<std::uint32_t, kMaximumWorkspaceRows> cache_positions{};
  std::array<std::uint32_t, 3U * kMaximumWorkspaceRows> mrope_positions{};
  for (std::uint32_t row = 0U; row < rows; ++row)
    cache_positions[row] = state.mtp_length + row;
  for (std::uint32_t row = 0U;
       row < rows && !state.prompt_mrope_positions.empty(); ++row) {
    const auto position = rotary_positions[row];
    if (position < state.prompt_mrope_positions.size() / 3U) {
      std::copy_n(state.prompt_mrope_positions.data() + 3U * position, 3U,
                  mrope_positions.data() + 3U * row);
    } else {
      const auto adjusted = static_cast<std::int64_t>(position) +
                            state.rope_delta;
      if (adjusted < 0 ||
          adjusted > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("adjusted MTP rotary position is invalid");
      std::fill_n(mrope_positions.data() + 3U * row, 3U,
                  static_cast<std::uint32_t>(adjusted));
    }
  }
  run_full_attention(*mtp_attention_, state,
                     std::span(cache_positions).first(rows),
                     rotary_positions, rows,
                     mtp_attention_->full_attention_slot,
                     state.prompt_mrope_positions.empty()
                         ? std::span<const std::uint32_t>{}
                         : std::span(mrope_positions).first(3U * rows));
  status_check(ec::add_in_place(hidden_, residual_, rows * hidden_size_,
                                nullptr, activation_bf16_));
  run_ffn(*mtp_ffn_, rows);
  state.mtp_length += rows;
  if (!produce_logits) {
    end_gpu_phase(phase_event);
    return std::nullopt;
  }
  const auto* final_hidden =
      hidden_ + static_cast<std::size_t>(rows - 1U) * hidden_size_;
  cuda_check(cudaMemcpy(
                 slot_mtp_last_hidden_ +
                     static_cast<std::size_t>(state.slot()) * hidden_size_,
                 final_hidden, hidden_size_ * sizeof(float),
                 cudaMemcpyDeviceToDevice),
             "retain final MTP hidden state");
  normalize_rows(final_hidden, binding(*exact_, "draft_norm").f32,
                 normalized_, 1U);
  ensure_logits_capacity(1U);
  const auto draft_vocabulary =
      exact_decode_abi_ >= 2U ? draft_vocabulary_size_ : vocabulary_size_;
  if (draft_vocabulary == vocabulary_size_)
    project(binding(*exact_, "output_head"), normalized_, logits_, 1U);
  else
    project_prefix(binding(*exact_, "output_head"), normalized_, logits_,
                   draft_vocabulary);
  MtpPrediction result;
  if (request) {
    result.distribution = distribution_from_logits(
        logits_, draft_vocabulary, *request, presence);
    result.token = er::sample_distribution(
        result.distribution,
        er::counter_uniform(
            request_parameter(*request, "sampling_seed"), sample_position,
            random_stream));
  } else {
    status_check(ec::argmax(logits_, draft_vocabulary, output_tokens_,
                            nullptr));
    cuda_check(cudaMemcpy(&result.token, output_tokens_, sizeof(result.token),
                          cudaMemcpyDeviceToHost),
               "copy final MTP draft token");
    result.distribution.entries.push_back({result.token, 1.0});
  }
  end_gpu_phase(phase_event);
  collect_gpu_phases();
  return result;
}

void DenseFp4Provider::extend_mtp_rollout(
    RequestState& state, const er::ProgramRequestContext& request,
    MtpPrediction first, std::uint32_t context_limit) {
  if (exact_decode_abi_ < 2U || draft_depth_ == 0U ||
      context_limit > max_context_ || state.mtp_length >= context_limit ||
      !(first.distribution.probability(first.token) > 0.0))
    throw std::runtime_error("invalid MTP rollout root");
  state.draft_predictions.clear();
  state.draft_predictions.reserve(draft_depth_);
  const auto committed_length = state.mtp_length;
  if (committed_length + 1U >= context_limit)
    throw std::runtime_error("MTP rollout root exceeds context");
  state.draft_predictions.push_back(std::move(first));
  cuda_check(cudaMemset(sampling_proposal_presence_ +
                            state.draft_predictions.back().token,
                        1, 1),
             "mark speculative proposal presence");
  while (state.draft_predictions.size() < draft_depth_ &&
         state.mtp_length + 2U < context_limit) {
    cuda_check(cudaMemcpy(
                   mtp_rollout_previous_hidden_,
                   slot_mtp_last_hidden_ +
                       static_cast<std::size_t>(state.slot()) * hidden_size_,
                   hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
               "retain MTP rollout hidden state");
    const auto sample_position = state.mtp_length;
    const std::array token{state.draft_predictions.back().token};
    const std::array rotary_position{state.mtp_length + 1U};
    auto prediction = run_mtp(
        state, token, mtp_rollout_previous_hidden_, rotary_position, true,
        &request, sampling_proposal_presence_, sample_position,
        1U + static_cast<std::uint32_t>(state.draft_predictions.size()));
    if (!prediction)
      throw std::runtime_error("MTP rollout produced no proposal");
    state.draft_predictions.push_back(std::move(*prediction));
    cuda_check(cudaMemset(sampling_proposal_presence_ +
                              state.draft_predictions.back().token,
                          1, 1),
               "mark speculative proposal presence");
  }
  state.draft_token = state.draft_predictions.front().token;
  state.draft_valid = !state.draft_predictions.empty();
  state.mtp_length = committed_length;
}

bool DenseFp4Provider::supports_program_sequence(
    const er::CompiledModelProgram& program) const noexcept {
  try {
    if (program.operations.size() != prepared_target_.size() ||
        program.operations.empty() ||
        program.inputs.size() != (vision_enabled_ ? 3U : 2U) ||
        program.outputs.size() != 1U ||
        std::any_of(prepared_target_.begin(), prepared_target_.end(),
                    [](const auto& operation) { return !operation; }))
      return false;

    std::optional<std::uint32_t> token_input;
    std::optional<std::uint32_t> position_input;
    std::optional<std::uint32_t> media_input;
    for (const auto& endpoint : program.inputs) {
      if (endpoint.value_index >= program.values.size()) return false;
      const auto& abi = program.values[endpoint.value_index].abi;
      if (abi == kTokenAbi && !token_input)
        token_input = endpoint.value_index;
      else if (abi == kPositionAbi && !position_input)
        position_input = endpoint.value_index;
      else if (abi == kMultimodalAbi && !media_input)
        media_input = endpoint.value_index;
      else
        return false;
    }
    if (!token_input || !position_input ||
        (vision_enabled_ != media_input.has_value()))
      return false;

    const auto output_endpoint = program.outputs.front();
    if (output_endpoint.value_index >= program.values.size() ||
        program.values[output_endpoint.value_index].abi != kTokenAbi)
      return false;

    const auto value_for_port = [](const er::CompiledOperationProgram& op,
                                   std::string_view port,
                                   bool output) -> std::optional<std::uint32_t> {
      const auto& bindings = output ? op.output_values : op.input_values;
      const auto found = std::find_if(
          bindings.begin(), bindings.end(),
          [port](const auto& binding) { return binding.port == port; });
      if (found == bindings.end()) return std::nullopt;
      return found->value_index;
    };

    std::optional<std::uint32_t> hidden;
    std::optional<std::uint32_t> hyper;
    std::optional<std::uint32_t> retained;
    std::optional<std::uint32_t> injection;
    std::optional<std::uint32_t> expert_input;
    std::optional<std::uint32_t> route_indices;
    std::optional<std::uint32_t> route_weights;
    std::optional<std::uint32_t> residual;
    std::optional<std::uint32_t> shared_output;
    for (std::size_t index = 0U; index < program.operations.size(); ++index) {
      const auto& compiled = program.operations[index];
      const auto& prepared = *prepared_target_[index];
      if (prepared.logical_operation != index) return false;
      switch (prepared.kernel) {
        case Kernel::embedding:
          if (index != 0U || compiled.input_values.size() != 1U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "token_ids", false) != token_input)
            return false;
          hidden = value_for_port(compiled, "hidden", true);
          if (!hidden) return false;
          break;
        case Kernel::hyper_initialize:
          if (!hidden || hyper || retained || injection ||
              compiled.input_values.size() != 1U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "hidden", false) != hidden)
            return false;
          hyper = value_for_port(compiled, "hyper", true);
          if (!hyper) return false;
          hidden.reset();
          break;
        case Kernel::ple:
          if (!hyper || hidden || retained || injection ||
              compiled.input_values.size() != 2U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "hyper", false) != hyper ||
              value_for_port(compiled, "token_ids", false) != token_input)
            return false;
          hyper = value_for_port(compiled, "hyper", true);
          if (!hyper) return false;
          break;
        case Kernel::hyper_read:
          if (!hyper || hidden || retained || injection ||
              compiled.input_values.size() != 1U ||
              compiled.output_values.size() != 3U ||
              value_for_port(compiled, "hyper", false) != hyper)
            return false;
          hidden = value_for_port(compiled, "hidden", true);
          retained = value_for_port(compiled, "retained", true);
          injection = value_for_port(compiled, "injection", true);
          if (!hidden || !retained || !injection) return false;
          hyper.reset();
          break;
        case Kernel::hyper_inject:
          if (!hidden || hyper || !retained || !injection ||
              compiled.input_values.size() != 3U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "retained", false) != retained ||
              value_for_port(compiled, "hidden", false) != hidden ||
              value_for_port(compiled, "injection", false) != injection)
            return false;
          hyper = value_for_port(compiled, "hyper", true);
          if (!hyper) return false;
          hidden.reset();
          retained.reset();
          injection.reset();
          break;
        case Kernel::hyper_reduce:
          if (!hyper || hidden || retained || injection ||
              compiled.input_values.size() != 1U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "hyper", false) != hyper)
            return false;
          hidden = value_for_port(compiled, "hidden", true);
          if (!hidden) return false;
          hyper.reset();
          break;
        case Kernel::vision:
          if (!vision_enabled_ || !hidden || !media_input ||
              compiled.input_values.size() != 2U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "hidden", false) != hidden ||
              value_for_port(compiled, "media", false) != media_input)
            return false;
          hidden = value_for_port(compiled, "hidden", true);
          if (!hidden) return false;
          break;
        case Kernel::full_attention:
        case Kernel::recurrent_attention:
          if (!hidden || compiled.input_values.size() != 2U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "hidden", false) != hidden ||
              value_for_port(compiled, "positions", false) !=
                  position_input)
            return false;
          hidden = value_for_port(compiled, "hidden", true);
          if (!hidden) return false;
          break;
        case Kernel::router: {
          const auto no_residual =
              prepared.capability ==
              "router.linear-topk.shared-swiglu.no-residual.v1";
          if (!hidden || expert_input || route_indices || route_weights ||
              residual || shared_output ||
              compiled.input_values.size() != 1U ||
              compiled.output_values.size() != (no_residual ? 4U : 5U) ||
              value_for_port(compiled, "hidden", false) != hidden)
            return false;
          expert_input = value_for_port(compiled, "expert_input", true);
          route_indices = value_for_port(compiled, "route_indices", true);
          route_weights = value_for_port(compiled, "route_weights", true);
          if (!no_residual)
            residual = value_for_port(compiled, "residual", true);
          shared_output = value_for_port(compiled, "shared_output", true);
          if (!expert_input || !route_indices || !route_weights ||
              (!no_residual && !residual) || !shared_output)
            return false;
          hidden.reset();
          break;
        }
        case Kernel::routed_moe: {
          const auto no_residual =
              prepared.capability ==
              "moe.swiglu.routed.merge-shared.no-residual.v1";
          if (hidden || !expert_input || !route_indices || !route_weights ||
              (!no_residual && !residual) || !shared_output ||
              compiled.input_values.size() != (no_residual ? 4U : 5U) ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "expert_input", false) !=
                  expert_input ||
              value_for_port(compiled, "route_indices", false) !=
                  route_indices ||
              value_for_port(compiled, "route_weights", false) !=
                  route_weights ||
              (!no_residual &&
               value_for_port(compiled, "residual", false) != residual) ||
              value_for_port(compiled, "shared_output", false) !=
                  shared_output)
            return false;
          hidden = value_for_port(compiled, "hidden", true);
          if (!hidden) return false;
          expert_input.reset();
          route_indices.reset();
          route_weights.reset();
          residual.reset();
          shared_output.reset();
          break;
        }
        case Kernel::ffn:
          if (!hidden || compiled.input_values.size() != 1U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "hidden", false) != hidden)
            return false;
          hidden = value_for_port(compiled, "hidden", true);
          if (!hidden) return false;
          break;
        case Kernel::head:
          if (!hidden || expert_input || route_indices || route_weights ||
              residual || shared_output ||
              index + 1U != program.operations.size() ||
              compiled.input_values.size() != 1U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "hidden", false) != hidden ||
              value_for_port(compiled, "token_ids", true) !=
                  output_endpoint.value_index)
            return false;
          hidden.reset();
          break;
        case Kernel::exact_decode:
          return false;
      }
    }
    return !hidden && !hyper && !retained && !injection && !expert_input &&
           !route_indices && !route_weights && !residual && !shared_output;
  } catch (...) {
    return false;
  }
}

er::OperationExecutionHandle DenseFp4Provider::execute_program_sequence(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const er::ProgramSequenceInvocation& invocation) {
  try {
    const auto request = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!request || !supports_program_sequence(invocation.program) ||
        invocation.operations.size() != prepared_target_.size() ||
        invocation.inputs.size() != invocation.program.inputs.size())
      throw std::runtime_error("dense FP4 program-sequence contract is invalid");
    ensure_sequence_workspace();

    auto sequence = std::make_shared<SequenceState>();
    sequence->request = request;
    sequence->generation = invocation.request;
    // The request flag expresses service intent.  Only arm the provider-side
    // synchronization contract when this artifact actually published and
    // prepared an exact-decode program; otherwise a normal target-only
    // prefill would leave rows waiting for a synchronization the executor
    // cannot issue.
    request->exact_decode_enabled =
        exact_ &&
        request_parameter(invocation.request, "exact_decode_enabled") != 0U;
    const auto retention =
        invocation.request.parameters.find("retention_checkpoint_position");
    if (retention != invocation.request.parameters.end()) {
      if (retention->second == 0U || retention->second > max_context_)
        throw std::runtime_error(
            "program-sequence retention checkpoint is invalid");
      sequence->retention_position =
          static_cast<std::uint32_t>(retention->second);
    }
    sequence->operations.reserve(invocation.operations.size());
    for (std::size_t index = 0U; index < invocation.operations.size(); ++index) {
      const auto* operation =
          dynamic_cast<const PreparedOperation*>(invocation.operations[index]);
      if (!operation || operation != prepared_target_[index].get())
        throw std::runtime_error(
            "program-sequence prepared operation order is invalid");
      sequence->operations.push_back(operation);
    }

    const er::ExecutionValue* token_value{};
    const er::ExecutionValue* position_value{};
    const er::ExecutionValue* media_value{};
    for (std::size_t index = 0U; index < invocation.inputs.size(); ++index) {
      const auto value_index = invocation.program.inputs[index].value_index;
      if (value_index >= invocation.program.values.size())
        throw std::runtime_error("program-sequence input value is invalid");
      const auto& abi = invocation.program.values[value_index].abi;
      if (abi == kTokenAbi)
        token_value = &invocation.inputs[index];
      else if (abi == kPositionAbi)
        position_value = &invocation.inputs[index];
      else if (abi == kMultimodalAbi)
        media_value = &invocation.inputs[index];
    }
    const auto copy_host_u32 = [this](const er::ExecutionValue* value,
                                      std::string_view abi,
                                      std::string_view description) {
      if (!value || !value->valid() || value->abi != abi ||
          value->memory_domain != "host" ||
          value->bytes % sizeof(std::uint32_t) != 0U ||
          value->bytes == 0U ||
          value->bytes / sizeof(std::uint32_t) > max_context_ ||
          reinterpret_cast<std::uintptr_t>(value->data) %
                  alignof(std::uint32_t) !=
              0U)
        throw std::runtime_error(std::string(description) + " ABI mismatch");
      std::vector<std::uint32_t> result(
          static_cast<std::size_t>(value->bytes / sizeof(std::uint32_t)));
      std::memcpy(result.data(), value->data,
                  result.size() * sizeof(result[0]));
      return result;
    };
    sequence->tokens =
        copy_host_u32(token_value, kTokenAbi, "program-sequence token batch");
    sequence->positions = copy_host_u32(
        position_value, kPositionAbi, "program-sequence position batch");
    if (sequence->tokens.size() != sequence->positions.size() ||
        sequence->tokens.size() > max_context_ ||
        std::any_of(sequence->tokens.begin(), sequence->tokens.end(),
                    [this](std::uint32_t token) {
                      return token >= vocabulary_size_;
                    }))
      throw std::runtime_error("program-sequence token stream is invalid");
    for (std::size_t row = 0U; row < sequence->positions.size(); ++row) {
      if (sequence->positions[row] >= max_context_ ||
          (row != 0U &&
           sequence->positions[row] != sequence->positions.front() + row))
        throw std::runtime_error(
            "program-sequence position stream is not contiguous");
    }
    if (vision_enabled_) {
      if (!media_value)
        throw std::runtime_error("program-sequence media input is absent");
      sequence->media = parse_media_payload(
          *media_value, static_cast<std::uint32_t>(sequence->tokens.size()),
          vision_patch_dimension_, vision_spatial_merge_size_);
      if (!sequence->media.empty()) {
        request->rope_delta = sequence->media.rope_delta;
        request->prompt_mrope_positions = sequence->media.positions_thw;
        slot_rope_deltas_.at(request->slot()) = request->rope_delta;
      }
    }
    if (sequence->retention_position > sequence->positions.back() + 1U)
      throw std::runtime_error(
          "program-sequence retention checkpoint is outside its positions");
    if (sequence->retention_position <= sequence->positions.front())
      sequence->retention_position = 0U;
    if (request->synchronization_rows != 0U)
      throw std::runtime_error("previous target batch was not synchronized");
    if (!sequence_tile_rows_ ||
        sequence->tokens.size() >
            std::numeric_limits<std::size_t>::max() / hidden_size_)
      throw std::runtime_error("program-sequence hidden state is too large");
    sequence->tile_rows =
        std::min<std::size_t>(sequence_tile_rows_, sequence->tokens.size());
    if (request->exact_decode_enabled)
      sequence->hidden.resize(sequence->tokens.size() * hidden_size_);

    return er::OperationExecutionHandle::from_callbacks(
        [this, sequence] { return poll_program_sequence(sequence); },
        [sequence] { sequence->cancelled.store(true); });
  } catch (const std::exception& error) {
    return completed_operation(
        {{er::ErrorCode::internal, error.what()}, {}});
  }
}

er::Status DenseFp4Provider::checkpoint_request_state(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    std::uint32_t next_position) {
  try {
    std::lock_guard lock(mutex_);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!state || next_position == 0U || next_position > max_context_ ||
        state->synchronization_rows != 0U)
      throw std::runtime_error("retention checkpoint position is invalid");
    if (state->retention_position == next_position &&
        state->current_position + 1U != next_position) {
      state->retention_valid = true;
      return er::Status::success();
    }
    if (state->current_position + 1U != next_position ||
        (exact_ && state->exact_decode_enabled &&
         state->mtp_length != next_position))
      throw std::runtime_error("retention checkpoint position is invalid");
    const auto conv_values = recurrent_conv_values_;
    const auto matrix_values = recurrent_matrix_values_;
    for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
      cuda_check(cudaMemcpy(
                     recurrent_conv_retention_checkpoint_[layer] +
                         static_cast<std::size_t>(state->slot()) * conv_values,
                     recurrent_conv(layer, state->slot()),
                     conv_values * sizeof(float), cudaMemcpyDeviceToDevice),
                 "checkpoint retained recurrent convolution state");
      cuda_check(cudaMemcpy(
                     recurrent_matrix_retention_checkpoint_[layer] +
                         static_cast<std::size_t>(state->slot()) *
                             matrix_values,
                     recurrent_matrix(layer, state->slot()),
                     matrix_values * sizeof(float), cudaMemcpyDeviceToDevice),
                 "checkpoint retained recurrent matrix state");
    }
    for (std::uint32_t layer = 0U; layer < ple_layers_; ++layer)
      cuda_check(cudaMemcpy(
                     ple_conv_retention_checkpoint_[layer] +
                         static_cast<std::size_t>(state->slot()) *
                             ple_conv_state_values_,
                     ple_conv_state_[layer] +
                         static_cast<std::size_t>(state->slot()) *
                             ple_conv_state_values_,
                     ple_conv_state_values_ * sizeof(float),
                     cudaMemcpyDeviceToDevice),
                 "checkpoint retained PLE convolution state");
    state->ple_retention_history = state->ple_history;
    cuda_check(cudaMemcpy(
                   slot_retention_last_hidden_ +
                       static_cast<std::size_t>(state->slot()) * hidden_size_,
                   slot_last_hidden(state->slot()),
                   hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
               "checkpoint retained target hidden state");
    checkpoint_window_state(state->slot());
    state->retention_position = next_position;
    state->retention_valid = true;
    return er::Status::success();
  } catch (const std::exception& error) {
    return {er::ErrorCode::internal, error.what()};
  }
}

er::Status DenseFp4Provider::rewind_request_state(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    std::uint32_t next_position) {
  try {
    std::lock_guard lock(mutex_);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!state || !state->retention_valid ||
        state->retention_position != next_position ||
        state->synchronization_rows != 0U)
      throw std::runtime_error("retention rewind position is invalid");
    const auto conv_values = recurrent_conv_values_;
    const auto matrix_values = recurrent_matrix_values_;
    for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
      cuda_check(cudaMemcpy(
                     recurrent_conv(layer, state->slot()),
                     recurrent_conv_retention_checkpoint_[layer] +
                         static_cast<std::size_t>(state->slot()) * conv_values,
                     conv_values * sizeof(float), cudaMemcpyDeviceToDevice),
                 "rewind retained recurrent convolution state");
      cuda_check(cudaMemcpy(
                     recurrent_matrix(layer, state->slot()),
                     recurrent_matrix_retention_checkpoint_[layer] +
                         static_cast<std::size_t>(state->slot()) *
                             matrix_values,
                     matrix_values * sizeof(float), cudaMemcpyDeviceToDevice),
                 "rewind retained recurrent matrix state");
    }
    for (std::uint32_t layer = 0U; layer < ple_layers_; ++layer)
      cuda_check(cudaMemcpy(
                     ple_conv_state_[layer] +
                         static_cast<std::size_t>(state->slot()) *
                             ple_conv_state_values_,
                     ple_conv_retention_checkpoint_[layer] +
                         static_cast<std::size_t>(state->slot()) *
                             ple_conv_state_values_,
                     ple_conv_state_values_ * sizeof(float),
                     cudaMemcpyDeviceToDevice),
                 "rewind retained PLE convolution state");
    state->ple_history = state->ple_retention_history;
    cuda_check(cudaMemcpy(
                   slot_last_hidden(state->slot()),
                   slot_retention_last_hidden_ +
                       static_cast<std::size_t>(state->slot()) * hidden_size_,
                   hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
               "rewind retained target hidden state");
    restore_window_checkpoint(state->slot());
    if (host_authoritative_fp16_kv()) {
      trim_host_kv(*state, next_position);
      if (state->target_mirror_enabled)
        trim_target_mirror(*state, next_position);
      else
        restore_target_mirror(*state, state->slot());
    }
    state->current_position = next_position - 1U;
    state->current_batch_first = next_position - 1U;
    state->current_batch_rows = 0U;
    state->mtp_length = next_position;
    state->synchronization_first = 0U;
    state->synchronization_rows = 0U;
    state->synchronization_consumed = 0U;
    state->draft_valid = false;
    state->draft_predictions.clear();
    state->sequence_target_hidden.clear();
    return er::Status::success();
  } catch (const std::exception& error) {
    return {er::ErrorCode::internal, error.what()};
  }
}

er::RequestStateParkingResult DenseFp4Provider::park_request_state(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    std::uint32_t next_position) {
  try {
    std::lock_guard lock(mutex_);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!state || state->parked() ||
        state->slot() >= capacity_ || !slot_in_use_[state->slot()] ||
        next_position == 0U || next_position > max_context_ ||
        state->current_position + 1U != next_position ||
        !state->retention_valid ||
        state->retention_position != next_position ||
        state->synchronization_rows != 0U)
      throw std::runtime_error("request state is not parkable");

    const auto page_count =
        (static_cast<std::uint64_t>(next_position) + kv_page_tokens_ - 1U) /
        kv_page_tokens_;
    if (host_authoritative_fp16_kv() &&
        (state->host_kv_populated_tokens < next_position ||
         state->host_kv_pages.size() < page_count))
      throw std::runtime_error(
          "authoritative FP16 KV prefix is incomplete at park");
    const auto conv_values = recurrent_conv_values_;
    const auto matrix_values = recurrent_matrix_values_;
    const auto recurrent_values =
        static_cast<std::uint64_t>(recurrent_layers_) *
        (conv_values + matrix_values);
    const auto ple_values = static_cast<std::uint64_t>(ple_layers_) *
                            ple_conv_state_values_;
    const auto ple_history_bytes =
        static_cast<std::uint64_t>(state->ple_history.size()) *
        sizeof(std::uint32_t);
    const auto state_bytes =
        (recurrent_values + ple_values) * sizeof(float) +
        ple_history_bytes +
        3U * static_cast<std::uint64_t>(hidden_size_) * sizeof(float);
    std::uint64_t resident_page_count{};
    for (std::uint64_t page_index = 0U; page_index < page_count;
         ++page_index)
      if (slot_pages_.at(state->slot()).at(
              static_cast<std::size_t>(page_index)))
        ++resident_page_count;
    if (!host_authoritative_fp16_kv() && resident_page_count != page_count)
      throw std::runtime_error("populated request KV page is absent");
    if ((kv_page_bytes_ == 0U && resident_page_count != 0U) ||
        (kv_page_bytes_ != 0U &&
         resident_page_count > std::numeric_limits<std::uint64_t>::max() /
                                   kv_page_bytes_))
      throw std::runtime_error("parked KV byte count overflows");
    const auto page_bytes = resident_page_count * kv_page_bytes_;
    const auto window_bytes = window_kv_bytes_per_slot_;
    if (page_bytes > std::numeric_limits<std::uint64_t>::max() - window_bytes ||
        state_bytes > std::numeric_limits<std::uint64_t>::max() -
                          page_bytes - window_bytes)
      throw std::runtime_error("parked request byte count overflows");
    const auto required_bytes = page_bytes + window_bytes + state_bytes;
    if (required_bytes > parking_ram_capacity_bytes_ ||
        host_kv_bytes_ + parked_request_bytes_ >
            parking_ram_capacity_bytes_ - required_bytes)
      return {{er::ErrorCode::backpressure,
               "request-state RAM parking capacity is exhausted"},
              0U, 0U};

    ParkedRequestState parked;
    parked.bytes = required_bytes;
    if (state->host_kv_bytes >
        std::numeric_limits<std::uint64_t>::max() - required_bytes)
      throw std::runtime_error("parked session byte count overflows");
    parked.reported_bytes = required_bytes + state->host_kv_bytes;
    if (page_count > std::numeric_limits<std::uint32_t>::max() ||
        required_bytes > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("parked request geometry overflows");
    parked.logical_pages = static_cast<std::uint32_t>(page_count);
    parked.page_payload_bytes = page_bytes;
    parked.window_payload_bytes = window_bytes;
    parked.page_indices.reserve(static_cast<std::size_t>(resident_page_count));
    parked.payload.allocate(static_cast<std::size_t>(required_bytes));
    const auto slot = state->slot();
    if (state->target_mirror_enabled) {
      release_target_mirror_pages(slot);
      state->target_mirror_enabled = false;
    }
    std::size_t page_ordinal{};
    for (std::uint64_t page_index = 0U; page_index < page_count;
         ++page_index) {
      const auto* page = slot_pages_.at(slot).at(
          static_cast<std::size_t>(page_index));
      if (!page) continue;
      parked.page_indices.push_back(static_cast<std::uint32_t>(page_index));
      cuda_check(cudaMemcpyAsync(
                     parked.payload.data + page_ordinal * kv_page_bytes_, page,
                     static_cast<std::size_t>(kv_page_bytes_),
                     cudaMemcpyDeviceToHost, parking_stream_),
                 "park request KV page");
      ++page_ordinal;
    }

    std::size_t payload_offset = static_cast<std::size_t>(page_bytes);
    if (window_bytes != 0U) {
      cuda_check(cudaMemcpyAsync(
                     parked.payload.data + payload_offset,
                     window_kv_ + static_cast<std::size_t>(slot) *
                                      window_kv_bytes_per_slot_,
                     static_cast<std::size_t>(window_bytes),
                     cudaMemcpyDeviceToHost, parking_stream_),
                 "park window-attention state");
      payload_offset += static_cast<std::size_t>(window_bytes);
    }
    const auto copy_layers = [&](const std::vector<float*>& sources,
                                 std::size_t values, std::string_view label) {
      for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
        cuda_check(cudaMemcpyAsync(
                       parked.payload.data + payload_offset,
                       sources.at(layer) +
                           static_cast<std::size_t>(slot) * values,
                       values * sizeof(float), cudaMemcpyDeviceToHost,
                       parking_stream_),
                   label);
        payload_offset += values * sizeof(float);
      }
    };
    copy_layers(recurrent_conv_state_, conv_values,
                "park recurrent convolution state");
    copy_layers(recurrent_matrix_state_,
                matrix_values, "park recurrent matrix state");
    for (std::uint32_t layer = 0U; layer < ple_layers_; ++layer) {
      cuda_check(cudaMemcpyAsync(
                     parked.payload.data + payload_offset,
                     ple_conv_state_[layer] +
                         static_cast<std::size_t>(slot) *
                             ple_conv_state_values_,
                     ple_conv_state_values_ * sizeof(float),
                     cudaMemcpyDeviceToHost, parking_stream_),
                 "park PLE convolution state");
      payload_offset += ple_conv_state_values_ * sizeof(float);
    }
    if (ple_history_bytes != 0U) {
      std::memcpy(parked.payload.data + payload_offset,
                  state->ple_history.data(),
                  static_cast<std::size_t>(ple_history_bytes));
      payload_offset += static_cast<std::size_t>(ple_history_bytes);
    }
    const auto hidden_bytes = static_cast<std::size_t>(hidden_size_) *
                              sizeof(float);
    cuda_check(cudaMemcpyAsync(parked.payload.data + payload_offset,
                               slot_last_hidden(slot), hidden_bytes,
                               cudaMemcpyDeviceToHost, parking_stream_),
               "park target hidden state");
    payload_offset += hidden_bytes;
    cuda_check(cudaMemcpyAsync(
                   parked.payload.data + payload_offset,
                   slot_mtp_last_hidden_ +
                       static_cast<std::size_t>(slot) * hidden_size_,
                   hidden_bytes, cudaMemcpyDeviceToHost, parking_stream_),
               "park MTP hidden state");
    payload_offset += hidden_bytes;
    cuda_check(cudaMemcpyAsync(
                   parked.payload.data + payload_offset,
                   slot_retention_last_hidden_ +
                       static_cast<std::size_t>(slot) * hidden_size_,
                   hidden_bytes, cudaMemcpyDeviceToHost, parking_stream_),
               "park retained target hidden state");
    payload_offset += hidden_bytes;
    if (payload_offset != required_bytes ||
        page_ordinal != parked.page_indices.size())
      throw std::runtime_error("parked request blob layout is inconsistent");

    const auto table_bytes = static_cast<std::size_t>(maximum_pages_per_slot_) *
                             sizeof(void*);
    cuda_check(cudaMemsetAsync(
                   device_page_table_ + static_cast<std::size_t>(slot) *
                                            maximum_pages_per_slot_,
                   0, table_bytes, parking_stream_),
               "unpublish parked KV page table");
    if (device_mtp_page_table_ != device_page_table_)
      cuda_check(cudaMemsetAsync(
                     device_mtp_page_table_ +
                         static_cast<std::size_t>(slot) *
                             maximum_pages_per_slot_,
                     0, table_bytes, parking_stream_),
                 "unpublish parked MTP KV page table");
    if (device_target_mirror_page_table_)
      cuda_check(cudaMemsetAsync(
                     device_target_mirror_page_table_ +
                         static_cast<std::size_t>(slot) *
                             maximum_pages_per_slot_,
                     0, table_bytes, parking_stream_),
                 "unpublish parked exact FP16 target mirror table");
    cuda_check(cudaStreamSynchronize(parking_stream_),
               "finish request-state parking transfers");

    for (std::uint32_t page_index = 0U;
         page_index < maximum_pages_per_slot_; ++page_index) {
      auto*& page = slot_pages_[slot][page_index];
      if (page) {
        free_kv_pages_.push_back(page);
        page = nullptr;
      }
    }
    slot_rope_deltas_[slot] = 0;
    slot_in_use_[slot] = false;
    state->slot_ = kNoSlot;
    state->parked_state.emplace(std::move(parked));
    parked_request_bytes_ += required_bytes;
    parked_session_bytes_ += state->parked_state->reported_bytes;
    park_device_to_host_bytes_ += required_bytes;
    ++park_calls_;
    return {er::Status::success(), page_count,
            state->parked_state->reported_bytes};
  } catch (const std::exception& error) {
    if (parking_stream_)
      static_cast<void>(cudaStreamSynchronize(parking_stream_));
    return {{er::ErrorCode::internal, error.what()}, 0U, 0U};
  }
}

er::RequestStateParkingResult DenseFp4Provider::restore_request_state(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state) {
  try {
    std::lock_guard lock(mutex_);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!state || !state->parked() ||
        state->slot() != kNoSlot)
      throw std::runtime_error("request state is not parked");
    const auto free = std::find(slot_in_use_.begin(), slot_in_use_.end(), false);
    if (free == slot_in_use_.end())
      return {{er::ErrorCode::backpressure,
               "dense FP4 provider has no free request slot"},
              0U, 0U};
    const auto& parked = *state->parked_state;
    const auto page_count = static_cast<std::uint64_t>(parked.logical_pages);
    const auto resident_page_count =
        static_cast<std::uint64_t>(parked.page_indices.size());
    if (page_count > maximum_pages_per_slot_ ||
        parked.page_payload_bytes != resident_page_count * kv_page_bytes_ ||
        parked.window_payload_bytes != window_kv_bytes_per_slot_ ||
        parked.payload.bytes != parked.bytes)
      throw std::runtime_error("parked request blob is invalid");
    if (parked.bytes > parked_request_bytes_ ||
        parked.reported_bytes > parked_session_bytes_)
      throw std::runtime_error("parked request accounting underflow");
    const auto unallocated = kv_page_capacity_ - all_kv_pages_.size();
    if (resident_page_count > free_kv_pages_.size() + unallocated)
      return {{er::ErrorCode::backpressure,
               "KV physical page budget is exhausted during restore"},
              0U, 0U};

    const auto slot = static_cast<std::uint32_t>(free - slot_in_use_.begin());
    *free = true;
    std::vector<std::uint32_t> published;
    published.reserve(static_cast<std::size_t>(resident_page_count));
    try {
      const auto table_bytes =
          static_cast<std::size_t>(maximum_pages_per_slot_) * sizeof(void*);
      const auto table_count =
          device_mtp_page_table_ == device_page_table_ ? 1U : 2U;
      PinnedHostBuffer host_tables;
      host_tables.allocate(table_count * table_bytes);
      std::memset(host_tables.data, 0, host_tables.bytes);
      auto** target_table = reinterpret_cast<void**>(host_tables.data);
      auto** mtp_table = table_count == 1U
          ? target_table
          : reinterpret_cast<void**>(host_tables.data + table_bytes);
      for (std::size_t ordinal = 0U;
           ordinal < parked.page_indices.size(); ++ordinal) {
        const auto page_index = parked.page_indices[ordinal];
        if (page_index >= page_count)
          throw std::runtime_error("parked KV page index is invalid");
        auto*& page = slot_pages_[slot][page_index];
        if (!free_kv_pages_.empty()) {
          page = free_kv_pages_.back();
          free_kv_pages_.pop_back();
        } else {
          cuda_check(cudaMalloc(&page, static_cast<std::size_t>(kv_page_bytes_)),
                     "allocate restored KV page");
          all_kv_pages_.push_back(page);
        }
        published.push_back(page_index);
        cuda_check(cudaMemcpyAsync(
                       page,
                       parked.payload.data + ordinal * kv_page_bytes_,
                       static_cast<std::size_t>(kv_page_bytes_),
                       cudaMemcpyHostToDevice, parking_stream_),
                   "restore request KV page");
        target_table[page_index] = page;
        if (device_mtp_page_table_ != device_page_table_) {
          mtp_table[page_index] = static_cast<void*>(
              static_cast<std::byte*>(page) + mtp_kv_page_offset_);
        }
      }

      const auto conv_values = recurrent_conv_values_;
      const auto matrix_values = recurrent_matrix_values_;
      std::size_t payload_offset =
          static_cast<std::size_t>(parked.page_payload_bytes);
      if (parked.window_payload_bytes != 0U) {
        auto* window = window_kv_ +
            static_cast<std::size_t>(slot) * window_kv_bytes_per_slot_;
        cuda_check(cudaMemcpyAsync(
                       window, parked.payload.data + payload_offset,
                       static_cast<std::size_t>(parked.window_payload_bytes),
                       cudaMemcpyHostToDevice, parking_stream_),
                   "restore window-attention state");
        cuda_check(cudaMemcpyAsync(
                       window_retention_kv_ +
                           static_cast<std::size_t>(slot) *
                               window_kv_bytes_per_slot_,
                       parked.payload.data + payload_offset,
                       static_cast<std::size_t>(parked.window_payload_bytes),
                       cudaMemcpyHostToDevice, parking_stream_),
                   "initialize retained window-attention state");
        payload_offset +=
            static_cast<std::size_t>(parked.window_payload_bytes);
      }
      const auto restore_layers = [&](const std::vector<float*>& destinations,
                                      std::size_t values,
                                      std::string_view label) {
        for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
          cuda_check(cudaMemcpyAsync(
                         destinations.at(layer) +
                             static_cast<std::size_t>(slot) * values,
                         parked.payload.data + payload_offset,
                         values * sizeof(float), cudaMemcpyHostToDevice,
                         parking_stream_),
                     label);
          payload_offset += values * sizeof(float);
        }
      };
      restore_layers(recurrent_conv_state_, conv_values,
                     "restore recurrent convolution state");
      restore_layers(recurrent_matrix_state_,
                     matrix_values, "restore recurrent matrix state");
      for (std::uint32_t layer = 0U; layer < ple_layers_; ++layer) {
        cuda_check(cudaMemcpyAsync(
                       ple_conv_state_[layer] +
                           static_cast<std::size_t>(slot) *
                               ple_conv_state_values_,
                       parked.payload.data + payload_offset,
                       ple_conv_state_values_ * sizeof(float),
                       cudaMemcpyHostToDevice, parking_stream_),
                   "restore PLE convolution state");
        payload_offset += ple_conv_state_values_ * sizeof(float);
      }
      const auto ple_history_bytes = state->ple_history.size() *
                                     sizeof(std::uint32_t);
      if (ple_history_bytes != 0U) {
        if (payload_offset > parked.payload.bytes ||
            ple_history_bytes > parked.payload.bytes - payload_offset)
          throw std::runtime_error("parked PLE history has invalid size");
        std::memcpy(state->ple_history.data(),
                    parked.payload.data + payload_offset,
                    ple_history_bytes);
        state->ple_retention_history = state->ple_history;
        payload_offset += ple_history_bytes;
      }
      const auto clone_device_layers = [&, slot](
        const std::vector<float*>& destinations,
          const std::vector<float*>& sources, std::size_t values,
          std::string_view label) {
        for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer)
          cuda_check(cudaMemcpyAsync(
                         destinations.at(layer) +
                             static_cast<std::size_t>(slot) * values,
                         sources.at(layer) +
                             static_cast<std::size_t>(slot) * values,
                         values * sizeof(float), cudaMemcpyDeviceToDevice,
                         parking_stream_),
                     label);
      };
      clone_device_layers(recurrent_conv_checkpoint_, recurrent_conv_state_,
                          conv_values,
                          "initialize recurrent convolution checkpoint");
      clone_device_layers(recurrent_matrix_checkpoint_,
                          recurrent_matrix_state_, matrix_values,
                          "initialize recurrent matrix checkpoint");
      clone_device_layers(recurrent_conv_retention_checkpoint_,
                          recurrent_conv_state_, conv_values,
                          "initialize retained recurrent convolution state");
      clone_device_layers(recurrent_matrix_retention_checkpoint_,
                          recurrent_matrix_state_, matrix_values,
                          "initialize retained recurrent matrix state");
      for (std::uint32_t layer = 0U; layer < ple_layers_; ++layer)
        cuda_check(cudaMemcpyAsync(
                       ple_conv_retention_checkpoint_[layer] +
                           static_cast<std::size_t>(slot) *
                               ple_conv_state_values_,
                       ple_conv_state_[layer] +
                           static_cast<std::size_t>(slot) *
                               ple_conv_state_values_,
                       ple_conv_state_values_ * sizeof(float),
                       cudaMemcpyDeviceToDevice, parking_stream_),
                   "initialize retained PLE convolution state");
      const auto hidden_bytes = static_cast<std::size_t>(hidden_size_) *
                                sizeof(float);
      if (payload_offset > parked.payload.bytes ||
          3U * hidden_bytes > parked.payload.bytes - payload_offset)
        throw std::runtime_error("parked hidden state has invalid size");
      cuda_check(cudaMemcpyAsync(slot_last_hidden(slot),
                                 parked.payload.data + payload_offset,
                                 hidden_bytes, cudaMemcpyHostToDevice,
                                 parking_stream_),
                 "restore target hidden state");
      payload_offset += hidden_bytes;
      cuda_check(cudaMemcpyAsync(
                     slot_mtp_last_hidden_ +
                         static_cast<std::size_t>(slot) * hidden_size_,
                     parked.payload.data + payload_offset, hidden_bytes,
                     cudaMemcpyHostToDevice, parking_stream_),
                 "restore MTP hidden state");
      payload_offset += hidden_bytes;
      cuda_check(cudaMemcpyAsync(
                     slot_retention_last_hidden_ +
                         static_cast<std::size_t>(slot) * hidden_size_,
                     parked.payload.data + payload_offset, hidden_bytes,
                     cudaMemcpyHostToDevice, parking_stream_),
                 "restore retained target hidden state");
      payload_offset += hidden_bytes;
      if (payload_offset != parked.payload.bytes)
        throw std::runtime_error("parked request blob layout changed");
      cuda_check(cudaMemcpyAsync(
                     device_page_table_ + static_cast<std::size_t>(slot) *
                                              maximum_pages_per_slot_,
                     target_table, table_bytes, cudaMemcpyHostToDevice,
                     parking_stream_),
                 "publish restored KV page table");
      if (device_mtp_page_table_ != device_page_table_)
        cuda_check(cudaMemcpyAsync(
                       device_mtp_page_table_ +
                           static_cast<std::size_t>(slot) *
                               maximum_pages_per_slot_,
                       mtp_table, table_bytes, cudaMemcpyHostToDevice,
                       parking_stream_),
                   "publish restored MTP KV page table");
      cuda_check(cudaStreamSynchronize(parking_stream_),
                 "finish request-state restore transfers");
      restore_target_mirror(*state, slot);
    } catch (...) {
      static_cast<void>(cudaStreamSynchronize(parking_stream_));
      for (const auto page_index : published) {
        auto*& page = slot_pages_[slot][page_index];
        if (page) free_kv_pages_.push_back(page);
        page = nullptr;
      }
      const auto table_bytes =
          static_cast<std::size_t>(maximum_pages_per_slot_) * sizeof(void*);
      static_cast<void>(cudaMemset(
          device_page_table_ + static_cast<std::size_t>(slot) *
                                   maximum_pages_per_slot_,
          0, table_bytes));
      if (device_mtp_page_table_ != device_page_table_)
        static_cast<void>(cudaMemset(
            device_mtp_page_table_ + static_cast<std::size_t>(slot) *
                                         maximum_pages_per_slot_,
            0, table_bytes));
      release_target_mirror_pages(slot);
      if (device_target_mirror_page_table_)
        static_cast<void>(cudaMemset(
            device_target_mirror_page_table_ +
                static_cast<std::size_t>(slot) * maximum_pages_per_slot_,
            0, table_bytes));
      *free = false;
      throw;
    }

    state->slot_ = slot;
    slot_rope_deltas_[slot] = state->rope_delta;
    const auto bytes = parked.bytes;
    const auto reported_bytes = parked.reported_bytes;
    parked_request_bytes_ -= bytes;
    parked_session_bytes_ -= reported_bytes;
    state->parked_state.reset();
    restore_host_to_device_bytes_ += bytes;
    ++restore_calls_;
    return {er::Status::success(), page_count, reported_bytes};
  } catch (const std::exception& error) {
    if (parking_stream_)
      static_cast<void>(cudaStreamSynchronize(parking_stream_));
    return {{er::ErrorCode::internal, error.what()}, 0U, 0U};
  }
}

er::RequestStateSnapshotResult
DenseFp4Provider::save_request_state_snapshot(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const std::filesystem::path& root, std::uint64_t generation) {
  try {
    std::lock_guard lock(mutex_);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!state || !state->parked() || state->slot() != kNoSlot ||
        generation == 0U || root.empty())
      throw std::runtime_error("request state is not persistable");
    const auto& parked = *state->parked_state;
    if (parked.payload.bytes != parked.bytes ||
        parked.page_payload_bytes !=
            parked.page_indices.size() * kv_page_bytes_ ||
        state->host_kv_bytes !=
            state->host_kv_pages.size() * host_fp16_target_page_bytes_ ||
        parked.reported_bytes != parked.bytes + state->host_kv_bytes)
      throw std::runtime_error("parked request accounting is inconsistent");

    DenseSnapshotManifest manifest;
    manifest.generation = generation;
    manifest.model_hash = descriptor_.content_hash;
    manifest.max_context = max_context_;
    manifest.page_tokens = kv_page_tokens_;
    manifest.page_bytes = kv_page_bytes_;
    manifest.host_page_bytes = host_fp16_target_page_bytes_;
    manifest.current_position = state->current_position;
    manifest.current_batch_first = state->current_batch_first;
    manifest.current_batch_rows = state->current_batch_rows;
    manifest.synchronization_first = state->synchronization_first;
    manifest.synchronization_rows = state->synchronization_rows;
    manifest.synchronization_consumed = state->synchronization_consumed;
    manifest.mtp_length = state->mtp_length;
    manifest.synchronized_token = state->synchronized_token;
    manifest.draft_token = state->draft_token;
    manifest.draft_distribution_offsets.push_back(0U);
    for (const auto& prediction : state->draft_predictions) {
      manifest.draft_tokens.push_back(prediction.token);
      for (const auto& entry : prediction.distribution.entries) {
        manifest.draft_distribution_tokens.push_back(entry.token);
        manifest.draft_distribution_probabilities.push_back(
            entry.probability);
      }
      manifest.draft_distribution_offsets.push_back(
          static_cast<std::uint32_t>(
              manifest.draft_distribution_tokens.size()));
    }
    manifest.retention_position = state->retention_position;
    manifest.rope_delta = state->rope_delta;
    manifest.draft_valid = state->draft_valid;
    manifest.retention_valid = state->retention_valid;
    manifest.exact_decode_enabled = state->exact_decode_enabled;
    manifest.host_kv_populated_tokens = state->host_kv_populated_tokens;
    manifest.logical_pages = parked.logical_pages;
    manifest.page_payload_bytes = parked.page_payload_bytes;
    manifest.window_payload_bytes = parked.window_payload_bytes;
    manifest.parked_bytes = parked.bytes;
    manifest.reported_bytes = parked.reported_bytes;
    manifest.page_indices = parked.page_indices;
    manifest.prompt_mrope_positions = state->prompt_mrope_positions;
    manifest.sequence_target_hidden = state->sequence_target_hidden;
    manifest.ple_history = state->ple_history;
    manifest.ple_retention_history = state->ple_retention_history;
    manifest.host_page_count =
        static_cast<std::uint32_t>(state->host_kv_pages.size());
    manifest.logical_data_bytes = parked.bytes + state->host_kv_bytes;

    ContentAddressedSnapshotWriter writer(root / "blobs", generation);
    writer.append({parked.payload.data,
                   static_cast<std::size_t>(parked.payload.bytes)});
    for (const auto& page : state->host_kv_pages) {
      if (!page.allocation)
        throw std::runtime_error("authoritative host KV page is absent");
      writer.append({reinterpret_cast<const std::byte*>(page.allocation),
                     static_cast<std::size_t>(
                         host_fp16_target_page_bytes_)});
    }
    manifest.chunks = writer.finish();
    const auto manifest_bytes = encode_snapshot_manifest(manifest);
    const auto manifest_path = snapshot_manifest_path(root, generation);
    write_atomic_file(manifest_path, manifest_bytes, generation);
    return {er::Status::success(), generation, parked.logical_pages,
            manifest.logical_data_bytes,
            writer.written_bytes() + manifest_bytes.size()};
  } catch (const std::exception& error) {
    return {{er::ErrorCode::internal, error.what()}, 0U, 0U, 0U, 0U};
  }
}

er::RequestStateSnapshotResult
DenseFp4Provider::load_request_state_snapshot(
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const std::filesystem::path& root, std::uint64_t generation) {
  std::vector<HostKvPage> host_pages;
  try {
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!state || state->parked() || state->slot() >= capacity_ ||
        generation == 0U || root.empty())
      throw std::runtime_error("snapshot load request state is invalid");
    const auto manifest_bytes = read_file(
        snapshot_manifest_path(root, generation), 64U << 20U);
    auto manifest = decode_snapshot_manifest(manifest_bytes);
    if (manifest.generation != generation ||
        !er::constant_time_equal(manifest.model_hash,
                                 descriptor_.content_hash) ||
        manifest.max_context != max_context_ ||
        manifest.page_tokens != kv_page_tokens_ ||
        manifest.page_bytes != kv_page_bytes_ ||
        manifest.host_page_bytes != host_fp16_target_page_bytes_ ||
        manifest.retention_position == 0U ||
        manifest.retention_position > max_context_ ||
        manifest.current_position + 1U != manifest.retention_position ||
        !manifest.retention_valid || manifest.current_batch_rows != 0U ||
        manifest.synchronization_rows != 0U)
      throw std::runtime_error("snapshot identity or request state mismatches");
    if (manifest.draft_distribution_tokens.size() !=
            manifest.draft_distribution_probabilities.size() ||
        manifest.draft_distribution_offsets.size() !=
            manifest.draft_tokens.size() + 1U ||
        manifest.draft_distribution_offsets.empty() ||
        manifest.draft_distribution_offsets.front() != 0U ||
        manifest.draft_distribution_offsets.back() !=
            manifest.draft_distribution_tokens.size() ||
        manifest.draft_tokens.size() > 4U ||
        manifest.draft_distribution_tokens.size() > 4U *
            kMaximumGpuSamplingTopK)
      throw std::runtime_error("snapshot draft distributions are invalid");
    std::vector<MtpPrediction> draft_predictions;
    draft_predictions.reserve(manifest.draft_tokens.size());
    for (std::size_t index = 0U; index < manifest.draft_tokens.size();
         ++index) {
      const auto first = manifest.draft_distribution_offsets[index];
      const auto last = manifest.draft_distribution_offsets[index + 1U];
      if (first >= last || first > last ||
          last > manifest.draft_distribution_tokens.size() ||
          manifest.draft_tokens[index] >= vocabulary_size_)
        throw std::runtime_error("snapshot draft distribution is malformed");
      MtpPrediction prediction;
      prediction.token = manifest.draft_tokens[index];
      prediction.distribution.entries.reserve(last - first);
      for (auto item = first; item < last; ++item) {
        const auto token = manifest.draft_distribution_tokens[item];
        const auto probability =
            manifest.draft_distribution_probabilities[item];
        if (token >= vocabulary_size_)
          throw std::runtime_error("snapshot draft token is invalid");
        prediction.distribution.entries.push_back({token, probability});
      }
      static_cast<void>(
          er::sample_distribution(prediction.distribution, 0.0));
      if (!(prediction.distribution.probability(prediction.token) > 0.0))
        throw std::runtime_error("snapshot proposal has zero probability");
      draft_predictions.push_back(std::move(prediction));
    }
    if ((exact_decode_abi_ >= 2U && manifest.draft_valid &&
         (draft_predictions.empty() ||
          draft_predictions.size() > draft_depth_)) ||
        (!manifest.draft_valid && !draft_predictions.empty()) ||
        (exact_decode_abi_ < 2U && !draft_predictions.empty()))
      throw std::runtime_error("snapshot draft depth disagrees with artifact");
    const auto expected_pages =
        (static_cast<std::uint64_t>(manifest.retention_position) +
         kv_page_tokens_ - 1U) /
        kv_page_tokens_;
    const auto expected_host_bytes =
        static_cast<std::uint64_t>(manifest.host_page_count) *
        host_fp16_target_page_bytes_;
    const auto expected_page_payload =
        static_cast<std::uint64_t>(manifest.page_indices.size()) *
        kv_page_bytes_;
    if (manifest.logical_pages != expected_pages ||
        manifest.logical_pages > maximum_pages_per_slot_ ||
        manifest.page_payload_bytes != expected_page_payload ||
        manifest.window_payload_bytes != window_kv_bytes_per_slot_ ||
        manifest.logical_data_bytes !=
            manifest.parked_bytes + expected_host_bytes ||
        manifest.reported_bytes != manifest.logical_data_bytes ||
        (host_authoritative_fp16_kv()
             ? manifest.host_page_count != expected_pages ||
                   manifest.host_kv_populated_tokens <
                       manifest.retention_position
             : manifest.host_page_count != 0U ||
                   manifest.host_kv_populated_tokens != 0U) ||
        (!host_authoritative_fp16_kv() &&
         manifest.page_indices.size() != expected_pages))
      throw std::runtime_error("snapshot storage geometry is invalid");
    for (std::size_t index = 0U; index < manifest.page_indices.size(); ++index)
      if (manifest.page_indices[index] >= expected_pages ||
          (index != 0U && manifest.page_indices[index - 1U] >=
                              manifest.page_indices[index]))
        throw std::runtime_error("snapshot page index is invalid");
    std::uint64_t chunk_bytes{};
    for (const auto& chunk : manifest.chunks) {
      if (chunk_bytes > std::numeric_limits<std::uint64_t>::max() -
                            chunk.bytes)
        throw std::runtime_error("snapshot chunk bytes overflow");
      chunk_bytes += chunk.bytes;
    }
    if (chunk_bytes != manifest.logical_data_bytes ||
        manifest.parked_bytes > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("snapshot chunk coverage is invalid");

    ParkedRequestState parked;
    parked.logical_pages = manifest.logical_pages;
    parked.page_payload_bytes = manifest.page_payload_bytes;
    parked.window_payload_bytes = manifest.window_payload_bytes;
    parked.bytes = manifest.parked_bytes;
    parked.reported_bytes = manifest.reported_bytes;
    parked.page_indices = manifest.page_indices;
    parked.payload.allocate(static_cast<std::size_t>(parked.bytes));
    host_pages.reserve(manifest.host_page_count);
    for (std::uint32_t index = 0U; index < manifest.host_page_count; ++index) {
      void* allocation{};
      cuda_check(cudaHostAlloc(
                     &allocation,
                     static_cast<std::size_t>(host_fp16_target_page_bytes_),
                     cudaHostAllocPortable),
                 "allocate persisted authoritative FP16 KV page");
      host_pages.push_back({static_cast<std::uint16_t*>(allocation)});
    }

    std::uint64_t cursor{};
    for (const auto& chunk : manifest.chunks) {
      const auto path = root / "blobs" /
                        (digest_hex(chunk.digest) + ".blob");
      const auto bytes = read_file(path, kSnapshotChunkBytes);
      if (bytes.size() != chunk.bytes ||
          !er::constant_time_equal(er::sha256(bytes), chunk.digest))
        throw std::runtime_error("snapshot data chunk is corrupt");
      std::size_t source{};
      while (source < bytes.size()) {
        if (cursor < parked.bytes) {
          const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
              parked.bytes - cursor, bytes.size() - source));
          std::memcpy(parked.payload.data + cursor, bytes.data() + source,
                      count);
          cursor += count;
          source += count;
          continue;
        }
        const auto host_cursor = cursor - parked.bytes;
        const auto page_index = host_cursor / host_fp16_target_page_bytes_;
        const auto page_offset = host_cursor % host_fp16_target_page_bytes_;
        if (page_index >= host_pages.size())
          throw std::runtime_error("snapshot data exceeds declared payload");
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
            host_fp16_target_page_bytes_ - page_offset,
            bytes.size() - source));
        std::memcpy(
            reinterpret_cast<std::byte*>(host_pages[page_index].allocation) +
                page_offset,
            bytes.data() + source, count);
        cursor += count;
        source += count;
      }
    }
    if (cursor != manifest.logical_data_bytes)
      throw std::runtime_error("snapshot data is truncated");

    // A newly created provider state owns an empty execution slot. Release it
    // before installing the independently validated parked state.
    release_slot(state->slot());
    {
      std::lock_guard lock(mutex_);
      if (manifest.parked_bytes > parking_ram_capacity_bytes_ ||
          host_kv_bytes_ + parked_request_bytes_ >
              parking_ram_capacity_bytes_ - manifest.parked_bytes ||
          expected_host_bytes > parking_ram_capacity_bytes_ -
                                    manifest.parked_bytes -
                                    host_kv_bytes_ - parked_request_bytes_)
        throw std::runtime_error(
            "persisted request exceeds host parking capacity");
      state->slot_ = kNoSlot;
      state->current_position = manifest.current_position;
      state->current_batch_first = manifest.current_batch_first;
      state->current_batch_rows = manifest.current_batch_rows;
      state->synchronization_first = manifest.synchronization_first;
      state->synchronization_rows = manifest.synchronization_rows;
      state->synchronization_consumed = manifest.synchronization_consumed;
      state->mtp_length = manifest.mtp_length;
      state->synchronized_token = manifest.synchronized_token;
      state->draft_token = manifest.draft_token;
      state->draft_predictions = std::move(draft_predictions);
      state->retention_position = manifest.retention_position;
      state->rope_delta = manifest.rope_delta;
      state->prompt_mrope_positions =
          std::move(manifest.prompt_mrope_positions);
      state->sequence_target_hidden =
          std::move(manifest.sequence_target_hidden);
      state->ple_history = std::move(manifest.ple_history);
      state->ple_retention_history =
          std::move(manifest.ple_retention_history);
      state->draft_valid = manifest.draft_valid;
      state->retention_valid = manifest.retention_valid;
      state->exact_decode_enabled = manifest.exact_decode_enabled;
      state->host_kv_pages = std::move(host_pages);
      state->host_kv_populated_tokens =
          manifest.host_kv_populated_tokens;
      state->host_kv_bytes = expected_host_bytes;
      state->target_mirror_enabled = false;
      state->parked_state.emplace(std::move(parked));
      host_kv_bytes_ += expected_host_bytes;
      parked_request_bytes_ += manifest.parked_bytes;
      parked_session_bytes_ += manifest.reported_bytes;
    }
    return {er::Status::success(), generation, manifest.logical_pages,
            manifest.logical_data_bytes, 0U};
  } catch (const std::exception& error) {
    for (auto& page : host_pages)
      if (page.allocation) static_cast<void>(cudaFreeHost(page.allocation));
    return {{er::ErrorCode::internal, error.what()}, 0U, 0U, 0U, 0U};
  }
}

er::Status DenseFp4Provider::prune_request_state_snapshots(
    const std::filesystem::path& root, std::uint64_t generation) {
  try {
    if (generation == 0U || root.empty())
      throw std::runtime_error("snapshot prune generation is invalid");
    const auto manifest_path = snapshot_manifest_path(root, generation);
    const auto manifest = decode_snapshot_manifest(
        read_file(manifest_path, 64U << 20U));
    if (manifest.generation != generation ||
        !er::constant_time_equal(manifest.model_hash,
                                 descriptor_.content_hash))
      throw std::runtime_error("snapshot prune identity mismatch");
    std::unordered_set<std::string> live;
    live.reserve(manifest.chunks.size());
    for (const auto& chunk : manifest.chunks)
      live.insert(digest_hex(chunk.digest) + ".blob");
    std::error_code error;
    for (const auto& item : std::filesystem::directory_iterator(root)) {
      const auto name = item.path().filename().string();
      if (item.is_regular_file() &&
          name.starts_with("manifest-") && item.path() != manifest_path)
        std::filesystem::remove(item.path(), error);
      if (name.find(".partial-") != std::string::npos)
        std::filesystem::remove_all(item.path(), error);
    }
    const auto blobs = root / "blobs";
    if (std::filesystem::exists(blobs))
      for (const auto& item : std::filesystem::directory_iterator(blobs)) {
        const auto name = item.path().filename().string();
        if ((item.is_regular_file() && !live.contains(name)) ||
            name.find(".partial-") != std::string::npos)
          std::filesystem::remove(item.path(), error);
      }
    return er::Status::success();
  } catch (const std::exception& error) {
    return {er::ErrorCode::internal, error.what()};
  }
}

std::optional<er::OperationExecutionResult>
DenseFp4Provider::poll_program_sequence(
    const std::shared_ptr<SequenceState>& sequence) {
  if (!sequence || sequence->terminal) return std::nullopt;
  if (sequence->cancelled.load()) {
    std::lock_guard lock(mutex_);
    release_staged_dense_weights();
    sequence->terminal = true;
    sequence->hidden.clear();
    return er::OperationExecutionResult{
        {er::ErrorCode::cancelled, "dense FP4 program-sequence was cancelled"},
        {}};
  }
  try {
    std::lock_guard lock(mutex_);
    if (sequence->next_operation >= sequence->operations.size())
      throw std::runtime_error("program-sequence advanced beyond its program");
    const auto& operation =
        *sequence->operations[sequence->next_operation];
    const auto total_rows = sequence->tokens.size();
    if (!sequence->tile_rows || sequence->tile_first > total_rows ||
        sequence->tile_rows > total_rows - sequence->tile_first)
      throw std::runtime_error("program-sequence tile state is invalid");
    auto* tile_hidden =
        sequence_hidden_ +
        static_cast<std::size_t>(sequence->request->slot()) *
            sequence_tile_rows_ * hidden_size_;
    auto* tile_hyper = hyper_enabled_
        ? sequence_hyper_ +
              static_cast<std::size_t>(sequence->request->slot()) *
                  sequence_tile_rows_ * hyper_width_
        : nullptr;
    auto* tile_injection = hyper_enabled_
        ? sequence_injection_ +
              static_cast<std::size_t>(sequence->request->slot()) *
                  sequence_tile_rows_ * hyper_count_
        : nullptr;

    if (operation.kernel == Kernel::head) {
      if (sequence->next_row != 0U || total_rows == 0U ||
          sequence->next_operation + 1U != sequence->operations.size())
        throw std::runtime_error("program-sequence head state is invalid");
      const auto tile_bytes = sequence->tile_rows * hidden_size_ *
                              sizeof(float);
      if (sequence->request->exact_decode_enabled) {
        cuda_check(cudaMemcpy(
                       sequence->hidden.data() +
                           sequence->tile_first * hidden_size_,
                       tile_hidden, tile_bytes, cudaMemcpyDeviceToHost),
                   "retain exact target hidden tile");
        program_sequence_host_bytes_ += tile_bytes;
      }
      if (sequence->retention_position != 0U) {
        const auto checkpoint_row = static_cast<std::size_t>(
            sequence->retention_position - sequence->positions.front() - 1U);
        if (checkpoint_row >= sequence->tile_first &&
            checkpoint_row < sequence->tile_first + sequence->tile_rows) {
          cuda_check(cudaMemcpy(
                         slot_retention_last_hidden_ +
                             static_cast<std::size_t>(
                                 sequence->request->slot()) *
                                 hidden_size_,
                         tile_hidden +
                             (checkpoint_row - sequence->tile_first) *
                                 hidden_size_,
                         hidden_size_ * sizeof(float),
                         cudaMemcpyDeviceToDevice),
                     "retain program-sequence checkpoint hidden state");
          sequence->request->retention_position =
              sequence->retention_position;
        }
      }
      ++program_sequence_tiles_;
      if (sequence->tile_first + sequence->tile_rows != total_rows) {
        sequence->tile_first += sequence->tile_rows;
        sequence->tile_rows = std::min<std::size_t>(
            sequence_tile_rows_, total_rows - sequence->tile_first);
        sequence->next_operation = 0U;
        sequence->next_row = 0U;
        return std::nullopt;
      }
      const auto phase_event = begin_gpu_phase(GpuPhase::head);
      std::vector<std::uint32_t> predictions;
      if (sequence->retention_position != 0U &&
          sequence->retention_position <= sequence->positions.back()) {
        cuda_check(cudaMemcpy(
                       hidden_,
                       slot_retention_last_hidden_ +
                           static_cast<std::size_t>(
                               sequence->request->slot()) * hidden_size_,
                       hidden_size_ * sizeof(float),
                       cudaMemcpyDeviceToDevice),
                   "load retained program-sequence hidden state");
        auto retained = run_head(
            operation, 1U, &sequence->generation, sequence->request.get(),
            sequence->retention_position - 1U, true);
        if (retained.size() != 1U)
          throw std::runtime_error(
              "program-sequence retained head returned invalid width");
        predictions.push_back(retained.front());
      }
      cuda_check(cudaMemcpy(
                     hidden_,
                     tile_hidden + (sequence->tile_rows - 1U) * hidden_size_,
                     hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                 "load final program-sequence hidden state");
      auto terminal = run_head(
          operation, 1U, &sequence->generation, sequence->request.get(),
          sequence->positions.back(), true);
      if (terminal.size() != 1U)
        throw std::runtime_error(
            "program-sequence terminal head returned invalid width");
      predictions.push_back(terminal.front());
      auto* retained_hidden = slot_target_hidden_batch_ +
                              static_cast<std::size_t>(
                                  sequence->request->slot()) *
                                  workspace_rows_ * hidden_size_;
      cuda_check(cudaMemcpy(retained_hidden, hidden_,
                            hidden_size_ * sizeof(float),
                            cudaMemcpyDeviceToDevice),
                 "retain final program-sequence target hidden state");
      cuda_check(cudaMemcpy(slot_last_hidden(sequence->request->slot()),
                            hidden_, hidden_size_ * sizeof(float),
                            cudaMemcpyDeviceToDevice),
                 "retain final program-sequence target state");
      end_gpu_phase(phase_event);
      collect_gpu_phases();

      sequence->request->current_batch_first =
          sequence->positions.front();
      sequence->request->current_batch_rows =
          static_cast<std::uint32_t>(total_rows);
      sequence->request->current_position = sequence->positions.back();
      if (sequence->request->exact_decode_enabled) {
        sequence->request->synchronization_first =
            sequence->positions.front();
        sequence->request->synchronization_rows =
            static_cast<std::uint32_t>(total_rows);
        sequence->request->synchronization_consumed = 0U;
        sequence->request->sequence_target_hidden =
            std::move(sequence->hidden);
      } else {
        sequence->request->synchronization_first = 0U;
        sequence->request->synchronization_rows = 0U;
        sequence->request->synchronization_consumed = 0U;
        sequence->request->sequence_target_hidden.clear();
        sequence->request->mtp_length = sequence->request->current_position + 1U;
        sequence->request->draft_valid = false;
        sequence->request->draft_predictions.clear();
      }

      auto owner = std::make_shared<std::vector<std::uint32_t>>(
          std::move(predictions));
      er::OperationExecutionResult result;
      result.status = er::Status::success();
      result.outputs.push_back(
          {std::string(kTokenAbi), "host", owner,
           reinterpret_cast<const std::byte*>(owner->data()),
           owner->size() * sizeof((*owner)[0])});
      program_steps_ += total_rows;
      ++prefill_batches_;
      prefill_tokens_ += total_rows;
      ++program_sequence_batches_;
      program_sequence_tokens_ += total_rows;
      release_staged_dense_weights();
      if (mtp_q8_kv_) release_sequence_workspace();
      sequence->terminal = true;
      return result;
    }

    if (operation.kernel == Kernel::vision) {
      if (sequence->next_row != 0U)
        throw std::runtime_error("program-sequence vision state is invalid");
      const auto phase_event = begin_gpu_phase(GpuPhase::vision);
      run_vision(operation, sequence->media, tile_hidden,
                 static_cast<std::uint32_t>(total_rows),
                 static_cast<std::uint32_t>(sequence->tile_first),
                 static_cast<std::uint32_t>(sequence->tile_rows),
                 !sequence->vision_prepared);
      sequence->vision_prepared = true;
      end_gpu_phase(phase_event);
      collect_gpu_phases();
      ++sequence->next_operation;
      return std::nullopt;
    }

    auto chunk_rows = std::min<std::size_t>(
        workspace_rows_, sequence->tile_rows - sequence->next_row);
    if (sequence->retention_position != 0U) {
      const auto checkpoint_offset = static_cast<std::size_t>(
          sequence->retention_position - sequence->positions.front());
      const auto global_row = sequence->tile_first + sequence->next_row;
      if (global_row < checkpoint_offset &&
          checkpoint_offset < global_row + chunk_rows)
        chunk_rows = checkpoint_offset - global_row;
    }
    const auto rows = static_cast<std::uint32_t>(chunk_rows);
    if (!rows)
      throw std::runtime_error("program-sequence operation has no rows");
    const auto tile_offset = sequence->next_row;
    const auto offset = sequence->tile_first + tile_offset;
    const auto hidden_bytes = static_cast<std::size_t>(rows) * hidden_size_ *
                              sizeof(float);
    auto phase_event = begin_gpu_phase(gpu_phase(operation.kernel));
    bool phase_ended = false;
    std::uint32_t operation_advance = 1U;
    const auto stage_candidate =
        rows > 8U && staged_dense_weight_capacity_bytes_ != 0U &&
        operation.kernel != Kernel::embedding &&
        operation.kernel != Kernel::routed_moe;
    if (stage_candidate && sequence->next_row == 0U)
      static_cast<void>(sequence_staging_has_headroom(*sequence, operation));
    auto staged = false;
    if (stage_candidate && !sequence->staging_disabled) {
      staged = activate_staged_weights(operation);
      if (!staged) {
        sequence->staging_disabled = true;
        ++staged_dense_weight_low_memory_fallbacks_;
      }
    }
    if (operation.kernel == Kernel::embedding) {
      run_embedding(operation, sequence->tokens.data() + offset, rows);
      cuda_check(cudaMemcpy(
                     tile_hidden + tile_offset * hidden_size_, hidden_,
                     hidden_bytes, cudaMemcpyDeviceToDevice),
                 "store program-sequence embedding tile");
      program_sequence_device_bytes_ += hidden_bytes;
    } else if (operation.kernel == Kernel::hyper_initialize) {
      if (!tile_hyper)
        throw std::runtime_error("program-sequence Hyper tile is absent");
      cuda_check(cudaMemcpy(hidden_,
                            tile_hidden + tile_offset * hidden_size_,
                            hidden_bytes, cudaMemcpyDeviceToDevice),
                 "load program-sequence Hyper initialization tile");
      status_check(ec::hyper_repeat_batch(
          hidden_, hyper_, rows, hidden_size_, hyper_count_, nullptr));
      const auto hyper_bytes = static_cast<std::size_t>(rows) * hyper_width_ *
                               sizeof(float);
      cuda_check(cudaMemcpy(tile_hyper + tile_offset * hyper_width_, hyper_,
                            hyper_bytes, cudaMemcpyDeviceToDevice),
                 "store program-sequence initialized Hyper tile");
      program_sequence_device_bytes_ += hidden_bytes + hyper_bytes;
    } else if (operation.kernel == Kernel::ple) {
      if (!tile_hyper)
        throw std::runtime_error("program-sequence PLE Hyper tile is absent");
      const auto hyper_bytes = static_cast<std::size_t>(rows) * hyper_width_ *
                               sizeof(float);
      cuda_check(cudaMemcpy(hyper_, tile_hyper + tile_offset * hyper_width_,
                            hyper_bytes, cudaMemcpyDeviceToDevice),
                 "load program-sequence PLE Hyper tile");
      run_ple(operation, *sequence->request,
              std::span(sequence->tokens).subspan(offset, rows), rows);
      cuda_check(cudaMemcpy(tile_hyper + tile_offset * hyper_width_, hyper_,
                            hyper_bytes, cudaMemcpyDeviceToDevice),
                 "store program-sequence PLE Hyper tile");
      program_sequence_device_bytes_ += 2U * hyper_bytes;
      if (sequence->retention_position != 0U &&
          offset + rows == static_cast<std::size_t>(
                               sequence->retention_position -
                               sequence->positions.front())) {
        cuda_check(cudaMemcpy(
                       ple_conv_retention_checkpoint_[operation.ple_slot] +
                           static_cast<std::size_t>(
                               sequence->request->slot()) *
                               ple_conv_state_values_,
                       ple_conv_state_[operation.ple_slot] +
                           static_cast<std::size_t>(
                               sequence->request->slot()) *
                               ple_conv_state_values_,
                       ple_conv_state_values_ * sizeof(float),
                       cudaMemcpyDeviceToDevice),
                   "checkpoint layer-major PLE convolution state");
        const auto history_values = ple_ngram_size_ - 1U;
        const auto history_offset = static_cast<std::size_t>(
            operation.ple_slot) * history_values;
        std::copy_n(sequence->request->ple_history.data() + history_offset,
                    history_values,
                    sequence->request->ple_retention_history.data() +
                        history_offset);
      }
    } else if (operation.kernel == Kernel::hyper_read ||
               operation.kernel == Kernel::hyper_reduce) {
      if (!tile_hyper ||
          (operation.kernel == Kernel::hyper_read && !tile_injection))
        throw std::runtime_error("program-sequence Hyper read tile is absent");
      const auto hyper_bytes = static_cast<std::size_t>(rows) * hyper_width_ *
                               sizeof(float);
      cuda_check(cudaMemcpy(hyper_, tile_hyper + tile_offset * hyper_width_,
                            hyper_bytes, cudaMemcpyDeviceToDevice),
                 "load program-sequence Hyper read tile");
      const auto reduce = operation.kernel == Kernel::hyper_reduce;
      run_hyper_read(operation, rows, reduce);
      cuda_check(cudaMemcpy(tile_hidden + tile_offset * hidden_size_, hidden_,
                            hidden_bytes, cudaMemcpyDeviceToDevice),
                 "store program-sequence Hyper read hidden tile");
      program_sequence_device_bytes_ += hyper_bytes + hidden_bytes;
      if (!reduce) {
        const auto injection_bytes = static_cast<std::size_t>(rows) *
                                     hyper_count_ * sizeof(float);
        cuda_check(cudaMemcpy(
                       tile_injection + tile_offset * hyper_count_,
                       hyper_injection_, injection_bytes,
                       cudaMemcpyDeviceToDevice),
                   "store program-sequence Hyper injection tile");
        program_sequence_device_bytes_ += injection_bytes;
      }
    } else if (operation.kernel == Kernel::hyper_inject) {
      if (!tile_hyper || !tile_injection)
        throw std::runtime_error("program-sequence Hyper injection tile is absent");
      const auto hyper_bytes = static_cast<std::size_t>(rows) * hyper_width_ *
                               sizeof(float);
      const auto injection_bytes = static_cast<std::size_t>(rows) *
                                   hyper_count_ * sizeof(float);
      cuda_check(cudaMemcpy(hyper_, tile_hyper + tile_offset * hyper_width_,
                            hyper_bytes, cudaMemcpyDeviceToDevice),
                 "load retained program-sequence Hyper tile");
      cuda_check(cudaMemcpy(hidden_,
                            tile_hidden + tile_offset * hidden_size_,
                            hidden_bytes, cudaMemcpyDeviceToDevice),
                 "load program-sequence Hyper block output");
      cuda_check(cudaMemcpy(
                     hyper_injection_,
                     tile_injection + tile_offset * hyper_count_,
                     injection_bytes, cudaMemcpyDeviceToDevice),
                 "load program-sequence Hyper injection coefficients");
      status_check(ec::hyper_inject_batch(
          hyper_, hidden_, hyper_injection_, hyper_, rows, hidden_size_,
          hyper_count_, nullptr));
      cuda_check(cudaMemcpy(tile_hyper + tile_offset * hyper_width_, hyper_,
                            hyper_bytes, cudaMemcpyDeviceToDevice),
                 "store injected program-sequence Hyper tile");
      program_sequence_device_bytes_ +=
          2U * hyper_bytes + hidden_bytes + injection_bytes;
    } else {
      cuda_check(cudaMemcpy(hidden_,
                            tile_hidden + tile_offset * hidden_size_,
                            hidden_bytes, cudaMemcpyDeviceToDevice),
                 "load program-sequence hidden tile");
      switch (operation.kernel) {
        case Kernel::hyper_initialize:
        case Kernel::ple:
        case Kernel::hyper_read:
        case Kernel::hyper_inject:
        case Kernel::hyper_reduce:
          throw std::runtime_error(
              "Hyper/PLE operation entered the hidden-only sequence path");
        case Kernel::full_attention:
        case Kernel::recurrent_attention: {
          const auto no_residual =
              operation.capability ==
                  "block.sparse-attention.qsa.output-gated.v1" ||
              operation.capability ==
                  "block.recurrent-linear-attention.split-gated-delta.no-residual.v1";
          if (!no_residual) {
            cuda_check(cudaMemcpy(residual_, hidden_,
                                  static_cast<std::size_t>(rows) * hidden_size_ *
                                      sizeof(float),
                                  cudaMemcpyDeviceToDevice),
                       "retain program-sequence attention residual");
            if (operation.kernel == Kernel::full_attention)
              normalize_operation_input(operation, hidden_, normalized_, rows);
            else
              normalize_rows(hidden_, binding(operation, "input_norm").f32,
                             normalized_, rows);
          }
          const auto positions = std::span(sequence->positions)
                                     .subspan(offset, rows);
          if (operation.kernel == Kernel::full_attention) {
            std::array<std::uint32_t, kMaximumWorkspaceRows> adjusted{};
            auto rotary = positions;
            if (sequence->media.positions_thw.empty() &&
                sequence->request->rope_delta != 0) {
              for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto value = static_cast<std::int64_t>(positions[row]) +
                                   sequence->request->rope_delta;
                if (value < 0 ||
                    value > std::numeric_limits<std::uint32_t>::max())
                  throw std::runtime_error(
                      "adjusted sequence rotary position is invalid");
                adjusted[row] = static_cast<std::uint32_t>(value);
              }
              rotary = std::span(adjusted).first(rows);
            }
            run_full_attention(
                operation, *sequence->request, positions, rotary, rows,
                operation.full_attention_slot,
                sequence->media.positions_thw.empty()
                    ? std::span<const std::uint32_t>{}
                    : std::span(sequence->media.positions_thw)
                          .subspan(3U * offset, 3U * rows));
            if (operation.attention_window_tokens != 0U &&
                sequence->retention_position != 0U &&
                offset + rows == static_cast<std::size_t>(
                                     sequence->retention_position -
                                     sequence->positions.front()))
              checkpoint_window_layer(sequence->request->slot(),
                                      operation.kv_layer_slot);
          } else
            run_recurrent_attention(operation, sequence->request->slot(), rows,
                                    RecurrentCheckpointMode::none);
          if (operation.kernel == Kernel::recurrent_attention &&
              sequence->retention_position != 0U &&
              offset + rows == static_cast<std::size_t>(
                                   sequence->retention_position -
                                   sequence->positions.front())) {
            const auto conv_values = recurrent_conv_values_;
            const auto matrix_values = recurrent_matrix_values_;
            cuda_check(cudaMemcpy(
                           recurrent_conv_retention_checkpoint_[
                               operation.recurrent_slot] +
                               static_cast<std::size_t>(
                                   sequence->request->slot()) *
                                   conv_values,
                           recurrent_conv(operation.recurrent_slot,
                                          sequence->request->slot()),
                           conv_values * sizeof(float),
                           cudaMemcpyDeviceToDevice),
                       "checkpoint layer-major recurrent convolution state");
            cuda_check(cudaMemcpy(
                           recurrent_matrix_retention_checkpoint_[
                               operation.recurrent_slot] +
                               static_cast<std::size_t>(
                                   sequence->request->slot()) *
                                   matrix_values,
                           recurrent_matrix(operation.recurrent_slot,
                                            sequence->request->slot()),
                           matrix_values * sizeof(float),
                           cudaMemcpyDeviceToDevice),
                       "checkpoint layer-major recurrent matrix state");
          }
          if (no_residual)
            cuda_check(cudaMemcpy(hidden_, residual_, hidden_bytes,
                                  cudaMemcpyDeviceToDevice),
                       "commit program-sequence no-residual attention output");
          else if (operation.kernel == Kernel::full_attention)
            finish_attention_block(operation, rows);
          else
            status_check(ec::add_in_place(hidden_, residual_,
                                          rows * hidden_size_, nullptr,
                                          activation_bf16_));
          break;
        }
        case Kernel::ffn:
          run_ffn(operation, rows);
          break;
        case Kernel::router: {
          if (sequence->next_operation + 1U >= sequence->operations.size())
            throw std::runtime_error(
                "program-sequence router has no routed consumer");
          const auto& routed =
              *sequence->operations[sequence->next_operation + 1U];
          if (routed.kernel != Kernel::routed_moe ||
              routed.logical_layer != operation.logical_layer ||
              routed.component_layer != operation.component_layer)
            throw std::runtime_error(
                "program-sequence routed pair is inconsistent");
          run_router(operation, rows);
          end_gpu_phase(phase_event);
          const auto routed_event = begin_gpu_phase(GpuPhase::routed_moe);
          run_routed_moe(routed, rows);
          end_gpu_phase(routed_event);
          phase_ended = true;
          operation_advance = 2U;
          break;
        }
        case Kernel::routed_moe:
          throw std::runtime_error(
              "program-sequence routed operation was not fused with router");
        case Kernel::embedding:
        case Kernel::vision:
        case Kernel::head:
        case Kernel::exact_decode:
          throw std::runtime_error(
              "invalid operation in program-sequence hidden phase");
      }
      cuda_check(cudaMemcpy(
                     tile_hidden + tile_offset * hidden_size_, hidden_,
                     hidden_bytes, cudaMemcpyDeviceToDevice),
                 "store program-sequence hidden tile");
      program_sequence_device_bytes_ += 2U * hidden_bytes;
    }
    if (staged) deactivate_staged_weights();
    if (!phase_ended) end_gpu_phase(phase_event);
    collect_gpu_phases();
    sequence->next_row += rows;
    if (sequence->next_row == sequence->tile_rows) {
      sequence->next_row = 0U;
      sequence->next_operation += operation_advance;
    }
    return std::nullopt;
  } catch (const std::exception& error) {
    release_staged_dense_weights();
    sequence->terminal = true;
    sequence->hidden.clear();
    active_gpu_events_ = 0U;
    return er::OperationExecutionResult{
        {er::ErrorCode::internal, error.what()}, {}};
  }
}

er::OperationExecutionHandle DenseFp4Provider::execute(
    const er::IPreparedOperation& opaque_operation,
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const er::OperationInvocation& invocation) {
  try {
    std::lock_guard lock(mutex_);
    const auto* operation =
        dynamic_cast<const PreparedOperation*>(&opaque_operation);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!operation || !state || operation->kernel == Kernel::exact_decode)
      throw std::runtime_error("dense FP4 invocation state is invalid");
    std::map<std::string, er::ExecutionValue, std::less<>> outputs;
    const auto phase_event = begin_gpu_phase(gpu_phase(operation->kernel));
    switch (operation->kernel) {
      case Kernel::embedding: {
        state->exact_decode_enabled =
            exact_ &&
            request_parameter(invocation.request, "exact_decode_enabled") !=
                0U;
        if (state->synchronization_rows != 0U)
          throw std::runtime_error(
              "previous target batch was not synchronized");
        const auto& input = invocation_input(*operation, invocation, "token_ids");
        const auto tokens =
            host_u32_batch(input, kTokenAbi, "embedding token batch");
        for (std::uint32_t row = 0U; row < tokens.size(); ++row) {
          if (tokens[row] >= vocabulary_size_)
            throw std::runtime_error("embedding token exceeds vocabulary");
        }
        run_embedding(*operation, tokens.data(),
                      static_cast<std::uint32_t>(tokens.size()));
        state->current_batch_rows = static_cast<std::uint32_t>(tokens.size());
        outputs.emplace("hidden", device_hidden_value(state->current_batch_rows));
        break;
      }
      case Kernel::hyper_initialize: {
        const auto rows = require_hidden_value(
            invocation_input(*operation, invocation, "hidden"));
        if (rows != state->current_batch_rows)
          throw std::runtime_error("Hyper batch width changed in program");
        status_check(ec::hyper_repeat_batch(
            hidden_, hyper_, rows, hidden_size_, hyper_count_, nullptr));
        outputs.emplace(
            "hyper", device_value(
                         kHyperAbi, hyper_,
                         static_cast<std::uint64_t>(rows) * hyper_width_ *
                             sizeof(float)));
        break;
      }
      case Kernel::ple: {
        const auto hyper_bytes =
            static_cast<std::uint64_t>(state->current_batch_rows) *
            hyper_width_ * sizeof(float);
        require_device_value(
            invocation_input(*operation, invocation, "hyper"), kHyperAbi,
            hyper_, hyper_bytes);
        const auto tokens = host_u32_batch(
            invocation_input(*operation, invocation, "token_ids"),
            kTokenAbi, "PLE token batch");
        if (tokens.size() != state->current_batch_rows)
          throw std::runtime_error("PLE token batch width changed");
        run_ple(*operation, *state, tokens, state->current_batch_rows);
        outputs.emplace("hyper",
                        device_value(kHyperAbi, hyper_, hyper_bytes));
        break;
      }
      case Kernel::hyper_read:
      case Kernel::hyper_reduce: {
        const auto rows = state->current_batch_rows;
        const auto hyper_bytes = static_cast<std::uint64_t>(rows) *
                                 hyper_width_ * sizeof(float);
        require_device_value(
            invocation_input(*operation, invocation, "hyper"), kHyperAbi,
            hyper_, hyper_bytes);
        const auto reduce = operation->kernel == Kernel::hyper_reduce;
        run_hyper_read(*operation, rows, reduce);
        outputs.emplace("hidden", device_hidden_value(rows));
        if (!reduce) {
          outputs.emplace("retained",
                          device_value(kHyperAbi, hyper_, hyper_bytes));
          outputs.emplace(
              "injection",
              device_value(kInjectionAbi, hyper_injection_,
                           static_cast<std::uint64_t>(rows) * hyper_count_ *
                               sizeof(float)));
        }
        break;
      }
      case Kernel::hyper_inject: {
        const auto rows = state->current_batch_rows;
        const auto hyper_bytes = static_cast<std::uint64_t>(rows) *
                                 hyper_width_ * sizeof(float);
        require_device_value(
            invocation_input(*operation, invocation, "retained"), kHyperAbi,
            hyper_, hyper_bytes);
        static_cast<void>(require_hidden_value(
            invocation_input(*operation, invocation, "hidden")));
        require_device_value(
            invocation_input(*operation, invocation, "injection"),
            kInjectionAbi, hyper_injection_,
            static_cast<std::uint64_t>(rows) * hyper_count_ * sizeof(float));
        status_check(ec::hyper_inject_batch(
            hyper_, hidden_, hyper_injection_, hyper_, rows, hidden_size_,
            hyper_count_, nullptr));
        outputs.emplace("hyper",
                        device_value(kHyperAbi, hyper_, hyper_bytes));
        break;
      }
      case Kernel::vision: {
        const auto rows = require_hidden_value(
            invocation_input(*operation, invocation, "hidden"));
        if (rows != state->current_batch_rows)
          throw std::runtime_error("vision batch width changed in program");
        const auto media = parse_media_payload(
            invocation_input(*operation, invocation, "media"), 0U,
            vision_patch_dimension_, vision_spatial_merge_size_);
        if (!media.empty())
          throw std::runtime_error(
              "image input requires program-sequence prefill");
        outputs.emplace("hidden", device_hidden_value(rows));
        break;
      }
      case Kernel::full_attention:
      case Kernel::recurrent_attention: {
        const auto rows = require_hidden_value(
            invocation_input(*operation, invocation, "hidden"));
        const auto& input =
            invocation_input(*operation, invocation, "positions");
        const auto positions =
            host_u32_batch(input, kPositionAbi, "attention position batch");
        if (positions.size() != rows || rows != state->current_batch_rows)
          throw std::runtime_error("attention batch width changed in program");
        for (std::uint32_t row = 0U; row < rows; ++row) {
          if (positions[row] >= max_context_ ||
              (row != 0U && positions[row] != positions[0] + row))
            throw std::runtime_error(
                "attention position batch is not contiguous");
        }
        state->current_batch_first = positions.front();
        state->current_position = positions.back();
        const auto no_residual =
            operation->capability ==
                "block.sparse-attention.qsa.output-gated.v1" ||
            operation->capability ==
                "block.recurrent-linear-attention.split-gated-delta.no-residual.v1";
        if (!no_residual) {
          cuda_check(cudaMemcpy(residual_, hidden_,
                                static_cast<std::size_t>(rows) * hidden_size_ *
                                    sizeof(float),
                                cudaMemcpyDeviceToDevice),
                     "retain attention residual batch");
          if (operation->kernel == Kernel::full_attention)
            normalize_operation_input(*operation, hidden_, normalized_, rows);
          else
            normalize_rows(hidden_, binding(*operation, "input_norm").f32,
                           normalized_, rows);
        }
        if (operation->kernel == Kernel::full_attention) {
          std::array<std::uint32_t, kMaximumWorkspaceRows> rotary{};
          for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto adjusted = static_cast<std::int64_t>(positions[row]) +
                                  state->rope_delta;
            if (adjusted < 0 ||
                adjusted > std::numeric_limits<std::uint32_t>::max())
              throw std::runtime_error("adjusted rotary position is invalid");
            rotary[row] = static_cast<std::uint32_t>(adjusted);
          }
          run_full_attention(*operation, *state, positions,
                             std::span(rotary).first(rows), rows,
                             operation->full_attention_slot);
        } else {
          run_recurrent_attention(*operation, state->slot(), rows,
                                  RecurrentCheckpointMode::none);
        }
        if (no_residual)
          cuda_check(cudaMemcpy(hidden_, residual_,
                                static_cast<std::size_t>(rows) * hidden_size_ *
                                    sizeof(float),
                                cudaMemcpyDeviceToDevice),
                     "commit no-residual attention output");
        else if (operation->kernel == Kernel::full_attention)
          finish_attention_block(*operation, rows);
        else
          status_check(ec::add_in_place(
              hidden_, residual_, rows * hidden_size_, nullptr,
              activation_bf16_));
        outputs.emplace("hidden", device_hidden_value(rows));
        break;
      }
      case Kernel::router: {
        const auto rows = require_hidden_value(
            invocation_input(*operation, invocation, "hidden"));
        if (rows != state->current_batch_rows)
          throw std::runtime_error("router batch width changed in program");
        run_router(*operation, rows);
        const auto hidden_bytes = static_cast<std::uint64_t>(rows) *
                                  hidden_size_ * sizeof(float);
        outputs.emplace("expert_input",
                        device_value(kHiddenAbi, normalized_, hidden_bytes));
        outputs.emplace(
            "route_indices",
            device_value(kRouteIndexAbi, routing_indices_,
                         static_cast<std::uint64_t>(rows) * route_width_ *
                             sizeof(std::uint32_t)));
        outputs.emplace(
            "route_weights",
            device_value(kRouteWeightAbi, routing_scores_,
                         static_cast<std::uint64_t>(rows) * route_width_ *
                             sizeof(float)));
        if (operation->capability !=
            "router.linear-topk.shared-swiglu.no-residual.v1")
          outputs.emplace("residual",
                          device_value(kHiddenAbi, hidden_, hidden_bytes));
        outputs.emplace("shared_output",
                        device_value(kHiddenAbi, shared_output_, hidden_bytes));
        break;
      }
      case Kernel::routed_moe: {
        const auto rows = state->current_batch_rows;
        const auto hidden_bytes = static_cast<std::uint64_t>(rows) *
                                  hidden_size_ * sizeof(float);
        require_device_value(
            invocation_input(*operation, invocation, "expert_input"),
            kHiddenAbi, normalized_, hidden_bytes);
        require_device_value(
            invocation_input(*operation, invocation, "route_indices"),
            kRouteIndexAbi, routing_indices_,
            static_cast<std::uint64_t>(rows) * route_width_ *
                sizeof(std::uint32_t));
        require_device_value(
            invocation_input(*operation, invocation, "route_weights"),
            kRouteWeightAbi, routing_scores_,
            static_cast<std::uint64_t>(rows) * route_width_ * sizeof(float));
        if (operation->capability !=
            "moe.swiglu.routed.merge-shared.no-residual.v1")
          require_device_value(
              invocation_input(*operation, invocation, "residual"),
              kHiddenAbi, hidden_, hidden_bytes);
        require_device_value(
            invocation_input(*operation, invocation, "shared_output"),
            kHiddenAbi, shared_output_, hidden_bytes);
        run_routed_moe(*operation, rows);
        outputs.emplace("hidden", device_hidden_value(rows));
        break;
      }
      case Kernel::ffn: {
        const auto rows = require_hidden_value(
            invocation_input(*operation, invocation, "hidden"));
        if (rows != state->current_batch_rows)
          throw std::runtime_error("FFN batch width changed in program");
        run_ffn(*operation, rows);
        outputs.emplace("hidden", device_hidden_value(rows));
        break;
      }
      case Kernel::head: {
        const auto rows = require_hidden_value(
            invocation_input(*operation, invocation, "hidden"));
        if (rows != state->current_batch_rows)
          throw std::runtime_error("head batch width changed in program");
        auto* retained_hidden =
            slot_target_hidden_batch_ +
            static_cast<std::size_t>(state->slot()) * workspace_rows_ *
                hidden_size_;
        cuda_check(cudaMemcpy(retained_hidden, hidden_,
                              static_cast<std::size_t>(rows) * hidden_size_ *
                                  sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "retain final target hidden batch");
        cuda_check(cudaMemcpy(
                       slot_last_hidden(state->slot()),
                       retained_hidden +
                           static_cast<std::size_t>(rows - 1U) * hidden_size_,
                       hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                   "retain final target hidden state");
        if (state->exact_decode_enabled) {
          state->synchronization_first = state->current_batch_first;
          state->synchronization_rows = rows;
          state->synchronization_consumed = 0U;
        } else {
          state->synchronization_first = 0U;
          state->synchronization_rows = 0U;
          state->synchronization_consumed = 0U;
          state->mtp_length = state->current_position + 1U;
          state->draft_valid = false;
          state->draft_predictions.clear();
        }
        auto owner =
            std::make_shared<std::vector<std::uint32_t>>(
                run_head(*operation, rows, &invocation.request, state.get(),
                         state->current_position, rows > 1U));
        outputs.emplace(
            "token_ids",
            er::ExecutionValue{std::string(kTokenAbi), "host", owner,
                               reinterpret_cast<const std::byte*>(
                                   owner->data()),
                               owner->size() * sizeof((*owner)[0])});
        program_steps_ += rows;
        if (rows > 1U) {
          ++prefill_batches_;
          prefill_tokens_ += rows;
        }
        break;
      }
      case Kernel::exact_decode:
        throw std::runtime_error("exact decode entered scalar execution");
    }
    end_gpu_phase(phase_event);
    if (operation->kernel == Kernel::head) collect_gpu_phases();
    er::OperationExecutionResult result;
    result.status = er::Status::success();
    result.outputs.reserve(operation->outputs.size());
    for (const auto& [port, abi] : operation->outputs) {
      auto found = outputs.find(port);
      if (found == outputs.end() || found->second.abi != abi)
        throw std::runtime_error("dense FP4 output ABI mismatch");
      result.outputs.push_back(std::move(found->second));
    }
    return completed_operation(std::move(result));
  } catch (const std::exception& error) {
    return completed_operation(
        {{er::ErrorCode::internal, error.what()}, {}});
  }
}

er::Status DenseFp4Provider::synchronize_exact_decode(
    const er::IPreparedOperation& opaque_operation,
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const er::ExactDecodeSynchronization& synchronization) {
  const std::array tokens{synchronization.next_token};
  return synchronize_exact_decode_batch(
      opaque_operation, opaque_state,
      er::ExactDecodeSynchronizationBatch{
          synchronization.request, tokens, synchronization.target_position,
          synchronization.produce_draft});
}

er::Status DenseFp4Provider::synchronize_exact_decode_batch(
    const er::IPreparedOperation& opaque_operation,
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const er::ExactDecodeSynchronizationBatch& synchronization) {
  try {
    std::lock_guard lock(mutex_);
    const auto* operation =
        dynamic_cast<const PreparedOperation*>(&opaque_operation);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!operation || operation != exact_.get() || !state ||
        synchronization.next_tokens.empty() ||
        synchronization.next_tokens.size() > workspace_rows_ ||
        state->synchronization_consumed > state->synchronization_rows ||
        synchronization.next_tokens.size() >
            state->synchronization_rows - state->synchronization_consumed ||
        std::any_of(synchronization.next_tokens.begin(),
                    synchronization.next_tokens.end(),
                    [this](std::uint32_t token) {
                      return token >= vocabulary_size_;
                    }) ||
        state->mtp_length != synchronization.first_target_position ||
        state->synchronization_consumed >= state->synchronization_rows ||
        synchronization.first_target_position !=
            state->synchronization_first +
                state->synchronization_consumed)
      throw std::runtime_error("MTP synchronization stream is invalid");
    const auto rows =
        static_cast<std::uint32_t>(synchronization.next_tokens.size());
    const auto synchronized_row = state->synchronization_consumed;
    auto* retained_hidden =
        slot_target_hidden_batch_ +
        static_cast<std::size_t>(state->slot()) * workspace_rows_ *
            hidden_size_;
    const float* previous_hidden{};
    if (!state->sequence_target_hidden.empty()) {
      if (state->sequence_target_hidden.size() !=
          static_cast<std::size_t>(state->synchronization_rows) * hidden_size_)
        throw std::runtime_error(
            "program-sequence target hidden state is inconsistent");
      cuda_check(cudaMemcpy(
                     retained_hidden,
                     state->sequence_target_hidden.data() +
                         static_cast<std::size_t>(synchronized_row) *
                             hidden_size_,
                     static_cast<std::size_t>(rows) * hidden_size_ *
                         sizeof(float),
                     cudaMemcpyHostToDevice),
                 "upload program-sequence target hidden chunk");
      previous_hidden = retained_hidden;
    } else {
      previous_hidden =
          retained_hidden +
          static_cast<std::size_t>(synchronized_row) * hidden_size_;
    }
    const auto* final_hidden =
        previous_hidden + static_cast<std::size_t>(rows - 1U) * hidden_size_;
    cuda_check(cudaMemcpy(slot_last_hidden(state->slot()), final_hidden,
                          hidden_size_ * sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "commit synchronized target hidden state");
    state->synchronization_consumed += rows;
    state->draft_valid = false;
    state->draft_predictions.clear();
    state->synchronized_token = synchronization.next_tokens.back();

    const auto mtp_rows = synchronization.first_target_position + 1U >=
                                  max_context_
                              ? 0U
                              : std::min<std::uint32_t>(
                                    rows, max_context_ -
                                              synchronization
                                                  .first_target_position -
                                              1U);
    if (mtp_rows != 0U) {
      std::array<std::uint32_t, kMaximumWorkspaceRows> positions{};
      for (std::uint32_t row = 0U; row < mtp_rows; ++row)
        positions[row] =
            synchronization.first_target_position + row + 1U;
      const auto produce_draft = synchronization.produce_final_draft &&
                                 mtp_rows == rows;
      if (produce_draft && exact_decode_abi_ >= 2U)
        cuda_check(cudaMemcpy(
                       sampling_proposal_presence_,
                       sampling_presence_ +
                           static_cast<std::size_t>(state->slot()) *
                               vocabulary_size_,
                       vocabulary_size_, cudaMemcpyDeviceToDevice),
                   "initialize speculative proposal presence");
      const auto sample_position =
          synchronization.first_target_position + mtp_rows;
      auto draft = run_mtp(
          *state, synchronization.next_tokens.first(mtp_rows), previous_hidden,
          std::span(positions).first(mtp_rows), produce_draft,
          produce_draft && exact_decode_abi_ >= 2U
              ? &synchronization.request
              : nullptr,
          produce_draft && exact_decode_abi_ >= 2U
              ? sampling_proposal_presence_
              : nullptr,
          sample_position, 1U);
      if (produce_draft) {
        if (!draft)
          throw std::runtime_error("MTP synchronization produced no draft");
        if (exact_decode_abi_ >= 2U) {
          const auto context_limit = static_cast<std::uint32_t>(
              request_parameter(synchronization.request,
                                "reserved_context_tokens"));
          extend_mtp_rollout(*state, synchronization.request,
                             std::move(*draft), context_limit);
        } else {
          state->draft_token = draft->token;
          state->draft_valid = true;
        }
      }
    }
    ++exact_sync_batches_;
    exact_sync_tokens_ += rows;
    if (state->synchronization_consumed == state->synchronization_rows) {
      state->synchronization_rows = state->synchronization_consumed = 0U;
      state->sequence_target_hidden.clear();
      state->sequence_target_hidden.shrink_to_fit();
    }
    return er::Status::success();
  } catch (const std::exception& error) {
    return {er::ErrorCode::internal, error.what()};
  }
}

er::ExactDecodeExecutionHandle DenseFp4Provider::execute_exact_decode(
    const er::IPreparedOperation& opaque_operation,
    const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
    const er::ExactDecodeInvocation& invocation) {
  try {
    std::lock_guard lock(mutex_);
    const auto* operation =
        dynamic_cast<const PreparedOperation*>(&opaque_operation);
    const auto state = std::dynamic_pointer_cast<RequestState>(opaque_state);
    if (!operation || operation != exact_.get() || !state ||
        !state->draft_valid ||
        invocation.guaranteed_token != state->synchronized_token ||
        invocation.position != state->mtp_length ||
        invocation.position + 1U >= invocation.context_limit ||
        invocation.context_limit > max_context_)
      throw std::runtime_error("exact MTP invocation stream is invalid");

    if (exact_decode_abi_ >= 2U) {
      if (state->draft_predictions.empty() ||
          state->draft_predictions.size() > draft_depth_ ||
          state->draft_predictions.size() + 1U >
              kMaximumExactDecodeRows ||
          invocation.position + state->draft_predictions.size() >=
              invocation.context_limit)
        throw std::runtime_error("multi-draft MTP state is invalid");
      const auto proposal_count = state->draft_predictions.size();
      std::vector<std::uint32_t> target_tokens;
      std::vector<std::uint32_t> target_positions;
      target_tokens.reserve(proposal_count + 1U);
      target_positions.reserve(proposal_count + 1U);
      target_tokens.push_back(invocation.guaranteed_token);
      target_positions.push_back(invocation.position);
      for (std::size_t index = 0U; index < proposal_count; ++index) {
        const auto& proposal = state->draft_predictions[index];
        if (!(proposal.distribution.probability(proposal.token) > 0.0))
          throw std::runtime_error("MTP proposal has zero draft mass");
        target_tokens.push_back(proposal.token);
        target_positions.push_back(
            invocation.position + static_cast<std::uint32_t>(index) + 1U);
      }

      static_cast<void>(
          run_target(*state, target_tokens, target_positions,
                     RecurrentCheckpointMode::every_row));
      cuda_check(cudaMemcpy(
                     sampling_proposal_presence_,
                     sampling_presence_ +
                         static_cast<std::size_t>(state->slot()) *
                             vocabulary_size_,
                     vocabulary_size_, cudaMemcpyDeviceToDevice),
                 "initialize target verification presence");
      std::vector<er::SamplingDistribution> target_distributions;
      target_distributions.reserve(proposal_count + 1U);
      for (std::size_t row = 0U; row <= proposal_count; ++row) {
        target_distributions.push_back(distribution_from_logits(
            logits_ + row * vocabulary_size_, vocabulary_size_,
            invocation.request, sampling_proposal_presence_));
        if (row < proposal_count)
          cuda_check(cudaMemset(
                         sampling_proposal_presence_ +
                             state->draft_predictions[row].token,
                         1, 1),
                     "advance target verification presence");
      }

      const auto seed =
          request_parameter(invocation.request, "sampling_seed");
      std::size_t accepted{};
      std::uint32_t next_token{};
      bool rejected{};
      for (; accepted < proposal_count; ++accepted) {
        const auto& proposal = state->draft_predictions[accepted];
        const auto decision = er::rejection_sample(
            proposal.token, target_distributions[accepted],
            proposal.distribution,
            er::counter_uniform(
                seed,
                invocation.position + static_cast<std::uint32_t>(accepted),
                100U, static_cast<std::uint32_t>(accepted)),
            er::counter_uniform(
                seed,
                invocation.position + static_cast<std::uint32_t>(accepted),
                200U, static_cast<std::uint32_t>(accepted)));
        if (!decision.accepted) {
          next_token = decision.token;
          rejected = true;
          break;
        }
      }
      if (!rejected)
        next_token = er::sample_distribution(
            target_distributions.back(),
            er::counter_uniform(
                seed,
                invocation.position + static_cast<std::uint32_t>(accepted),
                300U));

      for (std::size_t index = 0U; index < accepted; ++index)
        trace_sampling(
            invocation.request,
            invocation.position + static_cast<std::uint32_t>(index),
            state->draft_predictions[index].token, "mtp_accepted",
            &target_distributions[index],
            &state->draft_predictions[index].distribution);
      trace_sampling(
          invocation.request,
          invocation.position + static_cast<std::uint32_t>(accepted),
          next_token, rejected ? "mtp_residual" : "mtp_bonus",
          &target_distributions[accepted],
          rejected ? &state->draft_predictions[accepted].distribution
                   : nullptr);

      if (rejected) {
        restore_speculative_recurrent_checkpoint(
            state->slot(), static_cast<std::uint32_t>(accepted));
      }
      cuda_check(cudaMemcpy(
                     slot_last_hidden(state->slot()),
                     hidden_ + accepted * hidden_size_,
                     hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                 "commit sampled target hidden state");

      auto* emitted_presence = sampling_presence_ +
          static_cast<std::size_t>(state->slot()) * vocabulary_size_;
      for (std::size_t index = 0U; index < accepted; ++index)
        cuda_check(cudaMemset(
                       emitted_presence +
                           state->draft_predictions[index].token,
                       1, 1),
                   "commit accepted proposal presence");
      cuda_check(cudaMemset(emitted_presence + next_token, 1, 1),
                 "commit sampled next-token presence");

      std::vector<std::uint32_t> mtp_tokens;
      std::vector<std::uint32_t> mtp_positions;
      mtp_tokens.reserve(accepted + 1U);
      mtp_positions.reserve(accepted + 1U);
      for (std::size_t index = 0U; index < accepted; ++index)
        mtp_tokens.push_back(state->draft_predictions[index].token);
      mtp_tokens.push_back(next_token);
      for (std::size_t index = 0U; index < mtp_tokens.size(); ++index)
        mtp_positions.push_back(
            invocation.position + static_cast<std::uint32_t>(index) + 1U);
      const auto positions_advanced =
          static_cast<std::uint32_t>(accepted + 1U);
      const auto future_exact =
          invocation.position + positions_advanced + 1U <
          invocation.context_limit;
      state->mtp_length = invocation.position;
      state->draft_valid = false;
      state->draft_predictions.clear();
      cuda_check(cudaMemcpy(sampling_proposal_presence_, emitted_presence,
                            vocabulary_size_, cudaMemcpyDeviceToDevice),
                 "initialize next MTP rollout presence");
      auto first = run_mtp(
          *state, mtp_tokens, hidden_, mtp_positions, future_exact,
          future_exact ? &invocation.request : nullptr,
          future_exact ? sampling_proposal_presence_ : nullptr,
          invocation.position + positions_advanced, 1U);
      if (future_exact) {
        if (!first)
          throw std::runtime_error("next MTP rollout produced no root");
        extend_mtp_rollout(*state, invocation.request, std::move(*first),
                           invocation.context_limit);
      }
      state->synchronized_token = next_token;

      er::ExactDecodeExecutionResult result;
      result.status = er::Status::success();
      result.emitted_tokens.push_back(invocation.guaranteed_token);
      for (std::size_t index = 0U; index < accepted; ++index)
        result.emitted_tokens.push_back(target_tokens[index + 1U]);
      result.next_token = next_token;
      result.positions_advanced = positions_advanced;
      ++exact_calls_;
      accepted_drafts_ += accepted;
      ++accepted_depth_calls_.at(accepted);
      program_steps_ += positions_advanced;
      return completed_exact(std::move(result));
    }

    const std::array target_tokens{invocation.guaranteed_token,
                                   state->draft_token};
    const std::array target_positions{invocation.position,
                                      invocation.position + 1U};
    const auto target_outputs =
        run_target(*state, target_tokens, target_positions,
                   RecurrentCheckpointMode::after_first);
    if (target_outputs.size() != 2U)
      throw std::runtime_error("two-position target verification is incomplete");
    const bool accepted = target_outputs[0] == state->draft_token;
    const auto positions_advanced = accepted ? 2U : 1U;
    const auto next_token = accepted ? target_outputs[1] : target_outputs[0];
    if (!accepted) restore_recurrent_checkpoint(state->slot());
    const auto selected_row = accepted ? 1U : 0U;
    cuda_check(cudaMemcpy(slot_last_hidden(state->slot()),
                          hidden_ + static_cast<std::size_t>(selected_row) *
                                        hidden_size_,
                          hidden_size_ * sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "commit exact target hidden state");

    state->draft_valid = false;
    state->draft_predictions.clear();
    const auto future_exact =
        invocation.position + positions_advanced + 1U <
        invocation.context_limit;
    if (accepted && invocation.position + 2U < max_context_) {
      const std::array mtp_tokens{state->draft_token, target_outputs[1]};
      const std::array mtp_positions{invocation.position + 1U,
                                     invocation.position + 2U};
      auto drafts = run_mtp(*state, mtp_tokens, hidden_, mtp_positions,
                            future_exact);
      if (future_exact) {
        if (!drafts)
          throw std::runtime_error("legacy MTP produced no draft");
        state->draft_token = drafts->token;
        state->draft_valid = true;
      }
    } else if (!accepted) {
      const std::array mtp_tokens{next_token};
      const std::array mtp_positions{invocation.position + 1U};
      auto drafts = run_mtp(*state, mtp_tokens, hidden_, mtp_positions,
                            future_exact && !accepted);
      if (future_exact) {
        if (!drafts)
          throw std::runtime_error("legacy MTP produced no draft");
        state->draft_token = drafts->token;
        state->draft_valid = true;
      }
    } else {
      // The accepted token advances the shifted MTP cache. The following
      // guaranteed token would be outside the model's maximum context.
      const std::array mtp_tokens{target_tokens[1]};
      const std::array mtp_positions{invocation.position + 1U};
      static_cast<void>(
          run_mtp(*state, mtp_tokens, hidden_, mtp_positions, false));
    }
    state->synchronized_token = next_token;

    er::ExactDecodeExecutionResult result;
    result.status = er::Status::success();
    result.emitted_tokens.push_back(invocation.guaranteed_token);
    if (accepted) result.emitted_tokens.push_back(target_tokens[1]);
    result.next_token = next_token;
    result.positions_advanced = positions_advanced;
    ++exact_calls_;
    accepted_drafts_ += accepted ? 1U : 0U;
    ++accepted_depth_calls_.at(accepted ? 1U : 0U);
    program_steps_ += positions_advanced;
    return completed_exact(std::move(result));
  } catch (const std::exception& error) {
    return completed_exact(
        {{er::ErrorCode::internal, error.what()}, {}, 0U, 0U});
  }
}

}  // namespace

er::WorkerProviderDefinition make_sm86_dense_fp4_provider() {
  return {"sm86-dense-fp4", 200U, provider_capabilities(), nullptr};
}

er::CreateExecutionProviderModuleResult make_sm86_dense_fp4_callable_provider(
    const std::filesystem::path& artifact_root, std::uint32_t max_context,
    std::uint32_t capacity, std::uint64_t ram_cache_bytes,
    std::uint64_t vram_cache_bytes, std::uint64_t kv_cache_bytes,
    std::uint32_t kv_page_tokens, std::string_view kv_cache_dtype,
    std::string_view placement_profile, std::string_view routed_vram_policy,
    bool profile_gpu_phases, bool discover_active_expert_devices,
    std::vector<int> active_expert_devices,
    std::uint64_t active_expert_device_cache_bytes,
    std::uint64_t active_expert_host_cache_bytes) {
  try {
    if (discover_active_expert_devices)
      active_expert_devices = ec::discover_pascal_active_expert_devices();
    if (active_expert_devices.empty()) {
      active_expert_device_cache_bytes = 0U;
      active_expert_host_cache_bytes = 0U;
    }
    auto artifact = std::make_shared<er::ModelArtifact>();
    auto status = er::ModelArtifact::load(artifact_root, *artifact);
    if (!status.ok()) return {status, {}};
    auto tensor_store = std::make_shared<er::MappedModelTensorStore>();
    status = er::MappedModelTensorStore::create(*artifact, *tensor_store);
    if (!status.ok()) return {status, {}};
    const auto routed = !artifact->model().routed_components.empty();
    if (routed && placement_profile != "latency" &&
        placement_profile != "balanced" &&
        placement_profile != "capacity")
      throw std::runtime_error("unsupported routed FP4 placement profile");
    auto implementation = std::make_shared<DenseFp4Provider>(
        artifact, tensor_store, max_context, capacity, ram_cache_bytes,
        vram_cache_bytes, kv_cache_bytes, kv_page_tokens, kv_cache_dtype,
        placement_profile, routed_vram_policy, profile_gpu_phases,
        std::move(active_expert_devices),
        active_expert_device_cache_bytes,
        active_expert_host_cache_bytes);
    er::ExecutionProviderModule module;
    module.definition = {"sm86-dense-fp4", 200U, provider_capabilities(),
                         implementation};
    module.tensor_store = std::move(tensor_store);
    const auto mtp = artifact->model().exact_decode_program.has_value();
    module.service = {
        "causal_layer_major",
        implementation->prefill_batch_rows(),
        implementation->supports_request_state_retention(),
        "per_request_nonblocking",
        "artifact",
        std::string(implementation->target_kv_dtype()),
        "paged_on_demand",
        implementation->kv_page_tokens(),
        implementation->kv_page_bytes(),
        implementation->kv_page_capacity(),
        routed ? "budgeted" : "resident",
        routed ? std::string(placement_profile) : "resident",
        routed ? implementation->routed_ram_cache_bytes() : 0U,
        vram_cache_bytes,
        false,
        "disabled",
        0U,
        mtp,
        mtp,
        mtp,
        false,
        false,
        true};
    module.service.session_parking =
        implementation->supports_request_state_parking() &&
        implementation->parking_ram_capacity_bytes() != 0U;
    module.service.session_persistence =
        module.service.session_parking &&
        implementation->supports_request_state_persistence();
    module.service.session_park_ram_bytes =
        module.service.session_parking
            ? implementation->parking_ram_capacity_bytes()
            : 0U;
    module.service.session_park_page_capacity =
        module.service.session_parking
            ? implementation->parking_ram_capacity_bytes() /
                  implementation->kv_page_bytes()
            : 0U;
    module.service.routed_vram_policy = std::string(routed_vram_policy);
    module.telemetry = [implementation] { return implementation->telemetry(); };
    module.finalize_service = [implementation](auto& service) {
      auto status = implementation->finalize_startup();
      if (status.ok())
        service.vram_cache_bytes =
            implementation->effective_vram_cache_bytes();
      return status;
    };
    return {er::Status::success(), std::move(module)};
  } catch (const std::exception& error) {
    return {{er::ErrorCode::internal, error.what()}, {}};
  }
}
