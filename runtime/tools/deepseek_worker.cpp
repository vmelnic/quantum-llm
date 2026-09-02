#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "expert/core/json.hpp"
#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_decode.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/deepseek_request.hpp"
#include "expert/runtime/cuda/deepseek_mtp_request.hpp"
#include "expert/runtime/cuda/deepseek_scheduler.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cpu/deepseek_packed_executor.hpp"
#include "expert/runtime/deepseek_artifacts.hpp"
#include "expert/runtime/deepseek_catalog.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/execution_provider.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/model_descriptor.hpp"
#include "expert/runtime/program_executor.hpp"
#include "expert/runtime/resource_governor.hpp"
#include "expert/runtime/resident_expert_set.hpp"
#include "expert/runtime/route_census.hpp"
#include "expert/runtime/routed_expert_runtime.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"
#include "expert/runtime/worker_contract.hpp"
#include "expert/runtime/worker_provider.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <future>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace er = expert::runtime;

namespace {

er::Status validate_compressed_sparse_provider(
    const er::ModelDescriptor& candidate) {
  const auto decoder = std::max_element(
      candidate.routed_components.begin(), candidate.routed_components.end(),
      [](const auto& left, const auto& right) {
        return left.layer_count < right.layer_count;
      });
  if (decoder == candidate.routed_components.end() ||
      candidate.hidden_size != 4096U ||
      decoder->layer_count != candidate.layer_program.size() ||
      decoder->experts_per_layer != 256U || decoder->route_width != 6U ||
      decoder->intermediate_size != 2048U)
    return {er::ErrorCode::invalid_argument,
            "compressed sparse SM86 geometry is unsupported"};
  for (std::size_t layer = 0U; layer < candidate.layer_program.size(); ++layer) {
    const auto& instruction = candidate.layer_program[layer];
    const auto ratio = instruction.parameters.find("compression_ratio");
    if (instruction.logical_layer != layer ||
        instruction.block_capability !=
            "block.compressed-sparse-attention.hca.v1" ||
        instruction.component_layer >= decoder->layer_count ||
        ratio == instruction.parameters.end() ||
        (ratio->second != 0U && ratio->second != 4U &&
         ratio->second != 128U))
      return {er::ErrorCode::invalid_argument,
              "compressed sparse layer program is unsupported"};
  }
  return er::Status::success();
}

std::vector<er::KernelCapability> provider_capabilities() {
  return {
      {"embedding.lookup.int8-row.v1", 1U, 1U,
       validate_compressed_sparse_provider},
      {"block.compressed-sparse-attention.hca.v1", 1U, 1U,
       validate_compressed_sparse_provider},
      {"router.deepseek.v4.topk.v1", 1U, 1U},
      {"moe.swiglu.routed.v1", 1U, 1U},
      {"head.rmsnorm.argmax.int8-row.v1", 1U, 1U},
      {"decode.speculative.verify.v1", 1U, 1U}};
}

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

template <typename T>
class PinnedBuffer final {
 public:
  explicit PinnedBuffer(std::size_t count) : count_(count) {
    require(count_ != 0U, "pinned buffer cannot be empty");
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&data_),
                             count_ * sizeof(T), cudaHostAllocPortable),
               "allocate DeepSeek sequence host frontier");
  }
  ~PinnedBuffer() {
    if (data_) static_cast<void>(cudaFreeHost(data_));
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] T& operator[](std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] const T& operator[](std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  T* data_{};
  std::size_t count_{};
};

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> result;
  for (;;) {
    const auto separator = line.find('\t');
    result.push_back(line.substr(0U, separator));
    if (separator == std::string_view::npos) return result;
    line.remove_prefix(separator + 1U);
  }
}

std::vector<std::uint32_t> parse_tokens(std::string_view text,
                                        std::uint32_t vocabulary_size) {
  std::vector<std::uint32_t> result;
  for (;;) {
    const auto separator = text.find(',');
    const auto value = std::stoull(std::string(text.substr(0U, separator)));
    require(value < vocabulary_size, "token is outside model vocabulary");
    result.push_back(static_cast<std::uint32_t>(value));
    if (separator == std::string_view::npos) break;
    text.remove_prefix(separator + 1U);
  }
  require(!result.empty(), "empty token sequence");
  return result;
}

std::uint8_t nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::runtime_error("invalid worker model hash");
}

er::Sha256Digest digest(std::string_view text) {
  require(text.size() == 64U, "worker model hash has the wrong length");
  er::Sha256Digest result{};
  for (std::size_t index = 0U; index < result.size(); ++index)
    result[index] = static_cast<std::byte>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  return result;
}

struct Bundle final {
  std::filesystem::path checkpoint;
  std::filesystem::path dense;
  std::filesystem::path typed;
  std::filesystem::path shared;
  std::filesystem::path routed;
  std::filesystem::path census;
  std::filesystem::path mtp;
  std::filesystem::path mtp_routed;
  er::Sha256Digest model_hash{};
  er::ModelDescriptor descriptor;
  er::CompiledModelProgram program;
  double cpu_ns{};
  double gpu_ns{};
  double h2d_bytes_per_second{};
  bool mtp_available{};
  bool mtp_runtime_ready{};
};

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open model manifest");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::uint32_t manifest_u32(
    const expert::core::json::Value::Object& object, std::string_view key,
    std::uint32_t fallback) {
  const auto found = object.find(key);
  if (found == object.end()) return fallback;
  const auto value = found->second.AsU64(key);
  require(value <= std::numeric_limits<std::uint32_t>::max(),
          "model manifest integer exceeds uint32");
  return static_cast<std::uint32_t>(value);
}

std::uint64_t manifest_u64(
    const expert::core::json::Value::Object& object, std::string_view key,
    std::uint64_t fallback) {
  const auto found = object.find(key);
  return found == object.end() ? fallback : found->second.AsU64(key);
}

er::ModelDescriptor adapt_deepseek_manifest(
    const std::filesystem::path& root, const er::Sha256Digest&,
    std::uint64_t main_namespace, bool has_mtp) {
  using expert::core::json::Required;
  const auto document = expert::core::json::Parse(read_text(root / "manifest.json"));
  const auto& manifest = document.AsObject("manifest");
  const auto& format =
      Required(manifest, "format", "manifest").AsObject("format");
  require(Required(format, "name", "format").AsString("format.name") ==
              "deepseek-worker-bundle",
          "worker bundle manifest has an unsupported format");
  const auto& architecture =
      Required(manifest, "architecture", "manifest").AsObject("architecture");
  const auto& integrity =
      Required(manifest, "integrity", "manifest").AsObject("integrity");
  const auto artifact_hash = digest(
      Required(integrity, "content_sha256", "integrity")
          .AsString("integrity.content_sha256"));
  const auto model_program = manifest.find("model_program");
  require(model_program != manifest.end(),
          "worker bundle requires an immutable runtime model program");
  const auto& metadata = model_program->second.AsObject("model_program");
    expert::core::json::RequireExactKeys(
        metadata, {"format", "path", "bytes", "sha256"}, {},
        "model_program");
    require(Required(metadata, "format", "model_program")
                    .AsString("model_program.format") ==
                "expert-runtime-model-v1",
            "worker bundle has an unsupported model program format");
    const std::filesystem::path relative =
        Required(metadata, "path", "model_program")
            .AsString("model_program.path");
    const auto escapes = std::any_of(
        relative.begin(), relative.end(),
        [](const auto& part) { return part == ".."; });
    require(!relative.empty() && !relative.is_absolute() && !escapes,
            "worker bundle model program path escapes its root");
    auto parsed = er::load_model_descriptor_artifact(
        root / relative,
        Required(metadata, "bytes", "model_program")
            .AsU64("model_program.bytes"),
        digest(Required(metadata, "sha256", "model_program")
                   .AsString("model_program.sha256")),
        artifact_hash, main_namespace);
    require(parsed.status.ok(), parsed.status.message());
    const auto& descriptor = parsed.descriptor;
    const auto* decoder = er::find_routed_component(descriptor, "decoder");
    const auto* draft = er::find_routed_component(descriptor, "draft");
    require(
        descriptor.architecture_id ==
                Required(architecture, "model_type", "architecture")
                    .AsString("architecture.model_type") &&
            descriptor.hidden_size ==
                manifest_u32(architecture, "hidden_size", 0U) &&
            descriptor.vocab_size ==
                manifest_u32(architecture, "vocab_size", 0U) &&
            descriptor.max_context_tokens == manifest_u32(
                architecture, "max_position_embeddings", 0U) &&
            decoder != nullptr && decoder->namespace_id == main_namespace &&
            decoder->layer_count == manifest_u32(architecture, "layers", 0U) &&
            decoder->experts_per_layer ==
                manifest_u32(architecture, "experts_per_layer", 0U) &&
            decoder->route_width ==
                manifest_u32(architecture, "active_experts", 0U) &&
            decoder->intermediate_size == manifest_u32(
                architecture, "expert_intermediate_size", 0U) &&
            decoder->shared_experts_per_layer ==
                manifest_u32(architecture, "shared_experts_per_layer", 0U) &&
            decoder->hidden_size == descriptor.hidden_size &&
            decoder->source_abi == er::kExpertSourceAbiSplitFp4Block32V1 &&
            decoder->encoding_abi == er::kExpertEncodingAbiFp4Block32 &&
            decoder->encoding == "fp4.e2m1.ue8m0.block32" &&
            manifest_u64(architecture, "namespace_id", 0U) ==
                main_namespace &&
            ((has_mtp && draft != nullptr &&
              draft->namespace_id == main_namespace + 1U &&
              draft->layer_count == manifest_u32(
                  architecture, "multi_token_prediction_layers", 0U) &&
              draft->experts_per_layer == decoder->experts_per_layer &&
              draft->route_width == decoder->route_width &&
              draft->hidden_size == decoder->hidden_size &&
              draft->intermediate_size == decoder->intermediate_size &&
              draft->source_abi == decoder->source_abi &&
              draft->encoding_abi == decoder->encoding_abi &&
              draft->encoding == decoder->encoding &&
              manifest_u64(architecture, "mtp_namespace_id", 0U) ==
                  draft->namespace_id) ||
             (!has_mtp && draft == nullptr)),
        "runtime model program disagrees with worker bundle geometry");
  return std::move(parsed.descriptor);
}

const er::RoutedExpertComponentDescriptor& required_component(
    const er::ModelDescriptor& descriptor, std::string_view name) {
  const auto* component = er::find_routed_component(descriptor, name);
  require(component != nullptr, "model descriptor lacks a routed component");
  return *component;
}

er::ExpertCatalogConfig split_fp4_catalog_config(
    const std::filesystem::path& catalog_root,
    const std::filesystem::path& source_root,
    const er::RoutedExpertComponentDescriptor& component) {
  const auto elements = static_cast<std::uint64_t>(component.hidden_size) *
                        component.intermediate_size;
  require(component.encoding == "fp4.e2m1.ue8m0.block32" &&
              component.source_abi ==
                  er::kExpertSourceAbiSplitFp4Block32V1 &&
              component.encoding_abi == er::kExpertEncodingAbiFp4Block32 &&
              component.hidden_size % er::kExpertFp4BlockSize == 0U &&
              component.intermediate_size % er::kExpertFp4BlockSize == 0U,
          "artifact adapter cannot map routed component encoding");
  const auto stored = 3ULL * (elements / 2U +
                              elements / er::kExpertFp4BlockSize);
  return {catalog_root,
          source_root,
          component.layer_count,
          component.experts_per_layer,
          component.hidden_size,
          component.intermediate_size,
          er::kExpertFp4BlockSize,
          stored,
          3ULL * elements * sizeof(float),
          stored,
          component.source_abi,
          0U,
          0U,
          er::kExpertPackAlignment,
          {{"deepseek-routed-pack-extents-v1",
            "deepseek-routed-pack-catalog-v1", 1U, true},
           {"deepseek-routed-extents-v1", "deepseek-routed-catalog-v1", 6U,
            false}}};
}

std::filesystem::path bundle_path(const std::filesystem::path& root,
                                  const std::string& text) {
  std::filesystem::path path(text);
  if (path.is_absolute()) return path.lexically_normal();
  for (const auto& part : path)
    require(part != "..", "worker bundle path escapes its root");
  return (root / path).lexically_normal();
}

Bundle load_bundle(const std::filesystem::path& root) {
  std::ifstream input(root / "runtime.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              (line == "deepseek-worker-bundle-v1" ||
               line == "deepseek-worker-bundle-v2" ||
               line == "deepseek-worker-bundle-v3"),
          "invalid DeepSeek worker bundle header");
  const bool has_mtp = line != "deepseek-worker-bundle-v1";
  const bool has_mtp_runtime = line == "deepseek-worker-bundle-v3";
  std::map<std::string, std::string> values;
  while (std::getline(input, line)) {
    const auto fields = split_tabs(line);
    require(fields.size() == 2U && !fields[0].empty() &&
                values.emplace(std::string(fields[0]),
                               std::string(fields[1])).second,
            "invalid or duplicate DeepSeek worker bundle field");
  }
  const auto expected_fields = has_mtp_runtime ? 13U : has_mtp ? 12U : 11U;
  require(input.eof() && values.size() == expected_fields,
          "incomplete DeepSeek worker bundle");
  Bundle result;
  result.checkpoint = bundle_path(root, values.at("checkpoint"));
  result.dense = bundle_path(root, values.at("dense"));
  result.typed = bundle_path(root, values.at("typed"));
  result.shared = bundle_path(root, values.at("shared"));
  result.routed = bundle_path(root, values.at("routed"));
  result.census = bundle_path(root, values.at("census"));
  if (has_mtp) result.mtp = bundle_path(root, values.at("mtp"));
  if (has_mtp_runtime)
    result.mtp_routed = bundle_path(root, values.at("mtp_routed"));
  result.model_hash = digest(values.at("model_sha256"));
  const auto main_namespace = std::stoull(values.at("model_id"));
  require(main_namespace != 0U, "worker bundle namespace is zero");
  result.descriptor = adapt_deepseek_manifest(
      root, result.model_hash, main_namespace, has_mtp_runtime);
  er::ExecutionProviderRegistry provider_registry;
  auto registered = provider_registry.add(
      {"sm86-compressed-sparse-moe", 100U, provider_capabilities()});
  require(registered.ok(), registered.message());
  auto bound = provider_registry.bind(result.descriptor);
  require(bound.status.ok(), bound.status.message());
  result.program = std::move(bound.provider.program);
  result.cpu_ns = std::stod(values.at("cpu_ns_per_selection"));
  result.gpu_ns = std::stod(values.at("gpu_ns_per_selection"));
  result.h2d_bytes_per_second = std::stod(values.at("h2d_bytes_per_second"));
  result.mtp_available = has_mtp;
  result.mtp_runtime_ready = has_mtp_runtime;
  require(std::filesystem::is_directory(result.checkpoint) &&
              std::filesystem::is_directory(result.dense) &&
              std::filesystem::is_directory(result.typed) &&
              std::filesystem::is_directory(result.shared) &&
              std::filesystem::is_directory(result.routed) &&
              (!has_mtp ||
               (std::filesystem::is_directory(result.mtp) &&
                std::filesystem::is_regular_file(result.mtp / "manifest.json") &&
                std::filesystem::is_regular_file(
                    result.mtp / "routed" / "catalog.tsv"))) &&
              (!has_mtp_runtime ||
               (std::filesystem::is_directory(result.mtp_routed) &&
                std::filesystem::is_regular_file(
                    result.mtp_routed / "manifest.json") &&
                std::filesystem::is_regular_file(
                    result.mtp_routed / "catalog.tsv"))) &&
              std::isfinite(result.cpu_ns) && result.cpu_ns > 0.0 &&
              std::isfinite(result.gpu_ns) && result.gpu_ns > 0.0 &&
              std::isfinite(result.h2d_bytes_per_second) &&
              result.h2d_bytes_per_second > 0.0,
          "DeepSeek worker bundle dependency is unavailable");
  return result;
}

std::array<float, 32U> rope_values(std::uint32_t position, bool sine,
                                   bool compressed) {
  std::array<float, 32U> result{};
  constexpr double dimension = 64.0, factor = 16.0;
  constexpr double original_context = 65536.0;
  const double base = compressed ? 160000.0 : 10000.0;
  const auto correction = [&](double rotations) {
    return dimension *
           std::log(original_context /
                    (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(base));
  };
  const double low = std::clamp(std::floor(correction(32.0)), 0.0, 31.0);
  const double high = std::clamp(std::ceil(correction(1.0)), 0.0, 31.0);
  for (std::size_t index = 0U; index < result.size(); ++index) {
    double frequency = std::pow(base, -(2.0 * index) / dimension);
    if (compressed) {
      const double ramp = std::clamp(
          (static_cast<double>(index) - low) / std::max(high - low, 1e-3),
          0.0, 1.0);
      frequency = frequency / factor * ramp + frequency * (1.0 - ramp);
    }
    const double angle = static_cast<double>(position) * frequency;
    result[index] = static_cast<float>(sine ? std::sin(angle) : std::cos(angle));
  }
  return result;
}

std::array<std::array<float, 32U>, 8U> rope_row(std::uint32_t position) {
  std::array<std::array<float, 32U>, 8U> result{};
  result[0] = rope_values(position, false, false);
  result[1] = rope_values(position, true, false);
  result[2] = rope_values(position, false, true);
  result[3] = rope_values(position, true, true);
  const auto start4 = position + 1U >= 4U ? position + 1U - 4U : 0U;
  const auto start128 = position + 1U >= 128U ? position + 1U - 128U : 0U;
  result[4] = rope_values(start4, false, true);
  result[5] = rope_values(start4, true, true);
  result[6] = rope_values(start128, false, true);
  result[7] = rope_values(start128, true, true);
  return result;
}

struct PromptRouteEvidence final {
  std::uint32_t count{};
  std::uint64_t last_seen{};
};

struct RouteTraceStep final {
  std::uint32_t position{};
  std::uint32_t route_rows{};
  std::vector<er::cuda::DeepSeekRouteTraceEntry> routes;
};

struct Request final {
  std::shared_ptr<er::cuda::DeepSeekRequestState> state;
  std::shared_ptr<er::cuda::DeepSeekDecodeController> controller;
  std::shared_ptr<er::cuda::DeepSeekVerifyState> verify;
  std::shared_ptr<er::cuda::DeepSeekMtpRequestState> mtp;
  cudaStream_t stream{};
  er::cuda::DeepSeekDecodeTelemetry controller_telemetry;
  std::uint32_t predicted{};
  std::uint32_t draft{};
  std::uint32_t next_position{};
  std::uint32_t context_limit{};
  std::uint32_t slot{};
  bool draft_ready{};
  bool speculation_suppressed{};
  std::vector<std::map<std::uint32_t, PromptRouteEvidence>> prompt_routes;
  std::uint64_t prompt_route_clock{};
  std::vector<RouteTraceStep> route_trace_steps;
  std::uint64_t route_trace_dropped_steps{};
  ~Request() {
    controller.reset();
    verify.reset();
    mtp.reset();
    state.reset();
    if (stream) static_cast<void>(cudaStreamDestroy(stream));
  }
};

struct WorkerTelemetry final {
  std::uint64_t model_steps{};
  std::uint64_t model_rows{};
  std::uint64_t model_step_ns{};
  std::uint64_t embed_rope_submit_ns{};
  std::uint64_t scheduler_poll_ns{};
  std::uint64_t output_head_ns{};
  std::uint64_t attention_route_submit_ns{};
  std::uint64_t directory_plan_ns{};
  std::uint64_t ffn_submit_ns{};
  std::uint64_t directory_release_ns{};
  std::uint64_t callable_selection_launches{};
  std::uint64_t callable_selections{};
  std::uint64_t callable_overlap_launches{};
  std::uint64_t callable_cpu_launches{};
  std::uint64_t callable_cpu_selections{};
  std::uint64_t callable_cpu_activation_bytes{};
  std::uint64_t callable_cpu_output_bytes{};
  std::uint64_t callable_cpu_host_resolves_launched{};
  std::uint64_t callable_cpu_host_resolves_completed{};
  std::uint64_t callable_cpu_device_admission_fallbacks{};
  std::uint64_t callable_resolves_launched{};
  std::uint64_t callable_resolves_completed{};
  std::uint64_t callable_stale_plan_replans{};
  std::uint64_t callable_prefetch_routes_launched{};
  std::uint64_t callable_prefetch_routes_completed{};
  std::uint64_t callable_prefetch_prediction_wait_ns{};
  std::uint64_t callable_prefetch_candidates{};
  std::uint64_t callable_prefetch_vram_candidates{};
  std::uint64_t callable_prefetch_ram_candidates{};
  std::uint64_t callable_prefetch_storage_candidates{};
  std::uint64_t callable_prefetch_storage_selected{};
  std::uint64_t callable_prefetch_storage_incorrect{};
  std::uint64_t callable_prefetch_resolves_launched{};
  std::uint64_t callable_prefetch_resolves_completed{};
  std::uint64_t callable_prefetch_selected{};
  std::uint64_t callable_prefetch_useful{};
  std::uint64_t callable_prefetch_late{};
  std::uint64_t callable_prefetch_incorrect{};
  std::uint64_t callable_prefetch_cancelled{};
  std::uint64_t callable_prefetch_errors{};
  std::uint64_t callable_remote_resolves{};
  std::uint64_t callable_remote_selections_launched{};
  std::uint64_t callable_remote_selections_completed{};
  std::uint64_t callable_remote_errors{};
  std::uint64_t callable_remote_cancellations{};
  std::uint64_t callable_remote_activation_tx_bytes{};
  std::uint64_t callable_remote_activation_rx_bytes{};
  std::uint64_t callable_remote_weight_tx_bytes{};
  std::uint64_t callable_remote_owner_weight_read_bytes{};
  std::uint64_t callable_remote_owner_storage_read_bytes{};
  std::uint64_t callable_remote_owner_ram_read_bytes{};
  std::uint64_t callable_remote_owner_vram_read_bytes{};
  std::uint64_t callable_remote_owner_execution_ns{};
  std::uint64_t callable_remote_transport_wait_ns{};
  std::uint64_t gpu_attention_route_plan_ns{};
  std::uint64_t gpu_ffn_release_ns{};
  std::uint64_t gpu_attention_ns{};
  std::uint64_t gpu_route_ns{};
  std::uint64_t gpu_directory_plan_ns{};
  std::uint64_t gpu_ffn_ns{};
  std::uint64_t gpu_directory_release_ns{};
  std::uint64_t gpu_attention_hca_pre_norm_ns{};
  std::uint64_t gpu_attention_projection_ns{};
  std::uint64_t gpu_sparse_attention_ns{};
  std::uint64_t gpu_attention_output_projection_ns{};
  std::uint64_t gpu_attention_hca_post_ns{};
  std::uint64_t gpu_ffn_routed_ns{};
  std::uint64_t gpu_ffn_aggregate_ns{};
  std::uint64_t gpu_ffn_shared_ns{};
  std::uint64_t gpu_ffn_merge_ns{};
  std::uint64_t gpu_ffn_hca_post_ns{};
  std::uint64_t warm_start_candidates{};
  std::uint64_t warm_start_loaded{};
  std::uint64_t warm_start_failed{};
  std::uint64_t warm_start_cancelled{};
  std::uint64_t warm_start_demand_pauses{};
  std::uint64_t warm_start_inflight_max{};
  std::uint64_t warm_start_loop_errors{};
  std::uint64_t warm_start_bytes{};
  std::uint64_t warm_start_ns{};
  std::uint64_t census_namespace_rebinds{};
  std::uint64_t warm_vram_candidates{};
  std::uint64_t warm_vram_loaded{};
  std::uint64_t warm_vram_failed{};
  std::uint64_t warm_vram_cancelled{};
  std::uint64_t warm_vram_demand_pauses{};
  std::uint64_t warm_vram_inflight_max{};
  std::uint64_t warm_vram_bytes{};
  std::uint64_t warm_vram_ns{};
  std::uint64_t mtp_drafts{};
  std::uint64_t mtp_accepted{};
  std::uint64_t mtp_rejected{};
  std::uint64_t verify_pairs{};
  std::uint64_t useful_tokens{};
  std::uint64_t mtp_suppressions{};
  std::uint64_t mtp_acquire_batches{};
  std::uint64_t mtp_acquires_launched{};
  std::uint64_t mtp_acquire_batch_width_max{};
  std::uint64_t mtp_acquire_wait_ns{};
  std::uint64_t prefill_protection_candidates{};
  std::uint64_t prefill_protection_promoted{};
  std::uint64_t sequence_blocks{};
  std::uint64_t sequence_rows{};
  std::uint64_t sequence_layers{};
  std::uint64_t sequence_attention_hca_pre_norm_ns{};
  std::uint64_t sequence_attention_projection_ns{};
  std::uint64_t sequence_causal_attention_ns{};
  std::uint64_t sequence_attention_output_projection_ns{};
  std::uint64_t sequence_route_ns{};
  std::uint64_t sequence_expert_wait_ns{};
  std::uint64_t sequence_expert_execute_ns{};
  std::uint64_t sequence_finalize_ns{};
};

class Model final : public er::IOperationProvider,
                    public std::enable_shared_from_this<Model> {
 public:
  Model(const std::filesystem::path& root, std::uint32_t max_context,
        std::uint64_t ram_bytes, std::uint64_t vram_bytes,
        std::uint32_t capacity, std::uint64_t kv_cache_bytes,
        std::uint32_t kv_page_tokens, std::string placement,
        bool gpu_phase_timing, bool enable_mtp,
        bool enable_retained_route, bool enable_proactive_cpu_hybrid,
        bool enable_cpu_fallback,
        std::filesystem::path route_trace_path,
        std::size_t route_trace_max_steps,
        std::shared_ptr<const er::ActiveExpertOwnerDirectory> remote_owners =
            std::make_shared<const er::ActiveExpertOwnerDirectory>())
      : bundle_(load_bundle(root)), max_context_(max_context),
        capacity_(capacity), ram_bytes_(ram_bytes), vram_bytes_(vram_bytes),
        kv_cache_bytes_(kv_cache_bytes), kv_page_tokens_(kv_page_tokens),
        placement_(std::move(placement)), gpu_phase_timing_(gpu_phase_timing),
        mtp_enabled_(enable_mtp && bundle_.mtp_runtime_ready &&
                     bundle_.descriptor.exact_decode_program.has_value()),
        retained_route_enabled_(enable_retained_route),
        proactive_cpu_hybrid_enabled_(enable_proactive_cpu_hybrid),
        cpu_hybrid_enabled_(enable_proactive_cpu_hybrid ||
                            enable_cpu_fallback),
        route_trace_path_(std::move(route_trace_path)),
        route_trace_max_steps_(route_trace_max_steps),
        remote_owners_(std::move(remote_owners)) {
    require(max_context_ >= 2U && capacity_ != 0U && ram_bytes_ != 0U &&
                vram_bytes_ != 0U && kv_cache_bytes_ != 0U &&
                kv_page_tokens_ != 0U &&
                max_context_ <= bundle_.descriptor.max_context_tokens,
            "invalid DeepSeek worker limits");
    callable_provider_slots_.assign(capacity_, false);
    require(route_trace_path_.empty() || route_trace_max_steps_ != 0U,
            "DeepSeek route tracing requires a positive step bound");
    const auto& main_component =
        required_component(bundle_.descriptor, "decoder");
    compression_ratios_.reserve(main_component.layer_count);
    for (const auto& instruction : bundle_.descriptor.layer_program) {
      const auto ratio = instruction.parameters.find("compression_ratio");
      require(ratio != instruction.parameters.end() &&
                  instruction.logical_layer == compression_ratios_.size(),
              "runtime model has a non-canonical attention schedule");
      compression_ratios_.push_back(
          static_cast<std::uint32_t>(ratio->second));
    }
    const auto hash_layers = main_component.router.parameters.find(
        "hash_layers");
    require(hash_layers != main_component.router.parameters.end() &&
                hash_layers->second <= main_component.layer_count,
            "runtime model does not declare its router schedule");
    hash_router_layers_ = static_cast<std::uint32_t>(hash_layers->second);
    if (!route_trace_path_.empty()) {
      std::error_code error;
      const auto parent = route_trace_path_.parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent, error);
      require(!error, "cannot create DeepSeek route trace directory");
      route_trace_.open(route_trace_path_,
                        std::ios::out | std::ios::trunc);
      require(route_trace_.is_open(), "cannot open DeepSeek route trace");
    }
    auto loaded = er::load_deepseek_model_artifacts(
        bundle_.dense, bundle_.typed, bundle_.shared, bundle_.checkpoint,
        main_component.layer_count, main_component.namespace_id);
    require(loaded.status.ok(), loaded.status.message());
    artifacts_ = std::move(loaded.artifacts);
    const auto catalog_status = er::ExpertCatalog::load(
        split_fp4_catalog_config(bundle_.routed, bundle_.checkpoint,
                                 main_component),
        catalog_);
    require(catalog_status.ok(), catalog_status.message());
    if (mtp_enabled_) {
      const auto& draft_component =
          required_component(bundle_.descriptor, "draft");
      auto mtp_loaded = er::load_deepseek_tensor_artifacts(
          bundle_.mtp / "dense", bundle_.mtp / "typed-residency",
          bundle_.checkpoint, 7U, 19U);
      require(mtp_loaded.status.ok(), mtp_loaded.status.message());
      mtp_artifacts_ = std::move(mtp_loaded.artifacts);
      auto mtp_shared = er::load_deepseek_shared_artifacts(
          bundle_.mtp / "shared", bundle_.checkpoint,
          draft_component.layer_count, draft_component.namespace_id);
      require(mtp_shared.status.ok(), mtp_shared.status.message());
      mtp_shared_specs_ = std::move(mtp_shared.shared);
      const auto mtp_catalog_status = er::ExpertCatalog::load(
          split_fp4_catalog_config(bundle_.mtp_routed, bundle_.checkpoint,
                                   draft_component),
          mtp_catalog_);
      require(mtp_catalog_status.ok(), mtp_catalog_status.message());
      const auto mtp_request_size =
          er::cuda::deepseek_mtp_request_state_size(max_context_);
      require(mtp_request_size.status.ok(), mtp_request_size.status.message());
      mtp_request_bytes_ = mtp_request_size.total_bytes;
      mtp_cache_bytes_ = 512ULL << 20U;
    }
    require(!mtp_enabled_ || bundle_.mtp_runtime_ready,
            "MTP execution requires worker bundle v3 resources");
    // The MTP tier is instantiated and charged only when the selected serving
    // program can execute it. Merely shipping draft resources must not shrink
    // the exact scalar program's routed-expert budgets.
    const auto mtp_reserve_bytes =
        mtp_enabled_ ? mtp_cache_bytes_ : 0ULL;
    const auto request_size = er::cuda::deepseek_request_state_size(
        max_context_, compression_ratios_);
    require(request_size.status.ok(), request_size.status.message());
    request_bytes_ = request_size.total_bytes;
    if (mtp_enabled_) {
      const auto verify_size =
          er::cuda::deepseek_verify_state_size(max_context_,
                                                compression_ratios_);
      require(verify_size.status.ok(), verify_size.status.message());
      verify_request_bytes_ = verify_size.total_bytes;
    }
    const auto attention_per_token =
        (request_size.attention_bytes + max_context_ - 1U) / max_context_;
    require(attention_per_token <=
                std::numeric_limits<std::uint64_t>::max() / kv_page_tokens_,
            "DeepSeek KV page geometry overflow");
    kv_page_bytes_ = std::max<std::uint64_t>(
        1U, attention_per_token * kv_page_tokens_);
    kv_page_capacity_ = kv_cache_bytes_ / kv_page_bytes_;
    const auto pages_per_request =
        (max_context_ + kv_page_tokens_ - 1U) / kv_page_tokens_;
    require(kv_page_capacity_ >= pages_per_request * capacity_,
            "DeepSeek KV cache cannot reserve configured request capacity");
    const auto shared_bytes = std::accumulate(
        artifacts_.shared.begin(), artifacts_.shared.end(), std::uint64_t{0U},
        [](std::uint64_t total, const auto& item) {
          return total + (item.record.device_bytes == 0U
                              ? item.record.stored_bytes
                              : item.record.device_bytes);
        });
    const auto* representative = catalog_.find(0U, 0U);
    require(representative != nullptr && representative->stored_bytes != 0U,
            "routed catalog has no representative record");
    const auto routed_device_bytes = representative->device_bytes == 0U
                                         ? representative->stored_bytes
                                         : representative->device_bytes;
    require(vram_bytes_ >=
                shared_bytes +
                    (static_cast<std::uint64_t>(main_component.route_width) +
                     1U) *
                        routed_device_bytes +
                               mtp_reserve_bytes &&
                ram_bytes_ > mtp_reserve_bytes,
            "DeepSeek VRAM cache cannot hold shared plus one route");
    std::size_t free{}, total{};
    cuda_check(cudaMemGetInfo(&free, &total), "inspect DeepSeek worker VRAM");
    const auto fixed_without_sequence =
        artifacts_.dense_device_bytes + artifacts_.typed_source_bytes +
        request_bytes_ * capacity_ + rope_table_bytes() +
        mtp_artifacts_.dense_device_bytes +
        mtp_artifacts_.typed_source_bytes +
        mtp_request_bytes_ * capacity_ +
        verify_request_bytes_ * capacity_;
    constexpr std::uint64_t device_reserve_bytes = 1ULL << 30U;
    constexpr std::uint32_t sequence_row_quantum = 32U;
    const auto minimum_attention_size =
        er::cuda::deepseek_attention_batch_workspace_size(
            max_context_, sequence_row_quantum);
    const auto minimum_ffn_bytes =
        er::cuda::deepseek_ffn_batch_workspace_size(sequence_row_quantum);
    require(minimum_attention_size.status.ok() && minimum_ffn_bytes != 0U,
            "DeepSeek worker cannot size its minimum exact workspace");
    const auto minimum_sequence_workspace_bytes =
        minimum_attention_size.bytes + minimum_ffn_bytes +
        2ULL * sequence_row_quantum * 4U * 4096U * sizeof(float);
    require(capacity_ <=
                std::numeric_limits<std::uint64_t>::max() /
                    minimum_sequence_workspace_bytes,
            "DeepSeek minimum workspace accounting overflow");
    const auto minimum_cache_bytes =
        shared_bytes +
        (static_cast<std::uint64_t>(main_component.route_width) + 1U) *
            routed_device_bytes +
        mtp_reserve_bytes;
    const auto fitted_cache = er::fit_device_cache_budget(
        {static_cast<std::uint64_t>(free), fixed_without_sequence,
         minimum_sequence_workspace_bytes * capacity_, device_reserve_bytes,
         vram_bytes_, minimum_cache_bytes});
    require(fitted_cache.status.ok(),
            std::string("DeepSeek worker VRAM cache fit failed: ") +
                std::string(fitted_cache.status.message()));
    vram_bytes_ = fitted_cache.effective_cache_bytes;
    const auto reserved_without_sequence =
        fixed_without_sequence + vram_bytes_ + device_reserve_bytes;
    require(reserved_without_sequence <= free,
            "DeepSeek worker VRAM preflight failed before sequence "
            "workspace: required=" +
                std::to_string(reserved_without_sequence) +
                ", free=" + std::to_string(free));
    const auto sequence_budget = free - reserved_without_sequence;
    for (std::uint32_t candidate = kMaximumSequenceTileRows;
         candidate >= sequence_row_quantum;
         candidate -= sequence_row_quantum) {
      const auto attention_size =
          er::cuda::deepseek_attention_batch_workspace_size(max_context_,
                                                             candidate);
      const auto ffn_bytes =
          er::cuda::deepseek_ffn_batch_workspace_size(candidate);
      if (!attention_size.status.ok() || ffn_bytes == 0U) continue;
      const auto workspace_bytes =
          attention_size.bytes + ffn_bytes +
          2ULL * candidate * 4U * 4096U * sizeof(float);
      if (workspace_bytes <= sequence_budget / capacity_) {
        sequence_tile_rows_ = candidate;
        sequence_workspace_bytes_ = workspace_bytes;
        break;
      }
    }
    require(sequence_tile_rows_ != 0U,
            "DeepSeek worker VRAM preflight cannot fit a 32-row exact "
            "sequence tile: budget=" + std::to_string(sequence_budget));

    // Four IOCP workers match the Qwen runner and the useful NCQ depth of
    // the SATA pack drive; more outstanding random reads mostly add seeks.
    iocp_ = std::make_shared<er::WindowsIocpStorage>(4U);
    storage_ = std::make_shared<er::ExtentGatherStorage>(iocp_);
    const auto staging = std::max<std::uint64_t>(
        64ULL << 20U,
        std::max(artifacts_.maximum_source_record_bytes,
                 mtp_artifacts_.maximum_source_record_bytes));
    // Eight staging slots keep a full top-6 demand route in flight and still
    // leave room for two census prefetches during a demand burst.
    const auto staging_slots = std::max<std::uint32_t>(
        8U, main_component.route_width + 2U);
    MEMORYSTATUSEX memory{sizeof(memory)};
    require(GlobalMemoryStatusEx(&memory) != 0,
            "inspect DeepSeek worker RAM failed");
    constexpr std::uint64_t operating_system_reserve = 4ULL << 30U;
    require(ram_bytes_ <= std::numeric_limits<std::uint64_t>::max() -
                              staging * staging_slots -
                              operating_system_reserve &&
                ram_bytes_ + staging * staging_slots +
                        operating_system_reserve <=
                    memory.ullAvailPhys,
            "DeepSeek worker RAM preflight failed");
    buffers_ = std::make_shared<er::FixedBufferPool>(
        staging_slots, staging, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>(),
        main_component.route_width);
    model_ = std::make_shared<er::cuda::DeepSeekResidentModelState>();
    const auto model_status = er::cuda::DeepSeekResidentModelState::load(
        *storage_, *buffers_, artifacts_.dense, artifacts_.typed, *model_);
    require(model_status.ok(), model_status.message());
    if (mtp_enabled_) {
      mtp_model_ =
          std::make_shared<er::cuda::DeepSeekResidentTensorState>();
      const auto mtp_model_status =
          er::cuda::DeepSeekResidentTensorState::load(
              *storage_, *buffers_, mtp_artifacts_.dense,
              mtp_artifacts_.typed, *mtp_model_);
      require(mtp_model_status.ok(), mtp_model_status.message());
    }
    directory_ = std::make_shared<er::cuda::CudaExpertDirectory>(
        main_component.namespace_id, main_component.encoding_abi,
        main_component.layer_count,
        main_component.experts_per_layer +
            main_component.shared_experts_per_layer,
        std::max<std::uint32_t>(64U, capacity_ * 8U));
    uploader_ = std::make_shared<er::cuda::CudaExpertUploader>(
        er::cuda::CudaExpertUploaderOptions{
            0U, true, 0U, true});
    er::ExpertCacheConfig cache_config;
    const auto target_ram_bytes = ram_bytes_ - mtp_reserve_bytes;
    const auto target_vram_bytes = vram_bytes_ - mtp_reserve_bytes;
    route_trace_record_bytes_ = representative->stored_bytes;
    route_trace_device_bytes_ = routed_device_bytes;
    route_trace_ram_capacity_records_ =
        target_ram_bytes / representative->stored_bytes;
    cache_config.ram = {target_ram_bytes, target_ram_bytes,
                        target_ram_bytes * 15U / 16U};
    cache_config.vram = {target_vram_bytes, target_vram_bytes,
                         target_vram_bytes * 15U / 16U};
    const auto desired_census_bytes =
        static_cast<std::uint64_t>(main_component.layer_count) *
        std::min<std::uint32_t>(32U, main_component.experts_per_layer) *
        representative->stored_bytes;
    cache_config.placement.ram_protected_bytes = std::min(
        desired_census_bytes, target_ram_bytes * 3U / 4U);
    cache_ram_protected_bytes_ =
        cache_config.placement.ram_protected_bytes;
    const auto routed_vram_bytes = target_vram_bytes - shared_bytes;
    route_trace_vram_capacity_records_ =
        routed_vram_bytes / routed_device_bytes;
    const auto retained_records =
        static_cast<std::uint64_t>(capacity_) *
            main_component.layer_count * main_component.route_width +
        main_component.route_width;
    const auto retained_bytes = retained_records * routed_device_bytes;
    retain_previous_route_ = retained_route_enabled_ &&
        retained_bytes + 2ULL * main_component.route_width *
                             routed_device_bytes <= routed_vram_bytes &&
        retained_bytes <= routed_vram_bytes / 2U;
    cache_config.placement.vram_transient_bytes =
        retain_previous_route_
            ? retained_bytes
            : 2ULL * main_component.route_width * routed_device_bytes;
    cache_vram_resident_limit_bytes_ =
        target_vram_bytes - cache_config.placement.vram_transient_bytes;
    cache_config.retain_host_copy = true;
    // First-touch demand is retained in the bounded probationary LFU segment.
    // Only explicit census/session evidence enters the protected segment;
    // ordinary reuse raises temperature without permanently consuming the
    // census reservation.
    // Scheduler route feedback is published after the layer executes, so a
    // cold demand has frequency zero while its host copy is being admitted.
    cache_config.ram_retention_minimum_frequency = 0U;
    cache_config.trusted_immutable_source = true;
    cache_ = std::make_unique<er::ExpertCache>(
        cache_config, storage_, uploader_, buffers_, directory_);
    local_store_ = std::make_unique<er::LocalExpertStore>(*cache_);
    er::ActiveExpertComponentContract main_contract;
    main_contract.model_content_hash = bundle_.descriptor.content_hash;
    main_contract.namespace_id = main_component.namespace_id;
    main_contract.layer_count = main_component.layer_count;
    main_contract.experts_per_layer = main_component.experts_per_layer;
    main_contract.encoding_abi = main_component.encoding_abi;
    main_contract.execution_capability = main_component.execution_capability;
    main_contract.execution_abi = main_component.execution_abi;
    main_contract.source_abi = main_component.source_abi;
    remote_store_ = std::make_unique<er::RemoteExpertStore>(
        std::move(main_contract), *remote_owners_);
    placement_store_ = std::make_unique<er::PlacementExpertStore>(
        *local_store_, *remote_store_);
    routed_ = std::make_unique<er::RoutedExpertRuntime>(
        main_component, bundle_.model_hash, catalog_, *cache_,
        *placement_store_);
    const auto shared_status = er::ResidentExpertSet::load(
        *cache_, artifacts_.shared, shared_);
    require(shared_status.ok(), shared_status.message());
    if (mtp_enabled_) {
      const auto& draft_component =
          required_component(bundle_.descriptor, "draft");
      mtp_directory_ = std::make_shared<er::cuda::CudaExpertDirectory>(
          draft_component.namespace_id, draft_component.encoding_abi,
          draft_component.layer_count,
          draft_component.experts_per_layer +
              draft_component.shared_experts_per_layer,
          std::max<std::uint32_t>(16U, capacity_ * 8U));
      mtp_uploader_ = std::make_shared<er::cuda::CudaExpertUploader>(
          er::cuda::CudaExpertUploaderOptions{0U, true, 0U, true});
      er::ExpertCacheConfig mtp_cache_config;
      mtp_cache_config.ram = {mtp_cache_bytes_, mtp_cache_bytes_,
                              mtp_cache_bytes_ * 7U / 8U};
      mtp_cache_config.vram = {mtp_cache_bytes_, mtp_cache_bytes_,
                               mtp_cache_bytes_ * 7U / 8U};
      mtp_cache_config.retain_host_copy = true;
      mtp_cache_config.trusted_immutable_source = true;
      mtp_cache_ = std::make_unique<er::ExpertCache>(
          mtp_cache_config, storage_, mtp_uploader_, buffers_,
          mtp_directory_);
      mtp_local_store_ =
          std::make_unique<er::LocalExpertStore>(*mtp_cache_);
      er::ActiveExpertComponentContract draft_contract;
      draft_contract.model_content_hash = bundle_.descriptor.content_hash;
      draft_contract.namespace_id = draft_component.namespace_id;
      draft_contract.layer_count = draft_component.layer_count;
      draft_contract.experts_per_layer = draft_component.experts_per_layer;
      draft_contract.encoding_abi = draft_component.encoding_abi;
      draft_contract.execution_capability =
          draft_component.execution_capability;
      draft_contract.execution_abi = draft_component.execution_abi;
      draft_contract.source_abi = draft_component.source_abi;
      mtp_remote_store_ = std::make_unique<er::RemoteExpertStore>(
          std::move(draft_contract), *remote_owners_);
      mtp_placement_store_ = std::make_unique<er::PlacementExpertStore>(
          *mtp_local_store_, *mtp_remote_store_);
      mtp_routed_ = std::make_unique<er::RoutedExpertRuntime>(
          draft_component, bundle_.model_hash, mtp_catalog_, *mtp_cache_,
          *mtp_placement_store_);
      const auto mtp_shared_status = er::ResidentExpertSet::load(
          *mtp_cache_, mtp_shared_specs_, mtp_shared_);
      require(mtp_shared_status.ok(), mtp_shared_status.message());
    }
    initialize_rope_table();
    cpu_ = std::make_shared<er::cpu::DeepSeekPackedExecutor>(
        er::cpu::DeepSeekPackedExecutorConfig{
            std::clamp(std::thread::hardware_concurrency(), 1U, 64U),
            8U, 8U, 10.0F, true, true});
    planner_ = std::make_shared<er::HybridDispatchPlanner>(
        er::HybridDispatchConfig{bundle_.cpu_ns, bundle_.gpu_ns,
                                 bundle_.h2d_bytes_per_second,
                                 0.125, 256U, 256U,
                                 true, true, true, 1U});
    auto census_loaded = er::RouteCensus::load(
        bundle_.census, routed_->census_config());
    if (census_loaded.status.ok()) {
      census_namespace_rebound_ = census_loaded.namespace_rebound;
      census_ = std::shared_ptr<er::RouteCensus>(
          std::move(census_loaded.census));
      if (census_namespace_rebound_) {
        const auto migrated = census_->save(bundle_.census);
        require(migrated.ok(), migrated.message());
      }
    } else {
      require(census_loaded.status.code() == er::ErrorCode::open_failed,
              census_loaded.status.message());
      census_ = std::make_shared<er::RouteCensus>(routed_->census_config());
    }
    prepare_warm_from_census();
    scheduler_ = std::make_unique<er::cuda::DeepSeekDecodeScheduler>(
        er::cuda::DeepSeekDecodeSchedulerConfig{
            capacity_,
            std::max<std::uint32_t>(2U * main_component.route_width,
                                    capacity_ * main_component.route_width),
            capacity_, 4U, 2U, retain_previous_route_},
        *routed_,
        er::cuda::DeepSeekHybridSchedulerDependencies{
            cpu_hybrid_enabled_ ? cpu_ : nullptr,
            cpu_hybrid_enabled_ ? planner_ : nullptr, census_});
  }

  ~Model() {
    stop_background_warm();
    const auto saved = census_->save(bundle_.census);
    if (!saved.ok())
      std::cerr << "route census save failed: " << saved.message() << '\n';
    scheduler_.reset();
    if (rope_table_) static_cast<void>(cudaFree(rope_table_));
  }

  void start_background_warm() {
    if (warm_candidates_.empty() || warm_thread_.joinable()) return;
    warm_stop_.store(false, std::memory_order_release);
    warm_thread_ = std::thread([this] { background_warm_loop(); });
  }

  void wait_background_warm() {
    if (warm_thread_.joinable()) warm_thread_.join();
  }

  struct PreparedOperation final : er::IPreparedOperation {
    std::uint32_t kernel{};
    std::uint32_t logical_layer{};
    std::uint32_t component_layer{};
    std::optional<std::uint32_t> prefetch_target_component_layer;
    std::vector<std::pair<std::string, std::string>> outputs;
  };

  struct PreparedExactDecode final : er::IPreparedOperation {};

  enum class CallablePrefetchTier : std::uint8_t {
    device,
    host,
    storage,
  };

  struct CallableRequestState final : er::IOperationProviderRequestState {
    struct PrefetchItem final {
      std::uint32_t expert{};
      CallablePrefetchTier source_tier{CallablePrefetchTier::storage};
      er::ExpertResolveHandle handle;
      er::ResolvedExpert resolved;
      bool terminal{};
      bool failed{};
    };

    struct RetentionCheckpoint final {
      void* host_state{};
      std::uint64_t host_state_bytes{};
      std::uint32_t position{};
      std::uint32_t predicted{};
      std::uint32_t draft{};
      bool draft_ready{};
      bool speculation_suppressed{};
      bool valid{};
    };

    std::shared_ptr<Model> owner;
    std::shared_ptr<er::cuda::DeepSeekRequestState> state;
    std::shared_ptr<er::cuda::DeepSeekVerifyState> verify;
    std::shared_ptr<er::cuda::DeepSeekMtpRequestState> mtp;
    std::shared_ptr<er::cuda::DeepSeekDecodeController> verify_controller;
    std::shared_ptr<er::cuda::CudaDirectoryPlanWorkspace> directory_workspace;
    std::shared_ptr<er::cuda::DeepSeekRoutePredictionState>
        route_prediction_state;
    std::shared_ptr<er::cuda::DeepSeekAttentionBatchWorkspace>
        sequence_attention_workspace;
    std::shared_ptr<er::cuda::DeepSeekFfnBatchWorkspace>
        sequence_ffn_workspace;
    cudaStream_t stream{};
    cudaEvent_t remote_input_ready_event{};
    cudaEvent_t route_prediction_ready_event{};
    cudaEvent_t gpu_selection_started_event{};
    cudaEvent_t gpu_selection_finished_event{};
    float* remote_input_host{};
    float* remote_outputs_host{};
    std::uint32_t* route_prediction_host{};
    float* sequence_stream_allocation{};
    float* sequence_primary_streams{};
    float* sequence_attention_streams{};
    std::vector<PrefetchItem> prefetch_items;
    std::array<RetentionCheckpoint, 2U> retention_checkpoints;
    std::vector<std::map<std::uint32_t, PromptRouteEvidence>> prompt_routes;
    std::uint64_t prompt_route_clock{};
    std::vector<std::uint32_t> sequence_sync_successors;
    std::vector<float> sequence_final_target_streams;
    std::uint32_t sequence_sync_first{};
    std::uint32_t sequence_sync_consumed{};
    std::uint64_t next_remote_invocation{1U};
    std::uint32_t provider_slot{};
    std::uint32_t context_limit{};
    std::uint32_t next_position{};
    std::uint32_t current_token{};
    std::uint32_t predicted{};
    std::uint32_t draft{};
    std::uint32_t current_position{};
    std::uint32_t prediction_target_layer{
        std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t current_router_component_layer{
        std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t retention_target_position{};
    std::uint32_t selected_retention_checkpoint{
        std::numeric_limits<std::uint32_t>::max()};
    std::optional<std::uint32_t> current_prefetch_target_layer;
    bool embedded{};
    bool attention_ready{};
    bool route_ready{};
    bool prediction_copy_pending{};
    bool draft_ready{};
    bool speculation_suppressed{};
    bool exact_decode_enabled{};
    bool gpu_selection_observation_pending{};
    std::uint32_t gpu_selection_observation_selections{};

    ~CallableRequestState() override {
      for (auto& item : prefetch_items) item.handle.cancel();
      prefetch_items.clear();
      directory_workspace.reset();
      if (stream) static_cast<void>(cudaStreamSynchronize(stream));
      route_prediction_state.reset();
      sequence_attention_workspace.reset();
      sequence_ffn_workspace.reset();
      verify_controller.reset();
      verify.reset();
      mtp.reset();
      state.reset();
      if (route_prediction_ready_event)
        static_cast<void>(cudaEventDestroy(route_prediction_ready_event));
      if (remote_input_ready_event)
        static_cast<void>(cudaEventDestroy(remote_input_ready_event));
      if (gpu_selection_finished_event)
        static_cast<void>(cudaEventDestroy(gpu_selection_finished_event));
      if (gpu_selection_started_event)
        static_cast<void>(cudaEventDestroy(gpu_selection_started_event));
      if (route_prediction_host)
        static_cast<void>(cudaFreeHost(route_prediction_host));
      if (sequence_stream_allocation)
        static_cast<void>(cudaFree(sequence_stream_allocation));
      if (remote_outputs_host)
        static_cast<void>(cudaFreeHost(remote_outputs_host));
      if (remote_input_host)
        static_cast<void>(cudaFreeHost(remote_input_host));
      for (auto& checkpoint : retention_checkpoints)
        if (checkpoint.host_state)
          static_cast<void>(cudaFreeHost(checkpoint.host_state));
      if (stream) static_cast<void>(cudaStreamDestroy(stream));
      if (owner) owner->release_callable_provider_slot(provider_slot);
    }
  };

  struct CallableProgramSequence final {
    std::shared_ptr<CallableRequestState> request;
    er::ProgramRequestContext context;
    std::vector<const PreparedOperation*> operations;
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> positions;
    std::atomic<bool> cancelled{false};
    std::mutex result_mutex;
    std::optional<er::OperationExecutionResult> result;
    std::thread worker;
    bool consumed{};

    ~CallableProgramSequence() {
      cancelled.store(true, std::memory_order_release);
      if (worker.joinable()) worker.join();
    }

    void publish(er::OperationExecutionResult value) {
      std::lock_guard lock(result_mutex);
      result = std::move(value);
    }

    [[nodiscard]] std::optional<er::OperationExecutionResult> poll() {
      std::optional<er::OperationExecutionResult> completed;
      {
        std::lock_guard lock(result_mutex);
        if (consumed || !result) return std::nullopt;
        consumed = true;
        completed = std::move(result);
        result.reset();
      }
      if (worker.joinable()) worker.join();
      return completed;
    }
  };

  er::PrepareOperationResult prepare(
      const er::OperationPreparationContext& context) override {
    try {
      if (context.model.content_hash != bundle_.descriptor.content_hash ||
          context.model.schema_version < 3U ||
          context.model.architecture_id != bundle_.descriptor.architecture_id ||
          context.model.hidden_size != bundle_.descriptor.hidden_size ||
          context.model.vocab_size != bundle_.descriptor.vocab_size ||
          context.model.layer_program.size() !=
              bundle_.descriptor.layer_program.size() ||
          !context.tensors.empty())
        return {{er::ErrorCode::invalid_argument,
                 "compressed sparse provider received a foreign operation"},
                {}};
      auto prepared = std::make_shared<PreparedOperation>();
      prepared->logical_layer = context.compiled.logical_layer;
      prepared->component_layer = context.compiled.component_layer;
      const auto& capability = context.operation.capability;
      if (capability == "embedding.lookup.int8-row.v1") {
        prepared->kernel = kCallableEmbedding;
      } else if (capability ==
                 "block.compressed-sparse-attention.hca.v1") {
        prepared->kernel = kCallableAttention;
      } else if (capability == "router.deepseek.v4.topk.v1") {
        prepared->kernel = kCallableRouter;
      } else if (capability == "moe.swiglu.routed.v1") {
        prepared->kernel = kCallableRoutedMoe;
      } else if (capability == "head.rmsnorm.argmax.int8-row.v1") {
        prepared->kernel = kCallableHead;
      } else {
        return {{er::ErrorCode::invalid_argument,
                 "compressed sparse callable operation is unsupported"},
                {}};
      }
      const auto prefetch_target =
          context.compiled.parameters.find("prefetch_target_component_layer");
      if (prefetch_target != context.compiled.parameters.end()) {
        if (prepared->kernel != kCallableRouter ||
            !context.compiled.routed_component_index ||
            *context.compiled.routed_component_index >=
                context.model.routed_components.size() ||
            prefetch_target->second >
                std::numeric_limits<std::uint32_t>::max() ||
            prefetch_target->second >=
                context.model.routed_components
                    [*context.compiled.routed_component_index]
                        .layer_count ||
            prefetch_target->second == context.compiled.component_layer)
          return {{er::ErrorCode::invalid_argument,
                   "compressed sparse prefetch target is invalid"},
                  {}};
        prepared->prefetch_target_component_layer =
            static_cast<std::uint32_t>(prefetch_target->second);
      }
      prepared->outputs.reserve(context.compiled.output_values.size());
      for (const auto& output : context.compiled.output_values) {
        const auto source = context.operation.output_bindings.find(output.port);
        if (source == context.operation.output_bindings.end())
          return {{er::ErrorCode::invalid_argument,
                   "compressed sparse operation output is not declared"},
                  {}};
        prepared->outputs.emplace_back(output.port, source->second.abi);
      }
      return {er::Status::success(), std::move(prepared)};
    } catch (const std::exception& error) {
      return {{er::ErrorCode::invalid_argument, error.what()}, {}};
    }
  }

  er::PrepareOperationResult prepare_exact_decode(
      const er::ExactDecodePreparationContext& context) override {
    try {
      const auto target = context.compiled.parameters.find(
          "target_component_index");
      const auto draft = context.compiled.parameters.find(
          "draft_component_index");
      if (!mtp_enabled_ ||
          context.model.content_hash != bundle_.descriptor.content_hash ||
          context.program.capability != "decode.speculative.verify.v1" ||
          context.program.abi_version != 1U ||
          context.compiled.maximum_emitted_tokens != 2U ||
          target == context.compiled.parameters.end() ||
          draft == context.compiled.parameters.end() ||
          target->second >= context.model.routed_components.size() ||
          draft->second >= context.model.routed_components.size() ||
          target->second == draft->second)
        return {{er::ErrorCode::invalid_argument,
                 "compressed sparse exact decode contract is invalid"},
                {}};
      return {er::Status::success(),
              std::make_shared<PreparedExactDecode>()};
    } catch (const std::exception& error) {
      return {{er::ErrorCode::invalid_argument, error.what()}, {}};
    }
  }

  er::Status synchronize_exact_decode(
      const er::IPreparedOperation& opaque_operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      const er::ExactDecodeSynchronization& synchronization) override {
    const std::array<std::uint32_t, 1U> tokens{
        synchronization.next_token};
    return synchronize_exact_decode_batch(
        opaque_operation, opaque_state,
        {synchronization.request, tokens, synchronization.target_position,
         synchronization.produce_draft});
  }

  er::Status synchronize_exact_decode_batch(
      const er::IPreparedOperation& opaque_operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      const er::ExactDecodeSynchronizationBatch& synchronization) override {
    const auto* prepared =
        dynamic_cast<const PreparedExactDecode*>(&opaque_operation);
    const auto state =
        std::dynamic_pointer_cast<CallableRequestState>(opaque_state);
    if (!prepared || !state || state->owner.get() != this || !state->mtp ||
        synchronization.next_tokens.empty() ||
        std::any_of(synchronization.next_tokens.begin(),
                    synchronization.next_tokens.end(),
                    [this](std::uint32_t token) {
                      return token >= bundle_.descriptor.vocab_size;
                    }))
      return {er::ErrorCode::invalid_argument,
              "compressed sparse exact decode synchronization is invalid"};
    try {
      if (!state->sequence_sync_successors.empty()) {
        const auto total = state->sequence_sync_successors.size();
        const auto count = synchronization.next_tokens.size();
        if (state->sequence_sync_consumed > total ||
            count > total - state->sequence_sync_consumed ||
            synchronization.first_target_position !=
                state->sequence_sync_first + state->sequence_sync_consumed ||
            state->sequence_final_target_streams.size() !=
                4ULL * bundle_.descriptor.hidden_size)
          throw std::runtime_error(
              "DeepSeek sequence exact-sync cursor is invalid");
        for (std::size_t row = 0U; row < count; ++row) {
          const auto index = state->sequence_sync_consumed + row;
          if (index + 1U < total &&
              synchronization.next_tokens[row] !=
                  state->sequence_sync_successors[index])
            throw std::runtime_error(
                "DeepSeek sequence exact-sync successor changed");
          if (index + 1U == total) {
            cuda_check(
                cudaMemcpyAsync(
                    state->state->primary_streams(),
                    state->sequence_final_target_streams.data(),
                    state->sequence_final_target_streams.size() *
                        sizeof(float),
                    cudaMemcpyHostToDevice, state->stream),
                "upload final DeepSeek sequence exact-sync streams");
            advance_mtp_state(
                *state, synchronization.next_tokens[row],
                state->state->primary_streams(),
                synchronization.first_target_position +
                    static_cast<std::uint32_t>(row),
                synchronization.produce_final_draft);
          }
        }
        state->sequence_sync_consumed += static_cast<std::uint32_t>(count);
        if (state->sequence_sync_consumed == total) {
          state->sequence_sync_successors.clear();
          state->sequence_final_target_streams.clear();
          state->sequence_sync_first = 0U;
          state->sequence_sync_consumed = 0U;
        }
        return er::Status::success();
      }
      if (synchronization.next_tokens.size() != 1U ||
          synchronization.first_target_position + 1U !=
              state->next_position ||
          (synchronization.produce_final_draft &&
           synchronization.next_tokens.front() != state->predicted))
        throw std::runtime_error(
            "DeepSeek scalar exact-sync boundary is invalid");
      advance_mtp_state(*state, synchronization.next_tokens.front(),
                        state->state->current_streams(),
                        synchronization.first_target_position,
                        synchronization.produce_final_draft);
      return er::Status::success();
    } catch (const std::exception& error) {
      return {er::ErrorCode::internal, error.what()};
    }
  }

  er::ExactDecodeExecutionHandle execute_exact_decode(
      const er::IPreparedOperation& opaque_operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      const er::ExactDecodeInvocation& invocation) override {
    const auto* prepared =
        dynamic_cast<const PreparedExactDecode*>(&opaque_operation);
    const auto state =
        std::dynamic_pointer_cast<CallableRequestState>(opaque_state);
    if (!prepared || !state || state->owner.get() != this ||
        !state->verify_controller || !state->draft_ready ||
        invocation.guaranteed_token != state->predicted ||
        invocation.position != state->next_position ||
        invocation.context_limit > state->context_limit)
      return completed_exact_decode(
          {{er::ErrorCode::invalid_argument,
            "compressed sparse exact decode invocation is invalid"}});
    try {
      const auto before = state->next_position;
      auto emitted = verify_draft_state(*state, state->verify_controller);
      er::ExactDecodeExecutionResult result;
      result.status = er::Status::success();
      result.emitted_tokens = std::move(emitted);
      result.next_token = state->predicted;
      result.positions_advanced = state->next_position - before;
      return completed_exact_decode(std::move(result));
    } catch (const std::exception& error) {
      return completed_exact_decode(
          {{er::ErrorCode::internal, error.what()}});
    }
  }

  [[nodiscard]] std::uint64_t retention_checkpoint_bytes(
      const CallableRequestState& state) const {
    std::uint64_t bytes{};
    for (std::uint32_t layer = 0U; layer < state.state->layer_count();
         ++layer) {
      const auto view = state.state->layer(layer);
      require(view.attention_state != nullptr,
              "DeepSeek retention layer state is absent");
      const auto layer_bytes =
          view.attention_state->retention_checkpoint_bytes();
      require(layer_bytes <=
                  std::numeric_limits<std::uint64_t>::max() - bytes,
              "DeepSeek retention checkpoint size overflow");
      bytes += layer_bytes;
    }
    return bytes;
  }

  er::Status capture_retention_checkpoint(CallableRequestState& state,
                                          std::uint32_t position,
                                          bool select) {
    try {
      require(position != 0U && position == state.next_position,
              "DeepSeek retention checkpoint is not at a causal boundary");
      auto selected = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t index = 0U;
           index < state.retention_checkpoints.size(); ++index) {
        if (state.retention_checkpoints[index].valid &&
            state.retention_checkpoints[index].position == position) {
          selected = index;
          break;
        }
      }
      if (selected == std::numeric_limits<std::uint32_t>::max()) {
        for (std::uint32_t index = 0U;
             index < state.retention_checkpoints.size(); ++index) {
          if (!state.retention_checkpoints[index].valid ||
              index != state.selected_retention_checkpoint) {
            selected = index;
            break;
          }
        }
      }
      require(selected < state.retention_checkpoints.size(),
              "DeepSeek retention checkpoint ring is exhausted");
      auto& checkpoint = state.retention_checkpoints[selected];
      const auto bytes = retention_checkpoint_bytes(state);
      if (checkpoint.host_state_bytes != bytes) {
        if (checkpoint.host_state) {
          cuda_check(cudaFreeHost(checkpoint.host_state),
                     "release resized DeepSeek retention checkpoint");
          checkpoint.host_state = nullptr;
        }
        if (bytes != 0U)
          cuda_check(cudaHostAlloc(&checkpoint.host_state,
                                   static_cast<std::size_t>(bytes),
                                   cudaHostAllocPortable),
                     "allocate DeepSeek retention checkpoint");
        checkpoint.host_state_bytes = bytes;
      }
      auto* cursor = static_cast<std::byte*>(checkpoint.host_state);
      for (std::uint32_t layer = 0U; layer < state.state->layer_count();
           ++layer) {
        const auto view = state.state->layer(layer);
        const auto layer_bytes =
            view.attention_state->retention_checkpoint_bytes();
        const auto status = view.attention_state->checkpoint_retention_state(
            layer_bytes == 0U ? nullptr : cursor, state.stream);
        require(status.ok(), status.message());
        if (layer_bytes != 0U) cursor += layer_bytes;
      }
      cuda_check(cudaStreamSynchronize(state.stream),
                 "complete DeepSeek retention checkpoint");
      checkpoint.position = position;
      checkpoint.predicted = state.predicted;
      checkpoint.draft = state.draft;
      checkpoint.draft_ready = state.draft_ready;
      checkpoint.speculation_suppressed = state.speculation_suppressed;
      checkpoint.valid = true;
      if (select) state.selected_retention_checkpoint = selected;
      return er::Status::success();
    } catch (const std::exception& error) {
      return {er::ErrorCode::internal, error.what()};
    }
  }

  [[nodiscard]] bool supports_request_state_retention()
      const noexcept override {
    // Target recurrent state is checkpointed below, but the callable MTP and
    // in-flight sequence synchronization state have not passed an exact
    // checkpoint/rewind parity gate. Do not expose retention commands until
    // every selected state component is qualified.
    return false;
  }

  er::Status checkpoint_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      std::uint32_t position) override {
    const auto state =
        std::dynamic_pointer_cast<CallableRequestState>(opaque_state);
    if (!state || state->owner.get() != this)
      return {er::ErrorCode::invalid_argument,
              "DeepSeek retention checkpoint ownership is invalid"};
    for (std::uint32_t index = 0U;
         index < state->retention_checkpoints.size(); ++index) {
      if (state->retention_checkpoints[index].valid &&
          state->retention_checkpoints[index].position == position) {
        state->selected_retention_checkpoint = index;
        return er::Status::success();
      }
    }
    return capture_retention_checkpoint(*state, position, true);
  }

  er::Status rewind_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      std::uint32_t position) override {
    const auto state =
        std::dynamic_pointer_cast<CallableRequestState>(opaque_state);
    if (!state || state->owner.get() != this)
      return {er::ErrorCode::invalid_argument,
              "DeepSeek retention rewind ownership is invalid"};
    try {
      const auto found = std::find_if(
          state->retention_checkpoints.begin(),
          state->retention_checkpoints.end(),
          [position](const CallableRequestState::RetentionCheckpoint& item) {
            return item.valid && item.position == position;
          });
      require(found != state->retention_checkpoints.end(),
              "DeepSeek retention checkpoint is unavailable");
      reset_callable_prefetch(*state, true);
      auto* cursor = static_cast<const std::byte*>(found->host_state);
      for (std::uint32_t layer = 0U; layer < state->state->layer_count();
           ++layer) {
        const auto view = state->state->layer(layer);
        const auto layer_bytes =
            view.attention_state->retention_checkpoint_bytes();
        const auto status = view.attention_state->restore_retention_state(
            layer_bytes == 0U ? nullptr : cursor, state->stream);
        require(status.ok(), status.message());
        if (layer_bytes != 0U) cursor += layer_bytes;
      }
      cuda_check(cudaStreamSynchronize(state->stream),
                 "complete DeepSeek retention rewind");
      state->next_position = position;
      state->predicted = found->predicted;
      state->draft = found->draft;
      state->draft_ready = found->draft_ready;
      state->speculation_suppressed = found->speculation_suppressed;
      state->embedded = false;
      state->attention_ready = false;
      state->route_ready = false;
      state->current_prefetch_target_layer.reset();
      state->current_router_component_layer =
          std::numeric_limits<std::uint32_t>::max();
      state->selected_retention_checkpoint = static_cast<std::uint32_t>(
          found - state->retention_checkpoints.begin());
      state->retention_target_position = position;
      return er::Status::success();
    } catch (const std::exception& error) {
      return {er::ErrorCode::internal, error.what()};
    }
  }

  er::Status rebind_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      const er::ProgramRequestContext& request) override {
    const auto state =
        std::dynamic_pointer_cast<CallableRequestState>(opaque_state);
    const auto reserved = request.parameters.find("reserved_context_tokens");
    const auto checkpoint =
        request.parameters.find("retention_checkpoint_position");
    if (!state || state->owner.get() != this ||
        reserved == request.parameters.end() || reserved->second == 0U ||
        reserved->second > max_context_ ||
        (checkpoint != request.parameters.end() &&
         (checkpoint->second == 0U || checkpoint->second > reserved->second)))
      return {er::ErrorCode::invalid_argument,
              "DeepSeek retained request rebind is invalid"};
    state->context_limit = static_cast<std::uint32_t>(reserved->second);
    state->retention_target_position =
        checkpoint == request.parameters.end()
            ? 0U
            : static_cast<std::uint32_t>(checkpoint->second);
    return er::Status::success();
  }

  er::CreateOperationRequestStateResult create_request_state(
      const er::ProgramRequestContext& request) override {
    try {
      const auto reserved = request.parameters.find("reserved_context_tokens");
      if (reserved == request.parameters.end() || reserved->second == 0U ||
          reserved->second > max_context_ ||
          reserved->second > std::numeric_limits<std::uint32_t>::max())
        return {{er::ErrorCode::invalid_argument,
                 "compressed sparse request has no valid context reservation"},
                {}};
      const auto slot = acquire_callable_provider_slot();
      if (!slot)
        return {{er::ErrorCode::backpressure,
                 "compressed sparse provider capacity is exhausted"},
                {}};
      auto state = std::make_shared<CallableRequestState>();
      state->owner = shared_from_this();
      state->provider_slot = *slot;
      state->context_limit = static_cast<std::uint32_t>(reserved->second);
      const auto checkpoint =
          request.parameters.find("retention_checkpoint_position");
      if (checkpoint != request.parameters.end()) {
        if (checkpoint->second == 0U ||
            checkpoint->second > reserved->second ||
            checkpoint->second >
                std::numeric_limits<std::uint32_t>::max())
          return {{er::ErrorCode::invalid_argument,
                   "compressed sparse retention checkpoint is invalid"},
                  {}};
        state->retention_target_position =
            static_cast<std::uint32_t>(checkpoint->second);
      }
      try {
        auto created = er::cuda::create_deepseek_request_state(
            model_, {max_context_, request_bytes_, compression_ratios_,
                     hash_router_layers_});
        require(created.status.ok() && created.state,
                created.status.ok() ? "DeepSeek request returned no state"
                                    : created.status.message());
        state->state = std::move(created.state);
        auto prediction = er::cuda::create_deepseek_route_prediction_state();
        require(prediction.status.ok() && prediction.state,
                prediction.status.ok()
                    ? "DeepSeek route predictor returned no state"
                    : prediction.status.message());
        state->route_prediction_state = std::move(prediction.state);
        cuda_check(cudaStreamCreateWithFlags(&state->stream,
                                             cudaStreamNonBlocking),
                   "create callable DeepSeek stream");
        auto attention_workspace =
            er::cuda::create_deepseek_attention_batch_workspace(
                max_context_, sequence_tile_rows_);
        require(attention_workspace.status.ok() &&
                    attention_workspace.workspace,
                attention_workspace.status.ok()
                    ? "DeepSeek sequence attention workspace is absent"
                    : attention_workspace.status.message());
        state->sequence_attention_workspace =
            std::move(attention_workspace.workspace);
        auto ffn_workspace = er::cuda::create_deepseek_ffn_batch_workspace(
            sequence_tile_rows_);
        require(ffn_workspace.status.ok() && ffn_workspace.workspace,
                ffn_workspace.status.ok()
                    ? "DeepSeek sequence FFN workspace is absent"
                    : ffn_workspace.status.message());
        state->sequence_ffn_workspace = std::move(ffn_workspace.workspace);
        const auto active_sequence_stream_values =
            2ULL * sequence_tile_rows_ * 4U * 4096U;
        cuda_check(cudaMalloc(
                       reinterpret_cast<void**>(
                           &state->sequence_stream_allocation),
                       active_sequence_stream_values * sizeof(float)),
                   "allocate callable DeepSeek sequence streams");
        state->sequence_primary_streams = state->sequence_stream_allocation;
        state->sequence_attention_streams =
            state->sequence_primary_streams +
            static_cast<std::uint64_t>(sequence_tile_rows_) * 4U * 4096U;
        state->prompt_routes.resize(state->state->layer_count());
        if (mtp_enabled_) {
          auto mtp_state = er::cuda::create_deepseek_mtp_request_state(
              model_, mtp_model_, {max_context_, mtp_request_bytes_});
          require(mtp_state.status.ok() && mtp_state.state,
                  mtp_state.status.ok()
                      ? "callable MTP request returned no ownership"
                      : mtp_state.status.message());
          state->mtp = std::move(mtp_state.state);
          auto verify = er::cuda::create_deepseek_verify_state(
              state->state, verify_request_bytes_);
          require(verify.status.ok() && verify.state,
                  verify.status.ok()
                      ? "callable verify state returned no ownership"
                      : verify.status.message());
          state->verify = std::move(verify.state);
          auto controller = er::cuda::create_deepseek_decode_controller(
              state->state, directory_, state->stream);
          require(controller.status.ok() && controller.controller,
                  controller.status.ok()
                      ? "callable verify controller returned no ownership"
                      : controller.status.message());
          state->verify_controller = std::move(controller.controller);
          const auto configured =
              state->verify_controller->configure_verify(state->verify);
          require(configured.ok(), configured.message());
        }
        const auto hidden = bundle_.descriptor.hidden_size;
        const auto route_width = routed_->component().route_width;
        cuda_check(cudaHostAlloc(
                       reinterpret_cast<void**>(&state->remote_input_host),
                       static_cast<std::size_t>(hidden) * sizeof(float),
                       cudaHostAllocPortable),
                   "allocate callable remote expert input");
        cuda_check(cudaHostAlloc(
                       reinterpret_cast<void**>(&state->remote_outputs_host),
                       static_cast<std::size_t>(route_width) * hidden *
                           sizeof(float),
                       cudaHostAllocPortable),
                   "allocate callable remote expert outputs");
        cuda_check(cudaHostAlloc(
                       reinterpret_cast<void**>(&state->route_prediction_host),
                       static_cast<std::size_t>(route_width) *
                           sizeof(std::uint32_t),
                       cudaHostAllocPortable),
                   "allocate callable route prediction output");
        cuda_check(cudaEventCreateWithFlags(&state->remote_input_ready_event,
                                            cudaEventDisableTiming),
                   "create callable remote input event");
        cuda_check(cudaEventCreateWithFlags(
                       &state->route_prediction_ready_event,
                       cudaEventDisableTiming),
                   "create callable route prediction event");
        cuda_check(cudaEventCreate(&state->gpu_selection_started_event),
                   "create callable GPU selection start event");
        cuda_check(cudaEventCreate(&state->gpu_selection_finished_event),
                   "create callable GPU selection finish event");
        auto workspace = directory_->create_plan_workspace();
        require(workspace.status.ok() && workspace.workspace,
                workspace.status.ok()
                    ? "DeepSeek directory returned no callable workspace"
                    : workspace.status.message());
        state->directory_workspace = std::move(workspace.workspace);
      } catch (...) {
        state.reset();
        throw;
      }
      return {er::Status::success(), std::move(state)};
    } catch (const std::exception& error) {
      return {{er::ErrorCode::internal, error.what()}, {}};
    }
  }

  [[nodiscard]] bool supports_program_sequence(
      const er::CompiledModelProgram& program) const noexcept override {
    try {
      const auto layers = bundle_.descriptor.layer_program.size();
      if (layers == 0U || program.layers.size() != layers ||
          program.operations.size() != 2U + 3U * layers ||
          program.inputs.size() != 2U || program.outputs.size() != 1U)
        return false;
      bool token_input = false;
      bool position_input = false;
      for (const auto& endpoint : program.inputs) {
        if (endpoint.value_index >= program.values.size()) return false;
        const auto& abi = program.values[endpoint.value_index].abi;
        if (abi == token_abi && !token_input)
          token_input = true;
        else if (abi == position_abi && !position_input)
          position_input = true;
        else
          return false;
      }
      const auto output = program.outputs.front().value_index;
      return token_input && position_input && output < program.values.size() &&
             program.values[output].abi == token_abi;
    } catch (...) {
      return false;
    }
  }

  er::OperationExecutionHandle execute_program_sequence(
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      const er::ProgramSequenceInvocation& invocation) override {
    try {
      auto request =
          std::dynamic_pointer_cast<CallableRequestState>(opaque_state);
      if (!request || request->owner.get() != this ||
          !supports_program_sequence(invocation.program) ||
          invocation.operations.size() != invocation.program.operations.size() ||
          invocation.inputs.size() != invocation.program.inputs.size() ||
          request->embedded || request->attention_ready || request->route_ready)
        throw std::runtime_error(
            "compressed sparse program-sequence contract is invalid");

      auto sequence = std::make_shared<CallableProgramSequence>();
      sequence->request = std::move(request);
      sequence->context = invocation.request;
      sequence->operations.reserve(invocation.operations.size());
      for (const auto* opaque : invocation.operations) {
        const auto* prepared = dynamic_cast<const PreparedOperation*>(opaque);
        if (!prepared)
          throw std::runtime_error(
              "compressed sparse program-sequence operation is foreign");
        sequence->operations.push_back(prepared);
      }
      const auto layer_count = sequence->request->state->layer_count();
      if (sequence->operations.front()->kernel != kCallableEmbedding ||
          sequence->operations.back()->kernel != kCallableHead ||
          sequence->operations.size() != 2U + 3U * layer_count)
        throw std::runtime_error(
            "compressed sparse program-sequence order is invalid");
      for (std::uint32_t layer = 0U; layer < layer_count; ++layer) {
        const auto offset = 1U + 3U * layer;
        const auto* attention = sequence->operations[offset];
        const auto* router = sequence->operations[offset + 1U];
        const auto* moe = sequence->operations[offset + 2U];
        if (attention->kernel != kCallableAttention ||
            router->kernel != kCallableRouter ||
            moe->kernel != kCallableRoutedMoe ||
            attention->logical_layer >= layer_count ||
            router->component_layer >= layer_count ||
            router->component_layer != moe->component_layer)
          throw std::runtime_error(
              "compressed sparse program-sequence layer is invalid");
      }

      const er::ExecutionValue* token_value{};
      const er::ExecutionValue* position_value{};
      for (std::size_t index = 0U; index < invocation.inputs.size(); ++index) {
        const auto value_index = invocation.program.inputs[index].value_index;
        if (value_index >= invocation.program.values.size())
          throw std::runtime_error(
              "compressed sparse program-sequence input is invalid");
        const auto& abi = invocation.program.values[value_index].abi;
        if (abi == token_abi)
          token_value = &invocation.inputs[index];
        else if (abi == position_abi)
          position_value = &invocation.inputs[index];
      }
      sequence->tokens = host_u32_batch(
          token_value, token_abi, "DeepSeek sequence tokens", max_context_);
      sequence->positions = host_u32_batch(
          position_value, position_abi, "DeepSeek sequence positions",
          max_context_);
      if (sequence->tokens.size() != sequence->positions.size() ||
          sequence->tokens.empty() ||
          sequence->positions.front() != sequence->request->next_position ||
          sequence->positions.back() >= sequence->request->context_limit ||
          std::any_of(sequence->tokens.begin(), sequence->tokens.end(),
                      [this](std::uint32_t token) {
                        return token >= bundle_.descriptor.vocab_size;
                      }))
        throw std::runtime_error(
            "compressed sparse program-sequence rows are invalid");
      for (std::size_t row = 1U; row < sequence->positions.size(); ++row)
        if (sequence->positions[row] != sequence->positions.front() + row)
          throw std::runtime_error(
              "compressed sparse program-sequence positions are not contiguous");

      const auto exact = invocation.request.parameters.find(
          "exact_decode_enabled");
      sequence->request->exact_decode_enabled =
          exact != invocation.request.parameters.end() && exact->second != 0U;
      sequence->request->sequence_sync_successors.clear();
      sequence->request->sequence_final_target_streams.clear();
      sequence->request->sequence_sync_consumed = 0U;
      sequence->worker = std::thread([sequence] {
        try {
          sequence->publish(
              sequence->request->owner->run_callable_program_sequence(
                  *sequence));
        } catch (const std::exception& error) {
          sequence->publish({{er::ErrorCode::internal, error.what()}, {}});
        } catch (...) {
          sequence->publish(
              {{er::ErrorCode::internal,
                "compressed sparse program-sequence failed"}, {}});
        }
      });
      return er::OperationExecutionHandle::from_callbacks(
          [sequence] { return sequence->poll(); },
          [sequence] {
            sequence->cancelled.store(true, std::memory_order_release);
          });
    } catch (const std::exception& error) {
      return completed_operation(
          {{er::ErrorCode::invalid_argument, error.what()}, {}});
    }
  }

  er::OperationExecutionHandle execute(
      const er::IPreparedOperation& opaque_operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& opaque_state,
      const er::OperationInvocation& invocation) override {
    const auto* prepared =
        dynamic_cast<const PreparedOperation*>(&opaque_operation);
    const auto state =
        std::dynamic_pointer_cast<CallableRequestState>(opaque_state);
    if (!prepared || !state || state->owner.get() != this)
      return completed_operation(
          {{er::ErrorCode::invalid_argument,
            "compressed sparse invocation ownership is invalid"}, {}});
    try {
      const auto input = [&](std::string_view port) -> const er::ExecutionValue& {
        const auto found = std::find_if(
            invocation.operation.input_values.begin(),
            invocation.operation.input_values.end(),
            [port](const auto& item) { return item.port == port; });
        if (found == invocation.operation.input_values.end())
          throw std::runtime_error("callable DeepSeek input port is absent");
        const auto index = static_cast<std::size_t>(
            found - invocation.operation.input_values.begin());
        if (index >= invocation.inputs.size())
          throw std::runtime_error("callable DeepSeek input index is invalid");
        return invocation.inputs[index];
      };
      switch (prepared->kernel) {
        case kCallableEmbedding: {
          const auto token = host_u32(input("token_ids"), token_abi,
                                      "DeepSeek token");
          if (token >= bundle_.descriptor.vocab_size || state->embedded)
            throw std::runtime_error("callable DeepSeek token is invalid");
          auto status = state->state->embed(token, state->stream);
          require(status.ok(), status.message());
          state->current_token = token;
          state->embedded = true;
          state->attention_ready = false;
          state->route_ready = false;
          return completed_outputs(
              *prepared, {{"hidden", hca_value(state, true)}});
        }
        case kCallableAttention: {
          require_hca(input("hidden"), state, true);
          const auto position = host_u32(input("positions"), position_abi,
                                         "DeepSeek position");
          if (!state->embedded || position != state->next_position ||
              position >= state->context_limit ||
              prepared->logical_layer >= state->state->layer_count())
            throw std::runtime_error(
                "callable DeepSeek attention sequence is invalid");
          auto status = execute_callable_attention(
              *state, prepared->logical_layer, position);
          require(status.ok(), status.message());
          state->current_position = position;
          state->attention_ready = true;
          state->route_ready = false;
          return completed_outputs(
              *prepared, {{"hidden", hca_value(state, false)}});
        }
        case kCallableRouter: {
          require_hca(input("hidden"), state, false);
          const auto token = host_u32(input("token_ids"), token_abi,
                                      "DeepSeek router token");
          if (!state->attention_ready || token != state->current_token ||
              prepared->component_layer >= state->state->layer_count())
            throw std::runtime_error("callable DeepSeek router state is invalid");
          const auto view = state->state->layer(prepared->component_layer);
          auto status = er::cuda::deepseek_ffn_route(
              {view.ffn_weights, view.ffn_state,
               state->state->attention_streams(), token, 1e-6F, 20U,
               state->stream});
          require(status.ok(), status.message());
          state->route_ready = true;
          state->current_router_component_layer = prepared->component_layer;
          state->current_prefetch_target_layer =
              prepared->prefetch_target_component_layer;
          std::map<std::string, er::ExecutionValue, std::less<>> outputs;
          outputs.emplace("expert_input",
                          device_value(state, view.ffn_state->normalized_input(),
                                       bundle_.descriptor.hidden_size,
                                       hidden_abi));
          outputs.emplace("route_indices",
                          device_value(state, view.ffn_state->expert_indices(),
                                       routed_->component().route_width,
                                       route_index_abi));
          outputs.emplace("route_weights",
                          device_value(state, view.ffn_state->routing_weights(),
                                       routed_->component().route_width,
                                       route_weight_abi));
          outputs.emplace("residual", hca_value(state, false));
          return completed_outputs(*prepared, std::move(outputs));
        }
        case kCallableRoutedMoe: {
          require_hca(input("residual"), state, false);
          const auto view = state->state->layer(prepared->component_layer);
          require_device_value(input("expert_input"),
                               view.ffn_state->normalized_input(),
                               bundle_.descriptor.hidden_size * sizeof(float),
                               hidden_abi);
          require_device_value(input("route_indices"),
                               view.ffn_state->expert_indices(),
                               routed_->component().route_width *
                                   sizeof(std::uint32_t),
                               route_index_abi);
          require_device_value(input("route_weights"),
                               view.ffn_state->routing_weights(),
                               routed_->component().route_width * sizeof(float),
                               route_weight_abi);
          if (!state->route_ready)
            throw std::runtime_error(
                "callable DeepSeek routed operation has no exact route");
          return start_callable_routed(*prepared, std::move(state),
                                       invocation.request);
        }
        case kCallableHead: {
          require_hca(input("hidden"), state, true);
          if (!state->embedded || !state->attention_ready ||
              !state->route_ready)
            throw std::runtime_error("callable DeepSeek head state is invalid");
          auto status = state->state->project_logits(state->stream);
          require(status.ok(), status.message());
          auto token = std::make_shared<std::uint32_t>();
          cuda_check(cudaMemcpyAsync(token.get(), state->state->sampled_token(),
                                     sizeof(*token), cudaMemcpyDeviceToHost,
                                     state->stream),
                     "queue callable DeepSeek sampled token copy");
          cuda_check(cudaStreamSynchronize(state->stream),
                     "complete callable DeepSeek token step");
          state->predicted = *token;
          ++state->next_position;
          if (state->retention_target_position == state->next_position) {
            const auto checkpoint = capture_retention_checkpoint(
                *state, state->next_position, false);
            require(checkpoint.ok(), checkpoint.message());
          }
          state->embedded = false;
          er::ExecutionValue output{std::string(token_abi), "host", token,
                                    reinterpret_cast<const std::byte*>(
                                        token.get()),
                                    sizeof(*token)};
          return completed_outputs(
              *prepared, {{"token_ids", std::move(output)}});
        }
        default:
          throw std::runtime_error("callable DeepSeek kernel is unsupported");
      }
    } catch (const std::exception& error) {
      return completed_operation(
          {{er::ErrorCode::internal, error.what()}, {}});
    }
  }

  std::unique_ptr<Request> create_request() {
    auto state = er::cuda::create_deepseek_request_state(
        model_, {max_context_, request_bytes_, compression_ratios_,
                 hash_router_layers_});
    require(state.status.ok() && state.state, state.status.message());
    auto request = std::make_unique<Request>();
    request->state = std::move(state.state);
    request->prompt_routes.resize(routed_->component().layer_count);
    if (mtp_enabled_) {
      auto mtp_state = er::cuda::create_deepseek_mtp_request_state(
          model_, mtp_model_, {max_context_, mtp_request_bytes_});
      require(mtp_state.status.ok() && mtp_state.state,
              mtp_state.status.ok() ? "MTP request returned no ownership"
                                    : mtp_state.status.message());
      request->mtp = std::move(mtp_state.state);
    }
    cuda_check(cudaStreamCreateWithFlags(&request->stream,
                                         cudaStreamNonBlocking),
               "create DeepSeek request stream");
    auto controller = er::cuda::create_deepseek_decode_controller(
        request->state, directory_, request->stream);
    require(controller.status.ok() && controller.controller,
            controller.status.message());
    request->controller = std::move(controller.controller);
    if (mtp_enabled_) {
      auto verify = er::cuda::create_deepseek_verify_state(
          request->state, verify_request_bytes_);
      require(verify.status.ok() && verify.state,
              verify.status.ok() ? "verify state returned no ownership"
                                 : verify.status.message());
      request->verify = std::move(verify.state);
      const auto configured =
          request->controller->configure_verify(request->verify);
      require(configured.ok(), configured.message());
    }
    if (cpu_hybrid_enabled_) {
      auto workspace = er::cuda::create_deepseek_ffn_hybrid_workspace();
      require(workspace.status.ok() && workspace.workspace,
              workspace.status.ok()
                  ? "hybrid workspace returned no ownership"
                  : workspace.status.message());
      const auto configured = request->controller->configure_hybrid(
          cpu_, std::move(workspace.workspace));
      require(configured.ok(), configured.message());
    }
    if (gpu_phase_timing_) {
      const auto timing = request->controller->enable_gpu_phase_timing();
      require(timing.ok(), timing.message());
    }
    return request;
  }

  void observe_prefill_routes(Request& request) {
    for (const auto& trace : request.controller->route_trace()) {
      require(trace.layer < routed_->component().layer_count,
              "prefill route trace has an invalid layer");
      for (const auto expert : trace.routed_experts) {
        auto& evidence = request.prompt_routes[trace.layer][expert];
        if (evidence.count != std::numeric_limits<std::uint32_t>::max())
          ++evidence.count;
        evidence.last_seen = ++request.prompt_route_clock;
      }
    }
  }

  void protect_prefill_routes(Request& request) {
    const auto maximum_per_layer = routed_->component().route_width;
    for (std::uint32_t layer = 0U;
         layer < routed_->component().layer_count;
         ++layer) {
      std::vector<std::pair<std::uint32_t, PromptRouteEvidence>> ranked(
          request.prompt_routes[layer].begin(),
          request.prompt_routes[layer].end());
      std::sort(ranked.begin(), ranked.end(), [](const auto& left,
                                                 const auto& right) {
        if (left.second.count != right.second.count)
          return left.second.count > right.second.count;
        if (left.second.last_seen != right.second.last_seen)
          return left.second.last_seen > right.second.last_seen;
        return left.first < right.first;
      });
      if (ranked.size() > maximum_per_layer)
        ranked.resize(maximum_per_layer);
      for (const auto& [expert, evidence] : ranked) {
        (void)evidence;
        prefill_protection_candidates_.fetch_add(1U,
                                                  std::memory_order_relaxed);
        const auto key = routed_->key(layer, expert);
        static_cast<void>(cache_->protect(key, true, true));
        const auto snapshot = cache_->inspect(key);
        if (snapshot &&
            (!snapshot->has_host_copy || snapshot->ram_protected) &&
            (!snapshot->has_device_copy || snapshot->vram_resident)) {
          prefill_protection_promoted_.fetch_add(1U,
                                                 std::memory_order_relaxed);
        }
      }
      request.prompt_routes[layer].clear();
    }
    request.prompt_route_clock = 0U;
  }

  std::vector<std::uint32_t> forward(
      std::span<Request* const> requests,
      std::span<const std::uint32_t> tokens,
      std::span<const std::uint32_t> positions) {
    DemandActivity demand(demand_depth_);
    require(!requests.empty() && requests.size() == tokens.size() &&
                requests.size() == positions.size() &&
                requests.size() <= capacity_,
            "invalid DeepSeek worker batch");
    const auto model_step_started = std::chrono::steady_clock::now();
    const auto submit_started = model_step_started;
    std::vector<std::uint64_t> operations;
    operations.reserve(requests.size());
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      require(positions[index] < requests[index]->context_limit,
              "DeepSeek reserved context exhausted");
      const auto embedded = requests[index]->state->embed(
          tokens[index], requests[index]->stream);
      require(embedded.ok(), embedded.message());
      const auto operation = next_operation_++;
      const auto submitted = scheduler_->submit(
          operation, requests[index]->controller,
          {requests[index]->state->current_streams(),
           rope_at(positions[index]),
           positions[index], tokens[index], 0U,
           routed_->component().layer_count});
      require(submitted.ok(), submitted.message());
      operations.push_back(operation);
    }
    telemetry_.embed_rope_submit_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - submit_started)
            .count());
    const auto scheduler_started = std::chrono::steady_clock::now();
    std::size_t completed = 0U;
    while (completed != operations.size()) {
      const auto polled = scheduler_->poll();
      require(polled.ok(), polled.message());
      completed = 0U;
      for (const auto operation : operations) {
        const auto state = scheduler_->inspect(operation);
        require(state.has_value(), "DeepSeek scheduled operation disappeared");
        require(state->state != er::cuda::DeepSeekScheduledState::failed &&
                    state->state != er::cuda::DeepSeekScheduledState::cancelled,
                state->status.message());
        if (state->state == er::cuda::DeepSeekScheduledState::complete)
          ++completed;
      }
      if (completed != operations.size()) {
        const auto waited = scheduler_->wait_for_cuda_progress();
        require(waited.ok(), waited.message());
        if (scheduler_->snapshot().cuda_pending_requests == 0U)
          std::this_thread::yield();
      }
    }
    telemetry_.scheduler_poll_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - scheduler_started)
            .count());
    for (auto* request : requests) {
      const auto current = request->controller->telemetry();
      telemetry_.attention_route_submit_ns +=
          current.attention_route_submit_ns -
          request->controller_telemetry.attention_route_submit_ns;
      telemetry_.directory_plan_ns +=
          current.directory_plan_ns -
          request->controller_telemetry.directory_plan_ns;
      telemetry_.ffn_submit_ns +=
          current.ffn_submit_ns - request->controller_telemetry.ffn_submit_ns;
      telemetry_.directory_release_ns +=
          current.directory_release_ns -
          request->controller_telemetry.directory_release_ns;
      telemetry_.gpu_attention_route_plan_ns +=
          current.gpu_attention_route_plan_ns -
          request->controller_telemetry.gpu_attention_route_plan_ns;
      telemetry_.gpu_ffn_release_ns +=
          current.gpu_ffn_release_ns -
          request->controller_telemetry.gpu_ffn_release_ns;
      telemetry_.gpu_attention_ns +=
          current.gpu_attention_ns -
          request->controller_telemetry.gpu_attention_ns;
      telemetry_.gpu_route_ns +=
          current.gpu_route_ns - request->controller_telemetry.gpu_route_ns;
      telemetry_.gpu_directory_plan_ns +=
          current.gpu_directory_plan_ns -
          request->controller_telemetry.gpu_directory_plan_ns;
      telemetry_.gpu_ffn_ns +=
          current.gpu_ffn_ns - request->controller_telemetry.gpu_ffn_ns;
      telemetry_.gpu_directory_release_ns +=
          current.gpu_directory_release_ns -
          request->controller_telemetry.gpu_directory_release_ns;
      telemetry_.gpu_attention_hca_pre_norm_ns +=
          current.gpu_attention_hca_pre_norm_ns -
          request->controller_telemetry.gpu_attention_hca_pre_norm_ns;
      telemetry_.gpu_attention_projection_ns +=
          current.gpu_attention_projection_ns -
          request->controller_telemetry.gpu_attention_projection_ns;
      telemetry_.gpu_sparse_attention_ns +=
          current.gpu_sparse_attention_ns -
          request->controller_telemetry.gpu_sparse_attention_ns;
      telemetry_.gpu_attention_output_projection_ns +=
          current.gpu_attention_output_projection_ns -
          request->controller_telemetry.gpu_attention_output_projection_ns;
      telemetry_.gpu_attention_hca_post_ns +=
          current.gpu_attention_hca_post_ns -
          request->controller_telemetry.gpu_attention_hca_post_ns;
      telemetry_.gpu_ffn_routed_ns +=
          current.gpu_ffn_routed_ns -
          request->controller_telemetry.gpu_ffn_routed_ns;
      telemetry_.gpu_ffn_aggregate_ns +=
          current.gpu_ffn_aggregate_ns -
          request->controller_telemetry.gpu_ffn_aggregate_ns;
      telemetry_.gpu_ffn_shared_ns +=
          current.gpu_ffn_shared_ns -
          request->controller_telemetry.gpu_ffn_shared_ns;
      telemetry_.gpu_ffn_merge_ns +=
          current.gpu_ffn_merge_ns -
          request->controller_telemetry.gpu_ffn_merge_ns;
      telemetry_.gpu_ffn_hca_post_ns +=
          current.gpu_ffn_hca_post_ns -
          request->controller_telemetry.gpu_ffn_hca_post_ns;
      request->controller_telemetry = current;
    }
    for (std::size_t index = 0U; index < requests.size(); ++index)
      capture_route_step(*requests[index], positions[index], 1U);
    const auto output_started = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> result(requests.size());
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      const auto projected = requests[index]->state->project_logits(
          requests[index]->stream);
      require(projected.ok(), projected.message());
      cuda_check(cudaMemcpyAsync(&result[index],
                                 requests[index]->state->sampled_token(),
                                 sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                 requests[index]->stream),
                 "queue DeepSeek sampled token copy");
    }
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      cuda_check(cudaStreamSynchronize(requests[index]->stream),
                 "complete DeepSeek sampled token copy");
      const auto retired = scheduler_->retire(operations[index]);
      require(retired.ok(), retired.message());
    }
    telemetry_.output_head_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - output_started)
            .count());
    ++telemetry_.model_steps;
    telemetry_.model_rows += requests.size();
    telemetry_.model_step_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - model_step_started)
            .count());
    const auto elapsed_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - model_step_started)
            .count());
    update_moving_average(ordinary_ns_per_token_, ordinary_samples_,
                          elapsed_ns / requests.size());
    return result;
  }

  template <typename RequestState>
  void advance_mtp_state(RequestState& request, std::uint32_t next_token,
                         const float* target_streams,
                         std::uint32_t position, bool produce_draft) {
    DemandActivity demand(demand_depth_);
    require(mtp_enabled_ && request.mtp && mtp_cache_ && mtp_directory_,
            "MTP advance requires an enabled request runtime");
    const auto rope = rope_at(position);
    auto status = request.mtp->prepare(
        next_token, target_streams, position, rope.base_cosine,
        rope.base_sine, request.stream);
    require(status.ok(), status.message());
    if (!produce_draft) {
      status = request.mtp->abandon_draft();
      require(status.ok(), status.message());
      request.draft_ready = false;
      return;
    }

    std::vector<er::ExpertLease> leases;
    for (;;) {
      auto plan = mtp_directory_->pin_or_collect_misses(
          0U, request.mtp->expert_indices(),
          request.mtp->selection_count(), request.stream, true);
      require(plan.status.ok(), plan.status.message());
      if (plan.missing_experts.empty()) {
        require(plan.pin_id != 0U,
                "MTP directory returned no execution pin");
        status = request.mtp->complete(
            mtp_directory_->device_entries(),
            mtp_routed_->component().experts_per_layer +
                mtp_routed_->component().shared_experts_per_layer,
            request.stream);
        require(status.ok(), status.message());
        cuda_check(cudaMemcpyAsync(&request.draft,
                                   request.mtp->draft_token(),
                                   sizeof(request.draft),
                                   cudaMemcpyDeviceToHost, request.stream),
                   "copy MTP draft token");
        status = mtp_directory_->release_pins_async(plan.pin_id,
                                                    request.stream);
        require(status.ok(), status.message());
        cuda_check(cudaStreamSynchronize(request.stream),
                   "complete MTP draft");
        break;
      }
      if (plan.pin_id != 0U) {
        status = mtp_directory_->release_pins(plan.pin_id, request.stream);
        require(status.ok(), status.message());
      }
      struct PendingMtp final {
        std::uint32_t expert{};
        er::AcquireHandle handle;
      };
      std::vector<PendingMtp> pending;
      pending.reserve(plan.missing_experts.size());
      const auto acquire_started = std::chrono::steady_clock::now();
      for (const auto expert : plan.missing_experts) {
        require(expert < mtp_routed_->component().experts_per_layer,
                "always-resident MTP shared expert is unavailable");
        const auto* record = mtp_routed_->record(0U, expert);
        require(record != nullptr,
                "MTP route references an absent catalog expert");
        pending.push_back(
            {expert,
             mtp_cache_->acquire(
                 mtp_routed_->key(0U, expert), *record,
                 er::ExpertAcquireOptions{er::ExpertRequestPriority::demand,
                                          true, true, false})});
      }
      ++telemetry_.mtp_acquire_batches;
      telemetry_.mtp_acquires_launched += pending.size();
      telemetry_.mtp_acquire_batch_width_max = std::max<std::uint64_t>(
          telemetry_.mtp_acquire_batch_width_max, pending.size());
      for (std::size_t index = 0U; index < pending.size(); ++index) {
        while (pending[index].handle.wait_for(std::chrono::milliseconds(1)) !=
               std::future_status::ready)
          std::this_thread::yield();
        auto acquired = pending[index].handle.get();
        if (!acquired.status.ok() || !acquired.lease) {
          for (std::size_t cancel = index + 1U; cancel < pending.size();
               ++cancel) {
            pending[cancel].handle.cancel();
          }
          require(false,
                  acquired.status.ok() ? "MTP cache returned no device lease"
                                       : acquired.status.message());
        }
        leases.push_back(std::move(acquired.lease));
      }
      telemetry_.mtp_acquire_wait_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - acquire_started)
              .count());
    }
    request.draft_ready = true;
    ++telemetry_.mtp_drafts;
  }

  void advance_mtp(Request& request, std::uint32_t next_token,
                   const float* target_streams, std::uint32_t position,
                   bool produce_draft) {
    advance_mtp_state(request, next_token, target_streams, position,
                      produce_draft);
  }

  void capture_verify_route(Request& request, std::uint32_t position) {
    capture_route_step(request, position, 2U);
  }

  void capture_verify_route(CallableRequestState&,
                            std::uint32_t) noexcept {}

  template <typename RequestState>
  std::vector<std::uint32_t> verify_draft_state(
      RequestState& request,
      const std::shared_ptr<er::cuda::DeepSeekDecodeController>& controller) {
    DemandActivity demand(demand_depth_);
    require(mtp_enabled_ && request.verify && request.draft_ready &&
                request.next_position + 1U < request.context_limit,
            "invalid DeepSeek speculative verification request");
    const auto started = std::chrono::steady_clock::now();
    const auto guaranteed = request.predicted;
    const auto draft = request.draft;
    const auto position = request.next_position;
    auto status = request.state->embed(guaranteed, request.stream);
    require(status.ok(), status.message());
    status = request.verify->embed_speculative(draft, request.stream);
    require(status.ok(), status.message());
    const auto operation = next_operation_++;
    status = scheduler_->submit_verify(
        operation, controller,
        {{rope_at(position), rope_at(position + 1U)},
         {position, position + 1U}, {guaranteed, draft}, 0U,
         routed_->component().layer_count});
    require(status.ok(), status.message());
    for (;;) {
      status = scheduler_->poll();
      require(status.ok(), status.message());
      const auto scheduled = scheduler_->inspect(operation);
      require(scheduled.has_value(),
              "DeepSeek verify operation disappeared");
      require(scheduled->state != er::cuda::DeepSeekScheduledState::failed &&
                  scheduled->state !=
                      er::cuda::DeepSeekScheduledState::cancelled,
              scheduled->status.message());
      if (scheduled->state == er::cuda::DeepSeekScheduledState::complete)
        break;
      if (scheduler_->snapshot().cuda_pending_requests != 0U) {
        status = scheduler_->wait_for_cuda_progress();
        require(status.ok(), status.message());
      } else {
        std::this_thread::yield();
      }
    }
    capture_verify_route(request, position);
    status = request.verify->project_pair_logits(request.stream);
    require(status.ok(), status.message());
    std::array<std::uint32_t, 2U> sampled{};
    cuda_check(cudaMemcpyAsync(
                   sampled.data(), request.verify->primary_sampled_token(),
                   sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                   request.stream),
               "copy DeepSeek verifier token");
    cuda_check(cudaMemcpyAsync(
                   sampled.data() + 1U,
                   request.verify->bonus_sampled_token(),
                   sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                   request.stream),
               "copy DeepSeek verifier bonus token");
    cuda_check(cudaStreamSynchronize(request.stream),
               "complete DeepSeek pair vocabulary heads");

    const bool accepted = sampled[0] == draft;
    if (accepted) {
      // Bring the MTP causal cache through H_A/B before H_B/C becomes the next
      // draft boundary. The first result is intentionally abandoned because
      // the target verifier already produced the bonus token C.
      advance_mtp_state(request, draft, request.state->current_streams(),
                        position, false);
      status = request.verify->finish_transaction(true, request.stream);
      require(status.ok(), status.message());
      advance_mtp_state(request, sampled[1],
                        request.state->current_streams(), position + 1U,
                        true);
      request.predicted = sampled[1];
      request.next_position += 2U;
      ++telemetry_.mtp_accepted;
    } else {
      status = request.verify->finish_transaction(false, request.stream);
      require(status.ok(), status.message());
      advance_mtp_state(request, sampled[0],
                        request.state->current_streams(), position, true);
      request.predicted = sampled[0];
      ++request.next_position;
      ++telemetry_.mtp_rejected;
    }
    status = scheduler_->retire(operation);
    require(status.ok(), status.message());
    ++telemetry_.model_steps;
    telemetry_.model_rows += 2U;
    ++telemetry_.verify_pairs;
    telemetry_.model_step_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    const auto elapsed_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    update_moving_average(speculative_ns_per_useful_, speculative_samples_,
                          elapsed_ns / (accepted ? 2.0 : 1.0));
    // Suppression needs a real acceptance sample: with the EMA seeded from a
    // single pair, any one rejection trips the guard (a rejected pair scores
    // roughly 2x an ordinary step per useful token), so a sub-8-sample check
    // disables speculation on noise before the rate is measurable.
    if (speculative_samples_ >= 8U && ordinary_samples_ != 0U &&
        speculative_ns_per_useful_ > ordinary_ns_per_token_ * 1.05) {
      request.speculation_suppressed = true;
      request.draft_ready = false;
      ++telemetry_.mtp_suppressions;
    }
    return accepted ? std::vector<std::uint32_t>{guaranteed, draft}
                    : std::vector<std::uint32_t>{guaranteed};
  }

  std::vector<std::uint32_t> verify_draft(Request& request) {
    return verify_draft_state(request, request.controller);
  }

  std::uint32_t capacity() const noexcept { return capacity_; }
  std::uint32_t max_context() const noexcept { return max_context_; }
  std::uint32_t vocab_size() const noexcept {
    return bundle_.descriptor.vocab_size;
  }
  const er::ModelDescriptor& descriptor() const noexcept {
    return bundle_.descriptor;
  }
  std::uint64_t ram_bytes() const noexcept { return ram_bytes_; }
  std::uint64_t vram_bytes() const noexcept { return vram_bytes_; }
  std::uint32_t kv_page_tokens() const noexcept { return kv_page_tokens_; }
  std::uint32_t sequence_tile_rows() const noexcept {
    return sequence_tile_rows_;
  }
  std::uint64_t kv_page_bytes() const noexcept { return kv_page_bytes_; }
  std::uint64_t kv_page_capacity() const noexcept {
    return kv_page_capacity_;
  }
  std::uint64_t kv_pages_per_request() const noexcept {
    return (max_context_ + kv_page_tokens_ - 1U) / kv_page_tokens_;
  }
  const std::string& placement() const noexcept { return placement_; }
  bool retained_route_enabled() const noexcept {
    return retain_previous_route_;
  }
  bool cpu_hybrid_enabled() const noexcept { return cpu_hybrid_enabled_; }
  er::cuda::DeepSeekDecodeSchedulerSnapshot scheduler_snapshot() const {
    return scheduler_->snapshot();
  }
  er::RouteCensusSnapshot census_snapshot() const { return census_->snapshot(); }
  er::TelemetrySnapshot cache_snapshot() const { return cache_->telemetry(); }
  er::TelemetrySnapshot mtp_cache_snapshot() const {
    return mtp_cache_ ? mtp_cache_->telemetry() : er::TelemetrySnapshot{};
  }
  er::BufferPoolSnapshot buffer_snapshot() const {
    return buffers_->snapshot();
  }
  er::cuda::CudaExpertUploaderTelemetry uploader_snapshot() const {
    return uploader_->telemetry();
  }
  er::cpu::DeepSeekPackedExecutorTelemetry cpu_snapshot() const {
    return cpu_->telemetry();
  }
  er::HybridDispatchTelemetry planner_snapshot() const {
    return planner_->telemetry();
  }
  WorkerTelemetry worker_snapshot() const noexcept {
    auto result = telemetry_;
    result.callable_selection_launches =
        callable_selection_launches_.load(std::memory_order_relaxed);
    result.callable_selections =
        callable_selections_.load(std::memory_order_relaxed);
    result.callable_overlap_launches =
        callable_overlap_launches_.load(std::memory_order_relaxed);
    result.callable_cpu_launches =
        callable_cpu_launches_.load(std::memory_order_relaxed);
    result.callable_cpu_selections =
        callable_cpu_selections_.load(std::memory_order_relaxed);
    result.callable_cpu_activation_bytes =
        callable_cpu_activation_bytes_.load(std::memory_order_relaxed);
    result.callable_cpu_output_bytes =
        callable_cpu_output_bytes_.load(std::memory_order_relaxed);
    result.callable_cpu_host_resolves_launched =
        callable_cpu_host_resolves_launched_.load(std::memory_order_relaxed);
    result.callable_cpu_host_resolves_completed =
        callable_cpu_host_resolves_completed_.load(std::memory_order_relaxed);
    result.callable_cpu_device_admission_fallbacks =
        callable_cpu_device_admission_fallbacks_.load(
            std::memory_order_relaxed);
    result.callable_resolves_launched =
        callable_resolves_launched_.load(std::memory_order_relaxed);
    result.callable_resolves_completed =
        callable_resolves_completed_.load(std::memory_order_relaxed);
    result.callable_stale_plan_replans =
        callable_stale_plan_replans_.load(std::memory_order_relaxed);
    result.callable_prefetch_routes_launched =
        callable_prefetch_routes_launched_.load(std::memory_order_relaxed);
    result.callable_prefetch_routes_completed =
        callable_prefetch_routes_completed_.load(std::memory_order_relaxed);
    result.callable_prefetch_prediction_wait_ns =
        callable_prefetch_prediction_wait_ns_.load(
            std::memory_order_relaxed);
    result.callable_prefetch_candidates =
        callable_prefetch_candidates_.load(std::memory_order_relaxed);
    result.callable_prefetch_vram_candidates =
        callable_prefetch_vram_candidates_.load(std::memory_order_relaxed);
    result.callable_prefetch_ram_candidates =
        callable_prefetch_ram_candidates_.load(std::memory_order_relaxed);
    result.callable_prefetch_storage_candidates =
        callable_prefetch_storage_candidates_.load(std::memory_order_relaxed);
    result.callable_prefetch_storage_selected =
        callable_prefetch_storage_selected_.load(std::memory_order_relaxed);
    result.callable_prefetch_storage_incorrect =
        callable_prefetch_storage_incorrect_.load(std::memory_order_relaxed);
    result.callable_prefetch_resolves_launched =
        callable_prefetch_resolves_launched_.load(std::memory_order_relaxed);
    result.callable_prefetch_resolves_completed =
        callable_prefetch_resolves_completed_.load(std::memory_order_relaxed);
    result.callable_prefetch_selected =
        callable_prefetch_selected_.load(std::memory_order_relaxed);
    result.callable_prefetch_useful =
        callable_prefetch_useful_.load(std::memory_order_relaxed);
    result.callable_prefetch_late =
        callable_prefetch_late_.load(std::memory_order_relaxed);
    result.callable_prefetch_incorrect =
        callable_prefetch_incorrect_.load(std::memory_order_relaxed);
    result.callable_prefetch_cancelled =
        callable_prefetch_cancelled_.load(std::memory_order_relaxed);
    result.callable_prefetch_errors =
        callable_prefetch_errors_.load(std::memory_order_relaxed);
    result.callable_remote_resolves =
        callable_remote_resolves_.load(std::memory_order_relaxed);
    result.callable_remote_selections_launched =
        callable_remote_selections_launched_.load(std::memory_order_relaxed);
    result.callable_remote_selections_completed =
        callable_remote_selections_completed_.load(std::memory_order_relaxed);
    result.callable_remote_errors =
        callable_remote_errors_.load(std::memory_order_relaxed);
    result.callable_remote_cancellations =
        callable_remote_cancellations_.load(std::memory_order_relaxed);
    result.callable_remote_activation_tx_bytes =
        callable_remote_activation_tx_bytes_.load(std::memory_order_relaxed);
    result.callable_remote_activation_rx_bytes =
        callable_remote_activation_rx_bytes_.load(std::memory_order_relaxed);
    result.callable_remote_weight_tx_bytes =
        callable_remote_weight_tx_bytes_.load(std::memory_order_relaxed);
    result.callable_remote_owner_weight_read_bytes =
        callable_remote_owner_weight_read_bytes_.load(
            std::memory_order_relaxed);
    result.callable_remote_owner_storage_read_bytes =
        callable_remote_owner_storage_read_bytes_.load(
            std::memory_order_relaxed);
    result.callable_remote_owner_ram_read_bytes =
        callable_remote_owner_ram_read_bytes_.load(std::memory_order_relaxed);
    result.callable_remote_owner_vram_read_bytes =
        callable_remote_owner_vram_read_bytes_.load(std::memory_order_relaxed);
    result.callable_remote_owner_execution_ns =
        callable_remote_owner_execution_ns_.load(std::memory_order_relaxed);
    result.callable_remote_transport_wait_ns =
        callable_remote_transport_wait_ns_.load(std::memory_order_relaxed);
    result.warm_start_candidates =
        warm_candidates_count_.load(std::memory_order_relaxed);
    result.warm_start_loaded = warm_loaded_.load(std::memory_order_relaxed);
    result.warm_start_failed = warm_failed_.load(std::memory_order_relaxed);
    result.warm_start_cancelled =
        warm_cancelled_.load(std::memory_order_relaxed);
    result.warm_start_demand_pauses =
        warm_demand_pauses_.load(std::memory_order_relaxed);
    result.warm_start_inflight_max =
        warm_inflight_max_.load(std::memory_order_relaxed);
    result.warm_start_loop_errors =
        warm_loop_errors_.load(std::memory_order_relaxed);
    result.warm_start_bytes = warm_bytes_.load(std::memory_order_relaxed);
    result.warm_start_ns = warm_ns_.load(std::memory_order_relaxed);
    result.census_namespace_rebinds = census_namespace_rebound_ ? 1U : 0U;
    result.warm_vram_candidates =
        warm_vram_candidates_count_.load(std::memory_order_relaxed);
    result.warm_vram_loaded =
        warm_vram_loaded_.load(std::memory_order_relaxed);
    result.warm_vram_failed =
        warm_vram_failed_.load(std::memory_order_relaxed);
    result.warm_vram_cancelled =
        warm_vram_cancelled_.load(std::memory_order_relaxed);
    result.warm_vram_demand_pauses =
        warm_vram_demand_pauses_.load(std::memory_order_relaxed);
    result.warm_vram_inflight_max =
        warm_vram_inflight_max_.load(std::memory_order_relaxed);
    result.warm_vram_bytes =
        warm_vram_bytes_.load(std::memory_order_relaxed);
    result.warm_vram_ns = warm_vram_ns_.load(std::memory_order_relaxed);
    result.prefill_protection_candidates =
        prefill_protection_candidates_.load(std::memory_order_relaxed);
    result.prefill_protection_promoted =
        prefill_protection_promoted_.load(std::memory_order_relaxed);
    result.sequence_blocks =
        sequence_blocks_.load(std::memory_order_relaxed);
    result.sequence_rows = sequence_rows_.load(std::memory_order_relaxed);
    result.sequence_layers =
        sequence_layers_.load(std::memory_order_relaxed);
    result.sequence_attention_hca_pre_norm_ns =
        sequence_attention_hca_pre_norm_ns_.load(std::memory_order_relaxed);
    result.sequence_attention_projection_ns =
        sequence_attention_projection_ns_.load(std::memory_order_relaxed);
    result.sequence_causal_attention_ns =
        sequence_causal_attention_ns_.load(std::memory_order_relaxed);
    result.sequence_attention_output_projection_ns =
        sequence_attention_output_projection_ns_.load(
            std::memory_order_relaxed);
    result.sequence_route_ns =
        sequence_route_ns_.load(std::memory_order_relaxed);
    result.sequence_expert_wait_ns =
        sequence_expert_wait_ns_.load(std::memory_order_relaxed);
    result.sequence_expert_execute_ns =
        sequence_expert_execute_ns_.load(std::memory_order_relaxed);
    result.sequence_finalize_ns =
        sequence_finalize_ns_.load(std::memory_order_relaxed);
    return result;
  }
  void record_callable_selection_launch(std::uint32_t selections,
                                        bool overlapped) noexcept {
    callable_selection_launches_.fetch_add(1U, std::memory_order_relaxed);
    callable_selections_.fetch_add(selections, std::memory_order_relaxed);
    if (overlapped)
      callable_overlap_launches_.fetch_add(1U, std::memory_order_relaxed);
  }
  void refresh_hybrid_transfer_observation() noexcept {
    try {
      const auto snapshot = cache_->telemetry();
      std::lock_guard lock(hybrid_observation_mutex_);
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
    } catch (...) {
    }
  }
  void poll_callable_gpu_observation(CallableRequestState& request) noexcept {
    if (!request.gpu_selection_observation_pending) return;
    const auto query = cudaEventQuery(request.gpu_selection_finished_event);
    if (query == cudaErrorNotReady) return;
    if (query != cudaSuccess) {
      request.gpu_selection_observation_pending = false;
      return;
    }
    float milliseconds = 0.0F;
    if (cudaEventElapsedTime(&milliseconds, request.gpu_selection_started_event,
                             request.gpu_selection_finished_event) ==
            cudaSuccess &&
        milliseconds > 0.0F && request.gpu_selection_observation_selections) {
      planner_->observe_gpu(
          static_cast<std::uint64_t>(milliseconds * 1.0e6F),
          request.gpu_selection_observation_selections);
    }
    request.gpu_selection_observation_pending = false;
  }
  void record_remote_active_expert(
      const er::ActiveExpertExecutionEvidence& evidence) noexcept {
    callable_remote_activation_tx_bytes_.fetch_add(
        evidence.activation_input_bytes, std::memory_order_relaxed);
    callable_remote_activation_rx_bytes_.fetch_add(
        evidence.activation_output_bytes, std::memory_order_relaxed);
    callable_remote_weight_tx_bytes_.fetch_add(
        evidence.weight_transport_bytes, std::memory_order_relaxed);
    callable_remote_owner_weight_read_bytes_.fetch_add(
        evidence.owner_weight_read_bytes, std::memory_order_relaxed);
    callable_remote_owner_storage_read_bytes_.fetch_add(
        evidence.owner_storage_read_bytes, std::memory_order_relaxed);
    callable_remote_owner_ram_read_bytes_.fetch_add(
        evidence.owner_ram_read_bytes, std::memory_order_relaxed);
    callable_remote_owner_vram_read_bytes_.fetch_add(
        evidence.owner_vram_read_bytes, std::memory_order_relaxed);
    callable_remote_owner_execution_ns_.fetch_add(
        evidence.owner_execution_ns, std::memory_order_relaxed);
    callable_remote_transport_wait_ns_.fetch_add(
        evidence.transport_wait_ns, std::memory_order_relaxed);
  }
  const char* prefetch_state() const noexcept {
    if (placement_ == "capacity") return "disabled";
    return warm_loaded_.load(std::memory_order_relaxed) == 0U ? "observing"
                                                              : "ready";
  }
  bool prefetch_enabled() const noexcept {
    return warm_loaded_.load(std::memory_order_relaxed) != 0U;
  }
  bool gpu_phase_timing() const noexcept { return gpu_phase_timing_; }
  bool mtp_available() const noexcept { return bundle_.mtp_available; }
  bool mtp_runtime_ready() const noexcept {
    return bundle_.mtp_runtime_ready && mtp_model_ && mtp_cache_ &&
           mtp_directory_ && mtp_routed_ &&
           mtp_shared_.size() == mtp_routed_->component().layer_count;
  }
  bool mtp_enabled() const noexcept { return mtp_enabled_; }
  bool route_tracing_enabled() const noexcept {
    return route_trace_.is_open();
  }
  std::size_t route_trace_max_steps() const noexcept {
    return route_trace_max_steps_;
  }
  void emit_route_trace(std::uint64_t request_id, Request& request) {
    if (!route_trace_.is_open()) return;
    route_trace_ << "{\"schema_version\":1,\"request_id\":"
                 << request_id << ",\"record_bytes\":"
                 << route_trace_record_bytes_ << ",\"device_bytes\":"
                 << route_trace_device_bytes_
                 << ",\"ram_capacity_records\":"
                 << route_trace_ram_capacity_records_
                 << ",\"vram_capacity_records\":"
                 << route_trace_vram_capacity_records_
                 << ",\"initial_ram_records\":[";
    for (std::size_t index = 0U; index < warm_candidates_.size(); ++index) {
      if (index != 0U) route_trace_ << ',';
      route_trace_ << warm_candidates_[index].key.layer *
                              routed_->component().experts_per_layer +
                          warm_candidates_[index].key.expert;
    }
    route_trace_ << "],\"initial_vram_records\":[";
    for (std::size_t index = 0U; index < warm_vram_candidate_limit_; ++index) {
      if (index != 0U) route_trace_ << ',';
      const auto& item =
          warm_candidates_[warm_candidates_.size() - 1U - index];
      route_trace_ << item.key.layer *
                              routed_->component().experts_per_layer +
                          item.key.expert;
    }
    route_trace_ << ']'
                 << ",\"captured_steps\":"
                 << request.route_trace_steps.size()
                 << ",\"dropped_steps\":"
                 << request.route_trace_dropped_steps << ",\"steps\":[";
    for (std::size_t step_index = 0U;
         step_index < request.route_trace_steps.size(); ++step_index) {
      if (step_index != 0U) route_trace_ << ',';
      const auto& step = request.route_trace_steps[step_index];
      route_trace_ << "{\"position\":" << step.position
                   << ",\"rows\":" << step.route_rows
                   << ",\"batches\":[";
      for (std::uint32_t layer = 0U;
           layer < routed_->component().layer_count;
           ++layer) {
        if (layer != 0U) route_trace_ << ',';
        route_trace_ << '[';
        for (std::uint32_t row = 0U; row < step.route_rows; ++row) {
          const auto& trace =
              step.routes[layer * step.route_rows + row];
          for (std::size_t expert_index = 0U;
               expert_index < trace.routed_experts.size(); ++expert_index) {
            if (row != 0U || expert_index != 0U) route_trace_ << ',';
            route_trace_ << trace.layer *
                                    routed_->component().experts_per_layer +
                                trace.routed_experts[expert_index];
          }
        }
        route_trace_ << ']';
      }
      route_trace_ << "]}";
    }
    route_trace_ << "]}\n" << std::flush;
    require(route_trace_.good(), "cannot write DeepSeek route trace");
    request.route_trace_steps.clear();
    request.route_trace_dropped_steps = 0U;
  }
  void record_useful_tokens(std::size_t count) noexcept {
    telemetry_.useful_tokens += count;
  }
  bool speculation_active(const Request& request) const noexcept {
    return mtp_enabled_ && request.draft_ready &&
           !request.speculation_suppressed;
  }

 private:
  static constexpr std::uint32_t kCallableEmbedding = 0U;
  static constexpr std::uint32_t kCallableAttention = 1U;
  static constexpr std::uint32_t kCallableRouter = 2U;
  static constexpr std::uint32_t kCallableRoutedMoe = 3U;
  static constexpr std::uint32_t kCallableHead = 4U;
  // The provider advertises a 512-row kernel ceiling. Construction chooses
  // the largest multiple of 32 that fits the authenticated target/MTP state,
  // the configured expert cache and the mandatory 1 GiB device reserve.
  static constexpr std::uint32_t kMaximumSequenceTileRows =
      er::cuda::kDeepSeekMaximumSequenceRows;
  static constexpr std::uint32_t kSequenceBlockRows = 16384U;
  static constexpr std::uint32_t kSequenceStreams = 4U;
  static constexpr std::string_view token_abi =
      "batch.token-id.u32.host.v1";
  static constexpr std::string_view position_abi =
      "batch.position.u32.host.v1";
  static constexpr std::string_view hca_abi =
      "batch.hca4.hidden.f32.cuda.v1";
  static constexpr std::string_view hidden_abi =
      "batch.hidden.f32.cuda.v1";
  static constexpr std::string_view route_index_abi =
      "batch.route-index.u32.cuda.v1";
  static constexpr std::string_view route_weight_abi =
      "batch.route-weight.f32.cuda.v1";
  static constexpr std::string_view remote_expert_input_abi =
      "expert.swiglu.input.f32.host.v1";
  static constexpr std::string_view remote_expert_output_abi =
      "expert.swiglu.output.f32.host.v1";

  void observe_storage_prediction(bool selected) noexcept {
    if (selected) {
      callable_prefetch_storage_selected_.fetch_add(
          1U, std::memory_order_relaxed);
    } else {
      callable_prefetch_storage_incorrect_.fetch_add(
          1U, std::memory_order_relaxed);
    }
  }

  void reset_callable_prefetch(CallableRequestState& request,
                               bool count_cancellations) noexcept {
    for (auto& item : request.prefetch_items) {
      if (!item.terminal && item.handle.valid()) {
        item.handle.cancel();
        if (count_cancellations)
          callable_prefetch_cancelled_.fetch_add(
              1U, std::memory_order_relaxed);
      }
    }
    request.prefetch_items.clear();
    request.prediction_copy_pending = false;
    request.prediction_target_layer =
        std::numeric_limits<std::uint32_t>::max();
  }

  void poll_callable_prefetch(CallableRequestState& request) noexcept {
    try {
      if (request.prediction_copy_pending) {
        const auto query = cudaEventQuery(request.route_prediction_ready_event);
        if (query == cudaErrorNotReady) return;
        if (query != cudaSuccess) {
          callable_prefetch_errors_.fetch_add(1U,
                                              std::memory_order_relaxed);
          reset_callable_prefetch(request, true);
          return;
        }
        request.prediction_copy_pending = false;
        const auto width = routed_->component().route_width;
        std::set<std::uint32_t> unique;
        for (std::uint32_t slot = 0U; slot < width; ++slot) {
          const auto expert = request.route_prediction_host[slot];
          if (expert >= routed_->component().experts_per_layer ||
              !unique.insert(expert).second) {
            callable_prefetch_errors_.fetch_add(1U,
                                                std::memory_order_relaxed);
            reset_callable_prefetch(request, false);
            return;
          }
        }
        callable_prefetch_routes_completed_.fetch_add(
            1U, std::memory_order_relaxed);
        callable_prefetch_candidates_.fetch_add(width,
                                                std::memory_order_relaxed);
        request.prefetch_items.reserve(width);
        for (std::uint32_t slot = 0U; slot < width; ++slot) {
          const auto expert = request.route_prediction_host[slot];
          const auto key = routed_->key(request.prediction_target_layer,
                                        expert);
          const auto* record = routed_->record(request.prediction_target_layer,
                                               expert);
          if (!record) {
            callable_prefetch_errors_.fetch_add(1U,
                                                std::memory_order_relaxed);
            reset_callable_prefetch(request, false);
            return;
          }
          const auto snapshot = cache_->inspect(key);
          CallableRequestState::PrefetchItem item;
          item.expert = expert;
          bool resolve = false;
          auto target = er::ExpertResolveTarget::automatic;
          if (snapshot && snapshot->has_device_copy) {
            item.source_tier = CallablePrefetchTier::device;
            callable_prefetch_vram_candidates_.fetch_add(
                1U, std::memory_order_relaxed);
            resolve = true;
          } else if (snapshot && snapshot->has_host_copy) {
            item.source_tier = CallablePrefetchTier::host;
            callable_prefetch_ram_candidates_.fetch_add(
                1U, std::memory_order_relaxed);
            resolve = cache_->vram_admission_would_improve(key, *record);
          } else {
            item.source_tier = CallablePrefetchTier::storage;
            callable_prefetch_storage_candidates_.fetch_add(
                1U, std::memory_order_relaxed);
            // Cross-layer transition prediction remains shadow telemetry for
            // absent experts. Its measured hit rate did not reduce bytes, so
            // it must never create speculative storage traffic.
          }
          if (!resolve) {
            // RAM-to-VRAM prediction must prove that it improves admission;
            // absent storage predictions are deliberately observation-only.
            item.terminal = true;
            item.failed = true;
          } else {
            const std::array<std::uint32_t, 1U> one{expert};
            auto handle = routed_->resolve(
                request.prediction_target_layer, one, target,
                er::ExpertAcquireOptions{er::ExpertRequestPriority::prefetch,
                                         false, true, false, true});
            if (!handle.valid()) {
              item.terminal = true;
              item.failed = true;
              callable_prefetch_errors_.fetch_add(
                  1U, std::memory_order_relaxed);
            } else {
              item.handle = std::move(handle);
              callable_prefetch_resolves_launched_.fetch_add(
                  1U, std::memory_order_relaxed);
            }
          }
          request.prefetch_items.push_back(std::move(item));
        }
      }
      for (auto& item : request.prefetch_items) {
        if (item.terminal || !item.handle.valid()) continue;
        auto completed = item.handle.poll();
        if (!completed) continue;
        item.terminal = true;
        if (!completed->status.ok() || completed->experts.size() != 1U ||
            completed->experts.front().key !=
                routed_->key(request.prediction_target_layer, item.expert) ||
            !completed->experts.front()) {
          item.failed = true;
          callable_prefetch_errors_.fetch_add(1U,
                                              std::memory_order_relaxed);
          continue;
        }
        item.resolved = std::move(completed->experts.front());
        callable_prefetch_resolves_completed_.fetch_add(
            1U, std::memory_order_relaxed);
      }
    } catch (...) {
      callable_prefetch_errors_.fetch_add(1U, std::memory_order_relaxed);
      reset_callable_prefetch(request, true);
    }
  }

  er::Status launch_callable_prefetch(CallableRequestState& request,
                                      std::uint32_t source_layer,
                                      std::uint32_t target_layer) noexcept {
    if (request.prediction_copy_pending ||
        !request.prefetch_items.empty())
      return {er::ErrorCode::internal,
              "callable DeepSeek prefetch generation overlaps"};
    const auto source = request.state->layer(source_layer);
    const auto target = request.state->layer(target_layer);
    if (!source.ffn_state || !target.ffn_weights ||
        !request.route_prediction_state ||
        !request.route_prediction_host ||
        !request.route_prediction_ready_event)
      return {er::ErrorCode::invalid_argument,
              "callable DeepSeek prefetch state is incomplete"};
    auto status = er::cuda::deepseek_predict_route(
        {target.ffn_weights, request.route_prediction_state.get(),
         source.ffn_state->normalized_input(), request.current_token,
         request.stream});
    if (!status.ok()) return {status.code(), std::string(status.message())};
    auto error = cudaMemcpyAsync(
        request.route_prediction_host,
        request.route_prediction_state->expert_indices(),
        static_cast<std::size_t>(routed_->component().route_width) *
            sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost, request.stream);
    if (error == cudaSuccess)
      error = cudaEventRecord(request.route_prediction_ready_event,
                              request.stream);
    if (error != cudaSuccess)
      return {er::ErrorCode::internal,
              std::string("queue callable route prediction: ") +
                  cudaGetErrorString(error)};
    request.prediction_target_layer = target_layer;
    request.prediction_copy_pending = true;
    callable_prefetch_routes_launched_.fetch_add(
        1U, std::memory_order_relaxed);
    const auto wait_started = std::chrono::steady_clock::now();
    error = cudaEventSynchronize(request.route_prediction_ready_event);
    callable_prefetch_prediction_wait_ns_.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_started)
                .count()),
        std::memory_order_relaxed);
    if (error != cudaSuccess)
      return {er::ErrorCode::internal,
              std::string("complete callable route prediction: ") +
                  cudaGetErrorString(error)};
    poll_callable_prefetch(request);
    return er::Status::success();
  }

  struct CallableRoutedExecution final {
    enum class Stage : std::uint8_t { planning, resolving, terminal };

    struct PendingResolve final {
      std::uint32_t expert{};
      er::ExpertResolveTarget target{er::ExpertResolveTarget::automatic};
      er::ExpertResolveHandle handle;
    };

    struct HeldLease final {
      std::uint32_t expert{};
      er::ExpertLease lease;
    };

    struct HeldHostLease final {
      std::uint32_t expert{};
      std::uint32_t selection{};
      er::HostExpertLease lease;
    };

    struct RemoteLease final {
      std::uint32_t expert{};
      std::shared_ptr<er::IRemoteExpertLease> lease;
    };

    struct PendingRemoteExecution final {
      std::uint32_t expert{};
      std::uint32_t selection{};
      std::uint64_t invocation_id{};
      er::ActiveExpertExecutionHandle handle;
    };

    Model* model{};
    PreparedOperation prepared;
    std::shared_ptr<CallableRequestState> request;
    er::ProgramRequestContext context;
    std::vector<PendingResolve> pending;
    std::vector<HeldLease> leases;
    std::vector<HeldHostLease> host_leases;
    std::vector<RemoteLease> remote_leases;
    std::vector<PendingRemoteExecution> remote_pending;
    std::vector<std::uint32_t> remote_experts;
    std::vector<std::uint32_t> selected_experts;
    std::uint64_t completed_selection_mask{};
    Stage stage{Stage::planning};
    std::uint64_t pin_id{};
    bool demand_active{};
    bool remote_input_enqueued{};
    bool remote_input_ready{};
    bool stale_plan_replan_pending{};
    bool prefetch_transition_done{};
    std::chrono::steady_clock::time_point next_stall_diagnostic{
        std::chrono::steady_clock::now() + std::chrono::seconds(5)};

    void diagnose_stall() noexcept {
      const auto now = std::chrono::steady_clock::now();
      if (now < next_stall_diagnostic) return;
      next_stall_diagnostic = now + std::chrono::seconds(5);
      try {
        const auto cache = model->cache_->telemetry();
        std::cerr << "[callable-stall] request=" << context.request_id
                  << " layer=" << prepared.component_layer
                  << " stage="
                  << (stage == Stage::planning
                          ? "planning"
                          : stage == Stage::resolving ? "resolving"
                                                     : "terminal")
                  << " completed_mask=" << completed_selection_mask
                  << " pin=" << pin_id << " pending=" << pending.size()
                  << " device_leases=" << leases.size()
                  << " host_leases=" << host_leases.size()
                  << " remote_leases=" << remote_leases.size()
                  << " remote_pending=" << remote_pending.size()
                  << " ram_bytes=" << cache.ram_bytes
                  << " vram_bytes=" << cache.vram_bytes
                  << " staging_stalls="
                  << std::accumulate(cache.staging_stalls_by_priority.begin(),
                                     cache.staging_stalls_by_priority.end(),
                                     std::uint64_t{0U})
                  << " items=";
        for (const auto& item : pending) {
          const auto key = model->routed_->key(prepared.component_layer,
                                               item.expert);
          const auto snapshot = model->cache_->inspect(key);
          std::cerr << item.expert << ':'
                    << (item.target == er::ExpertResolveTarget::host
                            ? 'h'
                            : item.target == er::ExpertResolveTarget::device
                                  ? 'd'
                                  : 'a');
          if (snapshot) {
            std::cerr << ':' << static_cast<unsigned>(snapshot->state) << ':'
                      << snapshot->reference_count << ':'
                      << snapshot->waiter_count << ':'
                      << static_cast<unsigned>(snapshot->has_host_copy) << ':'
                      << static_cast<unsigned>(snapshot->has_device_copy);
          } else {
            std::cerr << ":missing";
          }
          std::cerr << ',';
        }
        std::cerr << '\n';
      } catch (...) {
      }
    }

    [[nodiscard]] bool selection_completed(
        std::uint32_t expert) const noexcept {
      const auto width = model->routed_->component().route_width;
      if (selected_experts.size() < width) return false;
      const auto found = std::find(selected_experts.begin(),
                                   selected_experts.begin() + width, expert);
      return found != selected_experts.begin() + width &&
             (completed_selection_mask &
              (std::uint64_t{1U} <<
               static_cast<std::uint32_t>(found - selected_experts.begin()))) !=
                 0U;
    }

    [[nodiscard]] er::Status prepare_cpu_placements(
        const er::cuda::DirectoryPlanResult& plan) {
      if (!model->proactive_cpu_hybrid_enabled_ || !model->cpu_ ||
          !model->planner_ ||
          plan.missing_experts.empty())
        return er::Status::success();
      const auto& component = model->routed_->component();
      model->refresh_hybrid_transfer_observation();
      model->poll_callable_gpu_observation(*request);
      std::vector<er::HybridDispatchCandidate> candidates;
      candidates.reserve(component.route_width);
      for (std::uint32_t slot = 0U; slot < component.route_width; ++slot) {
        const auto bit = std::uint64_t{1U} << slot;
        if ((completed_selection_mask & bit) != 0U) continue;
        const auto expert = plan.selected_experts.at(slot);
        if (std::any_of(host_leases.begin(), host_leases.end(),
                        [expert](const auto& held) {
                          return held.expert == expert;
                        }) ||
            std::any_of(pending.begin(), pending.end(),
                        [expert](const auto& item) {
                          return item.expert == expert &&
                                 item.target == er::ExpertResolveTarget::host;
                        }))
          continue;
        const auto* record = model->routed_->record(
            prepared.component_layer, expert);
        if (!record)
          return {er::ErrorCode::invalid_argument,
                  "callable CPU candidate is absent from catalog"};
        const auto key = model->routed_->key(prepared.component_layer, expert);
        if (model->remote_owners_ && model->remote_owners_->find(key)) continue;
        const bool ready =
            std::find(plan.ready_experts.begin(), plan.ready_experts.end(),
                      expert) != plan.ready_experts.end();
        const auto snapshot = model->cache_->inspect(key);
        const bool host_ready = snapshot && snapshot->has_host_copy &&
            (snapshot->state == er::CacheState::ram_ready ||
             snapshot->state == er::CacheState::vram_ready);
        candidates.push_back(
            {expert, 1U, record->stored_bytes, ready, host_ready, true,
             snapshot ? snapshot->placement_temperature : 0U,
             snapshot ? snapshot->last_access : 0U});
      }
      if (candidates.empty()) return er::Status::success();
      const auto placement = model->planner_->plan(candidates);
      if (!placement.status.ok())
        return {placement.status.code(),
                std::string(placement.status.message())};
      for (const auto& decision : placement.decisions) {
        if (decision.executor != er::HybridExecutor::cpu_local) continue;
        const auto selected = std::find(
            plan.selected_experts.begin(),
            plan.selected_experts.begin() + component.route_width,
            decision.expert);
        if (selected ==
            plan.selected_experts.begin() + component.route_width)
          return {er::ErrorCode::internal,
                  "callable CPU placement is outside exact route"};
        const auto* record = model->routed_->record(
            prepared.component_layer, decision.expert);
        auto lease = model->cache_->try_acquire_host(
            model->routed_->key(prepared.component_layer, decision.expert),
            *record, false, er::ExpertRequestPriority::demand);
        const auto selection = static_cast<std::uint32_t>(
            selected - plan.selected_experts.begin());
        // The planner is advisory, while this lease is the atomic tier gate.
        // If RAM was evicted after inspect(), preserve progress by leaving the
        // expert on the ordinary demand-to-GPU path. Never turn an NVMe read
        // into a proactive CPU placement.
        if (!lease) continue;
        for (std::size_t index = 0U; index < pending.size();) {
          if (pending[index].expert != decision.expert) {
            ++index;
            continue;
          }
          pending[index].handle.cancel();
          pending.erase(pending.begin() +
                        static_cast<std::ptrdiff_t>(index));
        }
        host_leases.push_back(
            {decision.expert, selection, std::move(*lease)});
      }
      return er::Status::success();
    }

    [[nodiscard]] er::Status enqueue_cpu_input() {
      if (host_leases.empty() || remote_input_enqueued)
        return er::Status::success();
      const auto view = request->state->layer(prepared.component_layer);
      const auto bytes = static_cast<std::size_t>(
                             model->routed_->component().hidden_size) *
                         sizeof(float);
      auto error = cudaMemcpyAsync(
          request->remote_input_host, view.ffn_state->normalized_input(), bytes,
          cudaMemcpyDeviceToHost, request->stream);
      if (error == cudaSuccess)
        error = cudaEventRecord(request->remote_input_ready_event,
                                request->stream);
      if (error != cudaSuccess)
        return {er::ErrorCode::internal,
                std::string("stage callable CPU activation: ") +
                    cudaGetErrorString(error)};
      remote_input_enqueued = true;
      return er::Status::success();
    }

    [[nodiscard]] er::Status execute_cpu_placements() {
      if (host_leases.empty()) return er::Status::success();
      auto error = cudaEventSynchronize(request->remote_input_ready_event);
      if (error != cudaSuccess)
        return {er::ErrorCode::internal,
                std::string("wait for callable CPU activation: ") +
                    cudaGetErrorString(error)};
      remote_input_ready = true;
      const auto& component = model->routed_->component();
      std::vector<er::cpu::DeepSeekPackedWorkGroup> groups;
      groups.reserve(host_leases.size());
      for (const auto& held : host_leases) {
        groups.push_back(
            {held.lease.bytes(), held.lease.compact_sections(),
             component.hidden_size, component.intermediate_size,
             {held.selection}, {held.selection}});
      }
      const auto started = std::chrono::steady_clock::now();
      auto status = model->cpu_->execute(
          groups,
          std::span<const float>(request->remote_input_host,
                                 component.hidden_size),
          1U, component.route_width,
          std::span<float>(
              request->remote_outputs_host,
              static_cast<std::size_t>(component.route_width) *
                  component.hidden_size));
      const auto elapsed = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count());
      model->planner_->observe_cpu(elapsed, host_leases.size());
      if (!status.ok()) return status;
      for (const auto& held : host_leases) {
        const auto view = request->state->layer(prepared.component_layer);
        status = er::cuda::deepseek_ffn_import_selection_output(
            {view.ffn_state, held.selection,
             request->remote_outputs_host +
                 static_cast<std::size_t>(held.selection) *
                     component.hidden_size,
             static_cast<std::uint64_t>(component.hidden_size) *
                 sizeof(float),
             request->stream});
        if (!status.ok()) return status;
        completed_selection_mask |= std::uint64_t{1U} << held.selection;
      }
      const auto selections = host_leases.size();
      host_leases.clear();
      model->callable_cpu_launches_.fetch_add(1U,
                                               std::memory_order_relaxed);
      model->callable_cpu_selections_.fetch_add(selections,
                                                 std::memory_order_relaxed);
      model->callable_cpu_activation_bytes_.fetch_add(
          static_cast<std::uint64_t>(component.hidden_size) * sizeof(float),
          std::memory_order_relaxed);
      model->callable_cpu_output_bytes_.fetch_add(
          static_cast<std::uint64_t>(selections) * component.hidden_size *
              sizeof(float),
          std::memory_order_relaxed);
      return er::Status::success();
    }

    [[nodiscard]] er::Status begin_plan() noexcept {
      const auto view = request->state->layer(prepared.component_layer);
      const auto status = model->directory_->begin_plan_async(
          *request->directory_workspace, prepared.component_layer,
          view.ffn_state->expert_indices(), view.ffn_state->selection_count(),
          request->stream, true);
      if (status.ok()) stage = Stage::planning;
      return {status.code(), std::string(status.message())};
    }

    [[nodiscard]] std::optional<er::OperationExecutionResult> fail(
        er::Status status) noexcept {
      cancel();
      return er::OperationExecutionResult{
          {status.code(), std::string(status.message())}, {}};
    }

    [[nodiscard]] er::Status retain_ready(
        std::span<const std::uint32_t> experts) {
      using namespace std::chrono_literals;
      for (const auto expert : experts) {
        if (expert == model->routed_->component().experts_per_layer) continue;
        if (std::any_of(leases.begin(), leases.end(), [expert](const auto& held) {
              return held.expert == expert;
            }))
          continue;
        const auto* record =
            model->routed_->record(prepared.component_layer, expert);
        if (!record)
          return {er::ErrorCode::invalid_argument,
                  "callable DeepSeek ready expert is absent from catalog"};
        auto acquired = model->cache_->acquire(
            model->routed_->key(prepared.component_layer, expert), *record,
            er::ExpertAcquireOptions{er::ExpertRequestPriority::demand, false,
                                     true, false});
        if (acquired.wait_for(0ms) != std::future_status::ready) {
          acquired.cancel();
          return {er::ErrorCode::internal,
                  "callable DeepSeek ready expert lost cache residency"};
        }
        auto result = acquired.get();
        if (!result.status.ok() || !result.lease)
          return result.status.ok()
                     ? er::Status(er::ErrorCode::internal,
                                  "callable DeepSeek ready lease is absent")
                     : er::Status(result.status.code(),
                                  std::string(result.status.message()));
        leases.push_back({expert, std::move(result.lease)});
      }
      return er::Status::success();
    }

    [[nodiscard]] er::Status validate_route(
        const er::cuda::DirectoryPlanResult& plan) {
      const auto& component = model->routed_->component();
      if (plan.selected_experts.size() != component.route_width + 1U ||
          plan.selected_experts.back() != component.experts_per_layer)
        return {er::ErrorCode::internal,
                "callable DeepSeek directory returned an invalid route"};
      if (selected_experts.empty()) {
        selected_experts = plan.selected_experts;
      } else if (selected_experts != plan.selected_experts) {
        return {er::ErrorCode::internal,
                "callable DeepSeek route changed while pages were supplied"};
      }
      if (!prefetch_transition_done) {
        auto status = consume_prefetch(plan);
        if (!status.ok()) return status;
        if (request->current_router_component_layer !=
            prepared.component_layer)
          return {er::ErrorCode::internal,
                  "callable DeepSeek router/FFN layer mismatch"};
        const auto target = request->current_prefetch_target_layer;
        request->current_prefetch_target_layer.reset();
        request->current_router_component_layer =
            std::numeric_limits<std::uint32_t>::max();
        if (target) {
          status = model->launch_callable_prefetch(
              *request, prepared.component_layer, *target);
          if (!status.ok()) return status;
        }
        prefetch_transition_done = true;
      }
      return er::Status::success();
    }

    [[nodiscard]] er::Status consume_prefetch(
        const er::cuda::DirectoryPlanResult& plan) {
      model->poll_callable_prefetch(*request);
      const auto no_layer = std::numeric_limits<std::uint32_t>::max();
      if (request->prediction_target_layer == no_layer)
        return er::Status::success();
      if (request->prediction_target_layer != prepared.component_layer ||
          request->prediction_copy_pending) {
        model->callable_prefetch_errors_.fetch_add(
            1U, std::memory_order_relaxed);
        model->reset_callable_prefetch(*request, true);
        return {er::ErrorCode::internal,
                "callable DeepSeek prefetch generation is out of order"};
      }
      const auto width = model->routed_->component().route_width;
      const auto selected_end = selected_experts.begin() + width;
      for (auto& item : request->prefetch_items) {
        const bool selected =
            std::find(selected_experts.begin(), selected_end, item.expert) !=
            selected_end;
        if (item.source_tier == CallablePrefetchTier::storage)
          model->observe_storage_prediction(selected);
        if (!selected) {
          model->callable_prefetch_incorrect_.fetch_add(
              1U, std::memory_order_relaxed);
          if (!item.terminal && item.handle.valid()) {
            item.handle.cancel();
            model->callable_prefetch_cancelled_.fetch_add(
                1U, std::memory_order_relaxed);
          }
          continue;
        }
        model->callable_prefetch_selected_.fetch_add(
            1U, std::memory_order_relaxed);
        bool host_page_ready = false;
        if (item.terminal && !item.failed && item.resolved) {
          if (item.resolved.placement == er::ExpertPlacementKind::device &&
              item.resolved.device_lease) {
            leases.push_back(
                {item.expert, std::move(item.resolved.device_lease)});
          } else if (item.resolved.placement ==
                         er::ExpertPlacementKind::remote &&
                     item.resolved.remote_lease) {
            remote_experts.push_back(item.expert);
            remote_leases.push_back(
                {item.expert, std::move(item.resolved.remote_lease)});
          } else if (item.resolved.placement ==
                         er::ExpertPlacementKind::host &&
                     item.resolved.host_lease) {
            host_page_ready = true;
          } else {
            item.failed = true;
          }
          if (!item.failed) {
            model->callable_prefetch_useful_.fetch_add(
                1U, std::memory_order_relaxed);
            if (!host_page_ready) {
              continue;
            }
          }
        }
        if (!host_page_ready) {
          model->callable_prefetch_late_.fetch_add(
              1U, std::memory_order_relaxed);
        }
        const bool missing =
            std::find(plan.missing_experts.begin(),
                      plan.missing_experts.end(), item.expert) !=
            plan.missing_experts.end();
        if (missing && !item.failed) {
          const std::array<std::uint32_t, 1U> one{item.expert};
          auto demand = model->routed_->resolve(
              prepared.component_layer, one,
              er::ExpertResolveTarget::automatic,
              er::ExpertAcquireOptions{er::ExpertRequestPriority::demand,
                                       false, true, false,
                                       model->cpu_hybrid_enabled_});
          if (!demand.valid())
            return {er::ErrorCode::backpressure,
                    "callable DeepSeek prefetch promotion was rejected"};
          pending.push_back({item.expert, er::ExpertResolveTarget::automatic,
                             std::move(demand)});
          model->callable_resolves_launched_.fetch_add(
              1U, std::memory_order_relaxed);
        }
        if (!item.terminal && item.handle.valid()) {
          item.handle.cancel();
          model->callable_prefetch_cancelled_.fetch_add(
              1U, std::memory_order_relaxed);
        }
      }
      request->prefetch_items.clear();
      request->prediction_target_layer = no_layer;
      return er::Status::success();
    }

    [[nodiscard]] std::uint64_t ready_selection_mask(
        const er::cuda::DirectoryPlanResult& plan) const noexcept {
      std::uint64_t mask = 0U;
      const auto width = model->routed_->component().route_width;
      for (std::size_t slot = 0U; slot < width; ++slot) {
        const auto bit = std::uint64_t{1U} << slot;
        if ((completed_selection_mask & bit) == 0U &&
            std::find(plan.ready_experts.begin(), plan.ready_experts.end(),
                      plan.selected_experts[slot]) != plan.ready_experts.end())
          mask |= bit;
      }
      return mask;
    }

    [[nodiscard]] er::Status execute_ready(
        const er::cuda::DirectoryPlanResult& plan) {
      const auto mask = ready_selection_mask(plan);
      if (mask == 0U) return er::Status::success();
      const auto view = request->state->layer(prepared.component_layer);
      model->poll_callable_gpu_observation(*request);
      const bool measure = !request->gpu_selection_observation_pending &&
          cudaEventRecord(request->gpu_selection_started_event,
                          request->stream) == cudaSuccess;
      auto status = er::cuda::deepseek_ffn_execute_selections(
          {view.ffn_weights, view.ffn_state,
           model->directory_->device_entries(), mask,
           model->directory_->experts_per_layer(), request->stream});
      if (!status.ok()) return {status.code(), std::string(status.message())};
      if (measure &&
          cudaEventRecord(request->gpu_selection_finished_event,
                          request->stream) == cudaSuccess) {
        request->gpu_selection_observation_pending = true;
        request->gpu_selection_observation_selections =
            static_cast<std::uint32_t>(std::popcount(mask));
      }
      completed_selection_mask |= mask;
      model->record_callable_selection_launch(
          static_cast<std::uint32_t>(std::popcount(mask)),
          !plan.missing_experts.empty());
      return er::Status::success();
    }

    [[nodiscard]] er::Status start_missing(
        std::span<const std::uint32_t> missing) {
      bool stale_local_page = false;
      for (const auto expert : missing) {
        if (expert >= model->routed_->component().experts_per_layer)
          return {er::ErrorCode::internal,
                  "callable DeepSeek shared expert is not resident"};
        if (selection_completed(expert)) continue;
        const bool held = std::any_of(
            leases.begin(), leases.end(), [expert](const auto& item) {
              return item.expert == expert;
            });
        const bool inflight = std::any_of(
            pending.begin(), pending.end(), [expert](const auto& item) {
              return item.expert == expert;
            });
        const bool remotely_owned =
            std::find(remote_experts.begin(), remote_experts.end(), expert) !=
            remote_experts.end();
        if (held) {
          // A local resolve can complete after an asynchronous directory plan
          // was enqueued but before that plan is consumed. The returned plan
          // is then an accurate snapshot of its enqueue point, but stale with
          // respect to the lease we now hold. Replan instead of treating the
          // already-resolved exact page as an impossible miss.
          stale_local_page = true;
          continue;
        }
        if (inflight || remotely_owned) continue;
        const std::array<std::uint32_t, 1U> one{expert};
        auto handle = model->routed_->resolve(
            prepared.component_layer, one, er::ExpertResolveTarget::automatic,
            er::ExpertAcquireOptions{er::ExpertRequestPriority::demand, false,
                                     true, false,
                                     model->cpu_hybrid_enabled_});
        if (!handle.valid())
          return {er::ErrorCode::backpressure,
                  "callable DeepSeek expert-page resolve was rejected"};
        pending.push_back({expert, er::ExpertResolveTarget::automatic,
                           std::move(handle)});
        model->callable_resolves_launched_.fetch_add(
            1U, std::memory_order_relaxed);
      }
      if (stale_local_page) {
        stale_plan_replan_pending = true;
        model->callable_stale_plan_replans_.fetch_add(
            1U, std::memory_order_relaxed);
      }
      if (pending.empty() && remote_leases.empty() &&
          remote_pending.empty() && !stale_plan_replan_pending)
        return {er::ErrorCode::internal,
                "callable DeepSeek missing route has no page resolve"};
      stage = Stage::resolving;
      return er::Status::success();
    }

    [[nodiscard]] er::Status poll_resolves(bool& replan) {
      for (std::size_t index = 0U; index < pending.size();) {
        auto completed = pending[index].handle.poll();
        if (!completed) {
          ++index;
          continue;
        }
        if (!completed->status.ok() &&
            completed->status.code() == er::ErrorCode::backpressure &&
            pending[index].target != er::ExpertResolveTarget::host &&
            model->cpu_hybrid_enabled_ && model->cpu_) {
          const auto expert = pending[index].expert;
          const std::array<std::uint32_t, 1U> one{expert};
          auto host = model->routed_->resolve(
              prepared.component_layer, one, er::ExpertResolveTarget::host,
              er::ExpertAcquireOptions{er::ExpertRequestPriority::demand,
                                       false, true, false});
          if (!host.valid())
            return {er::ErrorCode::backpressure,
                    "exact host fallback was rejected after device admission"};
          pending[index].target = er::ExpertResolveTarget::host;
          pending[index].handle = std::move(host);
          model->callable_resolves_launched_.fetch_add(
              1U, std::memory_order_relaxed);
          model->callable_cpu_host_resolves_launched_.fetch_add(
              1U, std::memory_order_relaxed);
          model->callable_cpu_device_admission_fallbacks_.fetch_add(
              1U, std::memory_order_relaxed);
          ++index;
          continue;
        }
        if (!completed->status.ok())
          return {completed->status.code(),
                  std::string(completed->status.message())};
        if (completed->experts.size() != 1U ||
            completed->experts.front().key.expert != pending[index].expert)
          return {er::ErrorCode::internal,
                  "callable DeepSeek page resolve identity mismatch"};
        auto& resolved = completed->experts.front();
        if (resolved.placement == er::ExpertPlacementKind::device &&
            resolved.device_lease) {
          leases.push_back(
              {pending[index].expert, std::move(resolved.device_lease)});
          replan = true;
        } else if (resolved.placement == er::ExpertPlacementKind::host &&
                   resolved.host_lease) {
          const auto width = model->routed_->component().route_width;
          const auto selected = std::find(
              selected_experts.begin(), selected_experts.begin() + width,
              pending[index].expert);
          if (selected == selected_experts.begin() + width)
            return {er::ErrorCode::internal,
                    "callable CPU host resolve is outside exact route"};
          host_leases.push_back(
              {pending[index].expert,
               static_cast<std::uint32_t>(selected -
                                          selected_experts.begin()),
               std::move(resolved.host_lease)});
          model->callable_cpu_host_resolves_completed_.fetch_add(
              1U, std::memory_order_relaxed);
          replan = true;
        } else if (resolved.placement == er::ExpertPlacementKind::remote &&
                   resolved.remote_lease &&
                   resolved.remote_lease->identity().key == resolved.key &&
                   resolved.remote_lease->identity().capability ==
                       model->routed_->component().execution_capability &&
                   resolved.remote_lease->identity().execution_abi ==
                       model->routed_->component().execution_abi) {
          remote_experts.push_back(pending[index].expert);
          remote_leases.push_back(
              {pending[index].expert, std::move(resolved.remote_lease)});
          model->callable_remote_resolves_.fetch_add(
              1U, std::memory_order_relaxed);
        } else {
          return {er::ErrorCode::internal,
                  "callable DeepSeek page resolve returned invalid ownership"};
        }
        pending.erase(pending.begin() + index);
        model->callable_resolves_completed_.fetch_add(
            1U, std::memory_order_relaxed);
      }
      return er::Status::success();
    }

    [[nodiscard]] er::Status enqueue_remote_input() {
      if (remote_input_enqueued || remote_leases.empty())
        return er::Status::success();
      const auto view = request->state->layer(prepared.component_layer);
      const auto bytes = static_cast<std::size_t>(
                             model->routed_->component().hidden_size) *
                         sizeof(float);
      auto error = cudaMemcpyAsync(
          request->remote_input_host, view.ffn_state->normalized_input(), bytes,
          cudaMemcpyDeviceToHost, request->stream);
      if (error == cudaSuccess)
        error = cudaEventRecord(request->remote_input_ready_event,
                                request->stream);
      if (error != cudaSuccess)
        return {er::ErrorCode::internal,
                std::string("stage remote expert activation: ") +
                    cudaGetErrorString(error)};
      remote_input_enqueued = true;
      return er::Status::success();
    }

    [[nodiscard]] er::Status dispatch_remote_leases() {
      if (!remote_input_ready || remote_leases.empty())
        return er::Status::success();
      const auto& component = model->routed_->component();
      const auto bytes = static_cast<std::uint64_t>(component.hidden_size) *
                         sizeof(float);
      for (auto& remote : remote_leases) {
        for (std::uint32_t slot = 0U; slot < component.route_width; ++slot) {
          const auto bit = std::uint64_t{1U} << slot;
          if ((completed_selection_mask & bit) != 0U ||
              selected_experts.at(slot) != remote.expert)
            continue;
          er::ActiveExpertInvocation invocation;
          invocation.request_id = context.request_id;
          invocation.invocation_id = request->next_remote_invocation++;
          invocation.selection_index = slot;
          invocation.route_width = component.route_width;
          invocation.deadline = context.deadline;
          invocation.input = {
              std::string(remote_expert_input_abi), "host.pinned", request,
              reinterpret_cast<const std::byte*>(request->remote_input_host),
              bytes};
          invocation.output_abi = remote_expert_output_abi;
          invocation.output_bytes = bytes;
          auto handle = remote.lease->execute(std::move(invocation));
          if (!handle.valid())
            return {er::ErrorCode::backpressure,
                    "remote expert owner rejected an exact selection"};
          remote_pending.push_back(
              {remote.expert, slot, request->next_remote_invocation - 1U,
               std::move(handle)});
          model->callable_remote_selections_launched_.fetch_add(
              1U, std::memory_order_relaxed);
        }
      }
      remote_leases.clear();
      return er::Status::success();
    }

    [[nodiscard]] er::Status poll_remote(bool& replan) {
      auto status = enqueue_remote_input();
      if (!status.ok()) return status;
      if (remote_input_enqueued && !remote_input_ready) {
        const auto query = cudaEventQuery(request->remote_input_ready_event);
        if (query == cudaSuccess) {
          remote_input_ready = true;
        } else if (query != cudaErrorNotReady) {
          return {er::ErrorCode::internal,
                  std::string("query remote expert activation: ") +
                      cudaGetErrorString(query)};
        }
      }
      status = dispatch_remote_leases();
      if (!status.ok()) return status;
      const auto hidden = model->routed_->component().hidden_size;
      const auto bytes = static_cast<std::uint64_t>(hidden) * sizeof(float);
      for (std::size_t index = 0U; index < remote_pending.size();) {
        auto completed = remote_pending[index].handle.poll();
        if (!completed) {
          ++index;
          continue;
        }
        if (!completed->status.ok()) {
          model->callable_remote_errors_.fetch_add(
              1U, std::memory_order_relaxed);
          return {completed->status.code(),
                  std::string(completed->status.message())};
        }
        const auto& item = remote_pending[index];
        if (completed->identity.key !=
                model->routed_->key(prepared.component_layer, item.expert) ||
            completed->request_id != context.request_id ||
            completed->invocation_id != item.invocation_id ||
            completed->selection_index != item.selection ||
            !completed->output.valid() ||
            completed->output.abi != remote_expert_output_abi ||
            completed->output.bytes != bytes ||
            completed->evidence.weight_transport_bytes != 0U)
          return {er::ErrorCode::checksum_mismatch,
                  "remote expert selection correlation mismatch"};
        auto* output = request->remote_outputs_host +
                       static_cast<std::size_t>(item.selection) * hidden;
        std::memcpy(output, completed->output.data,
                    static_cast<std::size_t>(bytes));
        const auto view = request->state->layer(prepared.component_layer);
        status = er::cuda::deepseek_ffn_import_selection_output(
            {view.ffn_state, item.selection, output, bytes, request->stream});
        if (!status.ok()) return status;
        completed_selection_mask |= std::uint64_t{1U} << item.selection;
        model->record_remote_active_expert(completed->evidence);
        model->callable_remote_selections_completed_.fetch_add(
            1U, std::memory_order_relaxed);
        remote_pending.erase(remote_pending.begin() + index);
        replan = true;
      }
      return er::Status::success();
    }

    [[nodiscard]] std::optional<er::OperationExecutionResult> finish_ffn(
        const er::cuda::DirectoryPlanResult& plan) {
      const auto width = model->routed_->component().route_width;
      const auto expected = (std::uint64_t{1U} << width) - 1U;
      if (completed_selection_mask != expected || pin_id == 0U)
        return fail({er::ErrorCode::internal,
                     "callable DeepSeek did not execute the exact route"});
      const auto view = request->state->layer(prepared.component_layer);
      auto status = er::cuda::deepseek_ffn_finalize(
          {view.ffn_weights, view.ffn_state,
           model->directory_->device_entries(),
           request->state->attention_streams(),
           request->state->primary_streams(),
           model->directory_->experts_per_layer(), request->stream});
      if (!status.ok()) return fail(std::move(status));
      status = model->directory_->release_pins_async(pin_id, request->stream);
      pin_id = 0U;
      if (!status.ok()) return fail(std::move(status));

      std::vector<er::ExpertAccess> accesses;
      accesses.reserve(width);
      for (std::size_t index = 0U; index < width; ++index) {
        if (std::find(remote_experts.begin(), remote_experts.end(),
                      selected_experts[index]) != remote_experts.end())
          continue;
        accesses.push_back(
            {model->routed_->key(prepared.component_layer,
                                 selected_experts[index]),
             1U});
      }
      static_cast<void>(model->cache_->record_accesses(accesses));
      if (model->census_) {
        const auto observed = model->census_->observe(
            prepared.component_layer,
            std::span<const std::uint32_t>(selected_experts.data(), width));
        if (!observed.ok()) return fail(std::move(observed));
      }
      leases.clear();
      host_leases.clear();
      remote_leases.clear();
      remote_pending.clear();
      request->route_ready = true;
      finish_demand();
      stage = Stage::terminal;
      return model->operation_result(
          prepared,
          {{"hidden", model->hca_value(request, true)}});
    }

    void finish_demand() noexcept {
      if (!demand_active) return;
      model->demand_depth_.fetch_sub(1U, std::memory_order_acq_rel);
      demand_active = false;
    }

    [[nodiscard]] std::optional<er::OperationExecutionResult> poll() {
      if (stage == Stage::terminal) return std::nullopt;
      diagnose_stall();
      model->poll_callable_prefetch(*request);
      bool replan = false;
      auto status = poll_resolves(replan);
      if (!status.ok()) return fail(std::move(status));
      status = poll_remote(replan);
      if (!status.ok()) return fail(std::move(status));
      if (stage == Stage::resolving) {
        const bool host_resolve_incomplete = std::any_of(
            pending.begin(), pending.end(), [](const auto& item) {
              return item.target == er::ExpertResolveTarget::host;
            });
        // A proactive CPU plan may resolve several immutable expert pages in
        // parallel. Replanning after the first completion fragments one MoE
        // route into many one-page executor calls and repeatedly stages the
        // same activation. Once any host page has completed, also wait for the
        // remaining exact-route resolves: a later device-admission fallback
        // must join this layer's CPU batch instead of creating another
        // activation transfer and executor call. Before a host page completes,
        // device-only progress remains asynchronous.
        if (host_resolve_incomplete ||
            (!host_leases.empty() && !pending.empty()))
          return std::nullopt;
        if (!replan && !stale_plan_replan_pending) return std::nullopt;
        status = begin_plan();
        if (!status.ok()) return fail(std::move(status));
        stale_plan_replan_pending = false;
        return std::nullopt;
      }

      auto polled = model->directory_->poll_plan_async(
          *request->directory_workspace);
      if (!polled.status.ok()) return fail(std::move(polled.status));
      if (!polled.complete) return std::nullopt;
      auto plan = std::move(polled.plan);
      if (!plan.status.ok()) return fail(std::move(plan.status));
      pin_id = plan.pin_id;
      status = validate_route(plan);
      if (!status.ok()) return fail(std::move(status));
      status = retain_ready(plan.ready_experts);
      if (!status.ok()) return fail(std::move(status));
      status = prepare_cpu_placements(plan);
      if (!status.ok()) return fail(std::move(status));
      status = enqueue_cpu_input();
      if (!status.ok()) return fail(std::move(status));
      status = execute_ready(plan);
      if (!status.ok()) return fail(std::move(status));
      status = execute_cpu_placements();
      if (!status.ok()) return fail(std::move(status));
      const auto expected =
          (std::uint64_t{1U} << model->routed_->component().route_width) - 1U;
      if (completed_selection_mask == expected) return finish_ffn(plan);
      if (plan.missing_experts.empty())
        return fail({er::ErrorCode::internal,
                     "callable DeepSeek route completed without every selection"});
      if (pin_id != 0U) {
        status = model->directory_->release_pins_async(pin_id, request->stream);
        pin_id = 0U;
        if (!status.ok()) return fail(std::move(status));
      }
      status = start_missing(plan.missing_experts);
      if (!status.ok()) return fail(std::move(status));
      return std::nullopt;
    }

    void cancel() noexcept {
      if (stage == Stage::terminal) return;
      if (stage == Stage::planning)
        static_cast<void>(model->directory_->cancel_plan_async(
            *request->directory_workspace));
      for (auto& item : pending) item.handle.cancel();
      pending.clear();
      for (auto& item : remote_pending) item.handle.cancel();
      if (!remote_pending.empty())
        model->callable_remote_cancellations_.fetch_add(
            remote_pending.size(), std::memory_order_relaxed);
      remote_pending.clear();
      remote_leases.clear();
      model->reset_callable_prefetch(*request, true);
      if (pin_id != 0U)
        static_cast<void>(
            model->directory_->release_pins(pin_id, request->stream));
      pin_id = 0U;
      leases.clear();
      host_leases.clear();
      request->route_ready = false;
      finish_demand();
      stage = Stage::terminal;
    }
  };

  [[nodiscard]] std::optional<std::uint32_t>
  acquire_callable_provider_slot() {
    std::lock_guard lock(callable_provider_mutex_);
    const auto found = std::find(callable_provider_slots_.begin(),
                                 callable_provider_slots_.end(), false);
    if (found == callable_provider_slots_.end()) return std::nullopt;
    *found = true;
    return static_cast<std::uint32_t>(found - callable_provider_slots_.begin());
  }

  void release_callable_provider_slot(std::uint32_t slot) noexcept {
    std::lock_guard lock(callable_provider_mutex_);
    if (slot < callable_provider_slots_.size())
      callable_provider_slots_[slot] = false;
  }

  static std::uint32_t host_u32(const er::ExecutionValue& value,
                                std::string_view abi,
                                std::string_view label) {
    if (!value.valid() || value.abi != abi || value.memory_domain != "host" ||
        value.bytes != sizeof(std::uint32_t))
      throw std::runtime_error(std::string(label) + " ABI mismatch");
    std::uint32_t result{};
    std::memcpy(&result, value.data, sizeof(result));
    return result;
  }

  static std::vector<std::uint32_t> host_u32_batch(
      const er::ExecutionValue* value, std::string_view abi,
      std::string_view label, std::uint32_t maximum_rows) {
    if (!value || !value->valid() || value->abi != abi ||
        value->memory_domain != "host" || value->bytes == 0U ||
        value->bytes % sizeof(std::uint32_t) != 0U ||
        value->bytes / sizeof(std::uint32_t) > maximum_rows ||
        reinterpret_cast<std::uintptr_t>(value->data) %
                alignof(std::uint32_t) !=
            0U)
      throw std::runtime_error(std::string(label) + " ABI mismatch");
    std::vector<std::uint32_t> result(
        static_cast<std::size_t>(value->bytes / sizeof(std::uint32_t)));
    std::memcpy(result.data(), value->data,
                result.size() * sizeof(result.front()));
    return result;
  }

  [[nodiscard]] er::OperationExecutionResult run_callable_program_sequence(
      CallableProgramSequence& sequence);

  static void require_device_value(const er::ExecutionValue& value,
                                   const void* expected,
                                   std::uint64_t bytes,
                                   std::string_view abi) {
    if (!value.valid() || value.abi != abi ||
        value.memory_domain != "cuda.device" || value.bytes != bytes ||
        value.data != reinterpret_cast<const std::byte*>(expected))
      throw std::runtime_error("callable DeepSeek device value ABI mismatch");
  }

  static void require_hca(
      const er::ExecutionValue& value,
      const std::shared_ptr<CallableRequestState>& state, bool primary) {
    const auto* expected = primary ? state->state->primary_streams()
                                   : state->state->attention_streams();
    require_device_value(value, expected,
                         4ULL * state->owner->bundle_.descriptor.hidden_size *
                             sizeof(float),
                         hca_abi);
  }

  template <typename T>
  er::ExecutionValue device_value(
      const std::shared_ptr<CallableRequestState>& state, const T* pointer,
      std::uint64_t elements, std::string_view abi) const {
    return {std::string(abi), "cuda.device", state,
            reinterpret_cast<const std::byte*>(pointer),
            elements * sizeof(T)};
  }

  er::ExecutionValue hca_value(
      const std::shared_ptr<CallableRequestState>& state,
      bool primary) const {
    const auto* pointer = primary ? state->state->primary_streams()
                                  : state->state->attention_streams();
    return device_value(state, pointer,
                        4ULL * bundle_.descriptor.hidden_size, hca_abi);
  }

  static er::OperationExecutionHandle completed_operation(
      er::OperationExecutionResult result) {
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

  static er::ExactDecodeExecutionHandle completed_exact_decode(
      er::ExactDecodeExecutionResult result) {
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

  er::OperationExecutionResult operation_result(
      const PreparedOperation& prepared,
      std::map<std::string, er::ExecutionValue, std::less<>> outputs) const {
    er::OperationExecutionResult result;
    result.status = er::Status::success();
    result.outputs.reserve(prepared.outputs.size());
    for (const auto& [port, abi] : prepared.outputs) {
      auto found = outputs.find(port);
      if (found == outputs.end() || found->second.abi != abi)
        return {{er::ErrorCode::internal,
                 "callable DeepSeek output ABI mismatch"}, {}};
      result.outputs.push_back(std::move(found->second));
    }
    return result;
  }

  er::OperationExecutionHandle completed_outputs(
      const PreparedOperation& prepared,
      std::map<std::string, er::ExecutionValue, std::less<>> outputs) const {
    return completed_operation(operation_result(prepared, std::move(outputs)));
  }

  er::OperationExecutionHandle start_callable_routed(
      const PreparedOperation& prepared,
      std::shared_ptr<CallableRequestState> request,
      er::ProgramRequestContext context) {
    auto state = std::make_shared<CallableRoutedExecution>();
    state->model = this;
    state->prepared = prepared;
    state->request = std::move(request);
    state->context = std::move(context);
    demand_depth_.fetch_add(1U, std::memory_order_acq_rel);
    state->demand_active = true;
    auto status = state->begin_plan();
    if (!status.ok()) {
      state->cancel();
      return completed_operation(
          {{status.code(), std::string(status.message())}, {}});
    }
    return er::OperationExecutionHandle::from_callbacks(
        [state] { return state->poll(); }, [state] { state->cancel(); });
  }

  er::Status execute_callable_attention(CallableRequestState& request,
                                        std::uint32_t layer,
                                        std::uint32_t position) noexcept {
    const auto view = request.state->layer(layer);
    const bool compressed = view.compress_ratio != 0U;
    const bool emits = compressed &&
        (position + 1U) % view.compress_ratio == 0U;
    const auto rope = rope_at(position);
    const float* group_cosine = nullptr;
    const float* group_sine = nullptr;
    if (emits && view.compress_ratio == 4U) {
      group_cosine = rope.ratio_four_group_cosine;
      group_sine = rope.ratio_four_group_sine;
    } else if (emits) {
      group_cosine = rope.ratio_128_group_cosine;
      group_sine = rope.ratio_128_group_sine;
    }
    return er::cuda::deepseek_attention_decode(
        {view.attention_weights, view.attention_state,
         request.state->primary_streams(), request.state->attention_streams(),
         compressed ? rope.compressed_cosine : rope.base_cosine,
         compressed ? rope.compressed_sine : rope.base_sine,
         group_cosine, group_sine, position, 1e-6F, 20U, request.stream});
  }

  struct DemandActivity final {
    explicit DemandActivity(std::atomic<std::uint32_t>& depth) noexcept
        : depth_(depth) {
      depth_.fetch_add(1U, std::memory_order_acq_rel);
    }
    ~DemandActivity() { depth_.fetch_sub(1U, std::memory_order_acq_rel); }
    std::atomic<std::uint32_t>& depth_;
  };

  static void update_moving_average(double& value, std::uint64_t& samples,
                                    double observation) noexcept {
    constexpr double alpha = 0.25;
    value = samples == 0U ? observation
                          : alpha * observation + (1.0 - alpha) * value;
    ++samples;
  }

  void capture_route_step(Request& request, std::uint32_t position,
                          std::uint32_t route_rows) {
    if (!route_trace_.is_open()) return;
    const auto trace = request.controller->route_trace();
    require((route_rows == 1U || route_rows == 2U) &&
                trace.size() == routed_->component().layer_count * route_rows,
            "DeepSeek route trace has invalid step geometry");
    for (std::uint32_t layer = 0U;
         layer < routed_->component().layer_count;
         ++layer) {
      for (std::uint32_t row = 0U; row < route_rows; ++row) {
        const auto& item = trace[layer * route_rows + row];
        require(item.layer == layer &&
                    std::all_of(
                        item.routed_experts.begin(), item.routed_experts.end(),
                        [this](std::uint32_t expert) {
                          return expert <
                                 routed_->component().experts_per_layer;
                        }),
                "DeepSeek route trace contains an invalid route");
      }
    }
    if (request.route_trace_steps.size() == route_trace_max_steps_) {
      ++request.route_trace_dropped_steps;
      return;
    }
    request.route_trace_steps.push_back(
        {position, route_rows,
         std::vector<er::cuda::DeepSeekRouteTraceEntry>(trace.begin(),
                                                        trace.end())});
  }
  static constexpr std::uint64_t rope_row_values = 8ULL * 32U;
  std::uint64_t rope_table_bytes() const noexcept {
    return static_cast<std::uint64_t>(max_context_) * rope_row_values *
           sizeof(float);
  }

  void initialize_rope_table() {
    std::vector<std::array<std::array<float, 32U>, 8U>> host(max_context_);
    for (std::uint32_t position = 0U; position < max_context_; ++position)
      host[position] = rope_row(position);
    float* candidate = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&candidate),
                          rope_table_bytes()),
               "allocate DeepSeek RoPE table");
    const auto copied = cudaMemcpy(candidate, host.data(), rope_table_bytes(),
                                   cudaMemcpyHostToDevice);
    if (copied != cudaSuccess) {
      static_cast<void>(cudaFree(candidate));
      cuda_check(copied, "upload DeepSeek RoPE table");
    }
    rope_table_ = candidate;
  }

  er::cuda::DeepSeekDecodeRope rope_at(std::uint32_t position) const {
    require(position < max_context_, "DeepSeek RoPE position is out of range");
    auto* row = rope_table_ + static_cast<std::size_t>(position) *
                                 rope_row_values;
    return {row, row + 32U, row + 64U, row + 96U,
            row + 128U, row + 160U, row + 192U, row + 224U};
  }

  void prepare_warm_from_census() {
    if (placement_ == "capacity") return;
    const auto* representative = catalog_.find(0U, 0U);
    require(representative != nullptr,
            "DeepSeek warm start has no representative routed record");
    require(representative->stored_bytes != 0U,
            "DeepSeek warm start record geometry is empty");
    const auto maximum_entries = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cache_ram_protected_bytes_ / representative->stored_bytes,
            static_cast<std::uint64_t>(routed_->component().layer_count) *
                std::min<std::uint32_t>(
                    32U, routed_->component().experts_per_layer)));
    if (maximum_entries == 0U) return;
    warm_candidates_ = census_->stable_warm_set(
        maximum_entries,
        std::min<std::uint32_t>(32U,
                                routed_->component().experts_per_layer));
    // The census is hottest-first. Background admission runs coldest-first so
    // the final bounded recency order leaves the hottest evidence newest.
    std::reverse(warm_candidates_.begin(), warm_candidates_.end());
    warm_candidates_count_.store(warm_candidates_.size(),
                                 std::memory_order_relaxed);
    const auto resident_used = cache_->telemetry().vram_resident_bytes;
    const auto resident_available =
        cache_vram_resident_limit_bytes_ > resident_used
            ? cache_vram_resident_limit_bytes_ - resident_used
            : 0U;
    const auto device_bytes = representative->device_bytes == 0U
                                  ? representative->stored_bytes
                                  : representative->device_bytes;
    warm_vram_candidate_limit_ = std::min<std::size_t>(
        warm_candidates_.size(), resident_available / device_bytes);
    warm_vram_candidates_count_.store(warm_vram_candidate_limit_,
                                      std::memory_order_relaxed);
  }

  void background_warm_loop() noexcept {
    try {
      struct Pending final {
        er::RouteCensusWarmEntry item;
        er::HostPreloadHandle handle;
        std::chrono::steady_clock::time_point started;
      };
      std::vector<Pending> pending;
      pending.reserve(2U);
      std::size_t next = 0U;
      bool demand_was_active = false;
      while (next < warm_candidates_.size() || !pending.empty()) {
        if (warm_stop_.load(std::memory_order_acquire)) {
          warm_cancelled_.fetch_add(pending.size(),
                                    std::memory_order_relaxed);
          for (auto& item : pending) item.handle.cancel();
          pending.clear();
          return;
        }
        bool progressed = false;
        for (std::size_t index = 0U; index < pending.size();) {
          if (pending[index].handle.wait_for(std::chrono::milliseconds(0)) !=
              std::future_status::ready) {
            ++index;
            continue;
          }
          auto result = pending[index].handle.get();
          if (result.status.ok() && result.retained) {
            const auto* record = catalog_.find(pending[index].item.key.layer,
                                               pending[index].item.key.expert);
            if (record) {
              static_cast<void>(cache_->record_access(
                  pending[index].item.key, 1U));
              warm_loaded_.fetch_add(1U, std::memory_order_relaxed);
              warm_bytes_.fetch_add(record->stored_bytes,
                                    std::memory_order_relaxed);
            }
          } else {
            warm_failed_.fetch_add(1U, std::memory_order_relaxed);
          }
          warm_ns_.fetch_add(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - pending[index].started)
                      .count()),
              std::memory_order_relaxed);
          pending.erase(pending.begin() +
                        static_cast<std::ptrdiff_t>(index));
          progressed = true;
        }
        const bool demand_active =
            demand_depth_.load(std::memory_order_acquire) != 0U;
        if (demand_active && !demand_was_active &&
            next < warm_candidates_.size()) {
          warm_demand_pauses_.fetch_add(1U, std::memory_order_relaxed);
        }
        demand_was_active = demand_active;
        while (demand_depth_.load(std::memory_order_acquire) == 0U &&
               pending.size() < 2U &&
               next < warm_candidates_.size()) {
          const auto item = warm_candidates_[next++];
          const auto* record = catalog_.find(item.key.layer, item.key.expert);
          if (!record) {
            warm_failed_.fetch_add(1U, std::memory_order_relaxed);
            continue;
          }
          pending.push_back(
              {item,
               cache_->preload_host(
                   item.key, *record,
                   er::HostPreloadOptions{er::ExpertRequestPriority::warm,
                                          true}),
               std::chrono::steady_clock::now()});
          auto maximum = warm_inflight_max_.load(std::memory_order_relaxed);
          while (maximum < pending.size() &&
                 !warm_inflight_max_.compare_exchange_weak(
                     maximum, pending.size(), std::memory_order_relaxed)) {
          }
          progressed = true;
        }
        if (!progressed)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      // The protected census records are now validated in RAM. Fill only the
      // resident VRAM class, hottest-first, leaving the independently bounded
      // transient class available for the scheduler's retained routes. Warm
      // acquisitions are lower priority than demand and therefore cannot
      // consume demand staging or jump an active request in the cache queue.
      struct PendingVram final {
        er::RouteCensusWarmEntry item;
        const er::PayloadRecord* record{};
        er::AcquireHandle handle;
        std::chrono::steady_clock::time_point started;
      };
      std::vector<PendingVram> pending_vram;
      pending_vram.reserve(2U);
      std::size_t next_vram = 0U;
      demand_was_active = false;
      while (next_vram < warm_vram_candidate_limit_ ||
             !pending_vram.empty()) {
        if (warm_stop_.load(std::memory_order_acquire)) {
          warm_vram_cancelled_.fetch_add(pending_vram.size(),
                                         std::memory_order_relaxed);
          for (auto& item : pending_vram) item.handle.cancel();
          pending_vram.clear();
          return;
        }
        bool progressed = false;
        for (std::size_t index = 0U; index < pending_vram.size();) {
          if (pending_vram[index].handle.wait_for(
                  std::chrono::milliseconds(0)) !=
              std::future_status::ready) {
            ++index;
            continue;
          }
          auto result = pending_vram[index].handle.get();
          if (result.status.ok() && result.lease) {
            warm_vram_loaded_.fetch_add(1U, std::memory_order_relaxed);
            warm_vram_bytes_.fetch_add(
                pending_vram[index].record->device_bytes == 0U
                    ? pending_vram[index].record->stored_bytes
                    : pending_vram[index].record->device_bytes,
                std::memory_order_relaxed);
          } else {
            warm_vram_failed_.fetch_add(1U, std::memory_order_relaxed);
          }
          warm_vram_ns_.fetch_add(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() -
                      pending_vram[index].started)
                      .count()),
              std::memory_order_relaxed);
          pending_vram.erase(pending_vram.begin() +
                             static_cast<std::ptrdiff_t>(index));
          progressed = true;
        }
        const bool demand_active =
            demand_depth_.load(std::memory_order_acquire) != 0U;
        if (demand_active && !demand_was_active &&
            next_vram < warm_vram_candidate_limit_) {
          warm_vram_demand_pauses_.fetch_add(1U,
                                             std::memory_order_relaxed);
        }
        demand_was_active = demand_active;
        while (demand_depth_.load(std::memory_order_acquire) == 0U &&
               pending_vram.size() < 2U &&
               next_vram < warm_vram_candidate_limit_) {
          const auto item =
              warm_candidates_[warm_candidates_.size() - 1U - next_vram++];
          const auto* record = catalog_.find(item.key.layer, item.key.expert);
          const auto cached = cache_->inspect(item.key);
          if (!record || !cached || !cached->has_host_copy) {
            warm_vram_failed_.fetch_add(1U, std::memory_order_relaxed);
            continue;
          }
          pending_vram.push_back(
              {item, record,
               cache_->acquire(
                   item.key, *record,
                   er::ExpertAcquireOptions{
                       er::ExpertRequestPriority::warm, false, true, true}),
               std::chrono::steady_clock::now()});
          auto maximum =
              warm_vram_inflight_max_.load(std::memory_order_relaxed);
          while (maximum < pending_vram.size() &&
                 !warm_vram_inflight_max_.compare_exchange_weak(
                     maximum, pending_vram.size(),
                     std::memory_order_relaxed)) {
          }
          progressed = true;
        }
        if (!progressed)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    } catch (...) {
      warm_loop_errors_.fetch_add(1U, std::memory_order_relaxed);
    }
  }

  void stop_background_warm() noexcept {
    warm_stop_.store(true, std::memory_order_release);
    if (warm_thread_.joinable()) warm_thread_.join();
  }

  Bundle bundle_;
  std::vector<std::uint32_t> compression_ratios_;
  std::uint32_t hash_router_layers_{};
  std::uint32_t max_context_{}, capacity_{}, sequence_tile_rows_{};
  std::uint64_t ram_bytes_{}, vram_bytes_{}, request_bytes_{},
      sequence_workspace_bytes_{}, next_operation_{1U};
  std::uint64_t mtp_request_bytes_{}, mtp_cache_bytes_{},
      verify_request_bytes_{};
  std::uint64_t kv_cache_bytes_{}, kv_page_bytes_{}, kv_page_capacity_{};
  std::uint64_t cache_ram_protected_bytes_{};
  std::uint64_t cache_vram_resident_limit_bytes_{};
  std::uint32_t kv_page_tokens_{};
  std::string placement_;
  bool gpu_phase_timing_{};
  bool mtp_enabled_{};
  bool census_namespace_rebound_{};
  bool retained_route_enabled_{true};
  bool proactive_cpu_hybrid_enabled_{};
  bool cpu_hybrid_enabled_{true};
  bool retain_previous_route_{};
  std::filesystem::path route_trace_path_;
  std::size_t route_trace_max_steps_{};
  std::shared_ptr<const er::ActiveExpertOwnerDirectory> remote_owners_;
  std::ofstream route_trace_;
  std::uint64_t route_trace_record_bytes_{};
  std::uint64_t route_trace_device_bytes_{};
  std::uint64_t route_trace_ram_capacity_records_{};
  std::uint64_t route_trace_vram_capacity_records_{};
  double ordinary_ns_per_token_{};
  double speculative_ns_per_useful_{};
  std::uint64_t ordinary_samples_{}, speculative_samples_{};
  er::DeepSeekModelArtifacts artifacts_;
  er::DeepSeekTensorArtifacts mtp_artifacts_;
  std::vector<er::ResidentExpertSpec> mtp_shared_specs_;
  er::ExpertCatalog catalog_;
  er::ExpertCatalog mtp_catalog_;
  std::shared_ptr<er::WindowsIocpStorage> iocp_;
  std::shared_ptr<er::ExtentGatherStorage> storage_;
  std::shared_ptr<er::FixedBufferPool> buffers_;
  std::shared_ptr<er::cuda::DeepSeekResidentModelState> model_;
  std::shared_ptr<er::cuda::DeepSeekResidentTensorState> mtp_model_;
  float* rope_table_{};
  std::shared_ptr<er::cuda::CudaExpertDirectory> directory_;
  std::shared_ptr<er::cuda::CudaExpertUploader> uploader_;
  std::unique_ptr<er::ExpertCache> cache_;
  std::unique_ptr<er::LocalExpertStore> local_store_;
  std::unique_ptr<er::RemoteExpertStore> remote_store_;
  std::unique_ptr<er::PlacementExpertStore> placement_store_;
  std::unique_ptr<er::RoutedExpertRuntime> routed_;
  er::ResidentExpertSet shared_;
  std::shared_ptr<er::cuda::CudaExpertDirectory> mtp_directory_;
  std::shared_ptr<er::cuda::CudaExpertUploader> mtp_uploader_;
  std::unique_ptr<er::ExpertCache> mtp_cache_;
  std::unique_ptr<er::LocalExpertStore> mtp_local_store_;
  std::unique_ptr<er::RemoteExpertStore> mtp_remote_store_;
  std::unique_ptr<er::PlacementExpertStore> mtp_placement_store_;
  std::unique_ptr<er::RoutedExpertRuntime> mtp_routed_;
  er::ResidentExpertSet mtp_shared_;
  std::shared_ptr<er::cpu::DeepSeekPackedExecutor> cpu_;
  std::shared_ptr<er::HybridDispatchPlanner> planner_;
  std::mutex hybrid_observation_mutex_;
  std::uint64_t observed_upload_bytes_{};
  std::uint64_t observed_upload_wait_ns_{};
  std::shared_ptr<er::RouteCensus> census_;
  std::unique_ptr<er::cuda::DeepSeekDecodeScheduler> scheduler_;
  std::vector<er::RouteCensusWarmEntry> warm_candidates_;
  std::size_t warm_vram_candidate_limit_{};
  std::thread warm_thread_;
  std::atomic<bool> warm_stop_{false};
  std::atomic<std::uint32_t> demand_depth_{0U};
  std::atomic<std::uint64_t> warm_candidates_count_{0U};
  std::atomic<std::uint64_t> warm_loaded_{0U};
  std::atomic<std::uint64_t> warm_failed_{0U};
  std::atomic<std::uint64_t> warm_cancelled_{0U};
  std::atomic<std::uint64_t> warm_demand_pauses_{0U};
  std::atomic<std::uint64_t> warm_inflight_max_{0U};
  std::atomic<std::uint64_t> warm_loop_errors_{0U};
  std::atomic<std::uint64_t> warm_bytes_{0U};
  std::atomic<std::uint64_t> warm_ns_{0U};
  std::atomic<std::uint64_t> warm_vram_candidates_count_{0U};
  std::atomic<std::uint64_t> warm_vram_loaded_{0U};
  std::atomic<std::uint64_t> warm_vram_failed_{0U};
  std::atomic<std::uint64_t> warm_vram_cancelled_{0U};
  std::atomic<std::uint64_t> warm_vram_demand_pauses_{0U};
  std::atomic<std::uint64_t> warm_vram_inflight_max_{0U};
  std::atomic<std::uint64_t> warm_vram_bytes_{0U};
  std::atomic<std::uint64_t> warm_vram_ns_{0U};
  std::atomic<std::uint64_t> callable_selection_launches_{0U};
  std::atomic<std::uint64_t> callable_selections_{0U};
  std::atomic<std::uint64_t> callable_overlap_launches_{0U};
  std::atomic<std::uint64_t> callable_cpu_launches_{0U};
  std::atomic<std::uint64_t> callable_cpu_selections_{0U};
  std::atomic<std::uint64_t> callable_cpu_activation_bytes_{0U};
  std::atomic<std::uint64_t> callable_cpu_output_bytes_{0U};
  std::atomic<std::uint64_t> callable_cpu_host_resolves_launched_{0U};
  std::atomic<std::uint64_t> callable_cpu_host_resolves_completed_{0U};
  std::atomic<std::uint64_t> callable_cpu_device_admission_fallbacks_{0U};
  std::atomic<std::uint64_t> callable_resolves_launched_{0U};
  std::atomic<std::uint64_t> callable_resolves_completed_{0U};
  std::atomic<std::uint64_t> callable_stale_plan_replans_{0U};
  std::atomic<std::uint64_t> callable_prefetch_routes_launched_{0U};
  std::atomic<std::uint64_t> callable_prefetch_routes_completed_{0U};
  std::atomic<std::uint64_t> callable_prefetch_prediction_wait_ns_{0U};
  std::atomic<std::uint64_t> callable_prefetch_candidates_{0U};
  std::atomic<std::uint64_t> callable_prefetch_vram_candidates_{0U};
  std::atomic<std::uint64_t> callable_prefetch_ram_candidates_{0U};
  std::atomic<std::uint64_t> callable_prefetch_storage_candidates_{0U};
  std::atomic<std::uint64_t> callable_prefetch_storage_selected_{0U};
  std::atomic<std::uint64_t> callable_prefetch_storage_incorrect_{0U};
  std::atomic<std::uint64_t> callable_prefetch_resolves_launched_{0U};
  std::atomic<std::uint64_t> callable_prefetch_resolves_completed_{0U};
  std::atomic<std::uint64_t> callable_prefetch_selected_{0U};
  std::atomic<std::uint64_t> callable_prefetch_useful_{0U};
  std::atomic<std::uint64_t> callable_prefetch_late_{0U};
  std::atomic<std::uint64_t> callable_prefetch_incorrect_{0U};
  std::atomic<std::uint64_t> callable_prefetch_cancelled_{0U};
  std::atomic<std::uint64_t> callable_prefetch_errors_{0U};
  std::atomic<std::uint64_t> callable_remote_resolves_{0U};
  std::atomic<std::uint64_t> callable_remote_selections_launched_{0U};
  std::atomic<std::uint64_t> callable_remote_selections_completed_{0U};
  std::atomic<std::uint64_t> callable_remote_errors_{0U};
  std::atomic<std::uint64_t> callable_remote_cancellations_{0U};
  std::atomic<std::uint64_t> callable_remote_activation_tx_bytes_{0U};
  std::atomic<std::uint64_t> callable_remote_activation_rx_bytes_{0U};
  std::atomic<std::uint64_t> callable_remote_weight_tx_bytes_{0U};
  std::atomic<std::uint64_t> callable_remote_owner_weight_read_bytes_{0U};
  std::atomic<std::uint64_t> callable_remote_owner_storage_read_bytes_{0U};
  std::atomic<std::uint64_t> callable_remote_owner_ram_read_bytes_{0U};
  std::atomic<std::uint64_t> callable_remote_owner_vram_read_bytes_{0U};
  std::atomic<std::uint64_t> callable_remote_owner_execution_ns_{0U};
  std::atomic<std::uint64_t> callable_remote_transport_wait_ns_{0U};
  std::atomic<std::uint64_t> prefill_protection_candidates_{0U};
  std::atomic<std::uint64_t> prefill_protection_promoted_{0U};
  std::atomic<std::uint64_t> sequence_blocks_{0U};
  std::atomic<std::uint64_t> sequence_rows_{0U};
  std::atomic<std::uint64_t> sequence_layers_{0U};
  std::atomic<std::uint64_t> sequence_attention_hca_pre_norm_ns_{0U};
  std::atomic<std::uint64_t> sequence_attention_projection_ns_{0U};
  std::atomic<std::uint64_t> sequence_causal_attention_ns_{0U};
  std::atomic<std::uint64_t> sequence_attention_output_projection_ns_{0U};
  std::atomic<std::uint64_t> sequence_route_ns_{0U};
  std::atomic<std::uint64_t> sequence_expert_wait_ns_{0U};
  std::atomic<std::uint64_t> sequence_expert_execute_ns_{0U};
  std::atomic<std::uint64_t> sequence_finalize_ns_{0U};
  std::mutex callable_provider_mutex_;
  std::vector<bool> callable_provider_slots_;
  WorkerTelemetry telemetry_;
};

er::OperationExecutionResult Model::run_callable_program_sequence(
    CallableProgramSequence& sequence) {
  constexpr std::size_t hidden = 4096U;
  constexpr std::size_t streams_per_row = kSequenceStreams * hidden;
  constexpr std::size_t kAcquireWindow = 8U;
  const auto top_k = routed_->component().route_width;
  const auto expert_count = routed_->component().experts_per_layer;
  require(top_k == 6U && expert_count != 0U,
          "DeepSeek sequence route geometry is unsupported");

  struct HostFrontier final {
    HostFrontier(std::size_t rows, std::size_t top_k,
                 std::size_t tile_rows, std::size_t expert_window,
                 std::size_t hidden_size, std::size_t stream_values)
        : primary(rows * stream_values),
          attention(rows * stream_values),
          ffn_inputs(rows * hidden_size),
          routing_weights(rows * top_k),
          expert_indices(rows * top_k),
          selection_outputs(rows * top_k * hidden_size),
          post(rows * kSequenceStreams),
          combination(rows * kSequenceStreams * kSequenceStreams),
          expert_input_window(expert_window * tile_rows * hidden_size),
          expert_output_window(expert_window * tile_rows * hidden_size) {}

    PinnedBuffer<float> primary;
    PinnedBuffer<float> attention;
    PinnedBuffer<float> ffn_inputs;
    PinnedBuffer<float> routing_weights;
    PinnedBuffer<std::uint32_t> expert_indices;
    PinnedBuffer<float> selection_outputs;
    PinnedBuffer<float> post;
    PinnedBuffer<float> combination;
    PinnedBuffer<float> expert_input_window;
    PinnedBuffer<float> expert_output_window;
  } host(std::min<std::size_t>(kSequenceBlockRows, sequence.tokens.size()),
         top_k, sequence_tile_rows_, kAcquireWindow, hidden,
         streams_per_row);

  auto& request = *sequence.request;
  auto& attention_workspace = *request.sequence_attention_workspace;
  auto& ffn_workspace = *request.sequence_ffn_workspace;
  const auto device_entries = directory_->device_entries();
  const auto experts_per_layer = expert_count +
      routed_->component().shared_experts_per_layer;
  const auto stream = request.stream;
  const auto cancelled = [&] {
    return sequence.cancelled.load(std::memory_order_acquire) ||
           std::chrono::steady_clock::now() > sequence.context.deadline;
  };
  const auto ensure_active = [&] {
    if (cancelled())
      throw std::runtime_error(
          "compressed sparse program-sequence was cancelled");
  };
  const auto copy_async = [&](void* destination, const void* source,
                              std::size_t bytes, cudaMemcpyKind kind,
                              const char* operation) {
    cuda_check(cudaMemcpyAsync(destination, source, bytes, kind, stream),
               operation);
  };
  DemandActivity demand(demand_depth_);

  struct SequenceProfileEvents final {
    SequenceProfileEvents() {
      for (auto& event : events)
        cuda_check(cudaEventCreate(&event),
                   "create DeepSeek sequence profile event");
    }
    ~SequenceProfileEvents() {
      for (auto event : events)
        if (event) static_cast<void>(cudaEventDestroy(event));
    }
    SequenceProfileEvents(const SequenceProfileEvents&) = delete;
    SequenceProfileEvents& operator=(const SequenceProfileEvents&) = delete;

    [[nodiscard]] std::uint64_t elapsed_ns(std::size_t first,
                                           std::size_t last) const {
      float milliseconds = 0.0F;
      cuda_check(cudaEventElapsedTime(&milliseconds, events[first],
                                      events[last]),
                 "measure DeepSeek sequence phase");
      return static_cast<std::uint64_t>(
          std::llround(static_cast<double>(milliseconds) * 1'000'000.0));
    }

    std::array<cudaEvent_t, 6U> events{};
  } profile;
  const er::cuda::DeepSeekAttentionBatchLaunch::ProfileEvents
      attention_profile{profile.events[1], profile.events[2],
                        profile.events[3], profile.events[4]};

  if (request.exact_decode_enabled) {
    request.sequence_sync_successors.resize(sequence.tokens.size());
    for (std::size_t row = 0U; row + 1U < sequence.tokens.size(); ++row)
      request.sequence_sync_successors[row] = sequence.tokens[row + 1U];
    request.sequence_sync_first = sequence.positions.front();
    request.sequence_sync_consumed = 0U;
    request.sequence_final_target_streams.resize(streams_per_row);
  }

  for (std::size_t block_first = 0U; block_first < sequence.tokens.size();
       block_first += kSequenceBlockRows) {
    ensure_active();
    const auto block_rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(kSequenceBlockRows,
                             sequence.tokens.size() - block_first));
    sequence_blocks_.fetch_add(1U, std::memory_order_relaxed);
    sequence_rows_.fetch_add(block_rows, std::memory_order_relaxed);

    // Embeddings are sparse row lookups; tile them only to bound device
    // storage. The expensive dense operations begin below.
    for (std::uint32_t tile_first = 0U; tile_first < block_rows;
         tile_first += sequence_tile_rows_) {
      const auto rows =
          std::min(sequence_tile_rows_, block_rows - tile_first);
      for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto global = block_first + tile_first + row;
        auto status = request.state->embed(sequence.tokens[global], stream);
        require(status.ok(), status.message());
        copy_async(request.sequence_primary_streams +
                       static_cast<std::size_t>(row) * streams_per_row,
                   request.state->primary_streams(),
                   streams_per_row * sizeof(float),
                   cudaMemcpyDeviceToDevice,
                   "stage DeepSeek sequence embedding");
      }
      copy_async(host.primary.data() +
                     static_cast<std::size_t>(tile_first) * streams_per_row,
                 request.sequence_primary_streams,
                 static_cast<std::size_t>(rows) * streams_per_row *
                     sizeof(float),
                 cudaMemcpyDeviceToHost,
                 "retain DeepSeek sequence embedding frontier");
      cuda_check(cudaStreamSynchronize(stream),
                 "complete DeepSeek sequence embedding tile");
    }

    for (std::uint32_t program_layer = 0U;
         program_layer < request.state->layer_count(); ++program_layer) {
      ensure_active();
      sequence_layers_.fetch_add(1U, std::memory_order_relaxed);
      const auto operation_offset = 1U + 3U * program_layer;
      const auto* attention_operation =
          sequence.operations[operation_offset];
      const auto* router_operation =
          sequence.operations[operation_offset + 1U];
      const auto component_layer = router_operation->component_layer;
      const auto attention_view =
          request.state->layer(attention_operation->logical_layer);
      const auto ffn_view = request.state->layer(component_layer);
      require(attention_view.attention_weights &&
                  attention_view.attention_state && ffn_view.ffn_weights &&
                  ffn_view.ffn_state,
              "DeepSeek sequence layer state is absent");

      for (std::uint32_t tile_first = 0U; tile_first < block_rows;
           tile_first += sequence_tile_rows_) {
        ensure_active();
        const auto rows =
            std::min(sequence_tile_rows_, block_rows - tile_first);
        copy_async(request.sequence_primary_streams,
                   host.primary.data() +
                       static_cast<std::size_t>(tile_first) * streams_per_row,
                   static_cast<std::size_t>(rows) * streams_per_row *
                       sizeof(float),
                   cudaMemcpyHostToDevice,
                   "upload DeepSeek sequence attention frontier");
        std::array<er::cuda::DeepSeekAttentionBatchRow,
                   kMaximumSequenceTileRows>
            row_state{};
        for (std::uint32_t row = 0U; row < rows; ++row) {
          const auto global = block_first + tile_first + row;
          const auto position = sequence.positions[global];
          const auto rope = rope_at(position);
          const auto compressed = attention_view.compress_ratio != 0U;
          const auto emits = compressed &&
              (position + 1U) % attention_view.compress_ratio == 0U;
          const float* group_cosine = nullptr;
          const float* group_sine = nullptr;
          if (emits && attention_view.compress_ratio == 4U) {
            group_cosine = rope.ratio_four_group_cosine;
            group_sine = rope.ratio_four_group_sine;
          } else if (emits) {
            group_cosine = rope.ratio_128_group_cosine;
            group_sine = rope.ratio_128_group_sine;
          }
          row_state[row] = {
              compressed ? rope.compressed_cosine : rope.base_cosine,
              compressed ? rope.compressed_sine : rope.base_sine,
              group_cosine, group_sine, position};
        }
        cuda_check(cudaEventRecord(profile.events[0], stream),
                   "start DeepSeek sequence attention profile");
        auto status = er::cuda::deepseek_attention_decode_batch(
            {attention_view.attention_weights,
             attention_view.attention_state, &attention_workspace,
             request.sequence_primary_streams,
             request.sequence_attention_streams, row_state.data(), rows,
             1e-6F, 20U, stream, &attention_profile});
        require(status.ok(), status.message());
        status = er::cuda::deepseek_ffn_route_batch(
            {ffn_view.ffn_weights, ffn_view.ffn_state, &ffn_workspace,
             request.sequence_attention_streams,
             sequence.tokens.data() + block_first + tile_first, rows, 1e-6F,
             20U, stream});
        require(status.ok(), status.message());
        cuda_check(cudaEventRecord(profile.events[5], stream),
                   "finish DeepSeek sequence route profile");

        const auto row_offset = static_cast<std::size_t>(tile_first);
        copy_async(host.attention.data() + row_offset * streams_per_row,
                   request.sequence_attention_streams,
                   static_cast<std::size_t>(rows) * streams_per_row *
                       sizeof(float),
                   cudaMemcpyDeviceToHost,
                   "retain DeepSeek sequence attention output");
        copy_async(host.ffn_inputs.data() + row_offset * hidden,
                   ffn_workspace.normalized_input(),
                   static_cast<std::size_t>(rows) * hidden * sizeof(float),
                   cudaMemcpyDeviceToHost,
                   "retain DeepSeek sequence expert inputs");
        copy_async(host.routing_weights.data() + row_offset * top_k,
                   ffn_workspace.routing_weights(),
                   static_cast<std::size_t>(rows) * top_k * sizeof(float),
                   cudaMemcpyDeviceToHost,
                   "retain DeepSeek sequence route weights");
        copy_async(host.expert_indices.data() + row_offset * top_k,
                   ffn_workspace.expert_indices(),
                   static_cast<std::size_t>(rows) * top_k *
                       sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost,
                   "retain DeepSeek sequence route indices");
        copy_async(host.post.data() + row_offset * kSequenceStreams,
                   ffn_workspace.post_control(),
                   static_cast<std::size_t>(rows) * kSequenceStreams *
                       sizeof(float),
                   cudaMemcpyDeviceToHost,
                   "retain DeepSeek sequence HCA post control");
        copy_async(
            host.combination.data() +
                row_offset * kSequenceStreams * kSequenceStreams,
            ffn_workspace.combination_control(),
            static_cast<std::size_t>(rows) * kSequenceStreams *
                kSequenceStreams * sizeof(float),
            cudaMemcpyDeviceToHost,
            "retain DeepSeek sequence HCA combination control");
        cuda_check(cudaStreamSynchronize(stream),
                   "complete DeepSeek sequence attention and route tile");
        sequence_attention_hca_pre_norm_ns_.fetch_add(
            profile.elapsed_ns(0U, 1U), std::memory_order_relaxed);
        sequence_attention_projection_ns_.fetch_add(
            profile.elapsed_ns(1U, 2U), std::memory_order_relaxed);
        sequence_causal_attention_ns_.fetch_add(
            profile.elapsed_ns(2U, 3U), std::memory_order_relaxed);
        sequence_attention_output_projection_ns_.fetch_add(
            profile.elapsed_ns(3U, 4U), std::memory_order_relaxed);
        sequence_route_ns_.fetch_add(
            profile.elapsed_ns(4U, 5U), std::memory_order_relaxed);
      }

      std::fill_n(host.selection_outputs.data(),
                  static_cast<std::size_t>(block_rows) * top_k * hidden,
                  0.0F);
      std::vector<std::vector<std::uint32_t>> selections(expert_count);
      std::vector<er::ExpertAccess> accesses;
      accesses.reserve(expert_count);
      for (std::uint32_t row = 0U; row < block_rows; ++row) {
        std::span<const std::uint32_t> route(
            host.expert_indices.data() +
                static_cast<std::size_t>(row) * top_k,
            top_k);
        for (std::uint32_t slot = 0U; slot < top_k; ++slot) {
          const auto expert = route[slot];
          require(expert < expert_count,
                  "DeepSeek sequence route references an invalid expert");
          selections[expert].push_back(row * top_k + slot);
          auto& evidence = request.prompt_routes[component_layer][expert];
          if (evidence.count != std::numeric_limits<std::uint32_t>::max())
            ++evidence.count;
          evidence.last_seen = ++request.prompt_route_clock;
        }
        const auto observed = census_->observe(component_layer, route);
        require(observed.ok(), observed.message());
      }

      struct PendingExpert final {
        std::uint32_t expert{};
        er::AcquireHandle handle;
      };
      std::vector<std::uint32_t> active_experts;
      active_experts.reserve(expert_count);
      for (std::uint32_t expert = 0U; expert < expert_count; ++expert) {
        if (selections[expert].empty()) continue;
        require(routed_->record(component_layer, expert) != nullptr,
                "DeepSeek sequence expert is absent from the catalog");
        active_experts.push_back(expert);
      }

      // Acquire ahead far enough to keep the eight staging buffers busy, but
      // never pin a whole layer's unique expert set. A completed acquire owns
      // a device lease even before get(); issuing every expert at once can
      // therefore exhaust the transient VRAM class with unevictable pages.
      std::vector<PendingExpert> pending;
      pending.reserve(kAcquireWindow);
      std::size_t next_expert = 0U;
      const auto launch_pending = [&] {
        while (pending.size() < kAcquireWindow &&
               next_expert < active_experts.size()) {
          const auto expert = active_experts[next_expert++];
          const auto* record = routed_->record(component_layer, expert);
          pending.push_back(
              {expert,
               cache_->acquire(
                   routed_->key(component_layer, expert), *record,
                   er::ExpertAcquireOptions{
                       er::ExpertRequestPriority::demand, false, true,
                       false})});
        }
      };
      const auto cancel_pending = [&] {
        for (auto& item : pending)
          if (item.handle.valid()) item.handle.cancel();
      };
      try {
        launch_pending();
        while (!pending.empty()) {
          struct AcquiredExpert final {
            std::uint32_t expert{};
            er::ExpertLease lease;
            std::size_t first{};
          };
          std::vector<AcquiredExpert> wave;
          wave.reserve(kAcquireWindow);
          while (wave.size() < kAcquireWindow && !pending.empty()) {
            auto item = std::move(pending.front());
            pending.erase(pending.begin());
            const auto wait_started = std::chrono::steady_clock::now();
            while (item.handle.wait_for(std::chrono::milliseconds(1)) !=
                   std::future_status::ready) {
              ensure_active();
            }
            auto acquired = item.handle.get();
            sequence_expert_wait_ns_.fetch_add(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wait_started)
                        .count()),
                std::memory_order_relaxed);
            require(acquired.status.ok() && acquired.lease,
                    acquired.status.ok()
                        ? "DeepSeek sequence expert lease is absent"
                        : acquired.status.message());
            wave.push_back(
                {item.expert, std::move(acquired.lease), 0U});
            launch_pending();
          }

          struct WaveTile final {
            std::size_t wave_index{};
            std::size_t first{};
            std::uint32_t rows{};
          };
          const auto execute_started = std::chrono::steady_clock::now();
          for (;;) {
            ensure_active();
            std::vector<WaveTile> tiles;
            tiles.reserve(wave.size());
            for (std::size_t wave_index = 0U; wave_index < wave.size();
                 ++wave_index) {
              auto& item = wave[wave_index];
              const auto& assigned = selections[item.expert];
              if (item.first >= assigned.size()) continue;
              const auto rows = static_cast<std::uint32_t>(
                  std::min<std::size_t>(sequence_tile_rows_,
                                       assigned.size() - item.first));
              auto* input_slot = host.expert_input_window.data() +
                  wave_index * sequence_tile_rows_ * hidden;
              auto* output_slot = host.expert_output_window.data() +
                  wave_index * sequence_tile_rows_ * hidden;
              for (std::uint32_t row = 0U; row < rows; ++row) {
                const auto selection = assigned[item.first + row];
                const auto input_row = selection / top_k;
                std::memcpy(
                    input_slot + static_cast<std::size_t>(row) * hidden,
                    host.ffn_inputs.data() +
                        static_cast<std::size_t>(input_row) * hidden,
                    hidden * sizeof(float));
              }
              copy_async(ffn_workspace.expert_inputs(), input_slot,
                         static_cast<std::size_t>(rows) * hidden *
                             sizeof(float),
                         cudaMemcpyHostToDevice,
                         "upload DeepSeek grouped expert inputs");
              const auto* expert = dynamic_cast<
                  const er::cuda::CudaCompactExpertAllocation*>(
                  item.lease.get());
              require(expert != nullptr,
                      "DeepSeek sequence expert is not direct packed FP4");
              const auto status = er::cuda::deepseek_ffn_execute_packed_batch(
                  {expert, &ffn_workspace, ffn_workspace.expert_inputs(),
                   ffn_workspace.expert_outputs(), rows, 10.0F, true,
                   stream});
              require(status.ok(), status.message());
              copy_async(output_slot, ffn_workspace.expert_outputs(),
                         static_cast<std::size_t>(rows) * hidden *
                             sizeof(float),
                         cudaMemcpyDeviceToHost,
                         "retain DeepSeek grouped expert outputs");
              tiles.push_back({wave_index, item.first, rows});
              item.first += rows;
            }
            if (tiles.empty()) break;
            cuda_check(cudaStreamSynchronize(stream),
                       "complete DeepSeek grouped expert wave");
            for (const auto& tile : tiles) {
              const auto& item = wave[tile.wave_index];
              const auto& assigned = selections[item.expert];
              const auto* output_slot = host.expert_output_window.data() +
                  tile.wave_index * sequence_tile_rows_ * hidden;
              for (std::uint32_t row = 0U; row < tile.rows; ++row) {
                const auto selection = assigned[tile.first + row];
                std::memcpy(
                    host.selection_outputs.data() +
                        static_cast<std::size_t>(selection) * hidden,
                    output_slot + static_cast<std::size_t>(row) * hidden,
                    hidden * sizeof(float));
              }
            }
          }
          sequence_expert_execute_ns_.fetch_add(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - execute_started)
                      .count()),
              std::memory_order_relaxed);
          for (const auto& item : wave) {
            const auto& assigned = selections[item.expert];
            accesses.push_back(
                {routed_->key(component_layer, item.expert),
                 static_cast<std::uint32_t>(std::min<std::size_t>(
                     assigned.size(),
                     std::numeric_limits<std::uint32_t>::max()))});
          }
        }
      } catch (...) {
        cancel_pending();
        throw;
      }
      static_cast<void>(cache_->record_accesses(accesses));

      for (std::uint32_t tile_first = 0U; tile_first < block_rows;
           tile_first += sequence_tile_rows_) {
        ensure_active();
        const auto finalize_started = std::chrono::steady_clock::now();
        const auto rows =
            std::min(sequence_tile_rows_, block_rows - tile_first);
        const auto row_offset = static_cast<std::size_t>(tile_first);
        copy_async(request.sequence_attention_streams,
                   host.attention.data() + row_offset * streams_per_row,
                   static_cast<std::size_t>(rows) * streams_per_row *
                       sizeof(float),
                   cudaMemcpyHostToDevice,
                   "upload DeepSeek sequence FFN residuals");
        copy_async(ffn_workspace.normalized_input(),
                   host.ffn_inputs.data() + row_offset * hidden,
                   static_cast<std::size_t>(rows) * hidden * sizeof(float),
                   cudaMemcpyHostToDevice,
                   "upload DeepSeek sequence FFN inputs");
        copy_async(ffn_workspace.routing_weights(),
                   host.routing_weights.data() + row_offset * top_k,
                   static_cast<std::size_t>(rows) * top_k * sizeof(float),
                   cudaMemcpyHostToDevice,
                   "upload DeepSeek sequence route weights");
        copy_async(ffn_workspace.expert_indices(),
                   host.expert_indices.data() + row_offset * top_k,
                   static_cast<std::size_t>(rows) * top_k *
                       sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice,
                   "upload DeepSeek sequence route indices");
        copy_async(ffn_workspace.selection_outputs(),
                   host.selection_outputs.data() +
                       row_offset * top_k * hidden,
                   static_cast<std::size_t>(rows) * top_k * hidden *
                       sizeof(float),
                   cudaMemcpyHostToDevice,
                   "upload DeepSeek sequence selection outputs");
        copy_async(ffn_workspace.post_control(),
                   host.post.data() + row_offset * kSequenceStreams,
                   static_cast<std::size_t>(rows) * kSequenceStreams *
                       sizeof(float),
                   cudaMemcpyHostToDevice,
                   "upload DeepSeek sequence HCA post control");
        copy_async(
            ffn_workspace.combination_control(),
            host.combination.data() +
                row_offset * kSequenceStreams * kSequenceStreams,
            static_cast<std::size_t>(rows) * kSequenceStreams *
                kSequenceStreams * sizeof(float),
            cudaMemcpyHostToDevice,
            "upload DeepSeek sequence HCA combination control");
        const auto status = er::cuda::deepseek_ffn_finalize_batch(
            {ffn_view.ffn_weights, ffn_view.ffn_state, &ffn_workspace,
             device_entries, request.sequence_attention_streams,
             request.sequence_primary_streams, rows, experts_per_layer,
             stream});
        require(status.ok(), status.message());
        copy_async(host.primary.data() + row_offset * streams_per_row,
                   request.sequence_primary_streams,
                   static_cast<std::size_t>(rows) * streams_per_row *
                       sizeof(float),
                   cudaMemcpyDeviceToHost,
                   "retain DeepSeek sequence layer frontier");
        cuda_check(cudaStreamSynchronize(stream),
                   "complete DeepSeek sequence FFN tile");
        sequence_finalize_ns_.fetch_add(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - finalize_started)
                    .count()),
            std::memory_order_relaxed);
      }
    }

    if (request.exact_decode_enabled) {
      for (std::uint32_t row = 0U; row < block_rows; ++row) {
        const auto global = block_first + row;
        if (global + 1U == sequence.tokens.size()) {
          std::memcpy(request.sequence_final_target_streams.data(),
                      host.primary.data() +
                          static_cast<std::size_t>(row) * streams_per_row,
                      streams_per_row * sizeof(float));
          break;
        }
        copy_async(request.state->primary_streams(),
                   host.primary.data() +
                       static_cast<std::size_t>(row) * streams_per_row,
                   streams_per_row * sizeof(float), cudaMemcpyHostToDevice,
                   "upload DeepSeek exact-sync target streams");
        advance_mtp_state(request, sequence.tokens[global + 1U],
                          request.state->primary_streams(),
                          sequence.positions[global], false);
      }
    }

    if (block_first + block_rows == sequence.tokens.size()) {
      copy_async(request.state->primary_streams(),
                 host.primary.data() +
                     static_cast<std::size_t>(block_rows - 1U) *
                         streams_per_row,
                 streams_per_row * sizeof(float), cudaMemcpyHostToDevice,
                 "commit DeepSeek sequence final streams");
      cuda_check(cudaStreamSynchronize(stream),
                 "complete DeepSeek sequence final-stream commit");
    }
  }

  ensure_active();
  auto status = request.state->project_logits(stream);
  require(status.ok(), status.message());
  auto token = std::make_shared<std::uint32_t>();
  copy_async(token.get(), request.state->sampled_token(), sizeof(*token),
             cudaMemcpyDeviceToHost,
             "copy DeepSeek sequence sampled token");
  cuda_check(cudaStreamSynchronize(stream),
             "complete DeepSeek sequence sampled token");
  request.predicted = *token;
  request.current_token = sequence.tokens.back();
  request.current_position = sequence.positions.back();
  request.next_position = sequence.positions.back() + 1U;
  request.embedded = false;
  request.attention_ready = true;
  request.route_ready = true;

  const auto maximum_per_layer = routed_->component().route_width;
  for (std::uint32_t layer = 0U; layer < request.prompt_routes.size();
       ++layer) {
    std::vector<std::pair<std::uint32_t, PromptRouteEvidence>> ranked(
        request.prompt_routes[layer].begin(),
        request.prompt_routes[layer].end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& left,
                                               const auto& right) {
      if (left.second.count != right.second.count)
        return left.second.count > right.second.count;
      if (left.second.last_seen != right.second.last_seen)
        return left.second.last_seen > right.second.last_seen;
      return left.first < right.first;
    });
    if (ranked.size() > maximum_per_layer)
      ranked.resize(maximum_per_layer);
    for (const auto& [expert, evidence] : ranked) {
      (void)evidence;
      prefill_protection_candidates_.fetch_add(1U,
                                               std::memory_order_relaxed);
      const auto key = routed_->key(layer, expert);
      static_cast<void>(cache_->protect(key, true, true));
      const auto snapshot = cache_->inspect(key);
      if (snapshot &&
          (!snapshot->has_host_copy || snapshot->ram_protected) &&
          (!snapshot->has_device_copy || snapshot->vram_resident))
        prefill_protection_promoted_.fetch_add(1U,
                                              std::memory_order_relaxed);
    }
    request.prompt_routes[layer].clear();
  }
  request.prompt_route_clock = 0U;
  if (request.retention_target_position == request.next_position) {
    const auto checkpoint = capture_retention_checkpoint(
        request, request.next_position, false);
    require(checkpoint.ok(), checkpoint.message());
  }

  er::ExecutionValue output{
      std::string(token_abi), "host", token,
      reinterpret_cast<const std::byte*>(token.get()), sizeof(*token)};
  return operation_result(*sequence.operations.back(),
                          {{"token_ids", std::move(output)}});
}

struct Active final {
  std::unique_ptr<Request> request;
};

std::string sanitize_error(std::string_view message) {
  std::string result;
  result.reserve(message.size());
  for (const char character : message) {
    if (character == '\t' || character == '\n' || character == '\r' ||
        character == '"')
      result += ' ';
    else if (character == '\\')
      result += '/';
    else
      result += character;
  }
  return result;
}

std::uint32_t free_slot(const std::unordered_map<std::uint64_t, Active>& active,
                        const std::unordered_map<std::uint64_t, Active>& retained,
                        std::uint32_t capacity) {
  std::vector<bool> used(capacity);
  for (const auto& [id, item] : active) {
    static_cast<void>(id);
    require(item.request->slot < capacity, "invalid active worker slot");
    used[item.request->slot] = true;
  }
  for (const auto& [key, item] : retained) {
    static_cast<void>(key);
    require(item.request->slot < capacity, "invalid retained worker slot");
    used[item.request->slot] = true;
  }
  const auto available = std::find(used.begin(), used.end(), false);
  require(available != used.end(), "no DeepSeek worker slot available");
  return static_cast<std::uint32_t>(available - used.begin());
}

int worker_loop(Model& model) {
  std::unordered_map<std::uint64_t, Active> active;
  std::unordered_map<std::uint64_t, Active> retained;
  const auto feed_prompt = [&](Request& request,
                               std::span<const std::uint32_t> tokens,
                               std::uint32_t start_position) {
    for (std::size_t index = 0U; index < tokens.size(); ++index) {
      const auto position =
          static_cast<std::uint32_t>(start_position + index);
      Request* pointer = &request;
      request.predicted = model.forward(
          std::span<Request* const>(&pointer, 1U),
          tokens.subspan(index, 1U),
          std::span<const std::uint32_t>(&position, 1U)).front();
      model.observe_prefill_routes(request);
      if (model.mtp_enabled()) {
        const bool final_prompt = index + 1U == tokens.size();
        const auto next_token =
            final_prompt ? request.predicted : tokens[index + 1U];
        model.advance_mtp(request, next_token,
                          request.state->current_streams(), position,
                          final_prompt);
      }
    }
    request.next_position =
        start_position + static_cast<std::uint32_t>(tokens.size());
    model.protect_prefill_routes(request);
  };
  const auto& descriptor = model.descriptor();
  const auto& routed_component =
      required_component(descriptor, "decoder");
  model.start_background_warm();
  model.wait_background_warm();
  std::cout << "{\"type\":\"ready\",\"protocol\":6,\"capacity\":"
            << model.capacity()
            << ",\"architecture_id\":\""
            << sanitize_error(descriptor.architecture_id) << '"'
            << ",\"vocab_size\":" << descriptor.vocab_size
            << ",\"max_context_tokens\":"
            << descriptor.max_context_tokens
            << ",\"routed_layers\":" << routed_component.layer_count
            << ",\"experts_per_layer\":"
            << routed_component.experts_per_layer
            << ",\"route_width\":" << routed_component.route_width
            << ",\"expert_encoding\":\""
            << sanitize_error(routed_component.encoding) << '"'
            << ",\"operation_capabilities\":[";
  for (std::size_t index = 0U; index < descriptor.required_kernels.size();
       ++index) {
    if (index != 0U) std::cout << ',';
    std::cout << '"'
              << sanitize_error(
                     descriptor.required_kernels[index].capability)
              << '"';
  }
  std::cout << ']'
            << ",\"prefill_mode\":\"causal_blocked_exact\""
            << ",\"prefill_chunk_tokens\":"
            << model.sequence_tile_rows()
            << ",\"session_retention\":"
            << (model.supports_request_state_retention() ? "true" : "false")
            << ",\"request_stream_mode\":\"per_request_nonblocking\""
            << ",\"rope_mode\":\"resident_table\""
            << ",\"kv_dtype\":\"bf16\""
            << ",\"kv_allocation\":\"preallocated\""
            << ",\"kv_page_tokens\":" << model.kv_page_tokens()
            << ",\"kv_page_bytes\":" << model.kv_page_bytes()
            << ",\"kv_page_capacity\":" << model.kv_page_capacity()
            << ",\"placement_profile\":\"" << model.placement()
            << "\",\"ram_cache_bytes\":" << model.ram_bytes()
            << ",\"vram_cache_bytes\":" << model.vram_bytes()
            << ",\"placement_prefetch_enabled\":"
            << (model.prefetch_enabled() ? "true" : "false")
            << ",\"placement_prefetch_state\":\""
            << model.prefetch_state() << '\"'
            << ",\"placement_minimum_observations\":"
            << (model.placement() == "latency" ? 1 : 2)
            << ",\"gpu_phase_timing\":"
            << (model.gpu_phase_timing() ? "true" : "false")
            << ",\"mtp_resource_available\":"
            << (model.mtp_available() ? "true" : "false")
            << ",\"mtp_runtime_ready\":"
            << (model.mtp_runtime_ready() ? "true" : "false")
            << ",\"mtp_enabled\":"
            << (model.mtp_enabled() ? "true" : "false")
            << ",\"retain_previous_route\":"
            << (model.retained_route_enabled() ? "true" : "false")
            << ",\"cpu_hybrid_enabled\":"
            << (model.cpu_hybrid_enabled() ? "true" : "false")
            << ",\"route_trace_enabled\":"
            << (model.route_tracing_enabled() ? "true" : "false")
            << ",\"route_trace_max_steps\":"
            << model.route_trace_max_steps()
            << "}\n"
            << std::flush;
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      const auto fields = split_tabs(line);
      if (fields[0] == "PING") {
        require(fields.size() == 1U, "invalid PING");
        std::cout << "{\"type\":\"pong\"}\n" << std::flush;
      } else if (fields[0] == "STATS") {
        require(fields.size() == 1U, "invalid STATS");
        const auto scheduler = model.scheduler_snapshot();
        const auto census = model.census_snapshot();
        const auto cache = model.cache_snapshot();
        const auto mtp_cache = model.mtp_cache_snapshot();
        const auto buffers = model.buffer_snapshot();
        const auto uploader = model.uploader_snapshot();
        const auto cpu = model.cpu_snapshot();
        const auto planner = model.planner_snapshot();
        const auto worker = model.worker_snapshot();
        std::uint64_t reserved_pages = 0U;
        for (const auto& [id, item] : active) {
          static_cast<void>(id);
          reserved_pages +=
              (item.request->context_limit + model.kv_page_tokens() - 1U) /
              model.kv_page_tokens();
        }
        for (const auto& [key, item] : retained) {
          static_cast<void>(key);
          reserved_pages +=
              (item.request->context_limit + model.kv_page_tokens() - 1U) /
              model.kv_page_tokens();
        }
        std::cout << "{\"type\":\"stats\",\"active_requests\":"
                  << active.size() << ",\"retained_sessions\":"
                  << retained.size() << ",\"kv_allocated_pages\":"
                  << (active.size() + retained.size()) *
                         model.kv_pages_per_request()
                  << ",\"kv_reserved_pages\":" << reserved_pages
                  << ",\"route_observations\":"
                  << scheduler.route_observations
                  << ",\"scheduler_prefetch_predictions\":"
                  << scheduler.prefetch_predictions
                  << ",\"scheduler_prefetch_scheduled\":"
                  << scheduler.prefetch_scheduled
                  << ",\"scheduler_prefetch_completed\":"
                  << scheduler.prefetch_completed
                  << ",\"scheduler_prefetch_useful\":"
                  << scheduler.prefetch_useful
                  << ",\"scheduler_prefetch_late\":"
                  << scheduler.prefetch_late
                  << ",\"scheduler_prefetch_incorrect\":"
                  << scheduler.prefetch_incorrect
                  << ",\"scheduler_prefetch_cancelled\":"
                  << scheduler.prefetch_cancelled
                  << ",\"scheduler_prefetch_evicted_before_use\":"
                  << scheduler.prefetch_evicted_before_use
                  << ",\"census_routes\":" << census.completed_routes
                  << ",\"cache_vram_hits\":" << cache.acquire_vram_hits
                  << ",\"cache_ram_hits\":" << cache.acquire_ram_hits
                  << ",\"cache_ssd_misses\":" << cache.acquire_ssd_misses
                  << ",\"cache_loads_started\":" << cache.load_started
                  << ",\"cache_loads_deduplicated\":"
                  << cache.load_deduplicated
                  << ",\"cache_loads_completed\":" << cache.load_completed
                  << ",\"cache_uploads_started\":" << cache.upload_started
                  << ",\"cache_uploads_completed\":" << cache.upload_completed
                  << ",\"cache_record_validations\":"
                  << cache.record_validations
                  << ",\"cache_validated_ram_reuses\":"
                  << cache.validated_ram_reuses
                  << ",\"cache_requested_bytes\":"
                  << cache.requested_bytes
                  << ",\"cache_useful_bytes\":" << cache.useful_bytes
                  << ",\"cache_read_bytes\":" << cache.read_bytes
                  << ",\"cache_uploaded_bytes\":" << cache.uploaded_bytes
                  << ",\"cache_storage_wait_ns\":" << cache.storage_wait_ns
                  << ",\"cache_ram_retention_copy_ns\":"
                  << cache.ram_retention_copy_ns
                  << ",\"cache_upload_wait_ns\":" << cache.upload_wait_ns
                  << ",\"cache_ram_bytes\":" << cache.ram_bytes
                  << ",\"cache_ram_high_water\":" << cache.ram_high_water
                  << ",\"cache_ram_probationary_high_water\":"
                  << cache.ram_probationary_high_water
                  << ",\"cache_ram_protected_high_water\":"
                  << cache.ram_protected_high_water
                  << ",\"cache_vram_bytes\":" << cache.vram_bytes
                  << ",\"cache_vram_high_water\":" << cache.vram_high_water
                  << ",\"cache_vram_resident_high_water\":"
                  << cache.vram_resident_high_water
                  << ",\"cache_vram_transient_high_water\":"
                  << cache.vram_transient_high_water
                  << ",\"cache_staging_bytes\":" << cache.staging_bytes
                  << ",\"cache_staging_high_water\":"
                  << cache.staging_high_water
                  << ",\"cache_evictions\":" << cache.eviction_count
                  << ",\"cache_eviction_scan_calls\":"
                  << cache.eviction_scan_calls
                  << ",\"cache_eviction_scan_candidates\":"
                  << cache.eviction_scan_candidates
                  << ",\"cache_eviction_scan_ns\":"
                  << cache.eviction_scan_ns
                  << ",\"cache_eviction_retire_retries\":"
                  << cache.eviction_retire_retries
                  << ",\"cache_vram_admission_scan_calls\":"
                  << cache.vram_admission_scan_calls
                  << ",\"cache_vram_admission_scan_candidates\":"
                  << cache.vram_admission_scan_candidates
                  << ",\"cache_vram_admission_scan_ns\":"
                  << cache.vram_admission_scan_ns
                  << ",\"cache_task_selection_calls\":"
                  << cache.task_selection_calls
                  << ",\"cache_task_selection_candidates\":"
                  << cache.task_selection_candidates
                  << ",\"cache_task_selection_ns\":"
                  << cache.task_selection_ns
                  << ",\"cache_mutex_acquisitions\":"
                  << cache.mutex_acquisitions
                  << ",\"cache_mutex_wait_ns\":" << cache.mutex_wait_ns
                  << ",\"cache_mutex_wait_max_ns\":"
                  << cache.mutex_wait_max_ns
                  << ",\"cache_same_partition_evictions\":"
                  << cache.same_partition_evictions
                  << ",\"cache_over_quota_evictions\":"
                  << cache.over_quota_evictions
                  << ",\"cache_stalled_by_budget\":"
                  << cache.stalled_by_budget
                  << ",\"cache_cancellations\":"
                  << cache.cancellation_count
                  << ",\"cache_short_read_errors\":"
                  << cache.short_read_errors
                  << ",\"cache_checksum_errors\":"
                  << cache.checksum_errors
                  << ",\"cache_io_errors\":" << cache.io_errors
                  << ",\"cache_upload_errors\":" << cache.upload_errors
                  << ",\"uploader_device_allocations\":"
                  << uploader.device_allocations
                  << ",\"uploader_recycled_acquires\":"
                  << uploader.recycled_acquires
                  << ",\"uploader_staging_allocations\":"
                  << uploader.staging_allocations
                  << ",\"uploader_compact_h2d_bytes\":"
                  << uploader.compact_h2d_bytes
                  << ",\"uploader_compact_cache_hits\":"
                  << uploader.compact_cache_hits
                  << ",\"uploader_compact_cache_misses\":"
                  << uploader.compact_cache_misses
                  << ",\"scheduler_layer_advances\":"
                  << scheduler.layer_advances
                  << ",\"scheduler_cuda_pending_polls\":"
                  << scheduler.cuda_pending_polls
                  << ",\"scheduler_cuda_waits\":" << scheduler.cuda_waits
                  << ",\"scheduler_cuda_wait_ns\":"
                  << scheduler.cuda_wait_ns
                  << ",\"scheduler_expert_suspensions\":"
                  << scheduler.expert_suspensions
                  << ",\"scheduler_acquires_started\":"
                  << scheduler.acquires_started
                  << ",\"scheduler_acquires_completed\":"
                  << scheduler.acquires_completed
                  << ",\"scheduler_host_resolves\":"
                  << scheduler.host_resolves
                  << ",\"scheduler_cpu_placements\":"
                  << scheduler.cpu_placements
                  << ",\"scheduler_hybrid_layers\":"
                  << scheduler.hybrid_layers
                  << ",\"scheduler_controller_advance_ns\":"
                  << scheduler.controller_advance_ns
                  << ",\"scheduler_expert_wait_ns\":"
                  << scheduler.expert_wait_ns
                  << ",\"scheduler_poll_ns\":" << scheduler.poll_ns
                  << ",\"cpu_workers_used_last\":"
                  << cpu.workers_used_last
                  << ",\"cpu_execute_calls\":" << cpu.execute_calls
                  << ",\"cpu_selections\":" << cpu.selections
                  << ",\"cpu_source_weight_bytes\":"
                  << cpu.source_weight_bytes
                  << ",\"cpu_compute_ns\":" << cpu.compute_ns
                  << ",\"planner_plans\":" << planner.plans
                  << ",\"planner_candidates\":" << planner.candidates
                  << ",\"planner_cpu_cost_wins\":"
                  << planner.cpu_cost_wins
                  << ",\"planner_gpu_cost_wins\":"
                  << planner.gpu_cost_wins
                  << ",\"worker_model_steps\":" << worker.model_steps
                  << ",\"worker_model_rows\":" << worker.model_rows
                  << ",\"worker_model_step_ns\":" << worker.model_step_ns
                  << ",\"worker_embed_rope_submit_ns\":"
                  << worker.embed_rope_submit_ns
                  << ",\"worker_scheduler_poll_ns\":"
                  << worker.scheduler_poll_ns
                  << ",\"worker_output_head_ns\":"
                  << worker.output_head_ns
                  << ",\"worker_attention_route_submit_ns\":"
                  << worker.attention_route_submit_ns
                  << ",\"worker_directory_plan_ns\":"
                  << worker.directory_plan_ns
                  << ",\"worker_ffn_submit_ns\":"
                  << worker.ffn_submit_ns
                  << ",\"worker_directory_release_ns\":"
                  << worker.directory_release_ns
                  << ",\"worker_gpu_attention_route_plan_ns\":"
                  << worker.gpu_attention_route_plan_ns
                  << ",\"worker_gpu_ffn_release_ns\":"
                  << worker.gpu_ffn_release_ns
                  << ",\"worker_gpu_attention_ns\":"
                  << worker.gpu_attention_ns
                  << ",\"worker_gpu_route_ns\":"
                  << worker.gpu_route_ns
                  << ",\"worker_gpu_directory_plan_ns\":"
                  << worker.gpu_directory_plan_ns
                  << ",\"worker_gpu_ffn_ns\":"
                  << worker.gpu_ffn_ns
                  << ",\"worker_gpu_directory_release_ns\":"
                  << worker.gpu_directory_release_ns
                  << ",\"worker_gpu_attention_hca_pre_norm_ns\":"
                  << worker.gpu_attention_hca_pre_norm_ns
                  << ",\"worker_gpu_attention_projection_ns\":"
                  << worker.gpu_attention_projection_ns
                  << ",\"worker_gpu_sparse_attention_ns\":"
                  << worker.gpu_sparse_attention_ns
                  << ",\"worker_gpu_attention_output_projection_ns\":"
                  << worker.gpu_attention_output_projection_ns
                  << ",\"worker_gpu_attention_hca_post_ns\":"
                  << worker.gpu_attention_hca_post_ns
                  << ",\"worker_gpu_ffn_routed_ns\":"
                  << worker.gpu_ffn_routed_ns
                  << ",\"worker_gpu_ffn_aggregate_ns\":"
                  << worker.gpu_ffn_aggregate_ns
                  << ",\"worker_gpu_ffn_shared_ns\":"
                  << worker.gpu_ffn_shared_ns
                  << ",\"worker_gpu_ffn_merge_ns\":"
                  << worker.gpu_ffn_merge_ns
                  << ",\"worker_gpu_ffn_hca_post_ns\":"
                  << worker.gpu_ffn_hca_post_ns
                  << ",\"worker_warm_start_candidates\":"
                  << worker.warm_start_candidates
                  << ",\"worker_warm_start_loaded\":"
                  << worker.warm_start_loaded
                  << ",\"worker_warm_start_bytes\":"
                  << worker.warm_start_bytes
                  << ",\"worker_warm_start_ns\":"
                  << worker.warm_start_ns
                  << ",\"worker_warm_vram_candidates\":"
                  << worker.warm_vram_candidates
                  << ",\"worker_warm_vram_loaded\":"
                  << worker.warm_vram_loaded
                  << ",\"worker_warm_vram_failed\":"
                  << worker.warm_vram_failed
                  << ",\"worker_warm_vram_cancelled\":"
                  << worker.warm_vram_cancelled
                  << ",\"worker_warm_vram_demand_pauses\":"
                  << worker.warm_vram_demand_pauses
                  << ",\"worker_warm_vram_inflight_max\":"
                  << worker.warm_vram_inflight_max
                  << ",\"worker_warm_vram_bytes\":"
                  << worker.warm_vram_bytes
                  << ",\"worker_warm_vram_ns\":"
                  << worker.warm_vram_ns
                  << ",\"worker_mtp_drafts\":" << worker.mtp_drafts
                  << ",\"worker_mtp_accepted\":" << worker.mtp_accepted
                  << ",\"worker_mtp_rejected\":" << worker.mtp_rejected
                  << ",\"worker_verify_pairs\":" << worker.verify_pairs
                  << ",\"worker_useful_tokens\":" << worker.useful_tokens
                  << ",\"worker_mtp_suppressions\":"
                  << worker.mtp_suppressions
                  << ",\"worker_warm_start_failed\":"
                  << worker.warm_start_failed
                  << ",\"worker_warm_start_cancelled\":"
                  << worker.warm_start_cancelled
                  << ",\"worker_warm_start_demand_pauses\":"
                  << worker.warm_start_demand_pauses
                  << ",\"worker_warm_start_inflight_max\":"
                  << worker.warm_start_inflight_max
                  << ",\"worker_warm_start_loop_errors\":"
                  << worker.warm_start_loop_errors
                  << ",\"worker_mtp_acquire_batches\":"
                  << worker.mtp_acquire_batches
                  << ",\"worker_mtp_acquires_launched\":"
                  << worker.mtp_acquires_launched
                  << ",\"worker_mtp_acquire_batch_width_max\":"
                  << worker.mtp_acquire_batch_width_max
                  << ",\"worker_mtp_acquire_wait_ns\":"
                  << worker.mtp_acquire_wait_ns
                  << ",\"mtp_cache_vram_hits\":"
                  << mtp_cache.acquire_vram_hits
                  << ",\"mtp_cache_ram_hits\":"
                  << mtp_cache.acquire_ram_hits
                  << ",\"mtp_cache_ssd_misses\":"
                  << mtp_cache.acquire_ssd_misses
                  << ",\"mtp_cache_loads_started\":"
                  << mtp_cache.load_started
                  << ",\"mtp_cache_loads_deduplicated\":"
                  << mtp_cache.load_deduplicated
                  << ",\"mtp_cache_loads_completed\":"
                  << mtp_cache.load_completed
                  << ",\"mtp_cache_reload_count\":"
                  << mtp_cache.reload_count
                  << ",\"mtp_cache_reread_bytes\":"
                  << mtp_cache.reread_bytes
                  << ",\"mtp_cache_read_bytes\":"
                  << mtp_cache.read_bytes
                  << ",\"mtp_cache_storage_wait_ns\":"
                  << mtp_cache.storage_wait_ns
                  << ",\"mtp_cache_host_validation_ns\":"
                  << mtp_cache.host_validation_ns
                  << ",\"mtp_cache_host_copy_bytes\":"
                  << mtp_cache.host_copy_bytes
                  << ",\"mtp_cache_uploads_started\":"
                  << mtp_cache.upload_started
                  << ",\"mtp_cache_uploads_completed\":"
                  << mtp_cache.upload_completed
                  << ",\"mtp_cache_uploaded_bytes\":"
                  << mtp_cache.uploaded_bytes
                  << ",\"mtp_cache_upload_wait_ns\":"
                  << mtp_cache.upload_wait_ns
                  << ",\"mtp_cache_ram_bytes\":" << mtp_cache.ram_bytes
                  << ",\"mtp_cache_vram_bytes\":" << mtp_cache.vram_bytes
                  << ",\"mtp_cache_evictions\":"
                  << mtp_cache.eviction_count
                  << ",\"mtp_cache_stalled_by_budget\":"
                  << mtp_cache.stalled_by_budget
                  << ",\"mtp_cache_cancellations\":"
                  << mtp_cache.cancellation_count
                  << ",\"mtp_cache_io_errors\":" << mtp_cache.io_errors
                  << ",\"mtp_cache_upload_errors\":"
                  << mtp_cache.upload_errors
                  << ",\"worker_prefill_protection_candidates\":"
                  << worker.prefill_protection_candidates
                  << ",\"worker_prefill_protection_promoted\":"
                  << worker.prefill_protection_promoted
                  << ",\"scheduler_retained_working_set_experts\":"
                  << scheduler.retained_working_set_experts
                  << ",\"cache_reload_count\":" << cache.reload_count
                  << ",\"cache_reread_bytes\":" << cache.reread_bytes
                  << ",\"cache_priority_upgrades\":"
                  << cache.priority_upgrades
                  << ",\"cache_host_preloads_requested\":"
                  << cache.host_preloads_requested
                  << ",\"cache_host_preloads_completed\":"
                  << cache.host_preloads_completed
                  << ",\"cache_host_validation_ns\":"
                  << cache.host_validation_ns
                  << ",\"cache_host_validation_failures\":"
                  << cache.host_validation_failures
                  << ",\"cache_host_copy_bytes\":"
                  << cache.host_copy_bytes
                  << ",\"cache_preloaded_host_useful\":"
                  << cache.preloaded_host_useful
                  << ",\"cache_preloaded_host_useful_bytes\":"
                  << cache.preloaded_host_useful_bytes
                  << ",\"cache_preloaded_host_wasted\":"
                  << cache.preloaded_host_wasted
                  << ",\"cache_preloaded_host_wasted_bytes\":"
                  << cache.preloaded_host_wasted_bytes
                  << ",\"cache_ram_probationary_bytes\":"
                  << cache.ram_probationary_bytes
                  << ",\"cache_ram_protected_bytes\":"
                  << cache.ram_protected_bytes
                  << ",\"cache_vram_resident_bytes\":"
                  << cache.vram_resident_bytes
                  << ",\"cache_vram_transient_bytes\":"
                  << cache.vram_transient_bytes
                  << ",\"cache_vram_referenced_entries\":"
                  << cache.vram_referenced_entries
                  << ",\"cache_vram_referenced_bytes\":"
                  << cache.vram_referenced_bytes
                  << ",\"cache_vram_transient_referenced_bytes\":"
                  << cache.vram_transient_referenced_bytes
                  << ",\"cache_ram_promotions\":" << cache.ram_promotions
                  << ",\"cache_vram_promotions\":" << cache.vram_promotions
                  << ",\"cache_ram_promotion_failures\":"
                  << cache.ram_promotion_failures
                  << ",\"cache_vram_promotion_failures\":"
                  << cache.vram_promotion_failures
                  << ",\"cache_ram_probationary_evictions\":"
                  << cache.ram_evictions_by_class[0]
                  << ",\"cache_ram_protected_evictions\":"
                  << cache.ram_evictions_by_class[1]
                  << ",\"cache_ram_probationary_evicted_bytes\":"
                  << cache.ram_evicted_bytes_by_class[0]
                  << ",\"cache_ram_protected_evicted_bytes\":"
                  << cache.ram_evicted_bytes_by_class[1]
                  << ",\"cache_vram_transient_evictions\":"
                  << cache.vram_evictions_by_class[0]
                  << ",\"cache_vram_resident_evictions\":"
                  << cache.vram_evictions_by_class[1]
                  << ",\"cache_vram_transient_evicted_bytes\":"
                  << cache.vram_evicted_bytes_by_class[0]
                  << ",\"cache_vram_resident_evicted_bytes\":"
                  << cache.vram_evicted_bytes_by_class[1]
                  << ",\"staging_slots_in_use\":" << buffers.slots_in_use
                  << ",\"staging_demand_slots_in_use\":"
                  << buffers.demand_slots_in_use
                  << ",\"staging_background_slots_in_use\":"
                  << buffers.background_slots_in_use
                  << ",\"staging_high_water_slots\":"
                  << buffers.high_water_slots
                  << ",\"staging_background_high_water_slots\":"
                  << buffers.background_high_water_slots
                  << ",\"staging_demand_acquires\":"
                  << buffers.demand_acquires
                  << ",\"staging_background_acquires\":"
                  << buffers.background_acquires
                  << ",\"staging_demand_stalls\":"
                  << buffers.demand_stalls
                  << ",\"staging_background_stalls\":"
                  << buffers.background_stalls;
        constexpr std::array<std::string_view, er::kExpertPriorityCount>
            priority_names{"warm", "prefetch", "demand"};
        for (std::size_t priority = 0U; priority < priority_names.size();
             ++priority) {
          const auto name = priority_names[priority];
          std::cout
              << ",\"cache_" << name << "_device_requests\":"
              << cache.device_requests[priority]
              << ",\"cache_" << name << "_host_requests\":"
              << cache.host_requests[priority]
              << ",\"cache_" << name << "_vram_hits\":"
              << cache.vram_hits_by_priority[priority]
              << ",\"cache_" << name << "_ram_hits\":"
              << cache.ram_hits_by_priority[priority]
              << ",\"cache_" << name << "_ssd_misses\":"
              << cache.ssd_misses_by_priority[priority]
              << ",\"cache_" << name << "_host_lookup_misses\":"
              << cache.host_lookup_misses_by_priority[priority]
              << ",\"cache_" << name << "_reads_started\":"
              << cache.reads_started_by_priority[priority]
              << ",\"cache_" << name << "_reads_completed\":"
              << cache.reads_completed_by_priority[priority]
              << ",\"cache_" << name << "_read_bytes\":"
              << cache.read_bytes_by_priority[priority]
              << ",\"cache_" << name << "_storage_wait_ns\":"
              << cache.storage_wait_ns_by_priority[priority]
              << ",\"cache_" << name << "_host_validation_ns\":"
              << cache.host_validation_ns_by_priority[priority]
              << ",\"cache_" << name << "_host_copy_bytes\":"
              << cache.host_copy_bytes_by_priority[priority]
              << ",\"cache_" << name << "_ram_retention_copy_ns\":"
              << cache.ram_retention_copy_ns_by_priority[priority]
              << ",\"cache_" << name << "_uploads_started\":"
              << cache.uploads_started_by_priority[priority]
              << ",\"cache_" << name << "_uploads_completed\":"
              << cache.uploads_completed_by_priority[priority]
              << ",\"cache_" << name << "_uploaded_bytes\":"
              << cache.uploaded_bytes_by_priority[priority]
              << ",\"cache_" << name << "_upload_wait_ns\":"
              << cache.upload_wait_ns_by_priority[priority]
              << ",\"cache_" << name << "_completed_waiters\":"
              << cache.completed_waiters_by_priority[priority]
              << ",\"cache_" << name << "_waiter_wait_ns\":"
              << cache.waiter_wait_ns_by_priority[priority]
              << ",\"cache_" << name << "_cancellations\":"
              << cache.cancellations_by_priority[priority]
              << ",\"cache_" << name << "_failed_waiters\":"
              << cache.failed_waiters_by_priority[priority]
              << ",\"cache_" << name << "_io_errors\":"
              << cache.io_errors_by_priority[priority]
              << ",\"cache_" << name << "_validation_errors\":"
              << cache.validation_errors_by_priority[priority]
              << ",\"cache_" << name << "_upload_errors\":"
              << cache.upload_errors_by_priority[priority]
              << ",\"cache_" << name << "_staging_stalls\":"
              << cache.staging_stalls_by_priority[priority];
        }
        constexpr std::array<std::string_view, 6U> cache_state_names{
            "absent", "ssd_loading", "ram_ready", "gpu_uploading",
            "vram_ready", "failed"};
        for (std::size_t from = 0U; from < cache_state_names.size(); ++from) {
          for (std::size_t to = 0U; to < cache_state_names.size(); ++to) {
            std::cout << ",\"cache_transition_" << cache_state_names[from]
                      << "_to_" << cache_state_names[to] << "\":"
                      << cache.state_transitions[
                             from * cache_state_names.size() + to];
          }
        }
        std::cout << "}\n" << std::flush;
      } else if (fields[0] == "BEGIN") {
        require(fields.size() == 4U || fields.size() == 6U, "invalid BEGIN");
        const auto id = std::stoull(std::string(fields[1]));
        const auto context = std::stoull(std::string(fields[2]));
        require(id != 0U && !active.contains(id) &&
                    context <= model.max_context(),
                "invalid BEGIN");
        const auto prompt = parse_tokens(fields[3], model.vocab_size());
        if (fields.size() == 6U) {
          require(fields[4] == "RESUME", "invalid BEGIN resume marker");
          const auto key = std::stoull(std::string(fields[5]));
          auto retained_iterator = retained.find(key);
          require(retained_iterator != retained.end(),
                  "unknown retained session");
          auto request = std::move(retained_iterator->second.request);
          retained.erase(retained_iterator);
          try {
            require(request->next_position + prompt.size() <= context,
                    "prompt exceeds reserved context");
            request->context_limit = std::max(
                request->context_limit, static_cast<std::uint32_t>(context));
            feed_prompt(*request, prompt, request->next_position);
            // A resumed turn re-evaluates speculation on fresh evidence: the
            // global cost EMA still guards, so a genuinely unprofitable draft
            // loop is suppressed again after the first new verify pair.
            request->speculation_suppressed = false;
            const auto slot = request->slot;
            active.emplace(id, Active{std::move(request)});
            std::cout << "{\"type\":\"begun\",\"id\":" << id
                      << ",\"slot\":" << slot << "}\n" << std::flush;
          } catch (...) {
            // A failed resume leaves the request state partially
            // overwritten; destroying it frees the slot for a fresh prefill.
            throw;
          }
        } else {
          require(active.size() + retained.size() < model.capacity(),
                  "over-capacity BEGIN");
          auto request = model.create_request();
          require(prompt.size() <= context, "prompt exceeds reserved context");
          request->context_limit = static_cast<std::uint32_t>(context);
          request->slot = free_slot(active, retained, model.capacity());
          feed_prompt(*request, prompt, 0U);
          const auto slot = request->slot;
          active.emplace(id, Active{std::move(request)});
          std::cout << "{\"type\":\"begun\",\"id\":" << id
                    << ",\"slot\":" << slot << "}\n" << std::flush;
        }
      } else if (fields[0] == "NEXT" || fields[0] == "STEP") {
        // Step modes: 0 = decode (speculative when MTP is active), 1 = final
        // emit-and-release, 2 = plain non-speculative decode that keeps the
        // slot. Mode 2 exists so a retained turn can end on an exact token
        // boundary: an accepted speculative pair would leave an unemitted
        // bonus token in the worker state that no client-echoed prompt can
        // match, poisoning session retention.
        std::vector<std::pair<std::uint64_t, std::uint32_t>> steps;
        std::set<std::uint64_t> unique_ids;
        const auto add = [&](std::string_view field) {
          const auto comma = field.find(',');
          require(comma != std::string_view::npos, "invalid step item");
          const auto id = std::stoull(std::string(field.substr(0U, comma)));
          const auto flag = field.substr(comma + 1U);
          require(active.contains(id) && unique_ids.insert(id).second &&
                      (flag == "0" || flag == "1" || flag == "2"),
                  "step request mismatch");
          steps.emplace_back(id, static_cast<std::uint32_t>(
                                     flag == "1" ? 1U : flag == "2" ? 2U : 0U));
        };
        if (fields[0] == "NEXT") {
          require(fields.size() == 3U, "invalid NEXT");
          add(std::string(fields[1]) + "," + std::string(fields[2]));
        } else {
          require(fields.size() >= 2U &&
                      fields.size() <= model.capacity() + 1U,
                  "invalid STEP");
          for (std::size_t index = 1U; index < fields.size(); ++index)
            add(fields[index]);
        }
        std::vector<std::vector<std::uint32_t>> emitted(steps.size());
        if (model.mtp_enabled()) {
          for (std::size_t index = 0U; index < steps.size(); ++index) {
            const auto [id, mode] = steps[index];
            auto& request = *active.at(id).request;
            if (mode == 1U) {
              emitted[index] = {request.predicted};
              continue;
            }
            if (mode == 0U && model.speculation_active(request)) {
              emitted[index] = model.verify_draft(request);
            } else {
              emitted[index] = {request.predicted};
              Request* pointer = &request;
              const auto token = request.predicted;
              const auto position = request.next_position;
              request.predicted = model.forward(
                  std::span<Request* const>(&pointer, 1U),
                  std::span<const std::uint32_t>(&token, 1U),
                  std::span<const std::uint32_t>(&position, 1U)).front();
              if (mode == 2U) {
                // Keep the MTP causal stream in lockstep with the target
                // model so the next turn's resume starts with a valid draft.
                model.advance_mtp(request, request.predicted,
                                  request.state->current_streams(), position,
                                  true);
              }
              ++request.next_position;
            }
          }
        } else {
          std::vector<Request*> advancing;
          std::vector<std::uint32_t> tokens, positions;
          for (std::size_t index = 0U; index < steps.size(); ++index) {
            const auto [id, mode] = steps[index];
            auto& request = *active.at(id).request;
            emitted[index] = {request.predicted};
            if (mode == 1U) continue;
            advancing.push_back(&request);
            tokens.push_back(request.predicted);
            positions.push_back(request.next_position);
          }
          if (!advancing.empty()) {
            const auto predicted = model.forward(advancing, tokens, positions);
            std::size_t predicted_index = 0U;
            for (const auto& [id, mode] : steps) {
              if (mode == 1U) continue;
              active.at(id).request->predicted =
                  predicted[predicted_index++];
              ++active.at(id).request->next_position;
            }
          }
        }
        if (fields[0] == "NEXT") {
          std::cout << "{\"type\":\"token\",\"id\":" << steps[0].first
                    << ",\"tokens\":[";
          for (std::size_t token = 0U; token < emitted[0].size(); ++token) {
            if (token) std::cout << ',';
            std::cout << emitted[0][token];
          }
          std::cout << "]}\n";
        } else {
          std::cout << "{\"type\":\"batch\",\"items\":[";
          for (std::size_t index = 0U; index < steps.size(); ++index) {
            if (index) std::cout << ',';
            const auto id = steps[index].first;
            std::cout << "{\"id\":" << id << ",\"tokens\":[";
            for (std::size_t token = 0U; token < emitted[index].size();
                 ++token) {
              if (token) std::cout << ',';
              std::cout << emitted[index][token];
            }
            std::cout << "]}";
          }
          std::cout << "]}\n";
        }
        for (const auto& tokens : emitted)
          model.record_useful_tokens(tokens.size());
        std::cout << std::flush;
        for (const auto& [id, mode] : steps) {
          if (mode != 1U) continue;
          auto iterator = active.find(id);
          require(iterator != active.end(), "final request disappeared");
          model.emit_route_trace(id, *iterator->second.request);
          active.erase(iterator);
        }
      } else if (fields[0] == "END") {
        require(fields.size() == 2U || fields.size() == 4U,
                "invalid END");
        const auto id = std::stoull(std::string(fields[1]));
        if (fields.size() == 4U) {
          require(fields[2] == "RETAIN", "invalid END retain marker");
          const auto key = std::stoull(std::string(fields[3]));
          require(!retained.contains(key), "duplicate retained session");
          auto iterator = active.find(id);
          require(id != 0U && iterator != active.end(),
                  "END request mismatch");
          const auto tokens = iterator->second.request->next_position;
          model.emit_route_trace(id, *iterator->second.request);
          retained.emplace(key, Active{std::move(iterator->second.request)});
          active.erase(iterator);
          std::cout << "{\"type\":\"ended\",\"id\":" << id
                    << ",\"retained_tokens\":" << tokens << "}\n"
                    << std::flush;
        } else {
          auto iterator = active.find(id);
          require(id != 0U && iterator != active.end(),
                  "END request mismatch");
          model.emit_route_trace(id, *iterator->second.request);
          active.erase(iterator);
          std::cout << "{\"type\":\"ended\",\"id\":" << id << "}\n"
                    << std::flush;
        }
      } else if (fields[0] == "DROP") {
        require(fields.size() == 2U, "invalid DROP");
        const auto key = std::stoull(std::string(fields[1]));
        const bool found = retained.erase(key) == 1U;
        std::cout << "{\"type\":\"dropped\",\"key\":" << key
                  << ",\"found\":" << (found ? "true" : "false") << "}\n"
                  << std::flush;
      } else if (fields[0] == "SHUTDOWN") {
        require(fields.size() == 1U && active.empty() && retained.empty(),
                "invalid SHUTDOWN");
        std::cout << "{\"type\":\"shutdown\"}\n" << std::flush;
        return 0;
      } else {
        throw std::runtime_error("unknown worker command");
      }
    } catch (const std::exception& error) {
      std::cerr << "worker command failed: " << error.what() << '\n';
      std::cout << "{\"type\":\"error\",\"active_requests\":"
                << active.size() << ",\"message\":\""
                << sanitize_error(error.what()) << "\"}\n" << std::flush;
    }
  }
  return 0;
}

}  // namespace

int expert_vm_compressed_sparse_moe_provider_main(int argc, char** argv) {
  try {
    if (argc < 3 || std::string_view(argv[2]) != "--worker") {
      std::cerr << "usage: expert-deepseek-worker <bundle> --worker "
                   "--max-context=N --ram-cache-gib=N --vram-cache-gib=N "
                   "--capacity=N --kv-cache-mib=N --kv-page-tokens=N "
                   "--placement-profile=NAME "
                   "[--profile-gpu-phases] "
                   "[--no-retain-previous-route] [--cpu-hybrid] "
                   "[--no-cpu-hybrid] "
                   "[--route-trace-file=<path>] "
                   "[--route-trace-max-steps=<count>]\n";
      return 64;
    }
    std::vector<std::string_view> raw_options;
    raw_options.reserve(static_cast<std::size_t>(argc - 3));
    for (int index = 3; index < argc; ++index)
      raw_options.emplace_back(argv[index]);
    auto parsed = er::parse_worker_launch_options(raw_options);
    require(parsed.status.ok(), parsed.status.message());
    auto options = std::move(parsed.options);
    const auto boolean_extension = [&](std::string_view name) {
      const auto found = options.extensions.find(name);
      if (found == options.extensions.end()) return false;
      require(!found->second, "boolean worker extension has a value");
      options.extensions.erase(found);
      return true;
    };
    const auto value_extension = [&](std::string_view name)
        -> std::optional<std::string> {
      const auto found = options.extensions.find(name);
      if (found == options.extensions.end()) return std::nullopt;
      require(found->second.has_value(), "worker extension requires a value");
      auto value = std::move(*found->second);
      options.extensions.erase(found);
      return value;
    };
    const bool profile_gpu_phases = options.profile_gpu_phases;
    const bool retain_previous_route =
        !boolean_extension("no-retain-previous-route");
    const bool requested_cpu_hybrid = boolean_extension("cpu-hybrid");
    const bool disable_cpu_hybrid = boolean_extension("no-cpu-hybrid");
    require(!(requested_cpu_hybrid && disable_cpu_hybrid),
            "conflicting CPU-hybrid worker extensions");
    // This provider owns an exact FP4 host executor and an adaptive split
    // planner, so hybrid execution is a provider capability rather than a
    // deployment experiment. Keep a direct-worker opt-out for diagnosis.
    const bool enable_cpu_hybrid = !disable_cpu_hybrid;
    require(!options.placement_settle_steps,
            "execution provider does not support placement settling");
    std::filesystem::path route_trace_path;
    std::size_t route_trace_max_steps = 4096U;
    if (auto trace = value_extension("route-trace-file"))
      route_trace_path = std::move(*trace);
    if (auto bound = value_extension("route-trace-max-steps")) {
      const auto value = std::stoull(*bound);
      require(value != 0U && value <= std::numeric_limits<std::size_t>::max(),
              "invalid route trace step bound");
      route_trace_max_steps = static_cast<std::size_t>(value);
    }
    require(options.extensions.empty(), "unsupported worker extension");
    require(options.ram_cache_gib <=
                (std::numeric_limits<std::uint64_t>::max() >> 30U) &&
                options.vram_cache_gib <=
                    (std::numeric_limits<std::uint64_t>::max() >> 30U) &&
                options.kv_cache_mib <=
                    (std::numeric_limits<std::uint64_t>::max() >> 20U),
            "worker resource bytes overflow");
    Model model(argv[1], options.max_context,
                options.ram_cache_gib << 30U,
                options.vram_cache_gib << 30U, options.capacity,
                options.kv_cache_mib << 20U, options.kv_page_tokens,
                options.placement_profile,
                profile_gpu_phases, true, retain_previous_route,
                enable_cpu_hybrid, enable_cpu_hybrid,
                std::move(route_trace_path),
                route_trace_max_steps);
    return worker_loop(model);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

expert::runtime::WorkerProviderDefinition
make_sm86_compressed_sparse_moe_provider() {
  return {"sm86-compressed-sparse-moe", 100U, provider_capabilities(),
          &expert_vm_compressed_sparse_moe_provider_main};
}

expert::runtime::CreateExecutionProviderModuleResult
make_sm86_compressed_sparse_moe_callable_provider(
    const std::filesystem::path& artifact_root, std::uint32_t max_context,
    std::uint32_t capacity, std::uint64_t ram_cache_bytes,
    std::uint64_t vram_cache_bytes, std::uint64_t kv_cache_bytes,
    std::uint32_t kv_page_tokens, std::string_view placement_profile,
    std::shared_ptr<const expert::runtime::ActiveExpertOwnerDirectory>
        remote_owners) {
  try {
    // Callable requests release exact route leases after each routed
    // operation. Reserving the legacy scheduler's full previous-route tier
    // therefore strands one route per layer without providing its pin/reuse
    // contract. Keep only the cache's bounded two-route transient ring; the
    // remaining VRAM is available to the authenticated census-hot set.
    constexpr bool callable_retains_previous_route = false;
    const bool callable_cpu_fallback = placement_profile != "capacity";
    auto implementation = std::make_shared<Model>(
        artifact_root, max_context, ram_cache_bytes, vram_cache_bytes,
        capacity, kv_cache_bytes, kv_page_tokens,
        std::string(placement_profile), false, true,
        callable_retains_previous_route, callable_cpu_fallback,
        callable_cpu_fallback,
        std::filesystem::path{}, 0U, std::move(remote_owners));
    implementation->start_background_warm();
    implementation->wait_background_warm();
    expert::runtime::ExecutionProviderModule module;
    module.definition = {"sm86-compressed-sparse-moe", 100U,
                         provider_capabilities(), implementation};
    module.service = {
        "causal_blocked_exact", implementation->sequence_tile_rows(),
        implementation->supports_request_state_retention(),
        "per_request_nonblocking",
        "resident_table", "bf16", "preallocated", kv_page_tokens,
        implementation->kv_page_bytes(), implementation->kv_page_capacity(),
        "budgeted", implementation->placement(), implementation->ram_bytes(),
        implementation->vram_bytes(), implementation->prefetch_enabled(),
        implementation->prefetch_state(),
        implementation->placement() == "latency" ? 1U : 2U,
        implementation->mtp_available(), implementation->mtp_runtime_ready(),
        implementation->mtp_enabled(),
        implementation->retained_route_enabled(),
        implementation->cpu_hybrid_enabled()};
    module.telemetry = [implementation] {
      const auto cache = implementation->cache_snapshot();
      const auto mtp_cache = implementation->mtp_cache_snapshot();
      const auto worker = implementation->worker_snapshot();
      const auto cpu = implementation->cpu_snapshot();
      const auto planner = implementation->planner_snapshot();
      return std::map<std::string, std::uint64_t, std::less<>>{
          {"cache_vram_hits", cache.acquire_vram_hits},
          {"cache_ram_hits", cache.acquire_ram_hits},
          {"cache_ssd_misses", cache.acquire_ssd_misses},
          {"cache_read_bytes", cache.read_bytes},
          {"cache_uploaded_bytes", cache.uploaded_bytes},
          {"cache_storage_wait_ns", cache.storage_wait_ns},
          {"cache_upload_wait_ns", cache.upload_wait_ns},
          {"cache_prefetch_device_requests", cache.device_requests[1]},
          {"cache_prefetch_vram_hits", cache.vram_hits_by_priority[1]},
          {"cache_prefetch_ram_hits", cache.ram_hits_by_priority[1]},
          {"cache_prefetch_ssd_misses", cache.ssd_misses_by_priority[1]},
          {"cache_prefetch_reads_started",
           cache.reads_started_by_priority[1]},
          {"cache_prefetch_reads_completed",
           cache.reads_completed_by_priority[1]},
          {"cache_prefetch_read_bytes", cache.read_bytes_by_priority[1]},
          {"cache_prefetch_storage_wait_ns",
           cache.storage_wait_ns_by_priority[1]},
          {"cache_prefetch_uploads_started",
           cache.uploads_started_by_priority[1]},
          {"cache_prefetch_uploads_completed",
           cache.uploads_completed_by_priority[1]},
          {"cache_prefetch_uploaded_bytes",
           cache.uploaded_bytes_by_priority[1]},
          {"cache_prefetch_upload_wait_ns",
           cache.upload_wait_ns_by_priority[1]},
          {"cache_prefetch_waiter_wait_ns",
           cache.waiter_wait_ns_by_priority[1]},
          {"cache_prefetch_cancellations",
           cache.cancellations_by_priority[1]},
          {"cache_prefetch_failed_waiters",
           cache.failed_waiters_by_priority[1]},
          {"cache_prefetch_staging_stalls",
           cache.staging_stalls_by_priority[1]},
          {"cache_priority_upgrades", cache.priority_upgrades},
          {"cache_reload_count", cache.reload_count},
          {"cache_reread_bytes", cache.reread_bytes},
          {"worker_model_steps", worker.model_steps},
          {"worker_model_step_ns", worker.model_step_ns},
          {"worker_mtp_drafts", worker.mtp_drafts},
          {"worker_mtp_accepted", worker.mtp_accepted},
          {"worker_mtp_rejected", worker.mtp_rejected},
          {"worker_verify_pairs", worker.verify_pairs},
          {"worker_mtp_suppressions", worker.mtp_suppressions},
          {"worker_mtp_acquire_batches", worker.mtp_acquire_batches},
          {"worker_mtp_acquires_launched",
           worker.mtp_acquires_launched},
          {"worker_mtp_acquire_batch_width_max",
           worker.mtp_acquire_batch_width_max},
          {"worker_mtp_acquire_wait_ns", worker.mtp_acquire_wait_ns},
          {"mtp_cache_vram_hits", mtp_cache.acquire_vram_hits},
          {"mtp_cache_ram_hits", mtp_cache.acquire_ram_hits},
          {"mtp_cache_ssd_misses", mtp_cache.acquire_ssd_misses},
          {"mtp_cache_read_bytes", mtp_cache.read_bytes},
          {"mtp_cache_storage_wait_ns", mtp_cache.storage_wait_ns},
          {"mtp_cache_uploaded_bytes", mtp_cache.uploaded_bytes},
          {"mtp_cache_upload_wait_ns", mtp_cache.upload_wait_ns},
          {"mtp_cache_ram_bytes", mtp_cache.ram_bytes},
          {"mtp_cache_vram_bytes", mtp_cache.vram_bytes},
          {"active_expert_selection_launches",
           worker.callable_selection_launches},
          {"active_expert_selections", worker.callable_selections},
          {"active_expert_overlap_launches",
           worker.callable_overlap_launches},
          {"active_expert_cpu_launches", worker.callable_cpu_launches},
          {"active_expert_cpu_selections", worker.callable_cpu_selections},
          {"active_expert_cpu_activation_bytes",
           worker.callable_cpu_activation_bytes},
          {"active_expert_cpu_output_bytes",
           worker.callable_cpu_output_bytes},
          {"active_expert_cpu_host_resolves_launched",
           worker.callable_cpu_host_resolves_launched},
          {"active_expert_cpu_host_resolves_completed",
           worker.callable_cpu_host_resolves_completed},
          {"active_expert_cpu_device_admission_fallbacks",
           worker.callable_cpu_device_admission_fallbacks},
          {"cache_device_admission_rejections",
           cache.device_admission_rejections},
          {"active_expert_cpu_compute_ns", cpu.compute_ns},
          {"active_expert_cpu_source_weight_bytes", cpu.source_weight_bytes},
          {"active_expert_cpu_ns_per_selection",
           static_cast<std::uint64_t>(std::llround(
               planner.cpu_ns_per_selection))},
          {"active_expert_planner_cpu_choices", planner.cpu_cost_wins +
                                                    planner.cpu_only +
                                                    planner.stable_ties},
          {"active_expert_planner_gpu_choices", planner.gpu_cost_wins +
                                                    planner.gpu_only +
                                                    planner.resident_gpu},
          {"active_expert_resolves_launched",
           worker.callable_resolves_launched},
          {"active_expert_resolves_completed",
           worker.callable_resolves_completed},
          {"active_expert_stale_plan_replans",
           worker.callable_stale_plan_replans},
          {"active_expert_prefetch_routes_launched",
           worker.callable_prefetch_routes_launched},
          {"active_expert_prefetch_routes_completed",
           worker.callable_prefetch_routes_completed},
          {"active_expert_prefetch_prediction_wait_ns",
           worker.callable_prefetch_prediction_wait_ns},
          {"active_expert_prefetch_candidates",
           worker.callable_prefetch_candidates},
          {"active_expert_prefetch_vram_candidates",
           worker.callable_prefetch_vram_candidates},
          {"active_expert_prefetch_ram_candidates",
           worker.callable_prefetch_ram_candidates},
          {"active_expert_prefetch_storage_candidates",
           worker.callable_prefetch_storage_candidates},
          {"active_expert_prefetch_storage_selected",
           worker.callable_prefetch_storage_selected},
          {"active_expert_prefetch_storage_incorrect",
           worker.callable_prefetch_storage_incorrect},
          {"active_expert_prefetch_resolves_launched",
           worker.callable_prefetch_resolves_launched},
          {"active_expert_prefetch_resolves_completed",
           worker.callable_prefetch_resolves_completed},
          {"active_expert_prefetch_selected",
           worker.callable_prefetch_selected},
          {"active_expert_prefetch_useful",
           worker.callable_prefetch_useful},
          {"active_expert_prefetch_late", worker.callable_prefetch_late},
          {"active_expert_prefetch_incorrect",
           worker.callable_prefetch_incorrect},
          {"active_expert_prefetch_cancelled",
           worker.callable_prefetch_cancelled},
          {"active_expert_prefetch_errors",
           worker.callable_prefetch_errors},
          {"active_expert_remote_resolves",
           worker.callable_remote_resolves},
          {"active_expert_remote_selections_launched",
           worker.callable_remote_selections_launched},
          {"active_expert_remote_selections_completed",
           worker.callable_remote_selections_completed},
          {"active_expert_remote_errors", worker.callable_remote_errors},
          {"active_expert_remote_cancellations",
           worker.callable_remote_cancellations},
          {"active_expert_remote_activation_tx_bytes",
           worker.callable_remote_activation_tx_bytes},
          {"active_expert_remote_activation_rx_bytes",
           worker.callable_remote_activation_rx_bytes},
          {"active_expert_remote_weight_tx_bytes",
           worker.callable_remote_weight_tx_bytes},
          {"active_expert_remote_owner_weight_read_bytes",
           worker.callable_remote_owner_weight_read_bytes},
          {"active_expert_remote_owner_storage_read_bytes",
           worker.callable_remote_owner_storage_read_bytes},
          {"active_expert_remote_owner_ram_read_bytes",
           worker.callable_remote_owner_ram_read_bytes},
          {"active_expert_remote_owner_vram_read_bytes",
           worker.callable_remote_owner_vram_read_bytes},
          {"active_expert_remote_owner_execution_ns",
           worker.callable_remote_owner_execution_ns},
          {"active_expert_remote_transport_wait_ns",
           worker.callable_remote_transport_wait_ns},
          {"warm_host_pages_loaded", worker.warm_start_loaded},
          {"warm_host_pages_candidates", worker.warm_start_candidates},
          {"warm_host_pages_failed", worker.warm_start_failed},
          {"warm_host_bytes", worker.warm_start_bytes},
          {"warm_vram_pages_candidates", worker.warm_vram_candidates},
          {"warm_vram_pages_loaded", worker.warm_vram_loaded},
          {"warm_vram_pages_failed", worker.warm_vram_failed},
          {"warm_vram_bytes", worker.warm_vram_bytes},
          {"sequence_blocks", worker.sequence_blocks},
          {"sequence_tile_rows", implementation->sequence_tile_rows()},
          {"sequence_rows", worker.sequence_rows},
          {"sequence_layers", worker.sequence_layers},
          {"sequence_attention_hca_pre_norm_ns",
           worker.sequence_attention_hca_pre_norm_ns},
          {"sequence_attention_projection_ns",
           worker.sequence_attention_projection_ns},
          {"sequence_causal_attention_ns",
           worker.sequence_causal_attention_ns},
          {"sequence_attention_output_projection_ns",
           worker.sequence_attention_output_projection_ns},
          {"sequence_route_ns", worker.sequence_route_ns},
          {"sequence_expert_wait_ns", worker.sequence_expert_wait_ns},
          {"sequence_expert_execute_ns",
           worker.sequence_expert_execute_ns},
          {"sequence_finalize_ns", worker.sequence_finalize_ns},
          {"census_namespace_rebinds",
           worker.census_namespace_rebinds}};
    };
    return {expert::runtime::Status::success(), std::move(module)};
  } catch (const std::exception& error) {
    return {{expert::runtime::ErrorCode::invalid_argument, error.what()}, {}};
  }
}

#ifndef EXPERT_VM_PROVIDER_LIBRARY
int main(int argc, char** argv) {
  return expert_vm_compressed_sparse_moe_provider_main(argc, argv);
}
#endif
