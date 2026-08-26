#include "expert/runtime/buffer_pool.hpp"
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
#include <cstring>
#include <filesystem>
#include <future>
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
#include <utility>
#include <vector>

namespace {

namespace er = expert::runtime;
namespace ec = expert::runtime::cuda;

constexpr std::string_view kHiddenAbi = "batch.hidden.f32.cuda.v1";
constexpr std::string_view kRouteIndexAbi =
    "batch.route-index.u32.cuda.v1";
constexpr std::string_view kRouteWeightAbi =
    "batch.route-weight.f32.cuda.v1";
constexpr std::string_view kTokenAbi = "batch.token-id.u32.host.v1";
constexpr std::string_view kPositionAbi = "batch.position.u32.host.v1";
constexpr std::string_view kMultimodalAbi =
    "request.multimodal.fp32.host.v1";
// Prefill reuses each decoded SM86 FP4 weight tile across four 128-row Tensor
// Core tiles, while scalar/speculative decode retains the lower-latency DP4A
// path. The provider publishes this geometry through its service contract;
// no model-family branch selects it.
constexpr std::uint32_t kWorkspaceRows = 512U;
constexpr std::uint32_t kMaximumVisionPatches = 4096U;
constexpr std::uint32_t kMaximumGpuSamplingTopK = 64U;
constexpr std::uint32_t kAttentionSplitTokens = 512U;
constexpr std::uint32_t kStagedPrefillSplitTokens = 8192U;

enum class TargetKvEncoding : std::uint8_t {
  artifact_fp4,
  fp8_e4m3_per_head,
  fp16
};

TargetKvEncoding target_kv_encoding(std::string_view value) {
  if (value == "artifact") return TargetKvEncoding::artifact_fp4;
  if (value == "fp8-e4m3-per-head")
    return TargetKvEncoding::fp8_e4m3_per_head;
  if (value == "fp16") return TargetKvEncoding::fp16;
  throw std::runtime_error("unsupported target KV cache dtype");
}

enum class Kernel : std::uint8_t {
  embedding,
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
      value.bytes / sizeof(std::uint32_t) > kWorkspaceRows ||
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

std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint32_t sample_sorted_candidates(
    std::span<const float> logits, std::span<const std::uint32_t> candidates,
    const er::ProgramRequestContext& request, std::uint32_t position) {
  const auto temperature_ppm =
      request_parameter(request, "sampling_temperature_ppm");
  const auto top_p_ppm = request_parameter(request, "sampling_top_p_ppm");
  const auto min_p_ppm = request_parameter(request, "sampling_min_p_ppm");
  const auto seed = request_parameter(request, "sampling_seed");
  if (logits.empty() || logits.size() != candidates.size() ||
      temperature_ppm == 0U ||
      temperature_ppm > 2'000'000U || top_p_ppm == 0U ||
      top_p_ppm > 1'000'000U || min_p_ppm > 1'000'000U ||
      !std::isfinite(logits.front()))
    throw std::runtime_error("invalid token sampling contract");

  const auto inverse_temperature =
      1'000'000.0 / static_cast<double>(temperature_ppm);
  const auto maximum = static_cast<double>(logits.front());
  const auto minimum_relative =
      static_cast<double>(min_p_ppm) / 1'000'000.0;
  std::vector<double> weights;
  weights.reserve(candidates.size());
  std::vector<std::uint32_t> retained_candidates;
  retained_candidates.reserve(candidates.size());
  std::size_t retained{};
  double total{};
  for (std::size_t index = 0U; index < candidates.size(); ++index) {
    const auto logit = static_cast<double>(logits[index]);
    const auto weight = std::isfinite(logit)
                            ? std::exp((logit - maximum) * inverse_temperature)
                            : 0.0;
    if (weight < minimum_relative) continue;
    retained_candidates.push_back(candidates[index]);
    ++retained;
    weights.push_back(weight);
    total += weight;
  }
  if (!retained || !std::isfinite(total) || !(total > 0.0))
    throw std::runtime_error("token sampling distribution is empty");

  const auto top_p = static_cast<double>(top_p_ppm) / 1'000'000.0;
  double cumulative{};
  std::size_t nucleus = weights.size();
  for (std::size_t index = 0U; index < weights.size(); ++index) {
    cumulative += weights[index] / total;
    if (cumulative >= top_p) {
      nucleus = index + 1U;
      break;
    }
  }
  weights.resize(nucleus);
  retained_candidates.resize(nucleus);
  total = std::accumulate(weights.begin(), weights.end(), 0.0);

  const auto random_bits = splitmix64(
      seed ^ (static_cast<std::uint64_t>(position) *
              0xd2b74407b1ce6e93ULL));
  const auto uniform = static_cast<double>(random_bits >> 11U) *
                       (1.0 / 9007199254740992.0);
  const auto threshold = uniform * total;
  cumulative = 0.0;
  for (std::size_t index = 0U; index < weights.size(); ++index) {
    cumulative += weights[index];
    if (threshold < cumulative) return retained_candidates[index];
  }
  return retained_candidates.back();
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
      "block.recurrent-linear-attention.split-gated-delta.v1");
  const auto mamba2 = has_capability("block.mamba2.ssm.v1");
  const auto standard_gqa =
      has_capability("block.full-attention.standard-gqa.v1") ||
      has_capability(
          "block.full-attention.standard-gqa-no-position.v1");
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
      query_heads / kv_heads >
          ((standard_gqa || normalized_gated_gqa)
               ? ec::kMaximumExactFp16GroupedQueryHeads
               : 8U) ||
      !head_dim || head_dim > 256U ||
      head_dim % 32U || !rotary || rotary > head_dim || rotary % 2U ||
      (model.exact_decode_program.has_value() && mtp != 1U) ||
      (!model.exact_decode_program.has_value() && mtp != 0U))
    return {er::ErrorCode::invalid_argument,
            "FP4 descriptor exceeds the SM86 provider geometry"};
  if (split_recurrent &&
      (!key_heads || !value_heads || value_heads % key_heads || !key_dim ||
       key_dim > 256U || !value_dim || value_dim > 256U || !conv ||
       conv > 16U || parameter("zero_centered_norm") != 1U))
    return {er::ErrorCode::invalid_argument,
            "split recurrent geometry exceeds the SM86 provider"};
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
    if (component.source_abi != er::kExpertSourceAbiExpertPackV1 ||
        component.encoding_abi != er::kExpertEncodingAbiFp4Block32 ||
        !((component.execution_capability ==
               "moe.swiglu.routed.merge-shared.v1" &&
           component.router.capability ==
               "router.linear-topk.shared-swiglu.v1") ||
          (component.execution_capability ==
               "moe.relu2.routed.merge-shared.v1" &&
           component.router.capability ==
               "router.sigmoid-bias.topk.shared-relu2.v1")) ||
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
      {"vision.patch-transformer-merge.fp4-block32.v1", 1U, 1U,
       validator},
      {"block.full-attention.output-gated.v1", 1U, 2U, validator},
      {"block.full-attention.standard-gqa.v1", 1U, 1U, validator},
      {"block.full-attention.standard-gqa-no-position.v1", 1U, 1U,
       validator},
      {"block.recurrent-linear-attention.split-gated-delta.v1", 1U, 1U,
       validator},
      {"block.mamba2.ssm.v1", 1U, 1U, validator},
      {"router.linear-topk.shared-swiglu.v1", 1U, 1U, validator},
      {"router.sigmoid-bias.topk.shared-relu2.v1", 1U, 1U, validator},
      {"moe.swiglu.routed.merge-shared.v1", 1U, 1U, validator},
      {"moe.relu2.routed.merge-shared.v1", 1U, 1U, validator},
      {"ffn.swiglu.dense.fp4-block32.v1", 1U, 2U, validator},
      {"head.rmsnorm.argmax.fp4-block32.v1", 1U, 1U, validator},
      {"head.rmsnorm.token-select.fp4-block32.v1", 1U, 2U, validator},
      {"decode.mtp.dense-full-attention.fp4-block32.exact.v1", 1U, 1U,
       validator},
  };
}

Kernel kernel_from_capability(std::string_view capability) {
  if (capability == "embedding.lookup.fp4-block32.v1")
    return Kernel::embedding;
  if (capability == "vision.patch-transformer-merge.fp4-block32.v1")
    return Kernel::vision;
  if (capability == "block.full-attention.output-gated.v1" ||
      capability == "block.full-attention.standard-gqa.v1" ||
      capability ==
          "block.full-attention.standard-gqa-no-position.v1")
    return Kernel::full_attention;
  if (capability ==
      "block.recurrent-linear-attention.split-gated-delta.v1")
    return Kernel::recurrent_attention;
  if (capability == "block.mamba2.ssm.v1")
    return Kernel::recurrent_attention;
  if (capability == "router.linear-topk.shared-swiglu.v1" ||
      capability == "router.sigmoid-bias.topk.shared-relu2.v1")
    return Kernel::router;
  if (capability == "moe.swiglu.routed.merge-shared.v1" ||
      capability == "moe.relu2.routed.merge-shared.v1")
    return Kernel::routed_moe;
  if (capability == "ffn.swiglu.dense.fp4-block32.v1")
    return Kernel::ffn;
  if (capability == "head.rmsnorm.argmax.fp4-block32.v1" ||
      capability == "head.rmsnorm.token-select.fp4-block32.v1")
    return Kernel::head;
  if (capability ==
      "decode.mtp.dense-full-attention.fp4-block32.exact.v1")
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
  const float* f32{};
  float* dequantized{};

  [[nodiscard]] ec::Fp4Block32Matrix matrix() const {
    if (encoding != "FP4_E2M1" || shape.size() != 2U || !fp4_data ||
        !fp4_scales)
      throw std::runtime_error(name + " is not a rank-2 FP4 matrix");
    return {fp4_data, fp4_scales, shape[0], shape[1], align32(shape[1])};
  }

  [[nodiscard]] ec::Fp4Block32Matrix flattened_matrix() const {
    if (encoding != "FP4_E2M1" || shape.size() < 2U || !fp4_data ||
        !fp4_scales)
      throw std::runtime_error(name + " is not an FP4 tensor matrix");
    const auto columns = checked_product(std::span(shape).subspan(1U));
    if (columns > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error(name + " flattened matrix is too wide");
    return {fp4_data, fp4_scales, shape[0],
            static_cast<std::uint32_t>(columns),
            align32(static_cast<std::uint32_t>(columns))};
  }
};

// Capability-owned sparse execution organ. It consumes only the routed
// component descriptor and authenticated artifact catalog; model-family and
// architecture identifiers never participate in placement or execution.
class Fp4RoutedExperts final {
 public:
  Fp4RoutedExperts(std::shared_ptr<er::ModelArtifact> artifact,
                   std::uint64_t ram_cache_bytes,
                   std::uint64_t vram_cache_bytes,
                   std::uint32_t maximum_rows)
      : artifact_(std::move(artifact)) {
    if (!artifact_ || artifact_->model().routed_components.size() != 1U ||
        !ram_cache_bytes || !vram_cache_bytes || !maximum_rows)
      throw std::runtime_error("invalid routed FP4 launch contract");
    component_ = artifact_->model().routed_components.front();
    relu2_ = component_.execution_capability ==
             "moe.relu2.routed.merge-shared.v1";
    if (component_.source_abi != er::kExpertSourceAbiExpertPackV1 ||
        component_.encoding_abi != er::kExpertEncodingAbiFp4Block32 ||
        (!relu2_ && component_.execution_capability !=
                       "moe.swiglu.routed.merge-shared.v1") ||
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
      }
    }
    if (!maximum_record_bytes ||
        maximum_record_bytes > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("routed FP4 record geometry is invalid");
    host_cache_capacity_bytes_ =
        std::min(ram_cache_bytes, total_record_bytes);
    whole_pool_host_ = host_cache_capacity_bytes_ == total_record_bytes;

    storage_ = std::make_shared<er::WindowsIocpStorage>(4U);
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
    return host_cache_capacity_bytes_;
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

  void execute(std::uint32_t layer, const float* input,
               const float* routing_weights,
               const std::uint32_t* routing_indices, std::uint32_t rows,
               float* intermediate, float* selection_output,
               std::int8_t* quantized_input, float* quantized_input_scales,
               std::int8_t* quantized_intermediate,
               float* quantized_intermediate_scales, float* output) {
    if (layer >= component_.layer_count || !input || !routing_weights ||
        !routing_indices || !rows || !intermediate || !selection_output ||
        !quantized_input || !quantized_input_scales ||
        !quantized_intermediate || !quantized_intermediate_scales || !output)
      throw std::runtime_error("invalid routed FP4 execution request");
    std::lock_guard execution_lock(execution_mutex_);
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
        bool selected = !relu2_ &&
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
                             quantized_intermediate_scales, rows);
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

 private:
  void launch_selection_graph(
      std::uint32_t layer, const float* input, const float* routing_weights,
      float* intermediate, float* selection_output,
      std::int8_t* quantized_input, float* quantized_input_scales,
      std::int8_t* quantized_intermediate,
      float* quantized_intermediate_scales, std::uint32_t rows) {
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
          directory_->device_entries(), layer, 0.0F, false, true});
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
           execution_stream_});
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
  bool host_bank_page_locked_{};
  bool relu2_{};
  std::uint64_t host_cache_capacity_bytes_{};
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
  std::map<std::string, std::uint64_t, std::less<>> parameters;
  std::map<std::string, const DeviceTensor*, std::less<>> tensors;
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
                   bool profile_gpu_phases)
      : artifact_(std::move(artifact)),
        descriptor_(artifact_ ? artifact_->model() : er::ModelDescriptor{}),
        tensor_store_(std::move(tensor_store)),
        max_context_(max_context), capacity_(capacity),
        ram_cache_bytes_(ram_cache_bytes), vram_cache_bytes_(vram_cache_bytes),
        kv_cache_bytes_(kv_cache_bytes), kv_page_tokens_(kv_page_tokens),
        target_kv_encoding_(target_kv_encoding(kv_cache_dtype)),
        profile_gpu_phases_(profile_gpu_phases),
        device_lifetime_(std::make_shared<std::uint8_t>(0U)) {
    status_check(validate_dense_fp4_descriptor(descriptor_));
    if (!artifact_ || !tensor_store_ || !tensor_store_->valid() || !capacity_ ||
        !max_context_ || max_context_ > descriptor_.max_context_tokens ||
        !kv_cache_bytes_ || !kv_page_tokens_)
      throw std::runtime_error("invalid dense FP4 provider launch contract");
    hidden_size_ = descriptor_.hidden_size;
    vocabulary_size_ = descriptor_.vocab_size;
    const auto has_capability = [&](std::string_view capability) {
      return std::any_of(
          descriptor_.operation_program.begin(),
          descriptor_.operation_program.end(),
          [capability](const auto& operation) {
            return operation.capability == capability;
          });
    };
    split_recurrent_enabled_ = has_capability(
        "block.recurrent-linear-attention.split-gated-delta.v1");
    mamba2_enabled_ = has_capability("block.mamba2.ssm.v1");
    standard_attention_enabled_ =
        has_capability("block.full-attention.standard-gqa.v1") ||
        has_capability(
            "block.full-attention.standard-gqa-no-position.v1");
    relu2_router_enabled_ = has_capability(
        "router.sigmoid-bias.topk.shared-relu2.v1");
    const auto zero_centered =
        descriptor_.attributes.find("zero_centered_norm");
    zero_centered_norm_ = zero_centered != descriptor_.attributes.end() &&
                          zero_centered->second == 1U;
    query_heads_ = attribute_u32("attention_heads");
    kv_heads_ = attribute_u32("kv_heads");
    head_dim_ = attribute_u32("head_dim");
    rotary_dimension_ = attribute_u32("rotary_dimension");
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
    if (!descriptor_.routed_components.empty()) {
      const auto& component = descriptor_.routed_components.front();
      expert_count_ = component.experts_per_layer;
      route_width_ = component.route_width;
      expert_width_ = component.intermediate_size;
      shared_intermediate_size_ = attribute_u32("shared_intermediate_size");
      routed_experts_ = std::make_unique<Fp4RoutedExperts>(
          artifact_, ram_cache_bytes_, vram_cache_bytes_, kWorkspaceRows);
      parking_ram_capacity_bytes_ =
          ram_cache_bytes_ - routed_experts_->host_cache_capacity_bytes();
    } else {
      parking_ram_capacity_bytes_ = ram_cache_bytes_;
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
    for (const auto& operation : descriptor_.operation_program) {
      if (operation.logical_layer == er::kModelLevelOperationLayer) continue;
      if (operation.logical_layer >= full_attention_slots_.size())
        throw std::runtime_error("operation layer exceeds the artifact topology");
      if (operation.capability == "block.full-attention.output-gated.v1" ||
          operation.capability == "block.full-attention.standard-gqa.v1" ||
          operation.capability ==
              "block.full-attention.standard-gqa-no-position.v1") {
        full_attention_slots_[operation.logical_layer] = target_full_layers_++;
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
               operation.capability == "block.mamba2.ssm.v1")
        recurrent_slots_[operation.logical_layer] = recurrent_layers_++;
    }
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
    const auto maximum_pages =
        (static_cast<std::uint64_t>(max_context_) + kv_page_tokens_ - 1U) /
        kv_page_tokens_;
    if (maximum_pages > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("maximum KV page count overflows");
    maximum_pages_per_slot_ = static_cast<std::uint32_t>(maximum_pages);
    auto fp16_target_page_bytes = static_cast<std::uint64_t>(
        global_attention_layers_);
    for (const auto factor : std::array<std::uint64_t, 5U>{
             2U, kv_page_tokens_, kv_heads_, head_dim_,
             sizeof(std::uint16_t)})
      fp16_target_page_bytes = checked_multiply(
          fp16_target_page_bytes, factor, "global FP16 KV page size");
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
        fp16_target_page_bytes, maximum_pages_per_slot_,
        "global FP16 KV context size");
    const auto window_reservation_bytes = checked_multiply(
        checked_multiply(2U, capacity_, "window FP16 KV reservation"),
        window_kv_bytes_per_slot_, "window FP16 KV reservation");
    const auto page_budget_bytes = window_reservation_bytes > kv_cache_bytes_
        ? 0U
        : kv_cache_bytes_ - window_reservation_bytes;
    device_resident_fp16_kv_ =
        exact_fp16_kv() && mtp_layers_ == 0U &&
        global_context_bytes <= page_budget_bytes;
    if (window_attention_layers_ != 0U && !device_resident_fp16_kv_)
      throw std::runtime_error(
          "windowed exact attention must fit its mixed device KV layout");
    if (target_kv_encoding_ == TargetKvEncoding::fp8_e4m3_per_head) {
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
          static_cast<std::uint64_t>(mtp_layers_) * fp4_layer_page_bytes;
      kv_page_bytes_ = target_kv_page_bytes_ + mtp_kv_page_bytes_;
    } else if (device_resident_fp16_kv()) {
      target_kv_page_bytes_ = fp16_target_page_bytes;
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ = 0U;
      kv_page_bytes_ = target_kv_page_bytes_;
    } else {
      const auto target_device_layers =
          host_authoritative_fp16_kv() ? 0U : target_full_layers_;
      target_kv_page_bytes_ = static_cast<std::uint64_t>(
          target_device_layers) * fp4_layer_page_bytes;
      mtp_kv_page_offset_ = target_kv_page_bytes_;
      mtp_kv_page_bytes_ =
          static_cast<std::uint64_t>(mtp_layers_) * fp4_layer_page_bytes;
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
    service_kv_page_bytes_ = host_authoritative_fp16_kv()
        ? fp16_target_page_bytes + mtp_kv_page_bytes_
        : kv_page_bytes_;
    service_kv_page_capacity_ = host_authoritative_fp16_kv()
        ? ram_cache_bytes_ / service_kv_page_bytes_
        : kv_page_capacity_;
    if (!service_kv_page_capacity_)
      throw std::runtime_error("KV RAM budget fits no service page");
    prepared_target_.resize(descriptor_.operation_program.size());
    slot_in_use_.assign(capacity_, false);
    slot_rope_deltas_.assign(capacity_, 0);
  }

  ~DenseFp4Provider() override {
    if (parking_stream_)
      static_cast<void>(cudaStreamDestroy(parking_stream_));
    for (auto& event : gpu_event_pool_) {
      if (event.start) static_cast<void>(cudaEventDestroy(event.start));
      if (event.stop) static_cast<void>(cudaEventDestroy(event.stop));
    }
    for (auto* page : all_kv_pages_)
      if (page) static_cast<void>(cudaFree(page));
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
      }
      for (const auto& binding : context.tensors) {
        if (!binding.tensor)
          throw std::runtime_error("operation tensor binding is null");
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
      if (context.model.content_hash != descriptor_.content_hash ||
          context.program.capability !=
              "decode.mtp.dense-full-attention.fp4-block32.exact.v1" ||
          context.program.abi_version != 1U ||
          context.compiled.maximum_emitted_tokens != 2U ||
          parameter_u32(context.compiled.parameters, "draft_layers") != 1U ||
          parameter_u32(context.compiled.parameters, "embedding_first") != 1U ||
          parameter_u32(context.compiled.parameters, "post_norm") != 1U)
        throw std::runtime_error("unsupported exact-decode artifact contract");
      auto prepared = std::make_shared<PreparedOperation>();
      prepared->kernel = Kernel::exact_decode;
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
  [[nodiscard]] bool supports_request_state_parking()
      const noexcept override {
    return true;
  }
  er::RequestStateParkingResult park_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& state,
      std::uint32_t next_position) override;
  er::RequestStateParkingResult restore_request_state(
      const std::shared_ptr<er::IOperationProviderRequestState>& state)
      override;
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
    if (exact_fp16_kv()) return "fp16";
    return target_kv_encoding_ == TargetKvEncoding::fp8_e4m3_per_head
               ? "fp8-e4m3-per-head"
               : "fp4-e2m1-ue8m0-block32";
  }
  [[nodiscard]] std::map<std::string, std::uint64_t, std::less<>> telemetry()
      const {
    std::uint64_t resident_pages{};
    for (const auto& slot : slot_pages_)
      resident_pages += static_cast<std::uint64_t>(std::count_if(
          slot.begin(), slot.end(), [](const void* page) { return page; }));
    auto result = std::map<std::string, std::uint64_t, std::less<>>{
            {"resident_tensor_bytes", uploaded_tensor_bytes_},
            {"provider_program_steps", program_steps_},
            {"provider_prefill_batches", prefill_batches_},
            {"provider_prefill_tokens", prefill_tokens_},
            {"provider_exact_sync_batches", exact_sync_batches_},
            {"provider_exact_sync_tokens", exact_sync_tokens_},
            {"provider_exact_calls", exact_calls_},
            {"provider_accepted_drafts", accepted_drafts_},
            {"provider_program_sequence_batches", program_sequence_batches_},
            {"provider_program_sequence_tokens", program_sequence_tokens_},
            {"provider_program_sequence_tiles", program_sequence_tiles_},
            {"provider_program_sequence_device_bytes",
             program_sequence_device_bytes_},
            {"provider_program_sequence_host_bytes",
             program_sequence_host_bytes_},
            {"provider_program_sequence_tile_rows", sequence_tile_rows_},
            {"provider_sampling_gpu_calls", sampling_gpu_calls_},
            {"provider_sampling_host_calls", sampling_host_calls_},
            {"provider_sampling_logit_transfer_bytes",
             sampling_logit_transfer_bytes_},
            {"provider_staged_dense_weight_bytes",
             staged_dense_weight_capacity_bytes_},
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
    std::uint32_t retention_position{};
    std::int32_t rope_delta{};
    std::vector<std::uint32_t> prompt_mrope_positions;
    std::vector<float> sequence_target_hidden;
    bool draft_valid{};
    bool retention_valid{};
    bool exact_decode_enabled{};
    std::vector<HostKvPage> host_kv_pages;
    std::uint32_t host_kv_populated_tokens{};
    std::uint64_t host_kv_bytes{};
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
    tensor->allocation_bytes = source.data_bytes + source.scale_bytes;
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
      cuda_check(cudaMemcpy(tensor->allocation + source.data_bytes,
                            host + source.scale_offset,
                            static_cast<std::size_t>(source.scale_bytes),
                            cudaMemcpyHostToDevice),
                 "upload dense tensor scales");
    if (source.encoding == "FP4_E2M1") {
      if (source.quant_abi != er::kExpertQuantAbiFp4Block32 ||
          source.shape.empty())
        throw std::runtime_error("FP4 tensor has the wrong quantization ABI");
      const auto rows = checked_product(std::span(source.shape).first(
          source.shape.size() - 1U));
      const auto padded = align32(source.shape.back());
      const auto expected_data = rows * padded / 2U;
      const auto expected_scales = rows * padded / 32U;
      if (source.data_bytes != expected_data ||
          source.scale_bytes != expected_scales)
        throw std::runtime_error("FP4 tensor storage geometry is inconsistent");
      tensor->fp4_data =
          reinterpret_cast<const std::uint8_t*>(tensor->allocation);
      tensor->fp4_scales = reinterpret_cast<const std::uint8_t*>(
          tensor->allocation + source.data_bytes);
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
    } else if (source.encoding == "F32") {
      if (source.quant_abi != 0U || source.scale_bytes != 0U ||
          source.data_bytes != checked_product(source.shape) * sizeof(float))
        throw std::runtime_error("F32 tensor storage geometry is inconsistent");
      tensor->f32 = reinterpret_cast<const float*>(tensor->allocation);
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

  const DeviceTensor& binding(const PreparedOperation& operation,
                              std::string_view role) const {
    const auto found = operation.tensors.find(role);
    if (found == operation.tensors.end() || found->second == nullptr)
      throw std::runtime_error("missing tensor role " + std::string(role));
    return *found->second;
  }

  static void expect_shape(const DeviceTensor& tensor,
                           std::initializer_list<std::uint32_t> shape,
                           std::string_view encoding) {
    if (tensor.encoding != encoding ||
        tensor.shape != std::vector<std::uint32_t>(shape))
      throw std::runtime_error("tensor role has an incompatible shape/encoding");
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
  void stage_operation_weights(const PreparedOperation& operation);
  void activate_staged_weights(const PreparedOperation& operation);
  void deactivate_staged_weights() noexcept;
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
  void quantize_rows(const float* input, std::uint32_t rows,
                     std::uint32_t columns);
  void project_quantized(const DeviceTensor& weight, float* output,
                         std::uint32_t rows);
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
  void trim_host_kv(RequestState& state, std::uint32_t populated_tokens)
      noexcept;
  void release_host_kv(RequestState& state) noexcept;
  [[nodiscard]] std::pair<std::uint16_t*, std::uint16_t*> host_kv_layer_page(
      const RequestState& state, std::uint32_t full_attention_slot,
      std::uint32_t page_index) const;
  void run_full_attention(const PreparedOperation& operation,
                          RequestState& state,
                          std::span<const std::uint32_t> cache_positions,
                          std::span<const std::uint32_t> rotary_positions,
                          std::uint32_t rows,
                          std::uint32_t full_attention_slot,
                          std::span<const std::uint32_t> mrope_positions = {});
  void run_recurrent_attention(const PreparedOperation& operation,
                               std::uint32_t slot, std::uint32_t rows,
                               bool checkpoint_after_first);
  void run_router(const PreparedOperation& operation, std::uint32_t rows);
  void run_routed_moe(const PreparedOperation& operation,
                      std::uint32_t rows);
  void run_ffn(const PreparedOperation& operation, std::uint32_t rows);
  std::vector<std::uint32_t> run_head(const PreparedOperation& operation,
                                      std::uint32_t rows,
                                      const er::ProgramRequestContext* request,
                                      std::uint32_t sample_position,
                                      bool terminal_only = false);
  std::vector<std::uint32_t> run_target(
      RequestState& state, std::span<const std::uint32_t> tokens,
      std::span<const std::uint32_t> positions, bool transactional_second);
  std::vector<std::uint32_t> run_mtp(
      RequestState& state, std::span<const std::uint32_t> tokens,
      const float* previous_hidden,
      std::span<const std::uint32_t> rotary_positions, bool produce_logits);
  void restore_recurrent_checkpoint(std::uint32_t slot);
  void checkpoint_recurrent_state(std::uint32_t slot);
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
  TargetKvEncoding target_kv_encoding_{TargetKvEncoding::artifact_fp4};
  bool device_resident_fp16_kv_{};
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
  bool relu2_router_enabled_{};
  bool zero_centered_norm_{};
  std::uint32_t mtp_layers_{};
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
  std::uint64_t window_kv_page_bytes_{};
  std::uint64_t window_kv_bytes_per_slot_{};
  std::uint64_t mtp_kv_page_offset_{};
  std::uint64_t mtp_kv_page_bytes_{};
  std::uint64_t service_kv_page_bytes_{};
  std::uint32_t maximum_pages_per_slot_{};
  std::uint64_t kv_page_capacity_{};
  std::uint64_t service_kv_page_capacity_{};
  std::uint64_t host_kv_bytes_{};
  std::uint64_t parked_request_bytes_{};
  std::uint64_t parked_session_bytes_{};
  std::uint64_t park_calls_{};
  std::uint64_t restore_calls_{};
  std::uint64_t park_device_to_host_bytes_{};
  std::uint64_t restore_host_to_device_bytes_{};
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
  std::uint64_t accepted_drafts_{};
  std::uint64_t program_sequence_batches_{};
  std::uint64_t program_sequence_tokens_{};
  std::uint64_t program_sequence_tiles_{};
  std::uint64_t program_sequence_device_bytes_{};
  std::uint64_t program_sequence_host_bytes_{};
  std::uint64_t sampling_gpu_calls_{};
  std::uint64_t sampling_host_calls_{};
  std::uint64_t sampling_logit_transfer_bytes_{};
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
  std::vector<std::shared_ptr<PreparedOperation>> prepared_target_;
  std::shared_ptr<PreparedOperation> exact_;
  std::shared_ptr<PreparedOperation> mtp_attention_;
  std::shared_ptr<PreparedOperation> mtp_ffn_;
  std::unique_ptr<Fp4RoutedExperts> routed_experts_;
  std::map<std::string, std::unique_ptr<DeviceTensor>, std::less<>> tensors_;
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

  // Workspace and state are declared below with the execution methods.
  float* hidden_{};
  float* sequence_hidden_{};
  std::uint32_t sequence_tile_rows_{};
  float* normalized_{};
  float* residual_{};
  float* query_gate_{};
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
  std::int8_t* q8_{};
  float* q8_scales_{};
  std::uint16_t* staged_dense_weights_{};
  std::uint16_t* staged_dense_input_{};
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
  std::vector<float*> recurrent_conv_retention_checkpoint_;
  std::vector<float*> recurrent_matrix_retention_checkpoint_;
  float* slot_retention_last_hidden_{};
  void** device_page_table_{};
  void** device_mtp_page_table_{};
  void** device_window_page_table_{};
  std::byte* window_kv_{};
  std::byte* window_retention_kv_{};
  std::vector<std::vector<void*>> slot_pages_;
  std::vector<void*> free_kv_pages_;
  std::vector<void*> all_kv_pages_;
  cudaStream_t parking_stream_{};
};

void DenseFp4Provider::validate_operation(PreparedOperation& operation) {
  switch (operation.kernel) {
    case Kernel::embedding:
      expect_shape(binding(operation, "weight"),
                   {vocabulary_size_, hidden_size_}, "FP4_E2M1");
      if (operation.abi_version >= 2U) {
        if (parameter_u32(operation.parameters, "output_norm_mode") != 1U)
          throw std::runtime_error("unsupported embedding ABI 2 norm mode");
        static_cast<void>(parameter_f32(
            operation.parameters, "output_norm_epsilon_f32_bits"));
      }
      break;
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
      expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
      const auto separated_gate = operation.abi_version >= 2U;
      expect_shape(
          binding(operation, "query_projection"),
          {((operation.capability == "block.full-attention.standard-gqa.v1" ||
             operation.capability ==
                 "block.full-attention.standard-gqa-no-position.v1" ||
             separated_gate)
                ? query_heads_
                : 2U * query_heads_) * head_dim_,
           hidden_size_},
          "FP4_E2M1");
      expect_shape(binding(operation, "key_projection"),
                   {kv_heads_ * head_dim_, hidden_size_}, "FP4_E2M1");
      expect_shape(binding(operation, "value_projection"),
                   {kv_heads_ * head_dim_, hidden_size_}, "FP4_E2M1");
      expect_shape(binding(operation, "output_projection"),
                   {hidden_size_, query_heads_ * head_dim_}, "FP4_E2M1");
      if (separated_gate) {
        expect_shape(binding(operation, "gate_projection"),
                     {query_heads_ * head_dim_, hidden_size_}, "FP4_E2M1");
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
        expect_shape(binding(operation, "input_projection"),
                     {projection, hidden_size_}, "FP4_E2M1");
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
        expect_shape(binding(operation, "output_projection"),
                     {hidden_size_, intermediate}, "FP4_E2M1");
        static_cast<void>(parameter_f32(operation.parameters,
                                        "time_step_min_f32_bits"));
        break;
      }
      const auto key_dimension = key_heads_ * key_head_dim_;
      const auto value_dimension = value_heads_ * value_head_dim_;
      const auto conv_dimension = 2U * key_dimension + value_dimension;
      expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
      expect_shape(binding(operation, "qkv_projection"),
                   {conv_dimension, hidden_size_}, "FP4_E2M1");
      expect_shape(binding(operation, "z_projection"),
                   {value_dimension, hidden_size_}, "FP4_E2M1");
      expect_shape(binding(operation, "b_projection"),
                   {value_heads_, hidden_size_}, "FP4_E2M1");
      expect_shape(binding(operation, "a_projection"),
                   {value_heads_, hidden_size_}, "FP4_E2M1");
      auto& convolution = const_cast<DeviceTensor&>(
          binding(operation, "convolution"));
      expect_shape(convolution, {conv_dimension, 1U, conv_kernel_},
                   "FP4_E2M1");
      if (!convolution.dequantized)
        throw std::runtime_error("recurrent convolution was not dequantized");
      expect_shape(binding(operation, "time_bias"), {value_heads_}, "F32");
      expect_shape(binding(operation, "decay_log"), {value_heads_}, "F32");
      expect_shape(binding(operation, "output_norm"), {value_head_dim_},
                   "F32");
      expect_shape(binding(operation, "output_projection"),
                   {hidden_size_, value_dimension}, "FP4_E2M1");
      break;
    }
    case Kernel::router:
      if (!routed_experts_ ||
          operation.logical_layer == er::kModelLevelOperationLayer ||
          operation.component_layer >=
              routed_experts_->component().layer_count)
        throw std::runtime_error("router has no routed component layer");
      expect_shape(binding(operation, "input_norm"), {hidden_size_}, "F32");
      expect_shape(binding(operation, "router_weight"),
                   {expert_count_, hidden_size_}, "F32");
      if (operation.capability ==
          "router.sigmoid-bias.topk.shared-relu2.v1") {
        expect_shape(binding(operation, "correction_bias"), {expert_count_},
                     "F32");
        expect_shape(binding(operation, "shared_up_projection"),
                     {shared_intermediate_size_, hidden_size_}, "FP4_E2M1");
        expect_shape(binding(operation, "shared_down_projection"),
                     {hidden_size_, shared_intermediate_size_}, "FP4_E2M1");
        static_cast<void>(parameter_f32(operation.parameters,
                                        "scale_f32_bits"));
      } else {
        expect_shape(binding(operation, "shared_gate_projection"),
                     {shared_intermediate_size_, hidden_size_}, "FP4_E2M1");
        expect_shape(binding(operation, "shared_up_projection"),
                     {shared_intermediate_size_, hidden_size_}, "FP4_E2M1");
        expect_shape(binding(operation, "shared_down_projection"),
                     {hidden_size_, shared_intermediate_size_}, "FP4_E2M1");
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
      if (gate.encoding != "FP4_E2M1" || gate.shape.size() != 2U ||
          gate.shape[1] != hidden_size_ || up.encoding != "FP4_E2M1" ||
          up.shape != gate.shape)
        throw std::runtime_error("dense FFN gate/up geometry is invalid");
      if (!intermediate_size_)
        intermediate_size_ = gate.shape[0];
      else if (intermediate_size_ != gate.shape[0])
        throw std::runtime_error("dense FFN width changes between layers");
      expect_shape(binding(operation, "down_projection"),
                   {hidden_size_, intermediate_size_}, "FP4_E2M1");
      break;
    }
    case Kernel::head:
      expect_shape(binding(operation, "norm"), {hidden_size_}, "F32");
      expect_shape(binding(operation, "weight"),
                   {vocabulary_size_, hidden_size_}, "FP4_E2M1");
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
  expect_shape(binding(operation, "token_embedding"),
               {vocabulary_size_, hidden_size_}, "FP4_E2M1");
  expect_shape(binding(operation, "output_head"),
               {vocabulary_size_, hidden_size_}, "FP4_E2M1");
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
    std::size_t operation_bytes{};
    for (const auto& [role, tensor] : operation->tensors) {
      static_cast<void>(role);
      if (!tensor || tensor->encoding != "FP4_E2M1" ||
          tensor->shape.size() != 2U || !unique.insert(tensor).second)
        continue;
      const auto matrix = tensor->matrix();
      const auto values =
          static_cast<std::size_t>(matrix.rows) * matrix.padded_columns;
      if (values > std::numeric_limits<std::size_t>::max() /
                       sizeof(std::uint16_t) ||
          operation_bytes >
              std::numeric_limits<std::size_t>::max() -
                  values * sizeof(std::uint16_t))
        throw std::runtime_error("staged dense weight workspace overflows");
      operation_bytes += values * sizeof(std::uint16_t);
    }
    staged_dense_weight_capacity_bytes_ =
        std::max(staged_dense_weight_capacity_bytes_, operation_bytes);
  }
  const auto maximum_columns = align32(std::max(
      {2U * hidden_size_, hidden_size_, intermediate_size_,
       shared_intermediate_size_, expert_width_,
       query_heads_ * head_dim_, value_heads_ * value_head_dim_,
       mamba_heads_ * mamba_head_dim_ + mamba_conv_size_ + mamba_heads_,
       vision_patch_dimension_, vision_hidden_size_,
       vision_intermediate_size_, vision_merged_width_}));
  staged_dense_input_capacity_bytes_ =
      static_cast<std::size_t>(kWorkspaceRows) * maximum_columns *
      sizeof(std::uint16_t);
  allocate_workspace();
  allocate_state();
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
       mamba_intermediate});
  const auto maximum_columns = std::max(
      {2U * hidden_size_, hidden_size_, workspace_intermediate,
       attention_width,
       value_dimension, mamba_projection, vision_patch_dimension_,
       vision_hidden_size_,
       vision_intermediate_size_, vision_merged_width_});
  const auto padded_columns = align32(maximum_columns);
  hidden_ = device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
  const auto hidden_row_bytes =
      static_cast<std::uint64_t>(hidden_size_) * sizeof(float);
  const auto minimum_sequence_bytes =
      static_cast<std::uint64_t>(capacity_) * kWorkspaceRows *
      hidden_row_bytes;
  const auto desired_sequence_bytes = std::max<std::uint64_t>(
      minimum_sequence_bytes,
      std::min<std::uint64_t>(128ULL << 20U, vram_cache_bytes_ / 64U));
  auto rows_per_slot = desired_sequence_bytes /
      (static_cast<std::uint64_t>(capacity_) * hidden_row_bytes);
  rows_per_slot = std::max<std::uint64_t>(kWorkspaceRows, rows_per_slot);
  rows_per_slot -= rows_per_slot % kWorkspaceRows;
  sequence_tile_rows_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(
      rows_per_slot, std::numeric_limits<std::uint32_t>::max()));
  sequence_hidden_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(capacity_) *
                        sequence_tile_rows_ * hidden_size_);
  normalized_ =
      device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
  residual_ = device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
  query_gate_ =
      device_allocate<float>(allocations_, kWorkspaceRows * query_width);
  key_ = device_allocate<float>(allocations_, kWorkspaceRows * key_value_width);
  value_ = device_allocate<float>(allocations_, kWorkspaceRows * key_value_width);
  attention_ =
      device_allocate<float>(allocations_, kWorkspaceRows * attention_width);
  projected_qkv_ =
      device_allocate<float>(allocations_, kWorkspaceRows * recurrent_projection);
  projected_z_ =
      device_allocate<float>(allocations_,
                             kWorkspaceRows * std::max(1U, value_dimension));
  projected_b_ =
      device_allocate<float>(allocations_,
                             kWorkspaceRows * std::max(1U, value_heads_));
  projected_a_ =
      device_allocate<float>(allocations_,
                             kWorkspaceRows * std::max(1U, value_heads_));
  conv_output_ =
      device_allocate<float>(allocations_,
                             kWorkspaceRows * recurrent_conv_output);
  delta_output_ =
      device_allocate<float>(allocations_, kWorkspaceRows * recurrent_output);
  gate_ =
      device_allocate<float>(allocations_, kWorkspaceRows * workspace_intermediate);
  up_ = device_allocate<float>(allocations_,
                               kWorkspaceRows * workspace_intermediate);
  intermediate_ =
      device_allocate<float>(allocations_,
                             kWorkspaceRows * workspace_intermediate);
  if (routed_experts_) {
    shared_output_ =
        device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
    shared_scalar_ = device_allocate<float>(allocations_, kWorkspaceRows);
    router_logits_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kWorkspaceRows) * expert_count_);
    routing_scores_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kWorkspaceRows) * route_width_);
    routing_indices_ = device_allocate<std::uint32_t>(
        allocations_, static_cast<std::size_t>(kWorkspaceRows) * route_width_);
    moe_intermediate_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kWorkspaceRows) * route_width_ *
                          expert_width_);
    moe_selection_output_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kWorkspaceRows) * route_width_ *
                          hidden_size_);
    moe_output_ =
        device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
    moe_q8_intermediate_ = device_allocate<std::int8_t>(
        allocations_, static_cast<std::size_t>(kWorkspaceRows) * route_width_ *
                          expert_width_);
    moe_q8_intermediate_scales_ = device_allocate<float>(
        allocations_, static_cast<std::size_t>(kWorkspaceRows) * route_width_);
  }
  logits_ =
      device_allocate<float>(allocations_, kWorkspaceRows * vocabulary_size_);
  output_tokens_ =
      device_allocate<std::uint32_t>(allocations_, kWorkspaceRows);
  sampling_top_logits_ = device_allocate<float>(
      allocations_, kMaximumGpuSamplingTopK);
  sampling_top_tokens_ = device_allocate<std::uint32_t>(
      allocations_, kMaximumGpuSamplingTopK);
  q8_ = device_allocate<std::int8_t>(allocations_,
                                     kWorkspaceRows * padded_columns);
  q8_scales_ = device_allocate<float>(allocations_, kWorkspaceRows);
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
  if (staged_dense_weight_capacity_bytes_ != 0U)
    staged_dense_weights_ = device_allocate<std::uint16_t>(
        allocations_, staged_dense_weight_capacity_bytes_ /
                          sizeof(std::uint16_t));
  staged_dense_input_ = device_allocate<std::uint16_t>(
      allocations_, staged_dense_input_capacity_bytes_ /
                        sizeof(std::uint16_t));
  attention_maximum_splits_ =
      (max_context_ + kAttentionSplitTokens - 1U) / kAttentionSplitTokens;
  const auto attention_state_rows =
      std::max(attention_maximum_splits_, kWorkspaceRows);
  partial_maxima_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(attention_state_rows) *
                        query_heads_);
  partial_sums_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(attention_state_rows) *
                        query_heads_);
  partial_outputs_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(attention_maximum_splits_) *
                        query_heads_ * head_dim_);
  staged_split_tokens_ = std::min(kStagedPrefillSplitTokens, max_context_);
  const auto staged_query_values =
      static_cast<std::size_t>(kWorkspaceRows) * query_heads_ * head_dim_;
  const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                staged_split_tokens_ * head_dim_;
  const auto staged_score_values =
      static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
      staged_split_tokens_;
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
      device_allocate<float>(allocations_, staged_score_values);
  staged_probabilities_ =
      device_allocate<std::uint16_t>(allocations_, staged_score_values);
  staged_accumulator_ =
      device_allocate<float>(allocations_, staged_query_values);
  mtp_embedding_ =
      device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
  mtp_embedding_norm_ =
      device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
  mtp_hidden_norm_ =
      device_allocate<float>(allocations_, kWorkspaceRows * hidden_size_);
  mtp_fusion_input_ =
      device_allocate<float>(allocations_, kWorkspaceRows * 2U * hidden_size_);
  slot_target_hidden_batch_ = device_allocate<float>(
      allocations_, static_cast<std::size_t>(capacity_) * kWorkspaceRows *
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
    recurrent_conv_retention_checkpoint_[layer] =
        device_allocate<float>(allocations_, capacity_ * conv_values);
    recurrent_matrix_retention_checkpoint_[layer] =
        device_allocate<float>(allocations_, capacity_ * matrix_values);
  }
  device_page_table_ = device_allocate<void*>(
      allocations_, static_cast<std::size_t>(capacity_) *
                        maximum_pages_per_slot_);
  cuda_check(cudaMemset(device_page_table_, 0,
                        static_cast<std::size_t>(capacity_) *
                            maximum_pages_per_slot_ * sizeof(void*)),
             "zero FP4 KV page table");
  if (target_kv_encoding_ == TargetKvEncoding::fp8_e4m3_per_head &&
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
    const auto free = std::find(slot_in_use_.begin(), slot_in_use_.end(), false);
    if (free == slot_in_use_.end())
      return {{er::ErrorCode::backpressure,
               "dense FP4 provider has no free request slot"},
              {}};
    const auto slot = static_cast<std::uint32_t>(free - slot_in_use_.begin());
    *free = true;
    const auto conv_bytes = recurrent_conv_values_ * sizeof(float);
    const auto matrix_bytes = recurrent_matrix_values_ * sizeof(float);
    for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
      cuda_check(cudaMemset(recurrent_conv(layer, slot), 0, conv_bytes),
                 "zero recurrent convolution state");
      cuda_check(cudaMemset(recurrent_matrix(layer, slot), 0, matrix_bytes),
                 "zero recurrent matrix state");
    }
    return {er::Status::success(),
            std::make_shared<RequestState>(*this, slot)};
  } catch (const std::exception& error) {
    return {{er::ErrorCode::internal, error.what()}, {}};
  }
}

er::ExecutionValue DenseFp4Provider::device_hidden_value(
    std::uint32_t rows) const {
  if (!rows || rows > kWorkspaceRows)
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
      value.bytes / row_bytes > kWorkspaceRows)
    throw std::runtime_error("hidden-state execution ABI mismatch");
  return static_cast<std::uint32_t>(value.bytes / row_bytes);
}

GpuPhase DenseFp4Provider::gpu_phase(Kernel kernel) {
  switch (kernel) {
    case Kernel::embedding:
      return GpuPhase::embedding;
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
    normalize_rows(input, binding(operation, "input_norm").f32, output, rows);
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
    status_check(ec::add_in_place(hidden_, residual_, values, nullptr));
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
  status_check(ec::add_in_place(hidden_, normalized_, values, nullptr));
}

void DenseFp4Provider::run_embedding(
    const PreparedOperation& operation, const std::uint32_t* tokens,
    std::uint32_t rows) {
  if (!tokens || !rows || rows > kWorkspaceRows)
    throw std::runtime_error("invalid embedding batch");
  cuda_check(cudaMemcpy(output_tokens_, tokens,
                        static_cast<std::size_t>(rows) * sizeof(tokens[0]),
                        cudaMemcpyHostToDevice),
             "upload embedding token batch");
  status_check(ec::fp4_embedding_batch(
      binding(operation, "weight").matrix(), output_tokens_, hidden_, rows,
      nullptr));
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

void DenseFp4Provider::quantize_rows(const float* input, std::uint32_t rows,
                                     std::uint32_t columns) {
  status_check(ec::quantize_q8_batch(input, q8_, q8_scales_, rows, columns,
                                     align32(columns), nullptr));
}

void DenseFp4Provider::stage_operation_weights(
    const PreparedOperation& operation) {
  if (staged_operation_ == &operation) return;
  staged_weight_bindings_.clear();
  std::set<const DeviceTensor*> unique;
  auto* cursor = reinterpret_cast<std::byte*>(staged_dense_weights_);
  std::size_t used{};
  for (const auto& [role, tensor] : operation.tensors) {
    static_cast<void>(role);
    if (!tensor || tensor->encoding != "FP4_E2M1" ||
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
}

void DenseFp4Provider::activate_staged_weights(
    const PreparedOperation& operation) {
  stage_operation_weights(operation);
  active_staged_weights_ = staged_weight_bindings_;
}

void DenseFp4Provider::deactivate_staged_weights() noexcept {
  active_staged_weights_.clear();
}

void DenseFp4Provider::project_quantized(const DeviceTensor& weight,
                                         float* output,
                                         std::uint32_t rows) {
  const auto matrix = weight.matrix();
  const auto staged = active_staged_weights_.find(&weight);
  if (rows > 8U && staged != active_staged_weights_.end())
    status_check(ec::bf16_gemm_q8_block32(
        matrix, staged->second.data, staged->second.bytes, q8_, q8_scales_,
        staged_dense_input_, staged_dense_input_capacity_bytes_, output, rows,
        nullptr));
  else if (rows > 8U)
    status_check(ec::fp4_gemm_q8_block32(
        matrix, q8_, q8_scales_, output, rows, nullptr));
  else if (rows > 1U)
    status_check(ec::fp4_gemv_q8_batch_weight_reuse(
        matrix, q8_, q8_scales_, output, rows, nullptr));
  else
    status_check(ec::fp4_gemv_q8_batch(matrix, q8_, q8_scales_, output, rows,
                                       nullptr));
}

void DenseFp4Provider::project(const DeviceTensor& weight, const float* input,
                               float* output, std::uint32_t rows) {
  const auto matrix = weight.matrix();
  quantize_rows(input, rows, matrix.columns);
  project_quantized(weight, output, rows);
}

void DenseFp4Provider::project_vision(const DeviceTensor& weight,
                                      const float* input, float* output,
                                      std::uint32_t rows) {
  if (!rows || rows > kMaximumVisionPatches)
    throw std::runtime_error("vision projection row count is invalid");
  const auto matrix = weight.matrix();
  for (std::uint32_t first = 0U; first < rows; first += kWorkspaceRows) {
    const auto chunk = std::min(kWorkspaceRows, rows - first);
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
       first += kWorkspaceRows) {
    const auto rows = std::min(kWorkspaceRows, media.patch_count - first);
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
        media.patch_count * vision_hidden_size_, nullptr));

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
        media.patch_count * vision_hidden_size_, nullptr));
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
    const auto page_values = static_cast<std::uint64_t>(kv_page_tokens_) *
                             kv_heads_ * head_dim_;
    const auto page_bytes = static_cast<std::uint64_t>(target_full_layers_) *
                            2U * page_values * sizeof(std::uint16_t);
    if (!page_bytes ||
        page_bytes > std::numeric_limits<std::size_t>::max() ||
        page_bytes > ram_cache_bytes_ ||
        host_kv_bytes_ + parked_request_bytes_ >
            ram_cache_bytes_ - page_bytes)
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
  const auto page_bytes = static_cast<std::uint64_t>(target_full_layers_) *
                          2U * kv_page_tokens_ * kv_heads_ * head_dim_ *
                          sizeof(std::uint16_t);
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

void DenseFp4Provider::run_full_attention(
    const PreparedOperation& operation, RequestState& state,
    std::span<const std::uint32_t> cache_positions,
    std::span<const std::uint32_t> rotary_positions, std::uint32_t rows,
    std::uint32_t full_attention_slot,
    std::span<const std::uint32_t> mrope_positions) {
  const auto slot = state.slot();
  if (!rows || rows > kWorkspaceRows || cache_positions.size() != rows ||
      rotary_positions.size() != rows ||
      (!mrope_positions.empty() && mrope_positions.size() != 3U * rows))
    throw std::runtime_error("invalid full-attention microbatch");
  for (std::uint32_t row = 1U; row < rows; ++row)
    if (cache_positions[row] != cache_positions.front() + row ||
        (mrope_positions.empty() &&
         rotary_positions[row] != rotary_positions.front() + row))
      throw std::runtime_error(
          "full-attention microbatch positions are not contiguous");
  quantize_rows(normalized_, rows, hidden_size_);
  project_quantized(binding(operation, "query_projection"), query_gate_, rows);
  project_quantized(binding(operation, "key_projection"), key_, rows);
  project_quantized(binding(operation, "value_projection"), value_, rows);
  if (operation.abi_version >= 2U) {
    if (!device_resident_fp16_kv() || !mrope_positions.empty() ||
        operation.kv_layer_slot == kNoSlot ||
        cache_positions.back() >= max_context_)
      throw std::runtime_error(
          "normalized output-gated GQA requires device FP16 KV");
    project_quantized(binding(operation, "gate_projection"), gate_, rows);
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

    const auto query_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * staged_split_tokens_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_, score_values * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(kWorkspaceRows) *
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
        attention_, gate_, rows * query_heads_ * head_dim_, nullptr));
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

    const auto query_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * staged_split_tokens_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_, score_values * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(kWorkspaceRows) *
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
        rope_theta_, nullptr));
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
    const auto continues_device_layer =
        (cache_positions.front() == 0U) ||
        (staged_device_slot_ == slot &&
         staged_device_layer_ == full_attention_slot &&
         staged_device_context_ == cache_positions.front());
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
    const auto query_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * staged_split_tokens_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_, score_values * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(kWorkspaceRows) *
                           query_heads_ * sizeof(float),
        staged_split_tokens_};
    if (continues_device_layer) {
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

    const auto query_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * head_dim_;
    const auto staged_kv_values = static_cast<std::size_t>(kv_heads_) *
                                  staged_split_tokens_ * head_dim_;
    const auto score_values = static_cast<std::size_t>(kWorkspaceRows) *
                              query_heads_ * staged_split_tokens_;
    const ec::HostFp16GatedGqaAttentionWorkspace workspace{
        staged_queries_, query_values * sizeof(std::uint16_t),
        staged_raw_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_raw_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_keys_, staged_kv_values * sizeof(std::uint16_t),
        staged_values_, staged_kv_values * sizeof(std::uint16_t),
        staged_scores_, score_values * sizeof(float),
        staged_probabilities_, score_values * sizeof(std::uint16_t),
        staged_accumulator_, query_values * sizeof(float), partial_maxima_,
        static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
            sizeof(float),
        partial_sums_, static_cast<std::size_t>(kWorkspaceRows) *
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
  const auto mixed_mtp = target_fp8 &&
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
    if (target_fp8 && !mixed_mtp)
      status_check(ec::store_gqa_kv_paged_fp8_batch(
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
    if (target_fp8 && !mixed_mtp) {
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
          kAttentionSplitTokens, attention_maximum_splits_, nullptr}));
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
          kAttentionSplitTokens, attention_maximum_splits_, nullptr}));
    }
  } else {
    if (target_fp8 && !mixed_mtp) {
      if (mrope_positions.empty())
        status_check(ec::gated_gqa_qkv_rope_cache_paged_fp8_batch(
          query_gate_, key_, value_, binding(operation, "query_norm").f32,
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
    if (rows * (query_heads_ / kv_heads_) <= 16U) {
      const ec::PagedFp4GatedGqaPrefillLaunch launch{
          query_gate_, page_table, attention_, partial_maxima_, partial_sums_,
          partial_outputs_, cache_positions.front() + 1U, rows,
          physical_layer, kv_page_tokens_, query_heads_, kv_heads_, head_dim_,
          kAttentionSplitTokens, attention_maximum_splits_, nullptr};
      if (target_fp8 && !mixed_mtp)
        status_check(
            ec::gated_gqa_attention_microbatch_paged_fp8_tensor_core(launch));
      else
        status_check(
            ec::gated_gqa_attention_microbatch_paged_fp4_tensor_core(launch));
    } else {
      const ec::PagedFp4GatedGqaPrefillLaunch launch{
          query_gate_, page_table, attention_, partial_maxima_, partial_sums_,
          partial_outputs_, cache_positions.front() + 1U, rows,
          physical_layer, kv_page_tokens_, query_heads_, kv_heads_,
          head_dim_, kAttentionSplitTokens,
          attention_maximum_splits_, nullptr};
      const auto query_values =
          static_cast<std::size_t>(kWorkspaceRows) * query_heads_ * head_dim_;
      const auto kv_values = static_cast<std::size_t>(kv_heads_) *
                             staged_split_tokens_ * head_dim_;
      const auto score_values =
          static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
          staged_split_tokens_;
      const ec::PagedFp4GatedGqaStagedPrefillWorkspace workspace{
          staged_queries_, query_values * sizeof(std::uint16_t),
          staged_keys_, kv_values * sizeof(std::uint16_t),
          staged_values_, kv_values * sizeof(std::uint16_t),
          staged_scores_, score_values * sizeof(float),
          staged_probabilities_, score_values * sizeof(std::uint16_t),
          staged_accumulator_, query_values * sizeof(float), partial_maxima_,
          static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
              sizeof(float),
          partial_sums_,
          static_cast<std::size_t>(kWorkspaceRows) * query_heads_ *
              sizeof(float),
          staged_split_tokens_};
      if (target_fp8 && !mixed_mtp)
        status_check(ec::gated_gqa_attention_staged_prefill_paged_fp8(
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
    std::uint32_t rows, bool checkpoint_after_first) {
  if (operation.capability == "block.mamba2.ssm.v1") {
    if (checkpoint_after_first)
      throw std::runtime_error(
          "Mamba2 does not advertise transactional draft checkpoints");
    quantize_rows(normalized_, rows, hidden_size_);
    project_quantized(binding(operation, "input_projection"), projected_qkv_,
                      rows);
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
  quantize_rows(normalized_, rows, hidden_size_);
  project_quantized(binding(operation, "qkv_projection"), projected_qkv_, rows);
  project_quantized(binding(operation, "z_projection"), projected_z_, rows);
  project_quantized(binding(operation, "b_projection"), projected_b_, rows);
  project_quantized(binding(operation, "a_projection"), projected_a_, rows);
  const auto conv_state_values =
      static_cast<std::size_t>(conv_dimension) * conv_kernel_;
  const auto matrix_state_values = static_cast<std::size_t>(value_heads_) *
                                   key_head_dim_ * value_head_dim_;
  if (rows > 1U && !checkpoint_after_first) {
    status_check(ec::split_gated_delta_prefill({
        projected_qkv_, projected_z_, projected_b_, projected_a_,
        binding(operation, "convolution").dequantized,
        binding(operation, "time_bias").f32,
        binding(operation, "decay_log").f32,
        binding(operation, "output_norm").f32,
        recurrent_conv(operation.recurrent_slot, slot),
        recurrent_matrix(operation.recurrent_slot, slot), conv_output_,
        delta_output_, rows, key_heads_, value_heads_, key_head_dim_,
        value_head_dim_, conv_kernel_, epsilon_, nullptr}));
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
        epsilon_, nullptr}));
    if (checkpoint_after_first && row == 0U) {
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
  if (!routed_experts_ || !rows || rows > kWorkspaceRows)
    throw std::runtime_error("invalid routed FP4 router batch");
  normalize_rows(hidden_, binding(operation, "input_norm").f32, normalized_,
                 rows);
  quantize_rows(normalized_, rows, hidden_size_);
  if (operation.capability ==
      "router.sigmoid-bias.topk.shared-relu2.v1") {
    project_quantized(binding(operation, "shared_up_projection"), up_, rows);
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
  project_quantized(binding(operation, "shared_gate_projection"), gate_, rows);
  project_quantized(binding(operation, "shared_up_projection"), up_, rows);
  status_check(ec::silu_product(
      gate_, up_, intermediate_, rows * shared_intermediate_size_, nullptr));
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
  if (!routed_experts_ || !rows || rows > kWorkspaceRows)
    throw std::runtime_error("invalid routed FP4 MoE batch");
  routed_experts_->execute(
      operation.component_layer, normalized_, routing_scores_,
      routing_indices_, rows, moe_intermediate_, moe_selection_output_, q8_,
      q8_scales_, moe_q8_intermediate_, moe_q8_intermediate_scales_,
      moe_output_);
  status_check(ec::add_in_place(moe_output_, shared_output_,
                                rows * hidden_size_, nullptr));
  status_check(ec::add_in_place(hidden_, moe_output_, rows * hidden_size_,
                                nullptr));
}

void DenseFp4Provider::run_ffn(const PreparedOperation& operation,
                               std::uint32_t rows) {
  cuda_check(cudaMemcpy(residual_, hidden_,
                        static_cast<std::size_t>(rows) * hidden_size_ *
                            sizeof(float),
                        cudaMemcpyDeviceToDevice),
             "retain FFN residual");
  normalize_operation_input(operation, hidden_, normalized_, rows);
  quantize_rows(normalized_, rows, hidden_size_);
  project_quantized(binding(operation, "gate_projection"), gate_, rows);
  project_quantized(binding(operation, "up_projection"), up_, rows);
  status_check(ec::silu_product(gate_, up_, intermediate_,
                                rows * intermediate_size_, nullptr));
  project(binding(operation, "down_projection"), intermediate_, hidden_, rows);
  if (operation.abi_version < 2U) {
    status_check(ec::add_in_place(hidden_, residual_, rows * hidden_size_,
                                  nullptr));
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
                                  rows * hidden_size_, nullptr));
    cuda_check(cudaMemcpy(hidden_, normalized_,
                          static_cast<std::size_t>(rows) * hidden_size_ *
                              sizeof(float),
                          cudaMemcpyDeviceToDevice),
               "commit post-normalized FFN update");
  }
}

std::vector<std::uint32_t> DenseFp4Provider::run_head(
    const PreparedOperation& operation, std::uint32_t rows,
    const er::ProgramRequestContext* request, std::uint32_t sample_position,
    bool terminal_only) {
  const auto head_rows = terminal_only ? 1U : rows;
  const auto* head_input =
      terminal_only
          ? hidden_ + static_cast<std::size_t>(rows - 1U) * hidden_size_
          : hidden_;
  normalize_rows(head_input, binding(operation, "norm").f32, normalized_,
                 head_rows);
  project(binding(operation, "weight"), normalized_, logits_, head_rows);
  if (operation.abi_version >= 2U)
    status_check(ec::scaled_tanh_in_place(
        logits_, head_rows * vocabulary_size_,
        parameter_f32(operation.parameters, "logit_multiplier_f32_bits"),
        parameter_f32(operation.parameters, "logit_softcap_f32_bits"),
        nullptr));
  if (request != nullptr &&
      request_parameter(*request, "sampling_temperature_ppm") != 0U) {
    if (head_rows != 1U)
      throw std::runtime_error(
          "sampling requires a single terminal head row");
    const auto top_k = request_parameter(*request, "sampling_top_k");
    if (top_k != 0U && top_k <= kMaximumGpuSamplingTopK) {
      status_check(ec::topk_logits(
          logits_, vocabulary_size_, static_cast<std::uint32_t>(top_k),
          sampling_top_logits_, sampling_top_tokens_, nullptr));
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
      return {sample_sorted_candidates(host_logits, host_tokens, *request,
                                       sample_position)};
    }
    std::vector<float> host_logits(vocabulary_size_);
    cuda_check(cudaMemcpy(host_logits.data(), logits_,
                          host_logits.size() * sizeof(host_logits[0]),
                          cudaMemcpyDeviceToHost),
               "copy dense FP4 sampling logits");
    ++sampling_host_calls_;
    sampling_logit_transfer_bytes_ +=
        host_logits.size() * sizeof(host_logits[0]);
    return {sample_token(host_logits, *request, sample_position)};
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
    std::span<const std::uint32_t> positions, bool transactional_second) {
  if (tokens.empty() || tokens.size() > kWorkspaceRows ||
      tokens.size() != positions.size())
    throw std::runtime_error("invalid target-model microbatch");
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  const auto slot = state.slot();
  std::vector<std::uint32_t> result;
  for (const auto& operation_pointer : prepared_target_) {
    const auto& operation = *operation_pointer;
    const auto phase_event = begin_gpu_phase(gpu_phase(operation.kernel));
    const auto stage = rows > 8U && operation.kernel != Kernel::embedding &&
                       operation.kernel != Kernel::vision &&
                       operation.kernel != Kernel::head &&
                       operation.kernel != Kernel::routed_moe;
    if (stage) activate_staged_weights(operation);
    switch (operation.kernel) {
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
        std::array<std::uint32_t, kWorkspaceRows> rotary{};
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
        run_recurrent_attention(operation, slot, rows,
                                transactional_second && rows == 2U);
        status_check(ec::add_in_place(hidden_, residual_, rows * hidden_size_,
                                      nullptr));
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
        result = run_head(operation, rows, nullptr, positions.back());
        break;
      case Kernel::exact_decode:
        throw std::runtime_error("exact service appeared in scalar program");
    }
    if (stage) deactivate_staged_weights();
    end_gpu_phase(phase_event);
    if (operation.kernel == Kernel::head) collect_gpu_phases();
  }
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

void DenseFp4Provider::checkpoint_recurrent_state(std::uint32_t slot) {
  const auto conv_values = recurrent_conv_values_;
  const auto matrix_values = recurrent_matrix_values_;
  for (std::uint32_t layer = 0U; layer < recurrent_layers_; ++layer) {
    cuda_check(cudaMemcpy(
                   recurrent_conv_checkpoint_[layer] +
                       static_cast<std::size_t>(slot) * conv_values,
                   recurrent_conv(layer, slot),
                   conv_values * sizeof(float), cudaMemcpyDeviceToDevice),
               "checkpoint exact-tier recurrent convolution state");
    cuda_check(cudaMemcpy(
                   recurrent_matrix_checkpoint_[layer] +
                       static_cast<std::size_t>(slot) * matrix_values,
                   recurrent_matrix(layer, slot),
                   matrix_values * sizeof(float), cudaMemcpyDeviceToDevice),
               "checkpoint exact-tier recurrent matrix state");
  }
}

std::vector<std::uint32_t> DenseFp4Provider::run_mtp(
    RequestState& state, std::span<const std::uint32_t> tokens,
    const float* previous_hidden,
    std::span<const std::uint32_t> rotary_positions, bool produce_logits) {
  if (!exact_ || !mtp_attention_ || !mtp_ffn_ || tokens.empty() ||
      tokens.size() > kWorkspaceRows || tokens.size() != rotary_positions.size() ||
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
  status_check(ec::fp4_embedding_batch(
      binding(*exact_, "token_embedding").matrix(), output_tokens_,
      mtp_embedding_, rows, nullptr));
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
  std::array<std::uint32_t, kWorkspaceRows> cache_positions{};
  std::array<std::uint32_t, 3U * kWorkspaceRows> mrope_positions{};
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
                                nullptr));
  run_ffn(*mtp_ffn_, rows);
  state.mtp_length += rows;
  if (!produce_logits) {
    end_gpu_phase(phase_event);
    return {};
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
  project(binding(*exact_, "output_head"), normalized_, logits_, 1U);
  status_check(ec::argmax_batch(logits_, vocabulary_size_, 1U,
                                output_tokens_, nullptr));
  std::vector<std::uint32_t> result(1U);
  cuda_check(cudaMemcpy(result.data(), output_tokens_, sizeof(result[0]),
                        cudaMemcpyDeviceToHost),
             "copy final MTP draft token");
  end_gpu_phase(phase_event);
  collect_gpu_phases();
  return result;
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
        case Kernel::router:
          if (!hidden || expert_input || route_indices || route_weights ||
              residual || shared_output ||
              compiled.input_values.size() != 1U ||
              compiled.output_values.size() != 5U ||
              value_for_port(compiled, "hidden", false) != hidden)
            return false;
          expert_input = value_for_port(compiled, "expert_input", true);
          route_indices = value_for_port(compiled, "route_indices", true);
          route_weights = value_for_port(compiled, "route_weights", true);
          residual = value_for_port(compiled, "residual", true);
          shared_output = value_for_port(compiled, "shared_output", true);
          if (!expert_input || !route_indices || !route_weights || !residual ||
              !shared_output)
            return false;
          hidden.reset();
          break;
        case Kernel::routed_moe:
          if (hidden || !expert_input || !route_indices || !route_weights ||
              !residual || !shared_output ||
              compiled.input_values.size() != 5U ||
              compiled.output_values.size() != 1U ||
              value_for_port(compiled, "expert_input", false) !=
                  expert_input ||
              value_for_port(compiled, "route_indices", false) !=
                  route_indices ||
              value_for_port(compiled, "route_weights", false) !=
                  route_weights ||
              value_for_port(compiled, "residual", false) != residual ||
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
    return !hidden && !expert_input && !route_indices && !route_weights &&
           !residual && !shared_output;
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
    cuda_check(cudaMemcpy(
                   slot_last_hidden(state->slot()),
                   slot_retention_last_hidden_ +
                       static_cast<std::size_t>(state->slot()) * hidden_size_,
                   hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
               "rewind retained target hidden state");
    restore_window_checkpoint(state->slot());
    if (host_authoritative_fp16_kv()) trim_host_kv(*state, next_position);
    state->current_position = next_position - 1U;
    state->current_batch_first = next_position - 1U;
    state->current_batch_rows = 0U;
    state->mtp_length = next_position;
    state->synchronization_first = 0U;
    state->synchronization_rows = 0U;
    state->synchronization_consumed = 0U;
    state->draft_valid = false;
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
    const auto state_bytes =
        recurrent_values * sizeof(float) +
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

std::optional<er::OperationExecutionResult>
DenseFp4Provider::poll_program_sequence(
    const std::shared_ptr<SequenceState>& sequence) {
  if (!sequence || sequence->terminal) return std::nullopt;
  if (sequence->cancelled.load()) {
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
      cuda_check(cudaMemcpy(
                     hidden_,
                     tile_hidden + (sequence->tile_rows - 1U) * hidden_size_,
                     hidden_size_ * sizeof(float), cudaMemcpyDeviceToDevice),
                 "load final program-sequence hidden state");
      const auto phase_event = begin_gpu_phase(GpuPhase::head);
      auto predictions = run_head(operation, 1U, &sequence->generation,
                                  sequence->positions.back(), true);
      auto* retained_hidden = slot_target_hidden_batch_ +
                              static_cast<std::size_t>(
                                  sequence->request->slot()) *
                                  kWorkspaceRows * hidden_size_;
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
        kWorkspaceRows, sequence->tile_rows - sequence->next_row);
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
    if (operation.kernel == Kernel::embedding) {
      run_embedding(operation, sequence->tokens.data() + offset, rows);
      cuda_check(cudaMemcpy(
                     tile_hidden + tile_offset * hidden_size_, hidden_,
                     hidden_bytes, cudaMemcpyDeviceToDevice),
                 "store program-sequence embedding tile");
      program_sequence_device_bytes_ += hidden_bytes;
    } else {
      cuda_check(cudaMemcpy(hidden_,
                            tile_hidden + tile_offset * hidden_size_,
                            hidden_bytes, cudaMemcpyDeviceToDevice),
                 "load program-sequence hidden tile");
      const auto single_tile = total_rows <= sequence_tile_rows_;
      if (single_tile) activate_staged_weights(operation);
      switch (operation.kernel) {
        case Kernel::full_attention:
        case Kernel::recurrent_attention: {
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
          const auto positions = std::span(sequence->positions)
                                     .subspan(offset, rows);
          if (operation.kernel == Kernel::full_attention) {
            std::array<std::uint32_t, kWorkspaceRows> adjusted{};
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
                                    false);
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
          if (operation.kernel == Kernel::full_attention)
            finish_attention_block(operation, rows);
          else
            status_check(ec::add_in_place(hidden_, residual_,
                                          rows * hidden_size_, nullptr));
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
      if (single_tile) deactivate_staged_weights();
      cuda_check(cudaMemcpy(
                     tile_hidden + tile_offset * hidden_size_, hidden_,
                     hidden_bytes, cudaMemcpyDeviceToDevice),
                 "store program-sequence hidden tile");
      program_sequence_device_bytes_ += 2U * hidden_bytes;
    }
    if (!phase_ended) end_gpu_phase(phase_event);
    collect_gpu_phases();
    sequence->next_row += rows;
    if (sequence->next_row == sequence->tile_rows) {
      sequence->next_row = 0U;
      sequence->next_operation += operation_advance;
    }
    return std::nullopt;
  } catch (const std::exception& error) {
    deactivate_staged_weights();
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
        if (operation->kernel == Kernel::full_attention) {
          std::array<std::uint32_t, kWorkspaceRows> rotary{};
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
          run_recurrent_attention(*operation, state->slot(), rows, false);
        }
        if (operation->kernel == Kernel::full_attention)
          finish_attention_block(*operation, rows);
        else
          status_check(ec::add_in_place(
              hidden_, residual_, rows * hidden_size_, nullptr));
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
        require_device_value(
            invocation_input(*operation, invocation, "residual"), kHiddenAbi,
            hidden_, hidden_bytes);
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
            static_cast<std::size_t>(state->slot()) * kWorkspaceRows *
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
        }
        auto owner =
            std::make_shared<std::vector<std::uint32_t>>(
                run_head(*operation, rows, &invocation.request,
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
        synchronization.next_tokens.size() > kWorkspaceRows ||
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
        static_cast<std::size_t>(state->slot()) * kWorkspaceRows *
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
      std::array<std::uint32_t, kWorkspaceRows> positions{};
      for (std::uint32_t row = 0U; row < mtp_rows; ++row)
        positions[row] =
            synchronization.first_target_position + row + 1U;
      const auto produce_draft = synchronization.produce_final_draft &&
                                 mtp_rows == rows;
      auto drafts = run_mtp(
          *state, synchronization.next_tokens.first(mtp_rows), previous_hidden,
          std::span(positions).first(mtp_rows), produce_draft);
      if (produce_draft) {
        if (drafts.size() != 1U)
          throw std::runtime_error("MTP synchronization produced no draft");
        state->draft_token = drafts.front();
        state->draft_valid = true;
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

    const std::array target_tokens{invocation.guaranteed_token,
                                   state->draft_token};
    const std::array target_positions{invocation.position,
                                      invocation.position + 1U};
    const auto target_outputs =
        run_target(*state, target_tokens, target_positions, true);
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
        state->draft_token = drafts.back();
        state->draft_valid = true;
      }
    } else if (!accepted) {
      const std::array mtp_tokens{next_token};
      const std::array mtp_positions{invocation.position + 1U};
      auto drafts = run_mtp(*state, mtp_tokens, hidden_, mtp_positions,
                            future_exact && !accepted);
      if (future_exact) {
        state->draft_token = drafts.back();
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
    std::string_view placement_profile, bool profile_gpu_phases) {
  try {
    auto artifact = std::make_shared<er::ModelArtifact>();
    auto status = er::ModelArtifact::load(artifact_root, *artifact);
    if (!status.ok()) return {status, {}};
    auto tensor_store = std::make_shared<er::MappedModelTensorStore>();
    status = er::MappedModelTensorStore::create(*artifact, *tensor_store);
    if (!status.ok()) return {status, {}};
    auto implementation = std::make_shared<DenseFp4Provider>(
        artifact, tensor_store, max_context, capacity, ram_cache_bytes,
        vram_cache_bytes, kv_cache_bytes, kv_page_tokens, kv_cache_dtype,
        profile_gpu_phases);
    er::ExecutionProviderModule module;
    module.definition = {"sm86-dense-fp4", 200U, provider_capabilities(),
                         implementation};
    module.tensor_store = std::move(tensor_store);
    const auto mtp = artifact->model().exact_decode_program.has_value();
    const auto routed = !artifact->model().routed_components.empty();
    if (routed && placement_profile != "latency" &&
        placement_profile != "balanced" &&
        placement_profile != "capacity")
      throw std::runtime_error("unsupported routed FP4 placement profile");
    module.service = {
        "causal_layer_major",
        kWorkspaceRows,
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
    module.service.session_park_ram_bytes =
        module.service.session_parking
            ? implementation->parking_ram_capacity_bytes()
            : 0U;
    module.service.session_park_page_capacity =
        module.service.session_parking
            ? implementation->parking_ram_capacity_bytes() /
                  implementation->kv_page_bytes()
            : 0U;
    module.telemetry = [implementation] { return implementation->telemetry(); };
    return {er::Status::success(), std::move(module)};
  } catch (const std::exception& error) {
    return {{er::ErrorCode::internal, error.what()}, {}};
  }
}
