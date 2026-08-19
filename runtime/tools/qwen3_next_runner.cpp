#include "expert/runtime/adaptive_placement.hpp"
#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cpu/expert_executor.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_catalog.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/execution_provider.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/model_descriptor.hpp"
#include "expert/runtime/program_executor.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/routed_expert_runtime.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"
#include "expert/runtime/worker_contract.hpp"
#include "expert/runtime/worker_provider.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
std::vector<expert::runtime::KernelCapability> provider_capabilities() {
  return {
      {"block.full-attention.output-gated.v1", 1U, 1U},
      {"block.recurrent-linear-attention.gated-delta.v1", 1U, 1U},
      {"router.linear-topk.shared-swiglu.v1", 1U, 1U},
      {"moe.swiglu.routed.merge-shared.v1", 1U, 1U},
      {"embedding.lookup.int8-row.v1", 1U, 1U},
      {"head.rmsnorm.argmax.int8-row.v1", 1U, 1U}};
}
void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}
void status_check(const expert::runtime::Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template <typename T>
T* device_allocate(std::size_t count) {
  void* raw = nullptr;
  cuda_check(cudaMalloc(&raw, count * sizeof(T)), "cudaMalloc");
  return reinterpret_cast<T*>(raw);
}
struct DevicePack final {
  std::byte* base{};
  std::uint64_t bytes{};
};

std::string digest_hex(const expert::runtime::Sha256Digest& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    const auto value = std::to_integer<unsigned>(digest[index]);
    result[2U * index] = digits[value >> 4U];
    result[2U * index + 1U] = digits[value & 0x0fU];
  }
  return result;
}

class MoeTraceWriter final {
 public:
  MoeTraceWriter(std::filesystem::path root,
                 std::vector<std::uint32_t> layers,
                 std::uint32_t hidden, std::uint32_t top_k,
                 std::uint64_t maximum_records)
      : root_(std::move(root)), layers_(std::move(layers)), hidden_(hidden),
        top_k_(top_k), maximum_records_(maximum_records) {
    if (layers_.empty() || !hidden_ || !top_k_)
      throw std::runtime_error("invalid MoE trace geometry");
    std::sort(layers_.begin(), layers_.end());
    if (std::adjacent_find(layers_.begin(), layers_.end()) != layers_.end())
      throw std::runtime_error("duplicate MoE trace layer");
    selected_layers_.insert(layers_.begin(), layers_.end());
    std::filesystem::create_directories(root_);
    open(input_, "input.f32");
    open(output_, "output.f32");
    open(layer_, "layer.u32");
    open(sequence_, "sequence.u32");
    open(position_, "position.u32");
    open(route_indices_, "route-indices.u32");
    open(route_scores_, "route-scores.f32");
  }

  bool selected(std::uint32_t layer) const noexcept {
    return selected_layers_.contains(layer) && !full();
  }
  bool full() const noexcept {
    return maximum_records_ && records_ >= maximum_records_;
  }
  std::uint64_t records() const noexcept { return records_; }

  void append(std::uint32_t layer, const float* device_input,
              const float* device_output,
              const std::uint32_t* device_route_indices,
              const float* device_route_scores,
              std::span<const std::uint32_t> positions,
              std::span<const std::uint32_t> state_slots,
              std::span<const std::uint32_t> sequence_ids) {
    if (!selected(layer)) return;
    if (positions.size() != state_slots.size())
      throw std::runtime_error("MoE trace row metadata mismatch");
    auto rows = positions.size();
    if (maximum_records_)
      rows = std::min<std::size_t>(
          rows, static_cast<std::size_t>(maximum_records_ - records_));
    if (!rows) return;

    const auto vector_values = rows * hidden_;
    const auto route_values = rows * top_k_;
    host_input_.resize(vector_values);
    host_output_.resize(vector_values);
    host_route_indices_.resize(route_values);
    host_route_scores_.resize(route_values);
    host_layers_.assign(rows, layer);
    host_sequences_.resize(rows);
    host_positions_.assign(positions.begin(), positions.begin() + rows);
    for (std::size_t row = 0; row < rows; ++row) {
      if (state_slots[row] >= sequence_ids.size())
        throw std::runtime_error("MoE trace state slot outside sequence map");
      host_sequences_[row] = sequence_ids[state_slots[row]];
    }

    cuda_check(cudaMemcpy(host_input_.data(), device_input,
                          vector_values * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy MoE trace input");
    cuda_check(cudaMemcpy(host_output_.data(), device_output,
                          vector_values * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy MoE trace output");
    cuda_check(cudaMemcpy(host_route_indices_.data(), device_route_indices,
                          route_values * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "copy MoE trace route indices");
    cuda_check(cudaMemcpy(host_route_scores_.data(), device_route_scores,
                          route_values * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy MoE trace route scores");

    write(input_, host_input_);
    write(output_, host_output_);
    write(layer_, host_layers_);
    write(sequence_, host_sequences_);
    write(position_, host_positions_);
    write(route_indices_, host_route_indices_);
    write(route_scores_, host_route_scores_);
    records_ += rows;
  }

  void finalize() {
    if (finalized_) return;
    finalized_ = true;
    close(input_);
    close(output_);
    close(layer_);
    close(sequence_);
    close(position_);
    close(route_indices_);
    close(route_scores_);
    std::ofstream manifest(root_ / "manifest.json", std::ios::binary);
    if (!manifest) throw std::runtime_error("cannot create MoE trace manifest");
    manifest << "{\n  \"format\": \"quantum-llm-moe-trace-v1\",\n"
             << "  \"record_count\": " << records_ << ",\n"
             << "  \"hidden_size\": " << hidden_ << ",\n"
             << "  \"top_k\": " << top_k_ << ",\n"
             << "  \"maximum_records\": " << maximum_records_ << ",\n"
             << "  \"layers\": [";
    for (std::size_t index = 0; index < layers_.size(); ++index) {
      if (index) manifest << ", ";
      manifest << layers_[index];
    }
    manifest << "],\n  \"files\": {\n";
    write_manifest_file(manifest, input_, true);
    write_manifest_file(manifest, output_, true);
    write_manifest_file(manifest, layer_, true);
    write_manifest_file(manifest, sequence_, true);
    write_manifest_file(manifest, position_, true);
    write_manifest_file(manifest, route_indices_, true);
    write_manifest_file(manifest, route_scores_, false);
    manifest << "  }\n}\n";
    if (!manifest) throw std::runtime_error("failed to write MoE trace manifest");
  }

 private:
  struct File final {
    std::string name;
    std::ofstream stream;
    expert::runtime::Sha256 hash;
    expert::runtime::Sha256Digest digest{};
    std::uint64_t bytes{};
  };

  void open(File& file, std::string name) {
    file.name = std::move(name);
    const auto path = root_ / file.name;
    if (std::filesystem::exists(path))
      throw std::runtime_error("MoE trace output already exists: " +
                               path.string());
    file.stream.open(path, std::ios::binary);
    if (!file.stream)
      throw std::runtime_error("cannot create MoE trace file: " + path.string());
  }

  template <typename T>
  void write(File& file, const std::vector<T>& values) {
    const auto bytes = std::as_bytes(std::span(values));
    file.stream.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
    if (!file.stream)
      throw std::runtime_error("failed writing MoE trace file: " + file.name);
    file.hash.update(bytes);
    file.bytes += bytes.size();
  }

  void close(File& file) {
    file.stream.close();
    if (!file.stream)
      throw std::runtime_error("failed closing MoE trace file: " + file.name);
    file.digest = file.hash.finalize();
  }

  static void write_manifest_file(std::ostream& output, const File& file,
                                  bool comma) {
    output << "    \"" << file.name << "\": {\"bytes\": " << file.bytes
           << ", \"sha256\": \"" << digest_hex(file.digest) << "\"}"
           << (comma ? "," : "") << "\n";
  }

  std::filesystem::path root_;
  std::vector<std::uint32_t> layers_;
  std::unordered_set<std::uint32_t> selected_layers_;
  std::uint32_t hidden_{}, top_k_{};
  std::uint64_t maximum_records_{}, records_{};
  bool finalized_{};
  File input_, output_, layer_, sequence_, position_, route_indices_,
      route_scores_;
  std::vector<float> host_input_, host_output_, host_route_scores_;
  std::vector<std::uint32_t> host_route_indices_, host_layers_, host_sequences_,
      host_positions_;
};

DevicePack upload_dense_pack(
    const std::filesystem::path& path, std::uint64_t expected_bytes,
    const expert::runtime::Sha256Digest& expected_sha) {
  if (std::filesystem::file_size(path) != expected_bytes)
    throw std::runtime_error("dense pack size mismatch");
  DevicePack result{device_allocate<std::byte>(
                        static_cast<std::size_t>(expected_bytes)),
                    expected_bytes};
  std::ifstream input(path, std::ios::binary);
  constexpr std::size_t chunk_bytes = 64U * 1024U * 1024U;
  std::vector<std::byte> chunk(chunk_bytes);
  expert::runtime::Sha256 hasher;
  std::uint64_t offset = 0;
  while (offset < expected_bytes) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(chunk.size(), expected_bytes - offset));
    input.read(reinterpret_cast<char*>(chunk.data()),
               static_cast<std::streamsize>(count));
    if (input.gcount() != static_cast<std::streamsize>(count))
      throw std::runtime_error("short dense pack read");
    hasher.update(std::span<const std::byte>(chunk.data(), count));
    cuda_check(cudaMemcpy(result.base + offset, chunk.data(), count,
                          cudaMemcpyHostToDevice),
               "upload dense pack");
    offset += count;
  }
  if (!expert::runtime::constant_time_equal(hasher.finalize(), expected_sha))
    throw std::runtime_error("dense pack SHA-256 mismatch");
  return result;
}

struct Tensor final {
  expert::runtime::cuda::Int8Matrix int8;
  const float* f32{};
  std::vector<std::uint32_t> shape;
  bool quantized{};
};

constexpr std::uint32_t kNoOperation =
    std::numeric_limits<std::uint32_t>::max();
struct LayerOperations final {
  std::uint32_t block_operation{kNoOperation};
  std::uint32_t router_operation{kNoOperation};
  std::uint32_t routed_operation{kNoOperation};
  std::uint32_t full_attention_slot{kNoOperation};
  bool full_attention{};
};

struct PhaseTelemetry final {
  std::uint64_t forward_calls{};
  std::uint64_t forward_wall_ns{};
  std::uint64_t dense_router_ns{};
  std::uint64_t attention_delta_ns{};
  std::uint64_t shared_router_ns{};
  std::uint64_t shared_expert_ns{};
  std::uint64_t router_ns{};
  std::uint64_t expert_cache_wait_ns{};
  std::uint64_t expert_compute_ns{};
  std::uint64_t cpu_expert_ns{};
  std::uint64_t gpu_expert_ns{};
  std::uint64_t cpu_gpu_overlap_ns{};
  std::uint64_t cpu_expert_groups{};
  std::uint64_t cpu_expert_selections{};
  std::uint64_t gpu_expert_selections{};
  std::uint64_t adaptive_promotions{};
  std::uint64_t frozen_promotions{};
  std::uint64_t frozen_promotion_bytes{};
  std::uint64_t useful_prefetches{};
  std::uint64_t useful_prefetch_bytes{};
  std::uint64_t wasted_prefetches{};
  std::uint64_t wasted_prefetch_bytes{};
  std::uint64_t stale_prefetch_cancellations{};
  std::uint64_t prefetch_candidate_evictions{};
  std::uint64_t prefetch_credit_rejections{};
  std::uint64_t cpu_result_h2d_bytes{};
  std::uint64_t cpu_result_map_h2d_bytes{};
  std::uint64_t final_head_ns{};
};

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point started) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started).count());
}

PhaseTelemetry phase_delta(const PhaseTelemetry& value,
                           const PhaseTelemetry& baseline) {
  return {
      value.forward_calls - baseline.forward_calls,
      value.forward_wall_ns - baseline.forward_wall_ns,
      value.dense_router_ns - baseline.dense_router_ns,
      value.attention_delta_ns - baseline.attention_delta_ns,
      value.shared_router_ns - baseline.shared_router_ns,
      value.shared_expert_ns - baseline.shared_expert_ns,
      value.router_ns - baseline.router_ns,
      value.expert_cache_wait_ns - baseline.expert_cache_wait_ns,
      value.expert_compute_ns - baseline.expert_compute_ns,
      value.cpu_expert_ns - baseline.cpu_expert_ns,
      value.gpu_expert_ns - baseline.gpu_expert_ns,
      value.cpu_gpu_overlap_ns - baseline.cpu_gpu_overlap_ns,
      value.cpu_expert_groups - baseline.cpu_expert_groups,
      value.cpu_expert_selections - baseline.cpu_expert_selections,
      value.gpu_expert_selections - baseline.gpu_expert_selections,
      value.adaptive_promotions - baseline.adaptive_promotions,
      value.frozen_promotions - baseline.frozen_promotions,
      value.frozen_promotion_bytes - baseline.frozen_promotion_bytes,
      value.useful_prefetches - baseline.useful_prefetches,
      value.useful_prefetch_bytes - baseline.useful_prefetch_bytes,
      value.wasted_prefetches - baseline.wasted_prefetches,
      value.wasted_prefetch_bytes - baseline.wasted_prefetch_bytes,
      value.stale_prefetch_cancellations -
          baseline.stale_prefetch_cancellations,
      value.prefetch_candidate_evictions -
          baseline.prefetch_candidate_evictions,
      value.prefetch_credit_rejections -
          baseline.prefetch_credit_rejections,
      value.cpu_result_h2d_bytes - baseline.cpu_result_h2d_bytes,
      value.cpu_result_map_h2d_bytes - baseline.cpu_result_map_h2d_bytes,
      value.final_head_ns - baseline.final_head_ns,
  };
}

void print_phase_json(std::ostream& output, const PhaseTelemetry& phase) {
  const auto classified = phase.dense_router_ns + phase.expert_cache_wait_ns +
                          phase.expert_compute_ns + phase.final_head_ns;
  const auto unattributed = phase.forward_wall_ns > classified
                                ? phase.forward_wall_ns - classified
                                : 0ULL;
  constexpr double ns_per_second = 1'000'000'000.0;
  const auto cpu_ns_per_selection =
      phase.cpu_expert_selections
          ? static_cast<double>(phase.cpu_expert_ns) /
                static_cast<double>(phase.cpu_expert_selections)
          : 0.0;
  const auto gpu_ns_per_selection =
      phase.gpu_expert_selections
          ? static_cast<double>(phase.gpu_expert_ns) /
                static_cast<double>(phase.gpu_expert_selections)
          : 0.0;
  const auto overlap_denominator =
      std::min(phase.cpu_expert_ns, phase.gpu_expert_ns);
  const auto overlap_ratio =
      overlap_denominator
          ? static_cast<double>(phase.cpu_gpu_overlap_ns) /
                static_cast<double>(overlap_denominator)
          : 0.0;
  output << ",\"forward_calls\":" << phase.forward_calls
         << ",\"forward_wall_seconds\":"
         << phase.forward_wall_ns / ns_per_second
         << ",\"dense_attention_router_seconds\":"
         << phase.dense_router_ns / ns_per_second
         << ",\"attention_delta_seconds\":"
         << phase.attention_delta_ns / ns_per_second
         << ",\"shared_router_seconds\":"
         << phase.shared_router_ns / ns_per_second
         << ",\"shared_expert_seconds\":"
         << phase.shared_expert_ns / ns_per_second
         << ",\"router_seconds\":" << phase.router_ns / ns_per_second
         << ",\"expert_cache_wait_seconds\":"
         << phase.expert_cache_wait_ns / ns_per_second
         << ",\"expert_compute_seconds\":"
         << phase.expert_compute_ns / ns_per_second
         << ",\"cpu_expert_seconds\":"
         << phase.cpu_expert_ns / ns_per_second
         << ",\"gpu_expert_seconds\":"
         << phase.gpu_expert_ns / ns_per_second
         << ",\"cpu_gpu_overlap_seconds\":"
         << phase.cpu_gpu_overlap_ns / ns_per_second
         << ",\"cpu_expert_groups\":" << phase.cpu_expert_groups
         << ",\"cpu_expert_selections\":" << phase.cpu_expert_selections
         << ",\"gpu_expert_selections\":" << phase.gpu_expert_selections
         << ",\"cpu_expert_ns_per_selection\":"
         << cpu_ns_per_selection
         << ",\"gpu_expert_ns_per_selection\":"
         << gpu_ns_per_selection
         << ",\"cpu_gpu_overlap_ratio\":" << overlap_ratio
         << ",\"adaptive_promotions\":" << phase.adaptive_promotions
         << ",\"frozen_promotions\":" << phase.frozen_promotions
         << ",\"frozen_promotion_bytes\":" << phase.frozen_promotion_bytes
         << ",\"useful_prefetches\":" << phase.useful_prefetches
         << ",\"useful_prefetch_bytes\":" << phase.useful_prefetch_bytes
         << ",\"wasted_prefetches\":" << phase.wasted_prefetches
         << ",\"wasted_prefetch_bytes\":" << phase.wasted_prefetch_bytes
         << ",\"stale_prefetch_cancellations\":"
         << phase.stale_prefetch_cancellations
         << ",\"prefetch_candidate_evictions\":"
         << phase.prefetch_candidate_evictions
         << ",\"prefetch_credit_rejections\":"
         << phase.prefetch_credit_rejections
         << ",\"cpu_result_h2d_bytes\":" << phase.cpu_result_h2d_bytes
         << ",\"cpu_result_map_h2d_bytes\":"
         << phase.cpu_result_map_h2d_bytes
         << ",\"final_head_seconds\":"
         << phase.final_head_ns / ns_per_second
         << ",\"unattributed_seconds\":"
         << unattributed / ns_per_second;
}

void print_dispatch_json(
    std::ostream& output,
    const expert::runtime::HybridDispatchTelemetry& telemetry,
    const expert::runtime::HybridDispatchTelemetry& baseline) {
  output << ",\"dispatch_plans\":" << telemetry.plans - baseline.plans
         << ",\"dispatch_candidates\":"
         << telemetry.candidates - baseline.candidates
         << ",\"dispatch_resident_gpu\":"
         << telemetry.resident_gpu - baseline.resident_gpu
         << ",\"dispatch_cpu_only\":"
         << telemetry.cpu_only - baseline.cpu_only
         << ",\"dispatch_gpu_only\":"
         << telemetry.gpu_only - baseline.gpu_only
         << ",\"dispatch_cpu_cost_wins\":"
         << telemetry.cpu_cost_wins - baseline.cpu_cost_wins
         << ",\"dispatch_gpu_cost_wins\":"
         << telemetry.gpu_cost_wins - baseline.gpu_cost_wins
         << ",\"dispatch_stable_ties\":"
         << telemetry.stable_ties - baseline.stable_ties
         << ",\"dispatch_rejected_plans\":"
         << telemetry.rejected_plans - baseline.rejected_plans
         << ",\"dispatch_pre_measurement_plans\":" << baseline.plans
         << ",\"dispatch_pre_measurement_cpu_cost_wins\":"
         << baseline.cpu_cost_wins
         << ",\"dispatch_pre_measurement_gpu_cost_wins\":"
         << baseline.gpu_cost_wins
         << ",\"dispatch_pre_measurement_cpu_only\":"
         << baseline.cpu_only
         << ",\"dispatch_pre_measurement_gpu_only\":"
         << baseline.gpu_only
         << ",\"dispatch_cpu_ns_per_selection\":"
         << telemetry.cpu_ns_per_selection
         << ",\"dispatch_gpu_ns_per_selection\":"
         << telemetry.gpu_ns_per_selection
         << ",\"dispatch_h2d_bytes_per_second\":"
         << telemetry.h2d_bytes_per_second;
}

void print_cpu_executor_json(
    std::ostream& output,
    const expert::runtime::cpu::ExpertExecutorTelemetry& telemetry,
    const expert::runtime::cpu::ExpertExecutorTelemetry& baseline) {
  const auto compute_ns = telemetry.compute_ns - baseline.compute_ns;
  const auto weight_bytes = telemetry.effective_weight_bytes -
                            baseline.effective_weight_bytes;
  const auto bytes_per_second =
      compute_ns ? static_cast<double>(weight_bytes) * 1.0e9 /
                       static_cast<double>(compute_ns)
                 : 0.0;
  output << ",\"cpu_executor_maximum_threads\":"
         << telemetry.maximum_threads
         << ",\"cpu_executor_selected_threads\":"
         << telemetry.selected_threads
         << ",\"cpu_executor_gate_chunk\":" << telemetry.gate_chunk
         << ",\"cpu_executor_down_chunk\":" << telemetry.down_chunk
         << ",\"cpu_executor_calibration_runs\":"
         << telemetry.calibration_runs
         << ",\"cpu_executor_calibration_seconds\":"
         << telemetry.calibration_ns / 1.0e9
         << ",\"cpu_executor_calls\":"
         << telemetry.execute_calls - baseline.execute_calls
         << ",\"cpu_executor_selections\":"
         << telemetry.selections - baseline.selections
         << ",\"cpu_executor_effective_weight_bytes\":" << weight_bytes
         << ",\"cpu_executor_compute_seconds\":" << compute_ns / 1.0e9
         << ",\"cpu_executor_effective_weight_bytes_per_second\":"
         << bytes_per_second;
}

// W4: minimum victim age in cache access-clock ticks (one tick per routed
// selection fed back to the cache) before an unreferenced VRAM resident
// counts as a dead-topic corpse eligible for frozen re-promotion
// displacement. Measured pacing: a 48-token chat turn with a short delta
// prefill advances the clock by roughly 60-80k ticks, so 2^19 ticks lets a
// resident survive about seven unrouted turns — long enough that a repeating
// or drifting route never displaces itself, short enough that a settled new
// topic reclaims VRAM from dead ones. A 2^16 threshold was measured to
// ping-pong: at about one turn of tolerance every topic change re-uploaded
// the whole turn's misses (10-20 GiB H2D per turn) instead of using the CPU
// executor overflow.
constexpr std::uint64_t kFrozenPromotionMinVictimAge = 1ULL << 19U;

class Qwen3NextModel final : public expert::runtime::IOperationProvider,
                             public expert::runtime::IModelTensorStore {
 public:
  Qwen3NextModel(const std::filesystem::path& root, std::uint32_t max_context,
                 std::uint64_t ram_cache_bytes,
                 std::uint64_t vram_cache_bytes,
                 std::uint32_t capacity = 1,
                 std::uint64_t kv_cache_bytes = 2ULL << 30U,
                 std::uint32_t kv_page_tokens = 256,
                 std::string_view placement_profile = "balanced",
                 std::uint32_t prefill_chunk_tokens = 0)
      : root_(root), max_context_(max_context), capacity_(capacity),
        ram_cache_bytes_(ram_cache_bytes),
        vram_cache_bytes_(vram_cache_bytes), kv_cache_bytes_(kv_cache_bytes),
        kv_page_tokens_(kv_page_tokens),
        placement_profile_(placement_profile) {
    if (!capacity_ || !max_context_ || !kv_cache_bytes_ || !kv_page_tokens_)
      throw std::runtime_error("zero request/KV capacity");
    // Prefill batching is decoupled from the decode batch capacity: wide
    // causal chunks reuse the same per-slot state but need their own row
    // workspace.
    prefill_chunk_tokens_ =
        prefill_chunk_tokens ? prefill_chunk_tokens : capacity_;
    if (prefill_chunk_tokens_ > max_context_)
      throw std::runtime_error("prefill chunk exceeds configured context");
    workspace_rows_ = std::max(capacity_, prefill_chunk_tokens_);
    expert::runtime::AdaptivePlacementConfig placement_config;
    if (placement_profile_ == "latency") {
      placement_config.minimum_recent_observations = 1;
      placement_config.admission_margin = 1.0;
    } else if (placement_profile_ == "capacity") {
      placement_config.enable_prefetch = false;
    } else if (placement_profile_ != "balanced") {
      throw std::runtime_error(
          "placement profile must be latency, balanced, or capacity");
    }
    status_check(expert::runtime::ModelArtifact::load_expert_pack_v1(
        root, artifact_));
    model_descriptor_ = artifact_.model();
    if (model_descriptor_.routed_components.size() != 1U)
      throw std::runtime_error(
          "hybrid-delta provider requires one routed component");
    const auto& component = model_descriptor_.routed_components.front();
    if (component.source_abi !=
            expert::runtime::kExpertSourceAbiExpertPackV1 ||
        component.encoding_abi !=
            expert::runtime::kExpertEncodingAbiFp4Block32)
      throw std::runtime_error(
          "hybrid-delta provider requires an FP4 Expert Pack component");
    model_hash_ = model_descriptor_.content_hash;
    model_id_ = component.namespace_id;
    encoding_abi_ = component.encoding_abi;
    hidden_ = model_descriptor_.hidden_size;
    expert_width_ = component.intermediate_size;
    vocab_ = model_descriptor_.vocab_size;
    layers_ = component.layer_count;
    experts_ = component.experts_per_layer;
    top_k_ = component.route_width;
    shared_experts_ = component.shared_experts_per_layer;
    query_heads_ = descriptor_u32("attention_heads");
    kv_heads_ = descriptor_u32("kv_heads");
    head_dim_ = descriptor_u32("head_dim");
    conv_kernel_ = descriptor_u32("linear_conv_kernel");
    key_head_dim_ = descriptor_u32("linear_key_head_dim");
    value_head_dim_ = descriptor_u32("linear_value_head_dim");
    key_heads_ = descriptor_u32("linear_key_heads");
    value_heads_ = descriptor_u32("linear_value_heads");
    shared_width_ = descriptor_u32("shared_intermediate_size");
    epsilon_ = descriptor_f32("norm_epsilon_f32_bits");
    rope_theta_ = descriptor_f32("rope_theta_f32_bits");
    rotary_dim_ = descriptor_u32("rotary_dimension");
    if (max_context_ > model_descriptor_.max_context_tokens)
      throw std::runtime_error("configured context exceeds model descriptor");
    if (!query_heads_ || !kv_heads_ || query_heads_ % kv_heads_ ||
        !head_dim_ || head_dim_ > 256U || !rotary_dim_ ||
        rotary_dim_ > head_dim_ || rotary_dim_ % 2U || top_k_ > 64U ||
        !key_heads_ || !value_heads_ || value_heads_ % key_heads_ ||
        !key_head_dim_ || key_head_dim_ > 256U || !value_head_dim_ ||
        value_head_dim_ > 256U || !conv_kernel_ || conv_kernel_ > 16U ||
        shared_width_ == 0U)
      throw std::runtime_error(
          "model geometry exceeds the registered SM86 operation capabilities");
    if (component.hidden_size != hidden_ || shared_experts_ != 1U ||
        model_descriptor_.layer_program.size() != layers_)
      throw std::runtime_error(
          "hybrid-delta operation geometry is inconsistent");
    expert::runtime::ExecutionProviderRegistry provider_registry;
    auto registered = provider_registry.add(
        {"sm86-hybrid-delta-moe", 100U, provider_capabilities()});
    if (!registered.ok())
      throw std::runtime_error(std::string(registered.message()));
    auto bound = provider_registry.bind(model_descriptor_);
    if (!bound.status.ok())
      throw std::runtime_error(std::string(bound.status.message()));
    compiled_program_ = std::move(bound.provider.program);
    for (const auto& pack : artifact_.packs()) {
      total_pack_bytes_ += pack.bytes;
      if (pack.kind == "dense") {
        dense_packs_.emplace(
            pack.name, upload_dense_pack(pack.path, pack.bytes, pack.sha256));
        dense_read_bytes_ += pack.bytes;
      }
    }
    if (dense_packs_.empty())
      throw std::runtime_error("artifact has no dense tensor pack");
    std::fprintf(stderr, "[load] dense pack uploaded\n");
    for (const auto& tensor : artifact_.dense_tensors()) add_tensor(tensor);
    provider_slots_.assign(capacity_, false);
    const auto* artifact_component = artifact_.find_component(component.name);
    if (artifact_component == nullptr)
      throw std::runtime_error("artifact has no routed component catalog");
    catalog_ = &artifact_component->catalog;
    for (std::uint32_t layer = 0; layer < layers_; ++layer)
      for (std::uint32_t expert = 0; expert < experts_; ++expert)
        max_expert_record_bytes_ = std::max(
            max_expert_record_bytes_,
            catalog_->find(layer, expert)->stored_bytes);
    std::fprintf(stderr, "[load] tensors + expert index ok (max record %llu)\n",
                 static_cast<unsigned long long>(max_expert_record_bytes_));
    route_access_counts_.resize(experts_);
    route_score_sums_.resize(experts_);
    route_score_maxima_.resize(experts_);
    route_accesses_.reserve(experts_);
    routed_expert_keys_.reserve(workspace_rows_ * top_k_);
    missing_expert_keys_.reserve(workspace_rows_ * top_k_);

    const auto slot_bytes = static_cast<std::size_t>(max_expert_record_bytes_);
    const auto slot_count = std::max<std::size_t>(top_k_ * 2U, 32U);
    storage_ = std::make_shared<expert::runtime::WindowsIocpStorage>(4);
    uploader_ = std::make_shared<expert::runtime::cuda::CudaExpertUploader>();
    directory_ =
        std::make_shared<expert::runtime::cuda::CudaExpertDirectory>(
            model_id_, encoding_abi_, layers_, experts_,
            workspace_rows_ * top_k_);
    {
      auto plan_workspace = directory_->create_plan_workspace();
      if (!plan_workspace.status.ok() || !plan_workspace.workspace)
        throw std::runtime_error(
            std::string("CUDA directory plan workspace: ") +
            std::string(plan_workspace.status.ok()
                            ? "unavailable"
                            : plan_workspace.status.message()));
      plan_workspace_ = std::move(plan_workspace.workspace);
    }
    std::fprintf(stderr, "[load] CUDA directory ok\n");
    buffers_ = std::make_shared<expert::runtime::FixedBufferPool>(
        slot_count, slot_bytes, expert::runtime::kExpertPackAlignment,
        std::make_shared<expert::runtime::CudaPinnedAllocator>());
    std::fprintf(stderr, "[load] buffer pool ok\n");
    const auto budget = [](std::uint64_t capacity) {
      if (!capacity) throw std::runtime_error("cache capacity is zero");
      return expert::runtime::TierBudget{
          capacity, capacity, capacity - std::min<std::uint64_t>(capacity / 8U,
                                                                 1ULL << 30U)};
    };
    const auto shared_burst = [](std::uint64_t capacity,
                                 std::uint64_t maximum) {
      return std::min(capacity / 8U, maximum);
    };
    cache_ = std::make_unique<expert::runtime::ExpertCache>(
        expert::runtime::ExpertCacheConfig{budget(ram_cache_bytes),
                                            budget(vram_cache_bytes), true,
                                            {layers_, 1,
                                             shared_burst(ram_cache_bytes,
                                                          2ULL << 30U),
                                             shared_burst(vram_cache_bytes,
                                                          1ULL << 30U)}},
        storage_, uploader_, buffers_, directory_);
    routed_ = std::make_unique<expert::runtime::RoutedExpertRuntime>(
        model_descriptor_.routed_components.front(),
        model_hash_, *catalog_, *cache_);
    const auto logical_threads = std::max(1U, std::thread::hardware_concurrency());
    cpu_executor_ =
        std::make_unique<expert::runtime::cpu::ExpertExecutor>(
            logical_threads);
    placement_ = std::make_unique<expert::runtime::AdaptivePlacementPlanner>(
        *cache_, placement_config);
    dispatch_ = std::make_unique<expert::runtime::HybridDispatchPlanner>();
    std::fprintf(stderr, "[load] cache + planners ok\n");
    initialize_program_contract();
    allocate_workspace();
    std::fprintf(stderr, "[load] workspace ok\n");
    cuda_check(cudaEventCreate(&layer_start_event_), "create layer-start event");
    cuda_check(cudaEventCreate(&attention_done_event_),
               "create attention-done event");
    cuda_check(cudaEventCreate(&shared_done_event_), "create shared-done event");
    cuda_check(cudaEventCreate(&router_done_event_), "create router-done event");
    expert_start_events_.resize(layers_);
    expert_done_events_.resize(layers_);
    gpu_selections_by_layer_.resize(layers_);
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      cuda_check(cudaEventCreate(&expert_start_events_[layer]),
                 "create expert-start event");
      cuda_check(cudaEventCreate(&expert_done_events_[layer]),
                 "create expert-done event");
    }
  }

  struct PreparedOperation final : expert::runtime::IPreparedOperation {
    std::uint32_t kernel{};
    std::map<std::string, std::size_t, std::less<>> input_indices;
    std::vector<std::pair<std::string, std::string>> outputs;
  };

  class ProviderRequestState final
      : public expert::runtime::IOperationProviderRequestState {
   public:
    ProviderRequestState(Qwen3NextModel& model, std::uint32_t slot) noexcept
        : model_(model), slot_(slot) {}
    ~ProviderRequestState() override {
      model_.release_callable_provider_slot(slot_);
    }
    [[nodiscard]] std::uint32_t slot() const noexcept { return slot_; }
    std::uint32_t current_position{};

   private:
    Qwen3NextModel& model_;
    std::uint32_t slot_{};
  };

  expert::runtime::PrepareOperationResult prepare(
      const expert::runtime::OperationPreparationContext& context) override {
    try {
      if (context.model.content_hash != model_descriptor_.content_hash)
        return {{expert::runtime::ErrorCode::invalid_argument,
                 "provider received a different model artifact"},
                {}};
      const auto capabilities = provider_capabilities();
      const auto found = std::find_if(
          capabilities.begin(), capabilities.end(), [&](const auto& item) {
            return item.capability == context.operation.capability &&
                   context.operation.abi_version >= item.minimum_abi &&
                   context.operation.abi_version <= item.maximum_abi;
          });
      if (found == capabilities.end())
        return {{expert::runtime::ErrorCode::invalid_argument,
                 "provider cannot prepare the operation capability"},
                {}};
      for (const auto& binding : context.tensors) {
        if (!binding.tensor || !tensors_.contains(binding.tensor->name))
          return {{expert::runtime::ErrorCode::invalid_argument,
                   "provider received an unavailable immutable tensor"},
                  {}};
      }
      auto prepared = std::make_shared<PreparedOperation>();
      prepared->kernel = static_cast<std::uint32_t>(found - capabilities.begin());
      for (std::size_t index = 0U;
           index < context.compiled.input_values.size(); ++index)
        prepared->input_indices.emplace(
            context.compiled.input_values[index].port, index);
      for (const auto& binding : context.compiled.output_values) {
        const auto source = context.operation.output_bindings.find(binding.port);
        if (source == context.operation.output_bindings.end())
          return {{expert::runtime::ErrorCode::invalid_argument,
                   "compiled provider output port is absent"},
                  {}};
        prepared->outputs.emplace_back(binding.port, source->second.abi);
      }
      return {expert::runtime::Status::success(), std::move(prepared)};
    } catch (const std::exception& error) {
      return {{expert::runtime::ErrorCode::internal, error.what()}, {}};
    }
  }

  expert::runtime::CreateOperationRequestStateResult create_request_state(
      const expert::runtime::ProgramRequestContext& request) override {
    const auto context = request.parameters.find("reserved_context_tokens");
    if (context == request.parameters.end() || context->second == 0U ||
        context->second > max_context_ ||
        context->second > std::numeric_limits<std::uint32_t>::max())
      return {{expert::runtime::ErrorCode::invalid_argument,
               "hybrid-delta provider requires a valid context reservation"},
              {}};
    const auto slot = acquire_callable_provider_slot();
    if (!slot)
      return {{expert::runtime::ErrorCode::backpressure,
               "hybrid-delta provider has no free request slot"},
              {}};
    try {
      reserve_slot(*slot, static_cast<std::uint32_t>(context->second));
      return {expert::runtime::Status::success(),
              std::make_shared<ProviderRequestState>(*this, *slot)};
    } catch (const std::exception& error) {
      release_callable_provider_slot(*slot);
      return {{expert::runtime::ErrorCode::internal, error.what()}, {}};
    }
  }

  expert::runtime::OperationExecutionHandle execute(
      const expert::runtime::IPreparedOperation& opaque_operation,
      const std::shared_ptr<expert::runtime::IOperationProviderRequestState>&
          opaque_state,
      const expert::runtime::OperationInvocation& invocation) override {
    try {
      const auto* prepared =
          dynamic_cast<const PreparedOperation*>(&opaque_operation);
      const auto state =
          std::dynamic_pointer_cast<ProviderRequestState>(opaque_state);
      if (prepared == nullptr || !state)
        return completed_operation({
            {expert::runtime::ErrorCode::invalid_argument,
             "hybrid-delta invocation state is invalid"},
            {}});
      const auto input = [&](std::string_view port)
          -> const expert::runtime::ExecutionValue& {
        const auto found = prepared->input_indices.find(port);
        if (found == prepared->input_indices.end() ||
            found->second >= invocation.inputs.size())
          throw std::runtime_error(
              "hybrid-delta operation input is absent");
        return invocation.inputs[found->second];
      };
      constexpr std::string_view hidden_abi = "batch.hidden.f32.cuda.v1";
      constexpr std::string_view token_abi = "batch.token-id.u32.host.v1";
      constexpr std::string_view position_abi = "batch.position.u32.host.v1";
      constexpr std::string_view route_index_abi =
          "batch.route-index.u32.cuda.v1";
      constexpr std::string_view route_weight_abi =
          "batch.route-weight.f32.cuda.v1";
      const auto require_device_value = [&](const auto& value,
                                            const void* pointer,
                                            std::uint64_t bytes,
                                            std::string_view abi) {
        if (value.abi != abi || value.memory_domain != "cuda.device" ||
            value.data != reinterpret_cast<const std::byte*>(pointer) ||
            value.bytes != bytes)
          throw std::runtime_error(
              "hybrid-delta intermediate value ABI mismatch");
      };
      const auto require_hidden = [&](const auto& value, const float* pointer) {
        require_device_value(value, pointer,
                             static_cast<std::uint64_t>(hidden_) *
                                 sizeof(float),
                             hidden_abi);
      };

      std::map<std::string, expert::runtime::ExecutionValue, std::less<>>
          outputs;
      switch (prepared->kernel) {
        case kEmbedding: {
          const auto& tokens = input("token_ids");
          if (tokens.abi != token_abi || tokens.memory_domain != "host" ||
              tokens.bytes != sizeof(std::uint32_t))
            throw std::runtime_error("embedding token ABI mismatch");
          std::uint32_t token{};
          std::memcpy(&token, tokens.data, sizeof(token));
          if (token >= vocab_)
            throw std::runtime_error("token exceeds vocabulary");
          frozen_stale_budget_valid_ = false;
          status_check(expert::runtime::cuda::embedding(
              matrix(operation_binding(invocation.operation, "weight")),
              token, hidden_state_, nullptr));
          outputs.emplace("hidden", device_value(hidden_state_, hidden_));
          break;
        }
        case kFullAttention:
        case kDeltaAttention: {
          require_hidden(input("hidden"), hidden_state_);
          const auto& positions = input("positions");
          if (positions.abi != position_abi ||
              positions.memory_domain != "host" ||
              positions.bytes != sizeof(std::uint32_t))
            throw std::runtime_error("attention position ABI mismatch");
          std::memcpy(&state->current_position, positions.data,
                      sizeof(state->current_position));
          if (state->current_position >= max_context_)
            throw std::runtime_error("attention position exceeds context");
          ensure_kv_page(state->slot(), state->current_position);
          cuda_check(cudaEventRecord(layer_start_event_),
                     "record callable layer start");
          status_check(expert::runtime::cuda::qwen3_next_rms_norm(
              hidden_state_,
              fp32(operation_binding(invocation.operation, "input_norm")),
              normalized_, hidden_, epsilon_, nullptr));
          const std::array positions_batch{state->current_position};
          const std::array slots{state->slot()};
          if (prepared->kernel == kFullAttention) {
            const auto logical_layer = invocation.operation.logical_layer;
            if (logical_layer >= layer_operations_.size() ||
                !layer_operations_[logical_layer].full_attention)
              throw std::runtime_error(
                  "full-attention operation has no artifact layer mapping");
            run_full_attention(
                invocation.operation,
                layer_operations_[logical_layer].full_attention_slot,
                positions_batch, slots, 1U);
          } else {
            run_delta(invocation.operation,
                      invocation.operation.logical_layer, slots, 1U);
          }
          status_check(expert::runtime::cuda::add_in_place(
              hidden_state_, residual_, hidden_, nullptr));
          cuda_check(cudaEventRecord(attention_done_event_),
                     "record callable attention done");
          outputs.emplace("hidden", device_value(hidden_state_, hidden_));
          break;
        }
        case kRouter:
          require_hidden(input("hidden"), hidden_state_);
          status_check(expert::runtime::cuda::qwen3_next_rms_norm(
              hidden_state_,
              fp32(operation_binding(invocation.operation, "input_norm")),
              normalized_, hidden_, epsilon_, nullptr));
          run_router(invocation.operation, 1U);
          outputs.emplace("expert_input", device_value(normalized_, hidden_));
          outputs.emplace("route_indices",
                          device_value(routing_indices_, top_k_,
                                       route_index_abi));
          outputs.emplace("route_weights",
                          device_value(routing_scores_, top_k_,
                                       route_weight_abi));
          outputs.emplace("residual", device_value(hidden_state_, hidden_));
          outputs.emplace("shared_output",
                          device_value(shared_output_, hidden_));
          break;
        case kRoutedMoe: {
          require_hidden(input("expert_input"), normalized_);
          require_device_value(
              input("route_indices"), routing_indices_,
              static_cast<std::uint64_t>(top_k_) * sizeof(std::uint32_t),
              route_index_abi);
          require_device_value(input("route_weights"), routing_scores_,
                               static_cast<std::uint64_t>(top_k_) *
                                   sizeof(float),
                               route_weight_abi);
          require_hidden(input("residual"), hidden_state_);
          require_hidden(input("shared_output"), shared_output_);
          const std::array positions{state->current_position};
          const std::array slots{state->slot()};
          run_routed_moe(invocation.operation, positions, slots, 1U);
          outputs.emplace("hidden", device_value(hidden_state_, hidden_));
          break;
        }
        case kHead: {
          require_hidden(input("hidden"), hidden_state_);
          status_check(expert::runtime::cuda::qwen3_next_rms_norm(
              hidden_state_,
              fp32(operation_binding(invocation.operation, "norm")),
              normalized_, hidden_, epsilon_, nullptr));
          status_check(expert::runtime::cuda::gemv_batch(
              matrix(operation_binding(invocation.operation, "weight")),
              normalized_, logits_, 1U, nullptr));
          status_check(expert::runtime::cuda::argmax_batch(
              logits_, vocab_, 1U, output_token_, nullptr));
          auto host = std::make_shared<std::uint32_t>();
          cuda_check(cudaMemcpy(host.get(), output_token_, sizeof(*host),
                                cudaMemcpyDeviceToHost),
                     "copy callable output token");
          outputs.emplace(
              "token_ids",
              expert::runtime::ExecutionValue{
                  std::string(token_abi), "host", host,
                  reinterpret_cast<const std::byte*>(host.get()),
                  sizeof(*host)});
          break;
        }
        default:
          throw std::runtime_error(
              "hybrid-delta callable provider kernel is unsupported");
      }

      expert::runtime::OperationExecutionResult result;
      result.status = expert::runtime::Status::success();
      result.outputs.reserve(prepared->outputs.size());
      for (const auto& [port, abi] : prepared->outputs) {
        auto found = outputs.find(port);
        if (found == outputs.end() || found->second.abi != abi)
          throw std::runtime_error("callable provider output ABI mismatch");
        result.outputs.push_back(std::move(found->second));
      }
      return completed_operation(std::move(result));
    } catch (const std::exception& error) {
      return completed_operation(
          {{expert::runtime::ErrorCode::internal, error.what()}, {}});
    }
  }

  expert::runtime::ResolveModelTensorResult resolve(
      std::string_view name) override {
    const auto entry = tensor_entries_.find(std::string(name));
    if (entry == tensor_entries_.end())
      return {{expert::runtime::ErrorCode::invalid_argument,
               "hybrid-delta tensor is absent"},
              {}};
    const auto pack = dense_packs_.find(entry->second.pack);
    if (pack == dense_packs_.end())
      return {{expert::runtime::ErrorCode::internal,
               "hybrid-delta dense pack is absent"},
              {}};
    auto tensor = std::make_shared<expert::runtime::ImmutableModelTensor>();
    tensor->name = entry->second.name;
    tensor->encoding = entry->second.encoding;
    tensor->quant_abi = entry->second.quant_abi;
    tensor->shape = entry->second.shape;
    tensor->data_offset = entry->second.data_offset;
    tensor->data_bytes = entry->second.data_bytes;
    tensor->scale_offset = entry->second.scale_offset;
    tensor->scale_bytes = entry->second.scale_bytes;
    tensor->value = {
        "artifact.dense-record.v1", "cuda.device", dense_lifetime_,
        pack->second.base + entry->second.record_offset,
        entry->second.stored_bytes};
    return {expert::runtime::Status::success(), std::move(tensor)};
  }

  std::uint32_t forward(std::uint32_t token, std::uint32_t position) {
    const std::array tokens{token};
    const std::array positions{position};
    return forward_batch(tokens, positions).front();
  }

  std::vector<std::uint32_t> forward_batch(
      std::span<const std::uint32_t> tokens,
      std::span<const std::uint32_t> positions,
      std::span<const std::uint32_t> state_slots = {},
      bool causal_same_slot = false,
      std::span<const std::uint32_t> head_rows = {}) {
    const auto forward_started = std::chrono::steady_clock::now();
    const auto cpu_expert_before = phase_.cpu_expert_ns;
    const auto gpu_expert_before = phase_.gpu_expert_ns;
    const auto expert_compute_before = phase_.expert_compute_ns;
    if (tokens.empty() || tokens.size() != positions.size() ||
        tokens.size() > workspace_rows_ ||
        (!state_slots.empty() && state_slots.size() != tokens.size()))
      throw std::runtime_error("invalid Qwen3-Next microbatch");
    if (causal_same_slot && state_slots.empty())
      throw std::runtime_error("causal chunk requires an explicit state slot");
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    // W4: the frozen re-promotion victim budget is rescanned at most once per
    // forward pass and shared by every layer's misses.
    frozen_stale_budget_valid_ = false;
    std::vector<std::uint32_t> default_slots;
    if (state_slots.empty()) {
      default_slots.resize(rows);
      std::iota(default_slots.begin(), default_slots.end(), 0U);
      state_slots = default_slots;
    }
    // Only the rows listed in head_rows run the vocabulary head; an empty
    // list means every row (the decode path). The logits/argmax workspace is
    // sized for capacity_ rows, so partial heads must fit within it.
    std::vector<std::uint32_t> default_heads;
    if (head_rows.empty()) {
      default_heads.resize(rows);
      std::iota(default_heads.begin(), default_heads.end(), 0U);
      head_rows = default_heads;
    }
    if (head_rows.size() > capacity_)
      throw std::runtime_error("head rows exceed logits workspace");
    for (const auto row : head_rows)
      if (row >= rows)
        throw std::runtime_error("head row outside microbatch");
    std::vector<bool> seen_slots(capacity_);
    for (std::uint32_t row = 0; row < rows; ++row) {
      if (positions[row] >= max_context_ || tokens[row] >= vocab_)
        throw std::runtime_error("token/position outside capacity");
      if (state_slots[row] >= capacity_)
        throw std::runtime_error("invalid request state slot");
      if (causal_same_slot) {
        if (row && (state_slots[row] != state_slots[0] ||
                    positions[row] != positions[row - 1U] + 1U))
          throw std::runtime_error(
              "causal chunk must use one slot and consecutive positions");
      } else if (seen_slots[state_slots[row]]) {
        throw std::runtime_error("duplicate request state slot");
      }
      if (!slot_context_limits_[state_slots[row]] ||
          positions[row] >= slot_context_limits_[state_slots[row]])
        throw std::runtime_error("token/position outside reserved context");
      seen_slots[state_slots[row]] = true;
      ensure_kv_page(state_slots[row], positions[row]);
      status_check(expert::runtime::cuda::embedding(
          matrix(model_binding("token_embedding")), tokens[row],
          hidden_state_ + static_cast<std::size_t>(row) * hidden_, nullptr));
    }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      const auto& layer_program = layer_operations_.at(layer);
      const auto& block =
          compiled_program_.operations.at(layer_program.block_operation);
      const auto& router =
          compiled_program_.operations.at(layer_program.router_operation);
      const auto& routed =
          compiled_program_.operations.at(layer_program.routed_operation);
      cuda_check(cudaEventRecord(layer_start_event_), "record layer start");
      for (std::uint32_t row = 0; row < rows; ++row)
        status_check(expert::runtime::cuda::qwen3_next_rms_norm(
            hidden_state_ + static_cast<std::size_t>(row) * hidden_,
            fp32(operation_binding(block, "input_norm")),
            normalized_ + static_cast<std::size_t>(row) * hidden_, hidden_,
            epsilon_, nullptr));
      if (layer_program.full_attention) {
        run_full_attention(block, layer_program.full_attention_slot, positions,
                           state_slots, rows);
      } else {
        run_delta(block, layer, state_slots, rows);
      }
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state_, residual_, rows * hidden_, nullptr));
      for (std::uint32_t row = 0; row < rows; ++row)
        status_check(expert::runtime::cuda::qwen3_next_rms_norm(
            hidden_state_ + static_cast<std::size_t>(row) * hidden_,
            fp32(operation_binding(router, "input_norm")),
            normalized_ + static_cast<std::size_t>(row) * hidden_, hidden_,
            epsilon_, nullptr));
      cuda_check(cudaEventRecord(attention_done_event_),
                 "record attention done");
      run_router(router, rows);
      run_routed_moe(routed, positions, state_slots, rows);
    }
    const auto final_head_started = std::chrono::steady_clock::now();
    for (const auto row : head_rows)
      status_check(expert::runtime::cuda::qwen3_next_rms_norm(
          hidden_state_ + static_cast<std::size_t>(row) * hidden_,
          fp32(model_binding("final_norm")),
          normalized_ + static_cast<std::size_t>(row) * hidden_, hidden_,
          epsilon_, nullptr));
    const auto head_count = static_cast<std::uint32_t>(head_rows.size());
    std::vector<std::uint32_t> result(head_rows.size());
    if (head_count == rows) {
      if (rows == 1) {
        status_check(expert::runtime::cuda::gemv_batch(
            matrix(model_binding("output_head")), normalized_, logits_, rows,
            nullptr));
      } else {
        status_check(expert::runtime::cuda::gemv_batch_weight_reuse(
            matrix(model_binding("output_head")), normalized_, logits_, rows,
            nullptr));
      }
    } else {
      for (std::uint32_t head = 0; head < head_count; ++head)
        status_check(expert::runtime::cuda::gemv_batch(
            matrix(model_binding("output_head")),
            normalized_ +
                static_cast<std::size_t>(head_rows[head]) * hidden_,
            logits_ + static_cast<std::size_t>(head) * vocab_, 1, nullptr));
    }
    status_check(expert::runtime::cuda::argmax_batch(
        logits_, vocab_, head_count, output_token_, nullptr));
    cuda_check(cudaMemcpy(result.data(), output_token_,
                          result.size() * sizeof(result[0]),
                          cudaMemcpyDeviceToHost),
               "copy generated tokens");
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      float milliseconds = 0.0F;
      cuda_check(cudaEventElapsedTime(&milliseconds,
                                      expert_start_events_[layer],
                                      expert_done_events_[layer]),
                 "measure GPU expert lane");
      phase_.gpu_expert_ns +=
          static_cast<std::uint64_t>(milliseconds * 1'000'000.0F);
      dispatch_->observe_gpu(
          static_cast<std::uint64_t>(milliseconds * 1'000'000.0F),
          gpu_selections_by_layer_[layer]);
    }
    const auto cpu_expert_delta = phase_.cpu_expert_ns - cpu_expert_before;
    const auto gpu_expert_delta = phase_.gpu_expert_ns - gpu_expert_before;
    const auto expert_compute_delta =
        phase_.expert_compute_ns - expert_compute_before;
    if (cpu_expert_delta + gpu_expert_delta > expert_compute_delta) {
      phase_.cpu_gpu_overlap_ns +=
          cpu_expert_delta + gpu_expert_delta - expert_compute_delta;
    }
    phase_.final_head_ns += elapsed_ns(final_head_started);
    ++phase_.forward_calls;
    phase_.forward_wall_ns += elapsed_ns(forward_started);
    return result;
  }

  std::uint32_t prefill(std::uint32_t slot,
                        std::span<const std::uint32_t> prompt,
                        std::uint32_t start_position = 0) {
    if (prompt.empty()) throw std::runtime_error("empty prefill prompt");
    std::uint32_t predicted = 0;
    std::vector<std::uint32_t> positions;
    std::vector<std::uint32_t> slots;
    positions.reserve(prefill_chunk_tokens_);
    slots.reserve(prefill_chunk_tokens_);
    for (std::size_t offset = 0; offset < prompt.size();
         offset += prefill_chunk_tokens_) {
      const auto rows = static_cast<std::uint32_t>(std::min<std::size_t>(
          prefill_chunk_tokens_, prompt.size() - offset));
      positions.resize(rows);
      slots.assign(rows, slot);
      for (std::uint32_t row = 0; row < rows; ++row)
        positions[row] =
            static_cast<std::uint32_t>(start_position + offset + row);
      // Only the last row of a chunk needs the vocabulary head; the rest
      // only extend the slot's attention/recurrent state.
      const std::array<std::uint32_t, 1> head{rows - 1U};
      predicted = forward_batch(prompt.subspan(offset, rows), positions, slots,
                                true, head).back();
    }
    return predicted;
  }

  expert::runtime::TelemetrySnapshot telemetry() const {
    auto snapshot = cache_->telemetry();
    snapshot.acquire_vram_hits += directory_vram_hits_;
    return snapshot;
  }
  PhaseTelemetry phase_telemetry() const noexcept {
    auto result = phase_;
    const auto placement = placement_->telemetry();
    result.adaptive_promotions = placement.completed;
    result.useful_prefetches = placement.useful_prefetches;
    result.useful_prefetch_bytes = placement.useful_prefetch_bytes;
    result.wasted_prefetches = placement.wasted_prefetches;
    result.wasted_prefetch_bytes = placement.wasted_prefetch_bytes;
    result.stale_prefetch_cancellations = placement.stale_cancellations;
    result.prefetch_candidate_evictions = placement.candidate_evictions;
    result.prefetch_credit_rejections = placement.credit_rejections;
    return result;
  }
  expert::runtime::HybridDispatchTelemetry dispatch_telemetry() const noexcept {
    return dispatch_->telemetry();
  }
  expert::runtime::cpu::ExpertExecutorTelemetry
  cpu_executor_telemetry() const noexcept {
    return cpu_executor_->telemetry();
  }
  void settle_placement() {
    status_check(placement_->quiesce(std::chrono::seconds(30)));
  }
  std::uint64_t total_pack_bytes() const noexcept { return total_pack_bytes_; }
  std::uint64_t dense_read_bytes() const noexcept { return dense_read_bytes_; }
  std::uint32_t capacity() const noexcept { return capacity_; }
  const expert::runtime::ModelDescriptor& descriptor() const noexcept {
    return model_descriptor_;
  }
  std::uint32_t prefill_chunk_tokens() const noexcept {
    return prefill_chunk_tokens_;
  }
  const std::string& placement_profile() const noexcept {
    return placement_profile_;
  }
  std::uint64_t ram_cache_bytes() const noexcept { return ram_cache_bytes_; }
  std::uint64_t vram_cache_bytes() const noexcept { return vram_cache_bytes_; }
  bool placement_prefetch_enabled() const noexcept {
    return placement_profile_ != "capacity";
  }
  bool placement_frozen() const noexcept { return placement_->frozen(); }
  std::uint32_t placement_minimum_observations() const noexcept {
    return placement_profile_ == "latency" ? 1U : 2U;
  }
  std::uint64_t kv_page_bytes() const noexcept { return kv_page_bytes_; }
  std::uint64_t kv_page_capacity() const noexcept { return kv_page_capacity_; }
  std::uint64_t kv_allocated_pages() const noexcept {
    return kv_allocated_pages_;
  }
  std::uint64_t kv_reserved_pages() const noexcept {
    return kv_reserved_pages_;
  }
  std::uint32_t kv_page_tokens() const noexcept { return kv_page_tokens_; }

  void enable_moe_trace(const std::filesystem::path& root,
                        std::vector<std::uint32_t> layers,
                        std::uint64_t maximum_records = 0) {
    if (moe_trace_)
      throw std::runtime_error("MoE tracing is already enabled");
    for (const auto layer : layers)
      if (layer >= layers_)
        throw std::runtime_error("MoE trace layer outside model");
    moe_trace_ = std::make_unique<MoeTraceWriter>(
        root, std::move(layers), hidden_, top_k_, maximum_records);
  }
  void set_trace_sequence_id(std::uint32_t slot, std::uint32_t sequence_id) {
    if (slot >= capacity_)
      throw std::runtime_error("trace sequence slot outside capacity");
    trace_sequence_ids_[slot] = sequence_id;
  }
  std::uint64_t moe_trace_records() const noexcept {
    return moe_trace_ ? moe_trace_->records() : 0U;
  }
  bool moe_trace_full() const noexcept {
    return moe_trace_ && moe_trace_->full();
  }
  void finalize_moe_trace() {
    if (moe_trace_) moe_trace_->finalize();
  }

  void reserve_slot(std::uint32_t slot, std::uint32_t context_tokens) {
    if (slot >= capacity_ || !context_tokens || context_tokens > max_context_)
      throw std::runtime_error("invalid slot context reservation");
    if (slot_context_limits_[slot])
      throw std::runtime_error("slot already has a context reservation");
    const auto pages = (static_cast<std::uint64_t>(context_tokens) +
                        kv_page_tokens_ - 1U) / kv_page_tokens_;
    if (pages > kv_page_capacity_ - kv_reserved_pages_)
      throw std::runtime_error("KV page credits exhausted");
    slot_kv_pages_[slot].assign(static_cast<std::size_t>(pages), nullptr);
    slot_context_limits_[slot] = context_tokens;
    kv_reserved_pages_ += pages;
    reset_slot(slot);
  }

  void release_slot(std::uint32_t slot) {
    if (slot >= capacity_) throw std::runtime_error("state slot out of range");
    auto& pages = slot_kv_pages_[slot];
    if (!slot_context_limits_[slot] && pages.empty()) return;
    cuda_check(cudaDeviceSynchronize(), "synchronize KV slot release");
    for (auto* page : pages)
      if (page) free_kv_pages_.push_back(page);
    kv_reserved_pages_ -= pages.size();
    pages.clear();
    slot_context_limits_[slot] = 0;
    cuda_check(cudaMemset(
                   device_kv_page_table_ +
                       static_cast<std::size_t>(slot) * max_kv_pages_per_slot_,
                   0, static_cast<std::size_t>(max_kv_pages_per_slot_) *
                          sizeof(void*)),
               "clear KV page table slot");
  }
  void reset_slot(std::uint32_t slot) {
    if (slot >= capacity_) throw std::runtime_error("state slot out of range");
    const auto conv_elements =
        (static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
         static_cast<std::size_t>(value_heads_) * value_head_dim_) *
        conv_kernel_;
    const auto recurrent_elements = static_cast<std::size_t>(value_heads_) *
                                    key_head_dim_ * value_head_dim_;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      if (!layer_operations_.at(layer).full_attention) {
        cuda_check(cudaMemset(conv_state_[layer] + slot * conv_elements, 0,
                              conv_elements * sizeof(float)),
                   "reset delta conv slot");
        cuda_check(cudaMemset(
                       recurrent_state_[layer] + slot * recurrent_elements, 0,
                       recurrent_elements * sizeof(float)),
                   "reset delta recurrent slot");
      }
    }
  }
  void grow_slot(std::uint32_t slot, std::uint32_t context_tokens) {
    if (slot >= capacity_ || !context_tokens || context_tokens > max_context_)
      throw std::runtime_error("invalid slot context growth");
    if (!slot_context_limits_[slot])
      throw std::runtime_error("slot has no context reservation to grow");
    if (context_tokens <= slot_context_limits_[slot]) return;
    const auto pages = (static_cast<std::uint64_t>(context_tokens) +
                        kv_page_tokens_ - 1U) / kv_page_tokens_;
    const auto current =
        static_cast<std::uint64_t>(slot_kv_pages_[slot].size());
    if (pages - current > kv_page_capacity_ - kv_reserved_pages_)
      throw std::runtime_error("KV page credits exhausted");
    slot_kv_pages_[slot].resize(static_cast<std::size_t>(pages), nullptr);
    kv_reserved_pages_ += pages - current;
    slot_context_limits_[slot] = context_tokens;
  }

  void reset_request() {
    const auto conv_elements =
        (static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
         static_cast<std::size_t>(value_heads_) * value_head_dim_) *
        conv_kernel_;
    const auto recurrent_elements = static_cast<std::size_t>(value_heads_) *
                                    key_head_dim_ * value_head_dim_;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      if (!layer_operations_.at(layer).full_attention) {
        cuda_check(cudaMemset(conv_state_[layer], 0,
                              capacity_ * conv_elements * sizeof(float)),
                   "reset delta conv state");
        cuda_check(cudaMemset(recurrent_state_[layer], 0,
                              capacity_ * recurrent_elements * sizeof(float)),
                   "reset delta recurrent state");
      }
    }
  }

 private:
  static constexpr std::uint32_t kFullAttention = 0U;
  static constexpr std::uint32_t kDeltaAttention = 1U;
  static constexpr std::uint32_t kRouter = 2U;
  static constexpr std::uint32_t kRoutedMoe = 3U;
  static constexpr std::uint32_t kEmbedding = 4U;
  static constexpr std::uint32_t kHead = 5U;

  template <typename T>
  expert::runtime::ExecutionValue device_value(
      const T* pointer, std::uint64_t elements,
      std::string_view abi = "batch.hidden.f32.cuda.v1") const {
    return {std::string(abi), "cuda.device", workspace_lifetime_,
            reinterpret_cast<const std::byte*>(pointer),
            elements * sizeof(T)};
  }

  static expert::runtime::OperationExecutionHandle completed_operation(
      expert::runtime::OperationExecutionResult result) {
    struct State final {
      expert::runtime::OperationExecutionResult result;
      bool terminal{};
    };
    auto state = std::make_shared<State>();
    state->result = std::move(result);
    return expert::runtime::OperationExecutionHandle::from_callbacks(
        [state]() -> std::optional<expert::runtime::OperationExecutionResult> {
          if (state->terminal) return std::nullopt;
          state->terminal = true;
          return std::move(state->result);
        },
        [state] { state->terminal = true; });
  }

  std::optional<std::uint32_t> acquire_callable_provider_slot() {
    std::lock_guard lock(provider_slot_mutex_);
    const auto found = std::find(provider_slots_.begin(),
                                 provider_slots_.end(), false);
    if (found == provider_slots_.end()) return std::nullopt;
    *found = true;
    return static_cast<std::uint32_t>(found - provider_slots_.begin());
  }

  void release_callable_provider_slot(std::uint32_t slot) noexcept {
    try {
      release_slot(slot);
    } catch (...) {
    }
    std::lock_guard lock(provider_slot_mutex_);
    if (slot < provider_slots_.size()) provider_slots_[slot] = false;
  }

  std::uint32_t descriptor_u32(std::string_view key) const {
    const auto found = model_descriptor_.attributes.find(key);
    if (found == model_descriptor_.attributes.end() ||
        found->second > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("missing or invalid model parameter " +
                               std::string(key));
    return static_cast<std::uint32_t>(found->second);
  }
  float descriptor_f32(std::string_view key) const {
    return std::bit_cast<float>(descriptor_u32(key));
  }
  const std::string& model_binding(std::string_view role) const {
    const auto found = model_descriptor_.tensor_bindings.find(role);
    if (found == model_descriptor_.tensor_bindings.end())
      throw std::runtime_error("missing model tensor role " +
                               std::string(role));
    return found->second;
  }
  static const std::string& operation_binding(
      const expert::runtime::CompiledOperationProgram& operation,
      std::string_view role) {
    const auto found = operation.tensor_bindings.find(role);
    if (found == operation.tensor_bindings.end())
      throw std::runtime_error("missing operation tensor role " +
                               std::string(role));
    return found->second;
  }
  const std::string& operation_capability(
      const expert::runtime::CompiledOperationProgram& operation) const {
    const auto& binding = compiled_program_.kernels.at(operation.kernel_binding);
    return model_descriptor_.required_kernels.at(binding.requirement_index)
        .capability;
  }
  const Tensor& tensor(std::string_view name) const {
    const auto found = tensors_.find(std::string(name));
    if (found == tensors_.end())
      throw std::runtime_error("tensor binding is absent from pack: " +
                               std::string(name));
    return found->second;
  }
  void expect_tensor(std::string_view name,
                     std::initializer_list<std::uint32_t> shape,
                     bool quantized) const {
    const auto& item = tensor(name);
    if (item.quantized != quantized ||
        item.shape != std::vector<std::uint32_t>(shape))
      throw std::runtime_error(
          "tensor binding has incompatible dtype/shape: " +
          std::string(name));
  }
  void initialize_program_contract() {
    expect_tensor(model_binding("token_embedding"), {vocab_, hidden_}, true);
    expect_tensor(model_binding("final_norm"), {hidden_}, false);
    expect_tensor(model_binding("output_head"), {vocab_, hidden_}, true);
    layer_operations_.assign(layers_, {});
    bool embedding_operation = false;
    bool head_operation = false;
    for (std::size_t index = 0; index < compiled_program_.operations.size();
         ++index) {
      const auto& operation = compiled_program_.operations[index];
      const auto& capability = operation_capability(operation);
      if (operation.logical_layer ==
          expert::runtime::kModelLevelOperationLayer) {
        if (capability == "embedding.lookup.int8-row.v1") {
          if (embedding_operation)
            throw std::runtime_error("duplicate model embedding operation");
          embedding_operation = true;
          expect_tensor(operation_binding(operation, "weight"),
                        {vocab_, hidden_}, true);
        } else if (capability == "head.rmsnorm.argmax.int8-row.v1") {
          if (head_operation)
            throw std::runtime_error("duplicate model output-head operation");
          head_operation = true;
          expect_tensor(operation_binding(operation, "norm"), {hidden_},
                        false);
          expect_tensor(operation_binding(operation, "weight"),
                        {vocab_, hidden_}, true);
        } else {
          throw std::runtime_error(
              "unsupported model-level operation capability");
        }
        continue;
      }
      if (operation.logical_layer >= layers_)
        throw std::runtime_error("compiled operation layer is out of range");
      auto& layer = layer_operations_[operation.logical_layer];
      const auto bind = [&](std::uint32_t& slot) {
        if (slot != kNoOperation)
          throw std::runtime_error("duplicate operation role in logical layer");
        slot = static_cast<std::uint32_t>(index);
      };
      if (capability == "block.full-attention.output-gated.v1") {
        bind(layer.block_operation);
        layer.full_attention = true;
        layer.full_attention_slot = full_attention_layers_++;
        expect_tensor(operation_binding(operation, "input_norm"), {hidden_},
                      false);
        expect_tensor(operation_binding(operation, "query_projection"),
                      {2U * query_heads_ * head_dim_, hidden_}, true);
        expect_tensor(operation_binding(operation, "key_projection"),
                      {kv_heads_ * head_dim_, hidden_}, true);
        expect_tensor(operation_binding(operation, "value_projection"),
                      {kv_heads_ * head_dim_, hidden_}, true);
        expect_tensor(operation_binding(operation, "output_projection"),
                      {hidden_, query_heads_ * head_dim_}, true);
        expect_tensor(operation_binding(operation, "query_norm"), {head_dim_},
                      false);
        expect_tensor(operation_binding(operation, "key_norm"), {head_dim_},
                      false);
      } else if (capability ==
                 "block.recurrent-linear-attention.gated-delta.v1") {
        bind(layer.block_operation);
        const auto key_dim = key_heads_ * key_head_dim_;
        const auto value_dim = value_heads_ * value_head_dim_;
        expect_tensor(operation_binding(operation, "input_norm"), {hidden_},
                      false);
        expect_tensor(operation_binding(operation, "qkvz_projection"),
                      {2U * key_dim + 2U * value_dim, hidden_}, true);
        expect_tensor(operation_binding(operation, "ba_projection"),
                      {2U * value_heads_, hidden_}, true);
        expect_tensor(operation_binding(operation, "convolution"),
                      {2U * key_dim + value_dim, 1U, conv_kernel_}, false);
        expect_tensor(operation_binding(operation, "time_bias"), {value_heads_},
                      false);
        expect_tensor(operation_binding(operation, "decay_log"), {value_heads_},
                      false);
        expect_tensor(operation_binding(operation, "output_norm"),
                      {value_head_dim_}, false);
        expect_tensor(operation_binding(operation, "output_projection"),
                      {hidden_, value_dim}, true);
      } else if (capability == "router.linear-topk.shared-swiglu.v1") {
        bind(layer.router_operation);
        if (operation.routed_component_index != 0U)
          throw std::runtime_error("router binds the wrong routed component");
        expect_tensor(operation_binding(operation, "input_norm"), {hidden_},
                      false);
        expect_tensor(operation_binding(operation, "router_weight"),
                      {experts_, hidden_}, false);
        expect_tensor(operation_binding(operation, "shared_gate_projection"),
                      {shared_width_, hidden_}, true);
        expect_tensor(operation_binding(operation, "shared_up_projection"),
                      {shared_width_, hidden_}, true);
        expect_tensor(operation_binding(operation, "shared_down_projection"),
                      {hidden_, shared_width_}, true);
        expect_tensor(operation_binding(operation, "shared_router"),
                      {1U, hidden_}, true);
      } else if (capability == "moe.swiglu.routed.merge-shared.v1") {
        bind(layer.routed_operation);
        if (operation.routed_component_index != 0U)
          throw std::runtime_error(
              "routed execution binds the wrong component");
      } else {
        throw std::runtime_error(
            "hybrid-delta provider compiled an unknown operation");
      }
    }
    if (!embedding_operation || !head_operation)
      throw std::runtime_error("model-level operation program is incomplete");
    if (full_attention_layers_ == 0U)
      throw std::runtime_error("operation program has no full-attention layer");
    for (const auto& layer : layer_operations_)
      if (layer.block_operation == kNoOperation ||
          layer.router_operation == kNoOperation ||
          layer.routed_operation == kNoOperation)
        throw std::runtime_error("logical layer operation program is incomplete");
  }
  const expert::runtime::cuda::Int8Matrix& matrix(const std::string& name) const {
    const auto& tensor = tensors_.at(name);
    if (!tensor.quantized) throw std::runtime_error(name + " is not INT8");
    return tensor.int8;
  }
  const float* fp32(const std::string& name) const {
    const auto& tensor = tensors_.at(name);
    if (tensor.quantized) throw std::runtime_error(name + " is not FP32");
    return tensor.f32;
  }
  void add_tensor(const expert::runtime::ArtifactDenseTensor& entry) {
    Tensor tensor;
    tensor.shape = entry.shape;
    const auto pack = dense_packs_.find(entry.pack);
    if (pack == dense_packs_.end())
      throw std::runtime_error("tensor references a non-resident dense pack");
    auto* data_pointer = pack->second.base + entry.record_offset +
                         entry.data_offset;
    tensor.quantized = entry.encoding == "I8";
    if (tensor.quantized) {
      tensor.int8 = {
          reinterpret_cast<const std::int8_t*>(data_pointer),
          reinterpret_cast<const float*>(
              pack->second.base + entry.record_offset + entry.scale_offset),
          tensor.shape.at(0), tensor.shape.at(1)};
    } else {
      tensor.f32 = reinterpret_cast<const float*>(data_pointer);
    }
    if (!tensors_.emplace(entry.name, std::move(tensor)).second)
      throw std::runtime_error("duplicate dense tensor " + entry.name);
    if (!tensor_entries_.emplace(entry.name, entry).second)
      throw std::runtime_error("duplicate dense tensor metadata " +
                               entry.name);
  }

  const expert::runtime::PayloadRecord& expert_record(
      std::uint32_t layer, std::uint32_t expert) const {
    const auto* record = catalog_->find(layer, expert);
    if (record == nullptr) throw std::runtime_error("expert is absent from catalog");
    return *record;
  }
  void allocate_workspace() {
    const auto query_size = static_cast<std::size_t>(2U) * query_heads_ * head_dim_;
    const auto key_value_size = static_cast<std::size_t>(kv_heads_) * head_dim_;
    const auto projected_size =
        static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
        static_cast<std::size_t>(2U) * value_heads_ * value_head_dim_;
    const auto conv_size = static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
                           static_cast<std::size_t>(value_heads_) * value_head_dim_;
    const auto rows = workspace_rows_;
    hidden_state_ = device_allocate<float>(rows * hidden_);
    normalized_ = device_allocate<float>(rows * hidden_);
    residual_ = device_allocate<float>(rows * hidden_);
    query_gate_ = device_allocate<float>(rows * query_size);
    key_ = device_allocate<float>(rows * key_value_size);
    value_ = device_allocate<float>(rows * key_value_size);
    attention_ = device_allocate<float>(rows * query_heads_ * head_dim_);
    projected_qkvz_ = device_allocate<float>(rows * projected_size);
    projected_ba_ = device_allocate<float>(rows * 2U * value_heads_);
    delta_output_ =
        device_allocate<float>(rows * value_heads_ * value_head_dim_);
    conv_output_ = device_allocate<float>(rows * conv_size);
    shared_gate_ = device_allocate<float>(rows * shared_width_);
    shared_up_ = device_allocate<float>(rows * shared_width_);
    shared_intermediate_ = device_allocate<float>(rows * shared_width_);
    shared_output_ = device_allocate<float>(rows * hidden_);
    shared_scalar_ = device_allocate<float>(rows);
    router_logits_ = device_allocate<float>(rows * experts_);
    routing_scores_ = device_allocate<float>(rows * top_k_);
    routing_indices_ =
        device_allocate<std::uint32_t>(rows * top_k_);
    moe_intermediate_ = device_allocate<float>(
        static_cast<std::size_t>(rows) * top_k_ * expert_width_);
    moe_selection_output_ = device_allocate<float>(
        static_cast<std::size_t>(rows) * top_k_ * hidden_);
    cpu_selection_output_device_ = device_allocate<float>(
        static_cast<std::size_t>(rows) * top_k_ * hidden_);
    gpu_selection_mask_ =
        device_allocate<std::uint8_t>(rows * top_k_);
    cpu_slot_by_selection_ =
        device_allocate<std::uint32_t>(rows * top_k_);
    moe_output_ = device_allocate<float>(rows * hidden_);
    moe_q8_input_ = device_allocate<std::int8_t>(rows * hidden_);
    moe_q8_input_scales_ = device_allocate<float>(rows);
    moe_q8_intermediate_ = device_allocate<std::int8_t>(
        static_cast<std::size_t>(rows) * top_k_ * expert_width_);
    moe_q8_intermediate_scales_ =
        device_allocate<float>(static_cast<std::size_t>(rows) * top_k_);
    // The vocabulary head only ever runs for decode rows or one prefill row,
    // so logits stay sized by the decode batch capacity.
    logits_ = device_allocate<float>(capacity_ * vocab_);
    output_token_ = device_allocate<std::uint32_t>(capacity_);
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_normalized_),
                   static_cast<std::size_t>(rows) * hidden_ * sizeof(float),
                   cudaHostAllocDefault),
               "cudaHostAlloc CPU expert input");
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_routing_indices_),
                   static_cast<std::size_t>(rows) * top_k_ *
                       sizeof(std::uint32_t),
                   cudaHostAllocDefault),
               "cudaHostAlloc CPU route indices");
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_routing_scores_),
                   static_cast<std::size_t>(rows) * top_k_ *
                       sizeof(float),
                   cudaHostAllocDefault),
               "cudaHostAlloc CPU route scores");
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_cpu_selection_output_),
                   static_cast<std::size_t>(rows) * top_k_ * hidden_ *
                       sizeof(float),
                   cudaHostAllocDefault),
               "cudaHostAlloc CPU expert output");
    conv_state_.resize(layers_);
    recurrent_state_.resize(layers_);
    max_kv_pages_per_slot_ =
        (max_context_ + kv_page_tokens_ - 1U) / kv_page_tokens_;
    const auto page_elements = static_cast<std::uint64_t>(kv_page_tokens_) *
                               kv_heads_ * head_dim_;
    kv_page_bytes_ = static_cast<std::uint64_t>(full_attention_layers_) * 2U *
                     page_elements * sizeof(std::uint16_t);
    kv_page_capacity_ = std::min<std::uint64_t>(
        kv_cache_bytes_ / kv_page_bytes_,
        static_cast<std::uint64_t>(capacity_) * max_kv_pages_per_slot_);
    if (!kv_page_capacity_)
      throw std::runtime_error("KV cache budget fits no page");
    device_kv_page_table_ = device_allocate<void*>(
        static_cast<std::size_t>(capacity_) * max_kv_pages_per_slot_);
    cuda_check(cudaMemset(
                   device_kv_page_table_, 0,
                   static_cast<std::size_t>(capacity_) *
                       max_kv_pages_per_slot_ * sizeof(void*)),
               "zero KV page table");
    slot_kv_pages_.resize(capacity_);
    slot_context_limits_.resize(capacity_);
    trace_sequence_ids_.resize(capacity_);
    const auto conv_elements = conv_size * conv_kernel_;
    const auto recurrent_elements = static_cast<std::size_t>(value_heads_) *
                                    key_head_dim_ * value_head_dim_;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      if (!layer_operations_.at(layer).full_attention) {
        conv_state_[layer] =
            device_allocate<float>(capacity_ * conv_elements);
        recurrent_state_[layer] =
            device_allocate<float>(capacity_ * recurrent_elements);
        cuda_check(cudaMemset(conv_state_[layer], 0,
                              capacity_ * conv_elements * sizeof(float)),
                   "zero delta conv state");
        cuda_check(cudaMemset(recurrent_state_[layer], 0,
                              capacity_ * recurrent_elements * sizeof(float)),
                   "zero delta recurrent state");
      }
    }
  }
  void run_full_attention(
                          const expert::runtime::CompiledOperationProgram& operation,
                          std::uint32_t full_attention_layer,
                          std::span<const std::uint32_t> positions,
                          std::span<const std::uint32_t> state_slots,
                          std::uint32_t rows) {
    const auto query_size = 2U * query_heads_ * head_dim_;
    const auto kv_size = kv_heads_ * head_dim_;
    const auto attention_size = query_heads_ * head_dim_;
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(operation, "query_projection")), normalized_,
        query_gate_,
        rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(operation, "key_projection")), normalized_,
        key_, rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(operation, "value_projection")), normalized_,
        value_, rows, nullptr));
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto page_index = positions[row] / kv_page_tokens_;
      auto* page = slot_kv_pages_[state_slots[row]][page_index];
      status_check(expert::runtime::cuda::qwen3_next_qkv_rope_cache_paged_fp16(
          query_gate_ + static_cast<std::size_t>(row) * query_size,
          key_ + static_cast<std::size_t>(row) * kv_size,
          value_ + static_cast<std::size_t>(row) * kv_size,
          fp32(operation_binding(operation, "query_norm")),
          fp32(operation_binding(operation, "key_norm")),
          page, full_attention_layer, kv_page_tokens_, positions[row],
          query_heads_, kv_heads_, head_dim_, rotary_dim_, epsilon_,
          rope_theta_, nullptr));
      const auto* table = reinterpret_cast<const void* const*>(
          device_kv_page_table_ + static_cast<std::size_t>(state_slots[row]) *
                                      max_kv_pages_per_slot_);
      status_check(
          expert::runtime::cuda::qwen3_next_attention_decode_paged_fp16(
          query_gate_ + static_cast<std::size_t>(row) * query_size,
          table,
          attention_ + static_cast<std::size_t>(row) * attention_size,
          positions[row] + 1U, full_attention_layer, kv_page_tokens_,
          query_heads_, kv_heads_, head_dim_, nullptr));
    }
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(operation, "output_projection")), attention_,
        residual_, rows, nullptr));
  }

  void ensure_kv_page(std::uint32_t slot, std::uint32_t position) {
    const auto page_index = position / kv_page_tokens_;
    auto& page = slot_kv_pages_[slot].at(page_index);
    if (page) return;
    if (!free_kv_pages_.empty()) {
      page = free_kv_pages_.back();
      free_kv_pages_.pop_back();
    } else {
      if (kv_allocated_pages_ >= kv_page_capacity_)
        throw std::runtime_error("KV physical page capacity exhausted");
      cuda_check(cudaMalloc(&page, static_cast<std::size_t>(kv_page_bytes_)),
                 "allocate KV page");
      ++kv_allocated_pages_;
    }
    cuda_check(cudaMemcpy(
                   device_kv_page_table_ +
                       static_cast<std::size_t>(slot) * max_kv_pages_per_slot_ +
                       page_index,
                   &page, sizeof(page), cudaMemcpyHostToDevice),
               "publish KV page");
  }

  void run_delta(const expert::runtime::CompiledOperationProgram& operation,
                 std::uint32_t layer,
                 std::span<const std::uint32_t> state_slots,
                 std::uint32_t rows) {
    const auto projected_size = 2U * key_heads_ * key_head_dim_ +
                                2U * value_heads_ * value_head_dim_;
    const auto ba_size = 2U * value_heads_;
    const auto delta_size = value_heads_ * value_head_dim_;
    const auto conv_size = 2U * key_heads_ * key_head_dim_ + delta_size;
    const auto conv_state_size = conv_size * conv_kernel_;
    const auto recurrent_size = value_heads_ * key_head_dim_ * value_head_dim_;
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(operation, "qkvz_projection")), normalized_,
        projected_qkvz_, rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(operation, "ba_projection")), normalized_,
        projected_ba_, rows, nullptr));
    for (std::uint32_t row = 0; row < rows; ++row)
      status_check(expert::runtime::cuda::qwen3_next_delta_decode({
          projected_qkvz_ + static_cast<std::size_t>(row) * projected_size,
          projected_ba_ + static_cast<std::size_t>(row) * ba_size,
          fp32(operation_binding(operation, "convolution")),
          fp32(operation_binding(operation, "time_bias")),
          fp32(operation_binding(operation, "decay_log")),
          fp32(operation_binding(operation, "output_norm")),
          conv_state_[layer] +
              static_cast<std::size_t>(state_slots[row]) * conv_state_size,
          recurrent_state_[layer] +
              static_cast<std::size_t>(state_slots[row]) * recurrent_size,
          conv_output_ + static_cast<std::size_t>(row) * conv_size,
          delta_output_ + static_cast<std::size_t>(row) * delta_size,
          key_heads_, value_heads_, key_head_dim_, value_head_dim_,
          conv_kernel_, epsilon_, nullptr}));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(operation, "output_projection")), delta_output_,
        residual_, rows, nullptr));
  }

  void run_router(
      const expert::runtime::CompiledOperationProgram& router,
      std::uint32_t rows) {
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(router, "shared_gate_projection")), normalized_,
        shared_gate_, rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(router, "shared_up_projection")), normalized_,
        shared_up_, rows, nullptr));
    status_check(expert::runtime::cuda::silu_product(
        shared_gate_, shared_up_, shared_intermediate_, rows * shared_width_,
        nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(router, "shared_down_projection")),
        shared_intermediate_, shared_output_, rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(operation_binding(router, "shared_router")), normalized_,
        shared_scalar_, rows, nullptr));
    for (std::uint32_t row = 0; row < rows; ++row)
      status_check(expert::runtime::cuda::sigmoid_scale_in_place(
          shared_output_ + static_cast<std::size_t>(row) * hidden_,
          shared_scalar_ + row, hidden_, nullptr));
    cuda_check(cudaEventRecord(shared_done_event_), "record shared expert done");
    status_check(expert::runtime::cuda::router_topk_normalized_batch(
        normalized_, fp32(operation_binding(router, "router_weight")), rows,
        hidden_, experts_,
        top_k_, router_logits_, routing_scores_, routing_indices_, nullptr));
    cuda_check(cudaEventRecord(router_done_event_), "record router done");
  }

  void run_routed_moe(
      const expert::runtime::CompiledOperationProgram& routed,
      std::span<const std::uint32_t> positions,
      std::span<const std::uint32_t> state_slots,
      std::uint32_t rows) {
    const auto logical_layer = routed.logical_layer;
    const auto layer = routed.component_layer;
    if (logical_layer >= layer_operations_.size() ||
        layer_operations_[logical_layer].routed_operation !=
            routed.logical_operation)
      throw std::runtime_error(
          "routed operation disagrees with the artifact layer program");

    // Launch the directory plan on the same stream right behind the router;
    // its results are consumed through a completion event instead of a
    // stream-wide host synchronization.
    struct PlanGuard final {
      expert::runtime::cuda::CudaExpertDirectory* directory{};
      expert::runtime::cuda::CudaDirectoryPlanWorkspace* workspace{};
      bool active{};
      ~PlanGuard() {
        if (active && directory != nullptr && workspace != nullptr)
          static_cast<void>(directory->cancel_plan_async(*workspace));
      }
    } plan_guard{directory_.get(), plan_workspace_.get(), false};
    status_check(directory_->begin_plan_async(
        *plan_workspace_, layer, routing_indices_, rows * top_k_, nullptr,
        true));
    plan_guard.active = true;

    const auto cache_started = std::chrono::steady_clock::now();
    std::vector<expert::runtime::ExpertLease> leases;
    std::vector<expert::runtime::HostExpertLease> host_leases;
    std::vector<expert::runtime::cpu::ExpertWorkGroup> cpu_groups;
    std::vector<std::uint32_t> cpu_experts;
    std::vector<std::uint32_t> cpu_group_expert;
    if (!placement_->frozen()) placement_->poll();
    std::uint64_t route_pin_id = 0U;
    struct PinGuard final {
      expert::runtime::cuda::CudaExpertDirectory* directory{};
      std::uint64_t* pin_id{};
      ~PinGuard() {
        if (directory != nullptr && pin_id != nullptr && *pin_id != 0U) {
          static_cast<void>(directory->release_pins(*pin_id, nullptr));
        }
      }
    } pin_guard{directory_.get(), &route_pin_id};
    bool split_execution = false;
    std::uint32_t compact_cpu_selection_count = 0;
    status_check(directory_->wait_plan_async(*plan_workspace_));
    auto polled_plan = directory_->poll_plan_async(*plan_workspace_);
    status_check(polled_plan.status);
    if (!polled_plan.complete)
      throw std::runtime_error("directory plan incomplete after wait");
    plan_guard.active = false;
    auto plan = std::move(polled_plan.plan);
    status_check(plan.status);
    route_pin_id = plan.pin_id;
    routed_expert_keys_.clear();
    missing_expert_keys_.clear();
    for (const auto expert : plan.ready_experts)
      routed_expert_keys_.push_back(routed_->key(layer, expert));
    for (const auto expert : plan.missing_experts)
      missing_expert_keys_.push_back(routed_->key(layer, expert));
    placement_->observe_routes(routed_expert_keys_, missing_expert_keys_);
    const auto selection_count = rows * top_k_;
    const bool placement_feedback = !placement_->frozen();
    // W4: once placement is frozen the per-selection D2H copies stay retired,
    // but the asynchronous directory plan already carries the exact route to
    // the host. Keep feeding the cache LFU from it so post-freeze placement
    // (eviction victims and re-promotion admission) tracks the observed
    // routing instead of the warmup snapshot.
    const bool frozen_route_feedback = placement_->frozen();
    if (!plan.missing_experts.empty() || placement_feedback) {
      cuda_check(cudaMemcpy(host_routing_indices_, routing_indices_,
                            static_cast<std::size_t>(selection_count) *
                                sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "copy expert route to host");
    }
    if (placement_feedback) {
      cuda_check(cudaMemcpy(host_routing_scores_, routing_scores_,
                            static_cast<std::size_t>(selection_count) *
                                sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy expert routing scores to host");
    }
    if (!plan.missing_experts.empty() || placement_feedback) {
      std::fill(route_access_counts_.begin(), route_access_counts_.end(), 0U);
      if (placement_feedback) {
        std::fill(route_score_sums_.begin(), route_score_sums_.end(), 0.0);
        std::fill(route_score_maxima_.begin(), route_score_maxima_.end(), 0.0);
      }
      for (std::uint32_t selection = 0; selection < selection_count;
           ++selection) {
        const auto expert = host_routing_indices_[selection];
        if (expert >= experts_)
          throw std::runtime_error("router expert out of range");
        ++route_access_counts_[expert];
        if (placement_feedback) {
          const auto score = static_cast<double>(host_routing_scores_[selection]);
          route_score_sums_[expert] += score;
          route_score_maxima_[expert] =
              std::max(route_score_maxima_[expert], score);
        }
      }
    } else if (frozen_route_feedback) {
      // Fully resident frozen layer: no host route copy was needed, so count
      // the plan's host-side selection list instead.
      std::fill(route_access_counts_.begin(), route_access_counts_.end(), 0U);
      for (const auto expert : plan.selected_experts) {
        if (expert >= experts_)
          throw std::runtime_error("router expert out of range");
        ++route_access_counts_[expert];
      }
    }
    if (placement_feedback || frozen_route_feedback) {
      route_accesses_.clear();
      for (std::uint32_t expert = 0; expert < experts_; ++expert) {
        if (route_access_counts_[expert] != 0) {
          // Routing scores ride the pre-freeze D2H copy only; frozen feedback
          // carries access counts, and the cache ignores non-positive scores.
          route_accesses_.push_back(
              {routed_->key(layer, expert),
               route_access_counts_[expert],
               placement_feedback ? route_score_sums_[expert] : 0.0,
               placement_feedback ? route_score_maxima_[expert] : 0.0});
        }
      }
      static_cast<void>(cache_->record_accesses(route_accesses_));
    }
    if (plan.missing_experts.empty()) {
      directory_vram_hits_ += plan.unique_experts;
      if (route_pin_id == 0U)
        throw std::runtime_error("directory returned no route pin");
    } else {
      split_execution = true;
      if (route_pin_id == 0U)
        throw std::runtime_error("directory returned no miss-route pin");
      const auto acquire_device = [&](std::span<const std::uint32_t> experts) {
        std::vector<expert::runtime::AcquireHandle> handles;
        handles.reserve(experts.size());
        for (const auto expert : experts) {
          if (expert >= experts_)
            throw std::runtime_error("router expert out of range");
          const auto& record = expert_record(layer, expert);
          handles.push_back(
              cache_->acquire(routed_->key(layer, expert), record));
        }
        for (std::size_t slot = 0; slot < handles.size(); ++slot) {
          if (handles[slot].wait_for(std::chrono::seconds(30)) !=
              std::future_status::ready) {
            handles[slot].cancel();
            throw std::runtime_error(
                "expert acquire timeout at layer " + std::to_string(layer) +
                ", expert " + std::to_string(experts[slot]));
          }
          auto acquired = handles[slot].get();
          if (!acquired.status.ok())
            throw std::runtime_error(
                "expert acquire failed: " +
                std::string(acquired.status.message()));
          leases.push_back(std::move(acquired.lease));
        }
      };

      std::unordered_map<std::uint32_t, std::size_t> host_slot_by_expert;
      host_leases.reserve(plan.missing_experts.size());
      for (const auto expert : plan.missing_experts) {
        if (expert >= experts_)
          throw std::runtime_error("router expert out of range");
        const auto& record = expert_record(layer, expert);
        auto host = cache_->try_acquire_host(
            routed_->key(layer, expert), record, false);
        if (host) {
          host_slot_by_expert.emplace(expert, host_leases.size());
          host_leases.push_back(std::move(*host));
        }
      }
      const auto has_cold_fallback =
          host_slot_by_expert.size() != plan.missing_experts.size();
      const auto may_upload_from_ram = placement_profile_ != "capacity";
      // W4: scan the stale-victim budget once per forward pass, ahead of the
      // admission decisions below. With no long-unrouted residents there is
      // nothing to displace, so no protection leases are needed either.
      if (frozen_route_feedback && may_upload_from_ram &&
          !frozen_stale_budget_valid_) {
        frozen_stale_budget_bytes_ =
            cache_->vram_stale_resident_bytes(kFrozenPromotionMinVictimAge);
        frozen_stale_budget_valid_ = true;
      }
      if (has_cold_fallback ||
          (may_upload_from_ram &&
           (!frozen_route_feedback || frozen_stale_budget_bytes_ > 0))) {
        // Cache references mirroring the directory pins keep current-route
        // entries hot and preferred by the admission heuristics. Eviction
        // correctness no longer depends on this lease: the cache skips
        // route-pinned victims structurally (route_pinned/try_retire).
        acquire_device(plan.ready_experts);
      }

      std::vector<expert::runtime::HybridDispatchCandidate> candidates;
      candidates.reserve(plan.ready_experts.size() +
                         plan.missing_experts.size());
      for (const auto expert : plan.ready_experts) {
        const auto& record = expert_record(layer, expert);
        candidates.push_back({expert, route_access_counts_[expert],
                              record.stored_bytes, true, false, true});
      }
      for (const auto expert : plan.missing_experts) {
        const auto& record = expert_record(layer, expert);
        candidates.push_back(
            {expert, route_access_counts_[expert], record.stored_bytes, false,
             false, true});
      }
      const auto dispatch_plan = dispatch_->plan(candidates);
      status_check(dispatch_plan.status);

      std::vector<std::uint32_t> gpu_upload_experts;
      std::uint64_t observed_upload_bytes = 0;
      bool uploads_are_ram_resident = true;
      cpu_experts.reserve(plan.missing_experts.size());
      gpu_upload_experts.reserve(plan.missing_experts.size());
      for (const auto& decision : dispatch_plan.decisions) {
        if (decision.executor == expert::runtime::HybridExecutor::cpu_local) {
          cpu_experts.push_back(decision.expert);
        } else if (decision.executor ==
                   expert::runtime::HybridExecutor::gpu_upload) {
          gpu_upload_experts.push_back(decision.expert);
          const auto host_available =
              host_slot_by_expert.contains(decision.expert);
          uploads_are_ram_resident =
              uploads_are_ram_resident && host_available;
          if (host_available) {
            observed_upload_bytes +=
                expert_record(layer, decision.expert).stored_bytes;
          }
        }
      }
      directory_vram_hits_ += plan.ready_experts.size();
      if (!gpu_upload_experts.empty()) {
        const auto upload_started = std::chrono::steady_clock::now();
        acquire_device(gpu_upload_experts);
        const auto upload_elapsed = elapsed_ns(upload_started);
        if (uploads_are_ram_resident && observed_upload_bytes != 0)
          dispatch_->observe_h2d(upload_elapsed, observed_upload_bytes);
      }

      std::unordered_map<std::uint32_t, std::size_t> cpu_group_by_expert;
      cpu_groups.reserve(cpu_experts.size());
      cpu_group_expert.clear();
      cpu_group_expert.reserve(cpu_experts.size());
      for (std::size_t index = 0; index < cpu_experts.size(); ++index) {
        const auto host_slot = host_slot_by_expert.at(cpu_experts[index]);
        cpu_group_by_expert.emplace(cpu_experts[index], index);
        cpu_group_expert.push_back(cpu_experts[index]);
        cpu_groups.push_back({host_leases[host_slot].bytes(),
                              host_leases[host_slot].sections(), {}, {}});
      }
      if (!cpu_groups.empty()) {
        cuda_check(cudaMemcpy(host_normalized_, normalized_,
                              static_cast<std::size_t>(rows) * hidden_ *
                                  sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy CPU expert activations");
      }
      std::vector<std::uint8_t> gpu_mask(selection_count, 1);
      std::vector<std::uint32_t> cpu_slot_by_selection(selection_count, 0);
      // The CPU executor blocks an expert's work group at 32 selections;
      // wide prefill chunks must spill into follow-up groups for the same
      // expert instead of violating that bound.
      constexpr std::size_t kCpuGroupSelectionLimit = 32;
      for (std::uint32_t selection = 0; selection < selection_count;
           ++selection) {
        const auto expert = host_routing_indices_[selection];
        const auto cpu_group = cpu_group_by_expert.find(expert);
        if (cpu_group != cpu_group_by_expert.end()) {
          gpu_mask[selection] = 0;
          std::size_t group_index = cpu_group->second;
          if (cpu_groups[group_index].selections.size() ==
              kCpuGroupSelectionLimit) {
            cpu_groups.push_back({cpu_groups[group_index].record_bytes,
                                  cpu_groups[group_index].sections, {}, {}});
            group_index = cpu_groups.size() - 1U;
            cpu_group->second = group_index;
            cpu_group_expert.push_back(expert);
          }
          cpu_groups[group_index].selections.push_back(selection);
          cpu_groups[group_index].output_slots.push_back(
              compact_cpu_selection_count);
          cpu_slot_by_selection[selection] = compact_cpu_selection_count++;
        }
      }
      if (compact_cpu_selection_count) {
        cuda_check(cudaMemcpy(gpu_selection_mask_, gpu_mask.data(),
                              gpu_mask.size() * sizeof(gpu_mask[0]),
                              cudaMemcpyHostToDevice),
                   "copy GPU expert selection mask");
        cuda_check(cudaMemcpy(cpu_slot_by_selection_,
                              cpu_slot_by_selection.data(),
                              cpu_slot_by_selection.size() *
                                  sizeof(cpu_slot_by_selection[0]),
                              cudaMemcpyHostToDevice),
                   "copy compact CPU selection map");
      }
    }
    const auto event_ns = [](cudaEvent_t begin, cudaEvent_t end) {
      float milliseconds = 0.0F;
      cuda_check(cudaEventElapsedTime(&milliseconds, begin, end),
                 "measure CUDA phase");
      return static_cast<std::uint64_t>(milliseconds * 1'000'000.0F);
    };
    const auto attention_ns =
        event_ns(layer_start_event_, attention_done_event_);
    const auto shared_ns = event_ns(attention_done_event_, shared_done_event_);
    const auto router_ns = event_ns(shared_done_event_, router_done_event_);
    phase_.attention_delta_ns += attention_ns;
    phase_.shared_expert_ns += shared_ns;
    phase_.router_ns += router_ns;
    phase_.shared_router_ns += shared_ns + router_ns;
    phase_.dense_router_ns += attention_ns + shared_ns + router_ns;
    phase_.expert_cache_wait_ns += elapsed_ns(cache_started);
    const auto expert_started = std::chrono::steady_clock::now();
    cuda_check(cudaEventRecord(expert_start_events_[logical_layer]),
               "record expert lane start");
    // FP4 packs always take the selection-batch path, whose kernels consume
    // the packed E2M1 payload directly through the shared dp4a GEMVs.
    split_execution = true;
    if (!split_execution) {
        status_check(expert::runtime::cuda::launch_moe_batch({
            normalized_, nullptr, nullptr, nullptr, nullptr, routing_scores_,
            routing_indices_, moe_intermediate_, moe_output_, rows, hidden_,
            expert_width_, top_k_, experts_, nullptr,
            directory_->device_entries(), layer}));
        phase_.gpu_expert_selections += selection_count;
        gpu_selections_by_layer_[logical_layer] = selection_count;
        cuda_check(cudaEventRecord(expert_done_events_[logical_layer]),
                   "record expert lane done");
      } else {
        if (route_pin_id != 0U) {
          status_check(expert::runtime::cuda::launch_moe_selection_batch({
              normalized_, routing_scores_, routing_indices_,
              compact_cpu_selection_count ? gpu_selection_mask_ : nullptr,
              moe_intermediate_, moe_selection_output_,
              moe_q8_input_, moe_q8_input_scales_, moe_q8_intermediate_,
              moe_q8_intermediate_scales_, rows, hidden_, expert_width_,
              top_k_, experts_, nullptr, directory_->device_entries(), layer,
              0.0F, false,
              true}));
        }
        phase_.gpu_expert_selections +=
            selection_count - compact_cpu_selection_count;
        gpu_selections_by_layer_[logical_layer] =
            selection_count - compact_cpu_selection_count;
        cuda_check(cudaEventRecord(expert_done_events_[logical_layer]),
                   "record expert lane done");
        if (!cpu_groups.empty()) {
          const auto cpu_started = std::chrono::steady_clock::now();
          status_check(cpu_executor_->execute(
              cpu_groups,
              std::span<const float>(host_normalized_,
                                     static_cast<std::size_t>(rows) * hidden_),
              rows, top_k_,
              std::span<float>(
                  host_cpu_selection_output_,
                  static_cast<std::size_t>(compact_cpu_selection_count) *
                      hidden_)));
          const auto cpu_elapsed = elapsed_ns(cpu_started);
          phase_.cpu_expert_ns += cpu_elapsed;
          phase_.cpu_expert_groups += cpu_groups.size();
          std::uint64_t cpu_selections = 0;
          for (const auto& group : cpu_groups)
            cpu_selections += group.selections.size();
          phase_.cpu_expert_selections += cpu_selections;
          if (!placement_->frozen())
            placement_->observe_cpu_batch(cpu_elapsed, cpu_selections);
          dispatch_->observe_cpu(cpu_elapsed, cpu_selections);
          const auto output_bytes =
              static_cast<std::size_t>(compact_cpu_selection_count) *
              hidden_ * sizeof(float);
          cuda_check(cudaMemcpyAsync(cpu_selection_output_device_,
                                     host_cpu_selection_output_, output_bytes,
                                     cudaMemcpyHostToDevice, nullptr),
                     "copy CPU expert outputs");
          phase_.cpu_result_h2d_bytes += output_bytes;
          phase_.cpu_result_map_h2d_bytes +=
              static_cast<std::uint64_t>(selection_count) *
              (sizeof(std::uint8_t) + sizeof(std::uint32_t));
        }
        status_check(expert::runtime::cuda::launch_moe_aggregate({
            moe_selection_output_,
            compact_cpu_selection_count ? cpu_selection_output_device_
                                         : nullptr,
            compact_cpu_selection_count ? gpu_selection_mask_ : nullptr,
            compact_cpu_selection_count ? cpu_slot_by_selection_ : nullptr,
            routing_scores_, moe_output_, compact_cpu_selection_count, rows,
            hidden_, top_k_, nullptr}));
    }
      status_check(expert::runtime::cuda::add_in_place(
          moe_output_, shared_output_, rows * hidden_, nullptr));
      if (moe_trace_ && moe_trace_->selected(logical_layer))
        moe_trace_->append(logical_layer, normalized_, moe_output_,
                           routing_indices_, routing_scores_, positions,
                           state_slots, trace_sequence_ids_);
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state_, moe_output_, rows * hidden_, nullptr));
      if (route_pin_id != 0U) {
        status_check(directory_->release_pins_async(route_pin_id, nullptr));
        route_pin_id = 0U;
      } else {
        cuda_check(cudaStreamSynchronize(nullptr),
                   "complete CPU-only expert layer");
      }
      if (!placement_->frozen()) {
        for (std::size_t index = 0; index < cpu_groups.size(); ++index) {
          const auto expert = cpu_group_expert.at(index);
          const auto& record = expert_record(layer, expert);
          placement_->consider(
              routed_->key(layer, expert), record,
              static_cast<std::uint32_t>(cpu_groups[index].selections.size()),
              route_score_sums_[expert]);
        }
      }
    phase_.expert_compute_ns += elapsed_ns(expert_started);
  }

  std::filesystem::path root_;
  std::uint32_t max_context_{}, capacity_{}, full_attention_layers_{};
  std::uint32_t prefill_chunk_tokens_{}, workspace_rows_{};
  std::uint32_t hidden_{}, expert_width_{}, vocab_{}, layers_{};
  std::uint32_t query_heads_{}, kv_heads_{}, head_dim_{}, experts_{}, top_k_{};
  std::uint32_t conv_kernel_{}, key_head_dim_{}, value_head_dim_{}, key_heads_{},
      value_heads_{}, shared_width_{}, rotary_dim_{}, shared_experts_{};
  float epsilon_{}, rope_theta_{};
  std::uint64_t model_id_{};
  expert::runtime::Sha256Digest model_hash_{};
  expert::runtime::ModelArtifact artifact_;
  expert::runtime::ModelDescriptor model_descriptor_;
  expert::runtime::CompiledModelProgram compiled_program_;
  std::uint32_t encoding_abi_{expert::runtime::kExpertEncodingAbiFp4Block32};
  std::uint64_t total_pack_bytes_{}, dense_read_bytes_{},
      max_expert_record_bytes_{};
  std::uint64_t ram_cache_bytes_{}, vram_cache_bytes_{}, kv_cache_bytes_{},
      kv_page_bytes_{}, kv_page_capacity_{}, kv_allocated_pages_{},
      kv_reserved_pages_{};
  std::uint32_t kv_page_tokens_{}, max_kv_pages_per_slot_{};
  std::string placement_profile_;
  std::uint64_t directory_vram_hits_{};
  // W4 frozen re-promotion budget: bytes of long-unrouted VRAM residents
  // available as displacement victims, valid for one forward pass.
  std::uint64_t frozen_stale_budget_bytes_{};
  bool frozen_stale_budget_valid_{};
  PhaseTelemetry phase_;
  std::unordered_map<std::string, DevicePack> dense_packs_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::unordered_map<std::string, expert::runtime::ArtifactDenseTensor>
      tensor_entries_;
  std::shared_ptr<const void> dense_lifetime_{
      this, [](const void*) noexcept {}};
  std::shared_ptr<const void> workspace_lifetime_{
      this, [](const void*) noexcept {}};
  std::mutex provider_slot_mutex_;
  std::vector<bool> provider_slots_;
  const expert::runtime::ExpertCatalog* catalog_{};
  std::vector<LayerOperations> layer_operations_;
  std::vector<std::uint32_t> route_access_counts_;
  std::vector<double> route_score_sums_, route_score_maxima_;
  std::vector<expert::runtime::ExpertAccess> route_accesses_;
  std::vector<expert::runtime::ExpertKey> routed_expert_keys_;
  std::vector<expert::runtime::ExpertKey> missing_expert_keys_;
  std::shared_ptr<expert::runtime::WindowsIocpStorage> storage_;
  std::shared_ptr<expert::runtime::cuda::CudaExpertUploader> uploader_;
  std::shared_ptr<expert::runtime::cuda::CudaExpertDirectory> directory_;
  std::shared_ptr<expert::runtime::cuda::CudaDirectoryPlanWorkspace>
      plan_workspace_;
  std::shared_ptr<expert::runtime::FixedBufferPool> buffers_;
  std::unique_ptr<expert::runtime::ExpertCache> cache_;
  std::unique_ptr<expert::runtime::RoutedExpertRuntime> routed_;
  std::unique_ptr<expert::runtime::cpu::ExpertExecutor> cpu_executor_;
  std::unique_ptr<expert::runtime::AdaptivePlacementPlanner> placement_;
  std::unique_ptr<expert::runtime::HybridDispatchPlanner> dispatch_;
  float *hidden_state_{}, *normalized_{}, *residual_{}, *query_gate_{}, *key_{},
      *value_{}, *attention_{}, *projected_qkvz_{}, *projected_ba_{},
      *delta_output_{}, *conv_output_{}, *shared_gate_{}, *shared_up_{},
      *shared_intermediate_{}, *shared_output_{}, *shared_scalar_{},
      *router_logits_{}, *routing_scores_{}, *moe_intermediate_{},
      *moe_selection_output_{}, *cpu_selection_output_device_{},
      *moe_output_{}, *logits_{}, *host_normalized_{}, *host_routing_scores_{},
      *host_cpu_selection_output_{};
  std::uint32_t *routing_indices_{}, *output_token_{}, *host_routing_indices_{},
      *cpu_slot_by_selection_{};
  std::uint8_t* gpu_selection_mask_{};
  // FP4 block-32 packs run the dp4a GEMVs, which read q8-quantized
  // activations; these buffers hold that transient representation.
  std::int8_t *moe_q8_input_{}, *moe_q8_intermediate_{};
  float *moe_q8_input_scales_{}, *moe_q8_intermediate_scales_{};
  void** device_kv_page_table_{};
  std::vector<std::vector<void*>> slot_kv_pages_;
  std::vector<void*> free_kv_pages_;
  std::vector<std::uint32_t> slot_context_limits_, trace_sequence_ids_;
  std::vector<float*> conv_state_, recurrent_state_;
  std::vector<std::uint32_t> gpu_selections_by_layer_;
  std::vector<cudaEvent_t> expert_start_events_, expert_done_events_;
  cudaEvent_t layer_start_event_{}, attention_done_event_{},
      shared_done_event_{}, router_done_event_{};
  std::unique_ptr<MoeTraceWriter> moe_trace_;
};

std::vector<std::uint32_t> parse_tokens(std::string_view text) {
  std::vector<std::uint32_t> result;
  while (!text.empty()) {
    const auto comma = text.find(',');
    result.push_back(static_cast<std::uint32_t>(
        std::stoul(std::string(text.substr(0, comma)))));
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
  }
  if (result.empty()) throw std::runtime_error("empty prompt");
  return result;
}

std::vector<std::vector<std::uint32_t>> parse_prompt_batch(
    std::string_view text) {
  std::vector<std::vector<std::uint32_t>> result;
  while (true) {
    const auto separator = text.find(';');
    const auto prompt = text.substr(0, separator);
    if (prompt.empty()) throw std::runtime_error("empty prompt in batch");
    result.push_back(parse_tokens(prompt));
    if (separator == std::string_view::npos) break;
    text.remove_prefix(separator + 1U);
  }
  return result;
}

std::vector<std::vector<std::uint32_t>> read_prompt_file(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open prompt file " + path.string());
  std::vector<std::vector<std::uint32_t>> prompts;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.front() == '#') continue;
    prompts.push_back(parse_tokens(line));
  }
  if (prompts.empty()) throw std::runtime_error("prompt file contains no prompts");
  return prompts;
}

std::vector<std::uint32_t> parse_layers(std::string_view text) {
  auto layers = parse_tokens(text);
  std::sort(layers.begin(), layers.end());
  if (std::adjacent_find(layers.begin(), layers.end()) != layers.end())
    throw std::runtime_error("duplicate trace layer");
  return layers;
}

double percentile_ms(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size()))) - 1U;
  return values[std::min(index, values.size() - 1U)];
}

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> fields;
  while (true) {
    const auto tab = line.find('\t');
    fields.push_back(line.substr(0, tab));
    if (tab == std::string_view::npos) break;
    line.remove_prefix(tab + 1U);
  }
  return fields;
}

struct WorkerRequest final {
  std::uint32_t slot{};
  std::uint32_t predicted{};
  std::uint32_t next_position{};
};

struct RetainedRequest final {
  std::uint32_t slot{};
  std::uint32_t tokens{};
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

int worker_loop(Qwen3NextModel& model, std::uint32_t settle_after_steps) {
  std::unordered_map<std::uint64_t, WorkerRequest> active;
  std::unordered_map<std::uint64_t, RetainedRequest> retained;
  std::vector<bool> used_slots(model.capacity());
  // Decode-step counter driving the one-time placement freeze. Once the
  // warmup boundary is reached the planner is quiesced and frozen, which
  // retires the per-layer routing-feedback D2H copies on the hot path.
  std::uint64_t decode_steps = 0;
  bool placement_settled = false;
  const auto maybe_settle_placement = [&]() {
    if (placement_settled || settle_after_steps == 0U) return;
    if (++decode_steps < settle_after_steps) return;
    model.settle_placement();
    placement_settled = true;
    std::cerr << "worker placement settled after " << decode_steps
              << " decode steps\n";
  };
  const auto& descriptor = model.descriptor();
  const auto& routed_component = descriptor.routed_components.front();
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
            << ",\"prefill_mode\":\"causal_chunked\""
            << ",\"prefill_chunk_tokens\":"
            << model.prefill_chunk_tokens()
            << ",\"session_retention\":true"
            << ",\"kv_page_tokens\":"
            << model.kv_page_tokens() << ",\"kv_page_bytes\":"
            << model.kv_page_bytes() << ",\"kv_page_capacity\":"
            << model.kv_page_capacity()
            << ",\"kv_dtype\":\"fp16\""
            << ",\"kv_allocation\":\"paged_on_demand\""
            << ",\"placement_profile\":\""
            << model.placement_profile() << "\",\"ram_cache_bytes\":"
            << model.ram_cache_bytes() << ",\"vram_cache_bytes\":"
            << model.vram_cache_bytes()
            << ",\"placement_prefetch_enabled\":"
            << (model.placement_prefetch_enabled() ? "true" : "false")
            << ",\"placement_prefetch_state\":\""
            << (model.placement_prefetch_enabled() ? "ready" : "disabled")
            << "\""
            << ",\"placement_minimum_observations\":"
            << model.placement_minimum_observations()
            << ",\"placement_settle_after_steps\":" << settle_after_steps
            << "}\n" << std::flush;
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      const auto fields = split_tabs(line);
      if (fields[0] == "PING") {
        if (fields.size() != 1) throw std::runtime_error("invalid PING");
        std::cout << "{\"type\":\"pong\"}\n" << std::flush;
      } else if (fields[0] == "STATS") {
        if (fields.size() != 1) throw std::runtime_error("invalid STATS");
        const auto phase = model.phase_telemetry();
        const auto cache = model.telemetry();
        std::cout << "{\"type\":\"stats\",\"placement_frozen\":"
                  << (model.placement_frozen() ? "true" : "false")
                  << ",\"kv_allocated_pages\":"
                  << model.kv_allocated_pages()
                  << ",\"kv_reserved_pages\":"
                  << model.kv_reserved_pages()
                  << ",\"retained_sessions\":" << retained.size()
                  << ",\"forward_calls\":" << phase.forward_calls
                  << ",\"forward_wall_ns\":" << phase.forward_wall_ns
                  << ",\"dense_router_ns\":" << phase.dense_router_ns
                  << ",\"attention_delta_ns\":" << phase.attention_delta_ns
                  << ",\"shared_expert_ns\":" << phase.shared_expert_ns
                  << ",\"router_ns\":" << phase.router_ns
                  << ",\"expert_cache_wait_ns\":"
                  << phase.expert_cache_wait_ns
                  << ",\"expert_compute_ns\":" << phase.expert_compute_ns
                  << ",\"cpu_expert_ns\":" << phase.cpu_expert_ns
                  << ",\"gpu_expert_ns\":" << phase.gpu_expert_ns
                  << ",\"cpu_gpu_overlap_ns\":" << phase.cpu_gpu_overlap_ns
                  << ",\"final_head_ns\":" << phase.final_head_ns
                  << ",\"cpu_expert_selections\":"
                  << phase.cpu_expert_selections
                  << ",\"gpu_expert_selections\":"
                  << phase.gpu_expert_selections
                  << ",\"useful_prefetches\":" << phase.useful_prefetches
                  << ",\"wasted_prefetches\":" << phase.wasted_prefetches
                  << ",\"frozen_promotions\":" << phase.frozen_promotions
                  << ",\"frozen_promotion_bytes\":"
                  << phase.frozen_promotion_bytes
                  << ",\"cache_read_bytes\":" << cache.read_bytes
                  << ",\"cache_uploaded_bytes\":" << cache.uploaded_bytes
                  << ",\"cache_vram_hits\":" << cache.acquire_vram_hits
                  << ",\"cache_ram_hits\":" << cache.acquire_ram_hits
                  << ",\"cache_ssd_misses\":" << cache.acquire_ssd_misses
                  << ",\"cache_storage_wait_ns\":" << cache.storage_wait_ns
                  << ",\"cache_upload_wait_ns\":" << cache.upload_wait_ns
                  << "}\n" << std::flush;
      } else if (fields[0] == "BEGIN") {
        if (fields.size() != 4 && fields.size() != 6)
          throw std::runtime_error("invalid BEGIN");
        const auto request_id = std::stoull(std::string(fields[1]));
        const auto context_limit_u64 = std::stoull(std::string(fields[2]));
        if (context_limit_u64 > std::numeric_limits<std::uint32_t>::max())
          throw std::runtime_error("BEGIN context limit exceeds u32");
        const auto context_limit =
            static_cast<std::uint32_t>(context_limit_u64);
        if (!request_id || active.contains(request_id))
          throw std::runtime_error("invalid or duplicate request id");
        const auto prompt = parse_tokens(fields[3]);
        if (fields.size() == 6) {
          if (fields[4] != "RESUME")
            throw std::runtime_error("invalid BEGIN resume marker");
          const auto key = std::stoull(std::string(fields[5]));
          const auto retained_iterator = retained.find(key);
          if (retained_iterator == retained.end())
            throw std::runtime_error("unknown retained session");
          const auto slot = retained_iterator->second.slot;
          const auto base = retained_iterator->second.tokens;
          if (static_cast<std::uint64_t>(base) + prompt.size() >
              context_limit)
            throw std::runtime_error("BEGIN context limit below resume length");
          try {
            model.grow_slot(slot, context_limit);
            const auto predicted = model.prefill(slot, prompt, base);
            active.emplace(request_id,
                           WorkerRequest{slot, predicted,
                                         base + static_cast<std::uint32_t>(
                                                    prompt.size())});
            retained.erase(retained_iterator);
          } catch (...) {
            // A failed resume leaves the slot state partially overwritten;
            // drop it so the server can fall back to a fresh prefill.
            model.release_slot(slot);
            used_slots[slot] = false;
            retained.erase(retained_iterator);
            throw;
          }
          std::cout << "{\"type\":\"begun\",\"id\":" << request_id
                    << ",\"slot\":" << slot << ",\"resumed_tokens\":" << base
                    << "}\n" << std::flush;
        } else {
          const auto available =
              std::find(used_slots.begin(), used_slots.end(), false);
          if (available == used_slots.end())
            throw std::runtime_error("worker request capacity exhausted");
          const auto slot = static_cast<std::uint32_t>(
              std::distance(used_slots.begin(), available));
          used_slots[slot] = true;
          try {
            model.reserve_slot(slot, context_limit);
            const auto predicted = model.prefill(slot, prompt);
            active.emplace(
                request_id, WorkerRequest{
                                slot, predicted,
                                static_cast<std::uint32_t>(prompt.size())});
          } catch (...) {
            model.release_slot(slot);
            used_slots[slot] = false;
            throw;
          }
          std::cout << "{\"type\":\"begun\",\"id\":" << request_id
                    << ",\"slot\":" << slot << "}\n" << std::flush;
        }
      } else if (fields[0] == "NEXT") {
        if (fields.size() != 3) throw std::runtime_error("invalid NEXT");
        const auto id = std::stoull(std::string(fields[1]));
        const auto iterator = active.find(id);
        if (!id || iterator == active.end())
          throw std::runtime_error("NEXT request mismatch");
        if (fields[2] != "0" && fields[2] != "1")
          throw std::runtime_error("NEXT final flag must be 0 or 1");
        const bool final = fields[2] == "1";
        const auto token = iterator->second.predicted;
        if (!final) {
          const std::array tokens{token}, positions{iterator->second.next_position},
              slots{iterator->second.slot};
          iterator->second.predicted =
              model.forward_batch(tokens, positions, slots).front();
          ++iterator->second.next_position;
        }
          std::cout << "{\"type\":\"token\",\"id\":" << id
                    << ",\"tokens\":[" << token << "]}\n" << std::flush;
        if (!final) maybe_settle_placement();
        if (final) {
          model.release_slot(iterator->second.slot);
          used_slots[iterator->second.slot] = false;
          active.erase(iterator);
        }
      } else if (fields[0] == "STEP") {
        if (fields.size() < 2 || fields.size() > model.capacity() + 1U)
          throw std::runtime_error("invalid STEP field count");
        struct Step final {
          std::uint64_t id{};
          bool final{};
          std::uint32_t token{};
        };
        std::vector<Step> steps;
        std::vector<std::uint32_t> tokens, positions, slots;
        std::vector<std::uint64_t> advancing_ids;
        steps.reserve(fields.size() - 1U);
        for (std::size_t field = 1; field < fields.size(); ++field) {
          const auto separator = fields[field].find(',');
          if (separator == std::string_view::npos)
            throw std::runtime_error("invalid STEP item");
          const auto id = std::stoull(std::string(fields[field].substr(0, separator)));
          const auto flag = fields[field].substr(separator + 1U);
          // Modes: 0 = decode, 1 = final emit-and-release, 2 = plain decode
          // that keeps the slot ("hold"). Without MTP speculation a hold is
          // exactly a plain decode here; the server emits it for retained
          // turns when MTP is enabled, so accept it before Qwen grows MTP.
          if (!id || (flag != "0" && flag != "1" && flag != "2") ||
              std::any_of(steps.begin(), steps.end(),
                          [&](const Step& step) { return step.id == id; }))
            throw std::runtime_error("invalid STEP request");
          const auto iterator = active.find(id);
          if (iterator == active.end())
            throw std::runtime_error("STEP request mismatch");
          const bool final = flag == "1";
          steps.push_back({id, final, iterator->second.predicted});
          if (!final) {
            tokens.push_back(iterator->second.predicted);
            positions.push_back(iterator->second.next_position);
            slots.push_back(iterator->second.slot);
            advancing_ids.push_back(id);
          }
        }
        if (!tokens.empty()) {
          const auto predicted = model.forward_batch(tokens, positions, slots);
          for (std::size_t index = 0; index < advancing_ids.size(); ++index) {
            auto& request = active.at(advancing_ids[index]);
            request.predicted = predicted[index];
            ++request.next_position;
          }
        }
        std::cout << "{\"type\":\"batch\",\"items\":[";
        for (std::size_t index = 0; index < steps.size(); ++index) {
          if (index) std::cout << ',';
          std::cout << "{\"id\":" << steps[index].id
                    << ",\"tokens\":[" << steps[index].token << "]}";
        }
        std::cout << "]}\n" << std::flush;
        if (!tokens.empty()) maybe_settle_placement();
        for (const auto& step : steps) {
          if (step.final) {
            const auto slot = active.at(step.id).slot;
            model.release_slot(slot);
            used_slots[slot] = false;
            active.erase(step.id);
          }
        }
      } else if (fields[0] == "END") {
        if (fields.size() != 2 && fields.size() != 4)
          throw std::runtime_error("END request mismatch");
        const auto id = std::stoull(std::string(fields[1]));
        const auto iterator = active.find(id);
        if (!id || iterator == active.end())
          throw std::runtime_error("END request mismatch");
        if (fields.size() == 4) {
          if (fields[2] != "RETAIN")
            throw std::runtime_error("invalid END retain marker");
          const auto key = std::stoull(std::string(fields[3]));
          if (retained.contains(key))
            throw std::runtime_error("duplicate retained session");
          const auto slot = iterator->second.slot;
          const auto tokens = iterator->second.next_position;
          retained.emplace(key, RetainedRequest{slot, tokens});
          active.erase(iterator);
          std::cout << "{\"type\":\"ended\",\"id\":" << id
                    << ",\"retained_tokens\":" << tokens << "}\n"
                    << std::flush;
        } else {
          model.release_slot(iterator->second.slot);
          used_slots[iterator->second.slot] = false;
          active.erase(iterator);
          std::cout << "{\"type\":\"ended\",\"id\":" << id << "}\n"
                    << std::flush;
        }
      } else if (fields[0] == "DROP") {
        if (fields.size() != 2) throw std::runtime_error("invalid DROP");
        const auto key = std::stoull(std::string(fields[1]));
        const auto iterator = retained.find(key);
        const bool found = iterator != retained.end();
        if (found) {
          model.release_slot(iterator->second.slot);
          used_slots[iterator->second.slot] = false;
          retained.erase(iterator);
        }
        std::cout << "{\"type\":\"dropped\",\"key\":" << key
                  << ",\"found\":" << (found ? "true" : "false") << "}\n"
                  << std::flush;
      } else if (fields[0] == "SHUTDOWN") {
        if (fields.size() != 1 || !active.empty() || !retained.empty())
          throw std::runtime_error("invalid SHUTDOWN");
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

int expert_vm_hybrid_delta_moe_provider_main(int argc, char** argv) {
  try {
    if (argc >= 3 && std::string_view(argv[2]) == "--trace-moe") {
      if (argc < 5 || argc > 10)
        throw std::runtime_error(
            "trace usage: <container> --trace-moe <prompt-file> <output-dir> "
            "[layers-csv] [ram-gib] [vram-gib] [chunk-tokens] "
            "[maximum-records]");
      const auto prompts = read_prompt_file(argv[3]);
      const auto layers = parse_layers(argc >= 6 ? argv[5] : "20,21,22,23");
      const auto ram_gib = argc >= 7 ? std::stoull(argv[6]) : 48ULL;
      const auto vram_gib = argc >= 8 ? std::stoull(argv[7]) : 14ULL;
      const auto chunk_tokens = argc >= 9
          ? static_cast<std::uint32_t>(std::stoul(argv[8])) : 16U;
      const auto maximum_records = argc >= 10 ? std::stoull(argv[9]) : 0ULL;
      if (!ram_gib || !vram_gib || !chunk_tokens)
        throw std::runtime_error("zero MoE trace runtime setting");
      const auto longest_prompt = std::max_element(
          prompts.begin(), prompts.end(),
          [](const auto& left, const auto& right) {
            return left.size() < right.size();
          })->size();
      if (longest_prompt > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("trace prompt exceeds u32 context");
      const auto started = std::chrono::steady_clock::now();
      Qwen3NextModel model(
          argv[1], static_cast<std::uint32_t>(longest_prompt), ram_gib << 30U,
          vram_gib << 30U, chunk_tokens);
      model.enable_moe_trace(argv[4], layers, maximum_records);
      std::uint64_t input_tokens = 0;
      std::uint32_t processed_sequences = 0;
      for (std::uint32_t sequence = 0; sequence < prompts.size(); ++sequence) {
        if (model.moe_trace_full()) break;
        const auto& prompt = prompts[sequence];
        model.reserve_slot(0, static_cast<std::uint32_t>(prompt.size()));
        model.set_trace_sequence_id(0, sequence);
        static_cast<void>(model.prefill(0, prompt));
        model.release_slot(0);
        input_tokens += prompt.size();
        ++processed_sequences;
      }
      model.finalize_moe_trace();
      const auto seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::cout << "{\"type\":\"moe_trace\",\"records\":"
                << model.moe_trace_records() << ",\"input_tokens\":"
                << input_tokens << ",\"sequences\":" << processed_sequences
                << ",\"seconds\":" << seconds << "}\n";
      return 0;
    }
    if (argc >= 3 && std::string_view(argv[2]) == "--batch") {
      if (argc < 4 || argc > 13)
        throw std::runtime_error(
            "batch usage: <container> --batch <token-ids-csv> [new-tokens] "
            "[concurrency] [ram-gib] [vram-gib] [warmup-rounds] "
            "[kv-cache-mib] [kv-page-tokens] [placement-profile] "
            "[prefill-chunk-tokens]");
      auto prompts = parse_prompt_batch(argv[3]);
      const auto new_tokens = argc >= 5
          ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 8U;
      const auto concurrency = argc >= 6
          ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 4U;
      const auto ram_gib = argc >= 7 ? std::stoull(argv[6]) : 48ULL;
      const auto vram_gib = argc >= 8 ? std::stoull(argv[7]) : 14ULL;
      const auto warmup_rounds = argc >= 9
          ? static_cast<std::uint32_t>(std::stoul(argv[8])) : 1U;
      const auto kv_cache_mib = argc >= 10 ? std::stoull(argv[9]) : 2048ULL;
      const auto kv_page_tokens = argc >= 11
          ? static_cast<std::uint32_t>(std::stoul(argv[10])) : 256U;
      const std::string_view placement_profile =
          argc >= 12 ? argv[11] : "balanced";
      const auto prefill_chunk_tokens = argc >= 13
          ? static_cast<std::uint32_t>(std::stoul(argv[12])) : 0U;
      if (!new_tokens || !concurrency || !ram_gib || !vram_gib)
        throw std::runtime_error("zero batched runtime setting");
      if (prompts.size() == 1U) prompts.resize(concurrency, prompts.front());
      if (prompts.size() != concurrency)
        throw std::runtime_error(
            "batch prompt count must be one or equal concurrency");
      const bool identical_prompts = std::all_of(
          prompts.begin() + 1, prompts.end(),
          [&](const auto& prompt) { return prompt == prompts.front(); });
      const auto longest_prompt = std::max_element(
          prompts.begin(), prompts.end(),
          [](const auto& left, const auto& right) {
            return left.size() < right.size();
          })->size();
      const auto max_context = static_cast<std::uint32_t>(longest_prompt) +
                               new_tokens;
      const auto started_load = std::chrono::steady_clock::now();
      Qwen3NextModel model(argv[1], max_context, ram_gib << 30U,
                           vram_gib << 30U, concurrency, kv_cache_mib << 20U,
                           kv_page_tokens, placement_profile,
                           prefill_chunk_tokens);
      for (std::uint32_t row = 0; row < concurrency; ++row)
        model.reserve_slot(
            row, static_cast<std::uint32_t>(prompts[row].size()) + new_tokens);
      const auto load_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_load).count();
      std::vector<std::uint32_t> batch_tokens, positions, state_slots,
          predicted(concurrency);
      std::vector<std::uint32_t> all_slots(concurrency);
      std::iota(all_slots.begin(), all_slots.end(), 0U);
      const auto prefill = [&]() {
        std::fill(predicted.begin(), predicted.end(), 0U);
        for (std::uint32_t position = 0; position < longest_prompt; ++position) {
          batch_tokens.clear();
          positions.clear();
          state_slots.clear();
          for (std::uint32_t row = 0; row < concurrency; ++row) {
            if (position >= prompts[row].size()) continue;
            batch_tokens.push_back(prompts[row][position]);
            positions.push_back(position);
            state_slots.push_back(row);
          }
          const auto outputs =
              model.forward_batch(batch_tokens, positions, state_slots);
          for (std::size_t index = 0; index < state_slots.size(); ++index)
            predicted[state_slots[index]] = outputs[index];
        }
      };
      for (std::uint32_t round = 0; round < warmup_rounds; ++round) {
        model.reset_request();
        prefill();
        for (std::uint32_t step = 0; step + 1U < new_tokens; ++step) {
          positions.resize(concurrency);
          for (std::uint32_t row = 0; row < concurrency; ++row)
            positions[row] = static_cast<std::uint32_t>(prompts[row].size()) + step;
          predicted = model.forward_batch(predicted, positions, all_slots);
        }
      }
      model.settle_placement();
      const auto baseline_metrics = model.telemetry();
      const auto baseline_phase = model.phase_telemetry();
      const auto baseline_dispatch = model.dispatch_telemetry();
      const auto baseline_cpu_executor = model.cpu_executor_telemetry();
      model.reset_request();
      const auto started_prompt = std::chrono::steady_clock::now();
      prefill();
      const auto prompt_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_prompt).count();
      std::vector<std::vector<std::uint32_t>> generated(concurrency);
      std::vector<double> inter_token_ms;
      const auto started_decode = std::chrono::steady_clock::now();
      for (std::uint32_t step = 0; step < new_tokens; ++step) {
        for (std::uint32_t row = 0; row < concurrency; ++row)
          generated[row].push_back(predicted[row]);
        if (step + 1U < new_tokens) {
          positions.resize(concurrency);
          for (std::uint32_t row = 0; row < concurrency; ++row)
            positions[row] = static_cast<std::uint32_t>(prompts[row].size()) + step;
          const auto step_started = std::chrono::steady_clock::now();
          predicted = model.forward_batch(predicted, positions, all_slots);
          inter_token_ms.push_back(std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - step_started).count());
        }
      }
      const auto decode_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_decode).count();
      const auto forwards = static_cast<std::uint64_t>(new_tokens - 1U) *
                            concurrency;
      const bool identical_outputs = std::all_of(
          generated.begin() + 1, generated.end(),
          [&](const auto& sequence) { return sequence == generated.front(); });
      const auto metrics = model.telemetry();
      const auto phases = phase_delta(model.phase_telemetry(), baseline_phase);
      const auto dispatch = model.dispatch_telemetry();
      const auto cpu_executor = model.cpu_executor_telemetry();
      const auto measured_read_bytes =
          metrics.read_bytes - baseline_metrics.read_bytes;
      const auto measured_uploaded_bytes =
          metrics.uploaded_bytes - baseline_metrics.uploaded_bytes;
      const auto measured_vram_hits =
          metrics.acquire_vram_hits - baseline_metrics.acquire_vram_hits;
      const auto measured_ram_hits =
          metrics.acquire_ram_hits - baseline_metrics.acquire_ram_hits;
      const auto measured_ssd_misses =
          metrics.acquire_ssd_misses - baseline_metrics.acquire_ssd_misses;
      const auto prompt_forwards = std::accumulate(
          prompts.begin(), prompts.end(), std::uint64_t{0},
          [](std::uint64_t total, const auto& prompt) {
            return total + prompt.size();
          });
      const auto model_forwards = prompt_forwards + forwards;
      const auto acquires =
          measured_vram_hits + measured_ram_hits + measured_ssd_misses;

      std::vector<std::vector<std::uint32_t>> isolated(concurrency);
      for (std::uint32_t row = 0; row < concurrency; ++row) {
        model.reset_slot(row);
        std::uint32_t isolated_prediction = 0;
        const std::array slot{row};
        for (std::uint32_t position = 0; position < prompts[row].size(); ++position) {
          const std::array token{prompts[row][position]}, one_position{position};
          isolated_prediction =
              model.forward_batch(token, one_position, slot).front();
        }
        for (std::uint32_t step = 0; step < new_tokens; ++step) {
          isolated[row].push_back(isolated_prediction);
          if (step + 1U < new_tokens) {
            const std::array token{isolated_prediction}, one_position{
                static_cast<std::uint32_t>(prompts[row].size()) + step};
            isolated_prediction =
                model.forward_batch(token, one_position, slot).front();
          }
        }
      }
      const bool interleaving_match = isolated == generated;
      std::vector<std::vector<std::uint32_t>> chunked(concurrency);
      for (std::uint32_t row = 0; row < concurrency; ++row) {
        model.reset_slot(row);
        auto chunked_prediction = model.prefill(row, prompts[row]);
        const std::array slot{row};
        for (std::uint32_t step = 0; step < new_tokens; ++step) {
          chunked[row].push_back(chunked_prediction);
          if (step + 1U < new_tokens) {
            const std::array token{chunked_prediction}, one_position{
                static_cast<std::uint32_t>(prompts[row].size()) + step};
            chunked_prediction =
                model.forward_batch(token, one_position, slot).front();
          }
        }
      }
      const bool chunked_prefill_match = chunked == isolated;
      std::cout << "{\"tokens\":[";
      for (std::size_t i = 0; i < generated.front().size(); ++i) {
        if (i) std::cout << ',';
        std::cout << generated.front()[i];
      }
      std::cout << "],\"request_tokens\":[";
      for (std::size_t row = 0; row < generated.size(); ++row) {
        if (row) std::cout << ',';
        std::cout << '[';
        for (std::size_t token = 0; token < generated[row].size(); ++token) {
          if (token) std::cout << ',';
          std::cout << generated[row][token];
        }
        std::cout << ']';
      }
      std::cout << "],\"concurrency\":" << concurrency
                << ",\"placement_profile\":\""
                << model.placement_profile()
                << "\",\"ram_cache_bytes\":" << model.ram_cache_bytes()
                << ",\"vram_cache_bytes\":" << model.vram_cache_bytes()
                << ",\"placement_prefetch_enabled\":"
                << (model.placement_prefetch_enabled() ? "true" : "false")
                << ",\"placement_prefetch_state\":\""
                << (model.placement_prefetch_enabled() ? "ready" : "disabled")
                << "\""
                << ",\"placement_minimum_observations\":"
                << model.placement_minimum_observations()
                << ",\"warmup_rounds\":" << warmup_rounds
                << ",\"mixed_prompts\":"
                << (!identical_prompts ? "true" : "false")
                << ",\"identical_outputs\":"
                << (identical_outputs ? "true" : "false")
                << ",\"interleaving_match\":"
                << (interleaving_match ? "true" : "false")
                << ",\"chunked_prefill_match\":"
                << (chunked_prefill_match ? "true" : "false")
                << ",\"prefill_chunk_tokens\":"
                << model.prefill_chunk_tokens()
                << ",\"model_load_seconds\":" << load_seconds
                << ",\"prompt_seconds\":" << prompt_seconds
                << ",\"warm_ttft_seconds\":" << prompt_seconds
                << ",\"cold_ttft_seconds\":"
                << (load_seconds + prompt_seconds)
                << ",\"decode_seconds\":" << decode_seconds
                << ",\"aggregate_forward_tokens\":" << forwards
                << ",\"tokens_per_second\":"
                << (forwards ? forwards / decode_seconds : 0.0)
                << ",\"inter_token_p50_ms\":"
                << percentile_ms(inter_token_ms, 0.50)
                << ",\"inter_token_p95_ms\":"
                << percentile_ms(inter_token_ms, 0.95)
                << ",\"container_bytes\":" << model.total_pack_bytes()
                << ",\"kv_page_tokens\":" << model.kv_page_tokens()
                << ",\"kv_page_bytes\":" << model.kv_page_bytes()
                << ",\"kv_page_capacity\":" << model.kv_page_capacity()
                << ",\"kv_allocated_pages\":" << model.kv_allocated_pages()
                << ",\"kv_reserved_pages\":" << model.kv_reserved_pages()
                << ",\"expert_read_bytes\":" << measured_read_bytes
                << ",\"expert_h2d_bytes\":" << measured_uploaded_bytes
                << ",\"expert_vram_hits\":" << measured_vram_hits
                << ",\"expert_ram_hits\":" << measured_ram_hits
                << ",\"expert_ssd_misses\":" << measured_ssd_misses
                << ",\"expert_acquires\":"
                << acquires
                << ",\"cold_bytes_per_forward\":"
                << (model_forwards ? measured_read_bytes / model_forwards : 0)
                << ",\"vram_hit_ratio\":"
                << (acquires ? static_cast<double>(measured_vram_hits) /
                                   static_cast<double>(acquires) : 0.0)
                << ",\"expert_loads\":"
                << (metrics.load_completed - baseline_metrics.load_completed)
                << ",\"expert_deduplicated\":"
                << (metrics.load_deduplicated -
                    baseline_metrics.load_deduplicated)
                << ",\"record_validations\":"
                << (metrics.record_validations -
                    baseline_metrics.record_validations)
                << ",\"validated_ram_reuses\":"
                << (metrics.validated_ram_reuses -
                    baseline_metrics.validated_ram_reuses)
                << ",\"ram_high_water\":" << metrics.ram_high_water
                << ",\"vram_high_water\":" << metrics.vram_high_water
                << ",\"vram_resident_high_water\":"
                << metrics.vram_resident_high_water
                << ",\"vram_transient_high_water\":"
                << metrics.vram_transient_high_water
                << ",\"evictions\":" << metrics.eviction_count
                << ",\"same_partition_evictions\":"
                << metrics.same_partition_evictions
                << ",\"over_quota_evictions\":"
                << metrics.over_quota_evictions;
      print_phase_json(std::cout, phases);
      print_dispatch_json(std::cout, dispatch, baseline_dispatch);
      print_cpu_executor_json(std::cout, cpu_executor, baseline_cpu_executor);
      std::cout << "}\n";
      return interleaving_match && chunked_prefill_match ? 0 : 2;
    }
    if (argc >= 3 && std::string_view(argv[2]) == "--worker") {
      std::vector<std::string_view> raw_options;
      raw_options.reserve(static_cast<std::size_t>(argc - 3));
      for (int index = 3; index < argc; ++index)
        raw_options.emplace_back(argv[index]);
      auto parsed =
          expert::runtime::parse_worker_launch_options(raw_options);
      if (!parsed.status.ok())
        throw std::runtime_error(std::string(parsed.status.message()));
      auto options = std::move(parsed.options);
      if (!options.extensions.empty())
        throw std::runtime_error("execution provider does not support requested extension");
      if (options.ram_cache_gib >
              (std::numeric_limits<std::uint64_t>::max() >> 30U) ||
          options.vram_cache_gib >
              (std::numeric_limits<std::uint64_t>::max() >> 30U) ||
          options.kv_cache_mib >
              (std::numeric_limits<std::uint64_t>::max() >> 20U))
        throw std::runtime_error("worker resource bytes overflow");
      const auto prefill_chunk_tokens =
          options.prefill_chunk_limit.value_or(0U);
      // Warmup boundary for the one-time placement freeze in worker mode;
      // 0 keeps placement adaptive forever (benchmarking).
      const auto settle_after_steps =
          options.placement_settle_steps.value_or(8U);
      Qwen3NextModel model(argv[1], options.max_context,
                           options.ram_cache_gib << 30U,
                           options.vram_cache_gib << 30U, options.capacity,
                           options.kv_cache_mib << 20U,
                           options.kv_page_tokens, options.placement_profile,
                           prefill_chunk_tokens);
      return worker_loop(model, settle_after_steps);
    }
    if (argc < 3 || argc > 10) {
      std::cerr << "usage: expert-qwen3-next-runner <container> <token-ids-csv> "
                   "[new-tokens] [ram-cache-gib] [vram-cache-gib] "
                   "[warmup-rounds] [kv-cache-mib] [kv-page-tokens] "
                   "[placement-profile]\n";
      return 64;
    }
    auto tokens = parse_tokens(argv[2]);
    const auto new_tokens = argc >= 4 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 8U;
    const auto ram_gib = argc >= 5 ? std::stoull(argv[4]) : 48ULL;
    const auto vram_gib = argc >= 6 ? std::stoull(argv[5]) : 14ULL;
    const auto warmup_rounds = argc >= 7
        ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 1U;
    const auto kv_cache_mib = argc >= 8 ? std::stoull(argv[7]) : 2048ULL;
    const auto kv_page_tokens = argc >= 9
        ? static_cast<std::uint32_t>(std::stoul(argv[8])) : 256U;
    const std::string_view placement_profile =
        argc >= 10 ? argv[9] : "balanced";
    if (!new_tokens || !ram_gib || !vram_gib) throw std::runtime_error("zero runtime budget");
    const auto max_context = static_cast<std::uint32_t>(tokens.size()) + new_tokens;
    const auto started_load = std::chrono::steady_clock::now();
    Qwen3NextModel model(argv[1], max_context, ram_gib << 30U,
                         vram_gib << 30U, 1, kv_cache_mib << 20U,
                         kv_page_tokens, placement_profile);
    model.reserve_slot(0, max_context);
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_load).count();
    std::uint32_t predicted = 0;
    for (std::uint32_t round = 0; round < warmup_rounds; ++round) {
      model.reset_request();
      for (std::uint32_t position = 0; position < tokens.size(); ++position)
        predicted = model.forward(tokens[position], position);
      for (std::uint32_t step = 0; step + 1U < new_tokens; ++step)
        predicted = model.forward(
            predicted, static_cast<std::uint32_t>(tokens.size()) + step);
    }
    model.settle_placement();
    const auto baseline_metrics = model.telemetry();
    const auto baseline_phase = model.phase_telemetry();
    const auto baseline_dispatch = model.dispatch_telemetry();
    const auto baseline_cpu_executor = model.cpu_executor_telemetry();
    model.reset_request();
    const auto started_prompt = std::chrono::steady_clock::now();
    for (std::uint32_t position = 0; position < tokens.size(); ++position)
      predicted = model.forward(tokens[position], position);
    const auto prompt_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_prompt).count();
    const auto started_decode = std::chrono::steady_clock::now();
    std::vector<double> inter_token_ms;
    for (std::uint32_t generated = 0; generated < new_tokens; ++generated) {
      tokens.push_back(predicted);
      if (generated + 1U < new_tokens)
      {
        const auto step_started = std::chrono::steady_clock::now();
        predicted = model.forward(
            predicted, static_cast<std::uint32_t>(tokens.size() - 1U));
        inter_token_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - step_started).count());
      }
    }
    const auto decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_decode).count();
    const auto metrics = model.telemetry();
    const auto phases = phase_delta(model.phase_telemetry(), baseline_phase);
    const auto dispatch = model.dispatch_telemetry();
    const auto cpu_executor = model.cpu_executor_telemetry();
    const auto measured_read_bytes =
        metrics.read_bytes - baseline_metrics.read_bytes;
    const auto measured_uploaded_bytes =
        metrics.uploaded_bytes - baseline_metrics.uploaded_bytes;
    const auto measured_vram_hits =
        metrics.acquire_vram_hits - baseline_metrics.acquire_vram_hits;
    const auto measured_ram_hits =
        metrics.acquire_ram_hits - baseline_metrics.acquire_ram_hits;
    const auto measured_ssd_misses =
        metrics.acquire_ssd_misses - baseline_metrics.acquire_ssd_misses;
    const auto forwards = new_tokens > 0 ? new_tokens - 1U : 0U;
    const auto model_forwards =
        static_cast<std::uint64_t>(tokens.size() - new_tokens) + forwards;
    const auto acquires =
        measured_vram_hits + measured_ram_hits + measured_ssd_misses;
    std::cout << "{\"tokens\":[";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << tokens[i];
    }
    std::cout << "],\"placement_profile\":\""
              << model.placement_profile()
              << "\",\"ram_cache_bytes\":" << model.ram_cache_bytes()
              << ",\"vram_cache_bytes\":" << model.vram_cache_bytes()
              << ",\"placement_prefetch_enabled\":"
              << (model.placement_prefetch_enabled() ? "true" : "false")
              << ",\"placement_prefetch_state\":\""
              << (model.placement_prefetch_enabled() ? "ready" : "disabled")
              << "\""
              << ",\"placement_minimum_observations\":"
              << model.placement_minimum_observations()
              << ",\"model_load_seconds\":" << load_seconds
              << ",\"kv_page_tokens\":" << model.kv_page_tokens()
              << ",\"kv_page_bytes\":" << model.kv_page_bytes()
              << ",\"kv_page_capacity\":" << model.kv_page_capacity()
              << ",\"kv_allocated_pages\":" << model.kv_allocated_pages()
              << ",\"kv_reserved_pages\":" << model.kv_reserved_pages()
              << ",\"warmup_rounds\":" << warmup_rounds
              << ",\"prompt_seconds\":" << prompt_seconds
              << ",\"warm_ttft_seconds\":" << prompt_seconds
              << ",\"cold_ttft_seconds\":"
              << (load_seconds + prompt_seconds)
              << ",\"decode_seconds\":" << decode_seconds
              << ",\"tokens_per_second\":"
              << (forwards ? forwards / decode_seconds : 0.0)
              << ",\"inter_token_p50_ms\":"
              << percentile_ms(inter_token_ms, 0.50)
              << ",\"inter_token_p95_ms\":"
              << percentile_ms(inter_token_ms, 0.95)
              << ",\"container_bytes\":" << model.total_pack_bytes()
              << ",\"startup_dense_read_bytes\":" << model.dense_read_bytes()
              << ",\"startup_dense_h2d_bytes\":" << model.dense_read_bytes()
              << ",\"expert_read_bytes\":" << measured_read_bytes
              << ",\"expert_h2d_bytes\":" << measured_uploaded_bytes
              << ",\"expert_vram_hits\":" << measured_vram_hits
              << ",\"expert_ram_hits\":" << measured_ram_hits
              << ",\"expert_ssd_misses\":" << measured_ssd_misses
              << ",\"expert_acquires\":"
              << acquires
              << ",\"cold_bytes_per_forward\":"
              << (model_forwards ? measured_read_bytes / model_forwards : 0)
              << ",\"vram_hit_ratio\":"
              << (acquires ? static_cast<double>(measured_vram_hits) /
                                 static_cast<double>(acquires) : 0.0)
              << ",\"expert_loads\":"
              << (metrics.load_completed - baseline_metrics.load_completed)
              << ",\"expert_deduplicated\":"
              << (metrics.load_deduplicated -
                  baseline_metrics.load_deduplicated)
              << ",\"record_validations\":"
              << (metrics.record_validations -
                  baseline_metrics.record_validations)
              << ",\"validated_ram_reuses\":"
              << (metrics.validated_ram_reuses -
                  baseline_metrics.validated_ram_reuses)
              << ",\"ram_high_water\":" << metrics.ram_high_water
              << ",\"vram_high_water\":" << metrics.vram_high_water
              << ",\"vram_resident_high_water\":"
              << metrics.vram_resident_high_water
              << ",\"vram_transient_high_water\":"
              << metrics.vram_transient_high_water
              << ",\"evictions\":" << metrics.eviction_count
              << ",\"same_partition_evictions\":"
              << metrics.same_partition_evictions
              << ",\"over_quota_evictions\":"
              << metrics.over_quota_evictions;
    print_phase_json(std::cout, phases);
    print_dispatch_json(std::cout, dispatch, baseline_dispatch);
    print_cpu_executor_json(std::cout, cpu_executor, baseline_cpu_executor);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3-Next runner: " << error.what() << '\n';
    return 1;
  }
}

expert::runtime::WorkerProviderDefinition
make_sm86_hybrid_delta_moe_provider() {
  return {"sm86-hybrid-delta-moe", 100U, provider_capabilities(),
          &expert_vm_hybrid_delta_moe_provider_main};
}

expert::runtime::CreateExecutionProviderModuleResult
make_sm86_hybrid_delta_moe_callable_provider(
    const std::filesystem::path& artifact_root, std::uint32_t max_context,
    std::uint32_t capacity, std::uint64_t ram_cache_bytes,
    std::uint64_t vram_cache_bytes, std::uint64_t kv_cache_bytes,
    std::uint32_t kv_page_tokens, std::string_view placement_profile) {
  try {
    auto implementation = std::make_shared<Qwen3NextModel>(
        artifact_root, max_context, ram_cache_bytes, vram_cache_bytes,
        capacity, kv_cache_bytes, kv_page_tokens, placement_profile, 1U);
    expert::runtime::ExecutionProviderModule module;
    module.definition = {"sm86-hybrid-delta-moe", 100U,
                         provider_capabilities(), implementation};
    module.tensor_store = implementation;
    module.service = {
        "causal_sequential", 1U, true, "per_request_nonblocking", "artifact",
        "fp16", "paged_on_demand", implementation->kv_page_tokens(),
        implementation->kv_page_bytes(), implementation->kv_page_capacity(),
        "budgeted", implementation->placement_profile(), ram_cache_bytes,
        vram_cache_bytes, implementation->placement_prefetch_enabled(),
        implementation->placement_prefetch_enabled() ? "ready" : "disabled",
        implementation->placement_minimum_observations(), false, false, false,
        true, true};
    module.telemetry = [implementation] {
      const auto cache = implementation->telemetry();
      const auto phase = implementation->phase_telemetry();
      return std::map<std::string, std::uint64_t, std::less<>>{
          {"cache_vram_hits", cache.acquire_vram_hits},
          {"cache_ram_hits", cache.acquire_ram_hits},
          {"cache_ssd_misses", cache.acquire_ssd_misses},
          {"cache_read_bytes", cache.read_bytes},
          {"cache_uploaded_bytes", cache.uploaded_bytes},
          {"cache_storage_wait_ns", cache.storage_wait_ns},
          {"cache_upload_wait_ns", cache.upload_wait_ns},
          {"forward_calls", phase.forward_calls},
          {"forward_wall_ns", phase.forward_wall_ns}};
    };
    return {expert::runtime::Status::success(), std::move(module)};
  } catch (const std::exception& error) {
    return {{expert::runtime::ErrorCode::invalid_argument, error.what()}, {}};
  }
}

#ifndef EXPERT_VM_PROVIDER_LIBRARY
int main(int argc, char** argv) {
  return expert_vm_hybrid_delta_moe_provider_main(argc, argv);
}
#endif
