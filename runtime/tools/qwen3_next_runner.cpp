#include "expert/core/json.hpp"
#include "expert/runtime/adaptive_placement.hpp"
#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cpu/expert_executor.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
using expert::core::json::Required;
using expert::core::json::Value;

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
std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}
double number(const Value& value) {
  if (const auto* item = std::get_if<std::int64_t>(&value.data))
    return static_cast<double>(*item);
  if (const auto* item = std::get_if<std::uint64_t>(&value.data))
    return static_cast<double>(*item);
  if (const auto* item = std::get_if<Value::Number>(&value.data))
    return std::stod(item->token);
  throw std::runtime_error("JSON value is not numeric");
}
std::uint32_t u32(const Value::Object& object, std::string_view key,
                  std::string_view where) {
  const auto value = Required(object, key, where).AsU64(where);
  if (value > 0xffffffffULL) throw std::runtime_error("integer exceeds u32");
  return static_cast<std::uint32_t>(value);
}
expert::runtime::Sha256Digest parse_digest(std::string_view text) {
  if (text.size() != 64) throw std::runtime_error("invalid SHA-256 length");
  const auto nibble = [](char value) -> unsigned {
    if (value >= '0' && value <= '9') return static_cast<unsigned>(value - '0');
    if (value >= 'a' && value <= 'f') return static_cast<unsigned>(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return static_cast<unsigned>(value - 'A' + 10);
    throw std::runtime_error("invalid SHA-256 hex");
  };
  expert::runtime::Sha256Digest result{};
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::byte>((nibble(text[2 * i]) << 4U) |
                                       nibble(text[2 * i + 1]));
  return result;
}

struct DevicePack final {
  std::byte* base{};
  std::uint64_t bytes{};
};

DevicePack upload_dense_pack(const std::filesystem::path& path,
                             std::uint64_t expected_bytes,
                             std::string_view expected_sha) {
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
  if (!expert::runtime::constant_time_equal(hasher.finalize(),
                                             parse_digest(expected_sha)))
    throw std::runtime_error("dense pack SHA-256 mismatch");
  return result;
}

struct Tensor final {
  expert::runtime::cuda::Int8Matrix int8;
  const float* f32{};
  std::vector<std::uint32_t> shape;
  bool quantized{};
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
  std::uint64_t cpu_expert_groups{};
  std::uint64_t cpu_expert_selections{};
  std::uint64_t adaptive_promotions{};
  std::uint64_t cpu_result_h2d_bytes{};
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
      value.cpu_expert_groups - baseline.cpu_expert_groups,
      value.cpu_expert_selections - baseline.cpu_expert_selections,
      value.adaptive_promotions - baseline.adaptive_promotions,
      value.cpu_result_h2d_bytes - baseline.cpu_result_h2d_bytes,
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
         << ",\"cpu_expert_groups\":" << phase.cpu_expert_groups
         << ",\"cpu_expert_selections\":" << phase.cpu_expert_selections
         << ",\"adaptive_promotions\":" << phase.adaptive_promotions
         << ",\"cpu_result_h2d_bytes\":" << phase.cpu_result_h2d_bytes
         << ",\"final_head_seconds\":"
         << phase.final_head_ns / ns_per_second
         << ",\"unattributed_seconds\":"
         << unattributed / ns_per_second;
}

class Qwen3NextModel final {
 public:
  Qwen3NextModel(const std::filesystem::path& root, std::uint32_t max_context,
                 std::uint64_t ram_cache_bytes,
                 std::uint64_t vram_cache_bytes,
                 std::uint32_t capacity = 1)
      : root_(root), max_context_(max_context), capacity_(capacity) {
    if (!capacity_) throw std::runtime_error("request capacity is zero");
    const auto document = expert::core::json::Parse(read_text(root / "manifest.json"));
    const auto& manifest = document.AsObject("manifest");
    const auto& architecture =
        Required(manifest, "architecture", "manifest").AsObject("architecture");
    if (Required(architecture, "family", "architecture").AsString("family") !=
        "qwen3_next")
      throw std::runtime_error("runner requires qwen3_next Expert Pack");
    hidden_ = u32(architecture, "hidden_size", "architecture");
    expert_width_ = u32(architecture, "intermediate_size", "architecture");
    vocab_ = u32(architecture, "vocab_size", "architecture");
    layers_ = u32(architecture, "num_hidden_layers", "architecture");
    query_heads_ = u32(architecture, "num_attention_heads", "architecture");
    kv_heads_ = u32(architecture, "num_key_value_heads", "architecture");
    head_dim_ = u32(architecture, "head_dim", "architecture");
    experts_ = u32(architecture, "num_experts", "architecture");
    top_k_ = u32(architecture, "num_experts_per_token", "architecture");
    full_interval_ = u32(architecture, "full_attention_interval", "architecture");
    conv_kernel_ = u32(architecture, "linear_conv_kernel_dim", "architecture");
    key_head_dim_ = u32(architecture, "linear_key_head_dim", "architecture");
    value_head_dim_ = u32(architecture, "linear_value_head_dim", "architecture");
    key_heads_ = u32(architecture, "linear_num_key_heads", "architecture");
    value_heads_ = u32(architecture, "linear_num_value_heads", "architecture");
    shared_width_ = u32(architecture, "shared_expert_intermediate_size", "architecture");
    epsilon_ = static_cast<float>(number(
        Required(architecture, "rms_norm_epsilon", "architecture")));
    rotary_dim_ = static_cast<std::uint32_t>(
        head_dim_ * number(Required(architecture, "partial_rotary_factor",
                                    "architecture")));
    const auto& rope = Required(architecture, "rope", "architecture").AsObject("rope");
    rope_theta_ = static_cast<float>(number(Required(rope, "theta", "rope")));
    if (hidden_ != 2048 || expert_width_ != 512 || query_heads_ != 16 ||
        kv_heads_ != 2 || head_dim_ != 256 || top_k_ != 10 ||
        key_heads_ != 16 || value_heads_ != 32 || key_head_dim_ != 128 ||
        value_head_dim_ != 128 || conv_kernel_ != 4 || rotary_dim_ != 64)
      throw std::runtime_error("unsupported Qwen3-Next geometry");

    const auto& packs = Required(manifest, "packs", "manifest").AsArray("packs");
    for (const auto& value : packs) {
      const auto& entry = value.AsObject("pack");
      const auto name = Required(entry, "name", "pack").AsString("pack.name");
      const auto bytes = Required(entry, "bytes", "pack").AsU64("pack.bytes");
      const auto path = root / name;
      if (!std::filesystem::is_regular_file(path) ||
          std::filesystem::file_size(path) != bytes)
        throw std::runtime_error("missing or truncated pack: " + name);
      total_pack_bytes_ += bytes;
      if (name == "dense.qpack") {
        dense_pack_ = upload_dense_pack(
            path, bytes, Required(entry, "sha256", "pack").AsString("pack.sha256"));
        dense_read_bytes_ = bytes;
      }
    }
    if (!dense_pack_.base) throw std::runtime_error("manifest has no dense pack");
    for (const auto& value : Required(manifest, "tensors", "manifest").AsArray("tensors"))
      add_tensor(value.AsObject("tensor"));
    build_expert_index(
        Required(manifest, "experts", "manifest").AsArray("experts"));
    route_access_counts_.resize(experts_);
    route_accesses_.reserve(experts_);

    const auto slot_bytes = static_cast<std::size_t>(max_expert_record_bytes_);
    const auto slot_count = std::max<std::size_t>(top_k_ * 2U, 32U);
    storage_ = std::make_shared<expert::runtime::WindowsIocpStorage>(4);
    uploader_ = std::make_shared<expert::runtime::cuda::CudaExpertUploader>();
    directory_ =
        std::make_shared<expert::runtime::cuda::CudaExpertDirectory>(
            model_id_, 1, layers_, experts_, capacity_ * top_k_);
    buffers_ = std::make_shared<expert::runtime::FixedBufferPool>(
        slot_count, slot_bytes, expert::runtime::kExpertPackAlignment,
        std::make_shared<expert::runtime::CudaPinnedAllocator>());
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
    const auto logical_threads = std::max(1U, std::thread::hardware_concurrency());
    cpu_executor_ =
        std::make_unique<expert::runtime::cpu::ExpertExecutor>(
            logical_threads);
    placement_ =
        std::make_unique<expert::runtime::AdaptivePlacementPlanner>(*cache_);
    allocate_workspace();
    cuda_check(cudaEventCreate(&layer_start_event_), "create layer-start event");
    cuda_check(cudaEventCreate(&attention_done_event_),
               "create attention-done event");
    cuda_check(cudaEventCreate(&shared_done_event_), "create shared-done event");
    cuda_check(cudaEventCreate(&router_done_event_), "create router-done event");
  }

  std::uint32_t forward(std::uint32_t token, std::uint32_t position) {
    const std::array tokens{token};
    const std::array positions{position};
    return forward_batch(tokens, positions).front();
  }

  std::vector<std::uint32_t> forward_batch(
      std::span<const std::uint32_t> tokens,
      std::span<const std::uint32_t> positions,
      std::span<const std::uint32_t> state_slots = {}) {
    const auto forward_started = std::chrono::steady_clock::now();
    if (tokens.empty() || tokens.size() != positions.size() ||
        tokens.size() > capacity_ ||
        (!state_slots.empty() && state_slots.size() != tokens.size()))
      throw std::runtime_error("invalid Qwen3-Next microbatch");
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    std::vector<std::uint32_t> default_slots;
    if (state_slots.empty()) {
      default_slots.resize(rows);
      std::iota(default_slots.begin(), default_slots.end(), 0U);
      state_slots = default_slots;
    }
    std::vector<bool> seen_slots(capacity_);
    for (std::uint32_t row = 0; row < rows; ++row) {
      if (positions[row] >= max_context_ || tokens[row] >= vocab_)
        throw std::runtime_error("token/position outside capacity");
      if (state_slots[row] >= capacity_ || seen_slots[state_slots[row]])
        throw std::runtime_error("invalid or duplicate request state slot");
      seen_slots[state_slots[row]] = true;
      status_check(expert::runtime::cuda::embedding(
          matrix("model.embed_tokens.weight"), tokens[row],
          hidden_state_ + static_cast<std::size_t>(row) * hidden_, nullptr));
    }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      cuda_check(cudaEventRecord(layer_start_event_), "record layer start");
      const auto prefix = "model.layers." + std::to_string(layer) + ".";
      for (std::uint32_t row = 0; row < rows; ++row)
        status_check(expert::runtime::cuda::qwen3_next_rms_norm(
            hidden_state_ + static_cast<std::size_t>(row) * hidden_,
            fp32(prefix + "input_layernorm.weight"),
            normalized_ + static_cast<std::size_t>(row) * hidden_, hidden_,
            epsilon_, nullptr));
      if ((layer + 1U) % full_interval_ == 0) {
        run_full_attention(prefix, layer, positions, state_slots, rows);
      } else {
        run_delta(prefix, layer, state_slots, rows);
      }
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state_, residual_, rows * hidden_, nullptr));
      for (std::uint32_t row = 0; row < rows; ++row)
        status_check(expert::runtime::cuda::qwen3_next_rms_norm(
            hidden_state_ + static_cast<std::size_t>(row) * hidden_,
            fp32(prefix + "post_attention_layernorm.weight"),
            normalized_ + static_cast<std::size_t>(row) * hidden_, hidden_,
            epsilon_, nullptr));
      cuda_check(cudaEventRecord(attention_done_event_),
                 "record attention done");
      run_moe(prefix, layer, rows);
    }
    const auto final_head_started = std::chrono::steady_clock::now();
    for (std::uint32_t row = 0; row < rows; ++row)
      status_check(expert::runtime::cuda::qwen3_next_rms_norm(
          hidden_state_ + static_cast<std::size_t>(row) * hidden_,
          fp32("model.norm.weight"),
          normalized_ + static_cast<std::size_t>(row) * hidden_, hidden_,
          epsilon_, nullptr));
    if (rows == 1) {
      status_check(expert::runtime::cuda::gemv_batch(
          matrix("lm_head.weight"), normalized_, logits_, rows, nullptr));
    } else {
      status_check(expert::runtime::cuda::gemv_batch_weight_reuse(
          matrix("lm_head.weight"), normalized_, logits_, rows, nullptr));
    }
    status_check(expert::runtime::cuda::argmax_batch(
        logits_, vocab_, rows, output_token_, nullptr));
    std::vector<std::uint32_t> result(rows);
    cuda_check(cudaMemcpy(result.data(), output_token_,
                          result.size() * sizeof(result[0]),
                          cudaMemcpyDeviceToHost),
               "copy generated tokens");
    phase_.final_head_ns += elapsed_ns(final_head_started);
    ++phase_.forward_calls;
    phase_.forward_wall_ns += elapsed_ns(forward_started);
    return result;
  }

  expert::runtime::TelemetrySnapshot telemetry() const {
    auto snapshot = cache_->telemetry();
    snapshot.acquire_vram_hits += directory_vram_hits_;
    return snapshot;
  }
  PhaseTelemetry phase_telemetry() const noexcept {
    auto result = phase_;
    result.adaptive_promotions = placement_->telemetry().completed;
    return result;
  }
  void settle_placement() {
    status_check(placement_->quiesce(std::chrono::seconds(30)));
  }
  std::uint64_t total_pack_bytes() const noexcept { return total_pack_bytes_; }
  std::uint64_t dense_read_bytes() const noexcept { return dense_read_bytes_; }
  std::uint32_t capacity() const noexcept { return capacity_; }
  void reset_slot(std::uint32_t slot) {
    if (slot >= capacity_) throw std::runtime_error("state slot out of range");
    const auto conv_elements =
        (static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
         static_cast<std::size_t>(value_heads_) * value_head_dim_) *
        conv_kernel_;
    const auto recurrent_elements = static_cast<std::size_t>(value_heads_) *
                                    key_head_dim_ * value_head_dim_;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      if ((layer + 1U) % full_interval_ != 0) {
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
  void reset_request() {
    const auto conv_elements =
        (static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
         static_cast<std::size_t>(value_heads_) * value_head_dim_) *
        conv_kernel_;
    const auto recurrent_elements = static_cast<std::size_t>(value_heads_) *
                                    key_head_dim_ * value_head_dim_;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      if ((layer + 1U) % full_interval_ != 0) {
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
  void add_tensor(const Value::Object& entry) {
    const auto name = Required(entry, "name", "tensor").AsString("tensor.name");
    const auto record = Required(entry, "offset", "tensor").AsU64("tensor.offset");
    const auto& sections = Required(entry, "sections", "tensor").AsObject("sections");
    const auto& data = Required(sections, "data", "sections").AsObject("data");
    const auto& scales = Required(sections, "scales", "sections").AsObject("scales");
    Tensor tensor;
    for (const auto& dimension :
         Required(entry, "source_shape", "tensor").AsArray("shape"))
      tensor.shape.push_back(static_cast<std::uint32_t>(dimension.AsU64("shape")));
    auto* data_pointer = dense_pack_.base + record +
        Required(data, "offset", "data").AsU64("data.offset");
    tensor.quantized =
        Required(entry, "stored_dtype", "tensor").AsString("dtype") == "I8";
    if (tensor.quantized) {
      tensor.int8 = {
          reinterpret_cast<const std::int8_t*>(data_pointer),
          reinterpret_cast<const float*>(
              dense_pack_.base + record +
              Required(scales, "offset", "scales").AsU64("scales.offset")),
          tensor.shape.at(0), tensor.shape.at(1)};
    } else {
      tensor.f32 = reinterpret_cast<const float*>(data_pointer);
    }
    tensors_.emplace(name, std::move(tensor));
  }
  void build_expert_index(const Value::Array& entries) {
    expert_records_.resize(static_cast<std::size_t>(layers_) * experts_);
    std::vector<bool> seen(expert_records_.size());
    for (const auto& value : entries) {
      const auto& entry = value.AsObject("expert");
      const auto layer = u32(entry, "layer", "expert");
      const auto expert = u32(entry, "expert", "expert");
      if (layer >= layers_ || expert >= experts_)
        throw std::runtime_error("expert index outside architecture");
      const auto index = static_cast<std::size_t>(layer) * experts_ + expert;
      if (seen[index]) throw std::runtime_error("duplicate expert record");
      seen[index] = true;
      auto& record = expert_records_[index];
      record.path = root_ /
          Required(entry, "pack", "expert").AsString("expert.pack");
      record.record_offset = Required(entry, "offset", "expert").AsU64("offset");
      record.stored_bytes =
          Required(entry, "stored_bytes", "expert").AsU64("stored_bytes");
      record.decoded_bytes =
          Required(entry, "decoded_bytes", "expert").AsU64("decoded_bytes");
      record.payload_sha256 = parse_digest(
          Required(entry, "payload_sha256", "expert").AsString("payload_sha256"));
      max_expert_record_bytes_ =
          std::max(max_expert_record_bytes_, record.stored_bytes);
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
      throw std::runtime_error("incomplete expert index");
  }
  void allocate_workspace() {
    const auto query_size = static_cast<std::size_t>(2U) * query_heads_ * head_dim_;
    const auto key_value_size = static_cast<std::size_t>(kv_heads_) * head_dim_;
    const auto projected_size =
        static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
        static_cast<std::size_t>(2U) * value_heads_ * value_head_dim_;
    const auto conv_size = static_cast<std::size_t>(2U) * key_heads_ * key_head_dim_ +
                           static_cast<std::size_t>(value_heads_) * value_head_dim_;
    hidden_state_ = device_allocate<float>(capacity_ * hidden_);
    normalized_ = device_allocate<float>(capacity_ * hidden_);
    residual_ = device_allocate<float>(capacity_ * hidden_);
    query_gate_ = device_allocate<float>(capacity_ * query_size);
    key_ = device_allocate<float>(capacity_ * key_value_size);
    value_ = device_allocate<float>(capacity_ * key_value_size);
    attention_ = device_allocate<float>(capacity_ * query_heads_ * head_dim_);
    projected_qkvz_ = device_allocate<float>(capacity_ * projected_size);
    projected_ba_ = device_allocate<float>(capacity_ * 2U * value_heads_);
    delta_output_ =
        device_allocate<float>(capacity_ * value_heads_ * value_head_dim_);
    conv_output_ = device_allocate<float>(capacity_ * conv_size);
    shared_gate_ = device_allocate<float>(capacity_ * shared_width_);
    shared_up_ = device_allocate<float>(capacity_ * shared_width_);
    shared_intermediate_ = device_allocate<float>(capacity_ * shared_width_);
    shared_output_ = device_allocate<float>(capacity_ * hidden_);
    shared_scalar_ = device_allocate<float>(capacity_);
    router_logits_ = device_allocate<float>(capacity_ * experts_);
    routing_scores_ = device_allocate<float>(capacity_ * top_k_);
    routing_indices_ =
        device_allocate<std::uint32_t>(capacity_ * top_k_);
    moe_intermediate_ = device_allocate<float>(
        static_cast<std::size_t>(capacity_) * top_k_ * expert_width_);
    moe_selection_output_ = device_allocate<float>(
        static_cast<std::size_t>(capacity_) * top_k_ * hidden_);
    cpu_selection_output_device_ = device_allocate<float>(
        static_cast<std::size_t>(capacity_) * top_k_ * hidden_);
    gpu_selection_mask_ =
        device_allocate<std::uint8_t>(capacity_ * top_k_);
    moe_output_ = device_allocate<float>(capacity_ * hidden_);
    logits_ = device_allocate<float>(capacity_ * vocab_);
    output_token_ = device_allocate<std::uint32_t>(capacity_);
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_normalized_),
                   static_cast<std::size_t>(capacity_) * hidden_ * sizeof(float),
                   cudaHostAllocDefault),
               "cudaHostAlloc CPU expert input");
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_routing_indices_),
                   static_cast<std::size_t>(capacity_) * top_k_ *
                       sizeof(std::uint32_t),
                   cudaHostAllocDefault),
               "cudaHostAlloc CPU route indices");
    cuda_check(cudaHostAlloc(
                   reinterpret_cast<void**>(&host_cpu_selection_output_),
                   static_cast<std::size_t>(capacity_) * top_k_ * hidden_ *
                       sizeof(float),
                   cudaHostAllocDefault),
               "cudaHostAlloc CPU expert output");
    key_cache_.resize(layers_);
    value_cache_.resize(layers_);
    conv_state_.resize(layers_);
    recurrent_state_.resize(layers_);
    const auto kv_elements = static_cast<std::size_t>(max_context_) * kv_heads_ * head_dim_;
    const auto conv_elements = conv_size * conv_kernel_;
    const auto recurrent_elements = static_cast<std::size_t>(value_heads_) *
                                    key_head_dim_ * value_head_dim_;
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      if ((layer + 1U) % full_interval_ == 0) {
        key_cache_[layer] = device_allocate<float>(capacity_ * kv_elements);
        value_cache_[layer] = device_allocate<float>(capacity_ * kv_elements);
      } else {
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
  void run_full_attention(const std::string& prefix, std::uint32_t layer,
                          std::span<const std::uint32_t> positions,
                          std::span<const std::uint32_t> state_slots,
                          std::uint32_t rows) {
    const auto query_size = 2U * query_heads_ * head_dim_;
    const auto kv_size = kv_heads_ * head_dim_;
    const auto attention_size = query_heads_ * head_dim_;
    const auto cache_stride = static_cast<std::size_t>(max_context_) * kv_size;
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "self_attn.q_proj.weight"), normalized_, query_gate_,
        rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "self_attn.k_proj.weight"), normalized_, key_, rows,
        nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "self_attn.v_proj.weight"), normalized_, value_, rows,
        nullptr));
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto state_offset =
          static_cast<std::size_t>(state_slots[row]) * cache_stride;
      status_check(expert::runtime::cuda::qwen3_next_qkv_rope_cache(
          query_gate_ + static_cast<std::size_t>(row) * query_size,
          key_ + static_cast<std::size_t>(row) * kv_size,
          value_ + static_cast<std::size_t>(row) * kv_size,
          fp32(prefix + "self_attn.q_norm.weight"),
          fp32(prefix + "self_attn.k_norm.weight"),
          key_cache_[layer] + state_offset,
          value_cache_[layer] + state_offset,
          positions[row], query_heads_, kv_heads_, head_dim_, rotary_dim_,
          epsilon_, rope_theta_, nullptr));
      status_check(expert::runtime::cuda::qwen3_next_attention_decode(
          query_gate_ + static_cast<std::size_t>(row) * query_size,
          key_cache_[layer] + state_offset,
          value_cache_[layer] + state_offset,
          attention_ + static_cast<std::size_t>(row) * attention_size,
          positions[row] + 1U, query_heads_, kv_heads_, head_dim_, nullptr));
    }
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "self_attn.o_proj.weight"), attention_, residual_, rows,
        nullptr));
  }

  void run_delta(const std::string& prefix, std::uint32_t layer,
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
        matrix(prefix + "linear_attn.in_proj_qkvz.weight"), normalized_,
        projected_qkvz_, rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "linear_attn.in_proj_ba.weight"), normalized_,
        projected_ba_, rows, nullptr));
    for (std::uint32_t row = 0; row < rows; ++row)
      status_check(expert::runtime::cuda::qwen3_next_delta_decode({
          projected_qkvz_ + static_cast<std::size_t>(row) * projected_size,
          projected_ba_ + static_cast<std::size_t>(row) * ba_size,
          fp32(prefix + "linear_attn.conv1d.weight"),
          fp32(prefix + "linear_attn.dt_bias"),
          fp32(prefix + "linear_attn.A_log"),
          fp32(prefix + "linear_attn.norm.weight"),
          conv_state_[layer] +
              static_cast<std::size_t>(state_slots[row]) * conv_state_size,
          recurrent_state_[layer] +
              static_cast<std::size_t>(state_slots[row]) * recurrent_size,
          conv_output_ + static_cast<std::size_t>(row) * conv_size,
          delta_output_ + static_cast<std::size_t>(row) * delta_size,
          key_heads_, value_heads_, key_head_dim_, value_head_dim_,
          conv_kernel_, epsilon_, nullptr}));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "linear_attn.out_proj.weight"), delta_output_,
        residual_, rows, nullptr));
  }

  void run_moe(const std::string& prefix, std::uint32_t layer,
               std::uint32_t rows) {
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "mlp.shared_expert.gate_proj.weight"), normalized_,
        shared_gate_, rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "mlp.shared_expert.up_proj.weight"), normalized_,
        shared_up_, rows, nullptr));
    status_check(expert::runtime::cuda::silu_product(
        shared_gate_, shared_up_, shared_intermediate_, rows * shared_width_,
        nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "mlp.shared_expert.down_proj.weight"),
        shared_intermediate_, shared_output_, rows, nullptr));
    status_check(expert::runtime::cuda::gemv_batch(
        matrix(prefix + "mlp.shared_expert_gate.weight"), normalized_,
        shared_scalar_, rows, nullptr));
    for (std::uint32_t row = 0; row < rows; ++row)
      status_check(expert::runtime::cuda::sigmoid_scale_in_place(
          shared_output_ + static_cast<std::size_t>(row) * hidden_,
          shared_scalar_ + row, hidden_, nullptr));
    cuda_check(cudaEventRecord(shared_done_event_), "record shared expert done");
    status_check(expert::runtime::cuda::router_topk_normalized_batch(
        normalized_, fp32(prefix + "mlp.gate.weight"), rows, hidden_, experts_,
        top_k_, router_logits_, routing_scores_, routing_indices_, nullptr));
    cuda_check(cudaEventRecord(router_done_event_), "record router done");

    const auto cache_started = std::chrono::steady_clock::now();
    std::vector<expert::runtime::ExpertLease> leases;
    std::vector<expert::runtime::HostExpertLease> host_leases;
    std::vector<expert::runtime::cpu::ExpertWorkGroup> cpu_groups;
    std::vector<std::uint32_t> cpu_experts;
    if (!placement_->frozen()) placement_->poll();
    bool route_pinned = false;
    struct PinGuard final {
      expert::runtime::cuda::CudaExpertDirectory* directory{};
      bool* active{};
      ~PinGuard() {
        if (directory != nullptr && active != nullptr && *active) {
          static_cast<void>(directory->release_pins(nullptr));
        }
      }
    } pin_guard{directory_.get(), &route_pinned};
    bool split_execution = false;
    auto plan = directory_->pin_or_collect_misses(
        layer, routing_indices_, rows * top_k_, nullptr, true);
    status_check(plan.status);
    const auto selection_count = rows * top_k_;
    const bool placement_feedback = !placement_->frozen();
    if (!plan.missing_experts.empty() || placement_feedback) {
      cuda_check(cudaMemcpy(host_routing_indices_, routing_indices_,
                            static_cast<std::size_t>(selection_count) *
                                sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "copy expert route to host");
    }
    if (placement_feedback) {
      std::fill(route_access_counts_.begin(), route_access_counts_.end(), 0U);
      route_accesses_.clear();
      for (std::uint32_t selection = 0; selection < selection_count;
           ++selection) {
        const auto expert = host_routing_indices_[selection];
        if (expert >= experts_)
          throw std::runtime_error("router expert out of range");
        ++route_access_counts_[expert];
      }
      for (std::uint32_t expert = 0; expert < experts_; ++expert) {
        if (route_access_counts_[expert] != 0) {
          route_accesses_.push_back(
              {{model_id_, layer, expert, 1}, route_access_counts_[expert]});
        }
      }
      static_cast<void>(cache_->record_accesses(route_accesses_));
    }
    if (plan.missing_experts.empty()) {
      directory_vram_hits_ += plan.unique_experts;
      route_pinned = true;
    } else {
      split_execution = true;
      route_pinned = true;
      cuda_check(cudaMemcpy(host_normalized_, normalized_,
                            static_cast<std::size_t>(rows) * hidden_ *
                                sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy CPU expert activations");
      const auto acquire_device = [&](std::span<const std::uint32_t> experts) {
        std::vector<expert::runtime::AcquireHandle> handles;
        handles.reserve(experts.size());
        for (const auto expert : experts) {
          if (expert >= experts_)
            throw std::runtime_error("router expert out of range");
          const auto& record = expert_records_.at(
              static_cast<std::size_t>(layer) * experts_ + expert);
          handles.push_back(
              cache_->acquire({model_id_, layer, expert, 1}, record));
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

      std::vector<std::uint32_t> fallback_experts;
      cpu_experts.reserve(plan.missing_experts.size());
      fallback_experts.reserve(plan.missing_experts.size());
      for (const auto expert : plan.missing_experts) {
        if (expert >= experts_)
          throw std::runtime_error("router expert out of range");
        const auto& record = expert_records_.at(
            static_cast<std::size_t>(layer) * experts_ + expert);
        auto host = cache_->try_acquire_host(
            {model_id_, layer, expert, 1}, record, false);
        if (host) {
          cpu_experts.push_back(expert);
          host_leases.push_back(std::move(*host));
        } else {
          fallback_experts.push_back(expert);
        }
      }
      directory_vram_hits_ += plan.ready_experts.size();
      if (!fallback_experts.empty()) {
        // A cold upload can invoke eviction. Mirror the retained device pins
        // with cache references so capacity selection cannot choose a pinned
        // ready entry and wait for itself.
        acquire_device(plan.ready_experts);
        acquire_device(fallback_experts);
      }

      std::unordered_map<std::uint32_t, std::size_t> cpu_group_by_expert;
      cpu_groups.reserve(cpu_experts.size());
      for (std::size_t index = 0; index < cpu_experts.size(); ++index) {
        cpu_group_by_expert.emplace(cpu_experts[index], index);
        cpu_groups.push_back({host_leases[index].bytes(),
                              host_leases[index].sections(), {}});
      }
      std::vector<std::uint8_t> gpu_mask(selection_count, 1);
      for (std::uint32_t selection = 0; selection < selection_count;
           ++selection) {
        const auto expert = host_routing_indices_[selection];
        const auto cpu_group = cpu_group_by_expert.find(expert);
        if (cpu_group != cpu_group_by_expert.end()) {
          gpu_mask[selection] = 0;
          cpu_groups[cpu_group->second].selections.push_back(selection);
        }
      }
      cuda_check(cudaMemcpy(gpu_selection_mask_, gpu_mask.data(),
                            gpu_mask.size() * sizeof(gpu_mask[0]),
                            cudaMemcpyHostToDevice),
                 "copy GPU expert selection mask");
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
    if (!split_execution) {
        status_check(expert::runtime::cuda::launch_moe_batch({
            normalized_, nullptr, nullptr, nullptr, nullptr, routing_scores_,
            routing_indices_, moe_intermediate_, moe_output_, rows, hidden_,
            expert_width_, top_k_, experts_, nullptr,
            directory_->device_entries(), layer}));
      } else {
        if (route_pinned) {
          status_check(expert::runtime::cuda::launch_moe_selection_batch({
              normalized_, routing_scores_, routing_indices_,
              gpu_selection_mask_, moe_intermediate_, moe_selection_output_,
              rows, hidden_, expert_width_, top_k_, experts_, nullptr,
              directory_->device_entries(), layer}));
        }
        if (!cpu_groups.empty()) {
          const auto cpu_started = std::chrono::steady_clock::now();
          status_check(cpu_executor_->execute(
              cpu_groups,
              std::span<const float>(host_normalized_,
                                     static_cast<std::size_t>(rows) * hidden_),
              rows, top_k_,
              std::span<float>(
                  host_cpu_selection_output_,
                  static_cast<std::size_t>(rows) * top_k_ * hidden_)));
          const auto cpu_elapsed = elapsed_ns(cpu_started);
          phase_.cpu_expert_ns += cpu_elapsed;
          phase_.cpu_expert_groups += cpu_groups.size();
          std::uint64_t cpu_selections = 0;
          for (const auto& group : cpu_groups)
            cpu_selections += group.selections.size();
          phase_.cpu_expert_selections += cpu_selections;
          if (!placement_->frozen())
            placement_->observe_cpu_batch(cpu_elapsed, cpu_selections);
          const auto output_bytes = static_cast<std::size_t>(rows) * top_k_ *
                                    hidden_ * sizeof(float);
          cuda_check(cudaMemcpyAsync(cpu_selection_output_device_,
                                     host_cpu_selection_output_, output_bytes,
                                     cudaMemcpyHostToDevice, nullptr),
                     "copy CPU expert outputs");
          phase_.cpu_result_h2d_bytes += output_bytes;
        }
        status_check(expert::runtime::cuda::launch_moe_aggregate({
            moe_selection_output_, cpu_selection_output_device_,
            gpu_selection_mask_, routing_scores_, moe_output_, rows, hidden_,
            top_k_, nullptr}));
    }
      status_check(expert::runtime::cuda::add_in_place(
          moe_output_, shared_output_, rows * hidden_, nullptr));
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state_, moe_output_, rows * hidden_, nullptr));
      if (route_pinned) {
        status_check(directory_->release_pins(nullptr));
        route_pinned = false;
      } else {
        cuda_check(cudaStreamSynchronize(nullptr),
                   "complete CPU-only expert layer");
      }
      if (!placement_->frozen()) {
        for (std::size_t index = 0; index < cpu_groups.size(); ++index) {
          const auto expert = cpu_experts.at(index);
          const auto& record = expert_records_.at(
              static_cast<std::size_t>(layer) * experts_ + expert);
          placement_->consider(
              {model_id_, layer, expert, 1}, record,
              static_cast<std::uint32_t>(cpu_groups[index].selections.size()));
        }
      }
    phase_.expert_compute_ns += elapsed_ns(expert_started);
  }

  std::filesystem::path root_;
  std::uint32_t max_context_{}, capacity_{};
  std::uint32_t hidden_{}, expert_width_{}, vocab_{}, layers_{};
  std::uint32_t query_heads_{}, kv_heads_{}, head_dim_{}, experts_{}, top_k_{};
  std::uint32_t full_interval_{}, conv_kernel_{}, key_head_dim_{},
      value_head_dim_{}, key_heads_{}, value_heads_{}, shared_width_{},
      rotary_dim_{};
  float epsilon_{}, rope_theta_{};
  std::uint64_t model_id_{0x51334e4558540001ULL};
  std::uint64_t total_pack_bytes_{}, dense_read_bytes_{},
      max_expert_record_bytes_{};
  std::uint64_t directory_vram_hits_{};
  PhaseTelemetry phase_;
  DevicePack dense_pack_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::vector<expert::runtime::PayloadRecord> expert_records_;
  std::vector<std::uint32_t> route_access_counts_;
  std::vector<expert::runtime::ExpertAccess> route_accesses_;
  std::shared_ptr<expert::runtime::WindowsIocpStorage> storage_;
  std::shared_ptr<expert::runtime::cuda::CudaExpertUploader> uploader_;
  std::shared_ptr<expert::runtime::cuda::CudaExpertDirectory> directory_;
  std::shared_ptr<expert::runtime::FixedBufferPool> buffers_;
  std::unique_ptr<expert::runtime::ExpertCache> cache_;
  std::unique_ptr<expert::runtime::cpu::ExpertExecutor> cpu_executor_;
  std::unique_ptr<expert::runtime::AdaptivePlacementPlanner> placement_;
  float *hidden_state_{}, *normalized_{}, *residual_{}, *query_gate_{}, *key_{},
      *value_{}, *attention_{}, *projected_qkvz_{}, *projected_ba_{},
      *delta_output_{}, *conv_output_{}, *shared_gate_{}, *shared_up_{},
      *shared_intermediate_{}, *shared_output_{}, *shared_scalar_{},
      *router_logits_{}, *routing_scores_{}, *moe_intermediate_{},
      *moe_selection_output_{}, *cpu_selection_output_device_{},
      *moe_output_{}, *logits_{}, *host_normalized_{},
      *host_cpu_selection_output_{};
  std::uint32_t *routing_indices_{}, *output_token_{}, *host_routing_indices_{};
  std::uint8_t* gpu_selection_mask_{};
  std::vector<float*> key_cache_, value_cache_, conv_state_, recurrent_state_;
  cudaEvent_t layer_start_event_{}, attention_done_event_{},
      shared_done_event_{}, router_done_event_{};
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

int worker_loop(Qwen3NextModel& model) {
  std::unordered_map<std::uint64_t, WorkerRequest> active;
  std::vector<bool> used_slots(model.capacity());
  std::cout << "{\"type\":\"ready\",\"protocol\":2,\"capacity\":"
            << model.capacity() << "}\n" << std::flush;
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      const auto fields = split_tabs(line);
      if (fields[0] == "PING") {
        if (fields.size() != 1) throw std::runtime_error("invalid PING");
        std::cout << "{\"type\":\"pong\"}\n" << std::flush;
      } else if (fields[0] == "BEGIN") {
        if (fields.size() != 3) throw std::runtime_error("invalid BEGIN");
        const auto request_id = std::stoull(std::string(fields[1]));
        if (!request_id || active.contains(request_id))
          throw std::runtime_error("invalid or duplicate request id");
        const auto available =
            std::find(used_slots.begin(), used_slots.end(), false);
        if (available == used_slots.end())
          throw std::runtime_error("worker request capacity exhausted");
        const auto slot = static_cast<std::uint32_t>(
            std::distance(used_slots.begin(), available));
        const auto prompt = parse_tokens(fields[2]);
        model.reset_slot(slot);
        std::uint32_t predicted = 0;
        for (std::uint32_t position = 0; position < prompt.size(); ++position) {
          const std::array token{prompt[position]}, positions{position}, slots{slot};
          predicted = model.forward_batch(token, positions, slots).front();
        }
        used_slots[slot] = true;
        active.emplace(request_id,
                       WorkerRequest{slot, predicted,
                                     static_cast<std::uint32_t>(prompt.size())});
        std::cout << "{\"type\":\"begun\",\"id\":" << request_id
                  << ",\"slot\":" << slot << "}\n" << std::flush;
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
                  << ",\"token\":" << token << "}\n" << std::flush;
        if (final) {
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
          if (!id || (flag != "0" && flag != "1") ||
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
                    << ",\"token\":" << steps[index].token << '}';
        }
        std::cout << "]}\n" << std::flush;
        for (const auto& step : steps) {
          if (step.final) {
            used_slots[active.at(step.id).slot] = false;
            active.erase(step.id);
          }
        }
      } else if (fields[0] == "END") {
        if (fields.size() != 2)
          throw std::runtime_error("END request mismatch");
        const auto id = std::stoull(std::string(fields[1]));
        const auto iterator = active.find(id);
        if (!id || iterator == active.end())
          throw std::runtime_error("END request mismatch");
        used_slots[iterator->second.slot] = false;
        active.erase(iterator);
        std::cout << "{\"type\":\"ended\",\"id\":" << id << "}\n"
                  << std::flush;
      } else if (fields[0] == "SHUTDOWN") {
        if (fields.size() != 1 || !active.empty())
          throw std::runtime_error("invalid SHUTDOWN");
        std::cout << "{\"type\":\"shutdown\"}\n" << std::flush;
        return 0;
      } else {
        throw std::runtime_error("unknown worker command");
      }
    } catch (const std::exception& error) {
      std::cerr << "worker command failed: " << error.what() << '\n';
      std::cout << "{\"type\":\"error\",\"active_requests\":"
                << active.size() << "}\n" << std::flush;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 3 && std::string_view(argv[2]) == "--batch") {
      if (argc < 4 || argc > 9)
        throw std::runtime_error(
            "batch usage: <container> --batch <token-ids-csv> [new-tokens] "
            "[concurrency] [ram-gib] [vram-gib] [warmup-rounds]");
      auto prompts = parse_prompt_batch(argv[3]);
      const auto new_tokens = argc >= 5
          ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 8U;
      const auto concurrency = argc >= 6
          ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 4U;
      const auto ram_gib = argc >= 7 ? std::stoull(argv[6]) : 48ULL;
      const auto vram_gib = argc >= 8 ? std::stoull(argv[7]) : 14ULL;
      const auto warmup_rounds = argc >= 9
          ? static_cast<std::uint32_t>(std::stoul(argv[8])) : 1U;
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
                           vram_gib << 30U, concurrency);
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
        model.reset_slot(0);
        std::uint32_t isolated_prediction = 0;
        for (std::uint32_t position = 0; position < prompts[row].size(); ++position)
          isolated_prediction = model.forward(prompts[row][position], position);
        for (std::uint32_t step = 0; step < new_tokens; ++step) {
          isolated[row].push_back(isolated_prediction);
          if (step + 1U < new_tokens)
            isolated_prediction = model.forward(
                isolated_prediction,
                static_cast<std::uint32_t>(prompts[row].size()) + step);
        }
      }
      const bool interleaving_match = isolated == generated;
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
                << ",\"warmup_rounds\":" << warmup_rounds
                << ",\"mixed_prompts\":"
                << (!identical_prompts ? "true" : "false")
                << ",\"identical_outputs\":"
                << (identical_outputs ? "true" : "false")
                << ",\"interleaving_match\":"
                << (interleaving_match ? "true" : "false")
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
      std::cout << "}\n";
      return interleaving_match ? 0 : 2;
    }
    if (argc >= 3 && std::string_view(argv[2]) == "--worker") {
      if (argc > 7)
        throw std::runtime_error(
            "worker usage: <container> --worker [max-context] [ram-gib] "
            "[vram-gib] [capacity]");
      const auto max_context = argc >= 4
          ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 4096U;
      const auto ram_gib = argc >= 5 ? std::stoull(argv[4]) : 48ULL;
      const auto vram_gib = argc >= 6 ? std::stoull(argv[5]) : 14ULL;
      const auto capacity = argc >= 7
          ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 1U;
      Qwen3NextModel model(argv[1], max_context, ram_gib << 30U,
                           vram_gib << 30U, capacity);
      return worker_loop(model);
    }
    if (argc < 3 || argc > 7) {
      std::cerr << "usage: expert-qwen3-next-runner <container> <token-ids-csv> "
                   "[new-tokens] [ram-cache-gib] [vram-cache-gib] "
                   "[warmup-rounds]\n";
      return 64;
    }
    auto tokens = parse_tokens(argv[2]);
    const auto new_tokens = argc >= 4 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 8U;
    const auto ram_gib = argc >= 5 ? std::stoull(argv[4]) : 48ULL;
    const auto vram_gib = argc >= 6 ? std::stoull(argv[5]) : 14ULL;
    const auto warmup_rounds = argc >= 7
        ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 1U;
    if (!new_tokens || !ram_gib || !vram_gib) throw std::runtime_error("zero runtime budget");
    const auto max_context = static_cast<std::uint32_t>(tokens.size()) + new_tokens;
    const auto started_load = std::chrono::steady_clock::now();
    Qwen3NextModel model(argv[1], max_context, ram_gib << 30U, vram_gib << 30U);
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
    std::cout << "],\"model_load_seconds\":" << load_seconds
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
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3-Next runner: " << error.what() << '\n';
    return 1;
  }
}
