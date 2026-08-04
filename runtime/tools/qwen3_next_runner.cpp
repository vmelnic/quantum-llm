#include "expert/core/json.hpp"
#include "expert/runtime/buffer_pool.hpp"
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
#include <unordered_map>
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

class Qwen3NextModel final {
 public:
  Qwen3NextModel(const std::filesystem::path& root, std::uint32_t max_context,
                 std::uint64_t ram_cache_bytes,
                 std::uint64_t vram_cache_bytes)
      : root_(root), max_context_(max_context) {
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

    const auto slot_bytes = static_cast<std::size_t>(max_expert_record_bytes_);
    const auto slot_count = std::max<std::size_t>(top_k_ * 2U, 32U);
    storage_ = std::make_shared<expert::runtime::WindowsIocpStorage>(4);
    uploader_ = std::make_shared<expert::runtime::cuda::CudaExpertUploader>();
    buffers_ = std::make_shared<expert::runtime::FixedBufferPool>(
        slot_count, slot_bytes, expert::runtime::kExpertPackAlignment,
        std::make_shared<expert::runtime::CudaPinnedAllocator>());
    const auto budget = [](std::uint64_t capacity) {
      if (!capacity) throw std::runtime_error("cache capacity is zero");
      return expert::runtime::TierBudget{
          capacity, capacity, capacity - std::min<std::uint64_t>(capacity / 8U,
                                                                 1ULL << 30U)};
    };
    cache_ = std::make_unique<expert::runtime::ExpertCache>(
        expert::runtime::ExpertCacheConfig{budget(ram_cache_bytes),
                                            budget(vram_cache_bytes), true},
        storage_, uploader_, buffers_);
    allocate_workspace();
  }

  std::uint32_t forward(std::uint32_t token, std::uint32_t position) {
    if (position >= max_context_ || token >= vocab_)
      throw std::runtime_error("token/position outside capacity");
    status_check(expert::runtime::cuda::embedding(
        matrix("model.embed_tokens.weight"), token, hidden_state_, nullptr));
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      const auto prefix = "model.layers." + std::to_string(layer) + ".";
      status_check(expert::runtime::cuda::qwen3_next_rms_norm(
          hidden_state_, fp32(prefix + "input_layernorm.weight"), normalized_,
          hidden_, epsilon_, nullptr));
      if ((layer + 1U) % full_interval_ == 0) {
        run_full_attention(prefix, layer, position);
      } else {
        run_delta(prefix, layer);
      }
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state_, residual_, hidden_, nullptr));
      status_check(expert::runtime::cuda::qwen3_next_rms_norm(
          hidden_state_, fp32(prefix + "post_attention_layernorm.weight"),
          normalized_, hidden_, epsilon_, nullptr));
      run_moe(prefix, layer);
    }
    status_check(expert::runtime::cuda::qwen3_next_rms_norm(
        hidden_state_, fp32("model.norm.weight"), normalized_, hidden_, epsilon_,
        nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix("lm_head.weight"), normalized_, logits_, nullptr));
    status_check(expert::runtime::cuda::argmax(logits_, vocab_, output_token_,
                                               nullptr));
    std::uint32_t result = 0;
    cuda_check(cudaMemcpy(&result, output_token_, sizeof(result),
                          cudaMemcpyDeviceToHost),
               "copy generated token");
    return result;
  }

  expert::runtime::TelemetrySnapshot telemetry() const {
    return cache_->telemetry();
  }
  std::uint64_t total_pack_bytes() const noexcept { return total_pack_bytes_; }
  std::uint64_t dense_read_bytes() const noexcept { return dense_read_bytes_; }

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
    hidden_state_ = device_allocate<float>(hidden_);
    normalized_ = device_allocate<float>(hidden_);
    residual_ = device_allocate<float>(hidden_);
    query_gate_ = device_allocate<float>(query_size);
    key_ = device_allocate<float>(key_value_size);
    value_ = device_allocate<float>(key_value_size);
    attention_ = device_allocate<float>(query_heads_ * head_dim_);
    projected_qkvz_ = device_allocate<float>(projected_size);
    projected_ba_ = device_allocate<float>(2U * value_heads_);
    delta_output_ = device_allocate<float>(value_heads_ * value_head_dim_);
    conv_output_ = device_allocate<float>(conv_size);
    shared_gate_ = device_allocate<float>(shared_width_);
    shared_up_ = device_allocate<float>(shared_width_);
    shared_intermediate_ = device_allocate<float>(shared_width_);
    shared_output_ = device_allocate<float>(hidden_);
    shared_scalar_ = device_allocate<float>(1);
    router_logits_ = device_allocate<float>(experts_);
    routing_scores_ = device_allocate<float>(top_k_);
    routing_indices_ = device_allocate<std::uint32_t>(top_k_);
    moe_intermediate_ = device_allocate<float>(
        static_cast<std::size_t>(top_k_) * expert_width_);
    moe_output_ = device_allocate<float>(hidden_);
    logits_ = device_allocate<float>(vocab_);
    output_token_ = device_allocate<std::uint32_t>(1);
    d_gate_up_ = device_allocate<const std::int8_t*>(top_k_);
    d_gate_scales_ = device_allocate<const float*>(top_k_);
    d_down_ = device_allocate<const std::int8_t*>(top_k_);
    d_down_scales_ = device_allocate<const float*>(top_k_);

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
        key_cache_[layer] = device_allocate<float>(kv_elements);
        value_cache_[layer] = device_allocate<float>(kv_elements);
      } else {
        conv_state_[layer] = device_allocate<float>(conv_elements);
        recurrent_state_[layer] = device_allocate<float>(recurrent_elements);
        cuda_check(cudaMemset(conv_state_[layer], 0,
                              conv_elements * sizeof(float)),
                   "zero delta conv state");
        cuda_check(cudaMemset(recurrent_state_[layer], 0,
                              recurrent_elements * sizeof(float)),
                   "zero delta recurrent state");
      }
    }
  }
  void run_full_attention(const std::string& prefix, std::uint32_t layer,
                          std::uint32_t position) {
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "self_attn.q_proj.weight"), normalized_, query_gate_,
        nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "self_attn.k_proj.weight"), normalized_, key_, nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "self_attn.v_proj.weight"), normalized_, value_, nullptr));
    status_check(expert::runtime::cuda::qwen3_next_qkv_rope_cache(
        query_gate_, key_, value_, fp32(prefix + "self_attn.q_norm.weight"),
        fp32(prefix + "self_attn.k_norm.weight"), key_cache_[layer],
        value_cache_[layer], position, query_heads_, kv_heads_, head_dim_,
        rotary_dim_, epsilon_, rope_theta_, nullptr));
    status_check(expert::runtime::cuda::qwen3_next_attention_decode(
        query_gate_, key_cache_[layer], value_cache_[layer], attention_,
        position + 1U, query_heads_, kv_heads_, head_dim_, nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "self_attn.o_proj.weight"), attention_, residual_,
        nullptr));
  }
  void run_delta(const std::string& prefix, std::uint32_t layer) {
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "linear_attn.in_proj_qkvz.weight"), normalized_,
        projected_qkvz_, nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "linear_attn.in_proj_ba.weight"), normalized_,
        projected_ba_, nullptr));
    status_check(expert::runtime::cuda::qwen3_next_delta_decode({
        projected_qkvz_, projected_ba_, fp32(prefix + "linear_attn.conv1d.weight"),
        fp32(prefix + "linear_attn.dt_bias"), fp32(prefix + "linear_attn.A_log"),
        fp32(prefix + "linear_attn.norm.weight"), conv_state_[layer],
        recurrent_state_[layer], conv_output_, delta_output_, key_heads_,
        value_heads_, key_head_dim_, value_head_dim_, conv_kernel_, epsilon_,
        nullptr}));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "linear_attn.out_proj.weight"), delta_output_, residual_,
        nullptr));
  }
  void run_moe(const std::string& prefix, std::uint32_t layer) {
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "mlp.shared_expert.gate_proj.weight"), normalized_,
        shared_gate_, nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "mlp.shared_expert.up_proj.weight"), normalized_,
        shared_up_, nullptr));
    status_check(expert::runtime::cuda::silu_product(
        shared_gate_, shared_up_, shared_intermediate_, shared_width_, nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "mlp.shared_expert.down_proj.weight"),
        shared_intermediate_, shared_output_, nullptr));
    status_check(expert::runtime::cuda::gemv(
        matrix(prefix + "mlp.shared_expert_gate.weight"), normalized_,
        shared_scalar_, nullptr));
    status_check(expert::runtime::cuda::sigmoid_scale_in_place(
        shared_output_, shared_scalar_, hidden_, nullptr));
    status_check(expert::runtime::cuda::router_topk_normalized(
        normalized_, fp32(prefix + "mlp.gate.weight"), hidden_, experts_, top_k_,
        router_logits_, routing_scores_, routing_indices_, nullptr));

    std::vector<std::uint32_t> selected(top_k_);
    cuda_check(cudaMemcpy(selected.data(), routing_indices_,
                          selected.size() * sizeof(selected[0]),
                          cudaMemcpyDeviceToHost),
               "copy selected expert ids");
    std::vector<expert::runtime::AcquireHandle> handles;
    handles.reserve(top_k_);
    for (const auto expert : selected) {
      const auto& record = expert_records_.at(
          static_cast<std::size_t>(layer) * experts_ + expert);
      handles.push_back(cache_->acquire({model_id_, layer, expert, 1}, record));
    }
    std::vector<expert::runtime::ExpertLease> leases;
    std::vector<const std::int8_t*> gate(top_k_), down(top_k_);
    std::vector<const float*> gate_scales(top_k_), down_scales(top_k_);
    leases.reserve(top_k_);
    for (std::uint32_t slot = 0; slot < top_k_; ++slot) {
      auto acquired = handles[slot].get();
      if (!acquired.status.ok())
        throw std::runtime_error("expert acquire failed: " +
                                 std::string(acquired.status.message()));
      leases.push_back(std::move(acquired.lease));
      const auto* allocation =
          dynamic_cast<const expert::runtime::cuda::CudaExpertAllocation*>(
              leases.back().get());
      if (!allocation) throw std::runtime_error("unexpected expert allocation");
      gate[slot] = allocation->gate_up();
      gate_scales[slot] = allocation->gate_up_scales();
      down[slot] = allocation->down();
      down_scales[slot] = allocation->down_scales();
    }
    cuda_check(cudaMemcpy(d_gate_up_, gate.data(), top_k_ * sizeof(gate[0]),
                          cudaMemcpyHostToDevice), "copy gate pointers");
    cuda_check(cudaMemcpy(d_gate_scales_, gate_scales.data(),
                          top_k_ * sizeof(gate_scales[0]), cudaMemcpyHostToDevice),
               "copy gate scale pointers");
    cuda_check(cudaMemcpy(d_down_, down.data(), top_k_ * sizeof(down[0]),
                          cudaMemcpyHostToDevice), "copy down pointers");
    cuda_check(cudaMemcpy(d_down_scales_, down_scales.data(),
                          top_k_ * sizeof(down_scales[0]), cudaMemcpyHostToDevice),
               "copy down scale pointers");
    status_check(expert::runtime::cuda::launch_moe_single_token({
        normalized_, d_gate_up_, d_gate_scales_, d_down_, d_down_scales_,
        routing_scores_, nullptr, moe_intermediate_, moe_output_, hidden_,
        expert_width_, top_k_, top_k_, nullptr}));
    status_check(expert::runtime::cuda::add_in_place(
        moe_output_, shared_output_, hidden_, nullptr));
    status_check(expert::runtime::cuda::add_in_place(
        hidden_state_, moe_output_, hidden_, nullptr));
    // Leases protect pointers until all kernels that consume them complete.
    cuda_check(cudaDeviceSynchronize(), "complete expert layer");
  }

  std::filesystem::path root_;
  std::uint32_t max_context_{};
  std::uint32_t hidden_{}, expert_width_{}, vocab_{}, layers_{};
  std::uint32_t query_heads_{}, kv_heads_{}, head_dim_{}, experts_{}, top_k_{};
  std::uint32_t full_interval_{}, conv_kernel_{}, key_head_dim_{},
      value_head_dim_{}, key_heads_{}, value_heads_{}, shared_width_{},
      rotary_dim_{};
  float epsilon_{}, rope_theta_{};
  std::uint64_t model_id_{0x51334e4558540001ULL};
  std::uint64_t total_pack_bytes_{}, dense_read_bytes_{},
      max_expert_record_bytes_{};
  DevicePack dense_pack_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::vector<expert::runtime::PayloadRecord> expert_records_;
  std::shared_ptr<expert::runtime::WindowsIocpStorage> storage_;
  std::shared_ptr<expert::runtime::cuda::CudaExpertUploader> uploader_;
  std::shared_ptr<expert::runtime::FixedBufferPool> buffers_;
  std::unique_ptr<expert::runtime::ExpertCache> cache_;
  float *hidden_state_{}, *normalized_{}, *residual_{}, *query_gate_{}, *key_{},
      *value_{}, *attention_{}, *projected_qkvz_{}, *projected_ba_{},
      *delta_output_{}, *conv_output_{}, *shared_gate_{}, *shared_up_{},
      *shared_intermediate_{}, *shared_output_{}, *shared_scalar_{},
      *router_logits_{}, *routing_scores_{}, *moe_intermediate_{},
      *moe_output_{}, *logits_{};
  std::uint32_t *routing_indices_{}, *output_token_{};
  const std::int8_t **d_gate_up_{}, **d_down_{};
  const float **d_gate_scales_{}, **d_down_scales_{};
  std::vector<float*> key_cache_, value_cache_, conv_state_, recurrent_state_;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 6) {
      std::cerr << "usage: expert-qwen3-next-runner <container> <token-ids-csv> "
                   "[new-tokens] [ram-cache-gib] [vram-cache-gib]\n";
      return 64;
    }
    auto tokens = parse_tokens(argv[2]);
    const auto new_tokens = argc >= 4 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 8U;
    const auto ram_gib = argc >= 5 ? std::stoull(argv[4]) : 48ULL;
    const auto vram_gib = argc >= 6 ? std::stoull(argv[5]) : 14ULL;
    if (!new_tokens || !ram_gib || !vram_gib) throw std::runtime_error("zero runtime budget");
    const auto max_context = static_cast<std::uint32_t>(tokens.size()) + new_tokens;
    const auto started_load = std::chrono::steady_clock::now();
    Qwen3NextModel model(argv[1], max_context, ram_gib << 30U, vram_gib << 30U);
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_load).count();
    std::uint32_t predicted = 0;
    const auto started_prompt = std::chrono::steady_clock::now();
    for (std::uint32_t position = 0; position < tokens.size(); ++position)
      predicted = model.forward(tokens[position], position);
    const auto prompt_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_prompt).count();
    const auto started_decode = std::chrono::steady_clock::now();
    for (std::uint32_t generated = 0; generated < new_tokens; ++generated) {
      tokens.push_back(predicted);
      if (generated + 1U < new_tokens)
        predicted = model.forward(
            predicted, static_cast<std::uint32_t>(tokens.size() - 1U));
    }
    const auto decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_decode).count();
    const auto metrics = model.telemetry();
    const auto forwards = new_tokens > 0 ? new_tokens - 1U : 0U;
    std::cout << "{\"tokens\":[";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << tokens[i];
    }
    std::cout << "],\"model_load_seconds\":" << load_seconds
              << ",\"prompt_seconds\":" << prompt_seconds
              << ",\"decode_seconds\":" << decode_seconds
              << ",\"tokens_per_second\":"
              << (forwards ? forwards / decode_seconds : 0.0)
              << ",\"container_bytes\":" << model.total_pack_bytes()
              << ",\"startup_dense_read_bytes\":" << model.dense_read_bytes()
              << ",\"expert_read_bytes\":" << metrics.read_bytes
              << ",\"expert_h2d_bytes\":" << metrics.uploaded_bytes
              << ",\"expert_loads\":" << metrics.load_completed
              << ",\"expert_deduplicated\":" << metrics.load_deduplicated
              << ",\"ram_high_water\":" << metrics.ram_high_water
              << ",\"vram_high_water\":" << metrics.vram_high_water
              << ",\"evictions\":" << metrics.eviction_count << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3-Next runner: " << error.what() << '\n';
    return 1;
  }
}
