#include "expert/core/json.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/sha256.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
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
  if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
void status_check(const expert::runtime::Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
template <typename T> T* device_allocate(std::size_t count) {
  void* raw = nullptr;
  cuda_check(cudaMalloc(&raw, count * sizeof(T)), "cudaMalloc");
  return reinterpret_cast<T*>(raw);
}
std::string read_text(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream output; output << stream.rdbuf(); return output.str();
}
double number(const Value& value) {
  if (const auto* v = std::get_if<std::int64_t>(&value.data)) return static_cast<double>(*v);
  if (const auto* v = std::get_if<std::uint64_t>(&value.data)) return static_cast<double>(*v);
  if (const auto* v = std::get_if<Value::Number>(&value.data)) return std::stod(v->token);
  throw std::runtime_error("JSON value is not numeric");
}
std::uint32_t u32(const Value::Object& object, std::string_view key, std::string_view where) {
  const auto value = Required(object, key, where).AsU64(where);
  if (value > 0xffffffffULL) throw std::runtime_error("integer exceeds u32");
  return static_cast<std::uint32_t>(value);
}

struct DevicePack { std::byte* base{}; std::uint64_t bytes{}; };
expert::runtime::Sha256Digest parse_digest(std::string_view text) {
  if (text.size() != 64) throw std::runtime_error("invalid SHA-256 text length");
  expert::runtime::Sha256Digest result{};
  const auto nibble = [](char value) -> unsigned {
    if (value >= '0' && value <= '9') return static_cast<unsigned>(value - '0');
    if (value >= 'a' && value <= 'f') return static_cast<unsigned>(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return static_cast<unsigned>(value - 'A' + 10);
    throw std::runtime_error("invalid SHA-256 hex digit");
  };
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<std::byte>((nibble(text[2 * i]) << 4U) | nibble(text[2 * i + 1]));
  }
  return result;
}
DevicePack upload_pack(const std::filesystem::path& path, std::uint64_t expected,
                       std::string_view expected_sha256) {
  const auto actual = std::filesystem::file_size(path);
  if (actual != expected) throw std::runtime_error("pack size mismatch: " + path.string());
  DevicePack pack{device_allocate<std::byte>(static_cast<std::size_t>(actual)), actual};
  std::ifstream stream(path, std::ios::binary);
  constexpr std::size_t chunk_bytes = 64U * 1024U * 1024U;
  std::vector<std::byte> chunk(chunk_bytes);
  expert::runtime::Sha256 hasher;
  std::uint64_t offset = 0;
  while (offset < actual) {
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(), actual - offset));
    stream.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(count));
    if (stream.gcount() != static_cast<std::streamsize>(count)) throw std::runtime_error("short pack read");
    hasher.update(std::span<const std::byte>(chunk.data(), count));
    cuda_check(cudaMemcpy(pack.base + offset, chunk.data(), count, cudaMemcpyHostToDevice), "upload pack");
    offset += count;
  }
  if (!expert::runtime::constant_time_equal(hasher.finalize(), parse_digest(expected_sha256)))
    throw std::runtime_error("pack SHA-256 mismatch: " + path.string());
  return pack;
}

struct Tensor {
  expert::runtime::cuda::Int8Matrix int8;
  const float* f32{};
  std::vector<std::uint32_t> shape;
  bool quantized{};
};

class Model {
 public:
  Model(const std::filesystem::path& root, std::uint32_t max_tokens) : root_(root), max_tokens_(max_tokens) {
    const auto document = expert::core::json::Parse(read_text(root / "manifest.json"));
    const auto& manifest = document.AsObject("manifest");
    const auto& architecture = Required(manifest, "architecture", "manifest").AsObject("architecture");
    hidden = u32(architecture, "hidden_size", "architecture");
    intermediate = u32(architecture, "intermediate_size", "architecture");
    vocab = u32(architecture, "vocab_size", "architecture");
    layers = u32(architecture, "num_hidden_layers", "architecture");
    heads = u32(architecture, "num_attention_heads", "architecture");
    kv_heads = u32(architecture, "num_key_value_heads", "architecture");
    head_dim = u32(architecture, "head_dim", "architecture");
    experts = u32(architecture, "num_experts", "architecture");
    top_k = u32(architecture, "num_experts_per_token", "architecture");
    epsilon = static_cast<float>(number(Required(architecture, "rms_norm_epsilon", "architecture")));
    const auto& rope = Required(architecture, "rope", "architecture").AsObject("rope");
    rope_theta = static_cast<float>(number(Required(rope, "theta", "rope")));
    if (hidden != heads * head_dim || heads != kv_heads || head_dim != 128 || top_k > 64)
      throw std::runtime_error("P3 OLMoE runner requires MHA, head_dim 128, top-k <=64");
    if (!std::holds_alternative<std::nullptr_t>(Required(architecture, "clip_qkv", "architecture").data))
      throw std::runtime_error("clip_qkv is not implemented");

    const auto& pack_entries = Required(manifest, "packs", "manifest").AsArray("packs");
    for (const auto& value : pack_entries) {
      const auto& entry = value.AsObject("pack");
      const auto name = Required(entry, "name", "pack").AsString("pack.name");
      const auto bytes = Required(entry, "bytes", "pack").AsU64("pack.bytes");
      packs_.emplace(name, upload_pack(
          root / name, bytes, Required(entry, "sha256", "pack").AsString("pack.sha256")));
      pack_bytes += bytes;
    }
    const auto& dense_entries = Required(manifest, "tensors", "manifest").AsArray("tensors");
    for (const auto& value : dense_entries) add_tensor(value.AsObject("tensor"));
    build_expert_tables(Required(manifest, "experts", "manifest").AsArray("experts"));
    allocate_workspace();
  }

  std::uint32_t forward(std::uint32_t token, std::uint32_t position) {
    status_check(expert::runtime::cuda::embedding(matrix("model.embed_tokens.weight"), token, hidden_state, nullptr));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      const auto prefix = "model.layers." + std::to_string(layer) + ".";
      status_check(expert::runtime::cuda::rms_norm(hidden_state, fp32(prefix + "input_layernorm.weight"), normalized, hidden, epsilon, nullptr));
      status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.q_proj.weight"), normalized, query, nullptr));
      status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.k_proj.weight"), normalized, key, nullptr));
      status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.v_proj.weight"), normalized, value, nullptr));
      status_check(expert::runtime::cuda::qkv_rope_cache(query, key, value, fp32(prefix + "self_attn.q_norm.weight"), fp32(prefix + "self_attn.k_norm.weight"), key_cache[layer], value_cache[layer], position, heads, head_dim, epsilon, rope_theta, nullptr));
      status_check(expert::runtime::cuda::attention_decode(query, key_cache[layer], value_cache[layer], attention, position + 1, heads, head_dim, nullptr));
      status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.o_proj.weight"), attention, residual, nullptr));
      status_check(expert::runtime::cuda::add_in_place(hidden_state, residual, hidden, nullptr));
      status_check(expert::runtime::cuda::rms_norm(hidden_state, fp32(prefix + "post_attention_layernorm.weight"), normalized, hidden, epsilon, nullptr));
      status_check(expert::runtime::cuda::router_topk(normalized, fp32(prefix + "mlp.gate.weight"), hidden, experts, top_k, router_logits, routing_scores, routing_indices, nullptr));
      expert::runtime::cuda::MoeLaunch launch{normalized, d_gate_up + static_cast<std::size_t>(layer) * experts, d_gate_scales + static_cast<std::size_t>(layer) * experts, d_down + static_cast<std::size_t>(layer) * experts, d_down_scales + static_cast<std::size_t>(layer) * experts, routing_scores, routing_indices, moe_intermediate, residual, hidden, intermediate, top_k, experts, nullptr};
      status_check(expert::runtime::cuda::launch_moe_single_token(launch));
      status_check(expert::runtime::cuda::add_in_place(hidden_state, residual, hidden, nullptr));
    }
    status_check(expert::runtime::cuda::rms_norm(hidden_state, fp32("model.norm.weight"), normalized, hidden, epsilon, nullptr));
    status_check(expert::runtime::cuda::gemv(matrix("lm_head.weight"), normalized, output_logits, nullptr));
    status_check(expert::runtime::cuda::argmax(output_logits, vocab, output_token, nullptr));
    std::uint32_t result{};
    cuda_check(cudaMemcpy(&result, output_token, sizeof(result), cudaMemcpyDeviceToHost), "copy output token");
    return result;
  }

  std::vector<std::pair<std::uint32_t, float>> top_logits(std::size_t count) const {
    std::vector<float> logits(vocab);
    cuda_check(cudaMemcpy(logits.data(), output_logits, logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy output logits");
    std::vector<std::uint32_t> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0U);
    count = std::min(count, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                      [&](std::uint32_t left, std::uint32_t right) {
                        return logits[left] > logits[right];
                      });
    std::vector<std::pair<std::uint32_t, float>> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) result.emplace_back(indices[i], logits[indices[i]]);
    return result;
  }

  std::uint32_t hidden{}, intermediate{}, vocab{}, layers{}, heads{}, kv_heads{}, head_dim{}, experts{}, top_k{};
  std::uint64_t pack_bytes{};
  float epsilon{}, rope_theta{};

 private:
  const expert::runtime::cuda::Int8Matrix& matrix(const std::string& name) const {
    const auto& tensor = tensors_.at(name); if (!tensor.quantized) throw std::runtime_error(name + " is not INT8"); return tensor.int8;
  }
  const float* fp32(const std::string& name) const {
    const auto& tensor = tensors_.at(name); if (tensor.quantized) throw std::runtime_error(name + " is not F32"); return tensor.f32;
  }
  void add_tensor(const Value::Object& entry) {
    const auto name = Required(entry, "name", "tensor").AsString("tensor.name");
    const auto pack_name = Required(entry, "pack", "tensor").AsString("tensor.pack");
    const auto record_offset = Required(entry, "offset", "tensor").AsU64("tensor.offset");
    const auto& sections = Required(entry, "sections", "tensor").AsObject("tensor.sections");
    const auto& data = Required(sections, "data", "sections").AsObject("data");
    const auto& scales = Required(sections, "scales", "sections").AsObject("scales");
    Tensor tensor;
    for (const auto& dim : Required(entry, "source_shape", "tensor").AsArray("shape")) tensor.shape.push_back(static_cast<std::uint32_t>(dim.AsU64("shape")));
    auto* data_pointer = packs_.at(pack_name).base + record_offset + Required(data, "offset", "data").AsU64("data.offset");
    tensor.quantized = Required(entry, "stored_dtype", "tensor").AsString("dtype") == "I8";
    if (tensor.quantized) {
      tensor.int8.weights = reinterpret_cast<const std::int8_t*>(data_pointer);
      tensor.int8.rows = tensor.shape.at(0);
      tensor.int8.columns = tensor.shape.size() == 1 ? 1 : tensor.shape.at(1);
      tensor.int8.scales = reinterpret_cast<const float*>(packs_.at(pack_name).base + record_offset + Required(scales, "offset", "scales").AsU64("scales.offset"));
    } else tensor.f32 = reinterpret_cast<const float*>(data_pointer);
    tensors_.emplace(name, std::move(tensor));
  }
  void build_expert_tables(const Value::Array& entries) {
    const auto count = static_cast<std::size_t>(layers) * experts;
    std::vector<const std::int8_t*> gate(count), down(count);
    std::vector<const float*> gate_scale(count), down_scale(count);
    for (const auto& value : entries) {
      const auto& entry = value.AsObject("expert");
      const auto layer = u32(entry, "layer", "expert"), expert = u32(entry, "expert", "expert");
      const auto index = static_cast<std::size_t>(layer) * experts + expert;
      const auto pack_name = Required(entry, "pack", "expert").AsString("expert.pack");
      const auto record = Required(entry, "offset", "expert").AsU64("expert.offset");
      const auto& sections = Required(entry, "sections", "expert").AsObject("sections");
      const auto pointer = [&](std::string_view section_name) {
        const auto& section = Required(sections, section_name, "sections").AsObject("section");
        return packs_.at(pack_name).base + record + Required(section, "offset", "section").AsU64("section.offset");
      };
      gate[index] = reinterpret_cast<const std::int8_t*>(pointer("gate_up_q"));
      gate_scale[index] = reinterpret_cast<const float*>(pointer("gate_up_scales"));
      down[index] = reinterpret_cast<const std::int8_t*>(pointer("down_q"));
      down_scale[index] = reinterpret_cast<const float*>(pointer("down_scales"));
    }
    d_gate_up = device_allocate<const std::int8_t*>(count); d_gate_scales = device_allocate<const float*>(count);
    d_down = device_allocate<const std::int8_t*>(count); d_down_scales = device_allocate<const float*>(count);
    cuda_check(cudaMemcpy(d_gate_up, gate.data(), count * sizeof(gate[0]), cudaMemcpyHostToDevice), "copy expert table");
    cuda_check(cudaMemcpy(d_gate_scales, gate_scale.data(), count * sizeof(gate_scale[0]), cudaMemcpyHostToDevice), "copy expert scales");
    cuda_check(cudaMemcpy(d_down, down.data(), count * sizeof(down[0]), cudaMemcpyHostToDevice), "copy down table");
    cuda_check(cudaMemcpy(d_down_scales, down_scale.data(), count * sizeof(down_scale[0]), cudaMemcpyHostToDevice), "copy down scales");
  }
  void allocate_workspace() {
    hidden_state = device_allocate<float>(hidden); normalized = device_allocate<float>(hidden);
    query = device_allocate<float>(hidden); key = device_allocate<float>(hidden); value = device_allocate<float>(hidden);
    attention = device_allocate<float>(hidden); residual = device_allocate<float>(hidden);
    router_logits = device_allocate<float>(experts); routing_scores = device_allocate<float>(top_k);
    routing_indices = device_allocate<std::uint32_t>(top_k); moe_intermediate = device_allocate<float>(static_cast<std::size_t>(top_k) * intermediate);
    output_logits = device_allocate<float>(vocab); output_token = device_allocate<std::uint32_t>(1);
    key_cache.resize(layers); value_cache.resize(layers);
    const auto cache_elements = static_cast<std::size_t>(max_tokens_) * hidden;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      key_cache[layer] = device_allocate<float>(cache_elements); value_cache[layer] = device_allocate<float>(cache_elements);
    }
  }
  std::filesystem::path root_; std::uint32_t max_tokens_{};
  std::unordered_map<std::string, DevicePack> packs_;
  std::unordered_map<std::string, Tensor> tensors_;
  const std::int8_t** d_gate_up{}; const float** d_gate_scales{};
  const std::int8_t** d_down{}; const float** d_down_scales{};
  float *hidden_state{}, *normalized{}, *query{}, *key{}, *value{}, *attention{}, *residual{};
  float *router_logits{}, *routing_scores{}, *moe_intermediate{}, *output_logits{};
  std::uint32_t *routing_indices{}, *output_token{};
  std::vector<float*> key_cache, value_cache;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4) { std::cerr << "usage: expert-olmoe-runner <container> [new-tokens] [--trace-logits]\n"; return 64; }
    const auto new_tokens = argc >= 3 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 12U;
    const bool trace_logits = argc == 4 && std::string_view(argv[3]) == "--trace-logits";
    if (argc == 4 && !trace_logits) throw std::runtime_error("unknown option");
    const std::vector<std::uint32_t> prompt{510, 5347, 273, 6181, 310};
    const auto load_started = std::chrono::steady_clock::now();
    Model model(argv[1], static_cast<std::uint32_t>(prompt.size()) + new_tokens);
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_started).count();
    std::vector<std::uint32_t> full = prompt;
    std::uint32_t predicted = 0;
    const auto prompt_started = std::chrono::steady_clock::now();
    for (std::uint32_t position = 0; position < prompt.size(); ++position) predicted = model.forward(prompt[position], position);
    const auto prompt_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prompt_started).count();
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t generated = 0; generated < new_tokens; ++generated) {
      full.push_back(predicted);
      if (trace_logits) {
        std::cerr << "logits position=" << (prompt.size() + generated) << " top=";
        const auto top = model.top_logits(5);
        for (std::size_t i = 0; i < top.size(); ++i) {
          if (i) std::cerr << ',';
          std::cerr << top[i].first << ':' << top[i].second;
        }
        std::cerr << '\n';
      }
      if (generated + 1 < new_tokens) predicted = model.forward(predicted, static_cast<std::uint32_t>(prompt.size()) + generated);
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto decode_forwards = new_tokens > 0 ? new_tokens - 1U : 0U;
    const auto tokens_per_second = decode_forwards > 0 ? decode_forwards / seconds : 0.0;
    std::cout << "{\"tokens\":[";
    for (std::size_t i = 0; i < full.size(); ++i) { if (i) std::cout << ','; std::cout << full[i]; }
    std::cout << "],\"generated\":" << new_tokens
              << ",\"model_load_seconds\":" << load_seconds
              << ",\"startup_pack_read_bytes\":" << model.pack_bytes
              << ",\"startup_pack_h2d_bytes\":" << model.pack_bytes
              << ",\"hot_storage_read_bytes\":0,\"hot_h2d_bytes\":0"
              << ",\"prompt_tokens\":" << prompt.size()
              << ",\"prompt_seconds\":" << prompt_seconds
              << ",\"decode_forward_tokens\":" << decode_forwards
              << ",\"decode_seconds\":" << seconds
              << ",\"tokens_per_second\":" << tokens_per_second
              << ",\"trace_logits\":" << (trace_logits ? "true" : "false") << "}\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << "olmoe runner: " << error.what() << '\n'; return 1; }
}
