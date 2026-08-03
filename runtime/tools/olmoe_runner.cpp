#include "expert/core/json.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/sha256.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
  Model(const std::filesystem::path& root, std::uint32_t max_tokens,
        std::uint32_t capacity = 1)
      : root_(root), max_tokens_(max_tokens), capacity_(capacity) {
    if (capacity_ == 0) throw std::runtime_error("request capacity must be positive");
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
    const std::array tokens{token};
    const std::array positions{position};
    return forward_batch(tokens, positions).front();
  }

  std::vector<std::uint32_t> forward_batch(
      std::span<const std::uint32_t> tokens,
      std::span<const std::uint32_t> positions) {
    if (tokens.empty() || tokens.size() != positions.size() ||
        tokens.size() > capacity_)
      throw std::runtime_error("invalid request microbatch");
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
      if (positions[row] >= max_tokens_) throw std::runtime_error("context capacity exceeded");
      status_check(expert::runtime::cuda::embedding(
          matrix("model.embed_tokens.weight"), tokens[row],
          hidden_state + static_cast<std::size_t>(row) * hidden, nullptr));
    }
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      const auto prefix = "model.layers." + std::to_string(layer) + ".";
      for (std::uint32_t row = 0; row < rows; ++row) {
        const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
        auto* row_hidden = hidden_state + hidden_offset;
        auto* row_normalized = normalized + hidden_offset;
        auto* row_query = query + hidden_offset;
        auto* row_key = key + hidden_offset;
        auto* row_value = value + hidden_offset;
        auto* row_attention = attention + hidden_offset;
        auto* row_residual = residual + hidden_offset;
        const auto cache_offset = static_cast<std::size_t>(row) * max_tokens_ * hidden;
        status_check(expert::runtime::cuda::rms_norm(row_hidden, fp32(prefix + "input_layernorm.weight"), row_normalized, hidden, epsilon, nullptr));
        status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.q_proj.weight"), row_normalized, row_query, nullptr));
        status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.k_proj.weight"), row_normalized, row_key, nullptr));
        status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.v_proj.weight"), row_normalized, row_value, nullptr));
        status_check(expert::runtime::cuda::qkv_rope_cache(row_query, row_key, row_value, fp32(prefix + "self_attn.q_norm.weight"), fp32(prefix + "self_attn.k_norm.weight"), key_cache[layer] + cache_offset, value_cache[layer] + cache_offset, positions[row], heads, head_dim, epsilon, rope_theta, nullptr));
        status_check(expert::runtime::cuda::attention_decode(row_query, key_cache[layer] + cache_offset, value_cache[layer] + cache_offset, row_attention, positions[row] + 1, heads, head_dim, nullptr));
        status_check(expert::runtime::cuda::gemv(matrix(prefix + "self_attn.o_proj.weight"), row_attention, row_residual, nullptr));
        status_check(expert::runtime::cuda::add_in_place(row_hidden, row_residual, hidden, nullptr));
        status_check(expert::runtime::cuda::rms_norm(row_hidden, fp32(prefix + "post_attention_layernorm.weight"), row_normalized, hidden, epsilon, nullptr));
        status_check(expert::runtime::cuda::router_topk(row_normalized, fp32(prefix + "mlp.gate.weight"), hidden, experts, top_k, router_logits + static_cast<std::size_t>(row) * experts, routing_scores + static_cast<std::size_t>(row) * top_k, routing_indices + static_cast<std::size_t>(row) * top_k, nullptr));
      }
      expert::runtime::cuda::MoeBatchLaunch launch{
          normalized, d_gate_up + static_cast<std::size_t>(layer) * experts,
          d_gate_scales + static_cast<std::size_t>(layer) * experts,
          d_down + static_cast<std::size_t>(layer) * experts,
          d_down_scales + static_cast<std::size_t>(layer) * experts,
          routing_scores, routing_indices, moe_intermediate, residual, rows,
          hidden, intermediate, top_k, experts, nullptr};
      status_check(expert::runtime::cuda::launch_moe_batch(launch));
      for (std::uint32_t row = 0; row < rows; ++row)
        status_check(expert::runtime::cuda::add_in_place(
            hidden_state + static_cast<std::size_t>(row) * hidden,
            residual + static_cast<std::size_t>(row) * hidden, hidden, nullptr));
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
      const auto logits_offset = static_cast<std::size_t>(row) * vocab;
      status_check(expert::runtime::cuda::rms_norm(hidden_state + hidden_offset, fp32("model.norm.weight"), normalized + hidden_offset, hidden, epsilon, nullptr));
      status_check(expert::runtime::cuda::gemv(matrix("lm_head.weight"), normalized + hidden_offset, output_logits + logits_offset, nullptr));
      status_check(expert::runtime::cuda::argmax(output_logits + logits_offset, vocab, output_token + row, nullptr));
    }
    std::vector<std::uint32_t> result(rows);
    cuda_check(cudaMemcpy(result.data(), output_token, result.size() * sizeof(result[0]), cudaMemcpyDeviceToHost), "copy output tokens");
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
    const auto hidden_rows = static_cast<std::size_t>(capacity_) * hidden;
    hidden_state = device_allocate<float>(hidden_rows); normalized = device_allocate<float>(hidden_rows);
    query = device_allocate<float>(hidden_rows); key = device_allocate<float>(hidden_rows); value = device_allocate<float>(hidden_rows);
    attention = device_allocate<float>(hidden_rows); residual = device_allocate<float>(hidden_rows);
    router_logits = device_allocate<float>(static_cast<std::size_t>(capacity_) * experts);
    routing_scores = device_allocate<float>(static_cast<std::size_t>(capacity_) * top_k);
    routing_indices = device_allocate<std::uint32_t>(static_cast<std::size_t>(capacity_) * top_k);
    moe_intermediate = device_allocate<float>(static_cast<std::size_t>(capacity_) * top_k * intermediate);
    output_logits = device_allocate<float>(static_cast<std::size_t>(capacity_) * vocab);
    output_token = device_allocate<std::uint32_t>(capacity_);
    key_cache.resize(layers); value_cache.resize(layers);
    const auto cache_elements = static_cast<std::size_t>(capacity_) * max_tokens_ * hidden;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
      key_cache[layer] = device_allocate<float>(cache_elements); value_cache[layer] = device_allocate<float>(cache_elements);
    }
  }
  std::filesystem::path root_; std::uint32_t max_tokens_{}, capacity_{};
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
    if (argc < 2) { std::cerr << "usage: expert-olmoe-runner <container> [new-tokens] [--trace-logits] [--concurrency N] [--verify-interleaving]\n"; return 64; }
    std::uint32_t new_tokens = 12U, concurrency = 1U;
    bool trace_logits = false, verify_interleaving = false;
    int argument = 2;
    if (argument < argc && std::string_view(argv[argument]).find("--") != 0) {
      new_tokens = static_cast<std::uint32_t>(std::stoul(argv[argument++]));
    }
    while (argument < argc) {
      const std::string_view option(argv[argument++]);
      if (option == "--trace-logits") trace_logits = true;
      else if (option == "--verify-interleaving") verify_interleaving = true;
      else if (option == "--concurrency" && argument < argc)
        concurrency = static_cast<std::uint32_t>(std::stoul(argv[argument++]));
      else throw std::runtime_error("unknown or incomplete option");
    }
    if (new_tokens == 0 || concurrency == 0 || concurrency > 64)
      throw std::runtime_error("new-tokens/concurrency out of range");
    if (trace_logits && concurrency != 1)
      throw std::runtime_error("logit tracing requires concurrency 1");
    const std::vector<std::uint32_t> prompt{510, 5347, 273, 6181, 310};
    const auto load_started = std::chrono::steady_clock::now();
    Model model(argv[1], static_cast<std::uint32_t>(prompt.size()) + new_tokens,
                concurrency);
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_started).count();
    std::vector<std::vector<std::uint32_t>> prompts(concurrency, prompt);
    if (verify_interleaving) {
      for (std::uint32_t row = 0; row < concurrency; ++row) prompts[row][0] += row;
    }
    std::vector<std::vector<std::uint32_t>> isolated;
    if (verify_interleaving) {
      isolated.reserve(concurrency);
      for (std::uint32_t row = 0; row < concurrency; ++row) {
        auto sequence = prompts[row];
        std::vector<std::uint32_t> one_token(1), one_position(1), one_prediction;
        for (std::uint32_t position = 0; position < prompt.size(); ++position) {
          one_token[0] = prompts[row][position];
          one_position[0] = position;
          one_prediction = model.forward_batch(one_token, one_position);
        }
        for (std::uint32_t generated = 0; generated < new_tokens; ++generated) {
          sequence.push_back(one_prediction[0]);
          if (generated + 1 < new_tokens) {
            one_token = one_prediction;
            one_position[0] = static_cast<std::uint32_t>(prompt.size()) + generated;
            one_prediction = model.forward_batch(one_token, one_position);
          }
        }
        isolated.push_back(std::move(sequence));
      }
    }
    std::vector<std::vector<std::uint32_t>> full = prompts;
    std::vector<std::uint32_t> predicted(concurrency), batch_tokens(concurrency),
        batch_positions(concurrency);
    const auto prompt_started = std::chrono::steady_clock::now();
    for (std::uint32_t position = 0; position < prompt.size(); ++position) {
      for (std::uint32_t row = 0; row < concurrency; ++row)
        batch_tokens[row] = prompts[row][position];
      std::fill(batch_positions.begin(), batch_positions.end(), position);
      predicted = model.forward_batch(batch_tokens, batch_positions);
    }
    const auto prompt_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prompt_started).count();
    const auto started = std::chrono::steady_clock::now();
    std::vector<double> inter_token_ms;
    for (std::uint32_t generated = 0; generated < new_tokens; ++generated) {
      for (std::uint32_t row = 0; row < concurrency; ++row)
        full[row].push_back(predicted[row]);
      if (trace_logits) {
        std::cerr << "logits position=" << (prompt.size() + generated) << " top=";
        const auto top = model.top_logits(5);
        for (std::size_t i = 0; i < top.size(); ++i) {
          if (i) std::cerr << ',';
          std::cerr << top[i].first << ':' << top[i].second;
        }
        std::cerr << '\n';
      }
      if (generated + 1 < new_tokens) {
        std::fill(batch_positions.begin(), batch_positions.end(),
                  static_cast<std::uint32_t>(prompt.size()) + generated);
        const auto step_started = std::chrono::steady_clock::now();
        predicted = model.forward_batch(predicted, batch_positions);
        inter_token_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - step_started).count());
      }
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto decode_forwards = new_tokens > 0 ? new_tokens - 1U : 0U;
    const auto aggregate_forwards = static_cast<std::uint64_t>(decode_forwards) * concurrency;
    const auto tokens_per_second = aggregate_forwards > 0 ? aggregate_forwards / seconds : 0.0;
    std::sort(inter_token_ms.begin(), inter_token_ms.end());
    const auto percentile = [&](double fraction) {
      if (inter_token_ms.empty()) return 0.0;
      const auto index = static_cast<std::size_t>(
          std::ceil(fraction * static_cast<double>(inter_token_ms.size()))) - 1U;
      return inter_token_ms[std::min(index, inter_token_ms.size() - 1U)];
    };
    const bool identical_requests = std::all_of(
        full.begin() + 1, full.end(), [&](const auto& sequence) { return sequence == full[0]; });
    const bool interleaving_match = !verify_interleaving || full == isolated;
    std::cout << "{\"tokens\":[";
    for (std::size_t i = 0; i < full[0].size(); ++i) { if (i) std::cout << ','; std::cout << full[0][i]; }
    std::cout << "],\"generated\":" << new_tokens
              << ",\"model_load_seconds\":" << load_seconds
              << ",\"startup_pack_read_bytes\":" << model.pack_bytes
              << ",\"startup_pack_h2d_bytes\":" << model.pack_bytes
              << ",\"hot_storage_read_bytes\":0,\"hot_h2d_bytes\":0"
              << ",\"prompt_tokens\":" << prompt.size()
              << ",\"prompt_seconds\":" << prompt_seconds
              << ",\"concurrency\":" << concurrency
              << ",\"decode_forward_tokens\":" << decode_forwards
              << ",\"aggregate_decode_forwards\":" << aggregate_forwards
              << ",\"decode_seconds\":" << seconds
              << ",\"tokens_per_second\":" << tokens_per_second
              << ",\"inter_token_p50_ms\":" << percentile(0.50)
              << ",\"inter_token_p95_ms\":" << percentile(0.95)
              << ",\"fairness_token_skew\":0"
              << ",\"identical_request_outputs\":" << (identical_requests ? "true" : "false")
              << ",\"interleaving_verified\":" << (verify_interleaving ? "true" : "false")
              << ",\"interleaving_match\":" << (interleaving_match ? "true" : "false")
              << ",\"trace_logits\":" << (trace_logits ? "true" : "false") << "}\n";
    return interleaving_match ? 0 : 2;
  } catch (const std::exception& error) { std::cerr << "olmoe runner: " << error.what() << '\n'; return 1; }
}
