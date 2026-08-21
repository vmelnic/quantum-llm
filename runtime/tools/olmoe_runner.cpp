#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/execution_provider.hpp"
#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/moe_virtual_machine.hpp"
#include "expert/runtime/program_executor.hpp"
#include "expert/runtime/routed_expert_runtime.hpp"
#include "expert/runtime/sha256.hpp"
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
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
std::vector<expert::runtime::KernelCapability> provider_capabilities() {
  return {
      {"block.full-attention.causal.v1", 1U, 1U},
      {"router.linear.topk.v1", 1U, 1U},
      {"moe.swiglu.routed.v1", 1U, 1U},
      {"block.causal-short-conv.gated.v1", 1U, 1U},
      {"block.full-attention.gqa.qk-norm.v1", 1U, 1U},
      {"ffn.swiglu.dense.v1", 1U, 1U},
      {"router.sigmoid-bias.topk.v1", 1U, 1U},
      {"embedding.lookup.int8-row.v1", 1U, 1U},
      {"head.rmsnorm.argmax.int8-row.v1", 1U, 1U}};
}
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
struct DevicePack { std::byte* base{}; std::uint64_t bytes{}; };
DevicePack upload_pack(const std::filesystem::path& path, std::uint64_t expected,
                       const expert::runtime::Sha256Digest& expected_sha256) {
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
  if (!expert::runtime::constant_time_equal(hasher.finalize(), expected_sha256))
    throw std::runtime_error("pack SHA-256 mismatch: " + path.string());
  return pack;
}

struct Tensor {
  expert::runtime::cuda::Int8Matrix int8;
  const float* f32{};
  std::vector<std::uint32_t> shape;
  bool quantized{};
};

class Model final : public expert::runtime::IOperationProvider,
                    public expert::runtime::IModelTensorStore {
 public:
  Model(const std::filesystem::path& root, std::uint32_t max_tokens,
        std::uint32_t capacity = 1,
        std::uint64_t ram_cache_bytes = 1ULL << 30U,
        std::uint64_t vram_cache_bytes = 9ULL << 30U)
      : root_(root), max_tokens_(max_tokens), capacity_(capacity) {
    if (capacity_ == 0) throw std::runtime_error("request capacity must be positive");
    status_check(expert::runtime::ModelArtifact::load_expert_pack_v1(
        root, artifact_));
    model_descriptor_ = artifact_.model();
    {
      const auto& component = model_descriptor_.routed_components.front();
      if (component.source_abi !=
              expert::runtime::kExpertSourceAbiExpertPackV1 ||
          component.encoding_abi !=
              expert::runtime::kExpertEncodingAbiFp4Block32 ||
          component.hidden_size != model_descriptor_.hidden_size)
        throw std::runtime_error(
            "SM86 Expert VM requires an FP4 Expert Pack component");
      encoding_abi = component.encoding_abi;
      packed_fp4 = true;
      record_quant_abi = expert::runtime::kExpertRecordAbiFp4Block32;
      hidden = model_descriptor_.hidden_size;
      vocab = model_descriptor_.vocab_size;
      layers = static_cast<std::uint32_t>(model_descriptor_.layer_program.size());
      intermediate = component.intermediate_size;
      experts = component.experts_per_layer;
      top_k = component.route_width;
      heads = descriptor_u32("attention_heads");
      kv_heads = descriptor_u32("kv_heads");
      head_dim = descriptor_u32("head_dim");
      epsilon = descriptor_f32("norm_epsilon_f32_bits");
      if (max_tokens_ > model_descriptor_.max_context_tokens ||
          hidden != heads * head_dim || kv_heads > heads ||
          top_k == 0U || top_k > 32U)
        throw std::runtime_error("runtime model geometry exceeds SM86 provider limits");
      expert::runtime::ExecutionProviderRegistry provider_registry;
      auto registered = provider_registry.add(
          {"sm86-dense-moe", 100U, provider_capabilities()});
      if (!registered.ok())
        throw std::runtime_error(std::string(registered.message()));
      auto bound = provider_registry.bind(model_descriptor_);
      if (!bound.status.ok())
        throw std::runtime_error(std::string(bound.status.message()));
      compiled_program_ = bound.provider.program;
      bound_provider_ = std::move(bound.provider);
    }

    for (const auto& pack : artifact_.packs()) {
      if (pack.kind == "dense") {
        packs_.emplace(pack.name,
                       upload_pack(pack.path, pack.bytes, pack.sha256));
        startup_pack_bytes += pack.bytes;
      } else {
        pageable_pack_bytes += pack.bytes;
      }
      pack_bytes += pack.bytes;
    }
    for (const auto& tensor : artifact_.dense_tensors()) add_tensor(tensor);
    provider_slots_.assign(capacity_, false);
    initialize_paging(ram_cache_bytes, vram_cache_bytes);
    validate_program_contract();
    allocate_workspace();
  }

  struct PreparedOperation final : expert::runtime::IPreparedOperation {
    std::uint32_t kernel{};
    std::map<std::string, std::size_t, std::less<>> input_indices;
    std::vector<std::pair<std::string, std::string>> outputs;
  };

  class ProviderRequestState final
      : public expert::runtime::IOperationProviderRequestState {
   public:
    ProviderRequestState(Model& model, std::uint32_t slot) noexcept
        : model_(model), slot_(slot) {}
    ~ProviderRequestState() override { model_.release_provider_slot(slot_); }
    [[nodiscard]] std::uint32_t slot() const noexcept { return slot_; }
    bool reset_pending{true};

   private:
    Model& model_;
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
      const expert::runtime::ProgramRequestContext&) override {
    const auto slot = acquire_provider_slot();
    if (!slot)
      return {{expert::runtime::ErrorCode::backpressure,
               "dense MoE provider has no free request slot"},
              {}};
    return {expert::runtime::Status::success(),
            std::make_shared<ProviderRequestState>(*this, *slot)};
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
             "dense MoE invocation state is invalid"},
            {}});
      const auto input = [&](std::string_view port)
          -> const expert::runtime::ExecutionValue& {
        const auto found = prepared->input_indices.find(port);
        if (found == prepared->input_indices.end() ||
            found->second >= invocation.inputs.size())
          throw std::runtime_error("dense MoE operation input is absent");
        return invocation.inputs[found->second];
      };
      constexpr std::string_view hidden_abi =
          "batch.hidden.f32.cuda.v1";
      constexpr std::string_view token_abi =
          "batch.token-id.u32.host.v1";
      constexpr std::string_view position_abi =
          "batch.position.u32.host.v1";
      const auto require_device_hidden = [&](const auto& value) {
        if (value.abi != hidden_abi || value.memory_domain != "cuda.device" ||
            value.data != reinterpret_cast<const std::byte*>(hidden_state) ||
            value.bytes != static_cast<std::uint64_t>(hidden) * sizeof(float))
          throw std::runtime_error("dense MoE hidden-state ABI mismatch");
      };

      std::map<std::string, expert::runtime::ExecutionValue, std::less<>>
          outputs;
      switch (prepared->kernel) {
        case kEmbedding: {
          const auto& tokens = input("token_ids");
          if (tokens.abi != token_abi || tokens.bytes != sizeof(std::uint32_t))
            throw std::runtime_error("embedding token ABI mismatch");
          std::uint32_t token{};
          std::memcpy(&token, tokens.data, sizeof(token));
          if (token >= vocab) throw std::runtime_error("token exceeds vocabulary");
          status_check(expert::runtime::cuda::embedding(
              matrix(operation_binding(invocation.operation, "weight")),
              token, hidden_state, nullptr));
          outputs.emplace("hidden", device_value(hidden_state, hidden));
          break;
        }
        case kCausalAttention:
        case kGqaAttention: {
          require_device_hidden(input("hidden"));
          const auto& positions = input("positions");
          if (positions.abi != position_abi ||
              positions.bytes != sizeof(std::uint32_t))
            throw std::runtime_error("attention position ABI mismatch");
          std::uint32_t position{};
          std::memcpy(&position, positions.data, sizeof(position));
          if (position >= max_tokens_)
            throw std::runtime_error("attention position exceeds context");
          if (state->reset_pending) {
            if (position != 0U)
              throw std::runtime_error("new provider session does not start at zero");
            reset_sequence_state(state->slot());
            state->reset_pending = false;
          }
          const std::array position_batch{position};
          execute_attention(invocation.operation, 1U, position_batch,
                            prepared->kernel == kGqaAttention, state->slot());
          outputs.emplace("hidden", device_value(hidden_state, hidden));
          break;
        }
        case kCausalShortConv:
          require_device_hidden(input("hidden"));
          execute_short_conv(invocation.operation, 1U, state->slot());
          outputs.emplace("hidden", device_value(hidden_state, hidden));
          break;
        case kDenseSwiGlu:
          require_device_hidden(input("hidden"));
          execute_dense_ffn(invocation.operation, 1U);
          outputs.emplace("hidden", device_value(hidden_state, hidden));
          break;
        case kLinearRouter:
        case kSigmoidBiasRouter:
          require_device_hidden(input("hidden"));
          execute_router(invocation.operation, 1U,
                         prepared->kernel == kSigmoidBiasRouter);
          outputs.emplace("expert_input", device_value(normalized, hidden));
          outputs.emplace("residual", device_value(hidden_state, hidden));
          outputs.emplace("route_indices",
                          device_value(routing_indices, top_k,
                                       "batch.route-index.u32.cuda.v1"));
          outputs.emplace("route_weights",
                          device_value(routing_scores, top_k,
                                       "batch.route-weight.f32.cuda.v1"));
          break;
        case kRoutedMoe:
          require_device_hidden(input("residual"));
          execute_moe(invocation.operation, 1U);
          outputs.emplace("hidden", device_value(hidden_state, hidden));
          break;
        case kHead: {
          require_device_hidden(input("hidden"));
          status_check(expert::runtime::cuda::rms_norm(
              hidden_state,
              fp32(operation_binding(invocation.operation, "norm")),
              normalized, hidden, epsilon, nullptr));
          status_check(expert::runtime::cuda::gemv(
              matrix(operation_binding(invocation.operation, "weight")),
              normalized, output_logits, nullptr));
          status_check(expert::runtime::cuda::argmax(
              output_logits, vocab, output_token, nullptr));
          auto host = std::make_shared<std::uint32_t>();
          cuda_check(cudaMemcpy(host.get(), output_token, sizeof(*host),
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
          throw std::runtime_error("callable provider kernel is unsupported");
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
      return completed_operation({
          {expert::runtime::ErrorCode::internal, error.what()}, {}});
    }
  }

  expert::runtime::ResolveModelTensorResult resolve(
      std::string_view name) override {
    const auto entry = tensor_entries_.find(name);
    if (entry == tensor_entries_.end())
      return {{expert::runtime::ErrorCode::invalid_argument,
               "dense MoE tensor is absent"},
              {}};
    const auto pack = packs_.find(entry->second.pack);
    if (pack == packs_.end())
      return {{expert::runtime::ErrorCode::internal,
               "dense MoE tensor pack is absent"},
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
      std::span<const std::uint32_t> positions) {
    if (tokens.empty() || tokens.size() != positions.size() ||
        tokens.size() > capacity_)
      throw std::runtime_error("invalid request microbatch");
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    if (std::all_of(positions.begin(), positions.end(),
                    [](std::uint32_t position) { return position == 0U; }))
      reset_sequence_state();
    for (std::uint32_t row = 0; row < rows; ++row) {
      if (positions[row] >= max_tokens_) throw std::runtime_error("context capacity exceeded");
      status_check(expert::runtime::cuda::embedding(
          matrix(model_binding("token_embedding")), tokens[row],
          hidden_state + static_cast<std::size_t>(row) * hidden, nullptr));
    }
    for (const auto& operation : compiled_program_.operations) {
      const auto provider_kernel = compiled_program_.kernels.at(
          operation.kernel_binding).provider_capability_index;
      switch (provider_kernel) {
        case kCausalAttention:
          execute_attention(operation, rows, positions, false);
          break;
        case kLinearRouter:
          execute_router(operation, rows, false);
          break;
        case kRoutedMoe:
          execute_moe(operation, rows);
          break;
        case kCausalShortConv:
          execute_short_conv(operation, rows);
          break;
        case kGqaAttention:
          execute_attention(operation, rows, positions, true);
          break;
        case kDenseSwiGlu:
          execute_dense_ffn(operation, rows);
          break;
        case kSigmoidBiasRouter:
          execute_router(operation, rows, true);
          break;
        case kEmbedding:
        case kHead:
          break;
        default:
          throw std::runtime_error("compiled operation has no numeric dispatch");
      }
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
      const auto logits_offset = static_cast<std::size_t>(row) * vocab;
      status_check(expert::runtime::cuda::rms_norm(
          hidden_state + hidden_offset, fp32(model_binding("final_norm")),
          normalized + hidden_offset, hidden, epsilon, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(model_binding("output_head")), normalized + hidden_offset,
          output_logits + logits_offset, nullptr));
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

  const expert::runtime::ModelDescriptor& descriptor() const noexcept {
    return model_descriptor_;
  }
  expert::runtime::TelemetrySnapshot paging_telemetry() const noexcept {
    return cache_->telemetry();
  }
  expert::runtime::CacheUsage paging_usage() const noexcept {
    return cache_->usage();
  }
  std::uint32_t attention_layer_count() const noexcept {
    return static_cast<std::uint32_t>(std::count_if(
        key_cache.begin(), key_cache.end(),
        [](const float* pointer) { return pointer != nullptr; }));
  }
  std::uint64_t kv_page_bytes(std::uint32_t page_tokens) const noexcept {
    return static_cast<std::uint64_t>(attention_layer_count()) * page_tokens *
           kv_heads * head_dim * sizeof(float) * 2U;
  }

  std::uint32_t hidden{}, intermediate{}, vocab{}, layers{}, heads{}, kv_heads{}, head_dim{}, experts{}, top_k{};
  std::uint32_t record_quant_abi{}, encoding_abi{};
  std::uint64_t pack_bytes{}, startup_pack_bytes{}, pageable_pack_bytes{};
  float epsilon{}, rope_theta{};
  bool packed_fp4{};

 private:
  static constexpr std::uint32_t kCausalAttention = 0U;
  static constexpr std::uint32_t kLinearRouter = 1U;
  static constexpr std::uint32_t kRoutedMoe = 2U;
  static constexpr std::uint32_t kCausalShortConv = 3U;
  static constexpr std::uint32_t kGqaAttention = 4U;
  static constexpr std::uint32_t kDenseSwiGlu = 5U;
  static constexpr std::uint32_t kSigmoidBiasRouter = 6U;
  static constexpr std::uint32_t kEmbedding = 7U;
  static constexpr std::uint32_t kHead = 8U;

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

  std::optional<std::uint32_t> acquire_provider_slot() {
    std::lock_guard lock(provider_slot_mutex_);
    const auto found = std::find(provider_slots_.begin(),
                                 provider_slots_.end(), false);
    if (found == provider_slots_.end()) return std::nullopt;
    *found = true;
    return static_cast<std::uint32_t>(found - provider_slots_.begin());
  }

  void release_provider_slot(std::uint32_t slot) noexcept {
    std::lock_guard lock(provider_slot_mutex_);
    if (slot < provider_slots_.size()) provider_slots_[slot] = false;
  }

  std::uint32_t descriptor_u32(std::string_view key) const {
    const auto found = model_descriptor_.attributes.find(key);
    if (found == model_descriptor_.attributes.end() ||
        found->second > 0xffffffffULL)
      throw std::runtime_error("missing or invalid model parameter " +
                               std::string(key));
    return static_cast<std::uint32_t>(found->second);
  }
  float descriptor_f32(std::string_view key) const {
    return std::bit_cast<float>(descriptor_u32(key));
  }
  static std::uint32_t operation_u32(
      const expert::runtime::CompiledOperationProgram& operation,
      std::string_view key) {
    const auto found = operation.parameters.find(key);
    if (found == operation.parameters.end() || found->second > 0xffffffffULL)
      throw std::runtime_error("missing or invalid operation parameter " +
                               std::string(key));
    return static_cast<std::uint32_t>(found->second);
  }
  static float operation_f32(
      const expert::runtime::CompiledOperationProgram& operation,
      std::string_view key) {
    return std::bit_cast<float>(operation_u32(operation, key));
  }
  static std::uint32_t router_u32(
      const expert::runtime::RoutedExpertComponentDescriptor& component,
      std::string_view key) {
    const auto found = component.router.parameters.find(key);
    if (found == component.router.parameters.end() ||
        found->second > 0xffffffffULL)
      throw std::runtime_error("missing or invalid router parameter " +
                               std::string(key));
    return static_cast<std::uint32_t>(found->second);
  }
  static float router_f32(
      const expert::runtime::RoutedExpertComponentDescriptor& component,
      std::string_view key) {
    return std::bit_cast<float>(router_u32(component, key));
  }
  const std::string& model_binding(std::string_view role) const {
    const auto found = model_descriptor_.tensor_bindings.find(role);
    if (found == model_descriptor_.tensor_bindings.end())
      throw std::runtime_error("missing model tensor role " + std::string(role));
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
  const expert::runtime::cuda::Int8Matrix& matrix(const std::string& name) const {
    const auto& tensor = tensors_.at(name); if (!tensor.quantized) throw std::runtime_error(name + " is not INT8"); return tensor.int8;
  }
  const float* fp32(const std::string& name) const {
    const auto& tensor = tensors_.at(name); if (tensor.quantized) throw std::runtime_error(name + " is not F32"); return tensor.f32;
  }
  const Tensor& tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end())
      throw std::runtime_error("tensor binding is absent from pack: " + name);
    return found->second;
  }
  void expect_tensor(const std::string& name,
                     std::initializer_list<std::uint32_t> shape,
                     bool quantized) const {
    const auto& item = tensor(name);
    if (item.quantized != quantized ||
        item.shape != std::vector<std::uint32_t>(shape))
      throw std::runtime_error("tensor binding has incompatible dtype/shape: " +
                               name);
  }
  void validate_program_contract() const {
    expect_tensor(model_binding("token_embedding"), {vocab, hidden}, true);
    expect_tensor(model_binding("final_norm"), {hidden}, false);
    expect_tensor(model_binding("output_head"), {vocab, hidden}, true);
    bool route_ready = false;
    std::uint32_t route_layer = 0U;
    for (const auto& operation : compiled_program_.operations) {
      const auto kernel = compiled_program_.kernels.at(
          operation.kernel_binding).provider_capability_index;
      if (kernel == kEmbedding) {
        if (operation.logical_layer !=
            expert::runtime::kModelLevelOperationLayer)
          throw std::runtime_error("embedding operation is not model-level");
        expect_tensor(operation_binding(operation, "weight"),
                      {vocab, hidden}, true);
        continue;
      }
      if (kernel == kHead) {
        if (operation.logical_layer !=
            expert::runtime::kModelLevelOperationLayer)
          throw std::runtime_error("head operation is not model-level");
        expect_tensor(operation_binding(operation, "norm"), {hidden}, false);
        expect_tensor(operation_binding(operation, "weight"),
                      {vocab, hidden}, true);
        continue;
      }
      if (operation.logical_layer >= layers)
        throw std::runtime_error("compiled operation layer is out of range");
      if (kernel == kCausalAttention || kernel == kGqaAttention) {
        expect_tensor(operation_binding(operation, "input_norm"), {hidden}, false);
        expect_tensor(operation_binding(operation, "query_projection"),
                      {heads * head_dim, hidden}, true);
        expect_tensor(operation_binding(operation, "key_projection"),
                      {kv_heads * head_dim, hidden}, true);
        expect_tensor(operation_binding(operation, "value_projection"),
                      {kv_heads * head_dim, hidden}, true);
        expect_tensor(operation_binding(operation, "output_projection"),
                      {hidden, heads * head_dim}, true);
        const auto norm_elements = kernel == kGqaAttention ? head_dim : hidden;
        expect_tensor(operation_binding(operation, "query_norm"),
                      {norm_elements}, false);
        expect_tensor(operation_binding(operation, "key_norm"),
                      {kernel == kGqaAttention ? head_dim : kv_heads * head_dim},
                      false);
        if (operation_u32(operation, "head_dim") != head_dim ||
            !(operation_f32(operation, "norm_epsilon_f32_bits") > 0.0F) ||
            !(operation_f32(operation, "rope_theta_f32_bits") > 0.0F) ||
            (kernel == kCausalAttention && heads != kv_heads))
          throw std::runtime_error("attention operation geometry is invalid");
      } else if (kernel == kCausalShortConv) {
        const auto cache = operation_u32(operation, "conv_cache_length");
        expect_tensor(operation_binding(operation, "input_norm"), {hidden}, false);
        expect_tensor(operation_binding(operation, "input_projection"),
                      {3U * hidden, hidden}, true);
        expect_tensor(operation_binding(operation, "convolution"),
                      {hidden, 1U, cache}, false);
        expect_tensor(operation_binding(operation, "output_projection"),
                      {hidden, hidden}, true);
        if (cache == 0U || operation_u32(operation, "bias") != 0U ||
            !(operation_f32(operation, "norm_epsilon_f32_bits") > 0.0F))
          throw std::runtime_error("short-convolution operation is invalid");
      } else if (kernel == kDenseSwiGlu) {
        const auto width = operation_u32(operation, "intermediate_size");
        expect_tensor(operation_binding(operation, "input_norm"), {hidden}, false);
        expect_tensor(operation_binding(operation, "gate_projection"),
                      {width, hidden}, true);
        expect_tensor(operation_binding(operation, "up_projection"),
                      {width, hidden}, true);
        expect_tensor(operation_binding(operation, "down_projection"),
                      {hidden, width}, true);
        if (!(operation_f32(operation, "norm_epsilon_f32_bits") > 0.0F))
          throw std::runtime_error("dense SwiGLU operation is invalid");
      } else if (kernel == kLinearRouter || kernel == kSigmoidBiasRouter) {
        if (!operation.routed_component_index.has_value() ||
            *operation.routed_component_index != 0U)
          throw std::runtime_error("router does not bind the Expert Pack component");
        expect_tensor(operation_binding(operation, "input_norm"), {hidden}, false);
        expect_tensor(operation_binding(operation, "router_weight"),
                      {experts, hidden}, false);
        if (kernel == kSigmoidBiasRouter)
          expect_tensor(operation_binding(operation, "expert_bias"),
                        {experts}, false);
        if (!(operation_f32(operation, "norm_epsilon_f32_bits") > 0.0F))
          throw std::runtime_error("router operation is invalid");
        route_ready = true;
        route_layer = operation.component_layer;
      } else if (kernel == kRoutedMoe) {
        if (!operation.routed_component_index.has_value() ||
            *operation.routed_component_index != 0U || !route_ready ||
            route_layer != operation.component_layer)
          throw std::runtime_error("routed MoE is not immediately fed by its router");
        route_ready = false;
      } else {
        throw std::runtime_error("provider compiled an unknown numeric operation");
      }
    }
    if (route_ready)
      throw std::runtime_error("model program ends with an unconsumed route");
  }

  void execute_attention(
      const expert::runtime::CompiledOperationProgram& operation,
      std::uint32_t rows, std::span<const std::uint32_t> positions,
      bool grouped_query,
      std::optional<std::uint32_t> provider_slot = std::nullopt) {
    const auto norm_epsilon = operation_f32(operation, "norm_epsilon_f32_bits");
    const auto theta = operation_f32(operation, "rope_theta_f32_bits");
    const auto kv_size = kv_heads * head_dim;
    const auto cache_stride = static_cast<std::size_t>(max_tokens_) * kv_size;
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
      auto* row_hidden = hidden_state + hidden_offset;
      auto* row_normalized = normalized + hidden_offset;
      auto* row_query = query + hidden_offset;
      auto* row_key = key + hidden_offset;
      auto* row_value = value + hidden_offset;
      auto* row_attention = attention + hidden_offset;
      auto* row_residual = residual + hidden_offset;
      const auto cache_slot = provider_slot.value_or(row);
      auto* row_key_cache = key_cache.at(operation.logical_layer) +
                            static_cast<std::size_t>(cache_slot) * cache_stride;
      auto* row_value_cache = value_cache.at(operation.logical_layer) +
                              static_cast<std::size_t>(cache_slot) * cache_stride;
      status_check(expert::runtime::cuda::rms_norm(
          row_hidden, fp32(operation_binding(operation, "input_norm")),
          row_normalized, hidden, norm_epsilon, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "query_projection")),
          row_normalized, row_query, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "key_projection")),
          row_normalized, row_key, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "value_projection")),
          row_normalized, row_value, nullptr));
      if (grouped_query) {
        status_check(expert::runtime::cuda::gqa_qkv_rope_cache(
            row_query, row_key, row_value,
            fp32(operation_binding(operation, "query_norm")),
            fp32(operation_binding(operation, "key_norm")), row_key_cache,
            row_value_cache, positions[row], heads, kv_heads, head_dim,
            head_dim, norm_epsilon, theta, nullptr));
        status_check(expert::runtime::cuda::gqa_attention_decode(
            row_query, row_key_cache, row_value_cache, row_attention,
            positions[row] + 1U, heads, kv_heads, head_dim, nullptr));
      } else {
        status_check(expert::runtime::cuda::qkv_rope_cache(
            row_query, row_key, row_value,
            fp32(operation_binding(operation, "query_norm")),
            fp32(operation_binding(operation, "key_norm")), row_key_cache,
            row_value_cache, positions[row], heads, head_dim, norm_epsilon,
            theta, nullptr));
        status_check(expert::runtime::cuda::attention_decode(
            row_query, row_key_cache, row_value_cache, row_attention,
            positions[row] + 1U, heads, head_dim, nullptr));
      }
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "output_projection")),
          row_attention, row_residual, nullptr));
      status_check(expert::runtime::cuda::add_in_place(
          row_hidden, row_residual, hidden, nullptr));
    }
  }

  void execute_short_conv(
      const expert::runtime::CompiledOperationProgram& operation,
      std::uint32_t rows,
      std::optional<std::uint32_t> provider_slot = std::nullopt) {
    const auto norm_epsilon = operation_f32(operation, "norm_epsilon_f32_bits");
    const auto kernel = operation_u32(operation, "conv_cache_length");
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
      status_check(expert::runtime::cuda::rms_norm(
          hidden_state + hidden_offset,
          fp32(operation_binding(operation, "input_norm")),
          normalized + hidden_offset, hidden, norm_epsilon, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "input_projection")),
          normalized + hidden_offset,
          projected_bcx + static_cast<std::size_t>(row) * 3U * hidden,
          nullptr));
    }
    status_check(expert::runtime::cuda::causal_short_conv_decode({
        projected_bcx, fp32(operation_binding(operation, "convolution")),
        conv_state.at(operation.logical_layer) +
            static_cast<std::size_t>(provider_slot.value_or(0U)) * hidden *
                kernel,
        attention, rows, hidden,
        kernel, nullptr}));
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "output_projection")),
          attention + hidden_offset, residual + hidden_offset, nullptr));
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state + hidden_offset, residual + hidden_offset, hidden,
          nullptr));
    }
  }

  void execute_dense_ffn(
      const expert::runtime::CompiledOperationProgram& operation,
      std::uint32_t rows) {
    const auto width = operation_u32(operation, "intermediate_size");
    const auto norm_epsilon = operation_f32(operation, "norm_epsilon_f32_bits");
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
      const auto width_offset = static_cast<std::size_t>(row) * dense_width;
      status_check(expert::runtime::cuda::rms_norm(
          hidden_state + hidden_offset,
          fp32(operation_binding(operation, "input_norm")),
          normalized + hidden_offset, hidden, norm_epsilon, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "gate_projection")),
          normalized + hidden_offset, dense_gate + width_offset, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "up_projection")),
          normalized + hidden_offset, dense_up + width_offset, nullptr));
      status_check(expert::runtime::cuda::silu_product(
          dense_gate + width_offset, dense_up + width_offset,
          dense_product + width_offset, width, nullptr));
      status_check(expert::runtime::cuda::gemv(
          matrix(operation_binding(operation, "down_projection")),
          dense_product + width_offset, residual + hidden_offset, nullptr));
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state + hidden_offset, residual + hidden_offset, hidden,
          nullptr));
    }
  }

  void execute_router(
      const expert::runtime::CompiledOperationProgram& operation,
      std::uint32_t rows, bool sigmoid_bias) {
    const auto& component = model_descriptor_.routed_components.at(
        *operation.routed_component_index);
    const auto norm_epsilon = operation_f32(operation, "norm_epsilon_f32_bits");
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
      status_check(expert::runtime::cuda::rms_norm(
          hidden_state + hidden_offset,
          fp32(operation_binding(operation, "input_norm")),
          normalized + hidden_offset, hidden, norm_epsilon, nullptr));
    }
    if (sigmoid_bias) {
      status_check(expert::runtime::cuda::sigmoid_bias_router_topk_batch(
          normalized, fp32(operation_binding(operation, "router_weight")),
          fp32(operation_binding(operation, "expert_bias")), rows, hidden,
          experts, top_k,
          router_f32(component, "normalization_epsilon_f32_bits"),
          router_f32(component, "routed_scaling_factor_f32_bits"),
          router_logits, routing_scores, routing_indices, nullptr));
    } else {
      const auto normalize = router_u32(component, "normalize") != 0U;
      for (std::uint32_t row = 0; row < rows; ++row) {
        const auto hidden_offset = static_cast<std::size_t>(row) * hidden;
        const auto logits_offset = static_cast<std::size_t>(row) * experts;
        const auto route_offset = static_cast<std::size_t>(row) * top_k;
        const auto status = normalize
            ? expert::runtime::cuda::router_topk_normalized(
                  normalized + hidden_offset,
                  fp32(operation_binding(operation, "router_weight")), hidden,
                  experts, top_k, router_logits + logits_offset,
                  routing_scores + route_offset,
                  routing_indices + route_offset, nullptr)
            : expert::runtime::cuda::router_topk(
                  normalized + hidden_offset,
                  fp32(operation_binding(operation, "router_weight")), hidden,
                  experts, top_k, router_logits + logits_offset,
                  routing_scores + route_offset,
                  routing_indices + route_offset, nullptr);
        status_check(status);
      }
    }
  }

  void execute_moe(
      const expert::runtime::CompiledOperationProgram& operation,
      std::uint32_t rows) {
    std::vector<std::uint32_t> host_routes(
        static_cast<std::size_t>(rows) * top_k);
    cuda_check(cudaMemcpy(host_routes.data(), routing_indices,
                          host_routes.size() * sizeof(host_routes[0]),
                          cudaMemcpyDeviceToHost),
               "copy exact expert route");
    std::vector<expert::runtime::ExpertResolveHandle> handles;
    handles.reserve(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
      auto resolved = vm_.resolve_operation_route(
          operation.logical_operation,
          std::span<const std::uint32_t>(host_routes)
              .subspan(static_cast<std::size_t>(row) * top_k, top_k),
          expert::runtime::ExpertResolveTarget::device);
      status_check(resolved.status);
      handles.push_back(std::move(resolved.handle));
    }
    std::vector<expert::runtime::ResolvedExpert> leases;
    leases.reserve(static_cast<std::size_t>(rows) * top_k);
    for (auto& handle : handles) {
      std::optional<expert::runtime::ExpertResolveResult> resolved;
      while (!(resolved = handle.poll()))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      status_check(resolved->status);
      for (auto& expert : resolved->experts)
        leases.push_back(std::move(expert));
    }
    auto plan = directory_->pin_or_collect_misses(
        operation.component_layer, routing_indices, rows * top_k, nullptr);
    status_check(plan.status);
    if (!plan.missing_experts.empty() || plan.pin_id == 0U)
      throw std::runtime_error(
          "page resolution did not publish the exact route to CUDA");
    status_check(expert::runtime::cuda::launch_moe_selection_batch({
        normalized, routing_scores, routing_indices, nullptr,
        moe_intermediate, moe_selection_output, moe_q8_input,
        moe_q8_input_scales, moe_q8_intermediate,
        moe_q8_intermediate_scales, rows, hidden, intermediate, top_k,
        experts, nullptr, directory_->device_entries(),
        operation.component_layer, 0.0F, false, true}));
    status_check(expert::runtime::cuda::launch_moe_aggregate({
        moe_selection_output, nullptr, nullptr, nullptr, routing_scores,
        residual, 0U, rows, hidden, top_k, nullptr}));
    status_check(directory_->release_pins_async(plan.pin_id, nullptr));
    for (std::uint32_t row = 0; row < rows; ++row)
      status_check(expert::runtime::cuda::add_in_place(
          hidden_state + static_cast<std::size_t>(row) * hidden,
          residual + static_cast<std::size_t>(row) * hidden, hidden, nullptr));
  }
  void reset_sequence_state(
      std::optional<std::uint32_t> provider_slot = std::nullopt) {
    for (const auto& operation : compiled_program_.operations) {
      const auto kernel = compiled_program_.kernels.at(
          operation.kernel_binding).provider_capability_index;
      if (kernel != kCausalShortConv) continue;
      const auto conv_kernel =
          operation_u32(operation, "conv_cache_length");
      const auto elements = static_cast<std::size_t>(
                                provider_slot ? 1U : capacity_) *
                            hidden * conv_kernel;
      auto* destination = conv_state.at(operation.logical_layer);
      if (provider_slot)
        destination +=
            static_cast<std::size_t>(*provider_slot) * hidden * conv_kernel;
      cuda_check(cudaMemset(destination, 0,
                            elements * sizeof(float)),
                 "reset short-convolution state");
    }
  }
  void add_tensor(const expert::runtime::ArtifactDenseTensor& entry) {
    Tensor tensor;
    tensor.shape = entry.shape;
    auto* data_pointer = packs_.at(entry.pack).base + entry.record_offset +
                         entry.data_offset;
    tensor.quantized = entry.encoding == "I8";
    if (tensor.quantized) {
      tensor.int8.weights = reinterpret_cast<const std::int8_t*>(data_pointer);
      tensor.int8.rows = tensor.shape.at(0);
      tensor.int8.columns = tensor.shape.size() == 1 ? 1 : tensor.shape.at(1);
      tensor.int8.scales = reinterpret_cast<const float*>(
          packs_.at(entry.pack).base + entry.record_offset +
          entry.scale_offset);
    } else tensor.f32 = reinterpret_cast<const float*>(data_pointer);
    if (!tensors_.emplace(entry.name, std::move(tensor)).second)
      throw std::runtime_error("duplicate dense tensor " + entry.name);
    if (!tensor_entries_.emplace(entry.name, entry).second)
      throw std::runtime_error("duplicate dense tensor metadata " + entry.name);
  }
  void initialize_paging(std::uint64_t ram_cache_bytes,
                         std::uint64_t vram_cache_bytes) {
    const auto& component = model_descriptor_.routed_components.front();
    const auto* artifact_component = artifact_.find_component(component.name);
    if (artifact_component == nullptr)
      throw std::runtime_error("artifact has no catalog for routed component");
    std::uint64_t maximum_record_bytes = 0U;
    for (std::uint32_t layer = 0U; layer < component.layer_count; ++layer)
      for (std::uint32_t expert = 0U; expert < component.experts_per_layer;
           ++expert)
        maximum_record_bytes = std::max(
            maximum_record_bytes,
            artifact_component->catalog.find(layer, expert)->stored_bytes);
    if (maximum_record_bytes == 0U ||
        maximum_record_bytes > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error("expert staging record is invalid");
    storage_ = std::make_shared<expert::runtime::WindowsIocpStorage>(4U);
    uploader_ =
        std::make_shared<expert::runtime::cuda::CudaExpertUploader>();
    directory_ =
        std::make_shared<expert::runtime::cuda::CudaExpertDirectory>(
            component.namespace_id, component.encoding_abi,
            component.layer_count, component.experts_per_layer,
            capacity_ * component.route_width);
    const auto slot_count = std::max<std::size_t>(
        32U, static_cast<std::size_t>(capacity_) * top_k * 2U);
    buffers_ = std::make_shared<expert::runtime::FixedBufferPool>(
        slot_count, static_cast<std::size_t>(maximum_record_bytes),
        expert::runtime::kExpertPackAlignment,
        std::make_shared<expert::runtime::CudaPinnedAllocator>(), top_k);
    const auto budget = [](std::uint64_t capacity) {
      if (capacity == 0U) throw std::runtime_error("cache budget is zero");
      return expert::runtime::TierBudget{
          capacity, capacity,
          capacity - std::min<std::uint64_t>(capacity / 8U, 1ULL << 30U)};
    };
    cache_ = std::make_unique<expert::runtime::ExpertCache>(
        expert::runtime::ExpertCacheConfig{
            budget(ram_cache_bytes), budget(vram_cache_bytes), true,
            {component.layer_count, 1U,
             std::min<std::uint64_t>(ram_cache_bytes / 8U, 2ULL << 30U),
             std::min<std::uint64_t>(vram_cache_bytes / 8U, 1ULL << 30U),
             vram_cache_bytes / 4U, ram_cache_bytes / 2U},
            false, 1U},
        storage_, uploader_, buffers_, directory_);
    routed_ = std::make_unique<expert::runtime::RoutedExpertRuntime>(
        component, model_descriptor_.content_hash,
        artifact_component->catalog, *cache_);
    const std::array bindings{
        expert::runtime::MoeVmComponentBinding{component.name, routed_.get()}};
    status_check(expert::runtime::MoeVirtualMachine::create(
        model_descriptor_, std::move(bound_provider_), bindings, vm_));
  }
  void allocate_workspace() {
    const auto hidden_rows = static_cast<std::size_t>(capacity_) * hidden;
    hidden_state = device_allocate<float>(hidden_rows); normalized = device_allocate<float>(hidden_rows);
    query = device_allocate<float>(hidden_rows); key = device_allocate<float>(hidden_rows); value = device_allocate<float>(hidden_rows);
    attention = device_allocate<float>(hidden_rows); residual = device_allocate<float>(hidden_rows);
    projected_bcx = device_allocate<float>(3U * hidden_rows);
    dense_width = 1U;
    for (const auto& operation : compiled_program_.operations) {
      const auto kernel = compiled_program_.kernels.at(
          operation.kernel_binding).provider_capability_index;
      if (kernel == kDenseSwiGlu)
        dense_width = std::max(
            dense_width, operation_u32(operation, "intermediate_size"));
    }
    const auto dense_elements =
        static_cast<std::size_t>(capacity_) * dense_width;
    dense_gate = device_allocate<float>(dense_elements);
    dense_up = device_allocate<float>(dense_elements);
    dense_product = device_allocate<float>(dense_elements);
    router_logits = device_allocate<float>(static_cast<std::size_t>(capacity_) * experts);
    routing_scores = device_allocate<float>(static_cast<std::size_t>(capacity_) * top_k);
    routing_indices = device_allocate<std::uint32_t>(static_cast<std::size_t>(capacity_) * top_k);
    moe_intermediate = device_allocate<float>(static_cast<std::size_t>(capacity_) * top_k * intermediate);
    const auto selections = static_cast<std::size_t>(capacity_) * top_k;
    moe_selection_output = device_allocate<float>(selections * hidden);
    moe_q8_input = device_allocate<std::int8_t>(hidden_rows);
    moe_q8_input_scales = device_allocate<float>(capacity_);
    moe_q8_intermediate =
        device_allocate<std::int8_t>(selections * intermediate);
    moe_q8_intermediate_scales = device_allocate<float>(selections);
    output_logits = device_allocate<float>(static_cast<std::size_t>(capacity_) * vocab);
    output_token = device_allocate<std::uint32_t>(capacity_);
    key_cache.resize(layers, nullptr); value_cache.resize(layers, nullptr);
    conv_state.resize(layers, nullptr);
    const auto cache_elements = static_cast<std::size_t>(capacity_) *
                                max_tokens_ * kv_heads * head_dim;
    for (const auto& operation : compiled_program_.operations) {
      const auto kernel = compiled_program_.kernels.at(
          operation.kernel_binding).provider_capability_index;
      const auto layer = operation.logical_layer;
      if ((kernel == kCausalAttention || kernel == kGqaAttention) &&
          key_cache[layer] == nullptr) {
        key_cache[layer] = device_allocate<float>(cache_elements);
        value_cache[layer] = device_allocate<float>(cache_elements);
      } else if (kernel == kCausalShortConv && conv_state[layer] == nullptr) {
        const auto elements = static_cast<std::size_t>(capacity_) * hidden *
                              operation_u32(operation, "conv_cache_length");
        conv_state[layer] = device_allocate<float>(elements);
        cuda_check(cudaMemset(conv_state[layer], 0, elements * sizeof(float)),
                   "zero short-convolution state");
      }
    }
  }
  std::filesystem::path root_; std::uint32_t max_tokens_{}, capacity_{};
  expert::runtime::ModelArtifact artifact_;
  std::unordered_map<std::string, DevicePack> packs_;
  expert::runtime::ModelDescriptor model_descriptor_;
  expert::runtime::BoundExecutionProvider bound_provider_;
  expert::runtime::CompiledModelProgram compiled_program_;
  std::unordered_map<std::string, Tensor> tensors_;
  std::map<std::string, expert::runtime::ArtifactDenseTensor, std::less<>>
      tensor_entries_;
  std::shared_ptr<const void> dense_lifetime_{
      std::make_shared<std::uint8_t>(0U)};
  std::shared_ptr<const void> workspace_lifetime_{
      std::make_shared<std::uint8_t>(0U)};
  std::mutex provider_slot_mutex_;
  std::vector<bool> provider_slots_;
  std::shared_ptr<expert::runtime::WindowsIocpStorage> storage_;
  std::shared_ptr<expert::runtime::cuda::CudaExpertUploader> uploader_;
  std::shared_ptr<expert::runtime::cuda::CudaExpertDirectory> directory_;
  std::shared_ptr<expert::runtime::FixedBufferPool> buffers_;
  std::unique_ptr<expert::runtime::ExpertCache> cache_;
  std::unique_ptr<expert::runtime::RoutedExpertRuntime> routed_;
  expert::runtime::MoeVirtualMachine vm_;
  float *hidden_state{}, *normalized{}, *query{}, *key{}, *value{}, *attention{}, *residual{};
  float *projected_bcx{}, *dense_gate{}, *dense_up{}, *dense_product{};
  float *router_logits{}, *routing_scores{}, *moe_intermediate{}, *output_logits{};
  float *moe_selection_output{}, *moe_q8_input_scales{},
      *moe_q8_intermediate_scales{};
  std::int8_t *moe_q8_input{}, *moe_q8_intermediate{};
  std::uint32_t *routing_indices{}, *output_token{};
  std::uint32_t dense_width{};
  std::vector<float*> key_cache, value_cache, conv_state;
};

}  // namespace

namespace {

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> fields;
  while (true) {
    const auto position = line.find('\t');
    fields.push_back(line.substr(0, position));
    if (position == std::string_view::npos) break;
    line.remove_prefix(position + 1);
  }
  return fields;
}

std::vector<std::uint32_t> parse_token_ids(std::string_view text) {
  std::vector<std::uint32_t> result;
  while (!text.empty()) {
    const auto separator = text.find(',');
    const auto field = text.substr(0, separator);
    if (field.empty()) throw std::runtime_error("empty token id");
    const auto value = std::stoull(std::string(field));
    if (value > 0xffffffffULL) throw std::runtime_error("token id exceeds u32");
    result.push_back(static_cast<std::uint32_t>(value));
    if (separator == std::string_view::npos) break;
    text.remove_prefix(separator + 1);
  }
  if (result.empty()) throw std::runtime_error("prompt is empty");
  return result;
}

struct WorkerOptions final {
  std::uint32_t max_context{4096U};
  std::uint32_t capacity{1U};
  std::uint32_t kv_page_tokens{256U};
  std::uint64_t ram_cache_bytes{1ULL << 30U};
  std::uint64_t vram_cache_bytes{9ULL << 30U};
  std::uint64_t kv_cache_bytes{2ULL << 30U};
  std::string placement_profile{"balanced"};
  bool retain_previous_route{true};
  bool cpu_hybrid{true};
};

WorkerOptions parse_worker_options(int argc, char** argv) {
  std::vector<std::string_view> arguments;
  arguments.reserve(static_cast<std::size_t>(argc - 3));
  for (int index = 3; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  auto parsed = expert::runtime::parse_worker_launch_options(arguments);
  status_check(parsed.status);
  auto common = std::move(parsed.options);
  WorkerOptions options;
  options.max_context = common.max_context;
  options.capacity = common.capacity;
  options.kv_page_tokens = common.kv_page_tokens;
  options.placement_profile = std::move(common.placement_profile);
  if (common.ram_cache_gib >
          (std::numeric_limits<std::uint64_t>::max() >> 30U) ||
      common.vram_cache_gib >
          (std::numeric_limits<std::uint64_t>::max() >> 30U) ||
      common.kv_cache_mib >
          (std::numeric_limits<std::uint64_t>::max() >> 20U))
    throw std::runtime_error("worker cache byte count overflows");
  options.ram_cache_bytes = common.ram_cache_gib << 30U;
  options.vram_cache_bytes = common.vram_cache_gib << 30U;
  options.kv_cache_bytes = common.kv_cache_mib << 20U;
  for (const auto& [name, value] : common.extensions) {
    if (value)
      throw std::runtime_error("unsupported valued VM extension --" + name);
    if (name == "no-retain-previous-route")
      options.retain_previous_route = false;
    else if (name == "no-cpu-hybrid")
      options.cpu_hybrid = false;
    else
      throw std::runtime_error("unsupported VM extension --" + name);
  }
  return options;
}

int worker_loop(Model& model, const WorkerOptions& options) {
  std::uint64_t active_id = 0;
  std::uint32_t predicted = 0, next_position = 0, context_limit = 0;
  const auto& descriptor = model.descriptor();
  const auto& component = descriptor.routed_components.front();
  std::cout << "{\"type\":\"ready\",\"protocol\":6,\"capacity\":"
            << options.capacity << ",\"architecture_id\":\""
            << descriptor.architecture_id << "\",\"vocab_size\":"
            << descriptor.vocab_size << ",\"max_context_tokens\":"
            << descriptor.max_context_tokens << ",\"routed_layers\":"
            << component.layer_count << ",\"experts_per_layer\":"
            << component.experts_per_layer << ",\"route_width\":"
            << component.route_width << ",\"expert_encoding\":\""
            << component.encoding << "\",\"operation_capabilities\":[";
  for (std::size_t index = 0; index < descriptor.required_kernels.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << '\"' << descriptor.required_kernels[index].capability << '\"';
  }
  const auto page_capacity =
      (options.max_context + options.kv_page_tokens - 1U) /
      options.kv_page_tokens;
  std::cout << "],\"prefill_mode\":\"causal_sequential\""
            << ",\"prefill_chunk_tokens\":1,\"session_retention\":false"
            << ",\"request_stream_mode\":\"single_slot\""
            << ",\"gpu_phase_timing\":false"
            << ",\"mtp_resource_available\":false"
            << ",\"mtp_runtime_ready\":false,\"mtp_enabled\":false"
            << ",\"retain_previous_route\":"
            << (options.retain_previous_route ? "true" : "false")
            << ",\"cpu_hybrid_enabled\":"
            << (options.cpu_hybrid ? "true" : "false")
            << ",\"rope_mode\":\"artifact\",\"kv_dtype\":\"fp32\""
            << ",\"kv_allocation\":\"preallocated\",\"kv_page_tokens\":"
            << options.kv_page_tokens << ",\"kv_page_bytes\":"
            << model.kv_page_bytes(options.kv_page_tokens)
            << ",\"kv_page_capacity\":" << page_capacity
            << ",\"placement_mode\":\"budgeted\""
            << ",\"placement_profile\":\"" << options.placement_profile
            << "\",\"ram_cache_bytes\":" << options.ram_cache_bytes
            << ",\"vram_cache_bytes\":" << options.vram_cache_bytes
            << ",\"placement_prefetch_enabled\":false"
            << ",\"placement_prefetch_state\":\"disabled\""
            << ",\"placement_minimum_observations\":"
            << (options.placement_profile == "latency" ? 1 : 2)
            << "}\n" << std::flush;
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      const auto fields = split_tabs(line);
      if (fields.empty()) continue;
      if (fields[0] == "PING") {
        std::cout << "{\"type\":\"pong\"}\n" << std::flush;
        continue;
      }
      if (fields[0] == "BEGIN") {
        if (fields.size() != 4) throw std::runtime_error("BEGIN field count");
        if (active_id != 0) throw std::runtime_error("worker already has an active request");
        const auto id = std::stoull(std::string(fields[1]));
        if (id == 0) throw std::runtime_error("request id zero");
        const auto requested_context = std::stoull(std::string(fields[2]));
        if (requested_context < 2U || requested_context > options.max_context)
          throw std::runtime_error("request context limit is invalid");
        const auto tokens = parse_token_ids(fields[3]);
        if (tokens.size() >= requested_context)
          throw std::runtime_error("prompt exhausts request context");
        for (std::uint32_t position = 0; position < tokens.size(); ++position)
          predicted = model.forward(tokens[position], position);
        next_position = static_cast<std::uint32_t>(tokens.size());
        context_limit = static_cast<std::uint32_t>(requested_context);
        active_id = id;
        std::cout << "{\"type\":\"begun\",\"id\":" << active_id << "}\n" << std::flush;
        continue;
      }
      if (fields[0] == "STEP") {
        if (fields.size() != 2) throw std::runtime_error("STEP field count");
        const auto comma = fields[1].find(',');
        if (comma == std::string_view::npos)
          throw std::runtime_error("STEP item is invalid");
        const auto id = std::stoull(std::string(fields[1].substr(0, comma)));
        const auto mode = std::stoul(std::string(fields[1].substr(comma + 1)));
        if (id == 0 || id != active_id || mode > 2U)
          throw std::runtime_error("STEP request mismatch");
        const auto token = predicted;
        if (mode != 1U) {
          if (next_position >= context_limit)
            throw std::runtime_error("request context capacity exceeded");
          predicted = model.forward(token, next_position++);
        }
        std::cout << "{\"type\":\"batch\",\"items\":[{\"id\":" << id
                  << ",\"token\":" << token << "}]}\n" << std::flush;
        if (mode == 1U) active_id = 0;
        continue;
      }
      if (fields[0] == "NEXT") {
        if (fields.size() != 3) throw std::runtime_error("NEXT field count");
        const auto id = std::stoull(std::string(fields[1]));
        if (id == 0 || id != active_id) throw std::runtime_error("NEXT request mismatch");
        const bool final = fields[2] == "1";
        const auto token = predicted;
        if (!final) predicted = model.forward(token, next_position++);
        std::cout << "{\"type\":\"token\",\"id\":" << id
                  << ",\"token\":" << token << "}\n" << std::flush;
        if (final) active_id = 0;
        continue;
      }
      if (fields[0] == "END") {
        if (fields.size() != 2) throw std::runtime_error("END field count");
        const auto id = std::stoull(std::string(fields[1]));
        if (id != active_id) throw std::runtime_error("END request mismatch");
        active_id = 0;
        std::cout << "{\"type\":\"ended\",\"id\":" << id << "}\n" << std::flush;
        continue;
      }
      if (fields[0] == "STATS") {
        if (fields.size() != 1) throw std::runtime_error("STATS field count");
        const auto telemetry = model.paging_telemetry();
        const auto usage = model.paging_usage();
        std::cout << "{\"type\":\"stats\",\"kv_allocated_pages\":"
                  << page_capacity << ",\"kv_reserved_pages\":"
                  << (active_id == 0 ? 0 : page_capacity)
                  << ",\"worker_model_steps\":" << next_position
                  << ",\"expert_storage_read_bytes\":"
                  << telemetry.read_bytes
                  << ",\"expert_h2d_bytes\":" << telemetry.uploaded_bytes
                  << ",\"expert_ram_bytes\":" << usage.ram_bytes
                  << ",\"expert_vram_bytes\":" << usage.vram_bytes
                  << ",\"expert_ssd_misses\":"
                  << telemetry.acquire_ssd_misses << "}\n"
                  << std::flush;
        continue;
      }
      if (fields[0] == "SHUTDOWN") {
        if (active_id != 0) throw std::runtime_error("cannot shutdown active worker");
        std::cout << "{\"type\":\"shutdown\"}\n" << std::flush;
        return 0;
      }
      throw std::runtime_error("unknown worker command");
    } catch (const std::exception& error) {
      std::cerr << "worker command failed: " << error.what() << '\n';
      std::cout << "{\"type\":\"error\",\"active_id\":" << active_id
                << "}\n" << std::flush;
    }
  }
  return 0;
}

}  // namespace

int expert_vm_dense_moe_provider_main(int argc, char** argv) {
  try {
    if (argc < 2) { std::cerr << "usage: expert-moe-vm-runner <container> [new-tokens] [--prompt-tokens CSV] [--trace-logits] [--concurrency N] [--verify-interleaving]\n"; return 64; }
    if (argc >= 3 && std::string_view(argv[2]) == "--worker") {
      const auto options = parse_worker_options(argc, argv);
      Model model(argv[1], options.max_context, options.capacity,
                  options.ram_cache_bytes, options.vram_cache_bytes);
      return worker_loop(model, options);
    }
    std::uint32_t new_tokens = 12U, concurrency = 1U;
    bool trace_logits = false, verify_interleaving = false;
    std::vector<std::uint32_t> prompt{510, 5347, 273, 6181, 310};
    int argument = 2;
    if (argument < argc && std::string_view(argv[argument]).find("--") != 0) {
      new_tokens = static_cast<std::uint32_t>(std::stoul(argv[argument++]));
    }
    while (argument < argc) {
      const std::string_view option(argv[argument++]);
      if (option == "--trace-logits") trace_logits = true;
      else if (option == "--verify-interleaving") verify_interleaving = true;
      else if (option == "--prompt-tokens" && argument < argc)
        prompt = parse_token_ids(argv[argument++]);
      else if (option == "--concurrency" && argument < argc)
        concurrency = static_cast<std::uint32_t>(std::stoul(argv[argument++]));
      else throw std::runtime_error("unknown or incomplete option");
    }
    if (new_tokens == 0 || concurrency == 0 || concurrency > 64)
      throw std::runtime_error("new-tokens/concurrency out of range");
    if (trace_logits && concurrency != 1)
      throw std::runtime_error("logit tracing requires concurrency 1");
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
    const auto paging = model.paging_telemetry();
    std::cout << "{\"tokens\":[";
    for (std::size_t i = 0; i < full[0].size(); ++i) { if (i) std::cout << ','; std::cout << full[0][i]; }
    std::cout << "],\"generated\":" << new_tokens
              << ",\"model_load_seconds\":" << load_seconds
              << ",\"startup_pack_read_bytes\":" << model.startup_pack_bytes
              << ",\"startup_pack_h2d_bytes\":" << model.startup_pack_bytes
              << ",\"pageable_pack_bytes\":" << model.pageable_pack_bytes
              << ",\"hot_storage_read_bytes\":" << paging.read_bytes
              << ",\"hot_h2d_bytes\":" << paging.uploaded_bytes
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
  } catch (const std::exception& error) { std::cerr << "Expert VM runner: " << error.what() << '\n'; return 1; }
}

expert::runtime::WorkerProviderDefinition make_sm86_dense_moe_provider() {
  return {"sm86-dense-moe", 100U, provider_capabilities(),
          &expert_vm_dense_moe_provider_main};
}

expert::runtime::CreateExecutionProviderModuleResult
make_sm86_dense_moe_callable_provider(
    const std::filesystem::path& artifact_root, std::uint32_t max_context,
    std::uint32_t capacity, std::uint64_t ram_cache_bytes,
    std::uint64_t vram_cache_bytes, std::uint32_t kv_page_tokens) {
  try {
    auto implementation = std::make_shared<Model>(
        artifact_root, max_context, capacity, ram_cache_bytes,
        vram_cache_bytes);
    expert::runtime::ExecutionProviderModule module;
    module.definition = {"sm86-dense-moe", 100U,
                         provider_capabilities(), implementation};
    module.tensor_store = implementation;
    const auto page_capacity =
        static_cast<std::uint64_t>(capacity) *
        ((static_cast<std::uint64_t>(max_context) + kv_page_tokens - 1U) /
         kv_page_tokens);
    module.service = {
        "causal_sequential", 1U,
        implementation->supports_request_state_retention(),
        "provider_stream", "artifact",
        "fp32", "preallocated", kv_page_tokens,
        implementation->kv_page_bytes(kv_page_tokens),
        page_capacity, "budgeted", "balanced", ram_cache_bytes,
        vram_cache_bytes, false, "disabled", 2U, false, false, false, true,
        false};
    module.telemetry = [implementation] {
      const auto cache = implementation->paging_telemetry();
      return std::map<std::string, std::uint64_t, std::less<>>{
          {"cache_vram_hits", cache.acquire_vram_hits},
          {"cache_ram_hits", cache.acquire_ram_hits},
          {"cache_ssd_misses", cache.acquire_ssd_misses},
          {"cache_read_bytes", cache.read_bytes},
          {"cache_uploaded_bytes", cache.uploaded_bytes},
          {"cache_storage_wait_ns", cache.storage_wait_ns},
          {"cache_upload_wait_ns", cache.upload_wait_ns}};
    };
    return {expert::runtime::Status::success(), std::move(module)};
  } catch (const std::exception& error) {
    return {{expert::runtime::ErrorCode::invalid_argument, error.what()}, {}};
  }
}

#ifndef EXPERT_VM_PROVIDER_LIBRARY
int main(int argc, char** argv) {
  return expert_vm_dense_moe_provider_main(argc, argv);
}
#endif
