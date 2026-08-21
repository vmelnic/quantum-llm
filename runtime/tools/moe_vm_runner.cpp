#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/expert_store.hpp"
#include "expert/runtime/program_executor.hpp"
#include "expert/runtime/worker_contract.hpp"
#include "expert/runtime/worker_provider.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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

er::WorkerProviderDefinition make_sm86_dense_moe_provider();
er::CreateExecutionProviderModuleResult make_sm86_dense_moe_callable_provider(
    const std::filesystem::path&, std::uint32_t, std::uint32_t, std::uint64_t,
    std::uint64_t, std::uint32_t);
er::WorkerProviderDefinition make_sm86_hybrid_delta_moe_provider();
er::CreateExecutionProviderModuleResult
make_sm86_hybrid_delta_moe_callable_provider(
    const std::filesystem::path&, std::uint32_t, std::uint32_t, std::uint64_t,
    std::uint64_t, std::uint64_t, std::uint32_t, std::string_view);
er::WorkerProviderDefinition make_sm86_dense_fp4_provider();
er::CreateExecutionProviderModuleResult make_sm86_dense_fp4_callable_provider(
    const std::filesystem::path&, std::uint32_t, std::uint32_t, std::uint64_t,
    std::uint64_t, std::uint64_t, std::uint32_t, std::string_view);
#ifdef EXPERT_VM_HAS_DEEPSEEK_PROVIDER
er::WorkerProviderDefinition make_sm86_compressed_sparse_moe_provider();
er::CreateExecutionProviderModuleResult
make_sm86_compressed_sparse_moe_callable_provider(
    const std::filesystem::path&, std::uint32_t, std::uint32_t, std::uint64_t,
    std::uint64_t, std::uint64_t, std::uint32_t, std::string_view,
    std::shared_ptr<const er::ActiveExpertOwnerDirectory>);
#endif

namespace {

constexpr std::string_view kTokenAbi = "batch.token-id.u32.host.v1";
constexpr std::string_view kPositionAbi = "batch.position.u32.host.v1";

using CreateModule = std::function<er::CreateExecutionProviderModuleResult()>;

struct ModuleFactory final {
  er::WorkerProviderDefinition metadata;
  CreateModule create;
};

struct ServicePorts final {
  std::string token_input;
  std::string position_input;
  std::string token_output;
};

struct SamplingSettings final {
  std::uint32_t temperature_ppm{};
  std::uint32_t top_p_ppm{1000000U};
  std::uint32_t top_k{};
  std::uint32_t min_p_ppm{};
  std::uint64_t seed{};

  [[nodiscard]] bool enabled() const noexcept {
    return temperature_ppm != 0U;
  }
};

struct ActiveRequest final {
  er::ProgramExecutionSession session;
  std::uint32_t predicted{};
  std::uint32_t next_position{};
  std::uint32_t context_limit{};
  std::uint32_t retention_position{};
  std::uint64_t reserved_pages{};
  SamplingSettings sampling;
};

struct StartedStep final {
  std::uint64_t request_id{};
  std::size_t item_index{};
  er::ProgramExecutionHandle handle;
};

struct StartedExactDecode final {
  std::uint64_t request_id{};
  std::size_t item_index{};
  er::ExactDecodeExecutionHandle handle;
};

class CommandInbox final {
 public:
  CommandInbox() : state_(std::make_shared<State>()) {
    std::thread([state = state_] {
      std::string line;
      while (std::getline(std::cin, line)) {
        {
          std::lock_guard lock(state->mutex);
          state->lines.push_back(std::move(line));
        }
        state->ready.notify_one();
      }
      {
        std::lock_guard lock(state->mutex);
        state->closed = true;
      }
      state->ready.notify_all();
    }).detach();
  }

  [[nodiscard]] std::optional<std::string> next() {
    std::unique_lock lock(state_->mutex);
    state_->ready.wait(lock,
                       [this] { return state_->closed || !state_->lines.empty(); });
    if (state_->lines.empty()) return std::nullopt;
    auto line = std::move(state_->lines.front());
    state_->lines.pop_front();
    return line;
  }

  [[nodiscard]] bool take_cancel(std::uint64_t request_id) {
    const auto expected = std::string("CANCEL\t") +
                          std::to_string(request_id);
    std::lock_guard lock(state_->mutex);
    const auto found =
        std::find(state_->lines.begin(), state_->lines.end(), expected);
    if (found == state_->lines.end()) return false;
    state_->lines.erase(found);
    return true;
  }

 private:
  struct State final {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::string> lines;
    bool closed{};
  };
  std::shared_ptr<State> state_;
};

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

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

std::string json_text(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      result.push_back('\\');
      result.push_back(character);
    } else if (character == '\n' || character == '\r' || character == '\t') {
      result.push_back(' ');
    } else {
      result.push_back(character);
    }
  }
  return result;
}

er::ExecutionValue host_u32(std::uint32_t value, std::string abi) {
  auto owner = std::make_shared<std::uint32_t>(value);
  return {std::move(abi), "host", owner,
          reinterpret_cast<const std::byte*>(owner.get()), sizeof(value)};
}

er::ExecutionValue host_u32_batch(std::span<const std::uint32_t> values,
                                  std::string abi) {
  require(!values.empty(), "cannot publish an empty host batch");
  auto owner =
      std::make_shared<std::vector<std::uint32_t>>(values.begin(), values.end());
  return {std::move(abi), "host", owner,
          reinterpret_cast<const std::byte*>(owner->data()),
          owner->size() * sizeof((*owner)[0])};
}

std::vector<std::uint32_t> read_u32_batch(
    const er::ExecutionValue& value) {
  require(value.valid() && value.abi == kTokenAbi &&
              value.memory_domain == "host" &&
              value.bytes % sizeof(std::uint32_t) == 0U,
          "callable VM returned an invalid token value");
  std::vector<std::uint32_t> result(
      static_cast<std::size_t>(value.bytes / sizeof(std::uint32_t)));
  std::memcpy(result.data(), value.data,
              result.size() * sizeof(result[0]));
  return result;
}

std::uint32_t read_u32(const er::ExecutionValue& value) {
  auto result = read_u32_batch(value);
  require(result.size() == 1U,
          "callable VM returned a token batch to scalar decode");
  return result.front();
}

ServicePorts service_ports(const er::ModelDescriptor& model) {
  ServicePorts result;
  for (const auto& [role, endpoint] : model.program_inputs) {
    if (endpoint.abi == kTokenAbi) {
      require(result.token_input.empty(), "multiple token program inputs");
      result.token_input = role;
    } else if (endpoint.abi == kPositionAbi) {
      require(result.position_input.empty(),
              "multiple position program inputs");
      result.position_input = role;
    }
  }
  for (const auto& [role, endpoint] : model.program_outputs) {
    if (endpoint.abi != kTokenAbi) continue;
    require(result.token_output.empty(), "multiple token program outputs");
    result.token_output = role;
  }
  require(!result.token_input.empty() && !result.position_input.empty() &&
              !result.token_output.empty(),
          "artifact does not implement the token-step service ABI");
  return result;
}

er::ProgramRequestContext request_context(std::uint64_t request_id,
                                          std::uint32_t reserved_context,
                                          std::uint32_t retention_position,
                                          const SamplingSettings& sampling) {
  er::ProgramRequestContext result;
  result.request_id = request_id;
  result.deadline = std::chrono::steady_clock::time_point::max();
  result.parameters.emplace("reserved_context_tokens", reserved_context);
  if (retention_position != 0U)
    result.parameters.emplace("retention_checkpoint_position",
                              retention_position);
  result.parameters.emplace("sampling_temperature_ppm",
                            sampling.temperature_ppm);
  result.parameters.emplace("sampling_top_p_ppm", sampling.top_p_ppm);
  result.parameters.emplace("sampling_top_k", sampling.top_k);
  result.parameters.emplace("sampling_min_p_ppm", sampling.min_p_ppm);
  result.parameters.emplace("sampling_seed", sampling.seed);
  return result;
}

er::StartProgramExecutionResult start_step(
    ActiveRequest& request, const ServicePorts& ports, std::uint32_t token,
    std::uint32_t position) {
  std::map<std::string, er::ExecutionValue, std::less<>> inputs;
  inputs.emplace(ports.token_input,
                 host_u32(token, std::string(kTokenAbi)));
  inputs.emplace(ports.position_input,
                 host_u32(position, std::string(kPositionAbi)));
  return request.session.execute(std::move(inputs));
}

er::StartProgramExecutionResult start_prefill_batch(
    ActiveRequest& request, const ServicePorts& ports,
    std::span<const std::uint32_t> tokens, std::uint32_t first_position) {
  require(!tokens.empty() &&
              tokens.size() <=
                  std::numeric_limits<std::uint32_t>::max() &&
              first_position <=
                  std::numeric_limits<std::uint32_t>::max() -
                      static_cast<std::uint32_t>(tokens.size() - 1U),
          "invalid prefill batch");
  std::vector<std::uint32_t> positions(tokens.size());
  for (std::size_t index = 0U; index < positions.size(); ++index)
    positions[index] =
        first_position + static_cast<std::uint32_t>(index);
  std::map<std::string, er::ExecutionValue, std::less<>> inputs;
  inputs.emplace(ports.token_input,
                 host_u32_batch(tokens, std::string(kTokenAbi)));
  inputs.emplace(ports.position_input,
                 host_u32_batch(positions, std::string(kPositionAbi)));
  return request.session.execute(std::move(inputs));
}

er::StartProgramExecutionResult start_prefill_sequence(
    ActiveRequest& request, const ServicePorts& ports,
    std::span<const std::uint32_t> tokens, std::uint32_t first_position) {
  require(!tokens.empty() &&
              tokens.size() <=
                  std::numeric_limits<std::uint32_t>::max() &&
              first_position <=
                  std::numeric_limits<std::uint32_t>::max() -
                      static_cast<std::uint32_t>(tokens.size() - 1U),
          "invalid prefill program-sequence");
  std::vector<std::uint32_t> positions(tokens.size());
  for (std::size_t index = 0U; index < positions.size(); ++index)
    positions[index] = first_position + static_cast<std::uint32_t>(index);
  std::map<std::string, er::ExecutionValue, std::less<>> inputs;
  inputs.emplace(ports.token_input,
                 host_u32_batch(tokens, std::string(kTokenAbi)));
  inputs.emplace(ports.position_input,
                 host_u32_batch(positions, std::string(kPositionAbi)));
  return request.session.execute_program_sequence(std::move(inputs));
}

std::uint32_t complete_step(er::ProgramExecutionHandle& handle,
                            const ServicePorts& ports) {
  for (;;) {
    auto result = handle.poll();
    if (!result) {
      std::this_thread::yield();
      continue;
    }
    require(result->status.ok(), result->status.message());
    const auto output = result->outputs.find(ports.token_output);
    require(output != result->outputs.end(),
            "model program did not publish its token output");
    return read_u32(output->second);
  }
}

std::uint64_t page_count(std::uint32_t tokens, std::uint32_t page_tokens) {
  return (static_cast<std::uint64_t>(tokens) + page_tokens - 1U) /
         page_tokens;
}

er::ExecutionProviderModule create_module(
    const er::ModelArtifact& artifact, const std::filesystem::path& root,
    const er::WorkerLaunchOptions& options) {
  std::vector<ModuleFactory> factories;
  // Deployment ownership is common runtime data. An empty directory preserves
  // exact local placement; configured transports can populate the same object
  // without changing provider selection or the artifact program.
  auto active_expert_owners =
      std::make_shared<const er::ActiveExpertOwnerDirectory>();
  {
    auto metadata = make_sm86_dense_moe_provider();
    factories.push_back(
        {std::move(metadata),
         [&] {
           return make_sm86_dense_moe_callable_provider(
               root, options.max_context, options.capacity,
               options.ram_cache_gib << 30U, options.vram_cache_gib << 30U,
               options.kv_page_tokens);
         }});
  }
  {
    auto metadata = make_sm86_dense_fp4_provider();
    factories.push_back(
        {std::move(metadata),
         [&] {
           return make_sm86_dense_fp4_callable_provider(
               root, options.max_context, options.capacity,
               options.ram_cache_gib << 30U, options.vram_cache_gib << 30U,
               options.kv_cache_mib << 20U, options.kv_page_tokens,
               options.kv_cache_dtype);
         }});
  }
  {
    auto metadata = make_sm86_hybrid_delta_moe_provider();
    factories.push_back(
        {std::move(metadata),
         [&] {
           return make_sm86_hybrid_delta_moe_callable_provider(
               root, options.max_context, options.capacity,
               options.ram_cache_gib << 30U, options.vram_cache_gib << 30U,
               options.kv_cache_mib << 20U, options.kv_page_tokens,
               options.placement_profile);
         }});
  }
#ifdef EXPERT_VM_HAS_DEEPSEEK_PROVIDER
  {
    auto metadata = make_sm86_compressed_sparse_moe_provider();
    factories.push_back(
        {std::move(metadata),
         [&] {
           return make_sm86_compressed_sparse_moe_callable_provider(
               root, options.max_context, options.capacity,
               options.ram_cache_gib << 30U, options.vram_cache_gib << 30U,
               options.kv_cache_mib << 20U, options.kv_page_tokens,
               options.placement_profile, active_expert_owners);
         }});
  }
#endif

  std::vector<ModuleFactory*> compatible;
  for (auto& factory : factories) {
    if (er::provider_supports_model(artifact.model(),
                                    factory.metadata.capabilities)
            .ok())
      compatible.push_back(&factory);
  }
  require(compatible.size() == 1U,
          compatible.empty()
              ? "no callable provider implements the artifact program"
              : "artifact program ambiguously matches callable providers");
  auto created = compatible.front()->create();
  require(created.status.ok(), created.status.message());
  require(created.module.definition.implementation != nullptr,
          "callable provider module has no implementation");
  // Placement profile is a common launch policy. Providers that implement no
  // speculative placement still report the selected policy with a disabled
  // prefetch state, matching the same service contract.
  if (created.module.service.placement_mode == "budgeted") {
    created.module.service.placement_profile = options.placement_profile;
    created.module.service.placement_minimum_observations =
        options.placement_profile == "latency" ? 1U : 2U;
  }
  return std::move(created.module);
}

void validate_service_contract(
    const er::ModelDescriptor& descriptor,
    const er::ExecutionProviderModule& module,
    const er::WorkerLaunchOptions& options) {
  const auto& service = module.service;
  require(!service.prefill_mode.empty() && service.prefill_chunk_tokens != 0U &&
              module.definition.implementation != nullptr &&
              service.session_retention ==
                  module.definition.implementation
                      ->supports_request_state_retention() &&
              !service.request_stream_mode.empty() &&
              !service.rope_mode.empty() && !service.kv_dtype.empty() &&
              (options.kv_cache_dtype == "artifact" ||
               service.kv_dtype == options.kv_cache_dtype) &&
              !service.kv_allocation.empty() &&
              service.kv_page_tokens == options.kv_page_tokens &&
              service.kv_page_bytes != 0U && service.kv_page_capacity != 0U &&
              (service.placement_mode == "budgeted" ||
               service.placement_mode == "resident") &&
              ((service.placement_mode == "budgeted" &&
                service.placement_profile == options.placement_profile &&
                service.ram_cache_bytes == (options.ram_cache_gib << 30U) &&
                service.vram_cache_bytes == (options.vram_cache_gib << 30U)) ||
               (service.placement_mode == "resident" &&
                service.placement_profile == "resident" &&
                service.ram_cache_bytes == 0U &&
                service.vram_cache_bytes != 0U &&
                !service.placement_prefetch_enabled &&
                service.placement_minimum_observations == 0U)) &&
              !service.placement_prefetch_state.empty() &&
              service.mtp_enabled ==
                  descriptor.exact_decode_program.has_value() &&
              (!service.mtp_enabled ||
               (service.mtp_resource_available && service.mtp_runtime_ready)),
          "callable provider returned an invalid service contract");
}

void print_ready(const er::ModelDescriptor& descriptor,
                 const er::ExecutionProviderModule::ServiceContract& service,
                 std::uint32_t capacity) {
  const auto* routed = descriptor.routed_components.empty()
                           ? nullptr
                           : &descriptor.routed_components.front();
  std::cout << "{\"type\":\"ready\",\"protocol\":8,\"capacity\":"
            << capacity << ",\"architecture_id\":\""
            << json_text(descriptor.architecture_id)
            << "\",\"vocab_size\":" << descriptor.vocab_size
            << ",\"max_context_tokens\":" << descriptor.max_context_tokens
            << ",\"routed_layers\":"
            << (routed == nullptr ? 0U : routed->layer_count)
            << ",\"experts_per_layer\":"
            << (routed == nullptr ? 0U : routed->experts_per_layer)
            << ",\"route_width\":"
            << (routed == nullptr ? 0U : routed->route_width)
            << ",\"expert_encoding\":\""
            << (routed == nullptr ? std::string{} : json_text(routed->encoding))
            << "\",\"operation_capabilities\":[";
  for (std::size_t index = 0U; index < descriptor.required_kernels.size();
       ++index) {
    if (index) std::cout << ',';
    std::cout << '"'
              << json_text(descriptor.required_kernels[index].capability)
              << '"';
  }
  std::cout << "],\"prefill_mode\":\"" << service.prefill_mode
            << "\",\"prefill_chunk_tokens\":"
            << service.prefill_chunk_tokens
            << ",\"session_retention\":"
            << (service.session_retention ? "true" : "false")
            << ",\"request_stream_mode\":\""
            << service.request_stream_mode
            << "\",\"gpu_phase_timing\":false,\"sampling_supported\":"
            << (service.sampling_supported ? "true" : "false")
            << ",\"mtp_resource_available\":"
            << (service.mtp_resource_available ? "true" : "false")
            << ",\"mtp_runtime_ready\":"
            << (service.mtp_runtime_ready ? "true" : "false")
            << ",\"mtp_enabled\":"
            << (service.mtp_enabled ? "true" : "false")
            << ",\"retain_previous_route\":"
            << (service.retain_previous_route ? "true" : "false")
            << ",\"cpu_hybrid_enabled\":"
            << (service.cpu_hybrid_enabled ? "true" : "false")
            << ",\"rope_mode\":\"" << service.rope_mode
            << "\",\"kv_dtype\":\"" << service.kv_dtype
            << "\",\"kv_allocation\":\"" << service.kv_allocation
            << "\",\"kv_page_tokens\":" << service.kv_page_tokens
            << ",\"kv_page_bytes\":" << service.kv_page_bytes
            << ",\"kv_page_capacity\":" << service.kv_page_capacity
            << ",\"placement_mode\":\"" << service.placement_mode
            << "\",\"placement_profile\":\""
            << service.placement_profile << "\",\"ram_cache_bytes\":"
            << service.ram_cache_bytes << ",\"vram_cache_bytes\":"
            << service.vram_cache_bytes
            << ",\"placement_prefetch_enabled\":"
            << (service.placement_prefetch_enabled ? "true" : "false")
            << ",\"placement_prefetch_state\":\""
            << service.placement_prefetch_state
            << "\",\"placement_minimum_observations\":"
            << service.placement_minimum_observations << "}\n" << std::flush;
}

int worker_loop(er::MoeProgramExecutor& executor,
                const er::ModelDescriptor& descriptor,
                er::ExecutionProviderModule& module,
                const er::WorkerLaunchOptions& options) {
  const auto ports = service_ports(descriptor);
  std::unordered_map<std::uint64_t, ActiveRequest> active;
  std::unordered_map<std::uint64_t, ActiveRequest> retained;
  std::uint64_t program_steps{};
  std::uint64_t program_step_ns{};
  std::uint64_t useful_tokens{};
  std::uint64_t exact_decode_calls{};
  std::uint64_t exact_decode_positions{};
  std::uint64_t exact_decode_accepted_tokens{};
  std::uint64_t cancelled_requests{};
  CommandInbox inbox;
  print_ready(descriptor, module.service, options.capacity);

  const auto feed = [&](std::uint64_t request_id, ActiveRequest& request,
                        std::span<const std::uint32_t> tokens,
                        std::uint32_t first_position) {
    const auto complete_prefill = [&](er::ProgramExecutionHandle& handle) {
      for (;;) {
        if (inbox.take_cancel(request_id)) {
          handle.cancel();
          request.session.cancel();
          ++cancelled_requests;
          throw std::runtime_error("model prefill was cancelled");
        }
        auto result = handle.poll();
        if (!result) {
          std::this_thread::yield();
          continue;
        }
        require(result->status.ok(), result->status.message());
        const auto output = result->outputs.find(ports.token_output);
        require(output != result->outputs.end(),
                "model program did not publish its token output");
        return read_u32_batch(output->second);
      }
    };
    const auto chunk_tokens =
        std::max<std::uint32_t>(1U, module.service.prefill_chunk_tokens);
    if (tokens.size() > 1U &&
        request.session.program_sequence_available()) {
      auto step =
          start_prefill_sequence(request, ports, tokens, first_position);
      require(step.status.ok(), step.status.message());
      const auto started = std::chrono::steady_clock::now();
      auto predictions = complete_prefill(step.handle);
      require(predictions.size() == 1U,
              "program-sequence returned an invalid prediction width");
      request.predicted = predictions.front();
      program_steps += tokens.size();
      program_step_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count());
      if (request.session.exact_decode_available()) {
        for (std::size_t offset = 0U; offset < tokens.size();) {
          const auto count = std::min<std::size_t>(
              chunk_tokens, tokens.size() - offset);
          std::vector<std::uint32_t> successors(count);
          for (std::size_t row = 0U; row < count; ++row) {
            const auto index = offset + row;
            successors[row] = index + 1U == tokens.size()
                                  ? request.predicted
                                  : tokens[index + 1U];
          }
          const auto synchronized =
              request.session.synchronize_exact_decode_batch(
                  successors,
                  first_position + static_cast<std::uint32_t>(offset),
                  offset + count == tokens.size() &&
                      !request.sampling.enabled());
          require(synchronized.ok(), synchronized.message());
          offset += count;
        }
      }
      request.next_position =
          first_position + static_cast<std::uint32_t>(tokens.size());
      if (module.service.session_retention) {
        const auto retention_position = request.retention_position == 0U
                                            ? request.next_position
                                            : request.retention_position;
        const auto checkpoint =
            request.session.checkpoint_retention(retention_position);
        require(checkpoint.ok(), checkpoint.message());
        request.retention_position = retention_position;
      }
      return;
    }
    for (std::size_t offset = 0U; offset < tokens.size();) {
      const auto count = std::min<std::size_t>(
          chunk_tokens, tokens.size() - offset);
      const auto batch = tokens.subspan(offset, count);
      auto step = start_prefill_batch(
          request, ports, batch,
          first_position + static_cast<std::uint32_t>(offset));
      require(step.status.ok(), step.status.message());
      const auto started = std::chrono::steady_clock::now();
      auto predictions = complete_prefill(step.handle);
      require(predictions.size() == 1U || predictions.size() == count,
              "provider returned an invalid prefill prediction width");
      request.predicted = predictions.back();
      program_steps += count;
      program_step_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started)
              .count());
      if (request.session.exact_decode_available()) {
        std::vector<std::uint32_t> successors(count);
        for (std::size_t row = 0U; row < count; ++row) {
          const auto index = offset + row;
          const bool final = index + 1U == tokens.size();
          successors[row] = final ? request.predicted : tokens[index + 1U];
        }
        const auto synchronized =
            request.session.synchronize_exact_decode_batch(
                successors,
                first_position + static_cast<std::uint32_t>(offset),
                offset + count == tokens.size() &&
                    !request.sampling.enabled());
        require(synchronized.ok(), synchronized.message());
      }
      offset += count;
    }
    request.next_position =
        first_position + static_cast<std::uint32_t>(tokens.size());
    if (module.service.session_retention) {
      const auto retention_position = request.retention_position == 0U
                                          ? request.next_position
                                          : request.retention_position;
      const auto checkpoint =
          request.session.checkpoint_retention(retention_position);
      require(checkpoint.ok(), checkpoint.message());
      request.retention_position = retention_position;
    }
  };

  while (auto next_line = inbox.next()) {
    auto& line = *next_line;
    try {
      const auto fields = split_tabs(line);
      require(!fields.empty(), "empty worker command");
      if (fields[0] == "PING") {
        require(fields.size() == 1U, "invalid PING");
        std::cout << "{\"type\":\"pong\"}\n" << std::flush;
      } else if (fields[0] == "BEGIN") {
        require(fields.size() >= 4U, "invalid BEGIN");
        const auto id = std::stoull(std::string(fields[1]));
        const auto context = std::stoull(std::string(fields[2]));
        require(id != 0U && !active.contains(id) &&
                    context >= 2U && context <= options.max_context,
                "invalid BEGIN request identity or context");
        const auto prompt = parse_tokens(fields[3], descriptor.vocab_size);
        require(prompt.size() < context, "prompt exhausts request context");
        std::size_t field = 4U;
        bool resume{};
        std::uint64_t resume_key{};
        if (field < fields.size() && fields[field] == "RESUME") {
          require(field + 1U < fields.size(), "invalid BEGIN resume marker");
          resume = true;
          resume_key = std::stoull(std::string(fields[field + 1U]));
          field += 2U;
        }
        std::uint32_t checkpoint_position{};
        if (field < fields.size() && fields[field] == "CHECKPOINT") {
          require(field + 1U < fields.size(),
                  "invalid BEGIN checkpoint marker");
          const auto parsed =
              std::stoull(std::string(fields[field + 1U]));
          require(parsed != 0U && parsed <= options.max_context,
                  "invalid BEGIN checkpoint position");
          checkpoint_position = static_cast<std::uint32_t>(parsed);
          field += 2U;
        }
        SamplingSettings sampling;
        if (field < fields.size() && fields[field] == "SAMPLING") {
          require(field + 5U < fields.size(),
                  "invalid BEGIN sampling marker");
          const auto temperature =
              std::stoull(std::string(fields[field + 1U]));
          const auto top_p = std::stoull(std::string(fields[field + 2U]));
          const auto top_k = std::stoull(std::string(fields[field + 3U]));
          const auto min_p = std::stoull(std::string(fields[field + 4U]));
          sampling.seed = std::stoull(std::string(fields[field + 5U]));
          require(temperature <= 2000000U && top_p != 0U &&
                      top_p <= 1000000U && top_k <= descriptor.vocab_size &&
                      min_p <= 1000000U,
                  "invalid BEGIN sampling values");
          sampling.temperature_ppm =
              static_cast<std::uint32_t>(temperature);
          sampling.top_p_ppm = static_cast<std::uint32_t>(top_p);
          sampling.top_k = static_cast<std::uint32_t>(top_k);
          sampling.min_p_ppm = static_cast<std::uint32_t>(min_p);
          field += 6U;
        }
        require(field == fields.size(), "unknown BEGIN request fields");
        require(module.service.session_retention ||
                    (!resume && checkpoint_position == 0U),
                "callable provider does not support session retention");
        ActiveRequest request;
        if (resume) {
          const auto found = retained.find(resume_key);
          require(found != retained.end(), "unknown retained session");
          request = std::move(found->second);
          retained.erase(found);
          require(checkpoint_position == 0U ||
                      (checkpoint_position > request.next_position &&
                       checkpoint_position <=
                           request.next_position + prompt.size()),
                  "resumed checkpoint is outside the prompt delta");
          if (checkpoint_position != 0U)
            request.retention_position = checkpoint_position;
          request.sampling = sampling;
          auto rebound = request.session.rebind_request(
              request_context(id, options.max_context,
                              request.retention_position, request.sampling));
          require(rebound.ok(), rebound.message());
          require(request.next_position + prompt.size() <= context,
                  "resumed prompt exhausts request context");
          feed(id, request, prompt, request.next_position);
        } else {
          require(checkpoint_position == 0U ||
                      checkpoint_position <= prompt.size(),
                  "checkpoint is outside the prompt");
          request.retention_position = checkpoint_position;
          request.sampling = sampling;
          require(active.size() + retained.size() < options.capacity,
                  "callable provider capacity is exhausted");
          auto begun = executor.begin_session(
              request_context(id, options.max_context,
                              request.retention_position, request.sampling));
          require(begun.status.ok(), begun.status.message());
          request.session = std::move(begun.session);
          request.reserved_pages = page_count(
              static_cast<std::uint32_t>(context),
              module.service.kv_page_tokens);
          feed(id, request, prompt, 0U);
        }
        request.context_limit = static_cast<std::uint32_t>(context);
        active.emplace(id, std::move(request));
        std::cout << "{\"type\":\"begun\",\"id\":" << id << "}\n"
                  << std::flush;
      } else if (fields[0] == "STEP" || fields[0] == "NEXT") {
        std::vector<std::pair<std::uint64_t, std::uint32_t>> items;
        std::set<std::uint64_t> unique;
        const auto parse_item = [&](std::string_view item) {
          const auto comma = item.find(',');
          require(comma != std::string_view::npos, "invalid STEP item");
          const auto id = std::stoull(std::string(item.substr(0U, comma)));
          const auto mode = item.substr(comma + 1U);
          require(active.contains(id) && unique.insert(id).second &&
                      (mode == "0" || mode == "1" || mode == "2"),
                  "STEP request mismatch");
          items.emplace_back(id, mode == "1" ? 1U : mode == "2" ? 2U : 0U);
        };
        if (fields[0] == "NEXT") {
          require(fields.size() == 3U, "invalid NEXT");
          parse_item(std::string(fields[1]) + "," + std::string(fields[2]));
        } else {
          require(fields.size() >= 2U &&
                      fields.size() <= options.capacity + 1U,
                  "invalid STEP");
          for (std::size_t index = 1U; index < fields.size(); ++index)
            parse_item(fields[index]);
        }

        std::vector<std::vector<std::uint32_t>> emitted(items.size());
        std::vector<StartedStep> started;
        started.reserve(items.size());
        std::vector<StartedExactDecode> exact_started;
        exact_started.reserve(items.size());
        const auto batch_started = std::chrono::steady_clock::now();
        for (std::size_t index = 0U; index < items.size(); ++index) {
          const auto [id, mode] = items[index];
          auto& request = active.at(id);
          if (mode == 1U) {
            emitted[index] = {request.predicted};
            continue;
          }
          require(request.next_position < request.context_limit,
                  "request context is exhausted");
          if (mode == 0U && !request.sampling.enabled() &&
              request.session.exact_decode_available() &&
              request.next_position + 1U < request.context_limit) {
            auto exact = request.session.execute_exact_decode(
                request.predicted, request.next_position,
                request.context_limit);
            require(exact.status.ok(), exact.status.message());
            exact_started.push_back(
                {id, index, std::move(exact.handle)});
            continue;
          }
          emitted[index] = {request.predicted};
          auto step = start_step(request, ports, request.predicted,
                                 request.next_position);
          require(step.status.ok(), step.status.message());
          started.push_back({id, index, std::move(step.handle)});
        }
        std::size_t completed{};
        std::vector<bool> done(started.size());
        std::size_t exact_completed{};
        std::vector<bool> exact_done(exact_started.size());
        while (completed != started.size() ||
               exact_completed != exact_started.size()) {
          bool progress = false;
          for (std::size_t index = 0U; index < started.size(); ++index) {
            if (done[index]) continue;
            auto result = started[index].handle.poll();
            if (!result) continue;
            require(result->status.ok(), result->status.message());
            const auto output = result->outputs.find(ports.token_output);
            require(output != result->outputs.end(),
                    "model program did not publish its token output");
            auto& request = active.at(started[index].request_id);
            request.predicted = read_u32(output->second);
            ++request.next_position;
            if (request.session.exact_decode_available()) {
              const auto synchronized =
                  request.session.synchronize_exact_decode(
                  request.predicted, request.next_position - 1U,
                  !request.sampling.enabled());
              require(synchronized.ok(), synchronized.message());
            }
            done[index] = true;
            ++completed;
            ++program_steps;
            progress = true;
          }
          for (std::size_t index = 0U; index < exact_started.size(); ++index) {
            if (exact_done[index]) continue;
            auto result = exact_started[index].handle.poll();
            if (!result) continue;
            require(result->status.ok(), result->status.message());
            auto& request = active.at(exact_started[index].request_id);
            emitted[exact_started[index].item_index] =
                std::move(result->emitted_tokens);
            request.predicted = result->next_token;
            request.next_position += result->positions_advanced;
            program_steps += result->positions_advanced;
            ++exact_decode_calls;
            exact_decode_positions += result->positions_advanced;
            exact_decode_accepted_tokens +=
                result->positions_advanced - 1U;
            exact_done[index] = true;
            ++exact_completed;
            progress = true;
          }
          if (!progress) std::this_thread::yield();
        }
        program_step_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - batch_started)
                .count());
        for (const auto& tokens : emitted) useful_tokens += tokens.size();

        const auto print_tokens = [](const auto& tokens) {
          for (std::size_t index = 0U; index < tokens.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << tokens[index];
          }
        };

        if (fields[0] == "NEXT") {
          std::cout << "{\"type\":\"token\",\"id\":" << items[0].first
                    << ",\"tokens\":[";
          print_tokens(emitted[0]);
          std::cout << "]}\n";
        } else {
          std::cout << "{\"type\":\"batch\",\"items\":[";
          for (std::size_t index = 0U; index < items.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << "{\"id\":" << items[index].first
                      << ",\"tokens\":[";
            print_tokens(emitted[index]);
            std::cout << "]}";
          }
          std::cout << "]}\n";
        }
        std::cout << std::flush;
        for (const auto [id, mode] : items) {
          if (mode != 1U) continue;
          active.erase(id);
        }
      } else if (fields[0] == "CANCEL") {
        require(fields.size() == 2U, "invalid CANCEL");
        const auto id = std::stoull(std::string(fields[1]));
        const auto found = active.find(id);
        if (found != active.end()) {
          found->second.session.cancel();
          active.erase(found);
          ++cancelled_requests;
        }
      } else if (fields[0] == "END") {
        require(fields.size() == 2U || fields.size() == 4U ||
                    fields.size() == 6U,
                "invalid END");
        const auto id = std::stoull(std::string(fields[1]));
        const auto found = active.find(id);
        require(found != active.end(), "unknown active request");
        if (fields.size() == 4U || fields.size() == 6U) {
          require(module.service.session_retention,
                  "callable provider does not support session retention");
          require(fields[2] == "RETAIN", "invalid END retention marker");
          const auto key = std::stoull(std::string(fields[3]));
          require(key != 0U && !retained.contains(key),
                  "duplicate retained session");
          if (fields.size() == 6U) {
            require(fields[4] == "AT", "invalid END checkpoint marker");
            const auto position =
                std::stoull(std::string(fields[5]));
            require(position == found->second.retention_position,
                    "retention checkpoint position mismatch");
            const auto rewound = found->second.session.rewind_retention(
                static_cast<std::uint32_t>(position));
            require(rewound.ok(), rewound.message());
            found->second.next_position =
                static_cast<std::uint32_t>(position);
          }
          const auto tokens = found->second.next_position;
          retained.emplace(key, std::move(found->second));
          active.erase(found);
          std::cout << "{\"type\":\"ended\",\"id\":" << id
                    << ",\"retained_tokens\":" << tokens << "}\n"
                    << std::flush;
        } else {
          active.erase(found);
          ++cancelled_requests;
          std::cout << "{\"type\":\"ended\",\"id\":" << id << "}\n"
                    << std::flush;
        }
      } else if (fields[0] == "DROP") {
        require(fields.size() == 2U, "invalid DROP");
        require(module.service.session_retention,
                "callable provider does not support session retention");
        const auto key = std::stoull(std::string(fields[1]));
        require(retained.erase(key) == 1U, "unknown retained session");
        std::cout << "{\"type\":\"dropped\"}\n" << std::flush;
      } else if (fields[0] == "STATS") {
        require(fields.size() == 1U, "invalid STATS");
        std::uint64_t reserved_pages{};
        for (const auto& [id, request] : active) {
          (void)id;
          reserved_pages += request.reserved_pages;
        }
        for (const auto& [key, request] : retained) {
          (void)key;
          reserved_pages += request.reserved_pages;
        }
        auto telemetry = module.telemetry
                             ? module.telemetry()
                             : std::map<std::string, std::uint64_t,
                                        std::less<>>{};
        telemetry["program_steps"] = program_steps;
        telemetry["program_step_ns"] = program_step_ns;
        telemetry["useful_tokens"] = useful_tokens;
        telemetry["exact_decode_calls"] = exact_decode_calls;
        telemetry["exact_decode_positions"] = exact_decode_positions;
        telemetry["exact_decode_accepted_tokens"] =
            exact_decode_accepted_tokens;
        telemetry["cancelled_requests"] = cancelled_requests;
        telemetry["active_requests"] = active.size();
        telemetry["retained_sessions"] = retained.size();
        telemetry.try_emplace("kv_allocated_pages", reserved_pages);
        telemetry.try_emplace("kv_reserved_pages", reserved_pages);
        std::cout << "{\"type\":\"stats\"";
        for (const auto& [name, value] : telemetry)
          std::cout << ",\"" << json_text(name) << "\":" << value;
        std::cout << "}\n" << std::flush;
      } else if (fields[0] == "SHUTDOWN") {
        require(fields.size() == 1U && active.empty() && retained.empty(),
                "SHUTDOWN requires no retained state");
        std::cout << "{\"type\":\"shutdown\"}\n" << std::flush;
        return 0;
      } else {
        throw std::runtime_error("unknown worker command");
      }
    } catch (const std::exception& error) {
      std::cout << "{\"type\":\"error\",\"message\":\""
                << json_text(error.what()) << "\"}\n" << std::flush;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || std::string_view(argv[2]) != "--worker") {
      std::cerr << "usage: expert-moe-vm-runner <artifact> --worker "
                   "<resource-options>\n";
      return 64;
    }
    std::vector<std::string_view> arguments;
    arguments.reserve(static_cast<std::size_t>(argc - 3));
    for (int index = 3; index < argc; ++index) arguments.emplace_back(argv[index]);
    auto parsed = er::parse_worker_launch_options(arguments);
    require(parsed.status.ok(), parsed.status.message());
    require(parsed.options.extensions.empty(),
            "callable VM service does not accept provider-specific launch flags");
    require(parsed.options.ram_cache_gib <=
                    (std::numeric_limits<std::uint64_t>::max() >> 30U) &&
                parsed.options.vram_cache_gib <=
                    (std::numeric_limits<std::uint64_t>::max() >> 30U) &&
                parsed.options.kv_cache_mib <=
                    (std::numeric_limits<std::uint64_t>::max() >> 20U),
            "worker resource byte count overflows");

    const std::filesystem::path root(argv[1]);
    er::ModelArtifact artifact;
    auto status = er::ModelArtifact::load(root, artifact);
    require(status.ok(), status.message());
    require(artifact.model().schema_version >= 3U,
            "callable VM service requires a schema v3 artifact");
    auto module = create_module(artifact, root, parsed.options);
    validate_service_contract(artifact.model(), module, parsed.options);

    er::ExecutionProviderRegistry registry;
    status = registry.add(module.definition);
    require(status.ok(), status.message());
    auto bound = registry.bind(
        artifact.model(), er::ExecutionProviderBindingMode::executable);
    require(bound.status.ok(), bound.status.message());
    er::MoeProgramExecutor executor;
    status = er::MoeProgramExecutor::create(
        artifact.model(), std::move(bound.provider), module.tensor_store.get(),
        executor);
    require(status.ok(), status.message());
    return worker_loop(executor, artifact.model(), module, parsed.options);
  } catch (const std::exception& error) {
    std::cerr << "MoE VM runner: " << error.what() << '\n';
    return 1;
  }
}
