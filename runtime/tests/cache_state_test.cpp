#include "expert/runtime/adaptive_placement.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_catalog.hpp"
#include "expert/runtime/expert_store.hpp"
#include "expert/runtime/execution_provider.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/model_descriptor.hpp"
#include "expert/runtime/moe_virtual_machine.hpp"
#include "expert/runtime/placement_profile.hpp"
#include "expert/runtime/program_executor.hpp"
#include "expert/runtime/resource_governor.hpp"
#include "expert/runtime/route_census.hpp"
#include "expert/runtime/routed_expert_runtime.hpp"
#include "expert/runtime/cpu/fp4_host_executor.hpp"
#include "expert/runtime/cpu/expert_executor.hpp"
#include "expert/runtime/scheduler.hpp"
#include "expert/runtime/sha256.hpp"
#include "expert/runtime/speculative_sampling.hpp"
#include "expert/runtime/worker_contract.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace er = expert::runtime;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void test_exact_rejection_sampling_matches_independent_target_oracle() {
  const er::SamplingDistribution target{{
      {3U, 0.50}, {7U, 0.30}, {11U, 0.20},
  }};
  const er::SamplingDistribution draft{{
      {3U, 0.20}, {7U, 0.70}, {13U, 0.10},
  }};

  // Independent closed-form oracle: accepted mass is min(p, q), while all
  // rejection mass is redistributed according to normalized max(p-q, 0).
  const std::array tokens{3U, 7U, 11U, 13U};
  std::array<double, tokens.size()> expected{};
  double rejection_mass{};
  double residual_mass{};
  for (std::size_t index = 0U; index < tokens.size(); ++index) {
    const auto p = target.probability(tokens[index]);
    const auto q = draft.probability(tokens[index]);
    expected[index] = std::min(p, q);
    rejection_mass += q - std::min(p, q);
    residual_mass += std::max(0.0, p - q);
  }
  require(std::abs(rejection_mass - residual_mass) < 1e-12,
          "independent p/q mass identity failed");
  for (std::size_t index = 0U; index < tokens.size(); ++index)
    expected[index] += rejection_mass *
        std::max(0.0, target.probability(tokens[index]) -
                          draft.probability(tokens[index])) /
        residual_mass;

  constexpr std::uint32_t trials = 400'000U;
  std::array<std::uint32_t, tokens.size()> observed{};
  for (std::uint32_t trial = 0U; trial < trials; ++trial) {
    const auto proposal = er::sample_distribution(
        draft, er::counter_uniform(19U, trial, 1U));
    const auto result = er::rejection_sample(
        proposal, target, draft, er::counter_uniform(19U, trial, 2U),
        er::counter_uniform(19U, trial, 3U));
    const auto found = std::find(tokens.begin(), tokens.end(), result.token);
    require(found != tokens.end(), "p/q sampler emitted an unknown token");
    ++observed[static_cast<std::size_t>(found - tokens.begin())];
  }
  for (std::size_t index = 0U; index < tokens.size(); ++index) {
    const auto frequency =
        static_cast<double>(observed[index]) / static_cast<double>(trials);
    require(std::abs(frequency - expected[index]) < 0.004,
            "p/q sampler disagrees with independent target oracle");
  }

  const std::array logits{4.0F, 3.0F, 2.0F, 1.0F};
  const std::array ids{9U, 5U, 2U, 1U};
  const auto nucleus = er::make_sampling_distribution(
      logits, ids, 1'000'000U, 800'000U, 0U);
  require(nucleus.entries.size() == 2U &&
              nucleus.entries[0].token == 9U &&
              nucleus.entries[1].token == 5U &&
              std::abs(nucleus.entries[0].probability -
                       0.7310585786300049) < 1e-12,
          "sampling distribution changed scalar temperature/top-p math");
}


void test_universal_model_descriptor_negotiates_capabilities() {
  er::ModelDescriptor descriptor;
  descriptor.architecture_id = "fixture.sparse-transformer";
  descriptor.vocab_size = 32000U;
  descriptor.max_context_tokens = 131072U;
  descriptor.hidden_size = 4096U;
  descriptor.attributes.emplace("attention_heads", 32U);
  descriptor.routed_components.push_back(
      {"decoder", 0x101U, 91U, 1024U, 8U, 1U, 4096U, 1536U,
       "moe.fixture.vendor-fp4", 7U, 77U, 91U,
       "fp4.fixture.group64", {{"group_size", 64U}}, {}});
  descriptor.required_kernels = {
      {"attention.fixture.v1", 1U}, {"moe.fixture.vendor-fp4", 7U}};
  for (std::uint32_t layer = 0U; layer < 96U; ++layer) {
    descriptor.layer_program.push_back(
        {layer, "attention.fixture.v1", 1U,
         layer < 5U ? "" : "decoder", layer < 5U ? 0U : layer - 5U,
         {{"window", layer < 8U ? 4096U : 0U}}});
  }
  require(er::validate_model_descriptor(descriptor).ok() &&
              er::expert_table_entries(descriptor.routed_components.front()) ==
                  91ULL * 1024U,
          "universal descriptor rejected a valid sparse topology");
  const std::array supported{
      er::KernelCapability{"attention.fixture.v1", 1U, 1U},
      er::KernelCapability{
          "moe.fixture.vendor-fp4", 6U, 9U,
          [](const er::ModelDescriptor& candidate) {
            const auto* component =
                er::find_routed_component(candidate, "decoder");
            if (component == nullptr ||
                component->encoding != "fp4.fixture.group64" ||
                !component->attributes.contains("group_size") ||
                component->attributes.at("group_size") != 64U)
              return er::Status(er::ErrorCode::invalid_argument,
                                "fixture provider rejected numeric encoding");
            return er::Status::success();
          }}};
  require(er::provider_supports_model(descriptor, supported).ok(),
          "provider capability negotiation rejected supported operation ABIs");
  const auto compiled = er::compile_model_program(descriptor, supported);
  require(compiled.status.ok() && compiled.program.layers.size() == 96U &&
              compiled.program.kernels.size() == 2U &&
              !compiled.program.layers[4U].routed_component_index &&
              compiled.program.layers[73U].component_layer == 68U &&
              compiled.program.layers[73U].routed_component_index == 0U,
          "universal descriptor did not compile to numeric provider bindings");
  auto unsupported_encoding = descriptor;
  unsupported_encoding.routed_components.front().attributes["group_size"] =
      32U;
  require(!er::compile_model_program(unsupported_encoding, supported)
               .status.ok(),
          "provider-specific constraints leaked out of capability validation");
  const std::array incomplete{
      er::KernelCapability{"moe.fixture.vendor-fp4", 6U, 9U}};
  require(!er::provider_supports_model(descriptor, incomplete).ok(),
          "provider capability negotiation accepted a missing attention ABI");
  auto malformed = descriptor;
  malformed.layer_program.back().logical_layer = 97U;
  require(!er::validate_model_descriptor(malformed).ok(),
          "universal descriptor accepted a non-contiguous layer program");
}

void test_serialized_model_program_is_provider_neutral() {
  constexpr std::string_view artifact =
      "expert-runtime-model-v1\n"
      "model\t1\tfixture.hybrid.sparse\t64001\t262144\t6144\n"
      "attribute\tattention_heads\t48\n"
      "kernel\tblock.fixture.dense-prefix.v2\t2\n"
      "kernel\tblock.fixture.windowed.v3\t3\n"
      "kernel\tmoe.fixture.fp4-group64.v5\t5\n"
      "component\tdecoder\t0\t2\t1000\t7\t2\t6144\t1792\t"
      "moe.fixture.fp4-group64.v5\t5\t91\t42\t"
      "fp4.fixture.e2m1.group64\n"
      "component_attribute\tdecoder\tgroup_size\t64\n"
      "layer\t0\tblock.fixture.dense-prefix.v2\t2\t-\t0\n"
      "layer_parameter\t0\twindow\t8192\n"
      "layer\t1\tblock.fixture.windowed.v3\t3\tdecoder\t0\n"
      "layer_parameter\t1\twindow\t4096\n"
      "layer\t2\tblock.fixture.windowed.v3\t3\tdecoder\t1\n";
  er::Sha256Digest content_hash{};
  content_hash[0] = std::byte{0xa5};
  const auto parsed =
      er::parse_model_descriptor_artifact(artifact, content_hash, 0x9000U);
  require(parsed.status.ok() &&
              parsed.descriptor.architecture_id ==
                  "fixture.hybrid.sparse" &&
              parsed.descriptor.routed_components.size() == 1U &&
              parsed.descriptor.routed_components.front().namespace_id ==
                  0x9000U &&
              parsed.descriptor.routed_components.front().experts_per_layer ==
                  1000U &&
              parsed.descriptor.routed_components.front().encoding ==
                  "fp4.fixture.e2m1.group64" &&
              parsed.descriptor.layer_program.front().routed_component.empty() &&
              parsed.descriptor.layer_program.back().component_layer == 1U,
          "serialized model program retained provider/model assumptions");
  const auto root = std::filesystem::temp_directory_path() /
                    "expert-runtime-model-program";
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
  std::filesystem::create_directories(root);
  const auto path = root / "runtime-model.tsv";
  {
    std::ofstream output(path, std::ios::binary);
    output.write(artifact.data(), static_cast<std::streamsize>(artifact.size()));
  }
  const auto artifact_bytes = std::as_bytes(
      std::span<const char>(artifact.data(), artifact.size()));
  const auto loaded = er::load_model_descriptor_artifact(
      path, artifact.size(), er::sha256(artifact_bytes), content_hash, 0x9000U);
  require(loaded.status.ok() &&
              loaded.descriptor.layer_program.size() == 3U,
          "model program file did not pass its size/SHA binding");
  auto wrong_sha = er::sha256(artifact_bytes);
  wrong_sha[0] ^= std::byte{1};
  require(!er::load_model_descriptor_artifact(
               path, artifact.size(), wrong_sha, content_hash, 0x9000U)
               .status.ok(),
          "model program file accepted the wrong artifact SHA");
  const auto duplicate = std::string(artifact) +
                         "layer_parameter\t1\twindow\t2048\n";
  require(!er::parse_model_descriptor_artifact(duplicate, content_hash,
                                                0x9000U)
               .status.ok(),
          "serialized model program accepted duplicate parameters");
  require(!er::parse_model_descriptor_artifact(artifact, content_hash, 0U)
               .status.ok(),
          "serialized model program accepted an unbound namespace");
  std::filesystem::remove_all(root, cleanup_error);
}

void test_schema_v2_artifact_binds_unknown_model_without_architecture_branch() {
  constexpr std::string_view artifact =
      "expert-runtime-model-v1\n"
      "model\t2\tunknown.vendor.fp4.moe\t70001\t524288\t5120\n"
      "model_tensor\ttoken_embedding\tmodel.embed.weight\n"
      "kernel\tblock.vendor.windowed.v4\t4\n"
      "kernel\trouter.vendor.grouped-topk.v3\t3\n"
      "kernel\tmoe.vendor.fp4-block64.v8\t8\n"
      "component\tdecoder\t0\t1\t513\t9\t1\t5120\t1664\t"
      "moe.vendor.fp4-block64.v8\t8\t77\t91\t"
      "fp4.vendor.e2m1.block64\n"
      "component_attribute\tdecoder\tgroup_size\t64\n"
      "router\tdecoder\trouter.vendor.grouped-topk.v3\t3\n"
      "router_parameter\tdecoder\tnormalize\t1\n"
      "router_parameter\tdecoder\tgroups\t8\n"
      "layer\t0\tblock.vendor.windowed.v4\t4\tdecoder\t0\n"
      "layer_parameter\t0\twindow\t16384\n"
      "operation\t0\t0\tblock.vendor.windowed.v4\t4\t-\t0\n"
      "operation_parameter\t0\twindow\t16384\n"
      "operation_tensor\t0\tinput_norm\tmodel.layers.0.norm.weight\n"
      "operation\t1\t0\trouter.vendor.grouped-topk.v3\t3\tdecoder\t0\n"
      "operation\t2\t0\tmoe.vendor.fp4-block64.v8\t8\tdecoder\t0\n";
  er::Sha256Digest content_hash{};
  content_hash[0] = std::byte{0x5a};
  auto parsed =
      er::parse_model_descriptor_artifact(artifact, content_hash, 0x4100U);
  require(parsed.status.ok() && parsed.descriptor.schema_version == 2U &&
              parsed.descriptor.routed_components.front()
                      .router.parameters.at("groups") == 8U &&
              parsed.descriptor.tensor_bindings.at("token_embedding") ==
                  "model.embed.weight" &&
              parsed.descriptor.operation_program.front()
                      .tensor_bindings.at("input_norm") ==
                  "model.layers.0.norm.weight" &&
              parsed.descriptor.operation_program.size() == 3U,
          "schema v2 artifact did not preserve router and operation IR");

  er::ExecutionProviderRegistry registry;
  require(registry
              .add({"incomplete-high-priority", 100U,
                    {er::KernelCapability{"block.vendor.windowed.v4", 4U,
                                          4U},
                     er::KernelCapability{"moe.vendor.fp4-block64.v8", 8U,
                                          8U}}})
              .ok(),
          "provider registry rejected an internally valid partial provider");
  require(registry
              .add({"generic-sm86-fp4", 10U,
                    {er::KernelCapability{"block.vendor.windowed.v4", 4U,
                                          4U},
                     er::KernelCapability{"router.vendor.grouped-topk.v3", 3U,
                                          3U},
                     er::KernelCapability{
                         "moe.vendor.fp4-block64.v8", 8U, 8U,
                         [](const er::ModelDescriptor& descriptor) {
                           const auto* component =
                               er::find_routed_component(descriptor, "decoder");
                           return component != nullptr &&
                                          component->encoding ==
                                              "fp4.vendor.e2m1.block64"
                                      ? er::Status::success()
                                      : er::Status(
                                            er::ErrorCode::invalid_argument,
                                            "unsupported fixture encoding");
                         }}}})
              .ok(),
          "provider registry rejected the complete generic provider");
  const auto bound = registry.bind(parsed.descriptor);
  require(bound.status.ok() && bound.provider.providers.size() == 2U &&
              bound.provider.providers[0U].name ==
                  "incomplete-high-priority" &&
              bound.provider.providers[1U].name == "generic-sm86-fp4" &&
              bound.provider.program.operations.size() == 3U &&
              bound.provider.program.operations[1U].logical_layer == 0U &&
              bound.provider.program.operations.front()
                      .tensor_bindings.at("input_norm") ==
                  "model.layers.0.norm.weight" &&
              bound.provider.program.operations[1U]
                      .routed_component_index == 0U,
          "unknown artifact was not composed solely by operation capabilities");

  auto incomplete_ir = parsed.descriptor;
  incomplete_ir.operation_program.pop_back();
  require(!registry.bind(incomplete_ir).status.ok(),
          "VM accepted a routed layer without its expert operation");
  const auto duplicate_binding = std::string(artifact) +
      "operation_tensor\t0\tinput_norm\tmodel.other.weight\n";
  require(!er::parse_model_descriptor_artifact(
               duplicate_binding, content_hash, 0x4100U)
               .status.ok(),
          "VM artifact accepted a duplicate operation tensor role");
}

er::ExecutionValue fixture_scalar(
    std::uint64_t value,
    std::string abi = "fixture.scalar.u64.v1") {
  auto owner = std::make_shared<std::uint64_t>(value);
  return {std::move(abi), "host.fixture", owner,
          reinterpret_cast<const std::byte*>(owner.get()), sizeof(value)};
}

std::uint64_t fixture_scalar_value(const er::ExecutionValue& value) {
  std::uint64_t result{};
  require(value.valid() && value.bytes == sizeof(result),
          "fixture execution value is not a scalar");
  std::memcpy(&result, value.data, sizeof(result));
  return result;
}

class FixtureModelTensorStore final : public er::IModelTensorStore {
 public:
  FixtureModelTensorStore() {
    auto tensor = std::make_shared<er::ImmutableModelTensor>();
    tensor->name = "fixture.bias";
    tensor->encoding = "fixture.u64";
    tensor->quant_abi = 1U;
    tensor->shape = {1U};
    tensor->value = fixture_scalar(7U, "fixture.tensor.u64.v1");
    tensor_ = std::move(tensor);
  }

  er::ResolveModelTensorResult resolve(std::string_view name) override {
    ++resolves;
    if (name != tensor_->name)
      return {{er::ErrorCode::invalid_argument,
               "fixture tensor does not exist"},
              {}};
    return {er::Status::success(), tensor_};
  }

  std::uint32_t resolves{};

 private:
  std::shared_ptr<const er::ImmutableModelTensor> tensor_;
};

struct FixtureOperationProviderControl final {
  std::map<std::uint32_t, std::uint32_t> prepared;
  std::map<std::uint32_t, std::uint32_t> executed;
  std::optional<std::uint32_t> delayed_operation;
  std::optional<std::uint32_t> failed_operation;
  std::optional<std::uint32_t> wrong_abi_operation;
  std::uint32_t states_created{};
  std::uint32_t states_destroyed{};
  std::uint32_t states_alive{};
  std::uint32_t cancellations{};
  std::uint32_t exact_prepared{};
  std::uint32_t exact_synchronizations{};
  std::uint32_t exact_executions{};
  std::vector<bool> exact_draft_requests;
};

class FixtureOperationRequestState final
    : public er::IOperationProviderRequestState {
 public:
  explicit FixtureOperationRequestState(
      std::shared_ptr<FixtureOperationProviderControl> control)
      : control_(std::move(control)) {
    ++control_->states_alive;
  }
  ~FixtureOperationRequestState() override {
    --control_->states_alive;
    ++control_->states_destroyed;
  }

 private:
  std::shared_ptr<FixtureOperationProviderControl> control_;
};

class FixturePreparedOperation final : public er::IPreparedOperation {
 public:
  std::uint32_t logical_operation{};
  std::uint64_t add{};
  std::string output_abi;
};

class FixturePreparedExactDecode final : public er::IPreparedOperation {};

class FixtureCallableOperationProvider final : public er::IOperationProvider {
 public:
  explicit FixtureCallableOperationProvider(
      std::shared_ptr<FixtureOperationProviderControl> control)
      : control_(std::move(control)) {}

  er::PrepareOperationResult prepare(
      const er::OperationPreparationContext& context) override {
    if (context.operation.output_bindings.size() != 1U)
      return {{er::ErrorCode::invalid_argument,
               "fixture operation requires one output"},
              {}};
    auto prepared = std::make_shared<FixturePreparedOperation>();
    prepared->logical_operation = context.operation.logical_operation;
    const auto parameter = context.operation.parameters.find("add");
    if (parameter != context.operation.parameters.end())
      prepared->add = parameter->second;
    for (const auto& tensor : context.tensors) {
      if (tensor.role != "bias" || !tensor.tensor)
        return {{er::ErrorCode::invalid_argument,
                 "fixture tensor binding is invalid"},
                {}};
      prepared->add += fixture_scalar_value(tensor.tensor->value);
    }
    prepared->output_abi =
        context.operation.output_bindings.begin()->second.abi;
    ++control_->prepared[prepared->logical_operation];
    return {er::Status::success(), std::move(prepared)};
  }

  er::CreateOperationRequestStateResult create_request_state(
      const er::ProgramRequestContext&) override {
    ++control_->states_created;
    return {er::Status::success(),
            std::make_shared<FixtureOperationRequestState>(control_)};
  }

  er::OperationExecutionHandle execute(
      const er::IPreparedOperation& operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& request_state,
      const er::OperationInvocation& invocation) override {
    const auto* prepared =
        dynamic_cast<const FixturePreparedOperation*>(&operation);
    if (prepared == nullptr || !request_state || invocation.inputs.size() != 1U)
      return {};
    ++control_->executed[prepared->logical_operation];
    struct InvocationState final {
      std::shared_ptr<FixtureOperationProviderControl> control;
      std::shared_ptr<er::IOperationProviderRequestState> request_state;
      std::uint32_t logical_operation{};
      std::uint64_t value{};
      std::string output_abi;
      std::uint32_t polls{};
      bool terminal{};
    };
    auto state = std::make_shared<InvocationState>();
    state->control = control_;
    state->request_state = request_state;
    state->logical_operation = prepared->logical_operation;
    state->value = fixture_scalar_value(invocation.inputs.front()) +
                   prepared->add;
    state->output_abi = prepared->output_abi;
    return er::OperationExecutionHandle::from_callbacks(
        [state]() -> std::optional<er::OperationExecutionResult> {
          if (state->terminal) return std::nullopt;
          if (state->control->delayed_operation ==
                  state->logical_operation &&
              state->polls++ == 0U)
            return std::nullopt;
          state->terminal = true;
          if (state->control->failed_operation == state->logical_operation)
            return er::OperationExecutionResult{
                {er::ErrorCode::internal, "fixture operation failed"}, {}};
          auto abi = state->output_abi;
          if (state->control->wrong_abi_operation ==
              state->logical_operation)
            abi = "fixture.wrong-abi.v1";
          return er::OperationExecutionResult{
              er::Status::success(),
              {fixture_scalar(state->value, std::move(abi))}};
        },
        [state] {
          if (state->terminal) return;
          state->terminal = true;
          ++state->control->cancellations;
        });
  }

  er::PrepareOperationResult prepare_exact_decode(
      const er::ExactDecodePreparationContext& context) override {
    if (context.compiled.maximum_emitted_tokens != 2U ||
        context.tensors.size() != 1U ||
        context.tensors.front().role != "bias" ||
        !context.tensors.front().tensor ||
        context.tensors.front().tensor->name != "fixture.bias")
      return {{er::ErrorCode::invalid_argument,
               "fixture exact decode binding is invalid"},
              {}};
    ++control_->exact_prepared;
    return {er::Status::success(),
            std::make_shared<FixturePreparedExactDecode>()};
  }

  er::Status synchronize_exact_decode(
      const er::IPreparedOperation& operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& request_state,
      const er::ExactDecodeSynchronization& synchronization) override {
    if (dynamic_cast<const FixturePreparedExactDecode*>(&operation) == nullptr ||
        !request_state)
      return {er::ErrorCode::invalid_argument,
              "fixture exact decode synchronization is invalid"};
    ++control_->exact_synchronizations;
    control_->exact_draft_requests.push_back(synchronization.produce_draft);
    return er::Status::success();
  }

  er::ExactDecodeExecutionHandle execute_exact_decode(
      const er::IPreparedOperation& operation,
      const std::shared_ptr<er::IOperationProviderRequestState>& request_state,
      const er::ExactDecodeInvocation& invocation) override {
    if (dynamic_cast<const FixturePreparedExactDecode*>(&operation) == nullptr ||
        !request_state)
      return {};
    ++control_->exact_executions;
    struct State final {
      er::ExactDecodeExecutionResult result;
      bool terminal{};
    };
    auto state = std::make_shared<State>();
    state->result = {er::Status::success(),
                     {invocation.guaranteed_token,
                      invocation.guaranteed_token + 1U},
                     invocation.guaranteed_token + 2U, 2U};
    return er::ExactDecodeExecutionHandle::from_callbacks(
        [state]() -> std::optional<er::ExactDecodeExecutionResult> {
          if (state->terminal) return std::nullopt;
          state->terminal = true;
          return std::move(state->result);
        },
        [state] { state->terminal = true; });
  }

 private:
  std::shared_ptr<FixtureOperationProviderControl> control_;
};

void test_schema_v3_callable_program_is_exact_and_family_neutral() {
  constexpr std::string_view artifact =
      "expert-runtime-model-v1\n"
      "model\t3\tunknown.future.architecture\t8192\t4096\t64\n"
      "model_tensor\tbias\tfixture.bias\n"
      "program_input\ttoken\trequest.token\tfixture.scalar.u64.v1\n"
      "program_output\tnext_token\tresponse.token\tfixture.scalar.u64.v1\n"
      "kernel\tfixture.embed.v7\t7\n"
      "kernel\tfixture.transform.v11\t11\n"
      "kernel\tfixture.head.v3\t3\n"
      "kernel\tfixture.exact-decode.v1\t1\n"
      "exact_decode\tfixture.exact-decode.v1\t1\t2\n"
      "exact_decode_parameter\tdraft_component_index\t0\n"
      "exact_decode_tensor\tbias\tfixture.bias\n"
      "layer\t0\tfixture.transform.v11\t11\t-\t0\n"
      "operation\t0\t-\tfixture.embed.v7\t7\t-\t0\n"
      "operation_parameter\t0\tadd\t1\n"
      "operation_input\t0\ttoken\trequest.token\tfixture.scalar.u64.v1\n"
      "operation_output\t0\thidden\tembedded.hidden\tfixture.scalar.u64.v1\n"
      "operation\t1\t0\tfixture.transform.v11\t11\t-\t0\n"
      "operation_tensor\t1\tbias\tfixture.bias\n"
      "operation_input\t1\thidden\tembedded.hidden\tfixture.scalar.u64.v1\n"
      "operation_output\t1\thidden\ttransformed.hidden\tfixture.scalar.u64.v1\n"
      "operation\t2\t-\tfixture.head.v3\t3\t-\t0\n"
      "operation_parameter\t2\tadd\t100\n"
      "operation_input\t2\thidden\ttransformed.hidden\tfixture.scalar.u64.v1\n"
      "operation_output\t2\ttoken\tresponse.token\tfixture.scalar.u64.v1\n";
  er::Sha256Digest content_hash{};
  content_hash[0] = std::byte{0x93};
  auto parsed =
      er::parse_model_descriptor_artifact(artifact, content_hash, 0x9300U);
  require(parsed.status.ok() && parsed.descriptor.schema_version == 3U &&
              parsed.descriptor.operation_program.front().logical_layer ==
                  er::kModelLevelOperationLayer &&
              parsed.descriptor.operation_program.back().logical_layer ==
                  er::kModelLevelOperationLayer,
          "schema v3 parser lost model-level operations or SSA endpoints");

  auto unavailable = parsed.descriptor;
  unavailable.operation_program[1U].input_bindings.at("hidden").value =
      "future.hidden";
  require(!er::validate_model_descriptor(unavailable).ok(),
          "schema v3 accepted use-before-produce data flow");
  auto duplicate = parsed.descriptor;
  duplicate.operation_program[1U].output_bindings.at("hidden").value =
      "embedded.hidden";
  require(!er::validate_model_descriptor(duplicate).ok(),
          "schema v3 accepted two producers for one SSA value");
  auto wrong_input_abi = parsed.descriptor;
  wrong_input_abi.operation_program[1U].input_bindings.at("hidden").abi =
      "fixture.scalar.u32.v1";
  require(!er::validate_model_descriptor(wrong_input_abi).ok(),
          "schema v3 accepted an operation input ABI mismatch");
  auto missing_exact_kernel = parsed.descriptor;
  missing_exact_kernel.required_kernels.pop_back();
  require(!er::validate_model_descriptor(missing_exact_kernel).ok(),
          "schema v3 accepted an unbound exact decode program");
  auto scalar_exact = parsed.descriptor;
  scalar_exact.exact_decode_program->maximum_emitted_tokens = 1U;
  require(!er::validate_model_descriptor(scalar_exact).ok(),
          "schema v3 accepted a scalar exact decode service");

  auto first_control = std::make_shared<FixtureOperationProviderControl>();
  auto second_control = std::make_shared<FixtureOperationProviderControl>();
  auto first_provider =
      std::make_shared<FixtureCallableOperationProvider>(first_control);
  auto second_provider =
      std::make_shared<FixtureCallableOperationProvider>(second_control);
  require(!first_provider->supports_request_state_retention() &&
              !second_provider->supports_request_state_retention(),
          "callable providers claimed retention without implementing it");
  er::ExecutionProviderRegistry registry;
  require(registry
              .add({"metadata-only-high-priority", 100U,
                    {er::KernelCapability{"fixture.embed.v7", 7U, 7U},
                     er::KernelCapability{"fixture.transform.v11", 11U, 11U},
                     er::KernelCapability{"fixture.head.v3", 3U, 3U},
                     er::KernelCapability{"fixture.exact-decode.v1", 1U,
                                          1U}}})
              .ok() &&
              registry
                  .add({"callable-edges", 20U,
                        {er::KernelCapability{"fixture.embed.v7", 7U, 7U},
                         er::KernelCapability{"fixture.head.v3", 3U, 3U},
                         er::KernelCapability{"fixture.exact-decode.v1", 1U,
                                              1U}},
                        first_provider})
                  .ok() &&
              registry
                  .add({"callable-transform", 10U,
                        {er::KernelCapability{"fixture.transform.v11", 11U,
                                              11U}},
                        second_provider})
                  .ok(),
          "schema v3 fixture provider registration failed");
  const auto metadata = registry.bind(parsed.descriptor);
  require(metadata.status.ok() && metadata.provider.providers.size() == 1U &&
              !metadata.provider.providers.front().implementation,
          "metadata binding unexpectedly required a callable provider");
  const auto executable = registry.bind(
      parsed.descriptor, er::ExecutionProviderBindingMode::executable);
  require(executable.status.ok() &&
              executable.provider.providers.size() == 2U &&
              executable.provider.program.values.size() == 4U &&
              executable.provider.program.inputs.size() == 1U &&
              executable.provider.program.outputs.size() == 1U &&
              executable.provider.program.exact_decode.has_value(),
          "executable binding did not compile the capability-composed SSA plan");

  er::MoeProgramExecutor rejected_metadata;
  require(!er::MoeProgramExecutor::create(
               parsed.descriptor, metadata.provider, nullptr,
               rejected_metadata)
               .ok(),
          "callable interpreter accepted a metadata-only provider plan");
  er::MoeProgramExecutor rejected_tensorless;
  require(!er::MoeProgramExecutor::create(
               parsed.descriptor, executable.provider, nullptr,
               rejected_tensorless)
               .ok(),
          "callable interpreter let a provider reopen an undeclared tensor");

  FixtureModelTensorStore tensor_store;
  er::MoeProgramExecutor executor;
  require(er::MoeProgramExecutor::create(
              parsed.descriptor, executable.provider, &tensor_store, executor)
              .ok() &&
              executor.valid() && tensor_store.resolves == 1U,
          "schema v3 callable program preparation failed");
  first_control->delayed_operation = std::nullopt;
  second_control->delayed_operation = 1U;
  er::ProgramExecutionRequest request;
  request.context.request_id = 17U;
  request.inputs.emplace("token", fixture_scalar(5U));
  auto started = executor.execute(std::move(request));
  require(started.status.ok() && started.handle.valid(),
          "schema v3 callable request was rejected");
  require(!started.handle.poll(),
          "asynchronous provider was not polled by the common interpreter");
  const auto completed = started.handle.poll();
  require(completed && completed->status.ok() &&
              completed->outputs.size() == 1U &&
              fixture_scalar_value(completed->outputs.at("next_token")) ==
                  113U &&
              first_control->executed[0U] == 1U &&
              second_control->executed[1U] == 1U &&
              first_control->executed[2U] == 1U &&
              first_control->states_created == 1U &&
              second_control->states_created == 1U &&
              first_control->states_alive == 0U &&
              second_control->states_alive == 0U,
          "common interpreter lost exact values, order, or request state lifecycle");

  second_control->delayed_operation = std::nullopt;
  second_control->failed_operation = 1U;
  er::ProgramExecutionRequest failing_request;
  failing_request.context.request_id = 18U;
  failing_request.inputs.emplace("token", fixture_scalar(8U));
  auto failing = executor.execute(std::move(failing_request));
  const auto head_before_failure = first_control->executed[2U];
  const auto failed = failing.handle.poll();
  require(failing.status.ok() && failed && !failed->status.ok() &&
              failed->outputs.empty() &&
              first_control->executed[2U] == head_before_failure,
          "operation failure published partial output or ran downstream work");

  second_control->failed_operation = std::nullopt;
  second_control->wrong_abi_operation = 1U;
  er::ProgramExecutionRequest wrong_abi_request;
  wrong_abi_request.context.request_id = 19U;
  wrong_abi_request.inputs.emplace("token", fixture_scalar(8U));
  auto wrong_abi = executor.execute(std::move(wrong_abi_request));
  const auto head_before_wrong_abi = first_control->executed[2U];
  const auto rejected_output = wrong_abi.handle.poll();
  require(wrong_abi.status.ok() && rejected_output &&
              !rejected_output->status.ok() &&
              rejected_output->outputs.empty() &&
              first_control->executed[2U] == head_before_wrong_abi,
          "interpreter accepted a provider output with the wrong ABI");

  second_control->wrong_abi_operation = std::nullopt;
  second_control->delayed_operation = 1U;
  er::ProgramExecutionRequest cancelled_request;
  cancelled_request.context.request_id = 20U;
  cancelled_request.inputs.emplace("token", fixture_scalar(9U));
  auto cancelled = executor.execute(std::move(cancelled_request));
  const auto head_before_cancel = first_control->executed[2U];
  require(cancelled.status.ok() && !cancelled.handle.poll(),
          "cancellation fixture did not reach its asynchronous operation");
  cancelled.handle.cancel();
  cancelled.handle.cancel();
  require(second_control->cancellations == 1U &&
              first_control->executed[2U] == head_before_cancel &&
              first_control->states_alive == 0U &&
              second_control->states_alive == 0U,
          "program cancellation was not idempotent or leaked provider state");

  er::ProgramExecutionRequest extra_input;
  extra_input.context.request_id = 21U;
  extra_input.inputs.emplace("token", fixture_scalar(1U));
  extra_input.inputs.emplace("implicit.hidden", fixture_scalar(2U));
  require(!executor.execute(std::move(extra_input)).status.ok(),
          "interpreter accepted a family-specific implicit input");

  second_control->delayed_operation = std::nullopt;
  const auto first_states_before_session = first_control->states_created;
  const auto second_states_before_session = second_control->states_created;
  er::ProgramRequestContext session_context;
  session_context.request_id = 30U;
  session_context.parameters.emplace("reserved_context_tokens", 128U);
  auto begun = executor.begin_session(std::move(session_context));
  require(begun.status.ok() && begun.session.valid() &&
              first_control->states_created ==
                  first_states_before_session + 1U &&
              second_control->states_created ==
                  second_states_before_session + 1U &&
              first_control->states_alive == 1U &&
              second_control->states_alive == 1U,
          "persistent VM session did not create one state per provider");
  for (const auto [input, expected] :
       std::array<std::pair<std::uint64_t, std::uint64_t>, 2U>{
           {{1U, 109U}, {2U, 110U}}}) {
    std::map<std::string, er::ExecutionValue, std::less<>> inputs;
    inputs.emplace("token", fixture_scalar(input));
    auto step = begun.session.execute(std::move(inputs));
    auto result = step.handle.poll();
    require(step.status.ok() && result && result->status.ok() &&
                fixture_scalar_value(result->outputs.at("next_token")) ==
                    expected &&
                first_control->states_created ==
                    first_states_before_session + 1U &&
                second_control->states_created ==
                    second_states_before_session + 1U &&
                first_control->states_alive == 1U &&
                second_control->states_alive == 1U,
            "VM session recreated or destroyed provider state between steps");
  }
  require(begun.session.exact_decode_available(),
          "artifact-declared exact decode service is unavailable");
  const auto synchronized =
      begun.session.synchronize_exact_decode(110U, 1U, true);
  const auto boundary_draft =
      begun.session.synchronize_exact_decode(110U, 124U, true);
  const auto boundary_no_draft =
      begun.session.synchronize_exact_decode(110U, 126U, true);
  auto exact = begun.session.execute_exact_decode(110U, 2U, 128U);
  const auto exact_result = exact.handle.poll();
  require(synchronized.ok() && boundary_draft.ok() &&
              boundary_no_draft.ok() && exact.status.ok() && exact_result &&
              exact_result->status.ok() &&
              exact_result->emitted_tokens ==
                  std::vector<std::uint32_t>({110U, 111U}) &&
              exact_result->next_token == 112U &&
              exact_result->positions_advanced == 2U &&
              first_control->exact_prepared == 1U &&
              first_control->exact_synchronizations == 3U &&
              first_control->exact_draft_requests ==
                  std::vector<bool>({true, true, false}) &&
              first_control->exact_executions == 1U,
          "exact decode binding lost synchronization or token semantics");
  begun.session.cancel();
  require(first_control->states_alive == 0U &&
              second_control->states_alive == 0U &&
              !begun.session.valid(),
          "closing a VM session leaked persistent provider state");

  second_control->delayed_operation = 1U;
  er::ProgramRequestContext active_context;
  active_context.request_id = 31U;
  auto active_session = executor.begin_session(std::move(active_context));
  std::map<std::string, er::ExecutionValue, std::less<>> active_inputs;
  active_inputs.emplace("token", fixture_scalar(3U));
  auto active_step = active_session.session.execute(std::move(active_inputs));
  require(active_session.status.ok() && active_step.status.ok() &&
              !active_step.handle.poll(),
          "persistent cancellation fixture did not become asynchronous");
  active_session.session.cancel();
  const auto active_cancelled = active_step.handle.poll();
  require(active_cancelled && !active_cancelled->status.ok() &&
              active_cancelled->status.code() == er::ErrorCode::cancelled &&
              first_control->states_alive == 0U &&
              second_control->states_alive == 0U,
          "session cancellation did not terminate its active step and state");
}

void test_schema_v2_expresses_model_derived_hybrid_moe_topology() {
  constexpr std::string_view convolution =
      "block.causal-short-conv.gated.v1";
  constexpr std::string_view attention =
      "block.full-attention.gqa.qk-norm.v1";
  constexpr std::string_view dense_ffn = "ffn.swiglu.dense.v1";
  constexpr std::string_view router = "router.sigmoid-bias.topk.v1";
  constexpr std::string_view routed_ffn = "moe.swiglu.routed.v1";
  constexpr std::array attention_layers{2U, 6U, 10U, 14U, 18U, 21U};

  er::ModelDescriptor descriptor;
  descriptor.schema_version = 2U;
  descriptor.architecture_id = "unseen.hybrid.moe";
  descriptor.content_hash[0] = std::byte{0x4c};
  descriptor.vocab_size = 65'536U;
  descriptor.max_context_tokens = 128'000U;
  descriptor.hidden_size = 2'048U;
  descriptor.attributes = {
      {"attention_heads", 32U},
      {"kv_heads", 8U},
      {"conv_cache_length", 3U},
      {"dense_prefix_layers", 2U},
  };

  er::RoutedExpertComponentDescriptor component;
  component.name = "decoder";
  component.namespace_id = 0x4c464d32U;
  component.layer_count = 22U;
  component.experts_per_layer = 32U;
  component.route_width = 4U;
  component.shared_experts_per_layer = 0U;
  component.hidden_size = 2'048U;
  component.intermediate_size = 1'792U;
  component.execution_capability = routed_ffn;
  component.execution_abi = 1U;
  component.source_abi = er::kExpertSourceAbiExpertPackV1;
  component.encoding_abi = er::kExpertEncodingAbiInt8PerRow;
  component.encoding = "int8.symmetric.per-row";
  component.router.capability = router;
  component.router.abi_version = 1U;
  component.router.parameters = {
      {"normalize", 1U},
      {"use_expert_bias", 1U},
  };
  descriptor.routed_components.push_back(component);
  descriptor.required_kernels = {
      {std::string(convolution), 1U},
      {std::string(attention), 1U},
      {std::string(dense_ffn), 1U},
      {std::string(router), 1U},
      {std::string(routed_ffn), 1U},
  };

  std::uint32_t logical_operation = 0U;
  for (std::uint32_t layer = 0U; layer < 24U; ++layer) {
    const bool uses_attention =
        std::find(attention_layers.begin(), attention_layers.end(), layer) !=
        attention_layers.end();
    const auto block = uses_attention ? attention : convolution;
    const bool dense = layer < 2U;
    descriptor.layer_program.push_back(
        {layer, std::string(block), 1U, dense ? "" : "decoder",
         dense ? 0U : layer - 2U, {}});
    descriptor.operation_program.push_back(
        {logical_operation++, layer, std::string(block), 1U, "", 0U, {}});
    if (dense) {
      descriptor.operation_program.push_back(
          {logical_operation++, layer, std::string(dense_ffn), 1U, "", 0U,
           {}});
      continue;
    }
    descriptor.operation_program.push_back(
        {logical_operation++, layer, std::string(router), 1U, "decoder",
         layer - 2U, {}});
    descriptor.operation_program.push_back(
        {logical_operation++, layer, std::string(routed_ffn), 1U, "decoder",
         layer - 2U, {}});
  }
  require(logical_operation == 70U &&
              er::validate_model_descriptor(descriptor).ok(),
          "model-derived hybrid MoE descriptor is not valid schema-v2 IR");

  er::ExecutionProviderRegistry registry;
  require(registry
              .add({"legacy-attention-moe", 100U,
                    {er::KernelCapability{std::string(attention), 1U, 1U},
                     er::KernelCapability{std::string(routed_ffn), 1U,
                                          1U}}})
              .ok(),
          "hybrid fixture rejected the incomplete legacy provider");
  require(!registry.bind(descriptor).status.ok(),
          "hybrid model bound without convolution, dense FFN, and router");

  const auto validate_geometry = [](const er::ModelDescriptor& candidate) {
    const auto* decoder = er::find_routed_component(candidate, "decoder");
    if (decoder == nullptr || decoder->layer_count != 22U ||
        decoder->experts_per_layer != 32U || decoder->route_width != 4U ||
        decoder->hidden_size != 2'048U ||
        decoder->intermediate_size != 1'792U ||
        decoder->source_abi != er::kExpertSourceAbiExpertPackV1 ||
        decoder->encoding_abi != er::kExpertEncodingAbiInt8PerRow ||
        decoder->router.parameters.at("normalize") != 1U ||
        decoder->router.parameters.at("use_expert_bias") != 1U)
      return er::Status(er::ErrorCode::invalid_argument,
                        "unsupported hybrid MoE geometry");
    return er::Status::success();
  };
  require(registry
              .add({"generic-hybrid-sm86", 10U,
                    {er::KernelCapability{std::string(convolution), 1U, 1U},
                     er::KernelCapability{std::string(attention), 1U, 1U},
                     er::KernelCapability{std::string(dense_ffn), 1U, 1U},
                     er::KernelCapability{std::string(router), 1U, 1U},
                     er::KernelCapability{std::string(routed_ffn), 1U, 1U,
                                          validate_geometry}}})
              .ok(),
          "hybrid fixture rejected the complete capability provider");
  const auto bound = registry.bind(descriptor);
  require(bound.status.ok() && bound.provider.providers.size() == 2U &&
              bound.provider.providers[0U].name ==
                  "legacy-attention-moe" &&
              bound.provider.providers[1U].name == "generic-hybrid-sm86" &&
              bound.provider.program.layers.size() == 24U &&
              bound.provider.program.operations.size() == 70U &&
              !bound.provider.program.layers[1U].routed_component_index &&
              bound.provider.program.layers[2U].routed_component_index == 0U &&
              bound.provider.program.layers[23U].component_layer == 21U &&
              bound.provider.program.operations[5U]
                      .routed_component_index == 0U &&
              bound.provider.program.operations.back().component_layer == 21U,
          "capability-only binding lost the hybrid model operation topology");
}

void test_generic_expert_catalog_uses_descriptor_cardinality() {
  const auto root = std::filesystem::temp_directory_path() /
                    "expert-runtime-generic-catalog";
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
  std::filesystem::create_directories(root);
  {
    std::ofstream extents(root / "extents.tsv", std::ios::binary);
    extents << "fixture-extents-v1\n";
    for (std::uint32_t index = 0U; index < 6U; ++index)
      extents << "0\t4\t" << index * 4U << "\tpack.bin\n";
  }
  {
    std::ofstream catalog(root / "catalog.tsv", std::ios::binary);
    catalog << "fixture-catalog-v1\n";
    const std::string hash(64U, '0');
    for (std::uint32_t layer = 0U; layer < 2U; ++layer) {
      for (std::uint32_t expert = 0U; expert < 3U; ++expert) {
        const auto index = layer * 3U + expert;
        catalog << layer << '\t' << expert << "\t4\t" << hash << '\t'
                << index << "\t1\n";
      }
    }
  }
  er::ExpertCatalog catalog;
  er::ExpertCatalogConfig config{
      root, root, 2U, 3U, 64U, 32U, 32U, 4U,
      3ULL * 64U * 32U * sizeof(float), 4U,
      er::kExpertSourceAbiExpertPackV1, 0U, 0U, 1U,
      {{"fixture-extents-v1", "fixture-catalog-v1", 1U, true}}};
  const auto loaded = er::ExpertCatalog::load(config, catalog);
  const auto* record = catalog.find(1U, 2U);
  require(loaded.ok() && catalog.layer_count() == 2U &&
              catalog.experts_per_layer() == 3U && catalog.size() == 6U &&
              record != nullptr && record->hidden == 64U &&
              record->intermediate == 32U &&
              record->extents.front().source_offset == 20U,
          "generic expert catalog retained model-specific cardinality");
  std::filesystem::remove_all(root, cleanup_error);
}

void test_universal_worker_launch_preserves_provider_extensions() {
  constexpr std::array arguments{
      std::string_view{"--max-context=131072"},
      std::string_view{"--ram-cache-gib=96"},
      std::string_view{"--vram-cache-gib=18"},
      std::string_view{"--routed-vram-policy=fit"},
      std::string_view{"--capacity=3"},
      std::string_view{"--kv-cache-mib=3072"},
      std::string_view{"--kv-page-tokens=128"},
      std::string_view{"--kv-cache-dtype=fp8-e4m3-per-head"},
      std::string_view{"--placement-profile=balanced"},
      std::string_view{"--prefill-chunk-limit=512"},
      std::string_view{"--profile-gpu-phases"},
      std::string_view{"--active-expert-devices=1,2"},
      std::string_view{"--active-expert-device-cache-gib=14"},
      std::string_view{"--active-expert-host-cache-gib=32"},
      std::string_view{"--provider-fp4-pipeline=vendor-x"},
      std::string_view{"--provider-background-compile"}};
  const auto parsed = er::parse_worker_launch_options(arguments);
  require(parsed.status.ok() && parsed.options.max_context == 131072U &&
              parsed.options.capacity == 3U &&
              parsed.options.routed_vram_policy == "fit" &&
              parsed.options.kv_cache_dtype == "fp8-e4m3-per-head" &&
              parsed.options.prefill_chunk_limit == 512U &&
              parsed.options.profile_gpu_phases &&
              !parsed.options.discover_active_expert_devices &&
              parsed.options.active_expert_devices ==
                  std::vector<int>({1, 2}) &&
              parsed.options.active_expert_device_cache_gib == 14U &&
              parsed.options.active_expert_host_cache_gib == 32U &&
              parsed.options.extensions.at("provider-fp4-pipeline") ==
                  "vendor-x" &&
              !parsed.options.extensions.at("provider-background-compile"),
          "universal worker launch lost an opaque provider extension");
  constexpr std::array auto_devices{
      std::string_view{"--max-context=131072"},
      std::string_view{"--ram-cache-gib=48"},
      std::string_view{"--vram-cache-gib=12"},
      std::string_view{"--capacity=1"},
      std::string_view{"--kv-cache-mib=5120"},
      std::string_view{"--kv-page-tokens=256"},
      std::string_view{"--placement-profile=balanced"},
      std::string_view{"--active-expert-devices=auto"},
      std::string_view{"--active-expert-device-cache-gib=14"},
      std::string_view{"--active-expert-host-cache-gib=32"}};
  const auto auto_parsed = er::parse_worker_launch_options(auto_devices);
  require(auto_parsed.status.ok() &&
              auto_parsed.options.discover_active_expert_devices &&
              auto_parsed.options.active_expert_devices.empty(),
          "universal worker launch rejected automatic secondary discovery");
  constexpr std::array duplicate{
      std::string_view{"--max-context=1"},
      std::string_view{"--max-context=2"}};
  require(!er::parse_worker_launch_options(duplicate).status.ok(),
          "universal worker launch accepted duplicate options");
}


template <typename T>
void write_le(std::byte* output, T value) {
  using Unsigned = std::make_unsigned_t<T>;
  const auto converted = static_cast<Unsigned>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output[index] = static_cast<std::byte>(
        (converted >> static_cast<unsigned>(index * 8U)) & 0xffU);
  }
}

struct FixtureRecord final {
  er::ExpertKey key;
  er::PayloadRecord record;
  std::vector<std::byte> bytes;
};

FixtureRecord make_record(std::uint32_t expert_id,
                          std::uint64_t file_offset = 0,
                          std::uint32_t layer = 3) {
  FixtureRecord result;
  result.key = {0x0123456789abcdefULL, layer, expert_id,
                er::kExpertQuantAbiInt8PerRow};
  result.bytes.resize(er::kExpertPackAlignment);
  for (std::size_t index = er::kExpertHeaderBytes; index < result.bytes.size();
       ++index) {
    result.bytes[index] = static_cast<std::byte>((index * 37U + expert_id) & 0xffU);
  }

  auto* header = result.bytes.data();
  std::memcpy(header, "EPEXPR01", 8);
  write_le<std::uint16_t>(header + 8, er::kExpertPackVersion);
  write_le<std::uint16_t>(header + 10, er::kExpertHeaderBytes);
  write_le<std::uint32_t>(header + 12, 0x0fU);
  write_le<std::uint32_t>(header + 16, er::kExpertQuantAbiInt8PerRow);
  write_le<std::int32_t>(header + 20, static_cast<std::int32_t>(layer));
  write_le<std::int32_t>(header + 24, static_cast<std::int32_t>(expert_id));
  write_le<std::uint32_t>(header + 28, 16);
  write_le<std::uint32_t>(header + 32, 8);
  write_le<std::uint32_t>(header + 36, 16);
  write_le<std::uint32_t>(header + 40, 0);
  write_le<std::uint64_t>(header + 44, result.bytes.size());
  write_le<std::uint64_t>(header + 52, 256);
  write_le<std::uint64_t>(header + 60, 256);
  write_le<std::uint64_t>(header + 68, 512);
  write_le<std::uint64_t>(header + 76, 64);
  write_le<std::uint64_t>(header + 84, 768);
  write_le<std::uint64_t>(header + 92, 128);
  write_le<std::uint64_t>(header + 100, 1024);
  write_le<std::uint64_t>(header + 108, 64);
  const auto digest = er::sha256(std::span<const std::byte>(result.bytes).subspan(
      er::kExpertHeaderBytes));
  std::copy(digest.begin(), digest.end(), header + 116);

  result.record.path = "fixture.qpack";
  result.record.record_offset = file_offset;
  result.record.stored_bytes = result.bytes.size();
  result.record.decoded_bytes = 3ULL * 16 * 8 * sizeof(float);
  result.record.header_bytes = er::kExpertHeaderBytes;
  result.record.alignment = er::kExpertPackAlignment;
  result.record.payload_sha256 = digest;
  return result;
}

FixtureRecord make_fp4_record(std::uint32_t expert_id,
                              std::uint64_t file_offset = 0,
                              std::uint32_t layer = 3) {
  // Block-32 aligned FP4-E2M1/UE8M0 record: hidden 32, intermediate 32.
  FixtureRecord result;
  result.key = {0x0123456789abcdefULL, layer, expert_id,
                er::kExpertEncodingAbiFp4Block32};
  result.bytes.resize(er::kExpertPackAlignment);
  for (std::size_t index = er::kExpertHeaderBytes; index < result.bytes.size();
       ++index) {
    result.bytes[index] = static_cast<std::byte>((index * 37U + expert_id) & 0xffU);
  }
  // Scale spans must carry valid UE8M0 codes in [1, 254].
  for (const auto& span : {std::pair{1280ULL, 64ULL}, {2048ULL, 32ULL}}) {
    for (std::size_t index = span.first; index < span.first + span.second;
         ++index) {
      const auto code = static_cast<unsigned>(result.bytes[index]) % 254U;
      result.bytes[index] = static_cast<std::byte>(code + 1U);
    }
  }

  auto* header = result.bytes.data();
  std::memcpy(header, "EPEXPR01", 8);
  write_le<std::uint16_t>(header + 8, er::kExpertPackVersion);
  write_le<std::uint16_t>(header + 10, er::kExpertHeaderBytes);
  write_le<std::uint32_t>(header + 12, 0x0fU);
  write_le<std::uint32_t>(header + 16, er::kExpertRecordAbiFp4Block32);
  write_le<std::int32_t>(header + 20, static_cast<std::int32_t>(layer));
  write_le<std::int32_t>(header + 24, static_cast<std::int32_t>(expert_id));
  write_le<std::uint32_t>(header + 28, 32);
  write_le<std::uint32_t>(header + 32, 32);
  write_le<std::uint32_t>(header + 36, 64);
  write_le<std::uint32_t>(header + 40, 0);
  write_le<std::uint64_t>(header + 44, result.bytes.size());
  write_le<std::uint64_t>(header + 52, 256);    // gate_up_q: 2*32*32/2
  write_le<std::uint64_t>(header + 60, 1024);
  write_le<std::uint64_t>(header + 68, 1280);   // gate_up scales: 2*32*32/32
  write_le<std::uint64_t>(header + 76, 64);
  write_le<std::uint64_t>(header + 84, 1536);   // down_q: 32*32/2
  write_le<std::uint64_t>(header + 92, 512);
  write_le<std::uint64_t>(header + 100, 2048);  // down scales: 32*32/32
  write_le<std::uint64_t>(header + 108, 32);
  const auto digest = er::sha256(std::span<const std::byte>(result.bytes).subspan(
      er::kExpertHeaderBytes));
  std::copy(digest.begin(), digest.end(), header + 116);

  result.record.path = "fixture.qpack";
  result.record.record_offset = file_offset;
  result.record.stored_bytes = result.bytes.size();
  result.record.decoded_bytes = 3ULL * 32 * 32 * sizeof(float);
  result.record.header_bytes = er::kExpertHeaderBytes;
  result.record.alignment = er::kExpertPackAlignment;
  result.record.source_abi = er::kExpertSourceAbiExpertPackV1;
  result.record.record_abi = er::kExpertRecordAbiFp4Block32;
  result.record.payload_sha256 = digest;
  return result;
}

FixtureRecord make_fp4_relu2_record(std::uint32_t expert_id,
                                    std::uint32_t layer = 3U) {
  FixtureRecord result;
  result.key = {0x0123456789abcdefULL, layer, expert_id,
                er::kExpertEncodingAbiFp4Block32};
  result.bytes.resize(er::kExpertPackAlignment);
  for (std::size_t index = er::kExpertHeaderBytes; index < result.bytes.size();
       ++index)
    result.bytes[index] =
        static_cast<std::byte>((index * 29U + expert_id) & 0xffU);
  for (const auto& span : {std::pair{768ULL, 32ULL}, {1536ULL, 32ULL}}) {
    for (std::size_t index = span.first; index < span.first + span.second;
         ++index) {
      const auto code = static_cast<unsigned>(result.bytes[index]) % 254U;
      result.bytes[index] = static_cast<std::byte>(code + 1U);
    }
  }
  auto* header = result.bytes.data();
  std::memcpy(header, "EPEXPR01", 8);
  write_le<std::uint16_t>(header + 8, er::kExpertPackVersion);
  write_le<std::uint16_t>(header + 10, er::kExpertHeaderBytes);
  write_le<std::uint32_t>(header + 12, 0x0dU);
  write_le<std::uint32_t>(header + 16,
                          er::kExpertRecordAbiFp4Relu2Block32);
  write_le<std::int32_t>(header + 20, static_cast<std::int32_t>(layer));
  write_le<std::int32_t>(header + 24, static_cast<std::int32_t>(expert_id));
  write_le<std::uint32_t>(header + 28, 32U);
  write_le<std::uint32_t>(header + 32, 32U);
  write_le<std::uint32_t>(header + 36, 32U);
  write_le<std::uint32_t>(header + 40, 0U);
  write_le<std::uint64_t>(header + 44, result.bytes.size());
  write_le<std::uint64_t>(header + 52, 256U);
  write_le<std::uint64_t>(header + 60, 512U);
  write_le<std::uint64_t>(header + 68, 768U);
  write_le<std::uint64_t>(header + 76, 32U);
  write_le<std::uint64_t>(header + 84, 1024U);
  write_le<std::uint64_t>(header + 92, 512U);
  write_le<std::uint64_t>(header + 100, 1536U);
  write_le<std::uint64_t>(header + 108, 32U);
  const auto digest = er::sha256(
      std::span<const std::byte>(result.bytes).subspan(er::kExpertHeaderBytes));
  std::copy(digest.begin(), digest.end(), header + 116);
  result.record.path = "fixture.qpack";
  result.record.stored_bytes = result.bytes.size();
  result.record.decoded_bytes = 2ULL * 32U * 32U * sizeof(float);
  result.record.header_bytes = er::kExpertHeaderBytes;
  result.record.alignment = er::kExpertPackAlignment;
  result.record.source_abi = er::kExpertSourceAbiExpertPackV1;
  result.record.record_abi = er::kExpertRecordAbiFp4Relu2Block32;
  result.record.payload_sha256 = digest;
  return result;
}

void test_fp4_block32_admission_validation() {
  auto fixture = make_fp4_record(7);
  const auto valid = er::validate_expert_admission(
      fixture.bytes, fixture.key, fixture.record);
  require(valid.status.ok() && valid.target.hidden == 32U &&
              valid.target.gate_up_q_bytes == 1024U &&
              valid.target.gate_up_scale_bytes == 64U &&
              valid.target.down_q_bytes == 512U &&
              valid.target.down_scale_bytes == 32U &&
              valid.compact.w1_weight_offset == 256U &&
              valid.compact.w3_weight_offset == 768U &&
              valid.compact.w1_scale_offset == 1280U &&
              valid.compact.w3_scale_offset == 1312U &&
              valid.compact.w2_weight_offset == 1536U &&
              valid.compact.w2_scale_offset == 2048U,
          "valid FP4 block-32 admission was rejected");

  auto corrupt = make_fp4_record(7);
  corrupt.bytes[er::kExpertHeaderBytes + 5] ^= std::byte{1};
  require(!er::validate_expert_admission(corrupt.bytes, corrupt.key,
                                         corrupt.record)
               .status.ok(),
          "corrupt FP4 block-32 admission was accepted");

  for (const auto bad_code : {std::byte{0x00}, std::byte{0xff}}) {
    auto invalid_scale = make_fp4_record(7);
    invalid_scale.bytes[1280] = bad_code;
    const auto digest = er::sha256(
        std::span<const std::byte>(invalid_scale.bytes)
            .subspan(er::kExpertHeaderBytes));
    std::copy(digest.begin(), digest.end(), invalid_scale.bytes.data() + 116);
    invalid_scale.record.payload_sha256 = digest;
    require(!er::validate_expert_admission(invalid_scale.bytes,
                                           invalid_scale.key,
                                           invalid_scale.record)
                 .status.ok(),
            "FP4 block-32 admission accepted an invalid UE8M0 code");
  }

  auto abi_mismatch = make_fp4_record(7);
  abi_mismatch.key.encoding_abi = er::kExpertQuantAbiInt8PerRow;
  require(!er::validate_expert_admission(abi_mismatch.bytes, abi_mismatch.key,
                                         abi_mismatch.record)
               .status.ok(),
          "FP4 record was admitted under the int8 ABI");
}

void test_fp4_relu2_block32_admission_validation() {
  const auto fixture = make_fp4_relu2_record(11U);
  const auto valid = er::validate_expert_admission(
      fixture.bytes, fixture.key, fixture.record);
  require(valid.status.ok() && valid.target.gate_up_q_bytes == 512U &&
              valid.target.gate_up_scale_bytes == 32U &&
              valid.target.down_q_bytes == 512U &&
              valid.compact.w1_weight_offset == 256U &&
              valid.compact.w3_weight_bytes == 0U &&
              valid.compact.w2_weight_offset == 1024U &&
              valid.compact.w2_scale_offset == 1536U,
          "valid FP4 ReLU-squared admission was rejected");
  auto wrong = fixture;
  write_le<std::uint32_t>(wrong.bytes.data() + 12, 0x0fU);
  require(!er::validate_expert_admission(wrong.bytes, wrong.key, wrong.record)
               .status.ok(),
          "FP4 ReLU-squared admission accepted gated-record flags");
}

class ControlledStorage final : public er::IAsyncStorage {
 public:
  struct Pending final {
    er::OperationId id{};
    er::ReadRequest request;
    er::ReadCompletion completion;
  };

  er::OperationId read(er::ReadRequest request,
                       er::ReadCompletion completion) override {
    std::lock_guard lock(mutex_);
    const auto id = next_id_++;
    pending_.push_back({id, std::move(request), std::move(completion)});
    ++read_count_;
    return id;
  }

  void cancel(er::OperationId operation) noexcept override {
    er::ReadCompletion completion;
    std::uint64_t requested = 0;
    {
      std::lock_guard lock(mutex_);
      const auto iterator = std::find_if(
          pending_.begin(), pending_.end(),
          [operation](const Pending& item) { return item.id == operation; });
      if (iterator == pending_.end()) {
        return;
      }
      requested = iterator->request.record.stored_bytes;
      completion = std::move(iterator->completion);
      pending_.erase(iterator);
    }
    completion({er::Status(er::ErrorCode::cancelled, "test read cancelled"),
                requested, 0});
  }

  void complete_success(const std::vector<std::byte>& source) {
    Pending pending = take_one();
    require(source.size() == pending.request.record.stored_bytes,
            "fixture/read size mismatch");
    require(pending.request.destination.capacity >= source.size(),
            "destination too small");
    require(!pending.request.direct ||
                reinterpret_cast<std::uintptr_t>(
                    pending.request.destination.data) %
                        er::kExpertPackAlignment ==
                    0,
            "direct-read destination is not 4096 aligned");
    std::copy(source.begin(), source.end(), pending.request.destination.data);
    pending.completion({er::Status::success(), source.size(), source.size()});
  }

  void complete_short(const std::vector<std::byte>& source) {
    Pending pending = take_one();
    const auto bytes = source.size() / 2;
    std::copy_n(source.begin(), bytes, pending.request.destination.data);
    pending.completion({er::Status::success(), source.size(), bytes});
  }

  [[nodiscard]] std::size_t read_count() const {
    std::lock_guard lock(mutex_);
    return read_count_;
  }

  [[nodiscard]] std::size_t pending_count() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
  }

 private:
  Pending take_one() {
    std::lock_guard lock(mutex_);
    require(!pending_.empty(), "no pending storage operation");
    auto pending = std::move(pending_.front());
    pending_.pop_front();
    return pending;
  }

  mutable std::mutex mutex_;
  std::deque<Pending> pending_;
  er::OperationId next_id_{1};
  std::size_t read_count_{};
};

void test_extent_gather_is_exact_and_bounded() {
  auto backing = std::make_shared<ControlledStorage>();
  er::ExtentGatherStorage gather(backing);
  std::array<std::byte, 12> destination{};
  er::PayloadRecord record;
  record.stored_bytes = destination.size();
  record.extents = {
      {"shard-a", 101U, 0U, 4U},
      {"shard-b", 202U, 4U, 3U},
      {"shard-a", 303U, 7U, 5U},
  };
  std::promise<er::ReadResult> promise;
  auto result = promise.get_future();
  static_cast<void>(gather.read(
      {record, {destination.data(), destination.size()}, true},
      [&promise](er::ReadResult read) { promise.set_value(std::move(read)); }));
  require(backing->pending_count() == 3U,
          "extent gather did not issue one bounded read per source range");
  backing->complete_success(
      {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}});
  backing->complete_success({std::byte{5}, std::byte{6}, std::byte{7}});
  backing->complete_success({std::byte{8}, std::byte{9}, std::byte{10},
                             std::byte{11}, std::byte{12}});
  const auto completed = result.get();
  require(completed.status.ok() && completed.requested_bytes == 12U &&
              completed.read_bytes == 12U &&
              destination[0] == std::byte{1} &&
              destination[6] == std::byte{7} &&
              destination[11] == std::byte{12},
          "extent gather changed source order or byte accounting");

  record.extents[1].destination_offset = 5U;
  bool rejected = false;
  static_cast<void>(gather.read(
      {record, {destination.data(), destination.size()}, true},
      [&rejected](er::ReadResult read) { rejected = !read.status.ok(); }));
  require(rejected && backing->pending_count() == 0U,
          "extent gather accepted a destination gap");
}

class TestDeviceAllocation final : public er::IDeviceAllocation {
 public:
  explicit TestDeviceAllocation(std::size_t size) : size_(size) {}
  [[nodiscard]] std::size_t bytes() const noexcept override { return size_; }

 private:
  std::size_t size_{};
};

class ControlledUploader final : public er::IDeviceUploader {
 public:
  struct Pending final {
    er::OperationId id{};
    er::UploadRequest request;
    er::UploadCompletion completion;
  };

  er::OperationId upload(er::UploadRequest request,
                         er::UploadCompletion completion) override {
    er::OperationId id{};
    {
      std::lock_guard lock(mutex_);
      id = next_id_++;
      pending_.push_back({id, request, std::move(completion)});
      ++upload_count_;
    }
    condition_.notify_all();
    return id;
  }

  void cancel(er::OperationId operation) noexcept override {
    er::UploadCompletion completion;
    {
      std::lock_guard lock(mutex_);
      const auto iterator = std::find_if(
          pending_.begin(), pending_.end(),
          [operation](const Pending& item) { return item.id == operation; });
      if (iterator == pending_.end()) {
        return;
      }
      completion = std::move(iterator->completion);
      pending_.erase(iterator);
    }
    completion({er::Status(er::ErrorCode::cancelled, "test upload cancelled"),
                {}, 0});
  }

  void complete_success(std::size_t device_bytes = 2048) {
    auto pending = take_one();
    pending.completion({er::Status::success(),
                        std::make_shared<TestDeviceAllocation>(device_bytes),
                        device_bytes});
  }

  void complete_failure() {
    auto pending = take_one();
    pending.completion(
        {er::Status(er::ErrorCode::upload_failed, "injected CUDA failure"), {}, 0});
  }

  [[nodiscard]] std::size_t upload_count() const {
    std::lock_guard lock(mutex_);
    return upload_count_;
  }

  [[nodiscard]] std::size_t pending_count() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
  }

  [[nodiscard]] bool wait_for_pending(std::size_t count) const {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, 1s,
                               [&] { return pending_.size() >= count; });
  }

 private:
  Pending take_one() {
    std::unique_lock lock(mutex_);
    require(condition_.wait_for(lock, 1s,
                                [&] { return !pending_.empty(); }),
            "no pending upload operation");
    auto pending = std::move(pending_.front());
    pending_.pop_front();
    return pending;
  }

  mutable std::mutex mutex_;
  mutable std::condition_variable condition_;
  std::deque<Pending> pending_;
  er::OperationId next_id_{1};
  std::size_t upload_count_{};
};

struct Harness final {
  std::shared_ptr<ControlledStorage> storage =
      std::make_shared<ControlledStorage>();
  std::shared_ptr<ControlledUploader> uploader =
      std::make_shared<ControlledUploader>();
  std::shared_ptr<er::FixedBufferPool> buffers;
  er::ExpertCache cache;

  explicit Harness(std::uint64_t budget = 8192, std::size_t slots = 2,
                   std::uint64_t vram_budget = 0,
                   er::CachePlacementConfig placement = {})
      : buffers(std::make_shared<er::FixedBufferPool>(
            slots, er::kExpertPackAlignment, er::kExpertPackAlignment)),
        cache({{budget, budget, std::min<std::uint64_t>(4096, budget)},
               {vram_budget ? vram_budget : budget,
                vram_budget ? vram_budget : budget,
                std::min<std::uint64_t>(4096,
                    vram_budget ? vram_budget : budget)}, true, placement,
               false, 1U, {}},
              storage, uploader, buffers) {}

  er::AcquireResult finish(er::AcquireHandle& handle,
                           const FixtureRecord& fixture) {
    storage->complete_success(fixture.bytes);
    require(handle.wait_for(0ms) == std::future_status::timeout,
            "expert became visible before upload completion");
    uploader->complete_success();
    auto result = handle.get();
    require(result.status.ok() && result.lease, "valid expert did not publish");
    return result;
  }
};

void test_vram_budget_fits_only_before_first_device_admission() {
  Harness harness(8192U, 2U, 8192U);
  require(harness.cache
              .configure_vram_budget({16384U, 16384U, 12288U}, 4096U)
              .ok(),
          "empty expert cache rejected a fitted VRAM budget");
  const auto fixture = make_record(73U, 0U);
  auto handle = harness.cache.acquire(fixture.key, fixture.record);
  auto result = harness.finish(handle, fixture);
  require(result.lease &&
              !harness.cache
                   .configure_vram_budget({32768U, 32768U, 28672U}, 4096U)
                   .ok(),
          "expert cache changed its VRAM contract after admission");
}

void test_expert_store_resolves_complete_ordered_union() {
  Harness harness(16'384, 2, 16'384);
  const auto first = make_record(31, 0);
  const auto second = make_record(32, er::kExpertPackAlignment);
  er::LocalExpertStore store(harness.cache);
  const std::array requests = {
      er::ExpertResolveRequest{second.key, second.record},
      er::ExpertResolveRequest{first.key, first.record},
  };
  auto batch = store.resolve(requests);
  require(batch.valid() && batch.size() == 2U && !batch.poll(),
          "expert store did not retain a pending union");

  harness.storage->complete_success(second.bytes);
  harness.uploader->complete_success();
  require(!batch.poll(), "expert store published a partial union");
  harness.storage->complete_success(first.bytes);
  harness.uploader->complete_success();
  auto resolved = batch.poll();
  require(resolved && resolved->status.ok() &&
              resolved->experts.size() == 2U &&
              resolved->experts[0].key == second.key &&
              resolved->experts[1].key == first.key &&
              resolved->experts[0].placement == er::ExpertPlacementKind::device &&
              resolved->experts[1].placement == er::ExpertPlacementKind::device &&
              resolved->experts[0].device_lease &&
              resolved->experts[1].device_lease,
          "expert store changed order or omitted a union member");
  require(!batch.valid() && !batch.poll(),
          "terminal expert store handle was reusable");

  const std::array duplicate = {
      er::ExpertResolveRequest{first.key, first.record},
      er::ExpertResolveRequest{first.key, first.record},
  };
  require(!store.resolve(duplicate).valid(),
          "expert store accepted duplicate immutable keys");

  const std::array host_request = {
      er::ExpertResolveRequest{second.key, second.record,
                               er::ExpertResolveTarget::host_ready}};
  auto host_batch = store.resolve(host_request);
  auto host_resolved = host_batch.poll();
  require(host_resolved && host_resolved->status.ok() &&
              host_resolved->experts.size() == 1U &&
              host_resolved->experts[0].placement ==
                  er::ExpertPlacementKind::host &&
              host_resolved->experts[0].host_lease &&
              !host_resolved->experts[0].device_lease,
          "expert store did not publish a retained host placement");

  const auto cold = make_record(33, 2U * er::kExpertPackAlignment);
  const std::array cold_host_request = {
      er::ExpertResolveRequest{
          cold.key, cold.record, er::ExpertResolveTarget::host,
          er::ExpertAcquireOptions{er::ExpertRequestPriority::demand, false,
                                   true, false}}};
  const auto uploads_before_host_resolve = harness.uploader->upload_count();
  auto cold_host_batch = store.resolve(cold_host_request);
  require(cold_host_batch.valid() && !cold_host_batch.poll(),
          "cold host resolve did not remain asynchronous");
  harness.storage->complete_success(cold.bytes);
  std::optional<er::ExpertResolveResult> cold_host_resolved;
  const auto host_deadline = std::chrono::steady_clock::now() + 1s;
  while (!cold_host_resolved &&
         std::chrono::steady_clock::now() < host_deadline) {
    cold_host_resolved = cold_host_batch.poll();
    if (!cold_host_resolved) std::this_thread::yield();
  }
  require(cold_host_resolved && cold_host_resolved->status.ok() &&
              cold_host_resolved->experts.size() == 1U &&
              cold_host_resolved->experts[0].key == cold.key &&
              cold_host_resolved->experts[0].placement ==
                  er::ExpertPlacementKind::host &&
              cold_host_resolved->experts[0].host_lease &&
              !cold_host_resolved->experts[0].device_lease &&
              harness.uploader->upload_count() == uploads_before_host_resolve,
          "cold host resolve uploaded weights or lost exact host ownership");

  Harness atomic_harness(4096, 2, 8192);
  er::LocalExpertStore atomic_store(atomic_harness.cache);
  const auto retained = make_record(34, 0U);
  const auto pressure = make_record(35, er::kExpertPackAlignment);
  const std::array retained_request = {
      er::ExpertResolveRequest{
          retained.key, retained.record, er::ExpertResolveTarget::host,
          er::ExpertAcquireOptions{er::ExpertRequestPriority::demand, false,
                                   true, false}}};
  auto retained_batch = atomic_store.resolve(retained_request);
  atomic_harness.storage->complete_success(retained.bytes);
  const auto ready_deadline = std::chrono::steady_clock::now() + 1s;
  for (;;) {
    const auto ready = atomic_harness.cache.inspect(retained.key);
    if (ready && ready->state == er::CacheState::ram_ready &&
        ready->has_host_copy)
      break;
    require(std::chrono::steady_clock::now() < ready_deadline,
            "atomic host lease fixture did not publish RAM residency");
    std::this_thread::yield();
  }
  auto pressure_preload = atomic_harness.cache.preload_host(
      pressure.key, pressure.record,
      {er::ExpertRequestPriority::demand, false});
  require(atomic_harness.storage->pending_count() == 0U,
          "host completion did not atomically retain its RAM ownership");
  auto retained_result = retained_batch.poll();
  require(retained_result && retained_result->status.ok() &&
              retained_result->experts.size() == 1U &&
              retained_result->experts[0].host_lease,
          "host resolve lost ownership before its consumer polled");
  retained_result->experts.clear();
  require(atomic_harness.storage->pending_count() == 1U,
          "released host ownership did not unblock RAM pressure");
  atomic_harness.storage->complete_success(pressure.bytes);
  require(pressure_preload.get().status.ok(),
          "RAM pressure fixture did not complete after lease release");
}

class FixtureRemoteLease final : public er::IRemoteExpertLease {
 public:
  FixtureRemoteLease() {
    identity_.model_content_hash[0] = std::byte{0x77};
    identity_.key = {0x7000U, 0U, 0U, 91U};
    identity_.capability = "moe.fixture.remote.v1";
    identity_.execution_abi = 1U;
    identity_.source_abi = 77U;
  }
  [[nodiscard]] std::string_view owner() const noexcept override {
    return "fixture-node-7";
  }
  [[nodiscard]] const er::ActiveExpertIdentity& identity()
      const noexcept override {
    return identity_;
  }
  [[nodiscard]] er::ActiveExpertExecutionHandle execute(
      er::ActiveExpertInvocation) override {
    return {};
  }

 private:
  er::ActiveExpertIdentity identity_;
};

class FixtureRemoteStore final : public er::IExpertStore {
 public:
  [[nodiscard]] er::ExpertResolveHandle resolve(
      std::span<const er::ExpertResolveRequest> requests) override {
    struct State final {
      std::vector<er::ExpertKey> keys;
      bool terminal{};
    };
    if (requests.empty() ||
        std::any_of(requests.begin(), requests.end(), [](const auto& request) {
          return request.target != er::ExpertResolveTarget::remote &&
                 request.target != er::ExpertResolveTarget::automatic;
        }))
      return {};
    auto state = std::make_shared<State>();
    for (const auto& request : requests) state->keys.push_back(request.key);
    return er::ExpertResolveHandle::from_callbacks(
        [state]() -> std::optional<er::ExpertResolveResult> {
          if (state->terminal) return std::nullopt;
          state->terminal = true;
          er::ExpertResolveResult result;
          result.status = er::Status::success();
          for (const auto& key : state->keys) {
            result.experts.push_back(
                {key, er::ExpertPlacementKind::remote, {}, {},
                 std::make_shared<FixtureRemoteLease>()});
          }
          return result;
        },
        [state] { state->terminal = true; }, requests.size());
  }
};

struct FixtureActiveExpertControl final {
  std::uint32_t executions{};
  std::uint32_t cancellations{};
  bool delay{};
};

class FixtureActiveExpertExecutor final : public er::IActiveExpertExecutor {
 public:
  explicit FixtureActiveExpertExecutor(
      std::shared_ptr<FixtureActiveExpertControl> control)
      : control_(std::move(control)) {}

  [[nodiscard]] std::string_view owner() const noexcept override {
    return "fixture-local-owner";
  }
  [[nodiscard]] bool remote() const noexcept override { return false; }
  [[nodiscard]] er::ActiveExpertExecutorTelemetry telemetry()
      const noexcept override {
    return {};
  }
  [[nodiscard]] er::ActiveExpertExecutionHandle execute(
      er::ActiveExpertExecutionRequest request) override {
    if (!request.invocation.input.valid() ||
        request.invocation.input.bytes != sizeof(std::uint64_t))
      return {};
    ++control_->executions;
    struct State final {
      std::shared_ptr<FixtureActiveExpertControl> control;
      er::ActiveExpertExecutionRequest request;
      std::uint32_t polls{};
      bool terminal{};
    };
    auto state = std::make_shared<State>();
    state->control = control_;
    state->request = std::move(request);
    return er::ActiveExpertExecutionHandle::from_callbacks(
        [state]() -> std::optional<er::ActiveExpertExecutionResult> {
          if (state->terminal) return std::nullopt;
          if (state->control->delay && state->polls++ == 0U)
            return std::nullopt;
          state->terminal = true;
          std::uint64_t input{};
          std::memcpy(&input, state->request.invocation.input.data,
                      sizeof(input));
          auto output = std::make_shared<std::uint64_t>(
              input + state->request.identity.key.expert + 100U);
          er::ActiveExpertExecutionResult result;
          result.status = er::Status::success();
          result.identity = state->request.identity;
          result.request_id = state->request.invocation.request_id;
          result.invocation_id = state->request.invocation.invocation_id;
          result.selection_index =
              state->request.invocation.selection_index;
          result.output = {state->request.invocation.output_abi, "host",
                           output,
                           reinterpret_cast<const std::byte*>(output.get()),
                           sizeof(*output)};
          result.evidence.owner_weight_read_bytes = 13'369'344U;
          result.evidence.owner_vram_read_bytes = 13'369'344U;
          result.evidence.owner_execution_ns = 777U;
          return result;
        },
        [state] {
          if (state->terminal) return;
          state->terminal = true;
          ++state->control->cancellations;
        });
  }

 private:
  std::shared_ptr<FixtureActiveExpertControl> control_;
};

class FixtureActiveExpertTransport final : public er::IActiveExpertTransport {
 public:
  explicit FixtureActiveExpertTransport(
      std::shared_ptr<er::ActiveExpertWireEndpoint> endpoint)
      : endpoint_(std::move(endpoint)) {}

  [[nodiscard]] er::ActiveExpertWireHandle submit(
      std::string_view owner, std::vector<std::byte> frame,
      std::chrono::steady_clock::time_point) override {
    if (owner != "fixture-remote-owner" || !endpoint_) return {};
    ++submissions;
    last_request_frame = frame;
    struct State final {
      FixtureActiveExpertTransport* transport{};
      er::ActiveExpertWireHandle inner;
      bool corrupt{};
      bool terminal{};
    };
    auto state = std::make_shared<State>();
    state->transport = this;
    state->inner = endpoint_->submit(std::move(frame));
    state->corrupt = std::exchange(corrupt_next_response, false);
    return er::ActiveExpertWireHandle::from_callbacks(
        [state]() -> std::optional<er::ActiveExpertWireResult> {
          if (state->terminal) return std::nullopt;
          auto result = state->inner.poll();
          if (!result) return std::nullopt;
          state->terminal = true;
          if (state->corrupt && result->frame.size() > 40U)
            result->frame[40U] ^= std::byte{1};
          return result;
        },
        [state] {
          if (state->terminal) return;
          state->inner.cancel();
          state->terminal = true;
          ++state->transport->cancellations;
        });
  }

  std::uint32_t submissions{};
  std::uint32_t cancellations{};
  bool corrupt_next_response{};
  std::vector<std::byte> last_request_frame;

 private:
  std::shared_ptr<er::ActiveExpertWireEndpoint> endpoint_;
};

er::ActiveExpertExecutionRequest fixture_active_expert_request(
    std::uint64_t request_id, std::uint64_t invocation_id,
    std::chrono::steady_clock::time_point deadline) {
  er::ActiveExpertExecutionRequest request;
  request.identity.model_content_hash[0] = std::byte{0xa7};
  request.identity.key = {0xa700U, 12U, 37U, 2U};
  request.identity.capability = "moe.fixture.fp4.swiglu.v1";
  request.identity.execution_abi = 1U;
  request.identity.source_abi = 2U;
  request.invocation.request_id = request_id;
  request.invocation.invocation_id = invocation_id;
  request.invocation.selection_index = 4U;
  request.invocation.route_width = 6U;
  request.invocation.deadline = deadline;
  auto input = std::make_shared<std::uint64_t>(9U);
  request.invocation.input = {
      "activation.hidden.f32.fixture.v1", "host.pinned", input,
      reinterpret_cast<const std::byte*>(input.get()), sizeof(*input)};
  request.invocation.output_abi = "activation.expert.f32.fixture.v1";
  request.invocation.output_bytes = sizeof(std::uint64_t);
  return request;
}

void test_active_expert_wire_moves_only_exact_activations() {
  auto control = std::make_shared<FixtureActiveExpertControl>();
  auto local = std::make_shared<FixtureActiveExpertExecutor>(control);
  er::ActiveExpertExecutorRegistry registry;
  require(registry
              .add({"fixture-fp4-owner", "moe.fixture.fp4.swiglu.v1", 1U,
                    1U, 10U,
                    [](const er::ActiveExpertIdentity& identity) {
                      return identity.key.encoding_abi == 2U &&
                                     identity.source_abi == 2U
                                 ? er::Status::success()
                                 : er::Status(
                                       er::ErrorCode::invalid_argument,
                                       "fixture owner rejected encoding");
                    },
                    local})
              .ok(),
          "active-expert registry rejected a local ABI provider");
  auto endpoint = std::make_shared<er::ActiveExpertWireEndpoint>(
      std::move(registry));
  auto transport =
      std::make_shared<FixtureActiveExpertTransport>(endpoint);
  auto remote = std::make_shared<er::RemoteActiveExpertExecutor>(
      "fixture-remote-owner", transport);

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::minutes(1);
  auto request = fixture_active_expert_request(71U, 9001U, deadline);
  auto encoded = er::encode_active_expert_request(request);
  require(encoded.status.ok() && encoded.frame.size() < 1024U,
          "activation-only request unexpectedly contains an expert payload");
  const auto decoded = er::decode_active_expert_request(encoded.frame);
  require(decoded.status.ok() &&
              decoded.request.identity == request.identity &&
              decoded.request.invocation.selection_index == 4U &&
              decoded.request.invocation.route_width == 6U &&
              decoded.request.invocation.input.bytes == sizeof(std::uint64_t),
          "active-expert request wire contract lost exact identity or top-k order");
  auto corrupt_request = encoded.frame;
  corrupt_request[40U] ^= std::byte{1};
  require(er::decode_active_expert_request(corrupt_request).status.code() ==
              er::ErrorCode::checksum_mismatch,
          "active-expert wire accepted a corrupt request");

  control->delay = true;
  auto execution = remote->execute(request);
  require(execution.valid() && !execution.poll(),
          "remote active-expert execution did not preserve async polling");
  auto completed = execution.poll();
  require(completed && completed->status.ok() &&
              completed->identity == request.identity &&
              completed->request_id == 71U &&
              completed->invocation_id == 9001U &&
              completed->selection_index == 4U &&
              fixture_scalar_value({completed->output.abi,
                                    completed->output.memory_domain,
                                    completed->output.owner,
                                    completed->output.data,
                                    completed->output.bytes}) == 146U &&
              completed->evidence.activation_input_bytes == 8U &&
              completed->evidence.activation_output_bytes == 8U &&
              completed->evidence.weight_transport_bytes == 0U &&
              completed->evidence.owner_weight_read_bytes == 13'369'344U &&
              completed->evidence.owner_vram_read_bytes == 13'369'344U &&
              completed->evidence.wire_request_bytes ==
                  transport->last_request_frame.size() &&
              completed->evidence.wire_response_bytes > 8U,
          "remote active-expert result lost exact output or traffic evidence");
  const auto client_telemetry = remote->telemetry();
  const auto owner_telemetry = endpoint->telemetry();
  require(client_telemetry.requests == 1U &&
              client_telemetry.completed == 1U &&
              client_telemetry.activation_input_bytes == 8U &&
              client_telemetry.activation_output_bytes == 8U &&
              client_telemetry.weight_transport_bytes == 0U &&
              owner_telemetry.requests == 1U &&
              owner_telemetry.completed == 1U &&
              owner_telemetry.weight_transport_bytes == 0U,
          "active-expert local/remote telemetry is not exact");

  auto cancelled = remote->execute(
      fixture_active_expert_request(72U, 9002U, deadline));
  require(cancelled.valid() && !cancelled.poll(),
          "active-expert cancellation fixture did not become pending");
  cancelled.cancel();
  cancelled.cancel();
  require(control->cancellations == 1U &&
              transport->cancellations == 1U &&
              remote->telemetry().cancelled == 1U &&
              endpoint->telemetry().cancelled == 1U,
          "active-expert cancellation was not propagated exactly once");

  auto expired = remote->execute(fixture_active_expert_request(
      73U, 9003U, std::chrono::steady_clock::now() - 1ms));
  auto expired_result = expired.poll();
  require(expired_result &&
              expired_result->status.code() ==
                  er::ErrorCode::deadline_exceeded &&
              transport->submissions == 2U,
          "expired active-expert work reached the transport");

  control->delay = false;
  transport->corrupt_next_response = true;
  auto corrupt = remote->execute(
      fixture_active_expert_request(74U, 9004U, deadline));
  auto corrupt_result = corrupt.poll();
  require(corrupt_result &&
              corrupt_result->status.code() ==
                  er::ErrorCode::checksum_mismatch &&
              remote->telemetry().checksum_failures == 1U,
          "remote active-expert executor accepted a corrupt response");

  er::ActiveExpertOwnerDirectory local_owners;
  require(local_owners
              .add({0xa700U, 0U, 43U, 0U, 256U, local})
              .ok() &&
              local_owners.find(request.identity.key) == local,
          "active-expert owner directory rejected an in-process executor");

  er::ActiveExpertOwnerDirectory owners;
  require(owners
              .add({0xa700U, 0U, 43U, 0U, 256U, remote})
              .ok() &&
              !owners
                   .add({0xa700U, 12U, 1U, 37U, 1U, remote})
                   .ok(),
          "active-expert owner directory accepted overlapping ownership");
  er::ActiveExpertComponentContract component;
  component.model_content_hash = request.identity.model_content_hash;
  component.namespace_id = 0xa700U;
  component.layer_count = 43U;
  component.experts_per_layer = 256U;
  component.encoding_abi = 2U;
  component.execution_capability = "moe.fixture.fp4.swiglu.v1";
  component.execution_abi = 1U;
  component.source_abi = 2U;
  er::RemoteExpertStore remote_store(component, std::move(owners));
  er::PayloadRecord record;
  record.stored_bytes = 13'369'344U;
  record.source_abi = 2U;
  const std::array resolve_requests{er::ExpertResolveRequest{
      request.identity.key, record, er::ExpertResolveTarget::automatic}};
  auto resolved = remote_store.resolve(resolve_requests);
  auto placement = resolved.poll();
  require(placement && placement->status.ok() &&
              placement->experts.size() == 1U &&
              placement->experts.front().placement ==
                  er::ExpertPlacementKind::remote &&
              placement->experts.front().remote_lease &&
              placement->experts.front().remote_lease->identity() ==
                  request.identity &&
              placement->experts.front().remote_lease->owner() ==
                  "fixture-remote-owner",
          "remote expert store did not bind artifact identity to its owner");
  auto lease_invocation = fixture_active_expert_request(
                              75U, 9005U, deadline)
                              .invocation;
  auto leased = placement->experts.front().remote_lease->execute(
      std::move(lease_invocation));
  auto leased_result = leased.poll();
  require(leased_result && leased_result->status.ok() &&
              leased_result->identity == request.identity &&
              leased_result->evidence.weight_transport_bytes == 0U,
          "remote expert lease did not execute through the activation contract");
}

void test_routed_runtime_accepts_injected_remote_store() {
  Harness harness(16'384, 2, 16'384);
  std::vector<er::PayloadRecord> records(3U);
  for (auto& record : records) {
    record.stored_bytes = 4U;
    record.hidden = 64U;
    record.intermediate = 32U;
    record.source_abi = 77U;
  }
  er::ExpertCatalog catalog;
  require(er::ExpertCatalog::from_records(1U, 3U, std::move(records), catalog)
              .ok(),
          "remote-store fixture catalog is invalid");
  er::RoutedExpertComponentDescriptor component;
  component.name = "decoder";
  component.namespace_id = 0x7000U;
  component.layer_count = 1U;
  component.experts_per_layer = 3U;
  component.route_width = 2U;
  component.hidden_size = 64U;
  component.intermediate_size = 32U;
  component.execution_capability = "moe.fixture.remote.v1";
  component.source_abi = 77U;
  component.encoding_abi = 91U;
  component.encoding = "fp4.fixture.block64";
  component.router = {"router.fixture.topk.v1", 1U,
                      {{"normalize", 1U}}};
  FixtureRemoteStore store;
  er::Sha256Digest content_hash{};
  content_hash[0] = std::byte{0x77};
  er::RoutedExpertRuntime routed(
      component, content_hash, catalog, harness.cache, store);
  auto draft_component = component;
  draft_component.name = "draft";
  draft_component.namespace_id = component.namespace_id + 1U;
  er::RoutedExpertRuntime draft_routed(
      draft_component, content_hash, catalog, harness.cache, store);
  er::ModelDescriptor descriptor;
  descriptor.schema_version = 2U;
  descriptor.architecture_id = "never-seen-before.fp4.sparse";
  descriptor.content_hash = content_hash;
  descriptor.vocab_size = 8192U;
  descriptor.max_context_tokens = 65536U;
  descriptor.hidden_size = 64U;
  descriptor.routed_components.push_back(component);
  descriptor.routed_components.push_back(draft_component);
  descriptor.required_kernels = {
      {"block.fixture.causal.v1", 1U},
      {"router.fixture.topk.v1", 1U},
      {"moe.fixture.remote.v1", 1U}};
  descriptor.layer_program.push_back(
      {0U, "block.fixture.causal.v1", 1U, "decoder", 0U, {}});
  descriptor.operation_program = {
      {0U, 0U, "block.fixture.causal.v1", 1U, "", 0U, {}},
      {1U, 0U, "router.fixture.topk.v1", 1U, "decoder", 0U, {}},
      {2U, 0U, "moe.fixture.remote.v1", 1U, "decoder", 0U, {}},
      {3U, 0U, "moe.fixture.remote.v1", 1U, "draft", 0U, {}}};
  er::ExecutionProviderRegistry registry;
  require(registry
              .add({"fixture-capability-provider", 1U,
                    {er::KernelCapability{"block.fixture.causal.v1", 1U,
                                          1U},
                     er::KernelCapability{"router.fixture.topk.v1", 1U,
                                          1U},
                     er::KernelCapability{"moe.fixture.remote.v1", 1U,
                                          1U}}})
              .ok(),
          "VM fixture provider registration failed");
  auto bound = registry.bind(descriptor);
  require(bound.status.ok(), "VM fixture provider binding failed");
  er::MoeVirtualMachine vm;
  const std::array bindings{
      er::MoeVmComponentBinding{"decoder", &routed},
      er::MoeVmComponentBinding{"draft", &draft_routed}};
  require(er::MoeVirtualMachine::create(
              descriptor, std::move(bound.provider), bindings, vm)
              .ok() &&
              vm.operations(0U).size() == 4U,
          "artifact-driven VM creation lost its operation program");
  constexpr std::array route{2U, 0U};
  require(routed.validate_route(0U, route).ok() &&
              !routed.validate_route(0U, std::span(route).first(1U)).ok(),
          "routed runtime did not enforce artifact top-k cardinality");
  auto resolved =
      vm.resolve_route(0U, route, er::ExpertResolveTarget::remote);
  require(resolved.status.ok(), "VM rejected an exact artifact top-k route");
  auto result = resolved.handle.poll();
  require(result && result->status.ok() && result->experts.size() == 2U &&
              result->experts[0].key.expert == 2U &&
              result->experts[1].key.expert == 0U &&
              result->experts[0].placement ==
                  er::ExpertPlacementKind::remote &&
              result->experts[0].remote_lease &&
              result->experts[0].remote_lease->owner() == "fixture-node-7",
          "injected remote store lost exact route order or ownership");
  auto draft_resolved = vm.resolve_operation_route(
      3U, route, er::ExpertResolveTarget::remote);
  require(draft_resolved.status.ok(),
          "VM failed to resolve the second sparse component on one layer");
  auto draft_result = draft_resolved.handle.poll();
  require(draft_result && draft_result->status.ok() &&
              draft_result->experts.size() == 2U &&
              draft_result->experts[0].key.model_id ==
                  draft_component.namespace_id,
          "operation routing silently fell back to the layer component");
}

class TestMemoryTier final : public er::ITrimmableMemoryTier {
 public:
  TestMemoryTier(std::string_view tier_name, er::MemoryDomain tier_domain,
                 std::uint64_t used, std::uint64_t protected_bytes)
      : name_(tier_name), domain_(tier_domain), used_(used),
        protected_(protected_bytes) {}
  [[nodiscard]] std::string_view name() const noexcept override { return name_; }
  [[nodiscard]] er::MemoryDomain domain() const noexcept override {
    return domain_;
  }
  [[nodiscard]] std::uint64_t used_bytes() const noexcept override {
    return used_;
  }
  [[nodiscard]] std::uint64_t protected_bytes() const noexcept override {
    return protected_;
  }
  [[nodiscard]] std::uint64_t trim_to(std::uint64_t target) override {
    used_ = std::max(protected_, std::min(used_, target));
    ++trims_;
    return used_;
  }
  [[nodiscard]] std::uint64_t trims() const noexcept { return trims_; }

 private:
  std::string name_;
  er::MemoryDomain domain_;
  std::uint64_t used_{};
  std::uint64_t protected_{};
  std::uint64_t trims_{};
};

void test_resource_governor_trims_before_reserving() {
  er::MemoryResourceGovernor governor({1000U, 1000U, 100U, 100U});
  auto transient = std::make_shared<TestMemoryTier>(
      "transient", er::MemoryDomain::device, 400U, 50U);
  auto hot = std::make_shared<TestMemoryTier>(
      "hot", er::MemoryDomain::device, 300U, 200U);
  governor.register_tier({hot, 20U});
  governor.register_tier({transient, 10U});
  require(governor.reserve(er::MemoryDomain::device, 500U).ok(),
          "resource governor rejected a trimmable reservation");
  auto snapshot = governor.snapshot();
  require(snapshot.device_reserved_bytes == 500U &&
              snapshot.device_tier_bytes == 400U &&
              transient->used_bytes() == 100U && hot->used_bytes() == 300U &&
              transient->trims() == 1U && hot->trims() == 0U,
          "resource governor ignored priority or target accounting");
  require(!governor.reserve(er::MemoryDomain::device, 250U).ok(),
          "resource governor overcommitted protected device memory");
  snapshot = governor.snapshot();
  require(snapshot.rejected_reservations == 1U &&
              snapshot.trimmed_bytes == 450U,
          "resource governor trim/rejection telemetry mismatch");
  governor.release(er::MemoryDomain::device, 500U);
  require(governor.snapshot().device_reserved_bytes == 0U,
          "resource governor did not release reservation credits");
}

void test_device_cache_budget_is_a_safe_provider_ceiling() {
  auto fitted = er::fit_device_cache_budget(
      {1000U, 200U, 100U, 100U, 700U, 300U});
  require(fitted.status.ok() && fitted.effective_cache_bytes == 600U,
          "device cache ceiling did not preserve fixed allocations");
  fitted = er::fit_device_cache_budget(
      {1000U, 200U, 100U, 100U, 400U, 300U});
  require(fitted.status.ok() && fitted.effective_cache_bytes == 400U,
          "device cache ceiling expanded a smaller explicit request");
  fitted = er::fit_device_cache_budget(
      {1000U, 400U, 200U, 150U, 500U, 300U});
  require(!fitted.status.ok() && fitted.effective_cache_bytes == 0U,
          "device cache ceiling admitted less than the provider minimum");
  fitted = er::fit_device_cache_budget(
      {1000U, std::numeric_limits<std::uint64_t>::max(), 1U, 1U, 500U,
       300U});
  require(!fitted.status.ok(),
          "device cache ceiling overflowed fixed allocation accounting");
}

void test_state_machine_and_sha256() {
  require(er::cache_state_name(er::CacheState::vram_ready) == "VRAM_READY",
          "cache state name");
  require(er::valid_cache_transition(er::CacheState::absent,
                                     er::CacheState::ssd_loading),
          "valid transition rejected");
  require(!er::valid_cache_transition(er::CacheState::absent,
                                      er::CacheState::vram_ready),
          "invalid publish transition accepted");
  const std::array input = {std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
  const auto digest = er::sha256(input);
  constexpr std::array<std::uint8_t, 32> expected = {
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
      0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
      0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
      0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    require(std::to_integer<std::uint8_t>(digest[index]) == expected[index],
            "SHA-256 implementation mismatch");
  }
  er::Sha256 streaming;
  streaming.update(std::span<const std::byte>(input).first(1));
  streaming.update(std::span<const std::byte>(input).subspan(1, 1));
  streaming.update(std::span<const std::byte>(input).subspan(2));
  require(er::constant_time_equal(digest, streaming.finalize()),
          "streaming SHA-256 differs from one-shot digest");
  require(er::constant_time_equal(digest, streaming.finalize()),
          "streaming SHA-256 finalize is not idempotent");
}

void test_buffer_pool_reserves_demand_capacity_globally() {
  er::FixedBufferPool pool(
      8U, er::kExpertPackAlignment, er::kExpertPackAlignment,
      std::make_shared<er::AlignedHostAllocator>(), 6U);
  std::vector<std::shared_ptr<er::FixedBufferPool::Lease>> background;
  for (std::size_t slot = 0; slot < 2U; ++slot) {
    auto lease = pool.try_acquire(er::kExpertPackAlignment,
                                  er::BufferPoolClass::background);
    require(static_cast<bool>(lease),
            "background staging stopped before its global limit");
    background.push_back(std::move(lease));
  }
  require(!pool.try_acquire(er::kExpertPackAlignment,
                            er::BufferPoolClass::background),
          "background staging consumed a demand-reserved slot");

  std::vector<std::shared_ptr<er::FixedBufferPool::Lease>> demand;
  for (std::size_t slot = 0; slot < 6U; ++slot) {
    auto lease = pool.try_acquire(er::kExpertPackAlignment,
                                  er::BufferPoolClass::demand);
    require(static_cast<bool>(lease),
            "demand could not consume its globally reserved capacity");
    demand.push_back(std::move(lease));
  }
  require(!pool.try_acquire(er::kExpertPackAlignment,
                            er::BufferPoolClass::demand),
          "staging pool overcommitted its physical slot count");
  const auto snapshot = pool.snapshot();
  require(snapshot.slots_in_use == 8U &&
              snapshot.demand_slots_in_use == 6U &&
              snapshot.background_slots_in_use == 2U &&
              snapshot.high_water_slots == 8U &&
              snapshot.background_high_water_slots == 2U &&
              snapshot.demand_acquires == 6U &&
              snapshot.background_acquires == 2U &&
              snapshot.demand_stalls == 1U &&
              snapshot.background_stalls == 1U,
          "global staging class telemetry is incomplete");
}

void test_host_preload_stays_in_ram_and_upgrades_without_reread() {
  Harness harness(8192, 2, 8192, {1, 1, 0, 0, 0, 4096});
  const auto fixture = make_record(62);
  auto preload = harness.cache.preload_host(
      fixture.key, fixture.record,
      {er::ExpertRequestPriority::warm, true});
  require(harness.storage->pending_count() == 1U &&
              harness.uploader->pending_count() == 0U,
          "host-only preload did not start exactly one disk read");
  harness.storage->complete_success(fixture.bytes);
  const auto preloaded = preload.get();
  require(preloaded.status.ok() && preloaded.retained &&
              harness.uploader->upload_count() == 0U,
          "host-only preload reserved or uploaded VRAM");
  const auto host_ready = harness.cache.inspect(fixture.key);
  require(host_ready && host_ready->state == er::CacheState::ram_ready &&
              host_ready->has_host_copy && !host_ready->has_device_copy &&
              host_ready->ram_protected,
          "host-only preload did not publish in protected RAM");

  auto demand = harness.cache.acquire(
      fixture.key, fixture.record,
      {er::ExpertRequestPriority::demand, false, true, false});
  require(harness.storage->read_count() == 1U,
          "device demand reread a host-preloaded expert from disk");
  harness.uploader->complete_success(er::kExpertPackAlignment);
  auto acquired = demand.get();
  require(acquired.status.ok() && acquired.lease,
          "host-preloaded expert did not upgrade to device residency");
  const auto metrics = harness.cache.telemetry();
  constexpr auto warm = static_cast<std::size_t>(
      er::ExpertRequestPriority::warm);
  constexpr auto demand_priority = static_cast<std::size_t>(
      er::ExpertRequestPriority::demand);
  require(metrics.host_preloads_requested == 1U &&
              metrics.host_preloads_completed == 1U &&
              metrics.preloaded_host_useful == 1U &&
              metrics.preloaded_host_useful_bytes ==
                  er::kExpertPackAlignment &&
              metrics.reads_started_by_priority[warm] == 1U &&
              metrics.reads_completed_by_priority[warm] == 1U &&
              metrics.uploads_started_by_priority[demand_priority] == 1U &&
              metrics.uploads_completed_by_priority[demand_priority] == 1U,
          "host preload attribution did not cover disk, RAM, and upload");
}

void test_warm_device_admission_uses_protected_ram_without_reread() {
  Harness harness(8192, 2, 12288, {1, 1, 0, 0, 4096, 4096});
  const auto fixture = make_record(71);
  auto preload = harness.cache.preload_host(
      fixture.key, fixture.record,
      {er::ExpertRequestPriority::warm, true});
  harness.storage->complete_success(fixture.bytes);
  require(preload.get().status.ok(),
          "warm-device fixture did not finish host preload");

  auto warm = harness.cache.acquire(
      fixture.key, fixture.record,
      {er::ExpertRequestPriority::warm, false, true, true});
  require(harness.storage->read_count() == 1U &&
              harness.uploader->pending_count() == 1U,
          "warm device admission reread disk or skipped the RAM upload");
  harness.uploader->complete_success(er::kExpertPackAlignment);
  auto acquired = warm.get();
  require(acquired.status.ok() && acquired.lease,
          "warm device admission did not publish a lease");
  const auto ready = harness.cache.inspect(fixture.key);
  const auto metrics = harness.cache.telemetry();
  constexpr auto warm_priority = static_cast<std::size_t>(
      er::ExpertRequestPriority::warm);
  require(ready && ready->has_host_copy && ready->has_device_copy &&
              ready->ram_protected && ready->vram_resident &&
              metrics.device_requests[warm_priority] == 1U &&
              metrics.ram_hits_by_priority[warm_priority] == 1U &&
              metrics.uploads_started_by_priority[warm_priority] == 1U &&
              metrics.uploads_completed_by_priority[warm_priority] == 1U &&
              metrics.read_bytes == er::kExpertPackAlignment,
          "warm device admission lost its protected placement or attribution");
}

void test_inflight_warm_read_is_upgraded_by_device_demand() {
  Harness harness(8192, 2, 8192, {1, 1, 0, 0, 0, 4096});
  const auto fixture = make_record(63);
  auto preload = harness.cache.preload_host(
      fixture.key, fixture.record,
      {er::ExpertRequestPriority::warm, true});
  auto demand = harness.cache.acquire(
      fixture.key, fixture.record,
      {er::ExpertRequestPriority::demand, false, true, false});
  require(harness.storage->read_count() == 1U,
          "concurrent warm and demand requests issued duplicate reads");
  harness.storage->complete_success(fixture.bytes);
  require(preload.get().status.ok(),
          "joined host waiter did not receive the validated RAM copy");
  harness.uploader->complete_success(er::kExpertPackAlignment);
  require(demand.get().status.ok(),
          "joined demand waiter did not receive the device copy");
  const auto metrics = harness.cache.telemetry();
  constexpr auto demand_priority = static_cast<std::size_t>(
      er::ExpertRequestPriority::demand);
  require(metrics.priority_upgrades == 1U && metrics.load_started == 1U &&
              metrics.uploads_started_by_priority[demand_priority] == 1U &&
              metrics.device_requests[demand_priority] == 1U,
          "in-flight priority upgrade was not attributed to device demand");
}

void test_reload_and_reread_bytes_are_attributed() {
  Harness harness(4096, 1, 4096);
  const auto fixture = make_record(64);
  auto first = harness.cache.acquire(fixture.key, fixture.record);
  auto first_result = harness.finish(first, fixture);
  first_result.lease = {};
  const auto released = harness.cache.trim();
  require(released != 0U &&
              harness.cache.inspect(fixture.key)->state ==
                  er::CacheState::absent,
          "reload fixture did not evict the complete first residency");

  auto second = harness.cache.acquire(fixture.key, fixture.record);
  auto second_result = harness.finish(second, fixture);
  require(second_result.status.ok() && second_result.lease,
          "reloaded expert did not republish");
  const auto metrics = harness.cache.telemetry();
  require(metrics.reload_count == 1U &&
              metrics.reread_bytes == er::kExpertPackAlignment &&
              metrics.read_bytes == 2U * er::kExpertPackAlignment,
          "reload telemetry did not attribute repeated disk traffic");
}

void test_protected_ram_survives_probationary_churn() {
  Harness harness(8192, 2, 12288, {1, 1, 0, 0, 0, 4096});
  const auto protected_record = make_record(65, 0);
  const auto probationary = make_record(66, 4096);
  const auto incoming = make_record(67, 8192);

  auto warm = harness.cache.preload_host(
      protected_record.key, protected_record.record,
      {er::ExpertRequestPriority::warm, true});
  harness.storage->complete_success(protected_record.bytes);
  require(warm.get().status.ok(),
          "protected RAM fixture did not finish host preload");

  auto probationary_handle =
      harness.cache.acquire(probationary.key, probationary.record);
  auto probationary_result =
      harness.finish(probationary_handle, probationary);
  probationary_result.lease = {};
  auto incoming_handle = harness.cache.acquire(incoming.key, incoming.record);
  auto incoming_result = harness.finish(incoming_handle, incoming);
  require(incoming_result.status.ok() && incoming_result.lease,
          "probationary churn fixture did not admit the replacement");

  const auto protected_after = harness.cache.inspect(protected_record.key);
  const auto probationary_after = harness.cache.inspect(probationary.key);
  const auto metrics = harness.cache.telemetry();
  require(protected_after && protected_after->has_host_copy &&
              protected_after->ram_protected && probationary_after &&
              !probationary_after->has_host_copy &&
              metrics.ram_evictions_by_class[0] == 1U &&
              metrics.ram_evictions_by_class[1] == 0U &&
              metrics.ram_evicted_bytes_by_class[0] ==
                  er::kExpertPackAlignment,
          "probationary churn evicted census-protected RAM");
}

void test_reuse_does_not_invade_census_protected_ram() {
  Harness harness(12288, 2, 12288, {1, 1, 0, 0, 0, 4096});
  const auto census = make_record(68, 0);
  const auto first = make_record(69, 4096);
  const auto second = make_record(70, 8192);

  auto warm = harness.cache.preload_host(
      census.key, census.record, {er::ExpertRequestPriority::warm, true});
  harness.storage->complete_success(census.bytes);
  require(warm.get().status.ok(), "census fixture did not preload");

  auto first_handle = harness.cache.acquire(first.key, first.record);
  auto first_result = harness.finish(first_handle, first);
  first_result.lease = {};
  require(harness.cache.record_access(first.key),
          "reuse fixture did not record its second access");
  auto second_handle = harness.cache.acquire(second.key, second.record);
  auto second_result = harness.finish(second_handle, second);
  require(second_result.status.ok() && second_result.lease,
          "second probationary fixture did not publish");

  const auto census_after = harness.cache.inspect(census.key);
  const auto first_after = harness.cache.inspect(first.key);
  const auto second_after = harness.cache.inspect(second.key);
  const auto metrics = harness.cache.telemetry();
  require(census_after && census_after->has_host_copy &&
              census_after->ram_protected && first_after &&
              first_after->has_host_copy && !first_after->ram_protected &&
              second_after && second_after->has_host_copy &&
              !second_after->ram_protected &&
              metrics.ram_protected_bytes == 4096U &&
              metrics.ram_probationary_bytes == 8192U &&
              metrics.ram_promotions == 0U,
          "ordinary reuse invaded the census-protected RAM reservation");
}

void test_expanding_admission_reserves_exact_device_bytes() {
  auto fixture = make_record(61);
  fixture.record.device_bytes = 8192;
  {
    Harness undersized(8192, 2, 4096);
    auto blocked = undersized.cache.acquire(fixture.key, fixture.record);
    require(undersized.storage->pending_count() == 0,
            "expanding admission started I/O without VRAM capacity");
    blocked.cancel();
    require(!blocked.get().status.ok(),
            "cancelled expanding admission unexpectedly succeeded");
  }
  Harness sized(8192, 2, 8192);
  auto admitted = sized.cache.acquire(fixture.key, fixture.record);
  require(sized.storage->pending_count() == 1,
          "valid expanding admission did not start I/O");
  sized.storage->complete_success(fixture.bytes);
  sized.uploader->complete_success(8192);
  auto result = admitted.get();
  require(result.status.ok() && result.lease &&
              result.lease.get()->bytes() == 8192,
          "expanding admission did not publish exact device allocation");
}

void test_ready_first_grouped_scheduler() {
  er::ContinuousBatchScheduler scheduler({2, 4, 8});
  require(scheduler.admit(1).ok() && scheduler.admit(2).ok(),
          "scheduler admission failed");
  require(!scheduler.admit(3).ok(), "scheduler ignored request capacity");
  const std::array first_routes = {
      er::RoutedExpert{7, 0.7F}, er::RoutedExpert{8, 0.2F}};
  const std::array second_routes = {
      er::RoutedExpert{7, 0.6F}, er::RoutedExpert{9, 0.3F}};
  require(scheduler.enqueue({1, 4, 0, 3, 20, first_routes}).ok(),
          "first token enqueue failed");
  require(scheduler.enqueue({2, 9, 1, 3, 10, second_routes}).ok(),
          "second token enqueue failed");
  auto batch = scheduler.schedule([](std::uint32_t, std::uint32_t expert) {
    return expert == 8 ? er::ExpertResidency::absent
                       : er::ExpertResidency::vram_ready;
  });
  require(batch.ready_items == 3 && batch.blocked_items == 1 &&
              batch.groups.size() == 2 && batch.unique_experts == 2,
          "ready-first grouping counts are wrong");
  require(batch.groups[0].expert == 7 && batch.groups[0].items.size() == 2 &&
              batch.groups[0].items[0].request == 2,
          "deadline fairness or expert reuse grouping is wrong");
  std::vector<er::WorkId> completed;
  for (const auto& group : batch.groups)
    for (const auto& item : group.items) completed.push_back(item.work);
  require(scheduler.complete(completed).ok(), "batch completion failed");
  require(!scheduler.token_complete(1, 4) && scheduler.token_complete(2, 9),
          "cold token blocked a ready token or completed too early");
  scheduler.retire_token(2, 9);
  batch = scheduler.schedule([](std::uint32_t, std::uint32_t) {
    return er::ExpertResidency::vram_ready;
  });
  require(batch.ready_items == 1 && batch.groups[0].expert == 8,
          "cold item did not resume after residency changed");
  completed = {batch.groups[0].items[0].work};
  require(scheduler.complete(completed).ok() && scheduler.token_complete(1, 4),
          "resumed token did not complete");
  scheduler.retire_token(1, 4);
  scheduler.finish(1);
  scheduler.cancel(2);
  const auto snapshot = scheduler.snapshot();
  require(snapshot.active_requests == 0 && snapshot.completed_tokens == 2 &&
              snapshot.reused_items == 1 && snapshot.cancelled_requests == 1,
          "scheduler accounting mismatch");
}

void test_concurrent_load_dedup_and_visibility() {
  Harness harness(8192, 2);
  const auto fixture = make_record(7);
  constexpr std::size_t kCallers = 16;
  std::vector<er::AcquireHandle> handles(kCallers);
  std::vector<std::thread> callers;
  callers.reserve(kCallers);
  for (std::size_t index = 0; index < kCallers; ++index) {
    callers.emplace_back([&, index] {
      handles[index] = harness.cache.acquire(fixture.key, fixture.record);
    });
  }
  for (auto& thread : callers) {
    thread.join();
  }
  require(harness.storage->read_count() == 1,
          "concurrent acquire issued a double-load");
  for (auto& handle : handles) {
    require(handle.wait_for(0ms) == std::future_status::timeout,
            "SSD_LOADING entry was visible");
  }

  harness.storage->complete_success(fixture.bytes);
  require(harness.uploader->wait_for_pending(1U) &&
              harness.uploader->upload_count() == 1,
          "deduplicated load issued multiple uploads");
  const auto uploading = harness.cache.inspect(fixture.key);
  require(uploading && uploading->state == er::CacheState::gpu_uploading &&
              !uploading->has_device_copy,
          "entry was published before upload completion");
  for (auto& handle : handles) {
    require(handle.wait_for(0ms) == std::future_status::timeout,
            "GPU_UPLOADING entry was visible");
  }

  harness.uploader->complete_success();
  std::vector<er::AcquireResult> results;
  for (auto& handle : handles) {
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "deduplicated waiter did not receive a lease");
    results.push_back(std::move(result));
  }
  const auto ready = harness.cache.inspect(fixture.key);
  require(ready && ready->state == er::CacheState::vram_ready &&
              ready->reference_count == kCallers,
          "ready leases are not reference-counted");
  const auto metrics = harness.cache.telemetry();
  require(metrics.load_started == 1 &&
              metrics.load_deduplicated == kCallers - 1 &&
              metrics.acquire_ssd_misses == kCallers,
          "load dedup telemetry mismatch");
}

void test_budget_eviction_refcount_and_cancellation() {
  Harness harness(8192, 2);
  const auto first = make_record(1, 0);
  const auto second = make_record(2, 4096);
  const auto third = make_record(3, 8192);
  const auto fourth = make_record(4, 12288);

  auto first_handle = harness.cache.acquire(first.key, first.record);
  auto first_result = harness.finish(first_handle, first);
  auto second_handle = harness.cache.acquire(second.key, second.record);
  auto second_result = harness.finish(second_handle, second);
  auto third_handle = harness.cache.acquire(third.key, third.record);
  require(harness.storage->pending_count() == 0,
          "referenced expert was evicted to admit a third record");

  first_result.lease = {};
  require(harness.storage->pending_count() == 1,
          "releasing a lease did not unblock budget admission");
  auto third_result = harness.finish(third_handle, third);
  const auto first_after = harness.cache.inspect(first.key);
  require(first_after && first_after->state == er::CacheState::vram_ready &&
              !first_after->has_host_copy && first_after->has_device_copy,
          "RAM pressure incorrectly coupled RAM and VRAM eviction");
  require(second_result.lease && third_result.lease,
          "live leases were invalidated by eviction");

  auto fourth_handle = harness.cache.acquire(fourth.key, fourth.record);
  require(harness.storage->pending_count() == 0,
          "budget overcommitted while all entries were referenced");
  fourth_handle.cancel();
  auto cancelled = fourth_handle.get();
  require(cancelled.status.code() == er::ErrorCode::cancelled,
          "pending acquisition did not cancel cleanly");
  const auto metrics = harness.cache.telemetry();
  require(metrics.ram_high_water <= 8192 && metrics.vram_high_water <= 8192 &&
              metrics.ram_bytes <= 8192 && metrics.vram_bytes <= 8192,
          "cache exceeded byte budget under churn");
  require(metrics.stalled_by_budget != 0 && metrics.cancellation_count == 1 &&
              metrics.eviction_count != 0,
          "budget/cancel/eviction telemetry missing");
  require(metrics.eviction_scan_calls != 0 &&
              metrics.eviction_scan_candidates >=
                  metrics.eviction_scan_calls &&
              metrics.task_selection_calls != 0 &&
              metrics.task_selection_candidates != 0 &&
              metrics.mutex_acquisitions != 0,
          "eviction/task-selection/mutex telemetry missing");
}

void test_device_admission_can_fail_fast_without_changing_default_waiting() {
  Harness harness(16'384, 4, 8192);
  const auto first = make_record(70, 0);
  const auto second = make_record(71, 4096);
  const auto incoming = make_record(72, 8192);

  const auto load_full_page = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "device-admission fixture did not publish a full page");
    return result;
  };
  auto first_result = load_full_page(first);
  auto second_result = load_full_page(second);

  auto rejected = harness.cache.acquire(
      incoming.key, incoming.record,
      er::ExpertAcquireOptions{er::ExpertRequestPriority::demand, false, true,
                               false, true});
  require(rejected.wait_for(0ms) == std::future_status::ready,
          "placement-aware admission remained pending without a victim");
  auto rejected_result = rejected.get();
  require(rejected_result.status.code() == er::ErrorCode::backpressure &&
              !rejected_result.lease && harness.storage->pending_count() == 0,
          "fail-fast admission read storage or returned device ownership");

  auto waiting = harness.cache.acquire(incoming.key, incoming.record);
  require(waiting.wait_for(0ms) == std::future_status::timeout &&
              harness.storage->pending_count() == 0,
          "default admission stopped waiting for releasable capacity");
  first_result.lease = {};
  require(harness.storage->pending_count() == 1,
          "default waiter did not resume after device capacity was released");
  auto admitted = harness.finish(waiting, incoming);
  require(admitted.status.ok() && admitted.lease && second_result.lease &&
              harness.cache.telemetry().device_admission_rejections == 1,
          "device admission rejection was not isolated or attributed");
}

void test_short_read_checksum_and_upload_fail_closed() {
  {
    Harness harness(4096, 1);
    const auto fixture = make_record(10);
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_short(fixture.bytes);
    auto result = handle.get();
    require(result.status.code() == er::ErrorCode::short_read && !result.lease,
            "short read was not fail-closed");
    require(harness.uploader->upload_count() == 0,
            "short record reached device uploader");
    require(harness.cache.inspect(fixture.key)->state == er::CacheState::failed,
            "short record was not quarantined");
  }
  {
    Harness harness(4096, 1);
    auto fixture = make_record(11);
    auto corrupted = fixture.bytes;
    corrupted[300] ^= std::byte{0x01};
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(corrupted);
    auto result = handle.get();
    require(result.status.code() == er::ErrorCode::checksum_mismatch &&
                !result.lease,
            "checksum mismatch was not fail-closed");
    require(harness.uploader->upload_count() == 0,
            "corrupt weights reached device uploader");
    require(harness.cache.telemetry().checksum_errors == 1,
            "checksum error telemetry missing");
  }
  {
    Harness harness(4096, 1);
    const auto fixture = make_record(12);
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_failure();
    auto result = handle.get();
    require(result.status.code() == er::ErrorCode::upload_failed && !result.lease,
            "device failure was not fail-closed");
    require(harness.cache.inspect(fixture.key)->state == er::CacheState::failed,
            "failed device copy became visible");
    require(harness.cache.telemetry().upload_errors == 1,
            "upload error telemetry missing");
  }
}

void test_ram_hit_reuploads_after_vram_eviction() {
  Harness harness(8192, 1, 4096);
  const auto first = make_record(20, 0);
  const auto second = make_record(21, 4096);
  auto first_handle = harness.cache.acquire(first.key, first.record);
  auto first_result = harness.finish(first_handle, first);
  first_result.lease = {};

  auto second_handle = harness.cache.acquire(second.key, second.record);
  auto second_result = harness.finish(second_handle, second);
  second_result.lease = {};
  const auto first_in_ram = harness.cache.inspect(first.key);
  require(first_in_ram && first_in_ram->state == er::CacheState::ram_ready &&
              first_in_ram->has_host_copy && !first_in_ram->has_device_copy,
          "VRAM pressure discarded the independent RAM tier");

  const auto reads_before = harness.storage->read_count();
  auto ram_hit = harness.cache.acquire(first.key, first.record);
  require(harness.storage->read_count() == reads_before &&
              harness.uploader->pending_count() == 1,
          "RAM hit reread SSD or failed to schedule H2D");
  harness.uploader->complete_success();
  auto result = ram_hit.get();
  require(result.status.ok() && result.lease,
          "RAM hit did not republish after VRAM reservation");
  auto vram_hit = harness.cache.acquire(first.key, first.record).get();
  require(vram_hit.status.ok() && vram_hit.lease,
          "VRAM hit did not return a lease immediately");
  const auto metrics = harness.cache.telemetry();
  require(metrics.upload_completed == 3 && metrics.acquire_ram_hits == 1 &&
              metrics.acquire_vram_hits == 1 &&
              metrics.acquire_ssd_misses == 2 &&
              metrics.record_validations == 2 &&
              metrics.validated_ram_reuses == 1,
          "RAM reupload telemetry mismatch");
}

void test_host_lease_protects_validated_ram_copy() {
  Harness harness(8192, 1, 4096);
  const auto first = make_record(24, 0);
  const auto second = make_record(25, 4096);
  auto first_handle = harness.cache.acquire(first.key, first.record);
  auto first_result = harness.finish(first_handle, first);
  first_result.lease = {};
  auto second_handle = harness.cache.acquire(second.key, second.record);
  auto second_result = harness.finish(second_handle, second);
  second_result.lease = {};

  auto host = harness.cache.try_acquire_host(first.key, first.record);
  require(host && host->bytes().size() == first.bytes.size() &&
              host->sections().hidden == 16,
          "validated host lease was not exposed from RAM tier");
  static_cast<void>(harness.cache.trim());
  const auto protected_entry = harness.cache.inspect(first.key);
  require(protected_entry && protected_entry->has_host_copy &&
              protected_entry->reference_count == 1,
          "host lease did not protect RAM copy from trim");
  host.reset();
  static_cast<void>(harness.cache.trim());
  const auto released_entry = harness.cache.inspect(first.key);
  require(released_entry && !released_entry->has_host_copy,
          "released host lease remained unevictable");
}

void test_cpu_executor_writes_compact_selection_outputs() {
  auto fixture = make_record(26);
  auto* bytes = fixture.bytes.data();
  std::fill_n(bytes + 256, 256, std::byte{1});
  std::fill_n(bytes + 768, 128, std::byte{1});
  const float scale = 0.01F;
  for (std::size_t offset = 512; offset < 576; offset += sizeof(float))
    std::memcpy(bytes + offset, &scale, sizeof(scale));
  for (std::size_t offset = 1024; offset < 1088; offset += sizeof(float))
    std::memcpy(bytes + offset, &scale, sizeof(scale));
  expert::runtime::cpu::ExpertExecutor executor(2);
  const er::ExpertSections sections{16, 8, 256, 512, 512, 64,
                                    768, 128, 1024, 64};
  const expert::runtime::cpu::ExpertWorkGroup group{
      fixture.bytes, sections, {0, 3}, {1, 0}};
  std::vector<float> inputs(2 * 16, 1.0F);
  std::vector<float> outputs(2 * 16, -123.0F);
  const auto status = executor.execute(std::span(&group, 1), inputs, 2, 2,
                                       outputs);
  require(status.ok(), "CPU expert executor rejected valid fixture");
  const auto executor_metrics = executor.telemetry();
  require(executor_metrics.maximum_threads == 2 &&
              executor_metrics.selected_threads >= 1 &&
              executor_metrics.selected_threads <= 2 &&
              executor_metrics.calibration_runs >= 1 &&
              executor_metrics.execute_calls == 1 &&
              executor_metrics.selections == 2 &&
              executor_metrics.effective_weight_bytes == 768 &&
              executor_metrics.compute_ns > 0,
          "CPU executor calibration or bandwidth telemetry mismatch");
  for (std::size_t column = 0; column < 16; ++column) {
    require(std::isfinite(outputs[column]) &&
                outputs[column] == outputs[16 + column],
            "CPU compact output is invalid or mapping changed the result");
  }
  auto duplicate = group;
  duplicate.output_slots = {0, 0};
  require(!executor.execute(std::span(&duplicate, 1), inputs, 2, 2,
                            outputs).ok(),
          "CPU executor accepted duplicate compact output slots");
}

void test_fp4_host_executor_accepts_prefill_rows() {
  constexpr std::uint32_t hidden = 32U;
  constexpr std::uint32_t intermediate = 32U;
  auto standard = make_fp4_record(7U);
  for (const auto& span : {std::pair{1280ULL, 64ULL},
                           std::pair{2048ULL, 32ULL}}) {
    std::fill_n(standard.bytes.begin() + span.first, span.second,
                std::byte{127});
  }
  const auto digest = er::sha256(
      std::span<const std::byte>(standard.bytes)
          .subspan(er::kExpertHeaderBytes));
  std::copy(digest.begin(), digest.end(), standard.bytes.data() + 116U);
  standard.record.payload_sha256 = digest;
  const auto admitted = er::validate_expert_admission(
      standard.bytes, standard.key, standard.record);
  require(admitted.status.ok(),
          "standard FP4 expert did not expose the universal host ABI");

  er::cpu::Fp4HostExecutor executor({4U, 8U, 8U, 0.0F, false, false});
  std::vector<float> input(hidden, 0.125F);
  const er::cpu::Fp4HostWorkGroup single_group{
      standard.bytes, admitted.compact, hidden, intermediate, {0U}, {0U}};
  std::vector<float> expected(hidden);
  require(executor.execute(std::span(&single_group, 1U), input, 1U, 1U,
                           expected).ok(),
          "standard FP4 host execution failed");

  constexpr std::uint32_t rows = 33U;
  std::vector<float> inputs(rows * hidden);
  std::vector<std::uint32_t> selections(rows);
  std::vector<std::uint32_t> slots(rows);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    std::copy(input.begin(), input.end(),
              inputs.begin() + static_cast<std::size_t>(row) * hidden);
    selections[row] = row;
    slots[row] = row;
  }
  const er::cpu::Fp4HostWorkGroup prefill_group{
      standard.bytes, admitted.compact, hidden, intermediate,
      std::move(selections), std::move(slots)};
  std::vector<float> outputs(rows * hidden);
  require(executor.execute(std::span(&prefill_group, 1U), inputs, rows, 1U,
                           outputs).ok(),
          "universal FP4 host ABI rejected a prefill-sized work group");
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t column = 0U; column < hidden; ++column) {
      const auto actual = outputs[static_cast<std::size_t>(row) * hidden + column];
      require(std::isfinite(expected[column]) && std::isfinite(actual) &&
                  expected[column] == actual,
              "prefill FP4 mismatch at row=" + std::to_string(row) +
                  " column=" + std::to_string(column) +
                  " expected=" + std::to_string(expected[column]) +
                  " actual=" + std::to_string(actual));
    }
  }
}

void test_layer_partitioned_eviction_protects_other_layers() {
  Harness harness(16384, 2, 6144, {2, 1, 0, 0});
  const auto layer0_old = make_record(30, 0, 0);
  const auto layer1_hot = make_record(31, 4096, 1);
  const auto layer0_new = make_record(32, 8192, 0);

  auto first_handle = harness.cache.acquire(layer0_old.key, layer0_old.record);
  auto first = harness.finish(first_handle, layer0_old);
  first.lease = {};
  auto second_handle = harness.cache.acquire(layer1_hot.key, layer1_hot.record);
  auto second = harness.finish(second_handle, layer1_hot);
  second.lease = {};

  auto third_handle = harness.cache.acquire(layer0_new.key, layer0_new.record);
  require(harness.cache.inspect(layer0_old.key)->state ==
              er::CacheState::ram_ready,
          "layer partition did not evict from the requesting layer");
  require(harness.cache.inspect(layer1_hot.key)->state ==
              er::CacheState::vram_ready,
          "layer partition evicted another layer's protected working set");
  auto third = harness.finish(third_handle, layer0_new);
  require(third.status.ok() && third.lease,
          "replacement expert did not become ready");
  require(harness.cache.telemetry().same_partition_evictions == 1,
          "same-partition eviction telemetry mismatch");
}

void test_frequency_admission_protects_reused_expert() {
  Harness harness(24576, 4, 12288, {2, 1, 8192, 0, 4096});
  const auto hot = make_record(40, 0, 0);
  const auto cold_a = make_record(41, 4096, 0);
  const auto incoming = make_record(42, 8192, 0);

  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    auto result = harness.finish(handle, fixture);
    result.lease = {};
  };
  load(hot);
  for (int reuse = 0; reuse < 3; ++reuse) {
    auto hit = harness.cache.acquire(hot.key, hot.record).get();
    require(hit.status.ok() && hit.lease, "hot expert reuse failed");
  }
  load(cold_a);

  auto incoming_handle = harness.cache.acquire(incoming.key, incoming.record);
  require(harness.cache.inspect(hot.key)->state == er::CacheState::vram_ready,
          "one-hit admission evicted the reused expert");
  require(harness.cache.inspect(cold_a.key)->state ==
              er::CacheState::ram_ready,
          "transient admission did not recycle the one-hit slot");
  auto incoming_result = harness.finish(incoming_handle, incoming);
  require(incoming_result.status.ok() && incoming_result.lease,
          "frequency-admitted expert did not publish");
}

void test_routing_score_temperature_breaks_frequency_ties() {
  Harness harness(16384, 4, 8192);
  const auto high_score = make_record(43, 0, 0);
  const auto low_score = make_record(44, 4096, 0);
  const auto incoming = make_record(45, 8192, 0);
  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "score-temperature fixture load failed");
    result.lease = {};
  };
  load(high_score);
  load(low_score);
  const std::array accesses{
      er::ExpertAccess{high_score.key, 1, 0.9, 0.9},
      er::ExpertAccess{low_score.key, 1, 0.1, 0.1},
  };
  require(harness.cache.record_accesses(accesses) == 2,
          "scored route feedback was not recorded");
  const auto high_snapshot = harness.cache.inspect(high_score.key);
  const auto low_snapshot = harness.cache.inspect(low_score.key);
  require(high_snapshot && low_snapshot &&
              high_snapshot->frequency == low_snapshot->frequency &&
              high_snapshot->routing_score_mass_q20 >
                  low_snapshot->routing_score_mass_q20 &&
              high_snapshot->placement_temperature >
                  low_snapshot->placement_temperature,
          "routing score did not affect bounded cache temperature");

  auto incoming_handle = harness.cache.acquire(incoming.key, incoming.record);
  const auto high_after = harness.cache.inspect(high_score.key);
  const auto low_after = harness.cache.inspect(low_score.key);
  if (!high_after || !low_after ||
      high_after->state != er::CacheState::vram_ready ||
      low_after->state != er::CacheState::ram_ready) {
    throw std::runtime_error(
        "score-aware eviction state mismatch: high=" +
        std::to_string(high_after ? static_cast<int>(high_after->state) : -1) +
        ", low=" +
        std::to_string(low_after ? static_cast<int>(low_after->state) : -1));
  }
  auto incoming_result = harness.finish(incoming_handle, incoming);
  require(incoming_result.status.ok() && incoming_result.lease,
          "score-aware replacement did not publish");
}

void test_vram_replacement_requires_a_strictly_colder_victim() {
  Harness harness(16384, 4, 8192);
  const auto candidate = make_record(50, 0, 0);
  const auto resident_a = make_record(51, 4096, 0);
  const auto resident_b = make_record(52, 8192, 0);
  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "placement fixture failed to load expert");
    result.lease = {};
  };
  load(candidate);
  load(resident_a);
  require(harness.cache.record_access(resident_a.key, 4),
          "fixture did not protect the intended resident");
  load(resident_b);
  require(harness.cache.inspect(candidate.key)->state ==
              er::CacheState::ram_ready,
          "fixture did not create a RAM promotion candidate");
  const std::array candidate_accesses{er::ExpertAccess{candidate.key, 8}};
  require(harness.cache.record_accesses(candidate_accesses) == 8,
          "batched route feedback did not update candidate frequency");

  auto lease_a = harness.cache.acquire(resident_a.key, resident_a.record).get();
  auto lease_b = harness.cache.acquire(resident_b.key, resident_b.record).get();
  require(lease_a.lease && lease_b.lease,
          "resident protection leases were not acquired");
  require(!harness.cache.vram_admission_would_improve(candidate.key,
                                                       candidate.record),
          "admission selected an in-flight VRAM victim");
  lease_a.lease = {};
  require(harness.cache.vram_admission_would_improve(candidate.key,
                                                      candidate.record),
          "hot RAM candidate did not outrank a colder VRAM resident");
  const auto admission_metrics = harness.cache.telemetry();
  require(admission_metrics.vram_admission_scan_calls == 2U &&
              admission_metrics.vram_admission_scan_candidates != 0U &&
              admission_metrics.mutex_acquisitions != 0U,
          "VRAM admission scan/mutex telemetry missing");
  er::AdaptivePlacementConfig placement_config;
  placement_config.enable_prefetch = true;
  placement_config.minimum_recent_observations = 1;
  er::AdaptivePlacementPlanner planner(harness.cache, placement_config);
  planner.consider(candidate.key, candidate.record, 1);
  require(planner.telemetry().scheduled == 1 &&
              harness.uploader->pending_count() == 1,
          "admitted promotion did not enter the asynchronous uploader");
  const std::array routed_candidate{candidate.key};
  planner.observe_routes(std::span<const er::ExpertKey>{}, routed_candidate);
  harness.uploader->complete_success(er::kExpertPackAlignment);
  planner.poll();
  require(planner.telemetry().completed == 1 &&
              planner.telemetry().useful_prefetches == 1 &&
              planner.telemetry().useful_prefetch_bytes ==
                  er::kExpertPackAlignment &&
              harness.cache.inspect(candidate.key)->state ==
                  er::CacheState::vram_ready,
          "useful asynchronous prefetch was not attributed or published");
  require(planner.quiesce(10ms).ok() && planner.frozen(),
          "placement epoch did not quiesce and freeze");
  planner.resume();
  require(!planner.frozen(), "placement epoch did not resume");
}

void test_vram_stale_resident_bytes_tracks_victim_recency() {
  Harness harness(16384, 4, 8192);
  const auto stale = make_record(60, 0, 0);
  const auto fresh = make_record(61, 4096, 0);
  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "staleness fixture failed to load expert");
    result.lease = {};
  };
  load(stale);
  load(fresh);
  require(harness.cache.vram_stale_resident_bytes(10) == 0,
          "freshly loaded residents counted as stale");
  require(harness.cache.record_access(fresh.key, 100),
          "staleness fixture did not advance the access clock");
  require(harness.cache.vram_stale_resident_bytes(50) ==
              er::kExpertPackAlignment,
          "long-unrouted resident did not count toward stale bytes");
  require(harness.cache.vram_stale_resident_bytes(1000) == 0,
          "resident counted as stale beyond its actual age");
  auto lease = harness.cache.acquire(fresh.key, fresh.record).get();
  require(static_cast<bool>(lease.lease),
          "fresh resident protection lease was not acquired");
  require(harness.cache.vram_stale_resident_bytes(50) ==
              er::kExpertPackAlignment,
          "referenced fresh resident changed the stale total");
}

void test_prefetch_credits_and_stale_epoch_cancel_pending_work() {
  Harness harness(16384, 4, 8192);
  const auto candidate = make_record(53, 0, 0);
  const auto resident_a = make_record(54, 4096, 0);
  const auto resident_b = make_record(55, 8192, 0);
  const auto rejected = make_record(56, 12288, 0);
  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "prefetch cancellation fixture failed to load");
    result.lease = {};
  };
  load(candidate);
  load(resident_a);
  load(resident_b);
  const std::array hot_accesses{er::ExpertAccess{candidate.key, 8, 4.0, 0.8}};
  require(harness.cache.record_accesses(hot_accesses) == 8,
          "prefetch candidate did not become hotter than residents");

  er::AdaptivePlacementConfig config;
  config.enable_prefetch = true;
  config.maximum_candidates = 1;
  config.minimum_recent_observations = 1;
  config.candidate_ttl_epochs = 1;
  er::AdaptivePlacementPlanner planner(harness.cache, config);
  planner.consider(candidate.key, candidate.record, 1, 0.8);
  require(harness.uploader->pending_count() == 1,
          "bounded prefetch did not consume its one in-flight credit");
  planner.consider(rejected.key, rejected.record, 1, 0.9);
  require(planner.telemetry().credit_rejections == 1,
          "busy candidate bound did not reject additional prefetch history");

  planner.observe_routes(std::span<const er::ExpertKey>{},
                         std::span<const er::ExpertKey>{});
  planner.observe_routes(std::span<const er::ExpertKey>{},
                         std::span<const er::ExpertKey>{});
  const auto telemetry = planner.telemetry();
  require(telemetry.stale_cancellations == 1 &&
              telemetry.wasted_prefetches == 1 &&
              telemetry.wasted_prefetch_bytes == er::kExpertPackAlignment &&
              telemetry.failed == 0 && harness.uploader->pending_count() == 0,
          "stale pending prefetch was not cancelled and attributed exactly");
}

void test_hybrid_dispatch_minimizes_measured_critical_path() {
  er::HybridDispatchPlanner planner({100.0, 10.0, 1.0e9, 0.5, 16, 16});
  const std::array candidates{
      er::HybridDispatchCandidate{3, 10, 0, true, true, true},
      er::HybridDispatchCandidate{1, 5, 10, false, true, true},
      er::HybridDispatchCandidate{2, 5, 10'000, false, true, true},
      er::HybridDispatchCandidate{4, 2, 0, false, true, false},
      er::HybridDispatchCandidate{5, 2, 100, false, false, true},
  };
  const auto plan = planner.plan(candidates);
  require(plan.status.ok() && plan.decisions.size() == candidates.size(),
          "hybrid dispatch rejected a feasible layer");
  const auto decision = [&](std::uint32_t expert) -> const auto& {
    const auto found = std::find_if(
        plan.decisions.begin(), plan.decisions.end(),
        [&](const auto& value) { return value.expert == expert; });
    require(found != plan.decisions.end(), "hybrid decision is missing");
    return *found;
  };
  require(decision(3).executor == er::HybridExecutor::gpu_resident &&
              decision(3).reason == er::HybridDispatchReason::resident_gpu,
          "resident expert did not retain GPU priority");
  require(decision(1).executor == er::HybridExecutor::gpu_upload &&
              decision(1).reason ==
                  er::HybridDispatchReason::gpu_lower_critical_path,
          "small upload did not shorten the projected critical path");
  require(decision(2).executor == er::HybridExecutor::cpu_local &&
              decision(2).reason ==
                  er::HybridDispatchReason::cpu_lower_critical_path,
          "large upload was not kept on CPU");
  require(decision(4).reason == er::HybridDispatchReason::cpu_only &&
              decision(5).reason == er::HybridDispatchReason::gpu_only,
          "forced executor reason was not preserved");

  planner.observe_cpu(1'000, 20);
  planner.observe_gpu(400, 20);
  planner.observe_h2d(1'000, 2'000);
  const auto telemetry = planner.telemetry();
  require(telemetry.cpu_ns_per_selection == 75.0 &&
              telemetry.gpu_ns_per_selection == 15.0 &&
              telemetry.h2d_bytes_per_second == 1.5e9 &&
              telemetry.plans == 1 && telemetry.candidates == 5,
          "hybrid EWMA or decision telemetry mismatch");
}

void test_hybrid_dispatch_ties_bounds_and_trace_are_deterministic() {
  er::HybridDispatchPlanner planner({100.0, 10.0, 1.0e9, 0.5, 2, 2});
  const std::array tie{
      er::HybridDispatchCandidate{7, 1, 90, false, true, true}};
  const auto tied = planner.plan(tie);
  require(tied.status.ok() &&
              tied.decisions.front().executor ==
                  er::HybridExecutor::cpu_local &&
              tied.decisions.front().reason ==
                  er::HybridDispatchReason::cpu_stable_tie,
          "hybrid tie did not use the stable CPU fallback");
  const std::array two{
      er::HybridDispatchCandidate{8, 1, 0, true, false, false},
      er::HybridDispatchCandidate{9, 1, 0, true, false, false}};
  require(planner.plan(two).status.ok(), "bounded hybrid plan failed");
  const auto trace = planner.trace();
  require(trace.size() == 2 && trace[0].expert == 8 && trace[1].expert == 9,
          "bounded trace did not retain the newest decisions in order");
  const std::array duplicate{
      er::HybridDispatchCandidate{1, 1, 0, true, false, false},
      er::HybridDispatchCandidate{1, 1, 0, true, false, false}};
  require(!planner.plan(duplicate).status.ok(),
          "hybrid planner accepted duplicate experts");
  const std::array too_many{
      er::HybridDispatchCandidate{1, 1, 0, true, false, false},
      er::HybridDispatchCandidate{2, 1, 0, true, false, false},
      er::HybridDispatchCandidate{3, 1, 0, true, false, false}};
  require(!planner.plan(too_many).status.ok() &&
              planner.telemetry().rejected_plans == 2,
          "hybrid planner did not enforce its candidate bound");
}

void test_hybrid_dispatch_requires_live_tiers_and_warms_hot_gpu_pages() {
  er::HybridDispatchPlanner planner(
      {100.0, 10.0, 1.0e9, 0.5, 8U, 16U, true, true, true, 1U});
  const std::array candidates{
      er::HybridDispatchCandidate{2U, 1U, 90U, false, true, true, 10U, 2U},
      er::HybridDispatchCandidate{1U, 1U, 90U, false, true, true, 50U, 5U}};
  const auto cold = planner.plan(candidates);
  require(cold.status.ok() &&
              std::all_of(cold.decisions.begin(), cold.decisions.end(),
                          [](const auto& item) {
                            return item.executor == er::HybridExecutor::gpu_upload;
                          }),
          "unmeasured H2D path was allowed to speculate on CPU");

  planner.observe_h2d(90U, 90U);
  const auto calibrated = planner.plan(candidates);
  const auto hot = std::find_if(calibrated.decisions.begin(),
                                calibrated.decisions.end(),
                                [](const auto& item) {
                                  return item.expert == 1U;
                                });
  const auto probe = std::find_if(calibrated.decisions.begin(),
                                  calibrated.decisions.end(),
                                  [](const auto& item) {
                                    return item.reason ==
                                           er::HybridDispatchReason::cpu_calibration;
                                  });
  require(hot != calibrated.decisions.end() &&
              hot->reason == er::HybridDispatchReason::gpu_cache_warm &&
              probe != calibrated.decisions.end(),
          "hybrid bootstrap did not warm the hottest page and calibrate CPU");
  planner.observe_cpu(200U, 1U);
  planner.observe_gpu(20U, 1U);
  const auto telemetry = planner.telemetry();
  require(telemetry.h2d_observations == 1U &&
              telemetry.cpu_observations == 1U &&
              telemetry.gpu_observations == 1U &&
              telemetry.cpu_calibrations == 1U &&
              telemetry.gpu_cache_warms == 2U,
          "hybrid live observation telemetry is incomplete");
}

void test_monotonic_host_allocator_is_bounded_and_aligned() {
  auto arena = std::make_shared<er::MonotonicHostAllocator>(
      4096U, 256U, std::make_shared<er::AlignedHostAllocator>());
  auto* first = arena->allocate(100U, 64U);
  auto* second = arena->allocate(200U, 256U);
  require(first && second &&
              reinterpret_cast<std::uintptr_t>(first) % 64U == 0U &&
              reinterpret_cast<std::uintptr_t>(second) % 256U == 0U &&
              arena->bytes_used() == 456U && !arena->page_locked(),
          "monotonic host allocator violated bank geometry");
  arena->deallocate(first);
  require(arena->bytes_used() == 456U &&
              arena->allocate(4096U, 256U) == nullptr,
          "monotonic host allocator reclaimed or exceeded its fixed bank");
}

void test_route_census_is_bounded_ranked_and_recoverable() {
  const std::array identity_bytes{std::byte{1}, std::byte{7}, std::byte{9}};
  er::RouteCensusConfig config{17U, er::sha256(identity_bytes), 3U,
                               2U, 8U, 3U, 2U};
  er::RouteCensus census(config);
  const std::array first{1U, 2U, 3U};
  const std::array first_cpu{3U};
  require(census.observe(0U, first, first_cpu).ok(),
          "route census rejected a valid first route");
  const auto root = std::filesystem::temp_directory_path() /
      ("quantum-route-census-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto prefix = root / "routes";
  require(census.save(prefix).ok(),
          "route census failed to publish generation one");
  require(std::filesystem::file_size(
              std::filesystem::path(prefix.string() + ".1")) ==
              census.snapshot().serialized_bytes,
          "route census serialized-size accounting changed");

  const std::array second{4U, 5U, 6U};
  const std::array third{1U, 2U, 4U};
  const std::array third_cpu{4U};
  require(census.observe(1U, second).ok() &&
              census.observe(0U, third, third_cpu).ok(),
          "route census rejected a valid continuation");
  const std::array duplicate{1U, 1U, 2U};
  require(!census.observe(0U, duplicate).ok(),
          "route census accepted a duplicate route");
  const auto prediction = census.predict_next(0U, first, 2U);
  require(prediction.size() == 1U && prediction[0].key.expert == 4U &&
              prediction[0].transition_score == 3U,
          "route census transition prediction is not stable");
  const auto warm = census.stable_warm_set(3U, 2U);
  require(warm.size() == 3U && warm[0].key.layer == 0U &&
              (warm[0].key.expert == 1U || warm[0].key.expert == 2U) &&
              std::count_if(warm.begin(), warm.end(), [](const auto& value) {
                return value.key.layer == 0U;
              }) <= 2,
          "route census warm ranking or per-layer bound is unstable");
  const auto before_save = census.snapshot();
  require(before_save.completed_routes == 3U &&
              before_save.total_selections == 9U &&
              before_save.consecutive_reuse_selections == 2U &&
              before_save.observed_experts == 7U &&
              census.save(prefix).ok(),
          "route census accounting or generation two save failed");

  auto loaded = er::RouteCensus::load(prefix, config);
  require(loaded.status.ok() && loaded.census &&
              !loaded.namespace_rebound &&
              loaded.census->snapshot().generation == 2U &&
              loaded.census->snapshot().completed_routes == 3U &&
              loaded.census->predict_next(0U, first, 2U).size() == 1U,
          "route census did not load its newest valid generation");
  auto relocated = config;
  relocated.model_id = 0x570116270568999ULL;
  auto rebound = er::RouteCensus::load(prefix, relocated);
  const auto rebound_warm = rebound.census
      ? rebound.census->stable_warm_set(1U, 1U)
      : std::vector<er::RouteCensusWarmEntry>{};
  const auto relocated_prefix = root / "relocated-routes";
  require(rebound.status.ok() && rebound.census &&
              rebound.namespace_rebound && rebound_warm.size() == 1U &&
              rebound_warm.front().key.model_id == relocated.model_id &&
              rebound.census->save(relocated_prefix).ok(),
          "route census did not safely rebind a runtime namespace");
  auto rebound_persisted =
      er::RouteCensus::load(relocated_prefix, relocated);
  require(rebound_persisted.status.ok() && rebound_persisted.census &&
              !rebound_persisted.namespace_rebound &&
              rebound_persisted.census->snapshot().generation == 3U,
          "route census namespace rebind was not persisted");
  auto mismatch = config;
  mismatch.model_content_hash[0] ^= std::byte{1};
  require(!er::RouteCensus::load(prefix, mismatch).status.ok(),
          "route census accepted a different model identity");

  const auto newest = std::filesystem::path(prefix.string() + ".0");
  std::fstream corrupt(newest, std::ios::binary | std::ios::in |
                                   std::ios::out);
  require(static_cast<bool>(corrupt),
          "route census corruption fixture could not open newest slot");
  corrupt.seekp(16);
  const char changed = static_cast<char>(0xa5);
  corrupt.write(&changed, 1);
  corrupt.close();
  auto recovered = er::RouteCensus::load(prefix, config);
  require(recovered.status.ok() && recovered.census &&
              recovered.census->snapshot().generation == 1U &&
              recovered.census->snapshot().completed_routes == 1U,
          "route census did not recover the previous authenticated slot");
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

void test_placement_profile_uses_measurements_and_exact_budgets() {
  const er::PlacementProfileInput input{
      {8'000'000.0, 700'000.0, 10.0e9, 6U, 6U, 64U << 20U},
      {64ULL << 30U, 8ULL << 30U, 4ULL << 30U, 16ULL << 20U, 6U, 1024U},
      {24ULL << 30U, 8ULL << 30U, 1ULL << 30U, 16ULL << 20U, 7U, 512U},
      0.25, 64U, 32U};
  const auto plan = er::solve_placement_profile(input);
  require(plan.status.ok() && plan.host_expert_slots == 1024U &&
              plan.host_cache_bytes == (16ULL << 30U) &&
              plan.device_expert_slots == 512U &&
              plan.device_cache_bytes == (8ULL << 30U) &&
              plan.governor.host_budget_bytes == (64ULL << 30U) &&
              plan.governor.device_budget_bytes == (24ULL << 30U) &&
              plan.dispatch.initial_cpu_ns_per_selection == 8'000'000.0 &&
              plan.dispatch.initial_gpu_ns_per_selection == 700'000.0 &&
              plan.dispatch.initial_h2d_bytes_per_second == 10.0e9,
          "placement profile did not preserve measurements and hard caps");

  auto insufficient = input;
  insufficient.device.minimum_expert_slots = 513U;
  require(!er::solve_placement_profile(insufficient).status.ok(),
          "placement profile silently reduced a required device minimum");
  auto unmeasured = input;
  unmeasured.costs.h2d_sample_bytes = 0U;
  require(!er::solve_placement_profile(unmeasured).status.ok(),
          "placement profile accepted an unmeasured transfer cost");
  auto overcommitted = input;
  overcommitted.host.fixed_bytes = 61ULL << 30U;
  require(!er::solve_placement_profile(overcommitted).status.ok(),
          "placement profile accepted fixed host overcommit");
}

}  // namespace

int main() {
  try {
    test_exact_rejection_sampling_matches_independent_target_oracle();
    test_universal_model_descriptor_negotiates_capabilities();
    test_serialized_model_program_is_provider_neutral();
    test_schema_v2_artifact_binds_unknown_model_without_architecture_branch();
    test_schema_v3_callable_program_is_exact_and_family_neutral();
    test_schema_v2_expresses_model_derived_hybrid_moe_topology();
    test_generic_expert_catalog_uses_descriptor_cardinality();
    test_universal_worker_launch_preserves_provider_extensions();
    test_vram_budget_fits_only_before_first_device_admission();
    test_fp4_block32_admission_validation();
    test_fp4_relu2_block32_admission_validation();
    test_extent_gather_is_exact_and_bounded();
    test_state_machine_and_sha256();
    test_buffer_pool_reserves_demand_capacity_globally();
    test_host_preload_stays_in_ram_and_upgrades_without_reread();
    test_warm_device_admission_uses_protected_ram_without_reread();
    test_inflight_warm_read_is_upgraded_by_device_demand();
    test_reload_and_reread_bytes_are_attributed();
    test_protected_ram_survives_probationary_churn();
    test_reuse_does_not_invade_census_protected_ram();
    test_expanding_admission_reserves_exact_device_bytes();
    test_expert_store_resolves_complete_ordered_union();
    test_active_expert_wire_moves_only_exact_activations();
    test_routed_runtime_accepts_injected_remote_store();
    test_resource_governor_trims_before_reserving();
    test_device_cache_budget_is_a_safe_provider_ceiling();
    test_ready_first_grouped_scheduler();
    test_concurrent_load_dedup_and_visibility();
    test_budget_eviction_refcount_and_cancellation();
    test_device_admission_can_fail_fast_without_changing_default_waiting();
    test_short_read_checksum_and_upload_fail_closed();
    test_ram_hit_reuploads_after_vram_eviction();
    test_host_lease_protects_validated_ram_copy();
    test_cpu_executor_writes_compact_selection_outputs();
    test_fp4_host_executor_accepts_prefill_rows();
    test_layer_partitioned_eviction_protects_other_layers();
    test_frequency_admission_protects_reused_expert();
    test_routing_score_temperature_breaks_frequency_ties();
    test_vram_replacement_requires_a_strictly_colder_victim();
    test_vram_stale_resident_bytes_tracks_victim_recency();
    test_prefetch_credits_and_stale_epoch_cancel_pending_work();
    test_hybrid_dispatch_minimizes_measured_critical_path();
    test_hybrid_dispatch_ties_bounds_and_trace_are_deterministic();
    test_hybrid_dispatch_requires_live_tiers_and_warms_hot_gpu_pages();
    test_monotonic_host_allocator_is_bounded_and_aligned();
    test_route_census_is_bounded_ranked_and_recoverable();
    test_placement_profile_uses_measurements_and_exact_budgets();
    std::cout << "expert_runtime_tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert_runtime_tests: FAIL: " << error.what() << '\n';
    return 1;
  }
}
