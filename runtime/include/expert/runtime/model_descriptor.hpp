#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace expert::runtime {

inline constexpr std::uint32_t kModelLevelOperationLayer =
    std::numeric_limits<std::uint32_t>::max();

struct ModelDescriptor;

// Provider-owned routing program for one sparse component. The common runtime
// deliberately does not enumerate router algorithms: linear, hash, grouped,
// biased, and future routers are negotiated by capability/ABI and immutable
// numeric parameters from the artifact.
struct RouterProgramDescriptor final {
  std::string capability;
  std::uint32_t abi_version{1U};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
};

// One independently placeable routed-expert namespace. A model may expose
// main, draft, vision, or other components without assigning them globally
// meaningful numeric IDs in code.
struct RoutedExpertComponentDescriptor final {
  std::string name;
  std::uint64_t namespace_id{};
  std::uint32_t layer_count{};
  std::uint32_t experts_per_layer{};
  std::uint32_t route_width{};
  std::uint32_t shared_experts_per_layer{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::string execution_capability;
  std::uint32_t execution_abi{1U};
  std::uint32_t source_abi{};
  std::uint32_t encoding_abi{};
  // Opaque identifiers owned by the artifact adapter/provider. Supporting a
  // new numeric format must not extend an enum in the common runtime.
  std::string encoding;
  std::map<std::string, std::uint64_t, std::less<>> attributes;
  RouterProgramDescriptor router;
};

// A provider-defined operation ABI required by the compiled model program.
// The common runtime treats capability names as opaque and only performs exact
// version negotiation. Adding a provider does not extend a central model enum.
struct KernelRequirement final {
  std::string capability;
  std::uint32_t abi_version{1U};
};

struct KernelCapability final {
  std::string capability;
  std::uint32_t minimum_abi{1U};
  std::uint32_t maximum_abi{1U};
  // Provider-owned startup validation for geometry, dtype, device, or any
  // other constraint that cannot be expressed by ABI range alone.
  std::function<Status(const ModelDescriptor&)> validate;

  KernelCapability(
      std::string capability_value, std::uint32_t minimum_abi_value = 1U,
      std::uint32_t maximum_abi_value = 1U,
      std::function<Status(const ModelDescriptor&)> validator = {})
      : capability(std::move(capability_value)),
        minimum_abi(minimum_abi_value), maximum_abi(maximum_abi_value),
        validate(std::move(validator)) {}
};

// Immutable startup program for one transformer layer. Parameters are
// provider-owned and are compiled into native bindings once at model load;
// strings and maps never enter the per-token hot path.
struct LayerProgramDescriptor final {
  std::uint32_t logical_layer{};
  std::string block_capability;
  std::uint32_t block_abi_version{1U};
  std::string routed_component;
  // Explicit catalog-local layer. It need not equal logical_layer, allowing
  // dense prefixes, interleaved MoE, and independently indexed components.
  std::uint32_t component_layer{};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
};

// Ordered provider-neutral VM instruction. Capabilities are opaque operation
// names, not model-family enums. A routed instruction binds to a component and
// its catalog-local layer; a model-level instruction leaves the component
// empty and component_layer zero.
struct OperationProgramDescriptor final {
  std::uint32_t logical_operation{};
  std::uint32_t logical_layer{};
  std::string capability;
  std::uint32_t abi_version{1U};
  std::string routed_component;
  std::uint32_t component_layer{};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
  // Provider-role -> immutable tensor name in the dense tensor index.
  std::map<std::string, std::string, std::less<>> tensor_bindings;
  // Provider port -> artifact-declared SSA value and opaque value ABI.
  // Schema v3 makes runtime data flow explicit instead of relying on a
  // family-specific hidden-state convention in a worker.
  struct ValueBinding final {
    std::string value;
    std::string abi;
  };
  std::map<std::string, ValueBinding, std::less<>> input_bindings;
  std::map<std::string, ValueBinding, std::less<>> output_bindings;

  OperationProgramDescriptor() = default;
  OperationProgramDescriptor(
      std::uint32_t logical_operation_value,
      std::uint32_t logical_layer_value, std::string capability_value,
      std::uint32_t abi_version_value, std::string routed_component_value,
      std::uint32_t component_layer_value,
      std::map<std::string, std::uint64_t, std::less<>> parameters_value = {},
      std::map<std::string, std::string, std::less<>> tensor_bindings_value =
          {},
      std::map<std::string, ValueBinding, std::less<>> input_bindings_value =
          {},
      std::map<std::string, ValueBinding, std::less<>> output_bindings_value =
          {})
      : logical_operation(logical_operation_value),
        logical_layer(logical_layer_value),
        capability(std::move(capability_value)),
        abi_version(abi_version_value),
        routed_component(std::move(routed_component_value)),
        component_layer(component_layer_value),
        parameters(std::move(parameters_value)),
        tensor_bindings(std::move(tensor_bindings_value)),
        input_bindings(std::move(input_bindings_value)),
        output_bindings(std::move(output_bindings_value)) {}
};

struct ProgramEndpointDescriptor final {
  std::string value;
  std::string abi;
};

// Optional exact token-generation service attached to the scalar model
// program. The capability remains provider-owned; the common runner only
// enforces the artifact-declared upper bound and exact token-stream contract.
// This is not an operation in the scalar SSA graph because it may execute and
// transactionally verify more than one causal position against the same
// request state.
struct ExactDecodeProgramDescriptor final {
  std::string capability;
  std::uint32_t abi_version{1U};
  std::uint32_t maximum_emitted_tokens{};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
  // Provider-role -> immutable tensor name. Exact decode is outside the SSA
  // graph, but it follows the same artifact-owned tensor binding contract as
  // ordinary operations.
  std::map<std::string, std::string, std::less<>> tensor_bindings;
};

struct ModelDescriptor final {
  std::uint32_t schema_version{1U};
  std::string architecture_id;
  Sha256Digest content_hash{};
  std::uint32_t vocab_size{};
  std::uint32_t max_context_tokens{};
  std::uint32_t hidden_size{};
  std::map<std::string, std::uint64_t, std::less<>> attributes;
  // Model-level endpoints such as token embedding, final norm, and output
  // head. Role names are provider-owned and architecture names stay opaque.
  std::map<std::string, std::string, std::less<>> tensor_bindings;
  std::vector<RoutedExpertComponentDescriptor> routed_components;
  std::vector<KernelRequirement> required_kernels;
  std::vector<LayerProgramDescriptor> layer_program;
  // Schema v2 requires an explicit ordered operation program and router
  // program for every routed component. Schema v3 additionally requires an
  // explicit SSA graph from program inputs to program outputs, including
  // model-level operations. Schema v1 remains readable and is compiled into
  // its legacy one-block-per-layer program.
  std::vector<OperationProgramDescriptor> operation_program;
  // External role -> internal SSA value. Roles and ABIs are artifact-owned;
  // the common interpreter never assigns meaning to either string.
  std::map<std::string, ProgramEndpointDescriptor, std::less<>> program_inputs;
  std::map<std::string, ProgramEndpointDescriptor, std::less<>> program_outputs;
  std::optional<ExactDecodeProgramDescriptor> exact_decode_program;
};

// Provider indices are resolved once at startup. The per-token path consumes
// this numeric program and never branches on architecture names or capability
// strings.
struct CompiledKernelBinding final {
  std::uint32_t requirement_index{};
  // Registry/provider ownership is resolved by ExecutionProviderRegistry.
  // compile_model_program() leaves this at zero for callers compiling against
  // one capability table directly.
  std::uint32_t provider_registry_index{};
  std::uint32_t provider_capability_index{};
};

struct CompiledLayerProgram final {
  std::uint32_t logical_layer{};
  std::uint32_t block_kernel_binding{};
  std::optional<std::uint32_t> routed_component_index;
  std::uint32_t component_layer{};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
};

struct CompiledOperationProgram final {
  std::uint32_t logical_operation{};
  std::uint32_t logical_layer{};
  std::uint32_t kernel_binding{};
  std::optional<std::uint32_t> routed_component_index;
  std::uint32_t component_layer{};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
  std::map<std::string, std::string, std::less<>> tensor_bindings;
  struct ValueBinding final {
    std::string port;
    std::uint32_t value_index{};
  };
  std::vector<ValueBinding> input_values;
  std::vector<ValueBinding> output_values;
};

struct CompiledProgramValue final {
  std::string name;
  std::string abi;
};

struct CompiledProgramEndpoint final {
  std::string role;
  std::uint32_t value_index{};
};

struct CompiledExactDecodeProgram final {
  std::uint32_t kernel_binding{};
  std::uint32_t maximum_emitted_tokens{};
  std::map<std::string, std::uint64_t, std::less<>> parameters;
  std::map<std::string, std::string, std::less<>> tensor_bindings;
};

struct CompiledModelProgram final {
  std::vector<CompiledKernelBinding> kernels;
  std::vector<CompiledLayerProgram> layers;
  std::vector<CompiledOperationProgram> operations;
  std::vector<CompiledProgramValue> values;
  std::vector<CompiledProgramEndpoint> inputs;
  std::vector<CompiledProgramEndpoint> outputs;
  std::optional<CompiledExactDecodeProgram> exact_decode;
};

struct CompileModelProgramResult final {
  Status status;
  CompiledModelProgram program;
};

struct ParseModelDescriptorResult final {
  Status status;
  ModelDescriptor descriptor;
};

[[nodiscard]] Status validate_model_descriptor(
    const ModelDescriptor& descriptor) noexcept;
[[nodiscard]] Status validate_routed_component(
    const RoutedExpertComponentDescriptor& component,
    std::uint32_t model_hidden_size) noexcept;
[[nodiscard]] Status provider_supports_model(
    const ModelDescriptor& descriptor,
    std::span<const KernelCapability> capabilities) noexcept;
[[nodiscard]] CompileModelProgramResult compile_model_program(
    const ModelDescriptor& descriptor,
    std::span<const KernelCapability> capabilities) noexcept;
// Parses the provider-neutral, line-oriented artifact program. Component
// namespace offsets are bound to a runtime-selected namespace base so the
// immutable artifact never embeds process-local cache identities.
[[nodiscard]] ParseModelDescriptorResult parse_model_descriptor_artifact(
    std::string_view text, const Sha256Digest& content_hash,
    std::uint64_t namespace_base) noexcept;
[[nodiscard]] ParseModelDescriptorResult load_model_descriptor_artifact(
    const std::filesystem::path& path, std::uint64_t expected_bytes,
    const Sha256Digest& expected_sha256, const Sha256Digest& content_hash,
    std::uint64_t namespace_base) noexcept;
[[nodiscard]] std::uint64_t expert_table_entries(
    const RoutedExpertComponentDescriptor& component) noexcept;
[[nodiscard]] const RoutedExpertComponentDescriptor* find_routed_component(
    const ModelDescriptor& descriptor, std::string_view name) noexcept;
[[nodiscard]] std::uint64_t namespace_id_from_content_hash(
    const Sha256Digest& digest) noexcept;

}  // namespace expert::runtime
