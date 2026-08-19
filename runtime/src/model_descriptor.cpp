#include "expert/runtime/model_descriptor.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace expert::runtime {
namespace {

Status invalid(std::string message) {
  return {ErrorCode::invalid_argument, std::move(message)};
}

std::vector<std::string_view> split_fields(std::string_view line) {
  std::vector<std::string_view> fields;
  while (true) {
    const auto delimiter = line.find('\t');
    fields.push_back(line.substr(0U, delimiter));
    if (delimiter == std::string_view::npos) break;
    line.remove_prefix(delimiter + 1U);
  }
  return fields;
}

bool parse_u64(std::string_view text, std::uint64_t& value) noexcept {
  if (text.empty()) return false;
  const auto* first = text.data();
  const auto* last = first + text.size();
  const auto parsed = std::from_chars(first, last, value);
  return parsed.ec == std::errc{} && parsed.ptr == last;
}

bool parse_u32(std::string_view text, std::uint32_t& value) noexcept {
  std::uint64_t wide{};
  if (!parse_u64(text, wide) ||
      wide > std::numeric_limits<std::uint32_t>::max())
    return false;
  value = static_cast<std::uint32_t>(wide);
  return true;
}

bool valid_atom(std::string_view text) noexcept {
  return !text.empty() && text.find_first_of("\t\r\n") == std::string_view::npos;
}

LayerProgramDescriptor* find_layer(ModelDescriptor& descriptor,
                                   std::uint32_t logical_layer) noexcept {
  const auto found = std::find_if(
      descriptor.layer_program.begin(), descriptor.layer_program.end(),
      [logical_layer](const auto& item) {
        return item.logical_layer == logical_layer;
      });
  return found == descriptor.layer_program.end() ? nullptr : &*found;
}

OperationProgramDescriptor* find_operation(
    ModelDescriptor& descriptor, std::uint32_t logical_operation) noexcept {
  const auto found = std::find_if(
      descriptor.operation_program.begin(), descriptor.operation_program.end(),
      [logical_operation](const auto& item) {
        return item.logical_operation == logical_operation;
      });
  return found == descriptor.operation_program.end() ? nullptr : &*found;
}

RoutedExpertComponentDescriptor* find_component(
    ModelDescriptor& descriptor, std::string_view name) noexcept {
  const auto found = std::find_if(
      descriptor.routed_components.begin(), descriptor.routed_components.end(),
      [name](const auto& item) { return item.name == name; });
  return found == descriptor.routed_components.end() ? nullptr : &*found;
}

}  // namespace

std::uint64_t expert_table_entries(
    const RoutedExpertComponentDescriptor& component) noexcept {
  return static_cast<std::uint64_t>(component.layer_count) *
         component.experts_per_layer;
}

const RoutedExpertComponentDescriptor* find_routed_component(
    const ModelDescriptor& descriptor, std::string_view name) noexcept {
  const auto found = std::find_if(
      descriptor.routed_components.begin(), descriptor.routed_components.end(),
      [name](const auto& item) { return item.name == name; });
  return found == descriptor.routed_components.end() ? nullptr : &*found;
}

std::uint64_t namespace_id_from_content_hash(
    const Sha256Digest& digest) noexcept {
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < sizeof(value); ++index) {
    value |= static_cast<std::uint64_t>(
                 std::to_integer<std::uint8_t>(digest[index]))
             << (index * 8U);
  }
  return value == 0U ? 1U : value;
}

Status validate_model_descriptor(const ModelDescriptor& descriptor) noexcept {
  if ((descriptor.schema_version != 1U && descriptor.schema_version != 2U &&
       descriptor.schema_version != 3U) ||
      descriptor.architecture_id.empty() ||
      descriptor.vocab_size == 0U || descriptor.max_context_tokens == 0U ||
      descriptor.hidden_size == 0U || descriptor.layer_program.empty() ||
      descriptor.required_kernels.empty())
    return invalid("model descriptor architecture is incomplete");

  for (const auto& [role, tensor] : descriptor.tensor_bindings) {
    if (!valid_atom(role) || !valid_atom(tensor))
      return invalid("model tensor binding is invalid");
  }

  std::set<std::string, std::less<>> component_names;
  std::set<std::uint64_t> namespaces;
  for (const auto& item : descriptor.routed_components) {
    const auto component_status =
        validate_routed_component(item, descriptor.hidden_size);
    if (!component_status.ok()) return component_status;
    if (!component_names.insert(item.name).second ||
        !namespaces.insert(item.namespace_id).second)
      return invalid("routed expert component geometry or identity is invalid");
  }

  std::set<std::pair<std::string, std::uint32_t>> requirements;
  for (const auto& item : descriptor.required_kernels) {
    if (item.capability.empty() || item.abi_version == 0U ||
        !requirements.emplace(item.capability, item.abi_version).second)
      return invalid("kernel requirement is invalid or duplicated");
  }
  if (descriptor.exact_decode_program) {
    const auto& decode = *descriptor.exact_decode_program;
    if (descriptor.schema_version < 3U || !valid_atom(decode.capability) ||
        decode.abi_version == 0U || decode.maximum_emitted_tokens < 2U ||
        !requirements.contains({decode.capability, decode.abi_version}))
      return invalid("exact decode program is invalid or unsupported");
    for (const auto& [key, value] : decode.parameters) {
      static_cast<void>(value);
      if (!valid_atom(key))
        return invalid("exact decode program parameter is invalid");
    }
    for (const auto& [role, tensor] : decode.tensor_bindings) {
      if (!valid_atom(role) || !valid_atom(tensor))
        return invalid("exact decode tensor binding is invalid");
    }
  }
  for (const auto& item : descriptor.routed_components) {
    if (!requirements.contains(
            {item.execution_capability, item.execution_abi}))
      return invalid(
          "routed component lacks its execution kernel requirement");
    const bool has_router = !item.router.capability.empty();
    if ((descriptor.schema_version >= 2U && !has_router) ||
        (has_router &&
         (item.router.abi_version == 0U ||
          !requirements.contains(
              {item.router.capability, item.router.abi_version}))))
      return invalid(
          "routed component lacks its routing kernel requirement");
  }

  std::set<std::uint32_t> logical_layers;
  for (const auto& layer : descriptor.layer_program) {
    if (layer.block_capability.empty() || layer.block_abi_version == 0U ||
        layer.logical_layer != logical_layers.size() ||
        !logical_layers.insert(layer.logical_layer).second ||
        !requirements.contains(
            {layer.block_capability, layer.block_abi_version}))
      return invalid("layer program is invalid or lacks a kernel requirement");
    if (!layer.routed_component.empty()) {
      const auto* component =
          find_routed_component(descriptor, layer.routed_component);
      if (component == nullptr ||
          layer.component_layer >= component->layer_count)
        return invalid("layer program references an unavailable expert component");
    } else if (layer.component_layer != 0U) {
      return invalid("dense layer program has a routed component index");
    }
  }
  if (*logical_layers.begin() != 0U ||
      *logical_layers.rbegin() + 1ULL != logical_layers.size())
    return invalid("layer program logical indices are not contiguous");

  if (descriptor.schema_version >= 2U &&
      descriptor.operation_program.empty())
    return invalid("schema v2 model descriptor lacks its operation program");
  if (!descriptor.operation_program.empty()) {
    std::set<std::uint32_t> logical_operations;
    std::set<std::uint32_t> operation_layers;
    std::uint32_t previous_layer{};
    bool first_operation = true;
    for (const auto& operation : descriptor.operation_program) {
      const bool model_level =
          operation.logical_layer == kModelLevelOperationLayer;
      if (operation.capability.empty() || operation.abi_version == 0U ||
          operation.logical_operation != logical_operations.size() ||
          !logical_operations.insert(operation.logical_operation).second ||
          (!model_level && !logical_layers.contains(operation.logical_layer)) ||
          (model_level && descriptor.schema_version < 3U) ||
          (!model_level && !first_operation &&
           operation.logical_layer < previous_layer) ||
          !requirements.contains(
              {operation.capability, operation.abi_version}))
        return invalid("operation program is invalid or unsupported");
      for (const auto& [role, tensor] : operation.tensor_bindings) {
        if (!valid_atom(role) || !valid_atom(tensor))
          return invalid("operation tensor binding is invalid");
      }
      for (const auto& [port, binding] : operation.input_bindings) {
        if (!valid_atom(port) || !valid_atom(binding.value) ||
            !valid_atom(binding.abi))
          return invalid("operation input binding is invalid");
      }
      for (const auto& [port, binding] : operation.output_bindings) {
        if (!valid_atom(port) || !valid_atom(binding.value) ||
            !valid_atom(binding.abi))
          return invalid("operation output binding is invalid");
      }
      if (!model_level) {
        previous_layer = operation.logical_layer;
        first_operation = false;
        operation_layers.insert(operation.logical_layer);
      }
      if (!operation.routed_component.empty()) {
        if (model_level)
          return invalid("model-level operation references an expert component");
        const auto* component =
            find_routed_component(descriptor, operation.routed_component);
        if (component == nullptr ||
            operation.component_layer >= component->layer_count)
          return invalid(
              "operation program references an unavailable expert component");
      } else if (operation.component_layer != 0U) {
        return invalid("model-level operation has a component-local index");
      }
    }
    if (*logical_operations.begin() != 0U ||
        *logical_operations.rbegin() + 1ULL != logical_operations.size() ||
        operation_layers != logical_layers)
      return invalid("operation program indices or layer coverage are invalid");

    if (descriptor.schema_version >= 2U) {
      for (const auto& layer : descriptor.layer_program) {
        if (layer.routed_component.empty()) continue;
        const auto* component =
            find_routed_component(descriptor, layer.routed_component);
        const auto contains = [&](std::string_view capability,
                                  std::uint32_t abi) {
          return std::any_of(
              descriptor.operation_program.begin(),
              descriptor.operation_program.end(), [&](const auto& operation) {
                return operation.logical_layer == layer.logical_layer &&
                       operation.routed_component == layer.routed_component &&
                       operation.component_layer == layer.component_layer &&
                       operation.capability == capability &&
                       operation.abi_version == abi;
              });
        };
        if (component == nullptr ||
            !contains(component->router.capability,
                      component->router.abi_version) ||
            !contains(component->execution_capability,
                      component->execution_abi))
          return invalid(
              "routed layer lacks explicit router or expert operations");
      }
    }
  }

  if (descriptor.schema_version < 3U) {
    if (!descriptor.program_inputs.empty() ||
        !descriptor.program_outputs.empty() ||
        descriptor.exact_decode_program ||
        std::any_of(descriptor.operation_program.begin(),
                    descriptor.operation_program.end(), [](const auto& item) {
                      return !item.input_bindings.empty() ||
                             !item.output_bindings.empty();
                    }))
      return invalid("legacy model descriptor contains schema v3 data flow");
    return Status::success();
  }

  if (descriptor.program_inputs.empty() || descriptor.program_outputs.empty())
    return invalid("schema v3 model descriptor lacks program endpoints");
  std::map<std::string, std::string, std::less<>> available_values;
  for (const auto& [role, endpoint] : descriptor.program_inputs) {
    if (!valid_atom(role) || !valid_atom(endpoint.value) ||
        !valid_atom(endpoint.abi) ||
        !available_values.emplace(endpoint.value, endpoint.abi).second)
      return invalid("schema v3 program input is invalid or duplicated");
  }
  for (const auto& operation : descriptor.operation_program) {
    if (operation.input_bindings.empty() &&
        operation.output_bindings.empty())
      return invalid("schema v3 operation lacks explicit data flow");
    for (const auto& [port, binding] : operation.input_bindings) {
      static_cast<void>(port);
      const auto value = available_values.find(binding.value);
      if (value == available_values.end() || value->second != binding.abi)
        return invalid(
            "schema v3 operation input is unavailable or has the wrong ABI");
    }
    for (const auto& [port, binding] : operation.output_bindings) {
      static_cast<void>(port);
      if (!available_values.emplace(binding.value, binding.abi).second)
        return invalid("schema v3 operation output has multiple producers");
    }
  }
  for (const auto& [role, endpoint] : descriptor.program_outputs) {
    const auto value = available_values.find(endpoint.value);
    if (!valid_atom(role) || !valid_atom(endpoint.value) ||
        !valid_atom(endpoint.abi) || value == available_values.end() ||
        value->second != endpoint.abi)
      return invalid("schema v3 program output is unavailable or invalid");
  }
  return Status::success();
}

Status validate_routed_component(
    const RoutedExpertComponentDescriptor& item,
    std::uint32_t model_hidden_size) noexcept {
  if (item.name.empty() || item.namespace_id == 0U || item.layer_count == 0U ||
      item.experts_per_layer == 0U || item.route_width == 0U ||
      item.route_width > item.experts_per_layer || item.hidden_size == 0U ||
      item.intermediate_size == 0U || item.source_abi == 0U ||
      item.encoding_abi == 0U || item.execution_capability.empty() ||
      item.execution_abi == 0U || item.encoding.empty() ||
      item.hidden_size != model_hidden_size)
    return invalid("routed expert component geometry or identity is invalid");
  if (expert_table_entries(item) >
      std::numeric_limits<std::size_t>::max() / 128U)
    return invalid("routed expert table exceeds addressable metadata");
  return Status::success();
}

Status provider_supports_model(
    const ModelDescriptor& descriptor,
    std::span<const KernelCapability> capabilities) noexcept {
  try {
    auto valid = validate_model_descriptor(descriptor);
    if (!valid.ok()) return valid;
    for (const auto& required : descriptor.required_kernels) {
      const auto found = std::find_if(
          capabilities.begin(), capabilities.end(),
          [&](const auto& available) {
            return available.capability == required.capability &&
                   required.abi_version >= available.minimum_abi &&
                   required.abi_version <= available.maximum_abi;
          });
      if (found == capabilities.end())
        return invalid(
            "execution provider lacks required kernel capability: " +
            required.capability);
      if (found->validate) {
        auto constrained = found->validate(descriptor);
        if (!constrained.ok()) return constrained;
      }
    }
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("execution provider validation failed: ") +
                error.what()};
  }
  return Status::success();
}

CompileModelProgramResult compile_model_program(
    const ModelDescriptor& descriptor,
    std::span<const KernelCapability> capabilities) noexcept {
  const auto supported = provider_supports_model(descriptor, capabilities);
  if (!supported.ok())
    return {{supported.code(), std::string(supported.message())}, {}};
  try {
    CompiledModelProgram program;
    program.kernels.reserve(descriptor.required_kernels.size());
    std::map<std::pair<std::string, std::uint32_t>, std::uint32_t>
        kernel_bindings;
    for (std::size_t requirement_index = 0U;
         requirement_index < descriptor.required_kernels.size();
         ++requirement_index) {
      const auto& required = descriptor.required_kernels[requirement_index];
      const auto available = std::find_if(
          capabilities.begin(), capabilities.end(), [&](const auto& item) {
            return item.capability == required.capability &&
                   required.abi_version >= item.minimum_abi &&
                   required.abi_version <= item.maximum_abi;
          });
      const auto binding = static_cast<std::uint32_t>(program.kernels.size());
      program.kernels.push_back(
          {static_cast<std::uint32_t>(requirement_index),
           0U,
           static_cast<std::uint32_t>(available - capabilities.begin())});
      kernel_bindings.emplace(
          std::pair{required.capability, required.abi_version}, binding);
    }
    std::map<std::string, std::uint32_t, std::less<>> component_indices;
    for (std::size_t index = 0U; index < descriptor.routed_components.size();
         ++index)
      component_indices.emplace(descriptor.routed_components[index].name,
                                static_cast<std::uint32_t>(index));
    program.layers.reserve(descriptor.layer_program.size());
    for (const auto& layer : descriptor.layer_program) {
      std::optional<std::uint32_t> component;
      if (!layer.routed_component.empty())
        component = component_indices.at(layer.routed_component);
      program.layers.push_back(
          {layer.logical_layer,
           kernel_bindings.at(
               {layer.block_capability, layer.block_abi_version}),
           component, layer.component_layer, layer.parameters});
    }
    if (descriptor.operation_program.empty()) {
      program.operations.reserve(descriptor.layer_program.size());
      for (const auto& layer : descriptor.layer_program) {
        std::optional<std::uint32_t> component;
        if (!layer.routed_component.empty())
          component = component_indices.at(layer.routed_component);
        program.operations.push_back(
            {static_cast<std::uint32_t>(program.operations.size()),
             layer.logical_layer,
             kernel_bindings.at(
                 {layer.block_capability, layer.block_abi_version}),
             component, layer.component_layer, layer.parameters, {}, {}, {}});
      }
    } else {
      program.operations.reserve(descriptor.operation_program.size());
      for (const auto& operation : descriptor.operation_program) {
        std::optional<std::uint32_t> component;
        if (!operation.routed_component.empty())
          component = component_indices.at(operation.routed_component);
        program.operations.push_back(
            {operation.logical_operation, operation.logical_layer,
             kernel_bindings.at(
                 {operation.capability, operation.abi_version}),
             component, operation.component_layer, operation.parameters,
             operation.tensor_bindings, {}, {}});
      }
    }
    if (descriptor.schema_version >= 3U) {
      std::map<std::string, std::uint32_t, std::less<>> value_indices;
      const auto add_value = [&](const std::string& name,
                                 const std::string& abi) {
        const auto index = static_cast<std::uint32_t>(program.values.size());
        program.values.push_back({name, abi});
        value_indices.emplace(name, index);
        return index;
      };
      program.inputs.reserve(descriptor.program_inputs.size());
      for (const auto& [role, endpoint] : descriptor.program_inputs)
        program.inputs.push_back(
            {role, add_value(endpoint.value, endpoint.abi)});
      for (std::size_t index = 0U;
           index < descriptor.operation_program.size(); ++index) {
        const auto& source = descriptor.operation_program[index];
        auto& compiled_operation = program.operations[index];
        compiled_operation.input_values.reserve(source.input_bindings.size());
        for (const auto& [port, binding] : source.input_bindings)
          compiled_operation.input_values.push_back(
              {port, value_indices.at(binding.value)});
        compiled_operation.output_values.reserve(
            source.output_bindings.size());
        for (const auto& [port, binding] : source.output_bindings)
          compiled_operation.output_values.push_back(
              {port, add_value(binding.value, binding.abi)});
      }
      program.outputs.reserve(descriptor.program_outputs.size());
      for (const auto& [role, endpoint] : descriptor.program_outputs)
        program.outputs.push_back({role, value_indices.at(endpoint.value)});
    }
    if (descriptor.exact_decode_program) {
      const auto& decode = *descriptor.exact_decode_program;
      program.exact_decode = CompiledExactDecodeProgram{
          kernel_bindings.at({decode.capability, decode.abi_version}),
          decode.maximum_emitted_tokens, decode.parameters,
          decode.tensor_bindings};
    }
    return {Status::success(), std::move(program)};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("model program compilation failed: ") + error.what()},
            {}};
  }
}

ParseModelDescriptorResult parse_model_descriptor_artifact(
    std::string_view text, const Sha256Digest& content_hash,
    std::uint64_t namespace_base) noexcept {
  try {
    if (namespace_base == 0U)
      return {invalid("model descriptor namespace base is zero"), {}};
    ModelDescriptor descriptor;
    descriptor.content_hash = content_hash;
    bool header = false;
    bool model = false;
    std::size_t line_number = 0U;
    while (!text.empty()) {
      ++line_number;
      const auto end = text.find('\n');
      auto line = text.substr(0U, end);
      if (!line.empty() && line.back() == '\r') line.remove_suffix(1U);
      if (end == std::string_view::npos)
        text = {};
      else
        text.remove_prefix(end + 1U);
      if (line.empty()) continue;
      if (!header) {
        if (line != "expert-runtime-model-v1")
          return {invalid("unknown model descriptor artifact header"), {}};
        header = true;
        continue;
      }
      const auto fields = split_fields(line);
      const auto malformed = [&]() {
        return ParseModelDescriptorResult{
            invalid("malformed model descriptor line " +
                    std::to_string(line_number)),
            {}};
      };
      if (fields.empty()) return malformed();
      if (fields[0] == "model") {
        if (model || fields.size() != 6U || !valid_atom(fields[2]) ||
            !parse_u32(fields[1], descriptor.schema_version) ||
            !parse_u32(fields[3], descriptor.vocab_size) ||
            !parse_u32(fields[4], descriptor.max_context_tokens) ||
            !parse_u32(fields[5], descriptor.hidden_size))
          return malformed();
        descriptor.architecture_id = fields[2];
        model = true;
      } else if (fields[0] == "attribute") {
        std::uint64_t value{};
        if (!model || fields.size() != 3U || !valid_atom(fields[1]) ||
            !parse_u64(fields[2], value) ||
            !descriptor.attributes.emplace(std::string(fields[1]), value)
                 .second)
          return malformed();
      } else if (fields[0] == "model_tensor") {
        if (!model || fields.size() != 3U || !valid_atom(fields[1]) ||
            !valid_atom(fields[2]) ||
            !descriptor.tensor_bindings
                 .emplace(std::string(fields[1]), std::string(fields[2]))
                 .second)
          return malformed();
      } else if (fields[0] == "program_input" ||
                 fields[0] == "program_output") {
        if (!model || fields.size() != 4U || !valid_atom(fields[1]) ||
            !valid_atom(fields[2]) || !valid_atom(fields[3]))
          return malformed();
        auto& endpoints = fields[0] == "program_input"
                              ? descriptor.program_inputs
                              : descriptor.program_outputs;
        if (!endpoints
                 .emplace(std::string(fields[1]),
                          ProgramEndpointDescriptor{std::string(fields[2]),
                                                    std::string(fields[3])})
                 .second)
          return malformed();
      } else if (fields[0] == "kernel") {
        KernelRequirement item;
        if (!model || fields.size() != 3U || !valid_atom(fields[1]) ||
            !parse_u32(fields[2], item.abi_version))
          return malformed();
        item.capability = fields[1];
        descriptor.required_kernels.push_back(std::move(item));
      } else if (fields[0] == "exact_decode") {
        ExactDecodeProgramDescriptor item;
        if (!model || descriptor.exact_decode_program ||
            fields.size() != 4U || !valid_atom(fields[1]) ||
            !parse_u32(fields[2], item.abi_version) ||
            !parse_u32(fields[3], item.maximum_emitted_tokens))
          return malformed();
        item.capability = fields[1];
        descriptor.exact_decode_program = std::move(item);
      } else if (fields[0] == "exact_decode_parameter") {
        std::uint64_t value{};
        if (!descriptor.exact_decode_program || fields.size() != 3U ||
            !valid_atom(fields[1]) || !parse_u64(fields[2], value) ||
            !descriptor.exact_decode_program->parameters
                 .emplace(std::string(fields[1]), value)
                 .second)
          return malformed();
      } else if (fields[0] == "exact_decode_tensor") {
        if (!descriptor.exact_decode_program || fields.size() != 3U ||
            !valid_atom(fields[1]) || !valid_atom(fields[2]) ||
            !descriptor.exact_decode_program->tensor_bindings
                 .emplace(std::string(fields[1]), std::string(fields[2]))
                 .second)
          return malformed();
      } else if (fields[0] == "component") {
        RoutedExpertComponentDescriptor item;
        std::uint64_t namespace_offset{};
        if (!model || fields.size() != 14U || !valid_atom(fields[1]) ||
            !valid_atom(fields[9]) || !valid_atom(fields[13]) ||
            !parse_u64(fields[2], namespace_offset) ||
            namespace_offset >
                std::numeric_limits<std::uint64_t>::max() - namespace_base ||
            !parse_u32(fields[3], item.layer_count) ||
            !parse_u32(fields[4], item.experts_per_layer) ||
            !parse_u32(fields[5], item.route_width) ||
            !parse_u32(fields[6], item.shared_experts_per_layer) ||
            !parse_u32(fields[7], item.hidden_size) ||
            !parse_u32(fields[8], item.intermediate_size) ||
            !parse_u32(fields[10], item.execution_abi) ||
            !parse_u32(fields[11], item.source_abi) ||
            !parse_u32(fields[12], item.encoding_abi))
          return malformed();
        item.name = fields[1];
        item.namespace_id = namespace_base + namespace_offset;
        item.execution_capability = fields[9];
        item.encoding = fields[13];
        descriptor.routed_components.push_back(std::move(item));
      } else if (fields[0] == "component_attribute") {
        std::uint64_t value{};
        auto* component = fields.size() == 4U
                              ? find_component(descriptor, fields[1])
                              : nullptr;
        if (component == nullptr || !valid_atom(fields[2]) ||
            !parse_u64(fields[3], value) ||
            !component->attributes.emplace(std::string(fields[2]), value)
                 .second)
          return malformed();
      } else if (fields[0] == "router") {
        auto* component = fields.size() == 4U
                              ? find_component(descriptor, fields[1])
                              : nullptr;
        if (component == nullptr || !valid_atom(fields[2]) ||
            !component->router.capability.empty() ||
            !parse_u32(fields[3], component->router.abi_version))
          return malformed();
        component->router.capability = fields[2];
      } else if (fields[0] == "router_parameter") {
        std::uint64_t value{};
        auto* component = fields.size() == 4U
                              ? find_component(descriptor, fields[1])
                              : nullptr;
        if (component == nullptr || component->router.capability.empty() ||
            !valid_atom(fields[2]) || !parse_u64(fields[3], value) ||
            !component->router.parameters
                 .emplace(std::string(fields[2]), value)
                 .second)
          return malformed();
      } else if (fields[0] == "layer") {
        LayerProgramDescriptor item;
        if (!model || fields.size() != 6U || !valid_atom(fields[2]) ||
            !parse_u32(fields[1], item.logical_layer) ||
            !parse_u32(fields[3], item.block_abi_version) ||
            !parse_u32(fields[5], item.component_layer))
          return malformed();
        item.block_capability = fields[2];
        if (fields[4] != "-") {
          if (!valid_atom(fields[4])) return malformed();
          item.routed_component = fields[4];
        }
        descriptor.layer_program.push_back(std::move(item));
      } else if (fields[0] == "layer_parameter") {
        std::uint32_t logical_layer{};
        std::uint64_t value{};
        auto* layer = fields.size() == 4U &&
                              parse_u32(fields[1], logical_layer)
                          ? find_layer(descriptor, logical_layer)
                          : nullptr;
        if (layer == nullptr || !valid_atom(fields[2]) ||
            !parse_u64(fields[3], value) ||
            !layer->parameters.emplace(std::string(fields[2]), value).second)
          return malformed();
      } else if (fields[0] == "operation") {
        OperationProgramDescriptor item;
        if (!model || fields.size() != 7U || !valid_atom(fields[3]) ||
            !parse_u32(fields[1], item.logical_operation) ||
            (fields[2] != "-" &&
             !parse_u32(fields[2], item.logical_layer)) ||
            !parse_u32(fields[4], item.abi_version) ||
            !parse_u32(fields[6], item.component_layer))
          return malformed();
        if (fields[2] == "-")
          item.logical_layer = kModelLevelOperationLayer;
        item.capability = fields[3];
        if (fields[5] != "-") {
          if (!valid_atom(fields[5])) return malformed();
          item.routed_component = fields[5];
        }
        descriptor.operation_program.push_back(std::move(item));
      } else if (fields[0] == "operation_parameter") {
        std::uint32_t logical_operation{};
        std::uint64_t value{};
        auto* operation = fields.size() == 4U &&
                                  parse_u32(fields[1], logical_operation)
                              ? find_operation(descriptor, logical_operation)
                              : nullptr;
        if (operation == nullptr || !valid_atom(fields[2]) ||
            !parse_u64(fields[3], value) ||
            !operation->parameters.emplace(std::string(fields[2]), value)
                 .second)
          return malformed();
      } else if (fields[0] == "operation_tensor") {
        std::uint32_t logical_operation{};
        auto* operation = fields.size() == 4U &&
                                  parse_u32(fields[1], logical_operation)
                              ? find_operation(descriptor, logical_operation)
                              : nullptr;
        if (operation == nullptr || !valid_atom(fields[2]) ||
            !valid_atom(fields[3]) ||
            !operation->tensor_bindings
                 .emplace(std::string(fields[2]), std::string(fields[3]))
                 .second)
          return malformed();
      } else if (fields[0] == "operation_input" ||
                 fields[0] == "operation_output") {
        std::uint32_t logical_operation{};
        auto* operation = fields.size() == 5U &&
                                  parse_u32(fields[1], logical_operation)
                              ? find_operation(descriptor, logical_operation)
                              : nullptr;
        if (operation == nullptr || !valid_atom(fields[2]) ||
            !valid_atom(fields[3]) || !valid_atom(fields[4]))
          return malformed();
        auto& bindings = fields[0] == "operation_input"
                             ? operation->input_bindings
                             : operation->output_bindings;
        if (!bindings
                 .emplace(std::string(fields[2]),
                          OperationProgramDescriptor::ValueBinding{
                              std::string(fields[3]), std::string(fields[4])})
                 .second)
          return malformed();
      } else {
        return malformed();
      }
    }
    if (!header || !model)
      return {invalid("model descriptor artifact is incomplete"), {}};
    auto valid = validate_model_descriptor(descriptor);
    if (!valid.ok()) return {std::move(valid), {}};
    return {Status::success(), std::move(descriptor)};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("model descriptor parsing failed: ") + error.what()},
            {}};
  }
}

ParseModelDescriptorResult load_model_descriptor_artifact(
    const std::filesystem::path& path, std::uint64_t expected_bytes,
    const Sha256Digest& expected_sha256, const Sha256Digest& content_hash,
    std::uint64_t namespace_base) noexcept {
  try {
    constexpr std::uint64_t maximum_descriptor_bytes = 64ULL << 20U;
    if (expected_bytes == 0U || expected_bytes > maximum_descriptor_bytes ||
        !std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) != expected_bytes)
      return {invalid("model descriptor artifact size is invalid"), {}};
    std::vector<std::byte> bytes(static_cast<std::size_t>(expected_bytes));
    std::ifstream input(path, std::ios::binary);
    if (!input ||
        !input.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size())) ||
        input.peek() != std::ifstream::traits_type::eof())
      return {invalid("model descriptor artifact read failed"), {}};
    if (sha256(bytes) != expected_sha256)
      return {invalid("model descriptor artifact checksum mismatch"), {}};
    return parse_model_descriptor_artifact(
        {reinterpret_cast<const char*>(bytes.data()), bytes.size()},
        content_hash, namespace_base);
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("model descriptor loading failed: ") + error.what()},
            {}};
  }
}

}  // namespace expert::runtime
