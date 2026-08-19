#include "expert/runtime/moe_virtual_machine.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <utility>

namespace expert::runtime {

Status MoeVirtualMachine::create(
    ModelDescriptor descriptor, BoundExecutionProvider provider,
    std::span<const MoeVmComponentBinding> bindings,
    MoeVirtualMachine& destination) noexcept {
  try {
    const auto valid = validate_model_descriptor(descriptor);
    if (!valid.ok())
      return {valid.code(), std::string(valid.message())};
    if (!destination.components_.empty() || provider.providers.empty() ||
        provider.model_content_hash != descriptor.content_hash ||
        provider.model_schema_version != descriptor.schema_version ||
        provider.program.layers.size() != descriptor.layer_program.size() ||
        provider.program.operations.empty() ||
        bindings.size() != descriptor.routed_components.size())
      return {ErrorCode::invalid_argument,
              "MoE VM provider or component binding is incomplete"};

    std::vector<RoutedExpertRuntime*> components(
        descriptor.routed_components.size(), nullptr);
    std::set<std::string_view, std::less<>> names;
    for (const auto& binding : bindings) {
      if (binding.name.empty() || binding.runtime == nullptr ||
          !names.insert(binding.name).second)
        return {ErrorCode::invalid_argument,
                "MoE VM component binding is invalid or duplicated"};
      const auto found = std::find_if(
          descriptor.routed_components.begin(),
          descriptor.routed_components.end(), [&](const auto& component) {
            return component.name == binding.name;
          });
      if (found == descriptor.routed_components.end())
        return {ErrorCode::invalid_argument,
                "MoE VM component binding is unknown to the artifact"};
      const auto& runtime_component = binding.runtime->component();
      if (runtime_component.name != found->name ||
          runtime_component.namespace_id != found->namespace_id ||
          runtime_component.layer_count != found->layer_count ||
          runtime_component.experts_per_layer != found->experts_per_layer ||
          runtime_component.route_width != found->route_width ||
          runtime_component.encoding_abi != found->encoding_abi)
        return {ErrorCode::invalid_argument,
                "MoE VM runtime component disagrees with the artifact"};
      components[static_cast<std::size_t>(
          found - descriptor.routed_components.begin())] = binding.runtime;
    }
    if (std::any_of(components.begin(), components.end(),
                    [](const auto* component) { return component == nullptr; }))
      return {ErrorCode::invalid_argument,
              "MoE VM lacks a routed component binding"};

    MoeVirtualMachine candidate;
    candidate.descriptor_ = std::move(descriptor);
    candidate.provider_ = std::move(provider);
    candidate.components_ = std::move(components);
    destination = std::move(candidate);
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("MoE VM creation failed: ") + error.what()};
  }
}

std::span<const CompiledOperationProgram> MoeVirtualMachine::operations(
    std::uint32_t logical_layer) const noexcept {
  const auto first = std::lower_bound(
      provider_.program.operations.begin(), provider_.program.operations.end(),
      logical_layer, [](const auto& operation, std::uint32_t layer) {
        return operation.logical_layer < layer;
      });
  const auto last = std::upper_bound(
      first, provider_.program.operations.end(), logical_layer,
      [](std::uint32_t layer, const auto& operation) {
        return layer < operation.logical_layer;
      });
  return {first, last};
}

MoeVmResolveResult MoeVirtualMachine::resolve_route(
    std::uint32_t logical_layer, std::span<const std::uint32_t> experts,
    ExpertResolveTarget target, ExpertAcquireOptions options) {
  if (logical_layer >= provider_.program.layers.size())
    return {{ErrorCode::invalid_argument,
             "MoE VM logical layer is outside the model program"},
            {}};
  const auto& layer = provider_.program.layers[logical_layer];
  if (!layer.routed_component_index ||
      *layer.routed_component_index >= components_.size())
    return {{ErrorCode::invalid_argument,
             "MoE VM logical layer has no routed component"},
            {}};
  return resolve_component_route(*layer.routed_component_index,
                                 layer.component_layer, experts, target,
                                 options);
}

MoeVmResolveResult MoeVirtualMachine::resolve_operation_route(
    std::uint32_t logical_operation,
    std::span<const std::uint32_t> experts, ExpertResolveTarget target,
    ExpertAcquireOptions options) {
  if (logical_operation >= provider_.program.operations.size())
    return {{ErrorCode::invalid_argument,
             "MoE VM logical operation is outside the model program"},
            {}};
  const auto& operation = provider_.program.operations[logical_operation];
  if (operation.logical_operation != logical_operation ||
      !operation.routed_component_index ||
      *operation.routed_component_index >= components_.size())
    return {{ErrorCode::invalid_argument,
             "MoE VM operation has no routed component"},
            {}};
  return resolve_component_route(*operation.routed_component_index,
                                 operation.component_layer, experts, target,
                                 options);
}

MoeVmResolveResult MoeVirtualMachine::resolve_component_route(
    std::uint32_t component_index, std::uint32_t component_layer,
    std::span<const std::uint32_t> experts, ExpertResolveTarget target,
    ExpertAcquireOptions options) {
  auto* runtime = components_[component_index];
  const auto route_status =
      runtime->validate_route(component_layer, experts);
  if (!route_status.ok())
    return {{route_status.code(), std::string(route_status.message())}, {}};
  auto handle = runtime->resolve(component_layer, experts, target, options);
  if (!handle.valid())
    return {{ErrorCode::backpressure,
             "MoE VM expert store rejected the exact route"},
            {}};
  return {Status::success(), std::move(handle)};
}

}  // namespace expert::runtime
