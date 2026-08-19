#pragma once

#include "expert/runtime/execution_provider.hpp"
#include "expert/runtime/routed_expert_runtime.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace expert::runtime {

struct MoeVmComponentBinding final {
  std::string_view name;
  RoutedExpertRuntime* runtime{};
};

struct MoeVmResolveResult final {
  Status status;
  ExpertResolveHandle handle;
};

// Artifact-driven sparse-model control plane. It owns the immutable model/VM
// program and maps component indices to injected expert stores. Numeric
// providers execute the returned operation stream; model family names never
// participate in binding, routing validation, or page resolution.
class MoeVirtualMachine final {
 public:
  MoeVirtualMachine() = default;
  MoeVirtualMachine(const MoeVirtualMachine&) = delete;
  MoeVirtualMachine& operator=(const MoeVirtualMachine&) = delete;
  MoeVirtualMachine(MoeVirtualMachine&&) noexcept = default;
  MoeVirtualMachine& operator=(MoeVirtualMachine&&) noexcept = default;

  [[nodiscard]] static Status create(
      ModelDescriptor descriptor, BoundExecutionProvider provider,
      std::span<const MoeVmComponentBinding> bindings,
      MoeVirtualMachine& destination) noexcept;

  [[nodiscard]] const ModelDescriptor& model() const noexcept {
    return descriptor_;
  }
  [[nodiscard]] const BoundExecutionProvider& provider() const noexcept {
    return provider_;
  }
  [[nodiscard]] std::span<const CompiledOperationProgram> operations(
      std::uint32_t logical_layer) const noexcept;
  [[nodiscard]] MoeVmResolveResult resolve_route(
      std::uint32_t logical_layer, std::span<const std::uint32_t> experts,
      ExpertResolveTarget target = ExpertResolveTarget::automatic,
      ExpertAcquireOptions options = {});
  // Resolves the routed component attached to one exact VM instruction. This
  // is the canonical API for programs with multiple sparse components on the
  // same logical layer (for example decoder and speculative draft routes).
  [[nodiscard]] MoeVmResolveResult resolve_operation_route(
      std::uint32_t logical_operation,
      std::span<const std::uint32_t> experts,
      ExpertResolveTarget target = ExpertResolveTarget::automatic,
      ExpertAcquireOptions options = {});

 private:
  [[nodiscard]] MoeVmResolveResult resolve_component_route(
      std::uint32_t component_index, std::uint32_t component_layer,
      std::span<const std::uint32_t> experts, ExpertResolveTarget target,
      ExpertAcquireOptions options);

  ModelDescriptor descriptor_;
  BoundExecutionProvider provider_;
  std::vector<RoutedExpertRuntime*> components_;
};

}  // namespace expert::runtime
