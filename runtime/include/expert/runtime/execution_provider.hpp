#pragma once

#include "expert/runtime/model_descriptor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace expert::runtime {

class IOperationProvider;

enum class ExecutionProviderBindingMode : std::uint8_t {
  metadata,
  executable,
};

// One execution backend registered by capability rather than model family.
// Higher priority wins only after the complete descriptor has passed exact
// capability, ABI, encoding, and geometry validation.
struct ExecutionProviderDefinition final {
  std::string name;
  std::uint32_t priority{};
  std::vector<KernelCapability> capabilities;
  // Optional for legacy schema/control-plane inspection. Executable binding
  // excludes definitions without a callable implementation.
  std::shared_ptr<IOperationProvider> implementation;

  ExecutionProviderDefinition() = default;
  ExecutionProviderDefinition(
      std::string name_value, std::uint32_t priority_value,
      std::vector<KernelCapability> capabilities_value,
      std::shared_ptr<IOperationProvider> implementation_value = {})
      : name(std::move(name_value)), priority(priority_value),
        capabilities(std::move(capabilities_value)),
        implementation(std::move(implementation_value)) {}
};

struct BoundExecutionProviderReference final {
  std::string name;
  std::uint32_t registry_index{};
  std::shared_ptr<IOperationProvider> implementation;
};

struct BoundExecutionProvider final {
  // The plan is intentionally composable: each kernel binding names the
  // provider that owns it. The common VM therefore never requires one backend
  // to implement an entire model family.
  std::vector<BoundExecutionProviderReference> providers;
  Sha256Digest model_content_hash{};
  std::uint32_t model_schema_version{};
  CompiledModelProgram program;
};

struct BindExecutionProviderResult final {
  Status status;
  BoundExecutionProvider provider;
};

// Deterministic, per-kernel provider negotiation for the VM control plane. No
// architecture_id is inspected: a previously unseen model binds when and only
// when the registered providers collectively implement every operation ABI
// declared by its artifact.
class ExecutionProviderRegistry final {
 public:
  [[nodiscard]] Status add(ExecutionProviderDefinition provider) noexcept;
  [[nodiscard]] BindExecutionProviderResult bind(
      const ModelDescriptor& descriptor,
      ExecutionProviderBindingMode mode =
          ExecutionProviderBindingMode::metadata) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return providers_.size(); }

 private:
  std::vector<ExecutionProviderDefinition> providers_;
};

}  // namespace expert::runtime
