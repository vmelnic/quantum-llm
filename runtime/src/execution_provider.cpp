#include "expert/runtime/execution_provider.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <utility>

namespace expert::runtime {

Status ExecutionProviderRegistry::add(
    ExecutionProviderDefinition provider) noexcept {
  try {
    if (provider.name.empty() || provider.capabilities.empty())
      return {ErrorCode::invalid_argument,
              "execution provider definition is incomplete"};
    if (std::any_of(providers_.begin(), providers_.end(), [&](const auto& item) {
          return item.name == provider.name;
        }))
      return {ErrorCode::invalid_argument,
              "execution provider name is duplicated"};
    std::set<std::string, std::less<>> names;
    for (const auto& capability : provider.capabilities) {
      if (capability.capability.empty() || capability.minimum_abi == 0U ||
          capability.maximum_abi < capability.minimum_abi ||
          !names.insert(capability.capability).second)
        return {ErrorCode::invalid_argument,
                "execution provider capability is invalid or duplicated"};
    }
    providers_.push_back(std::move(provider));
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("execution provider registration failed: ") +
                error.what()};
  }
}

BindExecutionProviderResult ExecutionProviderRegistry::bind(
    const ModelDescriptor& descriptor,
    ExecutionProviderBindingMode mode) const noexcept {
  try {
    const auto valid = validate_model_descriptor(descriptor);
    if (!valid.ok())
      return {{valid.code(), std::string(valid.message())}, {}};

    struct SelectedCapability final {
      std::uint32_t provider_index{};
      std::uint32_t capability_index{};
    };
    std::vector<SelectedCapability> selected;
    std::vector<KernelCapability> selected_capabilities;
    selected.reserve(descriptor.required_kernels.size());
    selected_capabilities.reserve(descriptor.required_kernels.size());
    for (const auto& required : descriptor.required_kernels) {
      const ExecutionProviderDefinition* best_provider = nullptr;
      const KernelCapability* best_capability = nullptr;
      std::uint32_t best_provider_index{};
      std::uint32_t best_capability_index{};
      for (std::size_t provider_index = 0U;
           provider_index < providers_.size(); ++provider_index) {
        const auto& candidate = providers_[provider_index];
        if (mode == ExecutionProviderBindingMode::executable &&
            !candidate.implementation)
          continue;
        for (std::size_t capability_index = 0U;
             capability_index < candidate.capabilities.size();
             ++capability_index) {
          const auto& capability = candidate.capabilities[capability_index];
          if (capability.capability != required.capability ||
              required.abi_version < capability.minimum_abi ||
              required.abi_version > capability.maximum_abi)
            continue;
          if (capability.validate) {
            const auto constrained = capability.validate(descriptor);
            if (!constrained.ok()) continue;
          }
          if (best_provider == nullptr ||
              candidate.priority > best_provider->priority ||
              (candidate.priority == best_provider->priority &&
               candidate.name < best_provider->name)) {
            best_provider = &candidate;
            best_capability = &capability;
            best_provider_index =
                static_cast<std::uint32_t>(provider_index);
            best_capability_index =
                static_cast<std::uint32_t>(capability_index);
          }
        }
      }
      if (best_provider == nullptr)
        return {{ErrorCode::invalid_argument,
                 "no execution provider implements required kernel: " +
                     required.capability},
                {}};
      selected.push_back({best_provider_index, best_capability_index});
      selected_capabilities.push_back(*best_capability);
    }

    auto compiled = compile_model_program(descriptor, selected_capabilities);
    if (!compiled.status.ok())
      return {{compiled.status.code(), std::string(compiled.status.message())},
              {}};
    for (std::size_t index = 0U; index < compiled.program.kernels.size();
         ++index) {
      const auto requirement =
          compiled.program.kernels[index].requirement_index;
      compiled.program.kernels[index].provider_registry_index =
          selected.at(requirement).provider_index;
      compiled.program.kernels[index].provider_capability_index =
          selected.at(requirement).capability_index;
    }

    BoundExecutionProvider plan;
    plan.model_content_hash = descriptor.content_hash;
    plan.model_schema_version = descriptor.schema_version;
    plan.program = std::move(compiled.program);
    std::set<std::uint32_t> referenced;
    for (const auto& kernel : plan.program.kernels)
      referenced.insert(kernel.provider_registry_index);
    plan.providers.reserve(referenced.size());
    for (const auto index : referenced)
      plan.providers.push_back({providers_.at(index).name, index,
                                providers_.at(index).implementation});
    return {Status::success(), std::move(plan)};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("execution provider binding failed: ") +
                 error.what()},
            {}};
  }
}

}  // namespace expert::runtime
