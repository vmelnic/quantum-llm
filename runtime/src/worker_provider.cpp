#include "expert/runtime/worker_provider.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace expert::runtime {

Status WorkerProviderRegistry::add(WorkerProviderDefinition provider) noexcept {
  try {
    if (provider.name.empty() || provider.capabilities.empty() ||
        provider.entry == nullptr)
      return {ErrorCode::invalid_argument,
              "worker provider definition is incomplete"};
    if (std::any_of(providers_.begin(), providers_.end(),
                    [&](const auto& item) { return item.name == provider.name; }))
      return {ErrorCode::invalid_argument,
              "worker provider name is duplicated"};
    providers_.push_back(std::move(provider));
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("worker provider registration failed: ") +
                error.what()};
  }
}

SelectWorkerProviderResult WorkerProviderRegistry::select(
    const ModelDescriptor& descriptor) const noexcept {
  try {
    const WorkerProviderDefinition* selected = nullptr;
    for (const auto& provider : providers_) {
      if (!provider_supports_model(descriptor, provider.capabilities).ok())
        continue;
      if (selected == nullptr || provider.priority > selected->priority ||
          (provider.priority == selected->priority &&
           provider.name < selected->name))
        selected = &provider;
    }
    if (selected == nullptr)
      return {{ErrorCode::invalid_argument,
               "no worker provider implements the artifact operation set"},
              nullptr};
    return {Status::success(), selected};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("worker provider selection failed: ") + error.what()},
            nullptr};
  }
}

}  // namespace expert::runtime
