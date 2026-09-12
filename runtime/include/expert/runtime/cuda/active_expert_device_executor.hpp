#pragma once

#include "expert/runtime/active_expert_executor.hpp"
#include "expert/runtime/expert_catalog.hpp"
#include "expert/runtime/model_descriptor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace expert::runtime::cuda {

// Optional in-process owner for artifact-declared routed experts. The primary
// provider continues to own routing, route weights and stable aggregation;
// this executor owns only immutable expert pages and exact expert evaluation
// on the configured secondary CUDA devices.
struct ActiveExpertDeviceExecutorConfig final {
  Sha256Digest model_content_hash{};
  RoutedExpertComponentDescriptor component;
  std::vector<int> device_ordinals;
  std::uint64_t device_cache_bytes_per_device{};
  // Kept free after the CUDA module, fixed execution workspace, and the
  // slot-aligned expert arena are resident. The configured cache is a ceiling
  // and is fitted independently on every selected device.
  std::uint64_t device_reserve_bytes_per_device{};
  std::uint64_t host_cache_bytes_total{};
  std::uint32_t staging_slots_per_device{};
  std::string input_abi;
  std::string output_abi;
  float activation_clamp{};
  bool round_intermediate_to_bf16{};
};

struct CreateActiveExpertDeviceExecutorResult final {
  Status status;
  std::shared_ptr<IActiveExpertExecutor> executor;
};

// Resolves secondary Pascal devices relative to the CUDA device selected by
// the primary provider. Discovery is launch policy, not a model-family rule.
[[nodiscard]] std::vector<int> discover_pascal_active_expert_devices();

[[nodiscard]] CreateActiveExpertDeviceExecutorResult
create_active_expert_device_executor(
    ActiveExpertDeviceExecutorConfig config,
    const ExpertCatalog& catalog,
    std::shared_ptr<IAsyncStorage> storage) noexcept;

}  // namespace expert::runtime::cuda
