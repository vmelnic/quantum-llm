#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace expert::runtime {

struct WorkerLaunchOptions final {
  std::uint32_t max_context{};
  std::uint64_t ram_cache_gib{};
  std::uint64_t vram_cache_gib{};
  // fixed: vram_cache_gib is a hard ceiling. fit: a compatible routed
  // provider derives the expert-page ceiling after its fixed CUDA organs are
  // resident and reserves maximum dynamic state plus emergency headroom.
  std::string routed_vram_policy{"fixed"};
  std::uint32_t capacity{};
  std::uint64_t kv_cache_mib{};
  std::uint32_t kv_page_tokens{};
  // "artifact" preserves the provider-declared default. Any explicit value
  // is a generic execution policy and must be confirmed by the selected
  // provider through its service contract.
  std::string kv_cache_dtype{"artifact"};
  std::string placement_profile;
  std::optional<std::uint32_t> prefill_chunk_limit;
  std::optional<std::uint32_t> placement_settle_steps;
  // Generic diagnostics policy. Providers must not collect synchronous GPU
  // timings unless the service explicitly requests them.
  bool profile_gpu_phases{};
  // Optional secondary CUDA tier for artifact-declared active expert pages.
  // Empty preserves the ordinary single-device provider path.
  bool discover_active_expert_devices{};
  std::vector<int> active_expert_devices;
  std::uint64_t active_expert_device_cache_gib{};
  std::uint64_t active_expert_host_cache_gib{};
  // Provider extensions are deliberately opaque to the common parser. A new
  // execution provider can add an option without changing this contract.
  std::map<std::string, std::optional<std::string>, std::less<>> extensions;
};

struct WorkerLaunchOptionsResult final {
  Status status;
  WorkerLaunchOptions options;
};

// Parses only canonical --name=value / --boolean-extension arguments. All
// common resource fields are mandatory so worker startup cannot silently use
// provider-specific defaults.
[[nodiscard]] WorkerLaunchOptionsResult parse_worker_launch_options(
    std::span<const std::string_view> arguments) noexcept;

}  // namespace expert::runtime
