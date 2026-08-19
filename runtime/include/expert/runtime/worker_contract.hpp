#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace expert::runtime {

struct WorkerLaunchOptions final {
  std::uint32_t max_context{};
  std::uint64_t ram_cache_gib{};
  std::uint64_t vram_cache_gib{};
  std::uint32_t capacity{};
  std::uint64_t kv_cache_mib{};
  std::uint32_t kv_page_tokens{};
  std::string placement_profile;
  std::optional<std::uint32_t> prefill_chunk_limit;
  std::optional<std::uint32_t> placement_settle_steps;
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
