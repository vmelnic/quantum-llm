#include "expert/runtime/worker_contract.hpp"

#include <algorithm>
#include <charconv>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace expert::runtime {
namespace {

std::uint64_t unsigned_integer(std::string_view text) {
  std::uint64_t value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("worker option is not canonical unsigned decimal");
  return value;
}

std::uint32_t u32(std::string_view text, bool allow_zero = false) {
  const auto value = unsigned_integer(text);
  if ((!allow_zero && value == 0U) ||
      value > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("worker option is outside uint32 range");
  return static_cast<std::uint32_t>(value);
}

std::string required_value(
    const std::map<std::string, std::optional<std::string>, std::less<>>& raw,
    std::string_view name) {
  const auto found = raw.find(name);
  if (found == raw.end() || !found->second || found->second->empty())
    throw std::invalid_argument("required worker option is absent: --" +
                                std::string(name));
  return *found->second;
}

std::vector<int> device_list(std::string_view text) {
  std::vector<int> result;
  std::size_t first = 0U;
  while (first < text.size()) {
    const auto separator = text.find(',', first);
    const auto part = text.substr(
        first, separator == std::string_view::npos ? text.size() - first
                                                   : separator - first);
    const auto value = unsigned_integer(part);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("active expert device is outside int range");
    result.push_back(static_cast<int>(value));
    if (separator == std::string_view::npos) break;
    first = separator + 1U;
  }
  if (result.empty())
    throw std::invalid_argument("active expert device list is empty");
  std::sort(result.begin(), result.end());
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    throw std::invalid_argument("active expert device is duplicated");
  return result;
}

}  // namespace

WorkerLaunchOptionsResult parse_worker_launch_options(
    std::span<const std::string_view> arguments) noexcept {
  try {
    std::map<std::string, std::optional<std::string>, std::less<>> raw;
    for (const auto argument : arguments) {
      if (!argument.starts_with("--") || argument.size() <= 2U)
        throw std::invalid_argument("worker launch contains a positional option");
      const auto separator = argument.find('=');
      const auto name = argument.substr(
          2U, separator == std::string_view::npos
                  ? std::string_view::npos
                  : separator - 2U);
      if (name.empty())
        throw std::invalid_argument("worker option name is empty");
      const auto valid_name = [](char value) {
        return (value >= 'a' && value <= 'z') ||
               (value >= '0' && value <= '9') || value == '-';
      };
      if (name.front() == '-' || name.back() == '-' ||
          !std::all_of(name.begin(), name.end(), valid_name))
        throw std::invalid_argument("worker option name is not canonical");
      std::optional<std::string> value;
      if (separator != std::string_view::npos) {
        const auto text = argument.substr(separator + 1U);
        if (text.empty())
          throw std::invalid_argument("worker option value is empty");
        value = std::string(text);
      }
      if (!raw.emplace(std::string(name), std::move(value)).second)
        throw std::invalid_argument("worker option is duplicated");
    }

    WorkerLaunchOptions options;
    options.max_context = u32(required_value(raw, "max-context"));
    options.ram_cache_gib =
        unsigned_integer(required_value(raw, "ram-cache-gib"));
    options.vram_cache_gib =
        unsigned_integer(required_value(raw, "vram-cache-gib"));
    if (const auto found = raw.find("routed-vram-policy"); found != raw.end())
      options.routed_vram_policy = required_value(raw, "routed-vram-policy");
    options.capacity = u32(required_value(raw, "capacity"));
    options.kv_cache_mib =
        unsigned_integer(required_value(raw, "kv-cache-mib"));
    options.kv_page_tokens = u32(required_value(raw, "kv-page-tokens"));
    if (const auto found = raw.find("kv-cache-dtype"); found != raw.end())
      options.kv_cache_dtype = required_value(raw, "kv-cache-dtype");
    options.placement_profile = required_value(raw, "placement-profile");
    if (options.ram_cache_gib == 0U || options.vram_cache_gib == 0U ||
        options.kv_cache_mib == 0U ||
        (options.routed_vram_policy != "fixed" &&
         options.routed_vram_policy != "fit") ||
        (options.placement_profile != "latency" &&
         options.placement_profile != "balanced" &&
         options.placement_profile != "capacity"))
      throw std::invalid_argument("worker resource contract is invalid");
    if (const auto found = raw.find("prefill-chunk-limit");
        found != raw.end())
      options.prefill_chunk_limit =
          u32(required_value(raw, "prefill-chunk-limit"));
    if (const auto found = raw.find("placement-settle-steps");
        found != raw.end())
      options.placement_settle_steps =
          u32(required_value(raw, "placement-settle-steps"), true);
    if (const auto found = raw.find("profile-gpu-phases");
        found != raw.end()) {
      if (found->second)
        throw std::invalid_argument(
            "boolean worker option profile-gpu-phases has a value");
      options.profile_gpu_phases = true;
    }
    if (const auto found = raw.find("active-expert-devices");
        found != raw.end()) {
      const auto devices = required_value(raw, "active-expert-devices");
      if (devices == "auto")
        options.discover_active_expert_devices = true;
      else
        options.active_expert_devices = device_list(devices);
    }
    if (const auto found = raw.find("active-expert-device-cache-gib");
        found != raw.end())
      options.active_expert_device_cache_gib = unsigned_integer(
          required_value(raw, "active-expert-device-cache-gib"));
    if (const auto found = raw.find("active-expert-host-cache-gib");
        found != raw.end())
      options.active_expert_host_cache_gib = unsigned_integer(
          required_value(raw, "active-expert-host-cache-gib"));
    const bool active_expert_tier_configured =
        options.discover_active_expert_devices ||
        !options.active_expert_devices.empty();
    if (active_expert_tier_configured !=
            (options.active_expert_device_cache_gib != 0U) ||
        active_expert_tier_configured !=
            (options.active_expert_host_cache_gib != 0U))
      throw std::invalid_argument(
          "active expert devices and cache budgets must be configured together");

    constexpr std::string_view common_names[]{
        "max-context",          "ram-cache-gib",
        "vram-cache-gib",      "routed-vram-policy", "capacity",
        "kv-cache-mib",        "kv-page-tokens",
        "kv-cache-dtype",
        "placement-profile",   "prefill-chunk-limit",
        "placement-settle-steps", "profile-gpu-phases",
        "active-expert-devices", "active-expert-device-cache-gib",
        "active-expert-host-cache-gib"};
    for (auto& [name, value] : raw) {
      if (std::find(std::begin(common_names), std::end(common_names), name) ==
          std::end(common_names))
        options.extensions.emplace(std::move(name), std::move(value));
    }
    return {Status::success(), std::move(options)};
  } catch (const std::exception& error) {
    return {{ErrorCode::invalid_argument,
             std::string("invalid worker launch contract: ") + error.what()},
            {}};
  }
}

}  // namespace expert::runtime
