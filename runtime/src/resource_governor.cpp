#include "expert/runtime/resource_governor.hpp"

#include "expert/runtime/expert_cache.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace expert::runtime {
namespace {

std::uint64_t saturating_add(std::uint64_t left,
                             std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

}  // namespace

ExpertCacheMemoryTier::ExpertCacheMemoryTier(
    ExpertCache& cache, MemoryDomain domain,
    std::uint64_t protected_bytes) noexcept
    : cache_(cache), domain_(domain), protected_bytes_(protected_bytes) {}

std::string_view ExpertCacheMemoryTier::name() const noexcept {
  return domain_ == MemoryDomain::host ? "expert-cache-host"
                                        : "expert-cache-device";
}

MemoryDomain ExpertCacheMemoryTier::domain() const noexcept { return domain_; }

std::uint64_t ExpertCacheMemoryTier::used_bytes() const noexcept {
  const auto usage = cache_.usage();
  return domain_ == MemoryDomain::host ? usage.ram_bytes : usage.vram_bytes;
}

std::uint64_t ExpertCacheMemoryTier::protected_bytes() const noexcept {
  return protected_bytes_;
}

std::uint64_t ExpertCacheMemoryTier::trim_to(std::uint64_t target_bytes) {
  const auto current = cache_.usage();
  const auto after = domain_ == MemoryDomain::host
                         ? cache_.trim_to(target_bytes, current.vram_bytes)
                         : cache_.trim_to(current.ram_bytes, target_bytes);
  return domain_ == MemoryDomain::host ? after.ram_bytes : after.vram_bytes;
}

struct MemoryResourceGovernor::Core final {
  explicit Core(MemoryGovernorConfig value) : config(value) {}

  MemoryGovernorConfig config;
  std::vector<MemoryTierRegistration> tiers;
  mutable std::mutex mutex;
  std::uint64_t host_reserved{};
  std::uint64_t device_reserved{};
  std::uint64_t trim_calls{};
  std::uint64_t trimmed_bytes{};
  std::uint64_t rejected{};

  [[nodiscard]] std::uint64_t budget(MemoryDomain domain) const noexcept {
    return domain == MemoryDomain::host ? config.host_budget_bytes
                                         : config.device_budget_bytes;
  }

  [[nodiscard]] std::uint64_t emergency(MemoryDomain domain) const noexcept {
    return domain == MemoryDomain::host
               ? config.host_emergency_reserve_bytes
               : config.device_emergency_reserve_bytes;
  }

  [[nodiscard]] std::uint64_t& reserved(MemoryDomain domain) noexcept {
    return domain == MemoryDomain::host ? host_reserved : device_reserved;
  }

  [[nodiscard]] std::uint64_t tier_bytes(MemoryDomain domain) const noexcept {
    std::uint64_t total = 0U;
    for (const auto& registration : tiers) {
      if (registration.tier->domain() == domain)
        total = saturating_add(total, registration.tier->used_bytes());
    }
    return total;
  }

  Status rebalance_locked(MemoryDomain domain, std::uint64_t additional) {
    const auto limit = budget(domain) - emergency(domain);
    auto required = saturating_add(reserved(domain), additional);
    auto tiers_used = tier_bytes(domain);
    if (required <= limit && tiers_used <= limit - required)
      return Status::success();

    ++trim_calls;
    for (auto& registration : tiers) {
      auto& tier = *registration.tier;
      if (tier.domain() != domain) continue;
      const auto used = tier.used_bytes();
      const auto other = tiers_used >= used ? tiers_used - used : 0U;
      const auto available = required <= limit ? limit - required : 0U;
      const auto target = other < available ? available - other : 0U;
      const auto protected_target = std::max(target, tier.protected_bytes());
      const auto after = tier.trim_to(protected_target);
      if (used > after) trimmed_bytes += used - after;
      tiers_used = saturating_add(other, after);
      if (required <= limit && tiers_used <= limit - required)
        return Status::success();
    }
    ++rejected;
    return {ErrorCode::backpressure,
            domain == MemoryDomain::host
                ? "host memory budget remains exhausted after trimming"
                : "device memory budget remains exhausted after trimming"};
  }
};

MemoryResourceGovernor::MemoryResourceGovernor(MemoryGovernorConfig config)
    : core_(std::make_unique<Core>(config)) {
  if (config.host_budget_bytes == 0U || config.device_budget_bytes == 0U ||
      config.host_emergency_reserve_bytes >= config.host_budget_bytes ||
      config.device_emergency_reserve_bytes >= config.device_budget_bytes) {
    throw std::invalid_argument("invalid memory governor configuration");
  }
}

MemoryResourceGovernor::~MemoryResourceGovernor() = default;

void MemoryResourceGovernor::register_tier(
    MemoryTierRegistration registration) {
  if (!registration.tier || registration.tier->name().empty())
    throw std::invalid_argument("invalid memory tier registration");
  std::lock_guard lock(core_->mutex);
  const auto duplicate = std::any_of(
      core_->tiers.begin(), core_->tiers.end(), [&](const auto& existing) {
        return existing.tier.get() == registration.tier.get();
      });
  if (duplicate) throw std::invalid_argument("duplicate memory tier");
  core_->tiers.push_back(std::move(registration));
  std::stable_sort(core_->tiers.begin(), core_->tiers.end(),
                   [](const auto& left, const auto& right) {
                     return left.eviction_priority < right.eviction_priority;
                   });
}

Status MemoryResourceGovernor::reserve(MemoryDomain domain,
                                       std::uint64_t bytes) {
  if (bytes == 0U)
    return {ErrorCode::invalid_argument, "zero-byte reservation"};
  std::lock_guard lock(core_->mutex);
  auto status = core_->rebalance_locked(domain, bytes);
  if (!status.ok()) return status;
  core_->reserved(domain) += bytes;
  return Status::success();
}

void MemoryResourceGovernor::release(MemoryDomain domain,
                                     std::uint64_t bytes) noexcept {
  std::lock_guard lock(core_->mutex);
  auto& reserved = core_->reserved(domain);
  reserved = bytes <= reserved ? reserved - bytes : 0U;
}

Status MemoryResourceGovernor::rebalance(MemoryDomain domain,
                                         std::uint64_t additional_bytes) {
  std::lock_guard lock(core_->mutex);
  return core_->rebalance_locked(domain, additional_bytes);
}

MemoryGovernorSnapshot MemoryResourceGovernor::snapshot() const noexcept {
  std::lock_guard lock(core_->mutex);
  return {core_->config.host_budget_bytes,
          core_->config.device_budget_bytes,
          core_->host_reserved,
          core_->device_reserved,
          core_->tier_bytes(MemoryDomain::host),
          core_->tier_bytes(MemoryDomain::device),
          core_->trim_calls,
          core_->trimmed_bytes,
          core_->rejected};
}

}  // namespace expert::runtime
