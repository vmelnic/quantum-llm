#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace expert::runtime {

class ExpertCache;

enum class MemoryDomain : std::uint8_t { host, device };

class ITrimmableMemoryTier {
 public:
  virtual ~ITrimmableMemoryTier() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual MemoryDomain domain() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t used_bytes() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t protected_bytes() const noexcept = 0;
  // Returns actual usage after trimming. Implementations may remain above the
  // target when live leases or in-flight kernels protect storage.
  [[nodiscard]] virtual std::uint64_t trim_to(
      std::uint64_t target_bytes) = 0;
};

class ExpertCacheMemoryTier final : public ITrimmableMemoryTier {
 public:
  ExpertCacheMemoryTier(ExpertCache& cache, MemoryDomain domain,
                        std::uint64_t protected_bytes = 0U) noexcept;
  [[nodiscard]] std::string_view name() const noexcept override;
  [[nodiscard]] MemoryDomain domain() const noexcept override;
  [[nodiscard]] std::uint64_t used_bytes() const noexcept override;
  [[nodiscard]] std::uint64_t protected_bytes() const noexcept override;
  [[nodiscard]] std::uint64_t trim_to(
      std::uint64_t target_bytes) override;

 private:
  ExpertCache& cache_;
  MemoryDomain domain_;
  std::uint64_t protected_bytes_{};
};

struct MemoryGovernorConfig final {
  std::uint64_t host_budget_bytes{};
  std::uint64_t device_budget_bytes{};
  std::uint64_t host_emergency_reserve_bytes{};
  std::uint64_t device_emergency_reserve_bytes{};
};

struct MemoryTierRegistration final {
  std::shared_ptr<ITrimmableMemoryTier> tier;
  // Lower values are trimmed first. Dense weights/KV use a higher priority or
  // remain outside trimmable tiers as explicit reservations.
  std::uint32_t eviction_priority{};
};

struct MemoryGovernorSnapshot final {
  std::uint64_t host_budget_bytes{};
  std::uint64_t device_budget_bytes{};
  std::uint64_t host_reserved_bytes{};
  std::uint64_t device_reserved_bytes{};
  std::uint64_t host_tier_bytes{};
  std::uint64_t device_tier_bytes{};
  std::uint64_t trim_calls{};
  std::uint64_t trimmed_bytes{};
  std::uint64_t rejected_reservations{};
};

// Computes a provider's trimmable device-cache budget after every hot,
// non-evictable allocation has been accounted for. The configured cache is a
// ceiling, not a promise that may overcommit the device. Providers still
// declare their own minimum useful cache from artifact geometry.
struct DeviceCacheBudgetRequest final {
  std::uint64_t available_device_bytes{};
  std::uint64_t fixed_device_bytes{};
  std::uint64_t execution_workspace_bytes{};
  std::uint64_t emergency_reserve_bytes{};
  std::uint64_t requested_cache_bytes{};
  std::uint64_t minimum_cache_bytes{};
};

struct DeviceCacheBudgetResult final {
  Status status;
  std::uint64_t effective_cache_bytes{};
};

[[nodiscard]] DeviceCacheBudgetResult fit_device_cache_budget(
    const DeviceCacheBudgetRequest& request) noexcept;

// One admission authority for dense state, KV/request reservations, and every
// trimmable cache tier. reserve() first asks lower-priority tiers to yield and
// fails closed if live storage still exceeds the physical budget.
class MemoryResourceGovernor final {
 public:
  explicit MemoryResourceGovernor(MemoryGovernorConfig config);
  ~MemoryResourceGovernor();
  MemoryResourceGovernor(const MemoryResourceGovernor&) = delete;
  MemoryResourceGovernor& operator=(const MemoryResourceGovernor&) = delete;
  void register_tier(MemoryTierRegistration registration);

  [[nodiscard]] Status reserve(MemoryDomain domain, std::uint64_t bytes);
  void release(MemoryDomain domain, std::uint64_t bytes) noexcept;
  [[nodiscard]] Status rebalance(MemoryDomain domain,
                                 std::uint64_t additional_bytes = 0U);
  [[nodiscard]] MemoryGovernorSnapshot snapshot() const noexcept;

 private:
  struct Core;
  std::unique_ptr<Core> core_;
};

}  // namespace expert::runtime
