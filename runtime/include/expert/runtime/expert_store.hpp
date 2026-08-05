#pragma once

#include "expert/runtime/expert_cache.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace expert::runtime {

enum class ExpertResolveTarget : std::uint8_t {
  device,
  host_ready,
};

enum class ExpertPlacementKind : std::uint8_t {
  device,
  host,
};

struct ExpertResolveRequest final {
  ExpertKey key;
  PayloadRecord record;
  ExpertResolveTarget target{ExpertResolveTarget::device};
};

struct ResolvedExpert final {
  ExpertKey key;
  ExpertPlacementKind placement{ExpertPlacementKind::device};
  ExpertLease device_lease;
  HostExpertLease host_lease;

  [[nodiscard]] explicit operator bool() const noexcept {
    return placement == ExpertPlacementKind::device
               ? static_cast<bool>(device_lease)
               : static_cast<bool>(host_lease);
  }
};

struct ExpertResolveResult final {
  Status status;
  std::vector<ResolvedExpert> experts;
};

// Pollable ownership token for one union resolve. Results preserve request
// order. Any failed member cancels the unresolved remainder, so a partial
// expert set is never published to a scheduler.
class ExpertResolveHandle final {
 public:
  ExpertResolveHandle() = default;
  ExpertResolveHandle(const ExpertResolveHandle&) = delete;
  ExpertResolveHandle& operator=(const ExpertResolveHandle&) = delete;
  ExpertResolveHandle(ExpertResolveHandle&&) noexcept;
  ExpertResolveHandle& operator=(ExpertResolveHandle&&) noexcept;
  ~ExpertResolveHandle();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::optional<ExpertResolveResult> poll();
  void cancel() noexcept;

 private:
  struct Core;
  friend class LocalExpertStore;
  explicit ExpertResolveHandle(std::unique_ptr<Core> core) noexcept;
  std::unique_ptr<Core> core_;
};

// Placement boundary. Other implementations may resolve to compact GPU
// storage, an all-core CPU lane, or a remote owner without changing request
// state machines.
class IExpertStore {
 public:
  virtual ~IExpertStore() = default;
  [[nodiscard]] virtual ExpertResolveHandle resolve(
      std::span<const ExpertResolveRequest> requests) = 0;
};

class LocalExpertStore final : public IExpertStore {
 public:
  explicit LocalExpertStore(ExpertCache& cache) noexcept;
  [[nodiscard]] ExpertResolveHandle resolve(
      std::span<const ExpertResolveRequest> requests) override;

 private:
  ExpertCache& cache_;
};

}  // namespace expert::runtime
