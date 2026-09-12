#pragma once

#include "expert/runtime/active_expert_executor.hpp"
#include "expert/runtime/expert_cache.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace expert::runtime {

enum class ExpertResolveTarget : std::uint8_t {
  device,
  // Ensure an authenticated pageable host copy exists. Unlike host_ready,
  // this target may asynchronously read the immutable record from storage.
  host,
  host_ready,
  remote,
  automatic,
};

enum class ExpertPlacementKind : std::uint8_t {
  device,
  host,
  remote,
};

// Ownership token for one externally owned logical expert page. The owner may
// be an in-process secondary accelerator or a transport-backed executor. The
// lease binds immutable artifact identity to the universal activation-only
// execution contract; it never exposes or transports the expert weights.
class IRemoteExpertLease {
 public:
  virtual ~IRemoteExpertLease() = default;
  [[nodiscard]] virtual std::string_view owner() const noexcept = 0;
  [[nodiscard]] virtual const ActiveExpertIdentity& identity()
      const noexcept = 0;
  [[nodiscard]] virtual ActiveExpertExecutionHandle execute(
      ActiveExpertInvocation invocation) = 0;
};

struct ExpertResolveRequest final {
  ExpertKey key;
  PayloadRecord record;
  ExpertResolveTarget target{ExpertResolveTarget::device};
  ExpertAcquireOptions options{};
};

struct ResolvedExpert final {
  ExpertKey key;
  ExpertPlacementKind placement{ExpertPlacementKind::device};
  ExpertLease device_lease;
  HostExpertLease host_lease;
  std::shared_ptr<IRemoteExpertLease> remote_lease;

  [[nodiscard]] explicit operator bool() const noexcept {
    switch (placement) {
      case ExpertPlacementKind::device:
        return static_cast<bool>(device_lease);
      case ExpertPlacementKind::host:
        return static_cast<bool>(host_lease);
      case ExpertPlacementKind::remote:
        return static_cast<bool>(remote_lease);
    }
    return false;
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
  using Poll = std::function<std::optional<ExpertResolveResult>()>;
  using Cancel = std::function<void()>;

  ExpertResolveHandle();
  ExpertResolveHandle(const ExpertResolveHandle&) = delete;
  ExpertResolveHandle& operator=(const ExpertResolveHandle&) = delete;
  ExpertResolveHandle(ExpertResolveHandle&&) noexcept;
  ExpertResolveHandle& operator=(ExpertResolveHandle&&) noexcept;
  ~ExpertResolveHandle();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::optional<ExpertResolveResult> poll();
  void cancel() noexcept;

  // Public construction boundary for local, remote, and future distributed
  // stores. Callbacks must publish exactly one terminal result and make cancel
  // idempotent.
  [[nodiscard]] static ExpertResolveHandle from_callbacks(
      Poll poll, Cancel cancel, std::size_t size = 1U);

 private:
  struct Core;
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

struct RemoteExpertOwnerRange final {
  std::uint64_t namespace_id{};
  std::uint32_t first_layer{};
  std::uint32_t layer_count{};
  std::uint32_t first_expert{};
  std::uint32_t expert_count{};
  std::shared_ptr<IActiveExpertExecutor> executor;
};

// Runtime placement table. Ranges are deployment data, not model-family
// cases. Overlap is rejected so one logical page has at most one execution
// owner. The executor itself declares whether a wire transport is involved.
class ActiveExpertOwnerDirectory final {
 public:
  [[nodiscard]] Status add(RemoteExpertOwnerRange range) noexcept;
  [[nodiscard]] std::shared_ptr<IActiveExpertExecutor> find(
      const ExpertKey& key) const noexcept;
  [[nodiscard]] bool owns(const ExpertKey& key) const noexcept {
    return static_cast<bool>(find(key));
  }

 private:
  std::vector<RemoteExpertOwnerRange> ranges_;
};

struct ActiveExpertComponentContract final {
  Sha256Digest model_content_hash{};
  std::uint64_t namespace_id{};
  std::uint32_t layer_count{};
  std::uint32_t experts_per_layer{};
  std::uint32_t encoding_abi{};
  std::string execution_capability;
  std::uint32_t execution_abi{};
  std::uint32_t source_abi{};
};

// Resolves logical pages to executable owner leases. The payload record is
// used only to validate source identity; no path, extent, or weight byte is
// exposed to the executor.
class RemoteExpertStore final : public IExpertStore {
 public:
  RemoteExpertStore(ActiveExpertComponentContract contract,
                    ActiveExpertOwnerDirectory owners);
  [[nodiscard]] ExpertResolveHandle resolve(
      std::span<const ExpertResolveRequest> requests) override;
  [[nodiscard]] bool owns(const ExpertKey& key) const noexcept;

 private:
  ActiveExpertComponentContract contract_;
  ActiveExpertOwnerDirectory owners_;
};

// Exact local/remote union resolver. Explicit device/host requests remain
// local; explicit remote requests remain remote; automatic requests follow the
// owner directory. Completion preserves original order and fails the complete
// union if either placement fails.
class PlacementExpertStore final : public IExpertStore {
 public:
  PlacementExpertStore(IExpertStore& local, RemoteExpertStore& remote) noexcept;
  [[nodiscard]] ExpertResolveHandle resolve(
      std::span<const ExpertResolveRequest> requests) override;

 private:
  IExpertStore& local_;
  RemoteExpertStore& remote_;
};

}  // namespace expert::runtime
