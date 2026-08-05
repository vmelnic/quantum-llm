#pragma once

#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cache_state.hpp"
#include "expert/runtime/expert_key.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/storage.hpp"
#include "expert/runtime/telemetry.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace expert::runtime {

class IDeviceAllocation {
 public:
  virtual ~IDeviceAllocation() = default;
  [[nodiscard]] virtual std::size_t bytes() const noexcept = 0;
};

// Optional device-side residency index. publish() runs only after upload
// completion. retire() must make the entry invisible and wait for device-side
// users of the old generation before returning.
class IDeviceResidencyDirectory {
 public:
  virtual ~IDeviceResidencyDirectory() = default;
  [[nodiscard]] virtual Status publish(
      const ExpertKey& key,
      std::shared_ptr<IDeviceAllocation> allocation) = 0;
  virtual void retire(const ExpertKey& key) noexcept = 0;
};

struct UploadRequest final {
  ExpertKey key;
  std::uint32_t source_abi{};
  ExpertSections sections;
  std::span<const std::byte> complete_record;
  DeepSeekCompactSections compact;
};

struct UploadResult final {
  Status status;
  std::shared_ptr<IDeviceAllocation> allocation;
  std::uint64_t uploaded_bytes{};
};

using UploadCompletion = std::function<void(UploadResult)>;

class IDeviceUploader {
 public:
  virtual ~IDeviceUploader() = default;
  virtual OperationId upload(UploadRequest request,
                             UploadCompletion completion) = 0;
  virtual void cancel(OperationId operation) noexcept = 0;
};

struct TierBudget final {
  std::uint64_t capacity_bytes{};
  std::uint64_t high_watermark_bytes{};
  std::uint64_t low_watermark_bytes{};
};

// The cache index is global, while capacity is protected per layer group.
// shared_burst_bytes remains available to the currently active partitions and
// prevents a hard per-layer quota from rejecting legitimate top-k bursts.
struct CachePlacementConfig final {
  std::uint32_t layer_partition_count{1};
  std::uint32_t layers_per_partition{1};
  std::uint64_t ram_shared_burst_bytes{};
  std::uint64_t vram_shared_burst_bytes{};
  // Optional protected streaming ring. Zero keeps one LFU-admitted pool.
  std::uint64_t vram_transient_bytes{};
};

struct ExpertCacheConfig final {
  TierBudget ram;
  TierBudget vram;
  bool retain_host_copy{true};
  CachePlacementConfig placement;
};

class ExpertLease final {
 public:
  ExpertLease() = default;
  ExpertLease(const ExpertLease&) = delete;
  ExpertLease& operator=(const ExpertLease&) = delete;
  ExpertLease(ExpertLease&& other) noexcept;
  ExpertLease& operator=(ExpertLease&& other) noexcept;
  ~ExpertLease();

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] const IDeviceAllocation* get() const noexcept;
  [[nodiscard]] std::shared_ptr<const IDeviceAllocation> share() const noexcept;

  // Internal construction point used by the cache after upload completion.
  ExpertLease(std::shared_ptr<IDeviceAllocation> allocation,
              std::function<void()> release) noexcept;

 private:
  void reset() noexcept;

  std::shared_ptr<IDeviceAllocation> allocation_;
  std::function<void()> release_;
};

// Immutable validated RAM view. It holds the cache entry reference until the
// CPU executor or slab packer has finished, preventing host eviction without
// forcing a promotion to VRAM.
class HostExpertLease final {
 public:
  HostExpertLease() = default;
  HostExpertLease(const HostExpertLease&) = delete;
  HostExpertLease& operator=(const HostExpertLease&) = delete;
  HostExpertLease(HostExpertLease&& other) noexcept;
  HostExpertLease& operator=(HostExpertLease&& other) noexcept;
  ~HostExpertLease();

  [[nodiscard]] explicit operator bool() const noexcept;
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
  [[nodiscard]] const ExpertSections& sections() const noexcept;

  HostExpertLease(std::shared_ptr<const std::vector<std::byte>> bytes,
                  ExpertSections sections,
                  std::function<void()> release) noexcept;

 private:
  void reset() noexcept;

  std::shared_ptr<const std::vector<std::byte>> bytes_;
  ExpertSections sections_{};
  std::function<void()> release_;
};

struct AcquireResult final {
  Status status;
  ExpertLease lease;
};

class AcquireHandle final {
 public:
  AcquireHandle() = default;
  AcquireHandle(const AcquireHandle&) = delete;
  AcquireHandle& operator=(const AcquireHandle&) = delete;
  AcquireHandle(AcquireHandle&&) noexcept = default;
  AcquireHandle& operator=(AcquireHandle&&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept { return future_.valid(); }
  [[nodiscard]] AcquireResult get() { return future_.get(); }
  [[nodiscard]] std::future_status wait_for(std::chrono::milliseconds timeout) {
    return future_.wait_for(timeout);
  }
  void cancel() noexcept;

 private:
  friend class ExpertCache;
  friend struct ExpertCacheCore;
  AcquireHandle(std::future<AcquireResult> future,
                std::function<void()> cancel) noexcept;

  std::future<AcquireResult> future_;
  std::function<void()> cancel_;
};

struct CacheEntrySnapshot final {
  CacheState state{CacheState::absent};
  std::uint64_t reference_count{};
  std::uint64_t waiter_count{};
  bool has_host_copy{};
  bool has_device_copy{};
  std::uint32_t frequency{};
  std::uint64_t routing_score_mass_q20{};
  std::uint32_t routing_score_peak_q20{};
  std::uint64_t placement_temperature{};
};

struct ExpertCacheCore;

struct ExpertAccess final {
  ExpertKey key;
  std::uint32_t count{};
  double routing_score_sum{};
  double routing_score_max{};
};

class ExpertCache final {
 public:
  ExpertCache(ExpertCacheConfig config,
              std::shared_ptr<IAsyncStorage> storage,
              std::shared_ptr<IDeviceUploader> uploader,
              std::shared_ptr<FixedBufferPool> buffers,
              std::shared_ptr<IDeviceResidencyDirectory> directory = {});
  ~ExpertCache();
  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;

  [[nodiscard]] AcquireHandle acquire(const ExpertKey& key,
                                      const PayloadRecord& record);
  // Non-blocking RAM-tier lookup. Cold entries continue through acquire(); the
  // scheduler can therefore add CPU-local execution without changing SSD
  // failure semantics in the first vertical slice.
  [[nodiscard]] std::optional<HostExpertLease> try_acquire_host(
      const ExpertKey& key, const PayloadRecord& record,
      bool record_access = true);
  // The device directory bypasses acquire() on a hit. Feed routed selections
  // back into the LFU index so placement reflects GPU and CPU use equally.
  [[nodiscard]] bool record_access(const ExpertKey& key,
                                   std::uint32_t count = 1);
  [[nodiscard]] std::uint64_t record_accesses(
      std::span<const ExpertAccess> accesses);
  // Requires the caller to hold leases for device entries used by the current
  // route. Returns true only when this RAM entry can displace strictly colder,
  // currently unreferenced VRAM entries (or unused VRAM already exists).
  [[nodiscard]] bool vram_admission_would_improve(
      const ExpertKey& key, const PayloadRecord& record);
  [[nodiscard]] std::optional<CacheEntrySnapshot> inspect(
      const ExpertKey& key) const;
  [[nodiscard]] TelemetrySnapshot telemetry() const noexcept;

  // Evicts every currently unreferenced copy and returns bytes released.
  std::uint64_t trim();

 private:
  std::shared_ptr<ExpertCacheCore> core_;
};

}  // namespace expert::runtime
