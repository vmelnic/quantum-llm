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

namespace expert::runtime {

class IDeviceAllocation {
 public:
  virtual ~IDeviceAllocation() = default;
  [[nodiscard]] virtual std::size_t bytes() const noexcept = 0;
};

struct UploadRequest final {
  ExpertKey key;
  ExpertSections sections;
  std::span<const std::byte> complete_record;
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

struct ExpertCacheConfig final {
  TierBudget ram;
  TierBudget vram;
  bool retain_host_copy{true};
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
};

struct ExpertCacheCore;

class ExpertCache final {
 public:
  ExpertCache(ExpertCacheConfig config,
              std::shared_ptr<IAsyncStorage> storage,
              std::shared_ptr<IDeviceUploader> uploader,
              std::shared_ptr<FixedBufferPool> buffers);
  ~ExpertCache();
  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;

  [[nodiscard]] AcquireHandle acquire(const ExpertKey& key,
                                      const PayloadRecord& record);
  [[nodiscard]] std::optional<CacheEntrySnapshot> inspect(
      const ExpertKey& key) const;
  [[nodiscard]] TelemetrySnapshot telemetry() const noexcept;

  // Evicts every currently unreferenced copy and returns bytes released.
  std::uint64_t trim();

 private:
  std::shared_ptr<ExpertCacheCore> core_;
};

}  // namespace expert::runtime
