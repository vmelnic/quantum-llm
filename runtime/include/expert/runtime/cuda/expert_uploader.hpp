#pragma once

#include "expert/runtime/expert_cache.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace expert::runtime::cuda {

struct CudaExpertPool;

struct CudaExpertUploaderOptions final {
  // Maximum bytes retained in idle, exactly-sized device slots. Live bytes
  // remain bounded by ExpertCache; zero preserves release-on-eviction.
  std::uint64_t recycled_capacity_bytes{};
  // Reuse one serialized compact-source staging allocation across admissions.
  bool persistent_staging{};
  // Retain immutable compact DeepSeek FP4 records on device independently of
  // expanded compute slots. Zero disables this L1 tier.
  std::uint64_t compact_cache_capacity_bytes{};
  // Publish routed DeepSeek FP4 records directly instead of materializing the
  // derived INT8-per-row allocation. Shared FP8 experts remain expanded.
  bool direct_compact_execution{};
};

struct CudaExpertUploaderTelemetry final {
  std::uint64_t device_allocations{};
  std::uint64_t recycled_acquires{};
  std::uint64_t recycled_releases{};
  std::uint64_t device_releases{};
  std::uint64_t recycled_bytes{};
  std::uint64_t device_bytes_high_water{};
  std::uint64_t staging_allocations{};
  std::uint64_t compact_cache_hits{};
  std::uint64_t compact_cache_misses{};
  std::uint64_t compact_cache_evictions{};
  std::uint64_t compact_h2d_bytes{};
  std::uint64_t compact_cache_bytes{};
  std::uint64_t compact_cache_high_water{};
};

class CudaExpertAllocation final : public IDeviceAllocation {
 public:
  CudaExpertAllocation(std::shared_ptr<CudaExpertPool> pool, void* storage,
                       std::size_t bytes,
                       const std::int8_t* gate_up, const float* gate_up_scales,
                       const std::int8_t* down, const float* down_scales,
                       std::uint32_t hidden = 0,
                       std::uint32_t intermediate = 0,
                       bool packed_fp4 = false,
                       bool relu2 = false,
                       bool native_nvfp4 = false) noexcept;
  ~CudaExpertAllocation() override;
  CudaExpertAllocation(const CudaExpertAllocation&) = delete;
  CudaExpertAllocation& operator=(const CudaExpertAllocation&) = delete;

  [[nodiscard]] std::size_t bytes() const noexcept override;
  [[nodiscard]] const std::int8_t* gate_up() const noexcept;
  [[nodiscard]] const float* gate_up_scales() const noexcept;
  [[nodiscard]] const std::int8_t* down() const noexcept;
  [[nodiscard]] const float* down_scales() const noexcept;
  // Nonzero only for FP4 block-32 records, where the directory derives the
  // w1/w3/w2 views from the contiguous gate-up/down sections.
  [[nodiscard]] std::uint32_t hidden() const noexcept;
  [[nodiscard]] std::uint32_t intermediate() const noexcept;
  [[nodiscard]] bool packed_fp4() const noexcept;
  [[nodiscard]] bool relu2() const noexcept;
  [[nodiscard]] bool native_nvfp4() const noexcept;

 private:
  std::shared_ptr<CudaExpertPool> pool_;
  void* storage_{};
  std::size_t bytes_{};
  const std::int8_t* gate_up_{};
  const float* gate_up_scales_{};
  const std::int8_t* down_{};
  const float* down_scales_{};
  std::uint32_t hidden_{};
  std::uint32_t intermediate_{};
  bool packed_fp4_{};
  bool relu2_{};
  bool native_nvfp4_{};
};

class CudaCompactExpertAllocation final : public IDeviceAllocation {
 public:
  CudaCompactExpertAllocation(std::shared_ptr<CudaExpertPool> pool,
                              void* storage, std::size_t bytes,
                              DeepSeekCompactSections sections) noexcept;
  ~CudaCompactExpertAllocation() override;
  CudaCompactExpertAllocation(const CudaCompactExpertAllocation&) = delete;
  CudaCompactExpertAllocation& operator=(
      const CudaCompactExpertAllocation&) = delete;

  [[nodiscard]] std::size_t bytes() const noexcept override;
  [[nodiscard]] const std::uint8_t* base() const noexcept;
  [[nodiscard]] const DeepSeekCompactSections& sections() const noexcept;

 private:
  std::shared_ptr<CudaExpertPool> pool_;
  void* storage_{};
  std::size_t bytes_{};
  DeepSeekCompactSections sections_{};
};

class CudaExpertUploader final : public IDeviceUploader {
 public:
  explicit CudaExpertUploader(CudaExpertUploaderOptions options = {});
  ~CudaExpertUploader() override;
  CudaExpertUploader(const CudaExpertUploader&) = delete;
  CudaExpertUploader& operator=(const CudaExpertUploader&) = delete;

  OperationId upload(UploadRequest request, UploadCompletion completion) override;
  void cancel(OperationId operation) noexcept override;
  [[nodiscard]] CudaExpertUploaderTelemetry telemetry() const noexcept;

 private:
  std::shared_ptr<CudaExpertPool> pool_;
  std::atomic<std::uint64_t> next_operation_{1};
};

}  // namespace expert::runtime::cuda
