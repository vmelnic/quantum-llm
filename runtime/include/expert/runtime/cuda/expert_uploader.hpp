#pragma once

#include "expert/runtime/expert_cache.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>

namespace expert::runtime::cuda {

class CudaExpertAllocation final : public IDeviceAllocation {
 public:
  CudaExpertAllocation(void* storage, std::size_t bytes,
                       const std::int8_t* gate_up, const float* gate_up_scales,
                       const std::int8_t* down, const float* down_scales) noexcept;
  ~CudaExpertAllocation() override;
  CudaExpertAllocation(const CudaExpertAllocation&) = delete;
  CudaExpertAllocation& operator=(const CudaExpertAllocation&) = delete;

  [[nodiscard]] std::size_t bytes() const noexcept override;
  [[nodiscard]] const std::int8_t* gate_up() const noexcept;
  [[nodiscard]] const float* gate_up_scales() const noexcept;
  [[nodiscard]] const std::int8_t* down() const noexcept;
  [[nodiscard]] const float* down_scales() const noexcept;

 private:
  void* storage_{};
  std::size_t bytes_{};
  const std::int8_t* gate_up_{};
  const float* gate_up_scales_{};
  const std::int8_t* down_{};
  const float* down_scales_{};
};

class CudaExpertUploader final : public IDeviceUploader {
 public:
  CudaExpertUploader();
  ~CudaExpertUploader() override;
  CudaExpertUploader(const CudaExpertUploader&) = delete;
  CudaExpertUploader& operator=(const CudaExpertUploader&) = delete;

  OperationId upload(UploadRequest request, UploadCompletion completion) override;
  void cancel(OperationId operation) noexcept override;

 private:
  void* stream_{};
  std::atomic<std::uint64_t> next_operation_{1};
  std::mutex stream_mutex_;
};

}  // namespace expert::runtime::cuda
