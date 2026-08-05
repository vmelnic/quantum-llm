#include "expert/runtime/cuda/deepseek_typed.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <string>

namespace expert::runtime::cuda {
namespace {

Status failure(cudaError_t error, const char* operation) {
  return {ErrorCode::upload_failed,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

}  // namespace

DeepSeekTypedTensor::DeepSeekTypedTensor(void* data, std::uint64_t bytes,
                                         DeepSeekDtype dtype) noexcept
    : data_(data), bytes_(bytes), dtype_(dtype) {}

DeepSeekTypedTensor::~DeepSeekTypedTensor() {
  if (data_) static_cast<void>(cudaFree(data_));
}

Status DeepSeekTypedTensor::upload(
    std::uint64_t offset, std::span<const std::byte> source) noexcept {
  if (source.empty() || offset > bytes_ || source.size() > bytes_ - offset)
    return {ErrorCode::invalid_argument, "invalid typed tensor upload range"};
  const auto error = cudaMemcpy(static_cast<std::byte*>(data_) + offset,
                                source.data(), source.size(),
                                cudaMemcpyHostToDevice);
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek typed tensor H2D");
}

DeepSeekTypedAllocationResult allocate_deepseek_typed_tensor(
    std::uint64_t bytes, DeepSeekDtype dtype) noexcept {
  const std::uint64_t element_bytes =
      dtype == DeepSeekDtype::bf16 ? 2U :
      dtype == DeepSeekDtype::f32 ? 4U : 8U;
  if (bytes == 0U || bytes % element_bytes != 0U ||
      bytes > std::numeric_limits<std::size_t>::max())
    return {{ErrorCode::invalid_argument, "invalid typed tensor allocation"}, {}};
  void* data = nullptr;
  const auto error = cudaMalloc(&data, static_cast<std::size_t>(bytes));
  if (error != cudaSuccess)
    return {failure(error, "DeepSeek typed tensor allocation"), {}};
  return {Status::success(),
          std::make_shared<DeepSeekTypedTensor>(data, bytes, dtype)};
}

}  // namespace expert::runtime::cuda
