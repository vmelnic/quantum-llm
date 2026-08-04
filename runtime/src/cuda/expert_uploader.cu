#include "expert/runtime/cuda/expert_uploader.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace expert::runtime::cuda {
namespace {

Status cuda_failure(const char* operation, cudaError_t error) {
  return Status(ErrorCode::upload_failed,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

}  // namespace

CudaExpertAllocation::CudaExpertAllocation(
    void* storage, std::size_t bytes, const std::int8_t* gate_up,
    const float* gate_up_scales, const std::int8_t* down,
    const float* down_scales) noexcept
    : storage_(storage), bytes_(bytes), gate_up_(gate_up),
      gate_up_scales_(gate_up_scales), down_(down), down_scales_(down_scales) {}

CudaExpertAllocation::~CudaExpertAllocation() {
  if (storage_ != nullptr) static_cast<void>(cudaFree(storage_));
}

std::size_t CudaExpertAllocation::bytes() const noexcept { return bytes_; }
const std::int8_t* CudaExpertAllocation::gate_up() const noexcept { return gate_up_; }
const float* CudaExpertAllocation::gate_up_scales() const noexcept { return gate_up_scales_; }
const std::int8_t* CudaExpertAllocation::down() const noexcept { return down_; }
const float* CudaExpertAllocation::down_scales() const noexcept { return down_scales_; }

CudaExpertUploader::CudaExpertUploader() {
  cudaStream_t stream{};
  if (const auto error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      error != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamCreate: ") + cudaGetErrorString(error));
  }
  stream_ = stream;
}

CudaExpertUploader::~CudaExpertUploader() {
  if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(static_cast<cudaStream_t>(stream_)));
}

OperationId CudaExpertUploader::upload(UploadRequest request,
                                       UploadCompletion completion) {
  const auto operation = next_operation_++;
  if (!completion) return operation;
  std::lock_guard stream_lock(stream_mutex_);
  const auto& sections = request.sections;
  const auto total = static_cast<std::size_t>(
      sections.gate_up_q_bytes + sections.gate_up_scale_bytes +
      sections.down_q_bytes + sections.down_scale_bytes);
  void* raw = nullptr;
  if (const auto error = cudaMalloc(&raw, total); error != cudaSuccess) {
    completion({cuda_failure("cudaMalloc expert", error), {}, 0});
    return operation;
  }
  auto* cursor = static_cast<std::byte*>(raw);
  auto* gate = reinterpret_cast<std::int8_t*>(cursor);
  cursor += sections.gate_up_q_bytes;
  auto* gate_scales = reinterpret_cast<float*>(cursor);
  cursor += sections.gate_up_scale_bytes;
  auto* down = reinterpret_cast<std::int8_t*>(cursor);
  cursor += sections.down_q_bytes;
  auto* down_scales = reinterpret_cast<float*>(cursor);
  const auto source = request.complete_record.data();
  const auto stream = static_cast<cudaStream_t>(stream_);
  const auto copy = [&](void* destination, std::uint64_t offset,
                        std::uint64_t bytes) {
    return cudaMemcpyAsync(destination, source + offset, static_cast<std::size_t>(bytes),
                           cudaMemcpyHostToDevice, stream);
  };
  cudaError_t error = copy(gate, sections.gate_up_q_offset, sections.gate_up_q_bytes);
  if (error == cudaSuccess)
    error = copy(gate_scales, sections.gate_up_scale_offset, sections.gate_up_scale_bytes);
  if (error == cudaSuccess)
    error = copy(down, sections.down_q_offset, sections.down_q_bytes);
  if (error == cudaSuccess)
    error = copy(down_scales, sections.down_scale_offset, sections.down_scale_bytes);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    static_cast<void>(cudaFree(raw));
    completion({cuda_failure("expert H2D", error), {}, 0});
    return operation;
  }
  auto allocation = std::make_shared<CudaExpertAllocation>(
      raw, total, gate, gate_scales, down, down_scales);
  completion({Status::success(), std::move(allocation), total});
  return operation;
}

void CudaExpertUploader::cancel(OperationId) noexcept {
  // v1 upload is bounded and synchronous; cancellation is observed before the
  // next cache operation. The asynchronous implementation will use events.
}

}  // namespace expert::runtime::cuda
