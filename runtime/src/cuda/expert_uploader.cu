#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/deepseek_admission.hpp"
#include "expert/runtime/expert_record.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace expert::runtime::cuda {
namespace {

Status cuda_failure(const char* operation, cudaError_t error) {
  return Status(ErrorCode::upload_failed,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

}  // namespace

struct CudaExpertPool final {
  cudaStream_t stream{};
  std::mutex mutex;

  ~CudaExpertPool() {
    if (stream != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream));
      static_cast<void>(cudaStreamDestroy(stream));
    }
  }
};

CudaExpertAllocation::CudaExpertAllocation(
    std::shared_ptr<CudaExpertPool> pool, void* storage, std::size_t bytes,
    const std::int8_t* gate_up,
    const float* gate_up_scales, const std::int8_t* down,
    const float* down_scales) noexcept
    : pool_(std::move(pool)), storage_(storage), bytes_(bytes), gate_up_(gate_up),
      gate_up_scales_(gate_up_scales), down_(down), down_scales_(down_scales) {}

CudaExpertAllocation::~CudaExpertAllocation() {
  if (storage_ != nullptr && pool_) {
    std::lock_guard lock(pool_->mutex);
    static_cast<void>(cudaFreeAsync(storage_, pool_->stream));
  }
}

std::size_t CudaExpertAllocation::bytes() const noexcept { return bytes_; }
const std::int8_t* CudaExpertAllocation::gate_up() const noexcept { return gate_up_; }
const float* CudaExpertAllocation::gate_up_scales() const noexcept { return gate_up_scales_; }
const std::int8_t* CudaExpertAllocation::down() const noexcept { return down_; }
const float* CudaExpertAllocation::down_scales() const noexcept { return down_scales_; }

CudaExpertUploader::CudaExpertUploader() : pool_(std::make_shared<CudaExpertPool>()) {
  int pools_supported = 0;
  if (const auto error = cudaDeviceGetAttribute(
          &pools_supported, cudaDevAttrMemoryPoolsSupported, 0);
      error != cudaSuccess || pools_supported == 0) {
    throw std::runtime_error("CUDA device does not support stream-ordered memory pools");
  }
  cudaStream_t stream{};
  if (const auto error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      error != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamCreate: ") + cudaGetErrorString(error));
  }
  pool_->stream = stream;
  cudaMemPool_t memory_pool{};
  if (const auto error = cudaDeviceGetDefaultMemPool(&memory_pool, 0);
      error != cudaSuccess) {
    throw std::runtime_error(std::string("cudaDeviceGetDefaultMemPool: ") +
                             cudaGetErrorString(error));
  }
  auto threshold = std::numeric_limits<std::uint64_t>::max();
  if (const auto error = cudaMemPoolSetAttribute(
          memory_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
      error != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemPoolSetAttribute: ") +
                             cudaGetErrorString(error));
  }
}

CudaExpertUploader::~CudaExpertUploader() = default;

OperationId CudaExpertUploader::upload(UploadRequest request,
                                       UploadCompletion completion) {
  const auto operation = next_operation_++;
  if (!completion) return operation;
  std::unique_lock stream_lock(pool_->mutex);
  const auto& sections = request.sections;
  const auto total = static_cast<std::size_t>(
      sections.gate_up_q_bytes + sections.gate_up_scale_bytes +
      sections.down_q_bytes + sections.down_scale_bytes);
  void* raw = nullptr;
  const auto stream = pool_->stream;
  auto error = cudaMallocAsync(&raw, total, stream);
  if (error != cudaSuccess) {
    stream_lock.unlock();
    completion({cuda_failure("cudaMallocAsync expert", error), {}, 0});
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
  if (request.key.quant_abi == kExpertQuantAbiDeepSeekSm86) {
    void* compact_raw = nullptr;
    error = cudaMallocAsync(&compact_raw, request.complete_record.size(), stream);
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(compact_raw, source, request.complete_record.size(),
                              cudaMemcpyHostToDevice, stream);
    }
    Status admission = error == cudaSuccess
                           ? Status::success()
                           : cuda_failure("DeepSeek compact H2D", error);
    const auto* compact = static_cast<const std::uint8_t*>(compact_raw);
    const auto launch = [&](std::uint64_t weight_offset,
                            std::uint64_t scale_offset, std::int8_t* output,
                            float* output_scales, std::uint32_t rows,
                            std::uint32_t columns) {
      if (!admission.ok()) return;
      if (request.source_abi == kExpertSourceAbiDeepSeekCompactV1) {
        admission = admit_deepseek_projection(
            {compact + weight_offset, compact + scale_offset, output,
             output_scales, rows, columns, stream, true});
      } else if (request.source_abi ==
                 kExpertSourceAbiDeepSeekFp8Block128V1) {
        admission = admit_deepseek_fp8_projection(
            {compact + weight_offset, compact + scale_offset, output,
             output_scales, rows, columns, stream});
      } else {
        admission = Status(ErrorCode::invalid_argument,
                           "unsupported DeepSeek CUDA source ABI");
      }
    };
    if (admission.ok()) {
      launch(request.compact.w1_weight_offset, request.compact.w1_scale_offset,
             gate, gate_scales, 2048U, 4096U);
      launch(request.compact.w3_weight_offset, request.compact.w3_scale_offset,
             gate + 2048ULL * 4096U, gate_scales + 2048U, 2048U, 4096U);
      launch(request.compact.w2_weight_offset, request.compact.w2_scale_offset,
             down, down_scales, 4096U, 2048U);
    }
    if (compact_raw != nullptr) {
      const auto free_error = cudaFreeAsync(compact_raw, stream);
      if (admission.ok() && free_error != cudaSuccess) {
        admission = cuda_failure("DeepSeek compact release", free_error);
      }
      const auto synchronize_error = cudaStreamSynchronize(stream);
      if (admission.ok() && synchronize_error != cudaSuccess) {
        admission = cuda_failure("DeepSeek compact admission", synchronize_error);
      }
    }
    if (!admission.ok()) {
      static_cast<void>(cudaFreeAsync(raw, stream));
      static_cast<void>(cudaStreamSynchronize(stream));
      stream_lock.unlock();
      completion({admission, {}, 0});
      return operation;
    }
    auto allocation = std::make_shared<CudaExpertAllocation>(
        pool_, raw, total, gate, gate_scales, down, down_scales);
    stream_lock.unlock();
    completion({Status::success(), std::move(allocation), total});
    return operation;
  }
  const auto copy = [&](void* destination, std::uint64_t offset,
                        std::uint64_t bytes) {
    return cudaMemcpyAsync(destination, source + offset,
                           static_cast<std::size_t>(bytes),
                           cudaMemcpyHostToDevice, stream);
  };
  error = copy(gate, sections.gate_up_q_offset, sections.gate_up_q_bytes);
  if (error == cudaSuccess)
    error = copy(gate_scales, sections.gate_up_scale_offset,
                 sections.gate_up_scale_bytes);
  if (error == cudaSuccess)
    error = copy(down, sections.down_q_offset, sections.down_q_bytes);
  if (error == cudaSuccess)
    error = copy(down_scales, sections.down_scale_offset,
                 sections.down_scale_bytes);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {
    static_cast<void>(cudaFreeAsync(raw, stream));
    static_cast<void>(cudaStreamSynchronize(stream));
    stream_lock.unlock();
    completion({cuda_failure("expert H2D", error), {}, 0});
    return operation;
  }
  auto allocation = std::make_shared<CudaExpertAllocation>(
      pool_, raw, total, gate, gate_scales, down, down_scales);
  stream_lock.unlock();
  completion({Status::success(), std::move(allocation), total});
  return operation;
}

void CudaExpertUploader::cancel(OperationId) noexcept {
  // v1 upload is bounded and synchronous; cache abandonment is observed before
  // the next cache operation.
}

}  // namespace expert::runtime::cuda
