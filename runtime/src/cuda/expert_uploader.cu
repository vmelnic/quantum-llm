#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/expert_record.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace expert::runtime::cuda {
namespace {

Status cuda_failure(const char* operation, cudaError_t error) {
  return Status(ErrorCode::upload_failed,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

}  // namespace

struct CudaExpertPool final {
  // One upload handed off to the completion thread. The allocation keeps the
  // device slot alive until the consumer's callback has observed the event;
  // on a stream error the allocation is dropped before the callback runs,
  // which returns the slot to the pool through the usual release path.
  struct PendingCompletion final {
    cudaEvent_t event{};
    UploadCompletion completion;
    std::shared_ptr<IDeviceAllocation> allocation;
    std::uint64_t bytes{};
  };

  cudaStream_t stream{};
  mutable std::mutex mutex;
  std::uint64_t recycled_capacity_bytes{};
  std::unordered_map<std::size_t, std::vector<void*>> recycled;
  std::uint64_t device_slot_bytes{};
  CudaExpertUploaderTelemetry metrics;
  // Uploads complete through events instead of stream-wide host syncs: the
  // enqueueing thread only records an event, and this thread blocks on it and
  // then invokes the completion callback. One stream keeps H2D copies,
  // admission kernels and staging reuse stream-ordered, so no host-side
  // serialization is required for correctness.
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  std::deque<PendingCompletion> pending_completions;
  bool completion_stop{false};
  std::thread completion_thread;

  void update_device_high_water_locked() noexcept {
    metrics.device_bytes_high_water = std::max(
        metrics.device_bytes_high_water,
        device_slot_bytes);
  }

  cudaError_t acquire_locked(std::size_t bytes, void** output) {
    auto iterator = recycled.find(bytes);
    if (iterator != recycled.end() && !iterator->second.empty()) {
      *output = iterator->second.back();
      iterator->second.pop_back();
      metrics.recycled_bytes -= bytes;
      ++metrics.recycled_acquires;
      return cudaSuccess;
    }
    const auto error = cudaMallocAsync(output, bytes, stream);
    if (error == cudaSuccess) {
      ++metrics.device_allocations;
      device_slot_bytes += bytes;
      update_device_high_water_locked();
    }
    return error;
  }

  void release_locked(void* storage, std::size_t bytes) noexcept {
    if (storage == nullptr) return;
    if (metrics.recycled_bytes + bytes <= recycled_capacity_bytes) {
      recycled[bytes].push_back(storage);
      metrics.recycled_bytes += bytes;
      ++metrics.recycled_releases;
      return;
    }
    if (cudaFreeAsync(storage, stream) == cudaSuccess) {
      device_slot_bytes -= bytes;
      ++metrics.device_releases;
      return;
    }
    // Keep ownership if CUDA cannot enqueue the release; the pool destructor
    // will retry instead of silently losing the pointer.
    recycled[bytes].push_back(storage);
    metrics.recycled_bytes += bytes;
  }

  void release(void* storage, std::size_t bytes) noexcept {
    std::lock_guard lock(mutex);
    release_locked(storage, bytes);
  }

  ~CudaExpertPool() {
    {
      std::lock_guard lock(completion_mutex);
      completion_stop = true;
    }
    completion_cv.notify_one();
    if (completion_thread.joinable()) completion_thread.join();
    if (stream != nullptr) {
      for (const auto& [_, slots] : recycled) {
        for (auto* slot : slots) static_cast<void>(cudaFreeAsync(slot, stream));
      }
      static_cast<void>(cudaStreamSynchronize(stream));
      static_cast<void>(cudaStreamDestroy(stream));
    }
  }
};

namespace {

// Completion-thread body: drains queued uploads in enqueue order, blocking on
// each event instead of the whole stream. Runs on its own thread so neither
// the decode thread nor a storage callback ever pays a stream-wide sync.
void upload_completion_loop(CudaExpertPool* pool) {
  for (;;) {
    CudaExpertPool::PendingCompletion work;
    {
      std::unique_lock lock(pool->completion_mutex);
      pool->completion_cv.wait(lock, [&] {
        return pool->completion_stop || !pool->pending_completions.empty();
      });
      if (pool->pending_completions.empty()) {
        if (pool->completion_stop) return;
        continue;
      }
      work = std::move(pool->pending_completions.front());
      pool->pending_completions.pop_front();
    }
    const auto error = cudaEventSynchronize(work.event);
    static_cast<void>(cudaEventDestroy(work.event));
    if (error != cudaSuccess) {
      // Dropping the allocation returns the slot to the pool; the consumer
      // only observes the failure status.
      work.allocation.reset();
      work.completion({cuda_failure("expert upload stream", error), {}, 0U});
      continue;
    }
    const auto bytes = work.bytes;
    work.completion({Status::success(), std::move(work.allocation), bytes});
  }
}

}  // namespace

CudaExpertAllocation::CudaExpertAllocation(
    std::shared_ptr<CudaExpertPool> pool, void* storage, std::size_t bytes,
    const std::int8_t* gate_up,
    const float* gate_up_scales, const std::int8_t* down,
    const float* down_scales, std::uint32_t hidden,
    std::uint32_t intermediate, bool packed_fp4, bool relu2,
    bool native_nvfp4) noexcept
    : pool_(std::move(pool)), storage_(storage), bytes_(bytes), gate_up_(gate_up),
      gate_up_scales_(gate_up_scales), down_(down), down_scales_(down_scales),
      hidden_(hidden), intermediate_(intermediate), packed_fp4_(packed_fp4),
      relu2_(relu2), native_nvfp4_(native_nvfp4) {}

CudaExpertAllocation::~CudaExpertAllocation() {
  if (storage_ != nullptr && pool_) {
    pool_->release(storage_, bytes_);
  }
}

std::size_t CudaExpertAllocation::bytes() const noexcept { return bytes_; }
const std::int8_t* CudaExpertAllocation::gate_up() const noexcept { return gate_up_; }
const float* CudaExpertAllocation::gate_up_scales() const noexcept { return gate_up_scales_; }
const std::int8_t* CudaExpertAllocation::down() const noexcept { return down_; }
const float* CudaExpertAllocation::down_scales() const noexcept { return down_scales_; }
std::uint32_t CudaExpertAllocation::hidden() const noexcept { return hidden_; }
std::uint32_t CudaExpertAllocation::intermediate() const noexcept { return intermediate_; }
bool CudaExpertAllocation::packed_fp4() const noexcept { return packed_fp4_; }
bool CudaExpertAllocation::relu2() const noexcept { return relu2_; }
bool CudaExpertAllocation::native_nvfp4() const noexcept {
  return native_nvfp4_;
}

CudaExpertUploader::CudaExpertUploader(CudaExpertUploaderOptions options)
    : pool_(std::make_shared<CudaExpertPool>()) {
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
  pool_->recycled_capacity_bytes = options.recycled_capacity_bytes;
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
  pool_->completion_thread =
      std::thread(upload_completion_loop, pool_.get());
}

CudaExpertUploader::~CudaExpertUploader() = default;

CudaExpertUploaderTelemetry CudaExpertUploader::telemetry() const noexcept {
  std::lock_guard lock(pool_->mutex);
  return pool_->metrics;
}

OperationId CudaExpertUploader::upload(UploadRequest request,
                                       UploadCompletion completion) {
  const auto operation = next_operation_++;
  if (!completion) return operation;
  std::unique_lock stream_lock(pool_->mutex);
  const auto& sections = request.sections;
  const auto total = static_cast<std::size_t>(
      sections.gate_up_q_bytes + sections.gate_up_scale_bytes +
      sections.down_q_bytes + sections.down_scale_bytes);
  const auto stream = pool_->stream;
  // Hands a fully enqueued upload to the completion thread. Only event
  // publication and admission-launch failures still complete inline; stream
  // execution errors surface when the completion thread syncs the event.
  const auto finish_async = [&](std::shared_ptr<IDeviceAllocation> allocation,
                                std::uint64_t bytes, Status admission) {
    if (admission.ok()) {
      cudaEvent_t event{};
      auto event_error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (event_error == cudaSuccess)
        event_error = cudaEventRecord(event, stream);
      if (event_error == cudaSuccess) {
        {
          std::lock_guard lock(pool_->completion_mutex);
          pool_->pending_completions.push_back(
              CudaExpertPool::PendingCompletion{
                  event, std::move(completion), std::move(allocation), bytes});
        }
        pool_->completion_cv.notify_one();
        stream_lock.unlock();
        return;
      }
      if (event != nullptr) static_cast<void>(cudaEventDestroy(event));
      admission = cuda_failure("expert upload event", event_error);
    }
    stream_lock.unlock();
    // The allocation returns its slot to the pool as it goes out of scope,
    // after the pool lock has been dropped.
    allocation.reset();
    completion({admission, {}, 0});
  };
  void* raw = nullptr;
  auto error = pool_->acquire_locked(total, &raw);
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
  if (error != cudaSuccess) {
    pool_->release_locked(raw, total);
    stream_lock.unlock();
    completion({cuda_failure("expert H2D", error), {}, 0});
    return operation;
  }
  finish_async(std::make_shared<CudaExpertAllocation>(
                   pool_, raw, total, gate, gate_scales, down, down_scales,
                   (request.key.encoding_abi == kExpertEncodingAbiFp4Block32 ||
                    request.key.encoding_abi ==
                        kExpertEncodingAbiNvfp4Block16W4A4)
                       ? sections.hidden
                       : 0U,
                   (request.key.encoding_abi == kExpertEncodingAbiFp4Block32 ||
                    request.key.encoding_abi ==
                        kExpertEncodingAbiNvfp4Block16W4A4)
                       ? sections.intermediate
                       : 0U,
                   request.key.encoding_abi == kExpertEncodingAbiFp4Block32 ||
                       request.key.encoding_abi ==
                           kExpertEncodingAbiNvfp4Block16W4A4,
                   request.record_abi == kExpertRecordAbiFp4Relu2Block32,
                   request.record_abi ==
                       kExpertRecordAbiNvfp4Block16W4A4),
               total, Status::success());
  return operation;
}

void CudaExpertUploader::cancel(OperationId) noexcept {
  // Uploads complete through the event queue shortly after enqueue; cache
  // abandonment is observed before the next cache operation, so there is no
  // mid-flight cancellation to perform here.
}

}  // namespace expert::runtime::cuda
