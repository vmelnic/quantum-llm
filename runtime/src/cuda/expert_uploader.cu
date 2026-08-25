#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cuda/deepseek_admission.hpp"
#include "expert/runtime/expert_record.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <iterator>
#include <list>
#include <limits>
#include <map>
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
  struct CompactEntry final {
    void* storage{};
    std::size_t bytes{};
    std::list<ExpertKey>::iterator lru;
  };

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
  std::uint64_t compact_cache_capacity_bytes{};
  bool persistent_staging{};
  bool direct_compact_execution{};
  std::unordered_map<std::size_t, std::vector<void*>> recycled;
  void* staging{};
  std::size_t staging_bytes{};
  std::uint64_t device_slot_bytes{};
  std::map<ExpertKey, CompactEntry> compact_cache;
  std::list<ExpertKey> compact_lru;
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
        device_slot_bytes + static_cast<std::uint64_t>(staging_bytes) +
            metrics.compact_cache_bytes);
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

  cudaError_t staging_locked(std::size_t bytes, void** output) {
    if (!persistent_staging) return cudaMallocAsync(output, bytes, stream);
    if (staging != nullptr && staging_bytes >= bytes) {
      *output = staging;
      return cudaSuccess;
    }
    if (staging != nullptr) {
      const auto error = cudaFreeAsync(staging, stream);
      if (error != cudaSuccess) return error;
      staging = nullptr;
      staging_bytes = 0U;
    }
    const auto error = cudaMallocAsync(&staging, bytes, stream);
    if (error == cudaSuccess) {
      staging_bytes = bytes;
      ++metrics.staging_allocations;
      update_device_high_water_locked();
      *output = staging;
    }
    return error;
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
      for (const auto& [_, entry] : compact_cache) {
        static_cast<void>(cudaFreeAsync(entry.storage, stream));
      }
      if (staging != nullptr) static_cast<void>(cudaFreeAsync(staging, stream));
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
    std::uint32_t intermediate, bool packed_fp4, bool relu2) noexcept
    : pool_(std::move(pool)), storage_(storage), bytes_(bytes), gate_up_(gate_up),
      gate_up_scales_(gate_up_scales), down_(down), down_scales_(down_scales),
      hidden_(hidden), intermediate_(intermediate), packed_fp4_(packed_fp4),
      relu2_(relu2) {}

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

CudaCompactExpertAllocation::CudaCompactExpertAllocation(
    std::shared_ptr<CudaExpertPool> pool, void* storage, std::size_t bytes,
    DeepSeekCompactSections sections) noexcept
    : pool_(std::move(pool)), storage_(storage), bytes_(bytes),
      sections_(sections) {}

CudaCompactExpertAllocation::~CudaCompactExpertAllocation() {
  if (storage_ != nullptr && pool_) pool_->release(storage_, bytes_);
}

std::size_t CudaCompactExpertAllocation::bytes() const noexcept {
  return bytes_;
}

const std::uint8_t* CudaCompactExpertAllocation::base() const noexcept {
  return static_cast<const std::uint8_t*>(storage_);
}

const DeepSeekCompactSections& CudaCompactExpertAllocation::sections()
    const noexcept {
  return sections_;
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
  pool_->compact_cache_capacity_bytes =
      options.compact_cache_capacity_bytes;
  pool_->persistent_staging = options.persistent_staging;
  pool_->direct_compact_execution = options.direct_compact_execution;
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
  if (request.key.encoding_abi == kExpertEncodingAbiFp4Block32 &&
      request.source_abi == kExpertSourceAbiDeepSeekCompactV1 &&
      pool_->direct_compact_execution) {
    const auto compact_bytes = request.complete_record.size();
    void* compact_raw = nullptr;
    auto error = pool_->acquire_locked(compact_bytes, &compact_raw);
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(compact_raw, request.complete_record.data(),
                              compact_bytes, cudaMemcpyHostToDevice, stream);
    }
    if (error != cudaSuccess) {
      if (compact_raw != nullptr)
        pool_->release_locked(compact_raw, compact_bytes);
      stream_lock.unlock();
      completion({cuda_failure("DeepSeek direct compact upload", error), {},
                  0U});
      return operation;
    }
    pool_->metrics.compact_h2d_bytes += compact_bytes;
    finish_async(std::make_shared<CudaCompactExpertAllocation>(
                     pool_, compact_raw, compact_bytes, request.compact),
                 compact_bytes, Status::success());
    return operation;
  }
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
  if (request.source_abi == kExpertSourceAbiDeepSeekCompactV1 ||
      request.source_abi == kExpertSourceAbiDeepSeekFp8Block128V1) {
    void* compact_raw = nullptr;
    bool compact_cached = false;
    const auto compact_bytes = request.complete_record.size();
    const bool compact_cache_eligible =
        request.source_abi == kExpertSourceAbiDeepSeekCompactV1 &&
        pool_->compact_cache_capacity_bytes >= compact_bytes;
    if (compact_cache_eligible) {
      const auto cached = pool_->compact_cache.find(request.key);
      if (cached != pool_->compact_cache.end()) {
        compact_raw = cached->second.storage;
        pool_->compact_lru.splice(pool_->compact_lru.end(),
                                  pool_->compact_lru,
                                  cached->second.lru);
        ++pool_->metrics.compact_cache_hits;
        compact_cached = true;
        error = cudaSuccess;
      } else {
        ++pool_->metrics.compact_cache_misses;
        while (pool_->metrics.compact_cache_bytes + compact_bytes >
               pool_->compact_cache_capacity_bytes) {
          if (pool_->compact_lru.empty()) break;
          const auto victim = pool_->compact_cache.find(
              pool_->compact_lru.front());
          if (victim == pool_->compact_cache.end()) break;
          error = cudaFreeAsync(victim->second.storage, stream);
          if (error != cudaSuccess) break;
          pool_->metrics.compact_cache_bytes -= victim->second.bytes;
          pool_->compact_lru.pop_front();
          pool_->compact_cache.erase(victim);
          ++pool_->metrics.compact_cache_evictions;
        }
        if (error == cudaSuccess &&
            pool_->metrics.compact_cache_bytes + compact_bytes <=
                pool_->compact_cache_capacity_bytes) {
          error = cudaMallocAsync(&compact_raw, compact_bytes, stream);
          if (error == cudaSuccess) {
            error = cudaMemcpyAsync(compact_raw, source, compact_bytes,
                                    cudaMemcpyHostToDevice, stream);
          }
          if (error == cudaSuccess) {
            pool_->compact_lru.push_back(request.key);
            pool_->compact_cache.emplace(
                request.key,
                CudaExpertPool::CompactEntry{
                    compact_raw, compact_bytes,
                    std::prev(pool_->compact_lru.end())});
            pool_->metrics.compact_cache_bytes += compact_bytes;
            pool_->metrics.compact_h2d_bytes += compact_bytes;
            pool_->metrics.compact_cache_high_water = std::max(
                pool_->metrics.compact_cache_high_water,
                pool_->metrics.compact_cache_bytes);
            pool_->update_device_high_water_locked();
            compact_cached = true;
          } else if (compact_raw != nullptr) {
            static_cast<void>(cudaFreeAsync(compact_raw, stream));
            compact_raw = nullptr;
          }
        }
      }
    } else {
      error = pool_->staging_locked(compact_bytes, &compact_raw);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(compact_raw, source, compact_bytes,
                                cudaMemcpyHostToDevice, stream);
      }
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
             gate, gate_scales, sections.intermediate, sections.hidden);
      launch(request.compact.w3_weight_offset, request.compact.w3_scale_offset,
             gate + static_cast<std::uint64_t>(sections.intermediate) *
                        sections.hidden,
             gate_scales + sections.intermediate, sections.intermediate,
             sections.hidden);
      launch(request.compact.w2_weight_offset, request.compact.w2_scale_offset,
             down, down_scales, sections.hidden, sections.intermediate);
    }
    if (compact_raw != nullptr) {
      if (!compact_cached && !pool_->persistent_staging) {
        const auto free_error = cudaFreeAsync(compact_raw, stream);
        if (admission.ok() && free_error != cudaSuccess) {
          admission = cuda_failure("DeepSeek compact release", free_error);
        }
      }
    }
    finish_async(std::make_shared<CudaExpertAllocation>(
                     pool_, raw, total, gate, gate_scales, down, down_scales),
                 total, std::move(admission));
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
  if (error != cudaSuccess) {
    pool_->release_locked(raw, total);
    stream_lock.unlock();
    completion({cuda_failure("expert H2D", error), {}, 0});
    return operation;
  }
  finish_async(std::make_shared<CudaExpertAllocation>(
                   pool_, raw, total, gate, gate_scales, down, down_scales,
                   request.key.encoding_abi == kExpertEncodingAbiFp4Block32
                       ? sections.hidden
                       : 0U,
                   request.key.encoding_abi == kExpertEncodingAbiFp4Block32
                       ? sections.intermediate
                       : 0U,
                   request.key.encoding_abi == kExpertEncodingAbiFp4Block32,
                   request.record_abi == kExpertRecordAbiFp4Relu2Block32),
               total, Status::success());
  return operation;
}

void CudaExpertUploader::cancel(OperationId) noexcept {
  // Uploads complete through the event queue shortly after enqueue; cache
  // abandonment is observed before the next cache operation, so there is no
  // mid-flight cancellation to perform here.
}

}  // namespace expert::runtime::cuda
