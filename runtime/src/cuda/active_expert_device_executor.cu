#include "expert/runtime/cuda/active_expert_device_executor.hpp"

#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/resource_governor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarpsPerBlock = kThreads / 32U;

Status copy_status(const Status& status) {
  return {status.code(), std::string(status.message())};
}

Status cuda_status(cudaError_t error, std::string_view operation) {
  if (error == cudaSuccess) return Status::success();
  return {ErrorCode::upload_failed,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

class DeviceGuard final {
 public:
  explicit DeviceGuard(int target) {
    if (cudaGetDevice(&previous_) != cudaSuccess) previous_ = -1;
    error_ = cudaSetDevice(target);
  }
  ~DeviceGuard() {
    if (error_ == cudaSuccess && previous_ >= 0)
      static_cast<void>(cudaSetDevice(previous_));
  }
  [[nodiscard]] cudaError_t error() const noexcept { return error_; }

 private:
  int previous_{-1};
  cudaError_t error_{cudaSuccess};
};

class ActiveExpertAllocationPool final {
 public:
  ActiveExpertAllocationPool(int device, std::size_t slot_bytes)
      : device_(device), slot_bytes_(slot_bytes) {}

  ~ActiveExpertAllocationPool() {
    DeviceGuard guard(device_);
    if (guard.error() != cudaSuccess) return;
    if (arena_) static_cast<void>(cudaFree(arena_));
  }

  [[nodiscard]] std::pair<Status, std::size_t> initialize(
      std::size_t maximum_slots, std::size_t minimum_slots,
      std::uint64_t reserve_bytes) {
    std::lock_guard lock(mutex_);
    if (arena_ || maximum_slots == 0U || minimum_slots == 0U ||
        maximum_slots < minimum_slots || slot_bytes_ == 0U ||
        maximum_slots > std::numeric_limits<std::size_t>::max() / slot_bytes_) {
      return {{ErrorCode::invalid_argument,
               "secondary CUDA expert arena configuration is invalid"},
              0U};
    }
    DeviceGuard guard(device_);
    if (guard.error() != cudaSuccess)
      return {cuda_status(guard.error(), "select secondary CUDA device"),
              0U};

    auto slots = maximum_slots;
    while (slots >= minimum_slots) {
      std::size_t free_bytes{}, total_bytes{};
      auto error = cudaMemGetInfo(&free_bytes, &total_bytes);
      if (error != cudaSuccess)
        return {cuda_status(error, "inspect secondary CUDA arena capacity"),
                0U};
      if (free_bytes <= reserve_bytes) break;
      const auto available_slots =
          static_cast<std::size_t>((free_bytes - reserve_bytes) / slot_bytes_);
      slots = std::min(slots, available_slots);
      if (slots < minimum_slots) break;

      const auto arena_bytes = slots * slot_bytes_;
      void* candidate{};
      error = cudaMalloc(&candidate, arena_bytes);
      if (error != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        --slots;
        continue;
      }
      std::size_t remaining_bytes{}, remaining_total{};
      error = cudaMemGetInfo(&remaining_bytes, &remaining_total);
      if (error == cudaSuccess && remaining_bytes >= reserve_bytes) {
        arena_ = candidate;
        free_.reserve(slots);
        auto* base = static_cast<std::byte*>(arena_);
        for (std::size_t index = 0U; index < slots; ++index)
          free_.push_back(base + index * slot_bytes_);
        return {Status::success(), slots};
      }
      static_cast<void>(cudaFree(candidate));
      if (error != cudaSuccess)
        return {cuda_status(error, "verify secondary CUDA arena reserve"),
                0U};
      const auto deficit = reserve_bytes - remaining_bytes;
      const auto slots_to_remove = static_cast<std::size_t>(
          std::max<std::uint64_t>(
              1U, (deficit + slot_bytes_ - 1U) / slot_bytes_));
      if (slots_to_remove > slots - minimum_slots) break;
      slots -= slots_to_remove;
    }
    return {{ErrorCode::backpressure,
             "secondary CUDA expert arena cannot preserve the device reserve"},
            0U};
  }

  [[nodiscard]] std::pair<Status, void*> acquire(std::size_t bytes) {
    if (bytes != slot_bytes_)
      return {{ErrorCode::invalid_argument,
               "secondary CUDA expert allocation has an invalid size"},
              nullptr};
    std::lock_guard lock(mutex_);
    if (!free_.empty()) {
      auto* result = free_.back();
      free_.pop_back();
      return {Status::success(), result};
    }
    return {{ErrorCode::backpressure,
             "secondary CUDA expert arena has no recyclable slot"},
            nullptr};
  }

  void release(void* pointer) noexcept {
    if (!pointer) return;
    std::lock_guard lock(mutex_);
    free_.push_back(pointer);
  }

  [[nodiscard]] int device() const noexcept { return device_; }

 private:
  int device_{};
  std::size_t slot_bytes_{};
  void* arena_{};
  std::mutex mutex_;
  std::vector<void*> free_;
};

class ActiveExpertPackedAllocation final : public IDeviceAllocation {
 public:
  ActiveExpertPackedAllocation(std::shared_ptr<ActiveExpertAllocationPool> pool,
                         void* pointer, std::size_t bytes,
                         SplitExpertSections sections) noexcept
      : pool_(std::move(pool)), pointer_(pointer), bytes_(bytes),
        sections_(sections) {}
  ~ActiveExpertPackedAllocation() override {
    if (pool_ && pointer_) pool_->release(pointer_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept override { return bytes_; }
  [[nodiscard]] const std::uint8_t* data() const noexcept {
    return static_cast<const std::uint8_t*>(pointer_);
  }
  [[nodiscard]] const SplitExpertSections& sections() const noexcept {
    return sections_;
  }

 private:
  std::shared_ptr<ActiveExpertAllocationPool> pool_;
  void* pointer_{};
  std::size_t bytes_{};
  SplitExpertSections sections_{};
};

class ActiveExpertPackedUploader final : public IDeviceUploader {
 public:
  ActiveExpertPackedUploader(std::shared_ptr<ActiveExpertAllocationPool> pool,
                       std::size_t source_record_bytes,
                       std::size_t device_record_bytes,
                       std::uint32_t hidden, std::uint32_t intermediate)
      : pool_(std::move(pool)),
        source_record_bytes_(source_record_bytes),
        device_record_bytes_(device_record_bytes),
        matrix_values_(static_cast<std::uint64_t>(hidden) * intermediate) {
    DeviceGuard guard(pool_->device());
    if (guard.error() != cudaSuccess)
      throw std::runtime_error(
          std::string(cudaGetErrorString(guard.error())));
    auto error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error == cudaSuccess)
      error = cudaHostAlloc(&upload_staging_, device_record_bytes_,
                            cudaHostAllocPortable);
    if (error != cudaSuccess) {
      if (upload_staging_) static_cast<void>(cudaFreeHost(upload_staging_));
      upload_staging_ = nullptr;
      if (stream_) static_cast<void>(cudaStreamDestroy(stream_));
      stream_ = nullptr;
      throw std::runtime_error(std::string("initialize secondary uploader: ") +
                               cudaGetErrorString(error));
    }
  }

  ~ActiveExpertPackedUploader() override {
    DeviceGuard guard(pool_->device());
    if (guard.error() == cudaSuccess && stream_)
      static_cast<void>(cudaStreamDestroy(stream_));
    if (upload_staging_) static_cast<void>(cudaFreeHost(upload_staging_));
  }

  OperationId upload(UploadRequest request,
                     UploadCompletion completion) override {
    const auto operation = next_.fetch_add(1U, std::memory_order_relaxed);
    if (!completion) return operation;
    std::lock_guard lock(mutex_);
    const auto in_range = [&](std::uint64_t offset, std::uint64_t bytes) {
      return offset <= source_record_bytes_ &&
             bytes <= source_record_bytes_ - offset;
    };
    const auto expected_weight_bytes = matrix_values_ / 2U;
    const auto expected_scale_bytes = matrix_values_ / kExpertFp4BlockSize;
    const auto expected_device_bytes =
        3U * (expected_weight_bytes + expected_scale_bytes);
    if (request.complete_record.size() != source_record_bytes_ ||
        device_record_bytes_ != expected_device_bytes ||
        request.key.encoding_abi != kExpertEncodingAbiFp4Block32 ||
        request.compact.w1_weight_bytes != expected_weight_bytes ||
        request.compact.w1_scale_bytes != expected_scale_bytes ||
        request.compact.w3_weight_bytes != expected_weight_bytes ||
        request.compact.w3_scale_bytes != expected_scale_bytes ||
        request.compact.w2_weight_bytes != expected_weight_bytes ||
        request.compact.w2_scale_bytes != expected_scale_bytes ||
        !in_range(request.compact.w1_weight_offset,
                  request.compact.w1_weight_bytes) ||
        !in_range(request.compact.w1_scale_offset,
                  request.compact.w1_scale_bytes) ||
        !in_range(request.compact.w3_weight_offset,
                  request.compact.w3_weight_bytes) ||
        !in_range(request.compact.w3_scale_offset,
                  request.compact.w3_scale_bytes) ||
        !in_range(request.compact.w2_weight_offset,
                  request.compact.w2_weight_bytes) ||
        !in_range(request.compact.w2_scale_offset,
                  request.compact.w2_scale_bytes)) {
      completion({{ErrorCode::invalid_argument,
                   "secondary CUDA uploader rejected the expert ABI"},
                  {}, 0U});
      return operation;
    }
    SplitExpertSections compact;
    compact.w1_weight_offset = 0U;
    compact.w1_weight_bytes = expected_weight_bytes;
    compact.w1_scale_offset = compact.w1_weight_bytes;
    compact.w1_scale_bytes = expected_scale_bytes;
    compact.w3_weight_offset =
        compact.w1_scale_offset + compact.w1_scale_bytes;
    compact.w3_weight_bytes = expected_weight_bytes;
    compact.w3_scale_offset =
        compact.w3_weight_offset + compact.w3_weight_bytes;
    compact.w3_scale_bytes = expected_scale_bytes;
    compact.w2_weight_offset =
        compact.w3_scale_offset + compact.w3_scale_bytes;
    compact.w2_weight_bytes = expected_weight_bytes;
    compact.w2_scale_offset =
        compact.w2_weight_offset + compact.w2_weight_bytes;
    compact.w2_scale_bytes = expected_scale_bytes;
    const auto copy_section = [&](std::uint64_t source_offset,
                                  std::uint64_t bytes,
                                  std::uint64_t destination_offset) {
      std::memcpy(static_cast<std::byte*>(upload_staging_) +
                      destination_offset,
                  request.complete_record.data() + source_offset,
                  static_cast<std::size_t>(bytes));
    };
    copy_section(request.compact.w1_weight_offset,
                 request.compact.w1_weight_bytes,
                 compact.w1_weight_offset);
    copy_section(request.compact.w1_scale_offset,
                 request.compact.w1_scale_bytes,
                 compact.w1_scale_offset);
    copy_section(request.compact.w3_weight_offset,
                 request.compact.w3_weight_bytes,
                 compact.w3_weight_offset);
    copy_section(request.compact.w3_scale_offset,
                 request.compact.w3_scale_bytes,
                 compact.w3_scale_offset);
    copy_section(request.compact.w2_weight_offset,
                 request.compact.w2_weight_bytes,
                 compact.w2_weight_offset);
    copy_section(request.compact.w2_scale_offset,
                 request.compact.w2_scale_bytes,
                 compact.w2_scale_offset);

    auto [allocation_status, pointer] = pool_->acquire(device_record_bytes_);
    if (!allocation_status.ok()) {
      completion({std::move(allocation_status), {}, 0U});
      return operation;
    }
    auto allocation = std::make_shared<ActiveExpertPackedAllocation>(
        pool_, pointer, device_record_bytes_, compact);
    DeviceGuard guard(pool_->device());
    auto error = guard.error();
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(pointer, upload_staging_, device_record_bytes_,
                              cudaMemcpyHostToDevice, stream_);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      allocation.reset();
      completion({cuda_status(error, "upload secondary CUDA expert page"),
                  {}, 0U});
      return operation;
    }
    completion(
        {Status::success(), std::move(allocation), device_record_bytes_});
    return operation;
  }

  void cancel(OperationId) noexcept override {}

 private:
  std::shared_ptr<ActiveExpertAllocationPool> pool_;
  std::size_t source_record_bytes_{};
  std::size_t device_record_bytes_{};
  std::uint64_t matrix_values_{};
  void* upload_staging_{};
  cudaStream_t stream_{};
  std::mutex mutex_;
  std::atomic<OperationId> next_{1U};
};

__device__ std::int8_t decode_fp4_twice(std::uint8_t code) {
  const auto index = code & 0x07U;
  const auto magnitude = index <= 4U ? static_cast<int>(index)
                         : index == 5U ? 6
                         : index == 6U ? 8
                                       : 12;
  return static_cast<std::int8_t>((code & 0x08U) ? -magnitude : magnitude);
}

__device__ float decode_ue8m0(std::uint8_t code) {
  return code == 0U ? __uint_as_float(0x00400000U)
                    : __uint_as_float(static_cast<unsigned>(code) << 23U);
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset != 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__device__ float round_bf16(float value) {
  auto bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) return value;
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return __uint_as_float(bits & 0xffff0000U);
}

__global__ void quantize_q8(const float* input, std::int8_t* output,
                            float* scales, std::uint32_t rows,
                            std::uint32_t columns) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= rows) return;
  const auto* source = input + static_cast<std::size_t>(row) * columns;
  auto* target = output + static_cast<std::size_t>(row) * columns;
  __shared__ float maxima[kThreads];
  float maximum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(source[column]));
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const auto scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) scales[row] = scale;
  __syncthreads();
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    auto value = __float2int_rn(source[column] / scale);
    value = max(-127, min(127, value));
    target[column] = static_cast<std::int8_t>(value);
  }
}

__device__ float packed_dot(const std::uint8_t* weights,
                            const std::uint8_t* scales,
                            const std::int8_t* activation,
                            float activation_scale, std::uint32_t row,
                            std::uint32_t columns) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto blocks = columns / 32U;
  const auto* row_weights = reinterpret_cast<const uint4*>(
      weights + static_cast<std::size_t>(row) * (columns / 2U));
  const auto* row_scales = scales + static_cast<std::size_t>(row) * blocks;
  float total = 0.0F;
  for (std::uint32_t block = lane; block < blocks; block += 32U) {
    const auto packed = row_weights[block];
    const std::uint32_t words[]{packed.x, packed.y, packed.z, packed.w};
    const auto* q = activation + static_cast<std::size_t>(block) * 32U;
    int block_total = 0;
#pragma unroll
    for (std::uint32_t byte = 0U; byte < 16U; ++byte) {
      const auto packed_weight = static_cast<std::uint8_t>(
          words[byte / 4U] >> (8U * (byte & 3U)));
      block_total +=
          static_cast<int>(decode_fp4_twice(packed_weight & 0x0fU)) *
              static_cast<int>(q[2U * byte]) +
          static_cast<int>(decode_fp4_twice(packed_weight >> 4U)) *
              static_cast<int>(q[2U * byte + 1U]);
    }
    total += static_cast<float>(block_total) * decode_ue8m0(row_scales[block]);
  }
  return warp_sum(total) * activation_scale * 0.5F;
}

__global__ void gate_up(const std::uint8_t* const* records,
                        const SplitExpertSections* sections,
                        const std::int8_t* input, const float* input_scale,
                        const std::uint32_t* input_rows,
                        float* intermediate, std::uint32_t hidden,
                        std::uint32_t width, std::uint32_t selections,
                        float clamp,
                        bool bf16_rounding) {
  const auto selection = static_cast<std::uint32_t>(blockIdx.y);
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / 32U;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto row = static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock +
                   warp;
  if (selection >= selections || row >= width) return;
  const auto* record = records[selection];
  const auto layout = sections[selection];
  const auto input_row = input_rows[selection];
  const auto* selected_input =
      input + static_cast<std::size_t>(input_row) * hidden;
  const auto gate_value = packed_dot(record + layout.w1_weight_offset,
                                     record + layout.w1_scale_offset,
                                     selected_input, input_scale[input_row],
                                     row, hidden);
  const auto up_value = packed_dot(record + layout.w3_weight_offset,
                                   record + layout.w3_scale_offset,
                                   selected_input, input_scale[input_row], row,
                                   hidden);
  if (lane == 0U) {
    const auto bounded_gate = clamp > 0.0F ? fminf(gate_value, clamp)
                                           : gate_value;
    const auto bounded_up = clamp > 0.0F
                                ? fminf(fmaxf(up_value, -clamp), clamp)
                                : up_value;
    auto value =
        (bounded_gate / (1.0F + expf(-bounded_gate))) * bounded_up;
    intermediate[static_cast<std::size_t>(selection) * width + row] =
        bf16_rounding ? round_bf16(value) : value;
  }
}

__global__ void down(const std::uint8_t* const* records,
                     const SplitExpertSections* sections,
                     const std::int8_t* intermediate,
                     const float* intermediate_scale, float* output,
                     std::uint32_t hidden, std::uint32_t width,
                     std::uint32_t selections) {
  const auto selection = static_cast<std::uint32_t>(blockIdx.y);
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / 32U;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto row = static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock +
                   warp;
  if (selection >= selections || row >= hidden) return;
  const auto* record = records[selection];
  const auto layout = sections[selection];
  const auto* selected_intermediate =
      intermediate + static_cast<std::size_t>(selection) * width;
  const auto value = packed_dot(
      record + layout.w2_weight_offset, record + layout.w2_scale_offset,
      selected_intermediate,
      intermediate_scale[selection], row, width);
  if (lane == 0U)
    output[static_cast<std::size_t>(selection) * hidden + row] = value;
}

struct AtomicTelemetry final {
  std::atomic<std::uint64_t> requests{0U};
  std::atomic<std::uint64_t> completed{0U};
  std::atomic<std::uint64_t> failed{0U};
  std::atomic<std::uint64_t> cancelled{0U};
  std::atomic<std::uint64_t> input_bytes{0U};
  std::atomic<std::uint64_t> output_bytes{0U};
  std::atomic<std::uint64_t> storage_bytes{0U};
  std::atomic<std::uint64_t> ram_bytes{0U};
  std::atomic<std::uint64_t> vram_bytes{0U};
  std::atomic<std::uint64_t> execution_ns{0U};

  [[nodiscard]] ActiveExpertExecutorTelemetry snapshot() const noexcept {
    ActiveExpertExecutorTelemetry result;
    result.requests = requests.load(std::memory_order_relaxed);
    result.completed = completed.load(std::memory_order_relaxed);
    result.failed = failed.load(std::memory_order_relaxed);
    result.cancelled = cancelled.load(std::memory_order_relaxed);
    result.activation_input_bytes =
        input_bytes.load(std::memory_order_relaxed);
    result.activation_output_bytes =
        output_bytes.load(std::memory_order_relaxed);
    result.owner_weight_read_bytes = result.completed;
    result.owner_storage_read_bytes =
        storage_bytes.load(std::memory_order_relaxed);
    result.owner_ram_read_bytes = ram_bytes.load(std::memory_order_relaxed);
    result.owner_vram_read_bytes = vram_bytes.load(std::memory_order_relaxed);
    result.owner_execution_ns = execution_ns.load(std::memory_order_relaxed);
    return result;
  }
};

struct Task final {
  explicit Task(ActiveExpertExecutionRequest value)
      : request(std::move(value)) {}
  ActiveExpertExecutionRequest request;
  std::atomic<bool> cancelled{false};
  std::mutex mutex;
  std::optional<ActiveExpertExecutionResult> result;
};

ActiveExpertExecutionResult failed_result(
    const ActiveExpertExecutionRequest& request, Status status) {
  ActiveExpertExecutionResult result;
  result.status = std::move(status);
  result.identity = request.identity;
  result.request_id = request.invocation.request_id;
  result.invocation_id = request.invocation.invocation_id;
  result.selection_index = request.invocation.selection_index;
  result.evidence.activation_input_bytes = request.invocation.input.bytes;
  return result;
}

class DeviceState final {
 public:
  DeviceState(int ordinal, const ActiveExpertDeviceExecutorConfig& config,
              std::shared_ptr<IAsyncStorage> storage,
              std::vector<PayloadRecord> records,
              AtomicTelemetry& telemetry)
      : ordinal_(ordinal), config_(config), records_(std::move(records)),
        telemetry_(telemetry) {
    DeviceGuard guard(ordinal_);
    if (guard.error() != cudaSuccess)
      throw std::runtime_error("cannot select secondary CUDA device");
    cudaDeviceProp properties{};
    auto error = cudaGetDeviceProperties(&properties, ordinal_);
    if (error != cudaSuccess || properties.major != 8 ||
        properties.minor < 6)
      throw std::runtime_error(
          "active expert device must support SM86 kernels");
    const auto source_record_bytes = records_.front().stored_bytes;
    const auto device_record_bytes = records_.front().device_bytes == 0U
                                         ? source_record_bytes
                                         : records_.front().device_bytes;
    const auto hidden_bytes =
        static_cast<std::size_t>(config_.component.hidden_size) *
        sizeof(float);
    const auto width_bytes =
        static_cast<std::size_t>(config_.component.intermediate_size) *
        sizeof(float);
    maximum_batch_ = config_.staging_slots_per_device;
    pool_ = std::make_shared<ActiveExpertAllocationPool>(
        ordinal_, static_cast<std::size_t>(device_record_bytes));
    uploader_ = std::make_shared<ActiveExpertPackedUploader>(
        pool_, static_cast<std::size_t>(source_record_bytes),
        static_cast<std::size_t>(device_record_bytes),
        config_.component.hidden_size,
        config_.component.intermediate_size);
    buffers_ = std::make_shared<FixedBufferPool>(
        config_.staging_slots_per_device,
        static_cast<std::size_t>(source_record_bytes), kExpertPackAlignment,
        std::make_shared<CudaPinnedAllocator>(),
        config_.staging_slots_per_device);
    output_buffers_ = std::make_shared<FixedBufferPool>(
        maximum_batch_, maximum_batch_ * hidden_bytes, kExpertPackAlignment,
        std::make_shared<CudaPinnedAllocator>(), maximum_batch_);

    error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error == cudaSuccess)
      error = cudaMalloc(&record_table_, maximum_batch_ * sizeof(void*));
    if (error == cudaSuccess)
      error = cudaMalloc(&input_row_table_,
                         maximum_batch_ * sizeof(std::uint32_t));
    if (error == cudaSuccess)
      error = cudaMalloc(&section_table_,
                         maximum_batch_ * sizeof(SplitExpertSections));
    if (error == cudaSuccess)
      error = cudaMalloc(&input_, maximum_batch_ * hidden_bytes);
    if (error == cudaSuccess)
      error = cudaMalloc(&q_input_, maximum_batch_ * config_.component.hidden_size);
    if (error == cudaSuccess)
      error = cudaMalloc(&input_scale_, maximum_batch_ * sizeof(float));
    if (error == cudaSuccess)
      error = cudaMalloc(&intermediate_, maximum_batch_ * width_bytes);
    if (error == cudaSuccess)
      error = cudaMalloc(&q_intermediate_, maximum_batch_ *
                         config_.component.intermediate_size);
    if (error == cudaSuccess)
      error = cudaMalloc(&intermediate_scale_,
                         maximum_batch_ * sizeof(float));
    if (error == cudaSuccess)
      error = cudaMalloc(&output_, maximum_batch_ * hidden_bytes);
    if (error == cudaSuccess) {
      quantize_q8<<<1U, kThreads, 0U, stream_>>>(
          static_cast<const float*>(input_),
          static_cast<std::int8_t*>(q_input_),
          static_cast<float*>(input_scale_), 0U,
          config_.component.hidden_size);
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) {
      gate_up<<<dim3(1U, 1U), kThreads, 0U, stream_>>>(
          reinterpret_cast<const std::uint8_t* const*>(record_table_),
          static_cast<const SplitExpertSections*>(section_table_),
          static_cast<const std::int8_t*>(q_input_),
          static_cast<const float*>(input_scale_),
          static_cast<const std::uint32_t*>(input_row_table_),
          static_cast<float*>(intermediate_), config_.component.hidden_size,
          config_.component.intermediate_size, 0U,
          config_.activation_clamp, config_.round_intermediate_to_bf16);
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) {
      down<<<dim3(1U, 1U), kThreads, 0U, stream_>>>(
          reinterpret_cast<const std::uint8_t* const*>(record_table_),
          static_cast<const SplitExpertSections*>(section_table_),
          static_cast<const std::int8_t*>(q_intermediate_),
          static_cast<const float*>(intermediate_scale_),
          static_cast<float*>(output_), config_.component.hidden_size,
          config_.component.intermediate_size, 0U);
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      release_device_workspace();
      throw std::runtime_error(std::string("initialize expert workspace: ") +
                               cudaGetErrorString(error));
    }

    // Query only after the complete executor module has been launched and all
    // permanent workspace allocations exist. CUDA lazy module loading can
    // otherwise make a nominal cache ceiling consume the emergency reserve on
    // the first real expert invocation.
    std::size_t free_bytes{}, total_bytes{};
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
      release_device_workspace();
      throw std::runtime_error(std::string("inspect secondary CUDA memory: ") +
                               cudaGetErrorString(error));
    }
    const auto fitted_cache = fit_device_cache_budget(
        {static_cast<std::uint64_t>(free_bytes), 0U, 0U,
         config_.device_reserve_bytes_per_device,
         config_.device_cache_bytes_per_device, 2U * device_record_bytes});
    if (!fitted_cache.status.ok()) {
      release_device_workspace();
      throw std::runtime_error(
          std::string("secondary CUDA expert cache fit failed: ") +
          std::string(fitted_cache.status.message()));
    }
    const auto maximum_slots = static_cast<std::size_t>(
        fitted_cache.effective_cache_bytes / device_record_bytes);
    const auto minimum_slots = static_cast<std::size_t>(
        std::max<std::uint64_t>(2U, config_.component.route_width));
    const auto [arena_status, arena_slots] = pool_->initialize(
        maximum_slots, minimum_slots,
        config_.device_reserve_bytes_per_device);
    if (!arena_status.ok()) {
      release_device_workspace();
      throw std::runtime_error(
          std::string("secondary CUDA expert arena initialization failed: ") +
          std::string(arena_status.message()));
    }
    const auto host_capacity = config_.host_cache_bytes_total /
                               config_.device_ordinals.size();
    const auto device_capacity =
        static_cast<std::uint64_t>(arena_slots) * device_record_bytes;
    const auto host_eviction_window =
        static_cast<std::uint64_t>(config_.component.route_width) *
        source_record_bytes;
    const auto device_eviction_window =
        static_cast<std::uint64_t>(config_.component.route_width) *
        device_record_bytes;
    ExpertCacheConfig cache_config;
    cache_config.ram = {
        host_capacity, host_capacity,
        host_capacity > host_eviction_window
            ? host_capacity - host_eviction_window
            : source_record_bytes};
    cache_config.vram = {
        device_capacity, device_capacity,
        device_capacity > device_eviction_window
            ? device_capacity - device_eviction_window
            : device_record_bytes};
    cache_config.retain_host_copy = true;
    cache_config.trusted_immutable_source = true;
    cache_config.ram_retention_minimum_frequency = 0U;
    cache_ = std::make_unique<ExpertCache>(
        cache_config, std::move(storage), uploader_, buffers_);
    thread_ = std::thread([this] { loop(); });
  }

  ~DeviceState() {
    {
      std::lock_guard lock(queue_mutex_);
      stop_ = true;
    }
    queue_ready_.notify_one();
    if (thread_.joinable()) thread_.join();
    cache_.reset();
    output_buffers_.reset();
    buffers_.reset();
    uploader_.reset();
    pool_.reset();
    DeviceGuard guard(ordinal_);
    if (guard.error() != cudaSuccess) return;
    release_device_workspace();
  }

  void submit(std::shared_ptr<Task> task) {
    {
      std::lock_guard lock(queue_mutex_);
      queue_.push_back(std::move(task));
    }
    queue_ready_.notify_one();
  }

 private:
  void release_device_workspace() noexcept {
    if (output_) static_cast<void>(cudaFree(output_));
    output_ = nullptr;
    if (intermediate_scale_) static_cast<void>(cudaFree(intermediate_scale_));
    intermediate_scale_ = nullptr;
    if (q_intermediate_) static_cast<void>(cudaFree(q_intermediate_));
    q_intermediate_ = nullptr;
    if (intermediate_) static_cast<void>(cudaFree(intermediate_));
    intermediate_ = nullptr;
    if (input_scale_) static_cast<void>(cudaFree(input_scale_));
    input_scale_ = nullptr;
    if (q_input_) static_cast<void>(cudaFree(q_input_));
    q_input_ = nullptr;
    if (input_) static_cast<void>(cudaFree(input_));
    input_ = nullptr;
    if (input_row_table_) static_cast<void>(cudaFree(input_row_table_));
    input_row_table_ = nullptr;
    if (section_table_) static_cast<void>(cudaFree(section_table_));
    section_table_ = nullptr;
    if (record_table_) static_cast<void>(cudaFree(record_table_));
    record_table_ = nullptr;
    if (stream_) static_cast<void>(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }

  void complete(const std::shared_ptr<Task>& task,
                ActiveExpertExecutionResult result) {
    if (task->cancelled.load(std::memory_order_acquire)) return;
    std::lock_guard lock(task->mutex);
    if (!task->result) task->result = std::move(result);
  }

  void run_batch(std::vector<std::shared_ptr<Task>> tasks) {
    enum class AcquisitionTier : std::uint8_t { storage, ram, vram };

    struct WorkItem final {
      std::shared_ptr<Task> task;
      const PayloadRecord* record{};
      AcquireHandle acquire;
      ExpertLease lease;
      const ActiveExpertPackedAllocation* allocation{};
      std::uint32_t input_row{};
      bool owns_input_transfer{};
      bool acquired{};
      AcquisitionTier acquisition_tier{AcquisitionTier::storage};
    };

    const auto started = std::chrono::steady_clock::now();
    std::vector<WorkItem> work;
    work.reserve(tasks.size());
    for (auto& task : tasks) {
      if (task->cancelled.load(std::memory_order_acquire)) continue;
      const auto& request = task->request;
      const auto index =
          static_cast<std::size_t>(request.identity.key.layer) *
              config_.component.experts_per_layer +
          request.identity.key.expert;
      const auto& record = records_.at(index);
      const auto before = cache_->telemetry();
      auto acquire = cache_->acquire(
          request.identity.key, record,
          ExpertAcquireOptions{ExpertRequestPriority::demand, true, true,
                               true, false});
      const auto after = cache_->telemetry();
      if (!acquire.valid()) {
        telemetry_.failed.fetch_add(1U, std::memory_order_relaxed);
        complete(task, failed_result(
                           request,
                           {ErrorCode::backpressure,
                            "secondary CUDA expert cache rejected demand"}));
        continue;
      }
      const auto vram_hits =
          after.acquire_vram_hits - before.acquire_vram_hits;
      const auto ram_hits = after.acquire_ram_hits - before.acquire_ram_hits;
      const auto ssd_misses =
          after.acquire_ssd_misses - before.acquire_ssd_misses;
      if (vram_hits + ram_hits + ssd_misses != 1U) {
        acquire.cancel();
        telemetry_.failed.fetch_add(1U, std::memory_order_relaxed);
        complete(task, failed_result(
                           request,
                           {ErrorCode::internal,
                            "secondary CUDA cache did not classify demand"}));
        continue;
      }
      const auto acquisition_tier =
          vram_hits != 0U
              ? AcquisitionTier::vram
              : (ram_hits != 0U ? AcquisitionTier::ram
                                : AcquisitionTier::storage);
      work.push_back({task, &record, std::move(acquire), {}, nullptr, 0U,
                      false, false, acquisition_tier});
    }

    std::size_t unresolved = work.size();
    while (unresolved != 0U) {
      bool progressed = false;
      for (std::size_t index = 0U; index < work.size();) {
        auto& item = work[index];
        if (item.acquired) {
          ++index;
          continue;
        }
        const auto& request = item.task->request;
        if (item.task->cancelled.load(std::memory_order_acquire)) {
          item.acquire.cancel();
          work.erase(work.begin() + static_cast<std::ptrdiff_t>(index));
          --unresolved;
          progressed = true;
          continue;
        }
        if (request.invocation.deadline !=
                std::chrono::steady_clock::time_point::max() &&
            request.invocation.deadline <= std::chrono::steady_clock::now()) {
          item.acquire.cancel();
          telemetry_.failed.fetch_add(1U, std::memory_order_relaxed);
          complete(item.task,
                   failed_result(request,
                                 {ErrorCode::deadline_exceeded,
                                  "secondary CUDA expert deadline expired"}));
          work.erase(work.begin() + static_cast<std::ptrdiff_t>(index));
          --unresolved;
          progressed = true;
          continue;
        }
        if (item.acquire.wait_for(std::chrono::milliseconds(0)) !=
            std::future_status::ready) {
          ++index;
          continue;
        }
        auto acquired = item.acquire.get();
        if (!acquired.status.ok() || !acquired.lease) {
          telemetry_.failed.fetch_add(1U, std::memory_order_relaxed);
          complete(item.task,
                   failed_result(
                       request,
                       acquired.status.ok()
                           ? Status{ErrorCode::internal,
                                    "secondary CUDA expert lease is absent"}
                           : copy_status(acquired.status)));
          work.erase(work.begin() + static_cast<std::ptrdiff_t>(index));
          --unresolved;
          progressed = true;
          continue;
        }
        item.lease = std::move(acquired.lease);
        item.allocation =
            dynamic_cast<const ActiveExpertPackedAllocation*>(item.lease.get());
        if (!item.allocation) {
          telemetry_.failed.fetch_add(1U, std::memory_order_relaxed);
          complete(item.task,
                   failed_result(
                       request,
                       {ErrorCode::internal,
                        "secondary CUDA expert allocation has the wrong ABI"}));
          work.erase(work.begin() + static_cast<std::ptrdiff_t>(index));
          --unresolved;
          progressed = true;
          continue;
        }
        item.acquired = true;
        --unresolved;
        ++index;
        progressed = true;
      }
      if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (work.empty()) return;

    const auto hidden = config_.component.hidden_size;
    const auto width = config_.component.intermediate_size;
    std::vector<const std::uint8_t*> record_pointers;
    record_pointers.reserve(work.size());
    std::vector<const std::byte*> unique_inputs;
    unique_inputs.reserve(work.size());
    std::vector<std::uint32_t> input_rows;
    input_rows.reserve(work.size());
    std::vector<SplitExpertSections> sections;
    sections.reserve(work.size());
    const auto output_bytes =
        static_cast<std::size_t>(work.front().task->request.invocation.output_bytes);
    auto output_lease = output_buffers_->try_acquire(work.size() * output_bytes);
    if (!output_lease) {
      telemetry_.failed.fetch_add(work.size(), std::memory_order_relaxed);
      for (auto& item : work)
        complete(item.task,
                 failed_result(item.task->request,
                               {ErrorCode::backpressure,
                                "secondary CUDA output pool exhausted"}));
      return;
    }
    const auto output_buffer = output_lease->buffer();
    auto error = cudaSuccess;
    for (std::size_t index = 0U; index < work.size(); ++index) {
      auto& item = work[index];
      record_pointers.push_back(item.allocation->data());
      sections.push_back(item.allocation->sections());
      const auto* input = item.task->request.invocation.input.data;
      const auto found = std::find(unique_inputs.begin(), unique_inputs.end(),
                                   input);
      if (found == unique_inputs.end()) {
        item.input_row = static_cast<std::uint32_t>(unique_inputs.size());
        item.owns_input_transfer = true;
        unique_inputs.push_back(input);
        error = cudaMemcpyAsync(
            static_cast<float*>(input_) +
                static_cast<std::size_t>(item.input_row) * hidden,
            input,
            static_cast<std::size_t>(item.task->request.invocation.input.bytes),
            cudaMemcpyHostToDevice, stream_);
        if (error != cudaSuccess) break;
      } else {
        item.input_row = static_cast<std::uint32_t>(
            std::distance(unique_inputs.begin(), found));
      }
      input_rows.push_back(item.input_row);
    }
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(record_table_, record_pointers.data(),
                              record_pointers.size() * sizeof(void*),
                              cudaMemcpyHostToDevice, stream_);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(input_row_table_, input_rows.data(),
                              input_rows.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice, stream_);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(section_table_, sections.data(),
                              sections.size() * sizeof(SplitExpertSections),
                              cudaMemcpyHostToDevice, stream_);
    if (error == cudaSuccess) {
      quantize_q8<<<static_cast<std::uint32_t>(unique_inputs.size()), kThreads, 0U,
                    stream_>>>(
          static_cast<const float*>(input_),
          static_cast<std::int8_t*>(q_input_),
          static_cast<float*>(input_scale_),
          static_cast<std::uint32_t>(unique_inputs.size()), hidden);
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) {
      const dim3 grid((width + kWarpsPerBlock - 1U) / kWarpsPerBlock,
                      static_cast<std::uint32_t>(work.size()));
      gate_up<<<grid, kThreads, 0U, stream_>>>(
          reinterpret_cast<const std::uint8_t* const*>(record_table_),
          static_cast<const SplitExpertSections*>(section_table_),
          static_cast<const std::int8_t*>(q_input_),
          static_cast<const float*>(input_scale_),
          static_cast<const std::uint32_t*>(input_row_table_),
          static_cast<float*>(intermediate_), hidden, width,
          static_cast<std::uint32_t>(work.size()),
          config_.activation_clamp,
          config_.round_intermediate_to_bf16);
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) {
      quantize_q8<<<static_cast<std::uint32_t>(work.size()), kThreads, 0U,
                    stream_>>>(
          static_cast<const float*>(intermediate_),
          static_cast<std::int8_t*>(q_intermediate_),
          static_cast<float*>(intermediate_scale_),
          static_cast<std::uint32_t>(work.size()), width);
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) {
      const dim3 grid((hidden + kWarpsPerBlock - 1U) / kWarpsPerBlock,
                      static_cast<std::uint32_t>(work.size()));
      down<<<grid, kThreads, 0U, stream_>>>(
          reinterpret_cast<const std::uint8_t* const*>(record_table_),
          static_cast<const SplitExpertSections*>(section_table_),
          static_cast<const std::int8_t*>(q_intermediate_),
          static_cast<const float*>(intermediate_scale_),
          static_cast<float*>(output_), hidden, width,
          static_cast<std::uint32_t>(work.size()));
      error = cudaPeekAtLastError();
    }
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(output_buffer.data, output_,
                              work.size() * output_bytes,
                              cudaMemcpyDeviceToHost, stream_);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream_);
    if (error != cudaSuccess) {
      const auto status = cuda_status(error, "execute secondary CUDA experts");
      telemetry_.failed.fetch_add(work.size(), std::memory_order_relaxed);
      for (auto& item : work)
        complete(item.task, failed_result(item.task->request,
                                          copy_status(status)));
      return;
    }

    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    const auto attributed_elapsed = elapsed / work.size();
    std::vector<ActiveExpertExecutionResult> results;
    results.reserve(work.size());
    for (auto& item : work) {
      const auto& request = item.task->request;
      ActiveExpertExecutionResult result;
      result.status = Status::success();
      result.identity = request.identity;
      result.request_id = request.invocation.request_id;
      result.invocation_id = request.invocation.invocation_id;
      result.selection_index = request.invocation.selection_index;
      result.output = {
          request.invocation.output_abi, "host.pinned", output_lease,
          output_buffer.data +
              static_cast<std::size_t>(&item - work.data()) * output_bytes,
          output_bytes};
      result.evidence.activation_input_bytes =
          item.owns_input_transfer ? request.invocation.input.bytes : 0U;
      result.evidence.activation_output_bytes = output_bytes;
      const auto device_bytes = item.record->device_bytes == 0U
                                    ? item.record->stored_bytes
                                    : item.record->device_bytes;
      result.evidence.owner_weight_read_bytes = device_bytes;
      if (item.acquisition_tier == AcquisitionTier::vram) {
        result.evidence.owner_vram_read_bytes = device_bytes;
        telemetry_.vram_bytes.fetch_add(device_bytes,
                                        std::memory_order_relaxed);
      } else if (item.acquisition_tier == AcquisitionTier::ram) {
        result.evidence.owner_ram_read_bytes = item.record->stored_bytes;
        telemetry_.ram_bytes.fetch_add(item.record->stored_bytes,
                                       std::memory_order_relaxed);
      } else {
        result.evidence.owner_storage_read_bytes = item.record->stored_bytes;
        telemetry_.storage_bytes.fetch_add(item.record->stored_bytes,
                                           std::memory_order_relaxed);
      }
      result.evidence.owner_execution_ns = attributed_elapsed;
      if (item.owns_input_transfer)
        telemetry_.input_bytes.fetch_add(request.invocation.input.bytes,
                                         std::memory_order_relaxed);
      telemetry_.output_bytes.fetch_add(output_bytes,
                                        std::memory_order_relaxed);
      results.push_back(std::move(result));
    }
    telemetry_.completed.fetch_add(work.size(), std::memory_order_relaxed);
    telemetry_.execution_ns.fetch_add(elapsed, std::memory_order_relaxed);
    for (std::size_t index = 0U; index < work.size(); ++index)
      complete(work[index].task, std::move(results[index]));
  }

  void loop() noexcept {
    static_cast<void>(cudaSetDevice(ordinal_));
    for (;;) {
      std::vector<std::shared_ptr<Task>> batch;
      {
        std::unique_lock lock(queue_mutex_);
        queue_ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stop_) return;
          continue;
        }
        if (!stop_ && queue_.size() < maximum_batch_)
          static_cast<void>(queue_ready_.wait_for(
              lock, std::chrono::microseconds(200),
              [&] { return stop_ || queue_.size() >= maximum_batch_; }));
        const auto count = std::min<std::size_t>(queue_.size(), maximum_batch_);
        batch.reserve(count);
        for (std::size_t index = 0U; index < count; ++index) {
          batch.push_back(std::move(queue_.front()));
          queue_.pop_front();
        }
      }
      try {
        run_batch(batch);
      } catch (const std::exception& error) {
        telemetry_.failed.fetch_add(batch.size(), std::memory_order_relaxed);
        for (auto& task : batch)
          complete(task, failed_result(
                             task->request,
                             {ErrorCode::internal, error.what()}));
      }
    }
  }

  int ordinal_{};
  ActiveExpertDeviceExecutorConfig config_;
  std::vector<PayloadRecord> records_;
  AtomicTelemetry& telemetry_;
  std::shared_ptr<ActiveExpertAllocationPool> pool_;
  std::shared_ptr<ActiveExpertPackedUploader> uploader_;
  std::shared_ptr<FixedBufferPool> buffers_;
  std::shared_ptr<FixedBufferPool> output_buffers_;
  std::unique_ptr<ExpertCache> cache_;
  std::size_t maximum_batch_{};
  cudaStream_t stream_{};
  void* record_table_{};
  void* input_row_table_{};
  void* section_table_{};
  void* input_{};
  void* q_input_{};
  void* input_scale_{};
  void* intermediate_{};
  void* q_intermediate_{};
  void* intermediate_scale_{};
  void* output_{};
  std::mutex queue_mutex_;
  std::condition_variable queue_ready_;
  std::deque<std::shared_ptr<Task>> queue_;
  bool stop_{};
  std::thread thread_;
};

class ActiveExpertDeviceExecutor final : public IActiveExpertExecutor {
 public:
  ActiveExpertDeviceExecutor(ActiveExpertDeviceExecutorConfig config,
                             const ExpertCatalog& catalog,
                             std::shared_ptr<IAsyncStorage> storage)
      : config_(std::move(config)) {
    const auto count = static_cast<std::size_t>(config_.component.layer_count) *
                       config_.component.experts_per_layer;
    std::vector<PayloadRecord> records;
    records.reserve(count);
    for (std::uint32_t layer = 0U;
         layer < config_.component.layer_count; ++layer) {
      for (std::uint32_t expert = 0U;
           expert < config_.component.experts_per_layer; ++expert) {
        const auto* record = catalog.find(layer, expert);
        if (!record)
          throw std::runtime_error(
              "active expert device catalog is incomplete");
        const auto record_device_bytes = record->device_bytes == 0U
                                             ? record->stored_bytes
                                             : record->device_bytes;
        const auto first_device_bytes = records.empty()
                                            ? record_device_bytes
                                            : (records.front().device_bytes == 0U
                                                   ? records.front().stored_bytes
                                                   : records.front().device_bytes);
        if (!records.empty() &&
            (record->stored_bytes != records.front().stored_bytes ||
             record_device_bytes != first_device_bytes))
          throw std::runtime_error(
              "active expert device catalog has variable record sizes");
        records.push_back(*record);
      }
    }
    record_bytes_ = records.front().device_bytes == 0U
                        ? records.front().stored_bytes
                        : records.front().device_bytes;
    owner_ = "cuda.devices";
    for (const auto ordinal : config_.device_ordinals)
      owner_ += "." + std::to_string(ordinal);
    workers_.reserve(config_.device_ordinals.size());
    for (const auto ordinal : config_.device_ordinals)
      workers_.push_back(std::make_unique<DeviceState>(
          ordinal, config_, storage, records, telemetry_));
  }

  [[nodiscard]] std::string_view owner() const noexcept override {
    return owner_;
  }
  [[nodiscard]] bool remote() const noexcept override { return false; }

  [[nodiscard]] ActiveExpertExecutionHandle execute(
      ActiveExpertExecutionRequest request) override {
    telemetry_.requests.fetch_add(1U, std::memory_order_relaxed);
    const auto expected_bytes =
        static_cast<std::uint64_t>(config_.component.hidden_size) *
        sizeof(float);
    const bool valid =
        request.identity.model_content_hash == config_.model_content_hash &&
        request.identity.key.model_id == config_.component.namespace_id &&
        request.identity.key.layer < config_.component.layer_count &&
        request.identity.key.expert <
            config_.component.experts_per_layer &&
        request.identity.key.encoding_abi ==
            config_.component.encoding_abi &&
        request.identity.capability ==
            config_.component.execution_capability &&
        request.identity.execution_abi == config_.component.execution_abi &&
        request.identity.source_abi == config_.component.source_abi &&
        request.invocation.input.valid() &&
        request.invocation.input.abi == config_.input_abi &&
        request.invocation.input.bytes == expected_bytes &&
        request.invocation.output_abi == config_.output_abi &&
        request.invocation.output_bytes == expected_bytes &&
        request.invocation.selection_index <
            request.invocation.route_width;
    if (!valid) {
      telemetry_.failed.fetch_add(1U, std::memory_order_relaxed);
      auto task = std::make_shared<Task>(std::move(request));
      task->result = failed_result(
          task->request,
          {ErrorCode::invalid_argument,
           "secondary CUDA expert invocation violates its artifact contract"});
      return ActiveExpertExecutionHandle::from_callbacks(
          [task]() -> std::optional<ActiveExpertExecutionResult> {
            std::lock_guard lock(task->mutex);
            if (!task->result) return std::nullopt;
            auto result = std::move(*task->result);
            task->result.reset();
            return result;
          },
          [task] { task->cancelled.store(true, std::memory_order_release); });
    }
    auto task = std::make_shared<Task>(std::move(request));
    const auto worker = task->request.invocation.selection_index %
                        workers_.size();
    workers_[worker]->submit(task);
    return ActiveExpertExecutionHandle::from_callbacks(
        [task]() -> std::optional<ActiveExpertExecutionResult> {
          std::lock_guard lock(task->mutex);
          if (!task->result) return std::nullopt;
          auto result = std::move(*task->result);
          task->result.reset();
          return result;
        },
        [this, task] {
          task->cancelled.store(true, std::memory_order_release);
          telemetry_.cancelled.fetch_add(1U, std::memory_order_relaxed);
        });
  }

  [[nodiscard]] ActiveExpertExecutorTelemetry telemetry()
      const noexcept override {
    auto result = telemetry_.snapshot();
    result.owner_weight_read_bytes *= record_bytes_;
    return result;
  }

 private:
  ActiveExpertDeviceExecutorConfig config_;
  std::string owner_;
  std::uint64_t record_bytes_{};
  mutable AtomicTelemetry telemetry_;
  std::vector<std::unique_ptr<DeviceState>> workers_;
};

}  // namespace

CreateActiveExpertDeviceExecutorResult create_active_expert_device_executor(
    ActiveExpertDeviceExecutorConfig config, const ExpertCatalog& catalog,
    std::shared_ptr<IAsyncStorage> storage) noexcept {
  try {
    const auto* representative = catalog.find(0U, 0U);
    const auto source_record_bytes = representative == nullptr
                                         ? 0U
                                         : representative->stored_bytes;
    const auto device_record_bytes =
        representative == nullptr
            ? 0U
            : (representative->device_bytes == 0U
                   ? representative->stored_bytes
                   : representative->device_bytes);
    const bool valid =
        storage && !config.device_ordinals.empty() &&
        config.device_cache_bytes_per_device >= device_record_bytes * 2U &&
        config.device_reserve_bytes_per_device != 0U &&
        config.host_cache_bytes_total >=
            source_record_bytes * config.device_ordinals.size() &&
        config.staging_slots_per_device != 0U &&
        config.component.layer_count != 0U &&
        config.component.experts_per_layer != 0U &&
        config.component.route_width != 0U &&
        config.component.hidden_size % kExpertFp4BlockSize == 0U &&
        config.component.intermediate_size % kExpertFp4BlockSize == 0U &&
        config.component.encoding_abi == kExpertEncodingAbiFp4Block32 &&
        config.component.encoding == "fp4.e2m1.ue8m0.block32" &&
        catalog.layer_count() == config.component.layer_count &&
        catalog.experts_per_layer() ==
            config.component.experts_per_layer &&
        !config.input_abi.empty() && !config.output_abi.empty() &&
        std::isfinite(config.activation_clamp) &&
        config.activation_clamp >= 0.0F;
    if (!valid)
      return {{ErrorCode::invalid_argument,
               "active expert device executor configuration is invalid"},
              {}};
    int primary_device = -1;
    int device_count = 0;
    auto cuda_error = cudaGetDevice(&primary_device);
    if (cuda_error == cudaSuccess) cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess)
      return {{ErrorCode::internal,
               std::string("inspect active expert CUDA devices: ") +
                   cudaGetErrorString(cuda_error)},
              {}};
    if (std::any_of(config.device_ordinals.begin(),
                    config.device_ordinals.end(),
                    [&](int ordinal) {
                      return ordinal < 0 || ordinal >= device_count ||
                             ordinal == primary_device;
                    }))
      return {{ErrorCode::invalid_argument,
               "active expert CUDA devices must be valid secondary ordinals"},
              {}};
    std::sort(config.device_ordinals.begin(), config.device_ordinals.end());
    if (std::adjacent_find(config.device_ordinals.begin(),
                           config.device_ordinals.end()) !=
        config.device_ordinals.end())
      return {{ErrorCode::invalid_argument,
               "active expert CUDA device is duplicated"},
              {}};
    auto executor = std::make_shared<ActiveExpertDeviceExecutor>(
        std::move(config), catalog, std::move(storage));
    return {Status::success(), std::move(executor)};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("create active expert device executor: ") +
                 error.what()},
            {}};
  }
}

std::vector<int> discover_active_expert_devices() {
  int primary_device = -1;
  int device_count = 0;
  auto error = cudaGetDevice(&primary_device);
  if (error == cudaSuccess) error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess)
    throw std::runtime_error(
        std::string("discover secondary CUDA devices: ") +
        cudaGetErrorString(error));
  std::vector<int> result;
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    if (ordinal == primary_device) continue;
    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, ordinal);
    if (error != cudaSuccess)
      throw std::runtime_error(
          std::string("inspect secondary CUDA device: ") +
          cudaGetErrorString(error));
    if (properties.major == 8 && properties.minor >= 6)
      result.push_back(ordinal);
  }
  return result;
}

}  // namespace expert::runtime::cuda
