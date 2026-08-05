#include "expert/runtime/cuda/expert_directory.hpp"

#include "expert/runtime/cuda/expert_uploader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint32_t kEmptyKey = std::numeric_limits<std::uint32_t>::max();
constexpr unsigned kThreads = 256;

Status checked(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return Status::success();
  return Status(ErrorCode::upload_failed,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

std::uint32_t hash_capacity(std::uint32_t maximum_selections) {
  std::uint32_t value = 1;
  const auto required = std::max<std::uint64_t>(
      2, static_cast<std::uint64_t>(maximum_selections) * 2U);
  while (value < required) {
    if (value > (1U << 30U)) {
      throw std::invalid_argument("expert route hash capacity overflow");
    }
    value <<= 1U;
  }
  return value;
}

__global__ void insert_route_keys(const std::uint32_t* selections,
                                  std::uint32_t selection_count,
                                  std::uint32_t expert_count,
                                  std::uint32_t* keys,
                                  std::uint32_t capacity,
                                  std::uint32_t* error) {
  for (auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x +
                                               threadIdx.x);
       index < selection_count; index += blockDim.x * gridDim.x) {
    const auto expert = selections[index];
    if (expert >= expert_count) {
      atomicExch(error, 1U);
      continue;
    }
    auto slot = (expert * 2654435761U) & (capacity - 1U);
    for (std::uint32_t probe = 0; probe < capacity; ++probe) {
      const auto previous = atomicCAS(keys + slot, kEmptyKey, expert);
      if (previous == kEmptyKey || previous == expert) break;
      slot = (slot + 1U) & (capacity - 1U);
      if (probe + 1U == capacity) atomicExch(error, 2U);
    }
  }
}

__global__ void pin_directory_entries(DeviceExpertEntry* directory,
                                      std::uint32_t layer_offset,
                                      const std::uint32_t* keys,
                                      std::uint8_t* pinned,
                                      std::uint32_t capacity,
                                      std::uint32_t* missing,
                                      std::uint32_t* missing_count,
                                      std::uint32_t* unique_count) {
  for (auto slot = static_cast<std::uint32_t>(blockIdx.x * blockDim.x +
                                              threadIdx.x);
       slot < capacity; slot += blockDim.x * gridDim.x) {
    const auto expert = keys[slot];
    if (expert == kEmptyKey) continue;
    atomicAdd(unique_count, 1U);
    auto* entry = directory + layer_offset + expert;
    const auto state = atomicAdd(&entry->state, 0U);
    if (state == static_cast<std::uint32_t>(DeviceExpertState::ready)) {
      atomicAdd(&entry->device_references, 1U);
      __threadfence();
      if (atomicAdd(&entry->state, 0U) ==
          static_cast<std::uint32_t>(DeviceExpertState::ready)) {
        pinned[slot] = 1;
        continue;
      }
      atomicSub(&entry->device_references, 1U);
    }
    missing[atomicAdd(missing_count, 1U)] = expert;
  }
}

__global__ void release_directory_pins(DeviceExpertEntry* directory,
                                       std::uint32_t layer_offset,
                                       const std::uint32_t* keys,
                                       std::uint8_t* pinned,
                                       std::uint32_t capacity) {
  for (auto slot = static_cast<std::uint32_t>(blockIdx.x * blockDim.x +
                                              threadIdx.x);
       slot < capacity; slot += blockDim.x * gridDim.x) {
    if (pinned[slot] == 0) continue;
    const auto expert = keys[slot];
    atomicSub(&directory[layer_offset + expert].device_references, 1U);
    pinned[slot] = 0;
  }
}

__global__ void release_selected_pins(DeviceExpertEntry* directory,
                                      std::uint32_t layer_offset,
                                      const std::uint32_t* experts,
                                      std::uint32_t count) {
  const auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x +
                                                threadIdx.x);
  if (index < count)
    atomicSub(&directory[layer_offset + experts[index]].device_references, 1U);
}

__global__ void try_retire_entry(DeviceExpertEntry* entry,
                                 std::uint32_t* retired) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const auto ready = static_cast<std::uint32_t>(DeviceExpertState::ready);
  const auto retiring =
      static_cast<std::uint32_t>(DeviceExpertState::retiring);
  const auto previous = atomicCAS(&entry->state, ready, retiring);
  if (previous == static_cast<std::uint32_t>(DeviceExpertState::absent)) {
    *retired = 1;
    return;
  }
  if (previous != ready) {
    *retired = 0;
    return;
  }
  __threadfence();
  if (atomicAdd(&entry->device_references, 0U) == 0U) {
    entry->gate_up = nullptr;
    entry->gate_up_scales = nullptr;
    entry->down = nullptr;
    entry->down_scales = nullptr;
    __threadfence();
    entry->state = static_cast<std::uint32_t>(DeviceExpertState::absent);
    *retired = 1;
  } else {
    __threadfence();
    entry->state = ready;
    *retired = 0;
  }
}

}  // namespace

struct CudaExpertDirectory::Impl final {
  std::uint64_t model_id{};
  std::uint32_t quant_abi{};
  std::uint32_t layers{};
  std::uint32_t experts{};
  std::uint32_t maximum_selections{};
  std::uint32_t maximum_active_pins{};
  std::uint32_t hash_slots{};
  DeviceExpertEntry* entries{};
  std::uint32_t* hash_keys{};
  std::uint8_t* pinned{};
  std::uint32_t* missing{};
  std::uint32_t* missing_count{};
  std::uint32_t* unique_count{};
  std::uint32_t* error{};
  std::uint32_t* retired{};
  std::vector<std::uint32_t> generations;
  struct PinSlot final {
    std::uint32_t* device_experts{};
    cudaEvent_t completion{};
    bool in_use{};
    bool release_pending{};
  };
  struct ActivePin final {
    std::uint32_t layer{};
    std::vector<std::uint32_t> experts;
    std::size_t slot{};
  };
  std::vector<PinSlot> pin_slots;
  std::unordered_map<std::uint64_t, ActivePin> active_pins;
  std::uint64_t next_pin_id{1U};
  std::mutex mutex;

  ~Impl() {
    for (auto& slot : pin_slots) {
      if (slot.release_pending && slot.completion)
        static_cast<void>(cudaEventSynchronize(slot.completion));
      if (slot.completion) static_cast<void>(cudaEventDestroy(slot.completion));
      static_cast<void>(cudaFree(slot.device_experts));
    }
    static_cast<void>(cudaFree(retired));
    static_cast<void>(cudaFree(error));
    static_cast<void>(cudaFree(unique_count));
    static_cast<void>(cudaFree(missing_count));
    static_cast<void>(cudaFree(missing));
    static_cast<void>(cudaFree(pinned));
    static_cast<void>(cudaFree(hash_keys));
    static_cast<void>(cudaFree(entries));
  }
};

CudaExpertDirectory::CudaExpertDirectory(
    std::uint64_t model_id, std::uint32_t quant_abi, std::uint32_t layers,
    std::uint32_t experts_per_layer, std::uint32_t maximum_selections,
    std::uint32_t maximum_active_pins)
    : impl_(std::make_unique<Impl>()) {
  if (layers == 0 || experts_per_layer == 0 || maximum_selections == 0 ||
      maximum_active_pins == 0) {
    throw std::invalid_argument("invalid CUDA expert directory geometry");
  }
  impl_->model_id = model_id;
  impl_->quant_abi = quant_abi;
  impl_->layers = layers;
  impl_->experts = experts_per_layer;
  impl_->maximum_selections = maximum_selections;
  impl_->maximum_active_pins = maximum_active_pins;
  impl_->hash_slots = hash_capacity(maximum_selections);
  const auto entry_count = static_cast<std::size_t>(layers) * experts_per_layer;
  impl_->generations.resize(entry_count);
  const auto allocate = [](auto** pointer, std::size_t bytes, const char* name) {
    if (const auto error = cudaMalloc(reinterpret_cast<void**>(pointer), bytes);
        error != cudaSuccess) {
      throw std::runtime_error(std::string(name) + ": " +
                               cudaGetErrorString(error));
    }
  };
  allocate(&impl_->entries, entry_count * sizeof(DeviceExpertEntry),
           "cudaMalloc expert directory");
  allocate(&impl_->hash_keys,
           impl_->hash_slots * sizeof(std::uint32_t),
           "cudaMalloc route hash");
  allocate(&impl_->pinned, impl_->hash_slots * sizeof(std::uint8_t),
           "cudaMalloc route pins");
  allocate(&impl_->missing, maximum_selections * sizeof(std::uint32_t),
           "cudaMalloc route misses");
  allocate(&impl_->missing_count, sizeof(std::uint32_t),
           "cudaMalloc route miss count");
  allocate(&impl_->unique_count, sizeof(std::uint32_t),
           "cudaMalloc route unique count");
  allocate(&impl_->error, sizeof(std::uint32_t), "cudaMalloc route error");
  allocate(&impl_->retired, sizeof(std::uint32_t),
           "cudaMalloc retire result");
  impl_->pin_slots.resize(maximum_active_pins);
  for (auto& slot : impl_->pin_slots) {
    allocate(&slot.device_experts,
             maximum_selections * sizeof(std::uint32_t),
             "cudaMalloc directory pin slot");
    if (const auto event_error =
            cudaEventCreateWithFlags(&slot.completion, cudaEventDisableTiming);
        event_error != cudaSuccess) {
      throw std::runtime_error(std::string("cudaEventCreate directory pin: ") +
                               cudaGetErrorString(event_error));
    }
  }
  const auto error = cudaMemset(impl_->entries, 0,
                                entry_count * sizeof(DeviceExpertEntry));
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemset expert directory: ") +
                             cudaGetErrorString(error));
  }
}

CudaExpertDirectory::~CudaExpertDirectory() = default;

Status CudaExpertDirectory::publish(
    const ExpertKey& key, std::shared_ptr<IDeviceAllocation> allocation) {
  std::lock_guard lock(impl_->mutex);
  if (key.model_id != impl_->model_id || key.quant_abi != impl_->quant_abi ||
      key.layer >= impl_->layers || key.expert >= impl_->experts) {
    return Status(ErrorCode::invalid_argument,
                  "expert key is outside CUDA directory");
  }
  const auto* cuda_allocation =
      dynamic_cast<const CudaExpertAllocation*>(allocation.get());
  const auto* compact_allocation =
      dynamic_cast<const CudaCompactExpertAllocation*>(allocation.get());
  if (!cuda_allocation && !compact_allocation) {
    return Status(ErrorCode::invalid_argument,
                  "CUDA directory received a non-CUDA allocation");
  }
  const auto index = static_cast<std::size_t>(key.layer) * impl_->experts +
                     key.expert;
  DeviceExpertEntry entry{};
  if (cuda_allocation) {
    entry.gate_up = cuda_allocation->gate_up();
    entry.gate_up_scales = cuda_allocation->gate_up_scales();
    entry.down = cuda_allocation->down();
    entry.down_scales = cuda_allocation->down_scales();
    entry.format = static_cast<std::uint32_t>(
        DeviceExpertFormat::int8_per_row);
  } else {
    const auto* base = compact_allocation->base();
    const auto& sections = compact_allocation->sections();
    entry.w1_fp4 = base + sections.w1_weight_offset;
    entry.w1_ue8m0 = base + sections.w1_scale_offset;
    entry.w3_fp4 = base + sections.w3_weight_offset;
    entry.w3_ue8m0 = base + sections.w3_scale_offset;
    entry.w2_fp4 = base + sections.w2_weight_offset;
    entry.w2_ue8m0 = base + sections.w2_scale_offset;
    entry.format = static_cast<std::uint32_t>(
        DeviceExpertFormat::deepseek_fp4_block32);
  }
  entry.generation = ++impl_->generations[index];
  entry.state = static_cast<std::uint32_t>(DeviceExpertState::ready);
  return checked(cudaMemcpy(impl_->entries + index, &entry, sizeof(entry),
                            cudaMemcpyHostToDevice),
                 "publish expert directory entry");
}

void CudaExpertDirectory::retire(const ExpertKey& key) noexcept {
  if (!impl_ || key.model_id != impl_->model_id ||
      key.quant_abi != impl_->quant_abi || key.layer >= impl_->layers ||
      key.expert >= impl_->experts) {
    return;
  }
  const auto index = static_cast<std::size_t>(key.layer) * impl_->experts +
                     key.expert;
  for (;;) {
    std::uint32_t retired = 0;
    static_cast<void>(cudaMemset(impl_->retired, 0, sizeof(std::uint32_t)));
    try_retire_entry<<<1, 1>>>(impl_->entries + index, impl_->retired);
    if (cudaPeekAtLastError() != cudaSuccess ||
        cudaMemcpy(&retired, impl_->retired, sizeof(retired),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return;
    }
    if (retired != 0) return;
    std::this_thread::yield();
  }
}

DirectoryPlanResult CudaExpertDirectory::pin_or_collect_misses(
    std::uint32_t layer, const std::uint32_t* device_expert_indices,
    std::uint32_t selection_count, void* raw_stream,
    bool keep_ready_pins_on_miss) {
  std::lock_guard lock(impl_->mutex);
  DirectoryPlanResult result{Status::success(), {}, {}, {}, 0, 0};
  if (layer >= impl_->layers || device_expert_indices == nullptr || selection_count == 0 ||
      selection_count > impl_->maximum_selections) {
    result.status = Status(ErrorCode::invalid_argument,
                           "invalid CUDA directory plan");
    return result;
  }
  const auto stream = static_cast<cudaStream_t>(raw_stream);
  auto error = cudaMemsetAsync(impl_->hash_keys, 0xff,
                               impl_->hash_slots * sizeof(std::uint32_t),
                               stream);
  if (error == cudaSuccess)
    error = cudaMemsetAsync(impl_->pinned, 0,
                            impl_->hash_slots * sizeof(std::uint8_t), stream);
  if (error == cudaSuccess)
    error = cudaMemsetAsync(impl_->missing_count, 0, sizeof(std::uint32_t),
                            stream);
  if (error == cudaSuccess)
    error = cudaMemsetAsync(impl_->unique_count, 0, sizeof(std::uint32_t),
                            stream);
  if (error == cudaSuccess)
    error = cudaMemsetAsync(impl_->error, 0, sizeof(std::uint32_t), stream);
  if (error != cudaSuccess) {
    result.status = checked(error, "initialize directory plan");
    return result;
  }
  const auto blocks = std::min(128U, (selection_count + kThreads - 1U) /
                                         kThreads);
  insert_route_keys<<<blocks, kThreads, 0, stream>>>(
      device_expert_indices, selection_count, impl_->experts,
      impl_->hash_keys, impl_->hash_slots, impl_->error);
  const auto hash_blocks =
      std::min(128U, (impl_->hash_slots + kThreads - 1U) / kThreads);
  pin_directory_entries<<<hash_blocks, kThreads, 0, stream>>>(
      impl_->entries, layer * impl_->experts, impl_->hash_keys, impl_->pinned,
      impl_->hash_slots, impl_->missing, impl_->missing_count,
      impl_->unique_count);
  std::uint32_t missing_count = 0;
  std::uint32_t plan_error = 0;
  std::vector<std::uint32_t> route_keys(impl_->hash_slots);
  result.selected_experts.resize(selection_count);
  error = cudaMemcpyAsync(&missing_count, impl_->missing_count,
                          sizeof(missing_count), cudaMemcpyDeviceToHost,
                          stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&result.unique_experts, impl_->unique_count,
                            sizeof(result.unique_experts),
                            cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(&plan_error, impl_->error, sizeof(plan_error),
                            cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(route_keys.data(), impl_->hash_keys,
                            impl_->hash_slots * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(result.selected_experts.data(),
                            device_expert_indices,
                            selection_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess || plan_error != 0) {
    release_directory_pins<<<hash_blocks, kThreads, 0, stream>>>(
        impl_->entries, layer * impl_->experts, impl_->hash_keys,
        impl_->pinned, impl_->hash_slots);
    static_cast<void>(cudaStreamSynchronize(stream));
    result.status = error != cudaSuccess
                        ? checked(error, "complete directory plan")
                        : Status(ErrorCode::invalid_argument,
                                 "route hash rejected expert indices");
    return result;
  }
  if (missing_count != 0) {
    result.missing_experts.resize(missing_count);
    error = cudaMemcpyAsync(result.missing_experts.data(), impl_->missing,
                            missing_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
      release_directory_pins<<<hash_blocks, kThreads, 0, stream>>>(
          impl_->entries, layer * impl_->experts, impl_->hash_keys,
          impl_->pinned, impl_->hash_slots);
      static_cast<void>(cudaStreamSynchronize(stream));
      result.status = checked(error, "copy directory misses");
    } else {
      result.ready_experts.reserve(result.unique_experts - missing_count);
      for (const auto expert : route_keys) {
        if (expert == kEmptyKey ||
            std::find(result.missing_experts.begin(),
                      result.missing_experts.end(), expert) !=
                result.missing_experts.end()) {
          continue;
        }
        result.ready_experts.push_back(expert);
      }
      if (!keep_ready_pins_on_miss) {
        release_directory_pins<<<hash_blocks, kThreads, 0, stream>>>(
            impl_->entries, layer * impl_->experts, impl_->hash_keys,
            impl_->pinned, impl_->hash_slots);
        const auto release_error = cudaStreamSynchronize(stream);
        if (release_error != cudaSuccess)
          result.status = checked(release_error,
                                  "release ready directory misses");
      }
    }
  }

  const bool retain = missing_count == 0U || keep_ready_pins_on_miss;
  if (!result.status.ok() || !retain) return result;
  if (impl_->active_pins.size() >= impl_->maximum_active_pins) {
    release_directory_pins<<<hash_blocks, kThreads, 0, stream>>>(
        impl_->entries, layer * impl_->experts, impl_->hash_keys,
        impl_->pinned, impl_->hash_slots);
    static_cast<void>(cudaStreamSynchronize(stream));
    result.status = Status(ErrorCode::backpressure,
                           "CUDA directory active pin capacity exhausted");
    return result;
  }
  std::vector<std::uint32_t> pinned_experts;
  pinned_experts.reserve(result.ready_experts.size());
  for (const auto expert : route_keys) {
    if (expert == kEmptyKey ||
        std::find(result.missing_experts.begin(),
                  result.missing_experts.end(), expert) !=
            result.missing_experts.end())
      continue;
    pinned_experts.push_back(expert);
  }
  std::size_t pin_slot = impl_->pin_slots.size();
  for (std::size_t index = 0U; index < impl_->pin_slots.size(); ++index) {
    auto& slot = impl_->pin_slots[index];
    if (slot.release_pending) {
      const auto query = cudaEventQuery(slot.completion);
      if (query == cudaSuccess) {
        slot.release_pending = false;
        slot.in_use = false;
      } else if (query != cudaErrorNotReady) {
        release_directory_pins<<<hash_blocks, kThreads, 0, stream>>>(
            impl_->entries, layer * impl_->experts, impl_->hash_keys,
            impl_->pinned, impl_->hash_slots);
        static_cast<void>(cudaStreamSynchronize(stream));
        result.status = checked(query, "query directory pin slot");
        return result;
      }
    }
    if (!slot.in_use) {
      pin_slot = index;
      break;
    }
  }
  if (pin_slot == impl_->pin_slots.size()) {
    release_directory_pins<<<hash_blocks, kThreads, 0, stream>>>(
        impl_->entries, layer * impl_->experts, impl_->hash_keys,
        impl_->pinned, impl_->hash_slots);
    static_cast<void>(cudaStreamSynchronize(stream));
    result.status = Status(ErrorCode::backpressure,
                           "CUDA directory pin slots exhausted");
    return result;
  }
  auto& slot = impl_->pin_slots[pin_slot];
  if (!pinned_experts.empty()) {
    const auto copy_error = cudaMemcpyAsync(
        slot.device_experts, pinned_experts.data(),
        pinned_experts.size() * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice, stream);
    if (copy_error != cudaSuccess) {
      release_directory_pins<<<hash_blocks, kThreads, 0, stream>>>(
          impl_->entries, layer * impl_->experts, impl_->hash_keys,
          impl_->pinned, impl_->hash_slots);
      static_cast<void>(cudaStreamSynchronize(stream));
      result.status = checked(copy_error, "stage directory pin slot");
      return result;
    }
  }
  slot.in_use = true;
  auto pin_id = impl_->next_pin_id++;
  while (pin_id == 0U || impl_->active_pins.contains(pin_id))
    pin_id = impl_->next_pin_id++;
  impl_->active_pins.emplace(
      pin_id, Impl::ActivePin{layer, std::move(pinned_experts), pin_slot});
  result.pin_id = pin_id;
  return result;
}

Status CudaExpertDirectory::release_pins(std::uint64_t pin_id,
                                         void* raw_stream) noexcept {
  std::lock_guard lock(impl_->mutex);
  const auto active = impl_->active_pins.find(pin_id);
  if (pin_id == 0U || active == impl_->active_pins.end()) {
    return Status(ErrorCode::invalid_argument,
                  "CUDA directory pin token is not active");
  }
  const auto stream = static_cast<cudaStream_t>(raw_stream);
  auto status = Status::success();
  auto& slot = impl_->pin_slots[active->second.slot];
  if (!active->second.experts.empty()) {
    const auto count = static_cast<std::uint32_t>(active->second.experts.size());
    release_selected_pins<<<(count + kThreads - 1U) / kThreads, kThreads, 0,
                            stream>>>(
        impl_->entries, active->second.layer * impl_->experts,
        slot.device_experts, count);
    auto error = cudaPeekAtLastError();
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    status = checked(error, "release directory pin token");
  }
  if (status.ok()) {
    slot.in_use = false;
    slot.release_pending = false;
  }
  if (status.ok()) impl_->active_pins.erase(active);
  return status;
}

Status CudaExpertDirectory::release_pins_async(std::uint64_t pin_id,
                                               void* raw_stream) noexcept {
  std::lock_guard lock(impl_->mutex);
  const auto active = impl_->active_pins.find(pin_id);
  if (pin_id == 0U || active == impl_->active_pins.end()) {
    return Status(ErrorCode::invalid_argument,
                  "CUDA directory pin token is not active");
  }
  auto& slot = impl_->pin_slots[active->second.slot];
  const auto stream = static_cast<cudaStream_t>(raw_stream);
  if (!active->second.experts.empty()) {
    const auto count = static_cast<std::uint32_t>(active->second.experts.size());
    release_selected_pins<<<(count + kThreads - 1U) / kThreads, kThreads, 0,
                            stream>>>(
        impl_->entries, active->second.layer * impl_->experts,
        slot.device_experts, count);
    auto error = cudaPeekAtLastError();
    if (error == cudaSuccess)
      error = cudaEventRecord(slot.completion, stream);
    if (error != cudaSuccess) {
      // A successfully launched release kernel still owns the reference
      // transition. Drain before discarding its bookkeeping when event
      // publication itself fails, preventing a retry from decrementing twice.
      static_cast<void>(cudaStreamSynchronize(stream));
      slot.in_use = false;
      slot.release_pending = false;
      impl_->active_pins.erase(active);
      return checked(error, "enqueue directory pin release");
    }
    slot.release_pending = true;
  } else {
    slot.in_use = false;
  }
  impl_->active_pins.erase(active);
  return Status::success();
}

const DeviceExpertEntry* CudaExpertDirectory::device_entries() const noexcept {
  return impl_->entries;
}

std::uint32_t CudaExpertDirectory::experts_per_layer() const noexcept {
  return impl_->experts;
}

}  // namespace expert::runtime::cuda
