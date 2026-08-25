#include "expert/runtime/buffer_pool.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace expert::runtime {

void* AlignedHostAllocator::allocate(std::size_t bytes, std::size_t alignment) {
#if defined(_WIN32)
  return _aligned_malloc(bytes, alignment);
#else
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, bytes) != 0) {
    return nullptr;
  }
  return pointer;
#endif
}

void AlignedHostAllocator::deallocate(void* pointer) noexcept {
#if defined(_WIN32)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

MonotonicHostAllocator::MonotonicHostAllocator(
    std::size_t capacity, std::size_t alignment,
    std::shared_ptr<IHostAllocator> upstream)
    : upstream_(std::move(upstream)), capacity_(capacity),
      alignment_(alignment) {
  if (!capacity_ || !alignment_ || (alignment_ & (alignment_ - 1U)) != 0U ||
      !upstream_)
    throw std::invalid_argument("invalid monotonic host allocator");
  base_ = upstream_->allocate(capacity_, alignment_);
  if (!base_) throw std::bad_alloc();
}

MonotonicHostAllocator::~MonotonicHostAllocator() {
  upstream_->deallocate(base_);
}

void* MonotonicHostAllocator::allocate(std::size_t bytes,
                                       std::size_t alignment) {
  if (!bytes || !alignment || (alignment & (alignment - 1U)) != 0U ||
      alignment > alignment_)
    return nullptr;
  std::lock_guard lock(mutex_);
  const auto aligned = (cursor_ + alignment - 1U) / alignment * alignment;
  if (aligned > capacity_ || bytes > capacity_ - aligned) return nullptr;
  cursor_ = aligned + bytes;
  return static_cast<std::byte*>(base_) + aligned;
}

void MonotonicHostAllocator::deallocate(void*) noexcept {}

bool MonotonicHostAllocator::page_locked() const noexcept {
  return upstream_->page_locked();
}

std::size_t MonotonicHostAllocator::bytes_used() const noexcept {
  std::lock_guard lock(mutex_);
  return cursor_;
}

struct FixedBufferPool::SharedState final {
  struct Slot final {
    void* pointer{};
    bool used{};
  };

  SharedState(std::size_t count, std::size_t bytes, std::size_t requested_alignment,
              std::shared_ptr<IHostAllocator> host_allocator,
              std::size_t demand_reserve)
      : slot_bytes(bytes),
        alignment(requested_alignment),
        allocator(std::move(host_allocator)),
        slots(count),
        reserved_demand_slots(demand_reserve) {
    if (count == 0 || bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1U)) != 0 || !allocator) {
      throw std::invalid_argument("invalid fixed buffer pool configuration");
    }
    if (reserved_demand_slots > count) {
      throw std::invalid_argument("invalid fixed buffer pool demand reserve");
    }
    try {
      for (auto& slot : slots) {
        slot.pointer = allocator->allocate(slot_bytes, alignment);
        if (slot.pointer == nullptr) {
          throw std::bad_alloc();
        }
      }
    } catch (...) {
      for (auto& slot : slots) {
        allocator->deallocate(slot.pointer);
        slot.pointer = nullptr;
      }
      throw;
    }
  }

  ~SharedState() {
    for (auto& slot : slots) {
      allocator->deallocate(slot.pointer);
    }
  }

  std::size_t slot_bytes{};
  std::size_t alignment{};
  std::shared_ptr<IHostAllocator> allocator;
  mutable std::mutex mutex;
  std::vector<Slot> slots;
  std::size_t used{};
  std::size_t background_used{};
  std::size_t reserved_demand_slots{};
  std::size_t high_water_used{};
  std::size_t background_high_water_used{};
  std::uint64_t demand_acquires{};
  std::uint64_t background_acquires{};
  std::uint64_t demand_stalls{};
  std::uint64_t background_stalls{};
};

FixedBufferPool::Lease::Lease(std::shared_ptr<SharedState> state,
                              std::size_t slot,
                              BufferPoolClass allocation_class) noexcept
    : state_(std::move(state)), slot_(slot),
      allocation_class_(allocation_class) {}

FixedBufferPool::Lease::~Lease() {
  if (!state_) {
    return;
  }
  std::lock_guard lock(state_->mutex);
  auto& slot = state_->slots.at(slot_);
  slot.used = false;
  --state_->used;
  if (allocation_class_ == BufferPoolClass::background) {
    --state_->background_used;
  }
}

MutableBuffer FixedBufferPool::Lease::buffer() const noexcept {
  return {static_cast<std::byte*>(state_->slots[slot_].pointer),
          state_->slot_bytes};
}

bool FixedBufferPool::Lease::page_locked() const noexcept {
  return state_->allocator->page_locked();
}

FixedBufferPool::FixedBufferPool(std::size_t slot_count,
                                 std::size_t slot_bytes,
                                 std::size_t alignment,
                                 std::shared_ptr<IHostAllocator> allocator,
                                 std::size_t reserved_demand_slots)
    : state_(std::make_shared<SharedState>(slot_count, slot_bytes, alignment,
                                           std::move(allocator),
                                           reserved_demand_slots)) {}

std::shared_ptr<FixedBufferPool::Lease> FixedBufferPool::try_acquire(
    std::size_t bytes, BufferPoolClass allocation_class) {
  if (bytes > state_->slot_bytes) {
    return {};
  }
  std::lock_guard lock(state_->mutex);
  const auto background_limit =
      state_->slots.size() - state_->reserved_demand_slots;
  if (allocation_class == BufferPoolClass::background &&
      state_->background_used >= background_limit) {
    ++state_->background_stalls;
    return {};
  }
  for (std::size_t index = 0; index < state_->slots.size(); ++index) {
    if (!state_->slots[index].used) {
      state_->slots[index].used = true;
      ++state_->used;
      if (allocation_class == BufferPoolClass::background) {
        ++state_->background_used;
        ++state_->background_acquires;
        state_->background_high_water_used =
            std::max(state_->background_high_water_used,
                     state_->background_used);
      } else {
        ++state_->demand_acquires;
      }
      state_->high_water_used =
          std::max(state_->high_water_used, state_->used);
      return std::shared_ptr<Lease>(
          new Lease(state_, index, allocation_class));
    }
  }
  if (allocation_class == BufferPoolClass::background)
    ++state_->background_stalls;
  else
    ++state_->demand_stalls;
  return {};
}

std::size_t FixedBufferPool::slot_count() const noexcept {
  return state_->slots.size();
}

std::size_t FixedBufferPool::slot_bytes() const noexcept {
  return state_->slot_bytes;
}

std::size_t FixedBufferPool::alignment() const noexcept {
  return state_->alignment;
}

std::size_t FixedBufferPool::bytes_in_use() const noexcept {
  std::lock_guard lock(state_->mutex);
  return state_->used * state_->slot_bytes;
}

BufferPoolSnapshot FixedBufferPool::snapshot() const noexcept {
  std::lock_guard lock(state_->mutex);
  return {state_->used,
          state_->used - state_->background_used,
          state_->background_used,
          state_->high_water_used,
          state_->background_high_water_used,
          state_->demand_acquires,
          state_->background_acquires,
          state_->demand_stalls,
          state_->background_stalls};
}

}  // namespace expert::runtime
