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

struct FixedBufferPool::SharedState final {
  struct Slot final {
    void* pointer{};
    bool used{};
  };

  SharedState(std::size_t count, std::size_t bytes, std::size_t requested_alignment,
              std::shared_ptr<IHostAllocator> host_allocator)
      : slot_bytes(bytes),
        alignment(requested_alignment),
        allocator(std::move(host_allocator)),
        slots(count) {
    if (count == 0 || bytes == 0 || alignment == 0 ||
        (alignment & (alignment - 1U)) != 0 || !allocator) {
      throw std::invalid_argument("invalid fixed buffer pool configuration");
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
};

FixedBufferPool::Lease::Lease(std::shared_ptr<SharedState> state,
                              std::size_t slot) noexcept
    : state_(std::move(state)), slot_(slot) {}

FixedBufferPool::Lease::~Lease() {
  if (!state_) {
    return;
  }
  std::lock_guard lock(state_->mutex);
  auto& slot = state_->slots.at(slot_);
  slot.used = false;
  --state_->used;
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
                                 std::shared_ptr<IHostAllocator> allocator)
    : state_(std::make_shared<SharedState>(slot_count, slot_bytes, alignment,
                                           std::move(allocator))) {}

std::shared_ptr<FixedBufferPool::Lease> FixedBufferPool::try_acquire(
    std::size_t bytes) {
  if (bytes > state_->slot_bytes) {
    return {};
  }
  std::lock_guard lock(state_->mutex);
  for (std::size_t index = 0; index < state_->slots.size(); ++index) {
    if (!state_->slots[index].used) {
      state_->slots[index].used = true;
      ++state_->used;
      return std::shared_ptr<Lease>(new Lease(state_, index));
    }
  }
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

}  // namespace expert::runtime

