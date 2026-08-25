#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace expert::runtime {

enum class BufferPoolClass : std::uint8_t {
  demand,
  background,
};

struct BufferPoolSnapshot final {
  std::size_t slots_in_use{};
  std::size_t demand_slots_in_use{};
  std::size_t background_slots_in_use{};
  std::size_t high_water_slots{};
  std::size_t background_high_water_slots{};
  std::uint64_t demand_acquires{};
  std::uint64_t background_acquires{};
  std::uint64_t demand_stalls{};
  std::uint64_t background_stalls{};
};

class IHostAllocator {
 public:
  virtual ~IHostAllocator() = default;
  [[nodiscard]] virtual void* allocate(std::size_t bytes,
                                       std::size_t alignment) = 0;
  virtual void deallocate(void* pointer) noexcept = 0;
  [[nodiscard]] virtual bool page_locked() const noexcept = 0;
};

class AlignedHostAllocator final : public IHostAllocator {
 public:
  [[nodiscard]] void* allocate(std::size_t bytes,
                               std::size_t alignment) override;
  void deallocate(void* pointer) noexcept override;
  [[nodiscard]] bool page_locked() const noexcept override { return false; }
};

// One bounded host allocation carved into immutable aligned records. This is
// useful when the complete expert pool fits RAM: it avoids thousands of OS or
// CUDA pin/unpin operations while keeping every retained record in one bank.
// Individual deallocations are intentionally no-ops; the bank is reclaimed as
// a unit after the cache releases all records.
class MonotonicHostAllocator final : public IHostAllocator {
 public:
  MonotonicHostAllocator(std::size_t capacity, std::size_t alignment,
                         std::shared_ptr<IHostAllocator> upstream);
  ~MonotonicHostAllocator() override;
  MonotonicHostAllocator(const MonotonicHostAllocator&) = delete;
  MonotonicHostAllocator& operator=(const MonotonicHostAllocator&) = delete;

  [[nodiscard]] void* allocate(std::size_t bytes,
                               std::size_t alignment) override;
  void deallocate(void* pointer) noexcept override;
  [[nodiscard]] bool page_locked() const noexcept override;
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t bytes_used() const noexcept;

 private:
  std::shared_ptr<IHostAllocator> upstream_;
  void* base_{};
  std::size_t capacity_{};
  std::size_t alignment_{};
  std::size_t cursor_{};
  mutable std::mutex mutex_;
};

#if defined(EXPERT_RUNTIME_HAS_CUDA_PINNED)
class CudaPinnedAllocator final : public IHostAllocator {
 public:
  [[nodiscard]] void* allocate(std::size_t bytes,
                               std::size_t alignment) override;
  void deallocate(void* pointer) noexcept override;
  [[nodiscard]] bool page_locked() const noexcept override { return true; }
};
#endif

class FixedBufferPool final {
 private:
  struct SharedState;

 public:
  class Lease final {
   public:
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&&) = delete;
    Lease& operator=(Lease&&) = delete;
    ~Lease();

    [[nodiscard]] MutableBuffer buffer() const noexcept;
    [[nodiscard]] bool page_locked() const noexcept;

   private:
    friend class FixedBufferPool;
    Lease(std::shared_ptr<SharedState> state, std::size_t slot,
          BufferPoolClass allocation_class) noexcept;

    std::shared_ptr<SharedState> state_;
    std::size_t slot_{};
    BufferPoolClass allocation_class_{BufferPoolClass::demand};
  };

  FixedBufferPool(std::size_t slot_count, std::size_t slot_bytes,
                  std::size_t alignment,
                  std::shared_ptr<IHostAllocator> allocator =
                      std::make_shared<AlignedHostAllocator>(),
                  std::size_t reserved_demand_slots = 0U);

  [[nodiscard]] std::shared_ptr<Lease> try_acquire(
      std::size_t bytes,
      BufferPoolClass allocation_class = BufferPoolClass::demand);
  [[nodiscard]] std::size_t slot_count() const noexcept;
  [[nodiscard]] std::size_t slot_bytes() const noexcept;
  [[nodiscard]] std::size_t alignment() const noexcept;
  [[nodiscard]] std::size_t bytes_in_use() const noexcept;
  [[nodiscard]] BufferPoolSnapshot snapshot() const noexcept;

 private:
  std::shared_ptr<SharedState> state_;
};

}  // namespace expert::runtime
