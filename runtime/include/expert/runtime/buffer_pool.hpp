#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace expert::runtime {

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
    Lease(std::shared_ptr<SharedState> state, std::size_t slot) noexcept;

    std::shared_ptr<SharedState> state_;
    std::size_t slot_{};
  };

  FixedBufferPool(std::size_t slot_count, std::size_t slot_bytes,
                  std::size_t alignment,
                  std::shared_ptr<IHostAllocator> allocator =
                      std::make_shared<AlignedHostAllocator>());

  [[nodiscard]] std::shared_ptr<Lease> try_acquire(std::size_t bytes);
  [[nodiscard]] std::size_t slot_count() const noexcept;
  [[nodiscard]] std::size_t slot_bytes() const noexcept;
  [[nodiscard]] std::size_t alignment() const noexcept;
  [[nodiscard]] std::size_t bytes_in_use() const noexcept;

 private:
  std::shared_ptr<SharedState> state_;
};

}  // namespace expert::runtime

