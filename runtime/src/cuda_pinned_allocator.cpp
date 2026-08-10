#include "expert/runtime/buffer_pool.hpp"

#if defined(EXPERT_RUNTIME_HAS_CUDA_PINNED)

#include <cuda_runtime_api.h>

#include <cstdint>

namespace expert::runtime {

void* CudaPinnedAllocator::allocate(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0 || alignment == 0 ||
      (alignment & (alignment - 1U)) != 0 ||
      alignment < sizeof(void*)) {
    return nullptr;
  }

  // cudaHostAlloc does NOT guarantee the pack/direct-I/O alignment (observed
  // on a 3090 host: a 512-byte-aligned pointer for a 1.6 MB request, which
  // used to fail the pool closed with std::bad_alloc). Over-allocate and
  // round up; the raw base pointer rides in a header slot just before the
  // aligned address so deallocate() can free the real allocation.
  const std::size_t total = bytes + alignment - 1U + sizeof(void*);
  void* raw = nullptr;
  if (cudaHostAlloc(&raw, total, cudaHostAllocDefault) != cudaSuccess) {
    return nullptr;
  }
  const auto aligned =
      (reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*) + alignment - 1U) &
      ~static_cast<std::uintptr_t>(alignment - 1U);
  reinterpret_cast<void**>(aligned)[-1] = raw;
  return reinterpret_cast<void*>(aligned);
}

void CudaPinnedAllocator::deallocate(void* pointer) noexcept {
  if (pointer != nullptr) {
    static_cast<void>(cudaFreeHost(reinterpret_cast<void**>(pointer)[-1]));
  }
}

}  // namespace expert::runtime

#endif
