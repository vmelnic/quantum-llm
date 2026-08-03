#include "expert/runtime/buffer_pool.hpp"

#if defined(EXPERT_RUNTIME_HAS_CUDA_PINNED)

#include <cuda_runtime_api.h>

#include <cstdint>

namespace expert::runtime {

void* CudaPinnedAllocator::allocate(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0 || alignment == 0 ||
      (alignment & (alignment - 1U)) != 0) {
    return nullptr;
  }

  void* pointer = nullptr;
  if (cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) != cudaSuccess) {
    return nullptr;
  }

  // Direct I/O requires the actual address, not merely the requested slot
  // size, to satisfy the pack/volume alignment. Fail closed if the CUDA
  // runtime ever returns a less-aligned host allocation.
  if (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
    static_cast<void>(cudaFreeHost(pointer));
    return nullptr;
  }
  return pointer;
}

void CudaPinnedAllocator::deallocate(void* pointer) noexcept {
  if (pointer != nullptr) {
    static_cast<void>(cudaFreeHost(pointer));
  }
}

}  // namespace expert::runtime

#endif
