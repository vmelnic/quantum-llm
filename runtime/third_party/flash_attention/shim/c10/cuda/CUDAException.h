#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define C10_CUDA_CHECK(expression)                                             \
  do {                                                                         \
    const cudaError_t c10_cuda_status = (expression);                          \
    if (c10_cuda_status != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA error: %s\n",                                \
                   cudaGetErrorString(c10_cuda_status));                       \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaPeekAtLastError())
