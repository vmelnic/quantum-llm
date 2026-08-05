#include "expert/runtime/cuda/deepseek_admission.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256U;

__device__ __constant__ float kFp4Values[16] = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

__device__ float decode_value(const std::uint8_t* packed,
                              const std::uint8_t* scales,
                              std::uint32_t row, std::uint32_t column,
                              std::uint32_t columns,
                              unsigned* invalid) {
  const auto packed_columns = columns / 2U;
  const auto byte = packed[static_cast<std::size_t>(row) * packed_columns +
                           column / 2U];
  const auto code = (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U;
  const auto scale_code =
      scales[static_cast<std::size_t>(row) * (columns / 32U) + column / 32U];
  if (scale_code == 255U) {
    atomicExch(invalid, 1U);
    return 0.0F;
  }
  return kFp4Values[code] * ldexpf(1.0F, static_cast<int>(scale_code) - 127);
}

__global__ void compact_to_int8(const std::uint8_t* packed,
                                const std::uint8_t* scales,
                                std::int8_t* output, float* row_scales,
                                std::uint32_t rows, std::uint32_t columns,
                                unsigned* invalid) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= rows) return;
  __shared__ float maxima[kThreads];
  float maximum = 0.0F;
  for (auto column = static_cast<std::uint32_t>(threadIdx.x); column < columns;
       column += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(decode_value(
        packed, scales, row, column, columns, invalid)));
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x],
                                  maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const auto scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) row_scales[row] = scale;
  __syncthreads();
  for (auto column = static_cast<std::uint32_t>(threadIdx.x); column < columns;
       column += blockDim.x) {
    const auto value = decode_value(packed, scales, row, column, columns, invalid);
    auto quantized = __float2int_rn(value / scale);
    quantized = quantized < -127 ? -127 : (quantized > 127 ? 127 : quantized);
    output[static_cast<std::size_t>(row) * columns + column] =
        static_cast<std::int8_t>(quantized);
  }
}

Status failure(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return Status::success();
  return Status(ErrorCode::upload_failed,
                std::string(operation) + ": " + cudaGetErrorString(error));
}

}  // namespace

Status admit_deepseek_projection(
    const DeepSeekAdmissionLaunch& launch) noexcept {
  if (launch.packed_fp4 == nullptr || launch.ue8m0_scales == nullptr ||
      launch.int8_rows == nullptr || launch.row_scales == nullptr ||
      launch.rows == 0U || launch.columns == 0U ||
      (launch.columns % 32U) != 0U) {
    return Status(ErrorCode::invalid_argument, "invalid DeepSeek admission launch");
  }
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  unsigned* invalid = nullptr;
  auto error = cudaMallocAsync(&invalid, sizeof(unsigned), stream);
  if (error == cudaSuccess) error = cudaMemsetAsync(invalid, 0, sizeof(unsigned), stream);
  if (error != cudaSuccess) return failure(error, "DeepSeek validity allocation");
  compact_to_int8<<<launch.rows, kThreads, 0, stream>>>(
      launch.packed_fp4, launch.ue8m0_scales, launch.int8_rows,
      launch.row_scales, launch.rows, launch.columns, invalid);
  error = cudaPeekAtLastError();
  unsigned host_invalid = 0U;
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(&host_invalid, invalid, sizeof(unsigned),
                            cudaMemcpyDeviceToHost, stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  static_cast<void>(cudaFreeAsync(invalid, stream));
  static_cast<void>(cudaStreamSynchronize(stream));
  if (error != cudaSuccess) return failure(error, "DeepSeek admission");
  if (host_invalid != 0U) {
    return Status(ErrorCode::checksum_mismatch, "DeepSeek source contains UE8M0 NaN");
  }
  return Status::success();
}

}  // namespace expert::runtime::cuda
