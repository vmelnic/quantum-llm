#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned kMaximumRouterTopK = 32;
constexpr unsigned kMaximumWeightReuseBatch = 8;
constexpr unsigned kFp4GemmBatchTile = 64;
constexpr unsigned kFp4GemmOutputTile = 32;
constexpr unsigned kFp4GemmBlockColumns = 32;
constexpr unsigned kFp4GemmThreads = 256;
constexpr unsigned kFp4GemmWarps = kFp4GemmThreads / kWarpSize;
constexpr unsigned kTensorCoreAttentionThreads = 512;
// A 16-token tile keeps decoded K/V values within the default 48-KiB SM86
// shared-memory limit. FP4 values times UE8M0 power-of-two scales are exactly
// representable in BF16, so the attention kernels can halve shared traffic
// without adding another quantization step.
constexpr unsigned kAttentionKvTile = 16;
constexpr unsigned kMaximumAttentionHeadDim = 256;
constexpr float kNegativeInfinity = -3.402823466e+38F;

__device__ float warp_sum(float value) {
  for (unsigned offset = kWarpSize / 2; offset; offset >>= 1U)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__device__ float warp_max(float value) {
  for (unsigned offset = kWarpSize / 2U; offset; offset >>= 1U)
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  return value;
}

__device__ float block_sum_warps(float value, float* workspace) {
  const auto lane = threadIdx.x % kWarpSize;
  const auto warp = threadIdx.x / kWarpSize;
  value = warp_sum(value);
  if (lane == 0U) workspace[warp] = value;
  __syncthreads();
  value = warp == 0U && lane < kWarpsPerBlock ? workspace[lane] : 0.0F;
  if (warp == 0U) value = warp_sum(value);
  if (threadIdx.x == 0U) workspace[0] = value;
  __syncthreads();
  return workspace[0];
}

__device__ float reduce_sum(float value) {
  __shared__ float shared[kThreads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

__device__ std::int8_t decode_fp4_twice(std::uint8_t code) {
  const auto index = code & 0x07U;
  const auto magnitude = index <= 4U ? static_cast<int>(index)
                         : index == 5U ? 6
                         : index == 6U ? 8
                                       : 12;
  return static_cast<std::int8_t>((code & 0x08U) != 0U ? -magnitude
                                                       : magnitude);
}

__device__ float decode_ue8m0(std::uint8_t code) {
  return code == 0U ? __uint_as_float(0x00400000U)
                    : __uint_as_float(static_cast<unsigned>(code) << 23U);
}

__device__ std::uint8_t encode_ue8m0_cover(float maximum) {
  if (maximum == 0.0F) return 127U;
  const auto exponent = static_cast<int>(ceilf(log2f(maximum / 6.0F)));
  return static_cast<std::uint8_t>(max(1, min(254, exponent + 127)));
}

__device__ std::uint8_t encode_fp4_nearest(float value, float scale) {
  const auto magnitude = fabsf(value / scale);
  std::uint8_t index{};
  if (magnitude <= 0.25F) index = 0U;
  else if (magnitude < 0.75F) index = 1U;
  else if (magnitude <= 1.25F) index = 2U;
  else if (magnitude < 1.75F) index = 3U;
  else if (magnitude <= 2.5F) index = 4U;
  else if (magnitude < 3.5F) index = 5U;
  else if (magnitude <= 5.0F) index = 6U;
  else index = 7U;
  return static_cast<std::uint8_t>(index | (value < 0.0F ? 8U : 0U));
}

__device__ int packed_fp4x4(std::uint8_t first, std::uint8_t second) {
  const auto selectors = static_cast<std::uint32_t>(first) |
                         (static_cast<std::uint32_t>(second) << 8U);
  const auto magnitudes =
      __byte_perm(0x03020100U, 0x0c080604U, selectors & 0x7777U);
  auto a = static_cast<int>(magnitudes & 0xffU);
  auto b = static_cast<int>((magnitudes >> 8U) & 0xffU);
  auto c = static_cast<int>((magnitudes >> 16U) & 0xffU);
  auto d = static_cast<int>((magnitudes >> 24U) & 0xffU);
  if ((first & 0x08U) != 0U) a = -a;
  if ((first & 0x80U) != 0U) b = -b;
  if ((second & 0x08U) != 0U) c = -c;
  if ((second & 0x80U) != 0U) d = -d;
  return static_cast<int>(static_cast<std::uint8_t>(a)) |
         (static_cast<int>(static_cast<std::uint8_t>(b)) << 8U) |
         (static_cast<int>(static_cast<std::uint8_t>(c)) << 16U) |
         (static_cast<int>(static_cast<std::uint8_t>(d)) << 24U);
}

__global__ void fp4_embedding_kernel(
    const std::uint8_t* weights, const std::uint8_t* scales,
    std::uint32_t token, std::uint32_t columns,
    std::uint32_t padded_columns, float* output) {
  const auto* row = weights +
      static_cast<std::size_t>(token) * (padded_columns / 2U);
  const auto* row_scales = scales +
      static_cast<std::size_t>(token) * (padded_columns / 32U);
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const auto packed = row[column / 2U];
    const auto code = (column & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
    output[column] = static_cast<float>(decode_fp4_twice(code)) * 0.5F *
                     decode_ue8m0(row_scales[column / 32U]);
  }
}

__global__ void fp4_embedding_batch_kernel(
    const std::uint8_t* weights, const std::uint8_t* scales,
    const std::uint32_t* tokens, std::uint32_t vocabulary,
    std::uint32_t columns, std::uint32_t padded_columns, float* output) {
  const auto request = static_cast<std::uint32_t>(blockIdx.x);
  const auto token = tokens[request];
  if (token >= vocabulary) return;
  const auto* row =
      weights + static_cast<std::size_t>(token) * (padded_columns / 2U);
  const auto* row_scales =
      scales + static_cast<std::size_t>(token) * (padded_columns / 32U);
  auto* target = output + static_cast<std::size_t>(request) * columns;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    const auto byte = row[column / 2U];
    const auto code = static_cast<std::uint8_t>(
        (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
    target[column] = static_cast<float>(decode_fp4_twice(code)) * 0.5F *
                     decode_ue8m0(row_scales[column / 32U]);
  }
}

__global__ void quantize_q8_batch_kernel(
    const float* input, std::int8_t* output, float* scales,
    std::uint32_t columns, std::uint32_t padded_columns) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  const auto* source = input + static_cast<std::size_t>(row) * columns;
  auto* target = output + static_cast<std::size_t>(row) * padded_columns;
  __shared__ float maxima[kThreads];
  float maximum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(source[column]));
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x],
                                  maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const auto scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) scales[row] = scale;
  __syncthreads();
  for (std::uint32_t column = threadIdx.x; column < padded_columns;
       column += blockDim.x) {
    auto quantized = column < columns
                         ? __float2int_rn(source[column] / scale)
                         : 0;
    quantized = max(-127, min(127, quantized));
    target[column] = static_cast<std::int8_t>(quantized);
  }
}

__device__ float fp4_q8_dot(
    const std::uint8_t* weights, const std::uint8_t* scales,
    const std::int8_t* activation, float activation_scale,
    std::uint32_t row, std::uint32_t padded_columns) {
  const auto lane = threadIdx.x % kWarpSize;
  const auto blocks = padded_columns / 32U;
  const auto* row_weights = reinterpret_cast<const int4*>(
      weights + static_cast<std::size_t>(row) * (padded_columns / 2U));
  const auto* row_scales = scales +
      static_cast<std::size_t>(row) * blocks;
  const auto* activation_blocks =
      reinterpret_cast<const int4*>(activation);
  float total = 0.0F;
  for (std::uint32_t block = lane; block < blocks; block += kWarpSize) {
    const auto packed_weights = row_weights[block];
    const auto activation_low = activation_blocks[2U * block];
    const auto activation_high = activation_blocks[2U * block + 1U];
    const auto weight_scale = decode_ue8m0(row_scales[block]);
    const std::uint32_t weight_words[]{
        static_cast<std::uint32_t>(packed_weights.x),
        static_cast<std::uint32_t>(packed_weights.y),
        static_cast<std::uint32_t>(packed_weights.z),
        static_cast<std::uint32_t>(packed_weights.w)};
    const int activation_words[]{
        activation_low.x, activation_low.y, activation_low.z,
        activation_low.w, activation_high.x, activation_high.y,
        activation_high.z, activation_high.w};
    int block_total = 0;
#pragma unroll
    for (std::uint32_t group = 0U; group < 8U; ++group) {
      const auto pair = static_cast<std::uint16_t>(
          weight_words[group / 2U] >> (16U * (group & 1U)));
      const auto packed = packed_fp4x4(
          static_cast<std::uint8_t>(pair),
          static_cast<std::uint8_t>(pair >> 8U));
      block_total = __dp4a(packed, activation_words[group], block_total);
    }
    total += static_cast<float>(block_total) * weight_scale;
  }
  return warp_sum(total) * activation_scale * 0.5F;
}

__global__ void fp4_gemv_q8_batch_kernel(
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    const std::int8_t* input, const float* input_scales, float* output,
    std::uint32_t rows, std::uint32_t padded_columns,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto item = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock +
                    warp;
  const auto total = static_cast<std::uint64_t>(rows) * batch;
  if (item >= total) return;
  const auto request = static_cast<std::uint32_t>(item / rows);
  const auto row = static_cast<std::uint32_t>(item % rows);
  const auto* activation = input +
      static_cast<std::size_t>(request) * padded_columns;
  const auto value = fp4_q8_dot(
      weights, weight_scales, activation, input_scales[request], row,
      padded_columns);
  if (threadIdx.x % kWarpSize == 0U)
    output[static_cast<std::size_t>(request) * rows + row] = value;
}

template <std::uint32_t Batch>
__global__ void fp4_gemv_q8_batch_weight_reuse_kernel(
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    const std::int8_t* input, const float* input_scales, float* output,
    std::uint32_t rows, std::uint32_t padded_columns) {
  static_assert(Batch > 0U && Batch <= kMaximumWeightReuseBatch);
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(
      blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  const auto blocks = padded_columns / 32U;
  const auto* row_weights = reinterpret_cast<const int4*>(
      weights + static_cast<std::size_t>(row) * (padded_columns / 2U));
  const auto* row_scales = weight_scales +
      static_cast<std::size_t>(row) * blocks;
  float totals[Batch]{};
  const int* activation_blocks[Batch];
#pragma unroll
  for (std::uint32_t request = 0U; request < Batch; ++request)
    activation_blocks[request] = reinterpret_cast<const int*>(
        input + static_cast<std::size_t>(request) * padded_columns);
  for (std::uint32_t block = lane; block < blocks; block += kWarpSize) {
    const auto packed_weights = row_weights[block];
    const auto weight_scale = decode_ue8m0(row_scales[block]);
    const std::uint32_t weight_words[]{
        static_cast<std::uint32_t>(packed_weights.x),
        static_cast<std::uint32_t>(packed_weights.y),
        static_cast<std::uint32_t>(packed_weights.z),
        static_cast<std::uint32_t>(packed_weights.w)};
    int block_totals[Batch]{};
#pragma unroll
    for (std::uint32_t group = 0U; group < 8U; ++group) {
      const auto pair = static_cast<std::uint16_t>(
          weight_words[group / 2U] >> (16U * (group & 1U)));
      const auto packed = packed_fp4x4(
          static_cast<std::uint8_t>(pair),
          static_cast<std::uint8_t>(pair >> 8U));
#pragma unroll
      for (std::uint32_t request = 0U; request < Batch; ++request) {
        block_totals[request] =
            __dp4a(packed, activation_blocks[request][block * 8U + group],
                   block_totals[request]);
      }
    }
#pragma unroll
    for (std::uint32_t request = 0U; request < Batch; ++request)
      totals[request] +=
          static_cast<float>(block_totals[request]) * weight_scale;
  }
#pragma unroll
  for (std::uint32_t request = 0U; request < Batch; ++request) {
    const auto value = warp_sum(totals[request]) *
                       input_scales[request] * 0.5F;
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] = value;
  }
}

// A CTA owns [up to 64 activation rows, 32 output rows]. Q8 integers and
// FP4-twice values multiplied by their power-of-two block scales are exactly
// representable in BF16. Baking the weight scale into the BF16 operand lets
// each warp keep one FP32 accumulator fragment across the complete K axis;
// the old integer path had to store, rescale, and reload a 64x32 product after
// every K=32 block.
__global__ void fp4_gemm_q8_block32_kernel(
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    const std::int8_t* input, const float* input_scales, float* output,
    std::uint32_t output_rows, std::uint32_t padded_columns,
    std::uint32_t batch) {
  using namespace nvcuda;
  __shared__ __align__(32) __nv_bfloat16
      activation_tile[kFp4GemmBatchTile][kFp4GemmBlockColumns];
  __shared__ __align__(32) __nv_bfloat16
      weight_tile[kFp4GemmOutputTile][kFp4GemmBlockColumns];
  __shared__ __align__(32) float
      product_tile[kFp4GemmBatchTile][kFp4GemmOutputTile];

  const auto local_thread = static_cast<std::uint32_t>(threadIdx.x);
  const auto warp = local_thread / kWarpSize;
  const auto first_batch =
      static_cast<std::uint32_t>(blockIdx.y) * kFp4GemmBatchTile;
  const auto first_output =
      static_cast<std::uint32_t>(blockIdx.x) * kFp4GemmOutputTile;
  wmma::fragment<wmma::matrix_a, 16, 16, 16,
                 __nv_bfloat16, wmma::row_major>
      activation_fragment;
  wmma::fragment<wmma::matrix_b, 16, 16, 16,
                 __nv_bfloat16, wmma::col_major>
      weight_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float>
      product_fragment;
  wmma::fill_fragment(product_fragment, 0.0F);
  const auto batch_warp = warp / 2U;
  const auto output_warp = warp % 2U;

  const auto blocks = padded_columns / kFp4GemmBlockColumns;
  for (std::uint32_t block = 0U; block < blocks; ++block) {
    for (std::uint32_t item = local_thread;
         item < kFp4GemmBatchTile * (kFp4GemmBlockColumns / 4U);
         item += kFp4GemmThreads) {
      const auto row = item / (kFp4GemmBlockColumns / 4U);
      const auto group = item % (kFp4GemmBlockColumns / 4U);
      const auto source_row = first_batch + row;
      char4 values{};
      if (source_row < batch)
        values = *reinterpret_cast<const char4*>(
            input + static_cast<std::size_t>(source_row) * padded_columns +
            block * kFp4GemmBlockColumns + 4U * group);
      const auto column = 4U * group;
      activation_tile[row][column] =
          __float2bfloat16(static_cast<float>(values.x));
      activation_tile[row][column + 1U] =
          __float2bfloat16(static_cast<float>(values.y));
      activation_tile[row][column + 2U] =
          __float2bfloat16(static_cast<float>(values.z));
      activation_tile[row][column + 3U] =
          __float2bfloat16(static_cast<float>(values.w));
    }
    for (std::uint32_t item = local_thread;
         item < kFp4GemmOutputTile * (kFp4GemmBlockColumns / 8U);
         item += kFp4GemmThreads) {
      const auto row = item / (kFp4GemmBlockColumns / 8U);
      const auto group = item % (kFp4GemmBlockColumns / 8U);
      const auto source_row = first_output + row;
      std::uint32_t packed{};
      float scale{};
      if (source_row < output_rows) {
        packed = *reinterpret_cast<const std::uint32_t*>(
            weights + static_cast<std::size_t>(source_row) *
                          (padded_columns / 2U) +
            block * (kFp4GemmBlockColumns / 2U) + 4U * group);
        scale = decode_ue8m0(
            weight_scales[static_cast<std::size_t>(source_row) * blocks +
                          block]);
      }
#pragma unroll
      for (std::uint32_t byte = 0U; byte < 4U; ++byte) {
        const auto codes = static_cast<std::uint8_t>(packed >> (8U * byte));
        const auto column = 8U * group + 2U * byte;
        weight_tile[row][column] = __float2bfloat16(
            static_cast<float>(decode_fp4_twice(codes & 0x0fU)) * scale);
        weight_tile[row][column + 1U] = __float2bfloat16(
            static_cast<float>(decode_fp4_twice(codes >> 4U)) * scale);
      }
    }
    __syncthreads();

    wmma::load_matrix_sync(
        activation_fragment, &activation_tile[batch_warp * 16U][0],
        kFp4GemmBlockColumns);
    wmma::load_matrix_sync(weight_fragment,
                           &weight_tile[output_warp * 16U][0],
                           kFp4GemmBlockColumns);
    wmma::mma_sync(product_fragment, activation_fragment, weight_fragment,
                   product_fragment);
    wmma::load_matrix_sync(
        activation_fragment, &activation_tile[batch_warp * 16U][16],
        kFp4GemmBlockColumns);
    wmma::load_matrix_sync(weight_fragment,
                           &weight_tile[output_warp * 16U][16],
                           kFp4GemmBlockColumns);
    wmma::mma_sync(product_fragment, activation_fragment, weight_fragment,
                   product_fragment);
    __syncthreads();
  }

  wmma::store_matrix_sync(
      &product_tile[batch_warp * 16U][output_warp * 16U],
      product_fragment, kFp4GemmOutputTile, wmma::mem_row_major);
  __syncthreads();

  for (std::uint32_t local = 0U;
       local < kFp4GemmBatchTile * kFp4GemmOutputTile /
                   kFp4GemmThreads;
       ++local) {
    const auto item = local_thread + local * kFp4GemmThreads;
    const auto batch_row = first_batch + item / kFp4GemmOutputTile;
    const auto output_row = first_output + item % kFp4GemmOutputTile;
    if (batch_row < batch && output_row < output_rows)
      output[static_cast<std::size_t>(batch_row) * output_rows + output_row] =
          product_tile[item / kFp4GemmOutputTile]
                      [item % kFp4GemmOutputTile] *
          input_scales[batch_row] * 0.5F;
  }
}

__global__ void embedding_kernel(const std::int8_t* weights,
                                 const float* scales, std::uint32_t token,
                                 std::uint32_t columns, float* output) {
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) {
    output[i] = static_cast<float>(weights[static_cast<std::size_t>(token) * columns + i]) * scales[token];
  }
}

__global__ void int8_gemv_kernel(const std::int8_t* weights,
                                 const float* scales, const float* input,
                                 float* output, std::uint32_t rows,
                                 std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize) {
    partial += static_cast<float>(weight[i]) * input[i];
  }
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial * scales[row];
}

__global__ void int8_gemv_vector_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(
      blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    const auto activation =
        *reinterpret_cast<const float4*>(input + column);
    partial += static_cast<float>(packed.x) * activation.x;
    partial += static_cast<float>(packed.y) * activation.y;
    partial += static_cast<float>(packed.z) * activation.z;
    partial += static_cast<float>(packed.w) * activation.w;
  }
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial * scales[row];
}

__global__ void int8_gemv_grouped_inputs_vector_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t rows_per_group) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(
      blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  const auto group = row / rows_per_group;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(group) * columns;
  float partial = 0.0F;
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    const auto values =
        *reinterpret_cast<const float4*>(activation + column);
    partial += static_cast<float>(packed.x) * values.x;
    partial += static_cast<float>(packed.y) * values.y;
    partial += static_cast<float>(packed.z) * values.z;
    partial += static_cast<float>(packed.w) * values.w;
  }
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial * scales[row];
}

__global__ void int8_gemv_grouped_inputs_batch_reuse_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t rows_per_group, std::uint32_t groups,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto group = row / rows_per_group;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  float partial[kMaximumWeightReuseBatch]{};
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    for (std::uint32_t request = 0U; request < batch; ++request) {
      const auto* activation = input +
          (static_cast<std::size_t>(request) * groups + group) * columns;
      const auto values =
          *reinterpret_cast<const float4*>(activation + column);
      partial[request] += static_cast<float>(packed.x) * values.x;
      partial[request] += static_cast<float>(packed.y) * values.y;
      partial[request] += static_cast<float>(packed.z) * values.z;
      partial[request] += static_cast<float>(packed.w) * values.w;
    }
  }
  for (std::uint32_t request = 0U; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] =
          sum * scales[row];
  }
}

__global__ void int8_gemv_batch_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto work = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const auto row = static_cast<std::uint32_t>(work / batch);
  const auto request = static_cast<std::uint32_t>(work % batch);
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(request) * columns;
  float partial = 0.0F;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += static_cast<float>(weight[i]) * activation[i];
  partial = warp_sum(partial);
  if (lane == 0)
    output[static_cast<std::size_t>(request) * rows + row] =
        partial * scales[row];
}

// Vectorized twin of int8_gemv_batch_kernel: identical scales/groups
// semantics, but each lane consumes four weights and four activations per
// iteration through char4/float4 loads. Requires columns % 4 == 0; row and
// request strides then stay naturally aligned for both vector types.
__global__ void int8_gemv_batch_vector_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto work = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const auto row = static_cast<std::uint32_t>(work / batch);
  const auto request = static_cast<std::uint32_t>(work % batch);
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(request) * columns;
  float partial = 0.0F;
  for (std::uint32_t column = lane * 4U; column < columns;
       column += kWarpSize * 4U) {
    const auto packed = *reinterpret_cast<const char4*>(weight + column);
    const auto values =
        *reinterpret_cast<const float4*>(activation + column);
    partial += static_cast<float>(packed.x) * values.x;
    partial += static_cast<float>(packed.y) * values.y;
    partial += static_cast<float>(packed.z) * values.z;
    partial += static_cast<float>(packed.w) * values.w;
  }
  partial = warp_sum(partial);
  if (lane == 0)
    output[static_cast<std::size_t>(request) * rows + row] =
        partial * scales[row];
}

__global__ void int8_gemv_batch_weight_reuse_kernel(
    const std::int8_t* weights, const float* scales, const float* input,
    float* output, std::uint32_t rows, std::uint32_t columns,
    std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  float partial[kMaximumWeightReuseBatch]{};
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t column = lane; column < columns; column += kWarpSize) {
    const auto value = static_cast<float>(weight[column]);
    for (std::uint32_t request = 0; request < batch; ++request) {
      partial[request] +=
          value * input[static_cast<std::size_t>(request) * columns + column];
    }
  }
  for (std::uint32_t request = 0; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0) {
      output[static_cast<std::size_t>(request) * rows + row] =
          sum * scales[row];
    }
  }
}

__global__ void f32_gemv_kernel(const float* weights, const float* input,
                                float* output, std::uint32_t rows,
                                std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += weight[i] * input[i];
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial;
}

__global__ void f32_gemv_batch_kernel(
    const float* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto work = static_cast<std::uint64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const auto row = static_cast<std::uint32_t>(work / batch);
  const auto request = static_cast<std::uint32_t>(work % batch);
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  const auto* activation = input + static_cast<std::size_t>(request) * columns;
  float partial = 0.0F;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += weight[i] * activation[i];
  partial = warp_sum(partial);
  if (lane == 0)
    output[static_cast<std::size_t>(request) * rows + row] = partial;
}

__global__ void f32_gemv_batch_reuse_kernel(
    const float* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  float partial[kMaximumWeightReuseBatch]{};
  for (std::uint32_t column = lane; column < columns;
       column += kWarpSize) {
    const auto value = weight[column];
    for (std::uint32_t request = 0U; request < batch; ++request)
      partial[request] += value *
          input[static_cast<std::size_t>(request) * columns + column];
  }
  for (std::uint32_t request = 0U; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] = sum;
  }
}

__global__ void bf16_gemv_kernel(const std::uint16_t* weights,
                                  const float* input, float* output,
                                  std::uint32_t rows,
                                  std::uint32_t columns) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(blockIdx.x * kWarpsPerBlock + warp);
  if (row >= rows) return;
  float partial = 0.0F;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  for (std::uint32_t i = lane; i < columns; i += kWarpSize)
    partial += __uint_as_float(static_cast<unsigned>(weight[i]) << 16U) * input[i];
  partial = warp_sum(partial);
  if (lane == 0) output[row] = partial;
}

__global__ void bf16_gemv_batch_reuse_kernel(
    const std::uint16_t* weights, const float* input, float* output,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t batch) {
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row =
      static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const auto* weight = weights + static_cast<std::size_t>(row) * columns;
  float partial[kMaximumWeightReuseBatch]{};
  for (std::uint32_t column = lane; column < columns;
       column += kWarpSize) {
    const auto value =
        __uint_as_float(static_cast<unsigned>(weight[column]) << 16U);
    for (std::uint32_t request = 0U; request < batch; ++request)
      partial[request] += value *
          input[static_cast<std::size_t>(request) * columns + column];
  }
  for (std::uint32_t request = 0U; request < batch; ++request) {
    const auto sum = warp_sum(partial[request]);
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] = sum;
  }
}

__global__ void rms_kernel(const float* input, const float* weight,
                           float* output, std::uint32_t count, float epsilon) {
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) output[i] = input[i] * inverse * weight[i];
}

__global__ void rms_bf16_weight_kernel(const float* input,
                                        const std::uint16_t* weight,
                                        float* output, std::uint32_t count,
                                        float epsilon) {
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const float scale =
        __uint_as_float(static_cast<unsigned>(weight[i]) << 16U);
    output[i] = input[i] * inverse * scale;
  }
}

__global__ void rms_bf16_weight_batch_kernel(
    const float* input, const std::uint16_t* weight, float* output,
    std::uint32_t count, float epsilon) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  input += static_cast<std::size_t>(row) * count;
  output += static_cast<std::size_t>(row) * count;
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const float scale =
        __uint_as_float(static_cast<unsigned>(weight[i]) << 16U);
    output[i] = input[i] * inverse * scale;
  }
}

__global__ void qwen_rms_kernel(const float* input, const float* weight,
                                float* output, std::uint32_t count,
                                float epsilon) {
  float square = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    square += input[i] * input[i];
  square = reduce_sum(square);
  const float inverse = rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x)
    output[i] = input[i] * inverse * (1.0F + weight[i]);
}

__global__ void zero_centered_rms_batch_kernel(
    const float* input, const float* weight, float* output,
    std::uint32_t count, float epsilon) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  input += static_cast<std::size_t>(row) * count;
  output += static_cast<std::size_t>(row) * count;
  float square = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x)
    square += input[index] * input[index];
  square = reduce_sum(square);
  const auto inverse =
      rsqrtf(square / static_cast<float>(count) + epsilon);
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x)
    output[index] = input[index] * inverse * (1.0F + weight[index]);
}

__global__ void add_kernel(float* destination, const float* source,
                           std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) destination[i] += source[i];
}

__global__ void silu_product_kernel(const float* gate, const float* up,
                                    float* output, std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) output[i] = (gate[i] / (1.0F + expf(-gate[i]))) * up[i];
}

__global__ void sigmoid_scale_kernel(float* values, const float* gate,
                                     std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) values[i] *= 1.0F / (1.0F + expf(-gate[0]));
}

__global__ void qkv_rope_kernel(float* query, float* key, const float* value,
                                float* key_cache, float* value_cache,
                                std::uint32_t position, std::uint32_t heads,
                                std::uint32_t head_dim, float theta) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head >= heads || dimension >= head_dim) return;
  const auto base = static_cast<std::size_t>(head) * head_dim;
  __shared__ float normalized_q[128];
  __shared__ float normalized_k[128];
  normalized_q[dimension] = query[base + dimension];
  normalized_k[dimension] = key[base + dimension];
  __syncthreads();
  const auto half = head_dim / 2U;
  const auto frequency_index = dimension % half;
  const float inverse_frequency = powf(theta, -2.0F * static_cast<float>(frequency_index) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(position) * inverse_frequency;
  const float cosine = cosf(angle), sine = sinf(angle);
  const float q_rotated = dimension < half ? -normalized_q[dimension + half] : normalized_q[dimension - half];
  const float k_rotated = dimension < half ? -normalized_k[dimension + half] : normalized_k[dimension - half];
  query[base + dimension] = normalized_q[dimension] * cosine + q_rotated * sine;
  const float final_key = normalized_k[dimension] * cosine + k_rotated * sine;
  key[base + dimension] = final_key;
  const auto cache = (static_cast<std::size_t>(position) * heads + head) * head_dim + dimension;
  key_cache[cache] = final_key;
  value_cache[cache] = value[base + dimension];
}

__global__ void attention_kernel(const float* query, const float* key_cache,
                                 const float* value_cache, float* output,
                                 std::uint32_t tokens, std::uint32_t heads,
                                 std::uint32_t head_dim) {
  extern __shared__ float scores[];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto q_base = static_cast<std::size_t>(head) * head_dim;
  for (std::uint32_t token = threadIdx.x; token < tokens; token += blockDim.x) {
    const auto cache_base = (static_cast<std::size_t>(token) * heads + head) * head_dim;
    float dot = 0.0F;
    for (std::uint32_t d = 0; d < head_dim; ++d) dot += query[q_base + d] * key_cache[cache_base + d];
    scores[token] = dot * rsqrtf(static_cast<float>(head_dim));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float maximum = kNegativeInfinity;
    for (std::uint32_t token = 0; token < tokens; ++token) maximum = fmaxf(maximum, scores[token]);
    float denominator = 0.0F;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      scores[token] = expf(scores[token] - maximum);
      denominator += scores[token];
    }
    scores[tokens] = denominator;
  }
  __syncthreads();
  if (dimension < head_dim) {
    float result = 0.0F;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      const auto cache = (static_cast<std::size_t>(token) * heads + head) * head_dim + dimension;
      result += (scores[token] / scores[tokens]) * value_cache[cache];
    }
    output[q_base + dimension] = result;
  }
}

__global__ void router_logits_kernel(const float* input, const float* weights,
                                     float* logits, std::uint32_t hidden,
                                     std::uint32_t experts) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x);
  if (expert >= experts) return;
  float partial = 0.0F;
  const auto* row = weights + static_cast<std::size_t>(expert) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) partial += row[i] * input[i];
  partial = reduce_sum(partial);
  if (threadIdx.x == 0) logits[expert] = partial;
}

__global__ void router_select_kernel(const float* logits, std::uint32_t experts,
                                     std::uint32_t top_k, float* scores,
                                     std::uint32_t* indices,
                                     bool normalize_selected) {
  if (threadIdx.x != 0) return;
  if (normalize_selected) {
    float selected_maximum = kNegativeInfinity;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      float best = kNegativeInfinity;
      std::uint32_t best_index = 0;
      for (std::uint32_t i = 0; i < experts; ++i) {
        bool used = false;
        for (std::uint32_t previous = 0; previous < slot; ++previous)
          used |= indices[previous] == i;
        if (!used && logits[i] > best) {
          best = logits[i];
          best_index = i;
        }
      }
      scores[slot] = best;
      indices[slot] = best_index;
      selected_maximum = fmaxf(selected_maximum, best);
    }
    float selected_sum = 0.0F;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      scores[slot] = expf(scores[slot] - selected_maximum);
      selected_sum += scores[slot];
    }
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      scores[slot] /= selected_sum;
    return;
  }
  float maximum = kNegativeInfinity;
  for (std::uint32_t i = 0; i < experts; ++i) maximum = fmaxf(maximum, logits[i]);
  float denominator = 0.0F;
  for (std::uint32_t i = 0; i < experts; ++i) denominator += expf(logits[i] - maximum);
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    float best = -1.0F;
    std::uint32_t best_index = 0;
    for (std::uint32_t i = 0; i < experts; ++i) {
      bool used = false;
      for (std::uint32_t previous = 0; previous < slot; ++previous) used |= indices[previous] == i;
      const float probability = expf(logits[i] - maximum) / denominator;
      if (!used && probability > best) { best = probability; best_index = i; }
    }
    scores[slot] = best;
    indices[slot] = best_index;
  }
}

__global__ void router_select_batch_kernel(
    const float* logits, std::uint32_t experts, std::uint32_t top_k,
    float* scores, std::uint32_t* indices) {
  const auto request = static_cast<std::uint32_t>(blockIdx.x);
  logits += static_cast<std::size_t>(request) * experts;
  scores += static_cast<std::size_t>(request) * top_k;
  indices += static_cast<std::size_t>(request) * top_k;
  __shared__ float candidate_values[kThreads];
  __shared__ std::uint32_t candidate_indices[kThreads];
  __shared__ float selected_values[kMaximumRouterTopK];
  __shared__ std::uint32_t selected_indices[kMaximumRouterTopK];
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    float best = kNegativeInfinity;
    std::uint32_t best_index = 0xffffffffU;
    for (std::uint32_t i = threadIdx.x; i < experts; i += blockDim.x) {
      bool used = false;
      for (std::uint32_t previous = 0; previous < slot; ++previous)
        used |= selected_indices[previous] == i;
      if (!used && (logits[i] > best ||
                    (logits[i] == best && i < best_index))) {
        best = logits[i];
        best_index = i;
      }
    }
    candidate_values[threadIdx.x] = best;
    candidate_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
      if (threadIdx.x < stride) {
        const auto other_value = candidate_values[threadIdx.x + stride];
        const auto other_index = candidate_indices[threadIdx.x + stride];
        if (other_value > candidate_values[threadIdx.x] ||
            (other_value == candidate_values[threadIdx.x] &&
             other_index < candidate_indices[threadIdx.x])) {
          candidate_values[threadIdx.x] = other_value;
          candidate_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected_values[slot] = candidate_values[0];
      selected_indices[slot] = candidate_indices[0];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float selected_maximum = kNegativeInfinity;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      selected_maximum = fmaxf(selected_maximum, selected_values[slot]);
    float selected_sum = 0.0F;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      scores[slot] = expf(selected_values[slot] - selected_maximum);
      indices[slot] = selected_indices[slot];
      selected_sum += scores[slot];
    }
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      scores[slot] /= selected_sum;
  }
}

__global__ void sigmoid_bias_router_select_batch_kernel(
    const float* logits, const float* expert_bias, std::uint32_t experts,
    std::uint32_t top_k, float normalization_epsilon,
    float routed_scaling_factor, float* scores, std::uint32_t* indices) {
  const auto request = static_cast<std::uint32_t>(blockIdx.x);
  logits += static_cast<std::size_t>(request) * experts;
  scores += static_cast<std::size_t>(request) * top_k;
  indices += static_cast<std::size_t>(request) * top_k;
  __shared__ float candidate_values[kThreads];
  __shared__ std::uint32_t candidate_indices[kThreads];
  __shared__ float selected_weights[kMaximumRouterTopK];
  __shared__ std::uint32_t selected_indices[kMaximumRouterTopK];
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    float best = kNegativeInfinity;
    std::uint32_t best_index = 0xffffffffU;
    for (std::uint32_t expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
      bool used = false;
      for (std::uint32_t previous = 0; previous < slot; ++previous)
        used |= selected_indices[previous] == expert;
      const float unbiased = 1.0F / (1.0F + expf(-logits[expert]));
      const float selection = unbiased + expert_bias[expert];
      if (!used &&
          (selection > best ||
           (selection == best && expert < best_index))) {
        best = selection;
        best_index = expert;
      }
    }
    candidate_values[threadIdx.x] = best;
    candidate_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
      if (threadIdx.x < stride) {
        const auto other_value = candidate_values[threadIdx.x + stride];
        const auto other_index = candidate_indices[threadIdx.x + stride];
        if (other_value > candidate_values[threadIdx.x] ||
            (other_value == candidate_values[threadIdx.x] &&
             other_index < candidate_indices[threadIdx.x])) {
          candidate_values[threadIdx.x] = other_value;
          candidate_indices[threadIdx.x] = other_index;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const auto selected = candidate_indices[0];
      selected_indices[slot] = selected;
      selected_weights[slot] =
          1.0F / (1.0F + expf(-logits[selected]));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float denominator = normalization_epsilon;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      denominator += selected_weights[slot];
    const float multiplier = routed_scaling_factor / denominator;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      indices[slot] = selected_indices[slot];
      scores[slot] = selected_weights[slot] * multiplier;
    }
  }
}

__device__ float deepseek_route_score(float logit) {
  const float softplus = logit > 20.0F ? logit : log1pf(expf(logit));
  return sqrtf(softplus);
}

__global__ void deepseek_hash_router_kernel(
    const float* logits, const std::int64_t* token_experts,
    std::uint32_t token_id, float route_scale, float* scores,
    std::uint32_t* indices) {
  const auto slot = static_cast<std::uint32_t>(threadIdx.x);
  __shared__ float selected[6];
  if (slot < 6U) {
    const auto expert = token_experts[
        static_cast<std::size_t>(token_id) * 6U + slot];
    if (expert < 0 || expert >= 256) {
      indices[slot] = 0xffffffffU;
      selected[slot] = 0.0F;
    } else {
      indices[slot] = static_cast<std::uint32_t>(expert);
      selected[slot] = deepseek_route_score(logits[expert]);
    }
  }
  __syncthreads();
  if (slot == 0U) {
    float total = 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      total += selected[index];
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      scores[index] = selected[index] * multiplier;
  }
}

__global__ void deepseek_learned_router_kernel(
    const float* logits, const float* bias, float route_scale,
    float* scores, std::uint32_t* indices) {
  __shared__ float candidates[kThreads];
  __shared__ std::uint32_t candidate_indices[kThreads];
  __shared__ float selected_scores[6];
  __shared__ std::uint32_t selected_indices[6];
  const auto expert = static_cast<std::uint32_t>(threadIdx.x);
  const float unbiased = deepseek_route_score(logits[expert]);
  for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
    bool used = false;
    for (std::uint32_t previous = 0U; previous < slot; ++previous)
      used |= selected_indices[previous] == expert;
    candidates[expert] = used ? kNegativeInfinity : unbiased + bias[expert];
    candidate_indices[expert] = expert;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride; stride >>= 1U) {
      if (expert < stride) {
        const auto other_value = candidates[expert + stride];
        const auto other_index = candidate_indices[expert + stride];
        if (other_value > candidates[expert] ||
            (other_value == candidates[expert] &&
             other_index < candidate_indices[expert])) {
          candidates[expert] = other_value;
          candidate_indices[expert] = other_index;
        }
      }
      __syncthreads();
    }
    if (expert == 0U) {
      selected_indices[slot] = candidate_indices[0];
      selected_scores[slot] =
          deepseek_route_score(logits[selected_indices[slot]]);
    }
    __syncthreads();
  }
  if (expert == 0U) {
    float total = 0.0F;
    for (const float value : selected_scores) total += value;
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
      indices[slot] = selected_indices[slot];
      scores[slot] = selected_scores[slot] * multiplier;
    }
  }
}

__global__ void deepseek_hash_router_batch_kernel(
    const float* logits, const std::int64_t* token_experts,
    std::uint32_t token_zero, std::uint32_t token_one, float route_scale,
    float* scores, std::uint32_t* indices) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  const auto slot = static_cast<std::uint32_t>(threadIdx.x);
  const auto token_id = row == 0U ? token_zero : token_one;
  logits += static_cast<std::size_t>(row) * 256U;
  scores += static_cast<std::size_t>(row) * 6U;
  indices += static_cast<std::size_t>(row) * 6U;
  __shared__ float selected[6];
  if (slot < 6U) {
    const auto expert = token_experts[
        static_cast<std::size_t>(token_id) * 6U + slot];
    if (expert < 0 || expert >= 256) {
      indices[slot] = 0xffffffffU;
      selected[slot] = 0.0F;
    } else {
      indices[slot] = static_cast<std::uint32_t>(expert);
      selected[slot] = deepseek_route_score(logits[expert]);
    }
  }
  __syncthreads();
  if (slot == 0U) {
    float total = 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      total += selected[index];
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t index = 0U; index < 6U; ++index)
      scores[index] = selected[index] * multiplier;
  }
}

__global__ void deepseek_learned_router_batch_kernel(
    const float* logits, const float* bias, float route_scale,
    float* scores, std::uint32_t* indices) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  const auto expert = static_cast<std::uint32_t>(threadIdx.x);
  logits += static_cast<std::size_t>(row) * 256U;
  scores += static_cast<std::size_t>(row) * 6U;
  indices += static_cast<std::size_t>(row) * 6U;
  __shared__ float candidates[kThreads];
  __shared__ std::uint32_t candidate_indices[kThreads];
  __shared__ float selected_scores[6];
  __shared__ std::uint32_t selected_indices[6];
  const float unbiased = deepseek_route_score(logits[expert]);
  for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
    bool used = false;
    for (std::uint32_t previous = 0U; previous < slot; ++previous)
      used |= selected_indices[previous] == expert;
    candidates[expert] = used ? kNegativeInfinity : unbiased + bias[expert];
    candidate_indices[expert] = expert;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride; stride >>= 1U) {
      if (expert < stride) {
        const auto other_value = candidates[expert + stride];
        const auto other_index = candidate_indices[expert + stride];
        if (other_value > candidates[expert] ||
            (other_value == candidates[expert] &&
             other_index < candidate_indices[expert])) {
          candidates[expert] = other_value;
          candidate_indices[expert] = other_index;
        }
      }
      __syncthreads();
    }
    if (expert == 0U) {
      selected_indices[slot] = candidate_indices[0];
      selected_scores[slot] =
          deepseek_route_score(logits[selected_indices[slot]]);
    }
    __syncthreads();
  }
  if (expert == 0U) {
    float total = 0.0F;
    for (const float value : selected_scores) total += value;
    const float multiplier = total > 0.0F ? route_scale / total : 0.0F;
    for (std::uint32_t slot = 0U; slot < 6U; ++slot) {
      indices[slot] = selected_indices[slot];
      scores[slot] = selected_scores[slot] * multiplier;
    }
  }
}

__global__ void qwen_qkv_rope_kernel(
    float* q_and_gate, float* key, const float* value,
    const float* q_weight, const float* k_weight, float* key_cache,
    float* value_cache, std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head < query_heads) {
    auto* query = q_and_gate + static_cast<std::size_t>(head) * 2U * head_dim;
    float square = dimension < head_dim ? query[dimension] * query[dimension] : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim) {
      const float normalized_value = query[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + q_weight[dimension]);
      normalized[dimension] = normalized_value;
    }
    __syncthreads();
    if (dimension < head_dim) {
      float final_query = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_query = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      query[dimension] = final_query;
    }
  }
  __syncthreads();
  if (head < kv_heads) {
    auto* key_head = key + static_cast<std::size_t>(head) * head_dim;
    float square = dimension < head_dim
                       ? key_head[dimension] * key_head[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim) {
      normalized[dimension] = key_head[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + k_weight[dimension]);
    }
    __syncthreads();
    if (dimension < head_dim) {
      float final_key = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_key = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      key_head[dimension] = final_key;
      const auto cache =
          (static_cast<std::size_t>(position) * kv_heads + head) * head_dim +
          dimension;
      key_cache[cache] = final_key;
      value_cache[cache] =
          value[static_cast<std::size_t>(head) * head_dim + dimension];
    }
  }
}

__global__ void qwen_attention_kernel(
    const float* q_and_gate, const float* key_cache,
    const float* value_cache, float* output, std::uint32_t tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  __shared__ float online_maximum;
  __shared__ float online_denominator;
  __shared__ float previous_scale;
  __shared__ float token_scale;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto kv_head = query_head / (query_heads / kv_heads);
  const auto query = q_and_gate +
      static_cast<std::size_t>(query_head) * 2U * head_dim;
  if (threadIdx.x == 0) {
    online_maximum = kNegativeInfinity;
    online_denominator = 0.0F;
  }
  __syncthreads();
  float result = 0.0F;
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto cache =
        (static_cast<std::size_t>(token) * kv_heads + kv_head) * head_dim;
    float dot = threadIdx.x < head_dim
                    ? query[threadIdx.x] * key_cache[cache + threadIdx.x]
                    : 0.0F;
    dot = reduce_sum(dot) * rsqrtf(static_cast<float>(head_dim));
    if (threadIdx.x == 0) {
      const float next_maximum = fmaxf(online_maximum, dot);
      previous_scale = expf(online_maximum - next_maximum);
      token_scale = expf(dot - next_maximum);
      online_denominator = online_denominator * previous_scale + token_scale;
      online_maximum = next_maximum;
    }
    __syncthreads();
    if (threadIdx.x < head_dim)
      result = result * previous_scale + token_scale *
          value_cache[cache + threadIdx.x];
    __syncthreads();
  }
  if (threadIdx.x < head_dim) {
    const float gate = query[head_dim + threadIdx.x];
    output[static_cast<std::size_t>(query_head) * head_dim + threadIdx.x] =
        (result / online_denominator) / (1.0F + expf(-gate));
  }
}

__global__ void gqa_qkv_rope_kernel(
    float* query, float* key, const float* value,
    const float* q_weight, const float* k_weight, float* key_cache,
    float* value_cache, std::uint32_t position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float theta) {
  __shared__ float normalized[kThreads];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head < query_heads) {
    auto* query_head =
        query + static_cast<std::size_t>(head) * head_dim;
    float square = dimension < head_dim
                       ? query_head[dimension] * query_head[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = query_head[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          q_weight[dimension];
    __syncthreads();
    if (dimension < head_dim) {
      float final_query = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half
                                ? -normalized[dimension + half]
                                : normalized[dimension - half];
        final_query = normalized[dimension] * cosf(angle) +
                      other * sinf(angle);
      }
      query_head[dimension] = final_query;
    }
  }
  __syncthreads();
  if (head < kv_heads) {
    auto* key_head = key + static_cast<std::size_t>(head) * head_dim;
    float square = dimension < head_dim
                       ? key_head[dimension] * key_head[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = key_head[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          k_weight[dimension];
    __syncthreads();
    if (dimension < head_dim) {
      float final_key = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half
                                ? -normalized[dimension + half]
                                : normalized[dimension - half];
        final_key = normalized[dimension] * cosf(angle) +
                    other * sinf(angle);
      }
      key_head[dimension] = final_key;
      const auto cache =
          (static_cast<std::size_t>(position) * kv_heads + head) * head_dim +
          dimension;
      key_cache[cache] = final_key;
      value_cache[cache] =
          value[static_cast<std::size_t>(head) * head_dim + dimension];
    }
  }
}

__global__ void gqa_attention_kernel(
    const float* query, const float* key_cache, const float* value_cache,
    float* output, std::uint32_t tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
  __shared__ float online_maximum;
  __shared__ float online_denominator;
  __shared__ float previous_scale;
  __shared__ float token_scale;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto kv_head = query_head / (query_heads / kv_heads);
  const auto* query_values =
      query + static_cast<std::size_t>(query_head) * head_dim;
  if (threadIdx.x == 0) {
    online_maximum = kNegativeInfinity;
    online_denominator = 0.0F;
  }
  __syncthreads();
  float result = 0.0F;
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto cache =
        (static_cast<std::size_t>(token) * kv_heads + kv_head) * head_dim;
    float dot = threadIdx.x < head_dim
                    ? query_values[threadIdx.x] *
                          key_cache[cache + threadIdx.x]
                    : 0.0F;
    dot = reduce_sum(dot) * rsqrtf(static_cast<float>(head_dim));
    if (threadIdx.x == 0) {
      const float next_maximum = fmaxf(online_maximum, dot);
      previous_scale = expf(online_maximum - next_maximum);
      token_scale = expf(dot - next_maximum);
      online_denominator = online_denominator * previous_scale + token_scale;
      online_maximum = next_maximum;
    }
    __syncthreads();
    if (threadIdx.x < head_dim)
      result = result * previous_scale + token_scale *
          value_cache[cache + threadIdx.x];
    __syncthreads();
  }
  if (threadIdx.x < head_dim)
    output[static_cast<std::size_t>(query_head) * head_dim + threadIdx.x] =
        result / online_denominator;
}

__global__ void causal_short_conv_decode_kernel(
    const float* projected, const float* weights, float* state,
    float* output, std::uint32_t hidden, std::uint32_t kernel) {
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto channel = static_cast<std::uint32_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (channel >= hidden) return;
  const auto projected_base = static_cast<std::size_t>(row) * 3U * hidden;
  const float b = projected[projected_base + channel];
  const float c = projected[projected_base + hidden + channel];
  const float x = projected[projected_base + 2U * hidden + channel];
  auto* channel_state = state +
      (static_cast<std::size_t>(row) * hidden + channel) * kernel;
  for (std::uint32_t index = 1U; index < kernel; ++index)
    channel_state[index - 1U] = channel_state[index];
  channel_state[kernel - 1U] = b * x;
  const auto* channel_weights =
      weights + static_cast<std::size_t>(channel) * kernel;
  float convolution = 0.0F;
  for (std::uint32_t index = 0U; index < kernel; ++index)
    convolution += channel_state[index] * channel_weights[index];
  output[static_cast<std::size_t>(row) * hidden + channel] = c * convolution;
}

__global__ void qwen_qkv_rope_paged_fp16_kernel(
    float* q_and_gate, float* key, const float* value,
    const float* q_weight, const float* k_weight, __half* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head < query_heads) {
    auto* query = q_and_gate + static_cast<std::size_t>(head) * 2U * head_dim;
    float square = dimension < head_dim ? query[dimension] * query[dimension] : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = query[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + q_weight[dimension]);
    __syncthreads();
    if (dimension < head_dim) {
      float final_query = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_query = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      query[dimension] = final_query;
    }
  }
  __syncthreads();
  if (head < kv_heads) {
    auto* key_head = key + static_cast<std::size_t>(head) * head_dim;
    float square = dimension < head_dim
                       ? key_head[dimension] * key_head[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = key_head[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + k_weight[dimension]);
    __syncthreads();
    if (dimension < head_dim) {
      float final_key = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_key = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      key_head[dimension] = final_key;
      const auto page_elements = static_cast<std::size_t>(page_tokens) *
                                 kv_heads * head_dim;
      auto* key_page = page + static_cast<std::size_t>(full_attention_layer) *
                                  2U * page_elements;
      auto* value_page = key_page + page_elements;
      const auto cache =
          (static_cast<std::size_t>(position % page_tokens) * kv_heads + head) *
              head_dim + dimension;
      key_page[cache] = __float2half_rn(final_key);
      value_page[cache] = __float2half_rn(
          value[static_cast<std::size_t>(head) * head_dim + dimension]);
    }
  }
}

__global__ void qwen_attention_paged_fp16_kernel(
    const float* q_and_gate, const void* const* page_table, float* output,
    std::uint32_t tokens, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
  __shared__ float online_maximum;
  __shared__ float online_denominator;
  __shared__ float previous_scale;
  __shared__ float token_scale;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto kv_head = query_head / (query_heads / kv_heads);
  const auto* query = q_and_gate +
      static_cast<std::size_t>(query_head) * 2U * head_dim;
  const auto page_elements = static_cast<std::size_t>(page_tokens) *
                             kv_heads * head_dim;
  if (threadIdx.x == 0) {
    online_maximum = kNegativeInfinity;
    online_denominator = 0.0F;
  }
  __syncthreads();
  float result = 0.0F;
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto* page = static_cast<const __half*>(page_table[token / page_tokens]);
    const auto* key_page = page +
        static_cast<std::size_t>(full_attention_layer) * 2U * page_elements;
    const auto* value_page = key_page + page_elements;
    const auto cache =
        (static_cast<std::size_t>(token % page_tokens) * kv_heads + kv_head) *
            head_dim + threadIdx.x;
    float dot = threadIdx.x < head_dim
                    ? query[threadIdx.x] * __half2float(key_page[cache])
                    : 0.0F;
    dot = reduce_sum(dot) * rsqrtf(static_cast<float>(head_dim));
    if (threadIdx.x == 0) {
      const float next_maximum = fmaxf(online_maximum, dot);
      previous_scale = expf(online_maximum - next_maximum);
      token_scale = expf(dot - next_maximum);
      online_denominator = online_denominator * previous_scale + token_scale;
      online_maximum = next_maximum;
    }
    __syncthreads();
    if (threadIdx.x < head_dim)
      result = result * previous_scale + token_scale *
          __half2float(value_page[cache]);
    __syncthreads();
  }
  if (threadIdx.x < head_dim) {
    const float gate = query[head_dim + threadIdx.x];
    output[static_cast<std::size_t>(query_head) * head_dim + threadIdx.x] =
        (result / online_denominator) / (1.0F + expf(-gate));
  }
}

__global__ void gated_gqa_qkv_rope_paged_fp4_kernel(
    float* q_and_gate, float* key, const float* value,
    const float* q_weight, const float* k_weight, std::uint8_t* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t cache_position, std::uint32_t rotary_position,
    std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head < query_heads) {
    auto* query = q_and_gate + static_cast<std::size_t>(head) * 2U * head_dim;
    float square = dimension < head_dim
                       ? query[dimension] * query[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = query[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + q_weight[dimension]);
    __syncthreads();
    if (dimension < head_dim) {
      float final_query = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle = static_cast<float>(rotary_position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const float other = dimension < half ? -normalized[dimension + half]
                                             : normalized[dimension - half];
        final_query = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      query[dimension] = final_query;
    }
  }
  __syncthreads();
  if (head >= kv_heads) return;

  auto* key_head = key + static_cast<std::size_t>(head) * head_dim;
  float square = dimension < head_dim
                     ? key_head[dimension] * key_head[dimension]
                     : 0.0F;
  square = reduce_sum(square);
  if (dimension < head_dim)
    normalized[dimension] = key_head[dimension] *
        rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
        (1.0F + k_weight[dimension]);
  __syncthreads();
  if (dimension < head_dim) {
    float final_key = normalized[dimension];
    if (dimension < rotary_dim) {
      const auto half = rotary_dim / 2U;
      const auto pair = dimension % half;
      const float angle = static_cast<float>(rotary_position) *
          powf(theta, -2.0F * static_cast<float>(pair) /
                           static_cast<float>(rotary_dim));
      const float other = dimension < half ? -normalized[dimension + half]
                                           : normalized[dimension - half];
      final_key = normalized[dimension] * cosf(angle) + other * sinf(angle);
    }
    key_head[dimension] = final_key;
  }
  __syncthreads();

  const auto record_bytes = head_dim / 2U + head_dim / 32U;
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  auto* key_records = page +
      static_cast<std::size_t>(full_attention_layer) * 2U *
          records_per_kind * record_bytes;
  auto* value_records = key_records + records_per_kind * record_bytes;
  const auto record_index =
      static_cast<std::size_t>(cache_position % page_tokens) * kv_heads + head;
  auto* key_record = key_records + record_index * record_bytes;
  auto* value_record = value_records + record_index * record_bytes;

  // One lane owns one block-32 record. This keeps scale selection and packed
  // nibble writes deterministic and avoids atomics on adjacent values.
  const auto block = dimension;
  if (block < head_dim / 32U) {
    const auto first = block * 32U;
    float key_maximum = 0.0F;
    float value_maximum = 0.0F;
    for (std::uint32_t index = 0U; index < 32U; ++index) {
      key_maximum = fmaxf(key_maximum, fabsf(key_head[first + index]));
      value_maximum = fmaxf(
          value_maximum,
          fabsf(value[static_cast<std::size_t>(head) * head_dim + first +
                      index]));
    }
    const auto key_scale_code = encode_ue8m0_cover(key_maximum);
    const auto value_scale_code = encode_ue8m0_cover(value_maximum);
    const auto key_scale = decode_ue8m0(key_scale_code);
    const auto value_scale = decode_ue8m0(value_scale_code);
    key_record[head_dim / 2U + block] = key_scale_code;
    value_record[head_dim / 2U + block] = value_scale_code;
    for (std::uint32_t index = 0U; index < 16U; ++index) {
      const auto offset = first + 2U * index;
      const auto key_low = encode_fp4_nearest(key_head[offset], key_scale);
      const auto key_high =
          encode_fp4_nearest(key_head[offset + 1U], key_scale);
      const auto value_low = encode_fp4_nearest(
          value[static_cast<std::size_t>(head) * head_dim + offset],
          value_scale);
      const auto value_high = encode_fp4_nearest(
          value[static_cast<std::size_t>(head) * head_dim + offset + 1U],
          value_scale);
      key_record[offset / 2U] =
          static_cast<std::uint8_t>(key_low | (key_high << 4U));
      value_record[offset / 2U] =
          static_cast<std::uint8_t>(value_low | (value_high << 4U));
    }
  }
}

__global__ void gated_gqa_qkv_rope_paged_fp4_batch_kernel(
    float* q_and_gate, float* key, const float* value,
    const float* q_weight, const float* k_weight,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t first_rotary_position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto cache_position = first_cache_position + row;
  const auto rotary_position = first_rotary_position + row;
  const auto query_width = 2U * query_heads * head_dim;
  const auto key_value_width = kv_heads * head_dim;
  q_and_gate += static_cast<std::size_t>(row) * query_width;
  key += static_cast<std::size_t>(row) * key_value_width;
  value += static_cast<std::size_t>(row) * key_value_width;

  if (head < query_heads) {
    auto* query = q_and_gate + static_cast<std::size_t>(head) * 2U * head_dim;
    float square = dimension < head_dim
                       ? query[dimension] * query[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] =
          query[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + q_weight[dimension]);
    __syncthreads();
    if (dimension < head_dim) {
      float final_query = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const float angle =
            static_cast<float>(rotary_position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                            static_cast<float>(rotary_dim));
        const float other = dimension < half
                                ? -normalized[dimension + half]
                                : normalized[dimension - half];
        final_query = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      query[dimension] = final_query;
    }
  }
  __syncthreads();
  if (head >= kv_heads) return;

  auto* key_head = key + static_cast<std::size_t>(head) * head_dim;
  float square = dimension < head_dim
                     ? key_head[dimension] * key_head[dimension]
                     : 0.0F;
  square = reduce_sum(square);
  if (dimension < head_dim)
    normalized[dimension] =
        key_head[dimension] *
        rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
        (1.0F + k_weight[dimension]);
  __syncthreads();
  if (dimension < head_dim) {
    float final_key = normalized[dimension];
    if (dimension < rotary_dim) {
      const auto half = rotary_dim / 2U;
      const auto pair = dimension % half;
      const float angle =
          static_cast<float>(rotary_position) *
          powf(theta, -2.0F * static_cast<float>(pair) /
                          static_cast<float>(rotary_dim));
      const float other = dimension < half ? -normalized[dimension + half]
                                           : normalized[dimension - half];
      final_key = normalized[dimension] * cosf(angle) + other * sinf(angle);
    }
    key_head[dimension] = final_key;
  }
  __syncthreads();

  auto* page = static_cast<std::uint8_t*>(
      const_cast<void*>(page_table[cache_position / page_tokens]));
  const auto record_bytes = head_dim / 2U + head_dim / 32U;
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  auto* key_records = page +
      static_cast<std::size_t>(full_attention_layer) * 2U *
          records_per_kind * record_bytes;
  auto* value_records = key_records + records_per_kind * record_bytes;
  const auto record_index =
      static_cast<std::size_t>(cache_position % page_tokens) * kv_heads + head;
  auto* key_record = key_records + record_index * record_bytes;
  auto* value_record = value_records + record_index * record_bytes;
  const auto block = dimension;
  if (block < head_dim / 32U) {
    const auto first = block * 32U;
    float key_maximum = 0.0F;
    float value_maximum = 0.0F;
    for (std::uint32_t index = 0U; index < 32U; ++index) {
      key_maximum = fmaxf(key_maximum, fabsf(key_head[first + index]));
      value_maximum = fmaxf(
          value_maximum,
          fabsf(value[static_cast<std::size_t>(head) * head_dim + first +
                      index]));
    }
    const auto key_scale_code = encode_ue8m0_cover(key_maximum);
    const auto value_scale_code = encode_ue8m0_cover(value_maximum);
    const auto key_scale = decode_ue8m0(key_scale_code);
    const auto value_scale = decode_ue8m0(value_scale_code);
    key_record[head_dim / 2U + block] = key_scale_code;
    value_record[head_dim / 2U + block] = value_scale_code;
    for (std::uint32_t index = 0U; index < 16U; ++index) {
      const auto offset = first + 2U * index;
      const auto key_low = encode_fp4_nearest(key_head[offset], key_scale);
      const auto key_high =
          encode_fp4_nearest(key_head[offset + 1U], key_scale);
      const auto value_low = encode_fp4_nearest(
          value[static_cast<std::size_t>(head) * head_dim + offset],
          value_scale);
      const auto value_high = encode_fp4_nearest(
          value[static_cast<std::size_t>(head) * head_dim + offset + 1U],
          value_scale);
      key_record[offset / 2U] =
          static_cast<std::uint8_t>(key_low | (key_high << 4U));
      value_record[offset / 2U] =
          static_cast<std::uint8_t>(value_low | (value_high << 4U));
    }
  }
}

__global__ void gated_gqa_attention_paged_fp4_split_kernel(
    const float* q_and_gate, const void* const* page_table,
    float* partial_maxima, float* partial_sums, float* partial_outputs,
    std::uint32_t tokens, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t split_tokens) {
  __shared__ __align__(32) __nv_bfloat16
      key_values[kAttentionKvTile][kMaximumAttentionHeadDim];
  __shared__ __align__(32) __nv_bfloat16
      value_values[kAttentionKvTile][kMaximumAttentionHeadDim];

  const auto kv_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto split = static_cast<std::uint32_t>(blockIdx.y);
  const auto grouped_heads = query_heads / kv_heads;
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / kWarpSize;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) % kWarpSize;
  const auto first_token = split * split_tokens;
  const auto last_token = min(tokens, first_token + split_tokens);
  const auto active = warp < grouped_heads;
  float maximum = kNegativeInfinity;
  float sum = 0.0F;
  float results[kMaximumAttentionHeadDim / kWarpSize]{};
  const auto record_bytes = head_dim / 2U + head_dim / 32U;
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  const auto query_head = kv_head * grouped_heads + warp;
  const auto* query = active
                          ? q_and_gate + static_cast<std::size_t>(query_head) *
                                             2U * head_dim
                          : q_and_gate;

  for (std::uint32_t tile_first = first_token; tile_first < last_token;
       tile_first += kAttentionKvTile) {
    const auto tile_tokens =
        min(kAttentionKvTile, last_token - tile_first);
    const auto blocks_per_record = head_dim / 32U;
    const auto blocks_per_kind = tile_tokens * blocks_per_record;
    for (std::uint32_t item = threadIdx.x; item < 2U * blocks_per_kind;
         item += blockDim.x) {
      const auto is_value = item >= blocks_per_kind;
      const auto local_item = item - (is_value ? blocks_per_kind : 0U);
      const auto local_token = local_item / blocks_per_record;
      const auto block = local_item % blocks_per_record;
      const auto token = tile_first + local_token;
      const auto* page =
          static_cast<const std::uint8_t*>(page_table[token / page_tokens]);
      const auto* page_keys = page +
          static_cast<std::size_t>(full_attention_layer) * 2U *
              records_per_kind * record_bytes;
      const auto* page_values =
          page_keys + records_per_kind * record_bytes;
      const auto record_index =
          static_cast<std::size_t>(token % page_tokens) * kv_heads + kv_head;
      const auto* source = (is_value ? page_values : page_keys) +
                           record_index * record_bytes;
      auto* target = is_value ? value_values[local_token]
                              : key_values[local_token];
      const auto scale = 0.5F * decode_ue8m0(
          source[head_dim / 2U + block]);
      const auto packed_first = block * 16U;
#pragma unroll
      for (std::uint32_t packed_offset = 0U; packed_offset < 16U;
           ++packed_offset) {
        const auto packed = source[packed_first + packed_offset];
        const auto dimension = 2U * (packed_first + packed_offset);
        *reinterpret_cast<__nv_bfloat162*>(target + dimension) =
            __floats2bfloat162_rn(
                static_cast<float>(decode_fp4_twice(packed & 0x0fU)) *
                    scale,
                static_cast<float>(decode_fp4_twice(packed >> 4U)) * scale);
      }
    }
    __syncthreads();

    if (active) {
      const auto score_scale = rsqrtf(static_cast<float>(head_dim));
      float lane_score = kNegativeInfinity;
      for (std::uint32_t local_token = 0U; local_token < tile_tokens;
           ++local_token) {
        float score = 0.0F;
        for (std::uint32_t local = 0U;
             local < kMaximumAttentionHeadDim / kWarpSize; ++local) {
          const auto dimension = lane + local * kWarpSize;
          if (dimension < head_dim)
            score = fmaf(
                query[dimension],
                __bfloat162float(key_values[local_token][dimension]), score);
        }
        score = warp_sum(score);
        score = __shfl_sync(0xffffffffU, score, 0U) * score_scale;
        if (lane == local_token) lane_score = score;
      }

      auto tile_maximum = warp_max(lane_score);
      tile_maximum = __shfl_sync(0xffffffffU, tile_maximum, 0U);
      float next_maximum{};
      float previous_scale{};
      if (lane == 0U) {
        next_maximum = fmaxf(maximum, tile_maximum);
        previous_scale = maximum == kNegativeInfinity
                             ? 0.0F
                             : expf(maximum - next_maximum);
      }
      next_maximum =
          __shfl_sync(0xffffffffU, next_maximum, 0U);
      previous_scale =
          __shfl_sync(0xffffffffU, previous_scale, 0U);
      const auto probability =
          lane < tile_tokens ? expf(lane_score - next_maximum) : 0.0F;
      const auto tile_sum = warp_sum(probability);
      if (lane == 0U) {
        sum = sum * previous_scale + tile_sum;
        maximum = next_maximum;
      }

      for (std::uint32_t local = 0U;
           local < kMaximumAttentionHeadDim / kWarpSize; ++local) {
        const auto dimension = lane + local * kWarpSize;
        if (dimension < head_dim) results[local] *= previous_scale;
      }
      for (std::uint32_t local_token = 0U;
           local_token < tile_tokens; ++local_token) {
        const auto token_probability =
            __shfl_sync(0xffffffffU, probability, local_token);
        for (std::uint32_t local = 0U;
             local < kMaximumAttentionHeadDim / kWarpSize; ++local) {
          const auto dimension = lane + local * kWarpSize;
          if (dimension < head_dim)
            results[local] =
                fmaf(token_probability,
                     __bfloat162float(
                         value_values[local_token][dimension]),
                     results[local]);
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    const auto partial =
        static_cast<std::size_t>(split) * query_heads + query_head;
    if (lane == 0U) {
      partial_maxima[partial] = maximum;
      partial_sums[partial] = sum;
    }
    for (std::uint32_t local = 0U;
         local < kMaximumAttentionHeadDim / kWarpSize; ++local) {
      const auto dimension = lane + local * kWarpSize;
      if (dimension < head_dim)
        partial_outputs[partial * head_dim + dimension] = results[local];
    }
  }
}

// Flash-decoding counterpart to the prefill Tensor Core kernel. Query heads
// in one GQA group are the matrix rows, so every packed K/V record is decoded
// once and reused by all heads that share the KV head. Padding the group to a
// 16-row WMMA tile changes neither routing nor attention visibility.
template <std::uint32_t BlockThreads, std::uint32_t HeadCapacity>
__global__ void gated_gqa_attention_paged_fp4_tensor_core_split_kernel(
    const float* q_and_gate, const void* const* page_table,
    float* partial_maxima, float* partial_sums, float* partial_outputs,
    std::uint32_t first_context_tokens, std::uint32_t rows,
    std::uint32_t maximum_splits, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t split_tokens) {
  using namespace nvcuda;
  static_assert(HeadCapacity == 128U || HeadCapacity == 256U);
  constexpr std::uint32_t kQueryTile = 16U;
  constexpr std::uint32_t kKeyTile = 16U;
  constexpr std::uint32_t kBlockWarps = BlockThreads / kWarpSize;
  __shared__ __align__(32) __nv_bfloat16
      query_values[kQueryTile][HeadCapacity];
  __shared__ __align__(32) __nv_bfloat16
      key_values[kKeyTile][HeadCapacity];
  __shared__ __align__(32) __nv_bfloat16
      value_values[kKeyTile][HeadCapacity];
  constexpr std::uint32_t kScoreWarps =
      HeadCapacity / 64U;
  __shared__ __align__(32) float
      score_partials[kScoreWarps][kQueryTile][kKeyTile];
  __shared__ __align__(32) __nv_bfloat16 probabilities[kQueryTile][kKeyTile];
  __shared__ __align__(32) float
      products[kQueryTile][HeadCapacity];
  __shared__ float row_maxima[kQueryTile];
  __shared__ float row_sums[kQueryTile];
  __shared__ float row_previous_scales[kQueryTile];

  const auto local_thread = static_cast<std::uint32_t>(threadIdx.x);
  const auto warp = local_thread / kWarpSize;
  const auto kv_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto split = static_cast<std::uint32_t>(blockIdx.y);
  const auto grouped_heads = query_heads / kv_heads;
  const auto matrix_rows = rows * grouped_heads;
  const auto first_token = split * split_tokens;
  const auto last_context_tokens = first_context_tokens + rows - 1U;
  const auto last_token =
      min(last_context_tokens, first_token + split_tokens);
  const auto record_bytes = head_dim / 2U + head_dim / 32U;
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  float results[kQueryTile * HeadCapacity / BlockThreads]{};

  for (std::uint32_t item = local_thread;
       item < kQueryTile * head_dim; item += blockDim.x) {
    const auto matrix_row = item / head_dim;
    const auto dimension = item % head_dim;
    float value{};
    if (matrix_row < matrix_rows) {
      const auto row = matrix_row / grouped_heads;
      const auto local_head = matrix_row % grouped_heads;
      const auto query_head = kv_head * grouped_heads + local_head;
      value = q_and_gate[
          (static_cast<std::size_t>(row) * query_heads + query_head) * 2U *
              head_dim +
          dimension];
    }
    query_values[matrix_row][dimension] = __float2bfloat16(value);
  }
  if (local_thread < kQueryTile) {
    row_maxima[local_thread] = kNegativeInfinity;
    row_sums[local_thread] = 0.0F;
  }
  if (local_thread < kQueryTile * kKeyTile)
    probabilities[local_thread / kKeyTile][local_thread % kKeyTile] =
        __float2bfloat16(0.0F);
  __syncthreads();

  for (std::uint32_t tile_first = first_token; tile_first < last_token;
       tile_first += kKeyTile) {
    const auto tile_tokens = min(kKeyTile, last_token - tile_first);
    const auto blocks_per_record = head_dim / 32U;
    const auto blocks_per_kind = tile_tokens * blocks_per_record;
    for (std::uint32_t item = local_thread; item < 2U * blocks_per_kind;
         item += blockDim.x) {
      const auto is_value = item >= blocks_per_kind;
      const auto local_item = item - (is_value ? blocks_per_kind : 0U);
      const auto local_token = local_item / blocks_per_record;
      const auto block = local_item % blocks_per_record;
      const auto token = tile_first + local_token;
      const auto* page =
          static_cast<const std::uint8_t*>(page_table[token / page_tokens]);
      const auto* page_keys = page +
          static_cast<std::size_t>(full_attention_layer) * 2U *
              records_per_kind * record_bytes;
      const auto* page_values =
          page_keys + records_per_kind * record_bytes;
      const auto record_index =
          static_cast<std::size_t>(token % page_tokens) * kv_heads + kv_head;
      const auto* source = (is_value ? page_values : page_keys) +
                           record_index * record_bytes;
      auto* target = is_value ? value_values[local_token]
                              : key_values[local_token];
      const auto scale =
          0.5F * decode_ue8m0(source[head_dim / 2U + block]);
      const auto packed_first = block * 16U;
      std::uint32_t packed_words[4]{};
      if ((record_bytes & 3U) == 0U) {
        const auto* source_words = reinterpret_cast<const std::uint32_t*>(
            source + packed_first);
#pragma unroll
        for (std::uint32_t word = 0U; word < 4U; ++word)
          packed_words[word] = source_words[word];
      }
#pragma unroll
      for (std::uint32_t packed_offset = 0U; packed_offset < 16U;
           ++packed_offset) {
        const auto packed = static_cast<std::uint8_t>(
            (record_bytes & 3U) == 0U
                ? packed_words[packed_offset / 4U] >>
                      (8U * (packed_offset & 3U))
                : source[packed_first + packed_offset]);
        const auto dimension = 2U * (packed_first + packed_offset);
        *reinterpret_cast<__nv_bfloat162*>(target + dimension) =
            __floats2bfloat162_rn(
                static_cast<float>(decode_fp4_twice(packed & 0x0fU)) * scale,
                static_cast<float>(decode_fp4_twice(packed >> 4U)) * scale);
      }
    }
    const auto trailing_tokens = kKeyTile - tile_tokens;
    for (std::uint32_t item = local_thread;
         item < 2U * trailing_tokens * head_dim; item += blockDim.x) {
      const auto values_per_kind = trailing_tokens * head_dim;
      const auto is_value = item >= values_per_kind;
      const auto local_item = item - (is_value ? values_per_kind : 0U);
      const auto local_token = tile_tokens + local_item / head_dim;
      const auto dimension = local_item % head_dim;
      auto* target = is_value ? value_values[local_token]
                              : key_values[local_token];
      target[dimension] = __float2bfloat16(0.0F);
    }
    __syncthreads();

    const auto active_score_warps = (head_dim + 63U) / 64U;
    if (warp < active_score_warps) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          query_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          key_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          score_fragment;
      wmma::fill_fragment(score_fragment, 0.0F);
      const auto dimension_end = min(head_dim, (warp + 1U) * 64U);
      for (std::uint32_t dimension = warp * 64U;
           dimension < dimension_end;
           dimension += 16U) {
        wmma::load_matrix_sync(query_fragment, &query_values[0][dimension],
                               HeadCapacity);
        wmma::load_matrix_sync(key_fragment, &key_values[0][dimension],
                               HeadCapacity);
        wmma::mma_sync(score_fragment, query_fragment, key_fragment,
                       score_fragment);
      }
      wmma::store_matrix_sync(&score_partials[warp][0][0], score_fragment,
                              kKeyTile, wmma::mem_row_major);
    }
    __syncthreads();

    if (warp < matrix_rows) {
      const auto matrix_row = warp;
      const auto row = matrix_row / grouped_heads;
      const auto lane = local_thread % kWarpSize;
      float lane_score = kNegativeInfinity;
      const auto context_tokens = first_context_tokens + row;
      if (lane < tile_tokens && tile_first + lane < context_tokens) {
        lane_score = 0.0F;
        for (std::uint32_t score_warp = 0U;
             score_warp < active_score_warps; ++score_warp)
          lane_score += score_partials[score_warp][matrix_row][lane];
        lane_score *= rsqrtf(static_cast<float>(head_dim));
      }
      auto tile_maximum = warp_max(lane_score);
      tile_maximum = __shfl_sync(0xffffffffU, tile_maximum, 0U);
      float next_maximum{};
      float previous_scale{};
      if (lane == 0U) {
        next_maximum = fmaxf(row_maxima[matrix_row], tile_maximum);
        previous_scale = row_maxima[matrix_row] == kNegativeInfinity
                             ? 0.0F
                             : expf(row_maxima[matrix_row] - next_maximum);
      }
      next_maximum = __shfl_sync(0xffffffffU, next_maximum, 0U);
      previous_scale = __shfl_sync(0xffffffffU, previous_scale, 0U);
      const auto probability = lane < tile_tokens &&
                                       tile_first + lane < context_tokens
                                   ? expf(lane_score - next_maximum)
                                   : 0.0F;
      const auto quantized = __float2bfloat16(probability);
      if (lane < kKeyTile)
        probabilities[matrix_row][lane] = quantized;
      const auto tile_sum = warp_sum(__bfloat162float(quantized));
      if (lane == 0U) {
        row_sums[matrix_row] =
            row_sums[matrix_row] * previous_scale + tile_sum;
        row_maxima[matrix_row] = next_maximum;
        row_previous_scales[matrix_row] = previous_scale;
      }
    }
    __syncthreads();

    for (std::uint32_t output_tile = warp;
         output_tile < head_dim / 16U;
         output_tile += kBlockWarps) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          probability_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          value_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          product_fragment;
      wmma::fill_fragment(product_fragment, 0.0F);
      wmma::load_matrix_sync(probability_fragment, &probabilities[0][0],
                             kKeyTile);
      wmma::load_matrix_sync(value_fragment,
                             &value_values[0][output_tile * 16U],
                             HeadCapacity);
      wmma::mma_sync(product_fragment, probability_fragment, value_fragment,
                     product_fragment);
      wmma::store_matrix_sync(&products[0][output_tile * 16U],
                              product_fragment, HeadCapacity,
                              wmma::mem_row_major);
    }
    __syncthreads();

    for (std::uint32_t local = 0U;
         local < kQueryTile * HeadCapacity / BlockThreads;
         ++local) {
      const auto item = local_thread + local * BlockThreads;
      const auto matrix_row = item / HeadCapacity;
      const auto dimension = item % HeadCapacity;
      if (matrix_row < matrix_rows && dimension < head_dim)
        results[local] =
            results[local] * row_previous_scales[matrix_row] +
            products[matrix_row][dimension];
    }
    __syncthreads();
  }

  for (std::uint32_t local = 0U;
       local < kQueryTile * HeadCapacity / BlockThreads;
       ++local) {
    const auto item = local_thread + local * BlockThreads;
    const auto matrix_row = item / HeadCapacity;
    const auto dimension = item % HeadCapacity;
    if (matrix_row >= matrix_rows || dimension >= head_dim) continue;
    const auto row = matrix_row / grouped_heads;
    const auto local_head = matrix_row % grouped_heads;
    const auto query_head = kv_head * grouped_heads + local_head;
    const auto partial =
        (static_cast<std::size_t>(row) * maximum_splits + split) *
            query_heads +
        query_head;
    if (dimension == 0U) {
      partial_maxima[partial] = row_maxima[matrix_row];
      partial_sums[partial] = row_sums[matrix_row];
    }
    partial_outputs[partial * head_dim + dimension] = results[local];
  }
}

__global__ void gated_gqa_attention_paged_fp4_combine_kernel(
    const float* q_and_gate, float* partial_maxima,
    const float* partial_sums, const float* partial_outputs, float* output,
    std::uint32_t splits, std::uint32_t query_heads,
    std::uint32_t head_dim) {
  __shared__ float global_sum;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (dimension == 0U) {
    float maximum = kNegativeInfinity;
    for (std::uint32_t split = 0U; split < splits; ++split)
      maximum = fmaxf(
          maximum,
          partial_maxima[static_cast<std::size_t>(split) * query_heads +
                         query_head]);
    float sum = 0.0F;
    for (std::uint32_t split = 0U; split < splits; ++split) {
      const auto partial =
          static_cast<std::size_t>(split) * query_heads + query_head;
      const auto rescale = expf(partial_maxima[partial] - maximum);
      sum += partial_sums[partial] * rescale;
      // The producer rewrites this scratch on every launch. Preserve the
      // rescale computed for the denominator so all head dimensions reuse the
      // same exact value instead of evaluating expf again.
      partial_maxima[partial] = rescale;
    }
    global_sum = sum;
  }
  __syncthreads();
  if (dimension >= head_dim) return;
  float result = 0.0F;
  for (std::uint32_t split = 0U; split < splits; ++split) {
    const auto partial =
        static_cast<std::size_t>(split) * query_heads + query_head;
    result += partial_outputs[partial * head_dim + dimension] *
              partial_maxima[partial];
  }
  const auto* query = q_and_gate +
      static_cast<std::size_t>(query_head) * 2U * head_dim;
  const float gate = query[head_dim + dimension];
  output[static_cast<std::size_t>(query_head) * head_dim + dimension] =
      (result / global_sum) / (1.0F + expf(-gate));
}

__global__ void gated_gqa_attention_paged_fp4_prefill_split_kernel(
    const float* q_and_gate, const void* const* page_table,
    float* partial_maxima, float* partial_sums, float* partial_outputs,
    std::uint32_t first_context_tokens, std::uint32_t rows,
    std::uint32_t maximum_splits, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t split_tokens) {
  using namespace nvcuda;
  constexpr std::uint32_t kQueryTile = 16U;
  constexpr std::uint32_t kKeyTile = 16U;
  __shared__ __align__(32) __nv_bfloat16
      query_values[kQueryTile][kMaximumAttentionHeadDim];
  __shared__ __align__(32) __nv_bfloat16
      key_values[kKeyTile][kMaximumAttentionHeadDim];
  __shared__ __align__(32) __nv_bfloat16
      value_values[kKeyTile][kMaximumAttentionHeadDim];
  __shared__ __align__(32) float scores[kQueryTile][kKeyTile];
  __shared__ __align__(32) __nv_bfloat16 probabilities[kQueryTile][kKeyTile];
  __shared__ __align__(32) float
      products[kQueryTile][kMaximumAttentionHeadDim];
  __shared__ float row_maxima[kQueryTile];
  __shared__ float row_sums[kQueryTile];
  __shared__ float row_previous_scales[kQueryTile];

  const auto local_thread = static_cast<std::uint32_t>(threadIdx.x);
  const auto warp = local_thread / kWarpSize;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto split = static_cast<std::uint32_t>(blockIdx.y);
  const auto query_tile = static_cast<std::uint32_t>(blockIdx.z);
  const auto first_query_row = query_tile * kQueryTile;
  const auto grouped_heads = query_heads / kv_heads;
  const auto kv_head = query_head / grouped_heads;
  const auto first_token = split * split_tokens;
  const auto last_query_row =
      min(rows - 1U, first_query_row + kQueryTile - 1U);
  const auto last_token =
      min(first_context_tokens + last_query_row,
          first_token + split_tokens);
  const auto record_bytes = head_dim / 2U + head_dim / 32U;
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  float results[kQueryTile * kMaximumAttentionHeadDim / kFp4GemmThreads]{};

  for (std::uint32_t item = local_thread;
       item < kQueryTile * head_dim; item += blockDim.x) {
    const auto local_row = item / head_dim;
    const auto dimension = item % head_dim;
    const auto query_row = first_query_row + local_row;
    float value{};
    if (query_row < rows)
      value = q_and_gate[
          (static_cast<std::size_t>(query_row) * query_heads + query_head) *
              2U * head_dim +
          dimension];
    query_values[local_row][dimension] = __float2bfloat16(value);
  }
  if (local_thread < kQueryTile) {
    row_maxima[local_thread] = kNegativeInfinity;
    row_sums[local_thread] = 0.0F;
  }
  __syncthreads();

  for (std::uint32_t tile_first = first_token; tile_first < last_token;
       tile_first += kKeyTile) {
    const auto tile_tokens = min(kKeyTile, last_token - tile_first);
    const auto packed_values = tile_tokens * (head_dim / 2U);
    for (std::uint32_t item = local_thread; item < 2U * packed_values;
         item += blockDim.x) {
      const auto is_value = item >= packed_values;
      const auto local_item = item - (is_value ? packed_values : 0U);
      const auto local_token = local_item / (head_dim / 2U);
      const auto packed_dimension = local_item % (head_dim / 2U);
      const auto token = tile_first + local_token;
      const auto* page =
          static_cast<const std::uint8_t*>(page_table[token / page_tokens]);
      const auto* page_keys = page +
          static_cast<std::size_t>(full_attention_layer) * 2U *
              records_per_kind * record_bytes;
      const auto* page_values =
          page_keys + records_per_kind * record_bytes;
      const auto record_index =
          static_cast<std::size_t>(token % page_tokens) * kv_heads + kv_head;
      const auto* source = (is_value ? page_values : page_keys) +
                           record_index * record_bytes;
      auto* target = is_value ? value_values[local_token]
                              : key_values[local_token];
      const auto packed = source[packed_dimension];
      const auto scale = 0.5F * decode_ue8m0(
          source[head_dim / 2U + packed_dimension / 16U]);
      const auto dimension = 2U * packed_dimension;
      target[dimension] = __float2bfloat16(
          static_cast<float>(decode_fp4_twice(packed & 0x0fU)) * scale);
      target[dimension + 1U] = __float2bfloat16(
          static_cast<float>(decode_fp4_twice(packed >> 4U)) * scale);
    }
    for (std::uint32_t item = local_thread + packed_values;
         item < kKeyTile * (head_dim / 2U); item += blockDim.x) {
      const auto local_token = item / (head_dim / 2U);
      const auto dimension = 2U * (item % (head_dim / 2U));
      key_values[local_token][dimension] = __float2bfloat16(0.0F);
      key_values[local_token][dimension + 1U] = __float2bfloat16(0.0F);
      value_values[local_token][dimension] = __float2bfloat16(0.0F);
      value_values[local_token][dimension + 1U] = __float2bfloat16(0.0F);
    }
    __syncthreads();

    if (warp == 0U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          query_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          key_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          score_fragment;
      wmma::fill_fragment(score_fragment, 0.0F);
      for (std::uint32_t dimension = 0U; dimension < head_dim;
           dimension += 16U) {
        wmma::load_matrix_sync(query_fragment, &query_values[0][dimension],
                               kMaximumAttentionHeadDim);
        wmma::load_matrix_sync(key_fragment, &key_values[0][dimension],
                               kMaximumAttentionHeadDim);
        wmma::mma_sync(score_fragment, query_fragment, key_fragment,
                       score_fragment);
      }
      wmma::store_matrix_sync(&scores[0][0], score_fragment, kKeyTile,
                              wmma::mem_row_major);
    }
    __syncthreads();

    if (local_thread < kQueryTile) {
      const auto local_row = local_thread;
      const auto query_row = first_query_row + local_row;
      const auto context_tokens = first_context_tokens + query_row;
      const auto score_scale = rsqrtf(static_cast<float>(head_dim));
      float tile_maximum = kNegativeInfinity;
      for (std::uint32_t local_token = 0U; local_token < tile_tokens;
           ++local_token) {
        if (query_row < rows && tile_first + local_token < context_tokens)
          tile_maximum = fmaxf(
              tile_maximum, scores[local_row][local_token] * score_scale);
      }
      if (tile_maximum == kNegativeInfinity) {
        row_previous_scales[local_row] = 1.0F;
        for (std::uint32_t local_token = 0U; local_token < kKeyTile;
             ++local_token)
          probabilities[local_row][local_token] = __float2bfloat16(0.0F);
      } else {
        const auto next_maximum =
            fmaxf(row_maxima[local_row], tile_maximum);
        const auto previous_scale =
            row_maxima[local_row] == kNegativeInfinity
                ? 0.0F
                : expf(row_maxima[local_row] - next_maximum);
        float tile_sum = 0.0F;
        for (std::uint32_t local_token = 0U; local_token < kKeyTile;
             ++local_token) {
          float probability{};
          if (local_token < tile_tokens && query_row < rows &&
              tile_first + local_token < context_tokens)
            probability = expf(scores[local_row][local_token] * score_scale -
                               next_maximum);
          const auto quantized = __float2bfloat16(probability);
          probabilities[local_row][local_token] = quantized;
          tile_sum += __bfloat162float(quantized);
        }
        row_sums[local_row] =
            row_sums[local_row] * previous_scale + tile_sum;
        row_maxima[local_row] = next_maximum;
        row_previous_scales[local_row] = previous_scale;
      }
    }
    __syncthreads();

    for (std::uint32_t output_tile = warp;
         output_tile < head_dim / 16U; output_tile += kFp4GemmWarps) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          probability_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major>
          value_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          product_fragment;
      wmma::fill_fragment(product_fragment, 0.0F);
      wmma::load_matrix_sync(probability_fragment, &probabilities[0][0],
                             kKeyTile);
      wmma::load_matrix_sync(value_fragment,
                             &value_values[0][output_tile * 16U],
                             kMaximumAttentionHeadDim);
      wmma::mma_sync(product_fragment, probability_fragment, value_fragment,
                     product_fragment);
      wmma::store_matrix_sync(&products[0][output_tile * 16U],
                              product_fragment,
                              kMaximumAttentionHeadDim,
                              wmma::mem_row_major);
    }
    __syncthreads();

    for (std::uint32_t local = 0U;
         local < kQueryTile * kMaximumAttentionHeadDim /
                     kFp4GemmThreads;
         ++local) {
      const auto item = local_thread + local * kFp4GemmThreads;
      const auto local_row = item / kMaximumAttentionHeadDim;
      const auto dimension = item % kMaximumAttentionHeadDim;
      if (first_query_row + local_row < rows && dimension < head_dim)
        results[local] =
            results[local] * row_previous_scales[local_row] +
            products[local_row][dimension];
    }
    __syncthreads();
  }

  for (std::uint32_t local = 0U;
       local < kQueryTile * kMaximumAttentionHeadDim / kFp4GemmThreads;
       ++local) {
    const auto item = local_thread + local * kFp4GemmThreads;
    const auto local_row = item / kMaximumAttentionHeadDim;
    const auto dimension = item % kMaximumAttentionHeadDim;
    const auto query_row = first_query_row + local_row;
    const auto context_tokens = first_context_tokens + query_row;
    if (query_row >= rows || dimension >= head_dim ||
        first_token >= context_tokens)
      continue;
    const auto partial =
        (static_cast<std::size_t>(query_row) * maximum_splits + split) *
            query_heads +
        query_head;
    if (dimension == 0U) {
      partial_maxima[partial] = row_maxima[local_row];
      partial_sums[partial] = row_sums[local_row];
    }
    partial_outputs[partial * head_dim + dimension] = results[local];
  }
}

__global__ void gated_gqa_attention_paged_fp4_prefill_combine_kernel(
    const float* q_and_gate, const float* partial_maxima,
    const float* partial_sums, const float* partial_outputs, float* output,
    std::uint32_t first_context_tokens, std::uint32_t maximum_splits,
    std::uint32_t split_tokens, std::uint32_t query_heads,
    std::uint32_t head_dim) {
  __shared__ float global_maximum;
  __shared__ float global_sum;
  const auto query_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto active_splits =
      (first_context_tokens + row + split_tokens - 1U) / split_tokens;
  if (dimension == 0U) {
    float maximum = kNegativeInfinity;
    for (std::uint32_t split = 0U; split < active_splits; ++split) {
      const auto partial =
          (static_cast<std::size_t>(row) * maximum_splits + split) *
              query_heads +
          query_head;
      maximum = fmaxf(maximum, partial_maxima[partial]);
    }
    float sum = 0.0F;
    for (std::uint32_t split = 0U; split < active_splits; ++split) {
      const auto partial =
          (static_cast<std::size_t>(row) * maximum_splits + split) *
              query_heads +
          query_head;
      sum += partial_sums[partial] *
             expf(partial_maxima[partial] - maximum);
    }
    global_maximum = maximum;
    global_sum = sum;
  }
  __syncthreads();
  if (dimension >= head_dim) return;
  float result = 0.0F;
  for (std::uint32_t split = 0U; split < active_splits; ++split) {
    const auto partial =
        (static_cast<std::size_t>(row) * maximum_splits + split) *
            query_heads +
        query_head;
    result += partial_outputs[partial * head_dim + dimension] *
              expf(partial_maxima[partial] - global_maximum);
  }
  const auto query = q_and_gate +
      (static_cast<std::size_t>(row) * query_heads + query_head) * 2U *
          head_dim;
  const auto gate = query[head_dim + dimension];
  output[(static_cast<std::size_t>(row) * query_heads + query_head) *
             head_dim +
         dimension] =
      (result / global_sum) / (1.0F + expf(-gate));
}

__global__ void qwen_delta_conv_kernel(
    const float* projected, const float* weights, float* state,
    float* output, std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t key_dim, std::uint32_t value_dim,
    std::uint32_t key_head_dim, std::uint32_t value_head_dim,
    std::uint32_t kernel) {
  const auto channel =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const auto conv_dim = 2U * key_dim + value_dim;
  if (channel >= conv_dim) return;
  const auto ratio = value_heads / key_heads;
  const auto group_width =
      2U * key_head_dim + 2U * ratio * value_head_dim;
  std::uint32_t source = 0;
  if (channel < key_dim) {
    const auto group = channel / key_head_dim;
    source = group * group_width + channel % key_head_dim;
  } else if (channel < 2U * key_dim) {
    const auto local = channel - key_dim;
    const auto group = local / key_head_dim;
    source = group * group_width + key_head_dim + local % key_head_dim;
  } else {
    const auto local = channel - 2U * key_dim;
    const auto value_head = local / value_head_dim;
    const auto group = value_head / ratio;
    const auto replica = value_head % ratio;
    source = group * group_width + 2U * key_head_dim +
             replica * value_head_dim + local % value_head_dim;
  }
  auto* channel_state = state + static_cast<std::size_t>(channel) * kernel;
  for (std::uint32_t i = 1; i < kernel; ++i)
    channel_state[i - 1] = channel_state[i];
  channel_state[kernel - 1] = projected[source];
  const auto* channel_weights =
      weights + static_cast<std::size_t>(channel) * kernel;
  float sum = 0.0F;
  for (std::uint32_t i = 0; i < kernel; ++i)
    sum += channel_state[i] * channel_weights[i];
  output[channel] = sum / (1.0F + expf(-sum));
}

__global__ void qwen_delta_recurrent_kernel(
    const float* projected_qkvz, const float* projected_ba,
    const float* conv, const float* dt_bias, const float* a_log,
    const float* norm_weight, float* recurrent, float* output,
    std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t key_head_dim, std::uint32_t value_head_dim,
    float epsilon) {
  __shared__ float query[256];
  __shared__ float key[256];
  __shared__ float per_dimension[256];
  const auto value_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto d = static_cast<std::uint32_t>(threadIdx.x);
  const auto ratio = value_heads / key_heads;
  const auto key_head = value_head / ratio;
  const auto key_dim = key_heads * key_head_dim;
  if (d < key_head_dim) {
    query[d] = conv[static_cast<std::size_t>(key_head) * key_head_dim + d];
    key[d] = conv[key_dim + static_cast<std::size_t>(key_head) * key_head_dim + d];
  }
  __syncthreads();
  float q_square = d < key_head_dim ? query[d] * query[d] : 0.0F;
  float k_square = d < key_head_dim ? key[d] * key[d] : 0.0F;
  q_square = reduce_sum(q_square);
  k_square = reduce_sum(k_square);
  if (d < key_head_dim) {
    query[d] *= rsqrtf(q_square + 1.0e-6F) *
                rsqrtf(static_cast<float>(key_head_dim));
    key[d] *= rsqrtf(k_square + 1.0e-6F);
  }
  __syncthreads();
  const auto group = value_head / ratio;
  const auto replica = value_head % ratio;
  const auto ba_stride = 2U * ratio;
  const float beta = 1.0F /
      (1.0F + expf(-projected_ba[group * ba_stride + replica]));
  const float a = projected_ba[group * ba_stride + ratio + replica];
  const float softplus = log1pf(expf(-fabsf(a + dt_bias[value_head]))) +
                         fmaxf(a + dt_bias[value_head], 0.0F);
  const float decay = expf(-expf(a_log[value_head]) * softplus);
  float core = 0.0F;
  if (d < value_head_dim) {
    auto* column = recurrent +
        static_cast<std::size_t>(value_head) * key_head_dim * value_head_dim + d;
    float memory = 0.0F;
    for (std::uint32_t i = 0; i < key_head_dim; ++i) {
      column[static_cast<std::size_t>(i) * value_head_dim] *= decay;
      memory += column[static_cast<std::size_t>(i) * value_head_dim] * key[i];
    }
    const float value = conv[2U * key_dim +
                             static_cast<std::size_t>(value_head) * value_head_dim + d];
    const float delta = (value - memory) * beta;
    for (std::uint32_t i = 0; i < key_head_dim; ++i) {
      auto& state_value = column[static_cast<std::size_t>(i) * value_head_dim];
      state_value += key[i] * delta;
      core += state_value * query[i];
    }
    per_dimension[d] = core * core;
  } else {
    per_dimension[d] = 0.0F;
  }
  __syncthreads();
  float square = reduce_sum(per_dimension[d]);
  if (d < value_head_dim) {
    const auto group_width =
        2U * key_head_dim + 2U * ratio * value_head_dim;
    const auto z_index = group * group_width + 2U * key_head_dim +
                         ratio * value_head_dim + replica * value_head_dim + d;
    const float z = projected_qkvz[z_index];
    const float gated = z / (1.0F + expf(-z));
    output[static_cast<std::size_t>(value_head) * value_head_dim + d] =
        core * rsqrtf(square / static_cast<float>(value_head_dim) + epsilon) *
        norm_weight[d] * gated;
  }
}

__global__ void split_delta_conv_kernel(
    const float* projected_qkv, const float* weights, float* state,
    float* output, std::uint32_t conv_dim, std::uint32_t kernel) {
  const auto channel =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (channel >= conv_dim) return;
  auto* channel_state = state + static_cast<std::size_t>(channel) * kernel;
  for (std::uint32_t index = 1U; index < kernel; ++index)
    channel_state[index - 1U] = channel_state[index];
  channel_state[kernel - 1U] = projected_qkv[channel];
  const auto* channel_weights =
      weights + static_cast<std::size_t>(channel) * kernel;
  float sum = 0.0F;
  for (std::uint32_t index = 0U; index < kernel; ++index)
    sum += channel_state[index] * channel_weights[index];
  output[channel] = sum / (1.0F + expf(-sum));
}

__global__ void split_delta_recurrent_kernel(
    const float* projected_z, const float* projected_b,
    const float* projected_a, const float* conv, const float* dt_bias,
    const float* a_log, const float* norm_weight, float* recurrent,
    float* output, std::uint32_t key_heads, std::uint32_t value_heads,
    std::uint32_t key_head_dim, std::uint32_t value_head_dim,
    float epsilon) {
  __shared__ float query[256];
  __shared__ float key[256];
  __shared__ float per_dimension[256];
  const auto value_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto ratio = value_heads / key_heads;
  const auto key_head = value_head / ratio;
  const auto key_dim = key_heads * key_head_dim;
  if (dimension < key_head_dim) {
    query[dimension] =
        conv[static_cast<std::size_t>(key_head) * key_head_dim + dimension];
    key[dimension] = conv[key_dim +
        static_cast<std::size_t>(key_head) * key_head_dim + dimension];
  }
  __syncthreads();
  float q_square = dimension < key_head_dim
                       ? query[dimension] * query[dimension]
                       : 0.0F;
  float k_square = dimension < key_head_dim
                       ? key[dimension] * key[dimension]
                       : 0.0F;
  q_square = reduce_sum(q_square);
  k_square = reduce_sum(k_square);
  if (dimension < key_head_dim) {
    query[dimension] *= rsqrtf(q_square + 1.0e-6F) *
                        rsqrtf(static_cast<float>(key_head_dim));
    key[dimension] *= rsqrtf(k_square + 1.0e-6F);
  }
  __syncthreads();
  const float beta = 1.0F / (1.0F + expf(-projected_b[value_head]));
  const float a = projected_a[value_head] + dt_bias[value_head];
  const float softplus = log1pf(expf(-fabsf(a))) + fmaxf(a, 0.0F);
  const float decay = expf(-expf(a_log[value_head]) * softplus);
  float core = 0.0F;
  if (dimension < value_head_dim) {
    auto* column = recurrent +
        static_cast<std::size_t>(value_head) * key_head_dim *
            value_head_dim +
        dimension;
    float memory = 0.0F;
    for (std::uint32_t index = 0U; index < key_head_dim; ++index) {
      auto& state = column[static_cast<std::size_t>(index) * value_head_dim];
      state *= decay;
      memory += state * key[index];
    }
    const float value =
        conv[2U * key_dim +
             static_cast<std::size_t>(value_head) * value_head_dim +
             dimension];
    const float delta = (value - memory) * beta;
    for (std::uint32_t index = 0U; index < key_head_dim; ++index) {
      auto& state = column[static_cast<std::size_t>(index) * value_head_dim];
      state += key[index] * delta;
      core += state * query[index];
    }
    per_dimension[dimension] = core * core;
  } else {
    per_dimension[dimension] = 0.0F;
  }
  __syncthreads();
  const float square = reduce_sum(per_dimension[dimension]);
  if (dimension < value_head_dim) {
    const float z =
        projected_z[static_cast<std::size_t>(value_head) * value_head_dim +
                    dimension];
    output[static_cast<std::size_t>(value_head) * value_head_dim + dimension] =
        core * rsqrtf(square / static_cast<float>(value_head_dim) + epsilon) *
        norm_weight[dimension] * (z / (1.0F + expf(-z)));
  }
}

__global__ void split_delta_conv_prefill_kernel(
    const float* projected_qkv, const float* weights, float* state,
    float* output, std::uint32_t rows, std::uint32_t conv_dim,
    std::uint32_t kernel) {
  const auto channel =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (channel >= conv_dim) return;
  float local_state[16]{};
  const auto* channel_state =
      state + static_cast<std::size_t>(channel) * kernel;
  for (std::uint32_t index = 0U; index < kernel; ++index)
    local_state[index] = channel_state[index];
  const auto* channel_weights =
      weights + static_cast<std::size_t>(channel) * kernel;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    for (std::uint32_t index = 1U; index < kernel; ++index)
      local_state[index - 1U] = local_state[index];
    local_state[kernel - 1U] =
        projected_qkv[static_cast<std::size_t>(row) * conv_dim + channel];
    float sum = 0.0F;
    for (std::uint32_t index = 0U; index < kernel; ++index)
      sum += local_state[index] * channel_weights[index];
    output[static_cast<std::size_t>(row) * conv_dim + channel] =
        sum / (1.0F + expf(-sum));
  }
  auto* final_state = state + static_cast<std::size_t>(channel) * kernel;
  for (std::uint32_t index = 0U; index < kernel; ++index)
    final_state[index] = local_state[index];
}

__global__ void split_delta_recurrent_prefill_kernel(
    const float* projected_z, const float* projected_b,
    const float* projected_a, const float* conv, const float* dt_bias,
    const float* a_log, const float* norm_weight, float* recurrent,
    float* output, std::uint32_t rows, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t key_head_dim,
    std::uint32_t value_head_dim, float epsilon) {
  __shared__ float query[256];
  __shared__ float key[256];
  __shared__ float per_dimension[256];
  const auto value_head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto ratio = value_heads / key_heads;
  const auto key_head = value_head / ratio;
  const auto key_dim = key_heads * key_head_dim;
  const auto value_dim = value_heads * value_head_dim;
  const auto conv_dim = 2U * key_dim + value_dim;
  for (std::uint32_t row = 0U; row < rows; ++row) {
    const auto* row_conv = conv + static_cast<std::size_t>(row) * conv_dim;
    if (dimension < key_head_dim) {
      query[dimension] =
          row_conv[static_cast<std::size_t>(key_head) * key_head_dim +
                   dimension];
      key[dimension] =
          row_conv[key_dim +
                   static_cast<std::size_t>(key_head) * key_head_dim +
                   dimension];
    }
    __syncthreads();
    float q_square = dimension < key_head_dim
                         ? query[dimension] * query[dimension]
                         : 0.0F;
    float k_square = dimension < key_head_dim
                         ? key[dimension] * key[dimension]
                         : 0.0F;
    q_square = reduce_sum(q_square);
    k_square = reduce_sum(k_square);
    if (dimension < key_head_dim) {
      query[dimension] *= rsqrtf(q_square + 1.0e-6F) *
                          rsqrtf(static_cast<float>(key_head_dim));
      key[dimension] *= rsqrtf(k_square + 1.0e-6F);
    }
    __syncthreads();
    const auto scalar =
        static_cast<std::size_t>(row) * value_heads + value_head;
    const float beta =
        1.0F / (1.0F + expf(-projected_b[scalar]));
    const float a = projected_a[scalar] + dt_bias[value_head];
    const float softplus =
        log1pf(expf(-fabsf(a))) + fmaxf(a, 0.0F);
    const float decay = expf(-expf(a_log[value_head]) * softplus);
    float core = 0.0F;
    if (dimension < value_head_dim) {
      auto* column = recurrent +
          static_cast<std::size_t>(value_head) * key_head_dim *
              value_head_dim +
          dimension;
      float memory = 0.0F;
      for (std::uint32_t index = 0U; index < key_head_dim; ++index) {
        auto& state =
            column[static_cast<std::size_t>(index) * value_head_dim];
        state *= decay;
        memory += state * key[index];
      }
      const float value =
          row_conv[2U * key_dim +
                   static_cast<std::size_t>(value_head) * value_head_dim +
                   dimension];
      const float delta = (value - memory) * beta;
      for (std::uint32_t index = 0U; index < key_head_dim; ++index) {
        auto& state =
            column[static_cast<std::size_t>(index) * value_head_dim];
        state += key[index] * delta;
        core += state * query[index];
      }
      per_dimension[dimension] = core * core;
    } else {
      per_dimension[dimension] = 0.0F;
    }
    __syncthreads();
    const float square = reduce_sum(per_dimension[dimension]);
    if (dimension < value_head_dim) {
      const auto value_index =
          static_cast<std::size_t>(row) * value_dim +
          static_cast<std::size_t>(value_head) * value_head_dim + dimension;
      const float z = projected_z[value_index];
      output[value_index] =
          core *
          rsqrtf(square / static_cast<float>(value_head_dim) + epsilon) *
          norm_weight[dimension] * (z / (1.0F + expf(-z)));
    }
    __syncthreads();
  }
}

__global__ void argmax_batch_kernel(const float* values, std::uint32_t count,
                                    std::uint32_t* output) {
  const auto request = static_cast<std::uint32_t>(blockIdx.x);
  values += static_cast<std::size_t>(request) * count;
  __shared__ float best_values[kThreads];
  __shared__ std::uint32_t best_indices[kThreads];
  float best = kNegativeInfinity;
  std::uint32_t index = 0xffffffffU;
  for (std::uint32_t item = threadIdx.x; item < count;
       item += blockDim.x) {
    const auto value = values[item];
    if (value > best || (value == best && item < index)) {
      best = value;
      index = item;
    }
  }
  best_values[threadIdx.x] = best;
  best_indices[threadIdx.x] = index;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      const auto other_value = best_values[threadIdx.x + stride];
      const auto other_index = best_indices[threadIdx.x + stride];
      if (other_value > best_values[threadIdx.x] ||
          (other_value == best_values[threadIdx.x] &&
           other_index < best_indices[threadIdx.x])) {
        best_values[threadIdx.x] = other_value;
        best_indices[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) output[request] = best_indices[0];
}

Status checked(cudaError_t error, const char* name) {
  return error == cudaSuccess ? Status::success() : Status(ErrorCode::upload_failed, std::string(name) + ": " + cudaGetErrorString(error));
}

}  // namespace

Status fp4_embedding(const Fp4Block32Matrix& m, std::uint32_t token,
                     float* output, void* raw) noexcept {
  if (!m.weights || !m.scales || !output || token >= m.rows || !m.columns ||
      m.padded_columns < m.columns || m.padded_columns % 32U)
    return Status(ErrorCode::invalid_argument, "invalid FP4 embedding");
  fp4_embedding_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, token, m.columns, m.padded_columns, output);
  return checked(cudaPeekAtLastError(), "FP4 embedding");
}
Status fp4_embedding_batch(const Fp4Block32Matrix& m,
                           const std::uint32_t* tokens, float* output,
                           std::uint32_t batch, void* raw) noexcept {
  if (!m.weights || !m.scales || !tokens || !output || !batch || !m.rows ||
      !m.columns || m.padded_columns < m.columns ||
      m.padded_columns % 32U)
    return Status(ErrorCode::invalid_argument, "invalid FP4 embedding batch");
  fp4_embedding_batch_kernel<<<batch, kThreads, 0,
                               static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, tokens, m.rows, m.columns, m.padded_columns,
      output);
  return checked(cudaPeekAtLastError(), "FP4 embedding batch");
}

Status quantize_q8_batch(const float* input, std::int8_t* output,
                         float* scales, std::uint32_t rows,
                         std::uint32_t columns,
                         std::uint32_t padded_columns, void* raw) noexcept {
  if (!input || !output || !scales || !rows || !columns ||
      padded_columns < columns || padded_columns % 32U)
    return Status(ErrorCode::invalid_argument,
                  "invalid Q8 activation quantization");
  quantize_q8_batch_kernel<<<rows, kThreads, 0,
                             static_cast<cudaStream_t>(raw)>>>(
      input, output, scales, columns, padded_columns);
  return checked(cudaPeekAtLastError(), "Q8 activation quantization");
}

Status fp4_gemv_q8_batch(const Fp4Block32Matrix& m,
                         const std::int8_t* input,
                         const float* input_scales, float* output,
                         std::uint32_t batch, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !input_scales || !output ||
      !m.rows || !m.columns || !batch || m.padded_columns < m.columns ||
      m.padded_columns % 32U)
    return Status(ErrorCode::invalid_argument, "invalid FP4 Q8 GEMV");
  const auto items = static_cast<std::uint64_t>(m.rows) * batch;
  const auto blocks = (items + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (blocks > 0xffffffffULL)
    return Status(ErrorCode::invalid_argument, "FP4 Q8 GEMV grid too large");
  fp4_gemv_q8_batch_kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                              static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, input_scales, output, m.rows,
      m.padded_columns, batch);
  return checked(cudaPeekAtLastError(), "FP4 Q8 GEMV");
}
Status fp4_gemv_q8_batch_weight_reuse(
    const Fp4Block32Matrix& m, const std::int8_t* input,
    const float* input_scales, float* output, std::uint32_t batch,
    void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !input_scales || !output ||
      !m.rows || !m.columns || m.padded_columns < m.columns ||
      m.padded_columns % 32U || !batch ||
      batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid FP4 Q8 weight-reuse GEMV");
  const auto blocks =
      (static_cast<std::uint64_t>(m.rows) + kWarpsPerBlock - 1U) /
      kWarpsPerBlock;
  if (blocks > 0xffffffffULL)
    return Status(ErrorCode::invalid_argument,
                  "FP4 Q8 weight-reuse GEMV grid too large");
  const auto grid = static_cast<unsigned>(blocks);
  const auto stream = static_cast<cudaStream_t>(raw);
#define LAUNCH_WEIGHT_REUSE(Batch)                                           \
  fp4_gemv_q8_batch_weight_reuse_kernel<Batch><<<grid, kThreads, 0, stream>>>(\
      m.weights, m.scales, input, input_scales, output, m.rows,              \
      m.padded_columns)
  switch (batch) {
    case 1U: LAUNCH_WEIGHT_REUSE(1U); break;
    case 2U: LAUNCH_WEIGHT_REUSE(2U); break;
    case 3U: LAUNCH_WEIGHT_REUSE(3U); break;
    case 4U: LAUNCH_WEIGHT_REUSE(4U); break;
    case 5U: LAUNCH_WEIGHT_REUSE(5U); break;
    case 6U: LAUNCH_WEIGHT_REUSE(6U); break;
    case 7U: LAUNCH_WEIGHT_REUSE(7U); break;
    case 8U: LAUNCH_WEIGHT_REUSE(8U); break;
    default: break;
  }
#undef LAUNCH_WEIGHT_REUSE
  return checked(cudaPeekAtLastError(), "FP4 Q8 weight-reuse GEMV");
}
Status fp4_gemm_q8_block32(const Fp4Block32Matrix& m,
                           const std::int8_t* input,
                           const float* input_scales, float* output,
                           std::uint32_t batch, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !input_scales || !output ||
      !m.rows || !m.columns || m.padded_columns < m.columns ||
      m.padded_columns % kFp4GemmBlockColumns || !batch)
    return Status(ErrorCode::invalid_argument,
                  "invalid FP4 Q8 block-32 GEMM");
  const dim3 grid((m.rows + kFp4GemmOutputTile - 1U) /
                      kFp4GemmOutputTile,
                  (batch + kFp4GemmBatchTile - 1U) /
                      kFp4GemmBatchTile);
  static_assert(kFp4GemmWarps == 8U);
  fp4_gemm_q8_block32_kernel<<<grid, kFp4GemmThreads, 0,
                              static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, input_scales, output, m.rows,
      m.padded_columns, batch);
  return checked(cudaPeekAtLastError(), "FP4 Q8 block-32 GEMM");
}

Status embedding(const Int8Matrix& m, std::uint32_t token, float* output, void* raw) noexcept {
  if (!m.weights || !m.scales || !output || token >= m.rows || !m.columns) return Status(ErrorCode::invalid_argument, "invalid embedding");
  embedding_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(m.weights, m.scales, token, m.columns, output);
  return checked(cudaPeekAtLastError(), "embedding");
}
Status gemv(const Int8Matrix& m, const float* input, float* output, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows || !m.columns) return Status(ErrorCode::invalid_argument, "invalid gemv");
  const auto blocks = (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (m.columns % 4U == 0U) {
    int8_gemv_vector_kernel<<<blocks, kThreads, 0,
                              static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns);
  } else {
    int8_gemv_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns);
  }
  return checked(cudaPeekAtLastError(), "int8 gemv");
}
Status gemv_grouped_inputs(const Int8Matrix& m, const float* input,
                           float* output, std::uint32_t groups,
                           void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !groups || m.rows % groups != 0U ||
      m.columns % 4U != 0U)
    return Status(ErrorCode::invalid_argument,
                  "invalid grouped-input gemv");
  const auto blocks = (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  int8_gemv_grouped_inputs_vector_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, output, m.rows, m.columns,
      m.rows / groups);
  return checked(cudaPeekAtLastError(), "grouped-input int8 gemv");
}
Status gemv_grouped_inputs_batch_weight_reuse(
    const Int8Matrix& m, const float* input, float* output,
    std::uint32_t groups, std::uint32_t batch, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !groups || m.rows % groups != 0U ||
      m.columns % 4U != 0U || !batch || batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid grouped-input weight-reuse batched gemv");
  const auto blocks = (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  int8_gemv_grouped_inputs_batch_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, output, m.rows, m.columns,
      m.rows / groups, groups, batch);
  return checked(cudaPeekAtLastError(),
                 "grouped-input weight-reuse int8 batched gemv");
}
Status gemv_batch(const Int8Matrix& m, const float* input, float* output,
                  std::uint32_t batch, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !batch)
    return Status(ErrorCode::invalid_argument, "invalid batched gemv");
  const auto blocks = static_cast<std::uint64_t>(m.rows) * batch;
  if (blocks > 0xffffffffULL)
    return Status(ErrorCode::invalid_argument, "batched gemv grid too large");
  const auto grid = (blocks + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  if (m.columns % 4U == 0U) {
    int8_gemv_batch_vector_kernel<<<static_cast<unsigned>(grid), kThreads, 0,
                                    static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns, batch);
  } else {
    int8_gemv_batch_kernel<<<static_cast<unsigned>(grid), kThreads, 0,
                             static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales, input, output, m.rows, m.columns, batch);
  }
  return checked(cudaPeekAtLastError(), "int8 batched gemv");
}
Status gemv_batch_weight_reuse(const Int8Matrix& m, const float* input,
                               float* output, std::uint32_t batch,
                               void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !batch || batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched gemv");
  const auto blocks =
      (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  int8_gemv_batch_weight_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, output, m.rows, m.columns, batch);
  return checked(cudaPeekAtLastError(), "weight-reuse int8 batched gemv");
}
Status gemv_f32(const float* matrix, std::uint32_t rows, std::uint32_t columns, const float* input, float* output, void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns) return Status(ErrorCode::invalid_argument, "invalid f32 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  f32_gemv_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(matrix, input, output, rows, columns);
  return checked(cudaPeekAtLastError(), "f32 gemv");
}
Status gemv_f32_batch(const float* matrix, std::uint32_t rows,
                      std::uint32_t columns, const float* input, float* output,
                      std::uint32_t batch, void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch)
    return Status(ErrorCode::invalid_argument, "invalid batched f32 gemv");
  const auto blocks = static_cast<std::uint64_t>(rows) * batch;
  if (blocks > 0xffffffffULL)
    return Status(ErrorCode::invalid_argument,
                  "batched f32 gemv grid too large");
  const auto grid = (blocks + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  f32_gemv_batch_kernel<<<static_cast<unsigned>(grid), kThreads, 0,
                          static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns, batch);
  return checked(cudaPeekAtLastError(), "f32 batched gemv");
}
Status gemv_f32_batch_weight_reuse(
    const float* matrix, std::uint32_t rows, std::uint32_t columns,
    const float* input, float* output, std::uint32_t batch,
    void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch ||
      batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched f32 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  f32_gemv_batch_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns, batch);
  return checked(cudaPeekAtLastError(),
                 "weight-reuse f32 batched gemv");
}
Status gemv_bf16(const std::uint16_t* matrix, std::uint32_t rows,
                  std::uint32_t columns, const float* input, float* output,
                  void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns)
    return Status(ErrorCode::invalid_argument, "invalid bf16 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  bf16_gemv_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns);
  return checked(cudaPeekAtLastError(), "bf16 gemv");
}
Status gemv_bf16_batch(const std::uint16_t* matrix, std::uint32_t rows,
                       std::uint32_t columns, const float* input,
                       float* output, std::uint32_t batch,
                       void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch ||
      batch > kMaximumWeightReuseBatch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched bf16 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  bf16_gemv_batch_reuse_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      matrix, input, output, rows, columns, batch);
  return checked(cudaPeekAtLastError(), "weight-reuse bf16 batched gemv");
}
Status rms_norm(const float* input, const float* weight, float* output, std::uint32_t elements, float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !elements || epsilon <= 0) return Status(ErrorCode::invalid_argument, "invalid rms norm");
  rms_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "rms norm");
}
Status rms_norm_bf16_weight(const float* input, const std::uint16_t* weight,
                            float* output, std::uint32_t elements,
                            float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !elements || epsilon <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid bf16-weight rms norm");
  rms_bf16_weight_kernel<<<1, kThreads, 0,
                           static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "bf16-weight rms norm");
}
Status rms_norm_bf16_weight_batch(
    const float* input, const std::uint16_t* weight, float* output,
    std::uint32_t rows, std::uint32_t elements, float epsilon,
    void* raw) noexcept {
  if (!input || !weight || !output || !rows || !elements || epsilon <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid batched bf16-weight rms norm");
  rms_bf16_weight_batch_kernel<<<rows, kThreads, 0,
                                 static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "batched bf16-weight rms norm");
}
Status qwen3_next_rms_norm(const float* input, const float* weight,
                           float* output, std::uint32_t elements,
                           float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !elements || epsilon <= 0)
    return Status(ErrorCode::invalid_argument, "invalid Qwen3-Next rms norm");
  qwen_rms_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "Qwen3-Next rms norm");
}
Status zero_centered_rms_norm_batch(
    const float* input, const float* weight, float* output,
    std::uint32_t rows, std::uint32_t elements, float epsilon,
    void* raw) noexcept {
  if (!input || !weight || !output || !rows || !elements || epsilon <= 0.0F)
    return Status(ErrorCode::invalid_argument,
                  "invalid zero-centered RMS norm batch");
  zero_centered_rms_batch_kernel<<<rows, kThreads, 0,
                                  static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "zero-centered RMS norm batch");
}
Status add_in_place(float* destination, const float* source, std::uint32_t elements, void* raw) noexcept {
  if (!destination || !source || !elements) return Status(ErrorCode::invalid_argument, "invalid add");
  add_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(destination, source, elements);
  return checked(cudaPeekAtLastError(), "add");
}
Status silu_product(const float* gate, const float* up, float* output,
                    std::uint32_t elements, void* raw) noexcept {
  if (!gate || !up || !output || !elements)
    return Status(ErrorCode::invalid_argument, "invalid silu product");
  silu_product_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                        static_cast<cudaStream_t>(raw)>>>(gate, up, output,
                                                          elements);
  return checked(cudaPeekAtLastError(), "silu product");
}
Status sigmoid_scale_in_place(float* values, const float* gate,
                              std::uint32_t elements, void* raw) noexcept {
  if (!values || !gate || !elements)
    return Status(ErrorCode::invalid_argument, "invalid sigmoid scale");
  sigmoid_scale_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0,
                         static_cast<cudaStream_t>(raw)>>>(values, gate,
                                                           elements);
  return checked(cudaPeekAtLastError(), "sigmoid scale");
}
Status qkv_rope_cache(float* q, float* k, const float* v, const float* qw, const float* kw, float* kc, float* vc, std::uint32_t pos, std::uint32_t heads, std::uint32_t dim, float eps, float theta, void* raw) noexcept {
  if (!q || !k || !v || !qw || !kw || !kc || !vc || !heads || dim != 128 || eps <= 0 || theta <= 0) return Status(ErrorCode::invalid_argument, "invalid qkv rope");
  auto stream = static_cast<cudaStream_t>(raw);
  const auto elements = heads * dim;
  rms_kernel<<<1, kThreads, 0, stream>>>(q, qw, q, elements, eps);
  if (auto status = checked(cudaPeekAtLastError(), "q rms norm"); !status.ok()) return status;
  rms_kernel<<<1, kThreads, 0, stream>>>(k, kw, k, elements, eps);
  if (auto status = checked(cudaPeekAtLastError(), "k rms norm"); !status.ok()) return status;
  qkv_rope_kernel<<<heads, 128, 0, stream>>>(q, k, v, kc, vc, pos, heads, dim, theta);
  return checked(cudaPeekAtLastError(), "qkv rope");
}
Status attention_decode(const float* q, const float* kc, const float* vc, float* output, std::uint32_t tokens, std::uint32_t heads, std::uint32_t dim, void* raw) noexcept {
  if (!q || !kc || !vc || !output || !tokens || !heads || dim > kThreads) return Status(ErrorCode::invalid_argument, "invalid attention");
  attention_kernel<<<heads, kThreads, (static_cast<std::size_t>(tokens) + 1) * sizeof(float), static_cast<cudaStream_t>(raw)>>>(q, kc, vc, output, tokens, heads, dim);
  return checked(cudaPeekAtLastError(), "attention");
}
Status router_topk(const float* input, const float* weights, std::uint32_t hidden, std::uint32_t experts, std::uint32_t top_k, float* logits, float* scores, std::uint32_t* indices, void* raw) noexcept {
  if (!input || !weights || !logits || !scores || !indices || !hidden || !experts || !top_k || top_k > experts) return Status(ErrorCode::invalid_argument, "invalid router");
  auto stream = static_cast<cudaStream_t>(raw);
  router_logits_kernel<<<experts, kThreads, 0, stream>>>(input, weights, logits, hidden, experts);
  if (auto status = checked(cudaPeekAtLastError(), "router logits"); !status.ok()) return status;
  router_select_kernel<<<1, 1, 0, stream>>>(logits, experts, top_k, scores,
                                             indices, false);
  return checked(cudaPeekAtLastError(), "router select");
}
Status router_topk_normalized(const float* input, const float* weights,
                              std::uint32_t hidden, std::uint32_t experts,
                              std::uint32_t top_k, float* logits,
                              float* scores, std::uint32_t* indices,
                              void* raw) noexcept {
  if (!input || !weights || !logits || !scores || !indices || !hidden ||
      !experts || !top_k || top_k > experts)
    return Status(ErrorCode::invalid_argument, "invalid normalized router");
  auto stream = static_cast<cudaStream_t>(raw);
  router_logits_kernel<<<experts, kThreads, 0, stream>>>(
      input, weights, logits, hidden, experts);
  if (auto status = checked(cudaPeekAtLastError(), "router logits");
      !status.ok())
    return status;
  router_select_kernel<<<1, 1, 0, stream>>>(logits, experts, top_k, scores,
                                             indices, true);
  return checked(cudaPeekAtLastError(), "normalized router select");
}
Status router_topk_normalized_batch(
    const float* input, const float* weights, std::uint32_t rows,
    std::uint32_t hidden, std::uint32_t experts, std::uint32_t top_k,
    float* logits, float* scores, std::uint32_t* indices,
    void* raw) noexcept {
  if (!input || !weights || !logits || !scores || !indices || !rows ||
      !hidden || !experts || !top_k || top_k > experts ||
      top_k > kMaximumRouterTopK)
    return Status(ErrorCode::invalid_argument,
                  "invalid normalized batched router");
  auto stream = static_cast<cudaStream_t>(raw);
  auto status = gemv_f32_batch(weights, experts, hidden, input, logits, rows,
                               raw);
  if (!status.ok()) return status;
  router_select_batch_kernel<<<rows, kThreads, 0, stream>>>(
      logits, experts, top_k, scores, indices);
  return checked(cudaPeekAtLastError(), "normalized batched router select");
}
Status sigmoid_bias_router_topk_batch(
    const float* input, const float* weights, const float* expert_bias,
    std::uint32_t rows, std::uint32_t hidden, std::uint32_t experts,
    std::uint32_t top_k, float normalization_epsilon,
    float routed_scaling_factor, float* logits, float* scores,
    std::uint32_t* indices, void* raw) noexcept {
  if (!input || !weights || !expert_bias || !logits || !scores || !indices ||
      !rows || !hidden || !experts || !top_k || top_k > experts ||
      top_k > kMaximumRouterTopK || !(normalization_epsilon > 0.0F) ||
      !(routed_scaling_factor > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid sigmoid-bias batched router");
  auto status = gemv_f32_batch(weights, experts, hidden, input, logits, rows,
                               raw);
  if (!status.ok()) return status;
  sigmoid_bias_router_select_batch_kernel<<<
      rows, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      logits, expert_bias, experts, top_k, normalization_epsilon,
      routed_scaling_factor, scores, indices);
  return checked(cudaPeekAtLastError(),
                 "sigmoid-bias batched router select");
}
Status deepseek_router_hash(
    const float* input, const std::uint16_t* weights,
    const std::int64_t* token_experts, std::uint32_t token_id,
    float* logits, float* scores, std::uint32_t* indices,
    float route_scale, void* raw) noexcept {
  if (!input || !weights || !token_experts || !logits || !scores ||
      !indices || token_id >= 129280U || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek hash router");
  auto status = gemv_bf16(weights, 256U, 4096U, input, logits, raw);
  if (!status.ok()) return status;
  deepseek_hash_router_kernel<<<1, 32, 0,
                                static_cast<cudaStream_t>(raw)>>>(
      logits, token_experts, token_id, route_scale, scores, indices);
  return checked(cudaPeekAtLastError(), "DeepSeek hash router select");
}
Status deepseek_router_learned(
    const float* input, const std::uint16_t* weights,
    const float* selection_bias, float* logits, float* scores,
    std::uint32_t* indices, float route_scale, void* raw) noexcept {
  if (!input || !weights || !selection_bias || !logits || !scores ||
      !indices || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek learned router");
  auto status = gemv_bf16(weights, 256U, 4096U, input, logits, raw);
  if (!status.ok()) return status;
  deepseek_learned_router_kernel<<<1, kThreads, 0,
                                   static_cast<cudaStream_t>(raw)>>>(
      logits, selection_bias, route_scale, scores, indices);
  return checked(cudaPeekAtLastError(), "DeepSeek learned router select");
}
Status deepseek_router_hash_batch(
    const float* input, const std::uint16_t* weights,
    const std::int64_t* token_experts, std::uint32_t token_zero,
    std::uint32_t token_one, float* logits, float* scores,
    std::uint32_t* indices, float route_scale, void* raw) noexcept {
  if (!input || !weights || !token_experts || !logits || !scores ||
      !indices || token_zero >= 129280U || token_one >= 129280U ||
      !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek batched hash router");
  auto status = gemv_bf16_batch(weights, 256U, 4096U, input, logits, 2U, raw);
  if (!status.ok()) return status;
  deepseek_hash_router_batch_kernel<<<2U, 32U, 0,
                                      static_cast<cudaStream_t>(raw)>>>(
      logits, token_experts, token_zero, token_one, route_scale, scores,
      indices);
  return checked(cudaPeekAtLastError(),
                 "DeepSeek batched hash router select");
}
Status deepseek_router_learned_batch(
    const float* input, const std::uint16_t* weights,
    const float* selection_bias, float* logits, float* scores,
    std::uint32_t* indices, float route_scale, void* raw) noexcept {
  if (!input || !weights || !selection_bias || !logits || !scores ||
      !indices || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek batched learned router");
  auto status = gemv_bf16_batch(weights, 256U, 4096U, input, logits, 2U, raw);
  if (!status.ok()) return status;
  deepseek_learned_router_batch_kernel<<<
      2U, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      logits, selection_bias, route_scale, scores, indices);
  return checked(cudaPeekAtLastError(),
                 "DeepSeek batched learned router select");
}
Status qwen3_next_qkv_rope_cache(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, float* key_cache,
    float* value_cache, std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* raw) noexcept {
  if (!q_and_gate || !key || !value || !q_norm_weight || !k_norm_weight ||
      !key_cache || !value_cache || !query_heads || !kv_heads ||
      query_heads % kv_heads || !head_dim || head_dim > kThreads ||
      !rotary_dim || rotary_dim > head_dim || rotary_dim % 2U ||
      epsilon <= 0 || rope_theta <= 0)
    return Status(ErrorCode::invalid_argument, "invalid Qwen3-Next qkv rope");
  qwen_qkv_rope_kernel<<<query_heads, kThreads, 0,
                         static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight, key_cache,
      value_cache, position, query_heads, kv_heads, head_dim, rotary_dim,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(), "Qwen3-Next qkv rope");
}
Status qwen3_next_attention_decode(
    const float* q_and_gate, const float* key_cache,
    const float* value_cache, float* output, std::uint32_t context_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* raw) noexcept {
  if (!q_and_gate || !key_cache || !value_cache || !output ||
      !context_tokens || !query_heads || !kv_heads || query_heads % kv_heads ||
      !head_dim || head_dim > kThreads)
    return Status(ErrorCode::invalid_argument,
                  "invalid Qwen3-Next attention decode");
  qwen_attention_kernel<<<query_heads, kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key_cache, value_cache, output, context_tokens, query_heads,
      kv_heads, head_dim);
  return checked(cudaPeekAtLastError(), "Qwen3-Next attention");
}
Status gqa_qkv_rope_cache(
    float* query, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, float* key_cache,
    float* value_cache, std::uint32_t position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* raw) noexcept {
  if (!query || !key || !value || !q_norm_weight || !k_norm_weight ||
      !key_cache || !value_cache || !query_heads || !kv_heads ||
      query_heads % kv_heads || !head_dim || head_dim > kThreads ||
      !rotary_dim || rotary_dim > head_dim || rotary_dim % 2U ||
      !(epsilon > 0.0F) || !(rope_theta > 0.0F))
    return Status(ErrorCode::invalid_argument, "invalid GQA QKV rope");
  gqa_qkv_rope_kernel<<<query_heads, kThreads, 0,
                        static_cast<cudaStream_t>(raw)>>>(
      query, key, value, q_norm_weight, k_norm_weight, key_cache,
      value_cache, position, query_heads, kv_heads, head_dim, rotary_dim,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(), "GQA QKV rope");
}
Status gqa_attention_decode(
    const float* query, const float* key_cache, const float* value_cache,
    float* output, std::uint32_t context_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* raw) noexcept {
  if (!query || !key_cache || !value_cache || !output || !context_tokens ||
      !query_heads || !kv_heads || query_heads % kv_heads || !head_dim ||
      head_dim > kThreads)
    return Status(ErrorCode::invalid_argument, "invalid GQA attention");
  gqa_attention_kernel<<<query_heads, kThreads, 0,
                         static_cast<cudaStream_t>(raw)>>>(
      query, key_cache, value_cache, output, context_tokens, query_heads,
      kv_heads, head_dim);
  return checked(cudaPeekAtLastError(), "GQA attention");
}
Status qwen3_next_qkv_rope_cache_paged_fp16(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* raw) noexcept {
  if (!q_and_gate || !key || !value || !q_norm_weight || !k_norm_weight ||
      !page || !page_tokens || !query_heads || !kv_heads ||
      query_heads % kv_heads || !head_dim || head_dim > kThreads ||
      !rotary_dim || rotary_dim > head_dim || rotary_dim % 2U ||
      epsilon <= 0 || rope_theta <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid paged Qwen3-Next qkv rope");
  qwen_qkv_rope_paged_fp16_kernel<<<query_heads, kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight,
      static_cast<__half*>(page), full_attention_layer, page_tokens, position,
      query_heads, kv_heads, head_dim, rotary_dim, epsilon, rope_theta);
  return checked(cudaPeekAtLastError(), "paged Qwen3-Next qkv rope");
}
Status qwen3_next_attention_decode_paged_fp16(
    const float* q_and_gate, const void* const* page_table,
    float* output, std::uint32_t context_tokens,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* raw) noexcept {
  if (!q_and_gate || !page_table || !output || !context_tokens ||
      !page_tokens || !query_heads || !kv_heads || query_heads % kv_heads ||
      !head_dim || head_dim > kThreads)
    return Status(ErrorCode::invalid_argument,
                  "invalid paged Qwen3-Next attention decode");
  qwen_attention_paged_fp16_kernel<<<query_heads, kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, page_table, output, context_tokens, full_attention_layer,
      page_tokens, query_heads, kv_heads, head_dim);
  return checked(cudaPeekAtLastError(), "paged Qwen3-Next attention");
}
Status gated_gqa_qkv_rope_cache_paged_fp4(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t position, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* raw) noexcept {
  return gated_gqa_qkv_rope_cache_paged_fp4_at(
      q_and_gate, key, value, q_norm_weight, k_norm_weight, page,
      full_attention_layer, page_tokens, position, position, query_heads,
      kv_heads, head_dim, rotary_dim, epsilon, rope_theta, raw);
}
Status gated_gqa_qkv_rope_cache_paged_fp4_at(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight, void* page,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t cache_position, std::uint32_t rotary_position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* raw) noexcept {
  if (!q_and_gate || !key || !value || !q_norm_weight || !k_norm_weight ||
      !page || !page_tokens || !query_heads || !kv_heads ||
      query_heads % kv_heads || query_heads / kv_heads > 8U || !head_dim ||
      head_dim > kThreads || head_dim % 32U || !rotary_dim ||
      rotary_dim > head_dim || rotary_dim % 2U || !(epsilon > 0.0F) ||
      !(rope_theta > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid paged FP4 gated GQA QKV rope");
  gated_gqa_qkv_rope_paged_fp4_kernel<<<
      query_heads, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight,
      static_cast<std::uint8_t*>(page), full_attention_layer, page_tokens,
      cache_position, rotary_position, query_heads, kv_heads, head_dim,
      rotary_dim, epsilon, rope_theta);
  return checked(cudaPeekAtLastError(),
                 "paged FP4 gated GQA QKV rope");
}
Status gated_gqa_qkv_rope_cache_paged_fp4_batch(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t first_rotary_position, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float rope_theta, void* raw) noexcept {
  if (!q_and_gate || !key || !value || !q_norm_weight || !k_norm_weight ||
      !page_table || !page_tokens || !rows || !query_heads || !kv_heads ||
      query_heads % kv_heads || query_heads / kv_heads > 8U || !head_dim ||
      head_dim > kThreads || head_dim % 32U || !rotary_dim ||
      rotary_dim > head_dim || rotary_dim % 2U || !(epsilon > 0.0F) ||
      !(rope_theta > 0.0F) ||
      first_cache_position > 0xffffffffU - (rows - 1U) ||
      first_rotary_position > 0xffffffffU - (rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid paged FP4 gated GQA QKV rope batch");
  const dim3 grid(query_heads, rows);
  gated_gqa_qkv_rope_paged_fp4_batch_kernel<<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight, page_table,
      full_attention_layer, page_tokens, first_cache_position,
      first_rotary_position, query_heads, kv_heads, head_dim, rotary_dim,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(),
                 "paged FP4 gated GQA QKV rope batch");
}
Status gated_gqa_attention_decode_paged_fp4(
    const PagedFp4GatedGqaAttentionLaunch& launch) noexcept {
  if (!launch.q_and_gate || !launch.page_table || !launch.output ||
      !launch.partial_maxima || !launch.partial_sums ||
      !launch.partial_outputs || !launch.context_tokens ||
      !launch.page_tokens || !launch.query_heads || !launch.kv_heads ||
      launch.query_heads % launch.kv_heads ||
      launch.query_heads / launch.kv_heads > 8U || !launch.head_dim ||
      launch.head_dim > kThreads || launch.head_dim % 32U ||
      !launch.split_tokens || !launch.maximum_splits)
    return Status(ErrorCode::invalid_argument,
                  "invalid paged FP4 gated GQA attention");
  const auto active_splits =
      (launch.context_tokens + launch.split_tokens - 1U) /
      launch.split_tokens;
  if (active_splits > launch.maximum_splits || active_splits > 65535U)
    return Status(ErrorCode::invalid_argument,
                  "paged FP4 gated GQA split workspace is too small");
  const dim3 producer_grid(launch.kv_heads, active_splits);
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  gated_gqa_attention_paged_fp4_split_kernel<<<
      producer_grid, kThreads, 0, stream>>>(
      launch.q_and_gate, launch.page_table, launch.partial_maxima,
      launch.partial_sums, launch.partial_outputs, launch.context_tokens,
      launch.full_attention_layer, launch.page_tokens, launch.query_heads,
      launch.kv_heads, launch.head_dim, launch.split_tokens);
  auto status = checked(cudaPeekAtLastError(),
                        "paged FP4 gated GQA attention split");
  if (!status.ok()) return status;
  gated_gqa_attention_paged_fp4_combine_kernel<<<
      launch.query_heads, kThreads, 0, stream>>>(
      launch.q_and_gate, launch.partial_maxima, launch.partial_sums,
      launch.partial_outputs, launch.output, active_splits,
      launch.query_heads, launch.head_dim);
  return checked(cudaPeekAtLastError(),
                 "paged FP4 gated GQA attention combine");
}
Status gated_gqa_attention_decode_paged_fp4_tensor_core(
    const PagedFp4GatedGqaAttentionLaunch& launch) noexcept {
  if (!launch.q_and_gate || !launch.page_table || !launch.output ||
      !launch.partial_maxima || !launch.partial_sums ||
      !launch.partial_outputs || !launch.context_tokens ||
      !launch.page_tokens || !launch.query_heads || !launch.kv_heads ||
      launch.query_heads % launch.kv_heads ||
      launch.query_heads / launch.kv_heads > 8U || !launch.head_dim ||
      launch.head_dim > kThreads || launch.head_dim % 32U ||
      !launch.split_tokens || !launch.maximum_splits)
    return Status(ErrorCode::invalid_argument,
                  "invalid paged FP4 Tensor Core gated GQA attention");
  const auto active_splits =
      (launch.context_tokens + launch.split_tokens - 1U) /
      launch.split_tokens;
  if (active_splits > launch.maximum_splits || active_splits > 65535U)
    return Status(ErrorCode::invalid_argument,
                  "paged FP4 Tensor Core GQA split workspace is too small");
  const dim3 producer_grid(launch.kv_heads, active_splits);
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  if (launch.head_dim <= 128U) {
    gated_gqa_attention_paged_fp4_tensor_core_split_kernel<
        kFp4GemmThreads, 128U><<<
        producer_grid, kFp4GemmThreads, 0, stream>>>(
        launch.q_and_gate, launch.page_table, launch.partial_maxima,
        launch.partial_sums, launch.partial_outputs, launch.context_tokens,
        1U, launch.maximum_splits, launch.full_attention_layer,
        launch.page_tokens, launch.query_heads, launch.kv_heads,
        launch.head_dim, launch.split_tokens);
  } else {
    gated_gqa_attention_paged_fp4_tensor_core_split_kernel<
        kFp4GemmThreads, 256U><<<
        producer_grid, kFp4GemmThreads, 0, stream>>>(
        launch.q_and_gate, launch.page_table, launch.partial_maxima,
        launch.partial_sums, launch.partial_outputs, launch.context_tokens,
        1U, launch.maximum_splits, launch.full_attention_layer,
        launch.page_tokens, launch.query_heads, launch.kv_heads,
        launch.head_dim, launch.split_tokens);
  }
  auto status = checked(cudaPeekAtLastError(),
                        "paged FP4 Tensor Core GQA attention split");
  if (!status.ok()) return status;
  gated_gqa_attention_paged_fp4_combine_kernel<<<
      launch.query_heads, kThreads, 0, stream>>>(
      launch.q_and_gate, launch.partial_maxima, launch.partial_sums,
      launch.partial_outputs, launch.output, active_splits,
      launch.query_heads, launch.head_dim);
  return checked(cudaPeekAtLastError(),
                 "paged FP4 Tensor Core GQA attention combine");
}
Status gated_gqa_attention_microbatch_paged_fp4_tensor_core(
    const PagedFp4GatedGqaPrefillLaunch& launch) noexcept {
  if (!launch.q_and_gate || !launch.page_table || !launch.output ||
      !launch.partial_maxima || !launch.partial_sums ||
      !launch.partial_outputs || !launch.first_context_tokens ||
      !launch.rows || !launch.page_tokens || !launch.query_heads ||
      !launch.kv_heads || launch.query_heads % launch.kv_heads ||
      launch.rows * (launch.query_heads / launch.kv_heads) > 16U ||
      !launch.head_dim || launch.head_dim > kThreads ||
      launch.head_dim % 32U || !launch.split_tokens ||
      !launch.maximum_splits ||
      launch.first_context_tokens > 0xffffffffU - (launch.rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid paged FP4 Tensor Core GQA microbatch");
  const auto last_context_tokens =
      launch.first_context_tokens + launch.rows - 1U;
  const auto active_splits =
      (last_context_tokens + launch.split_tokens - 1U) /
      launch.split_tokens;
  if (active_splits > launch.maximum_splits || active_splits > 65535U)
    return Status(
        ErrorCode::invalid_argument,
        "paged FP4 Tensor Core GQA microbatch workspace is too small");
  const dim3 producer_grid(launch.kv_heads, active_splits);
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  if (launch.head_dim <= 128U) {
    gated_gqa_attention_paged_fp4_tensor_core_split_kernel<
        kTensorCoreAttentionThreads, 128U><<<
        producer_grid, kTensorCoreAttentionThreads, 0, stream>>>(
        launch.q_and_gate, launch.page_table, launch.partial_maxima,
        launch.partial_sums, launch.partial_outputs,
        launch.first_context_tokens, launch.rows, launch.maximum_splits,
        launch.full_attention_layer, launch.page_tokens, launch.query_heads,
        launch.kv_heads, launch.head_dim, launch.split_tokens);
  } else {
    gated_gqa_attention_paged_fp4_tensor_core_split_kernel<
        kTensorCoreAttentionThreads, 256U><<<
        producer_grid, kTensorCoreAttentionThreads, 0, stream>>>(
        launch.q_and_gate, launch.page_table, launch.partial_maxima,
        launch.partial_sums, launch.partial_outputs,
        launch.first_context_tokens, launch.rows, launch.maximum_splits,
        launch.full_attention_layer, launch.page_tokens, launch.query_heads,
        launch.kv_heads, launch.head_dim, launch.split_tokens);
  }
  auto status = checked(
      cudaPeekAtLastError(),
      "paged FP4 Tensor Core GQA microbatch attention split");
  if (!status.ok()) return status;
  const dim3 combine_grid(launch.query_heads, launch.rows);
  gated_gqa_attention_paged_fp4_prefill_combine_kernel<<<
      combine_grid, kThreads, 0, stream>>>(
      launch.q_and_gate, launch.partial_maxima, launch.partial_sums,
      launch.partial_outputs, launch.output, launch.first_context_tokens,
      launch.maximum_splits, launch.split_tokens, launch.query_heads,
      launch.head_dim);
  return checked(cudaPeekAtLastError(),
                 "paged FP4 Tensor Core GQA microbatch combine");
}
Status gated_gqa_attention_prefill_paged_fp4(
    const PagedFp4GatedGqaPrefillLaunch& launch) noexcept {
  if (!launch.q_and_gate || !launch.page_table || !launch.output ||
      !launch.partial_maxima || !launch.partial_sums ||
      !launch.partial_outputs || !launch.first_context_tokens ||
      !launch.rows || !launch.page_tokens || !launch.query_heads ||
      !launch.kv_heads || launch.query_heads % launch.kv_heads ||
      launch.query_heads / launch.kv_heads > 8U || !launch.head_dim ||
      launch.head_dim > kThreads || launch.head_dim % 32U ||
      !launch.split_tokens || !launch.maximum_splits ||
      launch.first_context_tokens > 0xffffffffU - (launch.rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid paged FP4 gated GQA prefill");
  const auto last_context = launch.first_context_tokens + launch.rows - 1U;
  const auto active_splits =
      (last_context + launch.split_tokens - 1U) / launch.split_tokens;
  if (active_splits > launch.maximum_splits || active_splits > 65535U)
    return Status(ErrorCode::invalid_argument,
                  "paged FP4 prefill split workspace is too small");
  constexpr std::uint32_t kQueryTile = 16U;
  const dim3 producer_grid(
      launch.query_heads, active_splits,
      (launch.rows + kQueryTile - 1U) / kQueryTile);
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  gated_gqa_attention_paged_fp4_prefill_split_kernel<<<
      producer_grid, kThreads, 0, stream>>>(
      launch.q_and_gate, launch.page_table, launch.partial_maxima,
      launch.partial_sums, launch.partial_outputs,
      launch.first_context_tokens, launch.rows, launch.maximum_splits,
      launch.full_attention_layer, launch.page_tokens, launch.query_heads,
      launch.kv_heads, launch.head_dim, launch.split_tokens);
  auto status = checked(cudaPeekAtLastError(),
                        "paged FP4 gated GQA prefill split");
  if (!status.ok()) return status;
  const dim3 combine_grid(launch.query_heads, launch.rows);
  gated_gqa_attention_paged_fp4_prefill_combine_kernel<<<
      combine_grid, kThreads, 0, stream>>>(
      launch.q_and_gate, launch.partial_maxima, launch.partial_sums,
      launch.partial_outputs, launch.output, launch.first_context_tokens,
      launch.maximum_splits, launch.split_tokens, launch.query_heads,
      launch.head_dim);
  return checked(cudaPeekAtLastError(),
                 "paged FP4 gated GQA prefill combine");
}
Status qwen3_next_delta_decode(const Qwen3NextDeltaLaunch& launch) noexcept {
  if (!launch.projected_qkvz || !launch.projected_ba ||
      !launch.conv_weights || !launch.dt_bias || !launch.a_log ||
      !launch.norm_weight || !launch.conv_state || !launch.recurrent_state ||
      !launch.conv_output || !launch.output || !launch.key_heads ||
      !launch.value_heads || launch.value_heads % launch.key_heads ||
      !launch.key_head_dim || launch.key_head_dim > kThreads ||
      !launch.value_head_dim || launch.value_head_dim > kThreads ||
      !launch.conv_kernel || launch.conv_kernel > 16U || launch.epsilon <= 0)
    return Status(ErrorCode::invalid_argument,
                  "invalid Qwen3-Next delta launch");
  const auto key_dim = launch.key_heads * launch.key_head_dim;
  const auto value_dim = launch.value_heads * launch.value_head_dim;
  const auto conv_dim = 2U * key_dim + value_dim;
  auto stream = static_cast<cudaStream_t>(launch.stream);
  qwen_delta_conv_kernel<<<(conv_dim + kThreads - 1U) / kThreads, kThreads, 0,
                            stream>>>(
      launch.projected_qkvz, launch.conv_weights, launch.conv_state,
      launch.conv_output, launch.key_heads, launch.value_heads, key_dim,
      value_dim, launch.key_head_dim, launch.value_head_dim,
      launch.conv_kernel);
  if (auto status = checked(cudaPeekAtLastError(), "Qwen3-Next delta conv");
      !status.ok())
    return status;
  qwen_delta_recurrent_kernel<<<launch.value_heads, kThreads, 0, stream>>>(
      launch.projected_qkvz, launch.projected_ba, launch.conv_output,
      launch.dt_bias, launch.a_log, launch.norm_weight,
      launch.recurrent_state, launch.output, launch.key_heads,
      launch.value_heads, launch.key_head_dim, launch.value_head_dim,
      launch.epsilon);
  return checked(cudaPeekAtLastError(), "Qwen3-Next delta recurrent");
}
Status split_gated_delta_decode(const SplitGatedDeltaLaunch& launch) noexcept {
  if (!launch.projected_qkv || !launch.projected_z ||
      !launch.projected_b || !launch.projected_a || !launch.conv_weights ||
      !launch.dt_bias || !launch.a_log || !launch.norm_weight ||
      !launch.conv_state || !launch.recurrent_state || !launch.conv_output ||
      !launch.output || !launch.key_heads || !launch.value_heads ||
      launch.value_heads % launch.key_heads || !launch.key_head_dim ||
      launch.key_head_dim > kThreads || !launch.value_head_dim ||
      launch.value_head_dim > kThreads || !launch.conv_kernel ||
      launch.conv_kernel > 16U || launch.epsilon <= 0.0F)
    return Status(ErrorCode::invalid_argument,
                  "invalid split gated-delta launch");
  const auto key_dim = launch.key_heads * launch.key_head_dim;
  const auto value_dim = launch.value_heads * launch.value_head_dim;
  const auto conv_dim = 2U * key_dim + value_dim;
  auto stream = static_cast<cudaStream_t>(launch.stream);
  split_delta_conv_kernel<<<(conv_dim + kThreads - 1U) / kThreads, kThreads,
                              0, stream>>>(
      launch.projected_qkv, launch.conv_weights, launch.conv_state,
      launch.conv_output, conv_dim, launch.conv_kernel);
  if (auto status = checked(cudaPeekAtLastError(), "split delta conv");
      !status.ok())
    return status;
  split_delta_recurrent_kernel<<<launch.value_heads, kThreads, 0, stream>>>(
      launch.projected_z, launch.projected_b, launch.projected_a,
      launch.conv_output, launch.dt_bias, launch.a_log, launch.norm_weight,
      launch.recurrent_state, launch.output, launch.key_heads,
      launch.value_heads, launch.key_head_dim, launch.value_head_dim,
      launch.epsilon);
  return checked(cudaPeekAtLastError(), "split delta recurrent");
}
Status split_gated_delta_prefill(
    const SplitGatedDeltaPrefillLaunch& launch) noexcept {
  if (!launch.projected_qkv || !launch.projected_z ||
      !launch.projected_b || !launch.projected_a || !launch.conv_weights ||
      !launch.dt_bias || !launch.a_log || !launch.norm_weight ||
      !launch.conv_state || !launch.recurrent_state || !launch.conv_output ||
      !launch.output || !launch.rows || !launch.key_heads ||
      !launch.value_heads || launch.value_heads % launch.key_heads ||
      !launch.key_head_dim || launch.key_head_dim > kThreads ||
      !launch.value_head_dim || launch.value_head_dim > kThreads ||
      !launch.conv_kernel || launch.conv_kernel > 16U ||
      launch.epsilon <= 0.0F)
    return Status(ErrorCode::invalid_argument,
                  "invalid split gated-delta prefill launch");
  const auto key_dim = launch.key_heads * launch.key_head_dim;
  const auto value_dim = launch.value_heads * launch.value_head_dim;
  const auto conv_dim = 2U * key_dim + value_dim;
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  split_delta_conv_prefill_kernel<<<
      (conv_dim + kThreads - 1U) / kThreads, kThreads, 0, stream>>>(
      launch.projected_qkv, launch.conv_weights, launch.conv_state,
      launch.conv_output, launch.rows, conv_dim, launch.conv_kernel);
  auto status = checked(cudaPeekAtLastError(),
                        "split gated-delta prefill conv");
  if (!status.ok()) return status;
  split_delta_recurrent_prefill_kernel<<<
      launch.value_heads, kThreads, 0, stream>>>(
      launch.projected_z, launch.projected_b, launch.projected_a,
      launch.conv_output, launch.dt_bias, launch.a_log, launch.norm_weight,
      launch.recurrent_state, launch.output, launch.rows, launch.key_heads,
      launch.value_heads, launch.key_head_dim, launch.value_head_dim,
      launch.epsilon);
  return checked(cudaPeekAtLastError(),
                 "split gated-delta prefill recurrent");
}
Status causal_short_conv_decode(
    const CausalShortConvLaunch& launch) noexcept {
  if (!launch.projected_bcx || !launch.weights || !launch.state ||
      !launch.output || !launch.rows || !launch.hidden || !launch.kernel ||
      launch.kernel > 32U)
    return Status(ErrorCode::invalid_argument,
                  "invalid causal short-convolution launch");
  const dim3 grid((launch.hidden + kThreads - 1U) / kThreads, launch.rows);
  causal_short_conv_decode_kernel<<<
      grid, kThreads, 0, static_cast<cudaStream_t>(launch.stream)>>>(
      launch.projected_bcx, launch.weights, launch.state, launch.output,
      launch.hidden, launch.kernel);
  return checked(cudaPeekAtLastError(), "causal short convolution");
}
Status argmax(const float* values, std::uint32_t count, std::uint32_t* output, void* raw) noexcept {
  if (!values || !count || !output) return Status(ErrorCode::invalid_argument, "invalid argmax");
  argmax_batch_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(values, count, output);
  return checked(cudaPeekAtLastError(), "argmax");
}
Status argmax_batch(const float* values, std::uint32_t count,
                    std::uint32_t batch, std::uint32_t* output,
                    void* raw) noexcept {
  if (!values || !count || !batch || !output)
    return Status(ErrorCode::invalid_argument, "invalid batched argmax");
  argmax_batch_kernel<<<batch, kThreads, 0,
                        static_cast<cudaStream_t>(raw)>>>(values, count,
                                                          output);
  return checked(cudaPeekAtLastError(), "batched argmax");
}

}  // namespace expert::runtime::cuda
