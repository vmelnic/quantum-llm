#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace expert::runtime::cuda {
namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = kThreads / kWarpSize;
constexpr unsigned kMaximumRouterTopK = 32;
constexpr unsigned kMaximumWeightReuseBatch = 8;
constexpr unsigned kFp4GemmBatchTile = 128;
constexpr unsigned kFp4GemmOutputTile = 32;
constexpr unsigned kFp4GemmBlockColumns = 32;
constexpr unsigned kFp4GemmThreads = 256;
constexpr unsigned kFp4GemmWarps = kFp4GemmThreads / kWarpSize;
constexpr unsigned kFp4PrefillGemmThreads = 512;
constexpr unsigned kFp4PrefillGemmWarps =
    kFp4PrefillGemmThreads / kWarpSize;
constexpr unsigned kTensorCoreAttentionThreads = 512;
// A 16-token tile keeps decoded K/V values within the default 48-KiB SM86
// shared-memory limit. FP4 values times UE8M0 power-of-two scales are exactly
// representable in BF16, so the attention kernels can halve shared traffic
// without adding another quantization step.
constexpr unsigned kAttentionKvTile = 16;
constexpr unsigned kMaximumAttentionHeadDim = 256;
constexpr float kNegativeInfinity = -3.402823466e+38F;

enum class PagedKvEncoding : std::uint8_t { fp4, fp8 };

template <PagedKvEncoding Encoding>
__host__ __device__ constexpr std::uint32_t paged_kv_record_bytes(
    std::uint32_t head_dim) {
  if constexpr (Encoding == PagedKvEncoding::fp8)
    return head_dim + sizeof(std::uint16_t);
  return head_dim / 2U + head_dim / 32U;
}

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

__device__ float block_max_warps(float value, float* workspace) {
  const auto lane = threadIdx.x % kWarpSize;
  const auto warp = threadIdx.x / kWarpSize;
  value = warp_max(value);
  if (lane == 0U) workspace[warp] = value;
  __syncthreads();
  value = warp == 0U && lane < kWarpsPerBlock
              ? workspace[lane]
              : kNegativeInfinity;
  if (warp == 0U) value = warp_max(value);
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

__device__ float reduce_max(float value) {
  __shared__ float shared[kThreads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                 shared[threadIdx.x + stride]);
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

template <PagedKvEncoding Encoding>
__device__ void store_paged_kv_record(
    std::uint8_t* page, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t cache_position,
    std::uint32_t kv_head, std::uint32_t kv_heads,
    std::uint32_t head_dim, const float* key, const float* value) {
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto record_bytes = paged_kv_record_bytes<Encoding>(head_dim);
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  auto* key_records = page +
      static_cast<std::size_t>(full_attention_layer) * 2U *
          records_per_kind * record_bytes;
  auto* value_records = key_records + records_per_kind * record_bytes;
  const auto record_index =
      static_cast<std::size_t>(cache_position % page_tokens) * kv_heads +
      kv_head;
  auto* key_record = key_records + record_index * record_bytes;
  auto* value_record = value_records + record_index * record_bytes;

  if constexpr (Encoding == PagedKvEncoding::fp8) {
    const auto key_maximum = reduce_max(
        dimension < head_dim ? fabsf(key[dimension]) : 0.0F);
    const auto value_maximum = reduce_max(
        dimension < head_dim ? fabsf(value[dimension]) : 0.0F);
    const auto key_scale_half =
        __float2half_rn(fmaxf(key_maximum / 448.0F, 0x1p-24F));
    const auto value_scale_half =
        __float2half_rn(fmaxf(value_maximum / 448.0F, 0x1p-24F));
    const auto key_scale = __half2float(key_scale_half);
    const auto value_scale = __half2float(value_scale_half);
    if (dimension == 0U) {
      *reinterpret_cast<__half*>(key_record + head_dim) = key_scale_half;
      *reinterpret_cast<__half*>(value_record + head_dim) = value_scale_half;
    }
    if (dimension < head_dim) {
      reinterpret_cast<__nv_fp8_e4m3*>(key_record)[dimension] =
          __nv_fp8_e4m3(key[dimension] / key_scale);
      reinterpret_cast<__nv_fp8_e4m3*>(value_record)[dimension] =
          __nv_fp8_e4m3(value[dimension] / value_scale);
    }
    return;
  }

  // One lane owns one block-32 record. This keeps scale selection and packed
  // nibble writes deterministic and avoids atomics on adjacent values.
  const auto block = dimension;
  if (block >= head_dim / 32U) return;
  const auto first = block * 32U;
  float key_maximum = 0.0F;
  float value_maximum = 0.0F;
  for (std::uint32_t index = 0U; index < 32U; ++index) {
    key_maximum = fmaxf(key_maximum, fabsf(key[first + index]));
    value_maximum = fmaxf(value_maximum, fabsf(value[first + index]));
  }
  const auto key_scale_code = encode_ue8m0_cover(key_maximum);
  const auto value_scale_code = encode_ue8m0_cover(value_maximum);
  const auto key_scale = decode_ue8m0(key_scale_code);
  const auto value_scale = decode_ue8m0(value_scale_code);
  key_record[head_dim / 2U + block] = key_scale_code;
  value_record[head_dim / 2U + block] = value_scale_code;
  for (std::uint32_t index = 0U; index < 16U; ++index) {
    const auto offset = first + 2U * index;
    const auto key_low = encode_fp4_nearest(key[offset], key_scale);
    const auto key_high = encode_fp4_nearest(key[offset + 1U], key_scale);
    const auto value_low = encode_fp4_nearest(value[offset], value_scale);
    const auto value_high =
        encode_fp4_nearest(value[offset + 1U], value_scale);
    key_record[offset / 2U] =
        static_cast<std::uint8_t>(key_low | (key_high << 4U));
    value_record[offset / 2U] =
        static_cast<std::uint8_t>(value_low | (value_high << 4U));
  }
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

template <std::uint32_t Batch, bool CacheAllActivations = false,
          bool CacheActivationTiles = false>
__global__ void fp4_gemv_q8_batch_weight_reuse_kernel(
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    const std::int8_t* input, const float* input_scales, float* output,
    std::uint32_t rows, std::uint32_t padded_columns) {
  static_assert(Batch > 0U && Batch <= kMaximumWeightReuseBatch);
  const auto warp = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto row = static_cast<std::uint32_t>(
      blockIdx.x * kWarpsPerBlock + warp);
  const auto active = row < rows;
  const auto blocks = padded_columns / 32U;
  constexpr std::uint32_t kActivationWordsPerBlock = 8U;
  static_assert(!(CacheAllActivations && CacheActivationTiles));
  const auto activation_block_stride =
      CacheActivationTiles
          ? kWarpSize + 4U
          : ((blocks + kWarpSize - 1U) / kWarpSize) * kWarpSize + 4U;
  extern __shared__ int cached_activations[];
  if constexpr (CacheAllActivations) {
    static_assert(Batch == 2U);
    const auto words_per_request = blocks * kActivationWordsPerBlock;
    for (std::uint32_t item = threadIdx.x;
         item < Batch * words_per_request; item += blockDim.x) {
      const auto request = item / words_per_request;
      const auto local = item % words_per_request;
      const auto block = local / kActivationWordsPerBlock;
      const auto group = local % kActivationWordsPerBlock;
      cached_activations[
          (request * kActivationWordsPerBlock + group) *
              activation_block_stride +
          block] = reinterpret_cast<const int*>(
              input + static_cast<std::size_t>(request) * padded_columns)[
              local];
    }
    __syncthreads();
  }
  if constexpr (!CacheActivationTiles) {
    if (!active) return;
  }
  const auto* row_weights = reinterpret_cast<const int4*>(
      weights + static_cast<std::size_t>(active ? row : 0U) *
                    (padded_columns / 2U));
  const auto* row_scales = weight_scales +
      static_cast<std::size_t>(active ? row : 0U) * blocks;
  float totals[Batch]{};
  const int* activation_blocks[Batch];
#pragma unroll
  for (std::uint32_t request = 0U; request < Batch; ++request) {
    if constexpr (!CacheAllActivations && !CacheActivationTiles)
      activation_blocks[request] = reinterpret_cast<const int*>(
          input + static_cast<std::size_t>(request) * padded_columns);
  }
  for (std::uint32_t tile_first = 0U; tile_first < blocks;
       tile_first += kWarpSize) {
    const auto tile_blocks = min(kWarpSize, blocks - tile_first);
    if constexpr (CacheActivationTiles) {
      static_assert(Batch == 2U);
      const auto words_per_request =
          tile_blocks * kActivationWordsPerBlock;
      for (std::uint32_t item = threadIdx.x;
           item < Batch * words_per_request; item += blockDim.x) {
        const auto request = item / words_per_request;
        const auto local = item % words_per_request;
        const auto local_block = local / kActivationWordsPerBlock;
        const auto group = local % kActivationWordsPerBlock;
        const auto source = static_cast<std::size_t>(request) * blocks *
                                kActivationWordsPerBlock +
                            static_cast<std::size_t>(tile_first) *
                                kActivationWordsPerBlock +
                            local;
        cached_activations[
            (request * kActivationWordsPerBlock + group) *
                activation_block_stride +
            local_block] = reinterpret_cast<const int*>(input)[source];
      }
      __syncthreads();
    }
    const auto block = tile_first + lane;
    if (active && block < blocks) {
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
          int activation{};
          if constexpr (CacheAllActivations) {
            activation = cached_activations[
                (request * kActivationWordsPerBlock + group) *
                    activation_block_stride +
                block];
          } else if constexpr (CacheActivationTiles) {
            activation = cached_activations[
                (request * kActivationWordsPerBlock + group) *
                    activation_block_stride +
                lane];
          } else {
            activation = activation_blocks[request][
                block * kActivationWordsPerBlock + group];
          }
          block_totals[request] =
              __dp4a(packed, activation, block_totals[request]);
        }
      }
#pragma unroll
      for (std::uint32_t request = 0U; request < Batch; ++request)
        totals[request] +=
            static_cast<float>(block_totals[request]) * weight_scale;
    }
    if constexpr (CacheActivationTiles) __syncthreads();
  }
  if (!active) return;
#pragma unroll
  for (std::uint32_t request = 0U; request < Batch; ++request) {
    const auto value = warp_sum(totals[request]) *
                       input_scales[request] * 0.5F;
    if (lane == 0U)
      output[static_cast<std::size_t>(request) * rows + row] = value;
  }
}

// A CTA owns [up to 128 activation rows, 32 output rows]. Q8 integers and
// FP4-twice values multiplied by their power-of-two block scales are exactly
// representable in BF16. Baking the weight scale into the BF16 operand lets
// each of sixteen warps keep one FP32 accumulator fragment across the
// complete K axis;
// the old integer path had to store, rescale, and reload a 64x32 product after
// every K=32 block.
__global__ void fp4_gemm_q8_block32_kernel(
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    const std::int8_t* input, const float* input_scales, float* output,
    std::uint32_t output_rows, std::uint32_t padded_columns,
    std::uint32_t batch) {
  using namespace nvcuda;
  constexpr std::uint32_t kOperandStride = kFp4GemmBlockColumns + 8U;
  constexpr std::uint32_t kProductStride = kFp4GemmOutputTile + 4U;
  __shared__ __align__(32) __nv_bfloat16
      activation_tile[kFp4GemmBatchTile][kOperandStride];
  __shared__ __align__(32) __nv_bfloat16
      weight_tile[kFp4GemmOutputTile][kOperandStride];
  __shared__ __align__(32) float
      product_tile[kFp4GemmBatchTile][kProductStride];

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
         item += kFp4PrefillGemmThreads) {
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
         item += kFp4PrefillGemmThreads) {
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
        kOperandStride);
    wmma::load_matrix_sync(weight_fragment,
                           &weight_tile[output_warp * 16U][0],
                           kOperandStride);
    wmma::mma_sync(product_fragment, activation_fragment, weight_fragment,
                   product_fragment);
    wmma::load_matrix_sync(
        activation_fragment, &activation_tile[batch_warp * 16U][16],
        kOperandStride);
    wmma::load_matrix_sync(weight_fragment,
                           &weight_tile[output_warp * 16U][16],
                           kOperandStride);
    wmma::mma_sync(product_fragment, activation_fragment, weight_fragment,
                   product_fragment);
    __syncthreads();
  }

  wmma::store_matrix_sync(
      &product_tile[batch_warp * 16U][output_warp * 16U],
      product_fragment, kProductStride, wmma::mem_row_major);
  __syncthreads();

  for (std::uint32_t local = 0U;
       local < kFp4GemmBatchTile * kFp4GemmOutputTile /
                   kFp4PrefillGemmThreads;
       ++local) {
    const auto item = local_thread + local * kFp4PrefillGemmThreads;
    const auto batch_row = first_batch + item / kFp4GemmOutputTile;
    const auto output_row = first_output + item % kFp4GemmOutputTile;
    if (batch_row < batch && output_row < output_rows)
      output[static_cast<std::size_t>(batch_row) * output_rows + output_row] =
          product_tile[item / kFp4GemmOutputTile]
                      [item % kFp4GemmOutputTile] *
          input_scales[batch_row] * 0.5F;
  }
}

__global__ void fp4_decode_matrix_bf16_kernel(
    const std::uint8_t* weights, const std::uint8_t* weight_scales,
    __nv_bfloat16* decoded, std::uint32_t rows,
    std::uint32_t padded_columns) {
  const auto values = static_cast<std::size_t>(rows) * padded_columns;
  const auto scale_stride = padded_columns / 32U;
  const auto weight_stride = padded_columns / 2U;
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < values;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto row = item / padded_columns;
    const auto column = static_cast<std::uint32_t>(item % padded_columns);
    const auto packed = weights[row * weight_stride + column / 2U];
    const auto code = static_cast<std::uint8_t>(
        (column & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
    const auto scale = decode_ue8m0(
        weight_scales[row * scale_stride + column / 32U]);
    decoded[item] = __float2bfloat16(
        static_cast<float>(decode_fp4_twice(code)) * scale);
  }
}

__global__ void q8_to_bf16_kernel(
    const std::int8_t* input, __nv_bfloat16* decoded,
    std::size_t values) {
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < values;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    decoded[item] = __float2bfloat16(static_cast<float>(input[item]));
}

__global__ void scale_bf16_q8_gemm_kernel(
    float* output, const float* input_scales, std::uint32_t output_rows,
    std::size_t values) {
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < values;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[item] *= input_scales[item / output_rows] * 0.5F;
}

__global__ void embedding_kernel(const std::int8_t* weights,
                                 const float* scales, std::uint32_t token,
                                 std::uint32_t columns, float* output) {
  for (std::uint32_t i = threadIdx.x; i < columns; i += blockDim.x) {
    output[i] = static_cast<float>(weights[static_cast<std::size_t>(token) * columns + i]) * scales[token];
  }
}

__global__ void decode_int8_rows_f32_kernel(
    const std::int8_t* weights, const float* scales, float* decoded,
    std::uint64_t values, std::uint32_t columns) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < values;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    decoded[index] = static_cast<float>(weights[index]) *
                     scales[index / columns];
  }
}

__global__ void decode_bf16_rows_f32_kernel(
    const std::uint16_t* weights, float* decoded, std::uint64_t values) {
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < values;
       index += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    decoded[index] = __uint_as_float(
        static_cast<unsigned>(weights[index]) << 16U);
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

__global__ void rms_batch_kernel(const float* input, const float* weight,
                                 float* output, std::uint32_t count,
                                 float epsilon) {
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
    output[index] = input[index] * inverse * weight[index];
}

__global__ void weightless_rms_batch_kernel(
    const float* input, float* output, std::uint32_t count, float epsilon) {
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
    output[index] = input[index] * inverse;
}

__global__ void add_kernel(float* destination, const float* source,
                           std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) destination[i] += source[i];
}

__global__ void add_bias_kernel(float* destination, const float* bias,
                                std::uint32_t columns,
                                std::size_t values) {
  const auto item = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (item < values) destination[item] += bias[item % columns];
}

__global__ void layer_norm_batch_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    std::uint32_t elements, float epsilon) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  input += static_cast<std::size_t>(row) * elements;
  output += static_cast<std::size_t>(row) * elements;
  float sum = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < elements;
       index += blockDim.x)
    sum += input[index];
  const auto mean = reduce_sum(sum) / static_cast<float>(elements);
  float square = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < elements;
       index += blockDim.x) {
    const auto centered = input[index] - mean;
    square += centered * centered;
  }
  const auto inverse = rsqrtf(
      reduce_sum(square) / static_cast<float>(elements) + epsilon);
  for (std::uint32_t index = threadIdx.x; index < elements;
       index += blockDim.x)
    output[index] = (input[index] - mean) * inverse * weight[index] +
                    bias[index];
}

template <bool Exact>
__global__ void gelu_in_place_kernel(float* values, std::size_t count) {
  const auto item = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (item >= count) return;
  const auto value = values[item];
  if constexpr (Exact) {
    values[item] = 0.5F * value * (1.0F + erff(value * 0.7071067811865475F));
  } else {
    const auto cubic = value * value * value;
    values[item] = 0.5F * value *
        (1.0F + tanhf(0.7978845608028654F *
                      (value + 0.044715F * cubic)));
  }
}

__global__ void add_fp4_position_interpolation_kernel(
    float* hidden, const std::uint8_t* position_weights,
    const std::uint8_t* position_scales,
    const std::uint32_t* interpolation_indices,
    const float* interpolation_weights, std::uint32_t rows,
    std::uint32_t columns, std::uint32_t padded_columns) {
  const auto item = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  const auto values = static_cast<std::size_t>(rows) * columns;
  if (item >= values) return;
  const auto row = static_cast<std::uint32_t>(item / columns);
  const auto column = static_cast<std::uint32_t>(item % columns);
  float position = 0.0F;
  for (std::uint32_t tap = 0U; tap < 4U; ++tap) {
    const auto source = interpolation_indices[4U * row + tap];
    const auto packed = position_weights[
        static_cast<std::size_t>(source) * (padded_columns / 2U) +
        column / 2U];
    const auto code = static_cast<std::uint8_t>(
        (column & 1U) == 0U ? packed & 0x0fU : packed >> 4U);
    const auto scale = decode_ue8m0(position_scales[
        static_cast<std::size_t>(source) * (padded_columns / 32U) +
        column / 32U]);
    position += interpolation_weights[4U * row + tap] *
                static_cast<float>(decode_fp4_twice(code)) * 0.5F * scale;
  }
  hidden[item] += position;
}

__global__ void vision_qkv_rope_kernel(
    float* qkv, const std::uint32_t* row_positions,
    std::uint32_t heads, std::uint32_t head_dim, float theta) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head >= heads || dimension >= head_dim) return;
  const auto hidden = heads * head_dim;
  const auto half = head_dim / 2U;
  const auto frequency_dimension = half / 2U;
  const auto pair = dimension % half;
  const auto axis = pair < frequency_dimension ? 0U : 1U;
  const auto frequency = pair % frequency_dimension;
  const auto position = row_positions[2U * row + axis];
  const auto angle = static_cast<float>(position) *
      powf(theta, -2.0F * static_cast<float>(frequency) /
                       static_cast<float>(half));
  const auto cosine = cosf(angle);
  const auto sine = sinf(angle);
  const auto row_base = static_cast<std::size_t>(row) * 3U * hidden;
  for (std::uint32_t kind = 0U; kind < 2U; ++kind) {
    auto* vector = qkv + row_base + static_cast<std::size_t>(kind) * hidden +
                   static_cast<std::size_t>(head) * head_dim;
    const auto other = dimension < half ? -vector[dimension + half]
                                         : vector[dimension - half];
    const auto value = vector[dimension];
    __syncthreads();
    vector[dimension] = value * cosine + other * sine;
    __syncthreads();
  }
}

__global__ void vision_segment_attention_kernel(
    const float* qkv, const std::uint32_t* segment_first,
    const std::uint32_t* segment_last, float* output,
    std::uint32_t heads, std::uint32_t head_dim) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto query_row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto hidden = heads * head_dim;
  const auto query = qkv + static_cast<std::size_t>(query_row) * 3U * hidden +
                     static_cast<std::size_t>(head) * head_dim;
  float accumulator = 0.0F;
  float maximum = kNegativeInfinity;
  float denominator = 0.0F;
  const auto scale = rsqrtf(static_cast<float>(head_dim));
  for (auto key_row = segment_first[query_row];
       key_row < segment_last[query_row]; ++key_row) {
    const auto* key = qkv + static_cast<std::size_t>(key_row) * 3U * hidden +
                      hidden + static_cast<std::size_t>(head) * head_dim;
    const auto partial = dimension < head_dim
                             ? query[dimension] * key[dimension]
                             : 0.0F;
    const auto score = reduce_sum(partial) * scale;
    const auto next_maximum = fmaxf(maximum, score);
    const auto previous_scale = expf(maximum - next_maximum);
    const auto current_scale = expf(score - next_maximum);
    if (dimension < head_dim) {
      const auto* value = key + hidden;
      accumulator = accumulator * previous_scale +
                    value[dimension] * current_scale;
    }
    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }
  if (dimension < head_dim)
    output[static_cast<std::size_t>(query_row) * hidden +
           static_cast<std::size_t>(head) * head_dim + dimension] =
        accumulator / denominator;
}

__device__ std::uint32_t mrope_axis(std::uint32_t pair,
                                    std::uint32_t section_one,
                                    std::uint32_t section_two) {
  if (pair % 3U == 1U && pair < 3U * section_one) return 1U;
  if (pair % 3U == 2U && pair < 3U * section_two) return 2U;
  return 0U;
}

__global__ void gated_gqa_qk_norm_mrope_kernel(
    float* q_and_gate, float* key, const float* q_weight,
    const float* k_weight, const std::uint32_t* positions_thw,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim,
    std::uint32_t section_one, std::uint32_t section_two,
    float epsilon, float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto query_width = 2U * query_heads * head_dim;
  const auto key_width = kv_heads * head_dim;
  q_and_gate += static_cast<std::size_t>(row) * query_width;
  key += static_cast<std::size_t>(row) * key_width;
  const auto apply = [&](float* vector, const float* weight) {
    float square = dimension < head_dim
                       ? vector[dimension] * vector[dimension]
                       : 0.0F;
    square = reduce_sum(square);
    if (dimension < head_dim)
      normalized[dimension] = vector[dimension] *
          rsqrtf(square / static_cast<float>(head_dim) + epsilon) *
          (1.0F + weight[dimension]);
    __syncthreads();
    if (dimension < head_dim) {
      auto result = normalized[dimension];
      if (dimension < rotary_dim) {
        const auto half = rotary_dim / 2U;
        const auto pair = dimension % half;
        const auto axis = mrope_axis(pair, section_one, section_two);
        const auto position = positions_thw[3U * row + axis];
        const auto angle = static_cast<float>(position) *
            powf(theta, -2.0F * static_cast<float>(pair) /
                             static_cast<float>(rotary_dim));
        const auto other = dimension < half
                               ? -normalized[dimension + half]
                               : normalized[dimension - half];
        result = normalized[dimension] * cosf(angle) + other * sinf(angle);
      }
      vector[dimension] = result;
    }
    __syncthreads();
  };
  if (head < query_heads)
    apply(q_and_gate + static_cast<std::size_t>(head) * 2U * head_dim,
          q_weight);
  if (head < kv_heads)
    apply(key + static_cast<std::size_t>(head) * head_dim, k_weight);
}

template <PagedKvEncoding Encoding>
__global__ void store_gqa_kv_paged_batch_kernel(
    const float* key, const float* value, const void* const* page_table,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t first_cache_position, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto position = first_cache_position + row;
  const auto width = kv_heads * head_dim;
  auto* page = static_cast<std::uint8_t*>(
      const_cast<void*>(page_table[position / page_tokens]));
  store_paged_kv_record<Encoding>(
      page, full_attention_layer, page_tokens, position, head, kv_heads,
      head_dim, key + static_cast<std::size_t>(row) * width +
                    static_cast<std::size_t>(head) * head_dim,
      value + static_cast<std::size_t>(row) * width +
                      static_cast<std::size_t>(head) * head_dim);
}

__global__ void store_gqa_kv_fp16_batch_kernel(
    const float* key, const float* value, __half* fp16_keys,
    __half* fp16_values, std::size_t values) {
  const auto item = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (item < values) {
    fp16_keys[item] = __float2half_rn(key[item]);
    fp16_values[item] = __float2half_rn(value[item]);
  }
}

__global__ void silu_product_kernel(const float* gate, const float* up,
                                    float* output, std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) output[i] = (gate[i] / (1.0F + expf(-gate[i]))) * up[i];
}

__global__ void relu2_in_place_kernel(float* values, std::uint32_t count) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  const auto value = fmaxf(values[index], 0.0F);
  values[index] = value * value;
}

__global__ void deepseek_swiglu_product_kernel(
    const float* gate, const float* up, float* output, std::uint32_t count,
    float limit, bool bf16_output) {
  const auto i =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  auto gate_value = gate[i];
  auto up_value = up[i];
  if (limit > 0.0F) {
    gate_value = fminf(gate_value, limit);
    up_value = fminf(fmaxf(up_value, -limit), limit);
  }
  auto value = (gate_value / (1.0F + expf(-gate_value))) * up_value;
  if (bf16_output) value = __bfloat162float(__float2bfloat16_rn(value));
  output[i] = value;
}

__global__ void sigmoid_scale_kernel(float* values, const float* gate,
                                     std::uint32_t count) {
  const auto i = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) values[i] *= 1.0F / (1.0F + expf(-gate[0]));
}

__global__ void sigmoid_product_kernel(float* values, const float* gate,
                                       std::uint32_t count) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count)
    values[index] *= 1.0F / (1.0F + expf(-gate[index]));
}

__global__ void scaled_tanh_kernel(float* values, std::uint32_t count,
                                   float multiplier, float softcap) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count)
    values[index] = softcap * tanhf(values[index] * multiplier / softcap);
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

template <PagedKvEncoding Encoding>
__global__ void gated_gqa_qkv_rope_paged_kernel(
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

  store_paged_kv_record<Encoding>(
      page, full_attention_layer, page_tokens, cache_position, head, kv_heads,
      head_dim, key_head,
      value + static_cast<std::size_t>(head) * head_dim);
}

template <PagedKvEncoding Encoding>
__global__ void gated_gqa_qkv_rope_paged_batch_kernel(
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
  store_paged_kv_record<Encoding>(
      page, full_attention_layer, page_tokens, cache_position, head, kv_heads,
      head_dim, key_head,
      value + static_cast<std::size_t>(head) * head_dim);
}

__global__ void gated_gqa_qkv_rope_fp16_batch_kernel(
    float* q_and_gate, float* key, const float* value,
    const float* q_weight, const float* k_weight, __half* fp16_keys,
    __half* fp16_values, std::uint32_t first_rotary_position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float theta) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
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
  if (dimension >= head_dim) return;
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
  const auto output =
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim +
      dimension;
  fp16_keys[output] = __float2half_rn(final_key);
  fp16_values[output] = __float2half_rn(
      value[static_cast<std::size_t>(head) * head_dim + dimension]);
}

__device__ float standard_rope_value(const float* source,
                                     std::uint32_t dimension,
                                     std::uint32_t rotary_dim,
                                     std::uint32_t position, float theta) {
  if (dimension >= rotary_dim) return source[dimension];
  const auto half = rotary_dim / 2U;
  const auto pair = dimension % half;
  const float angle = static_cast<float>(position) *
      powf(theta, -2.0F * static_cast<float>(pair) /
                      static_cast<float>(rotary_dim));
  const float other = dimension < half ? -source[dimension + half]
                                       : source[dimension - half];
  return source[dimension] * cosf(angle) + other * sinf(angle);
}

__global__ void standard_gqa_qkv_rope_fp16_batch_kernel(
    float* query, float* key, const float* value, __half* fp16_keys,
    __half* fp16_values, std::uint32_t first_rotary_position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float theta) {
  __shared__ float source[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto position = first_rotary_position + row;
  auto* query_head = query +
      (static_cast<std::size_t>(row) * query_heads + head) * head_dim;
  if (dimension < head_dim) source[dimension] = query_head[dimension];
  __syncthreads();
  if (dimension < head_dim)
    query_head[dimension] = standard_rope_value(
        source, dimension, rotary_dim, position, theta);
  __syncthreads();
  if (head >= kv_heads) return;
  auto* key_head = key +
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim;
  if (dimension < head_dim) source[dimension] = key_head[dimension];
  __syncthreads();
  if (dimension >= head_dim) return;
  const auto encoded_key = standard_rope_value(
      source, dimension, rotary_dim, position, theta);
  key_head[dimension] = encoded_key;
  const auto target =
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim + dimension;
  fp16_keys[target] = __float2half_rn(encoded_key);
  fp16_values[target] = __float2half_rn(value[target]);
}

__global__ void normalized_gqa_qkv_fp16_batch_kernel(
    float* query, float* key, const float* value, __half* fp16_keys,
    __half* fp16_values, std::uint32_t first_rotary_position,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float query_scale, float rope_theta, bool apply_rope) {
  __shared__ float normalized[256];
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  const auto position = first_rotary_position + row;
  auto* query_head = query +
      (static_cast<std::size_t>(row) * query_heads + head) * head_dim;
  float square = dimension < head_dim
      ? query_head[dimension] * query_head[dimension]
      : 0.0F;
  square = reduce_sum(square);
  if (dimension < head_dim)
    normalized[dimension] = query_head[dimension] *
        rsqrtf(square / static_cast<float>(head_dim) + epsilon) * query_scale;
  __syncthreads();
  if (dimension < head_dim)
    query_head[dimension] = apply_rope
        ? standard_rope_value(normalized, dimension, rotary_dim, position,
                              rope_theta)
        : normalized[dimension];
  __syncthreads();
  if (head >= kv_heads) return;

  auto* key_head = key +
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim;
  square = dimension < head_dim
      ? key_head[dimension] * key_head[dimension]
      : 0.0F;
  square = reduce_sum(square);
  if (dimension < head_dim)
    normalized[dimension] = key_head[dimension] *
        rsqrtf(square / static_cast<float>(head_dim) + epsilon);
  __syncthreads();
  if (dimension >= head_dim) return;
  const auto encoded_key = apply_rope
      ? standard_rope_value(normalized, dimension, rotary_dim, position,
                            rope_theta)
      : normalized[dimension];
  key_head[dimension] = encoded_key;
  const auto target =
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim + dimension;
  fp16_keys[target] = __float2half_rn(encoded_key);
  fp16_values[target] = __float2half_rn(value[target]);
}

__global__ void standard_gqa_kv_fp16_batch_kernel(
    const float* key, const float* value, __half* fp16_keys,
    __half* fp16_values, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head >= kv_heads || dimension >= head_dim) return;
  const auto target =
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim + dimension;
  fp16_keys[target] = __float2half_rn(key[target]);
  fp16_values[target] = __float2half_rn(value[target]);
}

__global__ void pack_gqa_kv_fp16_to_paged_fp4_kernel(
    const __half* fp16_keys, const __half* fp16_values,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto block = static_cast<std::uint32_t>(threadIdx.x);
  if (head >= kv_heads || block >= head_dim / 32U) return;
  const auto cache_position = first_cache_position + row;
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
  const auto source =
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim +
      block * 32U;
  float key_maximum = 0.0F;
  float value_maximum = 0.0F;
  for (std::uint32_t index = 0U; index < 32U; ++index) {
    key_maximum =
        fmaxf(key_maximum, fabsf(__half2float(fp16_keys[source + index])));
    value_maximum = fmaxf(
        value_maximum, fabsf(__half2float(fp16_values[source + index])));
  }
  const auto key_scale_code = encode_ue8m0_cover(key_maximum);
  const auto value_scale_code = encode_ue8m0_cover(value_maximum);
  const auto key_scale = decode_ue8m0(key_scale_code);
  const auto value_scale = decode_ue8m0(value_scale_code);
  key_record[head_dim / 2U + block] = key_scale_code;
  value_record[head_dim / 2U + block] = value_scale_code;
  for (std::uint32_t index = 0U; index < 16U; ++index) {
    const auto offset = 2U * index;
    const auto key_low = encode_fp4_nearest(
        __half2float(fp16_keys[source + offset]), key_scale);
    const auto key_high = encode_fp4_nearest(
        __half2float(fp16_keys[source + offset + 1U]), key_scale);
    const auto value_low = encode_fp4_nearest(
        __half2float(fp16_values[source + offset]), value_scale);
    const auto value_high = encode_fp4_nearest(
        __half2float(fp16_values[source + offset + 1U]), value_scale);
    key_record[block * 16U + index] =
        static_cast<std::uint8_t>(key_low | (key_high << 4U));
    value_record[block * 16U + index] =
        static_cast<std::uint8_t>(value_low | (value_high << 4U));
  }
}

__global__ void store_gqa_kv_fp16_to_paged_kernel(
    const __half* fp16_keys, const __half* fp16_values,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto row = static_cast<std::uint32_t>(blockIdx.y);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head >= kv_heads || dimension >= head_dim) return;
  const auto cache_position = first_cache_position + row;
  auto* page = static_cast<__half*>(
      const_cast<void*>(page_table[cache_position / page_tokens]));
  const auto page_values = static_cast<std::size_t>(page_tokens) * kv_heads *
                           head_dim;
  auto* key_page = page + static_cast<std::size_t>(full_attention_layer) *
                              2U * page_values;
  auto* value_page = key_page + page_values;
  const auto page_offset =
      (static_cast<std::size_t>(cache_position % page_tokens) * kv_heads +
       head) * head_dim + dimension;
  const auto source =
      (static_cast<std::size_t>(row) * kv_heads + head) * head_dim +
      dimension;
  key_page[page_offset] = fp16_keys[source];
  value_page[page_offset] = fp16_values[source];
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
template <std::uint32_t BlockThreads, std::uint32_t HeadCapacity,
          PagedKvEncoding Encoding>
__device__ __forceinline__ void
gated_gqa_attention_paged_tensor_core_split_body(
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
  constexpr std::uint32_t kBf16Stride = HeadCapacity + 8U;
  constexpr std::uint32_t kScoreStride = kKeyTile + 4U;
  constexpr std::uint32_t kProbabilityStride = kKeyTile + 8U;
  constexpr std::uint32_t kProductStride = HeadCapacity + 4U;
  __shared__ __align__(32) __nv_bfloat16
      query_values[kQueryTile][kBf16Stride];
  __shared__ __align__(32) __nv_bfloat16
      key_values[kKeyTile][kBf16Stride];
  __shared__ __align__(32) __nv_bfloat16
      value_values[kKeyTile][kBf16Stride];
  constexpr std::uint32_t kScoreWarps =
      HeadCapacity / 64U;
  __shared__ __align__(32) float
      score_partials[kScoreWarps][kQueryTile][kScoreStride];
  __shared__ __align__(32) __nv_bfloat16
      probabilities[kQueryTile][kProbabilityStride];
  __shared__ __align__(32) float
      products[kQueryTile][kProductStride];
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
  const auto record_bytes = paged_kv_record_bytes<Encoding>(head_dim);
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
    constexpr std::uint32_t kPackedBytesPerThread = 8U;
    constexpr std::uint32_t kValuesPerChunk =
        Encoding == PagedKvEncoding::fp8 ? kPackedBytesPerThread
                                         : 2U * kPackedBytesPerThread;
    const auto chunks_per_record = head_dim / kValuesPerChunk;
    const auto chunks_per_kind = tile_tokens * chunks_per_record;
    // Each half-warp owns one complete K or V record. Consecutive lanes
    // therefore read eight packed bytes and write sixteen BF16 values
    // contiguously instead of hitting the same shared-memory banks at a
    // block-32 stride.
    for (std::uint32_t item = local_thread; item < 2U * chunks_per_kind;
         item += blockDim.x) {
      const auto is_value = item >= chunks_per_kind;
      const auto local_item = item - (is_value ? chunks_per_kind : 0U);
      const auto local_token = local_item / chunks_per_record;
      const auto chunk = local_item % chunks_per_record;
      const auto token = tile_first + local_token;
      const auto* page = static_cast<const std::uint8_t*>(
          page_table[token / page_tokens]);
      const auto* page_keys = page +
          static_cast<std::size_t>(full_attention_layer) * 2U *
              records_per_kind * record_bytes;
      const auto* page_values = page_keys + records_per_kind * record_bytes;
      const auto record_index =
          static_cast<std::size_t>(token % page_tokens) * kv_heads + kv_head;
      const auto* source = (is_value ? page_values : page_keys) +
                           record_index * record_bytes;
      auto* target = is_value ? value_values[local_token]
                              : key_values[local_token];
      const auto packed_first = chunk * kPackedBytesPerThread;
      if constexpr (Encoding == PagedKvEncoding::fp8) {
        const auto scale = __half2float(
            *reinterpret_cast<const __half*>(source + head_dim));
        const auto* encoded =
            reinterpret_cast<const __nv_fp8_e4m3*>(source);
#pragma unroll
        for (std::uint32_t offset = 0U; offset < kPackedBytesPerThread;
             ++offset)
          target[packed_first + offset] = __float2bfloat16(
              static_cast<float>(encoded[packed_first + offset]) * scale);
      } else {
        const auto block = packed_first / 16U;
        const auto scale =
            0.5F * decode_ue8m0(source[head_dim / 2U + block]);
        std::uint32_t packed_words[kPackedBytesPerThread / 4U]{};
        if ((record_bytes & 3U) == 0U) {
          const auto* source_words = reinterpret_cast<const std::uint32_t*>(
              source + packed_first);
#pragma unroll
          for (std::uint32_t word = 0U;
               word < kPackedBytesPerThread / 4U; ++word)
            packed_words[word] = source_words[word];
        } else {
#pragma unroll
          for (std::uint32_t byte = 0U; byte < kPackedBytesPerThread; ++byte)
            packed_words[byte / 4U] |=
                static_cast<std::uint32_t>(source[packed_first + byte])
                << (8U * (byte & 3U));
        }
#pragma unroll
        for (std::uint32_t packed_offset = 0U;
             packed_offset < kPackedBytesPerThread;
             packed_offset += 2U) {
          const auto packed_pair = static_cast<std::uint16_t>(
              packed_words[packed_offset / 4U] >>
              (8U * (packed_offset & 3U)));
          const auto decoded = static_cast<std::uint32_t>(packed_fp4x4(
              static_cast<std::uint8_t>(packed_pair),
              static_cast<std::uint8_t>(packed_pair >> 8U)));
          const auto dimension = 2U * (packed_first + packed_offset);
          *reinterpret_cast<__nv_bfloat162*>(target + dimension) =
              __floats2bfloat162_rn(
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded))) * scale,
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded >> 8U))) * scale);
          *reinterpret_cast<__nv_bfloat162*>(target + dimension + 2U) =
              __floats2bfloat162_rn(
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded >> 16U))) * scale,
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded >> 24U))) * scale);
        }
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
                               kBf16Stride);
        wmma::load_matrix_sync(key_fragment, &key_values[0][dimension],
                               kBf16Stride);
        wmma::mma_sync(score_fragment, query_fragment, key_fragment,
                       score_fragment);
      }
      wmma::store_matrix_sync(&score_partials[warp][0][0], score_fragment,
                              kScoreStride, wmma::mem_row_major);
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
                             kProbabilityStride);
      wmma::load_matrix_sync(value_fragment,
                             &value_values[0][output_tile * 16U],
                             kBf16Stride);
      wmma::mma_sync(product_fragment, probability_fragment, value_fragment,
                     product_fragment);
      wmma::store_matrix_sync(&products[0][output_tile * 16U],
                              product_fragment, kProductStride,
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

template <std::uint32_t BlockThreads, std::uint32_t HeadCapacity,
          PagedKvEncoding Encoding>
__global__ void gated_gqa_attention_paged_tensor_core_split_kernel(
    const float* q_and_gate, const void* const* page_table,
    float* partial_maxima, float* partial_sums, float* partial_outputs,
    std::uint32_t first_context_tokens, std::uint32_t rows,
    std::uint32_t maximum_splits, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t split_tokens) {
  gated_gqa_attention_paged_tensor_core_split_body<BlockThreads, HeadCapacity,
                                                    Encoding>(
      q_and_gate, page_table, partial_maxima, partial_sums, partial_outputs,
      first_context_tokens, rows, maximum_splits, full_attention_layer,
      page_tokens, query_heads, kv_heads, head_dim, split_tokens);
}

template <std::uint32_t BlockThreads, std::uint32_t HeadCapacity,
          PagedKvEncoding Encoding>
__global__ __launch_bounds__(BlockThreads, 2) void
gated_gqa_attention_paged_tensor_core_high_occupancy_split_kernel(
    const float* q_and_gate, const void* const* page_table,
    float* partial_maxima, float* partial_sums, float* partial_outputs,
    std::uint32_t first_context_tokens, std::uint32_t rows,
    std::uint32_t maximum_splits, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t split_tokens) {
  gated_gqa_attention_paged_tensor_core_split_body<BlockThreads, HeadCapacity,
                                                    Encoding>(
      q_and_gate, page_table, partial_maxima, partial_sums, partial_outputs,
      first_context_tokens, rows, maximum_splits, full_attention_layer,
      page_tokens, query_heads, kv_heads, head_dim, split_tokens);
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

__global__ __launch_bounds__(kFp4GemmThreads, 2) void
gated_gqa_attention_paged_fp4_prefill_split_kernel(
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
  constexpr std::uint32_t kBf16Stride = kMaximumAttentionHeadDim + 8U;
  constexpr std::uint32_t kScoreStride = kKeyTile + 4U;
  constexpr std::uint32_t kProbabilityStride = kKeyTile + 8U;
  constexpr std::uint32_t kProductStride = kMaximumAttentionHeadDim + 4U;
  constexpr std::uint32_t kScoreWarps =
      kMaximumAttentionHeadDim / 64U;
  __shared__ __align__(32) __nv_bfloat16
      query_values[kQueryTile][kBf16Stride];
  __shared__ __align__(32) __nv_bfloat16
      key_values[kKeyTile][kBf16Stride];
  __shared__ __align__(32) __nv_bfloat16
      value_values[kKeyTile][kBf16Stride];
  __shared__ __align__(32) float
      score_partials[kScoreWarps][kQueryTile][kScoreStride];
  __shared__ __align__(32) __nv_bfloat16
      probabilities[kQueryTile][kProbabilityStride];
  __shared__ __align__(32) float
      products[kQueryTile][kProductStride];
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
  if (local_thread < kQueryTile * kKeyTile)
    probabilities[local_thread / kKeyTile][local_thread % kKeyTile] =
        __float2bfloat16(0.0F);
  __syncthreads();

  for (std::uint32_t tile_first = first_token; tile_first < last_token;
       tile_first += kKeyTile) {
    const auto tile_tokens = min(kKeyTile, last_token - tile_first);
    constexpr std::uint32_t kPackedBytesPerThread = 8U;
    const auto chunks_per_record =
        head_dim / (2U * kPackedBytesPerThread);
    const auto chunks_per_kind = tile_tokens * chunks_per_record;
    for (std::uint32_t item = local_thread; item < 2U * chunks_per_kind;
         item += blockDim.x) {
      const auto is_value = item >= chunks_per_kind;
      const auto local_item = item - (is_value ? chunks_per_kind : 0U);
      const auto local_token = local_item / chunks_per_record;
      const auto chunk = local_item % chunks_per_record;
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
      const auto packed_first = chunk * kPackedBytesPerThread;
      const auto block = packed_first / 16U;
      const auto scale =
          0.5F * decode_ue8m0(source[head_dim / 2U + block]);
      std::uint32_t packed_words[kPackedBytesPerThread / 4U]{};
      if ((record_bytes & 3U) == 0U) {
        const auto* source_words = reinterpret_cast<const std::uint32_t*>(
            source + packed_first);
#pragma unroll
        for (std::uint32_t word = 0U;
             word < kPackedBytesPerThread / 4U; ++word)
          packed_words[word] = source_words[word];
      } else {
#pragma unroll
        for (std::uint32_t byte = 0U; byte < kPackedBytesPerThread; ++byte)
          packed_words[byte / 4U] |=
              static_cast<std::uint32_t>(source[packed_first + byte])
              << (8U * (byte & 3U));
      }
#pragma unroll
      for (std::uint32_t packed_offset = 0U;
           packed_offset < kPackedBytesPerThread;
           packed_offset += 2U) {
        const auto packed_pair = static_cast<std::uint16_t>(
            packed_words[packed_offset / 4U] >>
            (8U * (packed_offset & 3U)));
        const auto decoded = static_cast<std::uint32_t>(packed_fp4x4(
            static_cast<std::uint8_t>(packed_pair),
            static_cast<std::uint8_t>(packed_pair >> 8U)));
        const auto dimension = 2U * (packed_first + packed_offset);
        *reinterpret_cast<__nv_bfloat162*>(target + dimension) =
            __floats2bfloat162_rn(
                static_cast<float>(static_cast<std::int8_t>(
                    static_cast<std::uint8_t>(decoded))) * scale,
                static_cast<float>(static_cast<std::int8_t>(
                    static_cast<std::uint8_t>(decoded >> 8U))) * scale);
        *reinterpret_cast<__nv_bfloat162*>(target + dimension + 2U) =
            __floats2bfloat162_rn(
                static_cast<float>(static_cast<std::int8_t>(
                    static_cast<std::uint8_t>(decoded >> 16U))) * scale,
                static_cast<float>(static_cast<std::int8_t>(
                    static_cast<std::uint8_t>(decoded >> 24U))) * scale);
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
                               kBf16Stride);
        wmma::load_matrix_sync(key_fragment, &key_values[0][dimension],
                               kBf16Stride);
        wmma::mma_sync(score_fragment, query_fragment, key_fragment,
                       score_fragment);
      }
      wmma::store_matrix_sync(&score_partials[warp][0][0], score_fragment,
                              kScoreStride, wmma::mem_row_major);
    }
    __syncthreads();

    for (std::uint32_t local_row = warp; local_row < kQueryTile;
         local_row += kFp4GemmWarps) {
      const auto lane = local_thread % kWarpSize;
      const auto query_row = first_query_row + local_row;
      const auto context_tokens = first_context_tokens + query_row;
      const auto active_row = query_row < rows;
      const auto active_tile = active_row && tile_first < context_tokens;
      if (!active_tile) {
        if (lane < kKeyTile)
          probabilities[local_row][lane] = __float2bfloat16(0.0F);
        if (lane == 0U) row_previous_scales[local_row] = 1.0F;
        continue;
      }
      float lane_score = kNegativeInfinity;
      if (lane < tile_tokens && tile_first + lane < context_tokens) {
        lane_score = 0.0F;
        for (std::uint32_t score_warp = 0U;
             score_warp < active_score_warps; ++score_warp)
          lane_score += score_partials[score_warp][local_row][lane];
        lane_score *= rsqrtf(static_cast<float>(head_dim));
      }
      auto tile_maximum = warp_max(lane_score);
      tile_maximum = __shfl_sync(0xffffffffU, tile_maximum, 0U);
      float next_maximum{};
      float previous_scale{};
      if (lane == 0U) {
        next_maximum = fmaxf(row_maxima[local_row], tile_maximum);
        previous_scale = row_maxima[local_row] == kNegativeInfinity
                             ? 0.0F
                             : expf(row_maxima[local_row] - next_maximum);
      }
      next_maximum = __shfl_sync(0xffffffffU, next_maximum, 0U);
      previous_scale = __shfl_sync(0xffffffffU, previous_scale, 0U);
      const auto probability = lane < tile_tokens &&
                                       tile_first + lane < context_tokens
                                   ? expf(lane_score - next_maximum)
                                   : 0.0F;
      const auto quantized = __float2bfloat16(probability);
      if (lane < kKeyTile)
        probabilities[local_row][lane] = quantized;
      const auto tile_sum = warp_sum(__bfloat162float(quantized));
      if (lane == 0U) {
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
                             kProbabilityStride);
      wmma::load_matrix_sync(value_fragment,
                             &value_values[0][output_tile * 16U],
                             kBf16Stride);
      wmma::mma_sync(product_fragment, probability_fragment, value_fragment,
                     product_fragment);
      wmma::store_matrix_sync(&products[0][output_tile * 16U],
                              product_fragment, kProductStride,
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

__global__ void mamba2_conv_kernel(
    const float* xbc, const float* weights, const float* bias, float* state,
    float* output, std::uint32_t conv_size, std::uint32_t kernel) {
  const auto channel =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (channel >= conv_size) return;
  auto* channel_state = state + static_cast<std::size_t>(channel) * kernel;
  for (std::uint32_t index = 1U; index < kernel; ++index)
    channel_state[index - 1U] = channel_state[index];
  channel_state[kernel - 1U] = xbc[channel];
  const auto* channel_weights =
      weights + static_cast<std::size_t>(channel) * kernel;
  float sum = bias[channel];
  for (std::uint32_t index = 0U; index < kernel; ++index)
    sum += channel_state[index] * channel_weights[index];
  output[channel] = sum / (1.0F + expf(-sum));
}

__global__ void mamba2_state_kernel(
    const float* projected, const float* conv, const float* dt_bias,
    const float* a_log, const float* skip, float* state, float* output,
    std::uint32_t heads, std::uint32_t head_dim,
    std::uint32_t state_size, std::uint32_t groups,
    float time_step_min) {
  const auto head = static_cast<std::uint32_t>(blockIdx.x);
  const auto dimension = static_cast<std::uint32_t>(threadIdx.x);
  if (head >= heads || dimension >= head_dim) return;
  const auto intermediate = heads * head_dim;
  const auto group = head / (heads / groups);
  const auto x = conv[static_cast<std::size_t>(head) * head_dim + dimension];
  const auto dt_raw = projected[intermediate +
                                (intermediate + 2U * groups * state_size) +
                                head] + dt_bias[head];
  const auto dt = fmaxf(
      log1pf(expf(-fabsf(dt_raw))) + fmaxf(dt_raw, 0.0F), time_step_min);
  const auto decay = expf(-expf(a_log[head]) * dt);
  const auto* b = conv + intermediate +
                  static_cast<std::size_t>(group) * state_size;
  const auto* c = conv + intermediate +
                  static_cast<std::size_t>(groups) * state_size +
                  static_cast<std::size_t>(group) * state_size;
  auto* cell = state +
      (static_cast<std::size_t>(head) * head_dim + dimension) * state_size;
  float result = x * skip[head];
  for (std::uint32_t index = 0U; index < state_size; ++index) {
    cell[index] = cell[index] * decay + dt * b[index] * x;
    result += cell[index] * c[index];
  }
  output[static_cast<std::size_t>(head) * head_dim + dimension] = result;
}

__global__ void mamba2_gated_group_norm_kernel(
    const float* projected, float* values, const float* norm_weight,
    std::uint32_t intermediate, std::uint32_t groups, float epsilon) {
  const auto group = static_cast<std::uint32_t>(blockIdx.x);
  const auto group_size = intermediate / groups;
  const auto first = group * group_size;
  float square = 0.0F;
  for (std::uint32_t index = threadIdx.x; index < group_size;
       index += blockDim.x) {
    const auto offset = first + index;
    const auto gate = projected[offset];
    const auto gated = values[offset] * gate / (1.0F + expf(-gate));
    values[offset] = gated;
    square += gated * gated;
  }
  square = reduce_sum(square);
  __shared__ float inverse_rms;
  if (threadIdx.x == 0U)
    inverse_rms = rsqrtf(square / static_cast<float>(group_size) + epsilon);
  __syncthreads();
  for (std::uint32_t index = threadIdx.x; index < group_size;
       index += blockDim.x) {
    const auto offset = first + index;
    values[offset] *= inverse_rms * norm_weight[offset];
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

__global__ void staged_prefill_prepare_kernel(
    const float* q_and_gate, __nv_bfloat16* queries, float* accumulator,
    float* maxima, float* sums, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  const auto grouped_heads = query_heads / kv_heads;
  const auto matrix_rows = rows * grouped_heads;
  const auto values = static_cast<std::size_t>(rows) * query_heads * head_dim;
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < values;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto dimension = static_cast<std::uint32_t>(item % head_dim);
    const auto query = static_cast<std::uint32_t>(item / head_dim);
    const auto row = query / query_heads;
    const auto query_head = query % query_heads;
    const auto kv_head = query_head / grouped_heads;
    const auto local_head = query_head % grouped_heads;
    const auto matrix_row = row * grouped_heads + local_head;
    const auto packed =
        (static_cast<std::size_t>(kv_head) * matrix_rows + matrix_row) *
            head_dim +
        dimension;
    queries[packed] = __float2bfloat16(q_and_gate[
        (static_cast<std::size_t>(row) * query_heads + query_head) * 2U *
            head_dim +
        dimension]);
    accumulator[packed] = 0.0F;
    if (dimension == 0U) {
      maxima[static_cast<std::size_t>(kv_head) * matrix_rows + matrix_row] =
          kNegativeInfinity;
      sums[static_cast<std::size_t>(kv_head) * matrix_rows + matrix_row] =
          0.0F;
    }
  }
}

__global__ void staged_fp16_prepare_kernel(
    const float* q_and_gate, __half* queries, float* accumulator,
    float* maxima, float* sums, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, bool output_gated) {
  const auto grouped_heads = query_heads / kv_heads;
  const auto matrix_rows = rows * grouped_heads;
  const auto values = static_cast<std::size_t>(rows) * query_heads * head_dim;
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < values;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto dimension = static_cast<std::uint32_t>(item % head_dim);
    const auto query = static_cast<std::uint32_t>(item / head_dim);
    const auto row = query / query_heads;
    const auto query_head = query % query_heads;
    const auto kv_head = query_head / grouped_heads;
    const auto local_head = query_head % grouped_heads;
    const auto matrix_row = row * grouped_heads + local_head;
    const auto packed =
        (static_cast<std::size_t>(kv_head) * matrix_rows + matrix_row) *
            head_dim +
        dimension;
    const auto query_stride = (output_gated ? 2U : 1U) * head_dim;
    queries[packed] = __float2half_rn(q_and_gate[
        (static_cast<std::size_t>(row) * query_heads + query_head) *
            query_stride + dimension]);
    accumulator[packed] = 0.0F;
    if (dimension == 0U) {
      maxima[static_cast<std::size_t>(kv_head) * matrix_rows + matrix_row] =
          kNegativeInfinity;
      sums[static_cast<std::size_t>(kv_head) * matrix_rows + matrix_row] =
          0.0F;
    }
  }
}

__global__ void staged_fp16_transpose_kv_kernel(
    const __half* raw_keys, const __half* raw_values, __half* keys,
    __half* values, std::uint32_t tokens, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  const auto count =
      static_cast<std::size_t>(tokens) * kv_heads * head_dim;
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < count;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto dimension = static_cast<std::uint32_t>(item % head_dim);
    const auto record = item / head_dim;
    const auto head = static_cast<std::uint32_t>(record % kv_heads);
    const auto token = static_cast<std::uint32_t>(record / kv_heads);
    const auto target =
        (static_cast<std::size_t>(head) * tokens + token) * head_dim +
        dimension;
    keys[target] = raw_keys[item];
    values[target] = raw_values[item];
  }
}

__global__ void staged_fp16_transpose_paged_kv_kernel(
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_token,
    __half* keys, __half* values, std::uint32_t tokens,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
  const auto count =
      static_cast<std::size_t>(tokens) * kv_heads * head_dim;
  const auto page_values =
      static_cast<std::size_t>(page_tokens) * kv_heads * head_dim;
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < count;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto dimension = static_cast<std::uint32_t>(item % head_dim);
    const auto record = item / head_dim;
    const auto head = static_cast<std::uint32_t>(record % kv_heads);
    const auto token = static_cast<std::uint32_t>(record / kv_heads);
    const auto source_token = first_token + token;
    const auto* page = static_cast<const __half*>(
        page_table[source_token / page_tokens]);
    const auto* page_keys =
        page + static_cast<std::size_t>(full_attention_layer) * 2U *
                   page_values;
    const auto source =
        (static_cast<std::size_t>(source_token % page_tokens) * kv_heads +
         head) * head_dim + dimension;
    const auto target =
        (static_cast<std::size_t>(head) * tokens + token) * head_dim +
        dimension;
    keys[target] = page_keys[source];
    values[target] = page_keys[page_values + source];
  }
}

__global__ void staged_fp16_softmax_kernel(
    const float* scores, __half* probabilities, float* accumulator,
    float* maxima, float* sums, std::uint32_t first_context_tokens,
    std::uint32_t first_token, std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t grouped_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  __shared__ float reduction[kWarpsPerBlock];
  __shared__ float previous_scale_shared;
  const auto matrix_rows = rows * grouped_heads;
  const auto packed_row = static_cast<std::uint32_t>(blockIdx.x);
  const auto matrix_row = packed_row % matrix_rows;
  const auto row = matrix_row / grouped_heads;
  const auto context_tokens = first_context_tokens + row;
  const auto* row_scores =
      scores + static_cast<std::size_t>(packed_row) * tokens;
  auto* row_probabilities =
      probabilities + static_cast<std::size_t>(packed_row) * tokens;
  float local_maximum = kNegativeInfinity;
  for (std::uint32_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    if (first_token + token < context_tokens)
      local_maximum = fmaxf(
          local_maximum,
          row_scores[token] * rsqrtf(static_cast<float>(head_dim)));
  }
  const auto tile_maximum = block_max_warps(local_maximum, reduction);
  const auto old_maximum = maxima[packed_row];
  const auto next_maximum = fmaxf(old_maximum, tile_maximum);
  float local_sum = 0.0F;
  for (std::uint32_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    float probability{};
    if (first_token + token < context_tokens)
      probability = expf(
          row_scores[token] * rsqrtf(static_cast<float>(head_dim)) -
          next_maximum);
    const auto quantized = __float2half_rn(probability);
    row_probabilities[token] = quantized;
    local_sum += __half2float(quantized);
  }
  const auto tile_sum = block_sum_warps(local_sum, reduction);
  if (threadIdx.x == 0U) {
    const auto previous_scale = old_maximum == kNegativeInfinity
                                    ? 0.0F
                                    : expf(old_maximum - next_maximum);
    previous_scale_shared = previous_scale;
    maxima[packed_row] = next_maximum;
    sums[packed_row] = sums[packed_row] * previous_scale + tile_sum;
  }
  __syncthreads();
  auto* row_accumulator =
      accumulator + static_cast<std::size_t>(packed_row) * head_dim;
  for (std::uint32_t dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x)
    row_accumulator[dimension] *= previous_scale_shared;
}

template <PagedKvEncoding Encoding>
__global__ void staged_prefill_decode_kv_kernel(
    const void* const* page_table, __nv_bfloat16* keys,
    __nv_bfloat16* values, std::uint32_t first_token,
    std::uint32_t tokens, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  constexpr std::uint32_t kPackedBytesPerThread = 8U;
  constexpr std::uint32_t kValuesPerChunk =
      Encoding == PagedKvEncoding::fp8 ? kPackedBytesPerThread
                                       : 2U * kPackedBytesPerThread;
  const auto chunks_per_record = head_dim / kValuesPerChunk;
  const auto chunks_per_kind =
      static_cast<std::size_t>(tokens) * kv_heads * chunks_per_record;
  const auto total = 2U * chunks_per_kind;
  const auto record_bytes = paged_kv_record_bytes<Encoding>(head_dim);
  const auto records_per_kind =
      static_cast<std::size_t>(page_tokens) * kv_heads;
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < total;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto is_value = item >= chunks_per_kind;
    const auto local_item = item - (is_value ? chunks_per_kind : 0U);
    const auto chunk = static_cast<std::uint32_t>(
        local_item % chunks_per_record);
    const auto record = local_item / chunks_per_record;
    const auto kv_head = static_cast<std::uint32_t>(record % kv_heads);
    const auto local_token = static_cast<std::uint32_t>(record / kv_heads);
    const auto token = first_token + local_token;
    const auto* page =
        static_cast<const std::uint8_t*>(page_table[token / page_tokens]);
    const auto* page_keys = page +
        static_cast<std::size_t>(full_attention_layer) * 2U *
            records_per_kind * record_bytes;
    const auto* page_values = page_keys + records_per_kind * record_bytes;
    const auto record_index =
        static_cast<std::size_t>(token % page_tokens) * kv_heads + kv_head;
    const auto* source = (is_value ? page_values : page_keys) +
                         record_index * record_bytes;
    auto* target = (is_value ? values : keys) +
        (static_cast<std::size_t>(kv_head) * tokens + local_token) *
            head_dim;
    const auto packed_first = chunk * kPackedBytesPerThread;
    if constexpr (Encoding == PagedKvEncoding::fp8) {
      const auto scale = __half2float(
          *reinterpret_cast<const __half*>(source + head_dim));
      const auto* encoded =
          reinterpret_cast<const __nv_fp8_e4m3*>(source);
#pragma unroll
      for (std::uint32_t offset = 0U; offset < kPackedBytesPerThread;
           ++offset)
        target[packed_first + offset] = __float2bfloat16(
            static_cast<float>(encoded[packed_first + offset]) * scale);
    } else {
      const auto scale = 0.5F * decode_ue8m0(
          source[head_dim / 2U + packed_first / 16U]);
      std::uint32_t packed_words[kPackedBytesPerThread / 4U]{};
      if ((record_bytes & 3U) == 0U) {
        const auto* source_words = reinterpret_cast<const std::uint32_t*>(
            source + packed_first);
#pragma unroll
        for (std::uint32_t word = 0U;
             word < kPackedBytesPerThread / 4U; ++word)
          packed_words[word] = source_words[word];
      } else {
#pragma unroll
        for (std::uint32_t byte = 0U; byte < kPackedBytesPerThread; ++byte)
          packed_words[byte / 4U] |=
              static_cast<std::uint32_t>(source[packed_first + byte])
              << (8U * (byte & 3U));
      }
#pragma unroll
      for (std::uint32_t word = 0U; word < 2U; ++word) {
        const auto packed_word = packed_words[word];
#pragma unroll
        for (std::uint32_t pair = 0U; pair < 2U; ++pair) {
          const auto packed_pair = static_cast<std::uint16_t>(
              packed_word >> (16U * pair));
          const auto decoded = static_cast<std::uint32_t>(packed_fp4x4(
              static_cast<std::uint8_t>(packed_pair),
              static_cast<std::uint8_t>(packed_pair >> 8U)));
          const auto dimension =
              2U * (packed_first + 4U * word + 2U * pair);
          *reinterpret_cast<__nv_bfloat162*>(target + dimension) =
              __floats2bfloat162_rn(
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded))) * scale,
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded >> 8U))) * scale);
          *reinterpret_cast<__nv_bfloat162*>(target + dimension + 2U) =
              __floats2bfloat162_rn(
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded >> 16U))) * scale,
                  static_cast<float>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(decoded >> 24U))) * scale);
        }
      }
    }
  }
}

__global__ void staged_prefill_softmax_kernel(
    const float* scores, __nv_bfloat16* probabilities,
    float* accumulator, float* maxima, float* sums,
    std::uint32_t first_context_tokens, std::uint32_t first_token,
    std::uint32_t tokens, std::uint32_t rows,
    std::uint32_t grouped_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) {
  __shared__ float reduction[kWarpsPerBlock];
  __shared__ float previous_scale_shared;
  const auto matrix_rows = rows * grouped_heads;
  const auto packed_row = static_cast<std::uint32_t>(blockIdx.x);
  const auto matrix_row = packed_row % matrix_rows;
  const auto row = matrix_row / grouped_heads;
  const auto context_tokens = first_context_tokens + row;
  const auto* row_scores =
      scores + static_cast<std::size_t>(packed_row) * tokens;
  auto* row_probabilities =
      probabilities + static_cast<std::size_t>(packed_row) * tokens;
  float local_maximum = kNegativeInfinity;
  for (std::uint32_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    if (first_token + token < context_tokens)
      local_maximum = fmaxf(
          local_maximum,
          row_scores[token] * rsqrtf(static_cast<float>(head_dim)));
  }
  const auto tile_maximum = block_max_warps(local_maximum, reduction);
  const auto old_maximum = maxima[packed_row];
  const auto next_maximum = fmaxf(old_maximum, tile_maximum);
  float local_sum = 0.0F;
  for (std::uint32_t token = threadIdx.x; token < tokens;
       token += blockDim.x) {
    float probability{};
    if (first_token + token < context_tokens)
      probability = expf(
          row_scores[token] * rsqrtf(static_cast<float>(head_dim)) -
          next_maximum);
    const auto quantized = __float2bfloat16(probability);
    row_probabilities[token] = quantized;
    local_sum += __bfloat162float(quantized);
  }
  const auto tile_sum = block_sum_warps(local_sum, reduction);
  if (threadIdx.x == 0U) {
    const auto previous_scale = old_maximum == kNegativeInfinity
                                    ? 0.0F
                                    : expf(old_maximum - next_maximum);
    previous_scale_shared = previous_scale;
    maxima[packed_row] = next_maximum;
    sums[packed_row] = sums[packed_row] * previous_scale + tile_sum;
  }
  __syncthreads();
  auto* row_accumulator =
      accumulator + static_cast<std::size_t>(packed_row) * head_dim;
  for (std::uint32_t dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x)
    row_accumulator[dimension] *= previous_scale_shared;
}

__global__ void staged_prefill_finalize_kernel(
    const float* q_and_gate, const float* accumulator, const float* sums,
    float* output, std::uint32_t rows, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, bool output_gated) {
  const auto grouped_heads = query_heads / kv_heads;
  const auto matrix_rows = rows * grouped_heads;
  const auto values = static_cast<std::size_t>(rows) * query_heads * head_dim;
  for (std::size_t item =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < values;
       item += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto dimension = static_cast<std::uint32_t>(item % head_dim);
    const auto query = static_cast<std::uint32_t>(item / head_dim);
    const auto row = query / query_heads;
    const auto query_head = query % query_heads;
    const auto kv_head = query_head / grouped_heads;
    const auto local_head = query_head % grouped_heads;
    const auto matrix_row = row * grouped_heads + local_head;
    const auto packed_row =
        static_cast<std::size_t>(kv_head) * matrix_rows + matrix_row;
    auto result = accumulator[packed_row * head_dim + dimension] /
                  sums[packed_row];
    if (output_gated) {
      const auto gate = q_and_gate[
          (static_cast<std::size_t>(row) * query_heads + query_head) * 2U *
              head_dim + head_dim + dimension];
      result /= 1.0F + expf(-gate);
    }
    output[item] = result;
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

__global__ void topk_logits_kernel(const float* values, std::uint32_t count,
                                   std::uint32_t top_k,
                                   float* output_values,
                                   std::uint32_t* output_indices) {
  __shared__ float best_values[kThreads];
  __shared__ std::uint32_t best_indices[kThreads];
  for (std::uint32_t rank = 0U; rank < top_k; ++rank) {
    float best = kNegativeInfinity;
    std::uint32_t index = 0xffffffffU;
    for (std::uint32_t item = threadIdx.x; item < count;
         item += blockDim.x) {
      bool selected = false;
      for (std::uint32_t previous = 0U; previous < rank; ++previous)
        selected = selected || output_indices[previous] == item;
      const auto value = values[item];
      if (!selected && !isnan(value) &&
          (value > best || (value == best && item < index))) {
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
    if (threadIdx.x == 0U) {
      output_values[rank] = best_values[0];
      output_indices[rank] = best_indices[0];
    }
    __syncthreads();
  }
}

struct DeviceTopology final {
  cudaError_t status{cudaSuccess};
  int multiprocessors{};
  int maximum_threads_per_multiprocessor{};
  int maximum_shared_memory_per_multiprocessor{};
};

const DeviceTopology& current_device_topology() {
  static const DeviceTopology topology = [] {
    DeviceTopology result;
    int device{};
    result.status = cudaGetDevice(&device);
    if (result.status != cudaSuccess) return result;
    result.status = cudaDeviceGetAttribute(
        &result.multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (result.status != cudaSuccess) return result;
    result.status = cudaDeviceGetAttribute(
        &result.maximum_threads_per_multiprocessor,
        cudaDevAttrMaxThreadsPerMultiProcessor, device);
    if (result.status != cudaSuccess) return result;
    result.status = cudaDeviceGetAttribute(
        &result.maximum_shared_memory_per_multiprocessor,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    return result;
  }();
  return topology;
}

struct ThreadCublasContext final {
  int device{-1};
  cublasHandle_t handle{};

  ~ThreadCublasContext() {
    if (handle) static_cast<void>(cublasDestroy(handle));
  }
};

cublasStatus_t current_cublas_handle(cudaStream_t stream,
                                     cublasHandle_t& handle) {
  thread_local ThreadCublasContext context;
  int device{};
  auto cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!context.handle || context.device != device) {
    if (context.handle) {
      static_cast<void>(cublasDestroy(context.handle));
      context.handle = nullptr;
    }
    const auto create_status = cublasCreate(&context.handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) return create_status;
    context.device = device;
    const auto math_status =
        cublasSetMathMode(context.handle, CUBLAS_TENSOR_OP_MATH);
    if (math_status != CUBLAS_STATUS_SUCCESS) return math_status;
  }
  const auto stream_status = cublasSetStream(context.handle, stream);
  if (stream_status != CUBLAS_STATUS_SUCCESS) return stream_status;
  handle = context.handle;
  return CUBLAS_STATUS_SUCCESS;
}

Status checked(cublasStatus_t status, const char* name) {
  return status == CUBLAS_STATUS_SUCCESS
             ? Status::success()
             : Status(ErrorCode::upload_failed,
                      std::string(name) + ": cuBLAS status " +
                          std::to_string(static_cast<int>(status)));
}

Status checked(cudaError_t error, const char* name) {
  return error == cudaSuccess ? Status::success() : Status(ErrorCode::upload_failed, std::string(name) + ": " + cudaGetErrorString(error));
}

Status f32_prefill_gemm(const float* matrix, std::uint32_t rows,
                        std::uint32_t columns, const float* input,
                        std::uint32_t input_stride, float* output,
                        std::uint32_t output_stride, std::uint32_t batch,
                        cudaStream_t stream) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch ||
      input_stride < columns || output_stride < rows ||
      rows > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      columns >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      batch > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      input_stride >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      output_stride >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    return {ErrorCode::invalid_argument, "invalid FP32 prefill GEMM"};
  cublasHandle_t handle{};
  auto blas = current_cublas_handle(stream, handle);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return checked(blas, "prepare FP32 prefill GEMM");
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  blas = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows),
      static_cast<int>(batch), static_cast<int>(columns), &alpha, matrix,
      CUDA_R_32F, static_cast<int>(columns), input, CUDA_R_32F,
      static_cast<int>(input_stride), &beta, output, CUDA_R_32F,
      static_cast<int>(output_stride), CUBLAS_COMPUTE_32F_PEDANTIC,
      CUBLAS_GEMM_DEFAULT);
  return checked(blas, "FP32 prefill GEMM");
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
    case 2U: {
      const auto activation_blocks = m.padded_columns / 32U;
      const auto activation_block_stride =
          ((activation_blocks + kWarpSize - 1U) / kWarpSize) * kWarpSize +
          4U;
      const auto shared_bytes = static_cast<std::size_t>(2U) * 8U *
                                activation_block_stride * sizeof(int);
      const auto& topology = current_device_topology();
      if (topology.status != cudaSuccess)
        return checked(topology.status,
                       "query FP4 weight-reuse device topology");
      const auto maximum_thread_blocks =
          topology.maximum_threads_per_multiprocessor /
          static_cast<int>(kThreads);
      const auto minimum_active_blocks =
          std::max(1, maximum_thread_blocks - 1);
      const auto cache_activations =
          shared_bytes * static_cast<std::size_t>(minimum_active_blocks) <=
          static_cast<std::size_t>(
              topology.maximum_shared_memory_per_multiprocessor);
      if (cache_activations) {
        static const auto cache_status = checked(
            cudaFuncSetCacheConfig(
                fp4_gemv_q8_batch_weight_reuse_kernel<2U, true>,
                cudaFuncCachePreferShared),
            "configure FP4 Q8 activation-reuse cache");
        if (!cache_status.ok()) return cache_status;
        fp4_gemv_q8_batch_weight_reuse_kernel<2U, true><<<
            grid, kThreads, shared_bytes, stream>>>(
            m.weights, m.scales, input, input_scales, output, m.rows,
            m.padded_columns);
      } else {
        constexpr std::size_t tile_shared_bytes =
            2U * 8U * (kWarpSize + 4U) * sizeof(int);
        fp4_gemv_q8_batch_weight_reuse_kernel<2U, false, true><<<
            grid, kThreads, tile_shared_bytes, stream>>>(
            m.weights, m.scales, input, input_scales, output, m.rows,
            m.padded_columns);
      }
      break;
    }
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
  static_assert(kFp4PrefillGemmWarps == 16U);
  const dim3 grid((m.rows + kFp4GemmOutputTile - 1U) /
                      kFp4GemmOutputTile,
                  (batch + kFp4GemmBatchTile - 1U) /
                      kFp4GemmBatchTile);
  fp4_gemm_q8_block32_kernel<<<grid, kFp4PrefillGemmThreads, 0,
                              static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, input, input_scales, output, m.rows,
      m.padded_columns, batch);
  return checked(cudaPeekAtLastError(), "FP4 Q8 block-32 GEMM");
}

Status fp4_decode_matrix_bf16(const Fp4Block32Matrix& m,
                              void* decoded_weights,
                              std::size_t decoded_weight_bytes,
                              void* raw) noexcept {
  if (!m.weights || !m.scales || !decoded_weights || !m.rows ||
      !m.columns || m.padded_columns < m.columns ||
      m.padded_columns % kFp4GemmBlockColumns)
    return Status(ErrorCode::invalid_argument,
                  "invalid FP4 BF16 matrix decode");
  const auto values =
      static_cast<std::size_t>(m.rows) * m.padded_columns;
  if (values > std::numeric_limits<std::size_t>::max() /
                   sizeof(__nv_bfloat16) ||
      decoded_weight_bytes < values * sizeof(__nv_bfloat16))
    return Status(ErrorCode::invalid_argument,
                  "FP4 BF16 matrix decode workspace is too small");
  const auto blocks = static_cast<unsigned>(
      (values + kThreads - 1U) / kThreads);
  fp4_decode_matrix_bf16_kernel<<<blocks, kThreads, 0,
                                  static_cast<cudaStream_t>(raw)>>>(
      m.weights, m.scales, static_cast<__nv_bfloat16*>(decoded_weights),
      m.rows, m.padded_columns);
  return checked(cudaPeekAtLastError(), "decode FP4 matrix to BF16");
}

Status bf16_gemm_q8_block32(
    const Fp4Block32Matrix& m, const void* decoded_weights,
    std::size_t decoded_weight_bytes, const std::int8_t* input,
    const float* input_scales, void* decoded_input,
    std::size_t decoded_input_bytes, float* output, std::uint32_t batch,
    void* raw) noexcept {
  if (!m.weights || !m.scales || !decoded_weights || !input ||
      !input_scales || !decoded_input || !output || !m.rows || !m.columns ||
      m.padded_columns < m.columns ||
      m.padded_columns % kFp4GemmBlockColumns || !batch ||
      m.rows > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      m.padded_columns >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      batch > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    return Status(ErrorCode::invalid_argument,
                  "invalid staged BF16 Q8 GEMM");
  const auto weight_values =
      static_cast<std::size_t>(m.rows) * m.padded_columns;
  const auto input_values =
      static_cast<std::size_t>(batch) * m.padded_columns;
  const auto output_values = static_cast<std::size_t>(batch) * m.rows;
  if (weight_values > std::numeric_limits<std::size_t>::max() /
                          sizeof(__nv_bfloat16) ||
      input_values > std::numeric_limits<std::size_t>::max() /
                         sizeof(__nv_bfloat16) ||
      decoded_weight_bytes < weight_values * sizeof(__nv_bfloat16) ||
      decoded_input_bytes < input_values * sizeof(__nv_bfloat16))
    return Status(ErrorCode::invalid_argument,
                  "staged BF16 Q8 GEMM workspace is too small");
  const auto stream = static_cast<cudaStream_t>(raw);
  const auto input_blocks = static_cast<unsigned>(std::min<std::size_t>(
      (input_values + kThreads - 1U) / kThreads, 65535U));
  q8_to_bf16_kernel<<<input_blocks, kThreads, 0, stream>>>(
      input, static_cast<__nv_bfloat16*>(decoded_input), input_values);
  auto status = checked(cudaPeekAtLastError(), "decode Q8 input to BF16");
  if (!status.ok()) return status;

  cublasHandle_t blas{};
  auto blas_status = current_cublas_handle(stream, blas);
  if (blas_status != CUBLAS_STATUS_SUCCESS)
    return checked(blas_status, "initialize staged BF16 Q8 cuBLAS");
  const float alpha = 1.0F;
  const float zero = 0.0F;
  blas_status = cublasGemmEx(
      blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(m.rows),
      static_cast<int>(batch), static_cast<int>(m.padded_columns), &alpha,
      decoded_weights, CUDA_R_16BF, static_cast<int>(m.padded_columns),
      decoded_input, CUDA_R_16BF, static_cast<int>(m.padded_columns), &zero,
      output, CUDA_R_32F, static_cast<int>(m.rows), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (blas_status != CUBLAS_STATUS_SUCCESS)
    return checked(blas_status, "staged BF16 Q8 GEMM");
  const auto output_blocks = static_cast<unsigned>(std::min<std::size_t>(
      (output_values + kThreads - 1U) / kThreads, 65535U));
  scale_bf16_q8_gemm_kernel<<<output_blocks, kThreads, 0, stream>>>(
      output, input_scales, m.rows, output_values);
  return checked(cudaPeekAtLastError(), "scale staged BF16 Q8 GEMM");
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
      m.columns % 4U != 0U || !batch)
    return Status(ErrorCode::invalid_argument,
                  "invalid grouped-input weight-reuse batched gemv");
  const auto blocks = (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  for (std::uint32_t first = 0U; first < batch;
       first += kMaximumWeightReuseBatch) {
    const auto tile =
        std::min(kMaximumWeightReuseBatch, batch - first);
    int8_gemv_grouped_inputs_batch_reuse_kernel<<<
        blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales,
        input + static_cast<std::size_t>(first) * groups * m.columns,
        output + static_cast<std::size_t>(first) * m.rows, m.rows, m.columns,
        m.rows / groups, groups, tile);
  }
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
      !m.columns || !batch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched gemv");
  const auto blocks =
      (m.rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  for (std::uint32_t first = 0U; first < batch;
       first += kMaximumWeightReuseBatch) {
    const auto tile =
        std::min(kMaximumWeightReuseBatch, batch - first);
    int8_gemv_batch_weight_reuse_kernel<<<
        blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
        m.weights, m.scales,
        input + static_cast<std::size_t>(first) * m.columns,
        output + static_cast<std::size_t>(first) * m.rows, m.rows, m.columns,
        tile);
  }
  return checked(cudaPeekAtLastError(), "weight-reuse int8 batched gemv");
}
Status int8_gemm_f32_batch(
    const Int8Matrix& m, const float* input, float* output,
    std::uint32_t batch, float* decoded_matrix,
    std::uint64_t decoded_matrix_values, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !batch || !decoded_matrix)
    return {ErrorCode::invalid_argument, "invalid INT8 FP32 prefill GEMM"};
  const auto values = static_cast<std::uint64_t>(m.rows) * m.columns;
  if (values > decoded_matrix_values)
    return {ErrorCode::invalid_argument,
            "INT8 FP32 prefill GEMM workspace is too small"};
  const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(
      65535U, (values + kThreads - 1U) / kThreads));
  const auto stream = static_cast<cudaStream_t>(raw);
  decode_int8_rows_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      m.weights, m.scales, decoded_matrix, values, m.columns);
  auto status = checked(cudaPeekAtLastError(),
                        "decode INT8 matrix for prefill GEMM");
  if (!status.ok()) return status;
  return f32_prefill_gemm(decoded_matrix, m.rows, m.columns, input,
                          m.columns, output, m.rows, batch, stream);
}
Status int8_grouped_gemm_f32_batch(
    const Int8Matrix& m, const float* input, float* output,
    std::uint32_t groups, std::uint32_t batch, float* decoded_matrix,
    std::uint64_t decoded_matrix_values, void* raw) noexcept {
  if (!m.weights || !m.scales || !input || !output || !m.rows ||
      !m.columns || !groups || m.rows % groups != 0U || !batch ||
      !decoded_matrix)
    return {ErrorCode::invalid_argument,
            "invalid grouped INT8 FP32 prefill GEMM"};
  const auto values = static_cast<std::uint64_t>(m.rows) * m.columns;
  if (values > decoded_matrix_values)
    return {ErrorCode::invalid_argument,
            "grouped INT8 FP32 prefill GEMM workspace is too small"};
  const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(
      65535U, (values + kThreads - 1U) / kThreads));
  const auto stream = static_cast<cudaStream_t>(raw);
  decode_int8_rows_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      m.weights, m.scales, decoded_matrix, values, m.columns);
  auto status = checked(cudaPeekAtLastError(),
                        "decode grouped INT8 matrix for prefill GEMM");
  if (!status.ok()) return status;
  const auto group_rows = m.rows / groups;
  for (std::uint32_t group = 0U; group < groups; ++group) {
    status = f32_prefill_gemm(
        decoded_matrix + static_cast<std::uint64_t>(group) * group_rows *
                             m.columns,
        group_rows, m.columns,
        input + static_cast<std::uint64_t>(group) * m.columns,
        groups * m.columns,
        output + static_cast<std::uint64_t>(group) * group_rows, m.rows,
        batch, stream);
    if (!status.ok()) return status;
  }
  return Status::success();
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
  if (!matrix || !input || !output || !rows || !columns || !batch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched f32 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  for (std::uint32_t first = 0U; first < batch;
       first += kMaximumWeightReuseBatch) {
    const auto tile =
        std::min(kMaximumWeightReuseBatch, batch - first);
    f32_gemv_batch_reuse_kernel<<<
        blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
        matrix, input + static_cast<std::size_t>(first) * columns,
        output + static_cast<std::size_t>(first) * rows, rows, columns, tile);
  }
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
  if (!matrix || !input || !output || !rows || !columns || !batch)
    return Status(ErrorCode::invalid_argument,
                  "invalid weight-reuse batched bf16 gemv");
  const auto blocks = (rows + kWarpsPerBlock - 1U) / kWarpsPerBlock;
  for (std::uint32_t first = 0U; first < batch;
       first += kMaximumWeightReuseBatch) {
    const auto tile =
        std::min(kMaximumWeightReuseBatch, batch - first);
    bf16_gemv_batch_reuse_kernel<<<
        blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
        matrix, input + static_cast<std::size_t>(first) * columns,
        output + static_cast<std::size_t>(first) * rows, rows, columns, tile);
  }
  return checked(cudaPeekAtLastError(), "weight-reuse bf16 batched gemv");
}
Status bf16_gemm_f32_batch(
    const std::uint16_t* matrix, std::uint32_t rows,
    std::uint32_t columns, const float* input, float* output,
    std::uint32_t batch, float* decoded_matrix,
    std::uint64_t decoded_matrix_values, void* raw) noexcept {
  if (!matrix || !input || !output || !rows || !columns || !batch ||
      !decoded_matrix)
    return {ErrorCode::invalid_argument, "invalid BF16 FP32 prefill GEMM"};
  const auto values = static_cast<std::uint64_t>(rows) * columns;
  if (values > decoded_matrix_values)
    return {ErrorCode::invalid_argument,
            "BF16 FP32 prefill GEMM workspace is too small"};
  const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(
      65535U, (values + kThreads - 1U) / kThreads));
  const auto stream = static_cast<cudaStream_t>(raw);
  decode_bf16_rows_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      matrix, decoded_matrix, values);
  auto status = checked(cudaPeekAtLastError(),
                        "decode BF16 matrix for prefill GEMM");
  if (!status.ok()) return status;
  return f32_prefill_gemm(decoded_matrix, rows, columns, input, columns,
                          output, rows, batch, stream);
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
Status rms_norm_batch(const float* input, const float* weight, float* output,
                      std::uint32_t rows, std::uint32_t elements,
                      float epsilon, void* raw) noexcept {
  if (!input || !weight || !output || !rows || !elements || epsilon <= 0.0F)
    return Status(ErrorCode::invalid_argument, "invalid RMS norm batch");
  rms_batch_kernel<<<rows, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      input, weight, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "RMS norm batch");
}
Status weightless_rms_norm_batch(
    const float* input, float* output, std::uint32_t rows,
    std::uint32_t elements, float epsilon, void* raw) noexcept {
  if (!input || !output || !rows || !elements || epsilon <= 0.0F)
    return Status(ErrorCode::invalid_argument,
                  "invalid weightless RMS norm batch");
  weightless_rms_batch_kernel<<<rows, kThreads, 0,
                                static_cast<cudaStream_t>(raw)>>>(
      input, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "weightless RMS norm batch");
}
Status add_in_place(float* destination, const float* source, std::uint32_t elements, void* raw) noexcept {
  if (!destination || !source || !elements) return Status(ErrorCode::invalid_argument, "invalid add");
  add_kernel<<<(elements + kThreads - 1) / kThreads, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(destination, source, elements);
  return checked(cudaPeekAtLastError(), "add");
}
Status add_bias_in_place(float* destination, const float* bias,
                         std::uint32_t rows, std::uint32_t columns,
                         void* raw) noexcept {
  if (!destination || !bias || !rows || !columns)
    return Status(ErrorCode::invalid_argument, "invalid bias add");
  const auto values = static_cast<std::size_t>(rows) * columns;
  const auto blocks = static_cast<unsigned>(
      (values + kThreads - 1U) / kThreads);
  add_bias_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      destination, bias, columns, values);
  return checked(cudaPeekAtLastError(), "bias add");
}
Status layer_norm_batch(const float* input, const float* weight,
                        const float* bias, float* output,
                        std::uint32_t rows, std::uint32_t elements,
                        float epsilon, void* raw) noexcept {
  if (!input || !weight || !bias || !output || !rows || !elements ||
      !(epsilon > 0.0F))
    return Status(ErrorCode::invalid_argument, "invalid layer norm batch");
  layer_norm_batch_kernel<<<rows, kThreads, 0,
                            static_cast<cudaStream_t>(raw)>>>(
      input, weight, bias, output, elements, epsilon);
  return checked(cudaPeekAtLastError(), "layer norm batch");
}
Status gelu_tanh_in_place(float* values, std::uint32_t elements,
                          void* raw) noexcept {
  if (!values || !elements)
    return Status(ErrorCode::invalid_argument, "invalid tanh GELU");
  gelu_in_place_kernel<false><<<(elements + kThreads - 1U) / kThreads,
                                 kThreads, 0,
                                 static_cast<cudaStream_t>(raw)>>>(
      values, elements);
  return checked(cudaPeekAtLastError(), "tanh GELU");
}
Status gelu_exact_in_place(float* values, std::uint32_t elements,
                           void* raw) noexcept {
  if (!values || !elements)
    return Status(ErrorCode::invalid_argument, "invalid exact GELU");
  gelu_in_place_kernel<true><<<(elements + kThreads - 1U) / kThreads,
                                kThreads, 0,
                                static_cast<cudaStream_t>(raw)>>>(
      values, elements);
  return checked(cudaPeekAtLastError(), "exact GELU");
}
Status add_fp4_position_interpolation(
    float* hidden, const Fp4Block32Matrix& positions,
    const std::uint32_t* interpolation_indices,
    const float* interpolation_weights, std::uint32_t rows,
    void* raw) noexcept {
  if (!hidden || !positions.weights || !positions.scales ||
      !interpolation_indices || !interpolation_weights || !rows ||
      !positions.rows || !positions.columns ||
      positions.padded_columns < positions.columns ||
      positions.padded_columns % 32U)
    return Status(ErrorCode::invalid_argument,
                  "invalid FP4 position interpolation");
  const auto values = static_cast<std::size_t>(rows) * positions.columns;
  const auto blocks = static_cast<unsigned>(
      (values + kThreads - 1U) / kThreads);
  add_fp4_position_interpolation_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      hidden, positions.weights, positions.scales, interpolation_indices,
      interpolation_weights, rows, positions.columns,
      positions.padded_columns);
  return checked(cudaPeekAtLastError(), "FP4 position interpolation");
}
Status vision_qkv_rope_in_place(
    float* qkv, const std::uint32_t* row_positions, std::uint32_t rows,
    std::uint32_t heads, std::uint32_t head_dim, float rope_theta,
    void* raw) noexcept {
  if (!qkv || !row_positions || !rows || !heads || !head_dim ||
      head_dim > 1024U || head_dim % 4U || !(rope_theta > 0.0F))
    return Status(ErrorCode::invalid_argument, "invalid vision QKV rope");
  vision_qkv_rope_kernel<<<dim3(heads, rows), head_dim, 0,
                           static_cast<cudaStream_t>(raw)>>>(
      qkv, row_positions, heads, head_dim, rope_theta);
  return checked(cudaPeekAtLastError(), "vision QKV rope");
}
Status vision_segment_attention(
    const float* qkv, const std::uint32_t* segment_first,
    const std::uint32_t* segment_last, float* output, std::uint32_t rows,
    std::uint32_t heads, std::uint32_t head_dim, void* raw) noexcept {
  if (!qkv || !segment_first || !segment_last || !output || !rows ||
      !heads || !head_dim || head_dim > kThreads)
    return Status(ErrorCode::invalid_argument,
                  "invalid segmented vision attention");
  vision_segment_attention_kernel<<<
      dim3(heads, rows), kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      qkv, segment_first, segment_last, output, heads, head_dim);
  return checked(cudaPeekAtLastError(), "segmented vision attention");
}
Status gated_gqa_qk_norm_mrope_batch(
    float* q_and_gate, float* key, const float* q_norm_weight,
    const float* k_norm_weight, const std::uint32_t* positions_thw,
    std::uint32_t rows, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, std::uint32_t section_zero,
    std::uint32_t section_one, std::uint32_t section_two,
    float epsilon, float rope_theta, void* raw) noexcept {
  if (!q_and_gate || !key || !q_norm_weight || !k_norm_weight ||
      !positions_thw || !rows || !query_heads || !kv_heads ||
      query_heads % kv_heads || !head_dim || head_dim > kThreads ||
      !rotary_dim || rotary_dim > head_dim || rotary_dim % 2U ||
      !section_zero || !section_one || !section_two ||
      section_zero + section_one + section_two != rotary_dim / 2U ||
      !(epsilon > 0.0F) || !(rope_theta > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid three-axis GQA QKV mRoPE batch");
  gated_gqa_qk_norm_mrope_kernel<<<
      dim3(query_heads, rows), kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, q_norm_weight, k_norm_weight, positions_thw,
      query_heads, kv_heads, head_dim, rotary_dim, section_one, section_two,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(), "three-axis GQA QKV mRoPE batch");
}
template <PagedKvEncoding Encoding>
Status store_gqa_kv_paged_batch_impl(
    const float* key, const float* value, const void* const* page_table,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t first_cache_position, std::uint32_t rows,
    std::uint32_t kv_heads, std::uint32_t head_dim, void* raw,
    const char* name) noexcept {
  if (!key || !value || !page_table || !page_tokens || !rows || !kv_heads ||
      !head_dim || head_dim > kThreads || head_dim % 32U ||
      first_cache_position > 0xffffffffU - (rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid paged GQA KV store batch");
  store_gqa_kv_paged_batch_kernel<Encoding><<<
      dim3(kv_heads, rows), kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(
      key, value, page_table, full_attention_layer, page_tokens,
      first_cache_position, kv_heads, head_dim);
  return checked(cudaPeekAtLastError(), name);
}
Status store_gqa_kv_paged_fp4_batch(
    const float* key, const float* value, const void* const* page_table,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t first_cache_position, std::uint32_t rows,
    std::uint32_t kv_heads, std::uint32_t head_dim, void* raw) noexcept {
  return store_gqa_kv_paged_batch_impl<PagedKvEncoding::fp4>(
      key, value, page_table, full_attention_layer, page_tokens,
      first_cache_position, rows, kv_heads, head_dim, raw,
      "paged FP4 GQA KV store batch");
}
Status store_gqa_kv_paged_fp8_batch(
    const float* key, const float* value, const void* const* page_table,
    std::uint32_t full_attention_layer, std::uint32_t page_tokens,
    std::uint32_t first_cache_position, std::uint32_t rows,
    std::uint32_t kv_heads, std::uint32_t head_dim, void* raw) noexcept {
  return store_gqa_kv_paged_batch_impl<PagedKvEncoding::fp8>(
      key, value, page_table, full_attention_layer, page_tokens,
      first_cache_position, rows, kv_heads, head_dim, raw,
      "paged FP8 GQA KV store batch");
}
Status store_gqa_kv_fp16_batch(
    const float* key, const float* value, void* fp16_keys,
    void* fp16_values, std::uint32_t rows, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* raw) noexcept {
  if (!key || !value || !fp16_keys || !fp16_values || !rows || !kv_heads ||
      !head_dim)
    return Status(ErrorCode::invalid_argument,
                  "invalid FP16 GQA KV store batch");
  const auto values = static_cast<std::size_t>(rows) * kv_heads * head_dim;
  const auto blocks = static_cast<unsigned>(std::min<std::size_t>(
      (values + kThreads - 1U) / kThreads, 65535U));
  store_gqa_kv_fp16_batch_kernel<<<
      blocks, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      key, value, static_cast<__half*>(fp16_keys),
      static_cast<__half*>(fp16_values), values);
  return checked(cudaPeekAtLastError(), "FP16 GQA KV store batch");
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
Status relu2_in_place(float* values, std::uint32_t elements,
                      void* raw) noexcept {
  if (!values || !elements)
    return Status(ErrorCode::invalid_argument, "invalid ReLU squared input");
  relu2_in_place_kernel<<<(elements + kThreads - 1U) / kThreads, kThreads, 0,
                           static_cast<cudaStream_t>(raw)>>>(values, elements);
  return checked(cudaPeekAtLastError(), "ReLU squared");
}
Status deepseek_swiglu_product(
    const float* gate, const float* up, float* output,
    std::uint32_t elements, float limit, bool bf16_output,
    void* raw) noexcept {
  if (!gate || !up || !output || !elements || limit < 0.0F)
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek SwiGLU product");
  deepseek_swiglu_product_kernel<<<
      (elements + kThreads - 1U) / kThreads, kThreads, 0,
      static_cast<cudaStream_t>(raw)>>>(gate, up, output, elements, limit,
                                       bf16_output);
  return checked(cudaPeekAtLastError(), "DeepSeek SwiGLU product");
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
Status sigmoid_product_in_place(float* values, const float* gate,
                                std::uint32_t elements,
                                void* raw) noexcept {
  if (!values || !gate || !elements)
    return Status(ErrorCode::invalid_argument, "invalid sigmoid product");
  sigmoid_product_kernel<<<(elements + kThreads - 1U) / kThreads, kThreads,
                            0, static_cast<cudaStream_t>(raw)>>>(
      values, gate, elements);
  return checked(cudaPeekAtLastError(), "sigmoid product");
}
Status scaled_tanh_in_place(float* values, std::uint32_t elements,
                            float multiplier, float softcap,
                            void* raw) noexcept {
  if (!values || !elements || !(multiplier > 0.0F) || !(softcap > 0.0F))
    return Status(ErrorCode::invalid_argument, "invalid scaled tanh");
  scaled_tanh_kernel<<<(elements + kThreads - 1U) / kThreads, kThreads, 0,
                       static_cast<cudaStream_t>(raw)>>>(
      values, elements, multiplier, softcap);
  return checked(cudaPeekAtLastError(), "scaled tanh");
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
Status deepseek_router_hash_rows(
    const float* input, const std::uint16_t* weights,
    const std::int64_t* token_experts, const std::uint32_t* token_ids,
    std::uint32_t rows, float* logits, float* scores,
    std::uint32_t* indices, float route_scale, void* raw) noexcept {
  if (!input || !weights || !token_experts || !token_ids || !rows ||
      !logits || !scores || !indices || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek hash router rows");
  for (std::uint32_t row = 0U; row < rows; ++row) {
    if (token_ids[row] >= 129280U)
      return Status(ErrorCode::invalid_argument,
                    "DeepSeek hash router token exceeds vocabulary");
  }
  auto status = gemv_bf16_batch(weights, 256U, 4096U, input, logits, rows,
                                raw);
  if (!status.ok()) return status;
  const auto stream = static_cast<cudaStream_t>(raw);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    deepseek_hash_router_kernel<<<1U, 32U, 0, stream>>>(
        logits + static_cast<std::size_t>(row) * 256U, token_experts,
        token_ids[row], route_scale,
        scores + static_cast<std::size_t>(row) * 6U,
        indices + static_cast<std::size_t>(row) * 6U);
  }
  return checked(cudaPeekAtLastError(), "DeepSeek hash router rows select");
}
Status deepseek_router_learned_rows(
    const float* input, const std::uint16_t* weights,
    const float* selection_bias, std::uint32_t rows, float* logits,
    float* scores, std::uint32_t* indices, float route_scale,
    void* raw) noexcept {
  if (!input || !weights || !selection_bias || !rows || !logits || !scores ||
      !indices || !(route_scale > 0.0F))
    return Status(ErrorCode::invalid_argument,
                  "invalid DeepSeek learned router rows");
  auto status = gemv_bf16_batch(weights, 256U, 4096U, input, logits, rows,
                                raw);
  if (!status.ok()) return status;
  const auto stream = static_cast<cudaStream_t>(raw);
  for (std::uint32_t row = 0U; row < rows; ++row) {
    deepseek_learned_router_kernel<<<1U, kThreads, 0, stream>>>(
        logits + static_cast<std::size_t>(row) * 256U, selection_bias,
        route_scale, scores + static_cast<std::size_t>(row) * 6U,
        indices + static_cast<std::size_t>(row) * 6U);
  }
  return checked(cudaPeekAtLastError(),
                 "DeepSeek learned router rows select");
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
  gated_gqa_qkv_rope_paged_kernel<PagedKvEncoding::fp4><<<
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
  gated_gqa_qkv_rope_paged_batch_kernel<PagedKvEncoding::fp4><<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight, page_table,
      full_attention_layer, page_tokens, first_cache_position,
      first_rotary_position, query_heads, kv_heads, head_dim, rotary_dim,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(),
                 "paged FP4 gated GQA QKV rope batch");
}
Status gated_gqa_qkv_rope_cache_paged_fp8_at(
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
                  "invalid paged FP8 gated GQA QKV rope");
  gated_gqa_qkv_rope_paged_kernel<PagedKvEncoding::fp8><<<
      query_heads, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight,
      static_cast<std::uint8_t*>(page), full_attention_layer, page_tokens,
      cache_position, rotary_position, query_heads, kv_heads, head_dim,
      rotary_dim, epsilon, rope_theta);
  return checked(cudaPeekAtLastError(),
                 "paged FP8 gated GQA QKV rope");
}
Status gated_gqa_qkv_rope_cache_paged_fp8_batch(
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
                  "invalid paged FP8 gated GQA QKV rope batch");
  const dim3 grid(query_heads, rows);
  gated_gqa_qkv_rope_paged_batch_kernel<PagedKvEncoding::fp8><<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight, page_table,
      full_attention_layer, page_tokens, first_cache_position,
      first_rotary_position, query_heads, kv_heads, head_dim, rotary_dim,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(),
                 "paged FP8 gated GQA QKV rope batch");
}
Status gated_gqa_qkv_rope_fp16_batch(
    float* q_and_gate, float* key, const float* value,
    const float* q_norm_weight, const float* k_norm_weight,
    void* fp16_keys, void* fp16_values, std::uint32_t first_rotary_position,
    std::uint32_t rows, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float epsilon, float rope_theta,
    void* raw) noexcept {
  if (!q_and_gate || !key || !value || !q_norm_weight || !k_norm_weight ||
      !fp16_keys || !fp16_values || !rows || !query_heads || !kv_heads ||
      query_heads % kv_heads || query_heads / kv_heads > 8U || !head_dim ||
      head_dim > kThreads || head_dim % 32U || !rotary_dim ||
      rotary_dim > head_dim || rotary_dim % 2U || !(epsilon > 0.0F) ||
      !(rope_theta > 0.0F) ||
      first_rotary_position > 0xffffffffU - (rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid FP16 gated GQA QKV batch");
  const dim3 grid(query_heads, rows);
  gated_gqa_qkv_rope_fp16_batch_kernel<<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      q_and_gate, key, value, q_norm_weight, k_norm_weight,
      static_cast<__half*>(fp16_keys), static_cast<__half*>(fp16_values),
      first_rotary_position, query_heads, kv_heads, head_dim, rotary_dim,
      epsilon, rope_theta);
  return checked(cudaPeekAtLastError(), "FP16 gated GQA QKV batch");
}
Status standard_gqa_qkv_rope_fp16_batch(
    float* query, float* key, const float* value,
    void* fp16_keys, void* fp16_values, std::uint32_t first_rotary_position,
    std::uint32_t rows, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, void* raw) noexcept {
  if (!query || !key || !value || !fp16_keys || !fp16_values || !rows ||
      !query_heads || !kv_heads || query_heads % kv_heads ||
      query_heads / kv_heads > kMaximumExactFp16GroupedQueryHeads ||
      !head_dim || head_dim > kThreads ||
      head_dim % 32U || !rotary_dim || rotary_dim > head_dim ||
      rotary_dim % 2U || !(rope_theta > 0.0F) ||
      first_rotary_position > 0xffffffffU - (rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid FP16 standard GQA QKV batch");
  const dim3 grid(query_heads, rows);
  standard_gqa_qkv_rope_fp16_batch_kernel<<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      query, key, value, static_cast<__half*>(fp16_keys),
      static_cast<__half*>(fp16_values), first_rotary_position,
      query_heads, kv_heads, head_dim, rotary_dim, rope_theta);
  return checked(cudaPeekAtLastError(), "FP16 standard GQA QKV batch");
}
Status normalized_gqa_qkv_fp16_batch(
    float* query, float* key, const float* value,
    void* fp16_keys, void* fp16_values,
    std::uint32_t first_rotary_position, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, std::uint32_t rotary_dim, float epsilon,
    float query_scale, float rope_theta, bool apply_rope,
    void* raw) noexcept {
  if (!query || !key || !value || !fp16_keys || !fp16_values || !rows ||
      !query_heads || !kv_heads || query_heads % kv_heads ||
      query_heads / kv_heads > kMaximumExactFp16GroupedQueryHeads ||
      !head_dim || head_dim > kThreads || head_dim % 32U ||
      !rotary_dim || rotary_dim > head_dim || rotary_dim % 2U ||
      !(epsilon > 0.0F) || !(query_scale > 0.0F) ||
      (apply_rope && !(rope_theta > 0.0F)) ||
      first_rotary_position > 0xffffffffU - (rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid normalized FP16 GQA QKV batch");
  const dim3 grid(query_heads, rows);
  normalized_gqa_qkv_fp16_batch_kernel<<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      query, key, value, static_cast<__half*>(fp16_keys),
      static_cast<__half*>(fp16_values), first_rotary_position,
      query_heads, kv_heads, head_dim, rotary_dim, epsilon, query_scale,
      rope_theta, apply_rope);
  return checked(cudaPeekAtLastError(), "normalized FP16 GQA QKV batch");
}
Status standard_gqa_kv_fp16_batch(
    const float* key, const float* value,
    void* fp16_keys, void* fp16_values, std::uint32_t rows,
    std::uint32_t query_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim, void* raw) noexcept {
  if (!key || !value || !fp16_keys || !fp16_values || !rows ||
      !query_heads || !kv_heads || query_heads % kv_heads ||
      query_heads / kv_heads > kMaximumExactFp16GroupedQueryHeads ||
      !head_dim || head_dim > kThreads || head_dim % 32U)
    return Status(ErrorCode::invalid_argument,
                  "invalid FP16 position-free standard GQA QKV batch");
  const dim3 grid(kv_heads, rows);
  standard_gqa_kv_fp16_batch_kernel<<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      key, value, static_cast<__half*>(fp16_keys),
      static_cast<__half*>(fp16_values), kv_heads, head_dim);
  return checked(cudaPeekAtLastError(),
                 "FP16 position-free standard GQA QKV batch");
}
Status pack_gqa_kv_fp16_to_paged_fp4(
    const void* fp16_keys, const void* fp16_values,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t rows, std::uint32_t kv_heads, std::uint32_t head_dim,
    void* raw) noexcept {
  if (!fp16_keys || !fp16_values || !page_table || !page_tokens || !rows ||
      !kv_heads || !head_dim || head_dim % 32U || head_dim > 1024U ||
      first_cache_position > 0xffffffffU - (rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid FP16-to-paged-FP4 GQA cache pack");
  const dim3 grid(kv_heads, rows);
  pack_gqa_kv_fp16_to_paged_fp4_kernel<<<
      grid, head_dim / 32U, 0, static_cast<cudaStream_t>(raw)>>>(
      static_cast<const __half*>(fp16_keys),
      static_cast<const __half*>(fp16_values), page_table,
      full_attention_layer, page_tokens, first_cache_position, kv_heads,
      head_dim);
  return checked(cudaPeekAtLastError(),
                 "FP16-to-paged-FP4 GQA cache pack");
}
Status store_gqa_kv_fp16_to_paged(
    const void* fp16_keys, const void* fp16_values,
    const void* const* page_table, std::uint32_t full_attention_layer,
    std::uint32_t page_tokens, std::uint32_t first_cache_position,
    std::uint32_t rows, std::uint32_t kv_heads, std::uint32_t head_dim,
    void* raw) noexcept {
  if (!fp16_keys || !fp16_values || !page_table || !page_tokens || !rows ||
      !kv_heads || !head_dim || head_dim > kThreads ||
      first_cache_position > 0xffffffffU - (rows - 1U))
    return Status(ErrorCode::invalid_argument,
                  "invalid FP16 paged GQA cache store");
  const dim3 grid(kv_heads, rows);
  store_gqa_kv_fp16_to_paged_kernel<<<
      grid, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      static_cast<const __half*>(fp16_keys),
      static_cast<const __half*>(fp16_values), page_table,
      full_attention_layer, page_tokens, first_cache_position, kv_heads,
      head_dim);
  return checked(cudaPeekAtLastError(), "FP16 paged GQA cache store");
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
template <PagedKvEncoding Encoding>
Status gated_gqa_attention_decode_paged_tensor_core_impl(
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
    gated_gqa_attention_paged_tensor_core_split_kernel<
        kFp4GemmThreads, 128U, Encoding><<<
        producer_grid, kFp4GemmThreads, 0, stream>>>(
        launch.q_and_gate, launch.page_table, launch.partial_maxima,
        launch.partial_sums, launch.partial_outputs, launch.context_tokens,
        1U, launch.maximum_splits, launch.full_attention_layer,
        launch.page_tokens, launch.query_heads, launch.kv_heads,
        launch.head_dim, launch.split_tokens);
  } else {
    gated_gqa_attention_paged_tensor_core_split_kernel<
        kFp4GemmThreads, 256U, Encoding><<<
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
Status gated_gqa_attention_decode_paged_fp4_tensor_core(
    const PagedFp4GatedGqaAttentionLaunch& launch) noexcept {
  return gated_gqa_attention_decode_paged_tensor_core_impl<
      PagedKvEncoding::fp4>(launch);
}
Status gated_gqa_attention_decode_paged_fp8_tensor_core(
    const PagedFp8GatedGqaAttentionLaunch& launch) noexcept {
  return gated_gqa_attention_decode_paged_tensor_core_impl<
      PagedKvEncoding::fp8>(launch);
}

template <PagedKvEncoding Encoding>
Status gated_gqa_attention_microbatch_paged_tensor_core_impl(
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
    gated_gqa_attention_paged_tensor_core_split_kernel<
        kTensorCoreAttentionThreads, 128U, Encoding><<<
        producer_grid, kTensorCoreAttentionThreads, 0, stream>>>(
        launch.q_and_gate, launch.page_table, launch.partial_maxima,
        launch.partial_sums, launch.partial_outputs,
        launch.first_context_tokens, launch.rows, launch.maximum_splits,
        launch.full_attention_layer, launch.page_tokens, launch.query_heads,
        launch.kv_heads, launch.head_dim, launch.split_tokens);
  } else {
    const auto& topology = current_device_topology();
    if (topology.status != cudaSuccess)
      return checked(topology.status,
                     "query FP4 Tensor Core attention device topology");
    const auto grid_blocks = static_cast<std::uint64_t>(producer_grid.x) *
                             static_cast<std::uint64_t>(producer_grid.y);
    const auto use_high_occupancy = topology.multiprocessors > 0 &&
        grid_blocks >= 2U * static_cast<std::uint64_t>(
                                topology.multiprocessors);
    if (use_high_occupancy) {
      static const auto cache_status = checked(
          cudaFuncSetCacheConfig(
              gated_gqa_attention_paged_tensor_core_high_occupancy_split_kernel<
                  kTensorCoreAttentionThreads, 256U, Encoding>,
              cudaFuncCachePreferShared),
          "configure paged FP4 Tensor Core GQA microbatch cache");
      if (!cache_status.ok()) return cache_status;
      gated_gqa_attention_paged_tensor_core_high_occupancy_split_kernel<
          kTensorCoreAttentionThreads, 256U, Encoding><<<
          producer_grid, kTensorCoreAttentionThreads, 0, stream>>>(
          launch.q_and_gate, launch.page_table, launch.partial_maxima,
          launch.partial_sums, launch.partial_outputs,
          launch.first_context_tokens, launch.rows, launch.maximum_splits,
          launch.full_attention_layer, launch.page_tokens, launch.query_heads,
          launch.kv_heads, launch.head_dim, launch.split_tokens);
    } else {
      gated_gqa_attention_paged_tensor_core_split_kernel<
          kTensorCoreAttentionThreads, 256U, Encoding><<<
          producer_grid, kTensorCoreAttentionThreads, 0, stream>>>(
          launch.q_and_gate, launch.page_table, launch.partial_maxima,
          launch.partial_sums, launch.partial_outputs,
          launch.first_context_tokens, launch.rows, launch.maximum_splits,
          launch.full_attention_layer, launch.page_tokens, launch.query_heads,
          launch.kv_heads, launch.head_dim, launch.split_tokens);
    }
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
Status gated_gqa_attention_microbatch_paged_fp4_tensor_core(
    const PagedFp4GatedGqaPrefillLaunch& launch) noexcept {
  return gated_gqa_attention_microbatch_paged_tensor_core_impl<
      PagedKvEncoding::fp4>(launch);
}
Status gated_gqa_attention_microbatch_paged_fp8_tensor_core(
    const PagedFp8GatedGqaPrefillLaunch& launch) noexcept {
  return gated_gqa_attention_microbatch_paged_tensor_core_impl<
      PagedKvEncoding::fp8>(launch);
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
template <PagedKvEncoding Encoding>
Status gated_gqa_attention_staged_prefill_paged_impl(
    const PagedFp4GatedGqaPrefillLaunch& launch,
    const PagedFp4GatedGqaStagedPrefillWorkspace& workspace) noexcept {
  if (!launch.q_and_gate || !launch.page_table || !launch.output ||
      !launch.first_context_tokens || !launch.rows || !launch.page_tokens ||
      !launch.query_heads || !launch.kv_heads ||
      launch.query_heads % launch.kv_heads ||
      launch.query_heads / launch.kv_heads > 8U || !launch.head_dim ||
      launch.head_dim > kThreads || launch.head_dim % 32U ||
      launch.first_context_tokens > 0xffffffffU - (launch.rows - 1U) ||
      !workspace.queries || !workspace.keys || !workspace.values ||
      !workspace.scores || !workspace.probabilities ||
      !workspace.accumulator || !workspace.maxima || !workspace.sums ||
      !workspace.split_tokens)
    return Status(ErrorCode::invalid_argument,
                  "invalid staged paged FP4 gated GQA prefill");
  const auto grouped_heads = launch.query_heads / launch.kv_heads;
  const auto matrix_rows = launch.rows * grouped_heads;
  const auto split_tokens = std::min(
      workspace.split_tokens,
      launch.first_context_tokens + launch.rows - 1U);
  const auto query_values = static_cast<std::size_t>(launch.rows) *
                            launch.query_heads * launch.head_dim;
  const auto state_rows =
      static_cast<std::size_t>(launch.kv_heads) * matrix_rows;
  const auto kv_values = static_cast<std::size_t>(launch.kv_heads) *
                         split_tokens * launch.head_dim;
  const auto score_values = state_rows * split_tokens;
  const auto query_bytes = query_values * sizeof(__nv_bfloat16);
  const auto kv_bytes = kv_values * sizeof(__nv_bfloat16);
  const auto score_bytes = score_values * sizeof(float);
  const auto probability_bytes = score_values * sizeof(__nv_bfloat16);
  const auto accumulator_bytes = query_values * sizeof(float);
  const auto state_bytes = state_rows * sizeof(float);
  if (workspace.query_bytes < query_bytes || workspace.key_bytes < kv_bytes ||
      workspace.value_bytes < kv_bytes ||
      workspace.score_bytes < score_bytes ||
      workspace.probability_bytes < probability_bytes ||
      workspace.accumulator_bytes < accumulator_bytes ||
      workspace.maxima_bytes < state_bytes ||
      workspace.sum_bytes < state_bytes)
    return Status(ErrorCode::invalid_argument,
                  "staged paged FP4 prefill workspace is too small");

  const auto stream = static_cast<cudaStream_t>(launch.stream);
  const auto prepare_blocks = static_cast<unsigned>(
      (query_values + kThreads - 1U) / kThreads);
  staged_prefill_prepare_kernel<<<prepare_blocks, kThreads, 0, stream>>>(
      launch.q_and_gate,
      static_cast<__nv_bfloat16*>(workspace.queries),
      workspace.accumulator, workspace.maxima, workspace.sums, launch.rows,
      launch.query_heads, launch.kv_heads, launch.head_dim);
  auto status = checked(cudaPeekAtLastError(),
                        "prepare staged paged FP4 prefill");
  if (!status.ok()) return status;

  cublasHandle_t blas{};
  auto blas_status = current_cublas_handle(stream, blas);
  if (blas_status != CUBLAS_STATUS_SUCCESS)
    return checked(blas_status, "initialize staged prefill cuBLAS");
  const float alpha = 1.0F;
  const float zero = 0.0F;
  const float one = 1.0F;
  const auto last_context =
      launch.first_context_tokens + launch.rows - 1U;
  for (std::uint32_t first_token = 0U; first_token < last_context;
       first_token += split_tokens) {
    const auto tokens = std::min(split_tokens, last_context - first_token);
    const auto chunks_per_record =
        launch.head_dim /
        (Encoding == PagedKvEncoding::fp8 ? 8U : 16U);
    const auto decode_items = static_cast<std::size_t>(2U) * tokens *
                              launch.kv_heads * chunks_per_record;
    const auto decode_blocks = static_cast<unsigned>(
        (decode_items + kThreads - 1U) / kThreads);
    staged_prefill_decode_kv_kernel<Encoding><<<
        decode_blocks, kThreads, 0, stream>>>(
        launch.page_table, static_cast<__nv_bfloat16*>(workspace.keys),
        static_cast<__nv_bfloat16*>(workspace.values), first_token, tokens,
        launch.full_attention_layer, launch.page_tokens, launch.kv_heads,
        launch.head_dim);
    status = checked(cudaPeekAtLastError(),
                     "decode staged paged FP4 K/V");
    if (!status.ok()) return status;

    const auto query_stride = static_cast<long long>(matrix_rows) *
                              launch.head_dim;
    const auto kv_stride =
        static_cast<long long>(tokens) * launch.head_dim;
    const auto score_stride =
        static_cast<long long>(matrix_rows) * tokens;
    blas_status = cublasGemmStridedBatchedEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(tokens),
        static_cast<int>(matrix_rows), static_cast<int>(launch.head_dim),
        &alpha, workspace.keys, CUDA_R_16BF,
        static_cast<int>(launch.head_dim), kv_stride, workspace.queries,
        CUDA_R_16BF, static_cast<int>(launch.head_dim), query_stride, &zero,
        workspace.scores, CUDA_R_32F, static_cast<int>(tokens), score_stride,
        static_cast<int>(launch.kv_heads), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas_status != CUBLAS_STATUS_SUCCESS)
      return checked(blas_status, "staged paged FP4 QK GEMM");

    staged_prefill_softmax_kernel<<<
        static_cast<unsigned>(state_rows), kThreads, 0, stream>>>(
        workspace.scores,
        static_cast<__nv_bfloat16*>(workspace.probabilities),
        workspace.accumulator, workspace.maxima, workspace.sums,
        launch.first_context_tokens, first_token, tokens, launch.rows,
        grouped_heads, launch.kv_heads, launch.head_dim);
    status = checked(cudaPeekAtLastError(),
                     "staged paged FP4 online softmax");
    if (!status.ok()) return status;

    const auto output_stride =
        static_cast<long long>(matrix_rows) * launch.head_dim;
    blas_status = cublasGemmStridedBatchedEx(
        blas, CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(launch.head_dim), static_cast<int>(matrix_rows),
        static_cast<int>(tokens), &alpha, workspace.values, CUDA_R_16BF,
        static_cast<int>(launch.head_dim), kv_stride,
        workspace.probabilities, CUDA_R_16BF, static_cast<int>(tokens),
        score_stride, &one, workspace.accumulator, CUDA_R_32F,
        static_cast<int>(launch.head_dim), output_stride,
        static_cast<int>(launch.kv_heads), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas_status != CUBLAS_STATUS_SUCCESS)
      return checked(blas_status, "staged paged FP4 PV GEMM");
  }
  staged_prefill_finalize_kernel<<<prepare_blocks, kThreads, 0, stream>>>(
      launch.q_and_gate, workspace.accumulator, workspace.sums,
      launch.output, launch.rows, launch.query_heads, launch.kv_heads,
      launch.head_dim, true);
  return checked(cudaPeekAtLastError(),
                 "finalize staged paged FP4 prefill");
}
Status gated_gqa_attention_staged_prefill_paged_fp4(
    const PagedFp4GatedGqaPrefillLaunch& launch,
    const PagedFp4GatedGqaStagedPrefillWorkspace& workspace) noexcept {
  return gated_gqa_attention_staged_prefill_paged_impl<
      PagedKvEncoding::fp4>(launch, workspace);
}
Status gated_gqa_attention_staged_prefill_paged_fp8(
    const PagedFp8GatedGqaPrefillLaunch& launch,
    const PagedFp8GatedGqaStagedPrefillWorkspace& workspace) noexcept {
  return gated_gqa_attention_staged_prefill_paged_impl<
      PagedKvEncoding::fp8>(launch, workspace);
}
template <typename Launch>
Status gated_gqa_attention_staged_fp16_impl(
    const Launch& launch,
    const HostFp16GatedGqaAttentionWorkspace& workspace,
    bool source_is_host) noexcept {
  const auto paged_host_source = [&]() {
    if constexpr (std::is_same_v<Launch, HostFp16GatedGqaAttentionLaunch>)
      return launch.host_key_pages != nullptr ||
             launch.host_value_pages != nullptr;
    else
      return false;
  }();
  const auto paged_device_source = [&]() {
    if constexpr (std::is_same_v<Launch, DeviceFp16GatedGqaAttentionLaunch>)
      return launch.device_pages != nullptr;
    else
      return false;
  }();
  const auto* source_keys = [&]() -> const void* {
    if constexpr (std::is_same_v<Launch, HostFp16GatedGqaAttentionLaunch>)
      return launch.host_keys;
    else
      return launch.device_keys;
  }();
  const auto* source_values = [&]() -> const void* {
    if constexpr (std::is_same_v<Launch, HostFp16GatedGqaAttentionLaunch>)
      return launch.host_values;
    else
      return launch.device_values;
  }();
  const auto valid_source = [&]() {
    if constexpr (std::is_same_v<Launch, HostFp16GatedGqaAttentionLaunch>) {
      if (!paged_host_source) return source_keys && source_values;
      return launch.host_key_pages && launch.host_value_pages &&
             launch.host_page_tokens && launch.host_page_count &&
             launch.cache_capacity <=
                 static_cast<std::uint64_t>(launch.host_page_tokens) *
                     launch.host_page_count;
    } else {
      if (!paged_device_source) return source_keys && source_values;
      return launch.device_pages && launch.device_page_tokens &&
             launch.device_page_count &&
             launch.cache_capacity <=
                 static_cast<std::uint64_t>(launch.device_page_tokens) *
                     launch.device_page_count;
    }
  }();
  if (!launch.q_and_gate || !valid_source ||
      !launch.output || !launch.cache_capacity ||
      !launch.first_context_tokens || !launch.rows || !launch.query_heads ||
      !launch.kv_heads || launch.query_heads % launch.kv_heads ||
      launch.query_heads / launch.kv_heads >
          kMaximumExactFp16GroupedQueryHeads || !launch.head_dim ||
      launch.head_dim > kThreads || launch.head_dim % 32U ||
      launch.first_context_tokens > launch.cache_capacity ||
      launch.first_context_tokens > 0xffffffffU - (launch.rows - 1U) ||
      launch.first_context_tokens + launch.rows - 1U >
          launch.cache_capacity ||
      launch.retained_first_token >= launch.first_context_tokens ||
      !workspace.queries ||
      (source_is_host && (!workspace.raw_keys || !workspace.raw_values)) ||
      !workspace.keys || !workspace.values || !workspace.scores ||
      !workspace.probabilities || !workspace.accumulator ||
      !workspace.maxima || !workspace.sums || !workspace.split_tokens)
    return Status(ErrorCode::invalid_argument,
                  "invalid staged FP16 gated GQA attention");
  const auto grouped_heads = launch.query_heads / launch.kv_heads;
  const auto matrix_rows = launch.rows * grouped_heads;
  const auto last_context =
      launch.first_context_tokens + launch.rows - 1U;
  const auto retained_tokens = last_context - launch.retained_first_token;
  const auto split_tokens =
      std::min(workspace.split_tokens, retained_tokens);
  const auto query_values = static_cast<std::size_t>(launch.rows) *
                            launch.query_heads * launch.head_dim;
  const auto state_rows =
      static_cast<std::size_t>(launch.kv_heads) * matrix_rows;
  const auto maximum_kv_values =
      static_cast<std::size_t>(launch.kv_heads) * split_tokens *
      launch.head_dim;
  const auto maximum_score_values = state_rows * split_tokens;
  const auto query_bytes = query_values * sizeof(__half);
  const auto kv_bytes = maximum_kv_values * sizeof(__half);
  const auto score_bytes = maximum_score_values * sizeof(float);
  const auto probability_bytes = maximum_score_values * sizeof(__half);
  const auto accumulator_bytes = query_values * sizeof(float);
  const auto state_bytes = state_rows * sizeof(float);
  if (workspace.query_bytes < query_bytes ||
      (source_is_host && workspace.raw_key_bytes < kv_bytes) ||
      (source_is_host && workspace.raw_value_bytes < kv_bytes) ||
      workspace.key_bytes < kv_bytes ||
      workspace.value_bytes < kv_bytes ||
      workspace.score_bytes < score_bytes ||
      workspace.probability_bytes < probability_bytes ||
      workspace.accumulator_bytes < accumulator_bytes ||
      workspace.maxima_bytes < state_bytes ||
      workspace.sum_bytes < state_bytes)
    return Status(ErrorCode::invalid_argument,
                  "staged FP16 attention workspace is too small");

  const auto stream = static_cast<cudaStream_t>(launch.stream);
  const auto prepare_blocks = static_cast<unsigned>(
      (query_values + kThreads - 1U) / kThreads);
  staged_fp16_prepare_kernel<<<prepare_blocks, kThreads, 0, stream>>>(
      launch.q_and_gate, static_cast<__half*>(workspace.queries),
      workspace.accumulator, workspace.maxima, workspace.sums, launch.rows,
      launch.query_heads, launch.kv_heads, launch.head_dim,
      launch.output_gated);
  auto status = checked(cudaPeekAtLastError(),
                        "prepare staged FP16 attention");
  if (!status.ok()) return status;

  cublasHandle_t blas{};
  auto blas_status = current_cublas_handle(stream, blas);
  if (blas_status != CUBLAS_STATUS_SUCCESS)
    return checked(blas_status, "initialize staged FP16 attention");
  const float alpha = 1.0F;
  const float zero = 0.0F;
  const float one = 1.0F;
  const auto record_values = static_cast<std::size_t>(launch.kv_heads) *
                             launch.head_dim;
  const auto* fp16_keys = static_cast<const __half*>(source_keys);
  const auto* fp16_values = static_cast<const __half*>(source_values);
  for (std::uint32_t first_token = launch.retained_first_token;
       first_token < last_context;) {
    auto tokens = std::min(split_tokens, last_context - first_token);
    if constexpr (std::is_same_v<Launch,
                                 HostFp16GatedGqaAttentionLaunch>) {
      if (paged_host_source)
        tokens = std::min(tokens, launch.host_page_tokens -
                                      first_token % launch.host_page_tokens);
    }
    const auto raw_values =
        static_cast<std::size_t>(tokens) * record_values;
    const auto raw_bytes = raw_values * sizeof(__half);
    const __half* raw_keys{};
    const __half* raw_values_source{};
    if constexpr (std::is_same_v<Launch,
                                 HostFp16GatedGqaAttentionLaunch>) {
      if (paged_host_source) {
        const auto page = first_token / launch.host_page_tokens;
        if (page >= launch.host_page_count ||
            !launch.host_key_pages[page] ||
            !launch.host_value_pages[page])
          return Status(ErrorCode::invalid_argument,
                        "paged host FP16 source is incomplete");
        const auto offset = static_cast<std::size_t>(
                                first_token % launch.host_page_tokens) *
                            record_values;
        raw_keys = static_cast<const __half*>(launch.host_key_pages[page]) +
                   offset;
        raw_values_source =
            static_cast<const __half*>(launch.host_value_pages[page]) +
            offset;
      } else {
        raw_keys = fp16_keys +
                   static_cast<std::size_t>(first_token) * record_values;
        raw_values_source =
            fp16_values +
            static_cast<std::size_t>(first_token) * record_values;
      }
    } else {
      if (!paged_device_source) {
        raw_keys = fp16_keys +
                   static_cast<std::size_t>(first_token) * record_values;
        raw_values_source =
            fp16_values +
            static_cast<std::size_t>(first_token) * record_values;
      }
    }
    if (source_is_host) {
      auto copy = cudaMemcpyAsync(workspace.raw_keys, raw_keys, raw_bytes,
                                  cudaMemcpyHostToDevice, stream);
      if (copy != cudaSuccess)
        return checked(copy, "stage host FP16 keys");
      copy = cudaMemcpyAsync(workspace.raw_values, raw_values_source,
                             raw_bytes, cudaMemcpyHostToDevice, stream);
      if (copy != cudaSuccess)
        return checked(copy, "stage host FP16 values");
      raw_keys = static_cast<const __half*>(workspace.raw_keys);
      raw_values_source =
          static_cast<const __half*>(workspace.raw_values);
    }
    const auto transpose_blocks =
        static_cast<unsigned>((raw_values + kThreads - 1U) / kThreads);
    if constexpr (std::is_same_v<Launch,
                                 DeviceFp16GatedGqaAttentionLaunch>) {
      if (paged_device_source) {
        staged_fp16_transpose_paged_kv_kernel<<<transpose_blocks, kThreads,
                                                0, stream>>>(
            launch.device_pages, launch.device_page_layer,
            launch.device_page_tokens, first_token,
            static_cast<__half*>(workspace.keys),
            static_cast<__half*>(workspace.values), tokens, launch.kv_heads,
            launch.head_dim);
      } else {
        staged_fp16_transpose_kv_kernel<<<transpose_blocks, kThreads, 0,
                                          stream>>>(
            raw_keys, raw_values_source,
            static_cast<__half*>(workspace.keys),
            static_cast<__half*>(workspace.values), tokens, launch.kv_heads,
            launch.head_dim);
      }
    } else {
      staged_fp16_transpose_kv_kernel<<<transpose_blocks, kThreads, 0,
                                        stream>>>(
          raw_keys, raw_values_source,
          static_cast<__half*>(workspace.keys),
          static_cast<__half*>(workspace.values), tokens, launch.kv_heads,
          launch.head_dim);
    }
    status = checked(cudaPeekAtLastError(),
                     "transpose staged FP16 K/V");
    if (!status.ok()) return status;

    const auto query_stride = static_cast<long long>(matrix_rows) *
                              launch.head_dim;
    const auto kv_stride = static_cast<long long>(tokens) * launch.head_dim;
    const auto score_stride = static_cast<long long>(matrix_rows) * tokens;
    blas_status = cublasGemmStridedBatchedEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(tokens),
        static_cast<int>(matrix_rows), static_cast<int>(launch.head_dim),
        &alpha, workspace.keys, CUDA_R_16F, static_cast<int>(launch.head_dim),
        kv_stride, workspace.queries, CUDA_R_16F,
        static_cast<int>(launch.head_dim), query_stride, &zero,
        workspace.scores, CUDA_R_32F, static_cast<int>(tokens), score_stride,
        static_cast<int>(launch.kv_heads), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas_status != CUBLAS_STATUS_SUCCESS)
      return checked(blas_status, "staged FP16 QK GEMM");

    staged_fp16_softmax_kernel<<<
        static_cast<unsigned>(state_rows), kThreads, 0, stream>>>(
        workspace.scores, static_cast<__half*>(workspace.probabilities),
        workspace.accumulator, workspace.maxima, workspace.sums,
        launch.first_context_tokens, first_token, tokens, launch.rows,
        grouped_heads, launch.kv_heads, launch.head_dim);
    status = checked(cudaPeekAtLastError(),
                     "staged FP16 online softmax");
    if (!status.ok()) return status;

    const auto output_stride = static_cast<long long>(matrix_rows) *
                               launch.head_dim;
    blas_status = cublasGemmStridedBatchedEx(
        blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(launch.head_dim),
        static_cast<int>(matrix_rows), static_cast<int>(tokens), &alpha,
        workspace.values, CUDA_R_16F, static_cast<int>(launch.head_dim),
        kv_stride, workspace.probabilities, CUDA_R_16F,
        static_cast<int>(tokens), score_stride, &one, workspace.accumulator,
        CUDA_R_32F, static_cast<int>(launch.head_dim), output_stride,
        static_cast<int>(launch.kv_heads), CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (blas_status != CUBLAS_STATUS_SUCCESS)
      return checked(blas_status, "staged FP16 PV GEMM");
    first_token += tokens;
  }
  staged_prefill_finalize_kernel<<<prepare_blocks, kThreads, 0, stream>>>(
      launch.q_and_gate, workspace.accumulator, workspace.sums, launch.output,
      launch.rows, launch.query_heads, launch.kv_heads, launch.head_dim,
      launch.output_gated);
  return checked(cudaPeekAtLastError(),
                 "finalize staged FP16 attention");
}

Status gated_gqa_attention_staged_host_fp16(
    const HostFp16GatedGqaAttentionLaunch& launch,
    const HostFp16GatedGqaAttentionWorkspace& workspace) noexcept {
  return gated_gqa_attention_staged_fp16_impl(launch, workspace, true);
}

Status gated_gqa_attention_staged_device_fp16(
    const DeviceFp16GatedGqaAttentionLaunch& launch,
    const HostFp16GatedGqaAttentionWorkspace& workspace) noexcept {
  return gated_gqa_attention_staged_fp16_impl(launch, workspace, false);
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
Status mamba2_forward(const Mamba2BatchLaunch& launch) noexcept {
  if (!launch.projected || !launch.conv_weights || !launch.conv_bias ||
      !launch.dt_bias || !launch.a_log || !launch.skip ||
      !launch.norm_weight || !launch.conv_state ||
      !launch.recurrent_state || !launch.conv_output || !launch.output ||
      !launch.rows || !launch.heads || !launch.head_dim ||
      !launch.state_size || !launch.groups ||
      launch.heads % launch.groups || !launch.conv_kernel ||
      launch.conv_kernel > 16U || !(launch.epsilon > 0.0F) ||
      !(launch.time_step_min > 0.0F))
    return Status(ErrorCode::invalid_argument, "invalid Mamba2 launch");
  const auto intermediate = launch.heads * launch.head_dim;
  const auto conv_size = intermediate + 2U * launch.groups * launch.state_size;
  const auto projection_size = intermediate + conv_size + launch.heads;
  const auto stream = static_cast<cudaStream_t>(launch.stream);
  for (std::uint32_t row = 0U; row < launch.rows; ++row) {
    const auto* projected = launch.projected +
        static_cast<std::size_t>(row) * projection_size;
    auto* conv = launch.conv_output +
        static_cast<std::size_t>(row) * conv_size;
    auto* output = launch.output +
        static_cast<std::size_t>(row) * intermediate;
    mamba2_conv_kernel<<<(conv_size + kThreads - 1U) / kThreads, kThreads,
                           0, stream>>>(
        projected + intermediate, launch.conv_weights, launch.conv_bias,
        launch.conv_state, conv, conv_size, launch.conv_kernel);
    auto status = checked(cudaPeekAtLastError(), "Mamba2 causal convolution");
    if (!status.ok()) return status;
    mamba2_state_kernel<<<launch.heads, launch.head_dim, 0, stream>>>(
        projected, conv, launch.dt_bias, launch.a_log, launch.skip,
        launch.recurrent_state, output, launch.heads, launch.head_dim,
        launch.state_size, launch.groups, launch.time_step_min);
    status = checked(cudaPeekAtLastError(), "Mamba2 selective state update");
    if (!status.ok()) return status;
    mamba2_gated_group_norm_kernel<<<launch.groups, kThreads, 0, stream>>>(
        projected, output, launch.norm_weight, intermediate, launch.groups,
        launch.epsilon);
    status = checked(cudaPeekAtLastError(), "Mamba2 gated group RMSNorm");
    if (!status.ok()) return status;
  }
  return Status::success();
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

Status topk_logits(const float* values, std::uint32_t count,
                   std::uint32_t top_k, float* output_values,
                   std::uint32_t* output_indices, void* raw) noexcept {
  if (!values || !count || !top_k || top_k > 64U || top_k > count ||
      !output_values || !output_indices)
    return Status(ErrorCode::invalid_argument, "invalid top-k logits");
  topk_logits_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(raw)>>>(
      values, count, top_k, output_values, output_indices);
  return checked(cudaPeekAtLastError(), "top-k logits");
}

}  // namespace expert::runtime::cuda
