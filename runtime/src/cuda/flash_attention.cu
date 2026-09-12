#include "expert/runtime/cuda/flash_attention.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_LOCAL
#include "flash_fwd_launch_template.h"

namespace expert::runtime::cuda {
namespace {

Status checked(cudaError_t error, const char* operation) {
  return error == cudaSuccess
      ? Status::success()
      : Status(ErrorCode::upload_failed,
               std::string(operation) + ": " + cudaGetErrorString(error));
}

template <bool IsCausal>
Status launch_segment(const FlashGqaSegmentLaunch& launch,
                      flash::Flash_fwd_params& params,
                      cudaStream_t stream) {
  using Traits = Flash_fwd_kernel_traits<
      256, 64, 64, 4, false, false, cutlass::bfloat16_t>;
  auto kernel = &flash::flash_fwd_splitkv_kernel<
      Traits, IsCausal, false, false, false, true, false, true, false>;
  auto status = checked(
      cudaFuncSetAttribute(kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           Traits::kSmemSize),
      "configure FlashAttention shared memory");
  if (!status.ok()) return status;
  const dim3 grid((launch.rows + Traits::kBlockM - 1U) /
                      Traits::kBlockM,
                  1U, launch.query_heads);
  kernel<<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(params);
  return checked(cudaPeekAtLastError(), "launch FlashAttention segment");
}

}  // namespace

Status flash_gqa_segment_bf16(
    const FlashGqaSegmentLaunch& launch) noexcept {
  if (!launch.queries || !launch.keys || !launch.values || !launch.output ||
      !launch.lse || !launch.rows || !launch.tokens ||
      !launch.query_heads || !launch.kv_heads ||
      launch.query_heads % launch.kv_heads || launch.head_dim != 256U)
    return Status(ErrorCode::invalid_argument,
                  "invalid BF16 FlashAttention segment");

  flash::Flash_fwd_params params{};
  params.q_ptr = const_cast<void*>(launch.queries);
  params.k_ptr = const_cast<void*>(launch.keys);
  params.v_ptr = const_cast<void*>(launch.values);
  params.o_ptr = launch.output;
  params.oaccum_ptr = launch.output;
  params.softmax_lse_ptr = launch.lse;
  params.softmax_lseaccum_ptr = launch.lse;
  params.q_batch_stride =
      static_cast<std::int64_t>(launch.rows) * launch.query_heads *
      launch.head_dim;
  params.q_row_stride =
      static_cast<std::int64_t>(launch.query_heads) * launch.head_dim;
  params.q_head_stride = launch.head_dim;
  params.k_batch_stride =
      static_cast<std::int64_t>(launch.kv_heads) * launch.tokens *
      launch.head_dim;
  params.v_batch_stride = params.k_batch_stride;
  params.k_row_stride = launch.head_dim;
  params.v_row_stride = launch.head_dim;
  params.k_head_stride =
      static_cast<std::int64_t>(launch.tokens) * launch.head_dim;
  params.v_head_stride = params.k_head_stride;
  params.o_batch_stride = params.q_batch_stride;
  params.o_row_stride = params.q_row_stride;
  params.o_head_stride = launch.head_dim;
  params.b = 1;
  params.seqlen_q = static_cast<int>(launch.rows);
  params.seqlen_k = static_cast<int>(launch.tokens);
  params.seqlen_knew = 0;
  params.d = static_cast<int>(launch.head_dim);
  params.seqlen_q_rounded =
      static_cast<int>((launch.rows + 127U) & ~127U);
  params.seqlen_k_rounded =
      static_cast<int>((launch.tokens + 127U) & ~127U);
  params.d_rounded = static_cast<int>(launch.head_dim);
  params.total_q = static_cast<int>(launch.rows);
  params.h = static_cast<int>(launch.query_heads);
  params.h_k = static_cast<int>(launch.kv_heads);
  params.h_h_k_ratio =
      static_cast<int>(launch.query_heads / launch.kv_heads);
  params.scale_softmax =
      1.0F / std::sqrt(static_cast<float>(launch.head_dim));
  params.scale_softmax_log2 =
      params.scale_softmax * static_cast<float>(M_LOG2E);
  params.p_dropout = 1.0F;
  params.rp_dropout = 1.0F;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.window_size_left = -1;
  params.window_size_right = launch.causal ? 0 : -1;
  params.is_bf16 = true;
  params.is_causal = launch.causal;
  params.is_seqlens_k_cumulative = true;
  params.num_splits = 1;

  const auto stream = static_cast<cudaStream_t>(launch.stream);
  return launch.causal ? launch_segment<true>(launch, params, stream)
                       : launch_segment<false>(launch, params, stream);
}

}  // namespace expert::runtime::cuda
