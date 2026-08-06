#include "expert/runtime/cuda/deepseek_mtp.hpp"

#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint32_t kThreads = 256U;
constexpr std::uint64_t kTokenValues = kDeepSeekHidden;
constexpr std::uint64_t kStreamValues =
    static_cast<std::uint64_t>(kDeepSeekMtpStreams) * kDeepSeekHidden;
constexpr std::uint64_t kAllocationValues =
    kTokenValues + kStreamValues + kTokenValues + kStreamValues;
constexpr std::uint64_t kAllocationBytes = kAllocationValues * sizeof(float);

Status failure(cudaError_t error, const char* operation) {
  return {ErrorCode::internal,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

__global__ void broadcast_add_kernel(float* streams, const float* token) {
  const auto index =
      static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < kStreamValues)
    streams[index] += token[index % kDeepSeekHidden];
}

}  // namespace

DeepSeekMtpGlueState::DeepSeekMtpGlueState(
    void* allocation, std::uint64_t allocation_bytes,
    std::shared_ptr<DeepSeekIoState> head_state) noexcept
    : allocation_(allocation),
      allocation_bytes_(allocation_bytes),
      head_state_(std::move(head_state)) {
  auto* cursor = static_cast<float*>(allocation_);
  normalized_token_ = cursor;
  cursor += kTokenValues;
  normalized_streams_ = cursor;
  cursor += kStreamValues;
  token_projection_ = cursor;
  cursor += kTokenValues;
  mixed_ = cursor;
}

DeepSeekMtpGlueState::~DeepSeekMtpGlueState() {
  if (allocation_) static_cast<void>(cudaFree(allocation_));
}

std::uint64_t DeepSeekMtpGlueState::bytes() const noexcept {
  return allocation_bytes_ + (head_state_ ? head_state_->bytes() : 0U);
}

DeepSeekMtpGlueStateResult create_deepseek_mtp_glue_state() noexcept {
  auto head = create_deepseek_io_state();
  if (!head.status.ok()) return {head.status, {}};
  void* allocation = nullptr;
  auto error = cudaMalloc(&allocation, kAllocationBytes);
  if (error != cudaSuccess)
    return {failure(error, "allocate DeepSeek MTP glue state"), {}};
  error = cudaMemset(allocation, 0, kAllocationBytes);
  if (error != cudaSuccess) {
    static_cast<void>(cudaFree(allocation));
    return {failure(error, "reset DeepSeek MTP glue state"), {}};
  }
  return {Status::success(), std::shared_ptr<DeepSeekMtpGlueState>(
                                 new DeepSeekMtpGlueState(
                                     allocation, kAllocationBytes,
                                     std::move(head.state)))};
}

Status deepseek_mtp_mix(const DeepSeekMtpGlueBinding& weights,
                        const float* embedding,
                        const float* previous_streams,
                        DeepSeekMtpGlueState& state, float epsilon,
                        void* raw_stream) noexcept {
  if (!embedding || !previous_streams || !weights.embedding_norm ||
      !weights.hidden_norm || !weights.e_projection.weights ||
      !weights.h_projection.weights || epsilon <= 0.0F)
    return {ErrorCode::invalid_argument, "invalid DeepSeek MTP mix launch"};
  auto status = rms_norm_bf16_weight(
      embedding, weights.embedding_norm, state.normalized_token_,
      kDeepSeekHidden, epsilon, raw_stream);
  if (!status.ok()) return status;
  status = rms_norm_bf16_weight_batch(
      previous_streams, weights.hidden_norm, state.normalized_streams_,
      kDeepSeekMtpStreams, kDeepSeekHidden, epsilon, raw_stream);
  if (!status.ok()) return status;
  status = gemv(weights.e_projection, state.normalized_token_,
                state.token_projection_, raw_stream);
  if (!status.ok()) return status;
  status = gemv_batch(weights.h_projection, state.normalized_streams_,
                      state.mixed_, kDeepSeekMtpStreams, raw_stream);
  if (!status.ok()) return status;
  broadcast_add_kernel<<<
      (kStreamValues + kThreads - 1U) / kThreads, kThreads, 0,
      static_cast<cudaStream_t>(raw_stream)>>>(state.mixed_,
                                               state.token_projection_);
  const auto error = cudaPeekAtLastError();
  return error == cudaSuccess ? Status::success()
                              : failure(error, "DeepSeek MTP projection mix");
}

Status deepseek_mtp_collapse(const DeepSeekMtpGlueBinding& weights,
                             const float* decoder_streams,
                             DeepSeekMtpGlueState& state, float epsilon,
                             void* raw_stream) noexcept {
  DeepSeekIoBinding binding;
  binding.final_norm = weights.output_norm;
  binding.head_function = weights.head_function;
  binding.head_base = weights.head_base;
  binding.head_scale = weights.head_scale;
  return deepseek_hc_head(binding, decoder_streams, *state.head_state_, epsilon,
                          raw_stream);
}

}  // namespace expert::runtime::cuda
