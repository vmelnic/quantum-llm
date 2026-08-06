#include "expert/runtime/cuda/deepseek_mtp_request.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <string>
#include <utility>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint64_t kEmbeddingValues = 4096U;
constexpr std::uint64_t kStreamValues = 4ULL * 4096U;
constexpr std::uint64_t kWorkspaceBytes =
    (kEmbeddingValues + 2U * kStreamValues) * sizeof(float);

bool add_checked(std::uint64_t value, std::uint64_t& total) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += value;
  return true;
}

Status failure(cudaError_t error, const char* operation) {
  return {ErrorCode::internal,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

}  // namespace

DeepSeekMtpRequestStateSize deepseek_mtp_request_state_size(
    std::uint32_t max_context_tokens) noexcept {
  const auto attention = deepseek_attention_state_size(0U, max_context_tokens);
  if (!attention.status.ok())
    return {attention.status, 0U, 0U, 0U, 0U, 0U};
  DeepSeekMtpRequestStateSize result{
      Status::success(), attention.bytes, deepseek_ffn_state_size(),
      deepseek_mtp_glue_state_size(), kWorkspaceBytes, 0U};
  result.total_bytes = result.attention_bytes;
  if (!add_checked(result.ffn_bytes, result.total_bytes) ||
      !add_checked(result.glue_bytes, result.total_bytes) ||
      !add_checked(result.workspace_bytes, result.total_bytes))
    return {{ErrorCode::invalid_argument, "MTP request size overflow"},
            0U, 0U, 0U, 0U, 0U};
  return result;
}

DeepSeekMtpRequestState::~DeepSeekMtpRequestState() {
  if (workspace_allocation_)
    static_cast<void>(cudaFree(workspace_allocation_));
}

Status DeepSeekMtpRequestState::prepare(
    std::uint32_t next_token, const float* target_streams,
    std::uint32_t position, const float* cosine, const float* sine,
    void* raw_stream) noexcept {
  if (prepared_ || !target_streams || !cosine || !sine ||
      next_token >= kDeepSeekVocab || position >= max_context_tokens_)
    return {ErrorCode::invalid_argument, "invalid MTP request prepare"};
  auto status = deepseek_embed(target_io_, next_token, embedding_, raw_stream);
  if (!status.ok()) return status;
  if (position == 0U) {
    const auto error = cudaMemsetAsync(
        embedding_, 0, kEmbeddingValues * sizeof(float),
        static_cast<cudaStream_t>(raw_stream));
    if (error != cudaSuccess)
      return failure(error, "mask initial MTP embedding");
  }
  status = deepseek_mtp_mix(glue_weights_, embedding_, target_streams,
                            *glue_state_, 1e-6F, raw_stream);
  if (!status.ok()) return status;
  status = deepseek_attention_decode({
      &attention_weights_, attention_state_.get(),
      glue_state_->mixed_streams(), attention_streams_, cosine, sine,
      cosine, sine, position, 1e-6F, 20U, raw_stream});
  if (!status.ok()) return status;
  status = deepseek_ffn_route({
      &ffn_weights_, ffn_state_.get(), attention_streams_, next_token,
      1e-6F, 20U, raw_stream});
  if (!status.ok()) return status;
  prepared_ = true;
  complete_ = false;
  return Status::success();
}

Status DeepSeekMtpRequestState::complete(
    const DeviceExpertEntry* directory_entries,
    std::uint32_t experts_per_layer, void* raw_stream) noexcept {
  if (!prepared_ || complete_ || !directory_entries ||
      experts_per_layer != 257U)
    return {ErrorCode::invalid_argument, "invalid MTP request completion"};
  auto status = deepseek_ffn_execute({
      &ffn_weights_, ffn_state_.get(), directory_entries,
      attention_streams_, block_streams_, experts_per_layer,
      raw_stream, nullptr});
  if (!status.ok()) return status;
  status = deepseek_mtp_collapse(glue_weights_, block_streams_, *glue_state_,
                                 1e-6F, raw_stream);
  if (!status.ok()) return status;
  status = deepseek_mtp_vocab_head(target_io_.head, *glue_state_, raw_stream);
  if (!status.ok()) return status;
  prepared_ = false;
  complete_ = true;
  return Status::success();
}

Status DeepSeekMtpRequestState::abandon_draft() noexcept {
  if (!prepared_)
    return {ErrorCode::invalid_argument,
            "MTP request has no prepared draft to abandon"};
  prepared_ = false;
  complete_ = false;
  return Status::success();
}

DeepSeekMtpRequestStateResult create_deepseek_mtp_request_state(
    std::shared_ptr<const DeepSeekResidentModelState> target_model,
    std::shared_ptr<const DeepSeekResidentTensorState> mtp_model,
    const DeepSeekMtpRequestConfig& config) noexcept {
  if (!target_model || !mtp_model || config.device_state_budget_bytes == 0U)
    return {{ErrorCode::invalid_argument,
             "MTP request requires target/MTP models and a device budget"},
            {}};
  const auto estimate = deepseek_mtp_request_state_size(
      config.max_context_tokens);
  if (!estimate.status.ok()) return {estimate.status, {}};
  if (estimate.total_bytes > config.device_state_budget_bytes)
    return {{ErrorCode::backpressure,
             "MTP request state exceeds its CUDA memory budget"}, {}};

  auto candidate = std::shared_ptr<DeepSeekMtpRequestState>(
      new DeepSeekMtpRequestState());
  candidate->target_model_ = std::move(target_model);
  candidate->mtp_model_ = std::move(mtp_model);
  candidate->max_context_tokens_ = config.max_context_tokens;
  auto status = candidate->target_model_->bind_io(candidate->target_io_);
  if (!status.ok()) return {status, {}};
  status = candidate->mtp_model_->bind_mtp_glue(
      "mtp.0", candidate->glue_weights_);
  if (!status.ok()) return {status, {}};
  status = candidate->mtp_model_->bind_attention(
      "mtp.0", 0U, 0U, candidate->attention_weights_);
  if (!status.ok()) return {status, {}};
  status = candidate->mtp_model_->bind_ffn(
      "mtp.0", 0U, DeepSeekRouterKind::learned, candidate->ffn_weights_);
  if (!status.ok()) return {status, {}};

  auto glue = create_deepseek_mtp_glue_state();
  if (!glue.status.ok()) return {glue.status, {}};
  auto attention = create_deepseek_attention_state(
      0U, config.max_context_tokens);
  if (!attention.status.ok()) return {attention.status, {}};
  auto ffn = create_deepseek_ffn_state(0U);
  if (!ffn.status.ok()) return {ffn.status, {}};
  candidate->glue_state_ = std::move(glue.state);
  candidate->attention_state_ = std::move(attention.state);
  candidate->ffn_state_ = std::move(ffn.state);
  auto error = cudaMalloc(
      reinterpret_cast<void**>(&candidate->workspace_allocation_),
      estimate.workspace_bytes);
  if (error != cudaSuccess)
    return {failure(error, "allocate MTP request workspace"), {}};
  candidate->embedding_ = candidate->workspace_allocation_;
  candidate->attention_streams_ =
      candidate->embedding_ + kEmbeddingValues;
  candidate->block_streams_ =
      candidate->attention_streams_ + kStreamValues;
  error = cudaMemset(candidate->workspace_allocation_, 0,
                     estimate.workspace_bytes);
  if (error != cudaSuccess)
    return {failure(error, "reset MTP request workspace"), {}};
  candidate->bytes_ = estimate.total_bytes;
  return {Status::success(), std::move(candidate)};
}

}  // namespace expert::runtime::cuda
