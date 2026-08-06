#include "expert/runtime/cuda/deepseek_verify.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

namespace expert::runtime::cuda {
namespace {

constexpr std::array<std::uint32_t, kDeepSeekLayers> kCompressionRatios = {
    0U, 0U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U,
    128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
    4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U,
    128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 0U};
constexpr std::uint64_t kStreamValues = 4ULL * 4096U;
constexpr std::uint64_t kSecondaryStreamBytes =
    2ULL * kStreamValues * sizeof(float);

bool add_checked(std::uint64_t value, std::uint64_t& total) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += value;
  return true;
}

Status failure(cudaError_t error, const char* operation) noexcept {
  return {ErrorCode::internal,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

}  // namespace

DeepSeekVerifyStateSize deepseek_verify_state_size() noexcept {
  DeepSeekVerifyStateSize result{Status::success()};
  const auto ffn_bytes = deepseek_ffn_state_size();
  if (ffn_bytes > std::numeric_limits<std::uint64_t>::max() /
                      kDeepSeekLayers)
    return {{ErrorCode::invalid_argument,
             "DeepSeek verify FFN state size overflow"}};
  result.secondary_ffn_bytes = ffn_bytes * kDeepSeekLayers;
  result.pair_ffn_bytes = deepseek_ffn_pair_workspace_size();
  result.secondary_io_bytes = deepseek_io_state_size();
  result.secondary_stream_bytes = kSecondaryStreamBytes;
  for (const auto ratio : kCompressionRatios) {
    if (!add_checked(deepseek_attention_speculative_checkpoint_size(ratio),
                     result.rollback_bytes))
      return {{ErrorCode::invalid_argument,
               "DeepSeek verify rollback size overflow"}};
  }
  for (const auto value : {result.secondary_ffn_bytes, result.pair_ffn_bytes,
                           result.secondary_io_bytes,
                           result.secondary_stream_bytes,
                           result.rollback_bytes}) {
    if (!add_checked(value, result.total_bytes))
      return {{ErrorCode::invalid_argument,
               "DeepSeek verify state size overflow"}};
  }
  return result;
}

DeepSeekVerifyState::~DeepSeekVerifyState() {
  if (rollback_allocation_) static_cast<void>(cudaFree(rollback_allocation_));
  if (stream_allocation_) static_cast<void>(cudaFree(stream_allocation_));
}

DeepSeekVerifyLayerStateView DeepSeekVerifyState::layer(
    std::uint32_t index) const noexcept {
  if (!request_ || index >= kDeepSeekLayers) return {};
  const auto primary = request_->layer(index);
  return {primary.attention_weights, primary.attention_state,
          primary.ffn_weights,
          {primary.ffn_state, secondary_ffn_states_[index].get()},
          primary.compress_ratio, rollback_checkpoints_[index]};
}

const float* DeepSeekVerifyState::primary_streams() const noexcept {
  return request_ ? request_->streams_a_ : nullptr;
}

Status DeepSeekVerifyState::embed_speculative(std::uint32_t token,
                                              void* stream) noexcept {
  if (!request_ || !speculative_streams_a_)
    return {ErrorCode::internal, "DeepSeek verify state is incomplete"};
  return deepseek_embed(request_->io_weights_, token, speculative_streams_a_,
                        stream);
}

Status DeepSeekVerifyState::begin_transaction(
    std::uint32_t speculative_position) noexcept {
  if (active_ || !request_ ||
      speculative_position >= request_->max_context_tokens())
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek verification transaction"};
  active_ = true;
  speculative_position_ = speculative_position;
  recurrent_boundary_ = (speculative_position + 1U) % 4U == 0U;
  checkpointed_.fill(false);
  return Status::success();
}

Status DeepSeekVerifyState::checkpoint_layer(std::uint32_t index,
                                             void* stream) noexcept {
  if (!active_ || index >= kDeepSeekLayers || checkpointed_[index])
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek verification checkpoint"};
  if (!recurrent_boundary_ || kCompressionRatios[index] != 4U)
    return Status::success();
  const auto view = layer(index);
  if (!view.attention_state || !view.rollback_checkpoint)
    return {ErrorCode::internal,
            "DeepSeek verification checkpoint state is incomplete"};
  auto status = view.attention_state->checkpoint_speculative_state(
      view.rollback_checkpoint, stream);
  if (status.ok()) checkpointed_[index] = true;
  return status;
}

Status DeepSeekVerifyState::project_pair_logits(void* stream) noexcept {
  if (!active_ || !request_ || !request_->io_state_ || !secondary_io_state_)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek pair vocabulary projection"};
  auto status = deepseek_head(request_->io_weights_, request_->streams_a_,
                              *request_->io_state_, 1e-6F, stream);
  if (!status.ok()) return status;
  return deepseek_head(request_->io_weights_, speculative_streams_a_,
                       *secondary_io_state_, 1e-6F, stream);
}

const std::uint32_t* DeepSeekVerifyState::primary_sampled_token()
    const noexcept {
  return request_ && request_->io_state_ ? request_->io_state_->sampled_token()
                                         : nullptr;
}

const std::uint32_t* DeepSeekVerifyState::bonus_sampled_token() const noexcept {
  return secondary_io_state_ ? secondary_io_state_->sampled_token() : nullptr;
}

Status DeepSeekVerifyState::restore_checkpoints(void* stream) noexcept {
  auto result = Status::success();
  for (std::uint32_t index = 0U; index < kDeepSeekLayers; ++index) {
    if (!checkpointed_[index]) continue;
    const auto view = layer(index);
    const auto restored = view.attention_state->restore_speculative_state(
        view.rollback_checkpoint, stream);
    if (result.ok() && !restored.ok()) result = restored;
    checkpointed_[index] = false;
  }
  return result;
}

Status DeepSeekVerifyState::finish_transaction(bool accept,
                                               void* stream) noexcept {
  if (!active_)
    return {ErrorCode::invalid_argument,
            "DeepSeek verification transaction is not active"};
  Status status = Status::success();
  if (accept) {
    const auto error = cudaMemcpyAsync(
        request_->streams_a_, speculative_streams_a_,
        kStreamValues * sizeof(float), cudaMemcpyDeviceToDevice,
        static_cast<cudaStream_t>(stream));
    if (error != cudaSuccess)
      status = failure(error, "commit DeepSeek speculative streams");
    checkpointed_.fill(false);
  } else {
    status = restore_checkpoints(stream);
  }
  active_ = false;
  recurrent_boundary_ = false;
  return status;
}

Status DeepSeekVerifyState::abort_transaction(void* stream) noexcept {
  if (!active_) return Status::success();
  auto status = restore_checkpoints(stream);
  active_ = false;
  recurrent_boundary_ = false;
  return status;
}

DeepSeekVerifyStateResult create_deepseek_verify_state(
    std::shared_ptr<DeepSeekRequestState> request,
    std::uint64_t device_state_budget_bytes) noexcept {
  if (!request || device_state_budget_bytes == 0U)
    return {{ErrorCode::invalid_argument,
             "DeepSeek verify state requires a request and budget"}, {}};
  const auto estimate = deepseek_verify_state_size();
  if (!estimate.status.ok()) return {estimate.status, {}};
  if (estimate.total_bytes > device_state_budget_bytes)
    return {{ErrorCode::backpressure,
             "DeepSeek verify state exceeds its CUDA memory budget"}, {}};

  auto candidate = std::shared_ptr<DeepSeekVerifyState>(
      new DeepSeekVerifyState());
  candidate->request_ = std::move(request);
  std::uint64_t actual = 0U;
  for (std::uint32_t layer = 0U; layer < kDeepSeekLayers; ++layer) {
    auto ffn = create_deepseek_ffn_state(layer);
    if (!ffn.status.ok()) return {ffn.status, {}};
    candidate->secondary_ffn_states_[layer] = std::move(ffn.state);
    if (!add_checked(candidate->secondary_ffn_states_[layer]->bytes(), actual))
      return {{ErrorCode::internal,
               "DeepSeek verify allocation size overflow"}, {}};
  }
  auto pair = create_deepseek_ffn_pair_workspace();
  if (!pair.status.ok()) return {pair.status, {}};
  candidate->ffn_workspace_ = std::move(pair.workspace);
  if (!add_checked(candidate->ffn_workspace_->bytes(), actual))
    return {{ErrorCode::internal,
             "DeepSeek verify pair workspace size overflow"}, {}};
  auto io = create_deepseek_io_state();
  if (!io.status.ok()) return {io.status, {}};
  candidate->secondary_io_state_ = std::move(io.state);
  if (!add_checked(candidate->secondary_io_state_->bytes(), actual))
    return {{ErrorCode::internal,
             "DeepSeek verify I/O size overflow"}, {}};

  auto error = cudaMalloc(&candidate->stream_allocation_,
                          estimate.secondary_stream_bytes);
  if (error != cudaSuccess)
    return {failure(error, "allocate DeepSeek verify streams"), {}};
  candidate->speculative_streams_a_ =
      static_cast<float*>(candidate->stream_allocation_);
  candidate->speculative_streams_b_ =
      candidate->speculative_streams_a_ + kStreamValues;
  error = cudaMemset(candidate->stream_allocation_, 0,
                     estimate.secondary_stream_bytes);
  if (error != cudaSuccess)
    return {failure(error, "reset DeepSeek verify streams"), {}};
  if (!add_checked(estimate.secondary_stream_bytes, actual))
    return {{ErrorCode::internal,
             "DeepSeek verify stream size overflow"}, {}};

  if (estimate.rollback_bytes != 0U) {
    error = cudaMalloc(&candidate->rollback_allocation_,
                       estimate.rollback_bytes);
    if (error != cudaSuccess)
      return {failure(error, "allocate DeepSeek verify rollback state"), {}};
    auto* cursor = static_cast<std::byte*>(candidate->rollback_allocation_);
    for (std::uint32_t layer = 0U; layer < kDeepSeekLayers; ++layer) {
      const auto bytes = deepseek_attention_speculative_checkpoint_size(
          kCompressionRatios[layer]);
      if (bytes == 0U) continue;
      candidate->rollback_checkpoints_[layer] = cursor;
      cursor += bytes;
    }
    if (!add_checked(estimate.rollback_bytes, actual))
      return {{ErrorCode::internal,
               "DeepSeek verify rollback size overflow"}, {}};
  }
  if (actual != estimate.total_bytes)
    return {{ErrorCode::internal,
             "DeepSeek verify estimate does not match allocations"}, {}};
  candidate->bytes_ = actual;
  return {Status::success(), std::move(candidate)};
}

}  // namespace expert::runtime::cuda
