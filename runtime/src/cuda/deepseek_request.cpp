#include "expert/runtime/cuda/deepseek_request.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace expert::runtime::cuda {
namespace {

constexpr std::uint64_t kStreamValues = 4ULL * 4096U;
constexpr std::uint64_t kStreamBytes = 2ULL * kStreamValues * sizeof(float);

bool add_checked(std::uint64_t value, std::uint64_t& total) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += value;
  return true;
}

}  // namespace

DeepSeekRequestStateSize deepseek_request_state_size(
    std::uint32_t max_context_tokens,
    std::span<const std::uint32_t> compression_ratios) noexcept {
  if (compression_ratios.empty())
    return {{ErrorCode::invalid_argument,
             "DeepSeek layer program is empty"}, 0U, 0U, 0U, 0U, 0U};
  DeepSeekRequestStateSize result{Status::success(), 0U, 0U,
                                  deepseek_io_state_size(), kStreamBytes, 0U};
  for (const auto ratio : compression_ratios) {
    const auto size = deepseek_attention_state_size(ratio, max_context_tokens);
    if (!size.status.ok()) return {size.status, 0U, 0U, 0U, 0U, 0U};
    if (!add_checked(size.bytes, result.attention_bytes)) {
      return {{ErrorCode::invalid_argument,
               "DeepSeek attention state size overflow"},
              0U, 0U, 0U, 0U, 0U};
    }
  }
  const auto ffn_layer_bytes = deepseek_ffn_state_size();
  if (ffn_layer_bytes > std::numeric_limits<std::uint64_t>::max() /
                             compression_ratios.size()) {
    return {{ErrorCode::invalid_argument, "DeepSeek FFN state size overflow"},
            0U, 0U, 0U, 0U, 0U};
  }
  result.ffn_bytes = ffn_layer_bytes * compression_ratios.size();
  result.total_bytes = result.attention_bytes;
  if (!add_checked(result.ffn_bytes, result.total_bytes)) {
    return {{ErrorCode::invalid_argument, "DeepSeek request state size overflow"},
            0U, 0U, 0U, 0U, 0U};
  }
  if (!add_checked(result.io_bytes, result.total_bytes)) {
    return {{ErrorCode::invalid_argument, "DeepSeek I/O state size overflow"},
            0U, 0U, 0U, 0U, 0U};
  }
  if (!add_checked(result.stream_bytes, result.total_bytes)) {
    return {{ErrorCode::invalid_argument, "DeepSeek stream state size overflow"},
            0U, 0U, 0U, 0U, 0U};
  }
  return result;
}

DeepSeekRequestState::~DeepSeekRequestState() {
  if (stream_allocation_) static_cast<void>(cudaFree(stream_allocation_));
}

DeepSeekLayerStateView DeepSeekRequestState::layer(
    std::uint32_t index) const noexcept {
  if (index >= attention_weights_.size()) return {};
  return {&attention_weights_[index], attention_states_[index].get(),
          &ffn_weights_[index], ffn_states_[index].get(),
          compression_ratios_[index]};
}

Status DeepSeekRequestState::embed(std::uint32_t token,
                                   void* stream) noexcept {
  return deepseek_embed(io_weights_, token, streams_a_, stream);
}

Status DeepSeekRequestState::project_logits(void* stream) noexcept {
  if (!io_state_)
    return {ErrorCode::internal, "DeepSeek request has no I/O state"};
  return deepseek_head(io_weights_, streams_a_, *io_state_, 1e-6F, stream);
}

DeepSeekRequestStateResult create_deepseek_request_state(
    std::shared_ptr<const DeepSeekResidentModelState> model,
    const DeepSeekRequestConfig& config) noexcept {
  if (!model || config.device_state_budget_bytes == 0U) {
    return {{ErrorCode::invalid_argument,
             "DeepSeek request state requires a model and a nonzero budget"},
            {}};
  }
  const auto estimate = deepseek_request_state_size(
      config.max_context_tokens, config.compression_ratios);
  if (!estimate.status.ok()) return {estimate.status, {}};
  if (estimate.total_bytes > config.device_state_budget_bytes) {
    return {{ErrorCode::backpressure,
             "DeepSeek request state exceeds its CUDA memory budget"}, {}};
  }

  auto candidate = std::shared_ptr<DeepSeekRequestState>(
      new DeepSeekRequestState());
  candidate->model_ = std::move(model);
  candidate->max_context_tokens_ = config.max_context_tokens;
  candidate->compression_ratios_.assign(config.compression_ratios.begin(),
                                        config.compression_ratios.end());
  candidate->attention_weights_.resize(config.compression_ratios.size());
  candidate->ffn_weights_.resize(config.compression_ratios.size());
  candidate->attention_states_.resize(config.compression_ratios.size());
  candidate->ffn_states_.resize(config.compression_ratios.size());
  std::uint64_t actual_bytes = 0U;
  auto status = candidate->model_->bind_io(candidate->io_weights_);
  if (!status.ok()) return {status, {}};
  auto io = create_deepseek_io_state();
  if (!io.status.ok()) return {io.status, {}};
  candidate->io_state_ = std::move(io.state);
  if (!add_checked(candidate->io_state_->bytes(), actual_bytes)) {
    return {{ErrorCode::internal,
             "DeepSeek allocated I/O state size overflow"}, {}};
  }
  for (std::uint32_t layer = 0U;
       layer < config.compression_ratios.size(); ++layer) {
    status = candidate->model_->bind_attention(
        layer, config.compression_ratios[layer],
        candidate->attention_weights_[layer]);
    if (!status.ok()) return {status, {}};
    status = candidate->model_->bind_ffn(
                                         layer,
                                         layer < config.hash_router_layers
                                             ? DeepSeekRouterKind::hash
                                             : DeepSeekRouterKind::learned,
                                         candidate->ffn_weights_[layer]);
    if (!status.ok()) return {status, {}};

    auto attention = create_deepseek_attention_state(
        config.compression_ratios[layer], config.max_context_tokens);
    if (!attention.status.ok()) return {attention.status, {}};
    auto ffn = create_deepseek_ffn_state(layer);
    if (!ffn.status.ok()) return {ffn.status, {}};
    candidate->attention_states_[layer] = std::move(attention.state);
    candidate->ffn_states_[layer] = std::move(ffn.state);
    if (!add_checked(candidate->attention_states_[layer]->bytes(),
                     actual_bytes) ||
        !add_checked(candidate->ffn_states_[layer]->bytes(), actual_bytes)) {
      return {{ErrorCode::internal,
               "DeepSeek allocated request state size overflow"}, {}};
    }
  }
  auto error = cudaMalloc(reinterpret_cast<void**>(
                              &candidate->stream_allocation_),
                          estimate.stream_bytes);
  if (error != cudaSuccess) {
    return {{ErrorCode::internal,
             std::string("DeepSeek stream state allocation: ") +
                 cudaGetErrorString(error)}, {}};
  }
  candidate->streams_a_ = candidate->stream_allocation_;
  candidate->streams_b_ = candidate->streams_a_ + kStreamValues;
  error = cudaMemset(candidate->stream_allocation_, 0, estimate.stream_bytes);
  if (error != cudaSuccess) {
    return {{ErrorCode::internal,
             std::string("DeepSeek stream state reset: ") +
                 cudaGetErrorString(error)}, {}};
  }
  if (!add_checked(estimate.stream_bytes, actual_bytes)) {
    return {{ErrorCode::internal,
             "DeepSeek allocated stream state size overflow"}, {}};
  }
  if (actual_bytes != estimate.total_bytes) {
    return {{ErrorCode::internal,
             "DeepSeek request state estimate does not match allocations"},
            {}};
  }
  candidate->bytes_ = actual_bytes;
  return {Status::success(), std::move(candidate)};
}

}  // namespace expert::runtime::cuda
