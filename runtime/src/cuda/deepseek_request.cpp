#include "expert/runtime/cuda/deepseek_request.hpp"

#include <array>
#include <limits>
#include <memory>
#include <utility>

namespace expert::runtime::cuda {
namespace {

constexpr std::array<std::uint32_t, kDeepSeekLayers> kCompressionRatios = {
    0U, 0U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U,
    128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U,
    4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U,
    128U, 4U, 128U, 4U, 128U, 4U, 128U, 4U, 128U, 0U};

bool add_checked(std::uint64_t value, std::uint64_t& total) noexcept {
  if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += value;
  return true;
}

}  // namespace

DeepSeekRequestStateSize deepseek_request_state_size(
    std::uint32_t max_context_tokens) noexcept {
  DeepSeekRequestStateSize result{Status::success(), 0U, 0U, 0U};
  for (const auto ratio : kCompressionRatios) {
    const auto size = deepseek_attention_state_size(ratio, max_context_tokens);
    if (!size.status.ok()) return {size.status, 0U, 0U, 0U};
    if (!add_checked(size.bytes, result.attention_bytes)) {
      return {{ErrorCode::invalid_argument,
               "DeepSeek attention state size overflow"}, 0U, 0U, 0U};
    }
  }
  const auto ffn_layer_bytes = deepseek_ffn_state_size();
  if (ffn_layer_bytes > std::numeric_limits<std::uint64_t>::max() /
                             kDeepSeekLayers) {
    return {{ErrorCode::invalid_argument, "DeepSeek FFN state size overflow"},
            0U, 0U, 0U};
  }
  result.ffn_bytes = ffn_layer_bytes * kDeepSeekLayers;
  result.total_bytes = result.attention_bytes;
  if (!add_checked(result.ffn_bytes, result.total_bytes)) {
    return {{ErrorCode::invalid_argument, "DeepSeek request state size overflow"},
            0U, 0U, 0U};
  }
  return result;
}

DeepSeekLayerStateView DeepSeekRequestState::layer(
    std::uint32_t index) const noexcept {
  if (index >= kDeepSeekLayers) return {};
  return {&attention_weights_[index], attention_states_[index].get(),
          &ffn_weights_[index], ffn_states_[index].get(),
          kCompressionRatios[index]};
}

DeepSeekRequestStateResult create_deepseek_request_state(
    std::shared_ptr<const DeepSeekResidentModelState> model,
    const DeepSeekRequestConfig& config) noexcept {
  if (!model || config.device_state_budget_bytes == 0U) {
    return {{ErrorCode::invalid_argument,
             "DeepSeek request state requires a model and a nonzero budget"},
            {}};
  }
  const auto estimate = deepseek_request_state_size(config.max_context_tokens);
  if (!estimate.status.ok()) return {estimate.status, {}};
  if (estimate.total_bytes > config.device_state_budget_bytes) {
    return {{ErrorCode::backpressure,
             "DeepSeek request state exceeds its CUDA memory budget"}, {}};
  }

  auto candidate = std::shared_ptr<DeepSeekRequestState>(
      new DeepSeekRequestState());
  candidate->model_ = std::move(model);
  candidate->max_context_tokens_ = config.max_context_tokens;
  std::uint64_t actual_bytes = 0U;
  for (std::uint32_t layer = 0U; layer < kDeepSeekLayers; ++layer) {
    auto status = candidate->model_->bind_attention(
        layer, kCompressionRatios[layer],
        candidate->attention_weights_[layer]);
    if (!status.ok()) return {status, {}};
    status = candidate->model_->bind_ffn(layer,
                                         candidate->ffn_weights_[layer]);
    if (!status.ok()) return {status, {}};

    auto attention = create_deepseek_attention_state(
        kCompressionRatios[layer], config.max_context_tokens);
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
  if (actual_bytes != estimate.total_bytes) {
    return {{ErrorCode::internal,
             "DeepSeek request state estimate does not match allocations"},
            {}};
  }
  candidate->bytes_ = actual_bytes;
  return {Status::success(), std::move(candidate)};
}

}  // namespace expert::runtime::cuda
