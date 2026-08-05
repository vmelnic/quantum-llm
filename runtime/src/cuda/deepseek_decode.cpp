#include "expert/runtime/cuda/deepseek_decode.hpp"

#include <cuda_runtime_api.h>

#include <string>
#include <utility>

namespace expert::runtime::cuda {
namespace {

constexpr std::size_t kStreamValues = 4U * 4096U;

Status cuda_status(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return Status::success();
  return {ErrorCode::internal,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

}  // namespace

DeepSeekDecodeController::DeepSeekDecodeController(
    std::shared_ptr<DeepSeekRequestState> request,
    std::shared_ptr<CudaExpertDirectory> directory, void* stream) noexcept
    : request_(std::move(request)), directory_(std::move(directory)),
      stream_(stream) {}

DeepSeekDecodeController::~DeepSeekDecodeController() {
  static_cast<void>(cancel());
}

Status DeepSeekDecodeController::begin(
    const DeepSeekDecodeBegin& launch) noexcept {
  if (active_ || !launch.input_streams ||
      launch.position >= request_->max_context_tokens() ||
      launch.first_layer >= launch.layer_limit ||
      launch.layer_limit > kDeepSeekLayers || !launch.rope.base_cosine ||
      !launch.rope.base_sine || !launch.rope.compressed_cosine ||
      !launch.rope.compressed_sine) {
    return {ErrorCode::invalid_argument, "invalid DeepSeek decode begin"};
  }
  const auto stream = static_cast<cudaStream_t>(stream_);
  if (launch.input_streams != request_->streams_a_) {
    const auto status = cuda_status(
        cudaMemcpyAsync(request_->streams_a_, launch.input_streams,
                        kStreamValues * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream),
        "copy DeepSeek input streams");
    if (!status.ok()) return status;
  }
  rope_ = launch.rope;
  position_ = launch.position;
  token_id_ = launch.token_id;
  current_layer_ = launch.first_layer;
  layer_limit_ = launch.layer_limit;
  waiting_for_experts_ = false;
  complete_ = false;
  active_ = true;
  return Status::success();
}

DeepSeekDecodeAdvanceResult DeepSeekDecodeController::fail(
    Status status) noexcept {
  if (pin_id_ != 0U) {
    const auto release = directory_->release_pins(pin_id_, stream_);
    pin_id_ = 0U;
    if (status.ok() && !release.ok()) status = release;
  }
  active_ = false;
  waiting_for_experts_ = false;
  complete_ = false;
  return {std::move(status), DeepSeekDecodeProgress::layer_complete,
          current_layer_, {}};
}

DeepSeekDecodeAdvanceResult
DeepSeekDecodeController::plan_and_execute() noexcept {
  const auto view = request_->layer(current_layer_);
  auto plan = directory_->pin_or_collect_misses(
      current_layer_, view.ffn_state->expert_indices(),
      view.ffn_state->selection_count(), stream_, true);
  if (!plan.status.ok()) return fail(std::move(plan.status));
  pin_id_ = plan.pin_id;
  if (!plan.missing_experts.empty()) {
    waiting_for_experts_ = true;
    return {Status::success(), DeepSeekDecodeProgress::needs_experts,
            current_layer_, std::move(plan.missing_experts)};
  }
  if (pin_id_ == 0U)
    return fail({ErrorCode::internal,
                 "DeepSeek directory returned no execution pin"});

  const auto execute = deepseek_ffn_execute(
      {view.ffn_weights, view.ffn_state, directory_->device_entries(),
       request_->streams_b_, request_->streams_a_,
       directory_->experts_per_layer(), stream_});
  if (!execute.ok()) return fail(execute);
  const auto release = directory_->release_pins(pin_id_, stream_);
  pin_id_ = 0U;
  if (!release.ok()) return fail(release);

  const auto completed_layer = current_layer_++;
  waiting_for_experts_ = false;
  if (current_layer_ == layer_limit_) {
    active_ = false;
    complete_ = true;
    return {Status::success(), DeepSeekDecodeProgress::token_complete,
            completed_layer, {}};
  }
  return {Status::success(), DeepSeekDecodeProgress::layer_complete,
          completed_layer, {}};
}

DeepSeekDecodeAdvanceResult DeepSeekDecodeController::advance() noexcept {
  if (!active_) {
    return {{ErrorCode::invalid_argument,
             "DeepSeek decode controller is not active"},
            DeepSeekDecodeProgress::layer_complete, current_layer_, {}};
  }
  if (waiting_for_experts_) {
    const auto release = directory_->release_pins(pin_id_, stream_);
    pin_id_ = 0U;
    if (!release.ok()) return fail(release);
    return plan_and_execute();
  }

  const auto view = request_->layer(current_layer_);
  const bool compressed = view.compress_ratio != 0U;
  const bool emits = compressed &&
      (position_ + 1U) % view.compress_ratio == 0U;
  const float* group_cosine = nullptr;
  const float* group_sine = nullptr;
  if (emits && view.compress_ratio == 4U) {
    group_cosine = rope_.ratio_four_group_cosine;
    group_sine = rope_.ratio_four_group_sine;
  } else if (emits) {
    group_cosine = rope_.ratio_128_group_cosine;
    group_sine = rope_.ratio_128_group_sine;
  }
  if (emits && (!group_cosine || !group_sine)) {
    return fail({ErrorCode::invalid_argument,
                 "DeepSeek decode is missing group-start RoPE"});
  }
  const auto attention = deepseek_attention_decode(
      {view.attention_weights, view.attention_state, request_->streams_a_,
       request_->streams_b_,
       compressed ? rope_.compressed_cosine : rope_.base_cosine,
       compressed ? rope_.compressed_sine : rope_.base_sine,
       group_cosine, group_sine, position_, 1e-6F, 20U, stream_});
  if (!attention.ok()) return fail(attention);
  const auto route = deepseek_ffn_route(
      {view.ffn_weights, view.ffn_state, request_->streams_b_, token_id_,
       1e-6F, 20U, stream_});
  if (!route.ok()) return fail(route);
  return plan_and_execute();
}

Status DeepSeekDecodeController::cancel() noexcept {
  auto status = Status::success();
  if (pin_id_ != 0U) {
    status = directory_->release_pins(pin_id_, stream_);
    pin_id_ = 0U;
  }
  active_ = false;
  waiting_for_experts_ = false;
  complete_ = false;
  return status;
}

DeepSeekDecodeControllerResult create_deepseek_decode_controller(
    std::shared_ptr<DeepSeekRequestState> request,
    std::shared_ptr<CudaExpertDirectory> directory, void* stream) noexcept {
  if (!request || !directory || directory->experts_per_layer() != 257U) {
    return {{ErrorCode::invalid_argument,
             "invalid DeepSeek decode controller dependencies"}, {}};
  }
  return {Status::success(), std::shared_ptr<DeepSeekDecodeController>(
      new DeepSeekDecodeController(std::move(request), std::move(directory),
                                   stream))};
}

}  // namespace expert::runtime::cuda
