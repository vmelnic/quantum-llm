#include "expert/runtime/cuda/deepseek_decode.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <set>
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
    std::shared_ptr<CudaExpertDirectory> directory,
    std::shared_ptr<CudaDirectoryPlanWorkspace> workspace,
    void* stream) noexcept
    : request_(std::move(request)), directory_(std::move(directory)),
      directory_workspace_(std::move(workspace)), stream_(stream) {}

DeepSeekDecodeController::~DeepSeekDecodeController() {
  static_cast<void>(cancel());
  for (auto* event : {attention_start_event_, attention_stop_event_,
                      route_stop_event_, plan_stop_event_, ffn_start_event_,
                      ffn_stop_event_, release_stop_event_,
                      attention_hca_pre_norm_stop_event_,
                      attention_projection_stop_event_,
                      sparse_attention_stop_event_,
                      attention_output_projection_stop_event_,
                      ffn_routed_stop_event_, ffn_aggregate_stop_event_,
                      ffn_shared_stop_event_, ffn_merge_stop_event_})
    if (event)
      static_cast<void>(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
}

Status DeepSeekDecodeController::enable_gpu_phase_timing() noexcept {
  if (active_ || attention_start_event_ || attention_stop_event_ ||
      route_stop_event_ || plan_stop_event_ || ffn_start_event_ ||
      ffn_stop_event_ || release_stop_event_ ||
      attention_hca_pre_norm_stop_event_ ||
      attention_projection_stop_event_ || sparse_attention_stop_event_ ||
      attention_output_projection_stop_event_ || ffn_routed_stop_event_ ||
      ffn_aggregate_stop_event_ || ffn_shared_stop_event_ ||
      ffn_merge_stop_event_)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek GPU phase timing configuration"};
  void** destinations[] = {
      &attention_start_event_, &attention_stop_event_, &route_stop_event_,
      &plan_stop_event_, &ffn_start_event_, &ffn_stop_event_,
      &release_stop_event_, &attention_hca_pre_norm_stop_event_,
      &attention_projection_stop_event_, &sparse_attention_stop_event_,
      &attention_output_projection_stop_event_, &ffn_routed_stop_event_,
      &ffn_aggregate_stop_event_, &ffn_shared_stop_event_,
      &ffn_merge_stop_event_};
  for (auto** destination : destinations) {
    cudaEvent_t event{};
    const auto status = cuda_status(
        cudaEventCreateWithFlags(&event, cudaEventDefault),
        "create DeepSeek phase event");
    if (!status.ok()) return status;
    *destination = event;
  }
  return Status::success();
}

Status DeepSeekDecodeController::configure_hybrid(
    std::shared_ptr<cpu::DeepSeekPackedExecutor> executor,
    std::shared_ptr<DeepSeekFfnHybridWorkspace> workspace) noexcept {
  if (active_ || !executor || !workspace)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek hybrid controller configuration"};
  cpu_executor_ = std::move(executor);
  hybrid_workspace_ = std::move(workspace);
  return Status::success();
}

Status DeepSeekDecodeController::configure_verify(
    std::shared_ptr<DeepSeekVerifyState> verify) noexcept {
  if (active_ || !verify || verify->primary_request().get() != request_.get())
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek verify controller configuration"};
  verify_ = std::move(verify);
  return Status::success();
}

Status DeepSeekDecodeController::stage_cpu_placements(
    std::span<const DeepSeekCpuExpertPlacement> placements) noexcept {
  if (!active_ || !waiting_for_experts_ || !cpu_executor_ ||
      !hybrid_workspace_ || placements.empty() || placements.size() > 6U)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek CPU placement staging"};
  std::set<std::uint32_t> unique;
  for (const auto& placement : placements) {
    if (placement.expert >= 256U || placement.record_bytes.empty() ||
        !unique.insert(placement.expert).second)
      return {ErrorCode::invalid_argument,
              "invalid or duplicate DeepSeek CPU placement"};
  }
  cpu_placements_.assign(placements.begin(), placements.end());
  return Status::success();
}

void DeepSeekDecodeController::clear_cpu_placements() noexcept {
  cpu_placements_.clear();
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
  route_trace_.clear();
  route_trace_.reserve(layer_limit_ - current_layer_);
  waiting_for_experts_ = false;
  planning_ = false;
  complete_ = false;
  pair_mode_ = false;
  active_ = true;
  return Status::success();
}

Status DeepSeekDecodeController::begin_verify_pair(
    const DeepSeekVerifyBegin& launch) noexcept {
  if (active_ || !verify_ || launch.positions[1] != launch.positions[0] + 1U ||
      launch.positions[1] >= request_->max_context_tokens() ||
      launch.first_layer >= launch.layer_limit ||
      launch.layer_limit > kDeepSeekLayers) {
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek pair verification begin"};
  }
  for (const auto& rope : launch.rope) {
    if (!rope.base_cosine || !rope.base_sine || !rope.compressed_cosine ||
        !rope.compressed_sine)
      return {ErrorCode::invalid_argument,
              "DeepSeek pair verification is missing RoPE"};
  }
  auto status = verify_->begin_transaction(launch.positions[1]);
  if (!status.ok()) return status;
  pair_rope_ = launch.rope;
  pair_positions_ = launch.positions;
  pair_token_ids_ = launch.token_ids;
  current_layer_ = launch.first_layer;
  layer_limit_ = launch.layer_limit;
  route_trace_.clear();
  route_trace_.reserve(2U * (layer_limit_ - current_layer_));
  waiting_for_experts_ = false;
  planning_ = false;
  complete_ = false;
  pair_mode_ = true;
  active_ = true;
  return Status::success();
}

DeepSeekDecodeAdvanceResult DeepSeekDecodeController::fail(
    Status status) noexcept {
  if (planning_) {
    const auto cancelled = directory_->cancel_plan_async(*directory_workspace_);
    planning_ = false;
    if (status.ok() && !cancelled.ok()) status = cancelled;
  }
  if (pin_id_ != 0U) {
    const auto release = directory_->release_pins(pin_id_, stream_);
    pin_id_ = 0U;
    if (status.ok() && !release.ok()) status = release;
  }
  active_ = false;
  waiting_for_experts_ = false;
  complete_ = false;
  clear_cpu_placements();
  if (pair_mode_ && verify_) {
    const auto aborted = verify_->abort_transaction(stream_);
    if (status.ok() && !aborted.ok()) status = aborted;
  }
  pair_mode_ = false;
  return {std::move(status), DeepSeekDecodeProgress::layer_complete,
          current_layer_, {}, {}, {}};
}

DeepSeekDecodeAdvanceResult
DeepSeekDecodeController::start_plan() noexcept {
  const auto view = request_->layer(current_layer_);
  plan_started_ = std::chrono::steady_clock::now();
  const auto status = directory_->begin_plan_async(
      *directory_workspace_, current_layer_,
      pair_mode_ ? verify_->ffn_workspace()->expert_indices()
                 : view.ffn_state->expert_indices(),
      pair_mode_ ? verify_->ffn_workspace()->selection_count()
                 : view.ffn_state->selection_count(),
      stream_, true);
  if (!status.ok()) return fail(status);
  planning_ = true;
  return {Status::success(), DeepSeekDecodeProgress::pending_cuda,
          current_layer_, {}, {}, {}};
}

DeepSeekDecodeAdvanceResult
DeepSeekDecodeController::poll_plan() noexcept {
  auto polled = directory_->poll_plan_async(*directory_workspace_);
  if (!polled.status.ok()) return fail(std::move(polled.status));
  if (!polled.complete) {
    return {Status::success(), DeepSeekDecodeProgress::pending_cuda,
            current_layer_, {}, {}, {}};
  }
  planning_ = false;
  telemetry_.directory_plan_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - plan_started_)
          .count());
  if (attention_start_event_) {
    auto status = cuda_status(
        cudaEventRecord(static_cast<cudaEvent_t>(plan_stop_event_),
                        static_cast<cudaStream_t>(stream_)),
        "record DeepSeek plan phase stop");
    if (!status.ok()) return fail(status);
    status = cuda_status(
        cudaEventSynchronize(static_cast<cudaEvent_t>(plan_stop_event_)),
        "synchronize DeepSeek attention/route/plan phases");
    if (!status.ok()) return fail(status);
    const auto measure = [&](void* start, void* stop, const char* name,
                             std::uint64_t& destination) -> Status {
      float milliseconds = 0.0F;
      const auto measured = cuda_status(
          cudaEventElapsedTime(&milliseconds, static_cast<cudaEvent_t>(start),
                               static_cast<cudaEvent_t>(stop)),
          name);
      if (measured.ok())
        destination += static_cast<std::uint64_t>(
            static_cast<double>(milliseconds) * 1'000'000.0);
      return measured;
    };
    status = measure(attention_start_event_, attention_stop_event_,
                     "measure DeepSeek attention phase",
                     telemetry_.gpu_attention_ns);
    if (!status.ok()) return fail(status);
    status = measure(attention_start_event_,
                     attention_hca_pre_norm_stop_event_,
                     "measure DeepSeek attention HCA pre/norm phase",
                     telemetry_.gpu_attention_hca_pre_norm_ns);
    if (!status.ok()) return fail(status);
    status = measure(attention_hca_pre_norm_stop_event_,
                     attention_projection_stop_event_,
                     "measure DeepSeek attention projection phase",
                     telemetry_.gpu_attention_projection_ns);
    if (!status.ok()) return fail(status);
    status = measure(attention_projection_stop_event_,
                     sparse_attention_stop_event_,
                     "measure DeepSeek sparse attention phase",
                     telemetry_.gpu_sparse_attention_ns);
    if (!status.ok()) return fail(status);
    status = measure(sparse_attention_stop_event_,
                     attention_output_projection_stop_event_,
                     "measure DeepSeek attention output projection phase",
                     telemetry_.gpu_attention_output_projection_ns);
    if (!status.ok()) return fail(status);
    status = measure(attention_output_projection_stop_event_,
                     attention_stop_event_,
                     "measure DeepSeek attention HCA post phase",
                     telemetry_.gpu_attention_hca_post_ns);
    if (!status.ok()) return fail(status);
    status = measure(attention_stop_event_, route_stop_event_,
                     "measure DeepSeek route phase", telemetry_.gpu_route_ns);
    if (!status.ok()) return fail(status);
    status = measure(route_stop_event_, plan_stop_event_,
                     "measure DeepSeek directory plan phase",
                     telemetry_.gpu_directory_plan_ns);
    if (!status.ok()) return fail(status);
    status = measure(attention_start_event_, plan_stop_event_,
                     "measure DeepSeek attention/route/plan phases",
                     telemetry_.gpu_attention_route_plan_ns);
    if (!status.ok()) return fail(status);
  }
  if (!polled.plan.status.ok()) return fail(std::move(polled.plan.status));
  return execute_plan(std::move(polled.plan));
}

DeepSeekDecodeAdvanceResult DeepSeekDecodeController::execute_plan(
    DirectoryPlanResult plan) noexcept {
  const auto view = request_->layer(current_layer_);
  if (pair_mode_) {
    if (plan.selected_experts.size() != 14U ||
        plan.selected_experts[6U] != 256U ||
        plan.selected_experts[13U] != 256U)
      return fail({ErrorCode::internal,
                   "DeepSeek directory returned an invalid pair selection"});
    if (route_trace_.empty() || route_trace_.back().layer != current_layer_) {
      for (const auto offset : {0U, 7U}) {
        DeepSeekRouteTraceEntry trace;
        trace.layer = current_layer_;
        std::copy_n(plan.selected_experts.begin() + offset,
                    trace.routed_experts.size(),
                    trace.routed_experts.begin());
        route_trace_.push_back(trace);
      }
    }
    std::vector<std::uint32_t> routed_experts;
    routed_experts.reserve(12U);
    routed_experts.insert(routed_experts.end(),
                          plan.selected_experts.begin(),
                          plan.selected_experts.begin() + 6U);
    routed_experts.insert(routed_experts.end(),
                          plan.selected_experts.begin() + 7U,
                          plan.selected_experts.begin() + 13U);
    pin_id_ = plan.pin_id;
    if (!plan.missing_experts.empty()) {
      waiting_for_experts_ = true;
      return {Status::success(), DeepSeekDecodeProgress::needs_experts,
              current_layer_, std::move(plan.missing_experts),
              std::move(plan.ready_experts), std::move(routed_experts), 2U};
    }
    if (pin_id_ == 0U)
      return fail({ErrorCode::internal,
                   "DeepSeek pair directory returned no execution pin"});
    const auto pair_view = verify_->layer(current_layer_);
    const auto ffn_started = std::chrono::steady_clock::now();
    const auto execute = deepseek_ffn_execute_pair({
        pair_view.ffn_weights, pair_view.ffn_states,
        verify_->ffn_workspace(), directory_->device_entries(),
        {request_->streams_b_, verify_->speculative_streams_b_},
        {request_->streams_a_, verify_->speculative_streams_a_},
        directory_->experts_per_layer(), stream_});
    telemetry_.ffn_submit_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - ffn_started)
            .count());
    if (!execute.ok()) return fail(execute);
    const auto release_started = std::chrono::steady_clock::now();
    const auto release = directory_->release_pins_async(pin_id_, stream_);
    telemetry_.directory_release_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - release_started)
            .count());
    pin_id_ = 0U;
    if (!release.ok()) return fail(release);
    const auto completed_layer = current_layer_++;
    waiting_for_experts_ = false;
    if (current_layer_ == layer_limit_) {
      active_ = false;
      complete_ = true;
      return {Status::success(), DeepSeekDecodeProgress::token_complete,
              completed_layer, {}, {}, std::move(routed_experts), 2U};
    }
    return {Status::success(), DeepSeekDecodeProgress::layer_complete,
            completed_layer, {}, {}, std::move(routed_experts), 2U};
  }
  if (plan.selected_experts.size() != 7U ||
      plan.selected_experts.back() != 256U) {
    return fail({ErrorCode::internal,
                 "DeepSeek directory returned an invalid routed selection"});
  }
  if (route_trace_.empty() || route_trace_.back().layer != current_layer_) {
    DeepSeekRouteTraceEntry trace;
    trace.layer = current_layer_;
    std::copy_n(plan.selected_experts.begin(), trace.routed_experts.size(),
                trace.routed_experts.begin());
    route_trace_.push_back(trace);
  }
  std::vector<std::uint32_t> routed_experts(
      plan.selected_experts.begin(), plan.selected_experts.begin() + 6U);
  pin_id_ = plan.pin_id;
  std::vector<std::uint32_t> uncovered_missing;
  std::vector<cpu::DeepSeekPackedWorkGroup> cpu_groups;
  uncovered_missing.reserve(plan.missing_experts.size());
  cpu_groups.reserve(cpu_placements_.size());
  for (const auto missing : plan.missing_experts) {
    const auto placement = std::find_if(
        cpu_placements_.begin(), cpu_placements_.end(),
        [missing](const auto& value) { return value.expert == missing; });
    if (placement == cpu_placements_.end()) {
      uncovered_missing.push_back(missing);
      continue;
    }
    const auto selection = std::find(routed_experts.begin(),
                                     routed_experts.end(), missing);
    if (selection == routed_experts.end())
      return fail({ErrorCode::internal,
                   "staged CPU expert is absent from the routed selection"});
    const auto slot = static_cast<std::uint32_t>(
        std::distance(routed_experts.begin(), selection));
    cpu_groups.push_back({placement->record_bytes, placement->sections,
                          4096U, 2048U, {slot},
                          {static_cast<std::uint32_t>(cpu_groups.size())}});
  }
  if (!uncovered_missing.empty()) {
    waiting_for_experts_ = true;
    return {Status::success(), DeepSeekDecodeProgress::needs_experts,
            current_layer_, std::move(uncovered_missing),
            std::move(plan.ready_experts), std::move(routed_experts)};
  }
  if (pin_id_ == 0U)
    return fail({ErrorCode::internal,
                 "DeepSeek directory returned no execution pin"});

  if (ffn_start_event_) {
    const auto status = cuda_status(
        cudaEventRecord(static_cast<cudaEvent_t>(ffn_start_event_),
                        static_cast<cudaStream_t>(stream_)),
        "record DeepSeek FFN phase start");
    if (!status.ok()) return fail(status);
  }
  const auto ffn_started = std::chrono::steady_clock::now();
  const DeepSeekFfnExecuteLaunch::ProfileEvents ffn_profile{
      ffn_routed_stop_event_, ffn_aggregate_stop_event_,
      ffn_shared_stop_event_, ffn_merge_stop_event_};
  const auto execute = cpu_groups.empty()
      ? deepseek_ffn_execute(
            {view.ffn_weights, view.ffn_state, directory_->device_entries(),
             request_->streams_b_, request_->streams_a_,
             directory_->experts_per_layer(), stream_, nullptr,
             ffn_start_event_ ? &ffn_profile : nullptr})
      : deepseek_ffn_execute_hybrid(
            {view.ffn_weights, view.ffn_state, directory_->device_entries(),
             request_->streams_b_, request_->streams_a_,
             hybrid_workspace_.get(), cpu_executor_.get(), cpu_groups,
             directory_->experts_per_layer(), stream_,
             ffn_start_event_ ? &ffn_profile : nullptr});
  telemetry_.ffn_submit_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - ffn_started)
          .count());
  if (!execute.ok()) return fail(execute);
  if (ffn_start_event_) {
    const auto status = cuda_status(
        cudaEventRecord(static_cast<cudaEvent_t>(ffn_stop_event_),
                        static_cast<cudaStream_t>(stream_)),
        "record DeepSeek FFN phase stop");
    if (!status.ok()) return fail(status);
  }
  const auto release_started = std::chrono::steady_clock::now();
  const auto release = directory_->release_pins_async(pin_id_, stream_);
  telemetry_.directory_release_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - release_started)
          .count());
  pin_id_ = 0U;
  if (!release.ok()) return fail(release);
  if (ffn_start_event_) {
    auto status = cuda_status(
        cudaEventRecord(static_cast<cudaEvent_t>(release_stop_event_),
                        static_cast<cudaStream_t>(stream_)),
        "record DeepSeek release phase stop");
    if (!status.ok()) return fail(status);
    status = cuda_status(
        cudaEventSynchronize(static_cast<cudaEvent_t>(release_stop_event_)),
        "synchronize DeepSeek FFN/release phases");
    if (!status.ok()) return fail(status);
    const auto measure = [&](void* start, void* stop, const char* name,
                             std::uint64_t& destination) -> Status {
      float milliseconds = 0.0F;
      const auto measured = cuda_status(
          cudaEventElapsedTime(&milliseconds, static_cast<cudaEvent_t>(start),
                               static_cast<cudaEvent_t>(stop)),
          name);
      if (measured.ok())
        destination += static_cast<std::uint64_t>(
            static_cast<double>(milliseconds) * 1'000'000.0);
      return measured;
    };
    status = measure(ffn_start_event_, ffn_stop_event_,
                     "measure DeepSeek FFN phase", telemetry_.gpu_ffn_ns);
    if (!status.ok()) return fail(status);
    status = measure(ffn_start_event_, ffn_routed_stop_event_,
                     "measure DeepSeek routed FFN phase",
                     telemetry_.gpu_ffn_routed_ns);
    if (!status.ok()) return fail(status);
    status = measure(ffn_routed_stop_event_, ffn_aggregate_stop_event_,
                     "measure DeepSeek routed aggregate phase",
                     telemetry_.gpu_ffn_aggregate_ns);
    if (!status.ok()) return fail(status);
    status = measure(ffn_aggregate_stop_event_, ffn_shared_stop_event_,
                     "measure DeepSeek shared FFN phase",
                     telemetry_.gpu_ffn_shared_ns);
    if (!status.ok()) return fail(status);
    status = measure(ffn_shared_stop_event_, ffn_merge_stop_event_,
                     "measure DeepSeek FFN merge phase",
                     telemetry_.gpu_ffn_merge_ns);
    if (!status.ok()) return fail(status);
    status = measure(ffn_merge_stop_event_, ffn_stop_event_,
                     "measure DeepSeek FFN HCA post phase",
                     telemetry_.gpu_ffn_hca_post_ns);
    if (!status.ok()) return fail(status);
    status = measure(ffn_stop_event_, release_stop_event_,
                     "measure DeepSeek release phase",
                     telemetry_.gpu_directory_release_ns);
    if (!status.ok()) return fail(status);
    status = measure(ffn_start_event_, release_stop_event_,
                     "measure DeepSeek FFN/release phases",
                     telemetry_.gpu_ffn_release_ns);
    if (!status.ok()) return fail(status);
  }

  const auto completed_layer = current_layer_++;
  waiting_for_experts_ = false;
  clear_cpu_placements();
  if (current_layer_ == layer_limit_) {
    active_ = false;
    complete_ = true;
    return {Status::success(), DeepSeekDecodeProgress::token_complete,
            completed_layer, {}, {}, std::move(routed_experts)};
  }
  return {Status::success(), DeepSeekDecodeProgress::layer_complete,
          completed_layer, {}, {}, std::move(routed_experts)};
}

DeepSeekDecodeAdvanceResult DeepSeekDecodeController::advance() noexcept {
  if (!active_) {
    return {{ErrorCode::invalid_argument,
             "DeepSeek decode controller is not active"},
            DeepSeekDecodeProgress::layer_complete, current_layer_, {}, {}, {}};
  }
  if (waiting_for_experts_) {
    const auto release = directory_->release_pins_async(pin_id_, stream_);
    pin_id_ = 0U;
    if (!release.ok()) return fail(release);
    // The replacement plan is now the sole in-flight state. Keep any staged
    // CPU placements, but do not re-enter this release branch while its CUDA
    // event is pending.
    waiting_for_experts_ = false;
    return start_plan();
  }
  if (planning_) return poll_plan();

  if (pair_mode_) {
    const auto view = verify_->layer(current_layer_);
    const auto group_rope = [&](std::uint32_t row,
                                const float*& cosine,
                                const float*& sine) -> Status {
      cosine = nullptr;
      sine = nullptr;
      const bool emits = view.compress_ratio != 0U &&
          (pair_positions_[row] + 1U) % view.compress_ratio == 0U;
      if (!emits) return Status::success();
      if (view.compress_ratio == 4U) {
        cosine = pair_rope_[row].ratio_four_group_cosine;
        sine = pair_rope_[row].ratio_four_group_sine;
      } else {
        cosine = pair_rope_[row].ratio_128_group_cosine;
        sine = pair_rope_[row].ratio_128_group_sine;
      }
      return cosine && sine
          ? Status::success()
          : Status(ErrorCode::invalid_argument,
                   "DeepSeek pair verification is missing group RoPE");
    };
    const auto attention_route_started = std::chrono::steady_clock::now();
    const float* group_cosine[2]{};
    const float* group_sine[2]{};
    auto status = group_rope(0U, group_cosine[0], group_sine[0]);
    if (!status.ok()) return fail(status);
    status = group_rope(1U, group_cosine[1], group_sine[1]);
    if (!status.ok()) return fail(status);
    DeepSeekAttentionPairLaunch attention_pair{};
    attention_pair.weights = view.attention_weights;
    attention_pair.state = view.attention_state;
    attention_pair.workspace = verify_->attention_workspace();
    attention_pair.streams[0] = request_->streams_a_;
    attention_pair.streams[1] = verify_->speculative_streams_a_;
    attention_pair.updated_streams[0] = request_->streams_b_;
    attention_pair.updated_streams[1] = verify_->speculative_streams_b_;
    for (std::uint32_t row = 0U; row < 2U; ++row) {
      attention_pair.cosine[row] = view.compress_ratio
          ? pair_rope_[row].compressed_cosine
          : pair_rope_[row].base_cosine;
      attention_pair.sine[row] = view.compress_ratio
          ? pair_rope_[row].compressed_sine
          : pair_rope_[row].base_sine;
      attention_pair.compressed_cosine[row] = group_cosine[row];
      attention_pair.compressed_sine[row] = group_sine[row];
      attention_pair.positions[row] = pair_positions_[row];
    }
    attention_pair.speculative_checkpoint = view.rollback_checkpoint;
    attention_pair.epsilon = 1e-6F;
    attention_pair.sinkhorn_iterations = 20U;
    attention_pair.stream = stream_;
    status = deepseek_attention_decode_pair(attention_pair);
    if (!status.ok()) return fail(status);
    status = verify_->mark_layer_checkpointed(current_layer_);
    if (!status.ok()) return fail(status);
    status = deepseek_ffn_route({
        view.ffn_weights, view.ffn_states[0], request_->streams_b_,
        pair_token_ids_[0], 1e-6F, 20U, stream_});
    if (!status.ok()) return fail(status);
    status = deepseek_ffn_route({
        view.ffn_weights, view.ffn_states[1],
        verify_->speculative_streams_b_, pair_token_ids_[1], 1e-6F, 20U,
        stream_});
    if (!status.ok()) return fail(status);
    status = deepseek_ffn_gather_pair_routes(
        {{view.ffn_states[0], view.ffn_states[1]},
         verify_->ffn_workspace(), stream_});
    if (!status.ok()) return fail(status);
    telemetry_.attention_route_submit_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - attention_route_started)
            .count());
    return start_plan();
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
  const auto attention_route_started = std::chrono::steady_clock::now();
  if (attention_start_event_) {
    const auto status = cuda_status(
        cudaEventRecord(static_cast<cudaEvent_t>(attention_start_event_),
                        static_cast<cudaStream_t>(stream_)),
        "record DeepSeek attention phase start");
    if (!status.ok()) return fail(status);
  }
  const DeepSeekAttentionLaunch::ProfileEvents attention_profile{
      attention_hca_pre_norm_stop_event_, attention_projection_stop_event_,
      sparse_attention_stop_event_, attention_output_projection_stop_event_};
  const auto attention = deepseek_attention_decode(
      {view.attention_weights, view.attention_state, request_->streams_a_,
       request_->streams_b_,
       compressed ? rope_.compressed_cosine : rope_.base_cosine,
       compressed ? rope_.compressed_sine : rope_.base_sine,
       group_cosine, group_sine, position_, 1e-6F, 20U, stream_,
       attention_start_event_ ? &attention_profile : nullptr});
  if (!attention.ok()) return fail(attention);
  if (attention_start_event_) {
    const auto status = cuda_status(
        cudaEventRecord(static_cast<cudaEvent_t>(attention_stop_event_),
                        static_cast<cudaStream_t>(stream_)),
        "record DeepSeek attention phase stop");
    if (!status.ok()) return fail(status);
  }
  const auto route = deepseek_ffn_route(
      {view.ffn_weights, view.ffn_state, request_->streams_b_, token_id_,
       1e-6F, 20U, stream_});
  if (!route.ok()) return fail(route);
  if (attention_start_event_) {
    const auto status = cuda_status(
        cudaEventRecord(static_cast<cudaEvent_t>(route_stop_event_),
                        static_cast<cudaStream_t>(stream_)),
        "record DeepSeek route phase stop");
    if (!status.ok()) return fail(status);
  }
  telemetry_.attention_route_submit_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - attention_route_started)
          .count());
  return start_plan();
}

Status DeepSeekDecodeController::wait_for_cuda() noexcept {
  if (!active_ || !planning_)
    return {ErrorCode::invalid_argument,
            "DeepSeek decode has no pending CUDA plan"};
  return directory_->wait_plan_async(*directory_workspace_);
}

Status DeepSeekDecodeController::cancel() noexcept {
  auto status = Status::success();
  if (planning_) {
    status = directory_->cancel_plan_async(*directory_workspace_);
    planning_ = false;
  }
  if (pin_id_ != 0U) {
    const auto released = directory_->release_pins(pin_id_, stream_);
    if (status.ok()) status = released;
    pin_id_ = 0U;
  }
  active_ = false;
  waiting_for_experts_ = false;
  complete_ = false;
  clear_cpu_placements();
  if (pair_mode_ && verify_) {
    const auto aborted = verify_->abort_transaction(stream_);
    if (status.ok()) status = aborted;
  }
  pair_mode_ = false;
  return status;
}

DeepSeekDecodeControllerResult create_deepseek_decode_controller(
    std::shared_ptr<DeepSeekRequestState> request,
    std::shared_ptr<CudaExpertDirectory> directory, void* stream) noexcept {
  if (!request || !directory || directory->experts_per_layer() != 257U) {
    return {{ErrorCode::invalid_argument,
             "invalid DeepSeek decode controller dependencies"}, {}};
  }
  auto workspace = directory->create_plan_workspace();
  if (!workspace.status.ok() || !workspace.workspace)
    return {workspace.status.ok()
                ? Status(ErrorCode::internal,
                         "CUDA directory returned no planning workspace")
                : std::move(workspace.status),
            {}};
  return {Status::success(), std::shared_ptr<DeepSeekDecodeController>(
      new DeepSeekDecodeController(std::move(request), std::move(directory),
                                   std::move(workspace.workspace), stream))};
}

DeepSeekDecodeControllerResult create_deepseek_verify_controller(
    std::shared_ptr<DeepSeekVerifyState> verify,
    std::shared_ptr<CudaExpertDirectory> directory, void* stream) noexcept {
  if (!verify || !verify->primary_request() || !directory ||
      directory->experts_per_layer() != 257U)
    return {{ErrorCode::invalid_argument,
             "invalid DeepSeek verify controller dependencies"}, {}};
  auto workspace = directory->create_plan_workspace();
  if (!workspace.status.ok() || !workspace.workspace)
    return {workspace.status.ok()
                ? Status(ErrorCode::internal,
                         "CUDA directory returned no pair planning workspace")
                : std::move(workspace.status),
            {}};
  auto controller = std::shared_ptr<DeepSeekDecodeController>(
      new DeepSeekDecodeController(verify->primary_request(),
                                   std::move(directory),
                                   std::move(workspace.workspace), stream));
  controller->verify_ = std::move(verify);
  return {Status::success(), std::move(controller)};
}

}  // namespace expert::runtime::cuda
