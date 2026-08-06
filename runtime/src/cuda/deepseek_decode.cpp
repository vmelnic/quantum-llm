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
                      attention_output_projection_stop_event_})
    if (event)
      static_cast<void>(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
}

Status DeepSeekDecodeController::enable_gpu_phase_timing() noexcept {
  if (active_ || attention_start_event_ || attention_stop_event_ ||
      route_stop_event_ || plan_stop_event_ || ffn_start_event_ ||
      ffn_stop_event_ || release_stop_event_ ||
      attention_hca_pre_norm_stop_event_ ||
      attention_projection_stop_event_ || sparse_attention_stop_event_ ||
      attention_output_projection_stop_event_)
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek GPU phase timing configuration"};
  void** destinations[] = {
      &attention_start_event_, &attention_stop_event_, &route_stop_event_,
      &plan_stop_event_, &ffn_start_event_, &ffn_stop_event_,
      &release_stop_event_, &attention_hca_pre_norm_stop_event_,
      &attention_projection_stop_event_, &sparse_attention_stop_event_,
      &attention_output_projection_stop_event_};
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
  return {std::move(status), DeepSeekDecodeProgress::layer_complete,
          current_layer_, {}, {}, {}};
}

DeepSeekDecodeAdvanceResult
DeepSeekDecodeController::start_plan() noexcept {
  const auto view = request_->layer(current_layer_);
  plan_started_ = std::chrono::steady_clock::now();
  const auto status = directory_->begin_plan_async(
      *directory_workspace_, current_layer_,
      view.ffn_state->expert_indices(), view.ffn_state->selection_count(),
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
  const auto execute = cpu_groups.empty()
      ? deepseek_ffn_execute(
            {view.ffn_weights, view.ffn_state, directory_->device_entries(),
             request_->streams_b_, request_->streams_a_,
             directory_->experts_per_layer(), stream_})
      : deepseek_ffn_execute_hybrid(
            {view.ffn_weights, view.ffn_state, directory_->device_entries(),
             request_->streams_b_, request_->streams_a_,
             hybrid_workspace_.get(), cpu_executor_.get(), cpu_groups,
             directory_->experts_per_layer(), stream_});
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

}  // namespace expert::runtime::cuda
