#pragma once

#include "expert/runtime/active_expert_executor.hpp"
#include "expert/runtime/cuda/deepseek_request.hpp"
#include "expert/runtime/cuda/deepseek_verify.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/model_descriptor.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace expert::runtime::cuda {

struct DeepSeekDecodeRope final {
  const float* base_cosine{};
  const float* base_sine{};
  const float* compressed_cosine{};
  const float* compressed_sine{};
  const float* ratio_four_group_cosine{};
  const float* ratio_four_group_sine{};
  const float* ratio_128_group_cosine{};
  const float* ratio_128_group_sine{};
};

struct DeepSeekDecodeBegin final {
  const float* input_streams{};  // device [4, 4096]
  DeepSeekDecodeRope rope;
  std::uint32_t position{};
  std::uint32_t token_id{};
  // Half-open range. Zero selects the artifact-declared layer count; a strict subrange is
  // useful for pipeline ownership and independently qualified slices.
  std::uint32_t first_layer{};
  std::uint32_t layer_limit{};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::time_point::max()};
};

struct DeepSeekVerifyBegin final {
  std::array<DeepSeekDecodeRope, 2U> rope;
  std::array<std::uint32_t, 2U> positions{};
  std::array<std::uint32_t, 2U> token_ids{};
  std::uint32_t first_layer{};
  std::uint32_t layer_limit{};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::time_point::max()};
};

enum class DeepSeekDecodeProgress : std::uint8_t {
  pending_cuda,
  layer_complete,
  needs_experts,
  token_complete,
};

struct DeepSeekDecodeAdvanceResult final {
  Status status;
  DeepSeekDecodeProgress progress{DeepSeekDecodeProgress::layer_complete};
  std::uint32_t layer{};
  std::vector<std::uint32_t> missing_experts;
  // Ready entries are device-pinned across a cold-plan suspension. The
  // scheduler must convert them to cache leases before loading misses so an
  // eviction cannot wait on the controller's own directory pin.
  std::vector<std::uint32_t> ready_experts;
  // Exact routed selection excluding the invariant shared expert.
  std::vector<std::uint32_t> routed_experts;
  // Ordinary decode publishes one exact top-6 row. Pair verification publishes
  // two consecutive top-6 rows; duplicates across rows are valid and share one
  // directory pin transaction.
  std::uint32_t route_rows{1U};
  // The routed experts ran at an activation-only owner. The scheduler must
  // retain route census feedback, but must not attribute the route to or pin
  // pages in the primary device cache.
  bool active_expert_external{};
};

struct DeepSeekRouteTraceEntry final {
  std::uint32_t layer{};
  // The shared expert is invariant and intentionally excluded.
  std::array<std::uint32_t, 6U> routed_experts{};
};

struct DeepSeekCpuExpertPlacement final {
  std::uint32_t expert{};
  std::span<const std::byte> record_bytes;
  DeepSeekCompactSections sections;
};

// Artifact identity and the selected activation-only executor for the routed
// component. Routing, route weights, stable aggregation, shared experts and
// HCA remain owned by the primary DeepSeek provider.
struct DeepSeekActiveExpertConfig final {
  std::shared_ptr<IActiveExpertExecutor> executor;
  Sha256Digest model_content_hash{};
  RoutedExpertComponentDescriptor component;
  std::string input_abi;
  std::string output_abi;
};

struct DeepSeekDecodeControllerResult;

struct DeepSeekDecodeTelemetry final {
  std::uint64_t attention_route_submit_ns{};
  std::uint64_t directory_plan_ns{};
  std::uint64_t ffn_submit_ns{};
  std::uint64_t directory_release_ns{};
  std::uint64_t gpu_attention_route_plan_ns{};
  std::uint64_t gpu_ffn_release_ns{};
  std::uint64_t gpu_attention_ns{};
  std::uint64_t gpu_route_ns{};
  std::uint64_t gpu_directory_plan_ns{};
  std::uint64_t gpu_ffn_ns{};
  std::uint64_t gpu_directory_release_ns{};
  std::uint64_t gpu_attention_hca_pre_norm_ns{};
  std::uint64_t gpu_attention_projection_ns{};
  std::uint64_t gpu_sparse_attention_ns{};
  std::uint64_t gpu_attention_output_projection_ns{};
  std::uint64_t gpu_attention_hca_post_ns{};
  std::uint64_t gpu_ffn_routed_ns{};
  std::uint64_t gpu_ffn_aggregate_ns{};
  std::uint64_t gpu_ffn_shared_ns{};
  std::uint64_t gpu_ffn_merge_ns{};
  std::uint64_t gpu_ffn_hca_post_ns{};
};

// Executes one layer per advance() call. A cache miss returns control without
// recomputing attention/router; after the caller publishes the missing experts,
// advance() replans and resumes the same FFN. Independent directory pin tokens
// keep concurrent requests safe while they wait.
class DeepSeekDecodeController final {
 public:
  ~DeepSeekDecodeController();
  DeepSeekDecodeController(const DeepSeekDecodeController&) = delete;
  DeepSeekDecodeController& operator=(const DeepSeekDecodeController&) = delete;

  [[nodiscard]] Status begin(const DeepSeekDecodeBegin& launch) noexcept;
  [[nodiscard]] Status begin_verify_pair(
      const DeepSeekVerifyBegin& launch) noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult advance() noexcept;
  [[nodiscard]] Status wait_for_cuda() noexcept;
  [[nodiscard]] Status cancel() noexcept;
  // Configured once before begin(); the workspace is private to this request.
  [[nodiscard]] Status configure_hybrid(
      std::shared_ptr<cpu::DeepSeekPackedExecutor> executor,
      std::shared_ptr<DeepSeekFfnHybridWorkspace> workspace) noexcept;
  [[nodiscard]] Status configure_verify(
      std::shared_ptr<DeepSeekVerifyState> verify) noexcept;
  [[nodiscard]] Status configure_active_experts(
      DeepSeekActiveExpertConfig config) noexcept;
  // Profiling mode records and synchronizes CUDA events at the two existing
  // per-layer dependency boundaries. It is opt-in because the extra events
  // intentionally perturb production scheduling.
  [[nodiscard]] Status enable_gpu_phase_timing() noexcept;
  [[nodiscard]] bool hybrid_configured() const noexcept {
    return cpu_executor_ != nullptr && hybrid_workspace_ != nullptr;
  }
  // Called only after needs_experts. Spans remain owned by scheduler host
  // leases until the suspended layer completes or is cancelled.
  [[nodiscard]] Status stage_cpu_placements(
      std::span<const DeepSeekCpuExpertPlacement> placements) noexcept;

  [[nodiscard]] const float* output_streams() const noexcept {
    return complete_ ? request_->streams_a_ : nullptr;
  }
  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] DeepSeekDecodeTelemetry telemetry() const noexcept {
    return telemetry_;
  }
  [[nodiscard]] std::uint32_t current_layer() const noexcept {
    return current_layer_;
  }
  [[nodiscard]] std::span<const DeepSeekRouteTraceEntry> route_trace()
      const noexcept {
    return route_trace_;
  }

 private:
  friend DeepSeekDecodeControllerResult create_deepseek_decode_controller(
      std::shared_ptr<DeepSeekRequestState>,
      std::shared_ptr<CudaExpertDirectory>, void*) noexcept;
  friend DeepSeekDecodeControllerResult create_deepseek_verify_controller(
      std::shared_ptr<DeepSeekVerifyState>,
      std::shared_ptr<CudaExpertDirectory>, void*) noexcept;
  DeepSeekDecodeController(std::shared_ptr<DeepSeekRequestState> request,
                           std::shared_ptr<CudaExpertDirectory> directory,
                           std::shared_ptr<CudaDirectoryPlanWorkspace> workspace,
                           void* stream) noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult fail(Status status) noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult start_plan() noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult poll_plan() noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult execute_plan(
      DirectoryPlanResult plan) noexcept;
  [[nodiscard]] DeepSeekDecodeAdvanceResult execute_active_plan(
      std::span<const std::uint32_t> routed_experts) noexcept;
  void clear_cpu_placements() noexcept;

  std::shared_ptr<DeepSeekRequestState> request_;
  std::shared_ptr<DeepSeekVerifyState> verify_;
  std::shared_ptr<CudaExpertDirectory> directory_;
  std::shared_ptr<CudaDirectoryPlanWorkspace> directory_workspace_;
  void* stream_{};
  DeepSeekDecodeRope rope_{};
  std::array<DeepSeekDecodeRope, 2U> pair_rope_{};
  std::array<std::uint32_t, 2U> pair_positions_{};
  std::array<std::uint32_t, 2U> pair_token_ids_{};
  std::uint32_t position_{};
  std::uint32_t token_id_{};
  std::uint32_t current_layer_{};
  std::uint32_t layer_limit_{};
  std::uint64_t pin_id_{};
  std::vector<DeepSeekRouteTraceEntry> route_trace_;
  std::shared_ptr<cpu::DeepSeekPackedExecutor> cpu_executor_;
  std::shared_ptr<DeepSeekFfnHybridWorkspace> hybrid_workspace_;
  std::vector<DeepSeekCpuExpertPlacement> cpu_placements_;
  DeepSeekActiveExpertConfig active_expert_config_;
  std::shared_ptr<void> active_expert_host_owner_;
  float* active_expert_inputs_host_{};
  float* active_expert_outputs_host_{};
  std::uint64_t active_expert_request_id_{};
  std::uint64_t next_active_expert_invocation_{1U};
  std::chrono::steady_clock::time_point active_expert_deadline_{
      std::chrono::steady_clock::time_point::max()};
  DeepSeekDecodeTelemetry telemetry_;
  void* attention_start_event_{};
  void* attention_stop_event_{};
  void* route_stop_event_{};
  void* plan_stop_event_{};
  void* ffn_start_event_{};
  void* ffn_stop_event_{};
  void* release_stop_event_{};
  void* attention_hca_pre_norm_stop_event_{};
  void* attention_projection_stop_event_{};
  void* sparse_attention_stop_event_{};
  void* attention_output_projection_stop_event_{};
  void* ffn_routed_stop_event_{};
  void* ffn_aggregate_stop_event_{};
  void* ffn_shared_stop_event_{};
  void* ffn_merge_stop_event_{};
  std::chrono::steady_clock::time_point plan_started_{};
  bool active_{};
  bool planning_{};
  bool waiting_for_experts_{};
  bool complete_{};
  bool pair_mode_{};
};

struct DeepSeekDecodeControllerResult final {
  Status status;
  std::shared_ptr<DeepSeekDecodeController> controller;
};

[[nodiscard]] DeepSeekDecodeControllerResult create_deepseek_decode_controller(
    std::shared_ptr<DeepSeekRequestState> request,
    std::shared_ptr<CudaExpertDirectory> directory,
    void* stream = nullptr) noexcept;

[[nodiscard]] DeepSeekDecodeControllerResult create_deepseek_verify_controller(
    std::shared_ptr<DeepSeekVerifyState> verify,
    std::shared_ptr<CudaExpertDirectory> directory,
    void* stream = nullptr) noexcept;

}  // namespace expert::runtime::cuda
