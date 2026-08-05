#pragma once

#include "expert/runtime/expert_record.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace expert::runtime::cpu {

// One compact DeepSeek expert and every microbatch selection assigned to the
// CPU lane. The caller keeps the immutable host lease alive through execute().
struct DeepSeekPackedWorkGroup final {
  std::span<const std::byte> record_bytes;
  DeepSeekCompactSections sections;
  std::uint32_t hidden{};
  std::uint32_t intermediate{};
  std::vector<std::uint32_t> selections;
  std::vector<std::uint32_t> output_slots;
};

struct DeepSeekPackedExecutorConfig final {
  std::uint32_t maximum_threads{};
  std::uint32_t gate_chunk{8U};
  std::uint32_t down_chunk{8U};
  float swiglu_limit{10.0F};
  bool bf16_intermediate{true};
  bool pin_windows_threads{true};
};

struct DeepSeekPackedExecutorTelemetry final {
  std::uint32_t maximum_threads{};
  std::uint32_t workers_used_last{};
  std::uint64_t worker_mask_last{};
  std::uint64_t execute_calls{};
  std::uint64_t selections{};
  std::uint64_t source_weight_bytes{};
  std::uint64_t compute_ns{};
};

// Persistent all-core executor for the native E2M1/UE8M0 payload. It never
// expands an expert into an INT8 mirror: inputs/intermediates are quantized to
// Q8 once and every worker consumes the packed FP4 rows directly.
class DeepSeekPackedExecutor final {
 public:
  explicit DeepSeekPackedExecutor(DeepSeekPackedExecutorConfig config);
  ~DeepSeekPackedExecutor();
  DeepSeekPackedExecutor(const DeepSeekPackedExecutor&) = delete;
  DeepSeekPackedExecutor& operator=(const DeepSeekPackedExecutor&) = delete;

  [[nodiscard]] Status execute(
      std::span<const DeepSeekPackedWorkGroup> groups,
      std::span<const float> inputs, std::uint32_t rows, std::uint32_t top_k,
      std::span<float> selection_outputs);
  [[nodiscard]] DeepSeekPackedExecutorTelemetry telemetry() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace expert::runtime::cpu
