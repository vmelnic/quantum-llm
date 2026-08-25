#pragma once

#include "expert/runtime/expert_record.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace expert::runtime::cpu {

// One immutable FP4 block-32 SwiGLU expert and every microbatch selection
// assigned to the host lane. Source layout is described by the validated
// artifact sections; no model family participates in this execution ABI.
struct Fp4HostWorkGroup final {
  std::span<const std::byte> record_bytes;
  SplitExpertSections sections;
  std::uint32_t hidden{};
  std::uint32_t intermediate{};
  std::vector<std::uint32_t> selections;
  std::vector<std::uint32_t> output_slots;
};

struct Fp4HostExecutorConfig final {
  std::uint32_t maximum_threads{};
  std::uint32_t gate_chunk{8U};
  std::uint32_t down_chunk{8U};
  float swiglu_limit{10.0F};
  bool bf16_intermediate{true};
  bool pin_windows_threads{true};
};

struct Fp4HostExecutorTelemetry final {
  std::uint32_t maximum_threads{};
  std::uint32_t workers_used_last{};
  std::uint64_t worker_mask_last{};
  std::uint64_t execute_calls{};
  std::uint64_t selections{};
  std::uint64_t source_weight_bytes{};
  std::uint64_t compute_ns{};
};

// Persistent all-core executor for FP4-E2M1/UE8M0 block-32 payloads. It never
// expands an expert into an INT8 mirror: inputs and intermediates are quantized
// to Q8 once, while workers consume the published FP4 sections directly.
class Fp4HostExecutor final {
 public:
  explicit Fp4HostExecutor(Fp4HostExecutorConfig config);
  ~Fp4HostExecutor();
  Fp4HostExecutor(const Fp4HostExecutor&) = delete;
  Fp4HostExecutor& operator=(const Fp4HostExecutor&) = delete;

  [[nodiscard]] Status execute(
      std::span<const Fp4HostWorkGroup> groups,
      std::span<const float> inputs, std::uint32_t rows, std::uint32_t top_k,
      std::span<float> selection_outputs);
  [[nodiscard]] Fp4HostExecutorTelemetry telemetry() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace expert::runtime::cpu
