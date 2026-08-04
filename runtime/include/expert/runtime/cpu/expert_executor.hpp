#pragma once

#include "expert/runtime/expert_record.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace expert::runtime::cpu {

// One immutable expert and every (row, top-k slot) selection that routed to it.
// record_bytes remains owned by HostExpertLease in the caller until execute()
// returns.
struct ExpertWorkGroup final {
  std::span<const std::byte> record_bytes;
  ExpertSections sections;
  std::vector<std::uint32_t> selections;   // global row * top_k + slot
  std::vector<std::uint32_t> output_slots; // compact output row per selection
};

struct ExpertExecutorConfig final {
  std::uint32_t maximum_threads{};
  bool enable_autotune{true};
  std::uint32_t calibration_budget_ms{250};
};

struct ExpertExecutorTelemetry final {
  std::uint32_t maximum_threads{};
  std::uint32_t selected_threads{};
  std::uint32_t gate_chunk{};
  std::uint32_t down_chunk{};
  std::uint64_t calibration_runs{};
  std::uint64_t calibration_ns{};
  std::uint64_t execute_calls{};
  std::uint64_t selections{};
  std::uint64_t effective_weight_bytes{};
  std::uint64_t compute_ns{};
};

class ExpertExecutor final {
 public:
  explicit ExpertExecutor(std::uint32_t thread_count);
  explicit ExpertExecutor(ExpertExecutorConfig config);
  ~ExpertExecutor();
  ExpertExecutor(const ExpertExecutor&) = delete;
  ExpertExecutor& operator=(const ExpertExecutor&) = delete;

  // inputs is [rows, hidden]. selection_outputs is
  // [compact_output_count, hidden]. Each group maps its global selections to
  // distinct compact output_slots.
  [[nodiscard]] Status execute(std::span<const ExpertWorkGroup> groups,
                               std::span<const float> inputs,
                               std::uint32_t rows, std::uint32_t top_k,
                               std::span<float> selection_outputs);
  [[nodiscard]] ExpertExecutorTelemetry telemetry() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace expert::runtime::cpu
