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
  std::vector<std::uint32_t> selections;
};

class ExpertExecutor final {
 public:
  explicit ExpertExecutor(std::uint32_t thread_count);
  ~ExpertExecutor();
  ExpertExecutor(const ExpertExecutor&) = delete;
  ExpertExecutor& operator=(const ExpertExecutor&) = delete;

  // inputs is [rows, hidden]. selection_outputs is
  // [rows * top_k, hidden]. Only slots named by groups are written.
  [[nodiscard]] Status execute(std::span<const ExpertWorkGroup> groups,
                               std::span<const float> inputs,
                               std::uint32_t rows, std::uint32_t top_k,
                               std::span<float> selection_outputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace expert::runtime::cpu
