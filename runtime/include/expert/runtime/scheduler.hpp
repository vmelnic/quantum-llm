#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace expert::runtime {

using RequestId = std::uint64_t;
using WorkId = std::uint64_t;

enum class ExpertResidency : std::uint8_t {
  absent,
  ram_ready,
  vram_ready,
  failed,
};

struct SchedulerConfig final {
  std::size_t maximum_requests{};
  std::size_t maximum_queued_tokens{};
  std::size_t maximum_batch_items{};
};

struct RoutedExpert final {
  std::uint32_t expert{};
  float routing_weight{};
};

struct TokenWork final {
  RequestId request{};
  std::uint64_t sequence{};
  std::uint32_t row{};
  std::uint32_t layer{};
  std::uint64_t deadline_tick{};
  std::span<const RoutedExpert> experts;
};

struct ScheduledItem final {
  WorkId work{};
  RequestId request{};
  std::uint64_t sequence{};
  std::uint32_t row{};
  std::uint32_t layer{};
  std::uint32_t expert{};
  float routing_weight{};
};

struct ExpertGroup final {
  std::uint32_t layer{};
  std::uint32_t expert{};
  std::vector<ScheduledItem> items;
};

struct SchedulerBatch final {
  std::vector<ExpertGroup> groups;
  std::size_t ready_items{};
  std::size_t blocked_items{};
  std::size_t failed_items{};
  std::size_t unique_experts{};
};

struct SchedulerSnapshot final {
  std::size_t active_requests{};
  std::size_t queued_tokens{};
  std::size_t pending_items{};
  std::uint64_t admitted_requests{};
  std::uint64_t rejected_requests{};
  std::uint64_t cancelled_requests{};
  std::uint64_t completed_tokens{};
  std::uint64_t scheduled_items{};
  std::uint64_t reused_items{};
};

using ResidencyQuery =
    std::function<ExpertResidency(std::uint32_t layer, std::uint32_t expert)>;

class ContinuousBatchScheduler final {
 public:
  explicit ContinuousBatchScheduler(SchedulerConfig config);
  ~ContinuousBatchScheduler();
  ContinuousBatchScheduler(const ContinuousBatchScheduler&) = delete;
  ContinuousBatchScheduler& operator=(const ContinuousBatchScheduler&) = delete;

  [[nodiscard]] Status admit(RequestId request);
  [[nodiscard]] Status enqueue(const TokenWork& token);
  [[nodiscard]] SchedulerBatch schedule(const ResidencyQuery& residency);
  [[nodiscard]] Status complete(std::span<const WorkId> work);
  [[nodiscard]] bool token_complete(RequestId request,
                                    std::uint64_t sequence) const;
  void retire_token(RequestId request, std::uint64_t sequence);
  void cancel(RequestId request) noexcept;
  void finish(RequestId request) noexcept;
  [[nodiscard]] SchedulerSnapshot snapshot() const noexcept;

 private:
  struct Core;
  std::unique_ptr<Core> core_;
};

}  // namespace expert::runtime
