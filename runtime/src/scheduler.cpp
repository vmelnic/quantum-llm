#include "expert/runtime/scheduler.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace expert::runtime {
namespace {

struct TokenKey final {
  RequestId request{};
  std::uint64_t sequence{};
  auto operator<=>(const TokenKey&) const = default;
};

struct Item final {
  ScheduledItem value;
  std::uint64_t deadline{};
  std::uint64_t arrival{};
  bool in_flight{};
  bool complete{};
};

struct Token final {
  TokenKey key;
  std::uint64_t arrival{};
  std::vector<WorkId> work;
};

}  // namespace

struct ContinuousBatchScheduler::Core final {
  explicit Core(SchedulerConfig value) : config(value) {}
  SchedulerConfig config;
  std::unordered_set<RequestId> requests;
  std::map<TokenKey, Token> tokens;
  std::map<WorkId, Item> items;
  WorkId next_work{1};
  std::uint64_t clock{};
  SchedulerSnapshot metrics;
};

ContinuousBatchScheduler::ContinuousBatchScheduler(SchedulerConfig config)
    : core_(std::make_unique<Core>(config)) {
  if (config.maximum_requests == 0 || config.maximum_queued_tokens == 0 ||
      config.maximum_batch_items == 0) {
    throw std::invalid_argument("scheduler limits must be positive");
  }
}

ContinuousBatchScheduler::~ContinuousBatchScheduler() = default;

Status ContinuousBatchScheduler::admit(RequestId request) {
  if (request == 0) return Status(ErrorCode::invalid_argument, "request id zero");
  if (core_->requests.contains(request))
    return Status(ErrorCode::invalid_argument, "duplicate request id");
  if (core_->requests.size() >= core_->config.maximum_requests) {
    ++core_->metrics.rejected_requests;
    return Status(ErrorCode::backpressure, "request capacity exhausted");
  }
  core_->requests.insert(request);
  ++core_->metrics.admitted_requests;
  core_->metrics.active_requests = core_->requests.size();
  return Status::success();
}

Status ContinuousBatchScheduler::enqueue(const TokenWork& token) {
  if (!core_->requests.contains(token.request) || token.experts.empty())
    return Status(ErrorCode::invalid_argument, "unknown request or empty route");
  if (core_->tokens.size() >= core_->config.maximum_queued_tokens)
    return Status(ErrorCode::backpressure, "token queue capacity exhausted");
  const TokenKey key{token.request, token.sequence};
  if (core_->tokens.contains(key))
    return Status(ErrorCode::invalid_argument, "duplicate token sequence");
  Token stored{key, ++core_->clock, {}};
  stored.work.reserve(token.experts.size());
  for (const auto& route : token.experts) {
    const auto id = core_->next_work++;
    ScheduledItem scheduled{id, token.request, token.sequence, token.row,
                            token.layer, route.expert, route.routing_weight};
    core_->items.emplace(id, Item{scheduled, token.deadline_tick, stored.arrival});
    stored.work.push_back(id);
  }
  core_->tokens.emplace(key, std::move(stored));
  core_->metrics.queued_tokens = core_->tokens.size();
  core_->metrics.pending_items = core_->items.size();
  return Status::success();
}

SchedulerBatch ContinuousBatchScheduler::schedule(const ResidencyQuery& residency) {
  SchedulerBatch result;
  if (!residency) return result;
  std::vector<Item*> ready;
  for (auto& [id, item] : core_->items) {
    (void)id;
    if (item.complete || item.in_flight) continue;
    switch (residency(item.value.layer, item.value.expert)) {
      case ExpertResidency::vram_ready: ready.push_back(&item); break;
      case ExpertResidency::failed: ++result.failed_items; break;
      case ExpertResidency::ram_ready:
      case ExpertResidency::absent: ++result.blocked_items; break;
    }
  }
  std::sort(ready.begin(), ready.end(), [](const Item* left, const Item* right) {
    const auto left_deadline = left->deadline == 0 ? UINT64_MAX : left->deadline;
    const auto right_deadline = right->deadline == 0 ? UINT64_MAX : right->deadline;
    return std::tie(left_deadline, left->arrival, left->value.request,
                    left->value.sequence, left->value.expert) <
           std::tie(right_deadline, right->arrival, right->value.request,
                    right->value.sequence, right->value.expert);
  });
  if (ready.size() > core_->config.maximum_batch_items)
    ready.resize(core_->config.maximum_batch_items);
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<ScheduledItem>> grouped;
  for (auto* item : ready) {
    item->in_flight = true;
    grouped[{item->value.layer, item->value.expert}].push_back(item->value);
  }
  for (auto& [key, items] : grouped) {
    result.groups.push_back({key.first, key.second, std::move(items)});
  }
  result.ready_items = ready.size();
  result.unique_experts = result.groups.size();
  core_->metrics.scheduled_items += result.ready_items;
  core_->metrics.reused_items += result.ready_items - result.unique_experts;
  return result;
}

Status ContinuousBatchScheduler::complete(std::span<const WorkId> work) {
  std::set<WorkId> unique;
  for (const auto id : work) {
    if (!unique.insert(id).second)
      return Status(ErrorCode::invalid_argument, "duplicate completion id");
    const auto iterator = core_->items.find(id);
    if (iterator == core_->items.end() || !iterator->second.in_flight || iterator->second.complete)
      return Status(ErrorCode::invalid_argument, "completion is not in flight");
  }
  for (const auto id : work) {
    auto& item = core_->items.at(id);
    item.in_flight = false;
    item.complete = true;
  }
  return Status::success();
}

bool ContinuousBatchScheduler::token_complete(RequestId request,
                                               std::uint64_t sequence) const {
  const auto iterator = core_->tokens.find({request, sequence});
  if (iterator == core_->tokens.end()) return false;
  return std::all_of(iterator->second.work.begin(), iterator->second.work.end(),
                     [&](WorkId id) { return core_->items.at(id).complete; });
}

void ContinuousBatchScheduler::retire_token(RequestId request,
                                             std::uint64_t sequence) {
  const auto iterator = core_->tokens.find({request, sequence});
  if (iterator == core_->tokens.end() || !token_complete(request, sequence))
    throw std::logic_error("cannot retire incomplete token");
  for (const auto id : iterator->second.work) core_->items.erase(id);
  core_->tokens.erase(iterator);
  ++core_->metrics.completed_tokens;
  core_->metrics.queued_tokens = core_->tokens.size();
  core_->metrics.pending_items = core_->items.size();
}

void ContinuousBatchScheduler::cancel(RequestId request) noexcept {
  if (!core_->requests.erase(request)) return;
  for (auto iterator = core_->tokens.begin(); iterator != core_->tokens.end();) {
    if (iterator->first.request != request) { ++iterator; continue; }
    for (const auto id : iterator->second.work) core_->items.erase(id);
    iterator = core_->tokens.erase(iterator);
  }
  ++core_->metrics.cancelled_requests;
  core_->metrics.active_requests = core_->requests.size();
  core_->metrics.queued_tokens = core_->tokens.size();
  core_->metrics.pending_items = core_->items.size();
}

void ContinuousBatchScheduler::finish(RequestId request) noexcept {
  const auto has_tokens = std::any_of(core_->tokens.begin(), core_->tokens.end(),
      [&](const auto& entry) { return entry.first.request == request; });
  if (!has_tokens) core_->requests.erase(request);
  core_->metrics.active_requests = core_->requests.size();
}

SchedulerSnapshot ContinuousBatchScheduler::snapshot() const noexcept {
  return core_->metrics;
}

}  // namespace expert::runtime
