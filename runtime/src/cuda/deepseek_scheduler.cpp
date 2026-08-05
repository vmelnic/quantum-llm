#include "expert/runtime/cuda/deepseek_scheduler.hpp"

#include "expert/runtime/expert_record.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace expert::runtime::cuda {
namespace {

using namespace std::chrono_literals;

bool terminal(DeepSeekScheduledState state) noexcept {
  return state == DeepSeekScheduledState::complete ||
         state == DeepSeekScheduledState::failed ||
         state == DeepSeekScheduledState::cancelled;
}

Status copied_status(const Status& status) {
  return {status.code(), std::string(status.message())};
}

struct PendingAcquire final {
  std::uint32_t expert{};
  AcquireHandle handle;
};

struct HeldLease final {
  std::uint32_t expert{};
  ExpertLease lease;
};

struct ScheduledRequest final {
  std::uint64_t id{};
  std::shared_ptr<DeepSeekDecodeController> controller;
  DeepSeekScheduledState state{DeepSeekScheduledState::runnable};
  Status status;
  std::uint32_t layer{};
  std::deque<std::uint32_t> queued_experts;
  std::vector<PendingAcquire> inflight;
  std::vector<HeldLease> leases;
  bool runnable_queued{};
  bool acquire_queued{};
};

}  // namespace

struct DeepSeekDecodeScheduler::Core final {
  Core(DeepSeekDecodeSchedulerConfig value, ExpertCache& expert_cache,
       const DeepSeekExpertCatalog& expert_catalog)
      : config(value), cache(expert_cache), catalog(expert_catalog) {}

  DeepSeekDecodeSchedulerConfig config;
  ExpertCache& cache;
  const DeepSeekExpertCatalog& catalog;
  std::map<std::uint64_t, std::unique_ptr<ScheduledRequest>> requests;
  std::deque<std::uint64_t> runnable;
  std::deque<std::uint64_t> acquisition_round_robin;
  std::size_t inflight_acquires{};
  DeepSeekDecodeSchedulerSnapshot metrics;

  void enqueue_runnable(ScheduledRequest& request) {
    if (!request.runnable_queued &&
        request.state == DeepSeekScheduledState::runnable) {
      runnable.push_back(request.id);
      request.runnable_queued = true;
    }
  }

  void enqueue_acquisition(ScheduledRequest& request) {
    if (!request.acquire_queued && !request.queued_experts.empty() &&
        request.state == DeepSeekScheduledState::waiting_for_experts) {
      acquisition_round_robin.push_back(request.id);
      request.acquire_queued = true;
    }
  }

  void remove_from_queues(std::uint64_t id) {
    std::erase(runnable, id);
    std::erase(acquisition_round_robin, id);
  }

  void abandon_acquisitions(ScheduledRequest& request) noexcept {
    for (auto& pending : request.inflight) pending.handle.cancel();
    if (request.inflight.size() <= inflight_acquires)
      inflight_acquires -= request.inflight.size();
    else
      inflight_acquires = 0U;
    request.inflight.clear();
    request.queued_experts.clear();
    request.leases.clear();
    request.acquire_queued = false;
  }

  void fail(ScheduledRequest& request, Status status) noexcept {
    if (terminal(request.state)) return;
    abandon_acquisitions(request);
    static_cast<void>(request.controller->cancel());
    request.runnable_queued = false;
    request.state = DeepSeekScheduledState::failed;
    request.status = std::move(status);
    ++metrics.failed_requests;
  }

  Status wait_for_experts(
      ScheduledRequest& request,
      const DeepSeekDecodeAdvanceResult& result) {
    if (result.missing_experts.empty()) {
      return {ErrorCode::internal,
              "DeepSeek controller suspended without missing experts"};
    }
    std::set<std::uint32_t> unique;
    for (const auto expert : result.missing_experts) {
      if (expert >= kDeepSeekCatalogExperts) {
        return {ErrorCode::internal,
                "always-resident DeepSeek shared expert is unavailable"};
      }
      if (!unique.insert(expert).second) {
        return {ErrorCode::internal,
                "DeepSeek controller returned a duplicate missing expert"};
      }
      const auto held = std::any_of(
          request.leases.begin(), request.leases.end(),
          [expert](const HeldLease& value) { return value.expert == expert; });
      if (held) {
        return {ErrorCode::internal,
                "leased DeepSeek expert is absent from the device directory"};
      }
      if (!catalog.find(result.layer, expert)) {
        return {ErrorCode::invalid_argument,
                "DeepSeek routed expert is absent from the catalog"};
      }
    }
    request.state = DeepSeekScheduledState::waiting_for_experts;
    request.layer = result.layer;
    for (const auto expert : result.missing_experts)
      request.queued_experts.push_back(expert);
    enqueue_acquisition(request);
    return Status::success();
  }

  bool pump_acquisitions() {
    bool started = false;
    while (inflight_acquires < config.maximum_inflight_acquires &&
           !acquisition_round_robin.empty()) {
      const auto id = acquisition_round_robin.front();
      acquisition_round_robin.pop_front();
      const auto iterator = requests.find(id);
      if (iterator == requests.end()) continue;
      auto& request = *iterator->second;
      request.acquire_queued = false;
      if (request.state != DeepSeekScheduledState::waiting_for_experts ||
          request.queued_experts.empty())
        continue;
      const auto expert = request.queued_experts.front();
      request.queued_experts.pop_front();
      const auto* record = catalog.find(request.layer, expert);
      if (!record) {
        fail(request, {ErrorCode::invalid_argument,
                       "DeepSeek routed expert catalog lookup failed"});
        continue;
      }
      ExpertKey key{config.model_id, request.layer, expert,
                    kExpertQuantAbiDeepSeekSm86};
      request.inflight.push_back(
          {expert, cache.acquire(key, *record)});
      ++inflight_acquires;
      ++metrics.acquires_started;
      started = true;
      enqueue_acquisition(request);
    }
    return started;
  }

  bool collect_acquisitions() {
    bool completed = false;
    for (auto& [id, owned] : requests) {
      (void)id;
      auto& request = *owned;
      if (request.state != DeepSeekScheduledState::waiting_for_experts)
        continue;
      for (std::size_t index = 0U; index < request.inflight.size();) {
        auto& pending = request.inflight[index];
        if (pending.handle.wait_for(0ms) != std::future_status::ready) {
          ++index;
          continue;
        }
        auto result = pending.handle.get();
        const auto expert = pending.expert;
        request.inflight.erase(request.inflight.begin() + index);
        if (inflight_acquires > 0U) --inflight_acquires;
        ++metrics.acquires_completed;
        completed = true;
        if (!result.status.ok() || !result.lease) {
          fail(request, result.status.ok()
                            ? Status(ErrorCode::internal,
                                     "DeepSeek cache returned no expert lease")
                            : copied_status(result.status));
          break;
        }
        request.leases.push_back({expert, std::move(result.lease)});
      }
      if (request.state == DeepSeekScheduledState::waiting_for_experts &&
          request.queued_experts.empty() && request.inflight.empty()) {
        request.state = DeepSeekScheduledState::runnable;
        enqueue_runnable(request);
      }
    }
    return completed;
  }

  void service_acquisitions() {
    for (;;) {
      const bool completed = collect_acquisitions();
      const bool started = pump_acquisitions();
      if (!completed && !started) break;
    }
  }
};

DeepSeekDecodeScheduler::DeepSeekDecodeScheduler(
    DeepSeekDecodeSchedulerConfig config, ExpertCache& cache,
    const DeepSeekExpertCatalog& catalog)
    : core_(std::make_unique<Core>(config, cache, catalog)) {
  if (config.model_id == 0U || config.maximum_requests == 0U ||
      config.maximum_inflight_acquires == 0U ||
      config.maximum_layer_advances_per_poll == 0U ||
      catalog.size() != static_cast<std::size_t>(kDeepSeekCatalogLayers) *
                            kDeepSeekCatalogExperts) {
    throw std::invalid_argument("invalid DeepSeek decode scheduler contract");
  }
}

DeepSeekDecodeScheduler::~DeepSeekDecodeScheduler() {
  if (!core_) return;
  for (auto& [id, request] : core_->requests) {
    (void)id;
    core_->abandon_acquisitions(*request);
    static_cast<void>(request->controller->cancel());
  }
}

Status DeepSeekDecodeScheduler::submit(
    std::uint64_t request_id,
    std::shared_ptr<DeepSeekDecodeController> controller,
    const DeepSeekDecodeBegin& begin) {
  if (request_id == 0U || !controller || core_->requests.contains(request_id)) {
    ++core_->metrics.rejected_requests;
    return {ErrorCode::invalid_argument,
            "invalid or duplicate DeepSeek scheduled request"};
  }
  if (core_->requests.size() >= core_->config.maximum_requests) {
    ++core_->metrics.rejected_requests;
    return {ErrorCode::backpressure,
            "DeepSeek scheduled request capacity exhausted"};
  }
  const auto started = controller->begin(begin);
  if (!started.ok()) {
    ++core_->metrics.rejected_requests;
    return copied_status(started);
  }
  auto request = std::make_unique<ScheduledRequest>();
  request->id = request_id;
  request->controller = std::move(controller);
  request->layer = begin.first_layer;
  auto* published = request.get();
  core_->requests.emplace(request_id, std::move(request));
  core_->enqueue_runnable(*published);
  ++core_->metrics.submitted_requests;
  return Status::success();
}

Status DeepSeekDecodeScheduler::poll() {
  core_->service_acquisitions();
  std::size_t advances = 0U;
  while (advances < core_->config.maximum_layer_advances_per_poll &&
         !core_->runnable.empty()) {
    const auto id = core_->runnable.front();
    core_->runnable.pop_front();
    const auto iterator = core_->requests.find(id);
    if (iterator == core_->requests.end()) continue;
    auto& request = *iterator->second;
    request.runnable_queued = false;
    if (request.state != DeepSeekScheduledState::runnable) continue;
    auto result = request.controller->advance();
    ++advances;
    ++core_->metrics.layer_advances;
    request.layer = result.layer;
    if (!result.status.ok()) {
      core_->fail(request, copied_status(result.status));
      continue;
    }
    switch (result.progress) {
      case DeepSeekDecodeProgress::needs_experts: {
        ++core_->metrics.expert_suspensions;
        const auto status = core_->wait_for_experts(request, result);
        if (!status.ok()) core_->fail(request, copied_status(status));
        break;
      }
      case DeepSeekDecodeProgress::layer_complete:
        request.leases.clear();
        core_->enqueue_runnable(request);
        break;
      case DeepSeekDecodeProgress::token_complete:
        request.leases.clear();
        request.state = DeepSeekScheduledState::complete;
        ++core_->metrics.completed_requests;
        break;
    }
  }
  core_->service_acquisitions();
  return Status::success();
}

Status DeepSeekDecodeScheduler::cancel(std::uint64_t request_id) noexcept {
  const auto iterator = core_->requests.find(request_id);
  if (iterator == core_->requests.end())
    return {ErrorCode::invalid_argument, "unknown DeepSeek scheduled request"};
  auto& request = *iterator->second;
  if (terminal(request.state))
    return {ErrorCode::invalid_argument,
            "DeepSeek scheduled request is already terminal"};
  core_->abandon_acquisitions(request);
  const auto cancelled = request.controller->cancel();
  request.runnable_queued = false;
  request.state = DeepSeekScheduledState::cancelled;
  request.status = cancelled.ok()
                       ? Status(ErrorCode::cancelled,
                                "DeepSeek scheduled request cancelled")
                       : copied_status(cancelled);
  ++core_->metrics.cancelled_requests;
  return Status::success();
}

Status DeepSeekDecodeScheduler::retire(std::uint64_t request_id) {
  const auto iterator = core_->requests.find(request_id);
  if (iterator == core_->requests.end() || !terminal(iterator->second->state))
    return {ErrorCode::invalid_argument,
            "only a terminal DeepSeek request can be retired"};
  core_->remove_from_queues(request_id);
  core_->requests.erase(iterator);
  return Status::success();
}

std::optional<DeepSeekScheduledRequestSnapshot>
DeepSeekDecodeScheduler::inspect(std::uint64_t request_id) const {
  const auto iterator = core_->requests.find(request_id);
  if (iterator == core_->requests.end()) return std::nullopt;
  const auto& request = *iterator->second;
  return DeepSeekScheduledRequestSnapshot{
      request.state, copied_status(request.status), request.layer,
      request.queued_experts.size(), request.inflight.size(),
      request.leases.size()};
}

DeepSeekDecodeSchedulerSnapshot DeepSeekDecodeScheduler::snapshot() const
    noexcept {
  auto result = core_->metrics;
  result.requests = core_->requests.size();
  result.inflight_acquires = core_->inflight_acquires;
  result.runnable_requests = 0U;
  result.waiting_requests = 0U;
  for (const auto& [id, request] : core_->requests) {
    (void)id;
    if (request->state == DeepSeekScheduledState::runnable)
      ++result.runnable_requests;
    if (request->state == DeepSeekScheduledState::waiting_for_experts)
      ++result.waiting_requests;
  }
  return result;
}

}  // namespace expert::runtime::cuda
