#include "expert/runtime/cuda/deepseek_scheduler.hpp"

#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/expert_store.hpp"

#include <algorithm>
#include <array>
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

struct PendingResolve final {
  std::vector<std::uint32_t> experts;
  ExpertResolveHandle handle;
};

struct HeldLease final {
  std::uint32_t expert{};
  ExpertLease lease;
};

struct HeldHostLease final {
  std::uint32_t expert{};
  HostExpertLease lease;
};

using LayerWorkingSet =
    std::array<std::vector<HeldLease>, kDeepSeekLayers>;

struct ControllerWorkingSet final {
  std::weak_ptr<DeepSeekDecodeController> controller;
  LayerWorkingSet layers;
};

struct ScheduledRequest final {
  std::uint64_t id{};
  std::shared_ptr<DeepSeekDecodeController> controller;
  DeepSeekScheduledState state{DeepSeekScheduledState::runnable};
  Status status;
  std::uint32_t layer{};
  std::deque<std::uint32_t> queued_experts;
  std::vector<PendingResolve> inflight;
  std::vector<HeldLease> leases;
  std::vector<HeldHostLease> host_leases;
  std::vector<std::uint32_t> cpu_experts;
  std::chrono::steady_clock::time_point expert_wait_started{};
  bool runnable_queued{};
  bool acquire_queued{};
};

}  // namespace

struct DeepSeekDecodeScheduler::Core final {
  Core(DeepSeekDecodeSchedulerConfig value, ExpertCache& expert_cache,
       const DeepSeekExpertCatalog& expert_catalog,
       DeepSeekHybridSchedulerDependencies hybrid_dependencies)
      : config(value), cache(expert_cache), store(expert_cache),
        catalog(expert_catalog), hybrid(std::move(hybrid_dependencies)) {}

  DeepSeekDecodeSchedulerConfig config;
  ExpertCache& cache;
  LocalExpertStore store;
  const DeepSeekExpertCatalog& catalog;
  DeepSeekHybridSchedulerDependencies hybrid;
  std::map<std::uint64_t, std::unique_ptr<ScheduledRequest>> requests;
  std::map<DeepSeekDecodeController*, ControllerWorkingSet> working_sets;
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
    std::size_t cancelled = 0U;
    for (auto& pending : request.inflight) {
      cancelled += pending.experts.size();
      pending.handle.cancel();
    }
    if (cancelled <= inflight_acquires)
      inflight_acquires -= cancelled;
    else
      inflight_acquires = 0U;
    request.inflight.clear();
    request.queued_experts.clear();
    request.leases.clear();
    request.host_leases.clear();
    request.cpu_experts.clear();
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

  [[nodiscard]] bool route_contains(
      const DeepSeekDecodeAdvanceResult& result,
      std::uint32_t expert) const noexcept {
    return std::find(result.routed_experts.begin(),
                     result.routed_experts.end(), expert) !=
           result.routed_experts.end();
  }

  [[nodiscard]] bool cpu_contains(const ScheduledRequest& request,
                                  std::uint32_t expert) const noexcept {
    return std::find(request.cpu_experts.begin(), request.cpu_experts.end(),
                     expert) != request.cpu_experts.end();
  }

  [[nodiscard]] std::vector<HeldLease>* layer_working_set(
      ScheduledRequest& request, std::uint32_t layer) {
    if (!config.retain_previous_route) return nullptr;
    auto [iterator, inserted] = working_sets.try_emplace(
        request.controller.get());
    if (inserted) iterator->second.controller = request.controller;
    return &iterator->second.layers[layer];
  }

  void prune_working_sets() {
    std::erase_if(working_sets, [](const auto& item) {
      return item.second.controller.expired();
    });
  }

  Status acquire_ready_lease(ScheduledRequest& request,
                             std::uint32_t layer,
                             std::uint32_t expert) {
    const auto* record = catalog.find(layer, expert);
    if (!record) {
      return {ErrorCode::invalid_argument,
              "DeepSeek ready expert is absent from the catalog"};
    }
    ExpertKey key{config.model_id, layer, expert,
                  kExpertQuantAbiDeepSeekSm86};
    auto handle = cache.acquire(key, *record);
    if (handle.wait_for(0ms) != std::future_status::ready) {
      handle.cancel();
      return {ErrorCode::internal,
              "directory-ready DeepSeek expert is not cache-ready"};
    }
    auto acquired = handle.get();
    if (!acquired.status.ok() || !acquired.lease) {
      return acquired.status.ok()
                 ? Status(ErrorCode::internal,
                          "DeepSeek ready expert returned no cache lease")
                 : copied_status(acquired.status);
    }
    request.leases.push_back({expert, std::move(acquired.lease)});
    return Status::success();
  }

  Status reconcile_working_set(
      ScheduledRequest& request,
      const DeepSeekDecodeAdvanceResult& result) {
    if (result.routed_experts.size() != 6U) {
      return {ErrorCode::internal,
              "DeepSeek scheduler received an invalid routed set"};
    }
    std::set<std::uint32_t> unique(result.routed_experts.begin(),
                                   result.routed_experts.end());
    if (unique.size() != result.routed_experts.size() ||
        *unique.rbegin() >= kDeepSeekCatalogExperts) {
      return {ErrorCode::internal,
              "DeepSeek scheduler received invalid routed experts"};
    }
    if (auto* retained = layer_working_set(request, result.layer)) {
      std::erase_if(*retained, [&](const HeldLease& value) {
        return !route_contains(result, value.expert) ||
               cpu_contains(request, value.expert);
      });
    }
    return Status::success();
  }

  Status retain_completed_route(
      ScheduledRequest& request,
      const DeepSeekDecodeAdvanceResult& result) {
    if (hybrid.route_census) {
      const auto observed = hybrid.route_census->observe(
          result.layer, result.routed_experts, request.cpu_experts);
      if (!observed.ok()) return copied_status(observed);
      ++metrics.route_observations;
    }
    if (!config.retain_previous_route) {
      if (!request.cpu_experts.empty()) ++metrics.hybrid_layers;
      request.leases.clear();
      request.host_leases.clear();
      request.cpu_experts.clear();
      return Status::success();
    }
    auto status = reconcile_working_set(request, result);
    if (!status.ok()) return status;
    auto* retained = layer_working_set(request, result.layer);
    for (auto& held : request.leases) {
      const bool duplicate = std::any_of(
          retained->begin(), retained->end(), [&](const HeldLease& value) {
            return value.expert == held.expert;
          });
      if (route_contains(result, held.expert) && !duplicate) {
        retained->push_back({held.expert, std::move(held.lease)});
      }
    }
    request.leases.clear();
    for (const auto expert : result.routed_experts) {
      if (cpu_contains(request, expert)) continue;
      const bool held = std::any_of(
          retained->begin(), retained->end(), [expert](const HeldLease& value) {
            return value.expert == expert;
          });
      if (held) continue;
      status = acquire_ready_lease(request, result.layer, expert);
      if (!status.ok()) return status;
      retained->push_back(
          {request.leases.back().expert,
           std::move(request.leases.back().lease)});
      request.leases.pop_back();
    }
    const auto expected = result.routed_experts.size() -
                          request.cpu_experts.size();
    if (retained->size() != expected) {
      return {ErrorCode::internal,
              "DeepSeek retained route has the wrong cardinality"};
    }
    if (!request.cpu_experts.empty()) ++metrics.hybrid_layers;
    request.host_leases.clear();
    request.cpu_experts.clear();
    return Status::success();
  }

  Status plan_host_placements(
      ScheduledRequest& request,
      const DeepSeekDecodeAdvanceResult& result,
      std::set<std::uint32_t>& cpu_selected) {
    if (!hybrid.cpu_executor || !hybrid.planner ||
        !request.cpu_experts.empty())
      return Status::success();
    std::vector<HybridDispatchCandidate> candidates;
    candidates.reserve(result.routed_experts.size());
    for (const auto expert : result.routed_experts) {
      const auto* record = catalog.find(result.layer, expert);
      if (!record)
        return {ErrorCode::invalid_argument,
                "DeepSeek hybrid candidate is absent from catalog"};
      const auto ready = std::find(result.ready_experts.begin(),
                                   result.ready_experts.end(), expert) !=
                         result.ready_experts.end();
      const ExpertKey key{config.model_id, result.layer, expert,
                          kExpertQuantAbiDeepSeekSm86};
      const auto snapshot = cache.inspect(key);
      const auto host_ready = snapshot && snapshot->has_host_copy;
      candidates.push_back({expert, 1U, record->stored_bytes, ready,
                            host_ready, true});
    }
    const auto placement = hybrid.planner->plan(candidates);
    if (!placement.status.ok()) return copied_status(placement.status);
    std::vector<ExpertResolveRequest> host_requests;
    for (const auto& decision : placement.decisions) {
      if (decision.executor != HybridExecutor::cpu_local) continue;
      const auto* record = catalog.find(result.layer, decision.expert);
      host_requests.push_back(
          {ExpertKey{config.model_id, result.layer, decision.expert,
                     kExpertQuantAbiDeepSeekSm86},
           *record, ExpertResolveTarget::host_ready});
    }
    if (host_requests.empty()) return Status::success();
    auto handle = store.resolve(host_requests);
    auto resolved = handle.poll();
    if (!resolved || !resolved->status.ok() ||
        resolved->experts.size() != host_requests.size()) {
      // RAM placement is opportunistic. An eviction between inspect and lease
      // acquisition falls back to the ordinary device path.
      return Status::success();
    }
    std::vector<DeepSeekCpuExpertPlacement> staged;
    staged.reserve(resolved->experts.size());
    for (auto& expert : resolved->experts) {
      if (expert.placement != ExpertPlacementKind::host ||
          !expert.host_lease)
        return {ErrorCode::internal,
                "DeepSeek host resolve returned device placement"};
      cpu_selected.insert(expert.key.expert);
      request.cpu_experts.push_back(expert.key.expert);
      request.host_leases.push_back(
          {expert.key.expert, std::move(expert.host_lease)});
    }
    for (const auto& held : request.host_leases) {
      if (!cpu_selected.contains(held.expert)) continue;
      staged.push_back({held.expert, held.lease.bytes(),
                        held.lease.compact_sections()});
    }
    const auto staged_status = request.controller->stage_cpu_placements(staged);
    if (!staged_status.ok()) return copied_status(staged_status);
    metrics.host_resolves += staged.size();
    metrics.cpu_placements += staged.size();
    return Status::success();
  }

  Status wait_for_experts(
      ScheduledRequest& request,
      const DeepSeekDecodeAdvanceResult& result) {
    if (result.missing_experts.empty()) {
      return {ErrorCode::internal,
              "DeepSeek controller suspended without missing experts"};
    }
    auto status = reconcile_working_set(request, result);
    if (!status.ok()) return status;
    auto* retained = layer_working_set(request, result.layer);
    for (const auto expert : result.ready_experts) {
      if (expert == kDeepSeekCatalogExperts) continue;
      if (!route_contains(result, expert)) {
        return {ErrorCode::internal,
                "DeepSeek controller returned an unrelated ready expert"};
      }
      const bool already_held =
          (retained != nullptr &&
           std::any_of(retained->begin(), retained->end(),
                       [expert](const HeldLease& value) {
                         return value.expert == expert;
                       })) ||
          std::any_of(request.leases.begin(), request.leases.end(),
                      [expert](const HeldLease& value) {
                        return value.expert == expert;
                      });
      if (!already_held) {
        status = acquire_ready_lease(request, result.layer, expert);
        if (!status.ok()) return status;
      }
    }
    std::set<std::uint32_t> unique;
    std::set<std::uint32_t> cpu_selected;
    status = plan_host_placements(request, result, cpu_selected);
    if (!status.ok()) return status;
    for (const auto expert : result.missing_experts) {
      if (expert >= kDeepSeekCatalogExperts) {
        return {ErrorCode::internal,
                "always-resident DeepSeek shared expert is unavailable"};
      }
      if (!unique.insert(expert).second) {
        return {ErrorCode::internal,
                "DeepSeek controller returned a duplicate missing expert"};
      }
      if (cpu_selected.contains(expert)) continue;
      const auto held = std::any_of(
          request.leases.begin(), request.leases.end(),
          [expert](const HeldLease& value) { return value.expert == expert; }) ||
          (retained != nullptr &&
           std::any_of(retained->begin(), retained->end(),
                       [expert](const HeldLease& value) {
                         return value.expert == expert;
                       }));
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
    request.expert_wait_started = std::chrono::steady_clock::now();
    request.layer = result.layer;
    for (const auto expert : result.missing_experts) {
      if (!cpu_selected.contains(expert))
        request.queued_experts.push_back(expert);
    }
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
      const auto available =
          config.maximum_inflight_acquires - inflight_acquires;
      const auto count = std::min(available, request.queued_experts.size());
      std::vector<std::uint32_t> experts;
      std::vector<ExpertResolveRequest> resolves;
      experts.reserve(count);
      resolves.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        const auto expert = request.queued_experts.front();
        request.queued_experts.pop_front();
        const auto* record = catalog.find(request.layer, expert);
        if (!record) {
          fail(request, {ErrorCode::invalid_argument,
                         "DeepSeek routed expert catalog lookup failed"});
          break;
        }
        experts.push_back(expert);
        resolves.push_back({ExpertKey{config.model_id, request.layer, expert,
                                     kExpertQuantAbiDeepSeekSm86},
                            *record});
      }
      if (request.state != DeepSeekScheduledState::waiting_for_experts)
        continue;
      auto handle = store.resolve(resolves);
      if (!handle.valid()) {
        fail(request, {ErrorCode::internal,
                       "DeepSeek batch expert resolve was rejected"});
        continue;
      }
      request.inflight.push_back({std::move(experts), std::move(handle)});
      inflight_acquires += count;
      metrics.acquires_started += count;
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
        auto result = pending.handle.poll();
        if (!result) {
          ++index;
          continue;
        }
        const auto count = pending.experts.size();
        request.inflight.erase(request.inflight.begin() + index);
        inflight_acquires = count <= inflight_acquires
                                ? inflight_acquires - count
                                : 0U;
        metrics.acquires_completed += count;
        completed = true;
        if (!result->status.ok() || result->experts.size() != count) {
          fail(request, result->status.ok()
                            ? Status(ErrorCode::internal,
                                     "DeepSeek store returned incomplete batch")
                            : copied_status(result->status));
          break;
        }
        for (auto& resolved : result->experts) {
          if (resolved.placement != ExpertPlacementKind::device ||
              !resolved.device_lease) {
            fail(request, {ErrorCode::internal,
                           "DeepSeek device resolve returned host placement"});
            break;
          }
          request.leases.push_back(
              {resolved.key.expert, std::move(resolved.device_lease)});
        }
      }
      if (request.state == DeepSeekScheduledState::waiting_for_experts &&
          request.queued_experts.empty() && request.inflight.empty()) {
        if (request.expert_wait_started !=
            std::chrono::steady_clock::time_point{}) {
          metrics.expert_wait_ns += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() -
                  request.expert_wait_started)
                  .count());
          request.expert_wait_started = {};
        }
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
    const DeepSeekExpertCatalog& catalog,
    DeepSeekHybridSchedulerDependencies hybrid)
    : core_(std::make_unique<Core>(config, cache, catalog,
                                   std::move(hybrid))) {
  if (config.model_id == 0U || config.maximum_requests == 0U ||
      config.maximum_inflight_acquires == 0U ||
      config.maximum_layer_advances_per_poll == 0U ||
      static_cast<bool>(core_->hybrid.cpu_executor) !=
          static_cast<bool>(core_->hybrid.planner) ||
      (core_->hybrid.route_census &&
       (core_->hybrid.route_census->config().model_id != config.model_id ||
        core_->hybrid.route_census->config().quant_abi !=
            kExpertQuantAbiDeepSeekSm86 ||
        core_->hybrid.route_census->config().layer_count !=
            kDeepSeekCatalogLayers ||
        core_->hybrid.route_census->config().experts_per_layer !=
            kDeepSeekCatalogExperts ||
        core_->hybrid.route_census->config().route_width != 6U)) ||
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
  core_->prune_working_sets();
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
  if (core_->hybrid.cpu_executor && !controller->hybrid_configured()) {
    auto workspace = create_deepseek_ffn_hybrid_workspace();
    if (!workspace.status.ok() || !workspace.workspace) {
      ++core_->metrics.rejected_requests;
      return workspace.status.ok()
                 ? Status(ErrorCode::internal,
                          "DeepSeek hybrid workspace returned no ownership")
                 : copied_status(workspace.status);
    }
    const auto configured = controller->configure_hybrid(
        core_->hybrid.cpu_executor, std::move(workspace.workspace));
    if (!configured.ok()) {
      ++core_->metrics.rejected_requests;
      return copied_status(configured);
    }
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
  const auto poll_started = std::chrono::steady_clock::now();
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
    const auto advance_started = std::chrono::steady_clock::now();
    auto result = request.controller->advance();
    core_->metrics.controller_advance_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - advance_started)
            .count());
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
      case DeepSeekDecodeProgress::layer_complete: {
        const auto retained = core_->retain_completed_route(request, result);
        if (!retained.ok()) {
          core_->fail(request, copied_status(retained));
          break;
        }
        core_->enqueue_runnable(request);
        break;
      }
      case DeepSeekDecodeProgress::token_complete: {
        const auto retained = core_->retain_completed_route(request, result);
        if (!retained.ok()) {
          core_->fail(request, copied_status(retained));
          break;
        }
        request.state = DeepSeekScheduledState::complete;
        ++core_->metrics.completed_requests;
        break;
      }
    }
  }
  core_->service_acquisitions();
  core_->metrics.poll_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - poll_started)
          .count());
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
      request.leases.size(), request.host_leases.size()};
}

DeepSeekDecodeSchedulerSnapshot DeepSeekDecodeScheduler::snapshot() const
    noexcept {
  auto result = core_->metrics;
  result.requests = core_->requests.size();
  result.inflight_acquires = core_->inflight_acquires;
  result.runnable_requests = 0U;
  result.waiting_requests = 0U;
  result.retained_working_set_experts = 0U;
  for (const auto& [id, request] : core_->requests) {
    (void)id;
    if (request->state == DeepSeekScheduledState::runnable)
      ++result.runnable_requests;
    if (request->state == DeepSeekScheduledState::waiting_for_experts)
      ++result.waiting_requests;
  }
  for (const auto& [controller, working_set] : core_->working_sets) {
    (void)controller;
    for (const auto& layer : working_set.layers) {
      result.retained_working_set_experts += layer.size();
    }
  }
  return result;
}

}  // namespace expert::runtime::cuda
