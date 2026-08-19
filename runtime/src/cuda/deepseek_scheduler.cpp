#include "expert/runtime/cuda/deepseek_scheduler.hpp"

#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/expert_store.hpp"

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

enum class PredictionState : std::uint8_t {
  unavailable,
  queued,
  pending,
  ready,
  resident,
};

struct PendingPrefetch final {
  ExpertKey key;
  AcquireHandle handle;
};

using LayerWorkingSet = std::vector<std::vector<HeldLease>>;

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
  bool cuda_pending{};
};

}  // namespace

struct DeepSeekDecodeScheduler::Core final {
  Core(DeepSeekDecodeSchedulerConfig value, RoutedExpertRuntime& expert_runtime,
       DeepSeekHybridSchedulerDependencies hybrid_dependencies)
      : config(value), routed(expert_runtime), cache(expert_runtime.cache()),
        store(expert_runtime.store()), catalog(expert_runtime.catalog()),
        hybrid(std::move(hybrid_dependencies)),
        predictions(expert_runtime.component().layer_count) {}

  DeepSeekDecodeSchedulerConfig config;
  RoutedExpertRuntime& routed;
  ExpertCache& cache;
  IExpertStore& store;
  const ExpertCatalog& catalog;
  DeepSeekHybridSchedulerDependencies hybrid;
  std::map<std::uint64_t, std::unique_ptr<ScheduledRequest>> requests;
  std::map<DeepSeekDecodeController*, ControllerWorkingSet> working_sets;
  std::deque<std::uint64_t> runnable;
  std::deque<std::uint64_t> acquisition_round_robin;
  std::size_t inflight_acquires{};
  std::deque<ExpertKey> prefetch_queue;
  std::vector<PendingPrefetch> pending_prefetch;
  std::vector<std::map<std::uint32_t, PredictionState>> predictions;
  DeepSeekDecodeSchedulerSnapshot metrics;
  std::uint64_t observed_cpu_compute_ns{};
  std::uint64_t observed_cpu_selections{};

  void refresh_cpu_cost() noexcept {
    if (!hybrid.cpu_executor || !hybrid.planner) return;
    const auto current = hybrid.cpu_executor->telemetry();
    if (current.compute_ns >= observed_cpu_compute_ns &&
        current.selections >= observed_cpu_selections) {
      hybrid.planner->observe_cpu(
          current.compute_ns - observed_cpu_compute_ns,
          current.selections - observed_cpu_selections);
    }
    observed_cpu_compute_ns = current.compute_ns;
    observed_cpu_selections = current.selections;
  }

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
    request.cuda_pending = false;
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
    if (inserted || iterator->second.controller.expired()) {
      iterator->second.layers.clear();
      iterator->second.layers.resize(routed.component().layer_count);
      iterator->second.controller = request.controller;
    }
    if (layer >= iterator->second.layers.size()) return nullptr;
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
    const auto key = routed.key(layer, expert);
    auto handle = cache.acquire(
        key, *record,
        ExpertAcquireOptions{ExpertRequestPriority::demand, false, true,
                             false});
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
    if ((result.route_rows != 1U && result.route_rows != 2U) ||
        result.layer >= routed.component().layer_count ||
        result.routed_experts.empty() ||
        result.routed_experts.size() % result.route_rows != 0U ||
        result.routed_experts.size() / result.route_rows !=
            routed.component().route_width) {
      return {ErrorCode::internal,
              "DeepSeek scheduler received an invalid routed set"};
    }
    const auto route_width = result.routed_experts.size() / result.route_rows;
    for (std::uint32_t row = 0U; row < result.route_rows; ++row) {
      const auto first = result.routed_experts.begin() + row * route_width;
      std::set<std::uint32_t> unique(first, first + route_width);
      if (unique.size() != route_width ||
          *unique.rbegin() >= catalog.experts_per_layer())
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

  void collect_prefetch() {
    for (std::size_t index = 0U; index < pending_prefetch.size();) {
      auto& pending = pending_prefetch[index];
      if (pending.handle.wait_for(0ms) != std::future_status::ready) {
        ++index;
        continue;
      }
      auto result = pending.handle.get();
      auto& predicted = predictions[pending.key.layer];
      const auto item = predicted.find(pending.key.expert);
      if (result.status.ok() && result.lease) {
        if (item != predicted.end()) item->second = PredictionState::ready;
        ++metrics.prefetch_completed;
      } else if (item != predicted.end()) {
        item->second = PredictionState::unavailable;
      }
      pending_prefetch.erase(
          pending_prefetch.begin() + static_cast<std::ptrdiff_t>(index));
    }
  }

  void cancel_prefetch_for_exact_demand(
      std::uint32_t layer, std::span<const std::uint32_t> missing) {
    for (std::size_t index = 0U; index < pending_prefetch.size();) {
      const auto& key = pending_prefetch[index].key;
      const bool exact = key.layer == layer &&
          std::find(missing.begin(), missing.end(), key.expert) !=
              missing.end();
      if (exact) {
        ++index;
        continue;
      }
      pending_prefetch[index].handle.cancel();
      const auto item = predictions[key.layer].find(key.expert);
      if (item != predictions[key.layer].end() &&
          item->second == PredictionState::pending) {
        item->second = PredictionState::queued;
        prefetch_queue.push_back(key);
      }
      pending_prefetch.erase(
          pending_prefetch.begin() + static_cast<std::ptrdiff_t>(index));
      ++metrics.prefetch_cancelled;
    }
  }

  void attribute_predictions(
      std::uint32_t layer, std::span<const std::uint32_t> route) {
    collect_prefetch();
    auto& prior = predictions[layer];
    for (const auto& [expert, state] : prior) {
      const bool selected =
          std::find(route.begin(), route.end(), expert) != route.end();
      if (!selected) {
        ++metrics.prefetch_incorrect;
        continue;
      }
      const auto key = routed.key(layer, expert);
      const auto snapshot = cache.inspect(key);
      if ((state == PredictionState::ready ||
           state == PredictionState::resident) &&
          snapshot && snapshot->has_device_copy) {
        ++metrics.prefetch_useful;
      } else if (state == PredictionState::ready) {
        ++metrics.prefetch_evicted_before_use;
      } else {
        ++metrics.prefetch_late;
      }
    }
    std::erase_if(prefetch_queue, [layer](const ExpertKey& key) {
      return key.layer == layer;
    });
    for (std::size_t index = 0U; index < pending_prefetch.size();) {
      if (pending_prefetch[index].key.layer != layer) {
        ++index;
        continue;
      }
      pending_prefetch[index].handle.cancel();
      pending_prefetch.erase(
          pending_prefetch.begin() + static_cast<std::ptrdiff_t>(index));
      ++metrics.prefetch_cancelled;
    }
    prior.clear();
  }

  void publish_predictions(
      std::uint32_t layer, std::span<const std::uint32_t> route) {
    if (!hybrid.route_census ||
        config.transition_predictions_per_layer == 0U)
      return;
    auto& prior = predictions[layer];
    const auto next = hybrid.route_census->predict_next(
        layer, route, config.transition_predictions_per_layer);
    for (const auto& prediction : next) {
      const auto* record = catalog.find(layer, prediction.key.expert);
      if (!record) continue;
      ++metrics.prefetch_predictions;
      const auto snapshot = cache.inspect(prediction.key);
      if (snapshot && snapshot->has_device_copy) {
        prior.emplace(prediction.key.expert, PredictionState::resident);
      } else if (snapshot && snapshot->has_host_copy &&
                 cache.vram_admission_would_improve(prediction.key,
                                                    *record)) {
        prior.emplace(prediction.key.expert, PredictionState::queued);
        prefetch_queue.push_back(prediction.key);
      } else {
        prior.emplace(prediction.key.expert, PredictionState::unavailable);
      }
    }
  }

  void pump_prefetch() {
    collect_prefetch();
    // Prefetch deliberately runs while demand acquires are in flight: the
    // demand route is usually resident after a few tokens, and suspending
    // predictions during every demand read starved the pipeline.
    while (pending_prefetch.size() < config.maximum_inflight_prefetch &&
           !prefetch_queue.empty()) {
      const auto key = prefetch_queue.front();
      prefetch_queue.pop_front();
      auto item = predictions[key.layer].find(key.expert);
      if (item == predictions[key.layer].end() ||
          item->second != PredictionState::queued)
        continue;
      const auto* record = catalog.find(key.layer, key.expert);
      if (!record) {
        item->second = PredictionState::unavailable;
        continue;
      }
      item->second = PredictionState::pending;
      pending_prefetch.push_back(
          {key, cache.acquire(
                    key, *record,
                    ExpertAcquireOptions{ExpertRequestPriority::prefetch,
                                         false, true, false})});
      ++metrics.prefetch_scheduled;
    }
  }

  Status retain_completed_route(
      ScheduledRequest& request,
      const DeepSeekDecodeAdvanceResult& result) {
    auto status = reconcile_working_set(request, result);
    if (!status.ok()) return status;
    std::vector<ExpertAccess> route_accesses;
    route_accesses.reserve(result.routed_experts.size());
    for (const auto expert : result.routed_experts) {
      route_accesses.push_back(
          {routed.key(result.layer, expert),
           1U});
    }
    if (cache.record_accesses(route_accesses) != route_accesses.size()) {
      return {ErrorCode::internal,
              "DeepSeek route feedback referenced an absent cache entry"};
    }
    if (hybrid.route_census) {
      const auto route_width = routed.component().route_width;
      for (std::uint32_t row = 0U; row < result.route_rows; ++row) {
        const auto first = result.routed_experts.begin() + row * route_width;
        if (row == 0U)
          attribute_predictions(
              result.layer,
              std::span<const std::uint32_t>(first, first + route_width));
        const auto observed = hybrid.route_census->observe(
            result.layer,
            std::span<const std::uint32_t>(first, first + route_width),
            result.route_rows == 1U
                ? std::span<const std::uint32_t>(request.cpu_experts)
                : std::span<const std::uint32_t>());
        if (!observed.ok()) return copied_status(observed);
        ++metrics.route_observations;
        if (row + 1U == result.route_rows)
          publish_predictions(
              result.layer,
              std::span<const std::uint32_t>(first, first + route_width));
      }
    }
    if (!config.retain_previous_route) {
      if (!request.cpu_experts.empty()) ++metrics.hybrid_layers;
      request.leases.clear();
      request.host_leases.clear();
      request.cpu_experts.clear();
      return Status::success();
    }
    auto* retained = layer_working_set(request, result.layer);
    if (retained == nullptr)
      return {ErrorCode::internal,
              "routed working set references an invalid layer"};
    const auto retained_first =
        result.routed_experts.end() - routed.component().route_width;
    const auto retained_contains = [&](std::uint32_t expert) {
      return std::find(retained_first, result.routed_experts.end(), expert) !=
             result.routed_experts.end();
    };
    std::erase_if(*retained, [&](const HeldLease& value) {
      return !retained_contains(value.expert) ||
             cpu_contains(request, value.expert);
    });
    for (auto& held : request.leases) {
      const bool duplicate = std::any_of(
          retained->begin(), retained->end(), [&](const HeldLease& value) {
            return value.expert == held.expert;
          });
      if (retained_contains(held.expert) && !duplicate) {
        retained->push_back({held.expert, std::move(held.lease)});
      }
    }
    request.leases.clear();
    for (auto iterator = retained_first;
         iterator != result.routed_experts.end(); ++iterator) {
      const auto expert = *iterator;
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
    std::set<std::uint32_t> expected_experts(retained_first,
                                             result.routed_experts.end());
    for (const auto expert : request.cpu_experts)
      expected_experts.erase(expert);
    const auto expected = expected_experts.size();
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
    if (result.route_rows != 1U || !hybrid.cpu_executor || !hybrid.planner ||
        !request.cpu_experts.empty())
      return Status::success();
    refresh_cpu_cost();
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
      const auto key = routed.key(result.layer, expert);
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
          {routed.key(result.layer, decision.expert),
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
    cancel_prefetch_for_exact_demand(result.layer,
                                     result.missing_experts);
    auto* retained = layer_working_set(request, result.layer);
    for (const auto expert : result.ready_experts) {
      if (expert == catalog.experts_per_layer()) continue;
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
      if (expert >= catalog.experts_per_layer()) {
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
        resolves.push_back({routed.key(request.layer, expert),
                            *record, ExpertResolveTarget::device,
                            ExpertAcquireOptions{
                                ExpertRequestPriority::demand, false, true,
                                false}});
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
    pump_prefetch();
  }
};

DeepSeekDecodeScheduler::DeepSeekDecodeScheduler(
    DeepSeekDecodeSchedulerConfig config, RoutedExpertRuntime& routed,
    DeepSeekHybridSchedulerDependencies hybrid)
    : core_(std::make_unique<Core>(config, routed, std::move(hybrid))) {
  const auto& catalog = routed.catalog();
  if (config.maximum_requests == 0U ||
      config.maximum_inflight_acquires == 0U ||
      config.maximum_layer_advances_per_poll == 0U ||
      (config.transition_predictions_per_layer != 0U &&
       config.maximum_inflight_prefetch == 0U) ||
      static_cast<bool>(core_->hybrid.cpu_executor) !=
          static_cast<bool>(core_->hybrid.planner) ||
      (core_->hybrid.route_census &&
       (core_->hybrid.route_census->config().model_id !=
            routed.component().namespace_id ||
        core_->hybrid.route_census->config().encoding_abi !=
            routed.component().encoding_abi ||
        core_->hybrid.route_census->config().layer_count !=
            catalog.layer_count() ||
        core_->hybrid.route_census->config().experts_per_layer !=
            catalog.experts_per_layer() ||
        core_->hybrid.route_census->config().route_width !=
            routed.component().route_width)) ||
      catalog.size() != static_cast<std::size_t>(catalog.layer_count()) *
                            catalog.experts_per_layer()) {
    throw std::invalid_argument("invalid DeepSeek decode scheduler contract");
  }
}

DeepSeekDecodeScheduler::~DeepSeekDecodeScheduler() {
  if (!core_) return;
  for (auto& pending : core_->pending_prefetch) pending.handle.cancel();
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

Status DeepSeekDecodeScheduler::submit_verify(
    std::uint64_t request_id,
    std::shared_ptr<DeepSeekDecodeController> controller,
    const DeepSeekVerifyBegin& begin) {
  core_->prune_working_sets();
  if (request_id == 0U || !controller || core_->requests.contains(request_id)) {
    ++core_->metrics.rejected_requests;
    return {ErrorCode::invalid_argument,
            "invalid or duplicate DeepSeek verify request"};
  }
  if (core_->requests.size() >= core_->config.maximum_requests) {
    ++core_->metrics.rejected_requests;
    return {ErrorCode::backpressure,
            "DeepSeek scheduled request capacity exhausted"};
  }
  const auto started = controller->begin_verify_pair(begin);
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
  std::vector<std::uint64_t> deferred_cuda;
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
    request.layer = result.layer;
    if (!result.status.ok()) {
      request.cuda_pending = false;
      core_->fail(request, copied_status(result.status));
      continue;
    }
    switch (result.progress) {
      case DeepSeekDecodeProgress::pending_cuda: {
        request.cuda_pending = true;
        ++core_->metrics.cuda_pending_polls;
        deferred_cuda.push_back(request.id);
        break;
      }
      case DeepSeekDecodeProgress::needs_experts: {
        request.cuda_pending = false;
        ++core_->metrics.layer_advances;
        ++core_->metrics.expert_suspensions;
        const auto status = core_->wait_for_experts(request, result);
        if (!status.ok()) core_->fail(request, copied_status(status));
        break;
      }
      case DeepSeekDecodeProgress::layer_complete: {
        request.cuda_pending = false;
        ++core_->metrics.layer_advances;
        const auto retained = core_->retain_completed_route(request, result);
        if (!retained.ok()) {
          core_->fail(request, copied_status(retained));
          break;
        }
        core_->enqueue_runnable(request);
        break;
      }
      case DeepSeekDecodeProgress::token_complete: {
        request.cuda_pending = false;
        ++core_->metrics.layer_advances;
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
  for (const auto id : deferred_cuda) {
    const auto iterator = core_->requests.find(id);
    if (iterator != core_->requests.end())
      core_->enqueue_runnable(*iterator->second);
  }
  core_->service_acquisitions();
  core_->metrics.poll_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - poll_started)
          .count());
  return Status::success();
}

Status DeepSeekDecodeScheduler::wait_for_cuda_progress() {
  for (auto& [id, request] : core_->requests) {
    (void)id;
    if (!request->cuda_pending || terminal(request->state)) continue;
    const auto started = std::chrono::steady_clock::now();
    const auto status = request->controller->wait_for_cuda();
    core_->metrics.cuda_wait_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    ++core_->metrics.cuda_waits;
    return copied_status(status);
  }
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
  request.cuda_pending = false;
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
  result.cuda_pending_requests = 0U;
  result.retained_working_set_experts = 0U;
  for (const auto& [id, request] : core_->requests) {
    (void)id;
    if (request->state == DeepSeekScheduledState::runnable)
      ++result.runnable_requests;
    if (request->state == DeepSeekScheduledState::waiting_for_experts)
      ++result.waiting_requests;
    if (request->cuda_pending) ++result.cuda_pending_requests;
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
