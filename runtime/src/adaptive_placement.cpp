#include "expert/runtime/adaptive_placement.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace expert::runtime {

struct AdaptivePlacementPlanner::Impl final {
  struct Candidate final {
    PayloadRecord record;
    std::uint64_t debt_ns{};
    std::uint64_t last_seen_epoch{};
    std::uint64_t predicted_epoch{};
    std::uint64_t queue_generation{};
    std::uint32_t observations{};
    double routing_score_ewma{};
    bool queued{};
    bool pending{};
    bool demanded{};
    bool ready_prediction{};
  };

  struct Pending final {
    ExpertKey key;
    AcquireHandle handle;
    std::uint64_t scheduled_epoch{};
    std::uint64_t bytes{};
    bool cancelled{};
  };

  struct Eligible final {
    double priority{};
    ExpertKey key;
    std::uint64_t generation{};

    bool operator<(const Eligible& other) const noexcept {
      if (priority != other.priority) return priority < other.priority;
      return other.key < key;
    }
  };

  Impl(ExpertCache& value, AdaptivePlacementConfig settings)
      : cache(value), config(settings),
        cpu_ns_per_selection(settings.initial_cpu_ns_per_selection) {
    if (!std::isfinite(config.conservative_h2d_bytes_per_second) ||
        config.conservative_h2d_bytes_per_second <= 0.0 ||
        !std::isfinite(config.admission_margin) ||
        config.admission_margin < 1.0 ||
        !std::isfinite(config.initial_cpu_ns_per_selection) ||
        config.initial_cpu_ns_per_selection <= 0.0 ||
        !std::isfinite(config.cpu_cost_ewma_alpha) ||
        config.cpu_cost_ewma_alpha <= 0.0 ||
        config.cpu_cost_ewma_alpha > 1.0 ||
        !std::isfinite(config.routing_score_reuse_weight) ||
        config.routing_score_reuse_weight < 0.0 ||
        config.maximum_inflight_promotions == 0 ||
        config.maximum_candidates == 0 ||
        config.minimum_recent_observations == 0 ||
        config.candidate_ttl_epochs == 0) {
      throw std::invalid_argument("invalid adaptive placement configuration");
    }
  }

  void observe(std::uint64_t elapsed_ns, std::uint64_t selections) noexcept {
    if (elapsed_ns == 0 || selections == 0) return;
    const auto observed = static_cast<double>(elapsed_ns) / selections;
    cpu_ns_per_selection =
        cpu_ns_per_selection * (1.0 - config.cpu_cost_ewma_alpha) +
        observed * config.cpu_cost_ewma_alpha;
  }

  [[nodiscard]] std::uint64_t upload_cost(const Candidate& candidate) const {
    return std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(std::ceil(
               static_cast<double>(candidate.record.stored_bytes) * 1.0e9 /
               config.conservative_h2d_bytes_per_second *
               config.admission_margin)));
  }

  [[nodiscard]] double effective_debt(const Candidate& candidate) const {
    return static_cast<double>(candidate.debt_ns) *
           (1.0 + config.routing_score_reuse_weight *
                      candidate.routing_score_ewma);
  }

  void attribute_resident_use(Candidate& candidate) noexcept {
    if (candidate.ready_prediction) {
      ++metrics.useful_prefetches;
      metrics.useful_prefetch_bytes += candidate.record.stored_bytes;
      candidate.ready_prediction = false;
      candidate.debt_ns = 0;
    }
  }

  void attribute_missing_use(Candidate& candidate) noexcept {
    if (candidate.pending) {
      candidate.demanded = true;
    } else if (candidate.ready_prediction) {
      candidate.ready_prediction = false;
      candidate.debt_ns = 0;
      candidate.observations = 0;
      ++metrics.wasted_prefetches;
      metrics.wasted_prefetch_bytes += candidate.record.stored_bytes;
    }
  }

  void complete_ready() {
    for (std::size_t index = 0; index < pending.size();) {
      if (pending[index].handle.wait_for(std::chrono::milliseconds(0)) !=
          std::future_status::ready) {
        ++index;
        continue;
      }
      auto result = pending[index].handle.get();
      auto candidate = candidates.find(pending[index].key);
      if (candidate != candidates.end()) {
        candidate->second.pending = false;
        if (result.status.ok() && result.lease) {
          ++metrics.completed;
          if (candidate->second.demanded) {
            ++metrics.useful_prefetches;
            metrics.useful_prefetch_bytes += pending[index].bytes;
            candidate->second.demanded = false;
            candidate->second.debt_ns = 0;
          } else {
            candidate->second.ready_prediction = true;
            candidate->second.predicted_epoch = epoch;
          }
        } else if (!pending[index].cancelled) {
          ++metrics.failed;
        }
      }
      pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(index));
    }
  }

  void expire_stale() {
    for (auto& item : pending) {
      auto candidate = candidates.find(item.key);
      if (!item.cancelled && candidate != candidates.end() &&
          !candidate->second.demanded &&
          epoch - item.scheduled_epoch > config.candidate_ttl_epochs) {
        item.handle.cancel();
        item.cancelled = true;
        ++metrics.stale_cancellations;
        ++metrics.wasted_prefetches;
        metrics.wasted_prefetch_bytes += item.bytes;
      }
    }
    for (auto& [key, candidate] : candidates) {
      (void)key;
      if (candidate.ready_prediction &&
          epoch - candidate.predicted_epoch > config.candidate_ttl_epochs) {
        candidate.ready_prediction = false;
        candidate.debt_ns = 0;
        candidate.observations = 0;
        ++metrics.wasted_prefetches;
        metrics.wasted_prefetch_bytes += candidate.record.stored_bytes;
      }
      if (candidate.queued &&
          epoch - candidate.last_seen_epoch > config.candidate_ttl_epochs) {
        candidate.queued = false;
        ++candidate.queue_generation;
        ++metrics.stale_cancellations;
      }
    }
    complete_ready();
  }

  void observe_routes(std::span<const ExpertKey> gpu_resident,
                      std::span<const ExpertKey> missing) {
    complete_ready();
    ++epoch;
    for (const auto& key : gpu_resident) {
      const auto candidate = candidates.find(key);
      if (candidate != candidates.end())
        attribute_resident_use(candidate->second);
    }
    for (const auto& key : missing) {
      const auto candidate = candidates.find(key);
      if (candidate != candidates.end())
        attribute_missing_use(candidate->second);
    }
    if (!pending.empty() || (epoch & 0x3fU) == 0U) expire_stale();
  }

  bool ensure_candidate_capacity(const ExpertKey& key) {
    if (candidates.contains(key)) return true;
    if (candidates.size() < config.maximum_candidates) return true;
    auto victim = candidates.end();
    for (auto iterator = candidates.begin(); iterator != candidates.end();
         ++iterator) {
      const auto& candidate = iterator->second;
      if (candidate.queued || candidate.pending || candidate.ready_prediction)
        continue;
      if (victim == candidates.end() ||
          candidate.last_seen_epoch < victim->second.last_seen_epoch) {
        victim = iterator;
      }
    }
    if (victim == candidates.end()) {
      ++metrics.credit_rejections;
      return false;
    }
    candidates.erase(victim);
    ++metrics.candidate_evictions;
    return true;
  }

  void queue_if_profitable(const ExpertKey& key, Candidate& candidate) {
    const auto cost = upload_cost(candidate);
    if (!config.enable_prefetch || frozen || candidate.queued ||
        candidate.pending || cost == 0 ||
        candidate.observations < config.minimum_recent_observations ||
        effective_debt(candidate) < static_cast<double>(cost)) {
      return;
    }
    candidate.queued = true;
    ++candidate.queue_generation;
    eligible.push({effective_debt(candidate) / static_cast<double>(cost), key,
                   candidate.queue_generation});
    if (pending.size() >= config.maximum_inflight_promotions)
      ++metrics.credit_rejections;
  }

  void fill_slots() {
    if (frozen) return;
    while (pending.size() < config.maximum_inflight_promotions &&
           !eligible.empty()) {
      const auto next = eligible.top();
      eligible.pop();
      auto iterator = candidates.find(next.key);
      if (iterator == candidates.end() || !iterator->second.queued ||
          iterator->second.queue_generation != next.generation) {
        continue;
      }
      auto& candidate = iterator->second;
      candidate.queued = false;
      if (candidate.pending) continue;
      if (!cache.vram_admission_would_improve(next.key, candidate.record)) {
        ++metrics.admission_rejected;
        continue;
      }
      pending.push_back({next.key, cache.acquire(next.key, candidate.record),
                         epoch, candidate.record.stored_bytes, false});
      candidate.debt_ns = 0;
      candidate.pending = true;
      candidate.demanded = false;
      ++metrics.scheduled;
      metrics.scheduled_bytes += candidate.record.stored_bytes;
      complete_ready();
    }
  }

  void consider(const ExpertKey& key, const PayloadRecord& record,
                std::uint32_t selections, double routing_score_sum) {
    complete_ready();
    if (selections == 0 || !ensure_candidate_capacity(key)) return;
    ++metrics.considered;
    auto& candidate = candidates[key];
    candidate.record = record;
    if (candidate.last_seen_epoch != 0 &&
        epoch - candidate.last_seen_epoch > config.candidate_ttl_epochs) {
      candidate.debt_ns = 0;
      candidate.observations = 0;
      candidate.routing_score_ewma = 0.0;
    }
    candidate.last_seen_epoch = epoch;
    if (candidate.observations != std::numeric_limits<std::uint32_t>::max())
      ++candidate.observations;
    const auto average_score =
        std::isfinite(routing_score_sum)
            ? std::clamp(routing_score_sum / selections, 0.0, 1.0)
            : 0.0;
    candidate.routing_score_ewma =
        candidate.routing_score_ewma * 0.75 + average_score * 0.25;
    const auto addition = static_cast<std::uint64_t>(
        cpu_ns_per_selection * static_cast<double>(selections));
    candidate.debt_ns += std::min(
        std::numeric_limits<std::uint64_t>::max() - candidate.debt_ns,
        addition);
    queue_if_profitable(key, candidate);
    fill_slots();
  }

  void enqueue_profitable_candidates() {
    if (!config.enable_prefetch || frozen) return;
    for (auto& [key, candidate] : candidates)
      queue_if_profitable(key, candidate);
  }

  Status drain(std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    enqueue_profitable_candidates();
    for (;;) {
      complete_ready();
      fill_slots();
      if (pending.empty() && eligible.empty()) return Status::success();
      if (pending.empty()) continue;
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline)
        return {ErrorCode::cancelled, "adaptive placement drain timed out"};
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      static_cast<void>(pending.front().handle.wait_for(remaining));
    }
  }

  ExpertCache& cache;
  AdaptivePlacementConfig config;
  double cpu_ns_per_selection;
  std::uint64_t epoch{};
  std::map<ExpertKey, Candidate> candidates;
  std::priority_queue<Eligible> eligible;
  std::vector<Pending> pending;
  AdaptivePlacementTelemetry metrics;
  bool frozen{};
};

AdaptivePlacementPlanner::AdaptivePlacementPlanner(
    ExpertCache& cache, AdaptivePlacementConfig config)
    : impl_(std::make_unique<Impl>(cache, config)) {}

AdaptivePlacementPlanner::~AdaptivePlacementPlanner() = default;

void AdaptivePlacementPlanner::observe_cpu_batch(
    std::uint64_t elapsed_ns, std::uint64_t selections) noexcept {
  impl_->observe(elapsed_ns, selections);
}

void AdaptivePlacementPlanner::observe_routes(
    std::span<const ExpertKey> gpu_resident,
    std::span<const ExpertKey> missing) {
  impl_->observe_routes(gpu_resident, missing);
}

void AdaptivePlacementPlanner::consider(const ExpertKey& key,
                                        const PayloadRecord& record,
                                        std::uint32_t selections,
                                        double routing_score_sum) {
  impl_->consider(key, record, selections, routing_score_sum);
}

void AdaptivePlacementPlanner::poll() {
  impl_->complete_ready();
  impl_->fill_slots();
}

Status AdaptivePlacementPlanner::drain(std::chrono::milliseconds timeout) {
  return impl_->drain(timeout);
}

Status AdaptivePlacementPlanner::quiesce(std::chrono::milliseconds timeout) {
  auto status = impl_->drain(timeout);
  if (status.ok()) impl_->frozen = true;
  return status;
}

void AdaptivePlacementPlanner::resume() noexcept { impl_->frozen = false; }

bool AdaptivePlacementPlanner::frozen() const noexcept {
  return impl_->frozen;
}

AdaptivePlacementTelemetry AdaptivePlacementPlanner::telemetry() const noexcept {
  return impl_->metrics;
}

}  // namespace expert::runtime
