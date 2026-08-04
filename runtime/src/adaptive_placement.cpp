#include "expert/runtime/adaptive_placement.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace expert::runtime {

struct AdaptivePlacementPlanner::Impl final {
  struct Candidate final {
    PayloadRecord record;
    std::uint64_t debt_ns{};
    bool queued{};
    bool pending{};
  };

  struct Pending final {
    ExpertKey key;
    AcquireHandle handle;
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
        config.maximum_inflight_promotions == 0) {
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

  void complete_ready() {
    for (std::size_t index = 0; index < pending.size();) {
      if (pending[index].handle.wait_for(std::chrono::milliseconds(0)) !=
          std::future_status::ready) {
        ++index;
        continue;
      }
      auto result = pending[index].handle.get();
      if (result.status.ok() && result.lease) {
        ++metrics.completed;
      } else {
        ++metrics.failed;
      }
      candidates[pending[index].key].pending = false;
      pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(index));
    }
  }

  void fill_slots() {
    if (frozen) return;
    while (pending.size() < config.maximum_inflight_promotions &&
           !eligible.empty()) {
      const auto key = eligible.front();
      eligible.pop_front();
      auto iterator = candidates.find(key);
      if (iterator == candidates.end()) continue;
      auto& candidate = iterator->second;
      candidate.queued = false;
      if (candidate.pending) continue;
      if (!cache.vram_admission_would_improve(key, candidate.record)) {
        ++metrics.admission_rejected;
        continue;
      }
      pending.push_back({key, cache.acquire(key, candidate.record)});
      candidate.debt_ns = 0;
      candidate.pending = true;
      ++metrics.scheduled;
      metrics.scheduled_bytes += candidate.record.stored_bytes;
      complete_ready();
    }
  }

  void consider(const ExpertKey& key, const PayloadRecord& record,
                std::uint32_t selections) {
    complete_ready();
    if (selections == 0) return;
    ++metrics.considered;
    auto& candidate = candidates[key];
    candidate.record = record;
    const auto addition = static_cast<std::uint64_t>(
        cpu_ns_per_selection * static_cast<double>(selections));
    candidate.debt_ns += std::min(
        std::numeric_limits<std::uint64_t>::max() - candidate.debt_ns,
        addition);
    const auto upload_cost = static_cast<std::uint64_t>(
        static_cast<double>(record.stored_bytes) * 1.0e9 /
        config.conservative_h2d_bytes_per_second * config.admission_margin);
    if (!frozen && candidate.debt_ns >= upload_cost && !candidate.queued &&
        !candidate.pending) {
      candidate.queued = true;
      eligible.push_back(key);
    }
    fill_slots();
  }

  void enqueue_profitable_candidates() {
    if (frozen) return;
    for (auto& [key, candidate] : candidates) {
      const auto upload_cost = static_cast<std::uint64_t>(
          static_cast<double>(candidate.record.stored_bytes) * 1.0e9 /
          config.conservative_h2d_bytes_per_second *
          config.admission_margin);
      if (candidate.debt_ns >= upload_cost && !candidate.queued &&
          !candidate.pending) {
        candidate.queued = true;
        eligible.push_back(key);
      }
    }
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
      if (now >= deadline) {
        return {ErrorCode::cancelled, "adaptive placement drain timed out"};
      }
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - now);
      static_cast<void>(pending.front().handle.wait_for(remaining));
    }
    return Status::success();
  }

  ExpertCache& cache;
  AdaptivePlacementConfig config;
  double cpu_ns_per_selection;
  std::map<ExpertKey, Candidate> candidates;
  std::deque<ExpertKey> eligible;
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

void AdaptivePlacementPlanner::consider(const ExpertKey& key,
                                        const PayloadRecord& record,
                                        std::uint32_t selections) {
  impl_->consider(key, record, selections);
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
