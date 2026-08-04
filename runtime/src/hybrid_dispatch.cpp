#include "expert/runtime/hybrid_dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace expert::runtime {
namespace {

std::uint64_t saturating_add(std::uint64_t left,
                             std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

std::uint64_t rounded_cost(double value) noexcept {
  if (!std::isfinite(value) ||
      value >= static_cast<double>(std::numeric_limits<std::uint64_t>::max()))
    return std::numeric_limits<std::uint64_t>::max();
  return value <= 0.0 ? 0 : static_cast<std::uint64_t>(std::ceil(value));
}

void update_ewma(double& estimate, double observed, double alpha) noexcept {
  if (!std::isfinite(observed) || observed <= 0.0) return;
  estimate = estimate * (1.0 - alpha) + observed * alpha;
}

}  // namespace

HybridDispatchPlanner::HybridDispatchPlanner(HybridDispatchConfig config)
    : config_(config) {
  if (!std::isfinite(config_.initial_cpu_ns_per_selection) ||
      config_.initial_cpu_ns_per_selection <= 0.0 ||
      !std::isfinite(config_.initial_gpu_ns_per_selection) ||
      config_.initial_gpu_ns_per_selection <= 0.0 ||
      !std::isfinite(config_.initial_h2d_bytes_per_second) ||
      config_.initial_h2d_bytes_per_second <= 0.0 ||
      !std::isfinite(config_.observation_ewma_alpha) ||
      config_.observation_ewma_alpha <= 0.0 ||
      config_.observation_ewma_alpha > 1.0 ||
      config_.maximum_candidates == 0 ||
      config_.maximum_trace_decisions == 0) {
    throw std::invalid_argument("invalid hybrid dispatch configuration");
  }
  telemetry_.cpu_ns_per_selection = config_.initial_cpu_ns_per_selection;
  telemetry_.gpu_ns_per_selection = config_.initial_gpu_ns_per_selection;
  telemetry_.h2d_bytes_per_second = config_.initial_h2d_bytes_per_second;
  trace_.reserve(config_.maximum_trace_decisions);
}

void HybridDispatchPlanner::observe_cpu(std::uint64_t elapsed_ns,
                                        std::uint64_t selections) noexcept {
  if (elapsed_ns == 0 || selections == 0) return;
  update_ewma(telemetry_.cpu_ns_per_selection,
              static_cast<double>(elapsed_ns) / selections,
              config_.observation_ewma_alpha);
}

void HybridDispatchPlanner::observe_gpu(std::uint64_t elapsed_ns,
                                        std::uint64_t selections) noexcept {
  if (elapsed_ns == 0 || selections == 0) return;
  update_ewma(telemetry_.gpu_ns_per_selection,
              static_cast<double>(elapsed_ns) / selections,
              config_.observation_ewma_alpha);
}

void HybridDispatchPlanner::observe_h2d(std::uint64_t elapsed_ns,
                                        std::uint64_t bytes) noexcept {
  if (elapsed_ns == 0 || bytes == 0) return;
  update_ewma(telemetry_.h2d_bytes_per_second,
              static_cast<double>(bytes) * 1.0e9 /
                  static_cast<double>(elapsed_ns),
              config_.observation_ewma_alpha);
}

HybridDispatchPlan HybridDispatchPlanner::plan(
    std::span<const HybridDispatchCandidate> candidates) {
  HybridDispatchPlan result;
  if (candidates.empty()) {
    result.status = {ErrorCode::invalid_argument,
                     "hybrid dispatch requires at least one candidate"};
    ++telemetry_.rejected_plans;
    return result;
  }
  if (candidates.size() > config_.maximum_candidates) {
    result.status = {ErrorCode::backpressure,
                     "hybrid dispatch candidate bound exceeded"};
    ++telemetry_.rejected_plans;
    return result;
  }

  std::set<std::uint32_t> unique_experts;
  std::vector<HybridDispatchCandidate> fixed;
  std::vector<HybridDispatchCandidate> flexible;
  fixed.reserve(candidates.size());
  flexible.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (candidate.selections == 0 ||
        (!candidate.cpu_available && !candidate.gpu_available &&
         !candidate.gpu_resident) ||
        !unique_experts.insert(candidate.expert).second) {
      result.status = {ErrorCode::invalid_argument,
                       "invalid or duplicate hybrid dispatch candidate"};
      ++telemetry_.rejected_plans;
      return result;
    }
    if (!candidate.gpu_resident && candidate.cpu_available &&
        candidate.gpu_available) {
      flexible.push_back(candidate);
    } else {
      fixed.push_back(candidate);
    }
  }

  const auto cpu_cost = [&](const HybridDispatchCandidate& candidate) {
    return rounded_cost(telemetry_.cpu_ns_per_selection *
                        static_cast<double>(candidate.selections));
  };
  const auto gpu_cost = [&](const HybridDispatchCandidate& candidate) {
    return rounded_cost(telemetry_.gpu_ns_per_selection *
                        static_cast<double>(candidate.selections));
  };
  const auto h2d_cost = [&](const HybridDispatchCandidate& candidate) {
    return rounded_cost(static_cast<double>(candidate.record_bytes) * 1.0e9 /
                        telemetry_.h2d_bytes_per_second);
  };
  const auto record_trace = [&](const HybridDispatchDecision& decision) {
    if (trace_.size() < config_.maximum_trace_decisions) {
      trace_.push_back(decision);
    } else {
      trace_[next_trace_slot_] = decision;
      next_trace_slot_ =
          (next_trace_slot_ + 1) % config_.maximum_trace_decisions;
    }
  };
  const auto append = [&](HybridDispatchDecision decision) {
    result.decisions.push_back(decision);
    record_trace(decision);
    switch (decision.reason) {
      case HybridDispatchReason::resident_gpu:
        ++telemetry_.resident_gpu;
        break;
      case HybridDispatchReason::cpu_only:
        ++telemetry_.cpu_only;
        break;
      case HybridDispatchReason::gpu_only:
        ++telemetry_.gpu_only;
        break;
      case HybridDispatchReason::cpu_lower_critical_path:
        ++telemetry_.cpu_cost_wins;
        break;
      case HybridDispatchReason::gpu_lower_critical_path:
        ++telemetry_.gpu_cost_wins;
        break;
      case HybridDispatchReason::cpu_stable_tie:
        ++telemetry_.stable_ties;
        break;
    }
  };

  std::sort(fixed.begin(), fixed.end(),
            [](const auto& left, const auto& right) {
              return left.expert < right.expert;
            });
  for (const auto& candidate : fixed) {
    if (candidate.gpu_resident) {
      result.projected_gpu_ns =
          saturating_add(result.projected_gpu_ns, gpu_cost(candidate));
      append({candidate.expert, HybridExecutor::gpu_resident,
              HybridDispatchReason::resident_gpu, 0,
              saturating_add(result.projected_h2d_ns,
                             result.projected_gpu_ns)});
    } else if (candidate.cpu_available) {
      result.projected_cpu_ns =
          saturating_add(result.projected_cpu_ns, cpu_cost(candidate));
      append({candidate.expert, HybridExecutor::cpu_local,
              HybridDispatchReason::cpu_only, result.projected_cpu_ns, 0});
    } else {
      result.projected_h2d_ns =
          saturating_add(result.projected_h2d_ns, h2d_cost(candidate));
      result.projected_gpu_ns =
          saturating_add(result.projected_gpu_ns, gpu_cost(candidate));
      append({candidate.expert, HybridExecutor::gpu_upload,
              HybridDispatchReason::gpu_only, 0,
              saturating_add(result.projected_h2d_ns,
                             result.projected_gpu_ns)});
    }
  }

  std::sort(flexible.begin(), flexible.end(),
            [&](const auto& left, const auto& right) {
              const auto left_cost = cpu_cost(left);
              const auto right_cost = cpu_cost(right);
              return left_cost != right_cost ? left_cost > right_cost
                                             : left.expert < right.expert;
            });
  for (const auto& candidate : flexible) {
    const auto cpu_finish =
        saturating_add(result.projected_cpu_ns, cpu_cost(candidate));
    const auto current_gpu_finish =
        saturating_add(result.projected_h2d_ns, result.projected_gpu_ns);
    const auto cpu_critical = std::max(cpu_finish, current_gpu_finish);
    const auto h2d_finish =
        saturating_add(result.projected_h2d_ns, h2d_cost(candidate));
    const auto gpu_compute_finish =
        saturating_add(result.projected_gpu_ns, gpu_cost(candidate));
    const auto gpu_finish = saturating_add(h2d_finish, gpu_compute_finish);
    const auto gpu_critical = std::max(result.projected_cpu_ns, gpu_finish);
    if (gpu_critical < cpu_critical) {
      result.projected_h2d_ns = h2d_finish;
      result.projected_gpu_ns = gpu_compute_finish;
      append({candidate.expert, HybridExecutor::gpu_upload,
              HybridDispatchReason::gpu_lower_critical_path, cpu_critical,
              gpu_critical});
    } else {
      result.projected_cpu_ns = cpu_finish;
      append({candidate.expert, HybridExecutor::cpu_local,
              gpu_critical == cpu_critical
                  ? HybridDispatchReason::cpu_stable_tie
                  : HybridDispatchReason::cpu_lower_critical_path,
              cpu_critical, gpu_critical});
    }
  }

  std::sort(result.decisions.begin(), result.decisions.end(),
            [](const auto& left, const auto& right) {
              return left.expert < right.expert;
            });
  result.projected_critical_ns =
      std::max(result.projected_cpu_ns,
               saturating_add(result.projected_h2d_ns,
                              result.projected_gpu_ns));
  result.status = Status::success();
  ++telemetry_.plans;
  telemetry_.candidates += candidates.size();
  return result;
}

HybridDispatchTelemetry HybridDispatchPlanner::telemetry() const noexcept {
  return telemetry_;
}

std::vector<HybridDispatchDecision> HybridDispatchPlanner::trace() const {
  if (trace_.size() < config_.maximum_trace_decisions || next_trace_slot_ == 0)
    return trace_;
  std::vector<HybridDispatchDecision> ordered;
  ordered.reserve(trace_.size());
  ordered.insert(ordered.end(), trace_.begin() + next_trace_slot_, trace_.end());
  ordered.insert(ordered.end(), trace_.begin(), trace_.begin() + next_trace_slot_);
  return ordered;
}

}  // namespace expert::runtime
