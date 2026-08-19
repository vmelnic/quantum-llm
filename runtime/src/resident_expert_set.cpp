#include "expert/runtime/resident_expert_set.hpp"

#include <set>
#include <utility>

namespace expert::runtime {

Status ResidentExpertSet::load(ExpertCache& cache,
                               std::span<const ResidentExpertSpec> specs,
                               ResidentExpertSet& destination) {
  if (specs.empty() || destination.size() != 0U) {
    return {ErrorCode::invalid_argument,
            "resident expert load requires non-empty specs and destination"};
  }
  std::set<ExpertKey> keys;
  for (const auto& spec : specs) {
    if (!keys.insert(spec.key).second) {
      return {ErrorCode::invalid_argument,
              "resident expert set contains a duplicate key"};
    }
  }

  ResidentExpertSet candidate;
  candidate.leases_.reserve(specs.size());
  for (const auto& spec : specs) {
    auto acquired = cache.acquire(
        spec.key, spec.record,
        ExpertAcquireOptions{ExpertRequestPriority::demand, false, false,
                             true})
                        .get();
    if (!acquired.status.ok()) {
      candidate.clear();
      static_cast<void>(cache.trim());
      return acquired.status;
    }
    if (!acquired.lease || acquired.lease.get() == nullptr) {
      candidate.clear();
      static_cast<void>(cache.trim());
      return {ErrorCode::internal,
              "resident expert acquisition returned no allocation"};
    }
    candidate.bytes_ += acquired.lease.get()->bytes();
    candidate.leases_.push_back(std::move(acquired.lease));
  }
  destination = std::move(candidate);
  return Status::success();
}

void ResidentExpertSet::clear() noexcept {
  leases_.clear();
  bytes_ = 0U;
}

}  // namespace expert::runtime
