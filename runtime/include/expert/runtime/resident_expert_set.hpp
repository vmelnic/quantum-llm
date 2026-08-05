#pragma once

#include "expert/runtime/expert_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace expert::runtime {

struct ResidentExpertSpec final {
  ExpertKey key;
  PayloadRecord record;
};

// Owns long-lived cache leases for always-active model state. Loading is
// intentionally serial through one fixed staging slot: peak RAM is bounded and
// the complete set either publishes or is released on the first failure.
class ResidentExpertSet final {
 public:
  ResidentExpertSet() = default;
  ResidentExpertSet(const ResidentExpertSet&) = delete;
  ResidentExpertSet& operator=(const ResidentExpertSet&) = delete;
  ResidentExpertSet(ResidentExpertSet&&) noexcept = default;
  ResidentExpertSet& operator=(ResidentExpertSet&&) noexcept = default;

  [[nodiscard]] static Status load(ExpertCache& cache,
                                   std::span<const ResidentExpertSpec> specs,
                                   ResidentExpertSet& destination);
  [[nodiscard]] std::size_t size() const noexcept { return leases_.size(); }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  void clear() noexcept;

 private:
  std::vector<ExpertLease> leases_;
  std::uint64_t bytes_{};
};

}  // namespace expert::runtime
