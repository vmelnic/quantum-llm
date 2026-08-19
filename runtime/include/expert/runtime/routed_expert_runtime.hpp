#pragma once

#include "expert/runtime/expert_catalog.hpp"
#include "expert/runtime/expert_store.hpp"
#include "expert/runtime/model_descriptor.hpp"
#include "expert/runtime/route_census.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace expert::runtime {

// Common control plane for one sparse component. It owns no model kernels and
// makes no assumptions about attention, routing math, or model family.
class RoutedExpertRuntime final {
 public:
  RoutedExpertRuntime(RoutedExpertComponentDescriptor component,
                      Sha256Digest model_content_hash,
                      const ExpertCatalog& catalog, ExpertCache& cache);
  RoutedExpertRuntime(RoutedExpertComponentDescriptor component,
                      Sha256Digest model_content_hash,
                      const ExpertCatalog& catalog, ExpertCache& cache,
                      IExpertStore& store);
  RoutedExpertRuntime(const RoutedExpertRuntime&) = delete;
  RoutedExpertRuntime& operator=(const RoutedExpertRuntime&) = delete;

  [[nodiscard]] const RoutedExpertComponentDescriptor& component()
      const noexcept {
    return component_;
  }
  [[nodiscard]] const ExpertCatalog& catalog() const noexcept {
    return catalog_;
  }
  [[nodiscard]] ExpertCache& cache() const noexcept { return cache_; }
  [[nodiscard]] IExpertStore& store() noexcept { return *store_; }
  [[nodiscard]] ExpertKey key(std::uint32_t layer,
                              std::uint32_t expert) const noexcept;
  [[nodiscard]] const PayloadRecord* record(
      std::uint32_t layer, std::uint32_t expert) const noexcept;
  [[nodiscard]] Status validate_route(
      std::uint32_t layer,
      std::span<const std::uint32_t> experts) const noexcept;
  [[nodiscard]] ExpertResolveHandle resolve(
      std::uint32_t layer, std::span<const std::uint32_t> experts,
      ExpertResolveTarget target,
      ExpertAcquireOptions options = {});
  [[nodiscard]] RouteCensusConfig census_config(
      std::uint64_t decay_interval_observations = 4096U) const noexcept;

 private:
  RoutedExpertComponentDescriptor component_;
  Sha256Digest model_content_hash_{};
  const ExpertCatalog& catalog_;
  ExpertCache& cache_;
  std::unique_ptr<IExpertStore> owned_store_;
  IExpertStore* store_{};
};

}  // namespace expert::runtime
