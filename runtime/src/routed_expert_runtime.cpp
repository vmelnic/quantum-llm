#include "expert/runtime/routed_expert_runtime.hpp"

#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace expert::runtime {

RoutedExpertRuntime::RoutedExpertRuntime(
    RoutedExpertComponentDescriptor component, Sha256Digest model_content_hash,
    const ExpertCatalog& catalog, ExpertCache& cache)
    : component_(std::move(component)),
      model_content_hash_(model_content_hash), catalog_(catalog), cache_(cache),
      owned_store_(std::make_unique<LocalExpertStore>(cache)),
      store_(owned_store_.get()) {
  const auto status =
      validate_routed_component(component_, component_.hidden_size);
  if (!status.ok() || catalog_.layer_count() != component_.layer_count ||
      catalog_.experts_per_layer() != component_.experts_per_layer ||
      catalog_.size() != expert_table_entries(component_))
    throw std::invalid_argument(
        status.ok() ? "expert catalog does not match routed component"
                    : std::string(status.message()));
}

RoutedExpertRuntime::RoutedExpertRuntime(
    RoutedExpertComponentDescriptor component, Sha256Digest model_content_hash,
    const ExpertCatalog& catalog, ExpertCache& cache, IExpertStore& store)
    : component_(std::move(component)),
      model_content_hash_(model_content_hash), catalog_(catalog), cache_(cache),
      store_(&store) {
  const auto status =
      validate_routed_component(component_, component_.hidden_size);
  if (!status.ok() || catalog_.layer_count() != component_.layer_count ||
      catalog_.experts_per_layer() != component_.experts_per_layer ||
      catalog_.size() != expert_table_entries(component_))
    throw std::invalid_argument(
        status.ok() ? "expert catalog does not match routed component"
                    : std::string(status.message()));
}

ExpertKey RoutedExpertRuntime::key(std::uint32_t layer,
                                   std::uint32_t expert) const noexcept {
  return {component_.namespace_id, layer, expert, component_.encoding_abi};
}

const PayloadRecord* RoutedExpertRuntime::record(
    std::uint32_t layer, std::uint32_t expert) const noexcept {
  return catalog_.find(layer, expert);
}

Status RoutedExpertRuntime::validate_route(
    std::uint32_t layer,
    std::span<const std::uint32_t> experts) const noexcept {
  if (layer >= component_.layer_count ||
      experts.size() != component_.route_width)
    return {ErrorCode::invalid_argument, "routed expert selection has wrong shape"};
  std::set<std::uint32_t> unique;
  for (const auto expert : experts) {
    if (expert >= component_.experts_per_layer || !unique.insert(expert).second)
      return {ErrorCode::invalid_argument,
              "routed expert selection has an invalid member"};
  }
  return Status::success();
}

ExpertResolveHandle RoutedExpertRuntime::resolve(
    std::uint32_t layer, std::span<const std::uint32_t> experts,
    ExpertResolveTarget target, ExpertAcquireOptions options) {
  if (layer >= component_.layer_count || experts.empty()) return {};
  std::vector<ExpertResolveRequest> requests;
  requests.reserve(experts.size());
  std::set<std::uint32_t> unique;
  for (const auto expert : experts) {
    const auto* payload = record(layer, expert);
    if (payload == nullptr || !unique.insert(expert).second) {
      return {};
    }
    requests.push_back({key(layer, expert), *payload, target, options});
  }
  return store_->resolve(requests);
}

RouteCensusConfig RoutedExpertRuntime::census_config(
    std::uint64_t decay_interval_observations) const noexcept {
  return {component_.namespace_id, model_content_hash_, component_.encoding_abi,
          component_.layer_count, component_.experts_per_layer,
          component_.route_width, decay_interval_observations};
}

}  // namespace expert::runtime
