#pragma once

#include "expert/runtime/expert_key.hpp"
#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace expert::runtime {

struct RouteCensusConfig final {
  std::uint64_t model_id{};
  Sha256Digest model_content_hash{};
  std::uint32_t encoding_abi{};
  std::uint32_t layer_count{};
  std::uint32_t experts_per_layer{};
  std::uint32_t route_width{};
  // Heat halves after this many completed routed layers without another hit.
  std::uint64_t decay_interval_observations{4096U};
};

struct RouteCensusWarmEntry final {
  ExpertKey key;
  std::uint64_t total_selections{};
  std::uint64_t effective_heat_q20{};
  std::uint64_t last_seen_observation{};
  std::uint64_t cpu_selections{};
  std::uint64_t gpu_selections{};
};

struct RouteCensusPrediction final {
  ExpertKey key;
  std::uint64_t transition_score{};
};

struct RouteCensusSnapshot final {
  std::uint64_t generation{};
  std::uint64_t completed_routes{};
  std::uint64_t total_selections{};
  std::uint64_t consecutive_reuse_selections{};
  std::size_t observed_experts{};
  std::size_t serialized_bytes{};
};

struct RouteCensusLoadResult;

// Fixed-cardinality, model-bound evidence for stable warm placement. The
// complete table is bounded by layer_count * experts_per_layer; no prompt,
// request ID, activation, or unbounded trace is retained.
class RouteCensus final {
 public:
  explicit RouteCensus(RouteCensusConfig config);
  RouteCensus(const RouteCensus&) = delete;
  RouteCensus& operator=(const RouteCensus&) = delete;

  [[nodiscard]] Status observe(
      std::uint32_t layer, std::span<const std::uint32_t> routed_experts,
      std::span<const std::uint32_t> cpu_experts = {}) noexcept;

  // Returns the globally hottest bounded set. maximum_per_layer=0 means no
  // per-layer cap; a nonzero cap prevents one layer from consuming the budget.
  [[nodiscard]] std::vector<RouteCensusWarmEntry> stable_warm_set(
      std::size_t maximum_entries,
      std::size_t maximum_per_layer = 0U) const;
  // Bounded first-order prediction for the next route of the same layer.
  // Scores aggregate transitions from every expert in current_route. Experts
  // already present in current_route are omitted because the scheduler keeps
  // them protected as the deterministic working set.
  [[nodiscard]] std::vector<RouteCensusPrediction> predict_next(
      std::uint32_t layer, std::span<const std::uint32_t> current_route,
      std::size_t maximum_entries) const;
  [[nodiscard]] RouteCensusSnapshot snapshot() const noexcept;
  [[nodiscard]] const RouteCensusConfig& config() const noexcept {
    return config_;
  }

  // Double-buffered persistence. prefix.0 and prefix.1 are independently
  // authenticated; load selects the newest valid generation.
  [[nodiscard]] Status save(const std::filesystem::path& prefix) noexcept;
  [[nodiscard]] static RouteCensusLoadResult load(
      const std::filesystem::path& prefix,
      const RouteCensusConfig& expected) noexcept;

 private:
  struct Cell final {
    std::uint64_t total{};
    std::uint64_t heat_q20{};
    std::uint64_t heat_observation{};
    std::uint64_t last_seen{};
    std::uint64_t cpu{};
    std::uint64_t gpu{};
  };
  struct LayerState final {
    std::uint64_t observations{};
    std::uint64_t consecutive_reuse{};
    std::vector<std::uint32_t> previous_route;
  };

  [[nodiscard]] std::uint64_t effective_heat(const Cell& cell) const noexcept;
  [[nodiscard]] std::vector<std::byte> serialize(
      std::uint64_t generation) const;
  [[nodiscard]] static RouteCensusLoadResult decode_file(
      const std::filesystem::path& path,
      const RouteCensusConfig& expected);

  RouteCensusConfig config_;
  mutable std::mutex mutex_;
  std::vector<Cell> cells_;
  std::vector<LayerState> layers_;
  std::vector<std::uint32_t> transitions_;
  std::uint64_t generation_{};
  std::uint64_t observation_{};
  std::uint64_t completed_routes_{};
  std::uint64_t total_selections_{};
  std::uint64_t consecutive_reuse_selections_{};
};

struct RouteCensusLoadResult final {
  Status status;
  std::unique_ptr<RouteCensus> census;
  // Expert indices are semantic model data, while the namespace is a runtime
  // virtual-address assignment. A persisted census may therefore be rebound
  // when every immutable model, encoding, and geometry field still matches.
  bool namespace_rebound{};
};

}  // namespace expert::runtime
