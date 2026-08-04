#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace expert::core {

inline constexpr std::uint64_t kPartsPerMillion = 1'000'000;

struct ManifestResources {
    std::string manifest_sha256;
    std::uint64_t total_file_bytes{};
    std::uint64_t resident_dense_bytes{};
    std::uint64_t total_expert_bytes{};
    std::uint64_t active_expert_bytes_per_token{};
    std::uint64_t average_expert_record_bytes{};
};

struct HardwareResources {
    std::uint64_t total_ram_bytes{};
    std::uint64_t available_ram_bytes{};
    std::uint64_t vram_total_bytes{};
    std::uint64_t vram_available_bytes{};
    std::uint64_t disk_free_bytes{};
    std::uint64_t sustained_storage_read_bytes_per_second{};
    std::uint64_t sustained_h2d_bytes_per_second{};
    bool storage_bandwidth_measured{};
    bool h2d_bandwidth_measured{};
};

struct SloRequest {
    // Milli-tokens/s keeps planning integer-only (10 tok/s == 10000).
    std::uint64_t target_tokens_per_second_milli{};
    std::uint64_t concurrency{};
    std::uint64_t vram_expert_budget_bytes{};
    std::uint64_t ram_expert_budget_bytes{};
    std::uint64_t expected_vram_hit_ppm{};
    std::uint64_t expected_ram_hit_ppm{};
    std::uint64_t minimum_workspace_bytes{};
    std::uint64_t minimum_staging_bytes{};
    std::uint64_t kv_bytes_per_request{};
    // Optional evidence from the same model/backend. Zero means unknown.
    std::uint64_t measured_non_io_nanoseconds_per_token{};
};

enum class FeasibilityStatus { kFeasible, kDegraded, kImpossible };

struct Constraint {
    std::string resource;
    std::string formula;
    std::uint64_t required{};
    std::uint64_t available{};
    bool passes{};
};

struct FeasibilityDecision {
    FeasibilityStatus status{FeasibilityStatus::kImpossible};
    std::string limiting_resource;
    std::uint64_t cold_bytes_per_token_budget{};
    std::uint64_t h2d_bytes_per_token_budget{};
    std::uint64_t expected_cold_bytes_per_token{};
    std::uint64_t expected_h2d_bytes_per_token{};
    std::uint64_t required_storage_avoidance_ppm{};
    std::uint64_t required_vram_hit_ppm{};
    std::uint64_t theoretical_minimum_concurrency{};
    std::uint64_t vram_expert_capacity_count{};
    std::uint64_t ram_expert_capacity_count{};
    std::uint64_t measured_non_io_ceiling_tokens_per_second_milli{};
    std::vector<Constraint> constraints;
};

[[nodiscard]] FeasibilityDecision EvaluateFeasibility(
    const ManifestResources& manifest,
    const HardwareResources& hardware,
    const SloRequest& request);

[[nodiscard]] const char* ToString(FeasibilityStatus status) noexcept;
[[nodiscard]] std::string DecisionToCanonicalJson(
    const FeasibilityDecision& decision);

}  // namespace expert::core
