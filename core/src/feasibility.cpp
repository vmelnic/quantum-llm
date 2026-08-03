#include "expert/core/feasibility.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace expert::core {
namespace {

constexpr std::uint64_t kDegradedMarginPpm = 100'000;  // 10% headroom.

std::uint64_t SaturatingAdd(std::uint64_t left, std::uint64_t right) {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return left + right;
}

std::uint64_t SaturatingMultiply(std::uint64_t left, std::uint64_t right) {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return left * right;
}

std::uint64_t CeilDivide(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0U) throw std::invalid_argument("division by zero");
    return numerator / denominator + static_cast<std::uint64_t>(numerator % denominator != 0U);
}

std::uint64_t MultiplyDivideCeil(std::uint64_t value, std::uint64_t multiplier,
                                 std::uint64_t divisor) {
    const auto quotient = value / divisor;
    const auto remainder = value % divisor;
    const auto whole = SaturatingMultiply(quotient, multiplier);
    const auto partial = CeilDivide(SaturatingMultiply(remainder, multiplier), divisor);
    return SaturatingAdd(whole, partial);
}

std::uint64_t RequiredAvoidancePpm(std::uint64_t active_bytes,
                                   std::uint64_t byte_budget) {
    if (active_bytes == 0U || byte_budget >= active_bytes) return 0U;
    return MultiplyDivideCeil(active_bytes - byte_budget, kPartsPerMillion,
                              active_bytes);
}

std::string EscapeJson(std::string_view value) {
    std::string result;
    result.push_back('"');
    for (const char c : value) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result.push_back(c); break;
        }
    }
    result.push_back('"');
    return result;
}

bool HasLowMargin(const Constraint& constraint) {
    if (!constraint.passes || constraint.available == 0U) return false;
    const auto headroom = constraint.available - constraint.required;
    return MultiplyDivideCeil(headroom, kPartsPerMillion, constraint.available) <
           kDegradedMarginPpm;
}

}  // namespace

FeasibilityDecision EvaluateFeasibility(const ManifestResources& manifest,
                                        const HardwareResources& hardware,
                                        const SloRequest& request) {
    if (request.target_tokens_per_second_milli == 0U) {
        throw std::invalid_argument("target_tokens_per_second_milli must be positive");
    }
    if (request.concurrency == 0U) {
        throw std::invalid_argument("concurrency must be positive");
    }
    if (request.expected_vram_hit_ppm > kPartsPerMillion ||
        request.expected_ram_hit_ppm > kPartsPerMillion ||
        request.expected_vram_hit_ppm + request.expected_ram_hit_ppm >
            kPartsPerMillion) {
        throw std::invalid_argument("VRAM and RAM hit fractions must sum to <= 1000000 ppm");
    }
    if (manifest.average_expert_record_bytes == 0U &&
        manifest.total_expert_bytes != 0U) {
        throw std::invalid_argument("average_expert_record_bytes must be positive");
    }

    FeasibilityDecision result;
    result.cold_bytes_per_token_budget =
        SaturatingMultiply(hardware.sustained_storage_read_bytes_per_second, 1000U) /
        request.target_tokens_per_second_milli;
    result.h2d_bytes_per_token_budget =
        SaturatingMultiply(hardware.sustained_h2d_bytes_per_second, 1000U) /
        request.target_tokens_per_second_milli;

    const auto cold_fraction_ppm = kPartsPerMillion - request.expected_vram_hit_ppm -
                                   request.expected_ram_hit_ppm;
    const auto h2d_fraction_ppm = kPartsPerMillion - request.expected_vram_hit_ppm;
    result.expected_cold_bytes_per_token = MultiplyDivideCeil(
        manifest.active_expert_bytes_per_token, cold_fraction_ppm, kPartsPerMillion);
    result.expected_h2d_bytes_per_token = MultiplyDivideCeil(
        manifest.active_expert_bytes_per_token, h2d_fraction_ppm, kPartsPerMillion);
    result.required_storage_avoidance_ppm = RequiredAvoidancePpm(
        manifest.active_expert_bytes_per_token, result.cold_bytes_per_token_budget);
    result.required_vram_hit_ppm = RequiredAvoidancePpm(
        manifest.active_expert_bytes_per_token, result.h2d_bytes_per_token_budget);
    result.theoretical_minimum_concurrency =
        result.cold_bytes_per_token_budget == 0U
            ? std::numeric_limits<std::uint64_t>::max()
            : CeilDivide(manifest.active_expert_bytes_per_token,
                         result.cold_bytes_per_token_budget);
    if (manifest.average_expert_record_bytes != 0U) {
        result.vram_expert_capacity_count = request.vram_expert_budget_bytes /
                                            manifest.average_expert_record_bytes;
        result.ram_expert_capacity_count = request.ram_expert_budget_bytes /
                                           manifest.average_expert_record_bytes;
    }

    const auto kv_bytes = SaturatingMultiply(request.kv_bytes_per_request,
                                             request.concurrency);
    const auto vram_resident = SaturatingAdd(
        SaturatingAdd(manifest.resident_dense_bytes, request.minimum_workspace_bytes),
        kv_bytes);
    const auto vram_with_experts = SaturatingAdd(vram_resident,
                                                 request.vram_expert_budget_bytes);
    const auto ram_required = SaturatingAdd(request.minimum_staging_bytes,
                                            request.ram_expert_budget_bytes);
    const auto storage_required_bps = MultiplyDivideCeil(
        result.expected_cold_bytes_per_token,
        request.target_tokens_per_second_milli, 1000U);
    const auto h2d_required_bps = MultiplyDivideCeil(
        result.expected_h2d_bytes_per_token,
        request.target_tokens_per_second_milli, 1000U);

    auto add = [&result](std::string resource, std::string formula,
                         std::uint64_t required, std::uint64_t available) {
        result.constraints.push_back(Constraint{std::move(resource), std::move(formula),
                                                required, available,
                                                required <= available});
    };
    add("vram_resident", "dense + workspace + kv_per_request * concurrency <= available_vram",
        vram_resident, hardware.vram_available_bytes);
    add("vram_total", "resident_vram + vram_expert_budget <= available_vram",
        vram_with_experts, hardware.vram_available_bytes);
    add("ram", "staging + ram_expert_budget <= available_ram",
        ram_required, hardware.available_ram_bytes);
    add("disk_capacity", "container_file_bytes <= disk_free_bytes",
        manifest.total_file_bytes, hardware.disk_free_bytes);
    add("storage_bandwidth", "cold_bytes_per_token * target_tokens_per_second <= sustained_storage_read_bytes_per_second",
        storage_required_bps, hardware.sustained_storage_read_bytes_per_second);
    add("pcie_h2d_bandwidth", "h2d_bytes_per_token * target_tokens_per_second <= sustained_h2d_bytes_per_second",
        h2d_required_bps, hardware.sustained_h2d_bytes_per_second);

    const auto failure = std::find_if(result.constraints.begin(), result.constraints.end(),
                                      [](const Constraint& value) { return !value.passes; });
    if (failure != result.constraints.end()) {
        result.status = FeasibilityStatus::kImpossible;
        result.limiting_resource = failure->resource;
        return result;
    }

    const bool modeled_bandwidth = !hardware.storage_bandwidth_measured ||
                                   !hardware.h2d_bandwidth_measured;
    const auto low_margin = std::find_if(result.constraints.begin(), result.constraints.end(),
                                         HasLowMargin);
    if (modeled_bandwidth || low_margin != result.constraints.end()) {
        result.status = FeasibilityStatus::kDegraded;
        result.limiting_resource = modeled_bandwidth
            ? "unmeasured_bandwidth"
            : low_margin->resource;
    } else {
        result.status = FeasibilityStatus::kFeasible;
        result.limiting_resource = "none";
    }
    return result;
}

const char* ToString(FeasibilityStatus status) noexcept {
    switch (status) {
        case FeasibilityStatus::kFeasible: return "feasible";
        case FeasibilityStatus::kDegraded: return "degraded";
        case FeasibilityStatus::kImpossible: return "impossible";
    }
    return "impossible";
}

std::string DecisionToCanonicalJson(const FeasibilityDecision& decision) {
    std::ostringstream output;
    output << "{\"cold_bytes_per_token_budget\":"
           << decision.cold_bytes_per_token_budget
           << ",\"constraints\":[";
    for (std::size_t index = 0; index < decision.constraints.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& item = decision.constraints[index];
        output << "{\"available\":" << item.available
               << ",\"formula\":" << EscapeJson(item.formula)
               << ",\"passes\":" << (item.passes ? "true" : "false")
               << ",\"required\":" << item.required
               << ",\"resource\":" << EscapeJson(item.resource) << '}';
    }
    output << "],\"expected_cold_bytes_per_token\":"
           << decision.expected_cold_bytes_per_token
           << ",\"expected_h2d_bytes_per_token\":"
           << decision.expected_h2d_bytes_per_token
           << ",\"h2d_bytes_per_token_budget\":"
           << decision.h2d_bytes_per_token_budget
           << ",\"limiting_resource\":" << EscapeJson(decision.limiting_resource)
           << ",\"ram_expert_capacity_count\":"
           << decision.ram_expert_capacity_count
           << ",\"required_storage_avoidance_ppm\":"
           << decision.required_storage_avoidance_ppm
           << ",\"required_vram_hit_ppm\":"
           << decision.required_vram_hit_ppm
           << ",\"status\":" << EscapeJson(ToString(decision.status))
           << ",\"theoretical_minimum_concurrency\":"
           << decision.theoretical_minimum_concurrency
           << ",\"vram_expert_capacity_count\":"
           << decision.vram_expert_capacity_count << '}';
    return output.str();
}

}  // namespace expert::core
