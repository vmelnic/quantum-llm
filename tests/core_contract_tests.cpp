#include "expert/core/feasibility.hpp"
#include "expert/core/json.hpp"
#include "expert/core/sha256.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void TestCanonicalHash() {
    constexpr auto first = R"({ "z": [3, 2, 1], "a": {"enabled":true,"bytes":807403520} })";
    constexpr auto second = R"({"a":{"bytes":807403520,"enabled":true},"z":[3,2,1]})";
    const auto canonical_first = expert::core::json::Canonicalize(first);
    const auto canonical_second = expert::core::json::Canonicalize(second);
    Check(canonical_first == canonical_second,
          "equivalent manifests must have identical canonical bytes");
    Check(expert::core::Sha256Hex(canonical_first) ==
              expert::core::Sha256Hex(canonical_second),
          "equivalent manifests must have identical hashes");
    Check(expert::core::Sha256Hex("abc") ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 implementation failed the standard abc vector");

    bool rejected_duplicate = false;
    try {
        static_cast<void>(expert::core::json::Parse(R"({"a":1,"a":2})"));
    } catch (const expert::core::json::Error&) {
        rejected_duplicate = true;
    }
    Check(rejected_duplicate, "duplicate manifest keys must be rejected");
}

expert::core::ManifestResources OlmoeManifest() {
    expert::core::ManifestResources manifest;
    manifest.manifest_sha256 = std::string(64, '0');
    manifest.total_file_bytes = 7'416'456'383ULL;
    manifest.resident_dense_bytes = 953'806'399ULL;
    manifest.total_expert_bytes = 6'459'228'160ULL;
    manifest.active_expert_bytes_per_token = 807'403'520ULL;
    manifest.average_expert_record_bytes = 6'307'840ULL;
    return manifest;
}

expert::core::HardwareResources Box3090() {
    expert::core::HardwareResources hardware;
    hardware.total_ram_bytes = 67'032'328ULL * 1024ULL;
    hardware.available_ram_bytes = 58'430'572ULL * 1024ULL;
    hardware.vram_total_bytes = 24'576ULL << 20U;
    hardware.vram_available_bytes = 24'576ULL << 20U;
    hardware.disk_free_bytes = 340'901'289'984ULL;
    hardware.sustained_storage_read_bytes_per_second = 560'000'000ULL;
    hardware.sustained_h2d_bytes_per_second = 12'000'000'000ULL;
    hardware.storage_bandwidth_measured = false;
    hardware.h2d_bandwidth_measured = false;
    return hardware;
}

expert::core::SloRequest ColdTenTokensPerSecond() {
    expert::core::SloRequest request;
    request.target_tokens_per_second_milli = 10'000ULL;
    request.concurrency = 1;
    request.vram_expert_budget_bytes = 16ULL << 30U;
    request.ram_expert_budget_bytes = 32ULL << 30U;
    request.expected_vram_hit_ppm = 0;
    request.expected_ram_hit_ppm = 0;
    request.minimum_workspace_bytes = 1ULL << 30U;
    request.minimum_staging_bytes = 1ULL << 30U;
    request.kv_bytes_per_request = 512ULL << 20U;
    return request;
}

void TestImpossibleBandwidthRefusal() {
    const auto first = expert::core::EvaluateFeasibility(
        OlmoeManifest(), Box3090(), ColdTenTokensPerSecond());
    const auto second = expert::core::EvaluateFeasibility(
        OlmoeManifest(), Box3090(), ColdTenTokensPerSecond());
    Check(first.status == expert::core::FeasibilityStatus::kImpossible,
          "cold OLMoE must be refused at 10 tok/s on SATA");
    Check(first.limiting_resource == "storage_bandwidth",
          "refusal must identify storage bandwidth");
    Check(first.cold_bytes_per_token_budget == 56'000'000ULL,
          "cold budget must equal B/T");
    Check(first.expected_cold_bytes_per_token == 807'403'520ULL,
          "cold bytes must preserve exact manifest value");
    Check(first.required_storage_avoidance_ppm == 930'642ULL,
          "required avoidance must be reported as an exact ceiling in ppm");
    Check(first.theoretical_minimum_concurrency == 15ULL,
          "minimum perfect-reuse concurrency must be explicit");
    const auto first_json = expert::core::DecisionToCanonicalJson(first);
    const auto second_json = expert::core::DecisionToCanonicalJson(second);
    Check(first_json == second_json, "same inputs must yield same decision bytes");
    Check(expert::core::Sha256Hex(first_json) == expert::core::Sha256Hex(second_json),
          "same inputs must yield same decision hash");
}

void TestHotPlanFitsButUnmeasuredIsDegraded() {
    auto request = ColdTenTokensPerSecond();
    request.expected_vram_hit_ppm = 950'000ULL;
    const auto decision = expert::core::EvaluateFeasibility(
        OlmoeManifest(), Box3090(), request);
    Check(decision.status == expert::core::FeasibilityStatus::kDegraded,
          "an unmeasured but numerically fitting plan must be degraded");
    Check(decision.limiting_resource == "unmeasured_bandwidth",
          "modeled bandwidth must remain visible");
}

void TestHardVramRefusalPrecedesBandwidth() {
    auto manifest = OlmoeManifest();
    manifest.resident_dense_bytes = 30ULL << 30U;
    const auto decision = expert::core::EvaluateFeasibility(
        manifest, Box3090(), ColdTenTokensPerSecond());
    Check(decision.status == expert::core::FeasibilityStatus::kImpossible,
          "oversized dense allocation must be refused");
    Check(decision.limiting_resource == "vram_resident",
          "first hard limiting resource must be stable");
}

void TestMeasuredNonIoCeilingRefusesUnreachableSlo() {
    auto request = ColdTenTokensPerSecond();
    request.expected_vram_hit_ppm = 950'000ULL;
    request.measured_non_io_nanoseconds_per_token = 89'760'000ULL;
    request.target_tokens_per_second_milli = 30'000ULL;
    auto hardware = Box3090();
    hardware.storage_bandwidth_measured = true;
    hardware.h2d_bandwidth_measured = true;
    const auto decision = expert::core::EvaluateFeasibility(
        OlmoeManifest(), hardware, request);
    Check(decision.status == expert::core::FeasibilityStatus::kImpossible,
          "measured compute/control lower bound must refuse unreachable SLO");
    Check(decision.limiting_resource == "measured_non_io_ceiling",
          "measured non-I/O ceiling must identify the limiting resource");
    Check(decision.measured_non_io_ceiling_tokens_per_second_milli == 11'140ULL,
          "measured non-I/O tok/s ceiling must be explicit");
}

}  // namespace

int main() {
    try {
        TestCanonicalHash();
        TestImpossibleBandwidthRefusal();
        TestHotPlanFitsButUnmeasuredIsDegraded();
        TestHardVramRefusalPrecedesBandwidth();
        TestMeasuredNonIoCeilingRefusesUnreachableSlo();
        std::cout << "P0 core contract tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
