#include "expert/core/feasibility.hpp"
#include "expert/core/json.hpp"
#include "expert/core/sha256.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string ReadFile(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error(std::string("cannot open ") + path);
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        throw std::runtime_error(std::string("cannot read ") + path);
    }
    return contents.str();
}

expert::core::ManifestResources ParseManifest(std::string_view text) {
    using namespace expert::core::json;
    Value document = Parse(text);
    auto& root = document.AsObject("manifest");
    RequireExactKeys(root,
        {"schema", "format", "compatibility", "source", "architecture",
         "model_program", "quantization", "kernel_abi", "alignment",
         "tensors", "experts", "packs", "indexes", "masses",
         "requirements", "tokenizer", "integrity"},
        {"auxiliary_tensors"}, "manifest");
    const auto& model_program = Required(root, "model_program", "manifest")
                                    .AsObject("manifest.model_program");
    RequireExactKeys(model_program, {"format", "path", "bytes", "sha256"},
                     {}, "manifest.model_program");
    if (Required(model_program, "format", "manifest.model_program")
            .AsString("manifest.model_program.format") !=
            "expert-runtime-model-v1" ||
        Required(model_program, "path", "manifest.model_program")
            .AsString("manifest.model_program.path").empty() ||
        Required(model_program, "bytes", "manifest.model_program")
                .AsU64("manifest.model_program.bytes") == 0U ||
        Required(model_program, "sha256", "manifest.model_program")
                .AsString("manifest.model_program.sha256").size() != 64U) {
        throw Error("manifest.model_program is invalid");
    }
    const auto& masses = Required(root, "masses", "manifest")
                             .AsObject("manifest.masses");
    RequireExactKeys(masses,
        {"source_tensor_bytes", "pack_bytes", "dense_bytes", "expert_bytes",
         "active_expert_bytes_per_token", "resident_dense_bytes",
         "host_mapped_dense_bytes"}, {}, "manifest.masses");
    const auto& requirements = Required(root, "requirements", "manifest")
                                   .AsObject("manifest.requirements");
    const auto& expert_entries = Required(root, "experts", "manifest")
                                     .AsArray("manifest.experts");
    if (expert_entries.empty()) throw Error("manifest.experts must not be empty");

    auto& integrity = root.at("integrity").AsObject("manifest.integrity");
    const auto declared_hash = Required(integrity, "content_sha256", "manifest.integrity")
                                   .AsString("manifest.integrity.content_sha256");
    integrity["content_sha256"] = Value(Value::Storage(std::string{}));
    const auto content_hash = expert::core::Sha256Hex(Canonicalize(document));
    if (content_hash != declared_hash) throw Error("manifest content_sha256 mismatch");

    std::uint64_t expert_bytes_sum = 0;
    for (const auto& entry : expert_entries) {
        const auto& object = entry.AsObject("manifest.experts[]");
        const auto bytes = Required(object, "stored_bytes", "manifest.experts[]")
                               .AsU64("manifest.experts[].stored_bytes");
        if (expert_bytes_sum > std::numeric_limits<std::uint64_t>::max() - bytes) {
            throw Error("expert byte sum overflow");
        }
        expert_bytes_sum += bytes;
    }
    expert::core::ManifestResources result;
    result.manifest_sha256 = content_hash;
    result.total_file_bytes = Required(masses, "pack_bytes", "manifest.masses").AsU64("manifest.masses.pack_bytes");
    result.resident_dense_bytes = Required(requirements, "resident_dense_bytes", "manifest.requirements").AsU64("manifest.requirements.resident_dense_bytes");
    result.total_expert_bytes = Required(masses, "expert_bytes", "manifest.masses").AsU64("manifest.masses.expert_bytes");
    result.active_expert_bytes_per_token = Required(masses, "active_expert_bytes_per_token", "manifest.masses").AsU64("manifest.masses.active_expert_bytes_per_token");
    result.average_expert_record_bytes = expert_bytes_sum / expert_entries.size();
    return result;
}

expert::core::HardwareResources ParseHardware(std::string_view text) {
    using namespace expert::core::json;
    const auto document = Parse(text);
    const auto& root = document.AsObject("hardware");
    const auto& planner = Required(root, "planner", "hardware").AsObject("hardware.planner");
    RequireExactKeys(planner,
        {"total_ram_bytes", "available_ram_bytes", "vram_total_bytes",
         "vram_available_bytes", "disk_free_bytes",
         "sustained_storage_read_bytes_per_second",
         "sustained_h2d_bytes_per_second", "storage_bandwidth_measured",
         "h2d_bandwidth_measured"}, {}, "hardware.planner");
    expert::core::HardwareResources result;
    result.total_ram_bytes = Required(planner, "total_ram_bytes", "hardware.planner").AsU64("total_ram_bytes");
    result.available_ram_bytes = Required(planner, "available_ram_bytes", "hardware.planner").AsU64("available_ram_bytes");
    result.vram_total_bytes = Required(planner, "vram_total_bytes", "hardware.planner").AsU64("vram_total_bytes");
    result.vram_available_bytes = Required(planner, "vram_available_bytes", "hardware.planner").AsU64("vram_available_bytes");
    result.disk_free_bytes = Required(planner, "disk_free_bytes", "hardware.planner").AsU64("disk_free_bytes");
    result.sustained_storage_read_bytes_per_second = Required(planner, "sustained_storage_read_bytes_per_second", "hardware.planner").AsU64("sustained_storage_read_bytes_per_second");
    result.sustained_h2d_bytes_per_second = Required(planner, "sustained_h2d_bytes_per_second", "hardware.planner").AsU64("sustained_h2d_bytes_per_second");
    result.storage_bandwidth_measured = Required(planner, "storage_bandwidth_measured", "hardware.planner").AsBool("storage_bandwidth_measured");
    result.h2d_bandwidth_measured = Required(planner, "h2d_bandwidth_measured", "hardware.planner").AsBool("h2d_bandwidth_measured");
    return result;
}

expert::core::SloRequest ParseRequest(std::string_view text) {
    using namespace expert::core::json;
    const auto root = Parse(text).AsObject("request");
    RequireExactKeys(root,
        {"target_tokens_per_second_milli", "concurrency",
         "vram_expert_budget_bytes", "ram_expert_budget_bytes",
         "expected_vram_hit_ppm", "expected_ram_hit_ppm",
         "minimum_workspace_bytes", "minimum_staging_bytes",
         "kv_bytes_per_request"},
        {"measured_non_io_nanoseconds_per_token"}, "request");
    expert::core::SloRequest result;
    result.target_tokens_per_second_milli = Required(root, "target_tokens_per_second_milli", "request").AsU64("target_tokens_per_second_milli");
    result.concurrency = Required(root, "concurrency", "request").AsU64("concurrency");
    result.vram_expert_budget_bytes = Required(root, "vram_expert_budget_bytes", "request").AsU64("vram_expert_budget_bytes");
    result.ram_expert_budget_bytes = Required(root, "ram_expert_budget_bytes", "request").AsU64("ram_expert_budget_bytes");
    result.expected_vram_hit_ppm = Required(root, "expected_vram_hit_ppm", "request").AsU64("expected_vram_hit_ppm");
    result.expected_ram_hit_ppm = Required(root, "expected_ram_hit_ppm", "request").AsU64("expected_ram_hit_ppm");
    result.minimum_workspace_bytes = Required(root, "minimum_workspace_bytes", "request").AsU64("minimum_workspace_bytes");
    result.minimum_staging_bytes = Required(root, "minimum_staging_bytes", "request").AsU64("minimum_staging_bytes");
    result.kv_bytes_per_request = Required(root, "kv_bytes_per_request", "request").AsU64("kv_bytes_per_request");
    if (const auto measured = root.find("measured_non_io_nanoseconds_per_token");
        measured != root.end()) {
        result.measured_non_io_nanoseconds_per_token =
            measured->second.AsU64("measured_non_io_nanoseconds_per_token");
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: expert-feasibility <manifest.json> <hardware.json> <request.json>\n";
        return 64;
    }
    try {
        const auto manifest_text = ReadFile(argv[1]);
        const auto hardware_text = ReadFile(argv[2]);
        const auto request_text = ReadFile(argv[3]);
        const auto manifest = ParseManifest(manifest_text);
        const auto hardware = ParseHardware(hardware_text);
        const auto request = ParseRequest(request_text);
        const auto decision = expert::core::EvaluateFeasibility(manifest, hardware, request);
        const auto plan = expert::core::DecisionToCanonicalJson(decision);
        const auto input_hash = expert::core::Sha256Hex(
            expert::core::json::Canonicalize(manifest_text) + "\n" +
            expert::core::json::Canonicalize(hardware_text) + "\n" +
            expert::core::json::Canonicalize(request_text));
        std::cout << "{\"decision\":" << plan
                  << ",\"decision_sha256\":\"" << expert::core::Sha256Hex(plan)
                  << "\",\"input_sha256\":\"" << input_hash
                  << "\",\"manifest_sha256\":\"" << manifest.manifest_sha256
                  << "\"}\n";
        return decision.status == expert::core::FeasibilityStatus::kImpossible ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "feasibility error: " << error.what() << '\n';
        return 65;
    }
}
