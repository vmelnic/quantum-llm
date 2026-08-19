#include "expert/runtime/model_artifact.hpp"

#include "expert/core/json.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace expert::runtime {
namespace {
using expert::core::json::Required;
using expert::core::json::Value;

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::invalid_argument("cannot open artifact metadata");
  std::ostringstream output;
  output << input.rdbuf();
  if (input.bad()) throw std::invalid_argument("cannot read artifact metadata");
  return output.str();
}

std::uint8_t nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::invalid_argument("artifact SHA-256 contains an invalid digit");
}

Sha256Digest digest(std::string_view text) {
  if (text.size() != 64U)
    throw std::invalid_argument("artifact SHA-256 has the wrong length");
  Sha256Digest result{};
  for (std::size_t index = 0U; index < result.size(); ++index)
    result[index] = static_cast<std::byte>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  return result;
}

std::uint32_t u32(std::uint64_t value, std::string_view label) {
  if (value > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument(std::string(label) + " exceeds uint32");
  return static_cast<std::uint32_t>(value);
}

std::filesystem::path safe_child(const std::filesystem::path& root,
                                 std::string_view text) {
  const std::filesystem::path relative(text);
  if (relative.empty() || relative.is_absolute() || relative.has_root_name() ||
      relative.has_root_directory())
    throw std::invalid_argument("artifact path is not relative");
  for (const auto& part : relative)
    if (part == "..") throw std::invalid_argument("artifact path escapes root");
  return root / relative;
}

std::uint64_t fp4_device_bytes(std::uint32_t hidden,
                               std::uint32_t intermediate) {
  const auto elements = static_cast<std::uint64_t>(hidden) * intermediate;
  return elements + 2U * elements / kExpertFp4BlockSize + elements / 2U +
         elements / kExpertFp4BlockSize;
}

std::uint64_t runtime_u64(std::string_view text, std::string_view key) {
  const auto prefix = std::string(key) + "\t";
  std::optional<std::uint64_t> result;
  std::size_t offset{};
  while (offset < text.size()) {
    const auto end = text.find('\n', offset);
    auto line = text.substr(offset, end == std::string_view::npos
                                        ? text.size() - offset
                                        : end - offset);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1U);
    if (line.starts_with(prefix)) {
      if (result)
        throw std::invalid_argument("artifact runtime field is duplicated");
      const auto value = line.substr(prefix.size());
      std::uint64_t parsed{};
      const auto converted = std::from_chars(
          value.data(), value.data() + value.size(), parsed);
      if (value.empty() || converted.ec != std::errc{} ||
          converted.ptr != value.data() + value.size() || parsed == 0U)
        throw std::invalid_argument("artifact runtime integer is invalid");
      result = parsed;
    }
    if (end == std::string_view::npos) break;
    offset = end + 1U;
  }
  if (!result) throw std::invalid_argument("artifact runtime field is absent");
  return *result;
}

}  // namespace

Status ModelArtifact::load(const std::filesystem::path& root,
                           ModelArtifact& destination) noexcept {
  try {
    if (!destination.root_.empty() || !std::filesystem::is_directory(root) ||
        !std::filesystem::is_regular_file(root / "manifest.json"))
      return {ErrorCode::invalid_argument,
              "artifact destination is non-empty or manifest is absent"};
    const auto document = expert::core::json::Parse(
        read_text(root / "manifest.json"));
    const auto& manifest = document.AsObject("manifest");
    const auto& format = Required(manifest, "format", "manifest")
                             .AsObject("manifest.format");
    const auto name =
        Required(format, "name", "format").AsString("format.name");
    const auto version =
        Required(format, "version", "format").AsU64("format.version");
    if (name == "expert-pack" && version == 1U)
      return load_expert_pack_v1(root, destination);
    if (name == "deepseek-worker-bundle" && version == 3U)
      return load_deepseek_worker_bundle_v3(root, destination);
    return {ErrorCode::invalid_argument,
            "no artifact storage adapter implements manifest format"};
  } catch (const std::exception& error) {
    return {ErrorCode::invalid_argument,
            std::string("invalid artifact manifest: ") + error.what()};
  }
}

Status ModelArtifact::load_deepseek_worker_bundle_v3(
    const std::filesystem::path& root, ModelArtifact& destination) noexcept {
  try {
    const auto document = expert::core::json::Parse(
        read_text(root / "manifest.json"));
    const auto& manifest = document.AsObject("manifest");
    const auto& integrity = Required(manifest, "integrity", "manifest")
                                .AsObject("manifest.integrity");
    const auto content_hash = digest(
        Required(integrity, "content_sha256", "integrity")
            .AsString("integrity.content_sha256"));
    const auto runtime_text = read_text(root / "runtime.tsv");
    Sha256 runtime_hasher;
    runtime_hasher.update(std::as_bytes(std::span(runtime_text)));
    if (!constant_time_equal(runtime_hasher.finalize(), content_hash))
      return {ErrorCode::checksum_mismatch,
              "worker bundle runtime index hash mismatch"};
    const auto& program = Required(manifest, "model_program", "manifest")
                              .AsObject("manifest.model_program");
    if (Required(program, "format", "model_program")
            .AsString("model_program.format") != "expert-runtime-model-v1")
      return {ErrorCode::invalid_argument,
              "worker bundle model program format is unsupported"};
    const auto program_path = safe_child(
        root, Required(program, "path", "model_program")
                  .AsString("model_program.path"));
    auto parsed = load_model_descriptor_artifact(
        program_path,
        Required(program, "bytes", "model_program")
            .AsU64("model_program.bytes"),
        digest(Required(program, "sha256", "model_program")
                   .AsString("model_program.sha256")),
        content_hash, runtime_u64(runtime_text, "model_id"));
    if (!parsed.status.ok())
      return {parsed.status.code(), std::string(parsed.status.message())};
    ModelArtifact candidate;
    candidate.root_ = root;
    candidate.model_ = std::move(parsed.descriptor);
    destination = std::move(candidate);
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::invalid_argument,
            std::string("invalid worker bundle artifact: ") + error.what()};
  }
}

const ArtifactPack* ModelArtifact::find_pack(std::string_view name) const
    noexcept {
  const auto found = std::find_if(packs_.begin(), packs_.end(),
                                  [name](const auto& item) {
                                    return item.name == name;
                                  });
  return found == packs_.end() ? nullptr : &*found;
}

const ArtifactRoutedComponent* ModelArtifact::find_component(
    std::string_view name) const noexcept {
  const auto found = std::find_if(components_.begin(), components_.end(),
                                  [name](const auto& item) {
                                    return item.name == name;
                                  });
  return found == components_.end() ? nullptr : &*found;
}

Status ModelArtifact::load_expert_pack_v1(
    const std::filesystem::path& root, ModelArtifact& destination) noexcept {
  try {
    if (!destination.root_.empty() || !std::filesystem::is_directory(root) ||
        !std::filesystem::is_regular_file(root / "manifest.json") ||
        !std::filesystem::is_regular_file(root / "COMPLETED"))
      return {ErrorCode::invalid_argument,
              "artifact destination is non-empty or container is incomplete"};
    const auto document = expert::core::json::Parse(
        read_text(root / "manifest.json"));
    const auto& manifest = document.AsObject("manifest");
    const auto& format = Required(manifest, "format", "manifest")
                             .AsObject("manifest.format");
    if (Required(format, "name", "format").AsString("format.name") !=
            "expert-pack" ||
        Required(format, "version", "format").AsU64("format.version") != 1U)
      return {ErrorCode::invalid_argument,
              "artifact is not Expert Pack v1"};
    const auto& integrity = Required(manifest, "integrity", "manifest")
                                .AsObject("manifest.integrity");
    const auto content_hash = digest(
        Required(integrity, "content_sha256", "integrity")
            .AsString("integrity.content_sha256"));
    const auto namespace_base = namespace_id_from_content_hash(content_hash);

    const auto& program = Required(manifest, "model_program", "manifest")
                              .AsObject("manifest.model_program");
    if (Required(program, "format", "model_program")
            .AsString("model_program.format") != "expert-runtime-model-v1")
      return {ErrorCode::invalid_argument,
              "artifact model program format is unsupported"};
    const auto program_path = safe_child(
        root, Required(program, "path", "model_program")
                  .AsString("model_program.path"));
    auto parsed = load_model_descriptor_artifact(
        program_path,
        Required(program, "bytes", "model_program")
            .AsU64("model_program.bytes"),
        digest(Required(program, "sha256", "model_program")
                   .AsString("model_program.sha256")),
        content_hash, namespace_base);
    if (!parsed.status.ok())
      return {parsed.status.code(), std::string(parsed.status.message())};
    if (parsed.descriptor.routed_components.size() > 1U)
      return {ErrorCode::invalid_argument,
              "Expert Pack v1 indexes at most one routed component"};

    ModelArtifact candidate;
    candidate.root_ = root;
    candidate.model_ = std::move(parsed.descriptor);
    std::map<std::string, std::uint64_t, std::less<>> pack_sizes;
    for (const auto& value :
         Required(manifest, "packs", "manifest").AsArray("manifest.packs")) {
      const auto& item = value.AsObject("pack");
      ArtifactPack pack;
      pack.name = Required(item, "name", "pack").AsString("pack.name");
      pack.kind = Required(item, "kind", "pack").AsString("pack.kind");
      pack.path = safe_child(root, pack.name);
      pack.bytes = Required(item, "bytes", "pack").AsU64("pack.bytes");
      pack.sha256 = digest(Required(item, "sha256", "pack")
                               .AsString("pack.sha256"));
      if ((pack.kind != "dense" && pack.kind != "experts") ||
          !std::filesystem::is_regular_file(pack.path) ||
          std::filesystem::file_size(pack.path) != pack.bytes ||
          !pack_sizes.emplace(pack.name, pack.bytes).second)
        throw std::invalid_argument("artifact pack index is invalid");
      candidate.packs_.push_back(std::move(pack));
    }

    for (const auto& value : Required(manifest, "tensors", "manifest")
                                 .AsArray("manifest.tensors")) {
      const auto& item = value.AsObject("tensor");
      ArtifactDenseTensor tensor;
      tensor.name =
          Required(item, "name", "tensor").AsString("tensor.name");
      tensor.pack =
          Required(item, "pack", "tensor").AsString("tensor.pack");
      tensor.record_offset =
          Required(item, "offset", "tensor").AsU64("tensor.offset");
      tensor.stored_bytes = Required(item, "stored_bytes", "tensor")
                                .AsU64("tensor.stored_bytes");
      tensor.encoding = Required(item, "stored_dtype", "tensor")
                            .AsString("tensor.stored_dtype");
      tensor.quant_abi = u32(
          Required(item, "quant_abi", "tensor").AsU64("tensor.quant_abi"),
          "tensor quant ABI");
      for (const auto& dimension :
           Required(item, "source_shape", "tensor").AsArray("tensor.shape"))
        tensor.shape.push_back(
            u32(dimension.AsU64("tensor.dimension"), "tensor dimension"));
      const auto& sections = Required(item, "sections", "tensor")
                                 .AsObject("tensor.sections");
      const auto& data = Required(sections, "data", "sections")
                             .AsObject("tensor.data");
      const auto& scales = Required(sections, "scales", "sections")
                               .AsObject("tensor.scales");
      tensor.data_offset =
          Required(data, "offset", "data").AsU64("data.offset");
      tensor.data_bytes =
          Required(data, "bytes", "data").AsU64("data.bytes");
      tensor.scale_offset =
          Required(scales, "offset", "scales").AsU64("scales.offset");
      tensor.scale_bytes =
          Required(scales, "bytes", "scales").AsU64("scales.bytes");
      tensor.payload_sha256 = digest(
          Required(item, "payload_sha256", "tensor")
              .AsString("tensor.payload_sha256"));
      const auto pack = pack_sizes.find(tensor.pack);
      if (pack == pack_sizes.end() || tensor.shape.empty() ||
          tensor.record_offset > pack->second ||
          tensor.stored_bytes > pack->second - tensor.record_offset ||
          (tensor.encoding != "I8" && tensor.encoding != "F32" &&
           tensor.encoding != "FP4_E2M1"))
        throw std::invalid_argument("dense tensor index is invalid");
      candidate.dense_tensors_.push_back(std::move(tensor));
    }

    const auto& expert_values =
        Required(manifest, "experts", "manifest").AsArray("manifest.experts");
    if (candidate.model_.routed_components.empty()) {
      if (!expert_values.empty())
        throw std::invalid_argument(
            "dense-only artifact contains routed expert records");
      destination = std::move(candidate);
      return Status::success();
    }
    const auto& component = candidate.model_.routed_components.front();
    std::vector<PayloadRecord> records(expert_table_entries(component));
    std::vector<bool> present(records.size(), false);
    for (const auto& value : expert_values) {
      const auto& item = value.AsObject("expert");
      const auto layer = u32(
          Required(item, "layer", "expert").AsU64("expert.layer"),
          "expert layer");
      const auto expert = u32(
          Required(item, "expert", "expert").AsU64("expert.expert"),
          "expert index");
      if (layer >= component.layer_count ||
          expert >= component.experts_per_layer)
        throw std::invalid_argument("expert is outside component geometry");
      const auto index = static_cast<std::size_t>(layer) *
                             component.experts_per_layer +
                         expert;
      if (present[index]) throw std::invalid_argument("duplicate expert record");
      const auto pack_name =
          Required(item, "pack", "expert").AsString("expert.pack");
      const auto pack = pack_sizes.find(pack_name);
      if (pack == pack_sizes.end())
        throw std::invalid_argument("expert references an unknown pack");
      auto& record = records[index];
      record.path = safe_child(root, pack_name);
      record.record_offset =
          Required(item, "offset", "expert").AsU64("expert.offset");
      record.stored_bytes = Required(item, "stored_bytes", "expert")
                                .AsU64("expert.stored_bytes");
      record.decoded_bytes = Required(item, "decoded_bytes", "expert")
                                 .AsU64("expert.decoded_bytes");
      record.hidden = component.hidden_size;
      record.intermediate = component.intermediate_size;
      record.quant_block_size = kExpertFp4BlockSize;
      record.source_abi = component.source_abi;
      record.record_abi = u32(
          Required(item, "quant_abi", "expert").AsU64("expert.quant_abi"),
          "expert record ABI");
      record.header_bytes = kExpertHeaderBytes;
      record.alignment = kExpertPackAlignment;
      record.payload_sha256 = digest(
          Required(item, "payload_sha256", "expert")
              .AsString("expert.payload_sha256"));
      record.device_bytes =
          record.record_abi == kExpertRecordAbiFp4Block32
              ? fp4_device_bytes(record.hidden, record.intermediate)
              : 3ULL * record.hidden * record.intermediate +
                    static_cast<std::uint64_t>(
                        2U * record.intermediate + record.hidden) *
                        sizeof(float);
      if (record.record_offset > pack->second ||
          record.stored_bytes > pack->second - record.record_offset ||
          record.source_abi != kExpertSourceAbiExpertPackV1 ||
          (record.record_abi != kExpertRecordAbiInt8PerRow &&
           record.record_abi != kExpertRecordAbiFp4Block32))
        throw std::invalid_argument("expert record index is invalid");
      present[index] = true;
    }
    if (!std::all_of(present.begin(), present.end(), [](bool item) {
          return item;
        }))
      throw std::invalid_argument("routed component catalog is incomplete");
    ArtifactRoutedComponent routed;
    routed.name = component.name;
    const auto catalog_status = ExpertCatalog::from_records(
        component.layer_count, component.experts_per_layer,
        std::move(records), routed.catalog);
    if (!catalog_status.ok())
      return {catalog_status.code(), std::string(catalog_status.message())};
    candidate.components_.push_back(std::move(routed));
    destination = std::move(candidate);
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::invalid_argument,
            std::string("invalid model artifact: ") + error.what()};
  }
}

}  // namespace expert::runtime
