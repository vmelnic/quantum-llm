#include "expert/runtime/deepseek_artifacts.hpp"

#include "expert/runtime/deepseek_expert.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace expert::runtime {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

std::vector<std::string> fields(const std::string& line) {
  std::vector<std::string> result;
  std::size_t start = 0U;
  for (;;) {
    const auto separator = line.find('\t', start);
    result.push_back(line.substr(start, separator - start));
    if (separator == std::string::npos) return result;
    start = separator + 1U;
  }
}

std::filesystem::path relative_path(const std::string& text) {
  const std::filesystem::path path(text);
  require(!path.empty() && !path.is_absolute() && !path.has_root_name() &&
              !path.has_root_directory(),
          "artifact extent path is not relative");
  for (const auto& part : path)
    require(part != "..", "artifact extent path escapes its root");
  return path;
}

std::uint8_t nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::invalid_argument("artifact SHA-256 contains an invalid digit");
}

Sha256Digest digest(const std::string& text) {
  require(text.size() == 64U, "artifact SHA-256 has the wrong length");
  Sha256Digest result{};
  for (std::size_t index = 0U; index < result.size(); ++index)
    result[index] = static_cast<std::byte>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  return result;
}

std::vector<PayloadExtent> extents(const std::filesystem::path& descriptor,
                                   const std::filesystem::path& checkpoint,
                                   std::size_t expected) {
  std::ifstream input(descriptor);
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              line == "deepseek-compact-extents-v1",
          "invalid artifact extent header");
  std::vector<PayloadExtent> result;
  while (std::getline(input, line)) {
    const auto item = fields(line);
    require(item.size() == 4U, "invalid artifact extent row");
    result.push_back({checkpoint / relative_path(item[3]),
                      std::stoull(item[2]), std::stoull(item[0]),
                      std::stoull(item[1])});
  }
  require(input.eof() && result.size() == expected,
          "artifact tensor has the wrong extent count");
  return result;
}

cuda::DeepSeekDtype dtype(const std::string& text) {
  if (text == "BF16") return cuda::DeepSeekDtype::bf16;
  if (text == "F32") return cuda::DeepSeekDtype::f32;
  if (text == "I64") return cuda::DeepSeekDtype::i64;
  throw std::invalid_argument("unsupported artifact tensor dtype");
}

}  // namespace

DeepSeekTensorArtifactsResult load_deepseek_tensor_artifacts(
    const std::filesystem::path& dense_root,
    const std::filesystem::path& typed_root,
    const std::filesystem::path& checkpoint_root,
    std::size_t expected_dense, std::size_t expected_typed) noexcept {
  try {
    require(expected_dense != 0U && expected_typed != 0U,
            "artifact expected counts must be nonzero");
    DeepSeekTensorArtifacts result;
    std::ifstream dense_input(dense_root / "dense-set.tsv");
    std::string line;
    require(static_cast<bool>(std::getline(dense_input, line)) &&
                (line == "deepseek-dense-residency-v1" ||
                 line == "deepseek-mtp-dense-residency-v1"),
            "invalid dense artifact header");
    while (std::getline(dense_input, line)) {
      const auto item = fields(line);
      require(item.size() == 7U, "invalid dense artifact row");
      PayloadRecord record;
      record.extents =
          extents(dense_root / relative_path(item[6]), checkpoint_root, 2U);
      record.stored_bytes = std::stoull(item[3]);
      record.device_bytes = std::stoull(item[4]);
      record.source_abi = kExpertSourceAbiDeepSeekFp8Block128V1;
      record.alignment = kExpertPackAlignment;
      record.payload_sha256 = digest(item[5]);
      result.dense.push_back(
          {item[0], std::move(record),
           static_cast<std::uint32_t>(std::stoul(item[1])),
           static_cast<std::uint32_t>(std::stoul(item[2]))});
      result.dense_source_bytes += std::stoull(item[3]);
      result.dense_device_bytes += std::stoull(item[4]);
      result.maximum_source_record_bytes = std::max(
          result.maximum_source_record_bytes, std::stoull(item[3]));
    }
    require(dense_input.eof() && result.dense.size() == expected_dense,
            "dense artifact set is incomplete");

    std::ifstream typed_input(typed_root / "typed-set.tsv");
    require(static_cast<bool>(std::getline(typed_input, line)) &&
                line == "deepseek-typed-residency-v1",
            "invalid typed artifact header");
    while (std::getline(typed_input, line)) {
      const auto item = fields(line);
      require(item.size() == 6U, "invalid typed artifact row");
      PayloadRecord record;
      record.extents =
          extents(typed_root / relative_path(item[5]), checkpoint_root, 1U);
      record.stored_bytes = std::stoull(item[3]);
      record.payload_sha256 = digest(item[4]);
      result.typed.push_back({item[0], std::move(record), dtype(item[1])});
      result.typed_source_bytes += std::stoull(item[3]);
    }
    require(typed_input.eof() && result.typed.size() == expected_typed,
            "typed artifact set is incomplete");
    return {Status::success(), std::move(result)};
  } catch (const std::exception& error) {
    return {{ErrorCode::invalid_argument,
             std::string("invalid DeepSeek tensor artifacts: ") + error.what()},
            {}};
  }
}

DeepSeekModelArtifactsResult load_deepseek_model_artifacts(
    const std::filesystem::path& dense_root,
    const std::filesystem::path& typed_root,
    const std::filesystem::path& shared_root,
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expected_routed_layers,
    std::uint64_t namespace_id) noexcept {
  try {
    auto loaded = load_deepseek_tensor_artifacts(
        dense_root, typed_root, checkpoint_root, 236U, 834U);
    if (!loaded.status.ok())
      throw std::invalid_argument(std::string(loaded.status.message()));
    DeepSeekModelArtifacts result;
    result.dense = std::move(loaded.artifacts.dense);
    result.typed = std::move(loaded.artifacts.typed);
    result.dense_source_bytes = loaded.artifacts.dense_source_bytes;
    result.dense_device_bytes = loaded.artifacts.dense_device_bytes;
    result.typed_source_bytes = loaded.artifacts.typed_source_bytes;
    result.maximum_source_record_bytes =
        loaded.artifacts.maximum_source_record_bytes;
    auto shared = load_deepseek_shared_artifacts(
        shared_root, checkpoint_root, expected_routed_layers, namespace_id);
    if (!shared.status.ok())
      throw std::invalid_argument(std::string(shared.status.message()));
    result.shared = std::move(shared.shared);
    return {Status::success(), std::move(result)};
  } catch (const std::exception& error) {
    return {{ErrorCode::invalid_argument,
             std::string("invalid DeepSeek model artifacts: ") + error.what()},
            {}};
  }
}

DeepSeekSharedArtifactsResult load_deepseek_shared_artifacts(
    const std::filesystem::path& shared_root,
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expected_layers, std::uint64_t model_id) noexcept {
  try {
    require(expected_layers != 0U && model_id != 0U,
            "shared artifact namespace identity is invalid");
    std::vector<ResidentExpertSpec> result;
    result.reserve(expected_layers);
    std::string line;
    std::ifstream shared_input(shared_root / "shared-set.tsv");
    require(static_cast<bool>(std::getline(shared_input, line)) &&
                line == "deepseek-shared-residency-v1",
            "invalid shared artifact header");
    while (std::getline(shared_input, line)) {
      const auto item = fields(line);
      require(item.size() == 4U, "invalid shared artifact row");
      const auto layer = static_cast<std::uint32_t>(std::stoul(item[0]));
      const auto bytes = std::stoull(item[1]);
      require(layer == result.size() && layer < expected_layers &&
                  bytes == 25'167'360ULL,
              "shared artifact geometry is not canonical");
      PayloadRecord record;
      record.extents = extents(shared_root / relative_path(item[3]),
                               checkpoint_root, 6U);
      record.stored_bytes = bytes;
      record.decoded_bytes = 3ULL * 4096U * 2048U * sizeof(float);
      record.device_bytes = 25'198'592ULL;
      record.hidden = 4096U;
      record.intermediate = 2048U;
      record.quant_block_size = 128U;
      record.source_abi = kExpertSourceAbiDeepSeekFp8Block128V1;
      record.alignment = kExpertPackAlignment;
      record.header_bytes = 0U;
      record.payload_sha256 = digest(item[2]);
      result.push_back({{model_id, layer, 256U,
                         kExpertQuantAbiDeepSeekSm86},
                        std::move(record)});
    }
    require(shared_input.eof() && result.size() == expected_layers,
            "shared artifact set is incomplete");
    return {Status::success(), std::move(result)};
  } catch (const std::exception& error) {
    return {{ErrorCode::invalid_argument,
             std::string("invalid DeepSeek shared artifacts: ") +
                 error.what()},
            {}};
  }
}

}  // namespace expert::runtime
