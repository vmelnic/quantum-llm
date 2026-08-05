#include "expert/runtime/deepseek_catalog.hpp"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace expert::runtime {
namespace {

constexpr std::uint64_t kStoredBytes = 13'369'344U;
constexpr std::uint64_t kDecodedBytes = 3ULL * 4096U * 2048U * sizeof(float);
constexpr std::uint64_t kDeviceBytes = 25'198'592U;
constexpr std::size_t kExpertCount =
    static_cast<std::size_t>(kDeepSeekCatalogLayers) *
    kDeepSeekCatalogExperts;

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
  const std::filesystem::path path = text;
  if (path.empty() || path.is_absolute() || path.has_root_name() ||
      path.has_root_directory() || !path.parent_path().empty())
    throw std::invalid_argument("catalog source path is not relative");
  for (const auto& part : path) {
    if (part == "..")
      throw std::invalid_argument("catalog source path escapes its root");
  }
  return path;
}

std::uint64_t unsigned_integer(const std::string& text) {
  std::uint64_t result{};
  const auto parsed = std::from_chars(
      text.data(), text.data() + text.size(), result);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument("catalog integer is not canonical unsigned decimal");
  }
  return result;
}

std::uint8_t nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::invalid_argument("catalog SHA-256 contains an invalid digit");
}

Sha256Digest digest(const std::string& text) {
  if (text.size() != 64U)
    throw std::invalid_argument("catalog SHA-256 has the wrong length");
  Sha256Digest result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] = static_cast<std::byte>(
        (nibble(text[index * 2U]) << 4U) |
        nibble(text[index * 2U + 1U]));
  }
  return result;
}

}  // namespace

Status DeepSeekExpertCatalog::load(
    const std::filesystem::path& catalog_root,
    const std::filesystem::path& source_root,
    DeepSeekExpertCatalog& destination) {
  if (!destination.records_.empty()) {
    return {ErrorCode::invalid_argument,
            "DeepSeek catalog destination is not empty"};
  }
  try {
    std::ifstream extent_input(catalog_root / "extents.tsv");
    std::string line;
    if (!std::getline(extent_input, line)) {
      return {ErrorCode::invalid_argument,
              "invalid DeepSeek routed extent header"};
    }
    const bool packed = line == "deepseek-routed-pack-extents-v1";
    if (!packed && line != "deepseek-routed-extents-v1")
      return {ErrorCode::invalid_argument,
              "invalid DeepSeek routed extent header"};
    const std::size_t extents_per_expert = packed ? 1U : 6U;
    const auto payload_root = packed ? catalog_root : source_root;
    std::vector<PayloadExtent> extents;
    extents.reserve(kExpertCount * extents_per_expert);
    while (std::getline(extent_input, line)) {
      const auto item = fields(line);
      if (item.size() != 4U)
        return {ErrorCode::invalid_argument,
                "invalid DeepSeek routed extent row"};
      extents.push_back({payload_root / relative_path(item[3]),
                         unsigned_integer(item[2]), unsigned_integer(item[0]),
                         unsigned_integer(item[1])});
    }
    if (!extent_input.eof() ||
        extents.size() != kExpertCount * extents_per_expert) {
      return {ErrorCode::invalid_argument,
              "DeepSeek routed extent catalog is incomplete"};
    }

    std::ifstream catalog_input(catalog_root / "catalog.tsv");
    if (!std::getline(catalog_input, line) ||
        line != (packed ? "deepseek-routed-pack-catalog-v1"
                        : "deepseek-routed-catalog-v1")) {
      return {ErrorCode::invalid_argument,
              "invalid DeepSeek routed catalog header"};
    }
    DeepSeekExpertCatalog candidate;
    candidate.records_.reserve(kExpertCount);
    std::size_t expected_index = 0U;
    while (std::getline(catalog_input, line)) {
      const auto item = fields(line);
      if (item.size() != 6U)
        return {ErrorCode::invalid_argument,
                "invalid DeepSeek routed catalog row"};
      const auto layer = unsigned_integer(item[0]);
      const auto expert = unsigned_integer(item[1]);
      const auto stored = unsigned_integer(item[2]);
      const auto first = unsigned_integer(item[4]);
      const auto count = unsigned_integer(item[5]);
      if (layer != expected_index / kDeepSeekCatalogExperts ||
          expert != expected_index % kDeepSeekCatalogExperts ||
          stored != kStoredBytes ||
          first != expected_index * extents_per_expert ||
          count != extents_per_expert || first + count > extents.size()) {
        return {ErrorCode::invalid_argument,
                "DeepSeek routed catalog ordering/geometry mismatch"};
      }
      PayloadRecord record;
      const auto first_index = static_cast<std::size_t>(first);
      const auto extent_count = static_cast<std::size_t>(count);
      record.extents.assign(extents.begin() + first_index,
                            extents.begin() + first_index + extent_count);
      std::uint64_t destination_offset = 0U;
      for (const auto& extent : record.extents) {
        if (extent.destination_offset != destination_offset)
          return {ErrorCode::invalid_argument,
                  "DeepSeek routed extents do not exactly cover the payload"};
        destination_offset += extent.bytes;
      }
      if (destination_offset != stored)
        return {ErrorCode::invalid_argument,
                "DeepSeek routed extent bytes do not match the catalog"};
      record.stored_bytes = stored;
      record.decoded_bytes = kDecodedBytes;
      record.device_bytes = kDeviceBytes;
      record.source_abi = kExpertSourceAbiDeepSeekCompactV1;
      record.header_bytes = 0U;
      record.alignment = kExpertPackAlignment;
      record.payload_sha256 = digest(item[3]);
      candidate.source_bytes_ += stored;
      candidate.records_.push_back(std::move(record));
      ++expected_index;
    }
    if (!catalog_input.eof() || candidate.records_.size() != kExpertCount) {
      return {ErrorCode::invalid_argument,
              "DeepSeek routed catalog is incomplete"};
    }
    destination = std::move(candidate);
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::invalid_argument,
            std::string("invalid DeepSeek routed catalog: ") + error.what()};
  }
}

const PayloadRecord* DeepSeekExpertCatalog::find(
    std::uint32_t layer, std::uint32_t expert) const noexcept {
  if (layer >= kDeepSeekCatalogLayers || expert >= kDeepSeekCatalogExperts ||
      records_.size() != kExpertCount)
    return nullptr;
  return &records_[static_cast<std::size_t>(layer) *
                       kDeepSeekCatalogExperts + expert];
}

void DeepSeekExpertCatalog::clear() noexcept {
  records_.clear();
  source_bytes_ = 0U;
}

}  // namespace expert::runtime
