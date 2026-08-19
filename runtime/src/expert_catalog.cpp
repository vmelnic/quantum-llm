#include "expert/runtime/expert_catalog.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace expert::runtime {
namespace {

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
    throw std::invalid_argument("catalog source path is not a file name");
  for (const auto& part : path) {
    if (part == "..")
      throw std::invalid_argument("catalog source path escapes its root");
  }
  return path;
}

std::uint64_t unsigned_integer(const std::string& text) {
  std::uint64_t result{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("catalog integer is not canonical unsigned decimal");
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
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  }
  return result;
}

bool valid(const ExpertCatalogConfig& config) noexcept {
  return !config.catalog_root.empty() && !config.source_root.empty() &&
         config.layer_count != 0U && config.experts_per_layer != 0U &&
         config.hidden_size != 0U && config.intermediate_size != 0U &&
         config.stored_bytes != 0U && config.decoded_bytes != 0U &&
         config.device_bytes != 0U && config.source_abi != 0U &&
         config.alignment != 0U && !config.layouts.empty() &&
         static_cast<std::uint64_t>(config.layer_count) *
                 config.experts_per_layer <=
             std::numeric_limits<std::size_t>::max();
}

}  // namespace

Status ExpertCatalog::load(const ExpertCatalogConfig& config,
                           ExpertCatalog& destination) noexcept {
  if (!destination.records_.empty())
    return {ErrorCode::invalid_argument,
            "expert catalog destination is not empty"};
  if (!valid(config))
    return {ErrorCode::invalid_argument, "expert catalog config is invalid"};
  try {
    std::ifstream extent_input(config.catalog_root / "extents.tsv");
    std::string line;
    if (!std::getline(extent_input, line))
      throw std::invalid_argument("extent index has no header");
    const auto selected = std::find_if(
        config.layouts.begin(), config.layouts.end(),
        [&](const auto& layout) { return layout.extent_header == line; });
    if (selected == config.layouts.end() || selected->catalog_header.empty() ||
        selected->extents_per_expert == 0U)
      throw std::invalid_argument("extent index layout is unsupported");
    const auto expert_count = static_cast<std::size_t>(config.layer_count) *
                              config.experts_per_layer;
    const auto payload_root = selected->payload_is_catalog_relative
                                  ? config.catalog_root
                                  : config.source_root;
    std::vector<PayloadExtent> extents;
    extents.reserve(expert_count * selected->extents_per_expert);
    while (std::getline(extent_input, line)) {
      const auto item = fields(line);
      if (item.size() != 4U)
        throw std::invalid_argument("extent index row has the wrong shape");
      extents.push_back({payload_root / relative_path(item[3]),
                         unsigned_integer(item[2]), unsigned_integer(item[0]),
                         unsigned_integer(item[1])});
    }
    if (!extent_input.eof() ||
        extents.size() != expert_count * selected->extents_per_expert)
      throw std::invalid_argument("extent index is incomplete");

    std::ifstream catalog_input(config.catalog_root / "catalog.tsv");
    if (!std::getline(catalog_input, line) || line != selected->catalog_header)
      throw std::invalid_argument("catalog header does not match extent layout");
    ExpertCatalog candidate;
    candidate.records_.reserve(expert_count);
    candidate.layers_ = config.layer_count;
    candidate.experts_ = config.experts_per_layer;
    std::size_t expected_index = 0U;
    while (std::getline(catalog_input, line)) {
      const auto item = fields(line);
      if (item.size() != 6U)
        throw std::invalid_argument("catalog row has the wrong shape");
      const auto layer = unsigned_integer(item[0]);
      const auto expert = unsigned_integer(item[1]);
      const auto stored = unsigned_integer(item[2]);
      const auto first = unsigned_integer(item[4]);
      const auto count = unsigned_integer(item[5]);
      if (layer != expected_index / config.experts_per_layer ||
          expert != expected_index % config.experts_per_layer ||
          stored != config.stored_bytes ||
          first != expected_index * selected->extents_per_expert ||
          count != selected->extents_per_expert || first + count > extents.size())
        throw std::invalid_argument("catalog ordering or geometry is invalid");
      PayloadRecord record;
      const auto first_index = static_cast<std::size_t>(first);
      const auto extent_count = static_cast<std::size_t>(count);
      record.extents.assign(extents.begin() + first_index,
                            extents.begin() + first_index + extent_count);
      std::uint64_t destination_offset = 0U;
      for (const auto& extent : record.extents) {
        if (extent.destination_offset != destination_offset)
          throw std::invalid_argument("expert extents are not contiguous");
        destination_offset += extent.bytes;
      }
      if (destination_offset != stored)
        throw std::invalid_argument("expert extents do not cover the record");
      record.stored_bytes = stored;
      record.decoded_bytes = config.decoded_bytes;
      record.device_bytes = config.device_bytes;
      record.hidden = config.hidden_size;
      record.intermediate = config.intermediate_size;
      record.quant_block_size = config.quant_block_size;
      record.source_abi = config.source_abi;
      record.record_abi = config.record_abi;
      record.header_bytes = config.header_bytes;
      record.alignment = config.alignment;
      record.payload_sha256 = digest(item[3]);
      candidate.source_bytes_ += stored;
      candidate.records_.push_back(std::move(record));
      ++expected_index;
    }
    if (!catalog_input.eof() || candidate.records_.size() != expert_count)
      throw std::invalid_argument("catalog is incomplete");
    destination = std::move(candidate);
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::invalid_argument,
            std::string("invalid expert catalog: ") + error.what()};
  }
}

Status ExpertCatalog::from_records(
    std::uint32_t layer_count, std::uint32_t experts_per_layer,
    std::vector<PayloadRecord> records, ExpertCatalog& destination) noexcept {
  if (!destination.records_.empty() || layer_count == 0U ||
      experts_per_layer == 0U ||
      records.size() !=
          static_cast<std::uint64_t>(layer_count) * experts_per_layer)
    return {ErrorCode::invalid_argument,
            "in-memory expert catalog cardinality is invalid"};
  std::uint64_t source_bytes = 0U;
  for (const auto& record : records) {
    if (record.stored_bytes == 0U || record.source_abi == 0U)
      return {ErrorCode::invalid_argument,
              "in-memory expert catalog contains an invalid record"};
    if (source_bytes > std::numeric_limits<std::uint64_t>::max() -
                           record.stored_bytes)
      return {ErrorCode::invalid_argument,
              "in-memory expert catalog byte count overflows"};
    source_bytes += record.stored_bytes;
  }
  destination.records_ = std::move(records);
  destination.source_bytes_ = source_bytes;
  destination.layers_ = layer_count;
  destination.experts_ = experts_per_layer;
  return Status::success();
}

const PayloadRecord* ExpertCatalog::find(std::uint32_t layer,
                                         std::uint32_t expert) const noexcept {
  if (layer >= layers_ || expert >= experts_ ||
      records_.size() != static_cast<std::size_t>(layers_) * experts_)
    return nullptr;
  return &records_[static_cast<std::size_t>(layer) * experts_ + expert];
}

void ExpertCatalog::clear() noexcept {
  records_.clear();
  source_bytes_ = 0U;
  layers_ = 0U;
  experts_ = 0U;
}

}  // namespace expert::runtime
