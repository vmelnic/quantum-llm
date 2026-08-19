#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace expert::runtime {

struct ExpertCatalogLayout final {
  std::string extent_header;
  std::string catalog_header;
  std::size_t extents_per_expert{};
  bool payload_is_catalog_relative{};
};

struct ExpertCatalogConfig final {
  std::filesystem::path catalog_root;
  std::filesystem::path source_root;
  std::uint32_t layer_count{};
  std::uint32_t experts_per_layer{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t quant_block_size{};
  std::uint64_t stored_bytes{};
  std::uint64_t decoded_bytes{};
  std::uint64_t device_bytes{};
  std::uint32_t source_abi{};
  std::uint32_t record_abi{};
  std::uint32_t header_bytes{};
  std::uint32_t alignment{};
  std::vector<ExpertCatalogLayout> layouts;
};

// Immutable O(1) index for one routed-expert component. File names, extent
// count, record geometry, and layer/expert cardinality are adapter data rather
// than properties of this class.
class ExpertCatalog {
 public:
  ExpertCatalog() = default;
  ExpertCatalog(const ExpertCatalog&) = delete;
  ExpertCatalog& operator=(const ExpertCatalog&) = delete;
  ExpertCatalog(ExpertCatalog&&) noexcept = default;
  ExpertCatalog& operator=(ExpertCatalog&&) noexcept = default;

  [[nodiscard]] static Status load(const ExpertCatalogConfig& config,
                                   ExpertCatalog& destination) noexcept;
  [[nodiscard]] static Status from_records(
      std::uint32_t layer_count, std::uint32_t experts_per_layer,
      std::vector<PayloadRecord> records,
      ExpertCatalog& destination) noexcept;
  [[nodiscard]] const PayloadRecord* find(std::uint32_t layer,
                                          std::uint32_t expert) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
  [[nodiscard]] std::uint64_t source_bytes() const noexcept {
    return source_bytes_;
  }
  [[nodiscard]] std::uint32_t layer_count() const noexcept { return layers_; }
  [[nodiscard]] std::uint32_t experts_per_layer() const noexcept {
    return experts_;
  }
  void clear() noexcept;

 private:
  std::vector<PayloadRecord> records_;
  std::uint64_t source_bytes_{};
  std::uint32_t layers_{};
  std::uint32_t experts_{};
};

}  // namespace expert::runtime
