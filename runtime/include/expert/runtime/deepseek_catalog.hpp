#pragma once

#include "expert/runtime/expert_record.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace expert::runtime {

inline constexpr std::uint32_t kDeepSeekCatalogLayers = 43U;
inline constexpr std::uint32_t kDeepSeekCatalogExperts = 256U;

// Immutable, bounded metadata index for one routed-expert namespace. The
// checkpoint remains authoritative. A source catalog contains six immutable
// SafeTensors extents per expert; compact-pack v1 contains one aligned extent
// with identical authenticated bytes. Both publish the same payload ABI.
class DeepSeekExpertCatalog final {
 public:
  [[nodiscard]] static Status load(
      const std::filesystem::path& catalog_root,
      const std::filesystem::path& source_root,
      DeepSeekExpertCatalog& destination);
  [[nodiscard]] static Status load_namespace(
      const std::filesystem::path& catalog_root,
      const std::filesystem::path& source_root,
      std::uint32_t layers,
      DeepSeekExpertCatalog& destination);

  [[nodiscard]] const PayloadRecord* find(std::uint32_t layer,
                                          std::uint32_t expert) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
  [[nodiscard]] std::uint64_t source_bytes() const noexcept {
    return source_bytes_;
  }
  void clear() noexcept;

 private:
  std::vector<PayloadRecord> records_;
  std::uint64_t source_bytes_{};
  std::uint32_t layers_{};
};

}  // namespace expert::runtime
