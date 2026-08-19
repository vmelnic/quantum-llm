#pragma once

#include "expert/runtime/expert_catalog.hpp"

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
class DeepSeekExpertCatalog final : public ExpertCatalog {
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

};

}  // namespace expert::runtime
