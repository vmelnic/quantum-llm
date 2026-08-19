#include "expert/runtime/deepseek_catalog.hpp"

#include "expert/runtime/expert_record.hpp"

namespace expert::runtime {
namespace {

ExpertCatalogConfig deepseek_catalog_config(
    const std::filesystem::path& catalog_root,
    const std::filesystem::path& source_root, std::uint32_t layers) {
  return {catalog_root,
          source_root,
          layers,
          kDeepSeekCatalogExperts,
          4096U,
          2048U,
          kExpertFp4BlockSize,
          13'369'344U,
          3ULL * 4096U * 2048U * sizeof(float),
          13'369'344U,
          kExpertSourceAbiDeepSeekCompactV1,
          0U,
          0U,
          kExpertPackAlignment,
          {{"deepseek-routed-pack-extents-v1",
            "deepseek-routed-pack-catalog-v1", 1U, true},
           {"deepseek-routed-extents-v1", "deepseek-routed-catalog-v1", 6U,
            false}}};
}

}  // namespace

Status DeepSeekExpertCatalog::load(
    const std::filesystem::path& catalog_root,
    const std::filesystem::path& source_root,
    DeepSeekExpertCatalog& destination) {
  return load_namespace(catalog_root, source_root, kDeepSeekCatalogLayers,
                        destination);
}

Status DeepSeekExpertCatalog::load_namespace(
    const std::filesystem::path& catalog_root,
    const std::filesystem::path& source_root, std::uint32_t layers,
    DeepSeekExpertCatalog& destination) {
  return ExpertCatalog::load(
      deepseek_catalog_config(catalog_root, source_root, layers), destination);
}

}  // namespace expert::runtime
