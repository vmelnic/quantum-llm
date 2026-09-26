#pragma once

#include "expert/runtime/expert_catalog.hpp"
#include "expert/runtime/model_descriptor.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace expert::runtime {

struct ArtifactPack final {
  std::string name;
  std::string kind;
  std::filesystem::path path;
  std::uint64_t bytes{};
  Sha256Digest sha256{};
};

struct ArtifactDenseTensor final {
  std::string name;
  std::string pack;
  std::uint64_t record_offset{};
  std::uint64_t stored_bytes{};
  std::string encoding;
  std::uint32_t quant_abi{};
  std::vector<std::uint32_t> shape;
  std::uint64_t data_offset{};
  std::uint64_t data_bytes{};
  std::uint64_t scale_offset{};
  std::uint64_t scale_bytes{};
  Sha256Digest payload_sha256{};
};

struct ArtifactRoutedComponent final {
  std::string name;
  ExpertCatalog catalog;
};

// Authenticated, runtime-facing view of a published artifact. This is a
// storage-format adapter, not a model-family adapter: all topology and tensor
// roles still come from ModelDescriptor.
class ModelArtifact final {
 public:
  ModelArtifact() = default;
  ModelArtifact(const ModelArtifact&) = delete;
  ModelArtifact& operator=(const ModelArtifact&) = delete;
  ModelArtifact(ModelArtifact&&) noexcept = default;
  ModelArtifact& operator=(ModelArtifact&&) noexcept = default;

  [[nodiscard]] static Status load_expert_pack_v1(
      const std::filesystem::path& root, ModelArtifact& destination) noexcept;
  // Selects a storage adapter from manifest format/version. Architecture
  // identifiers are opaque and never participate in this dispatch.
  [[nodiscard]] static Status load(
      const std::filesystem::path& root, ModelArtifact& destination) noexcept;

  [[nodiscard]] const std::filesystem::path& root() const noexcept {
    return root_;
  }
  [[nodiscard]] const ModelDescriptor& model() const noexcept { return model_; }
  [[nodiscard]] const std::vector<ArtifactPack>& packs() const noexcept {
    return packs_;
  }
  [[nodiscard]] const std::vector<ArtifactDenseTensor>& dense_tensors()
      const noexcept {
    return dense_tensors_;
  }
  [[nodiscard]] const std::vector<ArtifactRoutedComponent>& components()
      const noexcept {
    return components_;
  }
  [[nodiscard]] const ArtifactPack* find_pack(
      std::string_view name) const noexcept;
  [[nodiscard]] const ArtifactRoutedComponent* find_component(
      std::string_view name) const noexcept;

 private:
  std::filesystem::path root_;
  ModelDescriptor model_;
  std::vector<ArtifactPack> packs_;
  std::vector<ArtifactDenseTensor> dense_tensors_;
  std::vector<ArtifactRoutedComponent> components_;
};

}  // namespace expert::runtime
