#pragma once

#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/resident_expert_set.hpp"

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace expert::runtime {

struct DeepSeekTensorArtifacts final {
  std::vector<cuda::DeepSeekDenseSpec> dense;
  std::vector<cuda::DeepSeekTypedSpec> typed;
  std::uint64_t dense_source_bytes{};
  std::uint64_t dense_device_bytes{};
  std::uint64_t typed_source_bytes{};
  std::uint64_t maximum_source_record_bytes{};
};

struct DeepSeekTensorArtifactsResult final {
  Status status;
  DeepSeekTensorArtifacts artifacts;
};

// Parses any authenticated DeepSeek tensor namespace. Expected counts are
// explicit so a partially published target or MTP namespace fails closed.
[[nodiscard]] DeepSeekTensorArtifactsResult load_deepseek_tensor_artifacts(
    const std::filesystem::path& dense_root,
    const std::filesystem::path& typed_root,
    const std::filesystem::path& checkpoint_root,
    std::size_t expected_dense, std::size_t expected_typed) noexcept;

struct DeepSeekModelArtifacts final {
  std::vector<cuda::DeepSeekDenseSpec> dense;
  std::vector<cuda::DeepSeekTypedSpec> typed;
  std::vector<ResidentExpertSpec> shared;
  std::uint64_t dense_source_bytes{};
  std::uint64_t dense_device_bytes{};
  std::uint64_t typed_source_bytes{};
  std::uint64_t maximum_source_record_bytes{};
};

struct DeepSeekModelArtifactsResult final {
  Status status;
  DeepSeekModelArtifacts artifacts;
};

// Loads only authenticated descriptors; tensor payloads remain lazy until the
// resident model/shared set loaders consume these exact records.
[[nodiscard]] DeepSeekModelArtifactsResult load_deepseek_model_artifacts(
    const std::filesystem::path& dense_root,
    const std::filesystem::path& typed_root,
    const std::filesystem::path& shared_root,
    const std::filesystem::path& checkpoint_root) noexcept;

}  // namespace expert::runtime
