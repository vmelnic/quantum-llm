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

struct DeepSeekSharedArtifactsResult final {
  Status status;
  std::vector<ResidentExpertSpec> shared;
};

// Parses one authenticated shared-expert namespace. model_id keeps cache and
// directory keys collision-free when the target and MTP namespaces both use
// logical layer zero.
[[nodiscard]] DeepSeekSharedArtifactsResult load_deepseek_shared_artifacts(
    const std::filesystem::path& shared_root,
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expected_layers, std::uint64_t model_id) noexcept;

// Loads only authenticated descriptors; tensor payloads remain lazy until the
// resident model/shared set loaders consume these exact records.
[[nodiscard]] DeepSeekModelArtifactsResult load_deepseek_model_artifacts(
    const std::filesystem::path& dense_root,
    const std::filesystem::path& typed_root,
    const std::filesystem::path& shared_root,
    const std::filesystem::path& checkpoint_root,
    std::uint32_t expected_routed_layers = 43U,
    std::uint64_t namespace_id = 17U) noexcept;

}  // namespace expert::runtime
