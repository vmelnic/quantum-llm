#pragma once

#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/resident_expert_set.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace expert::runtime {

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
