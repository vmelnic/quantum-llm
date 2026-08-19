#pragma once

#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/program_executor.hpp"

#include <memory>

namespace expert::runtime {

// Read-only, authenticated memory mapping of the dense tensor packs published
// by ModelArtifact. Mapping keeps the artifact authoritative without copying
// the full dense tier into pageable RAM. Providers may upload or directly
// consume tensor sections but never reopen model-family files.
class MappedModelTensorStore final : public IModelTensorStore {
 public:
  MappedModelTensorStore();
  MappedModelTensorStore(const MappedModelTensorStore&) = delete;
  MappedModelTensorStore& operator=(const MappedModelTensorStore&) = delete;
  MappedModelTensorStore(MappedModelTensorStore&&) noexcept;
  MappedModelTensorStore& operator=(MappedModelTensorStore&&) noexcept;
  ~MappedModelTensorStore() override;

  [[nodiscard]] static Status create(
      const ModelArtifact& artifact,
      MappedModelTensorStore& destination) noexcept;
  [[nodiscard]] ResolveModelTensorResult resolve(
      std::string_view name) override;
  [[nodiscard]] bool valid() const noexcept;

 private:
  struct Core;
  explicit MappedModelTensorStore(std::shared_ptr<Core> core) noexcept;
  std::shared_ptr<Core> core_;
};

}  // namespace expert::runtime
