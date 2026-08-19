#pragma once

#include "expert/runtime/model_descriptor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace expert::runtime {

using WorkerProviderEntry = int (*)(int argc, char** argv);

// End-to-end worker implementation selected solely by the operation ABIs in
// the artifact. Provider names are diagnostic; the common launcher never
// interprets architecture or model-family identifiers.
struct WorkerProviderDefinition final {
  std::string name;
  std::uint32_t priority{};
  std::vector<KernelCapability> capabilities;
  WorkerProviderEntry entry{};
};

struct SelectWorkerProviderResult final {
  Status status;
  const WorkerProviderDefinition* provider{};
};

class WorkerProviderRegistry final {
 public:
  [[nodiscard]] Status add(WorkerProviderDefinition provider) noexcept;
  [[nodiscard]] SelectWorkerProviderResult select(
      const ModelDescriptor& descriptor) const noexcept;

 private:
  std::vector<WorkerProviderDefinition> providers_;
};

}  // namespace expert::runtime
