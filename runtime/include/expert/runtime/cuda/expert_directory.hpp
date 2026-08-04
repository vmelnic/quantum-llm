#pragma once

#include "expert/runtime/expert_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace expert::runtime::cuda {

enum class DeviceExpertState : std::uint32_t {
  absent = 0,
  ready = 1,
  retiring = 2,
};

// Small compute-ready metadata entry. The full model is never materialized in
// this table; only addresses of currently resident slots are published.
struct alignas(16) DeviceExpertEntry final {
  const std::int8_t* gate_up{};
  const float* gate_up_scales{};
  const std::int8_t* down{};
  const float* down_scales{};
  std::uint32_t generation{};
  std::uint32_t state{};
  std::uint32_t device_references{};
  std::uint32_t reserved{};
};

struct DirectoryPlanResult final {
  Status status;
  std::vector<std::uint32_t> missing_experts;
  // Populated only on the cold path. The control plane may lease these ready
  // entries while it resolves misses, preventing capacity churn before retry.
  std::vector<std::uint32_t> ready_experts;
  std::uint32_t unique_experts{};
};

// One instance belongs to one immutable model/ABI. The hash table used to
// deduplicate a route is sized by maximum_selections rather than by the total
// expert count, keeping planning bounded for very large sparse models.
class CudaExpertDirectory final : public IDeviceResidencyDirectory {
 public:
  CudaExpertDirectory(std::uint64_t model_id, std::uint32_t quant_abi,
                      std::uint32_t layers, std::uint32_t experts_per_layer,
                      std::uint32_t maximum_selections);
  ~CudaExpertDirectory() override;
  CudaExpertDirectory(const CudaExpertDirectory&) = delete;
  CudaExpertDirectory& operator=(const CudaExpertDirectory&) = delete;

  [[nodiscard]] Status publish(
      const ExpertKey& key,
      std::shared_ptr<IDeviceAllocation> allocation) override;
  void retire(const ExpertKey& key) noexcept override;

  // A successful empty miss list holds one device reference for every unique
  // selected expert. release_pins() is mandatory after the MoE kernels finish.
  [[nodiscard]] DirectoryPlanResult pin_or_collect_misses(
      std::uint32_t layer, const std::uint32_t* device_expert_indices,
      std::uint32_t selection_count, void* stream,
      bool keep_ready_pins_on_miss = false);
  [[nodiscard]] Status release_pins(void* stream) noexcept;

  [[nodiscard]] const DeviceExpertEntry* device_entries() const noexcept;
  [[nodiscard]] std::uint32_t experts_per_layer() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace expert::runtime::cuda
