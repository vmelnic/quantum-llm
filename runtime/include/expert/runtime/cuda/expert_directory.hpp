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

enum class DeviceExpertFormat : std::uint32_t {
  int8_per_row = 0,
  deepseek_fp4_block32 = 1,
};

// Small compute-ready metadata entry. The full model is never materialized in
// this table; only addresses of currently resident slots are published.
struct alignas(16) DeviceExpertEntry final {
  const std::int8_t* gate_up{};
  const float* gate_up_scales{};
  const std::int8_t* down{};
  const float* down_scales{};
  const std::uint8_t* w1_fp4{};
  const std::uint8_t* w1_ue8m0{};
  const std::uint8_t* w3_fp4{};
  const std::uint8_t* w3_ue8m0{};
  const std::uint8_t* w2_fp4{};
  const std::uint8_t* w2_ue8m0{};
  std::uint32_t format{};
  std::uint32_t generation{};
  std::uint32_t state{};
  std::uint32_t device_references{};
};

struct DirectoryPlanResult final {
  Status status;
  std::vector<std::uint32_t> missing_experts;
  // Exact selection order copied from the router output. Unlike the hash-table
  // views below, this preserves duplicates and routing rank, making it suitable
  // for bounded route tracing and deterministic working-set placement.
  std::vector<std::uint32_t> selected_experts;
  // Populated only on the cold path. The control plane may lease these ready
  // entries while it resolves misses, preventing capacity churn before retry.
  std::vector<std::uint32_t> ready_experts;
  std::uint32_t unique_experts{};
  // Nonzero while this route owns device references. Independent tokens allow
  // multiple requests to keep disjoint or overlapping routes pinned.
  std::uint64_t pin_id{};
};

// One instance belongs to one immutable model/ABI. The hash table used to
// deduplicate a route is sized by maximum_selections rather than by the total
// expert count, keeping planning bounded for very large sparse models.
class CudaExpertDirectory final : public IDeviceResidencyDirectory {
 public:
  CudaExpertDirectory(std::uint64_t model_id, std::uint32_t quant_abi,
                      std::uint32_t layers, std::uint32_t experts_per_layer,
                      std::uint32_t maximum_selections,
                      std::uint32_t maximum_active_pins = 64U);
  ~CudaExpertDirectory() override;
  CudaExpertDirectory(const CudaExpertDirectory&) = delete;
  CudaExpertDirectory& operator=(const CudaExpertDirectory&) = delete;

  [[nodiscard]] Status publish(
      const ExpertKey& key,
      std::shared_ptr<IDeviceAllocation> allocation) override;
  void retire(const ExpertKey& key) noexcept override;

  // A returned pin_id holds one device reference for every ready unique expert.
  // release_pins(pin_id) is mandatory after the dependent kernels finish.
  [[nodiscard]] DirectoryPlanResult pin_or_collect_misses(
      std::uint32_t layer, const std::uint32_t* device_expert_indices,
      std::uint32_t selection_count, void* stream,
      bool keep_ready_pins_on_miss = false);
  [[nodiscard]] Status release_pins(std::uint64_t pin_id,
                                    void* stream) noexcept;

  [[nodiscard]] const DeviceExpertEntry* device_entries() const noexcept;
  [[nodiscard]] std::uint32_t experts_per_layer() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace expert::runtime::cuda
