#pragma once

#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>

namespace expert::runtime::cuda {

struct MoeLaunch final {
  const float* input{};
  const std::int8_t* const* gate_up_weights{};
  const float* const* gate_up_scales{};
  const std::int8_t* const* down_weights{};
  const float* const* down_scales{};
  const float* routing_weights{};
  const std::uint32_t* expert_indices{};  // null means slots already selected.
  float* intermediate{};  // [top_k, intermediate_size]
  float* output{};        // [hidden_size]
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t top_k{};
  std::uint32_t expert_table_size{};  // top_k when expert_indices is null.
  void* stream{};         // cudaStream_t without leaking CUDA headers.
  const DeviceExpertEntry* directory_entries{};
  std::uint32_t directory_layer{};
};

struct MoeBatchLaunch final {
  const float* input{};                 // [rows, hidden]
  const std::int8_t* const* gate_up_weights{};
  const float* const* gate_up_scales{};
  const std::int8_t* const* down_weights{};
  const float* const* down_scales{};
  const float* routing_weights{};       // [rows, top_k]
  const std::uint32_t* expert_indices{};  // [rows, top_k]
  float* intermediate{};                // [rows, top_k, intermediate]
  float* output{};                      // [rows, hidden]
  std::uint32_t rows{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t top_k{};
  std::uint32_t expert_table_size{};
  void* stream{};
  const DeviceExpertEntry* directory_entries{};
  std::uint32_t directory_layer{};
};

struct MoeSelectionBatchLaunch final {
  const float* input{};                   // [rows, hidden]
  const float* routing_weights{};         // retained for ABI symmetry
  const std::uint32_t* expert_indices{};  // [rows, top_k]
  const std::uint8_t* selection_mask{};   // [rows, top_k], null means all
  float* intermediate{};                  // [rows, top_k, intermediate]
  float* selection_outputs{};             // [rows, top_k, hidden]
  std::uint32_t rows{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t top_k{};
  std::uint32_t expert_table_size{};
  void* stream{};
  const DeviceExpertEntry* directory_entries{};
  std::uint32_t directory_layer{};
};

struct MoeAggregateLaunch final {
  const float* selection_outputs{};  // primary [rows, top_k, hidden]
  const float* alternate_outputs{};  // compact [alternate_output_count, hidden]
  const std::uint8_t* primary_mask{};  // 1 selects primary, 0 secondary
  const std::uint32_t* alternate_slot_by_selection{}; // [rows, top_k]
  const float* routing_weights{};     // [rows, top_k]
  float* output{};                    // [rows, hidden]
  std::uint32_t alternate_output_count{};
  std::uint32_t rows{};
  std::uint32_t hidden_size{};
  std::uint32_t top_k{};
  void* stream{};
};

// Exact Expert Pack INT8-per-row decode path. The first launch computes fused
// gate/up/SiLU for every selected expert. The second computes down projections
// and accumulates experts in routing order, avoiding atomics and preserving a
// deterministic numeric order.
[[nodiscard]] Status launch_moe_single_token(const MoeLaunch& launch) noexcept;
[[nodiscard]] Status launch_moe_batch(const MoeBatchLaunch& launch) noexcept;
[[nodiscard]] Status launch_moe_selection_batch(
    const MoeSelectionBatchLaunch& launch) noexcept;
[[nodiscard]] Status launch_moe_aggregate(
    const MoeAggregateLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
