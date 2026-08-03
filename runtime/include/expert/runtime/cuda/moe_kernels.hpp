#pragma once

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
};

// Exact Expert Pack INT8-per-row decode path. The first launch computes fused
// gate/up/SiLU for every selected expert. The second computes down projections
// and accumulates experts in routing order, avoiding atomics and preserving a
// deterministic numeric order.
[[nodiscard]] Status launch_moe_single_token(const MoeLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
