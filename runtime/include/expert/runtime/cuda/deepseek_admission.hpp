#pragma once

#include "expert/runtime/storage.hpp"

#include <cstdint>

namespace expert::runtime::cuda {

struct DeepSeekAdmissionLaunch {
  const std::uint8_t* packed_fp4{};
  const std::uint8_t* ue8m0_scales{};
  std::int8_t* int8_rows{};
  float* row_scales{};
  std::uint32_t rows{};
  std::uint32_t columns{};
  void* stream{};
};

struct DeepSeekFp8AdmissionLaunch {
  const std::uint8_t* fp8_e4m3{};
  const std::uint8_t* ue8m0_scales{};
  std::int8_t* int8_rows{};
  float* row_scales{};
  std::uint32_t rows{};
  std::uint32_t columns{};
  void* stream{};
};

// Converts one projection into deepseek-sm86-int8-per-row-v1. The call is
// synchronous only for the source-validity flag; cache publication happens
// after it returns success.
[[nodiscard]] Status admit_deepseek_projection(
    const DeepSeekAdmissionLaunch& launch) noexcept;

// Converts an FP8 E4M3FN matrix with UE8M0 128x128 block scales into the same
// SM86 INT8-per-row target used by routed experts.
[[nodiscard]] Status admit_deepseek_fp8_projection(
    const DeepSeekFp8AdmissionLaunch& launch) noexcept;

}  // namespace expert::runtime::cuda
