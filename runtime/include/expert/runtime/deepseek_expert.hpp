#pragma once

#include <cstdint>
#include <string_view>

namespace expert::runtime {

inline constexpr std::string_view kDeepSeekCompactAbi =
    "deepseek-fp4-e2m1-ue8m0-block32-v1";
inline constexpr std::string_view kDeepSeekFp8Block128Abi =
    "deepseek-fp8-e4m3-ue8m0-block128-v1";
inline constexpr std::string_view kDeepSeekSm86HotAbi =
    "deepseek-sm86-int8-per-row-v1";

enum class DeepSeekProjection : std::uint8_t { w1_gate, w3_up, w2_down };

struct DeepSeekMatrixGeometry {
  std::uint32_t rows{};
  std::uint32_t columns{};
};

struct DeepSeekExpertGeometry {
  std::uint32_t hidden{};
  std::uint32_t intermediate{};
  std::uint32_t fp4_block_size{};

  [[nodiscard]] static constexpr DeepSeekExpertGeometry v4_flash() noexcept {
    return {4096U, 2048U, 32U};
  }

  [[nodiscard]] DeepSeekMatrixGeometry matrix(
      DeepSeekProjection projection) const noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::uint64_t compact_weight_bytes(
      DeepSeekProjection projection) const noexcept;
  [[nodiscard]] std::uint64_t compact_scale_bytes(
      DeepSeekProjection projection) const noexcept;
  [[nodiscard]] std::uint64_t compact_expert_bytes() const noexcept;
};

struct DeepSeekHotSection {
  std::uint64_t offset{};
  std::uint64_t bytes{};
};

// Device slot layout. Gate (w1) and up (w3) are fused by row; down (w2) is
// separate. Every scale is little-endian FP32 and applies to one output row.
struct DeepSeekSm86HotLayout {
  DeepSeekHotSection gate_up_q;
  DeepSeekHotSection gate_up_scales;
  DeepSeekHotSection down_q;
  DeepSeekHotSection down_scales;
  std::uint64_t slot_bytes{};
  std::uint32_t alignment{};
};

[[nodiscard]] DeepSeekSm86HotLayout make_deepseek_sm86_hot_layout(
    const DeepSeekExpertGeometry& geometry,
    std::uint32_t alignment = 256U);

}  // namespace expert::runtime
