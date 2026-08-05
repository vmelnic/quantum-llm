#include "expert/runtime/deepseek_expert.hpp"

#include <initializer_list>
#include <stdexcept>

namespace expert::runtime {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint32_t alignment) {
  if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
    throw std::invalid_argument("DeepSeek hot-cache alignment must be a power of two");
  }
  return (value + alignment - 1U) & ~(static_cast<std::uint64_t>(alignment) - 1U);
}

}  // namespace

DeepSeekMatrixGeometry DeepSeekExpertGeometry::matrix(
    DeepSeekProjection projection) const noexcept {
  if (projection == DeepSeekProjection::w2_down) {
    return {hidden, intermediate};
  }
  return {intermediate, hidden};
}

bool DeepSeekExpertGeometry::valid() const noexcept {
  return hidden != 0U && intermediate != 0U && fp4_block_size != 0U &&
         (fp4_block_size % 2U) == 0U && (hidden % fp4_block_size) == 0U &&
         (intermediate % fp4_block_size) == 0U;
}

std::uint64_t DeepSeekExpertGeometry::compact_weight_bytes(
    DeepSeekProjection projection) const noexcept {
  const auto shape = matrix(projection);
  return static_cast<std::uint64_t>(shape.rows) * shape.columns / 2U;
}

std::uint64_t DeepSeekExpertGeometry::compact_scale_bytes(
    DeepSeekProjection projection) const noexcept {
  const auto shape = matrix(projection);
  return static_cast<std::uint64_t>(shape.rows) *
         (shape.columns / fp4_block_size);
}

std::uint64_t DeepSeekExpertGeometry::compact_expert_bytes() const noexcept {
  std::uint64_t bytes = 0U;
  for (const auto projection : {DeepSeekProjection::w1_gate,
                                DeepSeekProjection::w3_up,
                                DeepSeekProjection::w2_down}) {
    bytes += compact_weight_bytes(projection) + compact_scale_bytes(projection);
  }
  return bytes;
}

DeepSeekSm86HotLayout make_deepseek_sm86_hot_layout(
    const DeepSeekExpertGeometry& geometry, std::uint32_t alignment) {
  if (!geometry.valid()) {
    throw std::invalid_argument("invalid DeepSeek expert geometry");
  }
  DeepSeekSm86HotLayout layout{};
  layout.alignment = alignment;
  std::uint64_t cursor = 0U;
  const auto gate = geometry.matrix(DeepSeekProjection::w1_gate);
  const auto down = geometry.matrix(DeepSeekProjection::w2_down);

  layout.gate_up_q = {cursor,
                      2U * static_cast<std::uint64_t>(gate.rows) * gate.columns};
  cursor = align_up(cursor + layout.gate_up_q.bytes, alignment);
  layout.gate_up_scales = {
      cursor, 2U * static_cast<std::uint64_t>(gate.rows) * sizeof(float)};
  cursor = align_up(cursor + layout.gate_up_scales.bytes, alignment);
  layout.down_q = {cursor,
                   static_cast<std::uint64_t>(down.rows) * down.columns};
  cursor = align_up(cursor + layout.down_q.bytes, alignment);
  layout.down_scales = {
      cursor, static_cast<std::uint64_t>(down.rows) * sizeof(float)};
  layout.slot_bytes = align_up(cursor + layout.down_scales.bytes, alignment);
  return layout;
}

}  // namespace expert::runtime
