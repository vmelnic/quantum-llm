#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace expert::runtime {

// Identifies one immutable, compute-ready expert record. model_id is derived
// from the validated manifest hash; quant_abi prevents aliasing layouts.
struct ExpertKey final {
  std::uint64_t model_id{};
  std::uint32_t layer{};
  std::uint32_t expert{};
  std::uint32_t quant_abi{};

  friend constexpr bool operator==(const ExpertKey&, const ExpertKey&) = default;
  friend constexpr auto operator<=>(const ExpertKey&, const ExpertKey&) = default;
};

struct ExpertKeyHash final {
  [[nodiscard]] std::size_t operator()(const ExpertKey& key) const noexcept {
    auto value = static_cast<std::size_t>(key.model_id ^ (key.model_id >> 32U));
    const auto mix = [&value](std::uint32_t part) {
      value ^= static_cast<std::size_t>(part) + 0x9e3779b9U + (value << 6U) +
               (value >> 2U);
    };
    mix(key.layer);
    mix(key.expert);
    mix(key.quant_abi);
    return value;
  }
};

}  // namespace expert::runtime

