#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace expert::core {

[[nodiscard]] std::string Sha256Hex(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::string Sha256Hex(std::string_view bytes);

}  // namespace expert::core
