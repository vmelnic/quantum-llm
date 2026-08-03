#pragma once

#include "expert/runtime/storage.hpp"

#include <span>

namespace expert::runtime {

[[nodiscard]] Sha256Digest sha256(std::span<const std::byte> input) noexcept;
[[nodiscard]] bool constant_time_equal(const Sha256Digest& left,
                                       const Sha256Digest& right) noexcept;

}  // namespace expert::runtime

