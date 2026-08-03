#pragma once

#include "expert/runtime/storage.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace expert::runtime {

class Sha256 final {
 public:
  Sha256() noexcept;
  void update(std::span<const std::byte> input) noexcept;
  [[nodiscard]] Sha256Digest finalize() noexcept;

 private:
  std::array<std::uint32_t, 8> state_{};
  std::array<std::byte, 64> tail_{};
  std::size_t tail_bytes_{};
  std::uint64_t total_bytes_{};
  Sha256Digest digest_{};
  bool finalized_{};
};

[[nodiscard]] Sha256Digest sha256(std::span<const std::byte> input) noexcept;
[[nodiscard]] bool constant_time_equal(const Sha256Digest& left,
                                       const Sha256Digest& right) noexcept;

}  // namespace expert::runtime
