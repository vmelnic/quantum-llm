#pragma once

#include "expert/runtime/expert_key.hpp"
#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace expert::runtime {

inline constexpr std::uint16_t kExpertPackVersion = 1;
inline constexpr std::uint32_t kExpertHeaderBytes = 256;
inline constexpr std::uint32_t kExpertPackAlignment = 4096;
inline constexpr std::uint32_t kExpertQuantAbiInt8PerRow = 1;

struct ExpertSections final {
  std::uint32_t hidden{};
  std::uint32_t intermediate{};
  std::uint64_t gate_up_q_offset{};
  std::uint64_t gate_up_q_bytes{};
  std::uint64_t gate_up_scale_offset{};
  std::uint64_t gate_up_scale_bytes{};
  std::uint64_t down_q_offset{};
  std::uint64_t down_q_bytes{};
  std::uint64_t down_scale_offset{};
  std::uint64_t down_scale_bytes{};
};

struct ValidatedExpertRecord final {
  ExpertSections sections;
  std::span<const std::byte> complete_record;
  std::span<const std::byte> payload;
};

struct ExpertRecordValidation final {
  Status status;
  ValidatedExpertRecord record;
};

// Validates both the manifest claims and the EPEXPR01 header before returning
// any compute-visible span. SHA-256 covers bytes [record+256, record+stored),
// including section and end padding.
[[nodiscard]] ExpertRecordValidation validate_expert_record(
    std::span<const std::byte> bytes, const ExpertKey& expected_key,
    const PayloadRecord& expected_record) noexcept;

}  // namespace expert::runtime

