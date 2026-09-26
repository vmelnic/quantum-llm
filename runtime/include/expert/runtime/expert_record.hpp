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
inline constexpr std::uint32_t kExpertEncodingAbiInt8PerRow = 1;
inline constexpr std::uint32_t kExpertEncodingAbiFp4Block32 = 2;
inline constexpr std::uint32_t kExpertEncodingAbiNvfp4Block16W4A4 = 3;
// On-record quantization ABI IDs used by the Expert Pack v1 header.
inline constexpr std::uint32_t kExpertRecordAbiInt8PerRow = 1;
inline constexpr std::uint32_t kExpertRecordAbiFp4Block32 = 3;
inline constexpr std::uint32_t kExpertRecordAbiFp4Relu2Block32 = 4;
inline constexpr std::uint32_t kExpertRecordAbiNvfp4Block16W4A4 = 5;
inline constexpr std::uint32_t kDenseRecordAbiMxfp6E3m2Block32 = 6;
// Compatibility names for existing artifact adapters. New common code uses
// encoding ABI, source ABI, and record ABI as separate fields.
inline constexpr std::uint32_t kExpertQuantAbiInt8PerRow =
    kExpertEncodingAbiInt8PerRow;
// FP4-E2M1 packed nibbles with one UE8M0 scale per 32-value block along each
// output row, stored in a standard EPEXPR01 record.
inline constexpr std::uint32_t kExpertQuantAbiFp4Block32 =
    kExpertRecordAbiFp4Block32;
inline constexpr std::uint32_t kExpertFp4BlockSize = 32;
inline constexpr std::uint32_t kExpertNvfp4BlockSize = 16;
inline constexpr std::uint32_t kDenseMxfp6BlockSize = 32;
inline constexpr std::uint32_t kExpertSourceAbiExpertPackV1 = 1;

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

struct SplitExpertSections final {
  std::uint64_t w1_weight_offset{};
  std::uint64_t w1_weight_bytes{};
  std::uint64_t w1_scale_offset{};
  std::uint64_t w1_scale_bytes{};
  std::uint64_t w3_weight_offset{};
  std::uint64_t w3_weight_bytes{};
  std::uint64_t w3_scale_offset{};
  std::uint64_t w3_scale_bytes{};
  std::uint64_t w2_weight_offset{};
  std::uint64_t w2_weight_bytes{};
  std::uint64_t w2_scale_offset{};
  std::uint64_t w2_scale_bytes{};
};

struct ExpertAdmissionValidation final {
  Status status;
  ExpertSections target;
  SplitExpertSections compact;
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

// Representation-aware cache gate for Expert Pack records.
[[nodiscard]] ExpertAdmissionValidation validate_expert_admission(
    std::span<const std::byte> bytes, const ExpertKey& expected_key,
    const PayloadRecord& expected_record,
    bool verify_payload_sha256 = true) noexcept;

}  // namespace expert::runtime
