#include "expert/runtime/expert_record.hpp"

#include "expert/runtime/sha256.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <string>

namespace expert::runtime {
namespace {

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'E'}, std::byte{'P'}, std::byte{'E'}, std::byte{'X'},
    std::byte{'P'}, std::byte{'R'}, std::byte{'0'}, std::byte{'1'}};
constexpr std::uint32_t kRequiredFlags = 0x0fU;
constexpr std::size_t kStructuredHeaderBytes = 148;
constexpr std::uint32_t kSectionAlignment = 256;

template <typename T>
T read_le(const std::byte* bytes) noexcept {
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned value = 0;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    value |= static_cast<Unsigned>(std::to_integer<unsigned>(bytes[index]))
             << static_cast<unsigned>(index * 8U);
  }
  return static_cast<T>(value);
}

ExpertRecordValidation failure(ErrorCode code, std::string message) noexcept {
  return {Status(code, std::move(message)), {}};
}

bool multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t& result) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

bool valid_section(std::uint64_t offset, std::uint64_t bytes,
                   std::uint64_t previous_end,
                   std::uint64_t record_bytes) noexcept {
  return offset >= previous_end && offset % kSectionAlignment == 0 &&
         bytes <= record_bytes && offset <= record_bytes - bytes;
}

}  // namespace

ExpertRecordValidation validate_expert_record(
    std::span<const std::byte> bytes, const ExpertKey& key,
    const PayloadRecord& expected) noexcept {
  if (expected.header_bytes != kExpertHeaderBytes ||
      expected.alignment != kExpertPackAlignment ||
      expected.record_offset % kExpertPackAlignment != 0 ||
      expected.stored_bytes < kExpertHeaderBytes ||
      expected.stored_bytes % kExpertPackAlignment != 0 ||
      expected.stored_bytes != bytes.size()) {
    return failure(ErrorCode::invalid_argument,
                   "manifest record geometry violates Expert Pack v1");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    return failure(ErrorCode::checksum_mismatch, "expert record magic mismatch");
  }
  const auto* raw = bytes.data();
  const auto version = read_le<std::uint16_t>(raw + 8);
  const auto header_bytes = read_le<std::uint16_t>(raw + 10);
  const auto flags = read_le<std::uint32_t>(raw + 12);
  const auto quant_abi = read_le<std::uint32_t>(raw + 16);
  const auto layer = read_le<std::int32_t>(raw + 20);
  const auto expert = read_le<std::int32_t>(raw + 24);
  const auto hidden = read_le<std::uint32_t>(raw + 28);
  const auto intermediate = read_le<std::uint32_t>(raw + 32);
  const auto fused_rows = read_le<std::uint32_t>(raw + 36);
  const auto reserved = read_le<std::uint32_t>(raw + 40);
  const auto record_bytes = read_le<std::uint64_t>(raw + 44);

  if (version != kExpertPackVersion || header_bytes != kExpertHeaderBytes ||
      flags != kRequiredFlags || quant_abi != kExpertQuantAbiInt8PerRow ||
      quant_abi != key.quant_abi || reserved != 0 ||
      layer < 0 || expert < 0 || static_cast<std::uint32_t>(layer) != key.layer ||
      static_cast<std::uint32_t>(expert) != key.expert ||
      record_bytes != expected.stored_bytes || hidden == 0 || intermediate == 0 ||
      fused_rows != 2U * intermediate) {
    return failure(ErrorCode::checksum_mismatch,
                   "expert header/manifest ABI mismatch");
  }
  for (std::size_t index = kStructuredHeaderBytes;
       index < kExpertHeaderBytes; ++index) {
    if (raw[index] != std::byte{0}) {
      return failure(ErrorCode::checksum_mismatch,
                     "expert header padding is non-zero");
    }
  }

  ExpertSections sections;
  sections.hidden = hidden;
  sections.intermediate = intermediate;
  sections.gate_up_q_offset = read_le<std::uint64_t>(raw + 52);
  sections.gate_up_q_bytes = read_le<std::uint64_t>(raw + 60);
  sections.gate_up_scale_offset = read_le<std::uint64_t>(raw + 68);
  sections.gate_up_scale_bytes = read_le<std::uint64_t>(raw + 76);
  sections.down_q_offset = read_le<std::uint64_t>(raw + 84);
  sections.down_q_bytes = read_le<std::uint64_t>(raw + 92);
  sections.down_scale_offset = read_le<std::uint64_t>(raw + 100);
  sections.down_scale_bytes = read_le<std::uint64_t>(raw + 108);

  std::uint64_t hidden_intermediate = 0;
  if (!multiply(hidden, intermediate, hidden_intermediate) ||
      sections.gate_up_q_bytes != 2U * hidden_intermediate ||
      sections.gate_up_scale_bytes != 2ULL * intermediate * sizeof(float) ||
      sections.down_q_bytes != hidden_intermediate ||
      sections.down_scale_bytes != static_cast<std::uint64_t>(hidden) * sizeof(float)) {
    return failure(ErrorCode::checksum_mismatch,
                   "expert section dimensions are inconsistent");
  }
  std::uint64_t previous_end = kExpertHeaderBytes;
  const std::array<std::pair<std::uint64_t, std::uint64_t>, 4> spans = {{
      {sections.gate_up_q_offset, sections.gate_up_q_bytes},
      {sections.gate_up_scale_offset, sections.gate_up_scale_bytes},
      {sections.down_q_offset, sections.down_q_bytes},
      {sections.down_scale_offset, sections.down_scale_bytes},
  }};
  for (const auto& [offset, length] : spans) {
    if (!valid_section(offset, length, previous_end, record_bytes)) {
      return failure(ErrorCode::checksum_mismatch,
                     "expert section is unaligned, overlapping, or out of range");
    }
    previous_end = offset + length;
  }
  const auto expected_decoded = 3ULL * hidden_intermediate * sizeof(float);
  if (expected.decoded_bytes != 0 && expected.decoded_bytes != expected_decoded) {
    return failure(ErrorCode::checksum_mismatch,
                   "expert decoded byte count mismatch");
  }

  Sha256Digest header_digest{};
  std::copy_n(raw + 116, header_digest.size(), header_digest.begin());
  if (!constant_time_equal(header_digest, expected.payload_sha256)) {
    return failure(ErrorCode::checksum_mismatch,
                   "header and manifest payload digests disagree");
  }
  const auto payload = bytes.subspan(kExpertHeaderBytes);
  const auto actual_digest = sha256(payload);
  if (!constant_time_equal(actual_digest, header_digest)) {
    return failure(ErrorCode::checksum_mismatch,
                   "expert payload SHA-256 mismatch");
  }

  return {Status::success(), {sections, bytes, payload}};
}

}  // namespace expert::runtime
