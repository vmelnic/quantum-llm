#include "expert/runtime/expert_record.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace expert::runtime {
namespace {

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'E'}, std::byte{'P'}, std::byte{'E'}, std::byte{'X'},
    std::byte{'P'}, std::byte{'R'}, std::byte{'0'}, std::byte{'1'}};
constexpr std::uint32_t kRequiredFlags = 0x0fU;
constexpr std::uint32_t kRelu2RequiredFlags = 0x0dU;
constexpr std::uint32_t kNvfp4RequiredFlags = 0x07U;
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

ExpertAdmissionValidation admission_failure(ErrorCode code,
                                             std::string message) noexcept {
  return {Status(code, std::move(message)), {}, {}};
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

bool align_up(std::uint64_t value, std::uint64_t alignment,
              std::uint64_t& result) noexcept {
  if (alignment == 0U || value >
      std::numeric_limits<std::uint64_t>::max() - (alignment - 1U))
    return false;
  result = (value + alignment - 1U) / alignment * alignment;
  return true;
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
  const auto record_abi = read_le<std::uint32_t>(raw + 16);
  const auto layer = read_le<std::int32_t>(raw + 20);
  const auto expert = read_le<std::int32_t>(raw + 24);
  const auto hidden = read_le<std::uint32_t>(raw + 28);
  const auto intermediate = read_le<std::uint32_t>(raw + 32);
  const auto fused_rows = read_le<std::uint32_t>(raw + 36);
  const auto reserved = read_le<std::uint32_t>(raw + 40);
  const auto record_bytes = read_le<std::uint64_t>(raw + 44);
  const bool relu2 = record_abi == kExpertRecordAbiFp4Relu2Block32;
  const bool native_nvfp4 =
      record_abi == kExpertRecordAbiNvfp4Block16W4A4;

  if (version != kExpertPackVersion || header_bytes != kExpertHeaderBytes ||
      flags != (native_nvfp4 ? kNvfp4RequiredFlags :
               relu2 ? kRelu2RequiredFlags : kRequiredFlags) ||
      (record_abi != kExpertRecordAbiInt8PerRow &&
       record_abi != kExpertRecordAbiFp4Block32 &&
       record_abi != kExpertRecordAbiFp4Relu2Block32 &&
       record_abi != kExpertRecordAbiNvfp4Block16W4A4) ||
      (expected.record_abi != 0U && record_abi != expected.record_abi) ||
      (native_nvfp4 ? kExpertEncodingAbiNvfp4Block16W4A4 :
       record_abi == kExpertRecordAbiFp4Block32 || relu2
           ? kExpertEncodingAbiFp4Block32
           : kExpertEncodingAbiInt8PerRow) != key.encoding_abi ||
      reserved != 0 ||
      layer < 0 || expert < 0 || static_cast<std::uint32_t>(layer) != key.layer ||
      static_cast<std::uint32_t>(expert) != key.expert ||
      record_bytes != expected.stored_bytes || hidden == 0 || intermediate == 0 ||
      fused_rows != (relu2 ? intermediate : 2U * intermediate)) {
    return failure(ErrorCode::checksum_mismatch,
                   "expert header/manifest ABI mismatch");
  }
  const bool fp4 = record_abi == kExpertRecordAbiFp4Block32 || relu2;
  if (native_nvfp4 &&
      (hidden % kExpertNvfp4BlockSize != 0U ||
       intermediate % kExpertNvfp4BlockSize != 0U)) {
    return failure(ErrorCode::checksum_mismatch,
                   "NVFP4 expert geometry is not block-aligned");
  }
  if (fp4 && (hidden % kExpertFp4BlockSize != 0 ||
              intermediate % kExpertFp4BlockSize != 0)) {
    return failure(ErrorCode::checksum_mismatch,
                   "FP4 expert geometry is not block-aligned");
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
  if (!multiply(hidden, intermediate, hidden_intermediate)) {
    return failure(ErrorCode::checksum_mismatch,
                   "expert section dimensions are inconsistent");
  }
  if (native_nvfp4) {
    if (sections.gate_up_q_bytes != hidden_intermediate ||
        sections.gate_up_scale_bytes !=
            2ULL * (hidden_intermediate / kExpertNvfp4BlockSize +
                    2U * sizeof(float)) ||
        sections.down_q_bytes != hidden_intermediate / 2U ||
        sections.down_scale_bytes !=
            hidden_intermediate / kExpertNvfp4BlockSize +
                2U * sizeof(float)) {
      return failure(ErrorCode::checksum_mismatch,
                     "NVFP4 expert section dimensions are inconsistent");
    }
  } else if (fp4) {
    if (sections.gate_up_q_bytes !=
            (relu2 ? hidden_intermediate / 2U : hidden_intermediate) ||
        sections.gate_up_scale_bytes !=
            (relu2 ? hidden_intermediate : 2ULL * hidden_intermediate) /
                kExpertFp4BlockSize ||
        sections.down_q_bytes != hidden_intermediate / 2U ||
        sections.down_scale_bytes !=
            hidden_intermediate / kExpertFp4BlockSize) {
      return failure(ErrorCode::checksum_mismatch,
                     "expert section dimensions are inconsistent");
    }
  } else if (
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
  if (native_nvfp4) {
    const auto local = hidden_intermediate / kExpertNvfp4BlockSize;
    const std::array<std::uint64_t, 3U> scale_offsets = {
        sections.gate_up_scale_offset,
        sections.gate_up_scale_offset + local + 2U * sizeof(float),
        sections.down_scale_offset,
    };
    for (const auto offset : scale_offsets) {
      const auto begin = bytes.begin() + static_cast<std::size_t>(offset);
      const auto end = begin + static_cast<std::size_t>(local);
      if (std::find_if(begin, end, [](std::byte value) {
            const auto code = std::to_integer<std::uint8_t>(value);
            return (code & 0x80U) != 0U || (code & 0x7fU) == 0x7fU;
          }) != end) {
        return failure(ErrorCode::checksum_mismatch,
                       "NVFP4 expert contains an invalid E4M3FN scale");
      }
      float weight_global{};
      float input_global{};
      std::memcpy(&weight_global,
                  bytes.data() + offset + local, sizeof(float));
      std::memcpy(&input_global,
                  bytes.data() + offset + local + sizeof(float),
                  sizeof(float));
      if (!std::isfinite(weight_global) ||
          !std::isfinite(input_global) ||
          !(weight_global > 0.0F) || !(input_global > 0.0F)) {
        return failure(ErrorCode::checksum_mismatch,
                       "NVFP4 expert contains an invalid global divisor");
      }
    }
  } else if (fp4) {
    // UE8M0 scales must be finite (0xff is NaN) and unambiguous: the compiler
    // clamps codes to [1, 254] because code 0 decodes inconsistently between
    // toolchain and kernel paths.
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 2> scale_spans = {{
        {sections.gate_up_scale_offset, sections.gate_up_scale_bytes},
        {sections.down_scale_offset, sections.down_scale_bytes},
    }};
    for (const auto& [offset, length] : scale_spans) {
      const auto begin = bytes.begin() + static_cast<std::size_t>(offset);
      const auto end = begin + static_cast<std::size_t>(length);
      if (std::find_if(begin, end, [](std::byte code) {
            return code == std::byte{0x00} || code == std::byte{0xff};
          }) != end) {
        return failure(ErrorCode::checksum_mismatch,
                       "FP4 expert contains an invalid UE8M0 scale code");
      }
    }
  }
  const auto expected_decoded = (relu2 ? 2ULL : 3ULL) * hidden_intermediate *
                                sizeof(float);
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

ExpertAdmissionValidation validate_expert_admission(
    std::span<const std::byte> bytes, const ExpertKey& key,
    const PayloadRecord& expected, bool verify_payload_sha256) noexcept {
  if ((key.encoding_abi == kExpertEncodingAbiInt8PerRow ||
       key.encoding_abi == kExpertEncodingAbiFp4Block32 ||
       key.encoding_abi == kExpertEncodingAbiNvfp4Block16W4A4) &&
      expected.source_abi == kExpertSourceAbiExpertPackV1) {
    const auto validated = validate_expert_record(bytes, key, expected);
    SplitExpertSections split{};
    if (validated.status.ok() &&
        (key.encoding_abi == kExpertEncodingAbiFp4Block32 ||
         key.encoding_abi == kExpertEncodingAbiNvfp4Block16W4A4)) {
      const auto& sections = validated.record.sections;
      const auto matrix_bytes =
          static_cast<std::uint64_t>(sections.hidden) *
          sections.intermediate / 2U;
      const auto native =
          key.encoding_abi == kExpertEncodingAbiNvfp4Block16W4A4;
      const auto scale_bytes =
          static_cast<std::uint64_t>(sections.hidden) *
              sections.intermediate /
              (native ? kExpertNvfp4BlockSize : kExpertFp4BlockSize) +
          (native ? 2U * sizeof(float) : 0U);
      if (expected.record_abi == kExpertRecordAbiFp4Relu2Block32) {
        split = {
            sections.gate_up_q_offset, matrix_bytes,
            sections.gate_up_scale_offset, scale_bytes,
            0U, 0U, 0U, 0U,
            sections.down_q_offset, matrix_bytes,
            sections.down_scale_offset, scale_bytes};
      } else {
        split = {
            sections.gate_up_q_offset, matrix_bytes,
            sections.gate_up_scale_offset, scale_bytes,
            sections.gate_up_q_offset + matrix_bytes, matrix_bytes,
            sections.gate_up_scale_offset + scale_bytes, scale_bytes,
            sections.down_q_offset, matrix_bytes,
            sections.down_scale_offset, scale_bytes};
      }
    }
    return {validated.status, validated.record.sections, split};
  }
  if (key.encoding_abi != kExpertEncodingAbiFp4Block32 ||
      (expected.source_abi != kExpertSourceAbiDeepSeekCompactV1 &&
       expected.source_abi != kExpertSourceAbiDeepSeekFp8Block128V1)) {
    return admission_failure(ErrorCode::invalid_argument,
                             "unsupported source/target expert ABI pair");
  }
  const bool fp8 =
      expected.source_abi == kExpertSourceAbiDeepSeekFp8Block128V1;
  const auto hidden = expected.hidden == 0U ? 4096U : expected.hidden;
  const auto intermediate =
      expected.intermediate == 0U ? 2048U : expected.intermediate;
  const auto block = expected.quant_block_size == 0U
                         ? (fp8 ? 128U : kExpertFp4BlockSize)
                         : expected.quant_block_size;
  std::uint64_t elements = 0U;
  if (!multiply(hidden, intermediate, elements) ||
      (fp8 && (block != 128U || hidden % block != 0U ||
               intermediate % block != 0U)) ||
      (!fp8 && (block != kExpertFp4BlockSize || hidden % block != 0U ||
                intermediate % block != 0U)))
    return admission_failure(ErrorCode::checksum_mismatch,
                             "split expert source geometry is invalid");
  const std::uint64_t weight_bytes = fp8 ? elements : elements / 2U;
  const std::uint64_t scale_bytes =
      fp8 ? static_cast<std::uint64_t>(hidden / block) *
                (intermediate / block)
          : elements / block;
  const std::uint64_t source_bytes = 3U * (weight_bytes + scale_bytes);
  std::uint64_t gate_up_scale_offset = 0U;
  std::uint64_t down_offset = 0U;
  std::uint64_t down_scale_offset = 0U;
  std::uint64_t hot_bytes = 0U;
  const auto gate_up_bytes = 2U * elements;
  const auto gate_up_scale_bytes =
      2ULL * intermediate * sizeof(float);
  const auto down_bytes = elements;
  const auto down_scale_bytes = static_cast<std::uint64_t>(hidden) * sizeof(float);
  if (!align_up(gate_up_bytes, kSectionAlignment, gate_up_scale_offset) ||
      !align_up(gate_up_scale_offset + gate_up_scale_bytes,
                kSectionAlignment, down_offset) ||
      !align_up(down_offset + down_bytes, kSectionAlignment,
                down_scale_offset) ||
      !align_up(down_scale_offset + down_scale_bytes, kSectionAlignment,
                hot_bytes))
    return admission_failure(ErrorCode::checksum_mismatch,
                             "split expert target geometry overflows");
  // FP8 source records expand to INT8-per-row. FP4 source records execute
  // directly from the packed bytes, but target section metadata still
  // describes the optional expanded view.
  const std::uint64_t device_bytes = fp8 ? hot_bytes : source_bytes;
  if (expected.stored_bytes != source_bytes || bytes.size() != source_bytes ||
      expected.device_bytes != device_bytes || expected.header_bytes != 0U ||
      expected.decoded_bytes != 3ULL * elements * sizeof(float)) {
    return admission_failure(ErrorCode::checksum_mismatch,
                             "DeepSeek compact admission geometry mismatch");
  }
  if (verify_payload_sha256 &&
      !constant_time_equal(sha256(bytes), expected.payload_sha256)) {
    return admission_failure(ErrorCode::checksum_mismatch,
                             "DeepSeek compact payload SHA-256 mismatch");
  }
  DeepSeekCompactSections compact{};
  compact.w1_weight_offset = 0U;
  compact.w1_weight_bytes = weight_bytes;
  compact.w1_scale_offset = compact.w1_weight_offset + weight_bytes;
  compact.w1_scale_bytes = scale_bytes;
  compact.w3_weight_offset = compact.w1_scale_offset + scale_bytes;
  compact.w3_weight_bytes = weight_bytes;
  compact.w3_scale_offset = compact.w3_weight_offset + weight_bytes;
  compact.w3_scale_bytes = scale_bytes;
  compact.w2_weight_offset = compact.w3_scale_offset + scale_bytes;
  compact.w2_weight_bytes = weight_bytes;
  compact.w2_scale_offset = compact.w2_weight_offset + weight_bytes;
  compact.w2_scale_bytes = scale_bytes;
  if (!fp8) {
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 3U> scales = {{
        {compact.w1_scale_offset, compact.w1_scale_bytes},
        {compact.w3_scale_offset, compact.w3_scale_bytes},
        {compact.w2_scale_offset, compact.w2_scale_bytes},
    }};
    for (const auto& [offset, count] : scales) {
      const auto begin = bytes.begin() + static_cast<std::size_t>(offset);
      const auto end = begin + static_cast<std::size_t>(count);
      if (std::find(begin, end, std::byte{0xff}) != end) {
        return admission_failure(ErrorCode::checksum_mismatch,
                                 "DeepSeek compact source contains UE8M0 NaN");
      }
    }
  }
  ExpertSections target{};
  target.hidden = hidden;
  target.intermediate = intermediate;
  target.gate_up_q_offset = 0U;
  target.gate_up_q_bytes = gate_up_bytes;
  target.gate_up_scale_offset = gate_up_scale_offset;
  target.gate_up_scale_bytes = gate_up_scale_bytes;
  target.down_q_offset = down_offset;
  target.down_q_bytes = down_bytes;
  target.down_scale_offset = down_scale_offset;
  target.down_scale_bytes = down_scale_bytes;
  return {Status::success(), target, compact};
}

}  // namespace expert::runtime
