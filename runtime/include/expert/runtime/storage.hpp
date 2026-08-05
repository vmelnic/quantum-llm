#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace expert::runtime {

enum class ErrorCode : std::uint8_t {
  ok,
  cancelled,
  backpressure,
  invalid_argument,
  open_failed,
  io_failed,
  short_read,
  checksum_mismatch,
  upload_failed,
  internal,
};

class Status final {
 public:
  Status() = default;
  Status(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status success() { return {}; }
  [[nodiscard]] bool ok() const noexcept { return code_ == ErrorCode::ok; }
  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] std::string_view message() const noexcept { return message_; }

 private:
  ErrorCode code_{ErrorCode::ok};
  std::string message_;
};

using Sha256Digest = std::array<std::byte, 32>;
using OperationId = std::uint64_t;

// One immutable source range gathered into a logical payload. Extents permit
// compact experts to remain in their authoritative SafeTensors shards.
struct PayloadExtent final {
  std::filesystem::path path;
  std::uint64_t source_offset{};
  std::uint64_t destination_offset{};
  std::uint64_t bytes{};

  friend bool operator==(const PayloadExtent&, const PayloadExtent&) = default;
};

struct PayloadRecord final {
  std::filesystem::path path;
  // Empty selects the legacy contiguous path/record_offset read. Otherwise
  // these ranges must exactly and non-overlappingly cover stored_bytes.
  std::vector<PayloadExtent> extents;
  std::uint64_t record_offset{};
  std::uint64_t stored_bytes{};
  std::uint64_t decoded_bytes{};
  // Exact device allocation produced by admission. Zero preserves the legacy
  // invariant that stored bytes and device bytes are equal.
  std::uint64_t device_bytes{};
  // Source payload ABI. One is Expert Pack v1; two is the compact DeepSeek
  // staging order validated before CUDA admission.
  std::uint32_t source_abi{1};
  std::uint32_t header_bytes{256};
  std::uint32_t alignment{4096};
  Sha256Digest payload_sha256{};
};

struct MutableBuffer final {
  std::byte* data{};
  std::size_t capacity{};
};

struct ReadRequest final {
  PayloadRecord record;
  MutableBuffer destination;
  bool direct{true};
};

struct ReadResult final {
  Status status;
  std::uint64_t requested_bytes{};
  std::uint64_t read_bytes{};
};

using ReadCompletion = std::function<void(ReadResult)>;

class IAsyncStorage {
 public:
  virtual ~IAsyncStorage() = default;
  virtual OperationId read(ReadRequest request, ReadCompletion completion) = 0;
  virtual void cancel(OperationId operation) noexcept = 0;
};

}  // namespace expert::runtime
