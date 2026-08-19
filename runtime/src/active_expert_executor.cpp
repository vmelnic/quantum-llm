#include "expert/runtime/active_expert_executor.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>

namespace expert::runtime {
namespace {

constexpr std::array<std::byte, 8U> kWireMagic{
    static_cast<std::byte>('Q'), static_cast<std::byte>('M'),
    static_cast<std::byte>('O'), static_cast<std::byte>('E'),
    static_cast<std::byte>('A'), static_cast<std::byte>('X'),
    static_cast<std::byte>('0'), static_cast<std::byte>('1')};
constexpr std::uint16_t kWireVersion = 1U;
constexpr std::uint16_t kRequestFrame = 1U;
constexpr std::uint16_t kResponseFrame = 2U;
constexpr std::uint64_t kMaximumFrameBytes = 64ULL << 20U;
constexpr std::uint32_t kMaximumStringBytes = 4096U;

Status copy_status(const Status& status) {
  return {status.code(), std::string(status.message())};
}

bool nonzero(const Sha256Digest& digest) noexcept {
  return std::any_of(digest.begin(), digest.end(),
                     [](std::byte value) { return value != std::byte{0}; });
}

Status validate_identity(const ActiveExpertIdentity& identity) {
  if (!nonzero(identity.model_content_hash) || identity.key.model_id == 0U ||
      identity.key.encoding_abi == 0U || identity.capability.empty() ||
      identity.execution_abi == 0U || identity.source_abi == 0U)
    return {ErrorCode::invalid_argument,
            "active-expert identity is incomplete"};
  return Status::success();
}

Status validate_invocation(const ActiveExpertInvocation& invocation) {
  if (invocation.request_id == 0U || invocation.invocation_id == 0U ||
      invocation.route_width == 0U ||
      invocation.selection_index >= invocation.route_width ||
      !invocation.input.valid() || invocation.output_abi.empty() ||
      invocation.output_bytes == 0U)
    return {ErrorCode::invalid_argument,
            "active-expert invocation is incomplete"};
  if (invocation.deadline != std::chrono::steady_clock::time_point::max() &&
      invocation.deadline <= std::chrono::steady_clock::now())
    return {ErrorCode::deadline_exceeded,
            "active-expert invocation deadline expired"};
  return Status::success();
}

class Writer final {
 public:
  void u8(std::uint8_t value) { bytes_.push_back(static_cast<std::byte>(value)); }
  void u16(std::uint16_t value) {
    for (unsigned shift = 0U; shift < 16U; shift += 8U)
      u8(static_cast<std::uint8_t>(value >> shift));
  }
  void u32(std::uint32_t value) {
    for (unsigned shift = 0U; shift < 32U; shift += 8U)
      u8(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (unsigned shift = 0U; shift < 64U; shift += 8U)
      u8(static_cast<std::uint8_t>(value >> shift));
  }
  void raw(std::span<const std::byte> value) {
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void text(std::string_view value) {
    u32(static_cast<std::uint32_t>(value.size()));
    raw(std::as_bytes(std::span(value.data(), value.size())));
  }
  [[nodiscard]] std::vector<std::byte> finish() { return std::move(bytes_); }
  [[nodiscard]] std::span<const std::byte> view() const noexcept {
    return bytes_;
  }

 private:
  std::vector<std::byte> bytes_;
};

class Reader final {
 public:
  explicit Reader(std::span<const std::byte> bytes) noexcept : bytes_(bytes) {}

  bool u8(std::uint8_t& value) noexcept {
    if (remaining() < 1U) return false;
    value = std::to_integer<std::uint8_t>(bytes_[cursor_++]);
    return true;
  }
  bool u16(std::uint16_t& value) noexcept {
    std::uint64_t wide{};
    if (!integer(2U, wide)) return false;
    value = static_cast<std::uint16_t>(wide);
    return true;
  }
  bool u32(std::uint32_t& value) noexcept {
    std::uint64_t wide{};
    if (!integer(4U, wide)) return false;
    value = static_cast<std::uint32_t>(wide);
    return true;
  }
  bool u64(std::uint64_t& value) noexcept { return integer(8U, value); }
  bool raw(std::size_t count, std::span<const std::byte>& value) noexcept {
    if (count > remaining()) return false;
    value = bytes_.subspan(cursor_, count);
    cursor_ += count;
    return true;
  }
  bool text(std::string& value) {
    std::uint32_t bytes{};
    std::span<const std::byte> raw_value;
    if (!u32(bytes) || bytes > kMaximumStringBytes ||
        !raw(bytes, raw_value))
      return false;
    value.assign(reinterpret_cast<const char*>(raw_value.data()),
                 raw_value.size());
    return value.find_first_of("\0\r\n") == std::string::npos;
  }
  [[nodiscard]] std::size_t remaining() const noexcept {
    return bytes_.size() - cursor_;
  }

 private:
  bool integer(std::size_t count, std::uint64_t& value) noexcept {
    if (count > remaining()) return false;
    value = 0U;
    for (std::size_t index = 0U; index < count; ++index)
      value |= static_cast<std::uint64_t>(
                   std::to_integer<std::uint8_t>(bytes_[cursor_ + index]))
               << (index * 8U);
    cursor_ += count;
    return true;
  }

  std::span<const std::byte> bytes_;
  std::size_t cursor_{};
};

bool header(Reader& reader, std::uint16_t expected_kind) noexcept {
  std::span<const std::byte> magic;
  std::uint16_t version{};
  std::uint16_t kind{};
  return reader.raw(kWireMagic.size(), magic) &&
         std::equal(magic.begin(), magic.end(), kWireMagic.begin()) &&
         reader.u16(version) && version == kWireVersion && reader.u16(kind) &&
         kind == expected_kind;
}

void header(Writer& writer, std::uint16_t kind) {
  writer.raw(kWireMagic);
  writer.u16(kWireVersion);
  writer.u16(kind);
}

void identity(Writer& writer, const ActiveExpertIdentity& value) {
  writer.raw(value.model_content_hash);
  writer.u64(value.key.model_id);
  writer.u32(value.key.layer);
  writer.u32(value.key.expert);
  writer.u32(value.key.encoding_abi);
  writer.u32(value.execution_abi);
  writer.u32(value.source_abi);
  writer.text(value.capability);
}

bool identity(Reader& reader, ActiveExpertIdentity& value) {
  std::span<const std::byte> hash;
  if (!reader.raw(value.model_content_hash.size(), hash)) return false;
  std::copy(hash.begin(), hash.end(), value.model_content_hash.begin());
  return reader.u64(value.key.model_id) && reader.u32(value.key.layer) &&
         reader.u32(value.key.expert) &&
         reader.u32(value.key.encoding_abi) &&
         reader.u32(value.execution_abi) && reader.u32(value.source_abi) &&
         reader.text(value.capability);
}

std::uint64_t deadline_budget(
    std::chrono::steady_clock::time_point deadline) noexcept {
  if (deadline == std::chrono::steady_clock::time_point::max())
    return std::numeric_limits<std::uint64_t>::max();
  const auto now = std::chrono::steady_clock::now();
  if (deadline <= now) return 0U;
  const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         deadline - now)
                         .count();
  return value <= 0 ? 0U : static_cast<std::uint64_t>(value);
}

std::chrono::steady_clock::time_point decode_deadline(
    std::uint64_t budget) noexcept {
  if (budget == std::numeric_limits<std::uint64_t>::max())
    return std::chrono::steady_clock::time_point::max();
  const auto maximum = static_cast<std::uint64_t>(
      std::chrono::nanoseconds::max().count());
  return std::chrono::steady_clock::now() +
         std::chrono::nanoseconds(std::min(budget, maximum));
}

struct AtomicTelemetry final {
  std::atomic<std::uint64_t> requests{0U};
  std::atomic<std::uint64_t> completed{0U};
  std::atomic<std::uint64_t> failed{0U};
  std::atomic<std::uint64_t> cancelled{0U};
  std::atomic<std::uint64_t> deadline_exceeded{0U};
  std::atomic<std::uint64_t> checksum_failures{0U};
  std::atomic<std::uint64_t> activation_input_bytes{0U};
  std::atomic<std::uint64_t> activation_output_bytes{0U};
  std::atomic<std::uint64_t> wire_request_bytes{0U};
  std::atomic<std::uint64_t> wire_response_bytes{0U};
  std::atomic<std::uint64_t> weight_transport_bytes{0U};
  std::atomic<std::uint64_t> owner_weight_read_bytes{0U};
  std::atomic<std::uint64_t> owner_storage_read_bytes{0U};
  std::atomic<std::uint64_t> owner_ram_read_bytes{0U};
  std::atomic<std::uint64_t> owner_vram_read_bytes{0U};
  std::atomic<std::uint64_t> owner_execution_ns{0U};
  std::atomic<std::uint64_t> transport_wait_ns{0U};

  void add_completion(const ActiveExpertExecutionEvidence& value) noexcept {
    activation_output_bytes.fetch_add(value.activation_output_bytes,
                                      std::memory_order_relaxed);
    wire_response_bytes.fetch_add(value.wire_response_bytes,
                                  std::memory_order_relaxed);
    weight_transport_bytes.fetch_add(value.weight_transport_bytes,
                                     std::memory_order_relaxed);
    owner_weight_read_bytes.fetch_add(value.owner_weight_read_bytes,
                                      std::memory_order_relaxed);
    owner_storage_read_bytes.fetch_add(value.owner_storage_read_bytes,
                                       std::memory_order_relaxed);
    owner_ram_read_bytes.fetch_add(value.owner_ram_read_bytes,
                                   std::memory_order_relaxed);
    owner_vram_read_bytes.fetch_add(value.owner_vram_read_bytes,
                                    std::memory_order_relaxed);
    owner_execution_ns.fetch_add(value.owner_execution_ns,
                                 std::memory_order_relaxed);
    transport_wait_ns.fetch_add(value.transport_wait_ns,
                                std::memory_order_relaxed);
  }

  [[nodiscard]] ActiveExpertExecutorTelemetry snapshot() const noexcept {
    ActiveExpertExecutorTelemetry result;
#define ACTIVE_EXPERT_SNAPSHOT(field) \
  result.field = field.load(std::memory_order_relaxed)
    ACTIVE_EXPERT_SNAPSHOT(requests);
    ACTIVE_EXPERT_SNAPSHOT(completed);
    ACTIVE_EXPERT_SNAPSHOT(failed);
    ACTIVE_EXPERT_SNAPSHOT(cancelled);
    ACTIVE_EXPERT_SNAPSHOT(deadline_exceeded);
    ACTIVE_EXPERT_SNAPSHOT(checksum_failures);
    ACTIVE_EXPERT_SNAPSHOT(activation_input_bytes);
    ACTIVE_EXPERT_SNAPSHOT(activation_output_bytes);
    ACTIVE_EXPERT_SNAPSHOT(wire_request_bytes);
    ACTIVE_EXPERT_SNAPSHOT(wire_response_bytes);
    ACTIVE_EXPERT_SNAPSHOT(weight_transport_bytes);
    ACTIVE_EXPERT_SNAPSHOT(owner_weight_read_bytes);
    ACTIVE_EXPERT_SNAPSHOT(owner_storage_read_bytes);
    ACTIVE_EXPERT_SNAPSHOT(owner_ram_read_bytes);
    ACTIVE_EXPERT_SNAPSHOT(owner_vram_read_bytes);
    ACTIVE_EXPERT_SNAPSHOT(owner_execution_ns);
    ACTIVE_EXPERT_SNAPSHOT(transport_wait_ns);
#undef ACTIVE_EXPERT_SNAPSHOT
    return result;
  }
};

void evidence(Writer& writer, const ActiveExpertExecutionEvidence& value) {
  writer.u64(value.activation_input_bytes);
  writer.u64(value.activation_output_bytes);
  writer.u64(value.wire_request_bytes);
  writer.u64(value.wire_response_bytes);
  writer.u64(value.weight_transport_bytes);
  writer.u64(value.owner_weight_read_bytes);
  writer.u64(value.owner_storage_read_bytes);
  writer.u64(value.owner_ram_read_bytes);
  writer.u64(value.owner_vram_read_bytes);
  writer.u64(value.owner_execution_ns);
  writer.u64(value.transport_wait_ns);
}

bool evidence(Reader& reader, ActiveExpertExecutionEvidence& value) noexcept {
  return reader.u64(value.activation_input_bytes) &&
         reader.u64(value.activation_output_bytes) &&
         reader.u64(value.wire_request_bytes) &&
         reader.u64(value.wire_response_bytes) &&
         reader.u64(value.weight_transport_bytes) &&
         reader.u64(value.owner_weight_read_bytes) &&
         reader.u64(value.owner_storage_read_bytes) &&
         reader.u64(value.owner_ram_read_bytes) &&
         reader.u64(value.owner_vram_read_bytes) &&
         reader.u64(value.owner_execution_ns) &&
         reader.u64(value.transport_wait_ns);
}

std::vector<std::byte> encode_response_once(
    const ActiveExpertExecutionResult& result) {
  Writer writer;
  header(writer, kResponseFrame);
  writer.u64(result.request_id);
  writer.u64(result.invocation_id);
  writer.u32(result.selection_index);
  identity(writer, result.identity);
  writer.raw(result.request_sha256);
  writer.u8(static_cast<std::uint8_t>(result.status.code()));
  writer.text(result.status.message());
  writer.text(result.status.ok() ? result.output.abi : std::string_view{});
  writer.u64(result.status.ok() ? result.output.bytes : 0U);
  evidence(writer, result.evidence);
  if (result.status.ok())
    writer.raw({result.output.data,
                static_cast<std::size_t>(result.output.bytes)});
  const auto digest = sha256(writer.view());
  writer.raw(digest);
  return writer.finish();
}

Status encode_response(ActiveExpertExecutionResult& result,
                       std::vector<std::byte>& destination) {
  if (!validate_identity(result.identity).ok() || result.request_id == 0U ||
      result.invocation_id == 0U || !nonzero(result.request_sha256) ||
      result.evidence.weight_transport_bytes != 0U)
    return {ErrorCode::invalid_argument,
            "active-expert response identity or evidence is invalid"};
  if (result.status.ok() &&
      (!result.output.valid() || result.output.memory_domain != "host"))
    return {ErrorCode::invalid_argument,
            "active-expert wire response is not host accessible"};
  if (result.status.message().size() > kMaximumStringBytes ||
      (result.status.ok() && result.output.abi.size() > kMaximumStringBytes))
    return {ErrorCode::invalid_argument,
            "active-expert response metadata is too large"};
  destination = encode_response_once(result);
  result.evidence.wire_response_bytes = destination.size();
  destination = encode_response_once(result);
  if (destination.size() > kMaximumFrameBytes)
    return {ErrorCode::invalid_argument,
            "active-expert response frame is too large"};
  return Status::success();
}

struct DecodeResponseResult final {
  Status status;
  ActiveExpertExecutionResult result;
};

DecodeResponseResult decode_response(std::span<const std::byte> frame) {
  try {
    if (frame.size() <= kWireMagic.size() + 4U + 32U ||
        frame.size() > kMaximumFrameBytes)
      return {{ErrorCode::invalid_argument,
               "active-expert response frame size is invalid"}, {}};
    const auto payload = frame.first(frame.size() - 32U);
    Sha256Digest claimed{};
    std::copy(frame.end() - 32U, frame.end(), claimed.begin());
    if (!constant_time_equal(sha256(payload), claimed))
      return {{ErrorCode::checksum_mismatch,
               "active-expert response checksum mismatch"}, {}};
    Reader reader(payload);
    ActiveExpertExecutionResult result;
    std::span<const std::byte> request_digest;
    std::uint8_t code{};
    std::string message;
    std::string output_abi;
    std::uint64_t output_bytes{};
    if (!header(reader, kResponseFrame) || !reader.u64(result.request_id) ||
        !reader.u64(result.invocation_id) ||
        !reader.u32(result.selection_index) ||
        !identity(reader, result.identity) ||
        !reader.raw(result.request_sha256.size(), request_digest) ||
        !reader.u8(code) || code > static_cast<std::uint8_t>(ErrorCode::internal) ||
        !reader.text(message) || !reader.text(output_abi) ||
        !reader.u64(output_bytes) || !evidence(reader, result.evidence))
      return {{ErrorCode::invalid_argument,
               "active-expert response frame is malformed"}, {}};
    std::copy(request_digest.begin(), request_digest.end(),
              result.request_sha256.begin());
    result.status = {static_cast<ErrorCode>(code), std::move(message)};
    if (result.evidence.wire_response_bytes != frame.size() ||
        result.evidence.weight_transport_bytes != 0U)
      return {{ErrorCode::invalid_argument,
               "active-expert response transfer evidence is invalid"}, {}};
    if (result.status.ok()) {
      std::span<const std::byte> output;
      if (output_abi.empty() || output_bytes == 0U ||
          output_bytes > reader.remaining() ||
          !reader.raw(static_cast<std::size_t>(output_bytes), output))
        return {{ErrorCode::invalid_argument,
                 "active-expert response output is malformed"}, {}};
      auto owner = std::make_shared<std::vector<std::byte>>(output.begin(),
                                                            output.end());
      result.output = {std::move(output_abi), "host.wire", owner,
                       owner->data(), owner->size()};
    } else if (!output_abi.empty() || output_bytes != 0U) {
      return {{ErrorCode::invalid_argument,
               "failed active-expert response contains an output"}, {}};
    }
    if (reader.remaining() != 0U || !validate_identity(result.identity).ok() ||
        result.request_id == 0U || result.invocation_id == 0U ||
        !nonzero(result.request_sha256))
      return {{ErrorCode::invalid_argument,
               "active-expert response correlation is invalid"}, {}};
    return {Status::success(), std::move(result)};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("active-expert response decode failed: ") +
                 error.what()},
            {}};
  }
}

ActiveExpertExecutionResult failed_result(
    const ActiveExpertExecutionRequest& request, const Sha256Digest& digest,
    Status status) {
  ActiveExpertExecutionResult result;
  result.status = std::move(status);
  result.identity = request.identity;
  result.request_id = request.invocation.request_id;
  result.invocation_id = request.invocation.invocation_id;
  result.selection_index = request.invocation.selection_index;
  result.request_sha256 = digest;
  result.evidence.activation_input_bytes = request.invocation.input.bytes;
  return result;
}

ActiveExpertExecutionHandle completed_execution(
    ActiveExpertExecutionResult result) {
  struct State final {
    ActiveExpertExecutionResult result;
    bool terminal{};
  };
  auto state = std::make_shared<State>();
  state->result = std::move(result);
  return ActiveExpertExecutionHandle::from_callbacks(
      [state]() -> std::optional<ActiveExpertExecutionResult> {
        if (state->terminal) return std::nullopt;
        state->terminal = true;
        return std::move(state->result);
      },
      [state] { state->terminal = true; });
}

}  // namespace

struct ActiveExpertExecutionHandle::Core final {
  Poll poll;
  Cancel cancel;
  bool terminal{};
};

ActiveExpertExecutionHandle::ActiveExpertExecutionHandle() = default;
ActiveExpertExecutionHandle::ActiveExpertExecutionHandle(
    std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
ActiveExpertExecutionHandle::ActiveExpertExecutionHandle(
    ActiveExpertExecutionHandle&&) noexcept = default;
ActiveExpertExecutionHandle& ActiveExpertExecutionHandle::operator=(
    ActiveExpertExecutionHandle&&) noexcept = default;
ActiveExpertExecutionHandle::~ActiveExpertExecutionHandle() { cancel(); }

bool ActiveExpertExecutionHandle::valid() const noexcept {
  return core_ && !core_->terminal && static_cast<bool>(core_->poll);
}

std::optional<ActiveExpertExecutionResult> ActiveExpertExecutionHandle::poll() {
  if (!valid()) return std::nullopt;
  auto result = core_->poll();
  if (result) core_->terminal = true;
  return result;
}

void ActiveExpertExecutionHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  if (core_->cancel) core_->cancel();
  core_->terminal = true;
}

ActiveExpertExecutionHandle ActiveExpertExecutionHandle::from_callbacks(
    Poll poll, Cancel cancel) {
  if (!poll || !cancel) return {};
  auto core = std::make_unique<Core>();
  core->poll = std::move(poll);
  core->cancel = std::move(cancel);
  return ActiveExpertExecutionHandle(std::move(core));
}

Status ActiveExpertExecutorRegistry::add(
    ActiveExpertExecutorDefinition definition) noexcept {
  try {
    if (definition.name.empty() || definition.capability.empty() ||
        definition.minimum_abi == 0U ||
        definition.maximum_abi < definition.minimum_abi ||
        !definition.implementation || definition.implementation->remote())
      return {ErrorCode::invalid_argument,
              "active-expert executor definition is incomplete"};
    if (std::any_of(definitions_.begin(), definitions_.end(),
                    [&](const auto& item) {
                      return item.name == definition.name;
                    }))
      return {ErrorCode::invalid_argument,
              "active-expert executor name is duplicated"};
    definitions_.push_back(std::move(definition));
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("active-expert executor registration failed: ") +
                error.what()};
  }
}

SelectActiveExpertExecutorResult ActiveExpertExecutorRegistry::select(
    const ActiveExpertIdentity& identity_value) const noexcept {
  try {
    const auto valid = validate_identity(identity_value);
    if (!valid.ok()) return {copy_status(valid), {}};
    const ActiveExpertExecutorDefinition* selected = nullptr;
    for (const auto& candidate : definitions_) {
      if (candidate.capability != identity_value.capability ||
          identity_value.execution_abi < candidate.minimum_abi ||
          identity_value.execution_abi > candidate.maximum_abi)
        continue;
      if (candidate.validate && !candidate.validate(identity_value).ok())
        continue;
      if (selected == nullptr || candidate.priority > selected->priority ||
          (candidate.priority == selected->priority &&
           candidate.name < selected->name))
        selected = &candidate;
    }
    if (selected == nullptr)
      return {{ErrorCode::invalid_argument,
               "no active-expert executor implements the requested ABI"},
              {}};
    return {Status::success(), selected->implementation};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("active-expert executor selection failed: ") +
                 error.what()},
            {}};
  }
}

struct ActiveExpertWireHandle::Core final {
  Poll poll;
  Cancel cancel;
  bool terminal{};
};

ActiveExpertWireHandle::ActiveExpertWireHandle() = default;
ActiveExpertWireHandle::ActiveExpertWireHandle(
    std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
ActiveExpertWireHandle::ActiveExpertWireHandle(
    ActiveExpertWireHandle&&) noexcept = default;
ActiveExpertWireHandle& ActiveExpertWireHandle::operator=(
    ActiveExpertWireHandle&&) noexcept = default;
ActiveExpertWireHandle::~ActiveExpertWireHandle() { cancel(); }

bool ActiveExpertWireHandle::valid() const noexcept {
  return core_ && !core_->terminal && static_cast<bool>(core_->poll);
}

std::optional<ActiveExpertWireResult> ActiveExpertWireHandle::poll() {
  if (!valid()) return std::nullopt;
  auto result = core_->poll();
  if (result) core_->terminal = true;
  return result;
}

void ActiveExpertWireHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  if (core_->cancel) core_->cancel();
  core_->terminal = true;
}

ActiveExpertWireHandle ActiveExpertWireHandle::from_callbacks(
    Poll poll, Cancel cancel) {
  if (!poll || !cancel) return {};
  auto core = std::make_unique<Core>();
  core->poll = std::move(poll);
  core->cancel = std::move(cancel);
  return ActiveExpertWireHandle(std::move(core));
}

EncodeActiveExpertRequestResult encode_active_expert_request(
    const ActiveExpertExecutionRequest& request) noexcept {
  try {
    const auto identity_status = validate_identity(request.identity);
    if (!identity_status.ok()) return {copy_status(identity_status), {}, {}};
    const auto invocation_status = validate_invocation(request.invocation);
    if (!invocation_status.ok())
      return {copy_status(invocation_status), {}, {}};
    if (request.invocation.input.memory_domain != "host" &&
        request.invocation.input.memory_domain != "host.pinned" &&
        request.invocation.input.memory_domain != "host.wire")
      return {{ErrorCode::invalid_argument,
               "remote active-expert input is not host accessible"},
              {}, {}};
    if (request.identity.capability.size() > kMaximumStringBytes ||
        request.invocation.input.abi.size() > kMaximumStringBytes ||
        request.invocation.output_abi.size() > kMaximumStringBytes ||
        request.invocation.input.bytes > kMaximumFrameBytes ||
        request.invocation.output_bytes > kMaximumFrameBytes)
      return {{ErrorCode::invalid_argument,
               "active-expert request metadata or activation is too large"},
              {}, {}};
    const auto budget = deadline_budget(request.invocation.deadline);
    if (budget == 0U)
      return {{ErrorCode::deadline_exceeded,
               "active-expert request deadline expired"},
              {}, {}};
    Writer writer;
    header(writer, kRequestFrame);
    writer.u64(request.invocation.request_id);
    writer.u64(request.invocation.invocation_id);
    writer.u32(request.invocation.selection_index);
    writer.u32(request.invocation.route_width);
    writer.u64(budget);
    identity(writer, request.identity);
    writer.text(request.invocation.input.abi);
    writer.text(request.invocation.output_abi);
    writer.u64(request.invocation.output_bytes);
    writer.u64(request.invocation.input.bytes);
    writer.raw({request.invocation.input.data,
                static_cast<std::size_t>(request.invocation.input.bytes)});
    const auto digest = sha256(writer.view());
    writer.raw(digest);
    auto frame = writer.finish();
    if (frame.size() > kMaximumFrameBytes)
      return {{ErrorCode::invalid_argument,
               "active-expert request frame is too large"},
              {}, {}};
    return {Status::success(), std::move(frame), digest};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("active-expert request encode failed: ") +
                 error.what()},
            {}, {}};
  }
}

DecodeActiveExpertRequestResult decode_active_expert_request(
    std::span<const std::byte> frame) noexcept {
  try {
    if (frame.size() <= kWireMagic.size() + 4U + 32U ||
        frame.size() > kMaximumFrameBytes)
      return {{ErrorCode::invalid_argument,
               "active-expert request frame size is invalid"}, {}, {}};
    const auto payload = frame.first(frame.size() - 32U);
    Sha256Digest claimed{};
    std::copy(frame.end() - 32U, frame.end(), claimed.begin());
    if (!constant_time_equal(sha256(payload), claimed))
      return {{ErrorCode::checksum_mismatch,
               "active-expert request checksum mismatch"}, {}, {}};
    Reader reader(payload);
    ActiveExpertExecutionRequest request;
    std::uint64_t budget{};
    std::string input_abi;
    std::uint64_t input_bytes{};
    if (!header(reader, kRequestFrame) ||
        !reader.u64(request.invocation.request_id) ||
        !reader.u64(request.invocation.invocation_id) ||
        !reader.u32(request.invocation.selection_index) ||
        !reader.u32(request.invocation.route_width) || reader.remaining() == 0U ||
        !reader.u64(budget) || !identity(reader, request.identity) ||
        !reader.text(input_abi) ||
        !reader.text(request.invocation.output_abi) ||
        !reader.u64(request.invocation.output_bytes) ||
        !reader.u64(input_bytes) || input_bytes == 0U ||
        input_bytes > reader.remaining())
      return {{ErrorCode::invalid_argument,
               "active-expert request frame is malformed"}, {}, {}};
    std::span<const std::byte> input;
    if (!reader.raw(static_cast<std::size_t>(input_bytes), input) ||
        reader.remaining() != 0U || budget == 0U)
      return {{budget == 0U ? ErrorCode::deadline_exceeded
                            : ErrorCode::invalid_argument,
               budget == 0U ? "active-expert request deadline expired"
                             : "active-expert request has trailing data"},
              {}, {}};
    auto owner = std::make_shared<std::vector<std::byte>>(input.begin(),
                                                          input.end());
    request.invocation.deadline = decode_deadline(budget);
    request.invocation.input = {std::move(input_abi), "host.wire", owner,
                                owner->data(), owner->size()};
    const auto identity_status = validate_identity(request.identity);
    const auto invocation_status = validate_invocation(request.invocation);
    if (!identity_status.ok()) return {copy_status(identity_status), {}, {}};
    if (!invocation_status.ok())
      return {copy_status(invocation_status), {}, {}};
    return {Status::success(), std::move(request), claimed};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("active-expert request decode failed: ") +
                 error.what()},
            {}, {}};
  }
}

struct ActiveExpertWireEndpoint::Core final {
  explicit Core(ActiveExpertExecutorRegistry value)
      : registry(std::move(value)) {}
  ActiveExpertExecutorRegistry registry;
  AtomicTelemetry telemetry;
};

ActiveExpertWireEndpoint::ActiveExpertWireEndpoint(
    ActiveExpertExecutorRegistry registry)
    : core_(std::make_shared<Core>(std::move(registry))) {}
ActiveExpertWireEndpoint::ActiveExpertWireEndpoint(
    ActiveExpertWireEndpoint&&) noexcept = default;
ActiveExpertWireEndpoint& ActiveExpertWireEndpoint::operator=(
    ActiveExpertWireEndpoint&&) noexcept = default;
ActiveExpertWireEndpoint::~ActiveExpertWireEndpoint() = default;

ActiveExpertWireHandle ActiveExpertWireEndpoint::submit(
    std::vector<std::byte> request_frame) {
  if (!core_) return {};
  auto decoded = decode_active_expert_request(request_frame);
  if (!decoded.status.ok()) {
    if (decoded.status.code() == ErrorCode::checksum_mismatch)
      core_->telemetry.checksum_failures.fetch_add(1U,
                                                   std::memory_order_relaxed);
    struct FailureState final {
      Status status;
      bool terminal{};
    };
    auto failed = std::make_shared<FailureState>();
    failed->status = copy_status(decoded.status);
    return ActiveExpertWireHandle::from_callbacks(
        [failed]() -> std::optional<ActiveExpertWireResult> {
          if (failed->terminal) return std::nullopt;
          failed->terminal = true;
          return ActiveExpertWireResult{std::move(failed->status), {}};
        },
        [failed] { failed->terminal = true; });
  }
  auto selected = core_->registry.select(decoded.request.identity);
  if (!selected.status.ok()) {
    auto result = failed_result(decoded.request, decoded.request_sha256,
                                copy_status(selected.status));
    result.evidence.wire_request_bytes = request_frame.size();
    std::vector<std::byte> response;
    const auto encoded = encode_response(result, response);
    struct CompletedState final {
      ActiveExpertWireResult result;
      bool terminal{};
    };
    auto completed = std::make_shared<CompletedState>();
    completed->result = {copy_status(encoded), std::move(response)};
    core_->telemetry.requests.fetch_add(1U, std::memory_order_relaxed);
    core_->telemetry.failed.fetch_add(1U, std::memory_order_relaxed);
    core_->telemetry.wire_request_bytes.fetch_add(
        request_frame.size(), std::memory_order_relaxed);
    core_->telemetry.wire_response_bytes.fetch_add(
        completed->result.frame.size(), std::memory_order_relaxed);
    return ActiveExpertWireHandle::from_callbacks(
        [completed]() -> std::optional<ActiveExpertWireResult> {
          if (completed->terminal) return std::nullopt;
          completed->terminal = true;
          return std::move(completed->result);
        },
        [completed] { completed->terminal = true; });
  }

  struct State final {
    std::shared_ptr<Core> core;
    ActiveExpertExecutionRequest request;
    Sha256Digest request_sha256{};
    std::uint64_t request_wire_bytes{};
    ActiveExpertExecutionHandle execution;
    std::chrono::steady_clock::time_point started;
    bool terminal{};
  };
  auto state = std::make_shared<State>();
  state->core = core_;
  state->request = decoded.request;
  state->request_sha256 = decoded.request_sha256;
  state->request_wire_bytes = request_frame.size();
  state->started = std::chrono::steady_clock::now();
  state->execution = selected.executor->execute(std::move(decoded.request));
  core_->telemetry.requests.fetch_add(1U, std::memory_order_relaxed);
  core_->telemetry.activation_input_bytes.fetch_add(
      state->request.invocation.input.bytes, std::memory_order_relaxed);
  core_->telemetry.wire_request_bytes.fetch_add(
      state->request_wire_bytes, std::memory_order_relaxed);

  return ActiveExpertWireHandle::from_callbacks(
      [state]() -> std::optional<ActiveExpertWireResult> {
        if (state->terminal) return std::nullopt;
        if (state->request.invocation.deadline !=
                std::chrono::steady_clock::time_point::max() &&
            state->request.invocation.deadline <=
                std::chrono::steady_clock::now()) {
          state->execution.cancel();
          state->terminal = true;
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
          state->core->telemetry.deadline_exceeded.fetch_add(
              1U, std::memory_order_relaxed);
          auto result = failed_result(
              state->request, state->request_sha256,
              {ErrorCode::deadline_exceeded,
               "active-expert owner execution deadline expired"});
          result.evidence.wire_request_bytes = state->request_wire_bytes;
          std::vector<std::byte> response;
          auto encoded = encode_response(result, response);
          if (encoded.ok())
            state->core->telemetry.wire_response_bytes.fetch_add(
                response.size(), std::memory_order_relaxed);
          return ActiveExpertWireResult{std::move(encoded),
                                        std::move(response)};
        }
        if (!state->execution.valid()) {
          state->terminal = true;
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
          auto result = failed_result(
              state->request, state->request_sha256,
              {ErrorCode::internal,
               "active-expert owner returned no execution handle"});
          result.evidence.wire_request_bytes = state->request_wire_bytes;
          std::vector<std::byte> response;
          auto encoded = encode_response(result, response);
          return ActiveExpertWireResult{std::move(encoded),
                                        std::move(response)};
        }
        auto completed = state->execution.poll();
        if (!completed) return std::nullopt;
        state->terminal = true;
        const bool correlated =
            completed->identity == state->request.identity &&
            completed->request_id == state->request.invocation.request_id &&
            completed->invocation_id ==
                state->request.invocation.invocation_id &&
            completed->selection_index ==
                state->request.invocation.selection_index;
        if (!correlated) {
          *completed = failed_result(
              state->request, state->request_sha256,
              {ErrorCode::internal,
               "active-expert owner result correlation mismatch"});
        } else {
          completed->request_sha256 = state->request_sha256;
        }
        completed->evidence.activation_input_bytes =
            state->request.invocation.input.bytes;
        completed->evidence.wire_request_bytes = state->request_wire_bytes;
        if (completed->evidence.owner_execution_ns == 0U)
          completed->evidence.owner_execution_ns =
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - state->started)
                      .count());
        if (completed->status.ok()) {
          if (!completed->output.valid() ||
              completed->output.abi != state->request.invocation.output_abi ||
              completed->output.bytes !=
                  state->request.invocation.output_bytes ||
              completed->output.memory_domain != "host" ||
              completed->evidence.weight_transport_bytes != 0U) {
            *completed = failed_result(
                state->request, state->request_sha256,
                {ErrorCode::internal,
                 "active-expert owner violated the output/weight contract"});
            completed->evidence.wire_request_bytes =
                state->request_wire_bytes;
          } else {
            completed->evidence.activation_output_bytes =
                completed->output.bytes;
          }
        }
        std::vector<std::byte> response;
        auto encoded = encode_response(*completed, response);
        if (!encoded.ok()) {
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
          return ActiveExpertWireResult{std::move(encoded), {}};
        }
        state->core->telemetry.add_completion(completed->evidence);
        if (completed->status.ok()) {
          state->core->telemetry.completed.fetch_add(
              1U, std::memory_order_relaxed);
        } else {
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
        }
        return ActiveExpertWireResult{Status::success(),
                                      std::move(response)};
      },
      [state] {
        if (state->terminal) return;
        state->execution.cancel();
        state->terminal = true;
        state->core->telemetry.cancelled.fetch_add(
            1U, std::memory_order_relaxed);
      });
}

ActiveExpertExecutorTelemetry ActiveExpertWireEndpoint::telemetry()
    const noexcept {
  return core_ ? core_->telemetry.snapshot()
               : ActiveExpertExecutorTelemetry{};
}

struct RemoteActiveExpertExecutor::Core final {
  Core(std::string owner_value,
       std::shared_ptr<IActiveExpertTransport> transport_value)
      : owner(std::move(owner_value)), transport(std::move(transport_value)) {}
  std::string owner;
  std::shared_ptr<IActiveExpertTransport> transport;
  AtomicTelemetry telemetry;
};

RemoteActiveExpertExecutor::RemoteActiveExpertExecutor(
    std::string owner_value,
    std::shared_ptr<IActiveExpertTransport> transport)
    : core_(std::make_shared<Core>(std::move(owner_value),
                                   std::move(transport))) {}

std::string_view RemoteActiveExpertExecutor::owner() const noexcept {
  return core_ ? std::string_view(core_->owner) : std::string_view{};
}

ActiveExpertExecutionHandle RemoteActiveExpertExecutor::execute(
    ActiveExpertExecutionRequest request) {
  if (!core_ || core_->owner.empty() || !core_->transport)
    return {};
  core_->telemetry.requests.fetch_add(1U, std::memory_order_relaxed);
  auto encoded = encode_active_expert_request(request);
  if (!encoded.status.ok()) {
    core_->telemetry.failed.fetch_add(1U, std::memory_order_relaxed);
    if (encoded.status.code() == ErrorCode::deadline_exceeded)
      core_->telemetry.deadline_exceeded.fetch_add(
          1U, std::memory_order_relaxed);
    return completed_execution(failed_result(
        request, {}, copy_status(encoded.status)));
  }
  const auto wire_bytes = encoded.frame.size();
  auto transport = core_->transport->submit(
      core_->owner, std::move(encoded.frame), request.invocation.deadline);
  if (!transport.valid()) {
    core_->telemetry.failed.fetch_add(1U, std::memory_order_relaxed);
    return completed_execution(failed_result(
        request, encoded.request_sha256,
        {ErrorCode::backpressure,
         "active-expert transport rejected the invocation"}));
  }
  core_->telemetry.activation_input_bytes.fetch_add(
      request.invocation.input.bytes, std::memory_order_relaxed);
  core_->telemetry.wire_request_bytes.fetch_add(wire_bytes,
                                                std::memory_order_relaxed);
  struct State final {
    std::shared_ptr<Core> core;
    ActiveExpertExecutionRequest request;
    Sha256Digest request_sha256{};
    std::uint64_t wire_request_bytes{};
    ActiveExpertWireHandle transport;
    std::chrono::steady_clock::time_point started;
    bool terminal{};
  };
  auto state = std::make_shared<State>();
  state->core = core_;
  state->request = std::move(request);
  state->request_sha256 = encoded.request_sha256;
  state->wire_request_bytes = wire_bytes;
  state->transport = std::move(transport);
  state->started = std::chrono::steady_clock::now();
  return ActiveExpertExecutionHandle::from_callbacks(
      [state]() -> std::optional<ActiveExpertExecutionResult> {
        if (state->terminal) return std::nullopt;
        if (state->request.invocation.deadline !=
                std::chrono::steady_clock::time_point::max() &&
            state->request.invocation.deadline <=
                std::chrono::steady_clock::now()) {
          state->transport.cancel();
          state->terminal = true;
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
          state->core->telemetry.deadline_exceeded.fetch_add(
              1U, std::memory_order_relaxed);
          return failed_result(
              state->request, state->request_sha256,
              {ErrorCode::deadline_exceeded,
               "remote active-expert deadline expired"});
        }
        auto wire = state->transport.poll();
        if (!wire) return std::nullopt;
        state->terminal = true;
        if (!wire->status.ok()) {
          if (wire->status.code() == ErrorCode::checksum_mismatch)
            state->core->telemetry.checksum_failures.fetch_add(
                1U, std::memory_order_relaxed);
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
          return failed_result(state->request, state->request_sha256,
                               copy_status(wire->status));
        }
        auto decoded = decode_response(wire->frame);
        if (!decoded.status.ok()) {
          if (decoded.status.code() == ErrorCode::checksum_mismatch)
            state->core->telemetry.checksum_failures.fetch_add(
                1U, std::memory_order_relaxed);
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
          return failed_result(state->request, state->request_sha256,
                               copy_status(decoded.status));
        }
        auto& result = decoded.result;
        if (result.identity != state->request.identity ||
            result.request_id != state->request.invocation.request_id ||
            result.invocation_id !=
                state->request.invocation.invocation_id ||
            result.selection_index !=
                state->request.invocation.selection_index ||
            !constant_time_equal(result.request_sha256,
                                 state->request_sha256) ||
            result.evidence.weight_transport_bytes != 0U ||
            (result.status.ok() &&
             (result.output.abi != state->request.invocation.output_abi ||
              result.output.bytes !=
                  state->request.invocation.output_bytes))) {
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
          return failed_result(
              state->request, state->request_sha256,
              {ErrorCode::checksum_mismatch,
               "remote active-expert response correlation mismatch"});
        }
        result.evidence.wire_request_bytes = state->wire_request_bytes;
        result.evidence.wire_response_bytes = wire->frame.size();
        result.evidence.transport_wait_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - state->started)
                .count());
        state->core->telemetry.add_completion(result.evidence);
        if (result.status.ok()) {
          state->core->telemetry.completed.fetch_add(
              1U, std::memory_order_relaxed);
        } else {
          state->core->telemetry.failed.fetch_add(1U,
                                                  std::memory_order_relaxed);
        }
        return std::move(result);
      },
      [state] {
        if (state->terminal) return;
        state->transport.cancel();
        state->terminal = true;
        state->core->telemetry.cancelled.fetch_add(
            1U, std::memory_order_relaxed);
      });
}

ActiveExpertExecutorTelemetry RemoteActiveExpertExecutor::telemetry()
    const noexcept {
  return core_ ? core_->telemetry.snapshot()
               : ActiveExpertExecutorTelemetry{};
}

}  // namespace expert::runtime
