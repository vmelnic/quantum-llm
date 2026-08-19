#pragma once

#include "expert/runtime/expert_key.hpp"
#include "expert/runtime/storage.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace expert::runtime {

// Immutable activation or expert output owned by the producer. Weight payloads
// deliberately have no representation in this contract: local executors retain
// their weights and remote execution moves only activations and results.
struct ActiveExpertBuffer final {
  std::string abi;
  std::string memory_domain;
  std::shared_ptr<const void> owner;
  const std::byte* data{};
  std::uint64_t bytes{};

  [[nodiscard]] bool valid() const noexcept {
    return !abi.empty() && !memory_domain.empty() && owner && data != nullptr &&
           bytes != 0U;
  }
};

// Immutable logical page plus the numeric operation implemented by its owner.
// The model hash prevents two artifacts with coincidentally equal geometry
// from sharing an execution lease.
struct ActiveExpertIdentity final {
  Sha256Digest model_content_hash{};
  ExpertKey key;
  std::string capability;
  std::uint32_t execution_abi{};
  std::uint32_t source_abi{};

  friend bool operator==(const ActiveExpertIdentity&,
                         const ActiveExpertIdentity&) = default;
};

// Request-local fields supplied after exact routing. selection_index preserves
// the stable top-k slot; route weights remain at the caller and are applied by
// the ordinary exact aggregator after every selected output has arrived.
struct ActiveExpertInvocation final {
  std::uint64_t request_id{};
  std::uint64_t invocation_id{};
  std::uint32_t selection_index{};
  std::uint32_t route_width{};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::time_point::max()};
  ActiveExpertBuffer input;
  std::string output_abi;
  std::uint64_t output_bytes{};
};

struct ActiveExpertExecutionRequest final {
  ActiveExpertIdentity identity;
  ActiveExpertInvocation invocation;
};

// Per-invocation evidence is carried with the result, making attribution
// independent of global counter sampling. owner_* bytes describe work at the
// execution owner. weight_transport_bytes must always remain zero.
struct ActiveExpertExecutionEvidence final {
  std::uint64_t activation_input_bytes{};
  std::uint64_t activation_output_bytes{};
  std::uint64_t wire_request_bytes{};
  std::uint64_t wire_response_bytes{};
  std::uint64_t weight_transport_bytes{};
  std::uint64_t owner_weight_read_bytes{};
  std::uint64_t owner_storage_read_bytes{};
  std::uint64_t owner_ram_read_bytes{};
  std::uint64_t owner_vram_read_bytes{};
  std::uint64_t owner_execution_ns{};
  std::uint64_t transport_wait_ns{};
};

struct ActiveExpertExecutionResult final {
  Status status;
  ActiveExpertIdentity identity;
  std::uint64_t request_id{};
  std::uint64_t invocation_id{};
  std::uint32_t selection_index{};
  Sha256Digest request_sha256{};
  ActiveExpertBuffer output;
  ActiveExpertExecutionEvidence evidence;
};

class ActiveExpertExecutionHandle final {
 public:
  using Poll =
      std::function<std::optional<ActiveExpertExecutionResult>()>;
  using Cancel = std::function<void()>;

  ActiveExpertExecutionHandle();
  ActiveExpertExecutionHandle(const ActiveExpertExecutionHandle&) = delete;
  ActiveExpertExecutionHandle& operator=(const ActiveExpertExecutionHandle&) =
      delete;
  ActiveExpertExecutionHandle(ActiveExpertExecutionHandle&&) noexcept;
  ActiveExpertExecutionHandle& operator=(
      ActiveExpertExecutionHandle&&) noexcept;
  ~ActiveExpertExecutionHandle();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::optional<ActiveExpertExecutionResult> poll();
  void cancel() noexcept;
  [[nodiscard]] static ActiveExpertExecutionHandle from_callbacks(
      Poll poll, Cancel cancel);

 private:
  struct Core;
  explicit ActiveExpertExecutionHandle(std::unique_ptr<Core> core) noexcept;
  std::unique_ptr<Core> core_;
};

struct ActiveExpertExecutorTelemetry final {
  std::uint64_t requests{};
  std::uint64_t completed{};
  std::uint64_t failed{};
  std::uint64_t cancelled{};
  std::uint64_t deadline_exceeded{};
  std::uint64_t checksum_failures{};
  std::uint64_t activation_input_bytes{};
  std::uint64_t activation_output_bytes{};
  std::uint64_t wire_request_bytes{};
  std::uint64_t wire_response_bytes{};
  std::uint64_t weight_transport_bytes{};
  std::uint64_t owner_weight_read_bytes{};
  std::uint64_t owner_storage_read_bytes{};
  std::uint64_t owner_ram_read_bytes{};
  std::uint64_t owner_vram_read_bytes{};
  std::uint64_t owner_execution_ns{};
  std::uint64_t transport_wait_ns{};
};

// The same provider interface is registered locally at an expert owner and is
// wrapped by RemoteActiveExpertExecutor at a caller. Selection is by opaque
// capability/ABI, never by model family.
class IActiveExpertExecutor {
 public:
  virtual ~IActiveExpertExecutor() = default;
  [[nodiscard]] virtual std::string_view owner() const noexcept = 0;
  [[nodiscard]] virtual bool remote() const noexcept = 0;
  [[nodiscard]] virtual ActiveExpertExecutionHandle execute(
      ActiveExpertExecutionRequest request) = 0;
  [[nodiscard]] virtual ActiveExpertExecutorTelemetry telemetry()
      const noexcept = 0;
};

struct ActiveExpertExecutorDefinition final {
  std::string name;
  std::string capability;
  std::uint32_t minimum_abi{1U};
  std::uint32_t maximum_abi{1U};
  std::uint32_t priority{};
  std::function<Status(const ActiveExpertIdentity&)> validate;
  std::shared_ptr<IActiveExpertExecutor> implementation;
};

struct SelectActiveExpertExecutorResult final {
  Status status;
  std::shared_ptr<IActiveExpertExecutor> executor;
};

class ActiveExpertExecutorRegistry final {
 public:
  [[nodiscard]] Status add(
      ActiveExpertExecutorDefinition definition) noexcept;
  [[nodiscard]] SelectActiveExpertExecutorResult select(
      const ActiveExpertIdentity& identity) const noexcept;

 private:
  std::vector<ActiveExpertExecutorDefinition> definitions_;
};

struct ActiveExpertWireResult final {
  Status status;
  std::vector<std::byte> frame;
};

class ActiveExpertWireHandle final {
 public:
  using Poll = std::function<std::optional<ActiveExpertWireResult>()>;
  using Cancel = std::function<void()>;

  ActiveExpertWireHandle();
  ActiveExpertWireHandle(const ActiveExpertWireHandle&) = delete;
  ActiveExpertWireHandle& operator=(const ActiveExpertWireHandle&) = delete;
  ActiveExpertWireHandle(ActiveExpertWireHandle&&) noexcept;
  ActiveExpertWireHandle& operator=(ActiveExpertWireHandle&&) noexcept;
  ~ActiveExpertWireHandle();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::optional<ActiveExpertWireResult> poll();
  void cancel() noexcept;
  [[nodiscard]] static ActiveExpertWireHandle from_callbacks(
      Poll poll, Cancel cancel);

 private:
  struct Core;
  explicit ActiveExpertWireHandle(std::unique_ptr<Core> core) noexcept;
  std::unique_ptr<Core> core_;
};

class IActiveExpertTransport {
 public:
  virtual ~IActiveExpertTransport() = default;
  // The frame contains authenticated metadata plus activation bytes only.
  [[nodiscard]] virtual ActiveExpertWireHandle submit(
      std::string_view owner, std::vector<std::byte> frame,
      std::chrono::steady_clock::time_point deadline) = 0;
};

struct EncodeActiveExpertRequestResult final {
  Status status;
  std::vector<std::byte> frame;
  Sha256Digest request_sha256{};
};

struct DecodeActiveExpertRequestResult final {
  Status status;
  ActiveExpertExecutionRequest request;
  Sha256Digest request_sha256{};
};

[[nodiscard]] EncodeActiveExpertRequestResult encode_active_expert_request(
    const ActiveExpertExecutionRequest& request) noexcept;
[[nodiscard]] DecodeActiveExpertRequestResult decode_active_expert_request(
    std::span<const std::byte> frame) noexcept;

// Server-side wire endpoint. It binds an ordinary local executor from the
// registry, retains the asynchronous handle, and serializes an exact correlated
// response. A concrete network/RDMA implementation only has to carry frames.
class ActiveExpertWireEndpoint final {
 public:
  explicit ActiveExpertWireEndpoint(ActiveExpertExecutorRegistry registry);
  ActiveExpertWireEndpoint(const ActiveExpertWireEndpoint&) = delete;
  ActiveExpertWireEndpoint& operator=(const ActiveExpertWireEndpoint&) = delete;
  ActiveExpertWireEndpoint(ActiveExpertWireEndpoint&&) noexcept;
  ActiveExpertWireEndpoint& operator=(ActiveExpertWireEndpoint&&) noexcept;
  ~ActiveExpertWireEndpoint();

  [[nodiscard]] ActiveExpertWireHandle submit(
      std::vector<std::byte> request_frame);
  [[nodiscard]] ActiveExpertExecutorTelemetry telemetry() const noexcept;

 private:
  struct Core;
  std::shared_ptr<Core> core_;
};

// Client-side executor over an arbitrary frame transport. It verifies the
// response digest and complete request/selection/expert correlation before
// publishing the output to the provider.
class RemoteActiveExpertExecutor final : public IActiveExpertExecutor {
 public:
  RemoteActiveExpertExecutor(std::string owner,
                             std::shared_ptr<IActiveExpertTransport> transport);
  [[nodiscard]] std::string_view owner() const noexcept override;
  [[nodiscard]] bool remote() const noexcept override { return true; }
  [[nodiscard]] ActiveExpertExecutionHandle execute(
      ActiveExpertExecutionRequest request) override;
  [[nodiscard]] ActiveExpertExecutorTelemetry telemetry()
      const noexcept override;

 private:
  struct Core;
  std::shared_ptr<Core> core_;
};

}  // namespace expert::runtime
