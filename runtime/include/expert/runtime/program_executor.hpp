#pragma once

#include "expert/runtime/execution_provider.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace expert::runtime {

// Provider-neutral immutable value. The ABI and memory-domain strings are
// artifact/provider contracts, not enums owned by the common runtime. The
// owner keeps local, device, shared-memory, or transport-specific storage
// alive for as long as the value is reachable by an invocation.
struct ExecutionValue final {
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

// Authenticated immutable model tensor retained for a prepared operation.
// Providers receive this handle and must not reopen family-specific model
// files behind the VM's back.
struct ImmutableModelTensor final {
  std::string name;
  std::string encoding;
  std::uint32_t quant_abi{};
  std::vector<std::uint32_t> shape;
  std::uint64_t data_offset{};
  std::uint64_t data_bytes{};
  std::uint64_t scale_offset{};
  std::uint64_t scale_bytes{};
  ExecutionValue value;
};

struct ResolveModelTensorResult final {
  Status status;
  std::shared_ptr<const ImmutableModelTensor> tensor;
};

class IModelTensorStore {
 public:
  virtual ~IModelTensorStore() = default;
  [[nodiscard]] virtual ResolveModelTensorResult resolve(
      std::string_view name) = 0;
};

struct PreparedTensorBinding final {
  std::string role;
  std::shared_ptr<const ImmutableModelTensor> tensor;
};

class IPreparedOperation {
 public:
  virtual ~IPreparedOperation() = default;
};

class IOperationProviderRequestState {
 public:
  virtual ~IOperationProviderRequestState() = default;
};

struct OperationPreparationContext final {
  const ModelDescriptor& model;
  const OperationProgramDescriptor& operation;
  const CompiledOperationProgram& compiled;
  std::span<const PreparedTensorBinding> tensors;
};

struct PrepareOperationResult final {
  Status status;
  std::shared_ptr<const IPreparedOperation> operation;
};

struct ProgramRequestContext final {
  std::uint64_t request_id{};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::time_point::max()};
  // Service/request contract parameters such as reserved context length.
  // Names remain provider/service-ABI owned.
  std::map<std::string, std::uint64_t, std::less<>> parameters;
};

struct CreateOperationRequestStateResult final {
  Status status;
  // Null is a valid stateless provider result.
  std::shared_ptr<IOperationProviderRequestState> state;
};

struct OperationExecutionResult final {
  Status status;
  // Exact artifact output-port order. The interpreter rejects a partial set,
  // an extra value, or an ABI mismatch.
  std::vector<ExecutionValue> outputs;
};

class OperationExecutionHandle final {
 public:
  using Poll = std::function<std::optional<OperationExecutionResult>()>;
  using Cancel = std::function<void()>;

  OperationExecutionHandle();
  OperationExecutionHandle(const OperationExecutionHandle&) = delete;
  OperationExecutionHandle& operator=(const OperationExecutionHandle&) =
      delete;
  OperationExecutionHandle(OperationExecutionHandle&&) noexcept;
  OperationExecutionHandle& operator=(OperationExecutionHandle&&) noexcept;
  ~OperationExecutionHandle();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::optional<OperationExecutionResult> poll();
  void cancel() noexcept;

  [[nodiscard]] static OperationExecutionHandle from_callbacks(
      Poll poll, Cancel cancel);

 private:
  struct Core;
  explicit OperationExecutionHandle(std::unique_ptr<Core> core) noexcept;
  std::unique_ptr<Core> core_;
};

struct OperationInvocation final {
  ProgramRequestContext request;
  const CompiledOperationProgram& operation;
  std::span<const ExecutionValue> inputs;
};

// Optional whole-program prefill scheduling boundary. The compiled artifact
// program and prepared operations remain authoritative; providers may reorder
// evaluation only, never change declared math, tensor bindings, or output ABI.
struct ProgramSequenceInvocation final {
  ProgramRequestContext request;
  const CompiledModelProgram& program;
  std::span<const IPreparedOperation* const> operations;
  std::span<const ExecutionValue> inputs;
};

struct ExactDecodePreparationContext final {
  const ModelDescriptor& model;
  const ExactDecodeProgramDescriptor& program;
  const CompiledExactDecodeProgram& compiled;
  std::span<const PreparedTensorBinding> tensors;
};

struct ExactDecodeSynchronization final {
  ProgramRequestContext request;
  std::uint32_t next_token{};
  std::uint32_t target_position{};
  bool produce_draft{};
};

struct ExactDecodeSynchronizationBatch final {
  ProgramRequestContext request;
  std::span<const std::uint32_t> next_tokens;
  std::uint32_t first_target_position{};
  bool produce_final_draft{};
};

struct ExactDecodeInvocation final {
  ProgramRequestContext request;
  std::uint32_t guaranteed_token{};
  std::uint32_t position{};
  std::uint32_t context_limit{};
};

struct ExactDecodeExecutionResult final {
  Status status;
  // The first token is guaranteed_token. Any additional tokens have been
  // accepted by the exact target verifier; next_token remains unemitted.
  std::vector<std::uint32_t> emitted_tokens;
  std::uint32_t next_token{};
  std::uint32_t positions_advanced{};
};

class ExactDecodeExecutionHandle final {
 public:
  using Poll = std::function<std::optional<ExactDecodeExecutionResult>()>;
  using Cancel = std::function<void()>;

  ExactDecodeExecutionHandle();
  ExactDecodeExecutionHandle(const ExactDecodeExecutionHandle&) = delete;
  ExactDecodeExecutionHandle& operator=(const ExactDecodeExecutionHandle&) =
      delete;
  ExactDecodeExecutionHandle(ExactDecodeExecutionHandle&&) noexcept;
  ExactDecodeExecutionHandle& operator=(
      ExactDecodeExecutionHandle&&) noexcept;
  ~ExactDecodeExecutionHandle();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::optional<ExactDecodeExecutionResult> poll();
  void cancel() noexcept;
  [[nodiscard]] static ExactDecodeExecutionHandle from_callbacks(
      Poll poll, Cancel cancel);

 private:
  struct Core;
  explicit ExactDecodeExecutionHandle(std::unique_ptr<Core> core) noexcept;
  std::unique_ptr<Core> core_;
};

// One callable backend may implement any subset of operation capabilities.
// Preparation is startup-only; execute() is the hot-path boundary and may
// complete asynchronously. The common interpreter never switches on the
// capability name after binding.
class IOperationProvider {
 public:
  virtual ~IOperationProvider() = default;
  [[nodiscard]] virtual PrepareOperationResult prepare(
      const OperationPreparationContext& context) = 0;
  [[nodiscard]] virtual CreateOperationRequestStateResult create_request_state(
      const ProgramRequestContext& request) = 0;
  [[nodiscard]] virtual OperationExecutionHandle execute(
      const IPreparedOperation& operation,
      const std::shared_ptr<IOperationProviderRequestState>& request_state,
      const OperationInvocation& invocation) = 0;
  [[nodiscard]] virtual bool supports_program_sequence(
      const CompiledModelProgram&) const noexcept {
    return false;
  }
  [[nodiscard]] virtual OperationExecutionHandle execute_program_sequence(
      const std::shared_ptr<IOperationProviderRequestState>&,
      const ProgramSequenceInvocation&) {
    return {};
  }
  [[nodiscard]] virtual bool supports_request_state_retention()
      const noexcept {
    return false;
  }
  [[nodiscard]] virtual Status checkpoint_request_state(
      const std::shared_ptr<IOperationProviderRequestState>&,
      std::uint32_t) {
    return {ErrorCode::invalid_argument,
            "operation provider has no retention checkpoint implementation"};
  }
  [[nodiscard]] virtual Status rewind_request_state(
      const std::shared_ptr<IOperationProviderRequestState>&,
      std::uint32_t) {
    return {ErrorCode::invalid_argument,
            "operation provider has no retention rewind implementation"};
  }
  [[nodiscard]] virtual PrepareOperationResult prepare_exact_decode(
      const ExactDecodePreparationContext&) {
    return {{ErrorCode::invalid_argument,
             "operation provider has no exact decode implementation"},
            {}};
  }
  [[nodiscard]] virtual Status synchronize_exact_decode(
      const IPreparedOperation&,
      const std::shared_ptr<IOperationProviderRequestState>&,
      const ExactDecodeSynchronization&) {
    return {ErrorCode::invalid_argument,
            "operation provider has no exact decode implementation"};
  }
  [[nodiscard]] virtual Status synchronize_exact_decode_batch(
      const IPreparedOperation& operation,
      const std::shared_ptr<IOperationProviderRequestState>& state,
      const ExactDecodeSynchronizationBatch& synchronization) {
    if (synchronization.next_tokens.empty())
      return {ErrorCode::invalid_argument,
              "exact decode synchronization batch is empty"};
    for (std::size_t index = 0U;
         index < synchronization.next_tokens.size(); ++index) {
      auto status = synchronize_exact_decode(
          operation, state,
          {synchronization.request, synchronization.next_tokens[index],
           synchronization.first_target_position +
               static_cast<std::uint32_t>(index),
           synchronization.produce_final_draft &&
               index + 1U == synchronization.next_tokens.size()});
      if (!status.ok()) return status;
    }
    return Status::success();
  }
  [[nodiscard]] virtual ExactDecodeExecutionHandle execute_exact_decode(
      const IPreparedOperation&,
      const std::shared_ptr<IOperationProviderRequestState>&,
      const ExactDecodeInvocation&) {
    return {};
  }
};

// Loadable provider module boundary used by the common runner. A module may
// own an optimized tensor placement store, but execution is still negotiated
// per operation capability and never selected as a whole-model entry point.
struct ExecutionProviderModule final {
  struct ServiceContract final {
    std::string prefill_mode;
    std::uint32_t prefill_chunk_tokens{};
    // Must mirror IOperationProvider::supports_request_state_retention().
    // The common runner uses this capability to admit retention commands.
    bool session_retention{};
    std::string request_stream_mode;
    std::string rope_mode;
    std::string kv_dtype;
    std::string kv_allocation;
    std::uint32_t kv_page_tokens{};
    std::uint64_t kv_page_bytes{};
    std::uint64_t kv_page_capacity{};
    std::string placement_mode;
    std::string placement_profile;
    std::uint64_t ram_cache_bytes{};
    std::uint64_t vram_cache_bytes{};
    bool placement_prefetch_enabled{};
    std::string placement_prefetch_state;
    std::uint32_t placement_minimum_observations{};
    bool mtp_resource_available{};
    bool mtp_runtime_ready{};
    bool mtp_enabled{};
    bool retain_previous_route{};
    bool cpu_hybrid_enabled{};
    // True only when the selected provider implements request-scoped token
    // sampling for the artifact's token-selection capability.
    bool sampling_supported{};
  } service;
  ExecutionProviderDefinition definition;
  std::shared_ptr<IModelTensorStore> tensor_store;
  // Provider-owned counters are exposed as flat unsigned values so the common
  // service loop can publish request/runtime evidence without learning model
  // families or provider-specific telemetry structs.
  std::function<std::map<std::string, std::uint64_t, std::less<>>()> telemetry;
};

struct CreateExecutionProviderModuleResult final {
  Status status;
  ExecutionProviderModule module;
};

struct ProgramExecutionRequest final {
  ProgramRequestContext context;
  // External role -> value. The role set must match the schema v3 program
  // inputs exactly; family-specific implicit inputs are rejected.
  std::map<std::string, ExecutionValue, std::less<>> inputs;
};

struct ProgramExecutionResult final {
  Status status;
  std::map<std::string, ExecutionValue, std::less<>> outputs;
};

class ProgramExecutionHandle final {
 public:
  using Poll = std::function<std::optional<ProgramExecutionResult>()>;
  using Cancel = std::function<void()>;

  ProgramExecutionHandle();
  ProgramExecutionHandle(const ProgramExecutionHandle&) = delete;
  ProgramExecutionHandle& operator=(const ProgramExecutionHandle&) = delete;
  ProgramExecutionHandle(ProgramExecutionHandle&&) noexcept;
  ProgramExecutionHandle& operator=(ProgramExecutionHandle&&) noexcept;
  ~ProgramExecutionHandle();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::optional<ProgramExecutionResult> poll();
  void cancel() noexcept;
  [[nodiscard]] static ProgramExecutionHandle from_callbacks(
      Poll poll, Cancel cancel);

 private:
  friend class MoeProgramExecutor;
  struct Core;
  explicit ProgramExecutionHandle(std::unique_ptr<Core> core) noexcept;
  std::unique_ptr<Core> core_;
};

struct StartProgramExecutionResult final {
  Status status;
  ProgramExecutionHandle handle;
};

struct StartExactDecodeExecutionResult final {
  Status status;
  ExactDecodeExecutionHandle handle;
};

class ProgramExecutionSession final {
 public:
  ProgramExecutionSession();
  ProgramExecutionSession(const ProgramExecutionSession&) = delete;
  ProgramExecutionSession& operator=(const ProgramExecutionSession&) = delete;
  ProgramExecutionSession(ProgramExecutionSession&&) noexcept;
  ProgramExecutionSession& operator=(ProgramExecutionSession&&) noexcept;
  ~ProgramExecutionSession();

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] StartProgramExecutionResult execute(
      std::map<std::string, ExecutionValue, std::less<>> inputs) const noexcept;
  [[nodiscard]] bool program_sequence_available() const noexcept;
  [[nodiscard]] StartProgramExecutionResult execute_program_sequence(
      std::map<std::string, ExecutionValue, std::less<>> inputs) const noexcept;
  [[nodiscard]] Status checkpoint_retention(
      std::uint32_t next_position) const noexcept;
  [[nodiscard]] Status rewind_retention(
      std::uint32_t next_position) const noexcept;
  [[nodiscard]] Status rebind_request(
      ProgramRequestContext request) const noexcept;
  [[nodiscard]] bool exact_decode_available() const noexcept;
  [[nodiscard]] Status synchronize_exact_decode(
      std::uint32_t next_token, std::uint32_t target_position,
      bool produce_draft) const noexcept;
  [[nodiscard]] Status synchronize_exact_decode_batch(
      std::span<const std::uint32_t> next_tokens,
      std::uint32_t first_target_position,
      bool produce_final_draft) const noexcept;
  [[nodiscard]] StartExactDecodeExecutionResult execute_exact_decode(
      std::uint32_t guaranteed_token, std::uint32_t position,
      std::uint32_t context_limit) const noexcept;
  // Closes the complete request state. An active step is cancelled first.
  void cancel() noexcept;

 private:
  friend class MoeProgramExecutor;
  using Execute = std::function<StartProgramExecutionResult(
      std::map<std::string, ExecutionValue, std::less<>>)>;
  using SequenceAvailable = std::function<bool()>;
  using ExecuteSequence = Execute;
  using RetentionControl = std::function<Status(std::uint32_t)>;
  using Rebind = std::function<Status(ProgramRequestContext)>;
  using ExactDecodeAvailable = std::function<bool()>;
  using SynchronizeExactDecodeBatch = std::function<Status(
      std::span<const std::uint32_t>, std::uint32_t, bool)>;
  using ExecuteExactDecode = std::function<StartExactDecodeExecutionResult(
      std::uint32_t, std::uint32_t, std::uint32_t)>;
  using Cancel = std::function<void()>;
  struct Core;
  explicit ProgramExecutionSession(std::unique_ptr<Core> core) noexcept;
  [[nodiscard]] static ProgramExecutionSession from_callbacks(
      Execute execute, SequenceAvailable sequence_available,
      ExecuteSequence execute_sequence, RetentionControl checkpoint_retention,
      RetentionControl rewind_retention, Rebind rebind,
      ExactDecodeAvailable exact_available,
      SynchronizeExactDecodeBatch synchronize_exact,
      ExecuteExactDecode execute_exact, Cancel cancel);
  std::unique_ptr<Core> core_;
};

struct BeginProgramExecutionSessionResult final {
  Status status;
  ProgramExecutionSession session;
};

// Schema-v3 callable interpreter. It owns prepared operation and immutable
// tensor lifetimes, creates one state object per selected provider/request,
// and publishes outputs only after the complete SSA program succeeds.
class MoeProgramExecutor final {
 public:
  MoeProgramExecutor();
  MoeProgramExecutor(const MoeProgramExecutor&) = delete;
  MoeProgramExecutor& operator=(const MoeProgramExecutor&) = delete;
  MoeProgramExecutor(MoeProgramExecutor&&) noexcept;
  MoeProgramExecutor& operator=(MoeProgramExecutor&&) noexcept;
  ~MoeProgramExecutor();

  [[nodiscard]] static Status create(
      ModelDescriptor descriptor, BoundExecutionProvider provider,
      IModelTensorStore* tensor_store,
      MoeProgramExecutor& destination) noexcept;
  [[nodiscard]] StartProgramExecutionResult execute(
      ProgramExecutionRequest request) const noexcept;
  [[nodiscard]] BeginProgramExecutionSessionResult begin_session(
      ProgramRequestContext request) const noexcept;
  [[nodiscard]] bool valid() const noexcept;

 private:
  struct Core;
  explicit MoeProgramExecutor(std::shared_ptr<const Core> core) noexcept;
  std::shared_ptr<const Core> core_;
};

}  // namespace expert::runtime
