#include "expert/runtime/program_executor.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <mutex>
#include <set>
#include <utility>

namespace expert::runtime {
namespace {

Status copy_status(const Status& status) {
  return {status.code(), std::string(status.message())};
}

Status internal_error(std::string message) {
  return {ErrorCode::internal, std::move(message)};
}

}  // namespace

struct OperationExecutionHandle::Core final {
  Poll poll;
  Cancel cancel;
  bool terminal{};
};

OperationExecutionHandle::OperationExecutionHandle() = default;
OperationExecutionHandle::OperationExecutionHandle(
    std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
OperationExecutionHandle::OperationExecutionHandle(
    OperationExecutionHandle&&) noexcept = default;
OperationExecutionHandle& OperationExecutionHandle::operator=(
    OperationExecutionHandle&& other) noexcept {
  if (this != &other) {
    cancel();
    core_ = std::move(other.core_);
  }
  return *this;
}
OperationExecutionHandle::~OperationExecutionHandle() { cancel(); }

bool OperationExecutionHandle::valid() const noexcept {
  return core_ && !core_->terminal && static_cast<bool>(core_->poll);
}

std::optional<OperationExecutionResult> OperationExecutionHandle::poll() {
  if (!valid()) return std::nullopt;
  try {
    auto result = core_->poll();
    if (result) core_->terminal = true;
    return result;
  } catch (const std::exception& error) {
    if (core_->cancel) {
      try {
        core_->cancel();
      } catch (...) {
      }
    }
    core_->terminal = true;
    return OperationExecutionResult{
        internal_error(std::string("operation provider poll failed: ") +
                       error.what()),
        {}};
  } catch (...) {
    if (core_->cancel) {
      try {
        core_->cancel();
      } catch (...) {
      }
    }
    core_->terminal = true;
    return OperationExecutionResult{
        internal_error("operation provider poll failed"), {}};
  }
}

void OperationExecutionHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  if (core_->cancel) {
    try {
      core_->cancel();
    } catch (...) {
    }
  }
  core_->terminal = true;
}

OperationExecutionHandle OperationExecutionHandle::from_callbacks(
    Poll poll, Cancel cancel) {
  if (!poll || !cancel) return {};
  auto core = std::make_unique<Core>();
  core->poll = std::move(poll);
  core->cancel = std::move(cancel);
  return OperationExecutionHandle(std::move(core));
}

struct ExactDecodeExecutionHandle::Core final {
  Poll poll;
  Cancel cancel;
  bool terminal{};
};

ExactDecodeExecutionHandle::ExactDecodeExecutionHandle() = default;
ExactDecodeExecutionHandle::ExactDecodeExecutionHandle(
    std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
ExactDecodeExecutionHandle::ExactDecodeExecutionHandle(
    ExactDecodeExecutionHandle&&) noexcept = default;
ExactDecodeExecutionHandle& ExactDecodeExecutionHandle::operator=(
    ExactDecodeExecutionHandle&& other) noexcept {
  if (this != &other) {
    cancel();
    core_ = std::move(other.core_);
  }
  return *this;
}
ExactDecodeExecutionHandle::~ExactDecodeExecutionHandle() { cancel(); }

bool ExactDecodeExecutionHandle::valid() const noexcept {
  return core_ && !core_->terminal && static_cast<bool>(core_->poll);
}

std::optional<ExactDecodeExecutionResult>
ExactDecodeExecutionHandle::poll() {
  if (!valid()) return std::nullopt;
  try {
    auto result = core_->poll();
    if (result) core_->terminal = true;
    return result;
  } catch (const std::exception& error) {
    if (core_->cancel) {
      try {
        core_->cancel();
      } catch (...) {
      }
    }
    core_->terminal = true;
    return ExactDecodeExecutionResult{
        internal_error(std::string("exact decode provider poll failed: ") +
                       error.what())};
  } catch (...) {
    if (core_->cancel) {
      try {
        core_->cancel();
      } catch (...) {
      }
    }
    core_->terminal = true;
    return ExactDecodeExecutionResult{
        internal_error("exact decode provider poll failed")};
  }
}

void ExactDecodeExecutionHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  if (core_->cancel) {
    try {
      core_->cancel();
    } catch (...) {
    }
  }
  core_->terminal = true;
}

ExactDecodeExecutionHandle ExactDecodeExecutionHandle::from_callbacks(
    Poll poll, Cancel cancel) {
  if (!poll || !cancel) return {};
  auto core = std::make_unique<Core>();
  core->poll = std::move(poll);
  core->cancel = std::move(cancel);
  return ExactDecodeExecutionHandle(std::move(core));
}

struct ProgramExecutionHandle::Core final {
  Poll poll;
  Cancel cancel;
  bool terminal{};
};

ProgramExecutionHandle::ProgramExecutionHandle() = default;
ProgramExecutionHandle::ProgramExecutionHandle(
    std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
ProgramExecutionHandle::ProgramExecutionHandle(
    ProgramExecutionHandle&&) noexcept = default;
ProgramExecutionHandle& ProgramExecutionHandle::operator=(
    ProgramExecutionHandle&& other) noexcept {
  if (this != &other) {
    cancel();
    core_ = std::move(other.core_);
  }
  return *this;
}
ProgramExecutionHandle::~ProgramExecutionHandle() { cancel(); }

bool ProgramExecutionHandle::valid() const noexcept {
  return core_ && !core_->terminal && static_cast<bool>(core_->poll);
}

std::optional<ProgramExecutionResult> ProgramExecutionHandle::poll() {
  if (!valid()) return std::nullopt;
  try {
    auto result = core_->poll();
    if (result) core_->terminal = true;
    return result;
  } catch (const std::exception& error) {
    if (core_->cancel) {
      try {
        core_->cancel();
      } catch (...) {
      }
    }
    core_->terminal = true;
    return ProgramExecutionResult{
        internal_error(std::string("program execution poll failed: ") +
                       error.what()),
        {}};
  } catch (...) {
    if (core_->cancel) {
      try {
        core_->cancel();
      } catch (...) {
      }
    }
    core_->terminal = true;
    return ProgramExecutionResult{
        internal_error("program execution poll failed"), {}};
  }
}

void ProgramExecutionHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  if (core_->cancel) {
    try {
      core_->cancel();
    } catch (...) {
    }
  }
  core_->terminal = true;
}

ProgramExecutionHandle ProgramExecutionHandle::from_callbacks(
    Poll poll, Cancel cancel) {
  if (!poll || !cancel) return {};
  auto core = std::make_unique<Core>();
  core->poll = std::move(poll);
  core->cancel = std::move(cancel);
  return ProgramExecutionHandle(std::move(core));
}

struct ProgramExecutionSession::Core final {
  Execute execute;
  SequenceAvailable sequence_available;
  ExecuteSequence execute_sequence;
  RetentionControl checkpoint_retention;
  RetentionControl rewind_retention;
  Park park_retention;
  Restore restore_retention;
  SaveSnapshot save_snapshot;
  PruneSnapshots prune_snapshots;
  Rebind rebind;
  TransactionControl begin_transaction;
  TransactionControl end_transaction;
  ExactDecodeAvailable exact_available;
  SynchronizeExactDecodeBatch synchronize_exact;
  ExecuteExactDecode execute_exact;
  Cancel cancel;
  bool terminal{};
};

ProgramExecutionSession::ProgramExecutionSession() = default;
ProgramExecutionSession::ProgramExecutionSession(
    std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
ProgramExecutionSession::ProgramExecutionSession(
    ProgramExecutionSession&&) noexcept = default;
ProgramExecutionSession& ProgramExecutionSession::operator=(
    ProgramExecutionSession&& other) noexcept {
  if (this != &other) {
    cancel();
    core_ = std::move(other.core_);
  }
  return *this;
}
ProgramExecutionSession::~ProgramExecutionSession() { cancel(); }

bool ProgramExecutionSession::valid() const noexcept {
  return core_ && !core_->terminal && static_cast<bool>(core_->execute);
}

StartProgramExecutionResult ProgramExecutionSession::execute(
    std::map<std::string, ExecutionValue, std::less<>> inputs) const noexcept {
  if (!valid())
    return {{ErrorCode::cancelled, "model execution session is closed"}, {}};
  try {
    return core_->execute(std::move(inputs));
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("model execution session step failed: ") +
                 error.what()},
            {}};
  } catch (...) {
    return {{ErrorCode::internal, "model execution session step failed"}, {}};
  }
}

bool ProgramExecutionSession::program_sequence_available() const noexcept {
  if (!valid() || !core_->sequence_available) return false;
  try {
    return core_->sequence_available();
  } catch (...) {
    return false;
  }
}

StartProgramExecutionResult ProgramExecutionSession::execute_program_sequence(
    std::map<std::string, ExecutionValue, std::less<>> inputs) const noexcept {
  if (!valid() || !core_->execute_sequence)
    return {{ErrorCode::invalid_argument,
             "model execution session has no program-sequence provider"},
            {}};
  try {
    return core_->execute_sequence(std::move(inputs));
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("model program-sequence execution failed: ") +
                 error.what()},
            {}};
  } catch (...) {
    return {{ErrorCode::internal,
             "model program-sequence execution failed"},
            {}};
  }
}

Status ProgramExecutionSession::checkpoint_retention(
    std::uint32_t next_position) const noexcept {
  if (!valid() || !core_->checkpoint_retention)
    return {ErrorCode::cancelled, "model execution session is closed"};
  try {
    return core_->checkpoint_retention(next_position);
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("retention checkpoint failed: ") + error.what()};
  } catch (...) {
    return {ErrorCode::internal, "retention checkpoint failed"};
  }
}

Status ProgramExecutionSession::rewind_retention(
    std::uint32_t next_position) const noexcept {
  if (!valid() || !core_->rewind_retention)
    return {ErrorCode::cancelled, "model execution session is closed"};
  try {
    return core_->rewind_retention(next_position);
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("retention rewind failed: ") + error.what()};
  } catch (...) {
    return {ErrorCode::internal, "retention rewind failed"};
  }
}

RequestStateParkingResult ProgramExecutionSession::park_retention(
    std::uint32_t next_position) const noexcept {
  if (!valid() || !core_->park_retention)
    return {{ErrorCode::cancelled, "model execution session is closed"},
            0U, 0U};
  try {
    return core_->park_retention(next_position);
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("request-state parking failed: ") + error.what()},
            0U, 0U};
  } catch (...) {
    return {{ErrorCode::internal, "request-state parking failed"}, 0U, 0U};
  }
}

RequestStateParkingResult ProgramExecutionSession::restore_retention()
    const noexcept {
  if (!valid() || !core_->restore_retention)
    return {{ErrorCode::cancelled, "model execution session is closed"},
            0U, 0U};
  try {
    return core_->restore_retention();
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("request-state restore failed: ") + error.what()},
            0U, 0U};
  } catch (...) {
    return {{ErrorCode::internal, "request-state restore failed"}, 0U, 0U};
  }
}

RequestStateSnapshotResult ProgramExecutionSession::save_retention_snapshot(
    const std::filesystem::path& root,
    std::uint64_t generation) const noexcept {
  if (!valid() || !core_->save_snapshot)
    return {{ErrorCode::cancelled, "model execution session is closed"},
            0U, 0U, 0U, 0U};
  try {
    return core_->save_snapshot(root, generation);
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("request-state snapshot save failed: ") +
                 error.what()},
            0U, 0U, 0U, 0U};
  } catch (...) {
    return {{ErrorCode::internal, "request-state snapshot save failed"},
            0U, 0U, 0U, 0U};
  }
}

Status ProgramExecutionSession::prune_retention_snapshots(
    const std::filesystem::path& root,
    std::uint64_t generation) const noexcept {
  if (!valid() || !core_->prune_snapshots)
    return {ErrorCode::cancelled, "model execution session is closed"};
  try {
    return core_->prune_snapshots(root, generation);
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("request-state snapshot prune failed: ") +
                error.what()};
  } catch (...) {
    return {ErrorCode::internal, "request-state snapshot prune failed"};
  }
}

Status ProgramExecutionSession::rebind_request(
    ProgramRequestContext request) const noexcept {
  if (!valid() || !core_->rebind)
    return {ErrorCode::cancelled, "model execution session is closed"};
  try {
    return core_->rebind(std::move(request));
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("model execution session rebind failed: ") +
                error.what()};
  } catch (...) {
    return {ErrorCode::internal, "model execution session rebind failed"};
  }
}

Status ProgramExecutionSession::begin_retention_transaction() const noexcept {
  if (!valid() || !core_->begin_transaction)
    return {ErrorCode::cancelled, "model execution session is closed"};
  try {
    return core_->begin_transaction();
  } catch (...) {
    return {ErrorCode::internal, "retention transaction start failed"};
  }
}

Status ProgramExecutionSession::end_retention_transaction() const noexcept {
  if (!valid() || !core_->end_transaction)
    return {ErrorCode::cancelled, "model execution session is closed"};
  try {
    return core_->end_transaction();
  } catch (...) {
    return {ErrorCode::internal, "retention transaction end failed"};
  }
}

bool ProgramExecutionSession::exact_decode_available() const noexcept {
  if (!valid() || !core_->exact_available) return false;
  try {
    return core_->exact_available();
  } catch (...) {
    return false;
  }
}

Status ProgramExecutionSession::synchronize_exact_decode(
    std::uint32_t next_token, std::uint32_t target_position,
    bool produce_draft) const noexcept {
  const std::array tokens{next_token};
  return synchronize_exact_decode_batch(tokens, target_position,
                                        produce_draft);
}

Status ProgramExecutionSession::synchronize_exact_decode_batch(
    std::span<const std::uint32_t> next_tokens,
    std::uint32_t first_target_position,
    bool produce_final_draft) const noexcept {
  if (!valid() || !core_->synchronize_exact)
    return {ErrorCode::invalid_argument,
            "model execution session has no exact decode program"};
  try {
    return core_->synchronize_exact(next_tokens, first_target_position,
                                    produce_final_draft);
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("exact decode synchronization failed: ") +
                error.what()};
  } catch (...) {
    return {ErrorCode::internal, "exact decode synchronization failed"};
  }
}

StartExactDecodeExecutionResult
ProgramExecutionSession::execute_exact_decode(
    std::uint32_t guaranteed_token, std::uint32_t position,
    std::uint32_t context_limit) const noexcept {
  if (!valid() || !core_->execute_exact)
    return {{ErrorCode::invalid_argument,
             "model execution session has no exact decode program"},
            {}};
  try {
    return core_->execute_exact(guaranteed_token, position, context_limit);
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("exact decode session step failed: ") +
                 error.what()},
            {}};
  } catch (...) {
    return {{ErrorCode::internal, "exact decode session step failed"}, {}};
  }
}

void ProgramExecutionSession::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  if (core_->cancel) {
    try {
      core_->cancel();
    } catch (...) {
    }
  }
  core_->terminal = true;
}

ProgramExecutionSession ProgramExecutionSession::from_callbacks(
    Execute execute, SequenceAvailable sequence_available,
    ExecuteSequence execute_sequence, RetentionControl checkpoint_retention,
    RetentionControl rewind_retention, Park park_retention,
    Restore restore_retention, SaveSnapshot save_snapshot,
    PruneSnapshots prune_snapshots, Rebind rebind,
    TransactionControl begin_transaction,
    TransactionControl end_transaction,
    ExactDecodeAvailable exact_available,
    SynchronizeExactDecodeBatch synchronize_exact,
    ExecuteExactDecode execute_exact, Cancel cancel) {
  if (!execute || !sequence_available || !execute_sequence ||
      !checkpoint_retention || !rewind_retention || !park_retention ||
      !restore_retention || !save_snapshot || !prune_snapshots || !rebind ||
      !begin_transaction ||
      !end_transaction ||
      !exact_available || !synchronize_exact || !execute_exact || !cancel)
    return {};
  auto core = std::make_unique<Core>();
  core->execute = std::move(execute);
  core->sequence_available = std::move(sequence_available);
  core->execute_sequence = std::move(execute_sequence);
  core->checkpoint_retention = std::move(checkpoint_retention);
  core->rewind_retention = std::move(rewind_retention);
  core->park_retention = std::move(park_retention);
  core->restore_retention = std::move(restore_retention);
  core->save_snapshot = std::move(save_snapshot);
  core->prune_snapshots = std::move(prune_snapshots);
  core->rebind = std::move(rebind);
  core->begin_transaction = std::move(begin_transaction);
  core->end_transaction = std::move(end_transaction);
  core->exact_available = std::move(exact_available);
  core->synchronize_exact = std::move(synchronize_exact);
  core->execute_exact = std::move(execute_exact);
  core->cancel = std::move(cancel);
  return ProgramExecutionSession(std::move(core));
}

struct MoeProgramExecutor::Core final {
  struct PreparedInstruction final {
    std::uint32_t provider_registry_index{};
    std::shared_ptr<IOperationProvider> provider;
    std::shared_ptr<const IPreparedOperation> operation;
    std::vector<PreparedTensorBinding> tensors;
  };

  struct PreparedExactDecode final {
    std::uint32_t provider_registry_index{};
    std::shared_ptr<IOperationProvider> provider;
    std::shared_ptr<const IPreparedOperation> operation;
    std::vector<PreparedTensorBinding> tensors;
    std::uint32_t maximum_emitted_tokens{};
  };

  struct PreparedSequence final {
    std::uint32_t provider_registry_index{};
    std::shared_ptr<IOperationProvider> provider;
    std::vector<const IPreparedOperation*> operations;
  };

  struct RequestState;
  struct ExactDecodeState;

  struct SessionState final
      : public std::enable_shared_from_this<SessionState> {
    std::shared_ptr<const Core> program;
    ProgramRequestContext request;
    std::map<std::uint32_t, std::shared_ptr<IOperationProviderRequestState>>
        provider_states;
    std::mutex mutex;
    std::weak_ptr<RequestState> active_step;
    std::weak_ptr<ExactDecodeState> active_exact;
    bool active{};
    bool parked{};
    bool closed{};
    bool retention_transaction{};
    std::uint32_t parked_position{};
    std::uint64_t parked_pages{};
    std::uint64_t parked_bytes{};

    [[nodiscard]] StartProgramExecutionResult start(
        std::map<std::string, ExecutionValue, std::less<>> inputs,
        bool close_after_success);
    [[nodiscard]] bool sequence_available() const noexcept;
    [[nodiscard]] StartProgramExecutionResult start_sequence(
        std::map<std::string, ExecutionValue, std::less<>> inputs);
    [[nodiscard]] Status checkpoint_retention(
        std::uint32_t next_position) noexcept;
    [[nodiscard]] Status rewind_retention(
        std::uint32_t next_position) noexcept;
    [[nodiscard]] RequestStateParkingResult park_retention(
        std::uint32_t next_position) noexcept;
    [[nodiscard]] RequestStateParkingResult restore_retention() noexcept;
    [[nodiscard]] RequestStateSnapshotResult save_snapshot(
        const std::filesystem::path& root,
        std::uint64_t generation) noexcept;
    [[nodiscard]] RequestStateSnapshotResult load_snapshot(
        const std::filesystem::path& root, std::uint64_t generation,
        std::uint32_t next_position) noexcept;
    [[nodiscard]] Status prune_snapshots(
        const std::filesystem::path& root,
        std::uint64_t generation) noexcept;
    [[nodiscard]] bool exact_available() const noexcept;
    [[nodiscard]] Status synchronize_exact(
        std::span<const std::uint32_t> next_tokens,
        std::uint32_t first_target_position,
        bool produce_final_draft) noexcept;
    [[nodiscard]] StartExactDecodeExecutionResult start_exact(
        std::uint32_t guaranteed_token, std::uint32_t position,
        std::uint32_t context_limit);
    void finish(RequestState* step, bool success) noexcept;
    void finish_exact(ExactDecodeState* step, bool success) noexcept;
    [[nodiscard]] Status rebind(ProgramRequestContext replacement) noexcept;
    [[nodiscard]] Status begin_transaction() noexcept;
    [[nodiscard]] Status end_transaction() noexcept;
    void cancel() noexcept;
    void close() noexcept;
  };

  struct CreateSessionResult final {
    Status status;
    std::shared_ptr<SessionState> session;
  };

  struct RequestState final {
    std::shared_ptr<SessionState> session;
    std::vector<std::optional<ExecutionValue>> values;
    std::vector<ExecutionValue> current_inputs;
    OperationExecutionHandle inflight;
    std::size_t next_operation{};
    bool close_after_success{};
    bool direct_sequence{};
    bool terminal{};
    bool cancelled_result_pending{};

    [[nodiscard]] std::optional<ProgramExecutionResult> fail(Status status) {
      if (inflight.valid()) inflight.cancel();
      inflight = OperationExecutionHandle{};
      current_inputs.clear();
      values.clear();
      terminal = true;
      session->finish(this, false);
      return ProgramExecutionResult{std::move(status), {}};
    }

    [[nodiscard]] std::optional<ProgramExecutionResult> poll() {
      if (cancelled_result_pending) {
        cancelled_result_pending = false;
        return ProgramExecutionResult{
            {ErrorCode::cancelled, "model execution session was cancelled"},
            {}};
      }
      if (terminal) return std::nullopt;
      if (direct_sequence) {
        if (std::chrono::steady_clock::now() > session->request.deadline)
          return fail({ErrorCode::deadline_exceeded,
                       "model program-sequence deadline expired"});
        auto completed = inflight.poll();
        if (!completed) return std::nullopt;
        inflight = OperationExecutionHandle{};
        if (!completed->status.ok())
          return fail(copy_status(completed->status));
        if (completed->outputs.size() !=
            session->program->provider.program.outputs.size())
          return fail(internal_error(
              "program-sequence provider returned a partial or extra output set"));
        ProgramExecutionResult result;
        result.status = Status::success();
        for (std::size_t index = 0U; index < completed->outputs.size();
             ++index) {
          auto& value = completed->outputs[index];
          const auto& endpoint =
              session->program->provider.program.outputs[index];
          if (endpoint.value_index >=
                  session->program->provider.program.values.size() ||
              !value.valid() ||
              value.abi != session->program->provider.program
                               .values[endpoint.value_index]
                               .abi)
            return fail(internal_error(
                "program-sequence provider returned an invalid output value"));
          result.outputs.emplace(endpoint.role, std::move(value));
        }
        current_inputs.clear();
        terminal = true;
        session->finish(this, true);
        return result;
      }
      while (true) {
        if (std::chrono::steady_clock::now() > session->request.deadline)
          return fail({ErrorCode::deadline_exceeded,
                       "model program execution deadline expired"});

        if (inflight.valid()) {
          auto completed = inflight.poll();
          if (!completed) return std::nullopt;
          inflight = OperationExecutionHandle{};
          if (!completed->status.ok())
            return fail(copy_status(completed->status));
          const auto& compiled =
              session->program->provider.program.operations.at(next_operation);
          if (completed->outputs.size() != compiled.output_values.size())
            return fail(internal_error(
                "operation provider returned a partial or extra output set"));
          for (std::size_t index = 0U; index < completed->outputs.size();
               ++index) {
            auto& value = completed->outputs[index];
            const auto value_index = compiled.output_values[index].value_index;
            if (value_index >= values.size() || values[value_index] ||
                !value.valid() ||
                value.abi !=
                    session->program->provider.program.values[value_index].abi)
              return fail(internal_error(
                  "operation provider returned an invalid output value"));
            values[value_index].emplace(std::move(value));
          }
          current_inputs.clear();
          ++next_operation;
        }

        if (next_operation == session->program->prepared.size()) {
          ProgramExecutionResult result;
          result.status = Status::success();
          for (const auto& endpoint :
               session->program->provider.program.outputs) {
            if (endpoint.value_index >= values.size() ||
                !values[endpoint.value_index])
              return fail(internal_error(
                  "model program completed without an external output"));
            result.outputs.emplace(
                endpoint.role, *values[endpoint.value_index]);
          }
          current_inputs.clear();
          values.clear();
          terminal = true;
          session->finish(this, true);
          if (close_after_success) session->close();
          return result;
        }

        const auto& prepared = session->program->prepared[next_operation];
        const auto& compiled =
            session->program->provider.program.operations[next_operation];
        current_inputs.clear();
        current_inputs.reserve(compiled.input_values.size());
        for (const auto& binding : compiled.input_values) {
          if (binding.value_index >= values.size() ||
              !values[binding.value_index])
            return fail(internal_error(
                "model program attempted to consume an unavailable value"));
          current_inputs.push_back(*values[binding.value_index]);
        }
        const auto state = session->provider_states.find(
            prepared.provider_registry_index);
        if (state == session->provider_states.end())
          return fail(internal_error(
              "model program lacks provider request state"));
        inflight = prepared.provider->execute(
            *prepared.operation, state->second,
            OperationInvocation{session->request, compiled, current_inputs});
        if (!inflight.valid())
          return fail(internal_error(
              "operation provider rejected a bound invocation"));
      }
    }

    void cancel(bool publish_result) noexcept {
      if (terminal) return;
      if (inflight.valid()) inflight.cancel();
      inflight = OperationExecutionHandle{};
      current_inputs.clear();
      values.clear();
      terminal = true;
      cancelled_result_pending = publish_result;
      session->finish(this, false);
    }
  };

  struct ExactDecodeState final {
    std::shared_ptr<SessionState> session;
    ExactDecodeExecutionHandle inflight;
    std::uint32_t guaranteed_token{};
    std::uint32_t position{};
    std::uint32_t context_limit{};
    std::uint32_t vocabulary_size{};
    std::uint32_t maximum_emitted_tokens{};
    bool terminal{};

    [[nodiscard]] std::optional<ExactDecodeExecutionResult> poll() {
      if (terminal) return std::nullopt;
      auto completed = inflight.poll();
      if (!completed) return std::nullopt;
      inflight = ExactDecodeExecutionHandle{};
      if (!completed->status.ok()) {
        terminal = true;
        session->finish_exact(this, false);
        return completed;
      }
      const auto count = completed->emitted_tokens.size();
      if (count == 0U || count > maximum_emitted_tokens ||
          completed->emitted_tokens.front() != guaranteed_token ||
          completed->positions_advanced != count ||
          completed->next_token >= vocabulary_size ||
          std::any_of(completed->emitted_tokens.begin(),
                      completed->emitted_tokens.end(),
                      [this](std::uint32_t token) {
                        return token >= vocabulary_size;
                      }) ||
          position > context_limit ||
          count > static_cast<std::size_t>(context_limit - position)) {
        terminal = true;
        session->finish_exact(this, false);
        return ExactDecodeExecutionResult{
            internal_error("exact decode provider violated its artifact contract")};
      }
      terminal = true;
      session->finish_exact(this, true);
      return completed;
    }

    void cancel() noexcept {
      if (terminal) return;
      if (inflight.valid()) inflight.cancel();
      inflight = ExactDecodeExecutionHandle{};
      terminal = true;
      session->finish_exact(this, false);
    }
  };

  ModelDescriptor descriptor;
  BoundExecutionProvider provider;
  std::vector<PreparedInstruction> prepared;
  std::optional<PreparedExactDecode> exact_decode;
  std::optional<PreparedSequence> sequence;

  [[nodiscard]] static CreateSessionResult begin(
      std::shared_ptr<const Core> program,
      ProgramRequestContext request) noexcept;
};

MoeProgramExecutor::Core::CreateSessionResult MoeProgramExecutor::Core::begin(
    std::shared_ptr<const Core> program,
    ProgramRequestContext request) noexcept {
  try {
    if (!program || request.request_id == 0U)
      return {{ErrorCode::invalid_argument,
               "model execution request or executor is invalid"},
              {}};
    if (std::chrono::steady_clock::now() > request.deadline)
      return {{ErrorCode::deadline_exceeded,
               "model execution request deadline already expired"},
              {}};
    auto session = std::make_shared<SessionState>();
    session->program = std::move(program);
    session->request = std::move(request);
    for (const auto& prepared : session->program->prepared) {
      if (session->provider_states.contains(
              prepared.provider_registry_index))
        continue;
      auto created =
          prepared.provider->create_request_state(session->request);
      if (!created.status.ok())
        return {{created.status.code(), std::string(created.status.message())},
                {}};
      session->provider_states.emplace(prepared.provider_registry_index,
                                       std::move(created.state));
    }
    if (session->program->exact_decode &&
        !session->provider_states.contains(
            session->program->exact_decode->provider_registry_index)) {
      const auto& prepared = *session->program->exact_decode;
      auto created = prepared.provider->create_request_state(session->request);
      if (!created.status.ok())
        return {{created.status.code(), std::string(created.status.message())},
                {}};
      session->provider_states.emplace(prepared.provider_registry_index,
                                       std::move(created.state));
    }
    return {Status::success(), std::move(session)};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("model execution session start failed: ") +
                 error.what()},
            {}};
  } catch (...) {
    return {{ErrorCode::internal, "model execution session start failed"}, {}};
  }
}

StartProgramExecutionResult MoeProgramExecutor::Core::SessionState::start(
    std::map<std::string, ExecutionValue, std::less<>> inputs,
    bool close_after) {
  if (inputs.size() != program->provider.program.inputs.size())
    return {{ErrorCode::invalid_argument,
             "model execution input role set is incomplete"},
            {}};
  auto step = std::make_shared<RequestState>();
  step->session = shared_from_this();
  step->close_after_success = close_after;
  step->values.resize(program->provider.program.values.size());
  for (const auto& endpoint : program->provider.program.inputs) {
    const auto input = inputs.find(endpoint.role);
    if (input == inputs.end() || !input->second.valid() ||
        endpoint.value_index >= step->values.size() ||
        input->second.abi !=
            program->provider.program.values[endpoint.value_index].abi)
      return {{ErrorCode::invalid_argument,
               "model execution input is absent, invalid, or ABI-mismatched"},
              {}};
    step->values[endpoint.value_index].emplace(input->second);
  }
  {
    std::lock_guard lock(mutex);
    if (closed)
      return {{ErrorCode::cancelled,
               "model execution session is closed"},
              {}};
    if (active)
      return {{ErrorCode::backpressure,
               "model execution session already has an active step"},
              {}};
    active = true;
    active_step = step;
  }
  auto handle = ProgramExecutionHandle::from_callbacks(
      [step] { return step->poll(); },
      [step] { step->cancel(false); });
  if (!handle.valid()) {
    step->cancel(false);
    return {{ErrorCode::internal,
             "model execution handle construction failed"},
            {}};
  }
  return {Status::success(), std::move(handle)};
}

bool MoeProgramExecutor::Core::SessionState::sequence_available() const
    noexcept {
  return program && program->sequence.has_value();
}

StartProgramExecutionResult
MoeProgramExecutor::Core::SessionState::start_sequence(
    std::map<std::string, ExecutionValue, std::less<>> inputs) {
  if (!program || !program->sequence ||
      inputs.size() != program->provider.program.inputs.size())
    return {{ErrorCode::invalid_argument,
             "model program-sequence input role set is incomplete"},
            {}};

  auto step = std::make_shared<RequestState>();
  step->session = shared_from_this();
  step->direct_sequence = true;
  step->current_inputs.reserve(program->provider.program.inputs.size());
  for (const auto& endpoint : program->provider.program.inputs) {
    const auto input = inputs.find(endpoint.role);
    if (input == inputs.end() || !input->second.valid() ||
        endpoint.value_index >= program->provider.program.values.size() ||
        input->second.abi !=
            program->provider.program.values[endpoint.value_index].abi)
      return {{ErrorCode::invalid_argument,
               "model program-sequence input is absent, invalid, or "
               "ABI-mismatched"},
              {}};
    step->current_inputs.push_back(input->second);
  }

  std::shared_ptr<IOperationProviderRequestState> provider_state;
  const auto& sequence = *program->sequence;
  {
    std::lock_guard lock(mutex);
    const auto found =
        provider_states.find(sequence.provider_registry_index);
    if (closed)
      return {{ErrorCode::cancelled,
               "model execution session is closed"},
              {}};
    if (active)
      return {{ErrorCode::backpressure,
               "model execution session already has an active step"},
              {}};
    if (found == provider_states.end())
      return {internal_error(
                  "program-sequence provider request state is absent"),
              {}};
    active = true;
    active_step = step;
    provider_state = found->second;
  }

  step->inflight = sequence.provider->execute_program_sequence(
      provider_state,
      ProgramSequenceInvocation{request, program->provider.program,
                                sequence.operations,
                                step->current_inputs});
  if (!step->inflight.valid()) {
    step->cancel(false);
    return {internal_error(
                "operation provider rejected a program-sequence invocation"),
            {}};
  }

  auto handle = ProgramExecutionHandle::from_callbacks(
      [step] { return step->poll(); },
      [step] { step->cancel(false); });
  if (!handle.valid()) {
    step->cancel(false);
    return {{ErrorCode::internal,
             "model program-sequence handle construction failed"},
            {}};
  }
  return {Status::success(), std::move(handle)};
}

Status MoeProgramExecutor::Core::SessionState::checkpoint_retention(
    std::uint32_t next_position) noexcept {
  try {
    std::lock_guard lock(mutex);
    if (closed)
      return {ErrorCode::cancelled, "model execution session is closed"};
    if (active)
      return {ErrorCode::backpressure,
              "model execution session already has an active step"};
    for (const auto& [registry_index, state] : provider_states) {
      const auto prepared = std::find_if(
          program->prepared.begin(), program->prepared.end(),
          [registry_index](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index;
          });
      const auto implementation =
          prepared == program->prepared.end()
              ? program->exact_decode &&
                        program->exact_decode->provider_registry_index ==
                            registry_index
                    ? program->exact_decode->provider
                    : std::shared_ptr<IOperationProvider>{}
              : prepared->provider;
      if (!implementation)
        return internal_error("retention checkpoint provider is absent");
      const auto status =
          implementation->checkpoint_request_state(state, next_position);
      if (!status.ok()) return copy_status(status);
    }
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("retention checkpoint failed: ") + error.what()};
  } catch (...) {
    return {ErrorCode::internal, "retention checkpoint failed"};
  }
}

Status MoeProgramExecutor::Core::SessionState::rewind_retention(
    std::uint32_t next_position) noexcept {
  try {
    std::lock_guard lock(mutex);
    if (closed)
      return {ErrorCode::cancelled, "model execution session is closed"};
    if (active)
      return {ErrorCode::backpressure,
              "model execution session already has an active step"};
    for (const auto& [registry_index, state] : provider_states) {
      const auto prepared = std::find_if(
          program->prepared.begin(), program->prepared.end(),
          [registry_index](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index;
          });
      const auto implementation =
          prepared == program->prepared.end()
              ? program->exact_decode &&
                        program->exact_decode->provider_registry_index ==
                            registry_index
                    ? program->exact_decode->provider
                    : std::shared_ptr<IOperationProvider>{}
              : prepared->provider;
      if (!implementation)
        return internal_error("retention rewind provider is absent");
      const auto status =
          implementation->rewind_request_state(state, next_position);
      if (!status.ok()) return copy_status(status);
    }
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("retention rewind failed: ") + error.what()};
  } catch (...) {
    return {ErrorCode::internal, "retention rewind failed"};
  }
}

RequestStateParkingResult
MoeProgramExecutor::Core::SessionState::park_retention(
    std::uint32_t next_position) noexcept {
  try {
    std::lock_guard lock(mutex);
    if (closed)
      return {{ErrorCode::cancelled, "model execution session is closed"},
              0U, 0U};
    if (active)
      return {{ErrorCode::backpressure,
               "model execution session already has an active step"},
              0U, 0U};
    if (parked)
      return {{ErrorCode::invalid_argument,
               "model execution session is already parked"},
              0U, 0U};

    const auto provider_for = [this](std::uint32_t registry_index) {
      const auto prepared = std::find_if(
          program->prepared.begin(), program->prepared.end(),
          [registry_index](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index;
          });
      return prepared == program->prepared.end()
                 ? program->exact_decode &&
                           program->exact_decode->provider_registry_index ==
                               registry_index
                       ? program->exact_decode->provider
                       : std::shared_ptr<IOperationProvider>{}
                 : prepared->provider;
    };

    std::vector<std::pair<std::shared_ptr<IOperationProvider>,
                          std::shared_ptr<IOperationProviderRequestState>>>
        completed;
    std::uint64_t pages{};
    std::uint64_t bytes{};
    for (const auto& [registry_index, state] : provider_states) {
      const auto implementation = provider_for(registry_index);
      if (!implementation ||
          !implementation->supports_request_state_parking()) {
        for (auto item = completed.rbegin(); item != completed.rend(); ++item)
          static_cast<void>(item->first->restore_request_state(item->second));
        return {{ErrorCode::invalid_argument,
                 "request-state parking is not supported by every provider"},
                0U, 0U};
      }
      auto result =
          implementation->park_request_state(state, next_position);
      if (!result.status.ok()) {
        for (auto item = completed.rbegin(); item != completed.rend(); ++item)
          static_cast<void>(item->first->restore_request_state(item->second));
        return result;
      }
      if (pages > std::numeric_limits<std::uint64_t>::max() -
                      result.populated_pages ||
          bytes > std::numeric_limits<std::uint64_t>::max() -
                      result.parked_bytes) {
        static_cast<void>(implementation->restore_request_state(state));
        for (auto item = completed.rbegin(); item != completed.rend(); ++item)
          static_cast<void>(item->first->restore_request_state(item->second));
        return {{ErrorCode::internal,
                 "request-state parking accounting overflowed"},
                0U, 0U};
      }
      pages += result.populated_pages;
      bytes += result.parked_bytes;
      completed.emplace_back(implementation, state);
    }
    parked = true;
    parked_position = next_position;
    parked_pages = pages;
    parked_bytes = bytes;
    return {Status::success(), pages, bytes};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("request-state parking failed: ") + error.what()},
            0U, 0U};
  } catch (...) {
    return {{ErrorCode::internal, "request-state parking failed"}, 0U, 0U};
  }
}

RequestStateParkingResult
MoeProgramExecutor::Core::SessionState::restore_retention() noexcept {
  try {
    std::lock_guard lock(mutex);
    if (closed)
      return {{ErrorCode::cancelled, "model execution session is closed"},
              0U, 0U};
    if (active)
      return {{ErrorCode::backpressure,
               "model execution session already has an active step"},
              0U, 0U};
    if (!parked)
      return {{ErrorCode::invalid_argument,
               "model execution session is not parked"},
              0U, 0U};

    const auto provider_for = [this](std::uint32_t registry_index) {
      const auto prepared = std::find_if(
          program->prepared.begin(), program->prepared.end(),
          [registry_index](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index;
          });
      return prepared == program->prepared.end()
                 ? program->exact_decode &&
                           program->exact_decode->provider_registry_index ==
                               registry_index
                       ? program->exact_decode->provider
                       : std::shared_ptr<IOperationProvider>{}
                 : prepared->provider;
    };

    std::vector<std::pair<std::shared_ptr<IOperationProvider>,
                          std::shared_ptr<IOperationProviderRequestState>>>
        restored;
    for (const auto& [registry_index, state] : provider_states) {
      const auto implementation = provider_for(registry_index);
      if (!implementation)
        return {{ErrorCode::internal,
                 "request-state restore provider is absent"},
                0U, 0U};
      auto result = implementation->restore_request_state(state);
      if (!result.status.ok()) {
        for (auto item = restored.rbegin(); item != restored.rend(); ++item)
          static_cast<void>(item->first->park_request_state(
              item->second, parked_position));
        return result;
      }
      restored.emplace_back(implementation, state);
    }
    const auto pages = parked_pages;
    const auto bytes = parked_bytes;
    parked = false;
    parked_position = 0U;
    parked_pages = 0U;
    parked_bytes = 0U;
    return {Status::success(), pages, bytes};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("request-state restore failed: ") + error.what()},
            0U, 0U};
  } catch (...) {
    return {{ErrorCode::internal, "request-state restore failed"}, 0U, 0U};
  }
}

RequestStateSnapshotResult
MoeProgramExecutor::Core::SessionState::save_snapshot(
    const std::filesystem::path& root,
    std::uint64_t generation) noexcept {
  try {
    std::lock_guard lock(mutex);
    if (closed)
      return {{ErrorCode::cancelled, "model execution session is closed"},
              0U, 0U, 0U, 0U};
    if (active)
      return {{ErrorCode::backpressure,
               "model execution session already has an active step"},
              0U, 0U, 0U, 0U};
    if (!parked || generation == 0U || root.empty())
      return {{ErrorCode::invalid_argument,
               "only parked request state can be persisted"},
              0U, 0U, 0U, 0U};

    const auto provider_for = [this](std::uint32_t registry_index) {
      const auto prepared = std::find_if(
          program->prepared.begin(), program->prepared.end(),
          [registry_index](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index;
          });
      return prepared == program->prepared.end()
                 ? program->exact_decode &&
                           program->exact_decode->provider_registry_index ==
                               registry_index
                       ? program->exact_decode->provider
                       : std::shared_ptr<IOperationProvider>{}
                 : prepared->provider;
    };
    std::uint64_t pages{};
    std::uint64_t logical{};
    std::uint64_t written{};
    for (const auto& [registry_index, state] : provider_states) {
      const auto implementation = provider_for(registry_index);
      if (!implementation ||
          !implementation->supports_request_state_persistence())
        return {{ErrorCode::invalid_argument,
                 "request-state persistence is not supported by every "
                 "provider"},
                0U, 0U, 0U, 0U};
      const auto result = implementation->save_request_state_snapshot(
          state, root / ("provider-" + std::to_string(registry_index)),
          generation);
      if (!result.status.ok()) return result;
      if (pages > std::numeric_limits<std::uint64_t>::max() -
                      result.populated_pages ||
          logical > std::numeric_limits<std::uint64_t>::max() -
                        result.logical_bytes ||
          written > std::numeric_limits<std::uint64_t>::max() -
                        result.written_bytes)
        return {{ErrorCode::internal,
                 "request-state snapshot accounting overflowed"},
                0U, 0U, 0U, 0U};
      pages += result.populated_pages;
      logical += result.logical_bytes;
      written += result.written_bytes;
    }
    return {Status::success(), generation, pages, logical, written};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("request-state snapshot save failed: ") +
                 error.what()},
            0U, 0U, 0U, 0U};
  } catch (...) {
    return {{ErrorCode::internal, "request-state snapshot save failed"},
            0U, 0U, 0U, 0U};
  }
}

RequestStateSnapshotResult
MoeProgramExecutor::Core::SessionState::load_snapshot(
    const std::filesystem::path& root, std::uint64_t generation,
    std::uint32_t next_position) noexcept {
  try {
    std::lock_guard lock(mutex);
    if (closed || active || parked || generation == 0U || root.empty() ||
        next_position == 0U)
      return {{ErrorCode::invalid_argument,
               "request-state snapshot load target is invalid"},
              0U, 0U, 0U, 0U};
    const auto provider_for = [this](std::uint32_t registry_index) {
      const auto prepared = std::find_if(
          program->prepared.begin(), program->prepared.end(),
          [registry_index](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index;
          });
      return prepared == program->prepared.end()
                 ? program->exact_decode &&
                           program->exact_decode->provider_registry_index ==
                               registry_index
                       ? program->exact_decode->provider
                       : std::shared_ptr<IOperationProvider>{}
                 : prepared->provider;
    };
    std::uint64_t pages{};
    std::uint64_t logical{};
    for (const auto& [registry_index, state] : provider_states) {
      const auto implementation = provider_for(registry_index);
      if (!implementation ||
          !implementation->supports_request_state_persistence())
        return {{ErrorCode::invalid_argument,
                 "request-state persistence is not supported by every "
                 "provider"},
                0U, 0U, 0U, 0U};
      const auto result = implementation->load_request_state_snapshot(
          state, root / ("provider-" + std::to_string(registry_index)),
          generation);
      if (!result.status.ok()) return result;
      if (pages > std::numeric_limits<std::uint64_t>::max() -
                      result.populated_pages ||
          logical > std::numeric_limits<std::uint64_t>::max() -
                        result.logical_bytes)
        return {{ErrorCode::internal,
                 "request-state snapshot accounting overflowed"},
                0U, 0U, 0U, 0U};
      pages += result.populated_pages;
      logical += result.logical_bytes;
    }
    parked = true;
    parked_position = next_position;
    parked_pages = pages;
    parked_bytes = logical;
    return {Status::success(), generation, pages, logical, 0U};
  } catch (const std::exception& error) {
    return {{ErrorCode::internal,
             std::string("request-state snapshot load failed: ") +
                 error.what()},
            0U, 0U, 0U, 0U};
  } catch (...) {
    return {{ErrorCode::internal, "request-state snapshot load failed"},
            0U, 0U, 0U, 0U};
  }
}

Status MoeProgramExecutor::Core::SessionState::prune_snapshots(
    const std::filesystem::path& root,
    std::uint64_t generation) noexcept {
  try {
    std::lock_guard lock(mutex);
    if (closed || active || generation == 0U || root.empty())
      return {ErrorCode::invalid_argument,
              "request-state snapshot prune target is invalid"};
    const auto provider_for = [this](std::uint32_t registry_index) {
      const auto prepared = std::find_if(
          program->prepared.begin(), program->prepared.end(),
          [registry_index](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index;
          });
      return prepared == program->prepared.end()
                 ? program->exact_decode &&
                           program->exact_decode->provider_registry_index ==
                               registry_index
                       ? program->exact_decode->provider
                       : std::shared_ptr<IOperationProvider>{}
                 : prepared->provider;
    };
    for (const auto& [registry_index, state] : provider_states) {
      (void)state;
      const auto implementation = provider_for(registry_index);
      if (!implementation ||
          !implementation->supports_request_state_persistence())
        return {ErrorCode::invalid_argument,
                "request-state persistence is not supported by every "
                "provider"};
      const auto status = implementation->prune_request_state_snapshots(
          root / ("provider-" + std::to_string(registry_index)),
          generation);
      if (!status.ok()) return copy_status(status);
    }
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("request-state snapshot prune failed: ") +
                error.what()};
  } catch (...) {
    return {ErrorCode::internal, "request-state snapshot prune failed"};
  }
}

bool MoeProgramExecutor::Core::SessionState::exact_available() const noexcept {
  return program && program->exact_decode.has_value();
}

Status MoeProgramExecutor::Core::SessionState::synchronize_exact(
    std::span<const std::uint32_t> next_tokens,
    std::uint32_t first_target_position,
    bool produce_final_draft) noexcept {
  if (!program || !program->exact_decode || next_tokens.empty() ||
      next_tokens.size() > 0xffffffffULL ||
      first_target_position >
          0xffffffffU - static_cast<std::uint32_t>(next_tokens.size() - 1U) ||
      std::any_of(next_tokens.begin(), next_tokens.end(),
                  [this](std::uint32_t token) {
                    return token >= program->descriptor.vocab_size;
                  }))
    return {ErrorCode::invalid_argument,
            "exact decode synchronization is unavailable or invalid"};
  const auto& prepared = *program->exact_decode;
  std::shared_ptr<IOperationProviderRequestState> provider_state;
  {
    std::lock_guard lock(mutex);
    const auto found = provider_states.find(prepared.provider_registry_index);
    if (closed)
      return {ErrorCode::cancelled, "model execution session is closed"};
    if (active)
      return {ErrorCode::backpressure,
              "model execution session already has an active step"};
    if (found == provider_states.end())
      return internal_error("exact decode provider request state is absent");
    active = true;
    provider_state = found->second;
  }
  auto status = prepared.provider->synchronize_exact_decode_batch(
      *prepared.operation, provider_state,
      ExactDecodeSynchronizationBatch{request, next_tokens,
                                      first_target_position,
                                      produce_final_draft});
  {
    std::lock_guard lock(mutex);
    active = false;
    if (!status.ok()) {
      closed = true;
      provider_states.clear();
    }
  }
  return status;
}

StartExactDecodeExecutionResult
MoeProgramExecutor::Core::SessionState::start_exact(
    std::uint32_t guaranteed_token, std::uint32_t position,
    std::uint32_t context_limit) {
  if (!program || !program->exact_decode ||
      guaranteed_token >= program->descriptor.vocab_size ||
      position >= context_limit ||
      context_limit > program->descriptor.max_context_tokens)
    return {{ErrorCode::invalid_argument,
             "exact decode invocation is unavailable or invalid"},
            {}};
  const auto& prepared = *program->exact_decode;
  auto step = std::make_shared<ExactDecodeState>();
  step->session = shared_from_this();
  step->guaranteed_token = guaranteed_token;
  step->position = position;
  step->context_limit = context_limit;
  step->vocabulary_size = program->descriptor.vocab_size;
  step->maximum_emitted_tokens = prepared.maximum_emitted_tokens;
  std::shared_ptr<IOperationProviderRequestState> provider_state;
  {
    std::lock_guard lock(mutex);
    const auto found = provider_states.find(prepared.provider_registry_index);
    if (closed)
      return {{ErrorCode::cancelled, "model execution session is closed"}, {}};
    if (active)
      return {{ErrorCode::backpressure,
               "model execution session already has an active step"},
              {}};
    if (found == provider_states.end())
      return {internal_error("exact decode provider request state is absent"),
              {}};
    active = true;
    active_exact = step;
    provider_state = found->second;
  }
  step->inflight = prepared.provider->execute_exact_decode(
      *prepared.operation, provider_state,
      ExactDecodeInvocation{request, guaranteed_token, position,
                            context_limit});
  if (!step->inflight.valid()) {
    step->terminal = true;
    finish_exact(step.get(), false);
    return {internal_error("exact decode provider rejected its invocation"),
            {}};
  }
  auto handle = ExactDecodeExecutionHandle::from_callbacks(
      [step] { return step->poll(); }, [step] { step->cancel(); });
  if (!handle.valid()) {
    step->cancel();
    return {internal_error("exact decode handle construction failed"), {}};
  }
  return {Status::success(), std::move(handle)};
}

void MoeProgramExecutor::Core::SessionState::finish(
    RequestState* step, bool success) noexcept {
  std::lock_guard lock(mutex);
  const auto current = active_step.lock();
  if (current && current.get() == step) {
    active_step.reset();
    active = false;
  }
  if (!success && !retention_transaction) closed = true;
  if (closed) provider_states.clear();
}

void MoeProgramExecutor::Core::SessionState::finish_exact(
    ExactDecodeState* step, bool success) noexcept {
  std::lock_guard lock(mutex);
  const auto current = active_exact.lock();
  if (current && current.get() == step) {
    active_exact.reset();
    active = false;
  }
  if (!success && !retention_transaction) closed = true;
  if (closed) provider_states.clear();
}

void MoeProgramExecutor::Core::SessionState::close() noexcept {
  std::lock_guard lock(mutex);
  closed = true;
  if (!active) provider_states.clear();
}

Status MoeProgramExecutor::Core::SessionState::rebind(
    ProgramRequestContext replacement) noexcept {
  if (replacement.request_id == 0U ||
      std::chrono::steady_clock::now() > replacement.deadline)
    return {ErrorCode::invalid_argument,
            "replacement request context is invalid"};
  std::lock_guard lock(mutex);
  if (closed)
    return {ErrorCode::cancelled, "model execution session is closed"};
  if (active)
    return {ErrorCode::backpressure,
            "model execution session has an active step"};
  for (const auto& [name, value] : request.parameters)
    replacement.parameters.try_emplace(name, value);
  for (const auto& [registry_index, state] : provider_states) {
    const auto prepared = std::find_if(
        program->prepared.begin(), program->prepared.end(),
        [registry_index](const Core::PreparedInstruction& item) {
          return item.provider_registry_index == registry_index;
        });
    const auto implementation =
        prepared == program->prepared.end()
            ? program->exact_decode &&
                      program->exact_decode->provider_registry_index ==
                          registry_index
                  ? program->exact_decode->provider
                  : std::shared_ptr<IOperationProvider>{}
            : prepared->provider;
    if (!implementation)
      return internal_error("request rebind provider is absent");
    const auto rebound =
        implementation->rebind_request_state(state, replacement);
    if (!rebound.ok()) return copy_status(rebound);
  }
  request = std::move(replacement);
  return Status::success();
}

Status MoeProgramExecutor::Core::SessionState::begin_transaction() noexcept {
  std::lock_guard lock(mutex);
  if (closed)
    return {ErrorCode::cancelled, "model execution session is closed"};
  if (active || parked || retention_transaction)
    return {ErrorCode::backpressure,
            "model execution session cannot start a retention transaction"};
  retention_transaction = true;
  return Status::success();
}

Status MoeProgramExecutor::Core::SessionState::end_transaction() noexcept {
  std::lock_guard lock(mutex);
  if (closed)
    return {ErrorCode::cancelled, "model execution session is closed"};
  if (active || !retention_transaction)
    return {ErrorCode::backpressure,
            "model execution session cannot end a retention transaction"};
  retention_transaction = false;
  return Status::success();
}

void MoeProgramExecutor::Core::SessionState::cancel() noexcept {
  std::shared_ptr<RequestState> step;
  std::shared_ptr<ExactDecodeState> exact;
  {
    std::lock_guard lock(mutex);
    if (closed && !active) return;
    closed = true;
    step = active_step.lock();
    exact = active_exact.lock();
    if (!step && !exact) {
      active = false;
      provider_states.clear();
      return;
    }
  }
  if (step) step->cancel(true);
  if (exact) exact->cancel();
}

MoeProgramExecutor::MoeProgramExecutor() = default;
MoeProgramExecutor::MoeProgramExecutor(std::shared_ptr<const Core> core) noexcept
    : core_(std::move(core)) {}
MoeProgramExecutor::MoeProgramExecutor(MoeProgramExecutor&&) noexcept =
    default;
MoeProgramExecutor& MoeProgramExecutor::operator=(
    MoeProgramExecutor&&) noexcept = default;
MoeProgramExecutor::~MoeProgramExecutor() = default;

bool MoeProgramExecutor::valid() const noexcept { return core_ != nullptr; }

Status MoeProgramExecutor::create(
    ModelDescriptor descriptor, BoundExecutionProvider provider,
    IModelTensorStore* tensor_store,
    MoeProgramExecutor& destination) noexcept {
  try {
    if (destination.valid())
      return {ErrorCode::invalid_argument,
              "model program executor destination is already initialized"};
    const auto descriptor_status = validate_model_descriptor(descriptor);
    if (!descriptor_status.ok()) return copy_status(descriptor_status);
    if (descriptor.schema_version < 3U ||
        provider.model_content_hash != descriptor.content_hash ||
        provider.model_schema_version != descriptor.schema_version ||
        provider.program.operations.size() !=
            descriptor.operation_program.size() ||
        provider.program.values.empty() || provider.program.inputs.empty() ||
        provider.program.outputs.empty())
      return {ErrorCode::invalid_argument,
              "callable interpreter requires a matching schema v3 plan"};

    std::map<std::uint32_t, std::shared_ptr<IOperationProvider>> providers;
    for (const auto& item : provider.providers) {
      if (!item.implementation ||
          !providers.emplace(item.registry_index, item.implementation).second)
        return {ErrorCode::invalid_argument,
                "executable provider plan is incomplete or duplicated"};
    }

    auto core = std::make_shared<Core>();
    core->descriptor = std::move(descriptor);
    core->provider = std::move(provider);
    core->prepared.reserve(core->provider.program.operations.size());
    std::map<std::string, std::shared_ptr<const ImmutableModelTensor>,
             std::less<>>
        tensor_cache;
    for (std::size_t index = 0U;
         index < core->provider.program.operations.size(); ++index) {
      const auto& source = core->descriptor.operation_program[index];
      const auto& compiled = core->provider.program.operations[index];
      if (source.logical_operation != index ||
          compiled.logical_operation != index ||
          compiled.kernel_binding >= core->provider.program.kernels.size())
        return {ErrorCode::invalid_argument,
                "compiled operation order or kernel binding is invalid"};
      const auto registry_index =
          core->provider.program.kernels[compiled.kernel_binding]
              .provider_registry_index;
      const auto implementation = providers.find(registry_index);
      if (implementation == providers.end())
        return {ErrorCode::invalid_argument,
                "compiled operation references no callable provider"};

      Core::PreparedInstruction prepared;
      prepared.provider_registry_index = registry_index;
      prepared.provider = implementation->second;
      prepared.tensors.reserve(source.tensor_bindings.size());
      for (const auto& [role, name] : source.tensor_bindings) {
        auto retained = tensor_cache.find(name);
        if (retained == tensor_cache.end()) {
          if (tensor_store == nullptr)
            return {ErrorCode::invalid_argument,
                    "operation tensor has no artifact tensor store"};
          auto resolved = tensor_store->resolve(name);
          if (!resolved.status.ok()) return copy_status(resolved.status);
          if (!resolved.tensor || resolved.tensor->name != name ||
              resolved.tensor->encoding.empty() ||
              resolved.tensor->shape.empty() ||
              !resolved.tensor->value.valid())
            return {ErrorCode::invalid_argument,
                    "artifact tensor store returned an invalid tensor"};
          retained = tensor_cache.emplace(name, std::move(resolved.tensor)).first;
        }
        prepared.tensors.push_back({role, retained->second});
      }
      auto result = prepared.provider->prepare(
          {core->descriptor, source, compiled, prepared.tensors});
      if (!result.status.ok()) return copy_status(result.status);
      if (!result.operation)
        return {ErrorCode::invalid_argument,
                "callable provider returned no prepared operation"};
      prepared.operation = std::move(result.operation);
      core->prepared.push_back(std::move(prepared));
    }
    if (core->descriptor.exact_decode_program.has_value() !=
        core->provider.program.exact_decode.has_value())
      return {ErrorCode::invalid_argument,
              "compiled exact decode program does not match its artifact"};
    if (core->descriptor.exact_decode_program) {
      const auto& source = *core->descriptor.exact_decode_program;
      const auto& compiled = *core->provider.program.exact_decode;
      if (compiled.kernel_binding >= core->provider.program.kernels.size() ||
          compiled.maximum_emitted_tokens != source.maximum_emitted_tokens)
        return {ErrorCode::invalid_argument,
                "compiled exact decode binding is invalid"};
      const auto registry_index =
          core->provider.program.kernels[compiled.kernel_binding]
              .provider_registry_index;
      const auto implementation = providers.find(registry_index);
      if (implementation == providers.end())
        return {ErrorCode::invalid_argument,
                "exact decode binding references no callable provider"};
      Core::PreparedExactDecode exact;
      exact.provider_registry_index = registry_index;
      exact.provider = implementation->second;
      exact.maximum_emitted_tokens = compiled.maximum_emitted_tokens;
      exact.tensors.reserve(source.tensor_bindings.size());
      for (const auto& [role, name] : source.tensor_bindings) {
        auto retained = tensor_cache.find(name);
        if (retained == tensor_cache.end()) {
          if (tensor_store == nullptr)
            return {ErrorCode::invalid_argument,
                    "exact decode tensor has no artifact tensor store"};
          auto resolved = tensor_store->resolve(name);
          if (!resolved.status.ok()) return copy_status(resolved.status);
          if (!resolved.tensor || resolved.tensor->name != name ||
              resolved.tensor->encoding.empty() ||
              resolved.tensor->shape.empty() ||
              !resolved.tensor->value.valid())
            return {ErrorCode::invalid_argument,
                    "artifact tensor store returned an invalid exact decode tensor"};
          retained =
              tensor_cache.emplace(name, std::move(resolved.tensor)).first;
        }
        exact.tensors.push_back({role, retained->second});
      }
      auto prepared = implementation->second->prepare_exact_decode(
          {core->descriptor, source, compiled, exact.tensors});
      if (!prepared.status.ok()) return copy_status(prepared.status);
      if (!prepared.operation)
        return {ErrorCode::invalid_argument,
                "callable provider returned no exact decode program"};
      exact.operation = std::move(prepared.operation);
      core->exact_decode = std::move(exact);
    }
    if (!core->prepared.empty()) {
      const auto registry_index =
          core->prepared.front().provider_registry_index;
      const auto implementation = core->prepared.front().provider;
      const auto common_provider = std::all_of(
          core->prepared.begin(), core->prepared.end(),
          [&](const Core::PreparedInstruction& item) {
            return item.provider_registry_index == registry_index &&
                   item.provider == implementation;
          });
      if (common_provider &&
          implementation->supports_program_sequence(core->provider.program)) {
        Core::PreparedSequence sequence;
        sequence.provider_registry_index = registry_index;
        sequence.provider = implementation;
        sequence.operations.reserve(core->prepared.size());
        for (const auto& item : core->prepared)
          sequence.operations.push_back(item.operation.get());
        core->sequence = std::move(sequence);
      }
    }
    destination = MoeProgramExecutor(std::move(core));
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("model program preparation failed: ") + error.what()};
  } catch (...) {
    return {ErrorCode::internal, "model program preparation failed"};
  }
}

StartProgramExecutionResult MoeProgramExecutor::execute(
    ProgramExecutionRequest request) const noexcept {
  auto begun = Core::begin(core_, std::move(request.context));
  if (!begun.status.ok()) return {copy_status(begun.status), {}};
  return begun.session->start(std::move(request.inputs), true);
}

BeginProgramExecutionSessionResult MoeProgramExecutor::begin_session(
    ProgramRequestContext request) const noexcept {
  auto begun = Core::begin(core_, std::move(request));
  if (!begun.status.ok()) return {copy_status(begun.status), {}};
  auto state = std::move(begun.session);
  auto session = ProgramExecutionSession::from_callbacks(
      [state](std::map<std::string, ExecutionValue, std::less<>> inputs) {
        return state->start(std::move(inputs), false);
      },
      [state] { return state->sequence_available(); },
      [state](std::map<std::string, ExecutionValue, std::less<>> inputs) {
        return state->start_sequence(std::move(inputs));
      },
      [state](std::uint32_t position) {
        return state->checkpoint_retention(position);
      },
      [state](std::uint32_t position) {
        return state->rewind_retention(position);
      },
      [state](std::uint32_t position) {
        return state->park_retention(position);
      },
      [state] { return state->restore_retention(); },
      [state](const std::filesystem::path& root, std::uint64_t generation) {
        return state->save_snapshot(root, generation);
      },
      [state](const std::filesystem::path& root, std::uint64_t generation) {
        return state->prune_snapshots(root, generation);
      },
      [state](ProgramRequestContext request) {
        return state->rebind(std::move(request));
      },
      [state] { return state->begin_transaction(); },
      [state] { return state->end_transaction(); },
      [state] { return state->exact_available(); },
      [state](std::span<const std::uint32_t> tokens,
              std::uint32_t position, bool draft) {
        return state->synchronize_exact(tokens, position, draft);
      },
      [state](std::uint32_t token, std::uint32_t position,
              std::uint32_t context) {
        return state->start_exact(token, position, context);
      },
      [state] { state->cancel(); });
  if (!session.valid()) {
    state->cancel();
    return {{ErrorCode::internal,
             "model execution session handle construction failed"},
            {}};
  }
  return {Status::success(), std::move(session)};
}

BeginProgramExecutionSessionResult
MoeProgramExecutor::begin_session_from_snapshot(
    ProgramRequestContext request, const std::filesystem::path& root,
    std::uint64_t generation, std::uint32_t next_position) const noexcept {
  auto begun = Core::begin(core_, std::move(request));
  if (!begun.status.ok()) return {copy_status(begun.status), {}};
  auto state = std::move(begun.session);
  const auto loaded = state->load_snapshot(
      root, generation, next_position);
  if (!loaded.status.ok()) {
    state->cancel();
    return {copy_status(loaded.status), {}};
  }
  auto session = ProgramExecutionSession::from_callbacks(
      [state](std::map<std::string, ExecutionValue, std::less<>> inputs) {
        return state->start(std::move(inputs), false);
      },
      [state] { return state->sequence_available(); },
      [state](std::map<std::string, ExecutionValue, std::less<>> inputs) {
        return state->start_sequence(std::move(inputs));
      },
      [state](std::uint32_t position) {
        return state->checkpoint_retention(position);
      },
      [state](std::uint32_t position) {
        return state->rewind_retention(position);
      },
      [state](std::uint32_t position) {
        return state->park_retention(position);
      },
      [state] { return state->restore_retention(); },
      [state](const std::filesystem::path& path, std::uint64_t selected) {
        return state->save_snapshot(path, selected);
      },
      [state](const std::filesystem::path& path, std::uint64_t selected) {
        return state->prune_snapshots(path, selected);
      },
      [state](ProgramRequestContext replacement) {
        return state->rebind(std::move(replacement));
      },
      [state] { return state->begin_transaction(); },
      [state] { return state->end_transaction(); },
      [state] { return state->exact_available(); },
      [state](std::span<const std::uint32_t> tokens,
              std::uint32_t position, bool draft) {
        return state->synchronize_exact(tokens, position, draft);
      },
      [state](std::uint32_t token, std::uint32_t position,
              std::uint32_t context) {
        return state->start_exact(token, position, context);
      },
      [state] { state->cancel(); });
  if (!session.valid()) {
    state->cancel();
    return {{ErrorCode::internal,
             "model execution session handle construction failed"},
            {}};
  }
  return {Status::success(), std::move(session)};
}

}  // namespace expert::runtime
