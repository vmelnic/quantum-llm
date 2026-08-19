#include "expert/runtime/expert_store.hpp"

#include <chrono>
#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <utility>

namespace expert::runtime {
namespace {

Status copy_status(const Status& status) {
  return {status.code(), std::string(status.message())};
}

}  // namespace

struct LocalResolveState final {
  struct Item final {
    ExpertKey key;
    ExpertResolveTarget target{ExpertResolveTarget::device};
    PayloadRecord record;
    ExpertAcquireOptions options;
    AcquireHandle handle;
    HostPreloadHandle host_handle;
    std::optional<AcquireResult> result;
    std::optional<HostExpertLease> host;
    std::optional<Status> error;
  };
  ExpertCache* cache{};
  std::vector<Item> items;
  bool terminal{};

  std::optional<ExpertResolveResult> poll();
  void cancel() noexcept;
};

struct ExpertResolveHandle::Core final {
  Poll poll;
  Cancel cancel;
  std::size_t size{};
  bool terminal{};
};

ExpertResolveHandle::ExpertResolveHandle() = default;
ExpertResolveHandle::ExpertResolveHandle(std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
ExpertResolveHandle::ExpertResolveHandle(ExpertResolveHandle&&) noexcept =
    default;
ExpertResolveHandle& ExpertResolveHandle::operator=(
    ExpertResolveHandle&&) noexcept = default;
ExpertResolveHandle::~ExpertResolveHandle() { cancel(); }

bool ExpertResolveHandle::valid() const noexcept {
  return core_ && !core_->terminal && static_cast<bool>(core_->poll);
}

std::size_t ExpertResolveHandle::size() const noexcept {
  return valid() ? core_->size : 0U;
}

std::optional<ExpertResolveResult> LocalResolveState::poll() {
  if (terminal || items.empty()) return std::nullopt;
  bool complete = true;
  for (auto& item : items) {
    if (item.error) {
      for (auto& unresolved : items) {
        if (unresolved.target == ExpertResolveTarget::device &&
            !unresolved.result)
          unresolved.handle.cancel();
        if (unresolved.target == ExpertResolveTarget::host &&
            !unresolved.host)
          unresolved.host_handle.cancel();
      }
      terminal = true;
      return ExpertResolveResult{copy_status(*item.error), {}};
    }
    if (item.target == ExpertResolveTarget::host_ready) continue;
    if (item.target == ExpertResolveTarget::host) {
      if (item.host) continue;
      if (item.host_handle.wait_for(std::chrono::milliseconds(0)) !=
          std::future_status::ready) {
        complete = false;
        continue;
      }
      auto retained = item.host_handle.get();
      if (!retained.status.ok() || !retained.retained) {
        item.error.emplace(
            retained.status.ok()
                ? Status(ErrorCode::internal,
                         "host expert resolve completed without retention")
                : copy_status(retained.status));
        complete = false;
        continue;
      }
      item.host = cache->try_acquire_host(
          item.key, item.record, item.options.record_access,
          item.options.priority);
      if (!item.host) {
        item.error.emplace(ErrorCode::internal,
                           "host expert resolve lost retained ownership");
        complete = false;
      }
      continue;
    }
    if (item.result) continue;
    if (item.handle.wait_for(std::chrono::milliseconds(0)) !=
        std::future_status::ready) {
      complete = false;
      continue;
    }
    item.result.emplace(item.handle.get());
    if (!item.result->status.ok() || !item.result->lease) {
      const auto status = item.result->status.ok()
                              ? Status(ErrorCode::internal,
                                       "expert store member returned no lease")
                              : copy_status(item.result->status);
      for (auto& unresolved : items) {
        if (!unresolved.result) unresolved.handle.cancel();
      }
      terminal = true;
      return ExpertResolveResult{status, {}};
    }
  }
  if (!complete) return std::nullopt;

  ExpertResolveResult resolved;
  resolved.status = Status::success();
  resolved.experts.reserve(items.size());
  for (auto& item : items) {
    if (item.target == ExpertResolveTarget::host_ready ||
        item.target == ExpertResolveTarget::host) {
      resolved.experts.push_back({item.key, ExpertPlacementKind::host, {},
                                  std::move(*item.host), {}});
    } else {
      resolved.experts.push_back({item.key, ExpertPlacementKind::device,
                                  std::move(item.result->lease), {}, {}});
    }
  }
  terminal = true;
  return resolved;
}

void LocalResolveState::cancel() noexcept {
  if (terminal) return;
  for (auto& item : items) {
    if (item.target == ExpertResolveTarget::device && !item.result)
      item.handle.cancel();
    if (item.target == ExpertResolveTarget::host && !item.host)
      item.host_handle.cancel();
  }
  terminal = true;
}

std::optional<ExpertResolveResult> ExpertResolveHandle::poll() {
  if (!valid()) return std::nullopt;
  auto result = core_->poll();
  if (result) core_->terminal = true;
  return result;
}

void ExpertResolveHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  if (core_->cancel) core_->cancel();
  core_->terminal = true;
}

ExpertResolveHandle ExpertResolveHandle::from_callbacks(
    Poll poll, Cancel cancel, std::size_t size) {
  if (!poll || !cancel || size == 0U) return {};
  auto core = std::make_unique<Core>();
  core->poll = std::move(poll);
  core->cancel = std::move(cancel);
  core->size = size;
  return ExpertResolveHandle(std::move(core));
}

LocalExpertStore::LocalExpertStore(ExpertCache& cache) noexcept : cache_(cache) {}

ExpertResolveHandle LocalExpertStore::resolve(
    std::span<const ExpertResolveRequest> requests) {
  if (requests.empty()) return {};
  std::set<ExpertKey> unique;
  for (const auto& request : requests) {
    if (!unique.insert(request.key).second) return {};
  }
  auto state = std::make_shared<LocalResolveState>();
  state->cache = &cache_;
  state->items.reserve(requests.size());
  for (const auto& request : requests) {
    if (request.target == ExpertResolveTarget::remote)
      return {};
    LocalResolveState::Item item;
    item.key = request.key;
    item.record = request.record;
    item.options = request.options;
    item.target = request.target == ExpertResolveTarget::automatic
                      ? ExpertResolveTarget::device
                      : request.target;
    if (request.target == ExpertResolveTarget::host_ready) {
      item.host = cache_.try_acquire_host(
          request.key, request.record, request.options.record_access,
          request.options.priority);
      if (!item.host) {
        item.error.emplace(ErrorCode::backpressure,
                           "requested expert is not ready in host memory");
      }
    } else if (request.target == ExpertResolveTarget::host) {
      item.host_handle = cache_.preload_host(
          request.key, request.record,
          HostPreloadOptions{request.options.priority, false});
      if (!item.host_handle.valid()) {
        item.error.emplace(ErrorCode::backpressure,
                           "host expert-page resolve was rejected");
      }
    } else {
      item.handle = cache_.acquire(request.key, request.record,
                                   request.options);
    }
    state->items.push_back(std::move(item));
  }
  return ExpertResolveHandle::from_callbacks(
      [state] { return state->poll(); }, [state] { state->cancel(); },
      requests.size());
}

Status ActiveExpertOwnerDirectory::add(RemoteExpertOwnerRange range) noexcept {
  try {
    if (range.namespace_id == 0U || range.layer_count == 0U ||
        range.expert_count == 0U || !range.executor ||
        !range.executor->remote() || range.executor->owner().empty() ||
        range.first_layer >
            std::numeric_limits<std::uint32_t>::max() - range.layer_count ||
        range.first_expert >
            std::numeric_limits<std::uint32_t>::max() - range.expert_count)
      return {ErrorCode::invalid_argument,
              "remote expert owner range is invalid"};
    const auto layer_end = range.first_layer + range.layer_count;
    const auto expert_end = range.first_expert + range.expert_count;
    for (const auto& existing : ranges_) {
      if (existing.namespace_id != range.namespace_id) continue;
      const auto existing_layer_end =
          existing.first_layer + existing.layer_count;
      const auto existing_expert_end =
          existing.first_expert + existing.expert_count;
      const bool layer_overlap =
          range.first_layer < existing_layer_end &&
          existing.first_layer < layer_end;
      const bool expert_overlap =
          range.first_expert < existing_expert_end &&
          existing.first_expert < expert_end;
      if (layer_overlap && expert_overlap)
        return {ErrorCode::invalid_argument,
                "remote expert owner ranges overlap"};
    }
    ranges_.push_back(std::move(range));
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("remote expert owner registration failed: ") +
                error.what()};
  }
}

std::shared_ptr<IActiveExpertExecutor> ActiveExpertOwnerDirectory::find(
    const ExpertKey& key) const noexcept {
  const auto found = std::find_if(ranges_.begin(), ranges_.end(),
                                  [&](const auto& range) {
    return key.model_id == range.namespace_id &&
           key.layer >= range.first_layer &&
           key.layer - range.first_layer < range.layer_count &&
           key.expert >= range.first_expert &&
           key.expert - range.first_expert < range.expert_count;
  });
  return found == ranges_.end() ? nullptr : found->executor;
}

namespace {

bool valid_component_contract(
    const ActiveExpertComponentContract& contract) noexcept {
  return std::any_of(contract.model_content_hash.begin(),
                     contract.model_content_hash.end(),
                     [](std::byte value) { return value != std::byte{0}; }) &&
         contract.namespace_id != 0U && contract.layer_count != 0U &&
         contract.experts_per_layer != 0U && contract.encoding_abi != 0U &&
         !contract.execution_capability.empty() &&
         contract.execution_abi != 0U && contract.source_abi != 0U;
}

class BoundRemoteExpertLease final : public IRemoteExpertLease {
 public:
  BoundRemoteExpertLease(ActiveExpertIdentity identity,
                         std::shared_ptr<IActiveExpertExecutor> executor)
      : identity_(std::move(identity)), executor_(std::move(executor)) {}

  [[nodiscard]] std::string_view owner() const noexcept override {
    return executor_ ? executor_->owner() : std::string_view{};
  }
  [[nodiscard]] const ActiveExpertIdentity& identity()
      const noexcept override {
    return identity_;
  }
  [[nodiscard]] ActiveExpertExecutionHandle execute(
      ActiveExpertInvocation invocation) override {
    if (!executor_) return {};
    return executor_->execute(
        ActiveExpertExecutionRequest{identity_, std::move(invocation)});
  }

 private:
  ActiveExpertIdentity identity_;
  std::shared_ptr<IActiveExpertExecutor> executor_;
};

struct ImmediateRemoteResolveState final {
  ExpertResolveResult result;
  bool terminal{};
};

struct PlacementResolveState final {
  struct Batch final {
    std::vector<std::size_t> indices;
    ExpertResolveHandle handle;
    bool complete{};
  };
  std::vector<ExpertKey> expected;
  std::vector<std::optional<ResolvedExpert>> resolved;
  Batch local;
  Batch remote;
  bool terminal{};

  std::optional<ExpertResolveResult> poll() {
    if (terminal) return std::nullopt;
    const auto poll_batch = [&](Batch& batch) -> std::optional<Status> {
      if (batch.complete) return std::nullopt;
      auto result = batch.handle.poll();
      if (!result) return std::nullopt;
      batch.complete = true;
      if (!result->status.ok()) return copy_status(result->status);
      if (result->experts.size() != batch.indices.size())
        return Status(ErrorCode::internal,
                      "placement store batch cardinality mismatch");
      for (std::size_t index = 0U; index < batch.indices.size(); ++index) {
        const auto destination = batch.indices[index];
        if (!result->experts[index] ||
            result->experts[index].key != expected[destination])
          return Status(ErrorCode::internal,
                        "placement store batch identity mismatch");
        resolved[destination] = std::move(result->experts[index]);
      }
      return std::nullopt;
    };
    if (auto error = poll_batch(local)) {
      remote.handle.cancel();
      terminal = true;
      return ExpertResolveResult{std::move(*error), {}};
    }
    if (auto error = poll_batch(remote)) {
      local.handle.cancel();
      terminal = true;
      return ExpertResolveResult{std::move(*error), {}};
    }
    if (!local.complete || !remote.complete) return std::nullopt;
    ExpertResolveResult result;
    result.status = Status::success();
    result.experts.reserve(resolved.size());
    for (auto& item : resolved) {
      if (!item) {
        terminal = true;
        return ExpertResolveResult{
            {ErrorCode::internal,
             "placement store completed with an unresolved member"},
            {}};
      }
      result.experts.push_back(std::move(*item));
    }
    terminal = true;
    return result;
  }

  void cancel() noexcept {
    if (terminal) return;
    local.handle.cancel();
    remote.handle.cancel();
    terminal = true;
  }
};

}  // namespace

RemoteExpertStore::RemoteExpertStore(ActiveExpertComponentContract contract,
                                     ActiveExpertOwnerDirectory owners)
    : contract_(std::move(contract)), owners_(std::move(owners)) {}

bool RemoteExpertStore::owns(const ExpertKey& key) const noexcept {
  return valid_component_contract(contract_) &&
         key.model_id == contract_.namespace_id &&
         key.layer < contract_.layer_count &&
         key.expert < contract_.experts_per_layer &&
         key.encoding_abi == contract_.encoding_abi && owners_.owns(key);
}

ExpertResolveHandle RemoteExpertStore::resolve(
    std::span<const ExpertResolveRequest> requests) {
  if (requests.empty() || !valid_component_contract(contract_)) return {};
  std::set<ExpertKey> unique;
  auto state = std::make_shared<ImmediateRemoteResolveState>();
  state->result.status = Status::success();
  state->result.experts.reserve(requests.size());
  for (const auto& request : requests) {
    if ((request.target != ExpertResolveTarget::remote &&
         request.target != ExpertResolveTarget::automatic) ||
        !unique.insert(request.key).second || !owns(request.key) ||
        request.record.source_abi != contract_.source_abi) {
      return {};
    }
    auto executor = owners_.find(request.key);
    if (!executor) return {};
    ActiveExpertIdentity identity;
    identity.model_content_hash = contract_.model_content_hash;
    identity.key = request.key;
    identity.capability = contract_.execution_capability;
    identity.execution_abi = contract_.execution_abi;
    identity.source_abi = contract_.source_abi;
    state->result.experts.push_back(
        {request.key, ExpertPlacementKind::remote, {}, {},
         std::make_shared<BoundRemoteExpertLease>(std::move(identity),
                                                   std::move(executor))});
  }
  return ExpertResolveHandle::from_callbacks(
      [state]() -> std::optional<ExpertResolveResult> {
        if (state->terminal) return std::nullopt;
        state->terminal = true;
        return std::move(state->result);
      },
      [state] { state->terminal = true; }, requests.size());
}

PlacementExpertStore::PlacementExpertStore(IExpertStore& local,
                                           RemoteExpertStore& remote) noexcept
    : local_(local), remote_(remote) {}

ExpertResolveHandle PlacementExpertStore::resolve(
    std::span<const ExpertResolveRequest> requests) {
  if (requests.empty()) return {};
  std::set<ExpertKey> unique;
  std::vector<ExpertResolveRequest> local_requests;
  std::vector<ExpertResolveRequest> remote_requests;
  auto state = std::make_shared<PlacementResolveState>();
  state->expected.reserve(requests.size());
  state->resolved.resize(requests.size());
  for (std::size_t index = 0U; index < requests.size(); ++index) {
    auto request = requests[index];
    if (!unique.insert(request.key).second) return {};
    const bool use_remote =
        request.target == ExpertResolveTarget::remote ||
        (request.target == ExpertResolveTarget::automatic &&
         remote_.owns(request.key));
    if (use_remote) {
      if (!remote_.owns(request.key)) return {};
      request.target = ExpertResolveTarget::remote;
      state->remote.indices.push_back(index);
      remote_requests.push_back(std::move(request));
    } else {
      if (request.target == ExpertResolveTarget::automatic)
        request.target = ExpertResolveTarget::device;
      state->local.indices.push_back(index);
      local_requests.push_back(std::move(request));
    }
    state->expected.push_back(requests[index].key);
  }
  if (!local_requests.empty()) {
    state->local.handle = local_.resolve(local_requests);
    if (!state->local.handle.valid()) return {};
  } else {
    state->local.complete = true;
  }
  if (!remote_requests.empty()) {
    state->remote.handle = remote_.resolve(remote_requests);
    if (!state->remote.handle.valid()) {
      state->local.handle.cancel();
      return {};
    }
  } else {
    state->remote.complete = true;
  }
  return ExpertResolveHandle::from_callbacks(
      [state] { return state->poll(); }, [state] { state->cancel(); },
      requests.size());
}

}  // namespace expert::runtime
