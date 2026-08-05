#include "expert/runtime/expert_store.hpp"

#include <chrono>
#include <set>
#include <string>
#include <utility>

namespace expert::runtime {
namespace {

Status copy_status(const Status& status) {
  return {status.code(), std::string(status.message())};
}

}  // namespace

struct ExpertResolveHandle::Core final {
  struct Item final {
    ExpertKey key;
    ExpertResolveTarget target{ExpertResolveTarget::device};
    AcquireHandle handle;
    std::optional<AcquireResult> result;
    std::optional<HostExpertLease> host;
    std::optional<Status> error;
  };
  std::vector<Item> items;
  bool terminal{};
};

ExpertResolveHandle::ExpertResolveHandle(std::unique_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
ExpertResolveHandle::ExpertResolveHandle(ExpertResolveHandle&&) noexcept =
    default;
ExpertResolveHandle& ExpertResolveHandle::operator=(
    ExpertResolveHandle&&) noexcept = default;
ExpertResolveHandle::~ExpertResolveHandle() { cancel(); }

bool ExpertResolveHandle::valid() const noexcept {
  return core_ && !core_->terminal && !core_->items.empty();
}

std::size_t ExpertResolveHandle::size() const noexcept {
  return core_ ? core_->items.size() : 0U;
}

std::optional<ExpertResolveResult> ExpertResolveHandle::poll() {
  if (!valid()) return std::nullopt;
  bool complete = true;
  for (auto& item : core_->items) {
    if (item.error) {
      for (auto& unresolved : core_->items) {
        if (unresolved.target == ExpertResolveTarget::device &&
            !unresolved.result)
          unresolved.handle.cancel();
      }
      core_->terminal = true;
      return ExpertResolveResult{copy_status(*item.error), {}};
    }
    if (item.target == ExpertResolveTarget::host_ready) continue;
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
      for (auto& unresolved : core_->items) {
        if (!unresolved.result) unresolved.handle.cancel();
      }
      core_->terminal = true;
      return ExpertResolveResult{status, {}};
    }
  }
  if (!complete) return std::nullopt;

  ExpertResolveResult resolved;
  resolved.status = Status::success();
  resolved.experts.reserve(core_->items.size());
  for (auto& item : core_->items) {
    if (item.target == ExpertResolveTarget::host_ready) {
      resolved.experts.push_back({item.key, ExpertPlacementKind::host, {},
                                  std::move(*item.host)});
    } else {
      resolved.experts.push_back({item.key, ExpertPlacementKind::device,
                                  std::move(item.result->lease), {}});
    }
  }
  core_->terminal = true;
  return resolved;
}

void ExpertResolveHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  for (auto& item : core_->items) {
    if (item.target == ExpertResolveTarget::device && !item.result)
      item.handle.cancel();
  }
  core_->terminal = true;
}

LocalExpertStore::LocalExpertStore(ExpertCache& cache) noexcept : cache_(cache) {}

ExpertResolveHandle LocalExpertStore::resolve(
    std::span<const ExpertResolveRequest> requests) {
  if (requests.empty()) return {};
  std::set<ExpertKey> unique;
  for (const auto& request : requests) {
    if (!unique.insert(request.key).second) return {};
  }
  auto core = std::make_unique<ExpertResolveHandle::Core>();
  core->items.reserve(requests.size());
  for (const auto& request : requests) {
    ExpertResolveHandle::Core::Item item;
    item.key = request.key;
    item.target = request.target;
    if (request.target == ExpertResolveTarget::host_ready) {
      item.host = cache_.try_acquire_host(request.key, request.record);
      if (!item.host) {
        item.error.emplace(ErrorCode::backpressure,
                           "requested expert is not ready in host memory");
      }
    } else {
      item.handle = cache_.acquire(request.key, request.record);
    }
    core->items.push_back(std::move(item));
  }
  return ExpertResolveHandle(std::move(core));
}

}  // namespace expert::runtime
