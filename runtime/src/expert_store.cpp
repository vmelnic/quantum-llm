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
    AcquireHandle handle;
    std::optional<AcquireResult> result;
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
    resolved.experts.push_back({item.key, std::move(item.result->lease)});
  }
  core_->terminal = true;
  return resolved;
}

void ExpertResolveHandle::cancel() noexcept {
  if (!core_ || core_->terminal) return;
  for (auto& item : core_->items) {
    if (!item.result) item.handle.cancel();
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
    core->items.push_back(
        {request.key, cache_.acquire(request.key, request.record), {}});
  }
  return ExpertResolveHandle(std::move(core));
}

}  // namespace expert::runtime
