#include "expert/runtime/gather_storage.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace expert::runtime {
namespace {

Status validate_gather(const ReadRequest& request) {
  if (request.record.extents.empty()) {
    return {ErrorCode::invalid_argument,
            "extent gather requires a multi-extent payload"};
  }
  if (request.destination.data == nullptr || request.record.stored_bytes == 0U ||
      request.record.stored_bytes > request.destination.capacity) {
    return {ErrorCode::invalid_argument, "invalid extent gather destination"};
  }
  auto extents = request.record.extents;
  std::sort(extents.begin(), extents.end(),
            [](const PayloadExtent& left, const PayloadExtent& right) {
              return left.destination_offset < right.destination_offset;
            });
  std::uint64_t cursor = 0U;
  for (const auto& extent : extents) {
    if (extent.path.empty() || extent.bytes == 0U ||
        extent.destination_offset != cursor ||
        extent.bytes > request.record.stored_bytes - cursor ||
        extent.source_offset >
            std::numeric_limits<std::uint64_t>::max() - extent.bytes) {
      return {ErrorCode::invalid_argument,
              "payload extents must exactly and safely cover the destination"};
    }
    cursor += extent.bytes;
  }
  if (cursor != request.record.stored_bytes) {
    return {ErrorCode::invalid_argument,
            "payload extents do not cover stored_bytes"};
  }
  return Status::success();
}

}  // namespace

struct ExtentGatherStorage::Impl final
    : public std::enable_shared_from_this<ExtentGatherStorage::Impl> {
  struct Pending final {
    OperationId id{};
    ReadRequest request;
    ReadCompletion completion;
    std::mutex mutex;
    std::vector<OperationId> children;
    std::size_t remaining{};
    std::uint64_t read_bytes{};
    Status status;
    bool launching{true};
    bool cancelled{};
    bool finished{};
  };

  explicit Impl(std::shared_ptr<IAsyncStorage> child)
      : backing(std::move(child)) {
    if (!backing) throw std::invalid_argument("gather storage requires backing");
  }

  OperationId read(ReadRequest request, ReadCompletion completion) {
    const auto id = next_id.fetch_add(1U, std::memory_order_relaxed);
    if (!completion) return id;
    const auto validation = validate_gather(request);
    if (!validation.ok()) {
      completion({validation, request.record.stored_bytes, 0U});
      return id;
    }
    auto pending = std::make_shared<Pending>();
    pending->id = id;
    pending->request = std::move(request);
    pending->completion = std::move(completion);
    pending->remaining = pending->request.record.extents.size();
    {
      std::lock_guard lock(mutex);
      operations.emplace(id, pending);
    }
    auto self = shared_from_this();
    for (const auto& extent : pending->request.record.extents) {
      PayloadRecord child_record = pending->request.record;
      child_record.path = extent.path;
      child_record.extents.clear();
      child_record.record_offset = extent.source_offset;
      child_record.stored_bytes = extent.bytes;
      child_record.device_bytes = 0U;
      child_record.header_bytes = 0U;
      child_record.alignment = 1U;
      ReadRequest child_request{
          std::move(child_record),
          {pending->request.destination.data + extent.destination_offset,
           static_cast<std::size_t>(extent.bytes)},
          false};
      const auto child = backing->read(
          std::move(child_request),
          [self, pending](ReadResult result) {
            bool complete = false;
            {
              std::lock_guard lock(pending->mutex);
              if (!result.status.ok() && pending->status.ok()) {
                pending->status = result.status;
              }
              pending->read_bytes += result.read_bytes;
              if (pending->remaining != 0U) --pending->remaining;
              complete = pending->remaining == 0U && !pending->launching;
            }
            if (complete) self->finish(pending);
          });
      bool cancel_child = false;
      {
        std::lock_guard lock(pending->mutex);
        pending->children.push_back(child);
        cancel_child = pending->cancelled;
      }
      if (cancel_child) backing->cancel(child);
    }
    bool complete = false;
    {
      std::lock_guard lock(pending->mutex);
      pending->launching = false;
      complete = pending->remaining == 0U;
    }
    if (complete) finish(pending);
    return id;
  }

  void cancel(OperationId id) noexcept {
    std::shared_ptr<Pending> pending;
    {
      std::lock_guard lock(mutex);
      const auto found = operations.find(id);
      if (found == operations.end()) return;
      pending = found->second;
    }
    std::vector<OperationId> children;
    {
      std::lock_guard lock(pending->mutex);
      pending->cancelled = true;
      if (pending->status.ok()) {
        pending->status = {ErrorCode::cancelled, "extent gather cancelled"};
      }
      children = pending->children;
    }
    for (const auto child : children) backing->cancel(child);
  }

  void finish(const std::shared_ptr<Pending>& pending) noexcept {
    ReadCompletion callback;
    ReadResult result;
    {
      std::lock_guard lock(pending->mutex);
      if (pending->finished) return;
      pending->finished = true;
      callback = std::move(pending->completion);
      result = {pending->status, pending->request.record.stored_bytes,
                pending->read_bytes};
    }
    {
      std::lock_guard lock(mutex);
      operations.erase(pending->id);
    }
    try {
      callback(std::move(result));
    } catch (...) {
    }
  }

  void shutdown() noexcept {
    std::vector<OperationId> ids;
    {
      std::lock_guard lock(mutex);
      ids.reserve(operations.size());
      for (const auto& [id, pending] : operations) {
        (void)pending;
        ids.push_back(id);
      }
    }
    for (const auto id : ids) cancel(id);
  }

  std::shared_ptr<IAsyncStorage> backing;
  std::atomic<OperationId> next_id{1U};
  std::mutex mutex;
  std::map<OperationId, std::shared_ptr<Pending>> operations;
};

ExtentGatherStorage::ExtentGatherStorage(std::shared_ptr<IAsyncStorage> backing)
    : impl_(std::make_shared<Impl>(std::move(backing))) {}

ExtentGatherStorage::~ExtentGatherStorage() {
  if (impl_) impl_->shutdown();
}

OperationId ExtentGatherStorage::read(ReadRequest request,
                                      ReadCompletion completion) {
  return impl_->read(std::move(request), std::move(completion));
}

void ExtentGatherStorage::cancel(OperationId operation) noexcept {
  impl_->cancel(operation);
}

}  // namespace expert::runtime
