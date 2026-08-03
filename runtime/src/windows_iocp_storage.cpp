#if defined(_WIN32)

#include "expert/runtime/windows_iocp_storage.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace expert::runtime {
namespace {

constexpr ULONG_PTR kShutdownKey = 1;

Status windows_error(ErrorCode code, const char* operation, DWORD error) {
  std::string message(operation);
  message += ": ";
  message += std::system_category().message(static_cast<int>(error));
  return Status(code, std::move(message));
}

bool power_of_two(std::uint64_t value) noexcept {
  return value != 0 && (value & (value - 1U)) == 0;
}

}  // namespace

struct WindowsIocpStorage::Impl final {
  struct Pending final {
    OVERLAPPED overlapped{};  // Must remain first for completion recovery.
    OperationId id{};
    HANDLE file{INVALID_HANDLE_VALUE};
    ReadRequest request;
    ReadCompletion completion;
    std::atomic<bool> cancellation_requested{false};
  };

  static_assert(offsetof(Pending, overlapped) == 0);

  explicit Impl(std::size_t worker_count) {
    if (worker_count == 0) {
      throw std::invalid_argument("IOCP worker_count must be non-zero");
    }
    port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
    if (port == nullptr) {
      throw std::system_error(static_cast<int>(GetLastError()),
                              std::system_category(),
                              "CreateIoCompletionPort");
    }
    try {
      workers.reserve(worker_count);
      for (std::size_t index = 0; index < worker_count; ++index) {
        workers.emplace_back([this] { completion_loop(); });
      }
    } catch (...) {
      stopping.store(true, std::memory_order_release);
      for (std::size_t index = 0; index < workers.size(); ++index) {
        PostQueuedCompletionStatus(port, 0, kShutdownKey, nullptr);
      }
      for (auto& worker : workers) {
        worker.join();
      }
      CloseHandle(port);
      throw;
    }
  }

  ~Impl() {
    std::vector<std::pair<HANDLE, OVERLAPPED*>> outstanding;
    {
      std::lock_guard lock(mutex);
      stopping.store(true, std::memory_order_release);
      outstanding.reserve(pending.size());
      for (auto& [id, operation] : pending) {
        (void)id;
        operation->cancellation_requested.store(true, std::memory_order_release);
        outstanding.emplace_back(operation->file, &operation->overlapped);
      }
      if (pending.empty()) {
        post_shutdown_locked();
      }
    }
    for (const auto& [file, overlapped] : outstanding) {
      CancelIoEx(file, overlapped);
    }
    for (auto& worker : workers) {
      worker.join();
    }
    CloseHandle(port);
  }

  OperationId read(ReadRequest request, ReadCompletion completion) {
    const auto id = next_id.fetch_add(1, std::memory_order_relaxed);
    if (!completion) {
      return id;
    }
    const auto finish_now = [&](Status status, std::uint64_t read_bytes = 0) {
      completion({std::move(status), request.record.stored_bytes, read_bytes});
    };

    if (stopping.load(std::memory_order_acquire)) {
      finish_now(Status(ErrorCode::cancelled, "IOCP reader is shutting down"));
      return id;
    }
    if (request.record.stored_bytes == 0 ||
        request.record.stored_bytes > request.destination.capacity ||
        request.destination.data == nullptr ||
        request.record.stored_bytes > std::numeric_limits<DWORD>::max()) {
      finish_now(Status(ErrorCode::invalid_argument,
                        "invalid IOCP read size or destination"));
      return id;
    }
    if (request.direct) {
      const auto alignment = request.record.alignment;
      const auto address =
          reinterpret_cast<std::uintptr_t>(request.destination.data);
      if (alignment < 4096 || !power_of_two(alignment) ||
          request.record.record_offset % alignment != 0 ||
          request.record.stored_bytes % alignment != 0 ||
          address % alignment != 0) {
        finish_now(Status(
            ErrorCode::invalid_argument,
            "NO_BUFFERING requires aligned offset, length, and destination"));
        return id;
      }
    }

    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
    if (request.direct) {
      flags |= FILE_FLAG_NO_BUFFERING;
    }
    const auto file = CreateFileW(
        request.record.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, flags, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      finish_now(windows_error(ErrorCode::open_failed, "CreateFileW",
                               GetLastError()));
      return id;
    }

    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size)) {
      const auto error = GetLastError();
      CloseHandle(file);
      finish_now(windows_error(ErrorCode::io_failed, "GetFileSizeEx", error));
      return id;
    }
    const auto offset = request.record.record_offset;
    const auto length = request.record.stored_bytes;
    if (size.QuadPart < 0 || offset > static_cast<std::uint64_t>(size.QuadPart) ||
        length > static_cast<std::uint64_t>(size.QuadPart) - offset) {
      CloseHandle(file);
      finish_now(Status(ErrorCode::short_read,
                        "expert record extends beyond pack EOF"));
      return id;
    }
    if (CreateIoCompletionPort(file, port, 0, 0) == nullptr) {
      const auto error = GetLastError();
      CloseHandle(file);
      finish_now(windows_error(ErrorCode::io_failed,
                               "CreateIoCompletionPort(file)", error));
      return id;
    }

    auto operation = std::make_unique<Pending>();
    operation->id = id;
    operation->file = file;
    operation->request = std::move(request);
    operation->completion = std::move(completion);
    operation->overlapped.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
    operation->overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    auto* raw = operation.get();
    {
      std::lock_guard lock(mutex);
      if (stopping.load(std::memory_order_acquire)) {
        CloseHandle(file);
        auto callback = std::move(operation->completion);
        callback({Status(ErrorCode::cancelled, "IOCP reader is shutting down"),
                  length, 0});
        return id;
      }
      pending.emplace(id, std::move(operation));
    }

    const auto started = ReadFile(file, raw->request.destination.data,
                                  static_cast<DWORD>(length), nullptr,
                                  &raw->overlapped);
    if (!started) {
      const auto error = GetLastError();
      if (error != ERROR_IO_PENDING) {
        complete_synchronous_failure(id, error);
      }
    }
    return id;
  }

  void cancel(OperationId id) noexcept {
    HANDLE file = INVALID_HANDLE_VALUE;
    OVERLAPPED* overlapped = nullptr;
    {
      std::lock_guard lock(mutex);
      const auto iterator = pending.find(id);
      if (iterator == pending.end()) {
        return;
      }
      iterator->second->cancellation_requested.store(true,
                                                     std::memory_order_release);
      file = iterator->second->file;
      overlapped = &iterator->second->overlapped;
    }
    // Completion, including ERROR_OPERATION_ABORTED, is still consumed by the
    // IOCP worker; this function never races a second callback.
    CancelIoEx(file, overlapped);
  }

  void complete_synchronous_failure(OperationId id, DWORD error) noexcept {
    std::unique_ptr<Pending> operation;
    {
      std::lock_guard lock(mutex);
      const auto iterator = pending.find(id);
      if (iterator == pending.end()) {
        return;
      }
      operation = std::move(iterator->second);
      pending.erase(iterator);
      if (stopping.load(std::memory_order_acquire) && pending.empty()) {
        post_shutdown_locked();
      }
    }
    CloseHandle(operation->file);
    try {
      operation->completion(
          {windows_error(ErrorCode::io_failed, "ReadFile", error),
           operation->request.record.stored_bytes, 0});
    } catch (...) {
    }
  }

  void completion_loop() noexcept {
    for (;;) {
      DWORD transferred = 0;
      ULONG_PTR key = 0;
      OVERLAPPED* overlapped = nullptr;
      const auto succeeded =
          GetQueuedCompletionStatus(port, &transferred, &key, &overlapped,
                                    INFINITE);
      if (overlapped == nullptr) {
        if (key == kShutdownKey) {
          return;
        }
        continue;
      }
      auto* raw = reinterpret_cast<Pending*>(overlapped);
      const auto error = succeeded ? ERROR_SUCCESS : GetLastError();
      std::unique_ptr<Pending> operation;
      {
        std::lock_guard lock(mutex);
        const auto iterator = pending.find(raw->id);
        if (iterator == pending.end()) {
          continue;
        }
        operation = std::move(iterator->second);
        pending.erase(iterator);
        if (stopping.load(std::memory_order_acquire) && pending.empty()) {
          post_shutdown_locked();
        }
      }
      CloseHandle(operation->file);

      Status status = Status::success();
      if (operation->cancellation_requested.load(std::memory_order_acquire) ||
          error == ERROR_OPERATION_ABORTED) {
        status = Status(ErrorCode::cancelled, "IOCP read cancelled");
      } else if (!succeeded && (error == ERROR_HANDLE_EOF ||
                                error == ERROR_BROKEN_PIPE)) {
        status = Status(ErrorCode::short_read, "unexpected EOF in expert pack");
      } else if (!succeeded) {
        status = windows_error(ErrorCode::io_failed,
                               "GetQueuedCompletionStatus", error);
      } else if (transferred != operation->request.record.stored_bytes) {
        status = Status(ErrorCode::short_read,
                        "IOCP completed a partial expert record");
      }
      try {
        operation->completion(
            {std::move(status), operation->request.record.stored_bytes,
             transferred});
      } catch (...) {
      }
    }
  }

  void post_shutdown_locked() noexcept {
    if (shutdown_posted) {
      return;
    }
    shutdown_posted = true;
    for (std::size_t index = 0; index < workers.size(); ++index) {
      PostQueuedCompletionStatus(port, 0, kShutdownKey, nullptr);
    }
  }

  HANDLE port{nullptr};
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::unordered_map<OperationId, std::unique_ptr<Pending>> pending;
  std::atomic<OperationId> next_id{1};
  std::atomic<bool> stopping{false};
  bool shutdown_posted{};
};

WindowsIocpStorage::WindowsIocpStorage(std::size_t worker_count)
    : impl_(std::make_unique<Impl>(worker_count)) {}

WindowsIocpStorage::~WindowsIocpStorage() = default;

OperationId WindowsIocpStorage::read(ReadRequest request,
                                     ReadCompletion completion) {
  return impl_->read(std::move(request), std::move(completion));
}

void WindowsIocpStorage::cancel(OperationId operation) noexcept {
  impl_->cancel(operation);
}

}  // namespace expert::runtime

#endif
