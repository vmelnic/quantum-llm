#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <memory>

namespace expert::runtime {

#if defined(_WIN32)

// Native asynchronous pack reader. A fixed IOCP worker pool owns every
// OVERLAPPED until its completion packet is consumed.
class WindowsIocpStorage final : public IAsyncStorage {
 public:
  explicit WindowsIocpStorage(std::size_t worker_count = 1);
  ~WindowsIocpStorage() override;

  WindowsIocpStorage(const WindowsIocpStorage&) = delete;
  WindowsIocpStorage& operator=(const WindowsIocpStorage&) = delete;

  OperationId read(ReadRequest request, ReadCompletion completion) override;
  void cancel(OperationId operation) noexcept override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif

}  // namespace expert::runtime

