#pragma once

#include "expert/runtime/storage.hpp"

#include <memory>

namespace expert::runtime {

// Turns a logical multi-extent read into bounded asynchronous reads on an
// existing storage backend. SafeTensor payload offsets need not satisfy
// unbuffered-I/O alignment, so children use buffered overlapped reads while the
// final compact payload still lands in one fixed staging buffer.
class ExtentGatherStorage final : public IAsyncStorage {
 public:
  explicit ExtentGatherStorage(std::shared_ptr<IAsyncStorage> backing);
  ~ExtentGatherStorage() override;
  ExtentGatherStorage(const ExtentGatherStorage&) = delete;
  ExtentGatherStorage& operator=(const ExtentGatherStorage&) = delete;

  OperationId read(ReadRequest request, ReadCompletion completion) override;
  void cancel(OperationId operation) noexcept override;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace expert::runtime
