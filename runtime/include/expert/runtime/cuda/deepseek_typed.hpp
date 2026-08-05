#pragma once

#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace expert::runtime::cuda {

enum class DeepSeekDtype : std::uint8_t { bf16, f32, i64 };

class DeepSeekTypedTensor final {
 public:
  DeepSeekTypedTensor(void* data, std::uint64_t bytes,
                      DeepSeekDtype dtype) noexcept;
  ~DeepSeekTypedTensor();
  DeepSeekTypedTensor(const DeepSeekTypedTensor&) = delete;
  DeepSeekTypedTensor& operator=(const DeepSeekTypedTensor&) = delete;

  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] DeepSeekDtype dtype() const noexcept { return dtype_; }
  [[nodiscard]] Status upload(std::uint64_t offset,
                              std::span<const std::byte> source) noexcept;

 private:
  void* data_{};
  std::uint64_t bytes_{};
  DeepSeekDtype dtype_{};
};

struct DeepSeekTypedAllocationResult final {
  Status status;
  std::shared_ptr<DeepSeekTypedTensor> tensor;
};

[[nodiscard]] DeepSeekTypedAllocationResult allocate_deepseek_typed_tensor(
    std::uint64_t bytes, DeepSeekDtype dtype) noexcept;

struct DeepSeekTypedSpec final {
  std::string name;
  PayloadRecord record;
  DeepSeekDtype dtype{};
};

// Transactionally loads immutable model state through a fixed staging slot.
// Individual tensors may be arbitrarily larger than that slot; SHA-256 and H2D
// upload are both incremental. Source dtype is preserved on device.
class DeepSeekTypedSet final {
 public:
  DeepSeekTypedSet() = default;
  DeepSeekTypedSet(const DeepSeekTypedSet&) = delete;
  DeepSeekTypedSet& operator=(const DeepSeekTypedSet&) = delete;
  DeepSeekTypedSet(DeepSeekTypedSet&&) noexcept = default;
  DeepSeekTypedSet& operator=(DeepSeekTypedSet&&) noexcept = default;

  [[nodiscard]] static Status load(IAsyncStorage& storage,
                                   FixedBufferPool& buffers,
                                   std::span<const DeepSeekTypedSpec> specs,
                                   DeepSeekTypedSet& destination);
  [[nodiscard]] const DeepSeekTypedTensor* find(
      std::string_view name) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }
  void clear() noexcept;

 private:
  struct Entry final {
    std::string name;
    std::shared_ptr<DeepSeekTypedTensor> tensor;
  };
  std::vector<Entry> entries_;
  std::uint64_t bytes_{};
};

}  // namespace expert::runtime::cuda
