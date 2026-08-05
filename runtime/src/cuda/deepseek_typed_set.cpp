#include "expert/runtime/cuda/deepseek_typed.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <future>
#include <set>
#include <utility>

namespace expert::runtime::cuda {

Status DeepSeekTypedSet::load(IAsyncStorage& storage, FixedBufferPool& buffers,
                              std::span<const DeepSeekTypedSpec> specs,
                              DeepSeekTypedSet& destination) {
  if (specs.empty() || destination.size() != 0U || buffers.slot_bytes() == 0U)
    return {ErrorCode::invalid_argument,
            "typed set load requires specs, staging and empty destination"};
  std::set<std::string> names;
  for (const auto& spec : specs) {
    const auto element_bytes = spec.dtype == DeepSeekDtype::bf16 ? 2U :
                               spec.dtype == DeepSeekDtype::f32 ? 4U : 8U;
    if (spec.name.empty() || !names.insert(spec.name).second ||
        spec.record.stored_bytes == 0U ||
        spec.record.stored_bytes % element_bytes != 0U ||
        spec.record.extents.size() != 1U ||
        spec.record.extents.front().destination_offset != 0U ||
        spec.record.extents.front().bytes != spec.record.stored_bytes) {
      return {ErrorCode::invalid_argument,
              "typed set contains duplicate or invalid tensor geometry"};
    }
  }

  DeepSeekTypedSet candidate;
  candidate.entries_.reserve(specs.size());
  for (const auto& spec : specs) {
    auto allocation =
        allocate_deepseek_typed_tensor(spec.record.stored_bytes, spec.dtype);
    if (!allocation.status.ok()) return allocation.status;
    Sha256 hasher;
    std::uint64_t cursor = 0U;
    const auto& source = spec.record.extents.front();
    while (cursor < spec.record.stored_bytes) {
      const auto bytes = std::min<std::uint64_t>(
          buffers.slot_bytes(), spec.record.stored_bytes - cursor);
      auto lease = buffers.try_acquire(static_cast<std::size_t>(bytes));
      if (!lease)
        return {ErrorCode::backpressure,
                "typed set could not acquire its bounded staging slot"};
      PayloadRecord chunk;
      chunk.extents.push_back({source.path, source.source_offset + cursor, 0U,
                               bytes});
      chunk.stored_bytes = bytes;
      std::promise<ReadResult> promise;
      auto future = promise.get_future();
      static_cast<void>(storage.read(
          {chunk, lease->buffer(), true},
          [&promise](ReadResult result) {
            promise.set_value(std::move(result));
          }));
      const auto read = future.get();
      if (!read.status.ok()) return read.status;
      if (read.read_bytes != bytes)
        return {ErrorCode::short_read,
                "typed set storage returned an incomplete chunk"};
      const auto payload = std::span<const std::byte>(
          lease->buffer().data, static_cast<std::size_t>(bytes));
      hasher.update(payload);
      const auto upload = allocation.tensor->upload(cursor, payload);
      if (!upload.ok()) return upload;
      cursor += bytes;
    }
    if (!constant_time_equal(hasher.finalize(), spec.record.payload_sha256))
      return {ErrorCode::checksum_mismatch,
              "typed set tensor SHA-256 mismatch"};
    candidate.bytes_ += allocation.tensor->bytes();
    candidate.entries_.push_back({spec.name, std::move(allocation.tensor)});
  }
  destination = std::move(candidate);
  return Status::success();
}

const DeepSeekTypedTensor* DeepSeekTypedSet::find(
    std::string_view name) const noexcept {
  for (const auto& entry : entries_)
    if (entry.name == name) return entry.tensor.get();
  return nullptr;
}

void DeepSeekTypedSet::clear() noexcept {
  entries_.clear();
  bytes_ = 0U;
}

}  // namespace expert::runtime::cuda
