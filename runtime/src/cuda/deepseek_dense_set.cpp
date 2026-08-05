#include "expert/runtime/cuda/deepseek_dense.hpp"

#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/sha256.hpp"

#include <future>
#include <set>
#include <utility>

namespace expert::runtime::cuda {

Status DeepSeekDenseSet::load(IAsyncStorage& storage, FixedBufferPool& buffers,
                              std::span<const DeepSeekDenseSpec> specs,
                              DeepSeekDenseSet& destination) {
  if (specs.empty() || destination.size() != 0U) {
    return {ErrorCode::invalid_argument,
            "dense set load requires non-empty specs and destination"};
  }
  std::set<std::string> names;
  for (const auto& spec : specs) {
    const auto weight_bytes =
        static_cast<std::uint64_t>(spec.rows) * spec.columns;
    const auto scale_bytes = static_cast<std::uint64_t>(spec.rows / 128U) *
                             (spec.columns / 128U);
    if (spec.name.empty() || !names.insert(spec.name).second ||
        spec.rows == 0U || spec.columns == 0U || spec.rows % 128U != 0U ||
        spec.columns % 128U != 0U ||
        spec.record.stored_bytes != weight_bytes + scale_bytes ||
        spec.record.stored_bytes > buffers.slot_bytes()) {
      return {ErrorCode::invalid_argument,
              "dense set contains duplicate or invalid matrix geometry"};
    }
  }

  DeepSeekDenseSet candidate;
  candidate.entries_.reserve(specs.size());
  for (const auto& spec : specs) {
    auto lease = buffers.try_acquire(spec.record.stored_bytes);
    if (!lease) {
      return {ErrorCode::backpressure,
              "dense set could not acquire its bounded staging slot"};
    }
    std::promise<ReadResult> promise;
    auto future = promise.get_future();
    static_cast<void>(storage.read(
        {spec.record, lease->buffer(), true},
        [&promise](ReadResult result) {
          promise.set_value(std::move(result));
        }));
    const auto read = future.get();
    if (!read.status.ok()) return read.status;
    if (read.read_bytes != spec.record.stored_bytes) {
      return {ErrorCode::short_read,
              "dense set storage returned an incomplete matrix"};
    }
    const auto complete = std::span<const std::byte>(
        lease->buffer().data,
        static_cast<std::size_t>(spec.record.stored_bytes));
    if (!constant_time_equal(sha256(complete), spec.record.payload_sha256)) {
      return {ErrorCode::checksum_mismatch,
              "dense set matrix SHA-256 mismatch"};
    }
    const auto weight_bytes =
        static_cast<std::size_t>(spec.rows) * spec.columns;
    const auto scale_bytes = static_cast<std::size_t>(spec.rows / 128U) *
                             (spec.columns / 128U);
    auto admitted = admit_deepseek_dense_matrix(
        complete.first(weight_bytes),
        complete.subspan(weight_bytes, scale_bytes), spec.rows, spec.columns);
    if (!admitted.status.ok()) return admitted.status;
    candidate.bytes_ += admitted.matrix->bytes();
    candidate.entries_.push_back({spec.name, std::move(admitted.matrix)});
  }
  destination = std::move(candidate);
  return Status::success();
}

const DeepSeekDenseMatrix* DeepSeekDenseSet::find(
    std::string_view name) const noexcept {
  for (const auto& entry : entries_) {
    if (entry.name == name) return entry.matrix.get();
  }
  return nullptr;
}

void DeepSeekDenseSet::clear() noexcept {
  entries_.clear();
  bytes_ = 0U;
}

}  // namespace expert::runtime::cuda
