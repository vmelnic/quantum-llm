#include "expert/runtime/adaptive_placement.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/cpu/expert_executor.hpp"
#include "expert/runtime/scheduler.hpp"
#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace er = expert::runtime;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename T>
void write_le(std::byte* output, T value) {
  using Unsigned = std::make_unsigned_t<T>;
  const auto converted = static_cast<Unsigned>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output[index] = static_cast<std::byte>(
        (converted >> static_cast<unsigned>(index * 8U)) & 0xffU);
  }
}

struct FixtureRecord final {
  er::ExpertKey key;
  er::PayloadRecord record;
  std::vector<std::byte> bytes;
};

FixtureRecord make_record(std::uint32_t expert_id,
                          std::uint64_t file_offset = 0,
                          std::uint32_t layer = 3) {
  FixtureRecord result;
  result.key = {0x0123456789abcdefULL, layer, expert_id,
                er::kExpertQuantAbiInt8PerRow};
  result.bytes.resize(er::kExpertPackAlignment);
  for (std::size_t index = er::kExpertHeaderBytes; index < result.bytes.size();
       ++index) {
    result.bytes[index] = static_cast<std::byte>((index * 37U + expert_id) & 0xffU);
  }

  auto* header = result.bytes.data();
  std::memcpy(header, "EPEXPR01", 8);
  write_le<std::uint16_t>(header + 8, er::kExpertPackVersion);
  write_le<std::uint16_t>(header + 10, er::kExpertHeaderBytes);
  write_le<std::uint32_t>(header + 12, 0x0fU);
  write_le<std::uint32_t>(header + 16, er::kExpertQuantAbiInt8PerRow);
  write_le<std::int32_t>(header + 20, static_cast<std::int32_t>(layer));
  write_le<std::int32_t>(header + 24, static_cast<std::int32_t>(expert_id));
  write_le<std::uint32_t>(header + 28, 16);
  write_le<std::uint32_t>(header + 32, 8);
  write_le<std::uint32_t>(header + 36, 16);
  write_le<std::uint32_t>(header + 40, 0);
  write_le<std::uint64_t>(header + 44, result.bytes.size());
  write_le<std::uint64_t>(header + 52, 256);
  write_le<std::uint64_t>(header + 60, 256);
  write_le<std::uint64_t>(header + 68, 512);
  write_le<std::uint64_t>(header + 76, 64);
  write_le<std::uint64_t>(header + 84, 768);
  write_le<std::uint64_t>(header + 92, 128);
  write_le<std::uint64_t>(header + 100, 1024);
  write_le<std::uint64_t>(header + 108, 64);
  const auto digest = er::sha256(std::span<const std::byte>(result.bytes).subspan(
      er::kExpertHeaderBytes));
  std::copy(digest.begin(), digest.end(), header + 116);

  result.record.path = "fixture.qpack";
  result.record.record_offset = file_offset;
  result.record.stored_bytes = result.bytes.size();
  result.record.decoded_bytes = 3ULL * 16 * 8 * sizeof(float);
  result.record.header_bytes = er::kExpertHeaderBytes;
  result.record.alignment = er::kExpertPackAlignment;
  result.record.payload_sha256 = digest;
  return result;
}

class ControlledStorage final : public er::IAsyncStorage {
 public:
  struct Pending final {
    er::OperationId id{};
    er::ReadRequest request;
    er::ReadCompletion completion;
  };

  er::OperationId read(er::ReadRequest request,
                       er::ReadCompletion completion) override {
    std::lock_guard lock(mutex_);
    const auto id = next_id_++;
    pending_.push_back({id, std::move(request), std::move(completion)});
    ++read_count_;
    return id;
  }

  void cancel(er::OperationId operation) noexcept override {
    er::ReadCompletion completion;
    std::uint64_t requested = 0;
    {
      std::lock_guard lock(mutex_);
      const auto iterator = std::find_if(
          pending_.begin(), pending_.end(),
          [operation](const Pending& item) { return item.id == operation; });
      if (iterator == pending_.end()) {
        return;
      }
      requested = iterator->request.record.stored_bytes;
      completion = std::move(iterator->completion);
      pending_.erase(iterator);
    }
    completion({er::Status(er::ErrorCode::cancelled, "test read cancelled"),
                requested, 0});
  }

  void complete_success(const std::vector<std::byte>& source) {
    Pending pending = take_one();
    require(source.size() == pending.request.record.stored_bytes,
            "fixture/read size mismatch");
    require(pending.request.destination.capacity >= source.size(),
            "destination too small");
    require(reinterpret_cast<std::uintptr_t>(pending.request.destination.data) %
                    er::kExpertPackAlignment ==
                0,
            "direct-read destination is not 4096 aligned");
    std::copy(source.begin(), source.end(), pending.request.destination.data);
    pending.completion({er::Status::success(), source.size(), source.size()});
  }

  void complete_short(const std::vector<std::byte>& source) {
    Pending pending = take_one();
    const auto bytes = source.size() / 2;
    std::copy_n(source.begin(), bytes, pending.request.destination.data);
    pending.completion({er::Status::success(), source.size(), bytes});
  }

  [[nodiscard]] std::size_t read_count() const {
    std::lock_guard lock(mutex_);
    return read_count_;
  }

  [[nodiscard]] std::size_t pending_count() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
  }

 private:
  Pending take_one() {
    std::lock_guard lock(mutex_);
    require(!pending_.empty(), "no pending storage operation");
    auto pending = std::move(pending_.front());
    pending_.pop_front();
    return pending;
  }

  mutable std::mutex mutex_;
  std::deque<Pending> pending_;
  er::OperationId next_id_{1};
  std::size_t read_count_{};
};

class TestDeviceAllocation final : public er::IDeviceAllocation {
 public:
  explicit TestDeviceAllocation(std::size_t size) : size_(size) {}
  [[nodiscard]] std::size_t bytes() const noexcept override { return size_; }

 private:
  std::size_t size_{};
};

class ControlledUploader final : public er::IDeviceUploader {
 public:
  struct Pending final {
    er::OperationId id{};
    er::UploadRequest request;
    er::UploadCompletion completion;
  };

  er::OperationId upload(er::UploadRequest request,
                         er::UploadCompletion completion) override {
    std::lock_guard lock(mutex_);
    const auto id = next_id_++;
    pending_.push_back({id, request, std::move(completion)});
    ++upload_count_;
    return id;
  }

  void cancel(er::OperationId operation) noexcept override {
    er::UploadCompletion completion;
    {
      std::lock_guard lock(mutex_);
      const auto iterator = std::find_if(
          pending_.begin(), pending_.end(),
          [operation](const Pending& item) { return item.id == operation; });
      if (iterator == pending_.end()) {
        return;
      }
      completion = std::move(iterator->completion);
      pending_.erase(iterator);
    }
    completion({er::Status(er::ErrorCode::cancelled, "test upload cancelled"),
                {}, 0});
  }

  void complete_success(std::size_t device_bytes = 2048) {
    auto pending = take_one();
    pending.completion({er::Status::success(),
                        std::make_shared<TestDeviceAllocation>(device_bytes),
                        device_bytes});
  }

  void complete_failure() {
    auto pending = take_one();
    pending.completion(
        {er::Status(er::ErrorCode::upload_failed, "injected CUDA failure"), {}, 0});
  }

  [[nodiscard]] std::size_t upload_count() const {
    std::lock_guard lock(mutex_);
    return upload_count_;
  }

  [[nodiscard]] std::size_t pending_count() const {
    std::lock_guard lock(mutex_);
    return pending_.size();
  }

 private:
  Pending take_one() {
    std::lock_guard lock(mutex_);
    require(!pending_.empty(), "no pending upload operation");
    auto pending = std::move(pending_.front());
    pending_.pop_front();
    return pending;
  }

  mutable std::mutex mutex_;
  std::deque<Pending> pending_;
  er::OperationId next_id_{1};
  std::size_t upload_count_{};
};

struct Harness final {
  std::shared_ptr<ControlledStorage> storage =
      std::make_shared<ControlledStorage>();
  std::shared_ptr<ControlledUploader> uploader =
      std::make_shared<ControlledUploader>();
  std::shared_ptr<er::FixedBufferPool> buffers;
  er::ExpertCache cache;

  explicit Harness(std::uint64_t budget = 8192, std::size_t slots = 2,
                   std::uint64_t vram_budget = 0,
                   er::CachePlacementConfig placement = {})
      : buffers(std::make_shared<er::FixedBufferPool>(
            slots, er::kExpertPackAlignment, er::kExpertPackAlignment)),
        cache({{budget, budget, std::min<std::uint64_t>(4096, budget)},
               {vram_budget ? vram_budget : budget,
                vram_budget ? vram_budget : budget,
                std::min<std::uint64_t>(4096,
                    vram_budget ? vram_budget : budget)}, true, placement},
              storage, uploader, buffers) {}

  er::AcquireResult finish(er::AcquireHandle& handle,
                           const FixtureRecord& fixture) {
    storage->complete_success(fixture.bytes);
    require(handle.wait_for(0ms) == std::future_status::timeout,
            "expert became visible before upload completion");
    uploader->complete_success();
    auto result = handle.get();
    require(result.status.ok() && result.lease, "valid expert did not publish");
    return result;
  }
};

void test_state_machine_and_sha256() {
  require(er::cache_state_name(er::CacheState::vram_ready) == "VRAM_READY",
          "cache state name");
  require(er::valid_cache_transition(er::CacheState::absent,
                                     er::CacheState::ssd_loading),
          "valid transition rejected");
  require(!er::valid_cache_transition(er::CacheState::absent,
                                      er::CacheState::vram_ready),
          "invalid publish transition accepted");
  const std::array input = {std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
  const auto digest = er::sha256(input);
  constexpr std::array<std::uint8_t, 32> expected = {
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
      0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
      0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
      0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    require(std::to_integer<std::uint8_t>(digest[index]) == expected[index],
            "SHA-256 implementation mismatch");
  }
  er::Sha256 streaming;
  streaming.update(std::span<const std::byte>(input).first(1));
  streaming.update(std::span<const std::byte>(input).subspan(1, 1));
  streaming.update(std::span<const std::byte>(input).subspan(2));
  require(er::constant_time_equal(digest, streaming.finalize()),
          "streaming SHA-256 differs from one-shot digest");
  require(er::constant_time_equal(digest, streaming.finalize()),
          "streaming SHA-256 finalize is not idempotent");
}

void test_ready_first_grouped_scheduler() {
  er::ContinuousBatchScheduler scheduler({2, 4, 8});
  require(scheduler.admit(1).ok() && scheduler.admit(2).ok(),
          "scheduler admission failed");
  require(!scheduler.admit(3).ok(), "scheduler ignored request capacity");
  const std::array first_routes = {
      er::RoutedExpert{7, 0.7F}, er::RoutedExpert{8, 0.2F}};
  const std::array second_routes = {
      er::RoutedExpert{7, 0.6F}, er::RoutedExpert{9, 0.3F}};
  require(scheduler.enqueue({1, 4, 0, 3, 20, first_routes}).ok(),
          "first token enqueue failed");
  require(scheduler.enqueue({2, 9, 1, 3, 10, second_routes}).ok(),
          "second token enqueue failed");
  auto batch = scheduler.schedule([](std::uint32_t, std::uint32_t expert) {
    return expert == 8 ? er::ExpertResidency::absent
                       : er::ExpertResidency::vram_ready;
  });
  require(batch.ready_items == 3 && batch.blocked_items == 1 &&
              batch.groups.size() == 2 && batch.unique_experts == 2,
          "ready-first grouping counts are wrong");
  require(batch.groups[0].expert == 7 && batch.groups[0].items.size() == 2 &&
              batch.groups[0].items[0].request == 2,
          "deadline fairness or expert reuse grouping is wrong");
  std::vector<er::WorkId> completed;
  for (const auto& group : batch.groups)
    for (const auto& item : group.items) completed.push_back(item.work);
  require(scheduler.complete(completed).ok(), "batch completion failed");
  require(!scheduler.token_complete(1, 4) && scheduler.token_complete(2, 9),
          "cold token blocked a ready token or completed too early");
  scheduler.retire_token(2, 9);
  batch = scheduler.schedule([](std::uint32_t, std::uint32_t) {
    return er::ExpertResidency::vram_ready;
  });
  require(batch.ready_items == 1 && batch.groups[0].expert == 8,
          "cold item did not resume after residency changed");
  completed = {batch.groups[0].items[0].work};
  require(scheduler.complete(completed).ok() && scheduler.token_complete(1, 4),
          "resumed token did not complete");
  scheduler.retire_token(1, 4);
  scheduler.finish(1);
  scheduler.cancel(2);
  const auto snapshot = scheduler.snapshot();
  require(snapshot.active_requests == 0 && snapshot.completed_tokens == 2 &&
              snapshot.reused_items == 1 && snapshot.cancelled_requests == 1,
          "scheduler accounting mismatch");
}

void test_concurrent_load_dedup_and_visibility() {
  Harness harness(8192, 2);
  const auto fixture = make_record(7);
  constexpr std::size_t kCallers = 16;
  std::vector<er::AcquireHandle> handles(kCallers);
  std::vector<std::thread> callers;
  callers.reserve(kCallers);
  for (std::size_t index = 0; index < kCallers; ++index) {
    callers.emplace_back([&, index] {
      handles[index] = harness.cache.acquire(fixture.key, fixture.record);
    });
  }
  for (auto& thread : callers) {
    thread.join();
  }
  require(harness.storage->read_count() == 1,
          "concurrent acquire issued a double-load");
  for (auto& handle : handles) {
    require(handle.wait_for(0ms) == std::future_status::timeout,
            "SSD_LOADING entry was visible");
  }

  harness.storage->complete_success(fixture.bytes);
  require(harness.uploader->upload_count() == 1,
          "deduplicated load issued multiple uploads");
  const auto uploading = harness.cache.inspect(fixture.key);
  require(uploading && uploading->state == er::CacheState::gpu_uploading &&
              !uploading->has_device_copy,
          "entry was published before upload completion");
  for (auto& handle : handles) {
    require(handle.wait_for(0ms) == std::future_status::timeout,
            "GPU_UPLOADING entry was visible");
  }

  harness.uploader->complete_success();
  std::vector<er::AcquireResult> results;
  for (auto& handle : handles) {
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "deduplicated waiter did not receive a lease");
    results.push_back(std::move(result));
  }
  const auto ready = harness.cache.inspect(fixture.key);
  require(ready && ready->state == er::CacheState::vram_ready &&
              ready->reference_count == kCallers,
          "ready leases are not reference-counted");
  const auto metrics = harness.cache.telemetry();
  require(metrics.load_started == 1 &&
              metrics.load_deduplicated == kCallers - 1 &&
              metrics.acquire_ssd_misses == kCallers,
          "load dedup telemetry mismatch");
}

void test_budget_eviction_refcount_and_cancellation() {
  Harness harness(8192, 2);
  const auto first = make_record(1, 0);
  const auto second = make_record(2, 4096);
  const auto third = make_record(3, 8192);
  const auto fourth = make_record(4, 12288);

  auto first_handle = harness.cache.acquire(first.key, first.record);
  auto first_result = harness.finish(first_handle, first);
  auto second_handle = harness.cache.acquire(second.key, second.record);
  auto second_result = harness.finish(second_handle, second);
  auto third_handle = harness.cache.acquire(third.key, third.record);
  require(harness.storage->pending_count() == 0,
          "referenced expert was evicted to admit a third record");

  first_result.lease = {};
  require(harness.storage->pending_count() == 1,
          "releasing a lease did not unblock budget admission");
  auto third_result = harness.finish(third_handle, third);
  const auto first_after = harness.cache.inspect(first.key);
  require(first_after && first_after->state == er::CacheState::vram_ready &&
              !first_after->has_host_copy && first_after->has_device_copy,
          "RAM pressure incorrectly coupled RAM and VRAM eviction");
  require(second_result.lease && third_result.lease,
          "live leases were invalidated by eviction");

  auto fourth_handle = harness.cache.acquire(fourth.key, fourth.record);
  require(harness.storage->pending_count() == 0,
          "budget overcommitted while all entries were referenced");
  fourth_handle.cancel();
  auto cancelled = fourth_handle.get();
  require(cancelled.status.code() == er::ErrorCode::cancelled,
          "pending acquisition did not cancel cleanly");
  const auto metrics = harness.cache.telemetry();
  require(metrics.ram_high_water <= 8192 && metrics.vram_high_water <= 8192 &&
              metrics.ram_bytes <= 8192 && metrics.vram_bytes <= 8192,
          "cache exceeded byte budget under churn");
  require(metrics.stalled_by_budget != 0 && metrics.cancellation_count == 1 &&
              metrics.eviction_count != 0,
          "budget/cancel/eviction telemetry missing");
}

void test_short_read_checksum_and_upload_fail_closed() {
  {
    Harness harness(4096, 1);
    const auto fixture = make_record(10);
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_short(fixture.bytes);
    auto result = handle.get();
    require(result.status.code() == er::ErrorCode::short_read && !result.lease,
            "short read was not fail-closed");
    require(harness.uploader->upload_count() == 0,
            "short record reached device uploader");
    require(harness.cache.inspect(fixture.key)->state == er::CacheState::failed,
            "short record was not quarantined");
  }
  {
    Harness harness(4096, 1);
    auto fixture = make_record(11);
    auto corrupted = fixture.bytes;
    corrupted[300] ^= std::byte{0x01};
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(corrupted);
    auto result = handle.get();
    require(result.status.code() == er::ErrorCode::checksum_mismatch &&
                !result.lease,
            "checksum mismatch was not fail-closed");
    require(harness.uploader->upload_count() == 0,
            "corrupt weights reached device uploader");
    require(harness.cache.telemetry().checksum_errors == 1,
            "checksum error telemetry missing");
  }
  {
    Harness harness(4096, 1);
    const auto fixture = make_record(12);
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_failure();
    auto result = handle.get();
    require(result.status.code() == er::ErrorCode::upload_failed && !result.lease,
            "device failure was not fail-closed");
    require(harness.cache.inspect(fixture.key)->state == er::CacheState::failed,
            "failed device copy became visible");
    require(harness.cache.telemetry().upload_errors == 1,
            "upload error telemetry missing");
  }
}

void test_ram_hit_reuploads_after_vram_eviction() {
  Harness harness(8192, 1, 4096);
  const auto first = make_record(20, 0);
  const auto second = make_record(21, 4096);
  auto first_handle = harness.cache.acquire(first.key, first.record);
  auto first_result = harness.finish(first_handle, first);
  first_result.lease = {};

  auto second_handle = harness.cache.acquire(second.key, second.record);
  auto second_result = harness.finish(second_handle, second);
  second_result.lease = {};
  const auto first_in_ram = harness.cache.inspect(first.key);
  require(first_in_ram && first_in_ram->state == er::CacheState::ram_ready &&
              first_in_ram->has_host_copy && !first_in_ram->has_device_copy,
          "VRAM pressure discarded the independent RAM tier");

  const auto reads_before = harness.storage->read_count();
  auto ram_hit = harness.cache.acquire(first.key, first.record);
  require(harness.storage->read_count() == reads_before &&
              harness.uploader->pending_count() == 1,
          "RAM hit reread SSD or failed to schedule H2D");
  harness.uploader->complete_success();
  auto result = ram_hit.get();
  require(result.status.ok() && result.lease,
          "RAM hit did not republish after VRAM reservation");
  auto vram_hit = harness.cache.acquire(first.key, first.record).get();
  require(vram_hit.status.ok() && vram_hit.lease,
          "VRAM hit did not return a lease immediately");
  const auto metrics = harness.cache.telemetry();
  require(metrics.upload_completed == 3 && metrics.acquire_ram_hits == 1 &&
              metrics.acquire_vram_hits == 1 &&
              metrics.acquire_ssd_misses == 2 &&
              metrics.record_validations == 2 &&
              metrics.validated_ram_reuses == 1,
          "RAM reupload telemetry mismatch");
}

void test_host_lease_protects_validated_ram_copy() {
  Harness harness(8192, 1, 4096);
  const auto first = make_record(24, 0);
  const auto second = make_record(25, 4096);
  auto first_handle = harness.cache.acquire(first.key, first.record);
  auto first_result = harness.finish(first_handle, first);
  first_result.lease = {};
  auto second_handle = harness.cache.acquire(second.key, second.record);
  auto second_result = harness.finish(second_handle, second);
  second_result.lease = {};

  auto host = harness.cache.try_acquire_host(first.key, first.record);
  require(host && host->bytes().size() == first.bytes.size() &&
              host->sections().hidden == 16,
          "validated host lease was not exposed from RAM tier");
  static_cast<void>(harness.cache.trim());
  const auto protected_entry = harness.cache.inspect(first.key);
  require(protected_entry && protected_entry->has_host_copy &&
              protected_entry->reference_count == 1,
          "host lease did not protect RAM copy from trim");
  host.reset();
  static_cast<void>(harness.cache.trim());
  const auto released_entry = harness.cache.inspect(first.key);
  require(released_entry && !released_entry->has_host_copy,
          "released host lease remained unevictable");
}

void test_cpu_executor_writes_compact_selection_outputs() {
  auto fixture = make_record(26);
  auto* bytes = fixture.bytes.data();
  std::fill_n(bytes + 256, 256, std::byte{1});
  std::fill_n(bytes + 768, 128, std::byte{1});
  const float scale = 0.01F;
  for (std::size_t offset = 512; offset < 576; offset += sizeof(float))
    std::memcpy(bytes + offset, &scale, sizeof(scale));
  for (std::size_t offset = 1024; offset < 1088; offset += sizeof(float))
    std::memcpy(bytes + offset, &scale, sizeof(scale));
  expert::runtime::cpu::ExpertExecutor executor(2);
  const er::ExpertSections sections{16, 8, 256, 512, 512, 64,
                                    768, 128, 1024, 64};
  const expert::runtime::cpu::ExpertWorkGroup group{
      fixture.bytes, sections, {0, 3}, {1, 0}};
  std::vector<float> inputs(2 * 16, 1.0F);
  std::vector<float> outputs(2 * 16, -123.0F);
  const auto status = executor.execute(std::span(&group, 1), inputs, 2, 2,
                                       outputs);
  require(status.ok(), "CPU expert executor rejected valid fixture");
  for (std::size_t column = 0; column < 16; ++column) {
    require(std::isfinite(outputs[column]) &&
                outputs[column] == outputs[16 + column],
            "CPU compact output is invalid or mapping changed the result");
  }
  auto duplicate = group;
  duplicate.output_slots = {0, 0};
  require(!executor.execute(std::span(&duplicate, 1), inputs, 2, 2,
                            outputs).ok(),
          "CPU executor accepted duplicate compact output slots");
}

void test_layer_partitioned_eviction_protects_other_layers() {
  Harness harness(16384, 2, 6144, {2, 1, 0, 0});
  const auto layer0_old = make_record(30, 0, 0);
  const auto layer1_hot = make_record(31, 4096, 1);
  const auto layer0_new = make_record(32, 8192, 0);

  auto first_handle = harness.cache.acquire(layer0_old.key, layer0_old.record);
  auto first = harness.finish(first_handle, layer0_old);
  first.lease = {};
  auto second_handle = harness.cache.acquire(layer1_hot.key, layer1_hot.record);
  auto second = harness.finish(second_handle, layer1_hot);
  second.lease = {};

  auto third_handle = harness.cache.acquire(layer0_new.key, layer0_new.record);
  require(harness.cache.inspect(layer0_old.key)->state ==
              er::CacheState::ram_ready,
          "layer partition did not evict from the requesting layer");
  require(harness.cache.inspect(layer1_hot.key)->state ==
              er::CacheState::vram_ready,
          "layer partition evicted another layer's protected working set");
  auto third = harness.finish(third_handle, layer0_new);
  require(third.status.ok() && third.lease,
          "replacement expert did not become ready");
  require(harness.cache.telemetry().same_partition_evictions == 1,
          "same-partition eviction telemetry mismatch");
}

void test_frequency_admission_protects_reused_expert() {
  Harness harness(24576, 4, 12288, {2, 1, 8192, 0, 4096});
  const auto hot = make_record(40, 0, 0);
  const auto cold_a = make_record(41, 4096, 0);
  const auto incoming = make_record(42, 8192, 0);

  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    auto result = harness.finish(handle, fixture);
    result.lease = {};
  };
  load(hot);
  for (int reuse = 0; reuse < 3; ++reuse) {
    auto hit = harness.cache.acquire(hot.key, hot.record).get();
    require(hit.status.ok() && hit.lease, "hot expert reuse failed");
  }
  load(cold_a);

  auto incoming_handle = harness.cache.acquire(incoming.key, incoming.record);
  require(harness.cache.inspect(hot.key)->state == er::CacheState::vram_ready,
          "one-hit admission evicted the reused expert");
  require(harness.cache.inspect(cold_a.key)->state ==
              er::CacheState::ram_ready,
          "transient admission did not recycle the one-hit slot");
  auto incoming_result = harness.finish(incoming_handle, incoming);
  require(incoming_result.status.ok() && incoming_result.lease,
          "frequency-admitted expert did not publish");
}

void test_vram_replacement_requires_a_strictly_colder_victim() {
  Harness harness(16384, 4, 8192);
  const auto candidate = make_record(50, 0, 0);
  const auto resident_a = make_record(51, 4096, 0);
  const auto resident_b = make_record(52, 8192, 0);
  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "placement fixture failed to load expert");
    result.lease = {};
  };
  load(candidate);
  load(resident_a);
  require(harness.cache.record_access(resident_a.key, 4),
          "fixture did not protect the intended resident");
  load(resident_b);
  require(harness.cache.inspect(candidate.key)->state ==
              er::CacheState::ram_ready,
          "fixture did not create a RAM promotion candidate");
  const std::array candidate_accesses{er::ExpertAccess{candidate.key, 8}};
  require(harness.cache.record_accesses(candidate_accesses) == 8,
          "batched route feedback did not update candidate frequency");

  auto lease_a = harness.cache.acquire(resident_a.key, resident_a.record).get();
  auto lease_b = harness.cache.acquire(resident_b.key, resident_b.record).get();
  require(lease_a.lease && lease_b.lease,
          "resident protection leases were not acquired");
  require(!harness.cache.vram_admission_would_improve(candidate.key,
                                                       candidate.record),
          "admission selected an in-flight VRAM victim");
  lease_a.lease = {};
  require(harness.cache.vram_admission_would_improve(candidate.key,
                                                      candidate.record),
          "hot RAM candidate did not outrank a colder VRAM resident");
  er::AdaptivePlacementPlanner planner(harness.cache);
  planner.consider(candidate.key, candidate.record, 1);
  require(planner.telemetry().scheduled == 1 &&
              harness.uploader->pending_count() == 1,
          "admitted promotion did not enter the asynchronous uploader");
  harness.uploader->complete_success(er::kExpertPackAlignment);
  planner.poll();
  require(planner.telemetry().completed == 1 &&
              harness.cache.inspect(candidate.key)->state ==
                  er::CacheState::vram_ready,
          "asynchronous promotion did not publish its VRAM entry");
  require(planner.quiesce(10ms).ok() && planner.frozen(),
          "placement epoch did not quiesce and freeze");
  planner.resume();
  require(!planner.frozen(), "placement epoch did not resume");
}

void test_hybrid_dispatch_minimizes_measured_critical_path() {
  er::HybridDispatchPlanner planner({100.0, 10.0, 1.0e9, 0.5, 16, 16});
  const std::array candidates{
      er::HybridDispatchCandidate{3, 10, 0, true, true, true},
      er::HybridDispatchCandidate{1, 5, 10, false, true, true},
      er::HybridDispatchCandidate{2, 5, 10'000, false, true, true},
      er::HybridDispatchCandidate{4, 2, 0, false, true, false},
      er::HybridDispatchCandidate{5, 2, 100, false, false, true},
  };
  const auto plan = planner.plan(candidates);
  require(plan.status.ok() && plan.decisions.size() == candidates.size(),
          "hybrid dispatch rejected a feasible layer");
  const auto decision = [&](std::uint32_t expert) -> const auto& {
    const auto found = std::find_if(
        plan.decisions.begin(), plan.decisions.end(),
        [&](const auto& value) { return value.expert == expert; });
    require(found != plan.decisions.end(), "hybrid decision is missing");
    return *found;
  };
  require(decision(3).executor == er::HybridExecutor::gpu_resident &&
              decision(3).reason == er::HybridDispatchReason::resident_gpu,
          "resident expert did not retain GPU priority");
  require(decision(1).executor == er::HybridExecutor::gpu_upload &&
              decision(1).reason ==
                  er::HybridDispatchReason::gpu_lower_critical_path,
          "small upload did not shorten the projected critical path");
  require(decision(2).executor == er::HybridExecutor::cpu_local &&
              decision(2).reason ==
                  er::HybridDispatchReason::cpu_lower_critical_path,
          "large upload was not kept on CPU");
  require(decision(4).reason == er::HybridDispatchReason::cpu_only &&
              decision(5).reason == er::HybridDispatchReason::gpu_only,
          "forced executor reason was not preserved");

  planner.observe_cpu(1'000, 20);
  planner.observe_gpu(400, 20);
  planner.observe_h2d(1'000, 2'000);
  const auto telemetry = planner.telemetry();
  require(telemetry.cpu_ns_per_selection == 75.0 &&
              telemetry.gpu_ns_per_selection == 15.0 &&
              telemetry.h2d_bytes_per_second == 1.5e9 &&
              telemetry.plans == 1 && telemetry.candidates == 5,
          "hybrid EWMA or decision telemetry mismatch");
}

void test_hybrid_dispatch_ties_bounds_and_trace_are_deterministic() {
  er::HybridDispatchPlanner planner({100.0, 10.0, 1.0e9, 0.5, 2, 2});
  const std::array tie{
      er::HybridDispatchCandidate{7, 1, 90, false, true, true}};
  const auto tied = planner.plan(tie);
  require(tied.status.ok() &&
              tied.decisions.front().executor ==
                  er::HybridExecutor::cpu_local &&
              tied.decisions.front().reason ==
                  er::HybridDispatchReason::cpu_stable_tie,
          "hybrid tie did not use the stable CPU fallback");
  const std::array two{
      er::HybridDispatchCandidate{8, 1, 0, true, false, false},
      er::HybridDispatchCandidate{9, 1, 0, true, false, false}};
  require(planner.plan(two).status.ok(), "bounded hybrid plan failed");
  const auto trace = planner.trace();
  require(trace.size() == 2 && trace[0].expert == 8 && trace[1].expert == 9,
          "bounded trace did not retain the newest decisions in order");
  const std::array duplicate{
      er::HybridDispatchCandidate{1, 1, 0, true, false, false},
      er::HybridDispatchCandidate{1, 1, 0, true, false, false}};
  require(!planner.plan(duplicate).status.ok(),
          "hybrid planner accepted duplicate experts");
  const std::array too_many{
      er::HybridDispatchCandidate{1, 1, 0, true, false, false},
      er::HybridDispatchCandidate{2, 1, 0, true, false, false},
      er::HybridDispatchCandidate{3, 1, 0, true, false, false}};
  require(!planner.plan(too_many).status.ok() &&
              planner.telemetry().rejected_plans == 2,
          "hybrid planner did not enforce its candidate bound");
}

}  // namespace

int main() {
  try {
    test_state_machine_and_sha256();
    test_ready_first_grouped_scheduler();
    test_concurrent_load_dedup_and_visibility();
    test_budget_eviction_refcount_and_cancellation();
    test_short_read_checksum_and_upload_fail_closed();
    test_ram_hit_reuploads_after_vram_eviction();
    test_host_lease_protects_validated_ram_copy();
    test_cpu_executor_writes_compact_selection_outputs();
    test_layer_partitioned_eviction_protects_other_layers();
    test_frequency_admission_protects_reused_expert();
    test_vram_replacement_requires_a_strictly_colder_victim();
    test_hybrid_dispatch_minimizes_measured_critical_path();
    test_hybrid_dispatch_ties_bounds_and_trace_are_deterministic();
    std::cout << "expert_runtime_tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert_runtime_tests: FAIL: " << error.what() << '\n';
    return 1;
  }
}
