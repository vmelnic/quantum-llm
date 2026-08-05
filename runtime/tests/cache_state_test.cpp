#include "expert/runtime/adaptive_placement.hpp"
#include "expert/runtime/deepseek_expert.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/expert_store.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/resource_governor.hpp"
#include "expert/runtime/route_census.hpp"
#include "expert/runtime/cpu/deepseek_packed_executor.hpp"
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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
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

void test_deepseek_compact_and_sm86_hot_abi() {
  const auto geometry = er::DeepSeekExpertGeometry::v4_flash();
  require(geometry.valid(), "DeepSeek-V4 geometry is invalid");
  require(geometry.compact_weight_bytes(er::DeepSeekProjection::w1_gate) ==
              4'194'304U &&
              geometry.compact_scale_bytes(er::DeepSeekProjection::w1_gate) ==
                  262'144U &&
              geometry.compact_expert_bytes() == 13'369'344U,
          "DeepSeek compact ABI byte geometry changed");

  const auto layout = er::make_deepseek_sm86_hot_layout(geometry);
  require(layout.alignment == 256U && layout.gate_up_q.offset == 0U &&
              layout.gate_up_q.bytes == 16'777'216U &&
              layout.gate_up_scales.offset == 16'777'216U &&
              layout.gate_up_scales.bytes == 16'384U &&
              layout.down_q.offset == 16'793'600U &&
              layout.down_q.bytes == 8'388'608U &&
              layout.down_scales.offset == 25'182'208U &&
              layout.down_scales.bytes == 16'384U &&
              layout.slot_bytes == 25'198'592U,
          "DeepSeek SM86 hot-cache ABI layout changed");

  bool rejected = false;
  try {
    static_cast<void>(er::make_deepseek_sm86_hot_layout(geometry, 192U));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "DeepSeek hot-cache ABI accepted unsafe alignment");
}

void test_deepseek_compact_admission_validation() {
  std::vector<std::byte> bytes(13'369'344U);
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::byte>((index * 29U + 7U) & 0xffU);
  for (const auto offset : {4'194'304U, 8'650'752U, 13'107'200U}) {
    for (std::size_t index = offset; index < offset + 262'144U; ++index) {
      if (bytes[index] == std::byte{0xff}) bytes[index] = std::byte{0xfe};
    }
  }
  er::PayloadRecord record;
  record.stored_bytes = bytes.size();
  record.decoded_bytes = 3ULL * 4096U * 2048U * sizeof(float);
  record.device_bytes = 25'198'592U;
  record.source_abi = er::kExpertSourceAbiDeepSeekCompactV1;
  record.header_bytes = 0U;
  record.alignment = 1U;
  record.payload_sha256 = er::sha256(bytes);
  const er::ExpertKey key{17U, 0U, 0U, er::kExpertQuantAbiDeepSeekSm86};
  const auto valid = er::validate_expert_admission(bytes, key, record);
  require(valid.status.ok() && valid.target.hidden == 4096U &&
              valid.target.down_scale_offset == 25'182'208U &&
              valid.compact.w2_scale_offset == 13'107'200U,
          "valid DeepSeek compact admission was rejected");
  bytes.back() ^= std::byte{1};
  require(!er::validate_expert_admission(bytes, key, record).status.ok(),
          "corrupt DeepSeek compact admission was accepted");
  bytes = std::vector<std::byte>(13'369'344U);
  bytes[4'194'304U] = std::byte{0xff};
  require(!er::validate_expert_admission(bytes, key, record, false).status.ok(),
          "trusted DeepSeek compact admission accepted a UE8M0 NaN");
}

void test_deepseek_fp8_shared_admission_validation() {
  std::vector<std::byte> bytes(25'167'360U);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::byte>((index * 17U + 11U) & 0x7eU);
  }
  er::PayloadRecord record;
  record.stored_bytes = bytes.size();
  record.decoded_bytes = 3ULL * 4096U * 2048U * sizeof(float);
  record.device_bytes = 25'198'592U;
  record.source_abi = er::kExpertSourceAbiDeepSeekFp8Block128V1;
  record.header_bytes = 0U;
  record.alignment = 1U;
  record.payload_sha256 = er::sha256(bytes);
  const er::ExpertKey key{17U, 0U, 256U, er::kExpertQuantAbiDeepSeekSm86};
  const auto valid = er::validate_expert_admission(bytes, key, record);
  require(valid.status.ok() && valid.compact.w1_scale_offset == 8'388'608U &&
              valid.compact.w3_weight_offset == 8'389'120U &&
              valid.compact.w2_scale_offset == 25'166'848U &&
              valid.target.down_scale_offset == 25'182'208U,
          "valid DeepSeek FP8 shared admission was rejected");
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
    require(!pending.request.direct ||
                reinterpret_cast<std::uintptr_t>(
                    pending.request.destination.data) %
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

void test_extent_gather_is_exact_and_bounded() {
  auto backing = std::make_shared<ControlledStorage>();
  er::ExtentGatherStorage gather(backing);
  std::array<std::byte, 12> destination{};
  er::PayloadRecord record;
  record.stored_bytes = destination.size();
  record.extents = {
      {"shard-a", 101U, 0U, 4U},
      {"shard-b", 202U, 4U, 3U},
      {"shard-a", 303U, 7U, 5U},
  };
  std::promise<er::ReadResult> promise;
  auto result = promise.get_future();
  static_cast<void>(gather.read(
      {record, {destination.data(), destination.size()}, true},
      [&promise](er::ReadResult read) { promise.set_value(std::move(read)); }));
  require(backing->pending_count() == 3U,
          "extent gather did not issue one bounded read per source range");
  backing->complete_success(
      {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}});
  backing->complete_success({std::byte{5}, std::byte{6}, std::byte{7}});
  backing->complete_success({std::byte{8}, std::byte{9}, std::byte{10},
                             std::byte{11}, std::byte{12}});
  const auto completed = result.get();
  require(completed.status.ok() && completed.requested_bytes == 12U &&
              completed.read_bytes == 12U &&
              destination[0] == std::byte{1} &&
              destination[6] == std::byte{7} &&
              destination[11] == std::byte{12},
          "extent gather changed source order or byte accounting");

  record.extents[1].destination_offset = 5U;
  bool rejected = false;
  static_cast<void>(gather.read(
      {record, {destination.data(), destination.size()}, true},
      [&rejected](er::ReadResult read) { rejected = !read.status.ok(); }));
  require(rejected && backing->pending_count() == 0U,
          "extent gather accepted a destination gap");
}

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

void test_expert_store_resolves_complete_ordered_union() {
  Harness harness(16'384, 2, 16'384);
  const auto first = make_record(31, 0);
  const auto second = make_record(32, er::kExpertPackAlignment);
  er::LocalExpertStore store(harness.cache);
  const std::array requests = {
      er::ExpertResolveRequest{second.key, second.record},
      er::ExpertResolveRequest{first.key, first.record},
  };
  auto batch = store.resolve(requests);
  require(batch.valid() && batch.size() == 2U && !batch.poll(),
          "expert store did not retain a pending union");

  harness.storage->complete_success(second.bytes);
  harness.uploader->complete_success();
  require(!batch.poll(), "expert store published a partial union");
  harness.storage->complete_success(first.bytes);
  harness.uploader->complete_success();
  auto resolved = batch.poll();
  require(resolved && resolved->status.ok() &&
              resolved->experts.size() == 2U &&
              resolved->experts[0].key == second.key &&
              resolved->experts[1].key == first.key &&
              resolved->experts[0].placement == er::ExpertPlacementKind::device &&
              resolved->experts[1].placement == er::ExpertPlacementKind::device &&
              resolved->experts[0].device_lease &&
              resolved->experts[1].device_lease,
          "expert store changed order or omitted a union member");
  require(!batch.valid() && !batch.poll(),
          "terminal expert store handle was reusable");

  const std::array duplicate = {
      er::ExpertResolveRequest{first.key, first.record},
      er::ExpertResolveRequest{first.key, first.record},
  };
  require(!store.resolve(duplicate).valid(),
          "expert store accepted duplicate immutable keys");

  const std::array host_request = {
      er::ExpertResolveRequest{second.key, second.record,
                               er::ExpertResolveTarget::host_ready}};
  auto host_batch = store.resolve(host_request);
  auto host_resolved = host_batch.poll();
  require(host_resolved && host_resolved->status.ok() &&
              host_resolved->experts.size() == 1U &&
              host_resolved->experts[0].placement ==
                  er::ExpertPlacementKind::host &&
              host_resolved->experts[0].host_lease &&
              !host_resolved->experts[0].device_lease,
          "expert store did not publish a retained host placement");
}

class TestMemoryTier final : public er::ITrimmableMemoryTier {
 public:
  TestMemoryTier(std::string_view tier_name, er::MemoryDomain tier_domain,
                 std::uint64_t used, std::uint64_t protected_bytes)
      : name_(tier_name), domain_(tier_domain), used_(used),
        protected_(protected_bytes) {}
  [[nodiscard]] std::string_view name() const noexcept override { return name_; }
  [[nodiscard]] er::MemoryDomain domain() const noexcept override {
    return domain_;
  }
  [[nodiscard]] std::uint64_t used_bytes() const noexcept override {
    return used_;
  }
  [[nodiscard]] std::uint64_t protected_bytes() const noexcept override {
    return protected_;
  }
  [[nodiscard]] std::uint64_t trim_to(std::uint64_t target) override {
    used_ = std::max(protected_, std::min(used_, target));
    ++trims_;
    return used_;
  }
  [[nodiscard]] std::uint64_t trims() const noexcept { return trims_; }

 private:
  std::string name_;
  er::MemoryDomain domain_;
  std::uint64_t used_{};
  std::uint64_t protected_{};
  std::uint64_t trims_{};
};

void test_resource_governor_trims_before_reserving() {
  er::MemoryResourceGovernor governor({1000U, 1000U, 100U, 100U});
  auto transient = std::make_shared<TestMemoryTier>(
      "transient", er::MemoryDomain::device, 400U, 50U);
  auto hot = std::make_shared<TestMemoryTier>(
      "hot", er::MemoryDomain::device, 300U, 200U);
  governor.register_tier({hot, 20U});
  governor.register_tier({transient, 10U});
  require(governor.reserve(er::MemoryDomain::device, 500U).ok(),
          "resource governor rejected a trimmable reservation");
  auto snapshot = governor.snapshot();
  require(snapshot.device_reserved_bytes == 500U &&
              snapshot.device_tier_bytes == 400U &&
              transient->used_bytes() == 100U && hot->used_bytes() == 300U &&
              transient->trims() == 1U && hot->trims() == 0U,
          "resource governor ignored priority or target accounting");
  require(!governor.reserve(er::MemoryDomain::device, 250U).ok(),
          "resource governor overcommitted protected device memory");
  snapshot = governor.snapshot();
  require(snapshot.rejected_reservations == 1U &&
              snapshot.trimmed_bytes == 450U,
          "resource governor trim/rejection telemetry mismatch");
  governor.release(er::MemoryDomain::device, 500U);
  require(governor.snapshot().device_reserved_bytes == 0U,
          "resource governor did not release reservation credits");
}

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

void test_expanding_admission_reserves_exact_device_bytes() {
  auto fixture = make_record(61);
  fixture.record.device_bytes = 8192;
  {
    Harness undersized(8192, 2, 4096);
    auto blocked = undersized.cache.acquire(fixture.key, fixture.record);
    require(undersized.storage->pending_count() == 0,
            "expanding admission started I/O without VRAM capacity");
    blocked.cancel();
    require(!blocked.get().status.ok(),
            "cancelled expanding admission unexpectedly succeeded");
  }
  Harness sized(8192, 2, 8192);
  auto admitted = sized.cache.acquire(fixture.key, fixture.record);
  require(sized.storage->pending_count() == 1,
          "valid expanding admission did not start I/O");
  sized.storage->complete_success(fixture.bytes);
  sized.uploader->complete_success(8192);
  auto result = admitted.get();
  require(result.status.ok() && result.lease &&
              result.lease.get()->bytes() == 8192,
          "expanding admission did not publish exact device allocation");
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
  const auto executor_metrics = executor.telemetry();
  require(executor_metrics.maximum_threads == 2 &&
              executor_metrics.selected_threads >= 1 &&
              executor_metrics.selected_threads <= 2 &&
              executor_metrics.calibration_runs >= 1 &&
              executor_metrics.execute_calls == 1 &&
              executor_metrics.selections == 2 &&
              executor_metrics.effective_weight_bytes == 768 &&
              executor_metrics.compute_ns > 0,
          "CPU executor calibration or bandwidth telemetry mismatch");
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

void test_deepseek_packed_executor_engages_every_worker() {
  constexpr std::uint32_t hidden = 32U;
  constexpr std::uint32_t intermediate = 32U;
  constexpr std::size_t matrix_bytes = hidden * intermediate / 2U;
  constexpr std::size_t scale_bytes = hidden * intermediate / 32U;
  constexpr std::size_t record_bytes = 3U * (matrix_bytes + scale_bytes);
  std::vector<std::byte> record(record_bytes);
  const auto fill_matrix = [&](std::size_t weight_offset,
                               std::size_t scale_offset) {
    for (std::size_t index = 0U; index < matrix_bytes; ++index) {
      record[weight_offset + index] = static_cast<std::byte>(
          (index * 37U + weight_offset / 17U + 11U) & 0xffU);
    }
    std::fill_n(record.begin() + scale_offset, scale_bytes,
                std::byte{127});
  };
  const std::size_t w1_weight = 0U;
  const std::size_t w1_scale = w1_weight + matrix_bytes;
  const std::size_t w3_weight = w1_scale + scale_bytes;
  const std::size_t w3_scale = w3_weight + matrix_bytes;
  const std::size_t w2_weight = w3_scale + scale_bytes;
  const std::size_t w2_scale = w2_weight + matrix_bytes;
  fill_matrix(w1_weight, w1_scale);
  fill_matrix(w3_weight, w3_scale);
  fill_matrix(w2_weight, w2_scale);
  const er::DeepSeekCompactSections sections{
      w1_weight, matrix_bytes, w1_scale, scale_bytes,
      w3_weight, matrix_bytes, w3_scale, scale_bytes,
      w2_weight, matrix_bytes, w2_scale, scale_bytes};
  const expert::runtime::cpu::DeepSeekPackedWorkGroup group{
      record, sections, hidden, intermediate, {0U, 3U}, {1U, 0U}};
  expert::runtime::cpu::DeepSeekPackedExecutor executor(
      {4U, 8U, 8U, 0.0F, false, false});
  std::vector<float> inputs(2U * hidden);
  for (std::size_t column = 0U; column < hidden; ++column) {
    const auto value = static_cast<float>(
        static_cast<int>(column % 11U) - 5) * 0.125F;
    inputs[column] = value;
    inputs[hidden + column] = value;
  }
  std::vector<float> outputs(2U * hidden, -123.0F);
  const auto status = executor.execute(std::span(&group, 1U), inputs, 2U, 2U,
                                       outputs);
  require(status.ok(), "packed DeepSeek CPU executor rejected valid fixture");
  const auto decode = [](std::uint8_t code) {
    const auto index = code & 7U;
    const auto magnitude = index <= 4U ? static_cast<int>(index)
                           : index == 5U ? 6
                           : index == 6U ? 8
                                         : 12;
    return 0.5F * static_cast<float>((code & 8U) ? -magnitude : magnitude);
  };
  std::array<std::int8_t, hidden> q_input{};
  float input_maximum = 0.0F;
  for (std::size_t column = 0U; column < hidden; ++column)
    input_maximum = std::max(input_maximum, std::abs(inputs[column]));
  const auto input_scale = input_maximum / 127.0F;
  for (std::size_t column = 0U; column < hidden; ++column) {
    q_input[column] = static_cast<std::int8_t>(std::clamp(
        static_cast<int>(std::nearbyint(inputs[column] / input_scale)),
        -127, 127));
  }
  const auto projection = [&](std::size_t offset, std::uint32_t row,
                              const auto& activation, float scale) {
    float sum = 0.0F;
    for (std::size_t column = 0U; column < activation.size(); ++column) {
      const auto packed = std::to_integer<std::uint8_t>(
          record[offset + static_cast<std::size_t>(row) *
                              activation.size() / 2U +
                 column / 2U]);
      const auto code = (column & 1U) == 0U ? packed & 0x0fU : packed >> 4U;
      sum += decode(code) * static_cast<float>(activation[column]) * scale;
    }
    return sum;
  };
  std::array<float, intermediate> reference_intermediate{};
  for (std::uint32_t row = 0U; row < intermediate; ++row) {
    const auto gate = projection(w1_weight, row, q_input, input_scale);
    const auto up = projection(w3_weight, row, q_input, input_scale);
    reference_intermediate[row] =
        (gate / (1.0F + std::exp(-gate))) * up;
  }
  float intermediate_maximum = 0.0F;
  for (const auto value : reference_intermediate)
    intermediate_maximum = std::max(intermediate_maximum, std::abs(value));
  const auto intermediate_scale = intermediate_maximum > 0.0F
                                      ? intermediate_maximum / 127.0F
                                      : 1.0F;
  std::array<std::int8_t, intermediate> q_intermediate{};
  for (std::size_t column = 0U; column < intermediate; ++column) {
    q_intermediate[column] = static_cast<std::int8_t>(std::clamp(
        static_cast<int>(std::nearbyint(reference_intermediate[column] /
                                        intermediate_scale)),
        -127, 127));
  }
  for (std::size_t column = 0U; column < hidden; ++column) {
    const auto expected = projection(w2_weight,
                                     static_cast<std::uint32_t>(column),
                                     q_intermediate, intermediate_scale);
    require(std::isfinite(outputs[column]) &&
                std::abs(outputs[column] - outputs[hidden + column]) < 1e-3F &&
                std::abs(outputs[column] - expected) < 1e-3F,
            "packed DeepSeek CPU decode or output mapping is incorrect: actual=" +
                std::to_string(outputs[column]) + ", expected=" +
                std::to_string(expected) + ", column=" +
                std::to_string(column));
  }
  const auto metrics = executor.telemetry();
  require(metrics.maximum_threads == 4U && metrics.workers_used_last == 4U &&
              metrics.worker_mask_last == 0x0fU &&
              metrics.execute_calls == 1U && metrics.selections == 2U &&
              metrics.source_weight_bytes == record_bytes &&
              metrics.compute_ns > 0U,
          "packed DeepSeek CPU executor did not use every configured worker");
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

void test_routing_score_temperature_breaks_frequency_ties() {
  Harness harness(16384, 4, 8192);
  const auto high_score = make_record(43, 0, 0);
  const auto low_score = make_record(44, 4096, 0);
  const auto incoming = make_record(45, 8192, 0);
  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "score-temperature fixture load failed");
    result.lease = {};
  };
  load(high_score);
  load(low_score);
  const std::array accesses{
      er::ExpertAccess{high_score.key, 1, 0.9, 0.9},
      er::ExpertAccess{low_score.key, 1, 0.1, 0.1},
  };
  require(harness.cache.record_accesses(accesses) == 2,
          "scored route feedback was not recorded");
  const auto high_snapshot = harness.cache.inspect(high_score.key);
  const auto low_snapshot = harness.cache.inspect(low_score.key);
  require(high_snapshot && low_snapshot &&
              high_snapshot->frequency == low_snapshot->frequency &&
              high_snapshot->routing_score_mass_q20 >
                  low_snapshot->routing_score_mass_q20 &&
              high_snapshot->placement_temperature >
                  low_snapshot->placement_temperature,
          "routing score did not affect bounded cache temperature");

  auto incoming_handle = harness.cache.acquire(incoming.key, incoming.record);
  const auto high_after = harness.cache.inspect(high_score.key);
  const auto low_after = harness.cache.inspect(low_score.key);
  if (!high_after || !low_after ||
      high_after->state != er::CacheState::vram_ready ||
      low_after->state != er::CacheState::ram_ready) {
    throw std::runtime_error(
        "score-aware eviction state mismatch: high=" +
        std::to_string(high_after ? static_cast<int>(high_after->state) : -1) +
        ", low=" +
        std::to_string(low_after ? static_cast<int>(low_after->state) : -1));
  }
  auto incoming_result = harness.finish(incoming_handle, incoming);
  require(incoming_result.status.ok() && incoming_result.lease,
          "score-aware replacement did not publish");
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
  er::AdaptivePlacementConfig placement_config;
  placement_config.enable_prefetch = true;
  placement_config.minimum_recent_observations = 1;
  er::AdaptivePlacementPlanner planner(harness.cache, placement_config);
  planner.consider(candidate.key, candidate.record, 1);
  require(planner.telemetry().scheduled == 1 &&
              harness.uploader->pending_count() == 1,
          "admitted promotion did not enter the asynchronous uploader");
  const std::array routed_candidate{candidate.key};
  planner.observe_routes(std::span<const er::ExpertKey>{}, routed_candidate);
  harness.uploader->complete_success(er::kExpertPackAlignment);
  planner.poll();
  require(planner.telemetry().completed == 1 &&
              planner.telemetry().useful_prefetches == 1 &&
              planner.telemetry().useful_prefetch_bytes ==
                  er::kExpertPackAlignment &&
              harness.cache.inspect(candidate.key)->state ==
                  er::CacheState::vram_ready,
          "useful asynchronous prefetch was not attributed or published");
  require(planner.quiesce(10ms).ok() && planner.frozen(),
          "placement epoch did not quiesce and freeze");
  planner.resume();
  require(!planner.frozen(), "placement epoch did not resume");
}

void test_prefetch_credits_and_stale_epoch_cancel_pending_work() {
  Harness harness(16384, 4, 8192);
  const auto candidate = make_record(53, 0, 0);
  const auto resident_a = make_record(54, 4096, 0);
  const auto resident_b = make_record(55, 8192, 0);
  const auto rejected = make_record(56, 12288, 0);
  const auto load = [&](const FixtureRecord& fixture) {
    auto handle = harness.cache.acquire(fixture.key, fixture.record);
    harness.storage->complete_success(fixture.bytes);
    harness.uploader->complete_success(er::kExpertPackAlignment);
    auto result = handle.get();
    require(result.status.ok() && result.lease,
            "prefetch cancellation fixture failed to load");
    result.lease = {};
  };
  load(candidate);
  load(resident_a);
  load(resident_b);
  const std::array hot_accesses{er::ExpertAccess{candidate.key, 8, 4.0, 0.8}};
  require(harness.cache.record_accesses(hot_accesses) == 8,
          "prefetch candidate did not become hotter than residents");

  er::AdaptivePlacementConfig config;
  config.enable_prefetch = true;
  config.maximum_candidates = 1;
  config.minimum_recent_observations = 1;
  config.candidate_ttl_epochs = 1;
  er::AdaptivePlacementPlanner planner(harness.cache, config);
  planner.consider(candidate.key, candidate.record, 1, 0.8);
  require(harness.uploader->pending_count() == 1,
          "bounded prefetch did not consume its one in-flight credit");
  planner.consider(rejected.key, rejected.record, 1, 0.9);
  require(planner.telemetry().credit_rejections == 1,
          "busy candidate bound did not reject additional prefetch history");

  planner.observe_routes(std::span<const er::ExpertKey>{},
                         std::span<const er::ExpertKey>{});
  planner.observe_routes(std::span<const er::ExpertKey>{},
                         std::span<const er::ExpertKey>{});
  const auto telemetry = planner.telemetry();
  require(telemetry.stale_cancellations == 1 &&
              telemetry.wasted_prefetches == 1 &&
              telemetry.wasted_prefetch_bytes == er::kExpertPackAlignment &&
              telemetry.failed == 0 && harness.uploader->pending_count() == 0,
          "stale pending prefetch was not cancelled and attributed exactly");
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

void test_route_census_is_bounded_ranked_and_recoverable() {
  const std::array identity_bytes{std::byte{1}, std::byte{7}, std::byte{9}};
  er::RouteCensusConfig config{17U, er::sha256(identity_bytes), 3U,
                               2U, 8U, 3U, 2U};
  er::RouteCensus census(config);
  const std::array first{1U, 2U, 3U};
  const std::array first_cpu{3U};
  require(census.observe(0U, first, first_cpu).ok(),
          "route census rejected a valid first route");
  const auto root = std::filesystem::temp_directory_path() /
      ("quantum-route-census-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto prefix = root / "routes";
  require(census.save(prefix).ok(),
          "route census failed to publish generation one");
  require(std::filesystem::file_size(
              std::filesystem::path(prefix.string() + ".1")) ==
              census.snapshot().serialized_bytes,
          "route census serialized-size accounting changed");

  const std::array second{4U, 5U, 6U};
  const std::array third{1U, 2U, 4U};
  const std::array third_cpu{4U};
  require(census.observe(1U, second).ok() &&
              census.observe(0U, third, third_cpu).ok(),
          "route census rejected a valid continuation");
  const std::array duplicate{1U, 1U, 2U};
  require(!census.observe(0U, duplicate).ok(),
          "route census accepted a duplicate route");
  const auto warm = census.stable_warm_set(3U, 2U);
  require(warm.size() == 3U && warm[0].key.layer == 0U &&
              (warm[0].key.expert == 1U || warm[0].key.expert == 2U) &&
              std::count_if(warm.begin(), warm.end(), [](const auto& value) {
                return value.key.layer == 0U;
              }) <= 2,
          "route census warm ranking or per-layer bound is unstable");
  const auto before_save = census.snapshot();
  require(before_save.completed_routes == 3U &&
              before_save.total_selections == 9U &&
              before_save.consecutive_reuse_selections == 2U &&
              before_save.observed_experts == 7U &&
              census.save(prefix).ok(),
          "route census accounting or generation two save failed");

  auto loaded = er::RouteCensus::load(prefix, config);
  require(loaded.status.ok() && loaded.census &&
              loaded.census->snapshot().generation == 2U &&
              loaded.census->snapshot().completed_routes == 3U,
          "route census did not load its newest valid generation");
  auto mismatch = config;
  mismatch.model_content_hash[0] ^= std::byte{1};
  require(!er::RouteCensus::load(prefix, mismatch).status.ok(),
          "route census accepted a different model identity");

  const auto newest = std::filesystem::path(prefix.string() + ".0");
  std::fstream corrupt(newest, std::ios::binary | std::ios::in |
                                   std::ios::out);
  require(static_cast<bool>(corrupt),
          "route census corruption fixture could not open newest slot");
  corrupt.seekp(16);
  const char changed = static_cast<char>(0xa5);
  corrupt.write(&changed, 1);
  corrupt.close();
  auto recovered = er::RouteCensus::load(prefix, config);
  require(recovered.status.ok() && recovered.census &&
              recovered.census->snapshot().generation == 1U &&
              recovered.census->snapshot().completed_routes == 1U,
          "route census did not recover the previous authenticated slot");
  std::error_code cleanup_error;
  std::filesystem::remove_all(root, cleanup_error);
}

}  // namespace

int main() {
  try {
    test_deepseek_compact_and_sm86_hot_abi();
    test_deepseek_compact_admission_validation();
    test_deepseek_fp8_shared_admission_validation();
    test_extent_gather_is_exact_and_bounded();
    test_state_machine_and_sha256();
    test_expanding_admission_reserves_exact_device_bytes();
    test_expert_store_resolves_complete_ordered_union();
    test_resource_governor_trims_before_reserving();
    test_ready_first_grouped_scheduler();
    test_concurrent_load_dedup_and_visibility();
    test_budget_eviction_refcount_and_cancellation();
    test_short_read_checksum_and_upload_fail_closed();
    test_ram_hit_reuploads_after_vram_eviction();
    test_host_lease_protects_validated_ram_copy();
    test_cpu_executor_writes_compact_selection_outputs();
    test_deepseek_packed_executor_engages_every_worker();
    test_layer_partitioned_eviction_protects_other_layers();
    test_frequency_admission_protects_reused_expert();
    test_routing_score_temperature_breaks_frequency_ties();
    test_vram_replacement_requires_a_strictly_colder_victim();
    test_prefetch_credits_and_stale_epoch_cancel_pending_work();
    test_hybrid_dispatch_minimizes_measured_critical_path();
    test_hybrid_dispatch_ties_bounds_and_trace_are_deterministic();
    test_route_census_is_bounded_ranked_and_recoverable();
    std::cout << "expert_runtime_tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert_runtime_tests: FAIL: " << error.what() << '\n';
    return 1;
  }
}
