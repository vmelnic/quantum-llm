#include "expert/runtime/cpu/fp4_host_executor.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
#include <immintrin.h>
#define EXPERT_RUNTIME_X86_AVX2 1
#else
#define EXPERT_RUNTIME_X86_AVX2 0
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace expert::runtime::cpu {
namespace {

constexpr std::uint32_t kBlockColumns = 32U;

float decode_ue8m0(std::uint8_t code) noexcept {
  const std::uint32_t bits = code == 0U ? 0x00400000U
                                       : std::uint32_t{code} << 23U;
  return std::bit_cast<float>(bits);
}

[[maybe_unused]] std::int8_t decode_fp4_twice(std::uint8_t code) noexcept {
  const auto index = code & 0x07U;
  const auto magnitude = index <= 4U ? static_cast<int>(index)
                         : index == 5U ? 6
                         : index == 6U ? 8
                                       : 12;
  return static_cast<std::int8_t>((code & 0x08U) != 0U ? -magnitude
                                                       : magnitude);
}

#if EXPERT_RUNTIME_X86_AVX2
float horizontal_sum_f32(__m256 value) noexcept {
  auto lanes = _mm_add_ps(_mm256_castps256_ps128(value),
                          _mm256_extractf128_ps(value, 1));
  lanes = _mm_hadd_ps(lanes, lanes);
  lanes = _mm_hadd_ps(lanes, lanes);
  return _mm_cvtss_f32(lanes);
}

__m256i dot_fp4_q8_lanes(const std::uint8_t* packed,
                         const std::int8_t* activation) noexcept {
  const auto source = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(packed));
  const auto nibble_mask = _mm_set1_epi8(0x0f);
  const auto low = _mm_and_si128(source, nibble_mask);
  const auto high = _mm_and_si128(_mm_srli_epi16(source, 4), nibble_mask);
  const auto indices = _mm256_set_m128i(_mm_unpackhi_epi8(low, high),
                                        _mm_unpacklo_epi8(low, high));
  const auto decode_table = _mm256_broadcastsi128_si256(_mm_setr_epi8(
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
  const auto weights = _mm256_shuffle_epi8(decode_table, indices);
  const auto activations = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(activation));
  // vpmaddubsw accepts unsigned*signed bytes. Move the activation sign onto
  // the signed FP4 value and multiply its unsigned magnitude. Q8 is clamped
  // to [-127, 127], and a pair is bounded by 2*12*127, so the i16 pair sum
  // cannot saturate. The resulting i32 dot is bit-exact to the scalar path.
  const auto signed_weights = _mm256_sign_epi8(weights, activations);
  const auto magnitudes = _mm256_abs_epi8(activations);
  const auto pairs = _mm256_maddubs_epi16(magnitudes, signed_weights);
  return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}
#else
std::int32_t dot_fp4_q8_block(const std::uint8_t* packed,
                              const std::int8_t* activation) noexcept {
  std::int32_t sum = 0;
  for (std::uint32_t index = 0; index < kBlockColumns; ++index) {
    const auto byte = packed[index / 2U];
    const auto code = (index & 1U) == 0U ? byte & 0x0fU : byte >> 4U;
    sum += static_cast<std::int32_t>(decode_fp4_twice(code)) *
           static_cast<std::int32_t>(activation[index]);
  }
  return sum;
}
#endif

float packed_dot(const std::uint8_t* weights, const std::uint8_t* scales,
                 const std::int8_t* activation, float activation_scale,
                 std::uint32_t row, std::uint32_t columns) noexcept {
  const auto blocks = columns / kBlockColumns;
  const auto* row_weights =
      weights + static_cast<std::size_t>(row) * (columns / 2U);
  const auto* row_scales = scales + static_cast<std::size_t>(row) * blocks;
#if EXPERT_RUNTIME_X86_AVX2
  // Keep the eight partial lanes live across scale blocks and reduce once per
  // row. The old path performed three horizontal reductions for every
  // 32-value block, which serialized the hottest loop. Each i32 lane is
  // converted before applying its block's UE8M0 power-of-two scale, so the
  // quantization ABI and the mathematical dot product remain unchanged.
  auto lanes = _mm256_setzero_ps();
  for (std::uint32_t block = 0; block < blocks; ++block) {
    const auto products = dot_fp4_q8_lanes(
        row_weights + static_cast<std::size_t>(block) * 16U,
        activation + static_cast<std::size_t>(block) * 32U);
    const auto scale = _mm256_set1_ps(decode_ue8m0(row_scales[block]));
    lanes = _mm256_add_ps(
        lanes, _mm256_mul_ps(_mm256_cvtepi32_ps(products), scale));
  }
  const auto total = horizontal_sum_f32(lanes);
#else
  float total = 0.0F;
  for (std::uint32_t block = 0; block < blocks; ++block) {
    total += static_cast<float>(dot_fp4_q8_block(
                 row_weights + static_cast<std::size_t>(block) * 16U,
                 activation + static_cast<std::size_t>(block) * 32U)) *
             decode_ue8m0(row_scales[block]);
  }
#endif
  return total * activation_scale * 0.5F;
}

float round_bf16(float value) noexcept {
  auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) return value;
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(bits & 0xffff0000U);
}

float quantize_q8(std::span<const float> source,
                  std::span<std::int8_t> destination) noexcept {
  float maximum = 0.0F;
  for (const auto value : source) maximum = std::max(maximum, std::abs(value));
  const auto scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
  for (std::size_t index = 0; index < source.size(); ++index) {
    const auto rounded = static_cast<int>(std::nearbyint(source[index] / scale));
    destination[index] = static_cast<std::int8_t>(
        std::clamp(rounded, -127, 127));
  }
  return scale;
}

template <typename T>
const T* section(std::span<const std::byte> record,
                 std::uint64_t offset) noexcept {
  return reinterpret_cast<const T*>(record.data() + offset);
}

Status validate_group(const Fp4HostWorkGroup& group,
                      std::uint32_t rows, std::uint32_t top_k,
                      std::size_t input_values,
                      std::size_t output_values) {
  if (group.record_bytes.empty() || group.selections.empty() ||
      group.hidden == 0U || group.intermediate == 0U ||
      group.hidden % kBlockColumns != 0U ||
      group.intermediate % kBlockColumns != 0U ||
      group.selections.size() >
          static_cast<std::size_t>(rows) * top_k ||
      group.output_slots.size() != group.selections.size() ||
      input_values < static_cast<std::size_t>(rows) * group.hidden ||
      output_values < group.hidden || output_values % group.hidden != 0U) {
    return {ErrorCode::invalid_argument,
            "invalid FP4 host work group"};
  }
  const auto matrix_bytes = [](std::uint32_t output,
                               std::uint32_t input) -> std::uint64_t {
    return static_cast<std::uint64_t>(output) * input / 2U;
  };
  const auto scale_bytes = [](std::uint32_t output,
                              std::uint32_t input) -> std::uint64_t {
    return static_cast<std::uint64_t>(output) * input / kBlockColumns;
  };
  const auto& s = group.sections;
  if (s.w1_weight_bytes != matrix_bytes(group.intermediate, group.hidden) ||
      s.w3_weight_bytes != matrix_bytes(group.intermediate, group.hidden) ||
      s.w2_weight_bytes != matrix_bytes(group.hidden, group.intermediate) ||
      s.w1_scale_bytes != scale_bytes(group.intermediate, group.hidden) ||
      s.w3_scale_bytes != scale_bytes(group.intermediate, group.hidden) ||
      s.w2_scale_bytes != scale_bytes(group.hidden, group.intermediate)) {
    return {ErrorCode::invalid_argument,
            "FP4 host section geometry mismatch"};
  }
  const auto within = [&](std::uint64_t offset, std::uint64_t bytes) {
    return offset <= group.record_bytes.size() &&
           bytes <= group.record_bytes.size() - offset;
  };
  if (!within(s.w1_weight_offset, s.w1_weight_bytes) ||
      !within(s.w1_scale_offset, s.w1_scale_bytes) ||
      !within(s.w3_weight_offset, s.w3_weight_bytes) ||
      !within(s.w3_scale_offset, s.w3_scale_bytes) ||
      !within(s.w2_weight_offset, s.w2_weight_bytes) ||
      !within(s.w2_scale_offset, s.w2_scale_bytes)) {
    return {ErrorCode::invalid_argument,
            "FP4 host section exceeds record"};
  }
  for (const auto selection : group.selections) {
    if (selection >= rows * top_k)
      return {ErrorCode::invalid_argument,
              "FP4 host selection exceeds microbatch"};
  }
  for (const auto slot : group.output_slots) {
    if (slot >= output_values / group.hidden)
      return {ErrorCode::invalid_argument,
              "FP4 host output slot exceeds output"};
  }
  return Status::success();
}

}  // namespace

struct Fp4HostExecutor::Impl final {
  enum class Phase : std::uint8_t { gate_up, down };

  struct Batch final {
    std::span<const Fp4HostWorkGroup> groups;
    const float* inputs{};
    float* outputs{};
    std::uint32_t rows{};
    std::uint32_t top_k{};
    std::vector<std::size_t> offsets;
    std::vector<std::int8_t> q_inputs;
    std::vector<float> q_input_scales;
    std::vector<float> intermediate;
    std::vector<std::int8_t> q_intermediate;
    std::vector<float> q_intermediate_scales;
  };

  struct WorkItem final {
    std::size_t group{};
    std::uint32_t begin{};
    std::uint32_t end{};
  };

  explicit Impl(Fp4HostExecutorConfig value) : config(value) {
    if (config.maximum_threads == 0U || config.maximum_threads > 64U ||
        config.gate_chunk == 0U || config.down_chunk == 0U ||
        config.swiglu_limit < 0.0F)
      throw std::invalid_argument(
          "invalid FP4 host executor configuration");
    metrics.maximum_threads = config.maximum_threads;
    workers.reserve(config.maximum_threads);
    for (std::uint32_t index = 0; index < config.maximum_threads; ++index)
      workers.emplace_back([this, index] { worker_loop(index); });
  }

  ~Impl() {
    {
      std::lock_guard lock(work_mutex);
      stopping = true;
      ++generation;
    }
    work_ready.notify_all();
    for (auto& worker : workers) worker.join();
  }

  void run_gate(Batch& batch, const WorkItem& item) const noexcept {
    const auto& group = batch.groups[item.group];
    const auto& s = group.sections;
    const auto* w1 = section<std::uint8_t>(group.record_bytes,
                                           s.w1_weight_offset);
    const auto* w1_scales = section<std::uint8_t>(group.record_bytes,
                                                  s.w1_scale_offset);
    const auto* w3 = section<std::uint8_t>(group.record_bytes,
                                           s.w3_weight_offset);
    const auto* w3_scales = section<std::uint8_t>(group.record_bytes,
                                                  s.w3_scale_offset);
    for (std::uint32_t output = item.begin; output < item.end; ++output) {
      for (std::size_t selected = 0; selected < group.selections.size();
           ++selected) {
        const auto row = group.selections[selected] / batch.top_k;
        const auto* q = batch.q_inputs.data() +
                        static_cast<std::size_t>(row) * group.hidden;
        auto gate = packed_dot(w1, w1_scales, q,
                               batch.q_input_scales[row], output,
                               group.hidden);
        auto up = packed_dot(w3, w3_scales, q,
                             batch.q_input_scales[row], output,
                             group.hidden);
        if (config.swiglu_limit > 0.0F) {
          gate = std::min(gate, config.swiglu_limit);
          up = std::clamp(up, -config.swiglu_limit, config.swiglu_limit);
        }
        auto value = (gate / (1.0F + std::exp(-gate))) * up;
        if (config.bf16_intermediate) value = round_bf16(value);
        batch.intermediate[batch.offsets[item.group] +
                           selected * group.intermediate + output] = value;
      }
    }
  }

  void run_down(Batch& batch, const WorkItem& item) const noexcept {
    const auto& group = batch.groups[item.group];
    const auto& s = group.sections;
    const auto* w2 = section<std::uint8_t>(group.record_bytes,
                                           s.w2_weight_offset);
    const auto* w2_scales = section<std::uint8_t>(group.record_bytes,
                                                  s.w2_scale_offset);
    for (std::uint32_t output = item.begin; output < item.end; ++output) {
      for (std::size_t selected = 0; selected < group.selections.size();
           ++selected) {
        const auto offset = batch.offsets[item.group] +
                            selected * group.intermediate;
        const auto value = packed_dot(
            w2, w2_scales, batch.q_intermediate.data() + offset,
            batch.q_intermediate_scales[
                batch.offsets[item.group] / group.intermediate + selected],
            output, group.intermediate);
        batch.outputs[static_cast<std::size_t>(group.output_slots[selected]) *
                          group.hidden +
                      output] = value;
      }
    }
  }

  void worker_loop(std::uint32_t worker_index) noexcept {
#if defined(_WIN32)
    if (config.pin_windows_threads && worker_index < 64U) {
      static_cast<void>(SetThreadAffinityMask(
          GetCurrentThread(), DWORD_PTR{1} << worker_index));
    }
#endif
    std::uint64_t seen_generation = 0U;
    for (;;) {
      std::unique_lock lock(work_mutex);
      work_ready.wait(lock, [&] {
        return stopping || generation != seen_generation;
      });
      if (stopping) return;
      seen_generation = generation;
      auto* batch = active_batch;
      const auto active = active_threads;
      const auto current_phase = phase;
      const auto count = work_items.size();
      lock.unlock();
      if (worker_index < active) {
        worker_mask.fetch_or(std::uint64_t{1} << worker_index,
                             std::memory_order_relaxed);
        for (std::size_t index = worker_index; index < count; index += active) {
          if (current_phase == Phase::gate_up)
            run_gate(*batch, work_items[index]);
          else
            run_down(*batch, work_items[index]);
        }
        lock.lock();
        if (--remaining_workers == 0U) phase_done.notify_one();
      }
    }
  }

  Status dispatch(Batch& batch, Phase selected_phase) {
    work_items.clear();
    const auto chunk = selected_phase == Phase::gate_up
                           ? config.gate_chunk
                           : config.down_chunk;
    for (std::size_t group_index = 0; group_index < batch.groups.size();
         ++group_index) {
      const auto values = selected_phase == Phase::gate_up
                              ? batch.groups[group_index].intermediate
                              : batch.groups[group_index].hidden;
      for (std::uint32_t begin = 0; begin < values; begin += chunk) {
        work_items.push_back(
            {group_index, begin, std::min(values, begin + chunk)});
      }
    }
    if (work_items.empty()) return Status::success();
    std::unique_lock lock(work_mutex);
    active_batch = &batch;
    phase = selected_phase;
    active_threads = std::min<std::uint32_t>(
        config.maximum_threads,
        static_cast<std::uint32_t>(work_items.size()));
    remaining_workers = active_threads;
    worker_mask.store(0U, std::memory_order_relaxed);
    ++generation;
    work_ready.notify_all();
    phase_done.wait(lock, [&] { return remaining_workers == 0U; });
    metrics.worker_mask_last = worker_mask.load(std::memory_order_relaxed);
    metrics.workers_used_last =
        static_cast<std::uint32_t>(std::popcount(metrics.worker_mask_last));
    if (metrics.workers_used_last != active_threads) {
      return {ErrorCode::internal,
              "FP4 host executor did not engage every worker"};
    }
    return Status::success();
  }

  Status execute(std::span<const Fp4HostWorkGroup> groups,
                 std::span<const float> inputs, std::uint32_t rows,
                 std::uint32_t top_k, std::span<float> outputs) {
    if (groups.empty()) return Status::success();
    if (rows == 0U || top_k == 0U || inputs.empty() || outputs.empty())
      return {ErrorCode::invalid_argument,
              "invalid FP4 host batch"};
    std::lock_guard execution_lock(execution_mutex);
    auto& batch = scratch;
    batch.groups = groups;
    batch.inputs = inputs.data();
    batch.outputs = outputs.data();
    batch.rows = rows;
    batch.top_k = top_k;
    batch.offsets.clear();
    const auto hidden = groups.front().hidden;
    const auto intermediate_width = groups.front().intermediate;
    std::vector<bool> claimed(outputs.size() / hidden, false);
    std::size_t intermediate_values = 0U;
    std::uint64_t selection_count = 0U;
    std::uint64_t weight_bytes = 0U;
    for (const auto& group : groups) {
      if (group.hidden != hidden || group.intermediate != intermediate_width)
        return {ErrorCode::invalid_argument,
                "FP4 host groups disagree on geometry"};
      auto status = validate_group(group, rows, top_k, inputs.size(),
                                   outputs.size());
      if (!status.ok()) return status;
      for (const auto slot : group.output_slots) {
        if (claimed[slot])
          return {ErrorCode::invalid_argument,
                  "duplicate FP4 host output slot"};
        claimed[slot] = true;
      }
      batch.offsets.push_back(intermediate_values);
      intermediate_values += group.selections.size() * group.intermediate;
      selection_count += group.selections.size();
      weight_bytes += group.sections.w1_weight_bytes +
                      group.sections.w1_scale_bytes +
                      group.sections.w3_weight_bytes +
                      group.sections.w3_scale_bytes +
                      group.sections.w2_weight_bytes +
                      group.sections.w2_scale_bytes;
    }
    batch.q_inputs.resize(static_cast<std::size_t>(rows) * hidden);
    batch.q_input_scales.resize(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
      batch.q_input_scales[row] = quantize_q8(
          inputs.subspan(static_cast<std::size_t>(row) * hidden, hidden),
          std::span(batch.q_inputs).subspan(
              static_cast<std::size_t>(row) * hidden, hidden));
    }
    batch.intermediate.resize(intermediate_values);
    batch.q_intermediate.resize(intermediate_values);
    batch.q_intermediate_scales.resize(selection_count);
    const auto started = std::chrono::steady_clock::now();
    auto status = dispatch(batch, Phase::gate_up);
    if (!status.ok()) return status;
    std::size_t scale_slot = 0U;
    for (std::size_t group_index = 0U; group_index < groups.size();
         ++group_index) {
      const auto& group = groups[group_index];
      for (std::size_t selected = 0U; selected < group.selections.size();
           ++selected, ++scale_slot) {
        const auto offset = batch.offsets[group_index] +
                            selected * group.intermediate;
        batch.q_intermediate_scales[scale_slot] = quantize_q8(
            std::span(batch.intermediate).subspan(offset, group.intermediate),
            std::span(batch.q_intermediate).subspan(offset,
                                                    group.intermediate));
      }
    }
    status = dispatch(batch, Phase::down);
    if (!status.ok()) return status;
    ++metrics.execute_calls;
    metrics.selections += selection_count;
    metrics.source_weight_bytes += weight_bytes;
    metrics.compute_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    return Status::success();
  }

  Fp4HostExecutorTelemetry telemetry() const noexcept {
    std::lock_guard lock(execution_mutex);
    return metrics;
  }

  Fp4HostExecutorConfig config;
  mutable std::mutex execution_mutex;
  Fp4HostExecutorTelemetry metrics;
  std::vector<std::thread> workers;
  std::mutex work_mutex;
  std::condition_variable work_ready;
  std::condition_variable phase_done;
  std::vector<WorkItem> work_items;
  Batch* active_batch{};
  Phase phase{Phase::gate_up};
  std::uint32_t active_threads{};
  std::uint32_t remaining_workers{};
  std::uint64_t generation{};
  std::atomic<std::uint64_t> worker_mask{};
  Batch scratch;
  bool stopping{};
};

Fp4HostExecutor::Fp4HostExecutor(Fp4HostExecutorConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

Fp4HostExecutor::~Fp4HostExecutor() = default;

Status Fp4HostExecutor::execute(
    std::span<const Fp4HostWorkGroup> groups,
    std::span<const float> inputs, std::uint32_t rows, std::uint32_t top_k,
    std::span<float> selection_outputs) {
  return impl_->execute(groups, inputs, rows, top_k, selection_outputs);
}

Fp4HostExecutorTelemetry Fp4HostExecutor::telemetry()
    const noexcept {
  return impl_->telemetry();
}

}  // namespace expert::runtime::cpu
