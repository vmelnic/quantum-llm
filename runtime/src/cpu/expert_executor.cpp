#include "expert/runtime/cpu/expert_executor.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
#include <immintrin.h>
#define EXPERT_RUNTIME_X86_AVX2 1
#else
#define EXPERT_RUNTIME_X86_AVX2 0
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace expert::runtime::cpu {
namespace {

constexpr std::uint32_t kMaximumRowsPerExpert = 32;

#if EXPERT_RUNTIME_X86_AVX2
float horizontal_sum(__m256 value) noexcept {
  const auto low = _mm256_castps256_ps128(value);
  const auto high = _mm256_extractf128_ps(value, 1);
  auto sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

__m256 load_i8x8(const std::int8_t* source) noexcept {
  const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source));
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
}
#endif

template <typename T>
const T* section(std::span<const std::byte> record, std::uint64_t offset) {
  return reinterpret_cast<const T*>(record.data() + offset);
}

Status validate_group(const ExpertWorkGroup& group, std::uint32_t rows,
                      std::uint32_t top_k, std::size_t input_values,
                      std::size_t output_values) {
  const auto& s = group.sections;
  if (group.record_bytes.empty() || group.selections.empty() || s.hidden == 0 ||
      s.intermediate == 0 || (s.hidden % 8U) != 0 ||
      (s.intermediate % 8U) != 0 ||
      group.selections.size() > kMaximumRowsPerExpert ||
      group.output_slots.size() != group.selections.size() ||
      input_values < static_cast<std::size_t>(rows) * s.hidden ||
      output_values < s.hidden || (output_values % s.hidden) != 0) {
    return {ErrorCode::invalid_argument, "invalid CPU expert work group"};
  }
  const auto within = [&](std::uint64_t offset, std::uint64_t bytes) {
    return offset <= group.record_bytes.size() &&
           bytes <= group.record_bytes.size() - offset;
  };
  if (!within(s.gate_up_q_offset, s.gate_up_q_bytes) ||
      !within(s.gate_up_scale_offset, s.gate_up_scale_bytes) ||
      !within(s.down_q_offset, s.down_q_bytes) ||
      !within(s.down_scale_offset, s.down_scale_bytes)) {
    return {ErrorCode::invalid_argument, "CPU expert section exceeds record"};
  }
  for (const auto selection : group.selections) {
    if (selection >= rows * top_k) {
      return {ErrorCode::invalid_argument,
              "CPU expert selection exceeds microbatch"};
    }
  }
  for (const auto output_slot : group.output_slots) {
    if (output_slot >= output_values / s.hidden) {
      return {ErrorCode::invalid_argument,
              "CPU expert compact output slot exceeds output"};
    }
  }
  return Status::success();
}

void run_gate_range(const ExpertWorkGroup& group, const float* inputs,
                    std::uint32_t top_k, float* intermediate,
                    std::uint32_t begin, std::uint32_t end) {
  const auto& s = group.sections;
  const auto count = static_cast<std::uint32_t>(group.selections.size());
  const auto* gate_up = section<std::int8_t>(group.record_bytes,
                                             s.gate_up_q_offset);
  const auto* gate_scales = section<float>(group.record_bytes,
                                            s.gate_up_scale_offset);
#if EXPERT_RUNTIME_X86_AVX2
  std::array<const float*, kMaximumRowsPerExpert> input_rows{};
  for (std::uint32_t item = 0; item < count; ++item) {
    const auto row = group.selections[item] / top_k;
    input_rows[item] = inputs + static_cast<std::size_t>(row) * s.hidden;
  }
  std::array<__m256, kMaximumRowsPerExpert> gate_sum{};
  std::array<__m256, kMaximumRowsPerExpert> up_sum{};
  for (std::uint32_t output = begin; output < end; ++output) {
    std::fill_n(gate_sum.begin(), count, _mm256_setzero_ps());
    std::fill_n(up_sum.begin(), count, _mm256_setzero_ps());
    const auto* gate = gate_up + static_cast<std::size_t>(output) * s.hidden;
    const auto* up = gate_up +
                     static_cast<std::size_t>(s.intermediate + output) *
                         s.hidden;
    for (std::uint32_t column = 0; column < s.hidden; column += 8) {
      if (column + 64U < s.hidden) {
        _mm_prefetch(reinterpret_cast<const char*>(gate + column + 64U),
                     _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(up + column + 64U),
                     _MM_HINT_T0);
      }
      const auto gate_weight = load_i8x8(gate + column);
      const auto up_weight = load_i8x8(up + column);
      for (std::uint32_t item = 0; item < count; ++item) {
        if (column + 64U < s.hidden)
          _mm_prefetch(reinterpret_cast<const char*>(input_rows[item] +
                                                     column + 64U),
                       _MM_HINT_T0);
        const auto activation =
            _mm256_loadu_ps(input_rows[item] + column);
        gate_sum[item] =
            _mm256_fmadd_ps(gate_weight, activation, gate_sum[item]);
        up_sum[item] =
            _mm256_fmadd_ps(up_weight, activation, up_sum[item]);
      }
    }
    for (std::uint32_t item = 0; item < count; ++item) {
      const auto gate_value =
          horizontal_sum(gate_sum[item]) * gate_scales[output];
      const auto up_value = horizontal_sum(up_sum[item]) *
                            gate_scales[s.intermediate + output];
      intermediate[static_cast<std::size_t>(item) * s.intermediate + output] =
          (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
    }
  }
#else
  for (std::uint32_t item = 0; item < count; ++item) {
    const auto row = group.selections[item] / top_k;
    const auto* input = inputs + static_cast<std::size_t>(row) * s.hidden;
    auto* item_intermediate =
        intermediate + static_cast<std::size_t>(item) * s.intermediate;
    for (std::uint32_t output = begin; output < end; ++output) {
      double gate_sum_scalar = 0.0;
      double up_value = 0.0;
      const auto* gate_weights =
          gate_up + static_cast<std::size_t>(output) * s.hidden;
      const auto* up = gate_up +
                       static_cast<std::size_t>(s.intermediate + output) *
                           s.hidden;
      for (std::uint32_t column = 0; column < s.hidden; ++column) {
        gate_sum_scalar +=
            static_cast<float>(gate_weights[column]) * input[column];
        up_value += static_cast<float>(up[column]) * input[column];
      }
      const auto scaled_gate =
          static_cast<float>(gate_sum_scalar) * gate_scales[output];
      const auto scaled_up = static_cast<float>(up_value) *
                             gate_scales[s.intermediate + output];
      item_intermediate[output] =
          (scaled_gate / (1.0F + std::exp(-scaled_gate))) * scaled_up;
    }
  }
#endif
}

void run_down_range(const ExpertWorkGroup& group, std::uint32_t top_k,
                    const float* intermediate, float* selection_outputs,
                    std::uint32_t begin, std::uint32_t end) {
  const auto& s = group.sections;
  const auto count = static_cast<std::uint32_t>(group.selections.size());
  const auto* down = section<std::int8_t>(group.record_bytes, s.down_q_offset);
  const auto* down_scales =
      section<float>(group.record_bytes, s.down_scale_offset);
#if EXPERT_RUNTIME_X86_AVX2
  std::array<const float*, kMaximumRowsPerExpert> intermediate_rows{};
  std::array<float*, kMaximumRowsPerExpert> output_rows{};
  for (std::uint32_t item = 0; item < count; ++item) {
    intermediate_rows[item] =
        intermediate + static_cast<std::size_t>(item) * s.intermediate;
    output_rows[item] =
        selection_outputs +
        static_cast<std::size_t>(group.output_slots[item]) * s.hidden;
  }
  std::array<__m256, kMaximumRowsPerExpert> down_sum{};
  for (std::uint32_t output = begin; output < end; ++output) {
    std::fill_n(down_sum.begin(), count, _mm256_setzero_ps());
    const auto* weights =
        down + static_cast<std::size_t>(output) * s.intermediate;
    for (std::uint32_t column = 0; column < s.intermediate; column += 8) {
      if (column + 64U < s.intermediate)
        _mm_prefetch(reinterpret_cast<const char*>(weights + column + 64U),
                     _MM_HINT_T0);
      const auto weight = load_i8x8(weights + column);
      for (std::uint32_t item = 0; item < count; ++item) {
        if (column + 64U < s.intermediate)
          _mm_prefetch(reinterpret_cast<const char*>(intermediate_rows[item] +
                                                     column + 64U),
                       _MM_HINT_T0);
        const auto activation =
            _mm256_loadu_ps(intermediate_rows[item] + column);
        down_sum[item] =
            _mm256_fmadd_ps(weight, activation, down_sum[item]);
      }
    }
    for (std::uint32_t item = 0; item < count; ++item) {
      output_rows[item][output] =
          horizontal_sum(down_sum[item]) * down_scales[output];
    }
  }
#else
  for (std::uint32_t item = 0; item < count; ++item) {
    const auto* item_intermediate =
        intermediate + static_cast<std::size_t>(item) * s.intermediate;
    auto* item_output = selection_outputs +
                        static_cast<std::size_t>(group.output_slots[item]) *
                            s.hidden;
    for (std::uint32_t output = begin; output < end; ++output) {
      double sum = 0.0;
      const auto* weights =
          down + static_cast<std::size_t>(output) * s.intermediate;
      for (std::uint32_t column = 0; column < s.intermediate; ++column)
        sum += static_cast<float>(weights[column]) * item_intermediate[column];
      item_output[output] = static_cast<float>(sum) * down_scales[output];
    }
  }
#endif
  (void)top_k;
}

}  // namespace

struct ExpertExecutor::Impl final {
  struct Batch final {
    const float* inputs{};
    float* outputs{};
    std::uint32_t rows{};
    std::uint32_t top_k{};
    std::size_t input_values{};
    std::size_t output_values{};
    std::span<const ExpertWorkGroup> groups;
    std::span<const std::size_t> intermediate_offsets;
    std::span<float> intermediate;
    std::mutex mutex;
    std::condition_variable ready;
    std::size_t remaining{};
    Status status = Status::success();
  };

  struct Job final {
    enum class Phase { gate_up, down } phase{Phase::gate_up};
    std::size_t group{};
    std::uint32_t begin{};
    std::uint32_t end{};
    Batch* batch{};
  };

  struct Tuning final {
    std::uint32_t threads{};
    std::uint32_t gate_chunk{64};
    std::uint32_t down_chunk{128};
  };

  explicit Impl(ExpertExecutorConfig settings) : config(settings) {
    if (config.maximum_threads == 0 || config.calibration_budget_ms == 0)
      throw std::invalid_argument("invalid CPU expert executor configuration");
    tuning.threads = config.maximum_threads;
    active_threads = config.maximum_threads;
    metrics.maximum_threads = config.maximum_threads;
    metrics.selected_threads = tuning.threads;
    metrics.gate_chunk = tuning.gate_chunk;
    metrics.down_chunk = tuning.down_chunk;
    workers.reserve(config.maximum_threads);
    for (std::uint32_t index = 0; index < config.maximum_threads; ++index) {
      workers.emplace_back([this, index] { worker_loop(index); });
    }
  }

  ~Impl() {
    {
      std::lock_guard lock(queue_mutex);
      stopping = true;
    }
    queue_ready.notify_all();
    for (auto& worker : workers) worker.join();
  }

  void worker_loop(std::uint32_t worker_index) noexcept {
    for (;;) {
      Job job;
      {
        std::unique_lock lock(queue_mutex);
        queue_ready.wait(lock, [this, worker_index] {
          return stopping ||
                 (worker_index < active_threads && !jobs.empty());
        });
        if (stopping && jobs.empty()) return;
        job = std::move(jobs.front());
        jobs.pop_front();
      }
      Status status;
      try {
        const auto& group = job.batch->groups[job.group];
        auto* intermediate_pointer =
            job.batch->intermediate.data() +
            job.batch->intermediate_offsets[job.group];
        if (job.phase == Job::Phase::gate_up) {
          run_gate_range(group, job.batch->inputs, job.batch->top_k,
                         intermediate_pointer, job.begin, job.end);
        } else {
          run_down_range(group, job.batch->top_k, intermediate_pointer,
                         job.batch->outputs, job.begin, job.end);
        }
      } catch (const std::exception& error) {
        status = {ErrorCode::internal,
                  std::string("CPU expert worker failed: ") + error.what()};
      } catch (...) {
        status = {ErrorCode::internal, "CPU expert worker failed"};
      }
      {
        std::lock_guard lock(job.batch->mutex);
        if (!status.ok() && job.batch->status.ok()) {
          job.batch->status = std::move(status);
        }
        if (--job.batch->remaining == 0) job.batch->ready.notify_one();
      }
    }
  }

  Status dispatch_phase(Batch& batch, Job::Phase phase,
                        const Tuning& selected) {
    const auto chunk = phase == Job::Phase::gate_up ? selected.gate_chunk
                                                    : selected.down_chunk;
    std::size_t job_count = 0;
    for (const auto& group : batch.groups) {
      const auto values = phase == Job::Phase::gate_up
                              ? group.sections.intermediate
                              : group.sections.hidden;
      job_count += (values + chunk - 1U) / chunk;
    }
    {
      std::lock_guard lock(batch.mutex);
      batch.remaining = job_count;
    }
    {
      std::lock_guard lock(queue_mutex);
      if (stopping) return {ErrorCode::cancelled, "CPU executor is stopping"};
      active_threads = selected.threads;
      for (std::size_t group_index = 0;
           group_index < batch.groups.size(); ++group_index) {
        const auto& group = batch.groups[group_index];
        const auto values = phase == Job::Phase::gate_up
                                ? group.sections.intermediate
                                : group.sections.hidden;
        for (std::uint32_t begin = 0; begin < values; begin += chunk) {
          jobs.push_back({phase, group_index, begin,
                          std::min(values, begin + chunk), &batch});
        }
      }
    }
    queue_ready.notify_all();
    std::unique_lock lock(batch.mutex);
    batch.ready.wait(lock, [&] { return batch.remaining == 0; });
    return batch.status;
  }

  Status execute_once(Batch& batch, const Tuning& selected) {
    batch.status = Status::success();
    auto status = dispatch_phase(batch, Job::Phase::gate_up, selected);
    if (!status.ok()) return status;
    return dispatch_phase(batch, Job::Phase::down, selected);
  }

  Status execute(std::span<const ExpertWorkGroup> groups,
                 std::span<const float> inputs, std::uint32_t rows,
                 std::uint32_t top_k, std::span<float> outputs) {
    if (groups.empty()) return Status::success();
    if (rows == 0 || top_k == 0 || inputs.empty() || outputs.empty()) {
      return {ErrorCode::invalid_argument, "invalid CPU expert batch"};
    }
    std::lock_guard execution_lock(execution_mutex);
    Batch batch;
    batch.inputs = inputs.data();
    batch.outputs = outputs.data();
    batch.rows = rows;
    batch.top_k = top_k;
    batch.input_values = inputs.size();
    batch.output_values = outputs.size();
    batch.groups = groups;
    intermediate_offsets.clear();
    intermediate_offsets.reserve(groups.size());
    if (groups.front().sections.hidden == 0) {
      return {ErrorCode::invalid_argument,
              "CPU expert group has zero hidden size"};
    }
    claimed_outputs.assign(outputs.size() / groups.front().sections.hidden,
                           false);
    std::size_t intermediate_values = 0;
    std::uint64_t selection_count = 0;
    std::uint64_t weight_bytes = 0;
    for (const auto& group : groups) {
      if (group.sections.hidden != batch.groups.front().sections.hidden) {
        return {ErrorCode::invalid_argument,
                "CPU expert groups disagree on hidden size"};
      }
      const auto status = validate_group(
          group, rows, top_k, inputs.size(), outputs.size());
      if (!status.ok()) return status;
      for (const auto output_slot : group.output_slots) {
        if (claimed_outputs[output_slot]) {
          return {ErrorCode::invalid_argument,
                  "duplicate CPU expert compact output slot"};
        }
        claimed_outputs[output_slot] = true;
      }
      intermediate_offsets.push_back(intermediate_values);
      intermediate_values +=
          group.selections.size() * group.sections.intermediate;
      selection_count += group.selections.size();
      weight_bytes += group.sections.gate_up_q_bytes +
                      group.sections.gate_up_scale_bytes +
                      group.sections.down_q_bytes +
                      group.sections.down_scale_bytes;
    }
    intermediate.resize(intermediate_values);
    batch.intermediate_offsets =
        std::span<const std::size_t>(intermediate_offsets);
    batch.intermediate = std::span<float>(intermediate);

    const auto logical_started = std::chrono::steady_clock::now();
    Status status;
    if (!calibrated && config.enable_autotune) {
      const auto calibration_started = std::chrono::steady_clock::now();
      const auto deadline = calibration_started +
          std::chrono::milliseconds(config.calibration_budget_ms);
      std::vector<Tuning> candidates;
      const auto physical_guess =
          std::max(1U, config.maximum_threads / 2U);
      for (const auto threads : {physical_guess, config.maximum_threads}) {
        for (const auto tiles : {std::pair{64U, 128U},
                                 std::pair{128U, 256U}}) {
          const Tuning candidate{threads, tiles.first, tiles.second};
          if (std::find_if(candidates.begin(), candidates.end(),
                           [&](const Tuning& value) {
                             return value.threads == candidate.threads &&
                                    value.gate_chunk == candidate.gate_chunk &&
                                    value.down_chunk == candidate.down_chunk;
                           }) == candidates.end()) {
            candidates.push_back(candidate);
          }
        }
      }
      auto best_ns = std::numeric_limits<std::uint64_t>::max();
      for (const auto& candidate : candidates) {
        if (metrics.calibration_runs != 0 &&
            std::chrono::steady_clock::now() >= deadline)
          break;
        const auto started = std::chrono::steady_clock::now();
        status = execute_once(batch, candidate);
        if (!status.ok()) return status;
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        ++metrics.calibration_runs;
        if (elapsed < best_ns) {
          best_ns = elapsed;
          tuning = candidate;
        }
      }
      metrics.calibration_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - calibration_started).count());
      calibrated = true;
      metrics.selected_threads = tuning.threads;
      metrics.gate_chunk = tuning.gate_chunk;
      metrics.down_chunk = tuning.down_chunk;
    } else {
      status = execute_once(batch, tuning);
      if (!status.ok()) return status;
      calibrated = true;
    }
    const auto logical_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - logical_started).count());
    ++metrics.execute_calls;
    metrics.selections += selection_count;
    metrics.effective_weight_bytes += weight_bytes;
    metrics.compute_ns += logical_ns;
    return Status::success();
  }

  ExpertExecutorTelemetry telemetry() const noexcept {
    std::lock_guard lock(execution_mutex);
    return metrics;
  }

  ExpertExecutorConfig config;
  ExpertExecutorTelemetry metrics;
  Tuning tuning;
  bool calibrated{};
  mutable std::mutex execution_mutex;
  std::vector<std::size_t> intermediate_offsets;
  std::vector<float> intermediate;
  std::vector<bool> claimed_outputs;
  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  std::deque<Job> jobs;
  std::vector<std::thread> workers;
  std::uint32_t active_threads{};
  bool stopping{};
};

ExpertExecutor::ExpertExecutor(std::uint32_t thread_count)
    : ExpertExecutor(ExpertExecutorConfig{thread_count, true, 250}) {}

ExpertExecutor::ExpertExecutor(ExpertExecutorConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

ExpertExecutor::~ExpertExecutor() = default;

Status ExpertExecutor::execute(std::span<const ExpertWorkGroup> groups,
                               std::span<const float> inputs,
                               std::uint32_t rows, std::uint32_t top_k,
                               std::span<float> selection_outputs) {
  return impl_->execute(groups, inputs, rows, top_k, selection_outputs);
}

ExpertExecutorTelemetry ExpertExecutor::telemetry() const noexcept {
  return impl_->telemetry();
}

}  // namespace expert::runtime::cpu
