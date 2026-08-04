#include "expert/runtime/cpu/expert_executor.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
#include <immintrin.h>
#define EXPERT_RUNTIME_X86_AVX2 1
#else
#define EXPERT_RUNTIME_X86_AVX2 0
#endif

#include <algorithm>
#include <array>
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
      input_values < static_cast<std::size_t>(rows) * s.hidden ||
      output_values < static_cast<std::size_t>(rows) * top_k * s.hidden) {
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
      const auto gate_weight = load_i8x8(gate + column);
      const auto up_weight = load_i8x8(up + column);
      for (std::uint32_t item = 0; item < count; ++item) {
        const auto row = group.selections[item] / top_k;
        const auto activation = _mm256_loadu_ps(
            inputs + static_cast<std::size_t>(row) * s.hidden + column);
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
      double gate = 0.0;
      double up_value = 0.0;
      const auto* gate =
          gate_up + static_cast<std::size_t>(output) * s.hidden;
      const auto* up = gate_up +
                       static_cast<std::size_t>(s.intermediate + output) *
                           s.hidden;
      for (std::uint32_t column = 0; column < s.hidden; ++column) {
        gate += static_cast<float>(gate[column]) * input[column];
        up_value += static_cast<float>(up[column]) * input[column];
      }
      const auto scaled_gate = static_cast<float>(gate) * gate_scales[output];
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
  std::array<__m256, kMaximumRowsPerExpert> down_sum{};
  for (std::uint32_t output = begin; output < end; ++output) {
    std::fill_n(down_sum.begin(), count, _mm256_setzero_ps());
    const auto* weights =
        down + static_cast<std::size_t>(output) * s.intermediate;
    for (std::uint32_t column = 0; column < s.intermediate; column += 8) {
      const auto weight = load_i8x8(weights + column);
      for (std::uint32_t item = 0; item < count; ++item) {
        const auto activation = _mm256_loadu_ps(
            intermediate + static_cast<std::size_t>(item) * s.intermediate +
            column);
        down_sum[item] =
            _mm256_fmadd_ps(weight, activation, down_sum[item]);
      }
    }
    for (std::uint32_t item = 0; item < count; ++item) {
      const auto selection = group.selections[item];
      selection_outputs[static_cast<std::size_t>(selection) * s.hidden +
                        output] =
          horizontal_sum(down_sum[item]) * down_scales[output];
    }
  }
#else
  for (std::uint32_t item = 0; item < count; ++item) {
    const auto* item_intermediate =
        intermediate + static_cast<std::size_t>(item) * s.intermediate;
    auto* item_output = selection_outputs +
                        static_cast<std::size_t>(group.selections[item]) *
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
    std::vector<ExpertWorkGroup> groups;
    std::vector<std::size_t> intermediate_offsets;
    std::vector<float> intermediate;
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

  explicit Impl(std::uint32_t count) {
    if (count == 0) throw std::invalid_argument("CPU expert thread count is zero");
    workers.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
      (void)index;
      workers.emplace_back([this] { worker_loop(); });
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

  void worker_loop() noexcept {
    for (;;) {
      Job job;
      {
        std::unique_lock lock(queue_mutex);
        queue_ready.wait(lock, [this] { return stopping || !jobs.empty(); });
        if (stopping && jobs.empty()) return;
        job = std::move(jobs.front());
        jobs.pop_front();
      }
      Status status;
      try {
        const auto& group = job.batch->groups.at(job.group);
        auto* intermediate = job.batch->intermediate.data() +
                             job.batch->intermediate_offsets.at(job.group);
        if (job.phase == Job::Phase::gate_up) {
          run_gate_range(group, job.batch->inputs, job.batch->top_k,
                         intermediate, job.begin, job.end);
        } else {
          run_down_range(group, job.batch->top_k, intermediate,
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

  Status dispatch_phase(Batch& batch, Job::Phase phase) {
    constexpr std::uint32_t gate_chunk = 64;
    constexpr std::uint32_t down_chunk = 128;
    const auto chunk = phase == Job::Phase::gate_up ? gate_chunk : down_chunk;
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

  Status execute(std::span<const ExpertWorkGroup> groups,
                 std::span<const float> inputs, std::uint32_t rows,
                 std::uint32_t top_k, std::span<float> outputs) {
    if (groups.empty()) return Status::success();
    if (rows == 0 || top_k == 0 || inputs.empty() || outputs.empty()) {
      return {ErrorCode::invalid_argument, "invalid CPU expert batch"};
    }
    Batch batch;
    batch.inputs = inputs.data();
    batch.outputs = outputs.data();
    batch.rows = rows;
    batch.top_k = top_k;
    batch.input_values = inputs.size();
    batch.output_values = outputs.size();
    batch.groups.assign(groups.begin(), groups.end());
    batch.intermediate_offsets.reserve(groups.size());
    std::size_t intermediate_values = 0;
    for (const auto& group : batch.groups) {
      const auto status = validate_group(
          group, rows, top_k, inputs.size(), outputs.size());
      if (!status.ok()) return status;
      batch.intermediate_offsets.push_back(intermediate_values);
      intermediate_values +=
          group.selections.size() * group.sections.intermediate;
    }
    batch.intermediate.resize(intermediate_values);
    auto status = dispatch_phase(batch, Job::Phase::gate_up);
    if (!status.ok()) return status;
    return dispatch_phase(batch, Job::Phase::down);
  }

  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  std::deque<Job> jobs;
  std::vector<std::thread> workers;
  bool stopping{};
};

ExpertExecutor::ExpertExecutor(std::uint32_t thread_count)
    : impl_(std::make_unique<Impl>(thread_count)) {}

ExpertExecutor::~ExpertExecutor() = default;

Status ExpertExecutor::execute(std::span<const ExpertWorkGroup> groups,
                               std::span<const float> inputs,
                               std::uint32_t rows, std::uint32_t top_k,
                               std::span<float> selection_outputs) {
  return impl_->execute(groups, inputs, rows, top_k, selection_outputs);
}

}  // namespace expert::runtime::cpu
