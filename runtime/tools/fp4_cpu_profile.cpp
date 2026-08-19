#include "expert/runtime/cpu/deepseek_packed_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace er = expert::runtime;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> read_record(const std::filesystem::path& path,
                                   std::uint64_t offset,
                                   std::uint64_t bytes) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open FP4 record source");
  input.seekg(static_cast<std::streamoff>(offset));
  require(static_cast<bool>(input), "cannot seek to FP4 record");
  std::vector<std::byte> result(static_cast<std::size_t>(bytes));
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size()));
  require(static_cast<bool>(input), "truncated FP4 record");
  return result;
}

er::DeepSeekCompactSections sections(std::uint32_t hidden,
                                     std::uint32_t intermediate) {
  const auto matrix = [](std::uint32_t output, std::uint32_t input) {
    return static_cast<std::uint64_t>(output) * input / 2U;
  };
  const auto scales = [](std::uint32_t output, std::uint32_t input) {
    return static_cast<std::uint64_t>(output) * input / 32U;
  };
  er::DeepSeekCompactSections result{};
  result.w1_weight_bytes = matrix(intermediate, hidden);
  result.w1_scale_offset = result.w1_weight_bytes;
  result.w1_scale_bytes = scales(intermediate, hidden);
  result.w3_weight_offset = result.w1_scale_offset + result.w1_scale_bytes;
  result.w3_weight_bytes = matrix(intermediate, hidden);
  result.w3_scale_offset = result.w3_weight_offset + result.w3_weight_bytes;
  result.w3_scale_bytes = scales(intermediate, hidden);
  result.w2_weight_offset = result.w3_scale_offset + result.w3_scale_bytes;
  result.w2_weight_bytes = matrix(hidden, intermediate);
  result.w2_scale_offset = result.w2_weight_offset + result.w2_weight_bytes;
  result.w2_scale_bytes = scales(hidden, intermediate);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 5 || argc > 8) {
      std::cerr << "usage: expert-fp4-cpu-profile <record-file> <offset> "
                   "<hidden> <intermediate> [iterations] [threads] "
                   "[expert-groups]\n";
      return 64;
    }
    const auto hidden = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const auto intermediate =
        static_cast<std::uint32_t>(std::stoul(argv[4]));
    const auto iterations = argc >= 6
                                ? static_cast<std::uint32_t>(std::stoul(argv[5]))
                                : 8U;
    const auto threads = argc >= 7
                             ? static_cast<std::uint32_t>(std::stoul(argv[6]))
                             : std::max(1U, std::thread::hardware_concurrency());
    const auto group_count =
        argc == 8 ? static_cast<std::uint32_t>(std::stoul(argv[7])) : 1U;
    require(hidden != 0U && intermediate != 0U && hidden % 32U == 0U &&
                intermediate % 32U == 0U && iterations != 0U &&
                threads != 0U && threads <= 64U && group_count != 0U &&
                group_count <= 32U,
            "invalid FP4 CPU profile geometry");
    const auto layout = sections(hidden, intermediate);
    const auto record_bytes = layout.w2_scale_offset + layout.w2_scale_bytes;
    const auto first_offset = std::stoull(argv[2]);
    std::vector<std::vector<std::byte>> records;
    records.reserve(group_count);
    for (std::uint32_t group = 0U; group < group_count; ++group) {
      records.push_back(read_record(
          argv[1], first_offset + static_cast<std::uint64_t>(group) *
                                      record_bytes,
          record_bytes));
    }
    std::vector<float> inputs(hidden);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      const auto phase = static_cast<float>(index % 257U) * 0.03125F;
      inputs[index] = std::sin(phase) * 0.75F + std::cos(phase * 0.375F) * 0.25F;
    }
    std::vector<er::cpu::DeepSeekPackedWorkGroup> groups;
    groups.reserve(group_count);
    for (std::uint32_t group = 0U; group < group_count; ++group) {
      groups.push_back(
          {records[group], layout, hidden, intermediate, {group}, {group}});
    }
    std::vector<float> outputs(static_cast<std::size_t>(group_count) * hidden);
    er::cpu::DeepSeekPackedExecutor executor(
        {threads, 8U, 8U, 10.0F, true, true});
    auto status = executor.execute(groups, inputs, 1U, group_count, outputs);
    require(status.ok(), std::string(status.message()));
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
      status = executor.execute(groups, inputs, 1U, group_count, outputs);
      require(status.ok(), std::string(status.message()));
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    double checksum = 0.0;
    float maximum = 0.0F;
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      require(std::isfinite(outputs[index]), "FP4 CPU output is non-finite");
      checksum += static_cast<double>(outputs[index]) *
                  static_cast<double>((index % 101U) + 1U);
      maximum = std::max(maximum, std::abs(outputs[index]));
    }
    const auto selections_total =
        static_cast<double>(iterations) * group_count;
    const auto seconds_per_selection = elapsed / selections_total;
    const auto seconds_per_batch = elapsed / iterations;
    const auto metrics = executor.telemetry();
    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"encoding\":"
                 "\"fp4-e2m1-ue8m0-block32\",\"hidden\":"
              << hidden << ",\"intermediate\":" << intermediate
              << ",\"record_bytes\":" << record_bytes
              << ",\"threads\":" << threads
              << ",\"expert_groups\":" << group_count
              << ",\"iterations\":" << iterations
              << ",\"milliseconds_per_batch\":"
              << seconds_per_batch * 1000.0
              << ",\"milliseconds_per_selection\":"
              << seconds_per_selection * 1000.0
              << ",\"source_gib_per_second\":"
              << static_cast<double>(record_bytes) * group_count /
                     seconds_per_batch /
                     static_cast<double>(1ULL << 30U)
              << ",\"workers_used\":" << metrics.workers_used_last
              << ",\"output_max_abs\":" << maximum
              << ",\"output_checksum\":" << checksum << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
