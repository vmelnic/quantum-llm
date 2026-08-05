#include "expert/runtime/cuda/deepseek_admission.hpp"
#include "expert/runtime/cuda/moe_kernels.hpp"
#include "expert/runtime/deepseek_expert.hpp"
#include "expert/runtime/sha256.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace er = expert::runtime;

namespace {

void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path,
                                    std::size_t expected) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<std::size_t>(stream.tellg()) != expected) {
    throw std::runtime_error("compact fixture size mismatch: " + path.string());
  }
  std::vector<std::uint8_t> bytes(expected);
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short compact fixture read");
  return bytes;
}

struct DeviceInput {
  std::uint8_t* weight{};
  std::uint8_t* scale{};
  ~DeviceInput() {
    if (weight) static_cast<void>(cudaFree(weight));
    if (scale) static_cast<void>(cudaFree(scale));
  }
};

std::string hex_digest(const er::Sha256Digest& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto value : digest) {
    output << std::setw(2) << std::to_integer<unsigned>(value);
  }
  return output.str();
}

std::vector<float> cpu_expert(const std::vector<std::byte>& slot,
                              const er::DeepSeekSm86HotLayout& layout,
                              const er::DeepSeekExpertGeometry& geometry,
                              const std::vector<float>& input) {
  const auto* gate = reinterpret_cast<const std::int8_t*>(slot.data() + layout.gate_up_q.offset);
  const auto* scales = reinterpret_cast<const float*>(slot.data() + layout.gate_up_scales.offset);
  const auto* down = reinterpret_cast<const std::int8_t*>(slot.data() + layout.down_q.offset);
  const auto* down_scales = reinterpret_cast<const float*>(slot.data() + layout.down_scales.offset);
  std::vector<float> intermediate(geometry.intermediate);
  for (std::uint32_t row = 0; row < geometry.intermediate; ++row) {
    double g = 0.0, u = 0.0;
    for (std::uint32_t column = 0; column < geometry.hidden; ++column) {
      g += gate[static_cast<std::size_t>(row) * geometry.hidden + column] * input[column];
      u += gate[(static_cast<std::size_t>(geometry.intermediate + row) * geometry.hidden) + column] * input[column];
    }
    const auto gf = static_cast<float>(g) * scales[row];
    const auto uf = static_cast<float>(u) * scales[geometry.intermediate + row];
    intermediate[row] = (gf / (1.0F + std::exp(-gf))) * uf;
  }
  std::vector<float> output(geometry.hidden);
  for (std::uint32_t row = 0; row < geometry.hidden; ++row) {
    double sum = 0.0;
    for (std::uint32_t column = 0; column < geometry.intermediate; ++column)
      sum += down[static_cast<std::size_t>(row) * geometry.intermediate + column] * intermediate[column];
    output[row] = static_cast<float>(sum) * down_scales[row];
  }
  return output;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: expert-deepseek-admission-smoke <bundle> <expected-sha256>\n";
      return 64;
    }
    const std::filesystem::path root = argv[1];
    const std::string expected_hash = argv[2];
    const auto geometry = er::DeepSeekExpertGeometry::v4_flash();
    const auto layout = er::make_deepseek_sm86_hot_layout(geometry);
    const std::array projections{er::DeepSeekProjection::w1_gate,
                                 er::DeepSeekProjection::w3_up,
                                 er::DeepSeekProjection::w2_down};
    const std::array names{"w1", "w3", "w2"};
    std::array<DeviceInput, 3> device{};
    std::array<std::vector<std::uint8_t>, 3> weights;
    std::array<std::vector<std::uint8_t>, 3> scales;
    for (std::size_t index = 0; index < projections.size(); ++index) {
      weights[index] = read_file(root / (std::string(names[index]) + ".weight.bin"),
          geometry.compact_weight_bytes(projections[index]));
      scales[index] = read_file(root / (std::string(names[index]) + ".scale.bin"),
          geometry.compact_scale_bytes(projections[index]));
      check(cudaMalloc(reinterpret_cast<void**>(&device[index].weight),
                       weights[index].size()), "cudaMalloc compact weight");
      check(cudaMalloc(reinterpret_cast<void**>(&device[index].scale),
                       scales[index].size()), "cudaMalloc compact scale");
    }
    std::uint8_t* hot = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&hot), layout.slot_bytes),
          "cudaMalloc hot slot");
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    check(cudaEventRecord(start), "cudaEventRecord start");
    for (std::size_t index = 0; index < projections.size(); ++index) {
      check(cudaMemcpy(device[index].weight, weights[index].data(), weights[index].size(),
                       cudaMemcpyHostToDevice), "copy compact weight");
      check(cudaMemcpy(device[index].scale, scales[index].data(), scales[index].size(),
                       cudaMemcpyHostToDevice), "copy compact scale");
    }
    auto* gate_q = reinterpret_cast<std::int8_t*>(hot + layout.gate_up_q.offset);
    auto* gate_s = reinterpret_cast<float*>(hot + layout.gate_up_scales.offset);
    auto* down_q = reinterpret_cast<std::int8_t*>(hot + layout.down_q.offset);
    auto* down_s = reinterpret_cast<float*>(hot + layout.down_scales.offset);
    const auto gate_shape = geometry.matrix(er::DeepSeekProjection::w1_gate);
    const auto down_shape = geometry.matrix(er::DeepSeekProjection::w2_down);
    const std::array<er::cuda::DeepSeekAdmissionLaunch, 3> launches{{
        {device[0].weight, device[0].scale, gate_q, gate_s,
         gate_shape.rows, gate_shape.columns, nullptr},
        {device[1].weight, device[1].scale,
         gate_q + static_cast<std::size_t>(gate_shape.rows) * gate_shape.columns,
         gate_s + gate_shape.rows, gate_shape.rows, gate_shape.columns, nullptr},
        {device[2].weight, device[2].scale, down_q, down_s,
         down_shape.rows, down_shape.columns, nullptr},
    }};
    for (const auto& launch : launches) {
      const auto status = er::cuda::admit_deepseek_projection(launch);
      if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    }
    check(cudaEventRecord(stop), "cudaEventRecord stop");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed_ms = 0.0F;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    std::vector<std::byte> result(layout.slot_bytes);
    check(cudaMemcpy(result.data(), hot, result.size(), cudaMemcpyDeviceToHost),
          "copy hot slot");
    const auto actual_hash = hex_digest(er::sha256(result));
    if (actual_hash != expected_hash) {
      throw std::runtime_error("CUDA hot-slot hash mismatch: " + actual_hash);
    }
    std::vector<float> input(geometry.hidden);
    for (std::uint32_t i = 0; i < geometry.hidden; ++i)
      input[i] = std::sin(static_cast<float>(i) * 0.013F) * 0.25F;
    const auto reference = cpu_expert(result, layout, geometry, input);
    float *d_input{}, *d_intermediate{}, *d_output{}, *d_routing{};
    const std::int8_t** d_gate_table{};
    const std::int8_t** d_down_table{};
    const float** d_gate_scale_table{};
    const float** d_down_scale_table{};
    check(cudaMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(float)), "cudaMalloc input");
    check(cudaMalloc(reinterpret_cast<void**>(&d_intermediate), geometry.intermediate * sizeof(float)), "cudaMalloc intermediate");
    check(cudaMalloc(reinterpret_cast<void**>(&d_output), geometry.hidden * sizeof(float)), "cudaMalloc output");
    check(cudaMalloc(reinterpret_cast<void**>(&d_routing), sizeof(float)), "cudaMalloc routing");
    check(cudaMalloc(reinterpret_cast<void**>(&d_gate_table), sizeof(void*)), "cudaMalloc gate table");
    check(cudaMalloc(reinterpret_cast<void**>(&d_down_table), sizeof(void*)), "cudaMalloc down table");
    check(cudaMalloc(reinterpret_cast<void**>(&d_gate_scale_table), sizeof(void*)), "cudaMalloc gate scale table");
    check(cudaMalloc(reinterpret_cast<void**>(&d_down_scale_table), sizeof(void*)), "cudaMalloc down scale table");
    const std::int8_t* h_gate = gate_q;
    const std::int8_t* h_down = down_q;
    const float* h_gate_s = gate_s;
    const float* h_down_s = down_s;
    const float routing = 1.0F;
    check(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input");
    check(cudaMemcpy(d_routing, &routing, sizeof(float), cudaMemcpyHostToDevice), "copy routing");
    check(cudaMemcpy(d_gate_table, &h_gate, sizeof(void*), cudaMemcpyHostToDevice), "copy gate table");
    check(cudaMemcpy(d_down_table, &h_down, sizeof(void*), cudaMemcpyHostToDevice), "copy down table");
    check(cudaMemcpy(d_gate_scale_table, &h_gate_s, sizeof(void*), cudaMemcpyHostToDevice), "copy gate scales");
    check(cudaMemcpy(d_down_scale_table, &h_down_s, sizeof(void*), cudaMemcpyHostToDevice), "copy down scales");
    check(cudaEventRecord(start), "record GEMM start");
    const er::cuda::MoeLaunch moe{d_input, d_gate_table, d_gate_scale_table,
        d_down_table, d_down_scale_table, d_routing, nullptr, d_intermediate,
        d_output, geometry.hidden, geometry.intermediate, 1U, 1U, nullptr,
        nullptr, 0U};
    const auto moe_status = er::cuda::launch_moe_single_token(moe);
    if (!moe_status.ok()) throw std::runtime_error(std::string(moe_status.message()));
    check(cudaEventRecord(stop), "record GEMM stop");
    check(cudaEventSynchronize(stop), "synchronize GEMM");
    float gemm_ms = 0.0F;
    check(cudaEventElapsedTime(&gemm_ms, start, stop), "GEMM elapsed");
    std::vector<float> gpu_output(geometry.hidden);
    check(cudaMemcpy(gpu_output.data(), d_output, gpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy GEMM output");
    double squared = 0.0;
    float max_error = 0.0F;
    for (std::size_t i = 0; i < gpu_output.size(); ++i) {
      const auto error = std::abs(gpu_output[i] - reference[i]);
      max_error = std::max(max_error, error);
      squared += static_cast<double>(error) * error;
    }
    const auto rmse = std::sqrt(squared / gpu_output.size());
    std::cout << "{\"ok\":true,\"abi\":\"" << er::kDeepSeekSm86HotAbi
              << "\",\"bytes\":" << layout.slot_bytes
              << ",\"sha256\":\"" << actual_hash
              << "\",\"h2d_and_admission_ms\":" << elapsed_ms
              << ",\"expert_gemm_ms\":" << gemm_ms
              << ",\"output_rmse\":" << rmse
              << ",\"output_max_abs_error\":" << max_error << "}\n";
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check(cudaFree(hot), "cudaFree hot");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-admission-smoke: " << error.what() << '\n';
    return 1;
  }
}
