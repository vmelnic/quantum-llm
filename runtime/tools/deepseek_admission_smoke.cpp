#include "expert/runtime/cuda/deepseek_admission.hpp"
#include "expert/runtime/deepseek_expert.hpp"
#include "expert/runtime/sha256.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
    std::cout << "{\"ok\":true,\"abi\":\"" << er::kDeepSeekSm86HotAbi
              << "\",\"bytes\":" << layout.slot_bytes
              << ",\"sha256\":\"" << actual_hash
              << "\",\"h2d_and_admission_ms\":" << elapsed_ms << "}\n";
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check(cudaFree(hot), "cudaFree hot");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "expert-deepseek-admission-smoke: " << error.what() << '\n';
    return 1;
  }
}
