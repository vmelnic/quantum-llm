#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <latch>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kLayers = 43U;
constexpr std::uint32_t kTopK = 6U;
constexpr std::uint32_t kLocalExperts = kTopK / 2U;
constexpr std::uint32_t kHidden = 4096U;
constexpr std::uint32_t kIntermediate = 2048U;
constexpr std::uint32_t kThreads = 256U;
constexpr std::uint32_t kWarpsPerBlock = kThreads / 32U;
constexpr std::uint64_t kMatrixValues =
    static_cast<std::uint64_t>(kHidden) * kIntermediate;
constexpr std::uint64_t kWeightBytes = kMatrixValues / 2U;
constexpr std::uint64_t kScaleBytes = kMatrixValues / 32U;
constexpr std::uint64_t kRecordBytes = 3U * (kWeightBytes + kScaleBytes);
constexpr std::uint64_t kW1Weight = 0U;
constexpr std::uint64_t kW1Scale = kW1Weight + kWeightBytes;
constexpr std::uint64_t kW3Weight = kW1Scale + kScaleBytes;
constexpr std::uint64_t kW3Scale = kW3Weight + kWeightBytes;
constexpr std::uint64_t kW2Weight = kW3Scale + kScaleBytes;
constexpr std::uint64_t kW2Scale = kW2Weight + kWeightBytes;
constexpr double kTotalTokenGateMs = 100.0;
constexpr std::array<std::uint32_t, kLocalExperts> kFirstSlots{0U, 2U, 4U};
constexpr std::array<std::uint32_t, kLocalExperts> kSecondSlots{1U, 3U, 5U};
constexpr std::array<float, kTopK> kRouting{
    0.29F, 0.23F, 0.18F, 0.13F, 0.10F, 0.07F};

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class DeviceAllocation final {
 public:
  DeviceAllocation(int device, std::size_t bytes) : device_(device) {
    check(cudaSetDevice(device_), "select allocation device");
    check(cudaMalloc(&pointer_, bytes), "allocate device memory");
  }
  ~DeviceAllocation() {
    if (pointer_) {
      (void)cudaSetDevice(device_);
      (void)cudaFree(pointer_);
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(pointer_);
  }

 private:
  int device_{};
  void* pointer_{};
};

template <typename T>
class PinnedAllocation final {
 public:
  explicit PinnedAllocation(std::size_t count) : count_(count) {
    check(cudaHostAlloc(reinterpret_cast<void**>(&pointer_),
                        count * sizeof(T), cudaHostAllocPortable),
          "allocate portable pinned memory");
  }
  ~PinnedAllocation() {
    if (pointer_) (void)cudaFreeHost(pointer_);
  }
  PinnedAllocation(const PinnedAllocation&) = delete;
  PinnedAllocation& operator=(const PinnedAllocation&) = delete;
  [[nodiscard]] T* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  T* pointer_{};
  std::size_t count_{};
};

__device__ std::int8_t decode_fp4_twice(std::uint8_t code) {
  const auto index = code & 0x07U;
  const auto magnitude = index <= 4U ? static_cast<int>(index)
                         : index == 5U ? 6
                         : index == 6U ? 8
                                       : 12;
  return static_cast<std::int8_t>((code & 0x08U) ? -magnitude : magnitude);
}

__device__ float decode_ue8m0(std::uint8_t code) {
  return code == 0U ? __uint_as_float(0x00400000U)
                    : __uint_as_float(static_cast<unsigned>(code) << 23U);
}

__device__ float warp_sum(float value) {
  for (int offset = 16; offset != 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__device__ float round_bf16(float value) {
  auto bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) return value;
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return __uint_as_float(bits & 0xffff0000U);
}

__global__ void quantize_q8_kernel(const float* input, std::int8_t* output,
                                   float* scales, std::uint32_t rows,
                                   std::uint32_t columns) {
  const auto row = static_cast<std::uint32_t>(blockIdx.x);
  if (row >= rows) return;
  const auto* source = input + static_cast<std::size_t>(row) * columns;
  auto* target = output + static_cast<std::size_t>(row) * columns;
  __shared__ float maxima[kThreads];
  float maximum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(source[column]));
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = kThreads / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const auto scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) scales[row] = scale;
  __syncthreads();
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    auto quantized = __float2int_rn(source[column] / scale);
    quantized = max(-127, min(127, quantized));
    target[column] = static_cast<std::int8_t>(quantized);
  }
}

__device__ float fp4_q8_dot(const std::uint8_t* weights,
                            const std::uint8_t* scales,
                            const std::int8_t* activation,
                            float activation_scale, std::uint32_t row,
                            std::uint32_t columns) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto blocks = columns / 32U;
  const auto* row_weights = reinterpret_cast<const uint4*>(
      weights + static_cast<std::size_t>(row) * (columns / 2U));
  const auto* row_scales = scales + static_cast<std::size_t>(row) * blocks;
  float total = 0.0F;
  for (std::uint32_t block = lane; block < blocks; block += 32U) {
    const auto packed = row_weights[block];
    const std::uint32_t words[]{packed.x, packed.y, packed.z, packed.w};
    const auto* q = activation + static_cast<std::size_t>(block) * 32U;
    int block_total = 0;
#pragma unroll
    for (std::uint32_t byte = 0U; byte < 16U; ++byte) {
      const auto packed_weight = static_cast<std::uint8_t>(
          words[byte / 4U] >> (8U * (byte & 3U)));
      block_total +=
          static_cast<int>(decode_fp4_twice(packed_weight & 0x0fU)) *
              static_cast<int>(q[2U * byte]) +
          static_cast<int>(decode_fp4_twice(packed_weight >> 4U)) *
              static_cast<int>(q[2U * byte + 1U]);
    }
    total += static_cast<float>(block_total) * decode_ue8m0(row_scales[block]);
  }
  return warp_sum(total) * activation_scale * 0.5F;
}

__global__ void gate_up_kernel(const std::uint8_t* layer_records,
                               const std::int8_t* input,
                               const float* input_scale,
                               float* intermediate) {
  const auto local_expert = static_cast<std::uint32_t>(blockIdx.y);
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / 32U;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto row = static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock +
                   warp;
  if (local_expert >= kLocalExperts || row >= kIntermediate) return;
  const auto* record = layer_records +
      static_cast<std::uint64_t>(local_expert) * kRecordBytes;
  const auto gate_value = fp4_q8_dot(
      record + kW1Weight, record + kW1Scale, input, input_scale[0], row,
      kHidden);
  const auto up_value = fp4_q8_dot(
      record + kW3Weight, record + kW3Scale, input, input_scale[0], row,
      kHidden);
  if (lane == 0U) {
    const auto offset = static_cast<std::size_t>(local_expert) * kIntermediate +
                        row;
    const auto bounded_gate = fminf(gate_value, 10.0F);
    const auto bounded_up = fminf(fmaxf(up_value, -10.0F), 10.0F);
    intermediate[offset] = round_bf16(
        (bounded_gate / (1.0F + expf(-bounded_gate))) * bounded_up);
  }
}

__global__ void down_kernel(const std::uint8_t* layer_records,
                            const std::int8_t* intermediate,
                            const float* intermediate_scales, float* output) {
  const auto local_expert = static_cast<std::uint32_t>(blockIdx.y);
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / 32U;
  const auto lane = static_cast<std::uint32_t>(threadIdx.x) & 31U;
  const auto row = static_cast<std::uint32_t>(blockIdx.x) * kWarpsPerBlock +
                   warp;
  if (local_expert >= kLocalExperts || row >= kHidden) return;
  const auto* record = layer_records +
      static_cast<std::uint64_t>(local_expert) * kRecordBytes;
  const auto* q = intermediate +
      static_cast<std::size_t>(local_expert) * kIntermediate;
  const auto value = fp4_q8_dot(
      record + kW2Weight, record + kW2Scale, q,
      intermediate_scales[local_expert], row, kIntermediate);
  if (lane == 0U)
    output[static_cast<std::size_t>(local_expert) * kHidden + row] = value;
}

void launch_layer(const std::uint8_t* records, const float* input,
                  std::int8_t* q_input, float* q_input_scale,
                  float* intermediate,
                  std::int8_t* q_intermediate, float* q_intermediate_scales,
                  float* output, std::uint32_t layer, cudaStream_t stream) {
  const auto* layer_records = records +
      static_cast<std::uint64_t>(layer) * kLocalExperts * kRecordBytes;
  quantize_q8_kernel<<<1U, kThreads, 0, stream>>>(
      input, q_input, q_input_scale, 1U, kHidden);
  check(cudaPeekAtLastError(), "launch P100 input quantization");
  const dim3 gate_grid(
      (kIntermediate + kWarpsPerBlock - 1U) / kWarpsPerBlock,
      kLocalExperts);
  gate_up_kernel<<<gate_grid, kThreads, 0, stream>>>(
      layer_records, q_input, q_input_scale, intermediate);
  check(cudaPeekAtLastError(), "launch P100 gate/up");
  quantize_q8_kernel<<<kLocalExperts, kThreads, 0, stream>>>(
      intermediate, q_intermediate, q_intermediate_scales, kLocalExperts,
      kIntermediate);
  check(cudaPeekAtLastError(), "launch P100 intermediate quantization");
  const dim3 down_grid((kHidden + kWarpsPerBlock - 1U) / kWarpsPerBlock,
                       kLocalExperts);
  down_kernel<<<down_grid, kThreads, 0, stream>>>(
      layer_records, q_intermediate, q_intermediate_scales, output);
  check(cudaPeekAtLastError(), "launch P100 down");
}

std::filesystem::path layer_path(const std::filesystem::path& root,
                                 std::uint32_t layer) {
  std::ostringstream name;
  name << "experts-" << std::setw(2) << std::setfill('0') << layer << ".dsc";
  return root / name.str();
}

std::vector<std::uint8_t> read_expert(const std::filesystem::path& root,
                                      std::uint32_t layer,
                                      std::uint32_t expert) {
  const auto path = layer_path(root, layer);
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open " + path.string());
  input.seekg(static_cast<std::streamoff>(
      static_cast<std::uint64_t>(expert) * kRecordBytes));
  require(static_cast<bool>(input), "cannot seek in " + path.string());
  std::vector<std::uint8_t> result(static_cast<std::size_t>(kRecordBytes));
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size()));
  require(input.gcount() == static_cast<std::streamsize>(result.size()),
          "truncated expert in " + path.string());
  return result;
}

float cpu_scale(std::uint8_t code) {
  return std::ldexp(1.0F, code == 0U ? -127 : static_cast<int>(code) - 127);
}

int cpu_weight_twice(std::uint8_t code) {
  constexpr std::array<int, 16U> table{
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return table[code & 0x0fU];
}

float cpu_round_bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000U) == 0x7f800000U) return value;
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return std::bit_cast<float>(bits & 0xffff0000U);
}

float cpu_quantize(const float* input, std::int8_t* output,
                   std::uint32_t count) {
  float maximum = 0.0F;
  for (std::uint32_t index = 0U; index < count; ++index)
    maximum = std::max(maximum, std::abs(input[index]));
  const auto scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
  for (std::uint32_t index = 0U; index < count; ++index) {
    const auto rounded = static_cast<int>(std::nearbyint(input[index] / scale));
    output[index] = static_cast<std::int8_t>(
        std::clamp(rounded, -127, 127));
  }
  return scale;
}

float cpu_dot(const std::uint8_t* weights, const std::uint8_t* scales,
              const std::int8_t* input, float input_scale,
              std::uint32_t row, std::uint32_t columns) {
  const auto blocks = columns / 32U;
  const auto* packed = weights + static_cast<std::size_t>(row) * columns / 2U;
  const auto* row_scales = scales + static_cast<std::size_t>(row) * blocks;
  float total = 0.0F;
  for (std::uint32_t block = 0U; block < blocks; ++block) {
    int block_total = 0;
    for (std::uint32_t column = 0U; column < 32U; ++column) {
      const auto byte = packed[block * 16U + column / 2U];
      const auto code = static_cast<std::uint8_t>(
          (column & 1U) ? byte >> 4U : byte & 0x0fU);
      block_total += cpu_weight_twice(code) *
                     static_cast<int>(input[block * 32U + column]);
    }
    total += static_cast<float>(block_total) * cpu_scale(row_scales[block]);
  }
  return total * input_scale * 0.5F;
}

std::vector<float> cpu_oracle(const std::vector<std::uint8_t>& record,
                              const std::vector<float>& input) {
  std::vector<std::int8_t> q_input(kHidden);
  const auto input_scale = cpu_quantize(input.data(), q_input.data(), kHidden);
  std::vector<float> intermediate(kIntermediate);
  for (std::uint32_t row = 0U; row < kIntermediate; ++row) {
    auto gate = cpu_dot(record.data() + kW1Weight,
                        record.data() + kW1Scale, q_input.data(), input_scale,
                        row, kHidden);
    auto up = cpu_dot(record.data() + kW3Weight,
                      record.data() + kW3Scale, q_input.data(), input_scale,
                      row, kHidden);
    gate = std::min(gate, 10.0F);
    up = std::clamp(up, -10.0F, 10.0F);
    intermediate[row] = cpu_round_bf16(
        (gate / (1.0F + std::exp(-gate))) * up);
  }
  std::vector<std::int8_t> q_intermediate(kIntermediate);
  const auto intermediate_scale = cpu_quantize(
      intermediate.data(), q_intermediate.data(), kIntermediate);
  std::vector<float> output(kHidden);
  for (std::uint32_t row = 0U; row < kHidden; ++row)
    output[row] = cpu_dot(record.data() + kW2Weight,
                          record.data() + kW2Scale, q_intermediate.data(),
                          intermediate_scale, row, kIntermediate);
  return output;
}

struct Shared final {
  explicit Shared(float* input, std::array<float*, 2U> outputs)
      : host_input(input), host_outputs(outputs) {}
  std::latch ready{2};
  std::barrier<> command{3};
  std::barrier<> complete{3};
  std::atomic<bool> stop{false};
  std::atomic<std::uint32_t> layer{0U};
  float* host_input{};
  std::array<float*, 2U> host_outputs{};
  std::array<float, 2U> active_ms{};
  std::array<std::string, 2U> errors;
  std::array<std::string, 2U> names;
};

class WorkerJoiner final {
 public:
  WorkerJoiner(Shared& shared, std::thread& first, std::thread& second)
      : shared_(shared), first_(first), second_(second) {}
  ~WorkerJoiner() { stop(); }
  WorkerJoiner(const WorkerJoiner&) = delete;
  WorkerJoiner& operator=(const WorkerJoiner&) = delete;
  void stop() noexcept {
    if (!first_.joinable() && !second_.joinable()) return;
    shared_.stop.store(true, std::memory_order_release);
    shared_.command.arrive_and_wait();
    if (first_.joinable()) first_.join();
    if (second_.joinable()) second_.join();
  }

 private:
  Shared& shared_;
  std::thread& first_;
  std::thread& second_;
};

void p100_worker(std::uint32_t worker_index, int device,
                 const std::filesystem::path& root, Shared& shared) {
  bool command_loop = false;
  try {
    check(cudaSetDevice(device), "select P100");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "query P100");
    require(properties.major == 6 && properties.minor == 0,
            "route gate device is not SM60");
    shared.names[worker_index] = properties.name;
    DeviceAllocation records(
        device, static_cast<std::size_t>(kLayers) * kLocalExperts *
                    static_cast<std::size_t>(kRecordBytes));
    DeviceAllocation input(device, kHidden * sizeof(float));
    DeviceAllocation q_input(device, kHidden * sizeof(std::int8_t));
    DeviceAllocation q_input_scale(device, sizeof(float));
    DeviceAllocation intermediate(
        device, kLocalExperts * kIntermediate * sizeof(float));
    DeviceAllocation q_intermediate(
        device, kLocalExperts * kIntermediate * sizeof(std::int8_t));
    DeviceAllocation q_intermediate_scales(
        device, kLocalExperts * sizeof(float));
    DeviceAllocation output(device, kLocalExperts * kHidden * sizeof(float));
    const auto& slots = worker_index == 0U ? kFirstSlots : kSecondSlots;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
      for (std::uint32_t local = 0U; local < kLocalExperts; ++local) {
        const auto record = read_expert(root, layer, slots[local]);
        auto* destination = records.as<std::uint8_t>() +
            (static_cast<std::uint64_t>(layer) * kLocalExperts + local) *
                kRecordBytes;
        check(cudaMemcpy(destination, record.data(), record.size(),
                         cudaMemcpyHostToDevice),
              "load resident P100 expert");
      }
    }
    cudaStream_t stream{};
    cudaEvent_t begin{}, end{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "create P100 stream");
    check(cudaEventCreate(&begin), "create P100 begin event");
    check(cudaEventCreate(&end), "create P100 end event");
    shared.ready.count_down();
    command_loop = true;
    for (;;) {
      shared.command.arrive_and_wait();
      if (shared.stop.load(std::memory_order_acquire)) break;
      try {
        const auto layer = shared.layer.load(std::memory_order_acquire);
        check(cudaEventRecord(begin, stream), "record P100 layer begin");
        check(cudaMemcpyAsync(input.as<float>(), shared.host_input,
                              kHidden * sizeof(float), cudaMemcpyHostToDevice,
                              stream),
              "bounce input to P100");
        launch_layer(
            records.as<std::uint8_t>(), input.as<float>(),
            q_input.as<std::int8_t>(), q_input_scale.as<float>(),
            intermediate.as<float>(),
            q_intermediate.as<std::int8_t>(),
            q_intermediate_scales.as<float>(), output.as<float>(), layer,
            stream);
        check(cudaMemcpyAsync(shared.host_outputs[worker_index],
                              output.as<float>(),
                              kLocalExperts * kHidden * sizeof(float),
                              cudaMemcpyDeviceToHost, stream),
              "bounce P100 outputs to host");
        check(cudaEventRecord(end, stream), "record P100 layer end");
        check(cudaEventSynchronize(end), "wait P100 layer");
        check(cudaEventElapsedTime(&shared.active_ms[worker_index], begin, end),
              "measure P100 layer");
      } catch (const std::exception& error) {
        shared.errors[worker_index] = error.what();
      }
      shared.complete.arrive_and_wait();
    }
    (void)cudaEventDestroy(begin);
    (void)cudaEventDestroy(end);
    (void)cudaStreamDestroy(stream);
  } catch (const std::exception& error) {
    shared.errors[worker_index] = error.what();
    if (!command_loop) {
      shared.ready.count_down();
      for (;;) {
        shared.command.arrive_and_wait();
        if (shared.stop.load(std::memory_order_acquire)) break;
        shared.complete.arrive_and_wait();
      }
    }
  }
}

struct RouteResult final {
  double wall_ms{};
  double p100_critical_ms{};
};

RouteResult run_route(Shared& shared, int primary_device,
                      cudaStream_t primary_stream, float* primary_input,
                      float* stable_outputs, float* aggregate) {
  const auto started = std::chrono::steady_clock::now();
  double p100_critical_ms = 0.0;
  for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
    check(cudaSetDevice(primary_device), "select primary for input bounce");
    check(cudaMemcpyAsync(shared.host_input, primary_input,
                          kHidden * sizeof(float), cudaMemcpyDeviceToHost,
                          primary_stream),
          "bounce primary input to host");
    check(cudaStreamSynchronize(primary_stream), "wait primary input bounce");
    shared.layer.store(layer, std::memory_order_release);
    shared.command.arrive_and_wait();
    shared.complete.arrive_and_wait();
    require(shared.errors[0].empty() && shared.errors[1].empty(),
            "P100 route worker failed");
    p100_critical_ms += std::max(shared.active_ms[0], shared.active_ms[1]);
    for (std::uint32_t local = 0U; local < kLocalExperts; ++local) {
      std::copy_n(shared.host_outputs[0] +
                      static_cast<std::size_t>(local) * kHidden,
                  kHidden,
                  stable_outputs +
                      static_cast<std::size_t>(kFirstSlots[local]) * kHidden);
      std::copy_n(shared.host_outputs[1] +
                      static_cast<std::size_t>(local) * kHidden,
                  kHidden,
                  stable_outputs +
                      static_cast<std::size_t>(kSecondSlots[local]) * kHidden);
    }
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
      float total = 0.0F;
      for (std::uint32_t slot = 0U; slot < kTopK; ++slot)
        total += kRouting[slot] *
                 stable_outputs[static_cast<std::size_t>(slot) * kHidden +
                                column];
      aggregate[column] = total;
    }
    check(cudaSetDevice(primary_device), "select primary for output bounce");
    check(cudaMemcpyAsync(primary_input, aggregate, kHidden * sizeof(float),
                          cudaMemcpyHostToDevice, primary_stream),
          "bounce stable output to primary");
    check(cudaStreamSynchronize(primary_stream), "wait primary output bounce");
  }
  return {
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count(),
      p100_critical_ms};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: expert-pascal-moe-route-gate <compact-pack-root>\n";
      return 64;
    }
    const std::filesystem::path root(argv[1]);
    require(std::filesystem::is_regular_file(root / "manifest.json"),
            "compact pack manifest is missing");
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
      const auto path = layer_path(root, layer);
      require(std::filesystem::is_regular_file(path),
              "compact pack layer is missing");
      require(std::filesystem::file_size(path) == 256U * kRecordBytes,
              "compact pack layer size is invalid");
    }

    int devices = 0;
    check(cudaGetDeviceCount(&devices), "query CUDA devices");
    require(devices >= 3, "route gate requires RTX 3090 plus two P100s");
    int first_peer = 0, second_peer = 0;
    check(cudaDeviceCanAccessPeer(&first_peer, 0, 1), "query first peer");
    check(cudaDeviceCanAccessPeer(&second_peer, 0, 2), "query second peer");
    require(first_peer == 0 && second_peer == 0,
            "route gate expected the measured host-bounce topology");

    check(cudaSetDevice(0), "select RTX 3090");
    DeviceAllocation primary_input(0, kHidden * sizeof(float));
    cudaStream_t primary_stream{};
    check(cudaStreamCreateWithFlags(&primary_stream, cudaStreamNonBlocking),
          "create primary stream");
    PinnedAllocation<float> host_input(kHidden);
    PinnedAllocation<float> first_outputs(kLocalExperts * kHidden);
    PinnedAllocation<float> second_outputs(kLocalExperts * kHidden);
    PinnedAllocation<float> stable_outputs(kTopK * kHidden);
    PinnedAllocation<float> aggregate(kHidden);
    Shared shared(host_input.get(), {first_outputs.get(), second_outputs.get()});
    std::thread first([&] { p100_worker(0U, 1, root, shared); });
    std::thread second([&] { p100_worker(1U, 2, root, shared); });
    WorkerJoiner workers(shared, first, second);
    shared.ready.wait();
    require(shared.errors[0].empty() && shared.errors[1].empty(),
            "P100 resident expert load failed");

    std::vector<float> numeric_input(kHidden);
    for (std::uint32_t index = 0U; index < kHidden; ++index) {
      const auto phase = static_cast<float>(index % 257U) * 0.03125F;
      numeric_input[index] =
          std::sin(phase) * 0.75F + std::cos(phase * 0.375F) * 0.25F;
    }
    check(cudaSetDevice(0), "select primary for numeric input");
    check(cudaMemcpyAsync(primary_input.as<float>(), numeric_input.data(),
                          kHidden * sizeof(float), cudaMemcpyHostToDevice,
                          primary_stream),
          "upload numeric input");
    check(cudaStreamSynchronize(primary_stream), "wait numeric input");
    shared.layer.store(0U, std::memory_order_release);
    check(cudaMemcpyAsync(host_input.get(), primary_input.as<float>(),
                          kHidden * sizeof(float), cudaMemcpyDeviceToHost,
                          primary_stream),
          "numeric primary bounce");
    check(cudaStreamSynchronize(primary_stream), "wait numeric primary bounce");
    shared.command.arrive_and_wait();
    shared.complete.arrive_and_wait();
    require(shared.errors[0].empty() && shared.errors[1].empty(),
            "P100 numerical route failed");
    const auto first_record = read_expert(root, 0U, kFirstSlots[0]);
    const auto expected = cpu_oracle(first_record, numeric_input);
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum_error = 0.0;
    for (std::uint32_t index = 0U; index < kHidden; ++index) {
      const auto difference = static_cast<double>(first_outputs.get()[index]) -
                              static_cast<double>(expected[index]);
      squared_error += difference * difference;
      squared_reference += static_cast<double>(expected[index]) * expected[index];
      maximum_error = std::max(maximum_error, std::abs(difference));
    }
    const auto relative_l2 = std::sqrt(squared_error /
        std::max(squared_reference, std::numeric_limits<double>::min()));
    const bool numeric_pass = relative_l2 <= 2.0e-5 && maximum_error <= 0.05;

    std::fill_n(aggregate.get(), kHidden, 0.0F);
    check(cudaMemcpyAsync(primary_input.as<float>(), aggregate.get(),
                          kHidden * sizeof(float), cudaMemcpyHostToDevice,
                          primary_stream),
          "reset primary input");
    check(cudaStreamSynchronize(primary_stream), "wait primary reset");
    (void)run_route(shared, 0, primary_stream, primary_input.as<float>(),
                    stable_outputs.get(), aggregate.get());

    std::array<RouteResult, 3U> results{};
    for (auto& result : results) {
      std::fill_n(aggregate.get(), kHidden, 0.0F);
      check(cudaMemcpyAsync(primary_input.as<float>(), aggregate.get(),
                            kHidden * sizeof(float), cudaMemcpyHostToDevice,
                            primary_stream),
            "reset timed primary input");
      check(cudaStreamSynchronize(primary_stream), "wait timed primary reset");
      result = run_route(shared, 0, primary_stream, primary_input.as<float>(),
                         stable_outputs.get(), aggregate.get());
    }

    workers.stop();
    (void)cudaStreamDestroy(primary_stream);

    std::array<double, 3U> ordered{};
    std::transform(results.begin(), results.end(), ordered.begin(),
                   [](const auto& result) { return result.wall_ms; });
    std::sort(ordered.begin(), ordered.end());
    const auto median_ms = ordered[ordered.size() / 2U];
    const bool route_under_total_gate = median_ms < kTotalTokenGateMs;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema_version\":1,\"model\":\"deepseek-v4-flash\""
              << ",\"format\":\"fp4.e2m1.ue8m0.block32\""
              << ",\"layers\":" << kLayers
              << ",\"top_k\":" << kTopK
              << ",\"record_bytes\":" << kRecordBytes
              << ",\"selected_weight_bytes_per_token\":"
              << kLayers * kTopK * kRecordBytes
              << ",\"peer_access\":false"
              << ",\"devices\":[\"" << shared.names[0] << "\",\""
              << shared.names[1] << "\"]"
              << ",\"numeric_max_abs\":" << maximum_error
              << ",\"numeric_relative_l2\":" << relative_l2
              << ",\"numeric_pass\":" << (numeric_pass ? "true" : "false")
              << ",\"iterations\":[";
    for (std::size_t index = 0U; index < results.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << "{\"wall_ms\":" << results[index].wall_ms
                << ",\"p100_critical_ms\":"
                << results[index].p100_critical_ms << '}';
    }
    std::cout << "]"
              << ",\"median_route_wall_ms\":" << median_ms
              << ",\"total_token_gate_ms\":" << kTotalTokenGateMs
              << ",\"route_under_total_gate\":"
              << (route_under_total_gate ? "true" : "false")
              << ",\"admission_pass\":"
              << (numeric_pass && route_under_total_gate ? "true" : "false")
              << "}\n";
    return numeric_pass && route_under_total_gate ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "{\"schema_version\":1,\"admission_pass\":false,\"error\":"
              << std::quoted(std::string(error.what())) << "}\n";
    return 1;
  }
}
