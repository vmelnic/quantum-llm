#include "expert/runtime/model_artifact.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kGiB = 1ULL << 30U;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::uint64_t kDeviceReserveBytes = kGiB;
constexpr std::uint64_t kWorkspaceAllowanceBytes = 256ULL * kMiB;
constexpr double kRequiredBytesPerSecond = 500.0e9;
constexpr std::uint32_t kThreads = 256U;
constexpr std::string_view kMlpCapability =
    "ffn.swiglu.dense.fp4-block32.v1";

void cuda_check(cudaError_t error, std::string_view operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

void cublas_check(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS " +
                             std::to_string(static_cast<int>(status)));
}

std::uint32_t attribute_u32(const expert::runtime::ModelDescriptor& model,
                            std::string_view name) {
  const auto found = model.attributes.find(name);
  if (found == model.attributes.end() || found->second == 0U ||
      found->second > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("artifact attribute is absent or invalid: " +
                             std::string(name));
  return static_cast<std::uint32_t>(found->second);
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          std::string_view label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::runtime_error(std::string(label) + " overflows");
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               std::string_view label) {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left)
    throw std::runtime_error(std::string(label) + " overflows");
  return left * right;
}

class DeviceAllocation final {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(int device, std::size_t bytes) : device_(device), bytes_(bytes) {
    cuda_check(cudaSetDevice(device_), "select allocation device");
    cuda_check(cudaMalloc(&pointer_, bytes_), "allocate device memory");
  }
  ~DeviceAllocation() {
    if (pointer_) {
      static_cast<void>(cudaSetDevice(device_));
      static_cast<void>(cudaFree(pointer_));
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  DeviceAllocation(DeviceAllocation&& other) noexcept
      : device_(other.device_), pointer_(other.pointer_), bytes_(other.bytes_) {
    other.pointer_ = nullptr;
    other.bytes_ = 0U;
  }
  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
    if (this == &other) return *this;
    if (pointer_) {
      static_cast<void>(cudaSetDevice(device_));
      static_cast<void>(cudaFree(pointer_));
    }
    device_ = other.device_;
    pointer_ = other.pointer_;
    bytes_ = other.bytes_;
    other.pointer_ = nullptr;
    other.bytes_ = 0U;
    return *this;
  }
  [[nodiscard]] void* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  int device_{};
  void* pointer_{};
  std::size_t bytes_{};
};

template <typename T>
class PinnedAllocation final {
 public:
  explicit PinnedAllocation(std::size_t count) : count_(count) {
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&pointer_),
                             count * sizeof(T), cudaHostAllocPortable),
               "allocate portable pinned memory");
  }
  ~PinnedAllocation() {
    if (pointer_) static_cast<void>(cudaFreeHost(pointer_));
  }
  PinnedAllocation(const PinnedAllocation&) = delete;
  PinnedAllocation& operator=(const PinnedAllocation&) = delete;
  [[nodiscard]] T* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }

 private:
  T* pointer_{};
  std::size_t count_{};
};

struct MlpPlan final {
  std::uint32_t layers{};
  std::uint32_t hidden{};
  std::uint32_t intermediate{};
  std::uint32_t full_attention_layers{};
  std::uint32_t recurrent_layers{};
  std::uint64_t hot_weight_bytes{};
  std::uint64_t mlp_compact_bytes{};
  std::uint64_t exact_kv_bytes{};
  std::uint64_t recurrent_state_bytes{};
  std::uint8_t minimum_scale_code{255U};
  std::uint8_t maximum_scale_code{};
};

void scan_scale_codes(const expert::runtime::ModelArtifact& artifact,
                      const expert::runtime::ArtifactDenseTensor& tensor,
                      MlpPlan& plan) {
  const auto* pack = artifact.find_pack(tensor.pack);
  if (!pack || tensor.scale_bytes == 0U)
    throw std::runtime_error("FP4 MLP tensor has no scale section");
  std::ifstream input(pack->path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open FP4 pack for scale scan");
  const auto offset = checked_add(tensor.record_offset, tensor.scale_offset,
                                  "FP4 scale file offset");
  input.seekg(static_cast<std::streamoff>(offset));
  if (!input)
    throw std::runtime_error("cannot seek to FP4 scale section");
  std::vector<std::uint8_t> buffer(static_cast<std::size_t>(
      std::min<std::uint64_t>(tensor.scale_bytes, 4ULL * kMiB)));
  std::uint64_t remaining = tensor.scale_bytes;
  while (remaining != 0U) {
    const auto bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes))
      throw std::runtime_error("short FP4 scale section read");
    const auto bounds = std::minmax_element(buffer.begin(),
                                            buffer.begin() + bytes);
    plan.minimum_scale_code =
        std::min(plan.minimum_scale_code, *bounds.first);
    plan.maximum_scale_code =
        std::max(plan.maximum_scale_code, *bounds.second);
    remaining -= bytes;
  }
}

const expert::runtime::ArtifactDenseTensor& require_tensor(
    const std::map<std::string, const expert::runtime::ArtifactDenseTensor*,
                   std::less<>>& tensors,
    const expert::runtime::OperationProgramDescriptor& operation,
    std::string_view role) {
  const auto binding = operation.tensor_bindings.find(role);
  if (binding == operation.tensor_bindings.end())
    throw std::runtime_error("MLP operation lacks tensor role " +
                             std::string(role));
  const auto tensor = tensors.find(binding->second);
  if (tensor == tensors.end())
    throw std::runtime_error("operation references an absent tensor");
  return *tensor->second;
}

MlpPlan make_plan(const expert::runtime::ModelArtifact& artifact) {
  const auto& model = artifact.model();
  std::map<std::string, const expert::runtime::ArtifactDenseTensor*,
           std::less<>> tensors;
  for (const auto& tensor : artifact.dense_tensors())
    if (!tensors.emplace(tensor.name, &tensor).second)
      throw std::runtime_error("artifact tensor name is duplicated");

  std::set<std::string, std::less<>> hot_names;
  MlpPlan plan;
  for (const auto& operation : model.operation_program) {
    const auto model_input = operation.capability.starts_with("embedding.") ||
                             operation.capability.starts_with("vision.");
    if (!model_input)
      for (const auto& [role, name] : operation.tensor_bindings) {
        (void)role;
        hot_names.insert(name);
      }
    if (operation.capability == "block.full-attention.output-gated.v1")
      ++plan.full_attention_layers;
    if (operation.capability.starts_with("block.recurrent-linear-attention."))
      ++plan.recurrent_layers;
    if (operation.capability != kMlpCapability) continue;

    const auto& gate = require_tensor(tensors, operation, "gate_projection");
    const auto& up = require_tensor(tensors, operation, "up_projection");
    const auto& down = require_tensor(tensors, operation, "down_projection");
    if (gate.encoding != "FP4_E2M1" || up.encoding != "FP4_E2M1" ||
        down.encoding != "FP4_E2M1" || gate.quant_abi != up.quant_abi ||
        gate.quant_abi != down.quant_abi || gate.shape.size() != 2U ||
        up.shape != gate.shape || down.shape.size() != 2U ||
        down.shape[0] != gate.shape[1] || down.shape[1] != gate.shape[0])
      throw std::runtime_error("artifact MLP is not a compatible FP4 SwiGLU");
    if (plan.layers == 0U) {
      plan.hidden = gate.shape[1];
      plan.intermediate = gate.shape[0];
    } else if (plan.hidden != gate.shape[1] ||
               plan.intermediate != gate.shape[0]) {
      throw std::runtime_error("artifact MLP geometry changes between layers");
    }
    scan_scale_codes(artifact, gate, plan);
    scan_scale_codes(artifact, up, plan);
    scan_scale_codes(artifact, down, plan);
    ++plan.layers;
    plan.mlp_compact_bytes = checked_add(
        plan.mlp_compact_bytes,
        checked_add(checked_add(gate.data_bytes, gate.scale_bytes, "gate"),
                    checked_add(up.data_bytes, up.scale_bytes, "up"),
                    "gate/up"),
        "MLP compact bytes");
    plan.mlp_compact_bytes = checked_add(
        plan.mlp_compact_bytes,
        checked_add(down.data_bytes, down.scale_bytes, "down"),
        "MLP compact bytes");
  }
  if (plan.layers == 0U || plan.hidden == 0U || plan.intermediate == 0U ||
      plan.full_attention_layers == 0U)
    throw std::runtime_error("artifact has no compatible dense MLP program");

  for (const auto& name : hot_names) {
    const auto found = tensors.find(name);
    if (found == tensors.end())
      throw std::runtime_error("hot operation tensor is absent");
    const auto& tensor = *found->second;
    const auto bytes = tensor.encoding == "FP4_E2M1"
                           ? checked_add(tensor.data_bytes, tensor.scale_bytes,
                                         "hot FP4 tensor")
                           : tensor.data_bytes;
    plan.hot_weight_bytes =
        checked_add(plan.hot_weight_bytes, bytes, "hot weight bytes");
  }

  const auto max_context = model.max_context_tokens;
  const auto kv_heads = attribute_u32(model, "kv_heads");
  const auto head_dim = attribute_u32(model, "head_dim");
  plan.exact_kv_bytes = checked_multiply(
      checked_multiply(
          checked_multiply(
              checked_multiply(plan.full_attention_layers, 2U, "K/V"),
              max_context, "KV context"),
          kv_heads, "KV heads"),
      checked_multiply(head_dim, sizeof(std::uint16_t), "KV head bytes"),
      "exact KV bytes");

  const auto key_heads = attribute_u32(model, "linear_key_heads");
  const auto key_dim = attribute_u32(model, "linear_key_head_dim");
  const auto value_heads = attribute_u32(model, "linear_value_heads");
  const auto value_dim = attribute_u32(model, "linear_value_head_dim");
  const auto conv_kernel = attribute_u32(model, "linear_conv_kernel");
  const auto conv_values =
      (2ULL * key_heads * key_dim + 1ULL * value_heads * value_dim) *
      conv_kernel;
  const auto recurrent_values =
      1ULL * value_heads * key_dim * value_dim;
  plan.recurrent_state_bytes = checked_multiply(
      plan.recurrent_layers,
      checked_multiply(conv_values + recurrent_values, sizeof(float),
                       "recurrent layer state"),
      "recurrent state");
  return plan;
}

__device__ std::int8_t fp4_twice(std::uint8_t code) {
  constexpr std::int8_t values[16]{0, 1, 2, 3, 4, 6, 8, 12,
                                   0, -1, -2, -3, -4, -6, -8, -12};
  return values[code & 0x0fU];
}

__global__ void fill_fp16_kernel(std::uint16_t* values, std::size_t count,
                                 std::uint16_t bits) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    values[index] = bits;
}

__global__ void decode_fp4_fp16_kernel(const std::uint8_t* packed,
                                       const std::uint8_t* scales,
                                       __half* decoded, std::uint32_t rows,
                                       std::uint32_t columns) {
  const auto count = static_cast<std::size_t>(rows) * columns;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto row = index / columns;
    const auto column = static_cast<std::uint32_t>(index % columns);
    const auto byte = packed[index / 2U];
    const auto code = static_cast<std::uint8_t>(
        (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
    const auto exponent = static_cast<int>(
        scales[row * (columns / 32U) + column / 32U]) - 127;
    if (code == 8U) {
      reinterpret_cast<std::uint16_t*>(decoded)[index] = 0x8000U;
    } else {
      const auto value = ldexpf(static_cast<float>(fp4_twice(code)) * 0.5F,
                                exponent);
      decoded[index] = __float2half_rn(value);
    }
  }
}

__global__ void quantize_q8_kernel(const float* input, std::int8_t* output,
                                   float* output_scale,
                                   std::uint32_t columns) {
  __shared__ float maxima[kThreads];
  float maximum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x)
    maximum = fmaxf(maximum, fabsf(input[column]));
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0U; stride >>= 1U) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }
  const auto scale = maxima[0] > 0.0F ? maxima[0] / 127.0F : 1.0F;
  if (threadIdx.x == 0U) output_scale[0] = scale;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    auto value = __float2int_rn(input[column] / scale);
    value = max(-127, min(127, value));
    output[column] = static_cast<std::int8_t>(value);
  }
}

__global__ void q8_to_fp16_kernel(const std::int8_t* input, __half* output,
                                  std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x)
    output[index] = __int2half_rn(static_cast<int>(input[index]));
}

__global__ void scale_output_kernel(float* output, const float* scale,
                                    std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x)
    output[index] *= scale[0];
}

__global__ void silu_product_kernel(float* gate, const float* up,
                                    std::uint32_t count) {
  for (std::uint32_t index =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const auto value = gate[index];
    gate[index] = value / (1.0F + expf(-value)) * up[index];
  }
}

void launch_gemv(cublasHandle_t blas, const __half* weights,
                 const __half* input, const float* scale, float* output,
                 std::uint32_t rows, std::uint32_t columns,
                 cudaStream_t stream) {
  const float one = 1.0F;
  const float zero = 0.0F;
  cublas_check(cublasGemmEx(
                   blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(rows), 1,
                   static_cast<int>(columns), &one, weights, CUDA_R_16F,
                   static_cast<int>(columns), input, CUDA_R_16F,
                   static_cast<int>(columns), &zero, output, CUDA_R_32F,
                   static_cast<int>(rows), CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "resident FP16 GEMV");
  scale_output_kernel<<<(rows + kThreads - 1U) / kThreads, kThreads, 0,
                          stream>>>(output, scale, rows);
  cuda_check(cudaPeekAtLastError(), "scale resident FP16 GEMV output");
}

std::uint16_t cpu_fp16(float value) {
  const auto converted = __float2half_rn(value);
  std::uint16_t bits{};
  static_assert(sizeof(bits) == sizeof(converted));
  std::memcpy(&bits, &converted, sizeof(bits));
  return bits;
}

float cpu_fp16_value(std::uint16_t value) {
  __half converted{};
  std::memcpy(&converted, &value, sizeof(value));
  return __half2float(converted);
}

float cpu_fp4(std::uint8_t code) {
  constexpr float values[8]{0.0F, 0.5F, 1.0F, 1.5F,
                            2.0F, 3.0F, 4.0F, 6.0F};
  const auto value = values[code & 7U];
  return (code & 8U) == 0U ? value : -value;
}

struct OracleResult final {
  double maximum_absolute_error{};
  double relative_l2_error{};
  bool decoded_bitwise{};
};

OracleResult run_oracle(int device) {
  constexpr std::uint32_t hidden = 128U;
  constexpr std::uint32_t intermediate = 96U;
  constexpr auto matrix_values =
      static_cast<std::size_t>(hidden) * intermediate;
  std::vector<std::uint8_t> packed(matrix_values / 2U);
  std::vector<std::uint8_t> scales(matrix_values / 32U);
  std::vector<std::uint16_t> decoded(matrix_values);
  for (std::uint32_t row = 0; row < intermediate; ++row) {
    for (std::uint32_t block = 0; block < hidden / 32U; ++block)
      scales[static_cast<std::size_t>(row) * (hidden / 32U) + block] =
          static_cast<std::uint8_t>(117U + (row + block) % 8U);
    for (std::uint32_t column = 0; column < hidden; column += 2U) {
      const auto low = static_cast<std::uint8_t>((row + column) % 16U);
      const auto high =
          static_cast<std::uint8_t>((3U * row + column + 1U) % 16U);
      packed[(static_cast<std::size_t>(row) * hidden + column) / 2U] =
          static_cast<std::uint8_t>(low | (high << 4U));
    }
  }
  for (std::uint32_t row = 0; row < intermediate; ++row)
    for (std::uint32_t column = 0; column < hidden; ++column) {
      const auto byte =
          packed[(static_cast<std::size_t>(row) * hidden + column) / 2U];
      const auto code = static_cast<std::uint8_t>(
          (column & 1U) == 0U ? byte & 0x0fU : byte >> 4U);
      const auto scale = std::ldexp(
          1.0F,
          static_cast<int>(scales[static_cast<std::size_t>(row) *
                                  (hidden / 32U) + column / 32U]) -
              127);
      decoded[static_cast<std::size_t>(row) * hidden + column] =
          cpu_fp16(cpu_fp4(code) * scale);
    }

  std::vector<float> input(hidden);
  for (std::uint32_t index = 0; index < hidden; ++index)
    input[index] = std::sin(static_cast<float>(index + 1U) * 0.071F) * 0.7F;
  const auto maximum = *std::max_element(
      input.begin(), input.end(), [](float left, float right) {
        return std::abs(left) < std::abs(right);
      });
  const auto input_scale = std::abs(maximum) / 127.0F;
  std::vector<std::int8_t> q_input(hidden);
  for (std::uint32_t index = 0; index < hidden; ++index) {
    auto value = static_cast<int>(std::nearbyint(input[index] / input_scale));
    value = std::max(-127, std::min(127, value));
    q_input[index] = static_cast<std::int8_t>(value);
  }
  std::vector<float> expected(intermediate);
  for (std::uint32_t row = 0; row < intermediate; ++row) {
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < hidden; ++column)
      sum += static_cast<float>(q_input[column]) *
             cpu_fp16_value(decoded[static_cast<std::size_t>(row) * hidden +
                                     column]);
    expected[row] = sum * input_scale;
  }

  cuda_check(cudaSetDevice(device), "select oracle device");
  DeviceAllocation d_packed(device, packed.size());
  DeviceAllocation d_scales(device, scales.size());
  DeviceAllocation d_decoded(device, decoded.size() * sizeof(std::uint16_t));
  DeviceAllocation d_input(device, input.size() * sizeof(float));
  DeviceAllocation d_q_input(device, q_input.size());
  DeviceAllocation d_fp16_input(device, q_input.size() * sizeof(__half));
  DeviceAllocation d_scale(device, sizeof(float));
  DeviceAllocation d_output(device, expected.size() * sizeof(float));
  cuda_check(cudaMemcpy(d_packed.get(), packed.data(), packed.size(),
                        cudaMemcpyHostToDevice),
             "upload oracle FP4 values");
  cuda_check(cudaMemcpy(d_scales.get(), scales.data(), scales.size(),
                        cudaMemcpyHostToDevice),
             "upload oracle scales");
  cuda_check(cudaMemcpy(d_input.get(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice),
             "upload oracle input");
  decode_fp4_fp16_kernel<<<48U, kThreads>>>(
      static_cast<const std::uint8_t*>(d_packed.get()),
      static_cast<const std::uint8_t*>(d_scales.get()),
      static_cast<__half*>(d_decoded.get()), intermediate, hidden);
  cuda_check(cudaPeekAtLastError(), "launch oracle FP4 decode");
  quantize_q8_kernel<<<1U, kThreads>>>(
      static_cast<const float*>(d_input.get()),
      static_cast<std::int8_t*>(d_q_input.get()),
      static_cast<float*>(d_scale.get()), hidden);
  cuda_check(cudaPeekAtLastError(), "launch oracle Q8 quantization");
  q8_to_fp16_kernel<<<1U, kThreads>>>(
      static_cast<const std::int8_t*>(d_q_input.get()),
      static_cast<__half*>(d_fp16_input.get()), hidden);
  cuda_check(cudaPeekAtLastError(), "convert oracle Q8 input to FP16");
  cublasHandle_t blas{};
  cublas_check(cublasCreate(&blas), "create oracle cuBLAS handle");
  try {
    launch_gemv(blas, static_cast<const __half*>(d_decoded.get()),
                static_cast<const __half*>(d_fp16_input.get()),
                static_cast<const float*>(d_scale.get()),
                static_cast<float*>(d_output.get()), intermediate, hidden,
                nullptr);
  } catch (...) {
    static_cast<void>(cublasDestroy(blas));
    throw;
  }
  cublas_check(cublasDestroy(blas), "destroy oracle cuBLAS handle");
  cuda_check(cudaDeviceSynchronize(), "complete numerical oracle");

  std::vector<std::uint16_t> actual_decoded(decoded.size());
  std::vector<float> actual(expected.size());
  cuda_check(cudaMemcpy(actual_decoded.data(), d_decoded.get(),
                        actual_decoded.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost),
             "download decoded oracle weights");
  cuda_check(cudaMemcpy(actual.data(), d_output.get(),
                        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "download oracle output");
  double maximum_error = 0.0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const auto error = static_cast<double>(actual[index]) - expected[index];
    maximum_error = std::max(maximum_error, std::abs(error));
    squared_error += error * error;
    squared_reference +=
        static_cast<double>(expected[index]) * expected[index];
  }
  return {maximum_error,
          std::sqrt(squared_error / std::max(squared_reference, 1.0e-30)),
          actual_decoded == decoded};
}

struct CardResult final {
  int device{};
  std::string name;
  std::uint64_t free_before{};
  std::uint64_t arena_bytes{};
  std::vector<float> pass_ms;
  std::string error;
};

class CardContext final {
 public:
  CardContext(int device, std::uint32_t layers, std::uint32_t hidden,
              std::uint32_t local_intermediate, std::uint64_t arena_bytes)
      : device_(device), layers_(layers), hidden_(hidden),
        local_intermediate_(local_intermediate),
        weights_(device, static_cast<std::size_t>(arena_bytes)),
        input_(device, hidden * sizeof(float)),
        q_input_(device, hidden * sizeof(std::int8_t)),
        fp16_input_(device, hidden * sizeof(__half)),
        input_scale_(device, sizeof(float)),
        gate_up_(device, 2ULL * local_intermediate * sizeof(float)),
        q_intermediate_(device,
                        local_intermediate * sizeof(std::int8_t)),
        fp16_intermediate_(device, local_intermediate * sizeof(__half)),
        intermediate_scale_(device, sizeof(float)),
        output_(device, hidden * sizeof(float)),
        host_input_(hidden), host_output_(hidden) {
    cuda_check(cudaSetDevice(device_), "select P100 context");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device_),
               "query P100 properties");
    if (properties.major != 6 || properties.minor != 0)
      throw std::runtime_error("resident MLP gate requires SM60 P100 devices");
    name_ = properties.name;
    for (std::uint32_t index = 0; index < hidden_; ++index)
      host_input_.get()[index] =
          std::sin(static_cast<float>(index + 1U) * 0.013F) * 0.25F;
    const auto values = weights_.bytes() / sizeof(std::uint16_t);
    fill_fp16_kernel<<<65535U, kThreads>>>(
        static_cast<std::uint16_t*>(weights_.get()), values, 0x1400U);
    cuda_check(cudaPeekAtLastError(), "initialize resident FP16 arena");
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create P100 gate stream");
    cuda_check(cudaEventCreate(&begin_), "create P100 begin event");
    cuda_check(cudaEventCreate(&end_), "create P100 end event");
    cublas_check(cublasCreate(&blas_), "create P100 cuBLAS handle");
    cublas_check(cublasSetStream(blas_, stream_), "bind P100 cuBLAS stream");
    cuda_check(cudaDeviceSynchronize(), "complete resident arena initialization");
    run_layer(0U);
    cuda_check(cudaStreamSynchronize(stream_), "warm resident MLP layer");
  }

  ~CardContext() {
    static_cast<void>(cudaSetDevice(device_));
    if (blas_) static_cast<void>(cublasDestroy(blas_));
    if (end_) static_cast<void>(cudaEventDestroy(end_));
    if (begin_) static_cast<void>(cudaEventDestroy(begin_));
    if (stream_) static_cast<void>(cudaStreamDestroy(stream_));
  }
  CardContext(const CardContext&) = delete;
  CardContext& operator=(const CardContext&) = delete;

  [[nodiscard]] float run_pass() {
    cuda_check(cudaSetDevice(device_), "select P100 measurement device");
    cuda_check(cudaEventRecord(begin_, stream_), "record P100 pass begin");
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
      run_layer(layer);
      cuda_check(cudaStreamSynchronize(stream_),
                 "synchronize P100 MLP layer boundary");
    }
    cuda_check(cudaEventRecord(end_, stream_), "record P100 pass end");
    cuda_check(cudaEventSynchronize(end_), "wait P100 pass");
    float elapsed{};
    cuda_check(cudaEventElapsedTime(&elapsed, begin_, end_),
               "measure P100 pass");
    return elapsed;
  }

  [[nodiscard]] const std::string& name() const noexcept { return name_; }

 private:
  void run_layer(std::uint32_t layer) {
    const auto matrix_values =
        static_cast<std::size_t>(hidden_) * local_intermediate_;
    const auto layer_values = 3U * matrix_values;
    const auto* layer_weights =
        static_cast<const std::uint16_t*>(weights_.get()) +
        static_cast<std::size_t>(layer) * layer_values;
    const auto* gate_weights = layer_weights;
    const auto* up_weights = layer_weights + matrix_values;
    const auto* down_weights = layer_weights + 2U * matrix_values;
    cuda_check(cudaMemcpyAsync(input_.get(), host_input_.get(),
                               hidden_ * sizeof(float), cudaMemcpyHostToDevice,
                               stream_),
               "bounce MLP input to P100");
    quantize_q8_kernel<<<1U, kThreads, 0, stream_>>>(
        static_cast<const float*>(input_.get()),
        static_cast<std::int8_t*>(q_input_.get()),
        static_cast<float*>(input_scale_.get()), hidden_);
    cuda_check(cudaPeekAtLastError(), "quantize P100 MLP input");
    q8_to_fp16_kernel<<<(hidden_ + kThreads - 1U) / kThreads, kThreads, 0,
                         stream_>>>(
        static_cast<const std::int8_t*>(q_input_.get()),
        static_cast<__half*>(fp16_input_.get()), hidden_);
    cuda_check(cudaPeekAtLastError(), "convert P100 MLP input to FP16");
    launch_gemv(blas_, reinterpret_cast<const __half*>(gate_weights),
                static_cast<const __half*>(fp16_input_.get()),
                static_cast<const float*>(input_scale_.get()),
                static_cast<float*>(gate_up_.get()),
                2U * local_intermediate_, hidden_, stream_);
    silu_product_kernel<<<
        (local_intermediate_ + kThreads - 1U) / kThreads, kThreads, 0,
        stream_>>>(static_cast<float*>(gate_up_.get()),
                   static_cast<const float*>(gate_up_.get()) +
                       local_intermediate_,
                   local_intermediate_);
    cuda_check(cudaPeekAtLastError(), "launch P100 SwiGLU");
    quantize_q8_kernel<<<1U, kThreads, 0, stream_>>>(
        static_cast<const float*>(gate_up_.get()),
        static_cast<std::int8_t*>(q_intermediate_.get()),
        static_cast<float*>(intermediate_scale_.get()), local_intermediate_);
    cuda_check(cudaPeekAtLastError(), "quantize P100 MLP intermediate");
    q8_to_fp16_kernel<<<(local_intermediate_ + kThreads - 1U) / kThreads,
                         kThreads, 0, stream_>>>(
        static_cast<const std::int8_t*>(q_intermediate_.get()),
        static_cast<__half*>(fp16_intermediate_.get()), local_intermediate_);
    cuda_check(cudaPeekAtLastError(),
               "convert P100 MLP intermediate to FP16");
    launch_gemv(blas_, reinterpret_cast<const __half*>(down_weights),
                static_cast<const __half*>(fp16_intermediate_.get()),
                static_cast<const float*>(intermediate_scale_.get()),
                static_cast<float*>(output_.get()), hidden_,
                local_intermediate_, stream_);
    cuda_check(cudaMemcpyAsync(host_output_.get(), output_.get(),
                               hidden_ * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_),
               "bounce MLP output from P100");
  }

  int device_{};
  std::uint32_t layers_{}, hidden_{}, local_intermediate_{};
  std::string name_;
  DeviceAllocation weights_, input_, q_input_, fp16_input_, input_scale_,
      gate_up_, q_intermediate_, fp16_intermediate_, intermediate_scale_,
      output_;
  PinnedAllocation<float> host_input_, host_output_;
  cudaStream_t stream_{};
  cudaEvent_t begin_{}, end_{};
  cublasHandle_t blas_{};
};

double median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2U];
}

void print_json_string(std::string_view value) {
  std::cout << '"';
  for (const auto character : value) {
    if (character == '"' || character == '\\') std::cout << '\\';
    std::cout << character;
  }
  std::cout << '"';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: expert-pascal-resident-mlp-gate <artifact>\n";
      return 64;
    }
    int device_count{};
    cuda_check(cudaGetDeviceCount(&device_count), "enumerate CUDA devices");
    std::vector<int> pascal_devices;
    int primary_device = -1;
    for (int device = 0; device < device_count; ++device) {
      cudaDeviceProp properties{};
      cuda_check(cudaGetDeviceProperties(&properties, device),
                 "query CUDA device");
      if (properties.major == 6 && properties.minor == 0)
        pascal_devices.push_back(device);
      if (properties.major == 8 && properties.minor == 6)
        primary_device = device;
    }
    if (pascal_devices.size() != 2U || primary_device < 0)
      throw std::runtime_error(
          "gate requires exactly two SM60 P100s and one SM86 primary GPU");

    expert::runtime::ModelArtifact artifact;
    const auto loaded = expert::runtime::ModelArtifact::load(argv[1], artifact);
    if (!loaded.ok())
      throw std::runtime_error(std::string(loaded.message()));
    const auto plan = make_plan(artifact);
    const auto fp16_range_pass = plan.minimum_scale_code >= 104U &&
                                 plan.maximum_scale_code <= 140U;

    std::array<std::uint64_t, 2> free_bytes{};
    std::array<std::uint64_t, 2> total_bytes{};
    for (std::size_t index = 0; index < pascal_devices.size(); ++index) {
      cuda_check(cudaSetDevice(pascal_devices[index]), "select P100 capacity device");
      std::size_t free{}, total{};
      cuda_check(cudaMemGetInfo(&free, &total), "query P100 capacity");
      free_bytes[index] = free;
      total_bytes[index] = total;
    }
    const auto minimum_free = *std::min_element(free_bytes.begin(),
                                                free_bytes.end());
    if (minimum_free <= kDeviceReserveBytes + 64U * kMiB)
      throw std::runtime_error("P100 capacity has no admission headroom");
    const auto bytes_per_local_intermediate = checked_multiply(
        checked_multiply(plan.layers, 3ULL * plan.hidden, "MLP layer bytes"),
        sizeof(std::uint16_t), "decoded FP16 bytes");
    auto local_intermediate = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        plan.intermediate / 2U,
        (minimum_free - kDeviceReserveBytes - 64U * kMiB) /
            bytes_per_local_intermediate));
    local_intermediate = local_intermediate / 32U * 32U;
    if (local_intermediate == 0U)
      throw std::runtime_error("P100 capacity fits no aligned MLP shard");
    const auto arena_bytes = checked_multiply(
        bytes_per_local_intermediate, local_intermediate,
        "resident P100 arena");
    const auto offloaded_compact_bytes = checked_multiply(
        checked_multiply(
            checked_multiply(plan.layers, 3ULL * plan.hidden,
                             "offloaded MLP rows"),
            2ULL * local_intermediate, "offloaded P100 pair"),
        17U, "offloaded FP4 block bytes") /
        32U;
    if (offloaded_compact_bytes > plan.mlp_compact_bytes ||
        offloaded_compact_bytes > plan.hot_weight_bytes)
      throw std::runtime_error("offloaded MLP accounting exceeds artifact");
    const auto remaining_weight_bytes =
        plan.hot_weight_bytes - offloaded_compact_bytes;

    cuda_check(cudaSetDevice(primary_device), "select primary capacity device");
    std::size_t primary_free{}, primary_total{};
    cuda_check(cudaMemGetInfo(&primary_free, &primary_total),
               "query primary capacity");
    auto primary_required = checked_add(plan.exact_kv_bytes,
                                        remaining_weight_bytes,
                                        "primary KV and weights");
    primary_required = checked_add(primary_required,
                                   plan.recurrent_state_bytes,
                                   "primary recurrent state");
    primary_required = checked_add(primary_required,
                                   kWorkspaceAllowanceBytes,
                                   "primary workspace allowance");
    primary_required = checked_add(primary_required, kDeviceReserveBytes,
                                   "primary emergency reserve");
    const auto capacity_pass = primary_required <= primary_free;

    const auto oracle = run_oracle(pascal_devices.front());
    const auto numerical_pass = fp16_range_pass && oracle.decoded_bitwise &&
                                oracle.relative_l2_error <= 1.0e-5 &&
                                oracle.maximum_absolute_error <= 1.0e-4;

    std::array<CardResult, 2> results;
    std::array<std::unique_ptr<CardContext>, 2> contexts;
    for (std::size_t index = 0; index < contexts.size(); ++index) {
      results[index].device = pascal_devices[index];
      results[index].free_before = free_bytes[index];
      results[index].arena_bytes = arena_bytes;
      contexts[index] = std::make_unique<CardContext>(
          pascal_devices[index], plan.layers, plan.hidden,
          local_intermediate, arena_bytes);
      results[index].name = contexts[index]->name();
    }

    std::barrier start_barrier(3);
    std::array<std::thread, 2> workers;
    for (std::size_t index = 0; index < workers.size(); ++index) {
      workers[index] = std::thread([&, index] {
        try {
          start_barrier.arrive_and_wait();
          for (std::uint32_t pass = 0; pass < 3U; ++pass)
            results[index].pass_ms.push_back(contexts[index]->run_pass());
        } catch (const std::exception& error) {
          results[index].error = error.what();
        }
      });
    }
    const auto wall_started = std::chrono::steady_clock::now();
    start_barrier.arrive_and_wait();
    for (auto& worker : workers) worker.join();
    const auto parallel_wall_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() -
                                      wall_started)
                                      .count() /
                                  3.0;

    bool throughput_pass = true;
    double slowest_ms = 0.0;
    std::array<double, 2> median_ms{}, bytes_per_second{};
    for (std::size_t index = 0; index < results.size(); ++index) {
      if (!results[index].error.empty() || results[index].pass_ms.size() != 3U) {
        throughput_pass = false;
        continue;
      }
      median_ms[index] = median(results[index].pass_ms);
      slowest_ms = std::max(slowest_ms, median_ms[index]);
      bytes_per_second[index] =
          static_cast<double>(arena_bytes) * 1000.0 / median_ms[index];
      throughput_pass = throughput_pass &&
                        bytes_per_second[index] >= kRequiredBytesPerSecond;
    }
    const auto pass = capacity_pass && numerical_pass && throughput_pass;

    std::cout << "{\"type\":\"pascal_resident_mlp_gate\""
              << ",\"artifact_schema\":" << artifact.model().schema_version
              << ",\"layers\":" << plan.layers
              << ",\"hidden\":" << plan.hidden
              << ",\"intermediate\":" << plan.intermediate
              << ",\"local_intermediate_per_p100\":" << local_intermediate
              << ",\"offloaded_intermediate\":"
              << 2U * local_intermediate
              << ",\"hot_weight_bytes\":" << plan.hot_weight_bytes
              << ",\"mlp_compact_bytes\":" << plan.mlp_compact_bytes
              << ",\"offloaded_compact_bytes\":"
              << offloaded_compact_bytes
              << ",\"remaining_primary_weight_bytes\":"
              << remaining_weight_bytes
              << ",\"exact_kv_bytes\":" << plan.exact_kv_bytes
              << ",\"recurrent_state_bytes\":"
              << plan.recurrent_state_bytes
              << ",\"minimum_scale_code\":"
              << static_cast<unsigned>(plan.minimum_scale_code)
              << ",\"maximum_scale_code\":"
              << static_cast<unsigned>(plan.maximum_scale_code)
              << ",\"fp16_range_pass\":"
              << (fp16_range_pass ? "true" : "false")
              << ",\"primary_free_bytes\":" << primary_free
              << ",\"primary_required_bytes\":" << primary_required
              << ",\"capacity_pass\":"
              << (capacity_pass ? "true" : "false")
              << ",\"oracle_decoded_bitwise\":"
              << (oracle.decoded_bitwise ? "true" : "false")
              << ",\"oracle_max_abs\":" << oracle.maximum_absolute_error
              << ",\"oracle_relative_l2\":" << oracle.relative_l2_error
              << ",\"numerical_pass\":"
              << (numerical_pass ? "true" : "false")
              << ",\"required_bytes_per_second\":"
              << kRequiredBytesPerSecond
              << ",\"parallel_wall_ms_per_pass\":" << parallel_wall_ms
              << ",\"slowest_card_ms\":" << slowest_ms
              << ",\"cards\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << "{\"device\":" << results[index].device << ",\"name\":";
      print_json_string(results[index].name);
      std::cout << ",\"free_before\":" << results[index].free_before
                << ",\"total_bytes\":" << total_bytes[index]
                << ",\"arena_bytes\":" << arena_bytes
                << ",\"median_ms\":" << median_ms[index]
                << ",\"bytes_per_second\":" << bytes_per_second[index]
                << ",\"error\":";
      print_json_string(results[index].error);
      std::cout << '}';
    }
    std::cout << "],\"throughput_pass\":"
              << (throughput_pass ? "true" : "false")
              << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
    return pass ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "Pascal resident MLP gate: " << error.what() << '\n';
    return 1;
  }
}
