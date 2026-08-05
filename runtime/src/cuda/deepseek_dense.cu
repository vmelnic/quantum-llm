#include "expert/runtime/cuda/deepseek_dense.hpp"

#include "expert/runtime/cuda/deepseek_admission.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <string>

namespace expert::runtime::cuda {
namespace {

Status failure(cudaError_t error, const char* operation) {
  return {ErrorCode::upload_failed,
          std::string(operation) + ": " + cudaGetErrorString(error)};
}

}  // namespace

DeepSeekDenseMatrix::DeepSeekDenseMatrix(std::int8_t* weights, float* scales,
                                         std::uint32_t rows,
                                         std::uint32_t columns) noexcept
    : weights_(weights), scales_(scales), rows_(rows), columns_(columns) {}

DeepSeekDenseMatrix::~DeepSeekDenseMatrix() {
  if (weights_) static_cast<void>(cudaFree(weights_));
  if (scales_) static_cast<void>(cudaFree(scales_));
}

Int8Matrix DeepSeekDenseMatrix::view() const noexcept {
  return {weights_, scales_, rows_, columns_};
}

std::uint64_t DeepSeekDenseMatrix::bytes() const noexcept {
  return static_cast<std::uint64_t>(rows_) * columns_ +
         static_cast<std::uint64_t>(rows_) * sizeof(float);
}

DeepSeekDenseAdmissionResult admit_deepseek_dense_matrix(
    std::span<const std::byte> weights, std::span<const std::byte> scales,
    std::uint32_t rows, std::uint32_t columns) noexcept {
  if (rows == 0U || columns == 0U || rows % 128U != 0U ||
      columns % 128U != 0U ||
      static_cast<std::uint64_t>(rows) * columns != weights.size() ||
      static_cast<std::uint64_t>(rows / 128U) * (columns / 128U) !=
          scales.size() ||
      static_cast<std::uint64_t>(rows) * columns >
          std::numeric_limits<std::size_t>::max()) {
    return {{ErrorCode::invalid_argument,
             "invalid DeepSeek dense admission geometry"},
            {}};
  }
  std::uint8_t* source_weights = nullptr;
  std::uint8_t* source_scales = nullptr;
  std::int8_t* target_weights = nullptr;
  float* target_scales = nullptr;
  auto error = cudaMalloc(reinterpret_cast<void**>(&source_weights),
                          weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(reinterpret_cast<void**>(&source_scales), scales.size());
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(reinterpret_cast<void**>(&target_weights), weights.size());
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(reinterpret_cast<void**>(&target_scales),
                       static_cast<std::size_t>(rows) * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(source_weights, weights.data(), weights.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(source_scales, scales.data(), scales.size(),
                       cudaMemcpyHostToDevice);
  }
  Status status = error == cudaSuccess
                      ? admit_deepseek_fp8_projection(
                            {source_weights, source_scales, target_weights,
                             target_scales, rows, columns, nullptr})
                      : failure(error, "DeepSeek dense allocation/H2D");
  if (source_weights) static_cast<void>(cudaFree(source_weights));
  if (source_scales) static_cast<void>(cudaFree(source_scales));
  if (!status.ok()) {
    if (target_weights) static_cast<void>(cudaFree(target_weights));
    if (target_scales) static_cast<void>(cudaFree(target_scales));
    return {status, {}};
  }
  return {Status::success(),
          std::make_shared<DeepSeekDenseMatrix>(target_weights, target_scales,
                                                rows, columns)};
}

}  // namespace expert::runtime::cuda
