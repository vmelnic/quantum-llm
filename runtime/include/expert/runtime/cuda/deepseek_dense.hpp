#pragma once

#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace expert::runtime::cuda {

class DeepSeekDenseMatrix final {
 public:
  ~DeepSeekDenseMatrix();
  DeepSeekDenseMatrix(const DeepSeekDenseMatrix&) = delete;
  DeepSeekDenseMatrix& operator=(const DeepSeekDenseMatrix&) = delete;

  [[nodiscard]] Int8Matrix view() const noexcept;
  [[nodiscard]] std::uint64_t bytes() const noexcept;

  DeepSeekDenseMatrix(std::int8_t* weights, float* scales,
                      std::uint32_t rows, std::uint32_t columns) noexcept;

 private:
  std::int8_t* weights_{};
  float* scales_{};
  std::uint32_t rows_{};
  std::uint32_t columns_{};
};

struct DeepSeekDenseAdmissionResult final {
  Status status;
  std::shared_ptr<DeepSeekDenseMatrix> matrix;
};

// Source spans are raw E4M3FN weights and UE8M0 128x128 block scales. The
// resulting matrix is the generic SM86 INT8-per-row dense ABI.
[[nodiscard]] DeepSeekDenseAdmissionResult admit_deepseek_dense_matrix(
    std::span<const std::byte> weights, std::span<const std::byte> scales,
    std::uint32_t rows, std::uint32_t columns) noexcept;

}  // namespace expert::runtime::cuda
