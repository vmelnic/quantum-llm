#pragma once

#include "expert/runtime/cuda/deepseek_dense.hpp"
#include "expert/runtime/cuda/deepseek_typed.hpp"

#include <cstdint>
#include <span>

namespace expert::runtime::cuda {

// Non-owning, geometry-checked pointers for one attention sublayer. The
// resident model state must outlive every binding produced from it.
struct DeepSeekAttentionBinding final {
  std::uint32_t layer{};
  std::uint32_t compress_ratio{};

  Int8Matrix wq_a;
  Int8Matrix wq_b;
  Int8Matrix wkv;
  Int8Matrix wo_a;
  Int8Matrix wo_b;
  Int8Matrix index_wq_b;

  const std::uint16_t* attention_norm{};
  const std::uint16_t* query_norm{};
  const std::uint16_t* kv_norm{};
  const float* attention_sink{};
  const float* hca_function{};
  const float* hca_base{};
  const float* hca_scale{};

  const std::uint16_t* compressor_wkv{};
  const std::uint16_t* compressor_wgate{};
  const float* compressor_ape{};
  const std::uint16_t* compressor_norm{};

  const std::uint16_t* index_weights{};
  const std::uint16_t* index_compressor_wkv{};
  const std::uint16_t* index_compressor_wgate{};
  const float* index_compressor_ape{};
  const std::uint16_t* index_compressor_norm{};
};

// Publishes the FP8-expanded dense set and dtype-preserving model tensors as
// one transaction. A failed second phase cannot expose a partial model.
class DeepSeekResidentModelState final {
 public:
  DeepSeekResidentModelState() = default;
  DeepSeekResidentModelState(const DeepSeekResidentModelState&) = delete;
  DeepSeekResidentModelState& operator=(const DeepSeekResidentModelState&) = delete;
  DeepSeekResidentModelState(DeepSeekResidentModelState&&) noexcept = default;
  DeepSeekResidentModelState& operator=(DeepSeekResidentModelState&&) noexcept = default;

  [[nodiscard]] static Status load(
      IAsyncStorage& storage, FixedBufferPool& buffers,
      std::span<const DeepSeekDenseSpec> dense_specs,
      std::span<const DeepSeekTypedSpec> typed_specs,
      DeepSeekResidentModelState& destination);

  [[nodiscard]] Status bind_attention(
      std::uint32_t layer, std::uint32_t compress_ratio,
      DeepSeekAttentionBinding& destination) const noexcept;

  [[nodiscard]] std::uint64_t bytes() const noexcept {
    return dense_.bytes() + typed_.bytes();
  }
  [[nodiscard]] std::size_t dense_size() const noexcept { return dense_.size(); }
  [[nodiscard]] std::size_t typed_size() const noexcept { return typed_.size(); }
  void clear() noexcept;

 private:
  DeepSeekDenseSet dense_;
  DeepSeekTypedSet typed_;
};

}  // namespace expert::runtime::cuda
