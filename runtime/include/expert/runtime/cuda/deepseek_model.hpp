#pragma once

#include "expert/runtime/cuda/deepseek_dense.hpp"
#include "expert/runtime/cuda/deepseek_io.hpp"
#include "expert/runtime/cuda/deepseek_mtp.hpp"
#include "expert/runtime/cuda/deepseek_typed.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace expert::runtime::cuda {

enum class DeepSeekRouterKind : std::uint8_t { hash, learned };

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

// Non-owning controls for one FFN sublayer. Shared and routed expert weights
// live in the expert directory; this binding owns only the resident model
// tensors required before and after expert dispatch.
struct DeepSeekFfnBinding final {
  std::uint32_t layer{};
  bool hash_router{};

  const std::uint16_t* ffn_norm{};
  const std::uint16_t* router_weight{};
  const std::int64_t* token_experts{};
  const float* router_bias{};
  const float* hca_function{};
  const float* hca_base{};
  const float* hca_scale{};
};

// Owns an arbitrary authenticated DeepSeek tensor namespace. Bindings are
// formed by prefix, so the same attention/FFN kernels serve `layers.N`,
// `mtp.N`, or a future remotely placed block without model-specific loaders.
class DeepSeekResidentTensorState final {
 public:
  DeepSeekResidentTensorState() = default;
  DeepSeekResidentTensorState(const DeepSeekResidentTensorState&) = delete;
  DeepSeekResidentTensorState& operator=(const DeepSeekResidentTensorState&) = delete;
  DeepSeekResidentTensorState(DeepSeekResidentTensorState&&) noexcept = default;
  DeepSeekResidentTensorState& operator=(DeepSeekResidentTensorState&&) noexcept = default;

  [[nodiscard]] static Status load(
      IAsyncStorage& storage, FixedBufferPool& buffers,
      std::span<const DeepSeekDenseSpec> dense_specs,
      std::span<const DeepSeekTypedSpec> typed_specs,
      DeepSeekResidentTensorState& destination);
  [[nodiscard]] Status bind_attention(
      std::string_view prefix, std::uint32_t logical_layer,
      std::uint32_t compress_ratio,
      DeepSeekAttentionBinding& destination) const noexcept;
  [[nodiscard]] Status bind_ffn(
      std::string_view prefix, std::uint32_t logical_layer,
      DeepSeekRouterKind router,
      DeepSeekFfnBinding& destination) const noexcept;
  [[nodiscard]] Status bind_io(DeepSeekIoBinding& destination) const noexcept;
  [[nodiscard]] Status bind_mtp_glue(
      std::string_view prefix,
      DeepSeekMtpGlueBinding& destination) const noexcept;

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
  [[nodiscard]] Status bind_ffn(
      std::uint32_t layer, DeepSeekFfnBinding& destination) const noexcept;
  [[nodiscard]] Status bind_io(
      DeepSeekIoBinding& destination) const noexcept;

  [[nodiscard]] std::uint64_t bytes() const noexcept {
    return tensors_.bytes();
  }
  [[nodiscard]] std::size_t dense_size() const noexcept {
    return tensors_.dense_size();
  }
  [[nodiscard]] std::size_t typed_size() const noexcept {
    return tensors_.typed_size();
  }
  void clear() noexcept;

 private:
  DeepSeekResidentTensorState tensors_;
};

}  // namespace expert::runtime::cuda
