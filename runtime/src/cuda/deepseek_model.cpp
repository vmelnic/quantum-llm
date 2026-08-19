#include "expert/runtime/cuda/deepseek_model.hpp"

#include <string>
#include <utility>

namespace expert::runtime::cuda {
namespace {

bool bind_matrix(const DeepSeekDenseSet& set, const std::string& name,
                 std::uint32_t rows, std::uint32_t columns,
                 Int8Matrix& destination) {
  const auto* matrix = set.find(name);
  if (!matrix) return false;
  const auto view = matrix->view();
  if (view.rows != rows || view.columns != columns) return false;
  destination = view;
  return true;
}

template <typename T>
bool bind_tensor(const DeepSeekTypedSet& set, const std::string& name,
                 DeepSeekDtype dtype, std::uint64_t elements,
                 const T*& destination) {
  const auto* tensor = set.find(name);
  if (!tensor || tensor->dtype() != dtype ||
      tensor->bytes() != elements * sizeof(T))
    return false;
  destination = static_cast<const T*>(tensor->data());
  return true;
}

}  // namespace

Status DeepSeekResidentTensorState::load(
    IAsyncStorage& storage, FixedBufferPool& buffers,
    std::span<const DeepSeekDenseSpec> dense_specs,
    std::span<const DeepSeekTypedSpec> typed_specs,
    DeepSeekResidentTensorState& destination) {
  if (dense_specs.empty() || typed_specs.empty() ||
      destination.dense_size() != 0U || destination.typed_size() != 0U)
    return {ErrorCode::invalid_argument,
            "DeepSeek tensor load requires authenticated specs and an empty destination"};
  DeepSeekResidentTensorState candidate;
  auto status = DeepSeekDenseSet::load(storage, buffers, dense_specs,
                                       candidate.dense_);
  if (!status.ok()) return status;
  status = DeepSeekTypedSet::load(storage, buffers, typed_specs,
                                  candidate.typed_);
  if (!status.ok()) return status;
  destination = std::move(candidate);
  return Status::success();
}

Status DeepSeekResidentTensorState::bind_attention(
    std::string_view namespace_prefix, std::uint32_t logical_layer,
    std::uint32_t compress_ratio,
    DeepSeekAttentionBinding& destination) const noexcept {
  if (namespace_prefix.empty() ||
      (compress_ratio != 0U && compress_ratio != 4U &&
       compress_ratio != 128U))
    return {ErrorCode::invalid_argument,
            "invalid DeepSeek attention namespace or compression ratio"};

  DeepSeekAttentionBinding bound;
  bound.layer = logical_layer;
  bound.compress_ratio = compress_ratio;
  const std::string prefix(namespace_prefix);
  const auto attention = prefix + ".attn";
  bool valid =
      bind_matrix(dense_, attention + ".wq_a", 1024U, 4096U, bound.wq_a) &&
      bind_matrix(dense_, attention + ".wq_b", 32768U, 1024U, bound.wq_b) &&
      bind_matrix(dense_, attention + ".wkv", 512U, 4096U, bound.wkv) &&
      bind_matrix(dense_, attention + ".wo_a", 8192U, 4096U, bound.wo_a) &&
      bind_matrix(dense_, attention + ".wo_b", 4096U, 8192U, bound.wo_b) &&
      bind_tensor(typed_, prefix + ".attn_norm.weight", DeepSeekDtype::bf16,
                  4096U, bound.attention_norm) &&
      bind_tensor(typed_, attention + ".q_norm.weight", DeepSeekDtype::bf16,
                  1024U, bound.query_norm) &&
      bind_tensor(typed_, attention + ".kv_norm.weight", DeepSeekDtype::bf16,
                  512U, bound.kv_norm) &&
      bind_tensor(typed_, attention + ".attn_sink", DeepSeekDtype::f32,
                  64U, bound.attention_sink) &&
      bind_tensor(typed_, prefix + ".hc_attn_fn", DeepSeekDtype::f32,
                  24U * 4U * 4096U, bound.hca_function) &&
      bind_tensor(typed_, prefix + ".hc_attn_base", DeepSeekDtype::f32,
                  24U, bound.hca_base) &&
      bind_tensor(typed_, prefix + ".hc_attn_scale", DeepSeekDtype::f32,
                  3U, bound.hca_scale);

  if (compress_ratio != 0U) {
    const auto compressor = attention + ".compressor";
    const auto width = compress_ratio == 4U ? 1024U : 512U;
    valid = valid &&
        bind_tensor(typed_, compressor + ".wkv.weight", DeepSeekDtype::bf16,
                    static_cast<std::uint64_t>(width) * 4096U,
                    bound.compressor_wkv) &&
        bind_tensor(typed_, compressor + ".wgate.weight", DeepSeekDtype::bf16,
                    static_cast<std::uint64_t>(width) * 4096U,
                    bound.compressor_wgate) &&
        bind_tensor(typed_, compressor + ".ape", DeepSeekDtype::f32,
                    static_cast<std::uint64_t>(compress_ratio) * width,
                    bound.compressor_ape) &&
        bind_tensor(typed_, compressor + ".norm.weight", DeepSeekDtype::bf16,
                    512U, bound.compressor_norm);
  }

  if (compress_ratio == 4U) {
    const auto indexer = attention + ".indexer";
    const auto compressor = indexer + ".compressor";
    valid = valid &&
        bind_matrix(dense_, indexer + ".wq_b", 8192U, 1024U,
                    bound.index_wq_b) &&
        bind_tensor(typed_, indexer + ".weights_proj.weight",
                    DeepSeekDtype::bf16, 64U * 4096U,
                    bound.index_weights) &&
        bind_tensor(typed_, compressor + ".wkv.weight", DeepSeekDtype::bf16,
                    256U * 4096U, bound.index_compressor_wkv) &&
        bind_tensor(typed_, compressor + ".wgate.weight", DeepSeekDtype::bf16,
                    256U * 4096U, bound.index_compressor_wgate) &&
        bind_tensor(typed_, compressor + ".ape", DeepSeekDtype::f32,
                    4U * 256U, bound.index_compressor_ape) &&
        bind_tensor(typed_, compressor + ".norm.weight", DeepSeekDtype::bf16,
                    128U, bound.index_compressor_norm);
  }

  if (!valid)
    return {ErrorCode::invalid_argument,
            "DeepSeek attention binding is missing or has incompatible model state"};
  destination = bound;
  return Status::success();
}

Status DeepSeekResidentTensorState::bind_ffn(
    std::string_view namespace_prefix, std::uint32_t logical_layer,
    DeepSeekRouterKind router,
    DeepSeekFfnBinding& destination) const noexcept {
  if (namespace_prefix.empty())
    return {ErrorCode::invalid_argument, "invalid DeepSeek FFN namespace"};
  DeepSeekFfnBinding bound;
  bound.layer = logical_layer;
  bound.hash_router = router == DeepSeekRouterKind::hash;
  const std::string prefix(namespace_prefix);
  const auto ffn = prefix + ".ffn";
  bool valid =
      bind_tensor(typed_, prefix + ".ffn_norm.weight", DeepSeekDtype::bf16,
                  4096U, bound.ffn_norm) &&
      bind_tensor(typed_, ffn + ".gate.weight", DeepSeekDtype::bf16,
                  256U * 4096U, bound.router_weight) &&
      bind_tensor(typed_, prefix + ".hc_ffn_fn", DeepSeekDtype::f32,
                  24U * 4U * 4096U, bound.hca_function) &&
      bind_tensor(typed_, prefix + ".hc_ffn_base", DeepSeekDtype::f32,
                  24U, bound.hca_base) &&
      bind_tensor(typed_, prefix + ".hc_ffn_scale", DeepSeekDtype::f32,
                  3U, bound.hca_scale);
  if (bound.hash_router) {
    valid = valid &&
        bind_tensor(typed_, ffn + ".gate.tid2eid", DeepSeekDtype::i64,
                    129280U * 6U, bound.token_experts);
  } else {
    valid = valid &&
        bind_tensor(typed_, ffn + ".gate.bias", DeepSeekDtype::f32,
                    256U, bound.router_bias);
  }
  if (!valid)
    return {ErrorCode::invalid_argument,
            "DeepSeek FFN binding is missing or has incompatible model state"};
  destination = bound;
  return Status::success();
}

Status DeepSeekResidentTensorState::bind_io(
    DeepSeekIoBinding& destination) const noexcept {
  DeepSeekIoBinding bound;
  const bool valid =
      bind_tensor(typed_, "embed.weight", DeepSeekDtype::bf16,
                  static_cast<std::uint64_t>(kDeepSeekVocab) *
                      kDeepSeekHidden,
                  bound.embedding) &&
      bind_tensor(typed_, "norm.weight", DeepSeekDtype::bf16,
                  kDeepSeekHidden, bound.final_norm) &&
      bind_tensor(typed_, "head.weight", DeepSeekDtype::bf16,
                  static_cast<std::uint64_t>(kDeepSeekVocab) *
                      kDeepSeekHidden,
                  bound.head) &&
      bind_tensor(typed_, "hc_head_fn", DeepSeekDtype::f32,
                  4ULL * 4U * kDeepSeekHidden, bound.head_function) &&
      bind_tensor(typed_, "hc_head_base", DeepSeekDtype::f32, 4U,
                  bound.head_base) &&
      bind_tensor(typed_, "hc_head_scale", DeepSeekDtype::f32, 1U,
                  bound.head_scale);
  if (!valid)
    return {ErrorCode::invalid_argument,
            "DeepSeek I/O binding is missing or has incompatible model state"};
  destination = bound;
  return Status::success();
}

Status DeepSeekResidentTensorState::bind_mtp_glue(
    std::string_view namespace_prefix,
    DeepSeekMtpGlueBinding& destination) const noexcept {
  if (namespace_prefix.empty())
    return {ErrorCode::invalid_argument, "invalid DeepSeek MTP namespace"};
  const std::string prefix(namespace_prefix);
  DeepSeekMtpGlueBinding bound;
  const bool valid =
      bind_matrix(dense_, prefix + ".e_proj", 4096U, 4096U,
                  bound.e_projection) &&
      bind_matrix(dense_, prefix + ".h_proj", 4096U, 4096U,
                  bound.h_projection) &&
      bind_tensor(typed_, prefix + ".enorm.weight", DeepSeekDtype::bf16,
                  4096U, bound.embedding_norm) &&
      bind_tensor(typed_, prefix + ".hnorm.weight", DeepSeekDtype::bf16,
                  4096U, bound.hidden_norm) &&
      bind_tensor(typed_, prefix + ".norm.weight", DeepSeekDtype::bf16,
                  4096U, bound.output_norm) &&
      bind_tensor(typed_, prefix + ".hc_head_fn", DeepSeekDtype::f32,
                  4ULL * 4U * 4096U, bound.head_function) &&
      bind_tensor(typed_, prefix + ".hc_head_base", DeepSeekDtype::f32,
                  4U, bound.head_base) &&
      bind_tensor(typed_, prefix + ".hc_head_scale", DeepSeekDtype::f32,
                  1U, bound.head_scale);
  if (!valid)
    return {ErrorCode::invalid_argument,
            "DeepSeek MTP glue binding is missing or incompatible"};
  destination = bound;
  return Status::success();
}

void DeepSeekResidentTensorState::clear() noexcept {
  dense_.clear();
  typed_.clear();
}

Status DeepSeekResidentModelState::load(
    IAsyncStorage& storage, FixedBufferPool& buffers,
    std::span<const DeepSeekDenseSpec> dense_specs,
    std::span<const DeepSeekTypedSpec> typed_specs,
    DeepSeekResidentModelState& destination) {
  if (dense_specs.size() != 236U || typed_specs.size() != 834U ||
      destination.dense_size() != 0U || destination.typed_size() != 0U)
    return {ErrorCode::invalid_argument,
            "DeepSeek model load requires the complete pinned state and an empty destination"};
  DeepSeekResidentModelState candidate;
  auto status = DeepSeekResidentTensorState::load(
      storage, buffers, dense_specs, typed_specs, candidate.tensors_);
  if (!status.ok()) return status;
  destination = std::move(candidate);
  return Status::success();
}

Status DeepSeekResidentModelState::bind_attention(
    std::uint32_t layer, std::uint32_t compress_ratio,
    DeepSeekAttentionBinding& destination) const noexcept {
  return tensors_.bind_attention("layers." + std::to_string(layer), layer,
                                 compress_ratio, destination);
}

Status DeepSeekResidentModelState::bind_ffn(
    std::uint32_t layer, DeepSeekRouterKind router,
    DeepSeekFfnBinding& destination) const noexcept {
  return tensors_.bind_ffn(
      "layers." + std::to_string(layer), layer,
      router, destination);
}

Status DeepSeekResidentModelState::bind_io(
    DeepSeekIoBinding& destination) const noexcept {
  return tensors_.bind_io(destination);
}

void DeepSeekResidentModelState::clear() noexcept { tensors_.clear(); }

}  // namespace expert::runtime::cuda
