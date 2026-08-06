#pragma once

#include "expert/runtime/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <span>

namespace expert::runtime::cuda {

inline constexpr std::uint32_t kDeepSeekHcaStreams = 4;
inline constexpr std::uint32_t kDeepSeekHcaMixes = 24;

struct DeepSeekHcaView final {
  const float* function{};
  const float* base{};
  const float* scale{};
  std::uint32_t hidden{};
};

class DeepSeekHcaParameters final {
 public:
  DeepSeekHcaParameters(float* function, float* base, float* scale,
                        std::uint32_t hidden) noexcept;
  ~DeepSeekHcaParameters();
  DeepSeekHcaParameters(const DeepSeekHcaParameters&) = delete;
  DeepSeekHcaParameters& operator=(const DeepSeekHcaParameters&) = delete;

  [[nodiscard]] const float* function() const noexcept { return function_; }
  [[nodiscard]] const float* base() const noexcept { return base_; }
  [[nodiscard]] const float* scale() const noexcept { return scale_; }
  [[nodiscard]] std::uint32_t hidden() const noexcept { return hidden_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept;
  [[nodiscard]] DeepSeekHcaView view() const noexcept {
    return {function_, base_, scale_, hidden_};
  }

 private:
  float* function_{};
  float* base_{};
  float* scale_{};
  std::uint32_t hidden_{};
};

struct DeepSeekHcaAdmissionResult final {
  Status status;
  std::shared_ptr<DeepSeekHcaParameters> parameters;
};

// Source order is the original F32 fn/base/scale tensors. They remain F32 on
// device because this control projection is small and numerically sensitive.
[[nodiscard]] DeepSeekHcaAdmissionResult admit_deepseek_hca(
    std::span<const std::byte> function, std::span<const std::byte> base,
    std::span<const std::byte> scale, std::uint32_t hidden) noexcept;

struct DeepSeekHcaWorkspace final {
  float* normalized{};  // [4 * hidden]
  float* mixes{};       // [24]
};

// Implements the official DeepSeek-V4 mHC mapping and stream collapse. pre,
// post and comb contain 4, 4 and 16 F32 values respectively.
[[nodiscard]] Status deepseek_hca_pre(
    const DeepSeekHcaParameters& parameters, const float* streams,
    float* collapsed, float* pre, float* post, float* comb,
    const DeepSeekHcaWorkspace& workspace, float epsilon,
    std::uint32_t sinkhorn_iterations, void* stream) noexcept;

// Two independent rows share one read of the HCA control matrix. collapsed,
// pre/post/comb and the workspace are contiguous row-major pairs.
[[nodiscard]] Status deepseek_hca_pre_pair(
    const DeepSeekHcaView& parameters,
    const std::array<const float*, 2U>& streams, float* collapsed,
    float* pre, float* post, float* comb,
    const DeepSeekHcaWorkspace& workspace, float epsilon,
    std::uint32_t sinkhorn_iterations, void* stream) noexcept;

[[nodiscard]] Status deepseek_hca_pre(
    const DeepSeekHcaView& parameters, const float* streams,
    float* collapsed, float* pre, float* post, float* comb,
    const DeepSeekHcaWorkspace& workspace, float epsilon,
    std::uint32_t sinkhorn_iterations, void* stream) noexcept;

// updated[o,d] = post[o] * sublayer[d] + sum_i comb[i,o] * streams[i,d].
[[nodiscard]] Status deepseek_hca_post(
    const float* sublayer, const float* streams, const float* post,
    const float* comb, float* updated, std::uint32_t hidden,
    void* stream) noexcept;

}  // namespace expert::runtime::cuda
