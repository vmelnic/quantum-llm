#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace expert::runtime {

struct TokenProbability final {
  std::uint32_t token{};
  double probability{};
};

struct SamplingDistribution final {
  std::vector<TokenProbability> entries;

  [[nodiscard]] double probability(std::uint32_t token) const noexcept;
};

// Converts already-descending top-k logits into the exact host sampling
// distribution used by the runtime. Candidate ids remain authoritative for
// ties; temperature, min-p and nucleus truncation match the scalar sampler.
[[nodiscard]] SamplingDistribution make_sampling_distribution(
    std::span<const float> sorted_logits,
    std::span<const std::uint32_t> sorted_tokens,
    std::uint32_t temperature_ppm, std::uint32_t top_p_ppm,
    std::uint32_t min_p_ppm);

[[nodiscard]] std::uint32_t sample_distribution(
    const SamplingDistribution& distribution, double uniform);

struct RejectionSamplingResult final {
  bool accepted{};
  std::uint32_t token{};
  double target_probability{};
  double draft_probability{};
};

// Implements one exact speculative sampling decision. On rejection, the
// returned token is sampled from normalized max(p-q, 0). The caller owns the
// independent acceptance and correction random draws.
[[nodiscard]] RejectionSamplingResult rejection_sample(
    std::uint32_t proposal, const SamplingDistribution& target,
    const SamplingDistribution& draft, double acceptance_uniform,
    double correction_uniform);

// Stable counter-based uniform used to allocate disjoint scalar, proposer,
// acceptance and correction streams without mutable RNG ordering.
[[nodiscard]] double counter_uniform(std::uint64_t seed,
                                     std::uint32_t position,
                                     std::uint32_t stream,
                                     std::uint32_t index = 0U) noexcept;

}  // namespace expert::runtime
