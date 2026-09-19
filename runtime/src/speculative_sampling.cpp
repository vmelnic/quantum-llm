#include "expert/runtime/speculative_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace expert::runtime {
namespace {

std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

void require_uniform(double value) {
  if (!std::isfinite(value) || value < 0.0 || !(value < 1.0))
    throw std::invalid_argument("sampling uniform must be in [0, 1)");
}

void validate_distribution(const SamplingDistribution& distribution) {
  if (distribution.entries.empty())
    throw std::invalid_argument("sampling distribution is empty");
  double total{};
  std::vector<std::uint32_t> tokens;
  tokens.reserve(distribution.entries.size());
  for (const auto& entry : distribution.entries) {
    if (!std::isfinite(entry.probability) || entry.probability < 0.0)
      throw std::invalid_argument("sampling probability is invalid");
    total += entry.probability;
    tokens.push_back(entry.token);
  }
  std::sort(tokens.begin(), tokens.end());
  if (std::adjacent_find(tokens.begin(), tokens.end()) != tokens.end() ||
      !std::isfinite(total) || std::abs(total - 1.0) > 1e-10)
    throw std::invalid_argument("sampling distribution is not normalized");
}

}  // namespace

double SamplingDistribution::probability(std::uint32_t token) const noexcept {
  const auto found = std::find_if(
      entries.begin(), entries.end(),
      [token](const auto& entry) { return entry.token == token; });
  return found == entries.end() ? 0.0 : found->probability;
}

SamplingDistribution make_sampling_distribution(
    std::span<const float> sorted_logits,
    std::span<const std::uint32_t> sorted_tokens,
    std::uint32_t temperature_ppm, std::uint32_t top_p_ppm,
    std::uint32_t min_p_ppm) {
  if (sorted_logits.empty() || sorted_logits.size() != sorted_tokens.size() ||
      temperature_ppm == 0U || temperature_ppm > 2'000'000U ||
      top_p_ppm == 0U || top_p_ppm > 1'000'000U ||
      min_p_ppm > 1'000'000U || !std::isfinite(sorted_logits.front()))
    throw std::invalid_argument("invalid sorted sampling candidates");
  for (std::size_t index = 1U; index < sorted_logits.size(); ++index) {
    const auto previous = sorted_logits[index - 1U];
    const auto current = sorted_logits[index];
    if ((!std::isnan(current) && current > previous) ||
        (current == previous && sorted_tokens[index] < sorted_tokens[index - 1U]))
      throw std::invalid_argument("sampling candidates are not ordered");
  }

  const auto inverse_temperature =
      1'000'000.0 / static_cast<double>(temperature_ppm);
  const auto maximum = static_cast<double>(sorted_logits.front());
  const auto minimum_relative =
      static_cast<double>(min_p_ppm) / 1'000'000.0;
  std::vector<double> weights;
  std::vector<std::uint32_t> tokens;
  weights.reserve(sorted_logits.size());
  tokens.reserve(sorted_tokens.size());
  double total{};
  for (std::size_t index = 0U; index < sorted_logits.size(); ++index) {
    const auto logit = static_cast<double>(sorted_logits[index]);
    const auto weight = std::isfinite(logit)
                            ? std::exp((logit - maximum) * inverse_temperature)
                            : 0.0;
    if (weight < minimum_relative) continue;
    tokens.push_back(sorted_tokens[index]);
    weights.push_back(weight);
    total += weight;
  }
  if (weights.empty() || !std::isfinite(total) || !(total > 0.0))
    throw std::invalid_argument("sampling distribution has no retained mass");

  const auto top_p = static_cast<double>(top_p_ppm) / 1'000'000.0;
  double cumulative{};
  std::size_t nucleus = weights.size();
  for (std::size_t index = 0U; index < weights.size(); ++index) {
    cumulative += weights[index] / total;
    if (cumulative >= top_p) {
      nucleus = index + 1U;
      break;
    }
  }
  weights.resize(nucleus);
  tokens.resize(nucleus);
  total = std::accumulate(weights.begin(), weights.end(), 0.0);
  SamplingDistribution result;
  result.entries.reserve(nucleus);
  for (std::size_t index = 0U; index < nucleus; ++index)
    result.entries.push_back({tokens[index], weights[index] / total});
  return result;
}

std::uint32_t sample_distribution(const SamplingDistribution& distribution,
                                  double uniform) {
  validate_distribution(distribution);
  require_uniform(uniform);
  double cumulative{};
  for (const auto& entry : distribution.entries) {
    cumulative += entry.probability;
    if (uniform < cumulative) return entry.token;
  }
  return distribution.entries.back().token;
}

RejectionSamplingResult rejection_sample(
    std::uint32_t proposal, const SamplingDistribution& target,
    const SamplingDistribution& draft, double acceptance_uniform,
    double correction_uniform) {
  validate_distribution(target);
  validate_distribution(draft);
  require_uniform(acceptance_uniform);
  require_uniform(correction_uniform);
  const auto target_probability = target.probability(proposal);
  const auto draft_probability = draft.probability(proposal);
  if (!(draft_probability > 0.0))
    throw std::invalid_argument("proposal has zero draft probability");
  const auto acceptance =
      std::min(1.0, target_probability / draft_probability);
  if (acceptance_uniform < acceptance)
    return {true, proposal, target_probability, draft_probability};

  std::map<std::uint32_t, double> residual;
  for (const auto& entry : target.entries)
    residual[entry.token] = entry.probability;
  for (const auto& entry : draft.entries)
    residual[entry.token] =
        std::max(0.0, residual[entry.token] - entry.probability);
  double total{};
  for (const auto& [token, probability] : residual) {
    static_cast<void>(token);
    total += probability;
  }
  if (!std::isfinite(total) || !(total > 0.0))
    throw std::runtime_error("rejected proposal has no correction mass");
  const auto threshold = correction_uniform * total;
  double cumulative{};
  std::uint32_t last_positive{};
  bool found_positive{};
  for (const auto& [token, probability] : residual) {
    if (probability > 0.0) {
      last_positive = token;
      found_positive = true;
    }
    cumulative += probability;
    if (threshold < cumulative)
      return {false, token, target_probability, draft_probability};
  }
  if (!found_positive)
    throw std::runtime_error("rejected proposal has no positive correction");
  return {false, last_positive, target_probability, draft_probability};
}

double counter_uniform(std::uint64_t seed, std::uint32_t position,
                       std::uint32_t stream, std::uint32_t index) noexcept {
  auto counter = seed;
  counter ^= static_cast<std::uint64_t>(position) * 0xd2b74407b1ce6e93ULL;
  counter ^= static_cast<std::uint64_t>(stream) * 0xca5a826395121157ULL;
  counter ^= static_cast<std::uint64_t>(index) * 0x9e3779b185ebca87ULL;
  const auto bits = splitmix64(counter);
  return static_cast<double>(bits >> 11U) *
         (1.0 / 9007199254740992.0);
}

}  // namespace expert::runtime
