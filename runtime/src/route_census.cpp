#include "expert/runtime/route_census.hpp"

#include "expert/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace expert::runtime {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'Q'}, std::byte{'R'}, std::byte{'T'}, std::byte{'C'},
    std::byte{'E'}, std::byte{'N'}, std::byte{'S'}, std::byte{'1'}};
constexpr std::uint32_t kVersion = 2U;
constexpr std::uint32_t kLegacyVersion = 1U;
constexpr std::uint64_t kHeatUnitQ20 = 1ULL << 20U;
constexpr std::size_t kDigestBytes = 32U;
constexpr std::size_t kFixedHeaderBytes = 124U;

[[nodiscard]] bool nonzero_digest(const Sha256Digest& digest) noexcept {
  return std::any_of(digest.begin(), digest.end(),
                     [](std::byte value) { return value != std::byte{}; });
}

[[nodiscard]] bool valid_config(const RouteCensusConfig& config) noexcept {
  const auto cells = static_cast<std::uint64_t>(config.layer_count) *
                     config.experts_per_layer;
  const auto transitions = cells * config.experts_per_layer;
  return config.model_id != 0U && nonzero_digest(config.model_content_hash) &&
         config.quant_abi != 0U && config.layer_count != 0U &&
         config.experts_per_layer != 0U && config.route_width != 0U &&
         config.route_width <= config.experts_per_layer &&
         config.decay_interval_observations != 0U && cells <= 1'000'000U &&
         transitions <= 8'000'000U;
}

[[nodiscard]] std::uint64_t saturated_add(std::uint64_t left,
                                          std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left
             ? std::numeric_limits<std::uint64_t>::max()
             : left + right;
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
  for (std::uint32_t shift = 0; shift < 32U; shift += 8U)
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
  for (std::uint32_t shift = 0; shift < 64U; shift += 8U)
    output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
}

struct Reader final {
  std::span<const std::byte> bytes;
  std::size_t cursor{};

  [[nodiscard]] bool take_u32(std::uint32_t& value) noexcept {
    if (bytes.size() - cursor < sizeof(value)) return false;
    value = 0U;
    for (std::uint32_t shift = 0; shift < 32U; shift += 8U)
      value |= std::to_integer<std::uint32_t>(bytes[cursor++]) << shift;
    return true;
  }

  [[nodiscard]] bool take_u64(std::uint64_t& value) noexcept {
    if (bytes.size() - cursor < sizeof(value)) return false;
    value = 0U;
    for (std::uint32_t shift = 0; shift < 64U; shift += 8U)
      value |= std::to_integer<std::uint64_t>(bytes[cursor++]) << shift;
    return true;
  }

  [[nodiscard]] bool take(std::span<std::byte> output) noexcept {
    if (bytes.size() - cursor < output.size()) return false;
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
                output.size(), output.begin());
    cursor += output.size();
    return true;
  }
};

[[nodiscard]] std::filesystem::path slot_path(
    const std::filesystem::path& prefix, std::uint64_t slot) {
  return std::filesystem::path(prefix.string() + "." +
                               std::to_string(slot));
}

}  // namespace

RouteCensusLoadResult RouteCensus::decode_file(
    const std::filesystem::path& path,
    const RouteCensusConfig& expected) {
  std::error_code error;
  const auto file_bytes = std::filesystem::file_size(path, error);
  const auto expected_cells = static_cast<std::uint64_t>(expected.layer_count) *
                              expected.experts_per_layer;
  const auto expected_transitions =
      expected_cells * expected.experts_per_layer;
  const auto upper_bound = 264ULL + expected_cells * 48ULL +
                           static_cast<std::uint64_t>(expected.layer_count) *
                               (16ULL + 4ULL * expected.route_width) +
                           expected_transitions * sizeof(std::uint32_t);
  if (error || file_bytes < 128U || file_bytes > upper_bound) {
    return {{ErrorCode::io_failed, "invalid route census file size"}, {}};
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(file_bytes));
  std::ifstream input(path, std::ios::binary);
  if (!input ||
      !input.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()))) {
    return {{ErrorCode::io_failed, "route census read failed"}, {}};
  }
  if (bytes.size() <= kDigestBytes) {
    return {{ErrorCode::checksum_mismatch,
             "route census authentication is absent"}, {}};
  }
  const auto authenticated =
      std::span<const std::byte>(bytes).first(bytes.size() - kDigestBytes);
  Sha256Digest stored_digest{};
  std::copy_n(bytes.end() - static_cast<std::ptrdiff_t>(kDigestBytes),
              kDigestBytes, stored_digest.begin());
  if (!constant_time_equal(sha256(authenticated), stored_digest)) {
    return {{ErrorCode::checksum_mismatch,
             "route census checksum mismatch"}, {}};
  }

  Reader reader{authenticated};
  std::array<std::byte, kMagic.size()> magic{};
  std::uint32_t version{};
  std::uint64_t generation{};
  std::uint64_t model_id{};
  std::uint32_t quant_abi{};
  std::uint32_t layer_count{};
  std::uint32_t experts_per_layer{};
  std::uint32_t route_width{};
  std::uint64_t decay_interval{};
  Sha256Digest model_hash{};
  std::uint64_t observation{};
  std::uint64_t completed_routes{};
  std::uint64_t total_selections{};
  std::uint64_t consecutive_reuse{};
  std::uint64_t cell_count{};
  if (!reader.take(magic) || !reader.take_u32(version) ||
      !reader.take_u64(generation) || !reader.take_u64(model_id) ||
      !reader.take_u32(quant_abi) || !reader.take_u32(layer_count) ||
      !reader.take_u32(experts_per_layer) || !reader.take_u32(route_width) ||
      !reader.take_u64(decay_interval) || !reader.take(model_hash) ||
      !reader.take_u64(observation) || !reader.take_u64(completed_routes) ||
      !reader.take_u64(total_selections) ||
      !reader.take_u64(consecutive_reuse) || !reader.take_u64(cell_count)) {
    return {{ErrorCode::io_failed, "route census header is truncated"}, {}};
  }
  if (magic != kMagic ||
      (version != kVersion && version != kLegacyVersion) ||
      generation == 0U ||
      model_id != expected.model_id ||
      quant_abi != expected.quant_abi || layer_count != expected.layer_count ||
      experts_per_layer != expected.experts_per_layer ||
      route_width != expected.route_width ||
      decay_interval != expected.decay_interval_observations ||
      model_hash != expected.model_content_hash || cell_count != expected_cells) {
    return {{ErrorCode::invalid_argument,
             "route census model or schema identity mismatch"}, {}};
  }

  auto census = std::make_unique<RouteCensus>(expected);
  census->generation_ = generation;
  census->observation_ = observation;
  census->completed_routes_ = completed_routes;
  census->total_selections_ = total_selections;
  census->consecutive_reuse_selections_ = consecutive_reuse;
  for (auto& cell : census->cells_) {
    if (!reader.take_u64(cell.total) || !reader.take_u64(cell.heat_q20) ||
        !reader.take_u64(cell.heat_observation) ||
        !reader.take_u64(cell.last_seen) || !reader.take_u64(cell.cpu) ||
        !reader.take_u64(cell.gpu) || cell.heat_observation > observation ||
        cell.last_seen > observation || cell.cpu + cell.gpu < cell.cpu ||
        cell.cpu + cell.gpu != cell.total) {
      return {{ErrorCode::io_failed,
               "route census cell is invalid or truncated"}, {}};
    }
  }
  std::uint64_t observed_sum{};
  std::uint64_t selection_sum{};
  std::uint64_t reuse_sum{};
  for (auto& layer : census->layers_) {
    if (!reader.take_u64(layer.observations) ||
        !reader.take_u64(layer.consecutive_reuse) ||
        layer.observations >
            std::numeric_limits<std::uint64_t>::max() /
                expected.route_width ||
        layer.consecutive_reuse > layer.observations * expected.route_width) {
      return {{ErrorCode::io_failed, "route census layer state is invalid"},
              {}};
    }
    for (auto& expert : layer.previous_route) {
      if (!reader.take_u32(expert) || expert >= expected.experts_per_layer) {
        return {{ErrorCode::io_failed,
                 "route census previous route is invalid"}, {}};
      }
    }
    observed_sum = saturated_add(observed_sum, layer.observations);
    reuse_sum = saturated_add(reuse_sum, layer.consecutive_reuse);
  }
  if (version == kVersion) {
    std::uint64_t transition_count{};
    if (!reader.take_u64(transition_count) ||
        transition_count != census->transitions_.size())
      return {{ErrorCode::io_failed,
               "route census transition header is invalid"}, {}};
    for (auto& count : census->transitions_) {
      if (!reader.take_u32(count))
        return {{ErrorCode::io_failed,
                 "route census transition table is truncated"}, {}};
    }
  }
  for (const auto& cell : census->cells_)
    selection_sum = saturated_add(selection_sum, cell.total);
  if (reader.cursor != authenticated.size() || observation != completed_routes ||
      observed_sum != completed_routes ||
      selection_sum != total_selections || reuse_sum != consecutive_reuse) {
    return {{ErrorCode::io_failed,
             "route census aggregate accounting mismatch"}, {}};
  }
  return {Status::success(), std::move(census)};
}

RouteCensus::RouteCensus(RouteCensusConfig config) : config_(config) {
  if (!valid_config(config_))
    throw std::invalid_argument("invalid route census configuration");
  cells_.resize(static_cast<std::size_t>(config_.layer_count) *
                config_.experts_per_layer);
  layers_.resize(config_.layer_count);
  for (auto& layer : layers_)
    layer.previous_route.resize(config_.route_width);
  transitions_.resize(static_cast<std::size_t>(config_.layer_count) *
                      config_.experts_per_layer *
                      config_.experts_per_layer);
}

std::uint64_t RouteCensus::effective_heat(const Cell& cell) const noexcept {
  if (cell.heat_q20 == 0U || observation_ <= cell.heat_observation)
    return cell.heat_q20;
  const auto shifts = (observation_ - cell.heat_observation) /
                      config_.decay_interval_observations;
  return shifts >= 64U ? 0U : cell.heat_q20 >> shifts;
}

Status RouteCensus::observe(
    std::uint32_t layer, std::span<const std::uint32_t> routed_experts,
    std::span<const std::uint32_t> cpu_experts) noexcept {
  std::lock_guard lock(mutex_);
  if (layer >= config_.layer_count ||
      routed_experts.size() != config_.route_width ||
      cpu_experts.size() > routed_experts.size()) {
    return {ErrorCode::invalid_argument, "invalid route census observation"};
  }
  for (std::size_t index = 0U; index < routed_experts.size(); ++index) {
    const auto expert = routed_experts[index];
    if (expert >= config_.experts_per_layer ||
        std::find(routed_experts.begin(),
                  routed_experts.begin() +
                      static_cast<std::ptrdiff_t>(index),
                  expert) != routed_experts.begin() +
                                 static_cast<std::ptrdiff_t>(index))
      return {ErrorCode::invalid_argument,
              "route census expert is invalid or duplicated"};
  }
  for (std::size_t index = 0U; index < cpu_experts.size(); ++index) {
    const auto expert = cpu_experts[index];
    if (std::find(routed_experts.begin(), routed_experts.end(), expert) ==
            routed_experts.end() ||
        std::find(cpu_experts.begin(),
                  cpu_experts.begin() + static_cast<std::ptrdiff_t>(index),
                  expert) != cpu_experts.begin() +
                                 static_cast<std::ptrdiff_t>(index))
      return {ErrorCode::invalid_argument,
              "route census CPU selection is not a unique routed expert"};
  }

  observation_ = saturated_add(observation_, 1U);
  completed_routes_ = saturated_add(completed_routes_, 1U);
  total_selections_ = saturated_add(total_selections_, routed_experts.size());
  auto& layer_state = layers_[layer];
  if (layer_state.observations != 0U) {
    const auto layer_base = static_cast<std::size_t>(layer) *
        config_.experts_per_layer * config_.experts_per_layer;
    for (const auto previous : layer_state.previous_route) {
      for (const auto expert : routed_experts) {
        auto& transition = transitions_[
            layer_base + static_cast<std::size_t>(previous) *
                             config_.experts_per_layer + expert];
        if (transition != std::numeric_limits<std::uint32_t>::max())
          ++transition;
      }
    }
    for (const auto expert : routed_experts) {
      if (std::find(layer_state.previous_route.begin(),
                    layer_state.previous_route.end(), expert) !=
          layer_state.previous_route.end()) {
        layer_state.consecutive_reuse =
            saturated_add(layer_state.consecutive_reuse, 1U);
        consecutive_reuse_selections_ =
            saturated_add(consecutive_reuse_selections_, 1U);
      }
    }
  }
  layer_state.observations = saturated_add(layer_state.observations, 1U);
  std::copy(routed_experts.begin(), routed_experts.end(),
            layer_state.previous_route.begin());
  for (const auto expert : routed_experts) {
    auto& cell = cells_[static_cast<std::size_t>(layer) *
                        config_.experts_per_layer + expert];
    cell.heat_q20 = effective_heat(cell);
    cell.heat_observation = observation_;
    cell.heat_q20 = saturated_add(cell.heat_q20, kHeatUnitQ20);
    cell.total = saturated_add(cell.total, 1U);
    cell.last_seen = observation_;
    if (std::find(cpu_experts.begin(), cpu_experts.end(), expert) !=
        cpu_experts.end())
      cell.cpu = saturated_add(cell.cpu, 1U);
    else
      cell.gpu = saturated_add(cell.gpu, 1U);
  }
  return Status::success();
}

std::vector<RouteCensusWarmEntry> RouteCensus::stable_warm_set(
    std::size_t maximum_entries, std::size_t maximum_per_layer) const {
  std::lock_guard lock(mutex_);
  std::vector<RouteCensusWarmEntry> candidates;
  candidates.reserve(cells_.size());
  for (std::uint32_t layer = 0U; layer < config_.layer_count; ++layer) {
    for (std::uint32_t expert = 0U; expert < config_.experts_per_layer;
         ++expert) {
      const auto& cell = cells_[static_cast<std::size_t>(layer) *
                                config_.experts_per_layer + expert];
      if (cell.total == 0U) continue;
      candidates.push_back(
          {ExpertKey{config_.model_id, layer, expert, config_.quant_abi},
           cell.total, effective_heat(cell), cell.last_seen, cell.cpu,
           cell.gpu});
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& left, const auto& right) {
              if (left.effective_heat_q20 != right.effective_heat_q20)
                return left.effective_heat_q20 > right.effective_heat_q20;
              if (left.total_selections != right.total_selections)
                return left.total_selections > right.total_selections;
              if (left.last_seen_observation != right.last_seen_observation)
                return left.last_seen_observation >
                       right.last_seen_observation;
              return left.key < right.key;
            });
  std::vector<RouteCensusWarmEntry> result;
  result.reserve(std::min(maximum_entries, candidates.size()));
  std::vector<std::size_t> per_layer(config_.layer_count);
  for (auto& candidate : candidates) {
    if (result.size() == maximum_entries) break;
    if (maximum_per_layer != 0U &&
        per_layer[candidate.key.layer] >= maximum_per_layer)
      continue;
    ++per_layer[candidate.key.layer];
    result.push_back(std::move(candidate));
  }
  return result;
}

std::vector<RouteCensusPrediction> RouteCensus::predict_next(
    std::uint32_t layer, std::span<const std::uint32_t> current_route,
    std::size_t maximum_entries) const {
  std::lock_guard lock(mutex_);
  if (layer >= config_.layer_count ||
      current_route.size() != config_.route_width || maximum_entries == 0U)
    return {};
  for (const auto expert : current_route)
    if (expert >= config_.experts_per_layer) return {};
  const auto layer_base = static_cast<std::size_t>(layer) *
      config_.experts_per_layer * config_.experts_per_layer;
  std::vector<RouteCensusPrediction> candidates;
  candidates.reserve(config_.experts_per_layer);
  for (std::uint32_t candidate = 0U;
       candidate < config_.experts_per_layer; ++candidate) {
    if (std::find(current_route.begin(), current_route.end(), candidate) !=
        current_route.end())
      continue;
    std::uint64_t score = 0U;
    for (const auto previous : current_route)
      score = saturated_add(
          score, transitions_[layer_base +
              static_cast<std::size_t>(previous) *
                  config_.experts_per_layer + candidate]);
    if (score != 0U)
      candidates.push_back(
          {ExpertKey{config_.model_id, layer, candidate, config_.quant_abi},
           score});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& left, const auto& right) {
              return left.transition_score != right.transition_score
                  ? left.transition_score > right.transition_score
                  : left.key < right.key;
            });
  if (candidates.size() > maximum_entries)
    candidates.resize(maximum_entries);
  return candidates;
}

RouteCensusSnapshot RouteCensus::snapshot() const noexcept {
  std::lock_guard lock(mutex_);
  const auto observed = static_cast<std::size_t>(std::count_if(
      cells_.begin(), cells_.end(),
      [](const Cell& cell) { return cell.total != 0U; }));
  const auto serialized = kFixedHeaderBytes + cells_.size() * 48U +
      layers_.size() * (16U + config_.route_width * sizeof(std::uint32_t)) +
      sizeof(std::uint64_t) + transitions_.size() * sizeof(std::uint32_t) +
      kDigestBytes;
  return {generation_, completed_routes_, total_selections_,
          consecutive_reuse_selections_, observed, serialized};
}

std::vector<std::byte> RouteCensus::serialize(
    std::uint64_t generation) const {
  std::vector<std::byte> output;
  output.reserve(kFixedHeaderBytes + cells_.size() * 48U +
                 layers_.size() *
                     (16U + config_.route_width * sizeof(std::uint32_t)) +
                 sizeof(std::uint64_t) +
                 transitions_.size() * sizeof(std::uint32_t) +
                 kDigestBytes);
  output.insert(output.end(), kMagic.begin(), kMagic.end());
  append_u32(output, kVersion);
  append_u64(output, generation);
  append_u64(output, config_.model_id);
  append_u32(output, config_.quant_abi);
  append_u32(output, config_.layer_count);
  append_u32(output, config_.experts_per_layer);
  append_u32(output, config_.route_width);
  append_u64(output, config_.decay_interval_observations);
  output.insert(output.end(), config_.model_content_hash.begin(),
                config_.model_content_hash.end());
  append_u64(output, observation_);
  append_u64(output, completed_routes_);
  append_u64(output, total_selections_);
  append_u64(output, consecutive_reuse_selections_);
  append_u64(output, cells_.size());
  for (const auto& cell : cells_) {
    append_u64(output, cell.total);
    append_u64(output, cell.heat_q20);
    append_u64(output, cell.heat_observation);
    append_u64(output, cell.last_seen);
    append_u64(output, cell.cpu);
    append_u64(output, cell.gpu);
  }
  for (const auto& layer : layers_) {
    append_u64(output, layer.observations);
    append_u64(output, layer.consecutive_reuse);
    for (const auto expert : layer.previous_route) append_u32(output, expert);
  }
  append_u64(output, transitions_.size());
  for (const auto count : transitions_) append_u32(output, count);
  const auto digest = sha256(output);
  output.insert(output.end(), digest.begin(), digest.end());
  return output;
}

Status RouteCensus::save(const std::filesystem::path& prefix) noexcept {
  try {
    std::lock_guard lock(mutex_);
    if (prefix.empty())
      return {ErrorCode::invalid_argument, "route census prefix is empty"};
    const auto next_generation = saturated_add(generation_, 1U);
    const auto bytes = serialize(next_generation);
    const auto destination = slot_path(prefix, next_generation & 1U);
    const auto temporary = std::filesystem::path(destination.string() + ".tmp");
    std::error_code error;
    if (!destination.parent_path().empty())
      std::filesystem::create_directories(destination.parent_path(), error);
    if (error)
      return {ErrorCode::open_failed,
              "route census directory creation failed"};
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      return {ErrorCode::open_failed, "route census temporary open failed"};
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) {
      output.close();
      std::filesystem::remove(temporary, error);
      return {ErrorCode::io_failed, "route census write failed"};
    }
    output.close();
    std::filesystem::remove(destination, error);
    error.clear();
    std::filesystem::rename(temporary, destination, error);
    if (error) {
      std::filesystem::remove(temporary, error);
      return {ErrorCode::io_failed, "route census publication failed"};
    }
    generation_ = next_generation;
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::io_failed, error.what()};
  }
}

RouteCensusLoadResult RouteCensus::load(
    const std::filesystem::path& prefix,
    const RouteCensusConfig& expected) noexcept {
  try {
    if (prefix.empty() || !valid_config(expected))
      return {{ErrorCode::invalid_argument,
               "invalid route census load contract"}, {}};
    std::unique_ptr<RouteCensus> newest;
    Status last_error{ErrorCode::open_failed,
                      "no route census generation exists"};
    for (std::uint64_t slot = 0U; slot < 2U; ++slot) {
      const auto path = slot_path(prefix, slot);
      std::error_code error;
      if (!std::filesystem::exists(path, error) || error) continue;
      auto decoded = decode_file(path, expected);
      if (!decoded.status.ok()) {
        last_error = std::move(decoded.status);
        continue;
      }
      if (!newest || decoded.census->generation_ > newest->generation_)
        newest = std::move(decoded.census);
    }
    if (!newest) return {std::move(last_error), {}};
    return {Status::success(), std::move(newest)};
  } catch (const std::exception& error) {
    return {{ErrorCode::io_failed, error.what()}, {}};
  }
}

}  // namespace expert::runtime
