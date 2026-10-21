#include "expert/runtime/cuda/transformer_kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checked(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <class T>
struct Device final {
  T* pointer{};
  std::size_t count;
  explicit Device(std::size_t size) : count(size) {
    checked(cudaMalloc(reinterpret_cast<void**>(&pointer), size * sizeof(T)));
  }
  ~Device() { static_cast<void>(cudaFree(pointer)); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void upload(const std::vector<T>& values) {
    if (values.size() != count) throw std::runtime_error("upload size mismatch");
    checked(cudaMemcpy(pointer, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> result(count);
    checked(cudaMemcpy(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }
};

// Independent arithmetic BF16 decoder: no CUDA cache/decoder helper is used.
double bf16_value(std::uint16_t bits) {
  const auto exponent = (bits >> 7U) & 255U;
  const auto mantissa = bits & 127U;
  const auto magnitude = exponent == 0U
      ? std::ldexp(static_cast<double>(mantissa), -133)
      : std::ldexp(1.0 + static_cast<double>(mantissa) / 128.0,
                   static_cast<int>(exponent) - 127);
  return (bits & 32768U) ? -magnitude : magnitude;
}

std::uint16_t bf16_bits(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 32767U + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

struct Sample final {
  std::uint32_t state{0x14234567U};
  std::uint16_t next(unsigned exponent) {
    state = state * 1664525U + 1013904223U;
    return static_cast<std::uint16_t>(((state >> 16U) & 32768U) |
                                     (exponent << 7U) | ((state >> 8U) & 127U));
  }
};

// The tests use factor=1 and beta=0: pure RoPE still exercises interleaved
// source pairs, split output halves, cache commit and nonzero positions.
std::vector<double> rotate(const std::vector<double>& input, unsigned offset,
                           unsigned rope, unsigned position) {
  auto output = input;
  for (unsigned pair = 0; pair < rope / 2; ++pair) {
    const double angle = position / std::pow(10000.0, 2.0 * pair / rope);
    const auto even = input[offset + pair * 2];
    const auto odd = input[offset + pair * 2 + 1];
    output[offset + pair] = bf16_value(bf16_bits(static_cast<float>(
        even * std::cos(angle) - odd * std::sin(angle))));
    output[offset + rope / 2 + pair] = bf16_value(bf16_bits(static_cast<float>(
        odd * std::cos(angle) + even * std::sin(angle))));
  }
  return output;
}

void check_case(unsigned prefix, unsigned rows, unsigned heads,
                unsigned rank, unsigned nope, unsigned rope, unsigned value,
                float scale, bool split_workspace = true) {
  constexpr unsigned page_tokens = 64;
  constexpr unsigned layers = 2;
  constexpr unsigned layer = 1;
  const unsigned width = rank + rope;
  const unsigned tokens = prefix + rows;
  const unsigned page_count = (tokens + page_tokens - 1) / page_tokens;
  Sample generator;
  // Logical source arrays are the oracle inputs. It never decodes CUDA pages
  // or uses absorbed queries; it materializes K and V before attention.
  std::vector<std::vector<double>> cache(tokens, std::vector<double>(width));
  for (auto& token : cache)
    for (auto& element : token) element = bf16_value(generator.next(125));
  std::vector<float> latent(rows * width);
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned dim = 0; dim < width; ++dim)
      latent[row * width + dim] = static_cast<float>(cache[prefix + row][dim]);
    cache[prefix + row] = rotate(cache[prefix + row], rank, rope, prefix + row);
  }
  std::vector<float> queries(rows * heads * (nope + rope));
  std::vector<std::vector<double>> oracle_queries;
  for (unsigned index = 0; index < rows * heads; ++index) {
    std::vector<double> query(nope + rope);
    for (unsigned dim = 0; dim < nope + rope; ++dim) {
      query[dim] = bf16_value(generator.next(126));
      queries[index * (nope + rope) + dim] = static_cast<float>(query[dim]);
    }
    oracle_queries.push_back(rotate(query, nope, rope, prefix + index / heads));
  }
  const unsigned projected = nope + value;
  std::vector<std::uint16_t> weights(heads * projected * rank);
  std::vector<std::vector<double>> oracle_weights(heads * projected,
                                                 std::vector<double>(rank));
  for (unsigned index = 0; index < heads * projected; ++index)
    for (unsigned dim = 0; dim < rank; ++dim) {
      const auto bits = generator.next(120);
      weights[index * rank + dim] = bits;
      oracle_weights[index][dim] = bf16_value(bits);
    }
  std::vector<std::uint16_t> pages(page_count * layers * page_tokens * width,
                                  bf16_bits(0.25F));
  const auto page_offset = [&](unsigned token, unsigned dim) {
    // Encode the published two-plane page layout separately from the
    // logical dense [token, dimension] matrix used by the numerical oracle.
    return static_cast<std::size_t>(token / page_tokens) * layers * page_tokens * width +
        layer * page_tokens * width + (dim / (width / 2)) * page_tokens * (width / 2) +
        (token % page_tokens) * (width / 2) + dim % (width / 2);
  };
  for (unsigned token = 0; token < prefix; ++token)
    for (unsigned dim = 0; dim < width; ++dim)
      pages[page_offset(token, dim)] = bf16_bits(static_cast<float>(cache[token][dim]));
  Device<std::uint16_t> device_pages(pages.size());
  device_pages.upload(pages);
  std::vector<const void*> table;
  for (unsigned page = 0; page < page_count; ++page)
    table.push_back(device_pages.pointer + page * layers * page_tokens * width);
  Device<const void*> device_table(table.size());
  device_table.upload(table);
  Device<float> device_query(queries.size()); device_query.upload(queries);
  Device<float> device_latent(latent.size()); device_latent.upload(latent);
  Device<std::uint16_t> device_weights(weights.size()); device_weights.upload(weights);
  Device<float> device_absorbed(rows * heads * rank);
  Device<float> device_output(rows * heads * value);
  const auto splits = (tokens + expert::runtime::cuda::kMlaContextSplitTokens - 1U) /
                      expert::runtime::cuda::kMlaContextSplitTokens;
  const auto partials = rows * heads * splits;
  Device<float> maxima(partials + 16), sums(partials + 16);
  Device<float> partial_values(partials * rank + 16);
  maxima.upload(std::vector<float>(maxima.count, -123.0F));
  sums.upload(std::vector<float>(sums.count, -123.0F));
  partial_values.upload(std::vector<float>(partial_values.count, -123.0F));
  expert::runtime::cuda::MlaCausalLaunch launch{};
  launch.query = device_query.pointer;
  launch.latent = device_latent.pointer;
  launch.kv_up = device_weights.pointer;
  launch.page_table = device_table.pointer;
  launch.output = device_output.pointer;
  launch.latent_query = device_absorbed.pointer;
  if (split_workspace) {
    launch.partial_maxima = maxima.pointer;
    launch.partial_sums = sums.pointer;
    launch.partial_outputs = partial_values.pointer;
  }
  launch.layer = layer;
  launch.page_tokens = page_tokens;
  launch.first_position = prefix;
  launch.rows = rows;
  launch.heads = heads;
  launch.kv_rank = rank;
  launch.nope_dim = nope;
  launch.rope_dim = rope;
  launch.value_dim = value;
  launch.attention_scale = scale;
  launch.rope_theta = 10000.0F;
  launch.rope_factor = 1.0F;
  launch.rope_beta_fast = 32.0F;
  launch.rope_beta_slow = 1.0F;
  launch.rope_original_context = 4096;
  const auto status = expert::runtime::cuda::mla_causal_latent_bf16(launch);
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
  checked(cudaDeviceSynchronize());
  const auto actual = device_output.download();
  for (const auto* scratch : {&maxima, &sums, &partial_values}) {
    const auto values = scratch->download();
    for (auto index = values.size() - 16; index < values.size(); ++index)
      if (values[index] != -123.0F)
        throw std::runtime_error("MLA split scratch overrun");
    if (!split_workspace || rows > expert::runtime::cuda::kMlaContextSplitMaximumRows || tokens < 4096)
      if (std::any_of(values.begin(), values.end(), [](float v) { return v != -123.0F; }))
        throw std::runtime_error("MLA fallback unexpectedly wrote split scratch");
  }
  const auto committed = device_pages.download();
  for (unsigned token = 0; token < tokens; ++token)
    for (unsigned dim = 0; dim < width; ++dim) {
      const auto error = std::abs(bf16_value(committed[page_offset(token, dim)]) - cache[token][dim]);
      if (error > 0.004 * std::max(1.0, std::abs(cache[token][dim])))
        throw std::runtime_error("MLA cache commit numerical failure");
    }
  for (unsigned page = 0; page < page_count; ++page)
    for (unsigned index = 0; index < page_tokens * width; ++index)
      if (committed[page * layers * page_tokens * width + index] != bf16_bits(0.25F))
        throw std::runtime_error("MLA wrote another layer");
  double squared_error = 0.0, squared_reference = 0.0, maximum_error = 0.0;
  for (unsigned row = 0; row < rows; ++row) {
    const unsigned visible = prefix + row + 1;
    for (unsigned head = 0; head < heads; ++head) {
      const auto& query = oracle_queries[row * heads + head];
      std::vector<double> scores(visible);
      std::vector<std::vector<double>> expanded_values(visible,
                                                       std::vector<double>(value));
      double maximum = -INFINITY;
      for (unsigned token = 0; token < visible; ++token) {
        double score = 0.0;
        for (unsigned dim = 0; dim < projected; ++dim) {
          double projection = 0.0;
          for (unsigned latent_dim = 0; latent_dim < rank; ++latent_dim)
            projection += cache[token][latent_dim] *
                          oracle_weights[head * projected + dim][latent_dim];
          if (dim < nope) score += query[dim] * projection;
          else expanded_values[token][dim - nope] = projection;
        }
        for (unsigned dim = 0; dim < rope; ++dim)
          score += query[nope + dim] * cache[token][rank + dim];
        scores[token] = score * scale;
        maximum = std::max(maximum, scores[token]);
      }
      double denominator = 0.0;
      for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
      for (unsigned dim = 0; dim < value; ++dim) {
        double reference = 0.0;
        for (unsigned token = 0; token < visible; ++token)
          reference += scores[token] * expanded_values[token][dim] / denominator;
        reference = bf16_value(bf16_bits(static_cast<float>(reference)));
        const double observed = actual[(row * heads + head) * value + dim];
        const double error = std::abs(observed - reference);
        if (!std::isfinite(observed) || error > 0.0001 + 0.015625 * std::abs(reference))
          throw std::runtime_error("MLA materialized K/V oracle mismatch");
        maximum_error = std::max(maximum_error, error);
        squared_error += error * error;
        squared_reference += reference * reference;
      }
    }
  }
  const auto relative_l2 = std::sqrt(squared_error / std::max(1e-30, squared_reference));
  if (relative_l2 > 0.006) throw std::runtime_error("MLA relative L2 gate failed");
  std::cout << "MLA prefix=" << prefix << " rows=" << rows << " heads=" << heads
            << " rank=" << rank << " relative_l2=" << relative_l2
            << " max_abs=" << maximum_error << '\n';
}

void long_context_gate(unsigned tokens) {
  constexpr unsigned heads = 32, rank = 256, nope = 64, rope = 64, value = 128;
  constexpr unsigned page_tokens = 256, width = rank + rope;
  const auto pages_count = (tokens + page_tokens - 1) / page_tokens;
  std::vector<std::uint16_t> cache(pages_count * page_tokens * width);
  std::vector<double> reference(value);
  std::vector<float> last(width);
  for (unsigned token = 0; token < tokens; ++token)
    for (unsigned dim = 0; dim < rank; ++dim) {
      const float element = (static_cast<int>((token * 17U + dim * 31U) % 257U) - 128) / 128.0F;
      const auto offset = static_cast<std::size_t>(token / page_tokens) * page_tokens * width +
          (dim / (width / 2)) * page_tokens * (width / 2) +
          (token % page_tokens) * (width / 2) + dim % (width / 2);
      cache[offset] = bf16_bits(element);
      if (dim < value) reference[dim] += element;
      if (token == tokens - 1) last[dim] = element;
    }
  for (auto& element : reference) element /= tokens;
  // A zero query gives uniform softmax. Identity V projection means the
  // independently accumulated arithmetic mean is the exact FP64 oracle.
  std::vector<std::uint16_t> weights(heads * (nope + value) * rank);
  for (unsigned head = 0; head < heads; ++head)
    for (unsigned dim = 0; dim < value; ++dim)
      weights[(head * (nope + value) + nope + dim) * rank + dim] = bf16_bits(1.0F);
  Device<std::uint16_t> device_pages(cache.size()); device_pages.upload(cache);
  std::vector<const void*> table;
  for (unsigned page = 0; page < pages_count; ++page)
    table.push_back(device_pages.pointer + page * page_tokens * width);
  Device<const void*> device_table(table.size()); device_table.upload(table);
  Device<float> query(heads * (nope + rope));
  query.upload(std::vector<float>(query.count));
  Device<float> latent(width); latent.upload(last);
  Device<std::uint16_t> device_weights(weights.size()); device_weights.upload(weights);
  Device<float> absorbed(heads * rank), output(heads * value);
  const auto splits = (tokens + expert::runtime::cuda::kMlaContextSplitTokens - 1) /
                      expert::runtime::cuda::kMlaContextSplitTokens;
  Device<float> maxima(heads * splits), sums(heads * splits), partials(heads * splits * rank);
  expert::runtime::cuda::MlaCausalLaunch launch{};
  launch.query = query.pointer;
  launch.latent = latent.pointer;
  launch.kv_up = device_weights.pointer;
  launch.page_table = device_table.pointer;
  launch.output = output.pointer;
  launch.latent_query = absorbed.pointer;
  launch.page_tokens = page_tokens;
  launch.first_position = tokens - 1;
  launch.rows = 1; launch.heads = heads; launch.kv_rank = rank;
  launch.nope_dim = nope; launch.rope_dim = rope; launch.value_dim = value;
  launch.attention_scale = 0.05F; launch.rope_theta = 10000.0F;
  launch.rope_factor = 1.0F; launch.rope_beta_fast = 32.0F;
  launch.rope_beta_slow = 1.0F; launch.rope_original_context = 4096;
  const auto invoke = [&] {
    const auto status = expert::runtime::cuda::mla_causal_latent_bf16(launch);
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
  };
  const auto validate = [&] {
    const auto observed = output.download();
    double squared_error = 0.0, squared_reference = 0.0;
    for (unsigned head = 0; head < heads; ++head)
      for (unsigned dim = 0; dim < value; ++dim) {
        const auto expected = bf16_value(bf16_bits(static_cast<float>(reference[dim])));
        const auto error = observed[head * value + dim] - expected;
        if (!std::isfinite(error)) throw std::runtime_error("nonfinite full-length MLA result");
        squared_error += error * error;
        squared_reference += expected * expected;
      }
    if (std::sqrt(squared_error / std::max(1e-30, squared_reference)) > 0.006)
      throw std::runtime_error("full-length MLA arithmetic-mean oracle mismatch");
  };
  cudaEvent_t begin{}, end{};
  checked(cudaEventCreate(&begin)); checked(cudaEventCreate(&end));
  const auto measure = [&] {
    invoke(); checked(cudaDeviceSynchronize()); validate();
    checked(cudaEventRecord(begin));
    constexpr unsigned repetitions = 3;
    for (unsigned iteration = 0; iteration < repetitions; ++iteration) invoke();
    checked(cudaEventRecord(end)); checked(cudaEventSynchronize(end));
    float ms{}; checked(cudaEventElapsedTime(&ms, begin, end));
    validate();
    return ms / repetitions;
  };
  const auto tiled_ms = measure();
  launch.partial_maxima = maxima.pointer;
  launch.partial_sums = sums.pointer;
  launch.partial_outputs = partials.pointer;
  const auto split_ms = measure();
  checked(cudaEventDestroy(begin)); checked(cudaEventDestroy(end));
  std::cout << "MLA long tokens=" << tokens << " heads=" << heads
            << " tiled_ms=" << tiled_ms << " split_ms=" << split_ms
            << " speedup=" << tiled_ms / split_ms
            << " scratch_bytes=" << (maxima.count + sums.count + partials.count) * sizeof(float)
            << std::endl;
  if (tokens >= 65536 && tiled_ms / split_ms < 2.0F)
    throw std::runtime_error("MLA context split speed prerequisite failed");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--long-context") {
      long_context_gate(8192);
      long_context_gate(65536);
      long_context_gate(262144);
      return 0;
    }
    if (argc != 1) throw std::runtime_error("unknown MLA gate argument");
    check_case(0, 1, 2, 16, 8, 8, 16, 0.125F);
    check_case(7, 3, 2, 64, 16, 32, 32, 0.25F);
    check_case(60, 8, 2, 256, 32, 64, 64, 0.125F);
    check_case(63, 1, 32, 256, 256, 64, 256, 0.05F);
    check_case(128, 1, 32, 256, 64, 64, 128, 0.05F);
    check_case(64, 1, 2, 256, 32, 64, 64, 4.0F);
    check_case(511, 3, 2, 256, 32, 64, 64, 0.125F);
    check_case(8191, 1, 2, 256, 8, 64, 16, 0.05F);
    check_case(4095, 8, 2, 256, 8, 64, 16, 0.05F);
    check_case(4095, 9, 2, 64, 8, 32, 16, 0.05F);
    check_case(4095, 1, 32, 256, 64, 64, 128, 0.05F);
    check_case(4096, 1, 2, 64, 8, 32, 16, 4.0F);
    check_case(8191, 1, 2, 256, 8, 64, 16, 0.05F, false);
    std::cout << "MLA independent numerical gate passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
