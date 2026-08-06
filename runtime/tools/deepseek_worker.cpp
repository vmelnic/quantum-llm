#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "expert/runtime/buffer_pool.hpp"
#include "expert/runtime/cuda/deepseek_decode.hpp"
#include "expert/runtime/cuda/deepseek_model.hpp"
#include "expert/runtime/cuda/deepseek_request.hpp"
#include "expert/runtime/cuda/deepseek_mtp_request.hpp"
#include "expert/runtime/cuda/deepseek_scheduler.hpp"
#include "expert/runtime/cuda/expert_directory.hpp"
#include "expert/runtime/cuda/expert_uploader.hpp"
#include "expert/runtime/cpu/deepseek_packed_executor.hpp"
#include "expert/runtime/deepseek_artifacts.hpp"
#include "expert/runtime/deepseek_catalog.hpp"
#include "expert/runtime/expert_cache.hpp"
#include "expert/runtime/gather_storage.hpp"
#include "expert/runtime/hybrid_dispatch.hpp"
#include "expert/runtime/resident_expert_set.hpp"
#include "expert/runtime/route_census.hpp"
#include "expert/runtime/windows_iocp_storage.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace er = expert::runtime;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
}

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> result;
  for (;;) {
    const auto separator = line.find('\t');
    result.push_back(line.substr(0U, separator));
    if (separator == std::string_view::npos) return result;
    line.remove_prefix(separator + 1U);
  }
}

std::vector<std::uint32_t> parse_tokens(std::string_view text) {
  std::vector<std::uint32_t> result;
  for (;;) {
    const auto separator = text.find(',');
    const auto value = std::stoull(std::string(text.substr(0U, separator)));
    require(value < 129280U, "token is outside DeepSeek vocabulary");
    result.push_back(static_cast<std::uint32_t>(value));
    if (separator == std::string_view::npos) break;
    text.remove_prefix(separator + 1U);
  }
  require(!result.empty(), "empty token sequence");
  return result;
}

std::uint8_t nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10U;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10U;
  throw std::runtime_error("invalid worker model hash");
}

er::Sha256Digest digest(std::string_view text) {
  require(text.size() == 64U, "worker model hash has the wrong length");
  er::Sha256Digest result{};
  for (std::size_t index = 0U; index < result.size(); ++index)
    result[index] = static_cast<std::byte>(
        (nibble(text[index * 2U]) << 4U) | nibble(text[index * 2U + 1U]));
  return result;
}

struct Bundle final {
  std::filesystem::path checkpoint;
  std::filesystem::path dense;
  std::filesystem::path typed;
  std::filesystem::path shared;
  std::filesystem::path routed;
  std::filesystem::path census;
  std::filesystem::path mtp;
  std::filesystem::path mtp_routed;
  er::Sha256Digest model_hash{};
  double cpu_ns{};
  double gpu_ns{};
  double h2d_bytes_per_second{};
  bool mtp_available{};
  bool mtp_runtime_ready{};
};

std::filesystem::path bundle_path(const std::filesystem::path& root,
                                  const std::string& text) {
  std::filesystem::path path(text);
  if (path.is_absolute()) return path.lexically_normal();
  for (const auto& part : path)
    require(part != "..", "worker bundle path escapes its root");
  return (root / path).lexically_normal();
}

Bundle load_bundle(const std::filesystem::path& root) {
  std::ifstream input(root / "runtime.tsv");
  std::string line;
  require(static_cast<bool>(std::getline(input, line)) &&
              (line == "deepseek-worker-bundle-v1" ||
               line == "deepseek-worker-bundle-v2" ||
               line == "deepseek-worker-bundle-v3"),
          "invalid DeepSeek worker bundle header");
  const bool has_mtp = line != "deepseek-worker-bundle-v1";
  const bool has_mtp_runtime = line == "deepseek-worker-bundle-v3";
  std::map<std::string, std::string> values;
  while (std::getline(input, line)) {
    const auto fields = split_tabs(line);
    require(fields.size() == 2U && !fields[0].empty() &&
                values.emplace(std::string(fields[0]),
                               std::string(fields[1])).second,
            "invalid or duplicate DeepSeek worker bundle field");
  }
  const auto expected_fields = has_mtp_runtime ? 13U : has_mtp ? 12U : 11U;
  require(input.eof() && values.size() == expected_fields &&
              values.at("model_id") == "17",
          "incomplete DeepSeek worker bundle");
  Bundle result;
  result.checkpoint = bundle_path(root, values.at("checkpoint"));
  result.dense = bundle_path(root, values.at("dense"));
  result.typed = bundle_path(root, values.at("typed"));
  result.shared = bundle_path(root, values.at("shared"));
  result.routed = bundle_path(root, values.at("routed"));
  result.census = bundle_path(root, values.at("census"));
  if (has_mtp) result.mtp = bundle_path(root, values.at("mtp"));
  if (has_mtp_runtime)
    result.mtp_routed = bundle_path(root, values.at("mtp_routed"));
  result.model_hash = digest(values.at("model_sha256"));
  result.cpu_ns = std::stod(values.at("cpu_ns_per_selection"));
  result.gpu_ns = std::stod(values.at("gpu_ns_per_selection"));
  result.h2d_bytes_per_second = std::stod(values.at("h2d_bytes_per_second"));
  result.mtp_available = has_mtp;
  result.mtp_runtime_ready = has_mtp_runtime;
  require(std::filesystem::is_directory(result.checkpoint) &&
              std::filesystem::is_directory(result.dense) &&
              std::filesystem::is_directory(result.typed) &&
              std::filesystem::is_directory(result.shared) &&
              std::filesystem::is_directory(result.routed) &&
              (!has_mtp ||
               (std::filesystem::is_directory(result.mtp) &&
                std::filesystem::is_regular_file(result.mtp / "manifest.json") &&
                std::filesystem::is_regular_file(
                    result.mtp / "routed" / "catalog.tsv"))) &&
              (!has_mtp_runtime ||
               (std::filesystem::is_directory(result.mtp_routed) &&
                std::filesystem::is_regular_file(
                    result.mtp_routed / "manifest.json") &&
                std::filesystem::is_regular_file(
                    result.mtp_routed / "catalog.tsv"))) &&
              std::isfinite(result.cpu_ns) && result.cpu_ns > 0.0 &&
              std::isfinite(result.gpu_ns) && result.gpu_ns > 0.0 &&
              std::isfinite(result.h2d_bytes_per_second) &&
              result.h2d_bytes_per_second > 0.0,
          "DeepSeek worker bundle dependency is unavailable");
  return result;
}

std::array<float, 32U> rope_values(std::uint32_t position, bool sine,
                                   bool compressed) {
  std::array<float, 32U> result{};
  constexpr double dimension = 64.0, factor = 16.0;
  constexpr double original_context = 65536.0;
  const double base = compressed ? 160000.0 : 10000.0;
  const auto correction = [&](double rotations) {
    return dimension *
           std::log(original_context /
                    (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(base));
  };
  const double low = std::clamp(std::floor(correction(32.0)), 0.0, 31.0);
  const double high = std::clamp(std::ceil(correction(1.0)), 0.0, 31.0);
  for (std::size_t index = 0U; index < result.size(); ++index) {
    double frequency = std::pow(base, -(2.0 * index) / dimension);
    if (compressed) {
      const double ramp = std::clamp(
          (static_cast<double>(index) - low) / std::max(high - low, 1e-3),
          0.0, 1.0);
      frequency = frequency / factor * ramp + frequency * (1.0 - ramp);
    }
    const double angle = static_cast<double>(position) * frequency;
    result[index] = static_cast<float>(sine ? std::sin(angle) : std::cos(angle));
  }
  return result;
}

std::array<std::array<float, 32U>, 8U> rope_row(std::uint32_t position) {
  std::array<std::array<float, 32U>, 8U> result{};
  result[0] = rope_values(position, false, false);
  result[1] = rope_values(position, true, false);
  result[2] = rope_values(position, false, true);
  result[3] = rope_values(position, true, true);
  const auto start4 = position + 1U >= 4U ? position + 1U - 4U : 0U;
  const auto start128 = position + 1U >= 128U ? position + 1U - 128U : 0U;
  result[4] = rope_values(start4, false, true);
  result[5] = rope_values(start4, true, true);
  result[6] = rope_values(start128, false, true);
  result[7] = rope_values(start128, true, true);
  return result;
}

struct Request final {
  std::shared_ptr<er::cuda::DeepSeekRequestState> state;
  std::shared_ptr<er::cuda::DeepSeekDecodeController> controller;
  std::shared_ptr<er::cuda::DeepSeekVerifyState> verify;
  std::shared_ptr<er::cuda::DeepSeekMtpRequestState> mtp;
  cudaStream_t stream{};
  er::cuda::DeepSeekDecodeTelemetry controller_telemetry;
  std::uint32_t predicted{};
  std::uint32_t draft{};
  std::uint32_t next_position{};
  std::uint32_t context_limit{};
  std::uint32_t slot{};
  bool draft_ready{};
  bool speculation_suppressed{};
  ~Request() {
    controller.reset();
    verify.reset();
    mtp.reset();
    state.reset();
    if (stream) static_cast<void>(cudaStreamDestroy(stream));
  }
};

struct WorkerTelemetry final {
  std::uint64_t model_steps{};
  std::uint64_t model_rows{};
  std::uint64_t model_step_ns{};
  std::uint64_t embed_rope_submit_ns{};
  std::uint64_t scheduler_poll_ns{};
  std::uint64_t output_head_ns{};
  std::uint64_t attention_route_submit_ns{};
  std::uint64_t directory_plan_ns{};
  std::uint64_t ffn_submit_ns{};
  std::uint64_t directory_release_ns{};
  std::uint64_t gpu_attention_route_plan_ns{};
  std::uint64_t gpu_ffn_release_ns{};
  std::uint64_t gpu_attention_ns{};
  std::uint64_t gpu_route_ns{};
  std::uint64_t gpu_directory_plan_ns{};
  std::uint64_t gpu_ffn_ns{};
  std::uint64_t gpu_directory_release_ns{};
  std::uint64_t gpu_attention_hca_pre_norm_ns{};
  std::uint64_t gpu_attention_projection_ns{};
  std::uint64_t gpu_sparse_attention_ns{};
  std::uint64_t gpu_attention_output_projection_ns{};
  std::uint64_t gpu_attention_hca_post_ns{};
  std::uint64_t gpu_ffn_routed_ns{};
  std::uint64_t gpu_ffn_aggregate_ns{};
  std::uint64_t gpu_ffn_shared_ns{};
  std::uint64_t gpu_ffn_merge_ns{};
  std::uint64_t gpu_ffn_hca_post_ns{};
  std::uint64_t warm_start_candidates{};
  std::uint64_t warm_start_loaded{};
  std::uint64_t warm_start_bytes{};
  std::uint64_t warm_start_ns{};
  std::uint64_t mtp_drafts{};
  std::uint64_t mtp_accepted{};
  std::uint64_t mtp_rejected{};
  std::uint64_t verify_pairs{};
  std::uint64_t useful_tokens{};
  std::uint64_t mtp_suppressions{};
};

class Model final {
 public:
  Model(const std::filesystem::path& root, std::uint32_t max_context,
        std::uint64_t ram_bytes, std::uint64_t vram_bytes,
        std::uint32_t capacity, std::uint64_t kv_cache_bytes,
        std::uint32_t kv_page_tokens, std::string placement,
        bool gpu_phase_timing, bool enable_mtp)
      : bundle_(load_bundle(root)), max_context_(max_context),
        capacity_(capacity), ram_bytes_(ram_bytes), vram_bytes_(vram_bytes),
        kv_cache_bytes_(kv_cache_bytes), kv_page_tokens_(kv_page_tokens),
        placement_(std::move(placement)), gpu_phase_timing_(gpu_phase_timing),
        mtp_enabled_(enable_mtp) {
    require(max_context_ >= 2U && capacity_ != 0U && ram_bytes_ != 0U &&
                vram_bytes_ != 0U && kv_cache_bytes_ != 0U &&
                kv_page_tokens_ != 0U,
            "invalid DeepSeek worker limits");
    auto loaded = er::load_deepseek_model_artifacts(
        bundle_.dense, bundle_.typed, bundle_.shared, bundle_.checkpoint);
    require(loaded.status.ok(), loaded.status.message());
    artifacts_ = std::move(loaded.artifacts);
    const auto catalog_status = er::DeepSeekExpertCatalog::load(
        bundle_.routed, bundle_.checkpoint, catalog_);
    require(catalog_status.ok(), catalog_status.message());
    if (bundle_.mtp_runtime_ready) {
      auto mtp_loaded = er::load_deepseek_tensor_artifacts(
          bundle_.mtp / "dense", bundle_.mtp / "typed-residency",
          bundle_.checkpoint, 7U, 19U);
      require(mtp_loaded.status.ok(), mtp_loaded.status.message());
      mtp_artifacts_ = std::move(mtp_loaded.artifacts);
      auto mtp_shared = er::load_deepseek_shared_artifacts(
          bundle_.mtp / "shared", bundle_.checkpoint, 1U, 18U);
      require(mtp_shared.status.ok(), mtp_shared.status.message());
      mtp_shared_specs_ = std::move(mtp_shared.shared);
      const auto mtp_catalog_status =
          er::DeepSeekExpertCatalog::load_namespace(
              bundle_.mtp_routed, bundle_.checkpoint, 1U, mtp_catalog_);
      require(mtp_catalog_status.ok(), mtp_catalog_status.message());
      const auto mtp_request_size =
          er::cuda::deepseek_mtp_request_state_size(max_context_);
      require(mtp_request_size.status.ok(), mtp_request_size.status.message());
      mtp_request_bytes_ = mtp_request_size.total_bytes;
      mtp_cache_bytes_ = 512ULL << 20U;
    }
    require(!mtp_enabled_ || bundle_.mtp_runtime_ready,
            "MTP execution requires worker bundle v3 resources");
    const auto request_size = er::cuda::deepseek_request_state_size(max_context_);
    require(request_size.status.ok(), request_size.status.message());
    request_bytes_ = request_size.total_bytes;
    if (mtp_enabled_) {
      const auto verify_size =
          er::cuda::deepseek_verify_state_size(max_context_);
      require(verify_size.status.ok(), verify_size.status.message());
      verify_request_bytes_ = verify_size.total_bytes;
    }
    const auto attention_per_token =
        (request_size.attention_bytes + max_context_ - 1U) / max_context_;
    require(attention_per_token <=
                std::numeric_limits<std::uint64_t>::max() / kv_page_tokens_,
            "DeepSeek KV page geometry overflow");
    kv_page_bytes_ = std::max<std::uint64_t>(
        1U, attention_per_token * kv_page_tokens_);
    kv_page_capacity_ = kv_cache_bytes_ / kv_page_bytes_;
    const auto pages_per_request =
        (max_context_ + kv_page_tokens_ - 1U) / kv_page_tokens_;
    require(kv_page_capacity_ >= pages_per_request * capacity_,
            "DeepSeek KV cache cannot reserve configured request capacity");
    constexpr std::uint64_t shared_bytes = 43ULL * 25'198'592U;
    require(vram_bytes_ >= shared_bytes + 7ULL * 13'369'344U +
                               mtp_cache_bytes_ &&
                ram_bytes_ > mtp_cache_bytes_,
            "DeepSeek VRAM cache cannot hold shared plus one route");
    std::size_t free{}, total{};
    cuda_check(cudaMemGetInfo(&free, &total), "inspect DeepSeek worker VRAM");
    const auto fixed = artifacts_.dense_device_bytes +
                       artifacts_.typed_source_bytes +
                       request_bytes_ * capacity_ + rope_table_bytes() +
                       mtp_artifacts_.dense_device_bytes +
                       mtp_artifacts_.typed_source_bytes +
                       mtp_request_bytes_ * capacity_ +
                       verify_request_bytes_ * capacity_;
    require(fixed + vram_bytes_ + (1ULL << 30U) <= free,
            "DeepSeek worker VRAM preflight failed");

    iocp_ = std::make_shared<er::WindowsIocpStorage>(2U);
    storage_ = std::make_shared<er::ExtentGatherStorage>(iocp_);
    const auto staging = std::max<std::uint64_t>(
        64ULL << 20U,
        std::max(artifacts_.maximum_source_record_bytes,
                 mtp_artifacts_.maximum_source_record_bytes));
    MEMORYSTATUSEX memory{sizeof(memory)};
    require(GlobalMemoryStatusEx(&memory) != 0,
            "inspect DeepSeek worker RAM failed");
    constexpr std::uint64_t operating_system_reserve = 4ULL << 30U;
    require(ram_bytes_ <= std::numeric_limits<std::uint64_t>::max() -
                              staging * 4U - operating_system_reserve &&
                ram_bytes_ + staging * 4U + operating_system_reserve <=
                    memory.ullAvailPhys,
            "DeepSeek worker RAM preflight failed");
    buffers_ = std::make_shared<er::FixedBufferPool>(
        4U, staging, er::kExpertPackAlignment,
        std::make_shared<er::CudaPinnedAllocator>());
    model_ = std::make_shared<er::cuda::DeepSeekResidentModelState>();
    const auto model_status = er::cuda::DeepSeekResidentModelState::load(
        *storage_, *buffers_, artifacts_.dense, artifacts_.typed, *model_);
    require(model_status.ok(), model_status.message());
    if (bundle_.mtp_runtime_ready) {
      mtp_model_ =
          std::make_shared<er::cuda::DeepSeekResidentTensorState>();
      const auto mtp_model_status =
          er::cuda::DeepSeekResidentTensorState::load(
              *storage_, *buffers_, mtp_artifacts_.dense,
              mtp_artifacts_.typed, *mtp_model_);
      require(mtp_model_status.ok(), mtp_model_status.message());
    }
    directory_ = std::make_shared<er::cuda::CudaExpertDirectory>(
        17U, er::kExpertQuantAbiDeepSeekSm86, 43U, 257U,
        std::max<std::uint32_t>(64U, capacity_ * 8U));
    uploader_ = std::make_shared<er::cuda::CudaExpertUploader>(
        er::cuda::CudaExpertUploaderOptions{
            0U, true, 0U, true});
    er::ExpertCacheConfig cache_config;
    const auto target_ram_bytes = ram_bytes_ - mtp_cache_bytes_;
    const auto target_vram_bytes = vram_bytes_ - mtp_cache_bytes_;
    cache_config.ram = {target_ram_bytes, target_ram_bytes,
                        target_ram_bytes * 7U / 8U};
    cache_config.vram = {target_vram_bytes, target_vram_bytes,
                         target_vram_bytes * 7U / 8U};
    cache_config.retain_host_copy = true;
    cache_config.trusted_immutable_source = true;
    cache_ = std::make_unique<er::ExpertCache>(
        cache_config, storage_, uploader_, buffers_, directory_);
    const auto shared_status = er::ResidentExpertSet::load(
        *cache_, artifacts_.shared, shared_);
    require(shared_status.ok(), shared_status.message());
    if (bundle_.mtp_runtime_ready) {
      mtp_directory_ = std::make_shared<er::cuda::CudaExpertDirectory>(
          18U, er::kExpertQuantAbiDeepSeekSm86, 1U, 257U,
          std::max<std::uint32_t>(16U, capacity_ * 8U));
      mtp_uploader_ = std::make_shared<er::cuda::CudaExpertUploader>(
          er::cuda::CudaExpertUploaderOptions{0U, true, 0U, true});
      er::ExpertCacheConfig mtp_cache_config;
      mtp_cache_config.ram = {mtp_cache_bytes_, mtp_cache_bytes_,
                              mtp_cache_bytes_ * 7U / 8U};
      mtp_cache_config.vram = {mtp_cache_bytes_, mtp_cache_bytes_,
                               mtp_cache_bytes_ * 7U / 8U};
      mtp_cache_config.retain_host_copy = true;
      mtp_cache_config.trusted_immutable_source = true;
      mtp_cache_ = std::make_unique<er::ExpertCache>(
          mtp_cache_config, storage_, mtp_uploader_, buffers_,
          mtp_directory_);
      const auto mtp_shared_status = er::ResidentExpertSet::load(
          *mtp_cache_, mtp_shared_specs_, mtp_shared_);
      require(mtp_shared_status.ok(), mtp_shared_status.message());
    }
    initialize_rope_table();
    cpu_ = std::make_shared<er::cpu::DeepSeekPackedExecutor>(
        er::cpu::DeepSeekPackedExecutorConfig{
            std::max(1U, std::thread::hardware_concurrency()),
            8U, 8U, 10.0F, true, true});
    planner_ = std::make_shared<er::HybridDispatchPlanner>(
        er::HybridDispatchConfig{bundle_.cpu_ns, bundle_.gpu_ns,
                                 bundle_.h2d_bytes_per_second,
                                 0.125, 256U, 256U});
    auto census_loaded = er::RouteCensus::load(
        bundle_.census,
        {17U, bundle_.model_hash, er::kExpertQuantAbiDeepSeekSm86,
         43U, 256U, 6U, 4096U});
    if (census_loaded.status.ok()) {
      census_ = std::shared_ptr<er::RouteCensus>(
          std::move(census_loaded.census));
    } else {
      require(census_loaded.status.code() == er::ErrorCode::open_failed,
              census_loaded.status.message());
      census_ = std::make_shared<er::RouteCensus>(er::RouteCensusConfig{
          17U, bundle_.model_hash, er::kExpertQuantAbiDeepSeekSm86,
          43U, 256U, 6U, 4096U});
    }
    warm_from_census();
    scheduler_ = std::make_unique<er::cuda::DeepSeekDecodeScheduler>(
        er::cuda::DeepSeekDecodeSchedulerConfig{
            17U, capacity_, std::max<std::uint32_t>(6U, capacity_ * 2U),
            capacity_, 1U, 1U, true},
        *cache_, catalog_,
        er::cuda::DeepSeekHybridSchedulerDependencies{cpu_, planner_, census_});
  }

  ~Model() {
    const auto saved = census_->save(bundle_.census);
    if (!saved.ok())
      std::cerr << "route census save failed: " << saved.message() << '\n';
    scheduler_.reset();
    if (rope_table_) static_cast<void>(cudaFree(rope_table_));
  }

  std::unique_ptr<Request> create_request() {
    auto state = er::cuda::create_deepseek_request_state(
        model_, {max_context_, request_bytes_});
    require(state.status.ok() && state.state, state.status.message());
    auto request = std::make_unique<Request>();
    request->state = std::move(state.state);
    if (bundle_.mtp_runtime_ready) {
      auto mtp_state = er::cuda::create_deepseek_mtp_request_state(
          model_, mtp_model_, {max_context_, mtp_request_bytes_});
      require(mtp_state.status.ok() && mtp_state.state,
              mtp_state.status.ok() ? "MTP request returned no ownership"
                                    : mtp_state.status.message());
      request->mtp = std::move(mtp_state.state);
    }
    cuda_check(cudaStreamCreateWithFlags(&request->stream,
                                         cudaStreamNonBlocking),
               "create DeepSeek request stream");
    auto controller = er::cuda::create_deepseek_decode_controller(
        request->state, directory_, request->stream);
    require(controller.status.ok() && controller.controller,
            controller.status.message());
    request->controller = std::move(controller.controller);
    if (mtp_enabled_) {
      auto verify = er::cuda::create_deepseek_verify_state(
          request->state, verify_request_bytes_);
      require(verify.status.ok() && verify.state,
              verify.status.ok() ? "verify state returned no ownership"
                                 : verify.status.message());
      request->verify = std::move(verify.state);
      const auto configured =
          request->controller->configure_verify(request->verify);
      require(configured.ok(), configured.message());
    }
    auto workspace = er::cuda::create_deepseek_ffn_hybrid_workspace();
    require(workspace.status.ok() && workspace.workspace,
            workspace.status.ok() ? "hybrid workspace returned no ownership"
                                  : workspace.status.message());
    const auto configured = request->controller->configure_hybrid(
        cpu_, std::move(workspace.workspace));
    require(configured.ok(), configured.message());
    if (gpu_phase_timing_) {
      const auto timing = request->controller->enable_gpu_phase_timing();
      require(timing.ok(), timing.message());
    }
    return request;
  }

  std::vector<std::uint32_t> forward(
      std::span<Request* const> requests,
      std::span<const std::uint32_t> tokens,
      std::span<const std::uint32_t> positions) {
    require(!requests.empty() && requests.size() == tokens.size() &&
                requests.size() == positions.size() &&
                requests.size() <= capacity_,
            "invalid DeepSeek worker batch");
    const auto model_step_started = std::chrono::steady_clock::now();
    const auto submit_started = model_step_started;
    std::vector<std::uint64_t> operations;
    operations.reserve(requests.size());
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      require(positions[index] < requests[index]->context_limit,
              "DeepSeek reserved context exhausted");
      const auto embedded = requests[index]->state->embed(
          tokens[index], requests[index]->stream);
      require(embedded.ok(), embedded.message());
      const auto operation = next_operation_++;
      const auto submitted = scheduler_->submit(
          operation, requests[index]->controller,
          {requests[index]->state->current_streams(),
           rope_at(positions[index]),
           positions[index], tokens[index], 0U, 43U});
      require(submitted.ok(), submitted.message());
      operations.push_back(operation);
    }
    telemetry_.embed_rope_submit_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - submit_started)
            .count());
    const auto scheduler_started = std::chrono::steady_clock::now();
    std::size_t completed = 0U;
    while (completed != operations.size()) {
      const auto polled = scheduler_->poll();
      require(polled.ok(), polled.message());
      completed = 0U;
      for (const auto operation : operations) {
        const auto state = scheduler_->inspect(operation);
        require(state.has_value(), "DeepSeek scheduled operation disappeared");
        require(state->state != er::cuda::DeepSeekScheduledState::failed &&
                    state->state != er::cuda::DeepSeekScheduledState::cancelled,
                state->status.message());
        if (state->state == er::cuda::DeepSeekScheduledState::complete)
          ++completed;
      }
      if (completed != operations.size()) {
        const auto waited = scheduler_->wait_for_cuda_progress();
        require(waited.ok(), waited.message());
        if (scheduler_->snapshot().cuda_pending_requests == 0U)
          std::this_thread::yield();
      }
    }
    telemetry_.scheduler_poll_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - scheduler_started)
            .count());
    for (auto* request : requests) {
      const auto current = request->controller->telemetry();
      telemetry_.attention_route_submit_ns +=
          current.attention_route_submit_ns -
          request->controller_telemetry.attention_route_submit_ns;
      telemetry_.directory_plan_ns +=
          current.directory_plan_ns -
          request->controller_telemetry.directory_plan_ns;
      telemetry_.ffn_submit_ns +=
          current.ffn_submit_ns - request->controller_telemetry.ffn_submit_ns;
      telemetry_.directory_release_ns +=
          current.directory_release_ns -
          request->controller_telemetry.directory_release_ns;
      telemetry_.gpu_attention_route_plan_ns +=
          current.gpu_attention_route_plan_ns -
          request->controller_telemetry.gpu_attention_route_plan_ns;
      telemetry_.gpu_ffn_release_ns +=
          current.gpu_ffn_release_ns -
          request->controller_telemetry.gpu_ffn_release_ns;
      telemetry_.gpu_attention_ns +=
          current.gpu_attention_ns -
          request->controller_telemetry.gpu_attention_ns;
      telemetry_.gpu_route_ns +=
          current.gpu_route_ns - request->controller_telemetry.gpu_route_ns;
      telemetry_.gpu_directory_plan_ns +=
          current.gpu_directory_plan_ns -
          request->controller_telemetry.gpu_directory_plan_ns;
      telemetry_.gpu_ffn_ns +=
          current.gpu_ffn_ns - request->controller_telemetry.gpu_ffn_ns;
      telemetry_.gpu_directory_release_ns +=
          current.gpu_directory_release_ns -
          request->controller_telemetry.gpu_directory_release_ns;
      telemetry_.gpu_attention_hca_pre_norm_ns +=
          current.gpu_attention_hca_pre_norm_ns -
          request->controller_telemetry.gpu_attention_hca_pre_norm_ns;
      telemetry_.gpu_attention_projection_ns +=
          current.gpu_attention_projection_ns -
          request->controller_telemetry.gpu_attention_projection_ns;
      telemetry_.gpu_sparse_attention_ns +=
          current.gpu_sparse_attention_ns -
          request->controller_telemetry.gpu_sparse_attention_ns;
      telemetry_.gpu_attention_output_projection_ns +=
          current.gpu_attention_output_projection_ns -
          request->controller_telemetry.gpu_attention_output_projection_ns;
      telemetry_.gpu_attention_hca_post_ns +=
          current.gpu_attention_hca_post_ns -
          request->controller_telemetry.gpu_attention_hca_post_ns;
      telemetry_.gpu_ffn_routed_ns +=
          current.gpu_ffn_routed_ns -
          request->controller_telemetry.gpu_ffn_routed_ns;
      telemetry_.gpu_ffn_aggregate_ns +=
          current.gpu_ffn_aggregate_ns -
          request->controller_telemetry.gpu_ffn_aggregate_ns;
      telemetry_.gpu_ffn_shared_ns +=
          current.gpu_ffn_shared_ns -
          request->controller_telemetry.gpu_ffn_shared_ns;
      telemetry_.gpu_ffn_merge_ns +=
          current.gpu_ffn_merge_ns -
          request->controller_telemetry.gpu_ffn_merge_ns;
      telemetry_.gpu_ffn_hca_post_ns +=
          current.gpu_ffn_hca_post_ns -
          request->controller_telemetry.gpu_ffn_hca_post_ns;
      request->controller_telemetry = current;
    }
    const auto output_started = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> result(requests.size());
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      const auto projected = requests[index]->state->project_logits(
          requests[index]->stream);
      require(projected.ok(), projected.message());
      cuda_check(cudaMemcpyAsync(&result[index],
                                 requests[index]->state->sampled_token(),
                                 sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                 requests[index]->stream),
                 "queue DeepSeek sampled token copy");
    }
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      cuda_check(cudaStreamSynchronize(requests[index]->stream),
                 "complete DeepSeek sampled token copy");
      const auto retired = scheduler_->retire(operations[index]);
      require(retired.ok(), retired.message());
    }
    telemetry_.output_head_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - output_started)
            .count());
    ++telemetry_.model_steps;
    telemetry_.model_rows += requests.size();
    telemetry_.model_step_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - model_step_started)
            .count());
    const auto elapsed_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - model_step_started)
            .count());
    update_moving_average(ordinary_ns_per_token_, ordinary_samples_,
                          elapsed_ns / requests.size());
    return result;
  }

  void advance_mtp(Request& request, std::uint32_t next_token,
                   const float* target_streams, std::uint32_t position,
                   bool produce_draft) {
    require(mtp_enabled_ && request.mtp && mtp_cache_ && mtp_directory_,
            "MTP advance requires an enabled request runtime");
    const auto rope = rope_at(position);
    auto status = request.mtp->prepare(
        next_token, target_streams, position, rope.base_cosine,
        rope.base_sine, request.stream);
    require(status.ok(), status.message());
    if (!produce_draft) {
      status = request.mtp->abandon_draft();
      require(status.ok(), status.message());
      request.draft_ready = false;
      return;
    }

    std::vector<er::ExpertLease> leases;
    for (;;) {
      auto plan = mtp_directory_->pin_or_collect_misses(
          0U, request.mtp->expert_indices(),
          request.mtp->selection_count(), request.stream, true);
      require(plan.status.ok(), plan.status.message());
      if (plan.missing_experts.empty()) {
        require(plan.pin_id != 0U,
                "MTP directory returned no execution pin");
        status = request.mtp->complete(mtp_directory_->device_entries(), 257U,
                                       request.stream);
        require(status.ok(), status.message());
        cuda_check(cudaMemcpyAsync(&request.draft,
                                   request.mtp->draft_token(),
                                   sizeof(request.draft),
                                   cudaMemcpyDeviceToHost, request.stream),
                   "copy MTP draft token");
        status = mtp_directory_->release_pins_async(plan.pin_id,
                                                    request.stream);
        require(status.ok(), status.message());
        cuda_check(cudaStreamSynchronize(request.stream),
                   "complete MTP draft");
        break;
      }
      if (plan.pin_id != 0U) {
        status = mtp_directory_->release_pins(plan.pin_id, request.stream);
        require(status.ok(), status.message());
      }
      for (const auto expert : plan.missing_experts) {
        require(expert < 256U,
                "always-resident MTP shared expert is unavailable");
        const auto* record = mtp_catalog_.find(0U, expert);
        require(record != nullptr,
                "MTP route references an absent catalog expert");
        auto pending = mtp_cache_->acquire(
            {18U, 0U, expert, er::kExpertQuantAbiDeepSeekSm86}, *record);
        while (pending.wait_for(std::chrono::milliseconds(1)) !=
               std::future_status::ready)
          std::this_thread::yield();
        auto acquired = pending.get();
        require(acquired.status.ok() && acquired.lease,
                acquired.status.ok() ? "MTP cache returned no device lease"
                                     : acquired.status.message());
        leases.push_back(std::move(acquired.lease));
      }
    }
    request.draft_ready = true;
    ++telemetry_.mtp_drafts;
  }

  std::vector<std::uint32_t> verify_draft(Request& request) {
    require(mtp_enabled_ && request.verify && request.draft_ready &&
                request.next_position + 1U < request.context_limit,
            "invalid DeepSeek speculative verification request");
    const auto started = std::chrono::steady_clock::now();
    const auto guaranteed = request.predicted;
    const auto draft = request.draft;
    const auto position = request.next_position;
    auto status = request.state->embed(guaranteed, request.stream);
    require(status.ok(), status.message());
    status = request.verify->embed_speculative(draft, request.stream);
    require(status.ok(), status.message());
    const auto operation = next_operation_++;
    status = scheduler_->submit_verify(
        operation, request.controller,
        {{rope_at(position), rope_at(position + 1U)},
         {position, position + 1U}, {guaranteed, draft}, 0U, 43U});
    require(status.ok(), status.message());
    for (;;) {
      status = scheduler_->poll();
      require(status.ok(), status.message());
      const auto scheduled = scheduler_->inspect(operation);
      require(scheduled.has_value(),
              "DeepSeek verify operation disappeared");
      require(scheduled->state != er::cuda::DeepSeekScheduledState::failed &&
                  scheduled->state !=
                      er::cuda::DeepSeekScheduledState::cancelled,
              scheduled->status.message());
      if (scheduled->state == er::cuda::DeepSeekScheduledState::complete)
        break;
      if (scheduler_->snapshot().cuda_pending_requests != 0U) {
        status = scheduler_->wait_for_cuda_progress();
        require(status.ok(), status.message());
      } else {
        std::this_thread::yield();
      }
    }
    status = request.verify->project_pair_logits(request.stream);
    require(status.ok(), status.message());
    std::array<std::uint32_t, 2U> sampled{};
    cuda_check(cudaMemcpyAsync(
                   sampled.data(), request.verify->primary_sampled_token(),
                   sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                   request.stream),
               "copy DeepSeek verifier token");
    cuda_check(cudaMemcpyAsync(
                   sampled.data() + 1U,
                   request.verify->bonus_sampled_token(),
                   sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                   request.stream),
               "copy DeepSeek verifier bonus token");
    cuda_check(cudaStreamSynchronize(request.stream),
               "complete DeepSeek pair vocabulary heads");

    const bool accepted = sampled[0] == draft;
    if (accepted) {
      // Bring the MTP causal cache through H_A/B before H_B/C becomes the next
      // draft boundary. The first result is intentionally abandoned because
      // the target verifier already produced the bonus token C.
      advance_mtp(request, draft, request.state->current_streams(), position,
                  false);
      status = request.verify->finish_transaction(true, request.stream);
      require(status.ok(), status.message());
      advance_mtp(request, sampled[1], request.state->current_streams(),
                  position + 1U, true);
      request.predicted = sampled[1];
      request.next_position += 2U;
      ++telemetry_.mtp_accepted;
    } else {
      status = request.verify->finish_transaction(false, request.stream);
      require(status.ok(), status.message());
      advance_mtp(request, sampled[0], request.state->current_streams(),
                  position, true);
      request.predicted = sampled[0];
      ++request.next_position;
      ++telemetry_.mtp_rejected;
    }
    status = scheduler_->retire(operation);
    require(status.ok(), status.message());
    ++telemetry_.model_steps;
    telemetry_.model_rows += 2U;
    ++telemetry_.verify_pairs;
    telemetry_.model_step_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    const auto elapsed_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    update_moving_average(speculative_ns_per_useful_, speculative_samples_,
                          elapsed_ns / (accepted ? 2.0 : 1.0));
    if (speculative_samples_ >= 2U && ordinary_samples_ != 0U &&
        speculative_ns_per_useful_ > ordinary_ns_per_token_ * 1.05) {
      request.speculation_suppressed = true;
      request.draft_ready = false;
      ++telemetry_.mtp_suppressions;
    }
    return accepted ? std::vector<std::uint32_t>{guaranteed, draft}
                    : std::vector<std::uint32_t>{guaranteed};
  }

  std::uint32_t capacity() const noexcept { return capacity_; }
  std::uint32_t max_context() const noexcept { return max_context_; }
  std::uint64_t ram_bytes() const noexcept { return ram_bytes_; }
  std::uint64_t vram_bytes() const noexcept { return vram_bytes_; }
  std::uint32_t kv_page_tokens() const noexcept { return kv_page_tokens_; }
  std::uint64_t kv_page_bytes() const noexcept { return kv_page_bytes_; }
  std::uint64_t kv_page_capacity() const noexcept {
    return kv_page_capacity_;
  }
  std::uint64_t kv_pages_per_request() const noexcept {
    return (max_context_ + kv_page_tokens_ - 1U) / kv_page_tokens_;
  }
  const std::string& placement() const noexcept { return placement_; }
  er::cuda::DeepSeekDecodeSchedulerSnapshot scheduler_snapshot() const {
    return scheduler_->snapshot();
  }
  er::RouteCensusSnapshot census_snapshot() const { return census_->snapshot(); }
  er::TelemetrySnapshot cache_snapshot() const { return cache_->telemetry(); }
  er::cuda::CudaExpertUploaderTelemetry uploader_snapshot() const {
    return uploader_->telemetry();
  }
  er::cpu::DeepSeekPackedExecutorTelemetry cpu_snapshot() const {
    return cpu_->telemetry();
  }
  er::HybridDispatchTelemetry planner_snapshot() const {
    return planner_->telemetry();
  }
  WorkerTelemetry worker_snapshot() const noexcept { return telemetry_; }
  const char* prefetch_state() const noexcept {
    if (placement_ == "capacity") return "disabled";
    return telemetry_.warm_start_loaded == 0U ? "observing" : "ready";
  }
  bool prefetch_enabled() const noexcept {
    return telemetry_.warm_start_loaded != 0U;
  }
  bool gpu_phase_timing() const noexcept { return gpu_phase_timing_; }
  bool mtp_available() const noexcept { return bundle_.mtp_available; }
  bool mtp_runtime_ready() const noexcept {
    return bundle_.mtp_runtime_ready && mtp_model_ && mtp_cache_ &&
           mtp_directory_ && mtp_shared_.size() == 1U;
  }
  bool mtp_enabled() const noexcept { return mtp_enabled_; }
  void record_useful_tokens(std::size_t count) noexcept {
    telemetry_.useful_tokens += count;
  }
  bool speculation_active(const Request& request) const noexcept {
    return mtp_enabled_ && request.draft_ready &&
           !request.speculation_suppressed;
  }

 private:
  static void update_moving_average(double& value, std::uint64_t& samples,
                                    double observation) noexcept {
    constexpr double alpha = 0.25;
    value = samples == 0U ? observation
                          : alpha * observation + (1.0 - alpha) * value;
    ++samples;
  }
  static constexpr std::uint64_t rope_row_values = 8ULL * 32U;
  std::uint64_t rope_table_bytes() const noexcept {
    return static_cast<std::uint64_t>(max_context_) * rope_row_values *
           sizeof(float);
  }

  void initialize_rope_table() {
    std::vector<std::array<std::array<float, 32U>, 8U>> host(max_context_);
    for (std::uint32_t position = 0U; position < max_context_; ++position)
      host[position] = rope_row(position);
    float* candidate = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&candidate),
                          rope_table_bytes()),
               "allocate DeepSeek RoPE table");
    const auto copied = cudaMemcpy(candidate, host.data(), rope_table_bytes(),
                                   cudaMemcpyHostToDevice);
    if (copied != cudaSuccess) {
      static_cast<void>(cudaFree(candidate));
      cuda_check(copied, "upload DeepSeek RoPE table");
    }
    rope_table_ = candidate;
  }

  er::cuda::DeepSeekDecodeRope rope_at(std::uint32_t position) const {
    require(position < max_context_, "DeepSeek RoPE position is out of range");
    auto* row = rope_table_ + static_cast<std::size_t>(position) *
                                 rope_row_values;
    return {row, row + 32U, row + 64U, row + 96U,
            row + 128U, row + 160U, row + 192U, row + 224U};
  }

  void warm_from_census() {
    if (placement_ == "capacity") return;
    const auto usage = cache_->usage();
    const auto* representative = catalog_.find(0U, 0U);
    require(representative != nullptr,
            "DeepSeek warm start has no representative routed record");
    const auto device_record_bytes = representative->device_bytes == 0U
                                         ? representative->stored_bytes
                                         : representative->device_bytes;
    require(representative->stored_bytes != 0U && device_record_bytes != 0U,
            "DeepSeek warm start record geometry is empty");
    const auto warm_ram_limit = ram_bytes_ - mtp_cache_bytes_;
    const auto warm_vram_limit = vram_bytes_ - mtp_cache_bytes_;
    if (usage.ram_bytes >= warm_ram_limit ||
        usage.vram_bytes >= warm_vram_limit)
      return;
    // acquire() resolves all the way to a device lease. Bounding only by RAM
    // can leave the final waiter permanently queued once every VRAM entry is
    // hotter than the next census candidate. Until a host-only load primitive
    // exists, the safe warm set must fit both cache tiers.
    const auto ram_entries =
        (warm_ram_limit - usage.ram_bytes) / representative->stored_bytes;
    const auto vram_entries =
        (warm_vram_limit - usage.vram_bytes) / device_record_bytes;
    const auto maximum_entries = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            std::min(ram_entries, vram_entries),
            static_cast<std::uint64_t>(er::kDeepSeekCatalogLayers) * 6U));
    if (maximum_entries == 0U) return;
    // Startup warms at most one complete top-k working set per layer. Filling
    // the entire cache speculatively is both unnecessary for readiness and
    // vulnerable to allocator/driver headroom below the logical tier budget.
    constexpr std::size_t maximum_per_layer = 6U;
    auto warm = census_->stable_warm_set(maximum_entries, maximum_per_layer);
    telemetry_.warm_start_candidates = warm.size();
    if (warm.empty()) return;
    const auto started = std::chrono::steady_clock::now();
    // stable_warm_set() is hottest-first. Loading in reverse ensures that, if
    // the device tier fills, later/hotter entries displace earlier/cooler ones.
    for (auto item = warm.rbegin(); item != warm.rend(); ++item) {
      const auto* record = catalog_.find(item->key.layer, item->key.expert);
      require(record != nullptr, "route census references an absent expert");
      auto handle = cache_->acquire(item->key, *record);
      while (handle.wait_for(std::chrono::milliseconds(1)) !=
             std::future_status::ready) {
        std::this_thread::yield();
      }
      auto acquired = handle.get();
      require(acquired.status.ok() && acquired.lease,
              acquired.status.ok() ? "warm start returned no device lease"
                                   : acquired.status.message());
      acquired.lease = {};
      const auto evidence = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(item->total_selections, 255U));
      require(cache_->record_access(item->key, std::max(1U, evidence)),
              "warm-start expert disappeared before heat publication");
      ++telemetry_.warm_start_loaded;
      telemetry_.warm_start_bytes += record->stored_bytes;
    }
    telemetry_.warm_start_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
  }

  Bundle bundle_;
  std::uint32_t max_context_{}, capacity_{};
  std::uint64_t ram_bytes_{}, vram_bytes_{}, request_bytes_{}, next_operation_{1U};
  std::uint64_t mtp_request_bytes_{}, mtp_cache_bytes_{},
      verify_request_bytes_{};
  std::uint64_t kv_cache_bytes_{}, kv_page_bytes_{}, kv_page_capacity_{};
  std::uint32_t kv_page_tokens_{};
  std::string placement_;
  bool gpu_phase_timing_{};
  bool mtp_enabled_{};
  double ordinary_ns_per_token_{};
  double speculative_ns_per_useful_{};
  std::uint64_t ordinary_samples_{}, speculative_samples_{};
  er::DeepSeekModelArtifacts artifacts_;
  er::DeepSeekTensorArtifacts mtp_artifacts_;
  std::vector<er::ResidentExpertSpec> mtp_shared_specs_;
  er::DeepSeekExpertCatalog catalog_;
  er::DeepSeekExpertCatalog mtp_catalog_;
  std::shared_ptr<er::WindowsIocpStorage> iocp_;
  std::shared_ptr<er::ExtentGatherStorage> storage_;
  std::shared_ptr<er::FixedBufferPool> buffers_;
  std::shared_ptr<er::cuda::DeepSeekResidentModelState> model_;
  std::shared_ptr<er::cuda::DeepSeekResidentTensorState> mtp_model_;
  float* rope_table_{};
  std::shared_ptr<er::cuda::CudaExpertDirectory> directory_;
  std::shared_ptr<er::cuda::CudaExpertUploader> uploader_;
  std::unique_ptr<er::ExpertCache> cache_;
  er::ResidentExpertSet shared_;
  std::shared_ptr<er::cuda::CudaExpertDirectory> mtp_directory_;
  std::shared_ptr<er::cuda::CudaExpertUploader> mtp_uploader_;
  std::unique_ptr<er::ExpertCache> mtp_cache_;
  er::ResidentExpertSet mtp_shared_;
  std::shared_ptr<er::cpu::DeepSeekPackedExecutor> cpu_;
  std::shared_ptr<er::HybridDispatchPlanner> planner_;
  std::shared_ptr<er::RouteCensus> census_;
  std::unique_ptr<er::cuda::DeepSeekDecodeScheduler> scheduler_;
  WorkerTelemetry telemetry_;
};

struct Active final {
  std::unique_ptr<Request> request;
};

std::uint32_t free_slot(const std::unordered_map<std::uint64_t, Active>& active,
                        std::uint32_t capacity) {
  std::vector<bool> used(capacity);
  for (const auto& [id, item] : active) {
    static_cast<void>(id);
    require(item.request->slot < capacity, "invalid active worker slot");
    used[item.request->slot] = true;
  }
  const auto available = std::find(used.begin(), used.end(), false);
  require(available != used.end(), "no DeepSeek worker slot available");
  return static_cast<std::uint32_t>(available - used.begin());
}

int worker_loop(Model& model) {
  std::unordered_map<std::uint64_t, Active> active;
  std::cout << "{\"type\":\"ready\",\"protocol\":5,\"capacity\":"
            << model.capacity()
            << ",\"prefill_mode\":\"causal_sequential\""
            << ",\"prefill_chunk_tokens\":1"
            << ",\"request_stream_mode\":\"per_request_nonblocking\""
            << ",\"rope_mode\":\"resident_table\""
            << ",\"kv_dtype\":\"bf16\""
            << ",\"kv_allocation\":\"preallocated\""
            << ",\"kv_page_tokens\":" << model.kv_page_tokens()
            << ",\"kv_page_bytes\":" << model.kv_page_bytes()
            << ",\"kv_page_capacity\":" << model.kv_page_capacity()
            << ",\"placement_profile\":\"" << model.placement()
            << "\",\"ram_cache_bytes\":" << model.ram_bytes()
            << ",\"vram_cache_bytes\":" << model.vram_bytes()
            << ",\"placement_prefetch_enabled\":"
            << (model.prefetch_enabled() ? "true" : "false")
            << ",\"placement_prefetch_state\":\""
            << model.prefetch_state() << '\"'
            << ",\"placement_minimum_observations\":"
            << (model.placement() == "latency" ? 1 : 2)
            << ",\"gpu_phase_timing\":"
            << (model.gpu_phase_timing() ? "true" : "false")
            << ",\"mtp_resource_available\":"
            << (model.mtp_available() ? "true" : "false")
            << ",\"mtp_runtime_ready\":"
            << (model.mtp_runtime_ready() ? "true" : "false")
            << ",\"mtp_enabled\":"
            << (model.mtp_enabled() ? "true" : "false") << "}\n"
            << std::flush;
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      const auto fields = split_tabs(line);
      if (fields[0] == "PING") {
        require(fields.size() == 1U, "invalid PING");
        std::cout << "{\"type\":\"pong\"}\n" << std::flush;
      } else if (fields[0] == "STATS") {
        require(fields.size() == 1U, "invalid STATS");
        const auto scheduler = model.scheduler_snapshot();
        const auto census = model.census_snapshot();
        const auto cache = model.cache_snapshot();
        const auto uploader = model.uploader_snapshot();
        const auto cpu = model.cpu_snapshot();
        const auto planner = model.planner_snapshot();
        const auto worker = model.worker_snapshot();
        std::uint64_t reserved_pages = 0U;
        for (const auto& [id, item] : active) {
          static_cast<void>(id);
          reserved_pages +=
              (item.request->context_limit + model.kv_page_tokens() - 1U) /
              model.kv_page_tokens();
        }
        std::cout << "{\"type\":\"stats\",\"active_requests\":"
                  << active.size() << ",\"kv_allocated_pages\":"
                  << active.size() * model.kv_pages_per_request()
                  << ",\"kv_reserved_pages\":" << reserved_pages
                  << ",\"route_observations\":"
                  << scheduler.route_observations
                  << ",\"scheduler_prefetch_predictions\":"
                  << scheduler.prefetch_predictions
                  << ",\"scheduler_prefetch_scheduled\":"
                  << scheduler.prefetch_scheduled
                  << ",\"scheduler_prefetch_completed\":"
                  << scheduler.prefetch_completed
                  << ",\"scheduler_prefetch_useful\":"
                  << scheduler.prefetch_useful
                  << ",\"scheduler_prefetch_late\":"
                  << scheduler.prefetch_late
                  << ",\"scheduler_prefetch_incorrect\":"
                  << scheduler.prefetch_incorrect
                  << ",\"scheduler_prefetch_cancelled\":"
                  << scheduler.prefetch_cancelled
                  << ",\"scheduler_prefetch_evicted_before_use\":"
                  << scheduler.prefetch_evicted_before_use
                  << ",\"census_routes\":" << census.completed_routes
                  << ",\"cache_vram_hits\":" << cache.acquire_vram_hits
                  << ",\"cache_ram_hits\":" << cache.acquire_ram_hits
                  << ",\"cache_ssd_misses\":" << cache.acquire_ssd_misses
                  << ",\"cache_loads_started\":" << cache.load_started
                  << ",\"cache_loads_completed\":" << cache.load_completed
                  << ",\"cache_uploads_started\":" << cache.upload_started
                  << ",\"cache_uploads_completed\":" << cache.upload_completed
                  << ",\"cache_read_bytes\":" << cache.read_bytes
                  << ",\"cache_uploaded_bytes\":" << cache.uploaded_bytes
                  << ",\"cache_storage_wait_ns\":" << cache.storage_wait_ns
                  << ",\"cache_ram_retention_copy_ns\":"
                  << cache.ram_retention_copy_ns
                  << ",\"cache_upload_wait_ns\":" << cache.upload_wait_ns
                  << ",\"cache_ram_bytes\":" << cache.ram_bytes
                  << ",\"cache_ram_high_water\":" << cache.ram_high_water
                  << ",\"cache_vram_bytes\":" << cache.vram_bytes
                  << ",\"cache_vram_high_water\":" << cache.vram_high_water
                  << ",\"cache_evictions\":" << cache.eviction_count
                  << ",\"cache_stalled_by_budget\":"
                  << cache.stalled_by_budget
                  << ",\"cache_io_errors\":" << cache.io_errors
                  << ",\"cache_upload_errors\":" << cache.upload_errors
                  << ",\"uploader_device_allocations\":"
                  << uploader.device_allocations
                  << ",\"uploader_recycled_acquires\":"
                  << uploader.recycled_acquires
                  << ",\"uploader_staging_allocations\":"
                  << uploader.staging_allocations
                  << ",\"uploader_compact_h2d_bytes\":"
                  << uploader.compact_h2d_bytes
                  << ",\"uploader_compact_cache_hits\":"
                  << uploader.compact_cache_hits
                  << ",\"uploader_compact_cache_misses\":"
                  << uploader.compact_cache_misses
                  << ",\"scheduler_layer_advances\":"
                  << scheduler.layer_advances
                  << ",\"scheduler_cuda_pending_polls\":"
                  << scheduler.cuda_pending_polls
                  << ",\"scheduler_cuda_waits\":" << scheduler.cuda_waits
                  << ",\"scheduler_cuda_wait_ns\":"
                  << scheduler.cuda_wait_ns
                  << ",\"scheduler_expert_suspensions\":"
                  << scheduler.expert_suspensions
                  << ",\"scheduler_acquires_started\":"
                  << scheduler.acquires_started
                  << ",\"scheduler_acquires_completed\":"
                  << scheduler.acquires_completed
                  << ",\"scheduler_host_resolves\":"
                  << scheduler.host_resolves
                  << ",\"scheduler_cpu_placements\":"
                  << scheduler.cpu_placements
                  << ",\"scheduler_hybrid_layers\":"
                  << scheduler.hybrid_layers
                  << ",\"scheduler_controller_advance_ns\":"
                  << scheduler.controller_advance_ns
                  << ",\"scheduler_expert_wait_ns\":"
                  << scheduler.expert_wait_ns
                  << ",\"scheduler_poll_ns\":" << scheduler.poll_ns
                  << ",\"cpu_workers_used_last\":"
                  << cpu.workers_used_last
                  << ",\"cpu_execute_calls\":" << cpu.execute_calls
                  << ",\"cpu_selections\":" << cpu.selections
                  << ",\"cpu_source_weight_bytes\":"
                  << cpu.source_weight_bytes
                  << ",\"cpu_compute_ns\":" << cpu.compute_ns
                  << ",\"planner_plans\":" << planner.plans
                  << ",\"planner_candidates\":" << planner.candidates
                  << ",\"planner_cpu_cost_wins\":"
                  << planner.cpu_cost_wins
                  << ",\"planner_gpu_cost_wins\":"
                  << planner.gpu_cost_wins
                  << ",\"worker_model_steps\":" << worker.model_steps
                  << ",\"worker_model_rows\":" << worker.model_rows
                  << ",\"worker_model_step_ns\":" << worker.model_step_ns
                  << ",\"worker_embed_rope_submit_ns\":"
                  << worker.embed_rope_submit_ns
                  << ",\"worker_scheduler_poll_ns\":"
                  << worker.scheduler_poll_ns
                  << ",\"worker_output_head_ns\":"
                  << worker.output_head_ns
                  << ",\"worker_attention_route_submit_ns\":"
                  << worker.attention_route_submit_ns
                  << ",\"worker_directory_plan_ns\":"
                  << worker.directory_plan_ns
                  << ",\"worker_ffn_submit_ns\":"
                  << worker.ffn_submit_ns
                  << ",\"worker_directory_release_ns\":"
                  << worker.directory_release_ns
                  << ",\"worker_gpu_attention_route_plan_ns\":"
                  << worker.gpu_attention_route_plan_ns
                  << ",\"worker_gpu_ffn_release_ns\":"
                  << worker.gpu_ffn_release_ns
                  << ",\"worker_gpu_attention_ns\":"
                  << worker.gpu_attention_ns
                  << ",\"worker_gpu_route_ns\":"
                  << worker.gpu_route_ns
                  << ",\"worker_gpu_directory_plan_ns\":"
                  << worker.gpu_directory_plan_ns
                  << ",\"worker_gpu_ffn_ns\":"
                  << worker.gpu_ffn_ns
                  << ",\"worker_gpu_directory_release_ns\":"
                  << worker.gpu_directory_release_ns
                  << ",\"worker_gpu_attention_hca_pre_norm_ns\":"
                  << worker.gpu_attention_hca_pre_norm_ns
                  << ",\"worker_gpu_attention_projection_ns\":"
                  << worker.gpu_attention_projection_ns
                  << ",\"worker_gpu_sparse_attention_ns\":"
                  << worker.gpu_sparse_attention_ns
                  << ",\"worker_gpu_attention_output_projection_ns\":"
                  << worker.gpu_attention_output_projection_ns
                  << ",\"worker_gpu_attention_hca_post_ns\":"
                  << worker.gpu_attention_hca_post_ns
                  << ",\"worker_gpu_ffn_routed_ns\":"
                  << worker.gpu_ffn_routed_ns
                  << ",\"worker_gpu_ffn_aggregate_ns\":"
                  << worker.gpu_ffn_aggregate_ns
                  << ",\"worker_gpu_ffn_shared_ns\":"
                  << worker.gpu_ffn_shared_ns
                  << ",\"worker_gpu_ffn_merge_ns\":"
                  << worker.gpu_ffn_merge_ns
                  << ",\"worker_gpu_ffn_hca_post_ns\":"
                  << worker.gpu_ffn_hca_post_ns
                  << ",\"worker_warm_start_candidates\":"
                  << worker.warm_start_candidates
                  << ",\"worker_warm_start_loaded\":"
                  << worker.warm_start_loaded
                  << ",\"worker_warm_start_bytes\":"
                  << worker.warm_start_bytes
                  << ",\"worker_warm_start_ns\":"
                  << worker.warm_start_ns
                  << ",\"worker_mtp_drafts\":" << worker.mtp_drafts
                  << ",\"worker_mtp_accepted\":" << worker.mtp_accepted
                  << ",\"worker_mtp_rejected\":" << worker.mtp_rejected
                  << ",\"worker_verify_pairs\":" << worker.verify_pairs
                  << ",\"worker_useful_tokens\":" << worker.useful_tokens
                  << ",\"worker_mtp_suppressions\":"
                  << worker.mtp_suppressions
                  << "}\n" << std::flush;
      } else if (fields[0] == "BEGIN") {
        require(fields.size() == 4U, "invalid BEGIN");
        const auto id = std::stoull(std::string(fields[1]));
        const auto context = std::stoull(std::string(fields[2]));
        require(id != 0U && !active.contains(id) &&
                    active.size() < model.capacity() &&
                    context <= model.max_context(),
                "invalid or over-capacity BEGIN");
        auto request = model.create_request();
        const auto prompt = parse_tokens(fields[3]);
        require(prompt.size() <= context, "prompt exceeds reserved context");
        request->context_limit = static_cast<std::uint32_t>(context);
        request->slot = free_slot(active, model.capacity());
        for (std::uint32_t position = 0U; position < prompt.size(); ++position) {
          Request* pointer = request.get();
          request->predicted = model.forward(
              std::span<Request* const>(&pointer, 1U),
              std::span<const std::uint32_t>(&prompt[position], 1U),
              std::span<const std::uint32_t>(&position, 1U)).front();
          if (model.mtp_enabled()) {
            const bool final_prompt = position + 1U == prompt.size();
            const auto next_token = final_prompt
                ? request->predicted : prompt[position + 1U];
            model.advance_mtp(*request, next_token,
                              request->state->current_streams(), position,
                              final_prompt);
          }
        }
        request->next_position = static_cast<std::uint32_t>(prompt.size());
        const auto slot = request->slot;
        active.emplace(id, Active{std::move(request)});
        std::cout << "{\"type\":\"begun\",\"id\":" << id
                  << ",\"slot\":" << slot << "}\n" << std::flush;
      } else if (fields[0] == "NEXT" || fields[0] == "STEP") {
        std::vector<std::pair<std::uint64_t, bool>> steps;
        std::set<std::uint64_t> unique_ids;
        const auto add = [&](std::string_view field) {
          const auto comma = field.find(',');
          require(comma != std::string_view::npos, "invalid step item");
          const auto id = std::stoull(std::string(field.substr(0U, comma)));
          const auto flag = field.substr(comma + 1U);
          require(active.contains(id) && unique_ids.insert(id).second &&
                      (flag == "0" || flag == "1"),
                  "step request mismatch");
          steps.emplace_back(id, flag == "1");
        };
        if (fields[0] == "NEXT") {
          require(fields.size() == 3U, "invalid NEXT");
          add(std::string(fields[1]) + "," + std::string(fields[2]));
        } else {
          require(fields.size() >= 2U &&
                      fields.size() <= model.capacity() + 1U,
                  "invalid STEP");
          for (std::size_t index = 1U; index < fields.size(); ++index)
            add(fields[index]);
        }
        std::vector<std::vector<std::uint32_t>> emitted(steps.size());
        if (model.mtp_enabled()) {
          for (std::size_t index = 0U; index < steps.size(); ++index) {
            const auto [id, final] = steps[index];
            auto& request = *active.at(id).request;
            if (final) {
              emitted[index] = {request.predicted};
              continue;
            }
            if (model.speculation_active(request)) {
              emitted[index] = model.verify_draft(request);
            } else {
              emitted[index] = {request.predicted};
              Request* pointer = &request;
              const auto token = request.predicted;
              const auto position = request.next_position;
              request.predicted = model.forward(
                  std::span<Request* const>(&pointer, 1U),
                  std::span<const std::uint32_t>(&token, 1U),
                  std::span<const std::uint32_t>(&position, 1U)).front();
              ++request.next_position;
            }
          }
        } else {
          std::vector<Request*> advancing;
          std::vector<std::uint32_t> tokens, positions;
          for (std::size_t index = 0U; index < steps.size(); ++index) {
            const auto [id, final] = steps[index];
            auto& request = *active.at(id).request;
            emitted[index] = {request.predicted};
            if (final) continue;
            advancing.push_back(&request);
            tokens.push_back(request.predicted);
            positions.push_back(request.next_position);
          }
          if (!advancing.empty()) {
            const auto predicted = model.forward(advancing, tokens, positions);
            std::size_t predicted_index = 0U;
            for (const auto& [id, final] : steps) {
              if (final) continue;
              active.at(id).request->predicted =
                  predicted[predicted_index++];
              ++active.at(id).request->next_position;
            }
          }
        }
        if (fields[0] == "NEXT") {
          std::cout << "{\"type\":\"token\",\"id\":" << steps[0].first
                    << ",\"tokens\":[";
          for (std::size_t token = 0U; token < emitted[0].size(); ++token) {
            if (token) std::cout << ',';
            std::cout << emitted[0][token];
          }
          std::cout << "]}\n";
        } else {
          std::cout << "{\"type\":\"batch\",\"items\":[";
          for (std::size_t index = 0U; index < steps.size(); ++index) {
            if (index) std::cout << ',';
            const auto id = steps[index].first;
            std::cout << "{\"id\":" << id << ",\"tokens\":[";
            for (std::size_t token = 0U; token < emitted[index].size();
                 ++token) {
              if (token) std::cout << ',';
              std::cout << emitted[index][token];
            }
            std::cout << "]}";
          }
          std::cout << "]}\n";
        }
        for (const auto& tokens : emitted)
          model.record_useful_tokens(tokens.size());
        std::cout << std::flush;
        for (const auto& [id, final] : steps)
          if (final) active.erase(id);
      } else if (fields[0] == "END") {
        require(fields.size() == 2U, "invalid END");
        const auto id = std::stoull(std::string(fields[1]));
        require(active.erase(id) == 1U, "END request mismatch");
        std::cout << "{\"type\":\"ended\",\"id\":" << id << "}\n"
                  << std::flush;
      } else if (fields[0] == "SHUTDOWN") {
        require(fields.size() == 1U && active.empty(), "invalid SHUTDOWN");
        std::cout << "{\"type\":\"shutdown\"}\n" << std::flush;
        return 0;
      } else {
        throw std::runtime_error("unknown worker command");
      }
    } catch (const std::exception& error) {
      std::cerr << "worker command failed: " << error.what() << '\n';
      std::cout << "{\"type\":\"error\",\"active_requests\":"
                << active.size() << "}\n" << std::flush;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 10 || argc > 12 ||
        std::string_view(argv[2]) != "--worker") {
      std::cerr << "usage: expert-deepseek-worker <bundle> --worker "
                   "<max-context> <ram-gib> <vram-gib> <capacity> "
                   "<kv-cache-mib> <kv-page-tokens> <placement-profile> "
                   "[--profile-gpu-phases] [--enable-mtp]\n";
      return 64;
    }
    bool profile_gpu_phases = false;
    bool enable_mtp = false;
    for (int index = 10; index < argc; ++index) {
      const std::string_view flag = argv[index];
      require(flag == "--profile-gpu-phases" || flag == "--enable-mtp",
              "unknown DeepSeek worker option");
      if (flag == "--profile-gpu-phases") {
        require(!profile_gpu_phases, "duplicate GPU profiling option");
        profile_gpu_phases = true;
      } else {
        require(!enable_mtp, "duplicate MTP option");
        enable_mtp = true;
      }
    }
    const auto max_context = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const auto ram_gib = std::stoull(argv[4]);
    const auto vram_gib = std::stoull(argv[5]);
    const auto capacity = static_cast<std::uint32_t>(std::stoul(argv[6]));
    const auto kv_cache_mib = std::stoull(argv[7]);
    const auto kv_page_tokens =
        static_cast<std::uint32_t>(std::stoul(argv[8]));
    const std::string placement = argv[9];
    require(placement == "latency" || placement == "balanced" ||
                placement == "capacity",
            "invalid placement profile");
    Model model(argv[1], max_context, ram_gib << 30U, vram_gib << 30U,
                capacity, kv_cache_mib << 20U, kv_page_tokens, placement,
                profile_gpu_phases, enable_mtp);
    return worker_loop(model);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
