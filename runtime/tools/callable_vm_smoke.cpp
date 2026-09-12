#include "expert/runtime/execution_provider.hpp"
#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/program_executor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

expert::runtime::CreateExecutionProviderModuleResult
make_sm86_dense_moe_callable_provider(
    const std::filesystem::path& artifact_root, std::uint32_t max_context,
    std::uint32_t capacity, std::uint64_t ram_cache_bytes,
    std::uint64_t vram_cache_bytes, std::uint32_t kv_page_tokens);
expert::runtime::CreateExecutionProviderModuleResult
make_sm86_hybrid_delta_moe_callable_provider(
    const std::filesystem::path& artifact_root, std::uint32_t max_context,
    std::uint32_t capacity, std::uint64_t ram_cache_bytes,
    std::uint64_t vram_cache_bytes, std::uint64_t kv_cache_bytes,
    std::uint32_t kv_page_tokens, std::string_view placement_profile,
    bool discover_active_expert_devices,
    std::vector<int> active_expert_devices,
    std::uint64_t active_expert_device_cache_bytes,
    std::uint64_t active_expert_host_cache_bytes);

namespace {

expert::runtime::ExecutionValue host_u32(std::uint32_t value,
                                         std::string abi) {
  auto owner = std::make_shared<std::uint32_t>(value);
  return {std::move(abi), "host", owner,
          reinterpret_cast<const std::byte*>(owner.get()), sizeof(value)};
}

std::uint32_t read_u32(const expert::runtime::ExecutionValue& value) {
  if (!value.valid() || value.bytes != sizeof(std::uint32_t))
    throw std::runtime_error("callable VM returned an invalid token value");
  std::uint32_t result{};
  std::memcpy(&result, value.data, sizeof(result));
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      std::cerr << "usage: expert-callable-vm-smoke <artifact>\n";
      return 64;
    }
    const std::filesystem::path root(argv[1]);
    expert::runtime::ModelArtifact artifact;
    auto status = expert::runtime::ModelArtifact::load(root, artifact);
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    if (artifact.model().schema_version < 3U)
      throw std::runtime_error("callable smoke requires a schema v3 artifact");

    auto created = make_sm86_dense_moe_callable_provider(
        root, 4096U, 1U, 4ULL << 30U, 13ULL << 30U, 256U);
    std::string dense_error(created.status.message());
    if (!created.status.ok()) {
      created = make_sm86_hybrid_delta_moe_callable_provider(
          root, 4096U, 1U, 4ULL << 30U, 13ULL << 30U, 2ULL << 30U,
          256U, "balanced", false, {}, 0U, 0U);
    }
    if (!created.status.ok())
      throw std::runtime_error(
          "no callable operation provider accepted the artifact; "
          "dense=" + dense_error + "; hybrid-delta=" +
          std::string(created.status.message()));
    expert::runtime::ExecutionProviderRegistry registry;
    status = registry.add(std::move(created.module.definition));
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));
    auto bound = registry.bind(
        artifact.model(),
        expert::runtime::ExecutionProviderBindingMode::executable);
    if (!bound.status.ok())
      throw std::runtime_error(std::string(bound.status.message()));
    expert::runtime::MoeProgramExecutor executor;
    status = expert::runtime::MoeProgramExecutor::create(
        artifact.model(), std::move(bound.provider),
        created.module.tensor_store.get(), executor);
    if (!status.ok()) throw std::runtime_error(std::string(status.message()));

    expert::runtime::ProgramRequestContext request;
    request.request_id = 1U;
    request.deadline =
        std::chrono::steady_clock::now() + std::chrono::minutes(2);
    request.parameters.emplace("reserved_context_tokens", 4096U);
    auto begun = executor.begin_session(std::move(request));
    if (!begun.status.ok())
      throw std::runtime_error(std::string(begun.status.message()));
    std::map<std::string, expert::runtime::ExecutionValue, std::less<>> inputs;
    inputs.emplace("token_ids", host_u32(
        1U, "batch.token-id.u32.host.v1"));
    inputs.emplace("positions", host_u32(
        0U, "batch.position.u32.host.v1"));
    auto step = begun.session.execute(std::move(inputs));
    if (!step.status.ok())
      throw std::runtime_error(std::string(step.status.message()));
    std::optional<expert::runtime::ProgramExecutionResult> result;
    while (!(result = step.handle.poll()))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!result->status.ok())
      throw std::runtime_error(std::string(result->status.message()));
    const auto output = result->outputs.find("next_token_ids");
    if (output == result->outputs.end())
      throw std::runtime_error("callable VM did not publish next_token_ids");
    std::cout << "{\"schema\":\"callable-vm-smoke-v1\",\"token\":"
              << read_u32(output->second) << "}\n";
    begun.session.cancel();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "callable VM smoke: " << error.what() << '\n';
    return 1;
  }
}
