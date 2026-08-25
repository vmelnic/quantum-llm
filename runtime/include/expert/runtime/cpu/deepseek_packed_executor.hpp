#pragma once

#include "expert/runtime/cpu/fp4_host_executor.hpp"

namespace expert::runtime::cpu {

// Source-compatible aliases for the existing DeepSeek provider. New common
// providers use the capability-owned FP4 host ABI above.
using DeepSeekPackedWorkGroup = Fp4HostWorkGroup;
using DeepSeekPackedExecutorConfig = Fp4HostExecutorConfig;
using DeepSeekPackedExecutorTelemetry = Fp4HostExecutorTelemetry;
using DeepSeekPackedExecutor = Fp4HostExecutor;

}  // namespace expert::runtime::cpu
