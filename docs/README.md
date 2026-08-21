# Documentation map

Status: synchronized with the Qwen3.8 production-pilot state on 2026-08-19.

Start with [MoE VM current state and remaining work](moe-vm-next.md). It is the
canonical handoff for the next implementation session. Historical plans retain
measurements and decisions, but they are not active backlogs.

## Current contracts and operation

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | Current system boundaries, including the implemented and missing parts of the MoE VM |
| [Getting started](getting-started.md) | Build, Xet download, FP4 compilation, validation and foreground startup |
| [Deployment](deployment.md) | POSIX control host to Windows/CUDA lifecycle through `ops/model.sh` |
| [Operations](operations.md) | Short day-two runbook |
| [OpenAI-compatible API](openai-api.md) | Supported HTTP/SSE contract and explicit omissions |
| [Anthropic Messages API](anthropic-api.md) | Claude Code adapter, tool blocks, SSE and private-LAN configuration |
| [WSL2 Unsloth Qwen3.8](wsl2-unsloth-qwen38.md) | External production path for Qwen3.8 GGUF, llama.cpp and Claude Code on the RTX 3090 host |
| [Production readiness](production-readiness.md) | Pilot verdict, qualification matrix and release blockers |
| [Heterogeneous organ placement](heterogeneous-placement-research.md) | Research contract for executing model/state shards directly from VRAM/GPU and RAM/CPU |
| [Exact tiered tree verification](exact-tiered-tree-verification.md) | Active 262K exact-F16 decode research, equations and fail-fast gates for the 15 tok/s target |

## Artifact and runtime contracts

| Document | Purpose |
|---|---|
| [Expert Pack v1](expert-pack-v1.md) | FP4 placement container and record ABI |
| [DeepSeek compact pack v1](deepseek-compact-pack-v1.md) | Current DeepSeek routed-weight storage representation |
| [Compute-ready representations](compute-ready.md) | Distinction between source, placement, compute and device layouts |
| [Expert Runtime v1](expert-runtime.md) | Cache, placement, execution, worker protocol and telemetry invariants |

The current service has one launcher, one scheduled task and one binary:
`ops/model.sh`, `QuantumLLM-ExpertVm` and `expert-moe-vm-runner.exe`. This does
not yet mean one physical container or one composable numeric executor.
Expert Pack v1 and DeepSeek worker bundle v3 still enter through separate
storage adapters, and the common binary currently selects one complete worker
provider for the artifact's operation set. Qwen3.8 is the active dense FP4
target; DeepSeek remains the sparse demand-paged compatibility target.

## Evidence and historical records

| Document | Status |
|---|---|
| [Performance evidence](benchmarks.md) | Measured results; current functional VM acceptance plus historical performance campaigns |
| [Heterogeneous placement benchmark](heterogeneous-placement-benchmark.md) | Qwen3.8 manifest inventory, measured CUDA inputs and placement bounds |
| [Engineering history](history.md) | Completed, rejected and extracted work |
| [30 tok/s plan](inference-30toks-plan.md) | Executed historical campaign, not the active plan |
| [Scaling follow-on](inference-scaling-next.md) | Historical FP4/performance research; partially executed |
| [Performance architecture review](performance-architecture-review.md) | Historical diagnosis |
| [Neural CPU experiment](neural-cpu-experiment.md) | Completed negative research experiment; not active runtime work |
| [Roadmap](roadmap.md) | Compatibility pointer to the canonical handoff |

The external Memory Expert/KV-attach project is no longer maintained in this
repository. Its current architecture, results and runbooks live in
`../memory-expert`.
