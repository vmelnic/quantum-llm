# Documentation map

Status: synchronized with the repository state on 2026-08-11.

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
| [Production readiness](production-readiness.md) | Pilot verdict, qualification matrix and release blockers |

## Artifact and runtime contracts

| Document | Purpose |
|---|---|
| [Expert Pack v1](expert-pack-v1.md) | Qwen/LFM placement container and record ABI |
| [DeepSeek compact pack v1](deepseek-compact-pack-v1.md) | Current DeepSeek routed-weight storage representation |
| [Compute-ready representations](compute-ready.md) | Distinction between source, placement, compute and device layouts |
| [Expert Runtime v1](expert-runtime.md) | Cache, placement, execution, worker protocol and telemetry invariants |

The current service has one launcher, one scheduled task and one binary:
`ops/model.sh`, `QuantumLLM-ExpertVm` and `expert-moe-vm-runner.exe`. This does
not yet mean one physical container or one composable numeric executor.
Expert Pack v1 and DeepSeek worker bundle v3 still enter through separate
storage adapters, and the common binary currently selects one complete worker
provider for the artifact's operation set.

## Evidence and historical records

| Document | Status |
|---|---|
| [Performance evidence](benchmarks.md) | Measured results; current functional VM acceptance plus historical performance campaigns |
| [Engineering history](history.md) | Completed, rejected and extracted work |
| [30 tok/s plan](inference-30toks-plan.md) | Executed historical campaign, not the active plan |
| [Scaling follow-on](inference-scaling-next.md) | Historical FP4/performance research; partially executed |
| [Performance architecture review](performance-architecture-review.md) | Historical diagnosis |
| [Neural CPU experiment](neural-cpu-experiment.md) | Completed negative research experiment; not active runtime work |
| [Roadmap](roadmap.md) | Compatibility pointer to the canonical handoff |

The external Memory Expert/KV-attach project is no longer maintained in this
repository. Its current architecture, results and runbooks live in
`../memory-expert`.
