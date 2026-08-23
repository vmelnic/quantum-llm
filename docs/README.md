# Documentation

Status: authoritative documentation index, 2026-08-23.

The repository documents the implementation that exists today, the evidence
that has actually been measured, and the next accepted work. Detailed journals
for abandoned experiments were removed; their durable conclusions are
consolidated in [Research decisions](research-decisions.md), while the original
material remains recoverable from Git history.

## Start here

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | System boundaries, artifact VM, Qwen dense execution, DeepSeek expert paging, KV and session state |
| [Production readiness](production-readiness.md) | What is usable, what is not, and the release blockers |
| [Benchmarks](benchmarks.md) | Canonical measurements and capacity/bandwidth equations |
| [Roadmap](roadmap.md) | Dependency-ordered work that advances the active goals |
| [Research decisions](research-decisions.md) | Rejected mechanisms and current heterogeneous multi-GPU direction |

## Build and operate

| Document | Purpose |
|---|---|
| [Getting started](getting-started.md) | Host prerequisites, build, artifact preparation and first service run |
| [Deployment](deployment.md) | Stable artifact publication and the supported lifecycle |
| [Operations](operations.md) | Health, telemetry, diagnosis, rollback and cleanup |
| [Pi CLI](pi-cli.md) | Local coding-agent configuration, thinking control and session behavior |

## Interfaces and formats

| Document | Purpose |
|---|---|
| [Expert Runtime contract](expert-runtime.md) | Program/provider ABI, worker protocol, placement and request lifecycle |
| [Expert Pack v1](expert-pack-v1.md) | QPack records, manifest, FP4 encoding and publication rules |
| [DeepSeek compact pack v1](deepseek-compact-pack-v1.md) | Authenticated routed-expert storage and bundle layout |
| [OpenAI-compatible API](openai-api.md) | Models, Completions, Chat Completions and Responses |
| [Anthropic Messages API](anthropic-api.md) | Claude-compatible wire adapter and its explicit limitations |

## Claims policy

- A configured context limit is not a populated-context result.
- A short `hi` smoke proves lifecycle and basic generation, not quality or
  maximum-context performance.
- Post-first-token, end-to-end and aggregate rates are different metrics.
- Qwen FP4 storage, exact F16 KV, DeepSeek FP4 routed experts, FP8 shared
  tensors and FP32 runtime state are always reported separately.
- A theoretical hardware ceiling is not a service result.
- Removed research code is not a hidden roadmap; only
  [Roadmap](roadmap.md) is active.
