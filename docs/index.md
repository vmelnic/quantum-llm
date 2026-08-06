# Documentation

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | System boundaries, data flow, placement, and failure model |
| [Getting started](getting-started.md) | Requirements, build, model conversion, and first request |
| [Expert Pack v1](expert-pack-v1.md) | On-disk format and quantization ABI |
| [Compute-ready representations](compute-ready.md) | Source, placement, kernel ABI, and residency distinctions |
| [Runtime contract](expert-runtime.md) | Cache, scheduler, storage, CUDA, and request invariants |
| [OpenAI-compatible API](openai-api.md) | Endpoints, SDK examples, parameters, and errors |
| [Operations](operations.md) | Deploy, stop, observe, rollback, and recover |
| [Benchmarks](benchmarks.md) | Reproducible evidence and interpretation |
| [Production readiness](production-readiness.md) | Readiness matrix, known gaps, and release gates |
| [Roadmap](roadmap.md) | Long-context, performance, and distributed execution phases |
| [Hybrid execution plan](hybrid-execution-plan.md) | Staged CPU/GPU scheduler implementation and acceptance gates |
| [Larger-model decision](larger-model-decision.md) | Modern MoE candidates, compatibility gaps, workspace rules, and target selection |
| [DeepSeek-V4-Flash backend](deepseek-v4-backend.md) | Pinned source contract, quantization boundary, and SM86 placement direction |
| [DeepSeek compact pack v1](deepseek-compact-pack-v1.md) | Aligned, resumable native FP4 routed-expert container |
| [DeepSeek-V4-Flash serving](deepseek-serving.md) | Worker bundle, persistent CUDA process, API startup, and service gate |
| [Route census](route-census.md) | Bounded, authenticated route evidence and deterministic warm-set ranking |
| [Measured placement profile](placement-profile.md) | Hardware cost calibration and fail-closed RAM/VRAM budget solver |
| [DeepSeek performance tasks](deepseek-performance-tasks.md) | Ordered work and acceptance gates for the GPU-driven performance path |

The JSON schemas under `schemas/` and the compiler validator are normative for
Expert Pack. Documentation never overrides executable validation.
