# Documentation

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | System boundaries, data flow, placement, and failure model |
| [Getting started](getting-started.md) | Requirements, build, model conversion, and first request |
| [Expert Pack v1](expert-pack-v1.md) | On-disk format and quantization ABI |
| [Runtime contract](expert-runtime.md) | Cache, scheduler, storage, CUDA, and request invariants |
| [OpenAI-compatible API](openai-api.md) | Endpoints, SDK examples, parameters, and errors |
| [Operations](operations.md) | Deploy, stop, observe, rollback, and recover |
| [Benchmarks](benchmarks.md) | Reproducible evidence and interpretation |
| [Production readiness](production-readiness.md) | Readiness matrix, known gaps, and release gates |
| [Roadmap](roadmap.md) | Long-context, performance, and distributed execution phases |
| [Hybrid execution plan](hybrid-execution-plan.md) | Staged CPU/GPU scheduler implementation and acceptance gates |
| [Larger-model decision](larger-model-decision.md) | Modern MoE candidates, compatibility gaps, workspace rules, and target selection |

The JSON schemas under `schemas/` and the compiler validator are normative for
Expert Pack. Documentation never overrides executable validation.
