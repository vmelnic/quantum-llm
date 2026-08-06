# Expert Runtime

Expert Runtime is an experimental Windows/CUDA inference engine for Mixture of
Experts models whose checkpoints do not fit in GPU memory—and may not fit in
system RAM. It keeps dense weights on the GPU and places independently packed
experts across VRAM, RAM, and SSD.

Two native backends run on a single RTX 3090: Qwen3-Next 80B from an 81.9 GB
INT8 Expert Pack and DeepSeek-V4-Flash from its authenticated mixed-FP4/FP8
compact bundle. Both expose the same local HTTP API and lifecycle commands.

> **Project status:** research-quality runtime with a controlled pre-production
> pilot profile. It is not a general production inference server yet. The
> reference deployment accepts up to 65,536 context tokens and 8,192 output
> tokens, but only 4,096 context tokens have completed the historical
> long-context qualification gates. See
> [Production readiness](docs/production-readiness.md).

## Why this exists

Conventional runtimes usually assume that all weights fit in GPU memory, unified
memory, or RAM. Sparse MoE models activate only a fraction of their experts for
each token. Expert Runtime exploits that sparsity:

```text
SafeTensors checkpoint
        │ architecture-specific compiler
        ▼
Qwen Expert Pack / DeepSeek compact bundle
        ├── dense + shared state ───────────────► persistent GPU residency
        └── independently indexed experts ─► SSD/RAM/VRAM placement
                                                       │
OpenAI-compatible API ─► exact router ─► active experts ─┴─► stable aggregation
```

The runtime validates every manifest, pack, record ABI, size, and checksum
before readiness. Missing experts fail the request; they are never silently
dropped.

## What works

- deterministic Qwen Expert Pack and DeepSeek compact-bundle conversion;
- strict adapters for OLMoE, Qwen3-Next, and DeepSeek-V4-Flash;
- Windows IOCP storage, bounded RAM/VRAM caches, pinned staging, and CUDA SM86;
- native Qwen3-Next and DeepSeek-V4-Flash greedy inference;
- paged FP16 KV with per-request credits and constant-memory online attention;
- continuous decode batching with isolated request state;
- Responses, Chat Completions, and legacy Completions HTTP APIs;
- SSE streaming, usage, cancellation, overload handling, health and metrics;
- reproducible Windows build, correctness gates, and Task Scheduler deployment.

## Quick start

Read [Install, configure and use](docs/deployment.md) before downloading model
artifacts. After the Windows/CUDA runtime and at least one model are prepared,
copy the public configuration template on the POSIX control host:

```bash
cp .env.example .env
# Set the remote host and model artifact paths in .env.
./ops/model.sh install
./ops/model.sh start              # CHAT_MODEL from .env
./ops/model.sh chat
```

Switch models or release all GPU memory with the same command:

```bash
./ops/model.sh start qwen
./ops/model.sh start deepseek
./ops/model.sh stop all
```

## Documentation

- [Architecture](docs/architecture.md)
- [Getting started](docs/getting-started.md)
- [Install, configure and use](docs/deployment.md)
- [Expert Pack v1](docs/expert-pack-v1.md)
- [DeepSeek compact pack v1](docs/deepseek-compact-pack-v1.md)
- [Compute-ready representations](docs/compute-ready.md)
- [Runtime contract](docs/expert-runtime.md)
- [OpenAI-compatible API](docs/openai-api.md)
- [Operations](docs/operations.md)
- [Benchmarks and evidence](docs/benchmarks.md)
- [Performance architecture review](docs/performance-architecture-review.md)
- [Engineering history](docs/history.md)
- [Production readiness](docs/production-readiness.md)
- [Roadmap](docs/roadmap.md)

## Scope boundaries

This project is not llama.cpp, Colibri, or a wrapper around either. It uses a
compute-ready expert format and a GPU-driven heterogeneous placement runtime.
Metal and distributed expert workers are design targets, not current backends.

Model weights are not included. Users are responsible for the model's license
and for complying with its terms.

## License

The source code is licensed under the [Apache License 2.0](LICENSE).
