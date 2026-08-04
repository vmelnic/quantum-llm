# Expert Runtime

Expert Runtime is an experimental Windows/CUDA inference engine for Mixture of
Experts models whose checkpoints do not fit in GPU memory—and may not fit in
system RAM. It keeps dense weights on the GPU and places independently packed
experts across VRAM, RAM, and SSD.

The current vertical slice runs `Qwen/Qwen3-Next-80B-A3B-Instruct` on a single
RTX 3090. The original checkpoint is about 162.7 GB; its deterministic Expert
Pack v1 INT8 container is about 81.9 GB. The tested host has 24 GB VRAM and
approximately 64 GB RAM.

> **Project status:** research-quality runtime with a controlled pre-production
> pilot profile. It is not a general production inference server yet. The
> certified context window is 4096 tokens, despite the model architecture
> advertising 262K. See [Production readiness](docs/production-readiness.md).

## Why this exists

Conventional runtimes usually assume that all weights fit in GPU memory, unified
memory, or RAM. Sparse MoE models activate only a fraction of their experts for
each token. Expert Runtime exploits that sparsity:

```text
SafeTensors checkpoint
        │ streaming compiler
        ▼
Expert Pack v1 ── dense.qpack ───────────────► persistent GPU residency
        │
        └── experts-*.qpack ─► SSD ─► RAM ─► VRAM cache ─► grouped execution
                                                   │
OpenAI-compatible API ─► router ─► active experts ─┴─► stable aggregation
```

The runtime validates every manifest, pack, record ABI, size, and checksum
before readiness. Missing experts fail the request; they are never silently
dropped.

## What works

- deterministic SafeTensors → Expert Pack v1 conversion;
- strict adapters for OLMoE and Qwen3-Next;
- Windows IOCP storage, bounded RAM/VRAM caches, pinned staging, and CUDA SM86;
- exact Qwen3-Next attention/DeltaNet/MoE inference;
- paged FP16 KV with per-request credits and constant-memory online attention;
- continuous decode batching with isolated request state;
- Responses, Chat Completions, and legacy Completions HTTP APIs;
- SSE streaming, usage, cancellation, overload handling, health and metrics;
- reproducible Windows build, correctness gates, and Task Scheduler deployment.

## Quick start

Read [Getting started](docs/getting-started.md) before downloading the model.
The short path on the Windows GPU host is:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\server.txt

.\ops\windows\Invoke-BuildExpertRuntime.ps1 -Configuration Release
.\ops\windows\Invoke-ExpertPack.ps1 -Action Validate `
  -Path .\work\models\qwen3-next-80b-expert-pack-int8

.\ops\windows\Start-P6ExpertServer.ps1 -BuildId (git rev-parse --short HEAD)
```

Then, from another terminal:

```powershell
curl.exe http://127.0.0.1:8080/v1/responses `
  -H "Content-Type: application/json" `
  -d '{"model":"qwen3-next-80b-a3b-expert-pack-int8","input":"Hello","max_output_tokens":32,"temperature":0}'
```

## Documentation

- [Documentation index](docs/index.md)
- [Architecture](docs/architecture.md)
- [Getting started](docs/getting-started.md)
- [Expert Pack v1](docs/expert-pack-v1.md)
- [Runtime contract](docs/expert-runtime.md)
- [OpenAI-compatible API](docs/openai-api.md)
- [Operations](docs/operations.md)
- [Benchmarks and evidence](docs/benchmarks.md)
- [Production readiness](docs/production-readiness.md)
- [Roadmap](docs/roadmap.md)
- [Changelog](CHANGELOG.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

## Scope boundaries

This project is not llama.cpp, Colibri, or a wrapper around either. It uses a
compute-ready expert format and a GPU-driven heterogeneous placement runtime.
Metal and distributed expert workers are design targets, not current backends.

Model weights are not included. Users are responsible for the model's license
and for complying with its terms.

## License

The source code is licensed under the [Apache License 2.0](LICENSE).
