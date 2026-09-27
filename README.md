# Quantum LLM

Quantum LLM is a Windows/CUDA inference runtime for validated model artifacts
on one self-contained RTX 3090 host. It exposes OpenAI- and
Anthropic-compatible HTTP APIs and gives explicit control over model state in
VRAM, host RAM and NVMe.

The runtime is research/pilot software. Its strongest result is controlled,
observable execution of large dense, hybrid and sparse-MoE artifacts. The
product target remains open: Qwen3.8-27B at 262,144 populated positions, exact
IEEE F16 KV, about 15 useful output tokens/s and a real `xhigh` coding harness.

## Current boundary

The deployment publishes six in-scope models through one service and one
artifact-driven VM:

- Qwen3.8-27B and a separately published third-party Abliterated checkpoint;
- Qwen3.8-Flash-Next;
- Mistral Small 4;
- Muse-Glimmer-30B;
- Ornith-1.5-35B-A3B.

`qwen` is the operational Qwen alias. It uses standard FP4 weights, lossy
`q4-f16-per-head` target KV and artifact-declared MTP-4. `qwen-f16` selects the
same Qwen artifact with exact F16 target KV. The lossy default has passed live
32K and 128K throughput gates; it has not passed a live 262K coding-quality
gate and does not satisfy the exact-F16 product target.

The latest complete evidence and its limits are in
[Benchmarks](docs/benchmarks.md). Release blockers are in
[Production readiness](docs/production-readiness.md).

## System

```text
official checkpoint
  -> strict source adapter
  -> authenticated artifact (manifest, QPack, runtime-model.tsv)
  -> common Python HTTP service
  -> common native VM
  -> capability-matched CUDA provider
  -> VRAM / RAM / NVMe placement
```

The artifact declares tensor roles, operations, geometry, encodings, sampling,
response parsing and provider capabilities. Common service and VM code select
execution from those declarations. Source adapters are the only layer allowed
to know upstream tensor names.

Dense weights, recurrent state and active attention state are hot. Moving
them through RAM or NVMe can recover capacity but cannot provide interactive
decode. Unselected sparse-MoE experts are immutable logical pages and may be
cached or loaded on demand without changing router top-k semantics.

See [Architecture](docs/architecture.md) for the implemented boundaries.

## Quick start

The control machine is POSIX. The execution host is Windows with the RTX 3090.

```bash
cp .env.example .env
# Set QUANTUM_LLM_REMOTE, QUANTUM_LLM_REMOTE_ROOT, MODEL_ROOT and EXPERT_API_KEY.

./ops/model.sh sync
./ops/model.sh install
./ops/model.sh start qwen
./ops/model.sh chat qwen
./ops/pi.sh qwen
./ops/model.sh stop all
```

`MODEL_ROOT` is the only host-dependent model storage root. Model weights are
not included in this repository. Follow [Getting started](docs/getting-started.md)
before compiling or publishing an artifact.

## Repository map

| Path | Responsibility |
|---|---|
| `compiler/` | strict source adapters, QPack compilation and validation |
| `core/` | feasibility and shared artifact contracts |
| `runtime/` | native VM, providers, state, paging and CUDA kernels |
| `ops/` | build, download, publication, service and benchmark workflows |
| `tests/` | compiler, service, benchmark and native contract tests |
| `docs/` | current architecture, interfaces, evidence and release state |

Generated state belongs in `out/`, `work/`, `logs/`, `artifacts/` or below
`MODEL_ROOT`. Source code is licensed under [Apache License 2.0](LICENSE).
