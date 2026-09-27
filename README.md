# Quantum LLM

> **Qwen3.8-27B at 62.67 tok/s on one RTX 3090.**
>
> **9.30× the decode rate of pinned llama.cpp on the same 2,239-token Pi
> opening call.**
>
> **262,016 populated tokens completed through the exact-F16 path on the same
> GPU.**

Quantum LLM runs validated dense, hybrid and sparse-MoE artifacts through one
artifact-driven Windows/CUDA runtime on a single RTX 3090. It exposes OpenAI-
and Anthropic-compatible APIs and makes VRAM, RAM and NVMe placement visible
instead of hiding it behind an opaque model server.

The open product target is demanding by design: Qwen3.8-27B with 262,144
actually populated positions, exact IEEE F16 KV semantics, about 15 useful
output tokens/s and a real `xhigh` coding harness.

## Same call, same GPU

The first call of the pinned 256K Pi benchmark sent the same scripted `hi` and
the same 2,239-token prompt to both backends on the same RTX 3090. Quantum used
standard FP4 weights, Q4H target KV and MTP-4. llama.cpp v0.5.0 used the
official Q4_K_M GGUF, `q4_0` KV and its MTP-4 draft.

| Observed result | Quantum LLM | llama.cpp | Ratio |
|---|---:|---:|---:|
| Client TTFT | **2.813 s** | 6.281 s | **llama.cpp took 2.23× longer** |
| Post-first decode | **62.67 tok/s** | 6.74 tok/s | **Quantum reached 9.30× the rate** |
| Complete call wall | **3.547 s** | 11.765 s | **llama.cpp took 3.32× longer** |
| Generated output | 46 tokens | 37 tokens | — |

llama.cpp remained between 6.67 and 8.39 tok/s over its three completed calls
and was stopped because the early rate did not justify continuing. Quantum
completed 25 calls, reached a 58,664-token prompt and averaged 69.915 generated
tok/s across completed decode before a later idle-GPU stall.

The formats and generated outputs differ, so this is an observed comparison of
the same opening workload rather than a claim of numerical equivalence or a
completed 256K head-to-head. Raw conditions and the later failure are recorded
in [Benchmarks](docs/benchmarks.md#pinned-llamacpp-comparison).

## More measured results

These are live service results unless the row says otherwise. Decode rate is
post-first-token, so TTFT remains visible instead of being averaged away.

| Result | Path and boundary |
|---|---|
| **54.54 tok/s at 32,768 prompt tokens** | Qwen Q4H target KV + Q8 MTP-4; 39.627 s TTFT; 128 generated tokens |
| **47.16 tok/s at 131,072 prompt tokens** | Same live path; 203.250 s TTFT; 128 generated tokens |
| **60.58 tok/s through the OpenAI endpoint** | GuideLLM, 32,820 prompt tokens, 128 reasoning tokens; 39.861 s TTFT and 3.05 tok/s end to end |
| **262,016 populated tokens with exact F16 KV** | Retrieved facts planted near the start, middle and end; 47 generated tokens in 2,278.082 s |
| **4.566 GiB calculated context state at 262,144** | Q4H target KV plus Q8 MTP pages, compared with 16 GiB for exact F16 target KV alone; Q4H is lossy |
| **Six model artifacts through one service** | Each published model passed an isolated tokenizer, template and generation smoke |

The 262,016-token run proves capacity through the exact-F16 path, but its latency
misses the product target. The operational Q4H path is fast at measured 32K
and 128K prompts; a live 262K Q4H coding-quality result still does not exist.
The complete prompts, sampling settings, counters and negative results are in
[Benchmarks](docs/benchmarks.md).

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

Release blockers are tracked in
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
