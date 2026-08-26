# Quantum LLM

Quantum LLM is a native Windows/CUDA research runtime for exact inference when
one model's useful state must be placed across GPU memory, system memory and
NVMe. It is built for a self-contained RTX 3090 host and currently executes
four models through one artifact-driven VM:

- **Qwen3.8-27B FP4:** dense/hybrid weights in VRAM, recurrent state on the
  GPU and progressively allocated exact F16 long-context KV in host RAM;
- **Muse-Glimmer-30B FP4:** dense text weights in VRAM with exact F16 global
  KV plus model-declared exact sliding-window KV rings;
- **Ornith-1.5-35B-A3B FP4:** dense/recurrent organs plus exact top-8 standard
  FP4 expert pages split dynamically between its host bank and routed VRAM;
- **DeepSeek-V4-Flash:** resident dense/shared organs plus exact top-6 routed
  experts demand-paged through NVMe → RAM → VRAM.

The project exists to control and measure *where each model organ lives and
where its operation executes*. It does not pretend that generic weight
offload makes a dense model fast, and it never reduces top-k or silently swaps
exact F16 KV for a smaller representation.

> **Current status:** functional research/pilot software, not yet
> production-ready for the 262K coding target. Qwen short-context service
> works, but an exact-F16 262,001-token prefill measured 2,183.815 seconds.
> DeepSeek works beyond RAM+VRAM through exact expert paging, but its latest
> short `hi` gate measured 0.89 end-to-end tok/s on the reference host. See
> [Benchmarks](docs/benchmarks.md) and
> [Production readiness](docs/production-readiness.md).

## Why this architecture exists

An LLM is not one uniform byte array. Its organs have different access
patterns:

```text
                              every target call
 dense matrices / recurrent state ───────────────────────────► GPU

 Qwen full-attention KV       request-owned, every position ─► RAM today
                                  bounded staging + compute ─► GPU

 DeepSeek router/shared state every routed layer ────────────► GPU
 routed expert pages          selected top-6 only ─► NVMe ⇄ RAM ⇄ GPU
```

Streaming every dense layer from disk only solves capacity and usually
destroys tokens/second. Conversely, requiring every MoE expert in GPU memory
wastes the sparsity that makes a trillion-parameter checkpoint executable.
Quantum LLM makes those policies explicit in an authenticated artifact:

```text
SafeTensors checkpoint
        │ strict source adapter
        ▼
transactional artifact below ${MODEL_ROOT}
  manifest + hashes + tokenizer/processor assets
  compute-ready QPack or authenticated expert records
  runtime-model.tsv: operations, geometry, encodings, capabilities
        │
        ▼
common HTTP service ─► common native VM ─► capability providers
        │                                      │
 session/admission                    per-organ placement and telemetry
        └──────────────────────────────────────┘
```

The common service and VM do not branch on Qwen, Muse, Ornith, DeepSeek,
layer counts or family tensor paths. A source adapter understands an upstream checkpoint's
names; the published program then drives execution. New mathematics or a new
encoding needs a provider and numerical qualification, not a second server,
task or deployment path.

## What it provides today

- source-pinned, transactionally published Qwen, Muse, Ornith and DeepSeek artifacts;
- one capability-driven model program and native runner for all four models;
- packed FP4 E2M1/UE8M0 SM86 kernels and exact format reporting;
- Windows IOCP, bounded staging and protected RAM/VRAM expert caches;
- exact DeepSeek top-k with per-tier hits, reloads, bytes and wait telemetry;
- progressive exact-F16 Qwen KV, retained sessions, transactional resume,
  rewind and cancellation;
- OpenAI and Anthropic-compatible local APIs, tools, thinking and Qwen image
  input (not image generation);
- Pi integration, lifecycle automation and fail-closed artifact validation.

It does **not** yet provide fast exact 262K Qwen decode, true parallel Qwen
agents, heterogeneous multi-GPU execution, broad model coverage or a hardened
public-network edge.

## How it compares

| Project | Its strength | Why Quantum LLM is different |
|---|---|---|
| [Ollama](https://github.com/ollama/ollama) | simple local model acquisition and lifecycle on top of a broadly supported backend | Quantum LLM is not a desktop model manager; it exposes artifact ABIs, exact tier traffic and per-organ placement for four controlled model shapes |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | portable GGUF inference, many quantizations/backends, CPU+GPU offload and a mature [server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | llama.cpp is the better general local runtime; Quantum LLM trades breadth for compute-ready records, native Windows expert paging and exact placement/accounting experiments |
| [vLLM](https://docs.vllm.ai/) | high-throughput GPU serving, continuous batching, PagedAttention and tensor/pipeline parallelism | vLLM is the better resident/distributed production server; Quantum LLM targets a single host where sparse weights may exceed RAM+VRAM and storage traffic is part of the execution contract |
| [SGLang](https://docs.sglang.io/) / [TensorRT-LLM](https://docs.nvidia.com/tensorrt-llm/) | production scheduling, prefix reuse, optimized attention and multi-GPU kernels | Quantum LLM has a far smaller serving surface; its focus is exact heterogeneous state ownership on one controlled host, not replacing these schedulers |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | optimized CPU/GPU heterogeneous MoE execution with hot experts on GPU | Quantum LLM adds an explicit NVMe tier, immutable logical expert pages and Windows IOCP, but is currently slower and narrower |
| [AirLLM](https://github.com/lyogavin/airllm) | running oversized dense models by loading layers sequentially | Quantum LLM rejects per-token dense-layer streaming as its speed path; it pages only sparse routed experts and treats dense/KV organs separately |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | the closest overlap: expert offload/prefetch, activation caching, batching and multi-GPU MoE serving | SSD expert offload alone is therefore not a novelty claim. Quantum LLM's distinct work is its fail-closed compute-ready artifact/program ABI, exact byte/tier attribution and dense/sparse organ placement through one VM |

The honest advantage is not “faster than everything.” It is a controlled
system in which a huge sparse model can execute exactly beyond RAM+VRAM and
every representation, placement decision and transfer can be verified. The
active research question is whether that control can produce a demonstrated
maximum-context and novel-route throughput advantage on the self-contained
RTX 3090 host. Until the gates in [Roadmap](docs/roadmap.md) pass, that remains
a target rather than a result.

## Quick start

Read [Getting started](docs/getting-started.md) before downloading or
converting a checkpoint. With a validated artifact already below `MODEL_ROOT`:

```bash
cp .env.example .env
# Set QUANTUM_LLM_REMOTE, QUANTUM_LLM_REMOTE_ROOT, MODEL_ROOT and API key.
./ops/model.sh install
./ops/model.sh start qwen
./ops/model.sh chat qwen
./ops/model.sh stop all
```

The same public path selects DeepSeek:

```bash
./ops/model.sh start deepseek
./ops/model.sh chat deepseek
./ops/model.sh stop all
```

Ornith uses that same path and service task:

```bash
./ops/model.sh start ornith
./ops/model.sh chat ornith
./ops/model.sh stop all
```

Muse uses its artifact-declared 131K limit automatically even when the global
Qwen configuration requests 262K:

```bash
./ops/model.sh start muse
./ops/model.sh chat muse
./ops/model.sh stop all
```

For a coding harness, see [Pi CLI](docs/pi-cli.md). For the complete supported
surface, start at the [documentation index](docs/README.md).

## Repository map

| Path | Responsibility |
|---|---|
| `compiler/` | strict checkpoint inspection, QPack compilation, DeepSeek descriptors/oracles and validation |
| `core/` | platform-neutral artifact and feasibility contracts |
| `runtime/` | common VM, caches, providers, worker protocol and CUDA kernels |
| `ops/` | sync, build, download, publication, service and client workflows |
| `schemas/` | active external request schemas only; executable artifacts use the compiler/native fail-closed validators |
| `tests/` | compiler, API, behavior and integration contracts |
| `docs/` | current architecture, evidence, interfaces, decisions and roadmap |

Generated `work/`, `artifacts/`, `logs/`, `out/` and caches are not source.
The former KV-attach Memory Expert experiment lives in the standalone sibling
project `../memory-expert`.

Model weights are not included. Users are responsible for model licenses and
their terms. Source code is licensed under the [Apache License 2.0](LICENSE).
