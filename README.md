# Quantum LLM

Quantum LLM is a native Windows/CUDA inference runtime for running controlled
LLM artifacts on one RTX 3090 when model state must be placed deliberately
across VRAM, host RAM and NVMe.

The runtime treats a model as executable organs rather than one opaque weight
file:

- dense matrices, routers and recurrent state are hot and stay beside GPU
  compute;
- exact attention KV is request-owned and allocated progressively;
- routed MoE experts are immutable logical pages and may move through
  NVMe -> RAM -> VRAM only when selected;
- the artifact declares topology, operations, geometry, encodings, tokenizer
  behavior and provider capabilities;
- the common service and VM select providers from those declarations, not from
  model-family branches.

The current artifact registry exposes six selections through the same service:

| Alias | Artifact | Current execution |
|---|---|---|
| `qwen` | `qwen3.8-27b-fp4` | resident FP4 weights, hybrid recurrent/full attention, progressive exact F16 KV, image input |
| `qwen-flash` | `qwen3.8-flash-next-fp4` | Hyper/Gated DeltaNet/QSA/PLE plus exact top-10 paged FP4 experts |
| `mistral` | `mistral-small-4-119b-nvfp4` | source-native block-16 NVFP4 MoE, BF16 organs and latent KV |
| `muse` | `muse-glimmer-30b-fp4` | dense FP4 text path with exact global and sliding-window F16 KV; 131K artifact limit |
| `ornith` | `ornith-1.5-35b-a3b-fp4` | hybrid recurrent/attention path with exact top-8 FP4 experts |
| `deepseek` | `deepseek-v4-flash` | resident dense/shared organs and exact top-6 experts demand-paged through NVMe/RAM/VRAM |

## Why it exists

General runtimes already solve broad local inference well. This project exists
for a narrower problem: make representation, placement, transfer and execution
decisions explicit and measurable for models that do not fit a simple
all-resident layout.

```text
official checkpoint
        |
        v
strict source adapter
        |
        v
validated artifact under ${MODEL_ROOT}
  manifest + hashes + tokenizer/processor assets
  QPack or compact expert payloads
  runtime-model.tsv executable program
        |
        v
common authenticated HTTP service
        |
        v
common native VM -> capability provider -> VRAM / RAM / NVMe
```

This separation matters because ordinary offload is not a universal speed
solution. Dense weights and all retained K/V in full attention are hot every
decode call; moving them over PCIe or NVMe only recovers capacity. Unselected
MoE experts are genuinely cold, so exact demand paging is valid for DeepSeek
and similar sparse programs.

## Current status

The implementation is functional research/pilot software, not production-ready
for the maximum-context target. All six aliases pass a minimal real Pi CLI
`hi` gate without project context, tools or session state. Short Qwen, Muse and
Ornith requests are responsive; Qwen Flash, Mistral and especially DeepSeek
remain slow through Pi. The latest exact-F16 Qwen near-maximum prefill populated
262,001 prompt tokens in 2,183.815 seconds and generated one token; saturated
262K decode at the approximately 15 useful tok/s target is not qualified.

See [Benchmarks](docs/benchmarks.md) for measurements and
[Production readiness](docs/production-readiness.md) for the release boundary.

## Comparison

| Runtime | Best fit | Quantum LLM boundary |
|---|---|---|
| [Ollama](https://github.com/ollama/ollama) | simple local model lifecycle | Quantum LLM exposes executable artifact ABIs and tier traffic rather than a desktop model catalog |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | portable GGUF inference and CPU/GPU offload | Quantum LLM is narrower, native SM86, and separates hot dense/KV organs from pageable experts |
| [vLLM](https://docs.vllm.ai/) | resident/distributed throughput, batching and PagedAttention | Quantum LLM targets one controlled host where sparse expert storage participates in execution |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | heterogeneous CPU/GPU MoE execution | Quantum LLM adds an authenticated NVMe tier and fail-closed artifact/program contracts, but is less mature |
| [AirLLM](https://github.com/lyogavin/airllm) | capacity through sequential dense-layer loading | Quantum LLM does not use dense-layer streaming as a decode-throughput claim |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | expert offload, prefetch and activation-aware MoE serving | The overlap is substantial; Quantum LLM's useful distinction is exact byte/tier attribution and one artifact-driven VM, not SSD offload by itself |

The honest current advantage is control and observability, not a demonstrated
throughput win over mature resident-model servers.

## Quick start

```bash
cp .env.example .env
# Configure QUANTUM_LLM_REMOTE, QUANTUM_LLM_REMOTE_ROOT, MODEL_ROOT and API key.
./ops/model.sh install
./ops/model.sh start qwen
./ops/model.sh chat qwen
./ops/model.sh stop all
```

Replace `qwen` with `qwen-flash`, `mistral`, `muse`, `ornith` or `deepseek`.
For the coding harness:

```bash
./ops/model.sh start qwen
./ops/pi.sh qwen
```

Start with the [documentation index](docs/README.md).

## Repository map

| Path | Responsibility |
|---|---|
| `compiler/` | strict checkpoint adapters, compilation and artifact validation |
| `core/` | artifact and feasibility contracts |
| `runtime/` | VM, providers, caches, worker protocol and CUDA kernels |
| `ops/` | download, build, publication, service and client workflows |
| `tests/` | compiler, service and runtime contracts |
| `docs/` | current architecture, evidence, operations, decisions and roadmap |

`work/`, `out/`, `logs/`, `artifacts/`, model files and Hugging Face caches are
generated state, not source. Model weights are not included; users remain
responsible for model licenses. Source code is licensed under
[Apache License 2.0](LICENSE).
