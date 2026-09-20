# Quantum LLM

## Qwen on one RTX 3090

The default Qwen `q4-f16-per-head` policy (short name: **Q4H**) was measured through the real
OpenAI-compatible endpoint with one active request, session retention disabled
and the artifact's thinking sampler (`temperature=1.0`, `top_p=0.95`,
`top_k=20`). These are client-observed post-first-token rates, not provider
microbenchmarks.

| External benchmark | Actual prompt | Output | TTFT | Post-first decode |
|---|---:|---:|---:|---:|
| llama-benchy 0.4.0 | 34,817 tokens | 32 tokens | 42.131 s | **56.99 tok/s** |
| GuideLLM 0.7.4 synchronous, concurrency 1 | 32,820 tokens | 128 tokens | 39.861 s | **60.58 tok/s** |

The two runs used different prompts and reached 2.6667 versus 2.9767 verified
positions per target call, so their difference is expected MTP acceptance
variance. GuideLLM's 3.05 tok/s end-to-end rate includes cold prefill; its
16.506 ms inter-token latency is the decode measurement shown above. The packed
per-head Q4 target KV is explicitly lossy relative to IEEE F16. These speed
results do not qualify fidelity; see the exact conditions and raw-result links
in [Benchmarks](docs/benchmarks.md#external-llama-benchy-endpoint-gate).

## Measured snapshot

Primary RTX 3090, current published artifacts; P100 results are labelled.
Payload size excludes tokenizer, indexes and publication scratch. TTFT/rates
come from the documented direct-chat conditions, not model startup.

| Model | Validated payload | Context ceiling | Short TTFT | Short decode | Populated-context evidence |
|---|---:|---:|---:|---:|---|
| Qwen3.8-27B | 14.78 GB QPack | 262,144 | 0.827 s (`off`, historical K1) | 30.92 tok/s after first, historical K1 | 262,016: 546.173 s K1; 2,278.082 s F16 |
| Qwen3.8-27B Abliterated | 14.78 GB QPack | 262,144 | 0.910 s (`off`, historical K1) | 25.33 tok/s after first, historical K1 | not measured at maximum; third-party weights |
| Qwen3.8-Flash-Next | 95.92 GB | 262,144 | 27.642 s (`off`) | 0.58 tok/s after first | not measured at maximum |
| Mistral Small 4 | 70.80 GB source tensors | 262,144 | 271.256 s (`off`) | 1.12 tok/s after first | not measured at maximum |
| Muse-Glimmer-30B | 15.83 GB QPack | 131,072 | 2.559 s (`off`) | 18.49 tok/s end-to-end | not measured at maximum |
| Ornith-1.5-35B-A3B | 19.23 GB pack | 262,144 | 4.590 s (historical K1+fit cold) | 2.73 tok/s cold; 7.42 next turn, historical K1 | not measured at maximum |
| DeepSeek V4 Flash | 147.17 GB routed experts + fixed organs | 262,144 | 12.953 s (`off`, novel) | 0.55 tok/s novel; 6.54-6.69 settled primary | not measured at maximum |

The context ceiling is only an admission limit. Of these paths, only Qwen has
a recorded 262,016-token populated gate. The current per-head Q4 default is
lossy; exact IEEE F16 remains available through `qwen-f16` as the Qwen
reference. Full conditions and less favorable results are in
[Benchmarks](docs/benchmarks.md).

## What this is

Quantum LLM is a native Windows/CUDA inference runtime for deliberate model
placement across RTX 3090 VRAM, host RAM and NVMe. It treats a model as an
artifact-declared executable program rather than a family-specific runner:

- dense weights, recurrent state and active attention state stay near compute;
- selected MoE experts are immutable logical pages that may move through
  NVMe, RAM and VRAM without reducing top-k;
- topology, tensor roles, operations, quantization and tokenizer behavior come
  from the artifact;
- one authenticated service and native VM bind those declarations to available
  providers.

```text
official checkpoint -> strict source adapter -> validated artifact
                    -> common HTTP service -> native artifact VM
                    -> GPU / RAM / NVMe placement
```

The project exists for controlled inference experiments that ordinary model
offload hides: exact byte attribution, fail-closed artifact ABIs, progressive
request state and genuine sparse-expert paging. Its demonstrated advantage is
control and observability, not higher throughput than mature resident-model
servers.

## Runtime boundary

| Runtime | Main strength | Difference here |
|---|---|---|
| [Ollama](https://github.com/ollama/ollama) | simple local model lifecycle | this project exposes artifact operations and tier traffic |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | portable GGUF CPU/GPU inference | this project is narrower, native CUDA and expert-page oriented |
| [vLLM](https://docs.vllm.ai/) | batching, PagedAttention and resident/distributed throughput | this project targets one controlled host and storage-backed sparse experts |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | heterogeneous CPU/GPU MoE execution | this project adds authenticated artifact and NVMe page contracts, but is less mature |
| [AirLLM](https://github.com/lyogavin/airllm) | capacity through layer-wise loading | this project does not present dense layer streaming as decode acceleration |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | expert offload and prefetch | the overlap is substantial; this project emphasizes one artifact VM and exact tier telemetry |

## Current scope

The common service exposes these aliases:

| Alias | Artifact/policy |
|---|---|
| `qwen` | Qwen3.8-27B with direct packed `q4-f16-per-head` target KV |
| `qwen-abliterated` | third-party Qwen3.8-27B Abliterated with direct packed `q4-f16-per-head` target KV |
| `qwen-f16` | the same artifact with exact progressive F16 KV |
| `qwen-flash` | Qwen3.8-Flash-Next, exact top-10 routed FP4 experts |
| `mistral` | Mistral Small 4, source-native NVFP4/BF16 |
| `muse` | Muse-Glimmer, dense FP4 with F16 global/sliding KV |
| `ornith` | Ornith with direct packed `q4-f16-per-head` target KV |
| `deepseek` | DeepSeek compact top-6 experts demand-paged through NVMe/RAM/VRAM |

Two installed P100s can execute compatible Flash/DeepSeek experts, but current
end-to-end measurements do not justify enabling them for performance. Dense
Qwen does not use them.

## Quick start

```bash
cp .env.example .env
# Set QUANTUM_LLM_REMOTE, QUANTUM_LLM_REMOTE_ROOT, MODEL_ROOT and API key.
./ops/model.sh install
./ops/model.sh start qwen
./ops/model.sh chat qwen
# Optional: remove only Qwen's durable restart/resume snapshots.
./ops/model.sh clear-cache qwen
./ops/model.sh stop all
```

Qwen's provider can retain an exact completed Pi prefix across a service
restart in `${MODEL_ROOT}/.session-cache/qwen3.8-27b-fp4`. The configured
default is 64 GiB with a seven-day TTL; change `MODEL_SESSION_CACHE_GIB` and
`MODEL_SESSION_CACHE_TTL_SECONDS` in `.env`. This removes repeated prefill only
when the resumed transcript has an exact compatible prefix; it does not speed
the first prefill.

Thinking can be selected when opening direct chat:

```bash
./ops/model.sh chat deepseek --thinking off
```

For the maintained coding harness:

```bash
./ops/model.sh start qwen
./ops/pi.sh qwen
```

Start with the [documentation index](docs/README.md). The runtime is functional
research/pilot software; see [Production readiness](docs/production-readiness.md)
before treating an API smoke as a deployment result.

## Repository map

| Path | Responsibility |
|---|---|
| `compiler/` | strict source adapters, compilation and artifact validation |
| `core/` | artifact and feasibility contracts |
| `runtime/` | VM, providers, caches, worker protocol and CUDA kernels |
| `ops/` | download, build, publication, service and client workflows |
| `tests/` | compiler, service and runtime contracts |
| `docs/` | architecture, evidence, operations, decisions and roadmap |

`work/`, `out/`, `logs/`, `artifacts/`, model files and Hugging Face caches are
generated state. Model weights are not included; users remain responsible for
model licenses. Source code is licensed under [Apache License 2.0](LICENSE).
