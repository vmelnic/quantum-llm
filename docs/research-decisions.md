# Research decisions

Status: canonical decision and rejection ledger, 2026-09-12.

This file prevents failed mechanisms from returning under new names. Detailed
measurements live in [Benchmarks](benchmarks.md). Reopen a rejection only when
new evidence changes its capacity, bandwidth, fidelity or acceptance bound.

## Fixed targets

```text
Qwen3.8-27B
  host        one RTX 3090
  context     262,144 actually populated positions
  KV          exact IEEE F16
  harness     real coding agent, default xhigh
  target      about 15 useful output tok/s

DeepSeek
  host        same self-contained machine; NVMe+RAM+VRAM allowed
  routing     exact router, top-k and stable aggregation
  novel       about 1 tok/s acceptable
  settled     10-15 useful tok/s
```

The Qwen feasibility inequality is currently negative:

```text
hot target weights             13,625,700,352 bytes
exact F16 target KV            17,179,869,184 bytes
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
measured pinned PCIe link       12.46 GiB/s
full host-KV scan floor             1.284 s/call
```

Ordinary offload cannot reach 15 tok/s. A new runtime proposal must first prove
one of: enough compute-local capacity, lossless residency below the practical
bit threshold, or enough exact accepted tokens per full-KV traffic unit after
complete proposer/verifier cost.

## Accepted bounded mechanisms

| Mechanism | Accepted scope | Boundary |
|---|---|---|
| progressive exact-F16 mirror | Qwen prefixes that fit the disposable device mirror | preserves F16; does not solve 262K capacity |
| exact sparse-expert paging | DeepSeek and compatible MoE selected pages | makes models executable beyond RAM+VRAM; novel routes remain movement-bound |
| measured CPU/GPU expert split | exact selected experts and stable merge | first DeepSeek gate did not improve throughput |
| exact QSA tiering | Qwen Flash exact index plus selected F16 payload | QSA was only about 1.2% of measured short execution |
| startup routed-cache fitting | artifact-neutral, before first expert admission, after future-state reservation | admitted for `ornith-k1`; not a global cache increase |
| lazy dense/logits workspaces | allocate only the rows/operation requested | saved about 994 MiB; arithmetic unchanged |

## Experimental K1 exception

K1 stores block-32 FP4 values and FP4 keys with one aligned FP16 outlier
correction per key block. It is lossy. The operational `qwen` and
`ornith-k1` aliases opt into it; `qwen-f16` and `ornith` are fidelity references.

For Qwen at 262,144 positions:

```text
F16 target KV                 16.0000 GiB
K1 target KV                   4.7500 GiB
K1 target + MTP pages          5.015625 GiB
K1 attention, 16 layers       29.0161 ms (repeat)
K1 populated decode           14.39 tok/s after first token
```

A deterministic populated-262,016 request returned the same 47 output tokens
under K1 and F16; K1 took 546.173 s and F16 2,278.082 s. Both made the same
arithmetic error. One parity case is not a quality corpus, so K1 does not
satisfy the exact-F16 goal and cannot be described as lossless.

Promotion requires explicit acceptance of changed fidelity plus reproducible
long-context retrieval, reasoning and coding comparisons against F16. Until
then K1 is an operational experiment, not the target semantics.

## Rejected general mechanisms

| Direction | Failed prerequisite | Decision |
|---|---|---|
| FP8 target KV | 262K prefill 1,168.594 s, decode 3.064 tok/s and factual failure | wrong fidelity and too slow |
| Q4 target KV | user-confirmed material quality degradation | rejected for exact-F16 goal |
| uniform K8/V4 KV | 34.7668 ms for 16 attention layers versus 32.5 ms budget, before overhead | rejected before service integration |
| dense layer streaming/AirLLM | 12.690 GiB hot weights / 12.46 GiB/s >=1.018 s/call | capacity mechanism only |
| static lossless weight+KV coding | ideal result still 1,123,844,072 bytes over VRAM before runtime state | insufficient capacity |
| reversible F16 transforms | 13.265926 bits/value measured versus 11.617612 required | insufficient compression |
| Huffman/Zstd/zlib/LZ4 | best held-out Huffman 13.719539 bits/value | insufficient compression |
| exact MTP tree | 4.565 accepted/cycle; optimistic ceiling 2.8 tok/s | insufficient amplification |
| FP4-KV block-Jacobi | 3.25 accepted/cycle; corrected ceiling 1.55 tok/s | insufficient amplification |
| prompt n-gram/copy | 3.8485 useful/cycle versus 24 required | insufficient acceptance |
| exact page refinement | 2,029.014 MiB/call versus 272.822 MiB budget | traffic exceeds bound 7.437x |
| staged single-GPU exact attention | 91.763 ms for 16 layers before other work | misses complete 15 tok/s budget |
| CPU F16 V-tail | 226.279 ms/call at 12 threads | too slow |
| all-K-resident/selective-V | about 25,922 MiB before V staging | exceeds VRAM |
| pageable-to-pinned expert bounce | extra 1.385 GB copy and worse upload wait | removed |
| neural CPU replacement | failed structural/correctness boundary | removed |

## Rejected DeepSeek work

- Whole-prompt layer-major routing required about 176 GiB of selection streams
  at 1,048,576 tokens.
- Blocking into eight 131,072-token segments still required about 23.516 TiB
  of exact activation traffic.
- A real 1M-token attempt remained in block one after about 6.5 minutes.
- Exact two-row batching left one 131K block incomplete after 14 minutes.
- Shared-expert block-128 INT8 generated incoherent text and lacked a matching
  artifact oracle.

Do not reintroduce `layer_major_prefill`, `causal_layer_major`, row-width
tuning or the shared INT8 layout without a new quantitative prerequisite.

## P100 decision

The installed RTX 3090 and two P100s have no CUDA peer access; every boundary
uses pinned host memory.

| Placement | Measurement | Required | Decision |
|---|---:|---:|---|
| Qwen exact KV split | 77.101 ms attention compute-only | <=32 ms | reject |
| Qwen compact dense shard | 35.939 ms/card before integration | <=20 ms | reject |
| Qwen lossless resident FP16 MLP shard | 50.871 ms auxiliary wall; 307.7/311.0 GB/s | >=500 GB/s/card | reject |
| Qwen Flash selected experts | 0.20 tok/s vs 0.37 primary | faster than primary | reject for speed |
| DeepSeek selected experts | 3.18 tok/s settled | >=10 tok/s | reject for speed |

The P100 executor may remain an optional exact capacity path for compatible
routed components. It is not a throughput default, a dense-Qwen provider or a
reason to extend Pascal kernels without a newly passing end-to-end inequality.

## Model-specific parked work

- **Qwen Flash behavior:** corrected sampling/effort propagation and native
  parser tracing did not fix zero-edit/action-stall behavior. Do not tune the
  parser or thinking again without checkpoint, quantization or numerical
  evidence.
- **Bonsai/binary/ternary:** these are differently trained models, not lossless
  Qwen FP4 conversions. They require explicit model/quality approval.
- **Local/fast-weight/episodic memory:** changes model semantics and requires
  training; it is a model-research hypothesis, not inference optimization.
- **Nemotron generic FP4:** weak `hi` passed, normal instruction failed. Reopen
  only for an official NVFP4 integration with its actual encodings.

## Reopening rule

Before implementation, record the exact model, populated context, formats,
hardware and useful-output target, then calculate:

```text
weights + recurrent state + KV + draft/MTP + workspace + reserve
bytes touched per call and per accepted token
GPU + RAM + PCIe + NVMe lower bounds
acceptance * proposer/verifier cost, if speculative
```

If the resulting capacity, latency, traffic or fidelity prerequisite fails,
stop before runtime code.
