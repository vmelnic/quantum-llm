# Benchmarks and evidence

## Tested configuration

- model: `Qwen/Qwen3-Next-80B-A3B-Instruct`;
- source revision: `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`;
- source tensors: 162,659,161,528 bytes;
- Expert Pack v1 INT8: 81,903,198,208 bytes;
- dense pack: 4,191,133,696 bytes;
- hardware: RTX 3090 24 GB, approximately 64 GB RAM, Windows, local SSD;
- runtime profile: 48 GiB RAM cache, 18 GiB VRAM expert cache;
- worker: four request slots, 4096-token maximum context.

## Qualified hot-path gates

The final full-model gate produced:

| Gate | Result | Required |
|---|---:|---:|
| single-stream hot runtime | 47.6703 tok/s | 10 tok/s |
| four-request aggregate hot runtime | 42.5976 tok/s | 30 tok/s |

The gate used a warmup in the same process, retained expert residency, then
reset request state and measurement counters. It observed no expert-weight H2D
in the measured hot window, no pagefile growth, identical batched versus
isolated token sequences, and exact chunked-prefill versus scalar-prefill
sequences on the full model.

The mixed batch contains prompt lengths 3, 3, 4, and 7. With a four-token
prefill chunk, the seven-token case crosses a `4 + 3` boundary and proves that
KV and recurrent DeltaNet state survive more than one chunk.

### Compact CPU-result path

The mixed batch previously copied a dense masked CPU output buffer whenever any
selection used the CPU. Compact selection mapping changed only transport and
aggregation storage:

| Measurement | Dense masked | Compact |
|---|---:|---:|
| CPU selections | 9,281 | 9,521 |
| CPU-result H2D | 521,830,400 B | 77,996,032 B |
| aggregate throughput | 41.6653 tok/s | 42.5976 tok/s |

The compact byte count is exactly `9,521 × 2,048 × 4`; no unused top-k slots
are transferred. Despite minor route/cache-count variation between runs, H2D
fell 85.05%, output equality remained exact, and pagefile growth remained zero.

This gate uses paged FP16 KV and online-softmax attention. One 6 MiB page was
physically sufficient for each short gate request. The batch run retained exact
batched-versus-isolated token equality, used no pagefile growth, and kept at
least 24.3 GiB physical RAM free.

These numbers are runner throughput under a qualified hot placement. They are
not a promise of 31 user-visible tokens/s for a cold chat.

## Service smoke

The API deployment smoke covered four concurrent completions, legacy streaming,
Responses JSON and typed streaming, Chat Completions, usage streaming,
capability errors, identity, and disconnect cancellation. A representative
cold operational window reported:

- 21 decode batches / 32 rows;
- effective decode batch: 1.5238;
- TTFT p95: 16.203 seconds;
- inter-token p95: 2.375 seconds.

Smoke latencies include cold expert placement and short, heterogeneous requests.
They demonstrate lifecycle behavior, not the hot throughput SLO. The large gap
between hot gate and cold API latency is an open production problem.

## Real text probe

A streaming Chat Completions probe used a 25-token templated English prompt and
requested 32 tokens. Two identical sequential requests on the already-running
service produced:

| Placement state | TTFT | Total | Approx. post-first-token rate |
|---|---:|---:|---:|
| first request after heterogeneous smoke | 25.664 s | 56.216 s | 1.01 tok/s |
| immediately repeated request | 0.807 s | 2.429 s | 19.12 tok/s |

The output text and token usage were identical. The warm prefill processed the
25-token prompt at roughly 31 prompt tok/s, but this is not yet a formal prefill
throughput gate. The result exposes two separate remaining costs: cold expert
placement dominates the first request, while a longer prompt/output working set
still trails the short synthetic runner's fully hot 47.67 tok/s decode rate.

## Memory observation

At startup/cold idle, total GPU usage was approximately 5.7 GiB: dense weights,
FP32 request state, CUDA context/workspace, and Windows display. After the smoke
heated the expert cache, total usage reached approximately 17.4 GiB. Stopping
the full process tree returned the GPU to approximately 421 MiB used by the
Windows desktop.

## Benchmark rules

Any published result must include:

- exact Git commit and model content hash;
- hardware, driver, CUDA version, RAM/VRAM/cache budgets;
- prompt/context/output lengths and concurrency;
- cold, warm, or frozen placement state;
- whether tokens/s means one request, aggregate requests, or model forward rows;
- TTFT and inter-token latency alongside throughput;
- correctness comparison and memory/pagefile evidence.

Do not compare results that use different cache states as though they measured
the same workload.
