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
| single-stream hot runtime | 31.7312 tok/s | 10 tok/s |
| four-request aggregate hot runtime | 31.1004 tok/s | 30 tok/s |

The gate used a warmup in the same process, retained expert residency, then
reset request state and measurement counters. It observed no expert-weight H2D
in the measured hot window, no pagefile growth, and identical batched versus
isolated token sequences.

These numbers are runner throughput under a qualified hot placement. They are
not a promise of 31 user-visible tokens/s for a cold chat.

## Service smoke

The API deployment smoke covered four concurrent completions, legacy streaming,
Responses JSON and typed streaming, Chat Completions, usage streaming,
capability errors, identity, and disconnect cancellation. A representative
cold operational window reported:

- 21 decode batches / 32 rows;
- effective decode batch: 1.5238;
- TTFT p95: 23.812 seconds;
- inter-token p95: 2.469 seconds.

Smoke latencies include cold expert placement and short, heterogeneous requests.
They demonstrate lifecycle behavior, not the hot throughput SLO. The large gap
between hot gate and cold API latency is an open production problem.

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
