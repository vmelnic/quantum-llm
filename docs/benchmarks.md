# Performance evidence

## How to read the numbers

The project reports four different quantities and does not substitute one for
another:

- **runner hot single-stream**: one request in a warmed native worker;
- **runner hot aggregate**: total output across concurrent warmed requests;
- **API post-first-token**: decode rate after TTFT for one HTTP request;
- **API end-to-end**: output tokens divided by the entire request wall time.

Cold and warm results are always separate. “Warm” is route-specific: an expert
set warmed by one prompt does not make arbitrary future conversations hot.

## Qwen3-Next 80B

Reference configuration:

- model: `Qwen/Qwen3-Next-80B-A3B-Instruct`;
- source revision: `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`;
- source checkpoint: 162,659,161,528 bytes;
- Expert Pack v1 INT8: 81,903,198,208 bytes;
- dense pack: 4,191,133,696 bytes;
- RTX 3090 24 GB, approximately 64 GB RAM, local SATA SSD;
- 48 GiB RAM expert budget and 18 GiB VRAM expert budget.

### Qualified hot native gates

| Measurement | Result | Meaning |
|---|---:|---|
| single-stream hot runner | 47.6703 tok/s | exact short routes already resident |
| four-request hot aggregate runner | 42.5976 tok/s | aggregate service throughput, not one chat |

The measured windows had no expert SSD reads or expert H2D transfers. These
numbers are kernel/runtime capability under exact hot placement, not cold API
promises.

### Historical real-text API probe

Two identical sequential requests on an already-running service used a
25-token prompt and requested 32 output tokens:

| State | TTFT | Total | Post-first-token |
|---|---:|---:|---:|
| first request after heterogeneous traffic | 25.664 s | 56.216 s | 1.01 tok/s |
| immediately repeated request | 0.807 s | 2.429 s | 19.12 tok/s |

### Current lifecycle/chat verification

After repairing lifecycle, dependency, protocol and Unicode streaming issues,
an identical 21-token prompt producing nine tokens was measured before and
after route reuse:

| State | TTFT | Total | End-to-end | Post-first-token |
|---|---:|---:|---:|---:|
| first request after restart | 30.551 s | 42.020 s | 0.21 tok/s | 0.70 tok/s |
| immediate identical repeat | 0.601 s | 0.868 s | 10.37 tok/s | 30.07 tok/s |

The 30.07 tok/s repeat validates the warmed API decode path. It does not mean
general chat has reached 30 tok/s.

A real three-turn conversation with changing prompts/history observed:

| Prompt tokens | Output tokens | TTFT | End-to-end | Post-first-token |
|---:|---:|---:|---:|---:|
| 9 | 12 | 5.340 s | 0.65 tok/s | 0.84 tok/s |
| 44 | 46 | 10.395 s | 1.10 tok/s | 1.44 tok/s |
| 157 | 54 | 27.790 s | 0.73 tok/s | 1.14 tok/s |

This is the representative user-facing result: approximately 0.7–1.4 tok/s
after the first token for changing conversation routes, with TTFT increasing as
the full history is recomputed. The API currently resends/re-prefills history;
it does not retain a reusable conversation KV prefix between requests.

The Unicode verification after commit `86a4b3e` produced `Hello! 😊` once,
without a replacement character or replayed prefix. That was a correctness
gate, not a performance measurement.

## DeepSeek-V4-Flash

The 284B-class DeepSeek backend is functionally complete enough for greedy API
generation on the same host, but not performance-ready. Representative
end-to-end decode remains roughly 0.3–0.6 tok/s depending on route/cache state.

Measurements established:

- layer-major prefill substantially improved the original scalar prefill;
- compact routed FP4 storage reduced expansion and transfer volume;
- previous-route reuse removes repeated admissions but not first-touch traffic;
- a forced CPU/GPU split remained exact but was slower than the all-GPU hot
  layer on the six-core reference CPU;
- attention and request-state allocation are not the sole dominant bottleneck;
- expert working-set turnover, memory movement and expert compute require an
  order-of-magnitude improvement for a 30 tok/s single stream.

No DeepSeek result in this repository should be described as a 30 tok/s model
or chat result.

## Context and output limits

The deployed API advertises 65,536 context tokens and up to 8,192 output tokens.
Those are admission ceilings, not throughput evidence. Qwen KV is paged and
allocated on demand, so a short request does not physically allocate 65K KV.

Only 4,096 tokens have completed the historical long-context correctness gate.
Claims at 8K–65K require separate numerical, memory, TTFT and decode evidence.

## Current performance verdict

- Qwen native hot paths exceed 30 tok/s.
- Qwen repeated-route API decode can reach about 30 tok/s.
- Qwen arbitrary multi-turn chat is currently about 1 tok/s and misses the SLO.
- DeepSeek chat is below 1 tok/s and misses the SLO by a larger margin.
- `ready=true` means healthy/admitting, not warmed or SLO-compliant.

The acceptance target remains at least 30 useful output tok/s for a declared
workload. Any future claim must state model, prompt/history distribution,
cold/warm state, single/aggregate scope, TTFT, post-first-token rate and
end-to-end rate.
