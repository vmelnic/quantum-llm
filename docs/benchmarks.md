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

### W2 hot decode path (2026-08-08, build `w02`)

Changes: asynchronous directory planning on the per-token path
(`begin_plan_async`/`wait_plan_async`/`poll_plan_async` instead of
`pin_or_collect_misses`, non-blocking `release_pins_async`), worker-mode
placement freeze after an 8-decode-step warmup boundary (positional worker
argument `settle-after-decode-steps`, 0 disables), and vectorized
`char4`/`float4` dispatch in `gemv_batch`.

Six-turn chat suite (growing history, 48 max output tokens per turn,
post-first-token decode tok/s), same probe against the W1 build and the W2
build, both with session retention:

| Turn | W1 decode | W2 decode | W1 wall | W2 wall | W1 wait share | W2 wait share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.72 | 0.82 | 91.9 s | 69.8 s | 93% | 95% |
| 2 | 1.27 | 1.49 | 46.3 s | 38.4 s | 83% | 88% |
| 3 | 2.08 | 2.77 | 28.6 s | 21.9 s | 76% | 79% |
| 4 | 1.83 | 2.29 | 15.1 s | 12.3 s | 74% | 80% |
| 5 | 2.31 | 3.14 | 10.0 s | 7.8 s | 70% | 76% |
| 6 | 2.72 | 3.55 | 20.2 s | 15.7 s | 65% | 70% |

Wait share is `expert_cache_wait_ns` divided by request wall; on these turns
it is dominated by real storage/upload waits for not-yet-resident routes
(1–6 GiB per turn still crosses the disks), which is W3/W4 territory.

Repeated identical request (19-token prompt, 48 output tokens) on a freshly
restarted service, so the route becomes fully resident after the first run:

| Run | TTFT | Post-first-token |
|---:|---:|---:|
| 1 (cold, SSD load) | 26.890 s | 0.75 tok/s |
| 2 | 0.813 s | 27.16 tok/s |
| 3 | 0.462 s | 26.69 tok/s |
| 4 | 0.552 s | 27.69 tok/s |

On the resident runs the per-request telemetry shows zero expert storage
reads, zero H2D uploads, and `expert_cache_wait_ns` at ~24–30% of forward
wall (down from 65–95%); the residual is the event wait for in-flight GPU
layer work plus the CPU-executed share of the route (the 48-token route's
unique experts exceed the 18 GiB VRAM budget, so a fraction executes from
RAM on the CPU executor once placement is frozen).

### W3 async expert supply (2026-08-08, build `w03`)

Landed changes: uploads in `CudaExpertUploader` complete through per-upload
CUDA events consumed by a dedicated completion thread instead of
`cudaStreamSynchronize` under the pool lock (all three paths: Qwen INT8
sections, DeepSeek direct compact, DeepSeek compact expansion) — neither the
decode thread nor a storage callback pays a stream-wide sync on the hot path;
the DeepSeek scheduler no longer suspends prefetch while a demand acquire is
in flight and runs 4 in-flight prefetches with 2 transition predictions per
layer (was 1/1); the DeepSeek worker uses 4 IOCP threads (was 2) and 8
staging slots (was 4 — a full top-6 demand route plus two prefetches in
flight), charges the 512 MiB MTP reserve against the main cache budgets only
when MTP is enabled, and earns the pageable RAM-retention copy on the second
demand load of a record (`ram_retention_minimum_frequency = 2`), leaving
first-touch records pack-resident without the 13.4 MB memcpy on the miss
path.

Reverted experiment: a Qwen route-ahead prefetch (acquire layer i+1's
previous-token route during layer i's compute) was implemented, measured,
and removed. The telemetry showed it cannot help this regime: predicted
experts were always already RAM-resident (storage bytes identical to
baseline), so the prefetch only drove extra H2D uploads — 2–7× more uploaded
bytes per turn — which churned the 18 GiB VRAM tier and slightly regressed
every turn (novel-route mean 2.83 vs 3.24 tok/s). First-touch experts, the
actual misses, are by construction unpredictable from route history.

Qwen, six-turn chat suite (same probe as W1/W2, 48 max output tokens per
turn, freshly restarted service, post-first-token decode tok/s):

| Turn | W3-before | W3-after | before wait share | after wait share | storage read (both) |
|---:|---:|---:|---:|---:|---:|
| 1 (cold) | 0.76 | 0.75 | 97% | 97% | 23.9 GiB |
| 2 | 1.44 | 1.38 | 90% | 90% | 7.35 GiB |
| 3 | 2.51 | 2.44 | 82% | 82% | 3.69 GiB |
| 4 | 2.14 | 2.13 | 81% | 81% | 1.72 GiB |
| 5 | 2.99 | 2.82 | 77% | 77% | 1.05 GiB |
| 6 | 3.47 | 3.36 | 72% | 71% | 1.51 GiB |

Novel-route probe (8 independent single-turn questions on unrelated topics,
48 max output tokens): before 2.57–4.22 tok/s (mean 3.24, 0.84–3.65 GiB read
per turn, wait share 60–80%); after 2.43–4.15 tok/s (mean 3.06, identical
read bytes and wait shares). Qwen is unchanged within noise: the multi-turn
and novel-route cost is SATA first-touch reads, which neither the
event-driven uploader nor history-based prefetch can remove.

Repeated identical request (19-token prompt, 48 output tokens, freshly
restarted service), hot-path regression check:

| Run | W2 TTFT | W2 decode | W3 TTFT | W3 decode |
|---:|---:|---:|---:|---:|
| 1 (cold) | 26.890 s | 0.75 tok/s | 26.954 s | 0.75 tok/s |
| 2 | 0.813 s | 27.16 tok/s | 0.694 s | 28.22 tok/s |
| 3 | 0.462 s | 26.69 tok/s | 0.484 s | 27.27 tok/s |
| 4 | 0.552 s | 27.69 tok/s | 0.529 s | 26.99 tok/s |

DeepSeek-V4-Flash, 3-turn bounded probe (24/24/11 output tokens, freshly
restarted service; `w02r` is a reference build of the pre-W3 tree):

| Turn | w02r TTFT | w03 TTFT | w02r decode | w03 decode | w02r expert wait | w03 expert wait | w02r read | w03 read |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64.3 s | 51.9 s | 0.44 | 0.46 | 89.9 s | 80.5 s | 38.8 GiB | 56.1 GiB |
| 2 | 47.4 s | 45.4 s | 0.47 | 0.52 | 67.4 s | 62.8 s | 24.5 GiB | 29.8 GiB |
| 3 | 37.4 s | 31.8 s | 0.36 | 0.47 | 46.2 s | 35.3 s | 18.0 GiB | 14.6 GiB |

DeepSeek improves modestly (turn-3 wall −19%, expert wait −24%) but stays in
the ~0.4–0.5 tok/s class; the W3 pipeline work removes stalls, not the
bandwidth arithmetic (3.21 GiB/token). Caveats: the route census persists
across runs, so the w03 turns saw a census warmed by the w02r probe, and
turn 1 reads more under w03 because the widened prefetch window reads
predictions ahead of demand (wall time still improves). Both smokes pass on
w03 (`Invoke-P6ServiceSmoke.ps1`, `Invoke-DeepSeekServiceSmoke.ps1`).

### W4 router-aware hot-expert cache (2026-08-09, build `w04r3`)

Landed changes: the frozen placement now keeps receiving router feedback —
the asynchronous directory plan already carries the exact route to the host,
so per-layer access counts flow into the cache LFU from
`plan.selected_experts` without resurrecting the retired per-selection D2H
copies (`run_moe` in `qwen3_next_runner.cpp`); and a RAM-resident miss can be
re-promoted to VRAM while frozen. Re-promotion is gated by victim recency,
not by candidate hotness: once per forward pass the cache scans for
unreferenced VRAM residents that have not been routed for at least 2^19
access-clock ticks (`ExpertCache::vram_stale_resident_bytes`, roughly seven
48-token chat turns at the measured ~60-80k ticks per turn), and promotions
are bounded by that stale-victim byte budget. A repeating route whose working
set exceeds the VRAM budget has no stale residents, so its overflow stays on
the CPU executor as in W2; a conversation that has moved on displaces dead
topics' residents. Promotions upload from the RAM tier and add no storage
reads. The current route's protection leases are acquired only when the
stale budget is nonzero — unconditional lease acquire/release pairs each
force a full cache-map `drive()` scan, which alone cost ~8 ms/token on the
resident route in an intermediate build. New per-request telemetry:
`frozen_promotions`, `frozen_promotion_bytes` (worker STATS and
`request_telemetry`). DeepSeek code paths are untouched:
`vram_admission_would_improve` keeps its exact pre-W4 semantics and
`ram_retention_minimum_frequency = 2` is preserved.

Repeated identical request (19-token prompt, 48 output tokens, freshly
restarted service), resident-route regression gate:

| Run | W3 TTFT | W3 decode | W4 TTFT | W4 decode | W4 promotions |
|---:|---:|---:|---:|---:|---:|
| 1 (cold, SSD load) | 26.954 s | 0.75 tok/s | 26.906 s | 0.75 tok/s | 0 |
| 2 | 0.694 s | 28.22 tok/s | 0.676 s | 28.93 tok/s | 0 |
| 3 | 0.484 s | 27.27 tok/s | 0.570 s | 28.62 tok/s | 0 |
| 4 | 0.529 s | 26.99 tok/s | 0.461 s | 27.77 tok/s | 0 |

Resident runs show zero expert storage reads, zero H2D uploads and zero
frozen promotions — the gate holds.

Long diverse-session probe (one retained session, 16 turns drifting across
eight topics and returning to the opening topic at turns 15-16, 48 max
output tokens, freshly restarted service; turns 2-16 totals, W3 build `w03`
vs W4 build `w04r3`):

| Metric | W3 (before) | W4 (after) |
|---|---:|---:|
| total wall, turns 2-16 | 341.6 s | 337.3 s |
| total storage read, turns 2-16 | 35.87 GiB | 34.91 GiB |
| turns 2-11 (drift phase) | unchanged within noise | 0 promotions |
| turn 13 wall / storage-wait share / CPU expert | 31.73 s / 57.5% / 1.81 s | 22.44 s / 19.2% / 1.15 s (1,538 promotions) |
| turn 14 wall / storage-wait share / CPU expert | 33.28 s / 64.4% / 1.69 s | 18.81 s / 16.2% / 0.77 s (1,540 promotions) |
| turn 15-16 (revisit) wall | 29.66 s / 13.06 s | 37.95 s / 23.08 s |

The mechanism works where it can: once early topics are long-dead, their
VRAM residents are displaced by the current conversation's hot RAM-resident
experts (turns 13-14: wall −29%/−43%, storage-wait share down 3-4×, CPU
executor share down ~2×, fewer bytes re-read). The revisit turns are **not**
healed: by turn 15 the session had pushed ~55 GiB of unique experts through
the 48 GiB RAM tier, so the opening topic's experts were no longer
RAM-resident and had to come from SATA again (2.80 GiB read on turn 15) —
a RAM-capacity limit that VRAM re-promotion cannot address, and the extra
promotion work on top of a storage-bound turn made those two turns slower.
A dedicated revisit probe (two opening-topic turns, twelve drift turns, then
two revisit turns, freshly restarted service) hit the same boundary: ~57 GiB
had passed through the RAM tier by the revisit, so despite promotions firing
(441-1,593 per turn on turns 12-16) the revisit turns still read 2.80/1.19
GiB from storage.

Honest negatives (two failed gating variants, measured and discarded before
landing the recency gate):

- **Ungated re-promotion** (any admission-improving RAM-resident miss is
  uploaded): the resident repeat route collapsed to ~10 tok/s — ~3,800
  promotions and ~10 GiB of H2D per 48-token request, ping-ponging between
  equally hot entries of a route that exceeds the VRAM budget. Zero storage
  reads, pure VRAM churn.
- **Temperature-margin gating** (promote only if strictly hotter by 2×):
  still ~800 promotions per repeat request (a repeating route has intrinsic
  temperature spread) — resident route ~15 tok/s. A per-miss O(cache-map)
  admission scan added a further ~10 ms/token until the scan was hoisted to
  once per forward pass.
- **2^16-tick recency (~1 turn)**: resident gate clean, but every topic
  change re-uploaded the turn's misses (3-6k promotions, 10-20 GiB H2D per
  turn); long-session total wall regressed +23% (419.1 s vs 341.6 s).

Net: W4 lands as a safe, bounded self-healing mechanism — resident route
unchanged at ~27-29 tok/s, drift-phase behavior unchanged, settled new
topics reclaim VRAM from dead ones (up to −43% turn wall), no added storage
reads anywhere. It does not move the multi-turn chat totals on this probe,
because those turns sit at the SATA first-touch floor, and it cannot heal a
revisited topic whose experts have already left the 48 GiB RAM tier.
`Invoke-P6ServiceSmoke.ps1` passes on `w04r3`; the full build/ctest/Python
suite (46 tests) is green, including a new cache test
(`test_vram_stale_resident_bytes_tracks_victim_recency`).

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
