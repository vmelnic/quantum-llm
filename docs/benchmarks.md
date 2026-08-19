# Performance evidence

Status: current evidence index as of 2026-08-19. Dated W/S sections preserve
their original configurations and remain valid historical measurements. The
active implementation backlog is [MoE VM current state and remaining
work](moe-vm-next.md), not a benchmark section below.

The exact Qwen3.8 organ inventory, maximum-context KV geometry, current CUDA
kernel bandwidth inputs and heterogeneous placement bounds are recorded in
[Heterogeneous placement benchmark](heterogeneous-placement-benchmark.md).
That document distinguishes measured results from theoretical CPU/NVMe bounds
and contains no implementation claim.

## How to read the numbers

The project reports four different quantities and does not substitute one for
another:

- **runner hot single-stream**: one request in a warmed native worker;
- **runner hot aggregate**: total output across concurrent warmed requests;
- **API post-first-token**: decode rate after TTFT for one HTTP request;
- **API end-to-end**: output tokens divided by the entire request wall time.

Cold and warm results are always separate. “Warm” is route-specific: an expert
set warmed by one prompt does not make arbitrary future conversations hot.

## Common MoE VM functional acceptance (2026-08-17)

The final common-path gate used the same `hi` input through `ops/model.sh`, the
`QuantumLLM-ExpertVm` task, the OpenAI-compatible service and
`expert-moe-vm-runner.exe`. The Windows/CUDA build passed four CTest targets
and 59 Python tests before the gate.

| Artifact | TTFT | Total wall | Post-first-token | Result |
|---|---:|---:|---:|---|
| Qwen3-Next 80B FP4 | 11.755 s | 21.449 s | 1.13 tok/s | coherent greeting |
| DeepSeek-V4-Flash | 8.901 s | 15.728 s | 1.32 tok/s | coherent greeting |
| LFM2-8B-A1B FP4 | 7.508 s | 12.141 s | 2.37 tok/s | coherent greeting |

This is a lifecycle and functional-equivalence gate, not a throughput suite:
the outputs have different token counts and the services were cold/restarted.
It proves the shared public path, but it demonstrates no tok/s gain from the VM
refactor. This gate ran after moving the project, packs and build tree to the
Windows D: NVMe; Hugging Face source snapshots remained on C:. The
worker/service were stopped after the gate.

## Qwen3-Next 80B

Reference configuration:

- model: `Qwen/Qwen3-Next-80B-A3B-Instruct`;
- source revision: `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`;
- source checkpoint: 162,659,161,528 bytes;
- historical Expert Pack v1 INT8 comparison artifact: 81,903,198,208 bytes;
- dense pack: 4,191,133,696 bytes;
- RTX 3090 24 GB, approximately 64 GB RAM, local SATA SSD;
- 48 GiB RAM expert budget and 18 GiB VRAM expert budget.

### Host bandwidth (pinned measurement, 2026-08-10)

Probe: `work/fp4-s1/bandwidth_probe.py` (torch CUDA events for copies;
unbuffered 8 MiB sequential reads over the full 72.3 GiB int8 pack —
larger than RAM, so mostly cold). Artifact on the GPU host:
`work/fp4-s1/bandwidth-report.json`.

| Quantity | Measured |
|---|---:|
| H2D pinned (64/256/1024 MiB) | 12.45 / 12.46 / 12.46 GiB/s |
| D2H pinned | 12.26 GiB/s |
| H2D pageable | 12.05 GiB/s |
| PCIe link | Gen3 x16 max (idles at Gen1, ramps under load) |
| SATA sequential read | 0.47 GiB/s (all 19 shards, 0.46–0.48) |

Implied per-token ceilings (bandwidth ÷ bytes/token): int8 RAM tier
8.8 tok/s, FP4 RAM tier 17.8 tok/s, FP4 SATA tier 0.7 tok/s. The 30
tok/s goal at FP4 bytes/token (0.7 GiB) would need 21 GiB/s H2D — past
this bus — so on this host it is only reachable through VRAM residency
(hot native 47.67), not through any RAM-tier design.

### S1b: FP4 routed experts — measured results (2026-08-10)

Pack: `work/models/qwen3-next-80b-expert-pack-fp4`, 24,576 experts,
45.36 GB (vs 72.3 GiB int8), ABI 3, validator `valid:true`. Probes:
`work/fp4-s1/{fp4_weight_error,fp4_behavioral_probe,fp4_bench_probe}.py`;
artifacts `work/fp4-s1/{weight-error,behavioral-*,bench-*}.json`.
Same server configuration for both arms (context 4096, 48 GiB RAM /
18 GiB VRAM expert budgets).

Weight error (RTN, per-matrix relative L2 vs BF16 source): FP4 ~11.8%
(mean, max 12.0%), int8 ~0.85%. Expected for round-to-nearest FP4-E2M1
block-32; behavioral arm below is the quality gate that matters.

Two bugs were found and fixed before the arm could run:

- `cudaHostAlloc` does not guarantee 4096-byte alignment; the staging
  `FixedBufferPool` then failed every FP4 startup with `std::bad_alloc`.
  Fixed by over-allocating and rounding up inside the pinned allocator
  (`cuda_pinned_allocator.cpp`).
- FP4 livelock: with FP4 every miss goes to the GPU uploader
  (`gpu_available = true` regardless of admission), so eviction ran
  while the runner had skipped the ready-expert cache leases (the
  heuristic at `qwen3_next_runner.cpp:1588` predicted no uploads).
  Eviction could then pick a victim pinned by the in-flight route and
  `CudaExpertDirectory::retire()` spun forever waiting for a reference
  the blocked thread itself owned (`[retire-spin] ... refs=1`, 100% CPU
  on one core, GPU idle). Fixed structurally: eviction skips
  route-pinned victims via the host-side pin view
  (`route_pinned`), and `try_retire()` never spins — a busy entry is
  skipped, turning any future hole into a skipped victim instead of a
  hung server. The full FP4 arm completed with zero `[retire-busy]`
  events.

Throughput, decode tok/s per call (cold first call included):

| Mode | int8 | FP4 | FP4/int8 |
|---|---|---|---|
| resident (4 calls) | 2.40 / 6.20 / 7.79 / 7.75 | 7.90 / 39.56 / 45.59 / 45.59 | ~5.9x steady |
| novel (8 calls) | 1.89 – 3.98 | 5.32 – 16.90 | ~3.4x mean |
| suite (6 calls) | 3.10 – 5.24 | 16.43 – 25.27 | ~4.9x mean |

Behavioral quality: 10 fixed prompts, int8 vs FP4 — no identical texts
(expected at 11.8% weight error), but all FP4 answers are coherent,
on-topic, and comparable in length and structure; no truncation, no
degeneration, no factual collapse on inspection.

Gate reading: the plan's S1c gate "resident ≥ 27 tok/s" is met by FP4
within this probe (steady 39.6–45.6, close to the hot-native ceiling
47.67 — the routed working set of these prompts fits the 18 GiB VRAM
budget). Note the int8 arm of this probe measured 7.75 resident vs the
27–29 recorded in W4 under a different server configuration (65,536
context); the comparison above is valid because both arms here share
one configuration, but the absolute int8 numbers are not comparable
across configurations. FP4 novel-route calls reach 5.3–16.9 tok/s —
consistent with the 12.46 GiB/s H2D ceiling at 0.7 GiB/token
(theoretical 17.8) — and the whole 45.36 GB FP4 pack fits the 48 GiB
RAM budget, so the SATA floor is cold-start-only.


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

### Historical lifecycle/chat verification (pre-W1)

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

This was the pre-W1 representative user-facing result: approximately
0.7–1.4 tok/s after the first token for changing conversation routes, with
TTFT increasing as the full history was recomputed. At the time the API
resent/re-prefilled history on every turn; protocol v5 retained sessions
(W1, below) now prefill only the per-turn delta, and the W2–W4 decode-path
work raised these rates substantially (see the W2/W3 suites).

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
`cudaStreamSynchronize` under the pool lock (Expert Pack sections, DeepSeek
direct compact, DeepSeek compact expansion) — neither the
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

### W5 DeepSeek MTP speculative decode (2026-08-09, build `w05-mtp`)

The worker bundle v3 ships complete MTP resources (MTP dense/typed/shared
tensors plus a 3.42 GB one-layer routed compact pack, verified in place), so
no pack-side addition was needed; the MTP runtime (draft layer, pair
verification, adaptive suppression) already existed but had never been
enabled end-to-end. Landed changes:

- STEP protocol gains mode 2 ("hold": plain non-speculative decode that keeps
  the worker slot). A retained turn now ends on a hold step under MTP; the
  hold step also advances the MTP causal stream so the next resume starts
  with a valid draft. Without it, an accepted speculative pair on the last
  step of a turn leaves an unemitted bonus token inside the worker state; the
  retained session prefix then matches no client-echoed prompt, the resume
  silently fell back to a full re-prefill, and the capacity-1 slot was freed
  only by an LRU eviction on the next turn (`worker command failed:
  over-capacity BEGIN` in the pre-fix logs). Qwen step handling is unchanged;
  the server only emits mode 2 when the worker reports MTP enabled.
- Speculation suppression now requires 8 verify pairs before tripping (was
  2): with the EMA seeded from a single pair, any one rejection scores ~2x an
  ordinary step per useful token and disabled speculation on noise for the
  rest of the session. Suppression is also reset when a session resumes; the
  model-global cost EMA still guards, so a genuinely unprofitable draft loop
  is re-suppressed after the first new pair.
- A turn that still ends with an unemitted bonus (EOS cut through an accepted
  pair) is now explicitly not retained (`session_retain_skipped` log event)
  instead of storing a session that can never match.
- `request_telemetry` carries `worker_mtp_drafts`, `worker_mtp_accepted`,
  `worker_mtp_rejected`, `worker_verify_pairs`, `worker_mtp_suppressions`.

Three-turn bounded probe (24/24/11-12 output tokens, natural chat, freshly
restarted service, no route census present for either run, so both runs are
genuinely cold; `w05-base` is the same tree with MTP off):

| Turn | w05-base TTFT | w05-mtp TTFT | base decode | mtp decode | base expert wait | mtp expert wait | base read | mtp read | base steps | mtp steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 70.3 s | 70.9 s | 0.47 | 0.49 | 97.2 s | 96.9 s | 55.2 GiB | 57.1 GiB | 41 | 31 |
| 2 | 45.2 s | 45.6 s | 0.54 | 0.56 | 61.8 s | 61.1 s | 29.6 GiB | 28.8 GiB | 41 | 41 |
| 3 | 33.8 s | 33.9 s | 0.54 | 0.59 | 35.1 s | 35.4 s | 14.6 GiB | 14.6 GiB | 25 | 25 |

MTP acceptance on turn 1 (the only turn with enough speculation before the
cost EMA suppressed it): 10 accepted / 13 pairs (76.9%); probe total 11/15
(73%). Verify pairs cut model steps on turn 1 from 41 to 31 (−24%).

**MTP is throughput-neutral here, and that is now measured rather than
assumed.** A verify pair pays the union of two adjacent positions' expert
routes through the same bandwidth-bound supply pipeline: turn-1 storage reads
did not drop (57.1 vs 55.2 GiB) and expert wait is unchanged (96.9 vs 97.2 s)
despite 24% fewer model steps, because adjacent-token route overlap is too
small to amortize the 3.21 GiB/token supply. The only savings are per-step
fixed costs (attention, output head, scheduling), which are not the
bottleneck at ~2 s/token. The 8–15 tok/s W5 target would require the verify
pair to cost barely more than one ordinary forward; measured pair cost is
~2 forwards in expert bytes, so even perfect acceptance caps the gain near
zero in this regime. The adaptive suppression EMA reaches the same conclusion
on its own: after turn 1 it disables speculation within one pair per turn.

Honest negatives (both measured on intermediate builds and fixed before the
final numbers above):

- **Suppression after 2 samples**: the pre-W5 heuristic tripped on the first
  rejection (1 accept + 1 reject on turn 1), then stayed sticky in the
  retained session, so turns 2-3 ran plain decode with MTP enabled and the
  probe reproduced baseline numbers exactly.
- **Retention poisoning by the speculative bonus** (intermediate build, same
  probe): turn 2 fell back to a full 58-token re-prefill (TTFT 116.7 s vs
  45.2 s baseline, turn wall 155.9 s vs 87.6 s) because the retained session
  included the unemitted bonus token and matched no follow-up prompt.

The deployment keeps MTP enabled because the artifact declares draft/verify
operations; no model-specific launcher flag is required. With the hold-step
fix it is correct, retention-compatible and
self-suppressing when unprofitable, at a small extra read cost on cold turns
(+3.5% on turn 1). The full build/ctest/Python suite (48 tests, including
`test_hold_step_maps_to_worker_flag_2` and
`test_retained_mtp_turn_ends_with_hold_step`) is green on `w05-mtp`, and
`Invoke-DeepSeekServiceSmoke.ps1` passes with MTP active.


## DeepSeek-V4-Flash

The 284B-class DeepSeek backend is functionally complete enough for greedy API
generation on the same host, but not performance-ready. W3/W5 measured roughly
0.4–0.6 tok/s and S1-DeepSeek reached ~3.5 tok/s on settled turns. The later
priority/tier and indexed-supply follow-on below reached 6.54–6.69 tok/s only
on its settled identical-prompt workload; changing prompts remained much
slower.

### S1-DeepSeek: FP4 device accounting + warm-set recalibration (2026-08-10)

Premise correction: direct-FP4 execution for DeepSeek routed experts landed
in `05b474e` ("Execute DeepSeek experts directly from packed FP4") — upload
keeps the 13,369,344-byte compact record (`CudaCompactExpertAllocation`) and
compute runs the packed `__dp4a` selection-batch kernels. What was missing
was accounting: the routed catalog still declared
`device_bytes = 25,198,592` (the int8 SM86 expansion slot), so the VRAM
cache, preflight and census warm set sized every FP4 slot at ~2x its real
footprint, and `warm_from_census` was hard-capped at 6 experts/layer
(43x6=258 entries). Fixed in `deepseek_catalog.cpp` (device_bytes =
13,369,344 for compact routed records; shared FP8 expansion keeps 25.2 MB),
`expert_record.cpp` (admission validates the per-source device size),
`deepseek_worker.cpp` (per-layer warm cap derives from the VRAM entry
budget, clamped 6..32; the 43x6 hard cap is gone) and the residency tool
mirrors. Supporting measurement — route-skew from the persisted census
(`work/deepseek-census/census_skew.py`, artifact `census-skew.json`):
only 2,103/11,008 experts ever observed; top 13.1% cover 92.5% of route
mass; per-layer top-33 (12.9%) covers a median 93.1%; token-to-token
consecutive reuse 27.7%. Caveat: that skew is cumulative over repeated
probe traffic; fresh-topic turns route much wider.

3-turn bounded probe (same shape as W3, freshly restarted service, MTP on;
decode = generated/(wall - TTFT)):

| Turn | baseline w03 decode | S1-DeepSeek decode | read |
|---:|---:|---:|---:|
| 1 (cold) | 0.46 | 1.05 | 17.9 GiB |
| 2 | 0.52 | 2.00 | 19.7 GiB |
| 3 | 0.47 | 3.50 | 10.1 GiB |

~7.4x on turn 3. Historical MTP ablation (same build, the then-current
`-EnableMtp` launcher option removed): decode
1.31/0.91/0.96 with identical ~5 GiB/token reads — MTP no longer costs
supply (the W5 "neutral" verdict was measured in the int8-slot regime);
drafts 3/turn accepted 2/3, so MTP stays on and is now a 2-3.5x decode
multiplier. Bottleneck decomposition (turn 3, MTP on): cache_storage_wait
25s of 21s wall — SATA misses dominate; VRAM hit ~45-55% on fresh topics;
RAM tier (40 GiB, ~3,000 experts) cannot retain the 1,500-2,500 unique
experts a fresh request touches. The measured per-request supply is ~5
GiB/token vs the 3.21 theoretical (churn + prefetch-ahead reads). The
structural eviction fix from `40f595f` also proved itself here: a
`[retire-busy]` canary fired once (victim with an in-flight pin release on
another stream) — the victim was skipped and the request completed;
pre-fix that interleaving was a server hang.

At the S1 checkpoint, the hardware paths toward ~10 tok/s were:

- NVMe tier for the pack: the SATA miss path (0.47 GiB/s measured) is the
  dominant wait; NVMe (~5 GB/s) is ~10x on exactly that term and shortens
  warm start.
- RAM >= 192 GB: the whole 147 GB routed pack becomes RAM-resident, the
  disk leaves steady state entirely, every miss costs only PCIe
  (12.46 GiB/s). With VRAM hit ~50% and MTP on: ~8-12 tok/s effective.
- VRAM is fixed at 24 GB on this host; FP4 slots (12.75 MiB) instead of
  int8 (25.2 MB) are why residency doubled without new hardware.

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

### Priority tiers and indexed supply (2026-08-11)

The follow-on activated demand/prefetch/warm priority, host-only preload,
protected/probationary RAM, transient/protected VRAM, census warm-up,
prefill-route placement and parallel MTP acquisition. The complete cache path
now reports priority-attributed tier hits, waits, bytes, reload/reread traffic,
preload usefulness, occupancy, pins, staging and task-selection cost. A bounded
exact-route trace drives an offline LRU/Belady oracle.

Disabling synchronous DeepSeek CPU hybrid raised the settled identical-prompt
result from about 2.1 to 4.41–4.56 tok/s. Replacing full-map task discovery
(about 342 million inspected entries and ~4.0 s/request) with exact
priority-ordered read/upload key sets then produced:

| Probe | Result | Movement evidence |
|---|---:|---|
| settled identical prompt, 24 output tokens | 6.54–6.69 tok/s | zero SSD; about 47.8 GB H2D/request |
| eight fixed novel prompts, 12 output tokens | 321.47 s total, 10.4% faster | read/reread/H2D unchanged at 97.156/58.558/315.102 GiB |
| novel prompts before task indexing | 0.38–0.69 tok/s, mean 0.57 | 8.2–16.8 GiB SSD/request |
| retained six-turn suite before task indexing | 0.59–0.86 tok/s, mean 0.73 | 10.3–14.0 GiB SSD on turns 2–6 |

The task-index speedup therefore changed control overhead, not placement
policy. Eviction still spends about 0.96 s per settled request scanning the
entry map; two attempted residency indexes were slower and removed. Wider
prefill protection and frequency promotion also regressed the matched novel
suite and were removed. Full implementation detail and rejected variants are
preserved in [the executed 30 tok/s plan](inference-30toks-plan.md#indexed-supply-and-exact-route-follow-up).

## Context and output limits

The deployed API advertises 65,536 context tokens and up to 8,192 output tokens.
Those are admission ceilings, not throughput evidence. Qwen KV is paged and
allocated on demand, so a short request does not physically allocate 65K KV.

Only 4,096 tokens have completed the historical long-context correctness gate.
Claims at 8K–65K require separate numerical, memory, TTFT and decode evidence.

## Current performance verdict

- Qwen native hot paths exceed 30 tok/s.
- Qwen repeated-route API decode reaches 39.6–45.6 tok/s with the FP4 pack
  (§S1b).
- Qwen novel-route chat is SATA first-touch bound at 5.3–16.9 tok/s FP4. The
  30 tok/s SLO is met on resident routes only.
- DeepSeek progressed from ~0.4–0.6 tok/s (W3/W5), through ~3.5 tok/s at
  S1-DeepSeek, to 6.54–6.69 tok/s on a settled identical-prompt workload after
  tier separation and indexed supply. Novel/retained suites remain below one
  tok/s in the cited matched probes, so the SLO stays out of reach.
- `ready=true` means healthy/admitting, not warmed or SLO-compliant.
- The 2026-08-11 three-model VM gate above is the newest lifecycle evidence,
  but it did not rerun the controlled performance suites and does not supersede
  their scoped numbers.

The acceptance target remains at least 30 useful output tok/s for a declared
workload. Any future claim must state model, prompt/history distribution,
cold/warm state, single/aggregate scope, TTFT, post-first-token rate and
end-to-end rate.
