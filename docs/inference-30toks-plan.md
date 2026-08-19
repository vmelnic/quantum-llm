# Path to ±30 tok/s on the 3090 host — hypotheses and solutions

Status: executed, 2026-08-11 (v5 — W0–W5 and the DeepSeek tier-separation
follow-on landed; current measurements are in §10).
This is a historical execution record, not the active backlog. Resume current
work from [MoE VM current state and remaining work](moe-vm-next.md).
Follow-on work (FP4 routed experts S1a–S1c landed and measured 2026-08-10;
the model-specific S2/S3 items were left unexecuted) is recorded in
docs/inference-scaling-next.md. Line references
below were taken at audit time; the named symbols are the stable anchors.
Inputs: code audit of the serving path (runtime/, ops/python/,
ops/windows/), docs/benchmarks.md measured numbers, and published results
for comparable stacks. Hardware baseline: RTX 3090 24 GB, Ryzen 5 5600,
64 GB DDR4, SATA SSD (docs/benchmarks.md).

## 1. Where we were at plan time (pre-W0 baseline, measured)

These were the pre-campaign numbers the plan was written against; current
measured results are in docs/benchmarks.md and
docs/inference-scaling-next.md.

- Qwen3-Next-80B expert pack: 47.67 tok/s hot native single-stream, ~30
  tok/s repeated-identical-route API, **0.7–1.4 tok/s real multi-turn
  chat**, TTFT 5–28 s growing with history (docs/benchmarks.md).
- DeepSeek-V4-Flash worker bundle: **0.3–0.6 tok/s**, documented as "not
  performance-ready".
- Both models: KV/state caches exist and work within a request; nothing is
  recomputed per token. The 40× gap between hot-native and chat is entirely
  serving-path overhead.

The serving stack: `ops/python/expert_server.py` (stdlib http.server,
tokenization + admission + SSE) drives a C++ CUDA worker over pipes
(line-framed BEGIN/STEP/END protocol, one command at a time under a lock —
the `WorkerClient` command path in `expert_server.py`). All compute is in
the worker
(`runtime/tools/qwen3_next_runner.cpp`, `runtime/tools/deepseek_worker.cpp`).

## 2. Physics first (what is and is not possible)

Bytes that must move per generated token on the critical path:

- Qwen3-Next-80B: 48 layers × 10 routed experts ≈ **1.41 GiB/token**
  (docs/performance-architecture-review.md). At 30 tok/s that is ~42 GiB/s —
  feasible only if the routed working set is almost entirely VRAM/RAM
  resident; SATA (measured 0.47 GiB/s) gives ~0.3 tok/s, cold DDR4
  (~40 GB/s) gives ~1–2 tok/s, hot-resident DDR4 ~15–25, VRAM-resident 40+.
  The tier numbers were later pinned by measurement: H2D pinned 12.46 GiB/s
  (PCIe Gen3 x16), SATA sequential 0.47 GiB/s — see docs/benchmarks.md
  §Host bandwidth and the recalculated ladder in
  docs/inference-scaling-next.md §3.
- DeepSeek-V4-Flash (284B total / 13B active, external specs:
  [vLLM recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)):
  43 × 6 records ≈ **3.21 GiB/token**. 30 tok/s would need ~96 GiB/s of
  routed-expert bandwidth — beyond this host's RAM even fully resident.
  **30 tok/s for DeepSeek-V4-Flash on this box is physically infeasible.**
  The original 8–15 tok/s estimate assumed a much higher routed-byte hit rate
  and a 2–3× MTP gain; neither assumption held in the measured workload.
  Post-execution correction (§8, superseded by §10): MTP was measured
  throughput-neutral while CPU hybrid dispatch and expert movement dominated.
  Tier separation plus GPU-only routed execution later reached 4.41–4.56 tok/s
  on a settled route. The 3.21 GiB/token arithmetic wall still rules out
  30 tok/s on this host.
- Consequence for future models (Kimi K3): the deciding spec is **active
  parameters per token**, not total. ~3B active fits the 30 tok/s envelope
  on this host; ~13B active does not. Gate every candidate model on
  `active_bytes_per_token × 30 ≤ achievable_memory_bandwidth`.

## 3. Verified root causes (code-checked, file:line)

**R1 — Prefill is tied to batch capacity, and production runs capacity 1.**
`Runner::prefill` chunks the prompt by `capacity_` tokens
(`runtime/tools/qwen3_next_runner.cpp`); `prefill_chunk_tokens()`
returns `capacity_`; `capacity_` is the worker batch capacity,
which `expert_server.py` passes as `--worker-capacity` (default 1)
and `ops/model.sh` deploys with `-WorkerCapacity 1` for DeepSeek
(`-WorkerCapacity 4` for Qwen). A 2,000-token history therefore costs
500–2,000 full 48-layer forward passes instead of a handful of wide ones.
This alone explains TTFT 5–28 s, and every re-prefill replays the whole
expert working set through the cache, evicting useful residents (feeds R3).

**R2 — No session retention: full-history re-prefill every HTTP turn.**
The client resends the entire retained history each turn
(docs/deployment.md §4); the server creates a fresh worker slot per
request and discards state at END (`generate` in `expert_server.py`). No
prefix reuse anywhere in the protocol.

**R3 — Synchronous expert demand-paging on the per-token path.**
Qwen: `pin_or_collect_misses` does 5 D2H `cudaMemcpyAsync` +
`cudaStreamSynchronize` **per layer per token even when the route is fully
resident** (`runtime/src/cuda/expert_directory.cu`), and on miss
the forward blocks in `wait_for` per missing expert
(`run_moe` in `qwen3_next_runner.cpp`); the uploader H2D-syncs under the
shared stream lock (`runtime/src/cuda/expert_uploader.cu`).
`settle_placement()` is only called by benchmark mains
(`qwen3_next_runner.cpp`), never in worker mode, so routing
feedback D2H copies also run every layer: ~48+ hard syncs per
token. DeepSeek: the FFN cannot resume until the **entire exact top-6
route** is resident (`runtime/src/cuda/deepseek_scheduler.cpp`).

**R4 — DeepSeek supply pipeline structurally capped (its 0.3–0.6 tok/s).**
Prefetch is limited to 1 in-flight prediction and **suspended while any
demand acquire is in flight** (`deepseek_scheduler.cpp`, config in
`runtime/tools/deepseek_worker.cpp`) — i.e. almost always; 2 IOCP
threads and 4 staging buffers (`deepseek_worker.cpp`); a 512
MiB MTP reserve is charged **even when MTP is disabled**
(`deepseek_worker.cpp`); every 13.4 MB record
is memcpy'd for RAM retention before first upload
(`runtime/src/expert_cache.cpp`); eviction does linear scans under
one mutex; spin-yield scheduler loop.

No CUDA graphs anywhere. INT8 dequant is fused in the GEMV kernels.

## 4. External landscape (measured on the same wall)

- **AirLLM ([lyogavin/airllm](https://github.com/lyogavin/airllm)):
  rejected.** Layer-shard streaming from SSD is the *fitting* solution
  class (70B on 4 GB VRAM), not the *speed* one — streaming is precisely
  the ~1 tok/s regime we are leaving.
- **ktransformers ([kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)):
  the reference architecture for MoE on consumer GPUs** — attention +
  shared experts on GPU, routed experts in RAM, router-aware hot-expert
  caching in VRAM. Reported: DeepSeek-R1 ~12 tok/s on one 3090
  ([discussion #959](https://github.com/kvcache-ai/ktransformers/discussions/959));
  V4-Flash 24 GB + 256 GB RAM ≈ 5–15 tok/s. Confirms the R3/R4 direction
  and the DeepSeek ceiling.
- **llama.cpp `--n-cpu-moe` / ik_llama.cpp `-ser`** ([comparison](https://github.com/noonghunna/club-3090/blob/master/docs/INFERENCE_ENGINES.md)):
  Qwen3-Next-80B real numbers — 1×3090 ≈ 1.3 tok/s naive, **2×3090 ≈ 46
  tok/s** ([modelfit.io](https://modelfit.io/blog/claude-code-local-llm-setup/));
  4090 + naive CPU offload ≈ 2.7–3.3 tok/s
  ([genesis-kernel](https://github.com/Anuar81/genesis-kernel)). Naive
  offload ≈ 1 tok/s; resident/cached working set ≈ tens of tok/s.
- **MTP speculative decoding** (DeepSeek bundle v3 ships MTP and production
  enables it): the original expectation was 2–3× on eligible spans.
  Post-execution it was throughput-neutral in this bandwidth-bound regime
  (§8 W5); it is kept on because it is correct and self-suppressing.
- **moe-stream ([GOBA-AI-Labs/moe-stream](https://github.com/GOBA-AI-Labs/moe-stream)):
  same rejected SSD-streaming class, but its entropy-based dynamic-K
  (fewer experts per token when the router is confident) is a legitimate
  bytes/token reduction worth borrowing.

## 5. Work plan (all items required, ordered by dependency)

**W0 — measure first.** The runner already has phase telemetry
(`phase_.forward_wall_ns`, `TelemetrySnapshot`, `phase_telemetry()` in
`qwen3_next_runner.cpp`). Expose the per-phase breakdown through
the worker STATS command and log it per request in `expert_server.py`.
Without per-phase numbers no later gain is provable. Done when: per-request
metrics show the prefill/decode/wait-for-experts/upload split.

**W1 — session retention + decoupled prefill (the big win).**
Two changes that compound:
- (a) Server-side session map: keep the worker slot and its KV/state per
  conversation; BEGIN takes a session id and only the *delta* tokens of a
  turn. Change sites: `expert_server.py` `generate`, worker
  protocol (BEGIN extension), `deepseek_worker.cpp` request-state
  retention.
- (b) Decouple prefill chunking from batch capacity: prefill in wide
  chunks with per-position causal masking regardless of `capacity_`
  (change site: `Runner::prefill` in `qwen3_next_runner.cpp`; the
  ready-handshake check in `expert_server.py` must stop requiring
  `prefill_chunk_tokens <= requested_capacity`).
Done when: multi-turn chat TTFT is flat vs history length on the
benchmarks.md multi-turn suite.

**W2 — Qwen hot path (decode ceiling).**
- (a) Kill the per-layer planning syncs on resident routes: async plan
  with completion event — the pattern already exists as `wait_plan_async`
  used by DeepSeek (`runtime/src/cuda/deepseek_decode.cpp`); apply it
  in `expert_directory.cu`, read results one layer later
  (software pipeline).
- (b) Freeze placement in worker mode: call `settle_placement()` after a
  warmup boundary and skip the routing-feedback D2H copies once frozen
  (`run_moe` in `qwen3_next_runner.cpp`).
- (c) Route batch-1 `gemv_batch` to the vectorized kernel
  (`transformer_kernels.cu` dispatch).
Done when: telemetry shows wait/sync ≈ 0 on resident routes and the API
path (not just the benchmark mains) reproduces the hot-native rate.

**W3 — async expert supply (the structural fix, both models).**
- Overlap acquire with compute: while layer i computes, upload layer i+1's
  predicted misses; completion events instead of `cudaStreamSynchronize`
  in `expert_uploader.cu`.
- DeepSeek: parallel demand reads, prefetch during demand acquire (remove
  the suspension in `deepseek_scheduler.cpp`), widen the pipeline
  beyond 1 in-flight, drop the MTP reserve when MTP is off
  (`deepseek_worker.cpp`), skip the retention memcpy for pack-resident
  records (`expert_cache.cpp`).
Done when: novel-route chat sustains bandwidth-bound rates (Qwen 10–25
tok/s, DeepSeek 5–10) without TTFT spikes.

**W4 — router-aware hot-expert cache (ktransformers pattern).**
Track per-expert reuse from the existing placement telemetry
(`phase_telemetry()` exposes useful/wasted prefetch counters); retain hot
experts in VRAM preferentially; evaluate dynamic-K (drop low-weight
experts when router confidence is high, the moe-stream idea) to cut
bytes/token directly.
Done when: resident-hit rate ≥ 0.8 across the benchmarks.md chat traces
and Qwen sustains ~30 tok/s on typical chats.

**W5 — DeepSeek reality.** W3 applied; MTP is selected from the immutable VM
program when draft/verify operations are present (including the reserve fix in
R4), and the documented target is re-baselined at 8–15 tok/s.

Re-run the docs/benchmarks.md suite after every item against the pre-fix
baseline.

## 6. Verdicts per model

- **Qwen3-Next-80B: 30 tok/s conditionally feasible.** Conditions: W1+W2
  (serving path stops losing 40×) then W3+W4 (working set stays resident).
  The 47.67 tok/s hot-native number proves the compute side already can.
- **DeepSeek-V4-Flash: 30 tok/s is infeasible here.** The older 8–15 tok/s
  estimate was not demonstrated; the current settled-route result is
  4.41–4.56 tok/s (§10). Further runtime work is justified only when it
  measurably reduces SSD first touches or RAM-to-VRAM bytes, not as generic
  scheduler tuning. For 30 tok/s with a DeepSeek-class model: fewer active
  parameters, substantially more routed VRAM, or multiple GPUs.
- **Future Kimi K3 (or any candidate):** gate adoption on active
  params/token and MTP/speculative support; ~3B active fits this host,
  ~13B does not.

## 7. What not to do

- No SSD/layer-streaming engines (AirLLM, moe-stream) for this target —
  they are the 1 tok/s class by construction.
- No generic uniform `--cpu-offload-gb`-style offload (documented as
  catastrophic for MoE in the comparison above).
- No chasing DeepSeek 30 tok/s on this host.
- No per-turn protocol changes without W0 telemetry — otherwise gains are
  unprovable.

## 8. Measured outcomes (2026-08-09, all committed)

- **W0+W1 (protocol v5 retained sessions, telemetry):** landed. TTFT flat
  3–9 s (was 17–28 s growing with history); multi-turn prefills only the
  per-turn delta. Decode unchanged (~1–3 tok/s, expert-wait 82% of wall).
- **W2 (Qwen hot path):** landed. Async decode plan with completion
  events, placement frozen after warmup, vectorized GEMV. Resident route
  **27.2–27.7 tok/s with 0 disk bytes** (was ~1–2). Multi-turn decode
  +20–30%.
- **W3 (async expert supply):** landed with one measured revert.
  Event-driven uploads (no stream syncs on the hot path), DeepSeek
  prefetch pipeline (4 in-flight, 4 IOCP threads, 8 staging slots, MTP
  reserve charged only with MTP on, retention memcpy at second touch).
  Qwen multi-turn: neutral — first-touch SATA reads dominate and cannot
  be uploaded away; the route-ahead prefetch variant was reverted
  (predicted experts were already RAM-resident; 2–7× VRAM churn).
  DeepSeek: TTFT turn 3 37.4→31.8 s, expert-wait turn 3 −24%.
- **W4 (router-aware re-promotion):** landed as a bounded mechanism, plan
  criterion (resident-hit ≥ 0.8) not met. Frozen placement now re-promotes
  RAM-resident hot experts against a stale-victim byte budget; resident
  route holds 27.8–28.9 tok/s with zero added storage reads; settled-topic
  turns heal (e.g. wall 31.7→22.4 s, storage-wait 57.5%→19.2%); session
  totals are a wash because novel turns sit at the SATA first-touch floor
  and revisits past the 48 GiB RAM tier re-read from disk. Two aggressive
  variants measured and discarded (ping-pong churn).
- **W5 (DeepSeek MTP):** landed; verdict: **MTP is throughput-neutral in
  this regime and W5 did not reach the 8–15 tok/s estimate.** Acceptance
  73–77%, but
  a verify pair pays the union of two adjacent routes through the same
  bandwidth-bound pipe (reads slightly more), so accepted drafts amortize
  only the fixed per-step cost, which is not the bottleneck at ~2 s/token.
  Measured: 0.47–0.54 → 0.49–0.59 tok/s. Two real bugs fixed en route:
  premature speculation suppression (EMA seeded from 2 samples) and
  speculative-bonus poisoning of retained sessions (new STEP "hold" mode).
  Deployment keeps MTP on: correct, neutral, self-suppressing.

**State at the end of W5.** Qwen3-Next-80B met the target on the resident
route (~27–29 tok/s); non-resident chat remained SATA-first-touch-bound
(~3 tok/s). DeepSeek-V4-Flash was ~0.5 tok/s at this point. This paragraph
is historical; the later tier-separation and GPU-only measurements are in
§10.

## 9. DeepSeek tier-separation follow-on (implemented, 2026-08-10)

The S1-DeepSeek result later reached roughly 3.5 tok/s on settled turns but
still showed SATA first-touch and RAM/VRAM churn on new topics. A subsequent
code audit found that prediction could only move RAM-resident records to VRAM,
the census warm path was bounded by VRAM, the existing VRAM transient class was
disabled, MTP misses were acquired serially, and the cache did not attribute
reloads or work by request priority. The following structural changes are now
implemented:

- `ExpertCache` has distinct demand, prefetch, and warm priorities; demand is
  scheduled first, and the shared eight-slot staging pool reserves six slots
  globally for demand across the main and MTP caches.
- A host-only preload API admits authenticated records from storage into a
  bounded RAM tier without reserving or uploading VRAM. Device demand still
  reserves VRAM before starting its own I/O; demand joining an in-flight warm
  read upgrades that same read and reserves VRAM at upload admission.
- RAM is split into bounded probationary and protected classes. First-touch
  DeepSeek demand is retained in probationary RAM and reuse raises its bounded
  eviction temperature. The census warms at most 32 experts per layer into
  protected RAM (1,376 records, about 17.1 GiB for the current compact record
  geometry) after the worker publishes ready. Warm production stops while
  demand is active and is capped at two staging slots.
- The existing VRAM transient/resident mechanism is active. The transient ring
  is sized from the retained-route bound, census/session-hot records can be
  protected, route feedback now reaches the cache for every completed layer,
  and eviction skips every route-pinned or asynchronously busy victim before
  considering the next candidate.
- Prompt routes are counted during prefill and the top six per layer are
  promoted, within the bounded class budgets, for the following decode span.
  MTP launches all exact draft misses before waiting instead of serializing
  acquire/wait pairs. The main verify scheduler admits up to the 12-expert pair
  union in one acquisition round, subject to the physical staging bound.
- The cancelled-prefetch requeue bug is fixed. Transition prediction remains
  RAM-to-VRAM only: speculative disk-to-RAM prediction is intentionally off
  until the new byte and wait telemetry proves that its saved demand wait is
  larger than its added SATA traffic.

Instrumentation now covers the complete SSD -> staging -> validation/copy ->
RAM -> upload -> VRAM path: tier hits and misses by priority, waiter and stage
latency, bytes, reload/reread traffic, useful/wasted preload, RAM/VRAM class
occupancy and eviction, references/pins, every cache-state transition, global
staging arbitration, warm-loop behavior, prefill placement, task selection,
eviction/admission scan cost, mutex wait, and a separate MTP cache breakdown.
An opt-in bounded exact-route JSONL trace records ordinary and verify-pair
routes plus the initial RAM/VRAM seed; `deepseek_route_oracle.py` simulates LRU
and offline Belady bounds without changing serving. The server includes all
cumulative counters in per-request deltas; occupancy and high-water values
remain gauges.

This design is budget-driven, not total-model-size-driven: a 1 TB model uses
the same mechanisms, while its configured RAM/VRAM budgets determine which
working set is resident. It does not imply that a 1 TB pack fits this host or
that arbitrary first touches avoid storage.

Verification gate: the complete Release MSVC/CUDA build passed on the RTX 3090
host, all four CTest targets passed, and all 54 compiler/server Python tests
passed. The service benchmarks for this follow-on are recorded below.

## 10. DeepSeek measured result and remaining limits (2026-08-11)

The deployed configuration is a demand-streamed model, not a whole-model
load: the 11,008 routed records remain in the pack and each layer requests its
exact top six. The 48 GiB RAM and 13 GiB routed-VRAM budgets are cache tiers.
After ready, the census validates 1,376 records (about 17.1 GiB) into protected
RAM and promotes the hottest 658 (about 8.2 GiB) into resident VRAM. A 1 TB
model uses the same mechanism; only the hit rate and first-touch frequency
change as its pack grows.

The largest measured improvement came from disabling DeepSeek CPU hybrid
dispatch. GPU phase instrumentation showed that the planner selected CPU
execution 4,187 times and accumulated 41.22 s of synchronous CPU compute; the
planner assumed overlap that the actual scheduler cannot provide after exact
GPU acquisition. GPU-only dispatch reduced the settled identical-prompt wall
time from about 21.8 s to 10.1–9.7 s and raised decode from about 2.1 tok/s to
4.41–4.56 tok/s (roughly 2.1x). It is now the production default; profiling is
off and CPU hybrid remains an explicit diagnostic option.

The existing `work/fp4-s1/fp4_bench_probe.py` produced the following results
after a fresh start and completed census warm-up:

| Mode | Decode result | TTFT | Storage and transfer evidence |
| --- | ---: | ---: | --- |
| `resident`, four identical prompts, 24 output tokens | 0.47, 1.46, 4.41, **4.56 tok/s** | 47.47, 5.55, 4.88, **4.69 s** | Requests 3–4 read 0 bytes from SSD but uploaded 49.8/48.4 GB RAM-to-VRAM. |
| `novel`, eight unrelated prompts, 12 output tokens | **0.38–0.69 tok/s** (mean 0.57) | 14.75–38.00 s | 8.2–16.8 GiB SSD and 31.0–47.1 GiB H2D per request. |
| `suite`, six retained conversation turns, 12 output tokens | **0.59–0.86 tok/s** (mean 0.73) | 26.94–47.86 s | Turns 2–6 resumed the retained KV prefix, but still read 10.3–14.0 GiB SSD and uploaded 36.0–40.5 GiB H2D. |

The probe's legacy `storage_read_gib` annotation does not recognize the newer
`cache_read_bytes` key and prints zero. The byte figures above come from the
matching server `request_telemetry` records, not that stale annotation.

At the pre-index checkpoint, the settled-route bottleneck was no longer SATA
or CPU. Request 4 read
zero storage bytes, uploaded 48.38 GB over 40 model steps, and spent 6.29 s of
its 9.72 s wall time in scheduler expert wait (5.28 s in upload wait). That is
about 1.21 GB H2D per model step even after cache hits. At the measured 12.46
GiB/s pinned-H2D ceiling, the same byte rate alone bounds execution near 11
model steps/s before compute and control overhead; the observed end-to-end
rate was 4.56 tok/s. Reaching 30 tok/s would allow only about 0.42 GiB H2D per
token against 3.21 GiB of active routed weights, requiring at least about 87%
of routed bytes to hit VRAM before counting compute. The current hot trace is
approximately 65% by that arithmetic, and the 13 GiB routed cache cannot hold
arbitrary routes from the roughly 147 GB (137 GiB) routed pack.

### Indexed supply and exact-route follow-up

Task-selection instrumentation found a software bottleneck hidden inside the
expert wait: `next_task_locked` inspected about 342 million map entries and
spent about 4.0 s per settled request while looking for a few actionable
reads/uploads. The cache now maintains exact priority-ordered read and upload
key sets under the existing mutex. It preserves demand/prefetch/warm ordering
and admission policy while reducing the settled request to about 3,600 task
candidates and 0.79 s of inclusive task-selection time.

The matched service results are:

| Probe | Before task indexes | With task indexes | Evidence |
| --- | ---: | ---: | --- |
| Settled identical prompt, 24 output tokens | 4.41--4.56 tok/s | **6.54--6.69 tok/s** | Zero SSD in both arms; about 47.8 GB H2D per request remains. |
| Eight fixed novel prompts, 12 output tokens | 358.78 s total | **321.47 s total** | Read/reread/H2D stayed exactly 97.156/58.558/315.102 GiB, proving that the speedup did not change cache policy. |

This is roughly a 46--51% settled decode improvement and a 10.4% novel-suite
wall-time improvement. Eviction still scans roughly 49 million map candidates
and costs about 0.96 s per settled request. Two residency-index implementations
did not beat the deterministic map scan and were removed; only the validated
task indexes remain.

The exact trace contains 97,111 routed accesses over 365 captured steps. With
the real initial seed, the RAM oracle reports 9,978 LRU misses versus 6,383
offline-Belady misses; VRAM reports 43,928 versus 26,848. These are upper
bounds from future knowledge, not implementable throughput claims, but they
prove that policy headroom remains. Two direct approximations were rejected:

- frequency-based promotion of probationary demand into protected RAM caused
  1,001 promotions and 13.16 GiB of protected eviction on one novel request at
  threshold two; threshold 16 still increased a matched resident cold request
  from 95.5 s/30.9 GiB read to 124.2 s/41.7 GiB;
- widening exact prefill protection improved offline decode coverage from
  17.24% at top six to 21.45% at top eight and 28.26% at top twelve, but top
  twelve caused sustained late-suite churn. Top eight reduced bytes slightly
  yet regressed the matched novel total from 321.47 s to 328.91 s and raised
  storage wait from 385.1 s to 416.2 s. Production therefore remains top six.

Remaining work, ordered by the measured dependency:

1. Replace generic recency only with a policy that distinguishes prefill and
   verify multiplicity from future decode value. The top-eight/top-twelve and
   frequency-promotion experiments above show that a wider hit-count window is
   insufficient even when its raw coverage is higher. Any next policy must be
   evaluated first in the exact oracle and then reduce both bytes and wait in
   the matched service suite.
2. Add budgeted transition prefetch from disk to protected RAM for absent
   experts. It addresses the 8–17 GiB first touches in `novel`; cancel it on
   exact demand and automatically disable it when speculative read bytes rise
   without a corresponding storage-wait reduction. The current hot trace has
   about 1,702 incorrect of 3,440 predictions and schedules zero promotions
   once admission is saturated, so turning those predictions directly into
   SATA reads is not yet safe. NVMe helps this cold path but cannot improve the
   zero-SSD resident result.
3. Expose a cross-layer prediction handoff before current-layer FFN completion,
   then pipeline only RAM-resident next-layer uploads during that FFN. The
   current scheduler receives a ready route at `layer_complete`, after the
   overlap window has closed; merely increasing queue depth cannot create this
   overlap. Demand remains strictly prioritary, and the gate is lower
   `scheduler_expert_wait_ns` with no extra uploaded bytes.
4. Eliminate the failed `BEGIN` round trip for an unrelated new request by
   proactively evicting one unmatched retained worker slot. Reactive LRU
   eviction and retry already work; this is only a correctness-preserving
   control-path cleanup, not a tok/s lever.
5. Keep the validated exact task indexes. `evict_one_locked` still scans the
   full entry map under one mutex, but the measured residual is about 0.96 s per
   settled request; replace it only with a deterministic priority structure
   that beats that number. The attempted set/pointer residency indexes did not
   and were removed. This remains more important for packs much larger than the
   current 11,008 records.
6. Revisit MTP only after expert movement falls. In the current traces it
   usually launches three drafts and accepts zero or one; warming a distinct
   speculative future-route set may help, but the two attempted request-local
   gates increased SSD reads and were reverted.
Dynamic top-K is excluded from this plan because changing routed capacity can
reduce answer quality and increase hallucination. Smaller routed quantization
would likewise require a separate quality gate and is not part of these
cache/scheduler changes.

More RAM can eliminate cold SATA only if it holds the useful routed working
set (roughly 192 GB host RAM would hold this pack with operating headroom).
More routed VRAM or another GPU attacks the settled H2D bottleneck directly.
For larger models, including a 1 TB pack, no whole-model-fit assumption is
required, but arbitrary novel routes become slower unless their working set is
predictable or the storage/RAM tiers grow with it.
