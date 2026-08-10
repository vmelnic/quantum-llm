# Path to ±30 tok/s on the 3090 host — hypotheses and solutions

Status: executed, 2026-08-09 (v4 — W0–W5 landed, measured outcomes in §8).
Follow-on work (FP4 routed experts S1a–S1c landed and measured 2026-08-10;
S2/S3 pending) is tracked in docs/inference-scaling-next.md. Line references
below were taken at audit time; the named symbols are the stable anchors.
Inputs: code audit of the serving path (runtime/, ops/python/,
ops/windows/), docs/benchmarks.md measured numbers, and published results
for comparable stacks. Hardware baseline: RTX 3090 24 GB, Ryzen 5 5600,
64 GB DDR4, SATA SSD (docs/benchmarks.md).

## 1. Where we are (measured, not guessed)

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
  **30 tok/s for DeepSeek-V4-Flash on this box is physically infeasible;
  the realistic ceiling is ~8–15 tok/s**, and that only after fixing the
  supply pipeline (W3) and enabling MTP (2–3× on eligible spans).
  Post-execution correction (§8): W3 landed and MTP was measured
  throughput-neutral in this regime, so even the 8–15 tok/s ceiling is
  unreachable; the 3.21 GiB/token arithmetic wall stands.
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

**R5 — Byte-at-a-time INT8 GEMV on the hot GPU path (Qwen ceiling).**
`gemv_batch` (including batch 1) dispatches the non-vectorized
`int8_gemv_batch_kernel` — one byte per thread per iteration — while
`char4`/`float4` vectorized kernels exist in the same file but are only
reachable via other entry points
(`runtime/src/cuda/transformer_kernels.cu`).
~4× weight-read amplification on every resident-expert GEMV; invisible at
1 tok/s, decisive at 30.

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
- **MTP speculative decoding** (DeepSeek bundle v3 already ships MTP;
  `EnableMtp` off by default): 2–3× on eligible spans; mandatory for even
  15 tok/s on DeepSeek. Post-execution: measured throughput-neutral in
  this bandwidth-bound regime (§8 W5); kept on because it is correct and
  self-suppressing.
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

**W5 — DeepSeek reality.** W3 applied, MTP enabled (`EnableMtp`, plus the
reserve bug in R4), documented target re-baselined at 8–15 tok/s.

Re-run the docs/benchmarks.md suite after every item against the pre-fix
baseline.

## 6. Verdicts per model

- **Qwen3-Next-80B: 30 tok/s conditionally feasible.** Conditions: W1+W2
  (serving path stops losing 40×) then W3+W4 (working set stays resident).
  The 47.67 tok/s hot-native number proves the compute side already can.
- **DeepSeek-V4-Flash: 8–15 tok/s realistic, 30 infeasible here** — 3.21
  GiB/token fails bandwidth arithmetic before any code is written. For 30
  with a DeepSeek-class model: fewer active parameters or a second GPU.
  Post-execution (§8 W5): the 8–15 band proved unreachable too; measured
  ~0.5 tok/s with MTP on. No further runtime investment is justified on
  this host.
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
  this regime, 8–15 tok/s confirmed unreachable.** Acceptance 73–77%, but
  a verify pair pays the union of two adjacent routes through the same
  bandwidth-bound pipe (reads slightly more), so accepted drafts amortize
  only the fixed per-step cost, which is not the bottleneck at ~2 s/token.
  Measured: 0.47–0.54 → 0.49–0.59 tok/s. Two real bugs fixed en route:
  premature speculation suppression (EMA seeded from 2 samples) and
  speculative-bonus poisoning of retained sessions (new STEP "hold" mode).
  Deployment keeps MTP on: correct, neutral, self-suppressing.

**Final state.** Qwen3-Next-80B meets the target on the resident route
(~27–29 tok/s); non-resident chat is SATA-first-touch-bound (~3 tok/s).
DeepSeek-V4-Flash stays ~0.5 tok/s; no runtime change beats 3.21
GiB/token over this host's bandwidth. For 30 tok/s with a
DeepSeek-class model: fewer active bytes/token or faster storage/RAM —
not more scheduler work.
