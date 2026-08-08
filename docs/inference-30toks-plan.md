# Path to ±30 tok/s on the 3090 host — hypotheses and solutions

Status: work plan, 2026-08-08 (v3 — verified root causes, flat work list).
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
(line-framed BEGIN/STEP/END protocol, one command at a time under a lock,
`expert_server.py:303-310`). All compute is in the worker
(`runtime/tools/qwen3_next_runner.cpp`, `runtime/tools/deepseek_worker.cpp`).

## 2. Physics first (what is and is not possible)

Bytes that must move per generated token on the critical path:

- Qwen3-Next-80B: 48 layers × 10 routed experts ≈ **1.41 GiB/token**
  (docs/performance-architecture-review.md). At 30 tok/s that is ~42 GiB/s —
  feasible only if the routed working set is almost entirely VRAM/RAM
  resident; SATA (~0.5 GB/s) gives ~0.3 tok/s, cold DDR4 (~40 GB/s) gives
  ~1–2 tok/s, hot-resident DDR4 ~15–25, VRAM-resident 40+.
- DeepSeek-V4-Flash (284B total / 13B active, external specs:
  [vLLM recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)):
  43 × 6 records ≈ **3.21 GiB/token**. 30 tok/s would need ~96 GiB/s of
  routed-expert bandwidth — beyond this host's RAM even fully resident.
  **30 tok/s for DeepSeek-V4-Flash on this box is physically infeasible;
  the realistic ceiling is ~8–15 tok/s**, and that only after fixing the
  supply pipeline (W3) and enabling MTP (2–3× on eligible spans).
- Consequence for future models (Kimi K3): the deciding spec is **active
  parameters per token**, not total. ~3B active fits the 30 tok/s envelope
  on this host; ~13B active does not. Gate every candidate model on
  `active_bytes_per_token × 30 ≤ achievable_memory_bandwidth`.

## 3. Verified root causes (code-checked, file:line)

**R1 — Prefill is tied to batch capacity, and production runs capacity 1.**
`Runner::prefill` chunks the prompt by `capacity_` tokens
(`runtime/tools/qwen3_next_runner.cpp:824`); `prefill_chunk_tokens()`
returns `capacity_` (line 868); `capacity_` is the worker batch capacity,
which `expert_server.py` passes as `--worker-capacity` (default 1,
`expert_server.py:1377`) and `ops/model.sh:153` deploys with
`-WorkerCapacity 1`. A 2,000-token history therefore costs 500–2,000 full
48-layer forward passes instead of a handful of wide ones. This alone
explains TTFT 5–28 s, and every re-prefill replays the whole expert
working set through the cache, evicting useful residents (feeds R3).

**R2 — No session retention: full-history re-prefill every HTTP turn.**
The client resends the entire retained history each turn
(docs/deployment.md:172); the server creates a fresh worker slot per
request and discards state at END (`expert_server.py:828-866`). No
prefix reuse anywhere in the protocol.

**R3 — Synchronous expert demand-paging on the per-token path.**
Qwen: `pin_or_collect_misses` does 5 D2H `cudaMemcpyAsync` +
`cudaStreamSynchronize` **per layer per token even when the route is fully
resident** (`runtime/src/cuda/expert_directory.cu:746-765`), and on miss
the forward blocks in `wait_for` per missing expert
(`qwen3_next_runner.cpp:1376-1401`); the uploader H2D-syncs under the
shared stream lock (`runtime/src/cuda/expert_uploader.cu:411`).
`settle_placement()` is only called by benchmark mains
(`qwen3_next_runner.cpp:2080,2328`), never in worker mode, so routing
feedback D2H copies also run every layer (1321-1326): ~48+ hard syncs per
token. DeepSeek: the FFN cannot resume until the **entire exact top-6
route** is resident (`runtime/src/cuda/deepseek_scheduler.cpp:586-591`).

**R4 — DeepSeek supply pipeline structurally capped (its 0.3–0.6 tok/s).**
Prefetch is limited to 1 in-flight prediction and **suspended while any
demand acquire is in flight** (`deepseek_scheduler.cpp:349-352`, config
`runtime/tools/deepseek_worker.cpp:477-479`) — i.e. almost always; 2 IOCP
threads and 4 staging buffers (`deepseek_worker.cpp:381,396-398`); a 512
MiB MTP reserve is charged **even when MTP is disabled**
(`deepseek_worker.cpp:339`, subtracted at 419-420); every 13.4 MB record
is memcpy'd for RAM retention before first upload
(`runtime/src/expert_cache.cpp:659-663`); eviction does linear scans under
one mutex; spin-yield scheduler loop.

**R5 — Byte-at-a-time INT8 GEMV on the hot GPU path (Qwen ceiling).**
`gemv_batch` (including batch 1) dispatches the non-vectorized
`int8_gemv_batch_kernel` — one byte per thread per iteration — while
`char4`/`float4` vectorized kernels exist in the same file but are only
reachable via other entry points
(`runtime/src/cuda/transformer_kernels.cu:163-164`, dispatch at 1171-1184).
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
  15 tok/s on DeepSeek.
- **moe-stream ([GOBA-AI-Labs/moe-stream](https://github.com/GOBA-AI-Labs/moe-stream)):
  same rejected SSD-streaming class, but its entropy-based dynamic-K
  (fewer experts per token when the router is confident) is a legitimate
  bytes/token reduction worth borrowing.

## 5. Work plan (all items required, ordered by dependency)

**W0 — measure first.** The runner already has phase telemetry
(`phase_.forward_wall_ns`, `TelemetrySnapshot`, `phase_telemetry()` at
`qwen3_next_runner.cpp:836-855`). Expose the per-phase breakdown through
the worker STATS command and log it per request in `expert_server.py`.
Without per-phase numbers no later gain is provable. Done when: per-request
metrics show the prefill/decode/wait-for-experts/upload split.

**W1 — session retention + decoupled prefill (the big win).**
Two changes that compound:
- (a) Server-side session map: keep the worker slot and its KV/state per
  conversation; BEGIN takes a session id and only the *delta* tokens of a
  turn. Change sites: `expert_server.py` `generate` (828-866), worker
  protocol (BEGIN extension), `deepseek_worker.cpp` request-state
  retention.
- (b) Decouple prefill chunking from batch capacity: prefill in wide
  chunks with per-position causal masking regardless of `capacity_`
  (change site: `qwen3_next_runner.cpp:816-833`; the protocol check at
  `expert_server.py:263` must stop requiring
  `prefill_chunk_tokens <= requested_capacity`).
Done when: multi-turn chat TTFT is flat vs history length on the
benchmarks.md multi-turn suite.

**W2 — Qwen hot path (decode ceiling).**
- (a) Kill the per-layer planning syncs on resident routes: async plan
  with completion event — the pattern already exists as `wait_plan_async`
  used by DeepSeek (`runtime/src/cuda/deepseek_decode.cpp:685`); apply it
  in `expert_directory.cu:700-810`, read results one layer later
  (software pipeline).
- (b) Freeze placement in worker mode: call `settle_placement()` after a
  warmup boundary and skip the routing-feedback D2H copies once frozen
  (`qwen3_next_runner.cpp:1321-1326`).
- (c) Route batch-1 `gemv_batch` to the vectorized kernel
  (`transformer_kernels.cu:1171-1184` dispatch).
Done when: telemetry shows wait/sync ≈ 0 on resident routes and the API
path (not just the benchmark mains) reproduces the hot-native rate.

**W3 — async expert supply (the structural fix, both models).**
- Overlap acquire with compute: while layer i computes, upload layer i+1's
  predicted misses; completion events instead of `cudaStreamSynchronize`
  in `expert_uploader.cu`.
- DeepSeek: parallel demand reads, prefetch during demand acquire (remove
  the suspension at `deepseek_scheduler.cpp:351`), widen the pipeline
  beyond 1 in-flight, drop the MTP reserve when MTP is off
  (`deepseek_worker.cpp:339`), skip the retention memcpy for pack-resident
  records (`expert_cache.cpp:659-663`).
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
