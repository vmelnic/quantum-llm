# Path to ±30 tok/s on the 3090 host — hypotheses and solutions

Status: analysis, 2026-08-08. Inputs: code audit of the serving path
(runtime/, ops/python/, ops/windows/), docs/benchmarks.md measured numbers,
and published results for comparable stacks. Nothing in this document is yet
implemented. Hardware baseline: RTX 3090 24 GB, Ryzen 5 5600, 64 GB DDR4,
SATA SSD (docs/benchmarks.md).

## 1. Where we are (measured, not guessed)

- Qwen3-Next-80B expert pack: 47.67 tok/s hot native single-stream, ~30
  tok/s repeated-identical-route API, **0.7–1.4 tok/s real multi-turn chat**,
  TTFT 5–28 s growing with history (docs/benchmarks.md).
- DeepSeek-V4-Flash worker bundle: **0.3–0.6 tok/s**, documented as "not
  performance-ready".
- Both models: KV/state caches exist and work within a request; nothing is
  recomputed per token. The gap is elsewhere.

## 2. Physics first (what is and is not possible)

Bytes that must move per generated token on the critical path:

- Qwen3-Next-80B: 48 layers × 10 routed experts ≈ **1.41 GiB/token**
  (docs/performance-architecture-review.md). At 30 tok/s that is ~42 GiB/s —
  feasible only if the routed working set is almost entirely VRAM/RAM
  resident; SATA (~0.5 GB/s) gives ~0.3 tok/s, plain DDR4 (~40 GB/s) gives
  ~1–2 tok/s worst case, hot-resident DDR4 gives ~15–25, VRAM-resident gives
  40+.
- DeepSeek-V4-Flash (284B total / 13B active, external specs:
  [vLLM recipes](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)):
  43 × 6 records ≈ **3.21 GiB/token**. 30 tok/s would need ~96 GiB/s of
  routed-expert bandwidth — beyond this host's RAM even fully resident.
  **30 tok/s for DeepSeek-V4-Flash on this box is physically infeasible;
  the realistic ceiling is ~8–15 tok/s**, and that only after fixing the
  supply pipeline (H1/H4) and enabling MTP (2–3× on eligible spans).
- Consequence for future models (Kimi K3): the deciding spec is **active
  parameters per token**, not total. ~3B active fits the 30 tok/s envelope
  on this host; ~13B active does not. Evaluate every candidate model by
  `active_bytes_per_token × 30 ≤ achievable_memory_bandwidth`.

## 3. Bottleneck hypotheses in our code (ranked, with evidence)

**H1 — Synchronous expert demand-paging on the per-token path (both models;
dominant).** Every cache miss blocks the layer mid-forward: Qwen waits per
missing expert (`runtime/tools/qwen3_next_runner.cpp:1376-1401`), the
uploader does H2D then `cudaStreamSynchronize` under the shared stream lock
(`runtime/src/cuda/expert_uploader.cu:411`), DeepSeek requires the entire
top-6 route resident before FFN resume
(`runtime/src/cuda/deepseek_scheduler.cpp:586-591`). With caches holding
only a fraction of the pack (DeepSeek: 40 GiB RAM + 12 GiB VRAM vs 147 GB
routed pack ≈ 35%), route changes stream from SATA and produce exactly the
observed 0.15–3 tok/s.
Fix direction: event-driven async upload, overlap acquire with compute,
parallel demand reads, prefetch during demand acquire.

**H2 — Full-history re-prefill every HTTP turn (both models; dominant for
chat).** The client resends the entire retained history each turn
(docs/deployment.md:172), the server creates a fresh worker slot per
request (`ops/python/expert_server.py:828-866`), session KV is discarded at
request end, and prefill runs in **4-token chunks**
(`runtime/tools/qwen3_next_runner.cpp:816-868`). Measured TTFT 5.3→27.8 s as
history grows, and every re-prefill replays the whole expert working set
through the cache, evicting useful residents (cache churn feeding H1).
Fix direction: server-side session map with worker-state retention,
protocol-level prefix reuse (BEGIN with a session id instead of full
re-prefill), much larger prefill chunks.

**H3 — Per-layer host synchronization and never-frozen placement (Qwen
decode ceiling).** `pin_or_collect_misses` performs 5 D2H copies +
`cudaStreamSynchronize` per layer per token even on all-resident routes
(`runtime/src/cuda/expert_directory.cu:746-765`); worker mode never calls
`settle_placement()` (only benchmark mains do,
`qwen3_next_runner.cpp:2080,2328`), so routing feedback copies run every
layer. ~48+ hard syncs per token serialize the GPU pipeline.
Fix direction: async plan with completion event (the pattern already exists
as `wait_plan_async` used by DeepSeek at `deepseek_decode.cpp:685`), freeze
placement after a warmup boundary in worker mode.

**H4 — DeepSeek supply pipeline structurally capped (explains 0.3–0.6
tok/s).** Prefetch limited to 1 in-flight prediction and **suspended while
any demand acquire is in flight** (`deepseek_scheduler.cpp:349-352`, config
`runtime/tools/deepseek_worker.cpp:477-479`); only 2 IOCP threads and 4
staging buffers; a 512 MiB MTP reserve is charged **even when MTP is
disabled** (`deepseek_worker.cpp:339`, subtracted at 419-420); every 13.4 MB
record is memcpy'd for RAM retention before first upload
(`runtime/src/expert_cache.cpp:659-663`); eviction does linear scans under
one mutex; spin-yield scheduler loop.
Fix direction: widen prefetch pipeline, allow prefetch during demand, drop
the MTP reserve when MTP is off, avoid the retention memcpy for records
that stay pack-resident.

**H5 — Byte-at-a-time INT8 GEMV on the hot GPU path (Qwen hot-rate
multiplier).** `gemv_batch` (including batch 1) dispatches the non-vectorized
`int8_gemv_batch_kernel` — one byte per thread per iteration
(`runtime/src/cuda/transformer_kernels.cu:163-164`, dispatch at 1171-1184) —
while `char4`/`float4` vectorized kernels exist (lines 64-148) but are
reachable only via other entry points. ~4× weight-read amplification on
every resident-expert GEMV; at the 30 tok/s target this is the difference
between 12 and 40.
Fix direction: route batch-1 to the vectorized kernel; then address the
minor overheads (2 ms decode-batching window, per-token full
`tokenizer.decode`, one pipe round trip per token — negligible now, visible
at 30 tok/s).

## 4. External landscape (what others measured on the same wall)

- **AirLLM ([github.com/lyogavin/airllm](https://github.com/lyogavin/airllm)):
  rejected for our target.** It streams layer shards from SSD to fit 70B on
  4 GB VRAM — that is the *fitting* solution class, not the *speed* one;
  layer streaming over SATA/NVMe is precisely the ~1 tok/s regime we are
  trying to leave.
- **ktransformers ([github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)):
  the reference architecture for MoE on consumer GPUs.** Attention + shared
  experts on GPU, routed experts in system RAM, router-aware **hot-expert
  caching** in VRAM. Reported: DeepSeek-R1 ~12 tok/s on one RTX 3090
  ([discussion #959](https://github.com/kvcache-ai/ktransformers/discussions/959));
  DeepSeek-V4-Flash on 24 GB + 256 GB RAM ≈ 5–15 tok/s. Confirms our H1/H4
  direction and the DeepSeek ceiling.
- **llama.cpp `--n-cpu-moe` / ik_llama.cpp `-ser`:** layer-uniform expert
  offload with per-tensor overrides and smart expert reduction
  ([comparison table](https://github.com/noonghunna/club-3090/blob/master/docs/INFERENCE_ENGINES.md)).
  Real numbers for Qwen3-Next-80B (3B active): 1×3090 ≈ 1.3 tok/s naive,
  **2×3090 ≈ 46 tok/s** ([modelfit.io](https://modelfit.io/blog/claude-code-local-llm-setup/));
  4090 + naive CPU offload ≈ 2.7–3.3 tok/s
  ([genesis-kernel](https://github.com/Anuar81/genesis-kernel)). Confirms:
  naive offload ≈ 1 tok/s; resident/cached working set ≈ tens of tok/s.
- **MTP speculative decoding** (DeepSeek bundle v3 already contains MTP;
  disabled by default, `EnableMtp` switch): 2–3× on eligible spans; required
  to reach even 15 tok/s on DeepSeek. Qwen-side speculative options to be
  evaluated separately.
- **moe-stream ([github.com/GOBA-AI-Labs/moe-stream](https://github.com/GOBA-AI-Labs/moe-stream)):
  SSD-streaming MoE for Mac/Metal — same class as AirLLM, same rejection
  for the 30 tok/s target, but its dynamic-K expert selection idea is
  relevant to reducing bytes/token.

## 5. Verdicts per model

- **Qwen3-Next-80B: 30 tok/s is conditionally feasible on this host.**
  Conditions, in order: H1 (async supply) + H2 (session retention) + H3
  (kill per-layer syncs, freeze placement) + H5 (vectorized GEMV), plus a
  router-aware hot-expert cache sized to keep the measured working set
  ≥80% VRAM/RAM-resident. The 47.67 tok/s hot-native number proves the
  compute side can already do it; the serving path is what loses 40×.
- **DeepSeek-V4-Flash: 30 tok/s is infeasible here; target 8–15.** Path:
  H4 + H1, enable MTP, grow RAM beyond 64 GB (routed pack is 147 GB;
  hot-expert caching per ktransformers). If ±30 tok/s for a
  DeepSeek-class model is a hard requirement, the honest answers are a
  smaller active-parameter model or a second GPU — not more engineering on
  this box.
- **Future Kimi K3 (or any candidate):** gate adoption on
  `active params/token` and available MTP/speculative support; ~3B active
  fits this host, ~13B active does not.

## 6. Recommended order of work

1. H2 (session retention + prefix reuse): largest user-visible win
   (TTFT 5–28 s → sub-second), removes cache churn feeding H1.
2. H3 + H5 (Qwen hot path): cheap, raises the decode ceiling toward the
   native 47 tok/s.
3. H1 (async expert supply, both models): the structural fix; design per
   `wait_plan_async` precedent.
4. Hot-expert cache with router-aware retention (ktransformers pattern).
5. DeepSeek: H4 + enable MTP; re-baseline expectations at 8–15 tok/s.
6. Re-run docs/benchmarks.md suite after each step; keep the pre-fix
   numbers as the baseline.

## 7. What not to do

- Do not adopt SSD/layer-streaming engines (AirLLM, moe-stream) for this
  target — they are the 1 tok/s class by construction.
- Do not buy throughput with generic `--cpu-offload-gb`-style uniform
  offload (documented as catastrophic for MoE in the comparison above).
- Do not chase DeepSeek 30 tok/s on this host; it fails bandwidth
  arithmetic before any code is written.
