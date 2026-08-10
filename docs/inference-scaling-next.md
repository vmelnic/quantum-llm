# Scaling MoE serving beyond the 30 tok/s wall — next steps

Status: research synthesis, 2026-08-09; updated 2026-08-10 with measured
host bandwidth (§3) and S1a/S1c code status (§4). Inputs: internal
pack-format audit, measured W0–W5 results (docs/inference-30toks-plan.md
§8, docs/benchmarks.md), and published low-bit MoE evidence.

Premise: total model size is not the speed constraint — **routed
bytes/token and cache hit rate are**. A 1 TB model is fine if its
per-token footprint is small and the conversation's working set stays
cached. The plan below therefore does not rely on "shrink the model until
it fits in RAM"; every step helps at any model scale.

## 1. What the internal audit found (pack format, file:line)

- Quant today: `int8-symmetric-per-row-v1`, ABI 1
  (`compiler/expert_pack/constants.py:22-24`), one FP32 scale per output
  row (`compiler/expert_pack/quant.py:77-120`). Since 032cccd the runtime
  validator accepts ABI 1 and ABI 3
  (`runtime/src/expert_record.cpp:87-91`).
- **Dequant is in-kernel, not at upload** (`expert_uploader.cu:481-504`
  copies quantized bytes; `moe_kernels.cu:143-163` dequantizes per
  element). So a 4-bit format halves BOTH storage/RAM footprint AND the
  H2D bytes moved per token.
- The cache/uploader tiers are size-agnostic (manifest-driven
  `stored_bytes`); no tiering logic changes needed for smaller records.
- **A working 4-bit pipeline already exists in this repo**: DeepSeek FP4
  E2M1/UE8M0 block-32 — pack writer, `__dp4a` kernels
  (`moe_kernels.cu:68-118`), format dispatch in the `_selection_batch`
  kernels, direct-compact upload. Since 032cccd the same format is wired
  for Qwen routed experts as ABI 3
  (`compiler/expert_pack/constants.py:32`): compiler encoder, manifest +
  schema + validator, uploader/directory dispatch and packed `__dp4a`
  GEMV kernels — S1a/S1c code landed; S1b measured (2026-08-10): quality
  gate passed, resident 39.6–45.6 tok/s (docs/benchmarks.md §S1b).
- Risk: realizing the bandwidth win needs packed vectorized loads +
  `__dp4a` in the Qwen expert GEMVs; a naive nibble-unpack loop will not
  deliver 2x. The DeepSeek FP4 kernels are the template.

## 2. What the literature says (measured, not marketing)

- **4-bit MoE experts are essentially free in quality** — MoQE
  (arXiv:2310.02410), FineQuant (arXiv:2308.09723), QMoE (arXiv:2310.16795,
  sub-1-bit with minor loss), mixed-bit PTQ (arXiv:2406.08155: 4-bit on
  hot experts is the sweet spot). 2-bit with a dynamic scheme (high-bit
  down_proj/shared/attention) loses ~8-30% on hard tasks, not catastrophic
  (Unsloth DeepSeek-R1 dynamic measurements, empiric).
- **Cross-layer route prediction works on exactly this model class**:
  Fate (Fang et al. 2025) — gate inputs between adjacent layers correlate
  ~99%, prefetch accuracy >97%, 1.91x on Qwen3-30B-A3B in a llama.cpp
  fork; PreScope (arXiv:2509.23638) +141% with 2-layer lookahead; DyMoE
  +2.2x on Mixtral.
- **Spec-decode in the bandwidth-bound regime: our neutral measurement is
  confirmed by two independent sources** (thc1006 on RTX 3090, and
  pestopoppa MTP-1 at 78.5% acceptance -> 0.56x net). Do not invest there
  until hot experts live in VRAM.
- Dynamic top-K reduction (8->6): claimed by ktransformers without
  published quality numbers; treat as hypothesis to measure, not fact.
- HOBBIT (arXiv:2411.01433): mixed precision + token-level loading +
  adaptive layer prefetch, up to 9.93x on edge MoE — the closest
  published design to what this plan proposes.
- Caveat: ktransformers' 13.7 t/s numbers are DDR5 dual-socket servers
  (~10x our DDR4); not extrapolable. tinyserve "30 t/s on 8 GB GPU" is an
  unaudited RFC-author claim.

## 3. The physics ladder after each step

Measured host bandwidth (2026-08-10, `work/fp4-s1/bandwidth_probe.py`,
artifact `work/fp4-s1/bandwidth-report.json` on the GPU host):

- **H2D pinned: 12.46 GiB/s** — PCIe 3.0 x16 (`nvidia-smi`: link max
  Gen3 x16; idles at Gen1 and ramps under load). Pageable copies reach
  the same 12.05 GiB/s via staging. This is the hard ceiling for any
  RAM-tier design on this host.
- **SATA sequential read: 0.47 GiB/s** over the full 72.3 GiB int8 pack
  (larger than RAM, so mostly cold).

Implied per-token ceilings (bandwidth ÷ bytes/token):

| Tier | int8 (1.41 GiB/tok) | FP4 (0.7 GiB/tok) |
|---|---:|---:|
| SATA | 0.3 tok/s | 0.7 tok/s |
| RAM (over measured H2D) | 8.8 tok/s | 17.8 tok/s |
| VRAM-resident (hot native) | 47.7 tok/s | ≥47.7 expected |

Today (int8): SATA floor ~0.35-3 tok/s (real chat lives here), RAM tier
capped at ~8.8 by the bus, VRAM-resident 27-47.

- **S1 FP4 routed experts**: 0.7 GiB/token. Every tier doubles: SATA
  ~0.7, RAM ceiling ~17.8 (NOT 25-40 — the earlier ladder assumed
  17.5-28 GiB/s PCIe that this host does not have). Decisive structural
  effect: the routed FP4 pack ~36 GiB fits inside the 48 GiB RAM budget
  (72.3 GiB does not), so the SATA floor becomes cold-start-only.
- **Consequence for the 30 tok/s target**: 30 tok/s at 0.7 GiB/token
  needs 21 GiB/s H2D — impossible over the measured 12.46 GiB/s bus.
  On this host, 30 tok/s is reachable ONLY through VRAM residency
  (hot native is 47.7), i.e. capacity/hit-rate levers (W4 re-promotion,
  S3 pinning), not bandwidth levers. S2 buys proximity to the ~17.8
  RAM ceiling, not 30.
- **S2 predictive cross-layer prefetch (Fate-style)**: predict layer
  i+1/i+2 routes from the gate *inputs* at layer i (not from history —
  that is why the W3 route-ahead attempt churned and was reverted).
  With S1, a miss is a cheap 0.7 GiB/token H2D that overlaps compute;
  realistic target: real chat pinned near the ~17.8 RAM ceiling instead
  of the SATA floor. Prefetch must carry an admission gate against the
  stale-victim budget (W4 lesson), with churn-byte revert criteria.
- **S3 prefill-touch pinning**: the experts touched while prefilling the
  prompt are the strongest free predictor of the decode span's routes
  (same topic). Pin that set — budget-capped, frequency-weighted — at
  the RAM/VRAM tier boundary before decode starts. No learning, no model
  change, scales to any model size.

Optional later, only if measured supportive: top-K 10->8 with
renormalization (-20% bytes), 2-3-bit cold experts with high-bit
down_proj (-2x more), external-draft verification over K coherent
positions (union-of-routes amortization; contradicts W5 only if route
overlap over K>2 proves high — unmeasured today).

## 4. Work list

Flat list, dependency order, each step gated on measured before/after:

1. **S1a — compiler** ✅ landed (032cccd): FP4-E2M1/UE8M0 block-32
   encoder for Qwen routed experts in `compiler/expert_pack/`, ABI 3,
   manifest + schema + validator updates. Router/norms FP32, dense pack
   int8.
2. **S1b — quality gate** ✅ done (2026-08-10): FP4 pack compiled
   (45.36 GB, ABI 3, valid); weight error FP4 ~11.8% vs int8 ~0.85%
   relative L2 (RTN-expected); behavioral probe 10/10 coherent,
   on-topic, no degeneration; bench — resident steady 39.6–45.6 tok/s
   (int8 same-config 7.75), novel 5.3–16.9 (int8 1.9–4.0), suite
   16.4–25.3 (int8 3.1–5.2). Two runtime bugs fixed to get there:
   pinned-allocator alignment (cudaHostAlloc is not 4096-aligned) and
   the FP4 eviction livelock (route-pinned victim + unbounded retire
   spin; fixed structurally with route_pinned/try_retire). Full
   numbers: docs/benchmarks.md §S1b.
3. **S1c — runtime** ✅ landed (032cccd): FP4 dispatch in the uploader /
   directory, packed `__dp4a` selection-batch kernels for Qwen packs,
   cache-state tests (364 lines). Gate passed by S1b: resident route
   39.6–45.6 tok/s (≥27), close to the 47.67 hot-native ceiling.
4. **S2 — route predictor**: tap gate inputs per layer in
   `qwen3_next_runner.cpp`, predict next-layer routes, prefetch from the
   RAM tier through the existing event-driven uploader. Measure hit rate
   (target >90%) and novel-route tok/s; revert if churn reappears.
5. **S3 — prefill-touch pinning**: record experts touched during prefill,
   pin them for the decode span, release at turn end. Measure settled
   vs novel turn tok/s on the W4 long-session probe.

Not to do: spec-decode on DeepSeek-class bandwidth-bound regimes
(measured neutral, confirmed externally); trusting top-K reduction
claims without a quality measurement; chasing vendor tok/s numbers from
DDR5 dual-socket rigs.
