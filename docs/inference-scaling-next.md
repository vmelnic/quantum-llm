# Scaling MoE serving beyond the 30 tok/s wall — next steps

Status: research synthesis, 2026-08-09. Inputs: internal pack-format audit,
measured W0–W5 results (docs/inference-30toks-plan.md §8,
docs/benchmarks.md), and published low-bit MoE evidence.

Premise: total model size is not the speed constraint — **routed
bytes/token and cache hit rate are**. A 1 TB model is fine if its
per-token footprint is small and the conversation's working set stays
cached. The plan below therefore does not rely on "shrink the model until
it fits in RAM"; every step helps at any model scale.

## 1. What the internal audit found (pack format, file:line)

- Quant today: `int8-symmetric-per-row-v1`, ABI 1
  (`compiler/expert_pack/constants.py:22-24`), one FP32 scale per output
  row (`compiler/expert_pack/quant.py:77-120`). The runtime validator
  hard-rejects anything else (`runtime/src/expert_record.cpp:88-124`).
- **Dequant is in-kernel, not at upload** (`expert_uploader.cu:481-504`
  copies quantized bytes; `moe_kernels.cu:143-163` dequantizes per
  element). So a 4-bit format halves BOTH storage/RAM footprint AND the
  H2D bytes moved per token.
- The cache/uploader tiers are size-agnostic (manifest-driven
  `stored_bytes`); no tiering logic changes needed for smaller records.
- **A working 4-bit pipeline already exists in this repo**: DeepSeek FP4
  E2M1/UE8M0 block-32 — pack writer, ABI 2, `__dp4a` kernels
  (`moe_kernels.cu:68-118`), format dispatch in the `_selection_batch`
  kernels, direct-compact upload. Reusing that format for Qwen routed
  experts makes the runtime side mostly wiring; the compiler side is new
  calibration/encoding work.
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

Today (int8, 1.41 GiB/token): SATA floor ~0.35-3 tok/s (real chat lives
here), RAM tier ~15-25, VRAM-resident 27-47.

- **S1 int4 routed experts**: 0.7 GiB/token. Every tier doubles: SATA
  ~1-6, RAM ~25-40, and twice the working set fits VRAM-resident (W4
  re-promotion already in place). Pack 72.3 GiB -> ~36 GiB routed.
- **S2 predictive cross-layer prefetch (Fate-style)**: predict layer
  i+1/i+2 routes from the gate *inputs* at layer i (not from history —
  that is why the W3 route-ahead attempt churned and was reverted).
  With S1, a miss is a cheap 0.7 GiB/token H2D that overlaps compute;
  realistic target: real chat pinned near the RAM ceiling instead of the
  SATA floor.
- **S3 prefill-touch pinning**: the experts touched while prefilling the
  prompt are the strongest free predictor of the decode span's routes
  (same topic). Pin that set at the RAM/VRAM tier boundary before decode
  starts. No learning, no model change, scales to any model size.

Optional later, only if measured supportive: top-K 10->8 with
renormalization (-20% bytes), 2-3-bit cold experts with high-bit
down_proj (-2x more), external-draft verification over K coherent
positions (union-of-routes amortization; contradicts W5 only if route
overlap over K>2 proves high — unmeasured today).

## 4. Work list

Flat list, dependency order, each step gated on measured before/after:

1. **S1a — compiler**: int4-groupwise (or reuse FP4-E2M1/UE8M0 block-32)
   encoder for Qwen routed experts in `compiler/expert_pack/`, new ABI id,
   manifest + schema + validator updates. Calibrate from the BF16/FP32
   sources; keep router/norms FP32, dense pack int8.
2. **S1b — quality gate**: perplexity delta vs the int8 pack on a
   reference corpus + the KV-attach probes (Nacre 8/8, RO 4/5 must hold
   with the requantized pack). Literature predicts ~no regression at
   4-bit; if measured regression, stop here.
3. **S1c — runtime**: new device format in the `entry.format` dispatch,
   packed `__dp4a` GEMV kernels mirroring `packed_fp4_q8_dot`, uploader
   byte-count generalization, cache-state tests. Benchmark: resident
   route must stay >=27 tok/s; novel-route chat before/after.
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
