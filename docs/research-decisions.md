# Research decisions

Status: canonical decision and rejection ledger, 2026-09-20.

This file prevents failed mechanisms from returning under new names. Detailed
measurements live in [Benchmarks](benchmarks.md). Reopen a rejection only when
new evidence changes its capacity, bandwidth, fidelity or acceptance bound.

## Fixed targets

```text
Qwen3.8-27B
  host        one RTX 3090
  context     262,144 actually populated positions
  KV          exact IEEE F16
  harness     real coding agent, default xhigh
  target      about 15 useful output tok/s

DeepSeek
  host        same self-contained machine; NVMe+RAM+VRAM allowed
  routing     exact router, top-k and stable aggregation
  novel       about 1 tok/s acceptable
  settled     10-15 useful tok/s
```

The Qwen feasibility inequality is currently negative:

```text
hot target weights             13,625,700,352 bytes
exact F16 target KV            17,179,869,184 bytes
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
measured pinned PCIe link       12.46 GiB/s
full host-KV scan floor             1.284 s/call
```

Ordinary offload cannot reach 15 tok/s. A new runtime proposal must first prove
one of: enough compute-local capacity, lossless residency below the practical
bit threshold, or enough exact accepted tokens per full-KV traffic unit after
complete proposer/verifier cost.

## Accepted bounded mechanisms

| Mechanism | Accepted scope | Boundary |
|---|---|---|
| progressive exact-F16 mirror | Qwen prefixes that fit the disposable device mirror | preserves F16; does not solve 262K capacity |
| exact sparse-expert paging | DeepSeek and compatible MoE selected pages | makes models executable beyond RAM+VRAM; novel routes remain movement-bound |
| measured CPU/GPU expert split | exact selected experts and stable merge | first DeepSeek gate did not improve throughput |
| exact QSA tiering | Qwen Flash exact index plus selected F16 payload | QSA was only about 1.2% of measured short execution |
| startup routed-cache fitting | artifact-neutral, before first expert admission, after future-state reservation | measured with the former `ornith-k1` profile; not a global cache increase |
| lazy dense/logits workspaces | allocate only the rows/operation requested | saved about 994 MiB; arithmetic unchanged |
| exact hierarchical top-k | dense-provider sampling with artifact-declared `top_k <= 64` | preserves logits, candidate ids, tie order and NaN exclusion; does not change KV fidelity or prefill |

## Experimental K1 exception

K1 stores block-32 FP4 values and FP4 keys with one aligned FP16 outlier
correction per key block. It is lossy and remains implemented, but no active
alias selects it after the per-head-Q4 promotion. `qwen-f16` remains the exact
Qwen fidelity reference. The abliterated alias is a separately qualified
third-party weight artifact with the same dense geometry; its behavioral
quality is not inferred from the official model.

For Qwen at 262,144 positions:

```text
F16 target KV                 16.0000 GiB
K1 target KV                   4.7500 GiB
K1 target + MTP pages          5.015625 GiB
K1 attention, 16 layers       29.0161 ms (repeat)
K1 populated decode           14.39 -> 17.07 tok/s after first token
```

A deterministic populated-262,016 request returned the same 47 output tokens
under K1 and F16; K1 took 546.173 s and F16 2,278.082 s. Both made the same
arithmetic error. One parity case is not a quality corpus, so K1 does not
satisfy the exact-F16 goal and cannot be described as lossless.

The historical measurements remain evidence about K1, not the current default
or proof of per-head-Q4 fidelity.

### K1 attention-kernel gate, 2026-09-16

At 14.39 generated tokens/s, one populated-context token takes 69.493 ms. The
15 tokens/s gate allows 66.667 ms, so an attention-only change must save at
least 2.826 ms/token. With 16 full-attention layers measured at 29.0161 ms,
the necessary micro-gate was therefore:

```text
required 16-layer attention <= 26.190 ms
required per-layer attention <= 1.637 ms
required improvement >= 9.74%
```

Nsight Compute on the 262,144-position K1 kernel found 119 registers/thread,
48.06 KiB static shared memory/block, 33.33% theoretical occupancy, substantial
barrier/long-scoreboard stalls, 75% excessive global sectors and 44% excessive
shared wavefronts. This rejects the proposed naive `cp.async` double buffer:
a second complete tile would exceed the current two-block shared-memory budget,
while DRAM was not the sole limiter.

One 64-bit packed-load change reduced excessive global sectors from 75% to
58%, but changed the 16-layer micro-gate only from the 29.0161 ms recorded
baseline to 28.758 ms (0.89%). The single permitted correction reused the
existing 512-thread, two-block launch-bounds variant and regressed the gate to
30.9719 ms. Both runtime changes were removed.

A subsequent source-counter pass localized 25,165,824 of 36,438,016 excessive
shared wavefronts to the four 128-bit K/V decode stores. Token-striped chunk
ownership removed that dominant conflict: excessive shared wavefronts fell to
11,272,192 (44% to 20%), numerical parity remained exact, and the 16-layer
gate improved from 29.0161 to 28.5245 ms (1.69%). It simultaneously restored
75% excessive global sectors because adjacent lanes read different records.
The earlier coalesced-load experiment was worth only about 0.21 ms/16 layers;
even adding that gain optimistically leaves about 28.32 ms, before the extra
staging barrier required to combine both layouts. The two-stage correction
therefore failed its prerequisite and was not implemented. The token-striped
runtime change was removed. Reopen this kernel only with a data layout that
proves both coalesced global reads and conflict-free WMMA input without another
full-tile barrier, plus the unchanged <=26.190 ms micro-gate.

### Cold-prefill S-A/S-B/S-C review, 2026-09-17

Three proposed changes were checked against the current provider and rejected
in their proposed form before CUDA work:

1. **S-A, one-pass K1 FlashAttention:** the K1 path already calls the segmented
   FlashAttention implementation with online LSE merge. It does not materialize
   a full score arena or enter the cuBLAS fallback for Qwen's geometry. Exact
   causal QK+PV for 262,016 positions, 16 layers, 24 query heads and head
   dimension 256 is approximately 13.498 PFLOP. The RTX 3090 dense BF16 tensor
   peak gives an unattainable lower bound of about 95.1 seconds before softmax,
   K1 decode or launch overhead. The proposed <=80-second gate and 15-25-second
   estimate therefore fail the hardware bound. The measured 250.365 seconds is
   about 53.9 TFLOP/s, consistent with the reported 38% of peak.
2. **S-B, decode weights once for the whole prompt:** weights are already
   decoded once per operation and reused across every 1,024-row batch inside a
   6,144-row sequence tile. The 262,016-token run uses 43 tiles, not 256 weight
   decodes per matrix. Keeping reuse across all tiles requires layer-major
   execution. Its complete FP32 hidden stream is 5,366,087,680 bytes (4.998
   GiB), which does not fit beside hot weights, K1 KV, workspaces and reserve.
   Spilling it through the measured 12.46 GiB/s link for 64 layers has a
   51.3-second bidirectional floor before compute. The FFN alone contains
   17,112,760,320 weights and needs approximately 8.968 PFLOP for this prompt,
   a 63.2-second peak-compute floor. The proposed 20-35 seconds is impossible.
3. **S-C, logarithmic exact delta scan:** for one value head and output column,
   the transition is affine,
   `s_t = d_t (I - beta_t k_t k_t^T) s_(t-1) + beta_t v_t k_t`.
   Composition is associative, but products of the rank-one factors are not
   closed in the same compact form: both the linear map and accumulated offset
   become dense 128x128 objects. Materializing `(M,C)` prefixes for one
   1,024-row chunk and 48 value heads requires about 6 GiB before scan scratch
   or outputs. A different association also changes FP32 results. No compact
   closure, capacity bound or operation-count gate was supplied, so the claimed
   10-15 seconds is not an admitted implementation plan.

These failures do not prove the measured phases are optimal. They prove that
the stated mechanisms and latency gates do not follow from the current code or
hardware. Reopening cold prefill requires a new design whose exact operation
count, state representation, capacity and traffic pass before implementation.

### Persistent continuation snapshots: implemented restart/resume path

The live service already supports exact-prefix checkout plus provider parking
in pinned RAM. For Qwen the parked continuation contains the selected codec's
KV pages, all recurrent convolution/matrix states, three hidden vectors,
position/RoPE/media identity and the exact token-prefix identity. Dense MLP
intermediates are transient and are not part of a continuation snapshot.

Current Qwen page and fixed-state sizes are:

```text
K1 target+MTP page, 256 tokens       5,259,264 bytes
F16 target+compact-MTP page         17,055,744 bytes
recurrent plus hidden fixed state      158,920,704 bytes
full 262,144 K1 snapshot              about 5.164 GiB
full 262,144 F16 snapshot             about 16.414 GiB
```

The measured 120,187-token K1 snapshot was 2,630,774,784 bytes and the
129,217-token snapshot was 2,814,849,024 bytes. A content-addressed page chain
would write only 35 new K1 pages plus the replaced recurrent/hidden blob for
that transition, about 327 MiB, rather than rewriting the complete 2.62 GiB.

The implemented format is transactional and fail-closed: 4 MiB immutable
SHA-256 chunks, a versioned provider manifest, exact
artifact/tokenizer/template/codec/media and prefix identity, checksummed
recurrent/hidden state, byte-budgeted NVMe LRU and atomic
candidate-to-committed publication. RAM remains the hot tier. NVMe is read
once at request restore; active KV is never streamed from storage per decode
token. The `memory-expert` exact-prefix and split-versus-joint identity lesson
applies, but Qwen needs a complete continuation state, not KV alone.

The 2026-09-17 real Pi restart gate passed under K1: 443 retained tokens
occupied 169,443,798 bytes on NVMe; after a complete service restart the
snapshot restored in 0.203 s, prefilled only a 34-token suffix and completed in
0.938 s. The startup index retained zero RAM sessions and one disk session,
proving lazy restore. Corrupt metadata rejection and byte-bounded LRU eviction
are unit-tested. `MODEL_SESSION_CACHE_GIB` and
`MODEL_SESSION_CACHE_TTL_SECONDS` control automatic cleanup; `./ops/model.sh
clear-cache qwen` explicitly removes only that artifact's durable cache.

The deterministic parity gate also passed for K1: an initial request followed
by the same fixed-seed `temperature=0` continuation produced `PARITY-BETA` both
live and after complete service restart/restore. A large real coding transcript
still must be measured before claiming a practical Pi latency result.

The mechanism must keep these gates:

- full service restart followed by a disk hit and suffix-only provider prefill;
- uninterrupted versus restored next-token/logit parity under the same codec;
- exact byte accounting for disk read/write, H2D restore and resident state;
- rejection of corrupt, partial, ABI/model/template/codec or prefix mismatch;
- interrupted-write invisibility, byte-budget LRU eviction and final cleanup.

This mechanism can remove repeated prefill for an already-computed exact
prefix. It cannot improve the first cold prefill, and K1 persistence does not
turn K1 into exact F16.

CUDA Graph replay remains unimplemented: the available executable profile
attributed only about 0.23 ms/token outside the measured GPU phases, far below
the required 2.826 ms/token populated-context saving. Concurrent
micro-batching is also not a 262K solution: one K1 target plus MTP state is
5.015625 GiB and the
validated provider has one serialized hot slot. Moreover, completing two
requests in `1.25 * T1` would be 1.6x aggregate throughput, not the claimed
per-request doubling. Reopen either direction only with a whole-step trace and
an explicit capacity/latency inequality for the target workload.

### Exact top-k sampling gate, 2026-09-17

The recommended Qwen profile performs top-20 sampling over 248,320 logits.
The original one-block kernel rescanned the complete vocabulary once per rank:

```text
original Nsight trace        9.916 ms/decode call
required total saving        2.826 ms/decode call
accepted micro-gate         <=7.090 ms/decode call
hierarchical native smoke    0.423 ms/decode call
hierarchical worker trace    0.258 ms/decode call
```

The new kernel performs exact deterministic partial reductions over contiguous
4,096-item ranges followed by one exact final reduction per rank. An
independent CPU full sort matched all selected token ids with ties and NaNs;
the host sampler receives the same ordered values and ids. The real
262,016/128 service gate reported 536.469 s TTFT, 543.907 s total and 17.07
tok/s after-first, versus the prior 14.39 tok/s. This admits the sampler change
while leaving the K1 fidelity exception and cold-prefill failure unchanged.

## Rejected general mechanisms

| Direction | Failed prerequisite | Decision |
|---|---|---|
| FP8 target KV | 262K prefill 1,168.594 s, decode 3.064 tok/s and factual failure | wrong fidelity and too slow |
| Q4 target KV | user-confirmed material quality degradation | rejected for exact-F16 goal |
| uniform K8/V4 KV | 34.7668 ms for 16 attention layers versus 32.5 ms budget, before overhead | rejected before service integration |
| dense layer streaming/AirLLM | 12.690 GiB hot weights / 12.46 GiB/s >=1.018 s/call | capacity mechanism only |
| static lossless weight+KV coding | ideal result still 1,123,844,072 bytes over VRAM before runtime state | insufficient capacity |
| reversible F16 transforms | 13.265926 bits/value measured versus 11.617612 required | insufficient compression |
| Huffman/Zstd/zlib/LZ4 | best held-out Huffman 13.719539 bits/value | insufficient compression |
| exact MTP tree | 4.565 accepted/cycle; optimistic ceiling 2.8 tok/s | insufficient amplification |
| FP4-KV block-Jacobi | 3.25 accepted/cycle; corrected ceiling 1.55 tok/s | insufficient amplification |
| prompt n-gram/copy | 3.8485 useful/cycle versus 24 required | insufficient acceptance |
| exact page refinement | 2,029.014 MiB/call versus 272.822 MiB budget | traffic exceeds bound 7.437x |
| staged single-GPU exact attention | 91.763 ms for 16 layers before other work | misses complete 15 tok/s budget |
| CPU F16 V-tail | 226.279 ms/call at 12 threads | too slow |
| all-K-resident/selective-V | about 25,922 MiB before V staging | exceeds VRAM |
| pageable-to-pinned expert bounce | extra 1.385 GB copy and worse upload wait | removed |
| neural CPU replacement | failed structural/correctness boundary | removed |

The `exact MTP tree` rejection above applies to the exact-F16/offloaded traffic
inequality that produced it. It does not reject resident K1 target verification.
On 2026-09-19, the installed one-draft K1 service completed a real
262,016/128 gate at 26.00 useful tok/s after first token. It accepted 60 of 67
drafts, advanced 127 positions in 67 target calls (`A=1.8955`) and spent
72.91 ms per complete proposer/verifier cycle. At the measured conditional
acceptance `p=60/67`, MTP-4 projects `A=4.0588`; 40 tok/s permits 101.47 ms per
complete cycle, leaving 28.56 ms over the one-draft baseline. This is the new
quantitative prerequisite that admits a resident-K1 MTP-4 experiment. The
50 tok/s bound is 81.18 ms and is not admitted until the complete sampled
multi-draft cycle passes it.

The experiment must remain fail-closed on these boundaries:

- the target K1 distribution is authoritative and remains explicitly lossy
  relative to F16 KV;
- sampled speculation uses exact `p/q` rejection/correction, not argmax
  identity, and is independently checked against the scalar target sampler;
- target verification processes five causal rows while loading each target
  K/V tile once; a prefill fallback or five scalar attention launches is not
  the requested result;
- the draft vocabulary may be an artifact-declared 65,536-token prefix, but
  the target head remains the complete 248,320-token head and residual
  sampling retains target mass outside the draft prefix;
- Q8 applies only to the proposer cache. Its 528,384-byte page increases the
  1,024-page MTP allocation by 244 MiB. The measured 1,286 MiB post-gate free
  VRAM leaves only 18 MiB over the 1 GiB reserve after that increase, so the
  120 MiB sequence-prefill buffer must become releasable before Q8 promotion;
- RAM and NVMe carry no active per-token payload. The target lower bound is
  still 13,625,700,352 weight bytes plus 5,100,273,664 target-K1 bytes per
  target call. At `A=4.0588` this is at least 4.614 GB of target HBM traffic
  per useful token, before MTP weights/cache and workspaces.

The completed ABI-2 experiment failed this prerequisite. With exact sampled
`p/q` at `temperature=1.0`, `top_p=0.95`, `top_k=20`, MTP-4 reached only
`A=2.4615` at 32K and `A=2.6458` at 128K. End-to-end post-first rates were
15.67 and 8.59 useful tok/s respectively. Linear extrapolation of measured
seconds/useful-token gives 5.36 tok/s at 262K; no 262K service run was made for
this experiment. Because even 32K requires a 2.55x improvement to reach 40
tok/s, local kernel tuning cannot satisfy the admitted inequality. The stable
artifact was rolled back to exact-decode ABI 1. Do not promote the ABI-2
recurrent rollout without new measured evidence that changes both the
accepted-positions boundary (`A>=3.0`) and the complete-cycle bound; a
microbenchmark-only improvement is insufficient.

That rollback was not the final result. Executable profiling subsequently
proved that every rejection reran the target prefix even though the first
five-row verification pass had already produced the required recurrent states.
The experiment was reopened on this new end-to-end prerequisite. The target
pass now writes five FP32 recurrent checkpoints and restores the accepted row;
it never rereads target weights or target K/V after rejection. The final
context-boundary draft was also removed, and benchmark service starts disabled
session parking/snapshots.

Clean suffix-window gates improved to 26.19 useful tok/s at 32K and 23.89 at
128K. They advanced 127 positions in 54 and 45 target calls respectively,
giving `A=2.3519` and `A=2.8222`. Linear extrapolation of measured
seconds/useful-token is 21.39 tok/s at 262K; no live 262K run was made. This is
a real improvement and the MTP-4 implementation remains usable, but it still
does not satisfy 40-50 tok/s.

The updated blocker is cycle latency. At 128K, `A=2.8222` permits 70.56 ms for
40 tok/s while the measured cycle is 118.12 ms. The 32K/128K cycle slope
extrapolates to 155.90 ms at 262K, where even the perfect five positions per
MTP-4 cycle cap at 32.07 tok/s. A short-context phase trace attributes 25.31
ms/call to dense FFN and 15.38 ms/call to recurrent target matrices before
long-context attention.

Do not replace the selected small-batch DP4A path with BF16 Tensor Core weight
decode. The real batch-five gate measured 453.70/365.78 GB/s for DP4A on the
narrow/wide Qwen projections. The Tensor Core candidate reached only
148.60/67.14 GB/s; its one permitted barrier-removal correction regressed to
121.74/51.17 GB/s. Both were numerically valid, but the runtime candidate was
removed. Reopen target dense kernels only with a mechanism whose two-shape
micro-gate exceeds the selected DP4A bandwidth and whose projected complete
cycle can close the 47.56 ms 128K gap.

### Exact-F16 capacity and speed reopening gate, 2026-09-20

Making the raw F16 cache resident is not by itself a 40-50 tok/s result. The
current SM86 resident-F16 attention smoke reads one 262,144-token layer in
5.73952 ms at its best split, or 91.832 ms across the 16 full-attention layers
before target matrices, proposer, head or sampling. At the measured 128K
MTP-4 acceptance (`A=2.8222`), the complete 40 tok/s budget is only 70.56 ms.

The short-context trace leaves at least 47.35 ms/cycle for target FFN,
recurrent blocks, proposer and head. Exact-F16 attention would therefore have
to fit within 23.21 ms/cycle at the current acceptance: a 3.96x reduction from
the measured raw-F16 attention time, in addition to fitting beside
13,625,700,352 bytes of hot target weights, recurrent state, MTP state,
workspaces and the 1 GiB reserve.

The existing bit-exact measurements do not admit that path: the best tested
reversible F16 transform needs 13.265926 bits/value and the best held-out
Huffman result needs 13.719539 bits/value. New lossless work must first beat
both the actual resident-capacity budget and the `<=23.21 ms` fused-attention
gate on real Qwen K/V. A codec ratio measured only at rest is insufficient.
Lossy transform coding may meet the capacity/traffic gate, but it is a new
target-KV fidelity profile and requires an explicit quality gate against F16.

### Direct packed target-KV reopening gate, 2026-09-20

The user explicitly authorized a new lossy target-KV experiment and retained
the real-project Pi comparison as the final fidelity gate. This does not alter
the exact-F16 production objective or turn a runtime numerical oracle into a
model-quality result.

The admitted format is block-floating signed Q4 with a record-wide FP16 base,
one four-bit exponent per 32-value block and, for keys only, one aligned FP16
outlier correction per block. The packed payload is consumed by the batch-five
attention kernel. It is expanded only to transient INT8 Tensor Core operands;
there is no decoded F16 cache. For Qwen's 256-value head the exact record and
capacity equations are:

```text
K record = 128-byte Q4 + 4-byte exponents + 2-byte base
         + 8 * 4-byte FP16 outlier records             = 166 bytes
V record = 128-byte Q4 + 4-byte exponents + 2-byte base = 134 bytes
K+V per token/head/layer                                = 300 bytes
target page = 256 tokens * 4 KV heads * 16 layers * 300 = 4,915,200 bytes
target at 262,144 positions                             = 5,033,164,800 bytes
                                                         = 4.687500 GiB
Q8 MTP pages                                             = 541,065,216 bytes
target + MTP                                             = 5.191406 GiB
hot target weights                                       = 13,625,700,352 bytes
weights + target KV + MTP                                = 17.8815 GiB
```

The current K1 target page is 4,980,736 bytes. The candidate saves exactly
65,536 bytes/page, or 64 MiB at 1,024 pages, while leaving recurrent state,
MTP state, workspaces and the 1 GiB reserve unchanged. Capacity therefore
strictly improves on the already admitted K1+Q8 configuration; it does not
depend on RAM, PCIe or NVMe traffic.

At 262K, one target call touches at least 13,625,700,352 target-weight bytes
plus 5,033,164,800 target-KV bytes. At the measured `A=2.8222`, that is at
least 6.61 GB of target HBM traffic per useful token before proposer and
workspace traffic. The physical-HBM bound is not the failed prerequisite:
the measured target dense phases already consume about 40.69 ms/call and the
current batch-five 128K cycle is 118.12 ms.

For the requested 40 tok/s gate at 128K:

```text
complete-cycle budget = 2.8222 / 40                     = 70.56 ms
measured fixed non-long-attention work                  = 47.35 ms
available batch-five long-attention budget              = 23.21 ms
current inferred variable work = 118.12 - 47.35          = 70.77 ms
required variable-work improvement                      = 3.05x
```

At 32K, the measured `A=2.3519` permits 58.80 ms/cycle, leaving only
11.45 ms over the same fixed-work estimate; the current inferred variable
work is 42.44 ms, a 3.71x gate. At current acceptance, 50 tok/s at 32K would
permit 47.04 ms, slightly below the 47.35 ms fixed estimate, so 50+ also
requires either higher sampled acceptance or a separate dense-phase gain.

The packed layout is admitted because it attacks the measured shared-memory
and barrier blocker rather than claiming that 1.3% fewer global KV bytes is a
3x speedup. Head-major, plane-separated payloads give aligned coalesced reads.
A 32-token INT8 operand tile halves shared bytes per value and halves tile
rounds relative to the current 16-token BF16 operand tile. QK and PV use INT8
Tensor Cores; per-block powers of two are folded into the expanded integer
operands, and K outliers are corrected against the same transient Q8 query.
The implementation must pass an independent codec/attention oracle and the
complete Qwen 32K/128K service gates before any throughput claim.

The first monolithic direct-attention implementation passed the independent
layout, query-quantization and attention oracle with maximum absolute error
`3.72529e-09`. Its clean batch-five profiles measured 10.0004 ms over the 16
full-attention layers at 32K and 33.1325 ms at 128K. A single permitted local
correction made the PV result scratch warp-private and removed three
block-wide barriers per tile. The clean retest improved those values to
10.0004 -> 9.08698 ms at 32K and 33.1325 -> 30.0114 ms at 128K; the 128K
prerequisite still failed, so no service gate was run. (The separately sampled
post-build profile was 9.0624/29.9462 ms; it is diagnostic, not the gate.)

Nsight Compute on the corrected 128K producer measured 58 registers/thread,
47,744 bytes of static shared memory, two resident blocks per SM, 21.49% warp
stall at barriers, 16.73% MIO throttle, 12.70% short-scoreboard stall and only
8.49% DRAM throughput. This rejects further local barrier edits to the same
phase-imbalanced 512-thread kernel: QK uses only four of its sixteen warps,
while softmax and PV use all sixteen.

The next measured prerequisite was a two-stage direct-attention schedule. A QK
kernel reads packed K once and writes FP32 scores; a fused online-softmax+PV
kernel reads packed V once and writes the existing split accumulators. Packed
K/V is never expanded into an F16 cache. At 128K the score workspace is:

```text
5 rows * 24 query heads * 131,072 positions * 4 bytes = 62,914,560 bytes
                                                        = 60 MiB
```

At 262K it is 120 MiB. Releasing the existing 120 MiB prefill-only sequence
buffer and the Q4 format's 64 MiB saving over K1 leave approximately 202 MiB
above the reserve before this allocation, or 82 MiB after it. At 128K the
two-stage path touches the 150 MiB K+V payload plus 120 MiB of score writes and
reads per layer. Even that 270 MiB total has a 0.30 ms physical-HBM lower bound,
well below the required 1.450625 ms/layer. The candidate is admitted only if
the complete two-kernel attention call, including score traffic, split combine
and launch overhead, remains at or below 0.715625 ms/layer at 32K and
1.450625 ms/layer at 128K.

That prerequisite failed. The clean two-stage gate preserved the independent
oracle (`3.72529e-09`) but measured 9.76486 ms at 32K and 32.3932 ms at 128K
over 16 layers, versus 9.08698/30.0114 ms for the corrected monolithic path.
Nsight Compute attributed 0.861184 ms/layer to QK and 1.485600 ms/layer to
softmax+PV at 128K. The latter read 133,226,624 bytes, reached only 98.74 GB/s,
and reported 10.15% barrier, 19.80% MIO, 14.08% short-scoreboard and 16.20%
wait stalls. Even eliminating every measured PV barrier would save only about
0.15 ms/layer, below the approximately 0.57 ms/layer gap. The two-stage path
and its 120 MiB maximum score workspace were therefore removed from runtime
selection; the faster monolithic direct kernel remains the experimental Q4
implementation. Do not reintroduce a score spill without a micro-gate below
the monolithic result and the 1.450625 ms/layer acceptance bound.

The next implementation cycle keeps the monolithic, single-read schedule but
changes its warp mapping. QServe's Apache-2.0 W4A8KV4 kernel was inspected as
an external reference: its launch grid is `(query_head, batch)`, so Qwen's six
query heads per KV head and five speculative positions would traverse the same
KV head 30 times. It is therefore not an admissible batch-five implementation;
only its direct packed-load principle is relevant.

The admitted SM86 candidate uses the PTX-defined
`mma.m16n8k32.row.col.s32.s8.s8.s32` fragment layout. Eight warps cover the two
16-row query tiles and four 8-column key tiles, instead of leaving twelve of
sixteen warps idle during QK. PV keeps its integer accumulators in registers
and maps them to rows and dimensions using the documented PTX accumulator
layout. This removes the 16,384-byte shared `products` arena. The transient V
INT8 tile is dimension-major so each PTX B fragment is a pair of aligned
32-bit shared loads; no F16 representation is introduced. The shared-memory
accounting changes from:

```text
old static shared = 47,744 bytes
removed products  = 16 * 16 * 16 * 4 = 16,384 bytes
V bank padding    = 256 * (36 - 32) = 1,024 bytes
new static shared = 32,384 bytes
```

The first PTX implementation preserved the oracle but regressed to
12.4396/41.5232 ms. Nsight Compute measured 31,360 bytes of shared memory,
60 registers/thread, 32.9% MIO-queue stalls and 186,253,312 excessive shared
wavefronts (82% of all shared wavefronts). The dimension-major V transpose had
made all lanes write the same bank. The one permitted local correction uses a
36-byte dimension stride and XORs each four-token group by `dimension / 32`;
PV reverses that permutation for aligned 32-bit loads. For the token-major
decode-write mapping, the bank index becomes a permutation of all 32 banks per
warp rather than a 32-way collision. No global bytes or extra synchronization
are added.

The packed global traffic is unchanged: one 166-byte K record and one
134-byte V record per token/KV-head/full-attention layer. The required complete
attention gates remain `11.45 ms / 16 layers` at 32K and
`23.21 ms / 16 layers` at 128K (`0.715625` and `1.450625 ms/layer`). The
candidate is rejected before a service run if either gate fails or if the
independent layout/query/attention oracle changes from the established
`3.72529e-09` bound.

The corrected PTX cycle improved the clean profile to `8.81869 ms` at 32K and
`28.9157 ms` at 128K, preserving the `3.72529e-09` independent-oracle bound.
It therefore clears the 32K attention prerequisite but misses the 128K bound
by `5.7057 ms`, or `19.73%`; no service gate was run. A 128K Nsight Compute
capture measured 58 registers/thread, 32,384 bytes of static shared memory,
62.26% achieved occupancy, only 8.86% DRAM throughput, 20.04% MIO-queue stall
and 20.66% barrier stall. It reported 56,229,888 excessive shared wavefronts,
58% of all shared wavefronts, plus 8,493,824 excessive global sectors.

The next cycle retains the exact same packed ABI and numerical operations. Its
PTX operand staging rotates Q and K four-byte words by matrix row, and rotates
the eight four-token P and V words by row/dimension. Exhaustive host-side bank
mapping shows one distinct bank per lane for every QK and PV fragment load.
K corrections change from one token per lane with an eight-iteration loop to
eight lanes per token; four records then form one contiguous 128-byte global
transaction, while block-major 32-bit index/FP32-value planes make correction
reads bank-aligned. The candidate remains fail-fast: it must preserve the
independent oracle and meet both `11.45/23.21 ms` gates before any real Qwen
service run.

The first all-plane swizzle preserved the oracle but regressed to
`11.4668/38.1379 ms`. Nsight showed that shared loads fell from 85,836,050 to
33,671,158 wavefronts, while shared stores rose from 10,996,168 to 143,512,278:
the V rotation mapped every lane of each transpose instruction to one bank.
The single local correction restores the measured 36-byte/XOR V layout and
keeps the Q/K/P improvements. Correction metadata uses
`bank = token + 4 * block (mod 32)`, which is bijective for both the
eight-block-per-token writer and the fixed-block reader.

That correction passed the independent oracle and improved the clean gate to
`7.75782 ms` at 32K and `25.2314 ms` at 128K. Relative to the pre-PTX
monolithic path this is 14.63% and 15.93% faster, respectively. The 32K bound
passes, while 128K misses by `2.0214 ms/cycle`, or 8.01% of attention. Using
the measured fixed-work and acceptance values projects approximately 42.68
useful tok/s at 32K and 38.88 tok/s at 128K. No service gate was run.

The post-correction 128K profile measured 61 registers/thread, 62.26% active
warp occupancy, 10.24% DRAM throughput, 11.49% MIO stall, 15.06% barrier and
15.08% wait. The remaining launch geometry is quantifiably wasteful on the
82-SM RTX 3090: two resident blocks/SM give 164 slots, but 128 splits per KV
head produce 512 blocks, or three full waves plus a 20-block tail. With 32
tiles/block this has a 128 tile-wave makespan. A topology-driven split selects
`ceil(164 / kv_heads)` splits per KV head and rounds the token span to 32. For
Qwen this gives 41 splits, exactly 164 blocks: 100 tiles in one wave at 128K
and 25 at 32K, a 21.88% producer-makespan prerequisite versus the 8.01%
attention reduction required. The CUDA occupancy query, not a family/model
branch, determines this geometry; the caller split remains a lower bound.

The topology-balanced implementation passed the independent oracle unchanged
at `3.72529e-09`. Its clean batch-five attention gate measured `6.10714 ms`
over all 16 full-attention layers at 32K and `22.8332 ms` at 128K, so both
attention prerequisites pass. The 128K producer launches exactly 164 blocks
(`4 KV heads * 41 splits`) and sustains 110.216 GB/s of packed payload traffic.
Nsight Compute measured 61 registers/thread, two blocks/SM, 58.36% achieved
occupancy, 10.47% DRAM throughput, 15.22% barrier stall, 11.36% MIO throttle
and 15.47% wait stall.

The real sampled service gate does not inherit the old K1 acceptance. With the
actual `top_k=20`, temperature 1.0 and exact `p/q` rejection, the Q4 candidate
measured `A=2.64583` at 32K and `A=2.54` at 128K. The 128K cycle was
`75.6853 ms`, of which measured direct attention accounts for `22.8332 ms`;
the observed non-attention remainder is therefore `52.8521 ms`. At the
observed acceptance the complete 40 tok/s budget is only
`2.54 / 40 = 63.50 ms`, leaving `10.6479 ms` for attention. The remaining
gap is `12.1853 ms/cycle` or 16.10% of the complete cycle. A local layout edit
to the existing Q4 kernel is not an admitted route to the target unless its
micro-gate can remove at least 53.37% of the current attention time, or it is
paired with independently measured acceptance/non-attention improvement.

The next admitted cycle targets the lossless target dense batch-five work, not
the sampler or another Q4 layout tweak. The target has 13,625,700,352 bytes of
hot packed weights. The previous one-token phase attribution measured
40.685 ms/call in target FFN plus recurrent blocks; removing the observed
12.1853 ms complete-cycle gap requires that work to fall to at most 28.500 ms,
equivalent to at least 478.1 GB/s if every packed byte is read once. That is
51.1% of the RTX 3090's 936 GB/s physical bandwidth and is therefore feasible;
the physical lower bound is 14.56 ms. Five rows require about 128.25 GOP/call
for 25.65 billion four-bit values, only 0.72 ms at 62.5% utilization of the
card's advertised INT8 Tensor Core rate, so arithmetic is not the bound.

The candidate maps one 16-output-row by eight-request tile to
`mma.m16n8k32.row.col.s32.s8.s8.s32`, with the three unused request columns
zeroed. It reads the existing row-major FP4 payload once from its artifact,
expands nibbles only into registers, and applies each row/block UE8M0 scale to
the corresponding FP32 fragment before accumulation. It adds no persistent
weights, workspace or fidelity change. Admission requires the independent
scalar dense oracle to remain below `2e-4` and both real Qwen matrix shapes at
batch five to improve over the measured 453.70/365.78 GB/s DP4A baselines; a
kernel that fails either bandwidth comparison is removed before service gates.

The first register-fragment implementation preserved the independent scalar
oracle (`5.34058e-05`) but reached only 221.509 GB/s on the 17,408x5,120
matrix and 110.052 GB/s on 5,120x17,408. Its four 16-bit loads per lane and
K block fragmented the row-major FP4 stream. The one local correction loaded
one vector per packed row into a 2.25 KiB warp-private shared tile. It
regressed further to 206.968/100.796 GB/s. Both are below the
453.70/365.78 GB/s DP4A baselines, so neither implementation is eligible for
service execution.

End-to-end reassessment found that the schedule, not only the load layout,
starved the device: one warp produced 16 output rows, yielding only 1,088
independent warps for the narrow matrix and 320 for the wide matrix, or 13.27
and 3.90 warps per SM over the entire launch. The observed 221.509/110.052
ratio tracks that loss of available parallelism. A new split-K schedule is
therefore admitted on a distinct prerequisite: eight warps in one CTA process
disjoint K blocks for the same 16x8 output tile, read each packed weight once,
and reduce 4 KiB of FP32 partials once at the end. This makes 8,704/2,560 warp
work items available without a global partial workspace. It must still pass
the same scalar oracle and exceed both DP4A bandwidth baselines; otherwise the
Tensor Core direction is removed and batch five remains on stable DP4A.

The split-K schedule passed the independent oracle at `5.34058e-05` and
measured 597.287 GB/s on 17,408x5,120 and 525.455 GB/s on 5,120x17,408. This
is 31.65% and 43.65% above the stable DP4A baselines, respectively, and above
the 478.1 GB/s end-to-end prerequisite. The exact clean binary also preserved
the Q4 attention oracle (`3.72529e-09`) and measured `6.07846/22.7021 ms` for
all 16 full-attention layers at 32K/128K.

The real sampled gates retained exactly the previous output hashes and
acceptance values, while complete cycle latency fell from 56.437 to
50.0885 ms at 32K and from 75.6853 to 69.4494 ms at 128K. Useful throughput
rose to 52.82 and 36.57 tok/s. This admits the split-K kernel but does not close
the 128K target: at `A=2.54`, 40 tok/s still requires 63.50 ms/cycle, leaving a
5.9494 ms gap. With the observed 22.7021 ms attention call and unchanged
acceptance, an attention-only successor must reach at most 16.7527 ms, a
26.21% reduction. No 262K service run was performed.

The next attention cycle keeps the packed ABI and every numerical operation.
At 128K the balanced launch executes 164 blocks * 100 K/V tiles. Thirty active
query-head rows currently perform eight random shared-memory query loads per
tile for the key-outlier correction, or 3,936,000 warp load instructions.
Their random 0-31 indices serialize shared banks and contribute to the measured
35.754 million excessive shared-load wavefronts and the combined 34.58% MIO,
short-scoreboard and wait stalls. Each lane can instead load its own element
of the 32-value query block once and use `shfl` to select the outlier index for
its token. This preserves the exact Q8 value and correction order while making
the shared read conflict-free. It is admitted only if the independent oracle
is unchanged and the complete 16-layer 128K attention gate falls from
22.7021 ms to at most 16.7527 ms; otherwise no service run follows.

The shuffle candidate preserved the `3.72529e-09` oracle but regressed to
`6.29965/23.5827 ms` at 32K/128K. Eight shuffle instructions per score row and
tile cost more than the removed shared-bank serialization. No service gate was
run, and the conflict-swizzled shared-load implementation remains selected.
Do not reintroduce query shuffles without a schedule that amortizes them across
more than one correction or removes the correction loop itself.

The admitted correction-removal candidate is a separately named Q5-K/Q4-V
block-floating format, not a silent change to the Q4 ABI. For a 256-value head,
Q5 keys use 160 code bytes, four exponent bytes and one FP16 base; Q4 values
retain 128 code bytes, four exponent bytes and one FP16 base. The exact record
and capacity equations are therefore:

```text
K record = 256 * 5 / 8 + 4 + 2                       = 166 bytes
V record = 256 * 4 / 8 + 4 + 2                       = 134 bytes
K+V per token/head/layer                              = 300 bytes
target at 32,768 positions                            = 629,145,600 bytes
target at 131,072 positions                           = 2,516,582,400 bytes
target at 262,144 positions                           = 5,033,164,800 bytes
target + existing Q8 MTP at 262,144                  = 5.191406 GiB
hot target weights                                    = 13,625,700,352 bytes
weights + target KV + MTP                             = 17.8815 GiB
```

Thus capacity, target-KV HBM traffic, MTP Q8 state, workspaces and the 1 GiB
reserve are identical to the admitted Q4 candidate. No target KV moves through
RAM, PCIe or NVMe. At 128K a target call touches at least
`13,625,700,352 + 2,516,582,400 = 16,142,282,752` hot target bytes, or 17.25 ms
at the RTX 3090's 936 GB/s physical bandwidth and 6.36 GB per accepted token at
the observed `A=2.54`, before proposer/workspace traffic. The Q5 key codes are
expanded only into transient signed INT8 Tensor Core tiles; attention never
materializes an F16 cache.

The candidate replaces each block's Q4-plus-F16-outlier representation with
signed Q5 and a power-of-two exponent. Its integer range is `[-15, 15]` and
the exponent is limited to 0--3, so the transient Tensor Core operand remains
inside `[-120, 120]`. It removes all 3,936,000 128K correction-warp iterations
and their metadata/shared-memory accesses, while adding one packed code byte
per eight K values. With the measured complete cycle `69.4494 ms`, acceptance
`A=2.54` and non-attention remainder `69.4494 - 22.7021 = 46.7473 ms`, the
fail-fast gate is unchanged:

```text
40 tok/s complete-cycle budget = 2.54 / 40            = 63.5000 ms
maximum 16-layer attention time = 63.5000 - 46.7473   = 16.7527 ms
required reduction from 22.7021 ms                    = 26.21%
```

Admission requires an independent host encoder/decoder oracle below `2e-4`
and the clean 128K batch-five attention measurement at or below 16.7527 ms.
Failure stops before a real service run. Only the requested 32K and 128K
contexts are measured; 262K remains a two-point extrapolation.

The first Q5 code layout interleaved each group of eight signed values in five
bytes. It passed the independent layout, query-scale and attention oracles at
`0`, `0` and `3.72529e-09`, but measured `5.60538/20.9347 ms` over the 16
attention layers at 32K/128K. The one permitted local correction changed only
the 160-byte Q5 code payload to two coalesced planes: 128 bytes of magnitudes
and a 32-byte sign bitmap. It preserved the same numerical oracle and improved
the clean measurements to `5.54394/20.5885 ms`. That is 2.1136 ms faster than
the selected Q4 path at 128K, but 3.8358 ms slower than the `16.7527 ms`
attention-only prerequisite. No service gate was run. At unchanged acceptance
and non-attention cost it projects to a `67.3358 ms` cycle and only 37.72
useful tok/s. Further local Q5 unpack/layout edits are rejected without a new
end-to-end prerequisite.

### Batch-five verifier projection reopening gate, 2026-09-20

The Q5 result changes the remaining complete-cycle gap from 5.9494 ms to
3.8358 ms, without changing capacity, target weight traffic, MTP state or
acceptance. The artifact contains 12,937,920,512 stored bytes of target-layer
matrix records consumed by the batch-five projection kernel. Grouping the
actual artifact shapes by the two measured Qwen projection geometries gives:

```text
8,837,464,064 bytes / 597.287 GB/s = 14.7960 ms
4,100,456,448 bytes / 525.455 GB/s =  7.8036 ms
estimated target projection total             22.5996 ms
required projection saving                     3.8358 ms
required common bandwidth factor               1.20443x
17,408 x 5,120 gate                 >= 719.39 GB/s
 5,120 x 17,408 gate                >= 632.87 GB/s
```

The estimate deliberately credits no saving to recurrent-state math, norms,
sampling, the target head or the sequential MTP proposer. It is therefore a
fail-fast requirement for combining a lossless projection change with the
measured Q5 attention gain; passing only one matrix shape is insufficient.
The physical lower bound for all 13,625,700,352 hot target-weight bytes remains
14.56 ms at 936 GB/s, so the gate is demanding but not bandwidth-impossible.

The candidate keeps the existing artifact, FP4 decode, Q8 activations, FP32
block-scale accumulation and five verifier rows. It changes the split-K
schedule from eight to four warps per 16x8 output tile. Four warps still read
each weight once, while the final partial reduction falls from 4 KiB to 2 KiB
and a 128-thread CTA permits at least four resident blocks per SM. Admission
requires the independent scalar oracle to remain below `2e-4` and clean
batch-five bandwidth of at least `719.39/632.87 GB/s` on the two real Qwen
shapes. Failure stops before service execution.

The four-warp candidate retained the `5.34058e-05` oracle but collapsed to
`303.048/363.141 GB/s`. Its 46 registers/thread and 2 KiB shared allocation
did not make occupancy the blocker; halving the independent K streams exposed
the dependent MMA/load chain. The one permitted correction kept two
independent accumulator streams per warp. It recovered only to
`404.726/358.450 GB/s`, still far below both the selected
`597.287/525.455 GB/s` implementation and the new admission gate. The stable
eight-warp kernel remains selected. Do not revisit a smaller split-K without a
different dependency-breaking schedule and a new end-to-end prerequisite.

### Plain Q4 BFP target-KV reopening gate, 2026-09-20

The next candidate is a separately named lossy format, not a reinterpretation
of either Q4-key-outlier-1 or Q5-K/Q4-V. Both K and V use 128 signed-Q4 code
bytes, four packed block exponents and one FP16 record base. It uses the same
direct batch-five integer Tensor Core attention schedule and never creates an
F16 cache. Its capacity equations are:

```text
K record = V record = 128 + 4 + 2                       = 134 bytes
K+V per token/head/layer                                = 268 bytes
target page = 256 * 4 * 16 * 268                  = 4,390,912 bytes
target at 32,768                                  =   562,036,736 bytes
target at 131,072                                 = 2,248,146,944 bytes
target at 262,144                                 = 4,496,293,888 bytes
                                                     = 4.187500 GiB
target + existing Q8 MTP at 262,144               = 4.691406 GiB
weights + target KV + MTP                         = 17.3815 GiB
```

At 128K a target call touches at least
`13,625,700,352 + 2,248,146,944 = 15,873,847,296` hot bytes, or 16.96 ms at
936 GB/s and 6.25 GB per accepted token at `A=2.54`. RAM, PCIe and NVMe remain
outside the decode path. Relative to Q5-K/Q4-V, plain Q4 removes 268,435,456
key-code bytes per 128K target call plus every sign-plane load and
sign-magnitude expansion. To close the unchanged complete-cycle inequality it
must reduce the measured Q5 attention time from 20.5885 to at most
16.7527 ms, an 18.63% reduction. Admission requires independent layout/query
checks, an independent attention oracle below `2e-4`, and that clean 128K
batch-five timing. Failure stops before a real service run. This format has a
distinct Pi fidelity obligation because removing key outlier correction is
lossy relative both to exact F16 and to the other experimental packed formats.

The complete plain-Q4 implementation passed byte-exact layout and query checks
and the independent attention oracle at `3.72529e-09`. Its first clean profile
was `5.21421/19.2778 ms` over 16 layers at 32K/128K, so the 128K prerequisite
failed by `2.5251 ms`. A branchless PRMT/vector-byte decoder retained the oracle
and improved 128K to `18.4607 ms`, still `1.7080 ms` above admission. Nsight
Compute measured 361,236,952 instructions, 57.82% SM throughput, 56.02% L1/TEX
throughput, only 11.50% DRAM throughput, 14.42% barrier stall and 17.14% wait
stall. The remaining cost is decode/synchronization, not packed-byte bandwidth.

A distinct warp-specialized cycle tried to overlap V staging with QK on the
eight otherwise idle warps. It reduced the instruction count to 351,357,400,
but barrier stall rose to 23.06%; the clean 128K time was `18.9727 ms`. Splitting
V staging evenly before and during QK made the imbalance worse at
`5.1712/22.6263 ms`. The overlap schedule is rejected; do not revisit it without
a producer/consumer pipeline that changes the measured barrier inequality. No
Qwen service gate was run for plain Q4.

### Per-head Q4 (Q4H) target-KV reopening gate, 2026-09-20

The next candidate removes the hot block-exponent plane and every per-chunk
integer scale shift. K and V each store 128 signed-Q4 code bytes plus one FP16
scale for the complete 256-value head record. This is a separately named,
strictly lossier format; it is not plain Q4 BFP and it never materializes an
F16 cache before attention.

```text
K record = V record = 128 + 2                           = 130 bytes
K+V per token/head/layer                                = 260 bytes
target page = 256 * 4 * 16 * 260                  = 4,259,840 bytes
target at 32,768                                  =   545,259,520 bytes
target at 131,072                                 = 2,181,038,080 bytes
target at 262,144                                 = 4,362,076,160 bytes
                                                     = 4.062500 GiB
target + existing Q8 MTP at 262,144               = 4.566406 GiB
weights + target KV + MTP                         = 17.2565 GiB
```

At 128K the target call touches at least
`13,625,700,352 + 2,181,038,080 = 15,806,738,432` hot bytes, or 16.89 ms at
936 GB/s and 6.22 GB per accepted token at `A=2.54`. Relative to the best
plain-Q4 result it must reduce attention from `18.4607` to `16.7527 ms`, a
9.25% reduction. Admission still requires byte-exact independent layout/query
checks, an independent attention oracle below `2e-4`, and the 32K/128K focused
profile. Pi fidelity remains a separate user-run gate because one scale per
head can lose substantially more key detail than block-floating Q4.

The first complete clean gate was byte exact with a `3.72529e-09` independent
attention error and measured `4.62438/16.9247 ms` at 32K/128K. The 128K result
missed admission by 0.1720 ms. The one permitted local correction combined the
two four-value decoders for each 32-bit Q4 word, reusing its nibble mask and
shift while retaining two PRMT outputs. The clean retest preserved the oracle
and measured `4.54246/16.5949 ms`, passing the `16.7527 ms` prerequisite.

The real Qwen gates then reached 54.54 useful tok/s at 32K and 47.16 at 128K
with the actual `top_k=20` sampler and MTP-4. Their complete cycles were
48.512 and 64.124 ms with 2.6667 and 3.0476 positions/call respectively.
Cleanup returned allocated and reserved pages to zero. This admits the
per-head format for the throughput goal but not for quality: it remains lossy,
and the user-run real-project Pi fidelity gate is still required. No 262K or
DeepSeek gate was run.

The user explicitly approved this lossy policy as the default for compatible
models after the throughput gates passed. This is an operational promotion,
not a fidelity claim. The discarded ad-hoc retrieval probe, RULER and
LongBench were not admitted as quality gates; a relevant fixed-harness
ContextBench run and the user-owned real-project Pi gate remain separate.

### Long-agent repetition policy, 2026-09-20

The first real Qwen Abliterated Pi fidelity run failed qualitatively at long
context. Requests with 133,610 and 148,441 prompt tokens entered lexical
reasoning loops and were manually cancelled after 12,952 and 79,608 generated
tokens. A coherent 4,786-token request between the failures, plus full fresh
prefill on each request, rules out a permanently corrupted retained session.
No matched K1 or F16 replay was run, so this evidence does not isolate Q4H as
the cause. It does reject the prior short-smoke inference that the promoted
configuration was adequate for long-agent use.

The initial mitigation used an artifact-declared `presence_penalty=1.5` for
thinking and non-thinking on both operational Qwen artifacts. It produced
visible Romanian lexical and grammatical degradation: common words were
displaced by malformed or semantically wrong alternatives. That value is
rejected for operational use. The corrected Qwen policy uses
`presence_penalty=0.5` in both profiles, with a 32,768-token ceiling only for
thinking output. Ornith declares the same 0.5 penalty while retaining its
existing `temperature=1.0`, `top_p=0.95`, `top_k=20` defaults in both profiles
and no Qwen-specific output ceiling.

The 262,144-position context and non-thinking output capacity remain
unchanged. The penalty is applied consistently to target and MTP proposal
distributions and changes sampling semantics intentionally. The ceiling is the
firm circuit breaker, not a quality fix: 16,384 was rejected because the same
session contained a coherent 23,599-token tool-producing turn. Requalify
Romanian quality and real long-agent behavior under 0.5; do not attribute a
pass or failure specifically to Q4H without a matched codec control.

Context capacity, normal output capacity and the thinking circuit breaker are
independent universal limits. Pi therefore advertises 262,144 context positions
and 262,143 maximum output tokens for both Qwen Q4H models; the server clamps
only thinking requests to their artifact-declared 32,768 bound. Sampling-policy
schema labels do not select runtime behavior, and no model-family branch
implements this distinction.

## Rejected DeepSeek work

- Whole-prompt layer-major routing required about 176 GiB of selection streams
  at 1,048,576 tokens.
- Blocking into eight 131,072-token segments still required about 23.516 TiB
  of exact activation traffic.
- A real 1M-token attempt remained in block one after about 6.5 minutes.
- Exact two-row batching left one 131K block incomplete after 14 minutes.
- Shared-expert block-128 INT8 generated incoherent text and lacked a matching
  artifact oracle.

Do not reintroduce `layer_major_prefill`, `causal_layer_major`, row-width
tuning or the shared INT8 layout without a new quantitative prerequisite.

## P100 decision

The installed RTX 3090 and two P100s have no CUDA peer access; every boundary
uses pinned host memory.

| Placement | Measurement | Required | Decision |
|---|---:|---:|---|
| Qwen exact KV split | 77.101 ms attention compute-only | <=32 ms | reject |
| Qwen compact dense shard | 35.939 ms/card before integration | <=20 ms | reject |
| Qwen lossless resident FP16 MLP shard | 50.871 ms auxiliary wall; 307.7/311.0 GB/s | >=500 GB/s/card | reject |
| Qwen Flash selected experts | 0.20 tok/s vs 0.37 primary | faster than primary | reject for speed |
| DeepSeek selected experts | 3.18 tok/s settled | >=10 tok/s | reject for speed |

The P100 executor may remain an optional exact capacity path for compatible
routed components. It is not a throughput default, a dense-Qwen provider or a
reason to extend Pascal kernels without a newly passing end-to-end inequality.

## Model-specific parked work

- **Qwen Flash behavior:** corrected sampling/effort propagation and native
  parser tracing did not fix zero-edit/action-stall behavior. Do not tune the
  parser or thinking again without checkpoint, quantization or numerical
  evidence.
- **Bonsai/binary/ternary:** these are differently trained models, not lossless
  Qwen FP4 conversions. They require explicit model/quality approval.
- **Local/fast-weight/episodic memory:** changes model semantics and requires
  training; it is a model-research hypothesis, not inference optimization.
- **Nemotron generic FP4:** weak `hi` passed, normal instruction failed. Reopen
  only for an official NVFP4 integration with its actual encodings.

## Reopening rule

Before implementation, record the exact model, populated context, formats,
hardware and useful-output target, then calculate:

```text
weights + recurrent state + KV + draft/MTP + workspace + reserve
bytes touched per call and per accepted token
GPU + RAM + PCIe + NVMe lower bounds
acceptance * proposer/verifier cost, if speculative
```

If the resulting capacity, latency, traffic or fidelity prerequisite fails,
stop before runtime code.
