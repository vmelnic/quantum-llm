# Research decisions

Status: canonical decision and rejection ledger, 2026-09-17.

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
| startup routed-cache fitting | artifact-neutral, before first expert admission, after future-state reservation | admitted for `ornith-k1`; not a global cache increase |
| lazy dense/logits workspaces | allocate only the rows/operation requested | saved about 994 MiB; arithmetic unchanged |
| exact hierarchical top-k | dense-provider sampling with artifact-declared `top_k <= 64` | preserves logits, candidate ids, tie order and NaN exclusion; does not change KV fidelity or prefill |

## Experimental K1 exception

K1 stores block-32 FP4 values and FP4 keys with one aligned FP16 outlier
correction per key block. It is lossy. The operational `qwen`,
`qwen-abliterated` and `ornith-k1` aliases opt into it; `qwen-f16` and
`ornith` are fidelity references. The abliterated alias is a separately
qualified third-party weight artifact with the same dense geometry; its
behavioral quality is not inferred from the official model.

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

Promotion requires explicit acceptance of changed fidelity plus reproducible
long-context retrieval, reasoning and coding comparisons against F16. Until
then K1 is an operational experiment, not the target semantics.

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
