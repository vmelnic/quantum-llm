# DeepSeek-V4-Flash backend

DeepSeek-V4-Flash is the next architecture backend, not an input alias for the
Qwen runtime and not an automatic Expert Pack v1 conversion.

## Pinned source contract

The `deepseek-v4-flash-source-v1` contract validates the complete checkpoint
before payload decoding begins. It requires a one-to-one partition of every
tensor name and validates dtype, shape, byte length, layer/expert bounds, the
CSA compression schedule, MTP namespace, hash-router tensors, and FP4/FP8
quantization metadata.

Run the read-only gate with:

```powershell
.\.venv\Scripts\python.exe -m compiler inspect-source `
  --source C:\path\to\snapshot `
  --contract deepseek_v4
```

The pinned checkpoint contract covers:

- 43 transformer layers and one MTP layer;
- 256 routed experts per layer, top-6 routing, and one shared expert;
- packed FP4 E2M1 routed weights with UE8M0 block-32 scales;
- FP8 E4M3 dense/shared matrices with UE8M0 128×128 block scales;
- CSA compressors/indexers and manifold-constrained hyper-connections;
- 1,048,576 configured positions.

Passing this gate proves source identity and metadata compatibility. It does
not prove numeric decoding, model correctness, runtime support, context
capacity, or throughput.

The pinned revision has passed this gate on the target Windows host: 46 shards,
69,187 tensors, and 159,609,485,896 tensor bytes formed an exact contract and
role partition. No payload or source file was changed by the qualification.

## ABI boundary

Expert Pack v1 remains the tested OLMoE/Qwen INT8-per-row format. DeepSeek will
receive a separate, versioned representation because expanding all compact FP4
experts to INT8 would increase storage and I/O without benefiting every compute
tier.

The planned SM86 placement model is:

1. keep the cold expert directory in compact FP4 plus scales on SSD/RAM;
2. decode only admitted hot experts into a compute-ready RTX 3090 cache;
3. keep dense/attention state under a separate SM86 ABI;
4. preserve the compact source representation for future CPU and distributed
   expert workers.

No full conversion starts until the FP4/UE8M0 decoder, a real expert slice,
numeric comparison, and output-space estimate pass independently.

## Numeric source decoding

The dependency-free source decoder implements the published E2M1 finite table,
low-nibble-then-high-nibble packing order, one UE8M0 scale per 32 logical expert
values, and row/block streaming. All 16 FP4 codes and all 255 finite UE8M0 codes
are tested; the UE8M0 NaN code, invalid geometry, and non-finite output fail
closed.

This decoder establishes source semantics only. It does not select the final
GPU-cache representation and is not wired into Expert Pack v1.

## Representation estimate

`--estimate-representations` derives payload bytes from the validated tensor
geometry without opening weight payloads. It intentionally excludes container
headers, alignment, indexes, runtime buffers, dense decode expansion, and KV
cache, so it is a format comparison rather than a disk/VRAM capacity promise.

On the pinned checkpoint:

| Representation | Payload bytes | Consequence |
|---|---:|---|
| Source, compact routed FP4 | 159,609,485,896 | Baseline; routed experts occupy 150,592,290,816 bytes. |
| All routed experts expanded to FP8 | 292,502,338,120 | Adds 132,892,852,224 bytes before container overhead. |
| All routed experts expanded to INT8-per-row | 292,854,135,368 | Adds 133,244,649,472 bytes before container overhead. |

One main-model expert is 13,369,344 source bytes and 25,198,592 bytes in the
candidate INT8 compute cache. Six selected experts for one layer require
151,191,552 cache bytes if none are already resident. A nominal 1 GiB cache
holds 42 such experts, before allocator/workspace overhead.

Therefore eager expansion is rejected as the default direction. Compact cold
storage plus bounded, eviction-aware compute admission remains the working ABI
decision; the real expert slice must still validate quality and decode cost.

## Real expert vertical slice

`qualify-deepseek-expert` reads all three projections of one routed expert,
decodes them in bounded row chunks, compares every decoded FP32 bit with an
independent PyTorch `float8_e8m0fnu` reference, and incrementally hashes a
candidate INT8-per-row representation. It retains no converted weight file.

```powershell
.\.venv\Scripts\python.exe -m compiler qualify-deepseek-expert `
  --source C:\path\to\snapshot --layer 0 --expert 0 --row-chunk 128
```

The first complete slice (layer 0, expert 0) processed 25,165,824 logical
values from 13,369,344 source bytes. Reference equality was bitwise exact. The
25,198,592-byte INT8 candidate measured RMSE 0.0002269152 and maximum absolute
error 0.0007381886. End-to-end qualification, including the PyTorch comparison,
took about two seconds on the target host. Row chunks of 64 and 256 produced
the same candidate SHA-256, proving that chunking does not alter the candidate
byte stream.

This passes source decode correctness for one complete expert. It does not yet
prove model-level quality, CUDA kernel throughput, cache admission latency, or
the final on-disk/cache ABI.

## Frozen expert boundary v1

Two ABIs are now distinct and platform-neutral:

- `deepseek-fp4-e2m1-ue8m0-block32-v1` describes immutable compact source
  tensors. It is not a new copy of the checkpoint and remains suitable for
  SSD, RAM, CPU, or remote ownership.
- `deepseek-sm86-int8-per-row-v1` describes one admitted compute slot on RTX
  3090. It is derived and evictable; it is never the authoritative model copy.

The SM86 slot is 256-byte aligned and contains:

| Section | Offset | Bytes |
|---|---:|---:|
| fused w1 gate + w3 up INT8 rows | 0 | 16,777,216 |
| fused w1 + w3 FP32 row scales | 16,777,216 | 16,384 |
| w2 down INT8 rows | 16,793,600 | 8,388,608 |
| w2 FP32 row scales | 25,182,208 | 16,384 |
| total slot | 0 | 25,198,592 |

The runtime exposes this geometry as a backend-neutral contract rather than
changing Expert Pack v1 constants.

## SM86 admission result

The first CUDA admission gate now passes on RTX 3090. A small descriptor names
the six original SafeTensors ranges for one expert without decoding, copying,
or modifying the checkpoint. The production cache gathers the resulting
13,369,344 logical bytes with IOCP into a pinned buffer. CUDA then decodes
FP4/UE8M0, performs per-row INT8 quantization, and fills the exact
25,198,592-byte hot slot.

The complete device slot matched the independent Python/PyTorch candidate
SHA-256 byte-for-byte. Direct H2D plus admission measured about 9.5 ms for the
first real expert. A full cold cache acquisition—including unbuffered I/O,
SHA-256 validation, admission, publication, and two concurrent waiters—measured
49.6 ms with direct source extents. Both waiters shared exactly one logical
read and one upload. The CUDA directory
saw the entry only after completion, and eviction released the complete hot
allocation.

This number also constrains the scheduler: six serial cold admissions would
consume roughly 58 ms before GEMM, so 30 tok/s cannot depend on cold loading at
every layer. Hot residency, look-ahead prefetch, concurrent admission, and
request batching remain essential.

That GEMM gate now also passes for the same real expert. The existing SM86
INT8-per-row MoE kernels consumed the admitted slot without repacking and
completed gate + up + SiLU + down in 0.39–0.52 ms for one deterministic
activation.
Against the CPU calculation, output RMSE was `1.02e-8` and maximum absolute
error was `4.47e-8`.

This proves the compact source → CUDA admission → compute path end to end for
one expert. It does not imply model tok/s: top-6 scheduling across 43 layers,
shared experts, dense FP8 tensors, CSA/HCA attention, routing, KV state, cache
misses, and concurrent requests still have to be integrated and measured.

The shared cache now distinguishes immutable `stored_bytes` from exact
`device_bytes`. For DeepSeek these are 13,369,344 and 25,198,592 respectively.
VRAM capacity is reserved using the expanded hot size before SSD I/O starts;
an undersized admission remains absent instead of loading data it cannot
publish. Qwen retains its legacy equal-size behavior through a zero/default
device-size claim.

## Autoregressive route locality

The full-model runner now records the exact six routed expert IDs for every
layer and output-producing model step. The trace is bounded by the 43-layer
decode range, excludes the invariant shared expert, preserves router rank, and
is copied as part of the directory plan's existing host synchronization.

An eight-step real generation showed that the former 64-slot global INT8 cache
was a cyclic-scan policy failure, not evidence that every route was novel.
Consecutive routes reused 840 of 1,806 selections (46.5%), but only one of 301
layer transitions retained the complete six-expert set. Retaining more INT8
history cannot solve the capacity problem:

| Prior per-layer routes | Route hit ratio | Maximum routed slots | INT8 bytes |
|---:|---:|---:|---:|
| 1 | 46.5% | 258 | 6.50 GB |
| 3 | 57.8% | 526 | 13.25 GB |
| 7 | 75.2% | 905 | 22.80 GB |

The seven-route case no longer fits beside 7.77 GB of dense state and 1.08 GB
of shared experts. The production direction is therefore a tiered compact
cache: pageable compact FP4 in RAM, compact FP4 in VRAM, and only the current
six selections expanded into transient SM86 INT8 compute slots. Route traces
remain evidence for admission and prefetch; they are not a hard-coded routing
oracle.

The first tier is now wired into the full-model runner as a bounded pageable
RAM cache. `HostCacheGiB` accepts `0…48`, defaults to 32 for prompt runs, and
uses a 7/8 low watermark while pinned staging remains a separate fixed pool.
On the same eight-token sequence, a 32 GiB budget reached a 20.40 GB high
watermark, reduced total source reads from 35.80 GB to 20.40 GB, and recorded
1,152 decode RAM hits. Post-prefill decode read 7.94 GB and improved from 0.311
to 0.516 tok/s with identical token IDs. The extra pageable copy increased
cold prompt time, so this is an L2 placement result, not an SLO pass.

The CUDA uploader now owns a separate routed-only compact L1. Its byte budget
is independent of expanded slots, lookup is keyed by immutable expert identity,
and LRU touch/eviction are O(1). An 8 GiB run reached 8.58 GB, recorded 1,003
hits and 1,594 misses, evicted 952 records, and transferred 21.31 GB of compact
payload. The generated IDs were unchanged and decode improved only from 0.516
to 0.553 tok/s. This validates tier separation, but also isolates repeated
FP4→INT8 admission and compute-slot churn as the dominant miss cost.

The checkpoint remains authoritative: the qualification bundle now contains
only a manifest and a six-row extent descriptor, about 3 KiB total. No copied
expert payload is retained. Extents must cover the compact destination exactly
without gaps or overlap; the assembled payload still passes one whole-record
SHA-256 gate before admission. This keeps storage growth independent of model
parameter count and preserves the same cache/admission ABI for future 1T-class
checkpoints.

## FP8 shared expert

The always-active shared expert uses a third source ABI,
`deepseek-fp8-e4m3-ue8m0-block128-v1`. Its three projections contain
25,167,360 source bytes. E4M3FN values and UE8M0 block scales were decoded
bit-for-bit identically to PyTorch for the complete layer-0 shared expert.

The shared expert does not require another compute layout: admission produces
the same 25,198,592-byte `deepseek-sm86-int8-per-row-v1` slot used by routed
experts. The real slot matched the independent candidate SHA-256 exactly. A
full cold gather, checksum, CUDA conversion, publication, and two-waiter
single-flight acquisition measured 88.2 ms; execution measured 0.40 ms, with
maximum output difference `1.08e-8` against the CPU calculation.

Because one shared expert is used on every token at every layer, production
placement should treat these 43 slots as dense/resident model state rather than
router-driven cache entries. They occupy about 1.01 GiB in total before
allocator overhead. The validated cache lifecycle remains the loading and
integrity mechanism.

Startup residency now loads and pins all 43 shared experts through one fixed
25 MB staging slot. The source gather read 1,082,196,480 bytes and published
1,083,539,456 bytes of compute-ready slots in 3.59 seconds. Every layer was
visible in the CUDA directory, and teardown released the complete logical
allocation. VRAM headroom is checked before the first read; partial startup
failure releases every lease acquired by that startup transaction.

## Dense FP8 matrices

The main model contains 236 block-scaled FP8 matrices: 4,775,215,104 weight
bytes and 291,456 UE8M0 scale bytes. Another 2,988,256,348 bytes are dense
BF16/F32/I64 state and require their own typed loaders. The FP8 mass is small
enough to remain resident on a 24 GB GPU after the 1.01 GiB shared set, while
still leaving the routed-expert cache bounded separately.

`deepseek-sm86-int8-per-row-matrix-v1` is the generic dense compute ABI. The
first real projection, `layers.0.attn.wq_a` (1024×4096), was gathered directly
from SafeTensors and decoded bit-for-bit identically to PyTorch. Its
4,194,560-byte source became 4,198,400 device bytes; the candidate hash matched
exactly. Admission took 5.74 ms and the existing dense GEMV took 0.674 ms, with
maximum output difference `1.79e-7` against the CPU calculation.

This proves one reusable dense matrix boundary, not the complete attention
layer. Next the same loader must own all 236 FP8 matrices, while BF16/F32 HCA,
normalization, compressor and router tensors retain their declared types.

That resident dense loader now passes on the complete main-model FP8 set. It
read and checksummed 4,775,506,560 source bytes, admitted 4,783,917,056 device
bytes, and made all 236 matrices available by stable tensor name in 18.03
seconds. One 33,556,480-byte staging slot bounded host-pinned memory. The
RTX 3090 still reported about 19.68 GB free while the complete dense FP8 set
was resident in the qualification process.

The dense set is transactional: duplicate names or inconsistent geometry fail
before I/O, and any later checksum/admission failure destroys every matrix in
the partial candidate. This set will be composed with shared residency in the
model owner; neither allocation class competes with routed-cache eviction.

## Manifold-constrained hyper-connections

The HCA implementation follows the checkpoint author's
[reference inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)
and the independent
[Transformers DeepSeek-V4 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py).
Four residual streams are flattened and RMS-rescaled, then projected by the
site's F32 `fn` matrix into 24 mixing values. `pre` collapses the streams,
`post` places the sublayer result back into them, and the 4×4 `comb` matrix is
projected with the checkpoint's 20-step Sinkhorn normalization order.

The runtime retains `fn`, `base`, and `scale` in F32 and exposes reusable CUDA
`deepseek_hca_pre` and `deepseek_hca_post` primitives. It does not collapse the
four streams permanently or substitute an ordinary residual connection.

A real `layers.0.hc_attn_*` slice passed against a NumPy FP32 oracle generated
from the official equations. The 1,572,972 source bytes remain the same size on
device. Admission measured 0.48 ms; the initial unfused pre + post path measured
1.96 ms. Maximum absolute error was `3.87e-7` for `pre/post/comb` and `1.04e-7`
for collapsed/expanded streams. This closes the HCA correctness boundary, but
the measured path is not yet performance-final: its 24-row control GEMV and
small Sinkhorn kernel can later be fused after the complete layer is correct.

HCA and the CSA compressor, cache, sparse-attention, and ratio-four index
primitives are now independently qualified. The remaining boundary is their
composition with the resident dense and typed sets into a complete attention
sublayer, followed by workload-shaped kernel optimization.

## Typed model state

The remaining 834 main-model tensors use BF16, F32, or I64 and occupy
2,988,256,348 bytes. `DeepSeekTypedSet` retains those dtypes on device rather
than eagerly expanding BF16 to F32. Stable name lookup is confined to model
construction; bound execution paths consume typed pointers.

Loading is transactional and chunked. Each device tensor receives one final
allocation, but source reads, incremental SHA-256, and H2D copies pass through
one fixed 64 MiB pinned slot. Consequently the roughly 1 GiB embedding/head
tensors do not require equally large host staging allocations. A failed read,
hash, allocation, or upload releases the partial candidate and leaves the
published model state empty.

The complete real set loaded in 9.97 seconds. All 834 name lookups succeeded,
and resident bytes exactly matched the 2,988,256,348-byte source contract. The
RTX 3090 reported 21,452,816,384 free CUDA bytes while this set alone was
resident. This is the typed storage boundary required by HCA, normalization,
CSA compressors/indexers, routing, embedding, and the output head; individual
kernels still have to enforce each tensor's dtype and shape contract.

## Transactional resident model

`DeepSeekResidentModelState` combines the complete 236-matrix expanded dense
set and all 834 dtype-preserving tensors behind one publication boundary. The
dense phase may complete internally, but a failure while streaming typed state
destroys the candidate and leaves the destination empty. This prevents a
runner from observing a partially ready model.

On the real checkpoint the combined state published 1,070 named objects and
occupied 7,772,173,404 device bytes. It loaded through one 64 MiB pinned
staging slot in 28.0 seconds and left 16,665,018,368 CUDA bytes free on the
24 GiB qualification GPU. The source representation covered 7,763,762,908
bytes; expansion of dense FP8 matrices accounts for the device/source delta.

Layer bindings are non-owning views tied to the model state's lifetime. Model
construction resolves names once and rejects wrong dtype, byte length, matrix
geometry, layer ID, or compression schedule. Real bindings passed for layer 2
(ratio 4, including its indexer) and layer 3 (ratio 128). FFN bindings now
resolve the norm, BF16 router matrix, HCA controls, and exactly one routing
metadata form: the I64 token-to-expert table for hash layers 0–2 or the F32
selection bias for learned layers 3–42. Both real layer-2 and layer-3 contracts
passed. Decode can therefore consume direct pointers without names, source I/O,
or admission inside the per-token path.

## CSA decode compressor

The first CSA primitive now consumes the checkpoint's BF16 `wkv` and `wgate`
projections directly, adds the F32 per-position `ape`, maintains persistent
decode state, performs dimension-wise gated softmax pooling, and applies the
BF16 RMSNorm weight. Its behavior follows the checkpoint author's
[Compressor reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py).

One implementation covers both configured schedules:

- ratio 4 stores eight projected rows and combines the previous group's first
  512 dimensions with the current group's second 512 dimensions. After emit,
  the current four full-width rows become the next overlap history;
- ratio 128 stores 128 ordinary 512-wide rows and emits at the group boundary.

The layer-2 ratio-4 compressor passed against an independent NumPy oracle using
its complete real parameter slice. State occupies 65,536 bytes; four
projection/update steps took 1.15 ms and maximum output error was
`3.81e-6`. The layer-3 ratio-128 compressor passed on 8,651,776 source bytes;
state occupies 524,288 bytes, the 128-token group took 7.34 ms, and maximum
error was `1.43e-6`.

These measurements include both BF16 matrix-vector projections for every
token, state update, final pooling, and RMSNorm. They do not include compressed
KV RoPE/QAT simulation, cache publication, indexing, or sparse-attention time;
those boundaries are qualified separately below and will be timed together
only after composition into the complete attention layer.

The emitted vector now also crosses the real cache boundary. The runtime first
rounds the normalized vector to BF16, applies RoPE to only the final 64
dimensions, and runs the checkpoint's in-place MXFP8 E4M3 quantize/dequantize
simulation over seven 64-value blocks in the first 448 dimensions. Power-of-two
scales use the same ceiling rule and `1e-4` minimum as the official QAT kernel.
The result is stored as 512 BF16 values (1,024 bytes per compressed slot), not
as an F32 cache.

Both the ratio-4 and ratio-128 real slices matched independent PyTorch/NumPy
cache oracles with zero BF16-word differences. Keeping this cache typed halves
its VRAM footprint relative to F32 and freezes the producer ABI that the sparse
attention consumer will read next.

The first sparse-attention consumer now reads BF16 query/cache tensors, accepts
`-1` sentinel indices, includes the real per-head F32 `attn_sink` in the
softmax denominator with a zero value vector, and accumulates output with an
online softmax. Its workspace is constant with respect to context length and
selected-position count.

On the layer-2 fixture, 32,767 of 32,768 BF16 output values matched the
independent oracle exactly; the remaining value differed by `3.81e-6`. The
four-index fixture took 0.047 ms, but this is a correctness measurement rather
than a production top-k claim. The configured path may consume a 128-position
window plus as many as 512 indexed compressed positions, so the scalar
per-position loop must be tiled/batched after the real indexer and complete
attention layer establish the final workload.

## Ratio-four index selection

Ratio-four layers now run the checkpoint's second, 128-dimensional overlap
compressor from its real BF16 `wkv`, `wgate`, and norm tensors plus F32 `ape`.
The emitted position is rounded to BF16 after RoPE64, a scaled Hadamard-128
rotation, and the official block-32 E2M1 quantize/dequantize simulation. Query
rows cross the same preparation boundary.

Selection then computes 64 rectified query-to-position dot products, weights
them with the real BF16 `weights_proj` output and the checkpoint's two scale
factors, and returns a stable descending top-k. Lower position IDs win exact
score ties, so selection is deterministic.

The real layer-2 slice passed the independent oracle with:

- `1.97e-6` maximum error from the 128-wide index compressor;
- zero BF16-word differences after query/cache preparation;
- `2.98e-8` maximum score error;
- exact top-3 ordering.

The measured 0.076 ms selected three of six fixture slots. It establishes the
math and ABI only; it is not a production top-512 performance claim. The
current stable selector is intentionally simple and must be replaced by a
parallel bounded top-k after the complete layer fixes batching, context, and
cache-placement geometry. The already qualified dense FP8 set owns the query
projection used before this boundary; the vertical slice supplies its
deterministic output rather than duplicating that dense-kernel proof.

## Complete decode attention sublayer

`DeepSeekAttentionState` is a bounded per-request object. Creation takes a
compression schedule and admitted maximum context, allocates the BF16 window,
compressed and optional index caches plus all workspaces once, and initializes
the persistent compressors. `deepseek_attention_decode` performs no model-name
lookup, source I/O, admission, or allocation.

The one-token executor now composes:

1. HCA normalization, Sinkhorn controls, and stream collapse;
2. attention RMSNorm and the resident `wq_a`/`wq_b` query path;
3. per-head query normalization and RoPE64;
4. window KV projection, normalization, QAT boundary, and ring-cache update;
5. the main compressor and, for ratio four, the index compressor/query/top-k;
6. online sparse attention over window plus compressed positions;
7. inverse RoPE, eight grouped `wo_a` projections, `wo_b`, and HCA expansion.

An independent compiler oracle reconstructs every FP8 dense matrix through the
same declared SM86 INT8-per-row ABI, but reproduces admission and warp
accumulation outside the CUDA executor. Four sequential tokens on real layer 2
exercise real RoPE positions, retain the same state, and at token three emit
and consume the first main/index compressed slot. Across the 65,536 updated
HCA values, the executor matched with RMSE `3.07e-5` and maximum error
`4.02e-4` under a fail-closed `1e-3` composed limit. The gradual per-token
maximum (`2.7e-5`, `7.9e-5`, `1.65e-4`, `4.02e-4`) is consistent with the
different online/warp reduction order rather than a discontinuity at
compression. A context-4096 request state used 2,149,120 bytes.

The initial four-token composed measurement averaged 7.22 ms per layer-token.
Vectorizing aligned INT8 GEMV loads to four weights/activations per lane reduced
the same qualified path to 6.74 ms, a 6.6% improvement with no semantic change.
Vectorizing BF16 compressor loads and copying activations into shared memory
both regressed the complete layer and were removed.

A symmetric activation-INT8 DP4A path was also qualified and removed. It
quantized each reused activation once and applied DP4A to every dense attention
projection. The composed layer regressed to 7.45 ms per token, while the added
activation representation changed the prior output by RMSE `5.97e-3` and
maximum absolute error `5.34e-2`. This rules out the implemented per-vector
quantize-then-DP4A design on SM86; it does not rule out a fused or tensor-core
ABI that avoids standalone quantization and launch costs.

The eight `wo_a` group projections now share one grouped-input GEMV launch.
This preserves the original per-row accumulation and output bit behavior while
removing seven launches from each layer-token. The qualified path measured
6.74 ms per token versus 6.78 ms in the immediately preceding baseline run;
that difference is within run-to-run noise and is not counted as a throughput
gain.

This remains far from target throughput: multiplying 6.74 ms by 43 already
exceeds 289 ms before MoE. The generic one-warp-per-row GEMV still achieves
only a small fraction of RTX 3090 memory bandwidth. The next optimization
boundary remains the actual composed workload: DP4A/tensor-core-capable
weight/activation layouts with an explicit accuracy gate, fused
query/norm/RoPE, batched requests, and parallel top-k.
The rejected standalone activation-INT8 DP4A path must not be repeated.
Component-fixture timings will not be optimized in isolation.

## FFN route and dispatch boundary

FFN decode is split at the necessary control-plane boundary. `route` applies
FFN HCA pre, BF16 RMSNorm, the BF16 router projection, and checkpoint-exact
`sqrt(softplus)` scoring. Layers 0–2 select six experts through the immutable
I64 token table. Later layers select by unbiased score plus F32 correction
bias, but derive routing weights from the unbiased scores. Selected weights
are normalized and multiplied by the checkpoint route scale `1.5`.

The device route contains seven IDs: six routed experts followed by shared
expert 256. The control plane can therefore pin/load the complete dependency
set through one directory operation. Only after that succeeds does `execute`
run routed and shared SwiGLU, add their outputs, and apply FFN HCA post. Both
expert paths enforce the checkpoint's gate/up clamp of `10.0`; no expert
pointer is dereferenced before the directory pin boundary. SwiGLU output is
rounded to BF16 before the down projection, matching the checkpoint reference;
this behavior is explicit in the DeepSeek launch and does not alter Qwen.

On the four real layer-2 attention outputs, hash IDs matched the independent
checkpoint oracle exactly. Routing weights matched with RMSE `7.94e-6` and
maximum absolute error `2.39e-5`. HCA pre, FFN norm, router projection and
selection together measured 0.79 ms per token. The bounded FFN request state
uses 190,976 bytes.

The first complete transformer block is now qualified at layer 2, token three,
after that token emitted and consumed its first compressed attention slot. The
cache gathered 105,383,424 authoritative source bytes for the six selected
routed experts plus shared expert, admitted 176,390,144 compute-ready bytes,
and published all seven in 410 ms on the cold path. The directory then pinned
the exact seven dependencies before compute.

Hot execution measured 6.80 ms for attention, 0.78 ms for FFN HCA/norm/route,
and 7.19 ms for routed top-6 plus shared expert and FFN HCA post. The 16,384
final stream values matched the independent full-block oracle with RMSE
`1.04e-4` and maximum absolute error `4.76e-4`. Cold admission is not included
in the 14.77 ms composed compute path.

This is architectural correctness, not the throughput target: if every one of
43 layers had the same cost, dense plus MoE compute alone would exceed 630 ms
per generated token. The next phase must extend ownership across all layers
while replacing per-token GEMV-style work with batch/tensor-core execution and
overlapping expert readiness across requests.

### Pure sliding-window layers

Layers 0, 1, and 42 have `compress_ratio=0` in the pinned source contract. They
do not instantiate a compressor or indexer and use only the 128-token circular
window with base RoPE. The runtime now treats zero as a first-class attention
mode: its state allocates no compressed/index cache, its binding requires no
compressor tensors, and decode never performs compressor work or divides by a
ratio.

A real layer-0 four-token block passed the independent checkpoint oracle with
attention RMSE `1.18e-5`, maximum attention error `1.41e-4`, full-block RMSE
`3.55e-5`, and maximum full-block error `2.20e-4`. This validates the missing
schedule endpoint required before a 43-layer request owner can be constructed.

### Request ownership across 43 layers

`DeepSeekRequestState` now owns one persistent attention state and one FFN
state for every checkpoint layer. It retains a shared reference to the
immutable resident model, so pre-bound weight pointers cannot outlive their
owner. Construction is transactional and budgeted:

1. compute the exact state footprint without allocating;
2. reject an insufficient per-request CUDA budget with backpressure;
3. bind the fixed 3×ratio-zero, 20×ratio-four, 20×ratio-128 schedule;
4. allocate all layer states into a private candidate;
5. publish only after all 43 layers are complete.

At a 4,096-token context the complete state is 79,019,044 bytes: 70,060,544
bytes of attention/cache/workspace state, 8,211,968 bytes of FFN workspace,
615,460 bytes for HC-head/logits/argmax state, and 131,072 bytes for the two
four-stream ping-pong buffers.
The real model gate confirmed all 43 bindings and rejected a budget one byte
below the estimate before allocation. Execution order is owned separately by
the decode controller and its non-blocking outer scheduler, so request-state
allocation does not also become queue policy.

Directory pin lifetimes are now identified by independent bounded tokens. Two
requests can retain the same or different ready experts concurrently, and a
miss route may retain its ready subset while the remaining experts load. The
real seven-expert DeepSeek route held two simultaneous tokens, executed the
block, and released both independently. This removes the former single-global-
pin serialization point required before request-level overlap can be added.

The layer control loop is now suspendable. It executes attention and routing
once, returns the exact misses to the outer scheduler, retains no hidden
unbounded work, and resumes the same FFN after publication. On the real layer-2
route it suspended with seven cold misses, loaded those experts through the
normal extent-gather/cache/admission path, resumed, and reproduced the full
block oracle with `4.76e-4` maximum error. The all-resident path produced the
same bound. A full request can use range `[0, 43)`; strict subranges remain
available for qualification and future pipeline placement.

The production routed catalog now indexes all 11,008 `(layer, expert)` records
without generating per-route directories or copying model weights. A strict
runtime parser reconstructs the six exact-cover source extents, hash, compact
source ABI, and derived SM86 allocation for each key. Publication is atomic;
truncated, reordered, path-escaping, or geometrically incomplete catalogs fail
closed. A real boundary smoke loads `(0, 0)` and `(42, 255)` through IOCP,
extent gather, SHA-256 verification, compact FP4 admission, cache publication,
CUDA directory pinning, and independent release.

`DeepSeekDecodeScheduler` now owns the outer asynchronous boundary. It advances
runnable controllers in round-robin order, limits both request count and global
in-flight cache acquisitions, and never waits inside `poll()`. Cache acquisition
remains deduplicated by immutable expert key. A suspended request retains every
successful lease until the same layer replans and executes; failure or
cancellation releases acquire handles, leases, and directory pins explicitly.
The shared expert is a startup-resident invariant, so the cold scheduler accepts
only routed IDs 0–255 and fails closed if expert 256 disappears.

The real layer-2 gate kept the shared expert resident, admitted six routed
experts from the complete catalog with a two-acquire global limit, and resumed
without rerunning attention or routing. It observed six completed acquisitions,
a peak of two in flight, and reproduced the block oracle with `4.76122e-4`
maximum error.

The resident model now binds the untied BF16 embedding, final norm, output
head, and F32 HC-head controls without creating another weight allocation.
Request-owned I/O state expands one embedding row into four streams, applies
the official BF16 boundaries around HC collapse/final RMSNorm, projects all
129,280 logits, and computes greedy argmax. Against a bounded independent
oracle, embedding was bit-exact, logits had `6.89e-7` RMSE and `5.72e-6`
maximum error, and argmax matched token 65,270.

The native token loop now joins that front/back path to all 43 layers while
retaining request-owned attention state across prompt positions. The launcher
uses the checkpoint's official `encoding/encoding_dsv4.py` codec and
`tokenizer.json`; it does not approximate the chat format. RoPE is generated
per position for base, YaRN-compressed, ratio-four group-start, and ratio-128
group-start coordinates.

The first real chat prompt, `Hi`, encoded to the official five-token sequence
`[0, 128803, 23166, 128804, 128822]`. The runtime consumed all five positions,
executed 43 layers per position, projected the complete vocabulary, and
greedily produced token `19923`, which decodes to `Hello`.

Prefill now traverses layer-major: bounded device buffers retain every prompt
row while one layer processes positions in causal order. The exact same prompt
fell from 19.91 to 3.74 seconds (5.32x), and routed acquisitions fell from 1,286
to 859 because expert residency is reused across prompt rows before advancing
to the next layer. Grouped multi-row expert kernels remain a separate
optimization.

The serving boundary is still incomplete: this runner currently exposes one
generated token as a qualification path, not a persistent HTTP worker. Cold
expert admission also remains the dominant cost. A 43-partition cache was
rejected after it caused severe allocation/admission churn on changing routes;
the bounded global 64-expert window completed the same prompt more than 15×
faster. Persistent compute-ready expert packs are the next cold-path boundary.

An attempted direct FP4 CUDA prototype retained 13,369,344 bytes per routed
GPU entry instead of 25,198,592, kept `Hi → Hello` unchanged, and stayed
within the full-block tolerance (`4.23e-3` maximum error). Scalar on-the-fly
dequantization nevertheless regressed the FFN block from 6.94 to 38.64 ms and
the layer-major prompt from 3.74 to 26.75 seconds, so that execution path was
removed. A future compact path must expand during admission or use grouped
rows/Tensor Cores; it must not decode every weight inside a scalar GEMV.

CUDA expert storage now supports an explicitly bounded recycler. Cache
eviction retires the directory entry and returns its exact-size allocation to
the uploader instead of freeing it; later admission overwrites and republishes
the slot. Compact-source staging is also reusable because uploader execution is
serialized. The five-token gate reduced 902 logical publications to 107 device
allocations plus 795 slot reuses without changing output or materially changing
latency. The recycler capacity is supplied by placement policy rather than
being inferred from total model size, so metadata and storage remain bounded
for substantially larger models.

Autoregressive execution now continues after the first sampled token and
reports TTFT separately from decode latency. In the first two-token run, the
second token took 3.83 seconds and triggered 258 new routed acquisitions,
exactly six per layer. This establishes that global LRU/LFU turnover, rather
than request-state reconstruction, dominates current single-stream decode.

Routed down projections now materialize six selection outputs in parallel and
aggregate them in stable routing order. This reuses the runtime's qualified
selection-batch kernels, preserves the three-token greedy sequence, and keeps
the block maximum error below `5e-4`. It adds about 98 KiB per layer of FFN
request workspace. The change removes serialization across top-k down GEMVs;
it does not remove expert reads or implement Tensor Core weight kernels.
