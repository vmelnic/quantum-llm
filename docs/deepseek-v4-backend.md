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

The next dependency is CSA attention and its typed BF16/F32 compressor,
normalization, index, and cache state. After that, HCA can wrap a real attention
sublayer instead of the deterministic validation vector.
