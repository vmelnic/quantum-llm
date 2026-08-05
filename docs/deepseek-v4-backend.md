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
changing Expert Pack v1 constants. Next, CUDA admission must consume the
compact source, populate exactly this slot, publish it only after completion,
and reproduce the qualified candidate hash/output.

## SM86 admission result

The first CUDA admission gate now passes on RTX 3090. A bounded fixture copies
the six original compact tensors for one expert without decoding or modifying
the checkpoint. CUDA transfers 13,369,344 bytes, decodes FP4/UE8M0, performs
per-row INT8 quantization, and fills the exact 25,198,592-byte hot slot.

The complete device slot matched the independent Python/PyTorch candidate
SHA-256 byte-for-byte. Measured H2D plus admission latency was 9.73 ms for the
first real expert. The cache must not publish a slot until this conversion and
source-validity check succeed.

This number also constrains the scheduler: six serial cold admissions would
consume roughly 58 ms before GEMM, so 30 tok/s cannot depend on cold loading at
every layer. Hot residency, look-ahead prefetch, concurrent admission, and
request batching remain essential. The next gate executes gate/up/down GEMM
directly from the admitted slot and compares its output with a CPU reference.
