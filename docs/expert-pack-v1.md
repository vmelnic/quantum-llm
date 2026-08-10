# Expert Pack v1

Expert Pack is a compute-ready, placement-oriented container for sparse MoE
inference. It is not only a quantization format: dense tensors and individual
experts are independently indexed, checksummed, aligned, and addressable from
SSD, RAM, VRAM, or a future remote worker.

The executable sources of truth are:

- `schemas/expert-pack-v1.schema.json`;
- `compiler/expert_pack/` writer and validator;
- runtime record readers under `runtime/`.

Unknown fields, versions, ABIs, flags, or record layouts are rejected.

## Published container

```text
model.expert-pack/
  manifest.json
  COMPLETED
  conversion-report.json
  dense.qpack
  experts-000.qpack
  experts-001.qpack
  ...
  tokenizer/
```

Conversion occurs in a neighboring `.partial` directory. Publication requires
complete pack writes, fsync, independent validation, and atomic rename. A
directory without a valid `COMPLETED` marker is not deployable.

## Manifest

The strict top-level object contains exactly:

```text
schema, format, compatibility, source, architecture,
quantization, kernel_abi, alignment, tensors, experts,
packs, indexes, masses, requirements, tokenizer, integrity
```

Key responsibilities:

- `source`: immutable checkpoint identity, file sizes and SHA-256;
- `architecture`: complete OLMoE or Qwen3-Next geometry/semantics;
- `quantization`: `int8-symmetric-per-row-v1` (quant ABI 1) or
  `fp4-e2m1-ue8m0-block32-v1` (quant ABI 3);
- `kernel_abi`: layout, activation, ordering, target architecture;
- `alignment`: record, section, direct-I/O, pinned and CUDA alignment;
- `tensors` / `experts`: complete record indexes;
- `packs`: size, record count and SHA-256 per file;
- `indexes`: canonical hashes of dense and expert indexes;
- `masses`: source/container/dense/expert/active byte accounting;
- `requirements`: compiler lower bounds for feasibility;
- `tokenizer`: local files, hashes, template and special tokens;
- `integrity`: canonical manifest content hash.

Canonical JSON is UTF-8, keys sorted, no insignificant whitespace, and
`ensure_ascii=false`. To calculate `integrity.content_sha256`, set that field to
the empty string, serialize canonically, then hash the bytes.

`COMPLETED` contains format version, canonical manifest-content SHA-256, and
the SHA-256 of the actual `manifest.json` bytes.

## Pack rules

- all multibyte values are little-endian;
- record offset and stored size are multiples of pack alignment (at least 4096);
- fixed record header is 256 bytes;
- payload sections are aligned to 256 bytes;
- all alignment padding is zero;
- manifest indexes are authoritative; readers do not infer records by scanning;
- record payload SHA-256 covers bytes `[record + 256, record + stored_bytes)`,
  including zero padding.

Common flags:

| Bit | Name | Value |
|---:|---|---:|
| 0 | row major | `0x01` |
| 1 | gate/up fused | `0x02` |
| 2 | symmetric | `0x04` |
| 3 | per-row scales | `0x08` |

## Dense record (`EPDENS01`)

The dense header declares version, flags, quant ABI, rank/dimensions, stored
bytes, data/scales offsets and sizes, tensor-name SHA-256, and payload SHA-256.

- FP32 dense records contain row-major data and no scale section;
- INT8 dense matrices contain one signed byte per value and one FP32 scale per
  output row;
- router matrices and rank-one normalization tensors remain FP32;
- direct casting to a native C struct is forbidden because packed `u64` fields
  are not naturally aligned.

## Expert record (`EPEXPR01`)

An expert record declares layer, expert ID, hidden/intermediate geometry, ABI,
stored bytes, four section offsets/sizes, and payload hash. Sections are:

```text
gate_up_q       I8  [2*I, H]  # all gate rows, then all up rows
gate_up_scales  F32 [2*I]
down_q          I8  [H, I]    # output-major
down_scales     F32 [H]
```

The expert has all common flags (`0x0f`). Sections cannot overlap and appear in
the declared order. Under quant ABI 3 the same section layout carries packed
FP4-E2M1 rows (two values per byte) with UE8M0 block scales instead of I8/F32.

## Quantization profile

For each decoded FP32 row:

```text
maximum = max(abs(row))
scale   = maximum / 127, or 1.0 for an all-zero row
q[i]    = clamp(round_ties_to_even(row[i] / scale), -127, 127)
```

Non-finite values are rejected. Scales are finite positive FP32. There are no
zero points. The two compiler implementations (dependency-free and optional
NumPy acceleration) must produce the same ABI and deterministic bytes.

The alternative FP4 profile (`fp4-e2m1-ue8m0-block32-v1`, quant ABI 3) stores
routed expert weights as packed FP4-E2M1 values with one UE8M0 scale per
32-value block along each output row — the same device format as the DeepSeek
compact path, consumed directly by the packed `__dp4a` kernels (kernel ABI
`expert-pack-sm86-fp4-block32-v1`). Scale codes are clamped to [1, 254]: 255
is the UE8M0 NaN and code 0 decodes inconsistently between toolchain and CUDA
kernel, so the compiler never emits it. Dense tensors, router matrices and
normalizations keep their INT8/FP32 encodings under both profiles.

## Architecture semantics

The adapter owns semantics that cannot be inferred from shapes. Examples:

- OLMoE query/key normalization and tensor naming;
- Qwen3-Next alternating full attention and Gated DeltaNet;
- attention output gate and partial RoPE;
- Qwen3-Next normalization `(1 + weight)` where specified;
- global softmax, top-k and selected-probability renormalization;
- explicit preservation of the auxiliary MTP tensors.

An adapter must classify every source tensor. “Ignore unknown” is not allowed.

## Validation

The validator fails closed for:

- absent/invalid `COMPLETED`;
- unknown schema, format, quant or kernel ABI;
- missing/extra manifest fields or tensors;
- file size/hash mismatch;
- invalid geometry, offsets, alignment, overlap or flags;
- nonzero padding;
- record/index hash mismatch;
- inconsistent byte accounting;
- missing tokenizer files or mismatched tokenizer hashes.

Startup repeats the relevant validation before allocating large runtime state.
