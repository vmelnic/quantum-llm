# Artifact compiler

The compiler reads local SafeTensors checkpoints through bounded read-only
maps and publishes validated execution artifacts without instantiating a
Transformers model or materializing the whole checkpoint in RAM.

## Supported source contracts

- `hybrid_delta`: dense or MoE hybrid Transformer checkpoints with split
  Gated DeltaNet, periodic full attention, MTP and vision metadata;
- `muse_glimmer`: the strict official Muse-Glimmer source contract; it
  publishes dense text ABI-2 operations and preserves vision tensors as
  explicit auxiliary records;
- `qwen3_next`: Qwen3-Next MoE source adapter and MTP metadata;
- `lfm2_moe`: LFM2-MoE tensor and routed-expert source contract;
- `olmoe`: OLMoE source adapter;
- `deepseek_v4`: exhaustive read-only DeepSeek-V4-Flash contract used by the
  descriptor, compact-pack and independent-oracle commands.

Adapter names exist only at the source boundary. The published
`runtime-model.tsv` drives the common VM through roles, geometry, operations,
encodings and capabilities.

## Install

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\compiler
```

The optional `fast` extra accelerates bounded NumPy conversion but must produce
the same bytes as the dependency-free path.

## Inspect

```powershell
.\.venv\Scripts\python.exe -m compiler inspect-source `
  --source C:\path\to\snapshot --tensor-groups --adapter hybrid_delta
```

Inspection validates SafeTensors headers, index agreement, shard paths,
ranges, dtypes and the selected source contract without reading full tensor
payloads or writing an artifact.

## Compile a hybrid FP4 artifact

```powershell
.\.venv\Scripts\python.exe -m compiler compile `
  --source C:\path\to\snapshot `
  --output C:\path\to\qwen3.8-27b-fp4.candidate `
  --source-id Qwen/Qwen3.8-27B `
  --source-revision <immutable-commit> `
  --adapter hybrid_delta `
  --quant-profile fp4-e2m1-ue8m0-block32-v1

.\.venv\Scripts\python.exe -m compiler validate `
  C:\path\to\qwen3.8-27b-fp4.candidate
```

Normal operations should use
`ops/windows/Publish-HuggingFaceExpertPack.ps1`, which resolves the pinned Xet
snapshot, compiles a sibling candidate below `MODEL_ROOT`, validates it and
promotes it transactionally.

Compilation writes `<output>.partial` and resumes only when source bytes and
options are identical. `--reclaim-source-shards` is destructive and is never
part of the normal workflow.

Muse uses the same compiler/publication flow with `--adapter muse_glimmer`.
Its current executable program is text-only; preserved vision records do not
constitute a callable vision operation.

## DeepSeek commands

The CLI includes bounded commands to:

- validate the complete source geometry;
- qualify FP8/shared/expert numeric representations;
- export dense, typed, shared, routed and MTP descriptors;
- pack all routed experts into committed layer shards;
- export independent HCA, CSA, attention, MTP and model-I/O oracles.

These commands establish source and numerical contracts; a descriptor or
oracle is not itself an executable service artifact. See
[DeepSeek compact pack v1](../docs/deepseek-compact-pack-v1.md).

## Validation policy

Unknown tensors, fields, operation ABIs, record versions, geometries, hashes
or unsafe paths fail closed. FP4 completion also requires
`qualify-fp4-container` against the pinned source and execution through the
declared provider. The compiler/native validators—not a stale static JSON
schema—are the authoritative artifact contract.

See [Expert Pack v1](../docs/expert-pack-v1.md) and
[Expert Runtime contract](../docs/expert-runtime.md).
