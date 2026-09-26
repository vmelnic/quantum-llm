# Artifact compiler

The compiler inspects local SafeTensors checkpoints with bounded read-only
maps and publishes validated execution artifacts without loading a Transformers
model or materializing the checkpoint in RAM.

## Source adapters

Adapters are allowed only at the checkpoint boundary. They translate upstream
names and configuration into tensor roles, geometry, operation capabilities and
encodings. The common runtime never selects a model family from these names.

| Adapter | Source contract |
|---|---|
| `hybrid_delta` | supported dense or MoE hybrid DeltaNet/full-attention checkpoints; used by Qwen3.8-27B and Ornith |
| `qwen4_exp` | Hyper, PLE, Gated DeltaNet, QSA and routed-MoE source contract used by Qwen3.8-Flash-Next |
| `mistral4_nvfp4` | source-native Mistral Small 4 NVFP4/BF16 contract |
| `muse_glimmer` | Muse-Glimmer attention, normalization, head and auxiliary records |
| `qwen3_next` | Qwen3-Next MoE source contract |
| `lfm2_moe` | LFM2-MoE source contract |
| `olmoe` | OLMoE source contract |

## Install and inspect

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\compiler

.\.venv\Scripts\python.exe -m compiler inspect-source `
  --source C:\path\to\snapshot `
  --tensor-groups `
  --adapter hybrid_delta
```

Inspection validates SafeTensors headers, index agreement, shard paths,
ranges, dtypes and adapter geometry without writing an artifact.

## Compile and validate

```powershell
.\.venv\Scripts\python.exe -m compiler compile `
  --source C:\path\to\snapshot `
  --output C:\path\to\artifact.candidate `
  --source-id Qwen/Qwen3.8-27B `
  --source-revision <immutable-commit> `
  --adapter hybrid_delta `
  --quant-profile fp4-e2m1-ue8m0-block32-v1

.\.venv\Scripts\python.exe -m compiler validate `
  C:\path\to\artifact.candidate
```

Normal publication uses
`ops/windows/Publish-HuggingFaceExpertPack.ps1`. It resolves a pinned Xet
snapshot, writes a sibling candidate below `MODEL_ROOT`, validates it and
promotes it transactionally. Partial compilation resumes only when source
identity and options match exactly. Source-shard reclamation is destructive and
is not part of the supported publication path.

Unknown tensors, fields, paths, record versions, operation ABIs, geometries or
hashes fail closed. FP4 publication additionally requires source reconstruction
quality and real provider execution; parsing an artifact is not an FP4 gate.

See [Expert Pack v1](../docs/expert-pack-v1.md) and
[Runtime contract](../docs/expert-runtime.md).
