# Artifact compiler

The compiler reads local SafeTensors checkpoints with bounded mappings and
writes authenticated execution artifacts. It does not load a Transformers
model or materialize the complete checkpoint in RAM.

## Source adapters

Adapters are strict source contracts. They translate upstream configuration
and tensor names into semantic roles, geometry, operations and encodings.

| Adapter | Contract |
|---|---|
| `hybrid_delta` | hybrid recurrent/full-attention dense or MoE sources |
| `qwen4_exp` | Hyper, PLE, QSA, recurrent and routed-MoE sources |
| `mistral4_nvfp4` | source-native NVFP4/BF16 Mistral Small 4 |
| `muse_glimmer` | Muse-Glimmer text program and explicit auxiliary tensors |
| `qwen3_next` | Qwen3-Next MoE |
| `lfm2_moe` | LFM2-MoE |
| `olmoe` | OLMoE |

An adapter must account for every source tensor. Unknown tensors, inconsistent
geometry and unsupported dtypes fail before compilation.

## Commands

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\compiler

.\.venv\Scripts\python.exe -m compiler inspect-source `
  --source C:\path\to\snapshot --tensor-groups --adapter hybrid_delta

.\.venv\Scripts\python.exe -m compiler compile `
  --source C:\path\to\snapshot `
  --output C:\path\to\artifact.candidate `
  --source-id Qwen/Qwen3.8-27B `
  --source-revision <immutable-commit> `
  --adapter hybrid_delta `
  --quant-profile fp4-e2m1-ue8m0-block32-v1

.\.venv\Scripts\python.exe -m compiler validate `
  C:\path\to\artifact.candidate

.\.venv\Scripts\python.exe -m compiler qualify-fp4-container `
  --source C:\path\to\snapshot `
  --container C:\path\to\artifact.candidate
```

Additional commands clone a valid artifact while changing only authenticated
metadata:

- `refresh-model-program`;
- `refresh-sampling-profiles`;
- `upgrade-dense-mtp-program`.

Supported stored-weight profiles are symmetric row INT8, block-32 FP4 variants
and source-native block-16 NVFP4. A capability/role policy may select MXFP6 for
eligible dense tensors. The resulting manifest records the exact format and
resolved tensor set.

Normal deployment uses `Publish-HuggingFaceExpertPack.ps1` to compile a
sibling candidate, validate it and promote it transactionally. The
`--reclaim-source-shards` option deletes committed source shards and is not
part of that supported publication path.

See [Expert Pack v1](../docs/expert-pack-v1.md) and
[Runtime contract](../docs/expert-runtime.md).
