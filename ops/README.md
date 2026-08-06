# Operations automation

`ops/` contains deployment automation only. Product code lives in `core/`,
`compiler/`, and `runtime/`; public contracts live in `docs/` and `schemas/`.
The complete operator walkthrough is [Install, configure, and use](../docs/deployment.md).

## POSIX control-host wrappers

Copy `.env.example` to `.env` and set the remote host and immutable model
artifact paths. The lifecycle wrapper consumes that file:

```bash
./ops/model.sh config             # resolved, non-secret configuration
./ops/model.sh start              # CHAT_MODEL from .env; sync + replace + wait
./ops/model.sh status
./ops/model.sh chat
./ops/model.sh stop

./ops/model.sh start qwen         # explicit one-command model switch
./ops/model.sh start deepseek
./ops/model.sh stop all           # release all model processes and VRAM
./ops/model.sh sync               # sync without changing the running service
```

Starting a model first stops both known scheduled tasks because they share one
GPU and one API port. `MODEL_MAX_CONTEXT` and `MODEL_MAX_OUTPUT_TOKENS` become
the advertised API limits and are checked against `/model-info` before `start`
returns. `CHAT_MAX_TOKENS` is an optional independent per-turn ceiling; the
reference configuration gives chat the full model output allowance.

The sync includes only Git-visible, non-ignored files. It therefore excludes
`.git`, models, work, logs, artifacts, and build output. It does not delete
remote files.

Interactive streaming chat can own its SSH tunnel and clean it up on exit:

```bash
cp .env.example .env
# Set CHAT_SSH in .env, then:
./ops/chat.sh
```

## Windows scripts

| Script | Purpose |
|---|---|
| `Invoke-Inventory.ps1` | hardware/toolchain/model-cache inventory |
| `Invoke-BuildExpertRuntime.ps1` | configure, build, C++/CUDA/Python tests |
| `Start-P6ModelDownload.ps1` | resumable Hugging Face download task |
| `Get-P6ModelDownload.ps1` | independently verify download progress/completion |
| `Start-HuggingFaceModelDownload.ps1` | generic pinned, disk-checked Xet download task |
| `Get-HuggingFaceModelDownload.ps1` | progress and index/shard completion verification |
| `Start-DeepSeekV4FlashDownload.ps1` | pinned DeepSeek-V4-Flash Xet download profile |
| `Invoke-SourceInventory.ps1` | read-only validation and dtype/byte inventory for a cached checkpoint |
| `Invoke-DeepSeekExpertSlice.ps1` | full real-expert decode/reference/INT8 qualification |
| `Invoke-DeepSeekCudaAdmission.ps1` | compact H2D + SM86 admission hash gate |
| `Invoke-DeepSeekDenseAdmission.ps1` | block-scaled FP8 dense matrix → SM86 GEMV gate |
| `Invoke-DeepSeekCompactDenseScreen.ps1` | read-only compact dense representation screen across real projection geometries |
| `Invoke-DeepSeekDenseResidency.ps1` | atomically load all 236 main-model FP8 matrices |
| `Invoke-DeepSeekSharedAdmission.ps1` | FP8 shared expert → SM86 cache/compute gate |
| `Invoke-DeepSeekSharedResidency.ps1` | atomically pin all 43 shared experts at startup |
| `Invoke-DeepSeekHcaSlice.ps1` | real F32 HCA pre/Sinkhorn/post CUDA correctness gate |
| `Invoke-DeepSeekTypedResidency.ps1` | stream all 834 BF16/F32/I64 tensors into typed model state |
| `Invoke-DeepSeekModelResidency.ps1` | publish model/request state and qualify catalog-backed asynchronous layer resume |
| `Start-DeepSeekCompactPack.ps1` | start resumable durable DeepSeek compact-pack publication outside repository scratch |
| `Get-DeepSeekCompactPack.ps1` | report authenticated layer-shard progress and final publication state |
| `Invoke-DeepSeekCsaSlice.ps1` | validate real ratio-4/128 CSA compressor decode state on CUDA |
| `Invoke-P6Preflight.ps1` | exact disk/conversion feasibility for Qwen3-Next 80B |
| `Invoke-ExpertPack.ps1` | generic compile/validate wrapper |
| `Invoke-P6Conversion.ps1` | pinned Qwen3-Next conversion profile |
| `Invoke-P6Gate.ps1` | hot single/aggregate correctness and throughput gates |
| `Start-ExpertServer.ps1` | generic foreground HTTP/worker launcher |
| `Start-P6ExpertServer.ps1` | tested Qwen profile |
| `Install-ExpertServerTask.ps1` | install/replace pilot scheduled task |
| `Stop-ExpertServer.ps1` | stop task and full descendant process tree |
| `Uninstall-ExpertServerTask.ps1` | stop and remove task registration |
| `Invoke-P6ServiceSmoke.ps1` | identity, API, streaming, batching, cancellation smoke |
| `Get-ExpertServerStatus.ps1` | task state, readiness, identity and configured-limit verification |

Generated logs/artifacts/work directories are ignored by Git. Model deletion is
not part of normal automation. Source shard reclamation exists only behind an
explicit literal confirmation and is documented as destructive.

See [Getting started](../docs/getting-started.md) and
[Operations](../docs/operations.md) for complete procedures.
