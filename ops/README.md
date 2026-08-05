# Operations automation

`ops/` contains deployment automation only. Product code lives in `core/`,
`compiler/`, and `runtime/`; public contracts live in `docs/` and `schemas/`.

## POSIX control-host wrappers

```bash
export QUANTUM_LLM_REMOTE=user@gpu-host
export QUANTUM_LLM_REMOTE_ROOT=C:/quantum-llm
# Run the POSIX sync wrapper, then invoke a script from ops/windows through
# the remote-script wrapper.
```

The sync includes only Git-visible, non-ignored files. It therefore excludes
`.git`, models, work, logs, artifacts, and build output. It does not delete
remote files.

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
| `Invoke-DeepSeekDenseResidency.ps1` | atomically load all 236 main-model FP8 matrices |
| `Invoke-DeepSeekSharedAdmission.ps1` | FP8 shared expert → SM86 cache/compute gate |
| `Invoke-DeepSeekSharedResidency.ps1` | atomically pin all 43 shared experts at startup |
| `Invoke-DeepSeekHcaSlice.ps1` | real F32 HCA pre/Sinkhorn/post CUDA correctness gate |
| `Invoke-DeepSeekTypedResidency.ps1` | stream all 834 BF16/F32/I64 tensors into typed model state |
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

Generated logs/artifacts/work directories are ignored by Git. Model deletion is
not part of normal automation. Source shard reclamation exists only behind an
explicit literal confirmation and is documented as destructive.

See [Getting started](../docs/getting-started.md) and
[Operations](../docs/operations.md) for complete procedures.
