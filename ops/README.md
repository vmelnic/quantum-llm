# Operations automation

`ops/` is the supported control-host and Windows-host surface. Runtime code is
in `compiler/`, `core/` and `runtime/`; generated models, builds, logs and
caches do not belong here.

## Control-host commands

```bash
./ops/model.sh config qwen
./ops/model.sh install
./ops/model.sh sync
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/pi.sh qwen
./ops/model.sh stop all
```

`model.sh` reads `.env`, resolves `ops/model-aliases.tsv` and controls one
scheduled task and one common VM. `start` optionally syncs Git-visible files,
validates the selected artifact contract, clamps limits to artifact geometry,
stops the previous service, starts the selected artifact and waits for matching
readiness. Sync never copies or deletes ignored model/build/cache state.

`chat.sh` opens the optional SSH tunnel and reports prompt/output timing.
`pi.sh` selects the same provider/model registry, verifies the running model
and explicit KV codec through `/model-info`, and defaults to `xhigh` thinking;
all additional Pi arguments pass through unchanged. `qwen`,
`qwen-abliterated` and `ornith-k1` select K1 while `qwen-f16` and `ornith`
preserve the exact-F16 reference paths.

## Windows workflows

- host and build: `Invoke-Bootstrap.ps1`, `Invoke-Inventory.ps1`,
  `Invoke-BuildExpertRuntime.ps1`, `Install-ServerEnvironment.ps1`;
- Hugging Face Xet: `Start-HuggingFaceModelDownload.ps1`,
  `Invoke-HuggingFaceDownloadWorker.ps1`,
  `Get-HuggingFaceModelDownload.ps1`;
- QPack publication: `Invoke-SourceInventory.ps1`, `Invoke-ExpertPack.ps1`,
  source-quality gates, `Publish-HuggingFaceExpertPack.ps1` and
  `Promote-ModelArtifact.ps1`;
- DeepSeek publication: source inspection/descriptors/oracles,
  `Invoke-DeepSeekCompactPack.ps1` and `Publish-DeepSeekWorkerBundle.ps1`;
- service: `Get-ModelArtifactContract.ps1`, `Start-ExpertServer.ps1`,
  `Install-ExpertServerTask.ps1`, `Get-ExpertServerStatus.ps1` and
  `Stop-ExpertServer.ps1`;
- network maintenance: `Set-WireGuardSplitTunnel.ps1` for the documented
  split-tunnel configuration on 3090box.

## Operational invariants

- model storage derives only from `MODEL_ROOT`;
- downloads use the two maintained Xet entry points, never ad-hoc transfer;
- publish candidate -> validate -> promote -> start -> real smoke -> stop;
- the Windows Release/CUDA build uses `--clean-first`; `cmake --fresh` alone
  does not prove objects were rebuilt after an internal ABI change;
- a release gate ends with process and GPU cleanup;
- no normal lifecycle command deletes a published model or Hugging Face cache.

See [Getting started](../docs/getting-started.md),
[Deployment](../docs/deployment.md), [Operations](../docs/operations.md) and
[Pi CLI](../docs/pi-cli.md).
