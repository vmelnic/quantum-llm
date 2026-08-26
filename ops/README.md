# Operations automation

`ops/` contains the supported control-host and Windows-host workflows. Product
code is in `compiler/`, `core/` and `runtime/`; generated work, logs and model
artifacts do not belong here.

## Control-host commands

```bash
./ops/model.sh config qwen
./ops/model.sh install
./ops/model.sh sync
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/model.sh stop all

./ops/model.sh start deepseek
./ops/pi.sh deepseek
```

`model.sh` reads `.env`, resolves aliases from `model-aliases.tsv` and invokes
one common scheduled task/native VM. It contains no family runner or per-model
absolute storage root. Sync copies Git-visible files only and never deletes
remote `work/`, builds or `${MODEL_ROOT}`.

`chat.sh` owns an optional SSH tunnel and streams the local API with timings.
`pi.sh` selects the common Pi provider, exact advertised model and `xhigh`
thinking. See [Pi CLI](../docs/pi-cli.md).

## Windows workflows

The maintained PowerShell surface is grouped by purpose:

- host/build: `Invoke-Bootstrap.ps1`, `Invoke-Inventory.ps1`,
  `Invoke-BuildExpertRuntime.ps1`, `Install-ServerEnvironment.ps1`;
- Hugging Face Xet: `Start-HuggingFaceModelDownload.ps1`, its worker and
  `Get-HuggingFaceModelDownload.ps1`;
- Hugging Face QPack artifacts: `Invoke-SourceInventory.ps1`,
  `Invoke-ExpertPack.ps1`,
  `Invoke-Fp4SourceQualityGate.ps1`,
  `Publish-HuggingFaceExpertPack.ps1`, `Promote-ModelArtifact.ps1`;
- DeepSeek artifact: `Start-DeepSeekV4FlashDownload.ps1`, descriptor/oracle
  exporters, `Invoke-DeepSeekCompactPack.ps1` and
  `Publish-DeepSeekWorkerBundle.ps1`;
- service: `Get-ModelArtifactContract.ps1`, `Start-ExpertServer.ps1`,
  `Install-ExpertServerTask.ps1`, `Get-ExpertServerStatus.ps1`,
  `Stop-ExpertServer.ps1` and uninstall;
- reference behavior: the pinned Hugging Face reference environment and
  bounded behavior gates.

Former P6/Qwen3-Next deployment, Neural CPU, exact-tier and background
compact-pack wrappers were removed. Their durable negative results live in
[Research decisions](../docs/research-decisions.md), not in an ambiguous
second operator path.

## Safety rules

- downloads use only the pinned Xet scripts;
- host model paths derive only from `MODEL_ROOT`;
- artifacts are candidates until validated and transactionally promoted;
- start checks the pinned server environment before replacing a model;
- release gates test Qwen, Muse, Ornith and DeepSeek through the same public
  chat path;
- stop and verify process/GPU cleanup after every gate;
- no normal script deletes a published model or Hugging Face cache.

See [Getting started](../docs/getting-started.md),
[Deployment](../docs/deployment.md) and [Operations](../docs/operations.md).
