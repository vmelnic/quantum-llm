# Operations automation

`ops/` is the supported control and Windows-host surface. Generated models,
builds, logs and caches do not belong in this directory.

## Control-host entry points

```bash
./ops/model.sh config qwen
./ops/model.sh sync
./ops/model.sh install
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/pi.sh qwen
./ops/model.sh clear-cache qwen
./ops/model.sh stop all
```

`model.sh` reads `.env`, resolves `model-aliases.tsv` and manages the one
common scheduled task. It validates the artifact contract before start and
verifies the running model, limits and KV policy after readiness.

`pi.sh` verifies the running service against Pi's user-local model registry,
including the Q4H/MTP contract, then launches Pi with `xhigh` by default.

## Windows workflows

| Area | Maintained entry points |
|---|---|
| bootstrap/build | `Invoke-Bootstrap.ps1`, `Invoke-Inventory.ps1`, `Invoke-BuildExpertRuntime.ps1` |
| download | `Start-HuggingFaceModelDownload.ps1`, `Get-HuggingFaceModelDownload.ps1` |
| artifact | `Invoke-SourceInventory.ps1`, `Invoke-ExpertPack.ps1`, `Publish-HuggingFaceExpertPack.ps1`, `Promote-ModelArtifact.ps1` |
| service | `Get-ModelArtifactContract.ps1`, `Install-ExpertServerTask.ps1`, `Get-ExpertServerStatus.ps1`, `Stop-ExpertServer.ps1` |
| gates | populated-context, source-quality, behavior, Qwen speed and Pi 256K workflows |

Hugging Face downloads use the maintained Xet worker. Artifact publication is
candidate -> validate -> promote. Native deployment uses the clean-first
Windows Release/CUDA build. Every service gate ends with process, port and GPU
cleanup.

## Pi 256K benchmark

`Invoke-Pi256kBenchmark.ps1` exposes preparation, download, publication,
build, preflight, launch and status actions. It creates one backend-specific
scheduled task and writes durable per-call evidence below
`out/benchmarks/pi256k`. Quantum and llama.cpp backends run sequentially.

See [Getting started](../docs/getting-started.md),
[Deployment](../docs/deployment.md), [Operations](../docs/operations.md) and
[Pi CLI](../docs/pi-cli.md).
