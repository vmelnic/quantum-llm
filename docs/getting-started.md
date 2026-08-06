# Getting started

## Tested platform

- Windows 10/11 x64;
- NVIDIA RTX 3090 (SM86), 24 GB VRAM;
- approximately 64 GB system RAM;
- Visual Studio 2022 Build Tools with C++ workload;
- CMake 3.25 or newer;
- CUDA Toolkit 12.1;
- Python 3.10 or newer;
- enough disk for the source checkpoint, 81.9 GB container, and safety margin.

The portable core/tests build on Linux without CUDA. The complete Qwen runtime
is currently Windows/CUDA only.

For an existing built runtime and prepared DeepSeek/Qwen artifacts, use the
[end-to-end deployment guide](deployment.md) instead: it covers `.env`, remote
synchronization, model switching, interactive chat, and API use through the
single `ops/model.sh` control command.

## Clone and Python environment

```powershell
git clone <repository-url> quantum-llm
cd quantum-llm
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements\server.txt
```

The Qwen server does not execute model repository code. DeepSeek-V4-Flash lacks
a Transformers chat template, so its server executes only the pinned
checkpoint's official `encoding/encoding_dsv4.py` for prompt formatting; model
layers still run exclusively through the native runtime.

## Build and test

```powershell
.\ops\windows\Invoke-Inventory.ps1
.\ops\windows\Invoke-BuildExpertRuntime.ps1 -Configuration Release
```

The build runs C++ contract/runtime/CUDA tests and Python compiler/server tests.
The expected Qwen runner is:

```text
out/build/windows-msvc-release/runtime/Release/expert-qwen3-next-runner.exe
```

Portable development:

```bash
cmake --preset portable-release
cmake --build --preset portable-release
ctest --preset portable-release
python -m unittest -v tests.compiler.test_expert_pack tests.server.test_expert_server
```

## Obtain the model

Accept the upstream model license and authenticate Hugging Face first. Either
use the regular CLI or the resumable scheduled-task wrapper:

```powershell
hf download Qwen/Qwen3-Next-80B-A3B-Instruct --max-workers 4

# Optional managed download:
.\ops\windows\Start-P6ModelDownload.ps1
.\ops\windows\Get-P6ModelDownload.ps1
```

The tested revision is:

```text
9c7f2fbe84465e40164a94cc16cd30b6999b0cc7
```

## Compile Expert Pack

Review disk headroom first:

```powershell
.\ops\windows\Invoke-P6Preflight.ps1 -Snapshot C:\path\to\snapshot
```

The safe path preserves every source shard:

```powershell
.\ops\windows\Invoke-P6Conversion.ps1 `
  -Snapshot C:\path\to\snapshot `
  -Output C:\models\qwen3-next-80b-expert-pack-int8
```

Source reclamation is intentionally not shown as the default. It deletes
consumed SafeTensors shards and is only for a reviewed low-disk recovery flow.
The generic compiler/validator commands are documented in `compiler/README.md`.

Validate an existing pack independently:

```powershell
.\ops\windows\Invoke-ExpertPack.ps1 -Action Validate `
  -Path C:\models\qwen3-next-80b-expert-pack-int8
```

## Run in the foreground

```powershell
.\ops\windows\Start-P6ExpertServer.ps1 `
  -Container C:\models\qwen3-next-80b-expert-pack-int8 `
  -Python .\.venv\Scripts\python.exe `
  -MaximumContext 4096 `
  -WorkerCapacity 4 `
  -PlacementProfile balanced `
  -WorkerKvCacheMiB 2048 `
  -WorkerKvPageTokens 256 `
  -BuildId (git rev-parse --short HEAD)
```

Readiness can take tens of seconds because the server validates and uploads the
dense pack:

```powershell
curl.exe http://127.0.0.1:8080/health
curl.exe http://127.0.0.1:8080/ready
curl.exe http://127.0.0.1:8080/model-info
```

Use [the API guide](openai-api.md) for SDK and streaming examples.

`balanced` is the qualified reference-host default. Select `latency` for aggressive
single-stream warming or `capacity` to avoid speculative VRAM churn; see the
[runtime placement contract](expert-runtime.md#placement-policy) before
changing it. The choice is visible in `/model-info.worker_placement`.

## Install the pilot service

```powershell
.\ops\windows\Install-ExpertServerTask.ps1 -Start `
  -Container C:\models\qwen3-next-80b-expert-pack-int8 `
  -Python .\.venv\Scripts\python.exe `
  -BuildId (git rev-parse --short HEAD)

.\ops\windows\Invoke-P6ServiceSmoke.ps1 `
  -ExpectedBuildId (git rev-parse --short HEAD)
```

Stop the task and release model VRAM while keeping registration:

```powershell
.\ops\windows\Stop-ExpertServer.ps1
```

Remove registration and stop the entire process tree:

```powershell
.\ops\windows\Uninstall-ExpertServerTask.ps1
```

## Context configuration

`MaximumContext=4096` is the only certified value. The API rejects prompt plus
output beyond this capacity. Raising it no longer preallocates maximum KV for
every slot, but it still requires a matching aggregate page budget and
long-context qualification. Do not advertise the model's 262K architectural
maximum as an operational limit. See
[Production readiness](production-readiness.md).

KV capacity is an aggregate request budget. With the tested geometry, one
256-token page is 6 MiB. The 4096 × four-slot profile can reserve at most 64
pages (384 MiB); the larger 2048 MiB setting leaves room to qualify higher
contexts without changing the service interface.
