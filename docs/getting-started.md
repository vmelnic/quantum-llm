# Getting started

Status: current Windows/CUDA setup and Qwen3.8 FP4 artifact path as of
2026-08-19.
The resumable implementation handoff is [MoE VM current state and remaining
work](moe-vm-next.md).

## Tested platform

- Windows 10/11 x64;
- NVIDIA RTX 3090 (SM86), 24 GB VRAM;
- approximately 64 GB system RAM;
- Visual Studio 2022 Build Tools with C++ workload;
- CMake 3.25 or newer;
- CUDA Toolkit 12.1;
- Python 3.10 or newer;
- enough disk for the source checkpoint, FP4 container, and safety margin.

The portable core/tests build on Linux without CUDA. Complete Qwen3.8 and
DeepSeek execution is currently Windows/CUDA only.

For an existing built runtime and prepared Qwen/DeepSeek artifacts, use the
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

The common server does not execute arbitrary model repository code.
DeepSeek-V4-Flash lacks
a Transformers chat template, so its server executes only the pinned
checkpoint's official `encoding/encoding_dsv4.py` for prompt formatting; model
layers still run exclusively through the native runtime.

## Build and test

```powershell
.\ops\windows\Invoke-Inventory.ps1
.\ops\windows\Invoke-BuildExpertRuntime.ps1 -Configuration Release
```

The build runs C++ contract/runtime/CUDA tests and Python compiler/server tests.
The serving runner for every supported artifact is:

```text
out/build/windows-msvc-release/runtime/Release/expert-moe-vm-runner.exe
```

Portable development:

```bash
cmake --preset portable-release
cmake --build --preset portable-release
ctest --preset portable-release
python -m unittest -v tests.compiler.test_deepseek_quant \
  tests.compiler.test_expert_pack \
  tests.server.test_expert_server \
  tests.server.test_deepseek_route_oracle
```

## Obtain the model

Accept the upstream model license and authenticate Hugging Face first.
Downloads go only through the resumable scheduled-task wrapper (official `hf`
CLI with Xet). Supply values verified from the pinned upstream revision; do not
run ad-hoc `curl`, parallel downloaders or disable Xet:

```powershell
.\ops\windows\Start-HuggingFaceModelDownload.ps1 `
  -ModelId "Qwen/Qwen3.8-27B" `
  -Revision "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" `
  -ExpectedDownloadBytes 55615875685 `
  -ExpectedTensorBytes 55562855904 `
  -ExpectedShards 18

.\ops\windows\Get-HuggingFaceModelDownload.ps1
```

The tested revision is:

```text
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

## Compile Expert Pack

Compile through the generic wrapper and preserve every source shard:

```powershell
.\ops\windows\Invoke-ExpertPack.ps1 -Action Compile `
  -Path C:\path\to\snapshot `
  -Output (Join-Path $env:MODEL_ROOT "qwen3.8-27b-fp4") `
  -SourceId "Qwen/Qwen3.8-27B" `
  -SourceRevision "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" `
  -Adapter qwen3_5 `
  -QuantProfile fp4-e2m1-ue8m0-block32-v1 `
  -MaxExpertPackBytes 24GB `
  -Resume
```

Source reclamation is intentionally not shown as the default. It deletes
consumed SafeTensors shards and is only for a reviewed low-disk recovery flow.
The generic compiler/validator commands are documented in `compiler/README.md`.

Validate an existing pack independently:

```powershell
.\ops\windows\Invoke-ExpertPack.ps1 -Action Validate `
  -Path (Join-Path $env:MODEL_ROOT "qwen3.8-27b-fp4")
```

The supported Qwen path always selects FP4-E2M1/UE8M0 block-32 matrices. This
Qwen3.8 checkpoint is dense and has no routed experts. The artifact is
validated before transactional publication below `MODEL_ROOT`.

The equivalent generic compiler invocation is:

```powershell
.\.venv\Scripts\python.exe -m compiler compile `
  --source C:\path\to\snapshot `
  --output (Join-Path $env:MODEL_ROOT "qwen3.8-27b-fp4") `
  --source-id Qwen/Qwen3.8-27B `
  --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 `
  --adapter qwen3_5 `
  --quant-profile fp4-e2m1-ue8m0-block32-v1 `
  --max-expert-pack-bytes 25769803776

.\ops\windows\Invoke-ExpertPack.ps1 -Action Validate `
  -Path (Join-Path $env:MODEL_ROOT "qwen3.8-27b-fp4")
```

Serve it with `./ops/model.sh start qwen`; the `qwen` selector is FP4-only.

## Run in the foreground

```powershell
.\ops\windows\Start-ExpertServer.ps1 `
  -Container (Join-Path $env:MODEL_ROOT "qwen3.8-27b-fp4") `
  -Runner .\out\build\windows-msvc-release\runtime\Release\expert-moe-vm-runner.exe `
  -ModelId qwen3.8-27b-fp4 `
  -Python .\.venv\Scripts\python.exe `
  -MaximumContext 262144 `
  -MaximumNewTokens 8192 `
  -WorkerCapacity 1 `
  -PlacementProfile balanced `
  -WorkerKvCacheMiB 5120 `
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
  -Container (Join-Path $env:MODEL_ROOT "qwen3.8-27b-fp4") `
  -Runner .\out\build\windows-msvc-release\runtime\Release\expert-moe-vm-runner.exe `
  -ModelId qwen3.8-27b-fp4 `
  -Python .\.venv\Scripts\python.exe `
  -BuildId (git rev-parse --short HEAD)

.\ops\windows\Get-ExpertServerStatus.ps1 `
  -WaitSeconds 600 `
  -ExpectedModel qwen3.8-27b-fp4
```

Prefer `./ops/model.sh start qwen` from the control host for normal operation;
it supplies the common cache/queue contract and verifies all configured limits.

Stop the task and release model VRAM while keeping registration:

```powershell
.\ops\windows\Stop-ExpertServer.ps1
```

Remove registration and stop the entire process tree:

```powershell
.\ops\windows\Uninstall-ExpertServerTask.ps1
```

## Context configuration

The reference lifecycle advertises 262,144 context tokens and 8,192 output
tokens. A real 262,016-token prompt plus 128 generated tokens completed, so the
capacity path is qualified. That run required approximately 90 minutes to its
first generated token and decoded at approximately 3.24 tok/s afterward; it is
not a maximum-context performance or semantic-quality qualification. See
[Production readiness](production-readiness.md).

KV capacity is an aggregate request budget. The Qwen3.8 reference profile uses
5,120 MiB and one worker slot. Physical pages are committed on demand, so short
requests do not allocate the maximum context.
