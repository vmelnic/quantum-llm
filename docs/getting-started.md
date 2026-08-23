# Getting started

Status: supported setup path, 2026-08-23.

This project uses a POSIX control host and a self-contained Windows/CUDA
execution host. The reference execution host is `3090box`; paths below are
examples and all model storage is resolved through `MODEL_ROOT`.

## Prerequisites

The Windows host needs:

- Windows 11, an NVIDIA driver exposing the RTX 3090, and CUDA Toolkit 12.x;
- Visual Studio 2022 C++ build tools and CMake 3.25 or newer;
- Python 3 and PowerShell with permission to create a scheduled task;
- enough NVMe space for the immutable Hugging Face snapshot, conversion
  candidate and published artifact;
- OpenSSH access from the control host.

The control host needs Bash, Python 3, Git and SSH. The repository itself may
live anywhere; `.env` supplies the remote project root and model root.

## Configure

```bash
cp .env.example .env
```

Set at least:

```dotenv
QUANTUM_LLM_REMOTE=user@gpu-host
QUANTUM_LLM_REMOTE_ROOT=D:/quantum-llm
MODEL_ROOT=D:/quantum-llm/work/models
EXPERT_API_KEY=<random-secret>
```

`MODEL_ROOT` is the only model-storage root. Do not put per-model absolute
paths in scripts or source. Keep `MODEL_HOST=127.0.0.1` when accessing the
service through the managed SSH tunnel.

Inspect the resolved non-secret configuration before changing the host:

```bash
./ops/model.sh config qwen
```

## Prepare the host

```bash
./ops/model.sh sync
./ops/run-on-windows-host.sh Invoke-Inventory.ps1
./ops/run-on-windows-host.sh Invoke-BuildExpertRuntime.ps1 -Configuration Release
./ops/model.sh install
```

The build script configures the `windows-msvc-release` preset, builds the
native runtime, runs CTest and the supported Python contract suites. Server
dependencies are installed from pinned `requirements/server.txt` into the
repository-owned virtual environment.

## Download a checkpoint

Hugging Face downloads are pinned to an immutable commit and use Xet through
the repository scripts. Supply the exact repository byte and shard inventory:

```bash
./ops/run-on-windows-host.sh Start-HuggingFaceModelDownload.ps1 \
  -ModelId Qwen/Qwen3.8-27B \
  -Revision <40-hex-commit> \
  -ExpectedDownloadBytes <bytes> \
  -ExpectedTensorBytes <bytes> \
  -ExpectedShards <count>

./ops/run-on-windows-host.sh Get-HuggingFaceModelDownload.ps1
```

Do not use ad-hoc `curl`, parallel downloaders or `HF_HUB_DISABLE_XET`.

## Compile and publish Qwen FP4

The official checkpoint is compiled directly into a candidate below
`MODEL_ROOT`, validated, then atomically promoted:

```bash
./ops/run-on-windows-host.sh Publish-HuggingFaceExpertPack.ps1 \
  -ModelId Qwen/Qwen3.8-27B \
  -Revision <immutable-commit> \
  -StableName qwen3.8-27b-fp4 \
  -Adapter qwen3_5 \
  -QuantProfile fp4-e2m1-ue8m0-block32-v1
```

`qwen3_5` is the compiler adapter identifier inherited from the upstream
checkpoint ABI; it does not create or select a Qwen3.5 deployment. The
published service alias is only Qwen3.8. See [Expert Pack v1](expert-pack-v1.md).

DeepSeek uses its authenticated compact-bundle workflow documented in
[DeepSeek compact pack v1](deepseek-compact-pack-v1.md).

## First service run

```bash
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/model.sh stop all
```

The real smoke prompt is `hi`. A successful smoke validates lifecycle and the
common path; it is not a performance or quality qualification. Repeat the same
commands with `deepseek` when changing common runtime/service code.

Continue with [Deployment](deployment.md) and [Operations](operations.md).
