# Getting started

Status: supported setup path, 2026-09-02.

The control host is POSIX; the execution host is the self-contained Windows
RTX 3090 machine. The repository may live at any path. `.env` supplies the
remote project root and the only model-storage root.

## Prerequisites

Execution host:

- Windows 11, RTX 3090 driver and CUDA Toolkit 12.x;
- Visual Studio 2022 C++ tools and CMake 3.25 or newer;
- Python 3 and PowerShell permission to create the scheduled task;
- NVMe capacity for the pinned checkpoint, conversion candidate and published
  artifact;
- OpenSSH access from the control host.

Control host: Bash, Git, Python 3 and SSH. Pi is optional and required only for
the coding-harness path.

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

`MODEL_ROOT` is the only host-dependent model root. Do not add per-model
absolute paths to source, scripts or scheduled tasks. Keep
`MODEL_HOST=127.0.0.1` when clients use the SSH tunnel.

Inspect resolved non-secret settings:

```bash
./ops/model.sh config qwen
```

## Prepare and build

```bash
./ops/model.sh sync
./ops/run-on-windows-host.sh Invoke-Inventory.ps1
./ops/run-on-windows-host.sh Invoke-BuildExpertRuntime.ps1 -Configuration Release
./ops/model.sh install
```

The Windows build is authoritative for CUDA. It configures the maintained MSVC
preset, builds with `--clean-first`, runs CTest and runs the canonical Python
contract suites. `cmake --fresh` alone is insufficient after an internal ABI
change because it does not guarantee stale objects were removed.

## Download with Hugging Face Xet

Pin an immutable revision and the expected inventory:

```bash
./ops/run-on-windows-host.sh Start-HuggingFaceModelDownload.ps1 \
  -ModelId Qwen/Qwen3.8-27B \
  -Revision <40-hex-commit> \
  -ExpectedDownloadBytes <bytes> \
  -ExpectedTensorBytes <bytes> \
  -ExpectedShards <count>

./ops/run-on-windows-host.sh Get-HuggingFaceModelDownload.ps1
```

For nonstandard repositories, declare the actual config/index filenames. The
worker derives shard paths from the selected index. Do not use `curl`, an
ad-hoc downloader, invented scheduled tasks or `HF_HUB_DISABLE_XET`.

## Compile and publish

Example Qwen publication:

```bash
./ops/run-on-windows-host.sh Publish-HuggingFaceExpertPack.ps1 \
  -ModelId Qwen/Qwen3.8-27B \
  -Revision <immutable-commit> \
  -StableName qwen3.8-27b-fp4 \
  -Adapter hybrid_delta \
  -QuantProfile fp4-e2m1-ue8m0-block32-v1
```

Active source adapters are:

- `hybrid_delta`: Qwen3.8-27B and Ornith;
- `qwen4_exp`: Qwen3.8-Flash-Next;
- `mistral4_nvfp4`: Mistral Small 4 native NVFP4;
- `muse_glimmer`: Muse-Glimmer.

Adapters affect compilation only. Every published artifact uses the common VM
and service lifecycle. DeepSeek uses the descriptor/oracle/compact-bundle
workflow in [DeepSeek compact pack v1](deepseek-compact-pack-v1.md).

Publication is candidate -> validate -> promote. Never point `.env`, an alias
or a task at `.partial`, `.candidate` or a missing artifact.

## Run

```bash
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/model.sh stop all
```

The lifecycle aliases are `qwen`, `qwen-flash`, `mistral`, `muse`, `ornith`
and `deepseek`. A real `hi` validates wiring only. For a minimal Pi check:

```bash
./ops/model.sh start qwen
./ops/pi.sh qwen --no-context-files --no-tools --no-session -p hi
./ops/model.sh stop all
```

Continue with [Deployment](deployment.md), [Operations](operations.md) and
[Pi CLI](pi-cli.md).
