# Getting started

Status: supported setup path, 2026-09-27.

## Prerequisites

Execution host:

- Windows 11;
- one RTX 3090 with a supported NVIDIA driver;
- CUDA Toolkit and Visual Studio 2022 C++ tools;
- CMake 3.25 or newer, Python 3 and PowerShell;
- OpenSSH access from the control machine;
- enough NVMe space for the source checkpoint, candidate and published
  artifact.

Control machine:

- POSIX shell, Git, Python 3 and SSH;
- Pi only when the coding harness is required.

The Windows build is authoritative for CUDA execution.

## Configure

```bash
cp .env.example .env
```

Set:

```dotenv
QUANTUM_LLM_REMOTE=user@gpu-host
QUANTUM_LLM_REMOTE_ROOT=D:/quantum-llm
MODEL_ROOT=D:/quantum-llm/work/models
EXPERT_API_KEY=<random-secret>
```

`MODEL_ROOT` is the only host-dependent model-storage root. Keep
`MODEL_HOST=127.0.0.1` when clients connect through the maintained SSH
tunnel.

Inspect resolved non-secret settings:

```bash
./ops/model.sh config qwen
```

## Build

```bash
./ops/model.sh sync
./ops/run-on-windows-host.sh Invoke-Inventory.ps1
./ops/run-on-windows-host.sh Invoke-BuildExpertRuntime.ps1 -Configuration Release
./ops/model.sh install
```

The Release/CUDA workflow configures the MSVC preset, builds with
`--clean-first`, runs CTest and runs the canonical Python contract suites.
`cmake --fresh` does not remove stale objects and is insufficient after a
header or ABI change.

## Download

Hugging Face downloads use the repository's Xet workflow with an immutable
revision and expected inventory:

```bash
./ops/run-on-windows-host.sh Start-HuggingFaceModelDownload.ps1 \
  -ModelId Qwen/Qwen3.8-27B \
  -Revision <40-hex-commit> \
  -ExpectedDownloadBytes <bytes> \
  -ExpectedTensorBytes <bytes> \
  -ExpectedShards <count>

./ops/run-on-windows-host.sh Get-HuggingFaceModelDownload.ps1
```

The download worker derives shard paths from the selected SafeTensors index.
Do not replace this path with ad-hoc downloads or disable Xet.

## Compile and publish

```bash
./ops/run-on-windows-host.sh Publish-HuggingFaceExpertPack.ps1 \
  -ModelId Qwen/Qwen3.8-27B \
  -Revision <immutable-commit> \
  -StableName qwen3.8-27b-fp4 \
  -Adapter hybrid_delta \
  -QuantProfile fp4-e2m1-ue8m0-block32-v1
```

The compiler implements these source adapters:

| Adapter | Source contract |
|---|---|
| `hybrid_delta` | hybrid DeltaNet/full-attention checkpoints, including current Qwen and Ornith sources |
| `qwen4_exp` | Hyper, PLE, QSA, recurrent and routed-MoE checkpoints |
| `mistral4_nvfp4` | source-native NVFP4/BF16 Mistral Small 4 |
| `muse_glimmer` | Muse-Glimmer text program plus preserved auxiliary tensors |
| `qwen3_next` | Qwen3-Next MoE source contract |
| `lfm2_moe` | LFM2-MoE source contract |
| `olmoe` | OLMoE source contract |

Adapters affect compilation only. Published artifacts use the common service
and VM lifecycle.

Publication follows candidate -> validate -> promote. Never point an alias or
service task at a partial, candidate or absent directory.

## Run

```bash
./ops/model.sh start qwen
./ops/model.sh status
./ops/model.sh chat qwen
./ops/pi.sh qwen
./ops/model.sh stop all
```

A successful `hi` verifies artifact, service, tokenizer and generation
wiring. It does not qualify coding behavior, long context or throughput.

Continue with [Deployment](deployment.md), [Operations](operations.md) and
[Pi CLI](pi-cli.md).
