# Qwen3.8-27B with Unsloth and Claude Code on WSL2

Status: runtime and private-network service validated on 2026-08-20; model
publication and end-to-end Claude Code acceptance are pending the pinned Xet
payload.

This is the production-oriented external runtime path for Qwen3.8-27B on the
RTX 3090 host. It does not replace, extend, or qualify the repository's MoE VM
runtime. It deliberately uses the upstream Unsloth/llama.cpp integration rather
than the repository's experimental dense FP4 server.

## Immutable contract

| Item | Value |
|---|---|
| Hugging Face repository | `unsloth/Qwen3.8-27B-GGUF` |
| Pinned revision | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |
| Main model | `Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| Main model bytes | `17,559,178,144` |
| MTP draft | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` |
| MTP bytes | `1,369,590,656` |
| Selected payload bytes | `18,928,768,800` |
| Stable artifact | `${MODEL_ROOT}/qwen3.8-27b-ud-q4-k-xl` |
| Context | `262144` tokens |
| Parallel slots | `1` |

`UD-Q4_K_XL` is an Unsloth Dynamic mixed GGUF quant. It is not NVFP4, FP4,
or a uniform INT4 tensor set. The separate MTP payload is GGUF `Q4_0`. Runtime
activations and intermediate compute are selected by llama.cpp; they must not
be reported as FP4 execution.

The official `Qwen/Qwen3.8-27B` BF16 snapshot remains intact in the Windows
Hugging Face cache. The service artifact is the upstream Unsloth GGUF because
the approximately 60.2 GB BF16 snapshot cannot reside on the 24 GB RTX 3090.
Unsloth recommends the GGUF path for pre-Blackwell GPUs and reserves its NVFP4
path for RTX 50, DGX Spark, B200, and B300 systems. See the
[Qwen3.8-27B model guide](https://unsloth.ai/models/qwen3.8-27b).

## Host layout

Windows owns the repository, model store, and Hugging Face cache:

```text
D:\quantum-llm
D:\quantum-llm\work\models
C:\Users\vladi\.cache\huggingface\hub
```

`MODEL_ROOT` remains the only model-store authority:

```dotenv
MODEL_ROOT=D:/quantum-llm/work/models
```

Ubuntu 24.04 is installed as WSL2 on the D: drive. Docker is not involved.
The current VM contract is tracked in
`ops/wsl/3090box.wslconfig`:

```ini
[wsl2]
memory=48GB
processors=12
swap=8GB
networkingMode=mirrored
guiApplications=false
vmIdleTimeout=2147483647

[experimental]
autoMemoryReclaim=gradual
```

`guiApplications=false` prevents the unused WSLg/Xwayland process from taking
roughly 0.5 GiB of VRAM. Microsoft explicitly notes that systemd services do
not keep a WSL instance alive; `vmIdleTimeout` is the supported VM idle policy.
See [advanced WSL settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)
and [systemd on WSL](https://learn.microsoft.com/en-us/windows/wsl/systemd).

After editing `.wslconfig`, apply it with:

```powershell
wsl.exe --shutdown
wsl.exe -d Ubuntu-24.04 -- true
```

## Xet download

All Hugging Face payloads use the repository's pinned, resumable Xet workflow.
The `-Files` contract selects exact repository paths and prevents accidental
download of every GGUF quant:

```bash
./ops/run-on-windows-host.sh Start-HuggingFaceModelDownload.ps1 \
  -ModelId unsloth/Qwen3.8-27B-GGUF \
  -Revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  -ExpectedDownloadBytes 18928768800 \
  -ExpectedTensorBytes 18928768800 \
  -ExpectedShards 2 \
  -Files Qwen3.8-27B-UD-Q4_K_XL.gguf,MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  -MaxWorkers 4

./ops/run-on-windows-host.sh Get-HuggingFaceModelDownload.ps1
```

A successful worker exit is not sufficient. Publication requires both exact
paths, exact aggregate bytes, no incomplete Xet blobs, and the pinned revision.
Copy into a staging child of `MODEL_ROOT`, verify the payload there, write the
source manifest, and rename the directory to
`qwen3.8-27b-ud-q4-k-xl` only after validation.

## Runtime installation

The installed upstream stack is:

| Component | Version |
|---|---|
| Unsloth source | `bcd0f6a785af92fdd11f9d44fe8a21ad930b01ac` |
| Unsloth package | `2026.8.18` |
| llama.cpp | build `10472`, commit `7a556b8f9` |
| CUDA toolkit | `13.0.88` |
| GPU target | `sm_86` |

The source checkout is `/home/vladi/unsloth-src`; the isolated runtime is
`/home/vladi/.unsloth/studio`. The installer was run without the training
PyTorch stack because this deployment executes GGUF through llama.cpp:

```bash
cd /home/vladi/unsloth-src
UNSLOTH_SKIP_AUTOSTART=1 UNSLOTH_NO_TORCH=1 \
  ./install.sh --local --no-torch
```

The GPU gate is the actual llama.cpp device probe, not NVIDIA-SMI alone:

```bash
/home/vladi/.unsloth/llama.cpp/llama-server --list-devices
```

It must report `CUDA0: NVIDIA GeForce RTX 3090`. A CPU-only fallback does not
satisfy this deployment.

## Service and network

The tracked service is `ops/systemd/unsloth-studio.service`. Install and start
it inside WSL:

```bash
install -m 0644 /mnt/d/quantum-llm/ops/systemd/unsloth-studio.service \
  ~/.config/systemd/user/unsloth-studio.service
systemctl --user daemon-reload
systemctl --user enable --now unsloth-studio.service
```

The service exposes one inference slot on port 8888. It starts offline with
respect to Hugging Face; the model is always loaded from the validated local
artifact. Health is available without a bearer token:

```bash
curl http://127.0.0.1:8888/api/health
```

WSL mirrored networking makes the endpoint available through the Windows host
address. The Hyper-V firewall rule `QuantumLLM-Unsloth-8888` permits inbound
TCP 8888 only from `10.10.88.2`, the current Claude host. The endpoint is plain
HTTP and must not be exposed to the public internet.

The generated bootstrap admin password was rotated immediately. Running
`unsloth studio reset-password` rotates it again and revokes existing API keys.

## Load the model and connect Claude Code

After transactional publication, resolve the stable artifact from
`MODEL_ROOT`, convert that root to its `/mnt/<drive>/...` WSL spelling, and run:

```bash
unsloth start claude \
  --model /mnt/d/quantum-llm/work/models/qwen3.8-27b-ud-q4-k-xl/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --max-seq-length 262144 \
  --gpu-memory-mode auto \
  --reasoning auto \
  --disable-tools \
  --no-launch
```

`--disable-tools` disables Unsloth's server-side web/code tools; Claude Code's
own tools remain in the request and are relayed to the model. Tool-call healing
and nudging retain their upstream defaults. `--reasoning auto` preserves the
model's hybrid-thinking path instead of applying the coding-agent default that
turns reasoning off. Qwen's template defaults reasoning effort to `xhigh`.

The command prints the official Claude environment and mints a local bearer
token. The project-local `.claude/settings.local.json` must use the Windows
address, not WSL loopback, and must pin every Claude model role to the served
model. It must also contain:

```json
{
  "env": {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "262144",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "85",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192"
  }
}
```

Disabling the attribution header preserves reusable prompt prefixes. The 85%
threshold leaves room for model output and tool results before the 262144-token
server boundary. Do not set `DISABLE_COMPACT`, `DISABLE_PROMPT_CACHING`, or the
old custom-server capability flags.

## Acceptance

Completion requires all of the following on the real stable artifact:

1. `Get-HuggingFaceModelDownload.ps1` reports the exact two-file payload as
   complete.
2. The stable directory is validated after transactional publication.
3. Unsloth reports the main GGUF and MTP companion loaded from the stable path.
4. NVIDIA-SMI shows llama.cpp on the RTX 3090 and no retired model server.
5. `/v1/models`, `/v1/messages`, streaming messages, and token counting work
   with the minted bearer token.
6. Claude Code answers `hi`, performs one harmless local tool call, and
   continues after the tool result.
7. A service stop releases llama.cpp and GPU memory; a clean restart reloads
   the same stable artifact and repeats `hi`.

These are functional production gates, not throughput benchmarks. Record TTFT
and decode rate from the real Claude request because interactive usability is
part of the acceptance criterion.

## Day-two operation and rollback

```bash
# Windows: start the WSL VM after a Windows reboot
wsl.exe -d Ubuntu-24.04 -- true

# WSL: inspect and control the API
systemctl --user status unsloth-studio.service
journalctl --user -u unsloth-studio.service -n 200 --no-pager
systemctl --user restart unsloth-studio.service
systemctl --user stop unsloth-studio.service

# Windows: release the complete WSL VM and GPU state
wsl.exe --shutdown
```

Rollback does not mutate the official Qwen snapshot or Hugging Face cache:

1. stop `unsloth-studio.service` and verify llama.cpp/GPU cleanup;
2. remove or disable `QuantumLLM-Unsloth-8888`;
3. remove `guiApplications=false` and `vmIdleTimeout` from `.wslconfig` if the
   previous WSL lifecycle is desired, then run `wsl.exe --shutdown`;
4. keep or delete the stable GGUF artifact independently of the source cache;
5. restore the previous ignored `.claude/settings.local.json` only if returning
   to the experimental repository server.

