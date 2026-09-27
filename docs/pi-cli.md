# Pi CLI

Status: maintained coding harness, 2026-09-27.

## Configuration

Repository `.pi/settings.json` selects:

- provider `quantum-llm`;
- model `qwen3.8-27b-fp4`;
- thinking level `xhigh`;
- automatic retries disabled;
- compaction enabled with 32,768 recent tokens and an 8,192-token reserve.

The enabled catalog contains six model IDs:

- `qwen3.8-27b-fp4`;
- `qwen3.8-27b-abliterated-fp4`;
- `qwen3.8-flash-next-fp4`;
- `mistral-small-4-119b-nvfp4`;
- `muse-glimmer-30b-fp4`;
- `ornith-1.5-35b-a3b-fp4`.

The provider base URL and secret live in Pi's user-local model registry. Git
contains no credential.

## Launch

```bash
./ops/model.sh start qwen
./ops/pi.sh qwen
```

`pi.sh` resolves the repository and `.env` from its own path, then:

1. resolves the alias through `ops/model-aliases.tsv`;
2. verifies that Pi knows the advertised model;
3. queries `/model-info`;
4. verifies the running model, output limit and explicit KV codec;
5. verifies exact-decode ABI 2, draft depth, vocabulary and Q8 MTP state for a
   Q4H artifact that declares MTP;
6. launches Pi with `xhigh` and forwards remaining arguments.

Minimal wiring check:

```bash
./ops/pi.sh qwen \
  --no-context-files --no-tools --no-extensions --no-skills \
  --no-prompt-templates --no-session --thinking off -p hi
```

Pi still supplies its own base instructions. The provider's populated prompt
count is the authoritative context measurement.

## Thinking and sampling

Artifacts own sampling defaults. The operational Qwen profiles are:

```text
thinking:     temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.5
non-thinking: temperature=0.7, top_p=0.8,  top_k=20, presence_penalty=0.5
thinking output ceiling: 32,768
```

The Qwen context remains 262,144 and the normal service output ceiling remains
262,143. The server applies the 32,768 ceiling only to thinking requests.
Supported request fields may override individual sampling defaults but cannot
raise that ceiling.

Ornith has its own artifact profiles and no Qwen-specific thinking ceiling.
Changing thinking or sampling changes the workload and must be reported with a
benchmark result.

## Context and sessions

The service allocates KV for populated positions. Pi's context window,
compaction reserve and per-turn output limit are separate values. Several Pi
processes may queue and compatible sessions may retain exact prefixes, but the
service has one hot execution slot.

Qwen persistence can restore a matching provider state after a service restart.
Pi replaying its transcript does not guarantee a provider cache hit; telemetry
must report `resumed=true` or a disk restore.

## Autonomous 256K benchmark

The benchmark in `ops/python/pi256k_benchmark.py` runs a gradual coding-agent
dialogue on the Windows host. It records every completed model call at its
actual populated prompt size, writes results atomically and stops only after a
real prompt reaches 255,000 through 255,999 positions.

It can run the Quantum LLM QPack/Q4H/MTP-4 path and a pinned llama.cpp GGUF
comparison sequentially. Their formats and placement are different; the tool
records them rather than treating them as equivalent.

```bash
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 \
  -Action preflight -Backend quantum
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 \
  -Action launch -Backend quantum
./ops/run-on-windows-host.sh Invoke-Pi256kBenchmark.ps1 \
  -Action status -Backend quantum
```

The latest Quantum run is incomplete. It recorded 25 completed model calls,
reached 58,664 populated prompt tokens and then stalled with no provider
completion. No 256K result exists. Exact measurements belong in
[Benchmarks](benchmarks.md).
