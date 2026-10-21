# Pi CLI

Status: maintained coding harness, 2026-10-21.

## Configuration

Pi configuration lives in the user's home:

- `~/.pi/agent/models.json`: provider endpoints, credentials and model metadata;
- `~/.pi/agent/settings.json`: defaults, enabled models, thinking and compaction.

The project `.pi` directory has been removed. It is intentionally absent;
repository documentation and launch commands use the home configuration.

The home default remains `llama-3090/swift-1.5-iq3-xxs`, with xhigh
reasoning. Automatic retries are disabled. Compaction keeps 32768 recent
tokens and reserves 26214 tokens. Select Strata explicitly for the active
Strata service; starting a server does not change Pi's saved default.

| Local 3090 backend | Pi provider | Model selection | Serving operations |
|---|---|---|---|
| Artifact VM / QPack | `quantum-llm` | artifact model ID | `ops/model.sh`, `ops/pi.sh` |
| llama.cpp | `llama-3090` | `swift-1.5-iq3-xxs`, `swift-heretic`, `kat-coder-v2.5-iq4-xs`, `gpt-oss-20b-mxfp4` | [llama.cpp CLI](llama-cli.md) |
| Strata | `strata-3090` | `swift-flash-next-iq3-s` | [Strata CLI](strata-cli.md) |

The backends have separate serving processes. Use one active model on the
3090 and stop it through its own backend before switching to another backend.

## Strata through Pi

The independent Strata service is registered in the user's
`~/.pi/agent/models.json` as provider `strata-3090`, base URL
`http://10.10.88.4:8082/v1`, model `swift-flash-next-iq3-s`. User
`settings.json` enables `strata-3090/swift-flash-next-iq3-s:xhigh` for model
selection. It advertises 262144 context positions, 32768 maximum output tokens,
text input and native xhigh reasoning. Authentication uses the existing local
service key, stored only in private user configuration. Configuration backups
are under ignored `out/strata-20261021/pi-config-backups/`.

```bash
./ops/strata.sh start swift
pi --provider strata-3090 --model swift-flash-next-iq3-s --thinking xhigh
```

In an existing Pi client, open `/model` and select
`strata-3090/swift-flash-next-iq3-s`. The model registry reloads when the
picker opens. The enabled-model entry also makes it available in model cycling.

The [actual Pi measurement](benchmarks.md#actual-pi-request-with-embedded-code-2026-10-21)
records a larger embedded-code prompt, engine generation rate, prefill,
reasoning/visible latency, memory and test limits. Completion throughput
includes thinking; Pi's reported zero reasoning-token count is not a valid
breakdown. This was a single coding-generation response with tools disabled,
not a coding correctness or multi-turn tool benchmark.

## Artifact VM through Pi

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

## Artifact VM thinking and sampling

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

Pi's context window, compaction reserve and per-turn output limit are
separate values. Strata advertises 262144 context positions and 32768 maximum
output tokens; the larger-prompt Pi test does not qualify a populated262K
conversation. KV placement and session reuse depend on the selected backend.

The artifact VM allocates KV for populated positions. Several Pi processes
may queue and compatible sessions may retain exact prefixes, but the VM
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
