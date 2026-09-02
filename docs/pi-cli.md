# Pi CLI

Status: maintained coding-harness integration, 2026-09-02.

Pi is the preferred local harness because it can call the repository's
OpenAI-compatible endpoint without Claude Code's fixed Claude-oriented prompt.
It is still a real agent: even with context files and tools disabled, Pi may
send its own system instructions. A one-word user prompt is therefore not
necessarily a five-token model prefill.

## Configuration

Repository `.pi/settings.json` selects provider `quantum-llm`, Qwen by default,
all six active advertised IDs, `xhigh` thinking, disabled automatic retries and
a request timeout matching long local prefills. Compaction keeps 32,768 recent
tokens and reserves 8,192 tokens for the next answer/summary.

The base URL and secret live in Pi's user-local registry
`~/.pi/agent/models.json`; Git contains no credential. The local registry
should contain only these project models:

- `qwen3.8-27b-fp4`;
- `qwen3.8-flash-next-fp4`;
- `mistral-small-4-119b-nvfp4`;
- `muse-glimmer-30b-fp4`;
- `ornith-1.5-35b-a3b-fp4`;
- `deepseek-v4-flash`.

## Use

```bash
./ops/model.sh start qwen
./ops/pi.sh qwen
```

Replace the alias with `qwen-flash`, `mistral`, `muse`, `ornith` or
`deepseek`. `ops/pi.sh` validates the alias/key, selects the common provider and
model ID, defaults to `--thinking xhigh`, and forwards remaining arguments.

Minimal wiring gate:

```bash
./ops/pi.sh qwen \
  --no-context-files --no-tools --no-session -p hi
```

## Current minimal Pi evidence

On 2026-09-02 all six models returned valid text with default `xhigh` and no
context files, tools or session. Server telemetry is authoritative:

| Alias | Prefill | Generated | TTFT | Wall | Verdict |
|---|---:|---:|---:|---:|---|
| `qwen` | 5 | 33 | 0.625 s | 1.938 s | pass |
| `qwen-flash` | 477 | 67 | 41.829 s | 53.688 s | pass, slow |
| `mistral` | 439 | 67 | 54.797 s | 77.312 s | pass, slow |
| `muse` | 422 | 65 | 2.297 s | 4.297 s | pass |
| `ornith` | 439 | 34 | 3.703 s | 5.390 s | pass |
| `deepseek` | 504 | 256 | 81.985 s | 310.672 s | pass, impractical for trivial Pi turn |

This table proves Pi/API/template/generation wiring only. It does not qualify
tool use, coding quality, long context or concurrency.

## Thinking and sampling

The repository default is `xhigh`. Pi can append another supported
`--thinking` level for an explicit run, including `off`. The request path maps
thinking mode and reasoning effort into the artifact's official template.

Qwen-family default sampling comes from artifact metadata, not Pi:

```text
thinking:     temperature=1.0, top_p=0.95, top_k=20
non-thinking: temperature=0.7, top_p=0.8, top_k=20,
              presence_penalty=1.5
```

Supported explicit API/Pi parameters override individual defaults. Results
must report hidden reasoning separately; changing thinking is a quality and
latency experiment, not a runtime correctness setting.

## Context and concurrency

Pi's compaction reserve is not a model output or context limit. Qwen, Qwen
Flash, Mistral, Ornith and DeepSeek currently advertise up to 262,144 positions;
Muse advertises 131,072. The server clamps output after exact tokenization and
allocates KV only for populated positions.

Several Pi processes may queue and retain compatible sessions, but the current
service has one hot execution slot. They do not decode concurrently. DeepSeek
does not retain exact sessions, so its history must be replayed.

Representative coding qualification remains separate. Dense Qwen completed
the bounded Todo fixture at effective `medium`, but took 503.240 seconds; Flash
and Mistral made zero edits in their recorded gates. Ornith passed a bounded
read loop. Muse and DeepSeek are not qualified as production coding agents.
