# Pi CLI

Status: maintained coding-harness integration, 2026-09-12.

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

Replace the alias with `qwen-f16`, `qwen-flash`, `mistral`, `muse`, `ornith`,
`ornith-k1` or `deepseek`. `ops/pi.sh` validates the alias/key, reads Pi's configured provider
URL, then checks the running model ID and explicit KV codec through
`/model-info` before launching Pi. It defaults to `--thinking xhigh` and
forwards remaining arguments. The operational `qwen` alias and experimental
`ornith-k1` alias require K1; `qwen-f16` and `ornith` are their explicit
exact-F16 reference aliases.

Minimal wiring gate:

```bash
./ops/pi.sh qwen \
  --no-context-files --no-tools --no-session -p hi
```

## Current evidence

All six advertised model IDs and `ornith-k1` pass the no-context/no-tools/
no-session `hi` wiring gate. Qwen and Muse are responsive in that gate; Flash,
Mistral and especially DeepSeek are slow. Exact timings live only in
[Benchmarks](benchmarks.md). Wiring does not qualify tools, coding quality,
long context or concurrency.

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

Representative coding qualification remains separate. Qwen passed the bounded
Todo fixture but with poor wall time; Flash and Mistral made zero edits; Ornith
K1+fit reached only 2/3 tests. Muse and DeepSeek are not coding-qualified.
