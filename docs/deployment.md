# Install, configure, and use

This is the operator path for one POSIX control host and one Windows/CUDA GPU
host. It manages both supported deployments through one local `.env` and one
command, while keeping the large model artifacts on the Windows host.

## What the lifecycle wrapper manages

`ops/model.sh` synchronizes Git-visible source, stops the competing model,
installs or replaces the selected Windows scheduled task, starts it, waits for
readiness, and verifies model identity plus runtime limits.

It intentionally does not download checkpoints, compile model packs, build
CUDA binaries, delete artifacts, or copy `.env` to the remote host. Those are
separate installation and data-management operations.

## 1. Install the runtime

The control host needs Git, Bash, OpenSSH, and Python 3. The Windows GPU host
needs the prerequisites in [Getting started](getting-started.md), SSH access,
and the repository bootstrap/build:

```powershell
git clone <repository-url> C:\quantum-llm
cd C:\quantum-llm
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements\server.txt
.\ops\windows\Invoke-BuildExpertRuntime.ps1 -Configuration Release
```

Prepare at least one immutable model artifact:

- DeepSeek-V4-Flash worker bundle, as described in
  [DeepSeek serving](deepseek-serving.md);
- Qwen3-Next 80B Expert Pack, as described in
  [Getting started](getting-started.md#compile-expert-pack).

The lifecycle wrapper validates the selected artifact during service startup.
It never removes the original checkpoint or either prepared model.

## 2. Configure the control host

From the local repository:

```bash
cp .env.example .env
```

Edit `.env`. It is ignored by Git and may contain host-specific paths:

```dotenv
QUANTUM_LLM_REMOTE=user@gpu-host
QUANTUM_LLM_REMOTE_ROOT=C:/quantum-llm

CHAT_MODEL=deepseek-v4-flash
MODEL_MAX_CONTEXT=65536
MODEL_MAX_OUTPUT_TOKENS=8192
MODEL_PORT=8080
MODEL_SYNC_ON_START=1
MODEL_READY_TIMEOUT=600
MODEL_GENERATION_TIMEOUT_SECONDS=600

MODEL_DEEPSEEK_BUNDLE=C:/models/deepseek-v4-flash/worker-bundle-v3
MODEL_QWEN_CONTAINER=C:/models/qwen3-next-80b-expert-pack-int8

CHAT_SSH=user@gpu-host
CHAT_BASE_URL=http://127.0.0.1:8080
CHAT_MAX_TOKENS=8192
CHAT_LOCAL_PORT=18080
CHAT_READY_TIMEOUT=30
CHAT_PYTHON=python3
CHAT_SHOW_STATS=1
EXPERT_API_KEY=
```

The model selectors accepted by `CHAT_MODEL` are:

| Value | Deployment |
|---|---|
| `deepseek-v4-flash` | DeepSeek-V4-Flash compact worker bundle |
| `qwen3-next-80b-a3b-expert-pack-int8` | Qwen3-Next 80B Expert Pack |

Short aliases `deepseek` and `qwen` are accepted as command arguments.

`MODEL_MAX_CONTEXT` covers the entire tokenized request: instructions, chat
history, current input, and requested output. `MODEL_MAX_OUTPUT_TOKENS` is the
server ceiling for one response. `CHAT_MAX_TOKENS` is what the terminal client
requests and cannot exceed the server ceiling.

The reference 65,536/8,192 limits are operational configuration, not a claim
that long-context correctness or latency has been qualified. A request can
also end at `MODEL_GENERATION_TIMEOUT_SECONDS` before reaching its output
ceiling. Increase that deadline deliberately if a slow model must be allowed
to generate for longer.

Inspect the resolved non-secret control configuration before changing remote
state:

```bash
./ops/model.sh config
./ops/model.sh config qwen
```

## 3. Start and verify a model

Start the model selected by `CHAT_MODEL`:

```bash
./ops/model.sh start
```

The command returns only after `/ready` succeeds and `/model-info` matches the
selected model, context, output ceiling, and generation deadline. Inspect it
again at any time:

```bash
./ops/model.sh status
```

A healthy DeepSeek deployment reports fields like:

```json
{
  "ready": true,
  "model": "deepseek-v4-flash",
  "max_context": 65536,
  "maximum_new_tokens": 8192,
  "worker_capacity": 1,
  "active_requests": 0
}
```

`worker_capacity=1` means one admitted DeepSeek generation at a time;
`active_requests=0` means the service was idle when status was sampled.

## 4. Chat interactively

```bash
./ops/model.sh chat
```

The client creates and owns an SSH tunnel, streams text, and closes the tunnel
on `quit`, `exit`, or Ctrl+C. Interactive commands are:

| Command | Effect |
|---|---|
| `/info` | model/build, limits, caches, placement, capacity, and protocol |
| `/stats` | token counts, TTFT, elapsed time, and tokens/s for the last turn |
| `/clear` | discard local conversation history |
| `quit` or `exit` | close chat and its SSH tunnel |

The client resends the full retained history on every turn. Use `/clear`
before the history plus requested output reaches `MODEL_MAX_CONTEXT`.

## 5. Use the OpenAI-compatible API

For tools other than the terminal chat, keep a tunnel open:

```bash
ssh -N -L 18080:127.0.0.1:8080 user@gpu-host
```

Then use `http://127.0.0.1:18080/v1` as the OpenAI base URL:

```bash
curl http://127.0.0.1:18080/v1/models

curl http://127.0.0.1:18080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "input": "Reply briefly: what is an MoE model?",
    "max_output_tokens": 128,
    "temperature": 0,
    "stream": true
  }'
```

See [OpenAI-compatible API](openai-api.md) for SDK examples, supported fields,
stream formats, errors, and explicit capability gaps.

## 6. Switch, restart, stop, and sync

Only one model uses the GPU/API port at a time. Starting either model safely
stops both known scheduled tasks first:

```bash
./ops/model.sh start qwen
./ops/model.sh start deepseek
```

Other lifecycle operations:

```bash
./ops/model.sh restart             # selected CHAT_MODEL
./ops/model.sh stop                # selected CHAT_MODEL
./ops/model.sh stop all            # release all model processes and VRAM
./ops/model.sh sync                # source sync without a restart
./ops/model.sh status              # tasks plus live API identity
```

`stop` keeps task registration for inspection but kills the full server/worker
process tree and releases VRAM. `start` replaces the task definition from the
current `.env` values.

## Troubleshooting

- `model.sh: set QUANTUM_LLM_REMOTE...`: configure the SSH target in `.env`.
- `Container missing` or `runtime bundle is missing`: correct the immutable
  Qwen/DeepSeek artifact path; synchronization does not copy ignored models.
- readiness timeout: run `./ops/model.sh status`, then inspect the remote JSONL
  service log and Task Scheduler state.
- identity/limit mismatch: the start command fails closed; synchronize and
  replace the task again rather than using the stale service.
- `context_capacity_exhausted`: prompt plus output needs more KV credits than
  are currently available; shorten/clear history or lower concurrency.
- generation timeout: raise `MODEL_GENERATION_TIMEOUT_SECONDS` only after
  considering that one slow DeepSeek request occupies its sole worker slot.
- local port already in use: choose another `CHAT_LOCAL_PORT`; the remote
  service continues to use `MODEL_PORT`.

The built-in HTTP server binds to Windows loopback by default. Keep it behind
SSH or a trusted TLS/authenticated reverse proxy; do not expose it directly to
the public internet.
