# Pi CLI

Status: maintained local coding-agent integration, 2026-08-26.

Pi is the preferred harness because it can target the local OpenAI-compatible
service without Claude Code's fixed Claude-oriented prompt/caching behavior.
The repository config selects Qwen by default, exposes Qwen, Muse, Ornith and
DeepSeek, uses `xhigh` thinking and enables bounded compaction.

## Repository settings

`.pi/settings.json` contains:

- provider `quantum-llm` and model `qwen3.8-27b-fp4`;
- Qwen, Muse, Ornith and DeepSeek enabled at `xhigh`;
- compaction with 8,192 tokens of answer headroom and 32,768 recent tokens;
- a request timeout matching long local prefills;
- automatic retries disabled so failures and overload are visible.

The provider secret/base URL belong in Pi's user-local model registry, not in
Git. Configure a single OpenAI-compatible provider named `quantum-llm` that
points at the service URL and resolves `EXPERT_API_KEY`. The maintained local
catalog contains only the four supported models; repository settings remain
portable and contain no secret.

## Start and use

Start the service and keep the SSH tunnel/API reachable, then run:

```bash
./ops/model.sh start qwen
./ops/pi.sh qwen
```

For DeepSeek:

```bash
./ops/model.sh start deepseek
./ops/pi.sh deepseek
```

For Ornith:

```bash
./ops/model.sh start ornith
./ops/pi.sh ornith
```

For Muse text inference:

```bash
./ops/model.sh start muse
./ops/pi.sh muse
```

Muse currently advertises text only and an artifact-declared 131,072-token
maximum. The lifecycle clamps the global 262K Qwen setting to that contract;
declaring 131K does not prove it was populated or timed.

`ops/pi.sh` validates the model alias and secret, then launches Pi with the
common provider, selected advertised model and `--thinking xhigh`. Arguments
after the model are passed through to Pi.

## Thinking control

The default is `xhigh`. Pi's per-session thinking control maps to the request's
`reasoning_effort`/`enable_thinking` fields, so thinking can be enabled or
disabled without editing the artifact or server. Throughput comparisons must
report both visible and reasoning tokens and must not confuse thinking cost
with MTP policy.

## Context behavior

Pi compaction is intentional. A 262K model limit is a ceiling, while an agent
should normally keep a smaller active working set and compact old turns before
it becomes saturated. The runtime retains exact populated KV and feeds only a
suffix when the prefix is unchanged. Editing/truncating an old prefix requires
replay from the common prefix checkpoint.

The Qwen, Ornith and DeepSeek Pi entries advertise the 262,143-token output
ceiling; Muse advertises its artifact limit of 131,071. Pi subtracts its
estimated active context and a 4,096-token safety margin before each request;
the server then clamps the request again using the exact tokenized prompt.
Neither ceiling preallocates KV. The separate 8,192-token compaction reserve
controls when Pi compacts history and is not a model output limit.

Multiple Pi processes may keep sessions, but the current service has one
serialized hot provider slot. Expect queueing, not simultaneous decode. Actual
parked F16 KV bytes govern capacity; several near-262K sessions cannot be
inferred safe from the configured maximum.

The real Ornith gate uses normal Pi project context and tools. On 2026-08-25,
a 7,874-token project request completed without the former post-generation
parser failure. A separate 7,906-token request emitted a structured `read`
call, Pi executed it, and the retained follow-up returned the requested file
value. Qwen passed the same `read` loop. DeepSeek passed `model.sh chat`, but
its full Pi gate was stopped during the already-known slow harness prefill and
is not qualified here.

Streaming responses use one artifact-declared parser instance from the first
delta through final tool extraction. Pi never parses model-native tool syntax,
and the server no longer reparses the completed text and attempts to reconcile
it with content that has already been delivered.

## Qualification boundary

`hi`, tool calling and image understanding prove wiring only. A production
coding qualification must cover a representative repository, long tool loop,
cancellation/resume, compaction and useful-output quality. Current readiness
and blockers are listed in [Production readiness](production-readiness.md).
