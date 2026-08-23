# Pi CLI

Status: maintained local coding-agent integration, 2026-08-23.

Pi is the preferred harness because it can target the local OpenAI-compatible
service without Claude Code's fixed Claude-oriented prompt/caching behavior.
The repository config selects Qwen by default, exposes Qwen and DeepSeek, uses
`xhigh` thinking and enables bounded compaction.

## Repository settings

`.pi/settings.json` contains:

- provider `quantum-llm` and model `qwen3.8-27b-fp4`;
- Qwen and DeepSeek enabled at `xhigh`;
- compaction with 8,192 reserved output tokens and 32,768 recent tokens;
- a request timeout matching long local prefills;
- automatic retries disabled so failures and overload are visible.

The provider secret/base URL belong in Pi's user-local model registry, not in
Git. Configure a single OpenAI-compatible provider named `quantum-llm` that
points at the local tunnel URL and uses `EXPERT_API_KEY`.

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

Multiple Pi processes may keep sessions, but Qwen currently has one serialized
hot provider slot. Expect queueing, not simultaneous decode. Actual parked
F16 KV bytes govern capacity; several near-262K sessions cannot be inferred
safe from the configured maximum.

## Qualification boundary

`hi`, tool calling and image understanding prove wiring only. A production
coding qualification must cover a representative repository, long tool loop,
cancellation/resume, compaction and useful-output quality. Current readiness
and blockers are listed in [Production readiness](production-readiness.md).
