# Production readiness

Status: functional research/pilot runtime, not production-ready for the full
262K coding objective, 2026-08-23.

## Verdict

The native Qwen and DeepSeek services work through one artifact-driven VM,
HTTP lifecycle and Pi integration. Short-context Qwen chat, reasoning, tools,
images, exact-F16 progressive allocation and retained-session recovery have
passed bounded gates. DeepSeek exact expert paging and generation also work.

The product is not ready for unrestricted real-project use at maximum context:

- Qwen exact-F16 near-maximum prefill takes 2,183.815 seconds for 262,001
  prompt tokens plus one output;
- no saturated 262K exact-F16 decode qualification exists;
- the historical compact-KV maximum decoded at only about 3.24 tok/s;
- Claude Code's large automatic prompt failed latency and factual behavior;
- execution is serialized through one hot Qwen provider slot;
- long-context failure injection and broad quality qualification remain open.

## Ready components

| Area | State |
|---|---|
| Qwen FP4 artifact | transactionally published, source/hash/format validated |
| DeepSeek compact bundle | authenticated and consumable by the common VM |
| common lifecycle | install/start/status/chat/stop with artifact aliases |
| OpenAI API | Completions, Chat Completions, Responses, streaming and usage |
| Anthropic adapter | Messages/count_tokens and streaming wire compatibility |
| Pi integration | Qwen/DeepSeek selection, xhigh default, thinking on/off |
| Qwen text/tools/images | bounded functional gates passed |
| exact F16 KV | progressive 256-token pages, append growth and zero-delta reuse |
| session retention | park/restore, transactional resume and cancel rewind |
| introspection | cached nonblocking health/model/metrics path |
| DeepSeek routing | exact top-k, no dropped experts, tier-attributed telemetry |

## Not ready

| Area | Blocker |
|---|---|
| Qwen 262K interactive use | prefill latency and exact-F16 saturated decode |
| unrestricted coding harness | representative long tool-loop quality/reliability not qualified |
| true parallel agents | one serialized hot slot; parking gives continuity, not concurrency |
| multi-GPU | no per-device allocator/provider/P2P implementation |
| public network service | no integrated TLS, rate limiting or durable edge |
| automated recovery | long-prefix worker restart/KV exhaustion gates incomplete |
| broad model universality | artifact/provider model is generic, but only current capability sets are executable |

## Release invariants

- `MODEL_ROOT` is the only host-dependent model-storage root.
- A service artifact must contain a valid manifest and completion marker and be
  validated before `.env` or a scheduled task points at it.
- Qwen serving uses FP4 matrix payloads through the SM86 FP4 provider; INT8 is
  not an acceptable substitute.
- The deployed exact target-KV policy is F16. FP8/Q4 results cannot be promoted
  as equivalent.
- DeepSeek preserves exact top-k and stable aggregation.
- `ready=true` means the service can admit requests, not that caches are warm
  or an SLO is met.
- A release gate ends with `model.sh stop all` and verified process/GPU cleanup.

## Evidence required for production promotion

1. clean local and Windows Release/CUDA tests;
2. bounded official numerical and behavior comparisons for text, sampling,
   tools and images;
3. real Qwen and DeepSeek `model.sh chat` smokes through the common runner;
4. representative Pi coding histories with delta-only reuse;
5. populated 262K exact-F16 Qwen prefill/decode meeting the accepted latency
   and throughput target;
6. large-prefix cancellation, timeout, restart and memory-pressure recovery;
7. authenticated edge, supervision, observability and rollback.

See [Benchmarks](benchmarks.md) for current evidence and
[Roadmap](roadmap.md) for the work order.
