# Production readiness

## Verdict

Both Qwen3-Next 80B and DeepSeek-V4-Flash can be started, queried and stopped
through the same OpenAI-compatible service on the reference Windows/RTX 3090
host. This is a research/pilot runtime, not a production service.

Qwen is functionally usable but arbitrary multi-turn chat is currently about
0.7–1.4 tok/s after the first token. DeepSeek is slower. The configured 65K
context ceiling is not long-context-qualified. Neither backend meets the
30 tok/s user-facing objective for representative chat.

## Readiness matrix

| Area | Status | Current evidence or gap |
|---|---|---|
| immutable model artifacts | pilot-ready | strict pack/index/record hashes and completion markers |
| deterministic greedy execution | pilot-ready for tested paths | native correctness gates and real API generation |
| Qwen lifecycle | functional | install, start, identity/limit verification, chat and full stop tested |
| DeepSeek lifecycle | functional | compact bundle, persistent worker and API generation tested |
| Unicode streaming | fixed | incomplete byte-fallback sequences are held until a stable prefix exists |
| bounded memory/admission | pilot-ready | explicit RAM/VRAM/KV/queue budgets and fail-closed credits |
| configured context | 65,536 | enabled hard ceiling, not a quality/performance qualification |
| qualified context | 4,096 | larger staged gates remain undone |
| Qwen hot repeated route | capable | 30.07 tok/s post-first-token in the latest identical repeat |
| Qwen representative chat | not ready | changing routes/history measured about 0.7–1.4 tok/s |
| DeepSeek throughput | not ready | roughly 0.3–0.6 tok/s on the reference host |
| conversation KV reuse | missing | unchanged history is re-prefilled every turn |
| route-aware warm state | partial | cache/prefetch exists; no general chat-hot guarantee |
| API compatibility | partial | documented greedy text subset; no tools/multimodal/sampling/logprobs |
| authentication | partial | optional shared bearer key only |
| TLS and edge controls | missing | external proxy/firewall required |
| supervision | partial | Task Scheduler pilot, not a hardened service |
| observability | partial | JSONL/metrics exist; Qwen request-level placement telemetry is incomplete |
| reliability | missing | no 24-hour soak, chaos campaign, HA or failover |
| GPU CI/security review | missing | manual SM86 validation; no independent audit/fuzz campaign |

## Release blockers

1. Meet a declared user-facing workload SLO, not only hot runner gates.
2. Implement conversation prefix/KV reuse and workload-aware expert placement.
3. Qualify context sizes sequentially up to the advertised ceiling.
4. Add a hardened supervisor, external logging/metrics/alerts and rollback gate.
5. Add TLS/auth/rate limiting at an edge and keep the built-in server private.
6. Pass soak, cancellation storm, I/O corruption, disk-full, OOM and worker
   restart tests.

## Controlled pilot checklist

- pin commit, model revision, pack hashes, Python requirements, CUDA and driver;
- run `./ops/model.sh install`, then start the selected model;
- verify `/model-info` build, model, limits and pack identity;
- run one real tokenizer/chat request, not only a token-ID or readiness probe;
- record cold and warm TTFT, post-first-token and end-to-end rates;
- bind loopback or deploy an authenticated TLS proxy and firewall;
- retain the source checkpoint or an independently validated immutable pack;
- configure external log/metric retention and resource alerts;
- document capacity, rollback, maintenance and the full process-tree stop;
- obtain explicit acceptance of the known throughput/context limitations.

## Claims policy

- Checkpoint position metadata is not an operational context guarantee.
- `ready=true` means healthy and accepting work, not warmed or SLO-compliant.
- Hot runner throughput is not cold API or arbitrary-chat throughput.
- Aggregate throughput is not single-stream throughput.
- Draft/model rows are not useful output tokens unless verification accepts them.

Every published result must identify model, build, prompt/history distribution,
cold/warm state, single/aggregate scope, TTFT, decode rate and end-to-end rate.
