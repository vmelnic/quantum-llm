# Production readiness

Status: current pilot verdict as of 2026-08-11. VM completion work is tracked
in [the canonical handoff](moe-vm-next.md).

## Verdict

Qwen3-Next 80B FP4, DeepSeek-V4-Flash and LFM2-8B-A1B FP4 can be started,
queried and stopped through the same OpenAI-compatible service, task and binary
on the reference Windows/RTX 3090 host. This is a research/pilot runtime, not a
production service or a finished composable MoE VM.

Historical controlled Qwen tests reached 39.6–45.6 tok/s on FP4 resident
routes and 5.3–16.9 tok/s on their novel-route suite. Those numbers do not
describe every service configuration: the final common-path `hi` probes were
0.99 tok/s Qwen, 0.37 tok/s DeepSeek and 1.80 tok/s LFM after the first token.
That gate proved lifecycle correctness, not a performance improvement.
The configured 65K context ceiling is not long-context-qualified. No tested
artifact meets the 30 tok/s user-facing objective for representative chat.

## Readiness matrix

| Area | Status | Current evidence or gap |
|---|---|---|
| immutable model artifacts | pilot-ready | strict pack/index/record hashes and completion markers |
| deterministic greedy execution | pilot-ready for tested paths | native correctness gates and real API generation |
| Qwen lifecycle | functional | install, start, identity/limit verification, chat and full stop tested |
| DeepSeek lifecycle | functional | compact bundle, persistent worker and API generation tested |
| LFM lifecycle | functional | FP4 Expert Pack, common worker selection and API generation tested |
| common MoE VM control plane | functional | one launcher/task/binary and artifact-declared schema-2 program |
| composable VM executor | missing | common binary still selects one complete worker provider; operation providers do not yet own the shared model loop |
| unified physical container | missing | Expert Pack v1 and DeepSeek bundle/compact formats still coexist behind adapters |
| Unicode streaming | fixed | incomplete byte-fallback sequences are held until a stable prefix exists |
| bounded memory/admission | pilot-ready | explicit RAM/VRAM/KV/queue budgets and fail-closed credits |
| configured context | 65,536 | enabled hard ceiling, not a quality/performance qualification |
| qualified context | 4,096 | larger staged gates remain undone |
| Qwen hot repeated route | capable | 39.6–45.6 tok/s FP4 (S1b) post-first-token on a resident route |
| Qwen representative chat | not ready | historical changing-route suite measured 5.3–16.9 tok/s FP4; final VM smoke was slower and was not a suite rerun |
| DeepSeek throughput | not ready | indexed-supply follow-on reached 6.54–6.69 tok/s only on settled identical prompts; matched novel/retained suites remained below 1 tok/s; final cold/common-path `hi` measured 0.37 tok/s after first token |
| conversation KV reuse | implemented | protocol v5 retained sessions prefill only the per-turn delta; TTFT flat 3–9 s |
| route-aware warm state | partial | cache/prefetch exists; no general chat-hot guarantee |
| API compatibility | partial | documented greedy text subset; no tools/multimodal/sampling/logprobs |
| authentication | partial | optional shared bearer key only |
| TLS and edge controls | missing | external proxy/firewall required |
| supervision | partial | Task Scheduler pilot, not a hardened service |
| observability | partial | JSONL/metrics exist; Qwen request-level placement telemetry is incomplete |
| reliability | missing | no 24-hour soak, chaos campaign, HA or failover |
| GPU CI/security review | missing | manual SM86 validation; no independent audit/fuzz campaign |
| external Memory Expert | research, not qualified | KV-attach milestones 1–2 validated in the extracted standalone `memory-expert` project; serving integration here is not started |

## Release blockers

1. Meet a declared user-facing workload SLO, not only hot runner gates.
   Resident routes meet it; novel-route chat remains first-touch
   storage-bound, so the SLO is not met for representative chat.
2. Improve novel-route placement beyond the landed priority tiers, protected
   RAM/transient VRAM, census warm-up and prompt top-six protection. Wider
   protection and raw-frequency promotion already regressed and must not be
   repeated unchanged.
3. Qualify context sizes sequentially up to the advertised ceiling.
4. Add a hardened supervisor, external logging/metrics/alerts and rollback gate.
5. Add TLS/auth/rate limiting at an edge and keep the built-in server private.
6. Pass soak, cancellation storm, I/O corruption, disk-full, OOM and worker
   restart tests.
7. Replace whole-worker provider selection with the compiled per-operation VM
   executor and prove a fourth compatible FP4 MoE without common-path changes.

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
