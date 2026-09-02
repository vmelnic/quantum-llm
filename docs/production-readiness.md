# Production readiness

Status: functional research/pilot runtime; not production-ready for the full
maximum-context coding objective, 2026-09-02.

## Verdict

All six active artifacts start through one task/VM and return valid text through
both direct chat and the minimal Pi gate. Qwen text/reasoning/tools/image input,
exact progressive F16 KV and retained-session recovery have bounded evidence.
Muse and Ornith pass short text paths. Mistral's native NVFP4 and Qwen Flash's
extended program are callable. DeepSeek's self-contained exact paging bundle is
callable from NVMe/RAM/VRAM.

That is not yet production readiness:

- Qwen populated 262,001 exact-F16 prompt tokens in 2,183.815 s and generated
  only one token; saturated 262K decode at approximately 15 useful tok/s is
  unqualified;
- one hot provider slot serializes agents;
- Flash and Mistral pass wiring but fail current coding latency/quality gates;
- minimal DeepSeek Pi `hi` took 310.672 s and moved hundreds of GB through the
  cache hierarchy;
- DeepSeek cancellation may require manual restart;
- long-prefix failure injection, durable supervision and a hardened network
  edge are incomplete.

## Ready components

| Area | State |
|---|---|
| artifacts | six active artifacts under `MODEL_ROOT`, fail-closed manifests/programs and transactional publication |
| common lifecycle | alias-driven install/start/status/chat/stop through one task and VM |
| QPack execution | block-32 FP4 and source-native block-16 NVFP4 providers execute real service paths |
| DeepSeek paging | exact top-6, stable aggregation and measured NVMe/RAM/VRAM traffic |
| exact KV allocation | artifact-declared dtype/geometry, progressive pages and Qwen exact device mirror |
| retained sessions | transactional suffix resume/rewind for providers that advertise it |
| APIs | bounded OpenAI and Anthropic-compatible streaming surfaces |
| Pi wiring | all six aliases pass the no-context/no-tools/no-session `hi` gate |
| introspection | bounded health/readiness/model/metrics paths and request telemetry |

## Open blockers

| Area | Blocker |
|---|---|
| Qwen 262K target | capacity inequality, 36-minute exact prefill and no saturated 15 tok/s decode pass |
| real coding harness | only bounded Qwen/Ornith evidence; long real-project quality and wall time unqualified |
| Qwen Flash | slow startup/short Pi latency and repeated zero-edit action failure |
| Mistral | coherent native execution but high TTFT and zero edits in the recorded coding gate |
| Muse | text only; no populated 131K or representative coding gate |
| DeepSeek | novel movement cost, minimal Pi latency, settled rate below 10-15 and manual recovery after some cancellations |
| concurrency | one serialized hot slot; parking is continuity, not simultaneous decode |
| public service | no integrated TLS, public rate limiting, durable log retention or complete supervisor recovery |
| multi-GPU | not implemented and not part of the current one-3090 acceptance host |

## Release invariants

- `MODEL_ROOT` is the only host-dependent model-storage root.
- The current common routed-VRAM ceiling is 12 GiB. Do not raise it without
  Qwen and DeepSeek chat plus cleanup; DeepSeek currently fails at 13 GiB.
- Qwen/Muse/Ornith/Flash execute declared FP4 payloads; Mistral executes native
  NVFP4; format names are never substituted.
- Exact target F16 KV is not replaced with FP8/Q4 without explicit fidelity
  approval and a separate quality gate.
- DeepSeek preserves exact router/top-k and receives no unsupported session
  commands.
- Windows Release/CUDA verification is clean-first.
- `ready=true` is an admission/liveness state, not an SLO.
- Every release gate ends with service/process/GPU cleanup.

## Promotion evidence still required

1. clean Windows Release/CUDA and canonical Python tests;
2. official-reference numerical/behavior gates for the changed capabilities;
3. real common-path chat smokes for the affected model, Qwen and DeepSeek, plus
   every model claimed universal by the release;
4. representative Pi project histories, tools, compaction and suffix reuse;
5. populated maximum-context prefill/decode meeting the accepted model target;
6. large-prefix timeout/cancellation/restart/KV-pressure recovery;
7. authenticated TLS/rate-limited edge, supervision, observability and tested
   rollback.

See [Benchmarks](benchmarks.md) for evidence and [Roadmap](roadmap.md) for the
active order.
