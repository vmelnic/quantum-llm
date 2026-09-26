# Production readiness

Status: functional research/pilot runtime; not production-ready for the full
maximum-context coding objective, 2026-09-25.

## Verdict

The six advertised in-scope Pi models returned text through isolated `xhigh`
`hi` on the CUDA 13.4 clean build. This qualifies wiring only, not coding,
behavioral quality, long context or sustained throughput. The separate
abliterated-MSE artifact was not part of this Pi catalog. Artifact validation,
lifecycle, bounded API limits, telemetry, exact sparse routing and cleanup are
functional.

The complete product goal is not ready:

- Qwen exact-F16 populated 262,016 tokens but took 2,278.082 s; the lossy
  per-head-Q4 default passed 40 useful tok/s at 32K/128K but has no live 262K
  or real-project fidelity result;
- one hot provider slot serializes all agents;
- Qwen Flash and Mistral are callable but fail coding latency/behavior gates;
- Ornith's historical K1+fit run failed its coding fixture 2/3 after
  1,832.80 s; the new per-head-Q4 default has only a minimal Pi smoke;
- one Ornith direct `xhigh` `hi` stopped after reasoning without visible text;
  later requests passed, so intermittent empty-answer reliability is open;
- the CUDA 13.4 clean build has 14,992 vendored CUTLASS/FlashAttention
  warnings despite zero first-party and generated-stub warnings;
- durable supervision, long-prefix failure recovery and a hardened network
  edge remain incomplete.
- a Qwen native worker faulted inside the NVIDIA CUDA driver during sequential
  model switching on 2026-09-24; an unchanged retry passed, but restart
  reliability remains unqualified.

## Ready components

| Area | Current boundary |
|---|---|
| artifacts | six advertised in-scope Pi models under `MODEL_ROOT`, fail-closed manifests/programs and transactional publication |
| lifecycle | alias-driven install/start/status/chat/stop through one task and VM |
| execution | declared QPack FP4 and native NVFP4/BF16 execute real service paths |
| paging | exact top-k/stable merge with measured NVMe/RAM/VRAM traffic |
| KV/session state | artifact-declared progressive allocation; Qwen provider supports transactional RAM parking and durable, lazy NVMe restart/resume with TTL/LRU and explicit per-artifact cleanup |
| APIs | bounded OpenAI- and Anthropic-compatible streaming surfaces |
| harness wiring | current defaults for six advertised in-scope models pass isolated minimal Pi `hi` at `xhigh` |
| introspection | bounded health/readiness/model/metrics and per-request telemetry |

## Release blockers

| Area | Required result |
|---|---|
| Qwen 262K | populated exact-F16 real-harness prefill and about 15 useful tok/s |
| harness | reliable representative coding, tools, compaction and retained-prefix behavior |
| Flash/Mistral/Ornith/Muse | each model's documented latency, quality and populated-context gate |
| build warnings | eliminate or replace the vendored code emitting 14,992 warnings without suppression; repeat the clean build |
| concurrency | admission and continuity under real competing sessions; current decode remains serialized |
| operations | durable supervisor/logs/metrics, tested rollback, TLS and rate limiting |
| P100 | no release dependency: exact routes work, but measured throughput is worse than the preferred paths |

## Release invariants

- `MODEL_ROOT` is the only host-dependent model-storage root.
- The common fixed routed-VRAM ceiling is 12 GiB; startup fitting is admitted
  only after accounting maximum future state and reserve.
- Format and fidelity names remain exact; `q4-f16-per-head` is the explicit
  lossy default for compatible artifacts, while exact F16 remains the Qwen
  acceptance target and `qwen-f16` reference.
- Native releases use a clean-first Windows CUDA build.
- `ready=true` proves liveness/admission, not warm state or an SLO.
- Every gate ends with process and device-memory cleanup.

## Promotion gate

1. clean Windows Release/CUDA build, CTest and canonical Python tests;
2. independent numerical/reference gates for every changed capability;
3. real chat for the affected model, Qwen and every other model claimed
   universal;
4. representative Pi gate when harness behavior changed;
5. populated-context and failure-recovery gates relevant to the release;
6. stop and verify process/GPU cleanup.

Measurements: [Benchmarks](benchmarks.md). Active work: [Roadmap](roadmap.md).
