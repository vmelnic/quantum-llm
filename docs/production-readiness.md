# Production readiness

Status: functional research/pilot runtime; product target not ready,
2026-09-27.

## Verdict

Six published model IDs return text through the common service and isolated Pi
`xhigh` wiring checks. Artifact validation, transactional publication,
bounded APIs, native provider execution, telemetry, exact routing and cleanup
are functional.

The complete product objective is not ready:

- Qwen exact-F16 populated 262,016 positions but required 2,278.082 s;
- Q4H passed 32K and 128K decode gates but has no live 262K result or accepted
  long-project fidelity result;
- the latest Pi benchmark stalled at 58,664 populated prompt tokens;
- one hot provider slot serializes execution;
- Qwen Flash and Mistral are callable but fail practical coding latency;
- Muse and Ornith lack their complete representative qualification;
- one Ornith sampling path can stop without visible text;
- a Qwen worker previously faulted in the NVIDIA driver during sequential
  model switching; an unchanged retry passed, so restart reliability remains
  unqualified;
- the clean CUDA build emits 14,992 vendored warnings;
- durable supervision, large-prefix recovery and a hardened network edge are
  incomplete.

## Usable components

| Area | Current boundary |
|---|---|
| artifacts | six published model IDs, authenticated programs and transactional promotion |
| lifecycle | alias-driven install, start, status, chat, cache cleanup and stop |
| execution | declared QPack FP4, native NVFP4/BF16 and exact routing execute through the service |
| KV | progressive exact pages plus explicit packed policies selected by alias |
| sessions | transactional exact-prefix RAM parking; Qwen supports lazy persistent restart/restore |
| APIs | bounded OpenAI- and Anthropic-compatible text, reasoning, tools and declared image input |
| harness | six isolated Pi `hi` wiring passes |
| introspection | bounded health, readiness, model information, metrics and request telemetry |

## Release blockers

| Area | Required result |
|---|---|
| Qwen product target | 262,144 populated exact-F16 positions and about 15 useful tok/s in a real coding harness |
| Q4H operational path | complete near-256K Pi run, quality review and recovered stalled-request behavior |
| fidelity | causal exact-prefix logit diagnosis and a Romanian gate without throughput regression |
| other models | each model's roadmap latency, quality and populated-context gate |
| recovery | deterministic cancellation, timeout, failure and restart behavior at large prefix |
| concurrency | admission and continuity under competing real sessions |
| build | resolve the vendored warning boundary and repeat the clean build |
| operations | durable supervisor, bounded logs/metrics, tested rollback, TLS and rate limiting |

The P100 cards are not a release dependency.

## Release invariants

- `MODEL_ROOT` is the only host-dependent model-storage root.
- The common fixed routed-VRAM ceiling is 12 GiB.
- Q4H is labelled lossy; exact F16 remains the Qwen product target.
- Native releases use a clean-first Windows Release/CUDA build.
- Provider selection comes from artifact capabilities and ABIs.
- `ready=true` proves liveness and admission only.
- Every gate ends with process, port and GPU cleanup.

## Promotion gate

1. clean Windows Release/CUDA build;
2. CTest and canonical Python suites;
3. independent numerical/reference gates for every changed capability;
4. real chat for the affected model;
5. Qwen chat for a universal service or VM change;
6. representative Pi gate when harness behavior changed;
7. relevant populated-context and recovery gates;
8. stop and verify complete cleanup.

See [Benchmarks](benchmarks.md) for evidence and [Roadmap](roadmap.md) for
active work.
