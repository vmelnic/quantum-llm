# Production readiness

## Verdict

The repository is suitable for a **controlled pre-production pilot at a
4096-token context on the tested Windows/RTX 3090 host**. It is not ready for a
general multi-tenant or internet-facing production service.

## Readiness matrix

| Area | Status | Evidence / gap |
|---|---|---|
| model/container integrity | ready | strict schema, pack and record hashes, atomic completion marker |
| deterministic correctness | ready for tested model | isolated/batched equality and CUDA/compiler tests |
| bounded memory and queues | ready for pilot | explicit RAM/VRAM/staging/KV budgets and bounded admission |
| API client compatibility | ready for text/greedy subset | verified with OpenAI SDK; unsupported capabilities fail explicitly |
| cancellation and drain | ready for pilot | client FIN/RST cancellation and full descendant-tree stop |
| context window | limited | paged FP16 KV and bounded four-token causal prefill implemented; 4096 certified |
| cold latency | not ready | cold service p95 remains far below the hot throughput target |
| authentication | partial | one shared bearer key; no identity, tenant, or rotation service |
| TLS / edge security | missing | requires external reverse proxy and firewall |
| observability | partial | metrics and JSONL exist; no external retention, tracing, or alerting |
| supervision | partial | Task Scheduler pilot; not a hardened Windows service |
| high availability | missing | one process, one GPU, no replica/failover |
| long soak / chaos | missing | no 24h soak, forced I/O/CUDA failure campaign, or leak gate |
| GPU CI | manual | portable GitHub CI exists; SM86 validation needs a self-hosted runner |
| security review | missing | no independent audit or fuzz campaign |

## P0 blockers for broader production

### Long-context prefill and qualification

Completed: paged on-demand FP16 KV, per-request page credits, bounded reuse,
worker protocol v3, page observability, constant-shared-memory online decode
attention, and a four-token causal prefill slice with full-model equality
against scalar prefill.

Required work: a prefill workspace independent of decode concurrency, larger
adaptive chunks, FlashAttention-class kernels, mixed-length batch scheduling,
RoPE/numerical validation, and correctness/memory/SLO gates at 8K, 16K, 32K,
and 64K. Do not claim 262K before it passes.

### Cold-path latency

Hot placement crosses the throughput gate; cold API latency does not. Required
work: workload-aware warm sets, admission-aware placement, prefetch that cannot
starve ready work, and stable SLO tests using realistic prompt distributions.

### Service hardening

Required work: Windows Service or equivalent supervisor, non-interactive
identity, secret store integration, TLS reverse proxy, structured log rotation,
request correlation, resource alerts, and documented backup/restore drills.

### Reliability validation

Required work: long soak under mixed concurrency/context, cancellation storms,
overload recovery, SSD short reads, corrupted packs, CUDA reset/OOM, worker
crash/restart, disk-full behavior, and repeated deploy/rollback.

## Pilot release checklist

- [ ] pin commit, model revision, compiler profile, Python dependencies, CUDA and driver;
- [ ] validate all pack hashes and preserve the source checkpoint or verified backup;
- [ ] run Windows build/tests and both P6 gates;
- [ ] run service smoke and verify build/model hashes through `/model-info`;
- [ ] use 4096 or a separately qualified context limit;
- [ ] bind loopback, or deploy proxy/TLS/firewall/API key controls;
- [ ] configure external logs, metrics collection, disk/RAM/VRAM alerts;
- [ ] document capacity, owner, maintenance window, rollback commit, and stop command;
- [ ] run an environment-specific load test and record cold plus warm latency;
- [ ] obtain explicit operator acceptance of the known limitations.

## Claims policy

“Model supports 262K” describes checkpoint metadata. “Runtime supports 262K”
requires successful allocation, prefill, decode, numerical validation, and SLO
evidence at that context. The project currently makes only the first claim.
