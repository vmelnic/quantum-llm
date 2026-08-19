# Roadmap

Status: superseded on 2026-08-11.

The active, dependency-ordered implementation backlog is
[MoE VM current state and remaining work](moe-vm-next.md). Start the next work
session there. This file remains as a stable link for older documents.

The previous P0–P7 roadmap mixed several distinct programs: serving speed,
long-context qualification, service hardening, platform portability,
distributed experts, trillion-parameter scaling and external knowledge. Their
durable results are preserved in [Engineering history](history.md), and their
measurements are preserved in [Performance evidence](benchmarks.md).

Current tracks are intentionally separated:

1. finish the universal physical artifact and composable operation executor;
2. onboard a fourth compatible FP4 MoE without common-path code changes;
3. optimize representative chat only after it runs through that generic path;
4. qualify long context and production failure handling independently;
5. consider CPU/distributed/remote expert owners only after exact local
   placement, cancellation and resource contracts are complete.

The external Memory Expert/KV-attach experiment is owned by the standalone
`../memory-expert` project and is not part of this runtime roadmap.
