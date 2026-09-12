# Documentation

Status: canonical documentation index, 2026-09-12.

These documents describe the current implementation and measured state. Each
fact has one canonical owner: architecture describes mechanisms, benchmarks
records measurements, research decisions records rejected directions, roadmap
contains active work and production readiness defines the release boundary.
Historical journals and detached hypotheses remain in Git history.

## System and status

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | Artifact VM, providers, model organs, memory tiers and failure boundaries |
| [Production readiness](production-readiness.md) | What is usable now and what blocks promotion |
| [Benchmarks](benchmarks.md) | Canonical measurements and capacity/bandwidth equations |
| [Roadmap](roadmap.md) | Active dependency order only |
| [Research decisions](research-decisions.md) | Rejected mechanisms and the evidence needed to reopen them |

## Build and operate

| Document | Purpose |
|---|---|
| [Getting started](getting-started.md) | Configure, build, download, publish and run |
| [Deployment](deployment.md) | Stable artifacts, aliases, lifecycle and release sequence |
| [Operations](operations.md) | Health, telemetry, slow requests, failure recovery and cleanup |
| [Pi CLI](pi-cli.md) | Coding-agent configuration, context and the current harness boundary |

## Interfaces and formats

| Document | Purpose |
|---|---|
| [Runtime contract](expert-runtime.md) | Program/provider negotiation, worker lifecycle and placement |
| [Expert Pack v1](expert-pack-v1.md) | Current QPack record/container ABI |
| [DeepSeek compact pack v1](deepseek-compact-pack-v1.md) | Current compact expert and worker-bundle layout |
| [OpenAI-compatible API](openai-api.md) | Implemented OpenAI request/streaming surface |
| [Anthropic Messages API](anthropic-api.md) | Implemented adapter and its harness limitations |

## Claims policy

- configured context, populated context and harness overhead are different;
- a `hi` smoke proves wiring, not quality or maximum-context performance;
- TTFT, total wall time, useful output and hidden reasoning are reported
  separately when available;
- FP4, NVFP4, INT8, BF16, F16 KV and runtime intermediates are never conflated;
- a theoretical bandwidth ceiling or microbenchmark is not service throughput;
- only [Roadmap](roadmap.md) is active work; rejected ideas stay in the decision
  ledger unless new measured evidence changes their prerequisite.
