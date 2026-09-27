# Documentation

This index defines the current documentation set. Each fact has one owner; use
links instead of copying measurements or lifecycle rules into another file.

## System and status

| Document | Owns |
|---|---|
| [Architecture](architecture.md) | implemented components, state ownership and failure boundaries |
| [Benchmarks](benchmarks.md) | measured results and exact test conditions |
| [Research decisions](research-decisions.md) | accepted and rejected mechanisms with reopening gates |
| [Roadmap](roadmap.md) | active dependency order |
| [Production readiness](production-readiness.md) | current release boundary and promotion gate |

## Build and operate

| Document | Owns |
|---|---|
| [Getting started](getting-started.md) | first supported setup and run |
| [Deployment](deployment.md) | aliases, resource policy, publication and lifecycle |
| [Operations](operations.md) | health, telemetry, diagnosis, recovery and cleanup |
| [Pi CLI](pi-cli.md) | maintained coding harness and 256K benchmark contract |

## Interfaces and formats

| Document | Owns |
|---|---|
| [Runtime contract](expert-runtime.md) | artifact program, provider binding and worker protocol |
| [Expert Pack v1](expert-pack-v1.md) | QPack container and record ABI |
| [OpenAI-compatible API](openai-api.md) | implemented OpenAI request and stream surface |
| [Anthropic Messages API](anthropic-api.md) | implemented Anthropic adapter surface |

Historical files and Git history do not define active support. The current
runtime, alias registry and release boundary take precedence over retired
model-specific material.

## Evidence rules

- configured context, populated context and harness prompt overhead are
  different quantities;
- a `hi` response proves wiring only;
- TTFT, prefill, decode, reasoning, visible output and total wall time are
  reported separately when available;
- format names describe actual payloads and execution paths;
- a numerical gate or microbenchmark is not service throughput;
- only the roadmap defines active work; rejected mechanisms require a new
  quantitative prerequisite before implementation.
