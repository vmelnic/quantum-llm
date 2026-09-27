# Runtime contract

Status: implemented VM, provider and worker contract, 2026-09-27.

## Model program

Every callable artifact publishes `runtime-model.tsv` in
`expert-runtime-model-v1` schema 3. It declares:

- model and context geometry;
- program inputs and outputs;
- ordered operations and tensor bindings;
- routed components and exact routing geometry;
- required capability names and ABI versions;
- optional exact-decode and media contracts.

The native runner authenticates the program through the manifest, validates
it, selects exactly one compatible provider module and compiles the operation
sequence. An absent or ambiguous provider is a startup error.

## Worker startup

The Python service launches the native runner with explicit context, cache,
placement and diagnostic options. Startup completes these stages before
readiness:

1. artifact load and validation;
2. provider construction;
3. program binding and preparation;
4. provider finalization;
5. service-contract validation;
6. protocol-12 readiness handshake.

The handshake reports model identity, context, vocabulary, capabilities, KV
format and page geometry, placement, MTP state, sampling support, session
capabilities and telemetry configuration. The service rejects inconsistent
contracts.

## Request protocol

The line protocol supports fresh and resumed request begin, bounded prompt
feed, decode steps, cancellation, status and shutdown. Retain, drop, save,
load and prune are valid only when the selected provider advertises the
corresponding session capability.

Resume is transactional. The parked state is consumed only after request
rebind and suffix feed succeed. A zero-length suffix is valid. Cancellation
can rewind to a declared prompt checkpoint so a later client never inherits
unseen assistant tokens.

## Admission and accounting

Queue entries, hot worker slots, context pages, parked session bytes, routed
RAM, routed VRAM, workspaces and output limits are separate resources. A
configured context limit reserves none of them by itself. Admission uses real
owned state and may evict an eligible retained session before retrying a
capacity failure.

Telemetry separates prompt and generated tokens, reasoning and visible output,
TTFT, prefill and total wall time, KV pages and traffic, session park/restore,
routed tier hits and misses, storage reads, uploads, waits, expert decisions
and cancellation outcome. GPU phase timing is opt-in because it synchronizes
the measured path.

## Correctness rules

- selected experts, top-k and aggregation order are preserved;
- exact F16 KV is never silently replaced by a compact format;
- request sampling overrides only supported fields;
- artifact template, EOS, response parser and media declarations are
  authoritative;
- one parser instance owns each streamed response from first delta to final
  structured output;
- unsupported operations, sessions, tools or media fail explicitly;
- service success requires provider execution, not manifest parsing alone.
