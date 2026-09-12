# Runtime contract

Status: current VM, provider and worker contract, 2026-09-12.

## Program and provider binding

Every callable artifact publishes `runtime-model.tsv` in
`expert-runtime-model-v1` format. Schema 3 describes model geometry, ordered
operations, record references, encodings and required capabilities. The common
runner validates the program and binds exactly one compatible provider. Missing
or ambiguous bindings fail startup.

Source adapters may know upstream tensor names. Common service and VM code may
not branch on a model family, architecture ID, layer count or absolute tensor
path. Existing operations require only a new validated artifact; new
mathematics or encoding requires a provider and independent numerical oracle.

## Worker lifecycle

Startup performs artifact validation, provider construction, program
preparation and a capability handshake before readiness. The handshake reports
model identity, context/output limits, KV policy, session capabilities and
telemetry schema.

The protocol supports request begin, bounded prompt feed, generation,
cancellation, status and shutdown. Checkpoint, resume, rewind, retain and drop
are sent only when the selected provider advertises exact session retention.
DeepSeek does not; sending those commands is a protocol error.

For retaining providers, resume is transactional: parked state is consumed
only after rebind and suffix feed succeed, a zero-length suffix is valid, and
failure returns to the committed client-echoable prefix. Partial assistant
output is never silently retained after cancellation.

## Placement

Dense/hybrid provider:

- hot FP4/NVFP4 matrices and activation tiles remain on the GPU;
- recurrent state remains in the single hot slot;
- exact KV grows according to artifact-declared dtype and global/window/latent
  geometry;
- Qwen uses authoritative 256-token pinned-host F16 pages plus a bounded exact
  device mirror;
- inactive retaining sessions own real host bytes and a continuation blob.

Sparse provider:

- standard QPack experts and DeepSeek compact experts share logical page
  identity and exact CPU/GPU execution contracts;
- demand has priority over speculative work;
- leases and completion events protect in-flight records;
- RAM and VRAM retention classes isolate cold first touches from hot pages;
- DeepSeek absent pages may require NVMe, while a complete fitting routed pool
  such as Ornith can remain in the host bank.

MTP/draft resources are allocated only when the effective generation policy
can use their verifier. They are never counted as free or hidden from capacity
and throughput reports.

## Capacity and accounting

Worker slots, queued requests, pinned pages, routed RAM, routed VRAM, staging
buffers, workspaces, reserve and output tokens are independent credits. A
configured context limit reserves none of them by itself. Admission uses actual
owned bytes, including retained KV and continuation state.

Telemetry separates:

- prompt, generated, reasoning/useful tokens, TTFT and wall time;
- exact KV pages/bytes, mirror/staging traffic, park/restore bytes;
- routed VRAM/RAM hits, SSD misses, reload/reread bytes and storage/H2D wait;
- CPU/GPU expert decisions, prefetch/warm work and cancellations.

Status and metrics use bounded snapshots and must not block on a long prefill.
Profiling that introduces CUDA synchronization is disabled in normal serving.

## Correctness boundary

- unknown fields, operations, encodings, record versions or capabilities fail;
- selected experts are never dropped and top-k is never reduced;
- exact target F16 KV is not silently replaced with FP8/Q4;
- artifact sampling, template, EOS, response and media declarations are
  authoritative unless a supported request field explicitly overrides them;
- provider-specific session behavior is never assumed universal;
- service success requires real execution through the declared provider, not
  only manifest parsing.
