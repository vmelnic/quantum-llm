# DeepSeek-V4-Flash serving

The DeepSeek backend has a persistent CUDA worker compatible with the same
bounded OpenAI HTTP front-end used by the Qwen backend. It owns the resident
dense model, shared experts, request state, route census, RAM/VRAM caches, and
hybrid CPU/CUDA scheduler for its complete lifetime. HTTP request handlers do
not load checkpoints or create CUDA contexts.

## Runtime bundle

`Publish-DeepSeekWorkerBundle.ps1` turns qualified descriptors into a small,
atomically published runtime bundle:

```text
deepseek-worker-bundle-v1/
  manifest.json       public service identity and architecture
  runtime.tsv         exact runtime dependencies and measured placement seed
  dense/              authenticated FP8 dense descriptors
  typed/              authenticated BF16/F32/I64 descriptors
  shared/             authenticated shared-expert descriptors
```

The source checkpoint and routed expert pack remain external immutable
dependencies; they are not copied into the bundle. The route census is mutable
and lives in a separate state directory. `runtime.tsv` binds the worker to the
checkpoint-index SHA-256, quantization ABI, descriptor roots, routed catalog,
and measured CPU/CUDA/H2D seed. Unknown, duplicate, missing, corrupt, or
model-mismatched state fails closed. A missing census is the only state-load
condition that creates a new empty census.

For development, the routed catalog may point directly at authenticated
SafeTensors extents. For deployment, publish only after the 43-shard compact
pack is complete. The publisher records `source-extents` or `compact-pack` in
the manifest so the distinction is visible to operators.

```powershell
./ops/windows/Publish-DeepSeekWorkerBundle.ps1 `
  -Snapshot D:\models\deepseek-v4-flash\snapshot `
  -DescriptorBundle D:\qualification\deepseek-descriptors `
  -RoutedCatalog D:\models\deepseek-v4-flash\compact-pack-v1 `
  -Output D:\models\deepseek-v4-flash\worker-bundle-v1 `
  -StateDirectory D:\state\deepseek-v4-flash
```

The placement arguments default to the checked-in qualification sample. They
are not universal model constants. Re-measure CPU expert execution, routed
CUDA execution, and pinned H2D bandwidth when hardware, clocks, NUMA placement,
or memory configuration changes, then pass the measured values explicitly.

## Start the API

Build the CUDA runtime, install `requirements/server.txt` in the isolated
server environment, and start the model-specific wrapper:

```powershell
./ops/windows/Invoke-BuildExpertRuntime.ps1 -Configuration Release

python -m venv work\venv\server
./work/venv/server/Scripts/python.exe -m pip install `
  -r requirements/server.txt

./ops/windows/Start-DeepSeekExpertServer.ps1 `
  -Bundle D:\models\deepseek-v4-flash\worker-bundle-v1 `
  -HostAddress 127.0.0.1 `
  -Port 8080
```

The wrapper defaults to a 40 GiB host expert cache, a 12 GiB device expert
cache, two request slots, 4096 context tokens, and a balanced placement policy.
`placement_prefetch_state=observing` means the worker is collecting route
evidence only. It does not report prefetch as enabled until a bounded warm-load
or lookahead consumer is actually active.
When an authenticated prior census exists, balanced and latency profiles fill
the bounded RAM tier before worker readiness and report
`placement_prefetch_state=ready`. Capacity mode deliberately reports
`disabled`. The warm load is globally bounded by the configured RAM cache and
uses a per-layer cap so one layer cannot consume the complete tier.
The first production qualification loaded 864 routed records (11.55 GB) in
6.27 seconds. A repeated five-step route then used no scheduler suspension and
took 340.886 ms of worker model-step time while preserving token `19923`.

`STATS`, `/model-info`, and `/metrics` expose cumulative worker attribution.
The timing counters deliberately describe their measured boundary:

- `worker_model_step_ns` covers embedding through sampled-token readback;
- `worker_embed_rope_submit_ns`, `worker_scheduler_poll_ns`, and
  `worker_output_head_ns` partition that worker wall path;
- `scheduler_controller_advance_ns` is time inside controller advancement;
- `worker_attention_route_submit_ns` and `worker_ffn_submit_ns` are CPU launch
  time only;
- `worker_directory_plan_ns` includes the stream drain for attention/router,
  while `worker_directory_release_ns` includes the stream drain for routed and
  shared FFN work;
- `scheduler_expert_wait_ns` covers suspended route resolution;
- `cache_storage_wait_ns`, `cache_ram_retention_copy_ns`, and
  `cache_upload_wait_ns` measure the cache pipeline stages.

The scheduler/cache counters can overlap their parent worker counter and must
not be summed with it as independent wall time. They exist to attribute the
parent interval and compare deltas between two snapshots.

Each live request owns one non-blocking CUDA stream and one reusable hybrid
workspace. RoPE values for the configured context are precomputed once into a
bounded resident device table. `worker_execution` exposes these effective
contracts; no per-token workspace allocation or host RoPE upload remains in the
production path.
Startup performs hard RAM, VRAM, request-state, and logical KV-credit
preflights before the ready message. These defaults fit the qualified 64 GiB
RAM / 24 GiB VRAM class, but operators must lower them when other processes
consume memory. Pagefile use is not required to be disabled; physical-memory
preflight prevents treating swap as usable expert-cache capacity.

For a non-loopback bind, set `EXPERT_API_KEY`; the common server refuses an
unauthenticated public bind. See [OpenAI-compatible API](openai-api.md) for
request formats and supported compatibility behavior.

After the service is ready, run one model-specific integration gate:

```powershell
./ops/windows/Invoke-DeepSeekServiceSmoke.ps1 `
  -ExpectedBuildId (git rev-parse --short HEAD)
```

The production gate rejects a source-extent routed catalog. During backend
development only, `-AllowSourceExtents` permits that storage mode. The gate
checks model and bundle hashes, backend-specific prefill/KV metadata, one real
two-token completion, metrics, and complete request/KV cleanup.

For a controlled pilot, register a separate restartable task after the gate
passes. Registration does not start it unless `-Start` is supplied:

```powershell
./ops/windows/Install-DeepSeekExpertServerTask.ps1 `
  -Bundle D:\models\deepseek-v4-flash\worker-bundle-v1 `
  -BuildId (git rev-parse --short HEAD)

# Explicit lifecycle
Start-ScheduledTask QuantumLLM-DeepSeekV4Flash
./ops/windows/Stop-ExpertServer.ps1 `
  -TaskName QuantumLLM-DeepSeekV4Flash -Port 8080
./ops/windows/Uninstall-ExpertServerTask.ps1 `
  -TaskName QuantumLLM-DeepSeekV4Flash -Port 8080
```

## Verified vertical slice

The first service gate used the source-extent catalog while the durable pack
was still being produced. It proved:

- protocol 4 startup with the requested cache and KV geometry;
- one real token-ID request through `BEGIN`, inference, `NEXT`, and shutdown;
- one non-streaming `/v1/completions` request using the real tokenizer;
- correct request cleanup and VRAM release after service stop.

The one-token HTTP request completed in 4.86 seconds in that cold development
state. This is functional evidence, not the target throughput result. Final
latency and throughput qualification must use the completed compact pack,
declare cold/warm placement, and follow the benchmark rules.

The durable publication gate subsequently completed all 43 routed shards at
exactly 147,169,738,752 payload bytes, published a `compact-pack` worker bundle,
and passed the production-only service gate. With 4096 configured context,
two worker slots, a 40 GiB RAM cache, and a 12 GiB VRAM cache, the one-prompt/
two-output-token request completed in 17.24 seconds and returned `", I"`.
Context credits and physical request state returned to zero. The final clean
Windows build passed 4/4 native tests and 32/32 Python tests.

This proves packaging, startup, real inference, HTTP integration, accounting,
and cleanup. It does **not** satisfy the throughput objective. The measured
request is roughly 0.12 output tok/s when divided naively by wall time, and the
reported 17.20-second TTFT includes synchronous preparation of the following
token under protocol 4. Pack layout removed about 43% of the equivalent
source-extent wall time (30.52 seconds), but execution and memory movement on
this host remain the dominant product problem.

## Current boundary

The worker implements exact greedy decoding. Continuous decode batches rows
from concurrent HTTP requests into one scheduler poll cycle, while each routed
layer may place experts on all-core CPU execution or direct compact CUDA
execution. Sampling controls beyond the API's currently documented behavior,
long-context qualification beyond 4096, crash-supervised deployment, and the
30 tok/s performance target remain release work. The compact-pack service gate
and task-based pilot deployment path are implemented; a long soak and failure
campaign have not been run.
