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
Startup performs hard RAM, VRAM, request-state, and logical KV-credit
preflights before the ready message. These defaults fit the qualified 64 GiB
RAM / 24 GiB VRAM class, but operators must lower them when other processes
consume memory. Pagefile use is not required to be disabled; physical-memory
preflight prevents treating swap as usable expert-cache capacity.

For a non-loopback bind, set `EXPERT_API_KEY`; the common server refuses an
unauthenticated public bind. See [OpenAI-compatible API](openai-api.md) for
request formats and supported compatibility behavior.

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

## Current boundary

The worker implements exact greedy decoding. Continuous decode batches rows
from concurrent HTTP requests into one scheduler poll cycle, while each routed
layer may place experts on all-core CPU execution or direct compact CUDA
execution. Sampling controls beyond the API's currently documented behavior,
long-context qualification beyond 4096, crash-supervised deployment, and the
final compact-pack service gate remain release work.
