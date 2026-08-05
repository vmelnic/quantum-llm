# Route census and stable warm placement

The route census preserves bounded placement evidence across process restarts.
It is not a request log and is not on the per-expert compute hot path.

## Stored evidence

The table has exactly `layer_count × experts_per_layer` cells. Each cell stores:

- lifetime selection count;
- an aged Q20 heat value and its observation epoch;
- last-seen observation;
- CPU and GPU selection counts.

Per-layer state stores only the previous routed expert IDs, observation count,
and consecutive-route overlap count. The census never stores prompts, tokens,
activations, outputs, request IDs, network identities, or user metadata.

For DeepSeek-V4-Flash the fixed table is `43 × 256` cells. With six expert IDs
of previous-route state per layer, one serialized generation is 530,260 bytes.
Its size does not grow with requests, tokens, uptime, or a future checkpoint's
parameter count; it grows only with the declared layer/expert geometry.

## Identity and recovery

Every generation is bound to:

- schema version;
- model ID and full model-content SHA-256;
- quantization ABI;
- layer, expert, and route geometry;
- heat-decay interval.

The runtime writes alternating `<prefix>.0` and `<prefix>.1` slots. Each slot is
independently SHA-256 authenticated, and startup selects the newest valid
generation. A torn or corrupt newest write therefore falls back to the prior
generation. A file from another checkpoint, ABI, or geometry fails closed.

## Aging and ranking

One observation is one completed routed layer. A selected cell receives one
Q20 heat unit. Its heat halves after each configured interval without another
hit; aging is materialized lazily, so observation remains constant-time and
does not sweep all 11,008 cells. The DeepSeek default interval is 4,096 routed
layers, approximately 95 complete 43-layer tokens.

`stable_warm_set()` ranks observed experts by effective heat, lifetime count,
recency, then stable expert identity. Callers supply both a global entry budget
and an optional per-layer cap. The result is placement evidence, not an
unbounded or automatic VRAM reservation: the resource governor and current
hardware profile remain authoritative.

## Scheduler lifecycle

The DeepSeek scheduler records a route only after the FFN layer has completed.
It records the exact routed union and which selections ran on CPU before
releasing host/device leases. Cancelled, failed, or merely predicted work is
not counted. The real-model qualification covers a one-CPU/five-GPU route and
verifies six selections in the census without changing model output.

Persistence is called at safe service lifecycle boundaries rather than after
every layer. The production worker will load the newest generation during
model startup, use the bounded ranking to seed warm placement under the memory
governor's budget, and checkpoint it periodically and during graceful drain.
