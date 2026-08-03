# Expert Runtime v1 — contract Windows/CUDA

Acest document fixează lifecycle-ul și ABI-ul dintre core, storage/cache și
backend-ul CUDA. Platforma P0–P6 este Windows + RTX 3090 (SM86). Metal și
execuția distribuită sunt faze viitoare și nu fac parte din ABI-ul v1.

## Limita dintre module

```text
manifest/index -> feasibility/admission -> request state
                                           |
router exact -> work (layer,expert,row) -> scheduler expert-centric
                                           |
                         +-----------------+----------------+
                         | VRAM hit        | RAM hit        | SSD miss
                         v                 v                v
                     grouped CUDA     async H2D      IOCP direct read
                         +-----------------+----------------+
                                           v
                                weighted exact aggregation
```

Core-ul folosește IDs, byte budgets, stări și interfețe; handle-urile Win32 și
tipurile CUDA rămân în backend-urile lor. Nicio dependență Colibri/llama.cpp și
nicio cale Metal/network nu intră în build-ul curent.

## Mașina de stări a containerului

```text
UNOPENED -> MANIFEST_VALID -> PACKS_VALID -> PLANNED -> READY
    |             |              |             |
    +-------------+--------------+-------------+--> FAILED
READY -> DRAINING -> CLOSED
```

`READY` cere manifest strict, `COMPLETED`, hashes, ABI SM86 compatibil și plan
fezabil. `impossible` nu ajunge în READY cu promisiunea SLO; numai un flag
operator explicit poate porni `best-effort`, stare vizibilă în API și metrici.

## Mașina de stări a expertului

Cheia este `(model_content_hash, layer, expert, quant_abi)`.

```text
ABSENT -> SSD_LOADING -> RAM_READY -> GPU_UPLOADING -> VRAM_READY
   ^          |             |              |              |
   |          +--FAILED-----+--------------+--------------+
   +---------------- eviction după refcount/event --------+
```

Reguli obligatorii:

- un singur load și un singur upload in-flight per cheie;
- waiterii se atașează aceleiași operații, nu pornesc duplicate;
- slotul RAM devine vizibil numai după read complet + checksum;
- slotul VRAM devine vizibil numai după `cudaMemcpyAsync` + event complet;
- `refcount > 0`, rezervarea schedulerului sau un CUDA event incomplet interzic
  eviction;
- tranzițiile consumă credits înainte de alocare și le restituie exact o dată;
- cancellation elimină waiterul, nu invalidează un load încă necesar altora;
- short read, checksum sau CUDA error duc în `FAILED` și închid request-urile
  dependente; top-k incomplet nu continuă;
- high watermark oprește admission/load, low watermark îl reactivează;
- bugetele VRAM, RAM cache, staging și KV sunt independente și în bytes.

## Request și work item

Un request trece prin:

```text
QUEUED -> ADMITTED -> PREFILL -> DECODE_LAYER -> STREAMING -> COMPLETE
             |            |           |              |
             +------------+-----------+--------------+-> CANCELLING -> CANCELLED
             +------------+-----------+--------------+-> FAILED
```

Work item-ul immutable conține cel puțin request/sequence/row, layer, expert,
routing weight, KV slot, deadline și generația cache entry. Schedulerul poate
reordona work items între request-uri, dar nu poate schimba routerul, top-k sau
ordinea numerică de agregare dintr-un request.

Ready-first execută grupurile VRAM-ready, suprapune upload/read și aplică
fairness prin vârstă/deadline. Un request rece nu ține blocată coada ready.

## Contract storage Windows

- pack deschis read-only cu `FILE_FLAG_OVERLAPPED`;
- `FILE_FLAG_NO_BUFFERING` numai când offsetul, lungimea și bufferul satisfac
  alinierea efectivă a volumului și `manifest.alignment.direct_io_bytes`;
- IOCP unic/pool controlat, nu thread blocant per expert;
- buffers page-locked dintr-un pool fix, cu owner și generation explicite;
- un read direct acoperă `stored_bytes`, inclusiv padding determinist;
- coalescing numai pentru recorduri vecine compatibile, iar requested/useful/
  read/overfetch bytes se contorizează separat;
- EOF, short completion, device removal și checksum mismatch sunt fail-closed.

## ABI CUDA `expert-pack-sm86-int8-row-v1`

Container ABI:

- weight `I8`, simetric per output row, zero-point 0;
- scale `F32`, little-endian în pack, convertibil la device type numai explicit;
- `gate_up_q [2I,H]`, gate rows apoi up rows, row-contiguous;
- `gate_up_scales [2I]` în aceeași ordine;
- `down_q [H,I]`, output-major row-contiguous;
- `down_scales [H]`;
- activarea este SiLU; outputul expert este `down(silu(gate(x))*up(x))`;
- weighted accumulation este device-local și folosește o ordine stabilă
  `(request,row,expert_id)` pentru modul exact.

Descriptorul de dispatch nu conține pointeri host și nu expune layout implicit:

```text
ExpertBatchV1
  abi_id = 1
  H, I, row_count
  activation_dtype
  output_dtype
  x_device[row_count,H]
  gate_up_q_device, gate_up_scale_device
  down_q_device, down_scale_device
  routing_weight_device[row_count]
  output_device[row_count,H]
  workspace_device + workspace_bytes
  completion_event
```

Backend-ul validează `abi_id`, SM capability, dimensiunile și bounds înainte de
launch. Pointerii weight trebuie să provină dintr-un slot `VRAM_READY` rezervat.
Gate și up sunt un singur grouped dispatch logic; activarea de input nu se
recuantizează separat pentru cele două proiecții. SiLU/produsul rămân device-
local, apoi down și acumularea ponderată rulează fără round-trip CPU.

Streams logice:

- compute: dense/router/grouped experts/aggregation;
- H2D: upload din pinned RAM;
- preload: opțional, numai dacă nu întârzie ready work.

CUDA events exprimă dependențele și protejează eviction. Shapes/buffers pentru
microbatch-urile acceptate sunt prealocate; alocarea necontrolată în decode este
o eroare de runtime.

## Fezabilitate și admission

Calculatorul comun primește `manifest.json`, inventory și request-ul strict din
`schemas/feasibility-request-v1.schema.json`. Din manifest folosește
`masses.pack_bytes`, `masses.expert_bytes`,
`masses.active_expert_bytes_per_token`, `requirements.resident_dense_bytes` și
media `experts[].stored_bytes`. Workspace, staging și KV/request sunt tunables
runtime, nu un al doilea dialect de manifest.

Pentru ținta `T`:

```text
cold_budget_B_per_token = sustained_storage_Bps / T
h2d_budget_B_per_token  = sustained_H2D_Bps / T
required_storage_avoidance = ceil((active - cold_budget) / active * 1e6)
resident_vram = dense + workspace + kv_per_request * concurrency
```

Se verifică în ordine stabilă: VRAM rezident, VRAM total cu cache, RAM cu
staging/cache, disk capacity, storage bandwidth și PCIe/H2D bandwidth. Prima
constrângere eșuată este `limiting_resource`, împreună cu `required`, `available`
și formula. Bandwidth modelat, nu măsurat, nu poate produce `feasible`; produce
cel mult `degraded`.

Exit code-ul CLI este 0 pentru feasible/degraded, 2 pentru impossible și 65
pentru manifest/inventory/request invalid. Inputurile canonice și decizia au
SHA-256, astfel încât aceeași intrare produce aceeași decizie reproductibilă.

## Telemetrie minimă

Fiecare request și proces expun TTFT/inter-token/tok/s, wait admission/batch,
dense/router/expert/H2D/SSD time, rows și experți unici, hit VRAM/RAM/miss SSD,
requested/useful/read/uploaded bytes, overfetch/dedup, high-water marks, eviction,
budget stalls, cancellation și toate erorile checksum/I/O/CUDA. Identificatorii
build/model-content/config sunt obligatorii în health, logs și incidente.

## Invariante de corectitudine

1. Niciun slot nu este vizibil înainte de validare/completion.
2. Niciun expert rezervat sau în execuție nu este evicted.
3. Niciun failure nu devine weight zero sau top-k redus în tăcere.
4. Interleaving-ul schimbă scheduling-ul, nu rezultatul fiecărui request.
5. Memoria nu depășește bugetele; overload produce backpressure explicit.
6. Orice byte I/O/H2D și orice fallback sunt contorizate.
7. Startup-ul nu descarcă și nu modifică modelul.
