# Expert Runtime v1 — contract Windows/CUDA

Acest document fixează lifecycle-ul și ABI-ul dintre core, storage/cache și
backend-ul CUDA. Platforma P0–P6 este Windows + RTX 3090 (SM86). Metal și
execuția distribuită sunt faze viitoare și nu fac parte din ABI-ul v1.

## Limita dintre module

```text
manifest/index -> feasibility/admission -> request state
                                           |
router exact -> device dispatch plan -> scheduler expert-centric
                                           |
                +--------------------------+--------------------------+
                |                          |                          |
          VRAM resident              RAM -> pinned slab          RAM host lease
          grouped CUDA               -> H2D -> CUDA              -> CPU kernel
                |                          |                          |
                +---------- per-selection exact outputs -------------+
                                           |
                                stable weighted aggregation
```

Core-ul folosește IDs, byte budgets, stări și interfețe; handle-urile Win32 și
tipurile CUDA rămân în backend-urile lor. Nicio dependență Colibri/llama.cpp și
nicio cale Metal/network nu intră în build-ul curent. Contractul de placement
acceptă însă tiers locale sau remote, astfel încât un model 1T să nu impună o
rescriere a data plane-ului.

## Data plane GPU-driven, control plane ierarhic

GPU-driven descrie cine orchestrează traseul hot, nu locul unde se află modelul
complet. Routerul scrie selecțiile direct într-un plan device. Un kernel le
deduplică/grupează și consultă un director device cu intrări versionate. Ready
work continuă pe GPU; numai cheile lipsă sunt copiate într-un miss ring bounded.

CPU/storage nu reconstruiește lista completă de pointeri la fiecare strat. El
rezolvă miss-ul din RAM, SSD sau ulterior dintr-un worker și alege un executor.
Un miss poate fi admis într-un slot resident, calculat tranzitoriu într-un slab
VRAM sau calculat lângă copia RAM. Slotul/slab-ul nu poate fi reutilizat până
când toate event-urile generației vechi s-au terminat.

## Contractul executorului heterogen

Unitatea comună este un `ExpertWorkGroup`: un singur `(layer, expert,
quant_abi)` și toate selecțiile din microbatch care îl cer. El conține host sau
device lease, indicii rândurilor, sloturile top-k, deadline-ul și destinațiile
per-selecție. Executorii disponibili sunt:

- `cuda_resident`: consumă o generație publicată în director;
- `cuda_slab`: copiază mai multe grupuri RAM-ready într-un slab pinned/device
  prealocat și publică pointerii numai pentru durata generației tranzitorii;
- `cpu_local`: consumă host lease și calculează în thread pool-ul bounded;
- `remote_worker`: rezervat P8, cu aceeași ieșire semantică.

Plannerul alege după completion time măsurat, queue depth și credits. Nicio cale
nemăsurată nu este considerată gratuită. Alegerea executorului nu schimbă
routerul, routing weights, precizia declarată sau ordinea finală de acumulare.

Fiecare executor scrie `selection_output[row, top_k_slot, hidden]`. După event-
urile tuturor selecțiilor, un kernel device aplică routing weights în ordinea
stabilă din ABI. Astfel execuția concurentă CPU/GPU nu introduce o ordine
numerică dependentă de completion.

Costurile minime urmărite online sunt:

```text
resident = gpu_queue + gpu_kernel(rows, abi)
slab     = h2d_queue + host_pack + bytes/h2d_Bps + gpu_kernel(rows, abi)
cpu      = cpu_queue + cpu_kernel(bytes, rows, abi) + result_copy
```

EWMA-urile sunt separate după ABI, geometrie și bucket de rows. Timeout-ul sau
lipsa credits produce backpressure/failure explicit, nu drop de expert.

Indexul logic este global, dar memoria este partitionată pe layer/layer-group și
pe două clase: resident și transient. Astfel, parcurgerea secvențială a unui
model cu multe straturi nu produce evacuarea ciclică observată la Qwen3-Next.
Pentru modele 1T, directorul conține metadata de ordinul zecilor de bytes per
expert și poate fi windowed; weights rămân pagini în tiers, nu intrări în
director.

Admission-ul v1 este determinist și folosește LFU cu aging periodic bounded,
astfel încât o distribuție veche să nu blocheze permanent un workload nou.
Opțional, plannerul poate rezerva un pool transient separat: primul acces intră
în acel ring, iar promovarea cere reuse și o frecvență strict mai mare decât
victima resident. Mecanismul este o decizie de placement, nu un default: pe
Qwen3-Next 80B măsurarea lui a fost mai lentă decât pool-ul LFU comun.

Placement-ul adaptiv rulează în epoci. Într-o epocă de observație, toate
rutele GPU și CPU actualizează LFU în batch pe strat, iar costul CPU per
selecție este urmărit prin EWMA. Un expert RAM devine candidat numai după ce
reuse debt-ul său depășește costul conservator H2D și există suficienți
rezidenți VRAM strict mai reci. Promovarea folosește `AcquireHandle` asincron:
invocarea curentă continuă pe CPU, iar upload-ul poate ajuta numai tokeni
viitori.

La bariera de warmup/request, plannerul drenează promovările admise și îngheață
placement-ul. În epoch-ul frozen nu există promotion, H2D de weights, copiere
de rută pentru control sau actualizare LFU/debt. Miss-urile RAM continuă prin
executorul CPU exact. Această separare împiedică rebalansarea să introducă
jitter în decode și permite reluarea explicită a observației la o frontieră
sigură de workload.

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
- `HostExpertLease` protejează copia RAM validată cât timp este folosită de CPU
  sau copiată într-un slab; nu necesită promovare implicită în VRAM;
- slab-urile pinned/device au state și generații separate de cache entry și sunt
  restituite ca unitate după ultimul event dependent;
- memoria pageable de durată nu este sursă directă pentru mii de copii CUDA
  mici; grupurile alese pentru H2D sunt coalesced într-un staging pinned bounded.

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
- după H2D, copia RAM de durată este mutată într-un buffer pageable bugetat,
  iar slotul pinned revine imediat în pool; cache-ul RAM nu poate epuiza
  staging-ul fix prin simpla retenție a experților;
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

Descriptorul CUDA resident/slab nu conține pointeri host și nu expune layout
implicit:

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
launch. Pointerii weight trebuie să provină dintr-un slot `VRAM_READY` rezervat
sau dintr-o generație activă a ring-ului slab. Calea CPU are un descriptor
separat cu `HostExpertLease`; pointerii host nu sunt publicați în directorul
device. Gate și up sunt un singur grouped dispatch logic. SiLU/produsul rămân
device-local pentru CUDA, apoi down scrie ieșirea per selecție fără round-trip
CPU.

Kernelul row-major FP32-activation existent este oracle-ul exact al ABI-ului,
nu contractul de performanță. Kernelul production grupează expert-major și
reutilizează weights pentru toate rândurile acelui expert. Poate schimba
tiling-ul, tipul intern al activării sau primitivele SM86 numai dacă profilul
numeric declarat trece față de oracle; o asemenea schimbare este versionată în
`kernel_abi`, nu activată în tăcere.

Backend-ul Qwen3-Next adaugă două token mixers exacte: GQA full-attention cu
partial RoPE și output gate, respectiv Gated DeltaNet cu stare Conv1D și stare
recurentă persistentă. Normele dense folosesc `(1 + weight)`; norma gated din
DeltaNet folosește `weight` direct, conform checkpoint-ului. Routerul face
softmax global, top-k și renormalizare pe selecția top-k înainte de dispatch.
În microbatch, proiecțiile dense consumă toate activările într-o singură lansare
și ordonează blocurile astfel încât request-urile să reutilizeze aceeași linie de
weights. Selecțiile sunt deduplicate pe strat, apoi toate rândurile folosesc un
singur dispatch MoE batched și același cache global; stările KV/Conv/DeltaNet
rămân izolate per slot.

Workerul persistent Qwen folosește protocolul local v2: mai multe request-uri
pot deține simultan sloturi KV/Conv/DeltaNet, iar `STEP` avansează până la
capacitatea negociată într-un singur `forward_batch`. Front-end-ul adună pașii
concurenți într-o fereastră configurabilă și păstrează protocolul v1 pentru
runner-ele single-slot. Admission rezervă un slot înainte de a trimite headere
HTTP/SSE; lipsa unui slot produce overload explicit, nu suprascriere de stare.

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
compute_ceiling = 1e9 / measured_non_io_nanoseconds_per_token
```

Se verifică în ordine stabilă: VRAM rezident, VRAM total cu cache, RAM cu
staging/cache, disk capacity, storage bandwidth, PCIe/H2D bandwidth și plafonul
compute măsurat. Prima constrângere eșuată este `limiting_resource`, împreună cu
`required`, `available` și formula. Bandwidth sau compute modelat, nu măsurat,
nu poate produce `feasible`; produce cel mult `degraded`.

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
