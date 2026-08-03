# Plan QMOE — handoff pentru sesiunea următoare

## Obiectiv

Construim un runtime MoE propriu, expert-centric, care poate rula un model al
cărui container cuantizat este **mult mai mare decât RAM-ul fizic**, fără să
schimbe routerul sau să aproximeze în tăcere modelul.

Țintele de produs sunt:

- minimum **10 tok/s single-stream** după warmup, când working set-ul cererii
  este servit din VRAM/RAM;
- minimum **30 tok/s aggregate** la concurența configurată, prin reutilizarea
  experților între request-uri și continuous batching;
- modelul complet rămâne pe SSD; numai dense weights și working set-ul activ
  ocupă VRAM/RAM;
- degradarea la cold miss este vizibilă în metrici și nu schimbă rezultatul
  modelului cuantizat;
- procesul respectă bugete explicite de VRAM/RAM și nu se bazează pe swap.

Acesta nu este un fork nou de Colibri sau llama.cpp. Le folosim, împreună cu
rezultatele deja obținute, ca surse de idei și controale de corectitudine. Codul
de production va avea propriul format, loader, cache, scheduler și backend CUDA.

## Starea de la care pornim

Rezultatele istorice și ramurile închise sunt în `RESULTS.md`; nu se repetă în
acest plan ca lucru activ.

Fapte deja demonstrate pe 3090box:

- OLMoE INT8 RAM-resident a eliminat I/O-ul de experți din decode și a produs
  10,19 tok/s pe gate-ul scurt, față de 1,10 tok/s prin SSD/LRU;
- containerul are 6,459 GB de experți și 7,81 GB peak RSS în modul RAM;
- routerul OLMoE activează aproximativ 807 MB de payload expert per token;
- LRU 64 a atins numai 69,2% hit pe promptul măsurat;
- pilot-prefetch a produs overfetch și nu a crescut viteza;
- reprezentările HESR-AMC codebook și neuron-bank au fost respinse după ce
  controlerul corect calibrat a produs 100% fallback.

Consecința: nu continuăm aproximarea experților și nu optimizăm vechiul engine
OLMoE. Construim execuție exactă pentru un working set paginat.

## Limitele fizice care guvernează designul

Pentru o viteză `T` și o bandă de stocare susținută `B`, bugetul maxim rece este:

```text
cold_bytes_per_token <= B / T
```

Samsung 870 QVO SATA din 3090box oferă în cel mai bun caz aproximativ 560 MB/s
secvențial, înainte de penalizările pentru acces dispersat:

| Țintă | Buget rece teoretic |
|---:|---:|
| 10 tok/s | 56 MB/token |
| 30 tok/s | 19 MB/token |

Pentru cei aproximativ 807 MB/token ai OLMoE INT8, asta ar cere cel puțin 93%
hit/reuse la 10 tok/s și 97,7% la 30 tok/s. INT4 înjumătățește aproximativ masa,
dar nu elimină constrângerea.

Runtime-ul nu poate ascunde un deficit de bandă prin scheduling. El trebuie:

1. să țină dense weights și experții fierbinți în VRAM;
2. să țină următorul working set în RAM;
3. să amortizeze experții între mai mulți tokeni/request-uri;
4. să suprapună I/O-ul rece cu muncă GPU independentă;
5. să refuze SLO-ul dacă manifestul și hardware-ul arată că este imposibil.

3090box rămâne mașina de dezvoltare Windows/CUDA. Gate-ul final pentru un model
mult mai mare decât RAM poate necesita NVMe sau mașina mai puternică; schimbarea
hardware-ului nu schimbă arhitectura și nu relaxează formulele de fezabilitate.

## Arhitectura țintă

```text
OpenAI/API requests
        |
        v
request + KV slots -----> admission/backpressure
        |
        v
dense/attention/router exact pe GPU
        |
        v
work items (request, token, layer, expert, weight)
        |
        v
scheduler global expert-centric
   |                 |                    |
   | VRAM hit        | RAM hit            | SSD miss
   v                 v                    v
grouped CUDA      async H2D          IOCP read în buffer pinned
GEMM                 |                    |
   |                 +------> cache VRAM <-+
   +-------------------------> grouped CUDA GEMM
                                 |
                                 v
                         agregare top-k exactă
                                 |
                                 v
                         următorul strat/request
```

Unitatea de scheduling nu este modelul sau request-ul complet, ci expertul
selectat într-un strat. Request-urile independente pot avansa în ordine diferită;
ordinea operațiilor din interiorul fiecărui request și agregarea top-k rămân
semantic identice.

## 1. Formatul QMOE v1

Compilerul produce un container independent de checkpoint-ul sursă:

```text
qmoe-model/
├── manifest.json
├── dense.qpack
├── experts-000.qpack
├── experts-001.qpack
├── ...
└── tokenizer/
```

### Manifest

`manifest.json` trebuie să conțină cel puțin:

- versiunea formatului și compatibilitatea minimă a runtime-ului;
- identificatorul, revizia și hash-urile checkpoint-ului sursă;
- arhitectura, vocabularul, contextul maxim și configurația RoPE/attention;
- numărul de straturi, experți/strat, top-k și shared experts;
- shape, dtype sursă, quant format și layout pentru fiecare tensor;
- maparea `(layer, expert) -> pack, offset, stored_bytes, decoded_bytes`;
- checksum per record și checksum pentru index/manifest;
- alignment-ul necesar pentru SSD, pinned host buffers și CUDA kernels;
- masa totală, masa dense, masa expert și bytes activi/token;
- kernel ABI necesar: quant type, group size, fused gate+up și down layout;
- cerințele minime estimate de VRAM, RAM, disk bandwidth și PCIe bandwidth;
- chat template, tokenizer files și tokenii speciali;
- politica de compatibilitate: runtime-ul refuză câmpuri/ABI necunoscute.

Manifestul este autoritatea. Runtime-ul nu deduce formate cuantizate ambigue din
dimensiunea tensorilor.

### `dense.qpack`

Conține tensorii rezidenți sau frecvent folosiți:

- embeddings;
- attention projections și norme;
- routere;
- shared experts, dacă arhitectura îi cere;
- final norm și lm_head.

Tensorii sunt cuantizați și aranjați direct pentru kernelul țintă. Manifestul
poate marca explicit tensorii care trebuie păstrați într-o precizie mai mare.

### `experts-*.qpack`

Fiecare record expert este independent și compute-ready:

- header fix, versionat;
- `(layer, expert_id)` și quant ABI;
- gate+up concatenate într-un singur layout;
- down projection;
- scales/zero-points/outlier metadata, dacă formatul le cere;
- payload aliniat pentru `FILE_FLAG_NO_BUFFERING`;
- checksum al payload-ului;
- lungime totală cunoscută fără parsarea tensorilor interni.

Recordurile nu traversează inutil granițele shard-urilor. Pack-urile au o
mărime configurabilă pentru copiere, verificare și distribuție, nu un expert per
fișier.

### Compiler QMOE

Compilerul trebuie să:

1. citească config/index/headere SafeTensors fără a încărca modelul complet;
2. folosească adaptoare explicite per familie de arhitectură;
3. identifice fără euristici dense, routere, shared și routed experts;
4. cuantizeze conform profilului ales, inițial INT4 sau INT8;
5. fuzioneze gate+up și să scrie layout-ul direct consumat de CUDA;
6. scrie recorduri aliniate, index, checksums și manifest atomic;
7. poată relua conversia la limita ultimului pack valid;
8. valideze fiecare record prin decode/checksum și shape;
9. calculeze working-set bytes/token și fezabilitatea pe hardware-ul inventariat;
10. producă un raport reproductibil cu sursa, opțiunile, durata și hash-urile;
11. nu modifice checkpoint-ul original și să nu publice un container parțial ca
    fiind complet.

## 2. Runtime-ul Windows/CUDA

Structura inițială propusă:

```text
runtime/
├── model/       manifest, tokenizer, tensor/expert index
├── storage/     pack files, IOCP, direct aligned reads
├── cache/       RAM/VRAM residency și eviction
├── scheduler/   request/layer/expert queues
├── cuda/        fused/quantized grouped MoE kernels
├── dense/       attention, router, lm_head și KV cache
├── server/      API, streaming, admission și lifecycle
└── telemetry/   events, counters, latency și traces
```

### `model/`

- parsează și validează strict manifestul înainte de alocări mari;
- construiește indexul immutable al tensorilor și experților;
- verifică ABI-ul container/kernel și capabilitatea GPU;
- calculează planul inițial VRAM/RAM/SSD;
- expune config-ul arhitecturii fără dependență Hugging Face la runtime.

### `storage/`

Prima platformă este Windows nativ:

- `CreateFile` cu `FILE_FLAG_OVERLAPPED` și, când este sigur,
  `FILE_FLAG_NO_BUFFERING`;
- I/O Completion Ports, fără câte un thread blocant per expert;
- offset, lungime și buffer aliniate conform manifestului și sectorului;
- pool fix de buffers page-locked/pinned pentru H2D;
- coalescing numai pentru recorduri compatibile și apropiate;
- short-read, EOF, checksum failure și device error tratate fail-closed;
- deduplicarea încărcărilor concurente ale aceluiași expert;
- contoare separate requested/useful/read/overfetch bytes;
- niciun buffer temporar nelimitat și niciun apel dependent de page cache pentru
  traseul declarat direct.

Un backend Linux `io_uring` poate fi adăugat după stabilizarea ABI-ului; nu
dictează structura v1.

### `cache/`

Cache-ul este global și bugetat în bytes, nu `N experți per strat`.

Fiecare expert trece printr-o mașină de stări explicită:

```text
ABSENT -> SSD_LOADING -> RAM_READY -> GPU_UPLOADING -> VRAM_READY
                       <----------- evict -----------
```

Sunt obligatorii:

- un singur load/upload in-flight per cheie `(model, layer, expert, format)`;
- reference count/event CUDA înainte de eviction;
- slotul nu devine vizibil până la checksum și upload complet;
- bugete separate pentru VRAM, RAM cache și staging buffers;
- watermark high/low și backpressure înainte de OOM;
- expertul în execuție sau rezervat nu poate fi evicted;
- hotness și reuse distance măsurate global, pe strat și pe request class;
- politica de admission/eviction este pluggable, dar v1 începe determinist și
  explicabil; nu introducem un predictor înainte de traces reale;
- nicio copie duplicată în RAM dacă singurul owner valid este deja VRAM și
  politica permite eliberarea host copy.

### `scheduler/`

Schedulerul menține:

- request state și poziția curentă în model;
- KV slot și deadline/cancellation state;
- work queues indexate după `(layer, expert)`;
- ready queue pentru VRAM, upload queue pentru RAM și load queue pentru SSD;
- o fereastră de microbatch cu limită de latență configurabilă;
- fairness, astfel încât throughput-ul să nu înfometeze request-urile vechi.

Algoritmul per strat:

1. dense/attention/router rulează pe GPU;
2. top-k exact generează work items;
3. work items sunt grupate după expert peste request-urile disponibile;
4. experții VRAM-ready rulează imediat prin grouped GEMM;
5. pentru RAM-ready începe `cudaMemcpyAsync`;
6. pentru ABSENT începe I/O în buffer pinned;
7. între timp sunt servite alte grupuri/request-uri gata;
8. după terminarea tuturor top-k pentru un token, rezultatele sunt ponderate și
   agregate în ordinea numerică definită de ABI;
9. request-ul avansează la stratul următor.

Reordonarea între request-uri este permisă. Aproximarea routerului, drop-ul unui
expert sau schimbarea top-k nu sunt permise în modul exact.

### `cuda/`

Backend-ul inițial țintește RTX 3090, SM86:

- dense/attention/router/lm_head rezidente și executate pe GPU;
- kernel cuantizat fused gate+up pentru toate rândurile grupate ale expertului;
- activarea de intrare este cuantizată o singură dată pentru gate+up;
- SiLU și produsul gate×up sunt fuzionate sau păstrate device-local;
- grouped down projection și weighted accumulation device-local;
- streams separate pentru compute, H2D și eventual preload;
- CUDA events pentru dependențe și eviction sigur;
- buffers și graph shapes prealocate pentru batch sizes acceptate;
- niciun round-trip GPU->CPU între router și agregarea MoE;
- kernelul expune o ABI stabilă pentru quant type/group size/layout;
- o cale de referință precisă validează kernelul, dar nu intră în hot path.

Reducerile structurale urmărite față de prototipul OLMoE sunt verificabile:

- 384 apeluri expert `matmul_q`/token devin cel mult 32 dispatch-uri grouped
  gate+up/down pentru 16 straturi;
- activarea per expert este cuantizată de două ori în loc de trei ori;
- gate/up nu mai pornesc două echipe/dispatch-uri separate;
- experții identici selectați de mai multe request-uri folosesc o singură
  încărcare și o operație batched;
- weights rămân în layout compute-ready de la SSD până la kernel.

Nu atribuim anticipat un multiplicator tok/s acestor reduceri. Gate-ul este SLO-ul
end-to-end, iar metricile de fază explică unde rămâne timpul.

### `dense/`

- încarcă tensorii dense conform preciziei declarate în manifest;
- implementează embeddings, norme, attention/RoPE, router, residual și lm_head;
- KV cache prealocat și bugetat per slot;
- paged/slot KV pentru request-uri concurente;
- atenția și dense compute rămân pe GPU în v1;
- context overflow este refuzat sau gestionat printr-o politică explicită, nu
  prin reallocation necontrolat.

### `server/`

Procesul este persistent și oferă:

- endpoint OpenAI-compatible pentru chat/completions;
- streaming incremental;
- mai multe sloturi KV și continuous batching;
- admission control bazat pe VRAM/RAM/KV/deadline;
- coadă limitată, backpressure și răspuns explicit la overload;
- request cancellation care eliberează work items și references;
- timeout separat pentru queue, cold I/O și generation;
- health, readiness și model-info;
- graceful shutdown și drain;
- configurare versionată, fără tunables ascunse în variabile globale;
- crash log, ultimul model manifest/hash și motivul opririi;
- niciun download sau update de model implicit la startup.

### `telemetry/`

Metricile minime per proces și per request:

- TTFT, inter-token latency și tok/s single/aggregate;
- queue/admission/batch wait;
- timp dense, router, expert compute, H2D și SSD wait;
- batch rows și experți unici per strat;
- hit VRAM, hit RAM, miss SSD și reuse per expert;
- requested/useful/read/uploaded bytes per token;
- overfetch, load coalescing și load deduplication;
- VRAM/RAM/staging/KV high-water marks;
- eviction, stalled-by-budget și cancellation;
- erori checksum/I/O/CUDA și fallback-uri;
- model/config/build identifiers.

Tracing-ul detaliat trebuie să poată fi oprit. Contoarele de bază rămân ieftine
și active în production.

## 3. Fezabilitate și admission

Înainte de startup, plannerul combină manifestul cu inventarul hardware:

```text
resident_dense_bytes
vram_expert_budget
ram_expert_budget
ssd_bandwidth
pcie_bandwidth
active_expert_bytes_per_token
expected_concurrency/reuse
```

El produce:

- dacă dense + KV + minimum workspace încap în VRAM;
- numărul de experți care încap în VRAM și RAM;
- cold bytes/token maxim permis de fiecare SLO;
- hit/reuse minim necesar;
- concurența minimă necesară pentru amortizare;
- un plan `feasible`, `degraded` sau `impossible` cu explicație numerică.

`impossible` nu pornește cu promisiunea SLO. Operatorul poate porni explicit în
mod best-effort, iar API-ul și metricile trebuie să reflecte acest lucru.

## 4. Etapele de implementare

### P0 — contract și skeleton

Livrabile:

- `docs/qmoe-format-v1.md` cu schema binară și manifestul;
- `docs/qmoe-runtime.md` cu state machines și ABI CUDA;
- directoarele `compiler/` și `runtime/` cu build reproducibil;
- inventar hardware extins cu CUDA, VRAM, RAM, disk și PCIe;
- calculatorul de fezabilitate folosit atât de compiler, cât și de runtime.

Poarta P0:

- același manifest produce aceeași decizie și aceleași hash-uri;
- un model imposibil este refuzat cu formula și resursa limitativă;
- nu există dependență de codul Colibri în noul skeleton.

### P1 — compiler și container QMOE

Livrabile:

- primul adaptor SafeTensors pentru o familie MoE aleasă explicit;
- quant profile v1;
- writer atomic pentru dense/expert packs;
- gate+up fused layout;
- resume, checksums și validator independent;
- raport de conversie și manifest complet.

Poarta P1:

- toate tensor-ele sunt identificate fără ambiguitate;
- suma byte-ilor/indexului/checksum-urilor corespunde pack-urilor;
- fiecare expert poate fi încărcat independent și validat;
- containerul incomplet nu poate fi deschis ca model valid.

### P2 — storage și cache core

Livrabile:

- reader IOCP/direct;
- pool pinned fix;
- load deduplication;
- cache state machine RAM/VRAM;
- byte budgets, eviction sigur și backpressure;
- telemetry pentru fiecare tranziție și byte.

Poarta P2:

- zero double-load pentru aceeași cheie concurentă;
- niciun use-after-evict sau slot vizibil înainte de completion;
- memoria rămâne sub buget în churn și cancellation;
- short-read/checksum/CUDA failure închid request-ul corect, fără weights corupte.

### P3 — execuție exactă single-request

Livrabile:

- dense path complet pe GPU;
- router/top-k exact;
- fused grouped gate+up/down;
- weighted accumulation și KV persistent;
- un request poate traversa integral modelul din QMOE.

Poarta P3:

- logits/hidden și tokenii respectă toleranța declarată față de referința
  aceluiași model și quant profile;
- nu există I/O sau copie necontorizată;
- minimum 10 tok/s single-stream este evaluat numai după ce working set-ul este
  hot; dacă nu trece, metricile de fază indică componenta care consumă timpul.

OLMoE poate rămâne fixture de corectitudine ieftin, dar nu este ținta de produs
și nu lansăm un proiect separat de comparare a runtime-urilor.

### P4 — scheduler global și continuous batching

Livrabile:

- request slots și microbatch window;
- grouping `(layer, expert)` între request-uri;
- ready-first scheduling și overlap SSD/H2D/GPU;
- fairness, deadlines, cancellation și admission;
- aggregate telemetry și reuse accounting.

Poarta P4:

- minimum 30 tok/s aggregate la concurența declarată pe configurația fezabilă;
- rezultatul fiecărui request este independent de interleaving;
- un request rece nu blochează global request-urile ready;
- p95 și fairness sunt raportate împreună cu throughput-ul.

### P5 — serviciu production

Livrabile:

- API OpenAI-compatible și streaming;
- health/readiness/model-info;
- backpressure, overload, timeouts și graceful drain;
- configurare, logs și metric export;
- packaging și lansare pe 3090box;
- procedură documentată de conversion, deploy, rollback și recovery.

Poarta P5:

- restart-ul nu corupe containerul/cache-ul;
- request cancellation și overload nu depășesc bugetele;
- serviciul poate fi operat fără shell interactiv;
- toate build/model/config identifiers sunt vizibile într-un incident.

### P6 — demonstrația modelului mai mare decât RAM

Modelul este ales după manifest și criterii, nu după numele sau popularitatea
checkpoint-ului:

- container QMOE mai mare decât RAM-ul fizic al mașinii;
- dense + KV + workspace încap în VRAM;
- masa activă și top-k sunt compatibile cu SLO-ul;
- licența și tokenizer/config sunt complete;
- formatul sursă poate fi compilat fără euristici.

Poarta finală:

- modelul complet nu este rezident în RAM și sistemul nu folosește swap;
- cel puțin 10 tok/s single-stream hot;
- cel puțin 30 tok/s aggregate la concurența declarată;
- cold miss bytes/token, hit/reuse, TTFT și p95 sunt publicate;
- output-ul păstrează semantica modelului cuantizat;
- procesul respectă bugetele și rămâne disponibil sub churn.

## 5. Ce nu facem în această ramură

- nu facem un fork-feature race cu Colibri sau llama.cpp;
- nu facem benchmark în trei runtime-uri ca obiectiv de proiect;
- nu continuăm micro-codebook, neuron pruning sau HESR-AMC respins;
- nu schimbăm routerul/top-k pentru a fabrica hit-rate;
- nu promitem că prefetch-ul depășește limita de bandă;
- nu introducem route prediction înainte de traces care arată un beneficiu net;
- nu optimizăm CPU-only pentru ținta de 10–30 tok/s; RTX 3090 este backend-ul
  compute inițial;
- nu descărcăm încă un model mare înainte ca P0 să poată demonstra fezabilitatea;
- nu declarăm production pe baza unui prompt scurt sau doar a mediei tok/s.

## 6. Ordinea exactă pentru următoarea sesiune

1. recitim acest plan și `RESULTS.md`; nu redeschidem experimentele închise;
2. creăm specificația `qmoe-format-v1` și schema manifestului;
3. fixăm ABI-ul recordului expert: header, alignment, fused gate+up, down, scales,
   checksum și quant profile;
4. creăm skeleton-ul `compiler/` și `runtime/` cu CMake/build Windows;
5. implementăm calculatorul de fezabilitate și extindem inventarul hardware;
6. inventariem checkpoint-urile locale numai prin config/headere și selectăm
   primul adaptor după criteriile P1;
7. implementăm writer/index/checksum/resume pentru QMOE;
8. abia după un container valid începem `storage/` IOCP și cache state machine;
9. CUDA grouped MoE și schedulerul vin după ce ABI-ul și lifecycle-ul experților
   sunt stabile;
10. serverul vine după execuția exactă și bugetată, nu înainte.

Deciziile deja fixate pentru sesiunea următoare sunt: runtime propriu,
Windows+CUDA/RTX3090 prima platformă, QMOE compute-ready, semantică exactă,
cache global în bytes, I/O asincron, grouped MoE cross-request și production
admission/backpressure. Alegerea primului model mare și profilul final INT4/INT8
rămân deschise până când calculatorul P0 le poate valida numeric.
