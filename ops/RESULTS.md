# Rezultate experimentale

## Expert Pack P0–P2 — fundația runtime-ului propriu

Stare: **PASS PE 3090BOX, WINDOWS/CUDA TOOLCHAIN**.

Commitul `7473cf2` introduce contractul Expert Pack v1, calculatorul de
fezabilitate, compilerul/validatorul, cache-ul bugetat și backend-ul IOCP.
Build-ul Release cu Visual Studio 2022 și CUDA 12.1 a trecut 3/3 teste CTest,
iar compilerul a trecut 5/5 teste Python. Build-ul include pool-ul CUDA pinned
și compilează backend-ul IOCP nativ Windows.

Checkpoint-ul real `allenai/OLMoE-1B-7B-0125-Instruct` a fost apoi convertit,
nu doar fixture-ul sintetic:

- 3.219 tensori sursă și 13.838.721.960 bytes;
- 147 records dense și 1.024 records expert;
- 6.948.352.000 bytes Expert Pack INT8 în trei pack-uri;
- 484.929.536 bytes dense și 6.463.422.464 bytes experți;
- 807.927.808 bytes experți activi/token;
- conversie în 136,75 s cu NumPy pe 3090box;
- validator independent: 3/3 pack-uri, toate records și hash-urile valide;
- manifest content SHA-256:
  `82d0a6fa20fd440a1113cd0f89399c710c36c8a5c4d223d74971ec52276e3427`.

Containerul este în
`work/models/olmoe-expert-pack-int8` pe 3090box și devine intrarea P3 CUDA.
Conversia reală a corectat o presupunere din fixture: tensorii OLMoE `q_norm`
și `k_norm` au shape `[hidden_size]`, nu `[head_dim]`.
Manifestul canonic rezultat declară și `clip_qkv: null`; prima conversie,
anterioară acestui câmp semantic, a fost păstrată recuperabil sub numele
`olmoe-expert-pack-int8-pre-clip-contract` și nu este folosită de P3.

### P3.0 — fused CUDA MoE pe records reale

Stare: **PASS COMPONENTĂ, NU ÎNCĂ INFERENȚĂ END-TO-END**.

Backend-ul SM86 execută un strat single-token/top-8 în două dispatch-uri:
`gate+up+SiLU` grouped și `down+weighted accumulation` în ordine stabilă.
Smoke-ul a citit și validat primii opt experți reali din `experts-000.qpack`,
apoi a comparat RTX 3090 cu referința CPU a aceluiași quant ABI:

- hidden 2.048, intermediate 1.024, top-k 8;
- eroare absolută maximă `1,16415e-9`;
- eroare relativă maximă `7,29039e-5`;
- cosine `1,0`;
- `0,199588 ms` MoE hot per strat, media a 100 iterații după warmup;
- plafon izolat MoE pentru 16 straturi: `313,145 tok/s`.

Ultima valoare nu este raportată ca viteză de model: exclude dense, attention,
router, KV, lm_head și orice miss/upload. Ea demonstrează că kernelul MoE hot
nu consumă bugetul de 100 ms/token al porții single-stream de 10 tok/s.

### P3 — inferență exactă end-to-end din Expert Pack

Stare: **PASS PE 3090BOX, PROFIL INT8 EXPERT PACK**.

Runtime-ul CUDA traversează integral modelul: embedding, RMSNorm, Q/K/V,
normalizarea Q/K OLMoE pe vectorul proiectat complet, RoPE, KV persistent,
attention, router softmax/top-8, experții fused, residual, norma finală,
`lm_head` și argmax. Routerul și selecția experților rămân device-local.

Corectitudinea a fost verificată cu un oracle NumPy independent care citește
aceiași bytes Expert Pack, îi de-cuantizează conform ABI-ului și implementează
separat forward-ul. Pentru promptul canonic, CUDA și oracle-ul au produs exact
aceiași 12 tokeni:

```text
510,5347,273,6181,310,7785,15,187,187,510,3565,14731,273,6181,310,253,14029
```

Top-5 logits au coincis la precizia afișată, cu diferențe numai în ultimele
zecimale. Secvența BF16/Colibri anterioară nu este oracle pentru acest profil:
Colibri cuantizează numai experții, iar Expert Pack v1 cuantizează și matricile
dense. Cele două trasee coincid în primele șase tokenuri generate; apoi Expert
Pack alege `14731` cu logit `16,0087`, iar tokenul BF16 `3448` este al doilea cu
`15,9676` (marjă `0,0411`).

Măsurarea canonică Release, fără tracing, verifică SHA-256 al fiecărui pack la
startup și raportează separat startup/prompt/decode:

- 6.948.352.000 bytes citiți, verificați și copiați H2D la startup;
- `26,8875 s` model load, inclusiv SHA-256 și H2D;
- `0` bytes storage și `0` bytes H2D în decode hot;
- 5 tokeni prompt în `0,243322 s`;
- 11 forward-uri decode în `0,194159 s`;
- **56,6547 tok/s single-stream hot**.

Numărul de forward-uri, nu numărul de tokeni afișați, este folosit la calculul
tok/s. Rezultatul trece poarta de 10 tok/s fără să ascundă costul de startup.
Oracle-ul reproductibil este `ops/python/run_expert_pack_oracle.py`, iar
executabilul este `expert-olmoe-runner`.

### P4 — scheduler global și continuous microbatch

Stare: **PASS HOT PATH PE 3090BOX**.

Schedulerul runtime are admission și cozi limitate, fairness determinist după
deadline/vârstă, grupare `(layer, expert)`, ready-first, completare exactă și
cancellation. În scenariul determinist, un expert absent rămâne blocat fără să
oprească grupurile VRAM-ready ale altui request; după schimbarea residency,
work-ul rece este reluat și tokenul se finalizează. Reuse și toate stările de
coadă sunt contorizate.

Backend-ul CUDA acceptă acum mai multe rows/request slots cu KV separat.
Selecțiile tuturor request-urilor folosesc două dispatch-uri MoE per strat
(`gate+up` și `down+ordered accumulation`), nu câte două per request. Dense și
router rulează per slot, iar expert IDs/routing weights rămân pe device.

Măsurarea Release cu 4 request-uri, câte 24 tokeni generați, a produs:

- 92 forward rows decode în `1,36418 s`;
- **67,4398 tok/s aggregate**;
- inter-token p50 `57,6183 ms`, p95 `66,8937 ms`;
- fairness token skew `0`;
- output identic pentru cele patru request-uri identice.

O verificare separată a folosit patru prompturi diferite. Fiecare a fost rulat
mai întâi izolat, apoi toate patru intercalat în același microbatch/KV pool.
Toate cele patru secvențe batched au fost identice cu baseline-ul lor izolat
(`interleaving_match=true`); throughput-ul acelui run scurt a fost
`76,8394 tok/s`, p95 `55,2542 ms`, skew `0`.

Acest rezultat califică hot path-ul și depășește poarta de 30 tok/s aggregate.
Overlap-ul real SSD/IOCP→pinned RAM→H2D rămâne de exercitat de P6 cu un model
care nu încape în RAM/VRAM; OLMoE este complet VRAM-resident și ar fabrica un
test de cold I/O dacă l-am evacua artificial.

### P5 — serviciu persistent OpenAI-compatible

Stare: **PASS VERTICAL SLICE OPERAȚIONAL PE 3090BOX**.

Procesul CUDA are protocol persistent `BEGIN/NEXT/END/SHUTDOWN`; smoke-ul a
încărcat containerul o singură dată, a returnat tokenii canonici `7785, 15` și
s-a oprit curat. Frontend-ul Python folosește tokenizerul local și oferă:

- `/v1/completions` și `/v1/chat/completions`;
- SSE incremental, inclusiv chunk final cu `finish_reason` și `[DONE]`;
- `/health`, `/ready`, `/model-info`, `/v1/models`, `/metrics`;
- queue/admission limitate, timeout-uri, anulare la disconnect și drain;
- API key pentru bind non-loopback, logs JSONL și identificatori build/model;
- script foreground și Task Scheduler install/uninstall pentru operare fără
  shell interactiv.

Probe reale:

- completion pentru `The capital of France is` a returnat ` Paris.\n\n`;
- același request streaming a livrat deltele ` Paris`, `.`, newline;
- chat completions a traversat template-ul tokenizerului și a răspuns HTTP 200;
- la capacitate `active + queue = 3`, patru request-uri concurente de câte 100
  tokeni au produs trei răspunsuri HTTP 200 și un HTTP 503 explicit
  `overload_error`;
- după request, metricile au raportat `active=0`, `admitted=1`, `completed=1`,
  `generated_tokens=2`, zero failed/cancelled, readiness `1`;
- `SIGINT` a făcut drain și workerul a ieșit cu cod 0;
- logul `logs/expert-server.jsonl` a fost creat și actualizat.

Runnerul verifică la fiecare startup dimensiunea și SHA-256 pentru toate cele
6.948.352.000 bytes de pack. Un container corupt nu ajunge în readiness.

## Etapa B — analizor SafeTensors

Stare: **PASS**.

Checkpoint: `ibm-granite/granite-3.1-3b-a800m-base`, revizia
`d4dd87aa3a6c201bc374851d7d7ff4cf39a0b82a`.

- 6.597.577.728 bytes de tensori BF16;
- 290 tensori în două shard-uri, validați integral din headere și index;
- 32 straturi, 40 experți, top-8;
- 6.039.797.760 bytes în ponderile experților;
- 1.207.959.552 bytes de experți activi per token;
- 1.765.739.520 bytes de ponderi totale active per token;
- fracție activă față de modelul complet: 26,76%.

Artefact: `artifacts/model-analysis-latest.json`.

## Etapa C1 — bază comună + low-rank + sparse post-hoc

Stare: **NO-GO pentru această reprezentare directă**.

Stratul 0 conține 188.743.680 bytes BF16 de experți. Testul a inclus:

- baze comune cu 1, 2, 4 și 8 clustere;
- matching neuronal în două treceri pe tripletele gate/up/down;
- reziduuri SVD cu ranguri până la 64;
- corecție sparse cu costul indicilor inclus;
- șase prompturi, 115 tokeni și 920 apeluri rutate, 34/40 experți atinși;
- execuție exclusiv CPU.

Rezultate activation-weighted:

| Compresie | Rang | Sparse | Eroare normă ieșire MoE | Cosinus mediu | Cosinus p05 |
|---:|---:|---:|---:|---:|---:|
| 6,20x | 16 | 1% | 0,875 | 0,549 | 0,321 |
| 4,44x | 64 | 0% | 0,772 | 0,660 | 0,328 |

Matching-ul neuronal este funcțional exact: eroarea relativă a normei este
aproximativ `2,25e-7`, iar cosinusul este 1,0. Prin urmare, eroarea mare provine
din aproximarea ponderilor, nu dintr-o permutare implementată greșit.

Verdict: nu scalăm această variantă la opt straturi. Ea nu satisface poarta de
minimum 4x la calitate acceptabilă nici măcar pe un singur strat.

Artefacte:

- `artifacts/factorization-aligned-layer-0-latest.json`;
- `artifacts/activation-eval-layer-0-latest.json`.

## Decizia după Etapa C1

Ramura următoare a fost mutată de la aproximarea ponderilor la reprezentări
învățate pe activări, cu split train/validation, buget de maximum 25% din masa
activă și fallback exact. Rezultatele ei sunt consemnate în Etapa D1: cele două
reprezentări implementate au fost respinse, nu extinse artificial la mai multe
straturi.

No-go-ul C1 nu respinge în principiu AMC sau distilarea funcțională. Respinge
extrapolarea conform căreia experții pot fi comprimați post-hoc prin simpla
medie a matricelor și reziduuri SVD/sparse.

## Etapa D0 — baseline Colibri

Stare: **PASS PE 3090BOX, CPU-ONLY**.

Baseline-ul folosește `JustVugg/colibri` la commit-ul
`b085b48888a88d9a1c00b151a9979774b72cdbfd`, cu un patch minim care numără
încărcările și byte-ii experților, inclusiv prefetch, și adaugă `DIRECT=1`
pentru ponderile experților.

Modelul sursă este `allenai/OLMoE-1B-7B-0125-Instruct`, revizia
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`. Snapshot-ul conține 13.838.323.712
bytes de tensori BF16. Containerul Colibri rezultat are 7.416.456.383 bytes pe
disc și 7.412.649.984 bytes de payload în 35 shard-uri:

- 16 straturi, 64 experți/strat, top-8;
- 1.024/1.024 experți validați structural;
- 6.307.840 bytes payload per expert;
- aproximativ 807 MB payload expert eligibil per token înainte de cache hits;
- 992 experți pot fi citiți integral prin Windows `NO_BUFFERING`;
- 32 experți aflați la capătul shard-urilor folosesc fallback buffered pentru a
  nu extinde citirea aliniată peste EOF.

Oracle-ul independent Transformers BF16, rulat pe CPU din snapshot-ul original,
a generat aceeași secvență de 12 tokeni ca referința upstream. Smoke-ul Colibri
INT8 a reprodus exact 12/12 tokeni față de acest oracle.

Sweep-ul are o singură repetare per punct, 6 thread-uri, 12 tokeni generați,
`DIRECT=1`, hot pinning oprit și LRU Colibri gol la începutul fiecărui proces:

| Cache/strat | Pilot | Exact | Hit rate | Payload citit | Viteză | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 12/12 | 0,3% | 12,88 GB | 0,40 tok/s | 1,89 GB |
| 8 | 0 | 12/12 | 18,2% | 10,57 GB | 0,48 tok/s | 2,55 GB |
| 64 | 0 | 12/12 | 69,2% | 3,97 GB | 1,10 tok/s | 5,50 GB |
| 1 | 1 | 12/12 | 0,5% | 13,28 GB | 0,40 tok/s | 1,89 GB |
| 8 | 1 | 12/12 | 23,6% | 10,67 GB | 0,50 tok/s | 2,55 GB |
| 64 | 1 | 12/12 | 80,7% | 5,10 GB | 1,02 tok/s | 6,55 GB |

Fără pilot, cache 64 reduce payload-ul cu 69,1% și este de 2,75x mai rapid decât
cache 1, plătind 3,61 GB RSS în plus. Pe acest prompt scurt, pilotul nu produce
un câștig net: la cache 64 adaugă 413 încărcări speculative, crește payload-ul
cu 28,4% și scade viteza de la 1,10 la 1,02 tok/s. Acesta este un rezultat de
laborator, nu încă un benchmark stabil: sunt necesare prompturi mai lungi și
repetări pentru intervale de variație.

Artefacte principale:

- `artifacts/colibri-container-latest.json`;
- `artifacts/olmoe-oracle-latest.json`;
- `artifacts/colibri-benchmark-latest.json`;
- `artifacts/colibri-prepare-latest.json`.

### D0.1 — imagine exactă a experților în RAM

Stare: **ACCEPTAT PENTRU MODELELE CARE ÎNCAP ÎN RAM**.

`RAMCACHE=1` preîncarcă la startup toți cei 1.024 de experți INT8, apoi
resetează contoarele de I/O înaintea decodării. Nu este ramdisk: structurile
folosite direct de engine rămân rezidente în RAM, deci evităm încă o copie și
orice acces la filesystem pe calea critică.

Comparația exactă pe aceeași mașină, aceleași 6 thread-uri și aceeași secvență
de 12 tokeni:

| Mod | Preload experți | I/O experți în decode | Exact | Viteză decode | Peak RSS |
|---|---:|---:|---:|---:|---:|
| `RAMCACHE=0`, LRU 64 | — | 3,97 GB payload | 12/12 | 1,10 tok/s | 5,50 GB |
| `RAMCACHE=1` | 6,459 GB în 14,59 s | 0 bytes | 12/12 | 10,19 tok/s | 7,81 GB |

În RAM, engine-ul a raportat 2.048 hit-uri, zero miss-uri și zero încărcări de
experți în decode. Câștigul observat este de aproximativ **9,3x** față de cel
mai rapid punct SSD/LRU măsurat. Timpul wall al procesului scurt este 18,52 s
deoarece include preload-ul; într-o sesiune persistentă de chat acest cost se
plătește o singură dată.

Acesta devine traseul operațional implicit pe 3090box. `RAMCACHE=0` rămâne
flag-ul de comparație și fallback pentru modele mai mari decât RAM-ul disponibil.
Rezultatul relevant este în
`artifacts/colibri-benchmark-20260803-204824.json`.

## Etapa D1 — fast path aproximativ HESR-AMC

Stare: **NO-GO PENTRU CELE DOUĂ REPREZENTĂRI IMPLEMENTATE**.

Am testat pe stratul OLMoE 0 două fast path-uri antrenate pe activări, cu split
train/validation și fallback către expertul exact:

| Reprezentare | Reducere masă activă fast path | Eroare relativă validation | Cosinus mediu | Fallback validat |
|---|---:|---:|---:|---:|
| codebook 16×256, maximum 8 micro-experți | 4,00x | 1,099 | 0,581 | 100% |
| neuron bank width 252 + corector rank 32 | 4,02x | 0,580 | 0,768 | 100% |

Primul prototip de codebook 8×128 a părut să ofere 1,19x viteză, 6,9% fallback
și 73,6% acord top-1. Acel rezultat este **invalid ca poartă de calitate**:
normalizarea riscului era sub-calibrată și permitea activări nesigure. După
corectare, controlerul refuză corect toate ieșirile aproximative pe validation.

Nu extindem aceste ramuri la modelul complet și nu facem benchmark de runtime
pentru un fast path care cade integral în fallback. Artefactele canonice sunt:

- `artifacts/hesr-amc-compile-layer-0-latest.json`;
- `artifacts/amc-neuron-bank-compile-layer-0-latest.json`.

Concluzia actuală nu este „mai multe teste”, ci o separare de arhitectură:

1. când modelul cuantizat încape în RAM, folosim execuția exactă RAM-resident;
2. când nu încape, folosim LRU/SSD exact ca fallback măsurabil;
3. redeschidem aproximarea numai cu o reprezentare distilată mai puternică și
   o poartă validation fixată, nu prin scalarea variantelor respinse aici.

## P6 — fundația modelului mai mare decât RAM

Stare: **ÎN IMPLEMENTARE; compilerul și traseul cache→CUDA sunt validate**.

Ținta aleasă este `Qwen/Qwen3-Next-80B-A3B-Instruct`: 48 straturi, 512
experți/strat, top-10, hidden 2048 și expert width 512. Adaptorul strict
`qwen3_next` acoperă linear attention, full attention cu query/output gate,
shared expert și cei 1.553 de tensori MTP. Fixture-urile compilerului validează
inclusiv Conv1D rank-3 și conservarea explicită MTP.

Pe 3090box, primul vertical slice real a traversat:

```text
experts-000.qpack -> Windows IOCP -> pinned pool -> SHA-256/header validation
                  -> cache RAM/VRAM -> uploader CUDA -> fused MoE kernel
```

Pentru 8 experți OLMoE reali a raportat 50.495.488 bytes citiți,
50.462.720 bytes urcați, 8 load-uri și rezultat identic numeric cu referința
CPU (`cosine=1`, `max_abs=1.16415e-09`). Numai două sloturi pinned, în total
12.623.872 bytes high-water, au alimentat 50.495.488 bytes de cache RAM
pageable; aceasta validează reciclarea staging-ului. Kernelul MoE a măsurat
0,202168 ms/strat în această rulare.

Kernelurile Qwen3-Next au fost validate separat pe RTX 3090 contra calculelor
CPU pentru geometria reală: eroare maximă `2,38419e-7` la RMSNorm,
`7,45058e-9` la full attention GQA/output gate și `1,16415e-9` la Gated
DeltaNet recurent. Runnerul complet este construit: dense resident pe GPU,
router exact, shared expert, acquire numai pentru top-10, lease până la
completion și stări attention/delta persistente.

Traseul aggregate Qwen3-Next este de asemenea construit, dar încă nemăsurat pe
containerul real: proiecțiile INT8/F32 și routerul sunt batched, selecțiile
duplicate sunt încărcate o singură dată per strat, iar toate rândurile intră
într-un singur dispatch MoE. Smoke-ul RTX 3090 măsoară eroare maximă
`1,67638e-8` pentru GEMV batched și `2,98023e-8` pentru routerul batched.
Acestea sunt verificări de corectitudine a kernelurilor, nu rezultate tok/s.

Serviciul persistent nu mai serializează întregul request: protocolul Qwen v2
ține patru sloturi de stare izolate și `STEP` execută decode-ul concurent prin
același `forward_batch`. Batcherul front-end a trecut testul cu patru thread-uri
într-un singur worker step. Compatibilitatea protocolului OLMoE v1 a fost
revalidată end-to-end prin API: `/ready=true`, completarea canonică `Paris.` și
metrici `decode_batches=2`, `decode_rows=2` pentru două tokenuri.

Download-ul Qwen3-Next rulează prin Hugging Face Xet. Estimarea exactă a
containerului pentru geometria fixată este 81.749.057.536 bytes, peste cei
68.641.103.872 bytes RAM fizici. C: este singurul volum, deci sursa de
162.649.725.440 bytes și containerul nu încap simultan fără reclamare.
Preflight-ul confirmă că download-ul și conversia cu reclamare sunt fezabile,
dar conversia fără reclamare nu este.

Compilerul are acum un mod distructiv explicit, oprit implicit: după fsync-ul
fiecărui pack și al stării, jurnalizează hash-ul shard-ului consumat și elimină
numai shard-uri fără tensori viitori. Modul trece testul end-to-end, dar nu va
fi activat pe cache-ul real fără switch-ul operatorului; modelele existente nu
sunt șterse automat.
