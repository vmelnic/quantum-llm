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

Stare: **SINGLE PASS; CORECTITUDINE BATCH PASS; THROUGHPUT AGGREGATE RĂMÂNE
DESCHIS**.

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

Semantica a fost auditată și contra implementării Transformers `5.14.1`, fișier
`modeling_qwen3_next.py` cu SHA-256
`bc6ee64d65d9b42c021c2f5bef7a79a2823fc3fd511db7a0335b5e9bf8a088b4`.
Au fost verificate explicit: layout-ul query/output-gate, RMSNorm
zero-centered, partial RoPE, GQA scaling, ordinea Conv1D, Q/K L2 plus
`1/sqrt(head_dim)`, `exp(-exp(A_log)*softplus(a+dt_bias))`, update-ul recurent,
gated RMSNorm, shared expert și renormalizarea top-k. Nu s-a găsit o abatere de
formulă în decode-ul one-token.

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
După adăugarea ferestrei de latență, un al doilea smoke persistent cu trei
tokenuri a publicat TTFT `0,344 s`, inter-token p95 `0,031 s`, trei batch-uri și
trei rânduri. Aceste valori validează exportul metricilor pe OLMoE și nu sunt
folosite drept rezultat P6 pentru Qwen3-Next.
Serviciul oprește acum request-ul la tokenul EOS, anulează pasul deja pregătit
și raportează `finish_reason=stop` plus numărul real de tokenuri, nu limita
cerută de client; comportamentul de cleanup este acoperit de testul front-end.

Runnerul Qwen publică acum, pentru intervalul măsurat după warmup, timpul de
wall al forward-urilor și descompunerea fără sincronizări CUDA artificiale în
`dense_attention_router_seconds`, `expert_cache_wait_seconds`,
`expert_compute_seconds` și `unattributed_seconds`. Astfel un gate ratat poate
fi atribuit numeric înainte de orice optimizare. Citirea și transferul H2D al
`dense.qpack` sunt publicate separat la startup; byte-ii read/H2D și hit/miss ai
experților rămân contorizați de cache.

Deploy-ul P6 are profil Task Scheduler separat (`QuantumLLM-P6ExpertServer`) și
un smoke operațional canonic. Acesta verifică identity/build/hash-urile reale
din `/model-info`, health/readiness, patru completări concurente, SSE `[DONE]`,
cancellation prin client disconnect și delta contoarelor de batching/TTFT/p95.
Instalarea task-ului și smoke-ul real rămân după container și gate; pe 3090box
nu exista niciun task `QuantumLLM*` la verificarea curentă.

Download-ul Xet s-a încheiat autoritativ cu `41/41` shard-uri, zero transferuri
incomplete și exit code `0`; checkpoint-ul sursă ocupă `162.682.287.693` bytes
în cache. Conversia reală a fost făcută fără `--reclaim-source-shards`, după
curățarea celorlalte cache-uri Hugging Face. Sursa Qwen rămâne integrală pentru
repack, alături de fixture-ul OLMoE.

Containerul publicat atomic în
`work/models/qwen3-next-80b-expert-pack-int8` are:

- `20` pack-uri: un `dense.qpack` și `19` pack-uri de experți;
- `81.903.198.208` bytes în total, dintre care `4.191.133.696` dense și
  `77.712.064.512` experți;
- `2.216` tensori dense și `24.576` records de experți;
- `1.517.813.760` bytes activi per token pentru top-10 înainte de cache reuse;
- manifest content SHA-256
  `55cc761ae66f294f2ad423421cb4d87462aaafea9ccae0c4b89efdaab04656e3`;
- source checkpoint SHA-256
  `5557a28a802a5195953c9f63e15f9117b58c4e49ec56ec711e94e7b699d77a13`.

Conversia a durat `3.371,421 s`. Validatorul lansat separat de conversie a
recitit containerul complet și a raportat `valid=true`, aceleași `20` pack-uri,
`24.576` experți și același manifest hash. Astfel poarta P1 pentru containerul
real este închisă; următoarea dovadă este gate-ul de inferență P6.

Compilerul păstrează și modul distructiv explicit, oprit implicit. El poate
reclama numai shard-uri confirmate ca fiind consumate, dar nu a fost folosit la
această conversie și nu va fi aplicat checkpoint-ului Qwen păstrat pentru
repack.

### Gate-urile Qwen reale și revizia de arhitectură

Gate-ul single canonic, cu cache expert de 17 GiB, a produs `13,6968 tok/s`,
p95 `77,5434 ms`, zero SSD/H2D după warmup și zero evictions. Working set-ul
observat a fost `17.442.209.792` bytes RAM și `17.419.628.544` bytes VRAM. Modelul
de `81.903.198.208` bytes rămâne mai mare decât cei `68.641.103.872` bytes RAM
fizici, iar pagefile-ul nu a crescut.

Gate-ul mixt cu patru prompturi diferite a păstrat exact output-ul fiecărui
request față de execuția izolată (`interleaving_match=true`), dar a produs numai
`5,23099 tok/s` aggregate. Cauza măsurată nu a fost SSD-ul: zero miss-uri SSD,
dar un working set RAM de `35.440.951.296` bytes a concurat pentru aproximativ
18,25 GB VRAM, producând `21.080` evictions, `39.459.409.920` bytes H2D și
`12,598 s` cache wait.

Acest rezultat a invalidat două alegeri ale prototipului: LRU global și
dispatch-ul care copia expert IDs GPU->CPU și reconstruia patru tabele de
pointeri CPU->GPU la fiecare strat. Revizia implementată introduce:

- cote VRAM/RAM pe layer-group plus rezervă comună pentru burst;
- admission după frecvență cu aging, astfel încât traficul one-hit să nu
  evacueze automat working set-ul reutilizat;
- director GPU cu pointer, stare, generație și device refcount per expert;
- deduplicare prin hash dimensionat după selecțiile active, nu după numărul
  total de experți;
- hot path miss-only; lista completă ready este returnată CPU numai pe cold
  path, ca să fie lease-uită înaintea load-urilor;
- kernels MoE care consumă direct directorul device, fără pointer tables host.

Primul gate după director + partitionare, înainte de admission-ul LFU, a rămas
FAIL la `4,74454 tok/s`: `38.354.104.320` bytes H2D, `20.523` evictions și
`17,3081 s` cache wait. Acesta demonstrează că partitionarea singură nu este o
soluție și fixează baseline-ul pentru politica resident/transient. Rezultatul
nu este prezentat drept progres de viteză. Build-ul MSVC/NVCC, testele
deterministe și smoke-ul cu opt experți reali au trecut; smoke-ul directorului
a produs `cosine=1`, `max_abs=9,31323e-10`.

Admission-ul LFU cu aging a îmbunătățit gate-ul mixt la `5,37435 tok/s`, a
redus H2D la `31.289.622.528` bytes, evictions la `19.354` și cache wait la
`14,2896 s`, păstrând `interleaving_match=true`. Reducerea este reală, dar
insuficientă: scorul singur admitea în continuare fiecare miss în același pool.

O revizie separă fizic bugetul VRAM în resident și transient. Pentru profilul
de 17 GiB au fost testate 16 GiB resident + 1 GiB ring transient. Bugetele au
fost respectate exact (`17.179.611.136` și `1.073.741.824` bytes high-water),
dar gate-ul a regresat la `5,05507 tok/s`, `31.570.685.952` bytes H2D, `23.607`
evictions și `16,3525 s` cache wait. Corectitudinea a rămas PASS.

Mecanismul rămâne configurabil pentru modele la care izolarea cold traffic este
obligatorie, dar este dezactivat implicit pentru Qwen. Plannerul viitor trebuie
să-l activeze dintr-un model numeric de cost, nu dintr-un prag hard-coded.
Baseline-ul acceptat pentru continuarea compute-path este LFU-ul comun de
`5,37435 tok/s`.

### Step-back: costurile executorului heterogen

Stare: **CPU-LOCAL ALES PENTRU PRIMUL VERTICAL SLICE; SLAB H2D RĂMÂNE TIER
VALID**.

Inventarul relevant al 3090box este Ryzen 5 5600 6C/12T cu AVX2, 64 GiB în două
DIMM-uri DDR4-3200 și RTX 3090 conectat PCIe Gen3 x16. Alegerea nu se bazează pe
un al doilea GPU și nu cere schimbarea explicită la INT4.

`expert-path-probe` a folosit 64 de recorduri Qwen reale, în total 202.375.168
bytes, deci working set-ul probei depășește cache-urile CPU. Rezultatele sunt:

| Cale | Geometrie | Rezultat |
|---|---:|---:|
| RAM pageable -> slab pinned | 202,38 MB | 16,13–16,37 GiB/s |
| slab pinned -> RTX 3090 | 202,38 MB | 12,46 GiB/s |
| CPU AVX2 grouped | 1 row/expert, 6 threads | 6.203 experți/s; 18,27 GiB/s efectiv |
| CPU AVX2 grouped | 4 rows/expert, 6 threads | 5.059 experți/s; 14,90 GiB/s efectiv |

Kernelul CPU păstrează weights expert-major și le reutilizează între rânduri.
Față de referința scalară a aceluiași record a produs cosine `1` și max-abs
`2,32831e-9` pentru ambele bucket-uri.

În gate-ul LFU, cele 9.590 promovări RAM->VRAM pentru 124 tokeni înseamnă circa
77,34 grupuri expert miss/token. La 30 tok/s ar cere aproximativ 2.320 grupuri/s,
sub cele 5.059 grupuri/s măsurate chiar la patru rânduri. Aceasta nu dovedește
încă SLO-ul end-to-end: latența per strat, copiile activation/result și
concurența cu GPU-ul trebuie integrate și măsurate. Dovedește însă că pe această
mașină CPU-local este candidatul corect pentru primul vertical slice și că nu
trebuie să copiem automat fiecare RAM hit în VRAM.

Slab H2D rămâne important pentru alte distribuții și alte CPU-uri. Pentru
payload-ul observat de aproximativ 252 MB/token, link-ul măsurat are bandwidth
brut suficient pentru 30 tok/s, dar pack-ul în pinned staging consumă încă o
trecere prin RAM și trebuie suprapus prin double buffering. Nu îl implementăm
înaintea CPU-local pe 3090box, dar contractul executorului îl păstrează.

Separat, eliminarea cache wait-ului nu rezolvă P4. Baseline-ul LFU are aproximativ
11,13 s non-I/O pentru 124 tokeni, adică un plafon de circa 11,14 tok/s aggregate.
Kernelurile GPU grouped/persistent rămân obligatorii după verticala CPU-local.

### P6 final — executor heterogen și placement pe epoci

Verticala finală nu promovează obligatoriu fiecare RAM hit. Selecțiile
VRAM-ready rulează prin directorul CUDA, iar selecțiile RAM-ready folosesc
thread pool-ul AVX2 și întorc ieșiri per selecție pentru agregare stabilă pe
GPU. Plannerul generic urmărește costul CPU, reuse debt și frecvența tuturor
rutelor. Promovările RAM->VRAM sunt asincrone și admise numai dacă pot înlocui
rezidenți strict mai reci.

Rebalansarea se încheie la o barieră explicită, după care placement-ul rămâne
frozen pe durata decode-ului. Încercarea de a promova sincron în hot path a fost
respinsă: `14,5708 tok/s`, `4.973.875.200` bytes H2D și `1.575` promovări în
fereastra măsurată. Varianta asincronă fără epoch freeze a fost de asemenea
respinsă: `22,5733 tok/s`, `3.818.041.344` bytes H2D și `1.209` promovări.
Aceste rezultate rămân evidence pentru motivul barierei, nu configurații
acceptate.

Configurația finală folosește 48 GiB cache RAM și 18 GiB cache VRAM pe RTX 3090.
Artefactul `p6-gate-single-18g-pass.json` raportează:

- `31,7312 tok/s` single-stream, peste pragul de 10;
- correctness PASS, zero creștere pagefile;
- zero selecții CPU, zero promovări și zero H2D weights după warmup.

Artefactul `p6-gate-batch-18g-pass.json` raportează:

- `31,1004 tok/s` aggregate la patru prompturi diferite, peste pragul de 30;
- `interleaving_match=true` și correctness PASS;
- `3,98709 s` pentru 124 forward tokens;
- hit-rate VRAM `0,853185`;
- `9.847` selecții și `8.617` grupuri executate CPU în `1,10878 s`;
- zero promovări și zero H2D weights în fereastra măsurată;
- high-water `19.327.082.496` bytes cache VRAM și `35.440.951.296` bytes cache RAM;
- minimum `26.729.181.184` bytes RAM fizic liber și zero creștere pagefile.

Containerul de `81.903.198.208` bytes este mai mare decât cei
`68.641.103.872` bytes RAM fizici, iar ambele SLO-uri P6 sunt astfel închise
fără al doilea GPU și fără a șterge checkpoint-ul sursă.

### Deploy persistent P6

Task Scheduler rulează profilul `QuantumLLM-P6ExpertServer` pe loopback cu
build ID `2383494`, worker protocol 2, capacitate 4, cache RAM 48 GiB și cache
VRAM 18 GiB. `/model-info` publică manifest hash
`55cc761ae66f294f2ad423421cb4d87462aaafea9ccae0c4b89efdaab04656e3` și
experts index hash
`45c98005ae0897bbadfd8f9315c003f33c03a16a80b0a3cf6d66aa0551ca7bab`.

Primul smoke a găsit că disconnect-ul putea încăpea complet în send buffer și
nu producea `BrokenPipeError`; serverul a fost corectat să detecteze FIN/RST
prin peek non-blocant înaintea următorului decode și să închidă generatorul,
trimițând `END` workerului. Smoke-ul final a trecut cu:

- `completed_delta=5` pentru patru completări concurente plus streaming;
- `decode_batches_delta=10`, `decode_rows_delta=21`, batch efectiv `2,1`;
- SSE terminat cu `[DONE]` și `cancellation_observed=true`;
- model/build/container identity corecte;
- TTFT p95 `15,109 s` și inter-token p95 `2,563 s` pe smoke-ul operațional
  rece; acestea nu înlocuiesc gate-urile hot de throughput.

Installerul oprește acum arborele de procese al aceluiași repo/port înainte de
reînregistrare; aceasta previne ca un worker vechi să păstreze portul și VRAM la
redeploy sau rollback.
