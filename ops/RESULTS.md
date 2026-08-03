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
