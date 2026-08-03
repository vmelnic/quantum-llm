# Plan experimental HESR-AMC

## Alegerea modelelor

Folosim trei roluri distincte; un singur checkpoint nu poate răspunde corect la
toate întrebările experimentului.

1. `ibm-granite/granite-3.1-3b-a800m-base` BF16 este primul model de laborator.
   Are aproximativ 3,3B parametri totali, 800M activi, 40 experți și top-8. Testul
   post-hoc pe un strat a produs un no-go, documentat în `RESULTS.md`.
2. `allenai/OLMoE-1B-7B-0125-Instruct` BF16 este controlul cross-model. Are
   aproximativ 7B parametri totali, 1B activi, 64 experți și top-8, în circa
   13,84 GB. Revizia activă este fixată în `config/experiment.json`.
3. `deepseek-ai/DeepSeek-V2-Lite` BF16 este validarea secundară: aproximativ 16B
   parametri totali, 2,4B activi, 64 experți rutați și doi experți comuni per
   strat MoE. Îl descărcăm numai dacă primul model trece poarta de un strat.
4. `rdtand_Qwen3.6-35B-A3B-PrismaQuant-4.75bit-vllm`, deja în `slm-models`, este
   ținta locală pentru parserul de formate împachetate, simularea traficului și
   runtime. Nu îl folosim pentru a estima singur compresibilitatea BF16, deoarece
   este deja cuantizat mixt.

Qwen2.5-32B este confirmat dens și rămâne control pentru I/O. Snapshot-ul Xet al
Qwen3.6-27B-AWQ are momentan junction-uri inaccesibile; îl tratăm doar ca format
împachetat de control până când config-ul și tensor-ele pot fi inspectate.

## Etapa A — infrastructură și baseline fizic

Stare: bootstrap-ul, sync-ul și inventarul sunt implementate.

Următoarele măsurători rulează fără GPU:

- bandă RAM printr-un buffer mai mare decât cache-ul procesorului;
- citire SSD SATA secvențială și aleatorie la 4 KiB, 64 KiB, 1 MiB și 8 MiB;
- latență p50/p95/p99 și consum CPU;
- cold-cache și warm-cache raportate separat;
- verificare că niciun proces/container nu primește device GPU.

Artefact: `artifacts/io-baseline-*.json`.

## Etapa B — analizor SafeTensors, fără inferență

Analizorul citește header-ele și face mmap pe tensori; nu încarcă modelul complet.
El produce:

- maparea tensor -> shard, offset, dtype, shape și bytes;
- experții, straturile, routerele și experții comuni;
- bytes totali, bytes activi/token și distribuția dimensiunilor;
- paginarea ipotetică și traficul SSD pentru diferite hit-rate-uri;
- compatibilitatea checkpoint-ului cuantizat local cu extragerea per-expert.

Poarta B: toate tensor-ele MoE trebuie identificate fără euristici ambigue, iar
suma byte-ilor din manifest trebuie să corespundă fișierelor.

## Etapa C — un singur strat MoE

Pentru un strat timpuriu, unul median și unul târziu:

1. construim baseline-ul exact CPU pentru expert;
2. grupăm experții folosind statistici de ponderi și activări;
3. testăm `B_cluster + U_expert V_expert + S_expert` pentru grile de rang și
   sparsitate;
4. cuantizăm reziduurile doar după ce eroarea BF16 este cunoscută;
5. măsurăm bytes, FLOP, timp, eroare Frobenius, eroare pe ieșirea expertului și
   cosinusul ieșirilor;
6. publicăm curba Pareto, inclusiv costul de reconstrucție.

Calibrarea folosește mai multe domenii și rute, nu doar ponderi. Pragurile de
calitate sunt fixate după rularea baseline-ului exact, înainte de optimizarea
hiperparametrilor, pentru a evita alegerea retrospectivă a unei metrici favorabile.

Poarta C: trebuie să existe un punct Pareto care reduce cu cel puțin 2x bytes
pentru strat fără degradare semnificativă pe activările de calibrare. Dacă nu,
nu implementăm încă runtime-ul complet.

## Etapa D — ierarhia de execuție RAM/SSD

Implementarea este acum împărțită în niveluri explicite:

1. **Tier 0, exact RAM-resident**: `RAMCACHE=1` încarcă direct structurile INT8
   ale tuturor experților și elimină filesystem-ul din decode. Acesta este modul
   implicit când modelul încape în RAM.
2. **Tier 1, exact LRU/SSD**: `RAMCACHE=0` păstrează reader-ul Colibri, cache-ul
   per strat și fallback-ul măsurat. Acesta este traseul pentru modelele care nu
   încap integral în RAM.
3. **Tier 2, fast path aproximativ auditat**: este permis doar dacă o
   reprezentare trece validation și reduce masa activă; orice token riscant cade
   în Tier 0 sau Tier 1 fără schimbarea routerului exact.

Tier 0 este implementat și acceptat: 10,19 tok/s față de 1,10 tok/s pentru cel
mai rapid punct LRU/SSD măsurat, 12/12 tokeni identici și zero I/O de experți în
decode. Costul este 7,81 GB peak RSS și 14,59 s preload plătit o dată pe proces.

Două implementări Tier 2 au fost respinse pe stratul 0: codebook-ul de
micro-experți și neuron bank-ul cu corector low-rank au produs 100% fallback
după calibrarea corectă a riscului. Nu le extindem la modelul complet.

Următoarea dezvoltare utilă pentru modele peste capacitatea RAM este un cache
rezident cu buget global și telemetrie pe sesiuni reale, apoi eventual
distilarea unei reprezentări noi. PathPack/prefetch se justifică numai din
trasee lungi care arată predictibilitate și economie netă de I/O.

Poarta D pentru orice Tier 2 rămâne minimum 4x reducere a masei active, cu prag
de calitate fixat pe validation și fallback substanțial sub 100%. Hit-rate-ul,
traficul și costul fallback-ului trebuie raportate, nu presupuse.

## Etapa E — AMC și siguranța secvenței

Numai după ce o reprezentare nouă trece poarta D:

- router ierarhic pentru micro-experți;
- corector rezidual per strat;
- distilare on-policy;
- buget de derivă `D_t`, sondare, checkpoint și replay;
- comparații la 1, 128, 1K și 16K tokeni;
- moduri `calibrated`, `audited-probabilistic` și `exact-window-verified`.

Poarta E folosește hidden/KV/logit KL, rata și lungimea fallback-urilor, rata de
replay, masa activă amortizată și calitatea pe coada distribuției.

## Regula de scalare

3090box este o mașină de analiză și prototip: Ryzen 5 5600, 64 GB DDR4 și SSD
Samsung 870 QVO SATA. Nu folosim tokeni/secundă obținuți aici ca predicție directă
pentru serverul mare. Scalăm la mașina mai puternică numai după ce etapele B–D
produc curbe măsurate și un punct Pareto reproductibil.
