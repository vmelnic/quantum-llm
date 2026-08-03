# Baseline Colibri pe 3090box

## Scop

Colibri este baseline-ul exact pentru ierarhia SSD -> RAM -> CPU. El păstrează
routerul și top-k neschimbate și încarcă experții INT8 compleți. Experimentul
HESR-AMC trebuie să demonstreze un avantaj față de acest baseline prin reducerea
byte-ilor activi, nu doar prin reducerea rezidenței RAM.

Sursa este pinată la:

- repository: `https://github.com/JustVugg/colibri.git`;
- commit: `b085b48888a88d9a1c00b151a9979774b72cdbfd`;
- patch local: `ops/patches/colibri-olmoe-load-metrics.patch`.

Patch-ul nu schimbă aritmetica, rutarea sau politica de cache. El adaugă contoare
pentru încărcări/byte-i/prefetch și un mod opt-in `DIRECT=1`. Cu `DIRECT=0`, I/O
rămâne identic cu upstream. Cu `DIRECT=1`, numai tensorul mare
`merged_weight` este citit aliniat prin `O_DIRECT`/Windows `NO_BUFFERING`;
scalele și tensorii densi rămân buffered.

`RAMCACHE=1` este modul exact pentru modelele care încap integral în RAM. La
startup, toate payload-urile experților sunt citite o singură dată în sloturile
INT8 ale procesului; contoarele sunt apoi resetate, iar decode-ul trebuie să aibă
zero miss-uri și zero citiri de experți. `RAMCACHE=0` păstrează comportamentul
streaming/LRU.

Arhitectura efectivă este:

```text
checkpoint INT8 pe SSD
          |
          +-- RAMCACHE=1 --> preload o dată --> expert bank în RAM --+
          |                                                     |
router exact, top-8 --------------------------------------------+--> CPU decode
          |                                                     |
          +-- RAMCACHE=0 --> LRU per strat --> miss --> SSD ----+
```

Acesta este cache de obiecte gata de calcul, nu ramdisk. Un ramdisk ar păstra
formatul de fișier și ar repeta în continuare lookup-ul, citirea și copierea în
sloturile engine-ului; expert bank-ul elimină acești pași din decode.

## Etapele rulării

1. `Invoke-PrepareColibri.ps1` clonează exact commit-ul pinat, refuză modificări
   locale neașteptate, creează mediul Python izolat `work/venv/colibri`, aplică
   patch-ul și construiește `olmoe.exe` cu `ARCH=native`.
2. `Invoke-ConvertOlmoeForColibri.ps1` folosește exclusiv snapshot-ul local HF.
   Proiecțiile `gate/up/down` ale fiecărui expert sunt cuantizate row-wise INT8
   și concatenate pentru o singură citire per expert.
3. `inspect_colibri_olmoe.py` verifică din headere toate cele 16 x 64 perechi
   `merged_weight/qs`, dtype, dimensiuni și tensorii densi. El clasifică separat
   ponderile a căror citire aliniată la 4096 byte ar depăși EOF; engine-ul le
   citește buffered, fără overread.
4. `Invoke-ColibriSmoke.ps1` rulează o singură configurație și cere output
   token-exact față de referința pinată.
5. `Invoke-GenerateOlmoeOracle.ps1` produce separat o referință BF16 prin
   Transformers, CPU-only, din același snapshot local.
6. Smoke-ul este repetat cu `-UseIndependentOracle`.
7. `Invoke-ColibriBenchmark.ps1` compară implicit cache `1,8,64` și pilot `0,1`.

## Metrici

Fiecare rulare raportează:

- tokeni potriviți și totalul așteptat;
- tokeni/secundă și timp engine/wall;
- peak RSS;
- hit/miss pentru cererile reale ale routerului;
- încărcări totale și încărcări speculative;
- byte-i totali încărcați de engine;
- estimarea byte-ilor aferenți miss-urilor reale.

`expert_payload_bytes` măsoară payload-ul logic, `expert_io_bytes` include
overread-ul de aliniere, iar `expert_direct_bytes` numără cererile nebuffered.
Acestea sunt potrivite pentru compararea algoritmilor și overfetch-ului. Nu sunt
telemetrie internă a controlerului SSD, care poate avea propriul cache.

## Cache cold versus warm pe Windows

Colibri controlează propriul LRU. Windows păstrează separat paginile fișierelor
în standby/page cache, iar shim-ul `POSIX_FADV_DONTNEED` este intenționat no-op.
Prin urmare:

- fiecare proces începe cu LRU-ul Colibri gol;
- sistemul de operare poate servi aceleași citiri din RAM după prima rulare;
- cu `DIRECT=0`, scriptul marchează `os_file_cache_state: uncontrolled`;
- cu `DIRECT=1`, ponderile aliniate complet în shard ocolesc page cache-ul;
  ponderile de la capătul shard-ului folosesc fallback buffered, iar rularea
  eșuează dacă modul direct nu este observat deloc;
- nicio rulare buffered nu este declarată `cold SSD` numai pentru că are miss-uri.

Pentru un A/B buffered cold ar fi în continuare necesar reboot sau un mecanism
privilegiat de golire a standby list. Acesta nu este automatizat deoarece
afectează întreaga mașină; baseline-ul fizic implicit folosește în schimb
citirea directă a ponderilor expert.

## Mod interactiv RAM cache

După conversie și build, chat-ul exact pornește implicit cu imaginea experților
în RAM:

```bash
./ops/chat-on-3090box.sh
```

Comparația streaming rămâne disponibilă explicit:

```bash
./ops/chat-on-3090box.sh -RamCache 0 -Cache 8
```

## Siguranță și reproductibilitate

- GPU-ul este ascuns prin variabilele comune CPU-only.
- Dependențele Colibri rulează din `work/venv/colibri`, nu din Python-ul global.
- Snapshot-ul original nu este modificat.
- Conversia scrie numai în `work/models/olmoe-colibri-int8` și poate fi reluată.
- Un fișier `hot_pinned.bin` existent oprește benchmark-ul în loc să altereze
  silențios configurația.
- Smart App Control nu este dezactivat automat. Dacă Windows blochează executabilul
  local, rularea se oprește și decizia de securitate rămâne la operator.
- Instalarea Git/MSYS2/pachetelor Python are loc numai cu switch-uri explicite.
