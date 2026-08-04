# Operațiuni pentru 3090box

Acest director este sursa canonică pentru scripturile rulate pe 3090box.
Repository-ul este sincronizat în `C:\Users\vladi\quantum-llm`. Experimentele
istorice Colibri rămân CPU-only; runtime-ul Expert Pack nou țintește explicit
Windows/CUDA și RTX 3090.

## Expert Pack P0–P2

Build-ul, testele C++, cross-testul compiler→planner și testele compilerului se
rulează prin:

```bash
./ops/sync-to-3090box.sh
./ops/run-on-3090box.sh Invoke-Inventory.ps1 \
  -SustainedStorageReadBytesPerSecond 560000000 \
  -SustainedH2DBytesPerSecond 12000000000
./ops/run-on-3090box.sh Invoke-BuildExpertRuntime.ps1 -Configuration Release
```

Compilerul și validatorul Expert Pack rulează pe Windows printr-un singur
wrapper; checkpoint-ul sursă nu este modificat:

```powershell
Invoke-ExpertPack.ps1 -Action Compile -Path C:\path\snapshot `
  -Output C:\path\model.expert-pack -SourceId org/model -SourceRevision REV
Invoke-ExpertPack.ps1 -Action Validate -Path C:\path\model.expert-pack
```

Specificațiile normative sunt în `docs/`, codul produsului în `core/`,
`compiler/` și `runtime/`; `ops/` conține numai automatizarea operațională.

## Runtime CUDA și serviciul P3–P5

Build-ul Release compilează runnerul end-to-end, workerul persistent și
kernelurile batched. Smoke-ul protocolului încarcă modelul o singură dată și
verifică primii tokeni canonici:

```powershell
Invoke-BuildExpertRuntime.ps1 -Configuration Release
Invoke-ExpertWorkerSmoke.ps1
```

Serviciul OpenAI-compatible pornește implicit numai pe loopback:

```powershell
Start-ExpertServer.ps1 -HostAddress 127.0.0.1 -Port 8080 `
  -BuildId (git rev-parse --short HEAD)
```

Endpoint-uri:

- `POST /v1/completions` și `POST /v1/chat/completions`;
- streaming SSE prin `"stream": true`;
- `GET /health`, `/ready`, `/model-info`, `/v1/models` și `/metrics`;
- coadă limitată, timeout de queue/generation, cancellation la disconnect și
  graceful drain la `SIGINT`/`SIGTERM`.

Front-end-ul negociază protocolul workerului. Cu `-WorkerCapacity 1` păstrează
traseul v1 single-slot. Launcherul P6 folosește implicit patru sloturi și
microbatch decode de 2 ms; fiecare slot are stare KV/Conv/DeltaNet separată, dar
toate request-urile împart dense weights și cache-ul de experți:

```powershell
Start-P6ExpertServer.ps1 -WorkerCapacity 4 -MicrobatchWindowMs 2
```

`/metrics` expune `decode_batches_total` și `decode_rows_total`, astfel încât
batch size-ul efectiv poate fi calculat și nu este doar o setare declarată.
Aceeași fereastră bounded publică `ttft_seconds_p50/p95` și
`inter_token_seconds_p50/p95`; numărul de eșantioane este vizibil, iar memoria
metricilor nu crește odată cu durata procesului.

Pentru bind non-loopback se setează obligatoriu `EXPERT_API_KEY` în mediul
procesului și clienții trimit `Authorization: Bearer ...`. Containerul și
tokenizerul sunt strict locale; startup-ul nu descarcă nimic.

Lansarea fără shell interactiv se instalează în Task Scheduler și poate fi
pornită imediat:

```powershell
Install-ExpertServerTask.ps1 -Start -BuildId (git rev-parse --short HEAD)
Get-ScheduledTask QuantumLLM-ExpertServer
```

Logurile JSONL sunt în `logs/expert-server.jsonl`. Pentru drain/rollback:

```powershell
Stop-ScheduledTask QuantumLLM-ExpertServer
Uninstall-ExpertServerTask.ps1
```

Rollback-ul codului se face pe hostul de control la un commit validat, apoi
`sync-to-3090box.sh`, rebuild, worker smoke și reinstalarea task-ului. Modelul
Expert Pack nu este modificat de deploy/rollback. La recovery se verifică în
ordine logul, `/health`, `Invoke-ExpertPack.ps1 -Action Validate`, build-ul și
worker smoke-ul; un hash mismatch nu este ignorat și nu pornește modelul.

## P6 — Qwen3-Next > RAM

Checkpoint-ul țintă este `Qwen/Qwen3-Next-80B-A3B-Instruct`. Download-ul
rezumabil și statusul lui se lansează astfel:

```powershell
Start-P6ModelDownload.ps1 -MaxWorkers 4
Get-P6ModelDownload.ps1
Invoke-P6Preflight.ps1
```

Compilerul folosește `-Adapter qwen3_next`. Modelul include un decoder auxiliar
MTP; adaptorul îl clasifică și îl păstrează explicit. Nu se șterge și nu se mută
niciun checkpoint existent. Înaintea conversiei trebuie asigurat spațiu pentru
checkpoint plus container; pe instalația curentă există numai C:, deci fluxul
de conversie nu pornește dacă preflight-ul nu confirmă capacitatea.
Opțiunea `Invoke-ExpertPack.ps1 -ReclaimSourceShards` este distructivă și nu
este activată implicit: după ce un pack și starea sa au fost fsync-uite, ea
jurnalizează și elimină numai shard-urile sursă care nu mai au niciun tensor de
consumat. Astfel conversia poate continua cu headroom limitat; checkpoint-ul
sursă trebuie redescărcat ulterior dacă este necesar din nou.

Wrapperul P6 refuză un checkpoint incomplet și cere confirmarea literală:

```powershell
Invoke-P6Conversion.ps1 `
  -SourceReclamationConfirmation DELETE_CONSUMED_SHARDS
```

Fără acest argument, pe discul curent conversia este refuzată înainte de orice
scriere sau ștergere. Pentru recovery după întrerupere se redescărcă shard-urile
jurnalizate, apoi se folosește aceeași comandă cu `-Resume`.

După conversie și validare, serviciul Qwen3-Next folosește același front-end
OpenAI-compatible, dar runnerul și containerul P6 explicite. Startup-ul primește
un timeout mai mare deoarece validează și încarcă `dense.qpack` înainte de
readiness:

```powershell
Start-P6ExpertServer.ps1 -BuildId (git rev-parse --short HEAD)
```

Gate-ul aggregate folosește microbatching real în același proces și același
cache, nu mai multe copii ale modelului:

```powershell
expert-qwen3-next-runner.exe C:\Users\vladi\quantum-llm\work\models\qwen3-next-80b-expert-pack-int8 `
  --batch 151644,872,374 16 4 48 14
```

Rezultatul publică `concurrency`, `aggregate_forward_tokens`, tok/s aggregate și
telemetria comună SSD/RAM/VRAM. Numerele devin gate numai după conversia
checkpoint-ului complet; smoke-ul de kernel nu este tratat ca benchmark de model.
Implicit, runnerul face un warmup complet în același proces, resetează numai
starea request-ului și începe apoi contoarele gate-ului; cache-ul de experți
rămâne intact. Wrapperul canonic rulează ambele porți și monitorizează memoria și
pagefile-ul pe toată durata:

```powershell
Invoke-P6Gate.ps1 -Mode Both -NewTokens 32 -Concurrency 4
```

Gate-ul `no_swap` este strict: pagefile usage trebuie să fie zero înainte și
după fiecare rulare. Un delta zero peste un pagefile deja folosit nu este raportat
ca succes. Artefactul rezultat este `artifacts/p6-gate-latest.json`.

## Fluxul de lucru

De pe hostul de control (acest repository):

```bash
./ops/sync-to-3090box.sh
./ops/run-on-3090box.sh Invoke-Inventory.ps1
./ops/run-on-3090box.sh Invoke-AnalyzeModel.ps1
./ops/run-on-3090box.sh Invoke-FactorizeLayer.ps1
./ops/run-on-3090box.sh Invoke-EvaluateActivations.ps1
./ops/collect-from-3090box.sh
```

Scriptul de sync rulează automat bootstrap-ul după copiere. El nu șterge fișiere
de pe 3090box și nu descarcă modele.

Pentru verificarea dimensiunii modelului de laborator fără download:

```bash
./ops/run-on-3090box.sh Invoke-DownloadLabModel.ps1
```

Download-ul real este intenționat explicit:

```bash
./ops/run-on-3090box.sh Invoke-DownloadLabModel.ps1 -Download
```

## Baseline Colibri pentru OLMoE

După ce snapshot-ul OLMoE este complet, baseline-ul exact SSD/RAM se pregătește
în ordinea următoare:

```bash
./ops/sync-to-3090box.sh
./ops/run-on-3090box.sh Invoke-PrepareColibri.ps1 -InstallToolchain -InstallPythonDependencies
./ops/run-on-3090box.sh Invoke-ConvertOlmoeForColibri.ps1
./ops/run-on-3090box.sh Invoke-ColibriSmoke.ps1
./ops/run-on-3090box.sh Invoke-GenerateOlmoeOracle.ps1
./ops/run-on-3090box.sh Invoke-ColibriSmoke.ps1 -UseIndependentOracle
./ops/run-on-3090box.sh Invoke-ColibriBenchmark.ps1 -UseIndependentOracle
./ops/collect-from-3090box.sh
```

Comanda pipeline echivalentă este:

```bash
./ops/run-on-3090box.sh Invoke-ColibriPipeline.ps1 \
  -InstallToolchain -InstallPythonDependencies \
  -GenerateIndependentOracle -RunSweep
```

Prima rulare trebuie făcută etapizat, nu direct prin pipeline, pentru ca logul și
spațiul pe disc să poată fi verificate după conversie. Detaliile, controalele și
interpretarea metricilor sunt în [`COLIBRI.md`](COLIBRI.md).

Pentru utilizare interactivă exactă, cu toți experții OLMoE preîncărcați în RAM:

```bash
./ops/chat-on-3090box.sh
```

Pe 3090box, echivalentul direct este:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File C:\Users\vladi\quantum-llm\ops\windows\Invoke-Inventory.ps1
```

## Directoare pe 3090box

- `ops/windows/`: scripturile care rulează efectiv experimentul;
- `artifacts/`: inventare și rezultate JSON;
- `logs/`: loguri;
- `work/`: artefacte intermediare regenerabile.

`config/experiment.json` fixează numele mașinii, căile și modelele-candidat.
Checkpoint-urile existente nu sunt mutate sau modificate.

Checkout-ul terț Colibri și modelul convertit se află în `work/` și nu sunt
sincronizate înapoi. Scripturile și patch-ul folosit sunt însă păstrate în acest
repository și sincronizate pe 3090box.

Rezultatele și deciziile go/no sunt sintetizate în `RESULTS.md`.
