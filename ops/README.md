# Operațiuni pentru 3090box

Acest director este sursa canonică pentru scripturile experimentului. Repository-ul
este sincronizat în `C:\Users\vladi\quantum-llm`, iar scripturile PowerShell sunt
rulate pe 3090box. GPU-ul este inventariat, dar este dezactivat explicit pentru
procesele experimentului.

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
