# Independent Strata service

The user-authorized Strata installation serves Swift Flash-Next GSQ-RCO
IQ3_S on the A520M RTX 3090 independently of QPack and llama.cpp.
It uses stable release v0.2.1, Terminal 0.1.46.3 and signed engine 0.1.46.
These are the versions installed and tested on 2026-10-21.
Model assets belong under `${MODEL_ROOT}/strata`; runtime files are under
`work/strata`. Registry: `ops/strata-models.json`.

## Lifecycle

```bash
./ops/strata.sh status swift
./ops/strata.sh start swift
./ops/strata.sh stop swift
./ops/strata.sh restart swift
./ops/strata.sh config swift
```

`install` installs the reviewed upstream release and authenticates its engine.
`publish` authenticates the completed canonical Xet downloads and promotes
the assets. `deploy` prepares the pack and configuration without restarting.
No upstream model acquisition runs during preparation or serving.
HF downloads use only `ops/windows/Start-HuggingFaceModelDownload.ps1` and
`ops/windows/Get-HuggingFaceModelDownload.ps1`. The target GGUFs and the four
MTP files are pinned separately in the registry. The MTP source repository's
NVFP4 name does not mean NVFP4 target weights are downloaded or served.

The `QuantumLLM-Strata` Windows scheduled task owns the managed runtime and
remains alive after SSH disconnects, matching the existing llama service
lifecycle. Stop the active llama/QPack model through its own operations
script before starting Strata. Strata does not stop those services.
The task uses an interactive Windows user principal and is started on demand;
SSH independence does not imply automatic startup after reboot or sign-in.

The `swift`, `swift-flash` and `swift-flash-next-iq3-s` aliases select the
same Strata model. `swift` under `ops/llama.sh` selects a different checkpoint.

## API and Pi

API base: `http://10.10.88.4:8082/v1`; model ID:
`swift-flash-next-iq3-s`. Authentication uses `.env`'s `EXPERT_API_KEY` as
a Bearer token. The Windows inbound rule permits only the WireGuard subnet
`10.10.88.0/24`. Port defaults to 8082; override with `STRATA_PORT` when
preparing the configuration and starting the service.

Home Pi provider: `strata-3090`. The model is registered in
`~/.pi/agent/models.json` and enabled at xhigh in
`~/.pi/agent/settings.json`. The project `.pi` directory remains absent.

```bash
pi --model strata-3090/swift-flash-next-iq3-s --thinking xhigh
```

See [Pi CLI](pi-cli.md#strata-through-pi) for model selection and the
distinction between home defaults and the active server.

## Serving policy and evidence

| Setting | Prepared and tested value |
|---|---|
| Target checkpoint | Swift 1.5 Flash-Next GSQ-RCO GGUF IQ3_S |
| Hardware | A520M, one RTX 3090 24 GiB, Ryzen5 5600, 64 GiB RAM |
| Concurrency | one slot |
| Configured context | 262144 positions |
| Target KV | FP16, 32768 resident GPU cells with host backing |
| MTP | window 4 |
| Routed experts | full top-10, `STRATA_ROUTE_TAIL_SKIP=0` |
| Routed GPU cache | maximum 12 GiB |
| GPU reserve | 1024 MiB |

This model is different from the dense Swift 27B served by llama.cpp. Its
dense/state tensors and cached experts stay on GPU, while nonresident routed
experts are computed on CPU from host RAM. The configuration's
`reasoning_effort: high` selects Strata's native xhigh instruction; API
requests can use `reasoning_effort: xhigh`.

This approximately 84 GB checkpoint also needs a generated expert pack and
substantial host RAM. Initial engine readback reports 37.93 GiB pinned expert
RAM and 11.98 GiB GPU expert cache. Configured context is not proof of
populated-context performance. Qualification results and limits belong in
[Benchmarks](benchmarks.md#independent-strata-api-qualification-2026-10-21);
source decisions and authenticated asset details are in
[Research decisions](research-decisions.md#independent-strata-service-qualification-2026-10-21).

The larger-prompt Pi run measures generation including reasoning. Its first
visible response arrives much later than the first reasoning token at xhigh;
see the separate timings in the benchmark ledger. Generated repair code and
tests were not executed, and populated262K performance remains unqualified.

## Diagnosis

Runtime configuration and logs are under `work/strata/run/`:

- `swift-flash-next-iq3-s.json`: private server configuration;
- `swift-flash-next-iq3-s.startup.json`: managed startup result;
- `swift-flash-next-iq3-s.engine.log`: native engine log.

`./ops/strata.sh config swift` redacts the API key. Install integrity metadata
is in `artifacts/strata-install.json`. A ready status proves startup and API
identity, not a warm cache or populated-context latency.

If local API requests succeed but VPN requests time out, inspect the
`QuantumLLM-Strata-WireGuard` firewall rule and any explicit TCP block for
the versioned Strata Python executable. The launcher resolves the executable
through the `terminal/current` junction and applies the VPN-scoped allow rule.
An empty enabled-block rule set is valid. The launcher enumerates rules and
filters locally because CIM-side filters raise ObjectNotFound for no matches;
actual firewall enumeration/access failures still stop the launch.
After a stop, verify the owned native worker and 8082 listener are gone and
GPU residency returns to the desktop baseline. Do not inspect unrelated Pi
sessions or logs to diagnose this instance.
