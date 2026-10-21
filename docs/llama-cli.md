# llama.cpp CLI

The user requested an independent llama.cpp serving path for the downloaded
Swift and KAT GGUFs. This is an explicitly authorized operations exception;
it does not qualify either model for the artifact VM or change QPack launchers.

## Commands

```bash
# Install the latest stable Windows CUDA build; leave the active model running.
./ops/llama.sh update

./ops/llama.sh start swift
./ops/llama.sh pi swift

./ops/llama.sh start kat
./ops/llama.sh pi kat

./ops/llama.sh start swift-heretic
./ops/llama.sh pi swift-heretic

./ops/llama.sh status
./ops/llama.sh stop

# Publish and prepare the next launch; leave the active service alone.
./ops/llama.sh deploy swift
# Activate the prepared configuration when ready.
./ops/llama.sh restart swift
```

`swith` is accepted as an alias for `swift`. `pi` without a selector discovers
the running model. `restart` validates and restarts the selected model;
`publish` only validates and publishes it. `chat` is an alias for the Pi action.
`deploy` publishes and atomically prepares the launch file without start,
stop, task registration, health requests or removal of the process record.
The worker reads the launch file once at startup. Preparation takes effect
on the next restart, with the same Pi model ID and API address.

The script resolves `.env` relative to the repository, synchronizes source on
publication/start, and uses the Windows host configured by
`QUANTUM_LLM_REMOTE` and `QUANTUM_LLM_REMOTE_ROOT`. It does not stop QPack.
Both backends use the same RTX 3090, so stop QPack with `ops/model.sh stop`
before using llama.cpp if QPack is running; stop llama.cpp before starting
QPack. Switching between Swift and KAT stops the previous llama.cpp process.

## Artifact and service contract

`ops/llama-models.json` declares aliases, model IDs, pinned repositories,
revisions, file sizes, SHA256, context and KV policy. The generic publisher
validates the exact size, GGUF v3 header and SHA256 before creating a candidate.
It hard-links the unchanged downloaded file under `${MODEL_ROOT}/<artifact>`,
writes an authenticated manifest/completion record, then promotes the folder.
Existing artifacts must pass the same checks before start.

An optional strict `projector` object declares its pinned repository, revision,
file, bytes, SHA256 and boolean `offload` policy. Publication authenticates and
hard-links both the main GGUF and projector before promoting the candidate.
Launch preparation rechecks projector size/SHA256 and adds `--mmproj`;
`offload: false` also adds `--no-mmproj-offload` for CPU/RAM encoding.

Swift Heretic's current prepared artifact is
`${MODEL_ROOT}/swift-heretic-iq4-xs-mtp10-vision-cpu`, containing the unchanged
IQ4_XS backbone and the 629,247,712-byte Q8 visual projector. Context 262,144,
one slot, GPU layers all, fitting off and MTP-10 are preserved. The user Pi
catalog enables text/image input for the same `swift-heretic` ID, already
included in home settings. Preparation did not restart the active
service or run inference; the user activates vision with
`./ops/llama.sh restart swift-heretic`.

One generic scheduled task, `QuantumLLM-LlamaCpp`, runs the installed CUDA
binary declared in `artifacts/llama-install.json`. It uses CUDA
ordinal 0 on the independent A520M/RTX 3090 host; the X99E/P100 rig is not
involved. Both models use `--gpu-layers all --fit off`. The user requested
restoring Swift's single-slot policy after reporting severe slowdown with
two slots and automatic CPU/RAM fitting. Context shifting remains disabled.

Both models default to one slot with total context 262,144 and retain
Q4_0 K/V, flash attention,
512-token batches and 128-token microbatches. The optional `speculative`
artifact policy selects integrated MTP and declares its proposal limit and
draft KV encodings. It shares target weights; no separate draft model is loaded.
Swift's authorized MTP policy is `draft-mtp`, maximum 10 proposals,
Q8_0 draft K/V. KAT's policy is maximum 4 proposals with Q8_0 draft K/V,
using a separately published GGUF with the Qwen3.6-35B-A3B MTP head grafted
onto the Bartowski IQ4_XS backbone. The official KAT checkpoint has no native
MTP head; this borrowed-head variant was explicitly authorized after disclosure.
Swift restores the existing `swift-1.5-iq3-xxs-mtp10` single-slot artifact.
The previous parallel policy artifacts remain intact. No download or weight
change is needed for rollback.
Ten is a maximum proposal count, not an accepted-token count or speed promise.
Reasoning defaults to xhigh using each embedded template. The 32,768-token
per-turn output limit is a runaway guard, separate from the context window.
Prompt caching and recurrent checkpoints are enabled; cache reuse is best
effort, not a persistence or long-context speed guarantee.

The separate server listens on `LLAMA_HOST` (default `0.0.0.0`) and `LLAMA_PORT`
(default `8081`). `LLAMA_BINARY` can override the executable. It uses the
existing `EXPERT_API_KEY`; the key file has a user-only Windows ACL. Launch,
process and key files live under ignored `artifacts/`, with logs under `logs/`.
Start validates `/health`, `/v1/models`, `/props`, per-slot context and declared slots.
Stop checks process identity and does not terminate an unrelated port owner.

`update` resolves GitHub's latest stable release, verifies the tag's exact
source commit and clean checkout, then builds Windows Release/CUDA with
`--clean-first`, CUDA 13.4 and architecture 86 for the RTX 3090. It checks
the resulting executable's version and launcher options without loading any
model. Only after these gates does it atomically update the installation
record and replace the prepared launch's binary path, preserving all model
arguments. Existing executables remain available to the active process;
there is no service stop/start/restart or API request. Subsequent launches
verify the installed executable's SHA-256. The `LLAMA_BINARY` override remains
available. Historical benchmark scripts retain their original pinned binary.
Repeating `update` for the same verified installed release returns immediately
without rebuilding its potentially active directory. The installed stable
release is v0.6.0, commit `d81235049384534c167caea52b85a694f6103d14`, prepared
on 2026-10-07 without restarting the active model. Its executable version and
required CLI options passed inspection; model inference was not run.

## Local Pi

GPT-OSS-20B is available through aliases `gpt-oss`, `gpt-oss-20b` and
`gpt-oss-20b-mxfp4`. The exact published GGUF is the author's MXFP4 conversion
from `ggml-org/gpt-oss-20b-GGUF`; no Eagle draft is selected. It uses one slot,
native context 131,072, F16 K/V, maximum output 32,768 and high reasoning.
Pi xhigh/max map to native high; off maps to low. It is text-only.

```bash
./ops/llama.sh start gpt-oss
./ops/llama.sh pi gpt-oss
./ops/llama.sh stop
```

Home Pi enabled-model settings include
`llama-3090/gpt-oss-20b-mxfp4:xhigh`. Existing Swift defaults are preserved.
Publication and preparation passed file/argument gates without starting or
restarting a service; live coding/tool quality and speed have not been tested.

`~/.pi/agent/models.json` contains provider `llama-3090`, pointing from the
local Mac to `http://10.10.88.4:8081/v1`, with:

- `swift-1.5-iq3-xxs`;
- `swift-heretic`;
- `kat-coder-v2.5-iq4-xs`.

Home Pi settings include these models for `/model` and Ctrl+P. Swift is the
home default with xhigh reasoning. The project `.pi` directory has been
removed and remains absent; current selection uses `~/.pi/agent/`.
Other providers and compaction settings are preserved. Historical user
configuration backups remain beside the originals, and the former project
settings backup is under ignored `out/llama-config-backups/`.

The launcher verifies the running model and context against Pi's registry,
then starts the local Pi CLI with the selected provider and xhigh. Pi sends
ordinary OpenAI-compatible streaming requests and template kwargs; llama.cpp
owns template rendering and reasoning/tool parsing.

The user Pi registry declares 262,144 for Swift. The launcher checks each
model's declared slot count and per-slot context rather than requiring one
slot. The prepared single-slot rollback matches this client limit after
restart. A running server keeps its startup policy until restarted by the
user. A slot must fit both its input and requested output.

Explicit launch from any project after starting the server:

```bash
pi --provider llama-3090 --model swift-1.5-iq3-xxs --thinking xhigh
pi --provider llama-3090 --model swift-heretic --thinking xhigh
pi --provider llama-3090 --model kat-coder-v2.5-iq4-xs --thinking xhigh
```

## Capacity prerequisite

At 262,144 configured positions, the previously recorded geometry gives:

```text
Swift: 9.401 GiB file + 4.500 GiB Q4_0 KV + 0.141 GiB recurrent
       + 0 draft + 1 GiB reserve = 15.042 GiB before conv/workspaces
Swift MTP-10: previous subtotal + 0.325 GiB head + 0.531 GiB draft KV
              + 10 * 0.141 GiB recurrent rollback allowance
              = approximately 17.304 GiB before conv/workspaces
Swift MTP-20: same head/KV plus 20 * 0.140625 GiB rollback allowance
              = approximately 18.710 GiB before conv/workspaces
Swift Heretic MTP-10: 14.258 GiB complete GGUF (including head)
                     + 4.500 GiB target KV + 0.141 GiB recurrent
                     + 0.531 GiB draft KV + 1.406 GiB rollback + 1 GiB reserve
                     = approximately 21.836 GiB before conv/workspaces
KAT:  17.515 GiB file + 1.406 GiB Q4_0 KV + 0.059 GiB recurrent
       + 0 draft + 1 GiB reserve = 19.980 GiB before conv/workspaces
KAT MTP-4: 18.351 GiB file + 1.406 GiB target KV + 0.059 GiB recurrent
           + 0.266 GiB draft KV + 4 * 0.059 GiB rollback + 1 GiB reserve
           = approximately 21.316 GiB before conv/workspaces
```

Dense weights and retained full-attention KV remain hot per target call.
For scalar decode, accepted tokens/call = 1 and draft/proposer cost = 0.
For MTP, useful positions/cycle are A and total cycle time includes proposal,
target verification and rollback/sampling. Speed improves only when
`T_cycle/A` is smaller than scalar decode time at the same populated context.
The bandwidth lower bound is `bytes_touched / measured_GPU_bandwidth`, where
bytes touched include hot dense matrices, selected routed experts, active KV
and intermediates. Full GPU placement has zero planned RAM/PCIe/NVMe hot-state
transfers per accepted token. These equations are capacity prerequisites, not
measured throughput. Configured context is not actually populated context;
small integration probes do not qualify 256K coding speed or quality.

## Verified integration

The user-requested `swift-heretic` alias selects the unchanged
`mradermacher/Swift-1.5-Qwen3.8-27b-heretic-i1-GGUF` IQ4_XS file, revision
`45c709974da7cdcb04ea580c7edf55a1804d1953`. The downloaded file passed exact
size, GGUF v3 and pinned SHA-256 validation before transactional publication
under `${MODEL_ROOT}/swift-heretic-iq4-xs-mtp10`. Read-only inventory found
64 backbone blocks and one integrated MTP block with 15 tensors. Its policy
uses one slot, context 262,144, Q4_0 target/Q8_0 draft KV, ten maximum
proposals, full GPU layers and fitting disabled, through the existing generic
launcher. Pi inherits the existing Swift template compatibility and xhigh
mapping. Existing defaults, active service and prepared launch remain
unchanged. No inference, tool transport or quality gate was run for Heretic.

On 2026-10-02, both exact published files passed the Windows size/header/SHA256
gate. Both models advertised one slot and context 262,144. The local Mac Pi
CLI parsed streamed reasoning, executed a real `read` tool, sent its result
back to the server and returned the exact file marker. The initial tool-call
prompts contained 965 tokens for Swift and 927 for KAT. These probes establish
API/tool wiring only. Reasoning appeared as structured blocks even though the
reported reasoning-token usage counter was zero.

The revised process-handle lifecycle passed the real start/stop/status gate;
stop returned GPU use to 428 MiB. Existing Pi provider definitions and other
settings matched their backups, apart from the two added enabled model IDs.

The previous live integration evidence applies to the non-MTP files only.
For the requested MTP deployment, offline launch/policy gates are separate
from startup, native allocation, real Pi coding quality and speed. The user
explicitly reserves restart and live testing; preparing MTP does not pass
those gates. The existing CUDA binary and native source remain unchanged.

MTP preparation on 2026-10-02 passed the exact candidate file hash and an
independent byte comparison of all 851 original backbone tensor payloads;
only 15 MTP tensors were added. The candidate was transactionally published,
and `deploy swift` prepared the real host launch with maximum 10 proposals.
Serving task metadata and the process record remained unchanged. Offline
Windows tests also blocked every serving lifecycle/API/process call during
preparation. No MTP server startup or inference request was executed.

After the user reported 45–60 tok/s in Pi and roughly 19 GB VRAM at MTP-10,
the requested limit of 20 was prepared on the same date. Those numbers are
user observations with unknown populated context and rate accounting, not
controlled benchmarks. The MTP-20 policy artifact hard-links the same GGUF
and leaves the MTP-10 artifact intact. Real prepared-launch checks passed;
serving metadata remained unchanged. The user retains restart and live testing.
Additional rollback core state is estimated at 1.40625 GiB relative to ten
proposals. Workspace means intermediate GPU buffers used by matrix operations,
attention and verification; any additional allocation remains unmeasured.

The user subsequently reported 30–40 tok/s and roughly 20 GB VRAM at MTP-20
and requested returning to ten without restart. `deploy swift` now selects
the intact MTP-10 artifact and prepares the limit of ten for the next startup.
The existing file/manifest and prepared-launch contract passed validation.
No active session inspection, inference test, log read or serving restart was
performed. The reported rates do not establish a controlled causal comparison.

## KAT borrowed-head deployment

The official KAT-Coder-V2.5-Dev checkpoint declares zero MTP layers.
The selected [thread13 GGUF](https://huggingface.co/thread13/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF-MTP/tree/b2c680bf22e192eae1fc30e6060cac77dd144a91)
appends the original Qwen3.6-35B-A3B MTP head from Unsloth. Its publisher
states that the Bartowski IQ4_XS backbone payloads are preserved; independent
backbone comparison, numerical quality, native execution and Pi gates were
not run because the user prohibited tests. This is not a KAT-trained head.

On 2026-10-02, the exact pinned file was downloaded through the canonical Xet
path and transactionally published under `${MODEL_ROOT}/kat-coder-v2.5-iq4-xs-mtp4`:

```text
repository: thread13/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF-MTP
revision: b2c680bf22e192eae1fc30e6060cac77dd144a91
file: Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.bartowski.mtp.gguf
bytes: 19,704,404,096
sha256: 5037eb2b43466696aa5e3e918f4bfb8d12b6d3a769a28a36ceb84c9289a195a9
```

Publication checked file size, GGUF v3 header and the publisher SHA256.
Read-only metadata inventory records 40 backbone blocks plus one MTP block,
with 20 head tensors: Q8_0 matrices, BF16 routers and F32 norms. The pinned
llama.cpp binary already implements this one-block integrated MTP path;
no native source or binary was changed for KAT.

The existing `kat` alias and `kat-coder-v2.5-iq4-xs` Pi model ID select the
published MTP-4 policy on the user's next `./ops/llama.sh restart kat`.
This registry deployment leaves the prepared Swift MTP-10 launch file untouched.
The original scalar KAT artifact remains intact. No serving lifecycle, active
session inspection, inference request, benchmark or test was executed.
Only this download's own diagnostic status was inspected; active Pi/server
logs were not read. Actual residency, acceptance, quality and speed remain
unmeasured for this variant on the RTX 3090.
