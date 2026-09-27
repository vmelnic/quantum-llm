# Benchmarks and evidence

Status: measurements through 2026-09-26; documentation normalized
2026-09-27.

## Reporting contract

- TTFT is server time to the first generated token unless labelled otherwise.
- Decode rate excludes TTFT and the first token.
- Prompt, reasoning, visible and total generated tokens are separate.
- Configured context and populated context are separate.
- Cold, retained-prefix, disk-restored and warm runs are separate.
- A kernel or numerical gate is not service throughput.
- All rates below are single-request measurements unless stated otherwise.

## Validation baseline

The last documented Windows gate used MSVC 19.44 and CUDA 13.4 with a verified
clean build directory and `--clean-first`.

| Gate | Result |
|---|---|
| CTest | 8/8 passed |
| canonical Python suites | 99 passed |
| compiler/service tests in the current checkout | 99 passed, one optional NumPy skip |
| benchmark helper tests in the current checkout | 22 passed |
| build warnings | 14,992 from vendored CUTLASS/FlashAttention; zero first-party warnings |

The six models in the Pi catalog each returned visible text from an isolated
`xhigh` `hi` request on that build. This qualifies service, tokenizer,
template and generation wiring only. One Ornith request produced a coherent
answer inside reasoning without visible text; a later temperature-0.6 series
produced visible text in 30/30 attempts, but that sampling policy has not been
published.

## Artifact evidence

| Artifact | Validated payload evidence | Advertised context |
|---|---:|---:|
| Qwen3.8-27B | 14,775,390,208-byte QPack; 1,199 records | 262,144 |
| Qwen3.8-27B Abliterated | 14,775,390,208-byte QPack; separately published third-party weights | 262,144 |
| Qwen3.8-Flash-Next | 95,915,634,688 bytes; 1,562 dense and 24,576 expert records | 262,144 |
| Mistral Small 4 | 70,801,904,048 validated source-tensor bytes; native NVFP4/BF16 | 262,144 |
| Muse-Glimmer | 15,832,002,560-byte QPack | 131,072 |
| Ornith | 19,229,777,920-byte pack | 262,144 |

Source reconstruction is necessary but does not qualify provider execution,
model behavior or maximum context.

## Qwen capacity

```text
hot target weights             13,625,700,352 bytes = 12.690 GiB
exact F16 KV at 262,144        17,179,869,184 bytes = 16.000 GiB
subtotal                       30,805,569,536 bytes
RTX 3090 physical              25,769,803,776 bytes
```

The measured pinned-host link was 12.46 GiB/s. Scanning the full 16 GiB host KV
therefore has a 1.284-second transfer floor for one scalar target call.

Q4H stores a 260-byte K/V pair for each layer, KV head and token:

```text
target KV at 262,144 positions   4.062500 GiB
Q8 MTP pages                     0.503906 GiB
combined                         4.566406 GiB
```

Q4H is lossy relative to F16.

## Maximum populated context

A deterministic prompt placed retrieval and code facts at positions 2,068,
131,013 and 196,510 and queried them at 261,904.

| KV path | Populated prompt | Output | Wall | Result |
|---|---:|---:|---:|---|
| historical K1 | 262,016 | 47 | 546.173 s | retrieved all facts |
| exact F16 | 262,016 | 47 | 2,278.082 s | byte-identical output |

Both responses made the same unrelated arithmetic error. This single parity
case does not establish general K1 fidelity. Exact F16 proves capacity and
fails the interactive latency target.

The exact hierarchical top-k implementation reduced sampling from an observed
9.916 ms per target call to 0.258 ms in the worker trace. On the populated K1
request, post-first decode improved from 14.39 to 17.07 tok/s. This is an exact
sampling improvement; K1 remains lossy and is not an active default.

## Operational Q4H and MTP-4

The Qwen exact-decode ABI verifies one guaranteed target position plus up to
four MTP proposals in one target batch. Target sampling uses exact `p/q`
rejection and residual correction. Target KV is Q4H; proposer KV is Q8.

Live service gates used suffix windows from the same 262,016-token coding
prompt, 128 generated tokens, `temperature=1.0`, `top_p=0.95`,
`top_k=20` and seed 1. Session retention was disabled.

| Populated prompt | TTFT | Post-first decode | Target calls / positions | Positions/call |
|---:|---:|---:|---:|---:|
| 32,768 | 39.627 s | 54.54 tok/s | 48 / 128 | 2.6667 |
| 131,072 | 203.250 s | 47.16 tok/s | 42 / 128 | 3.0476 |

Only those two measured points produce this linear estimate:

```text
estimated 262,144 decode = 39.94 useful tok/s
```

No live Q4H 262K service gate has passed.

### External endpoint checks

Both tools called the real OpenAI-compatible endpoint with one active request,
no session retention and the artifact thinking sampler.

| Tool | Actual prompt | Output | TTFT | Post-first decode |
|---|---:|---:|---:|---:|
| llama-benchy 0.4.0 | 34,817 | 32 | 42.131 s | 56.99 tok/s |
| GuideLLM 0.7.4, concurrency 1 | 32,820 | 128 | 39.861 s | 60.58 tok/s |

GuideLLM generated reasoning only and measured 3.05 tok/s end to end including
prefill. The two decode rates differ with MTP acceptance. Neither run is a
quality or latency-distribution result.

## Long-agent and language quality

Two Qwen Abliterated Q4H Pi requests entered lexical loops:

| Prompt | Generated before cancellation | TTFT | Wall |
|---:|---:|---:|---:|
| 133,610 | 12,952 | 208.906 s | 449.750 s |
| 148,441 | 79,608 | 240.218 s | 1,533.172 s |

A coherent 4,786-token request completed between them. The evidence rejects a
claim of reliable long-agent behavior and does not isolate Q4H because no
matched codec replay was captured.

The current artifacts use `presence_penalty=0.5` and a 32,768-token thinking
circuit breaker. The earlier value 1.5 was removed after visible Romanian
degradation.

A controlled Romanian matrix crossed official and Abliterated weights, Q4H and
F16 target KV, MTP-4 and scalar target decode, presence penalties 0.5 and 0.0,
and temperatures 1.0 and 0.6. Material malformed Romanian remained in every
row. This rules out target-KV format, MTP and sampling policy as sufficient
fixes. It does not identify whether the remaining cause is source behavior,
stored FP4 weights or provider numerics.

An English synthetic request generated 39 complete numeric records and 38
complete prose records without observed numeric corruption, then hit the
32,768-token completion limit before record 40 and its checksum. It is a
bounded negative-corruption observation, not a general fidelity pass.

## Coding harness

| Model/path | Result |
|---|---|
| Qwen F16, effective `medium` | Todo fixture 3/3 in 503.240 s; about 17.8K final prompt |
| historical Qwen K1, `xhigh` | Todo fixture 3/3 in 979.17 s |
| Qwen Flash, corrected `medium` | valid tools, zero edits; exceeded 1,000 s |
| Mistral | 14 valid tool calls, zero edits at 30-minute cutoff |
| historical Ornith K1+fit, `xhigh` | 2/3 tests after 1,832.80 s |
| Muse | no representative coding result |

These runs use different model paths and reasoning modes. They establish only
the result stated in each row.

## Pi 256K benchmark

The current benchmark records every completed Pi model call at its actual
populated prompt size. It uses a gradual offline-first conference application,
real file operations and live web retrieval through Pi's standard PowerShell
tool. The terminal gate is a prompt from 255,000 through 255,999 tokens.

The latest Quantum run is incomplete:

| Metric | Result |
|---|---:|
| completed model calls | 25 |
| maximum populated prompt | 58,664 |
| generated / reasoning / visible tokens | 50,323 / 29,613 / 20,710 |
| provider prefill / decode wall | 1,327.951 / 719.779 s |
| weighted generated-token decode | 69.915 tok/s |
| visible end-to-end rate through last completed call | 9.950 tok/s |

The next call emitted 690 thinking deltas and then produced no event, tool call
or provider completion for more than two hours while the GPU was idle. The run
was stopped and cleaned up. No 256K result, stall root cause or correctness
pass exists.

All 25 completed calls reported full-prompt prefill and
`session_retain_skipped` because an unemitted speculative bonus prevented an
exact client-visible retained prefix.

### Pinned llama.cpp comparison

The comparative Pi benchmark ran the two backends sequentially on the same RTX
3090 with the same English dialogue, Pi `xhigh` and a configured 256,000-token
window. Quantum used official Qwen3.8-27B standard FP4 weights, Q4H target KV
and MTP-4. The comparison used pinned llama.cpp v0.5.0 with the official
Q4_K_M GGUF, `q4_0` target KV and an MTP-4 draft. The formats are not
numerically equivalent.

The opening `hi` call populated exactly 2,239 prompt tokens on both backends:

| Metric | Quantum LLM | llama.cpp | Observed ratio |
|---|---:|---:|---:|
| generated tokens | 46 | 37 | — |
| client TTFT | 2.813 s | 6.281 s | llama.cpp took 2.23× longer |
| provider prefill | 2.687 s | 6.061 s | llama.cpp took 2.26× longer |
| post-first decode | 62.670 tok/s | 6.740 tok/s | Quantum reached 9.30× the rate |
| complete client call | 3.547 s | 11.765 s | llama.cpp took 3.32× longer |

The other two completed llama.cpp calls decoded at 8.390 and 6.672 tok/s. The
run was stopped because its early rate did not justify continuing the
long-context dialogue. It reported 48/65 target layers and 66/66 draft layers
on GPU, 5,727 MiB of CPU-mapped model buffers and 1,125 MiB of CPU KV. Its
Q4_K_M weights plus Q4_0 target KV, MTP weights and the mandatory 1 GiB reserve
have a 24.630 GiB lower bound before workspaces, so full placement cannot fit
in 24 GiB.

Quantum completed 25 calls, reached 58,664 populated prompt tokens and recorded
69.915 generated tok/s weighted across completed decode. It later stalled with
an idle GPU before reaching the terminal context gate. Neither run reached
256K, and the difference in formats prevents a numerical-fidelity comparison.
The 9.30× figure applies only to the same measured opening call.

## Sparse and auxiliary paths

### Qwen Flash cold prefill

One real Pi prompt populated 4,190 tokens with exact F16 KV and the 12 GiB
routed-VRAM ceiling.

| Metric | Prior | Current |
|---|---:|---:|
| prefill batches | 10 | 1 |
| storage read | 158.8 GB | 62.03 GB |
| H2D upload | 236.8 GB | 61.98 GB |
| TTFT | 325.3 s | 163.640 s |

The grouped standard-FP4 route preserved the exact selected experts and passed
its independent numerical oracle. The request remained too slow for an
interactive coding claim.

### P100 admission

The two P100 PCIe cards have no peer access with the RTX 3090.

| Proposed path | Best result | Required | Decision |
|---|---:|---:|---|
| exact 262K attention split | 77.101 ms compute-only | <=32 ms | rejected |
| compact dense shard | 35.939 ms/card before integration | <=20 ms/card | rejected |
| lossless F16 MLP shards | 307.7/311.0 GB/s | >=500 GB/s/card | rejected |
| Qwen Flash selected experts | 0.20 tok/s | beat 0.37 tok/s primary | rejected for speed |

The secondary executor remains a capacity mechanism for compatible routed
pages.

## Sessions and cleanup

| Gate | Result |
|---|---|
| two exact-F16 sessions | four park/restores; suffix-only prefill; 351,952,896 parked bytes |
| alternating Pi sessions | about 6,741 tokens/session; 1,223,811,072 parked bytes |
| cancellation | active count returned to zero; committed prefix survived |
| progressive growth | 200 -> 300 -> 300 used 300 cumulative prefill tokens |

A historical K1 Qwen restart/resume gate persisted 443 tokens in 169,443,798
bytes. After a complete service restart it restored in 0.203 s, prefilled a
34-token suffix and completed in 0.938 s. Fixed-seed continuation produced the
same next text before and after restart. A large real coding-session
persistence gate and a matched current-Q4H restart gate remain unrun.

Recorded gates stopped the service and verified process, port and GPU cleanup.

## Current conclusion

- six published models pass isolated wiring smokes;
- Q4H plus MTP-4 provides fast 32K and 128K Qwen decode;
- no live Q4H 262K result exists;
- exact-F16 262K capacity works and its latency fails the product target;
- long-context Q4H behavior and Romanian fidelity are unqualified;
- the autonomous Pi run has not reached its 256K terminal gate;
- Flash, Mistral, Muse and Ornith remain below their complete qualification
  gates;
- persistent exact-prefix continuation works on the measured small Qwen gate;
- the P100s have not improved measured throughput.
