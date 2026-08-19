# Neural CPU experiment

The experiment has two levels. Only the second can produce a viability result:

1. `branch_screen.py` is a cheap NCPU-0 mechanism control on synthetic
   activations.
2. `make_prompts.py`, the runner's `--trace-moe` mode, `validate_trace.py`, and
   `experiment.py` implement the NCPU-1 boundary experiment on authenticated
   activations from exact Qwen inference. The full NCPU-1 decision additionally
   requires the in-model splice and free-running gates in the protocol.

The complete real-activation and downstream-splice protocol is documented in
[`docs/neural-cpu-experiment.md`](../../docs/neural-cpu-experiment.md).

## Real-activation pipeline

Generate tokenized, document-level train/validation/test splits:

```powershell
python experiments/neural_cpu/make_prompts.py `
  --tokenizer "$env:MODEL_ROOT/qwen3-next-80b-expert-pack-fp4/tokenizer" `
  --repo . `
  --output work/neural-cpu/prompts `
  --max-tokens 128 `
  --maximum-sequences 64
```

Capture selected complete MoE boundaries. The command refuses to overwrite an
existing trace unless `-ReplaceOutput` is explicit:

```powershell
ops/windows/Invoke-NeuralCpuTrace.ps1 `
  -Container "$env:MODEL_ROOT/qwen3-next-80b-expert-pack-fp4" `
  -PromptFile work/neural-cpu/prompts/prompts.csv `
  -OutputDirectory work/neural-cpu/trace
```

Validate every byte and run all byte-matched models:

```powershell
ops/windows/Invoke-NeuralCpuExperiment.ps1 `
  -TraceDirectory work/neural-cpu/trace `
  -PromptManifest work/neural-cpu/prompts/prompts-manifest.json `
  -Output work/neural-cpu/result.json
```

The training environment is deliberately separate from the API server
environment. It requires a CUDA-enabled PyTorch installation. Trace tensors,
checkpoints, and generated results remain under ignored `work/`; promoted
evidence belongs in `experiments/neural_cpu/results/`.

Create or verify that isolated environment with:

```powershell
ops/windows/Install-NeuralCpuEnvironment.ps1
ops/windows/Install-NeuralCpuEnvironment.ps1 -CheckOnly
```

## Recorded NCPU-0 outcomes

Three deterministic screens are retained as evidence:

- `fixed-point-screen.json`: the simple recurrent cell saturates after four
  cycles and the shuffled program becomes equivalent;
- `branch-accumulator-screen.json`: instruction order matters, but additional
  cycles worsen held-out error and a random-program readout is better;
- `iterative-solver-screen.json`: the learned fixed-point solver also worsens
  with additional cycles and loses to both static and random controls.

All three are negative results for those cells. They do not reject the literal
Neural CPU hypothesis on real model activations; NCPU-1 is the registered test.

The first real-activation pilot is retained as
`results/real-activation-pilot.json`. At the same approximately 3.96 MB budget,
the hard program improved from 0.5480 to 0.4496 test NMSE across 1--32 cycles,
but saturated after cycle 16. A shared-basis static baseline reached 0.3633 and
an ordinary recurrent baseline reached 0.3883. Instruction order and hard
branching were functional, but they did not provide an advantage over ordinary
compression. The registered pilot verdict is `kill/redesign`; no downstream
splice claim is made for this candidate.

`results/computed-dispatch-pilot.json` records the first redesign. It replaced
the fixed loop with hard 16-way paths through reusable low-rank subroutines.
At 3.93 MB it reached 0.3846 test NMSE, beating the ordinary recurrent baseline
but not shared-basis at 0.3633. It saturated after tick 16 and used only 14 of
256 possible two-tick paths. Its strict result is also `kill/redesign`.

Run that candidate against an existing authenticated pilot and reference result
with:

```powershell
python experiments/neural_cpu/dispatch_experiment.py `
  --trace work/neural-cpu/pilot-trace `
  --prompts-manifest work/neural-cpu/pilot-prompts/prompts-manifest.json `
  --reference-result work/neural-cpu/pilot-result.json `
  --output work/neural-cpu/dispatch-result.json
```

`results/torus-butterfly-pilot.json` records a fixed-trace reversible redesign.
It used a parameter-free round-robin address program and a 112-byte learned
determinant-one coupling bank over a width-240 latent. The best point was 0.4195
NMSE at 64 ticks; 128 ticks regressed to 0.4265. Strong schedule, bank, and
nonlinearity lesions confirmed that its loop carried function, but 99.90% of
bytes remained in encode/decode and it lost to every primary static/recurrent
reference. It is classified as a tied-depth structured map and receives `kill`.

Run it with:

```powershell
python experiments/neural_cpu/torus_experiment.py `
  --trace work/neural-cpu/pilot-trace `
  --prompts-manifest work/neural-cpu/pilot-prompts/prompts-manifest.json `
  --reference-result work/neural-cpu/pilot-result.json `
  --output work/neural-cpu/torus-result.json
```
