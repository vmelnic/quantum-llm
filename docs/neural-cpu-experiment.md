# Neural CPU experiment protocol

This document records the independent Kimi research proposal for testing a
literal Neural CPU, together with the evidence produced while executing it.
Proposed gates and measured results are labeled separately.

## Execution status

The real Qwen runner now exposes an explicit `--trace-moe` mode at the exact
post-attention-normalization to complete-MoE-output boundary. Trace v1 stores
inputs, outputs, layer/sequence/position identity, router diagnostics, exact
sizes, and SHA-256 for every file. A smoke run produced 236 authenticated
records: 59 tokens for each of layers 20--23. All vectors were finite and every
top-10 routing-weight sum was within `1.8e-7` of one.

Three NCPU-0 cells were then evaluated against one real packed Qwen expert on
deterministic synthetic activations. All failed their registered controls:

| cell | tick-1 NMSE | tick-32 NMSE | decisive failure |
|---|---:|---:|---|
| fixed-point recurrence | 1.2814 | 1.0066 | saturated by tick 4; shuffled became equivalent |
| hard branch + accumulator | 1.0027 | 1.1751 | time worsened; random readout reached 0.9893 |
| iterative latent solver | 1.0465 | 1.2383 | time worsened; static and random were better |

These are genuine negative results for the tested cells, not a rejection of
the full hypothesis: random Gaussian activations are outside the model's real
hidden-state manifold. The authoritative NCPU-1 run uses authenticated real
activations, grouped held-out splits, byte-matched controls, and hard/soft
evaluation. Raw result JSON is retained under
`experiments/neural_cpu/results/`.

The NCPU-1 real-activation pilot is also complete. It used 29,280 records from
7,320 tokens and four exact MoE boundaries, split into 23,680 train, 2,804
validation, and 2,796 test records by complete source group. Every model had a
persistent budget of approximately 3.96 MB.

| held-out test | persistent bytes | NMSE | cosine |
|---|---:|---:|---:|
| Neural CPU, 1 cycle | 3,958,964 | 0.5480 | 0.5727 |
| Neural CPU, 16 cycles | 3,958,964 | 0.4505 | 0.6648 |
| Neural CPU, 32 cycles | 3,958,964 | 0.4496 | 0.6652 |
| static GLU | 3,937,280 | 0.3800 | 0.7180 |
| independent low-rank | 3,932,160 | 0.3695 | 0.7332 |
| shared basis/core | 3,939,600 | **0.3633** | **0.7353** |
| recurrent without ISA | 3,955,392 | 0.3883 | 0.7135 |
| random program + readout | 3,958,964 | 0.8616 | 0.4146 |

The shuffled hard program reached 1.5871 NMSE, so instruction order carries
real function. Hard execution also beat its soft diagnostic, 0.4496 versus
0.5310. Nevertheless, the candidate failed the primary mechanism gates:

- improvement from one to 32 cycles was 17.96%, below the registered 20%;
- improvement from 16 to 32 cycles was only 0.21%, demonstrating saturation;
- the best byte-matched static baseline was 19.2% lower-error;
- the ordinary recurrent baseline was 13.6% lower-error.

A post-run independent review found that the implementation had accidentally
softened two pre-registered controls: it accepted shuffled degradation above
2× instead of 10× and required random/readout merely to be worse instead of
5× worse. The raw artifact preserves that original evaluator output. Under the
correct gates, shuffled degradation is only 3.53× and random/readout only 1.92×;
both controls fail. The evaluator is corrected for subsequent runs. The review
also noted that the first random control used a manually chosen alternate
instruction order; subsequent runs derive legal random bytecode from the
declared seed.

Therefore this ISA/cell receives a measured **kill/redesign** verdict. The
downstream Qwen splice is not justified for this candidate because its boundary
gate already failed. The result does not establish that every possible literal
Neural CPU is impossible; it establishes that hard branch selection plus the
tested low-rank iterative solver does not beat ordinary compression at this
budget and execution range.

### Structural diagnosis and redesign boundary

The failure is consistent with the implemented transition:

```text
Y <- Y + alpha * gate * (proposal - Y)
H <- rms_norm(input + feedback(Y))
```

This is a contractive fixed-point iteration. The eight-atom `used` mask forces
one traversal of the same palette every eight cycles; by cycle 16 the proposal
state is nearly converged, so later cycles recompute the same point. The
branch carries only `log2(8) = 3` bits per cycle, while the exact Qwen route
chooses ten of 512 experts plus continuous weights. Meanwhile the winning
shared-basis model spends its bytes directly on dense per-layer cores rather
than controller, feedback, and branch machinery.

Two distinct redesign families remain technically credible:

1. **Computed dispatch:** make the program counter itself encode a
   data-dependent route through reusable straight-line subroutines. A decisive
   test must audit path entropy and mutual information with teacher routes,
   permute CALL/JTABLE links, and still beat the byte-matched shared-basis map.
2. **Addressed matching pursuit:** repeatedly address a counted dictionary and
   add a selected direction with a computed gain. This is accepted only if the
   residual used for addressing is computable from candidate state alone. A
   true-target residual would leak the oracle and invalidate the experiment;
   therefore monotonic pursuit cannot be claimed merely from teacher error
   measured after the fact.

The next implementation candidate is computed dispatch because its route is
available from the input activation without access to teacher output. The
matching-pursuit design remains blocked until it defines a non-oracular
residual with an independently testable self-consistency law.

The proposed system is a deterministic register machine with a fixed
instruction set, a program counter, bounded loops, hard bytecode, and a small
constant pool. The original Qwen checkpoint is a teacher during compilation
and is absent from Neural CPU inference.

## Hypothesis

At a fixed total number of stored bytes, a hard-program Neural CPU can improve
its approximation of a real MoE function by executing more instructions. At a
sufficient instruction budget it must outperform the best static surrogate
trained on the same teacher data and constrained to the same stored bytes.

This is the distinguishing time-for-space claim:

```text
fixed program bytes + more execution ticks -> better function
```

If error stops improving as the instruction budget grows, or if a byte-matched
static model remains better, the literal Neural CPU hypothesis is rejected.

## Teacher boundary

The primary target is the routed MoE sublayer, including its router, shared
expert, routed experts, routing weights, and weighted aggregation. Attention,
DeltaNet, normalization outside the MoE boundary, KV state, and the output head
remain part of the exact surrounding model.

The full protocol selects four contiguous mid-model Qwen MoE sublayers, for
example layers 20--23. Exact model runs record, for each selected layer:

```text
post-attention normalized activation x_l
exact complete MoE residual y_l
layer identity
router expert IDs and scores for diagnostics only
conversation identity and token position
```

The Neural CPU receives the activation and layer identity. It does not receive
the teacher router weights. Route-dependent behavior must be implemented by
the compiled program.

The four functions are first trained from their individual real boundaries.
For the downstream decision, the same Neural CPU replaces all four MoE
sublayers inside the exact model while the intervening attention or DeltaNet
operations remain exact. This avoids the invalid assumption that four MoE
sublayers can be composed offline while ignoring the transformations between
them.

## Teacher data

The full proposal records approximately 500,000 activation/output pairs from
diverse real text:

- multi-turn general chat;
- source code and technical discussion;
- mathematics and structured reasoning;
- multilingual text;
- long-form prose and changing topics.

Data is split by complete conversation and topic, never randomly by token.
Held-out domains and conversations cannot contribute any prefix or near-copy to
training.

Teacher records are immutable and authenticated. Generated traces are research
artifacts and are not committed to Git.

## Machine contract

### Registers and state

- eight vector registers `R0`--`R7`, each with the model hidden width;
- four scalar registers;
- one predicate flag;
- one program counter;
- one bounded loop counter;
- optional vector stack with a fixed maximum depth.

Every inference starts with zeroed state except `R0`, which receives the input
activation, and the declared layer code. No mutable state may cross token or
request boundaries in this experiment. `HALT` returns the declared output
register.

Transient registers count as runtime memory. They may not contain trained
constants and may not be used to retain information between samples.

### Fixed instruction format

One proposed instruction word is 64 bits:

```text
opcode        6 bits
destination   4 bits
source A      4 bits
source B      4 bits
constant ID  20 bits
predicate     8 bits
immediate / reserved remainder
```

The exact packing may change before implementation, but it must be fixed before
training and used by every evaluated configuration.

### Initial ISA

```text
NOP       no operation
HALT      stop and publish the output register
MOV       copy a vector or scalar register
ADD       vector addition
SUB       vector subtraction
SCALE     multiply a vector by a scalar register
RMSN      RMS normalization
LMAT      low-rank matrix operation using counted constants
SVGLU     low-rank projections followed by SiLU/GLU
SMX8      softmax over at most eight scalar registers
MIX       weighted combination of vector registers
CMP       norm or dot-product comparison into the predicate
BRA       bounded relative branch
BNZ       predicate-controlled bounded branch
```

The ISA is intentionally small. It contains generic execution primitives, not
Qwen expert opcodes or hidden calls into the original runtime.

### Program and constant memory

The machine uses a Harvard layout:

```text
hard instruction stream
small constant pool
transient register file
```

The program may contain at most 4,096 instruction words. Every backward branch
is bounded, and a hard maximum tick count forces termination. Indirect access to
an unbounded database is forbidden.

All learned matrices, vectors, embeddings, tables, codes, and immediates reside
in the counted constant pool. Labeling trained data as microcode does not exempt
it from byte accounting.

## Byte accounting

For one compiled model:

```text
model bytes = hard bytecode
            + constant pool
            + retained controller parameters
            + layer/expert/tick tables
            + every other trained persistent value
```

The generic interpreter implementation is disclosed separately, fixed across
all configurations, and contains no trained model data. Baselines receive the
same persistent byte budget.

The proposed full budgets are 4 MiB and 16 MiB. The byte audit reports:

- instruction bytes;
- constant bytes by instruction family;
- controller or embedding bytes;
- quantization metadata;
- transient register memory;
- executed operations per stored byte.

If most bytes are embedded matrices and execution performs little parameter
reuse, the result is classified as a compressed lookup or ordinary surrogate,
not confirmation of the Neural CPU mechanism.

## Compilation and hardening

### Soft compilation

During compilation only, a small controller observes the program counter and
bounded summaries of the register file and emits a distribution over legal
instructions. Straight-through Gumbel selection allows gradient training while
temperature is annealed toward a discrete program.

The loss includes:

```text
normalized MoE-output error
downstream-logit KL from an in-model splice
instruction entropy regularization
program/constant byte penalties
termination and safety penalties
```

### Hard extraction

The highest-probability instruction sequence is extracted as deterministic
bytecode. Opcodes and control flow are frozen. A bounded constant-pool-only
fine-tune is allowed without changing the declared byte budget.

At inference the soft controller is deleted. If any controller remains, its
parameters are counted and the result is not called a fully compiled hard
program.

The soft and hard programs are evaluated independently. More than 20% relative
error regression after hardening is a kill condition.

## Mandatory baselines

Every baseline uses the same teacher records, training split, downstream
evaluation, and persistent-byte budget:

1. a static dense distilled FFN;
2. independent low-rank layer surrogates;
3. a shared-basis plus per-layer or per-expert code surrogate;
4. a parameter-matched recurrent neural block without a compiled ISA.

The fourth baseline distinguishes benefits of ordinary recurrence from benefits
of instruction sequencing and hard control flow.

## Negative controls

### Shuffled program

Execute the same instruction multiset and constants in a shuffled legal order.
The error must degrade by at least an order of magnitude. Otherwise the
constant pool, not the program, carries the function.

### Random program plus trained readout

Freeze random legal bytecode and train only a final readout. If it approaches
the compiled machine, the result is explained by random features or reservoir
behavior rather than a learned program.

### Soft-only controller

Retain the differentiable controller at evaluation as a diagnostic. A large
soft-to-hard gap means the proposed discrete ISA did not capture the learned
computation.

### Instruction and constant lesion

Remove or replace individual instruction regions and constant-pool groups. The
result must exhibit localized, reproducible functional loss rather than uniform
insensitivity.

### Route counterfactual

For a diagnostic variant that explicitly supplies a route code, recompute
teacher outputs under forced or permuted routes. The machine must respond to the
counterfactual route rather than reproduce the average output for the input.
This diagnostic is separate from the primary task, in which routing is internal
to the compiled function.

## Memorization controls

- split by complete conversation and topic;
- reserve code, mathematics, and multilingual domains for OOD evaluation;
- detect exact and near-duplicate prefixes;
- publish train/test nearest-neighbor distances in activation space;
- hold out complete route clusters or expert-heavy regions when enough data is
  available;
- compare degradation against every byte-matched baseline.

Holding out arbitrary expert IDs is not by itself a fair primary metric because
the student receives no representation of unseen expert behavior. It is a
secondary structural test: graceful degradation is evidence of shared
algorithmic structure, while collapse indicates memorization or missing
information.

## Time-for-space sweep

At fixed program and constant bytes, evaluate maximum tick counts:

```text
16, 32, 64, 128, 256
```

Increasing the tick count may not add constants, tick-specific matrices, or
program records. Loops must reuse the same instruction region and constants.

The primary graph is:

```text
held-out error / downstream KL
             versus
executed ticks at fixed persistent bytes
```

The Neural CPU hypothesis requires useful error reduction beyond the shallow
regime. Saturation by 32 ticks means additional clocked execution is not buying
model function for this design.

## Downstream integration

Boundary MSE is a diagnostic, not the decision metric. The compiled machine is
spliced into the exact Qwen runtime in place of the four selected MoE
sublayers. Evaluation includes:

- teacher-forced next-token KL on at least 20,000 held-out tokens;
- greedy token agreement;
- free-running generations of 256 tokens over 200 held-out prompts;
- task quality across general chat, code, mathematics, and multilingual text;
- error and fallback bursts;
- p50 and p99 token latency;
- deterministic repeatability.

No original Qwen expert weight may be read by the Neural CPU during this
evaluation.

## Latency accounting

The primary systems measurement is single-stream execution on the reference
CPU using real hard bytecode:

- cycles per instruction family;
- instructions and branches per sample;
- branch mispredictions;
- cache and persistent-byte working set;
- microseconds per selected MoE replacement;
- projected and measured full-model token time.

An accelerator implementation may be reported separately, but a simulation or
FLOP estimate cannot replace measured interpreter latency.

The initial full-model projection gate is at most 15 ms/token for the Neural
CPU portion, leaving budget for exact attention, state updates, normalization,
and the vocabulary head inside a 33 ms/token target.

## Pre-registered decision

### Go

All conditions are required at the 16 MiB budget:

1. at 128 ticks, downstream KL is at most 80% of the best byte-matched static
   baseline;
2. quality continues improving from 64 to 128 ticks without adding bytes;
3. free-running greedy agreement with the teacher is at least 92%;
4. hardening increases relative error by no more than 20%;
5. shuffled bytecode is at least 10 times worse and random-program/readout is at
   least five times worse;
6. OOD and route-cluster holdouts degrade gracefully relative to baselines;
7. the measured or qualified projected Neural CPU time is at most 15 ms/token;
8. the byte and instruction audit demonstrates material reuse rather than a
   constant-pool lookup table.

Passing is evidence that hard program execution provides a useful time-for-space
axis beyond ordinary distillation for the tested Qwen functions. It is not yet
proof that the entire model can be compiled.

### Kill or redesign

Any condition rejects the tested ISA/compiler configuration:

- held-out error is flat beyond 32 ticks;
- a byte-matched static surrogate is equal or better;
- hardening loses more than 20%;
- shuffled/random controls remain competitive;
- persistent bytes are dominated by effectively unshared lookup matrices;
- free-running drift fails the downstream gates;
- qualified latency exceeds the serving budget.

A failed ISA does not prove that every conceivable stored-program model is
impossible. It does answer the practical hypothesis tested here: this fixed
machine, compiler, byte budget, and instruction range do not provide the
required time-for-space advantage.

## Full-protocol scale

The complete evidence set requires:

- roughly 500,000 teacher records;
- several GiB of generated trace data;
- the complete byte/tick sweep at both byte budgets;
- every baseline, hardening control, and downstream splice;
- measured CPU interpreter qualification.

This is the full NCPU-1 protocol. A smaller NCPU-0 screening experiment may
reuse the same contracts with one MoE layer, fewer traces, one byte budget, and
a fixed hard loop program. NCPU-0 can cheaply reject the core time-for-space
mechanism, but only NCPU-1 supplies the four-layer downstream evidence above.
