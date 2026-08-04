# Larger-model decision record

This is a read-only inventory and model-selection record for the next runtime
target. It does not authorize deletion, download, conversion, or source-shard
reclamation. Measurements were taken on `3090box` on 2026-08-04.

## Storage inventory

The system volume has 930.61 GiB total, 444.13 GiB used, and 486.48 GiB free.
The following assets are protected:

| Asset | Path | Size |
|---|---|---:|
| Original Qwen checkpoint | `C:\Users\vladi\.cache\huggingface\hub\models--Qwen--Qwen3-Next-80B-A3B-Instruct` | 151.510 GiB |
| Validated Expert Pack | `C:\Users\vladi\quantum-llm\work\models\qwen3-next-80b-expert-pack-int8` | 76.326 GiB |

Neither asset is a cleanup candidate. The source remains necessary for a
repack if the format or kernel ABI changes.

Conservative reclaim candidates, before any download:

| Candidate | Measured size | Condition |
|---|---:|---|
| `C:\Windows\Temp` | 0.513 GiB | Delete only files not held by Windows. |
| `C:\Users\vladi\AppData\Local\Temp` | 0.001 GiB | Delete only stale files. |
| `C:\Users\vladi\.cache\huggingface\xet` | 0.235 GiB | Re-creatable transfer cache. |
| `C:\Users\vladi\.cache\huggingface\datasets` | 0.123 GiB | Re-creatable dataset cache. |
| `C:\Users\vladi\quantum-llm\out\build` | 0.018 GiB | Re-creatable build tree. |
| `models--allenai--OLMoE-1B-7B-0125-Instruct` | 12.892 GiB | Only if the OLMoE regression baseline is intentionally retired. |
| project virtual environment | 0.716 GiB | Only if dependency recreation is acceptable. |

The conservative maximum is about 14.50 GiB, of which 12.892 GiB sacrifices a
useful baseline. No deletion is required for the recommended checkpoint.
`C:\Users\vladi\Projects` (91.33 GiB) is explicitly out of scope.

Docker is not counted. Its daemon was stopped and neither the `slm-models`
volume nor a backing VHD was independently resolved. It must be inventoried
before making any cleanup claim.

## Candidate checkpoints

Sizes below are the sum of repository blobs returned by the Hugging Face API,
not marketing parameter estimates.

| Candidate | Total / active | Geometry | Context | Download | License | Decision |
|---|---:|---|---:|---:|---|---|
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 122B / 10B | 48 layers, 256 experts, top-8, hidden 3072 | 262K | 233.014 GiB BF16 | Apache-2.0 | Safest source dtype, but largest Qwen source. |
| [Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) | 122B / 10B | same | 262K | 118.460 GiB | Apache-2.0 | Recommended after FP8 input support. |
| [Qwen3.5-122B-A10B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4) | 122B / 10B | same | 262K | 73.486 GiB | Apache-2.0 | Smallest, but introduces a second quantization ABI and quality provenance. |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | 284B / 13B | 43 layers, 256 routed experts, top-6 + shared, hidden 4096 | 1M | 148.667 GiB mixed FP4/FP8 | MIT | Strong second target; much larger architecture jump. |
| [Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) | 397B / 17B | 60 layers, 512 experts, top-10, hidden 4096 | 262K | 378.302 GiB | Apache-2.0 | Download fits, source plus a new pack has poor safety margin. |
| [Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 1T / 32B | 61 layers, 384 routed experts, top-8 + shared, hidden 7168 | 256K | 554.328 GiB | Modified MIT | Future distributed/storage-sharded target; does not fit current free space. |

DeepSeek-V4-Pro at 1.6T total / 49B active is also a future target; its
checkpoint is larger than current free space. It is not a sensible first
compatibility step.

## Runtime compatibility gap

The current compiler accepts only BF16, F16, and F32 source tensors and only
the `olmoe` and `qwen3_next` adapters. The CUDA runner also validates the exact
Qwen3-Next 80B geometry. Therefore none of the candidates is runnable merely
by downloading it.

Qwen3.5-122B is the smallest useful architecture step:

- add a `qwen3_5_moe` adapter and explicitly select the language model from the
  multimodal wrapper;
- reject or intentionally preserve vision and MTP tensors rather than silently
  dropping them;
- support hidden 3072, 48 layers, 256 experts, top-8 and the Qwen3.5 hybrid
  attention layout;
- for the recommended checkpoint, decode block-scaled FP8 while streaming into
  the existing per-row INT8 Expert Pack, or define a new compute-ready ABI;
- qualify tokenizer/chat template, exact routing, dense kernels, KV layout and
  output against a trusted reference implementation.

DeepSeek-V4-Flash additionally requires a `deepseek_v4` adapter, mixed FP4/FP8
source decoding, CSA + HCA attention, manifold-constrained hyper-connections,
and the model's routing/shared-expert semantics. It is a deliberate second
backend milestone, not a small repack.

## Pack space and time estimate

The existing 80B conversion is measured, not inferred:

- 162,659,161,528 source tensor bytes;
- 81,903,198,208 output pack bytes;
- 3,371.421 seconds (56.19 minutes);
- no source shards reclaimed.

Scaling the same INT8-per-row format by parameter count gives a planning range
of 115–125 GiB and 85–100 minutes for Qwen3.5-122B after its adapter and source
decoder exist. This is an engineering estimate; the first representative
dense tensor and one complete routed layer must measure decode rate, output
size, and numerical error before a full conversion starts.

With 486.48 GiB currently free, either the 118.460 GiB FP8 source plus the
estimated pack, or the 233.014 GiB BF16 source plus the estimated pack, fits
while retaining both protected 80B assets. FP8 leaves materially more room for
atomic `.partial` output, validation, logs, and failure recovery.

## Recommendation and approval boundary

1. Implement and test the Qwen3.5 adapter plus streaming FP8 source decode on a
   representative shard; do not download the full checkpoint merely to test
   storage.
2. If the vertical slice preserves the existing pack ABI and numerical
   tolerance, download `Qwen/Qwen3.5-122B-A10B-FP8` at a pinned revision.
3. Keep the current Qwen3-Next source and pack. Produce the new pack atomically,
   validate every record independently, then run correctness/API gates before
   any performance claim.
4. Take DeepSeek-V4-Flash next because 284B total / 13B active exercises a
   genuinely larger directory and a different modern architecture.

Cleanup and download remain separate operator approvals. In particular,
`DELETE_CONSUMED_SHARDS` must stay disabled for both the protected source and
the next checkpoint until repacking and rollback are independently proven.
