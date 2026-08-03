# Expert Pack compiler

This directory contains the P1 compiler and an independent container validator.
It reads a local SafeTensors checkpoint one tensor at a time through read-only
memory mappings; it does not instantiate a Transformers model or materialize the
whole checkpoint.

The first explicit adapter is `olmoe`, verified against
`allenai/OLMoE-1B-7B-0125-Instruct`. The v1 quantization profile is symmetric
INT8 with one FP32 scale per output row. Expert `gate` and `up` rows are fused
in that order, followed by the `down` projection. Router matrices and rank-one
dense tensors remain FP32. Every record and section layout is declared in the
manifest; readers must not infer it from byte counts.

Run from the repository root:

```text
python -m compiler compile \
  --source C:\path\to\snapshot \
  --output C:\path\to\olmoe-expert-pack \
  --source-id allenai/OLMoE-1B-7B-0125-Instruct \
  --source-revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e

python -m compiler validate C:\path\to\olmoe-expert-pack
```

If conversion is interrupted, the final output directory is absent and a
neighboring `<output>.partial` directory remains. Resume only with identical
source bytes and options:

```text
python -m compiler compile --source ... --output ... --resume
```

Resume commits at complete pack boundaries. Uncommitted `.tmp` packs are
discarded. A directory without a valid `COMPLETED` marker is rejected.

The implementation has a dependency-free, row-streaming conversion path and
uses NumPy automatically when the optional `fast` dependency is installed.
Both paths produce the same quant ABI; the dependency-free path is correct but
substantially slower for large checkpoints.
