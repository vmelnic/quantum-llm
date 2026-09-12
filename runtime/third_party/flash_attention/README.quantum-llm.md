# Vendored FlashAttention forward kernel

This directory contains the minimal header closure used by the Quantum LLM
runtime's SM86 BF16 forward-attention adapter.

- FlashAttention commit: `0f3fb00d3f34196ca55f30d2f6c5dce0709b4667`
- CUTLASS commit: `7127592069c2fe01b041e174ba4345ef9b279671`
- Imported files: the dependency closure of `flash_fwd_launch_template.h` for
  the head-dimension-256 BF16 forward kernel only.
- Upstream licenses are retained in `LICENSE` and `csrc/cutlass/LICENSE.txt`.

The runtime does not depend on PyTorch. The local `shim` supplies only the two
CUDA error-check macros required by the upstream launch header; dropout,
ALiBi, soft-cap and local-window template branches are disabled for this
adapter. The runtime keeps its own validation, causal segmentation and
log-sum-exp merge.
