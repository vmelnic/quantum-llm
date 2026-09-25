"""Wire-format constants kept isolated until the P0 ABI is frozen.

Changing a value in this module is an on-disk ABI change.  Readers never infer
these values from record sizes.
"""

from __future__ import annotations

import struct

FORMAT_NAME = "expert-pack"
FORMAT_VERSION = 1
MIN_RUNTIME_VERSION = "0.1.0"
MANIFEST_SCHEMA = "expert-pack-manifest-v1"

PACK_ALIGNMENT = 4096
SECTION_ALIGNMENT = 256
HEADER_BYTES = 256
MAX_TENSOR_RANK = 5

HASH_ALGORITHM = "sha256"
QUANT_PROFILE = "int8-symmetric-per-row-v1"
QUANT_ABI_ID = 1
QUANT_GROUP_SIZE = 0  # 0 means one scale per complete output row.

# FP4 E2M1 payload with one UE8M0 scale per 32-value block along each output
# row, matching the device format already used by the DeepSeek compact path.
# Scale codes are clamped to [1, 254]: 255 is the UE8M0 NaN and code 0 decodes
# inconsistently between the toolchain decoder and the CUDA kernel, so the
# compiler never emits it.
FP4_QUANT_PROFILE = "fp4-e2m1-ue8m0-block32-v1"
FP4_MSE_QUANT_PROFILE = "fp4-e2m1-ue8m0-block32-mse-v2"
FP4_ACTIVATION_QUANT_PROFILE = (
    "fp4-e2m1-ue8m0-block32-activation-aware-v3"
)
FP4_ACTIVATION_CODE_QUANT_PROFILE = (
    "fp4-e2m1-ue8m0-block32-activation-codes-v4"
)
FP4_QUANT_ABI_ID = 3
# Routed ReLU2 experts use the same FP4 payload encoding as ABI 3, but their
# executable record contains only up/down matrices.  Keeping a distinct ABI
# prevents a SwiGLU runtime from interpreting the first section as gate+up.
FP4_RELU2_EXPERT_ABI_ID = 4
FP4_QUANT_GROUP_SIZE = 32
FP4_UE8M0_MIN_CODE = 1
FP4_UE8M0_MAX_CODE = 254

# OCP MXFP6 E3M2 payload with one UE8M0 scale per 32-value block. Four
# six-bit elements are packed little-endian into three bytes. This dense-only
# ABI is used for precision-sensitive semantic roles; routed experts retain
# their declared expert ABI.
MXFP6_QUANT_PROFILE = "mxfp6-e3m2-ue8m0-block32-v1"
MXFP6_QUANT_ABI_ID = 6
MXFP6_QUANT_GROUP_SIZE = 32

# Native NVIDIA FP4 checkpoint encoding. Packed E2M1 values use one E4M3FN
# scale per 16 input values plus checkpoint-stored FP32 global divisors for
# weights and activations. Unlike the block-32 profile above, this ABI copies
# the source payload bit-for-bit and preserves its W4A4 arithmetic.
NVFP4_QUANT_PROFILE = "nvfp4-e2m1-e4m3fn-block16-w4a4-v1"
NVFP4_QUANT_ABI_ID = 5
NVFP4_QUANT_GROUP_SIZE = 16

FP4_QUANT_PROFILES = (
    FP4_QUANT_PROFILE,
    FP4_MSE_QUANT_PROFILE,
    FP4_ACTIVATION_QUANT_PROFILE,
    FP4_ACTIVATION_CODE_QUANT_PROFILE,
)
QUANT_PROFILES = (QUANT_PROFILE, *FP4_QUANT_PROFILES, NVFP4_QUANT_PROFILE)

DENSE_MAGIC = b"EPDENS01"
EXPERT_MAGIC = b"EPEXPR01"
DENSE_RECORD_KIND = 1
EXPERT_RECORD_KIND = 2

FLAG_ROW_MAJOR = 1 << 0
FLAG_GATE_UP_FUSED = 1 << 1
FLAG_SYMMETRIC = 1 << 2
FLAG_PER_ROW_SCALES = 1 << 3

# Dense fields:
# magic, version, header bytes, flags, quant ABI, rank, four dimensions,
# record bytes, data offset/bytes, scale offset/bytes, source-name SHA-256,
# payload SHA-256.
DENSE_HEADER_STRUCT = struct.Struct("<8sHHIIIIIIIIQQQQQ32s32s")

# Expert fields:
# magic, version, header bytes, flags, quant ABI, layer, expert, hidden,
# intermediate, fused rows, reserved, record bytes, then four offset/length
# pairs (gate_up q/scales and down q/scales), payload SHA-256.
EXPERT_HEADER_STRUCT = struct.Struct("<8sHHIIiiIIIIQQQQQQQQQ32s")

if DENSE_HEADER_STRUCT.size > HEADER_BYTES:
    raise RuntimeError("dense header struct exceeds fixed header")
if EXPERT_HEADER_STRUCT.size > HEADER_BYTES:
    raise RuntimeError("expert header struct exceeds fixed header")

DTYPE_BYTES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
    # SafeTensors stores each supported float8 value in one byte.  Keep these
    # source dtypes distinct: their bit layouts and scale semantics are not
    # interchangeable even though their storage widths match.
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}

MODEL_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)

TOKENIZER_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)
