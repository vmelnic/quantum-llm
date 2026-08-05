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
MAX_TENSOR_RANK = 4

HASH_ALGORITHM = "sha256"
QUANT_PROFILE = "int8-symmetric-per-row-v1"
QUANT_ABI_ID = 1
QUANT_GROUP_SIZE = 0  # 0 means one scale per complete output row.

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
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)
