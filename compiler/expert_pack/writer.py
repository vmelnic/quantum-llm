"""Low-level Expert Pack record writers.

The writer only appends records.  Publication, resume state, and manifest
construction live in ``compile.py`` so this module remains a small ABI surface.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .adapters import ExpertSource
from .constants import (
    DENSE_HEADER_STRUCT,
    DENSE_MAGIC,
    EXPERT_HEADER_STRUCT,
    EXPERT_MAGIC,
    FLAG_GATE_UP_FUSED,
    FLAG_PER_ROW_SCALES,
    FLAG_ROW_MAJOR,
    FLAG_SYMMETRIC,
    FORMAT_VERSION,
    FP4_QUANT_ABI_ID,
    FP4_QUANT_GROUP_SIZE,
    HEADER_BYTES,
    PACK_ALIGNMENT,
    QUANT_ABI_ID,
    SECTION_ALIGNMENT,
)
from .quant import write_float32, write_fp4_block32_rows, write_int8_rows
from .safetensors import SafeTensorCheckpoint, TensorInfo
from .util import align_up, fsync_file, sha256_bytes, write_all, write_zeros


@dataclass(frozen=True)
class RecordResult:
    entry: dict[str, object]
    end_offset: int


def _pad_to(
    handle: BinaryIO,
    absolute_target: int,
    digest: object | None = None,
) -> None:
    current = handle.tell()
    if current > absolute_target:
        raise RuntimeError(f"writer passed target offset {absolute_target}")
    write_zeros(handle, absolute_target - current, digest)


def _shape5(shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    padded = shape + (0,) * (5 - len(shape))
    if len(padded) != 5:
        raise ValueError(f"tensor rank exceeds dense header capacity: {shape}")
    return padded


def dense_is_quantized(
    info: TensorInfo, preserve_float32: bool = False
) -> bool:
    # Semantic roles come from the architecture adapter. Tensor paths are not
    # a stable cross-model ABI and must not decide whether a router is lossy.
    # Rank-one norms/biases remain FP32; other matrices default to INT8.
    return len(info.shape) == 2 and not preserve_float32


def dense_record_size(
    info: TensorInfo,
    alignment: int = PACK_ALIGNMENT,
    preserve_float32: bool = False,
    quant_abi: int = QUANT_ABI_ID,
) -> int:
    elements = 1
    for dimension in info.shape:
        elements *= dimension
    quantized = (
        not preserve_float32
        and (
            quant_abi == FP4_QUANT_ABI_ID
            or dense_is_quantized(info, preserve_float32)
        )
    )
    if quantized and quant_abi == FP4_QUANT_ABI_ID:
        rows = math.prod(info.shape[:-1]) if len(info.shape) > 1 else 1
        columns = info.shape[-1]
        padded_columns = align_up(columns, FP4_QUANT_GROUP_SIZE)
        data_bytes = rows * padded_columns // 2
        scale_bytes = rows * padded_columns // FP4_QUANT_GROUP_SIZE
    elif quantized and quant_abi == QUANT_ABI_ID:
        data_bytes = elements
        scale_bytes = info.shape[0] * 4
    elif quantized:
        raise ValueError(f"unsupported dense quant ABI {quant_abi}")
    else:
        data_bytes = elements * 4
        scale_bytes = 0
    cursor = HEADER_BYTES + data_bytes
    if scale_bytes:
        cursor = align_up(cursor, SECTION_ALIGNMENT) + scale_bytes
    return align_up(cursor, alignment)


def _check_fp4_geometry(hidden: int, intermediate: int) -> None:
    if hidden % FP4_QUANT_GROUP_SIZE or intermediate % FP4_QUANT_GROUP_SIZE:
        raise ValueError(
            f"FP4 block-{FP4_QUANT_GROUP_SIZE} requires hidden/intermediate "
            f"multiples of {FP4_QUANT_GROUP_SIZE}, got {hidden}/{intermediate}"
        )


def expert_record_size(
    hidden: int,
    intermediate: int,
    alignment: int = PACK_ALIGNMENT,
    quant_abi: int = QUANT_ABI_ID,
) -> int:
    cursor = HEADER_BYTES
    if quant_abi == FP4_QUANT_ABI_ID:
        _check_fp4_geometry(hidden, intermediate)
        cursor += intermediate * hidden  # 2 * I * H values at 4 bits
        cursor = align_up(cursor, SECTION_ALIGNMENT)
        cursor += 2 * intermediate * hidden // FP4_QUANT_GROUP_SIZE
        cursor = align_up(cursor, SECTION_ALIGNMENT)
        cursor += hidden * intermediate // 2
        cursor = align_up(cursor, SECTION_ALIGNMENT)
        cursor += hidden * intermediate // FP4_QUANT_GROUP_SIZE
        return align_up(cursor, alignment)
    if quant_abi != QUANT_ABI_ID:
        raise ValueError(f"unsupported expert quant ABI {quant_abi}")
    cursor += 2 * intermediate * hidden
    cursor = align_up(cursor, SECTION_ALIGNMENT)
    cursor += 2 * intermediate * 4
    cursor = align_up(cursor, SECTION_ALIGNMENT)
    cursor += hidden * intermediate
    cursor = align_up(cursor, SECTION_ALIGNMENT)
    cursor += hidden * 4
    return align_up(cursor, alignment)


def write_dense_record(
    handle: BinaryIO,
    checkpoint: SafeTensorCheckpoint,
    info: TensorInfo,
    pack_name: str,
    alignment: int = PACK_ALIGNMENT,
    preserve_float32: bool = False,
    quant_abi: int = QUANT_ABI_ID,
) -> RecordResult:
    start = handle.tell()
    if start % alignment:
        raise RuntimeError("dense record start is not pack-aligned")
    write_zeros(handle, HEADER_BYTES)
    digest = hashlib.sha256()
    data_offset = HEADER_BYTES
    quantized = (
        not preserve_float32
        and (
            quant_abi == FP4_QUANT_ABI_ID
            or dense_is_quantized(info, preserve_float32)
        )
    )

    with checkpoint.open_tensor(info.name) as view:
        if quantized:
            if quant_abi == FP4_QUANT_ABI_ID:
                before = handle.tell()
                scales = write_fp4_block32_rows(view, handle, digest)
                data_bytes = handle.tell() - before
            elif quant_abi == QUANT_ABI_ID:
                scales = write_int8_rows(view, handle, digest)
                data_bytes = info.nbytes // (
                    2 if info.dtype in {"F16", "BF16"} else 4
                )
            else:
                raise ValueError(f"unsupported dense quant ABI {quant_abi}")
        else:
            data_bytes = write_float32(view, handle, digest)
            scales = b""

    scale_offset = 0
    scale_bytes = 0
    if scales:
        scale_offset = align_up(handle.tell() - start, SECTION_ALIGNMENT)
        _pad_to(handle, start + scale_offset, digest)
        write_all(handle, scales)
        digest.update(scales)
        scale_bytes = len(scales)

    record_bytes = align_up(handle.tell() - start, alignment)
    _pad_to(handle, start + record_bytes, digest)
    payload_hash = digest.digest()
    flags = FLAG_ROW_MAJOR
    stored_quant_abi = 0
    stored_dtype = "F32"
    layout = "row-major-f32"
    if quantized:
        flags |= FLAG_SYMMETRIC | FLAG_PER_ROW_SCALES
        stored_quant_abi = quant_abi
        if quant_abi == FP4_QUANT_ABI_ID:
            stored_dtype = "FP4_E2M1"
            layout = "row-major-fp4-e2m1-ue8m0-block32-padded"
        else:
            stored_dtype = "I8"
            layout = "output-major-row-contiguous-int8"
    dimensions = _shape5(info.shape)
    header = DENSE_HEADER_STRUCT.pack(
        DENSE_MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        flags,
        stored_quant_abi,
        len(info.shape),
        *dimensions,
        record_bytes,
        data_offset,
        data_bytes,
        scale_offset,
        scale_bytes,
        hashlib.sha256(info.name.encode("utf-8")).digest(),
        payload_hash,
    )
    handle.seek(start)
    write_all(handle, header)
    write_zeros(handle, HEADER_BYTES - len(header))
    handle.seek(start + record_bytes)

    elements = 1
    for dimension in info.shape:
        elements *= dimension
    entry: dict[str, object] = {
        "name": info.name,
        "pack": pack_name,
        "offset": start,
        "stored_bytes": record_bytes,
        "source_bytes": info.nbytes,
        "decoded_bytes": elements * 4,
        "source_dtype": info.dtype,
        "source_shape": list(info.shape),
        "stored_dtype": stored_dtype,
        "layout": layout,
        "quant_abi": stored_quant_abi,
        "payload_sha256": payload_hash.hex(),
        "sections": {
            "data": {"offset": data_offset, "bytes": data_bytes},
            "scales": {"offset": scale_offset, "bytes": scale_bytes},
        },
    }
    return RecordResult(entry=entry, end_offset=start + record_bytes)


def write_expert_record(
    handle: BinaryIO,
    checkpoint: SafeTensorCheckpoint,
    expert: ExpertSource,
    pack_name: str,
    hidden: int,
    intermediate: int,
    alignment: int = PACK_ALIGNMENT,
    quant_abi: int = QUANT_ABI_ID,
) -> RecordResult:
    fp4 = quant_abi == FP4_QUANT_ABI_ID
    if not fp4 and quant_abi != QUANT_ABI_ID:
        raise ValueError(f"unsupported expert quant ABI {quant_abi}")
    if fp4:
        _check_fp4_geometry(hidden, intermediate)
    start = handle.tell()
    if start % alignment:
        raise RuntimeError("expert record start is not pack-aligned")
    write_zeros(handle, HEADER_BYTES)
    digest = hashlib.sha256()

    def encode(view, destination, record_digest) -> bytes:
        if fp4:
            return write_fp4_block32_rows(view, destination, record_digest)
        return write_int8_rows(view, destination, record_digest)

    gate_up_q_offset = HEADER_BYTES
    with checkpoint.open_tensor(expert.gate.name) as gate:
        gate_scales = encode(gate, handle, digest)
    with checkpoint.open_tensor(expert.up.name) as up:
        up_scales = encode(up, handle, digest)
    if fp4:
        gate_up_q_bytes = intermediate * hidden
    else:
        gate_up_q_bytes = 2 * intermediate * hidden

    gate_up_scale_offset = align_up(handle.tell() - start, SECTION_ALIGNMENT)
    _pad_to(handle, start + gate_up_scale_offset, digest)
    gate_up_scales = gate_scales + up_scales
    write_all(handle, gate_up_scales)
    digest.update(gate_up_scales)
    gate_up_scale_bytes = len(gate_up_scales)

    down_q_offset = align_up(handle.tell() - start, SECTION_ALIGNMENT)
    _pad_to(handle, start + down_q_offset, digest)
    with checkpoint.open_tensor(expert.down.name) as down:
        down_scales = encode(down, handle, digest)
    down_q_bytes = hidden * intermediate // 2 if fp4 else hidden * intermediate

    down_scale_offset = align_up(handle.tell() - start, SECTION_ALIGNMENT)
    _pad_to(handle, start + down_scale_offset, digest)
    write_all(handle, down_scales)
    digest.update(down_scales)
    down_scale_bytes = len(down_scales)

    record_bytes = align_up(handle.tell() - start, alignment)
    _pad_to(handle, start + record_bytes, digest)
    payload_hash = digest.digest()
    flags = FLAG_ROW_MAJOR | FLAG_GATE_UP_FUSED | FLAG_SYMMETRIC | FLAG_PER_ROW_SCALES
    header = EXPERT_HEADER_STRUCT.pack(
        EXPERT_MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        flags,
        quant_abi,
        expert.layer,
        expert.expert,
        hidden,
        intermediate,
        2 * intermediate,
        0,
        record_bytes,
        gate_up_q_offset,
        gate_up_q_bytes,
        gate_up_scale_offset,
        gate_up_scale_bytes,
        down_q_offset,
        down_q_bytes,
        down_scale_offset,
        down_scale_bytes,
        payload_hash,
    )
    handle.seek(start)
    write_all(handle, header)
    write_zeros(handle, HEADER_BYTES - len(header))
    handle.seek(start + record_bytes)

    source_bytes = expert.gate.nbytes + expert.up.nbytes + expert.down.nbytes
    decoded_bytes = (3 * hidden * intermediate) * 4
    entry: dict[str, object] = {
        "layer": expert.layer,
        "expert": expert.expert,
        "pack": pack_name,
        "offset": start,
        "stored_bytes": record_bytes,
        "source_bytes": source_bytes,
        "decoded_bytes": decoded_bytes,
        "source_dtype": {
            "gate": expert.gate.dtype,
            "up": expert.up.dtype,
            "down": expert.down.dtype,
        },
        "source_shape": {
            "gate": list(expert.gate.shape),
            "up": list(expert.up.shape),
            "down": list(expert.down.shape),
        },
        "stored_dtype": "FP4_E2M1" if fp4 else "I8",
        "layout": (
            "gate-rows-then-up-rows;down-output-major;row-contiguous-fp4-e2m1-block32"
            if fp4
            else "gate-rows-then-up-rows;down-output-major;row-contiguous-int8"
        ),
        "quant_abi": quant_abi,
        "payload_sha256": payload_hash.hex(),
        "source_tensors": {
            "gate": expert.gate.name,
            "up": expert.up.name,
            "down": expert.down.name,
        },
        "sections": {
            "gate_up_q": {"offset": gate_up_q_offset, "bytes": gate_up_q_bytes},
            "gate_up_scales": {
                "offset": gate_up_scale_offset,
                "bytes": gate_up_scale_bytes,
            },
            "down_q": {"offset": down_q_offset, "bytes": down_q_bytes},
            "down_scales": {"offset": down_scale_offset, "bytes": down_scale_bytes},
        },
    }
    return RecordResult(entry=entry, end_offset=start + record_bytes)


def durable_close(handle: BinaryIO) -> None:
    fsync_file(handle)
    handle.close()


def truncate_to(path: Path, size: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(size)
        fsync_file(handle)
