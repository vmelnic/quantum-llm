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
    FP4_RELU2_EXPERT_ABI_ID,
    FP4_QUANT_GROUP_SIZE,
    NVFP4_QUANT_ABI_ID,
    NVFP4_QUANT_GROUP_SIZE,
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
    preserve_int64: bool = False,
    preserve_bfloat16: bool = False,
    native_nvfp4=None,
) -> int:
    if native_nvfp4 is not None:
        rows, columns = native_nvfp4.logical_shape
        data_bytes = rows * columns // 2
        scale_bytes = (
            rows * columns // NVFP4_QUANT_GROUP_SIZE +
            native_nvfp4.weight_global_scale.nbytes +
            native_nvfp4.input_global_scale.nbytes
        )
        return align_up(
            align_up(HEADER_BYTES + data_bytes, SECTION_ALIGNMENT) +
            scale_bytes,
            alignment,
        )
    elements = 1
    for dimension in info.shape:
        elements *= dimension
    if preserve_int64 and info.dtype != "I64":
        raise ValueError(f"raw I64 storage requires an I64 source: {info.name}")
    if preserve_bfloat16 and info.dtype != "BF16":
        raise ValueError(f"raw BF16 storage requires a BF16 source: {info.name}")
    if preserve_int64 and preserve_bfloat16:
        raise ValueError("dense tensor cannot preserve two raw encodings")
    quantized = (
        not preserve_int64
        and not preserve_bfloat16
        and
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
        data_bytes = elements * (
            8 if preserve_int64 else 2 if preserve_bfloat16 else 4
        )
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
    gated: bool = True,
) -> int:
    cursor = HEADER_BYTES
    if quant_abi == NVFP4_QUANT_ABI_ID:
        if not gated:
            raise ValueError("native NVFP4 expert ABI requires SwiGLU")
        _check_fp4_geometry(hidden, intermediate)
        elements = hidden * intermediate
        cursor += elements
        cursor = align_up(cursor, SECTION_ALIGNMENT)
        cursor += 2 * (elements // NVFP4_QUANT_GROUP_SIZE + 8)
        cursor = align_up(cursor, SECTION_ALIGNMENT)
        cursor += elements // 2
        cursor = align_up(cursor, SECTION_ALIGNMENT)
        cursor += elements // NVFP4_QUANT_GROUP_SIZE + 8
        return align_up(cursor, alignment)
    if quant_abi in (FP4_QUANT_ABI_ID, FP4_RELU2_EXPERT_ABI_ID):
        if (quant_abi == FP4_RELU2_EXPERT_ABI_ID) == gated:
            raise ValueError("expert record ABI disagrees with expert mathematics")
        _check_fp4_geometry(hidden, intermediate)
        cursor += (2 if gated else 1) * intermediate * hidden // 2
        cursor = align_up(cursor, SECTION_ALIGNMENT)
        cursor += ((2 if gated else 1) * intermediate * hidden
                   // FP4_QUANT_GROUP_SIZE)
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
    preserve_int64: bool = False,
    preserve_bfloat16: bool = False,
    native_nvfp4=None,
) -> RecordResult:
    start = handle.tell()
    if start % alignment:
        raise RuntimeError("dense record start is not pack-aligned")
    write_zeros(handle, HEADER_BYTES)
    digest = hashlib.sha256()
    data_offset = HEADER_BYTES
    if preserve_int64 and (preserve_float32 or preserve_bfloat16 or
                           native_nvfp4 is not None or info.dtype != "I64"):
        raise ValueError(f"invalid raw I64 storage declaration for {info.name}")
    if preserve_bfloat16 and (preserve_float32 or native_nvfp4 is not None or
                              info.dtype != "BF16"):
        raise ValueError(f"invalid raw BF16 storage declaration for {info.name}")
    quantized = (
        not preserve_int64
        and not preserve_bfloat16
        and native_nvfp4 is None
        and
        not preserve_float32
        and (
            quant_abi == FP4_QUANT_ABI_ID
            or dense_is_quantized(info, preserve_float32)
        )
    )

    logical_shape = info.shape
    source_bytes = info.nbytes
    source_tensors: dict[str, str] | None = None
    if native_nvfp4 is not None:
        if quant_abi != NVFP4_QUANT_ABI_ID or native_nvfp4.weight != info:
            raise ValueError(f"invalid native NVFP4 declaration for {info.name}")
        rows, columns = native_nvfp4.logical_shape
        if (columns % NVFP4_QUANT_GROUP_SIZE or info.dtype != "U8" or
                info.shape != (rows, columns // 2) or
                native_nvfp4.scale.dtype != "F8_E4M3" or
                native_nvfp4.scale.shape !=
                    (rows, columns // NVFP4_QUANT_GROUP_SIZE) or
                native_nvfp4.weight_global_scale.dtype != "F32" or
                native_nvfp4.weight_global_scale.shape != (1,) or
                native_nvfp4.input_global_scale.dtype not in {"F32", "BF16"} or
                native_nvfp4.input_global_scale.shape != (1,)):
            raise ValueError(f"native NVFP4 source geometry is invalid: {info.name}")
        with checkpoint.open_tensor(info) as view:
            payload = bytes(view.raw)
        write_all(handle, payload)
        digest.update(payload)
        data_bytes = len(payload)
        sidecars = bytearray()
        source_tensors = {}
        for role, tensor in (
            ("scale", native_nvfp4.scale),
            ("weight_global_scale", native_nvfp4.weight_global_scale),
            ("input_global_scale", native_nvfp4.input_global_scale),
        ):
            with checkpoint.open_tensor(tensor) as view:
                sidecars.extend(view.raw)
            source_bytes += tensor.nbytes
            source_tensors[role] = tensor.name
        source_tensors["weight"] = info.name
        scales = bytes(sidecars)
        logical_shape = native_nvfp4.logical_shape
    else:
        with checkpoint.open_tensor(info) as view:
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
            elif preserve_int64 or preserve_bfloat16:
                payload = bytes(view.raw)
                write_all(handle, payload)
                digest.update(payload)
                data_bytes = len(payload)
                scales = b""
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
    if preserve_int64:
        stored_dtype = "I64"
        layout = "row-major-i64-le"
    elif preserve_bfloat16:
        stored_dtype = "BF16"
        layout = "row-major-bfloat16-le"
    elif native_nvfp4 is not None:
        flags |= FLAG_SYMMETRIC
        stored_quant_abi = NVFP4_QUANT_ABI_ID
        stored_dtype = "FP4_E2M1"
        layout = "row-major-nvfp4-e2m1-e4m3fn-block16-w4a4"
    elif quantized:
        flags |= FLAG_SYMMETRIC | FLAG_PER_ROW_SCALES
        stored_quant_abi = quant_abi
        if quant_abi == FP4_QUANT_ABI_ID:
            stored_dtype = "FP4_E2M1"
            layout = "row-major-fp4-e2m1-ue8m0-block32-padded"
        else:
            stored_dtype = "I8"
            layout = "output-major-row-contiguous-int8"
    dimensions = _shape5(logical_shape)
    header = DENSE_HEADER_STRUCT.pack(
        DENSE_MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        flags,
        stored_quant_abi,
        len(logical_shape),
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
    for dimension in logical_shape:
        elements *= dimension
    entry: dict[str, object] = {
        "name": info.name,
        "pack": pack_name,
        "offset": start,
        "stored_bytes": record_bytes,
        "source_bytes": source_bytes,
        "decoded_bytes": elements * (
            8 if preserve_int64 else 2 if preserve_bfloat16 else 4
        ),
        "source_dtype": info.dtype,
        "source_shape": list(logical_shape),
        "stored_dtype": stored_dtype,
        "layout": layout,
        "quant_abi": stored_quant_abi,
        "payload_sha256": payload_hash.hex(),
        "sections": {
            "data": {"offset": data_offset, "bytes": data_bytes},
            "scales": {"offset": scale_offset, "bytes": scale_bytes},
        },
    }
    if source_tensors is not None:
        entry["source_tensors"] = source_tensors
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
    native_nvfp4 = quant_abi == NVFP4_QUANT_ABI_ID
    fp4 = quant_abi in (
        FP4_QUANT_ABI_ID, FP4_RELU2_EXPERT_ABI_ID, NVFP4_QUANT_ABI_ID
    )
    gated = expert.gate is not None
    if (quant_abi == FP4_RELU2_EXPERT_ABI_ID) == gated:
        raise ValueError("expert record ABI disagrees with expert mathematics")
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
        if native_nvfp4:
            raise ValueError("native NVFP4 must be copied with its sidecars")
        if fp4:
            return write_fp4_block32_rows(view, destination, record_digest)
        return write_int8_rows(view, destination, record_digest)

    def copy_native(matrix) -> bytes:
        if matrix is None:
            raise ValueError("native NVFP4 expert sidecar is absent")
        rows, columns = matrix.logical_shape
        if (columns % NVFP4_QUANT_GROUP_SIZE or
                matrix.weight.dtype != "U8" or
                matrix.weight.shape != (rows, columns // 2) or
                matrix.scale.dtype != "F8_E4M3" or
                matrix.scale.shape !=
                    (rows, columns // NVFP4_QUANT_GROUP_SIZE) or
                matrix.weight_global_scale.dtype != "F32" or
                matrix.weight_global_scale.shape != (1,) or
                matrix.input_global_scale.dtype != "F32" or
                matrix.input_global_scale.shape != (1,)):
            raise ValueError("native NVFP4 expert geometry is invalid")
        with checkpoint.open_tensor(matrix.weight) as view:
            payload = bytes(view.raw)
        write_all(handle, payload)
        digest.update(payload)
        sidecars = bytearray()
        for tensor in (matrix.scale, matrix.weight_global_scale,
                       matrix.input_global_scale):
            with checkpoint.open_tensor(tensor) as view:
                sidecars.extend(view.raw)
        return bytes(sidecars)

    gate_up_q_offset = HEADER_BYTES
    gate_scales = b""
    if native_nvfp4:
        gate_scales = copy_native(expert.nvfp4_gate)
        up_scales = copy_native(expert.nvfp4_up)
    else:
        if expert.gate is not None:
            with checkpoint.open_tensor(expert.gate) as gate:
                gate_scales = encode(gate, handle, digest)
        with checkpoint.open_tensor(expert.up) as up:
            up_scales = encode(up, handle, digest)
    if fp4:
        gate_up_q_bytes = (2 if gated else 1) * intermediate * hidden // 2
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
    if native_nvfp4:
        down_scales = copy_native(expert.nvfp4_down)
    else:
        with checkpoint.open_tensor(expert.down) as down:
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
    flags = FLAG_ROW_MAJOR | FLAG_SYMMETRIC
    if not native_nvfp4:
        flags |= FLAG_PER_ROW_SCALES
    if gated:
        flags |= FLAG_GATE_UP_FUSED
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
        (2 if gated else 1) * intermediate,
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

    source_roles = (("gate", expert.gate),) if expert.gate is not None else ()
    source_roles += (("up", expert.up), ("down", expert.down))
    source_bytes = sum(tensor.nbytes for _, tensor in source_roles)
    if native_nvfp4:
        source_bytes = sum(
            tensor.nbytes
            for matrix in (
                expert.nvfp4_gate, expert.nvfp4_up, expert.nvfp4_down
            )
            if matrix is not None
            for tensor in (
                matrix.weight, matrix.scale, matrix.weight_global_scale,
                matrix.input_global_scale,
            )
        )
    decoded_bytes = ((3 if gated else 2) * hidden * intermediate) * 4
    entry: dict[str, object] = {
        "layer": expert.layer,
        "expert": expert.expert,
        "pack": pack_name,
        "offset": start,
        "stored_bytes": record_bytes,
        "source_bytes": source_bytes,
        "decoded_bytes": decoded_bytes,
        "source_dtype": {role: tensor.dtype for role, tensor in source_roles},
        "source_shape": {
            role: list(tensor.shape) for role, tensor in source_roles
        },
        "stored_dtype": "FP4_E2M1" if fp4 else "I8",
        "layout": (
            ("gate-rows-then-up-rows;" if gated else "up-rows;")
            + (
                "down-output-major;row-contiguous-"
                "nvfp4-e2m1-e4m3fn-block16-w4a4"
                if native_nvfp4 else
                "down-output-major;row-contiguous-fp4-e2m1-block32"
            )
            if fp4
            else "gate-rows-then-up-rows;down-output-major;row-contiguous-int8"
        ),
        "quant_abi": quant_abi,
        "payload_sha256": payload_hash.hex(),
        "source_tensors": {
            role: tensor.name for role, tensor in source_roles
        },
        "source_regions": {
            role: {
                "tensor": tensor.physical_name,
                "byte_offset": tensor.source_byte_offset,
                "bytes": tensor.nbytes,
                "shape": list(tensor.shape),
                "tensor_bytes": checkpoint.tensors[tensor.physical_name].nbytes,
                "tensor_shape": list(checkpoint.tensors[tensor.physical_name].shape),
            }
            for role, tensor in source_roles
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
    if native_nvfp4:
        entry["nvfp4_sidecars"] = {
            role: {
                "scale": matrix.scale.name,
                "weight_global_scale": matrix.weight_global_scale.name,
                "input_global_scale": matrix.input_global_scale.name,
            }
            for role, matrix in (
                ("gate", expert.nvfp4_gate), ("up", expert.nvfp4_up),
                ("down", expert.nvfp4_down),
            )
            if matrix is not None
        }
    return RecordResult(entry=entry, end_offset=start + record_bytes)


def durable_close(handle: BinaryIO) -> None:
    fsync_file(handle)
    handle.close()


def truncate_to(path: Path, size: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(size)
        fsync_file(handle)
