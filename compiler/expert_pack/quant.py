"""Streaming quantization primitives.

The stdlib path converts one output row at a time.  It deliberately never
materializes a complete source tensor.  NumPy is not required for correctness;
the optional ``fast`` dependency can be introduced without changing the ABI.
"""

from __future__ import annotations

import math
import struct
import sys
from array import array
from collections.abc import Iterable
from typing import BinaryIO

from .constants import (
    FP4_QUANT_GROUP_SIZE,
    FP4_UE8M0_MAX_CODE,
    FP4_UE8M0_MIN_CODE,
)
from .errors import SourceFormatError
from .safetensors import TensorView
from .util import write_all

SUPPORTED_FLOAT_DTYPES = {"BF16", "F16", "F32"}

try:  # Optional acceleration; correctness does not depend on NumPy.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by the dependency-free CI path.
    _np = None


def _decode_float_row(raw: memoryview, dtype: str) -> Iterable[float]:
    if _np is not None:
        if dtype == "BF16":
            words = _np.frombuffer(raw, dtype="<u2").astype("<u4")
            return (words << 16).view("<f4")
        if dtype == "F16":
            return _np.frombuffer(raw, dtype="<f2").astype("<f4")
        if dtype == "F32":
            return _np.frombuffer(raw, dtype="<f4")
    if dtype == "BF16":
        words = array("H")
        words.frombytes(raw)
        if sys.byteorder != "little":
            words.byteswap()
        bits = array("I", (value << 16 for value in words))
        floats = array("f")
        floats.frombytes(bits.tobytes())
        if sys.byteorder != "little":
            floats.byteswap()
        return floats
    if dtype == "F16":
        count = len(raw) // 2
        return struct.unpack(f"<{count}e", raw)
    if dtype == "F32":
        values = array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return values
    raise SourceFormatError(f"quant profile does not accept source dtype {dtype}")


def _row_geometry(view: TensorView) -> tuple[int, int, int]:
    shape = view.info.shape
    if not shape or len(shape) > 2:
        raise SourceFormatError(
            f"quant profile accepts rank 1/2 tensors, got {shape} for {view.info.name}"
        )
    if view.info.dtype not in SUPPORTED_FLOAT_DTYPES:
        raise SourceFormatError(
            f"quant profile accepts BF16/F16/F32, got {view.info.dtype} for {view.info.name}"
        )
    rows = shape[0] if len(shape) == 2 else 1
    columns = shape[1] if len(shape) == 2 else shape[0]
    element_bytes = {"BF16": 2, "F16": 2, "F32": 4}[view.info.dtype]
    return rows, columns, columns * element_bytes


def write_int8_rows(view: TensorView, destination: BinaryIO, digest: object) -> bytes:
    """Write row-major int8 values and return little-endian FP32 scales."""
    rows, columns, row_bytes = _row_geometry(view)
    scales = bytearray()
    for row in range(rows):
        start = row * row_bytes
        values = _decode_float_row(view.raw[start : start + row_bytes], view.info.dtype)
        if _np is not None:
            numeric = _np.asarray(values, dtype="<f4")
            if numeric.size != columns:
                raise SourceFormatError(f"short decoded row in {view.info.name}")
            if not bool(_np.isfinite(numeric).all()):
                raise SourceFormatError(f"non-finite weight in {view.info.name}, row {row}")
            maximum = float(_np.max(_np.abs(numeric), initial=0.0))
            scale = maximum / 127.0 if maximum else 1.0
            payload = _np.clip(_np.rint(numeric / scale), -127, 127).astype("i1").tobytes()
            write_all(destination, payload)
            digest.update(payload)
            scales.extend(struct.pack("<f", scale))
            continue
        maximum = 0.0
        materialized: list[float] = []
        for value in values:
            scalar = float(value)
            if not math.isfinite(scalar):
                raise SourceFormatError(f"non-finite weight in {view.info.name}, row {row}")
            materialized.append(scalar)
            maximum = max(maximum, abs(scalar))
        if len(materialized) != columns:
            raise SourceFormatError(f"short decoded row in {view.info.name}")
        scale = maximum / 127.0 if maximum else 1.0
        quantized = array(
            "b",
            (
                max(-127, min(127, int(round(value / scale))))
                for value in materialized
            ),
        )
        payload = quantized.tobytes()
        write_all(destination, payload)
        digest.update(payload)
        scales.extend(struct.pack("<f", scale))
    return bytes(scales)


# E2M1 finite magnitudes in nibble-index order; the sign rides bit 3.  This is
# the exact inverse of the runtime decode table (2x these values, with the
# kernel's final 0.5 factor) and of deepseek_quant.FP4_E2M1_VALUES.
_FP4_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp4_block_code(maximum: float) -> int:
    """Pick the UE8M0 code whose scale covers a block maximum with E2M1."""

    if maximum == 0.0:
        return 127
    exponent = math.ceil(math.log2(maximum / 6.0))
    return min(FP4_UE8M0_MAX_CODE, max(FP4_UE8M0_MIN_CODE, exponent + 127))


def _fp4_nearest_index(magnitude: float) -> int:
    """Nearest E2M1 level index; exact ties go to the even index."""

    upper = 1
    while upper < len(_FP4_E2M1_LEVELS) and _FP4_E2M1_LEVELS[upper] < magnitude:
        upper += 1
    if upper == len(_FP4_E2M1_LEVELS):
        return len(_FP4_E2M1_LEVELS) - 1
    lower = upper - 1
    low_distance = magnitude - _FP4_E2M1_LEVELS[lower]
    high_distance = _FP4_E2M1_LEVELS[upper] - magnitude
    if high_distance < low_distance or (
        high_distance == low_distance and upper % 2 == 0
    ):
        return upper
    return lower


def _fp4_pack_nibbles(indices: list[int], signs: list[int]) -> bytes:
    packed = bytearray(len(indices) // 2)
    for pair in range(len(packed)):
        packed[pair] = (indices[2 * pair] | (signs[2 * pair] << 3)) | (
            (indices[2 * pair + 1] | (signs[2 * pair + 1] << 3)) << 4
        )
    return bytes(packed)


def write_fp4_block32_rows(view: TensorView, destination: BinaryIO, digest: object) -> bytes:
    """Write FP4-E2M1 packed rows and return UE8M0 block scales.

    Each flattened output row is encoded independently. The last source
    dimension is zero-padded to a complete block in storage; the original
    shape remains authoritative in the tensor index. Every block of
    ``FP4_QUANT_GROUP_SIZE`` values shares one UE8M0 scale chosen as the
    smallest power of two covering the block maximum, values round to the
    nearest E2M1 level (ties to the even level index), and nibbles pack
    low-then-high per byte.  NumPy and stdlib paths both compute in float64 so
    their bytes are identical.
    """

    shape = view.info.shape
    if not shape or len(shape) > 5:
        raise SourceFormatError(
            f"FP4 dense storage accepts rank 1-5 tensors, got {shape} "
            f"for {view.info.name}"
        )
    if view.info.dtype not in SUPPORTED_FLOAT_DTYPES:
        raise SourceFormatError(
            f"FP4 dense storage accepts BF16/F16/F32, got "
            f"{view.info.dtype} for {view.info.name}"
        )
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    padded_columns = (
        (columns + FP4_QUANT_GROUP_SIZE - 1) // FP4_QUANT_GROUP_SIZE
    ) * FP4_QUANT_GROUP_SIZE
    element_bytes = {"BF16": 2, "F16": 2, "F32": 4}[view.info.dtype]
    row_bytes = columns * element_bytes
    blocks = padded_columns // FP4_QUANT_GROUP_SIZE
    scales = bytearray()
    if _np is not None:
        # Amortize Python, mmap slicing and write overhead across a bounded
        # number of rows. The arithmetic remains float64 and preserves the
        # exact row-major payload/scale order of the dependency-free path.
        maximum_chunk_values = 256 * 1024
        chunk_rows = max(
            1, min(1024, maximum_chunk_values // max(padded_columns, 1))
        )
        levels = _np.asarray(_FP4_E2M1_LEVELS, dtype="<f8")
        for first_row in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - first_row)
            start = first_row * row_bytes
            raw = view.raw[start : start + count * row_bytes]
            numeric = _np.asarray(
                _decode_float_row(raw, view.info.dtype), dtype="<f8"
            )
            if numeric.size != count * columns:
                raise SourceFormatError(f"short decoded rows in {view.info.name}")
            if not bool(_np.isfinite(numeric).all()):
                raise SourceFormatError(
                    f"non-finite weight in {view.info.name}, rows "
                    f"{first_row}:{first_row + count}"
                )
            numeric = numeric.reshape(count, columns)
            if padded_columns != columns:
                numeric = _np.pad(
                    numeric, ((0, 0), (0, padded_columns - columns))
                )
            grid = numeric.reshape(count, blocks, FP4_QUANT_GROUP_SIZE)
            maxima = _np.max(_np.abs(grid), axis=2)
            positive = maxima != 0.0
            mantissas, exponents = _np.frexp(
                _np.where(positive, maxima / 6.0, 1.0)
            )
            covering_exponents = _np.where(
                mantissas == 0.5, exponents - 1, exponents
            )
            codes = _np.where(
                positive,
                _np.clip(
                    covering_exponents + 127,
                    FP4_UE8M0_MIN_CODE,
                    FP4_UE8M0_MAX_CODE,
                ),
                127,
            ).astype("<u1")
            block_scales = _np.ldexp(
                1.0, codes.astype("<i8") - 127
            )
            quotient = grid / block_scales[:, :, None]
            magnitude = _np.abs(quotient)
            upper = _np.clip(
                _np.searchsorted(levels, magnitude, side="left"),
                1,
                len(levels) - 1,
            )
            lower = upper - 1
            low_distance = magnitude - levels[lower]
            high_distance = levels[upper] - magnitude
            choose_upper = (high_distance < low_distance) | (
                (high_distance == low_distance) & (upper % 2 == 0)
            )
            indices = _np.where(choose_upper, upper, lower).astype("<u1")
            signs = (quotient < 0.0).astype("<u1")
            nibbles = (indices | (signs << 3)).reshape(
                count, padded_columns
            )
            payload = (
                nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
            ).astype("<u1").tobytes()
            write_all(destination, payload)
            digest.update(payload)
            scales.extend(codes.tobytes())
        return bytes(scales)

    for row in range(rows):
        start = row * row_bytes
        values = _decode_float_row(view.raw[start : start + row_bytes], view.info.dtype)
        materialized: list[float] = []
        for value in values:
            scalar = float(value)
            if not math.isfinite(scalar):
                raise SourceFormatError(f"non-finite weight in {view.info.name}, row {row}")
            materialized.append(scalar)
        if len(materialized) != columns:
            raise SourceFormatError(f"short decoded row in {view.info.name}")
        materialized.extend([0.0] * (padded_columns - columns))
        for block in range(blocks):
            chunk = materialized[
                block * FP4_QUANT_GROUP_SIZE : (block + 1) * FP4_QUANT_GROUP_SIZE
            ]
            code = _fp4_block_code(max(abs(value) for value in chunk))
            scale = math.ldexp(1.0, code - 127)
            indices = []
            signs = []
            for value in chunk:
                quotient = value / scale
                indices.append(_fp4_nearest_index(abs(quotient)))
                signs.append(1 if quotient < 0.0 else 0)
            payload = _fp4_pack_nibbles(indices, signs)
            write_all(destination, payload)
            digest.update(payload)
            scales.append(code)
    return bytes(scales)


def write_float32(view: TensorView, destination: BinaryIO, digest: object) -> int:
    """Write an unquantized tensor normalized to little-endian FP32."""

    if len(view.info.shape) > 2:
        if len(view.info.shape) > 4 or view.info.dtype not in SUPPORTED_FLOAT_DTYPES:
            raise SourceFormatError(
                f"float32 storage accepts rank <=4 BF16/F16/F32, got {view.info.shape} {view.info.dtype}"
            )
        values = _decode_float_row(view.raw, view.info.dtype)
        if _np is not None:
            payload = _np.asarray(values, dtype="<f4").tobytes()
        else:
            output = array("f", values)
            if sys.byteorder != "little":
                output.byteswap()
            payload = output.tobytes()
        write_all(destination, payload)
        digest.update(payload)
        return len(payload)

    rows, _columns, row_bytes = _row_geometry(view)
    written = 0
    for row in range(rows):
        start = row * row_bytes
        values = _decode_float_row(view.raw[start : start + row_bytes], view.info.dtype)
        if _np is not None:
            payload = _np.asarray(values, dtype="<f4").tobytes()
            write_all(destination, payload)
            digest.update(payload)
            written += len(payload)
            continue
        output = array("f", values)
        if sys.byteorder != "little":
            output.byteswap()
        payload = output.tobytes()
        write_all(destination, payload)
        digest.update(payload)
        written += len(payload)
    return written
