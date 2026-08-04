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
