"""Dependency-free numeric decoders for DeepSeek-V4 source quantization."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .errors import SourceFormatError


# Bit-exact E2M1 finite value mapping used by the published DeepSeek converter.
FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def decode_fp8_e4m3fn(code: int) -> float:
    """Decode one finite OCP E4M3FN byte; the two NaN encodings fail closed."""

    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255:
        raise SourceFormatError(f"E4M3 code must be one byte, got {code!r}")
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        raise SourceFormatError("E4M3FN NaN is not a valid model weight")
    if exponent == 0:
        return sign * math.ldexp(float(mantissa), -9)
    return sign * math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def decode_fp8_e4m3fn_values(
    raw: bytes | bytearray | memoryview,
) -> tuple[float, ...]:
    return tuple(decode_fp8_e4m3fn(code) for code in memoryview(raw).cast("B"))


def decode_ue8m0(code: int) -> float:
    """Decode one OCP E8M0FNU scale byte into a Python float.

    Codes 0..254 are powers of two with exponent bias 127. Code 255 is NaN
    and is rejected because weights/scales must be finite.
    """

    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 255:
        raise SourceFormatError(f"UE8M0 code must be one byte, got {code!r}")
    if code == 255:
        raise SourceFormatError("UE8M0 NaN scale is not a valid model weight scale")
    return math.ldexp(1.0, code - 127)


def decode_ue8m0_values(raw: bytes | bytearray | memoryview) -> tuple[float, ...]:
    return tuple(decode_ue8m0(code) for code in memoryview(raw).cast("B"))


def decode_fp4_e2m1_values(raw: bytes | bytearray | memoryview) -> tuple[float, ...]:
    """Unpack low nibble then high nibble, matching DeepSeek tensor order."""

    values: list[float] = []
    for packed in memoryview(raw).cast("B"):
        values.append(FP4_E2M1_VALUES[packed & 0x0F])
        values.append(FP4_E2M1_VALUES[(packed >> 4) & 0x0F])
    return tuple(values)


def iter_scaled_fp4_e2m1_blocks(
    packed: bytes | bytearray | memoryview,
    scale_codes: bytes | bytearray | memoryview,
    *,
    block_size: int = 32,
) -> Iterable[tuple[float, ...]]:
    """Yield decoded logical blocks without materializing a full matrix."""

    if block_size <= 0 or block_size % 2:
        raise SourceFormatError("FP4 block size must be a positive even integer")
    packed_view = memoryview(packed).cast("B")
    scale_view = memoryview(scale_codes).cast("B")
    packed_per_block = block_size // 2
    if len(packed_view) != len(scale_view) * packed_per_block:
        raise SourceFormatError(
            "FP4 payload/scale geometry mismatch: expected one scale per "
            f"{block_size} logical values"
        )
    for block, scale_code in enumerate(scale_view):
        start = block * packed_per_block
        unscaled = decode_fp4_e2m1_values(
            packed_view[start : start + packed_per_block]
        )
        scale = decode_ue8m0(scale_code)
        decoded = tuple(value * scale for value in unscaled)
        if not all(math.isfinite(value) for value in decoded):
            raise SourceFormatError(f"non-finite decoded FP4 block {block}")
        yield decoded


def decode_scaled_fp4_e2m1_row(
    packed: bytes | bytearray | memoryview,
    scale_codes: bytes | bytearray | memoryview,
    *,
    block_size: int = 32,
) -> tuple[float, ...]:
    """Decode one row; callers should stream matrices row by row."""

    return tuple(
        value
        for block in iter_scaled_fp4_e2m1_blocks(
            packed, scale_codes, block_size=block_size
        )
        for value in block
    )
