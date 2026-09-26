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
    MXFP6_QUANT_GROUP_SIZE,
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


def _float32(value: float) -> float:
    """Round one scalar to the canonical IEEE binary32 arithmetic domain."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def quantize_int8_row(
    values: Iterable[float], columns: int
) -> tuple[bytes, float]:
    """Encode one symmetric INT8 row with canonical binary32 arithmetic.

    Source values, the per-row scale, and division are all evaluated in
    IEEE binary32 before nearest-ties-to-even rounding. This is the artifact
    ABI used by the accelerated writer and must remain byte-identical when
    NumPy is unavailable or when the source-quality gate re-encodes a row.
    """
    if _np is not None:
        numeric = _np.asarray(values, dtype="<f4")
        if numeric.size != columns:
            raise SourceFormatError("short decoded INT8 row")
        if not bool(_np.isfinite(numeric).all()):
            raise SourceFormatError("non-finite weight in INT8 row")
        maximum = float(_np.max(_np.abs(numeric), initial=0.0))
        scale = _np.float32(maximum / 127.0 if maximum else 1.0)
        payload = _np.clip(
            _np.rint(numeric / scale), -127, 127
        ).astype("i1").tobytes()
        return payload, float(scale)

    materialized: list[float] = []
    maximum = 0.0
    for value in values:
        scalar = _float32(float(value))
        if not math.isfinite(scalar):
            raise SourceFormatError("non-finite weight in INT8 row")
        materialized.append(scalar)
        maximum = max(maximum, abs(scalar))
    if len(materialized) != columns:
        raise SourceFormatError("short decoded INT8 row")
    scale = _float32(maximum / 127.0 if maximum else 1.0)
    quantized = array(
        "b",
        (
            max(-127, min(127, int(round(_float32(value / scale)))))
            for value in materialized
        ),
    )
    return quantized.tobytes(), scale


def write_int8_rows(view: TensorView, destination: BinaryIO, digest: object) -> bytes:
    """Write row-major int8 values and return little-endian FP32 scales."""
    rows, columns, row_bytes = _row_geometry(view)
    scales = bytearray()
    for row in range(rows):
        start = row * row_bytes
        values = _decode_float_row(view.raw[start : start + row_bytes], view.info.dtype)
        try:
            payload, scale = quantize_int8_row(values, columns)
        except SourceFormatError as error:
            raise SourceFormatError(
                f"{error} in {view.info.name}, row {row}"
            ) from error
        write_all(destination, payload)
        digest.update(payload)
        scales.extend(struct.pack("<f", scale))
    return bytes(scales)


# E2M1 finite magnitudes in nibble-index order; the sign rides bit 3.  This is
# the exact inverse of the runtime decode table (2x these values, with the
# kernel's final 0.5 factor) and of fp4_reference.FP4_E2M1_VALUES.
_FP4_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_ACTIVATION_MAXIMUM_CANDIDATES = 64

# Positive OCP E3M2 values in exponent/mantissa bit order. Bit 5 is the sign.
_MXFP6_E3M2_LEVELS = (
    0.0, 0.0625, 0.125, 0.1875,
    0.25, 0.3125, 0.375, 0.4375,
    0.5, 0.625, 0.75, 0.875,
    1.0, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 3.5,
    4.0, 5.0, 6.0, 7.0,
    8.0, 10.0, 12.0, 14.0,
    16.0, 20.0, 24.0, 28.0,
)


def _fp4_block_code(maximum: float) -> int:
    """Pick the UE8M0 code whose scale covers a block maximum with E2M1."""

    if maximum == 0.0:
        return 127
    exponent = math.ceil(math.log2(maximum / 6.0))
    return min(FP4_UE8M0_MAX_CODE, max(FP4_UE8M0_MIN_CODE, exponent + 127))


def _fp4_min_error_block_code(values: Iterable[float]) -> int:
    """Choose the lower-error of the covering scale and one exponent lower.

    The smaller scale may clip the largest value to E2M1's finite magnitude
    6, but often represents the other 31 values more accurately.  The payload
    and UE8M0 ABI are unchanged; ties retain the non-clipping covering scale.
    """

    materialized = tuple(float(value) for value in values)
    covering = _fp4_block_code(max(abs(value) for value in materialized))
    candidate = max(FP4_UE8M0_MIN_CODE, covering - 1)
    if candidate == covering:
        return covering

    def squared_error(code: int) -> float:
        scale = math.ldexp(1.0, code - 127)
        return sum(
            (
                _FP4_E2M1_LEVELS[
                    _fp4_nearest_index(abs(value / scale))
                ] * scale - abs(value)
            ) ** 2
            for value in materialized
        )

    return candidate if squared_error(candidate) < squared_error(covering) \
        else covering


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


def _activation_aware_payload_indices(
    grid,
    indices,
    codes,
    activation_values,
    columns: int,
    levels,
):
    """Select adjacent E2M1 codes against the complete output residual.

    UE8M0 scales are fixed. For every output row, candidates are ranked by
    isolated calibration gain and the best complete-residual prefix is kept.
    This is an offline payload decision; it does not alter the FP4 ABI or the
    runtime kernel.
    """

    count, blocks, block_size = grid.shape
    padded_columns = blocks * block_size
    flat_grid = grid.reshape(count, padded_columns)
    flat_indices = indices.reshape(count, padded_columns).astype("<i2")
    value_scales = _np.repeat(
        _np.ldexp(
            _np.ones_like(codes, dtype="<f8"),
            codes.astype("<i2") - 127,
        ),
        block_size,
        axis=1,
    )
    signs = _np.where(flat_grid < 0.0, -1.0, 1.0)
    baseline = levels[flat_indices] * value_scales * signs
    calibration = activation_values.reshape(
        activation_values.shape[0], padded_columns
    )
    residual = (baseline - flat_grid) @ calibration.T
    baseline_sse = _np.sum(residual * residual, axis=1)

    lower_indices = _np.maximum(flat_indices - 1, 0)
    upper_indices = _np.minimum(flat_indices + 1, len(levels) - 1)
    lower_delta = (
        levels[lower_indices] * value_scales * signs - baseline
    )
    upper_delta = (
        levels[upper_indices] * value_scales * signs - baseline
    )
    correlation = residual @ calibration
    energy = _np.sum(calibration * calibration, axis=0)
    lower_objective = (
        2.0 * lower_delta * correlation
        + lower_delta * lower_delta * energy[None, :]
    )
    upper_objective = (
        2.0 * upper_delta * correlation
        + upper_delta * upper_delta * energy[None, :]
    )
    use_upper = upper_objective < lower_objective
    best_delta = _np.where(use_upper, upper_delta, lower_delta)
    best_indices = _np.where(
        use_upper, upper_indices, lower_indices
    ).astype("<u1")
    isolated_gain = -_np.where(
        use_upper, upper_objective, lower_objective
    )
    eligible = (best_delta != 0.0) & (isolated_gain > 0.0)
    if columns < padded_columns:
        eligible[:, columns:] = False
    if not bool(eligible.any()):
        return indices, 0

    ranked_gain = _np.where(eligible, isolated_gain, -_np.inf)
    candidate_count = min(
        _FP4_ACTIVATION_MAXIMUM_CANDIDATES, padded_columns
    )
    partition = _np.argpartition(
        -ranked_gain, candidate_count - 1, axis=1
    )[:, :candidate_count]
    partition_gain = _np.take_along_axis(
        ranked_gain, partition, axis=1
    )
    partition_order = _np.lexsort(
        (partition, -partition_gain), axis=1
    )
    order = _np.take_along_axis(partition, partition_order, axis=1)
    sorted_delta = _np.take_along_axis(best_delta, order, axis=1)
    sorted_eligible = _np.take_along_axis(eligible, order, axis=1)
    ordered_inputs = calibration.T[order]
    contributions = ordered_inputs * sorted_delta[:, :, None]
    _np.cumsum(contributions, axis=1, out=contributions)
    contributions += residual[:, None, :]
    _np.square(contributions, out=contributions)
    prefix_sse = _np.sum(contributions, axis=2)
    best_rank = _np.argmin(prefix_sse, axis=1)
    row_index = _np.arange(count)
    best_sse = prefix_sse[row_index, best_rank]
    eligible_count = _np.sum(sorted_eligible, axis=1)
    improve = (
        (eligible_count > 0)
        & (best_rank < eligible_count)
        & (best_sse < baseline_sse)
    )
    selected_sorted = (
        _np.arange(candidate_count)[None, :] <= best_rank[:, None]
    ) & improve[:, None] & sorted_eligible
    selected = _np.zeros_like(eligible)
    selected[row_index[:, None], order] = selected_sorted
    output = _np.where(
        selected, best_indices, flat_indices
    ).astype("<u1").reshape(count, blocks, block_size)
    return output, int(_np.count_nonzero(selected))


def write_fp4_block32_rows(
    view: TensorView,
    destination: BinaryIO,
    digest: object,
    optimize_mse: bool = False,
    activation_calibration=None,
) -> bytes:
    """Write FP4-E2M1 packed rows and return UE8M0 block scales.

    Each flattened output row is encoded independently. The last source
    dimension is zero-padded to a complete block in storage; the original
    shape remains authoritative in the tensor index. Every block of
    ``FP4_QUANT_GROUP_SIZE`` values shares one UE8M0 scale chosen as the
    smallest power of two covering the block maximum by default.  When
    ``optimize_mse`` is set, each block instead selects the lower-error of that
    covering scale and one power-of-two exponent lower. With exact runtime Q8
    calibration, the MSE choice is the baseline and covering-scale candidates
    are accepted greedily only when they reduce the complete output-row
    residual. Values round to the nearest E2M1 level (ties to the even level
    index), and nibbles pack low-then-high per byte. NumPy and stdlib paths both
    compute in float64 when no activation calibration is supplied.
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
    activation_values = None
    if activation_calibration is not None:
        if not optimize_mse:
            raise ValueError(
                "activation-aware FP4 selection requires the MSE-v2 baseline"
            )
        if _np is None:
            raise ValueError(
                "activation-aware FP4 selection requires NumPy"
            )
        activation_values = activation_calibration.load(
            view.info.name,
            columns=columns,
            padded_columns=padded_columns,
        )
        if activation_values is not None:
            if (
                activation_values.ndim != 2
                or activation_values.shape[1] != padded_columns
                or activation_values.shape[0] <= 0
                or not bool(_np.isfinite(activation_values).all())
            ):
                raise SourceFormatError(
                    f"invalid activation calibration for {view.info.name}"
                )
            activation_values = _np.asarray(
                activation_values, dtype="<f8"
            ).reshape(
                activation_values.shape[0], blocks, FP4_QUANT_GROUP_SIZE
            )
    activation_changes = 0
    payload_changes = 0
    select_payload_codes = (
        activation_values is not None
        and activation_calibration.selects_payload_codes(view.info.name)
    )
    if _np is not None:
        # Amortize Python, mmap slicing and write overhead across a bounded
        # number of rows. The arithmetic remains float64 and preserves the
        # exact row-major payload/scale order of the dependency-free path.
        maximum_chunk_values = 256 * 1024
        chunk_rows = max(
            1, min(1024, maximum_chunk_values // max(padded_columns, 1))
        )
        levels = _np.asarray(_FP4_E2M1_LEVELS, dtype="<f8")

        def nearest_indices(magnitudes):
            upper = _np.clip(
                _np.searchsorted(levels, magnitudes, side="left"),
                1,
                len(levels) - 1,
            )
            lower = upper - 1
            low_distance = magnitudes - levels[lower]
            high_distance = levels[upper] - magnitudes
            choose_upper = (high_distance < low_distance) | (
                (high_distance == low_distance) & (upper % 2 == 0)
            )
            return _np.where(choose_upper, upper, lower)

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
            absolute = _np.abs(grid)
            maxima = _np.max(absolute, axis=2)
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
            covering_scales = _np.ldexp(
                _np.ones_like(codes, dtype="<f8"),
                codes.astype("<i8") - 127,
            )
            covering_magnitudes = absolute / covering_scales[:, :, None]
            covering_indices = nearest_indices(covering_magnitudes)
            if optimize_mse:
                candidate_codes = _np.maximum(
                    codes.astype("<i2") - 1, FP4_UE8M0_MIN_CODE
                ).astype("<u1")
                candidate_indices = nearest_indices(
                    covering_magnitudes * 2.0
                )
                covering_error = _np.sum(
                    (levels[covering_indices] - covering_magnitudes) ** 2,
                    axis=2,
                )
                candidate_error = _np.sum(
                    (0.5 * levels[candidate_indices]
                     - covering_magnitudes) ** 2,
                    axis=2,
                )
                use_candidate = (
                    (candidate_codes != codes)
                    & (candidate_error < covering_error)
                )
                baseline_codes = _np.where(
                    use_candidate, candidate_codes, codes
                ).astype("<u1")
                baseline_indices = _np.where(
                    use_candidate[:, :, None],
                    candidate_indices,
                    covering_indices,
                ).astype("<u1")
                if activation_values is not None:
                    signs = _np.where(grid < 0.0, -1.0, 1.0)
                    covering_decoded = (
                        levels[covering_indices]
                        * covering_scales[:, :, None]
                        * signs
                    )
                    candidate_decoded = (
                        levels[candidate_indices]
                        * (covering_scales * 0.5)[:, :, None]
                        * signs
                    )
                    baseline_decoded = _np.where(
                        use_candidate[:, :, None],
                        candidate_decoded,
                        covering_decoded,
                    )
                    baseline_error = baseline_decoded - grid
                    residual = _np.einsum(
                        "sbk,rbk->rs",
                        activation_values,
                        baseline_error,
                        optimize=True,
                    )
                    deltas = _np.where(
                        use_candidate[:, :, None],
                        covering_decoded - candidate_decoded,
                        0.0,
                    )
                    output_deltas = _np.einsum(
                        "sbk,rbk->rbs",
                        activation_values,
                        deltas,
                        optimize=True,
                    )
                    current_error = _np.sum(residual * residual, axis=1)
                    isolated_error = _np.sum(
                        (residual[:, None, :] + output_deltas) ** 2,
                        axis=2,
                    )
                    isolated_gain = current_error[:, None] - isolated_error
                    order = _np.argsort(
                        -isolated_gain, axis=1, kind="stable"
                    )
                    selected = _np.zeros_like(use_candidate, dtype=bool)
                    row_index = _np.arange(count)
                    for rank in range(blocks):
                        block_index = order[:, rank]
                        valid = use_candidate[row_index, block_index]
                        delta = output_deltas[row_index, block_index]
                        next_residual = residual + delta
                        next_error = _np.sum(
                            next_residual * next_residual, axis=1
                        )
                        accept = valid & (next_error < current_error)
                        if bool(accept.any()):
                            residual[accept] = next_residual[accept]
                            current_error[accept] = next_error[accept]
                            selected[
                                row_index[accept], block_index[accept]
                            ] = True
                    activation_changes += int(_np.count_nonzero(selected))
                    codes = _np.where(
                        selected, codes, baseline_codes
                    ).astype("<u1")
                    indices = _np.where(
                        selected[:, :, None],
                        covering_indices,
                        baseline_indices,
                    ).astype("<u1")
                else:
                    codes = baseline_codes
                    indices = baseline_indices
            else:
                indices = covering_indices.astype("<u1")
            if select_payload_codes:
                indices, changed = _activation_aware_payload_indices(
                    grid,
                    indices,
                    codes,
                    activation_values,
                    columns,
                    levels,
                )
                payload_changes += changed
            signs = (grid < 0.0).astype("<u1")
            nibbles = (indices | (signs << 3)).reshape(
                count, padded_columns
            )
            payload = (
                nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
            ).astype("<u1").tobytes()
            write_all(destination, payload)
            digest.update(payload)
            scales.extend(codes.tobytes())
        if activation_values is not None:
            activation_calibration.record_selection(
                view.info.name,
                rows=rows,
                blocks=blocks,
                changed=activation_changes,
                payload_changed=payload_changes,
            )
        return bytes(scales)

    if activation_values is not None:
        raise ValueError(
            "activation-aware FP4 selection has no dependency-free path"
        )

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
            code = (
                _fp4_min_error_block_code(chunk)
                if optimize_mse
                else _fp4_block_code(max(abs(value) for value in chunk))
            )
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


def _mxfp6_block_code(maximum: float) -> int:
    """OCP MX scale: floor(log2(maximum)) relative to E3M2 value 16."""

    if maximum == 0.0:
        return 127
    exponent = math.floor(math.log2(maximum)) - 4
    return min(FP4_UE8M0_MAX_CODE, max(FP4_UE8M0_MIN_CODE, exponent + 127))


def _mxfp6_nearest_index(magnitude: float) -> int:
    """Nearest E3M2 magnitude; exact ties select an even encoded LSB."""

    upper = 1
    while (upper < len(_MXFP6_E3M2_LEVELS) and
           _MXFP6_E3M2_LEVELS[upper] < magnitude):
        upper += 1
    if upper == len(_MXFP6_E3M2_LEVELS):
        return len(_MXFP6_E3M2_LEVELS) - 1
    lower = upper - 1
    low_distance = magnitude - _MXFP6_E3M2_LEVELS[lower]
    high_distance = _MXFP6_E3M2_LEVELS[upper] - magnitude
    if high_distance < low_distance or (
            high_distance == low_distance and upper % 2 == 0):
        return upper
    return lower


def _mxfp6_pack_codes(codes: list[int]) -> bytes:
    if len(codes) % 4:
        raise ValueError("MXFP6 code count must be divisible by four")
    packed = bytearray(len(codes) * 3 // 4)
    for group in range(len(codes) // 4):
        word = (
            codes[4 * group]
            | (codes[4 * group + 1] << 6)
            | (codes[4 * group + 2] << 12)
            | (codes[4 * group + 3] << 18)
        )
        packed[3 * group] = word & 0xff
        packed[3 * group + 1] = (word >> 8) & 0xff
        packed[3 * group + 2] = (word >> 16) & 0xff
    return bytes(packed)


def write_mxfp6_e3m2_block32_rows(
    view: TensorView, destination: BinaryIO, digest: object
) -> bytes:
    """Write OCP MXFP6-E3M2 rows and return one UE8M0 scale per block32."""

    shape = view.info.shape
    if not shape or len(shape) > 5:
        raise SourceFormatError(
            f"MXFP6 dense storage accepts rank 1-5 tensors, got {shape} "
            f"for {view.info.name}"
        )
    if view.info.dtype not in SUPPORTED_FLOAT_DTYPES:
        raise SourceFormatError(
            f"MXFP6 dense storage accepts BF16/F16/F32, got "
            f"{view.info.dtype} for {view.info.name}"
        )
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    columns = shape[-1]
    padded_columns = (
        (columns + MXFP6_QUANT_GROUP_SIZE - 1) // MXFP6_QUANT_GROUP_SIZE
    ) * MXFP6_QUANT_GROUP_SIZE
    element_bytes = {"BF16": 2, "F16": 2, "F32": 4}[view.info.dtype]
    row_bytes = columns * element_bytes
    blocks = padded_columns // MXFP6_QUANT_GROUP_SIZE
    scales = bytearray()

    if _np is not None:
        maximum_chunk_values = 256 * 1024
        chunk_rows = max(
            1, min(1024, maximum_chunk_values // max(padded_columns, 1))
        )
        levels = _np.asarray(_MXFP6_E3M2_LEVELS, dtype="<f8")
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
            grid = numeric.reshape(count, blocks, MXFP6_QUANT_GROUP_SIZE)
            maxima = _np.max(_np.abs(grid), axis=2)
            positive = maxima != 0.0
            _mantissas, exponents = _np.frexp(
                _np.where(positive, maxima, 1.0)
            )
            floor_exponents = exponents - 1
            scale_codes = _np.where(
                positive,
                _np.clip(
                    floor_exponents - 4 + 127,
                    FP4_UE8M0_MIN_CODE,
                    FP4_UE8M0_MAX_CODE,
                ),
                127,
            ).astype("<u1")
            block_scales = _np.ldexp(
                1.0, scale_codes.astype("<i8") - 127
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
            codes = (indices | ((quotient < 0.0).astype("<u1") << 5)).reshape(
                count, padded_columns // 4, 4
            ).astype("<u4")
            words = (
                codes[:, :, 0]
                | (codes[:, :, 1] << 6)
                | (codes[:, :, 2] << 12)
                | (codes[:, :, 3] << 18)
            )
            packed = _np.empty((count, padded_columns // 4, 3), dtype="<u1")
            packed[:, :, 0] = words & 0xff
            packed[:, :, 1] = (words >> 8) & 0xff
            packed[:, :, 2] = (words >> 16) & 0xff
            payload = packed.tobytes()
            write_all(destination, payload)
            digest.update(payload)
            scales.extend(scale_codes.tobytes())
        return bytes(scales)

    for row in range(rows):
        start = row * row_bytes
        values = _decode_float_row(
            view.raw[start : start + row_bytes], view.info.dtype
        )
        materialized = [float(value) for value in values]
        if len(materialized) != columns:
            raise SourceFormatError(f"short decoded row in {view.info.name}")
        if not all(math.isfinite(value) for value in materialized):
            raise SourceFormatError(
                f"non-finite weight in {view.info.name}, row {row}"
            )
        materialized.extend([0.0] * (padded_columns - columns))
        for block in range(blocks):
            chunk = materialized[
                block * MXFP6_QUANT_GROUP_SIZE:
                (block + 1) * MXFP6_QUANT_GROUP_SIZE
            ]
            scale_code = _mxfp6_block_code(
                max(abs(value) for value in chunk)
            )
            scale = math.ldexp(1.0, scale_code - 127)
            codes = [
                _mxfp6_nearest_index(abs(value / scale))
                | (0x20 if value < 0.0 else 0)
                for value in chunk
            ]
            payload = _mxfp6_pack_codes(codes)
            write_all(destination, payload)
            digest.update(payload)
            scales.append(scale_code)
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
