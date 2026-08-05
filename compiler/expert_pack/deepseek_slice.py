"""Bounded real-expert qualification for the DeepSeek source/cache boundary."""

from __future__ import annotations

import hashlib
import math
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .deepseek_v4 import validate_deepseek_v4_source
from .errors import AdapterError, SourceFormatError
from .safetensors import SafeTensorCheckpoint
from .util import atomic_json, write_all

try:
    import numpy as np
except ImportError:  # pragma: no cover - qualification hosts install the fast extra.
    np = None


def _torch_reference(packed: object, scales: object) -> object:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on qualification host.
        raise SourceFormatError("PyTorch is required for the independent reference") from error

    packed_tensor = torch.from_numpy(packed.copy())
    scale_bytes = torch.from_numpy(scales.copy())
    scale_values = scale_bytes.view(torch.float8_e8m0fnu).float()
    table = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        dtype=torch.float32,
    )
    low = packed_tensor & 0x0F
    high = (packed_tensor >> 4) & 0x0F
    values = torch.stack((table[low.long()], table[high.long()]), dim=-1)
    values = values.flatten(1)
    values *= scale_values.repeat_interleave(32, dim=1)
    return values.numpy()


def _decode_numpy(packed: object, scale_codes: object) -> object:
    if bool((scale_codes == 255).any()):
        raise SourceFormatError("real expert slice contains a UE8M0 NaN scale")
    table = np.asarray(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        dtype=np.float32,
    )
    unpacked = np.empty((packed.shape[0], packed.shape[1], 2), dtype=np.float32)
    unpacked[:, :, 0] = table[packed & 0x0F]
    unpacked[:, :, 1] = table[(packed >> 4) & 0x0F]
    values = unpacked.reshape(packed.shape[0], packed.shape[1] * 2)
    exponents = scale_codes.astype(np.int16) - 127
    scales = np.ldexp(np.ones(scale_codes.shape, dtype=np.float32), exponents)
    values *= np.repeat(scales, 32, axis=1)
    if not bool(np.isfinite(values).all()):
        raise SourceFormatError("real expert slice decoded to a non-finite value")
    return values


def qualify_deepseek_expert(
    checkpoint: SafeTensorCheckpoint,
    *,
    layer: int,
    expert: int,
    row_chunk: int = 128,
    torch_reference: bool = True,
) -> dict[str, object]:
    """Decode one complete expert, compare a reference, and measure INT8 loss.

    No converted payload is retained. Candidate bytes feed SHA-256 incrementally
    so the slice is reproducible without creating a second model artifact.
    """

    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek expert qualification")
    validate_deepseek_v4_source(checkpoint)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < 43:
        raise AdapterError(f"DeepSeek layer must be in [0, 42], got {layer!r}")
    if isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < 256:
        raise AdapterError(f"DeepSeek expert must be in [0, 255], got {expert!r}")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    started = time.perf_counter()
    projections: dict[str, object] = {}
    candidate_payloads: dict[str, tuple[bytes, bytes]] = {}
    total_source_bytes = 0
    total_candidate_bytes = 0
    total_values = 0
    total_squared_error = 0.0
    maximum_absolute_error = 0.0
    reference_equal = True

    prefix = f"layers.{layer}.ffn.experts.{expert}"
    for projection in ("w1", "w2", "w3"):
        weight_name = f"{prefix}.{projection}.weight"
        scale_name = f"{prefix}.{projection}.scale"
        weight_info = checkpoint.tensors[weight_name]
        scale_info = checkpoint.tensors[scale_name]
        rows, packed_columns = weight_info.shape
        scale_rows, scale_columns = scale_info.shape
        if scale_rows != rows or scale_columns * 16 != packed_columns:
            raise SourceFormatError(f"FP4/scale geometry mismatch for {projection}")

        projection_values = 0
        projection_squared_error = 0.0
        projection_max_error = 0.0
        projection_reference_equal = True
        projection_candidate_bytes = 0
        projection_q_payload = bytearray()
        projection_scale_payload = bytearray()
        with checkpoint.open_tensor(weight_name) as weight_view, checkpoint.open_tensor(
            scale_name
        ) as scale_view:
            packed_matrix = np.frombuffer(weight_view.raw, dtype=np.uint8).reshape(
                rows, packed_columns
            )
            scale_matrix = np.frombuffer(scale_view.raw, dtype=np.uint8).reshape(
                rows, scale_columns
            )
            for first_row in range(0, rows, row_chunk):
                last_row = min(rows, first_row + row_chunk)
                packed = packed_matrix[first_row:last_row]
                scale_codes = scale_matrix[first_row:last_row]
                decoded = _decode_numpy(packed, scale_codes)
                if torch_reference:
                    reference = _torch_reference(packed, scale_codes)
                    if not bool(np.array_equal(decoded.view(np.uint32), reference.view(np.uint32))):
                        projection_reference_equal = False
                        reference_equal = False

                maxima = np.max(np.abs(decoded), axis=1)
                row_scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype("<f4")
                quantized = np.clip(
                    np.rint(decoded / row_scales[:, None]), -127, 127
                ).astype(np.int8)
                reconstructed = quantized.astype(np.float32) * row_scales[:, None]
                errors = np.abs(decoded - reconstructed)
                squared_error = np.square(
                    decoded.astype(np.float64) - reconstructed.astype(np.float64)
                )
                projection_squared_error += float(squared_error.sum())
                projection_max_error = max(
                    projection_max_error, float(errors.max(initial=0.0))
                )
                projection_values += int(decoded.size)
                q_bytes = quantized.tobytes(order="C")
                scale_bytes = row_scales.tobytes(order="C")
                projection_q_payload.extend(q_bytes)
                projection_scale_payload.extend(scale_bytes)
                projection_candidate_bytes += len(q_bytes) + len(scale_bytes)

            # NumPy views created with frombuffer export the read-only mmap.
            # Release every derived view before TensorView closes that mapping.
            del packed, scale_codes, decoded, quantized, reconstructed, errors
            if torch_reference:
                del reference
            del packed_matrix, scale_matrix

        candidate_payloads[projection] = (
            bytes(projection_q_payload), bytes(projection_scale_payload)
        )

        projection_mse = projection_squared_error / projection_values
        projections[projection] = {
            "logical_shape": [rows, packed_columns * 2],
            "source_bytes": weight_info.nbytes + scale_info.nbytes,
            "candidate_int8_bytes": projection_candidate_bytes,
            "reference_bitwise_equal": projection_reference_equal if torch_reference else None,
            "mean_squared_error": projection_mse,
            "root_mean_squared_error": math.sqrt(projection_mse),
            "maximum_absolute_error": projection_max_error,
        }
        total_source_bytes += weight_info.nbytes + scale_info.nbytes
        total_candidate_bytes += projection_candidate_bytes
        total_values += projection_values
        total_squared_error += projection_squared_error
        maximum_absolute_error = max(maximum_absolute_error, projection_max_error)

    total_mse = total_squared_error / total_values
    digest = hashlib.sha256()
    # Exact deepseek-sm86-int8-per-row-v1 slot order.
    for projection, section in (
        ("w1", 0), ("w3", 0), ("w1", 1),
        ("w3", 1), ("w2", 0), ("w2", 1),
    ):
        digest.update(candidate_payloads[projection][section])
    elapsed = time.perf_counter() - started
    return {
        "format": "deepseek-v4-expert-slice-qualification-v1",
        "layer": layer,
        "expert": expert,
        "complete_expert": True,
        "row_chunk": row_chunk,
        "reference": "pytorch-float8-e8m0fnu" if torch_reference else None,
        "reference_bitwise_equal": reference_equal if torch_reference else None,
        "source_bytes": total_source_bytes,
        "logical_values": total_values,
        "candidate_int8_bytes": total_candidate_bytes,
        "candidate_abi": "deepseek-sm86-int8-per-row-v1",
        "candidate_sha256": digest.hexdigest(),
        "mean_squared_error": total_mse,
        "root_mean_squared_error": math.sqrt(total_mse),
        "maximum_absolute_error": maximum_absolute_error,
        "elapsed_seconds": elapsed,
        "decoded_values_per_second": total_values / elapsed,
        "projections": projections,
    }


def _fp8_e4m3fn_table() -> object:
    table = np.empty(256, dtype=np.float32)
    for code in range(256):
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 0x0F and mantissa == 0x07:
            table[code] = np.nan
        elif exponent == 0:
            table[code] = math.ldexp(float(mantissa), -9)
        else:
            table[code] = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
        if code & 0x80:
            table[code] = -table[code]
    return table


def qualify_deepseek_shared_expert(
    checkpoint: SafeTensorCheckpoint, *, layer: int, row_chunk: int = 128
) -> dict[str, object]:
    """Qualify the always-active FP8 shared expert and its SM86 candidate."""

    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek shared qualification")
    validate_deepseek_v4_source(checkpoint)
    if not 0 <= layer < 43 or row_chunk <= 0:
        raise AdapterError("invalid DeepSeek shared expert qualification bounds")
    started = time.perf_counter()
    table = _fp8_e4m3fn_table()
    payloads: dict[str, tuple[bytes, bytes]] = {}
    source_bytes = 0
    logical_values = 0
    reference_equal = True
    prefix = f"layers.{layer}.ffn.shared_experts"
    for projection in ("w1", "w2", "w3"):
        weight_name = f"{prefix}.{projection}.weight"
        scale_name = f"{prefix}.{projection}.scale"
        weight_info = checkpoint.tensors[weight_name]
        scale_info = checkpoint.tensors[scale_name]
        rows, columns = weight_info.shape
        if scale_info.shape != (rows // 128, columns // 128):
            raise SourceFormatError(f"FP8 block geometry mismatch for {projection}")
        quantized_payload = bytearray()
        row_scale_payload = bytearray()
        with checkpoint.open_tensor(weight_name) as weight_view, checkpoint.open_tensor(
            scale_name
        ) as scale_view:
            weights = np.frombuffer(weight_view.raw, dtype=np.uint8).reshape(rows, columns)
            scales = np.frombuffer(scale_view.raw, dtype=np.uint8).reshape(
                rows // 128, columns // 128
            )
            for first in range(0, rows, row_chunk):
                last = min(rows, first + row_chunk)
                weight_codes = weights[first:last]
                scale_codes = scales[np.arange(first, last) // 128]
                if bool((scale_codes == 255).any()):
                    raise SourceFormatError("shared expert contains a UE8M0 NaN scale")
                decoded = table[weight_codes]
                decoded *= np.repeat(
                    np.ldexp(
                        np.ones(scale_codes.shape, dtype=np.float32),
                        scale_codes.astype(np.int16) - 127,
                    ),
                    128,
                    axis=1,
                )
                if not bool(np.isfinite(decoded).all()):
                    raise SourceFormatError("shared expert decoded to a non-finite value")
                try:
                    import torch
                except ImportError as error:  # pragma: no cover
                    raise SourceFormatError("PyTorch is required for FP8 reference") from error
                torch_weights = torch.from_numpy(weight_codes.copy()).view(
                    torch.float8_e4m3fn
                ).float()
                torch_scales = torch.from_numpy(scale_codes.copy()).view(
                    torch.float8_e8m0fnu
                ).float().repeat_interleave(128, dim=1)
                reference = (torch_weights * torch_scales).numpy()
                reference_equal &= bool(
                    np.array_equal(decoded.view(np.uint32), reference.view(np.uint32))
                )
                maxima = np.max(np.abs(decoded), axis=1)
                row_scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype("<f4")
                quantized = np.clip(
                    np.rint(decoded / row_scales[:, None]), -127, 127
                ).astype(np.int8)
                quantized_payload.extend(quantized.tobytes(order="C"))
                row_scale_payload.extend(row_scales.tobytes(order="C"))
                logical_values += int(decoded.size)
            del weights, scales, weight_codes, scale_codes, decoded, reference
        payloads[projection] = (bytes(quantized_payload), bytes(row_scale_payload))
        source_bytes += weight_info.nbytes + scale_info.nbytes
    digest = hashlib.sha256()
    for projection, section in (
        ("w1", 0), ("w3", 0), ("w1", 1),
        ("w3", 1), ("w2", 0), ("w2", 1),
    ):
        digest.update(payloads[projection][section])
    elapsed = time.perf_counter() - started
    return {
        "format": "deepseek-v4-shared-expert-qualification-v1",
        "layer": layer,
        "reference": "pytorch-float8-e4m3fn+float8-e8m0fnu",
        "reference_bitwise_equal": reference_equal,
        "source_bytes": source_bytes,
        "logical_values": logical_values,
        "candidate_int8_bytes": sum(len(part) for value in payloads.values() for part in value),
        "candidate_abi": "deepseek-sm86-int8-per-row-v1",
        "candidate_sha256": digest.hexdigest(),
        "elapsed_seconds": elapsed,
    }


def qualify_deepseek_fp8_matrix(
    checkpoint: SafeTensorCheckpoint, *, name: str, row_chunk: int = 128
) -> dict[str, object]:
    """Qualify one block-scaled FP8 matrix for the generic SM86 dense ABI."""

    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek dense qualification")
    validate_deepseek_v4_source(checkpoint)
    weight_name = name + ".weight"
    scale_name = name + ".scale"
    if weight_name not in checkpoint.tensors or scale_name not in checkpoint.tensors:
        raise AdapterError(f"DeepSeek FP8 matrix does not exist: {name}")
    weight_info = checkpoint.tensors[weight_name]
    scale_info = checkpoint.tensors[scale_name]
    if weight_info.dtype != "F8_E4M3" or scale_info.dtype != "F8_E8M0":
        raise AdapterError(f"DeepSeek matrix is not block-scaled FP8: {name}")
    rows, columns = weight_info.shape
    if rows % 128 or columns % 128 or scale_info.shape != (rows // 128, columns // 128):
        raise SourceFormatError("DeepSeek FP8 matrix has invalid 128x128 geometry")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    started = time.perf_counter()
    table = _fp8_e4m3fn_table()
    quantized_payload = bytearray()
    row_scale_payload = bytearray()
    reference_equal = True
    squared_error = 0.0
    maximum_error = 0.0
    with checkpoint.open_tensor(weight_name) as weight_view, checkpoint.open_tensor(
        scale_name
    ) as scale_view:
        weights = np.frombuffer(weight_view.raw, dtype=np.uint8).reshape(rows, columns)
        scales = np.frombuffer(scale_view.raw, dtype=np.uint8).reshape(
            rows // 128, columns // 128
        )
        for first in range(0, rows, row_chunk):
            last = min(rows, first + row_chunk)
            weight_codes = weights[first:last]
            scale_codes = scales[np.arange(first, last) // 128]
            if bool((scale_codes == 255).any()):
                raise SourceFormatError("dense FP8 matrix contains a UE8M0 NaN scale")
            decoded = table[weight_codes]
            decoded *= np.repeat(
                np.ldexp(
                    np.ones(scale_codes.shape, dtype=np.float32),
                    scale_codes.astype(np.int16) - 127,
                ),
                128,
                axis=1,
            )
            if not bool(np.isfinite(decoded).all()):
                raise SourceFormatError("dense FP8 matrix decoded non-finite values")
            try:
                import torch
            except ImportError as error:  # pragma: no cover
                raise SourceFormatError("PyTorch is required for FP8 reference") from error
            reference = (
                torch.from_numpy(weight_codes.copy()).view(torch.float8_e4m3fn).float()
                * torch.from_numpy(scale_codes.copy()).view(torch.float8_e8m0fnu)
                .float().repeat_interleave(128, dim=1)
            ).numpy()
            reference_equal &= bool(
                np.array_equal(decoded.view(np.uint32), reference.view(np.uint32))
            )
            maxima = np.max(np.abs(decoded), axis=1)
            row_scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype("<f4")
            quantized = np.clip(
                np.rint(decoded / row_scales[:, None]), -127, 127
            ).astype(np.int8)
            reconstructed = quantized.astype(np.float32) * row_scales[:, None]
            difference = decoded.astype(np.float64) - reconstructed.astype(np.float64)
            squared_error += float(np.square(difference).sum())
            maximum_error = max(
                maximum_error, float(np.max(np.abs(decoded - reconstructed)))
            )
            quantized_payload.extend(quantized.tobytes(order="C"))
            row_scale_payload.extend(row_scales.tobytes(order="C"))
        del weights, scales, weight_codes, scale_codes, decoded, reference
    digest = hashlib.sha256(quantized_payload)
    digest.update(row_scale_payload)
    values = rows * columns
    elapsed = time.perf_counter() - started
    return {
        "format": "deepseek-v4-fp8-matrix-qualification-v1",
        "name": name,
        "shape": [rows, columns],
        "reference_bitwise_equal": reference_equal,
        "source_bytes": weight_info.nbytes + scale_info.nbytes,
        "candidate_int8_bytes": len(quantized_payload) + len(row_scale_payload),
        "candidate_abi": "deepseek-sm86-int8-per-row-matrix-v1",
        "candidate_sha256": digest.hexdigest(),
        "root_mean_squared_error": math.sqrt(squared_error / values),
        "maximum_absolute_error": maximum_error,
        "elapsed_seconds": elapsed,
    }


def _deepseek_hca_reference(
    streams: object,
    fn: object,
    base: object,
    scale: object,
    *,
    epsilon: float = 1e-6,
    sinkhorn_iterations: int = 20,
) -> tuple[object, object, object, object, object]:
    """Reference the official DeepSeek-V4 mHC pre/post equations in FP32."""

    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek HCA qualification")
    streams = np.asarray(streams, dtype=np.float32)
    fn = np.asarray(fn, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if streams.ndim != 2 or fn.shape != (24, streams.size) or base.shape != (24,):
        raise SourceFormatError("invalid DeepSeek HCA reference geometry")
    if streams.shape[0] != 4 or scale.shape != (3,) or sinkhorn_iterations <= 0:
        raise SourceFormatError("invalid DeepSeek HCA reference configuration")
    flat = streams.reshape(-1)
    inverse_rms = np.float32(
        1.0 / math.sqrt(float(np.mean(np.square(flat, dtype=np.float32))) + epsilon)
    )
    mixes = np.matmul(fn, flat, dtype=np.float32) * inverse_rms
    pre_logits, post_logits, comb_logits = np.split(mixes, (4, 8))
    pre = 1.0 / (1.0 + np.exp(-(pre_logits * scale[0] + base[:4]))) + epsilon
    post = 2.0 / (1.0 + np.exp(-(post_logits * scale[1] + base[4:8])))
    comb_logits = comb_logits.reshape(4, 4) * scale[2] + base[8:].reshape(4, 4)
    shifted = comb_logits - np.max(comb_logits, axis=-1, keepdims=True)
    comb = np.exp(shifted)
    comb /= np.sum(comb, axis=-1, keepdims=True)
    comb += epsilon
    comb /= np.sum(comb, axis=-2, keepdims=True) + epsilon
    for _ in range(sinkhorn_iterations - 1):
        comb /= np.sum(comb, axis=-1, keepdims=True) + epsilon
        comb /= np.sum(comb, axis=-2, keepdims=True) + epsilon
    collapsed = np.sum(pre[:, None] * streams, axis=0, dtype=np.float32)

    # Exercise hc_post independently of attention/MLP with a stable sublayer
    # vector. The complete layer will later substitute the real sublayer output.
    dimension = streams.shape[1]
    positions = np.arange(dimension, dtype=np.float32)
    sublayer = np.sin(positions * np.float32(0.013)) * np.float32(0.125)
    updated = post[:, None] * sublayer[None, :] + np.matmul(comb.T, streams)
    return tuple(
        np.asarray(value, dtype="<f4")
        for value in (pre, post, comb, collapsed, updated)
    )


def _bf16_to_f32(raw: object, shape: tuple[int, ...]) -> object:
    if np is None:
        raise SourceFormatError("NumPy is required for BF16 decoding")
    words = np.frombuffer(raw, dtype="<u2").reshape(shape)
    return (words.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16_words(values: object) -> object:
    """Round FP32 to BF16 with round-to-nearest-even and return little-endian words."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype("<u2")


def _deepseek_compressed_kv_reference(
    normalized: object, cosine: object, sine: object
) -> object:
    """Apply the checkpoint's BF16, RoPE64 and in-place MXFP8 QAT boundary."""
    if np is None:
        raise SourceFormatError("NumPy is required for compressed KV qualification")
    base_words = _f32_to_bf16_words(normalized)
    base = _bf16_to_f32(base_words.tobytes(), (512,)).copy()
    result = base.copy()
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise SourceFormatError("PyTorch is required for compressed KV reference") from error
    for first in range(0, 448, 64):
        block = base[first:first + 64]
        maximum = max(float(np.max(np.abs(block))), 1e-4)
        scale = math.ldexp(1.0, math.ceil(math.log2(maximum / 448.0)))
        quantized = (
            torch.from_numpy((block / scale).copy())
            .to(torch.float8_e4m3fn).float().numpy() * scale
        )
        result[first:first + 64] = quantized
    cosine = np.asarray(cosine, dtype=np.float32)
    sine = np.asarray(sine, dtype=np.float32)
    for pair in range(32):
        left = base[448 + pair * 2]
        right = base[449 + pair * 2]
        result[448 + pair * 2] = left * cosine[pair] - right * sine[pair]
        result[449 + pair * 2] = right * cosine[pair] + left * sine[pair]
    return _f32_to_bf16_words(result)


def _deepseek_sm86_fp8_matvec(
    checkpoint: SafeTensorCheckpoint, name: str, vector: object,
    *, row_first: int = 0, row_last: int | None = None,
) -> object:
    """Independently reproduce FP8 admission and the SM86 warp GEMV ABI."""
    weight_info = checkpoint.tensors[name + ".weight"]
    scale_info = checkpoint.tensors[name + ".scale"]
    rows, columns = weight_info.shape
    row_last = rows if row_last is None else row_last
    if not 0 <= row_first < row_last <= rows or columns % 32:
        raise SourceFormatError("invalid DeepSeek reference GEMV rows")
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (columns,) or scale_info.shape != (rows // 128, columns // 128):
        raise SourceFormatError("invalid DeepSeek reference GEMV geometry")
    table = _fp8_e4m3fn_table()
    result = np.empty(row_last - row_first, dtype=np.float32)
    with checkpoint.open_tensor(name + ".weight") as weight_view, \
            checkpoint.open_tensor(name + ".scale") as scale_view:
        weights = np.frombuffer(weight_view.raw, dtype=np.uint8).reshape(rows, columns)
        scales = np.frombuffer(scale_view.raw, dtype=np.uint8).reshape(
            rows // 128, columns // 128
        )
        for first in range(row_first, row_last, 128):
            last = min(first + 128, row_last)
            codes = weights[first:last]
            scale_codes = scales[np.arange(first, last) // 128]
            if bool((scale_codes == 255).any()):
                raise SourceFormatError("DeepSeek reference GEMV contains NaN scale")
            decoded = table[codes]
            decoded *= np.repeat(
                np.ldexp(np.ones(scale_codes.shape, dtype=np.float32),
                         scale_codes.astype(np.int16) - 127),
                128, axis=1,
            )
            maxima = np.max(np.abs(decoded), axis=1)
            row_scales = np.where(maxima > 0, maxima / 127.0, 1.0).astype(np.float32)
            quantized = np.clip(
                np.rint(decoded / row_scales[:, None]), -127, 127
            ).astype(np.int8)
            # Match one CUDA warp per row: each lane accumulates stride-32
            # columns, followed by the fixed shuffle-down reduction tree.
            partial = np.zeros((last - first, 32), dtype=np.float32)
            for block in range(columns // 32):
                begin = block * 32
                partial += quantized[:, begin:begin + 32].astype(np.float32) * \
                    vector[begin:begin + 32]
            for offset in (16, 8, 4, 2, 1):
                partial[:, :offset] += partial[:, offset:2 * offset]
            destination = first - row_first
            result[destination:destination + last - first] = \
                partial[:, 0] * row_scales
        del weights, scales, codes, scale_codes
    return result


def export_deepseek_attention_oracle(
    checkpoint: SafeTensorCheckpoint, *, layer: int, output: Path
) -> dict[str, object]:
    """Emit an independent four-token oracle for one complete attention site."""
    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek attention oracle")
    validate_deepseek_v4_source(checkpoint)
    if layer not in (0, 2):
        raise AdapterError(
            "the complete attention oracle targets sliding-window layer 0 "
            "or ratio-4 layer 2"
        )
    prefix = f"layers.{layer}"
    ratio = int(checkpoint.config["compress_ratios"][layer])

    def tensor(name: str, shape: tuple[int, ...], dtype: str) -> object:
        info = checkpoint.tensors[name]
        if info.shape != shape or info.dtype != dtype:
            raise SourceFormatError(f"invalid attention oracle tensor: {name}")
        with checkpoint.open_tensor(name) as view:
            if dtype == "BF16":
                return _bf16_to_f32(view.raw, shape).copy()
            return np.frombuffer(view.raw, dtype="<f4").reshape(shape).copy()

    positions = np.arange(4 * 4 * 4096, dtype=np.float32).reshape(4, 4, 4096)
    streams = np.sin(positions * np.float32(0.0017)) * np.float32(0.08)
    fn = tensor(prefix + ".hc_attn_fn", (24, 4 * 4096), "F32")
    base = tensor(prefix + ".hc_attn_base", (24,), "F32")
    scale = tensor(prefix + ".hc_attn_scale", (3,), "F32")
    attention_norm = tensor(prefix + ".attn_norm.weight", (4096,), "BF16")
    query_norm = tensor(prefix + ".attn.q_norm.weight", (1024,), "BF16")
    kv_norm = tensor(prefix + ".attn.kv_norm.weight", (512,), "BF16")
    sink = tensor(prefix + ".attn.attn_sink", (64,), "F32")
    compressor_wkv = compressor_wgate = compressor_ape = compressor_norm = None
    if ratio:
        compressor = prefix + ".attn.compressor"
        width = 1024 if ratio == 4 else 512
        compressor_wkv = tensor(
            compressor + ".wkv.weight", (width, 4096), "BF16"
        )
        compressor_wgate = tensor(
            compressor + ".wgate.weight", (width, 4096), "BF16"
        )
        compressor_ape = tensor(
            compressor + ".ape", (ratio, width), "F32"
        )
        compressor_norm = tensor(compressor + ".norm.weight", (512,), "BF16")

    rope = checkpoint.config["rope_scaling"]
    rope_dim = 64
    rope_base = float(
        checkpoint.config["compress_rope_theta"] if ratio
        else checkpoint.config["rope_theta"]
    )
    frequencies = 1.0 / (
        rope_base ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim)
    )

    if ratio:
        original = int(rope["original_max_position_embeddings"])
        factor = float(rope["factor"])
        beta_fast = float(rope["beta_fast"])
        beta_slow = float(rope["beta_slow"])

        def correction(rotation: float) -> float:
            return rope_dim * math.log(original / (rotation * 2 * math.pi)) / \
                (2 * math.log(rope_base))

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), rope_dim - 1)
        ramp = np.clip(
            (np.arange(rope_dim // 2, dtype=np.float32) - low) /
            np.float32(high - low if high != low else 0.001), 0, 1,
        )
        smooth = 1 - ramp
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth
    angles = np.arange(4, dtype=np.float32)[:, None] * frequencies[None, :]
    cosines = np.cos(angles).astype(np.float32)
    sines = np.sin(angles).astype(np.float32)

    def rms(values: object, weight: object) -> object:
        values = np.asarray(values, dtype=np.float32)
        inverse = np.float32(
            1.0 / math.sqrt(float(np.mean(np.square(values, dtype=np.float32))) + 1e-6)
        )
        return np.asarray(values * inverse * weight, dtype=np.float32)

    attention_inputs: list[object] = []
    window_cache: list[object] = []
    updated_tokens: list[object] = []
    for position in range(4):
        pre, post, comb, collapsed, _ = _deepseek_hca_reference(
            streams[position], fn, base, scale
        )
        attention_input = rms(collapsed, attention_norm)
        attention_inputs.append(attention_input)
        qr = _deepseek_sm86_fp8_matvec(
            checkpoint, prefix + ".attn.wq_a", attention_input
        )
        qr = rms(qr, query_norm)
        query = _deepseek_sm86_fp8_matvec(
            checkpoint, prefix + ".attn.wq_b", qr
        ).reshape(64, 512)
        query *= np.asarray(
            1.0 / np.sqrt(
                np.mean(np.square(query, dtype=np.float32), axis=1) + 1e-6
            ), dtype=np.float32,
        )[:, None]
        for pair in range(32):
            left = query[:, 448 + pair * 2].copy()
            right = query[:, 449 + pair * 2].copy()
            query[:, 448 + pair * 2] = \
                left * cosines[position, pair] - right * sines[position, pair]
            query[:, 449 + pair * 2] = \
                right * cosines[position, pair] + left * sines[position, pair]
        query_words = _f32_to_bf16_words(query)
        query = _bf16_to_f32(query_words.tobytes(), (64, 512))

        kv = _deepseek_sm86_fp8_matvec(
            checkpoint, prefix + ".attn.wkv", attention_input
        )
        kv = rms(kv, kv_norm)
        cache_words = _deepseek_compressed_kv_reference(
            kv, cosines[position], sines[position]
        )
        window_cache.append(_bf16_to_f32(cache_words.tobytes(), (512,)))
        selected = list(window_cache)
        if ratio and (position + 1) % ratio == 0:
            compressed = _deepseek_csa_reference(
                np.asarray(attention_inputs), compressor_wkv,
                compressor_wgate, compressor_ape, compressor_norm, ratio=ratio,
            )
            compressed_words = _deepseek_compressed_kv_reference(
                compressed, cosines[position + 1 - ratio],
                sines[position + 1 - ratio]
            )
            selected.append(_bf16_to_f32(compressed_words.tobytes(), (512,)))
        selected_values = np.asarray(selected, dtype=np.float32)
        scores = np.matmul(query, selected_values.T, dtype=np.float32) * \
            np.float32(512.0 ** -0.5)
        maximum = np.maximum(np.max(scores, axis=1), sink)
        attention_weights = np.exp(scores - maximum[:, None])
        denominator = np.sum(attention_weights, axis=1, dtype=np.float32) + \
            np.exp(sink - maximum)
        attention = np.matmul(
            attention_weights, selected_values, dtype=np.float32
        ) / denominator[:, None]
        attention_words = _f32_to_bf16_words(attention)
        attention = _bf16_to_f32(attention_words.tobytes(), (64, 512)).copy()
        for pair in range(32):
            left = attention[:, 448 + pair * 2].copy()
            right = attention[:, 449 + pair * 2].copy()
            attention[:, 448 + pair * 2] = \
                left * cosines[position, pair] + right * sines[position, pair]
            attention[:, 449 + pair * 2] = \
                right * cosines[position, pair] - left * sines[position, pair]

        grouped = []
        flattened = attention.reshape(-1)
        for group in range(8):
            grouped.append(_deepseek_sm86_fp8_matvec(
                checkpoint, prefix + ".attn.wo_a",
                flattened[group * 4096:(group + 1) * 4096],
                row_first=group * 1024, row_last=(group + 1) * 1024,
            ))
        projected = _deepseek_sm86_fp8_matvec(
            checkpoint, prefix + ".attn.wo_b", np.concatenate(grouped)
        )
        updated_tokens.append(
            post[:, None] * projected[None, :] +
            np.matmul(comb.T, streams[position])
        )
    updated = np.asarray(updated_tokens, dtype="<f4")

    ffn_fn = tensor(prefix + ".hc_ffn_fn", (24, 4 * 4096), "F32")
    ffn_base = tensor(prefix + ".hc_ffn_base", (24,), "F32")
    ffn_scale = tensor(prefix + ".hc_ffn_scale", (3,), "F32")
    ffn_norm = tensor(prefix + ".ffn_norm.weight", (4096,), "BF16")
    router = tensor(prefix + ".ffn.gate.weight", (256, 4096), "BF16")
    token_expert_info = checkpoint.tensors[prefix + ".ffn.gate.tid2eid"]
    if token_expert_info.shape != (129280, 6) or token_expert_info.dtype != "I64":
        raise SourceFormatError("invalid hash router token table")
    with checkpoint.open_tensor(prefix + ".ffn.gate.tid2eid") as view:
        token_experts = np.frombuffer(view.raw, dtype="<i8").reshape(129280, 6)
        route_indices = token_experts[:4].astype("<i4").copy()
        del token_experts
    if bool((route_indices < 0).any()) or bool((route_indices >= 256).any()):
        raise SourceFormatError("hash router selected an invalid expert")

    route_weights = np.empty((4, 6), dtype=np.float32)
    ffn_inputs: list[object] = []
    ffn_posts: list[object] = []
    ffn_combinations: list[object] = []
    for position in range(4):
        _, ffn_post, ffn_comb, ffn_collapsed, _ = _deepseek_hca_reference(
            updated[position], ffn_fn, ffn_base, ffn_scale
        )
        ffn_input = rms(ffn_collapsed, ffn_norm)
        ffn_inputs.append(ffn_input)
        ffn_posts.append(ffn_post)
        ffn_combinations.append(ffn_comb)
        partial = np.zeros((256, 32), dtype=np.float32)
        for block in range(4096 // 32):
            begin = block * 32
            partial += router[:, begin:begin + 32] * ffn_input[begin:begin + 32]
        for offset in (16, 8, 4, 2, 1):
            partial[:, :offset] += partial[:, offset:2 * offset]
        logits = partial[:, 0]
        scores = np.sqrt(np.logaddexp(np.float32(0.0), logits)).astype(np.float32)
        selected = scores[route_indices[position]]
        route_weights[position] = selected / np.sum(selected, dtype=np.float32) * \
            np.float32(1.5)

    def admitted_matrix(name: str, source: str) -> tuple[object, object]:
        weight_info = checkpoint.tensors[name + ".weight"]
        scale_info = checkpoint.tensors[name + ".scale"]
        with checkpoint.open_tensor(name + ".weight") as weight_view, \
                checkpoint.open_tensor(name + ".scale") as scale_view:
            if source == "routed":
                rows, packed_columns = weight_info.shape
                packed = np.frombuffer(weight_view.raw, dtype=np.uint8).reshape(
                    rows, packed_columns
                ).copy()
                scale_codes = np.frombuffer(scale_view.raw, dtype=np.uint8).reshape(
                    rows, packed_columns // 16
                ).copy()
                decoded = _decode_numpy(packed, scale_codes)
                del packed, scale_codes
            else:
                rows, columns = weight_info.shape
                codes = np.frombuffer(weight_view.raw, dtype=np.uint8).reshape(
                    rows, columns
                ).copy()
                scale_codes = np.frombuffer(scale_view.raw, dtype=np.uint8).reshape(
                    rows // 128, columns // 128
                ).copy()
                if bool((scale_codes == 255).any()):
                    raise SourceFormatError("shared expert contains a UE8M0 NaN")
                decoded = _fp8_e4m3fn_table()[codes]
                decoded *= np.repeat(
                    np.ldexp(
                        np.ones((rows, columns // 128), dtype=np.float32),
                        scale_codes[np.arange(rows) // 128].astype(np.int16) - 127,
                    ),
                    128, axis=1,
                )
                del codes, scale_codes
        maxima = np.max(np.abs(decoded), axis=1)
        row_scales = np.where(maxima > 0.0, maxima / 127.0, 1.0).astype(np.float32)
        quantized = np.clip(
            np.rint(decoded / row_scales[:, None]), -127, 127
        ).astype(np.int8)
        return quantized, row_scales

    def block_matvec(matrix: object, scales: object, vector: object) -> object:
        rows, columns = matrix.shape
        partials = np.zeros((rows, 256), dtype=np.float32)
        vector = np.asarray(vector, dtype=np.float32)
        for block in range(columns // 256):
            begin = block * 256
            partials += matrix[:, begin:begin + 256].astype(np.float32) * \
                vector[begin:begin + 256]
        for offset in (128, 64, 32, 16, 8, 4, 2, 1):
            partials[:, :offset] += partials[:, offset:2 * offset]
        return partials[:, 0] * scales

    def expert_output(expert: int | str, vector: object) -> object:
        source = "shared" if expert == "shared" else "routed"
        base = prefix + ".ffn." + (
            "shared_experts" if source == "shared" else f"experts.{expert}"
        )
        gate_q, gate_scale = admitted_matrix(base + ".w1", source)
        up_q, up_scale = admitted_matrix(base + ".w3", source)
        gate = np.minimum(
            block_matvec(gate_q, gate_scale, vector), np.float32(10.0)
        )
        up = np.clip(
            block_matvec(up_q, up_scale, vector), -10.0, 10.0
        ).astype(np.float32)
        intermediate = gate / (np.float32(1.0) + np.exp(-gate)) * up
        words = _f32_to_bf16_words(intermediate)
        intermediate = _bf16_to_f32(words.tobytes(), (2048,))
        down_q, down_scale = admitted_matrix(base + ".w2", source)
        return block_matvec(down_q, down_scale, intermediate)

    block_position = 3
    routed_output = np.zeros(4096, dtype=np.float32)
    for slot, expert in enumerate(route_indices[block_position]):
        routed_output += route_weights[block_position, slot] * expert_output(
            int(expert), ffn_inputs[block_position]
        )
    ffn_output = routed_output + expert_output(
        "shared", ffn_inputs[block_position]
    )
    block_output = (
        ffn_posts[block_position][:, None] * ffn_output[None, :] +
        np.matmul(ffn_combinations[block_position].T, updated[block_position])
    ).astype("<f4")

    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"attention oracle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    ffn_root = partial / "ffn"
    ffn_root.mkdir()
    ffn_entries: list[dict[str, object]] = []
    for expert in sorted(set(int(value) for value in route_indices[3])):
        relative = Path("ffn") / f"expert-{expert:03d}"
        manifest = _export_deepseek_extents(
            checkpoint,
            names=tuple(
                f"{prefix}.ffn.experts.{expert}.{projection}.{kind}"
                for projection in ("w1", "w3", "w2")
                for kind in ("weight", "scale")
            ),
            output=partial / relative, layer=layer, expert=expert,
            format_name="deepseek-compact-expert-extents-v1",
            source_abi="deepseek-fp4-e2m1-ue8m0-block32-v1",
        )
        ffn_entries.append({
            "expert": expert, "kind": "routed", "bytes": manifest["bytes"],
            "sha256": manifest["combined"]["sha256"],
            "descriptor": str(relative / "extents.tsv"),
        })
    relative = Path("ffn") / "shared"
    shared_manifest = _export_deepseek_extents(
        checkpoint,
        names=tuple(
            f"{prefix}.ffn.shared_experts.{projection}.{kind}"
            for projection in ("w1", "w3", "w2")
            for kind in ("weight", "scale")
        ),
        output=partial / relative, layer=layer, expert="shared",
        format_name="deepseek-fp8-shared-expert-extents-v1",
        source_abi="deepseek-fp8-e4m3-ue8m0-block128-v1",
    )
    ffn_entries.append({
        "expert": 256, "kind": "shared", "bytes": shared_manifest["bytes"],
        "sha256": shared_manifest["combined"]["sha256"],
        "descriptor": str(relative / "extents.tsv"),
    })
    with (ffn_root / "ffn-set.tsv").open("x", encoding="utf-8", newline="\n") as index:
        index.write("deepseek-ffn-route-set-v2\n")
        for entry in ffn_entries:
            index.write(
                f"{layer}\t{entry['expert']}\t{entry['kind']}\t{entry['bytes']}\t"
                f"{entry['sha256']}\t{entry['descriptor']}\n"
            )
        index.flush()
        os.fsync(index.fileno())
    with (partial / "streams.f32").open("xb") as file:
        file.write(np.asarray(streams, dtype="<f4").tobytes())
        file.flush()
        os.fsync(file.fileno())
    with (partial / "output.f32").open("xb") as file:
        file.write(updated.tobytes())
        file.flush()
        os.fsync(file.fileno())
    with (partial / "cosine.f32").open("xb") as file:
        file.write(np.asarray(cosines, dtype="<f4").tobytes())
        file.flush()
        os.fsync(file.fileno())
    with (partial / "sine.f32").open("xb") as file:
        file.write(np.asarray(sines, dtype="<f4").tobytes())
        file.flush()
        os.fsync(file.fileno())
    with (partial / "router-scores.f32").open("xb") as file:
        file.write(np.asarray(route_weights, dtype="<f4").tobytes())
        file.flush()
        os.fsync(file.fileno())
    with (partial / "router-indices.i32").open("xb") as file:
        file.write(np.asarray(route_indices, dtype="<i4").tobytes())
        file.flush()
        os.fsync(file.fileno())
    with (partial / "block-output.f32").open("xb") as file:
        file.write(block_output.tobytes())
        file.flush()
        os.fsync(file.fileno())
    result = {
        "format": "deepseek-attention-decode4-oracle-v1",
        "layer": layer,
        "compress_ratio": ratio,
        "positions": [0, 3],
        "stream_values": int(streams.size),
        "output_values": int(updated.size),
        "router_tokens": 4,
        "router_top_k": 6,
        "block_position": block_position,
        "ffn_experts": [entry["expert"] for entry in ffn_entries],
    }
    atomic_json(partial / "manifest.json", result)
    os.replace(partial, output)
    return result


def _deepseek_index_prepare_reference(values: object, cosine: object,
                                      sine: object) -> object:
    """Reference RoPE64, Hadamard128 and block-32 E2M1 QAT for the indexer."""
    values = np.asarray(values, dtype=np.float32)
    rows = values.reshape(-1, 128)
    result = np.empty_like(rows)
    table = np.asarray((0, .5, 1, 1.5, 2, 3, 4, 6,
                        -0., -.5, -1, -1.5, -2, -3, -4, -6), dtype=np.float32)
    for row_index, row in enumerate(rows):
        base_words = _f32_to_bf16_words(row)
        transformed = _bf16_to_f32(base_words.tobytes(), (128,)).copy()
        for pair in range(32):
            left, right = transformed[64 + 2 * pair:66 + 2 * pair]
            transformed[64 + 2 * pair] = left * cosine[pair] - right * sine[pair]
            transformed[65 + 2 * pair] = right * cosine[pair] + left * sine[pair]
        stride = 1
        while stride < 128:
            previous = transformed.copy()
            for index in range(128):
                other = previous[index ^ stride]
                transformed[index] = previous[index] + other if not index & stride else other - previous[index]
            stride *= 2
        transformed *= np.float32(128.0 ** -0.5)
        for first in range(0, 128, 32):
            block = transformed[first:first + 32]
            maximum = max(float(np.max(np.abs(block))), 6.0 * 2.0**-126)
            scale = math.ldexp(1.0, math.ceil(math.log2(maximum / 6.0)))
            distances = np.abs(block[:, None] / scale - table[None, :])
            transformed[first:first + 32] = table[np.argmin(distances, axis=1)] * scale
        result[row_index] = transformed
    return _f32_to_bf16_words(result.reshape(values.shape))


def _deepseek_csa_ratio4_reference(
    inputs: object, wkv: object, wgate: object, ape: object, norm: object,
    *, epsilon: float = 1e-6,
) -> object:
    """Reference the official decode-phase overlap compressor for ratio four."""

    return _deepseek_csa_reference(
        inputs, wkv, wgate, ape, norm, ratio=4, epsilon=epsilon
    )


def _deepseek_csa_reference(
    inputs: object, wkv: object, wgate: object, ape: object, norm: object,
    *, ratio: int, epsilon: float = 1e-6,
) -> object:
    """Reference the official decode compressor for ratio four or 128."""

    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek CSA qualification")
    inputs = np.asarray(inputs, dtype=np.float32)
    if ratio not in (4, 128):
        raise SourceFormatError("unsupported DeepSeek CSA ratio")
    head_dim = int(norm.shape[0])
    if head_dim not in (128, 512):
        raise SourceFormatError("invalid DeepSeek compressor head dimension")
    width = 2 * head_dim if ratio == 4 else head_dim
    rows = 8 if ratio == 4 else ratio
    if inputs.shape != (ratio, 4096) or wkv.shape != (width, 4096):
        raise SourceFormatError("invalid DeepSeek CSA geometry")
    if wgate.shape != wkv.shape or ape.shape != (ratio, width):
        raise SourceFormatError("invalid DeepSeek CSA parameters")
    kv_state = np.zeros((rows, width), dtype=np.float32)
    score_state = np.full((rows, width), -np.inf, dtype=np.float32)
    for position in range(ratio):
        kv = np.matmul(wkv, inputs[position], dtype=np.float32)
        score = np.matmul(wgate, inputs[position], dtype=np.float32)
        state_row = ratio + position if ratio == 4 else position
        kv_state[state_row] = kv
        score_state[state_row] = score + ape[position]
    if ratio == 4:
        candidates = np.concatenate(
            (kv_state[:4, :head_dim], kv_state[4:, head_dim:]), axis=0
        )
        logits = np.concatenate(
            (score_state[:4, :head_dim], score_state[4:, head_dim:]), axis=0
        )
    else:
        candidates = kv_state
        logits = score_state
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    weights = np.exp(shifted)
    weights /= np.sum(weights, axis=0, keepdims=True)
    pooled = np.sum(candidates * weights, axis=0, dtype=np.float32)
    inverse = np.float32(
        1.0 / math.sqrt(float(np.mean(np.square(pooled, dtype=np.float32))) + epsilon)
    )
    return np.asarray(pooled * inverse * norm, dtype="<f4")


def export_deepseek_csa_slice(
    checkpoint: SafeTensorCheckpoint, *, layer: int, output: Path
) -> dict[str, object]:
    """Describe one ratio-four CSA compressor and emit a decode oracle."""

    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek CSA qualification")
    validate_deepseek_v4_source(checkpoint)
    ratios = checkpoint.config["compress_ratios"]
    if not 0 <= layer < 43 or ratios[layer] not in (4, 128):
        raise AdapterError("DeepSeek CSA slice requires a compressed-attention layer")
    ratio = ratios[layer]
    width = 1024 if ratio == 4 else 512
    prefix = f"layers.{layer}.attn.compressor"
    names = [
        prefix + ".wkv.weight",
        prefix + ".wgate.weight",
        prefix + ".ape",
        prefix + ".norm.weight",
        f"layers.{layer}.attn.attn_sink",
    ]
    expected = [
        ("BF16", (width, 4096)),
        ("BF16", (width, 4096)),
        ("F32", (ratio, width)),
        ("BF16", (512,)),
        ("F32", (64,)),
    ]
    if ratio == 4:
        names.extend((
            f"layers.{layer}.attn.indexer.weights_proj.weight",
            f"layers.{layer}.attn.indexer.compressor.wkv.weight",
            f"layers.{layer}.attn.indexer.compressor.wgate.weight",
            f"layers.{layer}.attn.indexer.compressor.ape",
            f"layers.{layer}.attn.indexer.compressor.norm.weight",
        ))
        expected.extend((
            ("BF16", (64, 4096)),
            ("BF16", (256, 4096)),
            ("BF16", (256, 4096)),
            ("F32", (4, 256)),
            ("BF16", (128,)),
        ))
    for name, (dtype, shape) in zip(names, expected):
        info = checkpoint.tensors.get(name)
        if info is None or info.dtype != dtype or info.shape != shape:
            raise SourceFormatError(f"invalid DeepSeek CSA tensor: {name}")
    manifest = _export_deepseek_extents(
        checkpoint, names=tuple(names), output=output, layer=layer,
        expert="attention_compressor", format_name="deepseek-csa-slice-v1",
        source_abi="deepseek-csa-mixed-v1",
        target_abi="deepseek-csa-sm86-f32-state-v1",
    )
    arrays: list[object] = []
    for name, (dtype, shape) in zip(names, expected):
        with checkpoint.open_tensor(name) as view:
            arrays.append(
                _bf16_to_f32(view.raw, shape)
                if dtype == "BF16"
                else np.frombuffer(view.raw, dtype="<f4").reshape(shape).copy()
            )
    wkv, wgate, ape, norm, attn_sink = arrays[:5]
    positions = np.arange(ratio * 4096, dtype=np.float32).reshape(ratio, 4096)
    inputs = (
        np.sin(positions * np.float32(0.005)) * np.float32(0.15)
        + np.cos(positions * np.float32(0.002)) * np.float32(0.025)
    ).astype("<f4")
    output_value = _deepseek_csa_reference(
        inputs, wkv, wgate, ape, norm, ratio=ratio
    )
    angles = np.arange(32, dtype=np.float32) * np.float32(0.03125)
    cosine = np.cos(angles).astype("<f4")
    sine = np.sin(angles).astype("<f4")
    cache_words = _deepseek_compressed_kv_reference(output_value, cosine, sine)
    base_cache = _bf16_to_f32(cache_words.tobytes(), (512,))
    dimensions = np.arange(512, dtype=np.float32)
    sparse_cache_values = np.stack(
        [
            base_cache * np.float32(0.85 + row * 0.04)
            + np.sin(dimensions * np.float32(0.01 + row * 0.001)) * np.float32(0.01)
            for row in range(6)
        ]
    )
    sparse_cache_words = _f32_to_bf16_words(sparse_cache_values)
    query_values = (
        np.sin(np.arange(64 * 512, dtype=np.float32).reshape(64, 512) * np.float32(0.004))
        * np.float32(0.12)
    )
    query_words = _f32_to_bf16_words(query_values)
    query = _bf16_to_f32(query_words.tobytes(), (64, 512))
    sparse_cache = _bf16_to_f32(sparse_cache_words.tobytes(), (6, 512))
    sparse_indices = np.asarray((0, 2, 5, -1), dtype="<i4")
    sparse_output = np.empty((64, 512), dtype=np.float32)
    attention_scale = np.float32(512.0 ** -0.5)
    selected = sparse_cache[[0, 2, 5]]
    for head in range(64):
        logits = np.matmul(selected, query[head], dtype=np.float32) * attention_scale
        maximum = max(float(np.max(logits)), float(attn_sink[head]))
        weights = np.exp(logits - maximum)
        denominator = float(np.sum(weights)) + math.exp(float(attn_sink[head]) - maximum)
        sparse_output[head] = np.matmul(weights, selected, dtype=np.float32) / denominator
    sparse_output_words = _f32_to_bf16_words(sparse_output)
    sparse_files = {
        "sparse-q.bf16": query_words,
        "sparse-cache.bf16": sparse_cache_words,
        "sparse-indices.i32": sparse_indices,
        "sparse-output.bf16": sparse_output_words,
    }
    index_selection = None
    if ratio == 4:
        index_weights = arrays[5]
        index_compressed = _deepseek_csa_reference(
            inputs, arrays[6], arrays[7], arrays[8], arrays[9], ratio=4
        )
        index_query_input = (
            np.sin(
                np.arange(64 * 128, dtype=np.float32).reshape(64, 128)
                * np.float32(0.009)
            )
            * np.float32(0.2)
        )
        index_cache_input = np.stack(
            [
                np.cos(
                    np.arange(128, dtype=np.float32)
                    * np.float32(0.017 + row * 0.002)
                )
                * np.float32(0.16)
                for row in range(6)
            ]
        )
        index_cache_input[0] = index_compressed
        index_query_words = _deepseek_index_prepare_reference(
            index_query_input, cosine, sine
        )
        index_cache_words = _deepseek_index_prepare_reference(
            index_cache_input, cosine, sine
        )
        index_query = _bf16_to_f32(index_query_words.tobytes(), (64, 128))
        index_cache = _bf16_to_f32(index_cache_words.tobytes(), (6, 128))
        index_head_weights = np.matmul(index_weights, inputs[-1], dtype=np.float32)
        index_scores = np.empty(6, dtype="<f4")
        index_scale = np.float32(128.0 ** -0.5 * 64.0 ** -0.5)
        for slot in range(6):
            dots = np.sum(index_query * index_cache[slot], axis=1, dtype=np.float32)
            index_scores[slot] = np.sum(
                np.maximum(dots, 0) * index_head_weights, dtype=np.float32
            ) * index_scale
        index_topk = np.argsort(-index_scores, kind="stable")[:3].astype("<i4")
        sparse_files.update({
            "index-compressor-output.f32": index_compressed,
            "index-q-input.f32": np.asarray(index_query_input, dtype="<f4"),
            "index-cache-input.f32": np.asarray(index_cache_input, dtype="<f4"),
            "index-q.bf16": index_query_words,
            "index-cache.bf16": index_cache_words,
            "index-scores.f32": index_scores,
            "index-topk.i32": index_topk,
        })
        index_selection = {
            "query_rows": 64, "cache_slots": 6, "top_k": 3, "head_dim": 128,
        }
    oracle_path = output / "oracle.f32"
    with oracle_path.open("xb") as oracle:
        oracle.write(inputs.tobytes(order="C"))
        oracle.write(output_value.tobytes(order="C"))
        oracle.write(cosine.tobytes(order="C"))
        oracle.write(sine.tobytes(order="C"))
        oracle.flush()
        os.fsync(oracle.fileno())
    cache_path = output / "cache.bf16"
    with cache_path.open("xb") as cache_file:
        cache_file.write(cache_words.tobytes(order="C"))
        cache_file.flush()
        os.fsync(cache_file.fileno())
    for filename, array in sparse_files.items():
        with (output / filename).open("xb") as sparse_file:
            sparse_file.write(array.tobytes(order="C"))
            sparse_file.flush()
            os.fsync(sparse_file.fileno())
    qualification = {
        "format": "deepseek-csa-decode-oracle-v1",
        "layer": layer,
        "compress_ratio": ratio,
        "overlap": ratio == 4,
        "hidden_size": 4096,
        "head_dim": 512,
        "oracle": oracle_path.name,
        "oracle_bytes": oracle_path.stat().st_size,
        "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "cache": cache_path.name,
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "output_l2": float(np.linalg.norm(output_value)),
        "sparse_attention": {
            "query": "sparse-q.bf16",
            "cache": "sparse-cache.bf16",
            "indices": "sparse-indices.i32",
            "output": "sparse-output.bf16",
            "heads": 64,
            "head_dim": 512,
            "cache_slots": 6,
            "selected_slots": 4,
        },
        "index_selection": index_selection,
    }
    atomic_json(output / "oracle.json", qualification)
    manifest["oracle"] = qualification
    atomic_json(output / "manifest.json", manifest)
    return manifest


def export_deepseek_hca_slice(
    checkpoint: SafeTensorCheckpoint,
    *,
    layer: int,
    site: str,
    output: Path,
) -> dict[str, object]:
    """Describe one real HCA site and emit a small deterministic FP32 oracle."""

    if np is None:
        raise SourceFormatError("NumPy is required for DeepSeek HCA qualification")
    validate_deepseek_v4_source(checkpoint)
    if not 0 <= layer < 43 or site not in ("attn", "ffn"):
        raise AdapterError("DeepSeek HCA key is outside configured bounds")
    prefix = f"layers.{layer}.hc_{site}"
    names = (f"{prefix}_fn", f"{prefix}_base", f"{prefix}_scale")
    expected = ((24, 16384), (24,), (3,))
    for name, shape in zip(names, expected):
        info = checkpoint.tensors.get(name)
        if info is None or info.dtype != "F32" or info.shape != shape:
            raise SourceFormatError(f"invalid DeepSeek HCA tensor: {name}")

    manifest = _export_deepseek_extents(
        checkpoint,
        names=names,
        output=output,
        layer=layer,
        expert=f"hc_{site}",
        format_name="deepseek-hca-slice-v1",
        source_abi="deepseek-hca-f32-v1",
        target_abi="deepseek-hca-sm86-f32-v1",
    )
    arrays: list[object] = []
    for name, shape in zip(names, expected):
        with checkpoint.open_tensor(name) as view:
            arrays.append(np.frombuffer(view.raw, dtype="<f4").reshape(shape).copy())
    fn, base, scale = arrays
    positions = np.arange(16384, dtype=np.float32)
    streams = (
        np.sin(positions * np.float32(0.007)) * np.float32(0.2)
        + np.cos(positions * np.float32(0.003)) * np.float32(0.05)
    ).reshape(4, 4096).astype("<f4")
    pre, post, comb, collapsed, updated = _deepseek_hca_reference(
        streams, fn, base, scale
    )
    sublayer = (
        np.sin(np.arange(4096, dtype=np.float32) * np.float32(0.013))
        * np.float32(0.125)
    ).astype("<f4")
    oracle_arrays = (streams, sublayer, pre, post, comb, collapsed, updated)
    oracle_path = output / "oracle.f32"
    with oracle_path.open("xb") as oracle:
        for array in oracle_arrays:
            oracle.write(array.tobytes(order="C"))
        oracle.flush()
        os.fsync(oracle.fileno())
    oracle_bytes = oracle_path.stat().st_size
    oracle_sha256 = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    qualification = {
        "format": "deepseek-hca-oracle-v1",
        "layer": layer,
        "site": site,
        "hidden_size": 4096,
        "hc_mult": 4,
        "sinkhorn_iterations": 20,
        "epsilon": 1e-6,
        "oracle": oracle_path.name,
        "oracle_f32_values": sum(int(array.size) for array in oracle_arrays),
        "oracle_bytes": oracle_bytes,
        "oracle_sha256": oracle_sha256,
        "pre_sum": float(np.sum(pre)),
        "post_sum": float(np.sum(post)),
        "comb_row_sums": [float(value) for value in np.sum(comb, axis=1)],
        "comb_column_sums": [float(value) for value in np.sum(comb, axis=0)],
    }
    atomic_json(output / "oracle.json", qualification)
    manifest["oracle"] = qualification
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _export_deepseek_extents(
    checkpoint: SafeTensorCheckpoint,
    *,
    names: tuple[str, ...],
    output: Path,
    layer: int,
    expert: int | str,
    format_name: str,
    source_abi: str,
    target_abi: str = "deepseek-sm86-int8-per-row-v1",
) -> dict[str, object]:
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"expert bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    tensors: list[dict[str, object]] = []
    total_bytes = 0
    try:
        combined_digest = hashlib.sha256()
        descriptor_path = partial / "extents.tsv"
        with descriptor_path.open("x", encoding="utf-8", newline="\n") as descriptor:
            descriptor.write("deepseek-compact-extents-v1\n")
            for name in names:
                info = checkpoint.tensors[name]
                digest = hashlib.sha256()
                with checkpoint.open_tensor(name) as view:
                    digest.update(view.raw)
                    combined_digest.update(view.raw)
                if "\t" in info.shard or "\n" in info.shard or "\r" in info.shard:
                    raise SourceFormatError("SafeTensors shard name is not descriptor-safe")
                descriptor.write(
                    f"{total_bytes}\t{info.nbytes}\t{info.offset}\t{info.shard}\n"
                )
                tensors.append({
                    "name": name,
                    "shard": info.shard,
                    "source_offset": info.offset,
                    "destination_offset": total_bytes,
                    "dtype": info.dtype,
                    "shape": list(info.shape),
                    "bytes": info.nbytes,
                    "sha256": digest.hexdigest(),
                })
                total_bytes += info.nbytes
            descriptor.flush()
            os.fsync(descriptor.fileno())
        manifest = {
            "format": format_name,
            "source_abi": source_abi,
            "target_abi": target_abi,
            "layer": layer,
            "expert": expert,
            "bytes": total_bytes,
            "combined": {
                "descriptor": descriptor_path.name,
                "bytes": total_bytes,
                "sha256": combined_digest.hexdigest(),
            },
            "tensors": tensors,
        }
        atomic_json(partial / "manifest.json", manifest)
        os.replace(partial, output)
        return manifest
    except Exception:
        # Preserve partial evidence; never delete or alter source tensors.
        raise


def export_deepseek_compact_expert(
    checkpoint: SafeTensorCheckpoint, *, layer: int, expert: int, output: Path
) -> dict[str, object]:
    """Describe one compact routed expert without copying weights."""

    validate_deepseek_v4_source(checkpoint)
    if not 0 <= layer < 43 or not 0 <= expert < 256:
        raise AdapterError("DeepSeek expert bundle key is outside configured bounds")
    prefix = f"layers.{layer}.ffn.experts.{expert}"
    names = tuple(
        f"{prefix}.{projection}.{kind}"
        for projection in ("w1", "w3", "w2")
        for kind in ("weight", "scale")
    )
    return _export_deepseek_extents(
        checkpoint, names=names, output=output, layer=layer, expert=expert,
        format_name="deepseek-compact-expert-extents-v1",
        source_abi="deepseek-fp4-e2m1-ue8m0-block32-v1",
    )


def export_deepseek_routed_catalog(
    checkpoint: SafeTensorCheckpoint, *, output: Path
) -> dict[str, object]:
    """Atomically index every routed expert without copying weight payloads."""

    validate_deepseek_v4_source(checkpoint)
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"routed catalog output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    expert_count = 43 * 256
    extent_count = expert_count * 6
    source_bytes = 0
    try:
        with ExitStack() as stack:
            catalog = stack.enter_context(
                (partial / "catalog.tsv").open(
                    "x", encoding="utf-8", newline="\n"
                )
            )
            extents = stack.enter_context(
                (partial / "extents.tsv").open(
                    "x", encoding="utf-8", newline="\n"
                )
            )
            # Cataloging touches 66,048 tensor extents. Keep one unbuffered
            # handle per shard, but hash through one fixed scratch allocation.
            # Long-lived mmaps make Windows retain the scanned pages in this
            # process's working set, which is unacceptable on a serving host.
            shard_handles: dict[str, object] = {}
            for shard in checkpoint.shards:
                shard_handles[shard] = stack.enter_context(
                    (checkpoint.root / shard).open("rb", buffering=0)
                )
            hash_buffer = bytearray(8 * 1024 * 1024)
            hash_view = memoryview(hash_buffer)
            catalog.write("deepseek-routed-catalog-v1\n")
            extents.write("deepseek-routed-extents-v1\n")
            extent_index = 0
            for layer in range(43):
                for expert in range(256):
                    prefix = f"layers.{layer}.ffn.experts.{expert}"
                    names = tuple(
                        f"{prefix}.{projection}.{kind}"
                        for projection in ("w1", "w3", "w2")
                        for kind in ("weight", "scale")
                    )
                    digest = hashlib.sha256()
                    destination = 0
                    first_extent = extent_index
                    for name in names:
                        info = checkpoint.tensors[name]
                        if any(character in info.shard for character in "\t\r\n"):
                            raise SourceFormatError(
                                "SafeTensors shard name is not descriptor-safe"
                            )
                        handle = shard_handles[info.shard]
                        handle.seek(info.offset)
                        remaining = info.nbytes
                        while remaining:
                            requested = min(remaining, len(hash_buffer))
                            count = handle.readinto(hash_view[:requested])
                            if count is None or count <= 0:
                                raise SourceFormatError(
                                    f"truncated DeepSeek tensor payload: {name}"
                                )
                            digest.update(hash_view[:count])
                            remaining -= count
                        extents.write(
                            f"{destination}\t{info.nbytes}\t{info.offset}\t"
                            f"{info.shard}\n"
                        )
                        destination += info.nbytes
                        extent_index += 1
                    if destination != 13_369_344:
                        raise SourceFormatError(
                            "DeepSeek routed expert has invalid compact bytes"
                        )
                    catalog.write(
                        f"{layer}\t{expert}\t{destination}\t"
                        f"{digest.hexdigest()}\t{first_extent}\t6\n"
                    )
                    source_bytes += destination
                catalog.flush()
                extents.flush()
                print(
                    f"cataloged DeepSeek routed layer {layer + 1}/43 "
                    f"({source_bytes} source bytes)",
                    flush=True,
                )
            if extent_index != extent_count:
                raise SourceFormatError("DeepSeek routed extent count mismatch")
            catalog.flush()
            os.fsync(catalog.fileno())
            extents.flush()
            os.fsync(extents.fileno())
            hash_view.release()
        result = {
            "format": "deepseek-routed-catalog-v1",
            "source_abi": "deepseek-fp4-e2m1-ue8m0-block32-v1",
            "target_abi": "deepseek-sm86-int8-per-row-v1",
            "layers": 43,
            "experts_per_layer": 256,
            "expert_count": expert_count,
            "extent_count": extent_count,
            "source_bytes": source_bytes,
            "device_bytes_per_expert": 25_198_592,
        }
        atomic_json(partial / "manifest.json", result)
        os.replace(partial, output)
        return result
    except Exception:
        raise


@contextmanager
def _exclusive_pack_lock(path: Path):
    """Hold a process-scoped lock without making crash recovery ambiguous."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b", buffering=0)
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise SourceFormatError(
                f"another DeepSeek compact pack process owns {path}"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def pack_deepseek_routed_catalog(
    *, catalog_root: Path, source_root: Path, output: Path, resume: bool = False
) -> dict[str, object]:
    output = output.resolve()
    lock = output.with_name(f".{output.name}.lock")
    with _exclusive_pack_lock(lock):
        return _pack_deepseek_routed_catalog_locked(
            catalog_root=catalog_root, source_root=source_root,
            output=output, resume=resume,
        )


def _pack_deepseek_routed_catalog_locked(
    *, catalog_root: Path, source_root: Path, output: Path, resume: bool
) -> dict[str, object]:
    """Repack routed experts into one aligned compact shard per layer.

    Payload bytes and per-expert SHA-256 remain identical to the authenticated
    source catalog. Completed layer shards are atomic resume boundaries; the
    source checkpoint is never modified or reclaimed.
    """

    stored_bytes = 13_369_344
    layers = 43
    experts_per_layer = 256
    expert_count = layers * experts_per_layer
    catalog_root = catalog_root.resolve()
    source_root = source_root.resolve()
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise SourceFormatError(f"DeepSeek compact pack already exists: {output}")
    if partial.exists() and not resume:
        raise SourceFormatError(
            f"DeepSeek compact pack partial exists; pass --resume: {partial}"
        )
    partial.mkdir(parents=True, exist_ok=resume)

    catalog_lines = (catalog_root / "catalog.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    extent_lines = (catalog_root / "extents.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    if not catalog_lines or catalog_lines[0] != "deepseek-routed-catalog-v1":
        raise SourceFormatError("compact pack requires routed catalog v1")
    if not extent_lines or extent_lines[0] != "deepseek-routed-extents-v1":
        raise SourceFormatError("compact pack requires routed extents v1")
    rows = [line.split("\t") for line in catalog_lines[1:]]
    extents = [line.split("\t") for line in extent_lines[1:]]
    if len(rows) != expert_count or len(extents) != expert_count * 6:
        raise SourceFormatError("DeepSeek source catalog is incomplete")

    normalized: list[tuple[int, int, str, list[tuple[int, int, Path]]]] = []
    for index, row in enumerate(rows):
        if len(row) != 6:
            raise SourceFormatError("invalid DeepSeek source catalog row")
        layer, expert, size, digest, first, count = row
        if (
            int(layer) != index // experts_per_layer
            or int(expert) != index % experts_per_layer
            or int(size) != stored_bytes
            or int(first) != index * 6
            or int(count) != 6
            or len(digest) != 64
        ):
            raise SourceFormatError("invalid DeepSeek source catalog geometry")
        record_extents: list[tuple[int, int, Path]] = []
        destination = 0
        for raw in extents[index * 6 : index * 6 + 6]:
            if len(raw) != 4:
                raise SourceFormatError("invalid DeepSeek source extent row")
            target, size_text, offset_text, relative_text = raw
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise SourceFormatError("DeepSeek source extent escapes root")
            size_value = int(size_text)
            if int(target) != destination or size_value <= 0:
                raise SourceFormatError("DeepSeek source extents are not exact-cover")
            destination += size_value
            record_extents.append((int(offset_text), size_value,
                                   source_root / relative))
        if destination != stored_bytes:
            raise SourceFormatError("DeepSeek source expert has wrong byte count")
        normalized.append((int(layer), int(expert), digest, record_extents))

    def pack_layer(layer: int) -> tuple[str, int]:
        shard_name = f"experts-{layer:02d}.dsc"
        shard = partial / shard_name
        commit = partial / f"{shard_name}.commit.json"
        layer_bytes = experts_per_layer * stored_bytes
        if shard.exists() and commit.exists() and shard.stat().st_size == layer_bytes:
            return "reused", layer_bytes
        temporary = partial / f"{shard_name}.tmp"
        if temporary.exists():
            temporary.unlink()
        buffer = bytearray(8 * 1024 * 1024)
        view = memoryview(buffer)
        layer_digest = hashlib.sha256()
        with ExitStack() as stack:
            handles: dict[Path, object] = {}
            destination = stack.enter_context(temporary.open("xb", buffering=0))
            for index in range(layer * experts_per_layer,
                               (layer + 1) * experts_per_layer):
                _, _, expected_digest, record_extents = normalized[index]
                expert_digest = hashlib.sha256()
                for offset, size_value, path in record_extents:
                    handle = handles.get(path)
                    if handle is None:
                        handle = stack.enter_context(path.open("rb", buffering=0))
                        handles[path] = handle
                    handle.seek(offset)
                    remaining = size_value
                    while remaining:
                        requested = min(remaining, len(buffer))
                        count = handle.readinto(view[:requested])
                        if count is None or count <= 0:
                            raise SourceFormatError(
                                f"truncated DeepSeek compact source: {path.name}"
                            )
                        write_all(destination, view[:count])
                        expert_digest.update(view[:count])
                        layer_digest.update(view[:count])
                        remaining -= count
                if expert_digest.hexdigest() != expected_digest:
                    raise SourceFormatError(
                        f"DeepSeek compact source hash mismatch at record {index}"
                    )
            destination.flush()
            os.fsync(destination.fileno())
        if temporary.stat().st_size != layer_bytes:
            raise SourceFormatError("DeepSeek compact layer has wrong byte count")
        os.replace(temporary, shard)
        atomic_json(commit, {
            "format": "deepseek-compact-layer-commit-v1",
            "layer": layer,
            "bytes": layer_bytes,
            "sha256": layer_digest.hexdigest(),
        })
        view.release()
        return "packed", layer_bytes

    pack_bytes = 0
    for layer in range(layers):
        action, layer_bytes = pack_layer(layer)
        pack_bytes += layer_bytes
        print(
            f"{action} DeepSeek compact layer {layer + 1}/{layers}",
            flush=True,
        )

    packed_catalog = partial / "catalog.tsv.tmp"
    packed_extents = partial / "extents.tsv.tmp"
    for stale in (packed_catalog, packed_extents,
                  partial / "catalog.tsv", partial / "extents.tsv"):
        if stale.exists():
            stale.unlink()
    with packed_catalog.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("deepseek-routed-pack-catalog-v1\n")
        for index, (layer, expert, digest, _) in enumerate(normalized):
            handle.write(
                f"{layer}\t{expert}\t{stored_bytes}\t{digest}\t{index}\t1\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    with packed_extents.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("deepseek-routed-pack-extents-v1\n")
        for layer in range(layers):
            shard_name = f"experts-{layer:02d}.dsc"
            for expert in range(experts_per_layer):
                handle.write(
                    f"0\t{stored_bytes}\t{expert * stored_bytes}\t{shard_name}\n"
                )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(packed_catalog, partial / "catalog.tsv")
    os.replace(packed_extents, partial / "extents.tsv")
    result = {
        "format": "deepseek-routed-compact-pack-v1",
        "source_abi": "deepseek-fp4-e2m1-ue8m0-block32-v1",
        "layers": layers,
        "experts_per_layer": experts_per_layer,
        "expert_count": expert_count,
        "stored_bytes_per_expert": stored_bytes,
        "pack_bytes": pack_bytes,
        "shards": layers,
        "alignment": 4096,
    }
    atomic_json(partial / "manifest.json", result)
    os.replace(partial, output)
    return result


def export_deepseek_io_oracle(
    checkpoint: SafeTensorCheckpoint, *, output: Path, token: int = 42,
    row_chunk: int = 512,
) -> dict[str, object]:
    """Emit an independent embedding → HC head → logits oracle."""

    if np is None:
        raise SourceFormatError("NumPy is required for the DeepSeek I/O oracle")
    validate_deepseek_v4_source(checkpoint)
    if not 0 <= token < 129_280 or row_chunk <= 0:
        raise AdapterError("invalid DeepSeek I/O oracle token or row chunk")
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"DeepSeek I/O oracle output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()

    def tensor(name: str, shape: tuple[int, ...], dtype: str) -> object:
        info = checkpoint.tensors[name]
        if info.shape != shape or info.dtype != dtype:
            raise SourceFormatError(f"invalid DeepSeek I/O tensor: {name}")
        with checkpoint.open_tensor(name) as view:
            if dtype == "BF16":
                return _bf16_to_f32(view.raw, shape).copy()
            return np.frombuffer(view.raw, dtype="<f4").reshape(shape).copy()

    try:
        embedding_info = checkpoint.tensors["embed.weight"]
        if embedding_info.shape != (129_280, 4096) or \
                embedding_info.dtype != "BF16":
            raise SourceFormatError("invalid DeepSeek embedding geometry")
        with (checkpoint.root / embedding_info.shard).open(
            "rb", buffering=0
        ) as source:
            source.seek(embedding_info.offset + token * 4096 * 2)
            raw = source.read(4096 * 2)
        if len(raw) != 4096 * 2:
            raise SourceFormatError("truncated DeepSeek embedding row")
        embedding = _bf16_to_f32(raw, (4096,)).copy()
        streams = np.repeat(embedding[None, :], 4, axis=0)
        flat = streams.reshape(-1).astype(np.float32)
        function = tensor("hc_head_fn", (4, 4 * 4096), "F32")
        base = tensor("hc_head_base", (4,), "F32")
        scale = tensor("hc_head_scale", (1,), "F32")
        inverse = np.float32(
            1.0 / math.sqrt(float(np.mean(np.square(flat, dtype=np.float32))) + 1e-6)
        )
        mixes = np.matmul(function, flat * inverse, dtype=np.float32)
        pre = 1.0 / (1.0 + np.exp(-(mixes * scale[0] + base))) + 1e-6
        collapsed = np.sum(pre[:, None] * streams, axis=0, dtype=np.float32)
        collapsed = _bf16_to_f32(
            _f32_to_bf16_words(collapsed).tobytes(), (4096,)
        ).copy()
        norm = tensor("norm.weight", (4096,), "BF16")
        inverse = np.float32(
            1.0 / math.sqrt(
                float(np.mean(np.square(collapsed, dtype=np.float32))) + 1e-6
            )
        )
        normalized = collapsed * inverse * norm
        normalized = _bf16_to_f32(
            _f32_to_bf16_words(normalized).tobytes(), (4096,)
        ).copy()

        head_info = checkpoint.tensors["head.weight"]
        if head_info.shape != (129_280, 4096) or head_info.dtype != "BF16":
            raise SourceFormatError("invalid DeepSeek output-head geometry")
        logits = np.empty(129_280, dtype="<f4")
        row_bytes = 4096 * 2
        with (checkpoint.root / head_info.shard).open(
            "rb", buffering=0
        ) as source:
            for first in range(0, 129_280, row_chunk):
                count = min(row_chunk, 129_280 - first)
                source.seek(head_info.offset + first * row_bytes)
                raw = source.read(count * row_bytes)
                if len(raw) != count * row_bytes:
                    raise SourceFormatError("truncated DeepSeek output-head rows")
                matrix = _bf16_to_f32(raw, (count, 4096))
                logits[first:first + count] = np.matmul(
                    matrix, normalized, dtype=np.float32
                )
        sampled = int(np.argmax(logits))
        (partial / "streams.f32").write_bytes(streams.astype("<f4").tobytes())
        (partial / "logits.f32").write_bytes(logits.tobytes())
        (partial / "sampled.u32").write_bytes(
            sampled.to_bytes(4, "little", signed=False)
        )
        (partial / "input.u32").write_bytes(
            token.to_bytes(4, "little", signed=False)
        )
        result = {
            "format": "deepseek-io-oracle-v1",
            "token": token,
            "hidden": 4096,
            "vocab": 129_280,
            "sampled_token": sampled,
            "row_chunk": row_chunk,
        }
        atomic_json(partial / "manifest.json", result)
        os.replace(partial, output)
        return result
    except Exception:
        raise


def export_deepseek_shared_expert(
    checkpoint: SafeTensorCheckpoint, *, layer: int, output: Path
) -> dict[str, object]:
    """Describe one always-active FP8 shared expert without copying weights."""

    validate_deepseek_v4_source(checkpoint)
    if not 0 <= layer < 43:
        raise AdapterError("DeepSeek shared expert layer is outside configured bounds")
    prefix = f"layers.{layer}.ffn.shared_experts"
    names = tuple(
        f"{prefix}.{projection}.{kind}"
        for projection in ("w1", "w3", "w2")
        for kind in ("weight", "scale")
    )
    return _export_deepseek_extents(
        checkpoint, names=names, output=output, layer=layer, expert="shared",
        format_name="deepseek-fp8-shared-expert-extents-v1",
        source_abi="deepseek-fp8-e4m3-ue8m0-block128-v1",
    )


def export_deepseek_fp8_matrix(
    checkpoint: SafeTensorCheckpoint, *, name: str, output: Path
) -> dict[str, object]:
    """Describe one block-scaled FP8 dense matrix without copying weights."""

    validate_deepseek_v4_source(checkpoint)
    weight_name = name + ".weight"
    scale_name = name + ".scale"
    if weight_name not in checkpoint.tensors or scale_name not in checkpoint.tensors:
        raise AdapterError(f"DeepSeek FP8 matrix does not exist: {name}")
    weight = checkpoint.tensors[weight_name]
    scale = checkpoint.tensors[scale_name]
    if weight.dtype != "F8_E4M3" or scale.dtype != "F8_E8M0":
        raise AdapterError(f"DeepSeek matrix is not block-scaled FP8: {name}")
    return _export_deepseek_extents(
        checkpoint, names=(weight_name, scale_name), output=output, layer=-1,
        expert=name, format_name="deepseek-fp8-matrix-extents-v1",
        source_abi="deepseek-fp8-e4m3-ue8m0-block128-v1",
    )


def export_deepseek_shared_set(
    checkpoint: SafeTensorCheckpoint, *, output: Path
) -> dict[str, object]:
    """Atomically describe every main-model shared expert without repacking."""

    validate_deepseek_v4_source(checkpoint)
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"shared set output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    entries: list[dict[str, object]] = []
    source_bytes = 0
    try:
        for layer in range(43):
            prefix = f"layers.{layer}.ffn.shared_experts"
            names = tuple(
                f"{prefix}.{projection}.{kind}"
                for projection in ("w1", "w3", "w2")
                for kind in ("weight", "scale")
            )
            relative = Path(f"layer-{layer:02d}")
            manifest = _export_deepseek_extents(
                checkpoint, names=names, output=partial / relative,
                layer=layer, expert="shared",
                format_name="deepseek-fp8-shared-expert-extents-v1",
                source_abi="deepseek-fp8-e4m3-ue8m0-block128-v1",
            )
            entry = {
                "layer": layer,
                "descriptor": str(relative / "extents.tsv"),
                "bytes": manifest["bytes"],
                "sha256": manifest["combined"]["sha256"],
            }
            entries.append(entry)
            source_bytes += int(manifest["bytes"])
        with (partial / "shared-set.tsv").open(
            "x", encoding="utf-8", newline="\n"
        ) as index:
            index.write("deepseek-shared-residency-v1\n")
            for entry in entries:
                index.write(
                    f"{entry['layer']}\t{entry['bytes']}\t{entry['sha256']}\t"
                    f"{entry['descriptor']}\n"
                )
            index.flush()
            os.fsync(index.fileno())
        result = {
            "format": "deepseek-shared-residency-v1",
            "source_abi": "deepseek-fp8-e4m3-ue8m0-block128-v1",
            "target_abi": "deepseek-sm86-int8-per-row-v1",
            "layers": entries,
            "source_bytes": source_bytes,
            "device_bytes": 43 * 25_198_592,
        }
        atomic_json(partial / "manifest.json", result)
        os.replace(partial, output)
        return result
    except Exception:
        raise


def export_deepseek_dense_set(
    checkpoint: SafeTensorCheckpoint, *, output: Path
) -> dict[str, object]:
    """Atomically describe all main-model FP8 dense matrices."""

    validate_deepseek_v4_source(checkpoint)
    bases = sorted(
        name.removesuffix(".weight")
        for name, info in checkpoint.tensors.items()
        if name.startswith("layers.")
        and name.endswith(".weight")
        and info.dtype == "F8_E4M3"
        and ".shared_experts." not in name
    )
    if len(bases) != 236:
        raise AdapterError(f"expected 236 main-model FP8 matrices, got {len(bases)}")
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"dense set output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    entries: list[dict[str, object]] = []
    total_source = 0
    total_device = 0
    maximum_source = 0
    try:
        for index, base in enumerate(bases):
            relative = Path(f"matrix-{index:03d}")
            manifest = _export_deepseek_extents(
                checkpoint, names=(base + ".weight", base + ".scale"),
                output=partial / relative, layer=-1, expert=base,
                format_name="deepseek-fp8-matrix-extents-v1",
                source_abi="deepseek-fp8-e4m3-ue8m0-block128-v1",
            )
            rows, columns = checkpoint.tensors[base + ".weight"].shape
            source_bytes = int(manifest["bytes"])
            device_bytes = rows * columns + rows * 4
            entry = {
                "name": base,
                "rows": rows,
                "columns": columns,
                "descriptor": str(relative / "extents.tsv"),
                "source_bytes": source_bytes,
                "device_bytes": device_bytes,
                "sha256": manifest["combined"]["sha256"],
            }
            entries.append(entry)
            total_source += source_bytes
            total_device += device_bytes
            maximum_source = max(maximum_source, source_bytes)
        with (partial / "dense-set.tsv").open(
            "x", encoding="utf-8", newline="\n"
        ) as index_file:
            index_file.write("deepseek-dense-residency-v1\n")
            for entry in entries:
                index_file.write(
                    f"{entry['name']}\t{entry['rows']}\t{entry['columns']}\t"
                    f"{entry['source_bytes']}\t{entry['device_bytes']}\t"
                    f"{entry['sha256']}\t{entry['descriptor']}\n"
                )
            index_file.flush()
            os.fsync(index_file.fileno())
        result = {
            "format": "deepseek-dense-residency-v1",
            "source_abi": "deepseek-fp8-e4m3-ue8m0-block128-v1",
            "target_abi": "deepseek-sm86-int8-per-row-matrix-v1",
            "matrix_count": len(entries),
            "source_bytes": total_source,
            "device_bytes": total_device,
            "maximum_source_bytes": maximum_source,
            "matrices": entries,
        }
        atomic_json(partial / "manifest.json", result)
        os.replace(partial, output)
        return result
    except Exception:
        raise


def export_deepseek_typed_set(
    checkpoint: SafeTensorCheckpoint, *, output: Path
) -> dict[str, object]:
    """Describe all 834 non-quantized main-model tensors without repacking."""

    validate_deepseek_v4_source(checkpoint)
    names = sorted(
        name
        for name, info in checkpoint.tensors.items()
        if not name.startswith("mtp.") and info.dtype in ("BF16", "F32", "I64")
    )
    if len(names) != 834:
        raise AdapterError(f"expected 834 main-model typed tensors, got {len(names)}")
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"typed set output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    entries: list[dict[str, object]] = []
    total_bytes = 0
    maximum_bytes = 0
    try:
        for index, name in enumerate(names):
            relative = Path(f"tensor-{index:03d}")
            info = checkpoint.tensors[name]
            manifest = _export_deepseek_extents(
                checkpoint,
                names=(name,),
                output=partial / relative,
                layer=-1,
                expert=name,
                format_name="deepseek-typed-tensor-extents-v1",
                source_abi=f"deepseek-{info.dtype.lower()}-v1",
                target_abi=f"deepseek-sm86-{info.dtype.lower()}-v1",
            )
            entry = {
                "name": name,
                "dtype": info.dtype,
                "shape": list(info.shape),
                "descriptor": str(relative / "extents.tsv"),
                "bytes": info.nbytes,
                "sha256": manifest["combined"]["sha256"],
            }
            entries.append(entry)
            total_bytes += info.nbytes
            maximum_bytes = max(maximum_bytes, info.nbytes)
        with (partial / "typed-set.tsv").open(
            "x", encoding="utf-8", newline="\n"
        ) as index_file:
            index_file.write("deepseek-typed-residency-v1\n")
            for entry in entries:
                shape = ",".join(str(value) for value in entry["shape"])
                index_file.write(
                    f"{entry['name']}\t{entry['dtype']}\t{shape}\t{entry['bytes']}\t"
                    f"{entry['sha256']}\t{entry['descriptor']}\n"
                )
            index_file.flush()
            os.fsync(index_file.fileno())
        result = {
            "format": "deepseek-typed-residency-v1",
            "tensor_count": len(entries),
            "source_bytes": total_bytes,
            "device_bytes": total_bytes,
            "maximum_tensor_bytes": maximum_bytes,
            "tensors": entries,
        }
        atomic_json(partial / "manifest.json", result)
        os.replace(partial, output)
        return result
    except Exception:
        raise
