"""Bounded real-expert qualification for the DeepSeek source/cache boundary."""

from __future__ import annotations

import hashlib
import math
import os
import time
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


def export_deepseek_compact_expert(
    checkpoint: SafeTensorCheckpoint, *, layer: int, expert: int, output: Path
) -> dict[str, object]:
    """Copy one immutable compact expert into a small CUDA fixture bundle."""

    validate_deepseek_v4_source(checkpoint)
    if not 0 <= layer < 43 or not 0 <= expert < 256:
        raise AdapterError("DeepSeek expert bundle key is outside configured bounds")
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise SourceFormatError(f"expert bundle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    tensors: list[dict[str, object]] = []
    total_bytes = 0
    try:
        prefix = f"layers.{layer}.ffn.experts.{expert}"
        combined_digest = hashlib.sha256()
        combined_path = partial / "expert.compact.bin"
        with combined_path.open("xb") as combined:
            for projection in ("w1", "w3", "w2"):
                for kind in ("weight", "scale"):
                    name = f"{prefix}.{projection}.{kind}"
                    info = checkpoint.tensors[name]
                    filename = f"{projection}.{kind}.bin"
                    digest = hashlib.sha256()
                    with checkpoint.open_tensor(name) as view, (
                        partial / filename
                    ).open("xb") as handle:
                        digest.update(view.raw)
                        combined_digest.update(view.raw)
                        write_all(handle, view.raw)
                        write_all(combined, view.raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                    tensors.append({
                        "name": name,
                        "file": filename,
                        "dtype": info.dtype,
                        "shape": list(info.shape),
                        "bytes": info.nbytes,
                        "sha256": digest.hexdigest(),
                    })
                    total_bytes += info.nbytes
            combined.flush()
            os.fsync(combined.fileno())
        manifest = {
            "format": "deepseek-compact-expert-fixture-v1",
            "source_abi": "deepseek-fp4-e2m1-ue8m0-block32-v1",
            "target_abi": "deepseek-sm86-int8-per-row-v1",
            "layer": layer,
            "expert": expert,
            "bytes": total_bytes,
            "combined": {
                "file": combined_path.name,
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
