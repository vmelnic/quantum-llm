"""Bounded source-to-container numerical qualification.

The qualifier is driven by the published manifest and model program. It does
not know model families, layer counts, or source tensor paths. Every tensor
record is checked against immutable SafeTensors metadata and deterministic
source samples. FP4 blocks are also re-encoded so the payload must match the
declared E2M1/UE8M0 ABI byte for byte.
"""

from __future__ import annotations

import hashlib
import math
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .constants import (
    DTYPE_BYTES,
    FP4_QUANT_ABI_ID,
    FP4_RELU2_EXPERT_ABI_ID,
    FP4_QUANT_GROUP_SIZE,
    QUANT_ABI_ID,
)
from .deepseek_quant import decode_scaled_fp4_e2m1_row
from .errors import ValidationError
from .quant import (
    _decode_float_row,
    _fp4_block_code,
    _fp4_nearest_index,
    _fp4_pack_nibbles,
)
from .safetensors import SafeTensorCheckpoint, TensorInfo, TensorView
from .util import load_json


@dataclass
class _Moments:
    values: int = 0
    source_energy: float = 0.0
    decoded_energy: float = 0.0
    dot: float = 0.0
    squared_error: float = 0.0
    maximum_absolute_error: float = 0.0

    def add(self, source: list[float], decoded: list[float]) -> None:
        if len(source) != len(decoded):
            raise ValidationError("source/decoded sample length mismatch")
        for expected, actual in zip(source, decoded):
            if not math.isfinite(expected) or not math.isfinite(actual):
                raise ValidationError("non-finite numerical quality sample")
            error = actual - expected
            self.values += 1
            self.source_energy += expected * expected
            self.decoded_energy += actual * actual
            self.dot += expected * actual
            self.squared_error += error * error
            self.maximum_absolute_error = max(
                self.maximum_absolute_error, abs(error)
            )

    def merge(self, other: "_Moments") -> None:
        self.values += other.values
        self.source_energy += other.source_energy
        self.decoded_energy += other.decoded_energy
        self.dot += other.dot
        self.squared_error += other.squared_error
        self.maximum_absolute_error = max(
            self.maximum_absolute_error, other.maximum_absolute_error
        )

    def report(self) -> dict[str, float | int]:
        relative_l2 = (
            math.sqrt(self.squared_error / self.source_energy)
            if self.source_energy > 0.0 else 0.0
        )
        denominator = math.sqrt(self.source_energy * self.decoded_energy)
        cosine = self.dot / denominator if denominator > 0.0 else 1.0
        return {
            "values": self.values,
            "relative_l2": relative_l2,
            "cosine": cosine,
            "maximum_absolute_error": self.maximum_absolute_error,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _safe_child(root: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and bool(relative), "manifest path is empty")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValidationError(
            f"manifest path escapes container: {relative}"
        ) from error
    return candidate


def _expert_source_region(
    checkpoint: SafeTensorCheckpoint,
    entry: dict[str, Any],
    role: str,
) -> TensorInfo:
    tensors = entry.get("source_tensors")
    shapes = entry.get("source_shape")
    regions = entry.get("source_regions")
    _require(
        isinstance(tensors, dict) and isinstance(shapes, dict),
        "expert source mapping is incomplete",
    )
    logical_name = tensors.get(role)
    shape = shapes.get(role)
    _require(
        isinstance(logical_name, str)
        and isinstance(shape, list)
        and all(isinstance(value, int) for value in shape),
        f"expert {role} source metadata is invalid",
    )
    if regions is None:
        _require(
            logical_name in checkpoint.tensors,
            f"unknown expert source tensor: {logical_name}",
        )
        return checkpoint.tensors[logical_name]
    _require(
        isinstance(regions, dict) and set(regions) == set(tensors),
        "expert source region map is invalid",
    )
    region = regions.get(role)
    _require(isinstance(region, dict), f"expert {role} source region is invalid")
    physical_name = region.get("tensor")
    byte_offset = region.get("byte_offset")
    byte_count = region.get("bytes")
    region_shape = region.get("shape")
    tensor_bytes = region.get("tensor_bytes")
    tensor_shape = region.get("tensor_shape")
    _require(
        isinstance(physical_name, str)
        and isinstance(byte_offset, int)
        and not isinstance(byte_offset, bool)
        and isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and region_shape == shape,
        f"expert {role} source region metadata is invalid",
    )
    info = checkpoint.tensor_region(
        physical_name, logical_name, tuple(shape), byte_offset
    )
    physical = checkpoint.tensors[physical_name]
    _require(
        info.nbytes == byte_count
        and tensor_bytes == physical.nbytes
        and tensor_shape == list(physical.shape),
        f"expert {role} source region size mismatch",
    )
    return info


def _semantic_bindings(program: Path) -> dict[str, tuple[str, ...]]:
    bindings: dict[str, list[str]] = {}
    operation_kernels: dict[str, str] = {}
    for line_number, line in enumerate(
            program.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if not fields:
            continue
        if fields[0] == "operation":
            _require(len(fields) >= 4, f"invalid operation at line {line_number}")
            operation_kernels[fields[1]] = fields[3]
        if not fields[0].endswith("_tensor"):
            continue
        _require(len(fields) >= 3, f"invalid tensor binding at line {line_number}")
        tensor = fields[-1]
        if fields[0] == "operation_tensor" and len(fields) >= 4:
            owner = operation_kernels.get(fields[1], f"operation.{fields[1]}")
            role = f"{owner}:{fields[-2]}"
        else:
            role = f"{fields[0]}:{fields[-2]}"
        bindings.setdefault(tensor, []).append(role)
    return {name: tuple(sorted(set(roles))) for name, roles in bindings.items()}


def _sample_indices(label: str, count: int, maximum: int) -> tuple[int, ...]:
    _require(count > 0, "cannot sample an empty tensor")
    target = min(count, maximum)
    selected = {0, count // 2, count - 1}
    counter = 0
    while len(selected) < target:
        digest = hashlib.sha256(f"{label}:{counter}".encode()).digest()
        selected.add(int.from_bytes(digest[:8], "little") % count)
        counter += 1
    return tuple(sorted(selected))


def _source_values(view: TensorView, start: int, count: int) -> list[float]:
    element_bytes = DTYPE_BYTES[view.info.dtype]
    raw = view.raw[start * element_bytes : (start + count) * element_bytes]
    values = [float(value) for value in _decode_float_row(raw, view.info.dtype)]
    _require(len(values) == count, f"short source sample for {view.info.name}")
    return values


def _expected_fp4(values: list[float]) -> tuple[bytes, int]:
    padded = values + [0.0] * (FP4_QUANT_GROUP_SIZE - len(values))
    code = _fp4_block_code(max(abs(value) for value in padded))
    scale = math.ldexp(1.0, code - 127)
    quotients = [value / scale for value in padded]
    indices = [_fp4_nearest_index(abs(value)) for value in quotients]
    signs = [1 if value < 0.0 else 0 for value in quotients]
    return _fp4_pack_nibbles(indices, signs), code


def _qualify_fp4(
        pack: BinaryIO, view: TensorView, entry: dict[str, Any],
        samples_per_tensor: int,
        ) -> tuple[_Moments, int, int]:
    shape = tuple(entry["source_shape"])
    columns = shape[-1]
    rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
    padded_columns = (
        (columns + FP4_QUANT_GROUP_SIZE - 1) // FP4_QUANT_GROUP_SIZE
        * FP4_QUANT_GROUP_SIZE
    )
    blocks_per_row = padded_columns // FP4_QUANT_GROUP_SIZE
    total_blocks = rows * blocks_per_row
    moments = _Moments()
    payload_mismatches = 0
    scale_mismatches = 0
    sections = entry["sections"]
    for flat_block in _sample_indices(
            entry["name"], total_blocks, samples_per_tensor):
        row, block = divmod(flat_block, blocks_per_row)
        logical_start = block * FP4_QUANT_GROUP_SIZE
        logical_count = max(
            0, min(FP4_QUANT_GROUP_SIZE, columns - logical_start)
        )
        source = _source_values(
            view, row * columns + logical_start, logical_count
        ) if logical_count else []
        data_offset = (
            entry["offset"] + sections["data"]["offset"]
            + row * (padded_columns // 2) + block * 16
        )
        scale_offset = (
            entry["offset"] + sections["scales"]["offset"]
            + row * blocks_per_row + block
        )
        pack.seek(data_offset)
        payload = pack.read(16)
        pack.seek(scale_offset)
        scale = pack.read(1)
        _require(len(payload) == 16 and len(scale) == 1,
                 f"short FP4 sample read for {entry['name']}")
        expected_payload, expected_scale = _expected_fp4(source)
        payload_mismatches += int(payload != expected_payload)
        scale_mismatches += int(scale[0] != expected_scale)
        decoded = list(decode_scaled_fp4_e2m1_row(payload, scale))
        moments.add(source, decoded[:logical_count])
    return moments, payload_mismatches, scale_mismatches


def _qualify_f32(
        pack: BinaryIO, view: TensorView, entry: dict[str, Any],
        samples_per_tensor: int,
        ) -> tuple[_Moments, int]:
    elements = math.prod(entry["source_shape"])
    block_count = (elements + FP4_QUANT_GROUP_SIZE - 1) // FP4_QUANT_GROUP_SIZE
    moments = _Moments()
    mismatches = 0
    data_offset = entry["offset"] + entry["sections"]["data"]["offset"]
    for block in _sample_indices(entry["name"], block_count, samples_per_tensor):
        start = block * FP4_QUANT_GROUP_SIZE
        count = min(FP4_QUANT_GROUP_SIZE, elements - start)
        source = _source_values(view, start, count)
        pack.seek(data_offset + start * 4)
        raw = pack.read(count * 4)
        _require(len(raw) == count * 4,
                 f"short F32 sample read for {entry['name']}")
        decoded = list(struct.unpack(f"<{count}f", raw))
        mismatches += sum(a != b for a, b in zip(source, decoded))
        moments.add(source, decoded)
    return moments, mismatches


def _qualify_int8(
        pack: BinaryIO, view: TensorView, entry: dict[str, Any],
        samples_per_tensor: int,
        ) -> tuple[_Moments, int, int]:
    shape = tuple(entry["source_shape"])
    _require(len(shape) == 2, f"INT8 tensor is not a matrix: {entry['name']}")
    rows, columns = shape
    moments = _Moments()
    payload_mismatches = 0
    scale_mismatches = 0
    data_offset = entry["offset"] + entry["sections"]["data"]["offset"]
    scale_offset = entry["offset"] + entry["sections"]["scales"]["offset"]
    for row in _sample_indices(entry["name"], rows, samples_per_tensor):
        source = _source_values(view, row * columns, columns)
        maximum = max((abs(value) for value in source), default=0.0)
        expected_scale = maximum / 127.0 if maximum else 1.0
        expected_q = bytes(
            max(-127, min(127, int(round(value / expected_scale)))) & 0xFF
            for value in source
        )
        pack.seek(data_offset + row * columns)
        payload = pack.read(columns)
        pack.seek(scale_offset + row * 4)
        scale_raw = pack.read(4)
        _require(len(payload) == columns and len(scale_raw) == 4,
                 f"short INT8 sample read for {entry['name']}")
        payload_mismatches += int(payload != expected_q)
        scale_mismatches += int(scale_raw != struct.pack("<f", expected_scale))
        scale = struct.unpack("<f", scale_raw)[0]
        decoded = [
            (value - 256 if value > 127 else value) * scale
            for value in payload
        ]
        moments.add(source, decoded)
    return moments, payload_mismatches, scale_mismatches


def _qualify_expert(
        pack: BinaryIO, checkpoint: SafeTensorCheckpoint,
        entry: dict[str, Any], samples_per_tensor: int,
        ) -> tuple[_Moments, int, int, list[dict[str, Any]]]:
    _require(entry.get("quant_abi") in (
                 FP4_QUANT_ABI_ID, FP4_RELU2_EXPERT_ABI_ID),
             "quality gate accepts only FP4 routed expert records")
    tensors = entry.get("source_tensors")
    shapes = entry.get("source_shape")
    sections = entry.get("sections")
    _require(
        isinstance(tensors, dict) and isinstance(shapes, dict)
        and isinstance(sections, dict),
        "expert source mapping is incomplete",
    )
    first_data = sections["gate_up_q"]["offset"]
    first_scales = sections["gate_up_scales"]["offset"]
    components = {
        "down": (
            sections["down_q"]["offset"],
            sections["down_scales"]["offset"],
        ),
    }
    if entry.get("quant_abi") == FP4_RELU2_EXPERT_ABI_ID:
        components["up"] = (first_data, first_scales)
    else:
        gate_elements = math.prod(shapes["gate"])
        components.update({
            "gate": (first_data, first_scales),
            "up": (
                first_data + gate_elements // 2,
                first_scales + gate_elements // FP4_QUANT_GROUP_SIZE,
            ),
        })
    aggregate = _Moments()
    payload_mismatches = scale_mismatches = 0
    reports: list[dict[str, Any]] = []
    for role, (data_offset, scale_offset) in components.items():
        name = tensors[role]
        source_info = _expert_source_region(checkpoint, entry, role)
        pseudo_entry = {
            "name": name,
            "source_shape": shapes[role],
            "offset": entry["offset"],
            "sections": {
                "data": {"offset": data_offset},
                "scales": {"offset": scale_offset},
            },
        }
        with checkpoint.open_tensor(source_info) as view:
            moments, payload_bad, scale_bad = _qualify_fp4(
                pack, view, pseudo_entry, samples_per_tensor
            )
        aggregate.merge(moments)
        payload_mismatches += payload_bad
        scale_mismatches += scale_bad
        reports.append({
            "name": name,
            "quant_abi": entry["quant_abi"],
            "roles": [f"routed_expert:{role}"],
            **moments.report(),
        })
    return aggregate, payload_mismatches, scale_mismatches, reports


def qualify_container_against_source(
        source: Path | str,
        container: Path | str,
        *,
        samples_per_tensor: int = 8,
        maximum_relative_l2: float = 0.20,
        minimum_cosine: float = 0.98,
        ) -> dict[str, Any]:
    """Compare a published container with its declared SafeTensors source."""

    if samples_per_tensor < 3:
        raise ValueError("samples_per_tensor must be at least three")
    if maximum_relative_l2 <= 0.0 or not -1.0 <= minimum_cosine <= 1.0:
        raise ValueError("invalid numerical quality thresholds")
    source_root = Path(source).resolve()
    container_root = Path(container).resolve()
    manifest = load_json(container_root / "manifest.json")
    _require(isinstance(manifest, dict), "container manifest is not an object")
    checkpoint = SafeTensorCheckpoint(source_root)
    entries = manifest.get("tensors")
    _require(isinstance(entries, list) and bool(entries), "manifest has no tensors")
    source_info = manifest.get("source")
    _require(isinstance(source_info, dict), "manifest source identity is absent")
    declared_revision = source_info.get("revision")
    revision_match = (
        isinstance(declared_revision, str) and bool(declared_revision)
        and source_root.name == declared_revision
    )
    declared_files = source_info.get("files")
    _require(isinstance(declared_files, list), "manifest source files are absent")
    sizes_match = all(
        isinstance(item, dict) and isinstance(item.get("path"), str)
        and isinstance(item.get("bytes"), int)
        and (source_root / item["path"]).is_file()
        and (source_root / item["path"]).stat().st_size == item["bytes"]
        for item in declared_files
    )
    experts = manifest.get("experts")
    _require(isinstance(experts, list), "manifest expert index is invalid")
    manifest_names = {entry.get("name") for entry in entries}
    expert_source_names = {
        name for entry in experts
        for name in entry.get("source_tensors", {}).values()
    }
    expert_physical_names = {
        (
            entry["source_regions"][role]["tensor"]
            if isinstance(entry.get("source_regions"), dict)
            else name
        )
        for entry in experts
        for role, name in entry.get("source_tensors", {}).items()
    }
    all_source_names = manifest_names | expert_physical_names
    metadata_match = (
        len(manifest_names) == len(entries)
        and len(expert_source_names) == sum(
            len(entry.get("source_tensors", {})) for entry in experts
        )
        and all_source_names == set(checkpoint.tensors)
        and all(
            checkpoint.tensors[entry["name"]].shape
                == tuple(entry.get("source_shape", ()))
            and checkpoint.tensors[entry["name"]].dtype
                == entry.get("source_dtype")
            for entry in entries
        )
        and all(
            _expert_source_region(checkpoint, entry, role).shape
                == tuple(entry["source_shape"][role])
            and _expert_source_region(checkpoint, entry, role).dtype
                == entry["source_dtype"][role]
            for entry in experts
            for role in entry["source_tensors"]
        )
    )
    program_info = manifest.get("model_program")
    _require(isinstance(program_info, dict), "manifest model program is absent")
    program = _safe_child(container_root, program_info.get("path"))
    bindings = _semantic_bindings(program)
    auxiliary = manifest.get("auxiliary_tensors", [])
    _require(
        isinstance(auxiliary, list)
        and all(isinstance(name, str) for name in auxiliary)
        and len(set(auxiliary)) == len(auxiliary)
        and set(auxiliary) <= manifest_names,
        "manifest auxiliary tensor index is invalid",
    )
    unreferenced = sorted(manifest_names - set(bindings) - set(auxiliary))
    unknown_bindings = sorted(set(bindings) - manifest_names)

    packs = manifest.get("packs")
    _require(isinstance(packs, list) and bool(packs),
             "manifest declares no packs")
    pack_index = {
        item.get("name"): item for item in packs
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    _require(len(pack_index) == len(packs),
             "manifest pack names are invalid or duplicated")
    tensor_pack_names = {
        entry.get("pack") for entry in entries + experts
    }
    _require(
        all(isinstance(name, str) and name in pack_index
            for name in tensor_pack_names),
        "tensor index references an undeclared pack",
    )
    pack_paths = {
        name: _safe_child(container_root, name)
        for name in tensor_pack_names
    }
    _require(
        all(path.stat().st_size == pack_index[name].get("bytes")
            for name, path in pack_paths.items()),
        "tensor pack size does not match manifest",
    )

    aggregate = _Moments()
    fp4_aggregate = _Moments()
    int8_aggregate = _Moments()
    f32_aggregate = _Moments()
    fp4_records = int8_records = f32_records = fp4_expert_records = 0
    payload_mismatches = scale_mismatches = 0
    int8_payload_mismatches = int8_scale_mismatches = f32_mismatches = 0
    per_tensor: list[dict[str, Any]] = []
    with ExitStack() as stack:
        pack_handles = {
            name: stack.enter_context(path.open("rb"))
            for name, path in pack_paths.items()
        }
        for entry in entries:
            name = entry["name"]
            pack = pack_handles[entry["pack"]]
            with checkpoint.open_tensor(name) as view:
                if entry.get("quant_abi") == FP4_QUANT_ABI_ID:
                    moments, payload_bad, scale_bad = _qualify_fp4(
                        pack, view, entry, samples_per_tensor
                    )
                    fp4_records += 1
                    fp4_aggregate.merge(moments)
                    payload_mismatches += payload_bad
                    scale_mismatches += scale_bad
                elif entry.get("quant_abi") == QUANT_ABI_ID:
                    moments, payload_bad, scale_bad = _qualify_int8(
                        pack, view, entry, samples_per_tensor
                    )
                    int8_records += 1
                    int8_aggregate.merge(moments)
                    int8_payload_mismatches += payload_bad
                    int8_scale_mismatches += scale_bad
                elif entry.get("quant_abi") == 0:
                    moments, f32_bad = _qualify_f32(
                        pack, view, entry, samples_per_tensor
                    )
                    f32_records += 1
                    f32_aggregate.merge(moments)
                    f32_mismatches += f32_bad
                else:
                    raise ValidationError(
                        f"quality gate does not accept quant ABI "
                        f"{entry.get('quant_abi')} for {name}"
                    )
            aggregate.merge(moments)
            per_tensor.append({
                "name": name,
                "quant_abi": entry["quant_abi"],
                "roles": list(bindings.get(name, ())),
                **moments.report(),
            })
        for entry in experts:
            pack = pack_handles[entry["pack"]]
            moments, payload_bad, scale_bad, reports = _qualify_expert(
                pack, checkpoint, entry, samples_per_tensor
            )
            aggregate.merge(moments)
            fp4_aggregate.merge(moments)
            payload_mismatches += payload_bad
            scale_mismatches += scale_bad
            fp4_expert_records += 1
            per_tensor.extend(reports)
    aggregate_report = aggregate.report()
    fp4_report = fp4_aggregate.report()
    worst_relative = sorted(
        per_tensor, key=lambda item: item["relative_l2"], reverse=True
    )[:10]
    lowest_cosine = sorted(per_tensor, key=lambda item: item["cosine"])[:10]
    thresholds_pass = (
        fp4_report["values"] > 0
        and fp4_report["relative_l2"] <= maximum_relative_l2
        and fp4_report["cosine"] >= minimum_cosine
    )
    valid = (
        bool(revision_match) and sizes_match and metadata_match
        and not unreferenced and not unknown_bindings
        and payload_mismatches == 0 and scale_mismatches == 0
        and int8_payload_mismatches == 0 and int8_scale_mismatches == 0
        and f32_mismatches == 0 and thresholds_pass
    )
    return {
        "schema_version": 1,
        "valid": valid,
        "source": {
            "id": source_info.get("id"),
            "revision": declared_revision,
            "revision_path_match": bool(revision_match),
            "file_sizes_match": sizes_match,
            "tensor_metadata_match": metadata_match,
        },
        "container": str(container_root),
        "records": {
            "total": len(entries) + len(experts),
            "tensor_fp4": fp4_records,
            "tensor_int8": int8_records,
            "tensor_f32": f32_records,
            "expert_fp4": fp4_expert_records,
            "unreferenced": unreferenced,
            "unknown_program_bindings": unknown_bindings,
        },
        "sampled": {
            "blocks_per_tensor": samples_per_tensor,
            "values": aggregate.values,
            "fp4_payload_mismatches": payload_mismatches,
            "fp4_scale_mismatches": scale_mismatches,
            "int8_payload_mismatches": int8_payload_mismatches,
            "int8_scale_mismatches": int8_scale_mismatches,
            "f32_value_mismatches": f32_mismatches,
        },
        "thresholds": {
            "maximum_relative_l2": maximum_relative_l2,
            "minimum_cosine": minimum_cosine,
            "pass": thresholds_pass,
        },
        "aggregate": {
            "all": aggregate_report,
            "fp4": fp4_report,
            "int8": int8_aggregate.report(),
            "f32": f32_aggregate.report(),
        },
        "worst_relative_l2": worst_relative,
        "lowest_cosine": lowest_cosine,
    }
