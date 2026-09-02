"""Independent fail-closed reader/validator for completed Expert Packs."""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any

from .constants import (
    DENSE_HEADER_STRUCT,
    DENSE_MAGIC,
    EXPERT_HEADER_STRUCT,
    EXPERT_MAGIC,
    FLAG_GATE_UP_FUSED,
    FLAG_PER_ROW_SCALES,
    FLAG_ROW_MAJOR,
    FLAG_SYMMETRIC,
    FORMAT_NAME,
    FORMAT_VERSION,
    FP4_QUANT_ABI_ID,
    FP4_RELU2_EXPERT_ABI_ID,
    FP4_QUANT_GROUP_SIZE,
    FP4_QUANT_PROFILE,
    FP4_UE8M0_MAX_CODE,
    FP4_UE8M0_MIN_CODE,
    NVFP4_QUANT_ABI_ID,
    NVFP4_QUANT_GROUP_SIZE,
    NVFP4_QUANT_PROFILE,
    HEADER_BYTES,
    MANIFEST_SCHEMA,
    PACK_ALIGNMENT,
    QUANT_ABI_ID,
    QUANT_PROFILE,
    SECTION_ALIGNMENT,
    DTYPE_BYTES,
)
from .errors import ValidationError
from .util import canonical_json_bytes, load_json, sha256_bytes, sha256_file


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _safe_child(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative, "manifest path is empty")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValidationError(f"manifest path escapes container: {relative}") from error
    return candidate


def _payload_hash(path: Path, start: int, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = size
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValidationError(f"short record read in {path.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _validate_scales(path: Path, absolute_offset: int, byte_count: int, label: str) -> None:
    _require(byte_count % 4 == 0, f"{label} scale bytes are not FP32-aligned")
    with path.open("rb") as handle:
        handle.seek(absolute_offset)
        raw = handle.read(byte_count)
    _require(len(raw) == byte_count, f"short {label} scale read")
    values = struct.iter_unpack("<f", raw)
    _require(all(math.isfinite(value[0]) and value[0] > 0.0 for value in values), f"invalid {label} scale")


def _validate_ue8m0_scales(path: Path, absolute_offset: int, byte_count: int, label: str) -> None:
    with path.open("rb") as handle:
        handle.seek(absolute_offset)
        raw = handle.read(byte_count)
    _require(len(raw) == byte_count, f"short {label} scale read")
    _require(
        all(FP4_UE8M0_MIN_CODE <= code <= FP4_UE8M0_MAX_CODE for code in raw),
        f"invalid {label} UE8M0 scale code",
    )


def _validate_e4m3fn_scale_bytes(
    path: Path, absolute_offset: int, byte_count: int, label: str
) -> None:
    with path.open("rb") as handle:
        handle.seek(absolute_offset)
        raw = handle.read(byte_count)
    _require(len(raw) == byte_count, f"short {label} E4M3FN scale read")
    _require(
        all((code & 0x7f) != 0x7f for code in raw),
        f"non-finite {label} E4M3FN scale",
    )


def _validate_positive_scalar(
    path: Path, absolute_offset: int, dtype: str, label: str
) -> None:
    size = 4 if dtype == "F32" else 2
    with path.open("rb") as handle:
        handle.seek(absolute_offset)
        raw = handle.read(size)
    _require(len(raw) == size, f"short {label} global scale")
    if dtype == "F32":
        value = struct.unpack("<f", raw)[0]
    else:
        bits = struct.unpack("<H", raw)[0]
        value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    _require(
        math.isfinite(value) and value > 0.0,
        f"invalid {label} global scale",
    )


def validate_dense_record(path: Path, entry: dict[str, Any], alignment: int) -> None:
    offset = entry.get("offset")
    stored_bytes = entry.get("stored_bytes")
    _require(isinstance(offset, int) and offset >= 0, "invalid dense record offset")
    _require(isinstance(stored_bytes, int) and stored_bytes >= HEADER_BYTES, "invalid dense size")
    _require(offset % alignment == 0 and stored_bytes % alignment == 0, "unaligned dense record")
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(HEADER_BYTES)
    _require(len(raw) == HEADER_BYTES, f"short dense header for {entry.get('name')}")
    _require(all(value == 0 for value in raw[DENSE_HEADER_STRUCT.size :]), "dense header padding is nonzero")
    (
        magic,
        version,
        header_bytes,
        flags,
        quant_abi,
        rank,
        dim0,
        dim1,
        dim2,
        dim3,
        dim4,
        record_bytes,
        data_offset,
        data_bytes,
        scale_offset,
        scale_bytes,
        name_hash,
        payload_hash,
    ) = DENSE_HEADER_STRUCT.unpack(raw[: DENSE_HEADER_STRUCT.size])
    _require(magic == DENSE_MAGIC and version == FORMAT_VERSION, "unknown dense record ABI")
    _require(header_bytes == HEADER_BYTES, "invalid dense header fields")
    _require(record_bytes == stored_bytes, "dense record/manifest size mismatch")
    _require(flags & FLAG_ROW_MAJOR, "dense record is not row-major")
    _require(1 <= rank <= 5, "invalid dense rank")
    dimensions = (dim0, dim1, dim2, dim3, dim4)[:rank]
    if rank < 5:
        _require(dim4 == 0, "invalid dense rank padding")
    _require(list(dimensions) == entry.get("source_shape"), "dense shape mismatch")
    name = entry.get("name")
    _require(isinstance(name, str), "dense record has no name")
    _require(hashlib.sha256(name.encode()).digest() == name_hash, "dense name hash mismatch")
    sections = entry.get("sections")
    _require(isinstance(sections, dict), "dense sections are absent")
    _require(
        sections.get("data") == {"offset": data_offset, "bytes": data_bytes},
        "dense data section mismatch",
    )
    _require(
        sections.get("scales") == {"offset": scale_offset, "bytes": scale_bytes},
        "dense scale section mismatch",
    )
    _require(data_offset >= HEADER_BYTES and data_offset + data_bytes <= record_bytes, "bad dense data span")
    elements = 1
    for dimension in dimensions:
        elements *= dimension
    if quant_abi == QUANT_ABI_ID:
        required = FLAG_SYMMETRIC | FLAG_PER_ROW_SCALES
        _require(flags & required == required, "INT8 dense flags are incomplete")
        _require(rank == 2, "INT8 dense storage requires a matrix")
        _require(scale_offset % SECTION_ALIGNMENT == 0, "unaligned dense scales")
        _require(scale_bytes == dimensions[0] * 4, "dense scale count mismatch")
        _require(data_bytes == elements, "INT8 dense data byte count mismatch")
        _validate_scales(path, offset + scale_offset, scale_bytes, "dense")
    elif quant_abi == FP4_QUANT_ABI_ID:
        required = FLAG_SYMMETRIC | FLAG_PER_ROW_SCALES
        _require(flags & required == required, "FP4 dense flags are incomplete")
        rows = math.prod(dimensions[:-1]) if rank > 1 else 1
        columns = dimensions[-1]
        padded_columns = (
            (columns + FP4_QUANT_GROUP_SIZE - 1)
            // FP4_QUANT_GROUP_SIZE
            * FP4_QUANT_GROUP_SIZE
        )
        _require(scale_offset % SECTION_ALIGNMENT == 0, "unaligned dense scales")
        _require(
            data_bytes == rows * padded_columns // 2,
            "FP4 dense data byte count mismatch",
        )
        _require(
            scale_bytes == rows * padded_columns // FP4_QUANT_GROUP_SIZE,
            "FP4 dense scale count mismatch",
        )
        _validate_ue8m0_scales(
            path, offset + scale_offset, scale_bytes, "dense"
        )
    elif quant_abi == NVFP4_QUANT_ABI_ID:
        _require(flags & FLAG_SYMMETRIC, "NVFP4 dense flags are incomplete")
        _require(rank == 2, "NVFP4 dense storage requires a matrix")
        rows, columns = dimensions
        _require(
            columns % NVFP4_QUANT_GROUP_SIZE == 0,
            "NVFP4 dense columns are not block-aligned",
        )
        local_bytes = rows * columns // NVFP4_QUANT_GROUP_SIZE
        source_tensors = entry.get("source_tensors")
        _require(
            isinstance(source_tensors, dict)
            and set(source_tensors) == {
                "weight", "scale", "weight_global_scale",
                "input_global_scale",
            },
            "NVFP4 dense source map is invalid",
        )
        _require(
            data_bytes == rows * columns // 2,
            "NVFP4 dense data byte count mismatch",
        )
        _require(scale_offset % SECTION_ALIGNMENT == 0,
                 "unaligned NVFP4 dense scales")
        _require(
            scale_bytes in (local_bytes + 6, local_bytes + 8),
            "NVFP4 dense scale byte count mismatch",
        )
        _validate_e4m3fn_scale_bytes(
            path, offset + scale_offset, local_bytes, "dense"
        )
        _validate_positive_scalar(
            path, offset + scale_offset + local_bytes, "F32", "weight"
        )
        _validate_positive_scalar(
            path, offset + scale_offset + local_bytes + 4,
            "F32" if scale_bytes == local_bytes + 8 else "BF16", "input"
        )
    else:
        _require(quant_abi == 0 and scale_offset == 0 and scale_bytes == 0, "unknown dense quant ABI")
        stored_dtype = entry.get("stored_dtype")
        _require(stored_dtype in {"F32", "I64", "BF16"}, "unknown raw dense dtype")
        element_bytes = 8 if stored_dtype == "I64" else 2 if stored_dtype == "BF16" else 4
        _require(data_bytes == elements * element_bytes, "raw dense data byte count mismatch")
    source_dtype = entry.get("source_dtype")
    _require(source_dtype in DTYPE_BYTES, "unknown dense source dtype")
    if quant_abi == NVFP4_QUANT_ABI_ID:
        _require(entry.get("source_bytes") == data_bytes + scale_bytes,
                 "NVFP4 dense source byte mismatch")
    else:
        _require(entry.get("source_bytes") == elements * DTYPE_BYTES[source_dtype], "dense source byte mismatch")
    decoded_element_bytes = 8 if entry.get("stored_dtype") == "I64" else 2 if entry.get("stored_dtype") == "BF16" else 4
    _require(entry.get("decoded_bytes") == elements * decoded_element_bytes, "dense decoded byte mismatch")
    actual_hash = _payload_hash(path, offset + HEADER_BYTES, record_bytes - HEADER_BYTES)
    _require(actual_hash == payload_hash.hex(), "dense payload/header checksum mismatch")
    _require(actual_hash == entry.get("payload_sha256"), "dense payload/manifest checksum mismatch")


def validate_expert_record(path: Path, entry: dict[str, Any], alignment: int) -> None:
    offset = entry.get("offset")
    stored_bytes = entry.get("stored_bytes")
    _require(isinstance(offset, int) and offset >= 0, "invalid expert record offset")
    _require(isinstance(stored_bytes, int) and stored_bytes >= HEADER_BYTES, "invalid expert size")
    _require(offset % alignment == 0 and stored_bytes % alignment == 0, "unaligned expert record")
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(HEADER_BYTES)
    _require(len(raw) == HEADER_BYTES, "short expert header")
    _require(all(value == 0 for value in raw[EXPERT_HEADER_STRUCT.size :]), "expert header padding is nonzero")
    unpacked = EXPERT_HEADER_STRUCT.unpack(raw[: EXPERT_HEADER_STRUCT.size])
    (
        magic,
        version,
        header_bytes,
        flags,
        quant_abi,
        layer,
        expert,
        hidden,
        intermediate,
        fused_rows,
        reserved,
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
    ) = unpacked
    _require(magic == EXPERT_MAGIC and version == FORMAT_VERSION, "unknown expert record ABI")
    _require(header_bytes == HEADER_BYTES and reserved == 0, "invalid expert header fields")
    relu2 = quant_abi == FP4_RELU2_EXPERT_ABI_ID
    native_nvfp4 = quant_abi == NVFP4_QUANT_ABI_ID
    required_flags = FLAG_ROW_MAJOR | FLAG_SYMMETRIC
    if not native_nvfp4:
        required_flags |= FLAG_PER_ROW_SCALES
    if not relu2:
        required_flags |= FLAG_GATE_UP_FUSED
    _require(flags == required_flags, "unknown expert quant/layout ABI")
    fp4 = quant_abi in (
        FP4_QUANT_ABI_ID, FP4_RELU2_EXPERT_ABI_ID,
        NVFP4_QUANT_ABI_ID,
    )
    _require(
        quant_abi in (QUANT_ABI_ID, FP4_QUANT_ABI_ID,
                      FP4_RELU2_EXPERT_ABI_ID, NVFP4_QUANT_ABI_ID),
        "unknown expert quant/layout ABI",
    )
    _require(layer == entry.get("layer") and expert == entry.get("expert"), "expert identity mismatch")
    _require(record_bytes == stored_bytes, "expert record/manifest size mismatch")
    first_rows = (1 if relu2 else 2) * intermediate
    _require(fused_rows == first_rows, "expert first-section row count mismatch")
    if native_nvfp4:
        matrix_elements = hidden * intermediate
        _require(
            hidden % NVFP4_QUANT_GROUP_SIZE == 0
            and intermediate % NVFP4_QUANT_GROUP_SIZE == 0,
            "NVFP4 expert geometry is not block-aligned",
        )
        _require(gate_up_q_bytes == matrix_elements,
                 "NVFP4 gate+up byte count mismatch")
        _require(
            gate_up_scale_bytes ==
            2 * (matrix_elements // NVFP4_QUANT_GROUP_SIZE + 8),
            "NVFP4 gate+up scale byte count mismatch",
        )
        _require(down_q_bytes == matrix_elements // 2,
                 "NVFP4 down byte count mismatch")
        _require(
            down_scale_bytes ==
            matrix_elements // NVFP4_QUANT_GROUP_SIZE + 8,
            "NVFP4 down scale byte count mismatch",
        )
    elif fp4:
        _require(
            hidden % FP4_QUANT_GROUP_SIZE == 0
            and intermediate % FP4_QUANT_GROUP_SIZE == 0,
            "FP4 expert geometry is not block-aligned",
        )
        _require(
            gate_up_q_bytes == first_rows * hidden // 2,
            "expert first-section byte count mismatch",
        )
        _require(
            gate_up_scale_bytes == first_rows * hidden // FP4_QUANT_GROUP_SIZE,
            "expert first-section scale count mismatch",
        )
        _require(down_q_bytes == hidden * intermediate // 2, "down byte count mismatch")
        _require(
            down_scale_bytes == hidden * intermediate // FP4_QUANT_GROUP_SIZE,
            "down scale count mismatch",
        )
    else:
        _require(gate_up_q_bytes == 2 * intermediate * hidden, "gate+up byte count mismatch")
        _require(gate_up_scale_bytes == 2 * intermediate * 4, "gate+up scale count mismatch")
        _require(down_q_bytes == hidden * intermediate, "down byte count mismatch")
        _require(down_scale_bytes == hidden * 4, "down scale count mismatch")
    section_values = {
        "gate_up_q": {"offset": gate_up_q_offset, "bytes": gate_up_q_bytes},
        "gate_up_scales": {"offset": gate_up_scale_offset, "bytes": gate_up_scale_bytes},
        "down_q": {"offset": down_q_offset, "bytes": down_q_bytes},
        "down_scales": {"offset": down_scale_offset, "bytes": down_scale_bytes},
    }
    _require(entry.get("sections") == section_values, "expert section index mismatch")
    previous_end = HEADER_BYTES
    for name in ("gate_up_q", "gate_up_scales", "down_q", "down_scales"):
        section = section_values[name]
        section_offset = section["offset"]
        section_bytes = section["bytes"]
        _require(section_offset % SECTION_ALIGNMENT == 0, f"unaligned expert section {name}")
        _require(section_offset >= previous_end, f"overlapping expert section {name}")
        _require(section_offset + section_bytes <= record_bytes, f"expert section exceeds record: {name}")
        previous_end = section_offset + section_bytes
    if native_nvfp4:
        matrix_local = hidden * intermediate // NVFP4_QUANT_GROUP_SIZE
        for section_offset, label in (
            (gate_up_scale_offset, "gate"),
            (gate_up_scale_offset + matrix_local + 8, "up"),
            (down_scale_offset, "down"),
        ):
            _validate_e4m3fn_scale_bytes(
                path, offset + section_offset, matrix_local, label
            )
            _validate_positive_scalar(
                path, offset + section_offset + matrix_local,
                "F32", label + " weight",
            )
            _validate_positive_scalar(
                path, offset + section_offset + matrix_local + 4,
                "F32", label + " input",
            )
    elif fp4:
        _validate_ue8m0_scales(
            path, offset + gate_up_scale_offset, gate_up_scale_bytes,
            "up" if relu2 else "gate+up",
        )
        _validate_ue8m0_scales(path, offset + down_scale_offset, down_scale_bytes, "down")
    else:
        _validate_scales(path, offset + gate_up_scale_offset, gate_up_scale_bytes, "gate+up")
        _validate_scales(path, offset + down_scale_offset, down_scale_bytes, "down")
    _require(
        entry.get("decoded_bytes") == (2 if relu2 else 3) * hidden * intermediate * 4,
        "expert decoded byte mismatch",
    )
    actual_hash = _payload_hash(path, offset + HEADER_BYTES, record_bytes - HEADER_BYTES)
    _require(actual_hash == payload_hash.hex(), "expert payload/header checksum mismatch")
    _require(actual_hash == entry.get("payload_sha256"), "expert payload/manifest checksum mismatch")


def _manifest_content_hash(manifest: dict[str, Any]) -> str:
    copied = dict(manifest)
    integrity = dict(copied.get("integrity", {}))
    integrity["content_sha256"] = ""
    copied["integrity"] = integrity
    return sha256_bytes(canonical_json_bytes(copied))


def _routed_component_cardinality(path: Path) -> tuple[int, int] | None:
    components: list[tuple[int, int]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(
            f"cannot read runtime model program: {error}"
        ) from error
    for line in lines:
        fields = line.split("\t")
        if not fields or fields[0] != "component":
            continue
        _require(len(fields) == 14, "invalid routed component record")
        try:
            layer_count = int(fields[3])
            expert_count = int(fields[4])
        except ValueError as error:
            raise ValidationError(
                "non-integer routed component cardinality"
            ) from error
        _require(
            layer_count > 0 and expert_count > 0,
            "invalid routed component cardinality",
        )
        components.append((layer_count, expert_count))
    _require(
        len(components) <= 1,
        "Expert Pack v1 indexes at most one routed component",
    )
    return components[0] if components else None


def validate_container(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    _require(root.is_dir(), f"container directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED"
    _require(manifest_path.is_file() and marker_path.is_file(), "container is incomplete")
    manifest = load_json(manifest_path)
    marker = load_json(marker_path)
    _require(isinstance(manifest, dict) and isinstance(marker, dict), "invalid completion metadata")
    expected_top_keys = {
        "schema",
        "format",
        "compatibility",
        "source",
        "architecture",
        "auxiliary_tensors",
        "model_program",
        "quantization",
        "kernel_abi",
        "alignment",
        "tensors",
        "experts",
        "packs",
        "indexes",
        "masses",
        "requirements",
        "tokenizer",
        "integrity",
    }
    _require(set(manifest) == expected_top_keys, "unknown or missing top-level manifest fields")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "unknown manifest schema")
    format_info = manifest.get("format")
    _require(isinstance(format_info, dict), "manifest format block missing")
    _require(format_info.get("name") == FORMAT_NAME, "unknown container format")
    _require(format_info.get("version") == FORMAT_VERSION, "unsupported format version")
    quant = manifest.get("quantization")
    _require(isinstance(quant, dict), "manifest quantization block missing")
    profile = quant.get("profile")
    expected_abis = {
        QUANT_PROFILE: {QUANT_ABI_ID},
        FP4_QUANT_PROFILE: {
            FP4_QUANT_ABI_ID, FP4_RELU2_EXPERT_ABI_ID
        },
        NVFP4_QUANT_PROFILE: {NVFP4_QUANT_ABI_ID},
    }
    _require(
        profile in expected_abis and quant.get("abi_id") in expected_abis[profile],
        "unsupported quant ABI",
    )
    alignment = manifest.get("alignment")
    _require(isinstance(alignment, dict), "manifest alignment block missing")
    pack_alignment = alignment.get("pack_bytes")
    _require(isinstance(pack_alignment, int) and pack_alignment >= PACK_ALIGNMENT, "invalid pack alignment")
    _require(pack_alignment & (pack_alignment - 1) == 0, "pack alignment is not a power of two")

    integrity = manifest.get("integrity")
    _require(isinstance(integrity, dict), "manifest integrity block missing")
    content_hash = _manifest_content_hash(manifest)
    _require(content_hash == integrity.get("content_sha256"), "manifest content checksum mismatch")
    _require(marker.get("manifest_content_sha256") == content_hash, "completion marker content hash mismatch")
    _require(marker.get("manifest_file_sha256") == sha256_file(manifest_path), "completion marker file hash mismatch")
    _require(
        set(marker) == {"format_version", "manifest_content_sha256", "manifest_file_sha256"}
        and marker.get("format_version") == FORMAT_VERSION,
        "invalid completion marker schema",
    )

    model_program = manifest.get("model_program")
    _require(
        isinstance(model_program, dict)
        and set(model_program) == {"format", "path", "bytes", "sha256"}
        and model_program.get("format") == "expert-runtime-model-v1",
        "invalid runtime model program metadata",
    )
    model_program_path = _safe_child(root, model_program["path"])
    _require(model_program_path.is_file(), "runtime model program is missing")
    _require(
        model_program_path.stat().st_size == model_program.get("bytes")
        and sha256_file(model_program_path) == model_program.get("sha256"),
        "runtime model program checksum mismatch",
    )
    with model_program_path.open("rb") as handle:
        _require(
            handle.readline() == b"expert-runtime-model-v1\n",
            "unknown runtime model program format",
        )

    dense = manifest.get("tensors")
    experts = manifest.get("experts")
    packs = manifest.get("packs")
    _require(isinstance(dense, list) and isinstance(experts, list) and isinstance(packs, list), "manifest indexes missing")
    indexes = manifest.get("indexes")
    _require(isinstance(indexes, dict), "manifest index checksums missing")
    _require(indexes.get("dense_sha256") == sha256_bytes(canonical_json_bytes(dense)), "dense index checksum mismatch")
    _require(indexes.get("experts_sha256") == sha256_bytes(canonical_json_bytes(experts)), "expert index checksum mismatch")

    pack_by_name: dict[str, dict[str, Any]] = {}
    for pack in packs:
        _require(isinstance(pack, dict) and isinstance(pack.get("name"), str), "invalid pack entry")
        name = pack["name"]
        _require(name not in pack_by_name, f"duplicate pack {name}")
        path = _safe_child(root, name)
        _require(path.is_file(), f"missing pack {name}")
        _require(path.stat().st_size == pack.get("bytes"), f"pack size mismatch: {name}")
        _require(sha256_file(path) == pack.get("sha256"), f"pack checksum mismatch: {name}")
        pack_by_name[name] = pack

    records_by_pack: dict[str, list[tuple[str, dict[str, Any]]]] = {name: [] for name in pack_by_name}
    dense_names: set[str] = set()
    dense_source_names: set[str] = set()
    for entry in dense:
        _require(isinstance(entry, dict), "invalid dense index entry")
        name = entry.get("name")
        pack_name = entry.get("pack")
        _require(isinstance(name, str) and name not in dense_names, "duplicate/invalid dense tensor")
        _require(pack_name in pack_by_name, f"dense tensor references unknown pack: {name}")
        dense_names.add(name)
        source_tensors = entry.get("source_tensors")
        if source_tensors is None:
            dense_source_names.add(name)
        else:
            _require(
                isinstance(source_tensors, dict)
                and all(isinstance(item, str)
                        for item in source_tensors.values()),
                "dense source tensor map is invalid",
            )
            dense_source_names.update(source_tensors.values())
        records_by_pack[pack_name].append(("dense", entry))

    auxiliary = manifest.get("auxiliary_tensors", [])
    _require(
        isinstance(auxiliary, list)
        and all(isinstance(name, str) for name in auxiliary)
        and len(set(auxiliary)) == len(auxiliary)
        and set(auxiliary) <= dense_names,
        "auxiliary tensor index is invalid",
    )

    expert_keys: set[tuple[int, int]] = set()
    source_expert_names: set[str] = set()
    source_expert_sidecar_names: set[str] = set()
    physical_source_regions: dict[
        str, tuple[int, tuple[int, ...], list[tuple[int, int]]]
    ] = {}
    for entry in experts:
        _require(isinstance(entry, dict), "invalid expert index entry")
        key = (entry.get("layer"), entry.get("expert"))
        _require(all(isinstance(value, int) for value in key), "invalid expert key")
        _require(key not in expert_keys, f"duplicate expert key: {key}")
        _require(
            entry.get("quant_abi") == quant.get("abi_id"),
            "expert record ABI disagrees with manifest",
        )
        pack_name = entry.get("pack")
        _require(pack_name in pack_by_name, f"expert references unknown pack: {key}")
        sources = entry.get("source_tensors")
        expected_roles = (
            {"up", "down"}
            if entry.get("quant_abi") == FP4_RELU2_EXPERT_ABI_ID
            else {"gate", "up", "down"}
        )
        _require(
            isinstance(sources, dict) and set(sources) == expected_roles,
            "expert source map invalid",
        )
        for source_name in sources.values():
            _require(isinstance(source_name, str) and source_name not in source_expert_names, "duplicate expert source tensor")
            source_expert_names.add(source_name)
        sidecars = entry.get("nvfp4_sidecars")
        if entry.get("quant_abi") == NVFP4_QUANT_ABI_ID:
            _require(
                isinstance(sidecars, dict)
                and set(sidecars) == {"gate", "up", "down"}
                and all(
                    isinstance(values, dict)
                    and set(values) == {
                        "scale", "weight_global_scale", "input_global_scale"
                    }
                    for values in sidecars.values()
                ),
                "NVFP4 expert sidecar map is invalid",
            )
            for values in sidecars.values():
                for source_name in values.values():
                    _require(
                        isinstance(source_name, str)
                        and source_name not in source_expert_names,
                        "duplicate NVFP4 expert sidecar",
                    )
                    source_expert_names.add(source_name)
                    source_expert_sidecar_names.add(source_name)
        else:
            _require(sidecars is None, "unexpected expert sidecars")
        regions = entry.get("source_regions")
        if regions is None:
            for role, source_name in sources.items():
                shape = entry.get("source_shape", {}).get(role)
                dtype = entry.get("source_dtype", {}).get(role)
                _require(
                    isinstance(shape, list)
                    and all(isinstance(value, int) and value >= 0 for value in shape)
                    and dtype in DTYPE_BYTES,
                    "legacy expert source metadata is invalid",
                )
                byte_count = math.prod(shape) * DTYPE_BYTES[dtype]
                physical_source_regions[source_name] = (
                    byte_count, tuple(shape), [(0, byte_count)]
                )
        else:
            _require(
                isinstance(regions, dict)
                and set(regions) == expected_roles,
                "expert source region map invalid",
            )
            for role, region in regions.items():
                _require(isinstance(region, dict), "expert source region invalid")
                physical_name = region.get("tensor")
                byte_offset = region.get("byte_offset")
                byte_count = region.get("bytes")
                tensor_bytes = region.get("tensor_bytes")
                shape = region.get("shape")
                tensor_shape = region.get("tensor_shape")
                dtype = entry.get("source_dtype", {}).get(role)
                _require(
                    isinstance(physical_name, str)
                    and isinstance(byte_offset, int) and not isinstance(byte_offset, bool)
                    and isinstance(byte_count, int) and not isinstance(byte_count, bool)
                    and isinstance(tensor_bytes, int) and not isinstance(tensor_bytes, bool)
                    and isinstance(shape, list)
                    and shape == entry.get("source_shape", {}).get(role)
                    and all(isinstance(value, int) and value >= 0 for value in shape)
                    and isinstance(tensor_shape, list)
                    and all(isinstance(value, int) and value >= 0 for value in tensor_shape)
                    and dtype in DTYPE_BYTES
                    and byte_offset >= 0
                    and byte_count == math.prod(shape) * DTYPE_BYTES[dtype]
                    and byte_offset + byte_count <= tensor_bytes,
                    "expert source region metadata is invalid",
                )
                previous = physical_source_regions.get(physical_name)
                if previous is None:
                    physical_source_regions[physical_name] = (
                        tensor_bytes, tuple(tensor_shape),
                        [(byte_offset, byte_offset + byte_count)],
                    )
                else:
                    previous_bytes, previous_shape, spans = previous
                    _require(
                        previous_bytes == tensor_bytes
                        and previous_shape == tuple(tensor_shape),
                        "physical expert tensor metadata is inconsistent",
                    )
                    spans.append((byte_offset, byte_offset + byte_count))
        expert_keys.add(key)
        records_by_pack[pack_name].append(("expert", entry))

    for physical_name, (tensor_bytes, _, spans) in physical_source_regions.items():
        cursor = 0
        for start, end in sorted(spans):
            _require(start == cursor, f"gap/overlap in source tensor {physical_name}")
            cursor = end
        _require(cursor == tensor_bytes, f"unindexed bytes in source tensor {physical_name}")

    architecture = manifest.get("architecture")
    _require(isinstance(architecture, dict), "architecture block missing")
    cardinality = _routed_component_cardinality(model_program_path)
    expected_keys = set()
    if cardinality is not None:
        layer_count, expert_count = cardinality
        expected_keys = {
            (layer, expert)
            for layer in range(layer_count)
            for expert in range(expert_count)
        }
    _require(expert_keys == expected_keys, "expert index is incomplete")

    for pack_name, records in records_by_pack.items():
        path = _safe_child(root, pack_name)
        cursor = 0
        for kind, entry in sorted(records, key=lambda item: item[1]["offset"]):
            _require(entry["offset"] == cursor, f"gap/overlap in pack {pack_name}")
            if kind == "dense":
                validate_dense_record(path, entry, pack_alignment)
            else:
                validate_expert_record(path, entry, pack_alignment)
            cursor += entry["stored_bytes"]
        _require(cursor == path.stat().st_size, f"unindexed bytes in pack {pack_name}")
        _require(len(records) == pack_by_name[pack_name].get("record_count"), f"record count mismatch: {pack_name}")

    actual_qpacks = {path.name for path in root.glob("*.qpack")}
    _require(actual_qpacks == set(pack_by_name), "unindexed or missing qpack files")
    masses = manifest.get("masses")
    _require(isinstance(masses, dict), "mass accounting missing")
    record_bytes = sum(entry["stored_bytes"] for entry in dense + experts)
    pack_bytes = sum(pack["bytes"] for pack in packs)
    _require(record_bytes == pack_bytes == masses.get("pack_bytes"), "pack byte accounting mismatch")
    _require(sum(entry["stored_bytes"] for entry in dense) == masses.get("dense_bytes"), "dense byte accounting mismatch")
    _require(sum(entry["stored_bytes"] for entry in experts) == masses.get("expert_bytes"), "expert byte accounting mismatch")
    resident_dense = masses.get("resident_dense_bytes")
    host_mapped_dense = masses.get("host_mapped_dense_bytes")
    _require(
        isinstance(resident_dense, int) and resident_dense >= 0
        and isinstance(host_mapped_dense, int) and host_mapped_dense >= 0
        and resident_dense + host_mapped_dense <= masses["dense_bytes"],
        "dense placement byte accounting mismatch",
    )
    source_count = manifest.get("source", {}).get("tensor_count")
    _require(
        source_count == len(
            dense_source_names | source_expert_sidecar_names |
            set(physical_source_regions)
        ),
        "source tensor accounting mismatch",
    )

    tokenizer = manifest.get("tokenizer")
    _require(isinstance(tokenizer, dict) and isinstance(tokenizer.get("files"), list), "tokenizer index missing")
    effort_map = tokenizer.get("reasoning_effort_map")
    if effort_map is not None:
        _require(
            isinstance(effort_map, dict) and
            set(effort_map) == {"off", "low", "medium", "xhigh"} and
            all(isinstance(value, str) and value and len(value) <= 32
                for value in effort_map.values()),
            "tokenizer reasoning effort map is invalid",
        )
    sampling = tokenizer.get("sampling")
    if sampling is not None:
        _require(
            isinstance(sampling, dict) and
            sampling.get("schema") == "sampling-profiles-v1",
            "sampling profiles have an unsupported schema",
        )
        profiles = sampling.get("profiles")
        _require(
            isinstance(profiles, dict) and
            set(profiles) == {"thinking", "non_thinking"},
            "sampling profiles are incomplete",
        )
        expected_fields = {
            "temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "frequency_penalty", "repetition_penalty",
        }
        for name, profile in profiles.items():
            _require(
                isinstance(profile, dict) and set(profile) == expected_fields,
                f"sampling profile {name!r} has unknown or missing fields",
            )
            numeric = (
                "temperature", "top_p", "min_p", "presence_penalty",
                "frequency_penalty", "repetition_penalty",
            )
            _require(
                all(not isinstance(profile[field], bool) and
                    isinstance(profile[field], (int, float))
                    for field in numeric) and
                not isinstance(profile["top_k"], bool) and
                isinstance(profile["top_k"], int),
                f"sampling profile {name!r} has an invalid value type",
            )
            _require(
                0.0 <= float(profile["temperature"]) <= 2.0 and
                0.0 < float(profile["top_p"]) <= 1.0 and
                profile["top_k"] >= 0 and
                0.0 <= float(profile["min_p"]) <= 1.0 and
                -2.0 <= float(profile["presence_penalty"]) <= 2.0 and
                float(profile["frequency_penalty"]) == 0.0 and
                float(profile["repetition_penalty"]) == 1.0,
                f"sampling profile {name!r} is outside runtime limits",
            )
    for file_entry in tokenizer["files"]:
        _require(isinstance(file_entry, dict), "invalid tokenizer entry")
        path = _safe_child(root, file_entry.get("path"))
        _require(path.is_file(), f"missing tokenizer/config file: {file_entry.get('path')}")
        _require(path.stat().st_size == file_entry.get("bytes"), "tokenizer/config size mismatch")
        _require(sha256_file(path) == file_entry.get("sha256"), "tokenizer/config checksum mismatch")

    report_path = root / "conversion-report.json"
    _require(report_path.is_file(), "conversion report is missing")
    report = load_json(report_path)
    _require(isinstance(report, dict), "conversion report is invalid")
    _require(report.get("schema") == "expert-pack-conversion-report-v1", "unknown conversion report schema")
    _require(report.get("manifest_content_sha256") == content_hash, "conversion report manifest hash mismatch")
    _require(report.get("source_tensor_count") == source_count, "conversion report tensor count mismatch")
    _require(report.get("dense_tensor_count") == len(dense), "conversion report dense count mismatch")
    _require(report.get("expert_count") == len(experts), "conversion report expert count mismatch")

    return {
        "valid": True,
        "manifest_content_sha256": content_hash,
        "dense_tensors": len(dense),
        "experts": len(experts),
        "packs": len(packs),
        "pack_bytes": pack_bytes,
    }
