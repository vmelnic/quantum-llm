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
    FP4_QUANT_GROUP_SIZE,
    FP4_QUANT_PROFILE,
    FP4_UE8M0_MAX_CODE,
    FP4_UE8M0_MIN_CODE,
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
        reserved,
        record_bytes,
        data_offset,
        data_bytes,
        scale_offset,
        scale_bytes,
        name_hash,
        payload_hash,
    ) = DENSE_HEADER_STRUCT.unpack(raw[: DENSE_HEADER_STRUCT.size])
    _require(magic == DENSE_MAGIC and version == FORMAT_VERSION, "unknown dense record ABI")
    _require(header_bytes == HEADER_BYTES and reserved == 0, "invalid dense header fields")
    _require(record_bytes == stored_bytes, "dense record/manifest size mismatch")
    _require(flags & FLAG_ROW_MAJOR, "dense record is not row-major")
    _require(1 <= rank <= 4, "invalid dense rank")
    dimensions = (dim0, dim1, dim2, dim3)[:rank]
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
        _require(scale_offset % SECTION_ALIGNMENT == 0, "unaligned dense scales")
        _require(scale_bytes == dimensions[0] * 4, "dense scale count mismatch")
        _require(data_bytes == elements, "INT8 dense data byte count mismatch")
        _validate_scales(path, offset + scale_offset, scale_bytes, "dense")
    else:
        _require(quant_abi == 0 and scale_offset == 0 and scale_bytes == 0, "unknown dense quant ABI")
        _require(data_bytes == elements * 4, "FP32 dense data byte count mismatch")
    source_dtype = entry.get("source_dtype")
    _require(source_dtype in DTYPE_BYTES, "unknown dense source dtype")
    _require(entry.get("source_bytes") == elements * DTYPE_BYTES[source_dtype], "dense source byte mismatch")
    _require(entry.get("decoded_bytes") == elements * 4, "dense decoded byte mismatch")
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
    required_flags = FLAG_ROW_MAJOR | FLAG_GATE_UP_FUSED | FLAG_SYMMETRIC | FLAG_PER_ROW_SCALES
    _require(flags == required_flags, "unknown expert quant/layout ABI")
    fp4 = quant_abi == FP4_QUANT_ABI_ID
    _require(quant_abi in (QUANT_ABI_ID, FP4_QUANT_ABI_ID), "unknown expert quant/layout ABI")
    _require(layer == entry.get("layer") and expert == entry.get("expert"), "expert identity mismatch")
    _require(record_bytes == stored_bytes, "expert record/manifest size mismatch")
    _require(fused_rows == 2 * intermediate, "expert fused-row count mismatch")
    if fp4:
        _require(
            hidden % FP4_QUANT_GROUP_SIZE == 0
            and intermediate % FP4_QUANT_GROUP_SIZE == 0,
            "FP4 expert geometry is not block-aligned",
        )
        _require(gate_up_q_bytes == intermediate * hidden, "gate+up byte count mismatch")
        _require(
            gate_up_scale_bytes == 2 * intermediate * hidden // FP4_QUANT_GROUP_SIZE,
            "gate+up scale count mismatch",
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
    if fp4:
        _validate_ue8m0_scales(path, offset + gate_up_scale_offset, gate_up_scale_bytes, "gate+up")
        _validate_ue8m0_scales(path, offset + down_scale_offset, down_scale_bytes, "down")
    else:
        _validate_scales(path, offset + gate_up_scale_offset, gate_up_scale_bytes, "gate+up")
        _validate_scales(path, offset + down_scale_offset, down_scale_bytes, "down")
    _require(entry.get("decoded_bytes") == 3 * hidden * intermediate * 4, "expert decoded byte mismatch")
    actual_hash = _payload_hash(path, offset + HEADER_BYTES, record_bytes - HEADER_BYTES)
    _require(actual_hash == payload_hash.hex(), "expert payload/header checksum mismatch")
    _require(actual_hash == entry.get("payload_sha256"), "expert payload/manifest checksum mismatch")


def _manifest_content_hash(manifest: dict[str, Any]) -> str:
    copied = dict(manifest)
    integrity = dict(copied.get("integrity", {}))
    integrity["content_sha256"] = ""
    copied["integrity"] = integrity
    return sha256_bytes(canonical_json_bytes(copied))


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
    expected_abi = {QUANT_PROFILE: QUANT_ABI_ID, FP4_QUANT_PROFILE: FP4_QUANT_ABI_ID}
    _require(
        profile in expected_abi and quant.get("abi_id") == expected_abi[profile],
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
    for entry in dense:
        _require(isinstance(entry, dict), "invalid dense index entry")
        name = entry.get("name")
        pack_name = entry.get("pack")
        _require(isinstance(name, str) and name not in dense_names, "duplicate/invalid dense tensor")
        _require(pack_name in pack_by_name, f"dense tensor references unknown pack: {name}")
        dense_names.add(name)
        records_by_pack[pack_name].append(("dense", entry))

    expert_keys: set[tuple[int, int]] = set()
    source_expert_names: set[str] = set()
    for entry in experts:
        _require(isinstance(entry, dict), "invalid expert index entry")
        key = (entry.get("layer"), entry.get("expert"))
        _require(all(isinstance(value, int) for value in key), "invalid expert key")
        _require(key not in expert_keys, f"duplicate expert key: {key}")
        pack_name = entry.get("pack")
        _require(pack_name in pack_by_name, f"expert references unknown pack: {key}")
        sources = entry.get("source_tensors")
        _require(isinstance(sources, dict) and set(sources) == {"gate", "up", "down"}, "expert source map invalid")
        for source_name in sources.values():
            _require(isinstance(source_name, str) and source_name not in source_expert_names, "duplicate expert source tensor")
            source_expert_names.add(source_name)
        expert_keys.add(key)
        records_by_pack[pack_name].append(("expert", entry))

    architecture = manifest.get("architecture")
    _require(isinstance(architecture, dict), "architecture block missing")
    layer_count = architecture.get("num_hidden_layers")
    expert_count = architecture.get("num_experts")
    _require(isinstance(layer_count, int) and isinstance(expert_count, int), "expert dimensions absent")
    expected_keys = {(layer, expert) for layer in range(layer_count) for expert in range(expert_count)}
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
    source_count = manifest.get("source", {}).get("tensor_count")
    _require(source_count == len(dense) + len(experts) * 3, "source tensor accounting mismatch")

    tokenizer = manifest.get("tokenizer")
    _require(isinstance(tokenizer, dict) and isinstance(tokenizer.get("files"), list), "tokenizer index missing")
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
