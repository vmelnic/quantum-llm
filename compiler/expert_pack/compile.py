"""Atomic, resumable SafeTensors to Expert Pack conversion."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adapters import AdaptedModel, adapt_checkpoint
from .constants import (
    FORMAT_NAME,
    FORMAT_VERSION,
    HASH_ALGORITHM,
    MANIFEST_SCHEMA,
    MIN_RUNTIME_VERSION,
    MODEL_CONFIG_FILES,
    PACK_ALIGNMENT,
    QUANT_ABI_ID,
    QUANT_GROUP_SIZE,
    QUANT_PROFILE,
    SECTION_ALIGNMENT,
    TOKENIZER_FILES,
)
from .errors import ResumeError
from .safetensors import SafeTensorCheckpoint
from .util import (
    atomic_json,
    canonical_json_bytes,
    fsync_directory,
    fsync_file,
    load_json,
    sha256_bytes,
    sha256_file,
)
from .validate import validate_container, validate_dense_record, validate_expert_record
from .writer import expert_record_size, write_dense_record, write_expert_record

STATE_SCHEMA = "expert-pack-compile-state-v1"
REPORT_SCHEMA = "expert-pack-conversion-report-v1"


@dataclass(frozen=True)
class CompileOptions:
    source: Path
    output: Path
    adapter: str = "olmoe"
    quant_profile: str = QUANT_PROFILE
    alignment: int = PACK_ALIGNMENT
    max_expert_pack_bytes: int = 2 * 1024 * 1024 * 1024
    source_id: str | None = None
    source_revision: str | None = None
    resume: bool = False
    reclaim_source_shards: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_inventory(checkpoint: SafeTensorCheckpoint) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in checkpoint.source_files():
        inventory.append(
            {
                "path": path.relative_to(checkpoint.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _option_contract(options: CompileOptions, source_files: list[dict[str, object]]) -> dict[str, object]:
    return {
        "adapter": options.adapter,
        "quant_profile": options.quant_profile,
        "alignment": options.alignment,
        "max_expert_pack_bytes": options.max_expert_pack_bytes,
        "source_id": options.source_id,
        "source_revision": options.source_revision,
        "reclaim_source_shards": options.reclaim_source_shards,
        "source_files": source_files,
    }


def _new_state(contract_hash: str, source_files: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": STATE_SCHEMA,
        "contract_sha256": contract_hash,
        "source_files": source_files,
        "dense": [],
        "experts": [],
        "reclaimed_source_shards": [],
    }


def _reclaim_consumed_source_shards(
    partial: Path,
    checkpoint: SafeTensorCheckpoint,
    adapted: AdaptedModel,
    next_expert: int,
    state: dict[str, object],
    source_files: list[dict[str, object]],
) -> None:
    """Delete only shards whose last compiled tensor is durably committed.

    This is deliberately opt-in. A journal is fsynced before each unlink, and
    Hugging Face blob targets are removed only when the snapshot symlink
    resolves inside the same model cache's ``blobs`` directory.
    """

    future_shards = {
        tensor.shard
        for expert in adapted.experts[next_expert:]
        for tensor in (expert.gate, expert.up, expert.down)
    }
    inventory = {str(item["path"]): item for item in source_files}
    reclaimed = state.setdefault("reclaimed_source_shards", [])
    if not isinstance(reclaimed, list):
        raise ResumeError("invalid reclaimed source shard state")
    reclaimed_names = {
        str(item.get("path")) for item in reclaimed if isinstance(item, dict)
    }
    journal_path = partial / "source-reclaim-journal.json"
    for shard in checkpoint.shards:
        if shard in future_shards:
            continue
        path = checkpoint.root / shard
        if not path.exists() and not path.is_symlink():
            if shard in reclaimed_names:
                continue
            raise ResumeError(f"consumed source shard disappeared without journal: {shard}")
        target: Path | None = None
        if path.is_symlink():
            target = path.resolve(strict=True)
            if checkpoint.root.parent.name != "snapshots":
                raise ValueError("refusing to reclaim a symlink outside a Hugging Face snapshot")
            blobs = checkpoint.root.parent.parent / "blobs"
            if target.parent != blobs:
                raise ValueError(f"refusing to unlink source target outside cache blobs: {target}")
        item = inventory.get(shard)
        if item is None:
            raise ResumeError(f"source inventory has no shard {shard}")
        entry = {
            "path": shard,
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "experts_committed": next_expert,
            "restore": "huggingface download of the same pinned revision",
        }
        if shard not in reclaimed_names:
            journal = [*reclaimed, entry]
            atomic_json(journal_path, {"schema": "expert-pack-source-reclaim-v1",
                                       "shards": journal})
            fsync_directory(partial)
        path.unlink()
        if target is not None and target.exists():
            target.unlink()
        if shard not in reclaimed_names:
            reclaimed.append(entry)
            reclaimed_names.add(shard)
            atomic_json(partial / "compile-state.json", state)
            fsync_directory(partial)


def _discard_uncommitted_files(partial: Path, state: dict[str, object]) -> None:
    for path in partial.glob("*.tmp"):
        path.unlink()
    committed = {entry["pack"] for entry in state.get("experts", [])}
    for path in partial.glob("experts-*.qpack"):
        if path.name not in committed:
            path.unlink()
    for name in ("manifest.json", "conversion-report.json", "COMPLETED"):
        path = partial / name
        if path.exists():
            path.unlink()


def _validate_resume_prefix(
    partial: Path,
    state: dict[str, object],
    adapted: AdaptedModel,
    alignment: int,
) -> None:
    dense_entries = state.get("dense")
    expert_entries = state.get("experts")
    if not isinstance(dense_entries, list) or not isinstance(expert_entries, list):
        raise ResumeError("partial conversion state has invalid record indexes")
    expected_dense = [info.name for info in adapted.dense]
    actual_dense = [entry.get("name") for entry in dense_entries if isinstance(entry, dict)]
    if actual_dense not in ([], expected_dense):
        raise ResumeError("dense resume state is not empty or complete deterministic prefix")
    if dense_entries:
        dense_path = partial / "dense.qpack"
        if not dense_path.is_file():
            raise ResumeError("committed dense pack is missing")
        for entry in dense_entries:
            validate_dense_record(dense_path, entry, alignment)
        expected_size = sum(entry["stored_bytes"] for entry in dense_entries)
        if dense_path.stat().st_size != expected_size:
            raise ResumeError("committed dense pack contains unindexed bytes")

    expected_keys = [(source.layer, source.expert) for source in adapted.experts]
    actual_keys = [
        (entry.get("layer"), entry.get("expert"))
        for entry in expert_entries
        if isinstance(entry, dict)
    ]
    if actual_keys != expected_keys[: len(actual_keys)]:
        raise ResumeError("expert resume state is not a deterministic prefix")
    records_by_pack: dict[str, list[dict[str, object]]] = {}
    for entry in expert_entries:
        records_by_pack.setdefault(entry["pack"], []).append(entry)
    for pack_name, entries in records_by_pack.items():
        path = partial / pack_name
        if not path.is_file():
            raise ResumeError(f"committed expert pack is missing: {pack_name}")
        cursor = 0
        for entry in entries:
            if entry["offset"] != cursor:
                raise ResumeError(f"gap in committed expert pack: {pack_name}")
            validate_expert_record(path, entry, alignment)
            cursor += entry["stored_bytes"]
        if path.stat().st_size != cursor:
            raise ResumeError(f"committed expert pack contains unindexed bytes: {pack_name}")


def _write_dense_pack(
    partial: Path,
    checkpoint: SafeTensorCheckpoint,
    adapted: AdaptedModel,
    alignment: int,
) -> list[dict[str, object]]:
    temporary = partial / "dense.qpack.tmp"
    entries: list[dict[str, object]] = []
    with temporary.open("w+b") as handle:
        for info in adapted.dense:
            result = write_dense_record(handle, checkpoint, info, "dense.qpack", alignment)
            entries.append(result.entry)
        fsync_file(handle)
    os.replace(temporary, partial / "dense.qpack")
    fsync_directory(partial)
    return entries


def _copy_model_metadata(checkpoint: SafeTensorCheckpoint, partial: Path) -> list[dict[str, object]]:
    destination = partial / "tokenizer"
    destination.mkdir(exist_ok=True)
    names = [name for name in MODEL_CONFIG_FILES if (checkpoint.root / name).is_file()]
    names.extend(name for name in TOKENIZER_FILES if (checkpoint.root / name).is_file())
    tokenizer_payloads = {"tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model"}
    if not tokenizer_payloads.intersection(names):
        raise ValueError("checkpoint has no supported tokenizer payload")
    if "tokenizer_config.json" not in names:
        raise ValueError("checkpoint has no tokenizer_config.json")
    result: list[dict[str, object]] = []
    for name in sorted(set(names)):
        source = checkpoint.root / name
        if not source.is_file():
            continue
        target = destination / name
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, temporary)
        # Windows FlushFileBuffers (used by os.fsync) requires a handle opened
        # for writing; a read-only descriptor fails with EBADF there.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        result.append(
            {
                "path": f"tokenizer/{name}",
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    fsync_directory(destination)
    return result


def _source_identity(options: CompileOptions, checkpoint: SafeTensorCheckpoint) -> tuple[str, str | None]:
    source_id = options.source_id
    if source_id is None:
        configured = checkpoint.config.get("_name_or_path")
        source_id = configured if isinstance(configured, str) and configured else checkpoint.root.name
    revision = options.source_revision
    if revision is None and checkpoint.root.parent.name == "snapshots":
        revision = checkpoint.root.name
    return source_id, revision


def _build_manifest(
    options: CompileOptions,
    checkpoint: SafeTensorCheckpoint,
    adapted: AdaptedModel,
    source_files: list[dict[str, object]],
    dense: list[dict[str, object]],
    experts: list[dict[str, object]],
    metadata_files: list[dict[str, object]],
    partial: Path,
) -> dict[str, object]:
    pack_names = ["dense.qpack"] + sorted({entry["pack"] for entry in experts})
    packs: list[dict[str, object]] = []
    for name in pack_names:
        path = partial / name
        kind = "dense" if name == "dense.qpack" else "experts"
        entries = dense if kind == "dense" else [entry for entry in experts if entry["pack"] == name]
        packs.append(
            {
                "name": name,
                "kind": kind,
                "bytes": path.stat().st_size,
                "record_count": len(entries),
                "sha256": sha256_file(path),
            }
        )
    dense_bytes = sum(entry["stored_bytes"] for entry in dense)
    expert_bytes = sum(entry["stored_bytes"] for entry in experts)
    source_bytes = sum(file["bytes"] for file in source_files if str(file["path"]).endswith(".safetensors"))
    architecture = adapted.architecture
    layers = int(architecture["num_hidden_layers"])
    top_k = int(architecture["num_experts_per_token"])
    active_expert_bytes = 0
    for layer in range(layers):
        layer_entries = [entry for entry in experts if entry["layer"] == layer]
        active_expert_bytes += round(
            top_k * sum(entry["stored_bytes"] for entry in layer_entries) / len(layer_entries)
        )
    source_id, source_revision = _source_identity(options, checkpoint)
    tokenizer_config = checkpoint.root / "tokenizer_config.json"
    tokenizer_metadata = load_json(tokenizer_config) if tokenizer_config.is_file() else {}
    if not isinstance(tokenizer_metadata, dict):
        tokenizer_metadata = {}
    config = checkpoint.config
    pack_bytes = dense_bytes + expert_bytes
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "format": {
            "name": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "minimum_runtime_version": MIN_RUNTIME_VERSION,
        },
        "compatibility": {
            "unknown_fields": "reject",
            "unknown_quant_abi": "reject",
            "unknown_record_version": "reject",
        },
        "source": {
            "id": source_id,
            "revision": source_revision,
            "checkpoint_sha256": sha256_bytes(canonical_json_bytes(source_files)),
            "tensor_count": adapted.source_tensor_count,
            "files": source_files,
        },
        "architecture": architecture,
        "quantization": {
            "profile": QUANT_PROFILE,
            "abi_id": QUANT_ABI_ID,
            "expert_weights": "symmetric-int8",
            "dense_matrix_weights": "symmetric-int8-except-router",
            "router_and_norms": "float32",
            "scale_dtype": "float32",
            "group_size": QUANT_GROUP_SIZE,
            "rounding": "nearest-ties-to-even",
            "zero_points": False,
        },
        "kernel_abi": {
            "id": "expert-pack-sm86-int8-row-v1",
            "quant_abi": QUANT_ABI_ID,
            "gate_up_fused": True,
            "gate_up_order": ["gate", "up"],
            "down_layout": "output-major-row-contiguous",
            "activation": architecture["hidden_activation"],
            "target": "cuda-sm86",
        },
        "alignment": {
            "pack_bytes": options.alignment,
            "section_bytes": SECTION_ALIGNMENT,
            "direct_io_bytes": options.alignment,
            "pinned_host_bytes": SECTION_ALIGNMENT,
            "cuda_bytes": SECTION_ALIGNMENT,
        },
        "tensors": dense,
        "experts": experts,
        "packs": packs,
        "indexes": {
            "dense_sha256": sha256_bytes(canonical_json_bytes(dense)),
            "experts_sha256": sha256_bytes(canonical_json_bytes(experts)),
        },
        "masses": {
            "source_tensor_bytes": source_bytes,
            "pack_bytes": pack_bytes,
            "dense_bytes": dense_bytes,
            "expert_bytes": expert_bytes,
            "active_expert_bytes_per_token": active_expert_bytes,
        },
        "requirements": {
            "resident_dense_bytes": dense_bytes,
            "minimum_vram_bytes": dense_bytes,
            "minimum_ram_bytes": max((entry["stored_bytes"] for entry in experts), default=0) * 2,
            "cold_disk_bytes_per_second_at_10_tps": active_expert_bytes * 10,
            "cold_disk_bytes_per_second_at_30_tps": active_expert_bytes * 30,
            "pcie_bytes_per_second_at_10_tps_without_reuse": active_expert_bytes * 10,
            "estimation_policy": "lower-bound; KV/workspace/cache are added by P0 feasibility planner",
        },
        "tokenizer": {
            "files": metadata_files,
            "chat_template": tokenizer_metadata.get("chat_template"),
            "special_tokens": {
                "bos_token_id": config.get("bos_token_id"),
                "eos_token_id": config.get("eos_token_id"),
                "pad_token_id": config.get("pad_token_id"),
            },
        },
        "integrity": {
            "algorithm": HASH_ALGORITHM,
            "content_sha256": "",
        },
    }
    manifest["integrity"]["content_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def compile_checkpoint(
    options: CompileOptions,
    _record_hook: Callable[[str, int], None] | None = None,
) -> dict[str, object]:
    started_wall = _utc_now()
    started = time.monotonic()
    source = Path(options.source).resolve()
    output = Path(options.output).resolve()
    if options.quant_profile != QUANT_PROFILE:
        raise ValueError(f"unsupported quant profile {options.quant_profile!r}")
    if options.alignment < PACK_ALIGNMENT or options.alignment & (options.alignment - 1):
        raise ValueError(f"alignment must be a power of two >= {PACK_ALIGNMENT}")
    if options.max_expert_pack_bytes < options.alignment:
        raise ValueError("max expert pack bytes is smaller than one aligned record")
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("output must not be inside the immutable source checkpoint")
    if output.exists():
        raise ResumeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")

    checkpoint = SafeTensorCheckpoint(source)
    adapted = adapt_checkpoint(checkpoint, options.adapter)
    source_files = _source_inventory(checkpoint)
    contract = _option_contract(options, source_files)
    contract_hash = sha256_bytes(canonical_json_bytes(contract))
    state_path = partial / "compile-state.json"

    if partial.exists():
        if not options.resume:
            raise ResumeError(f"partial conversion exists; pass --resume: {partial}")
        if (partial / "manifest.json").is_file() and (partial / "COMPLETED").is_file():
            validation = validate_container(partial)
            report = load_json(partial / "conversion-report.json")
            if state_path.exists():
                state_path.unlink()
                fsync_directory(partial)
            os.replace(partial, output)
            fsync_directory(output.parent)
            return {
                "output": str(output),
                "manifest_content_sha256": validation["manifest_content_sha256"],
                "validation": validation,
                "report": report,
            }
        if not state_path.is_file():
            raise ResumeError("partial conversion has no compile-state.json")
        state = load_json(state_path)
        if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
            raise ResumeError("unknown partial conversion state")
        if state.get("contract_sha256") != contract_hash:
            raise ResumeError("resume options or source checkpoint changed")
        _discard_uncommitted_files(partial, state)
        _validate_resume_prefix(partial, state, adapted, options.alignment)
    else:
        partial.mkdir()
        state = _new_state(contract_hash, source_files)
        atomic_json(state_path, state)
        fsync_directory(partial)

    dense_entries = state["dense"]
    if not dense_entries:
        dense_entries = _write_dense_pack(partial, checkpoint, adapted, options.alignment)
        state["dense"] = dense_entries
        atomic_json(state_path, state)
        if _record_hook:
            _record_hook("dense-pack", len(dense_entries))

    expert_entries = state["experts"]
    next_expert = len(expert_entries)
    if options.reclaim_source_shards:
        _reclaim_consumed_source_shards(
            partial, checkpoint, adapted, next_expert, state, source_files
        )
    pack_index = len({entry["pack"] for entry in expert_entries})
    hidden = int(adapted.architecture["hidden_size"])
    intermediate = int(adapted.architecture["intermediate_size"])
    predicted_record_bytes = expert_record_size(hidden, intermediate, options.alignment)
    while next_expert < len(adapted.experts):
        pack_name = f"experts-{pack_index:03d}.qpack"
        temporary = partial / (pack_name + ".tmp")
        pack_entries: list[dict[str, object]] = []
        with temporary.open("w+b") as handle:
            while next_expert < len(adapted.experts):
                if handle.tell() and handle.tell() + predicted_record_bytes > options.max_expert_pack_bytes:
                    break
                result = write_expert_record(
                    handle,
                    checkpoint,
                    adapted.experts[next_expert],
                    pack_name,
                    hidden,
                    intermediate,
                    options.alignment,
                )
                pack_entries.append(result.entry)
                next_expert += 1
            fsync_file(handle)
        os.replace(temporary, partial / pack_name)
        fsync_directory(partial)
        expert_entries.extend(pack_entries)
        state["experts"] = expert_entries
        atomic_json(state_path, state)
        if options.reclaim_source_shards:
            _reclaim_consumed_source_shards(
                partial, checkpoint, adapted, next_expert, state, source_files
            )
        pack_index += 1
        if _record_hook:
            _record_hook("expert-pack", len(expert_entries))

    metadata_files = _copy_model_metadata(checkpoint, partial)
    manifest = _build_manifest(
        options,
        checkpoint,
        adapted,
        source_files,
        dense_entries,
        expert_entries,
        metadata_files,
        partial,
    )
    atomic_json(partial / "manifest.json", manifest)
    report = {
        "schema": REPORT_SCHEMA,
        "started_at": started_wall,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "source": str(source),
        "output": str(output),
        "adapter": options.adapter,
        "quant_profile": options.quant_profile,
        "alignment": options.alignment,
        "max_expert_pack_bytes": options.max_expert_pack_bytes,
        "resumed": options.resume,
        "reclaim_source_shards": options.reclaim_source_shards,
        "reclaimed_source_shards": state.get("reclaimed_source_shards", []),
        "source_checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
        "manifest_content_sha256": manifest["integrity"]["content_sha256"],
        "source_tensor_count": adapted.source_tensor_count,
        "dense_tensor_count": len(dense_entries),
        "expert_count": len(expert_entries),
        "packs": manifest["packs"],
        "masses": manifest["masses"],
    }
    atomic_json(partial / "conversion-report.json", report)
    marker = {
        "format_version": FORMAT_VERSION,
        "manifest_content_sha256": manifest["integrity"]["content_sha256"],
        "manifest_file_sha256": sha256_file(partial / "manifest.json"),
    }
    atomic_json(partial / "COMPLETED", marker)
    fsync_directory(partial)
    validation = validate_container(partial)
    state_path.unlink()
    fsync_directory(partial)
    os.replace(partial, output)
    fsync_directory(output.parent)
    return {
        "output": str(output),
        "manifest_content_sha256": manifest["integrity"]["content_sha256"],
        "validation": validation,
        "report": report,
    }
