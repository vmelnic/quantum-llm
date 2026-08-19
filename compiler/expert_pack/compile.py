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
    FP4_QUANT_ABI_ID,
    FP4_QUANT_GROUP_SIZE,
    FP4_QUANT_PROFILE,
    HASH_ALGORITHM,
    MANIFEST_SCHEMA,
    MIN_RUNTIME_VERSION,
    MODEL_CONFIG_FILES,
    PACK_ALIGNMENT,
    QUANT_ABI_ID,
    QUANT_GROUP_SIZE,
    QUANT_PROFILE,
    QUANT_PROFILES,
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
    write_all,
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


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _expert_quant_abi(quant_profile: str) -> int:
    if quant_profile == QUANT_PROFILE:
        return QUANT_ABI_ID
    if quant_profile == FP4_QUANT_PROFILE:
        return FP4_QUANT_ABI_ID
    raise ValueError(f"unsupported quant profile {quant_profile!r}")


def _runtime_model_descriptor_bytes(adapted: AdaptedModel, expert_abi: int) -> bytes:
    topology = adapted.runtime_topology
    encoding_abi = 2 if expert_abi == FP4_QUANT_ABI_ID else 1
    encoding = (
        "fp4.e2m1.ue8m0.block32"
        if expert_abi == FP4_QUANT_ABI_ID
        else "int8.symmetric.per-row"
    )

    def atom(value: str) -> str:
        if not value or any(character in value for character in "\t\r\n"):
            raise ValueError(f"runtime model descriptor has invalid atom {value!r}")
        return value

    if tuple(layer.logical_layer for layer in topology.layers) != tuple(
        range(len(topology.layers))
    ):
        raise ValueError("runtime model layers are not contiguous and ordered")
    required = set(topology.required_kernels)
    components = {component.name: component for component in topology.components}
    dense_names = {tensor.name for tensor in adapted.dense}
    if len(components) != len(topology.components):
        raise ValueError("runtime model has duplicate component names")
    if (
        len({role for role, _ in topology.tensor_bindings}) !=
        len(topology.tensor_bindings)
        or any(name not in dense_names for _, name in topology.tensor_bindings)
    ):
        raise ValueError("runtime model has invalid model tensor bindings")
    if (
        not topology.program_inputs
        or not topology.program_outputs
        or len({role for role, _, _ in topology.program_inputs}) !=
        len(topology.program_inputs)
        or len({value for _, value, _ in topology.program_inputs}) !=
        len(topology.program_inputs)
        or len({role for role, _, _ in topology.program_outputs}) !=
        len(topology.program_outputs)
    ):
        raise ValueError("runtime model has invalid program endpoints")
    lines = [
        "expert-runtime-model-v1",
        "\t".join((
            "model", "3", atom(topology.architecture_id),
            str(topology.vocab_size), str(topology.max_context_tokens),
            str(topology.hidden_size),
        )),
    ]
    for key, value in topology.attributes:
        lines.append(f"attribute\t{atom(key)}\t{value}")
    for role, tensor_name in topology.tensor_bindings:
        lines.append(
            f"model_tensor\t{atom(role)}\t{atom(tensor_name)}"
        )
    for role, value, abi in topology.program_inputs:
        lines.append(
            f"program_input\t{atom(role)}\t{atom(value)}\t{atom(abi)}"
        )
    for role, value, abi in topology.program_outputs:
        lines.append(
            f"program_output\t{atom(role)}\t{atom(value)}\t{atom(abi)}"
        )
    for capability, abi in topology.required_kernels:
        lines.append(f"kernel\t{atom(capability)}\t{abi}")
    if topology.exact_decode is not None:
        decode = topology.exact_decode
        if (
            (decode.capability, decode.abi) not in required
            or decode.abi <= 0
            or decode.maximum_emitted_tokens < 2
            or len({key for key, _ in decode.parameters}) != len(decode.parameters)
            or len({role for role, _ in decode.tensor_bindings}) !=
            len(decode.tensor_bindings)
            or any(name not in dense_names for _, name in decode.tensor_bindings)
        ):
            raise ValueError("runtime model has an invalid exact decode program")
        lines.append(
            f"exact_decode\t{atom(decode.capability)}\t{decode.abi}\t"
            f"{decode.maximum_emitted_tokens}"
        )
        for key, value in decode.parameters:
            lines.append(f"exact_decode_parameter\t{atom(key)}\t{value}")
        for role, tensor_name in decode.tensor_bindings:
            lines.append(
                f"exact_decode_tensor\t{atom(role)}\t{atom(tensor_name)}"
            )
    for namespace_offset, component in enumerate(topology.components):
        if (component.execution_capability, component.execution_abi) not in required:
            raise ValueError(
                f"component {component.name} lacks its execution capability"
            )
        if (component.router_capability, component.router_abi) not in required:
            raise ValueError(
                f"component {component.name} lacks its router capability"
            )
        lines.append("\t".join((
            "component", atom(component.name), str(namespace_offset),
            str(component.layer_count), str(component.experts_per_layer),
            str(component.route_width), str(component.shared_experts_per_layer),
            str(component.hidden_size), str(component.intermediate_size),
            atom(component.execution_capability), str(component.execution_abi),
            "1", str(encoding_abi), encoding,
        )))
        for key, value in component.attributes:
            lines.append(
                f"component_attribute\t{atom(component.name)}\t{atom(key)}\t{value}"
            )
        lines.append("\t".join((
            "router", atom(component.name), atom(component.router_capability),
            str(component.router_abi),
        )))
        for key, value in component.router_parameters:
            lines.append(
                f"router_parameter\t{atom(component.name)}\t{atom(key)}\t{value}"
            )
    for layer in topology.layers:
        if (layer.block_capability, layer.block_abi) not in required:
            raise ValueError(
                f"layer {layer.logical_layer} lacks its block capability"
            )
        component = "-" if layer.routed_component is None else atom(
            layer.routed_component
        )
        if layer.routed_component is not None:
            routed = components.get(layer.routed_component)
            if routed is None or layer.component_layer >= routed.layer_count:
                raise ValueError(
                    f"layer {layer.logical_layer} has an invalid component mapping"
                )
        elif layer.component_layer != 0:
            raise ValueError(
                f"dense layer {layer.logical_layer} has a component-local index"
            )
        lines.append("\t".join((
            "layer", str(layer.logical_layer), atom(layer.block_capability),
            str(layer.block_abi), component, str(layer.component_layer),
        )))
        for key, value in layer.parameters:
            lines.append(
                f"layer_parameter\t{layer.logical_layer}\t{atom(key)}\t{value}"
            )
    if not topology.operations:
        raise ValueError("runtime model has no explicit operations")
    previous_layer = -1
    covered_layers: set[int] = set()
    available_values = {
        value: abi for _, value, abi in topology.program_inputs
    }
    for logical_operation, operation in enumerate(topology.operations):
        if operation.logical_layer is not None:
            if (
                operation.logical_layer < previous_layer
                or operation.logical_layer >= len(topology.layers)
            ):
                raise ValueError("runtime operations are not layer-ordered")
            previous_layer = operation.logical_layer
            covered_layers.add(operation.logical_layer)
        if (operation.capability, operation.abi) not in required:
            raise ValueError(
                f"operation {logical_operation} lacks its capability"
            )
        component = "-"
        if operation.routed_component is not None:
            if operation.logical_layer is None:
                raise ValueError(
                    f"model-level operation {logical_operation} is routed"
                )
            routed = components.get(operation.routed_component)
            if routed is None or operation.component_layer >= routed.layer_count:
                raise ValueError(
                    f"operation {logical_operation} has an invalid component mapping"
                )
            component = atom(routed.name)
        elif operation.component_layer != 0:
            raise ValueError(
                f"operation {logical_operation} has a component-local index"
            )
        layer_field = (
            "-" if operation.logical_layer is None
            else str(operation.logical_layer)
        )
        lines.append("\t".join((
            "operation", str(logical_operation), layer_field,
            atom(operation.capability), str(operation.abi), component,
            str(operation.component_layer),
        )))
        for key, value in operation.parameters:
            lines.append(
                f"operation_parameter\t{logical_operation}\t{atom(key)}\t{value}"
            )
        if (
            len({role for role, _ in operation.tensor_bindings}) !=
            len(operation.tensor_bindings)
            or any(name not in dense_names for _, name in operation.tensor_bindings)
        ):
            raise ValueError(
                f"operation {logical_operation} has invalid tensor bindings"
            )
        for role, tensor_name in operation.tensor_bindings:
            lines.append(
                f"operation_tensor\t{logical_operation}\t{atom(role)}\t"
                f"{atom(tensor_name)}"
            )
        if (
            not operation.input_bindings and not operation.output_bindings
        ):
            raise ValueError(
                f"operation {logical_operation} has no runtime data flow"
            )
        if (
            len({role for role, _, _ in operation.input_bindings}) !=
            len(operation.input_bindings)
            or len({role for role, _, _ in operation.output_bindings}) !=
            len(operation.output_bindings)
        ):
            raise ValueError(
                f"operation {logical_operation} has duplicate value ports"
            )
        for role, value, abi in operation.input_bindings:
            if available_values.get(value) != abi:
                raise ValueError(
                    f"operation {logical_operation} consumes unavailable value {value}"
                )
            lines.append(
                f"operation_input\t{logical_operation}\t{atom(role)}\t"
                f"{atom(value)}\t{atom(abi)}"
            )
        for role, value, abi in operation.output_bindings:
            if value in available_values:
                raise ValueError(
                    f"operation {logical_operation} duplicates value {value}"
                )
            available_values[value] = abi
            lines.append(
                f"operation_output\t{logical_operation}\t{atom(role)}\t"
                f"{atom(value)}\t{atom(abi)}"
            )
    for role, value, abi in topology.program_outputs:
        if available_values.get(value) != abi:
            raise ValueError(
                f"program output {role} references unavailable value {value}"
            )
    if covered_layers != set(range(len(topology.layers))):
        raise ValueError("runtime operations do not cover every logical layer")
    for layer in topology.layers:
        layer_operations = tuple(
            operation for operation in topology.operations
            if operation.logical_layer == layer.logical_layer
        )
        if not any(
            operation.capability == layer.block_capability
            and operation.abi == layer.block_abi
            for operation in layer_operations
        ):
            raise ValueError(
                f"layer {layer.logical_layer} lacks its block operation"
            )
        if layer.routed_component is None:
            continue
        routed = components[layer.routed_component]
        for capability, abi in (
            (routed.router_capability, routed.router_abi),
            (routed.execution_capability, routed.execution_abi),
        ):
            if not any(
                operation.capability == capability
                and operation.abi == abi
                and operation.routed_component == routed.name
                and operation.component_layer == layer.component_layer
                for operation in layer_operations
            ):
                raise ValueError(
                    f"layer {layer.logical_layer} lacks routed operation {capability}"
                )
    return (("\n".join(lines)) + "\n").encode("utf-8")


def _write_runtime_model_descriptor(
    partial: Path, adapted: AdaptedModel, expert_abi: int
) -> dict[str, object]:
    name = "runtime-model.tsv"
    payload = _runtime_model_descriptor_bytes(adapted, expert_abi)
    temporary = partial / (name + ".tmp")
    with temporary.open("wb") as handle:
        write_all(handle, payload)
        fsync_file(handle)
    os.replace(temporary, partial / name)
    fsync_directory(partial)
    return {
        "format": "expert-runtime-model-v1",
        "path": name,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


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
    quant_profile: str,
) -> list[dict[str, object]]:
    temporary = partial / "dense.qpack.tmp"
    entries: list[dict[str, object]] = []
    with temporary.open("w+b") as handle:
        for info in adapted.dense:
            dense_quant_abi = (
                FP4_QUANT_ABI_ID
                if quant_profile == FP4_QUANT_PROFILE
                and info.name in adapted.dense_fp4
                else QUANT_ABI_ID
            )
            result = write_dense_record(
                handle,
                checkpoint,
                info,
                "dense.qpack",
                alignment,
                info.name in adapted.dense_float32,
                dense_quant_abi,
            )
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
    model_program: dict[str, object],
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
    if len(adapted.runtime_topology.components) > 1:
        raise ValueError(
            "Expert Pack v1 indexes at most one routed expert component"
        )
    active_expert_bytes = 0
    if adapted.runtime_topology.components:
        routed_component = adapted.runtime_topology.components[0]
        for layer in range(routed_component.layer_count):
            layer_entries = [
                entry for entry in experts if entry["layer"] == layer
            ]
            if len(layer_entries) != routed_component.experts_per_layer:
                raise ValueError(
                    f"routed component layer {layer} has incomplete expert records"
                )
            active_expert_bytes += round(
                routed_component.route_width
                * sum(entry["stored_bytes"] for entry in layer_entries)
                / len(layer_entries)
            )
    elif experts:
        raise ValueError("dense-only model contains routed expert records")
    source_id, source_revision = _source_identity(options, checkpoint)
    tokenizer_config = checkpoint.root / "tokenizer_config.json"
    tokenizer_metadata = load_json(tokenizer_config) if tokenizer_config.is_file() else {}
    if not isinstance(tokenizer_metadata, dict):
        tokenizer_metadata = {}
    config = checkpoint.config
    pack_bytes = dense_bytes + expert_bytes
    expert_abi = _expert_quant_abi(options.quant_profile)
    fp4 = expert_abi == FP4_QUANT_ABI_ID
    dense_abis = sorted({int(entry["quant_abi"]) for entry in dense})
    dense_fp4 = FP4_QUANT_ABI_ID in dense_abis
    dense_int8 = QUANT_ABI_ID in dense_abis
    if dense_fp4 and dense_int8:
        dense_weights = "mixed-fp4-block32-and-int8-by-tensor-index"
    elif dense_fp4:
        dense_weights = "fp4-e2m1-ue8m0-block32-padded"
    elif dense_int8:
        dense_weights = "symmetric-int8-except-adapter-preserved-fp32"
    else:
        dense_weights = "float32"
    runtime_dense_names = {
        tensor for _, tensor in adapted.runtime_topology.tensor_bindings
    }
    runtime_dense_names.update(
        tensor
        for operation in adapted.runtime_topology.operations
        for _, tensor in operation.tensor_bindings
    )
    resident_dense_bytes = sum(
        int(entry["stored_bytes"])
        for entry in dense
        if entry["name"] in runtime_dense_names
    )
    has_routed = bool(adapted.runtime_topology.components)
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
        "model_program": model_program,
        "quantization": {
            "profile": options.quant_profile,
            "abi_id": expert_abi,
            "expert_weights": (
                "fp4-e2m1-block32" if fp4 else "symmetric-int8"
            ) if has_routed else "none",
            "dense_matrix_weights": dense_weights,
            "dense_tensor_quant_abis": dense_abis,
            "router_and_norms": "float32",
            "scale_dtype": "ue8m0" if fp4 else "float32",
            "group_size": FP4_QUANT_GROUP_SIZE if fp4 else QUANT_GROUP_SIZE,
            "rounding": "nearest-ties-to-even",
            "zero_points": False,
        },
        "kernel_abi": {
            "id": (
                ("expert-pack-sm86-fp4-block32-v1" if fp4
                 else "expert-pack-sm86-int8-row-v1")
                if has_routed else
                ("expert-pack-sm86-dense-fp4-block32-v1" if dense_fp4
                 else "expert-pack-sm86-dense-int8-row-v1")
            ),
            "quant_abi": expert_abi,
            "gate_up_fused": has_routed,
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
            "resident_dense_bytes": resident_dense_bytes,
            "minimum_vram_bytes": resident_dense_bytes,
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
    expert_abi = _expert_quant_abi(options.quant_profile)
    if options.quant_profile not in QUANT_PROFILES:
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
    if options.quant_profile not in adapted.supported_expert_quant_profiles:
        raise ValueError(
            f"adapter {adapted.family!r} does not support expert quant profile "
            f"{options.quant_profile!r}; supported: "
            f"{sorted(adapted.supported_expert_quant_profiles)}"
        )
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
        dense_entries = _write_dense_pack(
            partial, checkpoint, adapted, options.alignment,
            options.quant_profile,
        )
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
    if len(adapted.runtime_topology.components) > 1:
        raise ValueError(
            "Expert Pack v1 indexes at most one routed expert component"
        )
    routed_component = (
        adapted.runtime_topology.components[0]
        if adapted.runtime_topology.components else None
    )
    if routed_component is None and adapted.experts:
        raise ValueError("dense-only model contains routed expert sources")
    predicted_record_bytes = (
        expert_record_size(
            routed_component.hidden_size,
            routed_component.intermediate_size,
            options.alignment,
            expert_abi,
        )
        if routed_component is not None else 0
    )
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
                    routed_component.hidden_size,
                    routed_component.intermediate_size,
                    options.alignment,
                    expert_abi,
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
    model_program = _write_runtime_model_descriptor(
        partial, adapted, expert_abi
    )
    manifest = _build_manifest(
        options,
        checkpoint,
        adapted,
        source_files,
        dense_entries,
        expert_entries,
        metadata_files,
        model_program,
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


def refresh_runtime_model_program(
    container: Path,
    output: Path,
    source: Path,
    adapter: str,
) -> dict[str, object]:
    """Publish a cloned container with freshly compiled VM metadata only.

    Expert/dense payloads are reused with hard links when possible. The source
    container remains immutable and valid if publication is interrupted.
    """
    container = Path(container).resolve()
    output = Path(output).resolve()
    source = Path(source).resolve()
    if output.exists():
        raise ResumeError(f"output already exists: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ResumeError(f"partial refresh exists: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_json(container / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("source container manifest is invalid")
    if "model_program" in manifest:
        validation = validate_container(container)
    else:
        # Early Expert Pack v1 publications predate the mandatory VM program.
        # Authenticate their immutable manifest/marker here; the fully cloned
        # candidate below is then subjected to the complete current validator,
        # including every pack hash and every dense/expert record.
        marker = load_json(container / "COMPLETED")
        integrity = manifest.get("integrity")
        if not isinstance(marker, dict) or not isinstance(integrity, dict):
            raise ValueError("legacy source container completion metadata is invalid")
        copied = dict(manifest)
        copied_integrity = dict(integrity)
        copied_integrity["content_sha256"] = ""
        copied["integrity"] = copied_integrity
        content_hash = sha256_bytes(canonical_json_bytes(copied))
        if (
            integrity.get("content_sha256") != content_hash
            or marker.get("manifest_content_sha256") != content_hash
            or marker.get("manifest_file_sha256") !=
                sha256_file(container / "manifest.json")
        ):
            raise ValueError("legacy source container authentication failed")
        validation = {"manifest_content_sha256": content_hash}

    checkpoint = SafeTensorCheckpoint(source)
    adapted = adapt_checkpoint(checkpoint, adapter)
    if manifest.get("architecture") != adapted.architecture:
        raise ValueError("adapter architecture disagrees with source container")
    source_metadata = manifest.get("source")
    if (
        not isinstance(source_metadata, dict)
        or source_metadata.get("tensor_count") != adapted.source_tensor_count
    ):
        raise ValueError("adapter source tensor count disagrees with container")
    dense = manifest.get("tensors")
    experts = manifest.get("experts")
    if not isinstance(dense, list) or not isinstance(experts, list):
        raise ValueError("container tensor indexes are invalid")
    if [item.get("name") for item in dense] != [
        tensor.name for tensor in adapted.dense
    ]:
        raise ValueError("adapter dense tensor partition disagrees with container")
    expected_experts = [
        {
            "layer": item.layer,
            "expert": item.expert,
            "source_tensors": {
                "gate": item.gate.name,
                "up": item.up.name,
                "down": item.down.name,
            },
        }
        for item in adapted.experts
    ]
    actual_experts = [
        {
            "layer": item.get("layer"),
            "expert": item.get("expert"),
            "source_tensors": item.get("source_tensors"),
        }
        for item in experts
        if isinstance(item, dict)
    ]
    if actual_experts != expected_experts:
        raise ValueError("adapter expert tensor partition disagrees with container")
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict) or not isinstance(
        quantization.get("abi_id"), int
    ):
        raise ValueError("container quantization ABI is invalid")

    shutil.copytree(container, partial, copy_function=_link_or_copy)
    shutil.rmtree(partial / "tokenizer")
    metadata_files = _copy_model_metadata(checkpoint, partial)
    model_program = _write_runtime_model_descriptor(
        partial, adapted, quantization["abi_id"]
    )
    refreshed_manifest = load_json(partial / "manifest.json")
    refreshed_manifest["model_program"] = model_program
    tokenizer_metadata = load_json(source / "tokenizer_config.json")
    refreshed_manifest["tokenizer"] = {
        "files": metadata_files,
        "chat_template": (
            tokenizer_metadata.get("chat_template")
            if isinstance(tokenizer_metadata, dict)
            else None
        ),
    }
    refreshed_manifest["integrity"]["content_sha256"] = ""
    refreshed_manifest["integrity"]["content_sha256"] = sha256_bytes(
        canonical_json_bytes(refreshed_manifest)
    )
    atomic_json(partial / "manifest.json", refreshed_manifest)

    report = load_json(partial / "conversion-report.json")
    if not isinstance(report, dict):
        raise ValueError("container conversion report is invalid")
    report["output"] = str(output)
    report["adapter"] = adapter
    report["manifest_content_sha256"] = refreshed_manifest["integrity"][
        "content_sha256"
    ]
    atomic_json(partial / "conversion-report.json", report)
    marker = {
        "format_version": FORMAT_VERSION,
        "manifest_content_sha256": refreshed_manifest["integrity"][
            "content_sha256"
        ],
        "manifest_file_sha256": sha256_file(partial / "manifest.json"),
    }
    atomic_json(partial / "COMPLETED", marker)
    fsync_directory(partial)
    refreshed_validation = validate_container(partial)
    os.replace(partial, output)
    fsync_directory(output.parent)
    return {
        "output": str(output),
        "source_container": str(container),
        "source_manifest_content_sha256": validation[
            "manifest_content_sha256"
        ],
        "manifest_content_sha256": refreshed_validation[
            "manifest_content_sha256"
        ],
        "validation": refreshed_validation,
    }
