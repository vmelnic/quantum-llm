"""Atomic, resumable SafeTensors to Expert Pack conversion."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adapters import AdaptedModel, adapt_checkpoint
from .activation_calibration import DenseActivationCalibrationSet
from .constants import (
    FORMAT_NAME,
    FORMAT_VERSION,
    FP4_ACTIVATION_CODE_QUANT_PROFILE,
    FP4_QUANT_ABI_ID,
    FP4_ACTIVATION_QUANT_PROFILE,
    FP4_MSE_QUANT_PROFILE,
    FP4_RELU2_EXPERT_ABI_ID,
    FP4_QUANT_GROUP_SIZE,
    FP4_QUANT_PROFILE,
    FP4_QUANT_PROFILES,
    NVFP4_QUANT_ABI_ID,
    NVFP4_QUANT_GROUP_SIZE,
    NVFP4_QUANT_PROFILE,
    MXFP6_QUANT_ABI_ID,
    MXFP6_QUANT_PROFILE,
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
from .errors import ResumeError, ValidationError
from .safetensors import SafeTensorCheckpoint
from .util import (
    atomic_json,
    canonical_json_bytes,
    fsync_directory,
    fsync_file,
    load_json,
    sha256_bytes,
    sha256_file,
    publish_directory,
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
    sampling_profiles: Path | None = None
    activation_calibration: Path | None = None
    dense_encoding_policy: Path | None = None
    dense_activation_input: str = "q8"
    config_file: str = "config.json"
    index_file: str = "model.safetensors.index.json"
    resume: bool = False
    reclaim_source_shards: bool = False


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


_SAMPLING_PROFILE_NAMES = frozenset(("thinking", "non_thinking"))
_SAMPLING_PROFILE_FIELDS = frozenset((
    "temperature", "top_p", "top_k", "min_p", "presence_penalty",
    "frequency_penalty", "repetition_penalty",
))


def validate_sampling_profiles(value: object) -> dict[str, object]:
    """Validate artifact-declared, harness-overridable sampling defaults."""
    if not isinstance(value, dict):
        raise ValueError("sampling policy must be an object")
    if (not isinstance(value.get("schema"), str) or
            not value["schema"].strip()):
        raise ValueError("sampling policy schema is invalid")
    allowed_fields = {
        "schema", "profiles", "maximum_thinking_tokens",
    }
    if not {"schema", "profiles"} <= set(value) or set(value) - allowed_fields:
        raise ValueError("sampling profiles have unknown or missing fields")
    maximum = value.get("maximum_thinking_tokens")
    if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, int) or
            not 1 <= maximum <= 0xffff_ffff):
        raise ValueError(
            "sampling profiles maximum_thinking_tokens is invalid"
        )
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != _SAMPLING_PROFILE_NAMES:
        raise ValueError("sampling profiles must declare thinking and non_thinking")
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile) != _SAMPLING_PROFILE_FIELDS:
            raise ValueError(f"sampling profile {name!r} has unknown or missing fields")
        numeric = (
            "temperature", "top_p", "min_p", "presence_penalty",
            "frequency_penalty", "repetition_penalty",
        )
        if any(isinstance(profile[field], bool) or
               not isinstance(profile[field], (int, float)) for field in numeric):
            raise ValueError(f"sampling profile {name!r} has a non-numeric value")
        if isinstance(profile["top_k"], bool) or not isinstance(profile["top_k"], int):
            raise ValueError(f"sampling profile {name!r} top_k is not an integer")
        if not (0.0 <= float(profile["temperature"]) <= 2.0 and
                0.0 < float(profile["top_p"]) <= 1.0 and
                int(profile["top_k"]) >= 0 and
                0.0 <= float(profile["min_p"]) <= 1.0 and
                -2.0 <= float(profile["presence_penalty"]) <= 2.0 and
                float(profile["frequency_penalty"]) == 0.0 and
                float(profile["repetition_penalty"]) == 1.0):
            raise ValueError(f"sampling profile {name!r} is outside runtime limits")
    return value


def _load_sampling_profiles(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"sampling profiles file is missing: {path}")
    return validate_sampling_profiles(load_json(path))


_DENSE_ENCODING_POLICY_SCHEMA = "expert-pack-dense-encoding-policy-v1"
_DENSE_ACTIVATION_INPUTS = frozenset(("q8", "bf16"))
_BF16_ACTIVATION_KERNEL = ("dense.activation-input.bfloat16.v1", 1)


def _load_dense_encoding_policy(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"dense encoding policy file is missing: {path}")
    value = load_json(path)
    if not isinstance(value, dict) or set(value) != {"schema", "rules"}:
        raise ValueError("dense encoding policy has unknown or missing fields")
    if value.get("schema") != _DENSE_ENCODING_POLICY_SCHEMA:
        raise ValueError("dense encoding policy schema is unsupported")
    rules = value.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("dense encoding policy rules must be a non-empty list")
    normalized: list[dict[str, str]] = []
    identities: set[tuple[str, str, str]] = set()
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"encoding", "capability", "tensor_role"}
        ):
            raise ValueError("dense encoding rule has unknown or missing fields")
        if rule.get("encoding") != MXFP6_QUANT_PROFILE:
            raise ValueError(
                f"unsupported dense encoding {rule.get('encoding')!r}"
            )
        if any(
            not isinstance(rule.get(field), str) or not rule[field].strip()
            for field in ("capability", "tensor_role")
        ):
            raise ValueError("dense encoding capability and role must be non-empty")
        normalized_rule = {
            "encoding": str(rule["encoding"]),
            "capability": str(rule["capability"]),
            "tensor_role": str(rule["tensor_role"]),
        }
        identity = tuple(normalized_rule[field] for field in (
            "encoding", "capability", "tensor_role"
        ))
        if identity in identities:
            raise ValueError("dense encoding policy contains a duplicate rule")
        identities.add(identity)
        normalized.append(normalized_rule)
    return {"schema": _DENSE_ENCODING_POLICY_SCHEMA, "rules": normalized}


def _resolve_dense_encoding_policy(
    adapted: AdaptedModel,
    policy: dict[str, object] | None,
) -> tuple[AdaptedModel, dict[str, object] | None]:
    if policy is None:
        return adapted, None
    selected: set[str] = set()
    resolved_rules: list[dict[str, object]] = []
    for rule in policy["rules"]:
        matches = {
            tensor
            for operation in adapted.runtime_topology.operations
            if operation.capability == rule["capability"]
            for role, tensor in operation.tensor_bindings
            if role == rule["tensor_role"]
        }
        if not matches:
            raise ValueError(
                "dense encoding rule matched no operation tensor: "
                f"{rule['capability']} / {rule['tensor_role']}"
            )
        selected.update(matches)
        resolved_rules.append({**rule, "matched_tensors": len(matches)})
    dense_by_name = {info.name: info for info in adapted.dense}
    ineligible = sorted(selected - adapted.dense_fp4)
    if ineligible:
        raise ValueError(
            "dense encoding policy selected tensors that are not semantic "
            "FP4 matrices: " + ", ".join(ineligible[:8])
        )
    invalid_shape = sorted(
        name for name in selected
        if name not in dense_by_name or len(dense_by_name[name].shape) != 2
    )
    if invalid_shape:
        raise ValueError(
            "dense encoding policy requires rank-2 dense tensors: "
            + ", ".join(invalid_shape[:8])
        )
    selected_names = sorted(selected)
    resolved = {
        "schema": policy["schema"],
        "content_sha256": sha256_bytes(canonical_json_bytes(policy)),
        "rules": resolved_rules,
        "resolved_tensor_count": len(selected_names),
        "resolved_tensors": selected_names,
    }
    return replace(
        adapted,
        dense_mxfp6=frozenset(set(adapted.dense_mxfp6) | selected),
    ), resolved


def _configure_dense_activation_input(
    adapted: AdaptedModel, encoding: str,
) -> AdaptedModel:
    if encoding not in _DENSE_ACTIVATION_INPUTS:
        raise ValueError(f"unsupported dense activation input {encoding!r}")
    if encoding == "q8":
        return adapted
    if adapted.dense_mxfp6:
        raise ValueError(
            "BF16 dense activation input currently requires FP4 dense matrices"
        )
    topology = adapted.runtime_topology
    attributes = dict(topology.attributes)
    attributes["dense_activation_input_bf16"] = 1
    required = set(topology.required_kernels)
    required.add(_BF16_ACTIVATION_KERNEL)
    return replace(
        adapted,
        runtime_topology=replace(
            topology,
            attributes=tuple(sorted(attributes.items())),
            required_kernels=tuple(sorted(required)),
        ),
    )


def _expert_quant_abi(quant_profile: str) -> int:
    if quant_profile == QUANT_PROFILE:
        return QUANT_ABI_ID
    if quant_profile in FP4_QUANT_PROFILES:
        return FP4_QUANT_ABI_ID
    if quant_profile == NVFP4_QUANT_PROFILE:
        return NVFP4_QUANT_ABI_ID
    raise ValueError(f"unsupported quant profile {quant_profile!r}")


def _expert_record_abi(adapted: AdaptedModel, quant_abi: int) -> int:
    if not adapted.experts:
        return quant_abi
    gated = {expert.gate is not None for expert in adapted.experts}
    if len(gated) != 1:
        raise ValueError("one routed component cannot mix expert mathematics")
    if gated == {False}:
        if quant_abi != FP4_QUANT_ABI_ID:
            raise ValueError("ReLU2 routed experts currently require FP4")
        return FP4_RELU2_EXPERT_ABI_ID
    if quant_abi == NVFP4_QUANT_ABI_ID:
        if any(
            expert.nvfp4_gate is None or expert.nvfp4_up is None or
            expert.nvfp4_down is None
            for expert in adapted.experts
        ):
            raise ValueError("native NVFP4 profile requires complete expert sidecars")
        return NVFP4_QUANT_ABI_ID
    return quant_abi


def _runtime_model_descriptor_bytes(adapted: AdaptedModel, expert_abi: int) -> bytes:
    topology = adapted.runtime_topology
    fp4_experts = expert_abi in (
        FP4_QUANT_ABI_ID, FP4_RELU2_EXPERT_ABI_ID
    )
    native_nvfp4 = expert_abi == NVFP4_QUANT_ABI_ID
    encoding_abi = 3 if native_nvfp4 else (2 if fp4_experts else 1)
    encoding = (
        "nvfp4.e2m1.e4m3fn.block16.w4a4"
        if native_nvfp4 else
        ("fp4.e2m1.ue8m0.block32" if fp4_experts
         else "int8.symmetric.per-row")
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
        len(required) != len(topology.required_kernels)
        or len({key for key, _ in topology.attributes}) !=
        len(topology.attributes)
    ):
        raise ValueError("runtime model has duplicate declarations")
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
        if (
            len({key for key, _ in component.attributes}) !=
            len(component.attributes)
            or len({key for key, _ in component.router_parameters}) !=
            len(component.router_parameters)
        ):
            raise ValueError(
                f"component {component.name} has duplicate parameters"
            )
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
        if len({key for key, _ in layer.parameters}) != len(layer.parameters):
            raise ValueError(
                f"layer {layer.logical_layer} has duplicate parameters"
            )
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
        if (
            len({key for key, _ in operation.parameters}) !=
            len(operation.parameters)
        ):
            raise ValueError(
                f"operation {logical_operation} has duplicate parameters"
            )
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


def _option_contract(
    options: CompileOptions,
    source_files: list[dict[str, object]],
    activation_calibration: DenseActivationCalibrationSet | None,
    dense_encoding_policy: dict[str, object] | None,
) -> dict[str, object]:
    sampling_profiles = _load_sampling_profiles(options.sampling_profiles)
    return {
        "adapter": options.adapter,
        "quant_profile": options.quant_profile,
        "alignment": options.alignment,
        "max_expert_pack_bytes": options.max_expert_pack_bytes,
        "source_id": options.source_id,
        "source_revision": options.source_revision,
        "sampling_profiles": sampling_profiles,
        "activation_calibration": (
            activation_calibration.source_summary()
            if activation_calibration is not None else None
        ),
        "dense_encoding_policy": dense_encoding_policy,
        "dense_activation_input": options.dense_activation_input,
        "config_file": options.config_file,
        "index_file": options.index_file,
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
    activation_calibration: DenseActivationCalibrationSet | None,
) -> list[dict[str, object]]:
    temporary = partial / "dense.qpack.tmp"
    entries: list[dict[str, object]] = []
    native_nvfp4 = {
        matrix.weight.name: matrix for matrix in adapted.dense_nvfp4
    }
    with temporary.open("w+b") as handle:
        for info in adapted.dense:
            dense_quant_abi = (
                NVFP4_QUANT_ABI_ID
                if info.name in native_nvfp4
                else
                MXFP6_QUANT_ABI_ID
                if quant_profile in FP4_QUANT_PROFILES
                and info.name in adapted.dense_mxfp6
                else
                FP4_QUANT_ABI_ID
                if quant_profile in FP4_QUANT_PROFILES
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
                info.name in adapted.dense_int64,
                info.name in adapted.dense_bfloat16,
                native_nvfp4.get(info.name),
                quant_profile in (
                    FP4_MSE_QUANT_PROFILE,
                    FP4_ACTIVATION_QUANT_PROFILE,
                    FP4_ACTIVATION_CODE_QUANT_PROFILE,
                ),
                activation_calibration,
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
    if checkpoint.config_file not in names:
        names.append(checkpoint.config_file)
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
    activation_calibration: dict[str, object] | None,
    dense_encoding_policy: dict[str, object] | None,
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
    expert_abi = _expert_record_abi(
        adapted, _expert_quant_abi(options.quant_profile)
    )
    fp4 = expert_abi in (
        FP4_QUANT_ABI_ID, FP4_RELU2_EXPERT_ABI_ID,
        NVFP4_QUANT_ABI_ID,
    )
    native_nvfp4 = expert_abi == NVFP4_QUANT_ABI_ID
    relu2_experts = expert_abi == FP4_RELU2_EXPERT_ABI_ID
    dense_abis = sorted({int(entry["quant_abi"]) for entry in dense})
    dense_fp4 = FP4_QUANT_ABI_ID in dense_abis
    dense_mxfp6 = MXFP6_QUANT_ABI_ID in dense_abis
    dense_nvfp4 = NVFP4_QUANT_ABI_ID in dense_abis
    dense_int8 = QUANT_ABI_ID in dense_abis
    if dense_mxfp6 and (dense_fp4 or dense_int8 or dense_nvfp4):
        dense_weights = "mixed-mxfp6-fp4-and-other-quantized-tensors"
    elif dense_mxfp6:
        dense_weights = "mxfp6-e3m2-ue8m0-block32-padded"
    elif dense_nvfp4 and (dense_fp4 or dense_int8):
        dense_weights = "mixed-native-nvfp4-and-other-quantized-tensors"
    elif dense_nvfp4:
        dense_weights = "native-nvfp4-e2m1-e4m3fn-block16-w4a4"
    elif dense_fp4 and dense_int8:
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
        and entry["name"] not in adapted.host_mapped_dense
    )
    host_mapped_dense_bytes = sum(
        int(entry["stored_bytes"])
        for entry in dense
        if entry["name"] in adapted.host_mapped_dense
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
        "auxiliary_tensors": sorted(adapted.auxiliary_dense),
        "model_program": model_program,
        "quantization": {
            "profile": options.quant_profile,
            "abi_id": expert_abi,
            "expert_weights": (
                "nvfp4-e2m1-e4m3fn-block16-w4a4" if native_nvfp4 else
                "fp4-e2m1-block32" if fp4 else "symmetric-int8"
            ) if has_routed else "none",
            "dense_matrix_weights": dense_weights,
            "dense_tensor_quant_abis": dense_abis,
            "router_and_norms": "float32",
            "scale_dtype": (
                "float8-e4m3fn+float32-global-divisors"
                if native_nvfp4 else "ue8m0" if fp4 else "float32"
            ),
            "group_size": (
                NVFP4_QUANT_GROUP_SIZE if native_nvfp4 else
                FP4_QUANT_GROUP_SIZE if fp4 else QUANT_GROUP_SIZE
            ),
            "rounding": "nearest-ties-to-even",
            "zero_points": False,
            "dense_activation_input": options.dense_activation_input,
            **({"activation_calibration": activation_calibration}
               if activation_calibration is not None else {}),
            **({"dense_encoding_policy": dense_encoding_policy}
               if dense_encoding_policy is not None else {}),
        },
        "kernel_abi": {
            "id": (
                ("expert-pack-sm86-nvfp4-block16-w4a4-v1"
                 if native_nvfp4 else
                 "expert-pack-sm86-fp4-block32-v1" if fp4
                 else "expert-pack-sm86-int8-row-v1")
                if has_routed else
                ("expert-pack-sm86-dense-mixed-mxfp6-fp4-v1"
                 if dense_mxfp6 else
                 "expert-pack-sm86-dense-nvfp4-block16-w4a4-v1"
                 if dense_nvfp4 else
                 "expert-pack-sm86-dense-fp4-block32-v1" if dense_fp4
                 else "expert-pack-sm86-dense-int8-row-v1")
            ),
            "quant_abi": expert_abi,
            "gate_up_fused": has_routed and not relu2_experts,
            "gate_up_order": ["gate", "up"] if not relu2_experts else ["up"],
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
            "resident_dense_bytes": resident_dense_bytes,
            "host_mapped_dense_bytes": host_mapped_dense_bytes,
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
            **({"reasoning_effort_map": dict(
                adapted.template_reasoning_effort_map
            )} if adapted.template_reasoning_effort_map else {}),
            **({"sampling": _load_sampling_profiles(options.sampling_profiles)}
               if options.sampling_profiles is not None else {}),
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
    quant_abi = _expert_quant_abi(options.quant_profile)
    dense_encoding_policy = _load_dense_encoding_policy(
        options.dense_encoding_policy
    )
    if options.quant_profile not in QUANT_PROFILES:
        raise ValueError(f"unsupported quant profile {options.quant_profile!r}")
    if (
        dense_encoding_policy is not None
        and options.quant_profile not in FP4_QUANT_PROFILES
    ):
        raise ValueError("dense encoding policy requires an FP4 quant profile")
    if options.quant_profile in (
        FP4_ACTIVATION_QUANT_PROFILE,
        FP4_ACTIVATION_CODE_QUANT_PROFILE,
    ):
        if options.activation_calibration is None:
            raise ValueError(
                "activation-aware FP4 profile requires activation calibration"
            )
        activation_calibration = DenseActivationCalibrationSet(
            options.activation_calibration
        )
    else:
        if options.activation_calibration is not None:
            raise ValueError(
                "activation calibration requires the activation-aware FP4 profile"
            )
        activation_calibration = None
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

    checkpoint = SafeTensorCheckpoint(
        source, config_file=options.config_file, index_file=options.index_file
    )
    adapted = adapt_checkpoint(checkpoint, options.adapter)
    adapted, resolved_dense_encoding_policy = _resolve_dense_encoding_policy(
        adapted, dense_encoding_policy
    )
    adapted = _configure_dense_activation_input(
        adapted, options.dense_activation_input
    )
    expert_abi = _expert_record_abi(adapted, quant_abi)
    if options.quant_profile not in adapted.supported_expert_quant_profiles:
        raise ValueError(
            f"adapter {adapted.family!r} does not support expert quant profile "
            f"{options.quant_profile!r}; supported: "
            f"{sorted(adapted.supported_expert_quant_profiles)}"
        )
    if activation_calibration is not None:
        activation_calibration.configure_exclusions({
            name for name in adapted.dense_mxfp6
            if activation_calibration.contains(name)
        })
    if options.quant_profile == FP4_ACTIVATION_CODE_QUANT_PROFILE:
        regular_target_tensors = {
            tensor
            for operation in adapted.runtime_topology.operations
            if not operation.capability.startswith("embedding.lookup.")
            and not operation.capability.startswith("vision.")
            for _, tensor in operation.tensor_bindings
        }
        payload_targets = {
            info.name
            for info in adapted.dense
            if len(info.shape) == 2
            and info.name in adapted.dense_fp4
            and info.name not in adapted.dense_mxfp6
            and info.name in regular_target_tensors
            and activation_calibration.contains(info.name)
        }
        activation_calibration.configure_payload_targets(payload_targets)
    source_files = _source_inventory(checkpoint)
    contract = _option_contract(
        options, source_files, activation_calibration,
        resolved_dense_encoding_policy,
    )
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
            options.quant_profile, activation_calibration,
        )
        state["dense"] = dense_entries
        if activation_calibration is not None:
            state["activation_calibration"] = (
                activation_calibration.snapshot()
            )
        atomic_json(state_path, state)
        if _record_hook:
            _record_hook("dense-pack", len(dense_entries))
    elif activation_calibration is not None:
        activation_calibration.restore(
            state.get("activation_calibration")
        )

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
            adapted.experts[0].gate is not None,
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
                    options.quant_profile in (
                        FP4_MSE_QUANT_PROFILE,
                        FP4_ACTIVATION_QUANT_PROFILE,
                        FP4_ACTIVATION_CODE_QUANT_PROFILE,
                    ),
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
    activation_calibration_report = (
        activation_calibration.finalize()
        if activation_calibration is not None else None
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
        activation_calibration_report,
        resolved_dense_encoding_policy,
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
        "dense_activation_input": options.dense_activation_input,
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
        **({"activation_calibration": activation_calibration_report}
           if activation_calibration_report is not None else {}),
        **({"dense_encoding_policy": resolved_dense_encoding_policy}
           if resolved_dense_encoding_policy is not None else {}),
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
    config_file: str = "config.json",
    index_file: str = "model.safetensors.index.json",
    dense_activation_input: str | None = None,
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

    checkpoint = SafeTensorCheckpoint(
        source, config_file=config_file, index_file=index_file
    )
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
                "up": item.up.name,
                "down": item.down.name,
                **({"gate": item.gate.name} if item.gate is not None else {}),
            },
            "source_regions": {
                role: {
                    "tensor": tensor.physical_name,
                    "byte_offset": tensor.source_byte_offset,
                    "bytes": tensor.nbytes,
                    "shape": list(tensor.shape),
                    "tensor_bytes": checkpoint.tensors[tensor.physical_name].nbytes,
                    "tensor_shape": list(
                        checkpoint.tensors[tensor.physical_name].shape
                    ),
                }
                for role, tensor in (
                    *(((("gate", item.gate),)) if item.gate is not None else ()),
                    ("up", item.up),
                    ("down", item.down),
                )
            },
        }
        for item in adapted.experts
    ]
    actual_experts = [
        {
            "layer": item.get("layer"),
            "expert": item.get("expert"),
            "source_tensors": item.get("source_tensors"),
            "source_regions": item.get("source_regions"),
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
    selected_activation_input = (
        quantization.get("dense_activation_input", "q8")
        if dense_activation_input is None else dense_activation_input
    )
    if selected_activation_input not in _DENSE_ACTIVATION_INPUTS:
        raise ValueError(
            "unsupported dense activation input "
            f"{selected_activation_input!r}"
        )
    if selected_activation_input == "bf16" and any(
        isinstance(item, dict) and
        item.get("quant_abi") == MXFP6_QUANT_ABI_ID
        for item in dense
    ):
        raise ValueError(
            "BF16 dense activation input currently requires FP4 dense matrices"
        )
    adapted = _configure_dense_activation_input(
        adapted, selected_activation_input
    )

    shutil.copytree(container, partial, copy_function=_link_or_copy)
    shutil.rmtree(partial / "tokenizer")
    metadata_files = _copy_model_metadata(checkpoint, partial)
    model_program = _write_runtime_model_descriptor(
        partial, adapted, quantization["abi_id"]
    )
    refreshed_manifest = load_json(partial / "manifest.json")
    refreshed_manifest["model_program"] = model_program
    refreshed_manifest["quantization"]["dense_activation_input"] = (
        selected_activation_input
    )
    tokenizer_metadata = load_json(source / "tokenizer_config.json")
    previous_tokenizer = refreshed_manifest.get("tokenizer")
    previous_sampling = (
        previous_tokenizer.get("sampling")
        if isinstance(previous_tokenizer, dict) else None
    )
    refreshed_manifest["tokenizer"] = {
        "files": metadata_files,
        "chat_template": (
            tokenizer_metadata.get("chat_template")
            if isinstance(tokenizer_metadata, dict)
            else None
        ),
        **({"reasoning_effort_map": dict(
            adapted.template_reasoning_effort_map
        )} if adapted.template_reasoning_effort_map else {}),
        **({"sampling": previous_sampling}
           if previous_sampling is not None else {}),
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
    report["dense_activation_input"] = selected_activation_input
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


_DENSE_MTP_EXACT_V1 = (
    "decode.mtp.dense-full-attention.fp4-block32.exact.v1"
)
_DENSE_MTP_EXACT_V2 = (
    "decode.mtp.dense-full-attention.fp4-block32.exact.v2"
)


def _upgrade_dense_mtp_program_bytes(
    payload: bytes,
    *,
    draft_depth: int,
    draft_vocabulary_size: int,
) -> bytes:
    """Upgrade only the generic dense-MTP exact-decode record from ABI 1 to 2."""
    if draft_depth not in (3, 4):
        raise ValueError("dense MTP draft depth must be 3 or 4")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError("runtime model program is not UTF-8") from error
    if not lines or lines[0] != "expert-runtime-model-v1":
        raise ValueError("runtime model program format is unsupported")

    records = [line.split("\t") for line in lines]
    model_records = [fields for fields in records if fields[0] == "model"]
    if len(model_records) != 1 or len(model_records[0]) != 6:
        raise ValueError("runtime model program has an invalid model record")
    try:
        schema = int(model_records[0][1])
        target_vocabulary_size = int(model_records[0][3])
    except ValueError as error:
        raise ValueError("runtime model program has a non-integer model record") from error
    if schema != 3 or target_vocabulary_size <= 0:
        raise ValueError("dense MTP migration requires model program schema 3")
    if not 0 < draft_vocabulary_size <= target_vocabulary_size:
        raise ValueError(
            "draft vocabulary size must be positive and no larger than target vocabulary"
        )

    mtp_attributes = [
        fields for fields in records
        if len(fields) >= 2 and fields[:2] == ["attribute", "mtp_layers"]
    ]
    if mtp_attributes != [["attribute", "mtp_layers", "1"]]:
        raise ValueError("dense MTP v1 migration requires one source MTP layer")

    old_kernel = ["kernel", _DENSE_MTP_EXACT_V1, "1"]
    new_kernel = ["kernel", _DENSE_MTP_EXACT_V2, "2"]
    old_decode = ["exact_decode", _DENSE_MTP_EXACT_V1, "1", "2"]
    new_decode = [
        "exact_decode", _DENSE_MTP_EXACT_V2, "2", str(draft_depth + 1)
    ]
    if any(_DENSE_MTP_EXACT_V2 in fields for fields in records):
        raise ValueError("container already declares the dense MTP v2 contract")
    if records.count(old_kernel) != 1 or records.count(old_decode) != 1:
        raise ValueError("container does not declare the exact dense MTP v1 contract")

    parameter_records = [
        fields for fields in records if fields[0] == "exact_decode_parameter"
    ]
    if (any(len(fields) != 3 for fields in parameter_records) or
            {fields[1]: fields[2] for fields in parameter_records} != {
                "draft_layers": "1",
                "embedding_first": "1",
                "post_norm": "1",
            } or len(parameter_records) != 3):
        raise ValueError("dense MTP v1 parameters do not match the migratable contract")

    tensor_records = [
        fields for fields in records if fields[0] == "exact_decode_tensor"
    ]
    if (not tensor_records or any(len(fields) != 3 for fields in tensor_records) or
            not any(fields[1].startswith("layer.0.") for fields in tensor_records) or
            any(fields[1].startswith("layer.") and
                not fields[1].startswith("layer.0.")
                for fields in tensor_records)):
        raise ValueError("dense MTP v1 tensor roles are not a single reusable layer")

    upgraded: list[list[str]] = []
    for fields in records:
        if fields == old_kernel:
            upgraded.append(new_kernel)
        elif fields == old_decode:
            upgraded.append(new_decode)
        elif fields == ["exact_decode_parameter", "draft_layers", "1"]:
            upgraded.extend([
                ["exact_decode_parameter", "source_mtp_layers", "1"],
                ["exact_decode_parameter", "draft_depth", str(draft_depth)],
                [
                    "exact_decode_parameter", "draft_vocabulary_size",
                    str(draft_vocabulary_size),
                ],
                ["exact_decode_parameter", "mtp_kv_encoding", "1"],
            ])
        else:
            upgraded.append(fields)
    return ("\n".join("\t".join(fields) for fields in upgraded) + "\n").encode(
        "utf-8"
    )


def upgrade_dense_mtp_program(
    container: Path,
    output: Path,
    *,
    draft_depth: int = 4,
    draft_vocabulary_size: int = 65_536,
) -> dict[str, object]:
    """Publish an authenticated program-only dense-MTP ABI v1-to-v2 migration."""
    container = Path(container).resolve()
    output = Path(output).resolve()
    try:
        output.relative_to(container)
    except ValueError:
        pass
    else:
        raise ValueError("output must not be inside the immutable source container")
    if output.exists():
        raise ResumeError(f"output already exists: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ResumeError(f"partial migration already exists: {partial}")

    validation = validate_container(container)
    manifest = load_json(container / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("source container manifest is invalid")
    model_program = manifest.get("model_program")
    if (not isinstance(model_program, dict) or
            not isinstance(model_program.get("path"), str)):
        raise ValueError("source container model program metadata is invalid")
    relative_program = Path(model_program["path"])
    if relative_program.is_absolute() or ".." in relative_program.parts:
        raise ValueError("source container model program path is unsafe")
    source_program = container / relative_program
    source_payload = source_program.read_bytes()
    upgraded_payload = _upgrade_dense_mtp_program_bytes(
        source_payload,
        draft_depth=draft_depth,
        draft_vocabulary_size=draft_vocabulary_size,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(container, partial, copy_function=_link_or_copy)
    migrated_program = partial / relative_program
    temporary_program = migrated_program.with_name(migrated_program.name + ".tmp")
    with temporary_program.open("wb") as handle:
        write_all(handle, upgraded_payload)
        fsync_file(handle)
    os.replace(temporary_program, migrated_program)
    fsync_directory(migrated_program.parent)

    migrated_manifest = load_json(partial / "manifest.json")
    migrated_manifest["model_program"] = {
        "format": "expert-runtime-model-v1",
        "path": relative_program.as_posix(),
        "bytes": len(upgraded_payload),
        "sha256": sha256_bytes(upgraded_payload),
    }
    migrated_manifest["integrity"]["content_sha256"] = ""
    migrated_manifest["integrity"]["content_sha256"] = sha256_bytes(
        canonical_json_bytes(migrated_manifest)
    )
    atomic_json(partial / "manifest.json", migrated_manifest)

    report = load_json(partial / "conversion-report.json")
    if not isinstance(report, dict):
        raise ValueError("container conversion report is invalid")
    report["output"] = str(output)
    report["manifest_content_sha256"] = migrated_manifest["integrity"][
        "content_sha256"
    ]
    report["model_program_migration"] = {
        "source_capability": _DENSE_MTP_EXACT_V1,
        "target_capability": _DENSE_MTP_EXACT_V2,
        "source_program_sha256": sha256_bytes(source_payload),
        "target_program_sha256": sha256_bytes(upgraded_payload),
        "source_mtp_layers": 1,
        "draft_depth": draft_depth,
        "draft_vocabulary_size": draft_vocabulary_size,
        "mtp_kv_encoding": 1,
    }
    atomic_json(partial / "conversion-report.json", report)
    atomic_json(partial / "COMPLETED", {
        "format_version": FORMAT_VERSION,
        "manifest_content_sha256": migrated_manifest["integrity"][
            "content_sha256"
        ],
        "manifest_file_sha256": sha256_file(partial / "manifest.json"),
    })
    fsync_directory(partial)
    migrated_validation = validate_container(partial)
    publish_directory(partial, output)
    return {
        "output": str(output),
        "source_container": str(container),
        "source_manifest_content_sha256": validation[
            "manifest_content_sha256"
        ],
        "manifest_content_sha256": migrated_validation[
            "manifest_content_sha256"
        ],
        "source_program_sha256": sha256_bytes(source_payload),
        "target_program_sha256": sha256_bytes(upgraded_payload),
        "draft_depth": draft_depth,
        "draft_vocabulary_size": draft_vocabulary_size,
        "validation": migrated_validation,
    }


def refresh_sampling_profiles(
    container: Path,
    output: Path,
    profiles_path: Path,
) -> dict[str, object]:
    """Clone a valid artifact and transactionally replace sampling metadata."""
    container = Path(container).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ResumeError(f"output already exists: {output}")
    validation = validate_container(container)
    sampling = _load_sampling_profiles(profiles_path)
    if sampling is None:
        raise ValueError("sampling profiles are required")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ResumeError(f"partial refresh already exists: {partial}")

    shutil.copytree(container, partial, copy_function=_link_or_copy)
    manifest = load_json(partial / "manifest.json")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("container tokenizer metadata is invalid")
    tokenizer["sampling"] = sampling
    manifest["integrity"]["content_sha256"] = ""
    manifest["integrity"]["content_sha256"] = sha256_bytes(
        canonical_json_bytes(manifest)
    )
    atomic_json(partial / "manifest.json", manifest)

    report = load_json(partial / "conversion-report.json")
    if not isinstance(report, dict):
        raise ValueError("container conversion report is invalid")
    report["output"] = str(output)
    report["manifest_content_sha256"] = manifest["integrity"][
        "content_sha256"
    ]
    atomic_json(partial / "conversion-report.json", report)
    atomic_json(partial / "COMPLETED", {
        "format_version": FORMAT_VERSION,
        "manifest_content_sha256": manifest["integrity"][
            "content_sha256"
        ],
        "manifest_file_sha256": sha256_file(partial / "manifest.json"),
    })
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
