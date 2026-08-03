#!/usr/bin/env python3
"""Validate a converted Colibri OLMoE container without loading model weights."""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(merged_weight|qs)$"
)
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def tensor_offsets(shard: Path) -> dict[str, dict[str, int]]:
    with shard.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Invalid SafeTensors header prefix: {shard}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 256 * 1024 * 1024:
            raise ValueError(f"Unreasonable SafeTensors header length in {shard}")
        header = json.loads(stream.read(header_length).decode("utf-8"))
    data_start = 8 + header_length
    result = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        result[name] = {
            "absolute_start": data_start + int(start),
            "absolute_end": data_start + int(end),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-model")
    parser.add_argument("--source-revision")
    parser.add_argument("--colibri-commit")
    args = parser.parse_args()

    config_path = args.model / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Missing config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    experts = int(config.get("num_experts") or config.get("num_local_experts"))
    top_k = int(config["num_experts_per_tok"])

    shards = sorted(args.model.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No SafeTensors shards in {args.model}")

    tensors: dict[str, dict] = {}
    duplicate_names: list[str] = []
    tensor_bytes = 0
    for shard in shards:
        offsets = tensor_offsets(shard)
        with safe_open(str(shard), framework="numpy", device="cpu") as handle:
            for name in handle.keys():
                if name in tensors:
                    duplicate_names.append(name)
                    continue
                tensor_slice = handle.get_slice(name)
                shape = list(tensor_slice.get_shape())
                dtype = str(tensor_slice.get_dtype())
                if dtype not in DTYPE_BYTES:
                    raise SystemExit(f"Unsupported dtype {dtype} for {name}")
                size = product(shape) * DTYPE_BYTES[dtype]
                tensors[name] = {
                    "shard": shard.name,
                    "shape": shape,
                    "dtype": dtype,
                    "bytes": size,
                    **offsets[name],
                }
                tensor_bytes += size

    expert_parts: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for name, info in tensors.items():
        match = EXPERT_RE.fullmatch(name)
        if match:
            expert_parts[(int(match.group(1)), int(match.group(2)))][match.group(3)] = info

    failures: list[str] = []
    if duplicate_names:
        failures.append(f"duplicate tensor names: {len(duplicate_names)}")

    expected_weight_numel = 3 * hidden * intermediate
    expected_scale_numel = 2 * intermediate + hidden
    expert_payloads: list[int] = []
    missing_experts: list[str] = []
    malformed_experts: list[str] = []
    direct_unsafe_experts: list[str] = []
    direct_weight_read_bytes: list[int] = []

    for layer in range(layers):
        for expert in range(experts):
            parts = expert_parts.get((layer, expert), {})
            if set(parts) != {"merged_weight", "qs"}:
                missing_experts.append(f"{layer}:{expert}")
                continue
            weight = parts["merged_weight"]
            scales = parts["qs"]
            weight_numel = product(weight["shape"])
            scale_numel = product(scales["shape"])
            if (
                weight["dtype"] != "I8"
                or weight_numel != expected_weight_numel
                or scales["dtype"] != "F32"
                or scale_numel != expected_scale_numel
            ):
                malformed_experts.append(f"{layer}:{expert}")
                continue
            aligned_start = weight["absolute_start"] & ~4095
            aligned_end = (weight["absolute_end"] + 4095) & ~4095
            shard_size = (args.model / weight["shard"]).stat().st_size
            if aligned_end > shard_size:
                direct_unsafe_experts.append(f"{layer}:{expert}")
            expert_payloads.append(weight["bytes"] + scales["bytes"])
            if aligned_end <= shard_size:
                direct_weight_read_bytes.append(aligned_end - aligned_start)

    if missing_experts:
        failures.append(f"missing/incomplete experts: {len(missing_experts)}")
    if malformed_experts:
        failures.append(f"malformed experts: {len(malformed_experts)}")
    unexpected_experts = sorted(
        f"{layer}:{expert}"
        for layer, expert in expert_parts
        if layer >= layers or expert >= experts
    )
    if unexpected_experts:
        failures.append(f"unexpected experts: {len(unexpected_experts)}")

    required_dense = {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        required_dense.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate.weight",
            }
        )
    missing_dense = sorted(required_dense - tensors.keys())
    if missing_dense:
        failures.append(f"missing dense tensors: {len(missing_dense)}")

    metadata_files = ["tokenizer.json", "tokenizer_config.json"]
    missing_metadata = [name for name in metadata_files if not (args.model / name).is_file()]
    if missing_metadata:
        failures.append(f"missing metadata: {', '.join(missing_metadata)}")

    expert_bytes = sum(expert_payloads)
    payload_min = min(expert_payloads) if expert_payloads else None
    payload_max = max(expert_payloads) if expert_payloads else None
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "model_path": str(args.model.resolve()),
        "source_model": args.source_model,
        "source_revision": args.source_revision,
        "colibri_commit": args.colibri_commit,
        "architecture": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "layers": layers,
            "experts_per_layer": experts,
            "top_k": top_k,
        },
        "container": {
            "shards": len(shards),
            "tensors": len(tensors),
            "tensor_bytes": tensor_bytes,
            "expert_count_expected": layers * experts,
            "expert_count_valid": len(expert_payloads),
            "expert_bytes": expert_bytes,
            "dense_bytes": tensor_bytes - expert_bytes,
            "expert_payload_bytes_min": payload_min,
            "expert_payload_bytes_max": payload_max,
            "direct_weight_read_bytes_min": (
                min(direct_weight_read_bytes) if direct_weight_read_bytes else None
            ),
            "direct_weight_read_bytes_max": (
                max(direct_weight_read_bytes) if direct_weight_read_bytes else None
            ),
            "direct_io_safe": not direct_unsafe_experts,
            "direct_expert_count": layers * experts - len(direct_unsafe_experts),
            "buffered_tail_fallback_expert_count": len(direct_unsafe_experts),
            "active_expert_payload_bytes_per_token": (
                payload_max * top_k * layers if payload_max is not None else None
            ),
        },
        "checks": {
            "duplicate_tensor_names": duplicate_names[:20],
            "missing_experts": missing_experts[:20],
            "malformed_experts": malformed_experts[:20],
            "direct_unsafe_experts": direct_unsafe_experts[:20],
            "unexpected_experts": unexpected_experts[:20],
            "missing_dense_tensors": missing_dense[:20],
            "missing_metadata": missing_metadata,
            "failures": failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
