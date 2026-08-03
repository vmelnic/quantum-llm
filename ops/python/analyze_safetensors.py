#!/usr/bin/env python3
"""Analyze a sharded MoE SafeTensors checkpoint without loading tensor data."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
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

PACKED_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\."
    r"(?P<projection>input_linear|output_linear)\.weight$"
)
ROUTER_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.router\..*weight$"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_size(dtype: str, shape: list[int]) -> int | None:
    item_size = DTYPE_BYTES.get(dtype)
    if item_size is None:
        return None
    return math.prod(shape) * item_size


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    file_size = resolved.stat().st_size
    with resolved.open("rb") as handle:
        raw_header_length = handle.read(8)
        if len(raw_header_length) != 8:
            raise ValueError(f"{path}: missing 8-byte SafeTensors header length")
        header_length = struct.unpack("<Q", raw_header_length)[0]
        if header_length <= 0 or header_length > file_size - 8:
            raise ValueError(f"{path}: invalid header length {header_length}")
        header = json.loads(handle.read(header_length))

    data_start = 8 + header_length
    metadata = header.pop("__metadata__", {})
    shard = {
        "name": path.name,
        "snapshot_path": str(path),
        "resolved_path": str(resolved),
        "file_size": file_size,
        "header_bytes": header_length,
        "data_start": data_start,
        "metadata": metadata,
        "tensor_count": len(header),
    }
    return header, shard


def get_config_int(config: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = config.get(name)
        if value is not None:
            return int(value)
    return None


def analyze(snapshot: Path) -> dict[str, Any]:
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    config = read_json(config_path)
    index = read_json(index_path)
    weight_map: dict[str, str] = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    num_layers = get_config_int(config, "num_hidden_layers")
    num_experts = get_config_int(config, "num_local_experts", "num_experts")
    top_k = get_config_int(config, "num_experts_per_tok")
    if not num_layers or not num_experts or not top_k:
        raise ValueError("Config is missing num_hidden_layers, expert count, or top-k")

    records: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    found_names: set[str] = set()

    for shard_name in shard_names:
        header, shard_report = read_safetensors_header(snapshot / shard_name)
        max_data_end = 0
        shard_tensor_bytes = 0
        ranges: list[tuple[int, int, str]] = []

        for name, entry in header.items():
            dtype = entry["dtype"]
            shape = [int(value) for value in entry["shape"]]
            start, end = (int(value) for value in entry["data_offsets"])
            stored_bytes = end - start
            expected_bytes = tensor_size(dtype, shape)
            index_shard = weight_map.get(name)
            if index_shard != shard_name:
                errors.append(
                    f"index mismatch for {name}: header={shard_name}, index={index_shard}"
                )
            if expected_bytes is None:
                errors.append(f"unsupported dtype {dtype} for {name}")
            elif expected_bytes != stored_bytes:
                errors.append(
                    f"byte mismatch for {name}: offsets={stored_bytes}, shape={expected_bytes}"
                )
            if start < 0 or end < start:
                errors.append(f"invalid data offsets for {name}: {start}, {end}")

            category = "other"
            layer = None
            projection = None
            packed = PACKED_EXPERT_RE.match(name)
            router = ROUTER_RE.match(name)
            if packed:
                category = "packed_expert"
                layer = int(packed.group("layer"))
                projection = packed.group("projection")
                if not shape or shape[0] != num_experts:
                    errors.append(
                        f"packed expert dimension mismatch for {name}: "
                        f"shape={shape}, experts={num_experts}"
                    )
            elif router:
                category = "router"
                layer = int(router.group("layer"))

            records.append(
                {
                    "name": name,
                    "shard": shard_name,
                    "dtype": dtype,
                    "shape": shape,
                    "data_offset_start": start,
                    "data_offset_end": end,
                    "absolute_file_offset": shard_report["data_start"] + start,
                    "bytes": stored_bytes,
                    "category": category,
                    "layer": layer,
                    "projection": projection,
                }
            )
            found_names.add(name)
            shard_tensor_bytes += stored_bytes
            max_data_end = max(max_data_end, end)
            ranges.append((start, end, name))

        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                errors.append(
                    f"overlapping tensors in {shard_name}: {previous[2]} and {current[2]}"
                )

        expected_file_size = shard_report["data_start"] + max_data_end
        if expected_file_size != shard_report["file_size"]:
            errors.append(
                f"shard size mismatch for {shard_name}: "
                f"header expects {expected_file_size}, file has {shard_report['file_size']}"
            )
        shard_report["tensor_bytes"] = shard_tensor_bytes
        shard_report["expected_file_size"] = expected_file_size
        shard_reports.append(shard_report)

    missing = sorted(set(weight_map) - found_names)
    unexpected = sorted(found_names - set(weight_map))
    if missing:
        errors.append(f"{len(missing)} tensors from index are missing in shard headers")
    if unexpected:
        errors.append(f"{len(unexpected)} shard tensors are absent from index")

    records.sort(key=lambda item: item["name"])
    total_tensor_bytes = sum(item["bytes"] for item in records)
    index_total_size = int(index.get("metadata", {}).get("total_size", 0))
    if index_total_size and total_tensor_bytes != index_total_size:
        errors.append(
            f"index total_size={index_total_size}, tensor bytes={total_tensor_bytes}"
        )

    layers: list[dict[str, Any]] = []
    packed_expert_total = 0
    active_expert_total = 0
    router_total = 0
    for layer_id in range(num_layers):
        expert_records = [
            item
            for item in records
            if item["category"] == "packed_expert" and item["layer"] == layer_id
        ]
        router_records = [
            item
            for item in records
            if item["category"] == "router" and item["layer"] == layer_id
        ]
        expert_bytes = sum(item["bytes"] for item in expert_records)
        router_bytes = sum(item["bytes"] for item in router_records)
        per_expert_bytes = expert_bytes // num_experts
        active_bytes = per_expert_bytes * top_k
        packed_expert_total += expert_bytes
        active_expert_total += active_bytes
        router_total += router_bytes
        layers.append(
            {
                "layer": layer_id,
                "packed_expert_tensors": len(expert_records),
                "packed_expert_bytes": expert_bytes,
                "per_expert_bytes": per_expert_bytes,
                "active_expert_bytes_per_token": active_bytes,
                "router_bytes": router_bytes,
                "projections": {
                    item["projection"]: {
                        "shape": item["shape"],
                        "bytes": item["bytes"],
                    }
                    for item in expert_records
                },
            }
        )

    always_on_bytes = total_tensor_bytes - packed_expert_total
    active_model_bytes = always_on_bytes + active_expert_total
    traffic_grid = []
    for residual_fraction in (1.0, 0.5, 0.25, 0.1):
        for hit_rate in (0.0, 0.5, 0.9, 0.95, 0.99):
            traffic_grid.append(
                {
                    "residual_fraction": residual_fraction,
                    "ram_hit_rate": hit_rate,
                    "ssd_expert_bytes_per_token": round(
                        active_expert_total * residual_fraction * (1.0 - hit_rate)
                    ),
                }
            )

    return {
        "schema_version": 1,
        "snapshot": str(snapshot.resolve()),
        "config": {
            "architectures": config.get("architectures", []),
            "model_type": config.get("model_type"),
            "dtype": config.get("torch_dtype", config.get("dtype")),
            "num_hidden_layers": num_layers,
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_experts": num_experts,
            "num_experts_per_token": top_k,
        },
        "validation": {
            "ok": not errors,
            "errors": errors,
            "missing_tensors": missing,
            "unexpected_tensors": unexpected,
            "index_total_size": index_total_size,
            "tensor_bytes": total_tensor_bytes,
        },
        "summary": {
            "tensor_count": len(records),
            "shard_count": len(shard_reports),
            "dtype_counts": dict(Counter(item["dtype"] for item in records)),
            "total_tensor_bytes": total_tensor_bytes,
            "packed_expert_total_bytes": packed_expert_total,
            "always_on_weight_bytes": always_on_bytes,
            "active_expert_bytes_per_token": active_expert_total,
            "active_model_weight_bytes_per_token": active_model_bytes,
            "dense_equivalent_fraction": active_model_bytes / total_tensor_bytes,
        },
        "traffic_grid": traffic_grid,
        "layers": layers,
        "shards": shard_reports,
        "tensors": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = analyze(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    summary = report["summary"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation_ok": report["validation"]["ok"],
                "tensor_count": summary["tensor_count"],
                "total_tensor_bytes": summary["total_tensor_bytes"],
                "active_expert_bytes_per_token": summary[
                    "active_expert_bytes_per_token"
                ],
                "active_model_weight_bytes_per_token": summary[
                    "active_model_weight_bytes_per_token"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
