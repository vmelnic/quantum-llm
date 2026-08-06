"""Standard-library validator for a quantum-llm MoE trace."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def numbers(path: Path, typecode: str, block_values: int = 1 << 18):
    item_size = array.array(typecode).itemsize
    with path.open("rb") as handle:
        while block := handle.read(block_values * item_size):
            values = array.array(typecode)
            values.frombytes(block)
            yield values


def float_stats(path: Path) -> dict:
    count = 0
    total_square = 0.0
    minimum = math.inf
    maximum = -math.inf
    for values in numbers(path, "f"):
        for value in values:
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite value in {path.name}")
            count += 1
            total_square += value * value
            minimum = min(minimum, value)
            maximum = max(maximum, value)
    return {
        "values": count,
        "rms": math.sqrt(total_square / count),
        "minimum": minimum,
        "maximum": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    manifest_path = args.trace / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("format") != "quantum-llm-moe-trace-v1":
        raise RuntimeError("unsupported trace format")
    records = int(manifest["record_count"])
    hidden = int(manifest["hidden_size"])
    top_k = int(manifest["top_k"])
    sizes = {
        "input.f32": records * hidden * 4,
        "output.f32": records * hidden * 4,
        "layer.u32": records * 4,
        "sequence.u32": records * 4,
        "position.u32": records * 4,
        "route-indices.u32": records * top_k * 4,
        "route-scores.f32": records * top_k * 4,
    }
    for name, expected in sizes.items():
        path = args.trace / name
        entry = manifest["files"][name]
        if path.stat().st_size != expected or int(entry["bytes"]) != expected:
            raise RuntimeError(f"size mismatch for {name}")
        if digest(path) != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {name}")

    layer_counts: Counter[int] = Counter()
    for values in numbers(args.trace / "layer.u32", "I"):
        layer_counts.update(values)
    route_ids = [value for block in numbers(args.trace / "route-indices.u32", "I") for value in block]
    if route_ids and (min(route_ids) < 0 or max(route_ids) >= 512):
        raise RuntimeError("route expert ID outside Qwen geometry")
    score_sums = []
    pending: list[float] = []
    for values in numbers(args.trace / "route-scores.f32", "f"):
        pending.extend(values)
        complete = len(pending) // top_k
        for row in range(complete):
            scores = pending[row * top_k : (row + 1) * top_k]
            if any(not math.isfinite(value) or value < 0 for value in scores):
                raise RuntimeError("invalid route score")
            score_sums.append(sum(scores))
        pending = pending[complete * top_k :]
    if pending or len(score_sums) != records:
        raise RuntimeError("route-score row mismatch")

    result = {
        "status": "valid",
        "records": records,
        "hidden_size": hidden,
        "top_k": top_k,
        "layers": dict(sorted(layer_counts.items())),
        "input": float_stats(args.trace / "input.f32"),
        "output": float_stats(args.trace / "output.f32"),
        "route_expert_min": min(route_ids),
        "route_expert_max": max(route_ids),
        "route_score_sum_min": min(score_sums),
        "route_score_sum_max": max(score_sums),
        "manifest_sha256": digest(manifest_path),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
