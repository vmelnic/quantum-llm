"""Read-only, architecture-neutral checkpoint inventory."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import TypedDict

from .safetensors import SafeTensorCheckpoint


class DTypeInventory(TypedDict):
    tensor_count: int
    bytes: int


_NUMERIC_COMPONENT = re.compile(r"(?:(?<=\.)|^)\d+(?=\.|$)")
_TID_EID_COMPONENT = re.compile(r"tid\d+eid")


def _normalized_tensor_pattern(name: str) -> str:
    pattern = _NUMERIC_COMPONENT.sub("{n}", name)
    return _TID_EID_COMPONENT.sub("tid{n}eid", pattern)


def group_source_tensors(checkpoint: SafeTensorCheckpoint) -> list[dict[str, object]]:
    """Aggregate tensor metadata by names with numeric IDs normalized."""

    grouped: dict[str, dict[tuple[str, tuple[int, ...]], dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"tensor_count": 0, "bytes": 0})
    )
    for info in checkpoint.tensors.values():
        variant = grouped[_normalized_tensor_pattern(info.name)][(info.dtype, info.shape)]
        variant["tensor_count"] += 1
        variant["bytes"] += info.nbytes

    result: list[dict[str, object]] = []
    for pattern in sorted(grouped):
        variants = []
        pattern_count = 0
        pattern_bytes = 0
        for (dtype, shape), totals in sorted(grouped[pattern].items()):
            pattern_count += totals["tensor_count"]
            pattern_bytes += totals["bytes"]
            variants.append({
                "dtype": dtype,
                "shape": list(shape),
                **totals,
            })
        result.append({
            "pattern": pattern,
            "tensor_count": pattern_count,
            "bytes": pattern_bytes,
            "variants": variants,
        })
    return result


def inspect_source(
    checkpoint: SafeTensorCheckpoint, *, include_tensor_groups: bool = False
) -> dict[str, object]:
    """Return a deterministic summary after every shard header was validated.

    This function deliberately does not adapt, quantize, write, or mmap tensor
    payloads.  It is the safe first gate for an unfamiliar checkpoint.
    """

    by_dtype: dict[str, DTypeInventory] = defaultdict(
        lambda: {"tensor_count": 0, "bytes": 0}
    )
    total_bytes = 0
    for info in checkpoint.tensors.values():
        entry = by_dtype[info.dtype]
        entry["tensor_count"] += 1
        entry["bytes"] += info.nbytes
        total_bytes += info.nbytes

    config = checkpoint.config
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not all(
        isinstance(value, str) for value in architectures
    ):
        architectures = []

    result: dict[str, object] = {
        "format": "safetensors-source-inventory-v1",
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "shard_count": len(checkpoint.shards),
        "tensor_count": len(checkpoint.tensors),
        "tensor_bytes": total_bytes,
        "dtypes": {name: by_dtype[name] for name in sorted(by_dtype)},
    }
    if include_tensor_groups:
        result["tensor_groups"] = group_source_tensors(checkpoint)
    return result
