"""Read-only, architecture-neutral checkpoint inventory."""

from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

from .safetensors import SafeTensorCheckpoint


class DTypeInventory(TypedDict):
    tensor_count: int
    bytes: int


def inspect_source(checkpoint: SafeTensorCheckpoint) -> dict[str, object]:
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

    return {
        "format": "safetensors-source-inventory-v1",
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "shard_count": len(checkpoint.shards),
        "tensor_count": len(checkpoint.tensors),
        "tensor_bytes": total_bytes,
        "dtypes": {name: by_dtype[name] for name in sorted(by_dtype)},
    }
