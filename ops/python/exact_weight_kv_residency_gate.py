#!/usr/bin/env python3
"""Measure optimistic lossless FP4 entropy and gate exact FP16 KV residency.

The runtime program, rather than a model-family name, defines the target-call
tensor set.  Sparse embedding lookups are excluded from permanent residency;
their selected rows still have to be staged by a future executable design.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import mmap
import os
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # The offline analyzer declares this at execution.
    np = None  # type: ignore[assignment]


FP4_ENCODING = "FP4_E2M1"
SPARSE_OPERATION_CAPABILITIES = {"embedding.lookup.fp4-block32.v1"}


def _entropy_bits(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    nonzero = counts[counts != 0].astype(np.float64)
    return float(total * math.log2(total) - np.dot(nonzero, np.log2(nonzero)))


class _SymbolProfile:
    def __init__(self, alphabet: int, order2: bool) -> None:
        self.alphabet = alphabet
        self.raw = np.zeros(alphabet, dtype=np.uint64)
        self.xor = np.zeros(alphabet, dtype=np.uint64)
        self.delta = np.zeros(alphabet, dtype=np.uint64)
        self.order1 = np.zeros(alphabet * alphabet, dtype=np.uint64)
        self.order2 = (
            np.zeros(alphabet * alphabet * alphabet, dtype=np.uint64)
            if order2 else None
        )
        self.previous: int | None = None
        self.previous2: int | None = None
        self.symbols = 0

    def add(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.uint16)
        self.symbols += int(values.size)
        self.raw += np.bincount(values, minlength=self.alphabet).astype(np.uint64)

        prior = np.empty(values.size, dtype=np.uint16)
        prior[0] = values[0] if self.previous is None else self.previous
        prior[1:] = values[:-1]
        self.xor += np.bincount(
            np.bitwise_xor(values, prior), minlength=self.alphabet
        ).astype(np.uint64)
        self.delta += np.bincount(
            (values - prior) % self.alphabet, minlength=self.alphabet
        ).astype(np.uint64)

        if self.previous is None:
            transition_values = values[1:]
            transition_prior = values[:-1]
        else:
            transition_values = values
            transition_prior = prior
        if transition_values.size:
            pairs = transition_prior.astype(np.uint32) * self.alphabet + transition_values
            self.order1 += np.bincount(
                pairs, minlength=self.alphabet * self.alphabet
            ).astype(np.uint64)

        if self.order2 is not None:
            prefix: list[int] = []
            if self.previous2 is not None:
                prefix.append(self.previous2)
            if self.previous is not None:
                prefix.append(self.previous)
            combined = np.concatenate((np.asarray(prefix, dtype=np.uint16), values))
            if combined.size >= 3:
                triples = (
                    combined[:-2].astype(np.uint32) * self.alphabet * self.alphabet
                    + combined[1:-1].astype(np.uint32) * self.alphabet
                    + combined[2:]
                )
                self.order2 += np.bincount(
                    triples, minlength=self.alphabet ** 3
                ).astype(np.uint64)

        self.previous2 = int(values[-2]) if values.size >= 2 else self.previous
        self.previous = int(values[-1])

    @staticmethod
    def _conditional_bits(table: np.ndarray, alphabet: int) -> float:
        rows = table.reshape((-1, alphabet))
        result = 0.0
        for row in rows:
            result += _entropy_bits(row)
        return result

    def result(self) -> dict[str, object]:
        candidates = {
            "raw_h0": _entropy_bits(self.raw),
            "xor_h0": _entropy_bits(self.xor),
            "delta_h0": _entropy_bits(self.delta),
            "order1_conditional": self._conditional_bits(
                self.order1, self.alphabet
            ),
        }
        if self.order2 is not None:
            candidates["order2_conditional"] = self._conditional_bits(
                self.order2, self.alphabet
            )
        best = min(candidates, key=candidates.get)
        return {
            "symbols": self.symbols,
            "candidate_payload_bits": candidates,
            "best_predictor": best,
            "ideal_payload_bits": candidates[best],
            "ideal_bits_per_symbol": candidates[best] / self.symbols,
        }


def _profile_section(
    image: mmap.mmap, offset: int, byte_count: int, *, nibbles: bool,
    chunk_bytes: int,
) -> dict[str, object]:
    profiler = _SymbolProfile(16 if nibbles else 256, order2=nibbles)
    end = offset + byte_count
    for begin in range(offset, end, chunk_bytes):
        chunk = np.frombuffer(
            image, dtype=np.uint8, count=min(chunk_bytes, end - begin), offset=begin
        )
        if nibbles:
            symbols = np.empty(chunk.size * 2, dtype=np.uint16)
            symbols[0::2] = chunk & 0x0F
            symbols[1::2] = chunk >> 4
        else:
            symbols = chunk.astype(np.uint16)
        profiler.add(symbols)
        del symbols, chunk
    result = profiler.result()
    result["source_bytes"] = byte_count
    return result


def _parse_program(path: Path) -> tuple[dict[int, str], dict[int, set[str]],
                                        dict[str, str], dict[str, int]]:
    operations: dict[int, str] = {}
    bindings: dict[int, set[str]] = defaultdict(set)
    exact: dict[str, str] = {}
    attributes: dict[str, int] = {}
    with path.open(encoding="utf-8-sig") as stream:
        if stream.readline().rstrip("\r\n") != "expert-runtime-model-v1":
            raise ValueError("invalid runtime model header")
        for source in stream:
            fields = source.rstrip("\r\n").split("\t")
            if fields[0] == "operation" and len(fields) == 7:
                operations[int(fields[1])] = fields[3]
            elif fields[0] == "operation_tensor" and len(fields) == 4:
                bindings[int(fields[1])].add(fields[3])
            elif fields[0] == "exact_decode_tensor" and len(fields) == 3:
                exact[fields[1]] = fields[2]
            elif fields[0] == "attribute" and len(fields) == 3:
                attributes[fields[1]] = int(fields[2])
    if not operations or set(bindings) - set(operations):
        raise ValueError("runtime operation graph is incomplete")
    return operations, bindings, exact, attributes


def _required_recurrent_state_bytes(
    operations: dict[int, str], attributes: dict[str, int]
) -> int:
    recurrent = sum(
        capability == "block.recurrent-linear-attention.split-gated-delta.v1"
        for capability in operations.values()
    )
    if recurrent == 0:
        return 0
    required = (
        "linear_conv_kernel", "linear_key_head_dim", "linear_value_head_dim",
        "linear_key_heads", "linear_value_heads",
    )
    if any(key not in attributes for key in required):
        raise ValueError("recurrent-state geometry is incomplete")
    key_width = attributes["linear_key_heads"] * attributes["linear_key_head_dim"]
    value_width = (
        attributes["linear_value_heads"] * attributes["linear_value_head_dim"]
    )
    convolution = (2 * key_width + value_width) * attributes["linear_conv_kernel"]
    matrix = (
        attributes["linear_value_heads"]
        * attributes["linear_key_head_dim"]
        * attributes["linear_value_head_dim"]
    )
    return recurrent * (convolution + matrix) * 4


def _tensor_sets(
    operations: dict[int, str], bindings: dict[int, set[str]], exact: dict[str, str]
) -> tuple[set[str], set[str], set[str]]:
    sparse: set[str] = set()
    target: set[str] = set()
    for operation, capability in operations.items():
        if capability in SPARSE_OPERATION_CAPABILITIES:
            sparse.update(bindings[operation])
        else:
            target.update(bindings[operation])
    exact_dense = {
        tensor for role, tensor in exact.items() if role != "token_embedding"
    }
    if "token_embedding" in exact:
        sparse.add(exact["token_embedding"])
    return target, target | exact_dense, sparse


def _transactional_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def evaluate(
    artifact: Path, kv_profile_path: Path, device_gib: float = 24.0,
    chunk_mib: int = 16,
) -> dict[str, object]:
    if np is None:
        raise RuntimeError("exact weight profiling requires NumPy")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    program_metadata = manifest["model_program"]
    operations, bindings, exact, attributes = _parse_program(
        artifact / program_metadata["path"]
    )
    target, target_exact, sparse = _tensor_sets(operations, bindings, exact)
    tensor_index = {item["name"]: item for item in manifest["tensors"]}
    missing = (target_exact | sparse) - set(tensor_index)
    if missing:
        raise ValueError(f"runtime tensors absent from manifest: {sorted(missing)[:3]}")

    pack_handles: dict[str, object] = {}
    pack_images: dict[str, mmap.mmap] = {}
    profiles: dict[str, dict[str, object]] = {}
    try:
        for pack in manifest["packs"]:
            handle = (artifact / pack["name"]).open("rb")
            pack_handles[pack["name"]] = handle
            pack_images[pack["name"]] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        for name in sorted(target_exact):
            tensor = tensor_index[name]
            image = pack_images[tensor["pack"]]
            sections: dict[str, object] = {}
            ideal_bits = 0.0
            raw_bytes = 0
            for section_name in ("data", "scales"):
                section = tensor["sections"][section_name]
                byte_count = int(section["bytes"])
                if byte_count == 0:
                    continue
                section_profile = _profile_section(
                    image,
                    int(tensor["offset"]) + int(section["offset"]),
                    byte_count,
                    nibbles=(
                        tensor["stored_dtype"] == FP4_ENCODING
                        and section_name == "data"
                    ),
                    chunk_bytes=chunk_mib << 20,
                )
                sections[section_name] = section_profile
                ideal_bits += float(section_profile["ideal_payload_bits"])
                raw_bytes += byte_count
            profiles[name] = {
                "stored_dtype": tensor["stored_dtype"],
                "raw_allocation_bytes": raw_bytes,
                "ideal_payload_bits": ideal_bits,
                "sections": sections,
            }
    finally:
        for image in pack_images.values():
            image.close()
        for handle in pack_handles.values():
            handle.close()

    kv_profile = json.loads(kv_profile_path.read_text(encoding="utf-8-sig"))
    if kv_profile.get("format") != "fp16-kv-lossless-profile-v1":
        raise ValueError("invalid FP16 KV profile format")
    kv_raw = int(kv_profile["raw_fp16_bytes"])
    kv_ideal = int(kv_profile["ideal_entropy_payload_bytes"])
    device_bytes = int(device_gib * (1 << 30))
    recurrent_bytes = _required_recurrent_state_bytes(operations, attributes)

    def scenario(names: Iterable[str]) -> dict[str, object]:
        selected = list(names)
        raw = sum(int(profiles[name]["raw_allocation_bytes"]) for name in selected)
        ideal_bits = sum(float(profiles[name]["ideal_payload_bits"]) for name in selected)
        ideal = math.ceil(ideal_bits / 8.0)
        total = ideal + kv_ideal + recurrent_bytes
        return {
            "tensor_count": len(selected),
            "raw_weight_allocation_bytes": raw,
            "ideal_weight_payload_bytes": ideal,
            "ideal_weight_compression_ratio": None if ideal == 0 else raw / ideal,
            "ideal_weight_bits_per_raw_byte": ideal_bits / raw,
            "exact_fp16_kv_raw_bytes": kv_raw,
            "ideal_exact_fp16_kv_payload_bytes": kv_ideal,
            "required_recurrent_state_bytes": recurrent_bytes,
            "absolute_zero_overhead_total_bytes": total,
            "absolute_zero_overhead_headroom_bytes": device_bytes - total,
            "absolute_zero_overhead_capacity_pass": total <= device_bytes,
        }

    target_result = scenario(target)
    target_exact_result = scenario(target_exact)
    return {
        "format": "exact-weight-kv-residency-gate-v1",
        "artifact": str(artifact),
        "kv_profile": str(kv_profile_path),
        "device_bytes": device_bytes,
        "selection": {
            "target_tensor_count": len(target),
            "target_plus_exact_tensor_count": len(target_exact),
            "sparse_nonresident_tensor_count": len(sparse),
            "sparse_nonresident_tensors": sorted(sparse),
        },
        "target_only": target_result,
        "target_plus_exact_decode": target_exact_result,
        "gate_pass": bool(target_result["absolute_zero_overhead_capacity_pass"]),
        "interpretation": (
            "Failure rejects per-tensor raw/XOR/delta/order-1/order-2 static "
            "lossless coding even with ideal fractional-bit payloads, sparse "
            "embedding placement, and no codec metadata, decoder workspace, "
            "allocator, CUDA context, or display reservation. Success only "
            "admits a concrete executable codec bandwidth gate."
        ),
        "tensor_profiles": profiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("kv_profile", type=Path)
    parser.add_argument("--device-gib", type=float, default=24.0)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.artifact, args.kv_profile, args.device_gib, args.chunk_mib)
    if args.output:
        _transactional_json(args.output, result)
    summary = {key: value for key, value in result.items() if key != "tensor_profiles"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
