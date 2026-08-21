#!/usr/bin/env python3
"""Gate reversible structural FP16 KV coding on a captured runtime sample."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from pathlib import Path
import struct

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


MAGIC = b"QKVF16S1"
HEADER = struct.Struct("<8s8I")
HUFFMAN_ESCAPE_SYMBOL = 65536
HUFFMAN_ESCAPE_FREQUENCY_DIVISOR = 4096


def _huffman_code_lengths(
        frequencies: list[tuple[int, int]]) -> dict[int, int]:
    """Return deterministic binary Huffman lengths for positive frequencies."""
    heap: list[tuple[int, int, int]] = []
    children: dict[int, tuple[int, int]] = {}
    next_node = HUFFMAN_ESCAPE_SYMBOL + 1
    serial = 0
    for symbol, frequency in frequencies:
        if frequency <= 0:
            continue
        heap.append((frequency, serial, symbol))
        serial += 1
    if not heap:
        raise ValueError("Huffman alphabet is empty")
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][2]: 1}
    while len(heap) > 1:
        left_weight, _, left = heapq.heappop(heap)
        right_weight, _, right = heapq.heappop(heap)
        node = next_node
        next_node += 1
        children[node] = (left, right)
        heapq.heappush(
            heap, (left_weight + right_weight, serial, node))
        serial += 1
    lengths: dict[int, int] = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        branch = children.get(node)
        if branch is None:
            lengths[node] = max(1, depth)
            continue
        stack.append((branch[0], depth + 1))
        stack.append((branch[1], depth + 1))
    return lengths


def _held_out_raw_huffman(current: "np.ndarray") -> dict[str, object]:
    """Train on the first sampled half and price the disjoint second half."""
    split = current.shape[0] // 2
    if split == 0 or split == current.shape[0]:
        raise ValueError("Huffman holdout requires at least two sampled rows")
    train = np.bincount(current[:split].reshape(-1), minlength=65536)
    test = np.bincount(current[split:].reshape(-1), minlength=65536)
    train_total = int(train.sum())
    escape_frequency = max(1, train_total // HUFFMAN_ESCAPE_FREQUENCY_DIVISOR)
    frequencies = [
        (int(symbol), int(frequency))
        for symbol, frequency in enumerate(train)
        if frequency
    ]
    frequencies.append((HUFFMAN_ESCAPE_SYMBOL, escape_frequency))
    lengths = _huffman_code_lengths(frequencies)
    escape_length = lengths[HUFFMAN_ESCAPE_SYMBOL]
    encoded_bits = 0
    unseen_values = 0
    for symbol, frequency in enumerate(test):
        count = int(frequency)
        if not count:
            continue
        length = lengths.get(symbol)
        if length is None:
            unseen_values += count
            encoded_bits += count * (escape_length + 16)
        else:
            encoded_bits += count * length
    test_values = int(test.sum())
    # Canonical decoding needs only each observed 16-bit symbol and its 8-bit
    # code length. Codes are reconstructed from that table. The fixed header
    # carries the entry count and escape length.
    codebook_bytes = 8 + (len(frequencies) - 1) * 3
    return {
        "training_values": train_total,
        "test_values": test_values,
        "observed_symbols": len(frequencies) - 1,
        "escape_code_bits": escape_length,
        "unseen_test_values": unseen_values,
        "unseen_test_fraction": unseen_values / test_values,
        "encoded_test_bits": encoded_bits,
        "held_out_bits_per_value": encoded_bits / test_values,
        "canonical_codebook_bytes": codebook_bytes,
    }


def _entropy(counts: "np.ndarray") -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    nonzero = counts[counts != 0].astype(np.float64)
    return float(math.log2(total) - np.dot(nonzero, np.log2(nonzero)) / total)


def _conditional_entropy(counts: "np.ndarray") -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    row_totals = counts.sum(axis=1)
    nonempty = row_totals != 0
    row_weight = np.dot(
        row_totals[nonempty].astype(np.float64),
        np.log2(row_totals[nonempty].astype(np.float64)),
    )
    values = counts[counts != 0].astype(np.float64)
    symbol_weight = np.dot(values, np.log2(values))
    return float((row_weight - symbol_weight) / total)


def _stream_profile(current: "np.ndarray", previous: "np.ndarray",
                    heads: int, head_dim: int) -> dict[str, object]:
    samples, channels = current.shape
    raw = np.bincount(current.reshape(-1), minlength=65536)
    exponent = (current >> 10) & 31
    previous_exponent = (previous >> 10) & 31
    exponent_hist = np.bincount(exponent.reshape(-1), minlength=32)
    raw_h0 = _entropy(raw)
    exponent_h0 = _entropy(exponent_hist)
    residual_given_exponent = raw_h0 - exponent_h0

    channel = np.arange(channels, dtype=np.uint32)[None, :]
    exponent_by_channel = np.bincount(
        (channel * 32 + exponent).reshape(-1), minlength=channels * 32
    ).reshape((channels, 32))
    exponent_channel = _conditional_entropy(exponent_by_channel)

    temporal_codes = (
        ((channel * 32 + previous_exponent.astype(np.uint32)) * 32)
        + exponent.astype(np.uint32)
    )
    temporal = np.bincount(
        temporal_codes.reshape(-1), minlength=channels * 32 * 32
    ).reshape((channels * 32, 32))
    exponent_temporal = _conditional_entropy(temporal)

    high = current >> 8
    low = current & 255
    high_by_channel = np.bincount(
        (channel * 256 + high).reshape(-1), minlength=channels * 256
    ).reshape((channels, 256))
    high_channel = _conditional_entropy(high_by_channel)
    low_by_high = np.bincount(
        (high.astype(np.uint32) * 256 + low).reshape(-1), minlength=65536
    ).reshape((256, 256))
    low_given_high = _conditional_entropy(low_by_high)

    shaped = exponent.reshape((samples, heads, head_dim))
    palette_bits = 0
    modal_bits = 0
    for row in shaped.reshape((-1, head_dim)):
        counts = np.bincount(row, minlength=32)
        unique = int(np.count_nonzero(counts))
        width = 0 if unique <= 1 else math.ceil(math.log2(unique))
        palette_bits += 5 * unique + width * head_dim
        modal_bits += 5 + head_dim + 5 * (head_dim - int(counts.max()))
    values = samples * channels
    exponent_palette = palette_bits / values
    exponent_modal = modal_bits / values

    candidates = {
        "raw_h0": raw_h0,
        "exponent_channel": residual_given_exponent + exponent_channel,
        "exponent_temporal_channel": residual_given_exponent + exponent_temporal,
        "exponent_head_palette": residual_given_exponent + exponent_palette,
        "exponent_head_mode_exceptions": residual_given_exponent + exponent_modal,
        "high_byte_channel_low_given_high": high_channel + low_given_high,
    }
    best = min(candidates, key=candidates.get)
    huffman = _held_out_raw_huffman(current)
    return {
        "sampled_values": values,
        "raw_h0_bits_per_value": raw_h0,
        "exponent_h0_bits_per_value": exponent_h0,
        "residual_given_exponent_h0_bits_per_value": residual_given_exponent,
        "candidate_bits_per_value": candidates,
        "best_candidate": best,
        "best_candidate_bits_per_value": candidates[best],
        "held_out_raw_huffman": huffman,
    }


def _capacity(weight_gate: dict[str, object], kv_profile: dict[str, object],
              unavailable_mib: int, runtime_reserve_mib: int) -> dict[str, object]:
    target = weight_gate["target_only"]
    device = int(weight_gate["device_bytes"])
    weights = int(target["ideal_weight_payload_bytes"])
    recurrent = int(target["required_recurrent_state_bytes"])
    raw_kv = int(target["exact_fp16_kv_raw_bytes"])
    if raw_kv < int(kv_profile["raw_fp16_bytes"]):
        raise ValueError("target KV geometry is smaller than the captured profile")

    def threshold(reserved: int) -> float:
        return 16.0 * (device - weights - recurrent - reserved) / raw_kv

    unavailable = unavailable_mib << 20
    runtime = runtime_reserve_mib << 20
    return {
        "device_bytes": device,
        "ideal_weight_payload_bytes": weights,
        "required_recurrent_state_bytes": recurrent,
        "device_unavailable_bytes": unavailable,
        "minimum_runtime_reserve_bytes": runtime,
        "maximum_zero_overhead_bits_per_value": threshold(0),
        "maximum_after_device_reservation_bits_per_value": threshold(unavailable),
        "maximum_practical_bits_per_value": threshold(unavailable + runtime),
    }


def evaluate(profile_path: Path, weight_gate_path: Path,
             unavailable_mib: int, runtime_reserve_mib: int) -> dict[str, object]:
    if np is None:
        raise RuntimeError("structural KV profiling requires NumPy")
    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    weight_gate = json.loads(weight_gate_path.read_text(encoding="utf-8-sig"))
    if profile.get("format") != "fp16-kv-lossless-profile-v1" or \
            weight_gate.get("format") != "exact-weight-kv-residency-gate-v1":
        raise ValueError("invalid residency gate inputs")
    sample_path = profile_path.parent / profile["sample_file"]
    with sample_path.open("rb") as stream:
        header = stream.read(HEADER.size)
    magic, schema, streams, samples, channels, populated, heads, head_dim, rows = \
        HEADER.unpack(header)
    if magic != MAGIC or schema != 1 or rows != 2 or channels != heads * head_dim:
        raise ValueError("invalid FP16 KV sample header")
    expected = HEADER.size + streams * samples * rows * channels * 2
    if sample_path.stat().st_size != expected:
        raise ValueError("truncated FP16 KV sample")
    payload = np.memmap(sample_path, dtype="<u2", mode="r", offset=HEADER.size,
                        shape=(streams, samples, rows, channels))
    stream_profiles = []
    for index in range(streams):
        item = _stream_profile(payload[index, :, 0, :], payload[index, :, 1, :],
                               heads, head_dim)
        item["layer"] = index // 2
        item["kind"] = "key" if index % 2 == 0 else "value"
        stream_profiles.append(item)
    best = sum(float(item["best_candidate_bits_per_value"])
               for item in stream_profiles) / streams
    raw = sum(float(item["raw_h0_bits_per_value"])
              for item in stream_profiles) / streams
    huffman = sum(
        float(item["held_out_raw_huffman"]["held_out_bits_per_value"])
        for item in stream_profiles
    ) / streams
    huffman_codebook_bytes = sum(
        int(item["held_out_raw_huffman"]["canonical_codebook_bytes"])
        for item in stream_profiles
    )
    huffman_unseen = sum(
        int(item["held_out_raw_huffman"]["unseen_test_values"])
        for item in stream_profiles
    )
    huffman_test_values = sum(
        int(item["held_out_raw_huffman"]["test_values"])
        for item in stream_profiles
    )
    capacity = _capacity(weight_gate, profile, unavailable_mib,
                         runtime_reserve_mib)
    zero_pass = best <= float(capacity["maximum_zero_overhead_bits_per_value"])
    device_pass = best <= float(
        capacity["maximum_after_device_reservation_bits_per_value"])
    practical_pass = best <= float(capacity["maximum_practical_bits_per_value"])
    return {
        "format": "exact-kv-structural-gate-v1",
        "profile": str(profile_path),
        "weight_gate": str(weight_gate_path),
        "populated_tokens": populated,
        "sampled_tokens_per_stream": samples,
        "sampled_streams": streams,
        "average_raw_h0_bits_per_value": raw,
        "best_structural_bits_per_value": best,
        "held_out_raw_huffman_bits_per_value": huffman,
        "held_out_raw_huffman_codebook_bytes": huffman_codebook_bytes,
        "held_out_raw_huffman_unseen_fraction": (
            huffman_unseen / huffman_test_values),
        "capacity": capacity,
        "zero_overhead_capacity_pass": zero_pass,
        "device_reservation_capacity_pass": device_pass,
        "practical_capacity_pass": practical_pass,
        "gate_pass": practical_pass,
        "interpretation": (
            "Failure at zero overhead rejects all measured reversible structural "
            "candidates. Passing the practical sample gate only admits a full-"
            "context capture and executable codec bandwidth test."
        ),
        "streams": stream_profiles,
    }


def _transactional_write(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("weight_gate", type=Path)
    parser.add_argument("--device-unavailable-mib", type=int, default=0)
    parser.add_argument("--runtime-reserve-mib", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.profile, args.weight_gate,
                      args.device_unavailable_mib, args.runtime_reserve_mib)
    if args.output:
        _transactional_write(args.output, result)
    summary = {key: value for key, value in result.items() if key != "streams"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
