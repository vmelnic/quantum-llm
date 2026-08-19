#!/usr/bin/env python3
"""Measure exact lossless-compression headroom in fixed-size expert records.

The probe is deliberately format-neutral: callers describe the repeated
weight/scale segment geometry and provide one or more record files.  It never
rewrites the source files.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
from pathlib import Path
import zlib

import numpy as np


def _entropy(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probabilities = counts[counts != 0].astype(np.float64) / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _ratio(raw_bytes: int, encoded_bytes: int) -> float:
    return float(encoded_bytes) / raw_bytes if raw_bytes else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--record-bytes", type=int, required=True)
    parser.add_argument("--weight-bytes", type=int, required=True)
    parser.add_argument("--scale-bytes", type=int, required=True)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--records-per-file", type=int, required=True)
    parser.add_argument("--sample-records-per-file", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    segment_bytes = args.weight_bytes + args.scale_bytes
    if (
        args.record_bytes <= 0
        or args.weight_bytes <= 0
        or args.scale_bytes <= 0
        or args.segments <= 0
        or args.records_per_file <= 0
        or args.sample_records_per_file <= 0
        or segment_bytes * args.segments != args.record_bytes
    ):
        raise SystemExit("invalid fixed-record segment geometry")

    byte_counts = np.zeros(256, dtype=np.uint64)
    nibble_counts = np.zeros(16, dtype=np.uint64)
    scale_counts = np.zeros(256, dtype=np.uint64)
    weight_raw = scale_raw = record_raw = 0
    weight_zlib = scale_zlib = record_zlib = 0
    xor_raw = xor_zlib = xor_equal = 0
    sampled: list[dict[str, int | str]] = []
    previous_weights: np.ndarray | None = None

    for path in args.files:
        size = path.stat().st_size
        expected = args.record_bytes * args.records_per_file
        if size != expected:
            raise SystemExit(f"{path}: expected {expected} bytes, found {size}")
        sample_count = min(args.sample_records_per_file, args.records_per_file)
        record_indices = np.linspace(
            0, args.records_per_file - 1, sample_count, dtype=np.int64
        )
        with path.open("rb") as source, mmap.mmap(
            source.fileno(), 0, access=mmap.ACCESS_READ
        ) as mapped:
            for record_index in record_indices:
                begin = int(record_index) * args.record_bytes
                record_view = memoryview(mapped)[begin : begin + args.record_bytes]
                weights = bytearray()
                scales = bytearray()
                for segment in range(args.segments):
                    segment_begin = segment * segment_bytes
                    weights.extend(
                        record_view[
                            segment_begin : segment_begin + args.weight_bytes
                        ]
                    )
                    scales.extend(
                        record_view[
                            segment_begin
                            + args.weight_bytes : segment_begin
                            + segment_bytes
                        ]
                    )
                del record_view

                weight_array = np.frombuffer(weights, dtype=np.uint8)
                scale_array = np.frombuffer(scales, dtype=np.uint8)
                byte_counts += np.bincount(weight_array, minlength=256).astype(
                    np.uint64
                )
                nibble_counts += np.bincount(
                    weight_array & np.uint8(0x0F), minlength=16
                ).astype(np.uint64)
                nibble_counts += np.bincount(
                    weight_array >> np.uint8(4), minlength=16
                ).astype(np.uint64)
                scale_counts += np.bincount(scale_array, minlength=256).astype(
                    np.uint64
                )

                record = bytes(mapped[begin : begin + args.record_bytes])
                compressed_weights = zlib.compress(weights, level=6)
                compressed_scales = zlib.compress(scales, level=6)
                compressed_record = zlib.compress(record, level=6)
                weight_raw += len(weights)
                scale_raw += len(scales)
                record_raw += len(record)
                weight_zlib += len(compressed_weights)
                scale_zlib += len(compressed_scales)
                record_zlib += len(compressed_record)

                if previous_weights is not None:
                    xor = np.bitwise_xor(weight_array, previous_weights)
                    xor_raw += int(xor.nbytes)
                    xor_equal += int(np.count_nonzero(xor == 0))
                    xor_zlib += len(zlib.compress(xor.tobytes(), level=6))
                previous_weights = weight_array.copy()
                sampled.append({"file": str(path), "record": int(record_index)})
        previous_weights = None

    nibble_entropy = _entropy(nibble_counts)
    scale_entropy = _entropy(scale_counts)
    theoretical_weight_ratio = nibble_entropy / 4.0
    theoretical_scale_ratio = scale_entropy / 8.0
    theoretical_record_ratio = (
        theoretical_weight_ratio * weight_raw
        + theoretical_scale_ratio * scale_raw
    ) / record_raw
    result = {
        "schema_version": 1,
        "sampled_records": sampled,
        "sample_bytes": record_raw,
        "weight_nibble_entropy_bits": nibble_entropy,
        "scale_byte_entropy_bits": scale_entropy,
        "theoretical_independent_symbol_ratio": theoretical_record_ratio,
        "zlib_ratio": _ratio(record_raw, record_zlib),
        "weight_zlib_ratio": _ratio(weight_raw, weight_zlib),
        "scale_zlib_ratio": _ratio(scale_raw, scale_zlib),
        "adjacent_expert_equal_weight_byte_fraction": (
            float(xor_equal) / xor_raw if xor_raw else 0.0
        ),
        "adjacent_expert_xor_zlib_ratio": _ratio(xor_raw, xor_zlib),
        "weight_nibble_probabilities": [
            float(value) / int(nibble_counts.sum()) for value in nibble_counts
        ],
        "scale_byte_probabilities": {
            str(index): float(value) / int(scale_counts.sum())
            for index, value in enumerate(scale_counts)
            if value
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
