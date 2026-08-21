#!/usr/bin/env python3
"""Reject lossless FP16 KV residency when even ideal entropy cannot fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(profile_path: Path, device_gib: float,
             hot_weight_bytes: int) -> dict[str, object]:
    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    if profile.get("format") != "fp16-kv-lossless-profile-v1":
        raise ValueError("invalid FP16 KV profile format")
    bits = float(profile["best_predictor_h0_bits_per_value"])
    raw_bytes = int(profile["raw_fp16_bytes"])
    ideal_bytes = int(profile["ideal_entropy_payload_bytes"])
    device_bytes = int(device_gib * (1 << 30))
    if not 0.0 < bits <= 16.0 or raw_bytes <= 0 or ideal_bytes <= 0:
        raise ValueError("invalid FP16 KV entropy measurements")
    if hot_weight_bytes <= 0 or hot_weight_bytes >= device_bytes:
        raise ValueError("invalid hot-weight/device capacity")
    absolute_available = device_bytes - hot_weight_bytes
    maximum_bits = 16.0 * absolute_available / raw_bytes
    capacity_pass = ideal_bytes <= absolute_available
    return {
        "format": "exact-kv-residency-gate-v1",
        "profile": str(profile_path),
        "device_bytes": device_bytes,
        "hot_weight_bytes": hot_weight_bytes,
        "raw_fp16_kv_bytes": raw_bytes,
        "absolute_zero_overhead_kv_bytes": absolute_available,
        "maximum_zero_overhead_bits_per_value": maximum_bits,
        "measured_best_predictor_h0_bits_per_value": bits,
        "ideal_entropy_payload_bytes": ideal_bytes,
        "ideal_entropy_compression_ratio": raw_bytes / ideal_bytes,
        "zero_overhead_capacity_pass": capacity_pass,
        "gate_pass": capacity_pass,
        "interpretation": (
            "Failure rejects the measured raw/XOR/delta lossless residency "
            "family even with the impossible advantage of zero runtime, "
            "state, allocator, display, and codec metadata bytes. Success "
            "only admits an executable codec capacity/throughput gate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("--device-gib", type=float, default=24.0)
    parser.add_argument("--hot-weight-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.profile, args.device_gib, args.hot_weight_bytes)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
