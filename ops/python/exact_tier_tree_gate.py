#!/usr/bin/env python3
"""Evaluate the real bounded-proposer prerequisite for exact tiered decode."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TreeRecord:
    start_position: int
    horizon: int
    beam_width: int
    nodes: int
    proposer_ns: int
    peak_host_bytes: int
    accepted_depth: int
    termination: str


def read_tree_trace(path: Path) -> list[TreeRecord]:
    records: list[TreeRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = [
            "start_position", "horizon", "beam_width", "nodes",
            "proposer_ns", "peak_host_bytes", "accepted_depth",
            "termination",
        ]
        if reader.fieldnames != required:
            raise ValueError("exact-tier tree trace header is invalid")
        for row in reader:
            numeric = {name: int(row[name]) for name in required[:-1]}
            record = TreeRecord(**numeric, termination=row["termination"])
            if (record.horizon < 1 or record.beam_width < 1 or
                    record.nodes < 1 or record.proposer_ns < 0 or
                    record.peak_host_bytes < 0 or
                    not 0 <= record.accepted_depth <= record.horizon or
                    record.termination not in {"mismatch", "horizon",
                                               "request_end"}):
                raise ValueError("exact-tier tree trace contains an invalid row")
            records.append(record)
    if not records:
        raise ValueError("exact-tier tree trace is empty")
    return records


def evaluate(path: Path, resident_fraction_ppm: int, kv_gib: float,
             h2d_gib_per_second: float, required_tps: float,
             required_mean_accepted: float, node_budget: int) -> dict[str, object]:
    if not 0 <= resident_fraction_ppm < 1_000_000:
        raise ValueError("resident fraction must be inside [0, 1000000)")
    if min(kv_gib, h2d_gib_per_second, required_tps,
           required_mean_accepted, node_budget) <= 0:
        raise ValueError("gate parameters must be positive")
    records = read_tree_trace(path)
    cycles = len(records)
    accepted = sum(record.accepted_depth for record in records)
    emitted_upper_bound = sum(max(1, record.accepted_depth)
                              for record in records)
    proposer_seconds = sum(record.proposer_ns for record in records) / 1e9
    transfer_seconds_per_cycle = (
        (1.0 - resident_fraction_ppm / 1_000_000.0) * kv_gib /
        h2d_gib_per_second
    )
    necessary_cycle_seconds = (
        cycles * transfer_seconds_per_cycle + proposer_seconds
    )
    mean_accepted = accepted / cycles
    necessary_upper_bound_tps = emitted_upper_bound / necessary_cycle_seconds
    maximum_nodes = max(record.nodes for record in records)
    acceptance_pass = mean_accepted >= required_mean_accepted
    capacity_pass = maximum_nodes <= node_budget
    latency_pass = necessary_upper_bound_tps >= required_tps
    return {
        "format": "exact-tier-tree-gate-v1",
        "trace": str(path),
        "cycles": cycles,
        "mean_accepted_depth": mean_accepted,
        "minimum_accepted_depth": min(record.accepted_depth
                                      for record in records),
        "maximum_accepted_depth": max(record.accepted_depth
                                      for record in records),
        "mean_nodes": sum(record.nodes for record in records) / cycles,
        "maximum_nodes": maximum_nodes,
        "mean_proposer_seconds": proposer_seconds / cycles,
        "peak_proposer_host_bytes": max(record.peak_host_bytes
                                         for record in records),
        "resident_fraction_ppm": resident_fraction_ppm,
        "missing_kv_transfer_seconds_per_cycle": transfer_seconds_per_cycle,
        "necessary_upper_bound_tps_before_verifier_compute":
            necessary_upper_bound_tps,
        "required_mean_accepted_depth": required_mean_accepted,
        "required_tps": required_tps,
        "node_budget": node_budget,
        "acceptance_pass": acceptance_pass,
        "capacity_pass": capacity_pass,
        "latency_necessary_bound_pass": latency_pass,
        "gate_pass": acceptance_pass and capacity_pass and latency_pass,
        "interpretation": (
            "Failure rejects this proposer. Success only admits the streamed "
            "exact verifier implementation because verifier compute is not "
            "included in this upper bound."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--resident-fraction-ppm", type=int, default=250_000)
    parser.add_argument("--kv-gib", type=float, default=16.0)
    parser.add_argument("--h2d-gib-per-second", type=float, default=12.46)
    parser.add_argument("--required-tps", type=float, default=15.0)
    parser.add_argument("--required-mean-accepted", type=float, default=24.0)
    parser.add_argument("--node-budget", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.trace, args.resident_fraction_ppm, args.kv_gib,
        args.h2d_gib_per_second, args.required_tps,
        args.required_mean_accepted, args.node_budget,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
