#!/usr/bin/env python3
"""Evaluate the optimistic exact-path bound for an exact-tier gate trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    position: int
    fraction_ppm: int
    retained_first_token: int
    exact_token: int
    approx_token: int
    rank: int


def read_trace(path: Path) -> dict[int, list[Record]]:
    grouped: dict[int, list[Record]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "position", "fraction_ppm", "retained_first_token",
            "exact_token", "approx_token", "rank",
        }
        if set(reader.fieldnames or ()) != required:
            raise ValueError("exact-tier trace header is invalid")
        for row in reader:
            record = Record(**{name: int(row[name]) for name in required})
            if (record.position < 0 or
                    not 0 < record.fraction_ppm < 1_000_000):
                raise ValueError("trace contains an invalid position/fraction")
            if record.retained_first_token < 0 or record.rank < 1:
                raise ValueError("trace contains an invalid retained offset/rank")
            grouped[record.fraction_ppm].append(record)
    if not grouped:
        raise ValueError("exact-tier trace is empty")
    expected_positions: list[int] | None = None
    expected_tokens: list[int] | None = None
    for fraction, records in grouped.items():
        records.sort(key=lambda item: item.position)
        positions = [item.position for item in records]
        tokens = [item.exact_token for item in records]
        if len(set(positions)) != len(positions):
            raise ValueError(f"fraction {fraction} repeats a position")
        if any(right != left + 1 for left, right in zip(positions, positions[1:])):
            raise ValueError(f"fraction {fraction} positions are not contiguous")
        if expected_positions is None:
            expected_positions, expected_tokens = positions, tokens
        elif positions != expected_positions or tokens != expected_tokens:
            raise ValueError("fractions do not describe the same exact path")
    return dict(grouped)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def optimistic_blocks(ranks: list[int], horizon: int,
                      node_budget: int) -> dict[str, object]:
    accepted_depths: list[int] = []
    output_tokens: list[int] = []
    nodes_per_cycle: list[int] = []
    index = 0
    while index < len(ranks):
        nodes = 0
        depth = 0
        while depth < horizon and index + depth < len(ranks):
            next_nodes = ranks[index + depth]
            if nodes + next_nodes > node_budget:
                break
            nodes += next_nodes
            depth += 1
        accepted_depths.append(depth)
        # An exact verifier can still emit its root fallback when the draft
        # path misses immediately. Keep this distinct from accepted depth.
        emitted = max(1, depth)
        output_tokens.append(emitted)
        nodes_per_cycle.append(nodes)
        index += emitted
    return {
        "cycles": len(accepted_depths),
        "mean_accepted_depth": sum(accepted_depths) / len(accepted_depths),
        "mean_output_tokens_per_verify": sum(output_tokens) / len(output_tokens),
        "minimum_accepted_depth": min(accepted_depths),
        "maximum_accepted_depth": max(accepted_depths),
        "mean_oracle_nodes": sum(nodes_per_cycle) / len(nodes_per_cycle),
        "accepted_depths": accepted_depths,
    }


def evaluate(path: Path, horizon: int, node_budget: int,
             required_mean: float) -> dict[str, object]:
    if horizon < 1 or node_budget < 1 or required_mean <= 0:
        raise ValueError("gate bounds must be positive")
    grouped = read_trace(path)
    results: list[dict[str, object]] = []
    for fraction, records in sorted(grouped.items()):
        ranks = [record.rank for record in records]
        blocks = optimistic_blocks(ranks, horizon, node_budget)
        mean_depth = float(blocks["mean_accepted_depth"])
        results.append({
            "fraction_ppm": fraction,
            "records": len(records),
            "first_position": records[0].position,
            "last_position": records[-1].position,
            "top1_rate": sum(rank == 1 for rank in ranks) / len(ranks),
            "rank_p50": percentile(ranks, 0.50),
            "rank_p95": percentile(ranks, 0.95),
            "rank_max": max(ranks),
            "optimistic_oracle": blocks,
            "necessary_bound_pass": mean_depth >= required_mean,
        })
    return {
        "format": "exact-tier-gate-oracle-v1",
        "trace": str(path),
        "horizon": horizon,
        "node_budget": node_budget,
        "required_mean_accepted_depth": required_mean,
        "interpretation": (
            "This is an oracle-guided upper bound. Failure rejects the bounded "
            "tree configuration; success still requires a real proposer/tree."
        ),
        "fractions": results,
        "necessary_bound_pass": any(
            bool(result["necessary_bound_pass"]) for result in results
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--node-budget", type=int, default=128)
    parser.add_argument("--required-mean", type=float, default=24.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.trace, args.horizon, args.node_budget,
                      args.required_mean)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
