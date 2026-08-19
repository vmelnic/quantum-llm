#!/usr/bin/env python3
"""Offline cache-policy oracle for bounded DeepSeek route traces."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


@dataclasses.dataclass(frozen=True)
class Trace:
    batches: tuple[tuple[int, ...], ...]
    record_bytes: int
    device_bytes: int
    ram_capacity_records: int
    vram_capacity_records: int
    initial_ram_records: tuple[int, ...]
    initial_vram_records: tuple[int, ...]
    captured_steps: int
    dropped_steps: int


@dataclasses.dataclass(frozen=True)
class Simulation:
    policy: str
    capacity_records: int
    accesses: int
    hits: int
    misses: int

    def as_dict(self, bytes_per_record: int) -> dict[str, int | float | str]:
        return {
            "policy": self.policy,
            "capacity_records": self.capacity_records,
            "accesses": self.accesses,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / self.accesses if self.accesses else 0.0,
            "miss_bytes": self.misses * bytes_per_record,
        }


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_trace(path: Path, allow_truncated: bool = False) -> Trace:
    batches: list[tuple[int, ...]] = []
    metadata: tuple[int, int, int, int] | None = None
    initial_records: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    captured_steps = 0
    dropped_steps = 0
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on trace line {line_number}: {error}"
                ) from error
            if item.get("schema_version") != 1:
                raise ValueError(
                    f"unsupported schema on trace line {line_number}"
                )
            current = (
                _positive_int(item.get("record_bytes"), "record_bytes"),
                _positive_int(item.get("device_bytes"), "device_bytes"),
                _positive_int(
                    item.get("ram_capacity_records"), "ram_capacity_records"
                ),
                _positive_int(
                    item.get("vram_capacity_records"),
                    "vram_capacity_records",
                ),
            )
            if metadata is None:
                metadata = current
            elif metadata != current:
                raise ValueError("trace lines disagree on cache geometry")
            current_initial: list[tuple[int, ...]] = []
            for field, capacity in (
                ("initial_ram_records", current[2]),
                ("initial_vram_records", current[3]),
            ):
                values = item.get(field, [])
                if not isinstance(values, list) or any(
                    not isinstance(record, int)
                    or isinstance(record, bool)
                    or record < 0
                    for record in values
                ):
                    raise ValueError(f"{field} is invalid on line {line_number}")
                ordered = tuple(dict.fromkeys(values))
                if len(ordered) != len(values) or len(ordered) > capacity:
                    raise ValueError(f"{field} exceeds its exact capacity")
                current_initial.append(ordered)
            pair = (current_initial[0], current_initial[1])
            if initial_records is None:
                initial_records = pair
            elif initial_records != pair:
                raise ValueError("trace lines disagree on initial warm records")
            steps = item.get("steps")
            if not isinstance(steps, list):
                raise ValueError(f"steps must be a list on line {line_number}")
            declared_steps = item.get("captured_steps")
            if declared_steps != len(steps):
                raise ValueError(
                    f"captured_steps mismatch on line {line_number}"
                )
            captured_steps += len(steps)
            dropped = item.get("dropped_steps", 0)
            if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped < 0:
                raise ValueError(
                    f"dropped_steps is invalid on line {line_number}"
                )
            dropped_steps += dropped
            for step in steps:
                rows = step.get("rows")
                if rows not in (1, 2):
                    raise ValueError(f"invalid route row count on line {line_number}")
                step_batches = step.get("batches")
                if not isinstance(step_batches, list) or not step_batches:
                    raise ValueError(f"invalid batches on line {line_number}")
                for batch in step_batches:
                    if not isinstance(batch, list) or len(batch) != rows * 6:
                        raise ValueError(
                            f"invalid route batch on line {line_number}"
                        )
                    if any(
                        not isinstance(record, int)
                        or isinstance(record, bool)
                        or record < 0
                        for record in batch
                    ):
                        raise ValueError(
                            f"invalid record id on line {line_number}"
                        )
                    # Pair verification may select the same expert in both
                    # rows. The cache acquires that record once while the whole
                    # layer route is pinned, so retain first-seen order only.
                    batches.append(tuple(dict.fromkeys(batch)))
    if metadata is None:
        raise ValueError("route trace is empty")
    if dropped_steps and not allow_truncated:
        raise ValueError(
            f"route trace dropped {dropped_steps} steps; rerun with a larger "
            "bound or pass --allow-truncated"
        )
    assert initial_records is not None
    return Trace(
        tuple(batches), *metadata, *initial_records,
        captured_steps, dropped_steps
    )


def _validate_capacity(
    batches: Sequence[Sequence[int]], capacity: int
) -> int:
    if capacity <= 0:
        raise ValueError("cache capacity must be positive")
    maximum_batch = max((len(batch) for batch in batches), default=0)
    if maximum_batch > capacity:
        raise ValueError(
            f"cache capacity {capacity} cannot pin a route of {maximum_batch}"
        )
    return sum(len(batch) for batch in batches)


def simulate_lru(
    batches: Sequence[Sequence[int]], capacity: int,
    initial_cache: Sequence[int] = (),
) -> Simulation:
    accesses = _validate_capacity(batches, capacity)
    cache: set[int] = set()
    last_touch: dict[int, int] = {}
    eviction_heap: list[tuple[int, int]] = []
    clock = 0
    hits = 0
    initial = tuple(dict.fromkeys(initial_cache))
    if len(initial) > capacity:
        raise ValueError("initial LRU cache exceeds capacity")

    def touch(record: int) -> None:
        nonlocal clock
        clock += 1
        last_touch[record] = clock
        heapq.heappush(eviction_heap, (clock, record))

    def evict(protected: set[int]) -> None:
        skipped: list[tuple[int, int]] = []
        while eviction_heap:
            timestamp, record = heapq.heappop(eviction_heap)
            if record not in cache or last_touch.get(record) != timestamp:
                continue
            if record in protected:
                skipped.append((timestamp, record))
                continue
            cache.remove(record)
            last_touch.pop(record, None)
            for item in skipped:
                heapq.heappush(eviction_heap, item)
            return
        raise RuntimeError("LRU oracle could not evict an unpinned record")

    for record in initial:
        cache.add(record)
        touch(record)

    for raw_batch in batches:
        batch = tuple(dict.fromkeys(raw_batch))
        protected = set(batch)
        hits += sum(record in cache for record in batch)
        for record in batch:
            if record in cache:
                touch(record)
        for record in batch:
            if record in cache:
                continue
            while len(cache) >= capacity:
                evict(protected)
            cache.add(record)
            touch(record)
    return Simulation("lru", capacity, accesses, hits, accesses - hits)


def simulate_belady(
    batches: Sequence[Sequence[int]], capacity: int,
    initial_cache: Sequence[int] = (),
) -> Simulation:
    accesses = _validate_capacity(batches, capacity)
    normalized = tuple(tuple(dict.fromkeys(batch)) for batch in batches)
    future: dict[int, collections.deque[int]] = collections.defaultdict(
        collections.deque
    )
    for batch_index, batch in enumerate(normalized):
        for record in batch:
            future[record].append(batch_index)

    cache: set[int] = set()
    next_use: dict[int, int | float] = {}
    # Negative next-use turns heapq into a farthest-next-use heap.
    eviction_heap: list[tuple[float, int]] = []
    hits = 0
    initial = tuple(dict.fromkeys(initial_cache))
    if len(initial) > capacity:
        raise ValueError("initial Belady cache exceeds capacity")

    def update(record: int) -> None:
        value: int | float = future[record][0] if future[record] else math.inf
        next_use[record] = value
        heapq.heappush(eviction_heap, (-value, record))

    def evict(protected: set[int]) -> None:
        skipped: list[tuple[float, int]] = []
        while eviction_heap:
            priority, record = heapq.heappop(eviction_heap)
            if record not in cache or -priority != next_use.get(record):
                continue
            if record in protected:
                skipped.append((priority, record))
                continue
            cache.remove(record)
            next_use.pop(record, None)
            for item in skipped:
                heapq.heappush(eviction_heap, item)
            return
        raise RuntimeError("Belady oracle could not evict an unpinned record")

    for record in initial:
        cache.add(record)
        update(record)

    for batch_index, batch in enumerate(normalized):
        protected = set(batch)
        hits += sum(record in cache for record in batch)
        for record in batch:
            positions = future[record]
            if not positions or positions[0] != batch_index:
                raise RuntimeError("Belady future index is inconsistent")
            positions.popleft()
            if record in cache:
                update(record)
        for record in batch:
            if record in cache:
                continue
            while len(cache) >= capacity:
                evict(protected)
            cache.add(record)
            update(record)
    return Simulation("belady", capacity, accesses, hits, accesses - hits)


def analyze(trace: Trace, ram_records: int, vram_records: int) -> dict[str, object]:
    unique_records = len({record for batch in trace.batches for record in batch})

    def tier(
        name: str, capacity: int, bytes_per_record: int,
        initial: Sequence[int],
    ) -> dict[str, object]:
        lru = simulate_lru(trace.batches, capacity, initial)
        belady = simulate_belady(trace.batches, capacity, initial)
        return {
            "tier": name,
            "capacity_records": capacity,
            "initial_records": len(initial),
            "lru": lru.as_dict(bytes_per_record),
            "belady": belady.as_dict(bytes_per_record),
            "policy_avoidable_misses": lru.misses - belady.misses,
            "policy_avoidable_bytes":
                (lru.misses - belady.misses) * bytes_per_record,
        }

    return {
        "schema_version": 1,
        "captured_steps": trace.captured_steps,
        "dropped_steps": trace.dropped_steps,
        "route_batches": len(trace.batches),
        "route_accesses": sum(len(batch) for batch in trace.batches),
        "unique_records": unique_records,
        "ram": tier(
            "ram", ram_records, trace.record_bytes,
            trace.initial_ram_records,
        ),
        "vram": tier(
            "vram", vram_records, trace.device_bytes,
            trace.initial_vram_records,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--ram-records", type=int)
    parser.add_argument("--vram-records", type=int)
    parser.add_argument("--allow-truncated", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        trace = load_trace(args.trace, args.allow_truncated)
        ram_records = args.ram_records or trace.ram_capacity_records
        vram_records = args.vram_records or trace.vram_capacity_records
        result = analyze(trace, ram_records, vram_records)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
