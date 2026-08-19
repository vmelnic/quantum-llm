import json
import tempfile
import unittest
from pathlib import Path

from ops.python.deepseek_route_oracle import (
    analyze,
    load_trace,
    simulate_belady,
    simulate_lru,
)


class DeepSeekRouteOracleTests(unittest.TestCase):
    def test_belady_is_a_strict_oracle_over_lru(self) -> None:
        batches = tuple((record,) for record in (1, 2, 3, 1, 2, 4, 1, 2, 3, 4))
        lru = simulate_lru(batches, 3)
        belady = simulate_belady(batches, 3)
        self.assertEqual((lru.hits, lru.misses), (4, 6))
        self.assertEqual((belady.hits, belady.misses), (5, 5))

    def test_simulators_keep_the_current_route_batch_pinned(self) -> None:
        batches = ((1, 2), (2, 3), (1, 3))
        for simulation in (
            simulate_lru(batches, 2),
            simulate_belady(batches, 2),
        ):
            self.assertEqual(simulation.accesses, 6)
            self.assertEqual(simulation.hits + simulation.misses, 6)

    def test_trace_parser_deduplicates_pair_rows_per_layer(self) -> None:
        item = {
            "schema_version": 1,
            "request_id": 7,
            "record_bytes": 100,
            "device_bytes": 80,
            "ram_capacity_records": 20,
            "vram_capacity_records": 10,
            "initial_ram_records": [9, 8],
            "initial_vram_records": [9],
            "captured_steps": 1,
            "dropped_steps": 0,
            "steps": [{
                "position": 3,
                "rows": 2,
                "batches": [[0, 1, 2, 3, 4, 5, 0, 1, 6, 7, 8, 9]],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            trace = load_trace(path)
        self.assertEqual(trace.batches, ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),))
        self.assertEqual(trace.initial_ram_records, (9, 8))
        result = analyze(trace, 20, 10)
        self.assertEqual(result["route_accesses"], 10)
        self.assertEqual(result["unique_records"], 10)

    def test_truncated_trace_is_rejected_unless_explicitly_allowed(self) -> None:
        item = {
            "schema_version": 1,
            "request_id": 8,
            "record_bytes": 100,
            "device_bytes": 80,
            "ram_capacity_records": 20,
            "vram_capacity_records": 10,
            "initial_ram_records": [],
            "initial_vram_records": [],
            "captured_steps": 1,
            "dropped_steps": 2,
            "steps": [{
                "position": 3,
                "rows": 1,
                "batches": [[0, 1, 2, 3, 4, 5]],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dropped 2 steps"):
                load_trace(path)
            trace = load_trace(path, allow_truncated=True)
        self.assertEqual(trace.dropped_steps, 2)


if __name__ == "__main__":
    unittest.main()
