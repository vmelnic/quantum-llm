import tempfile
import unittest
from pathlib import Path

from ops.python.exact_tier_tree_gate import evaluate, read_tree_trace


class ExactTierTreeGateTest(unittest.TestCase):
    def write_trace(self, accepted: list[int], proposer_ns: int = 10_000_000,
                    nodes: int = 128) -> Path:
        temporary = tempfile.NamedTemporaryFile("w", delete=False, newline="")
        with temporary as stream:
            stream.write(
                "start_position\thorizon\tbeam_width\tnodes\tproposer_ns\t"
                "peak_host_bytes\taccepted_depth\ttermination\n"
            )
            for index, depth in enumerate(accepted):
                termination = "horizon" if depth == 32 else "mismatch"
                stream.write(
                    f"{100 + index * 32}\t32\t4\t{nodes}\t{proposer_ns}\t"
                    f"4194304\t{depth}\t{termination}\n"
                )
        self.addCleanup(Path(temporary.name).unlink)
        return Path(temporary.name)

    def test_real_tree_passes_acceptance_capacity_and_necessary_latency(self) -> None:
        path = self.write_trace([32, 31, 32, 31])
        result = evaluate(path, 250_000, 16.0, 12.46, 15.0, 24.0, 128)
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["mean_accepted_depth"], 31.5)
        self.assertEqual(result["peak_proposer_host_bytes"], 4_194_304)

    def test_tree_rejects_low_acceptance(self) -> None:
        path = self.write_trace([2, 1, 0, 3])
        result = evaluate(path, 250_000, 16.0, 12.46, 15.0, 24.0, 128)
        self.assertFalse(result["gate_pass"])
        self.assertFalse(result["acceptance_pass"])

    def test_tree_rejects_a_slow_proposer(self) -> None:
        path = self.write_trace([32, 32], proposer_ns=2_000_000_000)
        result = evaluate(path, 250_000, 16.0, 12.46, 15.0, 24.0, 128)
        self.assertFalse(result["latency_necessary_bound_pass"])

    def test_full_fp16_transfer_uses_zero_resident_fraction(self) -> None:
        path = self.write_trace([32, 32], proposer_ns=100_000_000)
        result = evaluate(path, 0, 16.0, 12.46, 15.0, 24.0, 128)
        self.assertAlmostEqual(
            result["missing_kv_transfer_seconds_per_cycle"],
            16.0 / 12.46,
        )

    def test_invalid_header_is_rejected(self) -> None:
        path = self.write_trace([32])
        path.write_text("wrong\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "header"):
            read_tree_trace(path)


if __name__ == "__main__":
    unittest.main()
