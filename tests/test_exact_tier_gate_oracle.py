import tempfile
import unittest
from pathlib import Path

from ops.python.exact_tier_gate_oracle import evaluate, read_trace


class ExactTierGateOracleTest(unittest.TestCase):
    def write_trace(self, ranks: dict[int, list[int]]) -> Path:
        temporary = tempfile.NamedTemporaryFile("w", delete=False, newline="")
        with temporary as stream:
            stream.write(
                "position\tfraction_ppm\tretained_first_token\t"
                "exact_token\tapprox_token\trank\n"
            )
            for fraction, values in ranks.items():
                for offset, rank in enumerate(values):
                    stream.write(
                        f"{100 + offset}\t{fraction}\t50\t{10 + offset}\t"
                        f"{20 + offset}\t{rank}\n"
                    )
        self.addCleanup(Path(temporary.name).unlink)
        return Path(temporary.name)

    def test_oracle_accepts_a_long_low_rank_path(self) -> None:
        path = self.write_trace({250_000: [1] * 64, 375_000: [2] * 64})
        result = evaluate(path, horizon=32, node_budget=128,
                          required_mean=24.0)
        self.assertTrue(result["necessary_bound_pass"])
        fractions = result["fractions"]
        self.assertEqual(fractions[0]["optimistic_oracle"][
            "mean_accepted_depth"], 32.0)
        self.assertEqual(fractions[1]["rank_p95"], 2)

    def test_oracle_rejects_ranks_outside_the_tree_budget(self) -> None:
        path = self.write_trace({375_000: [129] * 8})
        result = evaluate(path, horizon=32, node_budget=128,
                          required_mean=24.0)
        self.assertFalse(result["necessary_bound_pass"])
        oracle = result["fractions"][0]["optimistic_oracle"]
        self.assertEqual(oracle["mean_accepted_depth"], 0.0)
        self.assertEqual(oracle["mean_output_tokens_per_verify"], 1.0)

    def test_fractions_must_share_one_exact_path(self) -> None:
        path = self.write_trace({250_000: [1, 1], 375_000: [1]})
        with self.assertRaisesRegex(ValueError, "same exact path"):
            read_trace(path)


if __name__ == "__main__":
    unittest.main()
