from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ops.python.populated_context_gate import (
    metric_delta,
    parse_prometheus,
    read_token_ids,
)


class PopulatedContextGateTests(unittest.TestCase):
    def test_prometheus_parser_accepts_only_unlabelled_finite_scalars(self) -> None:
        parsed = parse_prometheus(
            "# TYPE one gauge\n"
            "one 7\n"
            "labelled{slot=\"0\"} 3\n"
            "bad nope\n"
            "infinite +Inf\n"
        )
        self.assertEqual(parsed, {"one": 7.0})

    def test_metric_delta_keeps_exact_integer_counters(self) -> None:
        self.assertEqual(
            metric_delta({"one": 4.0}, {"one": 9.0}, ("one", "missing")),
            {"one": 5},
        )

    def test_token_file_is_strict_uint32_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompt.tokens"
            path.write_text("1,2,4294967295\n", encoding="ascii")
            self.assertEqual(read_token_ids(path), [1, 2, 0xFFFFFFFF])
            self.assertEqual(read_token_ids(path, 2), [1, 2])
            self.assertEqual(
                read_token_ids(path, 2, "suffix"), [2, 0xFFFFFFFF]
            )
            with self.assertRaisesRegex(ValueError, "outside the available"):
                read_token_ids(path, 4)
            with self.assertRaisesRegex(ValueError, "unknown prompt token window"):
                read_token_ids(path, 2, "middle")
            path.write_text("1,-1\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "outside uint32"):
                read_token_ids(path)


if __name__ == "__main__":
    unittest.main()
