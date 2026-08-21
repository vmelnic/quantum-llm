import json
import tempfile
import unittest
from pathlib import Path

from ops.python.exact_kv_residency_gate import evaluate


class ExactKvResidencyGateTest(unittest.TestCase):
    def profile(self, bits: float) -> Path:
        raw_bytes = 16 * (1 << 30)
        payload = {
            "format": "fp16-kv-lossless-profile-v1",
            "best_predictor_h0_bits_per_value": bits,
            "raw_fp16_bytes": raw_bytes,
            "ideal_entropy_payload_bytes": int(raw_bytes * bits / 16),
        }
        temporary = tempfile.NamedTemporaryFile("w", delete=False)
        with temporary as stream:
            json.dump(payload, stream)
        path = Path(temporary.name)
        self.addCleanup(path.unlink)
        return path

    def test_rejects_entropy_that_cannot_fit_with_hot_weights(self) -> None:
        result = evaluate(self.profile(13.0), 24.0, 13 * (1 << 30))
        self.assertFalse(result["gate_pass"])
        self.assertEqual(result["maximum_zero_overhead_bits_per_value"], 11.0)

    def test_admits_only_capacity_before_executable_codec_work(self) -> None:
        result = evaluate(self.profile(10.0), 24.0, 13 * (1 << 30))
        self.assertTrue(result["gate_pass"])


if __name__ == "__main__":
    unittest.main()
