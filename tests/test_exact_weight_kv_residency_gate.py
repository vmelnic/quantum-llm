import json
import tempfile
import unittest
from pathlib import Path

from ops.python.exact_weight_kv_residency_gate import evaluate, np


@unittest.skipUnless(np is not None, "exact weight profiler requires NumPy")
class ExactWeightKvResidencyGateTest(unittest.TestCase):
    def make_artifact(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pack = bytes([0x00] * 32 + [0x11] * 32 + [127] * 4)
        (root / "dense.qpack").write_bytes(pack)
        manifest = {
            "model_program": {"path": "runtime-model.tsv"},
            "packs": [{"name": "dense.qpack"}],
            "tensors": [
                {
                    "name": "embedding", "pack": "dense.qpack", "offset": 0,
                    "stored_dtype": "FP4_E2M1",
                    "sections": {
                        "data": {"offset": 0, "bytes": 32},
                        "scales": {"offset": 64, "bytes": 2},
                    },
                },
                {
                    "name": "weight", "pack": "dense.qpack", "offset": 32,
                    "stored_dtype": "FP4_E2M1",
                    "sections": {
                        "data": {"offset": 0, "bytes": 32},
                        "scales": {"offset": 34, "bytes": 2},
                    },
                },
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "runtime-model.tsv").write_text(
            "\n".join((
                "expert-runtime-model-v1",
                "operation\t0\t-\tembedding.lookup.fp4-block32.v1\t1\t-\t0",
                "operation_tensor\t0\tweight\tembedding",
                "operation\t1\t0\tffn.swiglu.dense.fp4-block32.v1\t1\t-\t0",
                "operation_tensor\t1\tweight\tweight",
            )) + "\n", encoding="utf-8",
        )
        profile = root / "kv.json"
        profile.write_text(json.dumps({
            "format": "fp16-kv-lossless-profile-v1",
            "raw_fp16_bytes": 1024,
            "ideal_entropy_payload_bytes": 512,
        }), encoding="utf-8")
        return root, profile

    def test_uses_program_capability_to_exclude_sparse_embedding(self) -> None:
        root, profile = self.make_artifact()
        result = evaluate(root, profile, device_gib=1.0, chunk_mib=1)
        self.assertEqual(result["selection"]["sparse_nonresident_tensors"],
                         ["embedding"])
        self.assertEqual(result["target_only"]["raw_weight_allocation_bytes"], 34)
        self.assertEqual(result["selection"]["target_tensor_count"], 1)

    def test_constant_fp4_payload_has_zero_ideal_symbol_entropy(self) -> None:
        root, profile = self.make_artifact()
        result = evaluate(root, profile, device_gib=1.0, chunk_mib=1)
        tensor = result["tensor_profiles"]["weight"]
        self.assertEqual(tensor["sections"]["data"]["ideal_payload_bits"], 0.0)
        self.assertTrue(result["gate_pass"])


if __name__ == "__main__":
    unittest.main()
