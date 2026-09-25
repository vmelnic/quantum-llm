"""Offline contract tests for the Qwen speed benchmark recorder."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops" / "python"))
import qwen_speed_benchmark as bench  # noqa: E402


class QwenSpeedBenchmarkTests(unittest.TestCase):
    def test_artifact_requires_matching_completion_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "qwen3.8-27b-abliterated-fp4-activation-v3"
            artifact.mkdir()
            manifest = b'{"schema": 1}\n'
            (artifact / "manifest.json").write_bytes(manifest)
            marker = {"manifest_file_sha256": hashlib.sha256(manifest).hexdigest()}
            (artifact / "COMPLETED").write_text(json.dumps(marker), encoding="utf-8")
            resolved, digest = bench._validate_artifact(root, artifact.name)
            self.assertEqual(resolved, artifact)
            self.assertEqual(digest, marker["manifest_file_sha256"])
            (artifact / "manifest.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                bench._validate_artifact(root, artifact.name)

    def test_provider_and_http_rates_are_distinct(self) -> None:
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 11,
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
        }
        telemetry = {
            "prefill_tokens": 100, "generated_tokens": 11,
            "wall_seconds": 4.0, "prefill_wall_seconds": 1.0,
            "ttft_seconds": 1.5, "provider_accepted_drafts": 3,
            "provider_exact_calls": 7,
        }
        case = bench._case_metrics(response, 4.4, telemetry)
        self.assertEqual(case["decode_seconds"], 2.5)
        self.assertEqual(case["decode_tokens_per_second"], 4.0)
        self.assertEqual(case["end_to_end_tokens_per_second"], 2.5)
        self.assertEqual(case["visible_tokens"], 7)
        self.assertEqual(case["finish_reason"], "stop")

    def test_one_request_per_profile_uses_fixed_english_prompt(self) -> None:
        path = (Path(__file__).resolve().parents[1] / "ops" / "benchmarks" /
                "qwen27b_english_startups_prompt.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], "qwen27b-english-startups-speed-v1")
        self.assertTrue(document["question"].isascii())
        self.assertIn("10 original AI startup ideas", document["question"])
        tree = ast.parse(Path(bench.__file__).read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_run")
        def request_calls(node: ast.AST) -> list[ast.Call]:
            return [call for call in ast.walk(node) if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_request"]
        self.assertEqual(len(request_calls(run)), 1)
        self.assertFalse(any(request_calls(node) for node in ast.walk(run)
                             if isinstance(node, (ast.For, ast.While, ast.AsyncFor))))


if __name__ == "__main__":
    unittest.main()
