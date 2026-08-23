from __future__ import annotations

import unittest
from argparse import Namespace
from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory

from ops.python.behavior_quality_gate import (
    _compare,
    _reference_tool_calls,
    _validate,
)


class BehaviorQualityGateTests(unittest.TestCase):
    def test_reference_tool_call_parser_and_validator(self) -> None:
        raw = (
            "reasoning</think>\n<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nChisinau\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        calls = _reference_tool_calls(raw)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        case = {
            "expected": {
                "kind": "tool_call",
                "name": "get_weather",
                "arguments": {"city": "Chisinau"},
            }
        }
        valid, _ = _validate(case, {"raw_text": raw}, reference=True)
        self.assertTrue(valid)

    def test_objective_text_and_json_validators(self) -> None:
        exact = {"expected": {"kind": "exact_text", "value": "703"}}
        self.assertTrue(_validate(
            exact, {"content": " 703\n"}, reference=False,
        )[0])
        structured = {
            "expected": {"kind": "json_equal", "value": [0, 4, 10]}
        }
        self.assertTrue(_validate(
            structured, {"content": "[0, 4, 10]"}, reference=False,
        )[0])
        self.assertFalse(_validate(
            structured, {"content": "[0, 10, 4]"}, reference=False,
        )[0])

    def test_comparison_rejects_prompt_tokenization_drift(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.json"
            service = root / "service.json"
            reference = root / "reference.json"
            output = root / "output.json"
            corpus.write_text(
                '{"cases":[{"id":"x","expected":'
                '{"kind":"exact_text","value":"ok"}}]}',
                encoding="utf-8",
            )
            service.write_text(
                '{"backend":"service","cases":[{"id":"x",'
                '"content":"ok","input_tokens":11}]}',
                encoding="utf-8",
            )
            reference.write_text(
                '{"backend":"reference","source_revision":"r",'
                '"cases":[{"id":"x","content":"ok",'
                '"input_tokens":10}]}',
                encoding="utf-8",
            )
            result = _compare(Namespace(
                corpus=corpus, service=service, reference=reference,
                output=output,
            ))
            self.assertEqual(result, 1)
            document = loads(output.read_text(encoding="utf-8"))
            self.assertFalse(document["valid"])
            self.assertFalse(document["cases"][0]["input_tokens_match"])


if __name__ == "__main__":
    unittest.main()
