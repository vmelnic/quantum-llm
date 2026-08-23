#!/usr/bin/env python3
"""Capture and validate a bounded behavioral corpus across model backends."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _publish(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _service_capture(args: argparse.Namespace) -> int:
    corpus = _load(args.corpus)
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("behavior corpus has no cases")
    key = os.environ.get(args.api_key_environment, "")
    if not key:
        raise RuntimeError(f"{args.api_key_environment} is not configured")
    results = []
    for case in cases:
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": case["messages"],
            "temperature": 0,
            "max_tokens": int(case.get("maximum_new_tokens", 128)),
            "reasoning_effort": case.get("reasoning_effort", "xhigh"),
            "chat_template_kwargs": {
                "enable_thinking": bool(case.get("enable_thinking", True)),
                "preserve_thinking": True,
            },
        }
        if case.get("tools"):
            payload["tools"] = case["tools"]
            payload["tool_choice"] = "auto"
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.load(response)
        message = body["choices"][0]["message"]
        results.append({
            "id": case["id"],
            "input_tokens": body.get("usage", {}).get("prompt_tokens"),
            "output_tokens": body.get("usage", {}).get("completion_tokens"),
            "content": message.get("content") or "",
            "reasoning_content": (
                message.get("reasoning_content")
                or message.get("reasoning") or ""
            ),
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": body["choices"][0].get("finish_reason"),
        })
    document = {
        "schema_version": 1,
        "backend": "quantum-llm-fp4-service",
        "model": args.model,
        "cases": results,
    }
    _publish(args.output, document)
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0


def _reference_tool_calls(raw: str) -> list[dict[str, Any]]:
    calls = []
    for block in re.findall(
            r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
            raw, re.DOTALL):
        name, body = block
        arguments = {
            key.strip(): value.strip()
            for key, value in re.findall(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body,
                re.DOTALL,
            )
        }
        calls.append({
            "type": "function",
            "function": {"name": name.strip(), "arguments": arguments},
        })
    return calls


def _normalized_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function")
    if not isinstance(function, dict):
        return "", {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return str(function.get("name", "")), arguments


def _validate(case: dict[str, Any], result: dict[str, Any], *,
              reference: bool) -> tuple[bool, str]:
    expected = case.get("expected")
    if not isinstance(expected, dict):
        return False, "case has no expected predicate"
    kind = expected.get("kind")
    content = str(result.get("content", "")).strip()
    if kind == "exact_text":
        wanted = str(expected.get("value", "")).strip()
        return content == wanted, f"expected exact text {wanted!r}"
    if kind == "json_equal":
        try:
            actual = json.loads(content)
        except json.JSONDecodeError as error:
            return False, f"content is not JSON: {error}"
        wanted = expected.get("value")
        return actual == wanted, f"expected JSON {wanted!r}"
    if kind == "contains_all":
        wanted = expected.get("values")
        if not isinstance(wanted, list):
            return False, "contains_all values are invalid"
        missing = [str(item) for item in wanted if str(item) not in content]
        return not missing, f"missing substrings: {missing}"
    if kind == "tool_call":
        calls = (
            _reference_tool_calls(str(result.get("raw_text", "")))
            if reference else result.get("tool_calls", [])
        )
        if not isinstance(calls, list) or not calls:
            return False, "no tool call"
        name, arguments = _normalized_tool_call(calls[0])
        wanted_name = expected.get("name")
        wanted_arguments = expected.get("arguments")
        ok = name == wanted_name and arguments == wanted_arguments
        return ok, f"tool={name!r} arguments={arguments!r}"
    return False, f"unsupported expected predicate: {kind}"


def _compare(args: argparse.Namespace) -> int:
    corpus = _load(args.corpus)
    service = _load(args.service)
    reference = _load(args.reference)
    cases = {case["id"]: case for case in corpus.get("cases", [])}
    service_cases = {case["id"]: case for case in service.get("cases", [])}
    reference_cases = {case["id"]: case for case in reference.get("cases", [])}
    if set(cases) != set(service_cases) or set(cases) != set(reference_cases):
        raise ValueError("corpus and capture case IDs differ")
    results = []
    for case_id, case in cases.items():
        service_ok, service_detail = _validate(
            case, service_cases[case_id], reference=False,
        )
        reference_ok, reference_detail = _validate(
            case, reference_cases[case_id], reference=True,
        )
        service_input_tokens = service_cases[case_id].get("input_tokens")
        reference_input_tokens = reference_cases[case_id].get("input_tokens")
        input_tokens_match = (
            isinstance(service_input_tokens, int)
            and service_input_tokens == reference_input_tokens
        )
        results.append({
            "id": case_id,
            "service_pass": service_ok,
            "official_reference_pass": reference_ok,
            "input_tokens_match": input_tokens_match,
            "pass": service_ok and reference_ok and input_tokens_match,
            "service_detail": service_detail,
            "official_reference_detail": reference_detail,
            "service_input_tokens": service_input_tokens,
            "official_reference_input_tokens": reference_input_tokens,
        })
    document = {
        "schema_version": 1,
        "valid": all(item["pass"] for item in results),
        "service_backend": service.get("backend"),
        "reference_backend": reference.get("backend"),
        "source_revision": reference.get("source_revision"),
        "cases": results,
    }
    _publish(args.output, document)
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0 if document["valid"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    service = commands.add_parser("capture-service")
    service.add_argument("--corpus", type=Path, required=True)
    service.add_argument("--output", type=Path, required=True)
    service.add_argument("--base-url", required=True)
    service.add_argument("--model", required=True)
    service.add_argument("--api-key-environment", default="EXPERT_API_KEY")
    service.add_argument("--timeout", type=int, default=600)
    compare = commands.add_parser("compare")
    compare.add_argument("--corpus", type=Path, required=True)
    compare.add_argument("--service", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "capture-service":
        return _service_capture(args)
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
