#!/usr/bin/env python3
"""Measure real text TTFT and completion latency through the OpenAI API path."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def run_probe(base_url: str, model: str, prompt: str,
              maximum: int) -> dict[str, object]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": maximum,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("EXPERT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", body, headers,
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    fragments: list[str] = []
    usage: dict[str, object] | None = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    fragments.append(content)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"API returned HTTP {error.code}: {error.read().decode()}"
        ) from error
    finished = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError("stream completed without a content token")
    return {
        "ttft_seconds": first_content_at - started,
        "total_seconds": finished - started,
        "usage": usage,
        "text": "".join(fragments),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.rounds < 1:
        parser.error("max-tokens and rounds must be positive")
    results = [run_probe(args.base_url, args.model, args.prompt,
                         args.max_tokens)
               for _ in range(args.rounds)]
    print(json.dumps({
        "schema_version": 1,
        "endpoint": args.base_url,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "runs": results,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
