#!/usr/bin/env python3
"""Run a populated token-id prompt through the real streaming service path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_METRICS = (
    "expert_service_generated_tokens_total",
    "expert_worker_exact_decode_accepted_tokens",
    "expert_worker_exact_decode_calls",
    "expert_worker_exact_decode_positions",
    "expert_worker_program_step_ns",
    "expert_worker_program_steps",
    "expert_worker_provider_accepted_drafts",
    "expert_worker_provider_exact_calls",
    "expert_worker_provider_exact_sync_batches",
    "expert_worker_provider_exact_sync_tokens",
    "expert_worker_provider_gpu_ffn_ns",
    "expert_worker_provider_gpu_full_attention_ns",
    "expert_worker_provider_gpu_head_ns",
    "expert_worker_provider_gpu_mtp_ns",
    "expert_worker_provider_gpu_recurrent_attention_ns",
    "expert_worker_provider_prefill_tokens",
    "expert_worker_provider_program_sequence_tiles",
    "expert_worker_provider_program_sequence_tokens",
    "expert_worker_provider_program_steps",
    "expert_worker_provider_sampling_gpu_calls",
    "expert_worker_provider_sampling_host_calls",
    "expert_worker_provider_sampling_logit_transfer_bytes",
    "expert_worker_useful_tokens",
)


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("EXPERT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get(base_url: str, path: str, timeout: float = 10.0) -> bytes:
    request = urllib.request.Request(
        base_url.rstrip("/") + path, headers=_headers(), method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse the service's unlabelled scalar Prometheus exposition."""
    result: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or "{" in fields[0]:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(value):
            result[fields[0]] = value
    return result


def metric_delta(before: dict[str, float], after: dict[str, float],
                 names: tuple[str, ...] = DEFAULT_METRICS
                 ) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for name in names:
        if name not in before or name not in after:
            continue
        value = after[name] - before[name]
        result[name] = int(value) if value.is_integer() else value
    return result


def read_token_ids(
    path: Path, limit: int | None = None, window: str = "prefix"
) -> list[int]:
    encoded = path.read_text(encoding="ascii").strip()
    if not encoded:
        raise ValueError("prompt token file is empty")
    tokens = [int(item) for item in encoded.split(",")]
    if any(token < 0 or token > 0xFFFFFFFF for token in tokens):
        raise ValueError("prompt token is outside uint32")
    if limit is None:
        return tokens
    if limit < 1 or limit > len(tokens):
        raise ValueError(
            f"prompt token limit {limit} is outside the available {len(tokens)} tokens"
        )
    if window == "prefix":
        return tokens[:limit]
    if window == "suffix":
        return tokens[-limit:]
    raise ValueError(f"unknown prompt token window {window!r}")


def _completion_stream(base_url: str, model: str, tokens: list[int],
                       maximum: int, temperature: float, top_p: float,
                       top_k: int, seed: int, timeout: float
                       ) -> dict[str, Any]:
    body = json.dumps({
        "model": model,
        "prompt": tokens,
        "max_tokens": maximum,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions", body, _headers(),
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    fragments: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(f"service stream failed: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("text")
                if delta:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    fragments.append(str(delta))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"API returned HTTP {error.code}: {detail}"
        ) from error
    finished = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("stream ended without a token or final usage")
    text = "".join(fragments)
    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = finished - first_token_at
    return {
        "request_bytes": len(body),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": completion_tokens,
        "ttft_seconds": first_token_at - started,
        "wall_seconds": finished - started,
        "post_first_seconds": decode_seconds,
        "post_first_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds
            if completion_tokens > 1 and decode_seconds > 0.0 else None
        ),
        "finish_reason": finish_reason,
        "output_utf8_bytes": len(text.encode("utf-8")),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_prefix": text[:512],
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=Path)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument(
        "--prompt-window", choices=("prefix", "suffix"), default="prefix"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=14_400.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.max_tokens < 1 or args.timeout_seconds <= 0.0 or
            not 0.0 <= args.temperature <= 2.0 or
            not 0.0 < args.top_p <= 1.0 or args.top_k < 0):
        parser.error("invalid generation settings")

    tokens = read_token_ids(
        args.prompt_tokens, args.prompt_limit, args.prompt_window
    )
    before = parse_prometheus(
        _get(args.base_url, "/metrics").decode("utf-8")
    )
    info_before = json.loads(_get(args.base_url, "/model-info"))
    completion = _completion_stream(
        args.base_url, args.model, tokens, args.max_tokens,
        args.temperature, args.top_p, args.top_k, args.seed,
        args.timeout_seconds,
    )
    after = parse_prometheus(
        _get(args.base_url, "/metrics").decode("utf-8")
    )
    info_after = json.loads(_get(args.base_url, "/model-info"))
    result = {
        "schema_version": 1,
        "kind": "populated-context-service-gate",
        "model": args.model,
        "build_id": info_after.get("build_id"),
        "prompt_token_file": str(args.prompt_tokens.resolve()),
        "prompt_token_ids_sha256": hashlib.sha256(
            ",".join(str(token) for token in tokens).encode("ascii")
        ).hexdigest(),
        "requested_prompt_tokens": len(tokens),
        "prompt_window": args.prompt_window,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "completion": completion,
        "metric_delta": metric_delta(before, after),
        "worker_before": {
            "execution": info_before.get("worker_execution"),
            "kv": info_before.get("worker_kv"),
        },
        "worker_after": {
            "execution": info_after.get("worker_execution"),
            "kv": info_after.get("worker_kv"),
        },
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2,
                         sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
