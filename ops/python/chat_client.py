#!/usr/bin/env python3
"""Interactive streaming chat client for a local or SSH-forwarded runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("EXPERT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _ready(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                base_url.rstrip("/") + "/ready", headers=_headers(), method="GET"
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.load(response)
                if payload.get("ready") is True:
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.2)
    raise RuntimeError(f"API did not become ready: {last_error}")


def _events(response: Any) -> Iterator[dict[str, Any]]:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            return
        yield json.loads(line[6:])


@dataclass(frozen=True)
class TurnStats:
    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int
    first_visible_seconds: float
    total_seconds: float
    finish_reason: str


def _print_stats(stats: TurnStats) -> None:
    end_to_end = (stats.output_tokens / stats.total_seconds
                  if stats.total_seconds > 0 else 0.0)
    visible_decode_seconds = max(
        0.0, stats.total_seconds - stats.first_visible_seconds
    )
    post_first = (
        (stats.output_tokens - 1) / visible_decode_seconds
        if stats.reasoning_tokens == 0 and stats.output_tokens > 1 and
        visible_decode_seconds > 1e-6 else None
    )
    decode = (f"{post_first:.2f} tok/s" if post_first is not None else
              "n/a (hidden reasoning precedes visible text)"
              if stats.reasoning_tokens else "n/a")
    print(
        f"[stats] prompt={stats.prompt_tokens} tok | output={stats.output_tokens} tok | "
        f"reasoning={stats.reasoning_tokens} tok | "
        f"first-visible={stats.first_visible_seconds:.3f}s | "
        f"total={stats.total_seconds:.3f}s | "
        f"generated-end-to-end={end_to_end:.2f} tok/s | "
        f"after-first-visible={decode} | "
        f"finish={stats.finish_reason}"
    )


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path, headers=_headers(), method="GET"
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _resolve_model(base_url: str, configured: str) -> str:
    payload = _get_json(base_url, "/v1/models")
    deployed = [str(item.get("id")) for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")]
    if len(deployed) != 1:
        raise RuntimeError(
            f"expected exactly one deployed model, found {len(deployed)}"
        )
    active = deployed[0]
    if configured == "auto":
        return active
    if configured != active:
        print(
            f"Configured model {configured!r} is not active; using {active!r}.",
            file=sys.stderr,
        )
    return active


def _effective_maximum(base_url: str, configured: int) -> int:
    info = _get_json(base_url, "/model-info")
    runtime = info.get("runtime_config")
    if not isinstance(runtime, dict):
        raise RuntimeError("model info has no runtime configuration")
    service_maximum = runtime.get("maximum_new_tokens")
    if (isinstance(service_maximum, bool) or
            not isinstance(service_maximum, int) or service_maximum <= 0):
        raise RuntimeError("model info has no valid output limit")
    return min(configured, service_maximum)


def _print_info(base_url: str) -> None:
    info = _get_json(base_url, "/model-info")
    placement = info.get("worker_placement", {})
    execution = info.get("worker_execution", {})
    runtime = info.get("runtime_config", {})
    worker = info.get("worker_runtime", {})
    kv = info.get("worker_kv", {})
    gib = float(1 << 30)
    print(
        f"model={info.get('model')} build={info.get('build_id')} "
        f"profile={placement.get('profile')} MTP={execution.get('mtp_enabled')}"
    )
    print(
        f"context={runtime.get('max_context')} max_output={runtime.get('maximum_new_tokens')} "
        f"capacity={info.get('worker_capacity')} active={info.get('active_requests')}"
    )
    print(
        f"RAM cache={worker.get('cache_ram_bytes', 0) / gib:.2f} GiB | "
        f"VRAM cache={worker.get('cache_vram_bytes', 0) / gib:.2f} GiB | "
        f"KV pages={kv.get('allocated_pages', 0)}/{kv.get('page_capacity', 0)}"
    )
    print(
        f"storage={info.get('format', {}).get('routed_storage')} "
        f"prefetch={placement.get('prefetch_state')} "
        f"protocol={info.get('worker_protocol')}"
    )


def _chat(base_url: str, model: str, messages: list[dict[str, str]],
          maximum: int, request_timeout: float,
          thinking: str) -> tuple[str, TurnStats]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": maximum,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking != "off"},
    }
    if thinking != "off":
        payload["reasoning_effort"] = "xhigh" if thinking == "high" else thinking
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", body, _headers(),
        method="POST",
    )
    fragments: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = "unknown"
    started = time.perf_counter()
    first_content_at: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            print("assistant> ", end="", flush=True)
            for event in _events(response):
                if event.get("error"):
                    error = event["error"]
                    message = (error.get("message") if isinstance(error, dict)
                               else str(error))
                    raise RuntimeError(f"generation failed: {message}")
                if event.get("type") == "error":
                    raise RuntimeError(
                        f"generation failed: {event.get('message', 'unknown error')}"
                    )
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    finish_reason = str(choices[0]["finish_reason"])
                content = choices[0].get("delta", {}).get("content")
                if content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    fragments.append(content)
                    print(content, end="", flush=True)
            print()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {error.code}: {detail}") from error
    finished = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError("stream completed without a content token")
    completion_details = usage.get("completion_tokens_details") or {}
    return "".join(fragments), TurnStats(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        reasoning_tokens=int(completion_details.get("reasoning_tokens", 0)),
        first_visible_seconds=first_content_at - started,
        total_seconds=finished - started,
        finish_reason=finish_reason,
    )


def _tunnel(remote: str, local_port: int, remote_port: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen([
        "ssh", "-o", "ExitOnForwardFailure=yes", "-N", "-L",
        f"{local_port}:127.0.0.1:{remote_port}", remote,
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ssh", help="open and own an SSH tunnel to user@host")
    parser.add_argument("--local-port", type=int, default=18080)
    parser.add_argument("--remote-port", type=int, default=8080)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=660.0)
    parser.add_argument(
        "--thinking", choices=("off", "low", "medium", "high", "xhigh"),
        default="xhigh",
    )
    parser.add_argument("--show-stats", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    if (args.max_tokens < 1 or args.request_timeout <= 0 or
            not 1 <= args.local_port <= 65535 or
            not 1 <= args.remote_port <= 65535):
        parser.error("token and port values must be positive and valid")

    tunnel: subprocess.Popen[bytes] | None = None
    base_url = args.base_url
    try:
        if args.ssh:
            tunnel = _tunnel(args.ssh, args.local_port, args.remote_port)
            base_url = f"http://127.0.0.1:{args.local_port}"
        _ready(base_url, args.ready_timeout)
        model = _resolve_model(base_url, args.model)
        maximum = _effective_maximum(base_url, args.max_tokens)
        print(f"Connected to {model} at {base_url}")
        print("Commands: /help, /info, /stats, /clear, quit, exit. Ctrl+C closes.")
        messages: list[dict[str, str]] = []
        last_stats: TurnStats | None = None
        while True:
            prompt = input("you> ").strip()
            command = prompt.lower()
            if command in {"quit", "exit", "/quit", "/exit"}:
                return 0
            if command == "/help":
                print("/info runtime details | /stats last turn | /clear history | quit/exit")
                continue
            if command == "/clear":
                messages.clear()
                last_stats = None
                print("conversation cleared")
                continue
            if command == "/stats":
                print("no completed turn" if last_stats is None else "last turn:")
                if last_stats is not None:
                    _print_stats(last_stats)
                continue
            if command == "/info":
                try:
                    _print_info(base_url)
                except (OSError, RuntimeError, urllib.error.URLError) as error:
                    print(f"cannot read model info: {error}", file=sys.stderr)
                continue
            if not prompt:
                continue
            messages.append({"role": "user", "content": prompt})
            try:
                answer, last_stats = _chat(
                    base_url, model, messages, maximum,
                    args.request_timeout, args.thinking,
                )
            except RuntimeError as error:
                messages.pop()
                print(error, file=sys.stderr)
                continue
            messages.append({"role": "assistant", "content": answer})
            if args.show_stats:
                _print_stats(last_stats)
    except KeyboardInterrupt:
        print("\nbye")
        return 130
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()


if __name__ == "__main__":
    raise SystemExit(main())
