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
            with urllib.request.urlopen(
                    base_url.rstrip("/") + "/ready", timeout=2) as response:
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


def _chat(base_url: str, model: str, messages: list[dict[str, str]],
          maximum: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_completion_tokens": maximum,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", body, _headers(),
        method="POST",
    )
    fragments: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            print("assistant> ", end="", flush=True)
            for event in _events(response):
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    fragments.append(content)
                    print(content, end="", flush=True)
            print()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {error.code}: {detail}") from error
    return "".join(fragments)


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
    args = parser.parse_args()
    if args.max_tokens < 1 or not 1 <= args.local_port <= 65535 or not 1 <= args.remote_port <= 65535:
        parser.error("token and port values must be positive and valid")

    tunnel: subprocess.Popen[bytes] | None = None
    base_url = args.base_url
    try:
        if args.ssh:
            tunnel = _tunnel(args.ssh, args.local_port, args.remote_port)
            base_url = f"http://127.0.0.1:{args.local_port}"
        _ready(base_url, args.ready_timeout)
        print(f"Connected to {args.model} at {base_url}")
        print("Type quit or exit to close. Ctrl+C also closes the client.")
        messages: list[dict[str, str]] = []
        while True:
            prompt = input("you> ").strip()
            if prompt.lower() in {"quit", "exit"}:
                return 0
            if not prompt:
                continue
            messages.append({"role": "user", "content": prompt})
            try:
                answer = _chat(base_url, args.model, messages,
                               args.max_tokens)
            except RuntimeError as error:
                messages.pop()
                print(error, file=sys.stderr)
                continue
            messages.append({"role": "assistant", "content": answer})
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
