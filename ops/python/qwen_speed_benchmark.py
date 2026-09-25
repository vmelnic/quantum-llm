#!/usr/bin/env python3
"""Run one English xhigh request against one fresh Qwen profile.

This is a benchmark launcher, not a service profile or model alias. It refuses
an occupied port, validates the artifact marker, captures raw responses and
provider request telemetry, and stops only the process tree it created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


MODEL_ID = "qwen3.8-27b-abliterated-fp4"
ARTIFACT_PATTERN = re.compile(r"qwen3\.8-27b-abliterated-fp4(?:-[a-z0-9-]+)?\Z")
KV_OPTIONS = {
    "q4-f16-per-head",
    "fp4-e2m1-ue8m0-block32-key-outlier1",
    "fp16",
}
SEED = 314159
MAX_COMPLETION_TOKENS = 32768  # Runaway guard, not a reasoning target.


def _publish_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _publish_text(path: Path, value: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


def _validate_artifact(model_root: Path, name: str) -> tuple[Path, str]:
    if not ARTIFACT_PATTERN.fullmatch(name):
        raise ValueError("artifact must be a Qwen3.8-27B Abliterated FP4 name")
    artifact = model_root / name
    marker_path = artifact / "COMPLETED"
    manifest_path = artifact / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"artifact is not complete: {artifact}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = marker.get("manifest_file_sha256")
    actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not isinstance(expected, str) or expected.lower() != actual:
        raise ValueError(f"artifact manifest hash does not match COMPLETED: {name}")
    return artifact, actual


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
        conn.settimeout(0.5)
        return conn.connect_ex(("127.0.0.1", port)) == 0


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("EXPERT_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _get_json(port: int, path: str, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", headers=_headers(), method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"{path} returned a non-object")
    return value


def _wait_ready(process: subprocess.Popen[bytes], port: int) -> dict[str, Any]:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"benchmark server exited with {process.returncode}")
        try:
            if _get_json(port, "/ready").get("ready") is True:
                return _get_json(port, "/model-info")
        except (OSError, urllib.error.HTTPError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("benchmark server did not become ready")


def _request(port: int, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers=_headers(),
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=86_430) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
        ) from error
    elapsed = time.perf_counter() - started
    if not isinstance(result, dict):
        raise ValueError("chat response was not a JSON object")
    return result, elapsed


def _telemetry_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "request_telemetry":
            events.append(event)
    return events


def _wait_telemetry(path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        events = _telemetry_events(path)
        if len(events) == 1:
            return events[0]
        if len(events) > 1:
            raise RuntimeError("unexpected extra request telemetry")
        time.sleep(0.2)
    raise RuntimeError("missing request telemetry")


def _case_metrics(response: dict[str, Any], http_wall: float,
                  telemetry: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") or {}
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("expected exactly one response choice")
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reasoning_tokens = int(
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    )
    provider_wall = float(telemetry["wall_seconds"])
    prefill = float(telemetry["prefill_wall_seconds"])
    ttft = float(telemetry["ttft_seconds"])
    decode = provider_wall - ttft
    if (prompt_tokens < 1 or completion_tokens < 2 or
            int(telemetry["prefill_tokens"]) != prompt_tokens or
            int(telemetry["generated_tokens"]) != completion_tokens or
            not 0 < prefill <= ttft < provider_wall or decode <= 0 or
            http_wall <= 0):
        raise ValueError("inconsistent response/telemetry")
    finish_reason = choices[0].get("finish_reason")
    content = (choices[0].get("message") or {}).get("content") or ""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_tokens": completion_tokens - reasoning_tokens,
        "finish_reason": finish_reason,
        "prefill_seconds": prefill,
        "ttft_seconds": ttft,
        "decode_seconds": decode,
        "provider_wall_seconds": provider_wall,
        "http_end_to_end_seconds": http_wall,
        "prefill_tokens_per_second": prompt_tokens / prefill,
        "decode_tokens_per_second": (completion_tokens - 1) / decode,
        "end_to_end_tokens_per_second": completion_tokens / http_wall,
        "visible_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "request_id": telemetry.get("request_id"),
        "accepted_drafts": int(telemetry.get("provider_accepted_drafts", 0)),
        "exact_calls": int(telemetry.get("provider_exact_calls", 0)),
    }


def _markdown(summary: dict[str, Any]) -> str:
    profile = summary["profile"]
    case = summary["metrics"]
    lines = [
        f"#### English xhigh speed round `{profile['run_id']}`",
        "",
        f"Artifact: `{profile['artifact']}`; target KV: `{profile['kv']}`; "
        "MTP-4 enabled; one RTX 3090; one English request; "
        f"prompt SHA-256 `{profile['prompt_sha256']}`; seed {SEED}; "
        "temperature 1.0, top-p 0.95, top-k 20, min-p 0, "
        "presence penalty 0.5. Fresh service and no retained sessions.",
        "",
        "| Prompt | Completion | Reasoning | Prefill s | TTFT s | "
        "Decode tok/s | HTTP end-to-end s | End-to-end tok/s | Finish |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| {case['prompt_tokens']} | {case['completion_tokens']} | "
        f"{case['reasoning_tokens']} | {case['prefill_seconds']:.3f} | "
        f"{case['ttft_seconds']:.3f} | "
        f"{case['decode_tokens_per_second']:.2f} | "
        f"{case['http_end_to_end_seconds']:.3f} | "
        f"{case['end_to_end_tokens_per_second']:.2f} | "
        f"{case['finish_reason']} |",
        "",
    ]
    return "\n".join(lines)


def _gpu_ordinal() -> tuple[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    matches = []
    for line in result.stdout.splitlines():
        index, _, name = line.partition(",")
        if "RTX 3090" in name:
            matches.append((index.strip(), name.strip()))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one RTX 3090, found {len(matches)}")
    return matches[0]


def _run(args: argparse.Namespace) -> None:
    if os.name != "nt":
        raise RuntimeError("run this benchmark on the RTX 3090 Windows host")
    if args.kv not in KV_OPTIONS or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run_id):
        raise ValueError("invalid KV or round ID")
    artifact, manifest_sha = _validate_artifact(args.model_root, args.artifact)
    if not args.runner.is_file():
        raise FileNotFoundError(f"runner is missing: {args.runner}")
    if _port_open(args.port):
        raise RuntimeError(f"port {args.port} is already occupied; refusing to stop it")
    prompt_document = json.loads(args.prompt.read_text(encoding="utf-8"))
    question = prompt_document.get("question")
    if (prompt_document.get("schema") != "qwen27b-english-startups-speed-v1" or
            not isinstance(question, str) or not question.isascii() or not question):
        raise ValueError("invalid fixed English prompt")
    args.output.mkdir(parents=True, exist_ok=False)
    ordinal, gpu_name = _gpu_ordinal()
    log_path = args.output / "server.jsonl"
    startup_output = args.output / "startup.log"
    service_script = args.repo_root / "ops/windows/Start-ExpertServer.ps1"
    command = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(service_script), "-Container", str(artifact),
        "-Runner", str(args.runner), "-Python", sys.executable,
        "-ModelId", MODEL_ID, "-HostAddress", "127.0.0.1",
        "-Port", str(args.port), "-MaximumContext", "262144",
        "-MaximumNewTokens", str(MAX_COMPLETION_TOKENS),
        "-WorkerCapacity", "1", "-WorkerRamCacheGiB", "48",
        "-WorkerVramCacheGiB", "12", "-WorkerKvCacheMiB", "5136",
        "-WorkerKvPageTokens", "256", "-WorkerKvCacheDtype", args.kv,
        "-DisableSessionRetention", "-GenerationTimeoutSeconds", "86400",
        "-StartupTimeoutSeconds", "600", "-BuildId", args.run_id,
        "-LogFile", str(log_path),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ordinal
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": question}],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"enable_thinking": True},
        "speculative_decoding": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": SEED,
    }
    profile = {
        "run_id": args.run_id, "artifact": args.artifact, "kv": args.kv,
        "gpu": gpu_name, "gpu_ordinal": ordinal,
        "manifest_file_sha256": manifest_sha,
        "prompt_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "request": payload, "requests_per_profile": 1,
    }
    _publish_json(args.output / "profile.json", profile)
    process: subprocess.Popen[bytes] | None = None
    try:
        with startup_output.open("wb") as startup_log:
            startup_started = time.perf_counter()
            process = subprocess.Popen(
                command, stdout=startup_log, stderr=subprocess.STDOUT,
                env=environment, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            info = _wait_ready(process, args.port)
            profile["startup_seconds"] = time.perf_counter() - startup_started
            _publish_json(args.output / "profile.json", profile)
            if (info.get("model") != MODEL_ID or
                    (info.get("worker_kv") or {}).get("dtype") != args.kv or
                    not (info.get("worker_execution") or {}).get("mtp_enabled") or
                    info.get("manifest_content_sha256") !=
                    json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
                    ["integrity"]["content_sha256"]):
                raise RuntimeError("service contract differs from requested profile")
            _publish_json(args.output / "model-info.json", info)
            _publish_json(args.output / "request.json", payload)
            response, http_wall = _request(args.port, payload)
            _publish_json(args.output / "response.json", response)
            event = _wait_telemetry(log_path)
            case = _case_metrics(response, http_wall, event)
            summary = {"schema": "qwen-speed-round-v2", "profile": profile,
                       "metrics": case}
            _publish_json(args.output / "summary.json", summary)
            print(json.dumps({"run_id": args.run_id,
                              "decode_tps": case["decode_tokens_per_second"],
                              "end_to_end_tps": case["end_to_end_tokens_per_second"],
                              "finish": case["finish_reason"]}), flush=True)
            if case["finish_reason"] != "stop" or case["visible_tokens"] <= 0:
                raise RuntimeError("request did not finish normally")
            _publish_text(args.output / "round.md", _markdown(summary))
    finally:
        if process is not None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, text=True, check=False,
            )
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        deadline = time.monotonic() + 30
        while _port_open(args.port) and time.monotonic() < deadline:
            time.sleep(0.5)
        port_free = not _port_open(args.port)
        gpu_memory_mib: int | None = None
        try:
            gpu_result = subprocess.run(
                ["nvidia-smi", "--id", ordinal,
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            gpu_memory_mib = int(gpu_result.stdout.strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
        _publish_json(args.output / "cleanup.json", {
            "server_pid": process.pid if process is not None else None,
            "port_free": port_free,
            "gpu_memory_mib": gpu_memory_mib,
        })
        if not port_free:
            raise RuntimeError("benchmark port remained occupied after cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--kv", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
