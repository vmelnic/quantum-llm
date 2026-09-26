#!/usr/bin/env python3
"""One autonomous, sequential Pi long-context run on the Windows 3090 host.

The transcript, raw RPC events, provider logs, and one JSON result per turn are
durable on the host. No network connection to the operator is kept open.
"""

from __future__ import annotations

import argparse
from collections import deque
import ctypes
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/pi256k"))
from scenario import RESEARCH_TURN, prompt_for_turn


CONTEXT = 256_000
TERMINAL_MINIMUM = 255_000
MODEL = "qwen3.8-27b-fp4"
GGUF_MODEL = "qwen3.8-27b-gguf-q4-k-m"


class TargetContextReached(Exception):
    """End after saving the first real Pi call at the near-window context."""


def terminal_context_reached(populated: int) -> bool:
    return TERMINAL_MINIMUM <= populated < CONTEXT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def port_open(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(1)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def gpu() -> tuple[str, int]:
    found = []
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    for row in output.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) == 3 and "RTX 3090" in parts[1]:
            found.append((parts[0], int(parts[2])))
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one RTX 3090, found {len(found)}")
    return found[0]


def check_qwen(root: Path) -> tuple[Path, str]:
    artifact = root / MODEL
    marker = artifact / "COMPLETED"
    manifest = artifact / "manifest.json"
    if not marker.is_file() or not manifest.is_file():
        raise RuntimeError("official FP4 artifact is incomplete")
    expected = json.loads(marker.read_text(encoding="utf-8"))["manifest_file_sha256"]
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if expected.lower() != actual:
        raise RuntimeError("official FP4 manifest hash mismatch")
    return artifact, actual


def check_gguf(root: Path) -> tuple[Path, Path]:
    artifact = root / GGUF_MODEL
    manifest = artifact / "manifest.json"
    marker = artifact / "COMPLETED"
    if not manifest.is_file() or not marker.is_file():
        raise RuntimeError("published GGUF artifact is missing")
    marker_data = json.loads(marker.read_text(encoding="utf-8-sig"))
    if marker_data.get("manifest_file_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise RuntimeError("GGUF manifest integrity failure")
    metadata = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if metadata.get("schema") != "gguf-benchmark-artifact-v1" or metadata.get("revision") != "efbb3b1f70a21d97fd4495240648405f7228554f":
        raise RuntimeError("GGUF manifest is not the pinned benchmark artifact")
    weights = artifact / "Qwen3.8-27B-Q4_K_M.gguf"
    draft = artifact / "mtp-Qwen3.8-27B-Q4_0.gguf"
    expected_sizes = {"Qwen3.8-27B-Q4_K_M.gguf": 18_973_870_528,
                      "mtp-Qwen3.8-27B-Q4_0.gguf": 1_680_271_776}
    for path in (weights, draft):
        if (not path.is_file() or path.stat().st_size != expected_sizes[path.name]
                or path.name not in metadata["files"]):
            raise RuntimeError(f"GGUF file size/manifest failure: {path.name}")
    ram = system_memory()
    if ram["available_bytes"] < weights.stat().st_size + draft.stat().st_size + 2 * 1024**3:
        raise RuntimeError("insufficient free system RAM for GGUF CPU placement")
    return weights, draft


def gguf_required_bytes(weights_bytes: int, draft_bytes: int = 0) -> int:
    # Q4_0: 18 bytes per 32 values. F16 target KV is 16 GiB at 262144.
    kv_lower_bound = 16 * (CONTEXT / 262_144) * (18 / 32 / 2) * (1024**3)
    return int(weights_bytes + draft_bytes + kv_lower_bound + 1024**3)


def system_memory() -> dict[str, int]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_bytes", ctypes.c_ulonglong), ("available_bytes", ctypes.c_ulonglong),
                    ("total_pagefile", ctypes.c_ulonglong), ("available_pagefile", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"total_bytes": int(status.total_bytes),
            "available_bytes": int(status.available_bytes)}


def process_metrics(pid: int) -> dict[str, int] | None:
    """Windows working set and process I/O, not an SSD-only traffic claim."""
    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("page_faults", ctypes.c_ulong),
                    ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                    ("quota_peak_paged", ctypes.c_size_t), ("quota_paged", ctypes.c_size_t),
                    ("quota_peak_nonpaged", ctypes.c_size_t), ("quota_nonpaged", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
                    ("private_usage", ctypes.c_size_t)]
    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "read_operations", "write_operations", "other_operations",
            "read_transfer_bytes", "write_transfer_bytes", "other_transfer_bytes")]
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetProcessIoCounters.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        memory = MemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        io = IoCounters()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
            return None
        if not kernel.GetProcessIoCounters(handle, ctypes.byref(io)):
            return None
        return {"working_set_bytes": int(memory.working_set),
                "peak_working_set_bytes": int(memory.peak_working_set),
                "private_usage_bytes": int(memory.private_usage),
                "page_faults": int(memory.page_faults),
                "read_transfer_bytes": int(io.read_transfer_bytes),
                "write_transfer_bytes": int(io.write_transfer_bytes)}
    finally:
        kernel.CloseHandle(handle)


def parse_llama_placement(log: str) -> dict[str, Any]:
    buffers = []
    for match in re.finditer(
        r"(?m)^.*?\b(CPU(?:_Mapped)?|CUDA\d+)\s+(model|KV) buffer size\s*=\s*([0-9.]+)\s+MiB.*$", log):
        buffers.append({"name": f"{match[1]} {match[2]} buffer size",
                        "mib": float(match[3]), "line": match[0].strip()})
    layers = [{"gpu_layers": int(match[1]), "total_layers": int(match[2])}
              for match in re.finditer(r"offloaded\s+(\d+)/(\d+)\s+layers", log)]
    def size(prefix: str, kind: str) -> float:
        return sum(item["mib"] for item in buffers
                   if item["name"].startswith(prefix) and f" {kind} buffer size" in item["name"])
    cpu_mapped = size("CPU_Mapped", "model")
    cpu = size("CPU ", "model")
    gpu = sum(item["mib"] for item in buffers
              if item["name"].startswith("CUDA") and " model buffer size" in item["name"])
    cpu_kv = size("CPU ", "KV")
    gpu_kv = sum(item["mib"] for item in buffers
                 if item["name"].startswith("CUDA") and " KV buffer size" in item["name"])
    cpu_at_load = cpu_mapped + cpu + cpu_kv > 0 or any(
        item["gpu_layers"] < item["total_layers"] for item in layers)
    return {"buffers": buffers, "layer_records": layers,
            "cpu_mapped_model_buffer_mib": cpu_mapped,
            "cpu_allocated_model_buffer_mib": cpu,
            "gpu_model_buffer_mib": gpu,
            "cpu_kv_buffer_mib": cpu_kv,
            "gpu_kv_buffer_mib": gpu_kv,
            "cpu_placement_observed_at_load": cpu_at_load,
            "cpu_placement_selection_populated_prompt_tokens": 0 if cpu_at_load else None}


def resource_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Observed process/system deltas; neither RAM growth nor reads prove offload."""
    result = {
        "host_available_ram_change_mib":
            (after["ram"]["available_bytes"] - before["ram"]["available_bytes"]) / 1024**2,
        "gpu_used_change_mib": before["gpu_free_mib"] - after["gpu_free_mib"],
    }
    old, new = before.get("process"), after.get("process")
    if old is not None and new is not None:
        for key in ("working_set_bytes", "private_usage_bytes",
                    "read_transfer_bytes", "write_transfer_bytes", "page_faults"):
            result[key + "_change"] = new[key] - old[key]
        result["peak_working_set_bytes_after"] = new["peak_working_set_bytes"]
    return result


def qwen_paging(telemetry: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Exact provider counter deltas, distinct from process-level RAM guesses."""
    if not telemetry:
        return None
    keys = ("cache_read_bytes", "cache_uploaded_bytes", "cache_demand_ram_hits",
            "cache_demand_ssd_misses", "mtp_cache_read_bytes",
            "mtp_cache_uploaded_bytes", "mtp_cache_ram_hits", "mtp_cache_ssd_misses")
    if not any(key in event for event in telemetry for key in keys):
        return None
    totals = {key: sum(int(event.get(key) or 0) for event in telemetry) for key in keys}
    totals["paging_observed"] = any(value > 0 for value in totals.values())
    return totals


def is_live_research_tool(tool: dict[str, Any]) -> bool:
    if tool.get("name") not in ("powershell", "bash"):
        return False
    command = str((tool.get("args") or {}).get("command", ""))
    return (re.search(r"\b(?:Invoke-RestMethod|Invoke-WebRequest|curl(?:\.exe)?)\b",
                      command, re.IGNORECASE) is not None
            and "https://" in command.lower())


def research_succeeded(tool: dict[str, Any]) -> bool:
    return (is_live_research_tool(tool) and not tool.get("error")
            and tool.get("exit_code") in (0, None)
            and re.search(r"https?://\S+", str(tool.get("result_excerpt") or "")) is not None)


def readiness(port: int, process: subprocess.Popen[bytes], backend: str) -> dict[str, Any]:
    endpoint = "/model-info" if backend == "quantum" else "/v1/models"
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{endpoint}", timeout=3) as response:
                data = json.load(response)
            if backend == "quantum":
                if data.get("model") != MODEL or (data.get("worker_kv") or {}).get("dtype") != "q4-f16-per-head" or not (data.get("worker_execution") or {}).get("mtp_enabled") or (data.get("runtime_config") or {}).get("max_context") != CONTEXT:
                    raise RuntimeError("Qwen service format, MTP, or context mismatch")
            if backend == "llama":
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=3) as response:
                    props = json.load(response)
                effective = (props.get("default_generation_settings") or {}).get("n_ctx")
                if effective != CONTEXT:
                    raise RuntimeError(f"llama.cpp changed the canonical context: {effective}")
                return {"models": data, "props": props}
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            time.sleep(1)
    raise RuntimeError("server readiness timed out")


def provider_registry(agent_dir: Path, backend: str, port: int) -> None:
    model = MODEL if backend == "quantum" else GGUF_MODEL
    model_entry = {"id": model, "name": model, "reasoning": True,
                   "thinkingLevelMap": {"xhigh": "xhigh"},
                   "input": ["text"], "contextWindow": CONTEXT,
                   "maxTokens": 32_768,
                   "samplingParams": {"temperature": 1.0, "top_p": 0.95,
                                      "top_k": 20, "min_p": 0.0,
                                      "presence_penalty": 0.5},
                   "cost": {"input": 0, "output": 0,
                            "cacheRead": 0, "cacheWrite": 0}}
    compat = {
        "supportsDeveloperRole": True,
        "supportsReasoningEffort": True,
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens",
    }
    if backend == "quantum":
        compat.update({"thinkingFormat": "chat-template", "chatTemplateKwargs": {
            "enable_thinking": {"$var": "thinking.enabled"},
            "preserve_thinking": True,
            "reasoning_effort": {"$var": "thinking.effort", "omitWhenOff": True},
        }})
    write_json(agent_dir / "models.json", {"providers": {"benchmark": {
        "baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions",
        "apiKey": "benchmark-loopback", "authHeader": True, "compat": compat,
        "models": [model_entry],
    }}})
    write_json(agent_dir / "settings.json", {"defaultProvider": "benchmark",
        "defaultModel": model, "defaultThinkingLevel": "xhigh",
        "compaction": {"enabled": False}, "retry": {"enabled": False,
        "provider": {"timeoutMs": 86400000, "maxRetries": 0, "maxRetryDelayMs": 0}},
        "httpIdleTimeoutMs": 0, "enableInstallTelemetry": False})


class Rpc:
    def __init__(self, command: list[str], env: dict[str, str], cwd: Path, event_file: Path):
        self.stderr = event_file.with_suffix(".stderr.log").open("wb")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr,
            cwd=cwd, env=env, text=True, bufsize=1)
        self.items: queue.Queue[dict[str, Any]] = queue.Queue()
        self.pending: deque[dict[str, Any]] = deque()
        self.event_file = event_file.open("a", encoding="utf-8")
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        self.serial = 0

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "invalid_json", "raw": line}
            self.event_file.write(json.dumps({"received_at": time.time(), "event": event}) + "\n")
            self.event_file.flush()
            self.items.put(event)
        self.items.put({"type": "process_eof"})

    def next(self, timeout: float = 86400) -> dict[str, Any]:
        event = self.items.get(timeout=timeout)
        if event.get("type") == "process_eof":
            raise RuntimeError(f"Pi RPC exited: {self.process.poll()}")
        return event

    def call(self, command: str, **fields: Any) -> dict[str, Any]:
        self.serial += 1
        identifier = str(self.serial)
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"id": identifier, "type": command, **fields}) + "\n")
        self.process.stdin.flush()
        while True:
            event = self.next()
            if event.get("type") == "response" and event.get("id") == identifier:
                if not event.get("success"):
                    raise RuntimeError(f"Pi RPC {command} failed: {event}")
                return event
            self.pending.append(event)

    def turn(self, prompt: str, on_call: Callable[[dict[str, Any], int, float,
                                                  list[dict[str, Any]]], None] | None = None
             ) -> dict[str, Any]:
        started = time.monotonic()
        self.call("prompt", message=prompt)
        tools: dict[str, dict[str, Any]] = {}
        calls = []
        final_text = ""
        request_started = started
        first_delta: float | None = None
        first_visible_delta: float | None = None
        while True:
            if self.pending:
                event = self.pending.popleft()
            else:
                try:
                    event = self.next(timeout=5)
                except queue.Empty:
                    stalled = [tool for tool in tools.values()
                               if "started" in tool and
                               time.monotonic() - tool["started"] > 90]
                    if stalled:
                        raise RuntimeError("Pi tool exceeded the 90-second wall guard: " +
                                           ", ".join(str(tool["name"]) for tool in stalled))
                    continue
            kind = event.get("type")
            if kind == "message_update":
                delta = (event.get("assistantMessageEvent") or {})
                delta_type = delta.get("type")
                if delta_type in ("text_delta", "thinking_delta", "toolcall_delta"):
                    elapsed = time.monotonic() - request_started
                    if first_delta is None:
                        first_delta = elapsed
                    if delta_type == "text_delta" and first_visible_delta is None:
                        first_visible_delta = elapsed
            elif kind == "tool_execution_start":
                tools[event["toolCallId"]] = {"name": event.get("toolName"),
                    "args": event.get("args"), "started": time.monotonic()}
            elif kind == "tool_execution_end":
                tool = tools.get(event.get("toolCallId"))
                if tool:
                    tool["seconds"] = time.monotonic() - tool.pop("started")
                    tool["error"] = bool(event.get("isError"))
                    result = event.get("result") or {}
                    tool["result_excerpt"] = "\n".join(
                        str(block.get("text", "")) for block in result.get("content", [])
                        if block.get("type") == "text")[:4000]
                    details = result.get("details") or {}
                    if isinstance(details, dict):
                        tool["exit_code"] = details.get("exitCode")
                request_started = time.monotonic()
            elif kind == "message_end":
                message = event.get("message") or {}
                if message.get("role") == "assistant":
                    content = message.get("content") or []
                    call = {"usage": message.get("usage") or {},
                            "stop_reason": message.get("stopReason"),
                            "client_ttft_seconds": first_delta,
                            "client_time_to_first_visible_seconds": first_visible_delta,
                            "client_call_wall_seconds": time.monotonic() - request_started,
                            "visible_response": "".join(
                                block.get("text", "") for block in content
                                if block.get("type") == "text"),
                            "tool_call_names": [block.get("name") for block in content
                                                if block.get("type") == "toolCall"]}
                    calls.append(call)
                    final_text = call["visible_response"]
                    if on_call is not None:
                        on_call(call, len(calls) - 1, time.monotonic() - started,
                                list(tools.values()))
                    first_delta = None
                    first_visible_delta = None
            elif kind == "agent_end":
                return {"wall_seconds": time.monotonic() - started,
                        "model_calls": calls, "tools": list(tools.values()),
                        "visible_response": final_text}

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.thread.join(timeout=5)
        self.event_file.close()
        self.stderr.close()


def telemetry_after(path: Path, offset: int, backend: str) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], offset
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    # A request event can be in the middle of a write. Never consume a partial line.
    complete = data.rfind(b"\n") + 1
    if not complete:
        return [], offset
    data = data[:complete]
    next_offset = offset + complete
    events: list[dict[str, Any]] = []
    if backend == "quantum":
        for line in data.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "request_telemetry":
                events.append(event)
    else:
        text = data.decode("utf-8", errors="replace")
        prompts = list(re.finditer(
            r"prompt eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens", text))
        decodes = list(re.finditer(
            r"(?<!prompt )eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) (?:runs|tokens)", text))
        for index, match in enumerate(prompts):
            event = {"prefill_wall_seconds": float(match[1]) / 1000,
                     "prefill_tokens": int(match[2]), "source": "llama-server-log"}
            if index < len(decodes):
                seconds = float(decodes[index][1]) / 1000
                event["decode_seconds"] = seconds
                event["decode_tokens"] = int(decodes[index][2])
                event["decode_tokens_per_second"] = event["decode_tokens"] / max(0.001, seconds)
            events.append(event)
    return events, next_offset


def call_measurement(call: dict[str, Any], event: dict[str, Any], turn: int,
                     call_index: int, turn_elapsed: float, tool_seconds: float,
                     total_elapsed: float) -> dict[str, Any]:
    usage = call.get("usage") or {}
    populated = int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
    if populated <= 0:
        raise RuntimeError(f"Pi call {turn}.{call_index} has no populated prompt usage")
    if "prefill_tokens" not in event:
        raise RuntimeError(f"provider call {turn}.{call_index} has no prefill token count")
    processed = int(event["prefill_tokens"])
    if processed < 0:
        raise RuntimeError(f"provider call {turn}.{call_index} has invalid prefill count")
    event = dict(event)
    event["populated_prompt_tokens"] = populated
    if event.get("source") == "llama-server-log":
        event["reused_prefix_tokens_estimate"] = max(0, populated - processed)
        event["resumed"] = event["reused_prefix_tokens_estimate"] > 8
        event["prefix_classification"] = "inferred_from_prompt_eval"
    else:
        if event.get("resumed") is True:
            event["reused_prefix_tokens"] = max(0, populated - processed)
        event["prefix_classification"] = "provider_exact"
    output = int(usage.get("output") or 0)
    reasoning = usage.get("reasoning")
    reasoning = int(reasoning) if reasoning is not None else None
    ttft = event.get("ttft_seconds")
    wall = event.get("wall_seconds")
    generated = event.get("generated_tokens")
    prefill_seconds = event.get("prefill_wall_seconds")
    prefill_rate = (processed / float(prefill_seconds)
                    if prefill_seconds is not None and float(prefill_seconds) > 0 else None)
    decode_tokens = event.get("decode_tokens", generated)
    decode_seconds = event.get("decode_seconds")
    if decode_seconds is None and wall is not None and ttft is not None:
        decode_seconds = max(0.0, float(wall) - float(ttft))
    decode_rate = event.get("decode_tokens_per_second")
    if decode_rate is None and decode_tokens is not None and decode_seconds is not None:
        decode_rate = int(decode_tokens) / float(decode_seconds) if float(decode_seconds) > 0 else None
    client_wall = call.get("client_call_wall_seconds")
    return {"turn": turn, "call": call_index, "populated_prompt_tokens": populated,
            "completion_tokens": output, "reasoning_tokens": reasoning,
            "visible_tokens": output - reasoning if reasoning is not None else None,
            "stop_reason": call.get("stop_reason"),
            "visible_response": call.get("visible_response"),
            "tool_call_names": call.get("tool_call_names") or [],
            "correctness": "pending_review",
            "client_call_wall_seconds": client_wall,
            "client_completion_tokens_per_second": output / float(client_wall)
                if client_wall is not None and float(client_wall) > 0 else None,
            "client_visible_tokens_per_second": (output - reasoning) / float(client_wall)
                if reasoning is not None and client_wall is not None and float(client_wall) > 0 else None,
            "client_ttft_seconds": call.get("client_ttft_seconds"),
            "client_time_to_first_visible_seconds": call.get("client_time_to_first_visible_seconds"),
            "turn_wall_seconds_so_far": turn_elapsed,
            "tool_wall_seconds_so_far": tool_seconds,
            "end_to_end_wall_seconds_so_far": total_elapsed,
            "provider_wall_seconds": wall,
            "provider_generated_tokens": generated,
            "provider_useful_tokens": event.get("useful_tokens"),
            "provider_prefill_seconds": prefill_seconds,
            "provider_prefill_tokens": processed,
            "provider_prefill_tokens_per_second": prefill_rate,
            "cold_prefill_seconds": prefill_seconds if event.get("resumed") is False else None,
            "reused_prefill_seconds": prefill_seconds if event.get("resumed") is True else None,
            "reused_prefix_tokens": event.get("reused_prefix_tokens"),
            "reused_prefix_tokens_estimate": event.get("reused_prefix_tokens_estimate"),
            "provider_ttft_seconds": event.get("ttft_seconds"),
            "provider_decode_seconds": decode_seconds,
            "provider_decode_tokens": decode_tokens,
            "provider_decode_tokens_per_second": decode_rate,
            "provider_paging": qwen_paging([event]) if event.get("source") != "llama-server-log" else None,
            "provider_telemetry": event}


def enrich(turn: dict[str, Any], telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    usage = [call["usage"] for call in turn["model_calls"]]
    for index, event in enumerate(telemetry):
        if index >= len(usage):
            continue
        full = int(usage[index].get("input") or 0) + int(usage[index].get("cacheRead") or 0)
        event["populated_prompt_tokens"] = full
        processed = int(event.get("prefill_tokens") or 0)
        if event.get("source") == "llama-server-log":
            estimate = max(0, full - processed)
            event["reused_prefix_tokens_estimate"] = estimate
            event["resumed"] = estimate > 8
            event["prefix_classification"] = "inferred_from_prompt_eval"
        elif event.get("resumed") is True:
            event["reused_prefix_tokens"] = max(0, full - processed)
            event["prefix_classification"] = "provider_exact"
        else:
            event["prefix_classification"] = "provider_exact"
    populated = max((int(item.get("input") or 0) + int(item.get("cacheRead") or 0)
                     for item in usage), default=0)
    completion = sum(int(item.get("output") or 0) for item in usage)
    reasoning_values = [item.get("reasoning") for item in usage]
    reasoning = (sum(int(value) for value in reasoning_values)
                 if reasoning_values and all(value is not None for value in reasoning_values) else None)
    tool_seconds = sum(float(tool.get("seconds") or 0) for tool in turn["tools"])
    tests = [tool for tool in turn["tools"] if tool.get("name") in ("bash", "powershell") and
             re.search(r"\b(test|pytest|unittest|ctest|npm\s+test)\b",
                       str((tool.get("args") or {}).get("command", "")), re.IGNORECASE)]
    automated_test_status = ("not_run" if not tests else
        "failed" if any(tool.get("error") or tool.get("exit_code") not in (0, None)
                        for tool in tests) else "passed_reported")
    cold = sum(float(event.get("prefill_wall_seconds") or 0) for event in telemetry
               if event.get("resumed") is False)
    reused = sum(float(event.get("prefill_wall_seconds") or 0) for event in telemetry
                 if event.get("resumed") is True)
    cold_tokens = sum(int(event.get("prefill_tokens") or 0) for event in telemetry
                      if event.get("resumed") is False)
    reused_tokens = sum(int(event.get("prefill_tokens") or 0) for event in telemetry
                        if event.get("resumed") is True)
    decodes = []
    for event in telemetry:
        if event.get("decode_tokens_per_second") is not None:
            decodes.append(float(event["decode_tokens_per_second"]))
        elif event.get("generated_tokens") and event.get("ttft_seconds"):
            decodes.append(int(event["generated_tokens"]) /
                max(0.001, float(event.get("wall_seconds") or 0) - float(event["ttft_seconds"])))
    turn.update({"populated_prompt_tokens": populated, "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "visible_tokens": completion - reasoning if reasoning is not None else None,
        "tool_wall_seconds": tool_seconds, "model_and_overhead_wall_seconds": max(0, turn["wall_seconds"] - tool_seconds),
        "cold_prefill_seconds": cold if telemetry and any(event.get("resumed") is False for event in telemetry) else None,
        "reused_prefill_seconds": reused if telemetry and any(event.get("resumed") is True for event in telemetry) else None,
        "cold_prefill_tokens": cold_tokens if telemetry and any(event.get("resumed") is False for event in telemetry) else None,
        "reused_prefill_tokens": reused_tokens if telemetry and any(event.get("resumed") is True for event in telemetry) else None,
        "reused_prefix_tokens": sum(int(event.get("reused_prefix_tokens") or 0) for event in telemetry)
            if any(event.get("reused_prefix_tokens") is not None for event in telemetry) else None,
        "reused_prefix_tokens_estimate": sum(int(event.get("reused_prefix_tokens_estimate") or 0) for event in telemetry)
            if any(event.get("reused_prefix_tokens_estimate") is not None for event in telemetry) else None,
        "cold_prefill_tokens_per_second": cold_tokens / cold if cold > 0 else None,
        "reused_prefill_tokens_per_second": reused_tokens / reused if reused > 0 else None,
        "provider_prefill_seconds": sum(float(event.get("prefill_wall_seconds") or 0) for event in telemetry) if telemetry else None,
        "provider_ttft_seconds_per_call": [event.get("ttft_seconds") for event in telemetry],
        "client_ttft_seconds_per_call": [call.get("client_ttft_seconds") for call in turn["model_calls"]],
        "client_time_to_first_visible_seconds_per_call":
            [call.get("client_time_to_first_visible_seconds") for call in turn["model_calls"]],
        "decode_tokens_per_second_per_call": decodes,
        "provider_telemetry": telemetry,
        "automated_test_status": automated_test_status,
        "correctness": "pending_review"})
    return turn


def run(args: argparse.Namespace) -> None:
    if os.name != "nt":
        raise RuntimeError("the autonomous benchmark runs only on the Windows 3090 host")
    if port_open(args.port):
        raise RuntimeError(f"port {args.port} is occupied; refusing to stop its owner")
    ordinal, free_mib = gpu()
    if free_mib < 23_000:
        raise RuntimeError(f"RTX 3090 is not idle: {free_mib} MiB free")
    artifact_hash = None
    if args.backend == "quantum":
        artifact, artifact_hash = check_qwen(args.model_root)
    else:
        weights, draft = check_gguf(args.model_root)
    if args.output.exists():
        raise RuntimeError("run output already exists; refusing to overwrite or restart a partial session")
    args.output.mkdir(parents=True)
    workspace = args.output / "workspace"
    workspace.mkdir()
    agent_dir = args.output / "pi-agent"
    agent_dir.mkdir()
    provider_registry(agent_dir, args.backend, args.port)
    binary = args.repo_root / "out/benchmarks/pi256k-tools/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
    if not binary.is_file():
        raise RuntimeError("Pi benchmark-local installation missing; run Prepare-Pi.ps1")
    log_path = args.output / ("server.jsonl" if args.backend == "quantum" else "llama-server.log")
    if args.backend == "quantum":
        runner = args.repo_root / "out/build/windows-msvc-release/runtime/Release/expert-moe-vm-runner.exe"
        python = args.repo_root / ".venv/Scripts/python.exe"
        if not python.is_file():
            python = args.repo_root / "work/venv/server/Scripts/python.exe"
        if not runner.is_file() or not python.is_file():
            raise RuntimeError("Qwen server runner or Python environment missing")
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(args.repo_root / "ops/windows/Start-ExpertServer.ps1"),
            "-Container", str(artifact), "-Runner", str(runner), "-Python", str(python),
            "-ModelId", MODEL, "-HostAddress", "127.0.0.1", "-Port", str(args.port),
            "-MaximumContext", str(CONTEXT), "-MaximumNewTokens", "32768",
            "-WorkerCapacity", "1", "-WorkerRamCacheGiB", "48", "-WorkerVramCacheGiB", "12",
            "-WorkerKvCacheMiB", "5136", "-WorkerKvPageTokens", "256",
            "-WorkerKvCacheDtype", "q4-f16-per-head", "-GenerationTimeoutSeconds", "86400",
            "-StartupTimeoutSeconds", "900", "-LogFile", str(log_path)]
    else:
        binary_server = args.repo_root / "out/llama-v0.5.0/build/bin/Release/llama-server.exe"
        if not binary_server.is_file():
            raise RuntimeError("pinned llama.cpp CUDA server missing; run Build-Llama.ps1")
        command = [str(binary_server), "--model", str(weights), "--alias", GGUF_MODEL,
            "--host", "127.0.0.1", "--port", str(args.port), "--ctx-size", str(CONTEXT),
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
            "--flash-attn", "on", "--fit", "on", "--fit-target", "1024",
            "--fit-ctx", str(CONTEXT), "--parallel", "1", "--jinja", "--cache-prompt",
            "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
            "--presence-penalty", "0.5", "--repeat-penalty", "1.0",
            "--reasoning", "on", "--reasoning-preserve", "--no-context-shift", "--no-warmup",
            "--predict", "32768", "--timeout", "86400", "--spec-type", "draft-mtp",
            "--spec-draft-model", str(draft), "--spec-draft-n-max", "4",
            "--spec-draft-type-k", "q8_0", "--spec-draft-type-v", "q8_0",
            "--verbosity", "4"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ordinal
    environment["PI_CODING_AGENT_DIR"] = str(agent_dir)
    write_json(args.output / "STARTED.json", {"schema": "pi256k-v3", "backend": args.backend,
        "model": MODEL if args.backend == "quantum" else GGUF_MODEL,
        "context": CONTEXT, "thinking": "xhigh", "gpu_ordinal": ordinal,
        "measurement_policy": "every_completed_pi_model_call_at_actual_populated_prompt",
        "terminal_minimum_populated_prompt_tokens": TERMINAL_MINIMUM,
        "artifact_manifest_sha256": artifact_hash, "started_utc": time.time(),
        "gpu_free_mib_before": free_mib, "ram_before": system_memory(),
        "full_gpu_capacity_lower_bound_bytes": gguf_required_bytes(18_973_870_528, 1_680_271_776)
            if args.backend == "llama" else None,
        "command": command, "pi_package": "@earendil-works/pi-coding-agent@0.87.1",
        "scenario": "offline-first conference schedule app"})
    server: subprocess.Popen[bytes] | None = None
    rpc: Rpc | None = None
    completed = []
    last_call: dict[str, Any] | None = None
    terminal_call: dict[str, Any] | None = None
    calls_recorded = 0
    completed_wall_seconds = 0.0
    benchmark_started = time.monotonic()
    failure = None
    placement: dict[str, Any] | None = None
    first_paging_context: int | None = None
    try:
        startup_path = args.output / "startup.log" if args.backend == "quantum" else log_path
        with startup_path.open("wb") as startup:
            server = subprocess.Popen(command, stdout=startup, stderr=subprocess.STDOUT,
                env=environment, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            info = readiness(args.port, server, args.backend)
            write_json(args.output / "model-info.json", info)
            if args.backend == "llama":
                startup.flush()
                log = startup_path.read_text(encoding="utf-8", errors="replace")
                placement = parse_llama_placement(log)
                if not placement["buffers"] or not placement["layer_records"]:
                    raise RuntimeError("llama.cpp did not report CPU/GPU model placement")
                placement["gpu_free_mib_after_load"] = gpu()[1]
                placement["ram_after_load"] = system_memory()
                placement["process_after_load"] = process_metrics(server.pid)
                write_json(args.output / "placement.json", placement)
            pi_command = ["node", str(binary), "--mode", "rpc", "--provider", "benchmark",
                "--model", MODEL if args.backend == "quantum" else GGUF_MODEL,
                "--thinking", "xhigh", "--session", str(args.output / "session.jsonl"),
                "--no-context-files", "--no-extensions", "--tools",
                "read,powershell,edit,write,grep,find,ls",
                "--no-skills", "--no-prompt-templates", "--approve"]
            rpc = Rpc(pi_command, environment, workspace, args.output / "pi-events.jsonl")
            rpc.call("set_auto_compaction", enabled=False)
            state = rpc.call("get_state").get("data") or {}
            if state.get("autoCompactionEnabled") is not False or state.get("thinkingLevel") != "xhigh":
                raise RuntimeError("Pi compaction or thinking level differs from benchmark contract")
            offset = 0
            pending_events: deque[dict[str, Any]] = deque()
            index = 0
            while True:
                question = prompt_for_turn(index)
                process_before = process_metrics(server.pid) if args.backend == "llama" else None
                ram_before = system_memory()
                gpu_before = gpu()[1]
                provider_events: list[dict[str, Any]] = []
                previous_tool_count = 0

                def record_call(call: dict[str, Any], call_index: int,
                                turn_elapsed: float, tools_so_far: list[dict[str, Any]]) -> None:
                    nonlocal offset, first_paging_context, last_call
                    nonlocal calls_recorded, terminal_call, previous_tool_count
                    deadline = time.monotonic() + 10
                    while not pending_events and time.monotonic() < deadline:
                        new_events, offset = telemetry_after(log_path, offset, args.backend)
                        pending_events.extend(new_events)
                        if not pending_events:
                            time.sleep(0.2)
                    if not pending_events:
                        raise RuntimeError(f"missing provider telemetry for Pi call {index}.{call_index}")
                    provider_event = pending_events.popleft()
                    measurement = call_measurement(
                        call, provider_event, index, call_index, turn_elapsed,
                        sum(float(tool.get("seconds") or 0) for tool in tools_so_far),
                        completed_wall_seconds + turn_elapsed)
                    recent_tools = tools_so_far[previous_tool_count:]
                    previous_tool_count = len(tools_so_far)
                    measurement["tools_since_previous_call"] = [
                        {key: tool.get(key) for key in ("name", "seconds", "error", "exit_code")}
                        for tool in recent_tools]
                    measurement["tool_wall_seconds_since_previous_call"] = sum(
                        float(tool.get("seconds") or 0) for tool in recent_tools)
                    provider_events.append(measurement["provider_telemetry"])
                    last_call = {"turn": index, "call": call_index,
                                 "populated_prompt_tokens": measurement["populated_prompt_tokens"],
                                 "call_file": f"calls/{index:04d}-{call_index:04d}.json"}
                    if (measurement["provider_paging"] and
                            measurement["provider_paging"]["paging_observed"] and
                            first_paging_context is None):
                        first_paging_context = measurement["populated_prompt_tokens"]
                    write_json(args.output / "calls" / f"{index:04d}-{call_index:04d}.json",
                               measurement)
                    calls_recorded += 1
                    reached = terminal_context_reached(measurement["populated_prompt_tokens"])
                    if reached:
                        terminal_call = last_call
                    write_json(args.output / "PROGRESS.json", {"schema": "pi256k-v3",
                        "backend": args.backend, "turns": completed,
                        "calls_recorded": calls_recorded,
                        "last_call": last_call,
                        "terminal_call": terminal_call,
                        "first_provider_paging_observed_at_populated_prompt_tokens": first_paging_context,
                        "cpu_placement_observed_at_model_load":
                            placement["cpu_placement_observed_at_load"] if placement else None})
                    if call.get("stop_reason") in ("error", "aborted"):
                        raise RuntimeError(f"model call failed at {index}.{call_index}")
                    if reached:
                        raise TargetContextReached
                    if call.get("stop_reason") == "length":
                        raise RuntimeError(f"model call length-stopped at {index}.{call_index}")

                turn = rpc.turn(question, on_call=record_call)
                if len(provider_events) != len(turn["model_calls"]):
                    raise RuntimeError(f"Pi/provider call count mismatch at turn {index}")
                turn = enrich(turn, provider_events)
                paging = qwen_paging(provider_events) if args.backend == "quantum" else None
                if paging and paging["paging_observed"] and first_paging_context is None:
                    first_paging_context = turn["populated_prompt_tokens"]
                resources_before = {"process": process_before, "ram": ram_before,
                                    "gpu_free_mib": gpu_before}
                resources_after = {
                    "process": process_metrics(server.pid) if args.backend == "llama" else None,
                    "ram": system_memory(), "gpu_free_mib": gpu()[1]}
                research_tools = [tool for tool in turn["tools"] if is_live_research_tool(tool)]
                successful_research = sum(research_succeeded(tool) for tool in research_tools)
                turn.update({"turn": index, "user_prompt": question,
                    "live_research_calls": len(research_tools),
                    "successful_live_research_calls": successful_research,
                    "resources_before": resources_before,
                    "resources_after": resources_after,
                    "resource_deltas_during_turn": resource_deltas(resources_before, resources_after),
                    "provider_paging": paging,
                    "first_provider_paging_observed_at_populated_prompt_tokens": first_paging_context,
                    "placement_at_model_load": None if placement is None else {
                        "cpu_model_mapped_mib": placement["cpu_mapped_model_buffer_mib"],
                        "cpu_model_allocated_mib": placement["cpu_allocated_model_buffer_mib"],
                        "cpu_kv_mib": placement["cpu_kv_buffer_mib"],
                        "gpu_model_mib": placement["gpu_model_buffer_mib"],
                        "gpu_kv_mib": placement["gpu_kv_buffer_mib"],
                        "first_observed_populated_prompt_tokens":
                            placement["cpu_placement_selection_populated_prompt_tokens"]}})
                if index == RESEARCH_TURN and successful_research == 0:
                    raise RuntimeError("built-in Pi tool did not complete a live search on the research turn")
                write_json(args.output / "turns" / f"{index:04d}.json", turn)
                completed.append({"turn": index, "populated_prompt_tokens": turn["populated_prompt_tokens"],
                    "wall_seconds": turn["wall_seconds"],
                    "live_research_calls": turn["live_research_calls"],
                    "successful_live_research_calls": successful_research})
                completed_wall_seconds += turn["wall_seconds"]
                write_json(args.output / "PROGRESS.json", {"schema": "pi256k-v3",
                    "backend": args.backend, "turns": completed,
                    "calls_recorded": calls_recorded,
                    "last_call": last_call,
                    "terminal_call": terminal_call,
                    "last_turn": index, "last_populated_prompt_tokens": turn["populated_prompt_tokens"],
                    "first_provider_paging_observed_at_populated_prompt_tokens": first_paging_context,
                    "cpu_placement_observed_at_model_load":
                        placement["cpu_placement_observed_at_load"] if placement else None})
                if index > 0 and turn["populated_prompt_tokens"] <= completed[-2]["populated_prompt_tokens"]:
                    raise RuntimeError("populated context did not grow")
                index += 1
    except TargetContextReached:
        pass
    except Exception as error:
        failure = str(error)
        raise
    finally:
        if rpc:
            rpc.close()
        if server is not None:
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"],
                capture_output=True, text=True, check=False)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        cleanup_deadline = time.monotonic() + 30
        cleanup_free_mib = None
        while time.monotonic() < cleanup_deadline:
            try:
                _, cleanup_free_mib = gpu()
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            if not port_open(args.port) and cleanup_free_mib is not None and cleanup_free_mib >= free_mib - 512:
                break
            time.sleep(0.5)
        cleaned = not port_open(args.port) and cleanup_free_mib is not None and cleanup_free_mib >= free_mib - 512
        write_json(args.output / "COMPLETE.json", {"schema": "pi256k-v3",
            "backend": args.backend,
            "status": "failed" if failure else ("target_reached" if cleaned else "cleanup_failed"),
            "error": failure, "last_call": last_call, "terminal_call": terminal_call,
            "calls_recorded": calls_recorded,
            "turn_count": len(completed), "port_free": not port_open(args.port),
            "gpu_free_mib_before": free_mib, "gpu_free_mib_after": cleanup_free_mib,
            "gpu_cleanup_passed": cleaned,
            "benchmark_wall_seconds": time.monotonic() - benchmark_started,
            "finished_utc": time.time()})
        if not cleaned and failure is None:
            raise RuntimeError("benchmark service did not release its port/GPU budget")


def main() -> None:
    if sys.argv[1:] == ["--self-test-windows"]:
        if os.name != "nt":
            raise RuntimeError("Windows metrics self-test requires Windows")
        print(json.dumps({"gpu": gpu(), "system_memory": system_memory(),
                          "process_metrics": process_metrics(os.getpid())}))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("quantum", "llama"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
