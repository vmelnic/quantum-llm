#!/usr/bin/env python3
"""Bounded OpenAI-compatible HTTP front-end for the persistent CUDA worker."""

from __future__ import annotations

import argparse
import hmac
import importlib.util
import json
import os
import queue
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import unquote, urlsplit

try:
    from .response_protocols import install_declared_response_protocol
except ImportError:  # Direct script launch from Start-ExpertServer.ps1.
    from response_protocols import install_declared_response_protocol

from transformers import AutoTokenizer


LOG_FILE: Any = None


def _load_deepseek_chat_encoder(snapshot: Path) -> Callable[..., str]:
    path = snapshot / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official DeepSeek encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = getattr(module, "encode_messages", None)
    if not callable(encoder):
        raise RuntimeError(f"official DeepSeek encoder has no encode_messages: {path}")
    return encoder


def log(event: str, **fields: Any) -> None:
    line = json.dumps({"event": event, "time": time.time(), **fields}, separators=(",", ":"))
    # A scheduled task has no consumer for inherited stderr. Duplicating every
    # file log there eventually fills the Windows pipe and blocks an HTTP
    # handler while it still owns the GIL. Foreground mode keeps stderr; service
    # mode writes only to its explicit durable log.
    print(line, file=LOG_FILE if LOG_FILE is not None else sys.stderr,
          flush=True)


class WorkerError(RuntimeError):
    pass


def _worker_response_tokens(payload: Mapping[str, Any], message: str) -> list[int]:
    tokens = payload.get("tokens")
    legacy = payload.get("token")
    if tokens is None and isinstance(legacy, int) and not isinstance(legacy, bool):
        tokens = [legacy]
    if (not isinstance(tokens, list) or not tokens or
            any(not isinstance(token, int) or isinstance(token, bool)
                for token in tokens)):
        raise WorkerError(message)
    return [int(token) for token in tokens]


class RequestError(ValueError):
    def __init__(self, message: str, param: str | None = None,
                 code: str = "invalid_value") -> None:
        super().__init__(message)
        self.param = param
        self.code = code


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int

    @property
    def enabled(self) -> bool:
        return self.temperature > 0.0


@dataclass(frozen=True)
class GenerationRequest:
    endpoint: str
    prompt_ids: list[int]
    cache_prefix_tokens: int
    maximum: int
    stream: bool
    stop: tuple[str, ...]
    include_usage: bool
    instructions: str | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
    reasoning_effort: str = "xhigh"
    enable_thinking: bool = True
    preserve_thinking: bool = True
    sampling: SamplingSettings = SamplingSettings(0.0, 1.0, 0, 0.0, 0)
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | dict[str, Any] = "auto"


@dataclass(frozen=True)
class ToolCall:
    item_id: str
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantOutput:
    text: str
    reasoning: str
    reasoning_complete: bool
    tool_calls: tuple[ToolCall, ...]


class AssistantStreamParser:
    """Translate standard Transformers response events into API text deltas."""

    def __init__(self, application: "Application",
                 request: GenerationRequest) -> None:
        self.parser = application.response_stream_parser(request)

    @staticmethod
    def _deltas(events: list[dict[str, Any]]) -> tuple[str, str]:
        reasoning: list[str] = []
        content: list[str] = []
        for event in events:
            if event.get("type") != "region_chunk" or event.get("dirty"):
                continue
            text = event.get("text")
            if not isinstance(text, str) or not text:
                continue
            field = event.get("field")
            if field in {"thinking", "reasoning", "reasoning_content"}:
                reasoning.append(text)
            elif field == "content":
                content.append(text)
        return "".join(reasoning), "".join(content)

    def feed(self, delta: str) -> tuple[str, str]:
        if self.parser is None:
            return "", delta
        return self._deltas(self.parser.feed(delta))

    def finish(self) -> tuple[str, str]:
        if self.parser is None:
            return "", ""
        _message, events = self.parser.finalize()
        return self._deltas(events)


@dataclass
class Session:
    """A retained worker-side conversation prefix (LRU-managed)."""

    key: int
    tokens: list[int]
    pages: int
    last_used: float


@dataclass
class RequestContext:
    """Admission state owned by one HTTP request while it generates."""

    session: Session | None
    held_pages: int
    retained: bool = False


class StopFilter:
    """Hold possible stop-prefix suffixes so stop strings never leak to clients."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self.stops = stops
        self.pending = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped:
            return ""
        combined = self.pending + text
        match = min(
            (index for stop in self.stops
             if (index := combined.find(stop)) >= 0),
            default=-1,
        )
        if match >= 0:
            self.pending = ""
            self.stopped = True
            return combined[:match]
        held = 0
        for stop in self.stops:
            maximum = min(len(stop) - 1, len(combined))
            for size in range(maximum, 0, -1):
                if combined.endswith(stop[:size]):
                    held = max(held, size)
                    break
        if held:
            result, self.pending = combined[:-held], combined[-held:]
            return result
        self.pending = ""
        return combined

    def finish(self) -> str:
        if self.stopped:
            return ""
        result, self.pending = self.pending, ""
        return result


class IncrementalTextDecoder:
    """Emit only tokenizer text prefixes that cannot be rewritten later."""

    def __init__(self) -> None:
        self.emitted = ""

    def push(self, current: str, final: bool = False) -> str:
        if not current.startswith(self.emitted):
            raise WorkerError("tokenizer changed an already streamed text prefix")
        stable_end = len(current)
        if not final:
            replacement = current.find("\ufffd", len(self.emitted))
            if replacement >= 0:
                stable_end = replacement
        delta = current[len(self.emitted):stable_end]
        self.emitted = current[:stable_end]
        return delta


def _text_content(content: Any, param: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("message content must be text or an array of text parts", param)
    pieces: list[str] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise RequestError("message content parts must be objects", f"{param}.{index}")
        kind = part.get("type")
        if kind not in {"text", "input_text", "output_text"}:
            raise RequestError(
                f"content type {kind!r} is not supported by this text-only model",
                f"{param}.{index}.type", "unsupported_value",
            )
        if not isinstance(part.get("text"), str):
            raise RequestError("text content part requires a string", f"{param}.{index}.text")
        pieces.append(part["text"])
    return "".join(pieces)


class CudaWorker:
    def __init__(self, executable: Path, container: Path, max_context: int,
                 startup_timeout: float, requested_capacity: int,
                 ram_cache_gib: int, vram_cache_gib: int,
                 kv_cache_mib: int, kv_page_tokens: int,
                 placement_profile: str, profile_gpu_phases: bool,
                 prefill_chunk_tokens: int = 0,
                 placement_settle_steps: int | None = None,
                 retain_previous_route: bool | None = None,
                 enable_cpu_hybrid: bool | None = None,
                 route_trace_file: Path | None = None,
                 route_trace_max_steps: int = 4096) -> None:
        command = [
            str(executable), str(container), "--worker",
            f"--max-context={max_context}",
            f"--ram-cache-gib={ram_cache_gib}",
            f"--vram-cache-gib={vram_cache_gib}",
            f"--capacity={requested_capacity}",
            f"--kv-cache-mib={kv_cache_mib}",
            f"--kv-page-tokens={kv_page_tokens}",
            f"--placement-profile={placement_profile}",
        ]
        if prefill_chunk_tokens:
            command.append(f"--prefill-chunk-limit={prefill_chunk_tokens}")
        if placement_settle_steps is not None:
            command.append(
                f"--placement-settle-steps={placement_settle_steps}"
            )
        if profile_gpu_phases:
            command.append("--profile-gpu-phases")
        if retain_previous_route is False:
            command.append("--no-retain-previous-route")
        if enable_cpu_hybrid:
            command.append("--cpu-hybrid")
        if route_trace_file is not None:
            command.extend((
                f"--route-trace-file={route_trace_file}",
                f"--route-trace-max-steps={route_trace_max_steps}",
            ))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self._stderr_thread = threading.Thread(target=self._copy_stderr, daemon=True)
        self._stderr_thread.start()
        startup: queue.Queue[Any] = queue.Queue(maxsize=1)
        def wait_ready() -> None:
            try:
                startup.put(self._read())
            except Exception as error:
                startup.put(error)
        threading.Thread(target=wait_ready, daemon=True).start()
        try:
            response = startup.get(timeout=startup_timeout)
        except queue.Empty as error:
            self.process.kill()
            raise WorkerError("CUDA worker startup timed out") from error
        if isinstance(response, Exception) or response.get("type") != "ready":
            self.process.kill()
            raise WorkerError("CUDA worker did not become ready") from (
                response if isinstance(response, Exception) else None
            )
        self.protocol = int(response.get("protocol", 1))
        self.capacity = int(response.get("capacity", 1))
        self.architecture_id = str(response.get("architecture_id", ""))
        self.vocab_size = int(response.get("vocab_size", 0))
        self.model_max_context_tokens = int(
            response.get("max_context_tokens", 0)
        )
        self.routed_layers = int(response.get("routed_layers", 0))
        self.experts_per_layer = int(response.get("experts_per_layer", 0))
        self.route_width = int(response.get("route_width", 0))
        self.expert_encoding = str(response.get("expert_encoding", ""))
        raw_capabilities = response.get("operation_capabilities", [])
        self.operation_capabilities = (
            tuple(raw_capabilities) if isinstance(raw_capabilities, list)
            else ()
        )
        self.prefill_mode = str(response.get("prefill_mode", ""))
        self.prefill_chunk_tokens = int(response.get("prefill_chunk_tokens", 0))
        self.session_retention = bool(response.get("session_retention", False))
        self.request_stream_mode = str(
            response.get("request_stream_mode", "default")
        )
        self.gpu_phase_timing = bool(response.get("gpu_phase_timing", False))
        self.sampling_supported = bool(
            response.get("sampling_supported", False)
        )
        self.mtp_resource_available = bool(
            response.get("mtp_resource_available", False)
        )
        self.mtp_runtime_ready = bool(response.get("mtp_runtime_ready", False))
        self.mtp_enabled = bool(response.get("mtp_enabled", False))
        self.retain_previous_route = bool(
            response.get("retain_previous_route", True)
        )
        self.cpu_hybrid_enabled = bool(
            response.get("cpu_hybrid_enabled", False)
        )
        if self.mtp_runtime_ready and not self.mtp_resource_available:
            self.process.kill()
            raise WorkerError("MTP runtime cannot be ready without resources")
        if (retain_previous_route is not None and
                self.retain_previous_route != retain_previous_route):
            self.process.kill()
            raise WorkerError(
                "CUDA worker retained-route mode does not match the request"
            )
        if (enable_cpu_hybrid is not None and
                self.cpu_hybrid_enabled != enable_cpu_hybrid):
            self.process.kill()
            raise WorkerError(
                "CUDA worker CPU-hybrid mode does not match the request"
            )
        self.rope_mode = str(response.get("rope_mode", "per_step_upload"))
        self.kv_dtype = str(response.get("kv_dtype", ""))
        self.kv_allocation = str(response.get("kv_allocation", ""))
        self.kv_page_tokens = int(response.get("kv_page_tokens", 0))
        self.kv_page_bytes = int(response.get("kv_page_bytes", 0))
        self.kv_page_capacity = int(response.get("kv_page_capacity", 0))
        self.placement_profile = str(response.get("placement_profile", ""))
        self.placement_mode = str(response.get("placement_mode", "budgeted"))
        self.ram_cache_bytes = int(response.get("ram_cache_bytes", 0))
        self.vram_cache_bytes = int(response.get("vram_cache_bytes", 0))
        self.placement_prefetch_enabled = bool(
            response.get("placement_prefetch_enabled", False)
        )
        prefetch_state = response.get("placement_prefetch_state")
        # Protocol-4 Qwen workers published the effective boolean before the
        # explicit observing/ready state was added. Preserve that ABI while
        # newer workers report the richer state directly.
        if prefetch_state is None:
            prefetch_state = (
                "ready" if self.placement_prefetch_enabled else "disabled"
            )
        self.placement_prefetch_state = str(prefetch_state)
        self.placement_minimum_observations = int(
            response.get("placement_minimum_observations", 0)
        )
        expected_observations = 1 if placement_profile == "latency" else 2
        placement_invalid = (
            (
                self.placement_mode == "budgeted" and
                (
                    self.placement_profile != placement_profile or
                    self.ram_cache_bytes != ram_cache_gib << 30 or
                    self.vram_cache_bytes != vram_cache_gib << 30 or
                    self.placement_prefetch_state not in
                        {"disabled", "observing", "ready"} or
                    self.placement_prefetch_enabled !=
                        (self.placement_prefetch_state == "ready") or
                    (placement_profile == "capacity" and
                     self.placement_prefetch_state != "disabled") or
                    self.placement_minimum_observations !=
                        expected_observations
                )
            ) or
            (
                self.placement_mode == "resident" and
                (
                    self.placement_profile != "resident" or
                    self.ram_cache_bytes != 0 or
                    self.vram_cache_bytes <= 0 or
                    self.placement_prefetch_enabled or
                    self.placement_prefetch_state != "disabled" or
                    self.placement_minimum_observations != 0
                )
            ) or
            self.placement_mode not in {"budgeted", "resident"}
        )
        routed_descriptor_present = (
            self.routed_layers != 0 or self.experts_per_layer != 0 or
            self.route_width != 0 or bool(self.expert_encoding)
        )
        routed_descriptor_invalid = (
            (
                routed_descriptor_present and
                (
                    self.routed_layers <= 0 or self.experts_per_layer <= 0 or
                    not 1 <= self.route_width <= self.experts_per_layer or
                    not self.expert_encoding
                )
            ) or
            (
                not routed_descriptor_present and
                (
                    self.routed_layers != 0 or self.experts_per_layer != 0 or
                    self.route_width != 0 or bool(self.expert_encoding)
                )
            )
        )
        descriptor_invalid = self.protocol >= 6 and (
            not self.architecture_id or self.vocab_size <= 0 or
            self.model_max_context_tokens < max_context or
            routed_descriptor_invalid or not self.operation_capabilities or
            len(set(self.operation_capabilities)) !=
                len(self.operation_capabilities) or
            any(not isinstance(capability, str) or not capability
                for capability in self.operation_capabilities)
        )
        if (self.protocol < 4 or descriptor_invalid or
                self.capacity != requested_capacity or
                self.prefill_mode not in {
                    "causal_chunked",
                    "causal_layer_major",
                    "causal_sequential",
                } or
                not 1 <= self.prefill_chunk_tokens <= max_context or
                (prefill_chunk_tokens and
                 self.prefill_chunk_tokens > prefill_chunk_tokens) or
                self.kv_dtype not in {
                    "fp16", "bf16", "fp32",
                    "fp4-e2m1-ue8m0-block32",
                } or
                self.kv_allocation not in {"paged_on_demand", "preallocated"} or
                self.kv_page_tokens != kv_page_tokens or
                self.kv_page_bytes <= 0 or self.kv_page_capacity <= 0 or
                placement_invalid):
            self.process.kill()
            raise WorkerError(
                "CUDA worker does not support requested runtime contract"
            )
        self.active_ids: set[int] = set()
        self.command_lock = threading.Lock()
        self._responses: queue.Queue[Any] = queue.Queue()
        self._response_thread = threading.Thread(
            target=self._copy_responses, daemon=True
        )
        self._response_thread.start()

    def _copy_stderr(self) -> None:
        assert self.process.stderr
        for line in self.process.stderr:
            log("cuda_worker_stderr", line=line.rstrip())

    def _read(self) -> dict[str, Any]:
        assert self.process.stdout
        line = self.process.stdout.readline()
        if not line:
            raise WorkerError(f"CUDA worker exited with {self.process.poll()}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise WorkerError(f"invalid CUDA worker response: {line!r}") from error
        if payload.get("type") == "error":
            detail = payload.get("message")
            if isinstance(detail, str) and detail:
                raise WorkerError(f"CUDA worker rejected the command: {detail}")
            raise WorkerError("CUDA worker rejected the command")
        return payload

    def _copy_responses(self) -> None:
        while self.process.poll() is None:
            try:
                self._responses.put(self._read())
            except Exception as error:
                self._responses.put(error)
                if self.process.poll() is not None:
                    return

    def _command(self, command: str,
                 cancel_check: Callable[[], bool] | None = None,
                 cancel_id: int | None = None,
                 deadline: float | None = None) -> dict[str, Any]:
        with self.command_lock:
            if self.process.poll() is not None:
                raise WorkerError("CUDA worker is not running")
            assert self.process.stdin
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            interrupted = False
            while True:
                if not interrupted and (
                        (cancel_check is not None and cancel_check()) or
                        (deadline is not None and time.monotonic() > deadline)):
                    if cancel_id is not None:
                        self.process.stdin.write(f"CANCEL\t{cancel_id}\n")
                        self.process.stdin.flush()
                    interrupted = True
                try:
                    response = self._responses.get(timeout=0.05)
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise WorkerError("CUDA worker exited while awaiting response")
                    continue
                if isinstance(response, Exception):
                    raise response
                if interrupted:
                    raise WorkerError("CUDA worker request was cancelled")
                return response

    def begin(self, request_id: int, prompt_ids: list[int], context_limit: int,
              sampling: SamplingSettings,
              checkpoint_tokens: int | None = None,
              cancel_check: Callable[[], bool] | None = None,
              deadline: float | None = None) -> None:
        if request_id in self.active_ids:
            raise WorkerError("duplicate worker request")
        command = (f"BEGIN\t{request_id}\t{context_limit}\t" +
                   ",".join(str(token) for token in prompt_ids))
        if checkpoint_tokens is not None:
            command += f"\tCHECKPOINT\t{checkpoint_tokens}"
        command += self._sampling_command(sampling)
        response = self._command(
            command,
            cancel_check=cancel_check, cancel_id=request_id,
            deadline=deadline,
        )
        if response.get("type") != "begun" or response.get("id") != request_id:
            raise WorkerError("unexpected BEGIN response")
        self.active_ids.add(request_id)

    def begin_resume(self, request_id: int, session_key: int,
                     delta_ids: list[int], context_limit: int,
                     sampling: SamplingSettings,
                     checkpoint_tokens: int | None = None,
                     cancel_check: Callable[[], bool] | None = None,
                     deadline: float | None = None) -> None:
        if request_id in self.active_ids:
            raise WorkerError("duplicate worker request")
        if not delta_ids:
            raise WorkerError("resume requires at least one delta token")
        command = (f"BEGIN\t{request_id}\t{context_limit}\t" +
                   ",".join(str(token) for token in delta_ids) +
                   f"\tRESUME\t{session_key}")
        if checkpoint_tokens is not None:
            command += f"\tCHECKPOINT\t{checkpoint_tokens}"
        command += self._sampling_command(sampling)
        response = self._command(
            command, cancel_check=cancel_check,
            cancel_id=request_id, deadline=deadline,
        )
        if response.get("type") != "begun" or response.get("id") != request_id:
            raise WorkerError("unexpected BEGIN response")
        self.active_ids.add(request_id)

    def _sampling_command(self, settings: SamplingSettings) -> str:
        if not self.sampling_supported:
            if settings.enabled:
                raise WorkerError("CUDA worker does not support sampling")
            return ""
        return "\tSAMPLING\t{}\t{}\t{}\t{}\t{}".format(
            round(settings.temperature * 1_000_000),
            round(settings.top_p * 1_000_000),
            settings.top_k,
            round(settings.min_p * 1_000_000),
            settings.seed,
        )

    def end_retain(self, request_id: int, session_key: int,
                   checkpoint_tokens: int | None = None) -> int:
        command = f"END\t{request_id}\tRETAIN\t{session_key}"
        if checkpoint_tokens is not None:
            command += f"\tAT\t{checkpoint_tokens}"
        response = self._command(command)
        if response.get("type") != "ended" or response.get("id") != request_id:
            raise WorkerError("unexpected END response")
        self.active_ids.discard(request_id)
        return int(response.get("retained_tokens", 0))

    def drop_session(self, session_key: int) -> None:
        response = self._command(f"DROP\t{session_key}")
        if response.get("type") != "dropped":
            raise WorkerError("unexpected DROP response")

    def next(self, request_id: int, final: bool) -> list[int]:
        response = self._command(f"NEXT\t{request_id}\t{1 if final else 0}")
        if response.get("type") != "token" or response.get("id") != request_id:
            raise WorkerError("unexpected NEXT response")
        if final:
            self.active_ids.discard(request_id)
        return _worker_response_tokens(response, "NEXT response has no tokens")

    def step(self, items: list[tuple[int, int]]) -> dict[int, list[int]]:
        if not items or len(items) > self.capacity:
            raise WorkerError("invalid decode batch")
        if self.protocol < 2:
            if len(items) != 1:
                raise WorkerError("protocol v1 cannot batch decode")
            request_id, flag = items[0]
            return {request_id: self.next(request_id, flag == 1)}
        response = self._command("STEP\t" + "\t".join(
            f"{request_id},{flag}" for request_id, flag in items
        ))
        if response.get("type") != "batch" or not isinstance(response.get("items"), list):
            raise WorkerError("unexpected STEP response")
        result: dict[int, list[int]] = {}
        for item in response["items"]:
            result[int(item["id"])] = _worker_response_tokens(
                item, "STEP response item has no tokens"
            )
        expected = {request_id for request_id, _final in items}
        if set(result) != expected:
            raise WorkerError("STEP response request mismatch")
        for request_id, final in items:
            if final:
                self.active_ids.discard(request_id)
        return result

    def cancel(self, request_id: int) -> None:
        if request_id not in self.active_ids:
            return
        response = self._command(f"END\t{request_id}")
        if response.get("type") != "ended":
            raise WorkerError("unexpected END response")
        self.active_ids.discard(request_id)

    def healthy(self) -> bool:
        return self.process.poll() is None

    def stats(self) -> dict[str, int]:
        response = self._command("STATS")
        if response.get("type") != "stats":
            raise WorkerError("unexpected STATS response")
        result = {
            key: int(value) for key, value in response.items()
            if key != "type" and isinstance(value, (int, bool))
        }
        result.update({
            "allocated_pages": int(response["kv_allocated_pages"]),
            "reserved_pages": int(response["kv_reserved_pages"]),
        })
        return result

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            for request_id in list(self.active_ids):
                self.cancel(request_id)
            self._command("SHUTDOWN")
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()


class DecodeWaiter:
    def __init__(self, request_id: int, final: bool, hold: bool = False) -> None:
        self.request_id = request_id
        self.final = final
        self.hold = hold
        self.event = threading.Event()
        self.token: int | None = None
        self.error: Exception | None = None


class ContinuousDecodeBatcher:
    def __init__(self, worker: CudaWorker, window_ms: float,
                 on_batch: Any) -> None:
        self.worker = worker
        self.window_seconds = window_ms / 1000.0
        self.on_batch = on_batch
        self.pending: queue.Queue[DecodeWaiter | None] = queue.Queue()
        self.buffer_lock = threading.Lock()
        self.buffered: dict[int, list[int]] = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def step(self, request_id: int, final: bool, hold: bool = False) -> int:
        with self.buffer_lock:
            buffered = self.buffered.get(request_id)
            if buffered:
                token = buffered.pop(0)
                if not buffered:
                    self.buffered.pop(request_id, None)
                if final:
                    self.buffered.pop(request_id, None)
                    self.worker.cancel(request_id)
                return token
        waiter = DecodeWaiter(request_id, final, hold)
        self.pending.put(waiter)
        waiter.event.wait()
        if waiter.error is not None:
            raise waiter.error
        if waiter.token is None:
            raise WorkerError("decode batch returned no token")
        return waiter.token

    def _run(self) -> None:
        while True:
            first = self.pending.get()
            if first is None:
                return
            batch = [first]
            # With a single request in flight there is no partner to wait
            # for; dispatch immediately instead of burning the window
            # (~6% of every decode step at 30 tok/s). Under real concurrency
            # the queue is non-empty and the window still applies.
            deadline = time.monotonic() + self.window_seconds
            while not self.pending.empty() and len(batch) < self.worker.capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.pending.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.pending.put(None)
                    break
                batch.append(item)
            try:
                # Step modes: 0 = decode (speculative under MTP), 1 = final
                # emit-and-release, 2 = plain decode that keeps the slot. A
                # retained turn ends with mode 2 so the worker state stops on
                # an exact emitted-token boundary instead of holding an
                # unemitted speculative bonus token.
                tokens = self.worker.step([
                    (item.request_id,
                     1 if item.final else (2 if item.hold else 0))
                    for item in batch
                ])
                self.on_batch(len(batch))
                for item in batch:
                    raw = tokens[item.request_id]
                    produced = raw if isinstance(raw, list) else [int(raw)]
                    item.token = produced[0]
                    if not item.final and len(produced) > 1:
                        with self.buffer_lock:
                            self.buffered[item.request_id] = produced[1:]
            except Exception as error:
                for item in batch:
                    item.error = error
            finally:
                for item in batch:
                    item.event.set()

    def close(self) -> None:
        self.pending.put(None)
        self.thread.join(timeout=10)

    def discard(self, request_id: int) -> None:
        with self.buffer_lock:
            self.buffered.pop(request_id, None)

    def take_buffered(self, request_id: int) -> list[int]:
        with self.buffer_lock:
            return list(self.buffered.pop(request_id, []))


class Application:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.manifest = json.loads((args.container / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer), local_files_only=True, trust_remote_code=False
        )
        self.response_protocol = install_declared_response_protocol(
            self.tokenizer
        )
        generation_config_path = args.tokenizer / "generation_config.json"
        generation_config: dict[str, Any] = {}
        if generation_config_path.is_file():
            loaded_generation_config = json.loads(
                generation_config_path.read_text(encoding="utf-8")
            )
            if isinstance(loaded_generation_config, dict):
                generation_config = loaded_generation_config
        do_sample = generation_config.get("do_sample", False)
        if not isinstance(do_sample, bool):
            raise RuntimeError("generation_config do_sample must be boolean")
        self.default_sampling = SamplingSettings(
            temperature=float(generation_config.get(
                "temperature", 1.0 if do_sample else 0.0
            )),
            top_p=float(generation_config.get("top_p", 1.0)),
            top_k=int(generation_config.get("top_k", 0)),
            min_p=float(generation_config.get("min_p", 0.0)),
            seed=0,
        )
        self._validate_sampling(self.default_sampling, "generation_config")
        self.checkpoint_chat_encoder: Callable[..., str] | None = None
        if (not self.tokenizer.chat_template and
                (args.tokenizer / "encoding" / "encoding_dsv4.py").is_file()):
            self.checkpoint_chat_encoder = _load_deepseek_chat_encoder(
                args.tokenizer
            )
        eos = self.tokenizer.eos_token_id
        if eos is None:
            self.eos_token_ids: set[int] = set()
        elif isinstance(eos, int):
            self.eos_token_ids = {eos}
        else:
            self.eos_token_ids = {int(token) for token in eos}
        self.worker = CudaWorker(args.worker, args.container, args.max_context,
                                 args.startup_timeout, args.worker_capacity,
                                 args.worker_ram_cache_gib,
                                 args.worker_vram_cache_gib,
                                 args.worker_kv_cache_mib,
                                 args.worker_kv_page_tokens,
                                 args.placement_profile,
                                 args.profile_gpu_phases,
                                 args.worker_prefill_chunk_tokens,
                                 args.worker_placement_settle_steps,
                                 (False if args.disable_worker_retained_route
                                  else None),
                                 True if args.enable_worker_cpu_hybrid else None,
                                 args.worker_route_trace_file,
                                 args.worker_route_trace_max_steps)
        self.capacity = threading.BoundedSemaphore(
            args.maximum_queue + args.worker_capacity
        )
        self.worker_slots = threading.BoundedSemaphore(args.worker_capacity)
        self.kv_credit_lock = threading.Lock()
        self.kv_reserved_pages = 0
        self.session_lock = threading.Lock()
        self.sessions: OrderedDict[int, Session] = OrderedDict()
        self.next_session_key = 1
        self.id_lock = threading.Lock()
        self.next_id = 1
        self.draining = threading.Event()
        self.active = 0
        self.active_lock = threading.Lock()
        self.metric_lock = threading.Lock()
        self.metrics = {
            "admitted": 0, "rejected": 0, "completed": 0,
            "failed": 0, "cancelled": 0, "generated_tokens": 0,
            "decode_batches": 0, "decode_rows": 0,
        }
        self.latencies: dict[str, deque[float]] = {
            "ttft_seconds": deque(maxlen=args.latency_window),
            "inter_token_seconds": deque(maxlen=args.latency_window),
        }
        self.decode_batcher = ContinuousDecodeBatcher(
            self.worker, args.microbatch_window_ms, self.record_decode_batch
        )
        log("service_ready", model=args.model, build=args.build_id,
            manifest=self.manifest["indexes"]["experts_sha256"])

    def request_id(self) -> int:
        with self.id_lock:
            result = self.next_id
            self.next_id += 1
            return result

    def acquire(self) -> bool:
        if self.draining.is_set():
            self.increment("rejected")
            return False
        if not self.capacity.acquire(timeout=self.args.queue_timeout):
            self.increment("rejected")
            return False
        with self.active_lock:
            self.active += 1
        self.increment("admitted")
        return True

    def release(self) -> None:
        with self.active_lock:
            self.active -= 1
        self.capacity.release()

    def acquire_worker_slot(self) -> bool:
        return self.worker_slots.acquire(timeout=self.args.queue_timeout)

    def release_worker_slot(self) -> None:
        self.worker_slots.release()

    def acquire_context_credits(self, context_tokens: int) -> int:
        pages = (context_tokens + self.worker.kv_page_tokens - 1) // \
            self.worker.kv_page_tokens
        with self.kv_credit_lock:
            if self.kv_reserved_pages + pages > self.worker.kv_page_capacity:
                return 0
            self.kv_reserved_pages += pages
        return pages

    def release_context_credits(self, pages: int) -> None:
        with self.kv_credit_lock:
            if pages <= 0 or pages > self.kv_reserved_pages:
                raise RuntimeError("invalid KV context credit release")
            self.kv_reserved_pages -= pages

    def retention_enabled(self) -> bool:
        return (not self.args.disable_session_retention and
                getattr(self.worker, "session_retention", False))

    def _context_pages(self, context_tokens: int) -> int:
        return (context_tokens + self.worker.kv_page_tokens - 1) // \
            self.worker.kv_page_tokens

    def _acquire_pages(self, pages: int) -> bool:
        with self.kv_credit_lock:
            if self.kv_reserved_pages + pages > self.worker.kv_page_capacity:
                return False
            self.kv_reserved_pages += pages
        return True

    def _drop_worker_session(self, session_key: int) -> None:
        try:
            self.worker.drop_session(session_key)
        except WorkerError:
            # The worker already forgot the session (or never retained it);
            # the server-side entry is dropped either way.
            pass

    def _sweep_idle_sessions(self) -> None:
        if not self.sessions:
            return
        cutoff = time.monotonic() - self.args.session_idle_seconds
        expired: list[Session] = []
        with self.session_lock:
            for key, session in list(self.sessions.items()):
                if session.last_used < cutoff:
                    expired.append(self.sessions.pop(key))
        for session in expired:
            self._drop_worker_session(session.key)
            self.release_context_credits(session.pages)
            log("session_expired", key=session.key, tokens=len(session.tokens))

    def evict_lru_session(self) -> bool:
        with self.session_lock:
            if not self.sessions:
                return False
            _key, session = self.sessions.popitem(last=False)
        self._drop_worker_session(session.key)
        self.release_context_credits(session.pages)
        log("session_evicted", key=session.key, tokens=len(session.tokens))
        return True

    def checkout_session(self, prompt_ids: list[int]) -> Session | None:
        """Take the longest retained prefix of prompt_ids out of the map."""
        if not self.retention_enabled():
            return None
        self._sweep_idle_sessions()
        with self.session_lock:
            best: Session | None = None
            for session in self.sessions.values():
                length = len(session.tokens)
                if (length <= len(prompt_ids) and
                        prompt_ids[:length] == session.tokens and
                        (best is None or length > len(best.tokens))):
                    best = session
            if best is not None:
                del self.sessions[best.key]
        return best

    def store_session(self, session_key: int, tokens: list[int],
                      pages: int) -> None:
        duplicates: list[Session] = []
        with self.session_lock:
            for key, session in list(self.sessions.items()):
                if session.tokens == tokens:
                    duplicates.append(self.sessions.pop(key))
            self.sessions[session_key] = Session(
                key=session_key, tokens=tokens, pages=pages,
                last_used=time.monotonic(),
            )
        for duplicate in duplicates:
            self._drop_worker_session(duplicate.key)
            self.release_context_credits(duplicate.pages)

    def abandon_session(self, session: Session) -> None:
        """Drop a checked-out session whose request never started."""
        self._drop_worker_session(session.key)
        self.release_context_credits(session.pages)

    def allocate_session_key(self) -> int:
        with self.session_lock:
            key = self.next_session_key
            self.next_session_key += 1
            return key

    def acquire_request_context(self, prompt_ids: list[int],
                                maximum: int) -> RequestContext | None:
        session = self.checkout_session(prompt_ids)
        # One extra position covers the non-final trailing decode step that
        # retained turns require. At the model context boundary generate()
        # disables retention, so admission must not reserve an impossible
        # position beyond the advertised context window.
        context_tokens = len(prompt_ids) + maximum
        retain_headroom = int(
            self.retention_enabled() and context_tokens < self.args.max_context
        )
        total_pages = self._context_pages(context_tokens + retain_headroom)
        base_pages = session.pages if session is not None else 0
        needed = max(0, total_pages - base_pages)
        while needed > 0 and not self._acquire_pages(needed):
            if not self.evict_lru_session():
                if session is not None:
                    self.abandon_session(session)
                return None
        return RequestContext(session=session,
                              held_pages=base_pages + needed)

    def release_request_context(self, context: RequestContext) -> None:
        if context.retained:
            # The retained session keeps the worker slot and its KV pages.
            return
        if context.held_pages > 0:
            self.release_context_credits(context.held_pages)
        if context.session is not None:
            self._drop_worker_session(context.session.key)

    def worker_stats(self) -> dict[str, int]:
        stats = getattr(self.worker, "stats", None)
        if stats is None:
            return {}
        try:
            return stats()
        except Exception:
            return {}

    def record_decode_batch(self, rows: int) -> None:
        self.increment("decode_batches")
        self.increment("decode_rows", rows)

    def increment(self, name: str, value: int = 1) -> None:
        with self.metric_lock:
            self.metrics[name] += value

    def observe_latency(self, name: str, value: float) -> None:
        with self.metric_lock:
            self.latencies[name].append(value)

    def metrics_text(self) -> str:
        with self.active_lock:
            active = self.active
        with self.metric_lock:
            counters = dict(self.metrics)
            latency_samples = {
                name: sorted(values) for name, values in self.latencies.items()
            }
        with self.kv_credit_lock:
            kv_reserved_pages = self.kv_reserved_pages
        lines = [
            "# TYPE expert_service_active_requests gauge",
            f"expert_service_active_requests {active}",
            "# TYPE expert_service_kv_reserved_pages gauge",
            f"expert_service_kv_reserved_pages {kv_reserved_pages}",
        ]
        for name, value in counters.items():
            lines.extend((f"# TYPE expert_service_{name}_total counter",
                          f"expert_service_{name}_total {value}"))
        for name, values in latency_samples.items():
            def percentile(fraction: float) -> float:
                if not values:
                    return 0.0
                index = max(0, int(len(values) * fraction + 0.999999) - 1)
                return values[min(index, len(values) - 1)]
            lines.extend((
                f"# TYPE expert_service_{name}_p50 gauge",
                f"expert_service_{name}_p50 {percentile(0.50)}",
                f"# TYPE expert_service_{name}_p95 gauge",
                f"expert_service_{name}_p95 {percentile(0.95)}",
                f"# TYPE expert_service_{name}_count gauge",
                f"expert_service_{name}_count {len(values)}",
            ))
        lines.extend(("# TYPE expert_service_ready gauge",
                      f"expert_service_ready {int(self.worker.healthy() and not self.draining.is_set())}"))
        try:
            worker_stats = self.worker.stats()
        except WorkerError:
            worker_stats = {}
        for name, value in worker_stats.items():
            metric = f"expert_worker_{name}"
            lines.extend((f"# TYPE {metric} gauge", f"{metric} {value}"))
        return "\n".join(lines) + "\n"

    def _chat_prompt_ids(self, messages: list[dict[str, Any]],
                         add_generation_prompt: bool = True,
                         tools: tuple[dict[str, Any], ...] = (),
                         reasoning_effort: str = "xhigh",
                         enable_thinking: bool = True,
                         preserve_thinking: bool = True) -> list[int]:
        encoder = getattr(self, "checkpoint_chat_encoder", None)
        if encoder is not None:
            if tools:
                raise RequestError(
                    "the published tokenizer adapter does not declare tool calling",
                    "tools", "unsupported_value",
                )
            prompt = encoder(messages, thinking_mode="chat")
            return [int(token) for token in self.tokenizer.encode(
                prompt, add_special_tokens=False
            )]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=list(tools) if tools else None,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token) for token in ids]

    def response_stream_parser(self, request: GenerationRequest) -> Any:
        if self.response_protocol is None:
            return None
        return self.tokenizer.get_response_parser(
            prefix=request.prompt_ids,
        )

    def parse_assistant_output(
            self, text: str, request: GenerationRequest) -> AssistantOutput:
        if self.response_protocol is None:
            return AssistantOutput(text=text, reasoning="",
                                   reasoning_complete=True, tool_calls=())
        try:
            message = self.tokenizer.parse_response(
                text, prefix=request.prompt_ids,
                tools=list(request.tools) if request.tools else None,
            )
            if not isinstance(message, dict):
                raise ValueError("response parser returned a non-object")
            reasoning = message.get(
                "reasoning_content",
                message.get("reasoning", message.get("thinking", "")),
            )
            visible = message.get("content", "")
            if reasoning is None:
                reasoning = ""
            if visible is None:
                visible = ""
            if not isinstance(reasoning, str) or not isinstance(visible, str):
                raise ValueError("response parser returned non-text regions")
            calls: list[ToolCall] = []
            raw_calls = message.get("tool_calls", [])
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                raise ValueError("response parser returned invalid tool calls")
            complete_tool_markup = (
                text.count("<tool_call>") == text.count("</tool_call>") and
                text.count("<function=") == text.count("</function>")
            )
            if raw_calls and not complete_tool_markup:
                raise ValueError("model emitted an incomplete tool call")
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict) or raw_call.get("type") != "function":
                    raise ValueError("response parser returned an invalid tool call")
                function = raw_call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("parsed tool call has no function")
                name = function.get("name")
                arguments = function.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise ValueError("parsed tool call has no function name")
                if isinstance(arguments, str):
                    parsed_arguments = json.loads(arguments)
                    if not isinstance(parsed_arguments, dict):
                        raise ValueError("parsed tool arguments are not an object")
                    arguments = parsed_arguments
                if not isinstance(arguments, dict):
                    raise ValueError("parsed tool arguments are not an object")
                calls.append(ToolCall(
                    item_id="fc_" + uuid.uuid4().hex,
                    call_id="call_" + uuid.uuid4().hex,
                    name=name,
                    arguments=json.dumps(
                        arguments, separators=(",", ":"), ensure_ascii=False
                    ),
                ))
            return AssistantOutput(
                text=visible, reasoning=reasoning,
                reasoning_complete=True, tool_calls=tuple(calls),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            log("response_parse_failed", protocol=self.response_protocol,
                error=str(error))
            return AssistantOutput(text=text, reasoning="",
                                   reasoning_complete=False, tool_calls=())

    def text_token_count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _messages(self, raw: Any, param: str = "messages") -> list[dict[str, Any]]:
        if not isinstance(raw, list) or not raw:
            raise RequestError(f"{param} must be a non-empty array", param)
        result: list[dict[str, Any]] = []
        for index, message in enumerate(raw):
            item_param = f"{param}.{index}"
            if isinstance(message, str):
                result.append({"role": "user", "content": message})
                continue
            if not isinstance(message, dict):
                raise RequestError("message must be an object", item_param)
            item_type = message.get("type", "message")
            if item_type == "function_call_output":
                content = message.get("output")
                if not isinstance(content, str):
                    raise RequestError(
                        "function_call_output requires string output",
                        f"{item_param}.output",
                    )
                result.append({"role": "tool", "content": content,
                               "tool_call_id": message.get("call_id")})
                continue
            if item_type == "function_call":
                name = message.get("name")
                arguments = message.get("arguments", "{}")
                if not isinstance(name, str) or not name:
                    raise RequestError("function_call requires a name",
                                       f"{item_param}.name")
                if not isinstance(arguments, str):
                    raise RequestError("function_call arguments must be JSON text",
                                       f"{item_param}.arguments")
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise RequestError("function_call arguments must be valid JSON",
                                       f"{item_param}.arguments") from error
                if not isinstance(parsed_arguments, dict):
                    raise RequestError("function_call arguments must be a JSON object",
                                       f"{item_param}.arguments")
                result.append({
                    "role": "assistant", "content": "",
                    "tool_calls": [{"id": message.get("call_id"),
                                    "type": "function", "function": {
                                        "name": name,
                                        "arguments": parsed_arguments,
                                    }}],
                })
                continue
            if item_type != "message":
                raise RequestError("input item type is not supported",
                                   f"{item_param}.type", "unsupported_value")
            role = message.get("role")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant", "tool"}:
                raise RequestError(f"message role {role!r} is not supported",
                                   f"{item_param}.role", "unsupported_value")
            raw_content = message.get("content")
            if raw_content is None and role == "assistant":
                content = ""
            else:
                content = _text_content(raw_content, f"{item_param}.content")
            normalized: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                reasoning = message.get("reasoning_content")
                if reasoning is not None:
                    if not isinstance(reasoning, str):
                        raise RequestError("reasoning_content must be text",
                                           f"{item_param}.reasoning_content")
                    normalized["reasoning_content"] = reasoning
                raw_calls = message.get("tool_calls")
                if raw_calls is None and message.get("function_call") is not None:
                    raw_calls = [{"type": "function",
                                  "function": message["function_call"]}]
                if raw_calls is not None:
                    if not isinstance(raw_calls, list) or not raw_calls:
                        raise RequestError("tool_calls must be a non-empty array",
                                           f"{item_param}.tool_calls")
                    calls: list[dict[str, Any]] = []
                    for call_index, call in enumerate(raw_calls):
                        call_param = f"{item_param}.tool_calls.{call_index}"
                        if not isinstance(call, dict) or call.get("type", "function") != "function":
                            raise RequestError("only function tool calls are supported",
                                               call_param, "unsupported_value")
                        function = call.get("function")
                        if not isinstance(function, dict):
                            raise RequestError("tool call requires a function object",
                                               f"{call_param}.function")
                        name = function.get("name")
                        arguments = function.get("arguments", {})
                        if not isinstance(name, str) or not name:
                            raise RequestError("tool call requires a function name",
                                               f"{call_param}.function.name")
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError as error:
                                raise RequestError(
                                    "tool call arguments must be valid JSON",
                                    f"{call_param}.function.arguments",
                                ) from error
                        if not isinstance(arguments, dict):
                            raise RequestError(
                                "tool call arguments must be a JSON object",
                                f"{call_param}.function.arguments",
                            )
                        calls.append({"id": call.get("id"), "type": "function",
                                      "function": {"name": name,
                                                   "arguments": arguments}})
                    normalized["tool_calls"] = calls
            elif role == "tool":
                normalized["tool_call_id"] = message.get("tool_call_id")
            result.append(normalized)
        return result

    @staticmethod
    def _tools(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        raw_tools = payload.get("tools")
        legacy = payload.get("functions")
        if raw_tools not in (None, []) and legacy not in (None, []):
            raise RequestError("tools and functions cannot both be supplied", "tools")
        if legacy not in (None, []):
            if not isinstance(legacy, list):
                raise RequestError("functions must be an array", "functions")
            raw_tools = [{"type": "function", "function": item}
                         for item in legacy]
        if raw_tools in (None, []):
            return ()
        if not isinstance(raw_tools, list):
            raise RequestError("tools must be an array", "tools")
        tools: list[dict[str, Any]] = []
        names: set[str] = set()
        for index, tool in enumerate(raw_tools):
            param = f"tools.{index}"
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise RequestError("only function tools are supported", param,
                                   "unsupported_value")
            function = tool.get("function")
            if not isinstance(function, dict):
                raise RequestError("function tool requires a function object",
                                   f"{param}.function")
            name = function.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise RequestError("function tool names must be non-empty and unique",
                                   f"{param}.function.name")
            description = function.get("description")
            parameters = function.get("parameters", {"type": "object"})
            if description is not None and not isinstance(description, str):
                raise RequestError("function description must be text",
                                   f"{param}.function.description")
            if not isinstance(parameters, dict):
                raise RequestError("function parameters must be a JSON schema object",
                                   f"{param}.function.parameters")
            names.add(name)
            normalized_function: dict[str, Any] = {
                "name": name, "parameters": parameters,
            }
            if description is not None:
                normalized_function["description"] = description
            if "strict" in function:
                if not isinstance(function["strict"], bool):
                    raise RequestError("function strict must be boolean",
                                       f"{param}.function.strict")
                normalized_function["strict"] = function["strict"]
            tools.append({"type": "function", "function": normalized_function})
        return tuple(tools)

    @staticmethod
    def _tool_choice(payload: dict[str, Any],
                     tools: tuple[dict[str, Any], ...]) -> str | dict[str, Any]:
        choice = payload.get("tool_choice", payload.get("function_call", "auto"))
        if choice is None:
            choice = "auto"
        if isinstance(choice, str):
            if choice not in {"auto", "none", "required"}:
                if payload.get("function_call") == choice:
                    choice = {"type": "function", "function": {"name": choice}}
                else:
                    raise RequestError("unsupported tool_choice", "tool_choice",
                                       "unsupported_value")
        elif isinstance(choice, dict):
            function = choice.get("function")
            if choice.get("type", "function") != "function" or not isinstance(function, dict) or \
                    not isinstance(function.get("name"), str):
                raise RequestError("tool_choice function selection is invalid",
                                   "tool_choice")
        else:
            raise RequestError("tool_choice is invalid", "tool_choice")
        if not tools and choice not in {"auto", "none"}:
            raise RequestError("tool_choice requires tools", "tool_choice")
        if isinstance(choice, dict):
            selected = choice["function"]["name"]
            if selected not in {tool["function"]["name"] for tool in tools}:
                raise RequestError("tool_choice names an unknown function",
                                   "tool_choice")
        return choice

    @staticmethod
    def _validate_sampling(settings: SamplingSettings,
                           parameter: str) -> None:
        valid = (
            0.0 <= settings.temperature <= 2.0 and
            0.0 < settings.top_p <= 1.0 and
            0 <= settings.top_k <= 1_000_000 and
            0.0 <= settings.min_p <= 1.0 and
            0 <= settings.seed <= 0x7fff_ffff_ffff_ffff
        )
        if not valid:
            if parameter == "generation_config":
                raise RuntimeError("generation_config sampling values are invalid")
            raise RequestError("sampling parameters are outside runtime limits",
                               parameter, "unsupported_value")

    def _sampling(self, payload: dict[str, Any]) -> SamplingSettings:
        defaults = self.default_sampling

        def number(name: str, default: float) -> float:
            value = payload.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RequestError(f"{name} must be numeric", name)
            return float(value)

        top_k = payload.get("top_k", defaults.top_k)
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise RequestError("top_k must be an integer", "top_k")
        seed = payload.get("seed")
        if seed is None:
            seed = uuid.uuid4().int & 0x7fff_ffff_ffff_ffff
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise RequestError("seed must be an integer", "seed")
        settings = SamplingSettings(
            temperature=number("temperature", defaults.temperature),
            top_p=number("top_p", defaults.top_p),
            top_k=top_k,
            min_p=number("min_p", defaults.min_p),
            seed=seed,
        )
        self._validate_sampling(settings, "sampling")
        return settings

    @staticmethod
    def _stop_sequences(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        stops = [value] if isinstance(value, str) else value
        if (not isinstance(stops, list) or not 1 <= len(stops) <= 4 or
                not all(isinstance(stop, str) and stop for stop in stops)):
            raise RequestError("stop must be a non-empty string or up to four strings", "stop")
        return tuple(stops)

    @staticmethod
    def _stream_usage(payload: dict[str, Any]) -> bool:
        options = payload.get("stream_options")
        if options is None:
            return False
        if not isinstance(options, dict):
            raise RequestError("stream_options must be an object", "stream_options")
        include = options.get("include_usage", False)
        if not isinstance(include, bool):
            raise RequestError("include_usage must be boolean", "stream_options.include_usage")
        return include

    @staticmethod
    def _validate_compatibility(payload: dict[str, Any], endpoint: str) -> None:
        def only(field: str, allowed: tuple[Any, ...], message: str) -> None:
            if field in payload and payload[field] not in allowed:
                raise RequestError(message, field, "unsupported_value")

        only("n", (None, 1), "this runtime supports n=1")
        only("best_of", (None, 1), "this runtime supports best_of=1")
        only("presence_penalty", (None, 0, 0.0),
             "presence_penalty is not implemented")
        only("frequency_penalty", (None, 0, 0.0),
             "frequency_penalty is not implemented")
        only("repetition_penalty", (None, 1, 1.0),
             "repetition_penalty is not implemented")
        only("logprobs", (None, False, 0), "logprobs are not implemented")
        only("top_logprobs", (None, 0), "top_logprobs are not implemented")
        only("echo", (None, False), "echo is not implemented")
        only("background", (None, False), "background responses are not implemented")
        only("previous_response_id", (None,), "stored response chaining is not implemented")
        only("conversation", (None,), "server-side conversations are not implemented")
        only("truncation", (None, "disabled"), "automatic truncation is not implemented")

        if payload.get("logit_bias") not in (None, {}):
            raise RequestError("logit_bias is not implemented", "logit_bias", "unsupported_value")
        if payload.get("modalities") not in (None, ["text"]):
            raise RequestError("only text output is supported", "modalities", "unsupported_value")
        if payload.get("audio") is not None:
            raise RequestError("audio output is not supported", "audio", "unsupported_value")
        if payload.get("prediction") is not None:
            raise RequestError("predicted output is not implemented", "prediction", "unsupported_value")
        if payload.get("suffix") is not None:
            raise RequestError("suffix completion is not implemented", "suffix", "unsupported_value")

        response_format = payload.get("response_format")
        if response_format not in (None, {"type": "text"}):
            raise RequestError("only response_format type=text is supported",
                               "response_format", "unsupported_value")
        text = payload.get("text")
        if text is not None:
            if not isinstance(text, dict):
                raise RequestError("text must be an object", "text")
            text_format = text.get("format", {"type": "text"})
            if text_format != {"type": "text"}:
                raise RequestError("only text.format type=text is supported",
                                   "text.format", "unsupported_value")
        if endpoint == "responses" and payload.get("include") not in (None, []):
            raise RequestError("additional response fields are not available",
                               "include", "unsupported_value")

    def parse_request(self, payload: dict[str, Any], endpoint: str) -> GenerationRequest:
        if payload.get("model", self.args.model) != self.args.model:
            raise RequestError("unknown model", "model", "model_not_found")
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise RequestError("stream must be boolean", "stream")
        self._validate_compatibility(payload, endpoint)
        tools = self._tools(payload)
        if tools and self.response_protocol is None:
            raise RequestError(
                "the tokenizer does not declare a supported response protocol",
                "tools", "unsupported_value",
            )
        tool_choice = self._tool_choice(payload, tools)
        if endpoint == "completion" and tools:
            raise RequestError("tools require a chat or Responses request",
                               "tools", "unsupported_value")
        prompt_tools = tools
        if tool_choice == "none":
            prompt_tools = ()
        elif tool_choice == "required":
            raise RequestError(
                "required tool choice needs constrained decoding",
                "tool_choice", "unsupported_value",
            )
        elif isinstance(tool_choice, dict):
            raise RequestError(
                "named tool choice needs constrained decoding",
                "tool_choice", "unsupported_value",
            )
        template_kwargs = payload.get("chat_template_kwargs", {})
        if template_kwargs is None:
            template_kwargs = {}
        if not isinstance(template_kwargs, dict):
            raise RequestError("chat_template_kwargs must be an object",
                               "chat_template_kwargs")
        unknown_template_kwargs = set(template_kwargs) - {
            "enable_thinking", "preserve_thinking"
        }
        if unknown_template_kwargs:
            raise RequestError("unsupported chat template controls",
                               "chat_template_kwargs", "unsupported_value")
        enable_thinking = template_kwargs.get("enable_thinking", True)
        preserve_thinking = template_kwargs.get("preserve_thinking", True)
        if not isinstance(enable_thinking, bool):
            raise RequestError("enable_thinking must be boolean",
                               "chat_template_kwargs.enable_thinking")
        if not isinstance(preserve_thinking, bool):
            raise RequestError("preserve_thinking must be boolean",
                               "chat_template_kwargs.preserve_thinking")
        sampling = self._sampling(payload)
        reasoning_effort = payload.get("reasoning_effort")
        reasoning = payload.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise RequestError("reasoning must be an object", "reasoning")
            unknown = set(reasoning) - {"effort", "summary"}
            if unknown or reasoning.get("summary") not in (None, "auto"):
                raise RequestError("unsupported reasoning controls", "reasoning",
                                   "unsupported_value")
            if reasoning_effort is not None and reasoning.get("effort") is not None:
                raise RequestError("reasoning effort was specified twice",
                                   "reasoning_effort")
            reasoning_effort = reasoning.get("effort", reasoning_effort)
        if reasoning_effort is None:
            reasoning_effort = "xhigh"
        if reasoning_effort not in {"xhigh", "medium", "low"}:
            raise RequestError(
                "reasoning_effort must be xhigh, medium, or low",
                "reasoning_effort", "unsupported_value",
            )

        max_field = "max_output_tokens" if endpoint == "responses" else "max_tokens"
        maximum_value = payload.get("max_output_tokens") if endpoint == "responses" else payload.get(
            "max_completion_tokens", payload.get("max_tokens", 16)
        )
        if maximum_value is None:
            maximum_value = 16
        if isinstance(maximum_value, bool) or not isinstance(maximum_value, int):
            raise RequestError(f"{max_field} must be an integer", max_field)
        maximum = maximum_value
        if not 1 <= maximum <= self.args.maximum_new_tokens:
            raise RequestError(f"{max_field} is outside service limits", max_field)

        instructions = payload.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise RequestError("instructions must be a string", "instructions")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise RequestError("metadata must be an object", "metadata")
        user = payload.get("user")
        if user is not None and not isinstance(user, str):
            raise RequestError("user must be a string", "user")

        cache_prefix_tokens = 0
        if endpoint == "chat":
            messages = self._messages(payload.get("messages"))
            prompt_ids = self._chat_prompt_ids(
                messages, tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            stable_ids = self._chat_prompt_ids(
                messages, add_generation_prompt=False, tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            cache_prefix_tokens = len(stable_ids)
        elif endpoint == "responses":
            raw_input = payload.get("input")
            if isinstance(raw_input, str):
                messages = [{"role": "user", "content": raw_input}]
            else:
                messages = self._messages(raw_input, "input")
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            prompt_ids = self._chat_prompt_ids(
                messages, tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            stable_ids = self._chat_prompt_ids(
                messages, add_generation_prompt=False, tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            cache_prefix_tokens = len(stable_ids)
        else:
            prompt = payload.get("prompt")
            if isinstance(prompt, str):
                prompt_ids = [int(token) for token in self.tokenizer.encode(
                    prompt, add_special_tokens=False
                )]
            elif (isinstance(prompt, list) and prompt and
                  all(isinstance(item, int) and not isinstance(item, bool) for item in prompt)):
                prompt_ids = [int(token) for token in prompt]
            else:
                raise RequestError("prompt must be a string or token-id array", "prompt")
            cache_prefix_tokens = len(prompt_ids)

        if not prompt_ids or len(prompt_ids) >= self.args.max_context:
            raise RequestError("prompt is empty or exceeds context capacity",
                               "input" if endpoint == "responses" else "prompt")
        if len(prompt_ids) + maximum > self.args.max_context:
            raise RequestError("prompt plus output tokens exceeds context capacity", max_field)
        if endpoint in {"chat", "responses"} and (
                cache_prefix_tokens <= 0 or
                cache_prefix_tokens > len(prompt_ids) or
                prompt_ids[:cache_prefix_tokens] != stable_ids):
            raise RequestError("chat template has no stable cache prefix", "input")
        return GenerationRequest(
            endpoint=endpoint, prompt_ids=prompt_ids,
            cache_prefix_tokens=cache_prefix_tokens, maximum=maximum,
            stream=stream, stop=self._stop_sequences(payload.get("stop")),
            include_usage=self._stream_usage(payload), instructions=instructions,
            metadata=metadata, user=user, reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking, sampling=sampling,
            tools=tools, tool_choice=tool_choice,
        )

    _TELEMETRY_DELTA_KEYS = (
        "forward_calls", "forward_wall_ns", "expert_cache_wait_ns",
        "expert_compute_ns", "cpu_expert_ns", "gpu_expert_ns",
        "cpu_gpu_overlap_ns", "final_head_ns", "cache_read_bytes",
        "cache_uploaded_bytes", "cache_storage_wait_ns",
        "cache_upload_wait_ns", "worker_model_steps", "worker_model_step_ns",
        "frozen_promotions", "frozen_promotion_bytes",
        "worker_model_rows", "worker_embed_rope_submit_ns",
        "worker_scheduler_poll_ns", "worker_output_head_ns",
        "worker_attention_route_submit_ns", "worker_directory_plan_ns",
        "worker_ffn_submit_ns", "worker_directory_release_ns",
        "worker_gpu_attention_route_plan_ns", "worker_gpu_ffn_release_ns",
        "worker_gpu_attention_ns", "worker_gpu_route_ns",
        "worker_gpu_directory_plan_ns", "worker_gpu_ffn_ns",
        "worker_gpu_directory_release_ns",
        "worker_gpu_attention_hca_pre_norm_ns",
        "worker_gpu_attention_projection_ns",
        "worker_gpu_sparse_attention_ns",
        "worker_gpu_attention_output_projection_ns",
        "worker_gpu_attention_hca_post_ns", "worker_gpu_ffn_routed_ns",
        "worker_gpu_ffn_aggregate_ns", "worker_gpu_ffn_shared_ns",
        "worker_gpu_ffn_merge_ns", "worker_gpu_ffn_hca_post_ns",
        "scheduler_layer_advances", "scheduler_cuda_pending_polls",
        "scheduler_cuda_waits", "scheduler_cuda_wait_ns",
        "scheduler_expert_suspensions", "scheduler_acquires_started",
        "scheduler_acquires_completed", "scheduler_host_resolves",
        "scheduler_cpu_placements", "scheduler_hybrid_layers",
        "scheduler_controller_advance_ns", "scheduler_expert_wait_ns",
        "scheduler_poll_ns", "scheduler_prefetch_predictions",
        "scheduler_prefetch_scheduled", "scheduler_prefetch_completed",
        "scheduler_prefetch_useful", "scheduler_prefetch_late",
        "scheduler_prefetch_incorrect", "scheduler_prefetch_cancelled",
        "scheduler_prefetch_evicted_before_use", "cpu_execute_calls",
        "cpu_selections",
        "cpu_source_weight_bytes", "cpu_compute_ns", "planner_plans",
        "planner_candidates", "planner_cpu_cost_wins",
        "planner_gpu_cost_wins", "uploader_device_allocations",
        "uploader_recycled_acquires", "uploader_staging_allocations",
        "uploader_compact_h2d_bytes", "uploader_compact_cache_hits",
        "uploader_compact_cache_misses", "worker_mtp_drafts",
        "worker_mtp_accepted", "worker_mtp_rejected", "worker_verify_pairs",
        "worker_mtp_suppressions", "worker_mtp_acquire_batches",
        "worker_mtp_acquires_launched", "worker_mtp_acquire_wait_ns",
        "mtp_cache_vram_hits", "mtp_cache_ram_hits", "mtp_cache_ssd_misses",
        "mtp_cache_loads_started", "mtp_cache_loads_deduplicated",
        "mtp_cache_loads_completed", "mtp_cache_reload_count",
        "mtp_cache_reread_bytes", "mtp_cache_read_bytes",
        "mtp_cache_storage_wait_ns", "mtp_cache_host_validation_ns",
        "mtp_cache_host_copy_bytes", "mtp_cache_uploads_started",
        "mtp_cache_uploads_completed", "mtp_cache_uploaded_bytes",
        "mtp_cache_upload_wait_ns", "mtp_cache_evictions",
        "mtp_cache_stalled_by_budget", "mtp_cache_cancellations",
        "mtp_cache_io_errors", "mtp_cache_upload_errors",
        "worker_warm_start_loaded", "worker_warm_start_failed",
        "worker_warm_start_cancelled", "worker_warm_start_demand_pauses",
        "worker_warm_start_loop_errors",
        "worker_warm_start_bytes", "worker_warm_start_ns",
        "worker_warm_vram_loaded", "worker_warm_vram_failed",
        "worker_warm_vram_cancelled", "worker_warm_vram_demand_pauses",
        "worker_warm_vram_bytes", "worker_warm_vram_ns",
        "worker_prefill_protection_candidates",
        "worker_prefill_protection_promoted", "cache_loads_started",
        "cache_loads_deduplicated", "cache_loads_completed",
        "cache_uploads_started", "cache_uploads_completed",
        "cache_record_validations", "cache_validated_ram_reuses",
        "cache_requested_bytes", "cache_useful_bytes", "cache_reload_count",
        "cache_reread_bytes", "cache_priority_upgrades",
        "cache_host_preloads_requested", "cache_host_preloads_completed",
        "cache_host_validation_ns", "cache_host_validation_failures",
        "cache_host_copy_bytes", "cache_preloaded_host_useful",
        "cache_preloaded_host_useful_bytes", "cache_preloaded_host_wasted",
        "cache_preloaded_host_wasted_bytes", "cache_ram_promotions",
        "cache_vram_promotions", "cache_ram_promotion_failures",
        "cache_vram_promotion_failures", "cache_ram_probationary_evictions",
        "cache_ram_protected_evictions",
        "cache_ram_probationary_evicted_bytes",
        "cache_ram_protected_evicted_bytes",
        "cache_vram_transient_evictions", "cache_vram_resident_evictions",
        "cache_vram_transient_evicted_bytes",
        "cache_vram_resident_evicted_bytes", "cache_evictions",
        "cache_eviction_scan_calls", "cache_eviction_scan_candidates",
        "cache_eviction_scan_ns", "cache_eviction_retire_retries",
        "cache_vram_admission_scan_calls",
        "cache_vram_admission_scan_candidates",
        "cache_vram_admission_scan_ns", "cache_task_selection_calls",
        "cache_task_selection_candidates", "cache_task_selection_ns",
        "cache_mutex_acquisitions",
        "cache_mutex_wait_ns", "cache_mutex_wait_max_ns",
        "cache_same_partition_evictions", "cache_over_quota_evictions",
        "cache_stalled_by_budget", "cache_cancellations",
        "cache_short_read_errors", "cache_checksum_errors",
        "cache_io_errors", "cache_upload_errors", "staging_demand_acquires",
        "staging_background_acquires", "staging_demand_stalls",
        "staging_background_stalls",
    ) + tuple(
        f"cache_{priority}_{metric}"
        for priority in ("warm", "prefetch", "demand")
        for metric in (
            "device_requests", "host_requests", "vram_hits", "ram_hits",
            "ssd_misses", "host_lookup_misses", "reads_started",
            "reads_completed", "read_bytes", "storage_wait_ns",
            "host_validation_ns", "host_copy_bytes",
            "ram_retention_copy_ns", "uploads_started",
            "uploads_completed", "uploaded_bytes", "upload_wait_ns",
            "completed_waiters", "waiter_wait_ns", "cancellations",
            "failed_waiters", "io_errors", "validation_errors",
            "upload_errors", "staging_stalls",
        )
    ) + tuple(
        f"cache_transition_{source}_to_{target}"
        for source in (
            "absent", "ssd_loading", "ram_ready", "gpu_uploading",
            "vram_ready", "failed",
        )
        for target in (
            "absent", "ssd_loading", "ram_ready", "gpu_uploading",
            "vram_ready", "failed",
        )
    )

    # Protocol 7 providers own their counter vocabulary. Request attribution
    # therefore consumes every monotonic numeric counter returned by STATS
    # instead of requiring a server edit for each execution provider. Only
    # instantaneous resource gauges are excluded from deltas.
    _TELEMETRY_GAUGE_KEYS = frozenset({
        "active_requests", "retained_sessions", "allocated_pages",
        "reserved_pages", "kv_allocated_pages", "kv_reserved_pages",
    })

    @staticmethod
    def _capacity_error(error: Exception) -> bool:
        message = str(error)
        return "capacity" in message or "slot available" in message or \
            "credits" in message

    def generate(self, prompt_ids: list[int], maximum: int,
                 context: RequestContext | None = None,
                 cancel_check: Callable[[], bool] | None = None,
                 cache_prefix_tokens: int | None = None,
                 sampling: SamplingSettings | None = None,
                 ) -> Iterator[tuple[int, str]]:
        request_id = self.request_id()
        generated: list[int] = []
        decoder = IncrementalTextDecoder()
        started = time.monotonic()
        first_token_seconds: float | None = None
        previous_token_at: float | None = None
        session = context.session if context is not None else None
        retain = context is not None and self.retention_enabled()
        context_limit = len(prompt_ids) + maximum
        if retain:
            if context_limit < self.args.max_context:
                # Retained turns always end with a non-final decode step so
                # the slot survives; reserve one extra position for the
                # trailing feed (or speculative pair under MTP).
                context_limit += 1
            else:
                retain = False
        stats_before = self.worker_stats()
        prefill_tokens = len(prompt_ids)
        resumed = False
        finished = False
        deadline = started + self.args.generation_timeout
        checkpoint_tokens = cache_prefix_tokens or len(prompt_ids)
        effective_sampling = sampling or self.default_sampling
        while True:
            try:
                if session is not None:
                    delta = prompt_ids[len(session.tokens):]
                    if cancel_check is None:
                        self.worker.begin_resume(
                            request_id, session.key, delta, context_limit,
                            effective_sampling,
                        )
                    else:
                        self.worker.begin_resume(
                            request_id, session.key, delta, context_limit,
                            effective_sampling,
                            checkpoint_tokens=checkpoint_tokens,
                            cancel_check=cancel_check, deadline=deadline,
                        )
                    resumed = True
                    prefill_tokens = len(delta)
                else:
                    if cancel_check is None:
                        self.worker.begin(request_id, prompt_ids, context_limit,
                                          effective_sampling)
                    else:
                        self.worker.begin(
                            request_id, prompt_ids, context_limit,
                            effective_sampling,
                            checkpoint_tokens=checkpoint_tokens,
                            cancel_check=cancel_check, deadline=deadline,
                        )
                break
            except WorkerError as error:
                if self._capacity_error(error) and self.evict_lru_session():
                    continue
                if session is not None:
                    # The retained state is gone or inconsistent; fall back
                    # to a fresh full prefill.
                    log("session_resume_failed", key=session.key,
                        error=str(error))
                    self._drop_worker_session(session.key)
                    session = None
                    continue
                raise
        try:
            for index in range(maximum):
                if time.monotonic() - started > self.args.generation_timeout:
                    raise TimeoutError("generation deadline exceeded")
                # A final STEP makes the worker release the slot inline, which
                # is incompatible with retaining it; retained conversations
                # end with an explicit retaining END instead. Under MTP the
                # last step of a retained turn is a hold (mode 2): a
                # speculative pair could leave an unemitted bonus token in the
                # worker state, and no client-echoed prompt would match the
                # retained session afterwards.
                last = index + 1 == maximum
                token = self.decode_batcher.step(
                    request_id, last and not retain,
                    hold=last and retain and self.worker.mtp_enabled,
                )
                token_at = time.monotonic()
                if previous_token_at is None:
                    first_token_seconds = token_at - started
                    self.observe_latency("ttft_seconds", first_token_seconds)
                else:
                    self.observe_latency(
                        "inter_token_seconds", token_at - previous_token_at
                    )
                previous_token_at = token_at
                self.increment("generated_tokens")
                generated.append(token)
                current = self.tokenizer.decode(
                    generated, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                final_text = token in self.eos_token_ids or index + 1 == maximum
                delta = decoder.push(current, final=final_text)
                if final_text:
                    # The consumer closes the generator right after this
                    # final token; from here on the turn is complete and its
                    # state is safe to retain.
                    finished = True
                yield token, delta
                if token in self.eos_token_ids:
                    break
            finished = True
        finally:
            take_buffered = getattr(self.decode_batcher, "take_buffered", None)
            buffered = take_buffered(request_id) if take_buffered is not None \
                else []
            if finished and retain and not buffered:
                try:
                    session_key = session.key if session is not None \
                        else self.allocate_session_key()
                    try:
                        retained_tokens = self.worker.end_retain(
                            request_id, session_key, checkpoint_tokens
                        )
                    except TypeError as error:
                        if "positional" not in str(error):
                            raise
                        retained_tokens = self.worker.end_retain(
                            request_id, session_key
                        )
                    tokens = (prompt_ids + generated + buffered)[
                        :retained_tokens]
                    if 0 < retained_tokens == len(tokens):
                        self.store_session(session_key, tokens,
                                           context.held_pages)
                        context.retained = True
                        log("session_retained", key=session_key,
                            tokens=retained_tokens, resumed=resumed)
                    else:
                        self.worker.drop_session(session_key)
                        log("session_retain_mismatch", key=session_key,
                            retained_tokens=retained_tokens,
                            expected_tokens=len(prompt_ids) + len(generated) +
                            len(buffered))
                except WorkerError as error:
                    log("session_retain_failed", error=str(error))
            elif finished and retain:
                # An unemitted speculative bonus token remains buffered; the
                # worker state no longer matches any client-echoable prefix,
                # so this turn cannot be retained.
                log("session_retain_skipped", reason="speculative_bonus",
                    buffered_tokens=len(buffered))
            if request_id in self.worker.active_ids:
                self.worker.cancel(request_id)
            if context is not None and not context.retained and \
                    session is not None:
                self._drop_worker_session(session.key)
            stats_after = self.worker_stats()
            deltas = {
                key: stats_after[key] - stats_before[key]
                for key in stats_before.keys() & stats_after.keys()
                if key not in self._TELEMETRY_GAUGE_KEYS and
                isinstance(stats_before[key], int) and
                not isinstance(stats_before[key], bool) and
                isinstance(stats_after[key], int) and
                not isinstance(stats_after[key], bool) and
                stats_after[key] >= stats_before[key]
            }
            log("request_telemetry", request_id=request_id, resumed=resumed,
                prefill_tokens=prefill_tokens,
                generated_tokens=len(generated), finished=finished,
                wall_seconds=time.monotonic() - started,
                ttft_seconds=first_token_seconds, **deltas)

    def info(self) -> dict[str, Any]:
        with self.active_lock:
            active = self.active
        with self.session_lock:
            session_count = len(self.sessions)
            session_tokens = sum(len(session.tokens)
                                 for session in self.sessions.values())
            session_pages = sum(session.pages
                                for session in self.sessions.values())
        kv_stats = self.worker.stats()
        runtime_stats = {
            key: value for key, value in kv_stats.items()
            if key not in {"allocated_pages", "reserved_pages",
                           "kv_allocated_pages", "kv_reserved_pages"}
        }
        return {
            "model": self.args.model,
            "build_id": self.args.build_id,
            "source": self.manifest["source"],
            "format": self.manifest["format"],
            "quantization": self.manifest["quantization"],
            "manifest_content_sha256": self.manifest["integrity"]["content_sha256"],
            "dense_index_sha256": self.manifest["indexes"]["dense_sha256"],
            "experts_index_sha256": self.manifest["indexes"]["experts_sha256"],
            "architecture": self.manifest["architecture"],
            "masses": self.manifest["masses"],
            "active_requests": active,
            "maximum_queue": self.args.maximum_queue,
            "worker_capacity": self.args.worker_capacity,
            "worker_protocol": self.worker.protocol,
            "worker_model_descriptor": {
                "architecture_id": getattr(
                    self.worker, "architecture_id", ""
                ),
                "vocab_size": getattr(self.worker, "vocab_size", 0),
                "max_context_tokens": getattr(
                    self.worker, "model_max_context_tokens", 0
                ),
                "routed_layers": getattr(self.worker, "routed_layers", 0),
                "experts_per_layer": getattr(
                    self.worker, "experts_per_layer", 0
                ),
                "route_width": getattr(self.worker, "route_width", 0),
                "expert_encoding": getattr(
                    self.worker, "expert_encoding", ""
                ),
                "operation_capabilities": list(getattr(
                    self.worker, "operation_capabilities", ()
                )),
            },
            "worker_placement": {
                "mode": self.worker.placement_mode,
                "profile": self.worker.placement_profile,
                "ram_cache_bytes": self.worker.ram_cache_bytes,
                "vram_cache_bytes": self.worker.vram_cache_bytes,
                "prefetch_enabled": self.worker.placement_prefetch_enabled,
                "prefetch_state": self.worker.placement_prefetch_state,
                "minimum_recent_observations": (
                    self.worker.placement_minimum_observations
                ),
            },
            "worker_prefill": {
                "mode": self.worker.prefill_mode,
                "chunk_tokens": self.worker.prefill_chunk_tokens,
            },
            "worker_execution": {
                "request_stream_mode": self.worker.request_stream_mode,
                "rope_mode": self.worker.rope_mode,
                "gpu_phase_timing": self.worker.gpu_phase_timing,
                "mtp_resource_available": self.worker.mtp_resource_available,
                "mtp_runtime_ready": self.worker.mtp_runtime_ready,
                "mtp_enabled": self.worker.mtp_enabled,
                "retain_previous_route": self.worker.retain_previous_route,
                "cpu_hybrid_enabled": self.worker.cpu_hybrid_enabled,
                "response_protocol": self.response_protocol,
                "sampling": {
                    "temperature": self.default_sampling.temperature,
                    "top_p": self.default_sampling.top_p,
                    "top_k": self.default_sampling.top_k,
                    "min_p": self.default_sampling.min_p,
                    "source": "tokenizer/generation_config.json",
                },
            },
            "worker_kv": {
                "dtype": self.worker.kv_dtype,
                "allocation": self.worker.kv_allocation,
                "page_tokens": self.worker.kv_page_tokens,
                "page_bytes": self.worker.kv_page_bytes,
                "page_capacity": self.worker.kv_page_capacity,
                "allocated_pages": kv_stats["allocated_pages"],
                "reserved_pages": kv_stats["reserved_pages"],
            },
            "worker_sessions": {
                "enabled": self.retention_enabled(),
                "retained": session_count,
                "retained_tokens": session_tokens,
                "reserved_pages": session_pages,
            },
            "worker_runtime": runtime_stats,
            "runtime_config": {
                "host": self.args.host,
                "port": self.args.port,
                "max_context": self.args.max_context,
                "maximum_new_tokens": self.args.maximum_new_tokens,
                "maximum_queue": self.args.maximum_queue,
                "worker_capacity": self.args.worker_capacity,
                "worker_ram_cache_gib": self.args.worker_ram_cache_gib,
                "worker_vram_cache_gib": self.args.worker_vram_cache_gib,
                "placement_profile": self.args.placement_profile,
                "worker_kv_cache_mib": self.args.worker_kv_cache_mib,
                "worker_kv_page_tokens": self.args.worker_kv_page_tokens,
                "worker_prefill_chunk_tokens":
                    self.args.worker_prefill_chunk_tokens,
                "session_retention": self.retention_enabled(),
                "session_idle_seconds": self.args.session_idle_seconds,
                "microbatch_window_ms": self.args.microbatch_window_ms,
                "latency_window": self.args.latency_window,
                "queue_timeout_seconds": self.args.queue_timeout,
                "generation_timeout_seconds": self.args.generation_timeout,
                "maximum_body_bytes": self.args.maximum_body_bytes,
            },
            "draining": self.draining.is_set(),
        }

    def close(self) -> None:
        self.draining.set()
        deadline = time.monotonic() + self.args.drain_timeout
        while time.monotonic() < deadline:
            with self.active_lock:
                if self.active == 0:
                    break
            time.sleep(0.05)
        self.decode_batcher.close()
        with self.session_lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            self._drop_worker_session(session.key)
        self.worker.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "ExpertRuntime/2"

    @property
    def app(self) -> Application:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        log("http", client=self.client_address[0], message=format % args)

    def _authorized(self) -> bool:
        expected = self.app.args.api_key
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def _client_disconnected(self) -> bool:
        # A streaming response can fit in the kernel send buffer, so relying
        # only on a later BrokenPipeError may finish dozens of tokens after the
        # peer has already sent FIN/RST. A non-blocking peek observes that
        # closure before scheduling the next decode step.
        try:
            readable, _writable, _errors = select.select(
                [self.connection], [], [], 0
            )
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (ConnectionResetError, OSError, ValueError):
            return True

    def _request_id(self) -> str:
        value = getattr(self, "http_request_id", None)
        if value is None:
            value = "req_" + uuid.uuid4().hex
            self.http_request_id = value
        return value

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("x-request-id", self._request_id())
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error",
               param: str | None = None, code: str | None = None) -> None:
        self._json(status, {"error": {
            "message": message, "type": kind, "param": param, "code": code,
        }})

    def _sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("x-request-id", self._request_id())
        self.end_headers()

    def _sse(self, payload: dict[str, Any] | str) -> None:
        encoded = payload if isinstance(payload, str) else json.dumps(
            payload, separators=(",", ":")
        )
        self.wfile.write(b"data: " + encoded.encode() + b"\n\n")
        self.wfile.flush()

    def _model(self) -> dict[str, Any]:
        return {
            "id": self.app.args.model,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }

    @staticmethod
    def _usage(prompt_tokens: int, completion_tokens: int,
               reasoning_tokens: int = 0) -> dict[str, Any]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens, "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        }

    @staticmethod
    def _responses_usage(prompt_tokens: int, completion_tokens: int,
                         reasoning_tokens: int = 0) -> dict[str, Any]:
        return {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _run_generation(self, request: GenerationRequest, emit: Any,
                        context: RequestContext) -> tuple[str, int, str]:
        pieces: list[str] = []
        count = 0
        finish_reason = "length"
        stop_filter = StopFilter(request.stop)
        try:
            generation = self.app.generate(
                request.prompt_ids, request.maximum, context,
                cancel_check=self._client_disconnected,
                cache_prefix_tokens=request.cache_prefix_tokens,
                sampling=request.sampling,
            )
        except TypeError as error:
            if not any(name in str(error) for name in (
                    "cancel_check", "cache_prefix_tokens", "sampling")):
                raise
            generation = self.app.generate(
                request.prompt_ids, request.maximum, context
            )
        try:
            for token, delta in generation:
                if request.stream and self._client_disconnected():
                    raise BrokenPipeError("streaming client disconnected")
                count += 1
                safe = stop_filter.feed(delta)
                if safe:
                    pieces.append(safe)
                    emit(safe)
                if stop_filter.stopped or token in self.app.eos_token_ids:
                    finish_reason = "stop"
                    break
            tail = stop_filter.finish()
            if tail:
                pieces.append(tail)
                emit(tail)
        finally:
            generation.close()
        return "".join(pieces), count, finish_reason

    def _response_object(self, request: GenerationRequest, response_id: str,
                         message_id: str, created: int, text: str,
                         completion_tokens: int, status: str = "completed",
                         tool_calls: tuple[ToolCall, ...] = (),
                         reasoning_tokens: int = 0) -> dict[str, Any]:
        completed = status == "completed"
        output: list[dict[str, Any]] = []
        if completed and text:
            output.append({
                "id": message_id, "type": "message", "status": "completed",
                "role": "assistant", "content": [{
                    "type": "output_text", "text": text, "annotations": [],
                    "logprobs": [],
                }],
            })
        if completed:
            output.extend({
                "id": call.item_id,
                "call_id": call.call_id,
                "type": "function_call", "status": "completed",
                "name": call.name, "arguments": call.arguments,
            } for call in tool_calls)
        return {
            "id": response_id, "object": "response", "created_at": created,
            "status": status, "completed_at": int(time.time()) if completed else None,
            "error": None, "incomplete_details": None,
            "instructions": request.instructions,
            "max_output_tokens": request.maximum, "model": self.app.args.model,
            "output": output, "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": request.reasoning_effort, "summary": None},
            "store": False, "temperature": request.sampling.temperature,
            "text": {"format": {"type": "text"}},
            "tool_choice": request.tool_choice,
            "tools": list(request.tools), "top_p": request.sampling.top_p,
            "truncation": "disabled",
            "usage": self._responses_usage(
                len(request.prompt_ids), completion_tokens, reasoning_tokens
            )
                     if completed else None,
            "user": request.user, "metadata": request.metadata or {},
        }

    def do_GET(self) -> None:
        self.http_request_id = "req_" + uuid.uuid4().hex
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "invalid API key", "authentication_error",
                        code="invalid_api_key")
            return
        path = urlsplit(self.path).path
        if path == "/health":
            status = HTTPStatus.OK if self.app.worker.healthy() else HTTPStatus.SERVICE_UNAVAILABLE
            self._json(status, {"status": "ok" if status == 200 else "failed"})
        elif path == "/ready":
            ready = self.app.worker.healthy() and not self.app.draining.is_set()
            self._json(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                       {"ready": ready})
        elif path == "/model-info":
            self._json(HTTPStatus.OK, self.app.info())
        elif path == "/v1/models":
            self._json(HTTPStatus.OK, {"object": "list", "data": [self._model()]})
        elif path.startswith("/v1/models/"):
            model = unquote(path[len("/v1/models/"):])
            if model != self.app.args.model:
                self._error(HTTPStatus.NOT_FOUND, f"model {model!r} was not found",
                            "invalid_request_error", "model", "model_not_found")
            else:
                self._json(HTTPStatus.OK, self._model())
        elif path == "/metrics":
            encoded = self.app.metrics_text().encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("x-request-id", self._request_id())
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self._error(HTTPStatus.NOT_FOUND, "route not found", code="not_found")

    def do_POST(self) -> None:
        self.http_request_id = "req_" + uuid.uuid4().hex
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "invalid API key", "authentication_error",
                        code="invalid_api_key")
            return
        path = urlsplit(self.path).path
        endpoint = {
            "/v1/completions": "completion",
            "/v1/chat/completions": "chat",
            "/v1/responses": "responses",
        }.get(path)
        if endpoint is None:
            self._error(HTTPStatus.NOT_FOUND, "route not found", code="not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.app.args.maximum_body_bytes:
                raise RequestError("request body size is invalid", code="invalid_request_body")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise RequestError("request body must be an object", code="invalid_request_body")
            request = self.app.parse_request(payload, endpoint)
        except RequestError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error), param=error.param, code=error.code)
            return
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error), code="invalid_json")
            return
        except Exception as error:
            log("request_preprocessing_failed", error=str(error), endpoint=endpoint)
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "request preprocessing failed",
                "server_error",
                code="request_preprocessing_failed",
            )
            return
        if not self.app.acquire():
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "service overloaded or draining",
                        "server_error", code="overloaded")
            return
        if not self.app.acquire_worker_slot():
            self.app.release()
            self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                        "all model slots are busy", "server_error", code="overloaded")
            return
        context = self.app.acquire_request_context(
            request.prompt_ids, request.maximum
        )
        if context is None:
            self.app.release_worker_slot()
            self.app.release()
            self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                        "KV context capacity is exhausted", "server_error",
                        code="context_capacity_exhausted")
            return
        prefix = {"chat": "chatcmpl-", "completion": "cmpl-", "responses": "resp_"}[endpoint]
        request_uuid = prefix + uuid.uuid4().hex
        message_uuid = "msg_" + uuid.uuid4().hex
        created = int(time.time())
        stream_started = False
        try:
            if request.stream:
                self._sse_headers()
                stream_started = True
                if endpoint == "responses":
                    sequence = 0
                    initial = self._response_object(
                        request, request_uuid, message_uuid, created, "", 0, "in_progress"
                    )
                    self._sse({"type": "response.created", "response": initial,
                               "sequence_number": sequence})
                    sequence += 1
                    calls: tuple[ToolCall, ...] = ()
                    reasoning_text = ""
                    if request.tools and request.tool_choice != "none":
                        pieces: list[str] = []
                        raw_text, completion_count, _finish_reason = self._run_generation(
                            request, pieces.append, context
                        )
                        parsed = self.app.parse_assistant_output(raw_text, request)
                        text, calls = parsed.text, parsed.tool_calls
                        reasoning_text = parsed.reasoning
                    else:
                        item = {"id": message_uuid, "type": "message",
                                "status": "in_progress", "role": "assistant",
                                "content": []}
                        self._sse({"type": "response.output_item.added",
                                   "output_index": 0, "item": item,
                                   "sequence_number": sequence})
                        sequence += 1
                        part = {"type": "output_text", "text": "",
                                "annotations": [], "logprobs": []}
                        self._sse({"type": "response.content_part.added",
                                   "item_id": message_uuid, "output_index": 0,
                                   "content_index": 0, "part": part,
                                   "sequence_number": sequence})
                        sequence += 1

                        reasoning_filter = AssistantStreamParser(
                            self.app, request
                        )
                        visible_pieces: list[str] = []
                        reasoning_pieces: list[str] = []

                        def emit_response(delta: str) -> None:
                            nonlocal sequence
                            if self._client_disconnected():
                                raise BrokenPipeError("streaming client disconnected")
                            reasoning, visible = reasoning_filter.feed(delta)
                            if reasoning:
                                reasoning_pieces.append(reasoning)
                            if visible:
                                visible_pieces.append(visible)
                                self._sse({"type": "response.output_text.delta",
                                           "item_id": message_uuid,
                                           "output_index": 0,
                                           "content_index": 0, "delta": visible,
                                           "logprobs": [],
                                           "sequence_number": sequence})
                                sequence += 1

                        _raw_text, completion_count, _finish_reason = self._run_generation(
                            request, emit_response, context
                        )
                        reasoning_tail, visible_tail = reasoning_filter.finish()
                        if reasoning_tail:
                            reasoning_pieces.append(reasoning_tail)
                        if visible_tail:
                            visible_pieces.append(visible_tail)
                            self._sse({"type": "response.output_text.delta",
                                       "item_id": message_uuid,
                                       "output_index": 0,
                                       "content_index": 0,
                                       "delta": visible_tail, "logprobs": [],
                                       "sequence_number": sequence})
                            sequence += 1
                        text = "".join(visible_pieces)
                        reasoning_text = "".join(reasoning_pieces)

                    output_index = 0
                    if text or not calls:
                        if request.tools and request.tool_choice != "none":
                            item = {"id": message_uuid, "type": "message",
                                    "status": "in_progress", "role": "assistant",
                                    "content": []}
                            self._sse({"type": "response.output_item.added",
                                       "output_index": output_index, "item": item,
                                       "sequence_number": sequence})
                            sequence += 1
                            part = {"type": "output_text", "text": "",
                                    "annotations": [], "logprobs": []}
                            self._sse({"type": "response.content_part.added",
                                       "item_id": message_uuid,
                                       "output_index": output_index,
                                       "content_index": 0, "part": part,
                                       "sequence_number": sequence})
                            sequence += 1
                            if text:
                                self._sse({"type": "response.output_text.delta",
                                           "item_id": message_uuid,
                                           "output_index": output_index,
                                           "content_index": 0, "delta": text,
                                           "logprobs": [],
                                           "sequence_number": sequence})
                                sequence += 1
                        self._sse({"type": "response.output_text.done",
                                   "item_id": message_uuid,
                                   "output_index": output_index,
                                   "content_index": 0, "text": text,
                                   "logprobs": [], "sequence_number": sequence})
                        sequence += 1
                        done_part = {"type": "output_text", "text": text,
                                     "annotations": [], "logprobs": []}
                        self._sse({"type": "response.content_part.done",
                                   "item_id": message_uuid,
                                   "output_index": output_index,
                                   "content_index": 0, "part": done_part,
                                   "sequence_number": sequence})
                        sequence += 1
                        done_item = {"id": message_uuid, "type": "message",
                                     "status": "completed", "role": "assistant",
                                     "content": [done_part]}
                        self._sse({"type": "response.output_item.done",
                                   "output_index": output_index, "item": done_item,
                                   "sequence_number": sequence})
                        sequence += 1
                        output_index += 1
                    for call in calls:
                        call_item = {
                            "id": call.item_id,
                            "call_id": call.call_id, "type": "function_call",
                            "status": "in_progress", "name": call.name,
                            "arguments": "",
                        }
                        self._sse({"type": "response.output_item.added",
                                   "output_index": output_index,
                                   "item": call_item,
                                   "sequence_number": sequence})
                        sequence += 1
                        self._sse({"type": "response.function_call_arguments.done",
                                   "item_id": call_item["id"],
                                   "output_index": output_index,
                                   "arguments": call.arguments,
                                   "sequence_number": sequence})
                        sequence += 1
                        call_item = {**call_item, "status": "completed",
                                     "arguments": call.arguments}
                        self._sse({"type": "response.output_item.done",
                                   "output_index": output_index,
                                   "item": call_item,
                                   "sequence_number": sequence})
                        sequence += 1
                        output_index += 1
                    completed = self._response_object(
                        request, request_uuid, message_uuid, created, text,
                        completion_count, tool_calls=calls,
                        reasoning_tokens=self.app.text_token_count(reasoning_text),
                    )
                    self._sse({"type": "response.completed", "response": completed,
                               "sequence_number": sequence})
                else:
                    chat = endpoint == "chat"
                    base = {"id": request_uuid,
                            "object": "chat.completion.chunk" if chat else "text_completion",
                            "created": created, "model": self.app.args.model,
                            "system_fingerprint": "fp_" + self.app.args.build_id}
                    if chat:
                        self._sse({**base, "choices": [{"index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None, "finish_reason": None}]})
                    reasoning_filter = AssistantStreamParser(
                        self.app, request
                    ) if chat else None
                    reasoning_pieces: list[str] = []

                    def emit_completion(delta: str) -> None:
                        if self._client_disconnected():
                            raise BrokenPipeError("streaming client disconnected")
                        if chat:
                            assert reasoning_filter is not None
                            reasoning, visible = reasoning_filter.feed(delta)
                            if reasoning:
                                reasoning_pieces.append(reasoning)
                                self._sse({**base, "choices": [{
                                    "index": 0, "finish_reason": None,
                                    "logprobs": None,
                                    "delta": {"reasoning_content": reasoning},
                                }]})
                            if visible:
                                self._sse({**base, "choices": [{
                                    "index": 0, "finish_reason": None,
                                    "logprobs": None,
                                    "delta": {"content": visible},
                                }]})
                        else:
                            self._sse({**base, "choices": [{
                                "index": 0, "finish_reason": None,
                                "logprobs": None, "text": delta,
                            }]})

                    calls: tuple[ToolCall, ...] = ()
                    if chat and request.tools and request.tool_choice != "none":
                        pieces: list[str] = []
                        raw_text, completion_count, finish_reason = self._run_generation(
                            request, pieces.append, context
                        )
                        parsed = self.app.parse_assistant_output(raw_text, request)
                        calls = parsed.tool_calls
                        if parsed.reasoning:
                            reasoning_pieces.append(parsed.reasoning)
                            self._sse({**base, "choices": [{
                                "index": 0, "finish_reason": None,
                                "logprobs": None,
                                "delta": {"reasoning_content": parsed.reasoning},
                            }]})
                        if parsed.text:
                            self._sse({**base, "choices": [{
                                "index": 0, "finish_reason": None,
                                "logprobs": None,
                                "delta": {"content": parsed.text},
                            }]})
                        if calls:
                            self._sse({**base, "choices": [{
                                "index": 0, "finish_reason": None,
                                "logprobs": None, "delta": {"tool_calls": [{
                                    "index": index, "id": call.call_id,
                                    "type": "function", "function": {
                                        "name": call.name,
                                        "arguments": call.arguments,
                                    },
                                } for index, call in enumerate(calls)]},
                            }]})
                            finish_reason = "tool_calls"
                    else:
                        _text, completion_count, finish_reason = self._run_generation(
                            request, emit_completion, context
                        )
                        if reasoning_filter is None:
                            reasoning_tail, visible_tail = "", ""
                        else:
                            reasoning_tail, visible_tail = reasoning_filter.finish()
                        if chat and reasoning_tail:
                            reasoning_pieces.append(reasoning_tail)
                            self._sse({**base, "choices": [{
                                "index": 0, "finish_reason": None,
                                "logprobs": None,
                                "delta": {"reasoning_content": reasoning_tail},
                            }]})
                        if chat and visible_tail:
                            self._sse({**base, "choices": [{
                                "index": 0, "finish_reason": None,
                                "logprobs": None,
                                "delta": {"content": visible_tail},
                            }]})
                    final_choice: dict[str, Any] = {
                        "index": 0, "finish_reason": finish_reason, "logprobs": None,
                    }
                    final_choice["delta" if chat else "text"] = {} if chat else ""
                    self._sse({**base, "choices": [final_choice]})
                    if request.include_usage:
                        self._sse({**base, "choices": [],
                                   "usage": self._usage(len(request.prompt_ids),
                                        completion_count,
                                        self.app.text_token_count(
                                            "".join(reasoning_pieces)
                                        ))})
                    self._sse("[DONE]")
            else:
                pieces: list[str] = []
                text, completion_count, finish_reason = self._run_generation(
                    request, pieces.append, context
                )
                calls: tuple[ToolCall, ...] = ()
                reasoning = ""
                if endpoint in {"chat", "responses"}:
                    parsed = self.app.parse_assistant_output(text, request)
                    text, reasoning, calls = (
                        parsed.text, parsed.reasoning, parsed.tool_calls
                    )
                    if calls:
                        finish_reason = "tool_calls"
                if endpoint == "responses":
                    self._json(HTTPStatus.OK, self._response_object(
                        request, request_uuid, message_uuid, created, text,
                        completion_count, tool_calls=calls,
                        reasoning_tokens=self.app.text_token_count(reasoning),
                    ))
                else:
                    chat = endpoint == "chat"
                    choice: dict[str, Any] = {
                        "index": 0, "finish_reason": finish_reason, "logprobs": None,
                    }
                    if chat:
                        message: dict[str, Any] = {
                            "role": "assistant", "content": text or None,
                            "refusal": None, "annotations": [],
                        }
                        if reasoning:
                            message["reasoning_content"] = reasoning
                        if calls:
                            message["tool_calls"] = [{
                                "id": call.call_id, "type": "function",
                                "function": {"name": call.name,
                                             "arguments": call.arguments},
                            } for call in calls]
                        choice["message"] = message
                    else:
                        choice["text"] = text
                    self._json(HTTPStatus.OK, {"id": request_uuid,
                        "object": "chat.completion" if chat else "text_completion",
                        "created": created, "model": self.app.args.model,
                        "system_fingerprint": "fp_" + self.app.args.build_id,
                        "choices": [choice],
                        "usage": self._usage(
                            len(request.prompt_ids), completion_count,
                            self.app.text_token_count(reasoning),
                        ),
                        "service_tier": "default"})
            self.app.increment("completed")
        except (BrokenPipeError, ConnectionResetError):
            self.app.increment("cancelled")
            log("request_cancelled", id=request_uuid, reason="client_disconnect")
        except TimeoutError as error:
            self.app.increment("failed")
            if not stream_started:
                self._error(HTTPStatus.GATEWAY_TIMEOUT, str(error), "timeout_error")
            elif endpoint == "responses":
                self._sse({"type": "error", "code": "generation_timeout",
                           "message": str(error), "param": None})
            else:
                self._sse({"error": {"message": str(error),
                           "type": "timeout_error", "param": None,
                           "code": "generation_timeout"}})
                self._sse("[DONE]")
        except Exception as error:
            self.app.increment("failed")
            log("request_failed", id=request_uuid, error=repr(error))
            if not stream_started:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "generation failed",
                            "server_error", code="generation_failed")
            elif endpoint == "responses":
                self._sse({"type": "error", "code": "generation_failed",
                           "message": "generation failed", "param": None})
            else:
                self._sse({"error": {"message": "generation failed",
                           "type": "server_error", "param": None,
                           "code": "generation_failed"}})
                self._sse("[DONE]")
        finally:
            self.app.release_request_context(context)
            self.app.release_worker_slot()
            self.app.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--container", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument(
        "--model", default="expert-moe-vm"
    )
    parser.add_argument("--build-id", default=os.environ.get("EXPERT_BUILD_ID", "development"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default=os.environ.get("EXPERT_API_KEY", ""))
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--maximum-new-tokens", type=int, default=512)
    parser.add_argument("--maximum-queue", type=int, default=8)
    parser.add_argument("--worker-capacity", type=int, default=1)
    parser.add_argument("--worker-ram-cache-gib", type=int, default=48)
    parser.add_argument("--worker-vram-cache-gib", type=int, default=13)
    parser.add_argument(
        "--placement-profile", choices=("latency", "balanced", "capacity"),
        default="balanced",
    )
    parser.add_argument("--worker-kv-cache-mib", type=int, default=2048)
    parser.add_argument("--worker-kv-page-tokens", type=int, default=256)
    parser.add_argument(
        "--worker-prefill-chunk-tokens", type=int, default=0,
        help="maximum provider prefill chunk (0 uses the provider contract)",
    )
    parser.add_argument(
        "--worker-placement-settle-steps", type=int, default=None,
        help="decode steps before the Qwen worker freezes adaptive placement "
             "(0 disables the freeze; omitted keeps the worker default)",
    )
    parser.add_argument("--disable-session-retention", action="store_true")
    parser.add_argument("--session-idle-seconds", type=float, default=1800.0)
    parser.add_argument("--profile-gpu-phases", action="store_true")
    parser.add_argument(
        "--disable-worker-retained-route", action="store_true",
        help="disable DeepSeek previous-route VRAM leases for placement sweeps",
    )
    parser.add_argument(
        "--enable-worker-cpu-hybrid", action="store_true",
        help="allow a provider to execute routed misses on CPU",
    )
    parser.add_argument(
        "--worker-route-trace-file", type=Path,
        help="write bounded exact DeepSeek route traces as JSONL",
    )
    parser.add_argument(
        "--worker-route-trace-max-steps", type=int, default=4096,
        help="maximum exact model steps retained per request turn",
    )
    parser.add_argument("--microbatch-window-ms", type=float, default=2.0)
    parser.add_argument("--latency-window", type=int, default=4096)
    parser.add_argument("--maximum-body-bytes", type=int, default=1 << 20)
    parser.add_argument("--queue-timeout", type=float, default=1.0)
    parser.add_argument("--generation-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--drain-timeout", type=float, default=30.0)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> int:
    global LOG_FILE
    args = parse_args()
    if args.worker_route_trace_max_steps < 1:
        raise SystemExit("--worker-route-trace-max-steps must be positive")
    if (args.maximum_queue < 0 or args.max_context < 2 or
        args.maximum_new_tokens < 1 or args.worker_capacity < 1 or
        args.worker_ram_cache_gib < 1 or args.worker_vram_cache_gib < 1 or
        args.worker_kv_cache_mib < 1 or args.worker_kv_page_tokens < 1 or
        args.worker_prefill_chunk_tokens < 0 or
        (args.worker_placement_settle_steps is not None and
         args.worker_placement_settle_steps < 0) or
        args.session_idle_seconds < 0 or args.maximum_body_bytes < 1 or
        args.microbatch_window_ms < 0 or args.latency_window < 1):
        raise SystemExit("invalid service limits")
    if (args.host not in {"127.0.0.1", "::1", "localhost"} and
            not args.api_key):
        raise SystemExit("--api-key or EXPERT_API_KEY is required for non-loopback bind")
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE = args.log_file.open("a", encoding="utf-8")
    try:
        app = Application(args)
    except Exception as error:
        log("service_start_failed", error=str(error))
        raise
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app  # type: ignore[attr-defined]
    server.daemon_threads = True

    def stop(_signal: int, _frame: Any) -> None:
        app.draining.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        app.close()
        if LOG_FILE is not None:
            LOG_FILE.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
