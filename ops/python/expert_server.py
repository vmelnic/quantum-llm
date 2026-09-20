#!/usr/bin/env python3
"""Bounded OpenAI-compatible HTTP front-end for the persistent CUDA worker."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import queue
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
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
    from .artifact_chat_codec import discover_artifact_chat_codec
    from .response_protocols import select_response_protocol
    from .multimodal_input import (
        MultimodalInputError, PreparedMultimodalPrompt,
        create_image_processor, load_image, prepare_multimodal_prompt,
    )
except ImportError:  # Direct script launch from Start-ExpertServer.ps1.
    from artifact_chat_codec import discover_artifact_chat_codec
    from response_protocols import select_response_protocol
    from multimodal_input import (
        MultimodalInputError, PreparedMultimodalPrompt,
        create_image_processor, load_image, prepare_multimodal_prompt,
    )

from transformers import AutoTokenizer


LOG_FILE: Any = None

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


def _configured_eos_token_ids(
        generation_config: Mapping[str, Any], tokenizer_eos: Any) -> set[int]:
    """Resolve the checkpoint's complete, validated generation stop set."""
    value = generation_config.get("eos_token_id")
    if value is None:
        value = tokenizer_eos
    if value is None:
        return set()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    if not values or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in values):
        raise RuntimeError("generation_config eos_token_id is invalid")
    return set(values)


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
    presence_penalty: float = 0.0

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
    media_packet: bytes | None = None
    media_signature: bytes | None = None
    image_count: int = 0
    image_tokens: int = 0


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
    """Own one artifact-declared parser for an entire streamed response."""

    def __init__(self, application: "Application",
                 request: GenerationRequest) -> None:
        self.application = application
        self.request = request
        self.parser = application.response_stream_parser(request)
        self.raw_pieces: list[str] = []

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
        self.raw_pieces.append(delta)
        if self.parser is None:
            return "", delta
        return self._deltas(self.parser.feed(delta))

    def _trace(self, output: AssistantOutput, *,
               error: str | None = None) -> None:
        trace = getattr(self.application, "trace_assistant_response", None)
        if callable(trace):
            trace(
                self.request, "".join(self.raw_pieces), output, error=error,
            )

    def finish(self) -> tuple[str, str, AssistantOutput]:
        if self.parser is None:
            output = AssistantOutput(
                text="", reasoning="", reasoning_complete=True, tool_calls=()
            )
            self._trace(output)
            return "", "", output
        try:
            message, events = self.parser.finalize()
        except (AssertionError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            log("response_stream_finalize_failed",
                protocol=self.application.response_protocol,
                error=str(error))
            output = AssistantOutput(
                text="", reasoning="", reasoning_complete=False, tool_calls=()
            )
            self._trace(output, error=str(error))
            return "", "", output
        output = self.application.assistant_output_from_parsed_message(
            message, "".join(self.raw_pieces), fallback_text=""
        )
        self._trace(output)
        reasoning, content = self._deltas(events)
        return reasoning, content, output


class StreamingTextField:
    """Normalize a streamed parser text field without reparsing the response.

    Transformers' text response field strips its outer whitespace at close.
    Discard leading whitespace and hold a bounded trailing run until later
    content makes it internal. The streamed deltas remain authoritative; the
    final parsed message is used only for structured fields such as tool calls.
    """

    def __init__(self, maximum_ambiguous_whitespace: int = 4096) -> None:
        self.maximum_ambiguous_whitespace = maximum_ambiguous_whitespace
        self.pending_trailing_whitespace = ""
        self.discarded_leading_whitespace = 0
        self.started = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        combined = self.pending_trailing_whitespace + text
        self.pending_trailing_whitespace = ""
        if not self.started:
            stripped = combined.lstrip()
            self.discarded_leading_whitespace += len(combined) - len(stripped)
            if self.discarded_leading_whitespace > \
                    self.maximum_ambiguous_whitespace:
                raise WorkerError(
                    "response parser produced excessive leading whitespace"
                )
            combined = stripped
        stable = combined.rstrip()
        self.pending_trailing_whitespace = combined[len(stable):]
        if len(self.pending_trailing_whitespace) > \
                self.maximum_ambiguous_whitespace:
            raise WorkerError(
                "response parser produced excessive trailing whitespace"
            )
        if not stable:
            return ""
        self.started = True
        return stable

    def finish(self) -> None:
        """Discard whitespace that stayed outer through the final boundary."""
        self.pending_trailing_whitespace = ""


@dataclass
class Session:
    """A retained worker-side conversation prefix (LRU-managed)."""

    key: int
    tokens: list[int]
    pages: int
    last_used: float
    media_signature: bytes | None = None
    parked_bytes: int = 0
    persistent_id: str | None = None
    snapshot_generation: int = 0
    snapshot_path: Path | None = None
    snapshot_metadata: dict[str, int | bool] | None = None
    disk_bytes: int = 0
    resident: bool = True
    persistent_last_used: float = 0.0


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


class IncrementalTokenDecoder:
    """Bounded exact tokenizer decoding with a retained token overlap."""

    def __init__(self, tokenizer: Any, overlap_tokens: int = 64,
                 maximum_window_tokens: int = 256,
                 preserve_special_tokens: bool = False,
                 suppressed_token_ids: set[int] | None = None) -> None:
        if overlap_tokens < 1 or maximum_window_tokens <= overlap_tokens:
            raise ValueError("invalid incremental token decoder window")
        self.tokenizer = tokenizer
        self.overlap_tokens = overlap_tokens
        self.maximum_window_tokens = maximum_window_tokens
        self.preserve_special_tokens = preserve_special_tokens
        self.suppressed_token_ids = suppressed_token_ids or set()
        self.tokens: list[int] = []
        self.emitted = ""
        self.decode_prefix = ""

    def _decode_raw(self, tokens: list[int]) -> str:
        if self.suppressed_token_ids:
            tokens = [
                token for token in tokens
                if token not in self.suppressed_token_ids
            ]
        return self.tokenizer.decode(
            tokens, skip_special_tokens=not self.preserve_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def _decode(self, tokens: list[int]) -> str:
        current = self._decode_raw(tokens)
        if not current.startswith(self.decode_prefix):
            raise WorkerError(
                "tokenizer changed the retained streaming context"
            )
        return current[len(self.decode_prefix):]

    def _compact(self) -> bool:
        if len(self.tokens) <= self.overlap_tokens:
            return True
        suffix_tokens = self.tokens[-self.overlap_tokens:]
        raw_suffix = self._decode_raw(suffix_tokens)
        maximum = min(len(self.emitted), len(raw_suffix))
        common = 0
        while (common < maximum and
               self.emitted[-common - 1] == raw_suffix[-common - 1]):
            common += 1
        # A tiny coincidental suffix is not an authenticated boundary. Keep
        # the larger window and retry later unless at least half of the
        # retained decode is known to be the exact already-emitted suffix.
        if common == 0 or common * 2 < len(raw_suffix):
            return False
        self.tokens = suffix_tokens
        self.decode_prefix = raw_suffix[:-common]
        self.emitted = raw_suffix[-common:]
        return True

    def push(self, token: int, final: bool = False) -> str:
        if len(self.tokens) >= self.maximum_window_tokens and not self._compact():
            raise WorkerError(
                "tokenizer has no bounded stable streaming boundary"
            )
        self.tokens.append(token)
        current = self._decode(self.tokens)
        if not current.startswith(self.emitted):
            raise WorkerError("tokenizer changed an already streamed text prefix")
        stable_end = len(current)
        if not final:
            replacement = current.find("\ufffd", len(self.emitted))
            if replacement >= 0:
                stable_end = replacement
        delta = current[len(self.emitted):stable_end]
        self.emitted = current[:stable_end]
        if (stable_end == len(current) and
                len(self.tokens) > 2 * self.overlap_tokens):
            self._compact()
        if len(self.tokens) > self.maximum_window_tokens:
            raise WorkerError(
                "tokenizer has no bounded stable streaming boundary"
            )
        return delta


def _text_content(content: Any, param: str, *, allow_images: bool = False,
                  user_content: bool = False) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("message content must be text or an array of text parts", param)
    pieces: list[str] = []
    normalized: list[dict[str, Any]] = []
    has_image = False
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise RequestError("message content parts must be objects", f"{param}.{index}")
        kind = part.get("type")
        if kind in {"image_url", "input_image"}:
            if not allow_images or not user_content:
                raise RequestError(
                    "image input is not supported by the active artifact",
                    f"{param}.{index}.type", "unsupported_value",
                )
            reference = part.get("image_url")
            if isinstance(reference, dict):
                reference = reference.get("url")
            if not isinstance(reference, str):
                raise RequestError(
                    "image content requires image_url",
                    f"{param}.{index}.image_url",
                )
            try:
                image = load_image(reference)
            except MultimodalInputError as error:
                raise RequestError(
                    str(error), f"{param}.{index}.image_url"
                ) from error
            normalized.append({"type": "image", "image": image})
            has_image = True
            continue
        if kind not in {"text", "input_text", "output_text"}:
            raise RequestError(
                f"content type {kind!r} is not supported by this text-only model",
                f"{param}.{index}.type", "unsupported_value",
            )
        if not isinstance(part.get("text"), str):
            raise RequestError("text content part requires a string", f"{param}.{index}.text")
        pieces.append(part["text"])
        normalized.append({"type": "text", "text": part["text"]})
    return normalized if has_image else "".join(pieces)


class CudaWorker:
    def __init__(self, executable: Path, container: Path, max_context: int,
                 startup_timeout: float, requested_capacity: int,
                 ram_cache_gib: int, vram_cache_gib: int,
                 kv_cache_mib: int, kv_page_tokens: int,
                 placement_profile: str, profile_gpu_phases: bool,
                 kv_cache_dtype: str = "artifact",
                 prefill_chunk_tokens: int = 0,
                 placement_settle_steps: int | None = None,
                 retain_previous_route: bool | None = None,
                 enable_cpu_hybrid: bool | None = None,
                 route_trace_file: Path | None = None,
                 route_trace_max_steps: int = 4096,
                 active_expert_devices: str = "",
                 active_expert_device_cache_gib: int = 0,
                 active_expert_host_cache_gib: int = 0,
                 routed_vram_policy: str = "fixed") -> None:
        command = [
            str(executable), str(container), "--worker",
            f"--max-context={max_context}",
            f"--ram-cache-gib={ram_cache_gib}",
            f"--vram-cache-gib={vram_cache_gib}",
            f"--routed-vram-policy={routed_vram_policy}",
            f"--capacity={requested_capacity}",
            f"--kv-cache-mib={kv_cache_mib}",
            f"--kv-page-tokens={kv_page_tokens}",
            f"--kv-cache-dtype={kv_cache_dtype}",
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
        if active_expert_devices:
            command.extend((
                f"--active-expert-devices={active_expert_devices}",
                "--active-expert-device-cache-gib="
                f"{active_expert_device_cache_gib}",
                "--active-expert-host-cache-gib="
                f"{active_expert_host_cache_gib}",
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
        self.session_parking = bool(response.get("session_parking", False))
        self.session_persistence = bool(
            response.get("session_persistence", False)
        )
        self.session_park_ram_bytes = int(
            response.get("session_park_ram_bytes", 0)
        )
        self.session_park_page_capacity = int(
            response.get("session_park_page_capacity", 0)
        )
        self.request_stream_mode = str(
            response.get("request_stream_mode", "default")
        )
        self.gpu_phase_timing = bool(response.get("gpu_phase_timing", False))
        self.sampling_supported = bool(
            response.get("sampling_supported", False)
        )
        self.sampling_presence_penalty_supported = bool(
            response.get("sampling_presence_penalty_supported", False)
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
        self.routed_vram_policy = str(
            response.get("routed_vram_policy", "fixed")
        )
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
                    not 0 < self.ram_cache_bytes <= ram_cache_gib << 30 or
                    not 0 < self.vram_cache_bytes or
                    (routed_vram_policy == "fixed" and
                     self.vram_cache_bytes > vram_cache_gib << 30) or
                    self.routed_vram_policy != routed_vram_policy or
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
            self.placement_mode not in {"budgeted", "resident"} or
            (self.protocol >= 11 and
             self.routed_vram_policy not in {"fixed", "fit"})
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
        parking_invalid = self.protocol >= 9 and (
            (
                self.session_parking and
                (
                    not self.session_retention or
                    not 0 < self.session_park_ram_bytes <=
                        ram_cache_gib << 30 or
                    self.session_park_page_capacity <= 0
                )
            ) or
            (
                not self.session_parking and
                (self.session_park_ram_bytes != 0 or
                 self.session_park_page_capacity != 0)
            )
        )
        persistence_invalid = self.protocol >= 12 and (
            self.session_persistence and
            (not self.session_retention or not self.session_parking)
        )
        sampling_contract_invalid = self.protocol >= 10 and (
            self.sampling_presence_penalty_supported !=
            self.sampling_supported
        )
        if (self.protocol < 4 or descriptor_invalid or
                sampling_contract_invalid or
                parking_invalid or persistence_invalid or
                self.capacity != requested_capacity or
                self.prefill_mode not in {
                    "causal_blocked_exact",
                    "causal_chunked",
                    "causal_layer_major",
                    "causal_sequential",
                } or
                not 1 <= self.prefill_chunk_tokens <= max_context or
                (prefill_chunk_tokens and
                 self.prefill_chunk_tokens > prefill_chunk_tokens) or
                self.kv_dtype not in {
                    "fp16", "bf16", "bf16-latent", "fp32",
                    "fp4-e2m1-ue8m0-block32",
                    "fp4-e2m1-ue8m0-block32-key-outlier1",
                    "q4-bfp16-block32-key-outlier1",
                    "q4-bfp16-block32",
                    "q4-f16-per-head",
                    "q5-q4-bfp16-block32",
                    "fp8-e4m3-per-head",
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
                 deadline: float | None = None,
                 progress_callback: Callable[[], None] | None = None,
                 ) -> dict[str, Any]:
        with self.command_lock:
            if self.process.poll() is not None:
                raise WorkerError("CUDA worker is not running")
            assert self.process.stdin
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            interrupted = False
            progress_error: Exception | None = None
            while True:
                if not interrupted and (
                        (cancel_check is not None and cancel_check()) or
                        (deadline is not None and time.monotonic() > deadline)):
                    if cancel_id is not None:
                        self.process.stdin.write(f"CANCEL\t{cancel_id}\n")
                        self.process.stdin.flush()
                    interrupted = True
                if not interrupted and progress_callback is not None:
                    try:
                        progress_callback()
                    except Exception as error:
                        if cancel_id is not None:
                            self.process.stdin.write(f"CANCEL\t{cancel_id}\n")
                            self.process.stdin.flush()
                        interrupted = True
                        progress_error = error
                try:
                    response = self._responses.get(timeout=0.05)
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise WorkerError("CUDA worker exited while awaiting response")
                    continue
                if isinstance(response, Exception):
                    raise response
                if interrupted:
                    if progress_error is not None:
                        raise progress_error
                    raise WorkerError("CUDA worker request was cancelled")
                return response

    def begin(self, request_id: int, prompt_ids: list[int], context_limit: int,
              sampling: SamplingSettings,
              media_packet: bytes | None = None,
              checkpoint_tokens: int | None = None,
              cancel_check: Callable[[], bool] | None = None,
              deadline: float | None = None,
              progress_callback: Callable[[], None] | None = None) -> None:
        if request_id in self.active_ids:
            raise WorkerError("duplicate worker request")
        command = (f"BEGIN\t{request_id}\t{context_limit}\t" +
                   ",".join(str(token) for token in prompt_ids))
        media_path: str | None = None
        if media_packet is not None:
            with tempfile.NamedTemporaryFile(
                    mode="wb", prefix="quantum-llm-media-", suffix=".bin",
                    delete=False) as media_file:
                media_file.write(media_packet)
                media_file.flush()
                os.fsync(media_file.fileno())
                media_path = media_file.name
            if "\t" in media_path or "\n" in media_path or "\r" in media_path:
                os.unlink(media_path)
                raise WorkerError("temporary media path is not protocol safe")
            command += f"\tMULTIMODAL\t{media_path}"
        if checkpoint_tokens is not None:
            command += f"\tCHECKPOINT\t{checkpoint_tokens}"
        command += self._sampling_command(sampling)
        try:
            response = self._command(
                command,
                cancel_check=cancel_check, cancel_id=request_id,
                deadline=deadline, progress_callback=progress_callback,
            )
        finally:
            if media_path is not None:
                try:
                    os.unlink(media_path)
                except FileNotFoundError:
                    pass
        if response.get("type") != "begun" or response.get("id") != request_id:
            raise WorkerError("unexpected BEGIN response")
        self.active_ids.add(request_id)

    def begin_resume(self, request_id: int, session_key: int,
                     delta_ids: list[int], context_limit: int,
                     sampling: SamplingSettings,
                     checkpoint_tokens: int | None = None,
                     cancel_check: Callable[[], bool] | None = None,
                     deadline: float | None = None,
                     progress_callback: Callable[[], None] | None = None,
                     ) -> None:
        if request_id in self.active_ids:
            raise WorkerError("duplicate worker request")
        command = (f"BEGIN\t{request_id}\t{context_limit}\t" +
                   ",".join(str(token) for token in delta_ids) +
                   f"\tRESUME\t{session_key}")
        if checkpoint_tokens is not None:
            command += f"\tCHECKPOINT\t{checkpoint_tokens}"
        command += self._sampling_command(sampling)
        response = self._command(
            command, cancel_check=cancel_check,
            cancel_id=request_id, deadline=deadline,
            progress_callback=progress_callback,
        )
        if response.get("type") != "begun" or response.get("id") != request_id:
            raise WorkerError("unexpected BEGIN response")
        self.active_ids.add(request_id)

    def _sampling_command(self, settings: SamplingSettings) -> str:
        if not self.sampling_supported:
            if settings.enabled or settings.presence_penalty != 0.0:
                raise WorkerError(
                    "CUDA worker does not support requested sampling"
                )
            return ""
        if (settings.presence_penalty != 0.0 and
                not self.sampling_presence_penalty_supported):
            raise WorkerError(
                "CUDA worker does not support presence penalty"
            )
        if not self.sampling_presence_penalty_supported:
            return "\tSAMPLING\t{}\t{}\t{}\t{}\t{}".format(
                round(settings.temperature * 1_000_000),
                round(settings.top_p * 1_000_000),
                settings.top_k,
                round(settings.min_p * 1_000_000),
                settings.seed,
            )
        return "\tSAMPLING\t{}\t{}\t{}\t{}\t{}\t{}".format(
            round(settings.temperature * 1_000_000),
            round(settings.top_p * 1_000_000),
            settings.top_k,
            round(settings.min_p * 1_000_000),
            round(settings.presence_penalty * 1_000_000),
            settings.seed,
        )

    def end_retain(self, request_id: int, session_key: int,
                   checkpoint_tokens: int | None = None,
                   ) -> tuple[int, int, int]:
        command = f"END\t{request_id}\tRETAIN\t{session_key}"
        if checkpoint_tokens is not None:
            command += f"\tAT\t{checkpoint_tokens}"
        response = self._command(command)
        if response.get("type") != "ended" or response.get("id") != request_id:
            raise WorkerError("unexpected END response")
        retained_tokens = int(response.get("retained_tokens", 0))
        parked_pages = int(response.get("parked_pages", 0))
        parked_bytes = int(response.get("parked_bytes", 0))
        # kv_page_bytes is the maximum resident footprint of a logical page,
        # including optional provider state such as an MTP draft page.  It is
        # not a lower bound for a parked request: sampling requests
        # intentionally omit MTP state.  Validate the provider-authoritative
        # accounting against the negotiated global limits instead.
        if self.session_parking and (
                retained_tokens <= 0 or parked_pages <= 0 or
                parked_bytes <= 0 or
                parked_pages > self.session_park_page_capacity or
                parked_bytes > self.session_park_ram_bytes):
            try:
                self.drop_session(session_key)
            finally:
                self.active_ids.discard(request_id)
            raise WorkerError("worker returned invalid parked-state accounting")
        self.active_ids.discard(request_id)
        return retained_tokens, parked_pages, parked_bytes

    def drop_session(self, session_key: int) -> None:
        response = self._command(f"DROP\t{session_key}")
        if response.get("type") != "dropped":
            raise WorkerError("unexpected DROP response")

    @staticmethod
    def _snapshot_path(path: Path) -> str:
        value = str(path.resolve())
        if not value or any(marker in value for marker in ("\t", "\n", "\r")):
            raise WorkerError("session snapshot path is not protocol safe")
        return value

    def save_session(self, session_key: int, path: Path,
                     generation: int) -> dict[str, int | bool]:
        response = self._command(
            f"SAVE\t{session_key}\t{self._snapshot_path(path)}\t{generation}"
        )
        if response.get("type") != "saved" or \
                int(response.get("generation", 0)) != generation:
            raise WorkerError("unexpected SAVE response")
        keys = (
            "generation", "populated_pages", "logical_bytes",
            "written_bytes", "next_position", "predicted",
            "retention_predicted", "retention_prediction_valid",
            "sampling_temperature_ppm", "sampling_top_p_ppm",
            "sampling_top_k", "sampling_min_p_ppm",
            "sampling_presence_penalty_ppm", "sampling_seed",
        )
        return {key: response[key] for key in keys}

    def load_session(self, session_key: int, path: Path, generation: int,
                     metadata: Mapping[str, Any]) -> None:
        values = (
            int(metadata["next_position"]),
            int(metadata["populated_pages"]),
            int(metadata["predicted"]),
            int(metadata["retention_predicted"]),
            1 if bool(metadata["retention_prediction_valid"]) else 0,
            int(metadata["sampling_temperature_ppm"]),
            int(metadata["sampling_top_p_ppm"]),
            int(metadata["sampling_top_k"]),
            int(metadata["sampling_min_p_ppm"]),
            int(metadata["sampling_presence_penalty_ppm"]),
            int(metadata["sampling_seed"]),
        )
        response = self._command(
            "LOAD\t{}\t{}\t{}\t{}".format(
                session_key, self._snapshot_path(path), generation,
                "\t".join(str(value) for value in values),
            )
        )
        if (response.get("type") != "loaded" or
                int(response.get("key", 0)) != session_key or
                int(response.get("generation", 0)) != generation):
            raise WorkerError("unexpected LOAD response")

    def prune_session(self, session_key: int, path: Path,
                      generation: int) -> None:
        response = self._command(
            f"PRUNE\t{session_key}\t{self._snapshot_path(path)}\t{generation}"
        )
        if response.get("type") != "pruned" or \
                int(response.get("generation", 0)) != generation:
            raise WorkerError("unexpected PRUNE response")

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
        effort_map = self.manifest.get("tokenizer", {}).get(
            "reasoning_effort_map"
        )
        if effort_map is not None and (
                not isinstance(effort_map, dict) or
                set(effort_map) != {"off", "low", "medium", "xhigh"} or
                any(not isinstance(value, str) or not value or len(value) > 32
                    for value in effort_map.values())):
            raise RuntimeError("artifact tokenizer reasoning effort map is invalid")
        self.template_reasoning_effort_map = effort_map
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer), local_files_only=True, trust_remote_code=False
        )
        self.artifact_chat_codec = discover_artifact_chat_codec(
            args.tokenizer
        )
        self.response_protocol = select_response_protocol(
            self.tokenizer, self.artifact_chat_codec
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
        self.sampling_profiles = {
            "thinking": self.default_sampling,
            "non_thinking": self.default_sampling,
        }
        self.maximum_thinking_tokens: int | None = None
        artifact_sampling = self.manifest.get("tokenizer", {}).get("sampling")
        if artifact_sampling is not None:
            if (not isinstance(artifact_sampling, dict) or
                    not isinstance(artifact_sampling.get("schema"), str) or
                    not artifact_sampling["schema"].strip() or
                    not {"schema", "profiles"} <= set(artifact_sampling) or
                    set(artifact_sampling) - {
                        "schema", "profiles", "maximum_thinking_tokens"
                    } or
                    not isinstance(artifact_sampling.get("profiles"), dict) or
                    set(artifact_sampling["profiles"]) != {
                        "thinking", "non_thinking"
                    }):
                raise RuntimeError("artifact sampling profiles are invalid")
            maximum_thinking_tokens = artifact_sampling.get(
                "maximum_thinking_tokens"
            )
            if maximum_thinking_tokens is not None:
                if (isinstance(maximum_thinking_tokens, bool) or
                        not isinstance(maximum_thinking_tokens, int) or
                        not 1 <= maximum_thinking_tokens < args.max_context):
                    raise RuntimeError(
                        "artifact sampling maximum_thinking_tokens is invalid"
                    )
                self.maximum_thinking_tokens = maximum_thinking_tokens
            profiles: dict[str, SamplingSettings] = {}
            for name, profile in artifact_sampling["profiles"].items():
                if not isinstance(profile, dict):
                    raise RuntimeError("artifact sampling profile is invalid")
                unsupported_penalty = (
                    profile.get("frequency_penalty") not in (0, 0.0) or
                    profile.get("repetition_penalty") not in (1, 1.0)
                )
                if unsupported_penalty:
                    raise RuntimeError(
                        "artifact requests unsupported sampling penalties"
                    )
                try:
                    settings = SamplingSettings(
                        temperature=float(profile["temperature"]),
                        top_p=float(profile["top_p"]),
                        top_k=int(profile["top_k"]),
                        min_p=float(profile["min_p"]),
                        seed=0,
                        presence_penalty=float(profile["presence_penalty"]),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "artifact sampling profile is invalid"
                    ) from error
                self._validate_sampling(settings, "artifact sampling profile")
                profiles[name] = settings
            self.sampling_profiles = profiles
            self.default_sampling = profiles["thinking"]
        self.eos_token_ids = _configured_eos_token_ids(
            generation_config, self.tokenizer.eos_token_id
        )
        self.worker = CudaWorker(args.worker, args.container, args.max_context,
                                 args.startup_timeout, args.worker_capacity,
                                 args.worker_ram_cache_gib,
                                 args.worker_vram_cache_gib,
                                 args.worker_kv_cache_mib,
                                 args.worker_kv_page_tokens,
                                 args.placement_profile,
                                 args.profile_gpu_phases,
                                 args.worker_kv_cache_dtype,
                                 args.worker_prefill_chunk_tokens,
                                 args.worker_placement_settle_steps,
                                 (False if args.disable_worker_retained_route
                                  else None),
                                 True if args.enable_worker_cpu_hybrid else None,
                                 args.worker_route_trace_file,
                                 args.worker_route_trace_max_steps,
                                 args.worker_active_expert_devices,
                                 args.worker_active_expert_device_cache_gib,
                                 args.worker_active_expert_host_cache_gib,
                                 args.worker_routed_vram_policy)
        self.vision_capability = (
            "vision.patch-transformer-merge.fp4-block32.v1"
        )
        self.vision_enabled = (
            self.vision_capability in self.worker.operation_capabilities
        )
        self.image_processor = None
        if self.vision_enabled:
            preprocessor = args.tokenizer / "preprocessor_config.json"
            if not preprocessor.is_file():
                self.worker.close()
                raise RuntimeError(
                    "vision artifact is missing preprocessor_config.json"
                )
            self.image_processor = create_image_processor(str(args.tokenizer))
        self.capacity = threading.BoundedSemaphore(
            args.maximum_queue + args.worker_capacity
        )
        self.worker_slots = threading.BoundedSemaphore(args.worker_capacity)
        self.kv_credit_lock = threading.Lock()
        self.kv_reserved_pages = 0
        self.session_lock = threading.Lock()
        self.sessions: OrderedDict[int, Session] = OrderedDict()
        self.disk_sessions: OrderedDict[str, Session] = OrderedDict()
        self.next_session_key = 1
        identity_payload = {
            "schema": "persistent-session-v1",
            "artifact": self.manifest.get("integrity", {}).get(
                "content_sha256", ""
            ),
            "dense": self.manifest.get("indexes", {}).get(
                "dense_sha256", ""
            ),
            "kv_dtype": self.worker.kv_dtype,
            "kv_page_tokens": self.worker.kv_page_tokens,
            "response_protocol": self.response_protocol,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_size": len(self.tokenizer),
            "special_tokens": self.tokenizer.special_tokens_map,
            "chat_template": getattr(self.tokenizer, "chat_template", None),
        }
        self.session_cache_identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        self.session_cache_root = args.session_cache_root
        self.session_cache_enabled = bool(
            self.retention_enabled() and
            getattr(self.worker, "session_persistence", False) and
            self.session_cache_root is not None and
            args.session_cache_bytes > 0
        )
        if self.session_cache_enabled:
            self.session_cache_root.mkdir(parents=True, exist_ok=True)
            self._load_persistent_sessions()
            self._cleanup_persistent_sessions()
        self.id_lock = threading.Lock()
        self.next_id = 1
        self.draining = threading.Event()
        self.raw_response_trace_path = args.raw_response_trace_file
        self.raw_response_trace_lock = threading.Lock()
        self.raw_response_trace_remaining = 32
        if self.raw_response_trace_path is not None:
            self.raw_response_trace_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            self.raw_response_trace_path.write_text("", encoding="utf-8")
        self.active = 0
        self.active_lock = threading.Lock()
        self.metric_lock = threading.Lock()
        self._worker_stats_lock = threading.Lock()
        self._worker_stats_cache: dict[str, int] = {}
        self._worker_stats_updated_at = 0.0
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
        self.worker_stats(refresh=True)
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
            if self.kv_reserved_pages + pages > \
                    self._session_page_capacity():
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

    def _session_cache_directory(self, persistent_id: str) -> Path:
        if (not persistent_id or len(persistent_id) != 32 or
                any(character not in "0123456789abcdef"
                    for character in persistent_id)):
            raise ValueError("persistent session id is invalid")
        assert self.session_cache_root is not None
        return self.session_cache_root / persistent_id

    @staticmethod
    def _directory_bytes(path: Path) -> int:
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except OSError:
            return 0
        return total

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = path.with_name(path.name + ".partial")
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with candidate.open("wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(candidate, path)

    @staticmethod
    def _metadata_digest(payload: Mapping[str, Any]) -> str:
        authenticated = {
            key: value for key, value in payload.items()
            if key != "metadata_sha256"
        }
        return hashlib.sha256(json.dumps(
            authenticated, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    def _remove_persistent_session(self, session: Session,
                                   reason: str) -> None:
        if session.persistent_id is None:
            return
        with self.session_lock:
            self.disk_sessions.pop(session.persistent_id, None)
        if session.snapshot_path is not None:
            shutil.rmtree(session.snapshot_path, ignore_errors=True)
        log("session_snapshot_removed", persistent_id=session.persistent_id,
            tokens=len(session.tokens), bytes=session.disk_bytes,
            reason=reason)

    def _load_persistent_sessions(self) -> None:
        assert self.session_cache_root is not None
        now = time.time()
        loaded: list[Session] = []
        for directory in self.session_cache_root.iterdir():
            if not directory.is_dir():
                if ".partial" in directory.name:
                    try:
                        directory.unlink()
                    except OSError:
                        pass
                continue
            metadata_path = directory / "session.json"
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (not isinstance(payload, dict) or
                        not hmac.compare_digest(
                            str(payload.get("metadata_sha256", "")),
                            self._metadata_digest(payload))):
                    raise ValueError("persistent session checksum mismatch")
                persistent_id = str(payload["persistent_id"])
                if (payload.get("schema") != "persistent-session-v1" or
                        directory != self._session_cache_directory(
                            persistent_id) or
                        payload.get("identity") != self.session_cache_identity):
                    raise ValueError("persistent session identity mismatch")
                generation = int(payload["generation"])
                tokens = payload["tokens"]
                pages = int(payload["pages"])
                parked_bytes = int(payload["parked_bytes"])
                last_used_epoch = float(payload["last_used_epoch"])
                worker_metadata = payload["worker_metadata"]
                signature_hex = payload.get("media_signature")
                if (generation < 1 or not isinstance(tokens, list) or
                        not 0 < len(tokens) <= self.args.max_context or
                        any(not isinstance(token, int) or
                            isinstance(token, bool) or token < 0 or
                            token >= self.worker.vocab_size
                            for token in tokens) or
                        pages != self._context_pages(len(tokens)) or
                        parked_bytes <= 0 or
                        not isinstance(worker_metadata, dict) or
                        int(worker_metadata.get("generation", 0)) !=
                            generation or
                        int(worker_metadata.get("next_position", 0)) !=
                            len(tokens)):
                    raise ValueError("persistent session metadata is invalid")
                media_signature = (
                    bytes.fromhex(signature_hex)
                    if isinstance(signature_hex, str) else None
                )
                loaded.append(Session(
                    key=0, tokens=[int(token) for token in tokens],
                    pages=pages, last_used=time.monotonic(),
                    media_signature=media_signature,
                    parked_bytes=parked_bytes,
                    persistent_id=persistent_id,
                    snapshot_generation=generation,
                    snapshot_path=directory,
                    snapshot_metadata={
                        key: value for key, value in worker_metadata.items()
                        if isinstance(value, (int, bool))
                    },
                    disk_bytes=self._directory_bytes(directory),
                    resident=False, persistent_last_used=last_used_epoch,
                ))
            except Exception as error:
                shutil.rmtree(directory, ignore_errors=True)
                log("session_snapshot_rejected", path=str(directory),
                    error=str(error))
        loaded.sort(key=lambda session: session.persistent_last_used)
        with self.session_lock:
            for session in loaded:
                assert session.persistent_id is not None
                self.disk_sessions[session.persistent_id] = session
        log("session_snapshot_index_loaded", sessions=len(loaded),
            bytes=sum(session.disk_bytes for session in loaded))

    def _cleanup_persistent_sessions(self) -> None:
        if not getattr(self, "session_cache_enabled", False):
            return
        now = time.time()
        ttl = self.args.session_cache_ttl_seconds
        expired: list[Session] = []
        with self.session_lock:
            live_ids = {
                session.persistent_id for session in self.sessions.values()
                if session.persistent_id is not None
            }
            for persistent_id, session in list(self.disk_sessions.items()):
                if (ttl > 0 and now - session.persistent_last_used > ttl and
                        persistent_id not in live_ids):
                    expired.append(self.disk_sessions.pop(persistent_id))
            total = sum(session.disk_bytes
                        for session in self.disk_sessions.values())
            while total > self.args.session_cache_bytes:
                candidate = next((
                    (key, value) for key, value in self.disk_sessions.items()
                    if key not in live_ids
                ), None)
                if candidate is None:
                    break
                key, session = candidate
                self.disk_sessions.pop(key)
                total -= session.disk_bytes
                expired.append(session)
        for session in expired:
            if session.snapshot_path is not None:
                shutil.rmtree(session.snapshot_path, ignore_errors=True)
            log("session_snapshot_evicted",
                persistent_id=session.persistent_id,
                tokens=len(session.tokens), bytes=session.disk_bytes)

    def _persist_session(self, session: Session,
                         previous: Session | None) -> None:
        if not getattr(self, "session_cache_enabled", False):
            return
        persistent_id = (
            previous.persistent_id
            if previous is not None and previous.persistent_id is not None
            else uuid.uuid4().hex
        )
        previous_generation = (
            previous.snapshot_generation if previous is not None else 0
        )
        generation = max(previous_generation + 1, time.time_ns())
        directory = self._session_cache_directory(persistent_id)
        worker_metadata = self.worker.save_session(
            session.key, directory, generation
        )
        if (int(worker_metadata["next_position"]) != len(session.tokens) or
                int(worker_metadata["populated_pages"]) != session.pages or
                int(worker_metadata["logical_bytes"]) !=
                    session.parked_bytes):
            raise WorkerError("persisted session accounting mismatch")
        now = time.time()
        metadata = {
            "schema": "persistent-session-v1",
            "identity": self.session_cache_identity,
            "persistent_id": persistent_id,
            "generation": generation,
            "tokens": session.tokens,
            "pages": session.pages,
            "parked_bytes": session.parked_bytes,
            "media_signature": (
                session.media_signature.hex()
                if session.media_signature is not None else None
            ),
            "last_used_epoch": now,
            "worker_metadata": worker_metadata,
        }
        metadata["metadata_sha256"] = self._metadata_digest(metadata)
        self._write_json_atomic(directory / "session.json", metadata)
        try:
            self.worker.prune_session(session.key, directory, generation)
        except Exception as error:
            # The committed generation remains valid. Stale immutable blobs
            # cost disk space but cannot change which snapshot is selected.
            log("session_snapshot_prune_failed",
                persistent_id=persistent_id, generation=generation,
                error=str(error))
        session.persistent_id = persistent_id
        session.snapshot_generation = generation
        session.snapshot_path = directory
        session.snapshot_metadata = worker_metadata
        session.persistent_last_used = now
        session.disk_bytes = self._directory_bytes(directory)
        durable = Session(
            key=0, tokens=list(session.tokens), pages=session.pages,
            last_used=time.monotonic(),
            media_signature=session.media_signature,
            parked_bytes=session.parked_bytes,
            persistent_id=persistent_id,
            snapshot_generation=generation,
            snapshot_path=directory,
            snapshot_metadata=dict(worker_metadata),
            disk_bytes=session.disk_bytes, resident=False,
            persistent_last_used=now,
        )
        with self.session_lock:
            self.disk_sessions.pop(persistent_id, None)
            self.disk_sessions[persistent_id] = durable
        log("session_snapshot_committed", persistent_id=persistent_id,
            generation=generation, tokens=len(session.tokens),
            logical_bytes=session.parked_bytes,
            written_bytes=int(worker_metadata["written_bytes"]),
            disk_bytes=session.disk_bytes)
        self._cleanup_persistent_sessions()

    def _context_pages(self, context_tokens: int) -> int:
        return (context_tokens + self.worker.kv_page_tokens - 1) // \
            self.worker.kv_page_tokens

    def _session_page_capacity(self) -> int:
        if getattr(self.worker, "session_parking", False):
            return int(self.worker.session_park_page_capacity)
        return int(self.worker.kv_page_capacity)

    def _acquire_pages(self, pages: int) -> bool:
        with self.kv_credit_lock:
            if self.kv_reserved_pages + pages > self._session_page_capacity():
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

    def checkout_session(self, prompt_ids: list[int],
                         media_signature: bytes | None = None) -> Session | None:
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
                        session.media_signature == media_signature and
                        (best is None or length > len(best.tokens))):
                    best = session
            best_disk_id: str | None = None
            if getattr(self, "session_cache_enabled", False):
                for persistent_id, session in getattr(
                        self, "disk_sessions", {}).items():
                    length = len(session.tokens)
                    if (length <= len(prompt_ids) and
                            prompt_ids[:length] == session.tokens and
                            session.media_signature == media_signature and
                            (best is None or length > len(best.tokens))):
                        best = session
                        best_disk_id = persistent_id
            if best is not None and best.resident:
                del self.sessions[best.key]
            elif best_disk_id is not None:
                self.disk_sessions.pop(best_disk_id)
        return best

    def store_session(self, session_key: int, tokens: list[int],
                      pages: int,
                      media_signature: bytes | None = None,
                      parked_bytes: int = 0,
                      persistent_source: Session | None = None) -> Session:
        duplicates: list[Session] = []
        stored = Session(
            key=session_key, tokens=tokens, pages=pages,
            last_used=time.monotonic(), media_signature=media_signature,
            parked_bytes=parked_bytes,
            persistent_id=(persistent_source.persistent_id
                           if persistent_source is not None else None),
            snapshot_generation=(persistent_source.snapshot_generation
                                 if persistent_source is not None else 0),
            snapshot_path=(persistent_source.snapshot_path
                           if persistent_source is not None else None),
            snapshot_metadata=(persistent_source.snapshot_metadata
                               if persistent_source is not None else None),
            disk_bytes=(persistent_source.disk_bytes
                        if persistent_source is not None else 0),
            persistent_last_used=(persistent_source.persistent_last_used
                                  if persistent_source is not None else 0.0),
        )
        with self.session_lock:
            for key, session in list(self.sessions.items()):
                if (session.tokens == tokens and
                        session.media_signature == media_signature):
                    duplicates.append(self.sessions.pop(key))
            self.sessions[session_key] = stored
        for duplicate in duplicates:
            self._drop_worker_session(duplicate.key)
            self.release_context_credits(duplicate.pages)
        return stored

    def _return_disk_session(self, session: Session) -> None:
        if session.persistent_id is None:
            return
        session.key = 0
        session.resident = False
        with self.session_lock:
            disk_sessions = getattr(self, "disk_sessions", None)
            if disk_sessions is not None:
                disk_sessions.pop(session.persistent_id, None)
                disk_sessions[session.persistent_id] = session

    def abandon_session(self, session: Session) -> None:
        """Drop a checked-out session whose request never started."""
        if session.resident:
            self._drop_worker_session(session.key)
            self.release_context_credits(session.pages)
        self._return_disk_session(session)

    def allocate_session_key(self) -> int:
        with self.session_lock:
            key = self.next_session_key
            self.next_session_key += 1
            return key

    def acquire_request_context(self, prompt_ids: list[int],
                                maximum: int,
                                media_signature: bytes | None = None,
                                ) -> RequestContext | None:
        session = self.checkout_session(prompt_ids, media_signature)
        if session is not None and not session.resident:
            while not self._acquire_pages(session.pages):
                if not self.evict_lru_session():
                    self._return_disk_session(session)
                    session = None
                    break
            if session is not None:
                key = self.allocate_session_key()
                try:
                    assert session.snapshot_path is not None
                    assert session.snapshot_metadata is not None
                    started = time.monotonic()
                    self.worker.load_session(
                        key, session.snapshot_path,
                        session.snapshot_generation,
                        session.snapshot_metadata,
                    )
                    session.key = key
                    session.resident = True
                    log("session_snapshot_restored",
                        persistent_id=session.persistent_id,
                        generation=session.snapshot_generation,
                        tokens=len(session.tokens),
                        logical_bytes=session.parked_bytes,
                        wall_seconds=time.monotonic() - started)
                except Exception as error:
                    self.release_context_credits(session.pages)
                    log("session_snapshot_restore_failed",
                        persistent_id=session.persistent_id,
                        error=str(error))
                    self._return_disk_session(session)
                    session = None
        # The provider allocates execution KV on demand. Admission therefore
        # carries only an already-parked prefix; speculative output capacity
        # is never reserved as if it were populated state.
        base_pages = session.pages if session is not None else 0
        return RequestContext(session=session, held_pages=base_pages)

    def resize_request_context(self, context: RequestContext,
                               target_pages: int) -> bool:
        if target_pages < 0:
            raise RuntimeError("invalid retained context page count")
        if target_pages < context.held_pages:
            self.release_context_credits(context.held_pages - target_pages)
            context.held_pages = target_pages
            return True
        needed = target_pages - context.held_pages
        while needed > 0 and not self._acquire_pages(needed):
            if not self.evict_lru_session():
                return False
        context.held_pages = target_pages
        return True

    def release_request_context(self, context: RequestContext) -> None:
        if context.retained:
            # The retained session owns the exact parked-page credits.
            return
        if context.held_pages > 0:
            self.release_context_credits(context.held_pages)
        if context.session is not None:
            self._drop_worker_session(context.session.key)
            self._return_disk_session(context.session)

    def worker_stats(self, refresh: bool = True) -> dict[str, int]:
        lock = getattr(self, "_worker_stats_lock", None)
        if not refresh:
            if lock is None:
                return dict(getattr(self, "_worker_stats_cache", {}))
            with lock:
                return dict(self._worker_stats_cache)
        stats = getattr(self.worker, "stats", None)
        if stats is None:
            result: dict[str, int] = {}
        else:
            try:
                result = stats()
            except Exception:
                if lock is None:
                    return dict(getattr(self, "_worker_stats_cache", {}))
                with lock:
                    return dict(self._worker_stats_cache)
        if lock is None:
            self._worker_stats_cache = dict(result)
            self._worker_stats_updated_at = time.monotonic()
            return result
        with lock:
            self._worker_stats_cache = dict(result)
            self._worker_stats_updated_at = time.monotonic()
        return result

    def worker_stats_age_seconds(self) -> float:
        lock = getattr(self, "_worker_stats_lock", None)
        if lock is None:
            updated_at = getattr(self, "_worker_stats_updated_at", 0.0)
        else:
            with lock:
                updated_at = self._worker_stats_updated_at
        if updated_at <= 0.0:
            return 0.0
        return max(0.0, time.monotonic() - updated_at)

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
        worker_stats = self.worker_stats(refresh=False)
        lines.extend(("# TYPE expert_worker_stats_age_seconds gauge",
                      "expert_worker_stats_age_seconds "
                      f"{self.worker_stats_age_seconds()}"))
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
        codec = getattr(self, "artifact_chat_codec", None)
        if codec is not None:
            prompt = codec.encode(
                messages,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            return [int(token) for token in self.tokenizer.encode(
                prompt, add_special_tokens=False
            )]
        template_effort = self._template_reasoning_effort(
            reasoning_effort, enable_thinking
        )
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=list(tools) if tools else None,
            reasoning_effort=template_effort,
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

    def _template_reasoning_effort(
            self, reasoning_effort: str, enable_thinking: bool) -> str:
        mapping = getattr(self, "template_reasoning_effort_map", None)
        if mapping is None:
            return reasoning_effort
        return str(mapping[reasoning_effort if enable_thinking else "off"])

    @staticmethod
    def _has_images(messages: list[dict[str, Any]]) -> bool:
        return any(
            isinstance(message.get("content"), list) and
            any(isinstance(part, dict) and part.get("type") == "image"
                for part in message["content"])
            for message in messages
        )

    def _prepare_chat_prompt(
            self, messages: list[dict[str, Any]],
            add_generation_prompt: bool = True,
            tools: tuple[dict[str, Any], ...] = (),
            reasoning_effort: str = "xhigh",
            enable_thinking: bool = True,
            preserve_thinking: bool = True,
            ) -> tuple[list[int], bytes | None, bytes | None, int, int]:
        if not self._has_images(messages):
            return (self._chat_prompt_ids(
                messages, add_generation_prompt=add_generation_prompt,
                tools=tools, reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            ), None, None, 0, 0)
        if (not getattr(self, "vision_enabled", False) or
                getattr(self, "image_processor", None) is None):
            raise RequestError(
                "image input is not supported by the active artifact",
                "messages", "unsupported_value",
            )
        try:
            template_effort = self._template_reasoning_effort(
                reasoning_effort, enable_thinking
            )
            prepared: PreparedMultimodalPrompt = prepare_multimodal_prompt(
                self.tokenizer, self.image_processor, messages,
                add_generation_prompt=add_generation_prompt,
                tools=list(tools) if tools else None,
                reasoning_effort=template_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
                maximum_image_pixels=self.args.maximum_image_pixels,
                maximum_patch_tokens=self.args.maximum_image_patch_tokens,
            )
        except MultimodalInputError as error:
            raise RequestError(str(error), "messages") from error
        return (
            prepared.token_ids, prepared.packet, prepared.media_signature,
            prepared.image_count, prepared.image_tokens,
        )

    def response_stream_parser(self, request: GenerationRequest) -> Any:
        if self.response_protocol is None:
            return None
        codec = getattr(self, "artifact_chat_codec", None)
        if self.response_protocol == "artifact-chat-codec-v1":
            if codec is None or not codec.supports_response_parsing:
                raise RuntimeError(
                    "selected artifact response codec is unavailable"
                )
            return codec.stream_parser(request.enable_thinking)
        return self.tokenizer.get_response_parser(
            prefix=request.prompt_ids,
            tools=list(request.tools) if request.tools else None,
        )

    @staticmethod
    def _assistant_output_from_message(
            message: dict[str, Any]) -> AssistantOutput:
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
        for raw_call in raw_calls:
            if (not isinstance(raw_call, dict) or
                    raw_call.get("type") != "function"):
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

    def parse_assistant_output(
            self, text: str, request: GenerationRequest) -> AssistantOutput:
        if self.response_protocol is None:
            output = AssistantOutput(
                text=text, reasoning="", reasoning_complete=True,
                tool_calls=(),
            )
            self.trace_assistant_response(request, text, output)
            return output
        try:
            codec = getattr(self, "artifact_chat_codec", None)
            if self.response_protocol == "artifact-chat-codec-v1":
                if codec is None or not codec.supports_response_parsing:
                    raise RuntimeError(
                        "selected artifact response codec is unavailable"
                    )
                message = codec.parse(text, request.enable_thinking)
            else:
                message = self.tokenizer.parse_response(
                    text, prefix=request.prompt_ids,
                    tools=list(request.tools) if request.tools else None,
                )
            output = self._validated_assistant_output(message, text)
            self.trace_assistant_response(request, text, output)
            return output
        except (AssertionError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            log("response_parse_failed", protocol=self.response_protocol,
                error=str(error))
            output = AssistantOutput(
                text=text, reasoning="", reasoning_complete=False,
                tool_calls=(),
            )
            self.trace_assistant_response(
                request, text, output, error=str(error)
            )
            return output

    def trace_assistant_response(
            self, request: GenerationRequest, raw_text: str,
            output: AssistantOutput, *, error: str | None = None) -> None:
        """Record a bounded raw-to-structured diagnostic for tool responses.

        This is opt-in because native output can contain project data. The
        trace excludes prompts and tool schemas, keeps only the final 128 KiB
        of generated text, and stops after 32 responses per service start.
        """
        path = getattr(self, "raw_response_trace_path", None)
        if path is None or not request.tools:
            return
        lock = getattr(self, "raw_response_trace_lock", None)
        if lock is None:
            return
        with lock:
            remaining = getattr(self, "raw_response_trace_remaining", 0)
            if remaining <= 0:
                return
            self.raw_response_trace_remaining = remaining - 1
            maximum_characters = 128 * 1024
            raw_suffix = raw_text[-maximum_characters:]
            record = {
                "schema": "raw-response-trace-v1",
                "time": time.time(),
                "endpoint": request.endpoint,
                "response_protocol": self.response_protocol,
                "enable_thinking": request.enable_thinking,
                "reasoning_effort": request.reasoning_effort,
                "raw_sha256": hashlib.sha256(
                    raw_text.encode("utf-8")
                ).hexdigest(),
                "raw_characters": len(raw_text),
                "truncated_prefix_characters": (
                    len(raw_text) - len(raw_suffix)
                ),
                "raw_suffix": raw_suffix,
                "reasoning_complete": output.reasoning_complete,
                "parsed_tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in output.tool_calls
                ],
                "parse_error": error,
            }
            with path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(
                    record, separators=(",", ":"), ensure_ascii=False
                ) + "\n")

    def _validated_assistant_output(
            self, message: Any, text: str) -> AssistantOutput:
        if not isinstance(message, dict):
            raise ValueError("response parser returned a non-object")
        if (self.response_protocol != "artifact-chat-codec-v1" and
                message.get("tool_calls") and
                (text.count("<tool_call>") != text.count("</tool_call>") or
                 text.count("<function=") != text.count("</function>"))):
            raise ValueError("model emitted an incomplete tool call")
        return self._assistant_output_from_message(message)

    def assistant_output_from_parsed_message(
            self, message: Any, text: str, *, fallback_text: str
            ) -> AssistantOutput:
        """Validate the final message returned by the active stream parser."""
        try:
            return self._validated_assistant_output(message, text)
        except (AssertionError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            log("response_stream_parse_failed", protocol=self.response_protocol,
                error=str(error))
            return AssistantOutput(
                text=fallback_text, reasoning="", reasoning_complete=False,
                tool_calls=(),
            )

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
                content = _text_content(
                    raw_content, f"{item_param}.content",
                    allow_images=getattr(self, "vision_enabled", False),
                    user_content=role == "user",
                )
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
    def _anthropic_text(raw: Any, param: str) -> str:
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, list):
            raise RequestError("content must be text or an array of text blocks", param)
        pieces: list[str] = []
        for index, block in enumerate(raw):
            block_param = f"{param}.{index}"
            if not isinstance(block, dict) or block.get("type") != "text":
                raise RequestError(
                    "only text tool-result content is supported",
                    block_param, "unsupported_value",
                )
            text = block.get("text")
            if not isinstance(text, str):
                raise RequestError("text block requires text", f"{block_param}.text")
            pieces.append(text)
        return "".join(pieces)

    @classmethod
    def _anthropic_messages(
        cls, raw: Any
    ) -> tuple[list[dict[str, Any]], int | None]:
        if not isinstance(raw, list) or not raw:
            raise RequestError("messages must be a non-empty array", "messages")
        normalized: list[dict[str, Any]] = []
        cache_prefix_message_count: int | None = None
        for message_index, message in enumerate(raw):
            param = f"messages.{message_index}"
            if not isinstance(message, dict):
                raise RequestError("message must be an object", param)
            role = message.get("role")
            if role not in {"user", "assistant", "system"}:
                raise RequestError(
                    "Anthropic messages support user, assistant, and system roles",
                    f"{param}.role", "unsupported_value",
                )
            content = message.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_text = content
                elif isinstance(content, list) and content:
                    system_text = cls._anthropic_text(
                        content, f"{param}.content"
                    )
                else:
                    raise RequestError(
                        "system message content must be text or a non-empty block array",
                        f"{param}.content",
                    )
                if system_text:
                    # Historical reminders remain in Claude Code's message
                    # history while a new dynamic reminder is appended. The
                    # final boundary therefore advances the retained KV state
                    # each turn and excludes only the current reminder.
                    cache_prefix_message_count = len(normalized)
                    # Claude Code emits dynamic system reminders inside the
                    # message list. Qwen requires the only system role to be
                    # first, so retain the reminder at its conversational
                    # position as model-visible user context. Moving it into
                    # the leading system prompt would invalidate the large
                    # tools/system KV prefix on every turn.
                    normalized.append({"role": "user", "content": system_text})
                continue
            if isinstance(content, str):
                normalized.append({"role": role, "content": content})
                continue
            if not isinstance(content, list) or not content:
                raise RequestError(
                    "message content must be text or a non-empty block array",
                    f"{param}.content",
                )

            if role == "assistant":
                text: list[str] = []
                reasoning: list[str] = []
                calls: list[dict[str, Any]] = []
                for block_index, block in enumerate(content):
                    block_param = f"{param}.content.{block_index}"
                    if not isinstance(block, dict):
                        raise RequestError("content block must be an object", block_param)
                    kind = block.get("type")
                    if kind == "text":
                        value = block.get("text")
                        if not isinstance(value, str):
                            raise RequestError("text block requires text",
                                               f"{block_param}.text")
                        text.append(value)
                    elif kind == "thinking":
                        value = block.get("thinking")
                        if not isinstance(value, str):
                            raise RequestError("thinking block requires thinking text",
                                               f"{block_param}.thinking")
                        reasoning.append(value)
                    elif kind == "redacted_thinking":
                        # Redacted thinking contains no model-visible text. Its
                        # provider signature is intentionally not forwarded to
                        # a different local model.
                        continue
                    elif kind == "tool_use":
                        tool_id = block.get("id")
                        name = block.get("name")
                        arguments = block.get("input")
                        if not isinstance(tool_id, str) or not tool_id:
                            raise RequestError("tool_use requires an id",
                                               f"{block_param}.id")
                        if not isinstance(name, str) or not name:
                            raise RequestError("tool_use requires a name",
                                               f"{block_param}.name")
                        if not isinstance(arguments, dict):
                            raise RequestError("tool_use input must be an object",
                                               f"{block_param}.input")
                        calls.append({
                            "id": tool_id, "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        })
                    else:
                        raise RequestError(
                            f"content block type {kind!r} is not supported",
                            f"{block_param}.type", "unsupported_value",
                        )
                item: dict[str, Any] = {
                    "role": "assistant", "content": "".join(text),
                }
                if reasoning:
                    item["reasoning_content"] = "".join(reasoning)
                if calls:
                    item["tool_calls"] = calls
                normalized.append(item)
                continue

            tool_results: list[dict[str, Any]] = []
            text: list[str] = []
            user_parts: list[dict[str, Any]] = []
            has_image = False
            for block_index, block in enumerate(content):
                block_param = f"{param}.content.{block_index}"
                if not isinstance(block, dict):
                    raise RequestError("content block must be an object", block_param)
                kind = block.get("type")
                if kind == "tool_result":
                    if text:
                        raise RequestError(
                            "tool_result blocks must precede text blocks",
                            block_param,
                        )
                    tool_use_id = block.get("tool_use_id")
                    if not isinstance(tool_use_id, str) or not tool_use_id:
                        raise RequestError("tool_result requires tool_use_id",
                                           f"{block_param}.tool_use_id")
                    result_text = cls._anthropic_text(
                        block.get("content", ""), f"{block_param}.content"
                    )
                    is_error = block.get("is_error", False)
                    if not isinstance(is_error, bool):
                        raise RequestError("is_error must be boolean",
                                           f"{block_param}.is_error")
                    tool_results.append({
                        "role": "tool", "content": result_text,
                        "tool_call_id": tool_use_id,
                    })
                elif kind == "text":
                    value = block.get("text")
                    if not isinstance(value, str):
                        raise RequestError("text block requires text",
                                           f"{block_param}.text")
                    text.append(value)
                    user_parts.append({"type": "text", "text": value})
                elif kind == "image":
                    if tool_results:
                        raise RequestError(
                            "image blocks cannot follow tool results", block_param
                        )
                    source = block.get("source")
                    if not isinstance(source, dict):
                        raise RequestError(
                            "image block requires source", f"{block_param}.source"
                        )
                    source_type = source.get("type")
                    if source_type == "base64":
                        media_type = source.get("media_type")
                        data = source.get("data")
                        if not isinstance(media_type, str) or not isinstance(data, str):
                            raise RequestError(
                                "base64 image requires media_type and data",
                                f"{block_param}.source",
                            )
                        reference = f"data:{media_type};base64,{data}"
                    elif source_type == "url" and isinstance(source.get("url"), str):
                        reference = source["url"]
                    else:
                        raise RequestError(
                            "image source must be base64 or url",
                            f"{block_param}.source.type", "unsupported_value",
                        )
                    user_parts.append({
                        "type": "image_url", "image_url": {"url": reference}
                    })
                    has_image = True
                else:
                    raise RequestError(
                        f"content block type {kind!r} is not supported",
                        f"{block_param}.type", "unsupported_value",
                    )
            normalized.extend(tool_results)
            if user_parts:
                normalized.append({
                    "role": "user",
                    "content": user_parts if has_image else "".join(text),
                })
            if not tool_results and not user_parts:
                raise RequestError("user content has no supported blocks",
                                   f"{param}.content")
        return normalized, cache_prefix_message_count

    @classmethod
    def _anthropic_system(cls, raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if not isinstance(raw, list):
            raise RequestError("system must be text or an array of text blocks",
                               "system")
        pieces: list[str] = []
        for index, block in enumerate(raw):
            param = f"system.{index}"
            if not isinstance(block, dict) or block.get("type") != "text":
                raise RequestError("system supports only text blocks", param,
                                   "unsupported_value")
            text = block.get("text")
            if not isinstance(text, str):
                raise RequestError("system text block requires text",
                                   f"{param}.text")
            pieces.append(text)
        return "".join(pieces)

    @staticmethod
    def _anthropic_tools(raw: Any) -> list[dict[str, Any]]:
        if raw in (None, []):
            return []
        if not isinstance(raw, list):
            raise RequestError("tools must be an array", "tools")
        result: list[dict[str, Any]] = []
        for index, tool in enumerate(raw):
            param = f"tools.{index}"
            if not isinstance(tool, dict):
                raise RequestError("tool must be an object", param)
            if tool.get("type") not in (None, "custom"):
                raise RequestError("server-side tools are not supported", param,
                                   "unsupported_value")
            name = tool.get("name")
            description = tool.get("description")
            schema = tool.get("input_schema")
            if not isinstance(name, str) or not name:
                raise RequestError("tool requires a name", f"{param}.name")
            if description is not None and not isinstance(description, str):
                raise RequestError("tool description must be text",
                                   f"{param}.description")
            if not isinstance(schema, dict):
                raise RequestError("tool input_schema must be an object",
                                   f"{param}.input_schema")
            function: dict[str, Any] = {"name": name, "parameters": schema}
            if description is not None:
                function["description"] = description
            result.append({"type": "function", "function": function})
        return result

    @classmethod
    def anthropic_payload(cls, payload: dict[str, Any],
                          count_only: bool = False) -> dict[str, Any]:
        """Normalize Anthropic Messages input into the common chat contract."""
        if "max_tokens" not in payload and not count_only:
            raise RequestError("max_tokens is required", "max_tokens")
        messages, cache_prefix_message_count = cls._anthropic_messages(
            payload.get("messages")
        )
        normalized: dict[str, Any] = {
            "model": payload.get("model"),
            "messages": messages,
            "max_tokens": 1 if count_only else payload.get("max_tokens"),
            "stream": False if count_only else payload.get("stream", False),
        }
        if "system" in payload:
            system = cls._anthropic_system(payload["system"])
            if system:
                normalized["messages"].insert(
                    0, {"role": "system", "content": system}
                )
                if cache_prefix_message_count is not None:
                    cache_prefix_message_count += 1
        if cache_prefix_message_count is not None and \
                cache_prefix_message_count > 0:
            cache_messages = normalized["messages"][
                :cache_prefix_message_count
            ]
            if any(message["role"] == "user" for message in cache_messages):
                normalized["_cache_prefix_messages"] = cache_messages
        if "stop_sequences" in payload:
            normalized["stop"] = payload["stop_sequences"]
        for field in ("temperature", "top_p", "top_k"):
            if field in payload:
                normalized[field] = payload[field]
        tools = cls._anthropic_tools(payload.get("tools"))
        if tools:
            normalized["tools"] = tools
        choice = payload.get("tool_choice")
        if choice is not None:
            if not isinstance(choice, dict):
                raise RequestError("tool_choice must be an object", "tool_choice")
            kind = choice.get("type")
            if kind in {"auto", "none"}:
                normalized["tool_choice"] = kind
            elif kind == "any":
                normalized["tool_choice"] = "required"
            elif kind == "tool" and isinstance(choice.get("name"), str):
                normalized["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice["name"]},
                }
            else:
                raise RequestError("unsupported tool_choice", "tool_choice",
                                   "unsupported_value")
        metadata = payload.get("metadata")
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise RequestError("metadata must be an object", "metadata")
            user_id = metadata.get("user_id")
            if user_id is not None:
                if not isinstance(user_id, str):
                    raise RequestError("metadata.user_id must be text",
                                       "metadata.user_id")
                normalized["user"] = user_id

        enable_thinking = True
        thinking = payload.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, dict):
                raise RequestError("thinking must be an object", "thinking")
            kind = thinking.get("type")
            if kind == "disabled":
                enable_thinking = False
            elif kind not in {"enabled", "adaptive"}:
                raise RequestError("unsupported thinking mode", "thinking.type",
                                   "unsupported_value")
        normalized["chat_template_kwargs"] = {
            "enable_thinking": enable_thinking,
            "preserve_thinking": True,
        }

        output_config = payload.get("output_config")
        effort = None
        if output_config is not None:
            if not isinstance(output_config, dict):
                raise RequestError("output_config must be an object", "output_config")
            if output_config.get("format") is not None:
                raise RequestError("structured output is not supported",
                                   "output_config.format", "unsupported_value")
            effort = output_config.get("effort")
        if effort is not None:
            effort_map = {"low": "low", "medium": "medium",
                          "high": "xhigh", "xhigh": "xhigh",
                          "max": "xhigh"}
            if effort not in effort_map:
                raise RequestError("unsupported output effort",
                                   "output_config.effort", "unsupported_value")
            normalized["reasoning_effort"] = effort_map[effort]
        return normalized

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
            if "strict" in function:
                if not isinstance(function["strict"], bool):
                    raise RequestError("function strict must be boolean",
                                       f"{param}.function.strict")
            values: dict[str, Any] = {"name": name}
            if "description" in function:
                values["description"] = description
            values["parameters"] = parameters
            if "strict" in function:
                values["strict"] = function["strict"]
            # The official chat template serializes mappings in insertion
            # order. Preserve the client's order among supported fields so
            # validation does not silently change the model-visible token
            # stream. Append only fields whose API default was omitted.
            normalized_function = {
                field: values[field]
                for field in function
                if field in values
            }
            for field in ("name", "parameters"):
                if field not in normalized_function:
                    normalized_function[field] = values[field]
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
            -2.0 <= settings.presence_penalty <= 2.0 and
            0 <= settings.seed <= 0x7fff_ffff_ffff_ffff
        )
        if not valid:
            if parameter == "generation_config":
                raise RuntimeError("generation_config sampling values are invalid")
            raise RequestError("sampling parameters are outside runtime limits",
                               parameter, "unsupported_value")

    def _sampling(self, payload: dict[str, Any],
                  enable_thinking: bool) -> SamplingSettings:
        defaults = self.sampling_profiles[
            "thinking" if enable_thinking else "non_thinking"
        ]

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
            presence_penalty=number(
                "presence_penalty", defaults.presence_penalty
            ),
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
            "enable_thinking", "preserve_thinking", "reasoning_effort"
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
        sampling = self._sampling(payload, enable_thinking)
        reasoning_effort = payload.get("reasoning_effort")
        template_reasoning_effort = template_kwargs.get("reasoning_effort")
        if reasoning_effort is not None and template_reasoning_effort is not None:
            raise RequestError("reasoning effort was specified twice",
                               "reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = template_reasoning_effort
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
        if maximum_value < 1:
            raise RequestError(f"{max_field} is outside service limits", max_field)
        maximum = min(maximum_value, self.args.maximum_new_tokens)
        maximum_thinking_tokens = getattr(
            self, "maximum_thinking_tokens", None
        )
        if enable_thinking and maximum_thinking_tokens is not None:
            maximum = min(maximum, maximum_thinking_tokens)

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
        media_packet: bytes | None = None
        media_signature: bytes | None = None
        image_count = 0
        image_tokens = 0
        if endpoint in {"chat", "anthropic"}:
            messages = self._messages(payload.get("messages"))
            (prompt_ids, media_packet, media_signature,
             image_count, image_tokens) = self._prepare_chat_prompt(
                messages, tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            stable_messages = messages
            raw_cache_messages = payload.get("_cache_prefix_messages") \
                if endpoint == "anthropic" else None
            if raw_cache_messages is not None:
                stable_messages = self._messages(
                    raw_cache_messages, "_cache_prefix_messages"
                )
            stable_ids, _stable_packet, stable_signature, \
                _stable_images, _stable_image_tokens = self._prepare_chat_prompt(
                stable_messages, add_generation_prompt=False,
                tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            if stable_signature != media_signature:
                raise RequestError(
                    "cache prefix changes the request media set", "messages"
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
            (prompt_ids, media_packet, media_signature,
             image_count, image_tokens) = self._prepare_chat_prompt(
                messages, tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            stable_ids, _stable_packet, stable_signature, \
                _stable_images, _stable_image_tokens = self._prepare_chat_prompt(
                messages, add_generation_prompt=False, tools=prompt_tools,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking,
            )
            if stable_signature != media_signature:
                raise RequestError(
                    "cache prefix changes the request media set", "input"
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
        # max_tokens is a ceiling, not a request to preallocate that many KV
        # positions. Clamp it after exact tokenization so callers can advertise
        # the model's full output range without guessing template/tool overhead.
        maximum = min(maximum, self.args.max_context - len(prompt_ids))
        if endpoint in {"chat", "anthropic", "responses"} and (
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
            media_packet=media_packet, media_signature=media_signature,
            image_count=image_count, image_tokens=image_tokens,
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
        "parked_sessions", "parked_pages",
        "reserved_pages", "kv_allocated_pages", "kv_reserved_pages",
        "provider_parked_request_bytes", "provider_parked_session_bytes",
    })
    _TELEMETRY_CONFIGURATION_KEYS = frozenset({
        "provider_workspace_rows", "provider_compact_flash_prefill",
    })

    @classmethod
    def _request_telemetry_fields(
            cls, before: dict[str, Any], after: dict[str, Any],
            ) -> dict[str, int]:
        fields = {
            key: after[key] - before[key]
            for key in before.keys() & after.keys()
            if key not in cls._TELEMETRY_GAUGE_KEYS and
            key not in cls._TELEMETRY_CONFIGURATION_KEYS and
            isinstance(before[key], int) and
            not isinstance(before[key], bool) and
            isinstance(after[key], int) and
            not isinstance(after[key], bool) and
            after[key] >= before[key]
        }
        # Provider configuration is stable rather than monotonic. Preserve the
        # current value in every request record so a zero delta cannot conceal
        # which execution path actually served the request.
        fields.update({
            key: after[key]
            for key in cls._TELEMETRY_CONFIGURATION_KEYS
            if isinstance(after.get(key), int) and
            not isinstance(after[key], bool)
        })
        return fields

    @staticmethod
    def _capacity_error(error: Exception) -> bool:
        message = str(error)
        return "capacity" in message or "slot available" in message or \
            "no free request slot" in message or "credits" in message

    def end_retain_with_eviction(
            self, request_id: int, session_key: int,
            checkpoint_tokens: int | None,
            ) -> int | tuple[int, int, int]:
        while True:
            try:
                try:
                    return self.worker.end_retain(
                        request_id, session_key, checkpoint_tokens
                    )
                except TypeError as error:
                    if "positional" not in str(error):
                        raise
                    return self.worker.end_retain(request_id, session_key)
            except WorkerError as error:
                if self._capacity_error(error) and self.evict_lru_session():
                    continue
                raise

    def _retain_active_request(
            self, request_id: int, context: RequestContext,
            session: Session | None, checkpoint_tokens: int,
            visible_tokens: list[int], media_signature: bytes | None,
            resumed: bool, require_exact_checkpoint: bool = False) -> bool:
        if checkpoint_tokens <= 0 or checkpoint_tokens > len(visible_tokens):
            raise WorkerError("retention checkpoint is outside visible tokens")
        expected_pages = self._context_pages(checkpoint_tokens)
        if not self.resize_request_context(context, expected_pages):
            raise WorkerError("retained-session RAM credits are exhausted")
        session_key = session.key if session is not None \
            else self.allocate_session_key()
        retained_result = self.end_retain_with_eviction(
            request_id, session_key, checkpoint_tokens
        )
        if isinstance(retained_result, tuple):
            retained_tokens, parked_pages, parked_bytes = retained_result
        else:
            retained_tokens = int(retained_result)
            parked_pages = self._context_pages(retained_tokens)
            parked_bytes = 0
        retained_tokens_valid = (
            retained_tokens == checkpoint_tokens
            if require_exact_checkpoint
            else 0 < retained_tokens <= len(visible_tokens)
        )
        if not retained_tokens_valid:
            self.worker.drop_session(session_key)
            log("session_retain_mismatch", key=session_key,
                retained_tokens=retained_tokens,
                expected_tokens=(checkpoint_tokens
                                 if require_exact_checkpoint
                                 else len(visible_tokens)))
            return False
        retained_pages = self._context_pages(retained_tokens)
        if parked_pages not in {0, retained_pages}:
            self.worker.drop_session(session_key)
            raise WorkerError(
                "worker parked-page accounting does not match "
                "the retained prefix"
            )
        if not self.resize_request_context(context, retained_pages):
            self.worker.drop_session(session_key)
            raise WorkerError(
                "retained-session RAM credits changed during parking"
            )
        stored = self.store_session(
            session_key, visible_tokens[:retained_tokens], retained_pages,
            media_signature, parked_bytes, persistent_source=session,
        )
        context.retained = True
        try:
            self._persist_session(stored, session)
        except Exception as error:
            # Persistence is a durable acceleration tier. A failed write must
            # never invalidate the exact in-RAM retained state that already
            # passed the provider's checkpoint gate.
            log("session_snapshot_save_failed", key=session_key,
                error=str(error))
        log("session_retained", key=session_key, tokens=retained_tokens,
            resumed=resumed)
        return True

    def generate(self, prompt_ids: list[int], maximum: int,
                 context: RequestContext | None = None,
                 cancel_check: Callable[[], bool] | None = None,
                 cache_prefix_tokens: int | None = None,
                 sampling: SamplingSettings | None = None,
                 progress_callback: Callable[[], None] | None = None,
                 media_packet: bytes | None = None,
                 media_signature: bytes | None = None,
                 preserve_special_tokens: bool = False,
                 ) -> Iterator[tuple[int, str]]:
        request_id = self.request_id()
        generated: list[int] = []
        decoder = IncrementalTokenDecoder(
            self.tokenizer,
            preserve_special_tokens=preserve_special_tokens,
            # Protocol markers must survive tokenizer decoding, but every
            # declared EOS variant remains transport framing rather than
            # visible assistant content.
            suppressed_token_ids=self.eos_token_ids,
        )
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
        checkpoint_tokens = (cache_prefix_tokens or len(prompt_ids)) \
            if retain else None
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
                            progress_callback=progress_callback,
                        )
                    resumed = True
                    prefill_tokens = len(delta)
                else:
                    if cancel_check is None:
                        if media_packet is None:
                            self.worker.begin(
                                request_id, prompt_ids, context_limit,
                                effective_sampling,
                            )
                        else:
                            self.worker.begin(
                                request_id, prompt_ids, context_limit,
                                effective_sampling, media_packet=media_packet,
                            )
                    else:
                        begin_options = {
                            "checkpoint_tokens": checkpoint_tokens,
                            "cancel_check": cancel_check,
                            "deadline": deadline,
                            "progress_callback": progress_callback,
                        }
                        if media_packet is not None:
                            begin_options["media_packet"] = media_packet
                        self.worker.begin(
                            request_id, prompt_ids, context_limit,
                            effective_sampling, **begin_options,
                        )
                break
            except WorkerError as error:
                if self._capacity_error(error) and self.evict_lru_session():
                    continue
                if session is not None:
                    if "retained session preserved:" in str(error):
                        assert context is not None
                        self.store_session(
                            session.key, session.tokens, session.pages,
                            session.media_signature, session.parked_bytes,
                            persistent_source=session,
                        )
                        context.retained = True
                        log("session_resume_rolled_back", key=session.key,
                            error=str(error))
                        raise
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
                if progress_callback is not None:
                    progress_callback()
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
                if progress_callback is not None:
                    progress_callback()
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
                final_text = token in self.eos_token_ids or index + 1 == maximum
                delta = decoder.push(token, final=final_text)
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
                    assert context is not None
                    expected_retained_tokens = checkpoint_tokens or \
                        len(prompt_ids) + len(generated)
                    self._retain_active_request(
                        request_id, context, session,
                        expected_retained_tokens,
                        prompt_ids + generated + buffered,
                        media_signature, resumed,
                    )
                except WorkerError as error:
                    log("session_retain_failed", error=str(error))
            elif (not finished and retain and
                  request_id in self.worker.active_ids):
                # A client may disconnect after prefill while a decode token
                # is in flight. Rewind to the declared prompt checkpoint and
                # retain that exact, client-echoable prefix; partial assistant
                # output must never leak into a later turn's KV state.
                try:
                    assert context is not None
                    assert checkpoint_tokens is not None
                    if self._retain_active_request(
                            request_id, context, session, checkpoint_tokens,
                            prompt_ids, media_signature, resumed,
                            require_exact_checkpoint=True):
                        log("session_cancelled_prefix_retained",
                            tokens=checkpoint_tokens, resumed=resumed)
                except WorkerError as error:
                    log("session_cancelled_prefix_retain_failed",
                        error=str(error))
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
            deltas = self._request_telemetry_fields(stats_before, stats_after)
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
            session_parked_bytes = sum(
                session.parked_bytes for session in self.sessions.values()
            )
            disk_sessions = getattr(self, "disk_sessions", {})
            disk_session_count = len(disk_sessions)
            disk_session_tokens = sum(
                len(session.tokens) for session in disk_sessions.values()
            )
            disk_session_bytes = sum(
                session.disk_bytes for session in disk_sessions.values()
            )
        # Introspection must remain available during a long layer-major
        # prefill.  The worker command stream is deliberately serialized, so
        # querying STATS here would otherwise block behind the GPU request.
        kv_stats = self.worker_stats(refresh=False)
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
            "input_modalities": (
                ["text", "image"]
                if getattr(self, "vision_enabled", False) else ["text"]
            ),
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
                "routed_vram_policy": getattr(
                    self.worker, "routed_vram_policy", "fixed"
                ),
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
                    "presence_penalty": (
                        self.default_sampling.presence_penalty
                    ),
                    "profiles": {
                        name: {
                            "temperature": profile.temperature,
                            "top_p": profile.top_p,
                            "top_k": profile.top_k,
                            "min_p": profile.min_p,
                            "presence_penalty": profile.presence_penalty,
                        }
                        for name, profile in self.sampling_profiles.items()
                    },
                    "source": (
                        "manifest.tokenizer.sampling"
                        if self.manifest.get("tokenizer", {}).get("sampling")
                        is not None
                        else "tokenizer/generation_config.json"
                    ),
                },
            },
            "worker_kv": {
                "dtype": self.worker.kv_dtype,
                "allocation": self.worker.kv_allocation,
                "page_tokens": self.worker.kv_page_tokens,
                "page_bytes": self.worker.kv_page_bytes,
                "page_capacity": self.worker.kv_page_capacity,
                "allocated_pages": kv_stats.get("allocated_pages", 0),
                "reserved_pages": kv_stats.get("reserved_pages", 0),
            },
            "worker_sessions": {
                "enabled": self.retention_enabled(),
                "parking_enabled": getattr(
                    self.worker, "session_parking", False
                ),
                "persistence_supported": getattr(
                    self.worker, "session_persistence", False
                ),
                "persistence_enabled": getattr(
                    self, "session_cache_enabled", False
                ),
                "park_ram_bytes": getattr(
                    self.worker, "session_park_ram_bytes", 0
                ),
                "park_page_capacity": getattr(
                    self.worker, "session_park_page_capacity", 0
                ),
                "retained": session_count,
                "retained_tokens": session_tokens,
                "reserved_pages": session_pages,
                "parked_bytes": session_parked_bytes,
                "disk_retained": disk_session_count,
                "disk_retained_tokens": disk_session_tokens,
                "disk_bytes": disk_session_bytes,
                "disk_byte_limit": getattr(
                    self.args, "session_cache_bytes", 0
                ),
                "disk_ttl_seconds": getattr(
                    self.args, "session_cache_ttl_seconds", 0.0
                ),
            },
            "worker_runtime": {
                **runtime_stats,
                "stats_snapshot_age_seconds": self.worker_stats_age_seconds(),
            },
            "runtime_config": {
                "host": self.args.host,
                "port": self.args.port,
                "max_context": self.args.max_context,
                "maximum_new_tokens": self.args.maximum_new_tokens,
                "maximum_thinking_tokens": getattr(
                    self, "maximum_thinking_tokens", None
                ),
                "maximum_queue": self.args.maximum_queue,
                "worker_capacity": self.args.worker_capacity,
                "worker_ram_cache_gib": self.args.worker_ram_cache_gib,
                "worker_vram_cache_gib": self.args.worker_vram_cache_gib,
                "worker_routed_vram_policy":
                    getattr(self.args, "worker_routed_vram_policy", "fixed"),
                "worker_active_expert_devices":
                    getattr(self.args, "worker_active_expert_devices", ""),
                "worker_active_expert_device_cache_gib":
                    getattr(
                        self.args, "worker_active_expert_device_cache_gib", 0
                    ),
                "worker_active_expert_host_cache_gib":
                    getattr(
                        self.args, "worker_active_expert_host_cache_gib", 0
                    ),
                "placement_profile": self.args.placement_profile,
                "worker_kv_cache_mib": self.args.worker_kv_cache_mib,
                "worker_kv_page_tokens": self.args.worker_kv_page_tokens,
                "worker_kv_cache_dtype": getattr(
                    self.args, "worker_kv_cache_dtype", "artifact"
                ),
                "worker_prefill_chunk_tokens":
                    self.args.worker_prefill_chunk_tokens,
                "session_retention": self.retention_enabled(),
                "session_idle_seconds": self.args.session_idle_seconds,
                "session_cache_enabled": getattr(
                    self, "session_cache_enabled", False
                ),
                "session_cache_bytes": getattr(
                    self.args, "session_cache_bytes", 0
                ),
                "session_cache_ttl_seconds":
                    getattr(self.args, "session_cache_ttl_seconds", 0.0),
                "microbatch_window_ms": self.args.microbatch_window_ms,
                "latency_window": self.args.latency_window,
                "queue_timeout_seconds": self.args.queue_timeout,
                "generation_timeout_seconds": self.args.generation_timeout,
                "maximum_body_bytes": self.args.maximum_body_bytes,
                "maximum_image_pixels": getattr(
                    self.args, "maximum_image_pixels", 2 << 20
                ),
                "maximum_image_patch_tokens":
                    getattr(self.args, "maximum_image_patch_tokens", 4096),
                "raw_response_trace_enabled":
                    getattr(self, "raw_response_trace_path", None) is not None,
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
    anthropic_heartbeat_seconds = 10.0

    @property
    def app(self) -> Application:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        log("http", client=self.client_address[0], message=format % args)

    def _authorized(self) -> bool:
        expected = self.app.args.api_key
        if not expected:
            return True
        authorization = self.headers.get("Authorization", "")
        api_key = self.headers.get("x-api-key", "")
        return (hmac.compare_digest(authorization, f"Bearer {expected}") or
                hmac.compare_digest(api_key, expected))

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

    def _anthropic_error(self, status: int, message: str,
                         kind: str = "invalid_request_error") -> None:
        self._json(status, {
            "type": "error", "error": {"type": kind, "message": message},
            "request_id": self._request_id(),
        })

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

    def _anthropic_sse(self, event: str, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.wfile.write(b"event: " + event.encode() + b"\n")
        self.wfile.write(b"data: " + encoded + b"\n\n")
        self.wfile.flush()

    def _send_generation_error(
        self, endpoint: str, stream_started: bool, status: int,
        message: str, kind: str, code: str,
    ) -> bool:
        """Return false when the peer vanished before the error was delivered."""
        if self._client_disconnected():
            return False
        try:
            if not stream_started:
                if endpoint == "anthropic":
                    self._anthropic_error(status, message, "api_error")
                else:
                    self._error(status, message, kind, code=code)
            elif endpoint == "anthropic":
                self._anthropic_sse("error", {
                    "type": "error", "error": {
                        "type": "api_error", "message": message,
                    },
                })
            elif endpoint == "responses":
                self._sse({"type": "error", "code": code,
                           "message": message, "param": None})
            else:
                self._sse({"error": {"message": message,
                           "type": kind, "param": None, "code": code}})
                self._sse("[DONE]")
        except ConnectionError:
            return False
        return True

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

    def _run_generation(
        self, request: GenerationRequest, emit: Any,
        context: RequestContext,
        progress_callback: Callable[[], None] | None = None,
    ) -> tuple[str, int, str]:
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
                progress_callback=progress_callback,
                media_packet=request.media_packet,
                media_signature=request.media_signature,
                preserve_special_tokens=(
                    request.endpoint != "completion"
                    and self.app.response_protocol is not None
                ),
            )
        except TypeError as error:
            if not any(name in str(error) for name in (
                    "cancel_check", "cache_prefix_tokens", "sampling",
                    "progress_callback", "preserve_special_tokens")):
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

    @staticmethod
    def _anthropic_tool_id(call: ToolCall) -> str:
        suffix = call.call_id.removeprefix("call_")
        return "toolu_" + suffix

    @staticmethod
    def _anthropic_stop_reason(finish_reason: str,
                               calls: tuple[ToolCall, ...]) -> str:
        if calls:
            return "tool_use"
        return "max_tokens" if finish_reason == "length" else "end_turn"

    def _anthropic_content(self, text: str,
                           calls: tuple[ToolCall, ...]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if text or not calls:
            content.append({"type": "text", "text": text})
        for call in calls:
            content.append({
                "type": "tool_use", "id": self._anthropic_tool_id(call),
                "name": call.name, "input": json.loads(call.arguments),
            })
        return content

    def _anthropic_message(self, request: GenerationRequest, message_id: str,
                           text: str, completion_tokens: int,
                           finish_reason: str,
                           calls: tuple[ToolCall, ...]) -> dict[str, Any]:
        return {
            "id": message_id, "type": "message", "role": "assistant",
            "model": self.app.args.model,
            "content": self._anthropic_content(text, calls),
            "stop_reason": self._anthropic_stop_reason(finish_reason, calls),
            "stop_sequence": None,
            "usage": {
                "input_tokens": len(request.prompt_ids),
                "output_tokens": completion_tokens,
            },
        }

    def _serve_anthropic(self, request: GenerationRequest, message_id: str,
                         context: RequestContext) -> None:
        if not request.stream:
            raw_text, completion_count, finish_reason = self._run_generation(
                request, lambda _delta: None, context
            )
            parsed = self.app.parse_assistant_output(raw_text, request)
            calls = parsed.tool_calls
            self._json(HTTPStatus.OK, self._anthropic_message(
                request, message_id, parsed.text, completion_count,
                finish_reason, calls,
            ))
            return

        self._sse_headers()
        self._anthropic_sse("message_start", {
            "type": "message_start",
            "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "model": self.app.args.model, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": len(request.prompt_ids),
                          "output_tokens": 0},
            },
        })
        last_heartbeat = time.monotonic()

        def heartbeat() -> None:
            nonlocal last_heartbeat
            now = time.monotonic()
            if now - last_heartbeat < self.anthropic_heartbeat_seconds:
                return
            if self._client_disconnected():
                raise BrokenPipeError("streaming client disconnected")
            self._anthropic_sse("ping", {"type": "ping"})
            last_heartbeat = now

        index = 0
        text_open = False
        streamed_text: list[str] = []
        parser = AssistantStreamParser(self.app, request)
        text_field = StreamingTextField()

        def emit_confirmed_text(delta: str) -> None:
            nonlocal text_open
            if not delta:
                return
            if not text_open:
                self._anthropic_sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""},
                })
                text_open = True
            streamed_text.append(delta)
            self._anthropic_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": delta},
            })

        def emit_stream_text(delta: str) -> None:
            emit_confirmed_text(text_field.feed(delta))

        def emit(delta: str) -> None:
            _reasoning, visible = parser.feed(delta)
            emit_stream_text(visible)

        raw_text, completion_count, finish_reason = self._run_generation(
            request, emit, context, progress_callback=heartbeat,
        )
        _reasoning_tail, visible_tail, parsed = parser.finish()
        emit_stream_text(visible_tail)
        text_field.finish()
        calls = parsed.tool_calls
        if text_open:
            self._anthropic_sse("content_block_stop", {
                "type": "content_block_stop", "index": index,
            })
            index += 1
        if not calls and not text_open:
            self._anthropic_sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""},
            })
            self._anthropic_sse("content_block_stop", {
                "type": "content_block_stop", "index": index,
            })
            index += 1
        for call in calls:
            self._anthropic_sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {
                    "type": "tool_use", "id": self._anthropic_tool_id(call),
                    "name": call.name, "input": {},
                },
            })
            self._anthropic_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta",
                          "partial_json": call.arguments},
            })
            self._anthropic_sse("content_block_stop", {
                "type": "content_block_stop", "index": index,
            })
            index += 1

        self._anthropic_sse("message_delta", {
            "type": "message_delta",
            "delta": {
                "stop_reason": self._anthropic_stop_reason(finish_reason, calls),
                "stop_sequence": None,
            },
            "usage": {"output_tokens": completion_count},
        })
        self._anthropic_sse("message_stop", {"type": "message_stop"})

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
        path = urlsplit(self.path).path
        anthropic = path in {"/v1/messages", "/v1/messages/count_tokens"}
        if not self._authorized():
            if anthropic:
                self._anthropic_error(
                    HTTPStatus.UNAUTHORIZED, "invalid API key",
                    "authentication_error",
                )
            else:
                self._error(HTTPStatus.UNAUTHORIZED, "invalid API key",
                            "authentication_error", code="invalid_api_key")
            return
        endpoint = {
            "/v1/completions": "completion",
            "/v1/chat/completions": "chat",
            "/v1/responses": "responses",
            "/v1/messages": "anthropic",
            "/v1/messages/count_tokens": "anthropic_count",
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
            if endpoint in {"anthropic", "anthropic_count"}:
                normalized = self.app.anthropic_payload(
                    payload, count_only=endpoint == "anthropic_count"
                )
                request = self.app.parse_request(normalized, "anthropic")
            else:
                request = self.app.parse_request(payload, endpoint)
        except RequestError as error:
            if anthropic:
                self._anthropic_error(HTTPStatus.BAD_REQUEST, str(error))
            else:
                self._error(HTTPStatus.BAD_REQUEST, str(error),
                            param=error.param, code=error.code)
            return
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            if anthropic:
                self._anthropic_error(HTTPStatus.BAD_REQUEST, str(error))
            else:
                self._error(HTTPStatus.BAD_REQUEST, str(error), code="invalid_json")
            return
        except Exception as error:
            log("request_preprocessing_failed", error=str(error), endpoint=endpoint)
            if anthropic:
                self._anthropic_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "request preprocessing failed", "api_error",
                )
            else:
                self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "request preprocessing failed",
                    "server_error",
                    code="request_preprocessing_failed",
                )
            return
        if endpoint == "anthropic_count":
            self._json(HTTPStatus.OK, {"input_tokens": len(request.prompt_ids)})
            return
        if not self.app.acquire():
            if anthropic:
                self._anthropic_error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "service overloaded or draining", "overloaded_error",
                )
            else:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                            "service overloaded or draining",
                            "server_error", code="overloaded")
            return
        if not self.app.acquire_worker_slot():
            self.app.release()
            if anthropic:
                self._anthropic_error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "all model slots are busy", "overloaded_error",
                )
            else:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                            "all model slots are busy", "server_error",
                            code="overloaded")
            return
        if request.media_signature is None:
            context = self.app.acquire_request_context(
                request.prompt_ids, request.maximum
            )
        else:
            context = self.app.acquire_request_context(
                request.prompt_ids, request.maximum, request.media_signature
            )
        if context is None:
            self.app.release_worker_slot()
            self.app.release()
            if anthropic:
                self._anthropic_error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "KV context capacity is exhausted", "overloaded_error",
                )
            else:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                            "KV context capacity is exhausted", "server_error",
                            code="context_capacity_exhausted")
            return
        prefix = {"chat": "chatcmpl-", "completion": "cmpl-",
                  "responses": "resp_", "anthropic": "msg_"}[endpoint]
        request_uuid = prefix + uuid.uuid4().hex
        message_uuid = "msg_" + uuid.uuid4().hex
        created = int(time.time())
        stream_started = False
        try:
            if endpoint == "anthropic":
                stream_started = request.stream
                self._serve_anthropic(request, request_uuid, context)
                self.app.increment("completed")
                return
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
                        reasoning_tail, visible_tail, _parsed = \
                            reasoning_filter.finish()
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
                    visible_pieces: list[str] = []
                    tool_response = bool(
                        chat and request.tools and request.tool_choice != "none"
                    )
                    stream_tool_response = bool(
                        tool_response and reasoning_filter is not None and
                        reasoning_filter.parser is not None
                    )
                    text_field = StreamingTextField() \
                        if tool_response else None
                    reasoning_for_usage = ""

                    def emit_reasoning(reasoning: str) -> None:
                        if not reasoning:
                            return
                        reasoning_pieces.append(reasoning)
                        self._sse({**base, "choices": [{
                            "index": 0, "finish_reason": None,
                            "logprobs": None,
                            "delta": {"reasoning_content": reasoning},
                        }]})

                    def send_visible(visible: str) -> None:
                        if not visible:
                            return
                        visible_pieces.append(visible)
                        self._sse({**base, "choices": [{
                            "index": 0, "finish_reason": None,
                            "logprobs": None,
                            "delta": {"content": visible},
                        }]})

                    def emit_visible(visible: str) -> None:
                        if text_field is not None:
                            visible = text_field.feed(visible)
                        send_visible(visible)

                    def emit_completion(delta: str) -> None:
                        if self._client_disconnected():
                            raise BrokenPipeError("streaming client disconnected")
                        if chat:
                            assert reasoning_filter is not None
                            reasoning, visible = reasoning_filter.feed(delta)
                            emit_reasoning(reasoning)
                            emit_visible(visible)
                        else:
                            self._sse({**base, "choices": [{
                                "index": 0, "finish_reason": None,
                                "logprobs": None, "text": delta,
                            }]})

                    calls: tuple[ToolCall, ...] = ()
                    raw_text, completion_count, finish_reason = self._run_generation(
                        request,
                        emit_completion if not tool_response or stream_tool_response
                        else lambda _delta: None,
                        context,
                    )
                    streamed_output: AssistantOutput | None = None
                    if reasoning_filter is None or (
                            tool_response and not stream_tool_response):
                        reasoning_tail, visible_tail = "", ""
                    else:
                        reasoning_tail, visible_tail, streamed_output = \
                            reasoning_filter.finish()
                    if chat:
                        emit_reasoning(reasoning_tail)
                        emit_visible(visible_tail)

                    if tool_response:
                        parsed = streamed_output if stream_tool_response else \
                            self.app.parse_assistant_output(raw_text, request)
                        assert parsed is not None
                        calls = parsed.tool_calls
                        reasoning_for_usage = "".join(reasoning_pieces) \
                            if stream_tool_response else parsed.reasoning
                        if not stream_tool_response:
                            emit_reasoning(parsed.reasoning)
                            send_visible(parsed.text)
                        assert text_field is not None
                        text_field.finish()
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
                        reasoning_for_usage = "".join(reasoning_pieces)
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
                                            reasoning_for_usage
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
        except ConnectionError:
            self.app.increment("cancelled")
            log("request_cancelled", id=request_uuid, reason="client_disconnect")
        except TimeoutError as error:
            if self._send_generation_error(
                    endpoint, stream_started, HTTPStatus.GATEWAY_TIMEOUT,
                    str(error), "timeout_error", "generation_timeout"):
                self.app.increment("failed")
            else:
                self.app.increment("cancelled")
                log("request_cancelled", id=request_uuid,
                    reason="client_disconnect", error=repr(error))
        except Exception as error:
            if self._send_generation_error(
                    endpoint, stream_started,
                    HTTPStatus.INTERNAL_SERVER_ERROR, "generation failed",
                    "server_error", "generation_failed"):
                self.app.increment("failed")
                log("request_failed", id=request_uuid, error=repr(error))
            else:
                self.app.increment("cancelled")
                log("request_cancelled", id=request_uuid,
                    reason="client_disconnect", error=repr(error))
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
    parser.add_argument("--build-id", default="development")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default=os.environ.get("EXPERT_API_KEY", ""))
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--maximum-new-tokens", type=int, default=512)
    parser.add_argument("--maximum-queue", type=int, default=8)
    parser.add_argument("--worker-capacity", type=int, default=1)
    parser.add_argument("--worker-ram-cache-gib", type=int, default=48)
    parser.add_argument("--worker-vram-cache-gib", type=int, default=12)
    parser.add_argument(
        "--worker-routed-vram-policy", choices=("fixed", "fit"),
        default="fixed",
    )
    parser.add_argument("--worker-active-expert-devices", default="")
    parser.add_argument(
        "--worker-active-expert-device-cache-gib", type=int, default=0
    )
    parser.add_argument(
        "--worker-active-expert-host-cache-gib", type=int, default=0
    )
    parser.add_argument(
        "--placement-profile", choices=("latency", "balanced", "capacity"),
        default="balanced",
    )
    parser.add_argument("--worker-kv-cache-mib", type=int, default=2048)
    parser.add_argument("--worker-kv-page-tokens", type=int, default=256)
    parser.add_argument(
        "--worker-kv-cache-dtype", default="artifact",
        choices=(
            "artifact",
            "fp8-e4m3-per-head",
            "fp4-e2m1-ue8m0-block32-key-outlier1",
            "q4-bfp16-block32-key-outlier1",
            "q4-bfp16-block32",
            "q4-f16-per-head",
            "q5-q4-bfp16-block32",
            "fp16",
        ),
    )
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
    parser.add_argument("--session-cache-root", type=Path)
    parser.add_argument("--session-cache-bytes", type=int, default=0)
    parser.add_argument(
        "--session-cache-ttl-seconds", type=float, default=604800.0
    )
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
    parser.add_argument("--maximum-image-pixels", type=int, default=2 << 20)
    parser.add_argument(
        "--maximum-image-patch-tokens", type=int, default=4096
    )
    parser.add_argument("--queue-timeout", type=float, default=1.0)
    parser.add_argument("--generation-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--drain-timeout", type=float, default=30.0)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--raw-response-trace-file", type=Path)
    return parser.parse_args()


def main() -> int:
    global LOG_FILE
    args = parse_args()
    if args.worker_route_trace_max_steps < 1:
        raise SystemExit("--worker-route-trace-max-steps must be positive")
    if (args.maximum_queue < 0 or args.max_context < 2 or
        args.maximum_new_tokens < 1 or args.worker_capacity < 1 or
        args.worker_ram_cache_gib < 1 or args.worker_vram_cache_gib < 1 or
        args.worker_active_expert_device_cache_gib < 0 or
        args.worker_active_expert_host_cache_gib < 0 or
        args.worker_kv_cache_mib < 1 or args.worker_kv_page_tokens < 1 or
        args.worker_prefill_chunk_tokens < 0 or
        (args.worker_placement_settle_steps is not None and
         args.worker_placement_settle_steps < 0) or
        args.session_idle_seconds < 0 or args.session_cache_bytes < 0 or
        args.session_cache_ttl_seconds < 0 or
        ((args.session_cache_root is None) !=
         (args.session_cache_bytes == 0)) or
        args.maximum_body_bytes < 1 or
        args.maximum_image_pixels < 65536 or
        args.maximum_image_patch_tokens < 256 or
        args.microbatch_window_ms < 0 or args.latency_window < 1):
        raise SystemExit("invalid service limits")
    active_expert_configured = bool(args.worker_active_expert_devices)
    if (active_expert_configured !=
            (args.worker_active_expert_device_cache_gib > 0) or
            active_expert_configured !=
            (args.worker_active_expert_host_cache_gib > 0)):
        raise SystemExit(
            "secondary expert devices and cache budgets must be configured "
            "together"
        )
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
