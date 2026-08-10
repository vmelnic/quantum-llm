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
class GenerationRequest:
    endpoint: str
    prompt_ids: list[int]
    maximum: int
    stream: bool
    stop: tuple[str, ...]
    include_usage: bool
    instructions: str | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None


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
                 enable_mtp: bool, prefill_chunk_tokens: int = 0,
                 placement_settle_steps: int | None = None) -> None:
        command = [
            str(executable), str(container), "--worker", str(max_context),
            str(ram_cache_gib), str(vram_cache_gib), str(requested_capacity),
            str(kv_cache_mib), str(kv_page_tokens), placement_profile,
        ]
        if prefill_chunk_tokens:
            command.append(str(prefill_chunk_tokens))
        if placement_settle_steps is not None:
            # The settle-steps slot is positional after the prefill chunk.
            if not prefill_chunk_tokens:
                command.append("0")
            command.append(str(placement_settle_steps))
        if profile_gpu_phases:
            command.append("--profile-gpu-phases")
        if enable_mtp:
            command.append("--enable-mtp")
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
        self.prefill_mode = str(response.get("prefill_mode", ""))
        self.prefill_chunk_tokens = int(response.get("prefill_chunk_tokens", 0))
        self.session_retention = bool(response.get("session_retention", False))
        self.request_stream_mode = str(
            response.get("request_stream_mode", "default")
        )
        self.gpu_phase_timing = bool(response.get("gpu_phase_timing", False))
        self.mtp_resource_available = bool(
            response.get("mtp_resource_available", False)
        )
        self.mtp_runtime_ready = bool(response.get("mtp_runtime_ready", False))
        self.mtp_enabled = bool(response.get("mtp_enabled", False))
        if self.mtp_runtime_ready and not self.mtp_resource_available:
            self.process.kill()
            raise WorkerError("MTP runtime cannot be ready without resources")
        if self.mtp_enabled != enable_mtp:
            self.process.kill()
            raise WorkerError("CUDA worker MTP mode does not match the request")
        self.rope_mode = str(response.get("rope_mode", "per_step_upload"))
        self.kv_dtype = str(response.get("kv_dtype", ""))
        self.kv_allocation = str(response.get("kv_allocation", ""))
        self.kv_page_tokens = int(response.get("kv_page_tokens", 0))
        self.kv_page_bytes = int(response.get("kv_page_bytes", 0))
        self.kv_page_capacity = int(response.get("kv_page_capacity", 0))
        self.placement_profile = str(response.get("placement_profile", ""))
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
        if (self.protocol < 4 or self.capacity != requested_capacity or
                self.prefill_mode not in {"causal_chunked", "causal_sequential"} or
                not 1 <= self.prefill_chunk_tokens <= max_context or
                (prefill_chunk_tokens and
                 self.prefill_chunk_tokens != prefill_chunk_tokens) or
                self.kv_dtype not in {"fp16", "bf16"} or
                self.kv_allocation not in {"paged_on_demand", "preallocated"} or
                self.kv_page_tokens != kv_page_tokens or
                self.kv_page_bytes <= 0 or self.kv_page_capacity <= 0 or
                self.placement_profile != placement_profile or
                self.ram_cache_bytes != ram_cache_gib << 30 or
                self.vram_cache_bytes != vram_cache_gib << 30 or
                self.placement_prefetch_state not in
                    {"disabled", "observing", "ready"} or
                self.placement_prefetch_enabled !=
                    (self.placement_prefetch_state == "ready") or
                (placement_profile == "capacity" and
                 self.placement_prefetch_state != "disabled") or
                self.placement_minimum_observations != expected_observations):
            self.process.kill()
            raise WorkerError(
                "CUDA worker does not support requested runtime contract"
            )
        self.active_ids: set[int] = set()
        self.command_lock = threading.Lock()

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

    def _command(self, command: str) -> dict[str, Any]:
        with self.command_lock:
            if self.process.poll() is not None:
                raise WorkerError("CUDA worker is not running")
            assert self.process.stdin
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            return self._read()

    def begin(self, request_id: int, prompt_ids: list[int], context_limit: int) -> None:
        if request_id in self.active_ids:
            raise WorkerError("duplicate worker request")
        response = self._command(
            f"BEGIN\t{request_id}\t{context_limit}\t" +
            ",".join(str(token) for token in prompt_ids)
        )
        if response.get("type") != "begun" or response.get("id") != request_id:
            raise WorkerError("unexpected BEGIN response")
        self.active_ids.add(request_id)

    def begin_resume(self, request_id: int, session_key: int,
                     delta_ids: list[int], context_limit: int) -> None:
        if request_id in self.active_ids:
            raise WorkerError("duplicate worker request")
        if not delta_ids:
            raise WorkerError("resume requires at least one delta token")
        response = self._command(
            f"BEGIN\t{request_id}\t{context_limit}\t" +
            ",".join(str(token) for token in delta_ids) +
            f"\tRESUME\t{session_key}"
        )
        if response.get("type") != "begun" or response.get("id") != request_id:
            raise WorkerError("unexpected BEGIN response")
        self.active_ids.add(request_id)

    def end_retain(self, request_id: int, session_key: int) -> int:
        response = self._command(f"END\t{request_id}\tRETAIN\t{session_key}")
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
        self.checkpoint_chat_encoder: Callable[..., str] | None = None
        if args.model == "deepseek-v4-flash" and not self.tokenizer.chat_template:
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
        # The DeepSeek worker prefills sequentially (chunk 1) and rejects extra
        # positional arguments; only the Qwen runner takes a prefill chunk.
        prefill_chunk = (
            0 if args.model == "deepseek-v4-flash"
            else args.worker_prefill_chunk_tokens
        )
        settle_steps = (
            None if args.model == "deepseek-v4-flash"
            else args.worker_placement_settle_steps
        )
        self.worker = CudaWorker(args.worker, args.container, args.max_context,
                                 args.startup_timeout, args.worker_capacity,
                                 args.worker_ram_cache_gib,
                                 args.worker_vram_cache_gib,
                                 args.worker_kv_cache_mib,
                                 args.worker_kv_page_tokens,
                                 args.placement_profile,
                                 args.profile_gpu_phases,
                                 args.enable_mtp, prefill_chunk,
                                 settle_steps)
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
        # retained turns require.
        total_pages = self._context_pages(len(prompt_ids) + maximum + 1)
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

    def _chat_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        encoder = getattr(self, "checkpoint_chat_encoder", None)
        if encoder is not None:
            prompt = encoder(messages, thinking_mode="chat")
            return [int(token) for token in self.tokenizer.encode(
                prompt, add_special_tokens=False
            )]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token) for token in ids]

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
            if message.get("type", "message") != "message":
                raise RequestError("only message input items are supported",
                                   f"{item_param}.type", "unsupported_value")
            role = message.get("role")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant"}:
                raise RequestError(f"message role {role!r} is not supported",
                                   f"{item_param}.role", "unsupported_value")
            content = _text_content(message.get("content"), f"{item_param}.content")
            result.append({"role": role, "content": content})
        return result

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
        only("temperature", (None, 0, 0.0),
             "sampling is not implemented; omit temperature or use 0")
        only("top_p", (None, 1, 1.0),
             "nucleus sampling is not implemented; omit top_p or use 1")
        only("presence_penalty", (None, 0, 0.0),
             "presence_penalty is not implemented")
        only("frequency_penalty", (None, 0, 0.0),
             "frequency_penalty is not implemented")
        only("logprobs", (None, False, 0), "logprobs are not implemented")
        only("top_logprobs", (None, 0), "top_logprobs are not implemented")
        only("echo", (None, False), "echo is not implemented")
        only("background", (None, False), "background responses are not implemented")
        only("previous_response_id", (None,), "stored response chaining is not implemented")
        only("conversation", (None,), "server-side conversations are not implemented")
        only("truncation", (None, "disabled"), "automatic truncation is not implemented")

        if payload.get("logit_bias") not in (None, {}):
            raise RequestError("logit_bias is not implemented", "logit_bias", "unsupported_value")
        if payload.get("tools") not in (None, []):
            raise RequestError("tool calling is not implemented", "tools", "unsupported_value")
        if payload.get("functions") not in (None, []):
            raise RequestError("function calling is not implemented", "functions", "unsupported_value")
        if payload.get("modalities") not in (None, ["text"]):
            raise RequestError("only text output is supported", "modalities", "unsupported_value")
        if payload.get("audio") is not None:
            raise RequestError("audio output is not supported", "audio", "unsupported_value")
        if payload.get("prediction") is not None:
            raise RequestError("predicted output is not implemented", "prediction", "unsupported_value")
        if payload.get("reasoning") not in (None, {}):
            raise RequestError("reasoning controls are not implemented", "reasoning", "unsupported_value")
        if payload.get("suffix") is not None:
            raise RequestError("suffix completion is not implemented", "suffix", "unsupported_value")

        tools = payload.get("tools") or payload.get("functions")
        tool_choice = payload.get("tool_choice", payload.get("function_call"))
        if tools or tool_choice not in (None, "none", "auto"):
            raise RequestError("tool choice is not supported", "tool_choice", "unsupported_value")

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

        if endpoint == "chat":
            prompt_ids = self._chat_prompt_ids(self._messages(payload.get("messages")))
        elif endpoint == "responses":
            raw_input = payload.get("input")
            if isinstance(raw_input, str):
                messages = [{"role": "user", "content": raw_input}]
            else:
                messages = self._messages(raw_input, "input")
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            prompt_ids = self._chat_prompt_ids(messages)
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

        if not prompt_ids or len(prompt_ids) >= self.args.max_context:
            raise RequestError("prompt is empty or exceeds context capacity",
                               "input" if endpoint == "responses" else "prompt")
        if len(prompt_ids) + maximum > self.args.max_context:
            raise RequestError("prompt plus output tokens exceeds context capacity", max_field)
        return GenerationRequest(
            endpoint=endpoint, prompt_ids=prompt_ids, maximum=maximum,
            stream=stream, stop=self._stop_sequences(payload.get("stop")),
            include_usage=self._stream_usage(payload), instructions=instructions,
            metadata=metadata, user=user,
        )

    _TELEMETRY_DELTA_KEYS = (
        "forward_calls", "forward_wall_ns", "expert_cache_wait_ns",
        "expert_compute_ns", "cpu_expert_ns", "gpu_expert_ns",
        "cpu_gpu_overlap_ns", "final_head_ns", "cache_read_bytes",
        "cache_uploaded_bytes", "cache_storage_wait_ns",
        "cache_upload_wait_ns", "worker_model_steps", "worker_model_step_ns",
        "frozen_promotions", "frozen_promotion_bytes",
        "worker_scheduler_poll_ns", "worker_output_head_ns",
        "scheduler_expert_wait_ns", "worker_mtp_drafts",
        "worker_mtp_accepted", "worker_mtp_rejected", "worker_verify_pairs",
        "worker_mtp_suppressions",
    )

    @staticmethod
    def _capacity_error(error: Exception) -> bool:
        message = str(error)
        return "capacity" in message or "slot available" in message or \
            "credits" in message

    def generate(self, prompt_ids: list[int], maximum: int,
                 context: RequestContext | None = None
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
        while True:
            try:
                if session is not None:
                    delta = prompt_ids[len(session.tokens):]
                    self.worker.begin_resume(request_id, session.key, delta,
                                             context_limit)
                    resumed = True
                    prefill_tokens = len(delta)
                else:
                    self.worker.begin(request_id, prompt_ids, context_limit)
                break
            except WorkerError as error:
                if self._capacity_error(error) and self.evict_lru_session():
                    continue
                if session is not None:
                    # The retained state is gone or inconsistent; fall back
                    # to a fresh full prefill.
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
                for key in self._TELEMETRY_DELTA_KEYS
                if key in stats_before and key in stats_after
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
            "worker_placement": {
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
    def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0, "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        }

    @staticmethod
    def _responses_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
        return {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _run_generation(self, request: GenerationRequest, emit: Any,
                        context: RequestContext) -> tuple[str, int, str]:
        pieces: list[str] = []
        count = 0
        finish_reason = "length"
        stop_filter = StopFilter(request.stop)
        generation = self.app.generate(request.prompt_ids, request.maximum,
                                       context)
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
                         completion_tokens: int, status: str = "completed") -> dict[str, Any]:
        completed = status == "completed"
        output = [] if not completed else [{
            "id": message_id, "type": "message", "status": "completed",
            "role": "assistant", "content": [{
                "type": "output_text", "text": text, "annotations": [],
                "logprobs": [],
            }],
        }]
        return {
            "id": response_id, "object": "response", "created_at": created,
            "status": status, "completed_at": int(time.time()) if completed else None,
            "error": None, "incomplete_details": None,
            "instructions": request.instructions,
            "max_output_tokens": request.maximum, "model": self.app.args.model,
            "output": output, "parallel_tool_calls": True,
            "previous_response_id": None, "reasoning": {"effort": None, "summary": None},
            "store": False, "temperature": 0.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "none", "tools": [], "top_p": 1.0,
            "truncation": "disabled",
            "usage": self._responses_usage(len(request.prompt_ids), completion_tokens)
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
                    item = {"id": message_uuid, "type": "message", "status": "in_progress",
                            "role": "assistant", "content": []}
                    self._sse({"type": "response.output_item.added", "output_index": 0,
                               "item": item, "sequence_number": sequence})
                    sequence += 1
                    part = {"type": "output_text", "text": "", "annotations": [],
                            "logprobs": []}
                    self._sse({"type": "response.content_part.added", "item_id": message_uuid,
                               "output_index": 0, "content_index": 0, "part": part,
                               "sequence_number": sequence})
                    sequence += 1

                    def emit_response(delta: str) -> None:
                        nonlocal sequence
                        if self._client_disconnected():
                            raise BrokenPipeError("streaming client disconnected")
                        self._sse({"type": "response.output_text.delta",
                                   "item_id": message_uuid, "output_index": 0,
                                   "content_index": 0, "delta": delta,
                                   "logprobs": [], "sequence_number": sequence})
                        sequence += 1

                    text, completion_count, _finish_reason = self._run_generation(
                        request, emit_response, context
                    )
                    self._sse({"type": "response.output_text.done", "item_id": message_uuid,
                               "output_index": 0, "content_index": 0, "text": text,
                               "logprobs": [], "sequence_number": sequence})
                    sequence += 1
                    done_part = {"type": "output_text", "text": text,
                                 "annotations": [], "logprobs": []}
                    self._sse({"type": "response.content_part.done", "item_id": message_uuid,
                               "output_index": 0, "content_index": 0, "part": done_part,
                               "sequence_number": sequence})
                    sequence += 1
                    done_item = {"id": message_uuid, "type": "message",
                                 "status": "completed", "role": "assistant",
                                 "content": [done_part]}
                    self._sse({"type": "response.output_item.done", "output_index": 0,
                               "item": done_item, "sequence_number": sequence})
                    sequence += 1
                    completed = self._response_object(
                        request, request_uuid, message_uuid, created, text, completion_count
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

                    def emit_completion(delta: str) -> None:
                        if self._client_disconnected():
                            raise BrokenPipeError("streaming client disconnected")
                        choice: dict[str, Any] = {"index": 0, "finish_reason": None,
                                                  "logprobs": None}
                        choice["delta" if chat else "text"] = ({"content": delta}
                                                                 if chat else delta)
                        self._sse({**base, "choices": [choice]})

                    _text, completion_count, finish_reason = self._run_generation(
                        request, emit_completion, context
                    )
                    final_choice: dict[str, Any] = {
                        "index": 0, "finish_reason": finish_reason, "logprobs": None,
                    }
                    final_choice["delta" if chat else "text"] = {} if chat else ""
                    self._sse({**base, "choices": [final_choice]})
                    if request.include_usage:
                        self._sse({**base, "choices": [],
                                   "usage": self._usage(len(request.prompt_ids),
                                                        completion_count)})
                    self._sse("[DONE]")
            else:
                pieces: list[str] = []
                text, completion_count, finish_reason = self._run_generation(
                    request, pieces.append, context
                )
                if endpoint == "responses":
                    self._json(HTTPStatus.OK, self._response_object(
                        request, request_uuid, message_uuid, created, text, completion_count
                    ))
                else:
                    chat = endpoint == "chat"
                    choice: dict[str, Any] = {
                        "index": 0, "finish_reason": finish_reason, "logprobs": None,
                    }
                    if chat:
                        choice["message"] = {"role": "assistant", "content": text,
                                             "refusal": None, "annotations": []}
                    else:
                        choice["text"] = text
                    self._json(HTTPStatus.OK, {"id": request_uuid,
                        "object": "chat.completion" if chat else "text_completion",
                        "created": created, "model": self.app.args.model,
                        "system_fingerprint": "fp_" + self.app.args.build_id,
                        "choices": [choice],
                        "usage": self._usage(len(request.prompt_ids), completion_count),
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
        "--model", default="qwen3-next-80b-a3b-expert-pack-int8"
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
    parser.add_argument("--worker-vram-cache-gib", type=int, default=14)
    parser.add_argument(
        "--placement-profile", choices=("latency", "balanced", "capacity"),
        default="balanced",
    )
    parser.add_argument("--worker-kv-cache-mib", type=int, default=2048)
    parser.add_argument("--worker-kv-page-tokens", type=int, default=256)
    parser.add_argument("--worker-prefill-chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--worker-placement-settle-steps", type=int, default=None,
        help="decode steps before the Qwen worker freezes adaptive placement "
             "(0 disables the freeze; omitted keeps the worker default)",
    )
    parser.add_argument("--disable-session-retention", action="store_true")
    parser.add_argument("--session-idle-seconds", type=float, default=1800.0)
    parser.add_argument("--profile-gpu-phases", action="store_true")
    parser.add_argument("--enable-mtp", action="store_true")
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
    if (args.maximum_queue < 0 or args.max_context < 2 or
        args.maximum_new_tokens < 1 or args.worker_capacity < 1 or
        args.worker_ram_cache_gib < 1 or args.worker_vram_cache_gib < 1 or
        args.worker_kv_cache_mib < 1 or args.worker_kv_page_tokens < 1 or
        args.worker_prefill_chunk_tokens < 1 or
        (args.worker_placement_settle_steps is not None and
         args.worker_placement_settle_steps < 0) or
        args.session_idle_seconds < 0 or
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
