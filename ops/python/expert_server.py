#!/usr/bin/env python3
"""Bounded OpenAI-compatible HTTP front-end for the persistent CUDA worker."""

from __future__ import annotations

import argparse
import hmac
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
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit

from transformers import AutoTokenizer


LOG_FILE: Any = None


def log(event: str, **fields: Any) -> None:
    line = json.dumps({"event": event, "time": time.time(), **fields}, separators=(",", ":"))
    print(line, file=sys.stderr, flush=True)
    if LOG_FILE is not None:
        print(line, file=LOG_FILE, flush=True)


class WorkerError(RuntimeError):
    pass


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
                 kv_cache_mib: int, kv_page_tokens: int) -> None:
        command = [
            str(executable), str(container), "--worker", str(max_context),
            str(ram_cache_gib), str(vram_cache_gib), str(requested_capacity),
            str(kv_cache_mib), str(kv_page_tokens),
        ]
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
        self.prefill_chunk_tokens = int(response.get("prefill_chunk_tokens", 0))
        self.kv_page_tokens = int(response.get("kv_page_tokens", 0))
        self.kv_page_bytes = int(response.get("kv_page_bytes", 0))
        self.kv_page_capacity = int(response.get("kv_page_capacity", 0))
        if (self.protocol < 3 or self.capacity != requested_capacity or
                self.prefill_chunk_tokens != requested_capacity or
                self.kv_page_tokens != kv_page_tokens or
                self.kv_page_bytes <= 0 or self.kv_page_capacity <= 0):
            self.process.kill()
            raise WorkerError("CUDA worker does not support requested KV/batching contract")
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

    def next(self, request_id: int, final: bool) -> int:
        response = self._command(f"NEXT\t{request_id}\t{1 if final else 0}")
        if response.get("type") != "token" or response.get("id") != request_id:
            raise WorkerError("unexpected NEXT response")
        if final:
            self.active_ids.discard(request_id)
        return int(response["token"])

    def step(self, items: list[tuple[int, bool]]) -> dict[int, int]:
        if not items or len(items) > self.capacity:
            raise WorkerError("invalid decode batch")
        if self.protocol < 2:
            if len(items) != 1:
                raise WorkerError("protocol v1 cannot batch decode")
            request_id, final = items[0]
            return {request_id: self.next(request_id, final)}
        response = self._command("STEP\t" + "\t".join(
            f"{request_id},{1 if final else 0}" for request_id, final in items
        ))
        if response.get("type") != "batch" or not isinstance(response.get("items"), list):
            raise WorkerError("unexpected STEP response")
        result = {int(item["id"]): int(item["token"])
                  for item in response["items"]}
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
        return {
            "allocated_pages": int(response["kv_allocated_pages"]),
            "reserved_pages": int(response["kv_reserved_pages"]),
        }

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
    def __init__(self, request_id: int, final: bool) -> None:
        self.request_id = request_id
        self.final = final
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
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def step(self, request_id: int, final: bool) -> int:
        waiter = DecodeWaiter(request_id, final)
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
            deadline = time.monotonic() + self.window_seconds
            while len(batch) < self.worker.capacity:
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
                tokens = self.worker.step([
                    (item.request_id, item.final) for item in batch
                ])
                self.on_batch(len(batch))
                for item in batch:
                    item.token = tokens[item.request_id]
            except Exception as error:
                for item in batch:
                    item.error = error
            finally:
                for item in batch:
                    item.event.set()

    def close(self) -> None:
        self.pending.put(None)
        self.thread.join(timeout=10)


class Application:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.manifest = json.loads((args.container / "manifest.json").read_text(encoding="utf-8"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer), local_files_only=True, trust_remote_code=False
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
                                 args.worker_kv_page_tokens)
        self.capacity = threading.BoundedSemaphore(
            args.maximum_queue + args.worker_capacity
        )
        self.worker_slots = threading.BoundedSemaphore(args.worker_capacity)
        self.kv_credit_lock = threading.Lock()
        self.kv_reserved_pages = 0
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
        return "\n".join(lines) + "\n"

    def _chat_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
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

    def generate(self, prompt_ids: list[int], maximum: int) -> Iterator[tuple[int, str]]:
        request_id = self.request_id()
        generated: list[int] = []
        decoded = ""
        started = time.monotonic()
        previous_token_at: float | None = None
        self.worker.begin(request_id, prompt_ids, len(prompt_ids) + maximum)
        try:
            for index in range(maximum):
                if time.monotonic() - started > self.args.generation_timeout:
                    raise TimeoutError("generation deadline exceeded")
                token = self.decode_batcher.step(
                    request_id, index + 1 == maximum
                )
                token_at = time.monotonic()
                if previous_token_at is None:
                    self.observe_latency("ttft_seconds", token_at - started)
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
                delta = current[len(decoded):] if current.startswith(decoded) else current
                decoded = current
                yield token, delta
                if token in self.eos_token_ids:
                    break
        finally:
            if request_id in self.worker.active_ids:
                self.worker.cancel(request_id)

    def info(self) -> dict[str, Any]:
        with self.active_lock:
            active = self.active
        kv_stats = self.worker.stats()
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
            "worker_prefill": {
                "mode": "causal_chunked",
                "chunk_tokens": self.worker.prefill_chunk_tokens,
            },
            "worker_kv": {
                "dtype": "fp16",
                "page_tokens": self.worker.kv_page_tokens,
                "page_bytes": self.worker.kv_page_bytes,
                "page_capacity": self.worker.kv_page_capacity,
                **kv_stats,
            },
            "runtime_config": {
                "host": self.args.host,
                "port": self.args.port,
                "max_context": self.args.max_context,
                "maximum_new_tokens": self.args.maximum_new_tokens,
                "maximum_queue": self.args.maximum_queue,
                "worker_capacity": self.args.worker_capacity,
                "worker_ram_cache_gib": self.args.worker_ram_cache_gib,
                "worker_vram_cache_gib": self.args.worker_vram_cache_gib,
                "worker_kv_cache_mib": self.args.worker_kv_cache_mib,
                "worker_kv_page_tokens": self.args.worker_kv_page_tokens,
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

    def _run_generation(self, request: GenerationRequest, emit: Any) -> tuple[str, int, str]:
        pieces: list[str] = []
        count = 0
        finish_reason = "length"
        stop_filter = StopFilter(request.stop)
        generation = self.app.generate(request.prompt_ids, request.maximum)
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
        if not self.app.acquire():
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "service overloaded or draining",
                        "server_error", code="overloaded")
            return
        if not self.app.acquire_worker_slot():
            self.app.release()
            self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                        "all model slots are busy", "server_error", code="overloaded")
            return
        context_pages = self.app.acquire_context_credits(
            len(request.prompt_ids) + request.maximum
        )
        if not context_pages:
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
                        request, emit_response
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
                        request, emit_completion
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
                    request, pieces.append
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
        except Exception as error:
            self.app.increment("failed")
            log("request_failed", id=request_uuid, error=repr(error))
            if not stream_started:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "generation failed",
                            "server_error", code="generation_failed")
            elif endpoint == "responses":
                self._sse({"type": "error", "code": "generation_failed",
                           "message": "generation failed", "param": None})
        finally:
            self.app.release_context_credits(context_pages)
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
    parser.add_argument("--worker-kv-cache-mib", type=int, default=2048)
    parser.add_argument("--worker-kv-page-tokens", type=int, default=256)
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
        args.microbatch_window_ms < 0 or args.latency_window < 1):
        raise SystemExit("invalid service limits")
    if (args.host not in {"127.0.0.1", "::1", "localhost"} and
            not args.api_key):
        raise SystemExit("--api-key or EXPERT_API_KEY is required for non-loopback bind")
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE = args.log_file.open("a", encoding="utf-8")
    app = Application(args)
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
