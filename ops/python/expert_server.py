#!/usr/bin/env python3
"""Bounded OpenAI-compatible HTTP front-end for the persistent CUDA worker."""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer


LOG_FILE: Any = None


def log(event: str, **fields: Any) -> None:
    line = json.dumps({"event": event, "time": time.time(), **fields}, separators=(",", ":"))
    print(line, file=sys.stderr, flush=True)
    if LOG_FILE is not None:
        print(line, file=LOG_FILE, flush=True)


class WorkerError(RuntimeError):
    pass


class CudaWorker:
    def __init__(self, executable: Path, container: Path, max_context: int,
                 startup_timeout: float, requested_capacity: int,
                 ram_cache_gib: int, vram_cache_gib: int) -> None:
        command = [str(executable), str(container), "--worker", str(max_context)]
        if requested_capacity > 1:
            command.extend((str(ram_cache_gib), str(vram_cache_gib),
                            str(requested_capacity)))
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
        if requested_capacity > 1 and (
            self.protocol < 2 or self.capacity != requested_capacity
        ):
            self.process.kill()
            raise WorkerError("CUDA worker does not support requested batching")
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

    def begin(self, request_id: int, prompt_ids: list[int]) -> None:
        if request_id in self.active_ids:
            raise WorkerError("duplicate worker request")
        response = self._command(
            f"BEGIN\t{request_id}\t" + ",".join(str(token) for token in prompt_ids)
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
        self.worker = CudaWorker(args.worker, args.container, args.max_context,
                                 args.startup_timeout, args.worker_capacity,
                                 args.worker_ram_cache_gib,
                                 args.worker_vram_cache_gib)
        self.capacity = threading.BoundedSemaphore(
            args.maximum_queue + args.worker_capacity
        )
        self.worker_slots = threading.BoundedSemaphore(args.worker_capacity)
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
        lines = [
            "# TYPE expert_service_active_requests gauge",
            f"expert_service_active_requests {active}",
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

    def prompt_ids(self, payload: dict[str, Any], chat: bool) -> list[int]:
        if chat:
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty array")
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            prompt = payload.get("prompt")
            if isinstance(prompt, str):
                ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            elif isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
                ids = prompt
            else:
                raise ValueError("prompt must be a string or token-id array")
        ids = [int(token) for token in ids]
        if not ids or len(ids) >= self.args.max_context:
            raise ValueError("prompt is empty or exceeds context capacity")
        return ids

    def generate(self, prompt_ids: list[int], maximum: int) -> Iterator[tuple[int, str]]:
        request_id = self.request_id()
        generated: list[int] = []
        decoded = ""
        started = time.monotonic()
        previous_token_at: float | None = None
        self.worker.begin(request_id, prompt_ids)
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
                    generated, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                delta = current[len(decoded):] if current.startswith(decoded) else current
                decoded = current
                yield token, delta
        finally:
            if request_id in self.worker.active_ids:
                self.worker.cancel(request_id)

    def info(self) -> dict[str, Any]:
        with self.active_lock:
            active = self.active
        return {
            "model": self.args.model,
            "build_id": self.args.build_id,
            "source": self.manifest["source"],
            "format": self.manifest["format"],
            "quantization": self.manifest["quantization"],
            "architecture": self.manifest["architecture"],
            "masses": self.manifest["masses"],
            "active_requests": active,
            "maximum_queue": self.args.maximum_queue,
            "worker_capacity": self.args.worker_capacity,
            "worker_protocol": self.worker.protocol,
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
    server_version = "ExpertRuntime/1"

    @property
    def app(self) -> Application:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        log("http", client=self.client_address[0], message=format % args)

    def _authorized(self) -> bool:
        expected = self.app.args.api_key
        return not expected or self.headers.get("Authorization") == f"Bearer {expected}"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._json(status, {"error": {"message": message, "type": kind}})

    def do_GET(self) -> None:
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "invalid API key", "authentication_error")
            return
        if self.path == "/health":
            status = HTTPStatus.OK if self.app.worker.healthy() else HTTPStatus.SERVICE_UNAVAILABLE
            self._json(status, {"status": "ok" if status == 200 else "failed"})
        elif self.path == "/ready":
            ready = self.app.worker.healthy() and not self.app.draining.is_set()
            self._json(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                       {"ready": ready})
        elif self.path == "/model-info":
            self._json(HTTPStatus.OK, self.app.info())
        elif self.path == "/v1/models":
            self._json(HTTPStatus.OK, {"object": "list", "data": [{
                "id": self.app.args.model, "object": "model", "owned_by": "local"
            }]})
        elif self.path == "/metrics":
            encoded = self.app.metrics_text().encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self._error(HTTPStatus.NOT_FOUND, "route not found")

    def do_POST(self) -> None:
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "invalid API key", "authentication_error")
            return
        chat = self.path == "/v1/chat/completions"
        if not chat and self.path != "/v1/completions":
            self._error(HTTPStatus.NOT_FOUND, "route not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.app.args.maximum_body_bytes:
                raise ValueError("request body size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            if payload.get("model", self.app.args.model) != self.app.args.model:
                raise ValueError("unknown model")
            maximum = int(payload.get("max_completion_tokens", payload.get("max_tokens", 16)))
            if maximum < 1 or maximum > self.app.args.maximum_new_tokens:
                raise ValueError("max_tokens is outside service limits")
            prompt_ids = self.app.prompt_ids(payload, chat)
            if len(prompt_ids) + maximum > self.app.args.max_context:
                raise ValueError("prompt plus max_tokens exceeds context")
            stream = bool(payload.get("stream", False))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if not self.app.acquire():
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "service overloaded or draining", "overload_error")
            return
        if not self.app.acquire_worker_slot():
            self.app.release()
            self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                        "all model slots are busy", "overload_error")
            return
        request_uuid = "cmpl-" + uuid.uuid4().hex
        created = int(time.time())
        try:
            if stream:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for _token, delta in self.app.generate(prompt_ids, maximum):
                    choice = {"index": 0, "finish_reason": None}
                    choice["delta" if chat else "text"] = ({"content": delta} if chat else delta)
                    chunk = {"id": request_uuid, "object": "chat.completion.chunk" if chat else "text_completion",
                             "created": created, "model": self.app.args.model, "choices": [choice]}
                    self.wfile.write(b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n")
                    self.wfile.flush()
                final_choice: dict[str, Any] = {"index": 0, "finish_reason": "length"}
                final_choice["delta" if chat else "text"] = ({} if chat else "")
                final_chunk = {"id": request_uuid,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": created, "model": self.app.args.model,
                    "choices": [final_choice]}
                self.wfile.write(b"data: " + json.dumps(final_chunk, separators=(",", ":")).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                text = "".join(delta for _token, delta in self.app.generate(prompt_ids, maximum))
                choice: dict[str, Any] = {"index": 0, "finish_reason": "length"}
                if chat:
                    choice["message"] = {"role": "assistant", "content": text}
                else:
                    choice["text"] = text
                self._json(HTTPStatus.OK, {"id": request_uuid,
                    "object": "chat.completion" if chat else "text_completion",
                    "created": created, "model": self.app.args.model, "choices": [choice],
                    "usage": {"prompt_tokens": len(prompt_ids),
                              "completion_tokens": maximum,
                              "total_tokens": len(prompt_ids) + maximum}})
            self.app.increment("completed")
        except (BrokenPipeError, ConnectionResetError):
            self.app.increment("cancelled")
            log("request_cancelled", id=request_uuid, reason="client_disconnect")
        except TimeoutError as error:
            self.app.increment("failed")
            if not stream:
                self._error(HTTPStatus.GATEWAY_TIMEOUT, str(error), "timeout_error")
        except Exception as error:
            self.app.increment("failed")
            log("request_failed", id=request_uuid, error=repr(error))
            if not stream:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "generation failed", "server_error")
        finally:
            self.app.release_worker_slot()
            self.app.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--container", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--model", default="olmoe-expert-pack-int8")
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
        args.microbatch_window_ms < 0 or args.latency_window < 1):
        raise SystemExit("invalid service limits")
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
