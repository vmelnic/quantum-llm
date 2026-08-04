from __future__ import annotations

import sys
import socket
import threading
import types
import unittest
import http.client
import json
from http.server import ThreadingHTTPServer

# The batching coordinator itself has no tokenizer dependency. Keep this unit
# test runnable in the build Python used by CMake, where transformers is not a
# required package.
sys.modules.setdefault(
    "transformers", types.SimpleNamespace(AutoTokenizer=object)
)

from ops.python.expert_server import (
    Application, ContinuousDecodeBatcher, Handler, RequestError, StopFilter,
    _text_content,
)


class FakeWorker:
    capacity = 4

    def __init__(self) -> None:
        self.calls: list[list[tuple[int, bool]]] = []
        self.lock = threading.Lock()

    def step(self, items: list[tuple[int, bool]]) -> dict[int, int]:
        with self.lock:
            self.calls.append(list(items))
        return {request_id: request_id * 10 + int(final)
                for request_id, final in items}


class ContinuousDecodeBatcherTests(unittest.TestCase):
    def test_http_openai_response_and_stream_contracts(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, _messages: object, **_kwargs: object) -> list[int]:
                return [10, 11]

        app = Application.__new__(Application)
        app.args = types.SimpleNamespace(
            model="test-model", build_id="build", maximum_new_tokens=32,
            max_context=128, maximum_body_bytes=1 << 20, api_key="",
        )
        app.tokenizer = Tokenizer()
        app.eos_token_ids = set()
        app.worker = types.SimpleNamespace(healthy=lambda: True)
        app.draining = threading.Event()
        app.acquire = lambda: True
        app.release = lambda: None
        app.acquire_worker_slot = lambda: True
        app.release_worker_slot = lambda: None
        app.increment = lambda *_args, **_kwargs: None

        def generate(_prompt: list[int], maximum: int):
            for index, value in enumerate(("hello", " world")[:maximum]):
                yield index + 1, value

        app.generate = generate
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.app = app
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("POST", "/v1/responses", body=json.dumps({
                "model": "test-model", "input": "hello", "max_output_tokens": 2,
                "temperature": 0,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertTrue(response.getheader("x-request-id").startswith("req_"))
            self.assertEqual(payload["object"], "response")
            self.assertEqual(payload["output"][0]["content"][0]["text"], "hello world")
            self.assertEqual(payload["usage"]["output_tokens"], 2)

            connection.request("POST", "/v1/responses", body=json.dumps({
                "model": "test-model", "input": "hello", "max_output_tokens": 2,
                "temperature": 0, "stream": True,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            stream = response.read().decode()
            self.assertEqual(response.status, 200)
            self.assertIn('"type":"response.created"', stream)
            self.assertIn('"type":"response.output_text.delta"', stream)
            self.assertIn('"type":"response.completed"', stream)

            connection.request("POST", "/v1/chat/completions", body=json.dumps({
                "model": "test-model", "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 2, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True},
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            stream = response.read().decode()
            self.assertIn('"delta":{"role":"assistant","content":""}', stream)
            self.assertIn('"choices":[],"usage":', stream)
            self.assertTrue(stream.endswith("data: [DONE]\n\n"))
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_stop_filter_hides_cross_token_stop_sequence(self) -> None:
        stop_filter = StopFilter(("<END>",))
        self.assertEqual(stop_filter.feed("answer<EN"), "answer")
        self.assertEqual(stop_filter.feed("D>ignored"), "")
        self.assertTrue(stop_filter.stopped)
        self.assertEqual(stop_filter.finish(), "")

    def test_modern_text_parts_are_normalized(self) -> None:
        self.assertEqual(_text_content([
            {"type": "input_text", "text": "hello "},
            {"type": "text", "text": "world"},
        ], "input.0.content"), "hello world")
        with self.assertRaises(RequestError) as raised:
            _text_content([{"type": "input_image", "image_url": "x"}],
                          "input.0.content")
        self.assertEqual(raised.exception.code, "unsupported_value")

    def test_responses_request_uses_chat_template_and_greedy_contract(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, messages: object, **_kwargs: object) -> list[int]:
                self.messages = messages
                return [10, 11]

        app = Application.__new__(Application)
        app.args = types.SimpleNamespace(
            model="test-model", maximum_new_tokens=32, max_context=128,
        )
        app.tokenizer = Tokenizer()
        request = app.parse_request({
            "model": "test-model", "instructions": "be concise",
            "input": [{"role": "developer", "content": "policy"},
                      {"role": "user", "content": [
                          {"type": "input_text", "text": "hello"}
                      ]}],
            "max_output_tokens": 8, "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0, "top_p": 1,
        }, "responses")
        self.assertEqual(request.prompt_ids, [10, 11])
        self.assertEqual(request.maximum, 8)
        self.assertTrue(request.include_usage)
        self.assertEqual(app.tokenizer.messages[0],
                         {"role": "system", "content": "be concise"})
        with self.assertRaises(RequestError) as raised:
            app.parse_request({"model": "test-model", "input": "hello",
                               "temperature": 0.5}, "responses")
        self.assertEqual(raised.exception.param, "temperature")

        app.args.max_context = 9
        with self.assertRaises(RequestError) as raised:
            app.parse_request({
                "model": "test-model", "input": "hello",
                "max_output_tokens": 8, "temperature": 0,
            }, "responses")
        self.assertEqual(raised.exception.param, "max_output_tokens")
        self.assertIn("exceeds context capacity", str(raised.exception))

    def test_stream_disconnect_is_visible_before_next_decode(self) -> None:
        server_side, client_side = socket.socketpair()
        try:
            handler = Handler.__new__(Handler)
            handler.connection = server_side
            self.assertFalse(handler._client_disconnected())
            client_side.close()
            self.assertTrue(handler._client_disconnected())
        finally:
            server_side.close()
            client_side.close()

    def test_concurrent_rows_share_one_worker_step(self) -> None:
        worker = FakeWorker()
        observed: list[int] = []
        batcher = ContinuousDecodeBatcher(worker, 50.0, observed.append)
        barrier = threading.Barrier(5)
        results: dict[int, int] = {}

        def run(request_id: int) -> None:
            barrier.wait()
            results[request_id] = batcher.step(request_id, request_id == 4)

        threads = [threading.Thread(target=run, args=(request_id,))
                   for request_id in range(1, 5)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        batcher.close()

        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(len(worker.calls[0]), 4)
        self.assertEqual(observed, [4])
        self.assertEqual(results, {1: 10, 2: 20, 3: 30, 4: 41})

    def test_generation_stops_and_cancels_after_eos(self) -> None:
        class Worker:
            def __init__(self) -> None:
                self.active_ids: set[int] = set()

            def begin(self, request_id: int, _prompt: list[int]) -> None:
                self.active_ids.add(request_id)

            def cancel(self, request_id: int) -> None:
                self.active_ids.discard(request_id)

        class Batcher:
            def step(self, _request_id: int, _final: bool) -> int:
                return 7

        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return ",".join(str(token) for token in tokens)

        app = Application.__new__(Application)
        app.args = types.SimpleNamespace(generation_timeout=10.0)
        app.request_id = lambda: 1
        app.worker = Worker()
        app.decode_batcher = Batcher()
        app.tokenizer = Tokenizer()
        app.eos_token_ids = {7}
        app.increment = lambda *_args, **_kwargs: None
        app.observe_latency = lambda *_args, **_kwargs: None

        result = list(app.generate([3], 5))
        self.assertEqual(result, [(7, "7")])
        self.assertEqual(app.worker.active_ids, set())


if __name__ == "__main__":
    unittest.main()
