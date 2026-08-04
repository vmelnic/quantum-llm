from __future__ import annotations

import sys
import socket
import threading
import types
import unittest

# The batching coordinator itself has no tokenizer dependency. Keep this unit
# test runnable in the build Python used by CMake, where transformers is not a
# required package.
sys.modules.setdefault(
    "transformers", types.SimpleNamespace(AutoTokenizer=object)
)

from ops.python.expert_server import Application, ContinuousDecodeBatcher, Handler


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
