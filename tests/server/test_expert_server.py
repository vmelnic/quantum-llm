from __future__ import annotations

import sys
import threading
import types
import unittest

# The batching coordinator itself has no tokenizer dependency. Keep this unit
# test runnable in the build Python used by CMake, where transformers is not a
# required package.
sys.modules.setdefault(
    "transformers", types.SimpleNamespace(AutoTokenizer=object)
)

from ops.python.expert_server import ContinuousDecodeBatcher


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


if __name__ == "__main__":
    unittest.main()
