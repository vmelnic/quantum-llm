from __future__ import annotations

import contextlib
from collections import OrderedDict
import io
import sys
import socket
import tempfile
import time
import unittest.mock
import threading
import types
import unittest
import http.client
import json
from pathlib import Path
from http.server import ThreadingHTTPServer

# The batching coordinator itself has no tokenizer dependency. Keep this unit
# test runnable in the build Python used by CMake, where transformers is not a
# required package.
sys.modules.setdefault(
    "transformers", types.SimpleNamespace(AutoTokenizer=object)
)

import ops.python.expert_server as expert_server
from ops.python.chat_client import _effective_maximum
from ops.python.artifact_chat_codec import ArtifactChatCodec
from ops.python.expert_server import (
    Application, AssistantStreamParser, ContinuousDecodeBatcher, CudaWorker,
    Handler, IncrementalTextDecoder, IncrementalTokenDecoder, RequestError,
    SamplingSettings,
    StopFilter, _configured_eos_token_ids, _text_content,
    _worker_response_tokens,
)
from ops.python.response_protocols import (
    install_declared_response_protocol, select_response_protocol,
)


def application_fixture() -> Application:
    app = Application.__new__(Application)
    app.artifact_chat_codec = None
    app.response_protocol = None
    app.default_sampling = SamplingSettings(1.0, 0.95, 20, 0.0, 0)
    app.sampling_profiles = {
        "thinking": app.default_sampling,
        "non_thinking": app.default_sampling,
    }
    return app


class FakeWorker:
    capacity = 4

    def __init__(self) -> None:
        self.calls: list[list[tuple[int, int]]] = []
        self.lock = threading.Lock()

    def step(self, items: list[tuple[int, int]]) -> dict[int, int]:
        with self.lock:
            self.calls.append(list(items))
        return {request_id: request_id * 10 + int(flag)
                for request_id, flag in items}


class ContinuousDecodeBatcherTests(unittest.TestCase):
    def test_persistent_session_index_rejects_corruption_and_evicts_lru(self) -> None:
        """Disk snapshots are lazy, checksum-gated, and bounded independently of RAM."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = Application.__new__(Application)
            app.args = types.SimpleNamespace(
                max_context=64, session_cache_ttl_seconds=0.0,
                session_cache_bytes=600,
            )
            app.worker = types.SimpleNamespace(vocab_size=32, kv_page_tokens=4)
            app.session_cache_root = root
            app.session_cache_identity = "fixture-identity"
            app.session_cache_enabled = True
            app.session_lock = threading.Lock()
            app.disk_sessions = OrderedDict()
            app.sessions = OrderedDict()

            def write_snapshot(identifier: str, bytes_on_disk: int) -> Path:
                directory = root / identifier
                directory.mkdir()
                payload = {
                    "schema": "persistent-session-v1",
                    "identity": app.session_cache_identity,
                    "persistent_id": identifier,
                    "generation": 1,
                    "tokens": [1, 2, 3],
                    "pages": 1,
                    "parked_bytes": 12,
                    "last_used_epoch": 1.0,
                    "worker_metadata": {"generation": 1, "next_position": 3},
                    "media_signature": None,
                }
                payload["metadata_sha256"] = app._metadata_digest(payload)
                (directory / "session.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                (directory / "state.bin").write_bytes(b"x" * bytes_on_disk)
                return directory

            oldest = write_snapshot("a" * 32, 80)
            newest = write_snapshot("b" * 32, 80)
            corrupt = root / ("c" * 32)
            corrupt.mkdir()
            (corrupt / "session.json").write_text("not-json", encoding="utf-8")

            app._load_persistent_sessions()
            self.assertEqual(list(app.disk_sessions), ["a" * 32, "b" * 32])
            self.assertFalse(corrupt.exists())

            app._cleanup_persistent_sessions()
            self.assertNotIn("a" * 32, app.disk_sessions)
            self.assertIn("b" * 32, app.disk_sessions)
            self.assertFalse(oldest.exists())
            self.assertTrue(newest.exists())

    def test_request_telemetry_preserves_provider_configuration(self) -> None:
        before = {
            "provider_gpu_ffn_ns": 100,
            "provider_workspace_rows": 1024,
            "provider_compact_flash_prefill": 1,
            "allocated_pages": 12,
        }
        after = {
            "provider_gpu_ffn_ns": 140,
            "provider_workspace_rows": 1024,
            "provider_compact_flash_prefill": 1,
            "allocated_pages": 20,
        }
        self.assertEqual(
            Application._request_telemetry_fields(before, after),
            {
                "provider_gpu_ffn_ns": 40,
                "provider_workspace_rows": 1024,
                "provider_compact_flash_prefill": 1,
            },
        )

    def test_chat_output_limit_is_capped_by_active_service(self) -> None:
        with unittest.mock.patch(
                "ops.python.chat_client._get_json",
                return_value={"runtime_config": {"maximum_new_tokens": 127}},
        ):
            self.assertEqual(_effective_maximum("http://fixture", 1024), 127)
            self.assertEqual(_effective_maximum("http://fixture", 64), 64)

    def test_generation_config_declares_complete_eos_set(self) -> None:
        self.assertEqual(
            _configured_eos_token_ids(
                {"eos_token_id": [200001, 200008]}, 200001
            ),
            {200001, 200008},
        )
        self.assertEqual(_configured_eos_token_ids({}, 7), {7})
        self.assertEqual(
            _configured_eos_token_ids({"eos_token_id": None}, 7), {7}
        )
        self.assertEqual(_configured_eos_token_ids({}, (7, 9)), {7, 9})
        with self.assertRaisesRegex(RuntimeError, "eos_token_id is invalid"):
            _configured_eos_token_ids({"eos_token_id": [7, True]}, 7)

    def test_artifact_codec_controls_thinking_tools_and_stable_prefix(self) -> None:
        observed: dict[str, object] = {}

        def encode_messages(messages: list[dict[str, object]],
                            thinking_mode: str, drop_thinking: bool,
                            reasoning_effort: str | None = None) -> str:
            observed.update(messages=messages, thinking_mode=thinking_mode,
                            drop_thinking=drop_thinking,
                            reasoning_effort=reasoning_effort)
            return "BOS:user<A><think>"

        module = types.SimpleNamespace(
            encode_messages=encode_messages,
            parse_message_from_completion_text=lambda _text, _mode: {},
            eos_token="<eos>", thinking_start_token="<think>",
            thinking_end_token="</think>", ASSISTANT_SP_TOKEN="<A>",
            dsml_token="DSML", tool_calls_block_name="tool_calls",
        )
        codec = ArtifactChatCodec(module, Path("fixture.py"))
        prompt = codec.encode(
            [{"role": "user", "content": "weather"}],
            add_generation_prompt=False,
            tools=({"type": "function", "function": {
                "name": "weather", "parameters": {"type": "object"},
            }},),
            reasoning_effort="xhigh", enable_thinking=True,
            preserve_thinking=True,
        )
        self.assertEqual(prompt, "BOS:user")
        self.assertEqual(observed["thinking_mode"], "thinking")
        self.assertEqual(observed["reasoning_effort"], "max")
        self.assertFalse(observed["drop_thinking"])
        encoded_messages = observed["messages"]
        self.assertEqual(encoded_messages[0]["role"], "system")
        self.assertEqual(
            encoded_messages[0]["tools"][0]["function"]["name"],
            "weather",
        )

    def test_artifact_codec_streams_text_and_parses_dsml_without_leak(self) -> None:
        seen: dict[str, object] = {}

        def parse_message(text: str, thinking_mode: str) -> dict[str, object]:
            seen.update(text=text, thinking_mode=thinking_mode)
            return {
                "role": "assistant", "content": "answer",
                "reasoning_content": "reason",
                "tool_calls": [{"type": "function", "function": {
                    "name": "lookup", "arguments": '{"q":"value"}',
                }}],
            }

        module = types.SimpleNamespace(
            encode_messages=lambda _messages, thinking_mode: thinking_mode,
            parse_message_from_completion_text=parse_message,
            eos_token="<eos>", thinking_start_token="<think>",
            thinking_end_token="</think>", ASSISTANT_SP_TOKEN="<A>",
            dsml_token="DSML", tool_calls_block_name="tool_calls",
        )
        codec = ArtifactChatCodec(module, Path("fixture.py"))
        parser = codec.stream_parser(True)
        events = []
        events += parser.feed("rea")
        events += parser.feed("son</thi")
        events += parser.feed("nk>answer\n\n<DSMLtool_")
        events += parser.feed("calls>private")
        _message, tail = parser.finalize()
        events += tail
        self.assertEqual(
            [(event["field"], event["text"]) for event in events],
            [("reasoning_content", "rea"),
             ("reasoning_content", "son"),
             ("content", "answer")],
        )

        app = application_fixture()
        app.artifact_chat_codec = codec
        app.response_protocol = "artifact-chat-codec-v1"
        request = expert_server.GenerationRequest(
            endpoint="chat", prompt_ids=[1], cache_prefix_tokens=1,
            maximum=8, stream=False, stop=(), include_usage=False,
            enable_thinking=True,
        )
        output = app.parse_assistant_output("raw", request)
        self.assertEqual(output.text, "answer")
        self.assertEqual(output.reasoning, "reason")
        self.assertEqual(output.tool_calls[0].name, "lookup")
        self.assertEqual(json.loads(output.tool_calls[0].arguments),
                         {"q": "value"})
        self.assertEqual(seen, {
            "text": "raw<eos>", "thinking_mode": "thinking",
        })

    def test_artifact_codec_stream_hides_split_eos_terminal(self) -> None:
        module = types.SimpleNamespace(
            encode_messages=lambda _messages, thinking_mode: thinking_mode,
            parse_message_from_completion_text=lambda _text, thinking_mode: {
                "role": "assistant", "content": "answer",
            },
            eos_token="<eos>", thinking_start_token="<think>",
            thinking_end_token="</think>", ASSISTANT_SP_TOKEN="<A>",
            dsml_token="DSML", tool_calls_block_name="tool_calls",
        )
        parser = ArtifactChatCodec(module, Path("fixture.py")).stream_parser(
            False
        )
        events = parser.feed("answer<eo") + parser.feed("s>ignored")
        _message, tail = parser.finalize()
        events += tail
        self.assertEqual(
            [(event["field"], event["text"]) for event in events],
            [("content", "answer")],
        )

    def test_incremental_decoder_holds_incomplete_unicode_without_replay(self) -> None:
        decoder = IncrementalTextDecoder()
        self.assertEqual(decoder.push("Hello! \ufffd"), "Hello! ")
        self.assertEqual(decoder.push("Hello! 😊"), "😊")
        self.assertEqual(decoder.push("Hello! 😊 Next"), " Next")
        self.assertEqual(decoder.emitted, "Hello! 😊 Next")

    def test_incremental_decoder_flushes_replacement_only_at_final_boundary(self) -> None:
        decoder = IncrementalTextDecoder()
        self.assertEqual(decoder.push("value \ufffd"), "value ")
        self.assertEqual(decoder.push("value \ufffd", final=True), "\ufffd")

    def test_incremental_decoder_refuses_to_duplicate_changed_prefix(self) -> None:
        decoder = IncrementalTextDecoder()
        self.assertEqual(decoder.push("stable"), "stable")
        with self.assertRaises(expert_server.WorkerError):
            decoder.push("different")

    def test_incremental_token_decoder_is_bounded_and_context_exact(self) -> None:
        class Tokenizer:
            maximum_seen = 0

            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                self.maximum_seen = max(self.maximum_seen, len(tokens))
                pieces = {1: "Hello", 2: " world", 3: "!"}
                return "".join(pieces.get(token, chr(96 + token))
                               for token in tokens)

        tokenizer = Tokenizer()
        decoder = IncrementalTokenDecoder(
            tokenizer, overlap_tokens=4, maximum_window_tokens=8
        )
        output = "".join(decoder.push(token) for token in (
            1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6,
        ))
        expected = "Hello world!defghidef"
        self.assertEqual(output, expected)
        self.assertLessEqual(tokenizer.maximum_seen, 8)

    def test_incremental_token_decoder_holds_split_unicode(self) -> None:
        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                if tokens == [1]:
                    return "\ufffd"
                if tokens == [1, 2]:
                    return "😊"
                return ""

        decoder = IncrementalTokenDecoder(
            Tokenizer(), overlap_tokens=2, maximum_window_tokens=4
        )
        self.assertEqual(decoder.push(1), "")
        self.assertEqual(decoder.push(2), "😊")

    def test_incremental_token_decoder_preserves_declared_protocol_tokens(
            self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.skip_special_tokens: list[bool] = []

            def decode(self, tokens: list[int], **kwargs: object) -> str:
                skip = bool(kwargs["skip_special_tokens"])
                self.skip_special_tokens.append(skip)
                pieces = {
                    1: "to=self", 2: "<|message|>", 3: "reason",
                    4: "<eos>",
                }
                return "".join(
                    piece for token in tokens
                    if not (skip and token == 2)
                    for piece in (pieces[token],)
                )

        raw_tokenizer = Tokenizer()
        raw = IncrementalTokenDecoder(
            raw_tokenizer, overlap_tokens=4, maximum_window_tokens=8
        )
        self.assertEqual("".join(raw.push(token) for token in (1, 2, 3)),
                         "to=selfreason")
        self.assertTrue(all(raw_tokenizer.skip_special_tokens))

        protocol_tokenizer = Tokenizer()
        protocol = IncrementalTokenDecoder(
            protocol_tokenizer, overlap_tokens=4, maximum_window_tokens=8,
            preserve_special_tokens=True,
            suppressed_token_ids={4},
        )
        self.assertEqual(
            "".join(protocol.push(token) for token in (1, 2, 4, 3)),
            "to=self<|message|>reason",
        )
        self.assertFalse(any(protocol_tokenizer.skip_special_tokens))

    def test_incremental_token_decoder_never_exposes_suppressed_eos(self) -> None:
        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return "".join({1: "answer", 2: "<eos>"}[token]
                               for token in tokens)

        decoder = IncrementalTokenDecoder(
            Tokenizer(), overlap_tokens=4, maximum_window_tokens=8,
            suppressed_token_ids={2},
        )
        self.assertEqual(decoder.push(1), "answer")
        self.assertEqual(decoder.push(2, final=True), "")

    def test_incremental_token_decoder_strips_retained_context_prefix(self) -> None:
        class Tokenizer:
            maximum_seen = 0

            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                self.maximum_seen = max(self.maximum_seen, len(tokens))
                marker = "" if tokens and tokens[0] == 1 else "^"
                return marker + "".join(chr(96 + token) for token in tokens)

        tokenizer = Tokenizer()
        decoder = IncrementalTokenDecoder(
            tokenizer, overlap_tokens=4, maximum_window_tokens=8
        )
        tokens = list(range(1, 21))
        output = "".join(decoder.push(token) for token in tokens)
        self.assertEqual(output, "".join(chr(96 + token) for token in tokens))
        self.assertLessEqual(tokenizer.maximum_seen, 8)

    def test_standard_stream_events_split_reasoning_and_content(self) -> None:
        class Parser:
            def feed(self, delta: str) -> list[dict[str, object]]:
                field, text = delta.split(":", 1)
                return [{"type": "region_chunk", "field": field,
                         "text": text, "dirty": False}]

            def finalize(self) -> tuple[dict[str, object], list[dict[str, object]]]:
                return {}, []

        app = types.SimpleNamespace(
            response_stream_parser=lambda _request: Parser(),
            assistant_output_from_parsed_message=lambda message, _text,
                fallback_text: expert_server.Application.
                _assistant_output_from_message(message),
        )
        parser = AssistantStreamParser(app, object())
        self.assertEqual(parser.feed("thinking:check assumptions"),
                         ("check assumptions", ""))
        self.assertEqual(parser.feed("content:final answer"),
                         ("", "final answer"))
        reasoning, content, output = parser.finish()
        self.assertEqual((reasoning, content), ("", ""))
        self.assertEqual(output.tool_calls, ())

    def test_worker_tokens_accept_legacy_scalar_and_mtp_list(self) -> None:
        self.assertEqual(
            _worker_response_tokens({"token": 7}, "missing"), [7]
        )
        self.assertEqual(
            _worker_response_tokens({"tokens": [7, 8]}, "missing"), [7, 8]
        )
        with self.assertRaises(expert_server.WorkerError):
            _worker_response_tokens({"tokens": []}, "missing")

    def test_protocol8_serializes_request_sampling(self) -> None:
        worker = CudaWorker.__new__(CudaWorker)
        worker.sampling_supported = True
        worker.sampling_presence_penalty_supported = True
        self.assertEqual(
            worker._sampling_command(SamplingSettings(
                1.0, 0.95, 20, 0.0, 1234
            )),
            "\tSAMPLING\t1000000\t950000\t20\t0\t0\t1234",
        )

    def test_protocol9_end_retain_returns_exact_parking_accounting(self) -> None:
        worker = CudaWorker.__new__(CudaWorker)
        worker.session_parking = True
        worker.kv_page_bytes = 4096
        worker.session_park_ram_bytes = 1 << 20
        worker.session_park_page_capacity = 64
        worker.active_ids = {7}
        worker._command = lambda _command: {
            "type": "ended", "id": 7, "retained_tokens": 300,
            "parked_pages": 2, "parked_bytes": 16384,
        }

        self.assertEqual(worker.end_retain(7, 9, 300), (300, 2, 16384))
        self.assertNotIn(7, worker.active_ids)

    def test_protocol9_resume_accepts_an_exact_same_prefix(self) -> None:
        worker = CudaWorker.__new__(CudaWorker)
        worker.sampling_supported = True
        worker.sampling_presence_penalty_supported = True
        worker.active_ids = set()
        commands: list[str] = []

        def command(value: str, **_kwargs: object) -> dict[str, object]:
            commands.append(value)
            return {"type": "begun", "id": 7}

        worker._command = command
        worker.begin_resume(
            7, 9, [], 1024, SamplingSettings(0.0, 1.0, 0, 0.0, 1)
        )

        self.assertEqual(
            commands,
            ["BEGIN\t7\t1024\t\tRESUME\t9\tSAMPLING\t0\t1000000\t0\t0\t0\t1"],
        )
        self.assertIn(7, worker.active_ids)

    def test_protocol9_optional_page_state_may_use_less_than_page_ceiling(
            self) -> None:
        worker = CudaWorker.__new__(CudaWorker)
        worker.session_parking = True
        worker.kv_page_bytes = 4096
        worker.session_park_ram_bytes = 1 << 20
        worker.session_park_page_capacity = 64
        worker.active_ids = {7}
        worker._command = lambda _command: {
            "type": "ended", "id": 7, "retained_tokens": 300,
            "parked_pages": 2, "parked_bytes": 4096,
        }

        self.assertEqual(worker.end_retain(7, 9, 300), (300, 2, 4096))

    def test_protocol9_invalid_end_accounting_drops_worker_session(self) -> None:
        worker = CudaWorker.__new__(CudaWorker)
        worker.session_parking = True
        worker.kv_page_bytes = 4096
        worker.session_park_ram_bytes = 1 << 20
        worker.session_park_page_capacity = 64
        worker.active_ids = {7}
        commands: list[str] = []

        def command(value: str) -> dict[str, object]:
            commands.append(value)
            if value.startswith("END"):
                return {
                    "type": "ended", "id": 7, "retained_tokens": 300,
                    "parked_pages": 2, "parked_bytes": 0,
                }
            return {"type": "dropped"}

        worker._command = command
        with self.assertRaises(expert_server.WorkerError):
            worker.end_retain(7, 9, 300)

        self.assertEqual(commands, ["END\t7\tRETAIN\t9\tAT\t300", "DROP\t9"])
        self.assertNotIn(7, worker.active_ids)

    def test_cached_worker_stats_do_not_query_a_busy_worker(self) -> None:
        class Worker:
            def stats(self) -> dict[str, int]:
                raise AssertionError("cached stats must not query the worker")

        app = application_fixture()
        app.worker = Worker()
        app._worker_stats_lock = threading.Lock()
        app._worker_stats_cache = {"allocated_pages": 7}
        app._worker_stats_updated_at = time.monotonic()

        self.assertEqual(
            app.worker_stats(refresh=False), {"allocated_pages": 7}
        )
        self.assertLess(app.worker_stats_age_seconds(), 1.0)

    def test_parking_capacity_evicts_lru_and_retries_same_active_request(
            self) -> None:
        class Worker:
            def __init__(self) -> None:
                self.calls = 0

            def end_retain(self, _request_id: int, _session_key: int,
                           _checkpoint: int) -> tuple[int, int, int]:
                self.calls += 1
                if self.calls == 1:
                    raise expert_server.WorkerError(
                        "request-state RAM parking capacity is exhausted"
                    )
                return 300, 2, 16384

        app = application_fixture()
        app.worker = Worker()
        evictions: list[bool] = [True]
        app.evict_lru_session = lambda: evictions.pop() if evictions else False

        self.assertEqual(
            app.end_retain_with_eviction(7, 9, 300), (300, 2, 16384)
        )
        self.assertEqual(app.worker.calls, 2)

    def test_protocol4_worker_infers_legacy_prefetch_state(self) -> None:
        ready = {
            "type": "ready", "protocol": 4, "capacity": 4,
            "prefill_mode": "causal_chunked", "prefill_chunk_tokens": 4,
            "kv_dtype": "fp16", "kv_allocation": "paged_on_demand",
            "kv_page_tokens": 256, "kv_page_bytes": 6 << 20,
            "kv_page_capacity": 341, "placement_profile": "balanced",
            "ram_cache_bytes": 48 << 30, "vram_cache_bytes": 18 << 30,
            "placement_prefetch_enabled": True,
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process
        ) as popen:
            worker = CudaWorker(
                expert_server.Path("worker.exe"), expert_server.Path("pack"),
                65536, 1, 4, 48, 18, 2048, 256, "balanced", False,
            )
        self.assertTrue(worker.placement_prefetch_enabled)
        self.assertEqual(worker.placement_prefetch_state, "ready")

    def test_budgeted_worker_reports_effective_cache_below_ceiling(self) -> None:
        ready = {
            "type": "ready", "protocol": 4, "capacity": 1,
            "prefill_mode": "causal_sequential", "prefill_chunk_tokens": 1,
            "kv_dtype": "bf16", "kv_allocation": "preallocated",
            "kv_page_tokens": 256, "kv_page_bytes": 1024,
            "kv_page_capacity": 65536, "placement_mode": "budgeted",
            "placement_profile": "balanced",
            "ram_cache_bytes": 40 << 30, "vram_cache_bytes": 11 << 30,
            "placement_prefetch_enabled": False,
            "placement_prefetch_state": "observing",
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process):
            worker = CudaWorker(
                expert_server.Path("worker.exe"), expert_server.Path("pack"),
                65536, 1, 1, 48, 12, 2048, 256, "balanced", False,
            )
        self.assertEqual(worker.ram_cache_bytes, 40 << 30)
        self.assertEqual(worker.vram_cache_bytes, 11 << 30)

    def test_worker_accepts_exact_blocked_causal_prefill(self) -> None:
        ready = {
            "type": "ready", "protocol": 9, "capacity": 1,
            "architecture_id": "fixture.vendor.sparse",
            "vocab_size": 64000, "max_context_tokens": 262144,
            "routed_layers": 43, "experts_per_layer": 256,
            "route_width": 6, "expert_encoding": "fp4.vendor.group32",
            "operation_capabilities": ["attention.vendor.v2"],
            "prefill_mode": "causal_blocked_exact",
            "prefill_chunk_tokens": 256,
            "session_retention": False, "session_parking": False,
            "session_park_ram_bytes": 0, "session_park_page_capacity": 0,
            "kv_dtype": "bf16", "kv_allocation": "preallocated",
            "kv_page_tokens": 256, "kv_page_bytes": 1024,
            "kv_page_capacity": 65536, "placement_mode": "budgeted",
            "placement_profile": "balanced",
            "ram_cache_bytes": 40 << 30, "vram_cache_bytes": 12 << 30,
            "placement_prefetch_enabled": False,
            "placement_prefetch_state": "disabled",
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process):
            worker = CudaWorker(
                expert_server.Path("provider.exe"), expert_server.Path("pack"),
                262144, 1, 1, 40, 12, 5120, 256, "balanced", False,
                prefill_chunk_tokens=512,
            )
        self.assertEqual(worker.prefill_mode, "causal_blocked_exact")
        self.assertEqual(worker.prefill_chunk_tokens, 256)

    def test_worker_accepts_provider_cpu_policy_by_default(self) -> None:
        ready = {
            "type": "ready", "protocol": 5, "capacity": 1,
            "prefill_mode": "causal_sequential", "prefill_chunk_tokens": 1,
            "kv_dtype": "bf16", "kv_allocation": "preallocated",
            "kv_page_tokens": 256, "kv_page_bytes": 1024,
            "kv_page_capacity": 65536, "placement_profile": "balanced",
            "ram_cache_bytes": 48 << 30, "vram_cache_bytes": 13 << 30,
            "placement_prefetch_enabled": True,
            "placement_prefetch_state": "ready",
            "placement_minimum_observations": 2,
            "cpu_hybrid_enabled": False,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process
        ) as popen:
            worker = CudaWorker(
                expert_server.Path("worker.exe"), expert_server.Path("pack"),
                65536, 1, 1, 48, 13, 2048, 256, "balanced", False,
                enable_cpu_hybrid=None,
                route_trace_file=expert_server.Path("trace.jsonl"),
                route_trace_max_steps=17,
            )
        self.assertFalse(worker.cpu_hybrid_enabled)
        self.assertNotIn("--cpu-hybrid", popen.call_args.args[0])
        self.assertIn("--route-trace-file=trace.jsonl", popen.call_args.args[0])
        self.assertIn("--route-trace-max-steps=17", popen.call_args.args[0])

    def test_protocol6_launch_and_descriptor_are_provider_neutral(self) -> None:
        ready = {
            "type": "ready", "protocol": 6, "capacity": 2,
            "architecture_id": "fixture.vendor.sparse",
            "vocab_size": 64000, "max_context_tokens": 131072,
            "routed_layers": 57, "experts_per_layer": 1024,
            "route_width": 8, "expert_encoding": "fp4.vendor.group64",
            "operation_capabilities": [
                "attention.vendor.v2", "moe.vendor.fp4.v3",
            ],
            "prefill_mode": "causal_chunked", "prefill_chunk_tokens": 128,
            "kv_dtype": "bf16", "kv_allocation": "paged_on_demand",
            "kv_page_tokens": 128, "kv_page_bytes": 4096,
            "kv_page_capacity": 4096, "placement_profile": "balanced",
            "ram_cache_bytes": 96 << 30, "vram_cache_bytes": 18 << 30,
            "placement_prefetch_enabled": True,
            "placement_prefetch_state": "ready",
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process
        ) as popen:
            worker = CudaWorker(
                expert_server.Path("provider.exe"), expert_server.Path("pack"),
                65536, 1, 2, 96, 18, 2048, 128, "balanced", False,
                prefill_chunk_tokens=256,
            )
        command = popen.call_args.args[0]
        self.assertIn("--max-context=65536", command)
        self.assertIn("--prefill-chunk-limit=256", command)
        self.assertNotIn("65536", command)
        self.assertEqual(worker.architecture_id, "fixture.vendor.sparse")
        self.assertEqual(worker.operation_capabilities,
                         ("attention.vendor.v2", "moe.vendor.fp4.v3"))

    def test_protocol7_accepts_dense_only_artifact_descriptor(self) -> None:
        ready = {
            "type": "ready", "protocol": 7, "capacity": 1,
            "architecture_id": "fixture.dense", "vocab_size": 64000,
            "max_context_tokens": 262144,
            "routed_layers": 0, "experts_per_layer": 0,
            "route_width": 0, "expert_encoding": "",
            "operation_capabilities": [
                "block.recurrent-linear-attention.v1",
                "ffn.swiglu.dense.fp4-block32.v1",
            ],
            "prefill_mode": "causal_chunked", "prefill_chunk_tokens": 128,
            "kv_dtype": "fp16", "kv_allocation": "paged_on_demand",
            "kv_page_tokens": 128, "kv_page_bytes": 4096,
            "kv_page_capacity": 4096, "placement_mode": "budgeted",
            "placement_profile": "balanced",
            "ram_cache_bytes": 48 << 30, "vram_cache_bytes": 18 << 30,
            "placement_prefetch_enabled": False,
            "placement_prefetch_state": "disabled",
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process
        ) as popen:
            worker = CudaWorker(
                expert_server.Path("provider.exe"), expert_server.Path("pack"),
                262144, 1, 1, 48, 18, 2048, 128, "balanced", False,
                kv_cache_dtype="fp16",
            )
        self.assertIn("--kv-cache-dtype=fp16", popen.call_args.args[0])
        self.assertEqual(worker.kv_dtype, "fp16")
        self.assertEqual(worker.routed_layers, 0)
        self.assertEqual(worker.experts_per_layer, 0)
        self.assertEqual(worker.route_width, 0)
        self.assertEqual(worker.expert_encoding, "")

    def test_protocol10_accepts_artifact_declared_bf16_latent_kv(self) -> None:
        ready = {
            "type": "ready", "protocol": 10, "capacity": 1,
            "architecture_id": "fixture.mla", "vocab_size": 64000,
            "max_context_tokens": 262144,
            "routed_layers": 4, "experts_per_layer": 128,
            "route_width": 4, "expert_encoding": "nvfp4.fixture",
            "operation_capabilities": ["block.mla.bfloat16.v1"],
            "prefill_mode": "causal_layer_major",
            "prefill_chunk_tokens": 128,
            "session_retention": True, "session_parking": False,
            "session_park_ram_bytes": 0, "session_park_page_capacity": 0,
            "sampling_supported": True,
            "sampling_presence_penalty_supported": True,
            "kv_dtype": "bf16-latent",
            "kv_allocation": "paged_on_demand",
            "kv_page_tokens": 128, "kv_page_bytes": 4096,
            "kv_page_capacity": 2048, "placement_mode": "budgeted",
            "placement_profile": "balanced",
            "ram_cache_bytes": 48 << 30, "vram_cache_bytes": 12 << 30,
            "placement_prefetch_enabled": False,
            "placement_prefetch_state": "disabled",
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process):
            worker = CudaWorker(
                expert_server.Path("provider.exe"),
                expert_server.Path("pack"), 262144, 1, 1, 48, 12,
                2048, 128, "balanced", False,
            )
        self.assertEqual(worker.kv_dtype, "bf16-latent")

    def test_protocol11_accepts_fitted_routed_vram_above_fixed_ceiling(
            self) -> None:
        ready = {
            "type": "ready", "protocol": 11, "capacity": 1,
            "architecture_id": "fixture.routed", "vocab_size": 64000,
            "max_context_tokens": 262144,
            "routed_layers": 10, "experts_per_layer": 256,
            "route_width": 8, "expert_encoding": "fp4.fixture",
            "operation_capabilities": ["moe.fixture.v1"],
            "prefill_mode": "causal_layer_major",
            "prefill_chunk_tokens": 1024,
            "session_retention": True, "session_parking": False,
            "session_park_ram_bytes": 0, "session_park_page_capacity": 0,
            "sampling_supported": True,
            "sampling_presence_penalty_supported": True,
            "kv_dtype": "fp4-e2m1-ue8m0-block32-key-outlier1",
            "kv_allocation": "paged_on_demand",
            "kv_page_tokens": 256, "kv_page_bytes": 1556480,
            "kv_page_capacity": 1024, "placement_mode": "budgeted",
            "placement_profile": "balanced",
            "ram_cache_bytes": 18 << 30,
            "vram_cache_bytes": 16 << 30,
            "routed_vram_policy": "fit",
            "placement_prefetch_enabled": False,
            "placement_prefetch_state": "disabled",
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process
        ) as popen:
            worker = CudaWorker(
                expert_server.Path("provider.exe"),
                expert_server.Path("pack"), 262144, 1, 1, 48, 12,
                5136, 256, "balanced", False,
                kv_cache_dtype="fp4-e2m1-ue8m0-block32-key-outlier1",
                routed_vram_policy="fit",
            )
        self.assertEqual(worker.vram_cache_bytes, 16 << 30)
        self.assertEqual(worker.routed_vram_policy, "fit")
        self.assertIn("--routed-vram-policy=fit", popen.call_args.args[0])

    def test_protocol7_rejects_partial_routed_geometry(self) -> None:
        ready = {
            "type": "ready", "protocol": 7, "capacity": 1,
            "architecture_id": "fixture.invalid", "vocab_size": 64000,
            "max_context_tokens": 262144,
            "routed_layers": 4, "experts_per_layer": 0,
            "route_width": 0, "expert_encoding": "",
            "operation_capabilities": ["ffn.swiglu.dense.v1"],
            "prefill_mode": "causal_chunked", "prefill_chunk_tokens": 128,
            "kv_dtype": "fp16", "kv_allocation": "paged_on_demand",
            "kv_page_tokens": 128, "kv_page_bytes": 4096,
            "kv_page_capacity": 4096, "placement_mode": "budgeted",
            "placement_profile": "balanced",
            "ram_cache_bytes": 48 << 30, "vram_cache_bytes": 18 << 30,
            "placement_prefetch_enabled": False,
            "placement_prefetch_state": "disabled",
            "placement_minimum_observations": 2,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(
                    expert_server.WorkerError, "runtime contract"):
                CudaWorker(
                    expert_server.Path("provider.exe"),
                    expert_server.Path("pack"),
                    262144, 1, 1, 48, 18, 2048, 128, "balanced", False,
                )

    def test_protocol6_accepts_honest_resident_fp32_provider(self) -> None:
        ready = {
            "type": "ready", "protocol": 6, "capacity": 1,
            "architecture_id": "fixture.vm", "vocab_size": 32000,
            "max_context_tokens": 8192, "routed_layers": 8,
            "experts_per_layer": 32, "route_width": 4,
            "expert_encoding": "int8.row",
            "operation_capabilities": ["moe.swiglu.routed.v1"],
            "prefill_mode": "causal_sequential", "prefill_chunk_tokens": 1,
            "kv_dtype": "fp32", "kv_allocation": "preallocated",
            "kv_page_tokens": 256, "kv_page_bytes": 4096,
            "kv_page_capacity": 32, "placement_mode": "resident",
            "placement_profile": "resident", "ram_cache_bytes": 0,
            "vram_cache_bytes": 8 << 30,
            "placement_prefetch_enabled": False,
            "placement_prefetch_state": "disabled",
            "placement_minimum_observations": 0,
            "retain_previous_route": False,
            "cpu_hybrid_enabled": False,
        }
        process = unittest.mock.MagicMock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO(json.dumps(ready) + "\n")
        process.stderr = io.StringIO()
        with unittest.mock.patch.object(
                expert_server.subprocess, "Popen", return_value=process
        ) as popen:
            worker = CudaWorker(
                expert_server.Path("provider.exe"), expert_server.Path("pack"),
                4096, 1, 1, 48, 18, 2048, 256, "balanced", False,
                enable_cpu_hybrid=False,
            )
        self.assertNotIn("--no-retain-previous-route",
                         popen.call_args.args[0])
        self.assertFalse(worker.retain_previous_route)
        self.assertEqual(worker.placement_mode, "resident")
        self.assertEqual(worker.kv_dtype, "fp32")

    def test_service_log_does_not_duplicate_to_unconsumed_stderr(self) -> None:
        previous = expert_server.LOG_FILE
        service_log = io.StringIO()
        stderr = io.StringIO()
        try:
            expert_server.LOG_FILE = service_log
            with contextlib.redirect_stderr(stderr):
                expert_server.log("test", value=1)
        finally:
            expert_server.LOG_FILE = previous
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn('"event":"test"', service_log.getvalue())

    def test_http_openai_response_and_stream_contracts(self) -> None:
        class Tokenizer:
            parse_calls = 0

            def apply_chat_template(self, _messages: object, **_kwargs: object) -> list[int]:
                return [10, 11]

            def parse_response(self, text: str,
                               **_kwargs: object) -> dict[str, object]:
                self.parse_calls += 1
                if "<tool_call>" not in text:
                    return {"role": "assistant", "content": text.strip()}
                content = text.split("<tool_call>", 1)[0].strip()
                return {"role": "assistant", "content": content,
                        "tool_calls": [{
                    "type": "function", "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Chisinau"},
                    },
                }]}

            def get_response_parser(self, **_kwargs: object) -> object:
                tokenizer = self

                class Parser:
                    def __init__(self) -> None:
                        self.raw: list[str] = []

                    def feed(self, delta: str) -> list[dict[str, object]]:
                        self.raw.append(delta)
                        if "<tool_call>" in delta:
                            return []
                        return [{"type": "region_chunk", "field": "content",
                                 "text": delta, "dirty": False}]

                    def finalize(self) -> tuple[dict[str, object],
                                                list[dict[str, object]]]:
                        message = tokenizer.parse_response("".join(self.raw))
                        if message.get("tool_calls") and message.get("content"):
                            message["content"] = "different final rendering"
                        return message, []

                return Parser()

        app = application_fixture()
        app.args = types.SimpleNamespace(
            model="test-model", build_id="build", maximum_new_tokens=32,
            max_context=128, maximum_body_bytes=1 << 20, api_key="",
        )
        app.tokenizer = Tokenizer()
        app.eos_token_ids = set()
        app.worker = types.SimpleNamespace(
            healthy=lambda: True, kv_page_tokens=16, kv_page_capacity=32,
        )
        app.draining = threading.Event()
        app.acquire = lambda: True
        app.release = lambda: None
        app.acquire_worker_slot = lambda: True
        app.release_worker_slot = lambda: None
        app.acquire_request_context = lambda _ids, _maximum: types.SimpleNamespace(
            session=None, held_pages=0, retained=False
        )
        app.release_request_context = lambda _context: None
        app.increment = lambda *_args, **_kwargs: None

        app.generated = ("hello", " world")

        def generate(_prompt: list[int], maximum: int, _context: object = None):
            for index, value in enumerate(app.generated[:maximum]):
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

            tool = {"type": "function", "function": {
                "name": "get_weather", "description": "Read weather",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"},
                }, "required": ["city"]},
            }}
            app.response_protocol = "fixture"
            # Transformers marks the combined template separator + first
            # content byte as a clean stream chunk, but its final parser
            # removes the leading separator.
            app.generated = ("\n\nfirst ", "second")
            connection.request("POST", "/v1/chat/completions", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "explain"}],
                "tools": [tool], "tool_choice": "auto",
                "max_completion_tokens": 2, "stream": True,
                "stream_options": {"include_usage": True},
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            stream = response.read().decode()
            first = '"delta":{"content":"first"}'
            second = '"delta":{"content":" second"}'
            self.assertGreaterEqual(stream.find(first), 0)
            self.assertGreater(stream.find(second), stream.find(first))
            self.assertNotIn('"delta":{"content":"first second"}', stream)

            app.generated = ("hello", " world")
            connection.request("POST", "/v1/messages", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2, "temperature": 0,
            }), headers={"Content-Type": "application/json",
                         "anthropic-version": "2023-06-01"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["type"], "message")
            self.assertEqual(payload["role"], "assistant")
            self.assertEqual(payload["content"], [
                {"type": "text", "text": "hello world"}
            ])
            self.assertEqual(payload["usage"], {
                "input_tokens": 2, "output_tokens": 2,
            })

            connection.request("POST", "/v1/messages", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2, "temperature": 0, "stream": True,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            stream = response.read().decode()
            self.assertEqual(response.status, 200)
            self.assertIn("event: message_start\n", stream)
            self.assertIn('"type":"text_delta","text":"hello"', stream)
            self.assertIn("event: message_delta\n", stream)
            self.assertTrue(stream.endswith(
                'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            ))

            connection.request(
                "POST", "/v1/messages/count_tokens", body=json.dumps({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hi"}],
                }), headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertEqual(json.loads(response.read()), {"input_tokens": 2})

            tool_generation = (
                "<tool_call>\n<function=get_weather>\n"
                "<parameter=city>\nChisinau\n</parameter>\n"
                "</function>\n</tool_call>",
            )
            app.generated = tool_generation
            app.tokenizer.parse_calls = 0
            connection.request("POST", "/v1/chat/completions", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": [tool], "tool_choice": "auto",
                "max_completion_tokens": 2, "stream": True,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            stream = response.read().decode()
            self.assertIn('"delta":{"tool_calls":[', stream)
            self.assertNotIn("<tool_call>", stream)
            self.assertEqual(app.tokenizer.parse_calls, 1)

            app.generated = (
                "I will inspect the repository.\n\n",
                tool_generation[0],
            )
            connection.request("POST", "/v1/chat/completions", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": [tool], "tool_choice": "auto",
                "max_completion_tokens": 2, "stream": True,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            stream = response.read().decode()
            self.assertIn(
                '"delta":{"content":"I will inspect the repository."}',
                stream,
            )
            self.assertNotIn("repository.\\n", stream)
            self.assertNotIn("different final rendering", stream)
            self.assertIn('"delta":{"tool_calls":[', stream)
            self.assertEqual(app.tokenizer.parse_calls, 2)
            app.generated = tool_generation

            connection.request("POST", "/v1/chat/completions", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": [tool], "tool_choice": "auto",
                "max_completion_tokens": 2,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
            call = payload["choices"][0]["message"]["tool_calls"][0]
            self.assertEqual(call["function"]["name"], "get_weather")
            self.assertEqual(json.loads(call["function"]["arguments"]),
                             {"city": "Chisinau"})

            anthropic_tool = {
                "name": "get_weather", "description": "Read weather",
                "input_schema": {"type": "object", "properties": {
                    "city": {"type": "string"},
                }, "required": ["city"]},
            }
            connection.request("POST", "/v1/messages", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": [anthropic_tool], "tool_choice": {"type": "auto"},
                "max_tokens": 2,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(payload["stop_reason"], "tool_use")
            self.assertEqual(payload["content"][0]["type"], "tool_use")
            self.assertTrue(payload["content"][0]["id"].startswith("toolu_"))
            self.assertEqual(payload["content"][0]["input"],
                             {"city": "Chisinau"})
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_tokenizer_dependency_failure_is_structured_http_error(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, *_args: object,
                                    **_kwargs: object) -> list[int]:
                raise ImportError("jinja2 is missing")

        app = application_fixture()
        app.args = types.SimpleNamespace(
            model="test-model", maximum_new_tokens=32, max_context=128,
            maximum_body_bytes=1 << 20, api_key="",
        )
        app.tokenizer = Tokenizer()
        app.checkpoint_chat_encoder = None
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.app = app
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        try:
            connection.request("POST", "/v1/chat/completions", body=json.dumps({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 2,
            }), headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 500)
            self.assertEqual(
                payload["error"]["code"], "request_preprocessing_failed"
            )
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

    def test_user_image_parts_are_normalized_only_for_vision_artifacts(self) -> None:
        decoded = object()
        with unittest.mock.patch.object(
                expert_server, "load_image", return_value=decoded) as load:
            content = _text_content([
                {"type": "input_text", "text": "describe"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,AA==",
                }},
            ], "messages.0.content", allow_images=True, user_content=True)
        self.assertEqual(content, [
            {"type": "text", "text": "describe"},
            {"type": "image", "image": decoded},
        ])
        load.assert_called_once_with("data:image/png;base64,AA==")
        with self.assertRaises(RequestError) as raised:
            _text_content([
                {"type": "input_image", "image_url": "https://example/image.png"},
            ], "messages.0.content", allow_images=True, user_content=False)
        self.assertEqual(raised.exception.code, "unsupported_value")

    def test_anthropic_history_and_tools_normalize_to_common_chat(self) -> None:
        payload = Application.anthropic_payload({
            "model": "test-model", "max_tokens": 128,
            "system": [{"type": "text", "text": "Use tools."}],
            "messages": [
                {"role": "user", "content": "weather"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Need a lookup."},
                    {"type": "tool_use", "id": "toolu_1",
                     "name": "get_weather", "input": {"city": "Chisinau"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "18 C"}]},
                    {"type": "text", "text": "Summarize."},
                ]},
            ],
            "tools": [{
                "name": "get_weather", "description": "Read weather",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
        })
        self.assertEqual(payload["messages"][0],
                         {"role": "system", "content": "Use tools."})
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["id"],
                         "toolu_1")
        self.assertEqual(payload["messages"][3], {
            "role": "tool", "content": "18 C", "tool_call_id": "toolu_1",
        })
        self.assertEqual(payload["messages"][4],
                         {"role": "user", "content": "Summarize."})
        self.assertEqual(payload["tools"][0]["function"]["name"],
                         "get_weather")
        self.assertEqual(payload["reasoning_effort"], "xhigh")
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])

    def test_claude_cli_inline_system_message_joins_system_prompt(self) -> None:
        payload = Application.anthropic_payload({
            "model": "test-model", "max_tokens": 128,
            "system": [
                {"type": "text", "text": "Base instructions."},
                {"type": "text", "text": "Tool instructions.",
                 "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "text", "text": " there"},
                ]},
                {"role": "system", "content": [{
                    "type": "text", "text": "Runtime instructions.",
                    "cache_control": {"type": "ephemeral"},
                }]},
            ],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        })
        self.assertEqual(payload["messages"], [
            {"role": "system",
             "content": "Base instructions.Tool instructions."},
            {"role": "user", "content": "hi there"},
            {"role": "user", "content": "Runtime instructions."},
        ])
        self.assertEqual(payload["_cache_prefix_messages"], [
            {"role": "system",
             "content": "Base instructions.Tool instructions."},
            {"role": "user", "content": "hi there"},
        ])
        self.assertEqual(payload["reasoning_effort"], "xhigh")

    def test_anthropic_inline_system_sets_stable_kv_checkpoint(self) -> None:
        class Tokenizer:
            role_tokens = {"system": 10, "user": 20, "assistant": 30}

            def apply_chat_template(self, messages: object,
                                    add_generation_prompt: bool,
                                    **_kwargs: object) -> list[int]:
                tokens = [1]
                for message in messages:
                    tokens.extend((self.role_tokens[message["role"]],
                                   len(message["content"])))
                if add_generation_prompt:
                    tokens.append(99)
                return tokens

        app = application_fixture()
        app.args = types.SimpleNamespace(
            model="test-model", maximum_new_tokens=32, max_context=128,
        )
        app.tokenizer = Tokenizer()
        normalized = app.anthropic_payload({
            "model": "test-model", "max_tokens": 8,
            "system": [{"type": "text", "text": "Stable system."}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": [{
                    "type": "text", "text": "Dynamic reminder.",
                    "cache_control": {"type": "ephemeral"},
                }]},
            ],
        })

        request = app.parse_request(normalized, "anthropic")

        self.assertEqual(request.prompt_ids, [1, 10, 14, 20, 2, 20, 17, 99])
        self.assertEqual(request.cache_prefix_tokens, 5)
        self.assertEqual(
            request.prompt_ids[:request.cache_prefix_tokens],
            [1, 10, 14, 20, 2],
        )

    def test_anthropic_uses_last_inline_system_cache_boundary(self) -> None:
        payload = Application.anthropic_payload({
            "model": "test-model", "max_tokens": 8,
            "system": "Stable system.",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "system", "content": "historical reminder"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
                {"role": "system", "content": "current reminder"},
            ],
        })

        self.assertEqual(payload["_cache_prefix_messages"], [
            {"role": "system", "content": "Stable system."},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "historical reminder"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ])

    def test_anthropic_stream_heartbeats_while_generation_is_buffered(self) -> None:
        app = application_fixture()
        app.args = types.SimpleNamespace(model="test-model")
        handler = Handler.__new__(Handler)
        handler.server = types.SimpleNamespace(app=app)
        handler.anthropic_heartbeat_seconds = 0.0
        events: list[str] = []
        handler._sse_headers = lambda: None
        handler._anthropic_sse = lambda event, _payload: events.append(event)
        handler._client_disconnected = lambda: False

        def run_generation(_request: object, _emit: object, _context: object,
                           progress_callback: object = None
                           ) -> tuple[str, int, str]:
            self.assertIsNotNone(progress_callback)
            assert callable(progress_callback)
            progress_callback()
            return "hello", 1, "stop"

        handler._run_generation = run_generation
        request = expert_server.GenerationRequest(
            endpoint="anthropic", prompt_ids=[10, 11],
            cache_prefix_tokens=2, maximum=8, stream=True, stop=(),
            include_usage=False,
        )
        handler._serve_anthropic(request, "msg_test", object())

        self.assertIn("ping", events)
        self.assertLess(events.index("message_start"), events.index("ping"))
        self.assertLess(events.index("ping"), events.index("message_stop"))

    def test_anthropic_stream_holds_tool_grammar_leading_whitespace(self) -> None:
        class Parser:
            def feed(self, delta: str) -> list[dict[str, object]]:
                return [{"type": "region_chunk", "field": "content",
                         "text": delta, "dirty": False}]

            def finalize(self) -> tuple[dict[str, object],
                                        list[dict[str, object]]]:
                return {
                    "role": "assistant", "content": "",
                    "tool_calls": [{"type": "function", "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Chisinau"},
                    }}],
                }, []

        call = expert_server.ToolCall(
            item_id="item_1", call_id="call_1", name="get_weather",
            arguments='{"city":"Chisinau"}',
        )
        app = application_fixture()
        app.args = types.SimpleNamespace(model="test-model")
        app.response_stream_parser = lambda _request: Parser()
        app.parse_assistant_output = lambda _text, _request: \
            (_ for _ in ()).throw(AssertionError("stream response was reparsed"))
        handler = Handler.__new__(Handler)
        handler.server = types.SimpleNamespace(app=app)
        events: list[tuple[str, dict[str, object]]] = []
        handler._sse_headers = lambda: None
        handler._anthropic_sse = lambda event, payload: \
            events.append((event, payload))

        def run_generation(_request: object, emit: object, _context: object,
                           progress_callback: object = None
                           ) -> tuple[str, int, str]:
            emit("\n\n")
            return "<tool_call>...</tool_call>", 4, "stop"

        handler._run_generation = run_generation
        request = expert_server.GenerationRequest(
            endpoint="anthropic", prompt_ids=[10], cache_prefix_tokens=1,
            maximum=8, stream=True, stop=(), include_usage=False,
            tools=({"type": "function"},),
        )

        handler._serve_anthropic(request, "msg_test", object())

        deltas = [payload["delta"] for event, payload in events
                  if event == "content_block_delta"]
        self.assertEqual(deltas, [{
            "type": "input_json_delta",
            "partial_json": '{"city":"Chisinau"}',
        }])
        starts = [payload["content_block"] for event, payload in events
                  if event == "content_block_start"]
        self.assertEqual(starts[0]["type"], "tool_use")
        self.assertEqual(events[-1][0], "message_stop")

    def test_declared_qwen_grammar_installs_standard_response_parser(self) -> None:
        class Tokenizer:
            chat_template = (
                "<|im_start|>assistant <think></think> <tool_call>"
                "<function= <parameter= </parameter> </tool_call>"
            )
            response_template = None

            def parse_response(self, _text: str, **_kwargs: object) -> object:
                return {}

        tokenizer = Tokenizer()
        self.assertEqual(install_declared_response_protocol(tokenizer),
                         "xml-function-v1")
        self.assertEqual(
            tokenizer.response_template["fields"]["tool_calls"]["content"],
            "xml-inline",
        )

    def test_declared_bracket_grammar_installs_standard_response_parser(self) -> None:
        class Tokenizer:
            chat_template = (
                "[INST][/INST][THINK][/THINK][TOOL_CALLS][ARGS]"
                "[TOOL_RESULTS][/TOOL_RESULTS]"
            )
            response_template = None

            def parse_response(self, _text: str, **_kwargs: object) -> object:
                return {}

        tokenizer = Tokenizer()
        self.assertEqual(
            install_declared_response_protocol(tokenizer),
            "bracket-function-v1",
        )
        template = tokenizer.response_template
        self.assertEqual(
            template["start_anchor"], ["[/INST]", "[/TOOL_RESULTS]"]
        )
        self.assertTrue(template["fields"]["tool_calls"]["repeats"])
        self.assertEqual(
            template["fields"]["tool_calls"]["content"], "json"
        )

    def test_artifact_response_codec_precedes_checkpoint_parser(self) -> None:
        class Tokenizer:
            response_template = {"fields": {"content": {"content": "text"}}}

        codec = types.SimpleNamespace(supports_response_parsing=True)
        self.assertEqual(
            select_response_protocol(Tokenizer(), codec),
            "artifact-chat-codec-v1",
        )

    def test_standard_stream_parser_receives_prompt_and_tool_schema(self) -> None:
        class Tokenizer:
            def get_response_parser(self, **kwargs: object) -> object:
                self.kwargs = kwargs
                return object()

        app = application_fixture()
        app.response_protocol = "xml-function-v1"
        app.tokenizer = Tokenizer()
        tool = {"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object",
                "properties": {"limit": {"type": "integer"}}},
        }}
        request = expert_server.GenerationRequest(
            endpoint="chat", prompt_ids=[10, 11], cache_prefix_tokens=0,
            maximum=8, stream=True, stop=(), include_usage=False,
            tools=(tool,),
        )

        app.response_stream_parser(request)

        self.assertEqual(app.tokenizer.kwargs, {
            "prefix": [10, 11], "tools": [tool],
        })

    def test_standard_tool_schema_keyword_is_forwarded(self) -> None:
        class Tokenizer:
            def parse_response(self, _text: str, tools: object = None, *,
                               prefix: object = None) -> dict[str, object]:
                self.tools = tools
                self.prefix = prefix
                return {"role": "assistant", "content": "done"}

        app = application_fixture()
        app.response_protocol = "xml-function-v1"
        app.tokenizer = Tokenizer()
        tool = {"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object"},
        }}
        request = expert_server.GenerationRequest(
            endpoint="chat", prompt_ids=[10, 11], cache_prefix_tokens=0,
            maximum=8, stream=False, stop=(), include_usage=False,
            tools=(tool,),
        )
        parsed = app.parse_assistant_output("done", request)
        self.assertEqual(parsed.text, "done")
        self.assertEqual(app.tokenizer.tools, [tool])
        self.assertEqual(app.tokenizer.prefix, [10, 11])

    def test_raw_response_trace_preserves_pre_parser_tool_text(self) -> None:
        class Tokenizer:
            def parse_response(self, _text: str, **_kwargs: object
                               ) -> dict[str, object]:
                return {"role": "assistant", "tool_calls": [{
                    "type": "function", "function": {
                        "name": "read",
                        "arguments": {"path": "/Users/vm/project/app.js"},
                    },
                }]}

        with tempfile.TemporaryDirectory() as directory:
            app = application_fixture()
            app.response_protocol = "xml-function-v1"
            app.tokenizer = Tokenizer()
            app.raw_response_trace_path = Path(directory) / "trace.jsonl"
            app.raw_response_trace_lock = threading.Lock()
            app.raw_response_trace_remaining = 32
            tool = {"type": "function", "function": {
                "name": "read", "parameters": {"type": "object"},
            }}
            request = expert_server.GenerationRequest(
                endpoint="chat", prompt_ids=[10], cache_prefix_tokens=0,
                maximum=8, stream=False, stop=(), include_usage=False,
                tools=(tool,),
            )
            raw = (
                "<tool_call>\n<function=read>\n"
                "<parameter=path>/Users/vm/project/app.js</parameter>\n"
                "</function>\n</tool_call>"
            )
            output = app.parse_assistant_output(raw, request)
            record = json.loads(
                app.raw_response_trace_path.read_text(encoding="utf-8")
            )

        self.assertEqual(record["raw_suffix"], raw)
        self.assertEqual(
            json.loads(record["parsed_tool_calls"][0]["arguments"]),
            {"path": "/Users/vm/project/app.js"},
        )
        self.assertEqual(output.tool_calls[0].name, "read")

    def test_qwen_template_receives_xhigh_and_tools(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def apply_chat_template(self, _messages: object,
                                    **kwargs: object) -> list[int]:
                self.calls.append(kwargs)
                return [10, 11]

        app = application_fixture()
        app.args = types.SimpleNamespace(
            model="test-model", maximum_new_tokens=32, max_context=128,
        )
        app.tokenizer = Tokenizer()
        app.response_protocol = "fixture"
        request = app.parse_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {
                "name": "weather", "description": "Current weather",
                "parameters": {"type": "object"}, "strict": False,
            }}],
            "max_completion_tokens": 8,
        }, "chat")
        self.assertEqual(request.reasoning_effort, "xhigh")
        self.assertEqual(request.tools[0]["function"]["name"], "weather")
        self.assertEqual(
            list(request.tools[0]["function"]),
            ["name", "description", "parameters", "strict"],
        )
        self.assertEqual(len(app.tokenizer.calls), 2)
        for call in app.tokenizer.calls:
            self.assertEqual(call["reasoning_effort"], "xhigh")
            self.assertEqual(call["tools"][0]["function"]["name"], "weather")
            self.assertTrue(call["enable_thinking"])
            self.assertTrue(call["preserve_thinking"])

    def test_chat_template_thinking_effort_reaches_the_artifact(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def apply_chat_template(self, _messages: object,
                                    **kwargs: object) -> list[int]:
                self.calls.append(kwargs)
                return [10, 11]

        app = application_fixture()
        app.args = types.SimpleNamespace(
            model="test-model", maximum_new_tokens=32, max_context=128,
        )
        app.tokenizer = Tokenizer()
        request = app.parse_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "weather"}],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "medium",
            },
            "max_completion_tokens": 8,
        }, "chat")

        self.assertEqual(request.reasoning_effort, "medium")
        self.assertEqual(len(app.tokenizer.calls), 2)
        for call in app.tokenizer.calls:
            self.assertEqual(call["reasoning_effort"], "medium")

        with self.assertRaisesRegex(RequestError, "specified twice"):
            app.parse_request({
                "model": "test-model",
                "messages": [{"role": "user", "content": "weather"}],
                "reasoning_effort": "xhigh",
                "chat_template_kwargs": {"reasoning_effort": "medium"},
                "max_completion_tokens": 8,
            }, "chat")

    def test_artifact_maps_common_effort_to_tokenizer_vocabulary(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def apply_chat_template(self, _messages: object,
                                    **kwargs: object) -> list[int]:
                self.calls.append(kwargs)
                return [10, 11]

        app = application_fixture()
        app.tokenizer = Tokenizer()
        app.template_reasoning_effort_map = {
            "off": "none", "low": "high",
            "medium": "high", "xhigh": "high",
        }
        app._chat_prompt_ids(
            [{"role": "user", "content": "hi"}],
            reasoning_effort="medium", enable_thinking=True,
        )
        app._chat_prompt_ids(
            [{"role": "user", "content": "hi"}],
            reasoning_effort="xhigh", enable_thinking=False,
        )
        self.assertEqual(
            [call["reasoning_effort"] for call in app.tokenizer.calls],
            ["high", "none"],
        )

    def test_thinking_mode_selects_artifact_profile_and_request_overrides(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, _messages: object,
                                    **_kwargs: object) -> list[int]:
                return [10, 11]

        app = application_fixture()
        app.args = types.SimpleNamespace(
            model="test-model", maximum_new_tokens=32, max_context=128,
        )
        app.tokenizer = Tokenizer()
        app.sampling_profiles = {
            "thinking": SamplingSettings(1.0, 0.95, 20, 0.0, 0, 0.0),
            "non_thinking": SamplingSettings(0.7, 0.8, 20, 0.0, 0, 1.5),
        }

        request = app.parse_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_completion_tokens": 8,
        }, "chat")
        self.assertEqual(
            request.sampling,
            SamplingSettings(0.7, 0.8, 20, 0.0,
                             request.sampling.seed, 1.5),
        )

        overridden = app.parse_request({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.4,
            "presence_penalty": -0.25,
            "max_completion_tokens": 8,
        }, "chat")
        self.assertEqual(overridden.sampling.temperature, 0.4)
        self.assertEqual(overridden.sampling.top_p, 0.8)
        self.assertEqual(overridden.sampling.presence_penalty, -0.25)

    def test_responses_request_uses_artifact_sampling_contract(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, messages: object, **_kwargs: object) -> list[int]:
                self.messages = messages
                return [10, 11]

        app = application_fixture()
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
        sampled = app.parse_request({
            "model": "test-model", "input": "hello",
            "temperature": 0.5, "top_p": 0.8, "top_k": 7,
            "min_p": 0.1, "seed": 42,
        }, "responses")
        self.assertEqual(sampled.sampling,
                         SamplingSettings(0.5, 0.8, 7, 0.1, 42))

        app.args.max_context = 9
        clamped = app.parse_request({
            "model": "test-model", "input": "hello",
            "max_output_tokens": 8, "temperature": 0,
        }, "responses")
        self.assertEqual(clamped.maximum, 7)

    def test_artifact_encoder_fills_missing_chat_template(self) -> None:
        class Tokenizer:
            def encode(self, prompt: str, **kwargs: object) -> list[int]:
                self.prompt = prompt
                self.kwargs = kwargs
                return [0, 128803, 23166, 128804, 128822]

            def apply_chat_template(self, *_args: object,
                                    **_kwargs: object) -> list[int]:
                raise AssertionError("generic template must not be used")

        app = application_fixture()
        app.tokenizer = Tokenizer()
        seen: dict[str, object] = {}

        def encode_messages(messages: object, **kwargs: object) -> str:
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return "official prompt"

        app.artifact_chat_codec = ArtifactChatCodec(
            types.SimpleNamespace(encode_messages=encode_messages),
            Path("fixture.py"),
        )
        messages = [{"role": "user", "content": "Hi"}]
        self.assertEqual(app._chat_prompt_ids(messages),
                         [0, 128803, 23166, 128804, 128822])
        self.assertEqual(seen, {
            "messages": messages, "kwargs": {"thinking_mode": "thinking"},
        })
        self.assertEqual(app.tokenizer.prompt, "official prompt")
        self.assertEqual(app.tokenizer.kwargs,
                         {"add_special_tokens": False})

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

    def test_generation_error_is_not_written_after_client_disconnect(self) -> None:
        handler = Handler.__new__(Handler)
        handler._client_disconnected = lambda: True
        handler._sse = lambda _payload: self.fail(
            "a disconnected client must not receive another SSE event"
        )

        self.assertFalse(handler._send_generation_error(
            "chat", True, 500, "generation failed", "server_error",
            "generation_failed",
        ))

    def test_generation_error_absorbs_disconnect_during_sse_write(self) -> None:
        handler = Handler.__new__(Handler)
        handler._client_disconnected = lambda: False

        def disconnected_write(_payload: object) -> None:
            raise ConnectionResetError("peer reset")

        handler._sse = disconnected_write
        self.assertFalse(handler._send_generation_error(
            "chat", True, 500, "generation failed", "server_error",
            "generation_failed",
        ))

    def test_context_credits_are_bounded_and_reusable(self) -> None:
        app = application_fixture()
        app.worker = types.SimpleNamespace(
            kv_page_tokens=256, kv_page_capacity=3,
        )
        app.kv_credit_lock = threading.Lock()
        app.kv_reserved_pages = 0
        first = app.acquire_context_credits(257)
        self.assertEqual(first, 2)
        self.assertEqual(app.acquire_context_credits(512), 0)
        app.release_context_credits(first)
        self.assertEqual(app.acquire_context_credits(512), 2)

    def test_parked_sessions_use_host_capacity_not_gpu_page_capacity(
            self) -> None:
        app = application_fixture()
        app.worker = types.SimpleNamespace(
            kv_page_tokens=256, kv_page_capacity=2,
            session_parking=True, session_park_page_capacity=12,
        )
        app.kv_credit_lock = threading.Lock()
        app.kv_reserved_pages = 0

        self.assertEqual(app.acquire_context_credits(1024), 4)
        self.assertEqual(app.kv_reserved_pages, 4)

    def test_request_context_credits_shrink_to_populated_pages(self) -> None:
        app = application_fixture()
        app.worker = types.SimpleNamespace(
            kv_page_tokens=4, kv_page_capacity=4,
            session_parking=True, session_park_page_capacity=16,
        )
        app.kv_credit_lock = threading.Lock()
        app.kv_reserved_pages = 0
        context = expert_server.RequestContext(session=None, held_pages=0)

        self.assertTrue(app.resize_request_context(context, 9))
        self.assertEqual((context.held_pages, app.kv_reserved_pages), (9, 9))
        self.assertTrue(app.resize_request_context(context, 3))
        self.assertEqual((context.held_pages, app.kv_reserved_pages), (3, 3))

    def test_exact_max_context_does_not_reserve_retention_position(self) -> None:
        worker = types.SimpleNamespace(
            session_retention=True,
            kv_page_tokens=1,
            kv_page_capacity=8,
        )
        worker.drop_session = lambda _key: None
        app = application_fixture()
        app.args = types.SimpleNamespace(
            disable_session_retention=False,
            session_idle_seconds=1800.0,
            max_context=8,
        )
        app.worker = worker
        app.kv_credit_lock = threading.Lock()
        app.kv_reserved_pages = 0
        app.session_lock = threading.Lock()
        app.sessions = expert_server.OrderedDict()

        context = app.acquire_request_context([1, 2, 3, 4, 5, 6], 2)

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context.held_pages, 0)
        app.release_request_context(context)
        self.assertEqual(app.kv_reserved_pages, 0)

    def test_model_info_reports_effective_placement_contract(self) -> None:
        app = application_fixture()
        app.args = types.SimpleNamespace(
            model="test-model", build_id="build", maximum_queue=8,
            worker_capacity=4, host="127.0.0.1", port=8080,
            max_context=4096, maximum_new_tokens=512,
            worker_ram_cache_gib=48, worker_vram_cache_gib=18,
            placement_profile="capacity", worker_kv_cache_mib=2048,
            worker_kv_page_tokens=256, microbatch_window_ms=2.0,
            profile_gpu_phases=False, worker_prefill_chunk_tokens=256,
            enable_worker_cpu_hybrid=False,
            disable_session_retention=False, session_idle_seconds=1800.0,
            latency_window=4096, queue_timeout=1.0,
            generation_timeout=120.0, maximum_body_bytes=16 << 20,
        )
        app.manifest = {
            "source": {}, "format": {}, "quantization": {},
            "integrity": {"content_sha256": "manifest"},
            "indexes": {"dense_sha256": "dense", "experts_sha256": "experts"},
            "architecture": {}, "masses": {},
        }
        app.worker = types.SimpleNamespace(
            protocol=4, prefill_mode="causal_sequential",
            prefill_chunk_tokens=1, kv_dtype="bf16",
            request_stream_mode="per_request_nonblocking",
            rope_mode="resident_table",
            gpu_phase_timing=False,
            mtp_resource_available=True,
            mtp_runtime_ready=True, mtp_enabled=False,
            retain_previous_route=True,
            cpu_hybrid_enabled=True,
            kv_allocation="preallocated", kv_page_tokens=256,
            kv_page_bytes=1024, kv_page_capacity=8192,
            placement_mode="budgeted",
            placement_profile="capacity", ram_cache_bytes=48 << 30,
            vram_cache_bytes=18 << 30, placement_prefetch_enabled=False,
            placement_prefetch_state="disabled",
            placement_minimum_observations=2,
            stats=lambda: {"allocated_pages": 0, "reserved_pages": 0,
                           "cache_vram_hits": 7},
        )
        app.active = 0
        app.active_lock = threading.Lock()
        app.draining = threading.Event()
        app.session_lock = threading.Lock()
        app.sessions = {}
        app._worker_stats_lock = threading.Lock()
        app._worker_stats_cache = {}
        app._worker_stats_updated_at = 0.0
        app.worker_stats(refresh=True)

        info = app.info()
        self.assertEqual(info["worker_sessions"], {
            "enabled": False, "parking_enabled": False,
            "persistence_supported": False,
            "persistence_enabled": False,
            "park_ram_bytes": 0, "park_page_capacity": 0,
            "retained": 0, "retained_tokens": 0,
            "reserved_pages": 0, "parked_bytes": 0,
            "disk_retained": 0, "disk_retained_tokens": 0,
            "disk_bytes": 0, "disk_byte_limit": 0,
            "disk_ttl_seconds": 0.0,
        })
        self.assertEqual(info["worker_placement"], {
            "mode": "budgeted", "profile": "capacity",
            "routed_vram_policy": "fixed",
            "ram_cache_bytes": 48 << 30,
            "vram_cache_bytes": 18 << 30, "prefetch_enabled": False,
            "prefetch_state": "disabled",
            "minimum_recent_observations": 2,
        })
        self.assertEqual(info["runtime_config"]["placement_profile"],
                         "capacity")
        self.assertEqual(info["worker_prefill"], {
            "mode": "causal_sequential", "chunk_tokens": 1,
        })
        self.assertEqual(info["worker_execution"], {
            "request_stream_mode": "per_request_nonblocking",
            "rope_mode": "resident_table",
            "gpu_phase_timing": False,
            "mtp_resource_available": True,
            "mtp_runtime_ready": True,
            "mtp_enabled": False,
            "retain_previous_route": True,
            "cpu_hybrid_enabled": True,
            "response_protocol": None,
            "sampling": {
                "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 0.0,
                "profiles": {
                    "thinking": {
                        "temperature": 1.0, "top_p": 0.95,
                        "top_k": 20, "min_p": 0.0,
                        "presence_penalty": 0.0,
                    },
                    "non_thinking": {
                        "temperature": 1.0, "top_p": 0.95,
                        "top_k": 20, "min_p": 0.0,
                        "presence_penalty": 0.0,
                    },
                },
                "source": "tokenizer/generation_config.json",
            },
        })
        self.assertEqual(info["worker_kv"]["dtype"], "bf16")
        self.assertEqual(info["worker_kv"]["allocation"], "preallocated")
        self.assertEqual(info["worker_runtime"]["cache_vram_hits"], 7)

    def test_idle_request_dispatches_without_waiting_the_window(self) -> None:
        worker = FakeWorker()
        observed: list[int] = []
        # A huge window would make the step hang for seconds if the idle
        # fast path regressed; the fast path dispatches immediately.
        batcher = ContinuousDecodeBatcher(worker, 5000.0, observed.append)
        started = time.monotonic()
        self.assertEqual(batcher.step(1, False), 10)
        elapsed = time.monotonic() - started
        batcher.close()

        self.assertLess(elapsed, 1.0)
        self.assertEqual(worker.calls, [[(1, 0)]])
        self.assertEqual(observed, [1])

    def test_queued_rows_share_one_worker_step(self) -> None:
        class BlockingWorker:
            capacity = 4

            def __init__(self) -> None:
                self.calls: list[list[tuple[int, int]]] = []
                self.lock = threading.Lock()
                self.entered = threading.Event()
                self.release = threading.Event()

            def step(self, items: list[tuple[int, int]]) -> dict[int, int]:
                with self.lock:
                    self.calls.append(list(items))
                    first = len(self.calls) == 1
                if first:
                    self.entered.set()
                    self.release.wait(timeout=5)
                return {request_id: request_id * 10 + int(flag)
                        for request_id, flag in items}

        worker = BlockingWorker()
        observed: list[int] = []
        batcher = ContinuousDecodeBatcher(worker, 50.0, observed.append)
        results: dict[int, int] = {}

        def run(request_id: int) -> None:
            results[request_id] = batcher.step(request_id, request_id == 4)

        first = threading.Thread(target=run, args=(1,))
        first.start()
        self.assertTrue(worker.entered.wait(timeout=2))
        threads = [threading.Thread(target=run, args=(request_id,))
                   for request_id in range(2, 5)]
        for thread in threads:
            thread.start()
        # All three waiters enqueue while the worker is busy; when it frees
        # up they must share exactly one follow-up step.
        deadline = time.monotonic() + 2
        while batcher.pending.qsize() < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        worker.release.set()
        first.join(timeout=2)
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        batcher.close()

        self.assertEqual(len(worker.calls), 2)
        self.assertEqual(worker.calls[0], [(1, 0)])
        self.assertEqual(len(worker.calls[1]), 3)
        self.assertEqual(observed, [1, 3])
        self.assertEqual(results, {1: 10, 2: 20, 3: 30, 4: 41})

    def test_speculative_bonus_is_buffered_without_an_extra_worker_step(self) -> None:
        class Worker:
            capacity = 1

            def __init__(self) -> None:
                self.calls = 0
                self.active_ids = {7}

            def step(self, items: list[tuple[int, bool]]) -> dict[int, list[int]]:
                self.calls += 1
                return {items[0][0]: [101, 102]}

            def cancel(self, request_id: int) -> None:
                self.active_ids.discard(request_id)

        worker = Worker()
        batcher = ContinuousDecodeBatcher(worker, 0.0, lambda _rows: None)
        try:
            self.assertEqual(batcher.step(7, False), 101)
            self.assertEqual(batcher.step(7, True), 102)
            self.assertEqual(worker.calls, 1)
            self.assertEqual(worker.active_ids, set())
        finally:
            batcher.close()

    def test_hold_step_maps_to_worker_flag_2(self) -> None:
        worker = FakeWorker()
        batcher = ContinuousDecodeBatcher(worker, 0.0, lambda _rows: None)
        try:
            self.assertEqual(batcher.step(3, False, hold=True), 32)
        finally:
            batcher.close()
        self.assertEqual(worker.calls, [[(3, 2)]])

    def test_retained_mtp_turn_ends_with_hold_step(self) -> None:
        class Worker:
            mtp_enabled = True
            session_retention = True
            kv_page_tokens = 4
            kv_page_capacity = 64

            def __init__(self) -> None:
                self.active_ids: set[int] = set()
                self.retained: dict[int, int] = {}
                self.fed: dict[int, int] = {}

            def begin(self, request_id: int, prompt: list[int],
                      _context_limit: int,
                      _sampling: SamplingSettings) -> None:
                self.active_ids.add(request_id)
                self.fed[request_id] = len(prompt)

            def end_retain(self, request_id: int, session_key: int) -> int:
                self.active_ids.discard(request_id)
                self.retained[session_key] = self.fed.pop(request_id)
                return self.retained[session_key]

            def drop_session(self, session_key: int) -> None:
                self.retained.pop(session_key, None)

            def cancel(self, request_id: int) -> None:
                self.active_ids.discard(request_id)

            def stats(self) -> dict[str, int]:
                return {}

        class Batcher:
            def __init__(self, worker: Worker, tokens: list[int]) -> None:
                self.worker = worker
                self.tokens = tokens
                self.steps: list[tuple[bool, bool]] = []

            def step(self, request_id: int, final: bool,
                     hold: bool = False) -> int:
                self.steps.append((final, hold))
                if not final:
                    self.worker.fed[request_id] += 1
                return self.tokens.pop(0)

            def take_buffered(self, _request_id: int) -> list[int]:
                return []

        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return ",".join(str(token) for token in tokens)

        worker = Worker()
        app = application_fixture()
        app.args = types.SimpleNamespace(
            generation_timeout=60.0, disable_session_retention=False,
            session_idle_seconds=1800.0, max_context=64,
        )
        app.worker = worker
        app.tokenizer = Tokenizer()
        app.eos_token_ids = {22}
        app.request_id = iter(range(1, 100)).__next__
        app.increment = lambda *_args, **_kwargs: None
        app.observe_latency = lambda *_args, **_kwargs: None
        app.kv_credit_lock = threading.Lock()
        app.kv_reserved_pages = 0
        app.session_lock = threading.Lock()
        app.sessions = expert_server.OrderedDict()
        app.next_session_key = 1

        app.decode_batcher = Batcher(worker, [20, 21])
        context = app.acquire_request_context([10, 11, 12], 2)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(len(list(app.generate([10, 11, 12], 2, context))), 2)
        app.release_request_context(context)
        # The last step of a retained MTP turn is a hold (plain, slot kept),
        # never a speculative pair, so no unemitted bonus token can poison
        # the retained session prefix.
        self.assertEqual(app.decode_batcher.steps, [(False, False), (False, True)])
        self.assertTrue(context.retained)
        session = next(iter(app.sessions.values()))
        self.assertEqual(session.tokens, [10, 11, 12, 20, 21])


    def test_generation_stops_and_cancels_after_eos(self) -> None:
        class Worker:
            def __init__(self) -> None:
                self.active_ids: set[int] = set()

            def begin(self, request_id: int, _prompt: list[int],
                      _context_limit: int,
                      _sampling: SamplingSettings) -> None:
                self.active_ids.add(request_id)

            def cancel(self, request_id: int) -> None:
                self.active_ids.discard(request_id)

        class Batcher:
            def step(self, _request_id: int, _final: bool,
                     hold: bool = False) -> int:
                return 7

        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return ",".join(str(token) for token in tokens)

        app = application_fixture()
        app.args = types.SimpleNamespace(generation_timeout=10.0)
        app.request_id = lambda: 1
        app.worker = Worker()
        app.decode_batcher = Batcher()
        app.tokenizer = Tokenizer()
        app.eos_token_ids = {7}
        app.increment = lambda *_args, **_kwargs: None
        app.observe_latency = lambda *_args, **_kwargs: None

        result = list(app.generate([3], 5))
        self.assertEqual(result, [(7, "")])
        self.assertEqual(app.worker.active_ids, set())

    def test_non_retained_generation_omits_checkpoint(self) -> None:
        class Worker:
            session_retention = False
            mtp_enabled = False

            def __init__(self) -> None:
                self.active_ids: set[int] = set()
                self.checkpoint_tokens: int | None = -1

            def begin(self, request_id: int, _prompt: list[int],
                      _context_limit: int, _sampling: SamplingSettings,
                      checkpoint_tokens: int | None = None,
                      **_kwargs: object) -> None:
                self.checkpoint_tokens = checkpoint_tokens
                self.active_ids.add(request_id)

            def cancel(self, request_id: int) -> None:
                self.active_ids.discard(request_id)

            def stats(self) -> dict[str, int]:
                return {}

        class Batcher:
            def __init__(self, worker: Worker) -> None:
                self.worker = worker

            def step(self, request_id: int, final: bool,
                     hold: bool = False) -> int:
                if final:
                    self.worker.active_ids.discard(request_id)
                return 7

            def take_buffered(self, _request_id: int) -> list[int]:
                return []

        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return ",".join(str(token) for token in tokens)

        worker = Worker()
        app = application_fixture()
        app.args = types.SimpleNamespace(
            generation_timeout=10.0, disable_session_retention=False,
            session_idle_seconds=1800.0, max_context=64,
        )
        app.request_id = lambda: 1
        app.worker = worker
        app.decode_batcher = Batcher(worker)
        app.tokenizer = Tokenizer()
        app.eos_token_ids = {7}
        app.increment = lambda *_args, **_kwargs: None
        app.observe_latency = lambda *_args, **_kwargs: None
        context = types.SimpleNamespace(session=None, retained=False)

        result = list(app.generate(
            [3], 5, context, cancel_check=lambda: False,
            cache_prefix_tokens=1,
        ))

        self.assertEqual(result, [(7, "")])
        self.assertIsNone(worker.checkpoint_tokens)

    def test_retained_session_resumes_with_delta_tokens(self) -> None:
        class Worker:
            mtp_enabled = False

            def __init__(self) -> None:
                self.active_ids: set[int] = set()
                self.session_retention = True
                self.kv_page_tokens = 4
                self.kv_page_capacity = 64
                self.retained: dict[int, int] = {}
                self.fed: dict[int, int] = {}
                self.begins: list[tuple[str, list[int]]] = []

            def begin(self, request_id: int, prompt: list[int],
                      _context_limit: int,
                      _sampling: SamplingSettings) -> None:
                self.active_ids.add(request_id)
                self.fed[request_id] = len(prompt)
                self.begins.append(("fresh", list(prompt)))

            def begin_resume(self, request_id: int, session_key: int,
                             delta: list[int], _context_limit: int,
                             _sampling: SamplingSettings) -> None:
                assert session_key in self.retained
                self.active_ids.add(request_id)
                self.fed[request_id] = self.retained.pop(session_key) + len(delta)
                self.begins.append(("resume", list(delta)))

            def end_retain(self, request_id: int, session_key: int) -> int:
                self.active_ids.discard(request_id)
                self.retained[session_key] = self.fed.pop(request_id)
                return self.retained[session_key]

            def drop_session(self, session_key: int) -> None:
                self.retained.pop(session_key, None)

            def cancel(self, request_id: int) -> None:
                self.active_ids.discard(request_id)

            def stats(self) -> dict[str, int]:
                return {}

        class Batcher:
            def __init__(self, worker: Worker, tokens: list[int]) -> None:
                self.worker = worker
                self.tokens = tokens

            def step(self, request_id: int, _final: bool,
                     hold: bool = False) -> int:
                self.worker.fed[request_id] += 1
                return self.tokens.pop(0)

        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return ",".join(str(token) for token in tokens)

        worker = Worker()
        app = application_fixture()
        app.args = types.SimpleNamespace(
            generation_timeout=60.0, disable_session_retention=False,
            session_idle_seconds=1800.0, max_context=64,
        )
        app.worker = worker
        app.tokenizer = Tokenizer()
        app.eos_token_ids = {22}
        app.request_id = iter(range(1, 100)).__next__
        app.increment = lambda *_args, **_kwargs: None
        app.observe_latency = lambda *_args, **_kwargs: None
        app.kv_credit_lock = threading.Lock()
        app.kv_reserved_pages = 0
        app.session_lock = threading.Lock()
        app.sessions = expert_server.OrderedDict()
        app.next_session_key = 1

        app.decode_batcher = Batcher(worker, [20, 21])
        context = app.acquire_request_context([10, 11, 12], 2)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertIsNone(context.session)
        self.assertEqual(context.held_pages, 0)
        self.assertEqual(len(list(app.generate([10, 11, 12], 2, context))), 2)
        app.release_request_context(context)
        self.assertTrue(context.retained)
        self.assertEqual(app.kv_reserved_pages, 2)

        app.decode_batcher = Batcher(worker, [22])
        followup = [10, 11, 12, 20, 21, 30, 31]
        context = app.acquire_request_context(followup, 2)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertIsNotNone(context.session)
        self.assertEqual(context.held_pages, 2)
        self.assertEqual(len(list(app.generate(followup, 2, context))), 1)
        app.release_request_context(context)
        self.assertTrue(context.retained)
        self.assertEqual(worker.begins, [
            ("fresh", [10, 11, 12]),
            ("resume", [30, 31]),
        ])
        session = next(iter(app.sessions.values()))
        self.assertEqual(session.tokens, followup + [22])
        self.assertEqual(session.pages, 2)
        self.assertEqual(app.kv_reserved_pages, 2)

        self.assertTrue(app.evict_lru_session())
        self.assertEqual(worker.retained, {})
        self.assertEqual(app.kv_reserved_pages, 0)

    def test_cancelled_decode_retains_only_the_prompt_checkpoint(self) -> None:
        class Worker:
            session_retention = True
            mtp_enabled = False
            kv_page_tokens = 4
            kv_page_capacity = 64

            def __init__(self) -> None:
                self.active_ids: set[int] = set()
                self.retained: dict[int, int] = {}

            def begin(self, request_id: int, _prompt: list[int],
                      _context_limit: int,
                      _sampling: SamplingSettings,
                      **_kwargs: object) -> None:
                self.active_ids.add(request_id)

            def end_retain(self, request_id: int, session_key: int,
                           checkpoint_tokens: int) -> tuple[int, int, int]:
                self.active_ids.discard(request_id)
                self.retained[session_key] = checkpoint_tokens
                return checkpoint_tokens, 1, 4096

            def drop_session(self, session_key: int) -> None:
                self.retained.pop(session_key, None)

            def cancel(self, request_id: int) -> None:
                self.active_ids.discard(request_id)

            def stats(self) -> dict[str, int]:
                return {}

        class Batcher:
            def step(self, _request_id: int, _final: bool,
                     hold: bool = False) -> int:
                return 9

            def take_buffered(self, _request_id: int) -> list[int]:
                return []

        class Tokenizer:
            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return "".join(str(token) for token in tokens)

        worker = Worker()
        app = application_fixture()
        app.args = types.SimpleNamespace(
            generation_timeout=10.0, disable_session_retention=False,
            session_idle_seconds=1800.0, max_context=64,
        )
        app.request_id = lambda: 1
        app.worker = worker
        app.decode_batcher = Batcher()
        app.tokenizer = Tokenizer()
        app.eos_token_ids = set()
        app.increment = lambda *_args, **_kwargs: None
        app.observe_latency = lambda *_args, **_kwargs: None
        app.worker_stats = lambda refresh=True: {}
        app.kv_credit_lock = threading.Lock()
        app.kv_reserved_pages = 0
        app.session_lock = threading.Lock()
        app.sessions = expert_server.OrderedDict()
        app.next_session_key = 1
        app.id_lock = threading.Lock()
        context = expert_server.RequestContext(session=None, held_pages=0)

        stream = app.generate([1, 2, 3], 8, context=context)
        self.assertEqual(next(stream), (9, "9"))
        stream.close()

        self.assertTrue(context.retained)
        self.assertEqual(list(app.sessions.values())[0].tokens, [1, 2, 3])
        self.assertEqual(worker.retained, {1: 3})
        self.assertEqual(worker.active_ids, set())


if __name__ == "__main__":
    unittest.main()
