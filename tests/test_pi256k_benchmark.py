"""Offline contract tests for the autonomous Pi benchmark controller."""

import sys
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops/python"))
from pi256k_benchmark import (Rpc, call_measurement, check_gguf,
                              enrich, gguf_required_bytes,
                              is_live_research_tool, parse_llama_placement,
                              prompt_for_turn, research_succeeded,
                              provider_registry,
                              qwen_paging, resource_deltas, telemetry_after,
                              terminal_context_reached)


class Pi256kBenchmarkTests(unittest.TestCase):
    def test_gguf_contract_accepts_powershell_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "qwen3.8-27b-gguf-q4-k-m"
            artifact.mkdir()
            manifest = artifact / "manifest.json"
            marker = artifact / "COMPLETED"
            manifest_bytes = b"\xef\xbb\xbf" + json.dumps({
                "schema": "gguf-benchmark-artifact-v1",
                "revision": "efbb3b1f70a21d97fd4495240648405f7228554f",
                "files": {"Qwen3.8-27B-Q4_K_M.gguf": "pinned"},
            }).encode("utf-8")
            manifest.write_bytes(manifest_bytes)
            marker.write_bytes(b"\xef\xbb\xbf" + json.dumps({
                "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }).encode("utf-8"))
            with self.assertRaisesRegex(RuntimeError, "GGUF file size/manifest failure"):
                check_gguf(Path(directory))

    def test_pi_provider_keeps_xhigh_and_canonical_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider_registry(Path(directory), "quantum", 18080)
            entry = json.loads((Path(directory) / "models.json").read_text())[
                "providers"]["benchmark"]["models"][0]
        self.assertEqual(entry["thinkingLevelMap"]["xhigh"], "xhigh")
        self.assertEqual(entry["samplingParams"], {
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 0.5})

    def test_dialogue_starts_with_hi_and_grows_without_preloaded_answers(self) -> None:
        self.assertEqual(prompt_for_turn(0), "hi")
        self.assertIn("powershell", prompt_for_turn(1))
        self.assertNotEqual(prompt_for_turn(11), prompt_for_turn(12))
        self.assertLess(len(prompt_for_turn(120)), 600)

    def test_live_research_requires_standard_tool_and_returned_output(self) -> None:
        tool = {"name": "powershell", "args": {"command":
                "Invoke-RestMethod -Uri https://api.github.com/search/repositories?q=icalendar -TimeoutSec 20"},
                "error": False, "exit_code": 0,
                "result_excerpt": "https://github.com/example/repo"}
        self.assertTrue(is_live_research_tool(tool))
        self.assertTrue(research_succeeded(tool))
        self.assertFalse(research_succeeded({**tool, "result_excerpt": ""}))
        self.assertFalse(research_succeeded({**tool, "exit_code": 1}))
        self.assertFalse(is_live_research_tool({**tool, "name": "read"}))

    def test_full_gpu_gguf_capacity_exceeds_24_gib_with_mtp(self) -> None:
        main = 18_973_870_528
        draft = 1_680_271_776
        self.assertEqual(gguf_required_bytes(main), 24_766_204_352)
        self.assertEqual(gguf_required_bytes(main, draft), 26_446_476_128)
        self.assertGreater(gguf_required_bytes(main, draft), 24 * 1024**3)

    def test_llama_placement_distinguishes_mapped_cpu_and_gpu(self) -> None:
        log = ("load_tensors: CPU_Mapped model buffer size = 3072.00 MiB\n"
               "load_tensors: CPU model buffer size = 256.00 MiB\n"
               "load_tensors: CUDA0 model buffer size = 15000.50 MiB\n"
               "load_tensors: offloaded 53/65 layers to GPU\n"
               "llama_kv_cache: CPU KV buffer size = 480.00 MiB\n"
               "llama_kv_cache: CUDA0 KV buffer size = 512.00 MiB\n")
        result = parse_llama_placement(log)
        self.assertEqual(result["cpu_mapped_model_buffer_mib"], 3072)
        self.assertEqual(result["cpu_allocated_model_buffer_mib"], 256)
        self.assertEqual(result["gpu_model_buffer_mib"], 15000.5)
        self.assertEqual(result["cpu_kv_buffer_mib"], 480)
        self.assertEqual(result["gpu_kv_buffer_mib"], 512)
        self.assertTrue(result["cpu_placement_observed_at_load"])
        self.assertEqual(result["cpu_placement_selection_populated_prompt_tokens"], 0)
        self.assertEqual(result["layer_records"], [{"gpu_layers": 53,
                                                     "total_layers": 65}])

    def test_llama_placement_parses_pinned_server_trace_spacing_and_draft(self) -> None:
        log = (
            "0.04.279.224 I load_tensors: offloaded 48/65 layers to GPU\n"
            "0.04.279.228 I load_tensors:   CPU_Mapped model buffer size =  5044.97 MiB\n"
            "0.04.279.230 I load_tensors:        CUDA0 model buffer size = 13039.44 MiB\n"
            "0.08.769.274 I llama_kv_cache:        CPU KV buffer size =  1125.00 MiB\n"
            "0.08.899.097 I llama_kv_cache:      CUDA0 KV buffer size =  3375.00 MiB\n"
            "0.09.531.123 I load_tensors: offloaded 66/66 layers to GPU\n"
            "0.09.531.126 I load_tensors:   CPU_Mapped model buffer size =   682.03 MiB\n"
            "0.09.531.128 I load_tensors:        CUDA0 model buffer size =   909.96 MiB\n"
            "0.09.848.870 I llama_kv_cache:      CUDA0 KV buffer size =   531.25 MiB\n"
        )
        result = parse_llama_placement(log)
        self.assertEqual(result["cpu_mapped_model_buffer_mib"], 5727.0)
        self.assertAlmostEqual(result["gpu_model_buffer_mib"], 13949.4)
        self.assertEqual(result["cpu_kv_buffer_mib"], 1125.0)
        self.assertEqual(result["gpu_kv_buffer_mib"], 3906.25)
        self.assertEqual(result["layer_records"], [
            {"gpu_layers": 48, "total_layers": 65},
            {"gpu_layers": 66, "total_layers": 66},
        ])
        self.assertTrue(result["cpu_placement_observed_at_load"])

    def test_offload_metrics_do_not_call_process_reads_ssd_traffic(self) -> None:
        before = {"ram": {"available_bytes": 800 * 1024**2}, "gpu_free_mib": 900,
                  "process": {"working_set_bytes": 100, "private_usage_bytes": 80,
                              "read_transfer_bytes": 10, "write_transfer_bytes": 20,
                              "page_faults": 3, "peak_working_set_bytes": 110}}
        after = {"ram": {"available_bytes": 600 * 1024**2}, "gpu_free_mib": 850,
                 "process": {"working_set_bytes": 140, "private_usage_bytes": 100,
                             "read_transfer_bytes": 50, "write_transfer_bytes": 30,
                             "page_faults": 8, "peak_working_set_bytes": 150}}
        delta = resource_deltas(before, after)
        self.assertEqual(delta["host_available_ram_change_mib"], -200)
        self.assertEqual(delta["gpu_used_change_mib"], 50)
        self.assertEqual(delta["read_transfer_bytes_change"], 40)
        self.assertNotIn("ssd", delta)

    def test_qwen_paging_uses_provider_deltas(self) -> None:
        self.assertIsNone(qwen_paging([]))
        result = qwen_paging([{"cache_read_bytes": 10,
                               "cache_uploaded_bytes": 20,
                               "cache_demand_ram_hits": 2},
                              {"cache_read_bytes": 30,
                               "cache_uploaded_bytes": 40,
                               "mtp_cache_ram_hits": 1}])
        self.assertEqual(result["cache_read_bytes"], 40)
        self.assertEqual(result["cache_uploaded_bytes"], 60)
        self.assertEqual(result["cache_demand_ram_hits"], 2)
        self.assertTrue(result["paging_observed"])

    def test_metrics_keep_cold_and_reused_prefix_separate(self) -> None:
        turn = {"wall_seconds": 12.0, "model_calls": [
            {"usage": {"input": 1000, "output": 120, "reasoning": 80},
             "stop_reason": "stop", "client_ttft_seconds": 1.2,
             "client_time_to_first_visible_seconds": 4.5}], "tools": [
                {"name": "powershell", "seconds": 2.0, "error": False}],
            "visible_response": "answer"}
        telemetry = [
            {"resumed": False, "prefill_tokens": 900,
             "prefill_wall_seconds": 3.0, "ttft_seconds": 3.5,
             "generated_tokens": 120, "wall_seconds": 9.5},
            {"resumed": True, "prefill_tokens": 100,
             "prefill_wall_seconds": 0.5, "ttft_seconds": 0.7,
             "generated_tokens": 10, "wall_seconds": 1.2},
        ]
        result = enrich(turn, telemetry)
        self.assertEqual(result["populated_prompt_tokens"], 1000)
        self.assertEqual(result["cold_prefill_tokens"], 900)
        self.assertEqual(result["reused_prefill_tokens"], 100)
        self.assertEqual(result["reasoning_tokens"], 80)
        self.assertEqual(result["visible_tokens"], 40)
        self.assertEqual(result["tool_wall_seconds"], 2.0)
        self.assertEqual(result["client_ttft_seconds_per_call"], [1.2])
        self.assertEqual(result["provider_ttft_seconds_per_call"], [3.5, 0.7])
        self.assertEqual(result["automated_test_status"], "not_run")

    def test_only_near_window_actual_prompt_stops_the_trace(self) -> None:
        self.assertFalse(terminal_context_reached(2239))
        self.assertFalse(terminal_context_reached(17422))
        self.assertFalse(terminal_context_reached(254999))
        self.assertTrue(terminal_context_reached(255000))
        self.assertTrue(terminal_context_reached(255999))
        self.assertFalse(terminal_context_reached(256000))

    def test_per_call_record_keeps_actual_usage_and_provider_metrics(self) -> None:
        call = {"usage": {"input": 4000, "cacheRead": 100,
                          "output": 300, "reasoning": 200},
                "stop_reason": "toolUse", "client_ttft_seconds": 2.0,
                "client_time_to_first_visible_seconds": 4.0,
                "client_call_wall_seconds": 10.0}
        event = {"resumed": True, "prefill_tokens": 300,
                 "prefill_wall_seconds": 1.0, "ttft_seconds": 1.2,
                 "decode_tokens_per_second": 50.0}
        result = call_measurement(call, event, 2, 3, 12.0, 2.0, 42.0)
        self.assertEqual(result["populated_prompt_tokens"], 4100)
        self.assertEqual(result["reasoning_tokens"], 200)
        self.assertEqual(result["visible_tokens"], 100)
        self.assertEqual(result["provider_telemetry"]["reused_prefix_tokens"], 3800)
        self.assertEqual(result["end_to_end_wall_seconds_so_far"], 42.0)
        self.assertEqual(result["reused_prefill_seconds"], 1.0)
        self.assertEqual(result["provider_prefill_tokens_per_second"], 300.0)

    def test_decode_rate_is_derived_from_provider_wall_after_ttft(self) -> None:
        result = call_measurement(
            {"usage": {"input": 4000, "output": 100, "reasoning": 60}},
            {"resumed": False, "prefill_tokens": 4000, "generated_tokens": 100,
             "prefill_wall_seconds": 1.0, "ttft_seconds": 2.0,
             "wall_seconds": 4.0}, 0, 0, 4.0, 0.0, 4.0)
        self.assertEqual(result["provider_decode_tokens_per_second"], 50.0)
        self.assertEqual(result["cold_prefill_seconds"], 1.0)
        self.assertEqual(result["provider_decode_seconds"], 2.0)
        self.assertEqual(result["provider_prefill_tokens_per_second"], 4000.0)

    def test_missing_paging_counters_are_unknown_not_zero(self) -> None:
        self.assertIsNone(qwen_paging([{"event": "request_telemetry", "generated_tokens": 10}]))

    def test_provider_log_reader_does_not_consume_partial_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.jsonl"
            log.write_bytes(b'{"event":"request_telemetry","prefill_tokens":4}')
            self.assertEqual(telemetry_after(log, 0, "quantum"), ([], 0))
            with log.open("ab") as handle:
                handle.write(b"\n")
            events, offset = telemetry_after(log, 0, "quantum")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["prefill_tokens"], 4)
            self.assertEqual(telemetry_after(log, offset, "quantum"), ([], offset))

    def test_rpc_emits_each_call_before_turn_end(self) -> None:
        rpc = object.__new__(Rpc)
        rpc.pending = deque([
            {"type": "message_end", "message": {"role": "assistant",
                "usage": {"input": 4000, "output": 30}, "stopReason": "toolUse",
                "content": [{"type": "text", "text": "working"}]}},
            {"type": "agent_end"},
        ])
        rpc.call = lambda *_args, **_kwargs: {"success": True}
        observed = []
        turn = rpc.turn("hi", on_call=lambda call, index, elapsed, tools:
                        observed.append((index, call["usage"]["input"], elapsed, tools)))
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][1], 4000)
        self.assertEqual(len(turn["model_calls"]), 1)
        self.assertEqual(turn["model_calls"][0]["visible_response"], "working")


if __name__ == "__main__":
    unittest.main()
