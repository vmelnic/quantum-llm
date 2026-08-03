#!/usr/bin/env python3
"""Cross the real P1 compiler manifest into the shared P0 planner CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_compiler_fixture(repo: Path):
    path = repo / "tests" / "compiler" / "test_expert_pack.py"
    spec = importlib.util.spec_from_file_location("p1_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: feasibility_cli_cross_test.py <cli> <repo>")
    cli = Path(sys.argv[1])
    repo = Path(sys.argv[2])
    sys.path.insert(0, str(repo))
    fixture = _load_compiler_fixture(repo)

    from compiler.expert_pack.compile import CompileOptions, compile_checkpoint

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        fixture._make_fixture(source)
        output = root / "pack"
        compiled = compile_checkpoint(
            CompileOptions(
                source=source,
                output=output,
                source_id="synthetic/olmoe",
                source_revision="fixture-v1",
            )
        )
        hardware = {
            "planner": {
                "total_ram_bytes": 67032328 * 1024,
                "available_ram_bytes": 58430572 * 1024,
                "vram_total_bytes": 24576 * 1024 * 1024,
                "vram_available_bytes": 24576 * 1024 * 1024,
                "disk_free_bytes": 340901289984,
                "sustained_storage_read_bytes_per_second": 1,
                "sustained_h2d_bytes_per_second": 12000000000,
                "storage_bandwidth_measured": True,
                "h2d_bandwidth_measured": False,
            }
        }
        request = {
            "target_tokens_per_second_milli": 10000,
            "concurrency": 1,
            "vram_expert_budget_bytes": 0,
            "ram_expert_budget_bytes": 0,
            "expected_vram_hit_ppm": 0,
            "expected_ram_hit_ppm": 0,
            "minimum_workspace_bytes": 0,
            "minimum_staging_bytes": 1,
            "kv_bytes_per_request": 1,
        }
        hardware_path = root / "hardware.json"
        request_path = root / "request.json"
        hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run(
            [str(cli), str(output / "manifest.json"), str(hardware_path), str(request_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 2:
            raise AssertionError(f"expected impossible(2), got {result.returncode}: {result.stderr}")
        planned = json.loads(result.stdout)
        if planned["manifest_sha256"] != compiled["manifest_content_sha256"]:
            raise AssertionError("P0 and P1 computed different manifest content hashes")
        if planned["decision"]["status"] != "impossible":
            raise AssertionError("planner did not preserve impossible status")
        if planned["decision"]["limiting_resource"] != "storage_bandwidth":
            raise AssertionError("planner did not identify storage bandwidth")
    print("P1 manifest -> P0 planner cross-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
