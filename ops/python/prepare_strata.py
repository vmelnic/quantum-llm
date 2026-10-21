"""Prepare local, authenticated GGUF/MTP assets for the independent Strata service.

Uses the installed upstream packer without network access. No inference code or
model tensor values are changed here. The caller owns publication and lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    api_key = payload.get("api_key", "")
    if not api_key:
        raise SystemExit("An API key is required for the remote Strata endpoint")
    if not (args.model_dir / "COMPLETED").is_file():
        raise SystemExit("The GGUF publication is incomplete")
    target = sorted(args.model_dir.glob("*.gguf"))
    if len(target) != 2:
        raise SystemExit("Expected exactly two authenticated GGUF shards")
    mtp = args.data / "mtp" / "rt"
    for name in ("dense.bin", "dense.txt", "experts.bin", "draft_vocab.bin"):
        if not (mtp / name).is_file():
            raise SystemExit(f"Missing authenticated MTP asset: {name}")
    pack = args.data / "packs" / args.model_id
    pack.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "STRATA_NO_ACQUIRE": "1", "HF_HUB_OFFLINE": "1"}
    if not (pack / "native_experts.txt").is_file() or not (pack / "experts.bin").is_file():
        subprocess.run(
            [sys.executable, str(args.terminal / "tools" / "iq_pack.py"),
             "--gguf", str(target[0]), "--out", str(pack), "--experts-bin"],
            env=environment, check=True,
        )
    if not (pack / "tokenizer" / "vocab.json").is_file():
        raise SystemExit("Upstream packer did not publish a tokenizer")
    sys.path.insert(0, str(args.terminal / "tools"))
    from gguf_reader import GGUFFile

    ple = next((p for p in target if any(
        t.name == "per_layer_token_embd.weight" for t in GGUFFile(p).tensors)), None)
    if ple is None:
        raise SystemExit("No n-gram embedding table in the authenticated shards")
    expert_rows = [line.split() for line in (pack / "native_experts.txt").read_text().splitlines()
                   if line.strip() and not line.startswith("#")]
    maximum_blob = max(int(row[4]) for row in expert_rows)
    # A fixed slot budget below the existing 12 GiB routed-VRAM ceiling.
    slot_bytes = (maximum_blob + 255) // 256 * 256
    slots = (12 * 1024**3) // slot_bytes
    engine_args = [
        "--pack", str(pack), "--native", str(target[0]), "--ple-gguf", str(ple),
        "--expert-profile", str(args.terminal / "data" / "expert-profile.bin"),
        "--expert-cache", str(slots), "--expert-cache-per-layer",
        "--resident-experts", "--prefill", "auto", "--spec", "4",
        "--spec-min-p", "0.5", "--mtp", str(mtp),
        "--max-context", str(args.context), "--kv", "fp16",
        "--kv-resident", "32768", "--vram-reserve-mib", "1024",
    ]
    libraries = sorted({str(p.parent) for p in args.engine.parent.rglob("*.dll")})
    config = {
        "exe": str(args.engine), "args": engine_args, "cwd": str(args.terminal),
        "tokenizer": str(pack / "tokenizer"), "model_name": args.model_id,
        "log": str(args.config.with_suffix(".engine.log")), "lib_dirs": libraries,
        "port": args.port, "host": "0.0.0.0", "api_key": api_key,
        # Strata names this configuration level "high"; its HIGH_EFFORT
        # instruction is the model's native xhigh text. Requests accept xhigh.
        "reasoning_effort": "high", "parallel": 1,
        "env": {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_NO_ACQUIRE": "1",
                "HF_HUB_OFFLINE": "1"},
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.config.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.config)
    print(json.dumps({"status": "prepared", "model": args.model_id,
                      "context": args.context, "kv": "fp16", "route_tail_skip": 0,
                      "expert_slots": slots, "maximum_slot_bytes": slot_bytes,
                      "routed_vram_bound_bytes": slots * slot_bytes,
                      "config": str(args.config)}))


if __name__ == "__main__":
    main()
