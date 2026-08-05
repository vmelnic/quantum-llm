from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .compile import CompileOptions, compile_checkpoint
from .constants import PACK_ALIGNMENT, QUANT_PROFILE
from .deepseek_v4 import (
    estimate_deepseek_v4_representations,
    validate_deepseek_v4_source,
)
from .deepseek_slice import export_deepseek_compact_expert, qualify_deepseek_expert
from .errors import ExpertPackError
from .safetensors import SafeTensorCheckpoint
from .source_inventory import inspect_source
from .validate import validate_container


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expert-pack")
    commands = parser.add_subparsers(dest="command", required=True)

    compile_parser = commands.add_parser("compile", help="compile a local SafeTensors checkpoint")
    compile_parser.add_argument("--source", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--adapter", choices=("olmoe", "qwen3_next"), default="olmoe")
    compile_parser.add_argument("--quant-profile", choices=(QUANT_PROFILE,), default=QUANT_PROFILE)
    compile_parser.add_argument("--alignment", type=int, default=PACK_ALIGNMENT)
    compile_parser.add_argument("--max-expert-pack-bytes", type=int, default=2 * 1024**3)
    compile_parser.add_argument("--source-id")
    compile_parser.add_argument("--source-revision")
    compile_parser.add_argument("--resume", action="store_true")
    compile_parser.add_argument(
        "--reclaim-source-shards", action="store_true",
        help="DESTRUCTIVE: unlink source shards only after their records are durably committed",
    )

    validate_parser = commands.add_parser("validate", help="independently validate a completed container")
    validate_parser.add_argument("container", type=Path)

    inspect_parser = commands.add_parser(
        "inspect-source",
        help="validate all SafeTensors headers and report a read-only source inventory",
    )
    inspect_parser.add_argument("--source", type=Path, required=True)
    inspect_parser.add_argument(
        "--tensor-groups",
        action="store_true",
        help="include metadata grouped by tensor name pattern, dtype, and shape",
    )
    inspect_parser.add_argument(
        "--contract",
        choices=("deepseek_v4",),
        help="apply an exhaustive source-only architecture contract",
    )
    inspect_parser.add_argument(
        "--estimate-representations",
        action="store_true",
        help="derive DeepSeek compact/eager/hot-cache payload sizes without conversion",
    )

    slice_parser = commands.add_parser(
        "qualify-deepseek-expert",
        help="decode one complete real expert and compare candidate INT8 with PyTorch",
    )
    slice_parser.add_argument("--source", type=Path, required=True)
    slice_parser.add_argument("--layer", type=int, default=0)
    slice_parser.add_argument("--expert", type=int, default=0)
    slice_parser.add_argument("--row-chunk", type=int, default=128)
    slice_parser.add_argument("--no-torch-reference", action="store_true")

    export_parser = commands.add_parser(
        "export-deepseek-expert",
        help="describe one compact source expert as bounded SafeTensors extents",
    )
    export_parser.add_argument("--source", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--layer", type=int, default=0)
    export_parser.add_argument("--expert", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compile":
            result = compile_checkpoint(
                CompileOptions(
                    source=args.source,
                    output=args.output,
                    adapter=args.adapter,
                    quant_profile=args.quant_profile,
                    alignment=args.alignment,
                    max_expert_pack_bytes=args.max_expert_pack_bytes,
                    source_id=args.source_id,
                    source_revision=args.source_revision,
                    resume=args.resume,
                    reclaim_source_shards=args.reclaim_source_shards,
                )
            )
        elif args.command == "validate":
            result = validate_container(args.container)
        elif args.command == "inspect-source":
            checkpoint = SafeTensorCheckpoint(args.source)
            result = inspect_source(
                checkpoint,
                include_tensor_groups=args.tensor_groups,
            )
            if args.contract == "deepseek_v4":
                result["contract"] = validate_deepseek_v4_source(checkpoint)
            if args.estimate_representations:
                if args.contract != "deepseek_v4":
                    raise ValueError("--estimate-representations requires --contract deepseek_v4")
                result["representation_estimates"] = (
                    estimate_deepseek_v4_representations(checkpoint)
                )
        elif args.command == "qualify-deepseek-expert":
            result = qualify_deepseek_expert(
                SafeTensorCheckpoint(args.source),
                layer=args.layer,
                expert=args.expert,
                row_chunk=args.row_chunk,
                torch_reference=not args.no_torch_reference,
            )
        else:
            result = export_deepseek_compact_expert(
                SafeTensorCheckpoint(args.source), layer=args.layer,
                expert=args.expert, output=args.output,
            )
    except (ExpertPackError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "result": result}, sort_keys=True, indent=2))
    return 0
