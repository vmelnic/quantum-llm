from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters import ADAPTERS, adapt_checkpoint
from .compile import (
    CompileOptions,
    compile_checkpoint,
    refresh_sampling_profiles,
    refresh_runtime_model_program,
    upgrade_dense_mtp_program,
)
from .constants import PACK_ALIGNMENT, QUANT_PROFILE, QUANT_PROFILES
from .errors import ExpertPackError
from .quality import qualify_container_against_source
from .safetensors import SafeTensorCheckpoint
from .source_inventory import inspect_source
from .validate import validate_container


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expert-pack")
    commands = parser.add_subparsers(dest="command", required=True)

    compile_parser = commands.add_parser("compile", help="compile a local SafeTensors checkpoint")
    compile_parser.add_argument("--source", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument(
        "--adapter", choices=tuple(sorted(ADAPTERS)), default="olmoe"
    )
    compile_parser.add_argument("--quant-profile", choices=QUANT_PROFILES, default=QUANT_PROFILE)
    compile_parser.add_argument("--alignment", type=int, default=PACK_ALIGNMENT)
    compile_parser.add_argument("--max-expert-pack-bytes", type=int, default=2 * 1024**3)
    compile_parser.add_argument("--source-id")
    compile_parser.add_argument("--source-revision")
    compile_parser.add_argument("--config-file", default="config.json")
    compile_parser.add_argument(
        "--index-file", default="model.safetensors.index.json"
    )
    compile_parser.add_argument(
        "--sampling-profiles", type=Path,
        help="artifact-declared thinking/non-thinking sampling defaults",
    )
    compile_parser.add_argument(
        "--activation-calibration", type=Path,
        help="validated runtime Q8 captures for activation-aware FP4",
    )
    compile_parser.add_argument(
        "--dense-encoding-policy", type=Path,
        help="capability/role rules selecting alternate dense weight ABIs",
    )
    compile_parser.add_argument(
        "--dense-activation-input", choices=("q8", "bf16"), default="q8",
        help="artifact-declared activation operand encoding for dense FP4 projections",
    )
    compile_parser.add_argument("--resume", action="store_true")
    compile_parser.add_argument(
        "--reclaim-source-shards", action="store_true",
        help="DESTRUCTIVE: unlink source shards only after their records are durably committed",
    )

    validate_parser = commands.add_parser("validate", help="independently validate a completed container")
    validate_parser.add_argument("container", type=Path)

    quality_parser = commands.add_parser(
        "qualify-fp4-container",
        help="compare a published FP4 container with its declared SafeTensors source",
    )
    quality_parser.add_argument("--source", type=Path, required=True)
    quality_parser.add_argument("--container", type=Path, required=True)
    quality_parser.add_argument("--samples-per-tensor", type=int, default=8)
    quality_parser.add_argument("--maximum-relative-l2", type=float, default=0.20)
    quality_parser.add_argument("--minimum-cosine", type=float, default=0.98)

    refresh_parser = commands.add_parser(
        "refresh-model-program",
        help="clone a valid container and recompile only its VM metadata",
    )
    refresh_parser.add_argument("--container", type=Path, required=True)
    refresh_parser.add_argument("--output", type=Path, required=True)
    refresh_parser.add_argument("--source", type=Path, required=True)
    refresh_parser.add_argument(
        "--adapter", choices=tuple(sorted(ADAPTERS)), required=True
    )
    refresh_parser.add_argument("--config-file", default="config.json")
    refresh_parser.add_argument(
        "--index-file", default="model.safetensors.index.json"
    )
    refresh_parser.add_argument(
        "--dense-activation-input", choices=("q8", "bf16"),
        help="replace the artifact-declared dense activation operand encoding",
    )

    sampling_parser = commands.add_parser(
        "refresh-sampling-profiles",
        help="clone a valid container and replace only sampling metadata",
    )
    sampling_parser.add_argument("--container", type=Path, required=True)
    sampling_parser.add_argument("--output", type=Path, required=True)
    sampling_parser.add_argument("--profiles", type=Path, required=True)

    mtp_upgrade_parser = commands.add_parser(
        "upgrade-dense-mtp-program",
        help="clone an authenticated dense-MTP ABI v1 artifact and migrate its VM program to ABI v2",
    )
    mtp_upgrade_parser.add_argument("--container", type=Path, required=True)
    mtp_upgrade_parser.add_argument("--output", type=Path, required=True)
    mtp_upgrade_parser.add_argument(
        "--draft-depth", type=int, choices=(3, 4), default=4
    )
    mtp_upgrade_parser.add_argument(
        "--draft-vocabulary-size", type=int, default=65_536
    )

    inspect_parser = commands.add_parser(
        "inspect-source",
        help="validate all SafeTensors headers and report a read-only source inventory",
    )
    inspect_parser.add_argument("--source", type=Path, required=True)
    inspect_parser.add_argument("--config-file", default="config.json")
    inspect_parser.add_argument(
        "--index-file", default="model.safetensors.index.json"
    )
    inspect_parser.add_argument(
        "--tensor-groups",
        action="store_true",
        help="include metadata grouped by tensor name pattern, dtype, and shape",
    )
    inspect_parser.add_argument(
        "--adapter",
        choices=tuple(sorted(ADAPTERS)),
        help="apply a strict architecture adapter without writing a pack",
    )
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
                    sampling_profiles=args.sampling_profiles,
                    activation_calibration=args.activation_calibration,
                    dense_encoding_policy=args.dense_encoding_policy,
                    dense_activation_input=args.dense_activation_input,
                    config_file=args.config_file,
                    index_file=args.index_file,
                    resume=args.resume,
                    reclaim_source_shards=args.reclaim_source_shards,
                )
            )
        elif args.command == "validate":
            result = validate_container(args.container)
        elif args.command == "qualify-fp4-container":
            result = qualify_container_against_source(
                args.source,
                args.container,
                samples_per_tensor=args.samples_per_tensor,
                maximum_relative_l2=args.maximum_relative_l2,
                minimum_cosine=args.minimum_cosine,
            )
        elif args.command == "refresh-model-program":
            result = refresh_runtime_model_program(
                args.container, args.output, args.source, args.adapter,
                config_file=args.config_file, index_file=args.index_file,
                dense_activation_input=args.dense_activation_input,
            )
        elif args.command == "refresh-sampling-profiles":
            result = refresh_sampling_profiles(
                args.container, args.output, args.profiles
            )
        elif args.command == "upgrade-dense-mtp-program":
            result = upgrade_dense_mtp_program(
                args.container,
                args.output,
                draft_depth=args.draft_depth,
                draft_vocabulary_size=args.draft_vocabulary_size,
            )
        elif args.command == "inspect-source":
            checkpoint = SafeTensorCheckpoint(
                args.source, config_file=args.config_file,
                index_file=args.index_file,
            )
            result = inspect_source(
                checkpoint,
                include_tensor_groups=args.tensor_groups,
            )
            if args.adapter:
                adapted = adapt_checkpoint(checkpoint, args.adapter)
                topology = adapted.runtime_topology
                result["adapter"] = {
                    "name": args.adapter,
                    "dense_tensor_count": len(adapted.dense),
                    "expert_record_count": len(adapted.experts),
                    "source_tensor_count": adapted.source_tensor_count,
                    "architecture_id": topology.architecture_id,
                    "logical_layer_count": len(topology.layers),
                    "operation_count": len(topology.operations),
                    "required_kernels": [
                        {"capability": capability, "abi": abi}
                        for capability, abi in topology.required_kernels
                    ],
                    "routed_components": [
                        {
                            "name": component.name,
                            "layer_count": component.layer_count,
                            "experts_per_layer": component.experts_per_layer,
                            "route_width": component.route_width,
                            "hidden_size": component.hidden_size,
                            "intermediate_size": component.intermediate_size,
                        }
                        for component in topology.components
                    ],
                }
    except (ExpertPackError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "result": result}, sort_keys=True, indent=2))
    return 0
