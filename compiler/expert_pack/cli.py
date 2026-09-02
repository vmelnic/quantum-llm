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
)
from .constants import PACK_ALIGNMENT, QUANT_PROFILE, QUANT_PROFILES
from .deepseek_v4 import (
    estimate_deepseek_v4_representations,
    validate_deepseek_v4_source,
)
from .deepseek_slice import (
    export_deepseek_compact_expert,
    export_deepseek_routed_catalog,
    pack_deepseek_routed_catalog,
    export_deepseek_fp8_matrix,
    export_deepseek_hca_slice,
    export_deepseek_io_oracle,
    export_deepseek_mtp_glue_oracle,
    export_deepseek_mtp_block_oracle,
    export_deepseek_csa_slice,
    export_deepseek_attention_oracle,
    export_deepseek_typed_set,
    export_deepseek_mtp_set,
    export_deepseek_dense_set,
    export_deepseek_shared_expert,
    export_deepseek_shared_set,
    qualify_deepseek_expert,
    qualify_deepseek_fp8_matrix,
    qualify_deepseek_int8_organs,
    qualify_deepseek_compact_matrix,
    qualify_deepseek_shared_expert,
)
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

    sampling_parser = commands.add_parser(
        "refresh-sampling-profiles",
        help="clone a valid container and replace only sampling metadata",
    )
    sampling_parser.add_argument("--container", type=Path, required=True)
    sampling_parser.add_argument("--output", type=Path, required=True)
    sampling_parser.add_argument("--profiles", type=Path, required=True)

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
    shared_slice_parser = commands.add_parser(
        "qualify-deepseek-shared",
        help="decode one FP8 shared expert and compare the SM86 candidate",
    )
    shared_slice_parser.add_argument("--source", type=Path, required=True)
    shared_slice_parser.add_argument("--layer", type=int, default=0)
    shared_slice_parser.add_argument("--row-chunk", type=int, default=128)
    shared_slice_parser.add_argument("--no-torch-reference", action="store_true")
    dense_slice_parser = commands.add_parser(
        "qualify-deepseek-fp8-matrix",
        help="qualify one block-scaled FP8 dense matrix",
    )
    dense_slice_parser.add_argument("--source", type=Path, required=True)
    dense_slice_parser.add_argument("--name", required=True)
    dense_slice_parser.add_argument("--row-chunk", type=int, default=128)
    dense_slice_parser.add_argument("--no-torch-reference", action="store_true")
    organ_gate_parser = commands.add_parser(
        "qualify-deepseek-int8-organs",
        help="gate FP8-to-INT8 loss on representative always-active organs",
    )
    organ_gate_parser.add_argument("--source", type=Path, required=True)
    organ_gate_parser.add_argument(
        "--layers", type=int, nargs="+", default=(0, 21, 42)
    )
    organ_gate_parser.add_argument("--row-chunk", type=int, default=128)
    organ_gate_parser.add_argument(
        "--maximum-weight-relative-l2", type=float, default=0.02
    )
    organ_gate_parser.add_argument(
        "--maximum-projection-relative-l2", type=float, default=0.02
    )
    organ_gate_parser.add_argument("--minimum-cosine", type=float, default=0.999)
    compact_dense_parser = commands.add_parser(
        "qualify-deepseek-compact-matrix",
        help="screen compact symmetric dense candidates without publishing an ABI",
    )
    compact_dense_parser.add_argument("--source", type=Path, required=True)
    compact_dense_parser.add_argument("--name", required=True)
    compact_dense_parser.add_argument("--bits", type=int, nargs="+", default=(4, 5, 6))
    compact_dense_parser.add_argument(
        "--block-sizes", type=int, nargs="+", default=(32, 64, 128)
    )
    compact_dense_parser.add_argument("--row-chunk", type=int, default=128)

    export_parser = commands.add_parser(
        "export-deepseek-expert",
        help="describe one compact source expert as bounded SafeTensors extents",
    )
    export_parser.add_argument("--source", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--layer", type=int, default=0)
    export_parser.add_argument("--expert", type=int, default=0)
    routed_catalog_parser = commands.add_parser(
        "export-deepseek-routed-catalog",
        help="index all 43x256 routed experts as compact SafeTensors extents",
    )
    routed_catalog_parser.add_argument("--source", type=Path, required=True)
    routed_catalog_parser.add_argument("--output", type=Path, required=True)
    compact_pack_parser = commands.add_parser(
        "pack-deepseek-routed",
        help="repack routed experts into resumable aligned compact layer shards",
    )
    compact_pack_parser.add_argument("--catalog", type=Path, required=True)
    compact_pack_parser.add_argument("--source", type=Path, required=True)
    compact_pack_parser.add_argument("--output", type=Path, required=True)
    compact_pack_parser.add_argument("--resume", action="store_true")
    io_oracle_parser = commands.add_parser(
        "export-deepseek-io-oracle",
        help="emit an independent embedding and output-head oracle",
    )
    io_oracle_parser.add_argument("--source", type=Path, required=True)
    io_oracle_parser.add_argument("--output", type=Path, required=True)
    io_oracle_parser.add_argument("--token", type=int, default=42)
    io_oracle_parser.add_argument("--row-chunk", type=int, default=512)
    shared_parser = commands.add_parser(
        "export-deepseek-shared",
        help="describe one FP8 shared expert as bounded SafeTensors extents",
    )
    shared_parser.add_argument("--source", type=Path, required=True)
    shared_parser.add_argument("--output", type=Path, required=True)
    shared_parser.add_argument("--layer", type=int, default=0)
    shared_set_parser = commands.add_parser(
        "export-deepseek-shared-set",
        help="describe all 43 shared experts for bounded startup residency",
    )
    shared_set_parser.add_argument("--source", type=Path, required=True)
    shared_set_parser.add_argument("--output", type=Path, required=True)
    dense_parser = commands.add_parser(
        "export-deepseek-fp8-matrix",
        help="describe one block-scaled FP8 matrix as SafeTensors extents",
    )
    dense_parser.add_argument("--source", type=Path, required=True)
    dense_parser.add_argument("--output", type=Path, required=True)
    dense_parser.add_argument("--name", required=True)
    dense_set_parser = commands.add_parser(
        "export-deepseek-dense-set",
        help="describe all 236 main-model FP8 matrices for residency",
    )
    dense_set_parser.add_argument("--source", type=Path, required=True)
    dense_set_parser.add_argument("--output", type=Path, required=True)
    hca_parser = commands.add_parser(
        "export-deepseek-hca",
        help="describe one real HCA site and emit its deterministic FP32 oracle",
    )
    hca_parser.add_argument("--source", type=Path, required=True)
    hca_parser.add_argument("--output", type=Path, required=True)
    hca_parser.add_argument("--layer", type=int, default=0)
    hca_parser.add_argument("--site", choices=("attn", "ffn"), default="attn")
    typed_set_parser = commands.add_parser(
        "export-deepseek-typed-set",
        help="describe all 834 main-model BF16/F32/I64 tensors for residency",
    )
    typed_set_parser.add_argument("--source", type=Path, required=True)
    typed_set_parser.add_argument("--output", type=Path, required=True)
    mtp_set_parser = commands.add_parser(
        "export-deepseek-mtp-set",
        help="authenticate the native DeepSeek MTP resource set",
    )
    mtp_set_parser.add_argument("--source", type=Path, required=True)
    mtp_set_parser.add_argument("--output", type=Path, required=True)
    mtp_oracle_parser = commands.add_parser(
        "export-deepseek-mtp-glue-oracle",
        help="emit independent DeepSeek MTP input/output boundary oracles",
    )
    mtp_oracle_parser.add_argument("--source", type=Path, required=True)
    mtp_oracle_parser.add_argument("--output", type=Path, required=True)
    mtp_oracle_parser.add_argument("--token", type=int, default=42)
    mtp_oracle_parser.add_argument("--position", type=int, default=1)
    mtp_oracle_parser.add_argument("--previous-streams", type=Path)
    mtp_block_parser = commands.add_parser(
        "export-deepseek-mtp-block-oracle",
        help="emit a complete four-position DeepSeek MTP block oracle",
    )
    mtp_block_parser.add_argument("--source", type=Path, required=True)
    mtp_block_parser.add_argument("--output", type=Path, required=True)
    mtp_block_parser.add_argument(
        "--tokens", type=int, nargs=4,
        default=(0, 128803, 23166, 19923),
    )
    mtp_block_parser.add_argument("--row-chunk", type=int, default=512)
    csa_parser = commands.add_parser(
        "export-deepseek-csa",
        help="describe one ratio-four CSA compressor and emit its decode oracle",
    )
    csa_parser.add_argument("--source", type=Path, required=True)
    csa_parser.add_argument("--output", type=Path, required=True)
    csa_parser.add_argument("--layer", type=int, default=2)
    attention_parser = commands.add_parser(
        "export-deepseek-attention-oracle",
        help="emit an independent complete four-token attention oracle",
    )
    attention_parser.add_argument("--source", type=Path, required=True)
    attention_parser.add_argument("--output", type=Path, required=True)
    attention_parser.add_argument("--layer", type=int, default=2)
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
            )
        elif args.command == "refresh-sampling-profiles":
            result = refresh_sampling_profiles(
                args.container, args.output, args.profiles
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
        elif args.command == "qualify-deepseek-shared":
            result = qualify_deepseek_shared_expert(
                SafeTensorCheckpoint(args.source), layer=args.layer,
                row_chunk=args.row_chunk,
                torch_reference=not args.no_torch_reference,
            )
        elif args.command == "qualify-deepseek-fp8-matrix":
            result = qualify_deepseek_fp8_matrix(
                SafeTensorCheckpoint(args.source), name=args.name,
                row_chunk=args.row_chunk,
                torch_reference=not args.no_torch_reference,
            )
        elif args.command == "qualify-deepseek-int8-organs":
            result = qualify_deepseek_int8_organs(
                SafeTensorCheckpoint(args.source),
                layers=tuple(args.layers),
                row_chunk=args.row_chunk,
                maximum_weight_relative_l2=args.maximum_weight_relative_l2,
                maximum_projection_relative_l2=(
                    args.maximum_projection_relative_l2
                ),
                minimum_cosine=args.minimum_cosine,
            )
        elif args.command == "qualify-deepseek-compact-matrix":
            result = qualify_deepseek_compact_matrix(
                SafeTensorCheckpoint(args.source), name=args.name,
                bits=tuple(args.bits), block_sizes=tuple(args.block_sizes),
                row_chunk=args.row_chunk,
            )
        elif args.command == "export-deepseek-expert":
            result = export_deepseek_compact_expert(
                SafeTensorCheckpoint(args.source), layer=args.layer,
                expert=args.expert, output=args.output,
            )
        elif args.command == "export-deepseek-routed-catalog":
            result = export_deepseek_routed_catalog(
                SafeTensorCheckpoint(args.source), output=args.output,
            )
        elif args.command == "pack-deepseek-routed":
            result = pack_deepseek_routed_catalog(
                catalog_root=args.catalog, source_root=args.source,
                output=args.output, resume=args.resume,
            )
        elif args.command == "export-deepseek-io-oracle":
            result = export_deepseek_io_oracle(
                SafeTensorCheckpoint(args.source), output=args.output,
                token=args.token, row_chunk=args.row_chunk,
            )
        elif args.command == "export-deepseek-shared":
            result = export_deepseek_shared_expert(
                SafeTensorCheckpoint(args.source), layer=args.layer,
                output=args.output,
            )
        elif args.command == "export-deepseek-shared-set":
            result = export_deepseek_shared_set(
                SafeTensorCheckpoint(args.source), output=args.output,
            )
        elif args.command == "export-deepseek-fp8-matrix":
            result = export_deepseek_fp8_matrix(
                SafeTensorCheckpoint(args.source), name=args.name,
                output=args.output,
            )
        elif args.command == "export-deepseek-dense-set":
            result = export_deepseek_dense_set(
                SafeTensorCheckpoint(args.source), output=args.output,
            )
        elif args.command == "export-deepseek-hca":
            result = export_deepseek_hca_slice(
                SafeTensorCheckpoint(args.source), layer=args.layer,
                site=args.site, output=args.output,
            )
        elif args.command == "export-deepseek-typed-set":
            result = export_deepseek_typed_set(
                SafeTensorCheckpoint(args.source), output=args.output,
            )
        elif args.command == "export-deepseek-mtp-set":
            result = export_deepseek_mtp_set(
                SafeTensorCheckpoint(args.source), output=args.output,
            )
        elif args.command == "export-deepseek-mtp-glue-oracle":
            result = export_deepseek_mtp_glue_oracle(
                SafeTensorCheckpoint(args.source), output=args.output,
                token=args.token, position=args.position,
                previous_streams=args.previous_streams,
            )
        elif args.command == "export-deepseek-mtp-block-oracle":
            result = export_deepseek_mtp_block_oracle(
                SafeTensorCheckpoint(args.source), output=args.output,
                tokens=tuple(args.tokens), row_chunk=args.row_chunk,
            )
        elif args.command == "export-deepseek-csa":
            result = export_deepseek_csa_slice(
                SafeTensorCheckpoint(args.source), layer=args.layer,
                output=args.output,
            )
        else:
            result = export_deepseek_attention_oracle(
                SafeTensorCheckpoint(args.source), layer=args.layer,
                output=args.output,
            )
    except (ExpertPackError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "result": result}, sort_keys=True, indent=2))
    return 0
