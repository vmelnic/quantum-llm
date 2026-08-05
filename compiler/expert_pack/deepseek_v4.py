"""Strict source contract for the pinned DeepSeek-V4-Flash checkpoint."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .constants import DTYPE_BYTES
from .errors import AdapterError
from .safetensors import SafeTensorCheckpoint


CONTRACT_NAME = "deepseek-v4-flash-source-v1"
_MTP_PATTERN = re.compile(r"^mtp\.(\d+)\.")


@dataclass(frozen=True)
class ExpectedTensor:
    dtype: str
    shape: tuple[int, ...]
    role: str


def _require_geometry(config: dict[str, object]) -> None:
    expected = {
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "hidden_act": "silu",
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 43,
        "num_hash_layers": 3,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "hc_mult": 4,
        "vocab_size": 129280,
        "max_position_embeddings": 1048576,
    }
    mismatches = [
        f"{key}: expected {value!r}, got {config.get(key)!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "DeepseekV4ForCausalLM" not in architectures:
        mismatches.append("architectures must contain DeepseekV4ForCausalLM")

    quantization = config.get("quantization_config")
    expected_quantization = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    if quantization != expected_quantization:
        mismatches.append(
            f"quantization_config: expected {expected_quantization!r}, got {quantization!r}"
        )

    expected_ratios = [0, 0]
    expected_ratios.extend(4 if layer % 2 else 128 for layer in range(1, 42))
    expected_ratios.append(0)
    if config.get("compress_ratios") != expected_ratios:
        mismatches.append("compress_ratios does not match the 43-layer CSA schedule")
    if mismatches:
        raise AdapterError("DeepSeek-V4 source geometry mismatch; " + "; ".join(mismatches[:8]))


def _add(
    result: dict[str, ExpectedTensor],
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    role: str,
) -> None:
    if name in result:
        raise RuntimeError(f"duplicate internal DeepSeek-V4 tensor expectation: {name}")
    result[name] = ExpectedTensor(dtype, shape, role)


def _add_fp8_matrix(
    result: dict[str, ExpectedTensor],
    name: str,
    shape: tuple[int, int],
    role: str,
) -> None:
    rows, columns = shape
    if rows % 128 or columns % 128:
        raise RuntimeError(f"FP8 matrix is not 128x128 block aligned: {name} {shape}")
    _add(result, name + ".weight", "F8_E4M3", shape, role + "_weight")
    _add(
        result,
        name + ".scale",
        "F8_E8M0",
        (rows // 128, columns // 128),
        role + "_scale",
    )


def _add_fp4_expert(
    result: dict[str, ExpectedTensor],
    prefix: str,
    hidden: int,
    intermediate: int,
    role: str,
) -> None:
    logical_shapes = {
        "w1": (intermediate, hidden),
        "w2": (hidden, intermediate),
        "w3": (intermediate, hidden),
    }
    for projection, (rows, columns) in logical_shapes.items():
        if columns % 2 or columns % 32:
            raise RuntimeError("FP4 expert geometry is not pack/block aligned")
        _add(
            result,
            f"{prefix}.{projection}.weight",
            "I8",
            (rows, columns // 2),
            role + "_weight",
        )
        _add(
            result,
            f"{prefix}.{projection}.scale",
            "F8_E8M0",
            (rows, columns // 32),
            role + "_scale",
        )


def _add_shared_expert(
    result: dict[str, ExpectedTensor],
    prefix: str,
    hidden: int,
    intermediate: int,
    role: str,
) -> None:
    _add_fp8_matrix(result, prefix + ".w1", (intermediate, hidden), role)
    _add_fp8_matrix(result, prefix + ".w2", (hidden, intermediate), role)
    _add_fp8_matrix(result, prefix + ".w3", (intermediate, hidden), role)


def _add_hyper_connection(
    result: dict[str, ExpectedTensor], prefix: str, hidden: int, role: str
) -> None:
    _add(result, prefix + "_base", "F32", (24,), role)
    _add(result, prefix + "_fn", "F32", (24, 4 * hidden), role)
    _add(result, prefix + "_scale", "F32", (3,), role)


def _add_attention(
    result: dict[str, ExpectedTensor], prefix: str, hidden: int, role: str
) -> None:
    _add(result, prefix + ".q_norm.weight", "BF16", (1024,), role)
    _add(result, prefix + ".kv_norm.weight", "BF16", (512,), role)
    _add(result, prefix + ".attn_sink", "F32", (64,), role)
    _add_fp8_matrix(result, prefix + ".wq_a", (1024, hidden), role)
    _add_fp8_matrix(result, prefix + ".wq_b", (32768, 1024), role)
    _add_fp8_matrix(result, prefix + ".wkv", (512, hidden), role)
    _add_fp8_matrix(result, prefix + ".wo_a", (8192, hidden), role)
    _add_fp8_matrix(result, prefix + ".wo_b", (hidden, 8192), role)


def _build_expected(config: dict[str, object], mtp_id: int) -> dict[str, ExpectedTensor]:
    _require_geometry(config)
    hidden = 4096
    intermediate = 2048
    experts = 256
    layers = 43
    vocab = 129280
    result: dict[str, ExpectedTensor] = {}

    _add(result, "embed.weight", "BF16", (vocab, hidden), "dense")
    _add(result, "norm.weight", "BF16", (hidden,), "dense")
    _add(result, "head.weight", "BF16", (vocab, hidden), "dense")
    _add(result, "hc_head_base", "F32", (4,), "dense")
    _add(result, "hc_head_fn", "F32", (4, 4 * hidden), "dense")
    _add(result, "hc_head_scale", "F32", (1,), "dense")

    for layer in range(layers):
        prefix = f"layers.{layer}"
        _add(result, prefix + ".attn_norm.weight", "BF16", (hidden,), "dense")
        _add(result, prefix + ".ffn_norm.weight", "BF16", (hidden,), "dense")
        _add_hyper_connection(result, prefix + ".hc_attn", hidden, "dense")
        _add_hyper_connection(result, prefix + ".hc_ffn", hidden, "dense")
        _add_attention(result, prefix + ".attn", hidden, "dense")

        _add(result, prefix + ".ffn.gate.weight", "BF16", (experts, hidden), "dense")
        if layer < 3:
            _add(
                result,
                f"{prefix}.ffn.gate.tid2eid",
                "I64",
                (vocab, 6),
                "dense",
            )
        else:
            _add(result, prefix + ".ffn.gate.bias", "F32", (experts,), "dense")

        for expert in range(experts):
            _add_fp4_expert(
                result,
                f"{prefix}.ffn.experts.{expert}",
                hidden,
                intermediate,
                "routed_expert",
            )
        _add_shared_expert(
            result,
            prefix + ".ffn.shared_experts",
            hidden,
            intermediate,
            "shared_expert",
        )

        # The final schedule entry belongs to the MTP layer; transformer layer
        # N consumes compress_ratios[N] directly.
        ratio = config["compress_ratios"][layer]  # validated above
        if ratio:
            compressor_width = 1024 if ratio == 4 else 512
            compressor = prefix + ".attn.compressor"
            _add(result, compressor + ".wkv.weight", "BF16", (compressor_width, hidden), "dense")
            _add(result, compressor + ".wgate.weight", "BF16", (compressor_width, hidden), "dense")
            _add(result, compressor + ".norm.weight", "BF16", (512,), "dense")
            ape_shape = (4, 1024) if ratio == 4 else (128, 512)
            _add(result, compressor + ".ape", "F32", ape_shape, "dense")

        if ratio == 4:
            indexer = prefix + ".attn.indexer"
            _add_fp8_matrix(result, indexer + ".wq_b", (8192, 1024), "dense")
            _add(result, indexer + ".weights_proj.weight", "BF16", (64, hidden), "dense")
            compressor = indexer + ".compressor"
            _add(result, compressor + ".wkv.weight", "BF16", (256, hidden), "dense")
            _add(result, compressor + ".wgate.weight", "BF16", (256, hidden), "dense")
            _add(result, compressor + ".norm.weight", "BF16", (128,), "dense")
            _add(result, compressor + ".ape", "F32", (4, 256), "dense")

    prefix = f"mtp.{mtp_id}"
    for name in ("norm", "hnorm", "enorm", "attn_norm", "ffn_norm"):
        _add(result, f"{prefix}.{name}.weight", "BF16", (hidden,), "mtp_dense")
    _add(result, prefix + ".hc_head_base", "F32", (4,), "mtp_dense")
    _add(result, prefix + ".hc_head_fn", "F32", (4, 4 * hidden), "mtp_dense")
    _add(result, prefix + ".hc_head_scale", "F32", (1,), "mtp_dense")
    _add_hyper_connection(result, prefix + ".hc_attn", hidden, "mtp_dense")
    _add_hyper_connection(result, prefix + ".hc_ffn", hidden, "mtp_dense")
    _add_fp8_matrix(result, prefix + ".h_proj", (hidden, hidden), "mtp_dense")
    _add_fp8_matrix(result, prefix + ".e_proj", (hidden, hidden), "mtp_dense")
    _add_attention(result, prefix + ".attn", hidden, "mtp_dense")
    _add(result, prefix + ".ffn.gate.weight", "BF16", (experts, hidden), "mtp_dense")
    _add(result, prefix + ".ffn.gate.bias", "F32", (experts,), "mtp_dense")
    for expert in range(experts):
        _add_fp4_expert(
            result,
            f"{prefix}.ffn.experts.{expert}",
            hidden,
            intermediate,
            "mtp_routed_expert",
        )
    _add_shared_expert(
        result,
        prefix + ".ffn.shared_experts",
        hidden,
        intermediate,
        "mtp_shared_expert",
    )
    return result


def validate_deepseek_v4_source(checkpoint: SafeTensorCheckpoint) -> dict[str, object]:
    """Validate a one-to-one name, dtype, shape, and byte contract."""

    mtp_ids = {
        int(match.group(1))
        for name in checkpoint.tensors
        if (match := _MTP_PATTERN.match(name)) is not None
    }
    if len(mtp_ids) != 1:
        raise AdapterError(f"DeepSeek-V4 source requires one MTP namespace, got {sorted(mtp_ids)}")
    mtp_id = next(iter(mtp_ids))
    expected = _build_expected(checkpoint.config, mtp_id)
    actual_names = set(checkpoint.tensors)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise AdapterError(
            f"DeepSeek-V4 tensor partition mismatch; missing={missing[:8]}, extra={extra[:8]}"
        )

    roles: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "bytes": 0}
    )
    for name in sorted(expected):
        wanted = expected[name]
        actual = checkpoint.tensors[name]
        if actual.dtype != wanted.dtype or actual.shape != wanted.shape:
            raise AdapterError(
                f"DeepSeek-V4 metadata mismatch for {name}: expected "
                f"{wanted.dtype} {wanted.shape}, got {actual.dtype} {actual.shape}"
            )
        expected_bytes = DTYPE_BYTES[wanted.dtype]
        for dimension in wanted.shape:
            expected_bytes *= dimension
        if actual.nbytes != expected_bytes:
            raise AdapterError(
                f"DeepSeek-V4 byte count mismatch for {name}: "
                f"expected {expected_bytes}, got {actual.nbytes}"
            )
        roles[wanted.role]["tensor_count"] += 1
        roles[wanted.role]["bytes"] += actual.nbytes

    total_bytes = sum(item["bytes"] for item in roles.values())
    if total_bytes != sum(info.nbytes for info in checkpoint.tensors.values()):
        raise AdapterError("DeepSeek-V4 role classification is not a byte-exact partition")
    return {
        "name": CONTRACT_NAME,
        "valid": True,
        "mtp_namespace": mtp_id,
        "tensor_count": len(expected),
        "tensor_bytes": total_bytes,
        "roles": {name: roles[name] for name in sorted(roles)},
        "source_quantization": {
            "routed_experts": "packed-fp4-e2m1-with-ue8m0-block32",
            "dense_matrices": "fp8-e4m3-with-ue8m0-block128x128",
        },
    }
