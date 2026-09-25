from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from compiler.expert_pack.adapters import RuntimeOperationTopology, adapt_checkpoint
from compiler.expert_pack.compile import (
    CompileOptions,
    _runtime_model_descriptor_bytes,
    compile_checkpoint,
    refresh_runtime_model_program,
    refresh_sampling_profiles,
    upgrade_dense_mtp_program,
)
from compiler.expert_pack.constants import (
    EXPERT_HEADER_STRUCT,
    FP4_QUANT_ABI_ID,
    FP4_QUANT_PROFILE,
    HEADER_BYTES,
    MXFP6_QUANT_ABI_ID,
    PACK_ALIGNMENT,
    QUANT_ABI_ID,
)
from compiler.expert_pack.deepseek_v4 import (
    _build_expected,
    estimate_deepseek_v4_representations,
    validate_deepseek_v4_source,
)
from compiler.expert_pack.deepseek_slice import (
    _deepseek_mtp_mix_reference,
    _deepseek_mtp_partition,
)
from compiler.expert_pack.errors import AdapterError, ValidationError
from compiler.expert_pack.mistral4_nvfp4_adapter import _yarn_attention_scale
from compiler.expert_pack.quality import qualify_container_against_source
from compiler.expert_pack import quant
from compiler.expert_pack.safetensors import SafeTensorCheckpoint, TensorInfo
from compiler.expert_pack.source_inventory import group_source_tensors, inspect_source
from compiler.expert_pack.util import (
    canonical_json_bytes, load_json, sha256_bytes, sha256_file,
)
from compiler.expert_pack.util import publish_directory
from compiler.expert_pack.validate import validate_container


class Mistral4SemanticsTest(unittest.TestCase):
    def test_native_yarn_attention_scale(self) -> None:
        expected_mscale = 1.0 + 0.1 * math.log(128.0)
        expected = (128.0 ** -0.5) * expected_mscale * expected_mscale
        self.assertAlmostEqual(
            _yarn_attention_scale(128, 128.0, False), expected, places=12
        )
        self.assertAlmostEqual(
            _yarn_attention_scale(128, 128.0, True), 128.0 ** -0.5,
            places=12,
        )


def _values(name: str, count: int) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "little")
    return [((seed + index * 17) % 41 - 20) / 8.0 for index in range(count)]


def _tensor(name: str, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...], bytes]:
    count = 1
    for dimension in shape:
        count *= dimension
    values = _values(name, count)
    words = []
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        words.append(bits >> 16)
    return "BF16", shape, struct.pack(f"<{count}H", *words)


def _f32_tensor(
    name: str, shape: tuple[int, ...]
) -> tuple[str, tuple[int, ...], bytes]:
    count = 1
    for dimension in shape:
        count *= dimension
    return "F32", shape, struct.pack(f"<{count}f", *_values(name, count))


def _i64_tensor(
    name: str, shape: tuple[int, ...]
) -> tuple[str, tuple[int, ...], bytes]:
    count = math.prod(shape)
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    values = [((seed + index * 10007) << 25) | 1 for index in range(count)]
    return "I64", shape, struct.pack(f"<{count}q", *values)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def _make_fixture(root: Path, unknown: bool = False) -> dict[str, list[float]]:
    config = {
        "_name_or_path": "synthetic/olmoe",
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_act": "silu",
        "clip_qkv": None,
        "hidden_size": 4,
        "intermediate_size": 3,
        "max_position_embeddings": 16,
        "num_attention_heads": 2,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "vocab_size": 8,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 7,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (root / "chat_template.jinja").write_text(
        "{{ messages }}", encoding="utf-8"
    )
    (root / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {"type": "WordLevel", "vocab": {}}}),
        encoding="utf-8",
    )
    (root / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (8, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (8, 4),
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.post_attention_layernorm.weight": (4,),
        "model.layers.0.self_attn.q_proj.weight": (4, 4),
        "model.layers.0.self_attn.k_proj.weight": (2, 4),
        "model.layers.0.self_attn.v_proj.weight": (2, 4),
        "model.layers.0.self_attn.o_proj.weight": (4, 4),
        "model.layers.0.self_attn.q_norm.weight": (4,),
        "model.layers.0.self_attn.k_norm.weight": (4,),
        "model.layers.0.mlp.gate.weight": (2, 4),
    }
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        shapes[prefix + "gate_proj.weight"] = (3, 4)
        shapes[prefix + "up_proj.weight"] = (3, 4)
        shapes[prefix + "down_proj.weight"] = (4, 3)
    if unknown:
        shapes["unexpected.weight"] = (1,)

    names = sorted(shapes)
    split = len(names) // 2
    shards = [names[:split], names[split:]]
    weight_map: dict[str, str] = {}
    values: dict[str, list[float]] = {}
    for index, shard_names in enumerate(shards, start=1):
        shard_name = f"model-{index:05d}-of-00002.safetensors"
        tensors = {name: _tensor(name, shapes[name]) for name in shard_names}
        _write_safetensors(root / shard_name, tensors)
        for name in shard_names:
            weight_map[name] = shard_name
            count = 1
            for dimension in shapes[name]:
                count *= dimension
            values[name] = _values(name, count)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, sort_keys=True),
        encoding="utf-8",
    )
    return values


def _make_qwen3_next_fixture(root: Path, include_mtp: bool = False) -> None:
    config = {
        "_name_or_path": "synthetic/qwen3-next",
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "hidden_act": "silu",
        "hidden_size": 32,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "max_position_embeddings": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 1,
        "full_attention_interval": 2,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "partial_rotary_factor": 0.5,
        "norm_topk_prob": True,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "tie_word_embeddings": False,
        "vocab_size": 8,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 7,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    shapes = {
        "model.embed_tokens.weight": (8, 32),
        "model.norm.weight": (32,),
        "lm_head.weight": (8, 32),
        "model.layers.0.input_layernorm.weight": (32,),
        "model.layers.0.post_attention_layernorm.weight": (32,),
        "model.layers.0.linear_attn.dt_bias": (2,),
        "model.layers.0.linear_attn.A_log": (2,),
        "model.layers.0.linear_attn.conv1d.weight": (64, 1, 2),
        "model.layers.0.linear_attn.in_proj_qkvz.weight": (96, 32),
        "model.layers.0.linear_attn.in_proj_ba.weight": (4, 32),
        "model.layers.0.linear_attn.norm.weight": (16,),
        "model.layers.0.linear_attn.out_proj.weight": (32, 32),
        "model.layers.0.mlp.gate.weight": (2, 32),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": (32, 32),
        "model.layers.0.mlp.shared_expert.up_proj.weight": (32, 32),
        "model.layers.0.mlp.shared_expert.down_proj.weight": (32, 32),
        "model.layers.0.mlp.shared_expert_gate.weight": (1, 32),
    }
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        shapes[prefix + "gate_proj.weight"] = (32, 32)
        shapes[prefix + "up_proj.weight"] = (32, 32)
        shapes[prefix + "down_proj.weight"] = (32, 32)
    if include_mtp:
        shapes.update({
            "mtp.fc.weight": (32, 64),
            "mtp.pre_fc_norm_embedding.weight": (32,),
            "mtp.pre_fc_norm_hidden.weight": (32,),
            "mtp.norm.weight": (32,),
            "mtp.layers.0.input_layernorm.weight": (32,),
            "mtp.layers.0.post_attention_layernorm.weight": (32,),
            "mtp.layers.0.self_attn.q_proj.weight": (64, 32),
            "mtp.layers.0.self_attn.k_proj.weight": (16, 32),
            "mtp.layers.0.self_attn.v_proj.weight": (16, 32),
            "mtp.layers.0.self_attn.o_proj.weight": (32, 32),
            "mtp.layers.0.self_attn.q_norm.weight": (16,),
            "mtp.layers.0.self_attn.k_norm.weight": (16,),
            "mtp.layers.0.mlp.gate.weight": (2, 32),
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (32, 32),
            "mtp.layers.0.mlp.shared_expert.up_proj.weight": (32, 32),
            "mtp.layers.0.mlp.shared_expert.down_proj.weight": (32, 32),
            "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 32),
        })
        for expert in range(2):
            prefix = f"mtp.layers.0.mlp.experts.{expert}."
            shapes[prefix + "gate_proj.weight"] = (32, 32)
            shapes[prefix + "up_proj.weight"] = (32, 32)
            shapes[prefix + "down_proj.weight"] = (32, 32)
    tensors = {name: _tensor(name, shape) for name, shape in shapes.items()}
    _write_safetensors(root / "model.safetensors", tensors)


def _make_hybrid_delta_fixture(root: Path, *, moe: bool = False) -> None:
    """Build a small, topology-complete hybrid-delta checkpoint."""
    text = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "hidden_size": 32,
        "max_position_embeddings": 64,
        # Exercise the ABI's explicit output-gated head width: the query
        # projection width may exceed hidden_size and hidden_size need not be
        # divisible by the head count (as in the official 27B checkpoint).
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "num_hidden_layers": 4,
        "full_attention_interval": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "partial_rotary_factor": 0.5,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
        "vocab_size": 64,
    }
    if moe:
        text.update({
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        })
    else:
        text["intermediate_size"] = 32
    vision = {
        "model_type": "qwen3_5_moe_vision" if moe else "qwen3_5",
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_heads": 2,
        "num_position_embeddings": 16,
        "out_hidden_size": 32,
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 1,
        "spatial_merge_size": 2,
    }
    config = {
        "_name_or_path": "synthetic/qwen3.5",
        "model_type": "qwen3_5_moe" if moe else "qwen3_5",
        "architectures": [
            "Qwen3_5MoeForConditionalGeneration"
            if moe else "Qwen3_5ForConditionalGeneration"
        ],
        "language_model_only": False,
        "text_config": text,
        "vision_config": vision,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")

    hidden = int(text["hidden_size"])
    intermediate = int(
        text["moe_intermediate_size"] if moe else text["intermediate_size"]
    )
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    head_dim = int(text["head_dim"])
    key_heads = int(text["linear_num_key_heads"])
    value_heads = int(text["linear_num_value_heads"])
    key_head_dim = int(text["linear_key_head_dim"])
    value_head_dim = int(text["linear_value_head_dim"])
    key_dim = key_heads * key_head_dim
    value_dim = value_heads * value_head_dim
    conv_dim = 2 * key_dim + value_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.language_model.embed_tokens.weight": (64, hidden),
        "model.language_model.norm.weight": (hidden,),
        "lm_head.weight": (64, hidden),
    }
    for layer, layer_type in enumerate(text["layer_types"]):
        prefix = f"model.language_model.layers.{layer}."
        shapes.update({
            prefix + "input_layernorm.weight": (hidden,),
            prefix + "post_attention_layernorm.weight": (hidden,),
        })
        if moe:
            shapes.update({
                prefix + "mlp.gate.weight": (2, hidden),
                prefix + "mlp.shared_expert.gate_proj.weight":
                    (intermediate, hidden),
                prefix + "mlp.shared_expert.up_proj.weight":
                    (intermediate, hidden),
                prefix + "mlp.shared_expert.down_proj.weight":
                    (hidden, intermediate),
                prefix + "mlp.shared_expert_gate.weight": (1, hidden),
                prefix + "mlp.experts.gate_up_proj":
                    (2, 2 * intermediate, hidden),
                prefix + "mlp.experts.down_proj":
                    (2, hidden, intermediate),
            })
        else:
            shapes.update({
                prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                prefix + "mlp.up_proj.weight": (intermediate, hidden),
                prefix + "mlp.down_proj.weight": (hidden, intermediate),
            })
        if layer_type == "full_attention":
            shapes.update({
                prefix + "self_attn.q_proj.weight":
                    (2 * heads * head_dim, hidden),
                prefix + "self_attn.k_proj.weight":
                    (kv_heads * head_dim, hidden),
                prefix + "self_attn.v_proj.weight":
                    (kv_heads * head_dim, hidden),
                prefix + "self_attn.o_proj.weight":
                    (hidden, heads * head_dim),
                prefix + "self_attn.q_norm.weight": (head_dim,),
                prefix + "self_attn.k_norm.weight": (head_dim,),
            })
        else:
            shapes.update({
                prefix + "linear_attn.in_proj_qkv.weight":
                    (2 * key_dim + value_dim, hidden),
                prefix + "linear_attn.in_proj_z.weight": (value_dim, hidden),
                prefix + "linear_attn.in_proj_b.weight": (value_heads, hidden),
                prefix + "linear_attn.in_proj_a.weight": (value_heads, hidden),
                prefix + "linear_attn.conv1d.weight":
                    (conv_dim, 1, int(text["linear_conv_kernel_dim"])),
                prefix + "linear_attn.dt_bias": (value_heads,),
                prefix + "linear_attn.A_log": (value_heads,),
                prefix + "linear_attn.norm.weight": (value_head_dim,),
                prefix + "linear_attn.out_proj.weight": (hidden, value_dim),
            })

    shapes.update({
        "mtp.fc.weight": (hidden, 2 * hidden),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
        "mtp.norm.weight": (hidden,),
    })
    for layer in range(int(text["mtp_num_hidden_layers"])):
        prefix = f"mtp.layers.{layer}."
        shapes.update({
            prefix + "input_layernorm.weight": (hidden,),
            prefix + "post_attention_layernorm.weight": (hidden,),
            prefix + "self_attn.q_proj.weight":
                (2 * heads * head_dim, hidden),
            prefix + "self_attn.k_proj.weight":
                (kv_heads * head_dim, hidden),
            prefix + "self_attn.v_proj.weight":
                (kv_heads * head_dim, hidden),
            prefix + "self_attn.o_proj.weight":
                (hidden, heads * head_dim),
            prefix + "self_attn.q_norm.weight": (head_dim,),
            prefix + "self_attn.k_norm.weight": (head_dim,),
        })
        if moe:
            shapes.update({
                prefix + "mlp.gate.weight": (2, hidden),
                prefix + "mlp.shared_expert.gate_proj.weight":
                    (intermediate, hidden),
                prefix + "mlp.shared_expert.up_proj.weight":
                    (intermediate, hidden),
                prefix + "mlp.shared_expert.down_proj.weight":
                    (hidden, intermediate),
                prefix + "mlp.shared_expert_gate.weight": (1, hidden),
            })
            for expert in range(2):
                expert_prefix = prefix + f"mlp.experts.{expert}."
                shapes.update({
                    expert_prefix + "gate_proj.weight":
                        (intermediate, hidden),
                    expert_prefix + "up_proj.weight":
                        (intermediate, hidden),
                    expert_prefix + "down_proj.weight":
                        (hidden, intermediate),
                })
        else:
            shapes.update({
                prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                prefix + "mlp.up_proj.weight": (intermediate, hidden),
                prefix + "mlp.down_proj.weight": (hidden, intermediate),
            })

    vision_hidden = int(vision["hidden_size"])
    vision_intermediate = int(vision["intermediate_size"])
    visual = "model.visual."
    shapes.update({
        visual + "patch_embed.proj.weight": (
            vision_hidden,
            int(vision["in_channels"]),
            int(vision["temporal_patch_size"]),
            int(vision["patch_size"]),
            int(vision["patch_size"]),
        ),
        visual + "patch_embed.proj.bias": (vision_hidden,),
        visual + "pos_embed.weight":
            (int(vision["num_position_embeddings"]), vision_hidden),
    })
    for layer in range(int(vision["depth"])):
        prefix = f"{visual}blocks.{layer}."
        shapes.update({
            prefix + "attn.qkv.weight": (3 * vision_hidden, vision_hidden),
            prefix + "attn.qkv.bias": (3 * vision_hidden,),
            prefix + "attn.proj.weight": (vision_hidden, vision_hidden),
            prefix + "attn.proj.bias": (vision_hidden,),
            prefix + "mlp.linear_fc1.weight":
                (vision_intermediate, vision_hidden),
            prefix + "mlp.linear_fc1.bias": (vision_intermediate,),
            prefix + "mlp.linear_fc2.weight":
                (vision_hidden, vision_intermediate),
            prefix + "mlp.linear_fc2.bias": (vision_hidden,),
            prefix + "norm1.weight": (vision_hidden,),
            prefix + "norm1.bias": (vision_hidden,),
            prefix + "norm2.weight": (vision_hidden,),
            prefix + "norm2.bias": (vision_hidden,),
        })
    merged = vision_hidden * int(vision["spatial_merge_size"]) ** 2
    shapes.update({
        visual + "merger.norm.weight": (vision_hidden,),
        visual + "merger.norm.bias": (vision_hidden,),
        visual + "merger.linear_fc1.weight": (merged, merged),
        visual + "merger.linear_fc1.bias": (merged,),
        visual + "merger.linear_fc2.weight": (hidden, merged),
        visual + "merger.linear_fc2.bias": (hidden,),
    })

    tensors = {name: _tensor(name, shape) for name, shape in shapes.items()}
    _write_safetensors(root / "model.safetensors", tensors)


def _make_qwen4_exp_fixture(root: Path) -> None:
    hidden, hyper_count, lowrank = 32, 2, 32
    hyper = hidden * hyper_count
    layers, experts, expert_width, shared_width = 4, 2, 32, 32
    heads, kv_heads, head_dim = 2, 1, 32
    key_heads, value_heads, recurrent_dim = 1, 2, 32
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    text = {
        "model_type": "qwen4_exp_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "eos_token_id": 1,
        "full_attention_interval": 4,
        "hc_count": hyper_count,
        "hc_lowrank": lowrank,
        "head_dim": head_dim,
        "heads_per_ngram": 1,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "indexer_head_dim": 32,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 2,
        "layer_types": layer_types,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": recurrent_dim,
        "linear_num_key_heads": key_heads,
        "linear_num_value_heads": value_heads,
        "linear_value_head_dim": recurrent_dim,
        "make_ngram_vocab_size_divisible_by": 2,
        "max_position_embeddings": 64,
        "moe_intermediate_size": expert_width,
        "mtp_num_hidden_layers": 1,
        "ngram_size": 3,
        "ngram_vocab_size_base": 11,
        "num_attention_heads": heads,
        "num_experts": experts,
        "num_experts_per_tok": 1,
        "num_hidden_layers": layers,
        "num_key_value_heads": kv_heads,
        "norm_topk_prob": True,
        "output_gate_type": "sigmoid",
        "partial_rotary_factor": 0.5,
        "ple_conv_kernel_size": 2,
        "ple_embed_dim": hidden,
        "ple_layer_ids": [2],
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "partial_rotary_factor": 0.5,
            "rope_theta": 10_000_000.0,
        },
        "shared_expert_intermediate_size": shared_width,
        "split_ngram_parts": 2,
        "tie_word_embeddings": False,
        "vocab_size": 32,
    }
    vision = {
        "model_type": "qwen4_exp",
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_heads": 1,
        "num_position_embeddings": 4,
        "out_hidden_size": hidden,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
    }
    config = {
        "_name_or_path": "synthetic/qwen4-exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": text,
        "vision_config": vision,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")

    shapes: dict[str, tuple[int, ...]] = {
        "model.language_model.embed_tokens.weight": (32, hidden),
        "lm_head.weight": (32, hidden),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": (hyper,),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": (lowrank, hyper),
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": (hyper, lowrank),
    }
    conv_dim = 2 * key_heads * recurrent_dim + value_heads * recurrent_dim
    for layer, layer_type in enumerate(layer_types):
        prefix = f"model.language_model.layers.{layer}."
        for organ in ("attn_hyper_connection", "mlp_hyper_connection"):
            organ_prefix = prefix + organ + "."
            shapes.update({
                organ_prefix + "hc_norm.weight": (hyper,),
                organ_prefix + "input_mix_weight_down.weight": (lowrank, hyper),
                organ_prefix + "input_mix_weight_up.weight": (hyper, lowrank),
                organ_prefix + "block_inject_weight.weight": (hyper_count, hyper),
            })
        if layer_type == "linear_attention":
            linear = prefix + "linear_attn."
            shapes.update({
                linear + "in_proj_qkv.weight": (conv_dim, hidden),
                linear + "in_proj_z.weight": (value_heads * recurrent_dim, hidden),
                linear + "in_proj_b.weight": (value_heads, hidden),
                linear + "in_proj_a.weight": (value_heads, hidden),
                linear + "conv1d.weight": (conv_dim, 1, 2),
                linear + "dt_bias": (value_heads,),
                linear + "A_log": (value_heads,),
                linear + "norm.weight": (recurrent_dim,),
                linear + "out_proj.weight": (hidden, value_heads * recurrent_dim),
            })
        else:
            attention = prefix + "self_attn."
            shapes.update({
                attention + "q_proj.weight": (2 * heads * head_dim, hidden),
                attention + "k_proj.weight": (kv_heads * head_dim, hidden),
                attention + "v_proj.weight": (kv_heads * head_dim, hidden),
                attention + "o_proj.weight": (hidden, heads * head_dim),
                attention + "q_norm.weight": (head_dim,),
                attention + "k_norm.weight": (head_dim,),
                attention + "indexer.index_qk_proj.weight": (3 * 32, hidden),
                attention + "indexer.q_layernorm.weight": (32,),
                attention + "indexer.k_layernorm.weight": (32,),
            })
        mlp = prefix + "mlp."
        shapes.update({
            mlp + "gate.weight": (experts, hidden),
            mlp + "shared_expert.gate_proj.weight": (shared_width, hidden),
            mlp + "shared_expert.up_proj.weight": (shared_width, hidden),
            mlp + "shared_expert.down_proj.weight": (hidden, shared_width),
            mlp + "shared_expert_gate.weight": (1, hidden),
            mlp + "experts.gate_up_proj": (experts, 2 * expert_width, hidden),
            mlp + "experts.down_proj": (experts, hidden, expert_width),
        })
    ple = "model.language_model.layers.1.ple."
    shapes.update({
        ple + "key_proj.weight": (hyper, hidden),
        ple + "value_proj.weight": (hidden, hidden),
        ple + "norm_key.weight": (hyper,),
        ple + "norm_query.weight": (hyper,),
        ple + "norm_conv.weight": (hyper,),
        ple + "conv1d.weight": (hyper, 1, 2),
        ple + "ple_embedding.ngram_embedding.shard_0.weight": (12, 16),
        ple + "ple_embedding.ngram_embedding.shard_1.weight": (12, 16),
    })
    i64_shapes = {
        ple + "ple_embedding.layer_multipliers": (3,),
        ple + "ple_embedding.ngram_heads_vocab_sizes": (2,),
        ple + "ple_embedding.ngram_heads_offsets": (2,),
    }
    visual = "model.visual."
    shapes.update({
        visual + "patch_embed.proj.weight": (32, 3, 1, 2, 2),
        visual + "patch_embed.proj.bias": (32,),
        visual + "pos_embed.weight": (4, 32),
        visual + "blocks.0.attn.qkv.weight": (96, 32),
        visual + "blocks.0.attn.qkv.bias": (96,),
        visual + "blocks.0.attn.proj.weight": (32, 32),
        visual + "blocks.0.attn.proj.bias": (32,),
        visual + "blocks.0.mlp.linear_fc1.weight": (32, 32),
        visual + "blocks.0.mlp.linear_fc1.bias": (32,),
        visual + "blocks.0.mlp.linear_fc2.weight": (32, 32),
        visual + "blocks.0.mlp.linear_fc2.bias": (32,),
        visual + "blocks.0.norm1.weight": (32,),
        visual + "blocks.0.norm1.bias": (32,),
        visual + "blocks.0.norm2.weight": (32,),
        visual + "blocks.0.norm2.bias": (32,),
        visual + "merger.norm.weight": (32,),
        visual + "merger.norm.bias": (32,),
        visual + "merger.linear_fc1.weight": (128, 128),
        visual + "merger.linear_fc1.bias": (128,),
        visual + "merger.linear_fc2.weight": (32, 128),
        visual + "merger.linear_fc2.bias": (32,),
    })
    shapes.update({
        "mtp.fc_embedding.weight": (hidden, hidden),
        "mtp.fc_hidden.weight": (hidden, hidden),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hyper,),
        "mtp.hyper_connection_mixer.hc_norm.weight": (hyper,),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (lowrank, hyper),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (hyper, lowrank),
    })
    mtp = "mtp.layers.0."
    for organ in ("attn_hyper_connection", "mlp_hyper_connection"):
        organ_prefix = mtp + organ + "."
        shapes.update({
            organ_prefix + "hc_norm.weight": (hyper,),
            organ_prefix + "input_mix_weight_down.weight": (lowrank, hyper),
            organ_prefix + "input_mix_weight_up.weight": (hyper, lowrank),
            organ_prefix + "block_inject_weight.weight": (hyper_count, hyper),
        })
    attention = mtp + "self_attn."
    shapes.update({
        attention + "q_proj.weight": (2 * heads * head_dim, hidden),
        attention + "k_proj.weight": (kv_heads * head_dim, hidden),
        attention + "v_proj.weight": (kv_heads * head_dim, hidden),
        attention + "o_proj.weight": (hidden, heads * head_dim),
        attention + "q_norm.weight": (head_dim,),
        attention + "k_norm.weight": (head_dim,),
        attention + "indexer.index_qk_proj.weight": (3 * 32, hidden),
        attention + "indexer.q_layernorm.weight": (32,),
        attention + "indexer.k_layernorm.weight": (32,),
    })
    mtp_mlp = mtp + "mlp."
    shapes.update({
        mtp_mlp + "gate.weight": (experts, hidden),
        mtp_mlp + "shared_expert.gate_proj.weight": (shared_width, hidden),
        mtp_mlp + "shared_expert.up_proj.weight": (shared_width, hidden),
        mtp_mlp + "shared_expert.down_proj.weight": (hidden, shared_width),
        mtp_mlp + "shared_expert_gate.weight": (1, hidden),
        mtp_mlp + "experts.gate_up_proj": (experts, 2 * expert_width, hidden),
        mtp_mlp + "experts.down_proj": (experts, hidden, expert_width),
    })
    tensors = {name: _tensor(name, shape) for name, shape in shapes.items()}
    tensors.update({name: _i64_tensor(name, shape) for name, shape in i64_shapes.items()})
    _write_safetensors(root / "model.safetensors", tensors)


def _make_muse_glimmer_fixture(root: Path) -> None:
    hidden, intermediate, vocab = 32, 64, 64
    layers, heads, kv_heads, head_dim = 4, 1, 1, 32
    vision_hidden, vision_intermediate, vision_layers = 32, 64, 2
    patch, temporal, merge, projector = 2, 1, 2, 16
    config = {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "out_hidden_size": vision_hidden * merge * merge,
        "projector_hidden_act": "gelu",
        "projector_hidden_size": projector,
        "text_config": {
            "attention_bias": False,
            "final_logit_softcapping": 20.0,
            "head_dim": head_dim,
            "hidden_activation": "silu",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0],
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "max_position_embeddings": 64,
            "model_type": "muse_glimmer_text",
            "num_attention_heads": heads,
            "num_hidden_layers": layers,
            "num_key_value_heads": kv_heads,
            "output_multiplier": 0.19611613513818404,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 500000.0},
            "sliding_window": 32,
            "tie_word_embeddings": False,
            "vocab_size": vocab,
        },
        "vision_config": {
            "hidden_act": "gelu",
            "hidden_size": vision_hidden,
            "intermediate_size": vision_intermediate,
            "layer_types": ["window_attention", "full_attention"],
            "max_position_embeddings": 16,
            "merge_size": merge,
            "model_type": "muse_glimmer_vision",
            "num_attention_heads": 1,
            "num_hidden_layers": vision_layers,
            "patch_size": patch,
            "patch_temporal": temporal,
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    shapes: dict[str, tuple[int, ...]] = {
        "model.language_model.embed_tokens.weight": (vocab, hidden),
        "model.language_model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
        "model.vision_tower.patch_embedder.patch_embedding.weight":
            (vision_hidden, 3 * temporal * patch * patch),
        "model.vision_tower.patch_embedder.position_embedding_table.weight":
            (16, vision_hidden),
        "model.vision_tower.ln_pre.weight": (vision_hidden,),
        "model.vision_tower.ln_pre.bias": (vision_hidden,),
        "model.vision_tower.ln_post.weight": (vision_hidden,),
        "model.vision_tower.ln_post.bias": (vision_hidden,),
        "model.vision_adapter.fc1.weight":
            (projector, vision_hidden * merge * merge),
        "model.vision_adapter.fc2.weight": (projector, projector),
        "model.vision_projection.weight": (hidden, projector),
    }
    for layer in range(layers):
        prefix = f"model.language_model.layers.{layer}."
        for norm in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            shapes[prefix + norm] = (hidden,)
        for projection in ("q_proj", "gate_proj"):
            shapes[prefix + f"self_attn.{projection}.weight"] = (
                heads * head_dim, hidden
            )
        for projection in ("k_proj", "v_proj"):
            shapes[prefix + f"self_attn.{projection}.weight"] = (
                kv_heads * head_dim, hidden
            )
        shapes[prefix + "self_attn.o_proj.weight"] = (
            hidden, heads * head_dim
        )
        shapes[prefix + "mlp.gate_proj.weight"] = (intermediate, hidden)
        shapes[prefix + "mlp.up_proj.weight"] = (intermediate, hidden)
        shapes[prefix + "mlp.down_proj.weight"] = (hidden, intermediate)
    for layer in range(vision_layers):
        prefix = f"model.vision_tower.layers.{layer}."
        for norm in ("norm1", "norm2"):
            shapes[prefix + norm + ".weight"] = (vision_hidden,)
            shapes[prefix + norm + ".bias"] = (vision_hidden,)
        for projection in ("q_proj", "k_proj", "v_proj", "proj"):
            shapes[prefix + f"attn.{projection}.weight"] = (
                vision_hidden, vision_hidden
            )
            shapes[prefix + f"attn.{projection}.bias"] = (vision_hidden,)
        shapes[prefix + "mlp.fc1.weight"] = (
            vision_intermediate, vision_hidden
        )
        shapes[prefix + "mlp.fc1.bias"] = (vision_intermediate,)
        shapes[prefix + "mlp.fc2.weight"] = (
            vision_hidden, vision_intermediate
        )
        shapes[prefix + "mlp.fc2.bias"] = (vision_hidden,)
    _write_safetensors(
        root / "model.safetensors",
        {name: _tensor(name, shape) for name, shape in shapes.items()},
    )


def _make_lfm2_moe_fixture(root: Path, unknown: bool = False) -> None:
    config = {
        "_name_or_path": "synthetic/lfm2-moe",
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "hidden_size": 4,
        "intermediate_size": 6,
        "moe_intermediate_size": 2,
        "max_position_embeddings": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 4,
        "num_dense_layers": 1,
        "layer_types": ["conv", "full_attention", "conv", "full_attention"],
        "conv_L_cache": 3,
        "conv_bias": False,
        "norm_eps": 1e-5,
        "norm_topk_prob": True,
        "rope_theta": 1_000_000.0,
        "routed_scaling_factor": 1.0,
        "use_expert_bias": True,
        "tie_word_embeddings": True,
        "vocab_size": 8,
        "bos_token_id": 1,
        "eos_token_id": 7,
        "pad_token_id": 0,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (root / "chat_template.jinja").write_text(
        "{{ messages }}", encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (8, 4),
        "model.embedding_norm.weight": (4,),
    }
    for layer, layer_type in enumerate(config["layer_types"]):
        prefix = f"model.layers.{layer}."
        shapes[prefix + "operator_norm.weight"] = (4,)
        shapes[prefix + "ffn_norm.weight"] = (4,)
        if layer_type == "conv":
            shapes.update({
                prefix + "conv.in_proj.weight": (12, 4),
                prefix + "conv.conv.weight": (4, 1, 3),
                prefix + "conv.out_proj.weight": (4, 4),
            })
        else:
            shapes.update({
                prefix + "self_attn.q_proj.weight": (4, 4),
                prefix + "self_attn.k_proj.weight": (2, 4),
                prefix + "self_attn.v_proj.weight": (2, 4),
                prefix + "self_attn.out_proj.weight": (4, 4),
                prefix + "self_attn.q_layernorm.weight": (2,),
                prefix + "self_attn.k_layernorm.weight": (2,),
            })
        if layer == 0:
            shapes.update({
                prefix + "feed_forward.w1.weight": (6, 4),
                prefix + "feed_forward.w2.weight": (4, 6),
                prefix + "feed_forward.w3.weight": (6, 4),
            })
        else:
            shapes[prefix + "feed_forward.gate.weight"] = (2, 4)
            shapes[prefix + "feed_forward.expert_bias"] = (2,)
            for expert in range(2):
                expert_prefix = prefix + f"feed_forward.experts.{expert}."
                shapes[expert_prefix + "w1.weight"] = (2, 4)
                shapes[expert_prefix + "w2.weight"] = (4, 2)
                shapes[expert_prefix + "w3.weight"] = (2, 4)
    if unknown:
        shapes["model.unexpected.weight"] = (1,)
    tensors = {
        name: (
            _f32_tensor(name, shape)
            if name.endswith("expert_bias")
            else _tensor(name, shape)
        )
        for name, shape in shapes.items()
    }
    _write_safetensors(root / "model.safetensors", tensors)


def _deepseek_v4_config() -> dict[str, object]:
    ratios = [0, 0]
    ratios.extend(4 if layer % 2 else 128 for layer in range(1, 42))
    ratios.append(0)
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
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
        "compress_ratios": ratios,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def _deepseek_metadata_checkpoint() -> SimpleNamespace:
    config = _deepseek_v4_config()
    expected = _build_expected(config, 0)
    tensors = {}
    for name, item in expected.items():
        nbytes = 1
        for dimension in item.shape:
            nbytes *= dimension
        if item.dtype in ("BF16", "F16"):
            nbytes *= 2
        elif item.dtype == "F32":
            nbytes *= 4
        elif item.dtype in ("I64", "U64", "F64"):
            nbytes *= 8
        tensors[name] = TensorInfo(name, "synthetic.safetensors", item.dtype, item.shape, 0, nbytes)
    return SimpleNamespace(config=config, tensors=tensors)


class ExpertPackTests(unittest.TestCase):
    def test_sampling_profiles_are_artifact_declared_and_refreshable(self) -> None:
        profiles = {
            "schema": "sampling-profiles-v1",
            "profiles": {
                "thinking": {
                    "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                    "min_p": 0.0, "presence_penalty": 0.0,
                    "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                },
                "non_thinking": {
                    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "min_p": 0.0, "presence_penalty": 1.5,
                    "frequency_penalty": 0.0, "repetition_penalty": 1.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_fixture(source)
            profile_path = root / "sampling.json"
            profile_path.write_text(json.dumps(profiles), encoding="utf-8")
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source, output=output,
                sampling_profiles=profile_path,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            self.assertEqual(manifest["tokenizer"]["sampling"], profiles)

            # Exercise the exact metadata-only migration used by artifacts
            # published before auxiliary/placement accounting was mandatory.
            manifest.pop("auxiliary_tensors")
            manifest["masses"].pop("resident_dense_bytes")
            manifest["masses"].pop("host_mapped_dense_bytes")
            manifest["integrity"]["content_sha256"] = ""
            manifest["integrity"]["content_sha256"] = sha256_bytes(
                canonical_json_bytes(manifest)
            )
            (output / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) +
                "\n",
                encoding="utf-8",
            )
            report = load_json(output / "conversion-report.json")
            report["manifest_content_sha256"] = manifest["integrity"][
                "content_sha256"
            ]
            (output / "conversion-report.json").write_text(
                json.dumps(report, sort_keys=True, separators=(",", ":")) +
                "\n",
                encoding="utf-8",
            )
            (output / "COMPLETED").write_text(json.dumps({
                "format_version": 1,
                "manifest_content_sha256": manifest["integrity"][
                    "content_sha256"
                ],
                "manifest_file_sha256": sha256_file(
                    output / "manifest.json"
                ),
            }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

            updated = json.loads(json.dumps(profiles))
            updated["schema"] = "sampling-policy"
            updated["maximum_thinking_tokens"] = 32768
            updated["profiles"]["thinking"]["presence_penalty"] = 0.5
            updated["profiles"]["non_thinking"]["presence_penalty"] = 0.5
            updated_path = root / "updated-sampling.json"
            updated_path.write_text(json.dumps(updated), encoding="utf-8")
            refreshed = root / "pack-refreshed"
            refresh_result = refresh_sampling_profiles(
                output, refreshed, updated_path
            )
            self.assertTrue(refresh_result["validation"]["valid"])
            refreshed_manifest = load_json(refreshed / "manifest.json")
            self.assertEqual(
                refreshed_manifest["tokenizer"]["sampling"], updated
            )
            self.assertEqual(
                [item["sha256"] for item in refreshed_manifest["packs"]],
                [item["sha256"] for item in manifest["packs"]],
            )

    def test_numpy_fp4_row_batches_match_dependency_free_bytes(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            name = "matrix.weight"
            _write_safetensors(
                root / "model.safetensors", {name: _tensor(name, (1025, 37))}
            )
            checkpoint = SafeTensorCheckpoint(root)
            with checkpoint.open_tensor(name) as view:
                accelerated = io.BytesIO()
                accelerated_scales = quant.write_fp4_block32_rows(
                    view, accelerated, hashlib.sha256()
                )
            with checkpoint.open_tensor(name) as view, mock.patch.object(
                quant, "_np", None
            ):
                reference = io.BytesIO()
                reference_scales = quant.write_fp4_block32_rows(
                    view, reference, hashlib.sha256()
                )
            self.assertEqual(accelerated.getvalue(), reference.getvalue())
            self.assertEqual(accelerated_scales, reference_scales)

    def test_publish_directory_preserves_completed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / "artifact.partial"
            destination = root / "artifact"
            partial.mkdir()
            (partial / "manifest.json").write_text("{}\n", encoding="utf-8")
            publish_directory(partial, destination)
            self.assertFalse(partial.exists())
            self.assertEqual(
                (destination / "manifest.json").read_text(encoding="utf-8"),
                "{}\n",
            )

    def test_deepseek_v4_contract_is_byte_exact_and_fail_closed(self) -> None:
        checkpoint = _deepseek_metadata_checkpoint()
        result = validate_deepseek_v4_source(checkpoint)
        self.assertTrue(result["valid"])
        self.assertEqual(result["tensor_count"], 69187)
        self.assertEqual(result["tensor_bytes"], 159609485896)
        self.assertEqual(result["mtp_namespace"], 0)

        removed = checkpoint.tensors.pop("layers.42.ffn.experts.255.w3.scale")
        with self.assertRaisesRegex(AdapterError, "partition mismatch"):
            validate_deepseek_v4_source(checkpoint)
        checkpoint.tensors[removed.name] = removed

        name = "layers.0.ffn.experts.0.w1.weight"
        original = checkpoint.tensors[name]
        checkpoint.tensors[name] = TensorInfo(
            original.name, original.shard, original.dtype, (2048, 2047), original.offset,
            original.nbytes,
        )
        with self.assertRaisesRegex(AdapterError, "metadata mismatch"):
            validate_deepseek_v4_source(checkpoint)

    def test_deepseek_v4_representation_estimate_is_metadata_only(self) -> None:
        checkpoint = _deepseek_metadata_checkpoint()
        estimate = estimate_deepseek_v4_representations(checkpoint)
        self.assertTrue(estimate["payload_only"])
        self.assertEqual(estimate["source_checkpoint"]["bytes"], 159609485896)
        self.assertEqual(
            estimate["source_checkpoint"]["routed_expert_bytes"], 150592290816
        )
        self.assertEqual(estimate["eager_fp8_routed_experts"]["bytes"], 292502338120)
        self.assertEqual(
            estimate["eager_int8_per_row_routed_experts"]["bytes"], 292854135368
        )
        cache = estimate["hot_int8_cache"]
        self.assertEqual(cache["source_bytes_per_expert"], 13369344)
        self.assertEqual(cache["compute_bytes_per_expert"], 25198592)
        self.assertEqual(cache["compute_bytes_for_top_k_one_layer"], 151191552)
        self.assertEqual(cache["experts_per_gib"], 42)

    def test_deepseek_mtp_partition_is_exact_and_metadata_only(self) -> None:
        checkpoint = _deepseek_metadata_checkpoint()
        namespace, typed, dense, shared = _deepseek_mtp_partition(checkpoint)
        self.assertEqual(namespace, 0)
        self.assertEqual(len(typed), 19)
        self.assertEqual(len(dense), 7)
        self.assertEqual(len(shared), 6)
        self.assertTrue(all(name.startswith("mtp.0.") for name in typed))
        self.assertTrue(all(name.startswith("mtp.0.") for name in dense))
        self.assertEqual(len(set(typed)), len(typed))
        self.assertEqual(len(set(dense)), len(dense))

    def test_deepseek_mtp_mix_normalizes_last_dimension_and_broadcasts(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required")
        previous = np.asarray(
            [[1.0, 2.0], [2.0, 1.0], [-1.0, 2.0], [3.0, -2.0]],
            dtype=np.float32,
        )
        embedding = np.asarray([2.0, -1.0], dtype=np.float32)
        enorm = np.asarray([1.5, 0.5], dtype=np.float32)
        hnorm = np.asarray([0.25, 2.0], dtype=np.float32)
        matrix_e = np.asarray([[2.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        matrix_h = np.asarray([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32)
        normalized_token, normalized_streams, mixed = \
            _deepseek_mtp_mix_reference(
                previous, embedding, enorm, hnorm,
                lambda value: matrix_e @ value,
                lambda value: matrix_h @ value,
            )
        token_inverse = 1.0 / np.sqrt(np.mean(embedding * embedding) + 1e-6)
        expected_token = embedding * token_inverse * enorm
        stream_inverse = 1.0 / np.sqrt(
            np.mean(previous * previous, axis=1) + 1e-6
        )
        expected_streams = previous * stream_inverse[:, None] * hnorm[None, :]
        expected_mixed = expected_streams @ matrix_h.T + \
            (matrix_e @ expected_token)[None, :]
        np.testing.assert_allclose(normalized_token, expected_token, rtol=1e-6)
        np.testing.assert_allclose(normalized_streams, expected_streams, rtol=1e-6)
        np.testing.assert_allclose(mixed, expected_mixed, rtol=1e-6)

    def test_source_inventory_accepts_float8_metadata_without_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps({
                    "architectures": ["SyntheticForCausalLM"],
                    "model_type": "synthetic",
                }),
                encoding="utf-8",
            )
            _write_safetensors(
                root / "model.safetensors",
                {
                    "weight": ("F8_E4M3", (2, 3), bytes(range(6))),
                    "scale": ("F8_E8M0", (2,), bytes((127, 128))),
                },
            )
            inventory = inspect_source(SafeTensorCheckpoint(root))
            self.assertEqual(inventory["tensor_count"], 2)
            self.assertEqual(inventory["tensor_bytes"], 8)
            self.assertEqual(
                inventory["dtypes"],
                {
                    "F8_E4M3": {"tensor_count": 1, "bytes": 6},
                    "F8_E8M0": {"tensor_count": 1, "bytes": 2},
                },
            )
            groups = group_source_tensors(SafeTensorCheckpoint(root))
            self.assertEqual([group["pattern"] for group in groups], ["scale", "weight"])

    def test_source_inventory_groups_numeric_tensor_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps({"model_type": "synthetic"}), encoding="utf-8"
            )
            _write_safetensors(
                root / "model.safetensors",
                {
                    "layers.0.experts.17.w1.weight": ("I8", (2, 4), bytes(8)),
                    "layers.1.experts.2.w1.weight": ("I8", (2, 4), bytes(8)),
                    "layers.1.ffn.gate.tid0eid": ("I64", (2,), bytes(16)),
                },
            )
            groups = group_source_tensors(SafeTensorCheckpoint(root))
            by_pattern = {group["pattern"]: group for group in groups}
            experts = by_pattern["layers.{n}.experts.{n}.w1.weight"]
            self.assertEqual(experts["tensor_count"], 2)
            self.assertEqual(experts["bytes"], 16)
            self.assertEqual(
                by_pattern["layers.{n}.ffn.gate.tid{n}eid"]["tensor_count"], 1
            )

    def test_qwen3_next_adapter_and_rank3_dense_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_qwen3_next_fixture(source)
            adapted = adapt_checkpoint(SafeTensorCheckpoint(source), "qwen3_next")
            self.assertEqual(adapted.architecture["family"], "qwen3_next")
            self.assertEqual(len(adapted.experts), 2)
            rejected_output = root / "rejected-pack"
            with self.assertRaisesRegex(ValueError, "does not support"):
                compile_checkpoint(CompileOptions(
                    source=source,
                    output=rejected_output,
                    adapter="qwen3_next",
                    max_expert_pack_bytes=PACK_ALIGNMENT,
                ))
            self.assertFalse(rejected_output.exists())
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source, output=output, adapter="qwen3_next",
                quant_profile=FP4_QUANT_PROFILE,
                max_expert_pack_bytes=PACK_ALIGNMENT,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            model_program = (output / manifest["model_program"]["path"]).read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "component\tdecoder\t0\t1\t2\t1\t1\t32\t32",
                model_program,
            )
            self.assertIn(
                "layer\t0\tblock.recurrent-linear-attention.gated-delta.v1",
                model_program,
            )
            self.assertIn(
                "router\tdecoder\trouter.linear-topk.shared-swiglu.v1\t1",
                model_program,
            )
            self.assertIn(
                "operation_tensor\t1\tqkvz_projection\t"
                "model.layers.0.linear_attn.in_proj_qkvz.weight",
                model_program,
            )
            self.assertIn(
                "operation_tensor\t2\tshared_down_projection\t"
                "model.layers.0.mlp.shared_expert.down_proj.weight",
                model_program,
            )
            self.assertIn(
                "operation\t3\t0\tmoe.swiglu.routed.merge-shared.v1\t1\tdecoder\t0",
                model_program,
            )
            conv = next(item for item in manifest["tensors"] if item["name"].endswith("conv1d.weight"))
            self.assertEqual(conv["source_shape"], [64, 1, 2])
            self.assertEqual(conv["stored_dtype"], "F32")
            router = next(
                item for item in manifest["tensors"]
                if item["name"] == "model.layers.0.mlp.gate.weight"
            )
            self.assertEqual(router["stored_dtype"], "F32")
            self.assertEqual(
                manifest["quantization"]["abi_id"], FP4_QUANT_ABI_ID
            )

    def test_qwen3_next_optional_mtp_is_explicitly_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_qwen3_next_fixture(source, include_mtp=True)
            adapted = adapt_checkpoint(SafeTensorCheckpoint(source), "qwen3_next")
            self.assertEqual(adapted.architecture["multi_token_prediction_layers"], 1)
            self.assertIn("mtp.fc.weight", {tensor.name for tensor in adapted.dense})
            self.assertEqual(adapted.source_tensor_count,
                             len(adapted.dense) + 3 * len(adapted.experts))

    def test_muse_glimmer_reuses_dense_fp4_abi2_with_windowed_kv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            _make_muse_glimmer_fixture(source)
            checkpoint = SafeTensorCheckpoint(source)
            adapted = adapt_checkpoint(checkpoint, "muse_glimmer")
            self.assertEqual(adapted.family, "muse_glimmer")
            self.assertEqual(adapted.runtime_topology.components, ())
            self.assertEqual(len(adapted.runtime_topology.layers), 4)
            self.assertEqual(
                [layer.block_abi for layer in adapted.runtime_topology.layers],
                [2, 2, 2, 2],
            )
            self.assertIn(
                "model.vision_tower.layers.0.attn.q_proj.weight",
                adapted.auxiliary_dense,
            )
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                adapter="muse_glimmer",
                quant_profile=FP4_QUANT_PROFILE,
                max_expert_pack_bytes=PACK_ALIGNMENT,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            self.assertTrue(manifest["architecture"]["vision_auxiliary_only"])
            program = (output / "runtime-model.tsv").read_text(encoding="utf-8")
            self.assertIn(
                "kernel\tblock.full-attention.output-gated.v1\t2", program
            )
            self.assertIn(
                "operation_parameter\t1\tattention_window_tokens\t32",
                program,
            )
            self.assertIn(
                "operation_tensor\t1\tgate_projection\t"
                "model.language_model.layers.0.self_attn.gate_proj.weight",
                program,
            )
            self.assertIn(
                "operation_parameter\t9\tlogit_softcap_f32_bits", program
            )

    def test_hybrid_delta_dense_fp4_schema3_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)

            checkpoint = SafeTensorCheckpoint(source)
            adapted = adapt_checkpoint(checkpoint, "hybrid_delta")
            self.assertEqual(
                adapted.architecture["family"], "hybrid_delta_dense"
            )
            self.assertEqual(
                dict(adapted.runtime_topology.attributes)[
                    "minimum_exact_kv_bytes_per_token"
                ],
                2 * 1 * 16 * 2,
            )
            self.assertEqual(adapted.architecture["num_hidden_layers"], 4)
            self.assertEqual(adapted.source_tensor_count, len(adapted.dense))
            self.assertEqual(adapted.experts, ())
            self.assertEqual(adapted.runtime_topology.components, ())
            self.assertEqual(len(adapted.runtime_topology.layers), 4)
            self.assertEqual(len(adapted.runtime_topology.operations), 11)
            self.assertIsNotNone(adapted.runtime_topology.exact_decode)
            self.assertEqual(
                adapted.runtime_topology.exact_decode.maximum_emitted_tokens,
                5,
            )
            exact_parameters = dict(
                adapted.runtime_topology.exact_decode.parameters
            )
            self.assertEqual(exact_parameters["source_mtp_layers"], 1)
            self.assertEqual(exact_parameters["draft_depth"], 4)
            self.assertEqual(exact_parameters["draft_vocabulary_size"], 64)
            self.assertEqual(exact_parameters["mtp_kv_encoding"], 1)
            self.assertIn(
                "model.visual.patch_embed.proj.weight", adapted.dense_fp4
            )
            self.assertEqual(adapted.dense_mxfp6, frozenset())
            self.assertIn(
                "model.language_model.layers.0.linear_attn.A_log",
                adapted.dense_float32,
            )
            rejected = root / "rejected-pack"
            with self.assertRaisesRegex(ValueError, "does not support"):
                compile_checkpoint(CompileOptions(
                    source=source,
                    output=rejected,
                    adapter="hybrid_delta",
                ))
            self.assertFalse(rejected.exists())

            mxfp6_names = frozenset({
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
            })
            mxfp6_operations = tuple(
                replace(
                    operation,
                    capability=(
                        "embedding.lookup.mxfp6-e3m2-block32.v1"
                        if operation.capability ==
                           "embedding.lookup.fp4-block32.v1"
                        else "head.rmsnorm.token-select."
                             "mxfp6-e3m2-block32.v1"
                        if operation.capability ==
                           "head.rmsnorm.token-select.fp4-block32.v1"
                        else operation.capability
                    ),
                )
                for operation in adapted.runtime_topology.operations
            )
            mxfp6_required = tuple(
                (
                    "embedding.lookup.mxfp6-e3m2-block32.v1"
                    if capability == "embedding.lookup.fp4-block32.v1"
                    else "head.rmsnorm.token-select."
                         "mxfp6-e3m2-block32.v1"
                    if capability ==
                       "head.rmsnorm.token-select.fp4-block32.v1"
                    else "decode.mtp.dense-full-attention."
                         "fp4-mxfp6-io.exact.v3"
                    if capability ==
                       "decode.mtp.dense-full-attention."
                       "fp4-block32.exact.v2"
                    else capability,
                    abi,
                )
                for capability, abi in adapted.runtime_topology.required_kernels
            )
            mxfp6_adapted = replace(
                adapted,
                dense_mxfp6=mxfp6_names,
                runtime_topology=replace(
                    adapted.runtime_topology,
                    operations=mxfp6_operations,
                    required_kernels=mxfp6_required,
                    exact_decode=replace(
                        adapted.runtime_topology.exact_decode,
                        capability="decode.mtp.dense-full-attention."
                                   "fp4-mxfp6-io.exact.v3",
                    ),
                ),
            )
            output = root / "pack"
            with mock.patch(
                "compiler.expert_pack.compile.adapt_checkpoint",
                return_value=mxfp6_adapted,
            ):
                result = compile_checkpoint(CompileOptions(
                    source=source,
                    output=output,
                    adapter="hybrid_delta",
                    quant_profile=FP4_QUANT_PROFILE,
                    max_expert_pack_bytes=PACK_ALIGNMENT,
                    source_revision=source.name,
                ))
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(result["validation"]["experts"], 0)
            manifest = load_json(output / "manifest.json")
            self.assertEqual(manifest["experts"], [])
            self.assertEqual(manifest["quantization"]["expert_weights"], "none")
            self.assertEqual(
                manifest["quantization"]["dense_matrix_weights"],
                "mixed-mxfp6-fp4-and-other-quantized-tensors",
            )
            self.assertEqual(
                manifest["quantization"]["dense_tensor_quant_abis"],
                [0, FP4_QUANT_ABI_ID, MXFP6_QUANT_ABI_ID],
            )
            dense_by_name = {entry["name"]: entry for entry in manifest["tensors"]}
            for name in adapted.dense_fp4 - mxfp6_names:
                self.assertEqual(dense_by_name[name]["stored_dtype"], "FP4_E2M1")
                self.assertEqual(
                    dense_by_name[name]["quant_abi"], FP4_QUANT_ABI_ID
                )
            for name in mxfp6_names:
                self.assertEqual(
                    dense_by_name[name]["stored_dtype"], "MXFP6_E3M2"
                )
                self.assertEqual(
                    dense_by_name[name]["quant_abi"], MXFP6_QUANT_ABI_ID
                )
            for name in adapted.dense_float32:
                self.assertEqual(dense_by_name[name]["stored_dtype"], "F32")
                self.assertEqual(dense_by_name[name]["quant_abi"], 0)
            quality = qualify_container_against_source(
                source, output, samples_per_tensor=3,
                maximum_relative_l2=0.30, minimum_cosine=0.90,
            )
            self.assertTrue(quality["valid"])
            self.assertEqual(quality["records"]["tensor_mxfp6"], 2)
            self.assertEqual(
                quality["sampled"]["mxfp6_payload_mismatches"], 0
            )
            self.assertEqual(
                quality["sampled"]["mxfp6_scale_mismatches"], 0
            )
            self.assertEqual(quality["sampled"]["f32_value_mismatches"], 0)
            patch = dense_by_name["model.visual.patch_embed.proj.weight"]
            self.assertEqual(patch["source_shape"], [32, 3, 1, 2, 2])
            self.assertEqual(patch["sections"]["data"]["bytes"], 3072)
            self.assertEqual(patch["sections"]["scales"]["bytes"], 192)

            program = (output / manifest["model_program"]["path"]).read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "model\t3\thybrid_delta_dense\t64\t64\t32", program
            )
            self.assertNotIn("\ncomponent\t", program)
            self.assertIn(
                "layer\t3\tblock.full-attention.output-gated.v1\t1\t-\t0",
                program,
            )
            self.assertIn(
                "block.recurrent-linear-attention.split-gated-delta.v1",
                program,
            )
            self.assertIn("ffn.swiglu.dense.fp4-block32.v1", program)
            self.assertIn(
                "vision.patch-transformer-merge.fp4-block32.v1", program
            )
            self.assertIn(
                "input\tmultimodal\trequest.multimodal\t"
                "request.multimodal.fp32.host.v1",
                program,
            )
            self.assertIn(
                "exact_decode\t"
                "decode.mtp.dense-full-attention."
                "fp4-mxfp6-io.exact.v3\t2\t5",
                program,
            )
            self.assertIn(
                "exact_decode_tensor\tfusion_projection\tmtp.fc.weight",
                program,
            )
            self.assertIn(
                "exact_decode_tensor\tlayer.0.query_projection\t"
                "mtp.layers.0.self_attn.q_proj.weight",
                program,
            )

    def test_dense_mtp_program_only_upgrade_is_transactional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)
            current = root / "current"
            compile_checkpoint(CompileOptions(
                source=source,
                output=current,
                adapter="hybrid_delta",
                quant_profile=FP4_QUANT_PROFILE,
                max_expert_pack_bytes=PACK_ALIGNMENT,
            ))

            legacy = root / "legacy"
            shutil.copytree(current, legacy)
            legacy_program_path = legacy / "runtime-model.tsv"
            lines = legacy_program_path.read_text(encoding="utf-8").splitlines()
            legacy_lines: list[str] = []
            for line in lines:
                if line == (
                    "kernel\tdecode.mtp.dense-full-attention.fp4-block32."
                    "exact.v2\t2"
                ):
                    legacy_lines.append(
                        "kernel\tdecode.mtp.dense-full-attention.fp4-block32."
                        "exact.v1\t1"
                    )
                elif line == (
                    "exact_decode\tdecode.mtp.dense-full-attention."
                    "fp4-block32.exact.v2\t2\t5"
                ):
                    legacy_lines.append(
                        "exact_decode\tdecode.mtp.dense-full-attention."
                        "fp4-block32.exact.v1\t1\t2"
                    )
                elif line == "exact_decode_parameter\tsource_mtp_layers\t1":
                    legacy_lines.append(
                        "exact_decode_parameter\tdraft_layers\t1"
                    )
                elif line.startswith((
                    "exact_decode_parameter\tdraft_depth\t",
                    "exact_decode_parameter\tdraft_vocabulary_size\t",
                    "exact_decode_parameter\tmtp_kv_encoding\t",
                )):
                    continue
                else:
                    legacy_lines.append(line)
            legacy_payload = ("\n".join(legacy_lines) + "\n").encode("utf-8")
            legacy_program_path.write_bytes(legacy_payload)
            legacy_manifest = load_json(legacy / "manifest.json")
            legacy_manifest["model_program"]["bytes"] = len(legacy_payload)
            legacy_manifest["model_program"]["sha256"] = sha256_bytes(
                legacy_payload
            )
            legacy_manifest["integrity"]["content_sha256"] = ""
            legacy_manifest["integrity"]["content_sha256"] = sha256_bytes(
                canonical_json_bytes(legacy_manifest)
            )
            (legacy / "manifest.json").write_text(
                json.dumps(legacy_manifest, ensure_ascii=False, sort_keys=True,
                           indent=2) + "\n",
                encoding="utf-8",
            )
            legacy_report = load_json(legacy / "conversion-report.json")
            legacy_report["manifest_content_sha256"] = legacy_manifest[
                "integrity"
            ]["content_sha256"]
            (legacy / "conversion-report.json").write_text(
                json.dumps(legacy_report, ensure_ascii=False, sort_keys=True,
                           indent=2) + "\n",
                encoding="utf-8",
            )
            (legacy / "COMPLETED").write_text(json.dumps({
                "format_version": legacy_manifest["format"]["version"],
                "manifest_content_sha256": legacy_manifest["integrity"][
                    "content_sha256"
                ],
                "manifest_file_sha256": sha256_file(legacy / "manifest.json"),
            }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            self.assertTrue(validate_container(legacy)["valid"])

            original_manifest = load_json(legacy / "manifest.json")
            original_program = legacy_program_path.read_bytes()
            migrated = root / "migrated"
            result = upgrade_dense_mtp_program(
                legacy,
                migrated,
                draft_depth=4,
                draft_vocabulary_size=64,
            )
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(legacy_program_path.read_bytes(), original_program)
            upgraded = (migrated / "runtime-model.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "exact_decode\tdecode.mtp.dense-full-attention."
                "fp4-block32.exact.v2\t2\t5",
                upgraded,
            )
            self.assertIn(
                "exact_decode_parameter\tdraft_vocabulary_size\t64", upgraded
            )
            migrated_manifest = load_json(migrated / "manifest.json")
            self.assertEqual(
                migrated_manifest["packs"], original_manifest["packs"]
            )
            self.assertEqual(
                migrated_manifest["tokenizer"]["files"],
                original_manifest["tokenizer"]["files"],
            )
            with self.assertRaisesRegex(ValueError, "already declares"):
                upgrade_dense_mtp_program(
                    migrated,
                    root / "invalid-second-upgrade",
                    draft_depth=4,
                    draft_vocabulary_size=64,
                )


    def test_qwen4_exp_publishes_generic_hyper_qsa_ple_program(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_qwen4_exp_fixture(source)
            adapted = adapt_checkpoint(SafeTensorCheckpoint(source), "qwen4_exp")
            self.assertEqual(adapted.runtime_topology.architecture_id,
                             "hybrid-hyper-qsa-ple-moe-v1")
            self.assertEqual(len(adapted.experts), 8)
            self.assertEqual(len(adapted.dense_int64), 3)
            self.assertEqual(len(adapted.host_mapped_dense), 5)
            self.assertNotIn(
                "model.language_model.embed_tokens.weight",
                adapted.dense_fp4,
            )
            self.assertNotIn(
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                adapted.dense_fp4,
            )
            self.assertIn(
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                adapted.dense_float32,
            )
            self.assertIn(
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                adapted.dense_float32,
            )
            self.assertIn(
                "model.language_model.layers.0.attn_hyper_connection."
                "block_inject_weight.weight",
                adapted.dense_float32,
            )
            self.assertIn(
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                adapted.dense_fp4,
            )
            self.assertEqual(
                dict(adapted.runtime_topology.attributes)[
                    "minimum_exact_kv_bytes_per_token"
                ],
                (2 * 1 * 32 + 1 * 32) * 2,
            )
            self.assertIn(
                "block.sparse-attention.qsa.output-gated.v1",
                {operation.capability
                 for operation in adapted.runtime_topology.operations},
            )
            recurrent = [
                operation for operation in adapted.runtime_topology.operations
                if operation.capability ==
                "block.recurrent-linear-attention."
                "split-gated-delta.no-residual.v1"
            ]
            self.assertEqual(len(recurrent), 3)
            self.assertTrue(all(operation.abi == 2 for operation in recurrent))
            self.assertTrue(all(
                dict(operation.parameters)["output_gate_activation"] == 2
                for operation in recurrent
            ))
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source, output=output, adapter="qwen4_exp",
                quant_profile=FP4_QUANT_PROFILE,
                max_expert_pack_bytes=PACK_ALIGNMENT,
                source_revision="source",
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            by_name = {entry["name"]: entry for entry in manifest["tensors"]}
            self.assertEqual(
                manifest["quantization"]["dense_matrix_weights"],
                "mixed-fp4-block32-and-int8-by-tensor-index",
            )
            self.assertEqual(
                by_name["model.language_model.embed_tokens.weight"][
                    "stored_dtype"
                ],
                "I8",
            )
            self.assertEqual(
                by_name[
                    "model.language_model.layers.0.linear_attn."
                    "in_proj_qkv.weight"
                ]["stored_dtype"],
                "I8",
            )
            self.assertEqual(
                by_name[
                    "model.language_model.layers.0.linear_attn."
                    "in_proj_a.weight"
                ]["stored_dtype"],
                "F32",
            )
            self.assertEqual(
                by_name[
                    "model.language_model.layers.0.attn_hyper_connection."
                    "block_inject_weight.weight"
                ]["stored_dtype"],
                "F32",
            )
            ple_shard = next(
                name for name in adapted.host_mapped_dense
                if "ngram_embedding.shard_" in name
            )
            self.assertEqual(by_name[ple_shard]["stored_dtype"], "I8")
            multiplier = next(name for name in adapted.dense_int64
                              if name.endswith("layer_multipliers"))
            self.assertEqual(by_name[multiplier]["stored_dtype"], "I64")
            self.assertEqual(by_name[multiplier]["decoded_bytes"], 24)
            self.assertGreater(manifest["masses"]["host_mapped_dense_bytes"], 0)
            program = (output / "runtime-model.tsv").read_text(encoding="utf-8")
            self.assertIn("state.hyper-connection.initialize.v1", program)
            self.assertIn("embedding.lookup.v1", program)
            self.assertIn("embedding.ngram-ple.v1", program)
            self.assertIn("head.token-select.no-norm.v1", program)
            self.assertNotIn("embedding.ngram-ple.fp4-block32.v1", program)
            self.assertIn("router.linear-topk.shared-swiglu.no-residual.v1", program)
            self.assertIn("\toutput_gate_activation\t2", program)
            self.assertIn(
                "attribute\tminimum_exact_kv_bytes_per_token\t192", program
            )
            self.assertNotIn("qwen4_exp", program)

    def test_hybrid_delta_moe_fused_experts_are_logical_fp4_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source, moe=True)

            checkpoint = SafeTensorCheckpoint(source)
            adapted = adapt_checkpoint(checkpoint, "hybrid_delta")
            self.assertEqual(adapted.family, "hybrid_delta_moe")
            self.assertEqual(len(adapted.experts), 8)
            self.assertEqual(len(adapted.runtime_topology.components), 1)
            self.assertIsNone(adapted.runtime_topology.exact_decode)
            self.assertEqual(len(adapted.auxiliary_dense), 23)
            self.assertEqual(
                adapted.runtime_topology.components[0].execution_capability,
                "moe.swiglu.routed.merge-shared.v1",
            )
            shared_router = (
                "model.language_model.layers.0.mlp."
                "shared_expert_gate.weight"
            )
            self.assertIn(shared_router, adapted.dense_float32)
            self.assertNotIn(shared_router, adapted.dense_fp4)
            first, second = adapted.experts[:2]
            self.assertEqual(first.gate.physical_name,
                "model.language_model.layers.0.mlp.experts.gate_up_proj")
            self.assertEqual(first.gate.source_byte_offset, 0)
            self.assertEqual(first.up.source_byte_offset, 2048)
            self.assertEqual(second.gate.source_byte_offset, 4096)
            self.assertEqual(first.down.source_byte_offset, 0)
            self.assertEqual(second.down.source_byte_offset, 2048)

            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                adapter="hybrid_delta",
                quant_profile=FP4_QUANT_PROFILE,
                max_expert_pack_bytes=PACK_ALIGNMENT,
                source_revision="source",
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            self.assertEqual(
                manifest["architecture"]["family"], "hybrid_delta_moe"
            )
            self.assertEqual(
                manifest["architecture"]["upstream_model_type"],
                "qwen3_5_moe",
            )
            self.assertEqual(len(manifest["experts"]), 8)
            self.assertEqual(
                manifest["source"]["tensor_count"], len(checkpoint.tensors)
            )
            self.assertEqual(
                set(manifest["auxiliary_tensors"]), adapted.auxiliary_dense
            )
            first_entry = manifest["experts"][0]
            self.assertEqual(first_entry["stored_dtype"], "FP4_E2M1")
            dense_by_name = {
                entry["name"]: entry for entry in manifest["tensors"]
            }
            self.assertEqual(
                dense_by_name[shared_router]["stored_dtype"], "F32"
            )
            self.assertEqual(
                first_entry["source_regions"]["gate"],
                {
                    "tensor": (
                        "model.language_model.layers.0.mlp.experts."
                        "gate_up_proj"
                    ),
                    "byte_offset": 0,
                    "bytes": 2048,
                    "shape": [32, 32],
                    "tensor_bytes": 8192,
                    "tensor_shape": [2, 64, 32],
                },
            )
            dense_by_name = {
                entry["name"]: entry for entry in manifest["tensors"]
            }
            self.assertEqual(
                dense_by_name[
                    "model.language_model.layers.0.mlp.gate.weight"
                ]["stored_dtype"],
                "F32",
            )
            self.assertEqual(
                dense_by_name[
                    "model.language_model.layers.0.mlp.shared_expert."
                    "gate_proj.weight"
                ]["stored_dtype"],
                "FP4_E2M1",
            )
            program = (output / manifest["model_program"]["path"]).read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "component\tdecoder\t0\t4\t2\t1\t1\t32\t32\t",
                program,
            )
            self.assertIn(
                "block.recurrent-linear-attention.split-gated-delta.v1",
                program,
            )
            self.assertIn("router.linear-topk.shared-swiglu.v1", program)
            self.assertIn("moe.swiglu.routed.merge-shared.v1", program)
            self.assertNotIn("ffn.swiglu.dense.fp4-block32.v1", program)
            self.assertNotIn("exact_decode\t", program)
            quality = qualify_container_against_source(source, output)
            self.assertTrue(quality["valid"])
            self.assertEqual(quality["sampled"]["f32_value_mismatches"], 0)

    def test_lfm2_moe_adapter_compiles_dense_prefix_and_local_expert_layers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_lfm2_moe_fixture(source)
            adapted = adapt_checkpoint(
                SafeTensorCheckpoint(source), "lfm2_moe"
            )
            self.assertEqual(adapted.architecture["family"], "lfm2_moe")
            self.assertEqual(adapted.architecture["intermediate_size"], 6)
            self.assertEqual(adapted.architecture["moe_intermediate_size"], 2)
            self.assertEqual(len(adapted.dense), 37)
            self.assertEqual(len(adapted.experts), 6)
            self.assertEqual(
                [(item.layer, item.expert) for item in adapted.experts],
                [(layer, expert) for layer in range(3) for expert in range(2)],
            )
            first = adapted.experts[0]
            self.assertTrue(first.gate.name.endswith("experts.0.w1.weight"))
            self.assertTrue(first.up.name.endswith("experts.0.w3.weight"))
            self.assertTrue(first.down.name.endswith("experts.0.w2.weight"))
            topology = adapted.runtime_topology
            self.assertEqual(topology.components[0].layer_count, 3)
            self.assertIsNone(topology.layers[0].routed_component)
            self.assertEqual(topology.layers[1].component_layer, 0)
            self.assertEqual(len(topology.operations), 13)
            self.assertEqual(
                topology.operations[2].capability, "ffn.swiglu.dense.v1"
            )
            self.assertEqual(
                topology.operations[4].capability,
                "router.sigmoid-bias.topk.v1",
            )

            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                adapter="lfm2_moe",
                max_expert_pack_bytes=PACK_ALIGNMENT,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            self.assertTrue((output / "tokenizer/chat_template.jinja").is_file())
            self.assertIn(
                "tokenizer/chat_template.jinja",
                {item["path"] for item in manifest["tokenizer"]["files"]},
            )
            self.assertEqual(len(manifest["experts"]), 6)
            self.assertEqual(
                {(item["layer"], item["expert"]) for item in manifest["experts"]},
                {(layer, expert) for layer in range(3) for expert in range(2)},
            )
            self.assertEqual(
                manifest["masses"]["active_expert_bytes_per_token"],
                3 * PACK_ALIGNMENT,
            )
            first_entry = manifest["experts"][0]
            self.assertTrue(
                first_entry["source_tensors"]["gate"].endswith("w1.weight")
            )
            self.assertTrue(
                first_entry["source_tensors"]["up"].endswith("w3.weight")
            )
            self.assertTrue(
                first_entry["source_tensors"]["down"].endswith("w2.weight")
            )
            dense_by_name = {
                item["name"]: item for item in manifest["tensors"]
            }
            self.assertEqual(
                dense_by_name[
                    "model.layers.1.feed_forward.gate.weight"
                ]["stored_dtype"],
                "F32",
            )
            self.assertEqual(
                dense_by_name[
                    "model.layers.1.self_attn.q_proj.weight"
                ]["stored_dtype"],
                "I8",
            )
            program = (output / manifest["model_program"]["path"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("component\tdecoder\t0\t3\t2\t1\t0\t4\t2", program)
            self.assertIn(
                "layer\t0\tblock.causal-short-conv.gated.v1\t1\t-\t0",
                program,
            )
            self.assertIn(
                "layer\t1\tblock.full-attention.gqa.qk-norm.v1\t1\tdecoder\t0",
                program,
            )
            self.assertIn(
                "operation\t2\t0\tffn.swiglu.dense.v1\t1\t-\t0",
                program,
            )
            self.assertIn(
                "model_tensor\ttoken_embedding\tmodel.embed_tokens.weight",
                program,
            )
            self.assertIn(
                "operation_tensor\t1\tinput_norm\t"
                "model.layers.0.operator_norm.weight",
                program,
            )
            self.assertIn(
                "operation_tensor\t4\texpert_bias\t"
                "model.layers.1.feed_forward.expert_bias",
                program,
            )
            refreshed = root / "pack-refreshed"
            source_names = SafeTensorCheckpoint(source).tensors
            (source / "consolidated.safetensors.index.json").write_text(
                json.dumps({
                    "metadata": {},
                    "weight_map": {
                        name: "model.safetensors" for name in source_names
                    },
                }, sort_keys=True),
                encoding="utf-8",
            )
            (source / "config.json").rename(source / "params.json")
            refresh_result = refresh_runtime_model_program(
                output, refreshed, source, "lfm2_moe",
                config_file="params.json",
                index_file="consolidated.safetensors.index.json",
                dense_activation_input="bf16",
            )
            self.assertTrue(refresh_result["validation"]["valid"])
            refreshed_manifest = load_json(refreshed / "manifest.json")
            self.assertTrue(
                (refreshed / "tokenizer/chat_template.jinja").is_file()
            )
            self.assertEqual(
                [item["sha256"] for item in refreshed_manifest["packs"]],
                [item["sha256"] for item in manifest["packs"]],
            )
            refreshed_program = (refreshed / "runtime-model.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "attribute\tdense_activation_input_bf16\t1\n",
                refreshed_program,
            )
            self.assertIn(
                "kernel\tdense.activation-input.bfloat16.v1\t1\n",
                refreshed_program,
            )
            self.assertEqual(
                refreshed_manifest["quantization"]["dense_activation_input"],
                "bf16",
            )

    def test_lfm2_moe_adapter_rejects_unidentified_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_lfm2_moe_fixture(root, unknown=True)
            with self.assertRaisesRegex(AdapterError, "unidentified LFM2-MoE"):
                adapt_checkpoint(SafeTensorCheckpoint(root), "lfm2_moe")

    def test_checkpoint_reader_and_strict_olmoe_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root)
            checkpoint = SafeTensorCheckpoint(root)
            adapted = adapt_checkpoint(checkpoint, "olmoe")
            self.assertEqual(len(adapted.dense), 12)
            self.assertEqual(len(adapted.experts), 2)
            info = checkpoint.tensors["model.layers.0.mlp.experts.1.down_proj.weight"]
            with checkpoint.open_tensor(info.name) as view:
                self.assertEqual(len(view.raw), 4 * 3 * 2)

    def test_runtime_artifact_emits_adapter_ordered_extra_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root)
            adapted = adapt_checkpoint(SafeTensorCheckpoint(root), "olmoe")
            topology = adapted.runtime_topology
            operations = (
                topology.operations[0],
                RuntimeOperationTopology(
                    logical_layer=0,
                    capability="ffn.swiglu.dense.v1",
                    abi=1,
                    routed_component=None,
                    component_layer=0,
                    input_bindings=((
                        "hidden", "hidden.0", "batch.hidden.f32.cuda.v1",
                    ),),
                    output_bindings=((
                        "hidden", "fixture.extra.hidden",
                        "batch.hidden.f32.cuda.v1",
                    ),),
                ),
                *topology.operations[1:],
            )
            adapted = replace(
                adapted,
                runtime_topology=replace(
                    topology,
                    required_kernels=(
                        *topology.required_kernels,
                        ("ffn.swiglu.dense.v1", 1),
                    ),
                    operations=operations,
                ),
            )
            program = _runtime_model_descriptor_bytes(
                adapted, QUANT_ABI_ID
            ).decode("utf-8")
            self.assertIn(
                "operation\t1\t0\tffn.swiglu.dense.v1\t1\t-\t0",
                program,
            )
            self.assertIn(
                "operation\t3\t0\trouter.linear.topk.v1\t1\tdecoder\t0",
                program,
            )
            self.assertIn(
                "operation\t4\t0\tmoe.swiglu.routed.v1\t1\tdecoder\t0",
                program,
            )

    def test_runtime_artifact_rejects_duplicate_operation_parameters(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root)
            adapted = adapt_checkpoint(SafeTensorCheckpoint(root), "olmoe")
            topology = adapted.runtime_topology
            duplicate = replace(
                topology.operations[0],
                parameters=(("mode", 1), ("mode", 2)),
            )
            adapted = replace(
                adapted,
                runtime_topology=replace(
                    topology,
                    operations=(duplicate, *topology.operations[1:]),
                ),
            )
            with self.assertRaisesRegex(ValueError, "duplicate parameters"):
                _runtime_model_descriptor_bytes(adapted, QUANT_ABI_ID)

    def test_adapter_rejects_unidentified_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root, unknown=True)
            with self.assertRaisesRegex(AdapterError, "unidentified"):
                adapt_checkpoint(SafeTensorCheckpoint(root), "olmoe")

    def test_atomic_compile_layout_validation_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            source_values = _make_fixture(source)
            before = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
            outputs = [root / "pack-a", root / "pack-b"]
            manifests = []
            for output in outputs:
                result = compile_checkpoint(
                    CompileOptions(
                        source=source,
                        output=output,
                        max_expert_pack_bytes=PACK_ALIGNMENT,
                        source_id="synthetic/olmoe",
                        source_revision="fixture-v1",
                    )
                )
                self.assertTrue(result["validation"]["valid"])
                manifests.append((output / "manifest.json").read_bytes())
                validation = validate_container(output)
                self.assertEqual(validation["experts"], 2)
                manifest = load_json(output / "manifest.json")
                self.assertEqual(len(manifest["packs"]), 3)
                self.assertEqual(manifest["architecture"]["num_experts"], 2)
                self.assertEqual(manifest["quantization"]["profile"], "int8-symmetric-per-row-v1")
                program = (output / manifest["model_program"]["path"]).read_text(
                    encoding="utf-8"
                )
                self.assertIn("model\t3\tolmoe\t8\t16\t4", program)
                self.assertIn("block.full-attention.causal.v1", program)
                self.assertIn("router\tdecoder\trouter.linear.topk.v1\t1", program)
                self.assertIn(
                    "operation\t3\t0\tmoe.swiglu.routed.v1\t1\tdecoder\t0",
                    program,
                )
            self.assertEqual(manifests[0], manifests[1])
            after = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
            self.assertEqual(before, after)

            manifest = load_json(outputs[0] / "manifest.json")
            expert = manifest["experts"][0]
            pack_path = outputs[0] / expert["pack"]
            with pack_path.open("rb") as handle:
                header = handle.read(HEADER_BYTES)
                unpacked = EXPERT_HEADER_STRUCT.unpack(header[: EXPERT_HEADER_STRUCT.size])
                self.assertEqual(unpacked[0], b"EPEXPR01")
                section = expert["sections"]["gate_up_q"]
                handle.seek(expert["offset"] + section["offset"])
                gate_up = struct.unpack(f"<{section['bytes']}b", handle.read(section["bytes"]))
            gate_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
            gate_values = source_values[gate_name]
            scale = max(abs(value) for value in gate_values[:4]) / 127.0
            expected_first_row = tuple(round(value / scale) for value in gate_values[:4])
            self.assertEqual(gate_up[:4], expected_first_row)

    def test_interruption_is_incomplete_and_resume_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_fixture(source)
            output = root / "pack"

            def interrupt(kind: str, count: int) -> None:
                if kind == "dense-pack":
                    raise RuntimeError(f"simulated interruption after {count}")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                compile_checkpoint(CompileOptions(source=source, output=output), interrupt)
            self.assertFalse(output.exists())
            partial = root / "pack.partial"
            self.assertTrue((partial / "compile-state.json").is_file())
            with self.assertRaisesRegex(ValidationError, "incomplete"):
                validate_container(partial)

            result = compile_checkpoint(
                CompileOptions(source=source, output=output, resume=True)
            )
            self.assertTrue(result["validation"]["valid"])
            self.assertFalse(partial.exists())

    def test_opt_in_source_reclamation_only_after_committed_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_fixture(source)
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                reclaim_source_shards=True,
                max_expert_pack_bytes=PACK_ALIGNMENT,
            ))
            self.assertTrue(result["validation"]["valid"])
            self.assertFalse(any(source.glob("*.safetensors")))
            journal = load_json(output / "source-reclaim-journal.json")
            self.assertEqual(len(journal["shards"]), 2)
            self.assertEqual(len(result["report"]["reclaimed_source_shards"]), 2)

    def test_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_fixture(source)
            output = root / "pack"
            compile_checkpoint(CompileOptions(source=source, output=output))
            manifest = load_json(output / "manifest.json")
            pack = output / manifest["experts"][0]["pack"]
            with pack.open("r+b") as handle:
                handle.seek(HEADER_BYTES + 1)
                original = handle.read(1)
                handle.seek(HEADER_BYTES + 1)
                handle.write(bytes([original[0] ^ 1]))
            with self.assertRaisesRegex(ValidationError, "checksum"):
                validate_container(output)


if __name__ == "__main__":
    unittest.main()
