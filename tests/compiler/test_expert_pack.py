from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from compiler.expert_pack.adapters import adapt_checkpoint
from compiler.expert_pack.compile import CompileOptions, compile_checkpoint
from compiler.expert_pack.constants import EXPERT_HEADER_STRUCT, HEADER_BYTES, PACK_ALIGNMENT
from compiler.expert_pack.deepseek_v4 import (
    _build_expected,
    estimate_deepseek_v4_representations,
    validate_deepseek_v4_source,
)
from compiler.expert_pack.deepseek_slice import _deepseek_mtp_partition
from compiler.expert_pack.errors import AdapterError, ValidationError
from compiler.expert_pack.safetensors import SafeTensorCheckpoint, TensorInfo
from compiler.expert_pack.source_inventory import group_source_tensors, inspect_source
from compiler.expert_pack.util import load_json, sha256_file
from compiler.expert_pack.validate import validate_container


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
        "hidden_size": 4,
        "moe_intermediate_size": 2,
        "shared_expert_intermediate_size": 2,
        "max_position_embeddings": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 1,
        "full_attention_interval": 2,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
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
        "model.embed_tokens.weight": (8, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (8, 4),
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.post_attention_layernorm.weight": (4,),
        "model.layers.0.linear_attn.dt_bias": (2,),
        "model.layers.0.linear_attn.A_log": (2,),
        "model.layers.0.linear_attn.conv1d.weight": (8, 1, 2),
        "model.layers.0.linear_attn.in_proj_qkvz.weight": (12, 4),
        "model.layers.0.linear_attn.in_proj_ba.weight": (4, 4),
        "model.layers.0.linear_attn.norm.weight": (2,),
        "model.layers.0.linear_attn.out_proj.weight": (4, 4),
        "model.layers.0.mlp.gate.weight": (2, 4),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": (2, 4),
        "model.layers.0.mlp.shared_expert.up_proj.weight": (2, 4),
        "model.layers.0.mlp.shared_expert.down_proj.weight": (4, 2),
        "model.layers.0.mlp.shared_expert_gate.weight": (1, 4),
    }
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        shapes[prefix + "gate_proj.weight"] = (2, 4)
        shapes[prefix + "up_proj.weight"] = (2, 4)
        shapes[prefix + "down_proj.weight"] = (4, 2)
    if include_mtp:
        shapes.update({
            "mtp.fc.weight": (4, 8),
            "mtp.pre_fc_norm_embedding.weight": (4,),
            "mtp.pre_fc_norm_hidden.weight": (4,),
            "mtp.norm.weight": (4,),
            "mtp.layers.0.input_layernorm.weight": (4,),
            "mtp.layers.0.post_attention_layernorm.weight": (4,),
            "mtp.layers.0.self_attn.q_proj.weight": (8, 4),
            "mtp.layers.0.self_attn.k_proj.weight": (2, 4),
            "mtp.layers.0.self_attn.v_proj.weight": (2, 4),
            "mtp.layers.0.self_attn.o_proj.weight": (4, 4),
            "mtp.layers.0.self_attn.q_norm.weight": (2,),
            "mtp.layers.0.self_attn.k_norm.weight": (2,),
            "mtp.layers.0.mlp.gate.weight": (2, 4),
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (2, 4),
            "mtp.layers.0.mlp.shared_expert.up_proj.weight": (2, 4),
            "mtp.layers.0.mlp.shared_expert.down_proj.weight": (4, 2),
            "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 4),
        })
        for expert in range(2):
            prefix = f"mtp.layers.0.mlp.experts.{expert}."
            shapes[prefix + "gate_proj.weight"] = (2, 4)
            shapes[prefix + "up_proj.weight"] = (2, 4)
            shapes[prefix + "down_proj.weight"] = (4, 2)
    tensors = {name: _tensor(name, shape) for name, shape in shapes.items()}
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
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source, output=output, adapter="qwen3_next",
                max_expert_pack_bytes=PACK_ALIGNMENT,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            conv = next(item for item in manifest["tensors"] if item["name"].endswith("conv1d.weight"))
            self.assertEqual(conv["source_shape"], [8, 1, 2])
            self.assertEqual(conv["stored_dtype"], "F32")

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
