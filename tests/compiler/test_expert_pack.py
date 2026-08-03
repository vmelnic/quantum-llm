from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from compiler.expert_pack.adapters import adapt_checkpoint
from compiler.expert_pack.compile import CompileOptions, compile_checkpoint
from compiler.expert_pack.constants import EXPERT_HEADER_STRUCT, HEADER_BYTES, PACK_ALIGNMENT
from compiler.expert_pack.errors import AdapterError, ValidationError
from compiler.expert_pack.safetensors import SafeTensorCheckpoint
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
        "model.layers.0.self_attn.q_norm.weight": (2,),
        "model.layers.0.self_attn.k_norm.weight": (2,),
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


class ExpertPackTests(unittest.TestCase):
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
