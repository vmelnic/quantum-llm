from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

from compiler.expert_pack import quant as quant_module
from compiler.expert_pack.compile import CompileOptions, compile_checkpoint
from compiler.expert_pack.constants import (
    EXPERT_HEADER_STRUCT,
    FP4_QUANT_ABI_ID,
    FP4_QUANT_PROFILE,
    HEADER_BYTES,
    PACK_ALIGNMENT,
    QUANT_PROFILE,
)
from compiler.expert_pack.deepseek_quant import decode_scaled_fp4_e2m1_row
from compiler.expert_pack.errors import SourceFormatError
from compiler.expert_pack.quant import write_fp4_block32_rows
from compiler.expert_pack.safetensors import SafeTensorCheckpoint
from compiler.expert_pack.util import load_json
from compiler.expert_pack.validate import validate_container
from compiler.expert_pack.writer import expert_record_size

from tests.compiler.test_expert_pack import (
    _make_fixture,
    _tensor,
    _values,
    _write_safetensors,
)


def _float32_tensor(name: str, shape: tuple[int, ...], values: list[float]):
    return "F32", shape, struct.pack(f"<{len(values)}f", *values)


def _open_single_tensor(root: Path, name: str, shape: tuple[int, ...], values: list[float]):
    (root / "config.json").write_text(
        json.dumps({"model_type": "synthetic"}), encoding="utf-8"
    )
    _write_safetensors(root / "model.safetensors", {name: _float32_tensor(name, shape, values)})
    checkpoint = SafeTensorCheckpoint(root)
    return checkpoint.open_tensor(name)


def _encode(view, use_numpy):
    previous = quant_module._np
    if not use_numpy:
        quant_module._np = None
    try:
        destination = io.BytesIO()
        digest = hashlib.sha256()
        scales = write_fp4_block32_rows(view, destination, digest)
        return destination.getvalue(), scales, digest.digest()
    finally:
        quant_module._np = previous


def _make_wide_fixture(root: Path) -> dict[str, list[float]]:
    """The olmoe fixture scaled to block-32-aligned hidden/intermediate 32."""

    config = {
        "_name_or_path": "synthetic/olmoe-wide",
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_act": "silu",
        "clip_qkv": None,
        "hidden_size": 32,
        "intermediate_size": 32,
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
        "model.embed_tokens.weight": (8, 32),
        "model.norm.weight": (32,),
        "lm_head.weight": (8, 32),
        "model.layers.0.input_layernorm.weight": (32,),
        "model.layers.0.post_attention_layernorm.weight": (32,),
        "model.layers.0.self_attn.q_proj.weight": (32, 32),
        "model.layers.0.self_attn.k_proj.weight": (16, 32),
        "model.layers.0.self_attn.v_proj.weight": (16, 32),
        "model.layers.0.self_attn.o_proj.weight": (32, 32),
        "model.layers.0.self_attn.q_norm.weight": (32,),
        "model.layers.0.self_attn.k_norm.weight": (32,),
        "model.layers.0.mlp.gate.weight": (2, 32),
    }
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        shapes[prefix + "gate_proj.weight"] = (32, 32)
        shapes[prefix + "up_proj.weight"] = (32, 32)
        shapes[prefix + "down_proj.weight"] = (32, 32)
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


class Fp4Block32EncoderTests(unittest.TestCase):
    def test_exact_levels_round_trip_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
            values = [sign * level for level in levels for sign in (1.0, -1.0)] * 2
            view = _open_single_tensor(root, "w", (1, 32), values)
            with view:
                payload, scales, _ = _encode(view, use_numpy=True)
            # Block maximum is 6.0, so the scale is exactly 2^0 (code 127) and
            # every level encodes losslessly.
            self.assertEqual(scales, bytes((127,)))
            decoded = decode_scaled_fp4_e2m1_row(payload, scales)
            self.assertEqual(decoded, tuple(values))
            # Nibble order: values[2]=0.5 (index 1) in the low nibble,
            # values[3]=-0.5 (index 1, sign bit) in the high nibble.
            self.assertEqual(payload[0], 0x00)
            self.assertEqual(payload[1], 0x01 | (0x09 << 4))

    def test_nibble_packing_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [0.0] * 32
            values[0] = 1.0   # index 2, positive
            values[1] = -1.5  # index 3, sign bit
            values[31] = 6.0  # forces a unit block scale
            view = _open_single_tensor(root, "w", (1, 32), values)
            with view:
                payload, scales, _ = _encode(view, use_numpy=True)
            self.assertEqual(scales, bytes((127,)))
            self.assertEqual(payload[0], 0x02 | (0x0B << 4))
            self.assertTrue(all(byte == 0 for byte in payload[1:15]))
            self.assertEqual(payload[15], 0x70)

    def test_numpy_and_stdlib_paths_are_byte_identical(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = 7
            values = []
            for _ in range(3 * 64):
                seed = (seed * 1103515245 + 12345) % (2**31)
                values.append(((seed % 2001) - 1000) / 137.0)
            with _open_single_tensor(root, "w", (3, 64), values) as view:
                fast_payload, fast_scales, _ = _encode(view, use_numpy=True)
            with _open_single_tensor(root, "w", (3, 64), values) as view:
                slow_payload, slow_scales, _ = _encode(view, use_numpy=False)
            self.assertEqual(fast_payload, slow_payload)
            self.assertEqual(fast_scales, slow_scales)

    def test_relative_error_is_e2m1_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = 19
            values = []
            for _ in range(4 * 128):
                seed = (seed * 1103515245 + 12345) % (2**31)
                values.append(((seed % 4001) - 2000) / 333.0)
            view = _open_single_tensor(root, "w", (4, 128), values)
            with view:
                payload, scales, _ = _encode(view, use_numpy=True)
            blocks = 128 // 32
            for row in range(4):
                decoded = decode_scaled_fp4_e2m1_row(
                    payload[row * 64 : (row + 1) * 64],
                    scales[row * blocks : (row + 1) * blocks],
                )
                source = values[row * 128 : (row + 1) * 128]
                error = sum((a - b) ** 2 for a, b in zip(decoded, source))
                energy = sum(b * b for b in source)
                self.assertLess(error / energy, 0.02)

    def test_extreme_magnitudes_clamp_ue8m0_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [1e30, -1e-30, 0.0] + [1.0] * 29
            view = _open_single_tensor(root, "w", (1, 32), values)
            with view:
                payload, scales, _ = _encode(view, use_numpy=True)
            self.assertTrue(all(1 <= code <= 254 for code in scales))
            decoded = decode_scaled_fp4_e2m1_row(payload, scales)
            # E2M1 keeps the block maximum within one grid step.
            self.assertGreater(decoded[0], 1e30 * 0.5)
            self.assertLess(decoded[0], 1e30 * 1.5)

    def test_zero_block_uses_unit_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view = _open_single_tensor(root, "w", (1, 32), [0.0] * 32)
            with view:
                payload, scales, _ = _encode(view, use_numpy=True)
            self.assertEqual(scales, bytes((127,)))
            self.assertEqual(payload, bytes(16))

    def test_geometry_and_non_finite_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view = _open_single_tensor(root, "w", (1, 33), [1.0] * 33)
            with view:
                with self.assertRaises(SourceFormatError):
                    _encode(view, use_numpy=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [1.0] * 31 + [float("inf")]
            view = _open_single_tensor(root, "w", (1, 32), values)
            with view:
                with self.assertRaises(SourceFormatError):
                    _encode(view, use_numpy=True)


class Fp4ExpertPackCompileTests(unittest.TestCase):
    def test_unaligned_geometry_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_fixture(source)
            # The shared fixture uses hidden 4 / intermediate 3, which is not
            # block-32 aligned; the FP4 profile must refuse it.
            with self.assertRaisesRegex(ValueError, "block"):
                compile_checkpoint(
                    CompileOptions(
                        source=source,
                        output=root / "refused",
                        quant_profile=FP4_QUANT_PROFILE,
                    )
                )

    def test_fp4_compile_validates_and_decodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            source_values = _make_wide_fixture(source)
            output = root / "pack"
            result = compile_checkpoint(
                CompileOptions(
                    source=source,
                    output=output,
                    quant_profile=FP4_QUANT_PROFILE,
                    max_expert_pack_bytes=PACK_ALIGNMENT,
                    source_id="synthetic/olmoe-wide",
                    source_revision="fixture-v1",
                )
            )
            self.assertTrue(result["validation"]["valid"])
            self.assertEqual(validate_container(output)["experts"], 2)

            manifest = load_json(output / "manifest.json")
            quantization = manifest["quantization"]
            self.assertEqual(quantization["profile"], FP4_QUANT_PROFILE)
            self.assertEqual(quantization["abi_id"], FP4_QUANT_ABI_ID)
            self.assertEqual(quantization["group_size"], 32)
            self.assertEqual(quantization["scale_dtype"], "ue8m0")
            self.assertEqual(manifest["kernel_abi"]["quant_abi"], FP4_QUANT_ABI_ID)

            expert = manifest["experts"][0]
            self.assertEqual(expert["quant_abi"], FP4_QUANT_ABI_ID)
            self.assertEqual(expert["stored_dtype"], "FP4_E2M1")
            self.assertEqual(
                expert["stored_bytes"],
                expert_record_size(32, 32, PACK_ALIGNMENT, FP4_QUANT_ABI_ID),
            )
            sections = expert["sections"]
            self.assertEqual(sections["gate_up_q"]["bytes"], 2 * 32 * 32 // 2)
            self.assertEqual(sections["gate_up_scales"]["bytes"], 2 * 32 * 32 // 32)
            self.assertEqual(sections["down_q"]["bytes"], 32 * 32 // 2)
            self.assertEqual(sections["down_scales"]["bytes"], 32 * 32 // 32)
            # Dense tensors stay on the int8/FP32 profile.
            dense_abis = {item["quant_abi"] for item in manifest["tensors"]}
            self.assertEqual(dense_abis, {0, 1})

            pack_path = output / expert["pack"]
            with pack_path.open("rb") as handle:
                header = handle.read(HEADER_BYTES)
                unpacked = EXPERT_HEADER_STRUCT.unpack(header[: EXPERT_HEADER_STRUCT.size])
                self.assertEqual(unpacked[4], FP4_QUANT_ABI_ID)
                section = sections["gate_up_q"]
                handle.seek(expert["offset"] + section["offset"])
                payload = handle.read(section["bytes"])
                handle.seek(expert["offset"] + sections["gate_up_scales"]["offset"])
                scales = handle.read(sections["gate_up_scales"]["bytes"])
            self.assertTrue(all(1 <= code <= 254 for code in scales))
            gate_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
            gate_values = source_values[gate_name]
            decoded = decode_scaled_fp4_e2m1_row(payload[:16], scales[:1])
            source_row = gate_values[:32]
            error = sum((a - b) ** 2 for a, b in zip(decoded, source_row))
            energy = sum(b * b for b in source_row)
            self.assertLess(error / energy, 0.02)

    def test_int8_compile_still_validates_on_wide_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_wide_fixture(source)
            output = root / "pack"
            result = compile_checkpoint(
                CompileOptions(
                    source=source,
                    output=output,
                    quant_profile=QUANT_PROFILE,
                    max_expert_pack_bytes=PACK_ALIGNMENT,
                )
            )
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            self.assertEqual(manifest["quantization"]["abi_id"], 1)
            self.assertEqual({item["quant_abi"] for item in manifest["experts"]}, {1})

    def test_fp4_record_size_halves_int8_sections(self) -> None:
        int8_size = expert_record_size(2048, 512, PACK_ALIGNMENT)
        fp4_size = expert_record_size(2048, 512, PACK_ALIGNMENT, FP4_QUANT_ABI_ID)
        # 2*I*H + I*H int8 bytes vs half of that plus coarser scales: the FP4
        # record must be just over half the int8 record.
        self.assertLess(fp4_size, int8_size * 6 // 10)
        self.assertGreater(fp4_size, int8_size // 2)
        self.assertEqual(fp4_size % PACK_ALIGNMENT, 0)


if __name__ == "__main__":
    unittest.main()
