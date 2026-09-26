from __future__ import annotations

import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

from compiler.expert_pack import quant as quant_module
from compiler.expert_pack.activation_calibration import (
    DenseActivationCalibrationSet,
)
from compiler.expert_pack.compile import CompileOptions, compile_checkpoint
from compiler.expert_pack.constants import (
    EXPERT_HEADER_STRUCT,
    FP4_ACTIVATION_CODE_QUANT_PROFILE,
    FP4_ACTIVATION_QUANT_PROFILE,
    FP4_QUANT_ABI_ID,
    FP4_MSE_QUANT_PROFILE,
    FP4_QUANT_PROFILE,
    HEADER_BYTES,
    MXFP6_QUANT_ABI_ID,
    MXFP6_QUANT_PROFILE,
    PACK_ALIGNMENT,
    QUANT_PROFILE,
)
from compiler.expert_pack.fp4_reference import decode_scaled_fp4_e2m1_row
from compiler.expert_pack.errors import SourceFormatError
from compiler.expert_pack.quant import (
    write_fp4_block32_rows,
    write_int8_rows,
    write_mxfp6_e3m2_block32_rows,
)
from compiler.expert_pack.quality import (
    _qualify_int8,
    qualify_container_against_source,
)
from compiler.expert_pack.safetensors import SafeTensorCheckpoint
from compiler.expert_pack.util import load_json
from compiler.expert_pack.validate import validate_container
from compiler.expert_pack.writer import expert_record_size

from tests.compiler.test_expert_pack import (
    _make_hybrid_delta_fixture,
    _make_fixture,
    _tensor,
    _values,
    _write_safetensors,
)


def _write_dense_calibration(
    root: Path,
    name: str,
    values: list[list[int]],
    *,
    columns: int,
    scales: list[float] | None = None,
) -> Path:
    dense = root / "dense"
    dense.mkdir(parents=True, exist_ok=True)
    padded = (columns + 31) // 32 * 32
    rows = len(values)
    if rows <= 0 or any(len(row) != padded for row in values):
        raise ValueError("invalid test calibration rows")
    scales = scales or [1.0] * rows
    encoded = name.encode("utf-8")
    payload = bytearray(struct.pack(
        "<8sIIIIII", b"QLCALD01", 1, len(encoded), rows, rows,
        columns, padded,
    ))
    payload.extend(encoded)
    payload.extend(struct.pack(f"<{rows}I", *range(rows)))
    payload.extend(struct.pack(f"<{rows}f", *scales))
    payload.extend(bytes(value & 0xff for row in values for value in row))
    path = dense / (hashlib.sha256(encoded).hexdigest() + ".q8cal")
    path.write_bytes(payload)
    return path


def _float32_tensor(name: str, shape: tuple[int, ...], values: list[float]):
    return "F32", shape, struct.pack(f"<{len(values)}f", *values)


def _open_single_tensor(root: Path, name: str, shape: tuple[int, ...], values: list[float]):
    (root / "config.json").write_text(
        json.dumps({"model_type": "synthetic"}), encoding="utf-8"
    )
    _write_safetensors(root / "model.safetensors", {name: _float32_tensor(name, shape, values)})
    checkpoint = SafeTensorCheckpoint(root)
    return checkpoint.open_tensor(name)


def _encode(view, use_numpy, optimize_mse=False):
    previous = quant_module._np
    if not use_numpy:
        quant_module._np = None
    try:
        destination = io.BytesIO()
        digest = hashlib.sha256()
        scales = write_fp4_block32_rows(
            view, destination, digest, optimize_mse
        )
        return destination.getvalue(), scales, digest.digest()
    finally:
        quant_module._np = previous


def _encode_int8(view, use_numpy):
    previous = quant_module._np
    if not use_numpy:
        quant_module._np = None
    try:
        destination = io.BytesIO()
        digest = hashlib.sha256()
        scales = write_int8_rows(view, destination, digest)
        return destination.getvalue(), scales, digest.digest()
    finally:
        quant_module._np = previous


def _encode_mxfp6(view, use_numpy):
    previous = quant_module._np
    if not use_numpy:
        quant_module._np = None
    try:
        destination = io.BytesIO()
        digest = hashlib.sha256()
        scales = write_mxfp6_e3m2_block32_rows(view, destination, digest)
        return destination.getvalue(), scales, digest.digest()
    finally:
        quant_module._np = previous


def _decode_mxfp6(payload: bytes, scales: bytes) -> tuple[float, ...]:
    levels = (
        0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
        0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75,
        2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0,
        8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0,
    )
    decoded = []
    for block, scale_code in enumerate(scales):
        block_payload = payload[block * 24:(block + 1) * 24]
        scale = 2.0 ** (scale_code - 127)
        for group in range(8):
            word = int.from_bytes(block_payload[group * 3:group * 3 + 3],
                                  "little")
            for item in range(4):
                code = (word >> (6 * item)) & 0x3f
                value = levels[code & 0x1f] * scale
                decoded.append(-value if code & 0x20 else value)
    return tuple(decoded)


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
    def test_activation_aware_selection_uses_complete_output_residual(self) -> None:
        if quant_module._np is None:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [4.2] + [0.26] * 31
            calibration_root = root / "calibration"
            _write_dense_calibration(
                calibration_root,
                "w",
                [[1] + [0] * 31],
                columns=32,
            )
            calibration = DenseActivationCalibrationSet(calibration_root)
            baseline_source = root / "source"
            baseline_source.mkdir()
            candidate_source = root / "candidate"
            candidate_source.mkdir()
            with _open_single_tensor(
                baseline_source, "w", (1, 32), values
            ) as view:
                baseline_payload, baseline_scales, _ = _encode(
                    view, use_numpy=True, optimize_mse=True
                )
            with _open_single_tensor(
                candidate_source, "w", (1, 32), values
            ) as view:
                destination = io.BytesIO()
                digest = hashlib.sha256()
                candidate_scales = write_fp4_block32_rows(
                    view,
                    destination,
                    digest,
                    optimize_mse=True,
                    activation_calibration=calibration,
                )
                candidate_payload = destination.getvalue()
            report = calibration.finalize()
            self.assertEqual(baseline_scales, bytes((126,)))
            self.assertEqual(candidate_scales, bytes((127,)))
            self.assertNotEqual(candidate_payload, baseline_payload)
            self.assertEqual(report["covering_reselections"], 1)

    def test_activation_code_selection_uses_complete_output_residual(self) -> None:
        if quant_module._np is None:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [0.9, 0.9] + [0.0] * 30
            calibration_root = root / "calibration"
            _write_dense_calibration(
                calibration_root,
                "w",
                [[1, 1] + [0] * 30],
                columns=32,
            )
            calibration = DenseActivationCalibrationSet(calibration_root)
            calibration.configure_payload_targets({"w"})
            baseline_root = root / "baseline"
            baseline_root.mkdir()
            candidate_root = root / "candidate"
            candidate_root.mkdir()
            with _open_single_tensor(
                baseline_root, "w", (1, 32), values
            ) as view:
                baseline_payload, baseline_scales, _ = _encode(
                    view, use_numpy=True, optimize_mse=True
                )
            with _open_single_tensor(
                candidate_root, "w", (1, 32), values
            ) as view:
                destination = io.BytesIO()
                digest = hashlib.sha256()
                candidate_scales = write_fp4_block32_rows(
                    view,
                    destination,
                    digest,
                    optimize_mse=True,
                    activation_calibration=calibration,
                )
                candidate_payload = destination.getvalue()
            report = calibration.finalize()
            self.assertEqual(candidate_scales, baseline_scales)
            self.assertNotEqual(candidate_payload, baseline_payload)
            self.assertEqual(report["covering_reselections"], 0)
            self.assertEqual(report["payload_reselections"], 1)
            self.assertEqual(report["payload_matrix_names"], ["w"])

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
            with _open_single_tensor(root, "w", (3, 64), values) as view:
                optimized_fast = _encode(
                    view, use_numpy=True, optimize_mse=True
                )
            with _open_single_tensor(root, "w", (3, 64), values) as view:
                optimized_slow = _encode(
                    view, use_numpy=False, optimize_mse=True
                )
            self.assertEqual(optimized_fast, optimized_slow)

    def test_mse_scale_clips_one_outlier_and_reduces_block_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [6.1] + [0.5] * 31
            with _open_single_tensor(root, "w", (1, 32), values) as view:
                covering_payload, covering_scales, _ = _encode(
                    view, use_numpy=True
                )
            with _open_single_tensor(root, "w", (1, 32), values) as view:
                optimized_fast = _encode(
                    view, use_numpy=True, optimize_mse=True
                )
            with _open_single_tensor(root, "w", (1, 32), values) as view:
                optimized_slow = _encode(
                    view, use_numpy=False, optimize_mse=True
                )
            self.assertEqual(optimized_fast, optimized_slow)
            optimized_payload, optimized_scales, _ = optimized_fast
            self.assertEqual(covering_scales, bytes((128,)))
            self.assertEqual(optimized_scales, bytes((127,)))
            covering = decode_scaled_fp4_e2m1_row(
                covering_payload, covering_scales
            )
            optimized = decode_scaled_fp4_e2m1_row(
                optimized_payload, optimized_scales
            )
            covering_error = sum(
                (actual - expected) ** 2
                for actual, expected in zip(covering, values)
            )
            optimized_error = sum(
                (actual - expected) ** 2
                for actual, expected in zip(optimized, values)
            )
            self.assertLess(optimized_error, covering_error)

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

    def test_unaligned_columns_are_padded_and_non_finite_inputs_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view = _open_single_tensor(root, "w", (1, 33), [1.0] * 33)
            with view:
                payload, scales, _ = _encode(view, use_numpy=True)
            self.assertEqual(len(payload), 64 // 2)
            self.assertEqual(len(scales), 64 // 32)
            decoded = decode_scaled_fp4_e2m1_row(payload, scales)
            self.assertEqual(decoded[:33], (1.0,) * 33)
            self.assertEqual(decoded[33:], (0.0,) * 31)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [1.0] * 31 + [float("inf")]
            view = _open_single_tensor(root, "w", (1, 32), values)
            with view:
                with self.assertRaises(SourceFormatError):
                    _encode(view, use_numpy=True)


class Mxfp6E3m2EncoderTests(unittest.TestCase):
    def test_exact_levels_pack_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positive = [
                0.0, 0.0625, 0.125, 0.1875,
                0.25, 0.3125, 0.375, 0.4375,
                0.5, 0.625, 0.75, 0.875,
                1.0, 1.25, 1.5, 1.75,
                2.0, 2.5, 3.0, 3.5,
                4.0, 5.0, 6.0, 7.0,
                8.0, 10.0, 12.0, 14.0,
                16.0, 20.0, 24.0, 28.0,
            ]
            with _open_single_tensor(root, "w", (1, 32), positive) as view:
                payload, scales, _ = _encode_mxfp6(view, use_numpy=True)
            self.assertEqual(scales, bytes((127,)))
            self.assertEqual(len(payload), 24)
            self.assertEqual(_decode_mxfp6(payload, scales), tuple(positive))
            self.assertEqual(payload[:3], bytes((0x40, 0x20, 0x0c)))

    def test_numpy_and_stdlib_paths_are_byte_identical(self) -> None:
        if quant_module._np is None:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [((index * 37) % 211 - 105) / 17.0
                      for index in range(3 * 64)]
            with _open_single_tensor(root, "w", (3, 64), values) as view:
                fast = _encode_mxfp6(view, use_numpy=True)
            with _open_single_tensor(root, "w", (3, 64), values) as view:
                slow = _encode_mxfp6(view, use_numpy=False)
            self.assertEqual(fast, slow)

    def test_ties_round_to_even_and_columns_are_padded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [0.09375, 28.0] + [0.0] * 31
            with _open_single_tensor(root, "w", (1, 33), values) as view:
                payload, scales, _ = _encode_mxfp6(view, use_numpy=True)
            self.assertEqual(len(payload), 64 * 3 // 4)
            self.assertEqual(len(scales), 2)
            decoded = _decode_mxfp6(payload, scales)
            # 0.09375 is halfway between E3M2 codes 1 and 2; code 2 is even.
            self.assertEqual(decoded[0], 0.125)
            self.assertEqual(decoded[1], 28.0)
            self.assertEqual(decoded[33:], (0.0,) * 31)


class Int8RowEncoderTests(unittest.TestCase):
    def test_binary32_half_boundary_is_byte_identical_without_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [2.234375, 1.1171875]
            with _open_single_tensor(root, "w", (1, 2), values) as view:
                fallback_payload, fallback_scales, _ = _encode_int8(
                    view, use_numpy=False
                )
            self.assertEqual(fallback_payload, bytes((127, 63)))

            if quant_module._np is not None:
                with _open_single_tensor(root, "w", (1, 2), values) as view:
                    fast_payload, fast_scales, _ = _encode_int8(
                        view, use_numpy=True
                    )
                self.assertEqual(fast_payload, fallback_payload)
                self.assertEqual(fast_scales, fallback_scales)

            entry = {
                "name": "w",
                "source_shape": [1, 2],
                "offset": 0,
                "sections": {
                    "data": {"offset": 0, "bytes": 2},
                    "scales": {"offset": 2, "bytes": 4},
                },
            }
            packed = io.BytesIO(fallback_payload + fallback_scales)
            with _open_single_tensor(root, "w", (1, 2), values) as view:
                _, payload_bad, scale_bad = _qualify_int8(
                    packed, view, entry, samples_per_tensor=1
                )
            self.assertEqual(payload_bad, 0)
            self.assertEqual(scale_bad, 0)


class Fp4ExpertPackCompileTests(unittest.TestCase):
    def test_activation_aware_profile_preserves_fp4_abi_and_provenance(self) -> None:
        if quant_module._np is None:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)
            name = "model.language_model.layers.0.mlp.gate_proj.weight"
            calibration_root = root / "calibration"
            _write_dense_calibration(
                calibration_root,
                name,
                [[1] + [0] * 31],
                columns=32,
            )
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                adapter="hybrid_delta",
                quant_profile=FP4_ACTIVATION_QUANT_PROFILE,
                activation_calibration=calibration_root,
                source_revision=source.name,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            self.assertEqual(
                manifest["quantization"]["profile"],
                FP4_ACTIVATION_QUANT_PROFILE,
            )
            calibration = manifest["quantization"]["activation_calibration"]
            self.assertEqual(calibration["dense_files"], 1)
            self.assertEqual(calibration["matrix_names"], [name])
            entry = next(
                item for item in manifest["tensors"] if item["name"] == name
            )
            self.assertEqual(entry["quant_abi"], FP4_QUANT_ABI_ID)
            self.assertEqual(
                entry["fp4_scale_policy"],
                "activation-aware-complete-output-residual-v1",
            )
            quality = qualify_container_against_source(
                source,
                output,
                samples_per_tensor=3,
                maximum_relative_l2=0.40,
                minimum_cosine=0.80,
            )
            self.assertTrue(quality["valid"])
            self.assertEqual(
                quality["sampled"]["fp4_payload_mismatches"], 0
            )
            self.assertEqual(
                quality["sampled"]["fp4_scale_mismatches"], 0
            )

    def test_activation_code_profile_is_artifact_driven_and_qualified(self) -> None:
        if quant_module._np is None:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)
            name = "model.language_model.layers.0.mlp.gate_proj.weight"
            calibration_root = root / "calibration"
            _write_dense_calibration(
                calibration_root,
                name,
                [[1, 1] + [0] * 30],
                columns=32,
            )
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                adapter="hybrid_delta",
                quant_profile=FP4_ACTIVATION_CODE_QUANT_PROFILE,
                activation_calibration=calibration_root,
                source_revision=source.name,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            calibration = manifest["quantization"]["activation_calibration"]
            self.assertEqual(calibration["payload_dense_matrices"], 1)
            self.assertEqual(calibration["payload_matrix_names"], [name])
            entry = next(
                item for item in manifest["tensors"] if item["name"] == name
            )
            self.assertEqual(entry["quant_abi"], FP4_QUANT_ABI_ID)
            self.assertEqual(
                entry["fp4_payload_policy"],
                "activation-aware-adjacent-code-top64-complete-output-residual-v1",
            )
            quality = qualify_container_against_source(
                source,
                output,
                samples_per_tensor=3,
                maximum_relative_l2=0.40,
                minimum_cosine=0.80,
            )
            self.assertTrue(quality["valid"])
            self.assertEqual(
                quality["sampled"]["fp4_payload_mismatches"], 0
            )
            self.assertEqual(
                quality["sampled"]["fp4_scale_mismatches"], 0
            )

    def test_dense_encoding_policy_resolves_capability_role_and_excludes_calibration(self) -> None:
        if quant_module._np is None:
            self.skipTest("NumPy is required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)
            a_projections = [
                f"model.language_model.layers.{layer}.linear_attn."
                "in_proj_a.weight"
                for layer in range(3)
            ]
            a_projection = a_projections[0]
            fp4_projection = (
                "model.language_model.layers.0.mlp.gate_proj.weight"
            )
            calibration_root = root / "calibration"
            for name in (a_projection, fp4_projection):
                _write_dense_calibration(
                    calibration_root,
                    name,
                    [[1] + [0] * 31],
                    columns=32,
                )
            policy_path = root / "dense-encoding-policy.json"
            policy_path.write_text(json.dumps({
                "schema": "expert-pack-dense-encoding-policy-v1",
                "rules": [{
                    "encoding": MXFP6_QUANT_PROFILE,
                    "capability": (
                        "block.recurrent-linear-attention."
                        "split-gated-delta.v1"
                    ),
                    "tensor_role": "a_projection",
                }],
            }), encoding="utf-8")
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                adapter="hybrid_delta",
                quant_profile=FP4_ACTIVATION_CODE_QUANT_PROFILE,
                activation_calibration=calibration_root,
                dense_encoding_policy=policy_path,
                source_revision=source.name,
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            policy = manifest["quantization"]["dense_encoding_policy"]
            self.assertEqual(policy["resolved_tensor_count"], 3)
            self.assertEqual(policy["resolved_tensors"], a_projections)
            entries = {
                entry["name"]: entry for entry in manifest["tensors"]
            }
            self.assertTrue(all(
                entries[name]["quant_abi"] == MXFP6_QUANT_ABI_ID
                for name in a_projections
            ))
            self.assertEqual(
                entries[fp4_projection]["quant_abi"], FP4_QUANT_ABI_ID
            )
            calibration = manifest["quantization"]["activation_calibration"]
            self.assertEqual(
                calibration["excluded_matrix_names"], [a_projection]
            )
            self.assertEqual(calibration["matrix_names"], [fp4_projection])
            self.assertEqual(
                calibration["payload_matrix_names"], [fp4_projection]
            )

    def test_dense_encoding_policy_fails_closed_on_unknown_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)
            policy_path = root / "dense-encoding-policy.json"
            policy_path.write_text(json.dumps({
                "schema": "expert-pack-dense-encoding-policy-v1",
                "rules": [{
                    "encoding": MXFP6_QUANT_PROFILE,
                    "capability": (
                        "block.recurrent-linear-attention."
                        "split-gated-delta.v1"
                    ),
                    "tensor_role": "not_a_real_role",
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matched no operation"):
                compile_checkpoint(CompileOptions(
                    source=source,
                    output=root / "pack",
                    adapter="hybrid_delta",
                    quant_profile=FP4_QUANT_PROFILE,
                    dense_encoding_policy=policy_path,
                ))

    def test_bf16_dense_activation_input_is_artifact_declared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)
            output = root / "pack"
            result = compile_checkpoint(CompileOptions(
                source=source,
                output=output,
                adapter="hybrid_delta",
                quant_profile=FP4_QUANT_PROFILE,
                dense_activation_input="bf16",
            ))
            self.assertTrue(result["validation"]["valid"])
            manifest = load_json(output / "manifest.json")
            self.assertEqual(
                manifest["quantization"]["dense_activation_input"],
                "bf16",
            )
            program = (output / "runtime-model.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "attribute\tdense_activation_input_bf16\t1\n", program
            )
            self.assertIn(
                "kernel\tdense.activation-input.bfloat16.v1\t1\n", program
            )

    def test_bf16_dense_activation_input_rejects_mxfp6_mix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_hybrid_delta_fixture(source)
            policy_path = root / "dense-encoding-policy.json"
            policy_path.write_text(json.dumps({
                "schema": "expert-pack-dense-encoding-policy-v1",
                "rules": [{
                    "encoding": MXFP6_QUANT_PROFILE,
                    "capability": (
                        "block.recurrent-linear-attention."
                        "split-gated-delta.v1"
                    ),
                    "tensor_role": "a_projection",
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "requires FP4 dense matrices"
            ):
                compile_checkpoint(CompileOptions(
                    source=source,
                    output=root / "pack",
                    adapter="hybrid_delta",
                    quant_profile=FP4_QUANT_PROFILE,
                    dense_encoding_policy=policy_path,
                    dense_activation_input="bf16",
                ))

    def test_mse_profile_preserves_abi_and_passes_source_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_wide_fixture(source)
            output = root / "pack"
            compile_checkpoint(
                CompileOptions(
                    source=source,
                    output=output,
                    quant_profile=FP4_MSE_QUANT_PROFILE,
                    max_expert_pack_bytes=PACK_ALIGNMENT,
                    source_id="synthetic/mse-quality",
                    source_revision=source.name,
                )
            )
            manifest = load_json(output / "manifest.json")
            self.assertEqual(
                manifest["quantization"]["profile"],
                FP4_MSE_QUANT_PROFILE,
            )
            self.assertEqual(
                manifest["quantization"]["abi_id"], FP4_QUANT_ABI_ID
            )
            self.assertEqual(
                {entry["quant_abi"] for entry in manifest["experts"]},
                {FP4_QUANT_ABI_ID},
            )
            quality = qualify_container_against_source(
                source, output, samples_per_tensor=3,
                maximum_relative_l2=0.30, minimum_cosine=0.90,
            )
            self.assertTrue(quality["valid"])
            self.assertEqual(
                quality["sampled"]["fp4_payload_mismatches"], 0
            )
            self.assertEqual(
                quality["sampled"]["fp4_scale_mismatches"], 0
            )

    def test_fp4_source_quality_gate_is_artifact_driven_and_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            _make_wide_fixture(source)
            output = root / "pack"
            compile_checkpoint(
                CompileOptions(
                    source=source,
                    output=output,
                    quant_profile=FP4_QUANT_PROFILE,
                    max_expert_pack_bytes=PACK_ALIGNMENT,
                    source_id="synthetic/quality",
                    source_revision=source.name,
                )
            )
            result = qualify_container_against_source(
                source, output, samples_per_tensor=3,
                maximum_relative_l2=0.30, minimum_cosine=0.90,
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["sampled"]["fp4_payload_mismatches"], 0)
            self.assertEqual(result["sampled"]["int8_payload_mismatches"], 0)
            self.assertEqual(result["sampled"]["f32_value_mismatches"], 0)
            self.assertFalse(result["records"]["unreferenced"])
            fp4_values = result["aggregate"]["fp4"]["values"]
            f32_values = result["aggregate"]["f32"]["values"]
            int8_values = result["aggregate"]["int8"]["values"]
            self.assertGreater(fp4_values, 0)
            self.assertGreater(f32_values, 0)
            self.assertGreater(int8_values, 0)
            self.assertEqual(
                result["aggregate"]["all"]["values"],
                fp4_values + int8_values + f32_values,
            )

            manifest = load_json(output / "manifest.json")
            entry = manifest["experts"][0]
            pack = output / entry["pack"]
            position = (
                entry["offset"] + entry["sections"]["gate_up_q"]["offset"]
            )
            with pack.open("r+b") as handle:
                handle.seek(position)
                original = handle.read(1)
                handle.seek(position)
                handle.write(bytes((original[0] ^ 0x01,)))
            corrupted = qualify_container_against_source(
                source, output, samples_per_tensor=3,
                maximum_relative_l2=0.30, minimum_cosine=0.90,
            )
            self.assertFalse(corrupted["valid"])
            self.assertGreater(
                corrupted["sampled"]["fp4_payload_mismatches"], 0
            )

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
            model_program = (output / manifest["model_program"]["path"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("\t1\t2\tfp4.e2m1.ue8m0.block32", model_program)

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
