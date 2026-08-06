from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from compiler.expert_pack.deepseek_quant import (
    decode_fp8_e4m3fn,
    decode_fp8_e4m3fn_values,
    FP4_E2M1_VALUES,
    decode_fp4_e2m1_values,
    decode_scaled_fp4_e2m1_row,
    decode_ue8m0,
    decode_ue8m0_values,
    iter_scaled_fp4_e2m1_blocks,
)
from compiler.expert_pack.errors import SourceFormatError
from compiler.expert_pack.deepseek_slice import (
    _decode_numpy,
    _f32_to_bf16_words,
    _deepseek_csa_ratio4_reference,
    _deepseek_hca_reference,
    _exclusive_pack_lock,
    _reconstruct_compact_blocks,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class DeepSeekQuantTests(unittest.TestCase):
    @unittest.skipIf(np is None, "NumPy fast path is optional")
    def test_compact_dense_candidate_schemes_are_bounded(self) -> None:
        values = np.asarray(
            [[[-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
               0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0]]],
            dtype=np.float32,
        )
        symmetric, symmetric_scales = _reconstruct_compact_blocks(
            values, scheme="symmetric", bits=4
        )
        e2m1, e2m1_scales = _reconstruct_compact_blocks(
            values, scheme="e2m1", bits=4
        )
        self.assertEqual(symmetric.shape, values.shape)
        self.assertEqual(e2m1.shape, values.shape)
        self.assertEqual(symmetric_scales.dtype, np.float32)
        self.assertEqual(e2m1_scales.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(symmetric))), 6.001)
        self.assertTrue(np.array_equal(e2m1, values))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _reconstruct_compact_blocks(values, scheme="e2m1", bits=5)

    def test_compact_pack_lock_rejects_overlapping_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".compact.lock"
            with _exclusive_pack_lock(lock):
                with self.assertRaisesRegex(SourceFormatError, "another"):
                    with _exclusive_pack_lock(lock):
                        self.fail("overlapping compact pack lock was admitted")

    @unittest.skipIf(np is None, "NumPy fast path is optional")
    def test_bf16_rounding_uses_nearest_even(self) -> None:
        values = np.asarray([1.0, 1.00390625, 1.0078125], dtype=np.float32)
        words = _f32_to_bf16_words(values)
        self.assertEqual(tuple(int(value) for value in words), (0x3F80, 0x3F80, 0x3F81))

    @unittest.skipIf(np is None, "NumPy fast path is optional")
    def test_csa_ratio_four_uses_overlap_half_geometry(self) -> None:
        inputs = np.zeros((4, 4096), dtype=np.float32)
        wkv = np.zeros((1024, 4096), dtype=np.float32)
        wgate = np.zeros_like(wkv)
        ape = np.zeros((4, 1024), dtype=np.float32)
        norm = np.ones(512, dtype=np.float32)
        output = _deepseek_csa_ratio4_reference(inputs, wkv, wgate, ape, norm)
        self.assertEqual(output.shape, (512,))
        self.assertTrue(np.array_equal(output, np.zeros(512, dtype=np.float32)))

    @unittest.skipIf(np is None, "NumPy fast path is optional")
    def test_csa_ratio_four_supports_indexer_geometry(self) -> None:
        inputs = np.zeros((4, 4096), dtype=np.float32)
        wkv = np.zeros((256, 4096), dtype=np.float32)
        wgate = np.zeros_like(wkv)
        ape = np.zeros((4, 256), dtype=np.float32)
        norm = np.ones(128, dtype=np.float32)
        output = _deepseek_csa_ratio4_reference(inputs, wkv, wgate, ape, norm)
        self.assertEqual(output.shape, (128,))
        self.assertTrue(np.array_equal(output, np.zeros(128, dtype=np.float32)))

    @unittest.skipIf(np is None, "NumPy fast path is optional")
    def test_hca_reference_preserves_doubly_stochastic_contract(self) -> None:
        streams = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
        fn = np.zeros((24, 32), dtype=np.float32)
        base = np.zeros(24, dtype=np.float32)
        scale = np.ones(3, dtype=np.float32)
        pre, post, comb, collapsed, updated = _deepseek_hca_reference(
            streams, fn, base, scale
        )
        self.assertTrue(np.allclose(pre, 0.500001))
        self.assertTrue(np.allclose(post, 1.0))
        self.assertTrue(np.allclose(comb.sum(axis=0), 1.0, atol=2e-6))
        self.assertTrue(np.allclose(comb.sum(axis=1), 1.0, atol=2e-6))
        self.assertEqual(collapsed.shape, (8,))
        self.assertEqual(updated.shape, (4, 8))

    def test_e4m3fn_finite_table_and_nan_codes(self) -> None:
        self.assertEqual(decode_fp8_e4m3fn(0x00), 0.0)
        self.assertEqual(decode_fp8_e4m3fn(0x01), 2.0**-9)
        self.assertEqual(decode_fp8_e4m3fn(0x38), 1.0)
        self.assertEqual(decode_fp8_e4m3fn(0x7E), 448.0)
        self.assertEqual(decode_fp8_e4m3fn(0xFE), -448.0)
        finite = bytes(code for code in range(255) if code != 0x7F)
        self.assertEqual(len(decode_fp8_e4m3fn_values(finite)), 254)
        for code in (0x7F, 0xFF):
            with self.assertRaises(SourceFormatError):
                decode_fp8_e4m3fn(code)

    @unittest.skipIf(np is None, "NumPy fast path is optional")
    def test_numpy_slice_decoder_matches_dependency_free_decoder(self) -> None:
        packed = bytes((0x21,)) * 16 + bytes((0x76,)) * 16
        expected = decode_scaled_fp4_e2m1_row(packed, bytes((127, 128)))
        actual = _decode_numpy(
            np.frombuffer(packed, dtype=np.uint8).reshape(1, 32),
            np.frombuffer(bytes((127, 128)), dtype=np.uint8).reshape(1, 2),
        )
        self.assertEqual(tuple(float(value) for value in actual[0]), expected)
        with self.assertRaisesRegex(SourceFormatError, "NaN"):
            _decode_numpy(
                np.frombuffer(bytes(16), dtype=np.uint8).reshape(1, 16),
                np.frombuffer(bytes((255,)), dtype=np.uint8).reshape(1, 1),
            )

    def test_all_e2m1_codes_and_packed_order(self) -> None:
        packed = bytes((high << 4) | low for low, high in zip(range(8), range(8, 16)))
        expected = tuple(
            value
            for low, high in zip(FP4_E2M1_VALUES[:8], FP4_E2M1_VALUES[8:])
            for value in (low, high)
        )
        self.assertEqual(decode_fp4_e2m1_values(packed), expected)
        self.assertEqual(decode_fp4_e2m1_values(bytes((0x21,))), (0.5, 1.0))

    def test_all_finite_ue8m0_codes_are_exact_powers_of_two(self) -> None:
        decoded = decode_ue8m0_values(bytes(range(255)))
        self.assertEqual(len(decoded), 255)
        for code, value in enumerate(decoded):
            self.assertEqual(value, math.ldexp(1.0, code - 127))
        self.assertEqual(decode_ue8m0(127), 1.0)
        self.assertEqual(decode_ue8m0(128), 2.0)
        with self.assertRaisesRegex(SourceFormatError, "NaN"):
            decode_ue8m0(255)

    def test_block_scaling_and_geometry_are_fail_closed(self) -> None:
        packed = bytes((0x21,)) * 16 + bytes((0x76,)) * 16
        blocks = list(iter_scaled_fp4_e2m1_blocks(packed, bytes((127, 128))))
        self.assertEqual(blocks[0], (0.5, 1.0) * 16)
        self.assertEqual(blocks[1], (8.0, 12.0) * 16)
        row = decode_scaled_fp4_e2m1_row(packed, bytes((127, 128)))
        self.assertEqual(row, blocks[0] + blocks[1])

        with self.assertRaisesRegex(SourceFormatError, "geometry mismatch"):
            decode_scaled_fp4_e2m1_row(packed, bytes((127,)))
        with self.assertRaisesRegex(SourceFormatError, "positive even"):
            decode_scaled_fp4_e2m1_row(b"", b"", block_size=31)

    def test_invalid_scale_inputs_are_rejected(self) -> None:
        for value in (-1, 256, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(SourceFormatError):
                    decode_ue8m0(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
