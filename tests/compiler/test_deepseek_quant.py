from __future__ import annotations

import math
import unittest

from compiler.expert_pack.deepseek_quant import (
    FP4_E2M1_VALUES,
    decode_fp4_e2m1_values,
    decode_scaled_fp4_e2m1_row,
    decode_ue8m0,
    decode_ue8m0_values,
    iter_scaled_fp4_e2m1_blocks,
)
from compiler.expert_pack.errors import SourceFormatError


class DeepSeekQuantTests(unittest.TestCase):
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
