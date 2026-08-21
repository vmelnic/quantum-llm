import unittest

from ops.python.exact_kv_structural_gate import (
    HUFFMAN_ESCAPE_SYMBOL,
    _capacity,
    _huffman_code_lengths,
)


class ExactKvStructuralCapacityTest(unittest.TestCase):
    def test_huffman_lengths_cover_symbols_and_form_prefix_code(self) -> None:
        lengths = _huffman_code_lengths([
            (0, 10),
            (1, 4),
            (2, 2),
            (HUFFMAN_ESCAPE_SYMBOL, 1),
        ])
        self.assertEqual(set(lengths), {0, 1, 2, HUFFMAN_ESCAPE_SYMBOL})
        self.assertLessEqual(sum(2.0 ** -length for length in lengths.values()),
                             1.0)
        self.assertLessEqual(lengths[0], lengths[1])
        self.assertLessEqual(lengths[1], lengths[2])

    def test_capacity_accounts_for_unavailable_and_runtime_bytes(self) -> None:
        weight = {
            "device_bytes": 24 * (1 << 30),
            "target_only": {
                "ideal_weight_payload_bytes": 11 * (1 << 30),
                "required_recurrent_state_bytes": 1 << 30,
                "exact_fp16_kv_raw_bytes": 16 * (1 << 30),
            },
        }
        kv = {"raw_fp16_bytes": 512 * (1 << 20)}
        result = _capacity(weight, kv, 512, 512)
        self.assertEqual(result["maximum_zero_overhead_bits_per_value"], 12.0)
        self.assertEqual(result["maximum_practical_bits_per_value"], 11.0)


if __name__ == "__main__":
    unittest.main()
