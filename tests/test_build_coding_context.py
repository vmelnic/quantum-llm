from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ops.python.build_coding_context import exact_prompt_ids, repository_text


class CharacterTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "<system>" + messages[0]["content"] + "</system><user>" + \
            messages[1]["content"] + "</user><assistant>"

    def encode(self, text, add_special_tokens=False):
        self.assert_no_special_tokens = not add_special_tokens
        return [ord(character) for character in text]


class CodingContextTests(unittest.TestCase):
    def test_repository_capture_excludes_runtime_state_and_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime").mkdir()
            (root / "runtime" / "model.cpp").write_text("code\n")
            (root / "out").mkdir()
            (root / "out" / "binary.txt").write_text("not source")
            (root / ".env").write_text("SECRET=value")
            text, files = repository_text(root)
            self.assertIn("runtime/model.cpp", text)
            self.assertEqual([item["path"] for item in files],
                             ["runtime/model.cpp"])

    def test_exact_prompt_has_requested_populated_length(self):
        tokenizer = CharacterTokenizer()
        content = "real source text\n" * 100
        minimum, _, _ = exact_prompt_ids(tokenizer, content, 400)
        self.assertEqual(len(minimum), 400)


if __name__ == "__main__":
    unittest.main()
