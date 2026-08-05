#!/usr/bin/env python3
"""Encode/decode DeepSeek-V4 prompts with the checkpoint's official codec."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from tokenizers import Tokenizer


def _encoder(snapshot: Path):
    path = snapshot / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official DeepSeek encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    encode = subparsers.add_parser("encode")
    encode.add_argument("--prompt", required=True)
    encode.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    encode.add_argument("--output", type=Path, required=True)
    decode = subparsers.add_parser("decode")
    decode.add_argument("--tokens", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))
    if args.command == "encode":
        official = _encoder(args.snapshot)
        text = official.encode_messages(
            [{"role": "user", "content": args.prompt}],
            thinking_mode=args.thinking_mode,
        )
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        args.output.write_text("\n".join(str(token) for token in ids) + "\n", encoding="ascii")
        print(json.dumps({"token_ids": ids, "prompt_tokens": len(ids)}))
        return

    ids = [int(value) for value in args.tokens.split(",") if value]
    print(json.dumps({"token_ids": ids, "text": tokenizer.decode(ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
