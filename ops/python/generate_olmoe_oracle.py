#!/usr/bin/env python3
"""Generate a deterministic CPU teacher reference from a local OLMoE snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "void")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.snapshot), local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(args.prompt, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    with torch.inference_mode():
        generated = model.generate(
            encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_ids = encoded["input_ids"][0].tolist()
    full_ids = generated[0].tolist()
    payload = {
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "metadata": {
            "snapshot": str(args.snapshot.resolve()),
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "dtype": "bfloat16",
            "device": "cpu",
            "generated_text": tokenizer.decode(
                full_ids[len(prompt_ids) :], skip_special_tokens=True
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
