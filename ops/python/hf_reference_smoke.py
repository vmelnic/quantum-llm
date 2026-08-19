#!/usr/bin/env python3
"""Run a bounded, local-only Hugging Face CUDA reference smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompt", default="hi")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not (args.snapshot / "config.json").is_file():
        raise ValueError(f"snapshot has no config.json: {args.snapshot}")
    if not (args.snapshot / "model.safetensors.index.json").is_file():
        raise ValueError(
            f"snapshot has no model.safetensors.index.json: {args.snapshot}"
        )

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    config = json.loads((args.snapshot / "config.json").read_text("utf-8"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
    input_tokens = int(encoded["input_ids"].shape[-1])
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    output_ids = generated[0, input_tokens:].tolist()
    result = {
        "schema_version": 1,
        "snapshot": str(args.snapshot),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "prompt": args.prompt,
        "input_tokens": input_tokens,
        "input_token_ids": encoded["input_ids"][0].tolist(),
        "output_tokens": len(output_ids),
        "output_token_ids": output_ids,
        "output_text": tokenizer.decode(output_ids, skip_special_tokens=True),
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": (
            len(output_ids) / generation_seconds if generation_seconds else None
        ),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
