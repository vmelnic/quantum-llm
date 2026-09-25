#!/usr/bin/env python3
"""Run a bounded, local-only Hugging Face reference corpus.

The reference backend is selected from checkpoint capabilities.  It uses the
immutable source snapshot, its tokenizer and chat template, and corpus-declared
generation settings.  It never downloads model files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


_MAXIMUM_NEW_TOKENS = 32_768


class _PresencePenalty:
    """OpenAI presence semantics over generated tokens only."""

    def __init__(self, prompt_tokens: int, penalty: float) -> None:
        self.prompt_tokens = prompt_tokens
        self.penalty = penalty

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        emitted = input_ids[:, self.prompt_tokens:]
        for row in range(emitted.shape[0]):
            if emitted.shape[1]:
                scores[row, emitted[row].unique()] -= self.penalty
        return scores


class _ProgressStreamer:
    """Persist bounded generated-token checkpoints during a long reference run."""

    def __init__(self, tokenizer: Any, prompt_ids: list[int], output: Path,
                 case_id: str, interval: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_ids = prompt_ids
        self.output = output
        self.case_id = case_id
        self.interval = interval
        self.output_ids: list[int] = []
        self.received_prompt = False

    def _publish(self) -> None:
        document = {
            "schema_version": 1,
            "case_id": self.case_id,
            "output_tokens": len(self.output_ids),
            "output_token_ids": self.output_ids,
            "raw_text": self.tokenizer.decode(
                self.output_ids, skip_special_tokens=False,
            ),
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        partial = self.output.with_name(self.output.name + ".partial")
        partial.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        partial.replace(self.output)
        print(
            f"reference_progress case={self.case_id} "
            f"output_tokens={len(self.output_ids)}",
            flush=True,
        )

    def put(self, value: Any) -> None:
        tokens = [int(token) for token in value.detach().cpu().reshape(-1)]
        if not self.received_prompt:
            if tokens != self.prompt_ids:
                raise RuntimeError("reference streamer prompt token mismatch")
            self.received_prompt = True
            return
        self.output_ids.extend(tokens)
        if len(self.output_ids) % self.interval == 0:
            self._publish()

    def end(self) -> None:
        self._publish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gpu-memory", default="20GiB")
    parser.add_argument("--max-cpu-memory", default="42GiB")
    parser.add_argument("--offload-directory", type=Path, required=True)
    parser.add_argument("--progress-every-tokens", type=int, default=128)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _reasoning_and_content(text: str) -> tuple[str, str]:
    text = text.replace("<|im_end|>", "").strip()
    marker = "</think>"
    if marker not in text:
        return "", text
    reasoning, content = text.split(marker, 1)
    reasoning = reasoning.removeprefix("<think>").strip()
    return reasoning, content.strip()


def _device_for_inputs(model: Any) -> Any:
    embeddings = model.get_input_embeddings()
    device = getattr(getattr(embeddings, "weight", None), "device", None)
    if device is not None and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type not in {"meta", "cpu"}:
            return parameter.device
    return next(model.parameters()).device


def _token_digest(token_ids: list[int]) -> str:
    payload = b"".join(int(token).to_bytes(4, "little") for token in token_ids)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot.resolve()
    if not (snapshot / "config.json").is_file():
        raise ValueError(f"snapshot has no config.json: {snapshot}")
    if not (snapshot / "model.safetensors.index.json").is_file():
        raise ValueError(f"snapshot has no SafeTensors index: {snapshot}")
    corpus = _load_json(args.corpus.resolve())
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("behavior corpus has no cases")
    if args.progress_every_tokens <= 0:
        raise ValueError("progress token interval must be positive")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the isolated reference environment has no CUDA")

    config = _load_json(snapshot / "config.json")
    multimodal = bool(config.get("vision_config")) and not bool(
        config.get("language_model_only", False)
    )
    model_class = (
        AutoModelForMultimodalLM if multimodal else AutoModelForCausalLM
    )
    args.offload_directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = model_class.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: args.max_gpu_memory, "cpu": args.max_cpu_memory},
        offload_folder=str(args.offload_directory),
        offload_state_dict=True,
    )
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    input_device = _device_for_inputs(model)

    results: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("behavior corpus case is invalid")
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"case {case['id']} has no messages")
        tools = case.get("tools") or None
        maximum = int(case.get("maximum_new_tokens", 128))
        if maximum <= 0 or maximum > _MAXIMUM_NEW_TOKENS:
            raise ValueError(f"case {case['id']} has unsafe output limit")
        do_sample = bool(case.get("do_sample", False))
        temperature = float(case.get("temperature", 1.0))
        top_p = float(case.get("top_p", 1.0))
        top_k = int(case.get("top_k", 0))
        presence_penalty = float(case.get("presence_penalty", 0.0))
        seed = int(case.get("seed", 0))
        if not 0.0 < temperature or not 0.0 < top_p <= 1.0:
            raise ValueError(f"case {case['id']} has invalid sampling bounds")
        if top_k < 0 or not -2.0 <= presence_penalty <= 2.0 or seed < 0:
            raise ValueError(f"case {case['id']} has invalid sampling settings")
        template_kwargs = {
            "reasoning_effort": case.get("reasoning_effort", "xhigh"),
            "enable_thinking": bool(case.get("enable_thinking", True)),
            "preserve_thinking": True,
        }
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        input_ids = encoded["input_ids"][0].tolist()
        model_inputs = {
            name: value.to(input_device) for name, value in encoded.items()
        }
        progress = args.output.resolve().with_name(
            args.output.name + f".{case['id']}.progress.json"
        )
        streamer = _ProgressStreamer(
            tokenizer, input_ids, progress, case["id"],
            args.progress_every_tokens,
        )
        logits_processors = []
        if presence_penalty != 0.0:
            logits_processors.append(_PresencePenalty(
                len(input_ids), presence_penalty,
            ))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generation_options: dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": maximum,
            "use_cache": True,
            "streamer": streamer,
            "logits_processor": logits_processors,
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        }
        if do_sample:
            generation_options.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                **generation_options,
            )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        output_ids = generated[0, len(input_ids):].tolist()
        raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        reasoning, content = _reasoning_and_content(raw_text)
        results.append({
            "id": case["id"],
            "input_tokens": len(input_ids),
            "input_token_ids_sha256": _token_digest(input_ids),
            "output_tokens": len(output_ids),
            "output_token_ids": output_ids,
            "raw_text": raw_text,
            "reasoning_content": reasoning,
            "content": content,
            "generation_seconds": generation_seconds,
            "sampling": {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "rng_backend": "torch.manual_seed",
            },
        })

    document = {
        "schema_version": 1,
        "backend": "official-huggingface-bf16",
        "snapshot": str(snapshot),
        "source_revision": snapshot.name,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "model_class": model_class.__name__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "device_map": getattr(model, "hf_device_map", None),
        "load_seconds": load_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(args.output)
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
