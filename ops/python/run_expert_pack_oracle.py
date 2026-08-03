#!/usr/bin/env python3
"""Independent NumPy oracle for the OLMoE Expert Pack quantization profile."""

from __future__ import annotations

import argparse
import json
import mmap
import time
from pathlib import Path

import numpy as np


PROMPT = [510, 5347, 273, 6181, 310]


class Pack:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("rb")
        self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)

    def array(self, dtype: np.dtype, count: int, offset: int, shape: tuple[int, ...]) -> np.ndarray:
        return np.frombuffer(self.mapping, dtype=dtype, count=count, offset=offset).reshape(shape)


class Oracle:
    def __init__(self, root: Path, max_tokens: int) -> None:
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        architecture = self.manifest["architecture"]
        self.hidden = int(architecture["hidden_size"])
        self.intermediate = int(architecture["intermediate_size"])
        self.vocab = int(architecture["vocab_size"])
        self.layers = int(architecture["num_hidden_layers"])
        self.heads = int(architecture["num_attention_heads"])
        self.head_dim = int(architecture["head_dim"])
        self.experts = int(architecture["num_experts"])
        self.top_k = int(architecture["num_experts_per_token"])
        self.epsilon = np.float32(architecture["rms_norm_epsilon"])
        self.theta = np.float32(architecture["rope"]["theta"])
        if architecture["num_key_value_heads"] != self.heads or architecture["clip_qkv"] is not None:
            raise ValueError("oracle currently supports the canonical OLMoE MHA/clip_qkv=null contract")
        self.packs = {entry["name"]: Pack(root / entry["name"]) for entry in self.manifest["packs"]}
        self.tensors = {entry["name"]: entry for entry in self.manifest["tensors"]}
        self.expert_records = {(entry["layer"], entry["expert"]): entry for entry in self.manifest["experts"]}
        self.cache: dict[str, np.ndarray] = {}
        self.expert_cache: dict[tuple[int, int, str], np.ndarray] = {}
        self.keys = np.zeros((self.layers, max_tokens, self.heads, self.head_dim), dtype=np.float32)
        self.values = np.zeros_like(self.keys)
        half = self.head_dim // 2
        self.inverse_frequency = self.theta ** (
            -np.arange(0, self.head_dim, 2, dtype=np.float32) / np.float32(self.head_dim)
        )

    def tensor(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        entry = self.tensors[name]
        shape = tuple(int(value) for value in entry["source_shape"])
        base = int(entry["offset"])
        sections = entry["sections"]
        pack = self.packs[entry["pack"]]
        if entry["stored_dtype"] == "I8":
            count = int(np.prod(shape))
            quantized = pack.array(np.dtype("i1"), count, base + sections["data"]["offset"], shape)
            rows = shape[0] if len(shape) == 2 else 1
            scales = pack.array(np.dtype("<f4"), rows, base + sections["scales"]["offset"], (rows,))
            result = quantized.astype(np.float32) * scales.reshape((-1, 1))
            result = result.reshape(shape)
        else:
            count = int(np.prod(shape))
            result = pack.array(np.dtype("<f4"), count, base + sections["data"]["offset"], shape).copy()
        self.cache[name] = result
        return result

    def expert(self, layer: int, expert: int, section_name: str) -> np.ndarray:
        key = (layer, expert, section_name)
        if key in self.expert_cache:
            return self.expert_cache[key]
        entry = self.expert_records[(layer, expert)]
        section = entry["sections"]
        base = int(entry["offset"])
        pack = self.packs[entry["pack"]]
        if section_name == "gate_up":
            shape = (2 * self.intermediate, self.hidden)
            q_section, scale_section = section["gate_up_q"], section["gate_up_scales"]
        elif section_name == "down":
            shape = (self.hidden, self.intermediate)
            q_section, scale_section = section["down_q"], section["down_scales"]
        else:
            raise ValueError(section_name)
        rows, columns = shape
        quantized = pack.array(np.dtype("i1"), rows * columns, base + q_section["offset"], shape)
        scales = pack.array(np.dtype("<f4"), rows, base + scale_section["offset"], (rows, 1))
        result = quantized.astype(np.float32) * scales
        self.expert_cache[key] = result
        return result

    def norm(self, value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        variance = np.mean(value * value, dtype=np.float32)
        return value * np.float32(1.0 / np.sqrt(variance + self.epsilon)) * weight

    def forward(self, token: int, position: int) -> tuple[int, np.ndarray]:
        hidden = self.tensor("model.embed_tokens.weight")[token].copy()
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            normalized = self.norm(hidden, self.tensor(prefix + "input_layernorm.weight"))
            query = self.tensor(prefix + "self_attn.q_proj.weight") @ normalized
            key = self.tensor(prefix + "self_attn.k_proj.weight") @ normalized
            value = self.tensor(prefix + "self_attn.v_proj.weight") @ normalized
            query = self.norm(query, self.tensor(prefix + "self_attn.q_norm.weight"))
            key = self.norm(key, self.tensor(prefix + "self_attn.k_norm.weight"))
            query = query.reshape(self.heads, self.head_dim)
            key = key.reshape(self.heads, self.head_dim)
            value = value.reshape(self.heads, self.head_dim)
            angles = np.float32(position) * self.inverse_frequency
            cosine = np.concatenate((np.cos(angles), np.cos(angles))).astype(np.float32)
            sine = np.concatenate((np.sin(angles), np.sin(angles))).astype(np.float32)
            half = self.head_dim // 2
            query = query * cosine + np.concatenate((-query[:, half:], query[:, :half]), axis=1) * sine
            key = key * cosine + np.concatenate((-key[:, half:], key[:, :half]), axis=1) * sine
            self.keys[layer, position] = key
            self.values[layer, position] = value
            scores = np.einsum("hd,thd->ht", query, self.keys[layer, : position + 1], dtype=np.float32)
            scores *= np.float32(1.0 / np.sqrt(self.head_dim))
            scores -= scores.max(axis=1, keepdims=True)
            probabilities = np.exp(scores).astype(np.float32)
            probabilities /= probabilities.sum(axis=1, keepdims=True, dtype=np.float32)
            attention = np.einsum(
                "ht,thd->hd", probabilities, self.values[layer, : position + 1], dtype=np.float32
            ).reshape(self.hidden)
            hidden += self.tensor(prefix + "self_attn.o_proj.weight") @ attention
            normalized = self.norm(hidden, self.tensor(prefix + "post_attention_layernorm.weight"))
            logits = self.tensor(prefix + "mlp.gate.weight") @ normalized
            probabilities = np.exp(logits - logits.max()).astype(np.float32)
            probabilities /= probabilities.sum(dtype=np.float32)
            selected = np.argsort(-probabilities, kind="stable")[: self.top_k]
            moe = np.zeros(self.hidden, dtype=np.float32)
            for expert in selected:
                gate_up = self.expert(layer, int(expert), "gate_up") @ normalized
                gate, up = np.split(gate_up, 2)
                activated = (gate / (np.float32(1.0) + np.exp(-gate).astype(np.float32))) * up
                moe += probabilities[expert] * (self.expert(layer, int(expert), "down") @ activated)
            hidden += moe
        normalized = self.norm(hidden, self.tensor("model.norm.weight"))
        logits = self.tensor("lm_head.weight") @ normalized
        return int(np.argmax(logits)), logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("container", type=Path)
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--trace-logits", action="store_true")
    args = parser.parse_args()
    if args.new_tokens < 1:
        raise SystemExit("--new-tokens must be positive")
    oracle = Oracle(args.container, len(PROMPT) + args.new_tokens)
    predicted = 0
    for position, token in enumerate(PROMPT):
        predicted, logits = oracle.forward(token, position)
    full = list(PROMPT)
    started = time.perf_counter()
    for generated in range(args.new_tokens):
        full.append(predicted)
        if args.trace_logits:
            top = np.argsort(-logits, kind="stable")[:5]
            print(
                f"logits position={len(PROMPT) + generated} top="
                + ",".join(f"{int(index)}:{float(logits[index]):.7g}" for index in top),
                flush=True,
            )
        if generated + 1 < args.new_tokens:
            predicted, logits = oracle.forward(predicted, len(PROMPT) + generated)
    seconds = time.perf_counter() - started
    print(json.dumps({"tokens": full, "generated": args.new_tokens, "seconds": seconds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
