"""Build deterministic, group-split token traces for the Neural CPU experiment.

The generated CSV is consumed directly by ``--trace-moe``.  Complete source
documents are assigned to one split before chunking, so adjacent chunks from a
file can never leak across train and held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import AutoTokenizer


CURATED = (
    (
        "chat-ro",
        "Explică diferența dintre memorie virtuală, RAM și VRAM unui inginer "
        "care proiectează un runtime pentru modele MoE. Include două exemple.",
    ),
    (
        "chat-en",
        "A distributed service occasionally returns duplicate streaming chunks. "
        "Diagnose likely causes, propose an idempotent protocol, and discuss tradeoffs.",
    ),
    (
        "math-proof",
        "Prove that the sequence a_n=(1+1/n)^n is increasing and bounded. "
        "State every inequality you rely on and identify the limiting value.",
    ),
    (
        "math-discrete",
        "Find the number of length-12 binary strings with no three consecutive "
        "ones. Derive a recurrence and verify the initial conditions.",
    ),
    (
        "code-cpp",
        "Review a C++20 bounded binary protocol parser. Focus on integer overflow, "
        "allocation limits, ownership, cancellation, and deterministic errors.",
    ),
    (
        "code-python",
        "Write a Python iterator that merges timestamped event streams without "
        "loading them in memory. Explain stability, complexity, and malformed input.",
    ),
    (
        "systems",
        "Compare memory bandwidth, compute throughput, and latency as bottlenecks "
        "for autoregressive inference. Give a roofline-style argument.",
    ),
    (
        "science",
        "Describe how an experiment can distinguish a causal mechanism from a "
        "correlated proxy. Include positive controls, negative controls, and ablations.",
    ),
    (
        "long-form",
        "Draft a technical design review for a register machine that trades execution "
        "time for stored parameters. Separate falsifiable claims from speculation.",
    ),
    (
        "multilingual",
        "Rezumați în română, apoi în franceză și germană, avantajele și riscurile "
        "unui sistem distribuit care mută calculul către date.",
    ),
    (
        "structured",
        "Given an incident timeline, produce JSON with root_cause, contributing_factors, "
        "evidence, counterfactuals, and corrective_actions. Do not invent evidence.",
    ),
    (
        "creative",
        "Write a short dialogue between a compiler engineer and a physicist who "
        "disagree about whether recurrence is computation or compression.",
    ),
)

SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".md", ".py", ".ps1",
    ".sh", ".toml", ".yaml", ".yml",
}


@dataclass(frozen=True)
class Sequence:
    sequence_id: int
    group: str
    split: str
    tokens: int
    kind: str


def split_for(group: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:4], "little") % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def chunks(values: list[int], maximum: int) -> list[list[int]]:
    return [values[start : start + maximum] for start in range(0, len(values), maximum)]


def round_robin_groups(
    rows: list[tuple[str, str, str, list[int]]], count: int
) -> list[tuple[str, str, str, list[int]]]:
    grouped: dict[str, list[tuple[str, str, str, list[int]]]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)
    result: list[tuple[str, str, str, list[int]]] = []
    groups = sorted(grouped)
    depth = 0
    while len(result) < count:
        added = False
        for group in groups:
            values = grouped[group]
            if depth < len(values):
                result.append(values[depth])
                added = True
                if len(result) == count:
                    break
        if not added:
            break
        depth += 1
    return result


def encode_chat(tokenizer, text: str) -> list[int]:
    messages = [
        {"role": "system", "content": "Answer accurately and show the decisive reasoning."},
        {"role": "user", "content": text},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError("chat template unexpectedly returned a batch")
        encoded = encoded[0]
    return [int(token) for token in encoded]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--maximum-sequences", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 8:
        parser.error("--max-tokens must be at least 8")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    rows: list[tuple[str, str, str, list[int]]] = []
    for group, prompt in CURATED:
        token_ids = encode_chat(tokenizer, prompt)
        for index, part in enumerate(chunks(token_ids, args.max_tokens)):
            rows.append((group, split_for(group), f"curated:{index}", part))

    if not args.smoke:
        for path in sorted(args.repo.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in SOURCE_SUFFIXES
                or any(part in {".git", "out", "work", "artifacts"} for part in path.parts)
            ):
                continue
            relative = path.relative_to(args.repo).as_posix()
            try:
                text = path.read_text("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue
            prefix = (
                f"Analyze the following project document named {relative}. Preserve exact "
                "technical details and identify its contracts.\n\n"
            )
            token_ids = encode_chat(tokenizer, prefix + text)
            for index, part in enumerate(chunks(token_ids, args.max_tokens)):
                rows.append((relative, split_for(relative), f"source:{index}", part))

    if args.smoke:
        rows = rows[:1]
    elif args.maximum_sequences:
        held_out = max(1, args.maximum_sequences // 10)
        quotas = {
            "train": max(1, args.maximum_sequences - 2 * held_out),
            "validation": held_out,
            "test": held_out,
        }
        rows = [
            row
            for split in ("train", "validation", "test")
            for row in round_robin_groups(
                [candidate for candidate in rows if candidate[1] == split],
                quotas[split],
            )
        ]
    if not rows:
        raise RuntimeError("no prompt sequences were generated")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "prompts.csv"
    manifest_path = args.output / "prompts-manifest.json"
    metadata: list[Sequence] = []
    with csv_path.open("w", encoding="ascii", newline="\n") as handle:
        for sequence_id, (group, split, kind, token_ids) in enumerate(rows):
            handle.write(",".join(str(token) for token in token_ids) + "\n")
            metadata.append(Sequence(sequence_id, group, split, len(token_ids), kind))

    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    counts = {
        split: {
            "sequences": sum(item.split == split for item in metadata),
            "tokens": sum(item.tokens for item in metadata if item.split == split),
        }
        for split in ("train", "validation", "test")
    }
    manifest_path.write_text(
        json.dumps(
            {
                "format": "quantum-llm-neural-cpu-prompts-v1",
                "prompts_sha256": digest,
                "maximum_tokens_per_sequence": args.max_tokens,
                "counts": counts,
                "sequences": [asdict(item) for item in metadata],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "sha256": digest, "counts": counts}))


if __name__ == "__main__":
    main()
