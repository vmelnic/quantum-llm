from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .capability_corpus import (
        assert_no_forbidden_overlap,
        load_capability_corpus,
        training_texts,
        validate_capability_manifest,
    )
    from .pow import (
        PowConfig, causal_probe, evaluate, train, training_geometry,
    )
except ImportError:  # Direct execution on a worker.
    from capability_corpus import (
        assert_no_forbidden_overlap,
        load_capability_corpus,
        training_texts,
        validate_capability_manifest,
    )
    from pow import PowConfig, causal_probe, evaluate, train, training_geometry


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def read_forbidden_files(paths: Sequence[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        try:
            if path.suffix.lower() == ".jsonl":
                values = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                values = json.loads(raw)
            texts.extend(_walk_strings(values))
        except json.JSONDecodeError:
            texts.append(raw)
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and gate the multilingual Memory Expert operation"
    )
    parser.add_argument(
        "action", choices=("validate", "train", "probe", "evaluate", "run")
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--validation-limit", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-limit", type=int, default=30)
    parser.add_argument("--gate-rank", type=int, default=16)
    parser.add_argument("--gate-alpha", type=float, default=32.0)
    parser.add_argument("--knowledge-dropout", type=float, default=0.2)
    parser.add_argument("--maximum-memory-tokens", type=int, default=384)
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--forbidden-file", type=Path, action="append", default=[])
    parser.add_argument("--minimum-train-families", type=int, default=500)
    parser.add_argument("--minimum-eval-families", type=int, default=100)
    parser.add_argument("--memory-cache-bytes", type=int, default=4 << 30)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PowConfig(
        model=args.model,
        revision=args.revision,
        gate_rank=args.gate_rank,
        gate_alpha=args.gate_alpha,
        injection_every=1,
        knowledge_dropout=args.knowledge_dropout,
        maximum_memory_tokens=args.maximum_memory_tokens,
        maximum_new_tokens=args.maximum_new_tokens,
        seed=args.seed,
    )
    manifest = validate_capability_manifest(
        args.corpus.resolve(), "conflictqa-causal-memory-v4"
    )
    corpus = load_capability_corpus(
        args.corpus.resolve(),
        minimum_train_families_per_language=args.minimum_train_families,
        minimum_eval_families_per_language=args.minimum_eval_families,
    )
    forbidden = read_forbidden_files(args.forbidden_file)
    assert_no_forbidden_overlap(
        training_texts(*corpus), forbidden, ngram_width=8
    )
    if args.action == "validate":
        records, examples = corpus
        training_examples = [
            example for example in examples if example.split == "train"
        ]
        geometry = training_geometry(
            len(training_examples), args.batch_size,
            args.gradient_accumulation, args.epochs,
        )
        print(json.dumps({
            "event": "capability-validation-summary",
            "contract": manifest["contract"],
            "corpus_sha256": manifest["sha256"],
            "records": len(records),
            "examples": len(examples),
            "training_examples": len(training_examples),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "batches_per_epoch": geometry.batches_per_epoch,
            "optimizer_updates_per_epoch": geometry.optimizer_updates_per_epoch,
            "total_examples_seen": geometry.total_examples,
            "total_microsteps": geometry.total_microsteps,
            "total_optimizer_updates": geometry.total_optimizer_updates,
        }, indent=2))
        return 0
    checkpoint = args.checkpoint or args.output / "memory-expert.pt"
    if args.action in ("train", "run"):
        result = train(
            config, args.output, args.epochs, args.batch_size,
            args.learning_rate, args.gradient_accumulation, "adamw",
            corpus=corpus, corpus_name="conflictqa-causal-memory-v4",
            augment_examples=False,
            memory_cache_bytes=args.memory_cache_bytes,
            validation_limit=args.validation_limit,
            resume=args.resume,
        )
        print(json.dumps({
            "event": "capability-train-summary",
            "corpus_sha256": manifest["sha256"],
            **result,
        }, indent=2))
    if args.action in ("probe", "run"):
        result = causal_probe(checkpoint, args.output, family_limit=6, corpus=corpus)
        print(json.dumps({"event": "capability-probe-summary", **result}, indent=2))
        if args.action == "run" and not result["passed"]:
            raise RuntimeError("natural multilingual causal probe failed")
    if args.action in ("evaluate", "run"):
        result = evaluate(checkpoint, args.output, args.evaluation_limit, corpus=corpus)
        print(json.dumps({"event": "capability-evaluation-summary", **result}, indent=2))
        if args.action == "run" and not result["pow_passed"]:
            raise RuntimeError("natural multilingual held-out gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
