from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .capability_corpus import load_capability_corpus
    from .data_contract import read_records
    from .mechanism_probe import select_ood_record
    from .pow import (
        MemoryHook,
        PowConfig,
        candidate_nll,
        load_adapter,
        load_model,
        resolve_layers,
        seed_everything,
        encode_memory_sets,
    )
    from .real_query import read_questions
    from .synthetic_memory import (
        MemoryExample,
        MemoryRecord,
        normalized_contains,
    )
except ImportError:  # Direct execution on a worker.
    from capability_corpus import load_capability_corpus
    from data_contract import read_records
    from mechanism_probe import select_ood_record
    from pow import (
        MemoryHook,
        PowConfig,
        candidate_nll,
        load_adapter,
        load_model,
        resolve_layers,
        seed_everything,
        encode_memory_sets,
    )
    from real_query import read_questions
    from synthetic_memory import (
        MemoryExample,
        MemoryRecord,
        normalized_contains,
    )


def split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in text.replace("\n", " ").split(". ")
        if sentence.strip()
    ]
    return sentences or [text]


def rank_candidates(model, tokenizer, hooks, memory_cache,
                    memory_ids: tuple[str, ...], question: str,
                    candidates: list[str], gold_index: int,
                    device: torch.device) -> dict[str, object]:
    examples = [
        MemoryExample(
            example_id=f"candidate-{index}",
            question=question,
            answer=candidate,
            citations=(),
            memory_ids=memory_ids,
            kind="ranking",
            split="eval",
            source_slots=(0,),
            answer_support="dataset-label",
        )
        for index, candidate in enumerate(candidates)
    ]
    losses = candidate_nll(
        model, tokenizer, hooks, memory_cache, memory_ids, examples, device
    )
    order = sorted(range(len(candidates)), key=lambda index: losses[index])
    return {
        "gold_rank": order.index(gold_index) + 1,
        "gold_nll": losses[gold_index],
        "best_nll": losses[order[0]],
        "best_candidate": candidates[order[0]],
        "ranking": [
            {"candidate": candidates[index], "nll": losses[index]}
            for index in order
        ],
    }


def run_span_rank(checkpoint_path: Path, corpus_path: Path, ingest_root: Path,
                  questions_path: Path, output_path: Path, ood_cases: int,
                  device: torch.device) -> dict[str, object]:
    checkpoint_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PowConfig(**checkpoint_raw["config"])
    seed_everything(config.seed)
    # The corpus is loaded only for checkpoint fingerprint compatibility with
    # the other mechanism tools; ranking uses OOD oracle records only.
    load_capability_corpus(corpus_path)
    ood_knowledge = list(read_records(ingest_root))
    questions = read_questions(questions_path)

    tokenizer, model = load_model(config, device)
    layers = resolve_layers(model)
    checkpoint, _, adapter = load_adapter(checkpoint_path, model, device)
    layer_indices = tuple(int(index) for index in checkpoint["layer_indices"])
    hooks = {index: MemoryHook(adapter.adapter(index)) for index in layer_indices}
    handles = [
        layers[index].register_forward_hook(hooks[index]) for index in layer_indices
    ]

    ood_rows: list[tuple[dict[str, object], tuple[str, ...], str]] = []
    records_by_id: dict[str, MemoryRecord] = {}
    for row in questions[:ood_cases]:
        patterns = row.get("expected_sections", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        record = select_ood_record(ood_knowledge, [str(value) for value in patterns])
        record_id = f"OOD::{record.record_id}"
        records_by_id[record_id] = MemoryRecord(
            record_id=record_id,
            citation_id=record.citation_id,
            text=record.text,
            shard_key=f"ood:{record.document_id}",
            language=record.language,
            indexable=False,
        )
        expected = row.get("expected_answers", [])
        if isinstance(expected, str):
            expected = [expected]
        expected = [str(value) for value in expected]
        if not expected:
            raise ValueError(f"question {row['id']} has no expected answers")
        ood_rows.append((row, (record_id,), expected[0]))

    all_gold = [gold for _, _, gold in ood_rows]
    try:
        memory_cache = encode_memory_sets(
            model, tokenizer, hooks, records_by_id,
            [memory_ids for _, memory_ids, _ in ood_rows],
            config.maximum_memory_tokens, device,
        )
        cases: list[dict[str, object]] = []
        for index, (row, memory_ids, gold) in enumerate(ood_rows):
            question = str(row["question"])
            literal_test = rank_candidates(
                model, tokenizer, hooks, memory_cache, memory_ids, question,
                all_gold, index, device,
            )
            # No-memory control: identical ranking with the empty memory set.
            # If the gold rank survives this, the test measures question-type
            # matching rather than memory reading.
            control_test = rank_candidates(
                model, tokenizer, hooks, memory_cache, (), question,
                all_gold, index, device,
            )
            sentences = split_sentences(records_by_id[memory_ids[0]].text)
            gold_sentences = [
                position for position, sentence in enumerate(sentences)
                if normalized_contains(sentence, gold)
            ]
            sentence_test = None
            if gold_sentences and len(sentences) > 1:
                sentence_test = rank_candidates(
                    model, tokenizer, hooks, memory_cache, memory_ids, question,
                    sentences, gold_sentences[0], device,
                )
            case = {
                "case_id": str(row["id"]),
                "question": question,
                "gold": gold,
                "literal_gold_rank": literal_test["gold_rank"],
                "literal_best": literal_test["best_candidate"],
                "literal_gold_rank_no_memory": control_test["gold_rank"],
                "literal_best_no_memory": control_test["best_candidate"],
                "literal_ranking": literal_test["ranking"],
                "sentence_gold_rank": (
                    sentence_test["gold_rank"] if sentence_test else None
                ),
                "sentence_count": len(sentences),
            }
            cases.append(case)
            print(json.dumps({
                "event": "mechanism-span-rank-case",
                **{key: value for key, value in case.items()
                   if key != "literal_ranking"},
            }, ensure_ascii=False), flush=True)

        literal_ranks = [case["literal_gold_rank"] for case in cases]
        control_ranks = [case["literal_gold_rank_no_memory"] for case in cases]
        sentence_ranks = [
            case["sentence_gold_rank"] for case in cases
            if case["sentence_gold_rank"] is not None
        ]
        result = {
            "schema_version": 1,
            "contract": "quantum-llm-memory-span-rank-v1",
            "checkpoint": str(checkpoint_path),
            "cases": cases,
            "literal_rank1": sum(1 for rank in literal_ranks if rank == 1),
            "literal_total": len(literal_ranks),
            "literal_mean_rank": sum(literal_ranks) / max(1, len(literal_ranks)),
            "literal_rank1_no_memory": sum(
                1 for rank in control_ranks if rank == 1
            ),
            "literal_mean_rank_no_memory": (
                sum(control_ranks) / max(1, len(control_ranks))
            ),
            "sentence_rank1": sum(1 for rank in sentence_ranks if rank == 1),
            "sentence_total": len(sentence_ranks),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        for handle in handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the memory channel can rank the gold span first"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ood-cases", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_span_rank(
        args.checkpoint.resolve(), args.corpus.resolve(), args.ingest.resolve(),
        args.questions.resolve(), args.output.resolve(), args.ood_cases,
        torch.device(args.device),
    )
    print(json.dumps({
        "event": "mechanism-span-rank-summary",
        **{key: value for key, value in result.items() if key != "cases"},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
