from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path

try:
    from .capability_corpus import (
        LANGUAGES,
        assert_no_forbidden_overlap,
        load_capability_corpus,
        training_texts,
        validate_capability_manifest,
    )
    from .synthetic_memory import (
        UNKNOWN_ANSWER, corpus_fingerprint, format_memory, parse_response,
    )
    from .pow import build_epoch_training_schedule, training_geometry
except ImportError:
    from capability_corpus import (
        LANGUAGES,
        assert_no_forbidden_overlap,
        load_capability_corpus,
        training_texts,
        validate_capability_manifest,
    )
    from synthetic_memory import (
        UNKNOWN_ANSWER, corpus_fingerprint, format_memory, parse_response,
    )
    from pow import build_epoch_training_schedule, training_geometry


def _record(language: str, split: str, family: str, variant: int,
            answer: str) -> dict[str, object]:
    citation = f"MEM-C-{language.upper()}{split.upper()}{family}"
    return {
        "record_id": f"MEM-R-{language.upper()}{split.upper()}{family}{variant}",
        "citation_id": citation,
        "text": f"Authoritative memory. The recorded value is {answer}.",
        "shard_key": f"MEM-S-{language.upper()}{split.upper()}{family}",
        "generation": variant + 1,
        "indexable": variant == 0,
    }


def _fixture() -> list[dict[str, object]]:
    questions = {
        "ro": "Care este valoarea înregistrată?",
        "ru": "Какое значение записано?",
        "en": "What value was recorded?",
    }
    answers = {
        "ro": ("chihlimbar", "safir"),
        "ru": ("янтарь", "сапфир"),
        "en": ("amber", "sapphire"),
    }
    rows: list[dict[str, object]] = []
    for language in LANGUAGES:
        for split in ("train", "eval"):
            family = "A"
            question = questions[language] + (
                " Training case." if split == "train" else " Evaluation case."
            )
            records = [
                _record(language, split, family, variant, answer)
                for variant, answer in enumerate(answers[language])
            ]
            distractor = {
                "record_id": f"MEM-R-D-{language.upper()}{split.upper()}",
                "citation_id": f"MEM-C-D-{language.upper()}{split.upper()}",
                "text": "Independent memory with unrelated background material.",
                "shard_key": f"MEM-S-D-{language.upper()}{split.upper()}",
                "generation": 1,
                "indexable": True,
            }
            target_slot = 1 if language == "ru" else 0
            for variant, answer in enumerate(answers[language]):
                admitted = [records[variant], distractor]
                if target_slot == 1:
                    admitted.reverse()
                rows.append({
                    "schema_version": 1,
                    "family_id": f"MEM-F-{language.upper()}{split.upper()}{family}",
                    "example_id": f"MEM-E-{language.upper()}{split.upper()}{family}{variant}",
                    "split": split,
                    "language": language,
                    "question": question,
                    "answer": answer,
                    "answer_support": (
                        "dataset-label" if language == "ru" else "extractive"
                    ),
                    "citations": [records[variant]["citation_id"]],
                    "kind": "natural" if variant == 0 else "counterfactual",
                    "records": admitted,
                })
            rows.append({
                "schema_version": 1,
                "family_id": f"MEM-U-{language.upper()}{split.upper()}",
                "example_id": f"MEM-E-U-{language.upper()}{split.upper()}",
                "split": split,
                "language": language,
                "question": question + " Missing authority.",
                "answer": UNKNOWN_ANSWER,
                "answer_support": "absent",
                "citations": [],
                "kind": "unknown",
                "records": [records[0], distractor],
            })
    return rows


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="memory-capability-") as directory:
        path = Path(directory) / "fixture.jsonl"
        path.write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in _fixture()
        ), encoding="utf-8")
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps({
                "schema_version": 1,
                "rows": len(_fixture()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }) + "\n",
            encoding="utf-8",
        )
        validate_capability_manifest(path)
        records, examples = load_capability_corpus(path, 1, 1)
        second_records, _ = load_capability_corpus(path, 1, 1)
        assert corpus_fingerprint(records) == corpus_fingerprint(second_records)
        assert {record.language for record in records} == set(LANGUAGES)
        assert {example.language for example in examples} == set(LANGUAGES)
        assert all(len(example.memory_ids) == 2 for example in examples)
        assert all(example.family_id for example in examples)
        answerable = [example for example in examples if example.kind != "unknown"]
        assert {example.source_slots for example in answerable} == {(0,), (1,)}
        assert all(
            f"SOURCES: {example.source_slots[0]}" in example.target
            for example in answerable
        )
        assert all("CITATIONS:" not in example.target for example in examples)
        rendered = format_memory(records[:2])
        assert "SOURCE 0:" in rendered and "SOURCE 1:" in rendered
        assert parse_response("ANSWER: amber\nSOURCES: 1") == ("amber", (1,))
        train_questions = {
            (example.language, example.question) for example in examples
            if example.split == "train"
        }
        eval_questions = {
            (example.language, example.question) for example in examples
            if example.split == "eval"
        }
        assert train_questions.isdisjoint(eval_questions)
        train_examples = [
            example for example in examples if example.split == "train"
        ]
        geometry = training_geometry(
            len(train_examples), batch_size=2,
            gradient_accumulation=2, epochs=3,
        )
        assert geometry.examples_per_epoch == 9
        assert geometry.batches_per_epoch == 5
        assert geometry.optimizer_updates_per_epoch == 3
        assert geometry.total_examples == 27
        assert geometry.total_microsteps == 15
        assert geometry.total_optimizer_updates == 9
        schedule = build_epoch_training_schedule(
            train_examples, batch_size=2, gradient_accumulation=2,
            epochs=3, seed=20260807,
        )
        repeated = build_epoch_training_schedule(
            train_examples, batch_size=2, gradient_accumulation=2,
            epochs=3, seed=20260807,
        )
        assert [
            tuple(example.example_id for example in batch.examples)
            for batch in schedule
        ] == [
            tuple(example.example_id for example in batch.examples)
            for batch in repeated
        ]
        assert len(schedule) == geometry.total_microsteps
        assert sum(batch.optimizer_step for batch in schedule) == 9
        for epoch in range(1, 4):
            epoch_batches = [batch for batch in schedule if batch.epoch == epoch]
            visited = [
                example.example_id
                for batch in epoch_batches for example in batch.examples
            ]
            assert sorted(visited) == sorted(
                example.example_id for example in train_examples
            )
            assert [len(batch.examples) for batch in epoch_batches] == [2, 2, 2, 2, 1]
            assert [batch.accumulation_window_batches for batch in epoch_batches] == [
                2, 2, 2, 2, 1,
            ]
        resumed = build_epoch_training_schedule(
            train_examples, batch_size=2, gradient_accumulation=2,
            epochs=3, seed=20260807, start_epoch=2,
        )
        assert resumed[0].epoch == 3
        assert resumed[0].microstep == 11
        assert resumed[0].optimizer_update == 7
        assert resumed[-1].microstep == geometry.total_microsteps
        assert resumed[-1].optimizer_update == geometry.total_optimizer_updates
        corpus_text = training_texts(records, examples)
        assert_no_forbidden_overlap(corpus_text, ["unrelated benchmark sentence"])
        try:
            assert_no_forbidden_overlap(
                corpus_text, [corpus_text[0]], ngram_width=3
            )
        except ValueError:
            pass
        else:
            raise AssertionError("anti-leak canary missed an exact overlap")

        invalid = _fixture()
        invalid[0]["question"] = "Decode ZX123456Q."
        path.write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in invalid
        ), encoding="utf-8")
        try:
            load_capability_corpus(path, 1, 1)
        except ValueError as error:
            assert "shortcut" in str(error)
        else:
            raise AssertionError("synthetic shortcut was accepted")
    print("natural Memory Expert capability contract self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
