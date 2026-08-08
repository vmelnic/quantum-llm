from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    from .data_contract import split_sentences
except ImportError:  # Direct execution.
    from data_contract import split_sentences


SOURCE_REPOSITORY = "osunlp/ConflictQA"
SOURCE_REVISION = "056384049e63c1ddae853891c24610fa07d85744"
SOURCE_FILE = "conflictQA-popQA-chatgpt.json"
SOURCE_SHA256 = "835f7d80d009d10b077551779c0decfae6ede4ef7cfcfdc0c5148eda30516a2f"
SOURCE_LICENSE = "apache-2.0"
CORPUS_CONTRACT = "conflictqa-causal-memory-v4"
ROW_SCHEMA_VERSION = 2
UNKNOWN_ANSWER = "I don't know from the attached memory."

_STOP_TOKENS = frozenset(
    "a an the is was were are be been being of in on at to for with by from as "
    "it its this that and or not no do does did have has had he she they we "
    "you i his her their our your my who what which where when how s".split()
)


def _content_tokens(value: str) -> list[str]:
    return [
        token for token in _normalized_tokens(value) if token not in _STOP_TOKENS
    ]


def _stem_match(token: str, sentence_tokens: set[str]) -> bool:
    if token in sentence_tokens:
        return True
    if len(token) < 5:
        return False
    return any(
        len(candidate) >= 5
        and (candidate.startswith(token[:5]) or token.startswith(candidate[:5]))
        for candidate in sentence_tokens
    )


def _derive_answer_span(text: str, answer: str, sibling_answer: str
                        ) -> int | None:
    """Sentence index carrying the answer, or None when not confidently derivable.

    Discriminating tokens are the answer's content tokens absent from the
    sibling answer of the same causal family. The winning sentence must
    contain all of them (exact first, stem match as fallback); ties resolve to
    the highest full-answer coverage, then to the earliest sentence.
    """
    answer_tokens = _normalized_tokens(answer)
    content = _content_tokens(answer)
    sibling = set(_content_tokens(sibling_answer))
    distinctive = [token for token in content if token not in sibling] or content
    if not distinctive:
        return None
    sentences = split_sentences(text)
    for matcher in (
        lambda token, tokens: token in tokens,
        _stem_match,
    ):
        candidates: list[tuple[int, float]] = []
        for index, sentence in enumerate(sentences):
            tokens = set(_normalized_tokens(sentence))
            if all(matcher(token, tokens) for token in distinctive):
                coverage = (
                    sum(1 for token in answer_tokens if token in tokens)
                    / max(1, len(answer_tokens))
                )
                candidates.append((index, coverage))
        if candidates:
            best = max(candidates, key=lambda item: (item[1], -item[0]))
            return best[0]
    return None


def _digest(value: str, width: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:width].upper()


def _opaque_id(kind: str, value: str) -> str:
    return f"MEM-{kind}-{_digest(value)}"


def _normalized_tokens(value: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    plain = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.findall(r"\w+", plain, re.UNICODE)


def _contains_tokens(text: str, fragment: str) -> bool:
    haystack, needle = _normalized_tokens(text), _normalized_tokens(fragment)
    return bool(needle) and any(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"source row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("ConflictQA source is empty")
    return rows


def _require_text(row: dict[str, object], key: str, index: int) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"ConflictQA row {index} has no {key}")
    return value


def _record(
    family_id: str,
    variant: int,
    citation_id: str,
    context: str,
    indexable: bool,
) -> dict[str, object]:
    return {
        "record_id": _opaque_id("R", f"{family_id}:{variant}:{context}"),
        "citation_id": citation_id,
        "text": context,
        "shard_key": _opaque_id("S", family_id),
        "generation": variant + 1,
        "indexable": indexable,
    }


def _split(question: str) -> str:
    return "eval" if int(_digest(f"split:{question}", 8), 16) % 5 == 0 else "train"


def _source_rows(path: Path) -> list[dict[str, str]]:
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(
            f"ConflictQA source hash mismatch: expected {SOURCE_SHA256}, got {actual_hash}"
        )
    prepared: list[dict[str, str]] = []
    for index, raw in enumerate(_read_jsonl(path)):
        question = _require_text(raw, "question", index)
        factual_answer = _require_text(raw, "memory_answer", index)
        counterfactual_answer = _require_text(raw, "counter_answer", index)
        factual_context = _require_text(raw, "parametric_memory", index)
        counterfactual_context = _require_text(raw, "counter_memory", index)
        if factual_answer.casefold() == counterfactual_answer.casefold():
            continue
        if factual_context.casefold() == counterfactual_context.casefold():
            continue
        if (
            _contains_tokens(factual_context, counterfactual_answer)
            or _contains_tokens(counterfactual_context, factual_answer)
        ):
            continue
        identity = json.dumps(
            [question, factual_answer, counterfactual_answer],
            ensure_ascii=False, separators=(",", ":"),
        )
        family_id = _opaque_id("F", identity)
        prepared.append({
            "family_id": family_id,
            "split": _split(" ".join(question.casefold().split())),
            "question": question,
            "factual_answer": factual_answer,
            "counterfactual_answer": counterfactual_answer,
            "factual_context": factual_context,
            "counterfactual_context": counterfactual_context,
        })
    return prepared


def _choose_distractor(
    candidates: list[dict[str, object]],
    start: int,
    question: str,
    answers: tuple[str, str],
) -> dict[str, object]:
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        text = str(candidate["text"])
        if (
            str(candidate["question"]) != question
            and not any(_contains_tokens(text, answer) for answer in answers)
        ):
            return candidate
    raise ValueError("cannot find a non-answering ConflictQA distractor")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def build(output: Path, source: Path, seed: int) -> dict[str, object]:
    prepared = _source_rows(source)
    if len(prepared) < 1_000:
        raise ValueError("ConflictQA source contains too few valid causal families")
    rng = random.Random(seed)
    distractors: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in prepared:
        citation_id = _opaque_id("C", f"citation:{item['family_id']}")
        record = _record(
            item["family_id"], 0, citation_id, item["factual_context"], True
        )
        distractors[item["split"]].append({
            **record,
            "question": item["question"],
        })

    output_rows: list[dict[str, object]] = []
    rejected_no_span = 0
    for family_index, item in enumerate(prepared):
        family_id = item["family_id"]
        split = item["split"]
        question = item["question"]
        answers = (item["factual_answer"], item["counterfactual_answer"])
        citation_id = _opaque_id("C", f"citation:{family_id}")
        target_records = (
            _record(family_id, 0, citation_id, item["factual_context"], True),
            _record(family_id, 1, citation_id, item["counterfactual_context"], False),
        )
        spans = (
            _derive_answer_span(
                item["factual_context"], answers[0], answers[1]
            ),
            _derive_answer_span(
                item["counterfactual_context"], answers[1], answers[0]
            ),
        )
        if any(span is None for span in spans):
            rejected_no_span += 1
            continue
        pool = distractors[split]
        distractor = _choose_distractor(
            pool, int(_digest(f"distractor:{family_id}", 8), 16) % len(pool),
            question, answers,
        )
        distractor = {
            key: value for key, value in distractor.items() if key != "question"
        }
        target_slot = int(_digest(f"slot:{family_id}", 2), 16) % 2
        for variant, (answer, target, kind) in enumerate(zip(
            answers, target_records, ("natural", "counterfactual"), strict=True
        )):
            records = [distractor, target]
            if target_slot == 0:
                records.reverse()
            output_rows.append({
                "schema_version": ROW_SCHEMA_VERSION,
                "family_id": family_id,
                "example_id": _opaque_id("E", f"{family_id}:{variant}"),
                "split": split,
                "language": "en",
                "question": question,
                "answer": answer,
                "answer_support": "dataset-label",
                "answer_span": [target_slot, spans[variant]],
                "citations": [citation_id],
                "kind": kind,
                "records": records,
            })

        if family_index % 10 == 0:
            second = _choose_distractor(
                pool, (int(_digest(f"unknown:{family_id}", 8), 16) + 1) % len(pool),
                question, answers,
            )
            second = {key: value for key, value in second.items() if key != "question"}
            if second["record_id"] == distractor["record_id"]:
                continue
            unknown_family = _opaque_id("U", family_id)
            output_rows.append({
                "schema_version": ROW_SCHEMA_VERSION,
                "family_id": unknown_family,
                "example_id": _opaque_id("E", unknown_family),
                "split": split,
                "language": "en",
                "question": question,
                "answer": UNKNOWN_ANSWER,
                "answer_support": "absent",
                "answer_span": None,
                "citations": [],
                "kind": "unknown",
                "records": [distractor, second],
            })

    rng.shuffle(output_rows)
    row_count = _write_jsonl(output, output_rows)
    distribution = Counter(
        (str(row["split"]), str(row["language"]), str(row["kind"]))
        for row in output_rows
    )
    manifest = {
        "schema_version": 3,
        "contract": CORPUS_CONTRACT,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "source_license": SOURCE_LICENSE,
        "languages": ["en"],
        "causal_families": len(prepared),
        "families_rejected_no_span": rejected_no_span,
        "pointer_target": "sentence-v1",
        "rows": row_count,
        "examples_by_split_language_kind": {
            "/".join(key): value for key, value in sorted(distribution.items())
        },
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_built_corpus(path: Path) -> dict[str, object]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("built capability corpus or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 3
        or manifest.get("contract") != CORPUS_CONTRACT
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("capability corpus is stale or uses a superseded contract")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a causal Memory Expert corpus from pinned ConflictQA"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_built_corpus(args.output.resolve()), indent=2))
        return 0
    if args.source is None:
        parser.error("--source is required unless --validate-only is used")
    print(json.dumps(
        build(args.output.resolve(), args.source.resolve(), args.seed), indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
