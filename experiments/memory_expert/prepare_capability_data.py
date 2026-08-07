from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASET = "google/xquad"
REVISION = "51adfef1c1287aab1d2d91b5bead9bcfb9c68583"
LANGUAGES = ("ro", "ru", "en")
UNKNOWN_ANSWER = "I don't know from the attached memory."


def _digest(value: str, width: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:width].upper()


def _opaque_id(kind: str, value: str) -> str:
    return f"MEM-{kind}-{_digest(value)}"


def _fetch_language(language: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode({
            "dataset": DATASET,
            "config": f"xquad.{language}",
            "split": "validation",
            "offset": offset,
            "length": 100,
        })
        with urllib.request.urlopen(
            f"https://datasets-server.huggingface.co/rows?{query}", timeout=60
        ) as response:
            payload = json.load(response)
        page = [item["row"] for item in payload["rows"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= int(payload["num_rows_total"]):
            break
    return rows


def _answer(row: dict[str, object]) -> tuple[str, int]:
    answers = row["answers"]
    assert isinstance(answers, dict)
    texts = answers["text"]
    starts = answers["answer_start"]
    assert isinstance(texts, list) and isinstance(starts, list) and texts and starts
    text = str(texts[0]).strip()
    start = int(starts[0])
    context = str(row["context"])
    if context[start:start + len(text)] != text:
        start = context.find(text)
    if start < 0 or not text:
        raise ValueError(f"cannot locate answer for {row['id']}")
    return text, start


def _shape(answer: str) -> tuple[str, int]:
    tokens = answer.split()
    if any(character.isdigit() for character in answer):
        kind = "numeric"
    elif tokens and all(token[:1].isupper() for token in tokens if token[:1].isalpha()):
        kind = "named"
    else:
        kind = "phrase"
    return kind, min(4, max(1, len(tokens)))


def _counterfactual_answer(
    rng: random.Random,
    original: str,
    context: str,
    pool: dict[tuple[str, int], list[str]],
) -> str:
    folded_context = context.casefold()
    candidates = [
        item for item in pool[_shape(original)]
        if item != original and item.casefold() not in folded_context
    ]
    if not candidates:
        candidates = [
            item for values in pool.values() for item in values
            if item != original and item.casefold() not in folded_context
        ]
    if not candidates:
        raise ValueError("cannot construct a counterfactual answer pool")
    return rng.choice(candidates)


def _answer_window(context: str, start: int, answer_length: int,
                   flank_characters: int = 220) -> tuple[str, int]:
    """Select a natural bounded passage while preserving the exact answer span."""
    left = max(0, start - flank_characters)
    right = min(len(context), start + answer_length + flank_characters)
    if left:
        boundary = context.find(" ", left, min(start, left + 48))
        if boundary >= 0:
            left = boundary + 1
    if right < len(context):
        boundary = context.rfind(" ", max(start + answer_length, right - 48), right)
        if boundary >= 0:
            right = boundary
    window = context[left:right].strip()
    local_start = window.find(context[start:start + answer_length])
    if local_start < 0:
        raise ValueError("answer window lost the source span")
    return window, local_start


def _record(
    language: str,
    source_id: str,
    variant: int,
    citation_id: str,
    context: str,
    indexable: bool,
) -> dict[str, object]:
    record_id = _opaque_id("R", f"{language}:{source_id}:{variant}:{context}")
    return {
        "record_id": record_id,
        "citation_id": citation_id,
        "text": f"Memory record {citation_id}. {context}",
        "shard_key": _opaque_id("S", f"{language}:{source_id}"),
        "generation": variant + 1,
        "indexable": indexable,
    }


def _source_splits(
    by_language: dict[str, list[dict[str, object]]]
) -> dict[str, str]:
    """Keep translated rows and duplicate questions in one atomic split."""
    source_ids = {
        str(row["id"]) for rows in by_language.values() for row in rows
    }
    parent = {source_id: source_id for source_id in source_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    for rows in by_language.values():
        by_question: dict[str, str] = {}
        for row in rows:
            source_id = str(row["id"])
            question = " ".join(str(row["question"]).casefold().split())
            previous = by_question.setdefault(question, source_id)
            union(previous, source_id)
    components: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        components[find(source_id)].append(source_id)
    splits: dict[str, str] = {}
    for members in components.values():
        key = min(members)
        split = "eval" if int(_digest(key, 8), 16) % 5 == 0 else "train"
        for source_id in members:
            splits[source_id] = split
    return splits


def build(output: Path, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    source_sha = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{DATASET}", timeout=60
    ))["sha"]
    if source_sha != REVISION:
        raise RuntimeError(f"dataset revision changed: expected {REVISION}, got {source_sha}")
    by_language = {language: _fetch_language(language) for language in LANGUAGES}
    source_splits = _source_splits(by_language)
    answer_pools: dict[str, dict[tuple[str, int], list[str]]] = {}
    for language, rows in by_language.items():
        pool: dict[tuple[str, int], list[str]] = defaultdict(list)
        for row in rows:
            answer, _ = _answer(row)
            pool[_shape(answer)].append(answer)
        answer_pools[language] = pool

    output_rows: list[dict[str, object]] = []
    records_for_distractors: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    prepared: list[tuple[str, str, dict[str, object], str, int, str, str]] = []
    for language, rows in by_language.items():
        for row in rows:
            source_id = str(row["id"])
            split = source_splits[source_id]
            answer, start = _answer(row)
            context, local_start = _answer_window(
                str(row["context"]), start, len(answer)
            )
            citation_id = _opaque_id("C", f"citation:{language}:{source_id}")
            original = _record(
                language, source_id, 0, citation_id, context, True
            )
            records_for_distractors[(language, split)].append(original)
            prepared.append((
                language, split, row, answer, local_start, citation_id, context
            ))

    for language, split, row, answer, start, citation_id, context in prepared:
        source_id = str(row["id"])
        question = str(row["question"]).strip()
        family_id = _opaque_id("F", f"family:{language}:{source_id}")
        original = _record(language, source_id, 0, citation_id, context, True)
        alternate = _counterfactual_answer(
            rng, answer, context, answer_pools[language]
        )
        alternate_context = context[:start] + alternate + context[start + len(answer):]
        counterfactual = _record(
            language, source_id, 1, citation_id, alternate_context, False
        )
        distractors = [
            item for item in records_for_distractors[(language, split)]
            if item["citation_id"] != citation_id
            and answer.casefold() not in str(item["text"]).casefold()
            and alternate.casefold() not in str(item["text"]).casefold()
        ]
        if not distractors:
            raise ValueError("cannot construct a non-answering distractor")
        distractor = rng.choice(distractors)
        for variant, target, target_answer, kind in (
            (0, original, answer, "natural"),
            (1, counterfactual, alternate, "counterfactual"),
        ):
            output_rows.append({
                "schema_version": 1,
                "family_id": family_id,
                "example_id": _opaque_id("E", f"{family_id}:{variant}"),
                "split": split,
                "language": language,
                "question": question,
                "answer": target_answer,
                "citations": [citation_id],
                "kind": kind,
                "records": [target, distractor],
            })

    for language, rows in by_language.items():
        for index, row in enumerate(rows[::10]):
            source_id = str(row["id"])
            split = source_splits[source_id]
            original_answer, _ = _answer(row)
            distractors = [
                item for item in records_for_distractors[(language, split)]
                if original_answer.casefold() not in str(item["text"]).casefold()
            ]
            if len(distractors) < 2:
                raise ValueError("cannot construct an unanswerable memory set")
            first = rng.choice(distractors)
            second = rng.choice([
                item for item in distractors if item["record_id"] != first["record_id"]
            ])
            family_id = _opaque_id("U", f"unknown:{language}:{source_id}:{index}")
            output_rows.append({
                "schema_version": 1,
                "family_id": family_id,
                "example_id": _opaque_id("E", family_id),
                "split": split,
                "language": language,
                "question": str(row["question"]).strip(),
                "answer": UNKNOWN_ANSWER,
                "citations": [],
                "kind": "unknown",
                "records": [first, second],
            })

    rng.shuffle(output_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_license": "cc-by-sa-4.0",
        "languages": list(LANGUAGES),
        "rows": len(output_rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare natural multilingual Memory Expert QA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve(), args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
