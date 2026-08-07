from __future__ import annotations

import json
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .synthetic_memory import (
        MemoryExample,
        MemoryRecord,
        UNKNOWN_ANSWER,
        normalized_contains,
    )
except ImportError:
    from synthetic_memory import (
        MemoryExample,
        MemoryRecord,
        UNKNOWN_ANSWER,
        normalized_contains,
    )


LANGUAGES = ("ro", "ru", "en")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_FORBIDDEN_SHORTCUT = re.compile(
    r"(?:ZX\d{4,}Q|SPAN-\d+|MISSING-(?:train|eval))", re.IGNORECASE
)


def normalized_ngrams(text: str, width: int) -> set[tuple[str, ...]]:
    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    return {
        tuple(tokens[index:index + width])
        for index in range(max(0, len(tokens) - width + 1))
    }


def assert_no_forbidden_overlap(
    training_texts: Iterable[str],
    forbidden_texts: Iterable[str],
    ngram_width: int = 8,
) -> None:
    if ngram_width < 3:
        raise ValueError("anti-leak n-grams shorter than three tokens are too broad")
    forbidden = set().union(*(
        normalized_ngrams(text, ngram_width) for text in forbidden_texts
    ))
    if not forbidden:
        return
    for value in training_texts:
        overlap = normalized_ngrams(value, ngram_width).intersection(forbidden)
        if overlap:
            sample = " ".join(next(iter(overlap)))
            raise ValueError(f"training/evaluation leakage detected: {sample!r}")


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"capability row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("capability corpus is empty")
    return rows


def validate_capability_manifest(
    path: Path, expected_contract: str | None = None
) -> dict[str, object]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"capability manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in (1, 2):
        raise ValueError("invalid capability manifest")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("sha256") != actual_hash:
        raise ValueError("capability corpus hash does not match its manifest")
    actual_rows = sum(
        bool(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    if manifest.get("rows") != actual_rows:
        raise ValueError("capability corpus row count does not match its manifest")
    if expected_contract and manifest.get("contract") != expected_contract:
        raise ValueError(
            f"capability corpus contract is not {expected_contract}"
        )
    return manifest


def load_capability_corpus(
    path: Path,
    minimum_train_families_per_language: int = 500,
    minimum_eval_families_per_language: int = 100,
) -> tuple[list[MemoryRecord], list[MemoryExample]]:
    """Load natural multilingual QA with explicit causal memory variants.

    Each JSONL row contains a question/answer plus one or more records. Rows in
    the same family keep the question fixed while changing authoritative memory
    and answer. IDs are opaque and cannot be derived from question text.
    """
    rows = _read_rows(path)
    records: dict[str, MemoryRecord] = {}
    examples: list[MemoryExample] = []
    family_splits: dict[str, str] = {}
    family_rows: dict[str, list[MemoryExample]] = defaultdict(list)
    family_counts: Counter[tuple[str, str]] = Counter()
    unknown_counts: Counter[tuple[str, str]] = Counter()
    question_splits: dict[tuple[str, str], str] = {}
    multi_record = 0

    for index, row in enumerate(rows):
        if row.get("schema_version") != 1:
            raise ValueError(f"row {index} has unsupported schema_version")
        split = str(row.get("split", ""))
        language = str(row.get("language", ""))
        family_id = str(row.get("family_id", ""))
        example_id = str(row.get("example_id", ""))
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        kind = str(row.get("kind", "natural"))
        citations_raw = row.get("citations", [])
        records_raw = row.get("records", [])
        if split not in ("train", "eval") or language not in LANGUAGES:
            raise ValueError(f"row {index} has invalid split/language")
        if not family_id or not example_id or not question or not answer:
            raise ValueError(f"row {index} is missing identity or QA text")
        if family_splits.setdefault(family_id, split) != split:
            raise ValueError(f"family crosses train/eval boundary: {family_id}")
        normalized_question = " ".join(question.casefold().split())
        question_key = (language, normalized_question)
        previous_split = question_splits.setdefault(question_key, split)
        if previous_split != split:
            raise ValueError(
                f"question crosses train/eval boundary: {example_id}"
            )
        if not isinstance(citations_raw, list) or not isinstance(records_raw, list):
            raise ValueError(f"row {index} has invalid citations/records")
        if not records_raw or len(records_raw) > 4:
            raise ValueError(f"row {index} must admit one to four records")
        if len(records_raw) > 1:
            multi_record += 1
        shortcut_text = "\n".join(_strings({
            "question": question, "answer": answer, "records": records_raw,
        }))
        if _FORBIDDEN_SHORTCUT.search(shortcut_text):
            raise ValueError(f"row {index} contains a rejected synthetic shortcut")

        memory_ids: list[str] = []
        public_ids: list[str] = []
        memory_texts: list[str] = []
        for record_index, raw in enumerate(records_raw):
            if not isinstance(raw, dict):
                raise ValueError(f"row {index} record {record_index} is not an object")
            record_id = str(raw.get("record_id", ""))
            citation_id = str(raw.get("citation_id", ""))
            text = str(raw.get("text", "")).strip()
            if not record_id or not citation_id or not text:
                raise ValueError(f"row {index} has an incomplete record")
            if citation_id.casefold() in question.casefold():
                raise ValueError(f"citation ID leaks through question: {example_id}")
            record = MemoryRecord(
                record_id=record_id,
                citation_id=citation_id,
                text=text,
                shard_key=str(raw.get("shard_key", family_id)),
                generation=int(raw.get("generation", 1)),
                language=language,
                indexable=bool(raw.get("indexable", True)),
            )
            previous = records.setdefault(record_id, record)
            if previous != record:
                raise ValueError(f"record ID collision with different content: {record_id}")
            memory_ids.append(record_id)
            public_ids.append(citation_id)
            memory_texts.append(text)

        citations = tuple(str(item) for item in citations_raw)
        if len(set(public_ids)) != len(public_ids):
            raise ValueError(f"row {index} admits duplicate public source IDs")
        if not set(citations).issubset(set(public_ids)):
            raise ValueError(f"row {index} cites a record that was not admitted")
        source_slots = tuple(
            slot for slot, public_id in enumerate(public_ids)
            if public_id in set(citations)
        )
        if kind == "unknown":
            if answer != UNKNOWN_ANSWER or citations:
                raise ValueError("unknown rows require exact abstention and no citations")
            unknown_counts[(split, language)] += 1
        else:
            if not citations:
                raise ValueError("answerable rows require an authoritative citation")
            if not any(normalized_contains(text, answer) for text in memory_texts):
                raise ValueError(f"answer is not extractive from admitted memory: {example_id}")

        example = MemoryExample(
            example_id=example_id,
            question=question,
            answer=answer,
            citations=citations,
            memory_ids=tuple(memory_ids),
            kind=kind,
            split=split,
            language=language,
            family_id=family_id,
            source_slots=source_slots,
        )
        examples.append(example)
        family_rows[family_id].append(example)

    for family_id, members in family_rows.items():
        answerable = [member for member in members if member.kind != "unknown"]
        if not answerable:
            continue
        questions = {member.question for member in answerable}
        answers = {member.answer for member in answerable}
        if len(questions) != 1 or len(answers) < 2:
            raise ValueError(
                f"family {family_id} must contain same-question contradictory memories"
            )
        citation_sets = {member.citations for member in answerable}
        if len(citation_sets) != 1:
            raise ValueError(
                f"family {family_id} leaks the intervention through citations"
            )
        source_slot_sets = {member.source_slots for member in answerable}
        if len(source_slot_sets) != 1:
            raise ValueError(
                f"family {family_id} leaks the intervention through source position"
            )
        distractor_sets: set[tuple[str, ...]] = set()
        for member in answerable:
            authority = set(member.citations)
            authoritative_texts = [
                records[record_id].text for record_id in member.memory_ids
                if records[record_id].public_id in authority
            ]
            distractor_sets.add(tuple(sorted(
                record_id for record_id in member.memory_ids
                if records[record_id].public_id not in authority
            )))
            for sibling_answer in answers - {member.answer}:
                if any(
                    normalized_contains(text, sibling_answer)
                    for text in authoritative_texts
                ):
                    raise ValueError(
                        f"family {family_id} retains a sibling answer in authority"
                    )
        if len(distractor_sets) != 1:
            raise ValueError(
                f"family {family_id} changes distractors during intervention"
            )
        member = answerable[0]
        family_counts[(member.split, member.language)] += 1

    for language in LANGUAGES:
        if family_counts[("train", language)] < minimum_train_families_per_language:
            raise ValueError(f"insufficient train families for {language}")
        if family_counts[("eval", language)] < minimum_eval_families_per_language:
            raise ValueError(f"insufficient eval families for {language}")
        for split in ("train", "eval"):
            if unknown_counts[(split, language)] == 0:
                raise ValueError(f"missing unknown examples for {split}/{language}")
    if multi_record < len(rows) // 2:
        raise ValueError("at least half of capability rows must include distractor records")
    return list(records.values()), examples


def training_texts(
    records: Sequence[MemoryRecord], examples: Sequence[MemoryExample]
) -> list[str]:
    train_ids = {
        record_id
        for example in examples if example.split == "train"
        for record_id in example.memory_ids
    }
    values = [record.text for record in records if record.record_id in train_ids]
    values.extend(
        f"{example.question}\n{example.answer}"
        for example in examples if example.split == "train"
    )
    return values
