from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import defaultdict
from pathlib import Path

DATASET = "google/xquad"
REVISION = "51adfef1c1287aab1d2d91b5bead9bcfb9c68583"
LANGUAGES = ("ro", "ru", "en")
UNKNOWN_ANSWER = "I don't know from the attached memory."


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


def _token_occurrences(text: str, fragment: str) -> int:
    haystack, needle = _normalized_tokens(text), _normalized_tokens(fragment)
    if not needle:
        return 0
    return sum(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


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
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                break
            except HTTPError as error:
                if error.code != 429 or attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        page = [item["row"] for item in payload["rows"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= int(payload["num_rows_total"]):
            break
    return rows


def _load_languages(cache_root: Path) -> dict[str, list[dict[str, object]]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, list[dict[str, object]]] = {}
    for language in LANGUAGES:
        path = cache_root / f"xquad-{REVISION}-{language}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("revision") != REVISION or not isinstance(
                payload.get("rows"), list
            ):
                raise ValueError(f"invalid source cache: {path}")
            loaded[language] = payload["rows"]
            continue
        rows = _fetch_language(language)
        path.write_text(json.dumps({
            "revision": REVISION,
            "language": language,
            "rows": rows,
        }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        loaded[language] = rows
    return loaded


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
        "text": context,
        "shard_key": _opaque_id("S", f"{language}:{source_id}"),
        "generation": variant + 1,
        "indexable": indexable,
    }


def _rewrite_source_hash(question: str, context: str, answer: str) -> str:
    payload = json.dumps(
        {"question": question, "context": context, "answer": answer},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_rewrite_jobs(path: Path, prepared: list[dict[str, str]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in prepared:
            row = {
                "schema_version": 1,
                "family_id": item["family_id"],
                "language": item["language"],
                "split": item["split"],
                "question": item["question"],
                "original_context": item["context"],
                "original_answer": item["answer"],
                "source_sha256": item["source_sha256"],
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "contract": "memory-counterfactual-rewrite-jobs-v1",
        "rows": len(prepared),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _load_validated_rewrites(path: Path) -> dict[str, dict[str, str]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"rewrite manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != "memory-counterfactual-rewrites-v1"
        or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("rewrite artifact does not match its manifest")
    rewrites: dict[str, dict[str, str]] = {}
    row_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row_count += 1
        raw = json.loads(line)
        if raw.get("schema_version") != 1 or raw.get("status") != "accepted":
            continue
        validation = raw.get("validation")
        if not isinstance(validation, dict) or not all(
            validation.get(key) is True
            for key in ("question_answerable", "context_coherent", "language_match")
        ):
            raise ValueError(f"rewrite row {line_number} lacks independent validation")
        family_id = str(raw.get("family_id", ""))
        answer = str(raw.get("counterfactual_answer", "")).strip()
        context = str(raw.get("counterfactual_context", "")).strip()
        extracted = str(validation.get("extracted_answer", "")).strip()
        if not family_id or not answer or not context or extracted != answer:
            raise ValueError(f"rewrite row {line_number} is incomplete")
        if family_id in rewrites:
            raise ValueError(f"duplicate accepted rewrite: {family_id}")
        rewrites[family_id] = {
            "source_sha256": str(raw.get("source_sha256", "")),
            "answer": answer,
            "context": context,
        }
    if not rewrites:
        raise ValueError("rewrite artifact contains no accepted counterfactuals")
    if manifest.get("rows") != row_count or manifest.get("accepted") != len(rewrites):
        raise ValueError("rewrite artifact counts do not match its manifest")
    return rewrites


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


def build(
    output: Path,
    seed: int,
    source_cache: Path | None = None,
    rewrite_jobs: Path | None = None,
    rewrites_path: Path | None = None,
) -> dict[str, object]:
    rng = random.Random(seed)
    source_sha = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{DATASET}", timeout=60
    ))["sha"]
    if source_sha != REVISION:
        raise RuntimeError(f"dataset revision changed: expected {REVISION}, got {source_sha}")
    cache_root = source_cache or output.parent / "source-cache"
    by_language = _load_languages(cache_root)
    source_splits = _source_splits(by_language)

    output_rows: list[dict[str, object]] = []
    records_for_distractors: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    prepared: list[dict[str, str]] = []
    for language, rows in by_language.items():
        for row in rows:
            source_id = str(row["id"])
            split = source_splits[source_id]
            answer, start = _answer(row)
            context, _ = _answer_window(
                str(row["context"]), start, len(answer)
            )
            if _token_occurrences(context, answer) != 1:
                continue
            citation_id = _opaque_id("C", f"citation:{language}:{source_id}")
            original = _record(
                language, source_id, 0, citation_id, context, True
            )
            records_for_distractors[(language, split)].append(original)
            question = str(row["question"]).strip()
            family_id = _opaque_id("F", f"family:{language}:{source_id}")
            prepared.append({
                "language": language,
                "split": split,
                "source_id": source_id,
                "question": question,
                "answer": answer,
                "citation_id": citation_id,
                "context": context,
                "family_id": family_id,
                "source_sha256": _rewrite_source_hash(question, context, answer),
            })

    jobs_path = rewrite_jobs or output.parent / "counterfactual-rewrite-jobs.jsonl"
    jobs_manifest = _write_rewrite_jobs(jobs_path, prepared)
    if rewrites_path is None:
        return {
            "status": "rewrite-required",
            "rewrite_jobs": str(jobs_path),
            "rewrite_jobs_sha256": jobs_manifest["sha256"],
            "jobs": len(prepared),
        }
    rewrites = _load_validated_rewrites(rewrites_path)
    accepted_families: set[str] = set()
    for item in prepared:
        language = item["language"]
        split = item["split"]
        source_id = item["source_id"]
        question = item["question"]
        family_id = item["family_id"]
        answer = item["answer"]
        citation_id = item["citation_id"]
        context = item["context"]
        rewrite = rewrites.get(family_id)
        if rewrite is None:
            continue
        if rewrite["source_sha256"] != item["source_sha256"]:
            raise ValueError(f"stale counterfactual rewrite for {family_id}")
        alternate = rewrite["answer"]
        alternate_context = rewrite["context"]
        if alternate.casefold() == answer.casefold():
            raise ValueError(f"counterfactual answer did not change for {family_id}")
        if _token_occurrences(alternate_context, alternate) != 1:
            raise ValueError(f"counterfactual answer must occur exactly once: {family_id}")
        if _contains_tokens(alternate_context, answer):
            raise ValueError(f"counterfactual retained original answer: {family_id}")
        original = _record(language, source_id, 0, citation_id, context, True)
        counterfactual = _record(
            language, source_id, 1, citation_id, alternate_context, False
        )
        distractors = [
            item for item in records_for_distractors[(language, split)]
            if item["citation_id"] != citation_id
            and not _contains_tokens(str(item["text"]), answer)
            and not _contains_tokens(str(item["text"]), alternate)
        ]
        if not distractors:
            raise ValueError("cannot construct a non-answering distractor")
        distractor = rng.choice(distractors)
        target_slot = int(_digest(f"slot:{family_id}", 2), 16) % 2
        for variant, target, target_answer, kind in (
            (0, original, answer, "natural"),
            (1, counterfactual, alternate, "counterfactual"),
        ):
            admitted_records = [distractor, target]
            if target_slot == 0:
                admitted_records.reverse()
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
                "records": admitted_records,
            })
        accepted_families.add(family_id)

    family_counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in prepared:
        if item["family_id"] in accepted_families:
            family_counts[(item["split"], item["language"])] += 1
    for language in LANGUAGES:
        if family_counts[("train", language)] < 500:
            raise ValueError(f"fewer than 500 accepted train rewrites for {language}")
        if family_counts[("eval", language)] < 100:
            raise ValueError(f"fewer than 100 accepted eval rewrites for {language}")

    for language, rows in by_language.items():
        for index, row in enumerate(rows[::10]):
            source_id = str(row["id"])
            split = source_splits[source_id]
            original_answer, _ = _answer(row)
            distractors = [
                item for item in records_for_distractors[(language, split)]
                if not _contains_tokens(str(item["text"]), original_answer)
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
        "schema_version": 2,
        "contract": "xquad-coherent-counterfactual-memory-v2",
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "source_license": "cc-by-sa-4.0",
        "languages": list(LANGUAGES),
        "rewrite_artifact_sha256": hashlib.sha256(rewrites_path.read_bytes()).hexdigest(),
        "rewrite_jobs_sha256": jobs_manifest["sha256"],
        "rows": len(output_rows),
        "records": len({
            str(record["record_id"])
            for row in output_rows for record in row["records"]
        }),
        "examples_by_split_language_kind": {
            "/".join(key): value
            for key, value in sorted(
                {
                    key: sum(
                        row["split"] == key[0]
                        and row["language"] == key[1]
                        and row["kind"] == key[2]
                        for row in output_rows
                    )
                    for key in {
                        (str(row["split"]), str(row["language"]), str(row["kind"]))
                        for row in output_rows
                    }
                }.items()
            )
        },
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def validate_built_corpus(path: Path) -> dict[str, object]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("built capability corpus or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("contract") != "xquad-coherent-counterfactual-memory-v2"
        or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("capability corpus is stale or uses a superseded contract")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare natural multilingual Memory Expert QA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--rewrite-jobs", type=Path)
    parser.add_argument("--rewrites", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_built_corpus(args.output.resolve()), indent=2))
        return 0
    print(json.dumps(build(
        args.output.resolve(), args.seed,
        args.source_cache.resolve() if args.source_cache else None,
        args.rewrite_jobs.resolve() if args.rewrite_jobs else None,
        args.rewrites.resolve() if args.rewrites else None,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
