from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

try:
    from .prepare_capability_data import _contains_tokens, _token_occurrences
except ImportError:
    from prepare_capability_data import _contains_tokens, _token_occurrences


GENERATOR_PROMPT = """Create one coherent counterfactual reading-comprehension passage.
Keep the question unchanged and write in the same language as the input. Change
the underlying fact so that the unchanged question has a different answer.
Rewrite every sentence needed for grammar, agreement, references, dates, and
semantic coherence; do not perform a blind span substitution. The original
answer must not remain in the rewritten passage. The new answer must appear
verbatim exactly once. Preserve unrelated information. Return one JSON object
with exactly these string fields: counterfactual_context, counterfactual_answer.

LANGUAGE: {language}
QUESTION: {question}
ORIGINAL ANSWER: {answer}
ORIGINAL PASSAGE:
{context}"""


VERIFIER_PROMPT = """Independently audit the candidate passage below. Do not use the
claimed answer from another model. Answer the unchanged question using only the
candidate passage, then judge whether the passage is linguistically coherent
and whether it is written in the requested language. Return one JSON object
with exactly these fields: extracted_answer (string), question_answerable
(boolean), context_coherent (boolean), language_match (boolean).

REQUESTED LANGUAGE: {language}
QUESTION: {question}
CANDIDATE PASSAGE:
{context}"""


def _json_object(text: str) -> dict[str, object]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response was not a JSON object")
    return value


def _completion(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
) -> dict[str, object]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1200,
        "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(api_url, data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    return _json_object(str(content))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_jobs(path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"rewrite-jobs manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != "memory-counterfactual-rewrite-jobs-v1"
        or manifest.get("rows") != len(rows)
        or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("rewrite jobs do not match their manifest")


def generate(
    jobs_path: Path,
    output_path: Path,
    api_url: str,
    api_key: str,
    generator_model: str,
    verifier_model: str,
    limit: int,
    attempts: int,
    timeout: int,
) -> dict[str, object]:
    jobs = _read_jsonl(jobs_path)
    _validate_jobs(jobs_path, jobs)
    jobs_by_id = {str(row["family_id"]): row for row in jobs}
    if len(jobs_by_id) != len(jobs):
        raise ValueError("rewrite jobs contain duplicate family IDs")
    prior = _read_jsonl(output_path) if output_path.is_file() else []
    for row in prior:
        family_id = str(row.get("family_id", ""))
        current = jobs_by_id.get(family_id)
        if (
            row.get("status") == "accepted" and current is not None
            and row.get("source_sha256") != current.get("source_sha256")
        ):
            row["status"] = "superseded"
            row["error"] = "source job changed"
    if prior:
        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in prior:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    accepted = {
        family_id
        for row in prior
        if row.get("status") == "accepted"
        and (family_id := str(row["family_id"])) in jobs_by_id
        and row.get("source_sha256") == jobs_by_id[family_id].get("source_sha256")
    }
    pending = [row for row in jobs if str(row["family_id"]) not in accepted]
    if limit > 0:
        pending = pending[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_now = rejected_now = 0
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        for number, job in enumerate(pending, 1):
            last_error = "generation failed"
            result: dict[str, object] | None = None
            for attempt in range(1, attempts + 1):
                try:
                    generated = _completion(
                        api_url, api_key, generator_model,
                        GENERATOR_PROMPT.format(
                            language=job["language"], question=job["question"],
                            answer=job["original_answer"],
                            context=job["original_context"],
                        ),
                        0.2, timeout,
                    )
                    answer = str(generated["counterfactual_answer"]).strip()
                    context = str(generated["counterfactual_context"]).strip()
                    original = str(job["original_answer"])
                    if not answer or not context or answer.casefold() == original.casefold():
                        raise ValueError("counterfactual fact did not change")
                    if _token_occurrences(context, answer) != 1:
                        raise ValueError("new answer is not present exactly once")
                    if _contains_tokens(context, original):
                        raise ValueError("original answer remains in candidate passage")
                    verified = _completion(
                        api_url, api_key, verifier_model,
                        VERIFIER_PROMPT.format(
                            language=job["language"], question=job["question"],
                            context=context,
                        ),
                        0.0, timeout,
                    )
                    validation = {
                        "extracted_answer": str(verified.get("extracted_answer", "")).strip(),
                        "question_answerable": verified.get("question_answerable") is True,
                        "context_coherent": verified.get("context_coherent") is True,
                        "language_match": verified.get("language_match") is True,
                    }
                    if validation["extracted_answer"] != answer or not all(
                        validation[key] for key in (
                            "question_answerable", "context_coherent", "language_match"
                        )
                    ):
                        raise ValueError("independent verifier rejected candidate")
                    result = {
                        "schema_version": 1,
                        "status": "accepted",
                        "family_id": job["family_id"],
                        "source_sha256": job["source_sha256"],
                        "counterfactual_context": context,
                        "counterfactual_answer": answer,
                        "generator_model": generator_model,
                        "verifier_model": verifier_model,
                        "validation": validation,
                    }
                    break
                except Exception as error:  # Preserve a resumable rejection audit.
                    last_error = f"attempt {attempt}: {type(error).__name__}: {error}"
                    if attempt < attempts:
                        time.sleep(min(2 ** (attempt - 1), 8))
            if result is None:
                rejected_now += 1
                result = {
                    "schema_version": 1,
                    "status": "rejected",
                    "family_id": job["family_id"],
                    "source_sha256": job["source_sha256"],
                    "error": last_error,
                }
            else:
                accepted_now += 1
            stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            print(json.dumps({
                "event": "counterfactual-rewrite",
                "completed": number,
                "pending": len(pending),
                "status": result["status"],
                "family_id": result["family_id"],
            }), flush=True)
    all_rows = _read_jsonl(output_path)
    artifact_manifest = {
        "schema_version": 1,
        "contract": "memory-counterfactual-rewrites-v1",
        "rows": len(all_rows),
        "accepted": sum(row.get("status") == "accepted" for row in all_rows),
        "rejected": sum(row.get("status") == "rejected" for row in all_rows),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "jobs": len(jobs),
        "already_accepted": len(accepted),
        "attempted": len(pending),
        "accepted": accepted_now,
        "rejected": rejected_now,
        "artifact_sha256": artifact_manifest["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and independently validate coherent QA counterfactuals"
    )
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default=os.getenv("MEMORY_REWRITE_API_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("MEMORY_REWRITE_API_KEY", ""))
    parser.add_argument(
        "--generator-model", default=os.getenv("MEMORY_REWRITE_GENERATOR_MODEL", "")
    )
    parser.add_argument(
        "--verifier-model", default=os.getenv("MEMORY_REWRITE_VERIFIER_MODEL", "")
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not args.api_url or not args.generator_model or not args.verifier_model:
        parser.error(
            "configure MEMORY_REWRITE_API_URL, MEMORY_REWRITE_GENERATOR_MODEL, "
            "and MEMORY_REWRITE_VERIFIER_MODEL"
        )
    print(json.dumps(generate(
        args.jobs.resolve(), args.output.resolve(), args.api_url, args.api_key,
        args.generator_model, args.verifier_model, args.limit, args.attempts,
        args.timeout,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
