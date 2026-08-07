from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .data_contract import KnowledgeRecord, read_records
    from .pow import PowConfig, chat_prompt, generate, load_model, seed_everything
    from .real_query import read_questions, section_matches
    from .synthetic_memory import normalized_contains, parse_response
except ImportError:  # Direct execution on a worker.
    from data_contract import KnowledgeRecord, read_records
    from pow import PowConfig, chat_prompt, generate, load_model, seed_everything
    from real_query import read_questions, section_matches
    from synthetic_memory import normalized_contains, parse_response


def render_context(records: list[KnowledgeRecord]) -> str:
    """Render immutable records as request-local slots for the control arm."""
    return "\n\n".join(
        f"SOURCE {slot}:\n{record.retrieval_text}"
        for slot, record in enumerate(records)
    )


def run_control(ingest_root: Path, checkpoint_path: Path, questions_path: Path,
                output_path: Path, maximum_new_tokens: int,
                device: torch.device) -> dict[str, object]:
    started = time.perf_counter()
    records = list(read_records(ingest_root))
    questions = read_questions(questions_path)
    if not records:
        raise ValueError("ingest contains no records")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PowConfig(**checkpoint["config"])
    seed_everything(config.seed)
    tokenizer, model = load_model(config, device)
    loaded = time.perf_counter()
    context = render_context(records)
    empty_cache: dict[tuple[str, ...], dict[int, torch.Tensor]] = {(): {}}
    results: list[dict[str, object]] = []
    for row in questions:
        response = generate(
            model, tokenizer, {}, empty_cache, (),
            chat_prompt(tokenizer, str(row["question"]), context),
            maximum_new_tokens, device,
        )
        answer, slots = parse_response(response)
        expected_answers = row.get("expected_answers", [])
        if isinstance(expected_answers, str):
            expected_answers = [expected_answers]
        expected_sections = row.get("expected_sections", [])
        if isinstance(expected_sections, str):
            expected_sections = [expected_sections]
        slots_valid = (
            bool(slots)
            and len(set(slots)) == len(slots)
            and all(0 <= slot < len(records) for slot in slots)
        )
        selected = [records[slot] for slot in slots] if slots_valid else []
        answer_match = (
            not expected_answers
            or all(normalized_contains(answer, str(value)) for value in expected_answers)
        )
        source_match = (
            not expected_sections
            or all(
                any(section_matches(record, [str(pattern)]) for record in selected)
                for pattern in expected_sections
            )
        )
        result = {
            "id": row["id"],
            "question": row["question"],
            "response": response,
            "answer": answer,
            "source_slots": slots,
            "answer_text_match": answer_match,
            "source_selection_ok": slots_valid and source_match,
            "passed": answer_match and slots_valid and source_match,
        }
        results.append(result)
        print(json.dumps({"event": "context-control", **result}, ensure_ascii=False), flush=True)
    finished = time.perf_counter()
    count = len(results)
    summary = {
        "schema_version": 1,
        "contract": "quantum-llm-full-context-control-v1",
        "records": len(records),
        "questions": count,
        "strict_answer_text_match_rate": sum(
            bool(row["answer_text_match"]) for row in results
        ) / count,
        "source_selection_accuracy": sum(
            bool(row["source_selection_ok"]) for row in results
        ) / count,
        "joint_accuracy": sum(bool(row["passed"]) for row in results) / count,
        "timing_seconds": {
            "model_load": loaded - started,
            "generation": finished - loaded,
            "total": finished - started,
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen base model with all ingested records in context"
    )
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_control(
        args.ingest.resolve(), args.checkpoint.resolve(), args.questions.resolve(),
        args.output.resolve(), args.maximum_new_tokens, torch.device(args.device),
    )
    print(json.dumps({
        "event": "context-control-summary",
        **{key: value for key, value in summary.items() if key != "results"},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
