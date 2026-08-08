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
        chat_prompt,
        encode_memory_sets,
        generate,
        load_adapter,
        load_model,
        resolve_layers,
        score_example,
        seed_everything,
        select_eval_examples,
    )
    from .real_query import read_questions
    from .synthetic_memory import (
        MemoryRecord,
        normalized_contains,
        parse_response,
    )
except ImportError:  # Direct execution on a worker.
    from capability_corpus import load_capability_corpus
    from data_contract import read_records
    from mechanism_probe import select_ood_record
    from pow import (
        MemoryHook,
        PowConfig,
        chat_prompt,
        encode_memory_sets,
        generate,
        load_adapter,
        load_model,
        resolve_layers,
        score_example,
        seed_everything,
        select_eval_examples,
    )
    from real_query import read_questions
    from synthetic_memory import (
        MemoryRecord,
        normalized_contains,
        parse_response,
    )


# Inference-only intervention arms. (name, gate_scale, null_always)
ARMS = (
    ("baseline", 1.0, False),
    ("scale-1.5", 1.5, False),
    ("scale-2.0", 2.0, False),
    ("scale-3.0", 3.0, False),
    ("null-open", 1.0, True),
    ("scale-2.0-null-open", 2.0, True),
)


def run_interventions(checkpoint_path: Path, corpus_path: Path, ingest_root: Path,
                      questions_path: Path, output_path: Path, id_cases: int,
                      ood_cases: int, maximum_new_tokens: int,
                      device: torch.device) -> dict[str, object]:
    checkpoint_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PowConfig(**checkpoint_raw["config"])
    config = PowConfig(**{**config.__dict__, "maximum_new_tokens": maximum_new_tokens})
    seed_everything(config.seed)
    capability_records, capability_examples = load_capability_corpus(corpus_path)
    capability_by_id = {record.record_id: record for record in capability_records}
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

    records_by_id = dict(capability_by_id)
    ood_rows: list[tuple[dict[str, object], tuple[str, ...]]] = []
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
        ood_rows.append((row, (record_id,)))

    id_examples = [
        example
        for example in select_eval_examples(capability_examples, 64)
        if example.kind != "unknown"
    ][:id_cases]
    memory_sets = [example.memory_ids for example in id_examples]
    memory_sets.extend(memory_ids for _, memory_ids in ood_rows)
    try:
        memory_cache = encode_memory_sets(
            model, tokenizer, hooks, records_by_id, memory_sets,
            config.maximum_memory_tokens, device,
        )
        arms: list[dict[str, object]] = []
        for name, gate_scale, null_always in ARMS:
            for index in layer_indices:
                expert = adapter.adapter(index)
                expert.gate_scale = gate_scale
                expert.null_always = null_always
            ood_results: list[dict[str, object]] = []
            for row, memory_ids in ood_rows:
                response = generate(
                    model, tokenizer, hooks, memory_cache, memory_ids,
                    chat_prompt(tokenizer, str(row["question"])),
                    maximum_new_tokens, device,
                )
                answer, slots = parse_response(response)
                expected = row.get("expected_answers", [])
                if isinstance(expected, str):
                    expected = [expected]
                expected = [str(value) for value in expected]
                answer_ok = bool(expected) and all(
                    normalized_contains(answer, value) for value in expected
                )
                source_ok = slots == (0,)
                ood_results.append({
                    "case_id": str(row["id"]),
                    "response": response,
                    "answer_ok": answer_ok,
                    "source_ok": source_ok,
                    "passed": answer_ok and source_ok,
                })
            id_results: list[dict[str, object]] = []
            for example in id_examples:
                response = generate(
                    model, tokenizer, hooks, memory_cache, example.memory_ids,
                    chat_prompt(tokenizer, example.question),
                    maximum_new_tokens, device,
                )
                scored = score_example(
                    example, response, example.memory_ids, capability_by_id
                )
                id_results.append({
                    "case_id": example.example_id,
                    "response": response,
                    "passed": bool(scored["passed"]),
                })
            count_ood = max(1, len(ood_results))
            count_id = max(1, len(id_results))
            arm = {
                "arm": name,
                "gate_scale": gate_scale,
                "null_always": null_always,
                "ood_answer_ok": sum(r["answer_ok"] for r in ood_results),
                "ood_source_ok": sum(r["source_ok"] for r in ood_results),
                "ood_joint": sum(r["passed"] for r in ood_results),
                "ood_total": len(ood_results),
                "id_joint": sum(r["passed"] for r in id_results),
                "id_total": len(id_results),
                "ood_results": ood_results,
                "id_results": id_results,
            }
            arms.append(arm)
            print(json.dumps({
                "event": "mechanism-intervene-arm",
                **{key: value for key, value in arm.items()
                   if key not in ("ood_results", "id_results")},
            }, ensure_ascii=False), flush=True)
        result = {
            "schema_version": 1,
            "contract": "quantum-llm-memory-mechanism-intervene-v1",
            "checkpoint": str(checkpoint_path),
            "arms": arms,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        for index in layer_indices:
            expert = adapter.adapter(index)
            expert.gate_scale = 1.0
            expert.null_always = False
        for handle in handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only gate-scale and null-key interventions"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-cases", type=int, default=8)
    parser.add_argument("--ood-cases", type=int, default=8)
    parser.add_argument("--maximum-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_interventions(
        args.checkpoint.resolve(), args.corpus.resolve(), args.ingest.resolve(),
        args.questions.resolve(), args.output.resolve(), args.id_cases,
        args.ood_cases, args.maximum_new_tokens, torch.device(args.device),
    )
    print(json.dumps({
        "event": "mechanism-intervene-summary",
        "arms": [
            {
                "arm": arm["arm"],
                "ood_joint": f"{arm['ood_joint']}/{arm['ood_total']}",
                "id_joint": f"{arm['id_joint']}/{arm['id_total']}",
            }
            for arm in result["arms"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
