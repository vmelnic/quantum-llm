from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

try:
    from .capability_corpus import load_capability_corpus
    from .data_contract import KnowledgeRecord, read_records
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
    from .real_query import read_questions, section_matches
    from .synthetic_memory import (
        MemoryExample,
        MemoryRecord,
        normalized_contains,
        parse_response,
    )
except ImportError:  # Direct execution on a worker.
    from capability_corpus import load_capability_corpus
    from data_contract import KnowledgeRecord, read_records
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
    from real_query import read_questions, section_matches
    from synthetic_memory import (
        MemoryExample,
        MemoryRecord,
        normalized_contains,
        parse_response,
    )


METRICS = (
    "real_attention_mass",
    "null_attention_mass",
    "attention_entropy",
    "attention_max",
    "context_hidden_ratio",
    "gate_delta_hidden_ratio",
    "gate_delta_hidden_ratio_last",
)


class ProbeCollector:
    def __init__(self) -> None:
        self.case_id = ""
        self.events: list[dict[str, object]] = []

    def begin(self, case_id: str) -> int:
        self.case_id = case_id
        return len(self.events)

    def __call__(self, event: dict[str, object]) -> None:
        self.events.append({"case_id": self.case_id, **event})

    def since(self, offset: int) -> list[dict[str, object]]:
        return self.events[offset:]


def summarize_events(events: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        buckets[(int(event["layer"]), str(event["phase"]))].append(event)
    rows: list[dict[str, object]] = []
    for (layer, phase), values in sorted(buckets.items()):
        row: dict[str, object] = {
            "layer": layer,
            "phase": phase,
            "calls": len(values),
            "memory_tokens": statistics.fmean(
                float(value["memory_tokens"]) for value in values
            ),
        }
        for metric in METRICS:
            row[metric] = statistics.fmean(
                float(value[metric]) for value in values
            )
        rows.append(row)
    return rows


def summarize_group(cases: Sequence[dict[str, object]]) -> dict[str, object]:
    buckets: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        for row in case["layer_metrics"]:
            buckets[(int(row["layer"]), str(row["phase"]))].append(row)
    layers: list[dict[str, object]] = []
    for (layer, phase), values in sorted(buckets.items()):
        row: dict[str, object] = {
            "layer": layer,
            "phase": phase,
            "cases": len(values),
        }
        for metric in METRICS:
            row[metric] = statistics.fmean(
                float(value[metric]) for value in values
            )
        layers.append(row)
    prefill = [row for row in layers if row["phase"] == "prefill"]
    return {
        "cases": len(cases),
        "prefill_layer_mean": {
            metric: statistics.fmean(float(row[metric]) for row in prefill)
            if prefill else 0.0
            for metric in METRICS
        },
        "layers": layers,
    }


def select_ood_record(records: Sequence[KnowledgeRecord],
                      expected_sections: Sequence[str]) -> KnowledgeRecord:
    matches = [
        record for record in records
        if all(section_matches(record, [pattern]) for pattern in expected_sections)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one OOD oracle record, found {len(matches)} for {expected_sections}"
        )
    return matches[0]


def run_probe(checkpoint_path: Path, corpus_path: Path, ingest_root: Path,
              questions_path: Path, output_path: Path, in_distribution_cases: int,
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
    collector = ProbeCollector()
    hooks = {index: MemoryHook(adapter.adapter(index)) for index in layer_indices}
    for index in layer_indices:
        expert = adapter.adapter(index)
        expert.probe_layer_index = index
        expert.probe_sink = collector
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

    candidates = [
        example for example in select_eval_examples(capability_examples, 64)
        if example.kind != "unknown"
    ]
    memory_sets = [example.memory_ids for example in candidates]
    memory_sets.extend(memory_ids for _, memory_ids in ood_rows)
    try:
        memory_cache = encode_memory_sets(
            model, tokenizer, hooks, records_by_id, memory_sets,
            config.maximum_memory_tokens, device,
        )
        groups: dict[str, list[dict[str, object]]] = {
            "conflictqa_correct": [],
            "ood_correct": [],
            "ood_failed": [],
        }
        for example in candidates:
            if len(groups["conflictqa_correct"]) >= in_distribution_cases:
                break
            offset = collector.begin(f"conflictqa::{example.example_id}")
            response = generate(
                model, tokenizer, hooks, memory_cache, example.memory_ids,
                chat_prompt(tokenizer, example.question), maximum_new_tokens, device,
            )
            scored = score_example(
                example, response, example.memory_ids, capability_by_id
            )
            if not scored["passed"]:
                continue
            groups["conflictqa_correct"].append({
                "case_id": example.example_id,
                "question": example.question,
                "response": response,
                "memory_tokens": sum(
                    memory_cache[example.memory_ids][index].shape[0]
                    for index in layer_indices
                ) // len(layer_indices),
                "layer_metrics": summarize_events(collector.since(offset)),
            })
        if len(groups["conflictqa_correct"]) < in_distribution_cases:
            raise RuntimeError("could not collect enough correct ConflictQA cases")

        for row, memory_ids in ood_rows:
            case_id = str(row["id"])
            offset = collector.begin(f"ood::{case_id}")
            response = generate(
                model, tokenizer, hooks, memory_cache, memory_ids,
                chat_prompt(tokenizer, str(row["question"])),
                maximum_new_tokens, device,
            )
            answer, slots = parse_response(response)
            expected = row.get("expected_answers", [])
            if isinstance(expected, str):
                expected = [expected]
            answer_ok = all(
                normalized_contains(answer, str(value)) for value in expected
            )
            source_ok = slots == (0,)
            group = "ood_correct" if answer_ok and source_ok else "ood_failed"
            groups[group].append({
                "case_id": case_id,
                "question": row["question"],
                "response": response,
                "answer_ok": answer_ok,
                "source_ok": source_ok,
                "memory_tokens": sum(
                    memory_cache[memory_ids][index].shape[0]
                    for index in layer_indices
                ) // len(layer_indices),
                "layer_metrics": summarize_events(collector.since(offset)),
            })

        summaries = {
            name: summarize_group(cases) for name, cases in groups.items()
        }
        baseline_ratio = float(
            summaries["conflictqa_correct"]["prefill_layer_mean"]
            ["gate_delta_hidden_ratio_last"]
        )
        failed_ratio = float(
            summaries["ood_failed"]["prefill_layer_mean"]
            ["gate_delta_hidden_ratio_last"]
        )
        ratio = failed_ratio / baseline_ratio if baseline_ratio else None
        result = {
            "schema_version": 1,
            "contract": "quantum-llm-memory-mechanism-probe-v1",
            "checkpoint": str(checkpoint_path),
            "null_key_competes_with_memory": False,
            "note": (
                "The current forward pass masks the null key whenever real memory "
                "exists; null attention mass is therefore expected to be zero."
            ),
            "groups": summaries,
            "ood_to_conflictqa_gate_ratio_last": ratio,
            "cases": groups,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        for index in layer_indices:
            adapter.adapter(index).probe_sink = None
        for handle in handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare internal Memory Expert behavior in and out of distribution"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--in-distribution-cases", type=int, default=4)
    parser.add_argument("--ood-cases", type=int, default=8)
    parser.add_argument("--maximum-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_probe(
        args.checkpoint.resolve(), args.corpus.resolve(), args.ingest.resolve(),
        args.questions.resolve(), args.output.resolve(), args.in_distribution_cases,
        args.ood_cases, args.maximum_new_tokens, torch.device(args.device),
    )
    print(json.dumps({
        "event": "mechanism-probe-summary",
        "groups": {
            name: {
                "cases": group["cases"],
                "prefill_layer_mean": group["prefill_layer_mean"],
            }
            for name, group in result["groups"].items()
        },
        "ood_to_conflictqa_gate_ratio_last": result[
            "ood_to_conflictqa_gate_ratio_last"
        ],
        "null_key_competes_with_memory": result["null_key_competes_with_memory"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
