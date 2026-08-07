from __future__ import annotations

import argparse
import contextlib
import gc
import json
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    from .data_contract import KnowledgeRecord, read_records
    from .dense_index import TransformerDenseEncoder, search
    from .pow import (
        MemoryExpert,
        MemoryExpertStack,
        MemoryHook,
        PowConfig,
        chat_prompt,
        encode_memory_sets,
        generate,
        load_adapter,
        load_model,
        resolve_layers,
        seed_everything,
    )
    from .synthetic_memory import MemoryRecord, normalized_contains, parse_response
except ImportError:  # Direct execution on a worker.
    from data_contract import KnowledgeRecord, read_records
    from dense_index import TransformerDenseEncoder, search
    from pow import (
        MemoryHook,
        PowConfig,
        chat_prompt,
        encode_memory_sets,
        generate,
        load_adapter,
        load_model,
        resolve_layers,
        seed_everything,
    )
    from synthetic_memory import MemoryRecord, normalized_contains, parse_response


def read_questions(path: Path) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise ValueError(f"invalid question at line {line_number}")
            row.setdefault("id", f"Q{line_number:04d}")
            questions.append(row)
    if not questions:
        raise ValueError("question file is empty")
    return questions


def as_memory_record(record: KnowledgeRecord) -> MemoryRecord:
    return MemoryRecord(
        record_id=record.record_id,
        citation_id=record.citation_id,
        text=record.memory_text,
        shard_key=f"{record.namespace}:{record.document_id}",
        generation=1,
        acl=",".join(record.acl),
        language=record.language,
        indexable=True,
    )


def section_matches(record: KnowledgeRecord, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, record.section_label, re.IGNORECASE) for pattern in patterns)


def run_queries(ingest_root: Path, index_root: Path, checkpoint_path: Path,
                question_path: Path, output_path: Path, encoder_model: str,
                encoder_revision: str, top_k: int, maximum_memory_tokens: int,
                maximum_new_tokens: int, acl: set[str], language: str | None,
                device: torch.device) -> dict[str, object]:
    started = time.perf_counter()
    questions = read_questions(question_path)
    encoder_started = time.perf_counter()
    encoder = TransformerDenseEncoder(
        encoder_model, encoder_revision, device, maximum_tokens=1024
    )
    encoder_loaded = time.perf_counter()
    query_vectors = encoder.encode([str(row["question"]) for row in questions])
    encoded_queries = time.perf_counter()
    retrievals: list[list[object]] = []
    retrieval_seconds: list[float] = []
    for row, query_vector in zip(questions, query_vectors):
        query_started = time.perf_counter()
        retrievals.append(search(
            ingest_root, index_root, query_vector, maximum=top_k,
            allowed_acl=acl, language=language,
        ))
        retrieval_seconds.append(time.perf_counter() - query_started)
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    retrieval_finished = time.perf_counter()

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PowConfig(**raw_checkpoint["config"])
    config = PowConfig(
        **{
            **config.__dict__,
            "maximum_memory_tokens": maximum_memory_tokens,
            "maximum_new_tokens": maximum_new_tokens,
        }
    )
    seed_everything(config.seed)
    tokenizer, model = load_model(config, device)
    model_loaded = time.perf_counter()
    layers = resolve_layers(model)
    checkpoint, _, adapter = load_adapter(checkpoint_path, model, device)
    layer_indices = tuple(int(index) for index in checkpoint["layer_indices"])
    hooks = {index: MemoryHook(adapter.adapter(index)) for index in layer_indices}
    handles = [
        layers[index].register_forward_hook(hooks[index]) for index in layer_indices
    ]
    try:
        knowledge_by_id = {
            record.record_id: record for record in read_records(ingest_root)
        }
        selected_ids = [
            tuple(hit.record.record_id for hit in hits) for hits in retrievals
        ]
        admitted = {
            record_id: as_memory_record(knowledge_by_id[record_id])
            for key in selected_ids for record_id in key
        }
        memory_started = time.perf_counter()
        memory_cache = encode_memory_sets(
            model, tokenizer, hooks, admitted, selected_ids,
            config.maximum_memory_tokens, device, batch_size=1,
        )
        memory_finished = time.perf_counter()
        results: list[dict[str, object]] = []
        for row, hits, memory_ids, retrieval_elapsed in zip(
            questions, retrievals, selected_ids, retrieval_seconds
        ):
            generation_started = time.perf_counter()
            response = generate(
                model, tokenizer, hooks, memory_cache, memory_ids,
                chat_prompt(tokenizer, str(row["question"])),
                config.maximum_new_tokens, device,
            )
            generation_elapsed = time.perf_counter() - generation_started
            answer, model_citations = parse_response(response)
            hit_records = [hit.record for hit in hits]
            citation_map = {record.citation_id: record for record in hit_records}
            # Citation identifiers and quotes are authority-plane output, not
            # free-form model output. The model's attempted IDs remain in the
            # report as a diagnostic, but can never authorize a source.
            evidence_citations = tuple(record.citation_id for record in hit_records)
            expected_answers = row.get("expected_answers", [])
            if isinstance(expected_answers, str):
                expected_answers = [expected_answers]
            expected_sections = row.get("expected_sections", [])
            if isinstance(expected_sections, str):
                expected_sections = [expected_sections]
            retrieval_ok = (
                not expected_sections
                or all(
                    any(section_matches(record, [str(pattern)]) for record in hit_records)
                    for pattern in expected_sections
                )
            )
            answer_ok = (
                not expected_answers
                or all(normalized_contains(answer, str(value)) for value in expected_answers)
            )
            model_citations_authorized = bool(model_citations) and all(
                citation in citation_map for citation in model_citations
            )
            evidence_authorized = bool(hit_records) and all(
                acl.intersection(record.acl) for record in hit_records
            )
            result = {
                "id": row["id"],
                "question": row["question"],
                "response": response,
                "answer": answer,
                "model_citations": model_citations,
                "model_citations_authorized": model_citations_authorized,
                "citations": evidence_citations,
                "answer_ok": answer_ok,
                "retrieval_ok": retrieval_ok,
                "citations_authorized": evidence_authorized,
                "passed": answer_ok and retrieval_ok and evidence_authorized,
                "retrieval_seconds": retrieval_elapsed,
                "generation_seconds": generation_elapsed,
                "hits": [
                    {
                        "score": hit.score,
                        "record_id": hit.record.record_id,
                        "citation_id": hit.record.citation_id,
                        "section_label": hit.record.section_label,
                        "source_span": hit.record.to_dict()["source_span"],
                    }
                    for hit in hits
                ],
                "rendered_citations": [
                    {
                        "citation_id": citation,
                        "section_label": citation_map[citation].section_label,
                        "quote": citation_map[citation].text,
                    }
                    for citation in evidence_citations
                ],
            }
            results.append(result)
            print(json.dumps({"event": "real-query", **result}, ensure_ascii=False), flush=True)
        finished = time.perf_counter()
        count = len(results)
        summary = {
            "schema_version": 1,
            "contract": "quantum-llm-memory-query-report-v1",
            "questions": count,
            "passed": sum(bool(row["passed"]) for row in results),
            "answer_accuracy": sum(bool(row["answer_ok"]) for row in results) / count,
            "retrieval_recall": sum(bool(row["retrieval_ok"]) for row in results) / count,
            "authorized_evidence_rate": sum(
                bool(row["citations_authorized"]) for row in results
            ) / count,
            "authorized_model_citation_rate": sum(
                bool(row["model_citations_authorized"]) for row in results
            ) / count,
            "timing_seconds": {
                "encoder_load": encoder_loaded - encoder_started,
                "query_encode": encoded_queries - encoder_loaded,
                "retrieval_total": retrieval_finished - encoded_queries,
                "model_load": model_loaded - retrieval_finished,
                "memory_encode": memory_finished - memory_started,
                "generation_total": sum(float(row["generation_seconds"]) for row in results),
                "total": finished - started,
            },
            "results": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        for handle in handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query real ingested data through Memory Expert")
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--encoder-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--maximum-memory-tokens", type=int, default=768)
    parser.add_argument("--maximum-new-tokens", type=int, default=96)
    parser.add_argument("--acl", action="append")
    parser.add_argument("--language")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_queries(
        args.ingest.resolve(), args.index.resolve(), args.checkpoint.resolve(),
        args.questions.resolve(), args.output.resolve(), args.encoder_model,
        args.encoder_revision, args.top_k, args.maximum_memory_tokens,
        args.maximum_new_tokens, set(args.acl or ["public"]), args.language,
        torch.device(args.device),
    )
    print(json.dumps({
        "event": "real-query-summary",
        **{key: value for key, value in result.items() if key != "results"},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
