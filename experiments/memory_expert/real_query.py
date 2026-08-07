from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    from .data_contract import KnowledgeRecord, SourceSpan, content_sha256
    from .dense_index import SearchHit, TransformerDenseEncoder, search
    from .pow import (
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
    from data_contract import KnowledgeRecord, SourceSpan, content_sha256
    from dense_index import SearchHit, TransformerDenseEncoder, search
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


def as_memory_record(record: KnowledgeRecord,
                     record_format: str = "metadata") -> MemoryRecord:
    if record_format not in {"metadata", "raw"}:
        raise ValueError(f"unsupported memory record format: {record_format}")
    return MemoryRecord(
        record_id=record.record_id,
        citation_id=record.citation_id,
        text=record.text if record_format == "raw" else record.memory_text,
        shard_key=f"{record.namespace}:{record.document_id}",
        generation=1,
        acl=",".join(record.acl),
        language=record.language,
        indexable=True,
    )


def section_matches(record: KnowledgeRecord, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, record.section_label, re.IGNORECASE) for pattern in patterns)


def candidate_subspans(record: KnowledgeRecord, maximum_characters: int = 900
                       ) -> list[tuple[int, int]]:
    """Return exact paragraph/sentence spans for second-stage admission."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", record.text, re.DOTALL):
        text = match.group(0)
        if " ".join(text.split()) == " ".join(record.section_label.split()):
            continue
        if len(text) <= maximum_characters:
            spans.append((match.start(), match.end()))
            continue
        cursor = match.start()
        for sentence in re.finditer(r"\S.*?(?:[.!?;](?=\s|\Z)|\Z)", text, re.DOTALL):
            start = match.start() + sentence.start()
            end = match.start() + sentence.end()
            if end - start > maximum_characters:
                for offset in range(start, end, maximum_characters):
                    spans.append((offset, min(offset + maximum_characters, end)))
            else:
                spans.append((start, end))
            cursor = end
        if cursor < match.end():
            spans.append((cursor, match.end()))
    return spans or [(0, len(record.text))]


def rerank_admitted_segments(encoder: TransformerDenseEncoder,
                             query_vector: np.ndarray,
                             hits: Sequence[SearchHit]) -> list[SearchHit]:
    reranked: list[SearchHit] = []
    for hit in hits:
        spans = candidate_subspans(hit.record)
        texts = [
            f"{hit.record.section_label}\n{hit.record.text[start:end]}"
            for start, end in spans
        ]
        vectors = encoder.encode(texts, batch_size=min(16, len(texts)))
        scores = vectors @ np.asarray(query_vector, dtype=np.float32)
        selected = int(np.argmax(scores))
        relative_start, relative_end = spans[selected]
        selected_text = hit.record.text[relative_start:relative_end]
        absolute_start = hit.record.source_span.character_start + relative_start
        absolute_end = hit.record.source_span.character_start + relative_end
        selected_hash = content_sha256(selected_text)
        record = replace(
            hit.record,
            record_id=(
                f"{hit.record.record_id}-S{absolute_start}-{absolute_end}-"
                f"{selected_hash[:12]}"
            ),
            text=selected_text,
            content_sha256=selected_hash,
            source_span=SourceSpan(absolute_start, absolute_end),
            attributes={
                **hit.record.attributes,
                "parent_record_id": hit.record.record_id,
                "segment_reranked": True,
            },
        )
        reranked.append(SearchHit(record, hit.score, float(scores[selected])))
    return reranked


def run_queries(ingest_root: Path, index_root: Path, checkpoint_path: Path,
                question_path: Path, output_path: Path, encoder_model: str,
                encoder_revision: str, top_k: int, maximum_memory_tokens: int,
                maximum_new_tokens: int, acl: set[str], language: str | None,
                device: torch.device, memory_record_format: str = "metadata",
                ) -> dict[str, object]:
    started = time.perf_counter()
    questions = read_questions(question_path)
    encoder_started = time.perf_counter()
    encoder = TransformerDenseEncoder(
        encoder_model, encoder_revision, device, maximum_tokens=1024
    )
    encoder_loaded = time.perf_counter()
    query_vectors = encoder.encode([str(row["question"]) for row in questions])
    encoded_queries = time.perf_counter()
    retrievals: list[list[SearchHit]] = []
    retrieval_seconds: list[float] = []
    for row, query_vector in zip(questions, query_vectors):
        query_started = time.perf_counter()
        hits = search(
            ingest_root, index_root, query_vector, maximum=top_k,
            allowed_acl=acl, language=language,
        )
        retrievals.append(rerank_admitted_segments(encoder, query_vector, hits))
        retrieval_seconds.append(time.perf_counter() - query_started)
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    retrieval_finished = time.perf_counter()

    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PowConfig(**raw_checkpoint["config"])
    config = PowConfig(**{
        **config.__dict__,
        "maximum_memory_tokens": maximum_memory_tokens,
        "maximum_new_tokens": maximum_new_tokens,
    })
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
        selected_ids = [
            tuple(hit.record.record_id for hit in hits) for hits in retrievals
        ]
        admitted = {
            hit.record.record_id: as_memory_record(hit.record, memory_record_format)
            for hits in retrievals for hit in hits
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
            answer, model_source_slots = parse_response(response)
            hit_records = [hit.record for hit in hits]
            model_sources_authorized = (
                bool(model_source_slots)
                and len(set(model_source_slots)) == len(model_source_slots)
                and all(0 <= slot < len(hit_records) for slot in model_source_slots)
            )
            selected_records = (
                [hit_records[slot] for slot in model_source_slots]
                if model_sources_authorized else []
            )
            evidence_citations = tuple(
                record.citation_id for record in selected_records
            )
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
            answer_text_match = (
                not expected_answers
                or all(normalized_contains(answer, str(value)) for value in expected_answers)
            )
            evidence_authorized = bool(selected_records) and all(
                acl.intersection(record.acl) for record in selected_records
            )
            result = {
                "id": row["id"],
                "question": row["question"],
                "response": response,
                "answer": answer,
                "model_source_slots": model_source_slots,
                "model_sources_authorized": model_sources_authorized,
                "citations": evidence_citations,
                "answer_text_match": answer_text_match,
                "retrieval_ok": retrieval_ok,
                "citations_authorized": evidence_authorized,
                "passed": answer_text_match and retrieval_ok and evidence_authorized,
                "retrieval_seconds": retrieval_elapsed,
                "generation_seconds": generation_elapsed,
                "hits": [
                    {
                        "score": hit.score,
                        "segment_score": hit.segment_score,
                        "record_id": hit.record.record_id,
                        "parent_record_id": hit.record.attributes.get("parent_record_id"),
                        "citation_id": hit.record.citation_id,
                        "section_label": hit.record.section_label,
                        "source_span": hit.record.to_dict()["source_span"],
                    }
                    for hit in hits
                ],
                "rendered_citations": [
                    {
                        "source_slot": slot,
                        "citation_id": record.citation_id,
                        "section_label": record.section_label,
                        "quote": record.text,
                    }
                    for slot, record in zip(model_source_slots, selected_records)
                ],
            }
            results.append(result)
            print(json.dumps({"event": "real-query", **result}, ensure_ascii=False), flush=True)
        finished = time.perf_counter()
        count = len(results)
        summary = {
            "schema_version": 3,
            "contract": "quantum-llm-memory-query-report-v3",
            "memory_record_format": memory_record_format,
            "questions": count,
            "passed": sum(bool(row["passed"]) for row in results),
            "strict_text_match_rate": sum(
                bool(row["answer_text_match"]) for row in results
            ) / count,
            "semantic_answer_accuracy": None,
            "manual_semantic_review_required": True,
            "retrieval_recall": sum(bool(row["retrieval_ok"]) for row in results) / count,
            "authorized_evidence_rate": sum(
                bool(row["citations_authorized"]) for row in results
            ) / count,
            "authorized_model_source_rate": sum(
                bool(row["model_sources_authorized"]) for row in results
            ) / count,
            "timing_seconds": {
                "encoder_load": encoder_loaded - encoder_started,
                "query_encode": encoded_queries - encoder_loaded,
                "retrieval_total": retrieval_finished - encoded_queries,
                "model_load": model_loaded - retrieval_finished,
                "memory_encode": memory_finished - memory_started,
                "generation_total": sum(
                    float(result["generation_seconds"]) for result in results
                ),
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
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--acl", action="append")
    parser.add_argument("--language")
    parser.add_argument(
        "--memory-record-format", choices=("metadata", "raw"), default="metadata",
        help="Render admitted memory with source metadata or as its raw text body",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_queries(
        args.ingest.resolve(), args.index.resolve(), args.checkpoint.resolve(),
        args.questions.resolve(), args.output.resolve(), args.encoder_model,
        args.encoder_revision, args.top_k, args.maximum_memory_tokens,
        args.maximum_new_tokens, set(args.acl or ["public"]), args.language,
        torch.device(args.device), args.memory_record_format,
    )
    print(json.dumps({
        "event": "real-query-summary",
        **{key: value for key, value in result.items() if key != "results"},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
