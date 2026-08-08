from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
        load_adapter,
        load_model,
        resolve_layers,
        score_example,
        seed_everything,
        select_eval_examples,
        set_memory_batch,
    )
    from .real_query import read_questions
    from .synthetic_memory import (
        MemoryRecord,
        format_memory,
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
        load_adapter,
        load_model,
        resolve_layers,
        score_example,
        seed_everything,
        select_eval_examples,
        set_memory_batch,
    )
    from real_query import read_questions
    from synthetic_memory import (
        MemoryRecord,
        format_memory,
        normalized_contains,
        parse_response,
    )


class TraceCollector:
    """Collects per-layer sink events for exactly one forward call at a time."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def __call__(self, event: dict[str, object]) -> None:
        self.events.append(event)

    def take(self) -> list[dict[str, object]]:
        events, self.events = self.events, []
        return events


def tokenize_memory(tokenizer, records: Sequence[MemoryRecord],
                    maximum_tokens: int) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Reproduce encode_memory_sets tokenization and keep token/offset mapping."""
    text = format_memory(records)
    encoded = tokenizer(
        text, truncation=True, max_length=maximum_tokens,
        return_offsets_mapping=True,
    )
    return text, list(encoded["input_ids"]), [
        (int(start), int(end)) for start, end in encoded["offset_mapping"]
    ]


def gold_memory_indices(text: str, answer: str,
                        offsets: Sequence[tuple[int, int]]) -> list[int]:
    start = text.casefold().find(answer.casefold())
    if start < 0:
        return []
    end = start + len(answer)
    return [
        index for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]


def make_lens(model, device: torch.device):
    norm = getattr(getattr(model, "model", None), "norm", None)
    lm_head = model.get_output_embeddings()
    if norm is None or lm_head is None:
        raise RuntimeError("cannot locate final norm / output embeddings")

    def lens(vector: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return lm_head(
                norm(vector.to(device=device, dtype=torch.bfloat16))
            ).float().cpu()

    return lens


def token_ranks(logits: torch.Tensor, token_ids: Sequence[int]) -> dict[str, int]:
    return {
        str(token_id): int((logits > logits[token_id]).sum().item()) + 1
        for token_id in token_ids
    }


def process_step_events(events: Sequence[dict[str, object]], lens,
                        tokenizer, gold_indices: Sequence[int],
                        memory_token_strs: Sequence[str],
                        answer_token_ids: Sequence[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        layer = int(event["layer"])
        row: dict[str, object] = {
            "layer": layer,
            "phase": event["phase"],
            "gate_delta_hidden_ratio_last": event["gate_delta_hidden_ratio_last"],
        }
        weights = event.get("weights_last")
        if isinstance(weights, torch.Tensor):
            head_mean = weights[:, :-1].mean(dim=0)  # exclude the null column
            argmax_index = int(head_mean.argmax().item())
            row["attn_argmax_index"] = argmax_index
            row["attn_argmax_token"] = memory_token_strs[argmax_index]
            row["gold_attention_mass"] = (
                float(head_mean[list(gold_indices)].sum().item())
                if gold_indices else None
            )
        for name in ("context_last", "gated_last"):
            vector = event.get(name)
            if not isinstance(vector, torch.Tensor):
                continue
            logits = lens(vector)
            prefix = "context_lens" if name == "context_last" else "gated_lens"
            row[f"{prefix}_top1"] = tokenizer.decode([int(logits.argmax().item())])
            row[f"{prefix}_gold_ranks"] = token_ranks(logits, answer_token_ids)
        rows.append(row)
    rows.sort(key=lambda item: int(item["layer"]))
    return rows


@torch.inference_mode()
def traced_generate(model, tokenizer, hooks, memory_cache,
                    memory_ids: tuple[str, ...], prompt_ids: Sequence[int],
                    maximum_new_tokens: int, device: torch.device,
                    collector: TraceCollector, lens,
                    gold_indices: Sequence[int], memory_token_strs: Sequence[str],
                    answer_token_ids: Sequence[int]) -> tuple[str, list[dict[str, object]]]:
    """generate() with per-step sink capture; the math is identical to pow.generate."""
    set_memory_batch(
        hooks, memory_cache, [memory_ids], model.config.hidden_size, device
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    past = None
    generated: list[int] = []
    steps: list[dict[str, object]] = []
    for _ in range(maximum_new_tokens):
        collector.take()
        outputs = model(
            input_ids=input_ids if past is None else input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        events = collector.take()
        logits = outputs.logits[0, -1].float().cpu()
        token = int(logits.argmax().item())
        top = torch.topk(logits, 5)
        steps.append({
            "position": len(generated),
            "emitted_token": tokenizer.decode([token]),
            "emitted_token_id": token,
            "final_top5": [
                {"token": tokenizer.decode([int(item)]), "logit": float(value)}
                for item, value in zip(top.indices.tolist(), top.values.tolist())
            ],
            "final_gold_ranks": token_ranks(logits, answer_token_ids),
            "layers": process_step_events(
                events, lens, tokenizer, gold_indices, memory_token_strs,
                answer_token_ids,
            ),
        })
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        past = outputs.past_key_values
        input_ids = torch.cat((input_ids, torch.tensor([[token]], device=device)), dim=1)
        attention_mask = torch.ones_like(input_ids)
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response, steps


def memory_state_identity_lens(memory_cache, memory_ids: tuple[str, ...],
                               layer_indices: Sequence[int], lens,
                               gold_indices: Sequence[int],
                               memory_ids_vocab: Sequence[int],
                               memory_token_strs: Sequence[str]
                               ) -> list[dict[str, object]]:
    """Rank of each gold memory token's own id in the lens of its layer state."""
    rows: list[dict[str, object]] = []
    if not gold_indices:
        return rows
    states = memory_cache[memory_ids]
    for index in layer_indices:
        layer_states = states[index]
        ranks: dict[str, int] = {}
        for gold in gold_indices:
            token_id = int(memory_ids_vocab[gold])
            logits = lens(layer_states[gold])
            ranks[memory_token_strs[gold]] = (
                int((logits > logits[token_id]).sum().item()) + 1
            )
        rows.append({"layer": int(index), "gold_token_self_ranks": ranks})
    return rows


def run_trace(checkpoint_path: Path, corpus_path: Path, ingest_root: Path,
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
    lens = make_lens(model, device)
    collector = TraceCollector()
    hooks = {index: MemoryHook(adapter.adapter(index)) for index in layer_indices}
    for index in layer_indices:
        expert = adapter.adapter(index)
        expert.probe_layer_index = index
        expert.probe_sink = collector
        expert.probe_capture = True
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
        example
        for example in select_eval_examples(capability_examples, 64)
        if example.kind != "unknown"
    ]
    memory_sets = [example.memory_ids for example in candidates]
    memory_sets.extend(memory_ids for _, memory_ids in ood_rows)
    try:
        memory_cache = encode_memory_sets(
            model, tokenizer, hooks, records_by_id, memory_sets,
            config.maximum_memory_tokens, device,
        )
        cases: list[dict[str, object]] = []

        def trace_case(case_id: str, question: str, answer: str,
                       memory_ids: tuple[str, ...]) -> dict[str, object]:
            records = [records_by_id[item] for item in memory_ids]
            memory_text, memory_ids_vocab, offsets = tokenize_memory(
                tokenizer, records, config.maximum_memory_tokens
            )
            memory_token_strs = [
                tokenizer.decode([token_id]) for token_id in memory_ids_vocab
            ]
            gold_indices = gold_memory_indices(memory_text, answer, offsets)
            answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
            prompt = chat_prompt(tokenizer, question)
            response, steps = traced_generate(
                model, tokenizer, hooks, memory_cache, memory_ids, prompt,
                maximum_new_tokens, device, collector, lens, gold_indices,
                memory_token_strs, answer_token_ids,
            )
            return {
                "case_id": case_id,
                "question": question,
                "expected_answer": answer,
                "response": response,
                "memory_token_count": len(memory_ids_vocab),
                "memory_tokens": memory_token_strs,
                "gold_memory_indices": gold_indices,
                "gold_memory_tokens": [
                    memory_token_strs[index] for index in gold_indices
                ],
                "answer_vocab_token_ids": answer_token_ids,
                "memory_state_identity_lens": memory_state_identity_lens(
                    memory_cache, memory_ids, layer_indices, lens,
                    gold_indices, memory_ids_vocab, memory_token_strs,
                ),
                "steps": steps,
            }

        collected = 0
        for example in candidates:
            if collected >= in_distribution_cases:
                break
            traced = trace_case(
                f"conflictqa::{example.example_id}", example.question,
                example.answer, example.memory_ids,
            )
            scored = score_example(
                example, traced["response"], example.memory_ids, capability_by_id
            )
            if not scored["passed"]:
                continue
            traced["group"] = "conflictqa_correct"
            cases.append(traced)
            collected += 1
            print(json.dumps({
                "event": "mechanism-trace-case", "case_id": traced["case_id"],
                "group": traced["group"], "response": traced["response"],
            }, ensure_ascii=False), flush=True)

        for row, memory_ids in ood_rows:
            case_id = str(row["id"])
            expected = row.get("expected_answers", [])
            if isinstance(expected, str):
                expected = [expected]
            expected = [str(value) for value in expected] or [""]
            traced = trace_case(
                f"ood::{case_id}", str(row["question"]), expected[0], memory_ids,
            )
            answer, slots = parse_response(traced["response"])
            answer_ok = all(
                normalized_contains(answer, value) for value in expected
            )
            source_ok = slots == (0,)
            traced["group"] = (
                "ood_correct" if answer_ok and source_ok else "ood_failed"
            )
            traced["answer_ok"] = answer_ok
            traced["source_ok"] = source_ok
            cases.append(traced)
            print(json.dumps({
                "event": "mechanism-trace-case", "case_id": traced["case_id"],
                "group": traced["group"], "answer_ok": answer_ok,
                "source_ok": source_ok, "response": traced["response"],
            }, ensure_ascii=False), flush=True)

        result = {
            "schema_version": 1,
            "contract": "quantum-llm-memory-mechanism-trace-v1",
            "checkpoint": str(checkpoint_path),
            "cases": cases,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        for index in layer_indices:
            expert = adapter.adapter(index)
            expert.probe_sink = None
            expert.probe_capture = False
        for handle in handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-level attention and lens trace of the Memory Expert"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--in-distribution-cases", type=int, default=2)
    parser.add_argument("--ood-cases", type=int, default=8)
    parser.add_argument("--maximum-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_trace(
        args.checkpoint.resolve(), args.corpus.resolve(), args.ingest.resolve(),
        args.questions.resolve(), args.output.resolve(), args.in_distribution_cases,
        args.ood_cases, args.maximum_new_tokens, torch.device(args.device),
    )
    print(json.dumps({
        "event": "mechanism-trace-summary",
        "cases": [
            {"case_id": case["case_id"], "group": case["group"]}
            for case in result["cases"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
