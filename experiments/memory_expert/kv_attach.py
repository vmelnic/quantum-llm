"""KV-attach prototype milestone 1: mechanical validation.

Attached memory is prefilled once through frozen Qwen3-4B and its per-layer
K/V cache is kept. Query tokens then run on top of that cache, so the model's
own native attention reads the memory with contiguous prefix RoPE geometry —
no trained read-out, no hooks, no residual injection. This module validates
the mechanics before any gate/null training:

- identity: a split forward (prefill memory, then query on the cache) must
  reproduce a joint forward over the same tokens numerically;
- nacre: the invented-English OOD dossier admitted as attached KV must
  reproduce the full-context control quality (8/8 strict answers), with the
  full-context and no-memory arms rerun in the same artifact for direct
  comparison.

Design: docs/memory-kv-attach-v1.md. Gates and the learned null K/V arrive
in milestone 2; nothing here is trained.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import time
from pathlib import Path
from typing import Sequence

import torch
from transformers import DynamicCache

try:
    from .pow import chat_prompt
    from .synthetic_memory import (
        normalized_contains, parse_pointer, parse_response, split_sentences,
    )
except ImportError:  # Direct script execution on the Windows worker.
    from pow import chat_prompt
    from synthetic_memory import (
        normalized_contains, parse_pointer, parse_response, split_sentences,
    )

from transformers import AutoModelForCausalLM, AutoTokenizer


# The attach arm must not reuse pow.SYSTEM_PROMPT: that prompt explicitly
# tells the model "the source text is not present in this prompt", which is
# false here and caused the model to ignore the attached prefix.
ATTACH_SYSTEM_PROMPT = """You answer using an attached authoritative memory: source
records prepended to this conversation as an attached prefix. If no memory is
attached, or the attached memory does not contain evidence for the question,
answer exactly: I don't know from the attached memory. Otherwise, answer in
the same language as the user's question. Never use general knowledge for
factual answers. Return two lines only. Start the first with `ANSWER: `
followed by the answer. Start the second with `SOURCES: ` followed by
comma-separated zero-based source slots or `NONE`. Source slots order the
records as attached, starting at 0."""

MEMORY_HEADER = "Attached authoritative memory records:"

# Synthetic-turn wrapping: the attached memory occupies the position of a
# prior conversational exchange instead of a raw document prefix. The raw
# prefix arm fails because the model continues the document; a prior-turn
# position is the natural place where the model expects usable evidence.
MEMORY_TURN_USER = (
    "Here are the attached authoritative memory records. "
    "Remember them and answer only from them."
)
MEMORY_TURN_ACK = (
    "I have received the attached records and will answer only from them."
)


def _normalize_template_ids(result) -> list[int]:
    """Same normalization as pow.chat_prompt for apply_chat_template output."""
    if isinstance(result, dict) or hasattr(result, "keys"):
        result = result["input_ids"]
    if isinstance(result, torch.Tensor):
        result = result.flatten().tolist()
    elif isinstance(result, str):
        raise RuntimeError("chat template returned text despite tokenize=True")
    elif result and isinstance(result[0], (list, tuple)):
        if len(result) != 1:
            raise ValueError("chat template returned more than one sequence")
        result = result[0]
    if not all(isinstance(token, int) for token in result):
        raise TypeError("chat template returned non-integer tokens")
    return list(result)


def _rfind_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    for index in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[index:index + len(needle)]) == list(needle):
            return index
    return -1


def build_segments(tokenizer, record_texts: Sequence[str],
                   question: str) -> tuple[list[int], list[int]]:
    """Render the full conversation once at token level, then split.

    messages = system, memory user turn, assistant ack, real question. The
    attached prefix is the token ids before the real question turn; the query
    is the question turn plus the generation prompt. Splitting token ids
    (not strings) keeps special tokens exactly as the template produced them.
    """
    messages = [
        {"role": "system", "content": ATTACH_SYSTEM_PROMPT},
        {"role": "user", "content": MEMORY_TURN_USER + "\n\n" + "\n\n".join(record_texts)},
        {"role": "assistant", "content": MEMORY_TURN_ACK},
        {"role": "user", "content": question},
    ]
    options = dict(tokenize=True, add_generation_prompt=True)
    try:
        result = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **options
        )
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **options)
    full_ids = _normalize_template_ids(result)
    marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    split = _rfind_subsequence(full_ids, marker)
    if split < 0:
        raise RuntimeError("chat template split failed: no question turn")
    return full_ids[:split], full_ids[split:]


def attach_prompt(tokenizer, question: str) -> list[int]:
    return chat_prompt(tokenizer, question, system_prompt=ATTACH_SYSTEM_PROMPT)


def load_frozen_model(model_id: str, revision: str, dtype: torch.dtype,
                      device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def load_ingest_records(ingest_root: Path) -> list[dict[str, object]]:
    shards = sorted(ingest_root.glob("records-*.jsonl"))
    if not shards:
        raise ValueError(f"no records shards under {ingest_root}")
    records: list[dict[str, object]] = []
    for shard in shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_questions(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def expected_slot(records: Sequence[dict[str, object]],
                  sections: Sequence[str]) -> list[int]:
    slots = []
    labels = [str(record.get("section_label", "")) for record in records]
    for section in sections:
        wanted = section.lstrip("#").strip().casefold()
        for index, label in enumerate(labels):
            if label.lstrip("#").strip().casefold() == wanted:
                slots.append(index)
                break
        else:
            # Datasets like the Romanian criminal code express sections as
            # regular expressions against the label.
            for index, label in enumerate(labels):
                if re.search(section, label):
                    slots.append(index)
                    break
    return slots


def encode_memory(tokenizer, record_texts: Sequence[str],
                  maximum_tokens: int, header: str = "") -> list[int]:
    """Bare prefix: optional header, record bodies joined by blank lines."""
    token_ids: list[int] = []
    if header:
        token_ids.extend(tokenizer.encode(header, add_special_tokens=False))
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    for text in record_texts:
        if token_ids:
            token_ids.extend(separator)
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return token_ids[:maximum_tokens]


@torch.inference_mode()
def prefill_memory(model, memory_ids: Sequence[int],
                   device: torch.device,
                   lora_modules=None) -> tuple[tuple[tuple[torch.Tensor, ...], ...], int]:
    """Run the memory prefix once and keep per-layer (key, value) tensors.

    Keys carry contiguous RoPE positions 0..M-1, exactly as if the records
    had been read as a text prefix. Tensors are stored in legacy cache form
    so every query can rebuild a fresh DynamicCache without mutation risk.
    When a LoRA adapter is loaded, prefill always runs with it disabled:
    attached records keep their native frozen K/V, at serving as in training.
    """
    if not memory_ids:
        return tuple(), 0
    input_ids = torch.tensor([memory_ids], dtype=torch.long, device=device)
    if lora_modules:
        # Local import: kv_attach_train already imports this module.
        from kv_attach_train import lora_disabled
        guard_context = lora_disabled(lora_modules)
    else:
        guard_context = contextlib.nullcontext()
    with guard_context:
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
        )
    past = outputs.past_key_values
    if hasattr(past, "layers"):  # transformers layered-cache API
        states = tuple((layer.keys, layer.values) for layer in past.layers)
    elif hasattr(past, "to_legacy_cache"):
        states = tuple(past.to_legacy_cache())
    else:
        states = tuple(zip(past.key_cache, past.value_cache))
    return tuple(
        (key.detach(), value.detach()) for key, value in states
    ), len(memory_ids)


def fresh_cache(memory_states: tuple[tuple[torch.Tensor, ...], ...]) -> DynamicCache | None:
    if not memory_states:
        return None
    cache = DynamicCache()
    for layer_index, (key, value) in enumerate(memory_states):
        cache.update(key, value, layer_index)
    return cache


@torch.inference_mode()
def generate(model, tokenizer, prompt_ids: Sequence[int],
             memory_states: tuple[tuple[torch.Tensor, ...], ...],
             maximum_new_tokens: int, device: torch.device) -> str:
    past = fresh_cache(memory_states)
    # key shape: (batch, kv_heads, sequence, head_dim)
    past_length = int(memory_states[0][0].shape[2]) if memory_states else 0
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(maximum_new_tokens):
        attention_mask = torch.ones(
            1, past_length + input_ids.shape[1],
            dtype=torch.long, device=device,
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        token = int(torch.argmax(outputs.logits[0, -1]).item())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        past = outputs.past_key_values
        past_length = past.get_seq_length() if past is not None else 0
        input_ids = torch.tensor([[token]], dtype=torch.long, device=device)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def identity_check(model, tokenizer, memory_ids: Sequence[int],
                   prompt_ids: Sequence[int], device: torch.device) -> dict[str, object]:
    """Split forward (cache) must match the joint forward over same tokens."""
    joint_ids = torch.tensor(
        [list(memory_ids) + list(prompt_ids)], dtype=torch.long, device=device
    )
    joint = model(
        input_ids=joint_ids,
        attention_mask=torch.ones_like(joint_ids),
        use_cache=False,
        return_dict=True,
    ).logits[0, len(memory_ids):].float()

    memory_states, memory_length = prefill_memory(model, memory_ids, device)
    past = fresh_cache(memory_states)
    query = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones(1, memory_length + query.shape[1], dtype=torch.long, device=device)
    split = model(
        input_ids=query,
        attention_mask=mask,
        past_key_values=past,
        use_cache=False,
        return_dict=True,
    ).logits[0].float()

    difference = (joint - split).abs()
    return {
        "memory_tokens": memory_length,
        "query_tokens": len(prompt_ids),
        "max_abs_logit_diff": float(difference.max().item()),
        "mean_abs_logit_diff": float(difference.mean().item()),
        "argmax_agreement": float(
            (joint.argmax(dim=-1) == split.argmax(dim=-1)).float().mean().item()
        ),
    }


def score_case(row: dict[str, object], records: Sequence[dict[str, object]],
               response: str) -> dict[str, object]:
    answer, slots = parse_response(response)
    # Pointer contract: the authority plane renders the pointed sentence
    # verbatim; scoring checks the expected literal against the rendering.
    rendered = None
    pointer = parse_pointer(answer)
    if pointer is not None:
        slot, sentence_index = pointer
        if 0 <= slot < len(records):
            text = str(records[slot].get("text", ""))
            if sentence_index is None:
                # Slot-level pointer: the authority plane renders the record.
                rendered = text
            else:
                sentences = split_sentences(text)
                if 0 <= sentence_index < len(sentences):
                    rendered = sentences[sentence_index]
    answer_text = rendered if rendered is not None else answer
    expected_answers = [str(item) for item in row.get("expected_answers", [])]
    answer_strict = any(
        normalized_contains(answer_text, expected) for expected in expected_answers
    )
    wanted_slots = expected_slot(records, row.get("expected_sections", []))
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "response": response,
        "parsed_answer": answer,
        "rendered_answer": rendered,
        "parsed_source_slots": slots,
        "expected_slots": wanted_slots,
        "answer_strict": answer_strict,
        # parse_response returns a tuple; compare in list form.
        "source_ok": list(slots) == wanted_slots,
        "joint": answer_strict and list(slots) == wanted_slots,
    }


def summarize(cases: Sequence[dict[str, object]]) -> dict[str, object]:
    total = len(cases)
    return {
        "cases": total,
        "answer_strict": sum(1 for case in cases if case["answer_strict"]),
        "source_ok": sum(1 for case in cases if case["source_ok"]),
        "joint": sum(1 for case in cases if case["joint"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--revision", default="1cfa9a7208912126459214e8b04321603b3df60c"
    )
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-memory-tokens", type=int, default=768)
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    # Selection mode: a query report with per-question retrieval hits; each
    # question then attaches only its retrieved records (the real serving
    # path for corpora too large to attach whole).
    parser.add_argument("--selection", type=Path, default=None)
    # Optional trained LoRA checkpoint from kv_attach_train.py. The memory
    # prefill keeps LoRA disabled; only the query side uses it.
    parser.add_argument("--lora", type=Path, default=None)
    # float32 makes the identity check decisive; bf16 split-vs-joint logits
    # differ by accumulation order and obscure real plumbing bugs.
    parser.add_argument("--dtype", choices=("bfloat16", "float32"),
                        default="float32")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    started = time.perf_counter()
    device = torch.device(arguments.device)
    dtype = getattr(torch, arguments.dtype)
    tokenizer, model = load_frozen_model(
        arguments.model, arguments.revision, dtype, device
    )
    lora_modules = None
    if arguments.lora is not None:
        from kv_attach_train import load_lora_into
        checkpoint = torch.load(arguments.lora, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture_version") not in (
            "kv-attach-lora-v1", "kv-attach-lora-v2"
        ):
            raise RuntimeError("unrecognized LoRA checkpoint architecture")
        lora_modules = load_lora_into(model, checkpoint, device)

    records = load_ingest_records(arguments.ingest)
    questions = read_questions(arguments.questions)

    if arguments.selection is not None:
        report = json.loads(arguments.selection.read_text(encoding="utf-8"))
        hits_by_id = {
            str(result["id"]): result["hits"] for result in report["results"]
        }
        records_by_id = {
            str(record["record_id"]): record for record in records
        }
        arms = {"kv_attach_turn": [], "no_memory": []}
        for row in questions:
            question = str(row["question"])
            hits = hits_by_id.get(str(row["id"]), [])
            selected = [
                records_by_id[str(hit["record_id"])]
                for hit in hits
                if str(hit["record_id"]) in records_by_id
            ]
            selected_texts = [str(record["text"]) for record in selected]
            prefix_ids, query_ids = build_segments(
                tokenizer, selected_texts, question
            )
            prefix_ids = prefix_ids[:arguments.maximum_memory_tokens]
            states, _ = prefill_memory(model, prefix_ids, device, lora_modules)
            attached = generate(
                model, tokenizer, query_ids,
                states, arguments.maximum_new_tokens, device,
            )
            arms["kv_attach_turn"].append(score_case(row, selected, attached))
            blind = generate(
                model, tokenizer, attach_prompt(tokenizer, question),
                tuple(), arguments.maximum_new_tokens, device,
            )
            arms["no_memory"].append(score_case(row, selected, blind))
        artifact = {
            "schema_version": 5,
            "tool": "kv-attach-validate",
            "mode": "selection",
            "model": arguments.model,
            "revision": arguments.revision,
            "dtype": arguments.dtype,
            "records": len(records),
            "questions": len(questions),
            "summary": {arm: summarize(cases) for arm, cases in arms.items()},
            "arms": arms,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(artifact["summary"], indent=2))
        return 0

    record_texts = [str(record["text"]) for record in records]
    memory_ids = encode_memory(
        tokenizer, record_texts, arguments.maximum_memory_tokens,
        header=MEMORY_HEADER,
    )
    memory_states, memory_tokens = prefill_memory(
        model, memory_ids, device, lora_modules
    )

    # Synthetic-turn arm: full conversation rendered once at token level and
    # split before the real question turn; the prefix is question-independent.
    turn_prefix_ids, turn_query_ids = build_segments(
        tokenizer, record_texts, str(questions[0]["question"])
    )
    turn_prefix_ids = turn_prefix_ids[:arguments.maximum_memory_tokens]
    turn_states, turn_memory_tokens = prefill_memory(
        model, turn_prefix_ids, device, lora_modules
    )
    turn_token_audit = tokenizer.convert_ids_to_tokens(
        turn_prefix_ids[-8:] + turn_query_ids[:16]
    )

    probe_question = str(questions[0]["question"])
    identity = None
    if lora_modules is None:
        # The identity check compares against a joint forward, which is only
        # meaningful for the purely frozen model.
        identity = identity_check(
            model, tokenizer, memory_ids,
            attach_prompt(tokenizer, probe_question), device,
        )

    control_context = "\n\n".join(
        f"source {slot}:\n{text}" for slot, text in enumerate(record_texts)
    )
    arms: dict[str, list[dict[str, object]]] = {
        "kv_attach": [], "kv_attach_turn": [], "full_context": [],
        "no_memory": [],
    }
    for row in questions:
        question = str(row["question"])
        attached = generate(
            model, tokenizer, attach_prompt(tokenizer, question),
            memory_states, arguments.maximum_new_tokens, device,
        )
        arms["kv_attach"].append(score_case(row, records, attached))
        _, turn_query_ids = build_segments(tokenizer, record_texts, question)
        turn_attached = generate(
            model, tokenizer, turn_query_ids,
            turn_states, arguments.maximum_new_tokens, device,
        )
        arms["kv_attach_turn"].append(score_case(row, records, turn_attached))
        in_context = generate(
            model, tokenizer,
            chat_prompt(tokenizer, question, control_context=control_context),
            tuple(), arguments.maximum_new_tokens, device,
        )
        arms["full_context"].append(score_case(row, records, in_context))
        blind = generate(
            model, tokenizer, attach_prompt(tokenizer, question),
            tuple(), arguments.maximum_new_tokens, device,
        )
        arms["no_memory"].append(score_case(row, records, blind))

    artifact = {
        "schema_version": 5,
        "tool": "kv-attach-validate",
        "model": arguments.model,
        "revision": arguments.revision,
        "dtype": arguments.dtype,
        "lora": str(arguments.lora) if arguments.lora else None,
        "memory_header": MEMORY_HEADER,
        "turn_token_audit": turn_token_audit,
        "records": len(records),
        "questions": len(questions),
        "memory_tokens": memory_tokens,
        "turn_memory_tokens": turn_memory_tokens,
        "identity": identity,
        "summary": {arm: summarize(cases) for arm, cases in arms.items()},
        "arms": arms,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if identity is not None:
        print(json.dumps(identity))
    print(json.dumps(artifact["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
