"""KV-attach milestone 2: LoRA SFT for contract discipline.

Milestone 1 showed frozen Qwen3-4B reads an attached K/V prefix with native
attention at full-context quality. What it lacks is contract discipline:
abstention when the attached memory lacks evidence, valid source slots only,
and preferring the attached memory over parametric knowledge. Those are
output-distribution behaviours, so the trained surface is a LoRA on the
attention projections, not a memory-side gate.

Design invariants (see docs/memory-kv-attach-v1.md):

- the memory prefix is always prefilled with LoRA DISABLED, so attached
  records keep their native frozen K/V at training and at serving;
- the query+target forward runs with LoRA enabled on top of that frozen
  cache; gradients flow only through the query-side computation;
- targets reuse the v5 pointer contract (`ANSWER: @slot:sentence`), so the
  authority plane keeps owning verbatim text;
- unknown rows (memory attached, answer absent) train abstention.

Training geometry mirrors the v5 run: deterministic epochs, microbatch 1
with gradient accumulation, best checkpoint by held-out eval NLL.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional

try:
    from .capability_corpus import load_capability_corpus
    from .kv_attach import attach_prompt, build_segments, fresh_cache
    from .pow import resolve_layers, seed_everything
except ImportError:  # Direct script execution on the Windows worker.
    from capability_corpus import load_capability_corpus
    from kv_attach import attach_prompt, build_segments, fresh_cache
    from pow import resolve_layers, seed_everything
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRALinear(nn.Module):
    """Frozen base projection plus a trainable low-rank delta."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.enabled = True
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_a.weight, std=0.01)
        # Zero B preserves the frozen model exactly at initialization.
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.base(hidden)
        if self.enabled:
            delta = self.lora_b(self.lora_a(hidden.float())).to(hidden.dtype)
            output = output + delta * self.scale
        return output


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def apply_lora(model, rank: int, alpha: float,
               device: torch.device) -> list[LoRALinear]:
    modules: list[LoRALinear] = []
    for layer in resolve_layers(model):
        attention = layer.self_attn
        for name in LORA_TARGETS:
            base = getattr(attention, name, None)
            if not isinstance(base, nn.Linear):
                raise RuntimeError(f"cannot apply LoRA to attention.{name}")
            wrapped = LoRALinear(base, rank, alpha).to(device)
            wrapped.lora_a.weight.requires_grad_(True)
            wrapped.lora_b.weight.requires_grad_(True)
            setattr(attention, name, wrapped)
            modules.append(wrapped)
    return modules


@contextlib.contextmanager
def lora_disabled(modules: Sequence[LoRALinear]):
    previous = [module.enabled for module in modules]
    for module in modules:
        module.enabled = False
    try:
        yield
    finally:
        for module, flag in zip(modules, previous):
            module.enabled = flag


def lora_state_dict(modules: Sequence[LoRALinear]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for index, module in enumerate(modules):
        state[f"{index}.lora_a"] = module.lora_a.weight.detach().cpu()
        state[f"{index}.lora_b"] = module.lora_b.weight.detach().cpu()
    return state


def load_lora_into(model, checkpoint: dict[str, object],
                   device: torch.device) -> list[LoRALinear]:
    modules = apply_lora(
        model, int(checkpoint["rank"]), float(checkpoint["alpha"]), device
    )
    state = checkpoint["lora"]
    for index, module in enumerate(modules):
        module.lora_a.weight.data.copy_(state[f"{index}.lora_a"])
        module.lora_b.weight.data.copy_(state[f"{index}.lora_b"])
    for module in modules:
        module.eval()
    return modules


def load_trainable_model(model_id: str, revision: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


class PrefixCache:
    """Small LRU of frozen prefilled memory prefixes, keyed by record ids."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self.entries: OrderedDict[tuple[str, ...], object] = OrderedDict()

    def get(self, key: tuple[str, ...]):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key: tuple[str, ...], value: object) -> None:
        self.entries[key] = value
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


def prefill_frozen(model, lora_modules: Sequence[LoRALinear],
                   prefix_ids: Sequence[int], device: torch.device):
    """Prefill the memory prefix with LoRA disabled: native frozen K/V."""
    if not prefix_ids:
        return tuple()
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad(), lora_disabled(lora_modules):
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
        )
    past = outputs.past_key_values
    if hasattr(past, "layers"):
        states = tuple((layer.keys, layer.values) for layer in past.layers)
    else:
        states = tuple(zip(past.key_cache, past.value_cache))
    return tuple((key.detach(), value.detach()) for key, value in states)


def example_nll(model, lora_modules: Sequence[LoRALinear], tokenizer,
                prefix_states, query_ids: Sequence[int],
                target_ids: Sequence[int], device: torch.device,
                train: bool) -> torch.Tensor:
    past = fresh_cache(prefix_states)
    past_length = int(prefix_states[0][0].shape[2]) if prefix_states else 0
    ids = list(query_ids) + list(target_ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones(
        1, past_length + len(ids), dtype=torch.long, device=device
    )
    context = torch.no_grad() if not train else contextlib.nullcontext()
    with context:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=False,
            return_dict=True,
        ).logits.float()
    labels = torch.tensor(
        [list(target_ids)], dtype=torch.long, device=device
    )
    # Predict target tokens: logits at positions len(query)-1 .. end-1.
    start = len(query_ids) - 1
    prediction = logits[0, start:start + len(target_ids)]
    return functional.cross_entropy(prediction, labels[0])


def evaluate_nll(model, lora_modules, tokenizer, examples, records_by_id,
                 prefix_cache: PrefixCache, maximum_memory_tokens: int,
                 device: torch.device) -> float:
    total = 0.0
    for example in examples:
        key = example.memory_ids
        states = prefix_cache.get(key)
        if states is None:
            texts = [records_by_id[item].text for item in key]
            prefix_ids, query_ids = build_segments(
                tokenizer, texts, example.question
            )
            prefix_ids = prefix_ids[:maximum_memory_tokens]
            states = prefill_frozen(model, lora_modules, prefix_ids, device)
            prefix_cache.put(key, states)
        else:
            _, query_ids = build_segments(
                tokenizer, [records_by_id[item].text for item in key],
                example.question,
            )
        target_ids = tokenizer.encode(
            example.target + tokenizer.eos_token, add_special_tokens=False
        )
        loss = example_nll(
            model, lora_modules, tokenizer, states, query_ids, target_ids,
            device, train=False,
        )
        total += float(loss.item())
    return total / max(1, len(examples))


def train(arguments: argparse.Namespace) -> dict[str, object]:
    device = torch.device(arguments.device)
    seed_everything(arguments.seed)
    tokenizer, model = load_trainable_model(
        arguments.model, arguments.revision, device
    )
    lora_modules = apply_lora(model, arguments.rank, arguments.alpha, device)
    records, examples = load_capability_corpus(
        arguments.corpus,
        minimum_train_families_per_language=0,
        minimum_eval_families_per_language=0,
    )
    records_by_id = {record.record_id: record for record in records}
    train_examples = [example for example in examples if example.split == "train"]
    eval_pool = [example for example in examples if example.split == "eval"]
    rng = random.Random(arguments.seed)
    rng.shuffle(eval_pool)
    eval_examples = eval_pool[:arguments.eval_limit]
    if not train_examples or not eval_examples:
        raise RuntimeError("corpus lacks train or eval examples")

    prefix_cache = PrefixCache(capacity=arguments.prefix_cache)
    optimizer = torch.optim.AdamW(
        [parameter for module in lora_modules
         for parameter in (module.lora_a.weight, module.lora_b.weight)],
        lr=arguments.learning_rate, weight_decay=0.0,
    )

    started = time.perf_counter()
    best_nll = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    microstep = 0
    updates = 0
    total_microsteps = arguments.epochs * len(train_examples)
    history: list[dict[str, object]] = []
    for epoch in range(1, arguments.epochs + 1):
        order = list(train_examples)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for index, example in enumerate(order):
            # Empty-memory augmentation: unknown rows are also trained with
            # no attached prefix at all, so abstention covers the no-memory
            # case, not only the irrelevant-memory case.
            empty_memory = (
                example.kind == "unknown"
                and rng.random() < arguments.unknown_empty_prob
            )
            key = example.memory_ids
            texts = [records_by_id[item].text for item in key]
            if empty_memory:
                states = tuple()
                query_ids = attach_prompt(tokenizer, example.question)
            else:
                states = prefix_cache.get(key)
                prefix_ids, query_ids = build_segments(
                    tokenizer, texts, example.question
                )
                if states is None:
                    prefix_ids = prefix_ids[:arguments.maximum_memory_tokens]
                    states = prefill_frozen(
                        model, lora_modules, prefix_ids, device
                    )
                    prefix_cache.put(key, states)
            target_ids = tokenizer.encode(
                example.target + tokenizer.eos_token, add_special_tokens=False
            )
            loss = example_nll(
                model, lora_modules, tokenizer, states, query_ids,
                target_ids, device, train=True,
            ) / arguments.accumulation
            loss.backward()
            running += float(loss.item())
            microstep += 1
            if microstep % arguments.accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if microstep % 100 == 0:
                elapsed = time.perf_counter() - started
                rate = microstep / elapsed
                remaining = (total_microsteps - microstep) / max(rate, 1.0e-9)
                print(
                    f"[train] microstep {microstep}/{total_microsteps} "
                    f"epoch {epoch}/{arguments.epochs} updates {updates} "
                    f"nll {running / 100:.4f} "
                    f"elapsed {elapsed / 60:.1f}m eta {remaining / 60:.1f}m",
                    flush=True,
                )
                history.append({
                    "epoch": epoch, "microstep": microstep,
                    "mean_nll": round(running / 100, 4),
                })
                running = 0.0
        eval_nll = evaluate_nll(
            model, lora_modules, tokenizer, eval_examples, records_by_id,
            prefix_cache, arguments.maximum_memory_tokens, device,
        )
        improved = eval_nll < best_nll
        if improved:
            best_nll = eval_nll
            best_state = lora_state_dict(lora_modules)
        print(
            f"[eval] epoch {epoch}/{arguments.epochs} "
            f"eval_nll {eval_nll:.4f} best {best_nll:.4f}",
            flush=True,
        )
        history.append({
            "epoch": epoch, "eval_nll": round(eval_nll, 4),
            "best": improved,
        })

    if best_state is None:
        raise RuntimeError("no checkpoint improved over initialization")
    checkpoint = {
        "architecture_version": "kv-attach-lora-v2",
        "model": arguments.model,
        "revision": arguments.revision,
        "rank": arguments.rank,
        "alpha": arguments.alpha,
        "lora": best_state,
        "training": {
            "corpus": str(arguments.corpus),
            "epochs": arguments.epochs,
            "microsteps": microstep,
            "optimizer_updates": updates,
            "best_eval_nll": best_nll,
            "eval_limit": arguments.eval_limit,
            "seed": arguments.seed,
        },
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output / "kv-attach-lora.pt"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "schema_version": 1,
        "tool": "kv-attach-train",
        "checkpoint": str(checkpoint_path),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "microsteps": microstep,
        "optimizer_updates": updates,
        "best_eval_nll": best_nll,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (arguments.output / "train-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--revision", default="1cfa9a7208912126459214e8b04321603b3df60c"
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--eval-limit", type=int, default=128)
    # Fraction of unknown rows also trained with an empty memory prefix.
    parser.add_argument("--unknown-empty-prob", type=float, default=0.5)
    parser.add_argument("--prefix-cache", type=int, default=16)
    parser.add_argument("--maximum-memory-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    summary = train(parse_args())
    print(json.dumps({key: summary[key] for key in (
        "train_examples", "eval_examples", "microsteps",
        "optimizer_updates", "best_eval_nll", "elapsed_seconds",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
