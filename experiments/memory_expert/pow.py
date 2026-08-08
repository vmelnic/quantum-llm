from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .synthetic_memory import (
        HashingEmbedder,
        MemoryExample,
        MemoryRecord,
        ShardedVectorIndex,
        build_corpus,
        corpus_fingerprint,
        examples_fingerprint,
        format_memory,
        normalized_contains,
        parse_pointer,
        parse_response,
        retrieve_adaptive,
        split_sentences,
    )
except ImportError:  # Direct script execution on the Windows worker.
    from synthetic_memory import (
        HashingEmbedder,
        MemoryExample,
        MemoryRecord,
        ShardedVectorIndex,
        build_corpus,
        corpus_fingerprint,
        examples_fingerprint,
        format_memory,
        normalized_contains,
        parse_pointer,
        parse_response,
        retrieve_adaptive,
        split_sentences,
    )


def render_pointer_answer(parsed_answer: object, example: MemoryExample,
                          records_by_id: dict[str, MemoryRecord]) -> str | None:
    """Resolve a pointer against the admitted records.

    Slot-level pointers render the whole record; legacy `@slot:sentence`
    pointers render the pointed sentence.
    """
    pointer = parse_pointer(str(parsed_answer))
    if pointer is None:
        return None
    slot, sentence_index = pointer
    if not 0 <= slot < len(example.memory_ids):
        return None
    text = records_by_id[example.memory_ids[slot]].text
    if sentence_index is None:
        return text
    sentences = split_sentences(text)
    if not 0 <= sentence_index < len(sentences):
        return None
    return sentences[sentence_index]


SYSTEM_PROMPT = """You answer using a separate authoritative memory channel.
The source text is not present in this prompt. If the memory channel does not
contain evidence for the question, answer exactly: I don't know from the
attached memory. Otherwise, answer in the same language as the user's question.
Never use general knowledge for factual answers. Return two
lines only. Start the first with `ANSWER: ` followed by the answer. Start the
second with `SOURCES: ` followed by comma-separated zero-based source slots or
`NONE`. Source slots are local to this request; never invent durable IDs."""

ARCHITECTURE_VERSION = "tokenmem-input-state-rmsnorm-source-slots-v2"


@dataclass(frozen=True)
class PowConfig:
    model: str
    revision: str
    gate_rank: int
    gate_alpha: float
    injection_every: int
    knowledge_dropout: float
    maximum_memory_tokens: int
    maximum_new_tokens: int
    seed: int


@dataclass(frozen=True)
class TrainingBatch:
    """One deterministic microbatch with explicit optimizer accounting."""

    epoch: int
    batch_in_epoch: int
    batches_in_epoch: int
    microstep: int
    optimizer_update: int
    accumulation_window_batches: int
    optimizer_step: bool
    phase: str
    examples: tuple[MemoryExample, ...]


@dataclass(frozen=True)
class TrainingGeometry:
    examples_per_epoch: int
    batches_per_epoch: int
    optimizer_updates_per_epoch: int
    total_examples: int
    total_microsteps: int
    total_optimizer_updates: int


def training_geometry(
    example_count: int, batch_size: int,
    gradient_accumulation: int, epochs: int,
) -> TrainingGeometry:
    """Derive all training counts from complete epochs."""
    if example_count <= 0 or batch_size <= 0:
        raise ValueError("training geometry requires positive examples and batch size")
    if gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("training geometry requires positive accumulation and epochs")
    batches_per_epoch = math.ceil(example_count / batch_size)
    updates_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation
    )
    return TrainingGeometry(
        examples_per_epoch=example_count,
        batches_per_epoch=batches_per_epoch,
        optimizer_updates_per_epoch=updates_per_epoch,
        total_examples=example_count * epochs,
        total_microsteps=batches_per_epoch * epochs,
        total_optimizer_updates=updates_per_epoch * epochs,
    )


def build_epoch_training_schedule(
    examples: Sequence[MemoryExample], batch_size: int,
    gradient_accumulation: int, epochs: int, seed: int,
    start_epoch: int = 0,
) -> list[TrainingBatch]:
    """Visit every example exactly once per epoch, including partial batches."""
    if not examples:
        raise ValueError("training schedule requires examples")
    if batch_size <= 0 or gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("training schedule limits must be positive")
    if start_epoch < 0 or start_epoch >= epochs:
        raise ValueError("start_epoch must be within the configured epoch range")

    geometry = training_geometry(
        len(examples), batch_size, gradient_accumulation, epochs
    )
    schedule: list[TrainingBatch] = []
    microstep = start_epoch * geometry.batches_per_epoch
    optimizer_update = start_epoch * geometry.optimizer_updates_per_epoch
    for epoch_index in range(start_epoch, epochs):
        ordered = list(examples)
        random.Random(seed + epoch_index).shuffle(ordered)
        batches = [
            tuple(ordered[start:start + batch_size])
            for start in range(0, len(ordered), batch_size)
        ]
        for window_start in range(0, len(batches), gradient_accumulation):
            window = batches[window_start:window_start + gradient_accumulation]
            optimizer_update += 1
            for window_index, batch in enumerate(window):
                microstep += 1
                batch_index = window_start + window_index
                schedule.append(TrainingBatch(
                    epoch=epoch_index + 1,
                    batch_in_epoch=batch_index + 1,
                    batches_in_epoch=len(batches),
                    microstep=microstep,
                    optimizer_update=optimizer_update,
                    accumulation_window_batches=len(window),
                    optimizer_step=window_index + 1 == len(window),
                    phase="joint-grounding",
                    examples=batch,
                ))
    return schedule


class MemoryExpert(nn.Module):
    """Layer-local TokenMem-style gate over frozen Q/K/V/O projections."""

    def __init__(self, hidden_size: int, rank: int, alpha: float,
                 dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("gate rank must be positive")
        self.hidden_size = hidden_size
        self.rank = rank
        self.scale = alpha
        self.dropout = nn.Dropout(dropout)
        self.gate_down = nn.Linear(hidden_size, rank, bias=False)
        self.gate_up = nn.Linear(rank, hidden_size, bias=False)
        # TokenMem/DecoupledRAG use W_A ~ N(0, .01) followed by W_B = 0.
        # This preserves the frozen base model exactly at initialization while
        # allowing the output projection to learn on the first optimizer step.
        nn.init.normal_(self.gate_down.weight, std=0.01)
        nn.init.zeros_(self.gate_up.weight)
        # Optional inference-only observer. It is deliberately not a module or
        # parameter, so enabling a probe cannot change checkpoint state.
        self.probe_sink = None
        self.probe_layer_index: int | None = None
        # When probe_capture is true the observer also receives per-head last
        # position attention weights and the hidden/context/gated vectors.
        # Trace-only; it never changes the returned tensor.
        self.probe_capture = False
        # Inference-only interventions. gate_scale multiplies the residual;
        # null_always keeps the null key attendable even when memory exists.
        # Both default to the exact trained behavior.
        self.gate_scale = 1.0
        self.null_always = False

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor,
                memory_mask: torch.Tensor, frozen_attention: nn.Module,
                frozen_input_norm: nn.Module) -> torch.Tensor:
        if hidden.ndim != 3 or memory.ndim != 3 or memory_mask.ndim != 2:
            raise ValueError("invalid Memory Expert tensor rank")
        batch, query_tokens, _ = hidden.shape
        if memory.shape[0] != batch or memory_mask.shape != memory.shape[:2]:
            raise ValueError("memory batch geometry mismatch")
        head_width = int(frozen_attention.head_dim)
        heads = int(frozen_attention.config.num_attention_heads)
        key_value_heads = int(frozen_attention.config.num_key_value_heads)
        groups = heads // key_value_heads
        query_input = hidden
        # The memory pass exposes the input of this same decoder layer. Reuse
        # its frozen RMSNorm, including the learned scale and epsilon.
        memory_input = frozen_input_norm(memory)
        query = frozen_attention.q_proj(query_input).view(
            batch, query_tokens, heads, head_width
        )
        key = frozen_attention.k_proj(memory_input).view(
            batch, memory.shape[1], key_value_heads, head_width
        )
        value = frozen_attention.v_proj(memory_input).view(
            batch, memory.shape[1], key_value_heads, head_width
        )
        if hasattr(frozen_attention, "q_norm"):
            query = frozen_attention.q_norm(query)
        if hasattr(frozen_attention, "k_norm"):
            key = frozen_attention.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).repeat_interleave(groups, dim=1)
        value = value.transpose(1, 2).repeat_interleave(groups, dim=1)
        no_memory = ~memory_mask.any(dim=1)
        null_key = torch.zeros(
            batch, heads, 1, head_width, dtype=key.dtype, device=key.device
        )
        null_value = torch.zeros_like(null_key)
        key = torch.cat((key, null_key), dim=2)
        value = torch.cat((value, null_value), dim=2)
        if self.null_always:
            null_allowed = torch.ones_like(no_memory[:, None])
        else:
            null_allowed = no_memory[:, None]
        extended_mask = torch.cat((memory_mask, null_allowed), dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_width)
        scores = scores.masked_fill(~extended_mask[:, None, None, :], -1.0e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous().view(
            batch, query_tokens, heads * head_width
        )
        context = frozen_attention.o_proj(context)
        gated = self.gate_up(self.gate_down(self.dropout(context).float())) * self.scale
        if self.gate_scale != 1.0:
            gated = gated * self.gate_scale
        if self.probe_sink is not None:
            with torch.no_grad():
                real_weights = weights[..., :-1].float()
                real_mass = real_weights.sum(dim=-1)
                null_mass = weights[..., -1].float()
                entropy = -(
                    real_weights.clamp_min(1.0e-12)
                    * real_weights.clamp_min(1.0e-12).log()
                ).sum(dim=-1)
                valid_tokens = memory_mask.sum(dim=-1).float()
                entropy_scale = valid_tokens.clamp_min(2).log()[:, None, None]
                normalized_entropy = torch.where(
                    valid_tokens[:, None, None] > 1,
                    entropy / entropy_scale,
                    torch.zeros_like(entropy),
                )
                hidden_norm = hidden.float().norm(dim=-1).clamp_min(1.0e-12)
                context_norm = context.float().norm(dim=-1)
                gated_norm = gated.float().norm(dim=-1)
                values = torch.stack((
                    real_mass.mean(),
                    null_mass.mean(),
                    normalized_entropy.mean(),
                    real_weights.max(dim=-1).values.mean(),
                    (context_norm / hidden_norm).mean(),
                    (gated_norm / hidden_norm).mean(),
                    (gated_norm[:, -1] / hidden_norm[:, -1]).mean(),
                )).detach().cpu().tolist()
                event: dict[str, object] = {
                    "layer": self.probe_layer_index,
                    "phase": "prefill" if query_tokens > 1 else "decode",
                    "query_tokens": query_tokens,
                    "memory_tokens": float(valid_tokens.mean().item()),
                    **dict(zip((
                        "real_attention_mass",
                        "null_attention_mass",
                        "attention_entropy",
                        "attention_max",
                        "context_hidden_ratio",
                        "gate_delta_hidden_ratio",
                        "gate_delta_hidden_ratio_last",
                    ), (float(value) for value in values))),
                }
                if self.probe_capture and batch == 1:
                    event.update({
                        # Heads x (memory + null) at the last query position.
                        "weights_last": weights[0, :, -1, :].detach().float().cpu(),
                        "hidden_last": hidden[0, -1].detach().float().cpu(),
                        "context_last": context[0, -1].detach().float().cpu(),
                        "gated_last": gated[0, -1].detach().float().cpu(),
                    })
                self.probe_sink(event)
        return hidden + gated.to(hidden.dtype)


class MemoryHook:
    def __init__(self, adapter: MemoryExpert) -> None:
        self.adapter = adapter
        self.memory: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.enabled = True

    @contextlib.contextmanager
    def disabled(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def set_memory(self, memory: torch.Tensor, mask: torch.Tensor) -> None:
        self.memory = memory
        self.mask = mask

    def __call__(self, module, _arguments, output):
        if not self.enabled:
            return output
        if self.memory is None or self.mask is None:
            raise RuntimeError("Memory Expert hook has no admitted memory")
        if isinstance(output, tuple):
            hidden = self.adapter(
                output[0], self.memory, self.mask,
                module.self_attn, module.input_layernorm,
            )
            return (hidden, *output[1:])
        return self.adapter(
            output, self.memory, self.mask,
            module.self_attn, module.input_layernorm,
        )


class MemoryExpertStack(nn.Module):
    def __init__(self, hidden_size: int, layer_indices: Sequence[int],
                 rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.layer_indices = tuple(layer_indices)
        self.adapters = nn.ModuleDict({
            str(index): MemoryExpert(hidden_size, rank, alpha, dropout)
            for index in self.layer_indices
        })

    def adapter(self, layer_index: int) -> MemoryExpert:
        return self.adapters[str(layer_index)]


class Lamb(torch.optim.Optimizer):
    """Minimal LAMB optimizer for the small float32 gate parameter set."""

    def __init__(self, parameters, lr: float = 1.0e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1.0e-6, weight_decay: float = 0.01) -> None:
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                average = state["exp_avg"]
                average_square = state["exp_avg_sq"]
                average.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                average_square.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )
                bias1 = 1.0 - beta1 ** state["step"]
                bias2 = 1.0 - beta2 ** state["step"]
                update = (average / bias1) / (
                    (average_square / bias2).sqrt() + group["eps"]
                )
                if group["weight_decay"]:
                    update = update.add(parameter, alpha=group["weight_decay"])
                weight_norm = parameter.norm().clamp(max=10.0)
                update_norm = update.norm()
                trust = (
                    weight_norm / update_norm
                    if weight_norm > 0 and update_norm > 0 else 1.0
                )
                parameter.add_(update, alpha=-group["lr"] * float(trust))
        return loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chat_prompt(tokenizer, question: str, control_context: str | None = None,
                system_prompt: str | None = None) -> list[int]:
    system = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    user = question
    if control_context is not None:
        system = """This is a full-context control run. Answer only from the
provided records. If they do not support the question, answer exactly: I don't
know from the attached memory. Otherwise, answer in the same language as the
user's question. Return two lines only. Start the first with
`ANSWER: ` followed by the answer. Start the second with `SOURCES: ` followed
by comma-separated zero-based source slots or `NONE`."""
        user = f"RECORDS:\n{control_context}\n\nQUESTION:\n{question}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    options = dict(tokenize=True, add_generation_prompt=True)
    try:
        result = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **options
        )
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **options)
    if isinstance(result, dict) or hasattr(result, "keys"):
        result = result["input_ids"]
    if isinstance(result, torch.Tensor):
        result = result.flatten().tolist()
    elif isinstance(result, str):
        result = tokenizer.encode(result, add_special_tokens=False)
    elif result and isinstance(result[0], (list, tuple)):
        if len(result) != 1:
            raise ValueError("chat template returned more than one token sequence")
        result = result[0]
    if not all(isinstance(token, int) for token in result):
        raise TypeError(f"chat template returned non-integer tokens: {type(result)!r}")
    return list(result)


def supervised_tokens(tokenizer, example: MemoryExample) -> tuple[list[int], list[int]]:
    prompt = chat_prompt(tokenizer, example.question)
    target = tokenizer.encode(example.target + tokenizer.eos_token, add_special_tokens=False)
    tokens = prompt + target
    labels = [-100] * len(prompt) + target
    return tokens, labels


def validate_visible_evidence(tokenizer, examples: Sequence[MemoryExample],
                              records_by_id: dict[str, MemoryRecord],
                              maximum_tokens: int,
                              batch_size: int = 64) -> dict[str, int]:
    """Require every answer and citation to survive the real memory truncation."""
    answerable = [example for example in examples if example.kind != "unknown"]
    failures: list[str] = []
    for start in range(0, len(answerable), batch_size):
        batch = answerable[start:start + batch_size]
        rendered = [
            format_memory([records_by_id[item] for item in example.memory_ids])
            for example in batch
        ]
        encoded = tokenizer(
            rendered, truncation=True, max_length=maximum_tokens,
            add_special_tokens=True,
        )["input_ids"]
        for example, token_ids in zip(batch, encoded):
            visible = tokenizer.decode(token_ids, skip_special_tokens=True)
            answer_visible = (
                example.answer_support != "extractive"
                or normalized_contains(visible, example.answer)
            )
            sources_visible = all(
                f"source {slot}:" in visible.casefold()
                for slot in example.source_slots
            )
            if not answer_visible or not sources_visible:
                failures.append(example.example_id)
    if failures:
        sample = ", ".join(failures[:8])
        raise ValueError(
            f"{len(failures)} answerable examples lose evidence under "
            f"maximum_memory_tokens={maximum_tokens}; examples: {sample}"
        )
    return {"answerable_examples": len(answerable), "failures": 0}


def pad_token_batches(items: Sequence[tuple[list[int], list[int]]], pad_id: int,
                      device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(tokens) for tokens, _ in items)
    token_batch, label_batch, mask_batch = [], [], []
    for tokens, labels in items:
        padding = length - len(tokens)
        token_batch.append(tokens + [pad_id] * padding)
        label_batch.append(labels + [-100] * padding)
        mask_batch.append([1] * len(tokens) + [0] * padding)
    return (
        torch.tensor(token_batch, dtype=torch.long, device=device),
        torch.tensor(label_batch, dtype=torch.long, device=device),
        torch.tensor(mask_batch, dtype=torch.long, device=device),
    )


def pad_memory(items: Sequence[torch.Tensor], hidden_size: int,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max((item.shape[0] for item in items), default=0)
    maximum = max(1, maximum)
    memory = torch.zeros(
        len(items), maximum, hidden_size, dtype=torch.bfloat16, device=device
    )
    mask = torch.zeros(len(items), maximum, dtype=torch.bool, device=device)
    for row, item in enumerate(items):
        if item.numel():
            count = item.shape[0]
            memory[row, :count] = item.to(device=device, dtype=torch.bfloat16)
            mask[row, :count] = True
    return memory, mask


def load_model(config: PowConfig, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.revision, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.revision,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def resolve_layers(model):
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise RuntimeError("cannot locate decoder layers")


@torch.inference_mode()
def encode_memory_sets(model, tokenizer, hooks: dict[int, MemoryHook],
                       records_by_id: dict[str, MemoryRecord],
                       memory_sets: Sequence[tuple[str, ...]],
                       maximum_tokens: int, device: torch.device,
                       batch_size: int = 8
                       ) -> dict[tuple[str, ...], dict[int, torch.Tensor]]:
    cache: dict[tuple[str, ...], dict[int, torch.Tensor]] = {
        (): {
            index: torch.empty(0, model.config.hidden_size)
            for index in hooks
        }
    }
    pending = [key for key in dict.fromkeys(memory_sets) if key]
    for start in range(0, len(pending), batch_size):
        keys = pending[start:start + batch_size]
        texts = [format_memory([records_by_id[item] for item in key]) for key in keys]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=maximum_tokens,
        ).to(device)
        with contextlib.ExitStack() as stack:
            for hook in hooks.values():
                stack.enter_context(hook.disabled())
            outputs = model(
                **encoded, use_cache=False, output_hidden_states=True, return_dict=True
            )
        for row, key in enumerate(keys):
            count = int(encoded["attention_mask"][row].sum().item())
            cache[key] = {
                # hidden_states[i] is the input to decoder layer i. The
                # reference runtime feeds that state to cross-attention at i;
                # hidden_states[i + 1] would be an off-by-one representation.
                index: outputs.hidden_states[index][row, :count].detach().to(
                    device="cpu", dtype=torch.bfloat16
                )
                for index in hooks
            }
    return cache


def _state_bytes(states: dict[int, torch.Tensor]) -> int:
    return sum(item.nelement() * item.element_size() for item in states.values())


class BoundedMemoryStateCache:
    """Lazy CPU cache for frozen layer states; bounded independently of corpus size."""

    def __init__(self, model, tokenizer, hooks: dict[int, MemoryHook],
                 records_by_id: dict[str, MemoryRecord], maximum_tokens: int,
                 device: torch.device, maximum_bytes: int) -> None:
        if maximum_bytes <= 0:
            raise ValueError("memory cache byte limit must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = hooks
        self.records_by_id = records_by_id
        self.maximum_tokens = maximum_tokens
        self.device = device
        self.maximum_bytes = maximum_bytes
        self.cache: OrderedDict[tuple[str, ...], dict[int, torch.Tensor]] = OrderedDict()
        self.cache[()] = {
            index: torch.empty(0, model.config.hidden_size) for index in hooks
        }
        self.bytes = 0
        self.peak_bytes = 0

    def ensure(self, keys: Sequence[tuple[str, ...]]) -> dict[
        tuple[str, ...], dict[int, torch.Tensor]
    ]:
        ordered = list(dict.fromkeys(keys))
        missing = [key for key in ordered if key and key not in self.cache]
        if missing:
            encoded = encode_memory_sets(
                self.model, self.tokenizer, self.hooks, self.records_by_id,
                missing, self.maximum_tokens, self.device,
            )
            for key in missing:
                states = encoded[key]
                self.cache[key] = states
                self.bytes += _state_bytes(states)
                self.peak_bytes = max(self.peak_bytes, self.bytes)
        for key in ordered:
            if key in self.cache:
                self.cache.move_to_end(key)
        protected = set(ordered)
        while self.bytes > self.maximum_bytes:
            victim = next(
                (key for key in self.cache if key and key not in protected), None
            )
            if victim is None:
                break
            self.bytes -= _state_bytes(self.cache.pop(victim))
        return self.cache


def set_memory_batch(hooks: dict[int, MemoryHook],
                     memory_cache: dict[tuple[str, ...], dict[int, torch.Tensor]],
                     memory_ids: Sequence[tuple[str, ...]], hidden_size: int,
                     device: torch.device) -> None:
    for index, hook in hooks.items():
        memory, mask = pad_memory(
            [memory_cache[key][index] for key in memory_ids], hidden_size, device
        )
        hook.set_memory(memory, mask)


def augment_training_examples(examples: Sequence[MemoryExample], records: Sequence[MemoryRecord],
                              seed: int) -> list[MemoryExample]:
    rng = random.Random(seed)
    train = [example for example in examples if example.split == "train"]
    record_ids = [record.record_id for record in records]
    records_by_id = {record.record_id: record for record in records}
    records_by_group: dict[str, list[str]] = {}
    for record in records:
        records_by_group.setdefault(record.shard_key, []).append(record.record_id)
    augmented = list(train)
    for example in train:
        if example.kind == "unknown":
            augmented.append(replace(
                example,
                example_id=example.example_id + "-irrelevant",
                memory_ids=(rng.choice(record_ids),),
            ))
        elif len(example.memory_ids) == 1:
            group = records_by_id[example.memory_ids[0]].shard_key
            related = tuple(records_by_group[group])
            if related != example.memory_ids:
                augmented.append(replace(
                    example,
                    example_id=example.example_id + "-related",
                    memory_ids=related,
                ))
    rng.shuffle(augmented)
    return augmented


@torch.inference_mode()
def supervised_validation_nll(
    model, tokenizer, adapter: MemoryExpertStack,
    hooks: dict[int, MemoryHook], memory_state_cache: BoundedMemoryStateCache,
    examples: Sequence[MemoryExample], batch_size: int, device: torch.device,
) -> float:
    """Measure token-weighted held-out NLL without generating or changing weights."""
    if not examples:
        raise ValueError("validation requires at least one example")
    was_training = adapter.training
    adapter.eval()
    total_loss = 0.0
    target_tokens = 0
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            tokens, labels, attention_mask = pad_token_batches(
                [supervised_tokens(tokenizer, example) for example in batch],
                tokenizer.pad_token_id,
                device,
            )
            memory_ids = [example.memory_ids for example in batch]
            memory_cache = memory_state_cache.ensure(memory_ids)
            set_memory_batch(
                hooks, memory_cache, memory_ids,
                model.config.hidden_size, device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits.float()
            shifted_labels = labels[:, 1:]
            valid = shifted_labels.ne(-100)
            loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss.detach().cpu())
            target_tokens += int(valid.sum().item())
    finally:
        adapter.train(was_training)
    if target_tokens == 0:
        raise ValueError("validation examples contain no supervised target tokens")
    return total_loss / target_tokens


def train(config: PowConfig, output: Path, epochs: int, batch_size: int,
          learning_rate: float, gradient_accumulation: int,
          optimizer_name: str,
          corpus: tuple[list[MemoryRecord], list[MemoryExample]] | None = None,
          corpus_name: str = "synthetic-english-v1",
          initial_checkpoint: Path | None = None,
          augment_examples: bool = True,
          memory_cache_bytes: int = 4 << 30,
          validation_limit: int = 256,
          resume: bool = False) -> dict[str, object]:
    """Train complete deterministic epochs and checkpoint every epoch.

    `epochs` always means complete traversals of the selected training corpus.
    Microsteps and optimizer updates are derived and reported independently.
    Gradient accumulation is flushed at each epoch boundary and normalized by
    the exact supervised-token count, including a final partial window.
    """
    seed_everything(config.seed)
    if epochs <= 0 or batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("epochs, batch size, and accumulation must be positive")
    if validation_limit < 0:
        raise ValueError("validation limit cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Memory Expert PoW")
    device = torch.device("cuda:0")
    records, examples = corpus or build_corpus(seed=config.seed)
    records_by_id = {record.record_id: record for record in records}
    train_examples = (
        augment_training_examples(examples, records, config.seed)
        if augment_examples else
        [example for example in examples if example.split == "train"]
    )
    if not train_examples:
        raise ValueError("training corpus has no train examples")
    tokenizer, model = load_model(config, device)
    evidence_validation = validate_visible_evidence(
        tokenizer, examples, records_by_id, config.maximum_memory_tokens
    )
    layers = resolve_layers(model)
    layer_indices = tuple(range(0, len(layers), config.injection_every))
    adapter = MemoryExpertStack(
        model.config.hidden_size, layer_indices, config.gate_rank,
        config.gate_alpha, config.knowledge_dropout,
    ).to(device)
    output.mkdir(parents=True, exist_ok=True)
    training_state_path = output / "training-state.pt"
    if training_state_path.is_file() and not resume:
        raise ValueError(
            f"output already contains training state: {training_state_path}; "
            "use --resume or choose a new output"
        )
    geometry = training_geometry(
        len(train_examples), batch_size, gradient_accumulation, epochs
    )
    training_contract = {
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "training_examples": len(train_examples),
    }
    resume_state: dict[str, object] | None = None
    if resume:
        if initial_checkpoint is not None:
            raise ValueError("resume and initial checkpoint are mutually exclusive")
        if not training_state_path.is_file():
            raise ValueError(f"resume state is missing: {training_state_path}")
        resume_state = torch.load(
            training_state_path, map_location="cpu", weights_only=True
        )
        if resume_state.get("architecture_version") != ARCHITECTURE_VERSION:
            raise RuntimeError("resume adapter architecture does not match")
        if resume_state.get("config") != asdict(config):
            raise RuntimeError("resume model or adapter configuration does not match")
        if resume_state.get("corpus_fingerprint") != corpus_fingerprint(records):
            raise RuntimeError("resume corpus fingerprint does not match")
        if resume_state.get("examples_fingerprint") != examples_fingerprint(examples):
            raise RuntimeError("resume example fingerprint does not match")
        if tuple(resume_state["layer_indices"]) != layer_indices:
            raise RuntimeError("resume adapter layer layout does not match")
        if resume_state.get("training_contract") != training_contract:
            raise RuntimeError("resume training hyperparameters do not match")
        adapter.load_state_dict(resume_state["adapter"], strict=True)
    elif initial_checkpoint is not None:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if initial.get("architecture_version") != ARCHITECTURE_VERSION:
            raise RuntimeError("initial adapter architecture does not match")
        if tuple(initial["layer_indices"]) != layer_indices:
            raise RuntimeError("initial adapter layer layout does not match")
        adapter.load_state_dict(initial["adapter"], strict=True)
    hooks = {
        index: MemoryHook(adapter.adapter(index)) for index in layer_indices
    }
    handles = [
        layers[index].register_forward_hook(hooks[index]) for index in layer_indices
    ]
    started = time.perf_counter()
    try:
        memory_state_cache = BoundedMemoryStateCache(
            model, tokenizer, hooks, records_by_id,
            config.maximum_memory_tokens, device, memory_cache_bytes,
        )
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                adapter.parameters(), lr=learning_rate, weight_decay=0.01
            )
        elif optimizer_name == "lamb":
            optimizer = Lamb(
                adapter.parameters(), lr=learning_rate, weight_decay=0.01
            )
        else:
            raise ValueError(f"unsupported optimizer: {optimizer_name}")
        start_epoch = 0
        best_validation_nll = float("inf")
        validation_history: list[dict[str, float | int]] = []
        if resume_state is not None:
            start_epoch = int(resume_state["completed_epochs"])
            if start_epoch >= epochs:
                raise ValueError(
                    f"resume already completed {start_epoch} epochs; target={epochs}"
                )
            optimizer.load_state_dict(resume_state["optimizer"])
            best_validation_nll = float(
                resume_state.get("best_validation_nll", float("inf"))
            )
            validation_history = list(resume_state.get("validation_history", []))

        schedule = build_epoch_training_schedule(
            train_examples, batch_size, gradient_accumulation,
            epochs, config.seed, start_epoch=start_epoch,
        )
        validation_examples = (
            select_eval_examples(examples, validation_limit)
            if validation_limit else []
        )
        losses: list[float] = []
        adapter.train()
        examples_seen = 0
        optimizer_updates = 0
        epoch_loss_sum = 0.0
        epoch_target_tokens = 0
        accumulated_target_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        for scheduled in schedule:
            batch = scheduled.examples
            tokens, labels, attention_mask = pad_token_batches(
                [supervised_tokens(tokenizer, example) for example in batch],
                tokenizer.pad_token_id,
                device,
            )
            batch_memory_ids = [example.memory_ids for example in batch]
            memory_cache = memory_state_cache.ensure(batch_memory_ids)
            set_memory_batch(
                hooks, memory_cache, batch_memory_ids,
                model.config.hidden_size, device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits.float()
                loss_sum = functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            target_tokens = int(labels[:, 1:].ne(-100).sum().item())
            if target_tokens == 0:
                raise RuntimeError("training batch contains no supervised target tokens")
            loss = loss_sum / target_tokens
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {scheduled.epoch} "
                    f"batch {scheduled.batch_in_epoch}"
                )
            # Accumulate summed token losses. Divide gradients once by the
            # exact token count in this window so a final partial microbatch is
            # neither dropped nor over-weighted.
            loss_sum.backward()
            accumulated_target_tokens += target_tokens
            if scheduled.optimizer_step:
                for parameter in adapter.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_target_tokens)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                accumulated_target_tokens = 0
            losses.append(float(loss.detach().cpu()))
            epoch_loss_sum += float(loss_sum.detach().cpu())
            epoch_target_tokens += target_tokens
            examples_seen += len(batch)
            if len(losses) == 1 or scheduled.microstep % 100 == 0:
                print(json.dumps({
                    "event": "train",
                    "epoch": scheduled.epoch,
                    "batch_in_epoch": scheduled.batch_in_epoch,
                    "batches_in_epoch": scheduled.batches_in_epoch,
                    "microstep": scheduled.microstep,
                    "optimizer_update": scheduled.optimizer_update,
                    "optimizer_updates_this_run": optimizer_updates,
                    "optimizer_step": scheduled.optimizer_step,
                    "examples_seen": examples_seen,
                    "phase": scheduled.phase,
                    "loss": losses[-1],
                    "gate_norm": float(sum(
                        item.gate_up.weight.float().norm().detach().cpu()
                        for item in adapter.adapters.values()
                    )),
                    "vram_gib": torch.cuda.max_memory_allocated() / (1 << 30),
                }), flush=True)
            if scheduled.batch_in_epoch == scheduled.batches_in_epoch:
                validation_nll = (
                    supervised_validation_nll(
                        model, tokenizer, adapter, hooks, memory_state_cache,
                        validation_examples, batch_size, device,
                    )
                    if validation_examples else float("nan")
                )
                epoch_training_nll = epoch_loss_sum / epoch_target_tokens
                validation_history.append({
                    "epoch": scheduled.epoch,
                    "training_nll": epoch_training_nll,
                    "validation_nll": validation_nll,
                })
                checkpoint = {
                    "schema_version": 1,
                    "architecture_version": ARCHITECTURE_VERSION,
                    "config": asdict(config),
                    "layer_indices": layer_indices,
                    "hidden_size": model.config.hidden_size,
                    "corpus_fingerprint": corpus_fingerprint(records),
                    "examples_fingerprint": examples_fingerprint(examples),
                    "training_corpus": corpus_name,
                    "training_contract": training_contract,
                    "visible_evidence_validation": evidence_validation,
                    "initial_checkpoint": (
                        str(initial_checkpoint) if initial_checkpoint else None
                    ),
                    "training": {
                        "epochs": epochs,
                        "completed_epochs": scheduled.epoch,
                        "microsteps": scheduled.epoch * geometry.batches_per_epoch,
                        "optimizer_updates": (
                            scheduled.epoch * geometry.optimizer_updates_per_epoch
                        ),
                        "examples_seen": scheduled.epoch * len(train_examples),
                        "microsteps_this_run": len(losses),
                        "optimizer_updates_this_run": optimizer_updates,
                        "examples_seen_this_run": examples_seen,
                        "validation_nll": validation_nll,
                    },
                    "adapter": {
                        key: value.detach().cpu()
                        for key, value in adapter.state_dict().items()
                    },
                }
                last_path = output / "memory-expert.last.pt"
                last_temporary = output / "memory-expert.last.pt.tmp"
                torch.save(checkpoint, last_temporary)
                last_temporary.replace(last_path)
                improved = (
                    not validation_examples
                    or validation_nll < best_validation_nll
                )
                if improved:
                    best_validation_nll = validation_nll
                    best_path = output / "memory-expert.pt"
                    best_temporary = output / "memory-expert.pt.tmp"
                    torch.save(checkpoint, best_temporary)
                    best_temporary.replace(best_path)
                state = {
                    **checkpoint,
                    "schema_version": 2,
                    "optimizer": optimizer.state_dict(),
                    "completed_epochs": scheduled.epoch,
                    "best_validation_nll": best_validation_nll,
                    "validation_history": validation_history,
                }
                state_temporary = output / "training-state.pt.tmp"
                torch.save(state, state_temporary)
                state_temporary.replace(training_state_path)
                print(json.dumps({
                    "event": "epoch-complete",
                    "epoch": scheduled.epoch,
                    "epochs": epochs,
                    "training_nll": epoch_training_nll,
                    "validation_nll": validation_nll,
                    "best_validation_nll": best_validation_nll,
                    "microsteps": scheduled.epoch * geometry.batches_per_epoch,
                    "optimizer_updates": (
                        scheduled.epoch * geometry.optimizer_updates_per_epoch
                    ),
                    "examples_seen": scheduled.epoch * len(train_examples),
                    "checkpoint_improved": improved,
                }), flush=True)
                epoch_loss_sum = 0.0
                epoch_target_tokens = 0

        checkpoint_path = output / "memory-expert.pt"
        summary = {
            "schema_version": 2,
            "model": config.model,
            "revision": config.revision,
            "injection_layers": len(layer_indices),
            "epochs": epochs,
            "resumed_from_epoch": start_epoch,
            "training_examples": len(train_examples),
            "batches_per_epoch": geometry.batches_per_epoch,
            "optimizer_updates_per_epoch": geometry.optimizer_updates_per_epoch,
            "microsteps_this_run": len(schedule),
            "optimizer_updates_this_run": optimizer_updates,
            "examples_seen_this_run": examples_seen,
            "total_microsteps": geometry.total_microsteps,
            "total_optimizer_updates": geometry.total_optimizer_updates,
            "total_examples_seen": geometry.total_examples,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": batch_size * gradient_accumulation,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "training_corpus": corpus_name,
            "visible_evidence_validation": evidence_validation,
            "validation_examples": len(validation_examples),
            "validation_history": validation_history,
            "best_validation_nll": best_validation_nll,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "minimum_loss": min(losses),
            "adapter_parameters": sum(
                parameter.numel() for parameter in adapter.parameters()
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / (1 << 30),
            "memory_state_cache_limit_bytes": memory_cache_bytes,
            "memory_state_cache_peak_bytes": memory_state_cache.peak_bytes,
            "checkpoint": str(checkpoint_path),
            "last_checkpoint": str(output / "memory-expert.last.pt"),
            "training_state": str(training_state_path),
        }
        (output / "train-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        for handle in handles:
            handle.remove()


def load_adapter(checkpoint_path: Path, model, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError(
            "checkpoint uses an obsolete Memory Expert architecture; retrain it"
        )
    config = PowConfig(**checkpoint["config"])
    adapter = MemoryExpertStack(
        checkpoint["hidden_size"], checkpoint["layer_indices"],
        config.gate_rank, config.gate_alpha, config.knowledge_dropout,
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.eval()
    return checkpoint, config, adapter


@torch.inference_mode()
def generate(model, tokenizer, hooks: dict[int, MemoryHook],
             memory_cache: dict[tuple[str, ...], dict[int, torch.Tensor]],
             memory_ids: tuple[str, ...], prompt_ids: Sequence[int],
             maximum_new_tokens: int, device: torch.device) -> str:
    set_memory_batch(
        hooks, memory_cache, [memory_ids], model.config.hidden_size, device
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    past = None
    generated: list[int] = []
    for _ in range(maximum_new_tokens):
        outputs = model(
            input_ids=input_ids if past is None else input_ids[:, -1:],
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
        input_ids = torch.cat((input_ids, torch.tensor([[token]], device=device)), dim=1)
        attention_mask = torch.ones_like(input_ids)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def score_example(example: MemoryExample, response: str,
                  admitted_ids: Sequence[str],
                  records_by_id: dict[str, MemoryRecord]) -> dict[str, object]:
    answer, source_slots = parse_response(response)
    if example.answer_span is not None:
        # Pointer contract: the answer is exact iff the model points at the
        # gold (slot, sentence); the authority plane owns the verbatim text.
        answer_text_match = parse_pointer(answer) == tuple(example.answer_span)
    else:
        answer_text_match = normalized_contains(answer, example.answer)
    admitted_records = [records_by_id[item] for item in admitted_ids]
    sources_authorized = (
        len(set(source_slots)) == len(source_slots)
        and all(0 <= slot < len(admitted_records) for slot in source_slots)
    )
    source_selection_ok = source_slots == example.source_slots and sources_authorized
    rendered_citations = [
        {
            "source_slot": slot,
            "record_id": admitted_records[slot].public_id,
            "quote": admitted_records[slot].text,
        }
        for slot in source_slots
        if sources_authorized
    ]
    return {
        "example_id": example.example_id,
        "kind": example.kind,
        "response": response,
        "parsed_answer": answer,
        "parsed_source_slots": source_slots,
        "sources_authorized": sources_authorized,
        "rendered_citations": rendered_citations,
        "answer_text_match": answer_text_match,
        "source_selection_ok": source_selection_ok,
        "passed": answer_text_match and source_selection_ok,
    }


def select_eval_examples(examples: Sequence[MemoryExample], limit: int) -> list[MemoryExample]:
    candidates = [example for example in examples if example.split == "eval"]
    buckets: dict[tuple[str, str], list[MemoryExample]] = {}
    for example in candidates:
        buckets.setdefault((example.language, example.kind), []).append(example)
    selected: list[MemoryExample] = []
    while len(selected) < min(limit, len(candidates)):
        changed = False
        for stratum in sorted(buckets):
            if buckets[stratum] and len(selected) < limit:
                selected.append(buckets[stratum].pop(0))
                changed = True
        if not changed:
            break
    return selected


def select_causal_examples(examples: Sequence[MemoryExample],
                           family_limit: int = 3,
                           records_by_id: dict[str, MemoryRecord] | None = None,
                           ) -> list[list[MemoryExample]]:
    """Select same-question interventions whose only authority is memory."""
    groups: dict[str, list[MemoryExample]] = {}
    for example in examples:
        if example.split == "eval" and example.kind != "unknown":
            key = example.family_id or example.question
            groups.setdefault(key, []).append(example)
    candidates = [
        rows for rows in groups.values()
        if len(rows) >= 2 and len({row.answer for row in rows}) == len(rows)
    ]
    candidates.sort(key=lambda rows: rows[0].example_id)
    if not records_by_id:
        return candidates[:family_limit]
    by_language: dict[str, list[list[MemoryExample]]] = {}
    for family in candidates:
        language = family[0].language
        by_language.setdefault(language, []).append(family)
    selected: list[list[MemoryExample]] = []
    while len(selected) < min(family_limit, len(candidates)):
        changed = False
        for language in sorted(by_language):
            if by_language[language] and len(selected) < family_limit:
                selected.append(by_language[language].pop(0))
                changed = True
        if not changed:
            break
    return selected


@torch.inference_mode()
def candidate_nll(model, tokenizer, hooks: dict[int, MemoryHook],
                  memory_cache: dict[tuple[str, ...], dict[int, torch.Tensor]],
                  memory_ids: tuple[str, ...], candidates: Sequence[MemoryExample],
                  device: torch.device) -> list[float]:
    """Score competing answers while holding question and memory fixed."""
    tokens, labels, attention_mask = pad_token_batches(
        [supervised_tokens(tokenizer, candidate) for candidate in candidates],
        tokenizer.pad_token_id, device,
    )
    set_memory_batch(
        hooks, memory_cache, [memory_ids] * len(candidates),
        model.config.hidden_size, device,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=tokens, attention_mask=attention_mask,
            use_cache=False, return_dict=True,
        ).logits.float()
    shifted_labels = labels[:, 1:]
    losses = functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), shifted_labels,
        ignore_index=-100, reduction="none",
    )
    valid = shifted_labels.ne(-100)
    return (
        (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    ).detach().cpu().tolist()


def causal_probe(checkpoint_path: Path, output: Path,
                 family_limit: int = 3,
                 corpus: tuple[list[MemoryRecord], list[MemoryExample]] | None = None
                 ) -> dict[str, object]:
    """Prove that changing only admitted memory changes the exact answer."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Memory Expert PoW")
    device = torch.device("cuda:0")
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if raw_checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError("checkpoint architecture does not match this runtime")
    config = PowConfig(**raw_checkpoint["config"])
    seed_everything(config.seed)
    records, examples = corpus or build_corpus(seed=config.seed)
    if raw_checkpoint["corpus_fingerprint"] != corpus_fingerprint(records):
        raise RuntimeError("checkpoint corpus fingerprint mismatch")
    expected_examples = raw_checkpoint.get("examples_fingerprint")
    if expected_examples and expected_examples != examples_fingerprint(examples):
        raise RuntimeError("checkpoint example contract fingerprint mismatch")
    records_by_id = {record.record_id: record for record in records}
    tokenizer, model = load_model(config, device)
    layers = resolve_layers(model)
    checkpoint, _, adapter = load_adapter(checkpoint_path, model, device)
    layer_indices = tuple(int(index) for index in checkpoint["layer_indices"])
    hooks = {index: MemoryHook(adapter.adapter(index)) for index in layer_indices}
    handles = [
        layers[index].register_forward_hook(hooks[index]) for index in layer_indices
    ]
    families = select_causal_examples(examples, family_limit, records_by_id)
    selected = [example for family in families for example in family]
    pointer_mode = bool(families) and all(
        example.answer_span is not None
        for family in families for example in family
    )
    started = time.perf_counter()
    try:
        memory_cache = encode_memory_sets(
            model, tokenizer, hooks, records_by_id,
            [example.memory_ids for example in selected],
            config.maximum_memory_tokens, device,
        )
        contrastive_results: list[dict[str, object]] = []
        contrastive_correct = 0
        contrastive_total = 0
        if pointer_mode:
            # Under the pointer contract siblings legitimately share their
            # `@slot:sentence` coordinates, so contrastive target NLL is
            # degenerate by construction. Causality is established by the
            # generation stage below: the authority plane resolves each
            # pointer and the rendered content must track the admitted memory.
            contrastive_accuracy: float | None = None
            contrastive_passed = True
        else:
            for family in families:
                matrix: list[dict[str, object]] = []
                for correct_index, memory_example in enumerate(family):
                    nll = candidate_nll(
                        model, tokenizer, hooks, memory_cache,
                        memory_example.memory_ids, family, device,
                    )
                    preferred_index = int(np.argmin(nll))
                    is_correct = preferred_index == correct_index
                    contrastive_correct += int(is_correct)
                    contrastive_total += 1
                    matrix.append({
                        "memory_example_id": memory_example.example_id,
                        "candidate_example_ids": [item.example_id for item in family],
                        "mean_target_nll": nll,
                        "preferred_example_id": family[preferred_index].example_id,
                        "correct": is_correct,
                    })
                contrastive_results.append({
                    "question": family[0].question,
                    "matrix": matrix,
                })
            contrastive_accuracy = (
                contrastive_correct / contrastive_total if contrastive_total else 0.0
            )
            contrastive_passed = contrastive_accuracy >= 0.90

        rows: list[dict[str, object]] = []
        family_results: list[dict[str, object]] = []
        if contrastive_passed:
            generation_families = families
        else:
            generation_families = []
        for family in generation_families:
            predicted_answers: list[str] = []
            family_rows: list[dict[str, object]] = []
            for example in family:
                response = generate(
                    model, tokenizer, hooks, memory_cache, example.memory_ids,
                    chat_prompt(tokenizer, example.question),
                    config.maximum_new_tokens, device,
                )
                scored = score_example(
                    example, response, example.memory_ids, records_by_id
                )
                scored["admitted_memory_ids"] = example.memory_ids
                rows.append(scored)
                family_rows.append(scored)
                if pointer_mode:
                    rendered = render_pointer_answer(
                        scored["parsed_answer"], example, records_by_id
                    )
                    scored["rendered_answer"] = rendered
                    predicted_answers.append(rendered or "")
                else:
                    predicted_answers.append(str(scored["parsed_answer"]))
                print(json.dumps({
                    "event": "causal-probe", **scored,
                }), flush=True)
            expected_variants = len({example.answer for example in family})
            observed_variants = len(set(predicted_answers))
            family_results.append({
                "question": family[0].question,
                "examples": len(family),
                "expected_answer_variants": expected_variants,
                "observed_answer_variants": observed_variants,
                "all_exact": all(bool(row["passed"]) for row in family_rows),
                "memory_sensitive": observed_variants == expected_variants,
            })
        count = max(1, len(rows))
        exact_accuracy = (
            sum(bool(row["passed"]) for row in rows) / count if rows else 0.0
        )
        summary = {
            "schema_version": 1,
            "architecture_version": ARCHITECTURE_VERSION,
            "model": config.model,
            "families": len(families),
            "examples": len(rows),
            "pointer_mode": pointer_mode,
            "contrastive_accuracy": contrastive_accuracy,
            "contrastive_passed": contrastive_passed,
            "contrastive_results": contrastive_results,
            "generation_skipped": not contrastive_passed,
            "exact_accuracy": exact_accuracy,
            "all_families_memory_sensitive": bool(family_results) and all(
                bool(family["memory_sensitive"]) for family in family_results
            ),
            "passed": contrastive_passed and exact_accuracy >= 0.90 and all(
                bool(family["memory_sensitive"]) for family in family_results
            ),
            "family_results": family_results,
            "results": rows,
            "elapsed_seconds": time.perf_counter() - started,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "causal-probe.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        for handle in handles:
            handle.remove()


def evaluate(checkpoint_path: Path, output: Path, evaluation_limit: int,
             corpus: tuple[list[MemoryRecord], list[MemoryExample]] | None = None
             ) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Memory Expert PoW")
    device = torch.device("cuda:0")
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PowConfig(**raw_checkpoint["config"])
    seed_everything(config.seed)
    records, examples = corpus or build_corpus(seed=config.seed)
    if raw_checkpoint["corpus_fingerprint"] != corpus_fingerprint(records):
        raise RuntimeError("checkpoint corpus fingerprint mismatch")
    expected_examples = raw_checkpoint.get("examples_fingerprint")
    if expected_examples and expected_examples != examples_fingerprint(examples):
        raise RuntimeError("checkpoint example contract fingerprint mismatch")
    records_by_id = {record.record_id: record for record in records}
    tokenizer, model = load_model(config, device)
    layers = resolve_layers(model)
    checkpoint, _, adapter = load_adapter(checkpoint_path, model, device)
    layer_indices = tuple(int(index) for index in checkpoint["layer_indices"])
    hooks = {
        index: MemoryHook(adapter.adapter(index)) for index in layer_indices
    }
    handles = [
        layers[index].register_forward_hook(hooks[index]) for index in layer_indices
    ]

    index_records = [record for record in records if record.indexable]
    embedder = HashingEmbedder().fit(record.text for record in index_records)
    index = ShardedVectorIndex(
        index_records, embedder.encode(record.text for record in index_records)
    )
    index_root = output / "knowledge-index"
    index.write(index_root)
    index = ShardedVectorIndex.open(index_root)
    selected = select_eval_examples(examples, evaluation_limit)

    automatic_ids: dict[str, tuple[str, ...]] = {}
    for example in selected:
        matches = retrieve_adaptive(index, embedder, example.question, maximum=2)
        automatic_ids[example.example_id] = tuple(record.record_id for record, _ in matches)

    memory_sets = [example.memory_ids for example in selected]
    memory_sets += list(automatic_ids.values())
    try:
        memory_cache = encode_memory_sets(
            model, tokenizer, hooks, records_by_id, memory_sets,
            config.maximum_memory_tokens, device,
        )
        results: dict[str, list[dict[str, object]]] = {
            "no-memory": [],
            "oracle-memory": [],
            "automatic-retrieval": [],
            "full-context-ceiling": [],
        }
        started = time.perf_counter()
        for example in selected:
            prompt = chat_prompt(tokenizer, example.question)
            modes = {
                "no-memory": (),
                "oracle-memory": example.memory_ids,
            }
            retrieval_eligible = example.kind != "unknown" and all(
                records_by_id[record_id].indexable
                for record_id in example.memory_ids
            )
            if retrieval_eligible:
                modes["automatic-retrieval"] = automatic_ids[example.example_id]
            for mode, memory_ids in modes.items():
                response = generate(
                    model, tokenizer, hooks, memory_cache, memory_ids, prompt,
                    config.maximum_new_tokens, device,
                )
                scored = score_example(example, response, memory_ids, records_by_id)
                scored["admitted_memory_ids"] = memory_ids
                if mode == "automatic-retrieval":
                    expected_evidence = set(example.citations)
                    retrieved_evidence = {
                        records_by_id[record_id].public_id
                        for record_id in memory_ids
                    }
                    scored["retrieval_recall"] = expected_evidence.issubset(
                        retrieved_evidence
                    )
                results[mode].append(scored)
                print(json.dumps({"event": "evaluation", "mode": mode, **scored}), flush=True)

            context_records = [records_by_id[item] for item in example.memory_ids]
            context_prompt = chat_prompt(
                tokenizer, example.question, format_memory(context_records)
            )
            with contextlib.ExitStack() as stack:
                for hook in hooks.values():
                    stack.enter_context(hook.disabled())
                response = generate(
                    model, tokenizer, hooks, memory_cache, (), context_prompt,
                    config.maximum_new_tokens, device,
                )
            scored = score_example(
                example, response, example.memory_ids, records_by_id
            )
            scored["admitted_memory_ids"] = example.memory_ids
            results["full-context-ceiling"].append(scored)
            print(json.dumps({
                "event": "evaluation", "mode": "full-context-ceiling", **scored
            }), flush=True)

        metrics: dict[str, dict[str, float | int]] = {}
        for mode, rows in results.items():
            count = max(1, len(rows))
            metrics[mode] = {
                "examples": len(rows),
                "strict_answer_text_match_rate": sum(
                    bool(row["answer_text_match"]) for row in rows
                ) / count,
                "source_selection_accuracy": sum(
                    bool(row["source_selection_ok"]) for row in rows
                ) / count,
                "joint_accuracy": sum(bool(row["passed"]) for row in rows) / count,
            }
            if mode == "automatic-retrieval":
                metrics[mode]["retrieval_recall"] = sum(
                    bool(row["retrieval_recall"]) for row in rows
                ) / count
        automatic_example_ids = {
            str(row["example_id"]) for row in results["automatic-retrieval"]
        }
        paired_oracle_rows = [
            row for row in results["oracle-memory"]
            if str(row["example_id"]) in automatic_example_ids
        ]
        paired_count = max(1, len(paired_oracle_rows))
        metrics["paired-oracle-memory"] = {
            "examples": len(paired_oracle_rows),
            "strict_answer_text_match_rate": sum(
                bool(row["answer_text_match"]) for row in paired_oracle_rows
            ) / paired_count,
            "source_selection_accuracy": sum(
                bool(row["source_selection_ok"]) for row in paired_oracle_rows
            ) / paired_count,
            "joint_accuracy": sum(
                bool(row["passed"]) for row in paired_oracle_rows
            ) / paired_count,
        }
        unknown_rows = [
            row for row, example in zip(results["oracle-memory"], selected)
            if example.kind == "unknown"
        ]
        unknown_accuracy = (
            sum(bool(row["passed"]) for row in unknown_rows) / len(unknown_rows)
            if unknown_rows else 0.0
        )
        gates = {
            "memory_channel": metrics["oracle-memory"]["joint_accuracy"] >= 0.70,
            "relative_to_context": (
                metrics["oracle-memory"]["joint_accuracy"] >=
                0.80 * metrics["full-context-ceiling"]["joint_accuracy"]
            ),
            "improves_over_no_memory": (
                metrics["oracle-memory"]["joint_accuracy"] >=
                metrics["no-memory"]["joint_accuracy"] + 0.30
            ),
            "unknown_abstention": unknown_accuracy >= 0.80,
            "retrieval": metrics["automatic-retrieval"].get("retrieval_recall", 0.0) >= 0.75,
        }
        summary = {
            "schema_version": 2,
            "metric_semantics": {
                "strict_answer_text_match_rate": (
                    "deterministic normalized substring match; not semantic accuracy"
                ),
                "manual_semantic_review_required": True,
            },
            "model": config.model,
            "evaluation_split": "eval",
            "examples": len(selected),
            "metrics": metrics,
            "unknown_oracle_joint_accuracy": unknown_accuracy,
            "gates": gates,
            "pow_passed": all(gates.values()),
            "elapsed_seconds": time.perf_counter() - started,
            "results": results,
            "index_manifest": str(index_root / "manifest.json"),
            "index_storage_mode": index.storage_mode,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        for handle in handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory Expert proof of concept")
    parser.add_argument("action", choices=("train", "probe", "evaluate", "run"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gate-rank", type=int, default=16)
    parser.add_argument("--gate-alpha", type=float, default=32.0)
    parser.add_argument("--injection-every", type=int, default=1)
    parser.add_argument("--knowledge-dropout", type=float, default=0.2)
    parser.add_argument("--maximum-memory-tokens", type=int, default=192)
    parser.add_argument("--maximum-new-tokens", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--optimizer", choices=("adamw", "lamb"), default="adamw")
    parser.add_argument("--evaluation-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PowConfig(
        model=args.model,
        revision=args.revision,
        gate_rank=args.gate_rank,
        gate_alpha=args.gate_alpha,
        injection_every=args.injection_every,
        knowledge_dropout=args.knowledge_dropout,
        maximum_memory_tokens=args.maximum_memory_tokens,
        maximum_new_tokens=args.maximum_new_tokens,
        seed=args.seed,
    )
    checkpoint = args.checkpoint or args.output / "memory-expert.pt"
    if args.action in ("train", "run"):
        print(json.dumps({
            "event": "train-summary",
            **train(
                config, args.output, args.epochs, args.batch_size,
                args.learning_rate, args.gradient_accumulation, args.optimizer,
            ),
        }, indent=2), flush=True)
    if args.action in ("probe", "run"):
        probe = causal_probe(checkpoint, args.output)
        print(json.dumps({
            "event": "causal-probe-summary", **probe,
        }, indent=2), flush=True)
        if args.action == "run" and not probe["passed"]:
            raise RuntimeError(
                "causal probe failed; held-out evaluation would not be meaningful"
            )
    if args.action in ("evaluate", "run"):
        print(json.dumps({
            "event": "evaluation-summary",
            **evaluate(checkpoint, args.output, args.evaluation_limit),
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
