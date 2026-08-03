#!/usr/bin/env python3
"""Apply a compiled HESR-AMC layer inside the full OLMoE model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hesr_amc import MicroConfig, MicroExpertCodebook, original_topk


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class HybridHesrMoe(nn.Module):
    """Fast micro-codebook with an exact OLMoE expert fallback."""

    def __init__(
        self,
        original: nn.Module,
        student: MicroExpertCodebook,
        router_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.original = original
        self.student = student
        self.register_buffer("router_weight", router_weight.float())
        self.total_tokens = 0
        self.fast_tokens = 0
        self.fallback_tokens = 0
        self.micro_calls = 0
        self.risk_sum = 0.0
        self.coverage_sum = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        top_weights, top_indices = original_topk(
            flat.float(),
            self.router_weight,
            self.student.config.original_top_k,
            self.student.config.normalize_original_top_k,
        )
        fast_output, plan = self.student.forward_sparse(
            flat.float(), top_indices, top_weights
        )
        fallback = plan["fallback"]
        if fallback.any():
            exact = self.original.experts(
                flat[fallback], top_indices[fallback], top_weights[fallback]
            )
            fast_output[fallback] = exact.float()

        fast = ~fallback
        self.total_tokens += flat.shape[0]
        self.fast_tokens += int(fast.sum().item())
        self.fallback_tokens += int(fallback.sum().item())
        self.micro_calls += int(plan["counts"][fast].sum().item())
        self.risk_sum += float(plan["route_risk"].sum().item())
        self.coverage_sum += float(plan["coverage"].sum().item())
        return fast_output.to(hidden_states.dtype).reshape(shape)

    def counters(self) -> dict[str, float | int | None]:
        return {
            "tokens": self.total_tokens,
            "fast_path_tokens": self.fast_tokens,
            "fallback_tokens": self.fallback_tokens,
            "fallback_rate": self.fallback_tokens / max(1, self.total_tokens),
            "micro_expert_calls": self.micro_calls,
            "mean_micro_experts_per_fast_token": (
                self.micro_calls / self.fast_tokens if self.fast_tokens else None
            ),
            "mean_predicted_risk": self.risk_sum / max(1, self.total_tokens),
            "mean_route_coverage": self.coverage_sum / max(1, self.total_tokens),
        }


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float()
    candidate_f = candidate.float()
    difference = candidate_f - reference_f
    relative = torch.sqrt(
        difference.square().mean(dim=-1)
        / reference_f.square().mean(dim=-1).clamp_min(1e-12)
    )
    cosine = F.cosine_similarity(reference_f, candidate_f, dim=-1)
    reference_log_prob = F.log_softmax(reference_f, dim=-1)
    candidate_log_prob = F.log_softmax(candidate_f, dim=-1)
    kl = (
        reference_log_prob.exp()
        * (reference_log_prob - candidate_log_prob)
    ).sum(dim=-1)
    return {
        "relative_error_mean": float(relative.mean().item()),
        "relative_error_p95": float(relative.quantile(0.95).item()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_p05": float(cosine.quantile(0.05).item()),
        "kl_mean": float(kl.mean().item()),
        "top1_agreement": float(
            (reference_f.argmax(dim=-1) == candidate_f.argmax(dim=-1))
            .float()
            .mean()
            .item()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.set_num_threads(max(1, args.threads))
    manifest = read_json(args.compiled / "manifest.json")
    config = MicroConfig.from_manifest(manifest)
    layer = int(manifest["source"]["layer"])
    state = load_file(str(args.compiled / manifest["files"]["weights"]))
    student = MicroExpertCodebook(config)
    student.load_state_dict(
        {
            "gate_up": state["gate_up"],
            "down": state["down"],
            "route_logits": state["route_logits"],
            "expert_risk": state["expert_risk"],
        }
    )
    student.eval()

    prompts = read_json(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.snapshot), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    valid = encoded["attention_mask"].bool()

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.snapshot),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    load_seconds = time.perf_counter() - started

    with torch.inference_mode():
        exact_started = time.perf_counter()
        exact_logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        ).logits[valid].float().cpu()
        exact_seconds = time.perf_counter() - exact_started

        original = model.model.layers[layer].mlp
        hybrid = HybridHesrMoe(original, student, state["router_weight"])
        model.model.layers[layer].mlp = hybrid
        hybrid_started = time.perf_counter()
        hybrid_logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        ).logits[valid].float().cpu()
        hybrid_seconds = time.perf_counter() - hybrid_started

    counters = hybrid.counters()
    active = manifest["active_mass"]
    average_micro_parameters = (
        (counters["mean_micro_experts_per_fast_token"] or 0.0)
        * active["parameters_per_micro_expert"]
    )
    amortized_parameters = (
        (1.0 - counters["fallback_rate"]) * average_micro_parameters
        + counters["fallback_rate"] * active["original_parameters_per_token"]
    )
    report = {
        "schema_version": 1,
        "status": "pass",
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "compiled": str(args.compiled.resolve()),
        "layer": layer,
        "prompts": len(prompts),
        "tokens": int(valid.sum().item()),
        "policy": counters,
        "logits": comparison(exact_logits, hybrid_logits),
        "active_mass": {
            **active,
            "average_fast_path_parameters_per_token": average_micro_parameters,
            "amortized_parameters_per_token_including_fallback": amortized_parameters,
            "amortized_reduction": active["original_parameters_per_token"]
            / max(1.0, amortized_parameters),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "exact_full_forward_seconds": exact_seconds,
            "hybrid_full_forward_seconds": hybrid_seconds,
            "full_forward_speedup": exact_seconds / max(hybrid_seconds, 1e-12),
        },
        "fallback_backend": (
            "in-memory HuggingFace experts for this vertical slice; production "
            "backend is the measured Colibri exact path"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
