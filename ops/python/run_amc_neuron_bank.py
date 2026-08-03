#!/usr/bin/env python3
"""Run an activation-pruned AMC expert bank inside the full OLMoE model."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hesr_amc import NeuronBankConfig, PrunedExpertBank, original_topk


class HybridNeuronMoe(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        bank: PrunedExpertBank,
        router_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.original = original
        self.bank = bank
        self.register_buffer("router_weight", router_weight.float())
        self.tokens = 0
        self.fast_tokens = 0
        self.fallback_tokens = 0
        self.risk_sum = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        top_weights, top_indices = original_topk(
            flat.float(),
            self.router_weight,
            self.bank.config.top_k,
            self.bank.config.normalize_top_k,
        )
        plan = self.bank.plan(top_indices, top_weights)
        fallback = plan["fallback"]
        output = self.bank.forward_fast(
            flat.float(), top_indices, top_weights, fallback=fallback
        )
        if fallback.any():
            output[fallback] = self.original.experts(
                flat[fallback], top_indices[fallback], top_weights[fallback]
            ).float()
        self.tokens += flat.shape[0]
        self.fast_tokens += int((~fallback).sum().item())
        self.fallback_tokens += int(fallback.sum().item())
        self.risk_sum += float(plan["route_risk"].sum().item())
        return output.to(hidden_states.dtype).reshape(shape)

    def counters(self) -> dict[str, float | int]:
        return {
            "tokens": self.tokens,
            "fast_path_tokens": self.fast_tokens,
            "fallback_tokens": self.fallback_tokens,
            "fallback_rate": self.fallback_tokens / max(1, self.tokens),
            "mean_predicted_risk": self.risk_sum / max(1, self.tokens),
        }


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    relative = torch.sqrt(
        (candidate - reference).square().mean(dim=-1)
        / reference.square().mean(dim=-1).clamp_min(1e-12)
    )
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    p = F.log_softmax(reference, dim=-1)
    q = F.log_softmax(candidate, dim=-1)
    kl = (p.exp() * (p - q)).sum(dim=-1)
    return {
        "relative_error_mean": float(relative.mean().item()),
        "relative_error_p95": float(relative.quantile(0.95).item()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_p05": float(cosine.quantile(0.05).item()),
        "kl_mean": float(kl.mean().item()),
        "top1_agreement": float(
            (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean().item()
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
    manifest = json.loads((args.compiled / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "hesr-amc-neuron-bank-v1":
        raise ValueError("compiled directory is not an AMC neuron bank")
    config = NeuronBankConfig.from_manifest(manifest)
    state = load_file(str(args.compiled / manifest["files"]["weights"]))
    bank = PrunedExpertBank(config)
    bank.load_state_dict(
        {
            "gate_up": state["gate_up"],
            "down": state["down"],
            "correction_in": state["correction_in"],
            "correction_out": state["correction_out"],
            "correction_hidden_mean": state["correction_hidden_mean"],
            "correction_residual_mean": state["correction_residual_mean"],
            "expert_risk": state["expert_risk"],
        }
    )
    bank.eval()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True)
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
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.snapshot),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    with torch.inference_mode():
        exact_started = time.perf_counter()
        exact = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        ).logits[valid].float().cpu()
        exact_seconds = time.perf_counter() - exact_started
        layer = int(manifest["source"]["layer"])
        hybrid = HybridNeuronMoe(
            model.model.layers[layer].mlp, bank, state["router_weight"]
        )
        model.model.layers[layer].mlp = hybrid
        hybrid_started = time.perf_counter()
        candidate = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        ).logits[valid].float().cpu()
        hybrid_seconds = time.perf_counter() - hybrid_started

    counters = hybrid.counters()
    active = manifest["active_mass"]
    amortized = (
        (1.0 - counters["fallback_rate"]) * active["fast_path_parameters_per_token"]
        + counters["fallback_rate"] * active["original_parameters_per_token"]
    )
    report = {
        "schema_version": 1,
        "status": "pass",
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "layer": int(manifest["source"]["layer"]),
        "tokens": int(valid.sum().item()),
        "policy": counters,
        "logits": compare(exact, candidate),
        "active_mass": {
            **active,
            "amortized_parameters_per_token_including_fallback": amortized,
            "amortized_reduction": active["original_parameters_per_token"]
            / max(1.0, amortized),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "exact_full_forward_seconds": exact_seconds,
            "hybrid_full_forward_seconds": hybrid_seconds,
            "full_forward_speedup": exact_seconds / max(hybrid_seconds, 1e-12),
        },
        "fallback_backend": "in-memory exact experts; maps to RAMCACHE in the C runtime",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
