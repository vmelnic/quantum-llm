#!/usr/bin/env python3
"""Compile activation-aware structured OLMoE experts at a fixed AMC budget."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from hesr_amc import (
    NeuronBankConfig,
    PrunedExpertBank,
    original_topk,
    token_relative_error,
)


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    error = token_relative_error(reference, candidate)
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    return {
        "relative_error_mean": float(error.mean().item()),
        "relative_error_p50": float(error.quantile(0.50).item()),
        "relative_error_p95": float(error.quantile(0.95).item()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_p05": float(cosine.quantile(0.05).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--retained-width", type=int, default=252)
    parser.add_argument("--correction-rank", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=0.10)
    parser.add_argument("--risk-threshold", type=float, default=0.35)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.set_num_threads(max(1, args.threads))
    started = time.perf_counter()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
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

    model = AutoModelForCausalLM.from_pretrained(
        str(args.snapshot),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    mlp = model.model.layers[args.layer].mlp
    captured_input: list[torch.Tensor] = []
    captured_output: list[torch.Tensor] = []
    before = mlp.register_forward_pre_hook(
        lambda _module, values: captured_input.append(values[0].detach().float().cpu())
    )
    after = mlp.register_forward_hook(
        lambda _module, _values, output: captured_output.append(
            output.detach().float().cpu()
        )
    )
    with torch.inference_mode():
        model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        )
    before.remove()
    after.remove()

    valid = encoded["attention_mask"].bool()
    hidden = captured_input[0][valid]
    teacher = captured_output[0][valid]
    prompt_ids = torch.arange(len(prompts))[:, None].expand_as(valid)[valid]
    router_weight = mlp.gate.weight.detach().float().cpu().clone()
    original_gate_up = mlp.experts.gate_up_proj.detach().cpu().clone()
    original_down = mlp.experts.down_proj.detach().cpu().clone()
    top_k = int(mlp.gate.top_k)
    normalize = bool(mlp.gate.norm_topk_prob)
    del model, mlp, captured_input, captured_output
    gc.collect()

    intermediate = original_down.shape[-1]
    if args.retained_width < 1 or args.retained_width > intermediate:
        raise ValueError("retained width is outside the original intermediate size")
    top_weights, top_indices = original_topk(
        hidden, router_weight, top_k, normalize
    )
    train = prompt_ids.remainder(5) != 0
    validation = ~train
    num_experts = original_gate_up.shape[0]
    importance = torch.zeros(num_experts, intermediate)
    observed_calls = torch.zeros(num_experts, dtype=torch.int64)

    with torch.inference_mode():
        for expert in range(num_experts):
            locations = ((top_indices == expert) & train[:, None]).nonzero(
                as_tuple=False
            )
            down_norm = original_down[expert].float().square().sum(dim=0).sqrt()
            if locations.numel() == 0:
                gate = original_gate_up[expert, :intermediate].float()
                up = original_gate_up[expert, intermediate:].float()
                importance[expert] = (
                    gate.square().sum(dim=1).sqrt()
                    * up.square().sum(dim=1).sqrt()
                    * down_norm
                )
                continue
            token_indices = locations[:, 0]
            slots = locations[:, 1]
            current = hidden[token_indices]
            gate_up = F.linear(current, original_gate_up[expert].float())
            gate, up = gate_up.chunk(2, dim=-1)
            activation = F.silu(gate) * up
            route_weight = top_weights[token_indices, slots]
            importance[expert] = (
                (activation.abs() * route_weight[:, None]).mean(dim=0) * down_norm
            )
            observed_calls[expert] = locations.shape[0]

    selected = importance.topk(args.retained_width, dim=1).indices
    config = NeuronBankConfig(
        hidden_size=hidden.shape[1],
        original_intermediate_size=intermediate,
        retained_intermediate_size=args.retained_width,
        num_experts=num_experts,
        top_k=top_k,
        normalize_top_k=normalize,
        correction_rank=args.correction_rank,
        risk_threshold=args.risk_threshold,
    )
    bank = PrunedExpertBank(config)
    with torch.no_grad():
        for expert in range(num_experts):
            neurons = selected[expert]
            bank.gate_up[expert, : args.retained_width].copy_(
                original_gate_up[expert, :intermediate][neurons].float()
            )
            bank.gate_up[expert, args.retained_width :].copy_(
                original_gate_up[expert, intermediate:][neurons].float()
            )
            bank.down[expert].copy_(
                original_down[expert].index_select(1, neurons).float()
            )
    del original_gate_up, original_down
    gc.collect()

    with torch.inference_mode():
        train_candidate = bank.forward_fast(
            hidden[train], top_indices[train], top_weights[train]
        )
        residual = teacher[train] - train_candidate
        hidden_mean = hidden[train].mean(dim=0)
        residual_mean = residual.mean(dim=0)
        centered_hidden = hidden[train] - hidden_mean
        centered_residual = residual - residual_mean
        rank = min(args.correction_rank, centered_residual.shape[0])
        if rank > 0:
            _, _, right = torch.linalg.svd(centered_residual, full_matrices=False)
            basis = right[:rank]
            target_coefficients = centered_residual @ basis.T
            kernel = centered_hidden @ centered_hidden.T
            ridge_scale = args.ridge * kernel.diag().mean().clamp_min(1e-12)
            kernel.diagonal().add_(ridge_scale)
            alpha = torch.linalg.solve(kernel, target_coefficients)
            bank.correction_in[:rank].copy_(alpha.T @ centered_hidden)
            bank.correction_out[:, :rank].copy_(basis.T)
        bank.correction_hidden_mean.copy_(hidden_mean)
        bank.correction_residual_mean.copy_(residual_mean)

        validation_candidate = bank.forward_fast(
            hidden[validation], top_indices[validation], top_weights[validation]
        )
        validation_error = token_relative_error(
            teacher[validation], validation_candidate
        )
        risk_sum = torch.zeros(num_experts)
        risk_weight = torch.zeros(num_experts)
        for slot in range(top_k):
            experts = top_indices[validation, slot]
            weights = top_weights[validation, slot]
            risk_sum.index_add_(0, experts, validation_error * weights)
            risk_weight.index_add_(0, experts, weights)
        global_risk = validation_error.mean()
        bank.expert_risk.copy_(
            torch.where(
                risk_weight > 0,
                risk_sum / risk_weight.clamp_min(1e-12),
                global_risk,
            )
        )
        plan = bank.plan(top_indices[validation], top_weights[validation])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "model.safetensors"
    manifest_path = args.output_dir / "manifest.json"
    save_file(
        {
            "gate_up": bank.gate_up.detach().bfloat16().contiguous(),
            "down": bank.down.detach().bfloat16().contiguous(),
            "correction_in": bank.correction_in.detach().bfloat16().contiguous(),
            "correction_out": bank.correction_out.detach().bfloat16().contiguous(),
            "correction_hidden_mean": bank.correction_hidden_mean.detach().float().contiguous(),
            "correction_residual_mean": bank.correction_residual_mean.detach().float().contiguous(),
            "expert_risk": bank.expert_risk.detach().float().contiguous(),
            "router_weight": router_weight.bfloat16().contiguous(),
            "selected_neurons": selected.to(torch.int32).contiguous(),
        },
        str(weights_path),
    )
    original_active = top_k * 3 * config.hidden_size * intermediate
    fast_active = top_k * bank.parameters_per_expert + bank.correction_parameters
    manifest = {
        "schema_version": 1,
        "format": "hesr-amc-neuron-bank-v1",
        "source": {"snapshot": str(args.snapshot.resolve()), "layer": args.layer},
        "architecture": {
            "hidden_size": config.hidden_size,
            "original_intermediate_size": intermediate,
            "retained_intermediate_size": args.retained_width,
            "num_experts": num_experts,
            "top_k": top_k,
            "normalize_top_k": normalize,
            "correction_rank": args.correction_rank,
        },
        "policy": {
            "risk_threshold": args.risk_threshold,
            "fallback": "original expert path",
        },
        "active_mass": {
            "original_parameters_per_token": original_active,
            "fast_path_parameters_per_token": fast_active,
            "fast_path_reduction": original_active / fast_active,
            "resident_bank_parameters": num_experts * bank.parameters_per_expert,
            "correction_parameters": bank.correction_parameters,
        },
        "files": {"weights": weights_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "compiled",
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "layer": args.layer,
        "calibration": {
            "prompts": len(prompts),
            "tokens": int(hidden.shape[0]),
            "train_tokens": int(train.sum().item()),
            "validation_tokens": int(validation.sum().item()),
            "observed_expert_calls": observed_calls.tolist(),
        },
        "validation": metrics(teacher[validation], validation_candidate),
        "policy_validation": {
            "fast_path_tokens": int((~plan["fallback"]).sum().item()),
            "fallback_tokens": int(plan["fallback"].sum().item()),
            "fallback_rate": float(plan["fallback"].float().mean().item()),
            "mean_predicted_risk": float(plan["route_risk"].mean().item()),
        },
        "active_mass": manifest["active_mass"],
        "output": {
            "directory": str(args.output_dir.resolve()),
            "weights_bytes": weights_path.stat().st_size,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
