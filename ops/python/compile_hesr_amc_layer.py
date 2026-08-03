#!/usr/bin/env python3
"""Compile one real OLMoE layer into the first HESR-AMC micro-expert format."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from hesr_amc import MicroConfig, MicroExpertCodebook, original_topk, token_relative_error


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_kmeans(features: torch.Tensor, clusters: int) -> torch.Tensor:
    normalized = F.normalize(features - features.mean(dim=1, keepdim=True), dim=1)
    chosen = [int(normalized.square().sum(dim=1).argmax().item())]
    centers = [normalized[chosen[0]]]
    while len(centers) < clusters:
        similarity = normalized @ torch.stack(centers).T
        distance = 1.0 - similarity.max(dim=1).values
        distance[torch.tensor(chosen)] = -1
        index = int(distance.argmax().item())
        chosen.append(index)
        centers.append(normalized[index])
    assignments = torch.full((features.shape[0],), -1, dtype=torch.int64)
    centers_tensor = torch.stack(centers)
    for _ in range(30):
        updated_assignments = (normalized @ centers_tensor.T).argmax(dim=1)
        if torch.equal(assignments, updated_assignments):
            break
        assignments = updated_assignments
        next_centers = []
        for cluster in range(clusters):
            members = normalized[assignments == cluster]
            next_centers.append(
                F.normalize(members.mean(dim=0), dim=0)
                if members.numel()
                else centers_tensor[cluster]
            )
        centers_tensor = torch.stack(next_centers)
    return assignments


def sampled_expert_features(
    gate_up: torch.Tensor, down: torch.Tensor, samples: int = 4096
) -> torch.Tensor:
    parts = []
    for weights in (gate_up, down):
        flattened = weights.reshape(weights.shape[0], -1)
        count = min(samples, flattened.shape[1])
        indices = torch.linspace(0, flattened.shape[1] - 1, count).round().long()
        parts.append(flattened[:, indices].float())
    return torch.cat(parts, dim=1)


def initialize_from_teacher(
    student: MicroExpertCodebook,
    teacher_gate_up: torch.Tensor,
    teacher_down: torch.Tensor,
    assignments: torch.Tensor,
    route_counts: torch.Tensor,
) -> list[int]:
    representatives: list[int] = []
    width = student.config.micro_width
    intermediate = teacher_down.shape[-1]
    with torch.no_grad():
        student.route_logits.fill_(-4.0)
        for micro in range(student.config.num_micro_experts):
            members = (assignments == micro).nonzero(as_tuple=False).flatten()
            if members.numel() == 0:
                representative = micro % teacher_gate_up.shape[0]
            else:
                representative = int(
                    members[route_counts[members].argmax()].item()
                )
            representatives.append(representative)
            gate = teacher_gate_up[representative, :intermediate]
            up = teacher_gate_up[representative, intermediate:]
            down = teacher_down[representative]
            importance = (
                gate.float().square().sum(dim=1)
                + up.float().square().sum(dim=1)
                + down.float().square().sum(dim=0)
            )
            neurons = importance.topk(width).indices
            student.gate_up[micro, :width].copy_(gate[neurons].float())
            student.gate_up[micro, width:].copy_(up[neurons].float())
            student.down[micro].copy_(down[:, neurons].float())
        student.route_logits[torch.arange(assignments.shape[0]), assignments] = 4.0
    return representatives


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
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--micro-experts", type=int, default=8)
    parser.add_argument("--micro-width", type=int, default=128)
    parser.add_argument("--max-active-micro", type=int, default=4)
    parser.add_argument("--coverage-threshold", type=float, default=0.75)
    parser.add_argument("--risk-threshold", type=float, default=0.60)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-gate-up", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    started = time.perf_counter()
    prompts = read_json(args.prompts)
    if len(prompts) < 5:
        raise ValueError("at least five calibration prompts are required")

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

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.snapshot),
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    mlp = model.model.layers[args.layer].mlp
    captured_inputs: list[torch.Tensor] = []
    captured_outputs: list[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        captured_inputs.append(values[0].detach().float().cpu())

    def out_hook(
        _module: torch.nn.Module,
        _values: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured_outputs.append(output.detach().float().cpu())

    before = mlp.register_forward_pre_hook(pre_hook)
    after = mlp.register_forward_hook(out_hook)
    with torch.inference_mode():
        model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
        )
    before.remove()
    after.remove()
    if len(captured_inputs) != 1 or len(captured_outputs) != 1:
        raise RuntimeError("the selected OLMoE layer was not captured exactly once")

    mask = encoded["attention_mask"].bool()
    hidden = captured_inputs[0][mask]
    teacher = captured_outputs[0][mask]
    prompt_ids = torch.arange(len(prompts))[:, None].expand_as(mask)[mask]
    router_weight = mlp.gate.weight.detach().float().cpu().clone()
    teacher_gate_up = mlp.experts.gate_up_proj.detach().cpu().clone()
    teacher_down = mlp.experts.down_proj.detach().cpu().clone()
    top_k = int(mlp.gate.top_k)
    normalize = bool(mlp.gate.norm_topk_prob)
    load_capture_seconds = time.perf_counter() - load_started
    del model, mlp, captured_inputs, captured_outputs
    gc.collect()

    top_weights, top_indices = original_topk(
        hidden, router_weight, top_k, normalize
    )
    route_counts = torch.bincount(
        top_indices.reshape(-1), minlength=teacher_gate_up.shape[0]
    )
    assignments = deterministic_kmeans(
        sampled_expert_features(teacher_gate_up, teacher_down),
        args.micro_experts,
    )
    config = MicroConfig(
        hidden_size=hidden.shape[1],
        original_intermediate_size=teacher_down.shape[-1],
        num_original_experts=teacher_gate_up.shape[0],
        original_top_k=top_k,
        num_micro_experts=args.micro_experts,
        micro_width=args.micro_width,
        max_active_micro_experts=args.max_active_micro,
        normalize_original_top_k=normalize,
        coverage_threshold=args.coverage_threshold,
        risk_threshold=args.risk_threshold,
    )
    student = MicroExpertCodebook(config)
    representatives = initialize_from_teacher(
        student, teacher_gate_up, teacher_down, assignments, route_counts
    )
    if not args.train_gate_up:
        student.gate_up.requires_grad_(False)
    del teacher_gate_up, teacher_down
    gc.collect()

    validation_prompt = prompt_ids.remainder(5) == 0
    train_indices = (~validation_prompt).nonzero(as_tuple=False).flatten()
    validation_indices = validation_prompt.nonzero(as_tuple=False).flatten()
    if train_indices.numel() == 0 or validation_indices.numel() == 0:
        raise RuntimeError("prompt-based train/validation split is empty")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    history = []
    train_started = time.perf_counter()
    student.train()
    for epoch in range(args.epochs):
        permutation = train_indices[torch.randperm(train_indices.numel())]
        epoch_loss = 0.0
        batches = 0
        for offset in range(0, permutation.numel(), args.batch_size):
            indices = permutation[offset : offset + args.batch_size]
            prediction = student.forward_top_m_sparse(
                hidden[indices],
                top_indices[indices],
                top_weights[indices],
                args.max_active_micro,
            )
            target = teacher[indices]
            relative_mse = (
                (prediction - target).square().mean(dim=-1)
                / target.square().mean(dim=-1).clamp_min(1e-12)
            ).mean()
            cosine_loss = (
                1.0 - F.cosine_similarity(prediction, target, dim=-1)
            ).mean()
            code = F.softmax(student.route_logits, dim=-1)
            entropy = -(code * code.clamp_min(1e-9).log()).sum(dim=-1).mean()
            balance = (code.mean(dim=0) - 1.0 / args.micro_experts).square().mean()
            loss = relative_mse + 0.15 * cosine_loss + 0.002 * entropy + balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(epoch_loss / max(1, batches))
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={history[-1]:.6f}",
            flush=True,
        )
    training_seconds = time.perf_counter() - train_started

    student.eval()
    with torch.inference_mode():
        validation = {}
        for active in range(1, args.max_active_micro + 1):
            candidate = student.forward_top_m_sparse(
                hidden[validation_indices],
                top_indices[validation_indices],
                top_weights[validation_indices],
                active,
            )
            validation[str(active)] = metrics(teacher[validation_indices], candidate)

        max_candidate = student.forward_top_m_sparse(
            hidden[validation_indices],
            top_indices[validation_indices],
            top_weights[validation_indices],
            args.max_active_micro,
        )
        validation_error = token_relative_error(
            teacher[validation_indices], max_candidate
        )
        risk_sum = torch.zeros(config.num_original_experts)
        risk_weight = torch.zeros(config.num_original_experts)
        for slot in range(top_k):
            experts = top_indices[validation_indices, slot]
            weights = top_weights[validation_indices, slot]
            risk_sum.index_add_(0, experts, validation_error * weights)
            risk_weight.index_add_(0, experts, weights)
        global_risk = validation_error.mean()
        expert_risk = torch.where(
            risk_weight > 0,
            risk_sum / risk_weight.clamp_min(1e-12),
            global_risk,
        )
        student.expert_risk.copy_(expert_risk)
        _, planned = student.forward_sparse(
            hidden[validation_indices],
            top_indices[validation_indices],
            top_weights[validation_indices],
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "model.safetensors"
    manifest_path = args.output_dir / "manifest.json"
    save_file(
        {
            "gate_up": student.gate_up.detach().bfloat16().contiguous(),
            "down": student.down.detach().bfloat16().contiguous(),
            "route_logits": student.route_logits.detach().float().contiguous(),
            "expert_risk": student.expert_risk.detach().float().contiguous(),
            "router_weight": router_weight.bfloat16().contiguous(),
        },
        str(weights_path),
    )

    original_parameters = top_k * 3 * config.hidden_size * config.original_intermediate_size
    micro_parameters = 3 * config.hidden_size * config.micro_width
    manifest = {
        "schema_version": 1,
        "format": "hesr-amc-micro-codebook-v1",
        "source": {
            "snapshot": str(args.snapshot.resolve()),
            "layer": args.layer,
        },
        "architecture": {
            "hidden_size": config.hidden_size,
            "original_intermediate_size": config.original_intermediate_size,
            "num_original_experts": config.num_original_experts,
            "original_top_k": config.original_top_k,
            "normalize_original_top_k": config.normalize_original_top_k,
            "num_micro_experts": config.num_micro_experts,
            "micro_width": config.micro_width,
        },
        "policy": {
            "max_active_micro_experts": config.max_active_micro_experts,
            "coverage_threshold": config.coverage_threshold,
            "risk_threshold": config.risk_threshold,
            "fallback": "original expert path",
        },
        "active_mass": {
            "original_parameters_per_token": original_parameters,
            "parameters_per_micro_expert": micro_parameters,
            "max_fast_path_parameters_per_token": (
                micro_parameters * config.max_active_micro_experts
            ),
            "fast_path_reduction_at_max_budget": original_parameters
            / (micro_parameters * config.max_active_micro_experts),
            "resident_codebook_parameters": micro_parameters
            * config.num_micro_experts,
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
            "train_tokens": int(train_indices.numel()),
            "validation_tokens": int(validation_indices.numel()),
            "route_counts": route_counts.tolist(),
        },
        "initialization": {
            "cluster_assignments": assignments.tolist(),
            "representative_experts": representatives,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gate_up_trainable": args.train_gate_up,
            "loss_history": history,
        },
        "validation_by_active_micro_experts": validation,
        "policy_validation": {
            "fast_path_tokens": int((~planned["fallback"]).sum().item()),
            "fallback_tokens": int(planned["fallback"].sum().item()),
            "fallback_rate": float(planned["fallback"].float().mean().item()),
            "mean_active_micro_experts": float(
                planned["counts"][~planned["fallback"]].float().mean().item()
            )
            if (~planned["fallback"]).any()
            else None,
            "mean_predicted_risk": float(planned["route_risk"].mean().item()),
        },
        "active_mass": manifest["active_mass"],
        "output": {
            "directory": str(args.output_dir.resolve()),
            "manifest": str(manifest_path.resolve()),
            "weights": str(weights_path.resolve()),
            "weights_bytes": weights_path.stat().st_size,
        },
        "timing": {
            "model_load_and_capture_seconds": load_capture_seconds,
            "training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.report),
        "weights": str(weights_path),
        "validation": validation,
        "policy_validation": report["policy_validation"],
        "active_mass": report["active_mass"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
