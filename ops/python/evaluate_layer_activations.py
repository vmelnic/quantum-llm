#!/usr/bin/env python3
"""Evaluate one HESR approximation on real teacher activations, CPU only."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from factorize_moe_layer import (
    PROJECTIONS,
    align_expert_neurons,
    cluster_bases,
    deterministic_kmeans,
    sampled_features,
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_moe(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    suffix = f"layers.{layer}.block_sparse_moe"
    matches = [module for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one module ending in {suffix}, found {len(matches)}")
    return matches[0]


def reconstruct_weights(
    weights: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
    rank: int,
    sparse_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    experts, rows, columns = weights.shape
    bases = cluster_bases(weights, assignments, clusters)
    approximated = torch.empty_like(weights)
    sparse_values = 0
    started = time.perf_counter()

    for expert in range(experts):
        base = bases[int(assignments[expert].item())]
        residual = weights[expert] - base
        if rank:
            torch.manual_seed(seed + expert)
            u, singular, v = torch.svd_lowrank(residual, q=rank, niter=1)
            low_rank = (u[:, :rank] * singular[:rank]) @ v[:, :rank].T
        else:
            low_rank = torch.zeros_like(residual)
        tail = residual - low_rank
        count = int(round(tail.numel() * sparse_fraction))
        sparse = torch.zeros_like(tail)
        if count:
            indices = torch.topk(tail.abs().reshape(-1), count, sorted=False).indices
            sparse_flat = sparse.reshape(-1)
            tail_flat = tail.reshape(-1)
            sparse_flat[indices] = tail_flat[indices]
            sparse_values += count
        approximated[expert] = base + low_rank + sparse

    original_bytes = experts * rows * columns * 2
    representation_bytes = (
        clusters * rows * columns * 2
        + experts * rank * (rows + columns) * 2
        + sparse_values * 6
        + experts * 2
    )
    return approximated, {
        "shape": [experts, rows, columns],
        "original_bytes_bf16": original_bytes,
        "representation_bytes": representation_bytes,
        "compression_ratio": original_bytes / representation_bytes,
        "sparse_values": sparse_values,
        "runtime_seconds": time.perf_counter() - started,
    }


def expert_outputs(
    hidden: torch.Tensor,
    top_indices: torch.Tensor,
    top_gates: torch.Tensor,
    input_weights: torch.Tensor,
    output_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, hidden_size = hidden.shape
    experts = input_weights.shape[0]
    aggregate = torch.zeros(tokens, hidden_size, dtype=torch.float32)
    routed_outputs = []

    for expert in range(experts):
        locations = (top_indices == expert).nonzero(as_tuple=False)
        if locations.numel() == 0:
            continue
        token_indices = locations[:, 0]
        route_slots = locations[:, 1]
        expert_input = hidden.index_select(0, token_indices)
        projected = functional.linear(expert_input, input_weights[expert])
        gate, up = projected.chunk(2, dim=-1)
        activated = functional.silu(gate) * up
        output = functional.linear(activated, output_weights[expert])
        route_gates = top_gates[token_indices, route_slots]
        weighted = output * route_gates[:, None]
        aggregate.index_add_(0, token_indices, weighted)
        routed_outputs.append(output)
    return aggregate, torch.cat(routed_outputs, dim=0)


def relative_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = candidate - reference
    reference_energy = float(reference.square().sum().item())
    error_energy = float(difference.square().sum().item())
    cosine = functional.cosine_similarity(reference, candidate, dim=-1)
    return {
        "reference_norm": math.sqrt(reference_energy),
        "error_norm": math.sqrt(error_energy),
        "relative_error_energy": error_energy / max(reference_energy, 1e-30),
        "relative_error_norm": math.sqrt(error_energy / max(reference_energy, 1e-30)),
        "mean_cosine": float(cosine.mean().item()),
        "p05_cosine": float(torch.quantile(cosine, 0.05).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--sparse-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(os.cpu_count() or 1, 12))
    prompts = read_json(args.prompts)
    if not prompts:
        raise ValueError("calibration prompt list is empty")

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

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.snapshot),
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to("cpu")
    model.eval()
    load_seconds = time.perf_counter() - load_started

    moe = find_moe(model, args.layer)
    captured: list[torch.Tensor] = []

    def capture_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().to(torch.float32).cpu())

    hook = moe.register_forward_pre_hook(capture_input)
    forward_started = time.perf_counter()
    model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        use_cache=False,
    )
    forward_seconds = time.perf_counter() - forward_started
    hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one captured activation tensor, found {len(captured)}")

    valid = encoded["attention_mask"].bool().reshape(-1)
    hidden = captured[0].reshape(-1, captured[0].shape[-1])[valid]
    original_input = moe.input_linear.weight.detach().to(torch.float32).cpu().clone()
    original_output = moe.output_linear.weight.detach().to(torch.float32).cpu().clone()
    router_weight = moe.router.layer.weight.detach().to(torch.float32).cpu().clone()
    top_k = moe.router.top_k
    del model, moe, captured

    logits = functional.linear(hidden, router_weight)
    top_logits, top_indices = logits.topk(top_k, dim=1)
    top_gates = torch.softmax(top_logits, dim=1)
    route_counts = torch.bincount(top_indices.reshape(-1), minlength=original_input.shape[0])

    aligned_input, aligned_output, alignment = align_expert_neurons(
        original_input, original_output
    )
    features = torch.cat(
        (
            sampled_features(aligned_input, 4096),
            sampled_features(aligned_output, 4096),
        ),
        dim=1,
    )
    assignments = deterministic_kmeans(features, args.clusters)

    approximate_input, input_budget = reconstruct_weights(
        aligned_input,
        assignments,
        args.clusters,
        args.rank,
        args.sparse_fraction,
        args.seed,
    )
    approximate_output, output_budget = reconstruct_weights(
        aligned_output,
        assignments,
        args.clusters,
        args.rank,
        args.sparse_fraction,
        args.seed,
    )

    exact_aggregate, exact_routes = expert_outputs(
        hidden, top_indices, top_gates, original_input, original_output
    )
    aligned_aggregate, aligned_routes = expert_outputs(
        hidden, top_indices, top_gates, aligned_input, aligned_output
    )
    approximate_aggregate, approximate_routes = expert_outputs(
        hidden, top_indices, top_gates, approximate_input, approximate_output
    )

    total_original_bytes = (
        input_budget["original_bytes_bf16"] + output_budget["original_bytes_bf16"]
    )
    total_representation_bytes = (
        input_budget["representation_bytes"] + output_budget["representation_bytes"]
    )
    report = {
        "schema_version": 1,
        "snapshot": str(args.snapshot.resolve()),
        "layer": args.layer,
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "torch_cuda_available": torch.cuda.is_available(),
        "prompt_count": len(prompts),
        "calibration_tokens": int(hidden.shape[0]),
        "routed_calls": int(hidden.shape[0] * top_k),
        "top_k": top_k,
        "route_counts": route_counts.tolist(),
        "active_experts": int((route_counts > 0).sum().item()),
        "approximation": {
            "clusters": args.clusters,
            "cluster_sizes": [
                int((assignments == cluster).sum().item())
                for cluster in range(args.clusters)
            ],
            "rank": args.rank,
            "sparse_fraction": args.sparse_fraction,
            "original_bytes_bf16": total_original_bytes,
            "representation_bytes": total_representation_bytes,
            "compression_ratio": total_original_bytes / total_representation_bytes,
            "input": input_budget,
            "output": output_budget,
        },
        "alignment": alignment,
        "alignment_exactness": {
            "aggregate": relative_metrics(exact_aggregate, aligned_aggregate),
            "routed": relative_metrics(exact_routes, aligned_routes),
        },
        "approximation_error": {
            "aggregate_moe_output": relative_metrics(
                exact_aggregate, approximate_aggregate
            ),
            "routed_expert_outputs": relative_metrics(
                exact_routes, approximate_routes
            ),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "teacher_forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "limitations": [
            "Small calibration set; this is not a quality benchmark.",
            "Only layer-local MoE output is measured; hidden/KV/logit drift is not yet measured.",
            "Greedy neuron alignment is approximate, not an optimal Hungarian assignment.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "calibration_tokens": report["calibration_tokens"],
                "compression_ratio": report["approximation"]["compression_ratio"],
                "aggregate_relative_error_norm": report["approximation_error"][
                    "aggregate_moe_output"
                ]["relative_error_norm"],
                "aggregate_mean_cosine": report["approximation_error"][
                    "aggregate_moe_output"
                ]["mean_cosine"],
                "total_seconds": report["timing"]["total_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
