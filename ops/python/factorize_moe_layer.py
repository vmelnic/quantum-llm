#!/usr/bin/env python3
"""Weight-only HESR factorization experiment for one packed MoE layer."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


PROJECTIONS = ("input_linear", "output_linear")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tensor(snapshot: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = snapshot / weight_map[name]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def sampled_features(weights: torch.Tensor, sample_count: int) -> torch.Tensor:
    experts = weights.shape[0]
    flattened = weights.reshape(experts, -1)
    count = min(sample_count, flattened.shape[1])
    indices = torch.linspace(
        0, flattened.shape[1] - 1, steps=count, dtype=torch.float64
    ).round().to(torch.int64)
    return flattened[:, indices].to(torch.float32)


def neuron_features(
    input_weights: torch.Tensor,
    output_weights: torch.Tensor,
    sample_count: int = 128,
) -> torch.Tensor:
    """Build a joint gate/up/down signature for every intermediate neuron."""
    experts, twice_intermediate, hidden = input_weights.shape
    intermediate = twice_intermediate // 2
    if twice_intermediate != 2 * intermediate:
        raise ValueError("input_linear is not a concatenated gate/up tensor")
    if output_weights.shape != (experts, hidden, intermediate):
        raise ValueError(
            f"incompatible output_linear shape {list(output_weights.shape)}"
        )

    count = min(sample_count, hidden)
    indices = torch.linspace(0, hidden - 1, steps=count, dtype=torch.float64)
    indices = indices.round().to(torch.int64)
    gate = input_weights[:, :intermediate, :][:, :, indices]
    up = input_weights[:, intermediate:, :][:, :, indices]
    down = output_weights[:, indices, :].transpose(1, 2)
    features = torch.cat((gate, up, down), dim=2).to(torch.float32)
    centered = features - features.mean(dim=2, keepdim=True)
    return centered / centered.norm(dim=2, keepdim=True).clamp_min(1e-12)


def greedy_permutation(similarity: torch.Tensor) -> torch.Tensor:
    """Approximate one-to-one neuron assignment, reference row -> candidate row."""
    size = similarity.shape[0]
    if similarity.shape != (size, size):
        raise ValueError("similarity matrix must be square")
    order = torch.argsort(similarity.reshape(-1), descending=True)
    reference_used = torch.zeros(size, dtype=torch.bool)
    candidate_used = torch.zeros(size, dtype=torch.bool)
    permutation = torch.full((size,), -1, dtype=torch.int64)
    matched = 0
    for flat_index in order.tolist():
        reference = flat_index // size
        candidate = flat_index % size
        if not reference_used[reference] and not candidate_used[candidate]:
            permutation[reference] = candidate
            reference_used[reference] = True
            candidate_used[candidate] = True
            matched += 1
            if matched == size:
                break
    if (permutation < 0).any():
        raise RuntimeError("greedy assignment did not produce a full permutation")
    return permutation


def align_to_reference(
    original_input: torch.Tensor,
    original_output: torch.Tensor,
    reference_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float]]:
    candidate_features = neuron_features(original_input, original_output)
    experts, twice_intermediate, hidden = original_input.shape
    intermediate = twice_intermediate // 2
    aligned_input = torch.empty_like(original_input)
    aligned_output = torch.empty_like(original_output)
    identity_scores: list[float] = []
    matched_scores: list[float] = []

    for expert in range(experts):
        similarity = reference_features @ candidate_features[expert].T
        permutation = greedy_permutation(similarity)
        identity_scores.append(float(similarity.diag().mean().item()))
        matched_scores.append(
            float(similarity[torch.arange(intermediate), permutation].mean().item())
        )
        aligned_input[expert, :intermediate] = original_input[
            expert, :intermediate
        ][permutation]
        aligned_input[expert, intermediate:] = original_input[
            expert, intermediate:
        ][permutation]
        aligned_output[expert] = original_output[expert].index_select(1, permutation)
    return aligned_input, aligned_output, identity_scores, matched_scores


def align_expert_neurons(
    input_weights: torch.Tensor, output_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Two-pass deterministic weight matching across intermediate neurons."""
    initial_reference = neuron_features(
        input_weights[:1], output_weights[:1]
    )[0]
    first_input, first_output, first_identity, first_matched = align_to_reference(
        input_weights, output_weights, initial_reference
    )

    centroid_input = first_input.mean(dim=0, keepdim=True)
    centroid_output = first_output.mean(dim=0, keepdim=True)
    centroid_reference = neuron_features(centroid_input, centroid_output)[0]
    final_input, final_output, final_identity, final_matched = align_to_reference(
        input_weights, output_weights, centroid_reference
    )
    return final_input, final_output, {
        "method": "two-pass greedy joint gate-up-down matching",
        "feature_coordinates_per_branch": min(128, input_weights.shape[2]),
        "first_pass_mean_identity_cosine": sum(first_identity) / len(first_identity),
        "first_pass_mean_matched_cosine": sum(first_matched) / len(first_matched),
        "final_pass_mean_identity_cosine": sum(final_identity) / len(final_identity),
        "final_pass_mean_matched_cosine": sum(final_matched) / len(final_matched),
    }


def deterministic_kmeans(features: torch.Tensor, clusters: int, iterations: int = 30) -> torch.Tensor:
    experts = features.shape[0]
    if clusters < 1 or clusters > experts:
        raise ValueError(f"invalid cluster count {clusters} for {experts} experts")

    centered = features - features.mean(dim=1, keepdim=True)
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)

    chosen = [0]
    centers = [normalized[0]]
    while len(chosen) < clusters:
        center_matrix = torch.stack(centers)
        distances = torch.cdist(normalized, center_matrix).square().min(dim=1).values
        distances[torch.tensor(chosen, dtype=torch.int64)] = -1
        next_index = int(distances.argmax().item())
        chosen.append(next_index)
        centers.append(normalized[next_index])

    center_matrix = torch.stack(centers)
    assignments = torch.full((experts,), -1, dtype=torch.int64)
    for _ in range(iterations):
        new_assignments = torch.cdist(normalized, center_matrix).argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        updated = []
        for cluster in range(clusters):
            members = normalized[assignments == cluster]
            if members.shape[0] == 0:
                distances = torch.cdist(normalized, center_matrix).min(dim=1).values
                updated.append(normalized[int(distances.argmax().item())])
            else:
                updated.append(members.mean(dim=0))
        center_matrix = torch.stack(updated)
    return assignments


def cluster_bases(weights: torch.Tensor, assignments: torch.Tensor, clusters: int) -> list[torch.Tensor]:
    return [weights[assignments == cluster].mean(dim=0) for cluster in range(clusters)]


def base_sweep_for_projection(
    weights: torch.Tensor, assignments_by_count: dict[int, torch.Tensor]
) -> dict[int, dict[str, float | int]]:
    original_energy = float(weights.square().sum().item())
    results: dict[int, dict[str, float | int]] = {}
    for clusters, assignments in assignments_by_count.items():
        bases = cluster_bases(weights, assignments, clusters)
        residual_energy = 0.0
        for expert in range(weights.shape[0]):
            residual = weights[expert] - bases[int(assignments[expert].item())]
            residual_energy += float(residual.square().sum().item())
        results[clusters] = {
            "base_bytes_bf16": clusters * weights[0].numel() * 2,
            "relative_residual_energy": residual_energy / original_energy,
            "relative_residual_norm": math.sqrt(residual_energy / original_energy),
        }
    return results


def rank_grid(max_rank: int) -> list[int]:
    candidates = {0, max_rank}
    value = 4
    while value < max_rank:
        candidates.add(value)
        value *= 2
    return sorted(candidates)


def factorize_projection(
    weights: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
    ranks: list[int],
    sparse_fractions: list[float],
    seed: int,
) -> dict[str, Any]:
    experts, rows, columns = weights.shape
    bases = cluster_bases(weights, assignments, clusters)
    original_energy = float(weights.square().sum().item())
    base_bytes = clusters * rows * columns * 2
    variants: dict[tuple[int, float], dict[str, float | int]] = {}
    for rank in ranks:
        for fraction in sparse_fractions:
            variants[(rank, fraction)] = {
                "error_energy": 0.0,
                "low_rank_bytes_bf16": experts * rank * (rows + columns) * 2,
                "sparse_bytes": 0,
                "sparse_values": 0,
            }

    expert_timings: list[float] = []
    for expert in range(experts):
        started = time.perf_counter()
        residual = weights[expert] - bases[int(assignments[expert].item())]
        max_rank = max(ranks)
        if max_rank > 0:
            torch.manual_seed(seed + expert)
            u, singular, v = torch.svd_lowrank(residual, q=max_rank, niter=1)
        else:
            u = singular = v = None

        for rank in ranks:
            if rank == 0:
                tail = residual
            else:
                approximation = (u[:, :rank] * singular[:rank]) @ v[:, :rank].T
                tail = residual - approximation
            tail_energy = float(tail.square().sum().item())
            flattened = tail.reshape(-1)
            for fraction in sparse_fractions:
                count = int(round(flattened.numel() * fraction))
                count = min(max(count, 0), flattened.numel())
                captured = 0.0
                if count:
                    values = torch.topk(flattened.abs(), count, sorted=False).values
                    captured = float(values.square().sum().item())
                result = variants[(rank, fraction)]
                result["error_energy"] += max(0.0, tail_energy - captured)
                result["sparse_values"] += count
                # BF16 value plus a 32-bit linear index.
                result["sparse_bytes"] += count * 6
        expert_timings.append(time.perf_counter() - started)

    output_variants = []
    original_bytes = experts * rows * columns * 2
    for (rank, fraction), values in sorted(variants.items()):
        representation_bytes = (
            base_bytes
            + int(values["low_rank_bytes_bf16"])
            + int(values["sparse_bytes"])
            + experts * 2  # uint16 cluster assignment
        )
        error_energy = float(values["error_energy"])
        output_variants.append(
            {
                "rank": rank,
                "sparse_fraction": fraction,
                "base_bytes_bf16": base_bytes,
                "low_rank_bytes_bf16": int(values["low_rank_bytes_bf16"]),
                "sparse_values": int(values["sparse_values"]),
                "sparse_bytes": int(values["sparse_bytes"]),
                "representation_bytes": representation_bytes,
                "compression_ratio": original_bytes / representation_bytes,
                "relative_error_energy": error_energy / original_energy,
                "relative_error_norm": math.sqrt(error_energy / original_energy),
            }
        )

    return {
        "shape": [experts, rows, columns],
        "original_bytes_bf16": original_bytes,
        "original_energy": original_energy,
        "base_bytes_bf16": base_bytes,
        "expert_runtime_seconds": expert_timings,
        "variants": output_variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--align-neurons", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(os.cpu_count() or 1, 12))

    config = read_json(args.snapshot / "config.json")
    index = read_json(args.snapshot / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]
    num_experts = int(config.get("num_local_experts", config.get("num_experts")))
    if not 0 <= args.layer < int(config["num_hidden_layers"]):
        raise ValueError(f"layer {args.layer} is outside the model")
    if args.clusters > num_experts:
        raise ValueError("cluster count exceeds expert count")

    names = {
        projection: (
            f"model.layers.{args.layer}.block_sparse_moe.{projection}.weight"
        )
        for projection in PROJECTIONS
    }

    weights_by_projection: dict[str, torch.Tensor] = {}
    for projection in PROJECTIONS:
        packed = load_tensor(args.snapshot, weight_map, names[projection])
        if packed.shape[0] != num_experts:
            raise ValueError(f"unexpected expert dimension for {names[projection]}")
        weights_by_projection[projection] = packed.to(torch.float32)
        del packed

    alignment_report = None
    if args.align_neurons:
        aligned_input, aligned_output, alignment_report = align_expert_neurons(
            weights_by_projection["input_linear"],
            weights_by_projection["output_linear"],
        )
        weights_by_projection["input_linear"] = aligned_input
        weights_by_projection["output_linear"] = aligned_output

    feature_parts = [
        sampled_features(weights_by_projection[projection], args.sample_count)
        for projection in PROJECTIONS
    ]
    features = torch.cat(feature_parts, dim=1)

    cluster_counts = [value for value in (1, 2, 4, 8) if value <= num_experts]
    assignments_by_count = {
        count: deterministic_kmeans(features, count) for count in cluster_counts
    }
    selected_assignments = deterministic_kmeans(features, args.clusters)

    base_sweep: dict[int, dict[str, Any]] = {
        count: {
            "clusters": count,
            "cluster_sizes": [
                int((assignments_by_count[count] == cluster).sum().item())
                for cluster in range(count)
            ],
            "original_energy": 0.0,
            "residual_energy": 0.0,
            "base_bytes_bf16": 0,
            "original_bytes_bf16": 0,
        }
        for count in cluster_counts
    }

    ranks = rank_grid(args.max_rank)
    sparse_fractions = [0.0, 0.001, 0.01]
    projection_reports: dict[str, Any] = {}
    for projection in PROJECTIONS:
        weights = weights_by_projection[projection]

        projection_sweep = base_sweep_for_projection(weights, assignments_by_count)
        original_energy = float(weights.square().sum().item())
        original_bytes = weights.numel() * 2
        for count, sweep in projection_sweep.items():
            aggregate = base_sweep[count]
            aggregate["original_energy"] += original_energy
            aggregate["residual_energy"] += (
                float(sweep["relative_residual_energy"]) * original_energy
            )
            aggregate["base_bytes_bf16"] += int(sweep["base_bytes_bf16"])
            aggregate["original_bytes_bf16"] += original_bytes

        projection_reports[projection] = factorize_projection(
            weights=weights,
            assignments=selected_assignments,
            clusters=args.clusters,
            ranks=ranks,
            sparse_fractions=sparse_fractions,
            seed=args.seed,
        )

    base_sweep_output = []
    for count in cluster_counts:
        aggregate = base_sweep[count]
        ratio = aggregate["residual_energy"] / aggregate["original_energy"]
        aggregate["relative_residual_energy"] = ratio
        aggregate["relative_residual_norm"] = math.sqrt(ratio)
        aggregate["base_only_compression_ratio"] = (
            aggregate["original_bytes_bf16"] / aggregate["base_bytes_bf16"]
        )
        del aggregate["original_energy"]
        del aggregate["residual_energy"]
        base_sweep_output.append(aggregate)

    combined: dict[tuple[int, float], dict[str, float | int]] = {}
    total_original_energy = sum(
        report["original_energy"] for report in projection_reports.values()
    )
    total_original_bytes = sum(
        report["original_bytes_bf16"] for report in projection_reports.values()
    )
    for projection_report in projection_reports.values():
        projection_energy = projection_report["original_energy"]
        for variant in projection_report["variants"]:
            key = (variant["rank"], variant["sparse_fraction"])
            item = combined.setdefault(
                key,
                {"representation_bytes": 0, "error_energy": 0.0},
            )
            item["representation_bytes"] += variant["representation_bytes"]
            item["error_energy"] += (
                variant["relative_error_energy"] * projection_energy
            )

    combined_variants = []
    for (rank, sparse_fraction), values in sorted(combined.items()):
        representation_bytes = int(values["representation_bytes"])
        error_energy = float(values["error_energy"])
        combined_variants.append(
            {
                "rank": rank,
                "sparse_fraction": sparse_fraction,
                "representation_bytes": representation_bytes,
                "compression_ratio": total_original_bytes / representation_bytes,
                "relative_error_energy": error_energy / total_original_energy,
                "relative_error_norm": math.sqrt(error_energy / total_original_energy),
            }
        )

    report = {
        "schema_version": 1,
        "snapshot": str(args.snapshot.resolve()),
        "layer": args.layer,
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "seed": args.seed,
        "num_experts": num_experts,
        "selected_clusters": args.clusters,
        "neurons_aligned": args.align_neurons,
        "alignment": alignment_report,
        "selected_cluster_sizes": [
            int((selected_assignments == cluster).sum().item())
            for cluster in range(args.clusters)
        ],
        "ranks": ranks,
        "sparse_fractions": sparse_fractions,
        "base_sweep": base_sweep_output,
        "projections": projection_reports,
        "combined_variants": combined_variants,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Weight error only; activation, hidden-state, KV, and logit errors are not yet measured.",
            "Clusters are learned from deterministic sampled weight coordinates.",
            "Low-rank factors and sparse values are budgeted as BF16; sparse indices are uint32.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    best = min(
        (item for item in combined_variants if item["compression_ratio"] >= 4.0),
        key=lambda item: item["relative_error_norm"],
        default=None,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "layer": args.layer,
                "runtime_seconds": report["runtime_seconds"],
                "best_at_least_4x": best,
            }
        )
    )


if __name__ == "__main__":
    main()
