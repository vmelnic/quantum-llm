#!/usr/bin/env python3
"""Core modules for the first executable HESR-AMC micro-expert format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class MicroConfig:
    hidden_size: int
    original_intermediate_size: int
    num_original_experts: int
    original_top_k: int
    num_micro_experts: int
    micro_width: int
    max_active_micro_experts: int
    normalize_original_top_k: bool
    coverage_threshold: float
    risk_threshold: float

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "MicroConfig":
        architecture = manifest["architecture"]
        policy = manifest["policy"]
        return cls(
            hidden_size=int(architecture["hidden_size"]),
            original_intermediate_size=int(
                architecture["original_intermediate_size"]
            ),
            num_original_experts=int(architecture["num_original_experts"]),
            original_top_k=int(architecture["original_top_k"]),
            num_micro_experts=int(architecture["num_micro_experts"]),
            micro_width=int(architecture["micro_width"]),
            max_active_micro_experts=int(policy["max_active_micro_experts"]),
            normalize_original_top_k=bool(
                architecture["normalize_original_top_k"]
            ),
            coverage_threshold=float(policy["coverage_threshold"]),
            risk_threshold=float(policy["risk_threshold"]),
        )


@dataclass(frozen=True)
class NeuronBankConfig:
    hidden_size: int
    original_intermediate_size: int
    retained_intermediate_size: int
    num_experts: int
    top_k: int
    normalize_top_k: bool
    correction_rank: int
    risk_threshold: float

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "NeuronBankConfig":
        architecture = manifest["architecture"]
        policy = manifest["policy"]
        return cls(
            hidden_size=int(architecture["hidden_size"]),
            original_intermediate_size=int(
                architecture["original_intermediate_size"]
            ),
            retained_intermediate_size=int(
                architecture["retained_intermediate_size"]
            ),
            num_experts=int(architecture["num_experts"]),
            top_k=int(architecture["top_k"]),
            normalize_top_k=bool(architecture["normalize_top_k"]),
            correction_rank=int(architecture["correction_rank"]),
            risk_threshold=float(policy["risk_threshold"]),
        )


def original_topk(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(F.linear(hidden, router_weight), dim=-1)
    weights, indices = probabilities.topk(top_k, dim=-1)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights, indices


class MicroExpertCodebook(nn.Module):
    """Shared micro-experts addressed through a learned original->micro map.

    Dense mode evaluates the whole codebook and is used only by the offline
    compiler. Sparse mode groups tokens by selected micro-expert and performs
    only the matmuls admitted by the AMC budget.
    """

    def __init__(self, config: MicroConfig) -> None:
        super().__init__()
        self.config = config
        m = config.num_micro_experts
        w = config.micro_width
        d = config.hidden_size
        self.gate_up = nn.Parameter(torch.empty(m, 2 * w, d))
        self.down = nn.Parameter(torch.empty(m, d, w))
        self.route_logits = nn.Parameter(
            torch.empty(config.num_original_experts, m)
        )
        self.register_buffer(
            "expert_risk", torch.zeros(config.num_original_experts)
        )
        nn.init.normal_(self.gate_up, mean=0.0, std=d**-0.5)
        nn.init.normal_(self.down, mean=0.0, std=w**-0.5)
        nn.init.zeros_(self.route_logits)

    def micro_scores(
        self, top_indices: torch.Tensor, top_weights: torch.Tensor
    ) -> torch.Tensor:
        code = F.softmax(self.route_logits, dim=-1)
        scores = (code[top_indices] * top_weights[..., None]).sum(dim=1)
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _all_outputs(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = torch.einsum("nd,mkd->nmk", hidden, self.gate_up)
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return torch.einsum("nmw,mdw->nmd", activated, self.down)

    def forward_dense(
        self,
        hidden: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        active_micro_experts: int | None = None,
    ) -> torch.Tensor:
        scores = self.micro_scores(top_indices, top_weights)
        if active_micro_experts is not None:
            active = min(max(1, active_micro_experts), scores.shape[1])
            selected_weights, selected = scores.topk(active, dim=-1)
            selected_weights = selected_weights / selected_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            sparse_scores = torch.zeros_like(scores)
            sparse_scores.scatter_(1, selected, selected_weights)
            scores = sparse_scores
        return (self._all_outputs(hidden) * scores[..., None]).sum(dim=1)

    def forward_top_m_sparse(
        self,
        hidden: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        active_micro_experts: int,
    ) -> torch.Tensor:
        """Differentiable deployed-path forward without risk fallback."""
        scores = self.micro_scores(top_indices, top_weights)
        active_count = min(max(1, active_micro_experts), scores.shape[1])
        selected_weights, selected = scores.topk(active_count, dim=-1)
        selected_weights = selected_weights / selected_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        output = torch.zeros(
            hidden.shape[0], hidden.shape[1], dtype=hidden.dtype, device=hidden.device
        )
        for micro in range(self.config.num_micro_experts):
            locations = (selected == micro).nonzero(as_tuple=False)
            if locations.numel() == 0:
                continue
            token_indices = locations[:, 0]
            slots = locations[:, 1]
            current = hidden.index_select(0, token_indices)
            gate_up = F.linear(current, self.gate_up[micro])
            gate, up = gate_up.chunk(2, dim=-1)
            result = F.linear(F.silu(gate) * up, self.down[micro])
            weights = selected_weights[token_indices, slots]
            output.index_add_(0, token_indices, result * weights[:, None])
        return output

    def plan(
        self, top_indices: torch.Tensor, top_weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        scores = self.micro_scores(top_indices, top_weights)
        maximum = min(self.config.max_active_micro_experts, scores.shape[1])
        selected_weights, selected = scores.topk(maximum, dim=-1)
        cumulative = selected_weights.cumsum(dim=-1)
        reaches = cumulative >= self.config.coverage_threshold
        first = reaches.to(torch.int64).argmax(dim=-1) + 1
        reaches_any = reaches.any(dim=-1)
        counts = torch.where(
            reaches_any, first, torch.full_like(first, maximum)
        )
        coverage_at_budget = cumulative.gather(1, (counts - 1)[:, None]).squeeze(1)
        route_risk = (
            self.expert_risk[top_indices] * top_weights
        ).sum(dim=-1) / top_weights.sum(dim=-1).clamp_min(1e-12)
        fallback = (
            (route_risk > self.config.risk_threshold)
            | (coverage_at_budget < self.config.coverage_threshold)
        )
        positions = torch.arange(maximum, device=hidden_device(top_indices))[None, :]
        active = positions < counts[:, None]
        normalized = selected_weights * active
        normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        return {
            "scores": scores,
            "selected": selected,
            "selected_weights": normalized,
            "active": active,
            "counts": counts,
            "coverage": coverage_at_budget,
            "route_risk": route_risk,
            "fallback": fallback,
        }

    def forward_sparse(
        self,
        hidden: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        plan = self.plan(top_indices, top_weights)
        output = torch.zeros_like(hidden)
        usable = plan["active"] & ~plan["fallback"][:, None]
        for micro in range(self.config.num_micro_experts):
            locations = ((plan["selected"] == micro) & usable).nonzero(
                as_tuple=False
            )
            if locations.numel() == 0:
                continue
            token_indices = locations[:, 0]
            slots = locations[:, 1]
            current = hidden.index_select(0, token_indices)
            gate_up = F.linear(current, self.gate_up[micro])
            gate, up = gate_up.chunk(2, dim=-1)
            result = F.linear(F.silu(gate) * up, self.down[micro])
            weights = plan["selected_weights"][token_indices, slots]
            output.index_add_(0, token_indices, result * weights[:, None])
        return output, plan

    @property
    def parameters_per_micro_expert(self) -> int:
        return 3 * self.config.hidden_size * self.config.micro_width


class PrunedExpertBank(nn.Module):
    """Original expert neurons retained by activation-weighted contribution."""

    def __init__(self, config: NeuronBankConfig) -> None:
        super().__init__()
        self.config = config
        e = config.num_experts
        w = config.retained_intermediate_size
        d = config.hidden_size
        self.gate_up = nn.Parameter(torch.empty(e, 2 * w, d), requires_grad=False)
        self.down = nn.Parameter(torch.empty(e, d, w), requires_grad=False)
        r = config.correction_rank
        self.correction_in = nn.Parameter(torch.zeros(r, d), requires_grad=False)
        self.correction_out = nn.Parameter(torch.zeros(d, r), requires_grad=False)
        self.correction_hidden_mean = nn.Parameter(
            torch.zeros(d), requires_grad=False
        )
        self.correction_residual_mean = nn.Parameter(
            torch.zeros(d), requires_grad=False
        )
        self.register_buffer("expert_risk", torch.zeros(e))

    def plan(
        self, top_indices: torch.Tensor, top_weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        risk = (
            self.expert_risk[top_indices] * top_weights
        ).sum(dim=-1) / top_weights.sum(dim=-1).clamp_min(1e-12)
        return {"route_risk": risk, "fallback": risk > self.config.risk_threshold}

    def forward_fast(
        self,
        hidden: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        fallback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        usable_tokens = (
            torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
            if fallback is None
            else ~fallback
        )
        for expert in range(self.config.num_experts):
            locations = ((top_indices == expert) & usable_tokens[:, None]).nonzero(
                as_tuple=False
            )
            if locations.numel() == 0:
                continue
            token_indices = locations[:, 0]
            slots = locations[:, 1]
            current = hidden.index_select(0, token_indices)
            gate_up = F.linear(current, self.gate_up[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            result = F.linear(F.silu(gate) * up, self.down[expert])
            weights = top_weights[token_indices, slots]
            output.index_add_(0, token_indices, result * weights[:, None])
        corrected_tokens = usable_tokens.nonzero(as_tuple=False).flatten()
        if corrected_tokens.numel() and self.config.correction_rank > 0:
            current = hidden.index_select(0, corrected_tokens)
            coefficients = F.linear(
                current - self.correction_hidden_mean, self.correction_in
            )
            correction = F.linear(coefficients, self.correction_out)
            correction = correction + self.correction_residual_mean
            output.index_add_(0, corrected_tokens, correction)
        return output

    @property
    def parameters_per_expert(self) -> int:
        return (
            3
            * self.config.hidden_size
            * self.config.retained_intermediate_size
        )

    @property
    def correction_parameters(self) -> int:
        return (
            2 * self.config.hidden_size * self.config.correction_rank
            + 2 * self.config.hidden_size
        )


def hidden_device(tensor: torch.Tensor) -> torch.device:
    # Kept as a tiny helper so plan() remains torch.compile friendly later.
    return tensor.device


def token_relative_error(
    reference: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    numerator = (candidate - reference).square().mean(dim=-1)
    denominator = reference.square().mean(dim=-1).clamp_min(1e-12)
    return torch.sqrt(numerator / denominator)
