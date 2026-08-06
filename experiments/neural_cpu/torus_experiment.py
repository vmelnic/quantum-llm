"""Reversible torus/butterfly tied-depth experiment.

This candidate is intentionally classified as a tied-depth structured map, not
as evidence of data-dependent routing.  Its fixed program expands a register
dependency graph with an algorithmic round-robin pairing schedule and reuses
the same determinant-one coupling bank at every tick.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from experiment import (
    TraceData,
    device_context,
    iter_batches,
    load_trace,
    normalized_mse,
    parameter_bytes,
    sha256_file,
)


ISA = ("STRIDE", "COUPLE", "PGATE", "READ", "BRA")
INSTRUCTION_BYTES = 8


def round_robin_matching(width: int, round_id: int) -> tuple[list[int], list[int]]:
    """One factor from a complete round-robin factorization of K_width."""
    if width % 2:
        raise ValueError("torus width must be even")
    ring = width - 1
    pivot = round_id % ring
    left = [ring]
    right = [pivot]
    for offset in range(1, width // 2):
        left.append((pivot - offset) % ring)
        right.append((pivot + offset) % ring)
    return left, right


class TorusNeuralCpu(nn.Module):
    def __init__(
        self,
        hidden: int,
        width: int,
        layers: int,
        bank_size: int = 14,
        period: int = 32,
        schedule_seed: int = 20260810,
        random_constants: bool = False,
    ):
        super().__init__()
        if width % 2 or bank_size > width // 16 or period > width - 1:
            raise ValueError("invalid torus geometry")
        self.hidden = hidden
        self.width = width
        self.bank_size = bank_size
        self.period = period
        self.encode = nn.Linear(hidden, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        self.layer_code = nn.Embedding(layers, width)
        self.angle = nn.Parameter(torch.zeros(bank_size))
        self.log_scale = nn.Parameter(torch.zeros(bank_size))
        nn.init.zeros_(self.decode.weight)
        if random_constants:
            nn.init.uniform_(self.angle, -math.pi, math.pi)
            nn.init.normal_(self.log_scale, std=0.1)

        normal_rounds = [(phase * 7) % (width - 1) for phase in range(period)]
        seeded = random.Random(schedule_seed)
        random_rounds = seeded.sample(range(width - 1), period)
        self._register_schedule("normal", normal_rounds)
        self._register_schedule("random", random_rounds)
        self._register_schedule("single", [0] * period)
        self.register_buffer(
            "bank_by_pair",
            torch.arange(width // 2, dtype=torch.int64) % bank_size,
            persistent=False,
        )

    def _register_schedule(self, name: str, rounds: list[int]) -> None:
        left, right = [], []
        for round_id in rounds:
            one_left, one_right = round_robin_matching(self.width, round_id)
            left.append(one_left)
            right.append(one_right)
        self.register_buffer(
            f"{name}_left", torch.tensor(left, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            f"{name}_right", torch.tensor(right, dtype=torch.int64), persistent=False
        )

    @property
    def program_bytes(self) -> int:
        # Five instruction words plus period, multiplier, and fixed beta
        # immediates already encoded in those words.
        return len(ISA) * INSTRUCTION_BYTES

    def component_bytes(self) -> dict[str, int]:
        encode_decode = (
            self.encode.weight.numel() + self.decode.weight.numel()
        ) * self.encode.weight.element_size()
        layer_code = self.layer_code.weight.numel() * self.layer_code.weight.element_size()
        bank = (self.angle.numel() + self.log_scale.numel()) * self.angle.element_size()
        return {
            "encode_decode": encode_decode,
            "layer_code": layer_code,
            "coupling_bank": bank,
            "program": self.program_bytes,
        }

    def _schedule(self, kind: str) -> tuple[Tensor, Tensor]:
        if kind not in {"normal", "random", "single"}:
            raise ValueError(f"unknown torus schedule {kind}")
        return getattr(self, f"{kind}_left"), getattr(self, f"{kind}_right")

    def execute(
        self,
        inputs: Tensor,
        layers: Tensor,
        maximum_ticks: int,
        checkpoints: tuple[int, ...],
        *,
        schedule: str = "normal",
        identity_bank: bool = False,
        identity_nonlinearity: bool = False,
        return_states: bool = False,
    ):
        wanted = set(checkpoints)
        state = torch.tanh(self.encode(inputs) + self.layer_code(layers))
        outputs: dict[int, Tensor] = {}
        states: dict[int, Tensor] = {}
        left_schedule, right_schedule = self._schedule(schedule)
        bank = self.bank_by_pair
        for tick in range(1, maximum_ticks + 1):
            phase = (tick - 1) % self.period
            left = left_schedule[phase]
            right = right_schedule[phase]
            a = state[:, left]
            b = state[:, right]
            if not identity_bank:
                angle = self.angle[bank]
                scale = 0.05 * torch.tanh(self.log_scale[bank])
                cosine, sine = torch.cos(angle), torch.sin(angle)
                expand, contract = torch.exp(scale), torch.exp(-scale)
                transformed_a = cosine * expand * a + sine * contract * b
                transformed_b = -sine * expand * a + cosine * contract * b
                a, b = transformed_a, transformed_b
            if not identity_nonlinearity:
                # Two additive coupling steps are exactly invertible and each
                # has a triangular unit-determinant Jacobian.
                a = a + 0.1 * torch.tanh(b)
                b = b + 0.1 * torch.tanh(a)
            next_state = torch.empty_like(state)
            next_state[:, left] = a
            next_state[:, right] = b
            state = next_state
            if tick in wanted:
                outputs[tick] = self.decode(state)
                if return_states:
                    states[tick] = state
        return (outputs, states) if return_states else outputs


def persistent_bytes(model: TorusNeuralCpu) -> int:
    return parameter_bytes(model) + model.program_bytes


def train(
    name: str,
    model: TorusNeuralCpu,
    data: TraceData,
    args: argparse.Namespace,
    checkpoints: tuple[int, ...],
) -> list[dict]:
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    generator = torch.Generator().manual_seed(args.seed)
    weights = torch.tensor(checkpoints, device=device, dtype=torch.float32)
    weights /= weights.sum()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        samples = 0
        started = time.perf_counter()
        for selected in iter_batches(
            data.split_indices["train"], args.batch_size, generator
        ):
            inputs = data.inputs[selected].to(device)
            targets = data.outputs[selected].to(device)
            layers = data.layers[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            with device_context(device):
                outputs = model.execute(
                    inputs, layers, max(checkpoints), checkpoints
                )
                loss = torch.sum(
                    weights
                    * torch.stack(
                        [normalized_mse(outputs[tick], targets) for tick in checkpoints]
                    )
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(selected)
            samples += len(selected)
        row = {
            "model": name,
            "epoch": epoch + 1,
            "train_nmse": total / samples,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


@torch.inference_mode()
def evaluate_sweep(
    model: TorusNeuralCpu,
    data: TraceData,
    indices: Tensor,
    args: argparse.Namespace,
    *,
    schedule: str = "normal",
    identity_bank: bool = False,
    identity_nonlinearity: bool = False,
    checkpoints: tuple[int, ...] | None = None,
    collect_states: bool = False,
) -> tuple[dict[str, dict], dict[int, Tensor]]:
    chosen = checkpoints or args.checkpoints
    device = torch.device(args.device)
    model.eval()
    accumulators = {
        tick: {"error": 0.0, "target": 0.0, "cosine": 0.0}
        for tick in chosen
    }
    state_blocks: dict[int, list[Tensor]] = {tick: [] for tick in chosen}
    started = time.perf_counter()
    for start in range(0, len(indices), args.evaluation_batch_size):
        selected = indices[start : start + args.evaluation_batch_size]
        inputs = data.inputs[selected].to(device)
        targets = data.outputs[selected].to(device)
        layers = data.layers[selected].to(device)
        with device_context(device):
            result = model.execute(
                inputs,
                layers,
                max(chosen),
                chosen,
                schedule=schedule,
                identity_bank=identity_bank,
                identity_nonlinearity=identity_nonlinearity,
                return_states=collect_states,
            )
            if collect_states:
                outputs, states = result
            else:
                outputs = result
            for tick in chosen:
                prediction = outputs[tick]
                accumulators[tick]["error"] += float(
                    (prediction - targets).float().square().sum()
                )
                accumulators[tick]["target"] += float(targets.float().square().sum())
                accumulators[tick]["cosine"] += float(
                    F.cosine_similarity(
                        prediction.float(), targets.float(), dim=-1
                    ).sum()
                )
                if collect_states:
                    state_blocks[tick].append(states[tick].float().cpu())
    seconds = time.perf_counter() - started
    metrics = {
        str(tick): {
            "nmse": values["error"] / values["target"],
            "cosine": values["cosine"] / len(indices),
            "samples": len(indices),
            "sweep_seconds": seconds,
        }
        for tick, values in accumulators.items()
    }
    return metrics, {
        tick: torch.cat(blocks) for tick, blocks in state_blocks.items() if blocks
    }


def effective_rank(states: Tensor) -> float:
    centered = states - states.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(states) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return float(eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12))


def determinant_audit(model: TorusNeuralCpu) -> dict[str, float]:
    with torch.no_grad():
        angle = model.angle.float().cpu()
        scale = 0.05 * torch.tanh(model.log_scale.float().cpu())
        cosine, sine = torch.cos(angle), torch.sin(angle)
        expand, contract = torch.exp(scale), torch.exp(-scale)
        determinants = (cosine * expand) * (cosine * contract) - (
            sine * contract
        ) * (-sine * expand)
    return {
        "maximum_abs_det_minus_one": float((determinants - 1).abs().max()),
        "theoretical_tick_log_abs_det": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--prompts-manifest", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--bank-size", type=int, default=14)
    parser.add_argument("--period", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--maximum-ticks", type=int, default=128)
    parser.add_argument("--checkpoints", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    args.checkpoints = tuple(int(value) for value in args.checkpoints.split(","))
    if max(args.checkpoints) != args.maximum_ticks:
        parser.error("checkpoints must end at maximum ticks")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    data = load_trace(args.trace, args.prompts_manifest)
    reference = json.loads(args.reference_result.read_text("utf-8"))
    if reference["trace_manifest_sha256"] != sha256_file(args.trace / "manifest.json"):
        raise RuntimeError("reference result belongs to another trace")
    hidden = data.inputs.shape[1]
    layers = len(data.layer_values)
    candidate = TorusNeuralCpu(
        hidden,
        args.width,
        layers,
        args.bank_size,
        args.period,
        args.seed,
    )
    reference_budget = int(reference["byte_audit"]["neural_cpu"]["persistent_bytes"])
    if persistent_bytes(candidate) > reference_budget:
        raise RuntimeError("torus candidate exceeds reference byte budget")
    random_control = TorusNeuralCpu(
        hidden,
        args.width,
        layers,
        args.bank_size,
        args.period,
        args.seed + 1,
        random_constants=True,
    )
    for parameter in random_control.parameters():
        parameter.requires_grad = False
    for parameter in random_control.decode.parameters():
        parameter.requires_grad = True

    histories = {
        "torus": train("torus", candidate, data, args, args.checkpoints),
        "random_constants_readout": train(
            "random_constants_readout",
            random_control,
            data,
            args,
            (args.maximum_ticks,),
        ),
    }
    metrics = {"validation": {}, "test": {}}
    test_states = {}
    for split in ("validation", "test"):
        metrics[split]["normal"], states = evaluate_sweep(
            candidate,
            data,
            data.split_indices[split],
            args,
            collect_states=split == "test",
        )
        if states:
            test_states = states
        for name, options in {
            "random_schedule": {"schedule": "random"},
            "single_stride": {"schedule": "single"},
            "identity_bank": {"identity_bank": True},
            "identity_nonlinearity": {"identity_nonlinearity": True},
        }.items():
            value, _ = evaluate_sweep(
                candidate,
                data,
                data.split_indices[split],
                args,
                checkpoints=(args.maximum_ticks,),
                **options,
            )
            metrics[split][name] = value[str(args.maximum_ticks)]
        value, _ = evaluate_sweep(
            random_control,
            data,
            data.split_indices[split],
            args,
            checkpoints=(args.maximum_ticks,),
        )
        metrics[split]["random_constants_readout"] = value[str(args.maximum_ticks)]

    rank_audit = {
        str(tick): effective_rank(states) for tick, states in test_states.items()
    }
    normal = metrics["test"]["normal"]
    first = normal[str(args.checkpoints[0])]["nmse"]
    at_32 = normal["32"]["nmse"]
    at_64 = normal["64"]["nmse"]
    final = normal[str(args.maximum_ticks)]["nmse"]
    shared = reference["metrics"]["test"]["shared_basis_core"]["nmse"]
    static_glu = reference["metrics"]["test"]["static_glu"]["nmse"]
    recurrent = reference["metrics"]["test"]["recurrent_no_isa"]["nmse"]
    checks = {
        "improves_32_to_64_by_5_percent": at_64 <= at_32 * 0.95,
        "improves_64_to_128_by_5_percent": final <= at_64 * 0.95,
        "beats_shared_basis": final < shared,
        "beats_static_glu": final < static_glu,
        "beats_recurrent_no_isa": final < recurrent,
        "algorithmic_dividend_25_percent": final <= first * 0.75,
        "identity_bank_lesion_20_percent": metrics["test"]["identity_bank"]["nmse"]
        >= final * 1.2,
        "identity_nonlinearity_lesion_20_percent": metrics["test"][
            "identity_nonlinearity"
        ]["nmse"]
        >= final * 1.2,
        "single_stride_lesion_20_percent": metrics["test"]["single_stride"]["nmse"]
        >= final * 1.2,
        "random_schedule_10x_worse": metrics["test"]["random_schedule"]["nmse"]
        >= final * 10.0,
        "random_constants_5x_worse": metrics["test"][
            "random_constants_readout"
        ]["nmse"]
        >= final * 5.0,
        "effective_rank_retains_80_percent": rank_audit[str(args.maximum_ticks)]
        >= rank_audit[str(args.checkpoints[0])] * 0.8,
    }
    checks["pilot_viable"] = all(checks.values())
    component_bytes = candidate.component_bytes()
    result = {
        "format": "quantum-llm-torus-result-v1",
        "classification": "tied-depth structured map; no data-dependent routing claim",
        "seed": args.seed,
        "trace_manifest_sha256": sha256_file(args.trace / "manifest.json"),
        "reference_result_sha256": sha256_file(args.reference_result),
        "records": len(data.inputs),
        "layers": data.layer_values,
        "program": ISA,
        "geometry": {
            "width": args.width,
            "bank_size": args.bank_size,
            "period": args.period,
            "beta": 0.1,
        },
        "bytes": {
            "components": component_bytes,
            "persistent": persistent_bytes(candidate),
            "reference_budget": reference_budget,
            "encode_decode_fraction": component_bytes["encode_decode"]
            / persistent_bytes(candidate),
        },
        "history": histories,
        "metrics": metrics,
        "reference_test_nmse": {
            "shared_basis_core": shared,
            "static_glu": static_glu,
            "recurrent_no_isa": recurrent,
        },
        "determinant_audit": determinant_audit(candidate),
        "effective_rank": rank_audit,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    torch.save(
        {
            "torus": candidate.state_dict(),
            "random_control": random_control.state_dict(),
        },
        args.output.with_suffix(".pt"),
    )
    print(json.dumps({"result": str(args.output), "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
