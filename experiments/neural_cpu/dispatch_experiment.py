"""Computed-dispatch Neural CPU redesign on an authenticated MoE trace.

The program counter follows a hard 16-way path.  Tree entries dispatch to a
small palette of shared low-rank subroutines, so the path carries conditional
function without storing a separate matrix at every leaf.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from experiment import (
    MappedTensor,
    TraceData,
    device_context,
    iter_batches,
    load_trace,
    normalized_mse,
    parameter_bytes,
    rms_norm,
    sha256_file,
)


ISA = ("DOT16", "JTABLE", "CALL", "LMAT_A", "SVGLU", "LMAT_B", "ADD", "RMSN", "RET", "BRA")
INSTRUCTION_BYTES = 8


class DispatchNeuralCpu(nn.Module):
    def __init__(
        self,
        hidden: int,
        width: int,
        layers: int,
        fanout: int = 16,
        operators: int = 16,
        rank: int = 8,
    ):
        super().__init__()
        if fanout < 2 or operators < 2:
            raise ValueError("dispatch geometry is too small")
        self.hidden = hidden
        self.width = width
        self.fanout = fanout
        self.operators = operators
        self.rank = rank
        self.nodes = 1 + fanout
        self.encode = nn.Linear(hidden, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        self.layer_code = nn.Embedding(layers, width)
        self.probes = nn.Parameter(torch.empty(self.nodes, fanout, width))
        self.thresholds = nn.Parameter(torch.zeros(self.nodes, fanout))
        self.operator_logits = nn.Parameter(
            torch.zeros(self.nodes, fanout, operators)
        )
        self.operator_down = nn.Parameter(torch.empty(operators, rank, width))
        self.operator_up = nn.Parameter(torch.empty(operators, width, rank))
        self.operator_scale = nn.Parameter(torch.zeros(operators))
        self.register_bias = nn.Parameter(torch.zeros(width))
        self.step_logit = nn.Parameter(torch.tensor(-1.0))
        nn.init.xavier_uniform_(self.probes)
        nn.init.xavier_uniform_(self.operator_down)
        nn.init.xavier_uniform_(self.operator_up)
        nn.init.normal_(self.operator_logits, std=0.02)
        nn.init.zeros_(self.decode.weight)

    @property
    def bytecode_bytes(self) -> int:
        return len(ISA) * INSTRUCTION_BYTES + self.nodes * self.fanout

    def hard_table(self) -> Tensor:
        return self.operator_logits.argmax(dim=-1)

    def execute(
        self,
        inputs: Tensor,
        layers: Tensor,
        maximum_ticks: int,
        checkpoints: tuple[int, ...],
        *,
        soft_dispatch: bool = False,
        table_override: Tensor | None = None,
        return_paths: bool = False,
    ):
        wanted = set(checkpoints)
        source = torch.tanh(self.encode(inputs) + self.layer_code(layers))
        state = source
        node = torch.zeros(len(inputs), dtype=torch.int64, device=inputs.device)
        outputs: dict[int, Tensor] = {}
        paths: list[Tensor] = []
        step = 0.5 * torch.sigmoid(self.step_logit)
        override = (
            table_override.to(inputs.device) if table_override is not None else None
        )
        for tick in range(1, maximum_ticks + 1):
            probes = self.probes[node]
            scores = torch.bmm(probes, state.unsqueeze(-1)).squeeze(-1)
            scores = scores + self.thresholds[node]
            branch_soft = torch.softmax(scores.float(), dim=-1).to(state.dtype)
            branch_index = branch_soft.argmax(dim=-1)
            branch_hard = F.one_hot(branch_index, self.fanout).to(state.dtype)
            if soft_dispatch:
                branch = branch_soft
            elif self.training:
                branch = branch_hard + branch_soft - branch_soft.detach()
            else:
                branch = branch_hard

            if override is None:
                table_soft = torch.softmax(
                    self.operator_logits[node].float(), dim=-1
                ).to(state.dtype)
                table_index = table_soft.argmax(dim=-1)
                table_hard = F.one_hot(table_index, self.operators).to(state.dtype)
                if soft_dispatch:
                    table = table_soft
                elif self.training:
                    table = table_hard + table_soft - table_soft.detach()
                else:
                    table = table_hard
            else:
                selected_table = override[node]
                table = F.one_hot(selected_table, self.operators).to(state.dtype)
            operator = torch.einsum("bf,bfo->bo", branch, table)
            selected_down = torch.einsum(
                "bo,orw->brw", operator, self.operator_down
            )
            scratch = torch.bmm(selected_down, state.unsqueeze(-1)).squeeze(-1)
            selected_up = torch.einsum(
                "bo,owr->bwr", operator, self.operator_up
            )
            delta = torch.bmm(
                selected_up, F.silu(scratch).unsqueeze(-1)
            ).squeeze(-1)
            scale = 2.0 * torch.sigmoid(
                operator @ self.operator_scale
            ).unsqueeze(-1)
            state = rms_norm(
                state + step * scale * delta + 0.02 * source + self.register_bias
            )
            if return_paths:
                paths.append(branch_index.detach())
            if tick in wanted:
                outputs[tick] = self.decode(state)

            # The first dispatch enters one of 16 second-level code blocks.  A
            # second dispatch returns to root, forming an 8-bit path per pair.
            if tick % 2:
                node = 1 + branch_index
            else:
                node = torch.zeros_like(node)
        if return_paths:
            return outputs, torch.stack(paths, dim=1)
        return outputs


def persistent_bytes(model: DispatchNeuralCpu) -> int:
    return parameter_bytes(model) + model.bytecode_bytes


def train_dispatch(
    name: str,
    model: DispatchNeuralCpu,
    data: TraceData,
    args: argparse.Namespace,
    *,
    table_override: Tensor | None = None,
) -> list[dict]:
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint_weights = torch.tensor(
        args.checkpoints, device=device, dtype=torch.float32
    )
    checkpoint_weights /= checkpoint_weights.sum()
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
                    inputs,
                    layers,
                    args.maximum_ticks,
                    args.checkpoints,
                    table_override=table_override,
                )
                loss = torch.sum(
                    checkpoint_weights
                    * torch.stack(
                        [
                            normalized_mse(outputs[tick], targets)
                            for tick in args.checkpoints
                        ]
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
def evaluate(
    model: DispatchNeuralCpu,
    data: TraceData,
    indices: Tensor,
    args: argparse.Namespace,
    *,
    tick: int,
    soft_dispatch: bool = False,
    table_override: Tensor | None = None,
    collect_paths: bool = False,
):
    device = torch.device(args.device)
    model.eval()
    squared_error = squared_target = cosine = 0.0
    path_blocks = []
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
                tick,
                (tick,),
                soft_dispatch=soft_dispatch,
                table_override=table_override,
                return_paths=collect_paths,
            )
            if collect_paths:
                outputs, paths = result
                path_blocks.append(paths.cpu())
            else:
                outputs = result
            prediction = outputs[tick]
        squared_error += float((prediction - targets).float().square().sum())
        squared_target += float(targets.float().square().sum())
        cosine += float(
            F.cosine_similarity(prediction.float(), targets.float(), dim=-1).sum()
        )
    metrics = {
        "nmse": squared_error / squared_target,
        "cosine": cosine / len(indices),
        "samples": len(indices),
        "seconds": time.perf_counter() - started,
    }
    return metrics, (torch.cat(path_blocks) if path_blocks else None)


def information_audit(paths: Tensor, teacher_top1: Tensor, fanout: int) -> dict:
    if paths.shape[1] < 2:
        codes = paths[:, 0]
    else:
        codes = paths[:, 0] * fanout + paths[:, 1]
    pairs = collections.Counter(zip(codes.tolist(), teacher_top1.tolist()))
    path_counts = collections.Counter(codes.tolist())
    route_counts = collections.Counter(teacher_top1.tolist())
    total = len(codes)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in path_counts.values()
    )
    mutual_information = 0.0
    for (path, route), count in pairs.items():
        probability = count / total
        mutual_information += probability * math.log2(
            probability
            / ((path_counts[path] / total) * (route_counts[route] / total))
        )
    return {
        "path_entropy_bits": entropy,
        "path_route_mutual_information_bits": mutual_information,
        "unique_two_tick_paths": len(path_counts),
        "maximum_two_tick_paths": fanout * fanout,
        "most_common_path_fraction": max(path_counts.values()) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--prompts-manifest", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=211)
    parser.add_argument("--fanout", type=int, default=16)
    parser.add_argument("--operators", type=int, default=16)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--maximum-ticks", type=int, default=32)
    parser.add_argument("--checkpoints", default="1,2,4,8,16,32")
    parser.add_argument("--seed", type=int, default=20260809)
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
    candidate = DispatchNeuralCpu(
        hidden,
        args.width,
        layers,
        args.fanout,
        args.operators,
        args.rank,
    )
    reference_budget = int(
        reference["byte_audit"]["neural_cpu"]["persistent_bytes"]
    )
    if persistent_bytes(candidate) > reference_budget:
        raise RuntimeError("dispatch candidate exceeds reference byte budget")

    random_control = DispatchNeuralCpu(
        hidden,
        args.width,
        layers,
        args.fanout,
        args.operators,
        args.rank,
    )
    for parameter in random_control.parameters():
        parameter.requires_grad = False
    for parameter in random_control.decode.parameters():
        parameter.requires_grad = True
    random_generator = torch.Generator().manual_seed(args.seed + 1)
    random_table = torch.randint(
        args.operators,
        (random_control.nodes, args.fanout),
        generator=random_generator,
    )

    histories = {
        "computed_dispatch": train_dispatch(
            "computed_dispatch", candidate, data, args
        ),
        "random_bytecode_readout": train_dispatch(
            "random_bytecode_readout",
            random_control,
            data,
            args,
            table_override=random_table,
        ),
    }
    hard_table = candidate.hard_table().cpu()
    shuffled_table = hard_table.flatten().roll(1).reshape_as(hard_table)
    metrics = {"validation": {}, "test": {}}
    test_paths = None
    for split in ("validation", "test"):
        for tick in args.checkpoints:
            metric, paths = evaluate(
                candidate,
                data,
                data.split_indices[split],
                args,
                tick=tick,
                collect_paths=split == "test" and tick == args.maximum_ticks,
            )
            metrics[split][str(tick)] = metric
            if paths is not None:
                test_paths = paths
        metrics[split]["soft"], _ = evaluate(
            candidate,
            data,
            data.split_indices[split],
            args,
            tick=args.maximum_ticks,
            soft_dispatch=True,
        )
        metrics[split]["shuffled_table"], _ = evaluate(
            candidate,
            data,
            data.split_indices[split],
            args,
            tick=args.maximum_ticks,
            table_override=shuffled_table,
        )
        metrics[split]["random_bytecode_readout"], _ = evaluate(
            random_control,
            data,
            data.split_indices[split],
            args,
            tick=args.maximum_ticks,
            table_override=random_table,
        )

    route_map = MappedTensor(
        args.trace / "route-indices.u32",
        torch.int32,
        (len(data.inputs), int(json.loads((args.trace / "manifest.json").read_text())["top_k"])),
    )
    teacher_top1 = route_map.tensor[data.split_indices["test"], 0].to(torch.int64)
    path_audit = information_audit(test_paths, teacher_top1, args.fanout)
    route_map.close()

    final = metrics["test"][str(args.maximum_ticks)]["nmse"]
    first = metrics["test"][str(args.checkpoints[0])]["nmse"]
    at_16 = metrics["test"].get("16", metrics["test"][str(args.checkpoints[-2])])["nmse"]
    shared = reference["metrics"]["test"]["shared_basis_core"]["nmse"]
    recurrent = reference["metrics"]["test"]["recurrent_no_isa"]["nmse"]
    shuffled = metrics["test"]["shuffled_table"]["nmse"]
    random_nmse = metrics["test"]["random_bytecode_readout"]["nmse"]
    soft = metrics["test"]["soft"]["nmse"]
    checks = {
        "time_for_space_improves": final <= first * 0.8,
        "still_improves_16_to_32": final <= at_16 * 0.98,
        "beats_shared_basis": final < shared,
        "beats_recurrent_no_isa": final < recurrent,
        "shuffled_table_10x_worse": shuffled >= final * 10.0,
        "random_bytecode_5x_worse": random_nmse >= final * 5.0,
        "hardening_gap_below_20_percent": final <= soft * 1.2,
        "path_entropy_above_4_bits": path_audit["path_entropy_bits"] >= 4.0,
        "path_route_mi_above_0_1_bits": path_audit[
            "path_route_mutual_information_bits"
        ] >= 0.1,
    }
    checks["pilot_viable"] = all(checks.values())
    result = {
        "format": "quantum-llm-computed-dispatch-result-v1",
        "seed": args.seed,
        "trace_manifest_sha256": sha256_file(args.trace / "manifest.json"),
        "reference_result_sha256": sha256_file(args.reference_result),
        "records": len(data.inputs),
        "layers": data.layer_values,
        "program": ISA,
        "geometry": {
            "width": args.width,
            "fanout": args.fanout,
            "tree_nodes": candidate.nodes,
            "operators": args.operators,
            "rank": args.rank,
        },
        "bytes": {
            "parameters": parameter_bytes(candidate),
            "bytecode_and_hard_table": candidate.bytecode_bytes,
            "persistent": persistent_bytes(candidate),
            "reference_budget": reference_budget,
        },
        "history": histories,
        "metrics": metrics,
        "reference_test_nmse": {
            "shared_basis_core": shared,
            "recurrent_no_isa": recurrent,
        },
        "path_audit": path_audit,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    torch.save(
        {
            "computed_dispatch": candidate.state_dict(),
            "random_control": random_control.state_dict(),
            "random_table": random_table,
            "hard_table": hard_table,
        },
        args.output.with_suffix(".pt"),
    )
    print(json.dumps({"result": str(args.output), "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
