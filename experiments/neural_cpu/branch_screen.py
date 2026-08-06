"""NCPU-0b screen for the hard branch-and-accumulate machine.

The teacher is one exact packed Qwen expert. Inputs are synthetic, so this is a
mechanism screen only; the real-activation experiment remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from experiment import (
    PROGRAM,
    RANDOM_PROGRAM,
    SHUFFLED_PROGRAM,
    LiteralNeuralCpu,
    StaticGlu,
    largest_width,
    normalized_mse,
    parameter_bytes,
    rms_norm,
)
from screen import load_expert, teacher_outputs


def batches(count: int, size: int, generator: torch.Generator, device: torch.device):
    order = torch.randperm(count, generator=generator, device=device)
    for start in range(0, count, size):
        yield order[start : start + size]


def train_candidate(
    model: LiteralNeuralCpu,
    inputs: Tensor,
    targets: Tensor,
    layers: Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    checkpoints: tuple[int, ...],
    program: tuple[str, ...] = PROGRAM,
) -> float:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    generator = torch.Generator(device=inputs.device).manual_seed(20260807)
    completed = 0
    started = time.perf_counter()
    model.train()
    while completed < steps:
        for selected in batches(len(inputs), batch_size, generator, inputs.device):
            outputs = model.execute(
                inputs[selected], layers[selected], checkpoints[-1], checkpoints, program
            )
            weights = torch.tensor(
                checkpoints, device=inputs.device, dtype=torch.float32
            )
            weights = weights / weights.sum()
            loss = torch.sum(
                weights
                * torch.stack(
                    [
                        normalized_mse(outputs[cycle], targets[selected])
                        for cycle in checkpoints
                    ]
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed += 1
            if completed == steps:
                break
    return time.perf_counter() - started


def train_static(
    model: StaticGlu,
    inputs: Tensor,
    targets: Tensor,
    layers: Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=inputs.device).manual_seed(20260808)
    completed = 0
    started = time.perf_counter()
    model.train()
    while completed < steps:
        for selected in batches(len(inputs), batch_size, generator, inputs.device):
            loss = normalized_mse(model(inputs[selected], layers[selected]), targets[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            completed += 1
            if completed == steps:
                break
    return time.perf_counter() - started


@torch.inference_mode()
def metric(prediction: Tensor, target: Tensor) -> dict[str, float]:
    return {
        "nmse": float(normalized_mse(prediction, target)),
        "cosine": float(
            torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--width", type=int, default=216)
    parser.add_argument("--atoms", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--checkpoints", default="1,2,4,8,16,32")
    args = parser.parse_args()
    checkpoints = tuple(int(value) for value in args.checkpoints.split(","))
    torch.manual_seed(20260806)
    device = torch.device("cuda")
    weights = load_expert(args.container, args.layer, args.expert)
    hidden = weights.gate.shape[1]
    source = rms_norm(torch.randn(args.samples, hidden))
    target = teacher_outputs(weights, source, 64)
    order = torch.randperm(args.samples)
    train_count = args.samples * 3 // 4
    validation_count = args.samples // 8
    train = order[:train_count]
    validation = order[train_count : train_count + validation_count]
    test = order[train_count + validation_count :]
    source, target = source.to(device), target.to(device)
    layers = torch.zeros(args.samples, dtype=torch.int64, device=device)

    candidate = LiteralNeuralCpu(
        hidden, args.width, 1, atoms=args.atoms, rank=args.rank
    ).to(device)
    budget = parameter_bytes(candidate) + candidate.program_bytes
    static, static_width = largest_width(
        lambda width: StaticGlu(hidden, width, 1), budget
    )
    static = static.to(device)
    random_readout = LiteralNeuralCpu(
        hidden, args.width, 1, atoms=args.atoms, rank=args.rank
    ).to(device)
    for parameter in random_readout.parameters():
        parameter.requires_grad = False
    for parameter in random_readout.decode.parameters():
        parameter.requires_grad = True

    candidate_seconds = train_candidate(
        candidate,
        source[train],
        target[train],
        layers[train],
        args.steps,
        args.batch_size,
        args.learning_rate,
        checkpoints,
    )
    static_seconds = train_static(
        static,
        source[train],
        target[train],
        layers[train],
        args.steps,
        args.batch_size,
        args.learning_rate,
    )
    random_seconds = train_candidate(
        random_readout,
        source[train],
        target[train],
        layers[train],
        args.steps,
        args.batch_size,
        args.learning_rate,
        (checkpoints[-1],),
        RANDOM_PROGRAM,
    )

    candidate.eval()
    static.eval()
    random_readout.eval()
    def evaluate(selected: Tensor) -> dict:
        outputs = candidate.execute(
            source[selected], layers[selected], checkpoints[-1], checkpoints
        )
        shuffled = candidate.execute(
            source[selected],
            layers[selected],
            checkpoints[-1],
            (checkpoints[-1],),
            SHUFFLED_PROGRAM,
        )[checkpoints[-1]]
        soft = candidate.execute(
            source[selected],
            layers[selected],
            checkpoints[-1],
            (checkpoints[-1],),
            PROGRAM,
            True,
        )[checkpoints[-1]]
        random_output = random_readout.execute(
            source[selected],
            layers[selected],
            checkpoints[-1],
            (checkpoints[-1],),
            RANDOM_PROGRAM,
        )[checkpoints[-1]]
        return {
            "ticks": {
                str(cycle): metric(output, target[selected])
                for cycle, output in outputs.items()
            },
            "static": metric(static(source[selected], layers[selected]), target[selected]),
            "shuffled": metric(shuffled, target[selected]),
            "soft": metric(soft, target[selected]),
            "random_program_readout": metric(random_output, target[selected]),
        }

    result = {
        "format": "quantum-llm-neural-cpu-branch-screen-v1",
        "scope": "one exact Qwen expert with synthetic activations",
        "teacher": {"layer": args.layer, "expert": args.expert},
        "program": list(PROGRAM),
        "checkpoints": checkpoints,
        "bytes": {
            "neural_cpu": budget,
            "static": parameter_bytes(static),
            "random_program_readout": parameter_bytes(random_readout)
            + random_readout.program_bytes,
            "static_width": static_width,
        },
        "training_seconds": {
            "neural_cpu": candidate_seconds,
            "static": static_seconds,
            "random_program_readout": random_seconds,
        },
        "validation": evaluate(validation),
        "test": evaluate(test),
    }
    ticks = result["test"]["ticks"]
    first, final = ticks[str(checkpoints[0])]["nmse"], ticks[str(checkpoints[-1])]["nmse"]
    result["checks"] = {
        "time_improves": final < first * 0.8,
        "beats_static": final < result["test"]["static"]["nmse"],
        "order_matters": result["test"]["shuffled"]["nmse"] >= final * 10,
        "beats_random": result["test"]["random_program_readout"]["nmse"] >= final * 5,
        "hardening_gap": final <= result["test"]["soft"]["nmse"] * 1.2,
    }
    result["go"] = all(result["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
