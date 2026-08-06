"""Immediate NCPU-0 screen for the literal Neural CPU hypothesis.

The exact Qwen expert is read directly from Expert Pack v1 and used only as a
teacher.  The candidate is a deterministic hard program whose six instruction
words are executed repeatedly over a fixed register file.  Its constants do
not change as the tick count grows.

This screen answers one narrow question: does additional execution time buy a
better approximation at fixed persistent bytes than a static byte-matched
surrogate?  It does not claim full-model quality.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


EXPERT_MAGIC = b"EPEXPR01"
EXPERT_HEADER = struct.Struct("<8sHHIIiiIIIIQQQQQQQQQ32s")
CHECKPOINTS = (1, 2, 4, 8, 16, 32)
PROGRAM = ("MIX_A", "MIX_B", "GLU", "GATE", "ADD", "NORM")
SHUFFLED_PROGRAM = ("GLU", "GATE", "ADD", "NORM", "MIX_A", "MIX_B")


@dataclass(frozen=True)
class ExpertWeights:
    gate: Tensor
    up: Tensor
    down: Tensor
    stored_bytes: int


def _tensor_from_bytes(raw: bytes, dtype: torch.dtype) -> Tensor:
    # Clone because torch.frombuffer otherwise retains a read-only Python
    # buffer and emits a warning about writable aliases.
    return torch.frombuffer(bytearray(raw), dtype=dtype).clone()


def load_expert(container: Path, layer: int, expert: int) -> ExpertWeights:
    manifest = json.loads((container / "manifest.json").read_text("utf-8"))
    matches = [
        item
        for item in manifest["experts"]
        if int(item["layer"]) == layer and int(item["expert"]) == expert
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one expert record, found {len(matches)}")
    entry = matches[0]
    pack = container / str(entry["pack"])
    offset = int(entry["offset"])
    stored_bytes = int(entry["stored_bytes"])
    with pack.open("rb") as handle:
        handle.seek(offset)
        header_raw = handle.read(EXPERT_HEADER.size)
        if len(header_raw) != EXPERT_HEADER.size:
            raise RuntimeError("truncated expert header")
        values = EXPERT_HEADER.unpack(header_raw)
        (
            magic,
            version,
            header_bytes,
            _flags,
            _quant_abi,
            record_layer,
            record_expert,
            hidden,
            intermediate,
            fused_rows,
            _reserved,
            record_bytes,
            gate_up_q_offset,
            gate_up_q_bytes,
            gate_up_scale_offset,
            gate_up_scale_bytes,
            down_q_offset,
            down_q_bytes,
            down_scale_offset,
            down_scale_bytes,
            _payload_hash,
        ) = values
        if (
            magic != EXPERT_MAGIC
            or version != 1
            or header_bytes != 256
            or record_layer != layer
            or record_expert != expert
            or record_bytes != stored_bytes
            or fused_rows != 2 * intermediate
        ):
            raise RuntimeError("expert record identity or geometry mismatch")

        def read_section(relative: int, count: int) -> bytes:
            handle.seek(offset + relative)
            payload = handle.read(count)
            if len(payload) != count:
                raise RuntimeError("truncated expert section")
            return payload

        gate_up_q = _tensor_from_bytes(
            read_section(gate_up_q_offset, gate_up_q_bytes), torch.int8
        ).reshape(2 * intermediate, hidden)
        gate_up_scale = _tensor_from_bytes(
            read_section(gate_up_scale_offset, gate_up_scale_bytes), torch.float32
        ).reshape(2 * intermediate, 1)
        down_q = _tensor_from_bytes(
            read_section(down_q_offset, down_q_bytes), torch.int8
        ).reshape(hidden, intermediate)
        down_scale = _tensor_from_bytes(
            read_section(down_scale_offset, down_scale_bytes), torch.float32
        ).reshape(hidden, 1)

    gate_up = gate_up_q.float() * gate_up_scale
    down = down_q.float() * down_scale
    return ExpertWeights(
        gate=gate_up[:intermediate].contiguous(),
        up=gate_up[intermediate:].contiguous(),
        down=down.contiguous(),
        stored_bytes=stored_bytes,
    )


@torch.inference_mode()
def teacher_outputs(weights: ExpertWeights, inputs: Tensor, batch: int) -> Tensor:
    outputs: list[Tensor] = []
    for start in range(0, inputs.shape[0], batch):
        x = inputs[start : start + batch]
        gate = F.linear(x, weights.gate)
        up = F.linear(x, weights.up)
        outputs.append(F.linear(F.silu(gate) * up, weights.down).cpu())
    return torch.cat(outputs)


def rms_norm(value: Tensor) -> Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)


class NeuralCpu(nn.Module):
    """Hard register program with one bounded backwards branch."""

    def __init__(self, hidden: int, width: int):
        super().__init__()
        self.encode = nn.Linear(hidden, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        self.a_h = nn.Linear(width, width, bias=False)
        self.a_u = nn.Linear(width, width, bias=False)
        self.b_h = nn.Linear(width, width, bias=False)
        self.b_u = nn.Linear(width, width, bias=False)
        self.g_h = nn.Linear(width, width, bias=False)
        self.g_u = nn.Linear(width, width, bias=False)
        self.register_bias = nn.Parameter(torch.zeros(width))

    def execute(
        self,
        x: Tensor,
        maximum_ticks: int,
        program: tuple[str, ...] = PROGRAM,
        checkpoints: Iterable[int] = CHECKPOINTS,
    ) -> dict[int, Tensor]:
        wanted = set(checkpoints)
        u = torch.tanh(self.encode(x))
        registers = {
            "H": torch.zeros_like(u),
            "A": torch.zeros_like(u),
            "B": torch.zeros_like(u),
            "D": torch.zeros_like(u),
            "G": torch.zeros_like(u),
        }
        outputs: dict[int, Tensor] = {}
        pc = 0
        tick = 0
        while tick < maximum_ticks:
            opcode = program[pc]
            if opcode == "MIX_A":
                registers["A"] = self.a_h(registers["H"]) + self.a_u(u)
            elif opcode == "MIX_B":
                registers["B"] = self.b_h(registers["H"]) + self.b_u(u)
            elif opcode == "GLU":
                registers["D"] = F.silu(registers["A"]) * registers["B"]
            elif opcode == "GATE":
                registers["G"] = torch.sigmoid(
                    self.g_h(registers["H"]) + self.g_u(u)
                )
                registers["D"] = registers["D"] * registers["G"]
            elif opcode == "ADD":
                registers["H"] = registers["H"] + registers["D"]
            elif opcode == "NORM":
                registers["H"] = rms_norm(registers["H"] + self.register_bias)
            else:  # pragma: no cover - program is a frozen contract
                raise RuntimeError(f"illegal opcode {opcode}")
            pc += 1
            if pc == len(program):
                pc = 0
                tick += 1
                if tick in wanted:
                    outputs[tick] = self.decode(registers["H"])
        return outputs

    def forward(self, x: Tensor, maximum_ticks: int = 32) -> Tensor:
        return self.execute(x, maximum_ticks, checkpoints=(maximum_ticks,))[
            maximum_ticks
        ]


class StaticSurrogate(nn.Module):
    """Byte-comparable static function with no parameter reuse over time."""

    def __init__(self, hidden: int, width: int):
        super().__init__()
        self.encode_a = nn.Linear(hidden, width, bias=False)
        self.encode_b = nn.Linear(hidden, width, bias=False)
        self.hidden_a = nn.Linear(width, width, bias=False)
        self.hidden_b = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x: Tensor) -> Tensor:
        u = torch.tanh(self.encode_a(x))
        v = torch.tanh(self.encode_b(x))
        value = F.silu(self.hidden_a(u)) * self.hidden_b(v)
        value = value * torch.sigmoid(self.gate(u))
        return self.decode(rms_norm(value + self.bias))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def normalized_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().mean() / target.square().mean().clamp_min(1e-9)


def cosine_mean(prediction: Tensor, target: Tensor) -> Tensor:
    return F.cosine_similarity(prediction, target, dim=-1).mean()


def batches(indices: Tensor, batch_size: int, generator: torch.Generator):
    order = indices[
        torch.randperm(
            indices.numel(), generator=generator, device=indices.device
        )
    ]
    for start in range(0, order.numel(), batch_size):
        yield order[start : start + batch_size]


def train_neural_cpu(
    model: NeuralCpu,
    x: Tensor,
    y: Tensor,
    train_idx: Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    checkpoint_weights = {1: 0.02, 2: 0.03, 4: 0.05, 8: 0.10, 16: 0.25, 32: 0.55}
    completed = 0
    model.train()
    while completed < steps:
        for index in batches(train_idx, batch_size, generator):
            prediction = model.execute(x[index], 32)
            loss = sum(
                checkpoint_weights[tick] * normalized_mse(output, y[index])
                for tick, output in prediction.items()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed += 1
            if completed >= steps:
                break


def train_static(
    model: StaticSurrogate,
    x: Tensor,
    y: Tensor,
    train_idx: Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    completed = 0
    model.train()
    while completed < steps:
        for index in batches(train_idx, batch_size, generator):
            loss = normalized_mse(model(x[index]), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed += 1
            if completed >= steps:
                break


@torch.inference_mode()
def evaluate(
    neural_cpu: NeuralCpu,
    static: StaticSurrogate,
    x: Tensor,
    y: Tensor,
    index: Tensor,
) -> dict[str, object]:
    neural_cpu.eval()
    static.eval()
    cpu_outputs = neural_cpu.execute(x[index], 32)
    sweep = {
        str(tick): {
            "normalized_mse": float(normalized_mse(output, y[index])),
            "cosine": float(cosine_mean(output, y[index])),
        }
        for tick, output in cpu_outputs.items()
    }
    static_output = static(x[index])
    shuffled = neural_cpu.execute(
        x[index], 32, program=SHUFFLED_PROGRAM, checkpoints=(32,)
    )[32]
    return {
        "tick_sweep": sweep,
        "static": {
            "normalized_mse": float(normalized_mse(static_output, y[index])),
            "cosine": float(cosine_mean(static_output, y[index])),
        },
        "shuffled_program": {
            "normalized_mse": float(normalized_mse(shuffled, y[index])),
            "cosine": float(cosine_mean(shuffled, y[index])),
        },
    }


@torch.inference_mode()
def benchmark(model: NeuralCpu, device: torch.device, hidden: int, repeats: int) -> float:
    sample = rms_norm(torch.randn(1, hidden, device=device))
    for _ in range(5):
        model(sample, 32)
    started = time.perf_counter()
    for _ in range(repeats):
        model(sample, 32)
    return (time.perf_counter() - started) * 1e6 / repeats


def decision(metrics: dict[str, object]) -> tuple[str, list[str]]:
    sweep = metrics["tick_sweep"]
    e4 = float(sweep["4"]["normalized_mse"])
    e16 = float(sweep["16"]["normalized_mse"])
    e32 = float(sweep["32"]["normalized_mse"])
    static = float(metrics["static"]["normalized_mse"])
    shuffled = float(metrics["shuffled_program"]["normalized_mse"])
    checks = {
        "time_improves_4_to_32": e32 <= 0.80 * e4,
        "time_still_improves_16_to_32": e32 <= 0.98 * e16,
        "beats_static_by_10_percent": e32 <= 0.90 * static,
        "instruction_order_matters": shuffled >= 1.20 * e32,
    }
    return ("go" if all(checks.values()) else "kill", [
        f"{name}={str(value).lower()}" for name, value in checks.items()
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    weights = load_expert(args.container, args.layer, args.expert)
    hidden = weights.gate.shape[1]

    data_generator = torch.Generator().manual_seed(args.seed)
    inputs = rms_norm(torch.randn(args.samples, hidden, generator=data_generator))
    targets = teacher_outputs(weights, inputs, args.teacher_batch)
    permutation = torch.randperm(args.samples, generator=data_generator)
    train_count = int(args.samples * 0.75)
    validation_count = int(args.samples * 0.125)
    train_idx = permutation[:train_count]
    validation_idx = permutation[train_count : train_count + validation_count]
    test_idx = permutation[train_count + validation_count :]

    inputs = inputs.to(device)
    targets = targets.to(device)
    train_idx = train_idx.to(device)
    validation_idx = validation_idx.to(device)
    test_idx = test_idx.to(device)

    neural_cpu = NeuralCpu(hidden, args.width).to(device)
    static = StaticSurrogate(hidden, args.width).to(device)
    training_generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    started = time.perf_counter()
    train_neural_cpu(
        neural_cpu,
        inputs,
        targets,
        train_idx,
        args.train_steps,
        args.batch_size,
        args.learning_rate,
        training_generator,
    )
    cpu_train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    train_static(
        static,
        inputs,
        targets,
        train_idx,
        args.train_steps,
        args.batch_size,
        args.learning_rate,
        training_generator,
    )
    static_train_seconds = time.perf_counter() - started

    validation = evaluate(neural_cpu, static, inputs, targets, validation_idx)
    test = evaluate(neural_cpu, static, inputs, targets, test_idx)
    verdict, checks = decision(test)
    cpu_parameters = parameter_count(neural_cpu)
    static_parameters = parameter_count(static)
    result = {
        "schema": "neural-cpu-screen-v1",
        "scope": "one real Qwen routed expert on deterministic RMS-normalized synthetic activations",
        "teacher": {
            "layer": args.layer,
            "expert": args.expert,
            "hidden": hidden,
            "intermediate": int(weights.gate.shape[0]),
            "record_bytes": weights.stored_bytes,
        },
        "data": {
            "seed": args.seed,
            "samples": args.samples,
            "train": int(train_idx.numel()),
            "validation": int(validation_idx.numel()),
            "test": int(test_idx.numel()),
        },
        "machine": {
            "register_width": args.width,
            "program": list(PROGRAM),
            "program_bytes": len(PROGRAM) * 8,
            "parameters": cpu_parameters,
            "fp32_persistent_bytes": cpu_parameters * 4 + len(PROGRAM) * 8,
            "ticks": list(CHECKPOINTS),
        },
        "baseline": {
            "parameters": static_parameters,
            "fp32_persistent_bytes": static_parameters * 4,
            "parameter_ratio_cpu_to_static": cpu_parameters / static_parameters,
        },
        "training": {
            "steps_per_model": args.train_steps,
            "neural_cpu_seconds": cpu_train_seconds,
            "static_seconds": static_train_seconds,
        },
        "validation": validation,
        "test": test,
        "neural_cpu_batch1_microseconds_32_ticks_eager": benchmark(
            neural_cpu, device, hidden, 20
        ),
        "decision": verdict,
        "decision_checks": checks,
        "interpretation": (
            "GO only green-lights the time-for-space mechanism for a single expert; "
            "KILL rejects this hard ISA/cell, not every possible Neural CPU."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if verdict == "go" else 2


if __name__ == "__main__":
    raise SystemExit(main())
