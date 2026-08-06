"""Train and evaluate a literal hard-program Neural CPU on real MoE traces.

This is deliberately self-contained: it validates authenticated trace files,
enforces conversation-level splits, derives byte-matched baseline widths, and
emits a machine-readable result.  The teacher checkpoint is never opened.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import mmap
import random
import struct
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROGRAM = (
    "CMP", "BNZ", "LMAT_A", "SVGLU", "LMAT_B", "SCALE", "MIX", "MARK",
    "RMSN", "BRA",
)
SHUFFLED_PROGRAM = (
    "MARK", "LMAT_A", "CMP", "MIX", "RMSN", "LMAT_B", "BNZ", "SCALE",
    "SVGLU", "BRA",
)
_random_opcodes = list(PROGRAM[:-1])
random.Random(20260806).shuffle(_random_opcodes)
RANDOM_PROGRAM = tuple(_random_opcodes) + ("BRA",)
INSTRUCTION_BYTES = 8
OPCODES = {
    "CMP": 1,
    "BNZ": 2,
    "LMAT_A": 3,
    "SVGLU": 4,
    "LMAT_B": 5,
    "SCALE": 6,
    "MIX": 7,
    "MARK": 8,
    "RMSN": 9,
    "BRA": 10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def encode_program(program: tuple[str, ...]) -> bytes:
    # Opcode occupies the high six bits. Operand fields are zero in this first
    # fixed-register ABI; BNZ's bounded target is the beginning of the program.
    return b"".join(struct.pack("<Q", OPCODES[opcode] << 58) for opcode in program)


class MappedTensor:
    def __init__(self, path: Path, dtype: torch.dtype, shape: tuple[int, ...]):
        self.handle = path.open("rb")
        self.mapping = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_COPY)
        self.tensor = torch.frombuffer(self.mapping, dtype=dtype).reshape(shape)

    def close(self) -> None:
        self.tensor = torch.empty(0)
        gc.collect()
        self.mapping.close()
        self.handle.close()


@dataclass
class TraceData:
    inputs: Tensor
    outputs: Tensor
    layers: Tensor
    sequences: Tensor
    positions: Tensor
    layer_values: tuple[int, ...]
    split_indices: dict[str, Tensor]
    mappings: list[MappedTensor]

    def close(self) -> None:
        self.inputs = torch.empty(0)
        self.outputs = torch.empty(0)
        gc.collect()
        for mapping in self.mappings:
            mapping.close()
        self.mappings.clear()


def load_trace(trace_root: Path, prompts_manifest: Path) -> TraceData:
    manifest = json.loads((trace_root / "manifest.json").read_text("utf-8"))
    if manifest.get("format") != "quantum-llm-moe-trace-v1":
        raise RuntimeError("unsupported trace format")
    records = int(manifest["record_count"])
    hidden = int(manifest["hidden_size"])
    top_k = int(manifest["top_k"])
    if records <= 0 or hidden <= 0 or top_k <= 0:
        raise RuntimeError("invalid trace geometry")
    expected = {
        "input.f32": records * hidden * 4,
        "output.f32": records * hidden * 4,
        "layer.u32": records * 4,
        "sequence.u32": records * 4,
        "position.u32": records * 4,
        "route-indices.u32": records * top_k * 4,
        "route-scores.f32": records * top_k * 4,
    }
    for name, size in expected.items():
        path = trace_root / name
        entry = manifest["files"][name]
        if path.stat().st_size != size or int(entry["bytes"]) != size:
            raise RuntimeError(f"trace size mismatch: {name}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"trace digest mismatch: {name}")

    prompt_info = json.loads(prompts_manifest.read_text("utf-8"))
    sequence_splits = {
        int(item["sequence_id"]): str(item["split"])
        for item in prompt_info["sequences"]
    }
    mappings = [
        MappedTensor(trace_root / "input.f32", torch.float32, (records, hidden)),
        MappedTensor(trace_root / "output.f32", torch.float32, (records, hidden)),
        MappedTensor(trace_root / "layer.u32", torch.int32, (records,)),
        MappedTensor(trace_root / "sequence.u32", torch.int32, (records,)),
        MappedTensor(trace_root / "position.u32", torch.int32, (records,)),
    ]
    layers = mappings[2].tensor.to(torch.int64)
    sequences = mappings[3].tensor.to(torch.int64)
    layer_values = tuple(sorted(int(value) for value in torch.unique(layers)))
    layer_lookup = {value: index for index, value in enumerate(layer_values)}
    mapped_layers = torch.tensor(
        [layer_lookup[int(value)] for value in layers], dtype=torch.int64
    )
    split_indices: dict[str, Tensor] = {}
    for split in ("train", "validation", "test"):
        allowed = {
            sequence for sequence, value in sequence_splits.items() if value == split
        }
        split_indices[split] = torch.tensor(
            [index for index, value in enumerate(sequences.tolist()) if value in allowed],
            dtype=torch.int64,
        )
        if not len(split_indices[split]):
            raise RuntimeError(f"trace has no records for split {split}")
    return TraceData(
        mappings[0].tensor,
        mappings[1].tensor,
        mapped_layers,
        sequences,
        mappings[4].tensor.to(torch.int64),
        layer_values,
        split_indices,
        mappings,
    )


def rms_norm(value: Tensor) -> Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)


class LiteralNeuralCpu(nn.Module):
    """Hard branch-and-accumulate machine with a counted constant pool."""

    def __init__(
        self, hidden: int, width: int, layers: int, atoms: int = 8, rank: int = 16
    ):
        super().__init__()
        self.hidden = hidden
        self.width = width
        self.atoms = atoms
        self.rank = rank
        self.layer_code = nn.Embedding(layers, width)
        self.encode = nn.Linear(hidden, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        self.controller = nn.Linear(width, atoms, bias=False)
        self.atom_down = nn.Parameter(torch.empty(atoms, rank, width))
        self.atom_up = nn.Parameter(torch.empty(atoms, width, rank))
        self.atom_scale = nn.Parameter(torch.zeros(atoms))
        self.feedback = nn.Linear(width, width, bias=False)
        self.output_gate = nn.Linear(width, 1, bias=False)
        self.register_bias = nn.Parameter(torch.zeros(width))
        self.output_step_logit = nn.Parameter(torch.tensor(-1.5))
        nn.init.xavier_uniform_(self.atom_down)
        nn.init.xavier_uniform_(self.atom_up)
        nn.init.zeros_(self.decode.weight)

    def execute(
        self,
        inputs: Tensor,
        layers: Tensor,
        maximum_cycles: int,
        checkpoints: Iterable[int],
        program: tuple[str, ...] = PROGRAM,
        soft_branches: bool = False,
    ) -> dict[int, Tensor]:
        wanted = set(checkpoints)
        u = torch.tanh(self.encode(inputs) + self.layer_code(layers))
        registers = {
            "H": u,
            "Y": torch.zeros_like(u),
            "A": torch.zeros(
                inputs.shape[0], self.rank, device=inputs.device, dtype=u.dtype
            ),
            "D": torch.zeros_like(u),
            "LOGITS": torch.zeros(
                inputs.shape[0], self.atoms, device=inputs.device, dtype=u.dtype
            ),
            "SELECT": torch.zeros(
                inputs.shape[0], self.atoms, device=inputs.device, dtype=u.dtype
            ),
        }
        used = torch.zeros(
            inputs.shape[0], self.atoms, device=inputs.device, dtype=torch.bool
        )
        results: dict[int, Tensor] = {}
        output_step = torch.sigmoid(self.output_step_logit)
        pc = 0
        cycle = 0
        while cycle < maximum_cycles:
            opcode = program[pc]
            if opcode == "CMP":
                logits = self.controller(registers["H"])
                registers["LOGITS"] = logits.masked_fill(used, -1.0e4)
            elif opcode == "BNZ":
                soft = torch.softmax(registers["LOGITS"].float(), dim=-1).to(u.dtype)
                hard_index = soft.argmax(dim=-1)
                hard = F.one_hot(hard_index, self.atoms).to(u.dtype)
                if soft_branches:
                    registers["SELECT"] = soft
                elif self.training:
                    registers["SELECT"] = hard + soft - soft.detach()
                else:
                    registers["SELECT"] = hard
            elif opcode == "LMAT_A":
                selected = torch.einsum(
                    "bk,krw->brw", registers["SELECT"], self.atom_down
                )
                registers["A"] = torch.bmm(
                    selected, registers["H"].unsqueeze(-1)
                ).squeeze(-1)
            elif opcode == "SVGLU":
                registers["A"] = F.silu(registers["A"])
            elif opcode == "LMAT_B":
                selected = torch.einsum(
                    "bk,kwr->bwr", registers["SELECT"], self.atom_up
                )
                registers["D"] = torch.bmm(
                    selected, registers["A"].unsqueeze(-1)
                ).squeeze(-1)
            elif opcode == "SCALE":
                scale = 2.0 * torch.sigmoid(
                    registers["SELECT"] @ self.atom_scale
                ).unsqueeze(-1)
                registers["D"] = registers["D"] * scale
            elif opcode == "MIX":
                gate = torch.sigmoid(self.output_gate(registers["H"]))
                registers["Y"] = registers["Y"] + output_step * gate * (
                    registers["D"] - registers["Y"]
                )
            elif opcode == "MARK":
                selected_index = registers["SELECT"].argmax(dim=-1, keepdim=True)
                used = used.scatter(1, selected_index, True)
            elif opcode == "RMSN":
                registers["H"] = rms_norm(
                    u + self.feedback(registers["Y"]) + self.register_bias
                )
            elif opcode == "BRA":
                cycle += 1
                exhausted = used.all(dim=-1)
                if exhausted.any():
                    used = used & ~exhausted.unsqueeze(-1)
                if cycle in wanted:
                    results[cycle] = self.decode(registers["Y"])
            else:
                raise RuntimeError(f"illegal hard opcode {opcode}")
            pc = (pc + 1) % len(program)
        return results

    def forward(self, inputs: Tensor, layers: Tensor, cycles: int) -> Tensor:
        return self.execute(inputs, layers, cycles, (cycles,))[cycles]

    @property
    def program_bytes(self) -> int:
        return len(PROGRAM) * INSTRUCTION_BYTES


class StaticGlu(nn.Module):
    def __init__(self, hidden: int, width: int, layers: int):
        super().__init__()
        self.gate = nn.Linear(hidden, width, bias=False)
        self.up = nn.Linear(hidden, width, bias=False)
        self.down = nn.Linear(width, hidden, bias=False)
        self.gate_code = nn.Embedding(layers, width)
        self.up_code = nn.Embedding(layers, width)
        nn.init.zeros_(self.down.weight)

    def forward(self, inputs: Tensor, layers: Tensor) -> Tensor:
        return self.down(
            F.silu(self.gate(inputs) + self.gate_code(layers))
            * (self.up(inputs) + self.up_code(layers))
        )


class IndependentLowRank(nn.Module):
    def __init__(self, hidden: int, rank: int, layers: int):
        super().__init__()
        self.left = nn.Parameter(torch.empty(layers, rank, hidden))
        self.right = nn.Parameter(torch.empty(layers, hidden, rank))
        nn.init.xavier_uniform_(self.left)
        nn.init.zeros_(self.right)

    def forward(self, inputs: Tensor, layers: Tensor) -> Tensor:
        left = self.left[layers]
        right = self.right[layers]
        code = torch.bmm(left, inputs.unsqueeze(-1)).squeeze(-1)
        return torch.bmm(right, torch.tanh(code).unsqueeze(-1)).squeeze(-1)


class SharedBasisCore(nn.Module):
    def __init__(self, hidden: int, width: int, layers: int):
        super().__init__()
        self.encode = nn.Linear(hidden, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        self.cores = nn.Parameter(torch.empty(layers, width, width))
        nn.init.orthogonal_(self.cores)
        nn.init.zeros_(self.decode.weight)

    def forward(self, inputs: Tensor, layers: Tensor) -> Tensor:
        code = torch.tanh(self.encode(inputs))
        code = torch.bmm(self.cores[layers], code.unsqueeze(-1)).squeeze(-1)
        return self.decode(F.silu(code))


class RecurrentNoIsa(nn.Module):
    def __init__(self, hidden: int, width: int, layers: int):
        super().__init__()
        self.encode = nn.Linear(hidden, width, bias=False)
        self.decode = nn.Linear(width, hidden, bias=False)
        self.input_update = nn.Linear(width, width, bias=False)
        self.state_update = nn.Linear(width, width, bias=False)
        self.layer_code = nn.Embedding(layers, width)
        nn.init.zeros_(self.decode.weight)

    def forward(self, inputs: Tensor, layers: Tensor, cycles: int) -> Tensor:
        code = torch.tanh(self.encode(inputs) + self.layer_code(layers))
        state = code
        projected = self.input_update(code)
        for _ in range(cycles):
            state = rms_norm(state + 0.1 * torch.tanh(projected + self.state_update(state)))
        return self.decode(state)


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def largest_width(factory: Callable[[int], nn.Module], budget: int) -> tuple[nn.Module, int]:
    low, high = 1, 1
    while parameter_bytes(factory(high)) <= budget:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if parameter_bytes(factory(middle)) <= budget:
            low = middle
        else:
            high = middle
    return factory(low), low


def normalized_mse(prediction: Tensor, target: Tensor) -> Tensor:
    numerator = (prediction - target).square().mean(dim=-1)
    denominator = target.square().mean(dim=-1).clamp_min(1e-6)
    return (numerator / denominator).mean()


def iter_batches(indices: Tensor, batch_size: int, generator: torch.Generator):
    order = indices[torch.randperm(len(indices), generator=generator)]
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def device_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_model(
    name: str,
    model: nn.Module,
    data: TraceData,
    args: argparse.Namespace,
    recurrent: bool = False,
    neural_cpu: bool = False,
    program: tuple[str, ...] = PROGRAM,
) -> list[dict]:
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        samples = 0
        started = time.perf_counter()
        for batch_indices in iter_batches(
            data.split_indices["train"], args.batch_size, generator
        ):
            inputs = data.inputs[batch_indices].to(device, non_blocking=True)
            targets = data.outputs[batch_indices].to(device, non_blocking=True)
            layers = data.layers[batch_indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with device_context(device):
                if neural_cpu:
                    outputs = model.execute(
                        inputs,
                        layers,
                        args.maximum_cycles,
                        args.checkpoints,
                        program,
                    )
                    weights = torch.tensor(
                        args.checkpoints, device=inputs.device, dtype=torch.float32
                    )
                    weights = weights / weights.sum()
                    loss = torch.sum(
                        weights
                        * torch.stack(
                            [
                                normalized_mse(outputs[cycle], targets)
                                for cycle in args.checkpoints
                            ]
                        )
                    )
                elif recurrent:
                    loss = normalized_mse(
                        model(inputs, layers, args.maximum_cycles), targets
                    )
                else:
                    loss = normalized_mse(model(inputs, layers), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch_indices)
            total += float(loss.detach()) * count
            samples += count
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
    model: nn.Module,
    data: TraceData,
    indices: Tensor,
    args: argparse.Namespace,
    *,
    cycles: int | None = None,
    program: tuple[str, ...] | None = None,
    recurrent: bool = False,
    soft_branches: bool = False,
) -> dict[str, float]:
    device = torch.device(args.device)
    model.eval()
    squared_error = squared_target = cosine = 0.0
    samples = 0
    started = time.perf_counter()
    for start in range(0, len(indices), args.evaluation_batch_size):
        selected = indices[start : start + args.evaluation_batch_size]
        inputs = data.inputs[selected].to(device)
        targets = data.outputs[selected].to(device)
        layers = data.layers[selected].to(device)
        with device_context(device):
            if isinstance(model, LiteralNeuralCpu):
                chosen = cycles or args.maximum_cycles
                prediction = model.execute(
                    inputs,
                    layers,
                    chosen,
                    (chosen,),
                    program or PROGRAM,
                    soft_branches,
                )[chosen]
            elif recurrent:
                prediction = model(inputs, layers, cycles or args.maximum_cycles)
            else:
                prediction = model(inputs, layers)
        squared_error += float((prediction - targets).float().square().sum())
        squared_target += float(targets.float().square().sum())
        cosine += float(
            F.cosine_similarity(prediction.float(), targets.float(), dim=-1).sum()
        )
        samples += len(selected)
    return {
        "nmse": squared_error / max(squared_target, 1e-12),
        "cosine": cosine / samples,
        "samples": samples,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--prompts-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=216)
    parser.add_argument("--atoms", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--maximum-cycles", type=int, default=32)
    parser.add_argument("--checkpoints", default="1,2,4,8,16,32")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    args.checkpoints = tuple(int(value) for value in args.checkpoints.split(","))
    if not args.checkpoints or max(args.checkpoints) != args.maximum_cycles:
        parser.error("checkpoints must end at --maximum-cycles")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data = load_trace(args.trace, args.prompts_manifest)
    hidden = data.inputs.shape[1]
    layers = len(data.layer_values)
    candidate = LiteralNeuralCpu(
        hidden, args.width, layers, atoms=args.atoms, rank=args.rank
    )
    budget = parameter_bytes(candidate) + candidate.program_bytes
    static, static_width = largest_width(
        lambda width: StaticGlu(hidden, width, layers), budget
    )
    independent, independent_rank = largest_width(
        lambda rank: IndependentLowRank(hidden, rank, layers), budget
    )
    shared, shared_width = largest_width(
        lambda width: SharedBasisCore(hidden, width, layers), budget
    )
    recurrent, recurrent_width = largest_width(
        lambda width: RecurrentNoIsa(hidden, width, layers), budget
    )
    random_readout = LiteralNeuralCpu(
        hidden, args.width, layers, atoms=args.atoms, rank=args.rank
    )
    for parameter in random_readout.parameters():
        parameter.requires_grad = False
    for parameter in random_readout.decode.parameters():
        parameter.requires_grad = True
    models = {
        "neural_cpu": (
            candidate,
            {"width": args.width, "atoms": args.atoms, "rank": args.rank},
        ),
        "static_glu": (static, {"width": static_width}),
        "independent_low_rank": (independent, {"rank": independent_rank}),
        "shared_basis_core": (shared, {"width": shared_width}),
        "recurrent_no_isa": (recurrent, {"width": recurrent_width}),
        "random_program_readout": (
            random_readout,
            {"width": args.width, "atoms": args.atoms, "rank": args.rank},
        ),
    }
    byte_audit = {
        name: {
            **geometry,
            "parameter_bytes": parameter_bytes(model),
            "program_bytes": (
                candidate.program_bytes
                if name in {"neural_cpu", "random_program_readout"}
                else 0
            ),
            "persistent_bytes": parameter_bytes(model)
            + (
                candidate.program_bytes
                if name in {"neural_cpu", "random_program_readout"}
                else 0
            ),
        }
        for name, (model, geometry) in models.items()
    }
    print(json.dumps({"byte_audit": byte_audit}), flush=True)

    histories = {}
    for name, (model, _) in models.items():
        histories[name] = train_model(
            name,
            model,
            data,
            args,
            recurrent=name == "recurrent_no_isa",
            neural_cpu=name in {"neural_cpu", "random_program_readout"},
            program=RANDOM_PROGRAM if name == "random_program_readout" else PROGRAM,
        )

    metrics: dict[str, dict] = {}
    for split in ("validation", "test"):
        metrics[split] = {}
        for name, (model, _) in models.items():
            if name == "neural_cpu":
                metrics[split][name] = {
                    str(cycle): evaluate(
                        model, data, data.split_indices[split], args, cycles=cycle
                    )
                    for cycle in args.checkpoints
                }
                metrics[split]["neural_cpu_shuffled"] = evaluate(
                    model,
                    data,
                    data.split_indices[split],
                    args,
                    cycles=args.maximum_cycles,
                    program=SHUFFLED_PROGRAM,
                )
                metrics[split]["neural_cpu_soft"] = evaluate(
                    model,
                    data,
                    data.split_indices[split],
                    args,
                    cycles=args.maximum_cycles,
                    soft_branches=True,
                )
            elif name == "random_program_readout":
                metrics[split][name] = evaluate(
                    model,
                    data,
                    data.split_indices[split],
                    args,
                    cycles=args.maximum_cycles,
                    program=RANDOM_PROGRAM,
                )
            else:
                metrics[split][name] = evaluate(
                    model,
                    data,
                    data.split_indices[split],
                    args,
                    recurrent=name == "recurrent_no_isa",
                )

    test_ticks = metrics["test"]["neural_cpu"]
    first = test_ticks[str(args.checkpoints[0])]["nmse"]
    final = test_ticks[str(args.checkpoints[-1])]["nmse"]
    static_nmse = metrics["test"]["static_glu"]["nmse"]
    recurrent_nmse = metrics["test"]["recurrent_no_isa"]["nmse"]
    shuffled_nmse = metrics["test"]["neural_cpu_shuffled"]["nmse"]
    random_nmse = metrics["test"]["random_program_readout"]["nmse"]
    soft_nmse = metrics["test"]["neural_cpu_soft"]["nmse"]
    verdict = {
        "time_for_space_improves": final < first * 0.8,
        "beats_static_same_bytes": final < static_nmse,
        "beats_recurrent_no_isa": final < recurrent_nmse,
        "program_order_matters": shuffled_nmse >= final * 10.0,
        "beats_random_program_readout": random_nmse >= final * 5.0,
        "hardening_gap_below_20_percent": final <= soft_nmse * 1.2,
    }
    verdict["pilot_viable"] = all(verdict.values())
    result = {
        "format": "quantum-llm-neural-cpu-result-v1",
        "seed": args.seed,
        "trace_manifest_sha256": sha256_file(args.trace / "manifest.json"),
        "prompt_manifest_sha256": sha256_file(args.prompts_manifest),
        "records": len(data.inputs),
        "layers": data.layer_values,
        "checkpoints_cycles": args.checkpoints,
        "instructions_per_cycle": len(PROGRAM),
        "hard_bytecode_hex": encode_program(PROGRAM).hex(),
        "hard_bytecode_sha256": hashlib.sha256(encode_program(PROGRAM)).hexdigest(),
        "byte_audit": byte_audit,
        "history": histories,
        "metrics": metrics,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    torch.save(
        {name: model.state_dict() for name, (model, _) in models.items()},
        args.output.with_suffix(".pt"),
    )
    print(json.dumps({"result": str(args.output), "verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
