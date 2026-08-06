"""Fast structural self-test for the trace loader and candidate models."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from pathlib import Path

import torch

from experiment import (
    RANDOM_PROGRAM,
    SHUFFLED_PROGRAM,
    IndependentLowRank,
    LiteralNeuralCpu,
    RecurrentNoIsa,
    SharedBasisCore,
    StaticGlu,
    load_trace,
    normalized_mse,
)
from dispatch_experiment import DispatchNeuralCpu


def write_values(path: Path, code: str, values) -> dict:
    payload = struct.pack("<" + code * len(values), *values)
    path.write_bytes(payload)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        trace = root / "trace"
        trace.mkdir()
        records, hidden, top_k = 12, 16, 2
        inputs = torch.randn(records, hidden)
        targets = torch.tanh(inputs) * 0.2
        layers = [20 + index % 4 for index in range(records)]
        sequences = [index // 4 for index in range(records)]
        positions = [index % 4 for index in range(records)]
        route_indices = [index % 8 for index in range(records * top_k)]
        route_scores = [0.5] * (records * top_k)
        files = {
            "input.f32": write_values(trace / "input.f32", "f", inputs.flatten().tolist()),
            "output.f32": write_values(trace / "output.f32", "f", targets.flatten().tolist()),
            "layer.u32": write_values(trace / "layer.u32", "I", layers),
            "sequence.u32": write_values(trace / "sequence.u32", "I", sequences),
            "position.u32": write_values(trace / "position.u32", "I", positions),
            "route-indices.u32": write_values(
                trace / "route-indices.u32", "I", route_indices
            ),
            "route-scores.f32": write_values(
                trace / "route-scores.f32", "f", route_scores
            ),
        }
        (trace / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "quantum-llm-moe-trace-v1",
                    "record_count": records,
                    "hidden_size": hidden,
                    "top_k": top_k,
                    "layers": [20, 21, 22, 23],
                    "files": files,
                }
            ),
            "utf-8",
        )
        prompts = root / "prompts.json"
        prompts.write_text(
            json.dumps(
                {
                    "sequences": [
                        {"sequence_id": 0, "split": "train"},
                        {"sequence_id": 1, "split": "validation"},
                        {"sequence_id": 2, "split": "test"},
                    ]
                }
            ),
            "utf-8",
        )
        data = load_trace(trace, prompts)
        candidate = LiteralNeuralCpu(hidden, 8, 4)
        outputs = candidate.execute(data.inputs, data.layers, 2, (1, 2))
        candidate.execute(
            data.inputs, data.layers, 2, (2,), SHUFFLED_PROGRAM
        )
        candidate.execute(data.inputs, data.layers, 2, (2,), RANDOM_PROGRAM)
        dispatch = DispatchNeuralCpu(hidden, 8, 4, fanout=4, operators=4, rank=2)
        dispatch_outputs, dispatch_paths = dispatch.execute(
            data.inputs,
            data.layers,
            2,
            (1, 2),
            return_paths=True,
        )
        if dispatch_paths.shape != (records, 2):
            raise RuntimeError("invalid dispatch path geometry")
        shuffled_table = dispatch.hard_table().flatten().roll(1).reshape_as(
            dispatch.hard_table()
        )
        dispatch.execute(
            data.inputs,
            data.layers,
            2,
            (2,),
            table_override=shuffled_table,
        )
        loss = normalized_mse(outputs[1], data.outputs) + normalized_mse(
            outputs[2], data.outputs
        )
        loss.backward()
        models = [
            StaticGlu(hidden, 8, 4),
            IndependentLowRank(hidden, 3, 4),
            SharedBasisCore(hidden, 8, 4),
            RecurrentNoIsa(hidden, 8, 4),
        ]
        for model in models:
            if isinstance(model, RecurrentNoIsa):
                prediction = model(data.inputs, data.layers, 2)
            else:
                prediction = model(data.inputs, data.layers)
            if prediction.shape != data.outputs.shape or not torch.isfinite(prediction).all():
                raise RuntimeError(f"invalid self-test output from {type(model).__name__}")
        result_loss = float(loss.detach())
        print(
            json.dumps(
                {
                    "status": "pass",
                    "records": records,
                    "splits": {
                        name: len(indices) for name, indices in data.split_indices.items()
                    },
                    "candidate_loss": result_loss,
                }
            )
        )
        del prediction, outputs, loss, models, candidate, dispatch_outputs
        del dispatch_paths, dispatch
        data.close()


if __name__ == "__main__":
    main()
