from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from pow import MemoryExpert, continuation_candidates
from synthetic_memory import (
    HashingEmbedder,
    ShardedVectorIndex,
    build_corpus,
    retrieve_adaptive,
)


def main() -> int:
    records, examples = build_corpus(train_worlds=8, eval_worlds=4)
    assert len({record.record_id for record in records}) == len(records)
    assert {example.kind for example in examples} == {
        "single-hop", "two-hop", "unknown", "counterfactual"
    }
    counterfactual = [example for example in examples if example.kind == "counterfactual"]
    assert counterfactual
    assert any(
        left.question == right.question and
        left.citations == right.citations and
        left.answer != right.answer
        for left in counterfactual for right in counterfactual
    )

    index_records = [record for record in records if record.indexable]
    embedder = HashingEmbedder(256).fit(record.text for record in index_records)
    index = ShardedVectorIndex(
        index_records, embedder.encode(record.text for record in index_records), 5
    )
    known = [
        example for example in examples
        if example.kind in ("single-hop", "two-hop")
    ]
    hits = 0
    misses = []
    known_root_scores = []
    for example in known:
        selected = retrieve_adaptive(index, embedder, example.question, maximum=2)
        unfiltered = retrieve_adaptive(
            index, embedder, example.question, maximum=1, score_floor=0.0
        )
        known_root_scores.append(unfiltered[0][1])
        selected_ids = {record.record_id for record, _ in selected}
        hits += int(set(example.memory_ids).issubset(selected_ids))
        if not set(example.memory_ids).issubset(selected_ids):
            misses.append({
                "example": example.example_id,
                "expected": example.memory_ids,
                "selected": tuple(selected_ids),
            })
    print(json.dumps({
        "event": "retrieval-diagnostic",
        "known_examples": len(known),
        "recall": hits / len(known),
        "misses": misses,
    }), flush=True)
    assert hits / len(known) >= 0.70
    unknown = [example for example in examples if example.kind == "unknown"]
    unknown_root_scores = [
        retrieve_adaptive(
            index, embedder, example.question, maximum=1, score_floor=0.0
        )[0][1]
        for example in unknown
    ]
    print(json.dumps({
        "event": "retrieval-calibration",
        "known_minimum_root_score": min(known_root_scores),
        "unknown_maximum_root_score": max(unknown_root_scores),
    }), flush=True)
    assert all(
        not retrieve_adaptive(index, embedder, example.question, maximum=2)
        for example in unknown
    )

    with tempfile.TemporaryDirectory() as temporary:
        manifest_path = index.write(Path(temporary))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["record_count"] == len(index_records)
        assert len(manifest["shards"]) > 1
        disk_index = ShardedVectorIndex.open(Path(temporary), maximum_resident_bytes=0)
        assert disk_index.storage_mode == "mmap-sharded"
        for example in known:
            memory_selected = retrieve_adaptive(
                index, embedder, example.question, maximum=2
            )
            disk_selected = retrieve_adaptive(
                disk_index, embedder, example.question, maximum=2
            )
            assert [record.record_id for record, _ in disk_selected] == [
                record.record_id for record, _ in memory_selected
            ]

    attention = SimpleNamespace(
        head_dim=8,
        config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=2),
        q_proj=nn.Linear(32, 32, bias=False),
        k_proj=nn.Linear(32, 16, bias=False),
        v_proj=nn.Linear(32, 16, bias=False),
        o_proj=nn.Linear(32, 32, bias=False),
    )
    adapter = MemoryExpert(hidden_size=32, rank=4, alpha=8.0, dropout=0.0)
    input_norm = nn.RMSNorm(32)
    hidden = torch.randn(2, 3, 32)
    memory = torch.randn(2, 5, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    output = adapter(hidden, memory, mask, attention, input_norm)
    assert torch.equal(output, hidden)
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert adapter.gate_up.weight.grad is not None
    assert torch.isfinite(output).all()

    tokenizer = SimpleNamespace(
        decode=lambda tokens, skip_special_tokens=True: "".join(
            chr(token) for token in tokens
        )
    )
    source = [ord(character) for character in "code NOVA-9526 end"]
    generated = [ord(character) for character in "NOVA-"]
    assert continuation_candidates(generated, source, tokenizer) == {ord("9")}

    print(json.dumps({
        "records": len(records),
        "examples": len(examples),
        "retrieval_recall": hits / len(known),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "passed": True,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
