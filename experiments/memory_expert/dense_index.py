from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

try:
    from .data_contract import KnowledgeRecord, iter_record_shards, read_records
except ImportError:  # Direct execution on a worker.
    from data_contract import KnowledgeRecord, iter_record_shards, read_records


@dataclass(frozen=True)
class SearchHit:
    record: KnowledgeRecord
    score: float


class TransformerDenseEncoder:
    """Generic normalized CLS encoder backed by a pinned HF checkpoint."""

    def __init__(self, model_id: str, revision: str, device: torch.device,
                 maximum_tokens: int = 1024) -> None:
        if maximum_tokens < 8:
            raise ValueError("maximum_tokens must be at least 8")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.maximum_tokens = maximum_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, local_files_only=True,
            dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                list(texts[start:start + batch_size]),
                padding=True,
                truncation=True,
                max_length=self.maximum_tokens,
                return_tensors="pt",
            ).to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                output = self.model(**batch, return_dict=True)
            # BGE-M3's published SentenceTransformers configuration uses the
            # first/CLS token followed by L2 normalization.
            dense = functional.normalize(output.last_hidden_state[:, 0].float(), p=2, dim=1)
            rows.append(dense.cpu().numpy())
        if not rows:
            dimension = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, dimension), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def build_index(ingest_root: Path, output: Path, encoder: TransformerDenseEncoder,
                batch_size: int = 16) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"index output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    started = time.perf_counter()
    shards: list[dict[str, object]] = []
    record_count = 0
    dimension = 0
    try:
        for descriptor, record_path in iter_record_shards(ingest_root):
            records = [
                KnowledgeRecord.from_dict(json.loads(line))
                for line in record_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            encoded_at = time.perf_counter()
            vectors = encoder.encode(
                [record.retrieval_text for record in records], batch_size=batch_size
            )
            dimension = int(vectors.shape[1]) if vectors.ndim == 2 else dimension
            vector_name = f"vectors-{int(descriptor['shard_id']):05d}.npy"
            vector_path = temporary / vector_name
            np.save(vector_path, vectors.astype(np.float16), allow_pickle=False)
            shards.append({
                "shard_id": int(descriptor["shard_id"]),
                "record_count": len(records),
                "record_file": record_path.name,
                "record_sha256": descriptor["sha256"],
                "vectors": vector_name,
                "vector_sha256": _file_sha256(vector_path),
                "encode_seconds": time.perf_counter() - encoded_at,
            })
            record_count += len(records)
        ingest_manifest = json.loads(
            (ingest_root / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = {
            "schema_version": 1,
            "contract": "quantum-llm-dense-index-v1",
            "ingest_records_fingerprint": ingest_manifest["records_fingerprint"],
            "encoder": {
                "model_id": encoder.model_id,
                "revision": encoder.revision,
                "pooling": "cls",
                "normalization": "l2",
                "maximum_tokens": encoder.maximum_tokens,
                "dimension": dimension,
                "vector_dtype": "float16",
            },
            "record_count": record_count,
            "shards": shards,
            "timing_seconds": {"total": time.perf_counter() - started},
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_record_shard(path: Path) -> list[KnowledgeRecord]:
    return [
        KnowledgeRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def search(ingest_root: Path, index_root: Path, query_vector: np.ndarray,
           maximum: int = 4, allowed_acl: set[str] | None = None,
           language: str | None = None) -> list[SearchHit]:
    if maximum < 1:
        return []
    manifest = json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))
    ingest_manifest = json.loads((ingest_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("ingest_records_fingerprint") != ingest_manifest.get("records_fingerprint"):
        raise ValueError("dense index does not match the ingest generation")
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    candidates: list[SearchHit] = []
    for shard in manifest["shards"]:
        vectors = np.load(index_root / str(shard["vectors"]), mmap_mode="r")
        records = _read_record_shard(ingest_root / str(shard["record_file"]))
        if len(records) != len(vectors):
            raise ValueError("record/vector shard size mismatch")
        scores = np.asarray(vectors, dtype=np.float32) @ query
        local_count = min(maximum, len(scores))
        if local_count == 0:
            continue
        positions = np.argpartition(-scores, local_count - 1)[:local_count]
        for position in positions:
            record = records[int(position)]
            if allowed_acl is not None and not allowed_acl.intersection(record.acl):
                continue
            if language is not None and record.language != language:
                continue
            candidates.append(SearchHit(record, float(scores[int(position)])))
    candidates.sort(key=lambda hit: hit.score, reverse=True)
    return candidates[:maximum]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a generic sharded dense index")
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    encoder = TransformerDenseEncoder(
        args.model, args.revision, torch.device(args.device), args.maximum_tokens
    )
    result = build_index(
        args.ingest.resolve(), args.output.resolve(), encoder, args.batch_size
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
