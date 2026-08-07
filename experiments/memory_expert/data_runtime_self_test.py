from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

try:
    from .dense_index import build_index, search
    from .ingest import ingest
except ImportError:
    from dense_index import build_index, search
    from ingest import ingest


class FakeEncoder:
    model_id = "self-test/fake"
    revision = "0" * 40
    maximum_tokens = 32

    @staticmethod
    def encode(texts: list[str], batch_size: int = 2) -> np.ndarray:
        del batch_size
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vector = np.array([
                float("alpha" in lowered),
                float("beta" in lowered),
                0.25,
            ], dtype=np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="memory-data-runtime-test-") as directory:
        root = Path(directory)
        source = root / "source.txt"
        config = root / "config.json"
        source.write_text("alpha authority\n\nbeta authority", encoding="utf-8")
        config.write_text(json.dumps({
            "schema_version": 1,
            "source": {"format": "text"},
            "defaults": {
                "document_id": "runtime-test",
                "namespace": "test",
                "acl": ["principal-a"],
            },
            "segmentation": {
                "maximum_chars": 128,
                "overlap_paragraphs": 0,
            },
            "output": {"records_per_shard": 1},
        }), encoding="utf-8")
        ingest_root = root / "ingest"
        index_root = root / "index"
        ingest(source, config, ingest_root, "test://runtime")
        manifest = build_index(ingest_root, index_root, FakeEncoder(), batch_size=2)
        assert manifest["record_count"] == 1
        query = np.array([1.0, 0.0, 0.25], dtype=np.float32)
        query /= np.linalg.norm(query)
        hits = search(
            ingest_root, index_root, query, maximum=1,
            allowed_acl={"principal-a"},
        )
        assert len(hits) == 1
        assert "alpha authority" in hits[0].record.text
        assert not search(
            ingest_root, index_root, query, maximum=1,
            allowed_acl={"principal-b"},
        )
    print("memory-data runtime self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
