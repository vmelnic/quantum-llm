from __future__ import annotations

import json
import tempfile
from pathlib import Path

try:
    from .data_contract import read_records
    from .ingest import ingest
except ImportError:
    from data_contract import read_records
    from ingest import ingest


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="memory-data-test-") as directory:
        root = Path(directory)
        source = root / "source.json"
        config = root / "config.json"
        source.write_text(json.dumps({
            "id": "doc-7",
            "lang": "ro",
            "revision": 3,
            "body": (
                "Secțiunea 1. Alfa\n\nPrimul fapt este autoritativ.\n\n"
                "Secțiunea 2. Beta\n\nAl doilea fapt este verificabil."
            ),
        }, ensure_ascii=False), encoding="utf-8")
        config.write_text(json.dumps({
            "schema_version": 1,
            "source": {"format": "json", "text_pointer": "/body"},
            "identity": {
                "document_id_pointer": "/id",
                "language_pointer": "/lang",
                "generation_pointer": "/revision",
            },
            "defaults": {"namespace": "test", "acl": ["tenant-a"]},
            "segmentation": {
                "section_kind": "section",
                "section_pattern": "^(?P<label>Secțiunea [0-9]+\\.[^\\n]*)$",
                "maximum_chars": 128,
                "overlap_paragraphs": 0,
            },
            "output": {"records_per_shard": 1},
        }, ensure_ascii=False), encoding="utf-8")
        first = root / "first"
        second = root / "second"
        first_manifest = ingest(source, config, first, "test://document/7")
        second_manifest = ingest(source, config, second, "test://document/7")
        first_records = list(read_records(first))
        second_records = list(read_records(second))
        assert len(first_records) == 2
        assert first_manifest["records_fingerprint"] == second_manifest["records_fingerprint"]
        assert [row.record_id for row in first_records] == [row.record_id for row in second_records]
        assert all(row.source_uri == "test://document/7" for row in first_records)
        assert all(row.acl == ("tenant-a",) for row in first_records)
        assert all(row.document_title == "doc-7" for row in first_records)
        assert all(row.attributes == {} for row in first_records)
        source_text = json.loads(source.read_text(encoding="utf-8"))["body"]
        for record in first_records:
            span = record.source_span
            assert source_text[span.character_start:span.character_end] == record.text
            assert record.language == "ro"
            assert record.memory_text.startswith(f"Memory record {record.citation_id}.")
    print("memory-data self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
