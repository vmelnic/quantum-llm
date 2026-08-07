from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class SourceSpan:
    character_start: int
    character_end: int


@dataclass(frozen=True)
class KnowledgeRecord:
    """Model-independent unit exchanged by ingestion, retrieval, and serving."""

    record_id: str
    citation_id: str
    document_id: str
    document_title: str
    namespace: str
    language: str
    acl: tuple[str, ...]
    generation: str
    section_kind: str
    section_label: str
    chunk_index: int
    source_uri: str
    source_sha256: str
    content_sha256: str
    source_span: SourceSpan
    attributes: dict[str, object]
    text: str

    @property
    def retrieval_text(self) -> str:
        labels = " — ".join(dict.fromkeys(
            value for value in (
                self.document_title, self.section_label, self.document_id
            ) if value
        ))
        return f"{labels}\n{self.text}" if labels else self.text

    @property
    def memory_text(self) -> str:
        return f"Memory record {self.citation_id}. {self.retrieval_text}"

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["acl"] = list(self.acl)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "KnowledgeRecord":
        values = dict(row)
        span = values.get("source_span")
        if not isinstance(span, dict):
            raise ValueError("record has no valid source_span")
        values["source_span"] = SourceSpan(
            character_start=int(span["character_start"]),
            character_end=int(span["character_end"]),
        )
        acl = values.get("acl")
        if not isinstance(acl, list) or not all(isinstance(item, str) for item in acl):
            raise ValueError("record has no valid ACL")
        values["acl"] = tuple(acl)
        attributes = values.get("attributes")
        if not isinstance(attributes, dict):
            raise ValueError("record has no valid attributes object")
        return cls(**values)


def content_sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def safe_identifier(value: str, maximum: int = 48) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").upper()
    return normalized[:maximum] or "UNNAMED"


def iter_record_shards(root: Path) -> Iterator[tuple[dict[str, object], Path]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported ingest manifest schema")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("ingest manifest has no shards")
    for shard in shards:
        if not isinstance(shard, dict) or "records" not in shard:
            raise ValueError("invalid ingest shard descriptor")
        yield shard, root / str(shard["records"])


def read_records(root: Path) -> Iterator[KnowledgeRecord]:
    for _, path in iter_record_shards(root):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                        yield KnowledgeRecord.from_dict(row)
                    except Exception as error:
                        raise ValueError(f"invalid record at {path}:{line_number}") from error


def write_record_shards(records: Sequence[KnowledgeRecord], root: Path,
                        records_per_shard: int) -> list[dict[str, object]]:
    if records_per_shard < 1:
        raise ValueError("records_per_shard must be positive")
    shards: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, len(records), records_per_shard)):
        rows = records[start:start + records_per_shard]
        name = f"records-{shard_index:05d}.jsonl"
        path = root / name
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in rows:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        shards.append({
            "shard_id": shard_index,
            "record_count": len(rows),
            "records": name,
            "sha256": content_sha256(path.read_bytes()),
        })
    return shards


def records_fingerprint(records: Iterable[KnowledgeRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()
