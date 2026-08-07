from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from .data_contract import (
        KnowledgeRecord,
        SourceSpan,
        content_sha256,
        records_fingerprint,
        safe_identifier,
        write_record_shards,
    )
except ImportError:  # Direct execution on a worker.
    from data_contract import (
        KnowledgeRecord,
        SourceSpan,
        content_sha256,
        records_fingerprint,
        safe_identifier,
        write_record_shards,
    )


@dataclass(frozen=True)
class LoadedDocument:
    document_id: str
    title: str
    language: str
    acl: tuple[str, ...]
    generation: str
    metadata: dict[str, object]
    text: str
    source_uri: str
    source_sha256: str


@dataclass(frozen=True)
class Section:
    kind: str
    label: str
    start: int
    end: int


def json_pointer(value: object, pointer: str, default: Any = None) -> Any:
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must begin with '/': {pointer}")
    current: Any = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return default
    return current


def _string_field(payload: object, pointer: str, fallback: str) -> str:
    value = json_pointer(payload, pointer, fallback) if pointer else fallback
    return str(value) if value is not None else fallback


def load_document(path: Path, config: dict[str, object]) -> LoadedDocument:
    source_bytes = path.read_bytes()
    source = config.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("config.source must be an object")
    source_format = str(source.get("format", "json"))
    if source_format == "json":
        payload: object = json.loads(source_bytes.decode("utf-8-sig"))
        text = json_pointer(payload, str(source.get("text_pointer", "/text")))
        if not isinstance(text, str):
            raise ValueError("configured JSON text field is not a string")
    elif source_format == "text":
        payload = {}
        text = source_bytes.decode(str(source.get("encoding", "utf-8")))
    else:
        raise ValueError(f"unsupported source format: {source_format}")

    defaults = config.get("defaults", {})
    identity = config.get("identity", {})
    metadata_pointers = source.get("metadata_pointers", {})
    if not isinstance(defaults, dict) or not isinstance(identity, dict):
        raise ValueError("config defaults and identity must be objects")
    if not isinstance(metadata_pointers, dict):
        raise ValueError("source.metadata_pointers must be an object")
    metadata = {
        str(name): json_pointer(payload, str(pointer))
        for name, pointer in metadata_pointers.items()
    }
    document_id = _string_field(
        payload, str(identity.get("document_id_pointer", "")),
        str(defaults.get("document_id", path.stem)),
    )
    title = _string_field(
        payload, str(identity.get("title_pointer", "")),
        str(defaults.get("title", document_id)),
    )
    language = _string_field(
        payload, str(identity.get("language_pointer", "")),
        str(defaults.get("language", "und")),
    )
    generation = _string_field(
        payload, str(identity.get("generation_pointer", "")),
        str(defaults.get("generation", "1")),
    )
    default_acl = defaults.get("acl", ["public"])
    acl_pointer = str(identity.get("acl_pointer", ""))
    acl_value = json_pointer(payload, acl_pointer, default_acl) if acl_pointer else default_acl
    if isinstance(acl_value, str):
        acl_value = [acl_value]
    if not isinstance(acl_value, list) or not all(isinstance(item, str) for item in acl_value):
        raise ValueError("document ACL must be a string or an array of strings")
    return LoadedDocument(
        document_id=document_id,
        title=title,
        language=language,
        acl=tuple(acl_value),
        generation=generation,
        metadata=metadata,
        text=text,
        source_uri=str(path.resolve()),
        source_sha256=content_sha256(source_bytes),
    )


def detect_sections(text: str, segmentation: dict[str, object]) -> list[Section]:
    pattern = str(segmentation.get("section_pattern", ""))
    kind = str(segmentation.get("section_kind", "section"))
    if not pattern:
        return [Section(kind="document", label="", start=0, end=len(text))]
    expression = re.compile(pattern, re.MULTILINE | re.UNICODE)
    matches = list(expression.finditer(text))
    if not matches:
        return [Section(kind="document", label="", start=0, end=len(text))]
    sections: list[Section] = []
    if matches[0].start() > 0 and text[:matches[0].start()].strip():
        sections.append(Section("preamble", "Preamble", 0, matches[0].start()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label_group = segmentation.get("section_label_group", "label")
        try:
            label = match.group(label_group) if label_group is not None else match.group(0)
        except (IndexError, KeyError):
            label = match.group(0)
        sections.append(Section(kind, " ".join(label.split()), match.start(), end))
    return sections


def _paragraph_ranges(text: str, start: int, end: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r"\n\s*\n+", text[start:end]):
        paragraph_end = start + match.start()
        if text[cursor:paragraph_end].strip():
            left = cursor + len(text[cursor:paragraph_end]) - len(text[cursor:paragraph_end].lstrip())
            right = paragraph_end - (len(text[cursor:paragraph_end]) - len(text[cursor:paragraph_end].rstrip()))
            ranges.append((left, right))
        cursor = start + match.end()
    if text[cursor:end].strip():
        left = cursor + len(text[cursor:end]) - len(text[cursor:end].lstrip())
        right = end - (len(text[cursor:end]) - len(text[cursor:end].rstrip()))
        ranges.append((left, right))
    return ranges


def _split_oversize(text: str, start: int, end: int,
                    maximum_chars: int) -> Iterator[tuple[int, int]]:
    cursor = start
    sentence_end = re.compile(r"(?<=[.!?;])\s+")
    while end - cursor > maximum_chars:
        window_end = cursor + maximum_chars
        window = text[cursor:window_end]
        boundaries = [match.end() for match in sentence_end.finditer(window)]
        cut = cursor + (boundaries[-1] if boundaries and boundaries[-1] >= maximum_chars // 2 else maximum_chars)
        yield cursor, cut
        cursor = cut
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if cursor < end:
        yield cursor, end


def chunk_section(text: str, section: Section, maximum_chars: int,
                  overlap_paragraphs: int) -> list[tuple[int, int]]:
    paragraphs: list[tuple[int, int]] = []
    for start, end in _paragraph_ranges(text, section.start, section.end):
        if end - start <= maximum_chars:
            paragraphs.append((start, end))
        else:
            paragraphs.extend(_split_oversize(text, start, end, maximum_chars))
    if not paragraphs:
        return []
    chunks: list[tuple[int, int]] = []
    position = 0
    while position < len(paragraphs):
        start = paragraphs[position][0]
        end = paragraphs[position][1]
        next_position = position + 1
        while next_position < len(paragraphs):
            candidate_end = paragraphs[next_position][1]
            if candidate_end - start > maximum_chars:
                break
            end = candidate_end
            next_position += 1
        chunks.append((start, end))
        if next_position >= len(paragraphs):
            break
        position = max(position + 1, next_position - overlap_paragraphs)
    return chunks


def build_records(document: LoadedDocument, config: dict[str, object]) -> list[KnowledgeRecord]:
    defaults = config.get("defaults", {})
    segmentation = config.get("segmentation", {})
    if not isinstance(defaults, dict) or not isinstance(segmentation, dict):
        raise ValueError("config defaults and segmentation must be objects")
    namespace = str(defaults.get("namespace", "knowledge"))
    maximum_chars = int(segmentation.get("maximum_chars", 1200))
    overlap_paragraphs = int(segmentation.get("overlap_paragraphs", 1))
    if maximum_chars < 128 or overlap_paragraphs < 0:
        raise ValueError("invalid segmentation limits")
    document_key = safe_identifier(f"{namespace}-{document.document_id}", 56)
    records: list[KnowledgeRecord] = []
    for section_ordinal, section in enumerate(detect_sections(document.text, segmentation)):
        spans = chunk_section(
            document.text, section, maximum_chars, overlap_paragraphs
        )
        section_key = safe_identifier(section.label or f"{section.kind}-{section_ordinal}", 32)
        for chunk_index, (start, end) in enumerate(spans):
            text = document.text[start:end]
            digest = content_sha256(text)
            citation_id = f"{document_key}-{section_key}-P{chunk_index + 1}"
            record_id = f"{citation_id}-{digest[:12].upper()}"
            records.append(KnowledgeRecord(
                record_id=record_id,
                citation_id=citation_id,
                document_id=document.document_id,
                document_title=document.title,
                namespace=namespace,
                language=document.language,
                acl=document.acl,
                generation=document.generation,
                section_kind=section.kind,
                section_label=section.label,
                chunk_index=chunk_index,
                source_uri=document.source_uri,
                source_sha256=document.source_sha256,
                content_sha256=digest,
                source_span=SourceSpan(start, end),
                attributes=document.metadata,
                text=text,
            ))
    return records


def ingest(input_path: Path, config_path: Path, output: Path,
           source_uri: str | None = None) -> dict[str, object]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported ingestion config schema")
    document = load_document(input_path, config)
    if source_uri:
        document = replace(document, source_uri=source_uri)
    loaded_at = time.perf_counter()
    records = build_records(document, config)
    segmented_at = time.perf_counter()
    output_settings = config.get("output", {})
    if not isinstance(output_settings, dict):
        raise ValueError("config.output must be an object")
    records_per_shard = int(output_settings.get("records_per_shard", 128))
    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output_parent))
    try:
        shards = write_record_shards(records, temporary, records_per_shard)
        manifest = {
            "schema_version": 1,
            "contract": "quantum-llm-knowledge-record-v1",
            "document": {
                "document_id": document.document_id,
                "title": document.title,
                "language": document.language,
                "generation": document.generation,
                "metadata": document.metadata,
                "source_uri": document.source_uri,
                "source_sha256": document.source_sha256,
                "text_characters": len(document.text),
            },
            "record_count": len(records),
            "records_fingerprint": records_fingerprint(records),
            "segmentation": config.get("segmentation", {}),
            "shards": shards,
            "timing_seconds": {
                "load": loaded_at - started,
                "segment": segmented_at - loaded_at,
                "persist": 0.0,
                "total": 0.0,
            },
        }
        persisted_at = time.perf_counter()
        manifest["timing_seconds"]["persist"] = persisted_at - segmented_at
        manifest["timing_seconds"]["total"] = persisted_at - started
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic immutable knowledge ingestion")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-uri",
        help="Stable logical source URI stored in records instead of the local input path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = ingest(
        args.input.resolve(), args.config.resolve(), args.output.resolve(), args.source_uri
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
