"""Dependency-free SafeTensors metadata reader and per-tensor mapper."""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .constants import DTYPE_BYTES
from .errors import SourceFormatError
from .util import load_json

MAX_HEADER_BYTES = 256 * 1024 * 1024


def _element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension < 0:
            raise SourceFormatError(f"negative SafeTensors dimension: {shape}")
        result *= dimension
    return result


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


class TensorView:
    """Owns one read-only mmap while exposing only a single tensor slice."""

    def __init__(self, path: Path, info: TensorInfo) -> None:
        self.path = path
        self.info = info
        self._handle = path.open("rb")
        self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.raw = memoryview(self._mapping)[info.offset : info.offset + info.nbytes]

    def close(self) -> None:
        self.raw.release()
        self._mapping.close()
        self._handle.close()

    def __enter__(self) -> "TensorView":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class SafeTensorCheckpoint:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise SourceFormatError(f"checkpoint directory does not exist: {root}")
        config_path = self.root / "config.json"
        if not config_path.is_file():
            raise SourceFormatError("checkpoint has no config.json")
        self.config = load_json(config_path)
        if not isinstance(self.config, dict):
            raise SourceFormatError("config.json must contain an object")
        self.tensors, self.shards = self._read_index()

    def _read_index(self) -> tuple[dict[str, TensorInfo], tuple[str, ...]]:
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            index = load_json(index_path)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise SourceFormatError("SafeTensors index has no non-empty weight_map")
            for name, shard in weight_map.items():
                if not isinstance(name, str) or not isinstance(shard, str):
                    raise SourceFormatError("SafeTensors weight_map must map strings to strings")
            shard_names = tuple(sorted(set(weight_map.values())))
        else:
            single = self.root / "model.safetensors"
            if not single.is_file():
                raise SourceFormatError("checkpoint has neither SafeTensors index nor model.safetensors")
            shard_names = (single.name,)
            weight_map = None

        tensors: dict[str, TensorInfo] = {}
        for shard_name in shard_names:
            shard_path = self.root / shard_name
            if shard_path.parent != self.root or not shard_path.is_file():
                raise SourceFormatError(f"missing or unsafe shard path: {shard_name}")
            shard_tensors = _read_shard_header(shard_path, shard_name)
            duplicate = tensors.keys() & shard_tensors.keys()
            if duplicate:
                raise SourceFormatError(f"duplicate tensors across shards: {sorted(duplicate)[:3]}")
            tensors.update(shard_tensors)

        if weight_map is not None:
            indexed = set(weight_map)
            discovered = set(tensors)
            if indexed != discovered:
                missing = sorted(indexed - discovered)
                extra = sorted(discovered - indexed)
                raise SourceFormatError(
                    f"index/header tensor mismatch; missing={missing[:3]}, extra={extra[:3]}"
                )
            mismatched = [name for name, info in tensors.items() if weight_map[name] != info.shard]
            if mismatched:
                raise SourceFormatError(f"index maps tensors to wrong shard: {mismatched[:3]}")
        return tensors, shard_names

    def open_tensor(self, name: str) -> TensorView:
        try:
            info = self.tensors[name]
        except KeyError as error:
            raise SourceFormatError(f"unknown source tensor: {name}") from error
        return TensorView(self.root / info.shard, info)

    def source_files(self) -> Iterator[Path]:
        yield self.root / "config.json"
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            yield index
        for shard in self.shards:
            yield self.root / shard


def _read_shard_header(path: Path, shard_name: str) -> dict[str, TensorInfo]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        length_raw = handle.read(8)
        if len(length_raw) != 8:
            raise SourceFormatError(f"truncated SafeTensors prefix: {path}")
        header_bytes = struct.unpack("<Q", length_raw)[0]
        if header_bytes <= 1 or header_bytes > MAX_HEADER_BYTES:
            raise SourceFormatError(f"unsafe SafeTensors header size {header_bytes}: {path}")
        header_raw = handle.read(header_bytes)
        if len(header_raw) != header_bytes:
            raise SourceFormatError(f"truncated SafeTensors header: {path}")
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceFormatError(f"invalid SafeTensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise SourceFormatError(f"SafeTensors header is not an object: {path}")

    data_base = 8 + header_bytes
    result: dict[str, TensorInfo] = {}
    spans: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise SourceFormatError(f"malformed tensor metadata in {path}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            raise SourceFormatError(f"unsupported SafeTensors dtype {dtype!r} for {name}")
        if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
            raise SourceFormatError(f"invalid shape for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise SourceFormatError(f"invalid data_offsets for {name}")
        start, end = offsets
        expected = _element_count(tuple(shape)) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end - start != expected:
            raise SourceFormatError(f"byte size does not match dtype/shape for {name}")
        absolute_start = data_base + start
        absolute_end = data_base + end
        if absolute_end > file_size:
            raise SourceFormatError(f"tensor exceeds shard EOF: {name}")
        if name in result:
            raise SourceFormatError(f"duplicate tensor in shard: {name}")
        result[name] = TensorInfo(
            name=name,
            shard=shard_name,
            dtype=dtype,
            shape=tuple(shape),
            offset=absolute_start,
            nbytes=expected,
        )
        spans.append((absolute_start, absolute_end, name))
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if previous[1] > current[0]:
            raise SourceFormatError(
                f"overlapping tensors in {path}: {previous[2]} and {current[2]}"
            )
    if not result:
        raise SourceFormatError(f"SafeTensors shard has no tensors: {path}")
    return result

