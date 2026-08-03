from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_all(handle: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = handle.write(view)
        if written is None or written <= 0:
            raise OSError("short write")
        view = view[written:]


def write_zeros(handle: BinaryIO, count: int, digest: Any | None = None) -> None:
    zero = b"\0" * min(1024 * 1024, max(1, count))
    remaining = count
    while remaining:
        chunk = zero[: min(len(zero), remaining)]
        write_all(handle, chunk)
        if digest is not None:
            digest.update(chunk)
        remaining -= len(chunk)


def fsync_file(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    payload += "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fsync_directory(path: Path) -> None:
    # Windows does not permit opening a directory this way; atomic replace is
    # still used there, while POSIX gets the stronger durability barrier.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

