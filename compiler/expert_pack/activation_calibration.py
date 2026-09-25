"""Validated runtime activation captures for offline FP4 scale selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any

from .constants import FP4_QUANT_GROUP_SIZE
from .util import canonical_json_bytes, sha256_bytes, sha256_file

try:
    import numpy as _np
except ImportError:  # pragma: no cover - checked explicitly by the caller.
    _np = None


_DENSE_MAGIC = b"QLCALD01"
_DENSE_VERSION = 1
_DENSE_HEADER = struct.Struct("<8sIIIIII")


@dataclass(frozen=True)
class _DenseCapture:
    path: Path
    name: str
    source_rows: int
    sampled_rows: int
    columns: int
    padded_columns: int
    row_indices: tuple[int, ...]
    bytes: int
    sha256: str


class DenseActivationCalibrationSet:
    """Index exact Q8 runtime inputs without retaining every capture in RAM."""

    def __init__(self, root: Path | str):
        if _np is None:
            raise ValueError(
                "activation-aware FP4 compilation requires NumPy"
            )
        self.root = Path(root).resolve()
        dense_root = self.root / "dense"
        if not dense_root.is_dir():
            raise ValueError(
                f"activation calibration dense directory is missing: {dense_root}"
            )
        captures: dict[str, _DenseCapture] = {}
        inventory: list[dict[str, Any]] = []
        for path in sorted(dense_root.glob("*.q8cal")):
            capture = self._inspect(path)
            if capture.name in captures:
                raise ValueError(
                    f"duplicate activation calibration tensor: {capture.name}"
                )
            captures[capture.name] = capture
            inventory.append({
                "name": capture.name,
                "bytes": capture.bytes,
                "sha256": capture.sha256,
                "source_rows": capture.source_rows,
                "sampled_rows": capture.sampled_rows,
                "columns": capture.columns,
                "padded_columns": capture.padded_columns,
            })
        if not captures:
            raise ValueError(
                f"activation calibration has no dense captures: {dense_root}"
            )
        self._captures = captures
        self._inventory_sha256 = sha256_bytes(canonical_json_bytes(inventory))
        self._used: set[str] = set()
        self._excluded: set[str] = set()
        self._rows = 0
        self._blocks = 0
        self._covering_reselections = 0
        self._payload_targets: frozenset[str] | None = None
        self._payload_reselections = 0

    @staticmethod
    def _inspect(path: Path) -> _DenseCapture:
        with path.open("rb") as handle:
            raw = handle.read(_DENSE_HEADER.size)
            if len(raw) != _DENSE_HEADER.size:
                raise ValueError(f"short activation calibration header: {path}")
            (
                magic,
                version,
                name_bytes,
                source_rows,
                sampled_rows,
                columns,
                padded_columns,
            ) = _DENSE_HEADER.unpack(raw)
            if magic != _DENSE_MAGIC or version != _DENSE_VERSION:
                raise ValueError(f"unsupported activation calibration: {path}")
            if (
                not name_bytes
                or not source_rows
                or not sampled_rows
                or sampled_rows > source_rows
                or not columns
                or padded_columns < columns
                or padded_columns % FP4_QUANT_GROUP_SIZE
            ):
                raise ValueError(
                    f"invalid activation calibration geometry: {path}"
                )
            encoded_name = handle.read(name_bytes)
            if len(encoded_name) != name_bytes:
                raise ValueError(f"short activation calibration name: {path}")
            try:
                name = encoded_name.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"invalid activation calibration tensor name: {path}"
                ) from error
            expected_filename = hashlib.sha256(encoded_name).hexdigest() + ".q8cal"
            if path.name != expected_filename:
                raise ValueError(
                    f"activation calibration filename does not authenticate {name}"
                )
            raw_indices = handle.read(sampled_rows * 4)
            if len(raw_indices) != sampled_rows * 4:
                raise ValueError(f"short activation calibration indices: {path}")
            row_indices = struct.unpack(f"<{sampled_rows}I", raw_indices)
            if (
                len(set(row_indices)) != sampled_rows
                or any(index >= source_rows for index in row_indices)
            ):
                raise ValueError(f"invalid activation calibration indices: {path}")
            raw_scales = handle.read(sampled_rows * 4)
            if len(raw_scales) != sampled_rows * 4:
                raise ValueError(f"short activation calibration scales: {path}")
            scales = struct.unpack(f"<{sampled_rows}f", raw_scales)
            if any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
                raise ValueError(f"invalid activation calibration scales: {path}")
        expected_bytes = (
            _DENSE_HEADER.size + name_bytes + sampled_rows * 8
            + sampled_rows * padded_columns
        )
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"activation calibration size mismatch for {name}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        return _DenseCapture(
            path=path,
            name=name,
            source_rows=source_rows,
            sampled_rows=sampled_rows,
            columns=columns,
            padded_columns=padded_columns,
            row_indices=tuple(row_indices),
            bytes=actual_bytes,
            sha256=sha256_file(path),
        )

    def source_summary(self) -> dict[str, object]:
        summary = {
            "format": "q8-runtime-input-v1",
            "dense_files": len(self._captures),
            "inventory_sha256": self._inventory_sha256,
        }
        if self._payload_targets is not None:
            names = sorted(self._payload_targets)
            summary.update({
                "payload_target_matrices": len(names),
                "payload_target_sha256": sha256_bytes(
                    canonical_json_bytes(names)
                ),
            })
        if self._excluded:
            names = sorted(self._excluded)
            summary.update({
                "excluded_dense_matrices": len(names),
                "excluded_matrix_sha256": sha256_bytes(
                    canonical_json_bytes(names)
                ),
            })
        return summary

    def contains(self, name: str) -> bool:
        return name in self._captures

    def configure_exclusions(self, names: set[str]) -> None:
        """Exclude captures replaced by a different declared weight ABI."""
        if self._used:
            raise RuntimeError("calibration exclusions must be configured before loading")
        unknown = names - set(self._captures)
        if unknown:
            raise ValueError(
                "calibration exclusions are not captured tensors: "
                + ", ".join(sorted(unknown)[:8])
            )
        self._excluded = set(names)

    def configure_payload_targets(self, names: set[str]) -> None:
        """Declare runtime-target matrices eligible for adjacent FP4 codes."""
        if self._used:
            raise RuntimeError("payload targets must be configured before loading")
        unknown = names - set(self._captures)
        if unknown:
            raise ValueError(
                "payload targets lack activation captures: "
                + ", ".join(sorted(unknown)[:8])
            )
        if not names:
            raise ValueError("activation-code profile has no target matrices")
        self._payload_targets = frozenset(names)

    def selects_payload_codes(self, name: str) -> bool:
        return self._payload_targets is not None and name in self._payload_targets

    def load(
        self,
        name: str,
        *,
        columns: int,
        padded_columns: int,
    ):
        if name in self._excluded:
            raise RuntimeError(
                f"excluded activation calibration was requested: {name}"
            )
        capture = self._captures.get(name)
        if capture is None:
            return None
        if (
            capture.columns != columns
            or capture.padded_columns != padded_columns
        ):
            raise ValueError(
                f"activation calibration geometry disagrees with {name}: "
                f"capture=({capture.columns},{capture.padded_columns}), "
                f"weight=({columns},{padded_columns})"
            )
        with capture.path.open("rb") as handle:
            handle.seek(
                _DENSE_HEADER.size + len(name.encode("utf-8"))
                + capture.sampled_rows * 8
            )
            raw = handle.read(capture.sampled_rows * capture.padded_columns)
            if len(raw) != capture.sampled_rows * capture.padded_columns:
                raise ValueError(
                    f"short activation calibration values for {name}"
                )
            if handle.read(1):
                raise ValueError(
                    f"trailing activation calibration values for {name}"
                )
        with capture.path.open("rb") as handle:
            handle.seek(_DENSE_HEADER.size + len(name.encode("utf-8"))
                        + capture.sampled_rows * 4)
            scale_bytes = handle.read(capture.sampled_rows * 4)
        scales = _np.frombuffer(scale_bytes, dtype="<f4").astype("<f8")
        values = _np.frombuffer(raw, dtype="i1").reshape(
            capture.sampled_rows, capture.padded_columns
        )
        self._used.add(name)
        return values.astype("<f8") * scales[:, None]

    def record_selection(
        self,
        name: str,
        *,
        rows: int,
        blocks: int,
        changed: int,
        payload_changed: int = 0,
    ) -> None:
        if name not in self._used:
            raise RuntimeError(
                f"activation selection was recorded before loading {name}"
            )
        if rows <= 0 or blocks <= 0 or not 0 <= changed <= rows * blocks:
            raise RuntimeError(f"invalid activation selection counts for {name}")
        maximum_payload_changes = rows * blocks * FP4_QUANT_GROUP_SIZE
        if not 0 <= payload_changed <= maximum_payload_changes:
            raise RuntimeError(f"invalid payload selection counts for {name}")
        if payload_changed and not self.selects_payload_codes(name):
            raise RuntimeError(f"payload changes are not enabled for {name}")
        self._rows += rows
        self._blocks += rows * blocks
        self._covering_reselections += changed
        self._payload_reselections += payload_changed

    def snapshot(self) -> dict[str, object]:
        snapshot = {
            **self.source_summary(),
            "dense_matrices": len(self._used),
            "matrix_names": sorted(self._used),
            "output_rows": self._rows,
            "blocks": self._blocks,
            "covering_reselections": self._covering_reselections,
        }
        if self._excluded:
            snapshot["excluded_matrix_names"] = sorted(self._excluded)
        if self._payload_targets is not None:
            snapshot.update({
                "payload_dense_matrices": len(self._payload_targets),
                "payload_matrix_names": sorted(self._payload_targets),
                "payload_reselections": self._payload_reselections,
            })
        return snapshot

    def restore(self, snapshot: object) -> None:
        if not isinstance(snapshot, dict):
            raise ValueError("activation calibration resume state is missing")
        names = snapshot.get("matrix_names")
        if (
            snapshot.get("format") != "q8-runtime-input-v1"
            or snapshot.get("dense_files") != len(self._captures)
            or snapshot.get("inventory_sha256") != self._inventory_sha256
            or not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or set(names) | self._excluded != set(self._captures)
            or set(names) & self._excluded
        ):
            raise ValueError("activation calibration resume state is incompatible")
        excluded_names = snapshot.get("excluded_matrix_names", [])
        if (
            not isinstance(excluded_names, list)
            or any(not isinstance(name, str) for name in excluded_names)
            or set(excluded_names) != self._excluded
        ):
            raise ValueError("activation calibration exclusions changed")
        rows = snapshot.get("output_rows")
        blocks = snapshot.get("blocks")
        changed = snapshot.get("covering_reselections")
        if not all(isinstance(value, int) and value >= 0
                   for value in (rows, blocks, changed)):
            raise ValueError("activation calibration resume counts are invalid")
        self._used = set(names)
        self._rows = rows
        self._blocks = blocks
        self._covering_reselections = changed
        if self._payload_targets is not None:
            payload_names = snapshot.get("payload_matrix_names")
            payload_changed = snapshot.get("payload_reselections")
            if (
                not isinstance(payload_names, list)
                or set(payload_names) != set(self._payload_targets)
                or snapshot.get("payload_dense_matrices")
                    != len(self._payload_targets)
                or not isinstance(payload_changed, int)
                or payload_changed < 0
            ):
                raise ValueError(
                    "activation payload resume state is incompatible"
                )
            self._payload_reselections = payload_changed

    def finalize(self) -> dict[str, object]:
        unused = sorted(set(self._captures) - self._used - self._excluded)
        if unused:
            raise ValueError(
                "activation calibration contains captures that were not applied: "
                + ", ".join(unused[:8])
            )
        if self._blocks <= 0:
            raise ValueError("activation calibration selected no FP4 blocks")
        if (
            self._payload_targets is not None
            and not self._payload_targets <= self._used
        ):
            raise ValueError("activation payload targets were not compiled")
        return self.snapshot()
