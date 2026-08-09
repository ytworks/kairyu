"""Fail-closed mechanics shared by repository benchmark evidence gates.

The installed package owns byte encoding, hashing, strict JSON/JSONL framing,
and retained-manifest replay comparison.  Gate-specific schemas, thresholds,
and verdicts remain in the repository-only ``bench/*.py`` entrypoints.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from kairyu.bench.reporting import atomic_write_text


class EvidenceError(ValueError):
    """An evidence artifact violates the shared encoding or replay contract."""


_Error = TypeVar("_Error", bound=Exception)
_Manifest = dict[str, object]


def canonical_json(value: object) -> str:
    """Return the single UTF-8 JSON representation used for evidence hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_json(value: object) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_loads(
    value: str | bytes,
    *,
    error_type: type[_Error] = EvidenceError,
) -> object:
    """Parse RFC JSON while rejecting duplicate keys and non-finite numbers."""

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise error_type(f"duplicate JSON key {key!r}")
            result[key] = item
        return result

    try:
        return json.loads(
            value,
            object_pairs_hook=strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                error_type(f"non-finite JSON constant {constant}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise error_type("invalid JSON") from error


def read_json_object(
    path: str | Path,
    *,
    error_type: type[_Error] = EvidenceError,
) -> _Manifest:
    source = Path(path)
    try:
        value = strict_json_loads(
            source.read_bytes(),
            error_type=error_type,
        )
    except OSError as error:
        raise error_type(f"cannot read JSON object: {source}") from error
    if not isinstance(value, dict):
        raise error_type(f"{source} is not a JSON object")
    return value


def write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Normalize row indices and atomically publish canonical JSONL."""

    normalized = [
        {**dict(row), "row_index": index}
        for index, row in enumerate(rows)
    ]
    return atomic_write_text(
        path,
        "".join(canonical_json(row) + "\n" for row in normalized),
        encoding="utf-8",
    )


def read_jsonl(
    path: str | Path,
    *,
    error_type: type[_Error] = EvidenceError,
) -> list[_Manifest]:
    """Read non-empty, newline-framed object rows with contiguous indices."""

    rows, _ = read_jsonl_with_sha256(path, error_type=error_type)
    return rows


def read_jsonl_with_sha256(
    path: str | Path,
    *,
    error_type: type[_Error] = EvidenceError,
) -> tuple[list[_Manifest], str]:
    """Read strict JSONL and hash the exact bytes parsed from one descriptor."""

    source = Path(path)
    rows: list[_Manifest] = []
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.endswith(b"\n") or not raw.strip():
                    raise error_type(
                        f"invalid JSONL framing at {source}:{line_number}"
                    )
                try:
                    value = strict_json_loads(raw, error_type=error_type)
                except error_type as error:
                    raise error_type(
                        f"invalid JSON at {source}:{line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise error_type(
                        f"non-object JSONL row at {source}:{line_number}"
                    )
                rows.append(value)
    except OSError as error:
        raise error_type(f"cannot read JSONL: {source}") from error
    if not rows:
        raise error_type(f"{source} contains no rows")
    if any(row.get("row_index") != index for index, row in enumerate(rows)):
        raise error_type(f"non-contiguous row indices in {source}")
    return rows, digest.hexdigest()


def artifact_paths(
    path: str | Path,
    *,
    raw_name: str,
    manifest_name: str,
    error_type: type[_Error] = EvidenceError,
) -> tuple[Path, Path]:
    """Resolve a directory, raw file, or manifest file to the artifact pair."""

    source = Path(path)
    if source.is_dir():
        return source / raw_name, source / manifest_name
    if source.name == raw_name:
        return source, source.with_name(manifest_name)
    if source.name == manifest_name:
        return source.with_name(raw_name), source
    raise error_type(
        f"artifact must be a directory, {raw_name}, or {manifest_name}"
    )


def replay_manifest(
    raw_path: str | Path,
    *,
    recompute: Callable[[Sequence[_Manifest], str], _Manifest],
    error_type: type[_Error] = EvidenceError,
) -> _Manifest:
    """Recompute one manifest from raw bytes without trusting retained summary."""

    source = Path(raw_path)
    rows, raw_sha256 = read_jsonl_with_sha256(source, error_type=error_type)
    return recompute(rows, raw_sha256)


def verify_replayed_manifest(
    path: str | Path,
    *,
    raw_name: str,
    manifest_name: str,
    recompute: Callable[[Sequence[_Manifest], str], _Manifest],
    error_type: type[_Error] = EvidenceError,
) -> _Manifest:
    """Fail unless the retained manifest exactly equals independent raw replay."""

    raw_path, manifest_path = artifact_paths(
        path,
        raw_name=raw_name,
        manifest_name=manifest_name,
        error_type=error_type,
    )
    if not raw_path.is_file():
        raise error_type(f"raw evidence is missing: {raw_path}")
    if not manifest_path.is_file():
        raise error_type(f"manifest is missing: {manifest_path}")
    replayed = replay_manifest(
        raw_path,
        recompute=recompute,
        error_type=error_type,
    )
    retained = read_json_object(manifest_path, error_type=error_type)
    if retained != replayed:
        raise error_type("retained manifest differs from independent raw replay")
    return replayed


__all__ = [
    "EvidenceError",
    "artifact_paths",
    "canonical_json",
    "read_json_object",
    "read_jsonl",
    "read_jsonl_with_sha256",
    "replay_manifest",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "strict_json_loads",
    "verify_replayed_manifest",
    "write_jsonl",
]
