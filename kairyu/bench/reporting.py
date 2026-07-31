"""Reusable, deterministic reporting helpers for benchmark entrypoints."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

_Sample = TypeVar("_Sample")
PERCENTILE_METHOD = "nearest-rank-v1"


def nearest_rank_percentile(values: Iterable[_Sample], fraction: float) -> _Sample:
    """Return the nearest-rank percentile for ``0 < fraction <= 1``.

    The nearest-rank definition selects sorted rank ``ceil(fraction * n)``
    (ranks start at one).  Sorting inside this helper prevents callers from
    accidentally reporting a percentile over insertion order.
    """
    if (
        isinstance(fraction, bool)
        or not math.isfinite(fraction)
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError(f"fraction must satisfy 0 < fraction <= 1, got {fraction!r}")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("nearest-rank percentile requires at least one sample")
    rank = math.ceil(fraction * len(ordered))
    return ordered[rank - 1]


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
    validate_path: Callable[[Path], Path] | None = None,
) -> Path:
    """Atomically replace ``path`` with ``text`` and return the final path.

    ``validate_path`` lets a domain-specific store enforce containment and
    symlink policy for both the destination and randomized temporary file
    without maintaining a second atomic-write implementation.
    """
    target = Path(path)
    if validate_path is not None:
        target = validate_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if validate_path is not None:
        target = validate_path(target)
    fd: int | None = None
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            text=True,
        )
        # tempfile returns an absolute name even when ``dir`` was relative.
        # Keep the caller's path coordinate system for containment validators;
        # the reconstructed path names the same file under target.parent.
        temporary = target.parent / Path(temporary_name).name
        if validate_path is not None:
            temporary = validate_path(temporary)
        with os.fdopen(fd, "w", encoding=encoding) as output:
            fd = None
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        if validate_path is not None:
            target = validate_path(target)
            temporary = validate_path(temporary)
        os.replace(temporary, target)
        temporary = None
    finally:
        if fd is not None:
            os.close(fd)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    validate_path: Callable[[Path], Path] | None = None,
) -> Path:
    """Serialize ``payload`` once and atomically replace the destination."""
    return atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        ),
        validate_path=validate_path,
    )


__all__ = [
    "PERCENTILE_METHOD",
    "atomic_write_json",
    "atomic_write_text",
    "nearest_rank_percentile",
]
