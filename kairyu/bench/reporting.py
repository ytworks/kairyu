"""Reusable, deterministic reporting helpers for benchmark entrypoints."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

_Sample = TypeVar("_Sample")
PERCENTILE_METHOD = "nearest-rank-v1"


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _descriptor_matches(descriptor: int, expected: bytes) -> bool:
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        stream.seek(0)
        return stream.read(len(expected) + 1) == expected


def _unlink_if_identity(path: Path, expected: tuple[int, int]) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if _identity(metadata) != expected:
        return False
    path.unlink()
    return True


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
    overwrite: bool = True,
    validate_path: Callable[[Path], Path] | None = None,
) -> Path:
    """Atomically publish ``text`` at ``path`` and return the final path.

    With ``overwrite=False``, publication uses an atomic hard link and fails if
    any file, directory, or symlink already occupies the destination.
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
    encoded = text.encode(encoding)
    fd: int | None = None
    temporary: Path | None = None
    expected_identity: tuple[int, int] | None = None
    publication_validated = False
    try:
        fd, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        # tempfile returns an absolute name even when ``dir`` was relative.
        # Keep the caller's path coordinate system for containment validators;
        # the reconstructed path names the same file under target.parent.
        temporary = target.parent / Path(temporary_name).name
        if validate_path is not None:
            temporary = validate_path(temporary)
        with os.fdopen(os.dup(fd), "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        metadata = os.fstat(fd)
        path_metadata = temporary.lstat()
        expected_identity = _identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _identity(path_metadata) != expected_identity
            or metadata.st_size != len(encoded)
            or not _descriptor_matches(fd, encoded)
        ):
            raise RuntimeError("report temporary changed before publication")
        if validate_path is not None:
            target = validate_path(target)
            temporary = validate_path(temporary)
        if overwrite:
            os.replace(temporary, target)
            temporary = None
        else:
            os.link(temporary, target, follow_symlinks=False)
            target_descriptor: int | None = None
            try:
                target_descriptor = os.open(
                    target,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                )
                retained = os.fstat(target_descriptor)
                target_matches = _descriptor_matches(target_descriptor, encoded)
                final_target = target.lstat()
                if (
                    not stat.S_ISREG(retained.st_mode)
                    or _identity(retained) != expected_identity
                    or retained.st_size != len(encoded)
                    or not target_matches
                    or not stat.S_ISREG(final_target.st_mode)
                    or _identity(final_target) != expected_identity
                    or final_target.st_size != len(encoded)
                ):
                    raise RuntimeError("published report failed identity validation")
            except BaseException:
                try:
                    target_metadata = target.lstat()
                    target_identity = _identity(target_metadata)
                    if target_identity == expected_identity:
                        _unlink_if_identity(target, expected_identity)
                    else:
                        source_metadata = temporary.lstat()
                        if _identity(source_metadata) == target_identity:
                            _unlink_if_identity(target, target_identity)
                except OSError:
                    pass
                raise
            finally:
                if target_descriptor is not None:
                    os.close(target_descriptor)
            publication_validated = True
    finally:
        if fd is not None:
            os.close(fd)
        if temporary is not None:
            try:
                if publication_validated and expected_identity is not None:
                    _unlink_if_identity(temporary, expected_identity)
                else:
                    temporary.unlink(missing_ok=True)
            except OSError:
                # Once an exclusive target is identity/content validated,
                # private-temp cleanup cannot turn a successful publication
                # into a caller-visible failure and trigger an invalid rollback.
                pass
    return target


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    overwrite: bool = True,
    validate_path: Callable[[Path], Path] | None = None,
) -> Path:
    """Serialize ``payload`` once and atomically publish the destination."""
    return atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
        ),
        overwrite=overwrite,
        validate_path=validate_path,
    )


__all__ = [
    "PERCENTILE_METHOD",
    "atomic_write_json",
    "atomic_write_text",
    "nearest_rank_percentile",
]
