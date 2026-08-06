"""Optional, reusable PyTorch profiling for benchmark entrypoints.

The module itself is core-only: PyTorch is imported only when profiling is
explicitly enabled.  Callers own warm-up, synchronization, scheduling, and the
measured scope; this helper only normalizes profiler construction and publishes
bounded Chrome traces without overwriting existing evidence.
"""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRACE_FORMAT = "pytorch-chrome-trace-json-v1"
TRACE_SUFFIX = ".pt.trace.json"
MAX_TRACE_BYTES = 64 * 1024 * 1024

_ACTIVITY_NAMES = frozenset({"cpu", "cuda"})
_SCOPE_RE = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")
_RESULT_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
_TRACE_NAME_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,223}\.pt\.trace\.json\Z"
)
_RANGE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")


class ProfilingUnavailable(RuntimeError):
    """Raised when explicitly requested profiling cannot be provided."""


@dataclass(frozen=True)
class TraceArtifact:
    """Integrity metadata for one exclusively published Chrome trace."""

    path: Path
    size_bytes: int
    sha256: str
    format: str = TRACE_FORMAT
    _device: int = field(repr=False, default=-1)
    _inode: int = field(repr=False, default=-1)


def _load_torch() -> Any:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise ProfilingUnavailable(
            "torch profiling requires PyTorch; install the `engine` extra "
            "or the development dependencies"
        ) from error
    if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
        raise ProfilingUnavailable("the installed PyTorch build has no profiler support")
    return torch


def _resolved_activities(torch: Any, activities: Sequence[str]) -> tuple[Any, ...]:
    if isinstance(activities, (str, bytes)):
        raise ValueError("profiler activities must be a non-empty sequence of names")
    names = tuple(activities)
    if not names:
        raise ValueError("profiler activities must not be empty")
    if any(type(name) is not str or name not in _ACTIVITY_NAMES for name in names):
        raise ValueError("profiler activities must contain only 'cpu' and/or 'cuda'")
    if len(set(names)) != len(names):
        raise ValueError("profiler activities must not contain duplicates")
    if "cuda" in names and not bool(torch.cuda.is_available()):
        raise ProfilingUnavailable("CUDA profiling was requested but CUDA is unavailable")
    activity_type = torch.profiler.ProfilerActivity
    return tuple(getattr(activity_type, name.upper()) for name in names)


def profile_scope(
    enabled: bool = True,
    *,
    activities: Sequence[str] = ("cpu",),
    acc_events: bool = False,
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
) -> AbstractContextManager[Any | None]:
    """Return a native PyTorch profiler, or a torch-free null context.

    No synchronization, warm-up, schedule, profiler step, range annotation, or
    artifact write is implicit.  Those choices affect the measured contract and
    remain the responsibility of each benchmark or correctness gate.  The
    enabled path returns the native context manager directly so profiler stack
    capture does not acquire an extra generator frame.
    """

    options = {
        "enabled": enabled,
        "acc_events": acc_events,
        "record_shapes": record_shapes,
        "profile_memory": profile_memory,
        "with_stack": with_stack,
    }
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("profiler enablement and options must be booleans")
    if not enabled:
        return nullcontext(None)

    torch = _load_torch()
    resolved = _resolved_activities(torch, activities)
    return torch.profiler.profile(
        activities=list(resolved),
        acc_events=acc_events,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )


@contextmanager
def record_function_scope(enabled: bool, name: str) -> Iterator[None]:
    """Add one safe, explicit PyTorch range without importing torch when off."""

    if type(enabled) is not bool:
        raise ValueError("profile range enablement must be a boolean")
    if _RANGE_NAME_RE.fullmatch(name) is None:
        raise ValueError("profile range name must be a bounded safe identifier")
    if not enabled:
        yield
        return
    torch = _load_torch()
    with torch.profiler.record_function(name):
        yield


def profile_trace_path(result_json: str | os.PathLike[str], *, scope: str) -> Path:
    """Derive ``<result-stem>.<scope>.pt.trace.json`` from a result JSON path."""

    result = Path(result_json)
    if result.suffix != ".json" or _RESULT_STEM_RE.fullmatch(result.stem) is None:
        raise ValueError("profile result must have a bounded safe `.json` filename")
    if _SCOPE_RE.fullmatch(scope) is None:
        raise ValueError("profile scope must be a bounded lowercase identifier")
    return result.with_name(f"{result.stem}.{scope}{TRACE_SUFFIX}")


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Chrome trace contains a non-finite number")
        return parsed

    decoded = payload.decode("utf-8")
    parsed = json.loads(
        decoded,
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_object,
    )
    if not isinstance(parsed, dict) or not isinstance(parsed.get("traceEvents"), list):
        raise ValueError("Chrome trace must be a JSON object with a traceEvents list")
    return parsed


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _read_descriptor(descriptor: int, expected_size: int) -> bytes:
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        stream.seek(0)
        payload = stream.read(MAX_TRACE_BYTES + 1)
    if len(payload) != expected_size:
        raise ValueError("profiler trace changed while being retained")
    return payload


def _unlink_if_identity(path: Path, expected: tuple[int, int]) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if _identity(metadata) != expected:
        return False
    path.unlink()
    return True


def _artifact_paths(
    path: str | os.PathLike[str],
    artifact_root: str | os.PathLike[str],
) -> tuple[Path, Path]:
    root = Path(os.path.abspath(os.fspath(artifact_root)))
    target = Path(os.path.abspath(os.fspath(path)))
    if root.is_symlink():
        raise ValueError("profile artifact root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("profile artifact root must be a real directory")
    if target.parent != root:
        raise ValueError("profile artifact must be a direct child of its artifact root")
    if _TRACE_NAME_RE.fullmatch(target.name) is None:
        raise ValueError(f"profile artifact must use the {TRACE_SUFFIX!r} naming convention")
    if target.parent.resolve(strict=True) != root.resolve(strict=True):
        raise ValueError("profile artifact escapes its artifact root")
    return root, target


def validate_trace_artifact_path(
    path: str | os.PathLike[str],
    *,
    artifact_root: str | os.PathLike[str],
) -> Path:
    """Create/validate the root and reject an existing trace before a run."""

    root, target = _artifact_paths(path, artifact_root)
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "profile artifact already exists", target)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root,
        prefix=f".{target.name}.",
        suffix=".preflight",
    )
    os.close(descriptor)
    Path(temporary_name).unlink()
    return target


def write_trace_artifact(
    profiler: Any,
    path: str | os.PathLike[str],
    *,
    artifact_root: str | os.PathLike[str],
) -> TraceArtifact:
    """Export and exclusively publish one bounded, strict Chrome trace.

    Export happens in a private same-filesystem temporary file.  A hard-link
    publication makes an existing file, directory, symlink, or concurrent
    publisher a hard failure instead of replacing prior evidence.
    """

    exporter = getattr(profiler, "export_chrome_trace", None)
    if profiler is None or not callable(exporter):
        raise TypeError("a closed native PyTorch profiler is required")
    root, target = _artifact_paths(path, artifact_root)
    if os.path.lexists(target):
        raise FileExistsError(errno.EEXIST, "profile artifact already exists", target)

    descriptor: int | None = None
    temporary: Path | None = None
    expected_identity: tuple[int, int] | None = None
    publication_validated = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=root,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        # Kineto may atomically replace the pathname handed to
        # export_chrome_trace().  Do not retain the mkstemp inode as the
        # expected export identity; reopen the exported pathname without
        # following symlinks and pin that resulting inode instead.
        os.close(descriptor)
        descriptor = None
        exporter(os.fspath(temporary))

        descriptor = os.open(
            temporary,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        metadata = os.fstat(descriptor)
        path_metadata = temporary.lstat()
        if _identity(path_metadata) != _identity(metadata):
            raise ValueError("profiler trace path changed while being retained")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("profiler export did not produce a regular file")
        if metadata.st_size <= 0:
            raise ValueError("profiler export produced an empty trace")
        if metadata.st_size > MAX_TRACE_BYTES:
            raise ValueError(
                f"profiler trace exceeds the {MAX_TRACE_BYTES}-byte safety limit"
        )
        os.fchmod(descriptor, 0o600)
        payload = _read_descriptor(descriptor, metadata.st_size)
        _strict_json_object(payload)
        digest = hashlib.sha256(payload).hexdigest()
        os.fsync(descriptor)
        expected_identity = _identity(metadata)
        source_before_publish = temporary.lstat()
        if (
            _identity(source_before_publish) != expected_identity
            or source_before_publish.st_size != metadata.st_size
        ):
            raise ValueError("profiler trace changed before publication")
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            raise
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise FileExistsError(
                    errno.EEXIST,
                    "profile artifact already exists",
                    target,
                ) from error
            raise
        target_metadata = target.lstat()
        target_identity = _identity(target_metadata)
        if target_identity != expected_identity:
            try:
                source_after_publish = temporary.lstat()
            except FileNotFoundError:
                source_after_publish = None
            if (
                source_after_publish is not None
                and _identity(source_after_publish) == target_identity
            ):
                _unlink_if_identity(target, target_identity)
            raise RuntimeError("profile trace source changed during publication")

        target_descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            target_descriptor = os.open(target, flags)
            retained = os.fstat(target_descriptor)
            retained_payload = _read_descriptor(target_descriptor, retained.st_size)
            final_target = target.lstat()
            if (
                _identity(retained) != expected_identity
                or retained.st_size != metadata.st_size
                or retained_payload != payload
                or _identity(final_target) != expected_identity
                or final_target.st_size != metadata.st_size
            ):
                _unlink_if_identity(target, expected_identity)
                raise RuntimeError("published profile trace failed identity validation")
        except BaseException:
            _unlink_if_identity(target, expected_identity)
            raise
        finally:
            if target_descriptor is not None:
                os.close(target_descriptor)
        publication_validated = True
        return TraceArtifact(
            path=target,
            size_bytes=metadata.st_size,
            sha256=digest,
            _device=metadata.st_dev,
            _inode=metadata.st_ino,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                if publication_validated and expected_identity is not None:
                    _unlink_if_identity(temporary, expected_identity)
                else:
                    temporary.unlink(missing_ok=True)
            except OSError:
                # Publication success is authoritative once the target was
                # identity/content validated.  A private-temp cleanup failure
                # must not make callers roll back a valid result/trace pair.
                pass


def remove_trace_artifact(artifact: TraceArtifact) -> bool:
    """Quarantine, verify, and remove only the represented trace.

    Moving the pathname to a randomized private quarantine first prevents a
    concurrent publisher at the public name from being unlinked after an
    identity check.  If the move captured a replacement instead, it is restored
    exclusively and the cleanup fails without deleting that evidence.
    """

    expected = (artifact._device, artifact._inode)
    try:
        metadata = artifact.path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != expected:
        raise RuntimeError("refusing to remove a replaced profile artifact")

    quarantine_descriptor, quarantine_name = tempfile.mkstemp(
        dir=artifact.path.parent,
        prefix=f".{artifact.path.name}.",
        suffix=".rollback",
    )
    quarantine = Path(quarantine_name)
    placeholder_identity = _identity(os.fstat(quarantine_descriptor))
    descriptor: int | None = None
    moved = False
    try:
        try:
            os.replace(artifact.path, quarantine)
        except FileNotFoundError:
            return False
        moved = True
        quarantined = quarantine.lstat()
        if _identity(quarantined) != expected:
            try:
                os.link(quarantine, artifact.path, follow_symlinks=False)
            except FileExistsError as error:
                raise RuntimeError(
                    f"profile artifact changed during cleanup; replacement "
                    f"preserved at {quarantine}"
                ) from error
            quarantine.unlink()
            moved = False
            raise RuntimeError("refusing to remove a replaced profile artifact")

        descriptor = os.open(
            quarantine,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        retained = os.fstat(descriptor)
        if (
            _identity(retained) != expected
            or retained.st_size != artifact.size_bytes
            or hashlib.sha256(
                _read_descriptor(descriptor, retained.st_size)
            ).hexdigest()
            != artifact.sha256
        ):
            try:
                os.link(quarantine, artifact.path, follow_symlinks=False)
            except FileExistsError as error:
                raise RuntimeError(
                    f"modified profile artifact preserved at {quarantine}; "
                    "public path was concurrently claimed"
                ) from error
            quarantine.unlink()
            moved = False
            raise RuntimeError("refusing to remove a modified profile artifact")
        if not _unlink_if_identity(quarantine, expected):
            raise RuntimeError(
                f"profile artifact quarantine changed; evidence retained at {quarantine}"
            )
        moved = False
        return True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(quarantine_descriptor)
        if not moved:
            try:
                _unlink_if_identity(quarantine, placeholder_identity)
            except OSError:
                pass


__all__ = [
    "MAX_TRACE_BYTES",
    "TRACE_FORMAT",
    "TRACE_SUFFIX",
    "ProfilingUnavailable",
    "TraceArtifact",
    "profile_scope",
    "profile_trace_path",
    "record_function_scope",
    "remove_trace_artifact",
    "validate_trace_artifact_path",
    "write_trace_artifact",
]
