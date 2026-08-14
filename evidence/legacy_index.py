"""Strict catalog contract for retained checkout-only benchmark evidence."""

from __future__ import annotations

import json
import math
import re
import stat
import subprocess
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath

INDEX_RELATIVE_PATH = "bench/results/index.json"
INDEX_SCHEMA_VERSION = 1
RESULTS_ROOT = PurePosixPath("bench/results")

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:[._+-][a-z0-9]+)*$")
_VERDICTS = frozenset(
    {
        "accepted-deviation",
        "diagnostic",
        "discarded",
        "fail",
        "not-applicable",
        "pass",
        "reference",
    }
)
_INDEX_FIELDS = frozenset({"artifacts", "schema_version"})
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_path",
        "commit",
        "date",
        "gate_id",
        "hardware",
        "summary_path",
        "verdict",
    }
)


class ResultsIndexError(ValueError):
    """The retained-results catalog is malformed or stale."""


def _fail(message: str) -> ResultsIndexError:
    return ResultsIndexError(f"invalid benchmark results index: {message}")


@dataclass(frozen=True, slots=True)
class ResultArtifact:
    """Discovery metadata for one Git-tracked top-level result artifact."""

    artifact_path: str
    commit: str | None
    date: str | None
    gate_id: str | None
    hardware: str | None
    summary_path: str | None
    verdict: str | None

    @classmethod
    def from_mapping(cls, value: object, *, position: int) -> ResultArtifact:
        if not isinstance(value, dict):
            raise _fail(f"artifacts[{position}] must be an object")
        fields = set(value)
        missing = sorted(_ARTIFACT_FIELDS - fields)
        unknown = sorted(fields - _ARTIFACT_FIELDS)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unknown:
                details.append(f"unknown {', '.join(unknown)}")
            raise _fail(f"artifacts[{position}] has {'; '.join(details)}")

        artifact_path = _artifact_path(value["artifact_path"], position=position)
        commit = _optional_commit(value["commit"], position=position)
        recorded_date = _optional_date(value["date"], position=position)
        gate_id = _optional_slug(value["gate_id"], field="gate_id", position=position)
        hardware = _optional_slug(
            value["hardware"], field="hardware", position=position
        )
        summary_path = _summary_path(
            value["summary_path"], artifact_path=artifact_path, position=position
        )
        verdict = value["verdict"]
        if verdict is not None and (
            not isinstance(verdict, str) or verdict not in _VERDICTS
        ):
            raise _fail(
                f"artifacts[{position}].verdict must be null or one of "
                + ", ".join(sorted(_VERDICTS))
            )
        return cls(
            artifact_path=artifact_path,
            commit=commit,
            date=recorded_date,
            gate_id=gate_id,
            hardware=hardware,
            summary_path=summary_path,
            verdict=verdict,
        )


@dataclass(frozen=True, slots=True)
class ResultsIndex:
    """A validated, path-sorted retained-results catalog."""

    artifacts: tuple[ResultArtifact, ...]
    schema_version: int = INDEX_SCHEMA_VERSION


def _artifact_path(value: object, *, position: int) -> str:
    if not isinstance(value, str):
        raise _fail(f"artifacts[{position}].artifact_path must be a string")
    path = PurePosixPath(value)
    if (
        value != path.as_posix()
        or PureWindowsPath(value).drive
        or "\\" in value
        or len(path.parts) != 3
        or path.parts[:2] != RESULTS_ROOT.parts
        or path.name in {"", ".", "..", "index.json"}
    ):
        raise _fail(
            f"artifacts[{position}].artifact_path must be one top-level path "
            "beneath bench/results"
        )
    return value


def _summary_path(
    value: object, *, artifact_path: str, position: int
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _fail(f"artifacts[{position}].summary_path must be a string or null")
    path = PurePosixPath(value)
    artifact = PurePosixPath(artifact_path)
    if (
        value != path.as_posix()
        or PureWindowsPath(value).drive
        or "\\" in value
        or len(path.parts) <= len(artifact.parts)
        or path.parts[: len(artifact.parts)] != artifact.parts
    ):
        raise _fail(
            f"artifacts[{position}].summary_path must be a descendant of "
            f"{artifact_path}"
        )
    return value


def _optional_commit(value: object, *, position: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise _fail(
            f"artifacts[{position}].commit must be a full lowercase Git SHA or null"
        )
    return value


def _optional_date(value: object, *, position: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise _fail(f"artifacts[{position}].date must be an ISO date or null")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise _fail(f"artifacts[{position}].date is not a real calendar date") from error
    if parsed.isoformat() != value:
        raise _fail(f"artifacts[{position}].date must be canonical ISO format")
    return value


def _optional_slug(
    value: object, *, field: str, position: int
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SLUG_RE.fullmatch(value) is None:
        raise _fail(
            f"artifacts[{position}].{field} must be a lowercase slug or null"
        )
    return value


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise _fail(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _fail(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except ResultsIndexError:
        raise
    except (RecursionError, UnicodeError, ValueError) as error:
        raise _fail(f"cannot parse JSON: {error}") from error

    stack = [parsed]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise _fail("JSON contains a non-finite number")
    return parsed


def parse_results_index(text: str) -> ResultsIndex:
    """Parse and validate a results index without consulting a checkout."""

    payload = _strict_json_loads(text)
    if not isinstance(payload, dict):
        raise _fail("top-level value must be an object")
    fields = set(payload)
    missing = sorted(_INDEX_FIELDS - fields)
    unknown = sorted(fields - _INDEX_FIELDS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise _fail("top-level object has " + "; ".join(details))
    if type(payload["schema_version"]) is not int or (
        payload["schema_version"] != INDEX_SCHEMA_VERSION
    ):
        raise _fail(
            f"schema_version must be exactly {INDEX_SCHEMA_VERSION}, got "
            f"{payload['schema_version']!r}"
        )
    raw_artifacts = payload["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise _fail("artifacts must be a non-empty array")
    artifacts = tuple(
        ResultArtifact.from_mapping(value, position=position)
        for position, value in enumerate(raw_artifacts)
    )
    paths = [artifact.artifact_path for artifact in artifacts]
    if paths != sorted(paths):
        raise _fail("artifact_path values must be sorted")
    duplicates = sorted(path for path in set(paths) if paths.count(path) > 1)
    if duplicates:
        raise _fail("duplicate artifact paths: " + ", ".join(duplicates))
    return ResultsIndex(artifacts=artifacts)


def render_results_index(index: ResultsIndex) -> str:
    """Render the one canonical, review-friendly JSON representation."""

    payload = {
        "artifacts": [asdict(artifact) for artifact in index.artifacts],
        "schema_version": index.schema_version,
    }
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def load_results_index(path: str | Path, *, require_canonical: bool = True) -> ResultsIndex:
    """Load an index from disk, optionally requiring canonical bytes."""

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _fail(f"cannot read {path}: {error}") from error
    index = parse_results_index(text)
    if require_canonical and text != render_results_index(index):
        raise _fail(f"{path} is not canonical JSON")
    return index


def _tracked_results(root: Path) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--", RESULTS_ROOT.as_posix()],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _fail(f"cannot enumerate Git-tracked result artifacts: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _fail(f"git ls-files failed: {detail or completed.returncode}")
    try:
        decoded = completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise _fail("Git returned a non-UTF-8 result path") from error
    paths = decoded.split("\0")
    if paths and paths[-1] == "":
        paths.pop()
    if not paths:
        raise _fail("Git tracks no paths beneath bench/results")
    if len(paths) != len(set(paths)):
        raise _fail("git ls-files returned duplicate paths")
    return tuple(paths)


def _require_safe_checkout_path(root: Path, relative: str, *, kind: str) -> Path:
    pure = PurePosixPath(relative)
    if relative != pure.as_posix() or PureWindowsPath(relative).drive:
        raise _fail(f"tracked {kind} path is not canonical: {relative!r}")
    path = root
    for part in pure.parts:
        path = path / part
        try:
            metadata = path.lstat()
        except OSError as error:
            raise _fail(f"tracked {kind} path is missing: {relative}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise _fail(
                f"tracked {kind} path must not contain a symlink: {relative}"
            )
    try:
        path.resolve(strict=True).relative_to(
            root.joinpath(*RESULTS_ROOT.parts).resolve(strict=True)
        )
    except (OSError, ValueError) as error:
        raise _fail(f"tracked {kind} path escapes bench/results: {relative}") from error
    return path


def validate_results_repository(root: str | Path) -> ResultsIndex:
    """Validate the catalog against Git-tracked top-level checkout artifacts.

    Ignored and untracked runtime output is deliberately outside this catalog.
    Historical artifact contents remain owned by their gate-specific verifiers;
    this function validates discovery metadata and tracked-path coverage only.
    """

    unresolved_root = Path(root)
    try:
        root = unresolved_root.resolve(strict=True)
    except OSError as error:
        raise _fail(f"repository root does not exist: {unresolved_root}") from error
    if not root.is_dir():
        raise _fail(f"repository root is not a directory: {root}")
    index_relative = PurePosixPath(INDEX_RELATIVE_PATH)
    index_path = _require_safe_checkout_path(
        root, index_relative.as_posix(), kind="index"
    )
    index = load_results_index(index_path)
    tracked = _tracked_results(root)
    if INDEX_RELATIVE_PATH not in tracked:
        raise _fail(f"{INDEX_RELATIVE_PATH} is not Git-tracked")

    tracked_artifacts: set[str] = set()
    tracked_set = set(tracked)
    for relative in tracked:
        path = PurePosixPath(relative)
        if len(path.parts) < 3 or path.parts[:2] != RESULTS_ROOT.parts:
            raise _fail(f"Git returned an invalid result path: {relative!r}")
        if relative == INDEX_RELATIVE_PATH:
            continue
        candidate = _require_safe_checkout_path(root, relative, kind="artifact")
        if not candidate.is_file():
            raise _fail(f"tracked artifact is not a regular file: {relative}")
        tracked_artifacts.add(PurePosixPath(*path.parts[:3]).as_posix())

    indexed_artifacts = {artifact.artifact_path for artifact in index.artifacts}
    missing = sorted(tracked_artifacts - indexed_artifacts)
    stale = sorted(indexed_artifacts - tracked_artifacts)
    if missing or stale:
        details = []
        if missing:
            details.append("missing records: " + ", ".join(missing))
        if stale:
            details.append("stale records: " + ", ".join(stale))
        raise _fail("tracked artifact coverage differs; " + "; ".join(details))

    for artifact in index.artifacts:
        path = _require_safe_checkout_path(
            root, artifact.artifact_path, kind="artifact root"
        )
        if path.is_dir():
            if artifact.summary_path is None:
                raise _fail(
                    f"directory artifact requires summary_path: {artifact.artifact_path}"
                )
            summary = _require_safe_checkout_path(
                root, artifact.summary_path, kind="summary"
            )
            if artifact.summary_path not in tracked_set or not summary.is_file():
                raise _fail(
                    f"summary_path must name a tracked regular file: "
                    f"{artifact.summary_path}"
                )
        elif path.is_file():
            if artifact.summary_path is not None:
                raise _fail(
                    f"flat artifact must use null summary_path: {artifact.artifact_path}"
                )
        else:
            raise _fail(
                f"artifact root must be a regular file or directory: "
                f"{artifact.artifact_path}"
            )
    return index
