"""Versioned links between formal verification evidence artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from evidence import read_json_object, sha256_file

FORMAL_RECORD_SCHEMA_VERSION = 1
_ACCEPTED_CORRECTNESS_VERDICTS = frozenset({"accepted-deviation", "pass"})
_GATE_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FormalEvidenceError(ValueError):
    """Formal verification evidence is missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class CorrectnessReference:
    """An immutable reference to one accepted correctness record."""

    gate_id: str
    artifact_path: str
    sha256: str

    def as_mapping(self) -> dict[str, str]:
        return asdict(self)


def _artifact_path(path: Path) -> str:
    value = path.as_posix()
    pure = PurePosixPath(value)
    if (
        value != pure.as_posix()
        or PureWindowsPath(value).drive
        or ".." in pure.parts
    ):
        raise FormalEvidenceError("correctness artifact path must be canonical")
    return value


def load_correctness_reference(path: str | Path) -> CorrectnessReference:
    """Load and hash an accepted formal correctness record."""

    source = Path(path)
    payload = read_json_object(source, error_type=FormalEvidenceError)
    if payload.get("schema_version") != FORMAL_RECORD_SCHEMA_VERSION:
        raise FormalEvidenceError(
            "correctness artifact has an unsupported schema_version"
        )
    gate_id = payload.get("gate_id")
    if not isinstance(gate_id, str) or _GATE_ID_RE.fullmatch(gate_id) is None:
        raise FormalEvidenceError("correctness artifact has an invalid gate_id")
    if payload.get("kind") != "correctness":
        raise FormalEvidenceError("referenced artifact is not correctness evidence")
    if payload.get("verdict") not in _ACCEPTED_CORRECTNESS_VERDICTS:
        raise FormalEvidenceError("correctness artifact is not accepted")
    return CorrectnessReference(
        gate_id=gate_id,
        artifact_path=_artifact_path(source),
        sha256=sha256_file(source),
    )


def bind_correctness(
    payload: Mapping[str, object],
    correctness_artifact: str | Path,
) -> dict[str, object]:
    """Bind a performance record to exact accepted correctness bytes."""

    if payload.get("kind") != "performance":
        raise FormalEvidenceError("only performance evidence binds correctness")
    if "correctness_reference" in payload:
        raise FormalEvidenceError("performance evidence already has a correctness link")
    reference = load_correctness_reference(correctness_artifact)
    return {**dict(payload), "correctness_reference": reference.as_mapping()}


def validate_correctness_reference(value: object) -> CorrectnessReference:
    """Validate the closed correctness-reference object in retained evidence."""

    if not isinstance(value, dict):
        raise FormalEvidenceError("correctness_reference must be an object")
    expected = {"artifact_path", "gate_id", "sha256"}
    if set(value) != expected:
        raise FormalEvidenceError("correctness_reference fields are not exact")
    gate_id = value["gate_id"]
    artifact_path = value["artifact_path"]
    digest = value["sha256"]
    if not isinstance(gate_id, str) or _GATE_ID_RE.fullmatch(gate_id) is None:
        raise FormalEvidenceError("correctness_reference gate_id is invalid")
    if not isinstance(artifact_path, str):
        raise FormalEvidenceError("correctness_reference artifact_path is invalid")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise FormalEvidenceError("correctness_reference sha256 is invalid")
    _artifact_path(Path(artifact_path))
    return CorrectnessReference(
        gate_id=gate_id,
        artifact_path=artifact_path,
        sha256=digest,
    )


__all__ = [
    "CorrectnessReference",
    "FORMAL_RECORD_SCHEMA_VERSION",
    "FormalEvidenceError",
    "bind_correctness",
    "load_correctness_reference",
    "validate_correctness_reference",
]
