"""Append-only, source-bound history for installed-suite scoreboards.

Each JSONL row is a complete snapshot: the persisted ``run.json`` metadata,
the generated scoreboard, and the benchmark/target pair bindings.  Rows form a
SHA-256 chain and are serialized canonically.  A stable root-owned lock guards
copy-on-write replacement, giving concurrent writers a logical append-only
operation without exposing a torn final line after a crash.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from kairyu.bench.sampling import PROTOCOL_ID as SAMPLING_PROTOCOL_ID

INDEX_SCHEMA_VERSION = 1
INDEX_NAME = "scoreboards.jsonl"
LOCK_NAME = ".scoreboards.jsonl.lock"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REDACTED_EXTRA_BODY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESOLVED_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_KEY_RE = re.compile(
    r"(?:^|[-_])(?:password|passwd|secret|token|credential|authorization|api[-_]?key)(?:$|[-_])",
    re.IGNORECASE,
)
_PAIR_STATUSES = frozenset({"completed", "partial", "skipped", "failed"})
_CROSS_RUN_POLICIES = frozenset(
    {"allowed", "withheld_unresolved_runtime", "withheld_unpinned_execution"}
)
_MAX_SAFE_INTEGER = (1 << 53) - 1
_COMMON_EVALUATION_RESOURCES = frozenset(
    {
        ("kairyu.bench.adapters", "base.py"),
        ("kairyu.bench", "aggregate.py"),
        ("kairyu.bench", "cache.py"),
        ("kairyu.bench", "runner.py"),
        ("kairyu.bench", "targets.py"),
        ("kairyu.bench", "types.py"),
    }
)
_UNRESOLVED_RUNTIME_ADAPTERS = frozenset(
    {"swe-bench-pro", "terminal-bench", "tau-bench-banking"}
)
_NON_DATASET_ADAPTERS = frozenset({"terminal-bench", "tau-bench-banking"})
_CONTENT_ADDRESSED_EXECUTION_ADAPTERS = frozenset(
    {"livecodebench", "livecodebench-pro", "scicode"}
)
_ENTRY_FIELDS = frozenset(
    {
        "index_schema_version",
        "git_commit",
        "fingerprint",
        "run_id",
        "suite",
        "run",
        "scoreboard",
        "pairs",
        "previous_record_sha256",
        "record_sha256",
    }
)
_PAIR_FIELDS = frozenset(
    {
        "benchmark",
        "target",
        "run_fingerprint",
        "source_identity",
        "status",
        "score",
        "n",
        "n_scored",
        "reason",
        "comparable",
        "incomparable_reasons",
        "cross_run_policy",
        "cross_run_reason",
        "confidence_interval",
        "pair_sha256",
    }
)
_SOURCE_IDENTITY_FIELDS = (
    "git_commit",
    "git_commit_role",
    "source_kind",
    "source_tree_clean",
    "source_tree_sha256",
)
_RESOLVED_EXECUTION_FIELDS = (
    "resolved_image_id",
    "image_os",
    "image_architecture",
    "image_variant",
)
_FINGERPRINT_EXCLUSIONS = frozenset(
    {"run_id", "results_dir", "cache_dir", "rerun", "download", "progress"}
)
_SAMPLING_SENSITIVITY_FIELD = "sampling_sensitivity"
_SAMPLING_EVALUATION_RESOURCE = ("kairyu.bench", "sampling.py")


def _fail(message: str) -> ValueError:
    return ValueError(f"invalid scoreboard history: {message}")


def _validate_run_id(run_id: object) -> str:
    windows_drive = PureWindowsPath(run_id).drive if isinstance(run_id, str) else ""
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).is_absolute()
        or windows_drive
        or "/" in run_id
        or "\\" in run_id
    ):
        raise _fail(f"run id {run_id!r} is not one non-dot path component")
    return run_id


def _validate_json_value(value: object, *, path: str = "$") -> None:
    if value is None or isinstance(value, bool | str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise _fail(f"{path} contains non-JSON value {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    _validate_json_value(value)
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise _fail("record cannot be serialized as canonical JSON") from error


def _snapshot(value: object) -> Any:
    """Return a detached JSON value after strict canonical validation."""
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def stable_environment_identity(environment: object) -> dict | None:
    """Strip only volatile observation fields from a run environment.

    Source, interpreter, package, and execution-policy identity remains binding.
    A timestamp and the execution runner's momentary availability do not: they
    may legitimately differ when an interrupted run is resumed.
    """
    if not isinstance(environment, dict):
        return None
    stable = _snapshot(environment)
    stable.pop("created_at", None)
    execution = stable.get("execution")
    if isinstance(execution, dict):
        execution.pop("available", None)
        execution.pop("availability_detail", None)
    return stable


def resumable_environment_identity_equal(persisted: object, current: object) -> bool:
    """Compare stable runtime identity without making downtime a false drift.

    A successful Docker probe binds the resolved image/platform exactly. If
    either observation is explicitly unavailable, those fields are unknown for
    that observation and the immutable requested image/config still remains in
    the stable identity. Two available observations must match every resolved
    field.
    """

    left = stable_environment_identity(persisted)
    right = stable_environment_identity(current)
    if left is None or right is None:
        return False
    left_observed = persisted.get("execution") if isinstance(persisted, dict) else None
    right_observed = current.get("execution") if isinstance(current, dict) else None
    if (
        isinstance(left_observed, dict)
        and left_observed.get("available") is False
    ) or (
        isinstance(right_observed, dict)
        and right_observed.get("available") is False
    ):
        for identity in (left, right):
            execution = identity.get("execution")
            if isinstance(execution, dict):
                for field in _RESOLVED_EXECUTION_FIELDS:
                    execution.pop(field, None)
    return left == right


def _validate_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{field} must be a lowercase SHA-256")
    return value


def _validate_commit(value: object) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise _fail("git_commit must be a full lowercase Git object id")
    return value


def _nonempty_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field} must be a non-empty string")
    return value


def _validate_evaluation_resources(value: object, *, adapter: str) -> None:
    if not isinstance(value, list) or not value:
        raise _fail(f"benchmark {adapter!r} must disclose evaluator resources")
    seen: set[tuple[str, str]] = set()
    for resource in value:
        if not isinstance(resource, dict) or set(resource) != {"package", "resource", "sha256"}:
            raise _fail(f"benchmark {adapter!r} has malformed evaluator resource identity")
        package = _nonempty_name(resource["package"], field="evaluator resource package")
        name = _nonempty_name(resource["resource"], field="evaluator resource name")
        _validate_digest(resource["sha256"], field="evaluator resource sha256")
        key = (package, name)
        if key in seen:
            raise _fail(f"benchmark {adapter!r} repeats evaluator resource {package}/{name}")
        seen.add(key)
    missing = sorted(_COMMON_EVALUATION_RESOURCES - seen)
    if missing:
        rendered = ", ".join(f"{package}/{resource}" for package, resource in missing)
        raise _fail(f"benchmark {adapter!r} omits shared score-bearing resources: {rendered}")
    if not any(
        package.startswith("kairyu.bench.adapters") and resource != "base.py"
        for package, resource in seen
    ):
        raise _fail(f"benchmark {adapter!r} omits its adapter implementation resource")


def _validate_evaluation_distributions(value: object, *, adapter: str) -> dict[str, bool]:
    if not isinstance(value, list):
        raise _fail(f"benchmark {adapter!r} must disclose evaluator distributions")
    installed: dict[str, bool] = {}
    for distribution in value:
        if not isinstance(distribution, dict):
            raise _fail(f"benchmark {adapter!r} has malformed evaluator distribution identity")
        name = _nonempty_name(distribution.get("name"), field="evaluator distribution name")
        if name in installed:
            raise _fail(f"benchmark {adapter!r} repeats evaluator distribution {name!r}")
        present = distribution.get("installed")
        if type(present) is not bool:
            raise _fail(f"benchmark {adapter!r} distribution {name!r} has invalid installed state")
        if not present:
            if set(distribution) != {"name", "installed"}:
                raise _fail(
                    f"benchmark {adapter!r} absent distribution {name!r} has unexpected fields"
                )
            installed[name] = False
            continue
        expected = {
            "name",
            "installed",
            "version",
            "content_verified",
            "file_count",
            "content_sha256",
        }
        if set(distribution) != expected or distribution.get("content_verified") is not True:
            raise _fail(
                f"benchmark {adapter!r} installed distribution {name!r} is not content verified"
            )
        _nonempty_name(distribution.get("version"), field="evaluator distribution version")
        count = distribution.get("file_count")
        if type(count) is not int or not 0 < count <= _MAX_SAFE_INTEGER:
            raise _fail(
                f"benchmark {adapter!r} distribution {name!r} has invalid file_count"
            )
        _validate_digest(
            distribution.get("content_sha256"), field="evaluator distribution content_sha256"
        )
        installed[name] = True
    return installed


def _validate_evaluation_executables(
    value: object,
    *,
    adapter: str,
    distributions: dict[str, bool],
) -> None:
    if not isinstance(value, list):
        raise _fail(f"benchmark {adapter!r} must disclose evaluator executables")
    seen: set[str] = set()
    for executable in value:
        if not isinstance(executable, dict):
            raise _fail(f"benchmark {adapter!r} has malformed evaluator executable identity")
        name = _nonempty_name(executable.get("name"), field="evaluator executable name")
        if name in seen:
            raise _fail(f"benchmark {adapter!r} repeats evaluator executable {name!r}")
        seen.add(name)
        found = executable.get("found")
        if type(found) is not bool:
            raise _fail(f"benchmark {adapter!r} executable {name!r} has invalid found state")
        if not found:
            if set(executable) != {"name", "found"}:
                raise _fail(
                    f"benchmark {adapter!r} absent executable {name!r} has unexpected fields"
                )
            continue
        expected = {
            "name",
            "found",
            "content_verified",
            "content_sha256",
            "owned_by_distribution",
        }
        if set(executable) != expected or executable.get("content_verified") is not True:
            raise _fail(
                f"benchmark {adapter!r} executable {name!r} is not content verified"
            )
        _validate_digest(
            executable.get("content_sha256"), field="evaluator executable content_sha256"
        )
        owners = executable.get("owned_by_distribution")
        if (
            not isinstance(owners, list)
            or not owners
            or any(not isinstance(owner, str) or not owner for owner in owners)
            or len(owners) != len(set(owners))
            or any(distributions.get(owner) is not True for owner in owners)
        ):
            raise _fail(
                f"benchmark {adapter!r} executable {name!r} is PATH-shadowed or lacks "
                "verified distribution ownership"
            )


def _validate_asset_identities(value: object, *, adapter: str) -> None:
    if not isinstance(value, list):
        raise _fail(f"benchmark {adapter!r} assets must be an exact identity list")
    paths: list[str] = []
    for asset in value:
        if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
            raise _fail(f"benchmark {adapter!r} has malformed asset identity")
        path = asset.get("path")
        if not isinstance(path, str):
            raise _fail(f"benchmark {adapter!r} has malformed asset path")
        relative = PurePosixPath(path)
        if (
            not path
            or "\x00" in path
            or relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] != "assets"
            or relative.as_posix() != path
        ):
            raise _fail(f"benchmark {adapter!r} has unsafe or non-canonical asset path")
        _validate_digest(asset.get("sha256"), field="asset sha256")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise _fail(f"benchmark {adapter!r} asset identities must be unique and sorted")


def _dataset_identity_is_content_bound(identity: dict, *, adapter: str) -> bool:
    """Validate cache identity variants and return whether bytes are bound."""
    dataset = identity.get("dataset")
    revision = identity.get("revision")
    if adapter in _NON_DATASET_ADAPTERS:
        if dataset is not None or revision is not None:
            raise _fail(f"benchmark {adapter!r} must not claim a dataset cache identity")
    else:
        if not isinstance(dataset, str) or not dataset:
            raise _fail(f"benchmark {adapter!r} has malformed dataset identity")
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise _fail(f"benchmark {adapter!r} has malformed dataset revision")
        provenance = identity.get("history_provenance")
        if (
            isinstance(provenance, dict)
            and provenance.get("complete") is True
            and revision is None
        ):
            raise _fail(f"benchmark {adapter!r} source-complete dataset requires a revision")

    unavailable_present = "unavailable" in identity
    if unavailable_present:
        if identity.get("unavailable") is not True:
            raise _fail(f"benchmark {adapter!r} has invalid unavailable state")
        if "sha256" in identity or "assets" in identity:
            raise _fail(
                f"benchmark {adapter!r} cannot be unavailable and content-bound together"
            )
        return False

    has_digest = "sha256" in identity
    has_assets = "assets" in identity
    if has_digest != has_assets:
        raise _fail(f"benchmark {adapter!r} has incomplete cache content identity")
    if not has_digest:
        raise _fail(f"benchmark {adapter!r} cache availability is undisclosed")
    _validate_digest(identity.get("sha256"), field="dataset manifest sha256")
    _validate_asset_identities(identity.get("assets"), adapter=adapter)
    return True


def _validate_methodology_identity(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise _fail("run identity must disclose at least one benchmark adapter")
    names: set[str] = set()
    for identity in value:
        if not isinstance(identity, dict):
            raise _fail("run identity adapter entries must be objects")
        name = _nonempty_name(identity.get("name"), field="benchmark adapter name")
        if name in names:
            raise _fail(f"run identity repeats benchmark adapter {name!r}")
        names.add(name)
        provenance = identity.get("history_provenance")
        if not isinstance(provenance, dict) or set(provenance) != {
            "complete",
            "reason",
            "requires_content_addressed_execution",
        }:
            raise _fail(f"benchmark {name!r} has malformed history provenance policy")
        if type(provenance.get("complete")) is not bool or type(
            provenance.get("requires_content_addressed_execution")
        ) is not bool:
            raise _fail(f"benchmark {name!r} history provenance flags must be booleans")
        reason = provenance.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise _fail(f"benchmark {name!r} history provenance reason is malformed")
        if name in _CONTENT_ADDRESSED_EXECUTION_ADAPTERS and provenance.get(
            "requires_content_addressed_execution"
        ) is not True:
            raise _fail(f"benchmark {name!r} must disclose its code-execution requirement")
        protocol = identity.get("evaluation_protocol")
        if not isinstance(protocol, dict) or set(protocol) != {
            "resources",
            "distributions",
            "executables",
        }:
            raise _fail(f"benchmark {name!r} has malformed evaluation_protocol identity")
        _validate_evaluation_resources(protocol["resources"], adapter=name)
        distributions = _validate_evaluation_distributions(
            protocol["distributions"], adapter=name
        )
        _validate_evaluation_executables(
            protocol["executables"], adapter=name, distributions=distributions
        )
        _dataset_identity_is_content_bound(identity, adapter=name)


def _pair_has_sampling_evidence(value: object) -> bool:
    """Detect fresh pair evidence that depends on the sampling evaluator."""

    from kairyu.bench.types import PairResult

    try:
        pair = PairResult.model_validate(value)
    except (TypeError, ValueError):
        return False  # the ordinary pair validator reports the schema failure later
    if isinstance(pair.methodology.get("sampling"), dict):
        return True
    if any(item.sampling_attempts is not None for item in pair.items):
        return True
    return any(name.startswith("sampling_") for name in pair.metrics)


def _scoreboard_has_sampling_claim(scoreboard: dict) -> bool:
    cells = scoreboard.get("cells")
    if not isinstance(cells, dict):
        return False
    return any(
        isinstance(cell, dict) and _SAMPLING_SENSITIVITY_FIELD in cell
        for row in cells.values()
        if isinstance(row, dict)
        for cell in row.values()
    )


def _require_sampling_evaluation_resource(adapters: Sequence[object]) -> None:
    """Require sampling.py only for records that use the post-#371 protocol."""

    package, resource = _SAMPLING_EVALUATION_RESOURCE
    for identity in adapters:
        if not isinstance(identity, dict):  # validated by the caller
            continue
        protocol = identity.get("evaluation_protocol")
        resources = protocol.get("resources") if isinstance(protocol, dict) else None
        retained = set()
        if isinstance(resources, list):
            retained = {
                (entry.get("package"), entry.get("resource"))
                for entry in resources
                if isinstance(entry, dict)
            }
        if _SAMPLING_EVALUATION_RESOURCE not in retained:
            name = identity.get("name")
            raise _fail(
                f"benchmark {name!r} omits shared sampling evaluator resource "
                f"{package}/{resource}"
            )


def cross_run_policies(
    adapters: Sequence[Mapping[str, object]], environment: Mapping[str, object]
) -> dict[str, tuple[str, str | None]]:
    """Derive the non-forgeable per-adapter cross-run delta policy."""

    execution = environment.get("execution")
    resolved_execution = (
        isinstance(execution, dict)
        and execution.get("runner") == "docker"
        and execution.get("available") is True
        and isinstance(execution.get("resolved_image_id"), str)
        and _RESOLVED_IMAGE_ID_RE.fullmatch(execution["resolved_image_id"]) is not None
        and isinstance(execution.get("image_os"), str)
        and bool(execution["image_os"])
        and isinstance(execution.get("image_architecture"), str)
        and bool(execution["image_architecture"])
        and (
            execution.get("image_variant") is None
            or (
                isinstance(execution.get("image_variant"), str)
                and bool(execution["image_variant"])
            )
        )
    )
    policies: dict[str, tuple[str, str | None]] = {}
    for identity in adapters:
        name = identity.get("name")
        provenance = identity.get("history_provenance")
        if not isinstance(name, str) or not isinstance(provenance, Mapping):
            continue
        if name in _UNRESOLVED_RUNTIME_ADAPTERS or provenance.get("complete") is not True:
            policies[name] = (
                "withheld_unresolved_runtime",
                "runtime dataset, image, or executable content is unresolved",
            )
        elif provenance.get("requires_content_addressed_execution") is True and not (
            resolved_execution
        ):
            policies[name] = (
                "withheld_unpinned_execution",
                "code execution lacks a resolved immutable image and platform identity",
            )
        else:
            policies[name] = ("allowed", None)
    return policies


def _validate_execution_binding(config: dict, environment: dict) -> None:
    configured = config.get("execution")
    observed = environment.get("execution")
    if not isinstance(configured, dict) or not isinstance(observed, dict):
        raise _fail("scoreboard history requires configured and observed execution identities")
    required = {"runner", "image", "cpus", "pids_limit", "disk_mb", "pull_policy"}
    if not required <= set(configured):
        raise _fail("run config execution identity is incomplete")
    runner = configured.get("runner")
    if runner not in {"local", "docker"} or observed.get("runner") != runner:
        raise _fail("configured execution runner disagrees with observed runtime")
    if runner == "local":
        if {
            "image": configured.get("image"),
            "cpus": configured.get("cpus"),
            "pids_limit": configured.get("pids_limit"),
            "disk_mb": configured.get("disk_mb"),
            "pull_policy": configured.get("pull_policy"),
        } != {
            "image": None,
            "cpus": 1.0,
            "pids_limit": 64,
            "disk_mb": 256,
            "pull_policy": "never",
        }:
            raise _fail("local execution config is not the canonical runtime policy")
        return
    for field in ("image", "cpus", "pids_limit", "disk_mb", "pull_policy"):
        if observed.get(field) != configured.get(field):
            raise _fail(f"configured Docker execution {field} disagrees with observed runtime")


def validate_run_metadata(metadata: object, *, require_clean_source: bool) -> dict:
    """Validate and detach the persisted ``run.json`` metadata."""
    if not isinstance(metadata, dict):
        raise _fail("run metadata must be an object")
    required = {"fingerprint", "identity", "config", "environment", "run_id"}
    missing = sorted(required - set(metadata))
    if missing:
        raise _fail(f"run metadata is missing {', '.join(missing)}")
    run = _snapshot(metadata)
    run_id = _validate_run_id(run["run_id"])
    fingerprint = _validate_digest(run["fingerprint"], field="fingerprint")
    identity = run["identity"]
    config = run["config"]
    environment = run["environment"]
    if not isinstance(identity, dict):
        raise _fail("run identity must be an object")
    if not isinstance(config, dict):
        raise _fail("run config must be an object")
    _validate_tracked_config(config)
    if config.get("offline_fixtures") is not False:
        raise _fail("scoreboard history requires offline_fixtures to be explicitly false")
    _validate_methodology_identity(identity.get("adapters"))
    if not isinstance(environment, dict):
        raise _fail("run environment must be an object")
    immutable_config = {
        key: value for key, value in config.items() if key not in _FINGERPRINT_EXCLUSIONS
    }
    if identity.get("config") != immutable_config:
        raise _fail("run identity config does not match the disclosed config")
    if _sha256(identity) != fingerprint:
        raise _fail("fingerprint does not hash the complete run identity")
    if require_clean_source:
        _validate_execution_binding(config, environment)
        cross_run_policies(identity["adapters"], environment)
        commit = _validate_commit(environment.get("git_commit"))
        if environment.get("git_commit_role") != "local_benchmark_harness":
            raise _fail("git_commit_role must identify the local benchmark harness")
        if environment.get("source_kind") != "git-checkout":
            raise _fail("scoreboard history requires a git-checkout source")
        if environment.get("source_tree_clean") is not True:
            raise _fail("scoreboard history requires a clean tracked source tree")
        _validate_digest(environment.get("source_tree_sha256"), field="source_tree_sha256")
        if not isinstance(environment.get("python"), str) or not environment["python"]:
            raise _fail("scoreboard history requires a Python runtime identity")
        execution = environment.get("execution")
        if not isinstance(execution, dict) or not {
            key: value
            for key, value in execution.items()
            if key not in {"available", "availability_detail"}
        }:
            raise _fail("scoreboard history requires an execution runtime identity")
        if type(execution.get("available")) is not bool:
            raise _fail("execution.available must be a boolean")
        availability_detail = execution.get("availability_detail")
        if availability_detail is not None and not isinstance(availability_detail, str):
            raise _fail("execution.availability_detail must be a string or null")
        if run_id != metadata.get("run_id") or commit != environment["git_commit"]:
            raise _fail("run metadata changed during source validation")
    return run


def _validate_tracked_config(value: object, *, path: str = "config") -> None:
    """Reject credential-bearing literals from Git-trackable snapshots."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_tracked_config(item, path=f"{path}[{index}]")
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        field_path = f"{path}.{key}"
        if key == "extra_body_json":
            if item is not None and (
                not isinstance(item, str) or _REDACTED_EXTRA_BODY_RE.fullmatch(item) is None
            ):
                raise _fail(f"{field_path} must be an opaque SHA-256, not a retained request body")
            continue
        if key == "api_key_env":
            if item is not None and (
                not isinstance(item, str) or _ENV_NAME_RE.fullmatch(item) is None
            ):
                raise _fail(f"{field_path} must be an environment-variable name")
            continue
        if key != "api_key_env" and _SECRET_KEY_RE.search(key) and item is not None:
            raise _fail(f"{field_path} is a credential-capable field")
        if key == "base_url" and isinstance(item, str):
            parsed = urlsplit(item)
            if (
                "@" in parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise _fail(f"{field_path} contains URL credentials or parameters")
        _validate_tracked_config(item, path=field_path)


def scoreboard_key(run_metadata: object) -> tuple[str, str]:
    run = validate_run_metadata(run_metadata, require_clean_source=True)
    environment = run["environment"]
    return _validate_commit(environment["git_commit"]), run["fingerprint"]


def source_identity(environment: object) -> dict:
    """Return the exact source fields that must bind every stored pair."""
    if not isinstance(environment, dict):
        raise _fail("run environment must be an object")
    if any(field not in environment for field in _SOURCE_IDENTITY_FIELDS):
        raise _fail("run environment is missing its complete source identity")
    return _snapshot({field: environment[field] for field in _SOURCE_IDENTITY_FIELDS})


def _validate_name_list(value: object, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise _fail(f"scoreboard {field} must be a unique list of non-empty strings")
    return value


def _unit_number(value: object, *, field: str, allow_none: bool = True) -> int | float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(f"{field} must be {'null or ' if allow_none else ''}a number within [0, 1]")
    if isinstance(value, float) and not math.isfinite(value):
        raise _fail(f"{field} must be finite within [0, 1]")
    # Compare before any float conversion: JSON permits integers too large for
    # ``math.isfinite``/``float`` and malformed history must still fail closed.
    if value < 0 or value > 1:
        raise _fail(f"{field} must be within [0, 1]")
    return value


def _whole_count(value: object, *, field: str, allow_none: bool = True) -> int | None:
    if value is None and allow_none:
        return None
    if type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER:
        raise _fail(
            f"{field} must be {'null or ' if allow_none else ''}a non-negative safe integer"
        )
    return value


def _validate_confidence_interval(
    value: object,
    *,
    field: str,
    status: str,
    score: int | float | None,
    total: int | None,
    scored: int | None,
) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "method",
        "confidence",
        "successes",
        "trials",
        "lower",
        "upper",
    }:
        raise _fail(f"{field} must be null or a complete Wilson interval")
    if status != "completed" or score is None:
        raise _fail(f"{field} is only valid for a completed scored pair")
    if value.get("method") != "wilson":
        raise _fail(f"{field}.method must be wilson")
    confidence = _unit_number(
        value.get("confidence"), field=f"{field}.confidence", allow_none=False
    )
    if confidence != 0.95:
        raise _fail(f"{field}.confidence must be 0.95")
    successes = _whole_count(
        value.get("successes"), field=f"{field}.successes", allow_none=False
    )
    trials = _whole_count(value.get("trials"), field=f"{field}.trials", allow_none=False)
    assert successes is not None and trials is not None
    if trials == 0 or successes > trials or total != trials or scored != trials:
        raise _fail(f"{field} counts do not match the complete pair denominator")
    lower = _unit_number(value.get("lower"), field=f"{field}.lower", allow_none=False)
    upper = _unit_number(value.get("upper"), field=f"{field}.upper", allow_none=False)
    assert lower is not None and upper is not None
    if lower > upper or not lower <= score <= upper:
        raise _fail(f"{field} bounds do not contain the pair score")
    if not math.isclose(float(score), successes / trials, rel_tol=0.0, abs_tol=1e-12):
        raise _fail(f"{field} successes do not reproduce the pair score")
    return _snapshot(value)


def _cell_summary(cell: object, *, field: str) -> dict:
    if not isinstance(cell, dict):
        raise _fail(f"{field} is not an object")
    required = {
        "status",
        "score",
        "n",
        "n_scored",
        "confidence_interval",
        "reason",
        "comparable",
        "incomparable_reasons",
        "cross_run_policy",
        "cross_run_reason",
    }
    missing = sorted(required - set(cell))
    if missing:
        raise _fail(f"{field} is missing summary fields: {', '.join(missing)}")
    status = cell["status"]
    if not isinstance(status, str) or status not in _PAIR_STATUSES:
        raise _fail(f"{field} has an invalid status")
    if status == "failed":
        raise _fail(f"{field} is failed; history accepts only completed-run evidence")
    score = _unit_number(cell["score"], field=f"{field}.score")
    total = _whole_count(cell["n"], field=f"{field}.n")
    scored = _whole_count(cell["n_scored"], field=f"{field}.n_scored")
    if total is not None and scored is not None and scored > total:
        raise _fail(f"{field}.n_scored exceeds n")
    reason = cell["reason"]
    if reason is not None and not isinstance(reason, str):
        raise _fail(f"{field}.reason must be null or a string")
    comparable = cell["comparable"]
    if type(comparable) is not bool:
        raise _fail(f"{field}.comparable must be a boolean")
    reasons = cell["incomparable_reasons"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise _fail(f"{field}.incomparable_reasons must be a unique string list")
    interval = _validate_confidence_interval(
        cell["confidence_interval"],
        field=f"{field}.confidence_interval",
        status=status,
        score=score,
        total=total,
        scored=scored,
    )
    cross_run_policy = cell["cross_run_policy"]
    cross_run_reason = cell["cross_run_reason"]
    if (
        not isinstance(cross_run_policy, str)
        or cross_run_policy not in _CROSS_RUN_POLICIES
    ):
        raise _fail(f"{field}.cross_run_policy is invalid")
    if cross_run_policy == "allowed":
        if cross_run_reason is not None:
            raise _fail(f"{field}.cross_run_reason must be null when deltas are allowed")
    elif not isinstance(cross_run_reason, str) or not cross_run_reason:
        raise _fail(f"{field}.cross_run_reason is required when deltas are withheld")
    return {
        "status": status,
        "score": score,
        "n": total,
        "n_scored": scored,
        "reason": reason,
        "comparable": comparable,
        "incomparable_reasons": list(reasons),
        "confidence_interval": interval,
        "cross_run_policy": cross_run_policy,
        "cross_run_reason": cross_run_reason,
    }


def _scoreboard_matrix(scoreboard: dict) -> set[tuple[str, str]]:
    benchmarks = _validate_name_list(scoreboard.get("benchmarks"), field="benchmarks")
    targets = _validate_name_list(scoreboard.get("targets"), field="targets")
    cells = scoreboard.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(benchmarks):
        raise _fail("scoreboard cells do not match the declared benchmark rows")
    matrix: set[tuple[str, str]] = set()
    for benchmark in benchmarks:
        row = cells[benchmark]
        if not isinstance(row, dict) or set(row) != set(targets):
            raise _fail(f"scoreboard row {benchmark!r} does not match the declared targets")
        for target in targets:
            cell = row[target]
            _cell_summary(cell, field=f"scoreboard cell {benchmark!r}/{target!r}")
            matrix.add((benchmark, target))
    display_names = scoreboard.get("display_names")
    if (
        not isinstance(display_names, dict)
        or set(display_names) != set(benchmarks)
        or any(
            not isinstance(display_names[benchmark], str)
            or not display_names[benchmark]
            for benchmark in benchmarks
        )
    ):
        raise _fail(
            "scoreboard display_names must contain one non-empty string per benchmark row"
        )
    footnotes = scoreboard.get("footnotes")
    if not isinstance(footnotes, list) or any(
        not isinstance(note, str) for note in footnotes
    ):
        raise _fail("scoreboard footnotes must be a list of strings")
    return matrix


def _pair_result_binding(
    pair: object,
    *,
    declared_binary: bool | None = None,
    expected_sampling: dict[str, object] | None = None,
) -> dict:
    from kairyu.bench.aggregate import _binary_confidence_interval
    from kairyu.bench.types import PairResult, pair_result_evidence_error

    if isinstance(pair, Mapping):
        unknown = set(pair) - set(PairResult.model_fields)
        if unknown:
            raise _fail(f"pair result has unknown fields: {', '.join(sorted(unknown))}")
    try:
        result = PairResult.model_validate(pair)
    except (TypeError, ValueError) as error:
        raise _fail("pair result does not match PairResult schema") from error
    evidence_error = pair_result_evidence_error(
        result,
        expected_sampling=expected_sampling,
        require_history_safe_counts=True,
    )
    if evidence_error is not None:
        raise _fail(f"pair result contains invalid evidence: {evidence_error}")
    payload = result.model_dump(mode="json")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise _fail("pair result schema_version must be 1")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise _fail("pair result metrics must be an object")
    benchmark = payload.get("benchmark")
    summary_payload = {
        "status": payload.get("status"),
        "score": metrics.get("score"),
        "n": metrics.get("n_total"),
        "n_scored": metrics.get("n_scored"),
        "reason": payload.get("reason"),
        "comparable": payload.get("comparable"),
        "incomparable_reasons": payload.get("incomparable_reasons"),
        "cross_run_policy": payload.get("cross_run_policy"),
        "cross_run_reason": payload.get("cross_run_reason"),
        "confidence_interval": None,
    }
    # Reject malformed numeric evidence before aggregate helpers access
    # PairResult.score, which intentionally coerces normal metrics to float.
    _cell_summary(
        summary_payload,
        field=f"pair {benchmark!r}/{payload.get('target')!r}",
    )
    if declared_binary is None:
        from kairyu.bench.adapters import all_adapters

        adapter = all_adapters().get(benchmark) if isinstance(benchmark, str) else None
        binary_outcomes = bool(adapter is not None and adapter.info.binary_outcomes)
    elif type(declared_binary) is bool:
        binary_outcomes = declared_binary
    else:
        raise _fail("declared_binary must be a boolean when supplied")
    interval = _binary_confidence_interval(
        result,
        declared_binary=binary_outcomes,
    )
    summary_payload["confidence_interval"] = interval
    summary = _cell_summary(
        summary_payload,
        field=f"pair {benchmark!r}/{payload.get('target')!r}",
    )
    return {
        "benchmark": benchmark,
        "target": payload.get("target"),
        "run_fingerprint": payload.get("run_fingerprint"),
        "source_identity": payload.get("source_identity"),
        **summary,
        "pair_sha256": _sha256(payload),
    }


def _recomputed_sampling_sensitivity(pair: object, *, benchmark: str) -> dict | None:
    """Rebuild a fresh sampling claim without extending index schema 1.

    Sampling sensitivity is a derived scoreboard label rather than one of the
    schema-1 pair-binding fields.  Fresh history construction must nevertheless
    prove that label against the complete raw ``PairResult`` before discarding
    it from the durable scoreboard snapshot.
    """

    from kairyu.bench.adapters import all_adapters
    from kairyu.bench.aggregate import _sampling_sensitivity
    from kairyu.bench.types import PairResult

    try:
        result = PairResult.model_validate(pair)
    except (TypeError, ValueError) as error:  # pragma: no cover - bound just above
        raise _fail("pair result does not match PairResult schema") from error
    adapter = all_adapters().get(benchmark)
    declared_binary = bool(adapter is not None and adapter.info.binary_outcomes)
    return _sampling_sensitivity(result, declared_binary=declared_binary)


def _validate_fresh_sampling_claim(
    cell: dict,
    pair: object,
    *,
    benchmark: str,
    target: str,
) -> None:
    """Require exact agreement between a fresh cell and raw sampling evidence."""

    field = _SAMPLING_SENSITIVITY_FIELD
    expected = _recomputed_sampling_sensitivity(pair, benchmark=benchmark)
    if expected is None:
        if field in cell:
            raise _fail(
                f"scoreboard cell {benchmark!r}/{target!r} carries sampling "
                "sensitivity without complete raw evidence"
            )
        return
    if field not in cell or _canonical_bytes(cell[field]) != _canonical_bytes(expected):
        raise _fail(
            f"scoreboard cell {benchmark!r}/{target!r} sampling sensitivity "
            "disagrees with raw pair evidence"
        )


def pair_result_binding(
    pair: object,
    *,
    declared_binary: bool | None = None,
) -> dict:
    """Return a detached, history-safe canonical binding for one PairResult.

    ``declared_binary`` lets replay use the declaration already retained by an
    authoritative index binding instead of consulting a later adapter registry.
    Fresh index construction leaves it unset and uses the current adapter.
    """
    try:
        return _snapshot(
            _pair_result_binding(pair, declared_binary=declared_binary)
        )
    except RecursionError as error:
        raise _fail("pair result exceeds the supported nesting depth") from error


def _pair_bindings(
    scoreboard: dict,
    matrix: set[tuple[str, str]],
    pairs: Sequence[object],
    *,
    fingerprint: str,
    expected_source_identity: dict,
    stored: bool,
    sampling_expectations: Mapping[tuple[str, str], dict[str, object]],
) -> list[dict[str, Any]]:
    if not pairs:
        raise _fail("complete PairResult evidence is required for every scoreboard cell")
    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if stored:
            if not isinstance(pair, dict) or set(pair) != _PAIR_FIELDS:
                raise _fail("pair binding fields do not match index schema 1")
            binding = _snapshot(pair)
        else:
            raw_benchmark = (
                pair.benchmark
                if hasattr(pair, "benchmark")
                else pair.get("benchmark") if isinstance(pair, Mapping) else None
            )
            raw_target = (
                pair.target
                if hasattr(pair, "target")
                else pair.get("target") if isinstance(pair, Mapping) else None
            )
            binding = _pair_result_binding(
                pair,
                expected_sampling=sampling_expectations.get(
                    (raw_benchmark, raw_target)
                ),
            )
        benchmark = binding.get("benchmark")
        target = binding.get("target")
        pair_fingerprint = binding.get("run_fingerprint")
        pair_source_identity = binding.get("source_identity")
        if not isinstance(benchmark, str) or not benchmark:
            raise _fail("pair benchmark must be a non-empty string")
        if not isinstance(target, str) or not target:
            raise _fail("pair target must be a non-empty string")
        key = (benchmark, target)
        if key in seen:
            raise _fail(f"duplicate pair binding for {benchmark!r}/{target!r}")
        if key not in matrix:
            raise _fail(f"pair {benchmark!r}/{target!r} is outside the scoreboard matrix")
        seen.add(key)
        if pair_fingerprint != fingerprint:
            raise _fail(f"pair {benchmark!r}/{target!r} does not carry the run fingerprint")
        if pair_source_identity != expected_source_identity:
            raise _fail(
                f"pair {benchmark!r}/{target!r} does not carry the clean run source identity"
            )
        cell = scoreboard["cells"][benchmark][target]
        if stored:
            if _SAMPLING_SENSITIVITY_FIELD in cell:
                raise _fail(
                    f"stored scoreboard cell {benchmark!r}/{target!r} must not carry "
                    "sampling sensitivity in index schema 1"
                )
        else:
            _validate_fresh_sampling_claim(
                cell,
                pair,
                benchmark=benchmark,
                target=target,
            )
        summary = _cell_summary(binding, field=f"pair binding {benchmark!r}/{target!r}")
        expected_summary = _cell_summary(
            cell,
            field=f"scoreboard cell {benchmark!r}/{target!r}",
        )
        if _canonical_bytes(summary) != _canonical_bytes(expected_summary):
            raise _fail(f"pair {benchmark!r}/{target!r} summary disagrees with scoreboard cell")
        _validate_digest(binding.get("pair_sha256"), field="pair_sha256")
        bindings.append(
            {
                "benchmark": benchmark,
                "target": target,
                "run_fingerprint": pair_fingerprint,
                "source_identity": _snapshot(pair_source_identity),
                **summary,
                "pair_sha256": binding["pair_sha256"],
            }
        )
    if seen != matrix:
        raise _fail("pair bindings do not match the complete scoreboard cell matrix")
    return sorted(bindings, key=lambda item: (item["benchmark"], item["target"]))


def _validate_scoreboard(
    scoreboard: object,
    run: dict,
    pairs: Sequence[object],
    *,
    stored_pairs: bool,
) -> tuple[dict, list[dict[str, Any]]]:
    if not isinstance(scoreboard, dict):
        raise _fail("scoreboard must be an object")
    board = _snapshot(scoreboard)
    if type(board.get("schema_version")) is not int or board["schema_version"] != 1:
        raise _fail("scoreboard schema_version must be 1")
    if board.get("run_id") != run["run_id"]:
        raise _fail("scoreboard run_id does not match run metadata")
    if board.get("fingerprint") != run["fingerprint"]:
        raise _fail("scoreboard fingerprint does not match run metadata")
    config = run["config"]
    suite = config.get("suite")
    if not isinstance(suite, str) or not suite:
        raise _fail("run config suite must be a non-empty string")
    if board.get("suite") != suite:
        raise _fail("scoreboard suite does not match run config")
    if board.get("config") != config:
        raise _fail("scoreboard config does not match persisted run metadata")
    if board.get("environment") != run["environment"]:
        raise _fail("scoreboard environment does not match persisted run metadata")
    adapters = run["identity"].get("adapters")
    if not isinstance(adapters, list) or any(
        not isinstance(adapter, dict)
        or not isinstance(adapter.get("name"), str)
        or not adapter["name"]
        for adapter in adapters
    ):
        raise _fail("run identity adapters must disclose ordered benchmark names")
    adapter_names = [adapter["name"] for adapter in adapters]
    if len(adapter_names) != len(set(adapter_names)):
        raise _fail("run identity adapter names must be unique")
    protocol_marker = config.get("sampling_protocol")
    if protocol_marker is not None and protocol_marker != SAMPLING_PROTOCOL_ID:
        raise _fail("run config carries an unknown sampling protocol marker")
    raw_targets = config.get("targets")
    target_sampling_policy_used = isinstance(raw_targets, list) and any(
        isinstance(target, dict)
        and (
            target.get("temperature") is not None
            or target.get("sampling_mode") == "recommended"
        )
        for target in raw_targets
    )
    sampling_protocol_used = (
        protocol_marker == SAMPLING_PROTOCOL_ID
        or target_sampling_policy_used
        or _scoreboard_has_sampling_claim(board)
        or (not stored_pairs and any(_pair_has_sampling_evidence(pair) for pair in pairs))
    )
    if sampling_protocol_used:
        _require_sampling_evaluation_resource(adapters)
    configured_targets = config.get("targets")
    if not isinstance(configured_targets, list) or any(
        not isinstance(target, dict) for target in configured_targets
    ):
        raise _fail("run config targets must disclose ordered target identities")
    target_labels: list[str] = []
    configured_by_label: dict[str, dict] = {}
    for target in configured_targets:
        label = target.get("name") or target.get("model")
        if not isinstance(label, str) or not label:
            raise _fail("each run config target requires a non-empty label or model")
        target_labels.append(label)
        configured_by_label[label] = target
    if len(target_labels) != len(set(target_labels)):
        raise _fail("run config target labels must be unique")
    if board.get("benchmarks") != adapter_names:
        raise _fail("scoreboard benchmarks do not match run identity adapters")
    if board.get("targets") != target_labels:
        raise _fail("scoreboard targets do not match run config target labels")
    matrix = _scoreboard_matrix(board)
    from kairyu.bench.adapters import all_adapters
    from kairyu.bench.adapters.base import GenerativeAdapter

    registered = all_adapters()
    grouped_chat_attempts = (
        config.get("attempts", 1)
        if protocol_marker == SAMPLING_PROTOCOL_ID
        else 1
    )
    sampling_expectations = {
        (benchmark, target): {
            # ``attempts`` was added after index schema 1. Missing legacy
            # records are the original one-attempt protocol.
            "attempts": (
                grouped_chat_attempts
                if isinstance(registered.get(benchmark), GenerativeAdapter)
                else config.get("attempts", 1)
            ),
            "seed": configured_by_label[target].get("seed"),
            "mode": configured_by_label[target].get("sampling_mode", "adapter"),
            "temperature": configured_by_label[target].get("temperature"),
            "top_p": configured_by_label[target].get("top_p"),
            "required": isinstance(registered.get(benchmark), GenerativeAdapter),
        }
        for benchmark, target in matrix
    }
    from kairyu.bench.types import sampling_configuration_error

    for expectation in sampling_expectations.values():
        error = sampling_configuration_error(expectation)
        if error is not None:
            raise _fail(error)
    policies = cross_run_policies(adapters, run["environment"])
    identities_by_name = {identity["name"]: identity for identity in adapters}
    for benchmark in adapter_names:
        identity = identities_by_name[benchmark]
        has_scored_cell = any(
            board["cells"][benchmark][target]["status"] in {"completed", "partial"}
            for target in target_labels
        )
        if (
            benchmark not in _NON_DATASET_ADAPTERS
            and has_scored_cell
            and not _dataset_identity_is_content_bound(identity, adapter=benchmark)
        ):
            raise _fail(
                f"benchmark {benchmark!r} completed or partial evidence requires "
                "a content-bound dataset manifest and asset identity list"
            )
        expected_policy, expected_reason = policies[benchmark]
        for target in target_labels:
            summary = _cell_summary(
                board["cells"][benchmark][target],
                field=f"scoreboard cell {benchmark!r}/{target!r}",
            )
            if (
                summary["cross_run_policy"] != expected_policy
                or summary["cross_run_reason"] != expected_reason
            ):
                raise _fail(
                    f"scoreboard cell {benchmark!r}/{target!r} cross-run policy "
                    "does not match its runtime provenance"
                )
    bindings = _pair_bindings(
        board,
        matrix,
        pairs,
        fingerprint=run["fingerprint"],
        expected_source_identity=source_identity(run["environment"]),
        stored=stored_pairs,
        sampling_expectations=sampling_expectations,
    )
    if not stored_pairs:
        # Index schema 1 has no canonical pair-binding field for this derived
        # claim.  It was proven against the raw PairResult above; omit it from
        # the durable board so later index validation can never trust a detached
        # or attacker-rehashed pass@k value.
        for benchmark, target in matrix:
            board["cells"][benchmark][target].pop(_SAMPLING_SENSITIVITY_FIELD, None)
    return board, bindings


def _record_digest(entry: dict) -> str:
    payload = {key: value for key, value in entry.items() if key != "record_sha256"}
    return _sha256(payload)


def build_entry(
    scoreboard: object,
    run_metadata: object,
    *,
    pairs: Sequence[object] = (),
    previous_record_sha256: str | None,
) -> dict:
    run = validate_run_metadata(run_metadata, require_clean_source=True)
    board, bindings = _validate_scoreboard(scoreboard, run, pairs, stored_pairs=False)
    commit = _validate_commit(run["environment"]["git_commit"])
    entry = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "git_commit": commit,
        "fingerprint": run["fingerprint"],
        "run_id": run["run_id"],
        "suite": board["suite"],
        "run": run,
        "scoreboard": board,
        "pairs": bindings,
        "previous_record_sha256": previous_record_sha256,
    }
    if previous_record_sha256 is not None:
        _validate_digest(previous_record_sha256, field="previous_record_sha256")
    entry["record_sha256"] = _record_digest(entry)
    return entry


def _validate_entry(entry: object, *, previous: str | None) -> dict:
    if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
        raise _fail("record fields do not match index schema 1")
    if (
        type(entry.get("index_schema_version")) is not int
        or entry["index_schema_version"] != INDEX_SCHEMA_VERSION
    ):
        raise _fail("unknown index schema_version")
    recorded_previous = entry.get("previous_record_sha256")
    if recorded_previous != previous:
        raise _fail("record SHA-256 chain is broken")
    if recorded_previous is not None:
        _validate_digest(recorded_previous, field="previous_record_sha256")
    record_sha = _validate_digest(entry.get("record_sha256"), field="record_sha256")
    if _record_digest(entry) != record_sha:
        raise _fail("record_sha256 does not match the canonical record")
    run = validate_run_metadata(entry.get("run"), require_clean_source=True)
    stored_pairs = entry.get("pairs")
    if not isinstance(stored_pairs, list):
        raise _fail("pairs must be a list")
    board, canonical_pairs = _validate_scoreboard(
        entry.get("scoreboard"), run, stored_pairs, stored_pairs=True
    )
    commit = _validate_commit(run["environment"]["git_commit"])
    expected = {
        "git_commit": commit,
        "fingerprint": run["fingerprint"],
        "run_id": run["run_id"],
        "suite": board["suite"],
    }
    if any(entry.get(field) != value for field, value in expected.items()):
        raise _fail("record key fields disagree with its run and scoreboard snapshots")
    if stored_pairs != canonical_pairs:
        raise _fail("pair bindings are not in canonical matrix order")
    return entry


class _DuplicateObjectKey(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateObjectKey(key)
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def indexed_pair_binding(
    entry: object,
    *,
    run_id: str,
    benchmark: str,
    target: str,
) -> dict:
    """Revalidate an index row and return its unique requested pair binding."""
    selected_run_id = _validate_run_id(run_id)
    selected_benchmark = _nonempty_name(benchmark, field="benchmark")
    selected_target = _nonempty_name(target, field="target")

    try:
        candidate = _snapshot(entry)
        previous = (
            candidate.get("previous_record_sha256")
            if isinstance(candidate, dict)
            else None
        )
        validated = _validate_entry(candidate, previous=previous)
    except RecursionError as error:
        raise _fail("indexed pair entry exceeds the supported nesting depth") from error
    if validated["run_id"] != selected_run_id:
        raise _fail(
            f"store run id {selected_run_id!r} does not match indexed run "
            f"{validated['run_id']!r}"
        )

    matches = [
        binding
        for binding in validated["pairs"]
        if binding["benchmark"] == selected_benchmark
        and binding["target"] == selected_target
    ]
    if len(matches) != 1:
        raise _fail(
            f"indexed run {selected_run_id!r} has no unique pair binding for "
            f"{selected_benchmark!r}/{selected_target!r}"
        )
    return _snapshot(matches[0])


def validate_pair_binding(
    raw_pair: bytes,
    binding: object,
    *,
    benchmark: str,
    target: str,
):
    """Strictly parse one PairResult and match its complete canonical binding."""
    from kairyu.bench.types import PairResult

    selected_benchmark = _nonempty_name(benchmark, field="benchmark")
    selected_target = _nonempty_name(target, field="target")
    if not isinstance(binding, dict) or set(binding) != _PAIR_FIELDS:
        raise _fail("pair binding fields do not match index schema 1")
    try:
        expected = _snapshot(binding)
    except RecursionError as error:
        raise _fail("pair binding exceeds the supported nesting depth") from error
    if (
        expected.get("benchmark") != selected_benchmark
        or expected.get("target") != selected_target
    ):
        raise _fail("selected pair identity does not match its indexed binding")
    if not isinstance(raw_pair, bytes):
        raise _fail("persisted pair result must be supplied as bytes")

    try:
        parsed = json.loads(
            raw_pair.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _fail(
            f"persisted pair {selected_benchmark!r}/{selected_target!r} is not strict JSON"
        ) from error
    except RecursionError as error:
        raise _fail(
            f"persisted pair {selected_benchmark!r}/{selected_target!r} "
            "exceeds the supported nesting depth"
        ) from error

    try:
        actual = pair_result_binding(
            parsed,
            declared_binary=expected.get("confidence_interval") is not None,
        )
        if _canonical_bytes(actual) != _canonical_bytes(expected):
            raise _fail(
                f"persisted pair {selected_benchmark!r}/{selected_target!r} "
                "does not match its complete indexed binding"
            )
        # Revalidate and detach the returned model from the decoded mutable tree.
        return PairResult.model_validate(parsed)
    except RecursionError as error:
        raise _fail(
            f"persisted pair {selected_benchmark!r}/{selected_target!r} "
            "exceeds the supported nesting depth"
        ) from error


def validate_indexed_pair(
    entry: object,
    raw_pair: bytes,
    *,
    run_id: str,
    benchmark: str,
    target: str,
):
    """Strictly bind one persisted ``PairResult`` to a validated index row.

    ``entry`` is expected to originate from :func:`load_scoreboard_index`, but
    is revalidated here so callers cannot accidentally bypass the record,
    scoreboard-matrix, fingerprint, source-identity, or binding-uniqueness
    checks.  The complete canonical pair binding is compared, not only its
    aggregate score; this makes ``pair_sha256`` authoritative over item-level
    evidence and methodology as well.
    """
    binding = indexed_pair_binding(
        entry,
        run_id=run_id,
        benchmark=benchmark,
        target=target,
    )
    return validate_pair_binding(
        raw_pair,
        binding,
        benchmark=benchmark,
        target=target,
    )


def parse_index(raw: bytes) -> tuple[dict, ...]:
    """Parse one immutable byte snapshot of the canonical JSONL index."""
    if not raw:
        return ()
    if not raw.endswith(b"\n"):
        raise _fail("index has a torn final record")
    if b"\r" in raw:
        raise _fail("index must use canonical LF framing")
    entries: list[dict] = []
    keys: set[tuple[str, str]] = set()
    run_ids: set[str] = set()
    previous: str | None = None
    for line_number, line in enumerate(raw[:-1].split(b"\n"), start=1):
        if not line:
            raise _fail(f"blank record at line {line_number}")
        try:
            text = line.decode("utf-8")
            parsed = json.loads(
                text,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_constant,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            RecursionError,
        ) as error:
            raise _fail(f"line {line_number} is not strict JSON") from error
        try:
            entry = _validate_entry(parsed, previous=previous)
            if _canonical_bytes(entry) != line:
                raise _fail(f"line {line_number} is not canonical JSON")
        except RecursionError as error:
            raise _fail(
                f"line {line_number} exceeds the supported nesting depth"
            ) from error
        key = (entry["git_commit"], entry["fingerprint"])
        if key in keys:
            raise _fail(f"duplicate scoreboard key at line {line_number}")
        if entry["run_id"] in run_ids:
            raise _fail(f"duplicate run_id at line {line_number}")
        keys.add(key)
        run_ids.add(entry["run_id"])
        entries.append(entry)
        previous = entry["record_sha256"]
    return tuple(entries)


@contextmanager
def _opened_results_root(results_dir: Path, *, create: bool) -> Iterator[int]:
    """Pin the results directory inode before resolving any child artifact."""
    if create:
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise _fail("results root cannot be created") from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(results_dir, flags)
    except OSError as error:
        raise _fail("history refuses a symlinked or unsafe results root") from error
    try:
        opened = os.fstat(descriptor)
        current = results_dir.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or current.st_dev != opened.st_dev
            or current.st_ino != opened.st_ino
        ):
            raise _fail("results root changed while opening")
        yield descriptor
        try:
            current = results_dir.stat(follow_symlinks=False)
        except OSError as error:
            raise _fail("results root changed during history operation") from error
        if current.st_dev != opened.st_dev or current.st_ino != opened.st_ino:
            raise _fail("results root changed during history operation")
    finally:
        os.close(descriptor)


def _same_dir_entry(root_fd: int, name: str, opened: os.stat_result) -> None:
    try:
        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError as error:
        raise _fail(f"history artifact {name} changed while opening") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
    ):
        raise _fail(f"history refuses unsafe artifact {name}")


def _entry_exists(root_fd: int, name: str, *, label: str) -> bool:
    try:
        entry = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError as error:
        if error.errno == errno.ENOENT:
            return False
        raise _fail(f"history {label} cannot be inspected safely") from error
    if not stat.S_ISREG(entry.st_mode):
        raise _fail(f"history refuses an unsafe {label} artifact")
    return True


@contextmanager
def _root_lock(root_fd: int, *, exclusive: bool, create: bool = True) -> Iterator[None]:
    flags = (os.O_RDWR if exclusive else os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(LOCK_NAME, flags, 0o600, dir_fd=root_fd)
    except OSError as error:
        raise _fail("history lock cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        _same_dir_entry(root_fd, LOCK_NAME, opened)
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        _same_dir_entry(root_fd, LOCK_NAME, opened)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_regular_file(root_fd: int) -> bytes:
    # O_NONBLOCK prevents an attacker-controlled FIFO/device at the tracked
    # index path from hanging a read-only compare before fstat can reject it.
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(INDEX_NAME, flags, dir_fd=root_fd)
    except OSError as error:
        if error.errno == errno.ENOENT:
            return b""
        if error.errno in {errno.ELOOP, errno.ENXIO}:
            raise _fail(f"history refuses unsafe artifact {INDEX_NAME}") from error
        raise _fail(f"history index {INDEX_NAME} cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise _fail(f"history refuses unsafe artifact {INDEX_NAME}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_replace_index(root_fd: int, payload: bytes) -> None:
    """Publish one durable index snapshot using only the pinned root dirfd."""
    temporary = f".{INDEX_NAME}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=root_fd)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing scoreboard history")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            INDEX_NAME,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        os.fsync(root_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=root_fd)
        except FileNotFoundError:
            pass


def load_scoreboard_index(results_dir: str | Path) -> tuple[dict, ...]:
    root = Path(results_dir)
    if root.is_symlink():
        raise _fail("history refuses a symlinked results root")
    if not root.exists() and not root.is_symlink():
        return ()
    if not root.is_dir():
        raise _fail("results root is not a directory")
    with _opened_results_root(root, create=False) as root_fd:
        lock_exists = _entry_exists(root_fd, LOCK_NAME, label="lock")
        if lock_exists:
            with _root_lock(root_fd, exclusive=False, create=False):
                return parse_index(_read_regular_file(root_fd))
        # Atomic replacement makes either the old or new index a complete
        # immutable snapshot.  With no writer lock, read it directly so a fresh
        # compare-runs remains genuinely read-only.
        return parse_index(_read_regular_file(root_fd))


def _stable_payload(entry: dict) -> dict:
    return {
        key: value
        for key, value in entry.items()
        if key not in {"previous_record_sha256", "record_sha256"}
    }


def append_scoreboard_index(
    results_dir: str | Path,
    scoreboard: object,
    run_metadata: object,
    *,
    pairs: Sequence[object] = (),
) -> Path:
    root = Path(results_dir)
    index_path = root / INDEX_NAME
    with _opened_results_root(root, create=True) as root_fd:
        with _root_lock(root_fd, exclusive=True):
            raw = _read_regular_file(root_fd)
            entries = parse_index(raw)
            previous = entries[-1]["record_sha256"] if entries else None
            candidate = build_entry(
                scoreboard,
                run_metadata,
                pairs=pairs,
                previous_record_sha256=previous,
            )
            key = (candidate["git_commit"], candidate["fingerprint"])
            for existing in entries:
                if (existing["git_commit"], existing["fingerprint"]) == key:
                    if _canonical_bytes(_stable_payload(existing)) == _canonical_bytes(
                        _stable_payload(candidate)
                    ):
                        return index_path
                    raise _fail("scoreboard key already identifies different evidence")
                if existing["run_id"] == candidate["run_id"]:
                    raise _fail("run_id already identifies a different scoreboard key")
            updated = raw + _canonical_bytes(candidate) + b"\n"
            _atomic_replace_index(root_fd, updated)
            return index_path


def ensure_scoreboard_key_available(
    results_dir: str | Path,
    run_metadata: object,
    *,
    rerun: bool = False,
) -> None:
    run = validate_run_metadata(run_metadata, require_clean_source=True)
    key = scoreboard_key(run)
    for entry in load_scoreboard_index(results_dir):
        existing_key = (entry["git_commit"], entry["fingerprint"])
        if existing_key == key:
            if entry["run_id"] == run["run_id"] and not rerun:
                return
            raise _fail("scoreboard key is already indexed")
        if entry["run_id"] == run["run_id"]:
            raise _fail("run_id is already indexed under another scoreboard key")


def find_scoreboard_entry(
    results_dir: str | Path,
    run_id: str,
) -> dict | None:
    selected = _validate_run_id(run_id)
    matches = [entry for entry in load_scoreboard_index(results_dir) if entry["run_id"] == selected]
    if len(matches) > 1:  # load already rejects this; retain a defensive boundary.
        raise _fail(f"run_id {selected!r} is ambiguous")
    return matches[0] if matches else None


__all__ = [
    "INDEX_NAME",
    "INDEX_SCHEMA_VERSION",
    "LOCK_NAME",
    "append_scoreboard_index",
    "build_entry",
    "cross_run_policies",
    "ensure_scoreboard_key_available",
    "find_scoreboard_entry",
    "indexed_pair_binding",
    "load_scoreboard_index",
    "pair_result_binding",
    "parse_index",
    "scoreboard_key",
    "resumable_environment_identity_equal",
    "stable_environment_identity",
    "validate_indexed_pair",
    "validate_pair_binding",
    "validate_run_metadata",
]
