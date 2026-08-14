"""Fail-closed quantization x task-accuracy sweep artifacts.

One indexed ``quantization`` suite run contains seven served targets.  This
module transposes the benchmark-major scoreboard into a scheme-major table and
uses the public configuration A/B gate as the only statistical/evidence core
for every candidate-versus-dense comparison.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from evals import aggregate as aggregate_protocol
from evals import config_ab as config_ab_protocol
from evals import config_compare as config_statistics
from evals import history as scoreboard_history
from evals import store as store_protocol
from evals import types as types_protocol
from evals.adapters import QUANTIZATION_ROW_ORDER
from evals.config_ab import build_config_comparison
from evals.history import validate_run_metadata
from evals.types import (
    PairResult,
    QuantizationProfile,
    ServedConfigIdentity,
)
from kairyu.engine.core import kv_cache_dtype as kv_dtype_protocol
from kairyu.engine.core import quant_config as quant_config_protocol

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

# (public key, report label, checkpoint quant method, compute dtype, KV dtype)
_REQUIRED_PROFILES: tuple[tuple[str, str, str, str, str], ...] = (
    ("bf16", "BF16", "none", "bfloat16", "bfloat16"),
    ("fp8", "FP8", "fp8", "bfloat16", "bfloat16"),
    ("int8", "INT8", "int8", "bfloat16", "bfloat16"),
    ("awq", "AWQ", "awq", "bfloat16", "bfloat16"),
    ("gptq", "GPTQ", "gptq", "bfloat16", "bfloat16"),
    ("nvfp4", "NVFP4", "nvfp4", "bfloat16", "bfloat16"),
    ("fp8-kv", "BF16 + FP8 KV", "none", "bfloat16", "fp8_e4m3"),
)


class QuantizationSweepError(ValueError):
    """The indexed evidence cannot support a complete quantization sweep."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as error:
        raise QuantizationSweepError("sweep evidence is not canonical JSON") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _protocol_identity() -> dict:
    modules = (
        types_protocol,
        aggregate_protocol,
        scoreboard_history,
        store_protocol,
        config_statistics,
        config_ab_protocol,
        quant_config_protocol,
        kv_dtype_protocol,
    )
    resources: list[dict[str, str]] = []
    own_path = Path(__file__)
    try:
        resources.append(
            {
                "resource": "evals/quant_sweep.py",
                "sha256": hashlib.sha256(own_path.read_bytes()).hexdigest(),
            }
        )
        for module in modules:
            module_path = Path(module.__file__)
            resources.append(
                {
                    "resource": module.__name__.replace(".", "/") + ".py",
                    "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                }
            )
    except OSError as error:
        raise QuantizationSweepError(
            "cannot content-bind the quantization sweep protocol"
        ) from error
    return {
        "name": "kairyu_quantization_task_accuracy_v1",
        "resources": resources,
        "sha256": _sha256(resources),
    }


def _mapping(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise QuantizationSweepError(f"{field} must be an object")
    return value


def _name_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise QuantizationSweepError(
            f"{field} must be a unique list of non-empty strings"
        )
    return list(value)


def _safe_report_label(value: str, field: str) -> str:
    if "`" in value or any(
        unicodedata.category(character).startswith("C") for character in value
    ):
        raise QuantizationSweepError(
            f"{field} must not contain control characters or backticks"
        )
    return value


def _profile_dict(spec: tuple[str, str, str, str, str]) -> dict[str, str]:
    return {
        "weight_method": spec[2],
        "compute_dtype": spec[3],
        "kv_cache_dtype": spec[4],
    }


def required_quantization_profiles() -> tuple[dict[str, object], ...]:
    """Return a detached copy of the versioned seven-arm coverage contract."""

    return tuple(
        {
            "key": spec[0],
            "display_name": spec[1],
            "profile": _profile_dict(spec),
        }
        for spec in _REQUIRED_PROFILES
    )


def _profile_tuple(profile: QuantizationProfile) -> tuple[str, str, str]:
    return profile.weight_method, profile.compute_dtype, profile.kv_cache_dtype


def _snapshot(entry: object) -> dict:
    payload = _mapping(entry, "scoreboard index entry")
    run = validate_run_metadata(payload.get("run"), require_clean_source=True)
    board = _mapping(payload.get("scoreboard"), "scoreboard")
    record_sha256 = payload.get("record_sha256")
    if not isinstance(record_sha256, str) or _DIGEST_RE.fullmatch(record_sha256) is None:
        raise QuantizationSweepError("index entry has no valid record SHA-256")
    if payload.get("run_id") != run["run_id"] or board.get("run_id") != run["run_id"]:
        raise QuantizationSweepError("run id disagrees across indexed sweep evidence")
    _safe_report_label(run["run_id"], "run id")
    if (
        payload.get("fingerprint") != run["fingerprint"]
        or board.get("fingerprint") != run["fingerprint"]
    ):
        raise QuantizationSweepError(
            "run fingerprint disagrees across indexed sweep evidence"
        )
    if board.get("config") != run["config"] or board.get("environment") != run["environment"]:
        raise QuantizationSweepError("scoreboard disagrees with indexed run metadata")
    if (
        payload.get("suite") != "quantization"
        or board.get("suite") != "quantization"
        or run["config"].get("suite") != "quantization"
    ):
        raise QuantizationSweepError("sweep requires suite 'quantization'")

    tasks = _name_list(board.get("benchmarks"), "scoreboard tasks")
    if tasks != list(QUANTIZATION_ROW_ORDER):
        raise QuantizationSweepError(
            "quantization sweep requires the complete ordered task set "
            f"{list(QUANTIZATION_ROW_ORDER)!r}"
        )
    adapters = run["identity"].get("adapters")
    if not isinstance(adapters, list) or [
        adapter.get("name") if isinstance(adapter, dict) else None
        for adapter in adapters
    ] != tasks:
        raise QuantizationSweepError("adapter methodology does not match sweep tasks")

    targets = _name_list(board.get("targets"), "scoreboard targets")
    raw_targets = run["identity"]["config"].get("targets")
    if not isinstance(raw_targets, list) or any(
        not isinstance(target, dict) for target in raw_targets
    ):
        raise QuantizationSweepError("quantization target identities are malformed")
    targets_by_label: dict[str, dict] = {}
    target_by_profile: dict[tuple[str, str, str], str] = {}
    served_digests: set[str] = set()
    for raw_target in raw_targets:
        label = raw_target.get("name") or raw_target.get("model")
        if not isinstance(label, str) or not label or label in targets_by_label:
            raise QuantizationSweepError(
                "quantization target labels must be non-empty and unique"
            )
        _safe_report_label(label, f"target label {label!r}")
        raw_profile = raw_target.get("quantization")
        try:
            profile = QuantizationProfile.model_validate(raw_profile)
        except ValueError as error:
            raise QuantizationSweepError(
                f"target {label!r} has no valid quantization profile"
            ) from error
        if profile.model_dump(mode="json") != raw_profile:
            raise QuantizationSweepError(
                f"target {label!r} quantization profile is not canonical"
            )
        identity = raw_target.get("served_config")
        try:
            served_config = ServedConfigIdentity.model_validate(identity)
        except ValueError as error:
            raise QuantizationSweepError(
                f"target {label!r} requires a valid served_config identity"
            ) from error
        if served_config.model_dump(mode="json") != identity:
            raise QuantizationSweepError(
                f"target {label!r} served_config identity is not canonical"
            )
        if served_config.sha256 in served_digests:
            raise QuantizationSweepError(
                "every quantization arm requires a distinct served_config digest"
            )
        served_digests.add(served_config.sha256)
        profile_identity = _profile_tuple(profile)
        if profile_identity in target_by_profile:
            raise QuantizationSweepError(
                f"duplicate quantization profile on targets "
                f"{target_by_profile[profile_identity]!r} and {label!r}"
            )
        target_by_profile[profile_identity] = label
        targets_by_label[label] = deepcopy(raw_target)

    if targets != list(targets_by_label):
        raise QuantizationSweepError(
            "scoreboard target layout disagrees with quantization config"
        )
    expected = {
        (spec[2], spec[3], spec[4])
        for spec in _REQUIRED_PROFILES
    }
    actual = set(target_by_profile)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise QuantizationSweepError(
            f"quantization profile coverage is incomplete "
            f"(missing={missing!r}, extra={extra!r})"
        )
    return {
        "entry": payload,
        "run": run,
        "board": board,
        "record_sha256": record_sha256,
        "tasks": tasks,
        "targets": targets,
        "targets_by_label": targets_by_label,
        "target_by_profile": target_by_profile,
    }


def _validate_pair_matrix(
    pairs: Mapping[str, Mapping[str, PairResult]],
    *,
    targets: list[str],
    tasks: list[str],
) -> dict[str, dict[str, PairResult]]:
    if not isinstance(pairs, Mapping) or set(pairs) != set(targets):
        raise QuantizationSweepError(
            "raw pair matrix must exactly cover every quantization target"
        )
    validated: dict[str, dict[str, PairResult]] = {}
    for target in targets:
        cells = pairs[target]
        if not isinstance(cells, Mapping) or set(cells) != set(tasks):
            raise QuantizationSweepError(
                f"raw pair matrix for target {target!r} must exactly cover every task"
            )
        if any(not isinstance(cells[task], PairResult) for task in tasks):
            raise QuantizationSweepError(
                f"raw pair matrix for target {target!r} contains a non-PairResult cell"
            )
        validated[target] = {task: cells[task] for task in tasks}
    return validated


def _validate_policy(tolerances: Mapping[str, float], tasks: list[str]) -> dict[str, float]:
    if not isinstance(tolerances, Mapping) or set(tolerances) != set(tasks):
        missing = sorted(set(tasks) - set(tolerances)) if isinstance(tolerances, Mapping) else tasks
        extra = sorted(set(tolerances) - set(tasks)) if isinstance(tolerances, Mapping) else []
        raise QuantizationSweepError(
            "task tolerances must exactly cover the sweep "
            f"(missing={missing!r}, extra={extra!r})"
        )
    # The configuration A/B builder performs strict finite/range validation.
    return {task: tolerances[task] for task in tasks}


def _arm_identity(
    *,
    spec: tuple[str, str, str, str, str],
    target: str,
    target_config: dict,
    role: str,
) -> dict:
    return {
        "key": spec[0],
        "display_name": spec[1],
        "role": role,
        "profile": _profile_dict(spec),
        "target": target,
        "target_config": deepcopy(target_config),
        "target_config_sha256": _sha256(target_config),
        "served_config": deepcopy(target_config["served_config"]),
    }


def build_quantization_sweep(
    entry: dict,
    *,
    pairs: Mapping[str, Mapping[str, PairResult]],
    tolerances: Mapping[str, float],
) -> dict:
    """Build one complete seven-arm, four-task non-inferiority artifact.

    ``entry`` must come from :meth:`ResultStore.load_scoreboard_index` and each
    cell from :meth:`ResultStore.load_indexed_pair`.  The same task margin is
    applied to every candidate; scores are never averaged across tasks or arms.
    """

    snapshot = _snapshot(entry)
    pair_matrix = _validate_pair_matrix(
        pairs,
        targets=snapshot["targets"],
        tasks=snapshot["tasks"],
    )
    policy = _validate_policy(tolerances, snapshot["tasks"])
    target_for = snapshot["target_by_profile"]
    baseline_spec = _REQUIRED_PROFILES[0]
    baseline_target = target_for[(baseline_spec[2], baseline_spec[3], baseline_spec[4])]

    comparisons: list[dict] = []
    for spec in _REQUIRED_PROFILES[1:]:
        candidate_target = target_for[(spec[2], spec[3], spec[4])]
        comparison = build_config_comparison(
            snapshot["entry"],
            snapshot["entry"],
            baseline_pairs=pair_matrix[baseline_target],
            candidate_pairs=pair_matrix[candidate_target],
            tolerances=policy,
            baseline_target=baseline_target,
            candidate_target=candidate_target,
        )
        if comparison.get("suite") != "quantization":
            raise QuantizationSweepError("config comparison returned the wrong suite")
        rows = comparison.get("benchmarks")
        if not isinstance(rows, list) or [row.get("benchmark") for row in rows] != snapshot[
            "tasks"
        ]:
            raise QuantizationSweepError(
                "config comparison returned an incomplete task layout"
            )
        if comparison["baseline"]["pair_target"] != baseline_target:
            raise QuantizationSweepError("config comparison changed the baseline target")
        if comparison["candidate"]["pair_target"] != candidate_target:
            raise QuantizationSweepError("config comparison changed the candidate target")
        comparisons.append(comparison)

    first = comparisons[0]
    baseline = _arm_identity(
        spec=baseline_spec,
        target=baseline_target,
        target_config=snapshot["targets_by_label"][baseline_target],
        role="reference",
    )
    baseline["tasks"] = {
        row["benchmark"]: {
            "display_name": row["display_name"],
            "score": row["baseline_score"],
            "n_items": row["n_pairs"],
            "item_ids_sha256": row["item_ids_sha256"],
            "pair_sha256": row["evidence"]["baseline_pair_sha256"],
        }
        for row in first["benchmarks"]
    }
    baseline["overall_verdict"] = "reference"

    arms: list[dict] = [baseline]
    for spec, comparison in zip(_REQUIRED_PROFILES[1:], comparisons, strict=True):
        if comparison["methodology_fingerprint"] != first["methodology_fingerprint"]:
            raise QuantizationSweepError(
                "candidate comparisons disagree on methodology fingerprint"
            )
        if comparison["comparison_protocol"] != first["comparison_protocol"]:
            raise QuantizationSweepError(
                "candidate comparisons disagree on comparison protocol"
            )
        if comparison["comparison_runtime"] != first["comparison_runtime"]:
            raise QuantizationSweepError(
                "candidate comparisons disagree on comparator runtime"
            )
        candidate_target = comparison["candidate"]["pair_target"]
        arm = _arm_identity(
            spec=spec,
            target=candidate_target,
            target_config=snapshot["targets_by_label"][candidate_target],
            role="candidate",
        )
        tasks: dict[str, dict] = {}
        for row in comparison["benchmarks"]:
            baseline_task = baseline["tasks"][row["benchmark"]]
            if (
                row["baseline_score"] != baseline_task["score"]
                or row["n_pairs"] != baseline_task["n_items"]
                or row["item_ids_sha256"] != baseline_task["item_ids_sha256"]
                or row["evidence"]["baseline_pair_sha256"]
                != baseline_task["pair_sha256"]
            ):
                raise QuantizationSweepError(
                    "candidate comparisons do not share exact baseline evidence"
                )
            tasks[row["benchmark"]] = {
                "display_name": row["display_name"],
                "score": row["candidate_score"],
                "baseline_score": row["baseline_score"],
                "delta": row["delta"],
                "delta_percentage_points": row["delta_percentage_points"],
                "n_items": row["n_pairs"],
                "item_ids_sha256": row["item_ids_sha256"],
                "pair_sha256": row["evidence"]["candidate_pair_sha256"],
                "baseline_pair_sha256": row["evidence"]["baseline_pair_sha256"],
                "statistics": deepcopy(row["statistics"]),
                "gate": deepcopy(row["gate"]),
            }
        arm["tasks"] = tasks
        arm["config_comparison_sha256"] = comparison["comparison_sha256"]
        arm["overall_verdict"] = comparison["overall_verdict"]
        arms.append(arm)

    overall = (
        "pass"
        if all(arm["overall_verdict"] == "pass" for arm in arms[1:])
        else "fail"
    )
    environment = snapshot["run"]["environment"]
    sweep = {
        "schema_version": 1,
        "sweep_type": "quantization_task_accuracy",
        "suite": "quantization",
        "run": {
            "run_id": snapshot["run"]["run_id"],
            "run_fingerprint": snapshot["run"]["fingerprint"],
            "record_sha256": snapshot["record_sha256"],
            "git_commit": environment["git_commit"],
            "source_tree_sha256": environment["source_tree_sha256"],
        },
        "methodology_fingerprint": first["methodology_fingerprint"],
        "sweep_protocol": _protocol_identity(),
        "config_comparison_protocol": deepcopy(first["comparison_protocol"]),
        "comparison_runtime": deepcopy(first["comparison_runtime"]),
        "identity_evidence": "operator_declared_not_remotely_attested",
        "native_support": {
            "fp8_e4m3_kv": {
                "status": "disabled_in_this_kairyu_source",
                "reason": kv_dtype_protocol.FP8_KV_DISABLED_REASON,
                "measured_arm_scope": "external_or_experimental_deployment",
            }
        },
        "coverage": {
            "complete": True,
            "required_profiles": list(required_quantization_profiles()),
            "task_order": list(snapshot["tasks"]),
        },
        "gate_policy": {
            "all_tasks_per_candidate_required": True,
            "all_candidates_required": True,
            "task_averaging": "forbidden",
            "direction": "candidate_minus_bf16_reference",
            "one_sided_confidence": 0.95,
            "unit": "score_fraction",
            "display_unit": "percentage_points",
            "tolerances": {
                task: first["gate_policy"]["tolerances"][task]
                for task in snapshot["tasks"]
            },
        },
        "arms": arms,
        # Full per-candidate #365 artifacts make the retained sweep auditable
        # without weakening or duplicating the comparison contract.
        "config_comparisons": comparisons,
        "overall_verdict": overall,
    }
    sweep["sweep_sha256"] = _sha256(sweep)
    _canonical_bytes(sweep)
    return sweep


def _pp(value: float, *, signed: bool = False) -> str:
    points = value * 100.0
    sign = "+" if signed and points > 0 else ""
    return f"{sign}{points:.2f}"


def _cell_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|")


def render_quantization_sweep_markdown(sweep: dict) -> str:
    """Render the retained scheme-major accuracy and gate tables."""

    if not isinstance(sweep, dict) or sweep.get("sweep_type") != "quantization_task_accuracy":
        raise QuantizationSweepError("not a quantization task-accuracy artifact")
    arms = sweep.get("arms")
    if not isinstance(arms, list) or len(arms) != len(_REQUIRED_PROFILES):
        raise QuantizationSweepError("quantization artifact has incomplete arm coverage")
    tasks = sweep.get("coverage", {}).get("task_order")
    if tasks != list(QUANTIZATION_ROW_ORDER):
        raise QuantizationSweepError("quantization artifact has incomplete task coverage")

    run_id = _safe_report_label(sweep["run"]["run_id"], "run id")
    task_headers = [arms[0]["tasks"][task]["display_name"] for task in tasks]
    lines = [
        "# Quantization x task-accuracy sweep",
        "",
        f"- Run: `{run_id}`",
        f"- Source commit: `{sweep['run']['git_commit']}`",
        f"- Methodology fingerprint: `{sweep['methodology_fingerprint']}`",
        f"- Sweep protocol: `{sweep['sweep_protocol']['name']}` "
        f"(`{sweep['sweep_protocol']['sha256']}`)",
        f"- Overall verdict: **{sweep['overall_verdict'].upper()}**",
        "",
        "> **Deployment and quantization identities are operator-declared.** "
        "The retained manifest SHA-256 values are fingerprinted, but the "
        "benchmark client does not remotely attest the checkpoint, kernels, "
        "compute dtype, KV dtype, or replica-pool homogeneity.",
        "",
        "> `fp8_e4m3` KV is disabled by this Kairyu source after its failed "
        "quality bake. Its required row therefore describes only an explicitly "
        "served external or experimental deployment; this report does not claim "
        "native support.",
        "",
        "Every candidate must pass every task against the exact BF16 reference "
        "items. Tasks and schemes are never averaged.",
        "",
        "## Task accuracy",
        "",
        "| Profile | Weight method | Compute | KV cache | Target | "
        + " | ".join(_cell_text(header) for header in task_headers)
        + " | Verdict |",
        "|---|---|---|---|---|" + "---:|" * len(tasks) + "---|",
    ]
    for arm in arms:
        task_scores = [_pp(arm["tasks"][task]["score"]) for task in tasks]
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell_text(arm["display_name"]),
                    f"`{arm['profile']['weight_method']}`",
                    f"`{arm['profile']['compute_dtype']}`",
                    f"`{arm['profile']['kv_cache_dtype']}`",
                    f"`{_cell_text(arm['target'])}`",
                    *task_scores,
                    arm["overall_verdict"].upper(),
                )
            )
            + " |"
        )

    lines += [
        "",
        "## Paired non-inferiority gates",
        "",
        "| Profile | Task | Score | BF16 | Delta | One-sided 95% LCB | "
        "Tolerance | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in arms[1:]:
        for task in tasks:
            cell = arm["tasks"][task]
            lines.append(
                "| "
                + " | ".join(
                    (
                        _cell_text(arm["display_name"]),
                        _cell_text(cell["display_name"]),
                        _pp(cell["score"]),
                        _pp(cell["baseline_score"]),
                        _pp(cell["delta"], signed=True),
                        _pp(cell["statistics"]["one_sided"]["lower"], signed=True),
                        _pp(cell["gate"]["tolerance"]),
                        cell["gate"]["verdict"].upper(),
                    )
                )
                + " |"
            )

    lines += ["", "## Evidence bindings", ""]
    for arm in arms:
        for task in tasks:
            cell = arm["tasks"][task]
            lines.append(
                f"- `{arm['key']}` / `{task}`: n={cell['n_items']}, "
                f"items `{cell['item_ids_sha256']}`, pair `{cell['pair_sha256']}`"
            )
    lines += ["", f"Artifact SHA-256: `{sweep['sweep_sha256']}`", ""]
    return "\n".join(lines)


__all__ = [
    "QuantizationSweepError",
    "build_quantization_sweep",
    "render_quantization_sweep_markdown",
    "required_quantization_profiles",
]
