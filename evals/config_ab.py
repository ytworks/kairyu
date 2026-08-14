"""Fail-closed configuration A/B quality comparisons.

Unlike :mod:`evals.compare`'s cross-commit report, configuration A/B
intentionally permits the selected target declarations to differ.  Everything
else remains source-, methodology-, and item-bound.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from evals import aggregate as aggregate_protocol
from evals import config_compare as config_statistics
from evals import history as scoreboard_history
from evals import types as types_protocol
from evals.config_compare import (
    newcombe_paired_binary,
    paired_bounded_bootstrap,
    paired_clustered_bounded_bootstrap,
)
from evals.history import pair_result_binding, validate_run_metadata
from evals.types import PairResult, pair_result_evidence_error

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_TARGET_ARM_FIELDS = frozenset(
    {"name", "base_url", "api_key_env", "served_config", "quantization"}
)
_SOURCE_FIELDS = (
    "git_commit",
    "git_commit_role",
    "source_kind",
    "source_tree_clean",
    "source_tree_sha256",
    "python",
    "kairyu_version",
)


class ConfigComparisonError(ValueError):
    """The A/B inputs cannot support a quality-gate conclusion."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigComparisonError("comparison evidence is not canonical JSON") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _comparison_protocol_identity() -> dict:
    resources: list[dict[str, str]] = []
    for module_path in (
        Path(__file__),
        Path(config_statistics.__file__),
        Path(scoreboard_history.__file__),
        Path(aggregate_protocol.__file__),
        Path(types_protocol.__file__),
        Path(__file__).with_name("store.py"),
    ):
        try:
            payload = module_path.read_bytes()
        except OSError as error:
            raise ConfigComparisonError(
                f"cannot content-bind comparison protocol resource {module_path.name!r}"
            ) from error
        resources.append(
            {
                "resource": module_path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "name": "kairyu_config_ab_v1",
        "resources": resources,
        "sha256": _sha256(resources),
    }


def _mapping(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigComparisonError(f"{field} must be an object")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ConfigComparisonError(f"{field} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ConfigComparisonError(f"{field} contains duplicates")
    return list(value)


def _stable_execution(environment: dict, role: str) -> dict:
    execution = _mapping(environment.get("execution"), f"{role} execution runtime")
    stable = {
        key: value
        for key, value in execution.items()
        if key not in {"available", "availability_detail"}
    }
    if not stable:
        raise ConfigComparisonError(f"{role} execution runtime identity is empty")
    return stable


def _self_judge_lists(board: dict, targets: Sequence[str], role: str) -> tuple[set[str], set[str]]:
    values: list[set[str]] = []
    for field in ("self_judged_targets", "judge_independence_unknown_targets"):
        listed = _string_list(board.get(field), f"{role} {field}")
        if not set(listed) <= set(targets):
            raise ConfigComparisonError(f"{role} {field} contains an unknown target")
        values.append(set(listed))
    return values[0], values[1]


def _snapshot(entry: object, role: str) -> dict:
    payload = _mapping(entry, role)
    run = validate_run_metadata(payload.get("run"), require_clean_source=True)
    board = _mapping(payload.get("scoreboard"), f"{role} scoreboard")
    if board.get("run_id") != run["run_id"] or payload.get("run_id") != run["run_id"]:
        raise ConfigComparisonError(f"{role} run id disagrees across index evidence")
    if (
        board.get("fingerprint") != run["fingerprint"]
        or payload.get("fingerprint") != run["fingerprint"]
    ):
        raise ConfigComparisonError(f"{role} fingerprint disagrees across index evidence")
    if board.get("config") != run["config"] or board.get("environment") != run["environment"]:
        raise ConfigComparisonError(f"{role} scoreboard disagrees with run metadata")
    suite = run["config"].get("suite")
    if board.get("suite") != suite or payload.get("suite") != suite:
        raise ConfigComparisonError(f"{role} suite disagrees across index evidence")
    if board.get("schema_version") != 1:
        raise ConfigComparisonError(f"{role} scoreboard schema is unsupported")

    identity = _mapping(run.get("identity"), f"{role} run identity")
    identity_config = _mapping(identity.get("config"), f"{role} identity config")
    target_configs = identity_config.get("targets")
    if not isinstance(target_configs, list) or any(
        not isinstance(target, dict) for target in target_configs
    ):
        raise ConfigComparisonError(f"{role} target identities are malformed")
    target_labels: list[str] = []
    targets_by_label: dict[str, dict] = {}
    for target in target_configs:
        label = target.get("name") or target.get("model")
        if not isinstance(label, str) or not label or label in targets_by_label:
            raise ConfigComparisonError(f"{role} target labels must be non-empty and unique")
        target_labels.append(label)
        targets_by_label[label] = target

    benchmarks = _string_list(board.get("benchmarks"), f"{role} benchmarks")
    targets = _string_list(board.get("targets"), f"{role} targets")
    if targets != target_labels:
        raise ConfigComparisonError(f"{role} scoreboard target layout disagrees with config")
    adapters = identity.get("adapters")
    if not isinstance(adapters, list) or any(
        not isinstance(adapter, dict) for adapter in adapters
    ):
        raise ConfigComparisonError(f"{role} adapter methodology is malformed")
    adapter_names = [adapter.get("name") for adapter in adapters]
    if adapter_names != benchmarks:
        raise ConfigComparisonError(f"{role} benchmark layout disagrees with methodology")
    for adapter in adapters:
        if type(adapter.get("binary_outcomes")) is not bool:
            raise ConfigComparisonError(
                f"{role} adapter {adapter.get('name')!r} lacks a binary-outcome declaration"
            )
        if "paired_cluster_key" not in adapter:
            raise ConfigComparisonError(
                f"{role} adapter {adapter.get('name')!r} lacks a cluster declaration"
            )
        cluster_key = adapter["paired_cluster_key"]
        if cluster_key is not None and (
            not isinstance(cluster_key, str) or not cluster_key
        ):
            raise ConfigComparisonError(
                f"{role} adapter {adapter.get('name')!r} has a malformed cluster key"
            )
        uses_judge = adapter.get("uses_judge_template")
        if type(uses_judge) is not bool:
            raise ConfigComparisonError(
                f"{role} adapter {adapter.get('name')!r} lacks a judge-use declaration"
            )
        judge_template = adapter.get("judge_template")
        if uses_judge and (not isinstance(judge_template, dict) or not judge_template):
            raise ConfigComparisonError(
                f"{role} adapter {adapter.get('name')!r} has no judge-template identity"
            )
        if not uses_judge and "judge_template" in adapter:
            raise ConfigComparisonError(
                f"{role} non-judge adapter {adapter.get('name')!r} has a judge template"
            )
    display_names = _mapping(board.get("display_names"), f"{role} display names")
    if set(display_names) != set(benchmarks) or any(
        not isinstance(display_names[name], str) or not display_names[name]
        for name in benchmarks
    ):
        raise ConfigComparisonError(f"{role} display-name layout is malformed")
    cells = _mapping(board.get("cells"), f"{role} cells")
    if set(cells) != set(benchmarks):
        raise ConfigComparisonError(f"{role} cell rows do not match its benchmarks")
    for benchmark in benchmarks:
        row = _mapping(cells[benchmark], f"{role} cells[{benchmark!r}]")
        if set(row) != set(targets):
            raise ConfigComparisonError(f"{role} cell columns do not match its targets")

    record_sha256 = payload.get("record_sha256")
    if not isinstance(record_sha256, str) or _DIGEST_RE.fullmatch(record_sha256) is None:
        raise ConfigComparisonError(f"{role} index record requires a SHA-256 identity")
    git_commit = run["environment"].get("git_commit")
    if not isinstance(git_commit, str) or _COMMIT_RE.fullmatch(git_commit) is None:
        raise ConfigComparisonError(f"{role} local harness commit is malformed")
    self_judged, judge_unknown = _self_judge_lists(board, targets, role)
    return {
        "entry": payload,
        "run": run,
        "board": board,
        "suite": suite,
        "benchmarks": benchmarks,
        "targets": targets,
        "targets_by_label": targets_by_label,
        "adapters_by_name": dict(zip(adapter_names, adapters, strict=True)),
        "display_names": dict(display_names),
        "cells": cells,
        "record_sha256": record_sha256,
        "self_judged": self_judged,
        "judge_unknown": judge_unknown,
    }


def _selected_target(snapshot: dict, requested: str | None, role: str) -> tuple[str, dict]:
    labels = snapshot["targets"]
    if requested is None:
        if len(labels) != 1:
            raise ConfigComparisonError(
                f"{role} has {len(labels)} targets; select one explicitly"
            )
        requested = labels[0]
    if requested not in snapshot["targets_by_label"]:
        raise ConfigComparisonError(f"{role} target {requested!r} is not in the indexed run")
    return requested, snapshot["targets_by_label"][requested]


def _served_config(target: dict, role: str) -> dict:
    served = _mapping(target.get("served_config"), f"{role} served_config")
    if set(served) != {"label", "sha256"}:
        raise ConfigComparisonError(f"{role} served_config fields are malformed")
    label = served.get("label")
    digest = served.get("sha256")
    if not isinstance(label, str) or not label or label != label.strip():
        raise ConfigComparisonError(f"{role} served_config label is malformed")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ConfigComparisonError(f"{role} served_config SHA-256 is malformed")
    return {"label": label, "sha256": digest}


def _methodology_projection(snapshot: dict) -> dict:
    identity = deepcopy(snapshot["run"]["identity"])
    config = _mapping(identity.get("config"), "run identity config")
    if "targets" not in config:
        raise ConfigComparisonError("run identity config does not disclose targets")
    config.pop("targets")
    return identity


def _target_request_policy(target: dict) -> dict:
    return {key: value for key, value in target.items() if key not in _TARGET_ARM_FIELDS}


def _validate_run_pair(base: dict, candidate: dict) -> tuple[str, dict]:
    if base["suite"] != candidate["suite"]:
        raise ConfigComparisonError("suite differs between baseline and candidate")
    if base["benchmarks"] != candidate["benchmarks"]:
        raise ConfigComparisonError("ordered benchmark methodology differs between runs")
    if base["display_names"] != candidate["display_names"]:
        raise ConfigComparisonError("benchmark display names differ between runs")

    base_projection = _methodology_projection(base)
    candidate_projection = _methodology_projection(candidate)
    if _canonical_bytes(base_projection) != _canonical_bytes(candidate_projection):
        raise ConfigComparisonError(
            "methodology differs after removing the intended target configuration variable"
        )
    methodology_fingerprint = _sha256(base_projection)

    base_config = base["run"]["config"]
    candidate_config = candidate["run"]["config"]
    for role, config in (("baseline", base_config), ("candidate", candidate_config)):
        attempts = config.get("attempts")
        if type(attempts) is not int or attempts != 1:
            raise ConfigComparisonError(
                f"{role} configuration A/B requires attempts=1; got {attempts!r}"
            )
        if config.get("smoke") is not False or config.get("limit") is not None:
            raise ConfigComparisonError(f"{role} is a subset run, not a full-dataset gate")
        if config.get("offline_fixtures") is not False:
            raise ConfigComparisonError(f"{role} offline fixtures are not measurements")

    base_environment = base["run"]["environment"]
    candidate_environment = candidate["run"]["environment"]
    for field in _SOURCE_FIELDS:
        if base_environment.get(field) != candidate_environment.get(field):
            raise ConfigComparisonError(f"local harness/runtime provenance differs at {field}")
    base_execution = _stable_execution(base_environment, "baseline")
    candidate_execution = _stable_execution(candidate_environment, "candidate")
    if _canonical_bytes(base_execution) != _canonical_bytes(candidate_execution):
        raise ConfigComparisonError("execution runtime identity differs between runs")
    return methodology_fingerprint, {
        "measurement_python": base_environment["python"],
        "measurement_execution": base_execution,
    }


def _validate_tolerances(
    tolerances: Mapping[str, float], benchmarks: Sequence[str]
) -> dict[str, float]:
    if not isinstance(tolerances, Mapping) or not tolerances:
        raise ConfigComparisonError("at least one benchmark tolerance is required")
    validated: dict[str, float] = {}
    for benchmark, value in tolerances.items():
        if not isinstance(benchmark, str) or not benchmark:
            raise ConfigComparisonError("tolerance benchmark names must be non-empty strings")
        if benchmark not in benchmarks:
            raise ConfigComparisonError(f"unknown tolerance benchmark {benchmark!r}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigComparisonError(f"tolerance for {benchmark!r} must be finite")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ConfigComparisonError(
                f"tolerance for {benchmark!r} must be within [0, 1] score units"
            )
        validated[benchmark] = number
    return validated


def _binding(entry: dict, benchmark: str, target: str, role: str) -> dict:
    pairs = entry.get("pairs")
    if not isinstance(pairs, list):
        raise ConfigComparisonError(f"{role} index pair bindings are malformed")
    matched = [
        binding
        for binding in pairs
        if isinstance(binding, dict)
        and binding.get("benchmark") == benchmark
        and binding.get("target") == target
    ]
    if len(matched) != 1:
        raise ConfigComparisonError(
            f"{role} requires exactly one indexed pair for {benchmark!r}/{target!r}"
        )
    digest = matched[0].get("pair_sha256")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise ConfigComparisonError(f"{role} pair binding has no valid SHA-256")
    return matched[0]


def _validated_pair_binding(
    entry: dict,
    pair: PairResult,
    *,
    benchmark: str,
    target: str,
    role: str,
    declared_binary: bool,
) -> dict:
    """Prove that the in-memory pair is the exact evidence named by the index."""

    indexed = _binding(entry, benchmark, target, role)
    try:
        actual = pair_result_binding(
            pair,
            declared_binary=declared_binary,
        )
    except ValueError as error:
        raise ConfigComparisonError(f"{role} pair cannot be rebound safely") from error
    if _canonical_bytes(actual) != _canonical_bytes(indexed):
        raise ConfigComparisonError(
            f"{role} pair does not match its complete indexed binding"
        )
    return indexed


def _paired_items(
    baseline: PairResult,
    candidate: PairResult,
    *,
    benchmark: str,
    baseline_target: str,
    candidate_target: str,
) -> tuple[list[str], list[float], list[float]]:
    for role, pair, target in (
        ("baseline", baseline, baseline_target),
        ("candidate", candidate, candidate_target),
    ):
        error = pair_result_evidence_error(
            pair,
            expected_benchmark=benchmark,
            expected_target=target,
            require_history_safe_counts=True,
        )
        if error is not None:
            raise ConfigComparisonError(f"{role} pair evidence is invalid: {error}")
        if pair.status != "completed" or pair.reason is not None:
            raise ConfigComparisonError(f"{role} pair is not a complete measurement")
        if pair.cross_run_policy != "allowed" or pair.cross_run_reason is not None:
            raise ConfigComparisonError(f"{role} pair runtime provenance withholds comparison")
        if pair.metrics.get("n_total") != len(pair.items):
            raise ConfigComparisonError(f"{role} n_total disagrees with item evidence")
        if pair.metrics.get("n_scored") != len(pair.items):
            raise ConfigComparisonError(f"{role} n_scored disagrees with item evidence")
        if not pair.items:
            raise ConfigComparisonError(f"{role} pair contains no item evidence")

    if baseline.methodology != candidate.methodology:
        raise ConfigComparisonError("pair methodology differs between arms")
    if baseline.annotations != candidate.annotations:
        raise ConfigComparisonError("pair annotations differ between arms")
    if baseline.comparable != candidate.comparable:
        raise ConfigComparisonError("pair comparability declarations differ between arms")
    if baseline.incomparable_reasons != candidate.incomparable_reasons:
        raise ConfigComparisonError("pair comparability boundaries differ between arms")
    if not baseline.comparable and not baseline.incomparable_reasons:
        raise ConfigComparisonError("non-comparable pairs have no shared diagnostic boundary")

    by_role: list[dict[str, float]] = []
    for role, pair in (("baseline", baseline), ("candidate", candidate)):
        scores: dict[str, float] = {}
        for item in pair.items:
            if not isinstance(item.item_id, str) or not item.item_id:
                raise ConfigComparisonError(f"{role} pair contains an empty item id")
            if item.item_id in scores:
                raise ConfigComparisonError(f"{role} pair contains duplicate item ids")
            if item.status != "completed" or item.score is None:
                raise ConfigComparisonError(f"{role} item {item.item_id!r} is not scored")
            score = float(item.score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ConfigComparisonError(f"{role} item {item.item_id!r} score is invalid")
            scores[item.item_id] = score
        mean = sum(scores.values()) / len(scores)
        if pair.score is None or not math.isclose(
            pair.score, mean, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ConfigComparisonError(f"{role} aggregate score disagrees with item evidence")
        by_role.append(scores)

    baseline_scores, candidate_scores = by_role
    if set(baseline_scores) != set(candidate_scores):
        missing_candidate = sorted(set(baseline_scores) - set(candidate_scores))
        missing_baseline = sorted(set(candidate_scores) - set(baseline_scores))
        raise ConfigComparisonError(
            "paired item-id sets differ "
            f"(candidate missing={missing_candidate!r}, baseline missing={missing_baseline!r})"
        )
    item_ids = sorted(baseline_scores)
    return (
        item_ids,
        [baseline_scores[item_id] for item_id in item_ids],
        [candidate_scores[item_id] for item_id in item_ids],
    )


def _judging_is_independent(snapshot: dict, benchmark: str, target: str, role: str) -> None:
    adapter = snapshot["adapters_by_name"].get(benchmark)
    if not isinstance(adapter, dict):
        raise ConfigComparisonError(f"unknown benchmark adapter {benchmark!r}")
    if adapter["uses_judge_template"] is False:
        return
    judge_template = adapter["judge_template"]
    if not isinstance(judge_template, dict) or not judge_template:
        raise ConfigComparisonError(f"{role} judge-template identity is malformed")
    if target in snapshot["self_judged"]:
        raise ConfigComparisonError(f"{role} target is self-judged for {benchmark!r}")
    if target in snapshot["judge_unknown"]:
        raise ConfigComparisonError(
            f"{role} judge independence is unknown for {benchmark!r}"
        )


def _paired_cluster_ids(item_ids: Sequence[str], cluster_key: str) -> list[str]:
    if cluster_key != "item_id_before_final_dot_v1":
        raise ConfigComparisonError(f"unsupported paired cluster key {cluster_key!r}")
    clusters: list[str] = []
    for item_id in item_ids:
        cluster_id, separator, member_id = item_id.rpartition(".")
        if not separator or not cluster_id or not member_id:
            raise ConfigComparisonError(
                f"item id {item_id!r} does not satisfy paired cluster key {cluster_key!r}"
            )
        clusters.append(cluster_id)
    return clusters


def build_config_comparison(
    baseline_entry: dict,
    candidate_entry: dict,
    *,
    baseline_pairs: Mapping[str, PairResult],
    candidate_pairs: Mapping[str, PairResult],
    tolerances: Mapping[str, float],
    baseline_target: str | None = None,
    candidate_target: str | None = None,
) -> dict:
    """Build a source- and item-bound configuration non-inferiority gate.

    ``tolerances`` use score-fraction units (``0.01`` is one percentage point).
    Callers must obtain entries from ``ResultStore.load_scoreboard_index`` and
    pairs from ``ResultStore.load_indexed_pair``.
    """

    base = _snapshot(baseline_entry, "baseline")
    candidate = _snapshot(candidate_entry, "candidate")
    methodology_fingerprint, runtime = _validate_run_pair(base, candidate)
    baseline_label, baseline_config = _selected_target(base, baseline_target, "baseline")
    candidate_label, candidate_config = _selected_target(
        candidate, candidate_target, "candidate"
    )
    baseline_served = _served_config(baseline_config, "baseline")
    candidate_served = _served_config(candidate_config, "candidate")
    if baseline_served["sha256"] == candidate_served["sha256"]:
        raise ConfigComparisonError("baseline and candidate served_config digests are identical")
    if _canonical_bytes(_target_request_policy(baseline_config)) != _canonical_bytes(
        _target_request_policy(candidate_config)
    ):
        raise ConfigComparisonError(
            "target request policy differs; only deployment identity may vary in config A/B"
        )

    policy = _validate_tolerances(tolerances, base["benchmarks"])
    if set(baseline_pairs) != set(policy) or set(candidate_pairs) != set(policy):
        raise ConfigComparisonError("raw pair evidence must exactly cover configured gates")

    rows: list[dict] = []
    for benchmark in base["benchmarks"]:
        if benchmark not in policy:
            continue
        _judging_is_independent(base, benchmark, baseline_label, "baseline")
        _judging_is_independent(candidate, benchmark, candidate_label, "candidate")
        baseline_pair = baseline_pairs[benchmark]
        candidate_pair = candidate_pairs[benchmark]
        adapter_identity = base["adapters_by_name"][benchmark]
        declared_binary = adapter_identity["binary_outcomes"]
        cluster_key = adapter_identity.get("paired_cluster_key")
        baseline_binding = _validated_pair_binding(
            base["entry"],
            baseline_pair,
            benchmark=benchmark,
            target=baseline_label,
            role="baseline",
            declared_binary=declared_binary,
        )
        candidate_binding = _validated_pair_binding(
            candidate["entry"],
            candidate_pair,
            benchmark=benchmark,
            target=candidate_label,
            role="candidate",
            declared_binary=declared_binary,
        )
        item_ids, old_scores, new_scores = _paired_items(
            baseline_pair,
            candidate_pair,
            benchmark=benchmark,
            baseline_target=baseline_label,
            candidate_target=candidate_label,
        )
        item_ids_sha256 = _sha256(item_ids)
        baseline_score = sum(old_scores) / len(old_scores)
        candidate_score = sum(new_scores) / len(new_scores)
        delta = candidate_score - baseline_score
        if declared_binary:
            if any(score not in (0.0, 1.0) for score in (*old_scores, *new_scores)):
                raise ConfigComparisonError(
                    f"binary benchmark {benchmark!r} contains fractional item evidence"
                )
        if cluster_key is not None:
            cluster_ids = _paired_cluster_ids(item_ids, cluster_key)
            statistics = paired_clustered_bounded_bootstrap(
                old_scores,
                new_scores,
                item_ids=item_ids,
                cluster_ids=cluster_ids,
                baseline_pair_sha256=baseline_binding["pair_sha256"],
                candidate_pair_sha256=candidate_binding["pair_sha256"],
            )
            if statistics["item_ids_sha256"] != item_ids_sha256:
                raise ConfigComparisonError(
                    "cluster bootstrap item-id identity disagrees with evidence"
                )
            metric_kind = "clustered_binary" if declared_binary else "clustered_score"
        elif declared_binary:
            paired_counts = {
                "both_pass": 0,
                "candidate_win": 0,
                "candidate_loss": 0,
                "both_fail": 0,
            }
            for baseline_value, candidate_value in zip(
                old_scores, new_scores, strict=True
            ):
                if baseline_value == 1.0 and candidate_value == 1.0:
                    paired_counts["both_pass"] += 1
                elif baseline_value == 0.0 and candidate_value == 1.0:
                    paired_counts["candidate_win"] += 1
                elif baseline_value == 1.0 and candidate_value == 0.0:
                    paired_counts["candidate_loss"] += 1
                else:
                    paired_counts["both_fail"] += 1
            statistics = newcombe_paired_binary(**paired_counts)
            metric_kind = "binary"
        else:
            statistics = paired_bounded_bootstrap(
                old_scores,
                new_scores,
                item_ids=item_ids,
                baseline_pair_sha256=baseline_binding["pair_sha256"],
                candidate_pair_sha256=candidate_binding["pair_sha256"],
            )
            if statistics["item_ids_sha256"] != item_ids_sha256:
                raise ConfigComparisonError("bootstrap item-id identity disagrees with evidence")
            metric_kind = "bounded_score"
        if not math.isclose(
            statistics["delta"], delta, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ConfigComparisonError("statistical delta disagrees with paired evidence")
        tolerance = policy[benchmark]
        lower_bound = statistics["one_sided"]["lower"]
        passed = lower_bound >= -tolerance
        rows.append(
            {
                "benchmark": benchmark,
                "display_name": base["display_names"][benchmark],
                "metric_kind": metric_kind,
                "paired_cluster_key": cluster_key,
                "n_pairs": len(item_ids),
                "item_ids_sha256": item_ids_sha256,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "delta": delta,
                "delta_percentage_points": delta * 100.0,
                "statistics": statistics,
                "evidence": {
                    "baseline_pair_sha256": baseline_binding["pair_sha256"],
                    "candidate_pair_sha256": candidate_binding["pair_sha256"],
                },
                "gate": {
                    "tolerance": tolerance,
                    "tolerance_percentage_points": tolerance * 100.0,
                    "threshold": -tolerance,
                    "one_sided_confidence": 0.95,
                    "verdict": "pass" if passed else "fail",
                    "reason": None if passed else "insufficient evidence of non-inferiority",
                },
            }
        )

    overall = "pass" if all(row["gate"]["verdict"] == "pass" for row in rows) else "fail"
    environment = base["run"]["environment"]
    comparison = {
        "schema_version": 1,
        "comparison_type": "config_ab",
        "direction": "candidate_minus_baseline",
        "suite": base["suite"],
        "methodology_fingerprint": methodology_fingerprint,
        "comparison_protocol": _comparison_protocol_identity(),
        "confidence": {
            "two_sided_interval": 0.95,
            "one_sided_noninferiority_lcb": 0.95,
        },
        "baseline": {
            "run_id": base["run"]["run_id"],
            "run_fingerprint": base["run"]["fingerprint"],
            "record_sha256": base["record_sha256"],
            "pair_target": baseline_label,
            "target_config": deepcopy(baseline_config),
            "target_config_sha256": _sha256(baseline_config),
            "served_config": baseline_served,
            "git_commit": environment["git_commit"],
            "source_tree_sha256": environment["source_tree_sha256"],
        },
        "candidate": {
            "run_id": candidate["run"]["run_id"],
            "run_fingerprint": candidate["run"]["fingerprint"],
            "record_sha256": candidate["record_sha256"],
            "pair_target": candidate_label,
            "target_config": deepcopy(candidate_config),
            "target_config_sha256": _sha256(candidate_config),
            "served_config": candidate_served,
            "git_commit": candidate["run"]["environment"]["git_commit"],
            "source_tree_sha256": candidate["run"]["environment"][
                "source_tree_sha256"
            ],
        },
        "comparison_runtime": {
            **runtime,
            "comparator_python": platform.python_version(),
            "comparator_implementation": platform.python_implementation(),
        },
        "served_config_provenance": "operator_declared_not_remotely_attested",
        "gate_policy": {
            "all_required": True,
            "on_unevaluable": "error",
            "unit": "score_fraction",
            "display_unit": "percentage_points",
            "tolerances": dict(policy),
        },
        "benchmarks": rows,
        "overall_verdict": overall,
    }
    comparison["comparison_sha256"] = _sha256(comparison)
    return comparison


def _pp(value: float, *, signed: bool = False) -> str:
    number = value * 100.0
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:.2f}"


def render_config_comparison_markdown(comparison: dict) -> str:
    """Render a stable human-readable A/B gate report."""

    if comparison.get("comparison_type") != "config_ab":
        raise ConfigComparisonError("not a config_ab comparison artifact")
    baseline = _mapping(comparison.get("baseline"), "baseline")
    candidate = _mapping(comparison.get("candidate"), "candidate")
    lines = [
        "# Configuration A/B quality gate",
        "",
        f"- Direction: candidate `{candidate['run_id']}` minus baseline "
        f"`{baseline['run_id']}`",
        f"- Suite: `{comparison['suite']}`",
        f"- Methodology fingerprint: `{comparison['methodology_fingerprint']}`",
        f"- Comparison protocol: `{comparison['comparison_protocol']['name']}` "
        f"(`{comparison['comparison_protocol']['sha256']}`)",
        f"- Baseline arm: `{baseline['pair_target']}` / "
        f"`{baseline['served_config']['label']}` "
        f"(`{baseline['served_config']['sha256']}`)",
        f"- Candidate arm: `{candidate['pair_target']}` / "
        f"`{candidate['served_config']['label']}` "
        f"(`{candidate['served_config']['sha256']}`)",
        f"- Overall verdict: **{comparison['overall_verdict'].upper()}**",
        "",
        "> **Served configuration identity is operator-declared.** The recorded "
        "SHA-256 values are fingerprinted evidence, but this benchmark client does "
        "not remotely attest which build or quantization configuration answered "
        "each request.",
        "",
        "A row passes only when its one-sided 95% lower confidence bound is at "
        "least `-tolerance`. Scores, deltas, bounds, and tolerances below are "
        "percentage points; decisions use unrounded score fractions.",
        "",
        "| Benchmark | Candidate | Baseline | Δ | Two-sided 95% CI | "
        "One-sided 95% LCB | Tolerance | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison.get("benchmarks") or []:
        statistics = row["statistics"]
        interval = statistics["two_sided"]
        lcb = statistics["one_sided"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["display_name"]).replace("|", "\\|"),
                    _pp(row["candidate_score"]),
                    _pp(row["baseline_score"]),
                    _pp(row["delta"], signed=True),
                    f"[{_pp(interval['lower'])}, {_pp(interval['upper'])}]",
                    _pp(lcb["lower"], signed=True),
                    _pp(row["gate"]["tolerance"]),
                    row["gate"]["verdict"].upper(),
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Evidence bindings",
        "",
    ]
    for row in comparison.get("benchmarks") or []:
        lines.append(
            f"- `{row['benchmark']}`: n={row['n_pairs']}, "
            f"items `{row['item_ids_sha256']}`, baseline pair "
            f"`{row['evidence']['baseline_pair_sha256']}`, candidate pair "
            f"`{row['evidence']['candidate_pair_sha256']}`, method "
            f"`{row['statistics']['method']}`"
        )
    lines += ["", f"Artifact SHA-256: `{comparison['comparison_sha256']}`", ""]
    return "\n".join(lines)
