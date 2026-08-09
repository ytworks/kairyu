"""Accuracy report: a finished run next to the published Fugu-release scores.

The scoreboard says what kairyu measured. This says how that sits against the
numbers Sakana published — with the deltas, and with every reason the delta may
not mean what it looks like: a skipped or partial cell, a substituted dataset,
an uncompiled checker, a single attempt where the published figure used four.

Rules the renderer keeps:

- a missing measurement is `—`, never zero and never omitted;
- a `partial` cell is marked, because its denominator is not the full set;
- a row kairyu measures differently is marked NOT COMPARABLE and its delta is
  withheld rather than printed with a caveat somewhere further down;
- the baselines are labelled provider-reported, because the page says they are.
"""

from __future__ import annotations

import json
import math
import re

from kairyu.bench.reference import (
    NOT_COMPARABLE,
    PROVIDER_REPORTED,
    PUBLISHED_VARIANTS,
    published,
    published_models,
    reference_metadata,
)

#: Published column used as the delta baseline (the model the suite is named for).
DELTA_AGAINST = "Fugu"

_PAIR_STATUSES = frozenset({"completed", "partial", "skipped", "failed"})
_CROSS_RUN_POLICIES = frozenset(
    {"allowed", "withheld_unresolved_runtime", "withheld_unpinned_execution"}
)
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_OFFLINE_FIXTURE_NOTE = "offline fixture scores are synthetic diagnostics, not measurements"
_RESOLVED_EXECUTION_FIELDS = (
    "resolved_image_id",
    "image_os",
    "image_architecture",
    "image_variant",
)


def build_comparison(scoreboard: dict) -> dict:
    """Scoreboard + published values -> a comparison document."""
    from kairyu.bench.adapters import suite_info

    definition = suite_info(scoreboard.get("suite", "accuracy"))
    if not definition.published_comparison:
        raise ValueError(f"suite {definition.name!r} has no published comparison")
    targets = list(scoreboard.get("targets") or [])
    rows = []
    for benchmark in scoreboard.get("benchmarks") or []:
        cells = (scoreboard.get("cells") or {}).get(benchmark, {})
        reference = published(benchmark)
        measured = {}
        for target in targets:
            cell = cells.get(target) or {}
            score = cell.get("score")
            # Comparability is carried by the cell (static substitution, run-time
            # substitution, subset/fixture run); the static map below is only a
            # fallback for scoreboards written before that field existed.
            declared = cell.get("comparable")
            reasons = list(cell.get("incomparable_reasons") or [])
            if declared is None:
                declared = benchmark not in NOT_COMPARABLE
                if benchmark in NOT_COMPARABLE:
                    reasons = [NOT_COMPARABLE[benchmark]]
            measured[target] = {
                "score": None if score is None else round(score * 100, 1),
                "status": cell.get("status", "skipped"),
                "n": cell.get("n"),
                "n_scored": cell.get("n_scored"),
                "reason": cell.get("reason"),
                "incomparable_reasons": reasons,
                "comparable": bool(declared) and cell.get("status") == "completed",
            }
        rows.append(
            {
                "benchmark": benchmark,
                "display_name": (scoreboard.get("display_names") or {}).get(benchmark, benchmark),
                "measured": measured,
                "published": reference,
                "published_models": list(published_models(benchmark)),
                "variant": PUBLISHED_VARIANTS.get(benchmark),
                "not_comparable": NOT_COMPARABLE.get(benchmark),
                "deltas": {
                    target: _delta(values, reference) for target, values in measured.items()
                },
            }
        )
    return {
        "run_id": scoreboard.get("run_id"),
        "suite": scoreboard.get("suite"),
        "targets": targets,
        "delta_against": DELTA_AGAINST,
        "reference": reference_metadata(),
        "rows": rows,
        # the scoreboard's own footnotes: substituted datasets, uncompiled
        # checkers, self-judging, degraded cells
        "methodology_notes": list(scoreboard.get("footnotes") or []),
    }


def _delta(values: dict, reference: dict[str, float]) -> float | None:
    """Measured minus published `Fugu`, or None when the comparison is unsound.

    "Unsound" covers every way a cell can fail to be a full-suite measurement:
    no score, a partial or failed cell, a substituted dataset or harness, and a
    subset or fixture run.
    """
    baseline = reference.get(DELTA_AGAINST)
    if values["score"] is None or baseline is None:
        return None
    if not values["comparable"]:
        return None
    return round(values["score"] - baseline, 1)


def _measured_text(values: dict) -> str:
    """Score with its status marker; status stays visible without a score."""
    score = values["score"]
    marker = {"partial": "*", "failed": "!"}.get(values["status"], "")
    if score is None:
        # a failed cell usually has no score; without the marker it would read
        # identically to a cell that was never measured
        return f"—{marker}" if marker else "—"
    text = f"{score:.1f}{marker}"
    if values["n"]:
        scored = values.get("n_scored")
        if scored is not None and scored != values["n"]:
            text += f" ({scored}/{values['n']})"
        else:
            text += f" (n={values['n']})"
    return text


def _delta_text(delta: float | None, values: dict) -> str:
    """A delta only for a cell that is comparable; otherwise why not."""
    if delta is None:
        if values["score"] is None:
            return "—"
        return "n/c"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"


def render_comparison_markdown(comparison: dict) -> str:
    reference = comparison["reference"]
    targets = comparison["targets"]
    lines = [
        f"# Accuracy vs published Fugu scores — run {comparison['run_id']}",
        "",
        f"Published values transcribed from {reference['source_url']} on "
        f"{reference['retrieved_on']} (the page publishes them as images, so they "
        "cannot be fetched programmatically).",
        "",
        f"`Δ` is measured minus published **{comparison['delta_against']}**, and "
        "only for a cell that is a full-suite measurement of the same thing. "
        "`—` = not measured, `*` = partial, `!` = failed, `n/c` = measured but "
        "not comparable (see below). Item counts are shown next to each score.",
        "",
    ]
    lines += _banner(comparison)

    header = ["Benchmark"] + list(targets)
    published_columns: list[str] = []
    for row in comparison["rows"]:
        for model in row["published_models"]:
            if model not in published_columns:
                published_columns.append(model)
    header += published_columns + [f"Δ {target}" for target in targets]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")

    for row in comparison["rows"]:
        cells = [row["display_name"]]
        cells += [_measured_text(row["measured"][target]) for target in targets]
        for model in published_columns:
            value = row["published"].get(model)
            cells.append("—" if value is None else f"{value:.1f}")
        cells += [_delta_text(row["deltas"][target], row["measured"][target]) for target in targets]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Reading this table", ""]
    for note in reference["footnotes"]:
        lines.append(f"- {note}")
    lines.append(
        "- Baselines "
        f"({', '.join(sorted(PROVIDER_REPORTED))}) are provider-reported under "
        "unknown conditions; treat them as orientation, not as a measurement "
        "made under this harness."
    )

    caveats = _caveats(comparison)
    if caveats:
        lines += ["", "## Why a delta may not mean parity", ""]
        lines += [f"- {text}" for text in caveats]

    notes = comparison.get("methodology_notes") or []
    if notes:
        lines += ["", "## Methodology notes from this run", ""]
        lines += [f"- {note}" for note in notes]
    lines.append("")
    return "\n".join(lines)


def _banner(comparison: dict) -> list[str]:
    """Reasons that apply to every cell, stated once and up front."""
    shared: list[str] | None = None
    for row in comparison["rows"]:
        for values in row["measured"].values():
            reasons = list(values.get("incomparable_reasons") or [])
            shared = reasons if shared is None else [r for r in shared if r in reasons]
    if not shared:
        return []
    return (
        [
            "> **This run is not a full-suite measurement**, so no cell is compared "
            "with a published score.",
            ">",
        ]
        + [f"> - {reason}" for reason in shared]
        + [""]
    )


def _caveats(comparison: dict) -> list[str]:
    """Per-row reasons a delta is not a like-for-like comparison."""
    caveats: list[str] = []
    for row in comparison["rows"]:
        name = row["display_name"]
        if row["not_comparable"]:
            caveats.append(f"**{name}**: NOT COMPARABLE — {row['not_comparable']}.")
        for target, values in row["measured"].items():
            for reason in values.get("incomparable_reasons") or []:
                caveats.append(f"**{name}** × {target}: NOT COMPARABLE — {reason}.")
        variant = row["variant"]
        if variant:
            published_variant = ", ".join(
                f"{model} {score:.1f}" for model, score in variant["scores"].items()
            )
            caveats.append(
                f"**{name}**: the release also publishes a {variant['label']} "
                f"({published_variant}); the row above uses the headline table."
            )
        for target, values in row["measured"].items():
            if values["status"] in ("partial", "failed") and values["reason"]:
                caveats.append(f"**{name}** × {target}: {values['status']} — {values['reason']}.")
            elif values["status"] == "skipped" and values["reason"]:
                caveats.append(f"**{name}** × {target}: not measured — {values['reason']}.")
    return caveats


# -- Run-to-run comparison -----------------------------------------------------


def _required_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return list(value)


def _run_snapshot(value: object, role: str) -> dict:
    """Validate and unwrap one scoreboard-index entry.

    The durable index stores metadata next to an immutable ``scoreboard``
    snapshot.  Accepting an augmented scoreboard as well keeps the pure API
    useful to callers that already performed the same provenance validation.
    In either form, the comparison itself refuses anonymous fingerprints or
    commits: without those, a cross-commit delta has no defensible identity.
    """

    payload = _required_mapping(value, role)
    if "scoreboard" in payload:
        board = _required_mapping(payload["scoreboard"], f"{role}.scoreboard")
        metadata = payload
        run = payload.get("run")
    else:
        board = payload
        metadata = payload
        run = None

    run_id = metadata.get("run_id", board.get("run_id"))
    suite = metadata.get("suite", board.get("suite"))
    fingerprint = metadata.get("fingerprint", board.get("fingerprint"))
    environment = _required_mapping(board.get("environment"), f"{role}.scoreboard.environment")
    git_commit = metadata.get("git_commit", environment.get("git_commit"))

    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{role} run_id must be a non-empty string")
    if board.get("run_id") != run_id:
        raise ValueError(f"{role} run_id disagrees with its scoreboard")
    if not isinstance(suite, str) or not suite:
        raise ValueError(f"{role} suite must be a non-empty string")
    if board.get("suite") != suite:
        raise ValueError(f"{role} suite disagrees with its scoreboard")
    if not isinstance(fingerprint, str) or _FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"{role} fingerprint must be a lowercase SHA-256 digest")
    if board.get("fingerprint") != fingerprint:
        raise ValueError(f"{role} fingerprint disagrees with its scoreboard")
    if not isinstance(git_commit, str) or _GIT_COMMIT_RE.fullmatch(git_commit) is None:
        raise ValueError(f"{role} git_commit must be a full lowercase Git object id")
    if environment.get("git_commit") != git_commit:
        raise ValueError(f"{role} git_commit disagrees with its scoreboard environment")

    if run is not None:
        run = _required_mapping(run, f"{role}.run")
        if run.get("run_id") != run_id:
            raise ValueError(f"{role} run_id disagrees with run metadata")
        if run.get("fingerprint") != fingerprint:
            raise ValueError(f"{role} fingerprint disagrees with run metadata")
        run_environment = _required_mapping(run.get("environment"), f"{role}.run.environment")
        if run_environment.get("git_commit") != git_commit:
            raise ValueError(f"{role} git_commit disagrees with run metadata")
        if run_environment != environment:
            raise ValueError(f"{role} scoreboard environment disagrees with run metadata")
        run_config = _required_mapping(run.get("config"), f"{role}.run.config")
        if run_config.get("suite") != suite:
            raise ValueError(f"{role} suite disagrees with run metadata")

    schema_version = board.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"{role} scoreboard schema_version must be an integer")
    if schema_version != 1:
        raise ValueError(f"{role} scoreboard schema_version {schema_version!r} is unsupported")
    targets = _required_string_list(board.get("targets"), f"{role} targets")
    benchmarks = _required_string_list(board.get("benchmarks"), f"{role} benchmarks")
    display_names = _required_mapping(board.get("display_names"), f"{role} display_names")
    if set(display_names) != set(benchmarks) or any(
        not isinstance(display_names[name], str) or not display_names[name] for name in benchmarks
    ):
        raise ValueError(
            f"{role} display_names must contain exactly one non-empty name for every benchmark"
        )
    cells = _required_mapping(board.get("cells"), f"{role} cells")
    if set(cells) != set(benchmarks):
        raise ValueError(f"{role} cells do not match its benchmark structure")
    for benchmark in benchmarks:
        by_target = _required_mapping(cells[benchmark], f"{role} cells[{benchmark!r}]")
        if set(by_target) != set(targets):
            raise ValueError(f"{role} cells[{benchmark!r}] do not match its target structure")
        for target in targets:
            _validate_run_cell(
                by_target[target],
                f"{role} cells[{benchmark!r}][{target!r}]",
            )

    config = board.get("config")
    offline_fixtures = isinstance(config, dict) and config.get("offline_fixtures") is True
    if isinstance(run, dict):
        offline_fixtures = offline_fixtures or run["config"].get("offline_fixtures") is True
    if not offline_fixtures:
        offline_fixtures = any(
            "offline fixture" in reason.lower()
            for benchmark in benchmarks
            for target in targets
            for reason in _normalized_reasons(cells[benchmark][target])
        )

    footnotes = board.get("footnotes") or []
    if not isinstance(footnotes, list) or any(not isinstance(note, str) for note in footnotes):
        raise ValueError(f"{role} footnotes must be a list of strings")

    return {
        "run_id": run_id,
        "suite": suite,
        "fingerprint": fingerprint,
        "git_commit": git_commit,
        "created_at": environment.get("created_at"),
        "kairyu_version": environment.get("kairyu_version"),
        "comparison_runtime": _comparison_runtime(environment, role),
        "schema_version": schema_version,
        "targets": targets,
        "benchmarks": benchmarks,
        "display_names": dict(display_names),
        "cells": cells,
        "footnotes": list(footnotes),
        "offline_fixtures": offline_fixtures,
    }


def _comparison_runtime(environment: dict, role: str) -> dict:
    """Runtime fields that must match before a commit delta is meaningful."""
    python = environment.get("python")
    if not isinstance(python, str) or not python:
        raise ValueError(f"{role} environment requires a non-empty Python version")
    execution = _required_mapping(
        environment.get("execution"),
        f"{role}.scoreboard.environment.execution",
    )
    stable_execution = {
        key: value
        for key, value in execution.items()
        if key not in {"available", "availability_detail"}
    }
    if not stable_execution:
        raise ValueError(f"{role} execution runtime identity is empty")
    availability = execution.get("available")
    if availability is not None and type(availability) is not bool:
        raise ValueError(f"{role} execution availability must be a boolean or null")
    return {
        "python": python,
        "execution": stable_execution,
        "execution_available": availability,
    }


def _shared_comparison_runtime(baseline: dict, candidate: dict) -> dict:
    """Return the runtime identity that is known in both observations.

    A failed Docker probe cannot attest resolved image/platform fields.  When
    either side explicitly records that state, compare the immutable requested
    execution policy and let each code cell's cross-run policy withhold its own
    delta.  Two successful probes still require exact resolved identity.
    """
    if baseline["python"] != candidate["python"]:
        raise ValueError(
            "runtime mismatch: Python or execution-runner identity differs between runs"
        )
    baseline_execution = dict(baseline["execution"])
    candidate_execution = dict(candidate["execution"])
    if (
        baseline["execution_available"] is False
        or candidate["execution_available"] is False
    ):
        for execution in (baseline_execution, candidate_execution):
            for field in _RESOLVED_EXECUTION_FIELDS:
                execution.pop(field, None)
    if baseline_execution != candidate_execution:
        raise ValueError(
            "runtime mismatch: Python or execution-runner identity differs between runs"
        )
    return {"python": baseline["python"], "execution": baseline_execution}


def _validate_run_cell(value: object, label: str) -> None:
    cell = _required_mapping(value, label)
    missing_policy = {"cross_run_policy", "cross_run_reason"} - set(cell)
    if missing_policy:
        raise ValueError(f"{label} is missing cross_run policy fields")
    status = cell.get("status")
    if not isinstance(status, str) or status not in _PAIR_STATUSES:
        raise ValueError(f"{label}.status is invalid")
    comparable = cell.get("comparable", True)
    if not isinstance(comparable, bool):
        raise ValueError(f"{label}.comparable must be a boolean")
    reason = cell.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(f"{label}.reason must be a string or null")
    reasons = cell.get("incomparable_reasons") or []
    if not isinstance(reasons, list | tuple) or any(not isinstance(item, str) for item in reasons):
        raise ValueError(f"{label}.incomparable_reasons must be a string list")
    cross_run_policy = cell["cross_run_policy"]
    cross_run_reason = cell["cross_run_reason"]
    if not isinstance(cross_run_policy, str) or cross_run_policy not in _CROSS_RUN_POLICIES:
        raise ValueError(f"{label}.cross_run_policy is invalid")
    if cross_run_policy == "allowed":
        if cross_run_reason is not None:
            raise ValueError(f"{label}.cross_run_reason must be null when deltas are allowed")
    elif not isinstance(cross_run_reason, str) or not cross_run_reason:
        raise ValueError(f"{label}.cross_run_reason is required when deltas are withheld")


def _normalized_reasons(cell: dict) -> tuple[str, ...]:
    reasons = cell.get("incomparable_reasons") or []
    normalized = {
        " ".join(reason.split()) for reason in reasons if isinstance(reason, str) and reason.strip()
    }
    return tuple(sorted(normalized))


def _normalized_reason(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.split())


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        score = float(value)
    except OverflowError:
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _positive_whole(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, float) or not math.isfinite(value):
        return None
    return int(value) if value > 0 and value.is_integer() else None


def _denominator(cell: dict) -> tuple[int, int] | None:
    total = _positive_whole(cell.get("n"))
    if total is None:
        return None
    raw_scored = cell.get("n_scored")
    scored = total if raw_scored is None else _positive_whole(raw_scored)
    if scored is None or scored > total:
        return None
    return total, scored


def _run_cell(cell: dict) -> dict:
    raw_score = _finite_score(cell.get("score"))
    denominator = _denominator(cell)
    return {
        "score": None if raw_score is None else round(raw_score * 100, 1),
        "raw_score": raw_score,
        "status": cell["status"],
        "n": cell.get("n"),
        "n_scored": cell.get("n_scored"),
        "denominator": None if denominator is None else list(denominator),
        "reason": _normalized_reason(cell.get("reason")),
        "comparable": cell.get("comparable", True),
        "incomparable_reasons": list(_normalized_reasons(cell)),
        "confidence_interval": cell.get("confidence_interval"),
        "cross_run_policy": cell["cross_run_policy"],
        "cross_run_reason": _normalized_reason(cell.get("cross_run_reason")),
    }


def _run_delta(
    baseline: dict,
    candidate: dict,
    *,
    offline_fixtures: bool,
) -> tuple[float | None, str, list[str]]:
    reasons: list[str] = []
    baseline_score = baseline["raw_score"]
    candidate_score = candidate["raw_score"]
    missing = baseline_score is None or candidate_score is None
    if baseline_score is None:
        reasons.append("baseline has no valid finite score in [0, 1]")
    if candidate_score is None:
        reasons.append("candidate has no valid finite score in [0, 1]")

    if baseline["status"] != "completed":
        reasons.append(f"baseline status is {baseline['status']}")
    if candidate["status"] != "completed":
        reasons.append(f"candidate status is {candidate['status']}")
    if offline_fixtures:
        reasons.append(_OFFLINE_FIXTURE_NOTE)

    baseline_policy = baseline["cross_run_policy"]
    candidate_policy = candidate["cross_run_policy"]
    if baseline_policy != "allowed" or candidate_policy != "allowed":
        reasons.append(
            "cross-run policy withholds this delta "
            f"(baseline={baseline_policy!r}: {baseline['cross_run_reason']!r}, "
            f"candidate={candidate_policy!r}: {candidate['cross_run_reason']!r})"
        )

    baseline_reasons = tuple(baseline["incomparable_reasons"])
    candidate_reasons = tuple(candidate["incomparable_reasons"])
    if baseline_reasons != candidate_reasons:
        reasons.append(
            "incomparable reasons differ "
            f"(baseline={list(baseline_reasons)!r}, "
            f"candidate={list(candidate_reasons)!r})"
        )
    if baseline["comparable"] != candidate["comparable"]:
        reasons.append(
            "comparability declarations differ "
            f"(baseline={baseline['comparable']!r}, "
            f"candidate={candidate['comparable']!r})"
        )
    elif baseline["comparable"] is False and not baseline_reasons:
        reasons.append(
            "both cells are declared non-comparable without a shared diagnostic boundary"
        )

    baseline_reason = baseline["reason"]
    candidate_reason = candidate["reason"]
    if baseline_reason != candidate_reason:
        reasons.append(
            f"cell reasons differ (baseline={baseline_reason!r}, candidate={candidate_reason!r})"
        )

    baseline_denominator = baseline["denominator"]
    candidate_denominator = candidate["denominator"]
    if baseline_denominator is None:
        reasons.append("baseline has no valid positive denominator")
    if candidate_denominator is None:
        reasons.append("candidate has no valid positive denominator")
    if (
        baseline_denominator is not None
        and candidate_denominator is not None
        and baseline_denominator != candidate_denominator
    ):
        reasons.append(
            "denominators differ "
            f"(baseline={baseline_denominator!r}, "
            f"candidate={candidate_denominator!r})"
        )

    if reasons:
        return None, "missing" if missing else "not_comparable", reasons
    assert baseline_score is not None and candidate_score is not None
    delta = round((candidate_score - baseline_score) * 100, 1)
    if delta == 0:
        delta = 0.0
    return delta, "comparable", []


def build_run_comparison(baseline: dict, candidate: dict) -> dict:
    """Build a fail-closed ``BASE -> CANDIDATE`` accuracy comparison.

    Fingerprints bind the benchmark configuration and adapter identities but do
    not include the local git commit.  Exact fingerprint equality is therefore
    the cross-commit methodology gate; the remaining structural checks prevent
    malformed snapshots from shifting a target or benchmark onto another cell.
    """

    base = _run_snapshot(baseline, "baseline")
    current = _run_snapshot(candidate, "candidate")
    if base["suite"] != current["suite"]:
        raise ValueError(
            f"suite mismatch: baseline={base['suite']!r}, candidate={current['suite']!r}"
        )
    if base["fingerprint"] != current["fingerprint"]:
        raise ValueError("fingerprint mismatch: the runs do not have the same benchmark identity")
    if base["schema_version"] != current["schema_version"]:
        raise ValueError("scoreboard schema_version mismatch")
    if base["targets"] != current["targets"]:
        raise ValueError("target structure mismatch between baseline and candidate")
    if base["benchmarks"] != current["benchmarks"]:
        raise ValueError("benchmark structure mismatch between baseline and candidate")
    if base["display_names"] != current["display_names"]:
        raise ValueError("benchmark display names differ between baseline and candidate")
    comparison_runtime = _shared_comparison_runtime(
        base["comparison_runtime"], current["comparison_runtime"]
    )

    offline_fixtures = bool(base["offline_fixtures"] or current["offline_fixtures"])
    rows: list[dict] = []
    for benchmark in base["benchmarks"]:
        baseline_cells: dict[str, dict] = {}
        candidate_cells: dict[str, dict] = {}
        deltas: dict[str, float | None] = {}
        delta_states: dict[str, str] = {}
        delta_reasons: dict[str, list[str]] = {}
        for target in base["targets"]:
            old = _run_cell(base["cells"][benchmark][target])
            new = _run_cell(current["cells"][benchmark][target])
            delta, state, reasons = _run_delta(
                old,
                new,
                offline_fixtures=offline_fixtures,
            )
            baseline_cells[target] = old
            candidate_cells[target] = new
            deltas[target] = delta
            delta_states[target] = state
            delta_reasons[target] = reasons
        rows.append(
            {
                "benchmark": benchmark,
                "display_name": base["display_names"][benchmark],
                "baseline": baseline_cells,
                "candidate": candidate_cells,
                "deltas": deltas,
                "delta_states": delta_states,
                "delta_reasons": delta_reasons,
            }
        )

    return {
        "comparison_type": "runs",
        "schema_version": base["schema_version"],
        "suite": base["suite"],
        "fingerprint": base["fingerprint"],
        "baseline": {
            key: base[key] for key in ("run_id", "git_commit", "created_at", "kairyu_version")
        },
        "candidate": {
            key: current[key] for key in ("run_id", "git_commit", "created_at", "kairyu_version")
        },
        "comparison_runtime": comparison_runtime,
        "targets": list(base["targets"]),
        "benchmarks": list(base["benchmarks"]),
        "rows": rows,
        "offline_fixtures": offline_fixtures,
        "baseline_methodology_notes": list(base["footnotes"]),
        "candidate_methodology_notes": list(current["footnotes"]),
    }


def _markdown_table_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _run_cell_text(values: dict) -> str:
    score = values["score"]
    marker = {"partial": "*", "failed": "!"}.get(values["status"], "")
    if score is None:
        return f"—{marker}" if marker else "—"
    text = f"{score:.1f}{marker}"
    denominator = values.get("denominator")
    interval = None
    if values["status"] == "completed" and denominator is not None:
        # Keep the scoreboard and both run-comparison columns on the exact same
        # fail-closed Wilson contract.  A malformed or legacy interval remains
        # an ordinary point estimate rather than being labelled 95% CI.
        from kairyu.bench.aggregate import _stored_wilson_bounds

        interval = _stored_wilson_bounds(
            values.get("confidence_interval"),
            expected_trials=denominator[1],
            expected_score=values.get("raw_score"),
        )
    if interval is not None:
        text += f" [{interval[0] * 100:.1f}–{interval[1] * 100:.1f}]"
    if denominator is not None:
        total, scored = denominator
        text += f" (n={scored}/{total})" if scored != total else f" (n={total})"
    return text


def _run_delta_text(delta: float | None, state: str) -> str:
    if state == "missing":
        return "—"
    if state == "not_comparable":
        return "n/c"
    if state != "comparable" or delta is None:
        raise ValueError(f"invalid run comparison delta state {state!r}")
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"


def _run_comparison_caveats(comparison: dict) -> list[str]:
    caveats: list[str] = []
    baseline_id = comparison["baseline"]["run_id"]
    candidate_id = comparison["candidate"]["run_id"]
    for row in comparison["rows"]:
        name = row["display_name"]
        for target in comparison["targets"]:
            label = f"**{name}** × {target}"
            reasons = row["delta_reasons"][target]
            if reasons:
                caveats.append(f"{label}: delta withheld — {'; '.join(reasons)}.")
            baseline = row["baseline"][target]
            candidate = row["candidate"][target]
            if baseline["reason"]:
                caveats.append(
                    f"{label} in baseline {baseline_id}: "
                    f"{baseline['status']} — {baseline['reason']}."
                )
            if candidate["reason"]:
                caveats.append(
                    f"{label} in candidate {candidate_id}: "
                    f"{candidate['status']} — {candidate['reason']}."
                )
            matched = baseline["incomparable_reasons"]
            if matched and matched == candidate["incomparable_reasons"]:
                caveats.append(
                    f"{label}: both runs carry the same diagnostic boundary — {'; '.join(matched)}."
                )
    return caveats


def render_run_comparison_markdown(comparison: dict) -> str:
    """Render a run comparison without implying target-build provenance."""

    if comparison.get("comparison_type") != "runs":
        raise ValueError("not a run comparison document")
    baseline = _required_mapping(comparison.get("baseline"), "baseline")
    candidate = _required_mapping(comparison.get("candidate"), "candidate")
    targets = _required_string_list(comparison.get("targets"), "targets")
    lines = [
        f"# Accuracy run comparison — {baseline['run_id']} → {candidate['run_id']}",
        "",
        f"- Suite: `{comparison['suite']}`",
        f"- Fingerprint: `{comparison['fingerprint']}`",
        f"- Baseline `{baseline['run_id']}`: local harness commit `{baseline['git_commit']}`",
        f"- Candidate `{candidate['run_id']}`: local harness commit `{candidate['git_commit']}`",
        f"- Comparison runtime: Python `{comparison['comparison_runtime']['python']}`; "
        "execution `"
        + json.dumps(
            comparison["comparison_runtime"]["execution"],
            sort_keys=True,
            separators=(",", ":"),
        )
        + "`",
        "",
        "`Δ` is **candidate minus baseline**. Positive is a higher score; "
        "negative is a regression. Scores are percentages and brackets are "
        "validated 95% Wilson CIs for binary outcomes. `—` = one or both "
        "scores are missing, "
        "`*` = partial, `!` = failed, and `n/c` = measured but not comparable. "
        "Item counts are shown next to each score.",
        "",
        "> **Commit provenance is local-harness-only.** These commit IDs identify "
        "the checkout running the benchmark harness; the served target build or "
        "commit is not attested by these scoreboards.",
        "",
    ]
    if comparison.get("offline_fixtures"):
        lines += [
            "> **Offline fixture scores are not measurements.** All numeric deltas are withheld.",
            "",
        ]

    header = ["Benchmark"]
    header += [f"Candidate {candidate['run_id']} · {target}" for target in targets]
    header += [f"Baseline {baseline['run_id']} · {target}" for target in targets]
    header += [f"Δ {target}" for target in targets]
    lines.append("| " + " | ".join(_markdown_table_text(cell) for cell in header) + " |")
    lines.append("|---" * len(header) + "|")
    for row in comparison["rows"]:
        cells = [_markdown_table_text(row["display_name"])]
        cells += [_run_cell_text(row["candidate"][target]) for target in targets]
        cells += [_run_cell_text(row["baseline"][target]) for target in targets]
        cells += [
            _run_delta_text(row["deltas"][target], row["delta_states"][target])
            for target in targets
        ]
        lines.append("| " + " | ".join(cells) + " |")

    caveats = _run_comparison_caveats(comparison)
    if caveats:
        lines += ["", "## Comparison caveats", ""]
        lines += [f"- {text}" for text in caveats]

    for role, heading in (
        ("baseline", "Baseline methodology notes"),
        ("candidate", "Candidate methodology notes"),
    ):
        notes = comparison.get(f"{role}_methodology_notes") or []
        if notes:
            lines += ["", f"## {heading}", ""]
            lines += [f"- {note}" for note in notes]
    lines.append("")
    return "\n".join(lines)
