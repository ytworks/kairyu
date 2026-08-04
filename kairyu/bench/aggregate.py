"""Scoreboard aggregation: pair results -> scoreboard.json + Fugu-layout markdown.

Layout mirrors the Fugu release table: rows = benchmarks in FUGU_ROW_ORDER,
columns = targets. Cells carry footnote markers for annotations and
partial/skip reasons so a degraded run is still an honest artifact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

from kairyu.bench.adapters import FUGU_ROW_ORDER, all_adapters
from kairyu.bench.adapters.base import normalize_base_url
from kairyu.bench.types import (
    SCHEMA_VERSION,
    BenchTarget,
    JudgeConfig,
    PairResult,
)

_WILSON_CONFIDENCE = 0.95
_WILSON_Z = NormalDist().inv_cdf(0.5 + _WILSON_CONFIDENCE / 2)


def _resolved_identity(base_url: str, model: str) -> tuple[str, str]:
    return normalize_base_url(base_url), model


def _whole_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if not isinstance(value, float):
        return None
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        return None
    return int(value)


def _wilson_bounds(successes: int, trials: int) -> tuple[float, float] | None:
    """Return the two-sided 95% Wilson score interval for binary trials."""
    if trials <= 0 or successes < 0 or successes > trials:
        return None
    proportion = successes / trials
    z_squared = _WILSON_Z**2
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        _WILSON_Z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _binary_confidence_interval(
    pair: PairResult, *, declared_binary: bool
) -> dict | None:
    """Build a Wilson interval only when item evidence proves binomial data."""
    if not declared_binary or pair.status != "completed":
        return None
    score = pair.score
    if score is None or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    scored = [
        item.score
        for item in pair.items
        if item.status == "completed" and item.score is not None
    ]
    if not scored or any(value not in (0.0, 1.0) for value in scored):
        return None
    trials = _whole_count(pair.metrics.get("n_scored"))
    total = _whole_count(pair.metrics.get("n_total"))
    if trials is None or trials == 0 or trials != len(scored):
        return None
    if total is None or total != len(pair.items) or trials != total:
        return None
    successes = sum(value == 1.0 for value in scored)
    if not math.isclose(score, successes / trials, rel_tol=0.0, abs_tol=1e-12):
        return None
    bounds = _wilson_bounds(successes, trials)
    assert bounds is not None
    return {
        "method": "wilson",
        "confidence": _WILSON_CONFIDENCE,
        "successes": successes,
        "trials": trials,
        "lower": bounds[0],
        "upper": bounds[1],
    }


def build_scoreboard(
    *,
    run_id: str,
    suite: str,
    config: dict,
    environment: dict,
    pairs: list[PairResult],
    targets: list[str],
    target_configs: Sequence[BenchTarget] | None = None,
    judge: JudgeConfig | None = None,
    judge_identity_incomplete: bool = False,
) -> dict:
    adapters = all_adapters()
    display_names = {
        name: adapter.info.display_name for name, adapter in adapters.items()
    }
    by_key = {(pair.benchmark, pair.target): pair for pair in pairs}
    benchmarks = [name for name in FUGU_ROW_ORDER if any(p.benchmark == name for p in pairs)]

    footnotes: list[str] = []

    def footnote(text: str) -> int:
        if text not in footnotes:
            footnotes.append(text)
        return footnotes.index(text) + 1

    # Self-judging is an endpoint/model identity question, never a display-label
    # comparison. Legacy artifacts without enough identity data fail closed as
    # "independence unknown" instead of being declared independent.
    configured_by_label = {
        target.label(): target for target in (target_configs or ())
    }
    judge_requested = judge_identity_incomplete or (
        judge is not None
        and (judge.base_url is not None or judge.model is not None)
    )
    judge_identities = (
        [
            _resolved_identity(endpoint.base_url, endpoint.model)
            for endpoint in judge.grading_endpoints()
            if endpoint.base_url is not None and endpoint.model is not None
        ]
        if judge is not None and judge.enabled
        else []
    )
    self_judged: list[str] = []
    identity_unknown: list[str] = []
    if judge_requested:
        for label in targets:
            target = configured_by_label.get(label)
            if target is not None and (
                _resolved_identity(target.base_url, target.model) in judge_identities
            ):
                self_judged.append(label)
            if judge_identity_incomplete or not judge_identities or target is None:
                identity_unknown.append(label)

    cells: dict[str, dict[str, dict]] = {}
    for benchmark in benchmarks:
        uses_judge_template = (
            benchmark in adapters
            and adapters[benchmark].info.judge_template_name is not None
        )
        cells[benchmark] = {}
        for target in targets:
            pair = by_key.get((benchmark, target))
            if pair is None:
                cells[benchmark][target] = {
                    "status": "skipped",
                    "score": None,
                    "n": 0,
                    "confidence_interval": None,
                    "reason": "not run",
                    "footnotes": [footnote(f"{benchmark}/{target}: not run")],
                }
                continue
            notes = [footnote(f"{benchmark}: {text}") for text in pair.annotations]
            if pair.status in ("skipped", "partial", "failed") and pair.reason:
                notes.append(footnote(f"{benchmark}/{target}: {pair.status} — {pair.reason}"))
            for reason in pair.incomparable_reasons:
                notes.append(footnote(f"{benchmark}/{target}: NOT COMPARABLE — {reason}"))
            if uses_judge_template and target in self_judged:
                notes.append(
                    footnote(
                        f"{target}: self-judged "
                        "(one or more resolved judge endpoint/model identities "
                        "== target)"
                    )
                )
            if uses_judge_template and target in identity_unknown:
                notes.append(
                    footnote(
                        f"{target}: judge independence unknown "
                        "(resolved target or judge identity unavailable)"
                    )
                )
            cells[benchmark][target] = {
                "status": pair.status,
                "score": pair.score,
                "n": pair.metrics.get("n_total"),
                "n_scored": pair.metrics.get("n_scored"),
                "confidence_interval": _binary_confidence_interval(
                    pair,
                    declared_binary=(
                        benchmark in adapters
                        and adapters[benchmark].info.binary_outcomes
                    ),
                ),
                "reason": pair.reason,
                # structured comparability travels with the cell so the accuracy
                # report never has to infer it from a benchmark-name allow list
                "comparable": pair.comparable,
                "incomparable_reasons": list(pair.incomparable_reasons),
                "footnotes": notes,
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "suite": suite,
        "environment": environment,
        "config": config,
        "benchmarks": benchmarks,
        "display_names": {name: display_names.get(name, name) for name in benchmarks},
        "targets": targets,
        "self_judged_targets": self_judged,
        "judge_independence_unknown_targets": identity_unknown,
        "cells": cells,
        "footnotes": footnotes,
    }


def _stored_wilson_bounds(
    value: object, *, expected_trials: int | None, expected_score: object
) -> tuple[float, float] | None:
    """Validate stored interval metadata before labelling it Wilson 95%."""
    if not isinstance(value, dict) or value.get("method") != "wilson":
        return None
    confidence = value.get("confidence")
    lower = value.get("lower")
    upper = value.get("upper")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or isinstance(lower, bool)
        or not isinstance(lower, int | float)
        or isinstance(upper, bool)
        or not isinstance(upper, int | float)
        or not all(math.isfinite(number) for number in (confidence, lower, upper))
        or not math.isclose(
            float(confidence), _WILSON_CONFIDENCE, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        return None
    trials = _whole_count(value.get("trials"))
    successes = _whole_count(value.get("successes"))
    if (
        trials is None
        or trials == 0
        or trials != expected_trials
        or successes is None
        or isinstance(expected_score, bool)
        or not isinstance(expected_score, int | float)
        or not math.isfinite(expected_score)
        or not math.isclose(
            float(expected_score),
            successes / trials,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return None
    calculated = _wilson_bounds(successes, trials)
    if calculated is None or not all(
        math.isclose(stored, expected, rel_tol=0.0, abs_tol=1e-12)
        for stored, expected in zip((float(lower), float(upper)), calculated, strict=True)
    ):
        return None
    return float(lower), float(upper)


def _cell_text(cell: dict) -> str:
    marks = "".join(f"[^{n}]" for n in cell["footnotes"])
    if cell["status"] == "skipped":
        return f"—{marks}"
    if cell["score"] is None:
        return f"n/a{marks}"
    text = f"{cell['score'] * 100:.1f}"
    if cell["status"] in ("partial", "failed"):
        text += "*"
    total = _whole_count(cell.get("n"))
    scored = _whole_count(cell.get("n_scored"))
    interval = (
        _stored_wilson_bounds(
            cell.get("confidence_interval"),
            expected_trials=scored if scored is not None else total,
            expected_score=cell.get("score"),
        )
        if cell["status"] == "completed"
        else None
    )
    if interval is not None:
        text += f" [{interval[0] * 100:.1f}–{interval[1] * 100:.1f}]"
    if total is not None and scored is not None and scored != total:
        text += f" (n={scored}/{total})"
    elif total is not None:
        text += f" (n={total})"
    elif scored is not None:
        text += f" (n={scored})"
    return f"{text}{marks}"


def run_banner(scoreboard: dict) -> list[str]:
    """Loud, artifact-resident notice when no cell is a full-suite measurement.

    A shell warning does not survive into the file an operator opens hours later.
    """
    cells = scoreboard.get("cells") or {}
    shared: list[str] | None = None
    for by_target in cells.values():
        for cell in by_target.values():
            reasons = list(cell.get("incomparable_reasons") or [])
            shared = reasons if shared is None else [r for r in shared if r in reasons]
    if not shared:
        return []
    return ["> **This run is not a full-suite measurement.**", ">"] + [
        f"> - {reason}" for reason in shared
    ] + [""]


def render_markdown(scoreboard: dict) -> str:
    targets = scoreboard["targets"]
    lines = [
        f"# Fugu benchmark scoreboard — run {scoreboard['run_id']}",
        "",
        *run_banner(scoreboard),
        "Scores are percentages; brackets are 95% Wilson CIs for binary item "
        "outcomes; n is scored/total when they differ; — = skipped, "
        "* = partial/failed (see footnotes).",
        "",
        "| Benchmark | " + " | ".join(targets) + " |",
        "|---" * (len(targets) + 1) + "|",
    ]
    for benchmark in scoreboard["benchmarks"]:
        display = scoreboard["display_names"].get(benchmark, benchmark)
        row = [display] + [
            _cell_text(scoreboard["cells"][benchmark][target]) for target in targets
        ]
        lines.append("| " + " | ".join(row) + " |")
    if scoreboard["footnotes"]:
        lines.append("")
        for index, note in enumerate(scoreboard["footnotes"], start=1):
            lines.append(f"[^{index}]: {note}")
    lines.append("")
    return "\n".join(lines)
