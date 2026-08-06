"""Scoreboard aggregation: pair results -> scoreboard.json + markdown.

Rows follow the selected suite's canonical order and columns are targets. Cells
carry footnote markers for annotations and partial/skip reasons so a degraded
run is still an honest artifact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist, mean, stdev

from kairyu.bench.adapters import all_adapters, suite_info
from kairyu.bench.adapters.base import GenerativeAdapter, normalize_base_url
from kairyu.bench.sampling import (
    PROTOCOL_ID as SAMPLING_PROTOCOL_ID,
)
from kairyu.bench.sampling import (
    pass_at_k,
    reported_k_values,
    sampling_metrics,
    sampling_summary,
)
from kairyu.bench.types import (
    SCHEMA_VERSION,
    BenchTarget,
    JudgeConfig,
    PairResult,
    pair_result_evidence_error,
)

_WILSON_CONFIDENCE = 0.95
_WILSON_Z = NormalDist().inv_cdf(0.5 + _WILSON_CONFIDENCE / 2)


def _scoreboard_safe_pair(
    pair: PairResult,
    *,
    expected_sampling: dict[str, object] | None = None,
) -> PairResult:
    """Turn malformed pair metrics into a visible failed cell.

    Reports can be rebuilt directly from legacy or manually repaired pair
    artifacts, bypassing ``SuiteRunner``.  Keep aggregation total and preserve
    the cell identity/policy while withholding evidence that cannot be scored.
    """
    error = pair_result_evidence_error(pair, expected_sampling=expected_sampling)
    if error is None:
        return pair
    return pair.model_copy(
        update={
            "status": "failed",
            "reason": f"invalid pair evidence: {error}",
            "metrics": {"score": None, "n_total": 0},
            "items": (),
        }
    )


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


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


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
        * math.sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _binary_confidence_interval(pair: PairResult, *, declared_binary: bool) -> dict | None:
    """Build a Wilson interval only when item evidence proves binomial data."""
    if not declared_binary or pair.status != "completed":
        return None
    if any(item.sampling_attempts is not None for item in pair.items):
        # Multiple seeds of one dataset item are correlated repeated measures,
        # not extra Bernoulli items for an ordinary Wilson interval.
        return None
    score = pair.score
    if score is None or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    scored = [
        item.score for item in pair.items if item.status == "completed" and item.score is not None
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


def _sampling_sensitivity(pair: PairResult, *, declared_binary: bool) -> dict | None:
    """Recompute and bind a complete sampling summary from child evidence."""

    sampling = pair.methodology.get("sampling")
    if pair.status != "completed" or not isinstance(sampling, dict):
        return None
    attempts = sampling.get("attempts")
    seeds = sampling.get("seeds")
    mode = sampling.get("mode")
    wire_omitted = sampling.get("wire_omitted")
    expected_omissions = (
        ["temperature", "top_p", "top_k", "min_p", "repetition_penalty"]
        if mode == "recommended"
        else []
    )
    if (
        type(attempts) is not int
        or attempts < 2
        or not isinstance(seeds, list)
        or len(seeds) != attempts
        or any(type(seed) is not int for seed in seeds)
        or mode not in {"adapter", "recommended"}
        or wire_omitted != expected_omissions
        or sampling.get("recommended_defaults_attested") is not False
    ):
        return None
    try:
        summary = sampling_summary(
            pair.items,
            seeds=seeds,
            declared_binary=declared_binary,
        )
    except ValueError:
        return None
    expected = sampling_metrics(summary)
    n_items = int(summary["n_items"])
    if (
        pair.metrics.get("sampling_complete") != 1
        or pair.metrics.get("n_total") != n_items
        or pair.metrics.get("n_scored") != n_items
        or pair.metrics.get("sampling_base_items") != n_items
    ):
        return None
    for name, value in expected.items():
        stored = _finite_number(pair.metrics.get(name))
        if stored is None or not math.isclose(
            stored,
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return None
    score = pair.score
    if score is None or not math.isclose(
        score,
        float(summary["mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return None
    return {
        **summary,
        "mode": mode,
        "wire_omitted": list(wire_omitted),
        "recommended_defaults_attested": False,
    }


def build_scoreboard(
    *,
    run_id: str,
    suite: str,
    config: dict,
    environment: dict,
    fingerprint: str | None = None,
    pairs: list[PairResult],
    targets: list[str],
    target_configs: Sequence[BenchTarget] | None = None,
    judge: JudgeConfig | None = None,
    judge_identity_incomplete: bool = False,
) -> dict:
    definition = suite_info(suite)
    adapters = all_adapters()
    display_names = {name: adapter.info.display_name for name, adapter in adapters.items()}
    configured_by_label = {target.label(): target for target in (target_configs or ())}
    attempts = config.get("attempts", 1)
    grouped_chat_attempts = (
        attempts if config.get("sampling_protocol") == SAMPLING_PROTOCOL_ID else 1
    )
    pairs = [
        _scoreboard_safe_pair(
            pair,
            expected_sampling=(
                {
                    "attempts": (
                        grouped_chat_attempts
                        if isinstance(adapters.get(pair.benchmark), GenerativeAdapter)
                        else attempts
                    ),
                    "seed": configured_by_label[pair.target].seed,
                    "mode": configured_by_label[pair.target].sampling_mode,
                    "temperature": configured_by_label[pair.target].temperature,
                    "top_p": configured_by_label[pair.target].top_p,
                    "required": isinstance(
                        adapters.get(pair.benchmark), GenerativeAdapter
                    ),
                }
                if pair.target in configured_by_label
                else None
            ),
        )
        for pair in pairs
    ]
    by_key = {(pair.benchmark, pair.target): pair for pair in pairs}
    benchmarks = [
        name for name in definition.row_order if any(pair.benchmark == name for pair in pairs)
    ]

    footnotes: list[str] = []

    def footnote(text: str) -> int:
        if text not in footnotes:
            footnotes.append(text)
        return footnotes.index(text) + 1

    # Self-judging is an endpoint/model identity question, never a display-label
    # comparison. Legacy artifacts without enough identity data fail closed as
    # "independence unknown" instead of being declared independent.
    judge_requested = judge_identity_incomplete or (
        judge is not None and (judge.base_url is not None or judge.model is not None)
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
            benchmark in adapters and adapters[benchmark].info.judge_template_name is not None
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
                    "comparable": False,
                    "incomparable_reasons": ["pair was not run"],
                    "cross_run_policy": "allowed",
                    "cross_run_reason": None,
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
            declared_binary = (
                benchmark in adapters and adapters[benchmark].info.binary_outcomes
            )
            sensitivity = _sampling_sensitivity(
                pair,
                declared_binary=declared_binary,
            )
            sampling_policy = pair.methodology.get("sampling")
            if isinstance(sampling_policy, dict):
                if sampling_policy.get("mode") == "recommended":
                    notes.append(
                        footnote(
                            f"{benchmark}/{target}: recommended sampling requested; "
                            "generation-default fields were omitted from the request, "
                            "but endpoint defaults were not remotely attested"
                        )
                    )
                attempts = sampling_policy.get("attempts")
                if type(attempts) is int and attempts > 1 and sensitivity is None:
                    notes.append(
                        footnote(
                            f"{benchmark}/{target}: sampling sensitivity withheld "
                            "because the complete item-by-seed evidence did not validate"
                        )
                    )
            cell = {
                "status": pair.status,
                "score": pair.score,
                "n": pair.metrics.get("n_total"),
                "n_scored": pair.metrics.get("n_scored"),
                "confidence_interval": _binary_confidence_interval(
                    pair,
                    declared_binary=declared_binary,
                ),
                "reason": pair.reason,
                # structured comparability travels with the cell so the accuracy
                # report never has to infer it from a benchmark-name allow list
                "comparable": pair.comparable,
                "incomparable_reasons": list(pair.incomparable_reasons),
                "cross_run_policy": pair.cross_run_policy,
                "cross_run_reason": pair.cross_run_reason,
                "footnotes": notes,
            }
            if sensitivity is not None:
                cell["sampling_sensitivity"] = sensitivity
            cells[benchmark][target] = cell

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "suite": suite,
        "fingerprint": fingerprint,
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
    confidence_number = _finite_number(confidence)
    lower_number = _finite_number(lower)
    upper_number = _finite_number(upper)
    if confidence_number is None or lower_number is None or upper_number is None:
        return None
    if not math.isclose(
        confidence_number,
        _WILSON_CONFIDENCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return None
    trials = _whole_count(value.get("trials"))
    successes = _whole_count(value.get("successes"))
    expected_score_number = _finite_number(expected_score)
    if (
        trials is None
        or trials == 0
        or trials != expected_trials
        or successes is None
        or expected_score_number is None
        or not math.isclose(
            expected_score_number,
            successes / trials,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return None
    calculated = _wilson_bounds(successes, trials)
    if calculated is None or not all(
        math.isclose(stored, expected, rel_tol=0.0, abs_tol=1e-12)
        for stored, expected in zip((lower_number, upper_number), calculated, strict=True)
    ):
        return None
    return lower_number, upper_number


def _binary_margins_feasible(
    item_successes: list[int], seed_successes: list[int]
) -> bool:
    """Return whether the row/column sums describe some binary outcome matrix."""

    if sum(item_successes) != sum(seed_successes):
        return False
    rows = sorted(item_successes, reverse=True)
    return all(
        sum(rows[:count]) <= sum(min(count, degree) for degree in seed_successes)
        for count in range(1, len(rows) + 1)
    )


def _stored_sampling_sensitivity(
    value: object,
    *,
    expected_score: object,
    expected_total: object,
    expected_scored: object,
    expected_status: object,
    declared_binary: bool,
) -> dict | None:
    """Validate a stored scoreboard summary before rendering its labels."""

    if not isinstance(value, dict):
        return None
    attempts = _whole_count(value.get("attempts"))
    item_count = _whole_count(value.get("n_items"))
    seeds = value.get("seeds")
    seed_scores = value.get("seed_scores")
    expected_score_number = _finite_number(expected_score)
    total = _whole_count(expected_total)
    scored = _whole_count(expected_scored)
    if (
        attempts is None
        or attempts < 2
        or item_count is None
        or item_count < 1
        or not isinstance(seeds, list)
        or len(seeds) != attempts
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != attempts
        or not isinstance(seed_scores, list)
        or len(seed_scores) != attempts
        or expected_score_number is None
        or expected_status != "completed"
        or total != item_count
        or scored != item_count
    ):
        return None
    scores: list[float] = []
    for seed, entry in zip(seeds, seed_scores, strict=True):
        if not isinstance(entry, dict) or entry.get("seed") != seed:
            return None
        score = _finite_number(entry.get("score"))
        if score is None or not 0.0 <= score <= 1.0:
            return None
        scores.append(score)
    stored_mean = _finite_number(value.get("mean"))
    sample_sd = _finite_number(value.get("sample_sd"))
    minimum = _finite_number(value.get("min"))
    maximum = _finite_number(value.get("max"))
    if (
        stored_mean is None
        or sample_sd is None
        or sample_sd < 0.0
        or minimum is None
        or maximum is None
        or not 0.0 <= minimum <= stored_mean <= maximum <= 1.0
        or not math.isclose(stored_mean, mean(scores), rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(sample_sd, stdev(scores), rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(minimum, min(scores), rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(maximum, max(scores), rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(
            expected_score_number,
            stored_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return None
    passes = value.get("pass_at_k")
    item_successes = value.get("item_successes")
    if not declared_binary:
        if passes is not None or item_successes is not None:
            return None
    else:
        if (
            not isinstance(passes, list)
            or not isinstance(item_successes, list)
            or len(item_successes) != item_count
            or any(
                type(successes) is not int or not 0 <= successes <= attempts
                for successes in item_successes
            )
        ):
            return None
        expected_ks = reported_k_values(attempts)
        if len(passes) != len(expected_ks):
            return None
        for expected_k, entry in zip(expected_ks, passes, strict=True):
            if not isinstance(entry, dict):
                return None
            k = _whole_count(entry.get("k"))
            estimate = _finite_number(entry.get("estimate"))
            expected_estimate = mean(
                pass_at_k(successes, attempts, expected_k)
                for successes in item_successes
            )
            if (
                k != expected_k
                or estimate is None
                or not 0.0 <= estimate <= 1.0
                or not math.isclose(
                    estimate,
                    expected_estimate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                return None
        seed_successes: list[int] = []
        for score in scores:
            successes = round(score * item_count)
            if not math.isclose(
                score * item_count,
                successes,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return None
            seed_successes.append(successes)
        if not _binary_margins_feasible(item_successes, seed_successes):
            return None
    mode = value.get("mode")
    omissions = value.get("wire_omitted")
    expected_omissions = (
        ["temperature", "top_p", "top_k", "min_p", "repetition_penalty"]
        if mode == "recommended"
        else []
    )
    if (
        mode not in {"adapter", "recommended"}
        or omissions != expected_omissions
        or value.get("recommended_defaults_attested") is not False
    ):
        return None
    return value


def _cell_text(cell: dict, *, declared_binary: bool) -> str:
    marks = "".join(f"[^{n}]" for n in cell["footnotes"])
    if cell["status"] == "skipped":
        return f"—{marks}"
    if cell["score"] is None:
        return f"n/a{marks}"
    sensitivity = _stored_sampling_sensitivity(
        cell.get("sampling_sensitivity"),
        expected_score=cell.get("score"),
        expected_total=cell.get("n"),
        expected_scored=cell.get("n_scored"),
        expected_status=cell.get("status"),
        declared_binary=declared_binary,
    )
    if sensitivity is not None:
        text = (
            f"{cell['score'] * 100:.1f} ± "
            f"{sensitivity['sample_sd'] * 100:.1f} SD "
            f"[{sensitivity['min'] * 100:.1f}–{sensitivity['max'] * 100:.1f}]"
        )
        passes = sensitivity.get("pass_at_k")
        if isinstance(passes, list):
            text += "; " + ", ".join(
                f"pass@{entry['k']} {entry['estimate'] * 100:.1f}"
                for entry in passes
            )
    else:
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
    return (
        ["> **This run is not a full-suite measurement.**", ">"]
        + [f"> - {reason}" for reason in shared]
        + [""]
    )


def render_markdown(scoreboard: dict) -> str:
    targets = scoreboard["targets"]
    definition = suite_info(scoreboard.get("suite", "fugu"))
    adapters = all_adapters()
    has_sampling = any(
        _stored_sampling_sensitivity(
            cell.get("sampling_sensitivity"),
            expected_score=cell.get("score"),
            expected_total=cell.get("n"),
            expected_scored=cell.get("n_scored"),
            expected_status=cell.get("status"),
            declared_binary=(
                benchmark in adapters and adapters[benchmark].info.binary_outcomes
            ),
        )
        is not None
        for benchmark, by_target in scoreboard.get("cells", {}).items()
        for cell in by_target.values()
    )
    legend = (
        " Sampling sweeps show mean ± sample SD [min–max] over seeds; pass@k "
        "uses the per-item unbiased estimator."
        if has_sampling
        else ""
    )
    lines = [
        f"# {definition.display_name} benchmark scoreboard — run {scoreboard['run_id']}",
        "",
        *run_banner(scoreboard),
        "Scores are percentages; brackets are 95% Wilson CIs for binary item "
        "outcomes; n is scored/total when they differ; — = skipped, "
        f"* = partial/failed (see footnotes).{legend}",
        "",
        "| Benchmark | " + " | ".join(targets) + " |",
        "|---" * (len(targets) + 1) + "|",
    ]
    for benchmark in scoreboard["benchmarks"]:
        display = scoreboard["display_names"].get(benchmark, benchmark)
        declared_binary = (
            benchmark in adapters and adapters[benchmark].info.binary_outcomes
        )
        row = [display] + [
            _cell_text(
                scoreboard["cells"][benchmark][target],
                declared_binary=declared_binary,
            )
            for target in targets
        ]
        lines.append("| " + " | ".join(row) + " |")
    if scoreboard["footnotes"]:
        lines.append("")
        for index, note in enumerate(scoreboard["footnotes"], start=1):
            lines.append(f"[^{index}]: {note}")
    lines.append("")
    return "\n".join(lines)
