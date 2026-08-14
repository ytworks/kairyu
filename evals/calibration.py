"""Published-gold judge calibration and prompt-position bias evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.adapters.base import utc_now
from evals.judge import (
    JudgeVerdict,
    build_judge_client,
    judge_protocol_identity,
)
from evals.judge_prompts import judge_template_identity
from evals.targets import normalize_base_url
from evals.types import (
    BenchConfig,
    BenchTarget,
    JudgeConfig,
    JudgeEndpointConfig,
)
from evidence.reporting import atomic_write_text

# This is part of the calibration fingerprint. Bump it whenever metric or gate
# semantics change, even if the JSON remains structurally readable.
CALIBRATION_SCHEMA_VERSION = 1
_DEFAULT_SET = "judge-calibration.jsonl"
_TEMPLATES = ("hle", "charxiv-reasoning")
STANDARD_MIN_AGREEMENT = 11 / 12
STANDARD_MIN_COVERAGE = 1.0
STANDARD_MAX_POSITION_FLIP = 0.0
STANDARD_MAX_SELF_PREFERENCE_GAP = 0.2
HEADLINE_MIN_RESPONSES_PER_TEMPLATE = 12
HEADLINE_MIN_PROMPT_PAIRS_PER_TEMPLATE = 6
SELF_PREFERENCE_MIN_STRATUM = 2
_ATTESTED_LABEL_PROVENANCE = frozenset({"published-gold", "human-reviewed"})


class CalibrationCase(BaseModel):
    """One response plus an explicit correctness-label provenance claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    template: Literal["hle", "charxiv-reasoning"]
    question: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    response: str = Field(min_length=1)
    human_correct: bool
    label_provenance: Literal[
        "published-gold", "human-reviewed", "unverified"
    ] = "unverified"
    label_source: str | None = None
    label_source_revision: str | None = None
    label_source_record: str | None = None
    label_source_license: str | None = None
    # Optional real provenance for measuring self-preference. An unverified
    # built-in row deliberately makes no producer claim.
    response_base_url: str | None = None
    response_model: str | None = None

    @model_validator(mode="after")
    def _validate_provenance(self) -> CalibrationCase:
        label_fields = (
            "label_source",
            "label_source_revision",
            "label_source_record",
            "label_source_license",
        )
        for field_name in label_fields:
            value = getattr(self, field_name)
            if value is not None:
                stripped = value.strip()
                if not stripped:
                    raise ValueError(f"{field_name} must not be empty")
                object.__setattr__(self, field_name, stripped)
        if self.label_provenance in _ATTESTED_LABEL_PROVENANCE and any(
            getattr(self, field_name) is None for field_name in label_fields
        ):
            raise ValueError(
                "attested label provenance requires label_source, "
                "label_source_revision, label_source_record, and "
                "label_source_license"
            )
        present = (self.response_base_url is not None, self.response_model is not None)
        if present[0] != present[1]:
            raise ValueError(
                "response_base_url and response_model must be provided together"
            )
        if self.response_base_url is not None:
            assert self.response_model is not None
            if (
                not self.response_model.strip()
                or self.response_model != self.response_model.strip()
            ):
                raise ValueError(
                    "response_model must be non-empty and have no surrounding "
                    "whitespace"
                )
            object.__setattr__(
                self, "response_base_url", normalize_base_url(self.response_base_url)
            )
        return self

    @property
    def label_attested(self) -> bool:
        return self.label_provenance in _ATTESTED_LABEL_PROVENANCE

    def response_identity(self) -> tuple[str, str] | None:
        if self.response_base_url is None or self.response_model is None:
            return None
        return self.response_base_url, self.response_model


def _default_bytes() -> bytes:
    return (
        resources.files("evals.fixtures")
        .joinpath(_DEFAULT_SET)
        .read_bytes()
    )


def load_calibration_cases(
    path: str | Path | None = None,
) -> tuple[tuple[CalibrationCase, ...], bytes, str]:
    """Load and validate the packaged or caller-supplied JSONL set."""
    if path is None:
        raw = _default_bytes()
        source = f"package:evals.fixtures/{_DEFAULT_SET}"
    else:
        selected = Path(path)
        raw = selected.read_bytes()
        source = str(selected)
    cases: list[CalibrationCase] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(CalibrationCase.model_validate_json(line))
        except ValueError as error:
            raise ValueError(
                f"invalid judge calibration case at line {line_number}: {error}"
            ) from error
    if not cases:
        raise ValueError("judge calibration set is empty")
    ids = [case.id for case in cases]
    duplicates = sorted(name for name, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(
            "judge calibration case ids must be unique; duplicates: "
            + ", ".join(duplicates)
        )
    return tuple(cases), raw, source


def _set_summary(cases: tuple[CalibrationCase, ...]) -> dict:
    by_template: dict[str, dict] = {}
    balanced = True
    paired = True
    minimum_shape = True
    labels_attested = all(case.label_attested for case in cases)
    for template_name in _TEMPLATES:
        selected = [case for case in cases if case.template == template_name]
        positives = sum(case.human_correct for case in selected)
        negatives = len(selected) - positives
        template_balanced = bool(selected) and positives == negatives
        balanced = balanced and template_balanced
        prompt_groups: dict[tuple[str, str], list[bool]] = {}
        for case in selected:
            prompt_groups.setdefault((case.question, case.expected), []).append(
                case.human_correct
            )
        template_paired = bool(prompt_groups) and all(
            sorted(labels) == [False, True] for labels in prompt_groups.values()
        )
        paired = paired and template_paired
        template_minimum_shape = (
            len(selected) >= HEADLINE_MIN_RESPONSES_PER_TEMPLATE
            and len(prompt_groups) >= HEADLINE_MIN_PROMPT_PAIRS_PER_TEMPLATE
        )
        minimum_shape = minimum_shape and template_minimum_shape
        template_labels_attested = all(case.label_attested for case in selected)
        by_template[template_name] = {
            "responses": len(selected),
            "prompt_pairs": len(prompt_groups),
            "human_correct": positives,
            "human_incorrect": negatives,
            "paired": template_paired,
            "balanced": template_balanced,
            "minimum_headline_shape": template_minimum_shape,
            "labels_attested": template_labels_attested,
            "headline_shape": (
                template_paired
                and template_balanced
                and template_minimum_shape
                and template_labels_attested
            ),
        }
    prompt_groups: dict[tuple[str, str, str], list[bool]] = {}
    for case in cases:
        prompt_groups.setdefault(
            (case.template, case.question, case.expected), []
        ).append(case.human_correct)
    headline_shape = paired and balanced and minimum_shape and labels_attested
    return {
        "responses": len(cases),
        "prompt_pairs": len(prompt_groups),
        "paired": paired,
        "balanced": balanced,
        "minimum_headline_shape": minimum_shape,
        "labels_attested": labels_attested,
        "label_provenance": dict(
            sorted(Counter(case.label_provenance for case in cases).items())
        ),
        "headline_shape": headline_shape,
        "by_template": by_template,
    }


def _agreement(
    cases: tuple[CalibrationCase, ...], predictions: tuple[bool | None, ...]
) -> dict:
    if len(cases) != len(predictions):
        raise ValueError("calibration predictions do not match the case count")
    matrix = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    matches = 0
    parseable = 0
    for case, predicted in zip(cases, predictions, strict=True):
        if predicted is None:
            continue
        parseable += 1
        matches += predicted == case.human_correct
        if predicted and case.human_correct:
            matrix["true_positive"] += 1
        elif not predicted and not case.human_correct:
            matrix["true_negative"] += 1
        elif predicted:
            matrix["false_positive"] += 1
        else:
            matrix["false_negative"] += 1
    total = len(cases)
    return {
        "total": total,
        "parseable": parseable,
        "unparseable": total - parseable,
        "matches": matches,
        # Unparseable verdicts stay in the denominator: calibration fails closed.
        "agreement": matches / total if total else None,
        "parseable_agreement": matches / parseable if parseable else None,
        "coverage": parseable / total if total else None,
        "confusion_matrix": matrix,
    }


def _agreement_with_templates(
    cases: tuple[CalibrationCase, ...], predictions: tuple[bool | None, ...]
) -> dict:
    result = _agreement(cases, predictions)
    by_template = {}
    for template_name in _TEMPLATES:
        indexes = [
            index for index, case in enumerate(cases) if case.template == template_name
        ]
        selected_cases = tuple(cases[index] for index in indexes)
        selected_predictions = tuple(predictions[index] for index in indexes)
        by_template[template_name] = _agreement(
            selected_cases, selected_predictions
        )
    return {**result, "by_template": by_template}


def _position_bias(
    cases: tuple[CalibrationCase, ...],
    production: tuple[bool | None, ...],
    counterbalanced: tuple[bool | None, ...],
) -> dict:
    both_parseable = 0
    flips = 0
    for first, second in zip(production, counterbalanced, strict=True):
        if first is None or second is None:
            continue
        both_parseable += 1
        flips += first != second
    production_agreement = _agreement(cases, production)["agreement"]
    counterbalanced_agreement = _agreement(cases, counterbalanced)["agreement"]
    gap = None
    if production_agreement is not None and counterbalanced_agreement is not None:
        gap = counterbalanced_agreement - production_agreement
    return {
        "both_parseable": both_parseable,
        "flips": flips,
        "flip_rate": flips / both_parseable if both_parseable else None,
        "production_agreement": production_agreement,
        "counterbalanced_agreement": counterbalanced_agreement,
        "agreement_gap": gap,
    }


def _rate_for_label(
    cases: list[CalibrationCase], predictions: list[bool | None], *, label: bool
) -> float | None:
    selected = [
        prediction
        for case, prediction in zip(cases, predictions, strict=True)
        if case.human_correct is label
    ]
    if not selected or any(prediction is None for prediction in selected):
        return None
    return sum(prediction is True for prediction in selected) / len(selected)


def _self_preference(
    endpoint: JudgeEndpointConfig,
    cases: tuple[CalibrationCase, ...],
    production: tuple[bool | None, ...],
    counterbalanced: tuple[bool | None, ...],
    *,
    matches_target: bool | None,
    force_measurement: bool,
) -> dict:
    identity = endpoint.resolved_identity()
    if identity is None:
        return {"status": "unavailable", "reason": "judge identity unavailable"}
    if matches_target is False and not force_measurement:
        return {
            "status": "not_applicable",
            "reason": "judge identity does not match any bound target identity",
            "required_for_headline": False,
        }
    self_cases: list[CalibrationCase] = []
    self_indexes: list[int] = []
    nonself_cases: list[CalibrationCase] = []
    nonself_indexes: list[int] = []
    unknown = 0
    for index, case in enumerate(cases):
        source = case.response_identity()
        if source is None:
            unknown += 1
        elif source == identity:
            self_cases.append(case)
            self_indexes.append(index)
        else:
            nonself_cases.append(case)
            nonself_indexes.append(index)

    strata: dict[str, dict[str, dict[str, int]]] = {}
    missing_strata: list[str] = []
    for template_name in _TEMPLATES:
        strata[template_name] = {}
        for human_correct, label_name in (
            (True, "human_correct"),
            (False, "human_incorrect"),
        ):
            self_count = sum(
                case.template == template_name
                and case.human_correct is human_correct
                for case in self_cases
            )
            nonself_count = sum(
                case.template == template_name
                and case.human_correct is human_correct
                for case in nonself_cases
            )
            strata[template_name][label_name] = {
                "self": self_count,
                "nonself": nonself_count,
            }
            for source_name, count in (
                ("self", self_count),
                ("nonself", nonself_count),
            ):
                if count < SELF_PREFERENCE_MIN_STRATUM:
                    missing_strata.append(
                        f"{template_name}/{label_name}/{source_name}={count}"
                    )

    common = {
        "required_for_headline": matches_target is True or force_measurement,
        "n_self": len(self_cases),
        "n_nonself": len(nonself_cases),
        "n_unknown_provenance": unknown,
        "minimum_per_stratum": SELF_PREFERENCE_MIN_STRATUM,
        "strata": strata,
    }
    if unknown:
        return {
            "status": "unavailable",
            "reason": "every response requires producer provenance",
            **common,
        }
    if not all(case.label_attested for case in cases):
        return {
            "status": "unavailable",
            "reason": "self-preference requires attested correctness labels",
            **common,
        }
    if missing_strata:
        return {
            "status": "unavailable",
            "reason": (
                "requires at least two responses in every template/label/"
                "self-or-nonself stratum: " + ", ".join(missing_strata)
            ),
            **common,
        }

    def variant_metrics(predictions: tuple[bool | None, ...]) -> dict | None:
        self_predictions = [predictions[index] for index in self_indexes]
        nonself_predictions = [predictions[index] for index in nonself_indexes]
        self_tpr = _rate_for_label(self_cases, self_predictions, label=True)
        self_fpr = _rate_for_label(self_cases, self_predictions, label=False)
        nonself_tpr = _rate_for_label(
            nonself_cases, nonself_predictions, label=True
        )
        nonself_fpr = _rate_for_label(
            nonself_cases, nonself_predictions, label=False
        )
        values = (self_tpr, self_fpr, nonself_tpr, nonself_fpr)
        if any(value is None for value in values):
            return None
        assert all(value is not None for value in values)
        return {
            "self_true_positive_rate": self_tpr,
            "nonself_true_positive_rate": nonself_tpr,
            "true_positive_gap": self_tpr - nonself_tpr,
            "self_false_positive_rate": self_fpr,
            "nonself_false_positive_rate": nonself_fpr,
            "false_positive_gap": self_fpr - nonself_fpr,
            "max_self_favoring_gap": max(
                self_tpr - nonself_tpr, self_fpr - nonself_fpr
            ),
        }

    production_metrics = variant_metrics(production)
    counterbalanced_metrics = variant_metrics(counterbalanced)
    if production_metrics is None or counterbalanced_metrics is None:
        return {
            "status": "unavailable",
            "reason": "one or more provenance strata had an unparseable verdict",
            **common,
        }
    return {
        "status": "measured",
        **common,
        "production": production_metrics,
        "counterbalanced": counterbalanced_metrics,
        "max_self_favoring_gap": max(
            production_metrics["max_self_favoring_gap"],
            counterbalanced_metrics["max_self_favoring_gap"],
        ),
    }


def _target_identity(value: object) -> dict[str, str]:
    if isinstance(value, Mapping):
        base_url = value.get("base_url")
        model = value.get("model")
        label = value.get("label") or value.get("name") or model
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and len(value) == 2
    ):
        base_url, model = value
        label = model
    else:
        raise ValueError(
            "target identity must be {base_url, model} or a (base_url, model) pair"
        )
    if not isinstance(base_url, str) or not isinstance(model, str) or not model:
        raise ValueError("target identity requires non-empty base_url and model")
    if not isinstance(label, str) or not label:
        raise ValueError("target identity label must be a non-empty string")
    return {
        "label": label,
        "base_url": normalize_base_url(base_url),
        "model": model,
    }


def _verified_run_judge_adapters(
    adapter_identities: Sequence[Mapping[str, object]] | None,
) -> list[dict]:
    """Prove that a bound run used the templates/protocol calibrated now."""
    from evals.adapters import all_adapters

    registry = all_adapters()
    verified: list[dict] = []
    seen_adapters: set[str] = set()
    for raw_adapter in adapter_identities or ():
        if not isinstance(raw_adapter, Mapping):
            raise ValueError("bound run contains a malformed adapter identity")
        adapter_name = raw_adapter.get("name")
        if not isinstance(adapter_name, str) or not adapter_name:
            raise ValueError("bound run adapter identity lacks a name")
        if adapter_name in seen_adapters:
            raise ValueError(
                f"bound run contains duplicate adapter identity {adapter_name!r}"
            )
        seen_adapters.add(adapter_name)

        adapter = registry.get(adapter_name)
        expected_selector = (
            adapter.info.judge_template_name if adapter is not None else None
        )
        recorded_template = raw_adapter.get("judge_template")
        recorded_protocol = raw_adapter.get("judge_protocol")
        if expected_selector is None:
            if recorded_template is not None or recorded_protocol is not None:
                raise ValueError(
                    f"bound run has an unexpected judge identity for {adapter_name!r}"
                )
            continue
        if not isinstance(recorded_template, Mapping) or not isinstance(
            recorded_protocol, Mapping
        ):
            raise ValueError(
                f"bound run lacks judge template/protocol identity for {adapter_name!r}"
            )

        current_template = judge_template_identity(expected_selector)
        if dict(recorded_template) != current_template:
            raise ValueError(
                f"bound run judge template identity for {adapter_name!r} does not "
                "match the current calibration template"
            )
        current_protocol = judge_protocol_identity()
        if dict(recorded_protocol) != current_protocol:
            raise ValueError(
                f"bound run judge protocol identity for {adapter_name!r} does not "
                "match the current calibration protocol"
            )
        verified.append(
            {
                "adapter": adapter_name,
                "judge_template": current_template,
                "judge_protocol": current_protocol,
            }
        )
    return verified


def _run_binding(
    *,
    run_fingerprint: str | None,
    target_configs: Sequence[BenchTarget | Mapping[str, object]] | None,
    target_identities: Sequence[tuple[str, str] | Mapping[str, str]] | None,
    run_adapter_identities: Sequence[Mapping[str, object]] | None,
) -> dict:
    if run_fingerprint is not None and (
        len(run_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in run_fingerprint)
    ):
        raise ValueError("run_fingerprint must be a lowercase SHA-256 digest")

    validated_targets: list[BenchTarget] = []
    for raw_target in target_configs or ():
        validated_targets.append(
            raw_target
            if isinstance(raw_target, BenchTarget)
            else BenchTarget.model_validate(raw_target)
        )
    derived_identities = [
        {
            "label": target.label(),
            "base_url": target.base_url,
            "model": target.model,
        }
        for target in validated_targets
    ]
    declared_identities = [
        _target_identity(identity) for identity in (target_identities or ())
    ]
    if derived_identities and declared_identities:
        derived_pairs = [
            (identity["base_url"], identity["model"])
            for identity in derived_identities
        ]
        declared_pairs = [
            (identity["base_url"], identity["model"])
            for identity in declared_identities
        ]
        if derived_pairs != declared_pairs:
            raise ValueError(
                "target identities do not match the supplied target configs"
            )
    identities = derived_identities or declared_identities
    judge_adapters = _verified_run_judge_adapters(run_adapter_identities)
    bound = (
        run_fingerprint is not None
        and bool(validated_targets)
        and bool(identities)
        and bool(judge_adapters)
    )
    return {
        "bound": bound,
        "run_fingerprint": run_fingerprint,
        "target_identities": identities,
        "target_configs": [
            target.model_dump(mode="json") for target in validated_targets
        ],
        "judge_adapters": judge_adapters,
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _load_bound_run(
    run: str | Path,
    *,
    results_dir: str | Path,
) -> tuple[
    JudgeConfig,
    str,
    tuple[BenchTarget, ...],
    tuple[Mapping[str, object], ...],
]:
    """Load and integrity-check the immutable inputs of one benchmark run."""
    run_dir = Path(run)
    if not run_dir.is_dir():
        run_dir = Path(results_dir) / run_dir
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"no safe benchmark run directory: {run}")

    run_json = run_dir / "run.json"
    if not run_json.is_file() or run_json.is_symlink():
        raise ValueError(f"benchmark run has no safe run.json: {run_dir}")
    try:
        metadata = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"benchmark run.json is unreadable: {run_dir}") from error
    if not isinstance(metadata, dict):
        raise ValueError("benchmark run.json must contain an object")

    fingerprint = metadata.get("fingerprint")
    identity = metadata.get("identity")
    if not isinstance(fingerprint, str) or not isinstance(identity, dict):
        raise ValueError("benchmark run.json lacks a fingerprint-bound identity")
    expected_fingerprint = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError("benchmark run.json fingerprint does not match its identity")

    config = metadata.get("config")
    raw_adapters = identity.get("adapters")
    if not isinstance(raw_adapters, list) or not isinstance(config, dict):
        raise ValueError("benchmark run.json has an incomplete immutable identity")
    try:
        bench_config = BenchConfig.model_validate(config)
    except ValueError as error:
        raise ValueError("benchmark run.json has an invalid benchmark config") from error
    canonical_config = bench_config.model_dump(mode="json")
    if config != canonical_config:
        raise ValueError(
            "benchmark run.json config is not a complete canonical benchmark config"
        )

    from evals.runner import _run_identity

    expected_identity = _run_identity(bench_config, raw_adapters)
    if identity != expected_identity:
        raise ValueError(
            "benchmark run.json identity does not match the disclosed benchmark config"
        )

    targets = bench_config.targets
    judge = bench_config.judge
    if not targets:
        raise ValueError("benchmark run.json has no target configs")
    if not judge.enabled:
        raise ValueError("benchmark run did not configure an enabled judge")
    try:
        verified_judges = _verified_run_judge_adapters(raw_adapters)
    except ValueError as error:
        raise ValueError(f"benchmark run judge identity is incompatible: {error}") from error
    if not verified_judges:
        raise ValueError("benchmark run selected no judge-template adapter")
    return judge, fingerprint, targets, tuple(raw_adapters)


def _calibration_inputs_from_args(
    args,
) -> tuple[
    JudgeConfig,
    str | None,
    Sequence[BenchTarget | Mapping[str, object]] | None,
    Sequence[Mapping[str, object]] | None,
]:
    """Resolve either exploratory CLI inputs or one verified benchmark run."""
    from evals.config import build_judge_config

    run = getattr(args, "run", None)
    if run is None:
        return (
            build_judge_config(args),
            getattr(args, "run_fingerprint", None),
            getattr(args, "target_configs", None),
            getattr(args, "run_adapter_identities", None),
        )

    override_names = (
        "config",
        "judge_base_url",
        "judge_model",
        "judge_api_key_env",
        "judge_reasoning_effort",
        "judge_extra_body",
        "judge_secondary",
    )
    supplied = [
        name.replace("_", "-")
        for name in override_names
        if getattr(args, name, None) is not None
    ]
    if supplied:
        raise ValueError(
            "--run cannot be combined with judge/config overrides: "
            + ", ".join(f"--{name}" for name in supplied)
        )
    return _load_bound_run(
        run,
        results_dir=getattr(args, "results_dir", "bench/results/accuracy"),
    )


def _at_least(value: object, threshold: float) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= threshold


async def run_judge_calibration(
    config: JudgeConfig,
    *,
    calibration_set: str | Path | None = None,
    min_agreement: float = STANDARD_MIN_AGREEMENT,
    min_coverage: float = STANDARD_MIN_COVERAGE,
    max_position_flip: float = STANDARD_MAX_POSITION_FLIP,
    max_self_preference_gap: float = STANDARD_MAX_SELF_PREFERENCE_GAP,
    require_self_preference: bool = False,
    run_fingerprint: str | None = None,
    target_configs: Sequence[BenchTarget | Mapping[str, object]] | None = None,
    target_identities: Sequence[
        tuple[str, str] | Mapping[str, str]
    ] | None = None,
    run_adapter_identities: Sequence[Mapping[str, object]] | None = None,
    http_factory=None,
) -> dict:
    """Execute the gold-label agreement and counterbalanced-position gate."""
    if not config.enabled:
        raise ValueError("judge calibration requires judge.base_url and judge.model")
    for name, value in (
        ("min_agreement", min_agreement),
        ("min_coverage", min_coverage),
        ("max_position_flip", max_position_flip),
        ("max_self_preference_gap", max_self_preference_gap),
    ):
        if isinstance(value, bool) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    cases, raw, source = load_calibration_cases(calibration_set)
    set_summary = _set_summary(cases)
    run_binding = _run_binding(
        run_fingerprint=run_fingerprint,
        target_configs=target_configs,
        target_identities=target_identities,
        run_adapter_identities=run_adapter_identities,
    )
    templates = {
        name: {
            "production": judge_template_identity(name),
            "counterbalanced": judge_template_identity(
                name, counterbalanced=True
            ),
        }
        for name in _TEMPLATES
    }
    thresholds = {
        "min_agreement": min_agreement,
        "min_coverage": min_coverage,
        "max_position_flip": max_position_flip,
        "max_self_preference_gap": max_self_preference_gap,
        "require_self_preference": require_self_preference,
    }
    identity = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "judge": config.model_dump(mode="json"),
        "calibration_set_sha256": hashlib.sha256(raw).hexdigest(),
        "prompt_templates": templates,
        "judge_protocol": judge_protocol_identity(),
        "thresholds": thresholds,
        "run_binding": run_binding,
    }
    fingerprint = hashlib.sha256(_canonical_json(identity)).hexdigest()
    factory = http_factory or (lambda: httpx.AsyncClient())
    judge = build_judge_client(config, http_factory=factory)

    async def grade(case: CalibrationCase, *, counterbalanced: bool) -> JudgeVerdict:
        return await judge.grade(
            question=case.question,
            expected=case.expected,
            response=case.response,
            template_name=case.template,
            counterbalanced=counterbalanced,
        )

    async def grade_pair(case: CalibrationCase) -> tuple[JudgeVerdict, JudgeVerdict]:
        production, counterbalanced = await asyncio.gather(
            grade(case, counterbalanced=False),
            grade(case, counterbalanced=True),
        )
        return production, counterbalanced

    paired_verdicts = await asyncio.gather(*(grade_pair(case) for case in cases))
    production_verdicts = tuple(pair[0] for pair in paired_verdicts)
    counterbalanced_verdicts = tuple(pair[1] for pair in paired_verdicts)
    production_predictions = tuple(verdict.correct for verdict in production_verdicts)
    counterbalanced_predictions = tuple(
        verdict.correct for verdict in counterbalanced_verdicts
    )
    agreement = {
        "production": _agreement_with_templates(cases, production_predictions),
        "counterbalanced": _agreement_with_templates(
            cases, counterbalanced_predictions
        ),
    }
    position = _position_bias(
        cases, production_predictions, counterbalanced_predictions
    )

    endpoints = config.grading_endpoints()
    bound_target_pairs = {
        (identity["base_url"], identity["model"])
        for identity in run_binding["target_identities"]
    }
    member_reports = []
    for index, endpoint in enumerate(endpoints):
        production_member = tuple(
            verdict.constituent_verdicts()[index].correct
            for verdict in production_verdicts
        )
        counterbalanced_member = tuple(
            verdict.constituent_verdicts()[index].correct
            for verdict in counterbalanced_verdicts
        )
        endpoint_identity = endpoint.resolved_identity()
        matches_target = (
            endpoint_identity in bound_target_pairs
            if bound_target_pairs and endpoint_identity is not None
            else None
        )
        member_reports.append(
            {
                "endpoint": endpoint.model_dump(mode="json"),
                "target_relationship": (
                    "self"
                    if matches_target is True
                    else "nonself"
                    if matches_target is False
                    else "unknown"
                ),
                "agreement": {
                    "production": _agreement_with_templates(
                        cases, production_member
                    ),
                    "counterbalanced": _agreement_with_templates(
                        cases, counterbalanced_member
                    ),
                },
                "position_bias": _position_bias(
                    cases, production_member, counterbalanced_member
                ),
                "self_preference": _self_preference(
                    endpoint,
                    cases,
                    production_member,
                    counterbalanced_member,
                    matches_target=matches_target,
                    force_measurement=require_self_preference,
                ),
            }
        )

    checks = {
        "calibration_set_balanced": set_summary["balanced"]
        and set_summary["paired"],
        "production_coverage": _at_least(
            agreement["production"]["coverage"], min_coverage
        ),
        "counterbalanced_coverage": _at_least(
            agreement["counterbalanced"]["coverage"], min_coverage
        ),
        "production_agreement": _at_least(
            agreement["production"]["agreement"], min_agreement
        ),
        "counterbalanced_agreement": _at_least(
            agreement["counterbalanced"]["agreement"], min_agreement
        ),
        "position_flip_rate": position["flip_rate"] is not None
        and position["flip_rate"] <= max_position_flip,
    }
    for variant in ("production", "counterbalanced"):
        for template_name in _TEMPLATES:
            checks[f"{variant}_{template_name}_agreement"] = (
                _at_least(
                    agreement[variant]["by_template"][template_name]["agreement"],
                    min_agreement,
                )
            )
    for index, member_report in enumerate(member_reports):
        preference = member_report["self_preference"]
        measured = preference["status"] == "measured"
        if measured:
            preference_ok = (
                preference["max_self_favoring_gap"] <= max_self_preference_gap
            )
        else:
            preference_ok = not require_self_preference
        checks[f"member_{index}_self_preference"] = preference_ok

    standard_thresholds = (
        min_agreement >= STANDARD_MIN_AGREEMENT
        and min_coverage >= STANDARD_MIN_COVERAGE
        and max_position_flip <= STANDARD_MAX_POSITION_FLIP
        and max_self_preference_gap <= STANDARD_MAX_SELF_PREFERENCE_GAP
    )
    headline_checks = {
        "run_bound": run_binding["bound"],
        "calibration_set_headline_shape": set_summary["headline_shape"],
        "thresholds_not_weaker_than_standard": standard_thresholds,
        "production_coverage": _at_least(
            agreement["production"]["coverage"], STANDARD_MIN_COVERAGE
        ),
        "counterbalanced_coverage": _at_least(
            agreement["counterbalanced"]["coverage"], STANDARD_MIN_COVERAGE
        ),
        "production_agreement": _at_least(
            agreement["production"]["agreement"], STANDARD_MIN_AGREEMENT
        ),
        "counterbalanced_agreement": _at_least(
            agreement["counterbalanced"]["agreement"], STANDARD_MIN_AGREEMENT
        ),
        "position_flip_rate": position["flip_rate"] is not None
        and position["flip_rate"] <= STANDARD_MAX_POSITION_FLIP,
    }
    for variant in ("production", "counterbalanced"):
        for template_name in _TEMPLATES:
            headline_checks[f"{variant}_{template_name}_agreement"] = _at_least(
                agreement[variant]["by_template"][template_name]["agreement"],
                STANDARD_MIN_AGREEMENT,
            )
    for index, member_report in enumerate(member_reports):
        preference = member_report["self_preference"]
        relationship = member_report["target_relationship"]
        if relationship == "self":
            preference_ok = (
                preference["status"] == "measured"
                and preference["max_self_favoring_gap"]
                <= STANDARD_MAX_SELF_PREFERENCE_GAP
            )
        elif relationship == "nonself":
            if require_self_preference:
                preference_ok = (
                    preference["status"] == "measured"
                    and preference["max_self_favoring_gap"]
                    <= STANDARD_MAX_SELF_PREFERENCE_GAP
                )
            else:
                preference_ok = preference["status"] == "not_applicable"
        else:
            preference_ok = False
        headline_checks[f"member_{index}_self_preference"] = preference_ok

    passed = all(checks.values())
    headline_eligible = passed and all(headline_checks.values())

    evidence = []
    for case, first, second in zip(
        cases, production_verdicts, counterbalanced_verdicts, strict=True
    ):
        evidence.append(
            {
                "id": case.id,
                "template": case.template,
                "human_correct": case.human_correct,
                "label_provenance": case.label_provenance,
                "label_source": case.label_source,
                "label_source_revision": case.label_source_revision,
                "label_source_record": case.label_source_record,
                "label_source_license": case.label_source_license,
                "response_base_url": case.response_base_url,
                "response_model": case.response_model,
                "production": first.as_dict(),
                "counterbalanced": second.as_dict(),
            }
        )

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "created_at": utc_now(),
        "identity": identity,
        "run_binding": run_binding,
        "calibration_set": {
            "source": source,
            "sha256": hashlib.sha256(raw).hexdigest(),
            **set_summary,
        },
        "agreement": agreement,
        "position_bias": position,
        "members": member_reports,
        "checks": checks,
        "passed": passed,
        "headline_checks": headline_checks,
        "headline_eligible": headline_eligible,
        "cases": evidence,
    }


async def run_calibration_cli(args, *, http_factory=None) -> int:
    """CLI adapter that always emits the calibration evidence before exit."""
    (
        judge_config,
        run_fingerprint,
        target_configs,
        run_adapter_identities,
    ) = _calibration_inputs_from_args(args)

    report = await run_judge_calibration(
        judge_config,
        calibration_set=args.calibration_set,
        min_agreement=args.min_agreement,
        min_coverage=args.min_coverage,
        max_position_flip=args.max_position_flip,
        max_self_preference_gap=args.max_self_preference_gap,
        require_self_preference=args.require_self_preference,
        run_fingerprint=run_fingerprint,
        target_configs=target_configs,
        target_identities=getattr(args, "target_identities", None),
        run_adapter_identities=run_adapter_identities,
        http_factory=http_factory,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output is not None:
        atomic_write_text(Path(args.output), rendered + "\n")
    print(rendered)
    gate = report["headline_eligible"] if run_fingerprint is not None else report["passed"]
    return 0 if gate else 1
