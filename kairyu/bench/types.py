"""Schemas for benchmark suites: config, items, and per-pair results.

Everything is a frozen pydantic model (repo convention, m7 D3): configs are
loaded from YAML/CLI once and never mutated; results are written atomically
and re-read for resume, so round-tripping through JSON must be lossless.
"""

from __future__ import annotations

import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairyu.bench.targets import normalize_base_url, validate_api_key_env

SCHEMA_VERSION = 1

# `extra_body` is an escape hatch for vendor knobs, not a way to silently
# retarget or reshape the benchmark request. It is merged LAST, so anything
# built elsewhere would win if it were allowed through: the adapter's prompt and
# token budget (`messages`, `max_tokens`, `temperature`) and the typed sampling
# fields below. Allowing those would let the effective request disagree with the
# configuration that the run fingerprint and methodology record.
_RESERVED_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "temperature",
        "max_tokens",
        "reasoning_effort",
        "top_p",
        "seed",
    }
)


def _validate_extra_body(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"extra_body_json is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("extra_body_json must be a JSON object")
    clashes = sorted(_RESERVED_BODY_KEYS & set(parsed))
    if clashes:
        raise ValueError(
            f"extra_body_json may not override {', '.join(clashes)}: those fields "
            "are built from the adapter's request and this endpoint's typed "
            "sampling policy, which the run fingerprint records"
        )
    return value


class SamplingOptions(BaseModel):
    """Request knobs that belong to an endpoint, not to a benchmark.

    Fugu reports scores at each model's maximum reasoning effort, and its τ³
    user simulator ran at `low`; reproducing either needs the effort (and, for
    vendors that spell it differently, `extra_body`) to reach the wire.
    Adapters own prompts and `max_tokens`; the target owns sampling.

    Slots that issue their own chat requests pick this up through `call_chat`.
    The three external-harness slots (SWE-Bench Pro, Terminal-Bench, τ³) cannot:
    they drive a separate CLI, so each maps whatever its harness exposes and
    annotates what it cannot forward — see `wire_overrides()` callers.
    """

    reasoning_effort: str | None = None
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None
    # JSON object string: hashable, so the frozen models stay hashable, and
    # validated once at config load instead of at request time.
    extra_body_json: str | None = None

    @field_validator("extra_body_json")
    @classmethod
    def _check_extra_body(cls, value: str | None) -> str | None:
        return _validate_extra_body(value)

    def extra_body(self) -> dict:
        return json.loads(self.extra_body_json) if self.extra_body_json else {}

    def wire_overrides(self) -> dict:
        """Body fields this endpoint adds to every request."""
        body: dict = {}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(self.extra_body())
        return body


class BenchTarget(SamplingOptions):
    """One scoreboard column: a model name on an OpenAI-compatible endpoint.

    Single models and orchestrations are both just model names ("llama-70b"
    vs "kairyu-auto-max") — usually on the same gateway base_url.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    name: str = ""  # scoreboard label; defaults to model
    base_url: str
    model: str
    api_key_env: str | None = None  # env var NAME, never the key itself
    max_context_tokens: int | None = None  # gate for long-context items
    max_output_tokens: int = 8192
    supports_vision: bool = True

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return normalize_base_url(value)

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str | None) -> str | None:
        return validate_api_key_env(value)

    def label(self) -> str:
        return self.name or self.model


class JudgeEndpointConfig(SamplingOptions):
    """One OpenAI-compatible endpoint participating in LLM grading."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "KAIRYU_JUDGE_API_KEY"
    concurrency: int = Field(default=4, ge=1)
    max_retries: int = Field(default=3, ge=0)

    @field_validator("base_url")
    @classmethod
    def _normalize_optional_base_url(cls, value: str | None) -> str | None:
        return normalize_base_url(value) if value is not None else None

    @field_validator("model")
    @classmethod
    def _validate_optional_model(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("judge model must be non-empty and have no surrounding whitespace")
        return value

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str) -> str:
        validated = validate_api_key_env(value)
        if validated is None:  # pragma: no cover - field type rejects None first
            raise ValueError("api_key_env must be an environment-variable name")
        return validated

    @property
    def enabled(self) -> bool:
        return self.base_url is not None and self.model is not None

    def resolved_identity(self) -> tuple[str, str] | None:
        if self.base_url is None or self.model is None:
            return None
        return self.base_url, self.model


class JudgeConfig(JudgeEndpointConfig):
    """Primary judge plus optional independent grading-panel members.

    The primary endpoint also serves as the τ-bench user simulator, which
    remains a single-model role. ``additional_judges`` apply only to the
    pointwise HLE/CharXiv grading path. With two total judges, strict majority
    intentionally means unanimity; a disagreement is unjudged, never broken in
    favour of the primary.
    """

    aggregation: Literal["strict-majority"] = "strict-majority"
    additional_judges: tuple[JudgeEndpointConfig, ...] = ()

    @model_validator(mode="after")
    def _validate_panel(self) -> JudgeConfig:
        if self.additional_judges and not self.enabled:
            raise ValueError(
                "judge.base_url and judge.model are required when additional_judges are configured"
            )
        identities: list[tuple[str, str]] = []
        primary = self.resolved_identity()
        if primary is not None:
            identities.append(primary)
        for member in self.additional_judges:
            identity = member.resolved_identity()
            if identity is None:
                raise ValueError("each additional judge requires both base_url and model")
            identities.append(identity)
        if len(identities) != len(set(identities)):
            raise ValueError(
                "judge panel members must use distinct resolved endpoint/model identities"
            )
        return self

    def grading_endpoints(self) -> tuple[JudgeEndpointConfig, ...]:
        """Ordered primary-first endpoints used for pointwise grading."""
        if not self.enabled:
            return ()
        primary = JudgeEndpointConfig(
            base_url=self.base_url,
            model=self.model,
            api_key_env=self.api_key_env,
            concurrency=self.concurrency,
            max_retries=self.max_retries,
            reasoning_effort=self.reasoning_effort,
            top_p=self.top_p,
            seed=self.seed,
            extra_body_json=self.extra_body_json,
        )
        return (primary, *self.additional_judges)


_IMMUTABLE_IMAGE_RE = re.compile(r"(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})")


class ExecutionConfig(BaseModel):
    """How benchmark-generated code is executed.

    Docker images are accepted only by immutable content identity. A local
    image ID is useful immediately after building the supplied runtime, while
    ``repository@sha256:...`` is the portable registry form. Mutable tags are
    deliberately not resolved on behalf of a benchmark run.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    runner: Literal["local", "docker"] = "local"
    image: str | None = None
    cpus: float = Field(default=1.0, gt=0)
    pids_limit: int = Field(default=64, ge=2)
    disk_mb: int = Field(default=256, ge=1, le=65_536)
    pull_policy: Literal["never"] = "never"

    @field_validator("cpus", mode="before")
    @classmethod
    def _reject_boolean_cpus(cls, value):
        if isinstance(value, bool):
            raise ValueError("execution.cpus must be a finite positive number")
        return value

    @model_validator(mode="after")
    def _validate_runner_image(self) -> ExecutionConfig:
        if self.runner == "local":
            if self.image is not None:
                raise ValueError("execution.image is valid only for runner='docker'")
            if (self.cpus, self.pids_limit, self.disk_mb, self.pull_policy) != (
                1.0,
                64,
                256,
                "never",
            ):
                raise ValueError("Docker execution resource settings require runner='docker'")
            return self
        if self.image is None:
            raise ValueError("execution.image is required for runner='docker'")
        if _IMMUTABLE_IMAGE_RE.fullmatch(self.image) is None:
            raise ValueError(
                "execution.image must be an immutable sha256:<64 hex> image ID "
                "or repository@sha256:<64 hex> digest"
            )
        repository = self.image.split("@", 1)[0]
        if "@" in self.image and (repository.startswith("-") or repository.endswith(":")):
            raise ValueError("execution.image repository is invalid")
        return self


class BenchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    suite: Literal["fugu", "core"] = "fugu"
    targets: tuple[BenchTarget, ...] = Field(min_length=1)
    judge: JudgeConfig = JudgeConfig()
    execution: ExecutionConfig = ExecutionConfig()
    limit: int | None = Field(default=None, ge=1)  # None = full dataset
    smoke: bool = False  # preset: limit<=SMOKE_LIMIT
    offline_fixtures: bool = False  # read committed fixtures, no cache/network
    only: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    seed: int = 0
    # Trials per task for the agentic harnesses (Terminal-Bench `-k`, tau
    # `--num-trials`). Fugu reports tau-3 Banking as pass@4; the default stays 1
    # because each extra attempt is another full docker/agent run.
    attempts: int = Field(default=1, ge=1)
    concurrency: int = Field(default=8, ge=1)  # in-flight requests per pair
    request_timeout_s: float = Field(default=600.0, gt=0)
    retries: int = Field(default=2, ge=0)
    cache_dir: str | None = None
    results_dir: str = "bench/results/fugu"
    run_id: str | None = None  # reuse an id to resume
    rerun: bool = False  # ignore existing pair results
    download: bool = True  # auto-download missing datasets before running
    progress: bool = True  # live per-slot progress (bars on a TTY, lines in logs)

    @model_validator(mode="before")
    @classmethod
    def _suite_results_dir_default(cls, value):
        """Keep explicit paths while defaulting new runs under their suite."""
        if isinstance(value, dict) and value.get("results_dir") is None:
            value = {
                **value,
                "results_dir": f"bench/results/{value.get('suite', 'fugu')}",
            }
        return value


SMOKE_LIMIT = 20


class BenchItem(BaseModel):
    """One dataset row; payload is adapter-private normalized fields."""

    model_config = ConfigDict(frozen=True)

    id: str
    payload: dict


class ChatRequestSpec(BaseModel):
    """OpenAI wire-format request an adapter built for one item."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[dict, ...]
    max_tokens: int | None = None
    temperature: float = 0.0
    est_prompt_tokens: int | None = None  # chars/4 heuristic, for context gating


class LogLikelihoodRequestSpec(BaseModel):
    """Teacher-forced continuations an adapter ranks for one item.

    Each continuation is scored against the same exact context.  Ordering is
    significant: it is retained on the wire and provides the deterministic
    tie-break used by multiple-choice adapters.
    """

    model_config = ConfigDict(frozen=True)

    context: str
    continuations: tuple[str, ...]
    reduction: Literal["sum", "mean_token"] = "sum"
    est_prompt_tokens: int | None = None

    @field_validator("context")
    @classmethod
    def _validate_context(cls, value: str) -> str:
        if not value:
            raise ValueError("context must be non-empty")
        return value

    @field_validator("continuations")
    @classmethod
    def _validate_continuations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("continuations must contain at least one candidate")
        if any(not continuation for continuation in value):
            raise ValueError("continuations must be non-empty strings")
        if len(value) != len(set(value)):
            raise ValueError("continuations must be unique")
        return value


class ContinuationLogLikelihood(BaseModel):
    """Validated, selected-token evidence for one teacher-forced candidate."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    continuation: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    token_logprobs: tuple[float, ...]
    text_offsets: tuple[int, ...]
    sum_logprob: float = Field(le=0.0)
    score: float = Field(le=0.0)

    @field_validator("continuation")
    @classmethod
    def _validate_continuation(cls, value: str) -> str:
        if not value:
            raise ValueError("candidate continuation must be non-empty")
        return value

    @field_validator("token_ids", mode="before")
    @classmethod
    def _validate_token_ids(cls, value):
        if not isinstance(value, (list, tuple)) or any(
            type(token_id) is not int or token_id < 0 for token_id in value
        ):
            raise ValueError("token_ids must contain non-negative integers")
        return value

    @field_validator("text_offsets", mode="before")
    @classmethod
    def _validate_text_offsets(cls, value):
        if not isinstance(value, (list, tuple)) or any(type(offset) is not int for offset in value):
            raise ValueError("text_offsets must contain integers")
        return value

    @model_validator(mode="after")
    def _validate_parallel_evidence(self) -> ContinuationLogLikelihood:
        lengths = {
            len(self.token_ids),
            len(self.tokens),
            len(self.token_logprobs),
            len(self.text_offsets),
        }
        if lengths != {len(self.token_ids)} or not self.token_ids:
            raise ValueError("candidate token evidence must have equal nonzero lengths")
        if any(logprob > 0.0 for logprob in self.token_logprobs):
            raise ValueError("token_logprobs must be <= 0")
        if not math.isclose(
            self.sum_logprob,
            sum(self.token_logprobs),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("sum_logprob must equal the selected token_logprobs sum")
        return self


class LogLikelihoodResponse(BaseModel):
    """Ordered evidence returned for all candidates in one request spec."""

    model_config = ConfigDict(frozen=True)

    reduction: Literal["sum", "mean_token"]
    prompt_token_ids: tuple[int, ...]
    candidates: tuple[ContinuationLogLikelihood, ...]

    @field_validator("prompt_token_ids", mode="before")
    @classmethod
    def _validate_prompt_token_ids(cls, value):
        if not isinstance(value, (list, tuple)) or any(
            type(token_id) is not int or token_id < 0 for token_id in value
        ):
            raise ValueError("prompt_token_ids must contain non-negative integers")
        return value

    @model_validator(mode="after")
    def _validate_ranking_evidence(self) -> LogLikelihoodResponse:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        continuations = [candidate.continuation for candidate in self.candidates]
        if len(continuations) != len(set(continuations)):
            raise ValueError("candidate continuations must be unique")
        for candidate in self.candidates:
            expected = candidate.sum_logprob
            if self.reduction == "mean_token":
                expected /= len(candidate.token_ids)
            if not math.isclose(candidate.score, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("candidate score does not match the declared reduction")
        return self


class SkipItem(BaseModel):
    """build_request() verdict: this item cannot run against this target."""

    model_config = ConfigDict(frozen=True)

    reason: str


ItemStatus = Literal["completed", "failed", "unjudged", "skipped"]


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    item_id: str
    status: ItemStatus
    score: float | None = None  # 0..1
    response_excerpt: str | None = None  # capped, evidence only
    error: str | None = None
    # Deterministic scorer evidence (for example IFEval per-instruction booleans).
    details: dict | None = None
    # Single: {model, correct, raw_excerpt}; panels add aggregation + ordered votes.
    judge: dict | None = None
    latency_s: float | None = None

    @field_validator("score", mode="before")
    @classmethod
    def _validate_score(cls, value):
        if value is None:
            return None
        if type(value) not in (int, float):
            raise ValueError("item score must be a number within [0, 1]")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("item score must be finite within [0, 1]")
        if not 0 <= value <= 1:
            raise ValueError("item score must be within [0, 1]")
        return value

    @field_validator("latency_s", mode="before")
    @classmethod
    def _validate_latency(cls, value):
        if value is None:
            return None
        if type(value) not in (int, float):
            raise ValueError("item latency_s must be a finite non-negative number")
        if (isinstance(value, float) and not math.isfinite(value)) or value < 0:
            raise ValueError("item latency_s must be a finite non-negative number")
        return value


PairStatus = Literal["completed", "partial", "skipped", "failed"]
CrossRunPolicy = Literal[
    "allowed",
    "withheld_unresolved_runtime",
    "withheld_unpinned_execution",
]


class PairResult(BaseModel):
    """One scoreboard cell: one benchmark run against one target."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: int = SCHEMA_VERSION
    benchmark: str
    target: str
    run_fingerprint: str | None = None
    # Exact local harness source observed for this evidence.  Cross-commit
    # history accepts a pair only when this matches the clean run attestation.
    source_identity: dict | None = None
    status: PairStatus
    reason: str | None = None  # "docker unavailable", "dataset unavailable", ...
    # {"score":…, "n_total":…, "n_scored":…, "n_unjudged":…, "n_skipped":…, "n_failed":…}
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    items: tuple[ItemResult, ...] = ()  # per-item evidence (roadmap §6)
    methodology: dict = Field(default_factory=dict)
    annotations: tuple[str, ...] = ()
    # Whether this cell may be compared with a published full-suite score, and
    # why not. Static substitutions (LongBench for Long Context Reasoning),
    # run-time ones (the tau2 fallback) and run-level ones (a subset run, fixture
    # data) all land here, so the report never has to infer comparability from a
    # benchmark-name allow list.
    comparable: bool = True
    incomparable_reasons: tuple[str, ...] = ()
    # Separate from published-score comparability. An unresolved harness/data
    # runtime may remain visible in a complete Fugu scoreboard, but can never
    # produce a cross-commit regression delta even when both runs share it.
    cross_run_policy: CrossRunPolicy = "allowed"
    cross_run_reason: str | None = None
    started_at: str = ""
    finished_at: str = ""

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value):
        if type(value) is not int or value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        return value

    @field_validator("comparable", mode="before")
    @classmethod
    def _validate_comparable(cls, value):
        if type(value) is not bool:
            raise ValueError("comparable must be a boolean")
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def _preserve_metric_types(cls, value):
        """Reject coercions that could turn malformed evidence into a score."""
        if not isinstance(value, dict):
            raise ValueError("metrics must be an object")
        for name, metric in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if metric is None:
                continue
            if type(metric) not in (int, float):
                raise ValueError(f"metrics.{name} must be a number or null")
            if isinstance(metric, float) and not math.isfinite(metric):
                raise ValueError(f"metrics.{name} must be finite")
        return value

    @property
    def score(self) -> float | None:
        """Return a bounded primary score without trusting stored evidence.

        ``metrics`` is persisted adapter output and old artifacts may predate
        stricter runner checks.  In particular, converting an arbitrarily large
        JSON integer directly to ``float`` raises ``OverflowError``.  Reporters
        and resume progress are observability paths: malformed evidence must not
        be able to terminate the run from either one.
        """
        if not isinstance(self.metrics, dict):
            return None
        value = self.metrics.get("score")
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        # Compare while the value is still an integer.  Python can compare an
        # arbitrary-size integer safely even when it cannot convert it to float.
        if not 0 <= value <= 1:
            return None
        try:
            score = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None


_PAIR_COUNT_METRICS = frozenset(
    {"n_total", "n_scored", "n_unjudged", "n_skipped", "n_failed"}
)
_MAX_SAFE_INTEGER = (1 << 53) - 1


def _json_evidence_error(
    value: object,
    *,
    path: str,
    require_safe_integers: bool,
    active: set[int] | None = None,
) -> str | None:
    """Validate the raw tree without Pydantic's lossy JSON coercions."""
    if value is None or type(value) in (str, bool):
        return None
    if type(value) is int:
        if require_safe_integers and abs(value) > _MAX_SAFE_INTEGER:
            return f"{path} must be a safe integer"
        return None
    if type(value) is float:
        return None if math.isfinite(value) else f"{path} must be finite"

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        return f"{path} contains a reference cycle"
    active.add(identity)
    try:
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                error = _json_evidence_error(
                    getattr(value, name),
                    path=f"{path}.{name}",
                    require_safe_integers=require_safe_integers,
                    active=active,
                )
                if error is not None:
                    return error
            return None
        if type(value) in (list, tuple):
            for index, item in enumerate(value):
                error = _json_evidence_error(
                    item,
                    path=f"{path}[{index}]",
                    require_safe_integers=require_safe_integers,
                    active=active,
                )
                if error is not None:
                    return error
            return None
        if type(value) is dict:
            for name, item in value.items():
                if not isinstance(name, str):
                    return f"{path} contains a non-string object key"
                error = _json_evidence_error(
                    item,
                    path=f"{path}.{name}",
                    require_safe_integers=require_safe_integers,
                    active=active,
                )
                if error is not None:
                    return error
            return None
        return f"{path} contains non-JSON value {type(value).__name__}"
    finally:
        active.remove(identity)


def pair_result_evidence_error(
    value: object,
    *,
    expected_benchmark: str | None = None,
    expected_target: str | None = None,
    require_history_safe_counts: bool = False,
) -> str | None:
    """Explain why adapter/stored pair evidence is unsafe, or return ``None``.

    The Pydantic schema deliberately remains able to load legacy artifacts and
    adversarial fixtures.  This semantic boundary is used by the runner before
    progress, persistence, or history, and by aggregation before rendering.
    """
    if not isinstance(value, PairResult):
        return f"expected PairResult, got {type(value).__name__}"
    try:
        PairResult.model_validate(value)
    except (TypeError, ValueError):
        return "pair does not match the strict PairResult schema"
    if expected_benchmark is not None and value.benchmark != expected_benchmark:
        return "benchmark identity does not match the scheduled adapter"
    if expected_target is not None and value.target != expected_target:
        return "target identity does not match the scheduled target"
    if not isinstance(value.metrics, dict):
        return "metrics must be an object"

    for name, metric in value.metrics.items():
        if not isinstance(name, str) or not name:
            return "metric names must be non-empty strings"
        if metric is None:
            continue
        if isinstance(metric, bool) or not isinstance(metric, int | float):
            return f"metrics.{name} must be a number or null"
        if isinstance(metric, float) and not math.isfinite(metric):
            return f"metrics.{name} must be finite"
        if (
            type(metric) is int
            and name != "score"
            and name not in _PAIR_COUNT_METRICS
            and abs(metric) > _MAX_SAFE_INTEGER
        ):
            return f"metrics.{name} must be a safe integer"

    raw_score = value.metrics.get("score")
    if raw_score is not None and value.score is None:
        return "metrics.score must be finite and within [0, 1]"

    counts: dict[str, int] = {}
    for name in _PAIR_COUNT_METRICS:
        raw_count = value.metrics.get(name)
        if raw_count is None:
            continue
        if type(raw_count) is not int or raw_count < 0:
            return f"metrics.{name} must be a non-negative integer or null"
        if require_history_safe_counts and raw_count > _MAX_SAFE_INTEGER:
            return f"metrics.{name} must be a non-negative safe integer for history"
        counts[name] = raw_count
    total = counts.get("n_total")
    scored = counts.get("n_scored")
    if total is not None and scored is not None and scored > total:
        return "metrics.n_scored must not exceed metrics.n_total"
    if not isinstance(value.items, tuple) or any(
        not isinstance(item, ItemResult) for item in value.items
    ):
        return "items must contain ItemResult evidence"
    for index, item in enumerate(value.items):
        raw_item_score = item.score
        if raw_item_score is not None:
            if (
                isinstance(raw_item_score, bool)
                or not isinstance(raw_item_score, int | float)
                or (isinstance(raw_item_score, float) and not math.isfinite(raw_item_score))
                or not 0 <= raw_item_score <= 1
            ):
                return f"items[{index}].score must be finite and within [0, 1]"
        latency = item.latency_s
        if latency is not None:
            if (
                isinstance(latency, bool)
                or not isinstance(latency, int | float)
                or (isinstance(latency, float) and not math.isfinite(latency))
                or latency < 0
            ):
                return f"items[{index}].latency_s must be finite and non-negative"
    tree_error = _json_evidence_error(
        value,
        path="pair",
        require_safe_integers=require_history_safe_counts,
    )
    if tree_error is not None:
        return tree_error
    try:
        serialized = value.model_dump_json()
        if require_history_safe_counts:
            # History hashes canonical Python JSON after the pair is saved.  A
            # JSON integer beyond the interpreter's digit limit, including one
            # nested in methodology/item evidence, must be quarantined here.
            json.loads(serialized)
    except (OverflowError, RecursionError, TypeError, ValueError):
        return "pair evidence is not safely JSON serializable"
    return None


class DownloadReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    adapter: str
    status: Literal["ok", "cached", "gated", "unavailable", "extras_missing"]
    detail: str = ""


class BenchExtrasMissing(RuntimeError):
    """Raised when an optional dependency group is required but not installed."""

    def __init__(self, extra: str, purpose: str) -> None:
        super().__init__(f"{purpose} requires the [{extra}] extra: pip install 'kairyu[{extra}]'")
        self.extra = extra


class DatasetGated(RuntimeError):
    """Raised when a HF dataset needs license acceptance + token."""

    def __init__(self, dataset: str) -> None:
        super().__init__(
            f"dataset {dataset!r} is gated: accept the license at "
            f"https://huggingface.co/datasets/{dataset} and set HF_TOKEN"
        )
        self.dataset = dataset


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset cannot be fetched (missing repo, network, ...)."""
