"""Schemas for benchmark suites: config, items, and per-pair results.

Everything is a frozen pydantic model (repo convention, m7 D3): configs are
loaded from YAML/CLI once and never mutated; results are written atomically
and re-read for resume, so round-tripping through JSON must be lossless.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from evals.targets import normalize_base_url, validate_api_key_env

SCHEMA_VERSION = 1
JSON_SAFE_INTEGER_MAX = (1 << 53) - 1

# `extra_body` is an escape hatch for vendor knobs, not a way to silently
# retarget or reshape the benchmark request. The adapter's prompt and token
# budget (`messages`, `max_tokens`, `temperature`) and typed sampling fields
# below are reserved because allowing those would let the effective request
# disagree with the configuration that the run fingerprint and methodology
# record. Adapter-specific fields remain available to ordinary chat suites;
# suites that own such a field must reject it at their own precondition boundary.
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

    Adapters own prompts and `max_tokens`; the target owns sampling. Chat
    adapters apply these options through `call_chat`, and the run fingerprint
    records the exact target policy sent on the wire.
    """

    reasoning_effort: str | None = None
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None
    # JSON object string: hashable, so the frozen models stay hashable, and
    # validated once at config load instead of at request time.
    extra_body_json: str | None = None

    @field_validator("seed", mode="before")
    @classmethod
    def _require_integer_seed(cls, value):
        if value is not None and type(value) is not int:
            raise ValueError("sampling seed must be an integer")
        if value is not None and abs(value) > JSON_SAFE_INTEGER_MAX:
            raise ValueError("sampling seed must be a JSON-safe integer")
        return value

    @field_validator("top_p", mode="before")
    @classmethod
    def _require_numeric_top_p(cls, value):
        if value is not None and type(value) not in (int, float):
            raise ValueError("top_p must be a number in (0, 1]")
        return value

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


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ServedConfigIdentity(BaseModel):
    """Operator-declared immutable identity of the served deployment config."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    label: str
    sha256: str

    @field_validator("label")
    @classmethod
    def _normalize_label(cls, value: str) -> str:
        label = value.strip()
        if not label:
            raise ValueError("served config label must be non-empty")
        if "`" in label or any(
            unicodedata.category(character).startswith("C") for character in label
        ):
            raise ValueError(
                "served config label must not contain control characters or backticks"
            )
        return label

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError("served config sha256 must be 64 lowercase hexadecimal characters")
        return value


class QuantizationProfile(BaseModel):
    """Operator-declared effective quantization class for one served target.

    The profile classifies rows in the quantization sweep.  Exact checkpoint,
    kernel, scale, ignored-layer, and deployment details remain content-bound
    by :class:`ServedConfigIdentity`; this compact declaration must never be
    mistaken for remote attestation of the server that answered a request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    # Dense BF16 is represented honestly as unquantized weights plus BF16
    # compute.  ``bf16`` is not a checkpoint quantization method.
    weight_method: Literal["none", "fp8", "int8", "awq", "gptq", "nvfp4"]
    compute_dtype: Literal["bfloat16"]
    # Effective storage dtype, never the unresolved runtime request ``auto``.
    kv_cache_dtype: Literal["bfloat16", "fp8_e4m3"]


class BenchTarget(SamplingOptions):
    """One scoreboard column: a model name on an OpenAI-compatible endpoint.

    Single models and orchestrations are both just model names ("llama-70b"
    vs "kairyu-auto-max") — usually on the same gateway base_url.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    name: str = ""  # scoreboard label; defaults to model
    base_url: str
    model: str
    served_config: ServedConfigIdentity | None = None
    quantization: QuantizationProfile | None = None
    # ``adapter`` preserves the benchmark-authored request temperature (0.0
    # today). ``recommended`` deliberately omits generation-default fields so
    # a #351-capable backend may apply its checkpoint generation_config.json.
    # This is an operator-requested wire policy, not remote attestation that a
    # third-party endpoint actually loaded those defaults.
    sampling_mode: Literal["adapter", "recommended"] = "adapter"
    temperature: float | None = Field(default=None, ge=0.0)
    api_key_env: str | None = None  # env var NAME, never the key itself
    max_context_tokens: int | None = Field(default=None, ge=1)  # long-context gate
    max_output_tokens: int = Field(default=8192, ge=1)
    supports_vision: bool = True

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return normalize_base_url(value)

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str | None) -> str | None:
        return validate_api_key_env(value)

    @field_validator("temperature", mode="before")
    @classmethod
    def _validate_temperature(cls, value):
        if value is None:
            return None
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("temperature must be a finite non-negative number")
        return value

    @model_validator(mode="after")
    def _validate_sampling_policy(self) -> BenchTarget:
        if self.sampling_mode != "recommended":
            return self
        if self.temperature is not None or self.top_p is not None:
            raise ValueError(
                "recommended sampling cannot be combined with explicit "
                "temperature or top_p"
            )
        generation_overrides = {
            "top_k",
            "min_p",
            "repetition_penalty",
        } & set(self.extra_body())
        if generation_overrides:
            raise ValueError(
                "recommended sampling cannot be combined with explicit "
                f"{', '.join(sorted(generation_overrides))} in extra_body_json"
            )
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_absent_extensions(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict:
        """Keep pre-#371/pre-quantization target records byte-shape compatible."""
        serialized = handler(self)
        if self.quantization is None:
            serialized.pop("quantization", None)
        if self.sampling_mode == "adapter":
            serialized.pop("sampling_mode", None)
        if self.temperature is None:
            serialized.pop("temperature", None)
        return serialized

    def label(self) -> str:
        return self.name or self.model



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

    suite: Literal["core", "quantization", "structured", "long-context"]
    targets: tuple[BenchTarget, ...] = Field(min_length=1)
    execution: ExecutionConfig = ExecutionConfig()
    limit: int | None = Field(default=None, ge=1)  # None = full dataset
    smoke: bool = False  # preset: limit<=SMOKE_LIMIT
    offline_fixtures: bool = False  # read committed fixtures, no cache/network
    only: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    seed: int = 0
    # Trials per source item. Adapters retain an ordered seed sweep beneath each
    # dataset row. The default stays 1 because every attempt adds a model call.
    attempts: int = Field(default=1, ge=1)
    # Sixteen in-flight requests lets one engine continuously batch work for
    # per-GPU throughput; this is true request concurrency, not queue depth.
    concurrency: int = Field(default=16, ge=1)  # in-flight requests per pair
    request_timeout_s: float = Field(default=600.0, gt=0)
    retries: int = Field(default=2, ge=0)
    cache_dir: str | None = None
    results_dir: str | None = None
    run_id: str | None = None  # reuse an id to resume
    rerun: bool = False  # ignore existing pair results
    download: bool = True  # auto-download missing datasets before running
    progress: bool = True  # live per-slot progress (bars on a TTY, lines in logs)

    @field_validator("attempts", mode="before")
    @classmethod
    def _require_integer_attempts(cls, value):
        if (
            type(value) is not int
            or value < 1
            or value > JSON_SAFE_INTEGER_MAX
        ):
            raise ValueError("attempts must be a positive JSON-safe integer")
        return value

    @model_validator(mode="before")
    @classmethod
    def _suite_results_dir_default(cls, value):
        """Keep explicit paths while defaulting new runs under their suite."""
        if (
            isinstance(value, dict)
            and value.get("results_dir") is None
            and value.get("suite") is not None
        ):
            value = {
                **value,
                "results_dir": f"bench/results/{value['suite']}",
            }
        return value


SMOKE_LIMIT = 20


def run_configuration_incomparable_reasons(
    *,
    limit: int | None,
    offline_fixtures: bool,
) -> tuple[str, ...]:
    """Return the run-wide reasons every pair and scoreboard cell must retain."""

    reasons: list[str] = []
    if limit is not None:
        reasons.append(
            f"subset run: at most {limit} items per benchmark, not the full set"
        )
    if offline_fixtures:
        reasons.append(
            "offline fixture mode lacks cache-bound dataset identity; scores are "
            "diagnostics, not measurements"
        )
    return tuple(reasons)


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


class GenerationTimingEvidence(BaseModel):
    """Replayable timing facts from one successful streamed target request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: Literal["openai-chat-sse-v1"] = "openai-chat-sse-v1"
    ttft_s: float
    generation_span_s: float
    completion_tokens: int | None = None
    tps: float | None = None
    semantic_events: int = Field(ge=1)
    request_attempts: int = Field(default=1, ge=1)
    usage_missing: bool

    @field_validator("ttft_s", "generation_span_s", "tps", mode="before")
    @classmethod
    def _validate_timing_number(cls, value, info):
        if value is None and info.field_name == "tps":
            return None
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"generation timing {info.field_name} must be finite and non-negative")
        return float(value)

    @field_validator("completion_tokens", mode="before")
    @classmethod
    def _validate_completion_tokens(cls, value):
        if value is None:
            return None
        if type(value) is not int or not 0 <= value <= JSON_SAFE_INTEGER_MAX:
            raise ValueError("generation completion_tokens must be a safe non-negative integer")
        return value

    @field_validator("semantic_events", "request_attempts", mode="before")
    @classmethod
    def _validate_positive_count(cls, value, info):
        if type(value) is not int or not 1 <= value <= JSON_SAFE_INTEGER_MAX:
            raise ValueError(f"generation timing {info.field_name} must be a safe positive integer")
        return value

    @field_validator("usage_missing", mode="before")
    @classmethod
    def _validate_usage_missing(cls, value):
        if type(value) is not bool:
            raise ValueError("generation timing usage_missing must be a boolean")
        return value

    @model_validator(mode="after")
    def _validate_derived_rate(self) -> GenerationTimingEvidence:
        if self.usage_missing != (self.completion_tokens is None):
            raise ValueError("generation timing usage_missing disagrees with completion_tokens")
        rate_is_defined = (
            self.completion_tokens is not None
            and self.completion_tokens >= 2
            and self.generation_span_s > 0
        )
        if not rate_is_defined:
            if self.tps is not None:
                raise ValueError("generation timing TPS is defined without enough evidence")
            return self
        expected = (self.completion_tokens - 1) / self.generation_span_s
        if self.tps is None or not math.isclose(
            self.tps, expected, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("generation timing TPS disagrees with raw timing evidence")
        return self


class StructuredTokenUsage(BaseModel):
    """Strict token accounting retained from one Chat Completions arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens", mode="before")
    @classmethod
    def _require_safe_count(cls, value, info):
        if type(value) is not int or not 0 <= value <= JSON_SAFE_INTEGER_MAX:
            raise ValueError(f"structured usage {info.field_name} must be a safe count")
        return value

    @model_validator(mode="after")
    def _validate_total(self) -> StructuredTokenUsage:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("structured usage total_tokens must equal prompt plus completion")
        return self


class StructuredArmEvidence(BaseModel):
    """Raw, score-replayable evidence for one constrained or control request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: Literal["constrained", "unconstrained"]
    outcome: Literal["completed", "schema_rejected", "failed"]
    http_status: int
    content: str | None = None
    refusal: str | None = None
    message_auxiliary: str | None = None
    finish_reason: str | None = None
    usage: StructuredTokenUsage | None = None
    timing: GenerationTimingEvidence | None = None
    latency_s: float
    error: str | None = None

    @field_validator("http_status", mode="before")
    @classmethod
    def _validate_http_status(cls, value):
        if type(value) is not int or not 0 <= value <= 599:
            raise ValueError("structured arm http_status must be an integer from 0 to 599")
        return value

    @field_validator("latency_s", mode="before")
    @classmethod
    def _validate_latency(cls, value):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("structured arm latency_s must be finite and non-negative")
        return value

    @field_validator("content")
    @classmethod
    def _bound_content(cls, value: str | None) -> str | None:
        if value is not None and len(value) > 262_144:
            raise ValueError("structured arm content exceeds the retained evidence bound")
        return value

    @field_validator("refusal")
    @classmethod
    def _bound_refusal(cls, value: str | None) -> str | None:
        if value is not None and (not value or len(value) > 262_144):
            raise ValueError("structured arm refusal must be non-empty and bounded")
        return value

    @field_validator("message_auxiliary")
    @classmethod
    def _bound_message_auxiliary(cls, value: str | None) -> str | None:
        if value is not None and (not value or len(value) > 262_144):
            raise ValueError("structured arm auxiliary message payload is invalid")
        if value is not None:
            try:
                parsed = json.loads(value)
                canonical = json.dumps(
                    parsed,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (RecursionError, TypeError, ValueError) as error:
                raise ValueError(
                    "structured arm auxiliary message payload is not canonical JSON"
                ) from error
            if (
                not isinstance(parsed, dict)
                or not parsed
                or not set(parsed) <= {"tool_calls", "function_call"}
                or (
                    "tool_calls" in parsed
                    and (
                        not isinstance(parsed["tool_calls"], list)
                        or not parsed["tool_calls"]
                    )
                )
                or (
                    "function_call" in parsed
                    and (
                        not isinstance(parsed["function_call"], dict)
                        or not parsed["function_call"]
                    )
                )
                or canonical != value
            ):
                raise ValueError(
                    "structured arm auxiliary message payload is not canonical JSON"
                )
        return value

    @field_validator("error")
    @classmethod
    def _bound_error(cls, value: str | None) -> str | None:
        if value is not None and (not value or len(value) > 2_000):
            raise ValueError("structured arm error must be non-empty and bounded")
        return value

    @model_validator(mode="after")
    def _validate_outcome_shape(self) -> StructuredArmEvidence:
        if self.outcome == "completed":
            if (
                self.http_status != 200
                or (
                    self.refusal is not None
                    and (self.content is not None or self.message_auxiliary is not None)
                )
                or not self.finish_reason
                or self.error is not None
            ):
                raise ValueError("completed structured arm has inconsistent response evidence")
            return self
        if self.outcome == "schema_rejected":
            if (
                self.arm != "constrained"
                or self.http_status not in {400, 422}
                or self.content is not None
                or self.refusal is not None
                or self.message_auxiliary is not None
                or self.finish_reason is not None
                or self.usage is not None
                or self.timing is not None
                or self.error is None
            ):
                raise ValueError("schema-rejected structured arm has inconsistent evidence")
            return self
        if (
            self.content is not None
            or self.refusal is not None
            or self.message_auxiliary is not None
            or self.finish_reason is not None
            or self.usage is not None
            or self.timing is not None
            or self.error is None
        ):
            raise ValueError("failed structured arm has inconsistent response evidence")
        return self


class StructuredOutputEvidence(BaseModel):
    """The paired constrained/control observation retained beneath one item."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    protocol: Literal["paired-json-schema-v1"] = "paired-json-schema-v1"
    seed: int | None
    order: tuple[
        Literal["constrained", "unconstrained"],
        Literal["constrained", "unconstrained"],
    ]
    constrained: StructuredArmEvidence
    unconstrained: StructuredArmEvidence

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value):
        if value is not None and (
            type(value) is not int or abs(value) > JSON_SAFE_INTEGER_MAX
        ):
            raise ValueError("structured paired seed must be a JSON-safe integer or null")
        return value

    @model_validator(mode="after")
    def _validate_arm_identities(self) -> StructuredOutputEvidence:
        if set(self.order) != {"constrained", "unconstrained"}:
            raise ValueError("structured arm order must contain each arm exactly once")
        if self.constrained.arm != "constrained":
            raise ValueError("structured constrained evidence has the wrong arm identity")
        if self.unconstrained.arm != "unconstrained":
            raise ValueError("structured unconstrained evidence has the wrong arm identity")
        return self


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    item_id: str
    status: ItemStatus
    score: float | None = None  # 0..1
    response_excerpt: str | None = None  # capped, evidence only
    error: str | None = None
    # Deterministic scorer evidence (for example IFEval per-instruction booleans).
    details: dict | None = None
    latency_s: float | None = None
    # Successful streamed target-call timing for the measured target.
    timing: GenerationTimingEvidence | None = None
    # Raw paired-arm evidence for the structured-output conformance adapter.
    structured_output: StructuredOutputEvidence | None = None
    # Multi-seed chat evidence stays grouped under its source dataset item.
    # Flattening attempts into ``PairResult.items`` would make Wilson and
    # paired-comparison code falsely treat correlated trials as independent.
    sampling_attempts: tuple[SamplingAttemptResult, ...] | None = None

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

    @model_validator(mode="after")
    def _validate_sampling_attempts(self) -> ItemResult:
        attempts = self.sampling_attempts
        if attempts is None:
            if self.structured_output is not None:
                failed = any(
                    arm.outcome == "failed"
                    for arm in (
                        self.structured_output.constrained,
                        self.structured_output.unconstrained,
                    )
                )
                if failed:
                    if self.status != "failed" or self.score is not None or not self.error:
                        raise ValueError(
                            "failed structured arms require a failed unscored item"
                        )
                elif self.status != "completed" or self.score is None or self.error is not None:
                    raise ValueError(
                        "complete structured arms require a completed scored item"
                    )
            return self
        if self.structured_output is not None:
            raise ValueError("sampling parents cannot duplicate structured arm evidence")
        if not attempts:
            raise ValueError("sampling_attempts must be non-empty when present")
        expected = list(range(1, len(attempts) + 1))
        if [attempt.attempt for attempt in attempts] != expected:
            raise ValueError("sampling_attempts must use consecutive 1-based indices")
        seeds = [attempt.seed for attempt in attempts]
        if len(seeds) != len(set(seeds)):
            raise ValueError("sampling_attempts must use unique seeds")
        if any(attempt.result.item_id != self.item_id for attempt in attempts):
            raise ValueError("sampling attempt item_id must match its parent item")
        completed_scores = [
            attempt.result.score
            for attempt in attempts
            if attempt.result.status == "completed" and attempt.result.score is not None
        ]
        complete = len(completed_scores) == len(attempts)
        if complete:
            if self.status != "completed" or self.score is None:
                raise ValueError("complete sampling attempts require a completed parent")
            expected_score = sum(completed_scores) / len(completed_scores)
            if not math.isclose(self.score, expected_score, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("sampling parent score must equal the attempt mean")
            if self.error is not None:
                raise ValueError("a complete sampling parent cannot carry an error")
        else:
            child_statuses = [attempt.result.status for attempt in attempts]
            missing_score = any(
                attempt.result.status == "completed" and attempt.result.score is None
                for attempt in attempts
            )
            if "failed" in child_statuses or missing_score:
                expected_status = "failed"
            elif "unjudged" in child_statuses:
                expected_status = "unjudged"
            else:
                expected_status = "skipped"
            if self.status != expected_status:
                raise ValueError(
                    "sampling parent status must match its incomplete attempt outcomes"
                )
            if self.score is not None:
                raise ValueError("an incomplete sampling item cannot carry a score")
            counts = {
                status: child_statuses.count(status)
                for status in ("failed", "unjudged", "skipped")
            }
            if missing_score:
                counts["invalid"] = sum(
                    attempt.result.status == "completed" and attempt.result.score is None
                    for attempt in attempts
                )
            reason = ", ".join(
                f"{count} {name}" for name, count in counts.items() if count
            )
            if self.error != f"sampling sweep incomplete: {reason}":
                raise ValueError("sampling parent error must describe its attempt outcomes")
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_single_attempt_extension(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict:
        """Do not change any legacy/single-attempt ItemResult JSON shape."""
        serialized = handler(self)
        if self.sampling_attempts is None:
            serialized.pop("sampling_attempts", None)
        if self.structured_output is None:
            serialized.pop("structured_output", None)
        return serialized


class SamplingAttemptResult(BaseModel):
    """One seed-specific result retained beneath a source benchmark item."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    attempt: int = Field(ge=1)
    seed: int
    result: ItemResult

    @field_validator("attempt", "seed", mode="before")
    @classmethod
    def _require_integer_identity(cls, value, info):
        if type(value) is not int:
            raise ValueError(f"sampling {info.field_name} must be an integer")
        if abs(value) > JSON_SAFE_INTEGER_MAX:
            raise ValueError(
                f"sampling {info.field_name} must be a JSON-safe integer"
            )
        return value

    @model_validator(mode="after")
    def _reject_nested_sampling(self) -> SamplingAttemptResult:
        if self.result.sampling_attempts is not None:
            raise ValueError("sampling attempt results cannot contain nested attempts")
        return self


ItemResult.model_rebuild()


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
    # why not. Declared dataset substitutions and run-level restrictions such
    # as subsets or fixture data land here, so the report never infers
    # comparability from a benchmark-name allow list.
    comparable: bool = True
    incomparable_reasons: tuple[str, ...] = ()
    # Separate from published-score comparability. An unresolved harness/data
    # runtime may remain visible in a complete eval scoreboard, but can never
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
_MAX_SAFE_INTEGER = JSON_SAFE_INTEGER_MAX


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


def sampling_configuration_error(value: object) -> str | None:
    """Validate the sampling subset retained in run/target configuration."""

    if not isinstance(value, dict):
        return "expected sampling configuration must be an object"
    attempts = value.get("attempts")
    seed = value.get("seed")
    mode = value.get("mode")
    temperature = value.get("temperature")
    top_p = value.get("top_p")
    required = value.get("required")
    if (
        type(attempts) is not int
        or attempts < 1
        or attempts > JSON_SAFE_INTEGER_MAX
        or (seed is not None and type(seed) is not int)
        or (type(seed) is int and abs(seed) > JSON_SAFE_INTEGER_MAX)
        or mode not in {"adapter", "recommended"}
        or (
            temperature is not None
            and (
                type(temperature) not in (int, float)
                or not math.isfinite(temperature)
                or temperature < 0
            )
        )
        or (
            top_p is not None
            and (
                type(top_p) not in (int, float)
                or not math.isfinite(top_p)
                or not 0 < top_p <= 1
            )
        )
        or (mode == "recommended" and (temperature is not None or top_p is not None))
        or type(required) is not bool
    ):
        return "expected sampling configuration is invalid"
    if required and attempts > 1:
        first_seed = seed if seed is not None else 0
        if first_seed + attempts - 1 > JSON_SAFE_INTEGER_MAX:
            return "expected sampling seed schedule exceeds JSON-safe integers"
    return None


def pair_result_evidence_error(
    value: object,
    *,
    expected_benchmark: str | None = None,
    expected_target: str | None = None,
    expected_sampling: dict[str, object] | None = None,
    expected_structured_run: dict[str, object] | None = None,
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

    grouped_items = [item for item in value.items if item.sampling_attempts is not None]
    if grouped_items:
        if len(grouped_items) != len(value.items):
            return "sampling evidence must group every source item"
        item_ids = [item.item_id for item in value.items]
        if len(item_ids) != len(set(item_ids)):
            return "sampling source item ids must be unique"

        sampling = value.methodology.get("sampling")
        if not isinstance(sampling, dict):
            return "sampling evidence requires methodology.sampling"
        attempts = sampling.get("attempts")
        seeds = sampling.get("seeds")
        mode = sampling.get("mode")
        expected_omissions = (
            ["temperature", "top_p", "top_k", "min_p", "repetition_penalty"]
            if mode == "recommended"
            else []
        )
        if type(attempts) is not int or attempts < 2:
            return "methodology.sampling.attempts must be an integer of at least two"
        if (
            not isinstance(seeds, list)
            or len(seeds) != attempts
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != attempts
        ):
            return "methodology.sampling.seeds must be the unique ordered seed schedule"
        if mode not in {"adapter", "recommended"}:
            return "methodology.sampling.mode is invalid"
        if sampling.get("wire_omitted") != expected_omissions:
            return "methodology.sampling.wire_omitted does not match the sampling mode"
        if sampling.get("recommended_defaults_attested") is not False:
            return "methodology.sampling must not claim remote default attestation"

        for index, item in enumerate(value.items):
            retained = item.sampling_attempts
            assert retained is not None
            if len(retained) != attempts:
                return f"items[{index}].sampling_attempts does not match the attempt budget"
            if [attempt.seed for attempt in retained] != seeds:
                return f"items[{index}].sampling_attempts does not match the seed schedule"

        expected_counts = {
            "n_total": len(value.items),
            "n_scored": sum(
                item.status == "completed" and item.score is not None for item in value.items
            ),
            "n_unjudged": sum(item.status == "unjudged" for item in value.items),
            "n_skipped": sum(item.status == "skipped" for item in value.items),
            "n_failed": sum(item.status == "failed" for item in value.items),
        }
        for name, expected in expected_counts.items():
            if value.metrics.get(name) != expected:
                return f"metrics.{name} does not match grouped source-item evidence"
        if value.metrics.get("sampling_attempts") != attempts:
            return "metrics.sampling_attempts does not match methodology"
        if value.metrics.get("sampling_base_items") != len(value.items):
            return "metrics.sampling_base_items does not match grouped source items"

        complete = expected_counts["n_scored"] == len(value.items)
        if value.metrics.get("sampling_complete") != int(complete):
            return "metrics.sampling_complete does not match grouped evidence"
        scored_values = [
            float(item.score)
            for item in value.items
            if item.status == "completed" and item.score is not None
        ]
        expected_score = sum(scored_values) / len(scored_values) if scored_values else None
        stored_score = value.score
        if expected_score is None:
            if stored_score is not None:
                return "metrics.score does not match grouped source-item evidence"
        elif stored_score is None or not math.isclose(
            stored_score,
            expected_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return "metrics.score does not match grouped source-item evidence"

        completed_count = expected_counts["n_scored"]
        failed_count = expected_counts["n_failed"]
        if failed_count > len(value.items) / 2:
            expected_status = "failed"
        elif completed_count == len(value.items):
            expected_status = "completed"
        elif not scored_values:
            expected_status = "skipped"
        else:
            expected_status = "partial"
        if value.status != expected_status:
            return "pair status does not match grouped source-item evidence"
        reasons = []
        for name, label in (
            ("n_unjudged", "unjudgeable"),
            ("n_skipped", "skipped"),
            ("n_failed", "failed"),
        ):
            count = expected_counts[name]
            if count:
                reasons.append(f"{count}/{len(value.items)} items {label}")
        expected_reason = (
            None
            if expected_status == "completed"
            else "; ".join(reasons) or "no scoreable items"
        )
        if value.reason != expected_reason:
            return "pair reason does not match grouped source-item evidence"

    if expected_sampling is not None:
        from evals.sampling import seed_schedule

        config_error = sampling_configuration_error(expected_sampling)
        if config_error is not None:
            return config_error
        expected_attempts = expected_sampling.get("attempts")
        expected_seed = expected_sampling.get("seed")
        expected_mode = expected_sampling.get("mode")
        expected_temperature = expected_sampling.get("temperature")
        required = expected_sampling.get("required")
        assert isinstance(expected_attempts, int)
        assert expected_seed is None or isinstance(expected_seed, int)
        assert expected_mode in {"adapter", "recommended"}
        assert required is not None
        sampling = value.methodology.get("sampling")
        policy_requires_methodology = bool(required) and (
            expected_attempts > 1
            or expected_mode == "recommended"
            or expected_temperature is not None
        )
        if (
            policy_requires_methodology
            and value.status in {"completed", "partial"}
            and not isinstance(sampling, dict)
        ):
            return "pair omits the configured chat sampling methodology"
        if isinstance(sampling, dict):
            try:
                schedule = list(seed_schedule(expected_seed, expected_attempts))
            except ValueError as error:
                return f"expected sampling seed schedule is invalid: {error}"
            expected_omissions = (
                ["temperature", "top_p", "top_k", "min_p", "repetition_penalty"]
                if expected_mode == "recommended"
                else []
            )
            expected_seed_source = (
                "target.seed"
                if expected_seed is not None
                else (
                    "generated from zero"
                    if expected_attempts > 1
                    else "wire omission"
                )
            )
            wire_temperature = (
                None
                if expected_mode == "recommended"
                else (0.0 if expected_temperature is None else expected_temperature)
            )
            if sampling.get("attempts") != expected_attempts:
                return "sampling methodology attempt budget disagrees with run config"
            if sampling.get("seeds") != schedule:
                return "sampling methodology seeds disagree with target config"
            if sampling.get("mode") != expected_mode:
                return "sampling methodology mode disagrees with target config"
            if sampling.get("seed_source") != expected_seed_source:
                return "sampling methodology seed source disagrees with target config"
            if sampling.get("wire_omitted") != expected_omissions:
                return "sampling methodology omissions disagree with target config"
            if sampling.get("recommended_defaults_attested") is not False:
                return "sampling methodology must not claim remote default attestation"
            methodology_temperature = value.methodology.get("temperature")
            if (
                isinstance(methodology_temperature, bool)
                or methodology_temperature != wire_temperature
            ):
                return "sampling methodology temperature disagrees with target config"

    def has_structured_item(item: ItemResult) -> bool:
        if item.structured_output is not None:
            return True
        return bool(
            item.sampling_attempts
            and any(
                attempt.result.structured_output is not None
                for attempt in item.sampling_attempts
            )
        )

    structured_claimed = (
        value.benchmark == "structured-output"
        or "structured_output" in value.methodology
        or any(name.startswith("structured_") for name in value.metrics)
        or any(has_structured_item(item) for item in value.items)
    )
    if structured_claimed:
        if value.benchmark != "structured-output":
            return "structured-output evidence is attached to another benchmark"
        from evals.structured import structured_pair_evidence_error

        try:
            structured_error = structured_pair_evidence_error(
                value,
                expected_sampling=expected_sampling,
                expected_run=expected_structured_run,
            )
        except (BenchExtrasMissing, OSError, UnicodeError, ValueError) as error:
            return f"structured evidence validator unavailable: {error}"
        if structured_error is not None:
            return structured_error

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
