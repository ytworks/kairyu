"""Serving benchmark harness: TTFT p50/p99, TPOT, goodput over an OpenAI-compatible API.

Works against ANY OpenAI-compatible server (kairyu, vLLM, SGLang), so the same
script produces the M2 acceptance comparison on identical hardware. Prints only
measured values and labels the target endpoint; nothing is estimated.

Datasets:
  --dataset sharegpt.json   ShareGPT-format JSON (list of {"conversations": [...]})
  (omitted)                 synthetic prompts, clearly labeled as synthetic

Examples:
  uv run python verification/l1/performance/serving_bench.py --base-url http://localhost:8000 \
      --model kairyu-mock --num-requests 128 --concurrency 128
  uv run python verification/l1/performance/serving_bench.py --base-url http://localhost:8000 \
      --model kairyu --stage-trace
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _datetime
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from evidence.reporting import (
    PERCENTILE_METHOD,
    atomic_write_json,
    nearest_rank_percentile,
)
from verification.l1.performance.profiling import (
    profile_scope,
    profile_trace_path,
    record_function_scope,
    remove_trace_artifact,
    validate_trace_artifact_path,
    write_trace_artifact,
)
from verification.targets import (
    TARGET_SPEC_FORMAT,
    BenchTarget,
    normalize_base_url,
    parse_target_spec,
    target_api_key,
)

_SSE_PREFIX = "data: "
_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_MODEL = "kairyu-mock"
_TRACE_VERSION = "2.0"
_TRACE_HEADER = {"X-Kairyu-Trace": "1"}
_STAGE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_TRACE_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}\Z")
_EXECUTION_STATUS_RE = re.compile(r"[a-z_]+(?:,[a-z_]+){0,15}\Z")
_RFC3339_MILLISECONDS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\Z"
)
_ORCHESTRATION_KINDS = frozenset(
    {"routing", "classification", "generation", "verification", "synthesis", "execution"}
)
_TRACE_STATUSES = frozenset({"success", "skipped", "failed"})
_NATIVE_STAGE_NAMES = (
    "tokenize",
    "queue_wait",
    "schedule",
    "prefill",
    "decode_step",
    "detokenize",
    "sse_write",
)
_MAX_STAGE_DURATION_NS = (1 << 63) - 1
_MAX_STAGE_OCCURRENCES = (1 << 31) - 1
_MAX_TRACE_TOKEN_COUNT = (1 << 63) - 1
_MAX_TRACE_EVENTS = 256
_TRACE_DURATION_NAMES = (
    "duration_ms",
    "queue_wait_ms",
    "service_ms",
    "first_token_ms",
    "post_first_ms",
    "total_ms",
)
_CLIENT_PROFILE_SCOPE = "client"
_CLIENT_PROFILE_RANGE = "evals.serving.client-measurement"


@dataclass(frozen=True)
class TraceStageMetrics:
    """Privacy-minimized timing and scalar usage for one trace event."""

    stage: str
    node: str
    role: str | None
    kind: str
    status: str
    attempt: int
    duration_ms: float | None = None
    occurrences: int | None = None
    queue_wait_ms: float | None = None
    service_ms: float | None = None
    first_token_ms: float | None = None
    post_first_ms: float | None = None
    total_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    proposals: int | None = None
    execution_status: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "node": self.node,
            "role": self.role,
            "kind": self.kind,
            "status": self.status,
            "attempt": self.attempt,
            "duration_ms": self.duration_ms,
            "occurrences": self.occurrences,
            "queue_wait_ms": self.queue_wait_ms,
            "service_ms": self.service_ms,
            "first_token_ms": self.first_token_ms,
            "post_first_ms": self.post_first_ms,
            "total_ms": self.total_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "proposals": self.proposals,
            "execution_status": self.execution_status,
        }


@dataclass(frozen=True)
class RequestMetrics:
    ttft_s: float
    total_s: float
    output_chunks: int
    completion_tokens: int | None = None  # from the include_usage final chunk
    # AUTO usage is cumulative across private stages.  An optional tokenizer
    # oracle counts only the assistant content that crossed the L3 boundary.
    public_completion_tokens: int | None = None
    response_text: str = ""
    trace_status: str = "not_requested"
    trace_version: str | None = None
    trace_stages: tuple[TraceStageMetrics, ...] = ()

    @property
    def tpot_s(self) -> float:
        """Prefer exact public-token granularity, then reported API usage.

        Targets without either counter fall back to chunk granularity; the
        selected method is always labeled in the artifact.
        """
        tokens = (
            self.public_completion_tokens
            if self.public_completion_tokens is not None
            else self.completion_tokens
        )
        units = tokens - 1 if tokens is not None else self.output_chunks - 1
        if units is None or units <= 0:
            return 0.0
        return (self.total_s - self.ttft_s) / units

    @property
    def token_granular(self) -> bool:
        return (
            self.public_completion_tokens is not None
            or self.completion_tokens is not None
        )


def _parse_utc_timestamp(value: object, label: str) -> _datetime.datetime:
    if type(value) is not str or _RFC3339_MILLISECONDS_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an RFC 3339 UTC millisecond string")
    try:
        parsed = _datetime.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} is not RFC 3339") from error
    if parsed.tzinfo is None or parsed.utcoffset() != _datetime.timedelta(0):
        raise ValueError(f"{label} must be UTC")
    return parsed


def _trace_text(value: object, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not value or len(value) > 128 or not value.isprintable():
        raise ValueError(f"{label} must be a bounded printable string")
    return value


def _trace_identifier(
    value: object,
    label: str,
    *,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or _TRACE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded safe identifier")
    return value


def _duration_ms(start: _datetime.datetime, end: _datetime.datetime) -> float:
    return (end - start).total_seconds() * 1e3


def _trace_usage(event: dict) -> tuple[int | None, int | None, int | None]:
    usage = event.get("usage")
    if usage is None:
        return None, None, None
    if not isinstance(usage, dict):
        raise ValueError("event.usage must be an object or null")
    values = tuple(
        usage.get(name)
        for name in ("prompt_tokens", "completion_tokens", "cached_tokens")
    )
    if any(
        type(value) is not int or not 0 <= value <= _MAX_TRACE_TOKEN_COUNT
        for value in values
    ):
        raise ValueError("event.usage token counts must be bounded non-negative integers")
    return values


def _parse_trace(
    trace: object,
    *,
    response_id: object,
) -> tuple[str, tuple[TraceStageMetrics, ...]]:
    """Validate trace v2 and retain only identity, status and derived durations."""

    if not isinstance(trace, dict) or trace.get("trace_version") != _TRACE_VERSION:
        raise ValueError("unsupported or malformed trace version")
    if type(response_id) is not str or not response_id:
        raise ValueError("terminal SSE metadata has no response id")
    if trace.get("request_id") != response_id:
        raise ValueError("trace request id does not match the SSE response id")

    trace_started = _parse_utc_timestamp(trace.get("started_at"), "trace.started_at")
    trace_completed = _parse_utc_timestamp(trace.get("completed_at"), "trace.completed_at")
    if trace_completed < trace_started:
        raise ValueError("trace timestamps are reversed")
    events = trace.get("events")
    if not isinstance(events, list) or len(events) > _MAX_TRACE_EVENTS:
        raise ValueError("trace.events must be a bounded list")

    stages: list[TraceStageMetrics] = []
    direct_stage_names: set[str] = set()
    for expected_seq, event in enumerate(events, start=1):
        if (
            not isinstance(event, dict)
            or type(event.get("seq")) is not int
            or event["seq"] != expected_seq
        ):
            raise ValueError("trace event sequence is not contiguous")
        kind = _trace_text(event.get("kind"), "event.kind")
        assert kind is not None
        if kind != "stage" and kind not in _ORCHESTRATION_KINDS:
            # v2 is additive. Unknown event kinds do not invalidate or enter
            # artifacts; their detail and identities remain unretained.
            continue
        status = _trace_text(event.get("status"), "event.status")
        assert status is not None
        if status not in _TRACE_STATUSES:
            continue
        try:
            node = _trace_identifier(event.get("node"), "event.node")
            role = _trace_identifier(
                event.get("role"), "event.role", optional=True
            )
        except ValueError:
            # Deployments historically allow free-form orchestration role
            # labels. Do not persist such values or invalidate the rest of a
            # valid envelope; omit only the event whose identity is unsafe.
            continue
        attempt = event.get("attempt")
        if type(attempt) is not int or not 0 <= attempt <= _MAX_STAGE_OCCURRENCES:
            raise ValueError("event.attempt must be a bounded non-negative integer")
        assert node is not None and kind is not None and status is not None
        if kind == "stage":
            detail = event.get("detail")
            expected_detail = {
                "stage",
                "duration_ns",
                "occurrences",
                "aggregation",
                "scope",
            }
            if type(detail) is not dict or not expected_detail.issubset(detail):
                raise ValueError("stage event detail is missing required fields")
            stage_name = detail["stage"]
            duration_ns = detail["duration_ns"]
            occurrences = detail["occurrences"]
            if (
                type(stage_name) is not str
                or _STAGE_NAME_RE.fullmatch(stage_name) is None
                or node != stage_name
                or stage_name in direct_stage_names
            ):
                raise ValueError("stage event has an unsafe or inconsistent stage name")
            if (
                type(duration_ns) is not int
                or not 0 <= duration_ns <= _MAX_STAGE_DURATION_NS
            ):
                raise ValueError("stage duration_ns is outside the supported integer range")
            if (
                type(occurrences) is not int
                or not 1 <= occurrences <= _MAX_STAGE_OCCURRENCES
            ):
                raise ValueError("stage occurrences is outside the supported integer range")
            if detail["aggregation"] != "sum" or detail["scope"] != "request-observed":
                raise ValueError("stage aggregation or scope is unsupported")
            if event.get("timing") is not None:
                raise ValueError("stage events must use scalar durations, not wall timestamps")
            if role != "engine" or status != "success" or attempt != 0:
                raise ValueError("stage event identity is unsupported")
            direct_stage_names.add(stage_name)
            stages.append(
                TraceStageMetrics(
                    stage=stage_name,
                    node=node,
                    role=role,
                    kind=kind,
                    status=status,
                    attempt=attempt,
                    duration_ms=duration_ns / 1e6,
                    occurrences=occurrences,
                )
            )
            continue
        timing = event.get("timing")
        if timing is None and status != "success":
            # A stage skipped before dispatch (budget, caller intent, an
            # image-conditional role on a text request — DTO-D11) or failed
            # before any observation has no wall timestamps: nothing to
            # retain, and it must not invalidate an otherwise valid envelope.
            continue
        if not isinstance(timing, dict):
            raise ValueError("event.timing must be an object")
        started = _parse_utc_timestamp(timing.get("started_at"), "event.started_at")
        completed = _parse_utc_timestamp(
            timing.get("completed_at"), "event.completed_at"
        )
        queued_raw = timing.get("queued_at")
        first_raw = timing.get("first_token_at")
        queued = (
            None
            if queued_raw is None
            else _parse_utc_timestamp(queued_raw, "event.queued_at")
        )
        first = (
            None
            if first_raw is None
            else _parse_utc_timestamp(first_raw, "event.first_token_at")
        )
        ordered = [value for value in (queued, started, first, completed) if value is not None]
        if ordered != sorted(ordered):
            raise ValueError("event timestamps are reversed")
        if ordered[0] < trace_started or ordered[-1] > trace_completed:
            raise ValueError("event timestamps are outside the trace envelope")

        input_tokens, output_tokens, cached_tokens = _trace_usage(event)
        proposals = None
        execution_status = None
        if node == "moa" and role == "moa" and kind == "synthesis" and status == "success":
            detail = event.get("detail")
            proposals = detail.get("proposals") if isinstance(detail, dict) else None
            if type(proposals) is not int or not 1 <= proposals <= 16:
                raise ValueError("successful MoA trace must report 1..16 proposals")

        if kind == "execution":
            detail = event.get("detail")
            execution_status = (
                detail.get("execution_status") if isinstance(detail, dict) else None
            )
            if (
                type(execution_status) is not str
                or len(execution_status) > 128
                or _EXECUTION_STATUS_RE.fullmatch(execution_status) is None
            ):
                raise ValueError("execution trace must report a safe execution_status")

        stage = "/".join((node, role or "-", kind))
        stages.append(
            TraceStageMetrics(
                stage=stage,
                node=node,
                role=role,
                kind=kind,
                status=status,
                attempt=attempt,
                queue_wait_ms=(
                    _duration_ms(queued, started) if queued is not None else None
                ),
                service_ms=_duration_ms(started, completed),
                first_token_ms=(
                    _duration_ms(started, first) if first is not None else None
                ),
                post_first_ms=(
                    _duration_ms(first, completed) if first is not None else None
                ),
                total_ms=_duration_ms(queued or started, completed),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                proposals=proposals,
                execution_status=execution_status,
            )
        )
    return _TRACE_VERSION, tuple(stages)


def load_prompts(dataset: Path | None, num_requests: int) -> tuple[list[str], str]:
    if dataset is None:
        prompts = [
            f"Question {i}: summarize the trade-offs of approach {i % 7} in two sentences."
            for i in range(num_requests)
        ]
        return prompts, "synthetic"
    records = json.loads(dataset.read_text(encoding="utf-8"))
    prompts = []
    for record in records:
        turns = record.get("conversations", [])
        human_turns = [t["value"] for t in turns if t.get("from") in ("human", "user")]
        if human_turns:
            prompts.append(human_turns[0])
        if len(prompts) >= num_requests:
            break
    if len(prompts) < num_requests:
        raise ValueError(
            f"dataset has only {len(prompts)} usable prompts, need {num_requests}"
        )
    return prompts, f"sharegpt:{dataset.name}"


async def run_one(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int,
    request_usage: bool = True,
    *,
    temperature: float | None = None,
    seed: int | None = None,
    min_tokens: int | None = None,
    ignore_eos: bool = False,
    request_trace: bool = False,
    capture_response: bool = False,
) -> RequestMetrics:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if seed is not None:
        body["seed"] = seed
    if min_tokens is not None:
        body["min_tokens"] = min_tokens
    if ignore_eos:
        body["ignore_eos"] = True
    if request_usage:
        body["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    ttft = None
    chunks = 0
    completion_tokens = None
    response_id: str | None = None
    response_id_conflict = False
    terminal_traces: list[tuple[object, object, int]] = []
    json_event_count = 0
    done_count = 0
    trace_protocol_invalid = False
    response_parts: list[str] = []
    # Clients use the shared canonical API root (ending in /v1). Keep the
    # request relative so URL joining cannot produce /v1/v1.
    async with client.stream(
        "POST",
        "chat/completions",
        headers=_TRACE_HEADER if request_trace else None,
        json=body,
    ) as response:
        if response.status_code == 400 and request_usage:
            # target rejects stream_options: retry once without (labeled fallback)
            return await run_one(
                client,
                model,
                prompt,
                max_tokens,
                request_usage=False,
                temperature=temperature,
                seed=seed,
                min_tokens=min_tokens,
                ignore_eos=ignore_eos,
                request_trace=request_trace,
                capture_response=capture_response,
            )
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith(_SSE_PREFIX):
                continue
            data = line[len(_SSE_PREFIX):]
            if data == "[DONE]":
                if request_trace:
                    done_count += 1
                    if done_count != 1:
                        trace_protocol_invalid = True
                continue
            if request_trace:
                if done_count:
                    trace_protocol_invalid = True
                json_event_count += 1
            chunk = json.loads(data)
            if "error" in chunk:
                error = chunk["error"] if isinstance(chunk["error"], dict) else {}
                label = error.get("code") or error.get("type") or "unknown"
                raise RuntimeError(f"target stream failed ({label})")
            if request_trace:
                chunk_id = chunk.get("id")
                if isinstance(chunk_id, str) and chunk_id:
                    if response_id is None:
                        response_id = chunk_id
                    elif chunk_id != response_id:
                        response_id_conflict = True
            if chunk.get("usage"):  # final usage chunk (empty choices)
                completion_tokens = chunk["usage"].get("completion_tokens")
            if request_trace and "kairyu_trace_v2" in chunk:
                if chunk.get("choices") != []:
                    trace_protocol_invalid = True
                else:
                    terminal_traces.append(
                        (
                            chunk.get("id"),
                            chunk["kairyu_trace_v2"],
                            json_event_count,
                        )
                    )
            content_parts = [
                content
                for choice in chunk.get("choices", [])
                if isinstance(
                    content := (choice.get("delta") or {}).get("content"),
                    str,
                )
                and content
            ]
            if content_parts:
                if capture_response:
                    response_parts.extend(content_parts)
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    trace_status = "missing" if request_trace else "not_requested"
    trace_version = None
    trace_stages: tuple[TraceStageMetrics, ...] = ()
    if request_trace:
        if done_count != 1:
            trace_protocol_invalid = True
        if terminal_traces:
            trace_status = "invalid"
        if (
            len(terminal_traces) == 1
            and not response_id_conflict
            and not trace_protocol_invalid
            and terminal_traces[0][2] == json_event_count
        ):
            terminal_id, trace, _event_index = terminal_traces[0]
            try:
                trace_version, trace_stages = _parse_trace(
                    trace,
                    response_id=terminal_id or response_id,
                )
            except ValueError:
                pass
            else:
                native_stages = {
                    stage.stage for stage in trace_stages if stage.kind == "stage"
                }
                if native_stages:
                    trace_status = (
                        "valid"
                        if set(_NATIVE_STAGE_NAMES).issubset(native_stages)
                        else "partial"
                    )
                else:
                    trace_status = "valid" if trace_stages else "missing"
        elif trace_protocol_invalid:
            trace_status = "invalid"
    return RequestMetrics(
        ttft_s=ttft if ttft is not None else total,
        total_s=total,
        output_chunks=chunks,
        completion_tokens=completion_tokens,
        response_text="".join(response_parts),
        trace_status=trace_status,
        trace_version=trace_version,
        trace_stages=trace_stages,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Compatibility wrapper for the package-owned nearest-rank definition."""
    return nearest_rank_percentile(sorted_values, fraction)


def resolve_target(args: argparse.Namespace) -> BenchTarget:
    """Resolve the shared target contract or the legacy split endpoint flags."""
    target_spec = getattr(args, "target", None)
    base_url = getattr(args, "base_url", None)
    model = getattr(args, "model", None)
    api_key_env = getattr(args, "api_key_env", None)
    literal_api_key = getattr(args, "api_key", None)

    if target_spec is not None:
        conflicting = [
            flag
            for flag, value in (
                ("--base-url", base_url),
                ("--model", model),
                ("--api-key-env", api_key_env),
                ("--api-key", literal_api_key),
            )
            if value is not None
        ]
        if conflicting:
            raise ValueError(
                f"--target cannot be combined with {', '.join(conflicting)}"
            )
        return parse_target_spec(target_spec)

    if literal_api_key == "":
        raise ValueError("--api-key must not be empty")
    resolved_model = model or _DEFAULT_MODEL
    return BenchTarget(
        name=resolved_model,
        base_url=normalize_base_url(base_url or _DEFAULT_BASE_URL),
        model=resolved_model,
        api_key_env=api_key_env,
    )


def resolve_api_key(args: argparse.Namespace, target: BenchTarget) -> str | None:
    """Resolve auth without ever adding the secret value to run metadata."""
    literal_api_key = getattr(args, "api_key", None)
    if literal_api_key is not None:
        return literal_api_key
    return target_api_key(
        target,
        required=target.api_key_env is not None,
    )


def build_run_config(args: argparse.Namespace) -> dict:
    """Run config embedded in results so files carry topology (G2 §8, design m5 D6).

    The topology args (``--tensor-parallel``, ``--dp-replicas``, ``--pd``) are
    labels for the results file; the GPU phase wires them into engine behavior.
    """
    target = resolve_target(args)
    return {
        "target": target.label(),
        "base_url": normalize_base_url(target.base_url),
        "model": target.model,
        "api_key_env": target.api_key_env,
        "api_key_source": (
            "legacy-cli"
            if getattr(args, "api_key", None) is not None
            else "environment"
            if target.api_key_env is not None
            else "none"
        ),
        "dataset": args.dataset,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "min_tokens": args.min_tokens,
        "ignore_eos": args.ignore_eos,
        "public_tokenizer_url": getattr(args, "public_tokenizer_url", None),
        "public_tokenizer_model": getattr(args, "public_tokenizer_model", None),
        "stage_trace": getattr(args, "stage_trace", False),
        "profile": getattr(args, "profile", False),
        "ttft_slo_s": args.ttft_slo_s,
        "tensor_parallel": args.tensor_parallel,
        "dp_replicas": args.dp_replicas,
        "pd": args.pd,
    }


def serving_artifact_paths(
    args: argparse.Namespace,
    *,
    now: _datetime.datetime | None = None,
) -> tuple[Path | None, Path | None]:
    """Preflight one result/trace pair before any target request is sent."""

    _validate_run_args(args)
    profiling = getattr(args, "profile", False)
    if type(profiling) is not bool:
        raise ValueError("profile enablement must be a boolean")
    if profiling and not args.results_dir:
        raise ValueError("--profile requires a non-empty --results-dir")
    if not args.results_dir:
        return None, None

    moment = now or _datetime.datetime.now(_datetime.UTC)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("artifact timestamp must be timezone-aware")
    stamp = moment.astimezone(_datetime.UTC).strftime("%Y-%m-%dT%H%M%S.%fZ")
    results_dir = Path(args.results_dir)
    result = results_dir / f"{stamp}-serving.json"
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"serving result already exists: {result}")
    if not profiling:
        return result, None
    trace = profile_trace_path(result, scope=_CLIENT_PROFILE_SCOPE)
    validated = validate_trace_artifact_path(trace, artifact_root=results_dir)
    return result, validated


def _validate_run_args(args: argparse.Namespace) -> None:
    for name in (
        "num_requests",
        "concurrency",
        "max_tokens",
        "tensor_parallel",
        "dp_replicas",
    ):
        value = getattr(args, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    for name in ("ttft_slo_s", "timeout"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    temperature = args.temperature
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) < 0.0
    ):
        raise ValueError("--temperature must be finite and non-negative")
    min_tokens = args.min_tokens
    if min_tokens is not None and (
        type(min_tokens) is not int
        or min_tokens < 0
        or min_tokens > args.max_tokens
    ):
        raise ValueError("--min-tokens must be between zero and --max-tokens")
    tokenizer_url = getattr(args, "public_tokenizer_url", None)
    tokenizer_model = getattr(args, "public_tokenizer_model", None)
    if bool(tokenizer_url) != bool(tokenizer_model):
        raise ValueError(
            "--public-tokenizer-url and --public-tokenizer-model must be provided together"
        )


async def _count_public_completion_tokens(
    results: list[RequestMetrics],
    *,
    tokenizer_url: str,
    tokenizer_model: str,
    concurrency: int,
    timeout: float,
) -> list[RequestMetrics]:
    """Count only public L3 content outside the timed serving interval."""

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout) as client:

        async def count(metric: RequestMetrics) -> RequestMetrics:
            async with semaphore:
                response = await client.post(
                    tokenizer_url,
                    json={"model": tokenizer_model, "prompt": metric.response_text},
                )
            response.raise_for_status()
            value = response.json().get("count")
            if type(value) is not int or value < 0:
                raise RuntimeError("public tokenizer returned an invalid token count")
            return replace(
                metric,
                public_completion_tokens=value,
                response_text="",
            )

        return list(await asyncio.gather(*(count(metric) for metric in results)))


def summarize_results(
    results: list[RequestMetrics],
    *,
    wall_ns: int,
    dataset_label: str,
    ttft_slo_s: float,
) -> tuple[dict, list[dict]]:
    """Build summary plus lossless per-request timing/usage evidence."""
    if wall_ns <= 0:
        raise ValueError(f"wall_ns must be positive, got {wall_ns}")
    wall_s = wall_ns / 1e9
    ttfts = sorted(metric.ttft_s for metric in results)
    totals = sorted(metric.total_s for metric in results)
    tpots = [metric.tpot_s for metric in results if metric.tpot_s > 0]
    reported_token_granular = all(
        metric.completion_tokens is not None for metric in results
    )
    public_token_granular = all(
        metric.public_completion_tokens is not None for metric in results
    )
    tpot_method = (
        "public-token"
        if public_token_granular
        else "token"
        if reported_token_granular
        else "chunk"
    )
    within_slo = sum(metric.ttft_s <= ttft_slo_s for metric in results)
    completion_tokens_total = (
        sum(metric.completion_tokens or 0 for metric in results)
        if reported_token_granular
        else None
    )
    public_completion_tokens_total = (
        sum(metric.public_completion_tokens or 0 for metric in results)
        if public_token_granular
        else None
    )

    def generation_rates(field: str, *, post_first: bool) -> list[float]:
        rates = []
        for metric in results:
            tokens = getattr(metric, field)
            duration = (
                metric.total_s - metric.ttft_s if post_first else metric.total_s
            )
            units = tokens - 1 if post_first and tokens is not None else tokens
            if units is not None and duration > 0 and units > 0:
                rates.append(units / duration)
        return rates

    public_generation_rates = generation_rates(
        "public_completion_tokens", post_first=True
    )
    reported_generation_rates = generation_rates(
        "completion_tokens", post_first=True
    )
    effective_generation_rates = (
        public_generation_rates
        if public_token_granular
        else reported_generation_rates
    )
    reported_e2e_rates = generation_rates(
        "completion_tokens", post_first=False
    )
    trace_counts = {
        status: sum(metric.trace_status == status for metric in results)
        for status in (
            "valid",
            "partial",
            "missing",
            "invalid",
            "not_requested",
        )
    }
    requested_traces = (
        trace_counts["valid"]
        + trace_counts["partial"]
        + trace_counts["missing"]
        + trace_counts["invalid"]
    )
    stage_values: dict[str, dict[str, object]] = {}
    if requested_traces:
        for stage_name in _NATIVE_STAGE_NAMES:
            stage_values[stage_name] = {
                "node": stage_name,
                "role": "engine",
                "kind": "stage",
                "expected_native": True,
                "events": 0,
                "request_indexes": set(),
                "occurrences_total": None,
                "usage_totals": None,
                "proposal_counts": set(),
                **{name: [] for name in _TRACE_DURATION_NAMES},
            }
    for request_index, metric in enumerate(results):
        if metric.trace_status not in {"valid", "partial"}:
            continue
        for stage in metric.trace_stages:
            aggregate = stage_values.setdefault(
                stage.stage,
                {
                    "node": stage.node,
                    "role": stage.role,
                    "kind": stage.kind,
                    "expected_native": False,
                    "events": 0,
                    "request_indexes": set(),
                    "occurrences_total": None,
                    "usage_totals": None,
                    "proposal_counts": set(),
                    **{name: [] for name in _TRACE_DURATION_NAMES},
                },
            )
            aggregate["events"] = int(aggregate["events"]) + 1
            request_indexes = aggregate["request_indexes"]
            assert isinstance(request_indexes, set)
            request_indexes.add(request_index)
            if stage.occurrences is not None:
                aggregate["occurrences_total"] = (
                    int(aggregate["occurrences_total"] or 0) + stage.occurrences
                )
            if stage.input_tokens is not None:
                usage_totals = aggregate["usage_totals"]
                if usage_totals is None:
                    usage_totals = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_tokens": 0,
                    }
                    aggregate["usage_totals"] = usage_totals
                assert isinstance(usage_totals, dict)
                usage_totals["input_tokens"] += stage.input_tokens
                usage_totals["output_tokens"] += stage.output_tokens or 0
                usage_totals["cached_tokens"] += stage.cached_tokens or 0
            if stage.proposals is not None:
                proposal_counts = aggregate["proposal_counts"]
                assert isinstance(proposal_counts, set)
                proposal_counts.add(stage.proposals)
            for name in _TRACE_DURATION_NAMES:
                value = getattr(stage, name)
                if value is not None:
                    values = aggregate[name]
                    assert isinstance(values, list)
                    values.append(value)

    stage_latency_ms: dict[str, dict[str, object]] = {}
    for stage_name, aggregate in sorted(stage_values.items()):
        request_indexes = aggregate["request_indexes"]
        assert isinstance(request_indexes, set)
        requests_observed = len(request_indexes)
        stage_summary: dict[str, object] = {
            "node": aggregate["node"],
            "role": aggregate["role"],
            "kind": aggregate["kind"],
            "expected_native": aggregate["expected_native"],
            "events": aggregate["events"],
            "requests_observed": requests_observed,
            "requests_missing": requested_traces - requests_observed,
            "coverage_fraction": (
                requests_observed / requested_traces if requested_traces else None
            ),
            "occurrences_total": aggregate["occurrences_total"],
            "usage_totals": aggregate["usage_totals"],
            "proposal_counts": sorted(aggregate["proposal_counts"]),
        }
        for name in _TRACE_DURATION_NAMES:
            values = sorted(aggregate[name])
            stage_summary[name.removesuffix("_ms")] = {
                "samples": len(values),
                "p50_ms": (
                    round(nearest_rank_percentile(values, 0.50), 3) if values else None
                ),
                "p99_ms": (
                    round(nearest_rank_percentile(values, 0.99), 3) if values else None
                ),
            }
        stage_latency_ms[stage_name] = stage_summary

    samples = [
        {
            "request_index": index,
            "ttft_ms": metric.ttft_s * 1e3,
            "total_ms": metric.total_s * 1e3,
            "tpot_ms": metric.tpot_s * 1e3,
            "output_chunks": metric.output_chunks,
            "completion_tokens": metric.completion_tokens,
            **(
                {"public_completion_tokens": metric.public_completion_tokens}
                if metric.public_completion_tokens is not None
                else {}
            ),
            "trace": {
                "status": metric.trace_status,
                "version": metric.trace_version,
                "stages": [stage.as_dict() for stage in metric.trace_stages],
            },
        }
        for index, metric in enumerate(results)
    ]
    summary = {
        "dataset": dataset_label,
        "percentile_method": PERCENTILE_METHOD,
        "requests": len(results),
        "wall_ns": wall_ns,
        "wall_s": wall_s,
        "ttft_p50_ms": round(nearest_rank_percentile(ttfts, 0.50) * 1e3, 2),
        "ttft_p99_ms": round(nearest_rank_percentile(ttfts, 0.99) * 1e3, 2),
        "e2e_p50_ms": round(nearest_rank_percentile(totals, 0.50) * 1e3, 2),
        "e2e_p99_ms": round(nearest_rank_percentile(totals, 0.99) * 1e3, 2),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1e3, 3) if tpots else None,
        "tpot_method": tpot_method,
        "throughput_rps": round(len(results) / wall_s, 2),
        "goodput_rps": round(within_slo / wall_s, 2),
        "completion_tokens_total": completion_tokens_total,
        "public_completion_tokens_total": public_completion_tokens_total,
        "output_tokens_per_s": (
            round(completion_tokens_total / wall_s, 2)
            if completion_tokens_total is not None
            else None
        ),
        "public_output_tokens_per_s": (
            round(public_completion_tokens_total / wall_s, 2)
            if public_completion_tokens_total is not None
            else None
        ),
        "generation_tokens_per_s_mean": (
            round(statistics.mean(effective_generation_rates), 3)
            if effective_generation_rates
            else None
        ),
        "generation_tokens_per_s_p50": (
            round(statistics.median(effective_generation_rates), 3)
            if effective_generation_rates
            else None
        ),
        "public_generation_tokens_per_s_mean": (
            round(statistics.mean(public_generation_rates), 3)
            if public_generation_rates
            else None
        ),
        "public_generation_tokens_per_s_p50": (
            round(statistics.median(public_generation_rates), 3)
            if public_generation_rates
            else None
        ),
        "reported_output_tokens_per_e2e_s_mean": (
            round(statistics.mean(reported_e2e_rates), 3)
            if reported_e2e_rates
            else None
        ),
        "success_rate": 1.0,
        "trace_coverage": {
            "total_requests": len(results),
            "requested": requested_traces,
            "not_requested": trace_counts["not_requested"],
            "valid": trace_counts["valid"],
            "partial": trace_counts["partial"],
            "missing": trace_counts["missing"],
            "invalid": trace_counts["invalid"],
            "fraction": (
                trace_counts["valid"] / requested_traces if requested_traces else None
            ),
        },
        "stage_latency_ms": stage_latency_ms,
    }
    return summary, samples


async def run_benchmark(args: argparse.Namespace) -> None:
    result_path, trace_path = serving_artifact_paths(args)
    target = resolve_target(args)
    api_key = resolve_api_key(args, target)
    if getattr(args, "api_key", None) is not None:
        print(
            "warning: --api-key is deprecated because command-line arguments may "
            "be visible to other processes; use --api-key-env or --target "
            f"{TARGET_SPEC_FORMAT}",
            file=sys.stderr,
        )
    if getattr(args, "profile", False):
        print(
            "warning: --profile records only the local serving_bench CPU driver; "
            "it excludes the target server and makes this run diagnostic, not "
            "timing-comparable",
            file=sys.stderr,
        )
    print(f"config={json.dumps(build_run_config(args))}")
    prompts, dataset_label = load_prompts(
        Path(args.dataset) if args.dataset else None, args.num_requests
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(
        base_url=normalize_base_url(target.base_url),
        timeout=args.timeout,
        headers=headers,
    ) as client:

        async def bounded(prompt: str) -> RequestMetrics:
            async with semaphore:
                return await run_one(
                    client,
                    target.model,
                    prompt,
                    args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    min_tokens=args.min_tokens,
                    ignore_eos=args.ignore_eos,
                    request_trace=getattr(args, "stage_trace", False),
                    capture_response=args.public_tokenizer_url is not None,
                )

        with profile_scope(
            enabled=getattr(args, "profile", False),
            activities=("cpu",),
            acc_events=True,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            with record_function_scope(
                profiler is not None,
                _CLIENT_PROFILE_RANGE,
            ):
                wall_start_ns = time.perf_counter_ns()
                results = await asyncio.gather(*(bounded(p) for p in prompts))
                wall_ns = time.perf_counter_ns() - wall_start_ns

    if args.public_tokenizer_url is not None:
        results = await _count_public_completion_tokens(
            results,
            tokenizer_url=args.public_tokenizer_url,
            tokenizer_model=args.public_tokenizer_model,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )

    summary, samples = summarize_results(
        results,
        wall_ns=wall_ns,
        dataset_label=dataset_label,
        ttft_slo_s=args.ttft_slo_s,
    )
    print(
        f"target={normalize_base_url(target.base_url)} "
        f"model={target.model} dataset={dataset_label}"
    )
    print(
        f"requests={len(results)} concurrency={args.concurrency} "
        f"wall={summary['wall_s']:.2f}s"
    )
    print(
        f"TTFT p50={summary['ttft_p50_ms']}ms p99={summary['ttft_p99_ms']}ms"
    )
    print(
        f"E2E p50={summary['e2e_p50_ms']}ms p99={summary['e2e_p99_ms']}ms"
    )
    if summary["tpot_mean_ms"] is not None:
        print(
            f"TPOT mean={summary['tpot_mean_ms']}ms/token "
            f"({summary['tpot_method']}-granularity)"
        )
    coverage = summary["trace_coverage"]
    if coverage["requested"]:
        print(
            "trace="
            f"{coverage['valid']}/{coverage['requested']} valid "
            f"({coverage['partial']} partial, {coverage['missing']} missing, "
            f"{coverage['invalid']} invalid)"
        )
    else:
        print(f"trace=not requested ({coverage['not_requested']} requests)")
    for stage_name, stage in summary["stage_latency_ms"].items():
        rendered = [
            "coverage="
            f"{stage['requests_observed']}/"
            f"{stage['requests_observed'] + stage['requests_missing']}"
        ]
        for duration in (
            "duration",
            "queue_wait",
            "service",
            "first_token",
            "post_first",
            "total",
        ):
            values = stage[duration]
            if values["samples"]:
                rendered.append(
                    f"{duration}=p50 {values['p50_ms']}ms/p99 {values['p99_ms']}ms"
                )
        if stage["occurrences_total"] is not None:
            rendered.append(f"occurrences={stage['occurrences_total']}")
        print(f"trace stage={stage_name} " + "; ".join(rendered))
    print(
        f"throughput={summary['throughput_rps']} req/s; "
        f"output={summary['output_tokens_per_s']} token/s; "
        f"public_output={summary['public_output_tokens_per_s']} token/s; "
        f"goodput(TTFT<={args.ttft_slo_s}s)={summary['goodput_rps']} req/s"
    )
    if result_path is not None:
        payload = {
            "config": build_run_config(args),
            "summary": summary,
            "samples": samples,
        }
        # Prove the measured payload is strict JSON before publishing either
        # member of the profile/result pair.
        json.dumps(payload, allow_nan=False)
        profile_artifact = None
        if profiler is not None:
            assert trace_path is not None and args.results_dir
            profile_artifact = write_trace_artifact(
                profiler,
                trace_path,
                artifact_root=Path(args.results_dir),
            )
        if profile_artifact is not None:
            payload["artifacts"] = {
                "torch_profile": {
                    "path": profile_artifact.path.name,
                    "format": profile_artifact.format,
                    "size_bytes": profile_artifact.size_bytes,
                    "sha256": profile_artifact.sha256,
                    "producer": "torch.profiler",
                    "scope": "local-client-process",
                    "activities": ["cpu"],
                    "range": _CLIENT_PROFILE_RANGE,
                    "target_process_included": False,
                    "diagnostic_only": True,
                    "timing_comparable": False,
                }
            }
        try:
            atomic_write_json(
                result_path,
                payload,
                allow_nan=False,
                overwrite=False,
            )
        except BaseException as error:
            if profile_artifact is not None:
                try:
                    remove_trace_artifact(profile_artifact)
                except BaseException as cleanup_error:
                    raise ExceptionGroup(
                        "serving result publication and trace rollback both failed",
                        [error, cleanup_error],
                    ) from error
            raise
        if profile_artifact is not None:
            print(f"client profile written to {profile_artifact.path}")
        print(f"results written to {result_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=None,
        help=f"{TARGET_SPEC_FORMAT}; cannot be combined with split endpoint flags",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None, help="ShareGPT-format JSON path")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Explicit sampling temperature (omitted by default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit per-request sampling seed (omitted by default)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Minimum completion length (omitted by default)",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force generation to the requested max_tokens",
    )
    parser.add_argument("--ttft-slo-s", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--stage-trace",
        action="store_true",
        help="request and report Kairyu structured stage timing (opt-in)",
    )
    parser.add_argument(
        "--public-tokenizer-url",
        default=None,
        help=(
            "Optional exact tokenizer endpoint for public assistant content; "
            "tokenization runs after and outside the timed serving interval"
        ),
    )
    parser.add_argument(
        "--public-tokenizer-model",
        default=None,
        help="Model name sent to --public-tokenizer-url (required with that option)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "write a CPU-only PyTorch Chrome trace for the local benchmark "
            "client (diagnostic; requires torch and --results-dir)"
        ),
    )
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the target bearer token",
    )
    auth.add_argument(
        "--api-key",
        default=None,
        help=(
            "Deprecated literal bearer token; may be visible in process arguments. "
            "Prefer --api-key-env"
        ),
    )
    parser.add_argument("--results-dir", default="bench/results",
                        help="Write a timestamped results JSON here ('' to disable)")
    # M5 topology labels (design m5 D6); recorded in the run config for G2 §8.
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="TP degree of the target server (results label)")
    parser.add_argument("--dp-replicas", type=int, default=1,
                        help="DP replica count behind the target (results label)")
    parser.add_argument("--pd", action="store_true",
                        help="target runs prefill-decode disaggregated (results label)")
    return parser


def main() -> None:
    asyncio.run(run_benchmark(build_parser().parse_args()))


if __name__ == "__main__":
    main()
