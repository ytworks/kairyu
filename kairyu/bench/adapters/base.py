"""Adapter contract for installed suites + the shared generative run loop.

An adapter owns one scoreboard row: how its dataset is downloaded and
normalized, how an item becomes an OpenAI-wire request, and how a response
is scored. The runner owns everything else (targets, resume, aggregation).
Degradation is data, not control flow: every unmet precondition becomes a
PairResult(status="skipped", reason=...) so one command always completes.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import random
import re
import shutil
import sysconfig
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

from kairyu.bench.cache import BenchCache
from kairyu.bench.progress import NullProgress
from kairyu.bench.sampling import (
    combine_attempts,
    sampling_metrics,
    sampling_summary,
    seed_schedule,
)
from kairyu.bench.streaming import ChatSSEAccumulator, StreamingProtocolError
from kairyu.bench.targets import normalize_base_url, target_api_key
from kairyu.bench.types import (
    JSON_SAFE_INTEGER_MAX,
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    ContinuationLogLikelihood,
    DownloadReport,
    GenerationTimingEvidence,
    ItemResult,
    LogLikelihoodRequestSpec,
    LogLikelihoodResponse,
    PairResult,
    SamplingAttemptResult,
    SkipItem,
)

if TYPE_CHECKING:  # judge lands in its own module; adapters only see the protocol
    from kairyu.bench.execution import ExecutionRunner
    from kairyu.bench.judge import JudgeClient, MajorityJudgeClient
    from kairyu.bench.progress import ProgressReporter

_EXCERPT_CHARS = 2000
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class AdapterInfo:
    name: str  # registry key / directory name, kebab-case
    display_name: str  # scoreboard row label
    metric: str  # human name of the score ("accuracy", "pass@1", ...)
    hf_dataset: str | None = None
    hf_revision: str | None = None
    # Secondary sources this slot's rows depend on — testcase archives, golden
    # data — as (identifier, pin). They decide the tests and expected answers, so
    # they belong in cache invalidation and provenance next to the main dataset:
    # otherwise repinning one leaves a stale cache "ready" while the methodology
    # advertises the new pin.
    extra_sources: tuple[tuple[str, str], ...] = ()
    gated: bool = False
    needs_docker: bool = False
    needs_execution: bool = False
    needs_vision: bool = False
    judge_preferred: bool = False
    agentic: bool = False
    annotations: tuple[str, ...] = ()  # permanent footnotes (substitute slots)
    # False for a slot that measures a different thing than the published row
    # (a substituted dataset). Run-time reasons are added per pair.
    comparable_to_published: bool = True
    incomparable_reason: str = ""
    # True only when every scored item is contractually a Bernoulli outcome.
    # Observing only 0/1 values in one continuous/reward run is not enough.
    binary_outcomes: bool = False
    # Optional versioned grouping rule for paired uncertainty. Items inside one
    # group may be dependent and must be resampled together for config A/B.
    paired_cluster_key: str | None = None
    # Logical name in judge_prompts.judge_templates(). This binds the exact
    # scoring prompt to runs selecting this adapter without invalidating
    # unrelated benchmark runs.
    judge_template_name: str | None = None
    # Package resources that implement score-bearing prompt/parser/checker
    # semantics. Their content digests enter the run fingerprint so a resumed
    # run can never reuse pair evidence produced by different evaluator code.
    evaluation_resources: tuple[tuple[str, str], ...] = ()
    # Third-party harness distributions whose resolved implementation produces
    # the score.  Their version and installed content enter the fingerprint.
    evaluation_distributions: tuple[str, ...] = ()
    # PATH entry points that launch an external harness.  History requires each
    # resolved executable to be owned by a declared distribution.
    evaluation_executables: tuple[str, ...] = ()
    # False when runtime inputs cannot yet be resolved to immutable content
    # (for example a harness-managed remote task/image set). The run and cell
    # remain in history, but that cell is structurally withheld from deltas.
    history_provenance_complete: bool = True
    history_provenance_reason: str = ""

    def required_judge_template_name(self) -> str:
        if self.judge_template_name is None:
            raise RuntimeError(f"adapter {self.name!r} has no judge template")
        return self.judge_template_name


@dataclass(frozen=True)
class DownloadContext:
    cache: BenchCache
    force: bool = False


def _default_execution_runner() -> ExecutionRunner:
    """Keep directly constructed test/development contexts on the local runner."""
    from kairyu.bench.execution import build_execution_runner
    from kairyu.bench.types import ExecutionConfig

    return build_execution_runner(ExecutionConfig())


@dataclass
class RunContext:
    """Built once per suite run and shared by every (benchmark, target) pair."""

    cache: BenchCache
    http_factory: Callable[[], httpx.AsyncClient]
    judge: JudgeClient | MajorityJudgeClient | None = None
    limit: int | None = None
    seed: int = 0
    # Trials per source item: chat adapters retain a seed sweep; external
    # agentic adapters continue mapping this to their harness trial flag.
    attempts: int = 1
    run_id: str = ""  # provenance for artifacts the external harnesses name themselves
    # True means the caller explicitly requested fresh external-harness evidence.
    # Failed-pair resume keeps this false so costly partial artifacts can be reused.
    rerun: bool = False
    # Persistent root for expensive external-harness evidence. SuiteRunner
    # binds this below the immutable run directory; direct tests may override it.
    artifacts_dir: Path | None = None
    concurrency: int = 16
    retries: int = 2
    request_timeout_s: float = 600.0
    offline_fixtures: bool = False
    smoke: bool = False
    docker: tuple[bool, str] = (False, "docker not probed")
    execution_runner: ExecutionRunner = field(default_factory=_default_execution_runner)
    exec_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))
    download_failures: dict[str, str] = field(default_factory=dict)
    # Observer for live output. Defaults to silence so programmatic use and
    # tests behave exactly as before.
    progress: ProgressReporter = field(default_factory=NullProgress)


@runtime_checkable
class BenchmarkAdapter(Protocol):
    info: AdapterInfo

    def download(self, ctx: DownloadContext) -> DownloadReport: ...

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult: ...


class RequestFailed(RuntimeError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body


class LogLikelihoodUnsupported(RequestFailed):
    """A successful ordinary completion proved the extension was not honored."""


class LogLikelihoodSchemaError(RequestFailed):
    """The extension claimed support but returned unusable score evidence."""


def cache_pins(info: AdapterInfo) -> dict:
    """The pins every readiness check and identity record must agree on."""
    return {
        "dataset": info.hf_dataset,
        "revision": info.hf_revision,
        "sources": [list(source) for source in info.extra_sources],
    }


def _read_evaluation_resource(package: str, resource: str) -> bytes:
    """Read a packaged score-bearing resource through the wheel-safe API."""

    from importlib import resources

    return resources.files(package).joinpath(resource).read_bytes()


def _evaluation_distribution_identity(name: str) -> dict:
    """Return a path-free content identity for one installed harness."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "installed": False}
    try:
        files = distribution.files
        if files is None:
            raise OSError("distribution has no installed-file manifest")
        digest = hashlib.sha256()
        digest.update(b"kairyu-bench-harness-distribution-v1\0")
        count = 0
        for relative in sorted(files, key=str):
            relative_name = str(relative).replace("\\", "/")
            if "__pycache__" in relative_name or relative_name.endswith((".pyc", ".pyo")):
                continue
            basename = relative_name.rsplit("/", 1)[-1]
            if relative_name.endswith(".egg-link") or (
                relative_name.endswith(".pth")
                and basename.startswith(("__editable__.", "_editable_impl_"))
            ):
                raise OSError("editable distribution source is not content-bound")
            path = distribution.locate_file(relative)
            if not path.is_file():
                continue
            payload = path.read_bytes()
            if relative_name.endswith(".dist-info/direct_url.json"):
                try:
                    direct_url = json.loads(payload.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise OSError("distribution has malformed direct_url metadata") from error
                directory = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
                if isinstance(directory, dict) and directory.get("editable") is True:
                    raise OSError("editable distribution source is not content-bound")
            encoded_name = relative_name.encode("utf-8", errors="strict")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            count += 1
        if count == 0:
            raise OSError("distribution has no readable installed files")
    except (OSError, UnicodeError) as error:
        return {
            "name": name,
            "installed": True,
            "version": distribution.version,
            "content_verified": False,
            "detail": type(error).__name__,
        }
    return {
        "name": name,
        "installed": True,
        "version": distribution.version,
        "content_verified": True,
        "file_count": count,
        "content_sha256": digest.hexdigest(),
    }


def _evaluation_executable_identity(name: str, distributions: tuple[str, ...]) -> dict:
    resolved_text = shutil.which(name)
    if resolved_text is None:
        return {"name": name, "found": False}
    try:
        resolved = Path(resolved_text).resolve(strict=True)
        payload = resolved.read_bytes()
    except OSError as error:
        return {
            "name": name,
            "found": True,
            "content_verified": False,
            "detail": type(error).__name__,
        }

    owners: list[str] = []
    expected_script = Path(sysconfig.get_path("scripts")) / name
    for distribution_name in distributions:
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        declares_script = any(
            entry.group == "console_scripts" and entry.name == name
            for entry in distribution.entry_points
        )
        if not declares_script:
            continue
        try:
            if expected_script.resolve(strict=True) == resolved:
                owners.append(distribution_name)
        except OSError:
            continue
    return {
        "name": name,
        "found": True,
        "content_verified": True,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "owned_by_distribution": owners,
    }


def evaluation_protocol_identity(adapter: BenchmarkAdapter) -> dict:
    """Bind every built-in adapter to the code that can change its score.

    ``AdapterInfo.evaluation_resources`` remains the escape hatch for vendored
    checkers and other non-adapter resources.  The adapter implementation and
    shared request/scoring machinery are discovered automatically, so adding a
    new benchmark cannot silently omit its prompt/parser/scorer module from the
    cross-commit methodology fingerprint.
    """

    info = adapter.info
    resources: list[tuple[str, str]] = []
    for implementation in type(adapter).__mro__:
        module = implementation.__module__
        if not module.startswith("kairyu.bench.adapters."):
            continue
        package, _, module_name = module.rpartition(".")
        resources.append((package, f"{module_name}.py"))

    resources.extend(
        (
            ("kairyu.bench.adapters", "base.py"),
            ("kairyu.bench", "aggregate.py"),
            ("kairyu.bench", "cache.py"),
            ("kairyu.bench", "runner.py"),
            ("kairyu.bench", "sampling.py"),
            ("kairyu.bench", "streaming.py"),
            ("kairyu.bench", "types.py"),
            ("kairyu.bench", "targets.py"),
        )
    )
    if info.needs_execution:
        resources.extend(
            (
                ("kairyu.bench", "execution.py"),
                ("kairyu.bench", "sandbox.py"),
            )
        )
    if info.judge_template_name is not None:
        resources.extend(
            (
                ("kairyu.bench", "judge.py"),
                ("kairyu.bench", "judge_prompts.py"),
            )
        )
    resources.extend(info.evaluation_resources)

    ordered_resources = tuple(dict.fromkeys(resources))
    return {
        "resources": [
            {
                "package": package,
                "resource": resource,
                "sha256": hashlib.sha256(_read_evaluation_resource(package, resource)).hexdigest(),
            }
            for package, resource in ordered_resources
        ],
        "distributions": [
            _evaluation_distribution_identity(name) for name in info.evaluation_distributions
        ],
        "executables": [
            _evaluation_executable_identity(name, info.evaluation_distributions)
            for name in info.evaluation_executables
        ],
    }


def adapter_cache_ready(adapter: BenchmarkAdapter, cache: BenchCache) -> bool:
    """Return whether an adapter's rows and adapter-owned assets are usable.

    Most adapters only own normalized rows. Adapters with additional executable
    assets may expose ``additional_cache_ready(cache)``; every cache consumer
    goes through this helper so download, list, identity, resume, and runtime
    preconditions cannot disagree about readiness.
    """

    if not cache.is_ready(adapter.info.name, **cache_pins(adapter.info)):
        return False
    additional_ready = getattr(adapter, "additional_cache_ready", None)
    if additional_ready is None:
        return True
    try:
        return bool(additional_ready(cache))
    except Exception:
        # Readiness is a trust boundary: a broken validator must never make a
        # cache look usable.
        return False


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def estimate_tokens(text: str) -> int:
    """Chars/4 heuristic — recorded in methodology wherever it gates items."""
    return len(text) // 4 + 1


def mcq_prompt(question: str, choices: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question, "", "Choices:"]
    lines += [f"{letter}) {choice}" for letter, choice in zip(letters, choices, strict=True)]
    lines += [
        "",
        "Answer with the single letter of the correct choice. "
        'End your reply with "Answer: <letter>".',
    ]
    return "\n".join(lines)


def shuffle_choices(
    seed: int, item_id: str, correct: str, incorrect: list[str]
) -> tuple[list[str], str]:
    """Deterministic per-item choice order; returns (choices, correct letter).

    Pure function of (seed, item_id) so build_request and score agree without
    threading state through the payload.
    """
    choices = [correct, *incorrect]
    rng = random.Random(f"{seed}:{item_id}")
    rng.shuffle(choices)
    return choices, chr(ord("A") + choices.index(correct))


# The answer letter must be a bounded token right after the marker, NOT the
# first letter of the following word — `(?![A-Za-z])` stops "answer depends"
# from extracting "D" (B1). Every gap piece stays optional so "Answer: B",
# "the answer is (C)", and "answer**D**" all still match.
_ANSWER_RE = re.compile(r"(?i)\banswer\b\s*(?:is\b\s*)?:?\s*\**\(?([A-Z])\)?(?![A-Za-z])")


def extract_choice_letter(text: str, num_choices: int = 4) -> str | None:
    """Last 'Answer: X' style marker, else the last standalone UPPERCASE letter."""
    valid = {chr(ord("A") + i) for i in range(num_choices)}
    markers = [m.group(1).upper() for m in _ANSWER_RE.finditer(text)]
    for letter in reversed(markers):
        if letter in valid:
            return letter
    # uppercase-only fallback: a lone "a"/"i" (article/pronoun) is not an answer
    standalone = re.findall(r"\b([A-Z])\b", text)
    for letter in reversed(standalone):
        if letter in valid:
            return letter
    return None


_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    """Last fenced code block (models often restate; the last is the answer)."""
    blocks = _CODE_BLOCK_RE.findall(text)
    return blocks[-1].strip() if blocks else None


def excerpt(text: str | None) -> str | None:
    if text is None:
        return None
    return text[:_EXCERPT_CHARS]


def chat_request_body(
    target: BenchTarget,
    request: ChatRequestSpec,
    *,
    sampling_seed: int | None = None,
    response_format: dict | None = None,
    stream: bool = False,
) -> dict:
    """Build the shared chat body while preserving the legacy key order."""

    body = {
        "model": target.model,
        "messages": list(request.messages),
    }
    if target.sampling_mode == "adapter":
        body["temperature"] = (
            request.temperature if target.temperature is None else target.temperature
        )
    body["stream"] = stream
    if stream:
        body["stream_options"] = {"include_usage": True}
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    body.update(target.wire_overrides())
    if sampling_seed is not None:
        body["seed"] = sampling_seed
    if response_format is not None:
        body["response_format"] = response_format
    return body


@dataclass(frozen=True)
class ChatCompletionEvidence:
    """Strict standard fields retained by evidence-bearing chat adapters."""

    content: str | None
    refusal: str | None
    message_auxiliary: str | None
    finish_reason: str
    usage: tuple[int, int, int] | None
    timing: GenerationTimingEvidence | None = None


def _invalid_chat_response(message: str) -> RequestFailed:
    return RequestFailed(200, f"invalid chat completion response: {message}")


def _strict_chat_response(response: httpx.Response) -> ChatCompletionEvidence:
    raw = response.content
    if len(raw) > 1_048_576:
        raise _invalid_chat_response("HTTP body exceeds 1048576 bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", errors="strict")
        data = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (RecursionError, UnicodeError, ValueError) as error:
        raise _invalid_chat_response(f"strict JSON parse failed: {error}") from error
    stack = [(data, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > 256:
            raise _invalid_chat_response("JSON nesting exceeds 256 levels")
        if isinstance(value, dict):
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise _invalid_chat_response("JSON contains a non-finite number")
    if not isinstance(data, dict):
        raise _invalid_chat_response("top-level value is not an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise _invalid_chat_response("choices must contain exactly one item")
    choice = choices[0]
    if not isinstance(choice, dict) or type(choice.get("index")) is not int:
        raise _invalid_chat_response("choice must carry integer index 0")
    if choice["index"] != 0:
        raise _invalid_chat_response("choice index must be 0")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _invalid_chat_response("choice message must be an object")
    content_value = message.get("content")
    refusal_value = message.get("refusal")
    if content_value is not None and not isinstance(content_value, str):
        raise _invalid_chat_response("choice message content must be a string or null")
    if refusal_value is not None and (
        not isinstance(refusal_value, str) or not refusal_value
    ):
        raise _invalid_chat_response("choice message refusal must be null or non-empty text")
    auxiliary_value: dict[str, object] = {}
    if message.get("tool_calls") is not None:
        if not isinstance(message["tool_calls"], list):
            raise _invalid_chat_response("choice message tool_calls must be a list")
        if message["tool_calls"]:
            auxiliary_value["tool_calls"] = message["tool_calls"]
    if message.get("function_call") is not None:
        if not isinstance(message["function_call"], dict) or not message["function_call"]:
            raise _invalid_chat_response(
                "choice message function_call must be a non-empty object"
            )
        auxiliary_value["function_call"] = message["function_call"]
    if refusal_value is not None and (
        content_value is not None or auxiliary_value
    ):
        raise _invalid_chat_response(
            "choice refusal cannot coexist with content or call payloads"
        )
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise _invalid_chat_response("choice finish_reason must be a non-empty string")
    content = content_value
    refusal = refusal_value
    message_auxiliary = (
        json.dumps(
            auxiliary_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if auxiliary_value
        else None
    )
    if content is not None and len(content) > 262_144:
        raise _invalid_chat_response("choice content exceeds 262144 characters")
    if refusal is not None and len(refusal) > 262_144:
        raise _invalid_chat_response("choice refusal exceeds 262144 characters")
    if message_auxiliary is not None and len(message_auxiliary) > 262_144:
        raise _invalid_chat_response("choice call payload exceeds 262144 characters")

    usage_value = data.get("usage")
    usage: tuple[int, int, int] | None = None
    if usage_value is not None:
        if not isinstance(usage_value, dict):
            raise _invalid_chat_response("usage must be an object or null")
        names = ("prompt_tokens", "completion_tokens", "total_tokens")
        counts = tuple(usage_value.get(name) for name in names)
        if any(
            type(value) is not int or not 0 <= value <= JSON_SAFE_INTEGER_MAX
            for value in counts
        ):
            raise _invalid_chat_response(
                "usage token counts must be JSON-safe non-negative integers"
            )
        prompt_tokens, completion_tokens, total_tokens = counts
        if total_tokens != prompt_tokens + completion_tokens:
            raise _invalid_chat_response("usage total does not equal prompt plus completion")
        usage = (prompt_tokens, completion_tokens, total_tokens)
    return ChatCompletionEvidence(
        content=content,
        refusal=refusal,
        message_auxiliary=message_auxiliary,
        finish_reason=finish_reason,
        usage=usage,
        timing=None,
    )


def _permissive_chat_response(response: httpx.Response) -> ChatCompletionEvidence:
    """Compatibility fallback for JSON-only test and legacy endpoints.

    A JSON response to a request carrying ``stream=true`` cannot yield TTFT/TPS,
    so the returned timing is explicitly absent. Structured-output evidence uses
    the strict fallback instead.
    """

    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise _invalid_chat_response(f"JSON fallback parse failed: {error}") from error
    if content is not None and not isinstance(content, str):
        raise _invalid_chat_response("JSON fallback content must be text or null")
    return ChatCompletionEvidence(
        content=content,
        refusal=None,
        message_auxiliary=None,
        finish_reason=choice.get("finish_reason") or "stop",
        usage=None,
        timing=None,
    )


async def _call_streaming_chat_evidence(
    client: httpx.AsyncClient,
    target: BenchTarget,
    request: ChatRequestSpec,
    *,
    response_format: dict | None,
    retries: int,
    timeout_s: float,
    api_key: str | None,
    sampling_seed: int | None,
    strict_json_fallback: bool,
) -> ChatCompletionEvidence:
    url = f"{normalize_base_url(target.base_url)}/chat/completions"
    body = chat_request_body(
        target,
        request,
        sampling_seed=sampling_seed,
        response_format=response_format,
        stream=True,
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    attempt = 0
    while True:
        started = time.perf_counter()
        try:
            async with client.stream(
                "POST", url, json=body, headers=headers, timeout=timeout_s
            ) as response:
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/event-stream" not in content_type:
                        await response.aread()
                        return (
                            _strict_chat_response(response)
                            if strict_json_fallback
                            else _permissive_chat_response(response)
                        )
                    accumulator = ChatSSEAccumulator(request_attempts=attempt + 1)
                    try:
                        async for line in response.aiter_lines():
                            accumulator.feed_line(line, time.perf_counter() - started)
                        streamed = accumulator.finish()
                    except StreamingProtocolError as error:
                        raise _invalid_chat_response(str(error)) from error
                    return ChatCompletionEvidence(
                        content=streamed.content,
                        refusal=streamed.refusal,
                        message_auxiliary=streamed.message_auxiliary,
                        finish_reason=streamed.finish_reason,
                        usage=streamed.usage,
                        timing=streamed.timing,
                    )
                await response.aread()
                if response.status_code in _RETRYABLE_STATUS and attempt < retries:
                    await asyncio.sleep(0.5 * 2**attempt)
                    attempt += 1
                    continue
                raise RequestFailed(response.status_code, response.text)
        except RequestFailed:
            raise
        except httpx.HTTPError as error:
            if attempt >= retries:
                raise RequestFailed(0, f"transport error: {type(error).__name__}") from error
            await asyncio.sleep(0.5 * 2**attempt)
            attempt += 1


async def call_chat_evidence(
    client: httpx.AsyncClient,
    target: BenchTarget,
    request: ChatRequestSpec,
    *,
    response_format: dict | None,
    retries: int,
    timeout_s: float,
    api_key: str | None = None,
    sampling_seed: int | None = None,
) -> ChatCompletionEvidence:
    """Stream Chat Completions and validate the evidence-bearing envelope."""

    return await _call_streaming_chat_evidence(
        client,
        target,
        request,
        response_format=response_format,
        retries=retries,
        timeout_s=timeout_s,
        api_key=api_key,
        sampling_seed=sampling_seed,
        strict_json_fallback=True,
    )


async def call_chat(
    client: httpx.AsyncClient,
    target: BenchTarget,
    request: ChatRequestSpec,
    *,
    retries: int,
    timeout_s: float,
    api_key: str | None = None,
    sampling_seed: int | None = None,
) -> str:
    """Non-streaming POST /v1/chat/completions with backoff on 429/5xx/timeouts."""
    url = f"{normalize_base_url(target.base_url)}/chat/completions"
    body = chat_request_body(target, request, sampling_seed=sampling_seed)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    attempt = 0
    while True:
        try:
            response = await client.post(url, json=body, headers=headers, timeout=timeout_s)
        except httpx.HTTPError as error:
            if attempt >= retries:
                raise RequestFailed(0, f"transport error: {error}") from error
            await asyncio.sleep(0.5 * 2**attempt)
            attempt += 1
            continue
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content or ""
        if response.status_code in _RETRYABLE_STATUS and attempt < retries:
            await asyncio.sleep(0.5 * 2**attempt)
            attempt += 1
            continue
        raise RequestFailed(response.status_code, response.text)


def external_harness_sampling_incompatibility(
    target: BenchTarget,
    *,
    harness: str,
) -> str | None:
    """Reject chat-only sampling controls at external harness boundaries.

    These wrappers do not construct the OpenAI request body themselves.  An
    explicit temperature or generation-default omission therefore cannot be
    claimed unless the pinned harness exposes a verified equivalent.
    """

    if target.sampling_mode == "recommended":
        return (
            f"{harness} has no verified generation-default omission passthrough; "
            "sampling_mode='recommended' is supported only by chat adapters"
        )
    if target.temperature is not None:
        return (
            f"{harness} has no verified temperature passthrough; explicit target "
            "temperature is supported only by chat adapters"
        )
    return None


def _invalid_loglikelihood_response(message: str) -> RequestFailed:
    return LogLikelihoodSchemaError(200, f"invalid loglikelihood response: {message}")


def _parse_loglikelihood_choice(
    data: object,
    continuation: str,
    reduction: str,
) -> tuple[ContinuationLogLikelihood, tuple[int, ...]]:
    """Validate one extension response before any score can consume it."""

    if not isinstance(data, dict):
        raise _invalid_loglikelihood_response("top level must be a JSON object")
    if data.get("mode") != "loglikelihood":
        raise LogLikelihoodUnsupported(
            200,
            "response did not identify mode='loglikelihood'; target did not honor "
            "kairyu_continuation",
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise _invalid_loglikelihood_response("choices must contain exactly one entry")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise _invalid_loglikelihood_response("choice must be a JSON object")
    if type(choice.get("index")) is not int or choice["index"] != 0:
        raise _invalid_loglikelihood_response("choice.index must be integer 0")
    if choice.get("finish_reason") != "length":
        raise _invalid_loglikelihood_response(
            "choice.finish_reason must be 'length' for forced continuation scoring"
        )
    if choice.get("text") != continuation:
        raise _invalid_loglikelihood_response(
            "choice.text does not exactly match the requested continuation"
        )

    token_ids = choice.get("continuation_token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
    ):
        raise _invalid_loglikelihood_response(
            "continuation_token_ids must be a non-empty list of non-negative integers"
        )
    prompt_token_ids = choice.get("prompt_token_ids")
    if (
        not isinstance(prompt_token_ids, list)
        or not prompt_token_ids
        or any(type(token_id) is not int or token_id < 0 for token_id in prompt_token_ids)
    ):
        raise _invalid_loglikelihood_response(
            "prompt_token_ids must be a non-empty list of non-negative integers"
        )

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise _invalid_loglikelihood_response("choice.logprobs must be a JSON object")
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    text_offsets = logprobs.get("text_offset")
    length = len(token_ids)
    if (
        not isinstance(tokens, list)
        or len(tokens) != length
        or any(not isinstance(token, str) for token in tokens)
    ):
        raise _invalid_loglikelihood_response(
            "logprobs.tokens must be a string list matching continuation_token_ids"
        )
    if not isinstance(token_logprobs, list) or len(token_logprobs) != length:
        raise _invalid_loglikelihood_response(
            "logprobs.token_logprobs must match continuation_token_ids"
        )
    parsed_logprobs: list[float] = []
    for value in token_logprobs:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid_loglikelihood_response("token_logprobs must contain only finite numbers")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed > 0.0:
            raise _invalid_loglikelihood_response(
                "token_logprobs must contain only finite values <= 0"
            )
        parsed_logprobs.append(parsed)
    if (
        not isinstance(text_offsets, list)
        or len(text_offsets) != length
        or any(type(offset) is not int for offset in text_offsets)
    ):
        raise _invalid_loglikelihood_response(
            "logprobs.text_offset must be an integer list matching continuation_token_ids"
        )

    total = sum(parsed_logprobs)
    score = total if reduction == "sum" else total / length
    if not math.isfinite(total) or not math.isfinite(score):
        raise _invalid_loglikelihood_response("reduced token_logprobs must remain finite")
    return (
        ContinuationLogLikelihood(
            continuation=continuation,
            token_ids=tuple(token_ids),
            tokens=tuple(tokens),
            token_logprobs=tuple(parsed_logprobs),
            text_offsets=tuple(text_offsets),
            sum_logprob=total,
            score=score,
        ),
        tuple(prompt_token_ids),
    )


async def call_loglikelihood(
    client: httpx.AsyncClient,
    target: BenchTarget,
    request: LogLikelihoodRequestSpec,
    *,
    retries: int,
    timeout_s: float,
    api_key: str | None = None,
) -> LogLikelihoodResponse:
    """Score ordered continuations through Kairyu's teacher-forced extension.

    One non-streaming ``/v1/completions`` request is made per continuation.
    Target sampling overrides deliberately do not apply: this protocol fixes
    temperature and all score-bearing fields and records the exact selected
    token log probabilities returned by the backend.
    """

    url = f"{normalize_base_url(target.base_url)}/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    candidates: list[ContinuationLogLikelihood] = []
    common_prompt_token_ids: tuple[int, ...] | None = None
    for continuation in request.continuations:
        body = {
            "model": target.model,
            "prompt": request.context,
            "kairyu_continuation": continuation,
            "max_tokens": None,
            "temperature": 0.0,
            "stream": False,
            "logprobs": 0,
        }
        attempt = 0
        while True:
            try:
                response = await client.post(url, json=body, headers=headers, timeout=timeout_s)
            except httpx.HTTPError as error:
                if attempt >= retries:
                    raise RequestFailed(0, f"transport error: {error}") from error
                await asyncio.sleep(0.5 * 2**attempt)
                attempt += 1
                continue
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError as error:
                    raise _invalid_loglikelihood_response(
                        f"body is not valid JSON: {error}"
                    ) from error
                candidate, prompt_token_ids = _parse_loglikelihood_choice(
                    data, continuation, request.reduction
                )
                if common_prompt_token_ids is None:
                    common_prompt_token_ids = prompt_token_ids
                elif prompt_token_ids != common_prompt_token_ids:
                    raise _invalid_loglikelihood_response(
                        "prompt_token_ids changed across ordered candidates"
                    )
                candidates.append(candidate)
                break
            if response.status_code in _RETRYABLE_STATUS and attempt < retries:
                await asyncio.sleep(0.5 * 2**attempt)
                attempt += 1
                continue
            raise RequestFailed(response.status_code, response.text)
    return LogLikelihoodResponse(
        reduction=request.reduction,
        prompt_token_ids=common_prompt_token_ids or (),
        candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class ItemAttempt:
    """One backend call: either text (+latency) or a terminal ItemResult."""

    text: str | None = None
    latency_s: float = 0.0
    timing: GenerationTimingEvidence | None = None
    failure: ItemResult | None = None


@dataclass(frozen=True)
class LogLikelihoodAttempt:
    """One ranked-candidate call, a terminal item, or a capability verdict."""

    response: LogLikelihoodResponse | None = None
    latency_s: float = 0.0
    failure: ItemResult | None = None
    unsupported_reason: str | None = None
    fatal_reason: str | None = None


async def attempt_item(
    client: httpx.AsyncClient,
    target: BenchTarget,
    request: ChatRequestSpec,
    *,
    item_id: str,
    ctx: RunContext,
    api_key: str | None,
    sampling_seed: int | None = None,
) -> ItemAttempt:
    """Call the target for one item, converting request failures into data.

    A 400 that names an image/context problem is the target refusing input it
    cannot represent, so the item is `skipped`; anything else is `failed`.
    """
    start = time.perf_counter()
    try:
        completion = await _call_streaming_chat_evidence(
            client,
            target,
            request,
            response_format=None,
            retries=ctx.retries,
            timeout_s=ctx.request_timeout_s,
            api_key=api_key,
            sampling_seed=sampling_seed,
            strict_json_fallback=False,
        )
    except RequestFailed as error:
        lowered = error.body.lower()
        unrepresentable = error.status_code == 400 and (
            "image" in lowered or "context" in lowered or "too long" in lowered
        )
        latency_s = round(time.perf_counter() - start, 3)
        return ItemAttempt(
            failure=ItemResult(
                item_id=item_id,
                status="skipped" if unrepresentable else "failed",
                error=str(error),
                latency_s=latency_s,
            )
        )
    latency_s = time.perf_counter() - start
    text = completion.content or ""
    if not text.strip():
        return ItemAttempt(
            failure=ItemResult(
                item_id=item_id,
                status="failed",
                error="empty completion: target returned no response content",
                response_excerpt=excerpt(text),
                latency_s=round(latency_s, 3),
            )
        )
    return ItemAttempt(text=text, latency_s=latency_s, timing=completion.timing)


def _explicitly_unsupported_loglikelihood(error: RequestFailed) -> bool:
    """Recognize only an explicit extension rejection, never a generic 400."""

    if error.status_code not in {400, 422}:
        return False
    body = error.body.lower()
    return "kairyu_continuation" in body and (
        "unsupported" in body
        or "not supported" in body
        or "unknown parameter" in body
        or "unknown field" in body
        or "unknown argument" in body
        or "unrecognized field" in body
        or "unrecognized request argument" in body
        or "extra_forbidden" in body
        or "extra inputs are not permitted" in body
    )


def _fatal_probe_failure(error: RequestFailed) -> str | None:
    if isinstance(error, LogLikelihoodSchemaError):
        return "target returned malformed log-likelihood evidence during capability probe"
    if error.status_code == 0:
        return "target transport failed after capability-probe retries"
    if error.status_code in {401, 403}:
        return "target authentication or authorization failed during capability probe"
    if error.status_code == 404:
        return "target completion endpoint or model was not found during capability probe"
    if error.status_code in _RETRYABLE_STATUS:
        return "target remained unavailable after capability-probe retries"
    body = error.body.lower()
    if (
        error.status_code == 400
        and "kairyu_continuation" in body
        and "does not align with a model token boundary" in body
    ):
        return "target tokenizer rejected the fixed continuation boundary"
    return None


async def attempt_loglikelihood(
    client: httpx.AsyncClient,
    target: BenchTarget,
    request: LogLikelihoodRequestSpec,
    *,
    item_id: str,
    ctx: RunContext,
    api_key: str | None,
    capability_probe: bool = False,
) -> LogLikelihoodAttempt:
    """Convert wire/schema failures to evidence without manufacturing a score."""

    start = time.perf_counter()
    try:
        response = await call_loglikelihood(
            client,
            target,
            request,
            retries=ctx.retries,
            timeout_s=ctx.request_timeout_s,
            api_key=api_key,
        )
    except RequestFailed as error:
        latency_s = round(time.perf_counter() - start, 3)
        if capability_probe and (
            isinstance(error, LogLikelihoodUnsupported)
            or _explicitly_unsupported_loglikelihood(error)
        ):
            return LogLikelihoodAttempt(
                latency_s=latency_s,
                unsupported_reason=(
                    "target does not support the required kairyu_continuation "
                    "teacher-forced log-likelihood extension"
                ),
            )
        if capability_probe and (fatal_reason := _fatal_probe_failure(error)):
            return LogLikelihoodAttempt(
                latency_s=latency_s,
                failure=ItemResult(
                    item_id=item_id,
                    status="failed",
                    error=str(error),
                    latency_s=latency_s,
                ),
                fatal_reason=fatal_reason,
            )
        return LogLikelihoodAttempt(
            failure=ItemResult(
                item_id=item_id,
                status="failed",
                error=str(error),
                latency_s=latency_s,
            )
        )
    return LogLikelihoodAttempt(
        response=response,
        latency_s=time.perf_counter() - start,
    )


def select_items(items: list[BenchItem], limit: int | None, seed: int) -> list[BenchItem]:
    """Deterministic subset: seeded sample keeps subsets comparable across runs."""
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(seed)
    picked = rng.sample(range(len(items)), limit)
    return [items[i] for i in sorted(picked)]


def summarize_items(
    benchmark: str,
    target: str,
    items: list[ItemResult],
    *,
    methodology: dict,
    annotations: tuple[str, ...],
    started_at: str,
    score_fn: Callable[[list[ItemResult]], float | None] | None = None,
    comparable: bool = True,
    incomparable_reasons: tuple[str, ...] = (),
) -> PairResult:
    """Fold per-item results into the pair status per the suite-wide semantics."""
    n_total = len(items)
    by_status = {status: 0 for status in ("completed", "failed", "unjudged", "skipped")}
    for item in items:
        by_status[item.status] += 1
    scored = [item.score for item in items if item.status == "completed" and item.score is not None]

    if score_fn is not None:
        score = score_fn(items)
    else:
        score = sum(scored) / len(scored) if scored else None

    reasons = []
    if by_status["unjudged"]:
        reasons.append(f"{by_status['unjudged']}/{n_total} items unjudgeable")
    if by_status["skipped"]:
        reasons.append(f"{by_status['skipped']}/{n_total} items skipped")
    if by_status["failed"]:
        reasons.append(f"{by_status['failed']}/{n_total} items failed")

    if n_total == 0:
        status, reason = "skipped", "no items to run"
    elif by_status["failed"] > n_total / 2:
        status, reason = "failed", "; ".join(reasons)
    elif by_status["completed"] == n_total:
        status, reason = "completed", None
    elif not scored:
        status, reason = "skipped", "; ".join(reasons) or "no scoreable items"
    else:
        status, reason = "partial", "; ".join(reasons)

    return PairResult(
        benchmark=benchmark,
        target=target,
        status=status,
        reason=reason,
        metrics={
            "score": score,
            "n_total": n_total,
            "n_scored": len(scored),
            "n_unjudged": by_status["unjudged"],
            "n_skipped": by_status["skipped"],
            "n_failed": by_status["failed"],
        },
        items=tuple(items),
        methodology=methodology,
        annotations=annotations,
        comparable=comparable,
        incomparable_reasons=incomparable_reasons,
        started_at=started_at,
        finished_at=utc_now(),
    )


def skipped_pair(
    benchmark: str,
    target: str,
    reason: str,
    *,
    annotations: tuple[str, ...] = (),
    comparable: bool = True,
    incomparable_reasons: tuple[str, ...] = (),
) -> PairResult:
    now = utc_now()
    return PairResult(
        benchmark=benchmark,
        target=target,
        status="skipped",
        reason=reason,
        metrics={"score": None, "n_total": 0},
        annotations=annotations,
        comparable=comparable,
        incomparable_reasons=incomparable_reasons,
        started_at=now,
        finished_at=now,
    )


class DatasetAdapter(ABC):
    """Shared dataset, cache, precondition, and result lifecycle."""

    info: AdapterInfo

    # -- adapter-specific hooks -------------------------------------------------

    @abstractmethod
    def normalize(self, ctx: DownloadContext) -> list[dict]:
        """Fetch the upstream dataset and return normalized JSONL rows.

        Only called by download(); the only place allowed to import
        `datasets`/`huggingface_hub` (lazily).
        """

    # -- optional hooks ----------------------------------------------------------

    def additional_cache_ready(self, cache: BenchCache) -> bool:
        """Validate adapter-owned assets beyond normalized dataset rows."""

        return True

    def check_preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        """Return a skip reason, or None to proceed. Extend, don't replace."""
        if self.info.needs_vision and not target.supports_vision:
            return f"target {target.label()!r} does not support vision inputs"
        if not ctx.offline_fixtures and not adapter_cache_ready(self, ctx.cache):
            detail = ctx.download_failures.get(self.info.name, "run `kairyu bench download`")
            return f"dataset not in cache ({detail})"
        return None

    def methodology(self, ctx: RunContext) -> dict:
        base = {
            "metric": self.info.metric,
            "dataset": self.info.hf_dataset,
            "revision": self.info.hf_revision,
            "temperature": 0.0,
            "source": "fixtures" if ctx.offline_fixtures else "cache",
        }
        if not ctx.offline_fixtures and adapter_cache_ready(self, ctx.cache):
            manifest = ctx.cache.read_manifest(self.info.name)
            base["manifest"] = {
                key: manifest.get(key)
                for key in ("dataset", "revision", "sources", "rows", "sha256")
            }
        return base

    def pair_methodology(self, target: BenchTarget, ctx: RunContext) -> dict:
        """Target-aware methodology; ordinary adapters retain the old shape."""

        del target
        return self.methodology(ctx)

    def _incomparable_reasons(self) -> tuple[str, ...]:
        return (self.info.incomparable_reason,) if self.info.incomparable_reason else ()

    def _skipped_pair(self, target: BenchTarget, reason: str) -> PairResult:
        return skipped_pair(
            self.info.name,
            target.label(),
            reason,
            annotations=self.info.annotations,
            comparable=self.info.comparable_to_published,
            incomparable_reasons=self._incomparable_reasons(),
        )

    def _summarize_pair(
        self,
        target: BenchTarget,
        ctx: RunContext,
        results: list[ItemResult],
        *,
        started_at: str,
    ) -> PairResult:
        return summarize_items(
            self.info.name,
            target.label(),
            results,
            methodology=self.pair_methodology(target, ctx),
            annotations=self.info.annotations,
            started_at=started_at,
            comparable=self.info.comparable_to_published,
            incomparable_reasons=self._incomparable_reasons(),
        )

    # -- shared machinery ---------------------------------------------------------

    def fixture_name(self) -> str:
        return f"{self.info.name}.jsonl"

    def load_items(self, ctx: RunContext) -> list[BenchItem]:
        if ctx.offline_fixtures:
            from importlib import resources

            text = (
                resources.files("kairyu.bench.fixtures")
                .joinpath(self.fixture_name())
                .read_text(encoding="utf-8")
            )
            import json as _json

            rows = [_json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            rows = ctx.cache.read_rows(self.info.name)
        return [BenchItem(id=str(row["id"]), payload=row) for row in rows]

    def download(self, ctx: DownloadContext) -> DownloadReport:
        from kairyu.bench.types import BenchExtrasMissing, DatasetGated, DatasetUnavailable

        if not ctx.force and adapter_cache_ready(self, ctx.cache):
            return DownloadReport(adapter=self.info.name, status="cached")
        try:
            rows = self.normalize(ctx)
        except BenchExtrasMissing as error:
            return DownloadReport(
                adapter=self.info.name, status="extras_missing", detail=str(error)
            )
        except DatasetGated as error:
            return DownloadReport(adapter=self.info.name, status="gated", detail=str(error))
        except DatasetUnavailable as error:
            return DownloadReport(adapter=self.info.name, status="unavailable", detail=str(error))
        except Exception as error:
            # schema drift (KeyError), image/codec errors, unpickling failures —
            # any un-typed normalize() error must degrade THIS dataset to data,
            # not crash the whole suite run (B2: "degradation is data, not
            # control flow"). Treated as unavailable (a non-fatal skip).
            return DownloadReport(
                adapter=self.info.name,
                status="unavailable",
                detail=f"{type(error).__name__}: {error}",
            )
        ctx.cache.write_rows(self.info.name, rows, cache_pins(self.info))
        return DownloadReport(adapter=self.info.name, status="ok", detail=f"{len(rows)} rows")


class GenerativeAdapter(DatasetAdapter):
    """Shared run loop for chat-generation benchmarks."""

    @abstractmethod
    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem: ...

    @abstractmethod
    async def score(self, item: BenchItem, response_text: str, ctx: RunContext) -> ItemResult: ...

    def pair_methodology(self, target: BenchTarget, ctx: RunContext) -> dict:
        methodology = self.methodology(ctx)
        default_policy = (
            target.sampling_mode == "adapter"
            and target.temperature is None
            and ctx.attempts == 1
        )
        if default_policy:
            return methodology

        if target.sampling_mode == "recommended":
            methodology["temperature"] = None
            wire_omitted = [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
            ]
        else:
            methodology["temperature"] = (
                0.0 if target.temperature is None else target.temperature
            )
            wire_omitted = []
        schedule = seed_schedule(target.seed, ctx.attempts)
        methodology["sampling"] = {
            "mode": target.sampling_mode,
            "attempts": ctx.attempts,
            "seeds": list(schedule),
            "seed_source": (
                "target.seed"
                if target.seed is not None
                else ("generated from zero" if ctx.attempts > 1 else "wire omission")
            ),
            "wire_omitted": wire_omitted,
            "recommended_defaults_attested": False,
        }
        return methodology

    def _attach_sampling_metrics(
        self,
        pair: PairResult,
        target: BenchTarget,
        ctx: RunContext,
    ) -> PairResult:
        if ctx.attempts == 1:
            return pair
        seeds = seed_schedule(target.seed, ctx.attempts)
        metrics: dict[str, float | int | None] = {
            **pair.metrics,
            "sampling_attempts": ctx.attempts,
            "sampling_base_items": len(pair.items),
            "sampling_complete": 0,
        }
        try:
            summary = sampling_summary(
                pair.items,
                seeds=tuple(int(seed) for seed in seeds),
                declared_binary=self.info.binary_outcomes,
            )
        except ValueError:
            return pair.model_copy(update={"metrics": metrics})
        metrics.update(sampling_metrics(summary))
        metrics["sampling_complete"] = 1
        return pair.model_copy(update={"metrics": metrics})

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        skip_reason = self.check_preconditions(target, ctx)
        if skip_reason is not None:
            return self._skipped_pair(target, skip_reason)

        items = select_items(self.load_items(ctx), ctx.limit, ctx.seed)
        seeds = seed_schedule(target.seed, ctx.attempts)
        ctx.progress.items_total(len(items) * ctx.attempts)
        semaphore = asyncio.Semaphore(ctx.concurrency)
        api_key = target_api_key(target)

        async with ctx.http_factory() as client:

            async def run_item(item: BenchItem) -> ItemResult:
                if ctx.attempts > 1:
                    retained = await asyncio.gather(
                        *(
                            run_sampling_attempt(item, attempt, int(seed))
                            for attempt, seed in enumerate(seeds, start=1)
                        )
                    )
                    return combine_attempts(item.id, retained)
                try:
                    return await run_one(item)
                finally:
                    # every item advances the bar, including skips and failures
                    ctx.progress.item_done()

            async def run_sampling_attempt(
                item: BenchItem,
                attempt_index: int,
                seed: int,
            ) -> SamplingAttemptResult:
                try:
                    result = await run_one(item, sampling_seed=seed)
                    result = result.model_copy(update={"item_id": item.id})
                    return SamplingAttemptResult(
                        attempt=attempt_index,
                        seed=seed,
                        result=result,
                    )
                finally:
                    ctx.progress.item_done()

            async def run_one(
                item: BenchItem,
                *,
                sampling_seed: int | None = None,
            ) -> ItemResult:
                request = self.build_request(item, target, ctx)
                if isinstance(request, SkipItem):
                    return ItemResult(item_id=item.id, status="skipped", error=request.reason)
                async with semaphore:
                    attempt = await attempt_item(
                        client,
                        target,
                        request,
                        item_id=item.id,
                        ctx=ctx,
                        api_key=api_key,
                        sampling_seed=sampling_seed,
                    )
                if attempt.failure is not None:
                    return attempt.failure
                result = await self.score(item, attempt.text or "", ctx)
                return result.model_copy(
                    update={
                        "latency_s": round(attempt.latency_s, 3),
                        "timing": attempt.timing,
                    }
                )

            results = await asyncio.gather(*(run_item(item) for item in items))

        pair = self._summarize_pair(
            target,
            ctx,
            list(results),
            started_at=started_at,
        )
        return self._attach_sampling_metrics(pair, target, ctx)


class LogLikelihoodAdapter(DatasetAdapter):
    """Shared run loop for teacher-forced continuation-ranking benchmarks."""

    @abstractmethod
    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> LogLikelihoodRequestSpec | SkipItem: ...

    @abstractmethod
    async def score(
        self, item: BenchItem, response: LogLikelihoodResponse, ctx: RunContext
    ) -> ItemResult: ...

    def methodology(self, ctx: RunContext) -> dict:
        methodology = super().methodology(ctx)
        methodology.update(
            {
                "request_mode": "teacher-forced continuation log-likelihood",
                "endpoint": "/v1/completions",
                "wire_extension": "kairyu_continuation",
                "target_sampling_overrides": (
                    "not applied; temperature=0 and score-bearing fields are fixed"
                ),
            }
        )
        return methodology

    def probe_result_fatal_reason(self, result: ItemResult) -> str | None:
        """Classify adapter-specific, target-wide structural incompatibility."""

        del result
        return None

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        skip_reason = self.check_preconditions(target, ctx)
        if skip_reason is not None:
            return self._skipped_pair(target, skip_reason)

        items = select_items(self.load_items(ctx), ctx.limit, ctx.seed)
        ctx.progress.items_total(len(items))
        prepared: list[tuple[int, BenchItem, LogLikelihoodRequestSpec]] = []
        results: list[ItemResult | None] = [None] * len(items)
        for index, item in enumerate(items):
            request = self.build_request(item, target, ctx)
            if isinstance(request, SkipItem):
                results[index] = ItemResult(
                    item_id=item.id,
                    status="skipped",
                    error=request.reason,
                )
                ctx.progress.item_done()
            else:
                prepared.append((index, item, request))

        if not prepared:
            return self._summarize_pair(
                target,
                ctx,
                [result for result in results if result is not None],
                started_at=started_at,
            )

        semaphore = asyncio.Semaphore(ctx.concurrency)
        api_key = target_api_key(target)
        async with ctx.http_factory() as client:

            async def execute(
                entry: tuple[int, BenchItem, LogLikelihoodRequestSpec],
                *,
                capability_probe: bool = False,
            ) -> tuple[int, ItemResult | None, str | None, str | None]:
                index, item, request = entry
                try:
                    async with semaphore:
                        attempt = await attempt_loglikelihood(
                            client,
                            target,
                            request,
                            item_id=item.id,
                            ctx=ctx,
                            api_key=api_key,
                            capability_probe=capability_probe,
                        )
                    if attempt.unsupported_reason is not None:
                        return index, None, attempt.unsupported_reason, None
                    if attempt.fatal_reason is not None:
                        return index, attempt.failure, None, attempt.fatal_reason
                    if attempt.failure is not None:
                        return index, attempt.failure, None, None
                    if attempt.response is None:  # pragma: no cover - invariant
                        raise RuntimeError("loglikelihood attempt has no outcome")
                    result = await self.score(item, attempt.response, ctx)
                    result = result.model_copy(update={"latency_s": round(attempt.latency_s, 3)})
                    return (
                        index,
                        result,
                        None,
                        (self.probe_result_fatal_reason(result) if capability_probe else None),
                    )
                finally:
                    ctx.progress.item_done()

            # The first runnable item is both the capability probe and real
            # evidence.  An explicitly unsupported extension skips the pair
            # before concurrency can fan out over the remaining dataset.
            probe_index, probe_result, unsupported, fatal_reason = await execute(
                prepared[0], capability_probe=True
            )
            if unsupported is not None:
                for _ in prepared[1:]:
                    ctx.progress.item_done()
                return self._skipped_pair(target, unsupported)
            results[probe_index] = probe_result
            if fatal_reason is not None:
                for index, item, _ in prepared[1:]:
                    results[index] = ItemResult(
                        item_id=item.id,
                        status="failed",
                        error=f"not attempted after fatal capability probe: {fatal_reason}",
                    )
                    ctx.progress.item_done()
                pair = self._summarize_pair(
                    target,
                    ctx,
                    [result for result in results if result is not None],
                    started_at=started_at,
                )
                return pair.model_copy(update={"status": "failed", "reason": fatal_reason})

            remaining = await asyncio.gather(*(execute(entry) for entry in prepared[1:]))
            for index, result, unsupported, fatal_reason in remaining:
                # Only the serial first item is a capability probe.  A later
                # generic or explicit 400 is local failed evidence, never a
                # retroactive pair-level verdict.
                if unsupported is not None:  # pragma: no cover - probe=False
                    raise RuntimeError("unexpected non-probe capability verdict")
                if fatal_reason is not None:  # pragma: no cover - probe=False
                    raise RuntimeError("unexpected non-probe fatal verdict")
                results[index] = result

        if any(result is None for result in results):  # pragma: no cover - invariant
            raise RuntimeError("loglikelihood run did not produce every item result")
        return self._summarize_pair(
            target,
            ctx,
            [result for result in results if result is not None],
            started_at=started_at,
        )
