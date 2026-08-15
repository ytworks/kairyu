"""Sandboxed code-execution backend for L2 orchestration (ECO-D1/ECO-D3).

An executor role runs model-generated code against model-generated tests in an
isolated sandbox service and feeds the structured result back into later role
prompts as untrusted machine data. Executor stages consume orchestration steps
but never tokens, and a sandbox outage degrades to an ``unavailable`` report
instead of failing the request.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

_DEFAULT_TIMEOUT_MARGIN_S = 5.0
_DEFAULT_EXCERPT_BYTES = 4096
_RETRYABLE_STATUS = 429
_TIMEOUT_HEADER = "X-Kairyu-Timeout-Ms"

NOT_APPLICABLE_SENTINEL = "NOT_APPLICABLE"

_PYTHON_FENCE = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class ExecutionLimits:
    wall_time_s: float = 10.0
    cpu_time_s: float = 8.0
    memory_mb: int = 512
    processes: int = 64
    output_bytes: int = 32768

    def __post_init__(self) -> None:
        if self.wall_time_s <= 0 or self.cpu_time_s <= 0:
            raise ValueError("execution time limits must be positive")
        if self.memory_mb <= 0 or self.processes <= 0 or self.output_bytes <= 0:
            raise ValueError("execution resource limits must be positive")


@dataclass(frozen=True)
class ExecutionRequest:
    request_id: str
    language: str
    files: Mapping[str, str]
    entry: str = "pytest"
    limits: ExecutionLimits = ExecutionLimits()

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("execution request requires at least one file")
        object.__setattr__(self, "files", dict(self.files))


@dataclass(frozen=True)
class CaseOutcome:
    id: str
    outcome: str  # "passed" | "failed" | "error"


@dataclass(frozen=True)
class ExecutionReport:
    """One sandbox verdict; ``status`` other than ``ok`` carries no test rows."""

    status: str  # ok | timeout | oom | setup_error | unavailable | skipped | unsupported
    tests: tuple[CaseOutcome, ...] = ()
    passed: int = 0
    failed: int = 0
    errors: int = 0
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    duration_ms: int = 0
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }
        if self.tests:
            payload["tests"] = [
                {"id": outcome.id, "outcome": outcome.outcome} for outcome in self.tests
            ]
        if self.stdout_excerpt:
            payload["stdout_excerpt"] = self.stdout_excerpt
        if self.stderr_excerpt:
            payload["stderr_excerpt"] = self.stderr_excerpt
        if self.detail:
            payload["detail"] = self.detail
        return payload


@runtime_checkable
class ExecutionBackend(Protocol):
    async def execute(self, request: ExecutionRequest) -> ExecutionReport: ...

    async def shutdown(self) -> None: ...


@dataclass(frozen=True)
class ExecutorDescriptor:
    """Public identity of one deployment executor (mirrors EngineDescriptor)."""

    backend_type: str
    base_url: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"backend_type": self.backend_type, "base_url": self.base_url}


def _salvage_truncated(code: str) -> str | None:
    """Trim trailing lines until the fragment parses (bounded).

    A generation cut off at its token cap leaves an unclosed fence whose tail
    is mid-statement; the leading complete statements are still worth
    executing as advisory evidence.
    """

    lines = code.rstrip().splitlines()
    for _ in range(min(len(lines), 50)):
        candidate = "\n".join(lines).rstrip()
        if not candidate:
            return None
        try:
            ast.parse(candidate)
        except SyntaxError:
            lines.pop()
            continue
        return candidate
    return None


def extract_python_block(text: str) -> str | None:
    """Return the first complete ```python fenced block; salvage an unclosed
    trailing fence by trimming to the last parseable statement; else the whole
    text iff it parses as Python. Prose and non-Python fences return ``None``."""

    if not text:
        return None
    match = _PYTHON_FENCE.search(text)
    if match is not None:
        code = match.group(1).strip("\n")
        return code or None
    marker = text.find("```python")
    if marker != -1:
        tail = text[marker + len("```python") :].lstrip("\n")
        return _salvage_truncated(tail)
    stripped = text.strip()
    if not stripped or stripped == NOT_APPLICABLE_SENTINEL:
        return None
    if "```" in stripped:
        # A fenced block in another language is an explicit non-Python answer.
        return None
    try:
        ast.parse(stripped)
    except SyntaxError:
        return None
    return stripped


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


def skipped_report(reason: str) -> ExecutionReport:
    return ExecutionReport(status="skipped", detail=reason)


def unavailable_report(reason: str) -> ExecutionReport:
    return ExecutionReport(status="unavailable", detail=reason)


def render_single_report(report: ExecutionReport) -> str:
    return json.dumps(report.as_dict(), sort_keys=True)


def render_matrix_report(reports: Mapping[str, ExecutionReport]) -> str:
    """Combine per-candidate reports plus a per-test consensus signal.

    ``consensus`` counts, for each test id, how many candidates passed it: a
    test failed by every candidate is suspect, which downstream prompts are
    licensed to weigh against the test rather than the candidates.
    """

    consensus: dict[str, int] = {}
    observed: dict[str, int] = {}
    for report in reports.values():
        for outcome in report.tests:
            observed[outcome.id] = observed.get(outcome.id, 0) + 1
            if outcome.outcome == "passed":
                consensus[outcome.id] = consensus.get(outcome.id, 0) + 1
            else:
                consensus.setdefault(outcome.id, 0)
    payload = {
        "candidates": {name: report.as_dict() for name, report in reports.items()},
        "consensus": {
            test_id: {"passed": passed, "of": observed[test_id]}
            for test_id, passed in sorted(consensus.items())
        },
    }
    return json.dumps(payload, sort_keys=True)


@dataclass
class HttpExecutionBackend:
    """HTTP client for the compose-owned sandbox runner (degrade, never fail)."""

    base_url: str
    timeout_s: float = 25.0
    queue_wait_s: float = 0.0
    excerpt_bytes: int = _DEFAULT_EXCERPT_BYTES
    uds_path: str | None = None
    transport: httpx.AsyncBaseTransport | None = None
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.queue_wait_s < 0:
            raise ValueError("queue_wait_s must be non-negative")

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            transport = self.transport
            if transport is None and self.uds_path is not None:
                transport = httpx.AsyncHTTPTransport(uds=self.uds_path)
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_s,
                transport=transport,
            )
        return self._client

    async def execute(self, request: ExecutionRequest) -> ExecutionReport:
        payload = {
            "schema_version": 1,
            "language": request.language,
            "files": dict(request.files),
            "entry": {"kind": request.entry},
            "limits": {
                "wall_time_s": request.limits.wall_time_s,
                "cpu_time_s": request.limits.cpu_time_s,
                "memory_mb": request.limits.memory_mb,
                "processes": request.limits.processes,
                "output_bytes": request.limits.output_bytes,
            },
        }
        request_timeout = min(
            self.timeout_s,
            request.limits.wall_time_s + self.queue_wait_s + _DEFAULT_TIMEOUT_MARGIN_S,
        )
        try:
            response = await self._post_with_one_retry(payload, request_timeout)
        except (TimeoutError, httpx.HTTPError, OSError) as error:
            return unavailable_report(type(error).__name__)
        if response.status_code == 422:
            return ExecutionReport(status="unsupported", detail=request.language)
        if response.status_code >= 400:
            return unavailable_report(f"http_{response.status_code}")
        try:
            body = response.json()
        except ValueError:
            return unavailable_report("invalid_json")
        return self._parse_report(body)

    async def _post_with_one_retry(
        self,
        payload: Mapping[str, object],
        timeout: float,
    ) -> httpx.Response:
        client = self._get_client()
        deadline = asyncio.get_running_loop().time() + timeout

        async def post() -> httpx.Response:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError
            headers = {_TIMEOUT_HEADER: str(max(1, math.ceil(remaining * 1000)))}
            return await asyncio.wait_for(
                client.post(
                    "/v1/execute",
                    json=payload,
                    headers=headers,
                    timeout=remaining,
                ),
                timeout=remaining,
            )

        response = await post()
        if response.status_code == _RETRYABLE_STATUS:
            if deadline - asyncio.get_running_loop().time() <= 0.5:
                raise TimeoutError
            await asyncio.sleep(0.5)
            response = await post()
        return response

    def _parse_report(self, body: object) -> ExecutionReport:
        if not isinstance(body, Mapping):
            return unavailable_report("invalid_report")
        status = body.get("status")
        if status not in {"ok", "timeout", "oom", "setup_error"}:
            return unavailable_report("invalid_status")
        raw_tests = body.get("tests") or ()
        tests: list[CaseOutcome] = []
        if isinstance(raw_tests, list):
            for entry in raw_tests:
                if (
                    isinstance(entry, Mapping)
                    and isinstance(entry.get("id"), str)
                    and entry.get("outcome") in {"passed", "failed", "error"}
                ):
                    tests.append(CaseOutcome(id=entry["id"], outcome=entry["outcome"]))

        def _count(key: str) -> int:
            value = body.get(key)
            return value if isinstance(value, int) and value >= 0 else 0

        def _excerpt(key: str) -> str:
            value = body.get(key)
            return _truncate(value, self.excerpt_bytes) if isinstance(value, str) else ""

        return ExecutionReport(
            status=str(status),
            tests=tuple(tests),
            passed=_count("passed"),
            failed=_count("failed"),
            errors=_count("errors"),
            stdout_excerpt=_excerpt("stdout_excerpt"),
            stderr_excerpt=_excerpt("stderr_excerpt"),
            duration_ms=_count("duration_ms"),
        )

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
