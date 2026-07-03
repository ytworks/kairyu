"""Adapter contract for the Fugu suite + the shared generative run loop.

An adapter owns one scoreboard row: how its dataset is downloaded and
normalized, how an item becomes an OpenAI-wire request, and how a response
is scored. The runner owns everything else (targets, resume, aggregation).
Degradation is data, not control flow: every unmet precondition becomes a
PairResult(status="skipped", reason=...) so one command always completes.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

from kairyu.bench.cache import BenchCache
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DownloadReport,
    ItemResult,
    PairResult,
    SkipItem,
)

if TYPE_CHECKING:  # judge lands in its own module; adapters only see the protocol
    from kairyu.bench.judge import JudgeClient

_EXCERPT_CHARS = 2000
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class AdapterInfo:
    name: str  # registry key / directory name, kebab-case
    display_name: str  # Fugu release table row label
    metric: str  # human name of the score ("accuracy", "pass@1", ...)
    hf_dataset: str | None = None
    hf_revision: str | None = None
    gated: bool = False
    needs_docker: bool = False
    needs_execution: bool = False
    needs_vision: bool = False
    judge_preferred: bool = False
    agentic: bool = False
    annotations: tuple[str, ...] = ()  # permanent footnotes (substitute slots)


@dataclass(frozen=True)
class DownloadContext:
    cache: BenchCache
    force: bool = False


@dataclass
class RunContext:
    """Built once per suite run and shared by every (benchmark, target) pair."""

    cache: BenchCache
    http_factory: Callable[[], httpx.AsyncClient]
    judge: JudgeClient | None = None
    limit: int | None = None
    seed: int = 0
    concurrency: int = 8
    retries: int = 2
    request_timeout_s: float = 600.0
    offline_fixtures: bool = False
    smoke: bool = False
    docker: tuple[bool, str] = (False, "docker not probed")
    exec_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))
    download_failures: dict[str, str] = field(default_factory=dict)


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


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def estimate_tokens(text: str) -> int:
    """Chars/4 heuristic — recorded in methodology wherever it gates items."""
    return len(text) // 4 + 1


def normalize_base_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root if root.endswith("/v1") else f"{root}/v1"


def mcq_prompt(question: str, choices: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question, "", "Choices:"]
    lines += [
        f"{letter}) {choice}" for letter, choice in zip(letters, choices, strict=True)
    ]
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


_ANSWER_RE = re.compile(r"(?i)answer\s*(?:is|:)?\s*\**\(?([A-Z])\)?")


def extract_choice_letter(text: str, num_choices: int = 4) -> str | None:
    """Last 'Answer: X' style marker, else the last standalone letter."""
    valid = {chr(ord("A") + i) for i in range(num_choices)}
    markers = [m.group(1).upper() for m in _ANSWER_RE.finditer(text)]
    for letter in reversed(markers):
        if letter in valid:
            return letter
    standalone = re.findall(r"\b([A-Za-z])\b", text)
    for letter in reversed(standalone):
        if letter.upper() in valid:
            return letter.upper()
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


async def call_chat(
    client: httpx.AsyncClient,
    target: BenchTarget,
    request: ChatRequestSpec,
    *,
    retries: int,
    timeout_s: float,
    api_key: str | None = None,
) -> str:
    """Non-streaming POST /v1/chat/completions with backoff on 429/5xx/timeouts."""
    url = f"{normalize_base_url(target.base_url)}/chat/completions"
    body = {
        "model": target.model,
        "messages": list(request.messages),
        "temperature": request.temperature,
        "stream": False,
    }
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    attempt = 0
    while True:
        try:
            response = await client.post(
                url, json=body, headers=headers, timeout=timeout_s
            )
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
) -> PairResult:
    """Fold per-item results into the pair status per the suite-wide semantics."""
    n_total = len(items)
    by_status = {status: 0 for status in ("completed", "failed", "unjudged", "skipped")}
    for item in items:
        by_status[item.status] += 1
    scored = [item.score for item in items if item.status == "completed"]

    if score_fn is not None:
        score = score_fn(items)
    else:
        valid = [s for s in scored if s is not None]
        score = sum(valid) / len(valid) if valid else None

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
        started_at=started_at,
        finished_at=utc_now(),
    )


def skipped_pair(
    benchmark: str, target: str, reason: str, *, annotations: tuple[str, ...] = ()
) -> PairResult:
    now = utc_now()
    return PairResult(
        benchmark=benchmark,
        target=target,
        status="skipped",
        reason=reason,
        metrics={"score": None, "n_total": 0},
        annotations=annotations,
        started_at=now,
        finished_at=now,
    )


class GenerativeAdapter(ABC):
    """Shared run loop for request/response benchmarks (9 of 11 Fugu slots)."""

    info: AdapterInfo

    # -- adapter-specific hooks -------------------------------------------------

    @abstractmethod
    def normalize(self, ctx: DownloadContext) -> list[dict]:
        """Fetch the upstream dataset and return normalized JSONL rows.

        Only called by download(); the only place allowed to import
        `datasets`/`huggingface_hub` (lazily).
        """

    @abstractmethod
    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem: ...

    @abstractmethod
    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult: ...

    # -- optional hooks ----------------------------------------------------------

    def check_preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        """Return a skip reason, or None to proceed. Extend, don't replace."""
        if self.info.needs_vision and not target.supports_vision:
            return f"target {target.label()!r} does not support vision inputs"
        if not ctx.offline_fixtures and not ctx.cache.is_ready(self.info.name):
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
        if not ctx.offline_fixtures and ctx.cache.is_ready(self.info.name):
            manifest = ctx.cache.read_manifest(self.info.name)
            base["manifest"] = {
                key: manifest.get(key) for key in ("dataset", "revision", "rows", "sha256")
            }
        return base

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

        if not ctx.force and ctx.cache.is_ready(self.info.name):
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
            return DownloadReport(
                adapter=self.info.name, status="unavailable", detail=str(error)
            )
        ctx.cache.write_rows(
            self.info.name,
            rows,
            {"dataset": self.info.hf_dataset, "revision": self.info.hf_revision},
        )
        return DownloadReport(adapter=self.info.name, status="ok", detail=f"{len(rows)} rows")

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        started_at = utc_now()
        skip_reason = self.check_preconditions(target, ctx)
        if skip_reason is not None:
            return skipped_pair(
                self.info.name, target.label(), skip_reason, annotations=self.info.annotations
            )

        items = select_items(self.load_items(ctx), ctx.limit, ctx.seed)
        semaphore = asyncio.Semaphore(ctx.concurrency)
        api_key = target_api_key(target)

        async with ctx.http_factory() as client:

            async def run_item(item: BenchItem) -> ItemResult:
                request = self.build_request(item, target, ctx)
                if isinstance(request, SkipItem):
                    return ItemResult(item_id=item.id, status="skipped", error=request.reason)
                async with semaphore:
                    start = time.perf_counter()
                    try:
                        text = await call_chat(
                            client,
                            target,
                            request,
                            retries=ctx.retries,
                            timeout_s=ctx.request_timeout_s,
                            api_key=api_key,
                        )
                    except RequestFailed as error:
                        lowered = error.body.lower()
                        if error.status_code == 400 and (
                            "image" in lowered or "context" in lowered or "too long" in lowered
                        ):
                            return ItemResult(
                                item_id=item.id, status="skipped", error=str(error)
                            )
                        return ItemResult(item_id=item.id, status="failed", error=str(error))
                    latency = time.perf_counter() - start
                result = await self.score(item, text, ctx)
                return result.model_copy(update={"latency_s": round(latency, 3)})

            results = await asyncio.gather(*(run_item(item) for item in items))

        return summarize_items(
            self.info.name,
            target.label(),
            list(results),
            methodology=self.methodology(ctx),
            annotations=self.info.annotations,
            started_at=started_at,
        )


def target_api_key(target: BenchTarget) -> str | None:
    if target.api_key_env is None:
        return None
    import os

    return os.environ.get(target.api_key_env)
