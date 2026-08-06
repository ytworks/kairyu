"""Paired JSON-Schema conformance and task-quality benchmark."""

from __future__ import annotations

import asyncio
import re
import time

import httpx

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RequestFailed,
    RunContext,
    call_chat_evidence,
    excerpt,
    select_items,
)
from kairyu.bench.sampling import combine_attempts, seed_schedule
from kairyu.bench.structured import (
    CORPUS_ID,
    CORPUS_REVISION,
    StrictJsonError,
    arm_order,
    canonical_corpus_payload,
    evaluate_arm,
    jsonschema_dependency_error,
    load_packaged_corpus,
    methodology_claim,
    request_prompt,
    response_format,
    strict_json_loads,
    structured_metrics,
    structured_summary,
    validate_corpus_rows,
)
from kairyu.bench.targets import target_api_key
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    ItemResult,
    PairResult,
    SamplingAttemptResult,
    SkipItem,
    StructuredArmEvidence,
    StructuredOutputEvidence,
    StructuredTokenUsage,
)

_ERROR_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]{1,80}")
_SCHEMA_MARKERS = (
    "response_format",
    "json_schema",
    "json schema",
    "guided decoding",
    "unsupported schema",
    "invalid schema",
)


def _strict_error_envelope(body: str) -> tuple[str | None, str | None, str] | None:
    """Return safe error metadata only for a strict OpenAI error envelope."""

    if len(body.encode("utf-8", errors="replace")) > 1_048_576:
        return None

    try:
        payload = strict_json_loads(body)
    except StrictJsonError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    error = payload["error"]
    message = error.get("message")
    error_type = error.get("type")
    code = error.get("code")
    if not isinstance(message, str) or not message:
        return None
    if error_type is not None and not isinstance(error_type, str):
        return None
    if code is not None and not isinstance(code, str):
        return None
    return error_type, code, message


def _safe_error_token(value: str | None) -> str:
    if value is None or _ERROR_TOKEN_RE.fullmatch(value) is None:
        return "unspecified"
    return value


def _schema_rejection_label(error: RequestFailed) -> str | None:
    if error.status_code not in {400, 422}:
        return None
    parsed = _strict_error_envelope(error.body)
    if parsed is None:
        return None
    error_type, code, message = parsed
    marker_text = " ".join(
        value for value in (error_type, code, message) if isinstance(value, str)
    ).lower()
    if not any(marker in marker_text for marker in _SCHEMA_MARKERS):
        return None
    return (
        f"response_format schema rejected (HTTP {error.status_code}; "
        f"type={_safe_error_token(error_type)}; code={_safe_error_token(code)})"
    )


def _failure_label(error: RequestFailed) -> str:
    if error.status_code == 0:
        return "structured request transport failure"
    if error.status_code == 200:
        return "structured request invalid HTTP 200 envelope"
    return f"structured request failed with HTTP {error.status_code}"


class StructuredOutputAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="structured-output",
        display_name="Structured Output",
        metric="constrained schema-valid exact-task success rate",
        binary_outcomes=True,
        hf_dataset=CORPUS_ID,
        hf_revision=CORPUS_REVISION,
        annotations=(
            "paired constrained/control JSON-Schema corpus; token cost is endpoint-reported "
            "and currency cost is not calculated",
        ),
        evaluation_resources=(
            ("kairyu.bench.adapters", "structured_output.py"),
            ("kairyu.bench", "structured.py"),
            ("kairyu.bench.fixtures", "structured-output.jsonl"),
        ),
        evaluation_distributions=(
            "jsonschema",
            "jsonschema-specifications",
            "referencing",
            "rpds-py",
            "attrs",
        ),
    )

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        del ctx
        return load_packaged_corpus()

    def additional_cache_ready(self, cache) -> bool:
        try:
            cached = validate_corpus_rows(cache.read_rows(self.info.name))
            return canonical_corpus_payload(cached) == canonical_corpus_payload(
                load_packaged_corpus()
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

    def load_items(self, ctx: RunContext) -> list[BenchItem]:
        rows = (
            load_packaged_corpus()
            if ctx.offline_fixtures
            else validate_corpus_rows(ctx.cache.read_rows(self.info.name))
        )
        if canonical_corpus_payload(rows) != canonical_corpus_payload(
            load_packaged_corpus()
        ):
            raise ValueError("structured-output cache differs from the fixed packaged corpus")
        return [BenchItem(id=row["id"], payload=row) for row in rows]

    def check_preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        reason = super().check_preconditions(target, ctx)
        if reason is not None:
            return reason
        dependency_error = jsonschema_dependency_error()
        if dependency_error is not None:
            return dependency_error
        if target.extra_body_json is not None:
            return (
                "structured-output pairing rejects extra_body_json because an unknown vendor "
                "field could constrain the control arm"
            )
        stochastic = target.sampling_mode == "recommended" or (
            target.temperature is not None and target.temperature != 0.0
        )
        if stochastic and target.seed is None and ctx.attempts == 1:
            return "structured-output stochastic single-attempt pairing requires target.seed"
        return None

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        del ctx
        return ChatRequestSpec(
            messages=({"role": "user", "content": request_prompt(item.payload)},),
            max_tokens=min(target.max_output_tokens, item.payload["max_tokens"]),
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        del item, response_text, ctx
        raise RuntimeError("structured-output scoring requires the paired adapter run loop")

    def methodology(self, ctx: RunContext) -> dict:
        methodology = super().methodology(ctx)
        methodology.update(
            {
                "config": "fixed package-owned corpus",
                "split": (
                    "fixed five-row corpus; deterministic config limit/seed may select a subset"
                ),
                "max_tokens": "min(target.max_output_tokens, row.max_tokens)",
            }
        )
        return methodology

    async def _run_arm(
        self,
        *,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        target: BenchTarget,
        request: ChatRequestSpec,
        arm: str,
        row: dict,
        ctx: RunContext,
        api_key: str | None,
        sampling_seed: int | None,
    ) -> StructuredArmEvidence:
        started = time.perf_counter()
        try:
            async with semaphore:
                completion = await call_chat_evidence(
                    client,
                    target,
                    request,
                    response_format=response_format(row) if arm == "constrained" else None,
                    retries=ctx.retries,
                    timeout_s=ctx.request_timeout_s,
                    api_key=api_key,
                    sampling_seed=sampling_seed,
                )
        except RequestFailed as error:
            latency = time.perf_counter() - started
            rejection = _schema_rejection_label(error) if arm == "constrained" else None
            if rejection is not None:
                return StructuredArmEvidence(
                    arm="constrained",
                    outcome="schema_rejected",
                    http_status=error.status_code,
                    latency_s=latency,
                    error=rejection,
                )
            return StructuredArmEvidence(
                arm=arm,
                outcome="failed",
                http_status=error.status_code,
                latency_s=latency,
                error=_failure_label(error),
            )
        finally:
            ctx.progress.item_done()
        usage = (
            StructuredTokenUsage(
                prompt_tokens=completion.usage[0],
                completion_tokens=completion.usage[1],
                total_tokens=completion.usage[2],
            )
            if completion.usage is not None
            else None
        )
        return StructuredArmEvidence(
            arm=arm,
            outcome="completed",
            http_status=200,
            content=completion.content,
            refusal=completion.refusal,
            message_auxiliary=completion.message_auxiliary,
            finish_reason=completion.finish_reason,
            usage=usage,
            latency_s=time.perf_counter() - started,
        )

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        from kairyu.bench.adapters.base import utc_now

        started_at = utc_now()
        skip_reason = self.check_preconditions(target, ctx)
        if skip_reason is not None:
            return self._skipped_pair(target, skip_reason)

        items = select_items(self.load_items(ctx), ctx.limit, ctx.seed)
        seeds = seed_schedule(target.seed, ctx.attempts)
        ctx.progress.items_total(len(items) * len(seeds) * 2)
        semaphore = asyncio.Semaphore(ctx.concurrency)
        api_key = target_api_key(target)

        async with ctx.http_factory() as client:

            async def run_one(
                item: BenchItem,
                *,
                sampling_seed: int | None,
                observation_index: int,
            ) -> ItemResult:
                request = self.build_request(item, target, ctx)
                if isinstance(request, SkipItem):  # pragma: no cover - fixed corpus never skips
                    return ItemResult(item_id=item.id, status="skipped", error=request.reason)
                order = arm_order(observation_index)
                retained: dict[str, StructuredArmEvidence] = {}
                for arm in order:
                    retained[arm] = await self._run_arm(
                        client=client,
                        semaphore=semaphore,
                        target=target,
                        request=request,
                        arm=arm,
                        row=item.payload,
                        ctx=ctx,
                        api_key=api_key,
                        sampling_seed=sampling_seed,
                    )
                evidence = StructuredOutputEvidence(
                    seed=sampling_seed,
                    order=order,
                    constrained=retained["constrained"],
                    unconstrained=retained["unconstrained"],
                )
                failed = [
                    arm.arm
                    for arm in (evidence.constrained, evidence.unconstrained)
                    if arm.outcome == "failed"
                ]
                latency = evidence.constrained.latency_s + evidence.unconstrained.latency_s
                if failed:
                    return ItemResult(
                        item_id=item.id,
                        status="failed",
                        error="structured arm failure: " + ", ".join(failed),
                        response_excerpt=excerpt(evidence.constrained.content),
                        latency_s=latency,
                        structured_output=evidence,
                    )
                outcome = evaluate_arm(evidence.constrained, item.payload)
                return ItemResult(
                    item_id=item.id,
                    status="completed",
                    score=float(outcome["task_correct"]),
                    response_excerpt=excerpt(evidence.constrained.content),
                    latency_s=latency,
                    structured_output=evidence,
                )

            async def run_source(source_index: int, item: BenchItem) -> ItemResult:
                if len(seeds) == 1:
                    return await run_one(
                        item,
                        sampling_seed=seeds[0],
                        observation_index=source_index,
                    )
                attempts = await asyncio.gather(
                    *(
                        run_one(
                            item,
                            sampling_seed=seed,
                            observation_index=source_index * len(seeds) + attempt_index,
                        )
                        for attempt_index, seed in enumerate(seeds)
                    )
                )
                return combine_attempts(
                    item.id,
                    tuple(
                        SamplingAttemptResult(
                            attempt=index,
                            seed=int(seed),
                            result=result,
                        )
                        for index, (seed, result) in enumerate(
                            zip(seeds, attempts, strict=True), start=1
                        )
                    ),
                )

            results = await asyncio.gather(
                *(run_source(index, item) for index, item in enumerate(items))
            )

        pair = self._summarize_pair(
            target,
            ctx,
            list(results),
            started_at=started_at,
        )
        methodology = dict(pair.methodology)
        methodology["structured_output"] = methodology_claim(
            selected_ids=[item.id for item in items],
            seeds=list(seeds),
        )
        pair = pair.model_copy(update={"methodology": methodology})
        summary = structured_summary(pair)
        metrics = {
            **pair.metrics,
            "structured_complete": int(summary is not None),
            "structured_observations": len(items) * len(seeds),
        }
        if summary is not None:
            metrics.update(structured_metrics(summary))
        pair = pair.model_copy(update={"metrics": metrics})
        return self._attach_sampling_metrics(pair, target, ctx)
