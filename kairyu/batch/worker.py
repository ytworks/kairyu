"""In-gateway batch worker: drains jobs through the served engines (m7 D7).

The worker's fixed consumer pool caps its concurrency strictly below the
server's global guard so interactive latency is protected (goal G3 gate C4).
Requests go through the same engines mapping the HTTP path uses — a pool member
is a pool member whether the request arrived interactively or from a batch file.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from kairyu.batch.envelope import BatchLineEnvelope
from kairyu.batch.store import (
    BatchJob,
    BatchStoreProtocol,
    JsonlFileWriter,
    StaleBatchClaimError,
)
from kairyu.engine.backend import (
    EngineBackend,
    UpstreamClientError,
    backend_admission_upper_bound_async,
    prepare_backend_request,
)
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ExecutedChat,
    chat_error_from_upstream_client_error,
    execute_chat,
    validate_chat_policy,
    validate_chat_request_async,
)
from kairyu.entrypoints.server.errors import (
    invalid_request_payload,
    sanitize_backend_error,
)
from kairyu.entrypoints.server.metering import (
    TokenLimiterSink,
    UsageLedgerSink,
    record_tenant_usage,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.tenancy import TenantConfig

if TYPE_CHECKING:
    from kairyu.batch.postgres_store import BatchClaim
    from kairyu.entrypoints.server.slo import AdmissionController

logger = logging.getLogger("kairyu.batch")
_INPUT_QUEUE_FACTOR = 2
_INPUT_SENTINEL = object()
_ITERATION_SENTINEL = object()
_BATCH_PRESSURE_POLL_S = 0.01
_WORKER_ID_ENV = "KAIRYU_BATCH_WORKER_ID"


class _BatchClaimLeaseLost(RuntimeError):
    """Internal signal: stop work without publishing through a stale fence."""


@dataclass(frozen=True)
class BatchLineUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    quota_accounted: bool = False


class BatchWorker:
    def __init__(
        self,
        store: BatchStoreProtocol,
        engines: Mapping[str, EngineBackend],
        max_concurrency: int = 4,
        metrics=None,
        chat_templates: Mapping[str, ChatTemplate] | None = None,
        usage_ledger: UsageLedgerSink | None = None,
        tenant_limiter: TokenLimiterSink | None = None,
        tenant_config: TenantConfig | None = None,
        admission_controller: AdmissionController | None = None,
        worker_id: str | None = None,
        claim_poll_interval_s: float = 0.5,
        claim_lease_seconds: float = 30.0,
        legacy_chat_models: AbstractSet[str] | None = None,
    ) -> None:
        if claim_poll_interval_s <= 0:
            raise ValueError("claim_poll_interval_s must be positive")
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be positive")
        legacy_chat_models = frozenset(legacy_chat_models or ())
        validate_chat_policy(chat_templates, legacy_chat_models)
        missing_chat_policy = (
            set(engines) - set(chat_templates or {}) - set(legacy_chat_models)
        )
        if missing_chat_policy:
            logger.warning(
                "batch models have no configured chat rendering policy; chat "
                "requests will be rejected before dispatch until each model has a "
                "ChatTemplate or legacy_chat_models membership: %s",
                sorted(missing_chat_policy),
            )
        self._store = store
        self._engines = engines
        self._max_concurrency = max_concurrency
        self._metrics = metrics
        self._chat_templates = chat_templates
        self._legacy_chat_models = legacy_chat_models
        self._usage_ledger = usage_ledger
        self._tenant_limiter = tenant_limiter
        self._tenant_config = tenant_config or TenantConfig()
        self._admission_controller = admission_controller
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._claim_wakeup = asyncio.Event()
        claim_next = getattr(store, "claim_next_batch", None)
        self._claim_next = claim_next if callable(claim_next) else None
        self._worker_id = (
            worker_id
            or os.environ.get(_WORKER_ID_ENV)
            or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
        )
        self._claim_poll_interval_s = claim_poll_interval_s
        self._claim_lease_seconds = claim_lease_seconds

    def submit(self, batch_id: str) -> None:
        if self._claim_next is None:
            self._queue.put_nowait(batch_id)
        else:
            # Shared workers discover ownership through the store. This is only
            # a local low-latency hint; every other gateway polls the same queue.
            self._claim_wakeup.set()

    async def run(self) -> None:
        """Job loop; cancelled by the app lifespan on shutdown."""
        if self._claim_next is not None:
            await self._run_shared()
            return
        while True:
            batch_id = await self._queue.get()
            try:
                await self.process(batch_id)
            except Exception:
                logger.exception("batch job crashed", extra={"batch_id": batch_id})

    async def _run_shared(self) -> None:
        assert self._claim_next is not None
        while True:
            try:
                claim = await asyncio.to_thread(
                    self._claim_next,
                    self._worker_id,
                    lease_seconds=self._claim_lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("shared batch claim failed")
                await asyncio.sleep(self._claim_poll_interval_s)
                continue
            if claim is not None:
                try:
                    await self.process(claim.batch_id, claim=claim)
                except Exception:
                    logger.exception(
                        "shared batch job crashed",
                        extra={"batch_id": claim.batch_id},
                    )
                continue

            self._claim_wakeup.clear()
            try:
                await asyncio.wait_for(
                    self._claim_wakeup.wait(),
                    timeout=self._claim_poll_interval_s,
                )
            except TimeoutError:
                pass

    async def _cancelled(
        self,
        batch_id: str,
        claim_lost: asyncio.Event | None = None,
    ) -> bool:
        if self._claim_next is not None:
            # Shared cancellation atomically invalidates the fenced claim.
            # The dedicated heartbeat observes that state without issuing two
            # synchronous PostgreSQL reads for every JSONL line.
            return claim_lost is not None and claim_lost.is_set()
        job = await asyncio.to_thread(self._store.get_batch, batch_id)
        return job.status == "cancelled"

    async def _wait_for_interactive_slack(self) -> bool:
        controller = self._admission_controller
        waited = False
        while controller is not None and controller.batch_should_yield():
            waited = True
            await asyncio.sleep(_BATCH_PRESSURE_POLL_S)
        return waited

    async def _finish(
        self,
        job: BatchJob,
        state: str,
        claim: BatchClaim | None = None,
        publication_committed: list[bool] | None = None,
    ) -> None:
        job.status = state
        if claim is None:
            setattr(job, f"{state}_at", int(time.time()))
            await self._run_blocking_cancellation_safe(
                self._store.update_batch,
                job,
            )
            if publication_committed is not None:
                publication_committed[0] = True
        else:
            # The shared store fills the terminal timestamp from the same DB
            # clock that governs claim leases and their audit chronology.
            setattr(job, f"{state}_at", None)
            await self._run_blocking_cancellation_safe(
                self._store.finalize_claim,
                claim,
                job,
                completion_flag=publication_committed,
            )
        if self._metrics is not None:
            self._metrics.batch_jobs_total.labels(state=state).inc()

    @staticmethod
    async def _run_blocking_cancellation_safe(
        function: Callable[..., Any],
        *args: object,
        completion_flag: list[bool] | None = None,
        **kwargs: object,
    ) -> Any:
        """Let a started store operation settle before propagating cancellation.

        ``asyncio.to_thread`` cannot stop its underlying thread. Propagating
        cancellation immediately would let rollback race a still-running file
        commit, or delete files after a terminal DB commit. Shield the task,
        remember cancellation, and re-raise it only after the operation's
        durable outcome is known.
        """
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        cancelled: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancelled = error
        try:
            result = task.result()
        except Exception:
            if cancelled is not None:
                raise cancelled from None
            raise
        if completion_flag is not None:
            completion_flag[0] = True
        if cancelled is not None:
            raise cancelled
        return result

    async def _run_line(
        self,
        line: object,
        endpoint: str,
        seen_custom_ids: set[str],
        tenant: str = "default",
    ) -> tuple[dict | None, dict | None, BatchLineUsage | None]:
        """Execute one input line; return output, error, and successful usage."""
        # a line that is valid JSON but not an object (e.g. a bare `5`) must
        # become a per-line error record, never escape the pipeline and wedge
        # the whole job in "in_progress" forever (S1)
        custom_id = line.get("custom_id") if isinstance(line, dict) else None
        try:
            envelope = BatchLineEnvelope.model_validate(
                line, context={"endpoint": endpoint}
            )
            custom_id = envelope.custom_id
            if custom_id in seen_custom_ids:
                return self._line_error(
                    custom_id,
                    invalid_request_payload(
                        f"custom_id {custom_id!r} is duplicated"
                    ),
                )
            # No await occurs between membership and add, so this check remains
            # atomic across the fixed asyncio consumer pool.
            seen_custom_ids.add(custom_id)
            request = ChatCompletionRequest.model_validate(envelope.body)
        except ValidationError as error:
            return self._line_error(custom_id, invalid_request_payload(str(error)))

        quota_accounted = False
        metric_admitted = False
        try:
            record_preplacement = getattr(
                self._metrics,
                "record_preplacement_phase",
                None,
            )
            validation_started_ns = time.perf_counter_ns()
            try:
                validated = await validate_chat_request_async(
                    request,
                    self._engines,
                    self._chat_templates,
                    request_id=f"batch-{uuid.uuid4().hex[:12]}",
                    priority=self._tenant_config.limits_for(tenant).batch_priority,
                    scheduling_class="batch",
                    legacy_chat_models=self._legacy_chat_models,
                )
            finally:
                if callable(record_preplacement):
                    record_preplacement(
                        "batch",
                        "request_validation",
                        max(0, time.perf_counter_ns() - validation_started_ns),
                    )
            admission = None
            acquire = getattr(self._tenant_limiter, "acquire", None)
            if callable(acquire):
                admission = acquire(tenant)
            try:
                record_admission = getattr(
                    self._metrics,
                    "record_tenant_admission",
                    None,
                )
                if (
                    admission is not None
                    and not admission.admitted
                    and callable(record_admission)
                ):
                    record_admission(
                        tenant,
                        source="batch",
                        admitted=False,
                        reason=admission.reason,
                    )
                if admission is not None and not admission.admitted:
                    return self._line_error(
                        custom_id,
                        {
                            "message": (
                                f"tenant {tenant!r} admission limit exceeded "
                                f"({admission.reason})"
                            ),
                            "type": "rate_limit_error",
                            "code": "tenant_rate_limited",
                        },
                    )
                prepare_started_ns = time.perf_counter_ns()
                try:
                    await prepare_backend_request(
                        validated.engine,
                        validated.generation_request,
                    )
                except UpstreamClientError as error:
                    raise chat_error_from_upstream_client_error(error) from error
                except ValueError as error:
                    raise ChatRequestError(
                        str(error),
                        code=getattr(error, "code", "invalid_request"),
                    ) from error
                finally:
                    if callable(record_preplacement):
                        record_preplacement(
                            "batch",
                            "backend_prepare",
                            max(
                                0,
                                time.perf_counter_ns() - prepare_started_ns,
                            ),
                        )
                admission_started_ns = time.perf_counter_ns()
                try:
                    bound = await backend_admission_upper_bound_async(
                        validated.engine,
                        validated.generation_request,
                    )
                except ValueError as error:
                    raise ChatRequestError(
                        str(error),
                        code=getattr(error, "code", "invalid_request"),
                    ) from error
                admission_ns = max(
                    0,
                    time.perf_counter_ns() - admission_started_ns,
                )
                reserve_started_ns = time.perf_counter_ns()
                reserve = (
                    getattr(admission, "reserve_tokens", None)
                    if admission is not None
                    else None
                )
                if callable(reserve):
                    quota_accounted = reserve(
                        bound.tokens,
                        refundable_on_exact_usage=(
                            bound.refundable_on_exact_usage
                        ),
                    )
                    if callable(record_admission):
                        record_admission(
                            tenant,
                            source="batch",
                            admitted=quota_accounted,
                            reason=admission.reason,
                        )
                        metric_admitted = quota_accounted
                    if not quota_accounted:
                        return self._line_error(
                            custom_id,
                            {
                                "message": (
                                    f"tenant {tenant!r} admission limit exceeded "
                                    f"({admission.reason})"
                                ),
                                "type": "rate_limit_error",
                                "code": "tenant_rate_limited",
                            },
                        )
                elif admission is not None and callable(record_admission):
                    record_admission(
                        tenant,
                        source="batch",
                        admitted=True,
                        reason=admission.reason,
                    )
                    metric_admitted = True
                admission_ns += max(
                    0,
                    time.perf_counter_ns() - reserve_started_ns,
                )
                if callable(record_preplacement):
                    record_preplacement(
                        "batch",
                        "admission",
                        admission_ns,
                    )
                if self._metrics is not None:
                    self._metrics.record_priority(
                        validated.generation_request.scheduling_class,
                        source="batch",
                    )
                if quota_accounted:
                    admission.mark_dispatched()
                executed = await execute_chat(validated)
                usage = self._line_usage(
                    executed,
                    quota_accounted=quota_accounted,
                )
                if quota_accounted:
                    admission.settle_tokens(
                        usage.prompt_tokens + usage.completion_tokens,
                        exact=executed.result.usage is not None,
                    )
            finally:
                if admission is not None and admission.admitted:
                    admission.release()
                    record_release = getattr(
                        self._metrics,
                        "record_tenant_release",
                        None,
                    )
                    if callable(record_release) and metric_admitted:
                        record_release(tenant, source="batch")
            response = executed.response
            return (
                {
                    "id": f"batch_req_{uuid.uuid4().hex[:16]}",
                    "custom_id": custom_id,
                    "response": {"status_code": 200, "body": response.model_dump()},
                    "error": None,
                },
                None,
                usage,
            )
        except ChatRequestError as error:
            usage = (
                self._line_usage(
                    error.execution,
                    quota_accounted=quota_accounted,
                )
                if error.execution is not None
                else None
            )
            output, error_record, _ = self._line_error(custom_id, error.payload())
            return output, error_record, usage
        except Exception as error:
            logger.exception("batch request backend error")
            return self._line_error(custom_id, sanitize_backend_error(error))

    @staticmethod
    def _line_usage(
        executed: ExecutedChat,
        *,
        quota_accounted: bool = False,
    ) -> BatchLineUsage:
        details = executed.response.usage.prompt_tokens_details
        return BatchLineUsage(
            model=executed.response.model,
            prompt_tokens=executed.response.usage.prompt_tokens,
            completion_tokens=executed.response.usage.completion_tokens,
            cached_tokens=details.cached_tokens if details is not None else 0,
            quota_accounted=quota_accounted,
        )

    @staticmethod
    def _line_error(
        custom_id: object, payload: dict
    ) -> tuple[None, dict, None]:
        return (
            None,
            {
                "id": f"batch_req_{uuid.uuid4().hex[:16]}",
                "custom_id": custom_id,
                "error": payload,
            },
            None,
        )

    @staticmethod
    def _failure_class(error: BaseException) -> str:
        if isinstance(error, BaseExceptionGroup) and error.exceptions:
            return BatchWorker._failure_class(error.exceptions[0])
        return type(error).__name__

    async def _rollback_writer(self, writer: JsonlFileWriter | None) -> None:
        if writer is None:
            return
        try:
            await self._run_blocking_cancellation_safe(writer.rollback)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("failed to roll back batch spool")

    async def process(
        self,
        batch_id: str,
        claim: BatchClaim | None = None,
    ) -> None:
        if claim is None:
            job = await self._run_blocking_cancellation_safe(
                self._store.get_batch,
                batch_id,
            )
            if job.status == "cancelled":
                return
            job.status = "in_progress"
            job.in_progress_at = int(time.time())
            await self._run_blocking_cancellation_safe(
                self._store.update_batch,
                job,
            )
        else:
            if claim.batch_id != batch_id:
                raise ValueError("batch claim does not match the requested batch")
            # The atomic claim already moved the durable row to in_progress.
            job = claim.job.model_copy(deep=True)

        claim_state = [claim]
        claim_lost = asyncio.Event()
        lease_error: list[Exception | None] = [None]
        terminal_published = [False]
        heartbeat_task: asyncio.Task[None] | None = None

        def ensure_claim() -> None:
            if claim_lost.is_set():
                raise _BatchClaimLeaseLost from lease_error[0]

        if claim is not None:
            heartbeat_interval_s = max(0.05, self._claim_lease_seconds / 3.0)

            async def heartbeat() -> None:
                while True:
                    await asyncio.sleep(heartbeat_interval_s)
                    try:
                        claim_state[0] = await asyncio.to_thread(
                            self._store.renew_claim,
                            claim_state[0],
                            lease_seconds=self._claim_lease_seconds,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        lease_error[0] = error
                        claim_lost.set()
                        return

            heartbeat_task = asyncio.create_task(heartbeat())

        output_writer: JsonlFileWriter | None = None
        error_writer: JsonlFileWriter | None = None
        try:
            input_queue: asyncio.Queue[object] = asyncio.Queue(
                maxsize=max(1, self._max_concurrency * _INPUT_QUEUE_FACTOR)
            )
            input_error: Exception | None = None
            claim_writer = getattr(
                self._store,
                "create_claim_jsonl_writer",
                None,
            )
            if claim is not None and callable(claim_writer):
                output_writer = claim_writer(
                    claim,
                    filename=f"{batch_id}_output.jsonl",
                    purpose="batch_output",
                    owner=job.owner,
                )
                error_writer = claim_writer(
                    claim,
                    filename=f"{batch_id}_errors.jsonl",
                    purpose="batch_output",
                    owner=job.owner,
                )
            else:
                output_writer = self._store.create_jsonl_writer(
                    filename=f"{batch_id}_output.jsonl",
                    purpose="batch_output",
                    owner=job.owner,
                )
                error_writer = self._store.create_jsonl_writer(
                    filename=f"{batch_id}_errors.jsonl",
                    purpose="batch_output",
                    owner=job.owner,
                )
            seen_custom_ids: set[str] = set()

            async def produce() -> None:
                nonlocal input_error
                async def enqueue(raw_line: bytes) -> bool:
                    nonlocal input_error
                    if await self._cancelled(batch_id, claim_lost):
                        return False
                    if not raw_line.strip():
                        return True
                    try:
                        line = json.loads(raw_line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        input_error = error
                        return False
                    job.request_counts.total += 1
                    await input_queue.put(line)
                    return True

                async_iter_lines = getattr(
                    self._store,
                    "iter_file_lines_async",
                    None,
                )
                if callable(async_iter_lines):
                    input_lines = async_iter_lines(
                        job.input_file_id,
                        owner=job.owner,
                    )
                    try:
                        async for raw_line in input_lines:
                            if not await enqueue(raw_line):
                                break
                    except KeyError as error:
                        input_error = error
                    finally:
                        close = getattr(input_lines, "aclose", None)
                        if close is not None:
                            await close()
                else:
                    try:
                        input_lines = await asyncio.to_thread(
                            self._store.iter_file_lines,
                            job.input_file_id,
                            owner=job.owner,
                        )
                    except KeyError as error:
                        input_error = error
                    else:
                        try:
                            while True:
                                raw_line = await asyncio.to_thread(
                                    next,
                                    input_lines,
                                    _ITERATION_SENTINEL,
                                )
                                if raw_line is _ITERATION_SENTINEL:
                                    break
                                if not await enqueue(raw_line):
                                    break
                        finally:
                            close = getattr(input_lines, "close", None)
                            if close is not None:
                                await asyncio.to_thread(close)
                for _ in range(self._max_concurrency):
                    await input_queue.put(_INPUT_SENTINEL)

            async def consume() -> None:
                while True:
                    line = await input_queue.get()
                    try:
                        if line is _INPUT_SENTINEL:
                            return
                        if input_error is not None or await self._cancelled(
                            batch_id, claim_lost
                        ):
                            continue
                        waited = await self._wait_for_interactive_slack()
                        if waited and await self._cancelled(batch_id, claim_lost):
                            continue
                        output, error, usage = await self._run_line(
                            line, job.endpoint, seen_custom_ids, job.owner
                        )
                        if usage is not None:
                            record_tenant_usage(
                                tenant=job.owner,
                                model=usage.model,
                                prompt_tokens=usage.prompt_tokens,
                                completion_tokens=usage.completion_tokens,
                                cached_tokens=usage.cached_tokens,
                                ledger=self._usage_ledger,
                                metrics=self._metrics,
                                limiter=(
                                    None
                                    if usage.quota_accounted
                                    else self._tenant_limiter
                                ),
                            )
                        if output is not None:
                            output_writer.append(output)
                            job.request_counts.completed += 1
                        if error is not None:
                            error_writer.append(error)
                            job.request_counts.failed += 1
                    finally:
                        input_queue.task_done()

            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(produce())
                for _ in range(self._max_concurrency):
                    task_group.create_task(consume())

            ensure_claim()
            current = await asyncio.to_thread(
                self._store.get_batch,
                batch_id,
            )  # cancellation may have landed
            current.request_counts = job.request_counts.model_copy()
            if current.status == "cancelled":
                await self._rollback_writer(output_writer)
                await self._rollback_writer(error_writer)
                if claim is None:
                    current.cancelled_at = current.cancelled_at or int(time.time())
                    await self._run_blocking_cancellation_safe(
                        self._store.update_batch,
                        current,
                    )
                if self._metrics is not None:
                    self._metrics.batch_jobs_total.labels(state="cancelled").inc()
                return
            if input_error is not None:
                await self._rollback_writer(output_writer)
                await self._rollback_writer(error_writer)
                current.errors = {"message": f"invalid input file: {input_error}"}
                await self._finish(
                    current,
                    "failed",
                    claim_state[0],
                    terminal_published,
                )
                return

            if output_writer.has_content:
                output_file = await self._run_blocking_cancellation_safe(
                    output_writer.commit
                )
                current.output_file_id = output_file.id
            else:
                await self._run_blocking_cancellation_safe(output_writer.abort)
            if error_writer.has_content:
                error_file = await self._run_blocking_cancellation_safe(
                    error_writer.commit
                )
                current.error_file_id = error_file.id
            else:
                await self._run_blocking_cancellation_safe(error_writer.abort)
            ensure_claim()
            await self._finish(
                current,
                "completed",
                claim_state[0],
                terminal_published,
            )
        except asyncio.CancelledError:
            if not terminal_published[0]:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._rollback_writer(output_writer)
                with contextlib.suppress(asyncio.CancelledError):
                    await self._rollback_writer(error_writer)
            raise
        except (_BatchClaimLeaseLost, StaleBatchClaimError):
            await self._rollback_writer(output_writer)
            await self._rollback_writer(error_writer)
            logger.info(
                "batch claim was lost before publication",
                extra={"batch_id": batch_id, "worker_id": self._worker_id},
            )
            return
        except Exception as error:
            await self._rollback_writer(output_writer)
            await self._rollback_writer(error_writer)
            if claim_lost.is_set():
                logger.info(
                    "batch claim lease renewal failed; leaving the job for reclaim",
                    extra={"batch_id": batch_id, "worker_id": self._worker_id},
                )
                return
            current = await asyncio.to_thread(self._store.get_batch, batch_id)
            current.request_counts = job.request_counts.model_copy()
            current.output_file_id = None
            current.error_file_id = None
            if current.status == "cancelled":
                if claim is None:
                    current.cancelled_at = current.cancelled_at or int(time.time())
                    await self._run_blocking_cancellation_safe(
                        self._store.update_batch,
                        current,
                    )
                if self._metrics is not None:
                    self._metrics.batch_jobs_total.labels(state="cancelled").inc()
                return
            current.errors = {
                "message": (
                    f"batch processing failed ({self._failure_class(error)}); resubmit"
                )
            }
            try:
                await self._finish(
                    current,
                    "failed",
                    claim_state[0],
                    terminal_published,
                )
            except StaleBatchClaimError:
                logger.info(
                    "batch claim was lost while recording failure",
                    extra={"batch_id": batch_id, "worker_id": self._worker_id},
                )
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
