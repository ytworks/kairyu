"""In-gateway batch worker: drains jobs through the served engines (m7 D7).

The worker's fixed consumer pool caps its concurrency strictly below the
server's global guard so interactive latency is protected (goal G3 gate C4).
Requests go through the same engines mapping the HTTP path uses — a pool member
is a pool member whether the request arrived interactively or from a batch file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Mapping

from kairyu.batch.store import BatchJob, BatchStore, JsonlFileWriter
from kairyu.engine.backend import EngineBackend, GenerationRequest
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server.app import (
    completion_response,
    render_prompt,
    sampling_params_from,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest

logger = logging.getLogger("kairyu.batch")
_INPUT_QUEUE_FACTOR = 2
_INPUT_SENTINEL = object()


class BatchWorker:
    def __init__(
        self,
        store: BatchStore,
        engines: Mapping[str, EngineBackend],
        max_concurrency: int = 4,
        metrics=None,
        chat_templates: Mapping[str, ChatTemplate] | None = None,
    ) -> None:
        self._store = store
        self._engines = engines
        self._max_concurrency = max_concurrency
        self._metrics = metrics
        self._chat_templates = chat_templates
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    def submit(self, batch_id: str) -> None:
        self._queue.put_nowait(batch_id)

    async def run(self) -> None:
        """Job loop; cancelled by the app lifespan on shutdown."""
        while True:
            batch_id = await self._queue.get()
            try:
                await self.process(batch_id)
            except Exception:
                logger.exception("batch job crashed", extra={"batch_id": batch_id})

    def _cancelled(self, batch_id: str) -> bool:
        return self._store.get_batch(batch_id).status == "cancelled"

    def _finish(self, job: BatchJob, state: str) -> None:
        job.status = state
        setattr(job, f"{state}_at", int(time.time()))
        self._store.update_batch(job)
        if self._metrics is not None:
            self._metrics.batch_jobs_total.labels(state=state).inc()

    async def _run_line(self, line: object) -> tuple[dict | None, dict | None]:
        """Execute one input line; returns (output_line, error_line)."""
        # a line that is valid JSON but not an object (e.g. a bare `5`) must
        # become a per-line error record, never escape the pipeline and wedge
        # the whole job in "in_progress" forever (S1)
        custom_id = line.get("custom_id") if isinstance(line, dict) else None
        try:
            if not isinstance(line, dict):
                raise ValueError("input line is not a JSON object")
            request = ChatCompletionRequest.model_validate(line["body"])
            engine = self._engines.get(request.model)
            if engine is None:
                raise ValueError(f"model {request.model!r} not found")
            prompt = render_prompt(request, self._chat_templates)
            result = await engine.generate(
                GenerationRequest(
                    request_id=f"batch-{uuid.uuid4().hex[:12]}",
                    prompt=prompt,
                    sampling_params=sampling_params_from(request),
                )
            )
            response = completion_response(request, prompt, result.completions, result.usage)
            return (
                {
                    "id": f"batch_req_{uuid.uuid4().hex[:16]}",
                    "custom_id": custom_id,
                    "response": {"status_code": 200, "body": response.model_dump()},
                    "error": None,
                },
                None,
            )
        except Exception as error:
            return (
                None,
                {
                    "id": f"batch_req_{uuid.uuid4().hex[:16]}",
                    "custom_id": custom_id,
                    "error": {"message": str(error), "type": "batch_request_error"},
                },
            )

    @staticmethod
    def _failure_class(error: BaseException) -> str:
        if isinstance(error, BaseExceptionGroup) and error.exceptions:
            return BatchWorker._failure_class(error.exceptions[0])
        return type(error).__name__

    @staticmethod
    def _rollback_writer(writer: JsonlFileWriter | None) -> None:
        if writer is None:
            return
        try:
            writer.rollback()
        except Exception:
            logger.exception("failed to roll back batch spool")

    async def process(self, batch_id: str) -> None:
        job = self._store.get_batch(batch_id)
        if job.status == "cancelled":
            return
        job.status = "in_progress"
        job.in_progress_at = int(time.time())
        self._store.update_batch(job)
        output_writer: JsonlFileWriter | None = None
        error_writer: JsonlFileWriter | None = None
        try:
            input_queue: asyncio.Queue[object] = asyncio.Queue(
                maxsize=max(1, self._max_concurrency * _INPUT_QUEUE_FACTOR)
            )
            input_error: Exception | None = None
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

            async def produce() -> None:
                nonlocal input_error
                try:
                    input_lines = self._store.iter_file_lines(
                        job.input_file_id, owner=job.owner
                    )
                except KeyError as error:
                    input_error = error
                else:
                    try:
                        for raw_line in input_lines:
                            if self._cancelled(batch_id):
                                break
                            if not raw_line.strip():
                                continue
                            try:
                                line = json.loads(raw_line.decode("utf-8"))
                            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                                input_error = error
                                break
                            job.request_counts.total += 1
                            await input_queue.put(line)
                    finally:
                        close = getattr(input_lines, "close", None)
                        if close is not None:
                            close()
                for _ in range(self._max_concurrency):
                    await input_queue.put(_INPUT_SENTINEL)

            async def consume() -> None:
                while True:
                    line = await input_queue.get()
                    try:
                        if line is _INPUT_SENTINEL:
                            return
                        if input_error is not None or self._cancelled(batch_id):
                            continue
                        output, error = await self._run_line(line)
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

            current = self._store.get_batch(batch_id)  # cancellation may have landed
            current.request_counts = job.request_counts.model_copy()
            if current.status == "cancelled":
                self._rollback_writer(output_writer)
                self._rollback_writer(error_writer)
                current.cancelled_at = current.cancelled_at or int(time.time())
                self._store.update_batch(current)
                if self._metrics is not None:
                    self._metrics.batch_jobs_total.labels(state="cancelled").inc()
                return
            if input_error is not None:
                self._rollback_writer(output_writer)
                self._rollback_writer(error_writer)
                current.errors = {"message": f"invalid input file: {input_error}"}
                self._finish(current, "failed")
                return

            if output_writer.has_content:
                current.output_file_id = output_writer.commit().id
            else:
                output_writer.abort()
            if error_writer.has_content:
                current.error_file_id = error_writer.commit().id
            else:
                error_writer.abort()
            self._finish(current, "completed")
        except asyncio.CancelledError:
            self._rollback_writer(output_writer)
            self._rollback_writer(error_writer)
            raise
        except Exception as error:
            self._rollback_writer(output_writer)
            self._rollback_writer(error_writer)
            current = self._store.get_batch(batch_id)
            current.request_counts = job.request_counts.model_copy()
            current.output_file_id = None
            current.error_file_id = None
            if current.status == "cancelled":
                current.cancelled_at = current.cancelled_at or int(time.time())
                self._store.update_batch(current)
                if self._metrics is not None:
                    self._metrics.batch_jobs_total.labels(state="cancelled").inc()
                return
            current.errors = {
                "message": (
                    f"batch processing failed ({self._failure_class(error)}); resubmit"
                )
            }
            self._finish(current, "failed")
