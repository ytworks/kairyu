"""In-gateway batch worker: drains jobs through the served engines (m7 D7).

The worker's own semaphore caps its concurrency strictly below the server's
global guard so interactive latency is protected (goal G3 gate C4). Requests
go through the same engines mapping the HTTP path uses — a pool member is a
pool member whether the request arrived interactively or from a batch file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Mapping

from kairyu.batch.store import BatchJob, BatchStore
from kairyu.engine.backend import EngineBackend, GenerationRequest
from kairyu.entrypoints.chat_template import render_chat
from kairyu.entrypoints.server.app import completion_response, sampling_params_from
from kairyu.entrypoints.server.protocol import ChatCompletionRequest

logger = logging.getLogger("kairyu.batch")


class BatchWorker:
    def __init__(
        self,
        store: BatchStore,
        engines: Mapping[str, EngineBackend],
        max_concurrency: int = 4,
        metrics=None,
    ) -> None:
        self._store = store
        self._engines = engines
        self._max_concurrency = max_concurrency
        self._metrics = metrics
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

    async def _run_line(self, line: dict) -> tuple[dict | None, dict | None]:
        """Execute one input line; returns (output_line, error_line)."""
        custom_id = line.get("custom_id")
        try:
            request = ChatCompletionRequest.model_validate(line["body"])
            engine = self._engines.get(request.model)
            if engine is None:
                raise ValueError(f"model {request.model!r} not found")
            prompt = render_chat([message.model_dump() for message in request.messages])
            result = await engine.generate(
                GenerationRequest(
                    request_id=f"batch-{uuid.uuid4().hex[:12]}",
                    prompt=prompt,
                    sampling_params=sampling_params_from(request),
                )
            )
            texts = [(c.text, c.finish_reason) for c in result.completions]
            response = completion_response(request, prompt, texts)
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

    async def process(self, batch_id: str) -> None:
        job = self._store.get_batch(batch_id)
        if job.status == "cancelled":
            return
        try:
            raw = self._store.read_file_content(job.input_file_id)
            lines = [
                json.loads(line)
                for line in raw.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
            job.errors = {"message": f"invalid input file: {error}"}
            self._finish(job, "failed")
            return

        job.status = "in_progress"
        job.in_progress_at = int(time.time())
        job.request_counts.total = len(lines)
        self._store.update_batch(job)

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def guarded(line: dict) -> tuple[dict | None, dict | None] | None:
            async with semaphore:
                if self._cancelled(batch_id):
                    return None
                return await self._run_line(line)

        results = await asyncio.gather(*(guarded(line) for line in lines))

        job = self._store.get_batch(batch_id)  # re-read: cancel may have landed
        if job.status == "cancelled":
            job.cancelled_at = job.cancelled_at or int(time.time())
            self._store.update_batch(job)
            if self._metrics is not None:
                self._metrics.batch_jobs_total.labels(state="cancelled").inc()
            return

        outputs = [output for output, _ in filter(None, results) if output]
        errors = [error for _, error in filter(None, results) if error]
        if outputs:
            output_file = self._store.save_file(
                "\n".join(json.dumps(line) for line in outputs).encode("utf-8"),
                filename=f"{batch_id}_output.jsonl",
                purpose="batch_output",
            )
            job.output_file_id = output_file.id
        if errors:
            error_file = self._store.save_file(
                "\n".join(json.dumps(line) for line in errors).encode("utf-8"),
                filename=f"{batch_id}_errors.jsonl",
                purpose="batch_output",
            )
            job.error_file_id = error_file.id
        job.request_counts.completed = len(outputs)
        job.request_counts.failed = len(errors)
        self._finish(job, "completed")
