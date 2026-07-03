"""OpenAI-compatible /v1/files and /v1/batches routes (design m7 D7)."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker


class CreateBatchRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: dict | None = None


def _not_found(kind: str, object_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"{kind} {object_id!r} not found",
                "type": "invalid_request_error",
                "code": f"{kind}_not_found",
            }
        },
    )


def add_batch_routes(app: FastAPI, store: BatchStore, worker: BatchWorker) -> None:
    @app.post("/v1/files")
    async def upload_file(
        file: Annotated[UploadFile, File()], purpose: Annotated[str, Form()]
    ):
        content = await file.read()
        return store.save_file(content, filename=file.filename or "upload", purpose=purpose)

    @app.get("/v1/files/{file_id}")
    async def get_file(file_id: str):
        try:
            return store.get_file(file_id)
        except KeyError:
            return _not_found("file", file_id)

    @app.get("/v1/files/{file_id}/content")
    async def get_file_content(file_id: str):
        try:
            content = store.read_file_content(file_id)
        except KeyError:
            return _not_found("file", file_id)
        return Response(content=content, media_type="application/octet-stream")

    @app.post("/v1/batches")
    async def create_batch(request: CreateBatchRequest):
        try:
            job = store.create_batch(
                input_file_id=request.input_file_id,
                endpoint=request.endpoint,
                completion_window=request.completion_window,
                metadata=request.metadata,
            )
        except KeyError:
            return _not_found("file", request.input_file_id)
        except ValueError as error:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": str(error),
                        "type": "invalid_request_error",
                        "code": "unsupported_endpoint",
                    }
                },
            )
        worker.submit(job.id)
        return job

    @app.get("/v1/batches")
    async def list_batches(limit: int = 20):
        return {"object": "list", "data": [job.model_dump() for job in store.list_batches(limit)]}

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(batch_id: str):
        try:
            return store.get_batch(batch_id)
        except KeyError:
            return _not_found("batch", batch_id)

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str):
        try:
            job = store.get_batch(batch_id)
        except KeyError:
            return _not_found("batch", batch_id)
        if job.status in ("validating", "in_progress"):
            job.status = "cancelled"
            store.update_batch(job)
        return store.get_batch(batch_id)
