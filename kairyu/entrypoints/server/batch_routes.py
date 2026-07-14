"""OpenAI-compatible /v1/files and /v1/batches routes (design m7 D7)."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kairyu.batch.store import BatchStore, FileTooLargeError
from kairyu.batch.worker import BatchWorker


def _tenant_of(request: Request) -> str:
    """Owning tenant for this request (C3), or "default" in keyless mode."""
    return request.scope.get("state", {}).get("tenant", "default")


# cap the batch input upload so one client cannot OOM the gateway (S7)
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024


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
        request: Request,
        file: Annotated[UploadFile, File()],
        purpose: Annotated[str, Form()],
    ):
        async def chunks():
            while chunk := await file.read(_CHUNK_BYTES):
                yield chunk

        try:
            return await store.save_file_streaming(
                chunks(),
                filename=file.filename or "upload",
                purpose=purpose,
                owner=_tenant_of(request),
                max_bytes=_MAX_UPLOAD_BYTES,
            )
        except FileTooLargeError:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"file exceeds the {_MAX_UPLOAD_BYTES}-byte upload limit",
                        "type": "invalid_request_error",
                        "code": "file_too_large",
                    }
                },
            )

    @app.get("/v1/files/{file_id}")
    async def get_file(request: Request, file_id: str):
        try:
            return store.get_file(file_id, owner=_tenant_of(request))
        except KeyError:
            return _not_found("file", file_id)

    @app.get("/v1/files/{file_id}/content")
    async def get_file_content(request: Request, file_id: str):
        try:
            content = store.read_file_content(file_id, owner=_tenant_of(request))
        except KeyError:
            return _not_found("file", file_id)
        return Response(content=content, media_type="application/octet-stream")

    @app.post("/v1/batches")
    async def create_batch(request: Request, body: CreateBatchRequest):
        try:
            job = store.create_batch(
                input_file_id=body.input_file_id,
                endpoint=body.endpoint,
                completion_window=body.completion_window,
                metadata=body.metadata,
                owner=_tenant_of(request),
            )
        except KeyError:
            return _not_found("file", body.input_file_id)
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
    async def list_batches(request: Request, limit: int = 20):
        owner = _tenant_of(request)
        jobs = store.list_batches(limit, owner=owner)
        return {"object": "list", "data": [job.model_dump() for job in jobs]}

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(request: Request, batch_id: str):
        try:
            return store.get_batch(batch_id, owner=_tenant_of(request))
        except KeyError:
            return _not_found("batch", batch_id)

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(request: Request, batch_id: str):
        owner = _tenant_of(request)
        try:
            job = store.get_batch(batch_id, owner=owner)
        except KeyError:
            return _not_found("batch", batch_id)
        if job.status in ("validating", "in_progress"):
            job.status = "cancelled"
            store.update_batch(job)
        return store.get_batch(batch_id, owner=owner)
