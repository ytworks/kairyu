"""/v1/embeddings plus registration of the Responses API adapter.

Embeddings: ``EmbeddingBackend`` protocol; base64 is the OpenAI SDK's
DEFAULT encoding_format — both float and base64 are served. Responses lives in
``responses_service`` so its typed SSE and tool-call state machine remain
separate from embeddings.
"""

from __future__ import annotations

import base64
import struct
from collections.abc import Mapping
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kairyu.entrypoints.server.errors import model_not_found
from kairyu.entrypoints.server.metering import record_state_usage
from kairyu.entrypoints.server.responses_service import (
    ResponsesRequest,
    ResponseStore,
    add_responses_route,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingsRequest",
    "MockEmbeddingBackend",
    "ResponseStore",
    "ResponsesRequest",
    "add_extra_routes",
]

_MAX_EMBEDDING_INPUTS = 2048  # cap the embeddings batch (M6)


class EmbeddingBackend(Protocol):
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbeddingBackend:
    """Deterministic hash-based unit vectors (CPU tests, wire-format truth)."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        vectors: list[list[float]] = []
        for text in texts:
            values = []
            counter = 0
            while len(values) < self.dimensions:
                digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
                values.extend(b / 255.0 - 0.5 for b in digest)
                counter += 1
            norm = sum(v * v for v in values[: self.dimensions]) ** 0.5 or 1.0
            vectors.append([v / norm for v in values[: self.dimensions]])
        return vectors


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"  # the SDK sends base64 by default (A9)


def add_extra_routes(
    app: FastAPI,
    engines,
    *,
    embedding_backends: Mapping[str, EmbeddingBackend] | None = None,
    chat_templates=None,
) -> None:
    add_responses_route(app, engines, chat_templates=chat_templates)

    if embedding_backends:

        @app.post("/v1/embeddings")
        async def embeddings(request: EmbeddingsRequest, http_request: Request):
            backend = embedding_backends.get(request.model)
            if backend is None:
                return model_not_found(request.model)
            resolved_model = request.model
            http_request.state.model = resolved_model
            texts = [request.input] if isinstance(request.input, str) else request.input
            if not texts:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "input must not be empty",
                                       "type": "invalid_request_error", "code": None}},
                )
            if len(texts) > _MAX_EMBEDDING_INPUTS:  # M6: bound the batch
                return JSONResponse(
                    status_code=400,
                    content={"error": {
                        "message": f"input exceeds {_MAX_EMBEDDING_INPUTS} items",
                        "type": "invalid_request_error", "code": None}},
                )
            if request.encoding_format not in ("float", "base64"):  # M6: no silent fallthrough
                return JSONResponse(
                    status_code=400,
                    content={"error": {
                        "message": "encoding_format must be 'float' or 'base64'",
                        "type": "invalid_request_error", "code": None}},
                )
            vectors = await backend.embed(texts)
            data = []
            for index, vector in enumerate(vectors):
                if request.encoding_format == "base64":  # SDK default (A9)
                    packed = struct.pack(f"<{len(vector)}f", *vector)
                    payload = base64.b64encode(packed).decode()
                else:
                    payload = vector
                data.append({"object": "embedding", "index": index, "embedding": payload})
            prompt_tokens = sum(len(text.split()) for text in texts)
            response = {
                "object": "list",
                "data": data,
                "model": resolved_model,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }
            owner = getattr(http_request.state, "tenant", None) or "default"
            record_state_usage(
                http_request.app.state,
                tenant=owner,
                model=resolved_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
            )
            return response
