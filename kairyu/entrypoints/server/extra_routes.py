"""/v1/embeddings and /v1/responses (m11 D4, amendments A8/A9).

Embeddings: ``EmbeddingBackend`` protocol; base64 is the OpenAI SDK's
DEFAULT encoding_format — both float and base64 are served. Responses: the
reviewed subset (input str|messages, instructions, previous_response_id,
store) with the EXACT output-item shapes ``response.output_text`` needs;
usage names are input/output/total_tokens (NOT prompt/completion). Stream is
descoped (the typed response.* event protocol is its own milestone).
"""

from __future__ import annotations

import base64
import struct
import time
import uuid
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


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


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[dict] = ""
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool = True
    max_output_tokens: int | None = None
    stream: bool = False
    metadata: dict = Field(default_factory=dict)


class ResponseStore:
    """In-memory previous_response_id state (protocol-shaped for M11+)."""

    def __init__(self) -> None:
        self._items: dict[str, list[dict]] = {}

    def save(self, response_id: str, items: list[dict]) -> None:
        self._items[response_id] = items

    def get(self, response_id: str) -> list[dict] | None:
        return self._items.get(response_id)


def _input_items(payload: str | list[dict]) -> list[dict]:
    if isinstance(payload, str):
        return [{"type": "message", "role": "user", "content": payload}]
    return list(payload)


def _item_text(item: dict) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
            parts.append(part.get("text", ""))
    return "".join(parts)


def add_extra_routes(
    app: FastAPI,
    engines,
    *,
    embedding_backend=None,
    chat_templates=None,
) -> None:
    store = ResponseStore()
    app.state.response_store = store

    if embedding_backend is not None:

        @app.post("/v1/embeddings")
        async def embeddings(request: EmbeddingsRequest):
            texts = [request.input] if isinstance(request.input, str) else request.input
            if not texts:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "input must not be empty",
                                       "type": "invalid_request_error", "code": None}},
                )
            vectors = await embedding_backend.embed(texts)
            data = []
            for index, vector in enumerate(vectors):
                if request.encoding_format == "base64":  # SDK default (A9)
                    packed = struct.pack(f"<{len(vector)}f", *vector)
                    payload = base64.b64encode(packed).decode()
                else:
                    payload = vector
                data.append({"object": "embedding", "index": index, "embedding": payload})
            prompt_tokens = sum(len(text.split()) for text in texts)
            return {
                "object": "list",
                "data": data,
                "model": request.model,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest, http_request: Request):
        if request.stream:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "responses streaming is not supported yet",
                                   "type": "invalid_request_error", "code": None}},
            )
        engine = engines.get(request.model)
        if engine is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": f"model {request.model!r} not found",
                                   "type": "invalid_request_error",
                                   "code": "model_not_found"}},
            )
        context: list[dict] = []
        if request.previous_response_id:
            previous = store.get(request.previous_response_id)
            if previous is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": {"message": "previous response not found",
                                       "type": "invalid_request_error", "code": None}},
                )
            context.extend(previous)
        context.extend(_input_items(request.input))

        lines = []
        if request.instructions:
            lines.append(f"system: {request.instructions}")
        for item in context:
            lines.append(f"{item.get('role', 'user')}: {_item_text(item)}")
        lines.append("assistant:")
        prompt = "\n".join(lines)

        from kairyu.engine.backend import GenerationRequest
        from kairyu.sampling_params import SamplingParams

        generation = GenerationRequest(
            request_id=f"resp-{uuid.uuid4().hex[:12]}",
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=request.max_output_tokens or 1024
            ),
        )
        result = await engine.generate(generation)
        response_id = f"resp_{uuid.uuid4().hex}"
        output_item = {
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "status": "completed",
            # exact shape response.output_text is computed from (A8)
            "content": [{"type": "output_text", "text": result.text, "annotations": []}],
        }
        if request.store:
            store.save(
                response_id,
                context + [{"type": "message", "role": "assistant", "content": result.text}],
            )
        usage = result.usage
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": request.model,
            "output": [output_item],
            "previous_response_id": request.previous_response_id,
            "metadata": request.metadata,
            # A8: Responses usage names differ from chat completions
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": (usage.prompt_tokens + usage.completion_tokens) if usage else 0,
            },
        }
