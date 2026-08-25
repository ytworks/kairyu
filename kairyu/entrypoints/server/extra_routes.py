"""/v1/embeddings plus registration of the Responses API adapter.

Embeddings: ``EmbeddingBackend`` protocol; base64 is the OpenAI SDK's
DEFAULT encoding_format — both float and base64 are served. Responses lives in
``responses_service`` so its typed SSE and tool-call state machine remain
separate from embeddings.
"""

from __future__ import annotations

import base64
import math
import struct
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kairyu.engine.embedding import (
    EmbeddingBackend,
    EmbeddingCapacityError,
    EmbeddingResult,
    MockEmbeddingBackend,
)
from kairyu.entrypoints.server.errors import model_not_found, upstream_error
from kairyu.entrypoints.server.messages_service import (
    add_hello_probe,
    add_messages_route,
)
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
_MAX_EMBEDDING_UTF8_BYTES = 8 * 1024 * 1024


def _embedding_capacity_error() -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "1"},
        content={
            "error": {
                "message": "embedding backend is at max concurrency",
                "type": "rate_limit_error",
                "code": "embedding_concurrency_exceeded",
            }
        },
    )


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
    legacy_chat_models: AbstractSet[str] | None = None,
    orchestrated_models: AbstractSet[str] | None = None,
    chat_dispatch=None,
    responses_compaction_key: bytes,
) -> None:
    add_responses_route(
        app,
        engines,
        chat_templates=chat_templates,
        legacy_chat_models=legacy_chat_models,
        orchestrated_models=orchestrated_models,
        chat_dispatch=chat_dispatch,
        compaction_key=responses_compaction_key,
    )

    add_messages_route(
        app,
        engines,
        chat_templates=chat_templates,
        legacy_chat_models=legacy_chat_models,
        orchestrated_models=orchestrated_models,
        chat_dispatch=chat_dispatch,
    )
    add_hello_probe(app)

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
            input_bytes = sum(len(text.encode()) for text in texts)
            if input_bytes > _MAX_EMBEDDING_UTF8_BYTES:
                return JSONResponse(
                    status_code=400,
                    content={"error": {
                        "message": (
                            "input exceeds "
                            f"{_MAX_EMBEDDING_UTF8_BYTES} UTF-8 bytes"
                        ),
                        "type": "invalid_request_error",
                        "code": None,
                    }},
                )
            if request.encoding_format not in ("float", "base64"):  # M6: no silent fallthrough
                return JSONResponse(
                    status_code=400,
                    content={"error": {
                        "message": "encoding_format must be 'float' or 'base64'",
                        "type": "invalid_request_error", "code": None}},
                )
            owner = getattr(http_request.state, "tenant", None) or "default"
            admission = getattr(http_request.state, "tenant_admission", None)
            if admission is not None:
                # Reserve before tokenization/dispatch. Production backends
                # that return exact tokenizer usage may refund the difference;
                # mocks and legacy estimates retain the conservative bound.
                work_bound = sum(max(1, len(text.encode()) + 32) for text in texts)
                admitted = admission.reserve_tokens(
                    work_bound,
                    refundable_on_exact_usage=bool(
                        getattr(backend, "reports_exact_usage", False)
                    ),
                )
                metrics = getattr(http_request.app.state, "metrics", None)
                if metrics is not None:
                    metrics.record_tenant_admission(
                        owner,
                        source="http",
                        admitted=admitted,
                        reason=admission.reason,
                    )
                if not admitted:
                    return JSONResponse(
                        status_code=429,
                        headers={"Retry-After": "1"},
                        content={
                            "error": {
                                "message": (
                                    f"tenant {owner!r} admission limit exceeded "
                                    f"({admission.reason})"
                                ),
                                "type": "rate_limit_error",
                                "code": "tenant_rate_limited",
                            }
                        },
                    )
                http_request.state.tenant_metric_admitted = True
            try:
                # FastEmbed reserves its bounded execution slot synchronously.
                # Capacity rejection therefore remains pre-dispatch and the
                # tenant middleware refunds the conservative token reservation.
                work = backend.embed(texts)
            except EmbeddingCapacityError:
                return _embedding_capacity_error()
            except Exception as error:
                return upstream_error(error)
            if admission is not None:
                admission.mark_dispatched()
            try:
                result = await work
                if not isinstance(result, EmbeddingResult):
                    raise TypeError(
                        "embedding backend returned an unsupported result type"
                    )
                if len(result.vectors) != len(texts):
                    raise ValueError(
                        "embedding backend returned the wrong vector count"
                    )
                for vector in result.vectors:
                    if len(vector) != backend.dimensions:
                        raise ValueError(
                            "embedding backend returned the wrong dimensions"
                        )
                    if not all(math.isfinite(value) for value in vector):
                        raise ValueError(
                            "embedding backend returned a non-finite value"
                        )
                if result.prompt_tokens < 0:
                    raise ValueError(
                        "embedding backend returned negative token usage"
                    )
            except EmbeddingCapacityError:
                return _embedding_capacity_error()
            except Exception as error:
                return upstream_error(error)
            data = []
            for index, vector in enumerate(result.vectors):
                if request.encoding_format == "base64":  # SDK default (A9)
                    packed = struct.pack(f"<{len(vector)}f", *vector)
                    payload = base64.b64encode(packed).decode()
                else:
                    payload = vector
                data.append({"object": "embedding", "index": index, "embedding": payload})
            prompt_tokens = result.prompt_tokens
            response = {
                "object": "list",
                "data": data,
                "model": resolved_model,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }
            record_state_usage(
                http_request.app.state,
                tenant=owner,
                model=resolved_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                reservation=admission,
                usage_exact=result.usage_exact,
            )
            return response
