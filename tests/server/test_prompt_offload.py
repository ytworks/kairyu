import asyncio
import json
import threading

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from kairyu.engine.backend import AdmissionUpperBound
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.chat_template import ChatTemplate
from kairyu.entrypoints.server import app as server_app
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


async def test_blocked_chat_render_leaves_event_loop_responsive():
    render_started = threading.Event()
    release_render = threading.Event()
    render_threads: list[int] = []
    validation_threads: list[int] = []
    template = ChatTemplate("{{ messages[0].content }}")
    original_render = template.render

    def blocking_render(*args, **kwargs):
        render_threads.append(threading.get_ident())
        render_started.set()
        assert release_render.wait(2)
        return original_render(*args, **kwargs)

    template.render = blocking_render

    class ThreadRecordingBackend(MockBackend):
        def validate_request(self, request):
            validation_threads.append(threading.get_ident())
            super().validate_request(request)

    app = create_app(
        {"m": ThreadRecordingBackend()},
        chat_templates={"m": template},
    )
    loop_thread = threading.get_ident()
    watchdog = threading.Timer(1, release_render.set)
    watchdog.start()
    try:
        async with _client(app) as client:
            request_task = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )
            assert await asyncio.to_thread(render_started.wait, 0.5)
            # If render still owned the event loop, the watchdog would be the
            # only code able to release it and this assertion would fail.
            assert not release_render.is_set()
            health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
            assert health.status_code == 200
            release_render.set()
            response = await request_task
    finally:
        release_render.set()
        watchdog.cancel()

    assert response.status_code == 200
    assert render_threads and set(render_threads) != {loop_thread}
    # Generic validators can inspect ReplicaPool membership and stay owned by
    # the event loop; only pure prompt work crosses the thread boundary.
    assert validation_threads and set(validation_threads) == {loop_thread}


async def test_blocked_body_validation_leaves_event_loop_responsive(monkeypatch):
    validation_started = threading.Event()
    release_validation = threading.Event()
    validation_threads: list[int] = []
    original = server_app._prepare_request_body

    def blocking_validation(*args, **kwargs):
        validation_threads.append(threading.get_ident())
        validation_started.set()
        assert release_validation.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(server_app, "_prepare_request_body", blocking_validation)
    app = create_app({"m": MockBackend()}, legacy_chat_models={"m"})
    loop_thread = threading.get_ident()
    watchdog = threading.Timer(1, release_validation.set)
    watchdog.start()
    try:
        async with _client(app) as client:
            request_task = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                )
            )
            assert await asyncio.to_thread(validation_started.wait, 0.5)
            assert not release_validation.is_set()
            health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
            assert health.status_code == 200
            release_validation.set()
            response = await request_task
    finally:
        release_validation.set()
        watchdog.cancel()

    assert response.status_code == 200
    assert validation_threads and set(validation_threads) != {loop_thread}


def _reference_app() -> FastAPI:
    app = FastAPI()

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
    )
    async def chat_completions(request: ChatCompletionRequest):
        return request

    return app


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (b'{"model":"m"}', {"content-type": "application/json"}),
        (b'{"model":', {"content-type": "application/json"}),
        (b"\xff", {"content-type": "application/json"}),
        (b"", {"content-type": "application/json"}),
        (b"null", {"content-type": "application/json"}),
        (b"[]", {"content-type": "application/json"}),
        (
            b'{"model":"m","messages":[{"role":"user","content":"x"}]}',
            {"content-type": "text/plain"},
        ),
        (
            b'{"model":"m","messages":[{"role":"user","content":"x"}]}',
            {},
        ),
    ],
)
async def test_body_validation_preserves_fastapi_error_wire(body, headers):
    app = create_app({"m": MockBackend()}, legacy_chat_models={"m"})
    async with _client(_reference_app()) as reference_client:
        expected = await reference_client.post(
            "/v1/chat/completions",
            content=body,
            headers=headers,
        )
    async with _client(app) as client:
        actual = await client.post(
            "/v1/chat/completions",
            content=body,
            headers=headers,
        )

    assert actual.status_code == expected.status_code
    assert actual.json() == expected.json()


async def test_vendor_json_content_type_remains_supported():
    body = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()
    app = create_app({"m": MockBackend()}, legacy_chat_models={"m"})
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/vnd.kairyu+json"},
        )

    assert response.status_code == 200


async def test_typed_request_keeps_plain_dict_json_contract():
    app = FastAPI()
    app.router.route_class = server_app._OffloadedRequestBodyRoute

    @app.post("/probe")
    async def probe(body: ChatCompletionRequest, request: Request):
        parsed = await request.json()
        return {
            "plain_dict": type(parsed) is dict,
            "model": body.model,
            "parsed_model": parsed["model"],
        }

    async with _client(app) as client:
        response = await client.post(
            "/probe",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "plain_dict": True,
        "model": "m",
        "parsed_model": "m",
    }


async def test_custom_request_validation_handler_receives_parsed_body():
    app = FastAPI()
    app.router.route_class = server_app._OffloadedRequestBodyRoute

    @app.exception_handler(RequestValidationError)
    async def custom_handler(_request, error):
        return JSONResponse(
            status_code=499,
            content={
                "body_type": type(error.body).__name__,
                "function": error.endpoint_ctx.get("function"),
                "path": error.endpoint_ctx.get("path"),
                "cause": type(error.__cause__).__name__,
                "context": type(error.__context__).__name__,
            },
        )

    @app.post("/probe")
    async def probe(body: ChatCompletionRequest):
        return body

    async with _client(app) as client:
        response = await client.post(
            "/probe",
            json={"model": "m"},
        )
        malformed = await client.post(
            "/probe",
            content=b'{"model":',
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 499
    assert response.json() == {
        "body_type": "dict",
        "function": "probe",
        "path": "POST /probe",
        "cause": "NoneType",
        "context": "NoneType",
    }
    assert malformed.status_code == 499
    assert malformed.json() == {
        "body_type": "str",
        "function": "probe",
        "path": "POST /probe",
        "cause": "JSONDecodeError",
        "context": "JSONDecodeError",
    }


async def test_custom_body_parse_handler_receives_original_exception_cause():
    app = FastAPI()
    app.router.route_class = server_app._OffloadedRequestBodyRoute

    @app.exception_handler(StarletteHTTPException)
    async def custom_handler(_request, error):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "cause": type(error.__cause__).__name__,
                "context": type(error.__context__).__name__,
            },
        )

    @app.post("/probe")
    async def probe(body: ChatCompletionRequest):
        return body

    async with _client(app) as client:
        response = await client.post(
            "/probe",
            content=b"\xff",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "cause": "UnicodeDecodeError",
        "context": "UnicodeDecodeError",
    }


def test_chat_and_route_openapi_keep_typed_request_contracts():
    app = create_app({"m": MockBackend()}, legacy_chat_models={"m"})
    schema = app.openapi()

    chat_operation = schema["paths"]["/v1/chat/completions"]["post"]
    route_operation = schema["paths"]["/v1/route"]["post"]
    assert chat_operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ChatCompletionRequest"
    }
    assert route_operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RoutePreviewRequest"
    }
    assert "422" in chat_operation["responses"]
    assert "422" in route_operation["responses"]
    schemas = schema["components"]["schemas"]
    for name in (
        "ChatCompletionRequest",
        "RoutePreviewRequest",
        "ChatMessage",
        "HTTPValidationError",
        "ValidationError",
    ):
        assert name in schemas


async def test_completions_prepares_before_backend_admission_bound():
    events: list[str] = []

    class OrderedBackend(MockBackend):
        async def prepare_request(self, request):
            events.append("prepare")

        async def admission_upper_bound_async(self, request):
            assert events == ["prepare"]
            events.append("bound")
            return AdmissionUpperBound(1000, True)

        async def generate(self, request):
            events.append("generate")
            return await super().generate(request)

    app = create_app({"m": OrderedBackend()})
    async with _client(app) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "m", "prompt": "hello", "max_tokens": 1},
        )

    assert response.status_code == 200
    assert events == ["prepare", "bound", "generate"]
