from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.mock import MockBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.entrypoints.server.middleware import TracingMiddleware
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.replica import ReplicaPool
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams
from kairyu.telemetry import configure_tracing
from tests.server._legacy_chat import create_legacy_app


@pytest.fixture
def trace_recorder():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield provider, exporter
    finally:
        configure_tracing(False)
        provider.shutdown()


def _span_payload(spans) -> str:
    return json.dumps(
        [
            {
                "name": span.name,
                "attributes": dict(span.attributes),
                "events": [
                    {"name": event.name, "attributes": dict(event.attributes)}
                    for event in span.events
                ],
                "status": span.status.status_code.name,
                "description": span.status.description,
            }
            for span in spans
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


async def test_one_request_correlates_gateway_conductor_pool_and_remote_replica(
    trace_recorder,
):
    provider, exporter = trace_recorder
    secret_prompt = (
        "otel-prompt-canary: First research. Then plan. "
        "After that implement. Finally verify."
    )
    settings = ServerSettings(tracing=True, access_log=False, metrics=False)
    replica_app = create_legacy_app(
        engines={"llama": MockBackend()},
        settings=settings,
    )
    remote = OpenAICompatBackend(
        base_url="http://replica/v1",
        model="llama",
        api_key_env=None,
        transport=httpx.ASGITransport(app=replica_app),
        upstream="kairyu",
    )
    pool = ReplicaPool({"replica-1": remote})
    gateway_app = create_legacy_app(
        engines={},
        orchestrators={
            "auto": Orchestrator(
                {"tier1": pool, "tier2": pool},
            )
        },
        settings=settings,
    )
    # App construction enables tracing without taking ownership of a global
    # provider. The fixture injects one private deterministic provider for both
    # process-shaped ASGI services.
    configure_tracing(True, tracer_provider=provider)

    async with replica_app.router.lifespan_context(replica_app):
        async with gateway_app.router.lifespan_context(gateway_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway_app),
                base_url="http://gateway",
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "auto",
                        "messages": [{"role": "user", "content": secret_prompt}],
                        "max_tokens": 4,
                    },
                )

    assert response.status_code == 200
    secret_output = response.json()["choices"][0]["message"]["content"]
    spans = list(exporter.get_finished_spans())
    gateway = next(
        span
        for span in spans
        if span.name == "kairyu.request"
        and span.parent is None
        and span.attributes["http.route"] == "/v1/chat/completions"
    )
    trace_id = gateway.context.trace_id
    trace = [span for span in spans if span.context.trace_id == trace_id]
    assert len(trace) == len(spans)
    assert gateway.attributes["http.response.status_code"] == 200
    assert gateway.attributes["kairyu.response.complete"] is True

    route = next(span for span in trace if span.name == "kairyu.route")
    assert route.parent is not None
    assert route.parent.span_id == gateway.context.span_id
    assert route.attributes["kairyu.route.target"] == "multi_agent"

    stages = [span for span in trace if span.name == "kairyu.conductor.stage"]
    assert len(stages) >= 2
    for stage in stages:
        assert stage.parent is not None
        assert stage.parent.span_id == gateway.context.span_id
        assert stage.attributes["kairyu.stage.status"] == "success"
        assert stage.attributes["kairyu.stage.name"]
        assert stage.attributes["kairyu.stage.worker"]

        request_id = stage.attributes["kairyu.request_id"]
        placement = next(
            span
            for span in trace
            if span.name == "kairyu.pool.place"
            and span.attributes["kairyu.request_id"] == request_id
        )
        call = next(
            span
            for span in trace
            if span.name == "kairyu.replica.call"
            and span.attributes["kairyu.request_id"] == request_id
        )
        assert placement.parent is not None
        assert placement.parent.span_id == stage.context.span_id
        assert call.parent is not None
        assert call.parent.span_id == stage.context.span_id
        assert call.attributes["kairyu.call.status"] == "success"
        assert (
            placement.attributes["kairyu.replica.id"]
            == call.attributes["kairyu.replica.id"]
            == "replica-1"
        )
        remote_request = next(
            span
            for span in trace
            if span.name == "kairyu.request"
            and span.parent is not None
            and span.parent.span_id == call.context.span_id
        )
        assert remote_request.kind.name == "SERVER"
        assert remote_request.attributes["http.route"] == "/v1/chat/completions"
        assert remote_request.attributes["http.response.status_code"] == 200
        assert remote_request.attributes["kairyu.response.complete"] is True

    exported = _span_payload(trace)
    assert secret_prompt not in exported
    assert secret_output not in exported


class _SecretFailingBackend:
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise RuntimeError(f"private prompt/output: {request.prompt}")

    async def stream(self, request):  # pragma: no cover - unary fixture
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class _BlockingBackend:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.started.set()
        await asyncio.Event().wait()
        return GenerationResult(  # pragma: no cover - cancellation is required
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text="never", token_ids=()),),
        )

    async def stream(self, request):  # pragma: no cover - unary fixture
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


def _request(prompt: str) -> GenerationRequest:
    return GenerationRequest(
        request_id="trace-error-request",
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=1),
    )


async def test_replica_error_and_cancellation_are_privacy_safe(trace_recorder):
    from opentelemetry.trace import StatusCode

    provider, exporter = trace_recorder
    configure_tracing(True, tracer_provider=provider)
    secret = "error-cancel-secret-prompt"
    failing = ReplicaPool({"failing": _SecretFailingBackend()})

    with pytest.raises(RuntimeError, match="private prompt/output"):
        await failing.generate(_request(secret))

    blocking_backend = _BlockingBackend()
    blocking = ReplicaPool({"blocking": blocking_backend})
    task = asyncio.create_task(blocking.generate(_request(secret)))
    await blocking_backend.started.wait()
    task.cancel("private cancellation output")
    with pytest.raises(asyncio.CancelledError):
        await task

    call_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "kairyu.replica.call"
    ]
    assert len(call_spans) == 2
    failed = next(
        span for span in call_spans if span.attributes["error.type"] == "RuntimeError"
    )
    cancelled = next(
        span for span in call_spans if span.attributes["error.type"] == "CancelledError"
    )
    for span in (failed, cancelled):
        assert span.events == ()
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description is None
    assert cancelled.attributes["kairyu.cancelled"] is True
    exported = _span_payload(call_spans)
    assert secret not in exported
    assert "private cancellation output" not in exported


async def test_gateway_span_stays_open_until_the_final_stream_body(trace_recorder):
    provider, exporter = trace_recorder
    configure_tracing(True, tracer_provider=provider)
    first_body_sent = asyncio.Event()
    finish_stream = asyncio.Event()

    async def streaming_app(scope, receive, send):  # noqa: ARG001
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": b"first",
                "more_body": True,
            }
        )
        first_body_sent.set()
        await finish_stream.wait()
        await send(
            {
                "type": "http.response.body",
                "body": b"last",
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.disconnect"}

    sent: list[dict] = []

    async def send(message: dict):
        sent.append(message)

    task = asyncio.create_task(
        TracingMiddleware(streaming_app)(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "method": "POST",
                "headers": [],
                "state": {"request_id": "stream-response-request"},
            },
            receive,
            send,
        )
    )
    await first_body_sent.wait()
    assert exporter.get_finished_spans() == ()

    finish_stream.set()
    await task

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "kairyu.request"
    assert span.attributes["kairyu.request_id"] == "stream-response-request"
    assert span.attributes["http.response.status_code"] == 200
    assert span.attributes["kairyu.response.complete"] is True
    assert [message["body"] for message in sent if "body" in message] == [
        b"first",
        b"last",
    ]


async def test_gateway_marks_swallowed_stream_disconnect_as_cancelled(trace_recorder):
    from opentelemetry.trace import StatusCode

    provider, exporter = trace_recorder
    configure_tracing(True, tracer_provider=provider)

    async def disconnected_app(scope, receive, send):  # noqa: ARG001
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": b"partial",
                "more_body": True,
            }
        )
        assert (await receive())["type"] == "http.disconnect"
        # Starlette's StreamingResponse can return normally after observing the
        # disconnect, so the middleware must not rely on an exception alone.

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):  # noqa: ARG001
        return None

    await TracingMiddleware(disconnected_app)(
        {
            "type": "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "state": {"request_id": "disconnected-request"},
        },
        receive,
        send,
    )

    span = exporter.get_finished_spans()[0]
    assert span.attributes["kairyu.response.complete"] is False
    assert span.attributes["kairyu.cancelled"] is True
    assert span.attributes["error.type"] == "ClientDisconnect"
    assert span.events == ()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None


async def test_gateway_does_not_complete_when_final_send_disconnects(trace_recorder):
    from opentelemetry.trace import StatusCode

    provider, exporter = trace_recorder
    configure_tracing(True, tracer_provider=provider)
    secret = "private send disconnect detail"

    async def streaming_app(scope, receive, send):  # noqa: ARG001
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": b"partial",
                "more_body": True,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"last",
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if (
            message["type"] == "http.response.body"
            and not message.get("more_body", False)
        ):
            raise OSError(secret)

    with pytest.raises(OSError, match=secret):
        await TracingMiddleware(streaming_app)(
            {
                "type": "http",
                "path": "/v1/chat/completions",
                "method": "POST",
                "headers": [],
                "state": {"request_id": "send-disconnected-request"},
            },
            receive,
            send,
        )

    span = exporter.get_finished_spans()[0]
    assert span.attributes["kairyu.response.complete"] is False
    assert span.attributes["kairyu.cancelled"] is True
    assert span.attributes["error.type"] == "OSError"
    assert span.events == ()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert secret not in repr(span)


async def test_gateway_marks_completed_http_5xx_as_error(trace_recorder):
    from opentelemetry.trace import StatusCode

    provider, exporter = trace_recorder
    configure_tracing(True, tracer_provider=provider)

    async def failing_app(scope, receive, send):  # noqa: ARG001
        await send({"type": "http.response.start", "status": 503, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            }
        )

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):  # noqa: ARG001
        return None

    await TracingMiddleware(failing_app)(
        {
            "type": "http",
            "path": "/v1/chat/completions",
            "method": "POST",
            "headers": [],
            "state": {"request_id": "server-error-request"},
        },
        receive,
        send,
    )

    span = exporter.get_finished_spans()[0]
    assert span.attributes["http.response.status_code"] == 503
    assert span.attributes["kairyu.response.complete"] is True
    assert span.attributes["error.type"] == "503"
    assert span.events == ()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
