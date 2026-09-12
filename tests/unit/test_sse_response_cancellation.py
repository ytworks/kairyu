"""Disconnect during an SSE send must close task-backed deferred orchestration."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.requests import ClientDisconnect, Request

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import _stream_orchestrator
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.entrypoints.server.sse_response import sse_response
from kairyu.orchestration.conductor import RoleSpec
from kairyu.orchestration.orchestrator import Orchestrator
from kairyu.orchestration.router import RuleRouter


@pytest.mark.parametrize(
    ("asgi_version", "disconnect_at"), [("2.3", "send"), ("2.4", "send"), ("2.3", "next")]
)
@pytest.mark.parametrize("disconnect", [True, False])
async def test_deferred_audit_stream_ownership(
    monkeypatch, asgi_version, disconnect_at, disconnect
):
    import kairyu.orchestration.orchestrator as module

    monkeypatch.setattr(module, "_KEEPALIVE_INTERVAL_S", 60 if disconnect_at == "next" else 0.001)
    active, closed, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class AuditBackend(MockBackend):
        async def generate(self, request):
            if "[audit]" in str(request.prompt):
                active.set()
                try:
                    if disconnect:
                        await asyncio.Event().wait()
                finally:
                    # Cleanup itself may need asynchronous I/O (closing upstream HTTP).
                    await asyncio.sleep(0.02)
                    closed.set()
            return await super().generate(request)

    query = (
        "First research the options and summarize trade-offs. Then design a plan. "
        "After that implement it. Finally verify everything works end to end."
    )
    decision = RuleRouter().route(query)
    assert decision.target == "multi_agent"
    orchestrator = Orchestrator(
        engines={
            "tier1": AuditBackend(responses={"[head]": "Opening."}),
            "tier2": MockBackend(responses={"[answer]": "Remainder."}),
        },
        router=SimpleNamespace(route=lambda *a, **k: decision),
        roles=(
            RoleSpec(name="head", worker="tier1", role_type="head", prompt="[head] {query}"),
            RoleSpec(
                name="answer",
                worker="tier2",
                role_type="publisher",
                depends_on=("head",),
                prompt="[answer] {query}",
            ),
            RoleSpec(
                name="audit",
                worker="tier1",
                role_type="verifier",
                verifies="answer",
                depends_on=("answer",),
                prompt="[audit] {answer}",
            ),
        ),
    )
    request = ChatCompletionRequest(
        model="test", messages=[{"role": "user", "content": query}], stream=True
    )
    content = _stream_orchestrator(
        orchestrator,
        query,
        None,
        request,
        False,
        False,
        Request({"type": "http", "app": FastAPI()}),
    )
    response = sse_response(content)

    emitted = []

    async def send(message):
        if message["type"] == "http.response.body":
            emitted.append(message["body"])
        if (
            disconnect
            and disconnect_at == "send"
            and message["type"] == "http.response.body"
            and active.is_set()
        ):
            if asgi_version == "2.4":
                raise OSError("peer closed")
            disconnected.set()
            await asyncio.Event().wait()

    async def receive():
        if disconnect and disconnect_at == "next":
            await active.wait()
            await asyncio.sleep(0.001)
            return {"type": "http.disconnect"}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    scope = {"type": "http", "asgi": {"spec_version": asgi_version}}
    try:
        if disconnect and asgi_version == "2.4":
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(response(scope, receive, send), 2)
        else:
            await asyncio.wait_for(response(scope, receive, send), 2)
        if not disconnect:
            body = b"".join(emitted)
            assert b"Opening." in body
            assert b"Remainder." in body
            assert body.endswith(b"data: [DONE]\n\n")
        assert active.is_set(), "The test must reach deferred backend.generate"
        assert closed.is_set(), "HTTP response must await cancellation all the way to the backend"
    finally:
        # Do not leave the intentionally blocked backend behind on a red run.
        await content.aclose()
        await asyncio.sleep(0.01)


async def test_normal_sse_exhaustion_preserves_bytes_and_closes_generator():
    closed = asyncio.Event()
    chunks = [b"data: first\n\n", b"data: [DONE]\n\n"]

    async def content():
        try:
            for chunk in chunks:
                yield chunk
        finally:
            closed.set()

    emitted = []

    async def send(message):
        if message["type"] == "http.response.body":
            emitted.append(message["body"])

    async def receive():
        await asyncio.Event().wait()

    await sse_response(content())({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert b"".join(emitted) == b"".join(chunks)
    assert closed.is_set()
