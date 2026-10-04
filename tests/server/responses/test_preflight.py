"""Responses engine preflight: prepare, tenant reservation, and dispatch order."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import AdmissionUpperBound, UpstreamClientError
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.tenancy import TenantConfig
from tests.server.responses._helpers import _app, _tool


@pytest.mark.parametrize(
    ("stream", "with_tools"),
    [
        (False, False),
        (True, False),
        (True, True),
    ],
    ids=["unary", "live-stream", "buffered-tool-stream"],
)
def test_responses_prepare_backend_before_tenant_reservation_and_dispatch(
    tmp_path,
    stream: bool,
    with_tools: bool,
) -> None:
    events: list[str] = []

    class PreparedBoundBackend(MockBackend):
        limiter = None

        def admission_upper_bound(self, request):
            events.append("bound")
            return AdmissionUpperBound(
                tokens=7,
                refundable_on_exact_usage=True,
            )

        async def prepare_request(self, request):
            events.append("prepare-start")
            assert self.limiter.reservation_snapshot()["default"] == 0
            await asyncio.sleep(0)
            assert self.limiter.reservation_snapshot()["default"] == 0
            events.append("prepare-end")

        async def generate(self, request):
            events.append("generate")
            assert self.limiter.reservation_snapshot()["default"] == 7
            return await super().generate(request)

    backend = PreparedBoundBackend({"hello": "prepared response"})
    app = _app(tmp_path, backend, tenant_config=TenantConfig())
    backend.limiter = app.state.tenant_limiter
    payload = {
        "model": "m",
        "input": "hello",
        "stream": stream,
    }
    if with_tools:
        payload["tools"] = [_tool()]

    with TestClient(app) as http:
        response = http.post("/v1/responses", json=payload)

    assert response.status_code == 200
    assert events == ["prepare-start", "prepare-end", "bound", "generate"]
    assert backend.limiter.reservation_snapshot()["default"] == 0


@pytest.mark.parametrize(
    (
        "failure",
        "expected_status",
        "expected_message",
        "expected_type",
        "expected_code",
    ),
    [
        (
            UpstreamClientError(
                "private upstream detail",
                status_code=422,
                code="invalid_media",
                public_message="invalid public media",
            ),
            422,
            "invalid public media",
            "invalid_request_error",
            "invalid_media",
        ),
        (
            ValueError("invalid prepared prompt"),
            400,
            "invalid prepared prompt",
            "invalid_request_error",
            None,
        ),
        (
            RuntimeError("private runtime detail"),
            502,
            "upstream backend error (RuntimeError)",
            "upstream_error",
            "backend_error",
        ),
    ],
    ids=["public-client-error", "value-error", "sanitized-runtime-error"],
)
def test_responses_prepare_failure_precedes_stream_headers_and_dispatch(
    tmp_path,
    failure: Exception,
    expected_status: int,
    expected_message: str,
    expected_type: str,
    expected_code: str | None,
) -> None:
    class FailingPrepareBackend(MockBackend):
        async def prepare_request(self, request):
            raise failure

    backend = FailingPrepareBackend()
    with TestClient(_app(tmp_path, backend)) as http:
        response = http.post(
            "/v1/responses",
            json={"model": "m", "input": "hello", "stream": True},
        )

    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == {
        "message": expected_message,
        "type": expected_type,
        "code": expected_code,
    }
    if not isinstance(failure, ValueError):
        assert "private" not in response.text
    assert backend.prompts_seen == ()
