"""Production wiring for direct-chat TTFT SLO admission (issue #340)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from kairyu.engine.backend import GenerationResult
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.middleware import (
    _SLO_ADMISSION_LEASE_STATE_KEY,
    RequestIngressMiddleware,
)
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.slo import AdmissionController
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.outputs import CompletionOutput
from tests.server._legacy_chat import create_legacy_app

_LOWEST_SCHEDULER_PRIORITY = 2**63 - 1


def _chat_scope(encoded: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }


class _RecordingBackend(MockBackend):
    supports_slo_defer = True

    def __init__(self) -> None:
        super().__init__()
        self.prepared = []
        self.dispatched = []

    async def prepare_request(self, request) -> None:
        self.prepared.append(request)

    async def generate(self, request):
        self.dispatched.append(request)
        return await super().generate(request)


@pytest.mark.parametrize(
    ("ema", "expected_status", "expected_class", "expected_priority"),
    [
        pytest.param(0.01, 200, "interactive", 0, id="admit"),
        pytest.param(
            1.25,
            200,
            "batch",
            _LOWEST_SCHEDULER_PRIORITY,
            id="defer",
        ),
        pytest.param(2.25, 429, None, None, id="shed"),
    ],
)
def test_direct_chat_applies_slo_admit_defer_and_shed(
    ema,
    expected_status,
    expected_class,
    expected_priority,
):
    backend = _RecordingBackend()
    tenant_config = TenantConfig(
        limits={
            "default": TenantLimits(
                interactive_priority=-4 if expected_class == "batch" else 0,
                batch_priority=7 if expected_class == "batch" else 1,
            )
        }
    )
    app = create_legacy_app(
        {"m": backend},
        settings=ServerSettings(ttft_slo_s=1.0),
        tenant_config=tenant_config,
    )
    app.state.slo_admission._ttft_per_unit_ema = ema
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "stream": True,
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)
        metrics = client.get("/metrics").text

        assert response.status_code == expected_status
        if expected_status == 429:
            assert response.headers["retry-after"] == "1"
            assert response.json()["error"]["code"] == "slo_admission_shed"
            assert backend.prepared == []
            assert backend.dispatched == []
            invalid = client.post(
                "/v1/chat/completions",
                json={**body, "model": "missing"},
            )
            assert invalid.status_code == 404
        else:
            assert len(backend.prepared) == len(backend.dispatched) == 1
            assert backend.prepared[0] is backend.dispatched[0]
            assert backend.dispatched[0].scheduling_class == expected_class
            assert backend.dispatched[0].priority == expected_priority

    snapshot = app.state.slo_admission.snapshot()
    assert snapshot.in_flight == 0
    assert snapshot.interactive_in_flight == 0
    assert snapshot.deferred_in_flight == 0
    for name in (
        "kairyu_slo_admission_in_flight",
        "kairyu_slo_admission_interactive_in_flight",
        "kairyu_slo_admission_deferred_in_flight",
        "kairyu_slo_admission_ttft_per_unit_ema_seconds",
        "kairyu_slo_admission_predicted_ttft_seconds",
        "kairyu_slo_admission_defer_pressure_seconds",
    ):
        assert name in metrics


@pytest.mark.parametrize(
    ("ema", "expected_status", "expected_class", "expected_priority"),
    [
        pytest.param(0.01, 200, "interactive", 0, id="admit"),
        pytest.param(
            1.25,
            200,
            "batch",
            _LOWEST_SCHEDULER_PRIORITY,
            id="defer",
        ),
        pytest.param(2.25, 429, None, None, id="shed"),
    ],
)
def test_messages_applies_slo_admit_defer_and_shed(
    ema,
    expected_status,
    expected_class,
    expected_priority,
):
    backend = _RecordingBackend()
    tenant_config = TenantConfig(
        limits={
            "default": TenantLimits(
                interactive_priority=-4 if expected_class == "batch" else 0,
                batch_priority=7 if expected_class == "batch" else 1,
            )
        }
    )
    app = create_legacy_app(
        {"m": backend},
        settings=ServerSettings(ttft_slo_s=1.0),
        tenant_config=tenant_config,
    )
    app.state.slo_admission._ttft_per_unit_ema = ema

    with TestClient(app) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        )

    assert response.status_code == expected_status
    if expected_status == 429:
        assert response.headers["retry-after"] == "1"
        payload = response.json()
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "rate_limit_error"
        assert "predicted TTFT" in payload["error"]["message"]
        assert backend.prepared == []
        assert backend.dispatched == []
    else:
        assert len(backend.prepared) == len(backend.dispatched) == 1
        assert backend.prepared[0] is backend.dispatched[0]
        assert backend.dispatched[0].scheduling_class == expected_class
        assert backend.dispatched[0].priority == expected_priority

    snapshot = app.state.slo_admission.snapshot()
    assert snapshot.in_flight == 0
    assert snapshot.interactive_in_flight == 0
    assert snapshot.deferred_in_flight == 0


def test_defer_sheds_when_backend_cannot_isolate_running_batch():
    backend = MockBackend()
    app = create_legacy_app(
        {"m": backend},
        settings=ServerSettings(ttft_slo_s=1.0),
    )
    app.state.slo_admission._ttft_per_unit_ema = 1.25

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"]["code"] == "slo_admission_shed"
    assert backend.prompts_seen == ()
    assert app.state.slo_admission.in_flight == 0


async def test_defer_to_shed_releases_lease_before_sending_429():
    app = create_legacy_app(
        {"m": MockBackend()},
        settings=ServerSettings(ttft_slo_s=1.0),
    )
    app.state.slo_admission._ttft_per_unit_ema = 1.25
    encoded = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()
    request_sent = False
    snapshots = []
    statuses = []

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": encoded, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
        snapshots.append(app.state.slo_admission.snapshot())

    await app(_chat_scope(encoded), receive, send)

    assert statuses == [429]
    assert snapshots
    assert all(snapshot.in_flight == 0 for snapshot in snapshots)
    assert all(snapshot.deferred_in_flight == 0 for snapshot in snapshots)


def test_admitted_max_priority_is_clamped_even_when_defer_is_unsupported():
    backend = _RecordingBackend()
    backend.supports_slo_defer = False
    app = create_legacy_app(
        {"m": backend},
        settings=ServerSettings(ttft_slo_s=1.0),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "priority": _LOWEST_SCHEDULER_PRIORITY,
            },
        )

    assert response.status_code == 200
    assert backend.dispatched[0].priority == _LOWEST_SCHEDULER_PRIORITY - 1


class _SlowValidationBackend(_RecordingBackend):
    def validate_request(self, request) -> None:
        time.sleep(0.06)
        super().validate_request(request)


def test_ingress_elapsed_time_can_shed_before_backend_preparation():
    backend = _SlowValidationBackend()
    app = create_legacy_app(
        {"m": backend},
        settings=ServerSettings(ttft_slo_s=0.02),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 429
    assert backend.prepared == []
    assert backend.dispatched == []
    assert app.state.slo_admission.in_flight == 0


class _ClockedMultiChoiceBackend(_RecordingBackend):
    def __init__(self, clock) -> None:
        super().__init__()
        self._clock = clock

    async def stream(self, request):
        self.dispatched.append(request)
        final = self._result_for(request)
        self._clock["t"] = 0.5
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=tuple(
                CompletionOutput(
                    index=completion.index,
                    text=completion.text[:1],
                    token_ids=(),
                )
                for completion in final.completions
            ),
            finished=False,
        )
        self._clock["t"] = 10.0
        yield final


def test_first_stream_content_observes_ttft_once_across_choices():
    clock = {"t": 0.0}
    backend = _ClockedMultiChoiceBackend(clock)
    app = create_legacy_app(
        {"m": backend},
        settings=ServerSettings(ttft_slo_s=1.0),
    )
    app.state.slo_admission._now = lambda: clock["t"]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "upstream_error" not in response.text
    # 0.8 * the 0.01 prior + 0.2 * (0.5 seconds / concurrency 1).
    assert app.state.slo_admission.snapshot().ttft_per_unit_ema_s == pytest.approx(0.108)
    assert app.state.slo_admission.in_flight == 0


class _ClockedBufferedToolBackend(_RecordingBackend):
    def __init__(self, clock) -> None:
        super().__init__()
        self._clock = clock

    async def generate(self, request):
        self._clock["t"] = 0.5
        return await super().generate(request)


def test_buffered_tool_stream_observes_first_visible_delta():
    clock = {"t": 0.0}
    app = create_legacy_app(
        {"m": _ClockedBufferedToolBackend(clock)},
        settings=ServerSettings(ttft_slo_s=1.0),
    )
    app.state.slo_admission._now = lambda: clock["t"]

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "none",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert app.state.slo_admission.snapshot().ttft_per_unit_ema_s == pytest.approx(0.108)
    assert app.state.slo_admission.in_flight == 0


class _OneTokenThenBlockBackend(MockBackend):
    def __init__(self, clock) -> None:
        super().__init__()
        self._clock = clock

    async def stream(self, request):
        self._clock["t"] = 0.5
        yield GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(
                CompletionOutput(index=0, text="x", token_ids=(1,)),
            ),
            finished=False,
        )
        await asyncio.Event().wait()


async def test_failed_first_sse_body_send_does_not_train_ttft_ema():
    clock = {"t": 0.0}
    app = create_legacy_app(
        {"m": _OneTokenThenBlockBackend(clock)},
        settings=ServerSettings(ttft_slo_s=1.0),
    )
    app.state.slo_admission._now = lambda: clock["t"]
    encoded = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    ).encode()
    request_sent = False
    body_send_attempts = 0
    disconnected = asyncio.Event()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal body_send_attempts
        if message["type"] == "http.response.body" and message.get("body"):
            body_send_attempts += 1
            raise asyncio.CancelledError

    await asyncio.wait_for(app(_chat_scope(encoded), receive, send), timeout=2)

    snapshot = app.state.slo_admission.snapshot()
    assert body_send_attempts == 1
    assert snapshot.in_flight == 0
    assert snapshot.ttft_per_unit_ema_s == pytest.approx(0.01)


async def test_request_boundary_releases_slo_lease_on_cancellation():
    controller = AdmissionController(ttft_slo_s=1.0)
    lease = controller.begin()

    async def cancelled_app(scope, receive, send):  # noqa: ARG001
        raise asyncio.CancelledError

    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "state": {_SLO_ADMISSION_LEASE_STATE_KEY: lease},
    }
    with pytest.raises(asyncio.CancelledError):
        await RequestIngressMiddleware(cancelled_app)(scope, lambda: None, lambda _: None)

    assert controller.in_flight == 0
    assert _SLO_ADMISSION_LEASE_STATE_KEY not in scope["state"]
