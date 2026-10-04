"""Every Responses-path error is the OpenAI envelope (M20 WP-07, D10, O-2).

Framework (422, malformed JSON, 404, 405), middleware (401, 403, concurrency,
tenant quota) and handler (reservation, previous_response_id) failures on
``/v1/responses*`` all answer ``{"error": {message, type, param, code}}``.
Transient backpressure is 503 ``slow_down`` with ``Retry-After`` and
``retry-after-ms`` (Codex retries it; it treats any 429 as terminal), the wait
derived from the bucket refill and capped while the tenant's reservations are
in flight; only a reservation the tenant's bucket can never hold stays a 429,
marked ``x-should-retry: false`` for the OpenAI SDKs.

A 413 case joins this table with WP-08c: no body limit applies to
``/v1/responses`` before it.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import pytest
from fastapi import FastAPI

from kairyu.engine.backend import AdmissionUpperBound
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.settings import ServerSettings
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.orchestrator import Orchestrator
from tests.server._legacy_chat import create_legacy_app

_INVALID = "invalid_request_error"
_TURN = {"model": "m", "input": "hello"}
_TENANT_BOUND_TOKENS = 100


class _FixedBoundBackend(MockBackend):
    """Reserves a fixed bound, so a concurrent second turn finds the bucket short."""

    def admission_upper_bound(self, request) -> AdmissionUpperBound:
        return AdmissionUpperBound(tokens=_TENANT_BOUND_TOKENS, refundable_on_exact_usage=True)


def _plain() -> FastAPI:
    return create_legacy_app({"m": MockBackend()})


def _keyed() -> FastAPI:
    return create_legacy_app(
        {"m": MockBackend()},
        resolved_api_keys=frozenset({"data-key"}),
        resolved_admin_keys=frozenset({"admin-key"}),
    )


def _one_slot() -> FastAPI:
    return create_legacy_app(
        {"m": MockBackend(latency_s=0.2)}, settings=ServerSettings(max_concurrency=1)
    )


def _tenant(backend: MockBackend, limits: TenantLimits) -> FastAPI:
    return create_legacy_app(
        {"m": backend}, tenant_config=TenantConfig(limits={"default": limits})
    )


def _auto_tenant_budget_of_one_token() -> FastAPI:
    engine = MockBackend()
    return create_legacy_app(
        {"m": engine},
        orchestrators={"kairyu-auto": Orchestrator({"tier1": engine, "tier2": engine})},
        tenant_config=TenantConfig(limits={"default": TenantLimits(token_burst=1)}),
    )


@dataclass(frozen=True)
class _Case:
    app: Callable[[], FastAPI]
    method: str
    path: str
    status: int
    error_type: str
    code: str | None
    param: str | None = None
    message: str | None = None
    json: dict | None = field(default_factory=lambda: dict(_TURN))
    content: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    first: str | None = None  # "concurrent": occupy the slot; "completed": spend quota
    retry_after: str | None = None  # the exact Retry-After; None: no retry headers
    should_retry: str | None = None  # x-should-retry (OpenAI SDKs)


_CASES = {
    "schema-error": _Case(  # a union field reports the branch that matched the JSON type
        _plain, "POST", "/v1/responses", 400, _INVALID, "invalid_type", "input[0]",
        message="Invalid value for 'input[0]': Input should be a valid dictionary.",
        json={"model": "m", "input": [5]},
    ),
    "malformed-json": _Case(
        _plain, "POST", "/v1/responses", 400, _INVALID, "invalid_json",
        json=None, content=b"{", headers={"content-type": "application/json"},
    ),
    "unknown-subpath": _Case(
        _plain, "GET", "/v1/responses/resp_1/unknown", 404, _INVALID, None,
        message="Invalid URL (GET /v1/responses/resp_1/unknown)", json=None,
    ),
    "method-not-allowed": _Case(
        _plain, "DELETE", "/v1/responses", 405, _INVALID, None,
        message="Method not allowed (DELETE /v1/responses)", json=None,
    ),
    "codex-standalone-search": _Case(
        _plain, "POST", "/v1/alpha/search", 404, _INVALID, None,
        message="Invalid URL (POST /v1/alpha/search)", json={"query": "kairyu"},
    ),
    "missing-api-key": _Case(_keyed, "POST", "/v1/responses", 401, _INVALID, "invalid_api_key"),
    "admin-key-on-data-plane": _Case(
        _keyed, "POST", "/v1/responses", 403, _INVALID, "data_plane_required",
        headers={"authorization": "Bearer admin-key"},
    ),
    "concurrency-limit": _Case(
        _one_slot, "POST", "/v1/responses", 503, "service_unavailable_error", "slow_down",
        first="concurrent", retry_after="1",
    ),
    "tenant-request-quota": _Case(  # burst 1 refilled at 4/min: the wait is the 15 s refill
        lambda: _tenant(MockBackend(), TenantLimits(requests_per_minute=4, request_burst=1)),
        "POST", "/v1/responses", 503, "service_unavailable_error", "slow_down",
        first="completed", retry_after="15",
    ),
    "tenant-token-quota": _Case(  # a 3000 s refill, capped while the held turn settles
        lambda: _tenant(
            _FixedBoundBackend(latency_s=0.2),
            TenantLimits(tokens_per_minute=1, token_burst=_TENANT_BOUND_TOKENS * 3 // 2),
        ),
        "POST", "/v1/responses", 503, "service_unavailable_error", "slow_down",
        first="concurrent", retry_after="10",
    ),
    "engine-tenant-budget-too-small": _Case(
        lambda: _tenant(_FixedBoundBackend(), TenantLimits(token_burst=_TENANT_BOUND_TOKENS // 2)),
        "POST", "/v1/responses", 429, "rate_limit_error", "tenant_budget_too_small",
        should_retry="false",
    ),
    "auto-tenant-budget-too-small": _Case(
        _auto_tenant_budget_of_one_token, "POST", "/v1/responses", 429, "rate_limit_error",
        "tenant_budget_too_small", json={"model": "kairyu-auto", "input": "hello"},
        should_retry="false",
    ),
    "previous-response-not-found": _Case(
        _plain, "POST", "/v1/responses", 400, _INVALID, "previous_response_not_found",
        "previous_response_id", json={**_TURN, "previous_response_id": "resp_missing"},
    ),
}


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _send(client: httpx.AsyncClient, case: _Case) -> httpx.Response:
    return await client.request(
        case.method, case.path, json=case.json, content=case.content, headers=case.headers
    )


@pytest.mark.parametrize("case", list(_CASES.values()), ids=list(_CASES))
async def test_framework_errors_use_openai_envelope(case: _Case) -> None:
    async with _client(case.app()) as client:
        if case.first == "completed":
            assert (await client.post("/v1/responses", json=_TURN)).status_code == 200
        held = None
        if case.first == "concurrent":
            held = asyncio.create_task(client.post("/v1/responses", json=_TURN))
            await asyncio.sleep(0.05)  # the held turn occupies the only slot
        response = await _send(client, case)
        if held is not None:
            assert (await held).status_code == 200

    assert response.status_code == case.status
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert list(body) == ["error"]
    error = body["error"]
    assert sorted(error) == ["code", "message", "param", "type"]
    assert (error["type"], error["code"], error["param"]) == (
        case.error_type,
        case.code,
        case.param,
    )
    if case.message is not None:
        assert error["message"] == case.message
    assert response.headers.get("retry-after") == case.retry_after
    retry_after_ms = response.headers.get("retry-after-ms")
    if case.retry_after is None:
        assert retry_after_ms is None
    else:
        assert math.ceil(int(retry_after_ms) / 1000) == int(case.retry_after)
    assert response.headers.get("x-should-retry") == case.should_retry
