import json

import httpx
import pytest
from fastapi.testclient import TestClient

from kairyu import SamplingParams
from kairyu.batch.store import BatchStore
from kairyu.batch.worker import BatchWorker
from kairyu.engine.backend import GenerationRequest, admission_upper_bound
from kairyu.engine.kairyu_backend import KairyuBackend, build_engine_loop
from kairyu.engine.mock import MockBackend
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.metrics import ServerMetrics
from kairyu.entrypoints.server.tenancy import TenantConfig, TenantLimits
from kairyu.orchestration.replica import ReplicaPool


class _RecordingBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest):
        self.requests.append(request)
        return await super().generate(request)


def _chat_body(**extra) -> dict:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }


def test_tenant_priority_classes_must_preserve_interactive_precedence() -> None:
    with pytest.raises(
        ValueError,
        match="interactive_priority must be smaller than batch_priority",
    ):
        TenantLimits(interactive_priority=1, batch_priority=1)


def test_tenant_quota_rejects_before_shared_replica_dispatch() -> None:
    replica = _RecordingBackend()
    app = create_app(
        {"m": ReplicaPool([replica])},
        tenant_config=TenantConfig(
            key_tenants={
                "good-key": "good",
                "noisy-key": "noisy",
            },
            limits={
                "good": TenantLimits(requests_per_minute=60),
                "noisy": TenantLimits(requests_per_minute=1),
            },
        ),
        resolved_api_keys=frozenset({"good-key", "noisy-key"}),
    )

    with TestClient(app) as client:
        noisy = [
            client.post(
                "/v1/chat/completions",
                json=_chat_body(),
                headers={"Authorization": "Bearer noisy-key"},
            )
            for _ in range(10)
        ]
        good = client.post(
            "/v1/chat/completions",
            json=_chat_body(),
            headers={"Authorization": "Bearer good-key"},
        )
        metrics = client.get("/metrics").text

    assert [response.status_code for response in noisy].count(200) == 1
    assert [response.status_code for response in noisy].count(429) == 9
    assert all(
        response.json()["error"]["code"] == "tenant_rate_limited"
        for response in noisy
        if response.status_code == 429
    )
    assert good.status_code == 200
    assert len(replica.requests) == 2
    assert (
        'kairyu_tenant_admission_total{decision="admitted",'
        'reason="quota_available",source="http",tenant="noisy"} 1.0'
    ) in metrics
    assert (
        'kairyu_tenant_admission_total{decision="rejected",'
        'reason="request_quota",source="http",tenant="noisy"} 9.0'
    ) in metrics
    assert (
        'kairyu_tenant_admission_total{decision="admitted",'
        'reason="quota_available",source="http",tenant="good"} 1.0'
    ) in metrics
    assert (
        'kairyu_tenant_in_flight_requests{source="http",tenant="noisy"} 0.0'
    ) in metrics


def test_token_reservation_rejects_before_replica_and_leaves_zero_gauges() -> None:
    replica = _RecordingBackend()
    app = create_app(
        {"m": ReplicaPool([replica])},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    requests_per_minute=60,
                    tokens_per_minute=1,
                    token_burst=1,
                    max_in_flight=1,
                )
            }
        ),
    )

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_chat_body())
        metrics = client.get("/metrics").text

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "tenant_rate_limited"
    assert replica.requests == []
    assert (
        'kairyu_tenant_admission_total{decision="rejected",'
        'reason="token_quota",source="http",tenant="default"} 1.0'
    ) in metrics
    assert (
        'kairyu_tenant_in_flight_requests{source="http",tenant="default"} 0.0'
    ) in metrics
    assert 'kairyu_tenant_reserved_tokens{tenant="default"} 0.0' in metrics


def test_generation_bound_multiplies_prefill_and_decode_candidate_work() -> None:
    single = GenerationRequest(
        request_id="single",
        prompt="hello",
        sampling_params=SamplingParams(n=1, max_tokens=8),
    )
    candidates = GenerationRequest(
        request_id="candidates",
        prompt="hello",
        sampling_params=SamplingParams(n=2, best_of=3, max_tokens=8),
    )

    one = admission_upper_bound(single)
    three = admission_upper_bound(candidates)

    assert three.tokens == one.tokens * 3
    assert one.refundable_on_exact_usage
    assert not three.refundable_on_exact_usage


def test_direct_chat_accepts_vllm_priority_without_gateway_policy() -> None:
    backend = _RecordingBackend()
    app = create_app({"m": backend})

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_chat_body(priority=-7))

    assert response.status_code == 200
    assert backend.requests[-1].priority == -7


def test_authenticated_tenant_priority_replaces_untrusted_chat_value() -> None:
    backend = _RecordingBackend()
    tenants = TenantConfig(
        limits={
            "default": TenantLimits(
                interactive_priority=-3,
                batch_priority=4,
            )
        }
    )
    app = create_app({"m": backend}, tenant_config=tenants)

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_chat_body(priority=-99))

    assert response.status_code == 200
    assert backend.requests[-1].priority == -3


def test_responses_uses_the_same_trusted_tenant_priority() -> None:
    backend = _RecordingBackend()
    tenants = TenantConfig(
        limits={"default": TenantLimits(interactive_priority=-4)}
    )
    app = create_app({"m": backend}, tenant_config=tenants)

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "m", "input": "hello", "priority": -99},
        )
        metrics = client.get("/metrics").text

    assert response.status_code == 200
    assert backend.requests[-1].priority == -4
    assert backend.requests[-1].scheduling_class == "interactive"
    assert (
        'kairyu_priority_requests_total{request_class="interactive",'
        'source="http"} 1.0'
    ) in metrics


def test_legacy_completion_preserves_direct_priority() -> None:
    backend = _RecordingBackend()
    app = create_app({"m": backend})

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "m", "prompt": "hello", "priority": -7},
        )

    assert response.status_code == 200
    assert backend.requests[-1].priority == -7
    assert backend.requests[-1].scheduling_class == "interactive"


@pytest.mark.asyncio
async def test_batch_worker_replaces_client_value_with_batch_class(tmp_path) -> None:
    backend = _RecordingBackend()
    tenants = TenantConfig(
        limits={"gold": TenantLimits(interactive_priority=-10, batch_priority=-5)}
    )
    metrics = ServerMetrics()
    worker = BatchWorker(
        BatchStore(tmp_path),
        {"m": backend},
        tenant_config=tenants,
        metrics=metrics,
    )
    line = {
        "custom_id": "line-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": _chat_body(priority=-99),
    }

    output, error, _usage = await worker._run_line(
        line,
        "/v1/chat/completions",
        set(),
        "gold",
    )

    assert error is None
    assert output is not None
    assert backend.requests[-1].priority == -5
    assert backend.requests[-1].scheduling_class == "batch"
    rendered = metrics.render()[0].decode()
    assert (
        'kairyu_priority_requests_total{request_class="batch",source="batch"} 1.0'
        in rendered
    )


def test_native_engine_loop_admits_smaller_priority_first() -> None:
    loop, _cache, scheduler = build_engine_loop(
        tokenizer="toy",
        max_num_batched_tokens=1,
        priority_age_s=60.0,
    )
    try:
        loop.submit(
            "batch",
            "batch prompt",
            SamplingParams(max_tokens=1),
            priority=1,
        )
        loop.submit(
            "interactive",
            "interactive",
            SamplingParams(max_tokens=1),
            priority=0,
        )
        loop._drain_ops()

        plan = scheduler.schedule()

        assert plan.scheduled[0].request_id == "interactive"
        assert scheduler.states["interactive"].request.priority == 0
        assert scheduler.states["batch"].request.priority == 1
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_gateway_forwards_priority_to_kairyu_replica() -> None:
    captured: dict = {}
    captured_headers: dict = {}

    def handler(request):
        captured.update(json.loads(request.content))
        captured_headers.update(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-priority",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    backend = OpenAICompatBackend(
        base_url="https://replica.test/v1",
        model="m",
        api_key_env=None,
        upstream="kairyu",
        transport=httpx.MockTransport(handler),
    )
    await backend.generate(
        GenerationRequest(
            request_id="priority-wire",
            prompt="hello",
            sampling_params=SamplingParams(max_tokens=1),
            priority=6,
            scheduling_class="batch",
        )
    )

    assert captured["priority"] == 6
    assert captured_headers["x-kairyu-scheduling-class"] == "batch"


@pytest.mark.asyncio
async def test_tenant_gateway_pool_reaches_replica_scheduler_with_explicit_class() -> None:
    replica_engine = KairyuBackend(
        tokenizer="toy",
        max_num_batched_tokens=16,
        priority_age_s=60.0,
    )
    replica_app = create_app({"m": replica_engine})
    remote = OpenAICompatBackend(
        base_url="http://replica.test/v1",
        model="m",
        api_key_env=None,
        upstream="kairyu",
        transport=httpx.ASGITransport(app=replica_app),
    )
    gateway_app = create_app(
        {"m": ReplicaPool([remote])},
        tenant_config=TenantConfig(
            limits={
                "default": TenantLimits(
                    interactive_priority=5,
                    batch_priority=10,
                )
            }
        ),
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway_app),
            base_url="http://gateway.test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_body(priority=-(2**63), max_tokens=1),
            )
            gateway_metrics = (await client.get("/metrics")).text
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=replica_app),
            base_url="http://replica.test",
        ) as client:
            replica_metrics = (await client.get("/metrics")).text

        assert response.status_code == 200
        snapshot = replica_engine.scheduler_priority_metrics()
        assert snapshot["events"][("interactive", "enqueue")] == 1
        assert snapshot["events"][("interactive", "admit")] == 1
        assert snapshot["events"][("interactive", "complete")] == 1
        assert (
            'kairyu_priority_requests_total{request_class="interactive",'
            'source="http"} 1.0'
        ) in gateway_metrics
        assert (
            'kairyu_priority_requests_total{request_class="interactive",'
            'source="http"} 1.0'
        ) in replica_metrics
        assert (
            'kairyu_scheduler_priority_events_total{event="admit",model="m",'
            'request_class="interactive"} 1.0'
        ) in replica_metrics
    finally:
        await remote.shutdown()
        await replica_engine.shutdown()
