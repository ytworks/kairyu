import asyncio
import json

import httpx
import pytest

from kairyu.deploy.registry import PoolReconciler, ReplicaConfig
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.entrypoints.server.middleware import RequestIngressMiddleware
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog
from kairyu.sampling_params import SamplingParams


def _request(*, placement_started_ns: int | None = None) -> GenerationRequest:
    return GenerationRequest(
        request_id="latency",
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=1),
        placement_started_ns=placement_started_ns,
    )


def test_empty_pool_requires_explicit_dynamic_membership_opt_in() -> None:
    try:
        ReplicaPool({})
    except ValueError as error:
        assert "at least 1 replica" in str(error)
    else:
        raise AssertionError("empty static pool was accepted")

    pool = ReplicaPool({}, allow_empty=True)
    assert pool.replica_count == 0
    pool.add_replica("joined", MockBackend())
    assert pool.replica_ids == ("joined",)


async def test_replica_log_records_ingress_to_selection_latency(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "kairyu.orchestration.replica.time.perf_counter_ns",
        lambda: 1_750,
    )
    log_path = tmp_path / "placement.jsonl"
    log = JsonlRouterLog(log_path)
    pool = ReplicaPool({"replica": MockBackend()}, log=log)

    await pool.generate(_request(placement_started_ns=1_000))
    log.flush()
    row = json.loads(log_path.read_text().strip())

    assert row["kind"] == "replica"
    assert row["replica_id"] == "replica"
    assert row["placement_latency_ns"] == 750
    assert row["request_id"] == "latency"
    assert row["placement_started_ns"] == 1_000
    assert row["selected_at_ns"] == 1_750
    assert row["pool_size"] == 1
    assert row["eligible_size"] == 1
    await pool.shutdown()


async def test_same_id_replacement_has_distinct_serializable_generations(
    tmp_path,
) -> None:
    class Source:
        config = ReplicaConfig("http://old/v1", "model", None)

        def poll(self):
            return {"replica": self.config}

    source = Source()
    events: list[dict[str, object]] = []
    log_path = tmp_path / "replacement.jsonl"
    pool = ReplicaPool(
        {},
        allow_empty=True,
        log=JsonlRouterLog(log_path),
    )
    reconciler = PoolReconciler(
        pool,
        source,
        factory=lambda identity: (MockBackend(), None),
        event_sink=events.append,
    )

    await reconciler.reconcile()
    first_generation = pool.entry_generation("replica")
    await pool.generate(_request())
    source.config = ReplicaConfig("http://new/v1", "model", None)
    await reconciler.reconcile()
    second_generation = pool.entry_generation("replica")
    await pool.generate(_request())
    await reconciler.aclose()
    await pool.shutdown()

    assert isinstance(first_generation, str)
    assert isinstance(second_generation, str)
    assert first_generation != second_generation
    placement_rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "replica"
    ]
    assert [
        row["replica_generation"] for row in placement_rows
    ] == [first_generation, second_generation]
    removed = [
        event
        for event in events
        if event["reason"] == "replica_removed"
    ]
    added = [
        event
        for event in events
        if event["reason"] == "replica_added"
    ]
    assert removed[0]["replica_generation"] == first_generation
    assert added[-1]["replica_generation"] == second_generation


async def test_request_ingress_middleware_stamps_outer_http_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "kairyu.entrypoints.server.middleware.time.perf_counter_ns",
        lambda: 9_001,
    )
    observed = {}

    async def app(scope, receive, send) -> None:
        observed.update(scope["state"])

    middleware = RequestIngressMiddleware(app)
    scope = {"type": "http", "state": {}}

    async def receive():
        return {"type": "http.request"}

    async def send(message) -> None:
        return None

    await middleware(scope, receive, send)
    assert observed["placement_started_ns"] == 9_001


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "fleet",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        ),
        (
            "/v1/completions",
            {"model": "fleet", "prompt": "hello", "max_tokens": 1},
        ),
        (
            "/v1/responses",
            {"model": "fleet", "input": "hello", "max_output_tokens": 1},
        ),
    ],
)
async def test_http_response_request_id_correlates_with_placement_evidence(
    tmp_path,
    path: str,
    payload: dict,
) -> None:
    log_path = tmp_path / f"{path.rsplit('/', 1)[-1]}.jsonl"
    log = JsonlRouterLog(log_path)
    pool = ReplicaPool({"replica": MockBackend()}, log=log)
    app = create_app({"fleet": pool}, legacy_chat_models={"fleet"})

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(path, json=payload)

    assert response.status_code == 200
    response_request_id = response.headers["x-request-id"]
    log.flush()
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["request_id"] == response_request_id
    assert rows[0]["placement_latency_ns"] >= 0
    await pool.shutdown()


async def test_prompt_array_uses_unique_request_ids_under_concurrent_dispatch() -> None:
    class DuplicateRejectingBackend(MockBackend):
        def __init__(self) -> None:
            super().__init__()
            self.active: set[str] = set()
            self.seen: list[str] = []
            self.both_entered = asyncio.Event()

        async def generate(self, request):
            if request.request_id in self.active:
                raise ValueError(f"duplicate request_id {request.request_id!r}")
            self.active.add(request.request_id)
            self.seen.append(request.request_id)
            if len(self.active) == 2:
                self.both_entered.set()
            try:
                await asyncio.wait_for(self.both_entered.wait(), timeout=1)
                return await super().generate(request)
            finally:
                self.active.remove(request.request_id)

    backend = DuplicateRejectingBackend()
    app = create_app({"fleet": backend})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "fleet", "prompt": ["first", "second"]},
        )

    assert response.status_code == 200
    assert len(backend.seen) == 2
    assert len(set(backend.seen)) == 2
    request_id = response.headers["x-request-id"]
    assert backend.seen == [
        f"{request_id}-prompt-0",
        f"{request_id}-prompt-1",
    ]
