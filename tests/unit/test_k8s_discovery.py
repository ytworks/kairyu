"""Production Kubernetes EndpointSlice discovery and reconciliation loop."""

from __future__ import annotations

import asyncio
import ipaddress
import json
from pathlib import Path

import httpx
import pytest

from kairyu.deploy.registry import (
    KubernetesEndpointSliceDiscovery,
    PoolReconciler,
    ReplicaConfig,
)
from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    SamplingParams,
)
from kairyu.orchestration.replica import ReplicaPool
from kairyu.orchestration.router import JsonlRouterLog


class _Backend:
    def __init__(self) -> None:
        self.closed = False

    async def generate(self, request):
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(),
            finished=True,
        )

    async def stream(self, request):
        yield await self.generate(request)

    async def shutdown(self) -> None:
        self.closed = True


def _slice_payload() -> dict:
    return {
        "apiVersion": "discovery.k8s.io/v1",
        "kind": "EndpointSliceList",
        "items": [
            {
                "addressType": "IPv4",
                "ports": [{"name": "http", "protocol": "TCP", "port": 8000}],
                "endpoints": [
                    {
                        "addresses": ["10.2.0.1"],
                        "conditions": {"ready": True, "terminating": False},
                        "targetRef": {"uid": "uid-a", "name": "pod-a"},
                    },
                    {
                        "addresses": ["10.2.0.2"],
                        "conditions": {"ready": False, "terminating": False},
                        "targetRef": {"uid": "not-ready"},
                    },
                    {
                        "addresses": ["10.2.0.3"],
                        "conditions": {"ready": True, "terminating": True},
                        "targetRef": {"uid": "terminating"},
                    },
                ],
            },
            {
                "addressType": "IPv6",
                "ports": [{"name": "http", "protocol": "TCP", "port": 8000}],
                "endpoints": [
                    {
                        "addresses": ["2001:db8::17"],
                        "conditions": {"ready": True},
                        "targetRef": {"name": "pod-v6"},
                    }
                ],
            },
            {
                "addressType": "FQDN",
                "ports": [{"name": "http", "protocol": "TCP", "port": 8000}],
                "endpoints": [
                    {
                        "addresses": ["ignored.example"],
                        "conditions": {"ready": True},
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
async def test_endpoint_slice_query_rotates_auth_and_parses_ready_addresses(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_slice_payload())

    token_path = tmp_path / "token"
    token_path.write_text("token-one\n", encoding="utf-8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = KubernetesEndpointSliceDiscovery(
        "kairyu-replicas",
        "model-serving",
        "http",
        model="qwen",
        api_key_env=None,
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
    )

    expected = {
        "uid-a": ReplicaConfig(
            address="http://10.2.0.1:8000/v1",
            model="qwen",
            api_key_env=None,
        ),
        "pod-v6": ReplicaConfig(
            address="http://[2001:db8::17]:8000/v1",
            model="qwen",
            api_key_env=None,
        ),
    }
    assert await source.poll() == expected
    token_path.write_text("token-two", encoding="utf-8")
    assert await source.poll() == expected

    assert [request.headers["authorization"] for request in requests] == [
        "Bearer token-one",
        "Bearer token-two",
    ]
    assert all(
        request.url.path
        == (
            "/apis/discovery.k8s.io/v1/namespaces/"
            "model-serving/endpointslices"
        )
        for request in requests
    )
    assert all(
        request.url.params["labelSelector"]
        == "kubernetes.io/service-name=kairyu-replicas"
        for request in requests
    )
    await source.aclose()
    assert client.is_closed is False
    await client.aclose()


@pytest.mark.asyncio
async def test_numeric_port_and_address_fallback(tmp_path: Path) -> None:
    payload = {
        "items": [
            {
                "addressType": "IPv4",
                "ports": [{"protocol": "TCP", "port": 9000}],
                "endpoints": [
                    {
                        "addresses": ["10.3.0.4"],
                        "conditions": {"ready": True},
                    }
                ],
            }
        ]
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = KubernetesEndpointSliceDiscovery(
        "replicas",
        "default",
        "9000",
        scheme="https",
        base_path="",
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
    )

    # The token contents do not matter to the mock transport.
    assert await source.poll() == {
        "10.3.0.4": ReplicaConfig(
            address="https://10.3.0.4:9000",
        )
    }
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"items": "not-a-list"},
        {
            "items": [
                {
                    "addressType": "IPv4",
                    "ports": [{"name": "http", "port": 8000}],
                    "endpoints": [
                        {
                            "addresses": [],
                            "conditions": {"ready": True},
                        }
                    ],
                }
            ]
        },
        {
            "items": [
                {
                    "addressType": "IPv4",
                    "ports": [{"name": "http", "port": 8000}],
                    "endpoints": [
                        {
                            "addresses": ["2001:db8::1"],
                            "conditions": {"ready": True},
                        }
                    ],
                }
            ]
        },
    ],
)
async def test_malformed_endpoint_slice_fails_closed(
    tmp_path: Path, payload: object
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = KubernetesEndpointSliceDiscovery(
        "replicas",
        "default",
        "http",
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
    )

    with pytest.raises(ValueError):
        await source.poll()
    await client.aclose()


@pytest.mark.parametrize(
    ("preference", "expected_address"),
    [
        ("ipv4", "http://10.0.0.1:8000/v1"),
        ("ipv6", "http://[2001:0db8::1]:8000/v1"),
    ],
)
async def test_dual_stack_uid_is_one_deterministic_replica(
    preference: str,
    expected_address: str,
) -> None:
    ipv4 = {
        "addressType": "IPv4",
        "ports": [{"name": "http", "protocol": "TCP", "port": 8000}],
        "endpoints": [
            {
                "addresses": ["10.0.0.2", "10.0.0.1"],
                "conditions": {"ready": True},
                "targetRef": {"uid": "same-pod", "name": "replica-0"},
            }
        ],
    }
    ipv6 = {
        "addressType": "IPv6",
        "ports": [{"name": "http", "protocol": "TCP", "port": 8000}],
        "endpoints": [
            {
                # The last two spellings have the same numeric address. The
                # historical tuple then uses the original string as its final
                # tie-break, after family, numeric address, and port.
                "addresses": [
                    "2001:db8::2",
                    "2001:db8::1",
                    "2001:0db8::1",
                ],
                "conditions": {"ready": True},
                "targetRef": {"uid": "same-pod", "name": "replica-0"},
            }
        ],
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"items": []})
        )
    )
    source = KubernetesEndpointSliceDiscovery(
        "replicas",
        "default",
        "http",
        address_family_preference=preference,
        client=client,
    )

    expected = {
        "same-pod": ReplicaConfig(address=expected_address),
    }

    def legacy_expected(items: list[dict]) -> dict[str, ReplicaConfig]:
        preferred_version = 6 if preference == "ipv6" else 4
        candidates = []
        for item in items:
            port = item["ports"][0]["port"]
            for endpoint in item["endpoints"]:
                for address in endpoint["addresses"]:
                    parsed = ipaddress.ip_address(address)
                    candidates.append(
                        (
                            0 if parsed.version == preferred_version else 1,
                            int(parsed),
                            port,
                            address,
                        )
                    )
        _rank, _numeric, port, address = min(candidates)
        parsed = ipaddress.ip_address(address)
        host = f"[{address}]" if parsed.version == 6 else address
        return {
            "same-pod": ReplicaConfig(address=f"http://{host}:{port}/v1"),
        }

    for items in ([ipv4, ipv6], [ipv6, ipv4]):
        assert legacy_expected(items) == expected
        assert source._parse({"items": items}) == legacy_expected(items)
    await client.aclose()


@pytest.mark.asyncio
async def test_async_discovery_reconciles_and_emits_membership_event() -> None:
    class AsyncSource:
        async def poll(self):
            await asyncio.sleep(0)
            return {"new": ReplicaConfig("http://10.0.0.2:8000/v1", "model")}

    events: list[dict[str, object]] = []

    async def sink(event: dict[str, object]) -> None:
        await asyncio.sleep(0)
        events.append(event)

    pool = ReplicaPool({}, allow_empty=True)
    reconciler = PoolReconciler(
        pool,
        AsyncSource(),
        factory=lambda identity: (_Backend(), None),
        event_sink=sink,
    )

    assert await reconciler.reconcile() == {
        "added": ["new"],
        "draining": [],
        "removed": [],
    }
    assert pool.replica_ids == ("new",)
    assert len(events) == 1
    event = events[0]
    assert event["reason"] == "replica_added"
    assert event["added"] == ["new"]
    assert event["sequence"] == 1
    assert event["event_id"] == (
        f"{event['event_source_id']}:00000000000000000001"
    )
    assert isinstance(event["observed_ns"], int)
    assert event["generation_by_id"] == {
        "new": event["replica_generation"]
    }


@pytest.mark.asyncio
async def test_fail_once_sink_retries_identical_event_before_next_poll() -> None:
    class Source:
        polls = 0

        def poll(self):
            self.polls += 1
            return {"new": ReplicaConfig("http://new/v1", "model")}

    attempts: list[dict[str, object]] = []

    def sink(event: dict[str, object]) -> None:
        attempts.append(event)
        if len(attempts) == 1:
            raise RuntimeError("audit sink full")

    source = Source()
    pool = ReplicaPool({}, allow_empty=True)
    reconciler = PoolReconciler(
        pool,
        source,
        factory=lambda identity: (_Backend(), None),
        event_sink=sink,
    )

    with pytest.raises(RuntimeError, match="sink full"):
        await reconciler.reconcile()
    assert pool.replica_ids == ("new",)
    assert source.polls == 1

    assert (await reconciler.reconcile())["added"] == []
    assert source.polls == 2
    assert len(attempts) == 2
    assert attempts[0]["event_id"] == attempts[1]["event_id"]
    assert attempts[0]["sequence"] == attempts[1]["sequence"] == 1
    assert attempts[0]["observed_ns"] == attempts[1]["observed_ns"]


@pytest.mark.asyncio
async def test_membership_transition_time_precedes_placement_even_if_flushed_later(
    tmp_path: Path,
) -> None:
    class Source:
        def poll(self):
            return {"replica": ReplicaConfig("http://replica/v1", "model")}

    log = JsonlRouterLog(tmp_path / "causal.jsonl")
    pool = ReplicaPool({}, allow_empty=True, log=log)

    def sink(event: dict[str, object]) -> None:
        log.record_membership(
            event,
            replica_ids=tuple(event["replica_ids"]),
            healthy_ids=tuple(event["healthy_ids"]),
            eligible_ids=tuple(event["eligible_ids"]),
            generation_by_id=event["generation_by_id"],
        )

    reconciler = PoolReconciler(
        pool,
        Source(),
        factory=lambda identity: (_Backend(), "http://replica/readyz"),
        event_sink=sink,
    )
    await reconciler.reconcile()

    # The probe transition enters the reconciler outbox synchronously. Place a
    # request before the next reconcile flushes that membership row.
    await pool.probe("replica")
    await pool.generate(
        GenerationRequest(
            request_id="placed-after-probe",
            prompt="p",
            sampling_params=SamplingParams(),
        )
    )
    await reconciler.reconcile()
    log.flush()

    rows = [
        json.loads(line)
        for line in (tmp_path / "causal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    placement_index = next(
        index for index, row in enumerate(rows) if row["kind"] == "replica"
    )
    restored_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("reason") == "probe_succeeded"
    )
    placement = rows[placement_index]
    restored = rows[restored_index]

    assert placement_index < restored_index
    assert restored["observed_ns"] <= placement["selected_at_ns"]
    assert restored["generation_by_id"]["replica"] == placement[
        "replica_generation"
    ]

    await reconciler.aclose()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_partial_factory_failure_flushes_applied_mutations() -> None:
    class Source:
        def poll(self):
            return {
                replica_id: ReplicaConfig(f"http://{replica_id}/v1", "model")
                for replica_id in ("a", "b", "c")
            }

    events: list[dict[str, object]] = []

    def factory(identity):
        if identity.address == "http://b/v1":
            raise RuntimeError("factory b failed")
        return _Backend(), None

    pool = ReplicaPool({}, allow_empty=True)
    reconciler = PoolReconciler(
        pool,
        Source(),
        factory=factory,
        event_sink=events.append,
    )

    with pytest.raises(RuntimeError, match="factory b failed"):
        await reconciler.reconcile()
    assert pool.replica_ids == ("a",)
    assert [(event["reason"], event["replica_id"]) for event in events] == [
        ("replica_added", "a")
    ]


@pytest.mark.asyncio
async def test_partial_shutdown_failure_flushes_every_detached_generation() -> None:
    class ShutdownFails(_Backend):
        async def shutdown(self) -> None:
            raise RuntimeError("shutdown failed")

    class EmptySource:
        def poll(self):
            return {}

    first = _Backend()
    second = ShutdownFails()
    pool = ReplicaPool({"a": first, "b": second})
    events: list[dict[str, object]] = []
    reconciler = PoolReconciler(
        pool,
        EmptySource(),
        event_sink=events.append,
    )

    with pytest.raises(ExceptionGroup, match="shutdown failed"):
        await reconciler.reconcile()
    assert pool.replica_ids == ()
    assert first.closed is True
    assert [
        (event["reason"], event["replica_id"])
        for event in events
        if event["reason"] == "replica_removed"
    ] == [("replica_removed", "a"), ("replica_removed", "b")]
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )


@pytest.mark.asyncio
async def test_effective_eligibility_events_are_bounded_to_real_transitions() -> None:
    class FailingBackend(_Backend):
        failing = False

        async def generate(self, request):
            if self.failing:
                raise RuntimeError("backend failed")
            return await super().generate(request)

    class Source:
        def poll(self):
            return {"replica": ReplicaConfig("http://replica/v1", "model")}

    backend = FailingBackend()
    pool = ReplicaPool({}, allow_empty=True, unhealthy_after=1)
    events: list[dict[str, object]] = []
    reconciler = PoolReconciler(
        pool,
        Source(),
        factory=lambda identity: (backend, "http://replica/readyz"),
        event_sink=events.append,
    )
    await reconciler.reconcile()
    assert [event["reason"] for event in events] == ["replica_added"]

    await pool.probe("replica")
    await reconciler.reconcile()
    assert [event["reason"] for event in events] == [
        "replica_added",
        "probe_succeeded",
    ]
    await pool.probe("replica")
    await pool.probe("replica")
    await reconciler.reconcile()
    assert [event["reason"] for event in events].count("probe_succeeded") == 1

    backend.failing = True
    with pytest.raises(RuntimeError, match="backend failed"):
        await pool.generate(
            GenerationRequest(
                request_id="failure",
                prompt="p",
                sampling_params=SamplingParams(),
            )
        )
    await reconciler.reconcile()
    assert events[-1]["reason"] == "health_ejected"
    assert events[-1]["ineligible"] == ["replica"]

    backend.failing = False
    await pool.probe("replica")
    await reconciler.reconcile()
    assert events[-1]["reason"] == "probe_succeeded"
    assert events[-1]["eligible"] == ["replica"]

    pool.drain("replica")
    await reconciler.reconcile()
    assert events[-1]["reason"] == "manual_drain"
    pool.cancel_drain("replica")
    await reconciler.reconcile()
    assert events[-1]["reason"] == "manual_undrain"


@pytest.mark.asyncio
async def test_manual_drain_replacement_never_reports_new_generation_eligible() -> None:
    class Source:
        config = ReplicaConfig("http://old/v1", "model")

        def poll(self):
            return {"replica": self.config}

    source = Source()
    pool = ReplicaPool({}, allow_empty=True)
    events: list[dict[str, object]] = []
    reconciler = PoolReconciler(
        pool,
        source,
        factory=lambda identity: (_Backend(), None),
        event_sink=events.append,
    )
    await reconciler.reconcile()
    pool.drain("replica")
    await reconciler.reconcile()
    events.clear()

    source.config = ReplicaConfig("http://new/v1", "model")
    await reconciler.reconcile()

    replacement_add = next(
        event for event in events if event["reason"] == "replica_added"
    )
    assert replacement_add["eligible"] == []
    assert replacement_add["eligible_ids"] == []
    assert "manual_drain" not in {
        event["reason"] for event in events
    }


@pytest.mark.asyncio
async def test_run_is_immediate_resilient_and_propagates_cancellation() -> None:
    class FlakySource:
        def __init__(self) -> None:
            self.polls = 0
            self.closed = False

        async def poll(self):
            self.polls += 1
            if self.polls == 1:
                raise httpx.ConnectError("temporary apiserver failure")
            return {"new": ReplicaConfig("http://10.0.0.2:8000/v1", "model")}

        async def aclose(self) -> None:
            self.closed = True

    source = FlakySource()
    added = asyncio.Event()

    def sink(event: dict[str, list[str]]) -> None:
        if event["added"]:
            added.set()

    reconciler = PoolReconciler(
        ReplicaPool({}, allow_empty=True),
        source,
        factory=lambda identity: (_Backend(), None),
        event_sink=sink,
    )
    task = asyncio.create_task(reconciler.run(0.001))
    await asyncio.wait_for(added.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.polls >= 2
    assert source.closed is True


@pytest.mark.asyncio
async def test_reconciler_shutdown_releases_owned_drain_lease() -> None:
    backend = _Backend()
    pool = ReplicaPool({"busy": backend})
    pool._entries["busy"].outstanding = 1

    class EmptySource:
        def poll(self):
            return {}

    reconciler = PoolReconciler(pool, EmptySource())
    assert (await reconciler.reconcile())["draining"] == ["busy"]
    assert pool.is_draining("busy") is True

    await reconciler.aclose()
    assert pool.is_draining("busy") is False
    with pytest.raises(RuntimeError, match="closed"):
        await reconciler.reconcile()


@pytest.mark.asyncio
async def test_owned_injected_client_is_closed_idempotently(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"items": []})
        )
    )
    source = KubernetesEndpointSliceDiscovery(
        "replicas",
        "default",
        8000,
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        close_client=True,
    )

    await source.aclose()
    await source.aclose()
    assert client.is_closed is True
    with pytest.raises(RuntimeError, match="closed"):
        await source.poll()


@pytest.mark.asyncio
async def test_cancelled_endpoint_discovery_close_finishes_owned_client(
    tmp_path: Path,
) -> None:
    class BlockingClient:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = False

        async def aclose(self) -> None:
            self.started.set()
            await self.release.wait()
            self.closed = True

    token_path = tmp_path / "token"
    token_path.write_text("token", encoding="utf-8")
    client = BlockingClient()
    source = KubernetesEndpointSliceDiscovery(
        "replicas",
        "default",
        8000,
        api_server="https://kubernetes.example",
        token_path=token_path,
        client=client,
        close_client=True,
    )

    task = asyncio.create_task(source.aclose())
    await asyncio.wait_for(client.started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    client.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed is True
    await source.aclose()


@pytest.mark.asyncio
async def test_cancelled_reconciler_close_finishes_sink_and_source() -> None:
    class Source:
        def __init__(self) -> None:
            self.closed = False

        def poll(self):
            return {}

        async def aclose(self) -> None:
            self.closed = True

    started = asyncio.Event()
    release = asyncio.Event()
    delivered = []

    async def sink(event) -> None:
        started.set()
        await release.wait()
        delivered.append(event)

    source = Source()
    pool = ReplicaPool({}, allow_empty=True)
    reconciler = PoolReconciler(pool, source, event_sink=sink)
    pool.add_replica("replica", _Backend())

    task = asyncio.create_task(reconciler.aclose())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(delivered) == 1
    assert source.closed is True
    assert pool._membership_event_sink is None
    await reconciler.aclose()


@pytest.mark.asyncio
async def test_cancelled_candidate_cleanup_finishes_before_propagating() -> None:
    class MutableSource:
        config = ReplicaConfig("http://old:8000/v1", "model")

        def poll(self):
            return {"replica": self.config}

    class BlockingCandidate(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.was_cancelled = False

        async def shutdown(self) -> None:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.was_cancelled = True
                raise
            self.closed = True

    source = MutableSource()
    pool = ReplicaPool({"replica": _Backend()})
    candidate = BlockingCandidate()
    reconciler = PoolReconciler(
        pool,
        source,
        factory=lambda identity: (candidate, None),
    )
    await reconciler.reconcile()
    pool._entries["replica"].outstanding = 1
    source.config = ReplicaConfig("http://new:8000/v1", "model")

    task = asyncio.create_task(reconciler.reconcile())
    await asyncio.wait_for(candidate.started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    candidate.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert candidate.closed is True
    assert candidate.was_cancelled is False
