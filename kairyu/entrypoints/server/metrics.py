"""Prometheus metrics for the serve layer (design m7 D8).

Each app gets its own ``CollectorRegistry`` so multiple ``create_app`` calls
(tests, embedded use) never collide on timeseries names. Pool gauges are read
at scrape time through ``ReplicaPool``'s read-only accessors — the pool stays
passive (m5 D4).
"""

from __future__ import annotations

from collections.abc import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric

from kairyu.orchestration.replica import ReplicaPool


class _PoolCollector:
    """Scrape-time view over tracked ReplicaPools (no background sampling)."""

    def __init__(self) -> None:
        self._pools: dict[str, ReplicaPool] = {}

    def add(self, name: str, pool: ReplicaPool) -> None:
        self._pools[name] = pool

    def collect(self) -> Iterator[Metric]:
        outstanding = GaugeMetricFamily(
            "kairyu_replica_outstanding",
            "In-flight requests per replica",
            labels=["pool", "replica"],
        )
        healthy = GaugeMetricFamily(
            "kairyu_replica_healthy",
            "Replica health (1 = in the hash ring)",
            labels=["pool", "replica"],
        )
        decisions = CounterMetricFamily(
            "kairyu_pool_decisions",
            "Placement decisions by reason (session_affinity is the cache-affinity signal)",
            labels=["pool", "reason"],
        )
        for name, pool in self._pools.items():
            for index, count in enumerate(pool.outstanding):
                outstanding.add_metric([name, str(index)], count)
            for index, is_healthy in enumerate(pool.healthy):
                healthy.add_metric([name, str(index)], 1.0 if is_healthy else 0.0)
            for reason, count in pool.decision_counts.items():
                decisions.add_metric([name, reason], count)
        yield outstanding
        yield healthy
        yield decisions


class _SchedulerCollector:
    """Scrape-time native scheduler view with bounded class/event labels."""

    _CLASSES = ("interactive", "batch")
    _EVENTS = ("enqueue", "admit", "preempt", "complete")

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}

    def add(self, name: str, engine: object) -> None:
        if callable(getattr(engine, "scheduler_priority_metrics", None)):
            self._engines[name] = engine

    def collect(self) -> Iterator[Metric]:
        events = CounterMetricFamily(
            "kairyu_scheduler_priority_events",
            "Native scheduler events by bounded request class",
            labels=["model", "request_class", "event"],
        )
        depth = GaugeMetricFamily(
            "kairyu_scheduler_queue_depth",
            "Current native scheduler waiting depth by bounded request class",
            labels=["model", "request_class"],
        )
        high_watermark = GaugeMetricFamily(
            "kairyu_scheduler_queue_high_watermark",
            "Maximum native scheduler waiting depth by bounded request class",
            labels=["model", "request_class"],
        )
        for model, engine in self._engines.items():
            snapshot = engine.scheduler_priority_metrics()
            if not snapshot:
                continue
            event_counts = snapshot["events"]
            for request_class in self._CLASSES:
                for event in self._EVENTS:
                    events.add_metric(
                        [model, request_class, event],
                        event_counts.get((request_class, event), 0),
                    )
                depth.add_metric(
                    [model, request_class],
                    snapshot["queue_depth"].get(request_class, 0),
                )
                high_watermark.add_metric(
                    [model, request_class],
                    snapshot["queue_high_watermark"].get(request_class, 0),
                )
        yield events
        yield depth
        yield high_watermark


class _TenantLimiterCollector:
    """Scrape-time view of reservations not yet settled or consumed."""

    def __init__(self) -> None:
        self._limiter = None

    def set(self, limiter: object) -> None:
        self._limiter = limiter

    def collect(self) -> Iterator[Metric]:
        reserved = GaugeMetricFamily(
            "kairyu_tenant_reserved_tokens",
            "Worst-case tenant compute tokens reserved before shared dispatch",
            labels=["tenant"],
        )
        violations = CounterMetricFamily(
            "kairyu_tenant_reservation_bound_violations",
            "Actual tenant work exceeding its pre-dispatch reservation",
            labels=["tenant"],
        )
        if self._limiter is not None:
            snapshot = self._limiter.reservation_snapshot()
            for tenant, tokens in sorted(snapshot.items()):
                reserved.add_metric([tenant], tokens)
            for tenant, count in sorted(
                self._limiter.bound_violation_snapshot().items()
            ):
                violations.add_metric([tenant], count)
        yield reserved
        yield violations


class ServerMetrics:
    """Registry + the request-level metrics recorded by the middleware."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "kairyu_requests",
            "API requests by served model and HTTP status",
            ["model", "code"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "kairyu_request_duration_seconds",
            "Request wall time by path",
            ["path"],
            registry=self.registry,
        )
        self.batch_jobs_total = Counter(
            "kairyu_batch_jobs",
            "Batch jobs by terminal state",
            ["state"],
            registry=self.registry,
        )
        self.priority_requests_total = Counter(
            "kairyu_priority_requests",
            "Requests dispatched by bounded scheduling class and ingress source",
            ["request_class", "source"],
            registry=self.registry,
        )
        self.tenant_admission_total = Counter(
            "kairyu_tenant_admission",
            "Tenant admission decisions before shared downstream capacity",
            ["tenant", "source", "decision", "reason"],
            registry=self.registry,
        )
        self.tenant_in_flight_requests = Gauge(
            "kairyu_tenant_in_flight_requests",
            "Admitted tenant requests currently holding downstream capacity",
            ["tenant", "source"],
            registry=self.registry,
        )
        self.usage_requests_total = Counter(
            "kairyu_usage_requests",
            "Successful metered executions by tenant",
            ["tenant"],
            registry=self.registry,
        )
        self.usage_tokens_total = Counter(
            "kairyu_usage_tokens",
            "Metered tokens by tenant and token type",
            ["tenant", "type"],
            registry=self.registry,
        )
        self._pool_collector = _PoolCollector()
        self._scheduler_collector = _SchedulerCollector()
        self._tenant_limiter_collector = _TenantLimiterCollector()
        self.registry.register(self._pool_collector)
        self.registry.register(self._scheduler_collector)
        self.registry.register(self._tenant_limiter_collector)

    def track_pool(self, name: str, pool: ReplicaPool) -> None:
        self._pool_collector.add(name, pool)

    def track_scheduler(self, name: str, engine: object) -> None:
        self._scheduler_collector.add(name, engine)

    def track_tenant_limiter(self, limiter: object) -> None:
        self._tenant_limiter_collector.set(limiter)

    def record_priority(self, request_class: str, *, source: str) -> None:
        """Record an explicit bounded class without labeling priority integers."""

        if request_class not in {"interactive", "batch"}:
            raise ValueError(f"invalid scheduling class {request_class!r}")
        self.priority_requests_total.labels(
            request_class=request_class,
            source=source,
        ).inc()

    def record_tenant_admission(
        self,
        tenant: str,
        *,
        source: str,
        admitted: bool,
        reason: str,
    ) -> None:
        decision = "admitted" if admitted else "rejected"
        self.tenant_admission_total.labels(
            tenant=tenant,
            source=source,
            decision=decision,
            reason=reason,
        ).inc()
        in_flight = self.tenant_in_flight_requests.labels(
            tenant=tenant,
            source=source,
        )
        if admitted:
            in_flight.inc()

    def record_tenant_release(self, tenant: str, *, source: str) -> None:
        self.tenant_in_flight_requests.labels(
            tenant=tenant,
            source=source,
        ).dec()

    def record_usage(
        self,
        tenant: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
    ) -> None:
        """Mirror one accepted ledger row at the tenant aggregation boundary."""
        self.usage_requests_total.labels(tenant=tenant).inc()
        self.usage_tokens_total.labels(tenant=tenant, type="prompt").inc(
            prompt_tokens
        )
        self.usage_tokens_total.labels(tenant=tenant, type="completion").inc(
            completion_tokens
        )
        self.usage_tokens_total.labels(tenant=tenant, type="cached").inc(
            cached_tokens
        )
        self.usage_tokens_total.labels(tenant=tenant, type="uncached").inc(
            prompt_tokens - cached_tokens
        )

    def restore_usage_totals(
        self,
        totals: dict[str, dict[str, int]],
    ) -> None:
        """Restore process-local counters from the append-only ledger on startup."""
        for tenant, usage in totals.items():
            self.usage_requests_total.labels(tenant=tenant).inc(
                usage["requests"]
            )
            self.usage_tokens_total.labels(tenant=tenant, type="prompt").inc(
                usage["prompt_tokens"]
            )
            self.usage_tokens_total.labels(
                tenant=tenant,
                type="completion",
            ).inc(usage["completion_tokens"])
            self.usage_tokens_total.labels(tenant=tenant, type="cached").inc(
                usage["cached_tokens"]
            )
            self.usage_tokens_total.labels(tenant=tenant, type="uncached").inc(
                usage["uncached_tokens"]
            )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
