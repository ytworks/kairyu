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
        self.registry.register(self._pool_collector)

    def track_pool(self, name: str, pool: ReplicaPool) -> None:
        self._pool_collector.add(name, pool)

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
