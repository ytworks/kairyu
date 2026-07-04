"""Serve-layer health prober for ejected ReplicaPool members (design m7 D4).

Lives outside the pool on purpose: ``ReplicaPool`` stays pure hashing with no
background tasks (m5 D4). The prober GETs each ejected replica's readiness URL
(``/readyz`` — so a drained or wedged-but-alive node is NOT restored, O3) and on
200 calls the pool's existing ``probe(id)``.

Keyed by replica id, never ordinal (O2): the pool has dynamic membership, so an
ordinal frozen at construction can restore the wrong replica or IndexError after
an add/remove. URLs are resolved per id at check time, preferring an explicit
map and falling back to the pool's own per-entry ``health_url``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence

import httpx

from kairyu.orchestration.replica import ReplicaPool

logger = logging.getLogger("kairyu.prober")

_PROBE_TIMEOUT_S = 5.0


class HealthProber:
    def __init__(
        self,
        pool_name: str,
        pool: ReplicaPool,
        health_urls: Sequence[str | None] | Mapping[str, str | None],
        interval_s: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if isinstance(health_urls, Mapping):
            self._health_urls: dict[str, str | None] = dict(health_urls)
        else:
            health_urls = tuple(health_urls)
            if len(health_urls) != pool.replica_count:
                raise ValueError(
                    f"got {len(health_urls)} health URLs for {pool.replica_count} replicas"
                )
            # bind positional URLs to the current ids once; dynamically-added
            # replicas resolve their URL from the pool at check time
            self._health_urls = dict(zip(pool.replica_ids, health_urls, strict=True))
        self._pool_name = pool_name
        self._pool = pool
        self._interval_s = interval_s
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S)
        return self._client

    def _url_for(self, replica_id: str) -> str | None:
        url = self._health_urls.get(replica_id)
        if url is None:
            url = self._pool.health_url(replica_id)  # registry-added replicas
        return url

    async def check_once(self) -> tuple[str, ...]:
        """Probe every ejected replica once; returns the ids restored."""
        restored = []
        for replica_id, healthy in self._pool.healthy_by_id().items():
            if healthy:
                continue
            url = self._url_for(replica_id)
            if url is None:
                continue  # unprobeable member (no readiness endpoint declared)
            try:
                response = await self._get_client().get(url)
            except httpx.HTTPError:
                continue
            if response.status_code == 200:
                await self._pool.probe(replica_id)
                restored.append(replica_id)
                logger.info(
                    "replica restored",
                    extra={"pool": self._pool_name, "replica": replica_id, "url": url},
                )
        return tuple(restored)

    async def run(self) -> None:
        """Probe loop; cancelled by the app lifespan on shutdown."""
        try:
            while True:
                await asyncio.sleep(self._interval_s)
                try:
                    await self.check_once()
                except Exception:  # one bad tick must never kill the prober (O2)
                    logger.exception(
                        "prober tick failed", extra={"pool": self._pool_name}
                    )
        finally:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
