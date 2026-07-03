"""Serve-layer health prober for ejected ReplicaPool members (design m7 D4).

Lives outside the pool on purpose: ``ReplicaPool`` stays pure hashing with no
background tasks (m5 D4). The prober GETs each ejected replica's ``/health``
(replica nodes run the same server, so the endpoint exists by construction)
and on 200 calls the pool's existing ``probe(index)``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import httpx

from kairyu.orchestration.replica import ReplicaPool

logger = logging.getLogger("kairyu.prober")

_PROBE_TIMEOUT_S = 5.0


class HealthProber:
    def __init__(
        self,
        pool_name: str,
        pool: ReplicaPool,
        health_urls: Sequence[str | None],
        interval_s: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if len(health_urls) != pool.replica_count:
            raise ValueError(
                f"got {len(health_urls)} health URLs for {pool.replica_count} replicas"
            )
        self._pool_name = pool_name
        self._pool = pool
        self._health_urls = tuple(health_urls)
        self._interval_s = interval_s
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S)
        return self._client

    async def check_once(self) -> tuple[int, ...]:
        """Probe every ejected replica once; returns the indices restored."""
        restored = []
        for index, healthy in enumerate(self._pool.healthy):
            if healthy:
                continue
            url = self._health_urls[index]
            if url is None:
                continue  # unprobeable member (no health endpoint declared)
            try:
                response = await self._get_client().get(url)
            except httpx.HTTPError:
                continue
            if response.status_code == 200:
                await self._pool.probe(index)
                restored.append(index)
                logger.info(
                    "replica restored",
                    extra={"pool": self._pool_name, "replica": index, "url": url},
                )
        return tuple(restored)

    async def run(self) -> None:
        """Probe loop; cancelled by the app lifespan on shutdown."""
        try:
            while True:
                await asyncio.sleep(self._interval_s)
                await self.check_once()
        finally:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
