"""Serve-layer health prober for unknown/ejected ReplicaPool members (design m7 D4).

Lives outside the pool on purpose: ``ReplicaPool`` stays pure hashing with no
background tasks (m5 D4). The prober GETs each ejected replica's readiness URL
(``/readyz`` — so a drained or wedged-but-alive node is NOT restored, O3) and on
200 calls the pool's existing ``probe(id)``.

Keyed by replica id plus entry generation, never ordinal (O2): the pool has
dynamic membership, so an ordinal frozen at construction can restore the wrong
replica or IndexError after an add/remove. URLs are resolved per id at snapshot
time, preferring the current entry's ``health_url`` over the initial URL map.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence

import httpx

from kairyu.orchestration.replica import ReplicaPool

logger = logging.getLogger("kairyu.prober")

_PROBE_TIMEOUT_S = 5.0
_DEFAULT_MAX_CONCURRENCY = 16


class HealthProber:
    def __init__(
        self,
        pool_name: str,
        pool: ReplicaPool,
        health_urls: Sequence[str | None] | Mapping[str, str | None],
        interval_s: float,
        client: httpx.AsyncClient | None = None,
        *,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
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
        self._max_concurrency = max_concurrency
        for replica_id in pool.replica_ids:
            if self._url_for(replica_id) is not None:
                pool.require_probe(replica_id)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S)
        return self._client

    def _url_for(self, replica_id: str) -> str | None:
        url = self._pool.health_url(replica_id)
        if url is None:
            url = self._health_urls.get(replica_id)
        return url

    def _candidate_snapshot(self) -> tuple[tuple[str, object, str], ...]:
        candidates = []
        for replica_id, healthy in self._pool.healthy_by_id().items():
            if healthy:
                continue
            try:
                generation = self._pool.entry_generation(replica_id)
                url = self._url_for(replica_id)
            except ValueError:
                continue  # membership changed while this tick was taking its snapshot
            if url is not None:
                candidates.append((replica_id, generation, url))
        return tuple(candidates)

    async def _check_candidate(
        self,
        replica_id: str,
        generation: object,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> str | None:
        async with semaphore:
            try:
                response = await self._get_client().get(url)
            except httpx.HTTPError:
                return None
            except Exception:
                logger.exception(
                    "replica probe request failed",
                    extra={"pool": self._pool_name, "replica": replica_id, "url": url},
                )
                return None
        if response.status_code != 200:
            return None
        try:
            if self._pool.entry_generation(replica_id) is not generation:
                return None
            await self._pool.probe(replica_id)
        except ValueError:
            return None  # removed after the request completed
        except Exception:
            logger.exception(
                "replica restore failed",
                extra={"pool": self._pool_name, "replica": replica_id, "url": url},
            )
            return None
        logger.info(
            "replica restored",
            extra={"pool": self._pool_name, "replica": replica_id, "url": url},
        )
        return replica_id

    async def check_once(self) -> tuple[str, ...]:
        """Probe an id/generation snapshot of every unknown or ejected replica."""
        semaphore = asyncio.Semaphore(self._max_concurrency)
        results = await asyncio.gather(
            *(
                self._check_candidate(replica_id, generation, url, semaphore)
                for replica_id, generation, url in self._candidate_snapshot()
            )
        )
        return tuple(replica_id for replica_id in results if replica_id is not None)

    async def run(self) -> None:
        """Probe loop; cancelled by the app lifespan on shutdown."""
        try:
            while True:
                try:
                    await self.check_once()
                except Exception:  # one bad tick must never kill the prober (O2)
                    logger.exception(
                        "prober tick failed", extra={"pool": self._pool_name}
                    )
                await asyncio.sleep(self._interval_s)
        finally:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
