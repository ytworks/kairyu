"""Multi-tenant limits + usage metering (m11 D3).

Identity: AuthMiddleware stores the matched key in scope state (A6);
``TenantLimitMiddleware`` maps key → tenant and enforces a per-tenant token
bucket on /v1/* — it runs INSIDE auth (401 wins over 429; unauthenticated
requests never drain buckets). Keyless mode maps everything to "default".

Ledger (A7): O_APPEND single-writer JSONL, one record per request, written
from the handlers (middleware cannot see token usage); batch-worker
executions are NOT metered in v1 (recorded in the design).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TenantLimits:
    requests_per_minute: int = 600
    tokens_per_minute: int = 200_000


@dataclass(frozen=True)
class TenantConfig:
    """key -> tenant mapping plus per-tenant limits."""

    key_tenants: dict[str, str] = field(default_factory=dict)
    limits: dict[str, TenantLimits] = field(default_factory=dict)
    default_tenant: str = "default"

    def tenant_for_key(self, api_key: str | None) -> str:
        if api_key is None:
            return self.default_tenant
        return self.key_tenants.get(api_key, self.default_tenant)

    def limits_for(self, tenant: str) -> TenantLimits:
        return self.limits.get(tenant, TenantLimits())


class _Bucket:
    def __init__(self, per_minute: int, now: Callable[[], float]) -> None:
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self.updated = now()

    def take(self, amount: float, now: float) -> bool:
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True


class TenantLimiter:
    """Per-tenant request-rate token buckets (single-gateway; the distributed
    limiter is a G6 note)."""

    def __init__(self, config: TenantConfig, now: Callable[[], float] = time.monotonic):
        self._config = config
        self._now = now
        self._buckets: dict[str, _Bucket] = {}

    def admit(self, tenant: str) -> bool:
        bucket = self._buckets.get(tenant)
        if bucket is None:
            bucket = _Bucket(self._config.limits_for(tenant).requests_per_minute, self._now)
            self._buckets[tenant] = bucket
        return bucket.take(1.0, self._now())


class TenantLimitMiddleware:
    """Pure ASGI; requires auth to have stored the key in scope state (A6)."""

    def __init__(self, app, *, config: TenantConfig, limiter: TenantLimiter) -> None:
        self.app = app
        self._config = config
        self._limiter = limiter

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or not (
            path.startswith("/v1/") or path.startswith("/admin/usage")
        ):
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        tenant = self._config.tenant_for_key(state.get("api_key"))
        state["tenant"] = tenant
        if not path.startswith("/v1/"):
            await self.app(scope, receive, send)  # identity only, no bucket
            return
        if not self._limiter.admit(tenant):
            body = json.dumps(
                {
                    "error": {
                        "message": f"tenant {tenant!r} rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "tenant_rate_limited",
                    }
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"retry-after", b"1"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


class UsageLedger:
    """O_APPEND single-writer JSONL (A7): one line per completed request."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self, tenant: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        line = json.dumps(
            {
                "tenant": tenant,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ts": time.time(),
            }
        )
        with open(self._path, "a", encoding="utf-8") as handle:  # O_APPEND
            handle.write(line + "\n")

    def totals(self, tenant: str | None = None) -> dict[str, dict[str, int]]:
        """Aggregate by tenant (optionally filtered) for /admin/usage."""
        totals: dict[str, dict[str, int]] = {}
        if not self._path.is_file():
            return totals
        with open(self._path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if tenant is not None and record["tenant"] != tenant:
                    continue
                bucket = totals.setdefault(
                    record["tenant"],
                    {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0},
                )
                bucket["requests"] += 1
                bucket["prompt_tokens"] += record["prompt_tokens"]
                bucket["completion_tokens"] += record["completion_tokens"]
        return totals
