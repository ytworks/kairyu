"""Serve-layer settings for `create_app` (design m7 D4/D5/D8).

All fields default to the pre-M7 behavior (no auth, no concurrency cap,
metrics on) so existing callers of ``create_app(engines)`` are unchanged.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field


class ServerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_keys_env: str | None = Field(
        default=None,
        description=(
            "Env var holding comma-separated API keys; None disables auth "
            "(keyless node-to-node replicas, design m6 D2 / m7 D5)."
        ),
    )
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description="Global in-flight cap on /v1/* requests; None disables the guard.",
    )
    metrics: bool = Field(default=True, description="Expose /metrics (Prometheus).")
    protect_metrics: bool = Field(
        default=False, description="Require an API key for /metrics too."
    )
    access_log: bool = Field(
        default=True, description="Emit one JSON access-log line per request."
    )

    def resolve_api_keys(self) -> frozenset[str]:
        """Read keys from the configured env var; fail loud on an empty var."""
        if self.api_keys_env is None:
            return frozenset()
        raw = os.environ.get(self.api_keys_env, "")
        keys = frozenset(key.strip() for key in raw.split(",") if key.strip())
        if not keys:
            raise ValueError(
                f"api_keys_env={self.api_keys_env!r} is set but the env var "
                "contains no keys; unset api_keys_env to serve without auth"
            )
        return keys
