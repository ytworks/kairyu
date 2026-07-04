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
    tracing: bool = Field(
        default=False,
        description="Enable OTel spans (needs the otel extra; no-op without it).",
    )
    usage_ledger_path: str | None = Field(
        default=None,
        description="JSONL usage-ledger path; None disables metering (m11 D3).",
    )
    admin_keys_env: str | None = Field(
        default=None,
        description=(
            "Env var holding comma-separated ADMIN API keys; when set, /admin/* "
            "state changes (drain/undrain) require one of these, so an ordinary "
            "data-plane key cannot take the node out of service (S5)."
        ),
    )

    def resolve_api_keys(self) -> frozenset[str]:
        """Read keys from the configured env var; fail loud on an empty var."""
        return self._resolve_keys(self.api_keys_env)

    def resolve_admin_keys(self) -> frozenset[str]:
        """Admin keys for /admin/* mutations; empty when unconfigured."""
        return self._resolve_keys(self.admin_keys_env)

    @staticmethod
    def _resolve_keys(env_var: str | None) -> frozenset[str]:
        if env_var is None:
            return frozenset()
        raw = os.environ.get(env_var, "")
        keys = frozenset(key.strip() for key in raw.split(",") if key.strip())
        if not keys:
            raise ValueError(
                f"key env var {env_var!r} is set but contains no keys; "
                "unset it to disable that key set"
            )
        return keys
