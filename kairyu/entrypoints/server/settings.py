"""Serve-layer settings for `create_app` (design m7 D4/D5/D8).

All fields default to the pre-M7 behavior (no auth, no concurrency cap,
metrics on) so existing callers of ``create_app(engines)`` are unchanged.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

# The size cap, compressed and decoded, of a Content-Encoding request body (m20 D21, WP-41).
DEFAULT_MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024
# A browser Origin: scheme://host[:port], never a path or a trailing slash. Any
# RFC 3986 scheme, so desktop and extension webviews (tauri://localhost,
# vscode-webview://<id>, chrome-extension://<id>) can be listed too.
_CORS_ORIGIN = re.compile(r"[a-z][a-z0-9+.-]*://[^/\s]+")


def _check_cors_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [o for o in origins if o != "*" and not _CORS_ORIGIN.fullmatch(o)]
    if invalid:
        raise ValueError(
            "cors_allowed_origins entries are '*' or scheme://host[:port] "
            f"without a path: {invalid}"
        )
    return origins


# Shared with the deployment ServerSection, so a typo fails at load time there too.
CorsOrigins = Annotated[tuple[str, ...], AfterValidator(_check_cors_origins)]


class ServerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_keys_env: str | None = Field(
        default=None,
        description=(
            "Env var holding comma-separated API keys; None disables auth "
            "(keyless node-to-node replicas, design m6 D2 / m7 D5)."
        ),
    )
    responses_compaction_secret_env: str | None = Field(
        default=None,
        description=(
            "Env var holding a secret used to encrypt Responses compaction tokens; "
            "None uses a process-local ephemeral key."
        ),
    )
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Global active-plus-queued cap on /v1/* requests; None disables "
            "the guard."
        ),
    )
    admission_wait_timeout_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Maximum wait for a backend-aware active request slot; None "
            "preserves immediate saturation rejection."
        ),
    )
    ttft_slo_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "TTFT target for direct-chat SLO admission; None disables "
            "predictive admit/defer/shed decisions."
        ),
    )
    max_chat_body_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum raw POST body for /v1/chat/completions; None disables "
            "the process-level guard."
        ),
    )
    max_decompressed_bytes: int = Field(
        default=DEFAULT_MAX_DECOMPRESSED_BYTES,
        ge=1,
        description=(
            "Maximum size of a gzip or zstd (Content-Encoding) request body, both "
            "compressed and decoded; a larger body is refused with 413 before it "
            "is fully inflated."
        ),
    )
    cors_allowed_origins: CorsOrigins = Field(
        default=(),
        description=(
            "Browser origins allowed cross-origin ('*' for any). Empty installs "
            "no CORS handling; when set, preflights are answered before auth."
        ),
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

    @model_validator(mode="after")
    def _admission_queue_requires_total_bound(self) -> ServerSettings:
        if self.admission_wait_timeout_s is not None and self.max_concurrency is None:
            raise ValueError(
                "admission_wait_timeout_s requires max_concurrency"
            )
        return self

    def resolve_responses_compaction_key(self) -> bytes:
        """Resolve a 256-bit AEAD key without exposing the configured secret."""
        env_var = self.responses_compaction_secret_env
        if env_var is None:
            return secrets.token_bytes(32)
        raw = os.environ.get(env_var, "")
        secret = raw.encode("utf-8")
        if len(secret) < 32:
            raise ValueError(
                f"Responses compaction secret env var {env_var!r} must contain "
                "at least 32 UTF-8 bytes"
            )
        return hashlib.sha256(
            b"kairyu.responses.compaction.key.v1\0" + secret
        ).digest()

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
