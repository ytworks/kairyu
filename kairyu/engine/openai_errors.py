"""HTTP error mapping for OpenAI-compatible upstreams (M20 WP-04).

An upstream 4xx is a client-request error, never a replica health signal (O1).
A 400 that reports a prompt overflowing the upstream's context window becomes
``UpstreamClientError(code="context_length_exceeded")`` with a fixed public
message, so every surface can classify it without exposing upstream text.
"""

from __future__ import annotations

import json
import re

from kairyu.engine.backend import UpstreamClientError
from kairyu.engine.request_errors import CONTEXT_LENGTH_EXCEEDED

# Message shapes of a context-window overflow: OpenAI and vLLM's serving layer
# ("maximum context length"), vLLM's V1 input processor ("longer than the
# maximum model length"), and Kairyu replicas that predate the typed code.
_CONTEXT_OVERFLOW_MESSAGE = re.compile(
    r"maximum context length"
    r"|longer than the maximum model length"
    r"|exceed max_model_len \(\d+\)"
    r"|already fill max_model_len \(\d+\)",
    re.IGNORECASE,
)
# Error bodies are small; never parse an unexpectedly large payload.
_MAX_CLASSIFIED_BODY_CHARS = 64 * 1024
UPSTREAM_CONTEXT_OVERFLOW_MESSAGE = (
    "the request exceeds the model's maximum context length"
)


def _error_object(body: str) -> dict | None:
    """Return the OpenAI-style error object of a nested or flat JSON body."""

    if len(body) > _MAX_CLASSIFIED_BODY_CHARS:
        return None
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    nested = payload.get("error")
    return nested if isinstance(nested, dict) else payload


def is_context_overflow_body(body: str) -> bool:
    """Whether an upstream 400 body reports a context-window overflow."""

    error = _error_object(body)
    if error is None:
        return False
    if error.get("code") == CONTEXT_LENGTH_EXCEEDED:
        return True
    message = error.get("message")
    return isinstance(message, str) and bool(_CONTEXT_OVERFLOW_MESSAGE.search(message))


def classify_upstream_client_error(
    base_url: str,
    status_code: int,
    body: str,
) -> UpstreamClientError:
    """Build the typed client error for one upstream 4xx reply.

    The private message keeps a bounded body excerpt for server-side logs;
    only the fixed overflow message may cross the tenant boundary.
    """

    message = f"backend {base_url} returned HTTP {status_code}: {body[:500]}"
    if status_code == 400 and is_context_overflow_body(body):
        return UpstreamClientError(
            message,
            status_code,
            code=CONTEXT_LENGTH_EXCEEDED,
            public_message=UPSTREAM_CONTEXT_OVERFLOW_MESSAGE,
        )
    return UpstreamClientError(message, status_code)


def raise_for_status(base_url: str, status_code: int, body: str) -> None:
    """4xx is a client-request error (not a replica health signal, O1); 5xx and
    everything else is a transport/server failure the pool should count."""

    if 400 <= status_code < 500:
        raise classify_upstream_client_error(base_url, status_code, body)
    raise RuntimeError(f"backend {base_url} returned HTTP {status_code}: {body[:500]}")
