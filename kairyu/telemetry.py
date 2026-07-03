"""Tracing seam usable from ANY layer (m10a D4/A9 — L2 must not import L3).

``traced_span`` is a context manager: a real OTel span when tracing is
enabled AND opentelemetry is importable, a no-op otherwise (the server runs
without the dependency). Enablement is process-global and explicit —
``configure_tracing(True)`` is called by the server from ServerSettings.
"""

from __future__ import annotations

from contextlib import contextmanager

_ENABLED = False


def configure_tracing(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def tracing_enabled() -> bool:
    return _ENABLED


@contextmanager
def traced_span(name: str, attributes: dict | None = None):
    if not _ENABLED:
        yield None
        return
    try:
        from opentelemetry import trace  # deferred: otel extra / dev group
    except ImportError:
        yield None
        return
    tracer = trace.get_tracer("kairyu")
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        yield span
