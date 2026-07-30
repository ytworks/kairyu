"""Tracing seam usable from ANY layer (m10a D4/A9 — L2 must not import L3).

OpenTelemetry remains an optional dependency: every OTel import is deferred
until tracing is both configured and used.  A caller may inject a private
``TracerProvider`` for a deterministic fixture or exporter without replacing
the process-global provider.

The helpers deliberately use only the W3C Trace Context propagator.  Baggage is
not extracted or injected, which keeps arbitrary caller data out of Kairyu's
cross-service telemetry boundary.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from typing import Literal

_ENABLED = False
_TRACER_PROVIDER: object | None = None
_CONSOLE_PROVIDER: object | None = None
CONSOLE_SPAN_PREFIX = "KAIRYU_OTEL_SPAN "

SpanKindName = Literal["internal", "server", "client", "producer", "consumer"]


def _console_span_line(span: object) -> str:
    """Format one grep-safe JSON line for the opt-in deployed smoke exporter."""

    context = span.get_span_context()
    parent = span.parent
    service_name = span.resource.attributes.get("service.name", "kairyu")
    payload = {
        "name": span.name,
        "service_name": service_name,
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
        "parent_span_id": (
            f"{parent.span_id:016x}" if parent is not None and parent.is_valid else None
        ),
        "kind": span.kind.name,
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
    }
    return CONSOLE_SPAN_PREFIX + json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _console_provider_from_env() -> object | None:
    """Build one private SDK provider only for ``OTEL_TRACES_EXPORTER=console``."""

    global _CONSOLE_PROVIDER
    exporters = {
        item.strip()
        for item in os.environ.get("OTEL_TRACES_EXPORTER", "").split(",")
        if item.strip()
    }
    if "console" not in exporters:
        return None
    if _CONSOLE_PROVIDER is not None:
        return _CONSOLE_PROVIDER
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        return None
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": os.environ.get("OTEL_SERVICE_NAME", "kairyu")}
        )
    )
    provider.add_span_processor(
        SimpleSpanProcessor(
            ConsoleSpanExporter(
                out=sys.stdout,
                formatter=_console_span_line,
            )
        )
    )
    _CONSOLE_PROVIDER = provider
    return provider


def configure_tracing(
    enabled: bool,
    tracer_provider: object | None = None,
) -> None:
    """Enable tracing and optionally select a private tracer provider.

    ``configure_tracing(True/False)`` retains the original public contract.
    Supplying ``tracer_provider`` changes only this module's tracer source and
    never calls OpenTelemetry's process-global ``set_tracer_provider``.
    Disabling tracing also releases the injected provider.
    """

    global _ENABLED, _TRACER_PROVIDER
    _ENABLED = bool(enabled)
    if not _ENABLED:
        _TRACER_PROVIDER = None
    else:
        _TRACER_PROVIDER = (
            tracer_provider
            if tracer_provider is not None
            else _console_provider_from_env()
        )


def tracing_enabled() -> bool:
    return _ENABLED


def extract_trace_context(carrier: Mapping[str, object]) -> object | None:
    """Extract a W3C parent context, or return ``None`` when tracing is inert."""

    if not _ENABLED:
        return None
    try:
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError:
        return None
    return TraceContextTextMapPropagator().extract(carrier=carrier)


def inject_trace_context(
    carrier: MutableMapping[str, str],
    *,
    context: object | None = None,
) -> None:
    """Inject only W3C ``traceparent``/``tracestate`` fields into ``carrier``."""

    if not _ENABLED:
        return
    try:
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ImportError:
        return
    TraceContextTextMapPropagator().inject(carrier=carrier, context=context)


def _otel_span_kind(kind: SpanKindName | None, span_kind_type: object) -> object:
    if kind is None:
        kind = "internal"
    if not isinstance(kind, str):
        raise TypeError("span kind must be a string")
    try:
        return getattr(span_kind_type, kind.upper())
    except AttributeError:
        allowed = "internal, server, client, producer, consumer"
        raise ValueError(f"unsupported span kind {kind!r}; expected one of: {allowed}") from None


def mark_span_error(
    span: object | None,
    *,
    error_type: str | None = None,
    cancelled: bool = False,
) -> None:
    """Mark an existing span failed without recording application text."""

    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        return
    try:
        if error_type is not None:
            span.set_attribute("error.type", error_type)
        if cancelled:
            span.set_attribute("kairyu.cancelled", True)
        # Deliberately omit a description: SDK descriptions often include the
        # exception message, which may contain prompt or output text.
        span.set_status(Status(StatusCode.ERROR))
    except Exception:
        # Optional observability must never mask the application exception.
        pass


def _record_failure(span: object, error: BaseException) -> None:
    """Record a privacy-safe failure without replacing the original exception."""

    mark_span_error(
        span,
        error_type=type(error).__qualname__,
        cancelled=isinstance(error, asyncio.CancelledError),
    )


@contextmanager
def traced_span(
    name: str,
    attributes: Mapping[str, object] | None = None,
    *,
    context: object | None = None,
    kind: SpanKindName | None = None,
):
    """Create a privacy-safe span, or yield ``None`` when tracing is unavailable."""

    if not _ENABLED:
        yield None
        return
    try:
        from opentelemetry import trace  # deferred: otel extra / dev group
        from opentelemetry.trace import SpanKind
    except ImportError:
        yield None
        return
    provider = _TRACER_PROVIDER
    tracer = provider.get_tracer("kairyu") if provider is not None else trace.get_tracer("kairyu")
    otel_kind = _otel_span_kind(kind, SpanKind)
    with tracer.start_as_current_span(
        name,
        context=context,
        kind=otel_kind,
        attributes=attributes,
        # The SDK defaults would attach exception.message and stacktrace and
        # put the message in status.description.  Both are forbidden here.
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as error:
            _record_failure(span, error)
            raise
