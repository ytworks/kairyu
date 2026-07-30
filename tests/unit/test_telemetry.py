from __future__ import annotations

import asyncio
import builtins
import json

import pytest

import kairyu.telemetry as telemetry
from kairyu.telemetry import (
    CONSOLE_SPAN_PREFIX,
    configure_tracing,
    extract_trace_context,
    inject_trace_context,
    traced_span,
)


@pytest.fixture(autouse=True)
def _reset_tracing():
    configure_tracing(False)
    yield
    configure_tracing(False)


def _recording_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_disabled_helpers_do_not_import_otel(monkeypatch):
    real_import = builtins.__import__

    def reject_otel(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise AssertionError("disabled tracing imported OpenTelemetry")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_otel)
    carrier = {"existing": "value"}

    assert extract_trace_context(carrier) is None
    inject_trace_context(carrier)
    with traced_span("disabled") as span:
        assert span is None
    assert carrier == {"existing": "value"}


def test_missing_dependency_is_a_noop(monkeypatch):
    real_import = builtins.__import__

    def reject_otel(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("optional dependency is absent")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_otel)
    configure_tracing(True, tracer_provider=object())
    carrier = {"existing": "value"}

    assert extract_trace_context(carrier) is None
    inject_trace_context(carrier)
    with traced_span("missing") as span:
        assert span is None
    assert carrier == {"existing": "value"}


def test_injected_provider_does_not_replace_process_global_provider():
    from opentelemetry import trace

    provider, exporter = _recording_provider()
    global_provider = trace.get_tracer_provider()
    configure_tracing(True, tracer_provider=provider)

    with traced_span("private", {"safe": "attribute"}):
        pass

    assert trace.get_tracer_provider() is global_provider
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["private"]
    assert spans[0].attributes == {"safe": "attribute"}


def test_console_exporter_is_explicit_compact_and_service_scoped(
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "fixture-gateway")
    monkeypatch.setattr(telemetry, "_CONSOLE_PROVIDER", None)
    configure_tracing(True)

    with traced_span("fixture", {"safe": "attribute"}, kind="server"):
        pass

    line = capsys.readouterr().out.strip()
    assert line.startswith(CONSOLE_SPAN_PREFIX)
    payload = json.loads(line.removeprefix(CONSOLE_SPAN_PREFIX))
    assert payload["name"] == "fixture"
    assert payload["service_name"] == "fixture-gateway"
    assert payload["kind"] == "SERVER"
    assert payload["status"] == "UNSET"
    assert payload["attributes"] == {"safe": "attribute"}
    assert len(payload["trace_id"]) == 32
    assert len(payload["span_id"]) == 16
    assert payload["parent_span_id"] is None
    telemetry._CONSOLE_PROVIDER.shutdown()


def test_w3c_context_round_trip_preserves_parent_and_span_kind():
    from opentelemetry.trace import SpanKind

    provider, exporter = _recording_provider()
    tracer = provider.get_tracer("test")
    configure_tracing(True, tracer_provider=provider)
    carrier: dict[str, str] = {}

    with tracer.start_as_current_span("upstream") as upstream:
        parent_span_id = upstream.get_span_context().span_id
        inject_trace_context(carrier)

    assert set(carrier) <= {"traceparent", "tracestate"}
    assert "traceparent" in carrier
    parent_context = extract_trace_context(carrier)
    with traced_span("gateway", context=parent_context, kind="server"):
        pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    gateway = spans["gateway"]
    assert gateway.kind is SpanKind.SERVER
    assert gateway.parent is not None
    assert gateway.parent.span_id == parent_span_id
    assert gateway.context.trace_id == spans["upstream"].context.trace_id


def test_w3c_propagates_tracestate_but_never_baggage():
    from opentelemetry import baggage, trace
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        TraceState,
    )

    configure_tracing(True)
    parent = SpanContext(
        trace_id=1,
        span_id=2,
        is_remote=True,
        trace_flags=TraceFlags.SAMPLED,
        trace_state=TraceState([("vendor", "opaque")]),
    )
    context = trace.set_span_in_context(NonRecordingSpan(parent))
    context = baggage.set_baggage("private", "must-not-cross", context=context)
    carrier: dict[str, str] = {}

    inject_trace_context(carrier, context=context)

    assert carrier["tracestate"] == "vendor=opaque"
    assert set(carrier) == {"traceparent", "tracestate"}
    carrier["baggage"] = "private=must-not-cross"
    extracted = extract_trace_context(carrier)
    assert baggage.get_all(context=extracted) == {}
    extracted_span = trace.get_current_span(extracted).get_span_context()
    assert extracted_span.trace_state == parent.trace_state


def test_exception_records_only_type_and_description_free_status():
    from opentelemetry.trace import StatusCode

    provider, exporter = _recording_provider()
    configure_tracing(True, tracer_provider=provider)
    secret = "private prompt and output"

    with pytest.raises(ValueError, match=secret):
        with traced_span("failed"):
            raise ValueError(secret)

    span = exporter.get_finished_spans()[0]
    assert span.attributes == {"error.type": "ValueError"}
    assert span.events == ()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert secret not in repr(span)


def test_cancellation_is_marked_without_recording_its_message():
    from opentelemetry.trace import StatusCode

    provider, exporter = _recording_provider()
    configure_tracing(True, tracer_provider=provider)
    secret = "private cancellation detail"

    with pytest.raises(asyncio.CancelledError):
        with traced_span("cancelled", kind="client"):
            raise asyncio.CancelledError(secret)

    span = exporter.get_finished_spans()[0]
    assert span.attributes == {
        "error.type": "CancelledError",
        "kairyu.cancelled": True,
    }
    assert span.events == ()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert secret not in repr(span)


def test_unknown_span_kind_fails_before_starting_a_span():
    provider, exporter = _recording_provider()
    configure_tracing(True, tracer_provider=provider)

    with pytest.raises(ValueError, match="unsupported span kind"):
        with traced_span("invalid", kind="remote"):  # type: ignore[arg-type]
            pass

    assert exporter.get_finished_spans() == ()
