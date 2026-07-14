import pytest
from pydantic import ValidationError

from kairyu.batch.envelope import BatchLineEnvelope
from kairyu.entrypoints.server.errors import sanitize_backend_error


def _line(**overrides):
    line = {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "m",
            "messages": [{"role": "user", "content": "hello"}],
        },
    }
    line.update(overrides)
    return line


def _validate(line):
    return BatchLineEnvelope.model_validate(
        line, context={"endpoint": "/v1/chat/completions"}
    )


def test_batch_line_envelope_accepts_valid_input_and_is_frozen():
    envelope = _validate(_line())

    assert envelope.custom_id == "request-1"
    assert envelope.method == "POST"
    assert envelope.url == "/v1/chat/completions"
    assert envelope.body["model"] == "m"
    with pytest.raises(ValidationError):
        envelope.custom_id = "changed"


def test_batch_line_envelope_requires_job_endpoint_context():
    with pytest.raises(ValidationError, match="batch endpoint context"):
        BatchLineEnvelope.model_validate(_line())


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (5, "valid dictionary"),
        ({"method": "POST", "url": "/v1/chat/completions", "body": {}}, "custom_id"),
        (_line(custom_id=""), "custom_id"),
        (_line(custom_id="   "), "custom_id"),
        (_line(method="DELETE"), "POST"),
        (_line(url="/admin/usage"), "batch endpoint"),
        (_line(body=[]), "body"),
    ],
)
def test_batch_line_envelope_rejects_invalid_input(line, message):
    with pytest.raises(ValidationError, match=message):
        _validate(line)


def test_sanitize_backend_error_exposes_only_exception_class():
    payload = sanitize_backend_error(
        RuntimeError("http://replica-internal:9000 secret=abc")
    )

    assert payload == {
        "message": "upstream backend error (RuntimeError)",
        "type": "upstream_error",
        "code": "backend_error",
    }
    serialized = str(payload)
    assert "replica-internal" not in serialized
    assert "secret=abc" not in serialized
