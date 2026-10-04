"""Upstream 4xx classification at the OpenAI-compatible boundary (M20 WP-04).

vLLM's nested "maximum context length" body is driven end to end by
``tests/server/test_context_overflow.py``; these cases cover the other branches.
"""

from __future__ import annotations

import json

import pytest

from kairyu.engine.backend import UpstreamClientError
from kairyu.engine.openai_errors import raise_for_status

_SECRET = "SECRET-UPSTREAM-DETAIL"


@pytest.mark.parametrize(
    ("body", "overflow"),
    [
        (
            # OpenAI Responses: only the code identifies the overflow.
            {
                "error": {
                    "message": (
                        "Your input exceeds the context window of this model. "
                        f"Please adjust your input and try again. {_SECRET}"
                    ),
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": "context_length_exceeded",
                }
            },
            True,
        ),
        (
            # Older vLLM returns a flat error object from the V1 processor.
            {
                "object": "error",
                "message": (
                    "The decoder prompt (length 5000) is longer than the "
                    f"maximum model length of 4096. {_SECRET}"
                ),
                "type": "BadRequestError",
                "param": None,
                "code": 400,
            },
            True,
        ),
        (
            # A Kairyu replica from before this classification existed.
            {
                "error": {
                    "message": (
                        "prompt tokens (9) plus max_tokens (2) exceed "
                        f"max_model_len (5) {_SECRET}"
                    ),
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
            True,
        ),
        (
            # A per-field length limit is not a context-window overflow.
            {
                "error": {
                    "message": (
                        "Invalid 'messages[0].content': string too long. "
                        "Expected a string with maximum length 10485760."
                    ),
                    "type": "invalid_request_error",
                    "param": "messages[0].content",
                    "code": "string_above_max_length",
                }
            },
            False,
        ),
        (
            {
                "object": "error",
                "message": "temperature must be non-negative, got -1.0.",
                "type": "BadRequestError",
                "param": None,
                "code": 400,
            },
            False,
        ),
    ],
    ids=[
        "openai-code",
        "vllm-flat",
        "kairyu-legacy-text",
        "string-above-max-length",
        "unrelated-400",
    ],
)
def test_upstream_overflow_classifier(body, overflow):
    with pytest.raises(UpstreamClientError) as raised:
        raise_for_status("http://vllm.internal:8000/v1", 400, json.dumps(body))

    error = raised.value
    assert error.status_code == 400
    if overflow:
        assert error.code == "context_length_exceeded"
        # The tenant-visible text is fixed: upstream text and URLs never leak.
        assert error.public_message is not None
        assert _SECRET not in error.public_message
        assert "vllm.internal" not in error.public_message
    else:
        assert error.code == "invalid_request"
        assert error.public_message is None
