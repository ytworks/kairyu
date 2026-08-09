"""Allocation-light encoders for the dominant token SSE wire shapes."""

from __future__ import annotations

from pydantic_core import to_json

_JSON_LINE_SEPARATOR_BYTES = (
    (b"\xc2\x85", b"\\u0085"),
    (b"\xe2\x80\xa8", b"\\u2028"),
    (b"\xe2\x80\xa9", b"\\u2029"),
)


def _json_string_bytes(value: str) -> bytes:
    """Serialize one JSON string with the same UTF-8 policy as Pydantic.

    The three replacements keep JSON on one physical SSE line. ``bytes.replace``
    returns the original object when a separator is absent, so ordinary token
    deltas stay allocation-light without decoding back to ``str``.
    """

    if not isinstance(value, str):
        raise TypeError("SSE fast-path values must be strings")
    serialized = to_json(value)
    for separator, escape in _JSON_LINE_SEPARATOR_BYTES:
        serialized = serialized.replace(separator, escape)
    return serialized


def _require_integer(value: int) -> None:
    if type(value) is not int:
        raise TypeError("SSE fast-path integer fields must be integers")


def _integer_bytes(value: int) -> bytes:
    _require_integer(value)
    return str(value).encode("ascii")


class ChatContentSSEEncoder:
    """Encode role-less single-choice chat deltas without Pydantic models."""

    __slots__ = ("_choice_prefix", "_prefixes", "_suffix")

    def __init__(
        self,
        response_id: str,
        created: int,
        model: str,
        *,
        include_usage: bool,
    ) -> None:
        self._choice_prefix = b"".join(
            (
                b'data: {"id":',
                _json_string_bytes(response_id),
                b',"object":"chat.completion.chunk","created":',
                _integer_bytes(created),
                b',"model":',
                _json_string_bytes(model),
                b',"choices":[{"index":',
            )
        )
        self._suffix = b"".join(
            (
                b',"reasoning_content":null,"tool_calls":null},'
                b'"finish_reason":null,"logprobs":null}]',
                b',"usage":null}' if include_usage else b"}",
                b"\n\n",
            )
        )
        self._prefixes = {0: self._prefix(0)}

    def _prefix(self, index: int) -> bytes:
        return b"".join(
            (
                self._choice_prefix,
                _integer_bytes(index),
                b',"delta":{"role":null,"content":',
            )
        )

    def encode(self, index: int, content: str) -> bytes:
        _require_integer(index)
        prefix = self._prefixes.get(index)
        if prefix is None:
            prefix = self._prefix(index)
            self._prefixes[index] = prefix
        return b"".join((prefix, _json_string_bytes(content), self._suffix))


class CompletionTextSSEEncoder:
    """Encode unfinished single-choice legacy completion text deltas."""

    __slots__ = ("_choice_prefix", "_prefixes", "_suffix")

    def __init__(
        self,
        response_id: str,
        created: int,
        model: str,
        *,
        include_usage: bool,
    ) -> None:
        self._choice_prefix = b"".join(
            (
                b'data: {"id":',
                _json_string_bytes(response_id),
                b',"object":"text_completion","created":',
                _integer_bytes(created),
                b',"model":',
                _json_string_bytes(model),
                b',"choices":[{"index":',
            )
        )
        self._suffix = b"".join(
            (
                b',"logprobs":null,"finish_reason":null}]',
                b',"usage":null}' if include_usage else b"}",
                b"\n\n",
            )
        )
        self._prefixes = {0: self._prefix(0)}

    def _prefix(self, index: int) -> bytes:
        return b"".join(
            (
                self._choice_prefix,
                _integer_bytes(index),
                b',"text":',
            )
        )

    def encode(self, index: int, text: str) -> bytes:
        _require_integer(index)
        prefix = self._prefixes.get(index)
        if prefix is None:
            prefix = self._prefix(index)
            self._prefixes[index] = prefix
        return b"".join((prefix, _json_string_bytes(text), self._suffix))


class ResponsesTextDeltaSSEEncoder:
    """Encode the dominant ``response.output_text.delta`` event shape."""

    __slots__ = ("_after_sequence", "_prefix", "_suffix")

    def __init__(self, message_id: str) -> None:
        self._prefix = (
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","sequence_number":'
        )
        self._after_sequence = b"".join(
            (
                b',"item_id":',
                _json_string_bytes(message_id),
                b',"output_index":0,"content_index":0,"delta":',
            )
        )
        self._suffix = b',"logprobs":[]}\n\n'

    def encode(self, sequence_number: int, delta: str) -> bytes:
        _require_integer(sequence_number)
        return b"".join(
            (
                self._prefix,
                _integer_bytes(sequence_number),
                self._after_sequence,
                _json_string_bytes(delta),
                self._suffix,
            )
        )
