"""Strict OpenAI Chat Completions SSE reconstruction and timing evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from evals.types import JSON_SAFE_INTEGER_MAX, GenerationTimingEvidence

# A 32K-token completion can carry several MiB of SSE framing because every
# token is wrapped in its own JSON event.  Keep a hard bound, but size it for
# the maximum output used by retained long-context evals.
_MAX_STREAM_BYTES = 16_777_216
_MAX_TEXT_CHARS = 1_048_576


class StreamingProtocolError(ValueError):
    """The HTTP 200 SSE body cannot be trusted as one chat completion."""


@dataclass(frozen=True)
class StreamingChatResult:
    content: str | None
    refusal: str | None
    message_auxiliary: str | None
    finish_reason: str
    usage: tuple[int, int, int] | None
    timing: GenerationTimingEvidence | None


def _strict_json(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _usage(value: object) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        raise StreamingProtocolError("usage must be an object")
    names = ("prompt_tokens", "completion_tokens", "total_tokens")
    counts = tuple(value.get(name) for name in names)
    if any(
        type(count) is not int or not 0 <= count <= JSON_SAFE_INTEGER_MAX
        for count in counts
    ):
        raise StreamingProtocolError("usage token counts must be safe non-negative integers")
    prompt, completion, total = counts
    if total != prompt + completion:
        raise StreamingProtocolError("usage total does not equal prompt plus completion")
    return prompt, completion, total


class ChatSSEAccumulator:
    """Consume decoded SSE lines while retaining semantic event timestamps."""

    def __init__(self, *, request_attempts: int) -> None:
        if type(request_attempts) is not int or request_attempts < 1:
            raise ValueError("request_attempts must be positive")
        self._request_attempts = request_attempts
        self._bytes = 0
        self._done = False
        self._content: list[str] = []
        self._refusal: list[str] = []
        self._reasoning: list[str] = []
        self._function_call: dict[str, list[str]] = {"name": [], "arguments": []}
        self._tool_calls: dict[int, dict[str, object]] = {}
        self._finish_reason: str | None = None
        self._usage: tuple[int, int, int] | None = None
        self._first_semantic_s: float | None = None
        self._last_semantic_s: float | None = None
        self._semantic_events = 0

    def feed_line(self, line: str, elapsed_s: float) -> None:
        if not isinstance(line, str):
            raise StreamingProtocolError("SSE line is not decoded text")
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise StreamingProtocolError("SSE timestamp is invalid")
        if not line or line.startswith(":"):
            return
        if self._done:
            raise StreamingProtocolError("data appeared after [DONE]")
        if not line.startswith("data:"):
            raise StreamingProtocolError("SSE event contains a non-data field")
        payload = line[5:]
        if payload.startswith(" "):
            payload = payload[1:]
        self._bytes += len(payload.encode("utf-8"))
        if self._bytes > _MAX_STREAM_BYTES:
            raise StreamingProtocolError(
                f"SSE data exceeds {_MAX_STREAM_BYTES} bytes"
            )
        if payload == "[DONE]":
            self._done = True
            return
        try:
            data = _strict_json(payload)
        except (RecursionError, UnicodeError, ValueError) as error:
            raise StreamingProtocolError(f"strict SSE JSON parse failed: {error}") from error
        self._feed_chunk(data, elapsed_s)

    def _feed_chunk(self, data: object, elapsed_s: float) -> None:
        if not isinstance(data, dict):
            raise StreamingProtocolError("SSE chunk top level must be an object")
        if "error" in data:
            raise StreamingProtocolError("SSE chunk contains an error envelope")
        if data.get("usage") is not None:
            parsed = _usage(data["usage"])
            if self._usage is not None and self._usage != parsed:
                raise StreamingProtocolError("usage changed across SSE chunks")
            self._usage = parsed
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamingProtocolError("choices must be an empty or singleton list")
        if not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict) or type(choice.get("index")) is not int:
            raise StreamingProtocolError("stream choice must carry integer index 0")
        if choice["index"] != 0:
            raise StreamingProtocolError("stream choice index must be 0")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise StreamingProtocolError("stream choice delta must be an object")
        semantic = self._feed_delta(delta)
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str) or not finish_reason:
                raise StreamingProtocolError("finish_reason must be null or non-empty text")
            if self._finish_reason is not None and self._finish_reason != finish_reason:
                raise StreamingProtocolError("finish_reason changed across SSE chunks")
            self._finish_reason = finish_reason
        if semantic:
            self._semantic_events += 1
            if self._first_semantic_s is None:
                self._first_semantic_s = elapsed_s
            self._last_semantic_s = elapsed_s

    @staticmethod
    def _fragment(delta: dict, name: str) -> str | None:
        value = delta.get(name)
        if value is not None and not isinstance(value, str):
            raise StreamingProtocolError(f"delta.{name} must be text or null")
        return value

    def _feed_delta(self, delta: dict) -> bool:
        role = delta.get("role")
        if role is not None and not isinstance(role, str):
            raise StreamingProtocolError("delta.role must be text or null")
        semantic = False
        for name, destination in (
            ("content", self._content),
            ("refusal", self._refusal),
            ("reasoning_content", self._reasoning),
            ("reasoning", self._reasoning),
        ):
            fragment = self._fragment(delta, name)
            if fragment is not None:
                destination.append(fragment)
                semantic = semantic or bool(fragment)
        function_call = delta.get("function_call")
        if function_call is not None:
            if not isinstance(function_call, dict):
                raise StreamingProtocolError("delta.function_call must be an object")
            for name in ("name", "arguments"):
                value = function_call.get(name)
                if value is not None and not isinstance(value, str):
                    raise StreamingProtocolError(
                        f"delta.function_call.{name} must be text or null"
                    )
                if value is not None:
                    self._function_call[name].append(value)
                    semantic = semantic or bool(value)
        tool_calls = delta.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise StreamingProtocolError("delta.tool_calls must be a list")
            for value in tool_calls:
                semantic = self._feed_tool_call(value) or semantic
        return semantic

    def _feed_tool_call(self, value: object) -> bool:
        if not isinstance(value, dict) or type(value.get("index")) is not int:
            raise StreamingProtocolError("tool-call delta must carry an integer index")
        index = value["index"]
        if index < 0:
            raise StreamingProtocolError("tool-call delta index must be non-negative")
        retained = self._tool_calls.setdefault(
            index,
            {"id": [], "type": None, "function": {"name": [], "arguments": []}},
        )
        semantic = False
        call_id = value.get("id")
        if call_id is not None:
            if not isinstance(call_id, str):
                raise StreamingProtocolError("tool-call id must be text or null")
            retained["id"].append(call_id)
            semantic = semantic or bool(call_id)
        call_type = value.get("type")
        if call_type is not None:
            if not isinstance(call_type, str):
                raise StreamingProtocolError("tool-call type must be text or null")
            if retained["type"] not in (None, call_type):
                raise StreamingProtocolError("tool-call type changed across chunks")
            retained["type"] = call_type
            semantic = semantic or bool(call_type)
        function = value.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise StreamingProtocolError("tool-call function must be an object")
            stored_function = retained["function"]
            assert isinstance(stored_function, dict)
            for name in ("name", "arguments"):
                fragment = function.get(name)
                if fragment is not None and not isinstance(fragment, str):
                    raise StreamingProtocolError(
                        f"tool-call function {name} must be text or null"
                    )
                if fragment is not None:
                    stored_function[name].append(fragment)
                    semantic = semantic or bool(fragment)
        return semantic

    def finish(self) -> StreamingChatResult:
        if not self._done:
            raise StreamingProtocolError("stream ended without [DONE]")
        if self._finish_reason is None:
            raise StreamingProtocolError("stream ended without finish_reason")
        content = "".join(self._content) if self._content else None
        refusal = "".join(self._refusal) if self._refusal else None
        if refusal is not None and (not refusal or content is not None or self._tool_calls):
            raise StreamingProtocolError("refusal cannot coexist with content or tool calls")
        auxiliary: dict[str, object] = {}
        if self._tool_calls:
            indexes = sorted(self._tool_calls)
            if indexes != list(range(len(indexes))):
                raise StreamingProtocolError("tool-call indexes must be contiguous from zero")
            calls = []
            for index in indexes:
                stored = self._tool_calls[index]
                function = stored["function"]
                assert isinstance(function, dict)
                call = {
                    "id": "".join(stored["id"]),
                    "type": stored["type"],
                    "function": {
                        "name": "".join(function["name"]),
                        "arguments": "".join(function["arguments"]),
                    },
                }
                calls.append(call)
            auxiliary["tool_calls"] = calls
        if any(self._function_call.values()):
            auxiliary["function_call"] = {
                name: "".join(parts) for name, parts in self._function_call.items()
            }
        message_auxiliary = (
            json.dumps(
                auxiliary,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            if auxiliary
            else None
        )
        for label, text in (
            ("content", content),
            ("refusal", refusal),
            ("call payload", message_auxiliary),
        ):
            if text is not None and len(text) > _MAX_TEXT_CHARS:
                raise StreamingProtocolError(f"choice {label} exceeds retained evidence bound")

        timing = None
        if self._first_semantic_s is not None and self._last_semantic_s is not None:
            span = self._last_semantic_s - self._first_semantic_s
            completion_tokens = self._usage[1] if self._usage is not None else None
            tps = (
                (completion_tokens - 1) / span
                if completion_tokens is not None and completion_tokens >= 2 and span > 0
                else None
            )
            timing = GenerationTimingEvidence(
                ttft_s=self._first_semantic_s,
                generation_span_s=span,
                completion_tokens=completion_tokens,
                tps=tps,
                semantic_events=self._semantic_events,
                request_attempts=self._request_attempts,
                usage_missing=self._usage is None,
            )
        return StreamingChatResult(
            content=content,
            refusal=refusal,
            message_auxiliary=message_auxiliary,
            finish_reason=self._finish_reason,
            usage=self._usage,
            timing=timing,
        )
