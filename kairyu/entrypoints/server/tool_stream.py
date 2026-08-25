"""Incremental tool-call scanners for the Anthropic Messages adapter (#573).

Turns the append-only model text stream into typed events so ``/v1/messages``
can open ``content_block_start(tool_use)`` and emit ``input_json_delta``
fragments while the model is still generating. Canonical generic JSON
envelopes commit once the declared name and arguments-object opener are
complete; the object bytes then stream as generated. Other shapes retain
commit-on-close validation. Once a tool block is on the wire it cannot be
retracted, so a generic envelope that becomes invalid after early commit ends
the stream with an error.

The same scanners run in one-shot mode (``feed(text, final=True)`` +
``fold_tool_stream``) to build the unary ``/v1/messages`` response, so the
stream and unary bodies are reconstructions of one shared parse.

Documented divergences from the OpenAI-wire parse (``_parse_tool_calls``):

- ``/v1/messages`` keeps text and tool_use blocks coexisting (the Anthropic
  contract) where the OpenAI wire drops text when calls exist.
- QWEN/DSML retroactive invalidation (trailing prose voids every call in the
  unary rules) becomes an SSE ``error`` event on a stream that already
  committed a call; the unary fold keeps today's silent downgrade to text.
- Mixed generic/native envelopes are decided per envelope rather than
  all-or-nothing; only pathological model output can observe the difference.
- A canonical generic envelope that turns invalid AFTER its early commit
  (progressive streaming) invalidates the whole parse: the stream errors and
  the unary fold downgrades everything to text, where pre-progressive
  behavior flushed only that envelope. Pathological output only.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

from kairyu.entrypoints.chat_template import ToolCallProtocol
from kairyu.entrypoints.server.chat_service import (
    _DSML_INVOKE_PATTERN,
    _DSML_PARAMETER_PATTERN,
    _LLAMA_TOOL_PREFIX,
    _LLAMA_TOOL_SUFFIXES,
    _QWEN_FUNCTION_PATTERN,
    NormalizedToolChoice,
    _qwen_arguments,
    _strict_json_loads,
    _tool_parameters_schema,
)

_GENERIC_OPEN = "<tool_call>"
_GENERIC_CLOSE = "</tool_call>"
_DSML_BLOCK_OPEN = "<｜DSML｜tool_calls>"
_DSML_BLOCK_CLOSE = "</｜DSML｜tool_calls>"
_DSML_INVOKE_CLOSE = "</｜DSML｜invoke>"

_GENERIC_PROGRESSIVE_PREFIX = re.compile(
    r'^\s*\{\s*"name"\s*:\s*(?P<name>"(?:[^"\\]|\\.)*")\s*,\s*'
    r'"(?:arguments|parameters)"\s*:\s*(?P<arguments>\{)'
)


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolStart:
    id: str
    name: str


@dataclass(frozen=True)
class ToolArgsDelta:
    partial_json: str


@dataclass(frozen=True)
class ToolStop:
    pass


@dataclass(frozen=True)
class StreamInvalid:
    """The protocol became invalid after a call was put on the wire."""

    message: str


ToolStreamEvent = TextDelta | ToolStart | ToolArgsDelta | ToolStop | StreamInvalid


def _holdback_length(buffer: str, watches: tuple[str, ...]) -> int:
    """Longest buffer suffix that could still begin one of ``watches``.

    The ``ReasoningDeltaParser`` hold-back idiom: retaining up to
    ``len(watch) - 1`` trailing characters guarantees a marker split across
    chunk boundaries is never emitted as text.
    """

    limit = min(max(len(watch) for watch in watches) - 1, len(buffer))
    for keep in range(limit, 0, -1):
        tail = buffer[-keep:]
        if any(watch.startswith(tail) for watch in watches):
            return keep
    return 0


class ToolStreamScanner:
    """GENERIC protocol scanner; also the embedded text mode of every other
    protocol (the unary parse runs the generic scan first for all of them)."""

    protocol = ToolCallProtocol.GENERIC

    def __init__(
        self,
        tools: tuple[dict, ...] | list[dict] | None,
        tool_choice: NormalizedToolChoice,
    ) -> None:
        self._tools = tuple(tools or ())
        self._choice = tool_choice
        self._buffer = ""
        self._in_call = False
        self._raw_parts: list[str] = []
        self._progressive_name: str | None = None
        self._progressive_scan_at = 0
        self._progressive_depth = 0
        self._progressive_in_string = False
        self._progressive_escape = False
        self._progressive_arguments_complete = False
        self.committed_calls = 0
        self.invalidated = False

    @property
    def raw_text(self) -> str:
        return "".join(self._raw_parts)

    # -- protocol hooks ----------------------------------------------------

    def _text(self, text: str) -> list[ToolStreamEvent]:
        return [TextDelta(text)] if text else []

    def _close_envelope(self, body: str) -> list[ToolStreamEvent]:
        try:
            payload = _strict_json_loads(body)
        except (TypeError, ValueError, RecursionError):
            payload = None
        call = self._validated_call(payload)
        if call is None or not self._allowed(call[0]):
            # Mirrors the unary rules: an unparsable envelope is skipped (its
            # text survives) and a filtered name never becomes a call.
            return self._text(_GENERIC_OPEN + body + _GENERIC_CLOSE)
        return self._commit(call[0], call[1])

    def _dangling(self, buffered: str) -> list[ToolStreamEvent]:
        """The stream ended inside an open envelope."""

        return self._text(_GENERIC_OPEN + buffered)

    # -- shared helpers ----------------------------------------------------

    def _validated_call(self, payload: object) -> tuple[str, dict] | None:
        if not isinstance(payload, dict):
            return None
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        arguments = payload.get("arguments", payload.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = _strict_json_loads(arguments)
            except (TypeError, ValueError, RecursionError):
                return None
        if not isinstance(arguments, dict):
            return None
        try:
            json.dumps(arguments, allow_nan=False)
        except (TypeError, ValueError):
            return None
        return name, arguments

    def _allowed(self, name: str) -> bool:
        return name in self._choice.allowed_names and (
            self._choice.named is None or name == self._choice.named
        )

    def _commit(self, name: str, arguments: dict) -> list[ToolStreamEvent]:
        self.committed_calls += 1
        return [
            ToolStart(id=f"call_{uuid.uuid4().hex[:12]}", name=name),
            ToolArgsDelta(json.dumps(arguments, ensure_ascii=False)),
            ToolStop(),
        ]

    def _invalidate(self, message: str) -> list[ToolStreamEvent]:
        self.invalidated = True
        return [StreamInvalid(message)]

    def _progressive_events(self, body: str) -> list[ToolStreamEvent]:
        """Start and advance a canonical generic call when it is safe."""

        events: list[ToolStreamEvent] = []
        if self._progressive_name is None:
            match = _GENERIC_PROGRESSIVE_PREFIX.match(body)
            if match is None:
                return events
            try:
                name = _strict_json_loads(match.group("name"))
            except (TypeError, ValueError, RecursionError):
                return events
            if not isinstance(name, str) or not name.strip() or not self._allowed(name):
                return events
            self._progressive_name = name
            self._progressive_scan_at = match.start("arguments")
            self._progressive_depth = 0
            self._progressive_in_string = False
            self._progressive_escape = False
            self._progressive_arguments_complete = False
            self.committed_calls += 1
            events.append(
                ToolStart(id=f"call_{uuid.uuid4().hex[:12]}", name=name)
            )

        if self._progressive_arguments_complete:
            return events
        start = self._progressive_scan_at
        cursor = start
        while cursor < len(body):
            char = body[cursor]
            cursor += 1
            if self._progressive_in_string:
                if self._progressive_escape:
                    self._progressive_escape = False
                elif char == "\\":
                    self._progressive_escape = True
                elif char == '"':
                    self._progressive_in_string = False
                continue
            if char == '"':
                self._progressive_in_string = True
            elif char == "{":
                self._progressive_depth += 1
            elif char == "}":
                self._progressive_depth -= 1
                if self._progressive_depth == 0:
                    self._progressive_arguments_complete = True
                    break
        self._progressive_scan_at = cursor
        if cursor > start:
            events.append(ToolArgsDelta(body[start:cursor]))
        return events

    def _finish_progressive(self, body: str) -> list[ToolStreamEvent]:
        events = self._progressive_events(body)
        try:
            payload = _strict_json_loads(body)
        except (TypeError, ValueError, RecursionError):
            payload = None
        call = self._validated_call(payload)
        if (
            not self._progressive_arguments_complete
            or call is None
            or call[0] != self._progressive_name
            or not self._allowed(call[0])
        ):
            events.extend(
                self._invalidate(
                    "generic tool call became invalid after streaming arguments"
                )
            )
            return events
        events.append(ToolStop())
        self._progressive_name = None
        self._progressive_scan_at = 0
        self._progressive_depth = 0
        self._progressive_in_string = False
        self._progressive_escape = False
        self._progressive_arguments_complete = False
        return events

    # -- feed --------------------------------------------------------------

    def feed(self, delta: str, *, final: bool = False) -> list[ToolStreamEvent]:
        if self.invalidated:
            return []
        if delta:
            self._raw_parts.append(delta)
        self._buffer += delta
        events: list[ToolStreamEvent] = []
        while not self.invalidated:
            if self._in_call:
                close_at = self._buffer.find(_GENERIC_CLOSE)
                if close_at < 0:
                    if not final or self._progressive_name is not None:
                        events.extend(self._progressive_events(self._buffer))
                    if final:
                        if self._progressive_name is not None:
                            events.extend(
                                self._invalidate(
                                    "generic tool call ended after streaming "
                                    "unterminated arguments"
                                )
                            )
                        else:
                            events.extend(self._dangling(self._buffer))
                        self._buffer = ""
                        self._in_call = False
                    break
                body = self._buffer[:close_at]
                self._buffer = self._buffer[close_at + len(_GENERIC_CLOSE) :]
                self._in_call = False
                if self._progressive_name is None:
                    events.extend(self._close_envelope(body))
                else:
                    events.extend(self._finish_progressive(body))
                continue
            open_at = self._buffer.find(_GENERIC_OPEN)
            if open_at >= 0:
                text = self._buffer[:open_at]
                self._buffer = self._buffer[open_at + len(_GENERIC_OPEN) :]
                self._in_call = True
                events.extend(self._text(text))
                continue
            if final:
                text, self._buffer = self._buffer, ""
                events.extend(self._text(text))
            else:
                keep = _holdback_length(self._buffer, (_GENERIC_OPEN,))
                cut = len(self._buffer) - keep
                if cut > 0:
                    text = self._buffer[:cut]
                    self._buffer = self._buffer[cut:]
                    events.extend(self._text(text))
            break
        return events


class QwenToolStreamScanner(ToolStreamScanner):
    """QWEN: ``<tool_call>`` envelopes whose body is one ``<function=...>``
    element; any non-whitespace text outside envelopes voids all qwen calls."""

    protocol = ToolCallProtocol.QWEN

    def __init__(self, tools, tool_choice) -> None:
        super().__init__(tools, tool_choice)
        self._pure_so_far = True
        self._qwen_committed = False

    def _text(self, text: str) -> list[ToolStreamEvent]:
        if text.strip():
            if self._qwen_committed:
                return self._invalidate(
                    "text outside Qwen tool calls voids the committed calls"
                )
            self._pure_so_far = False
        return super()._text(text)

    def _close_envelope(self, body: str) -> list[ToolStreamEvent]:
        try:
            payload = _strict_json_loads(body)
        except (TypeError, ValueError, RecursionError):
            payload = None
        call = self._validated_call(payload)
        if call is not None:
            if not self._allowed(call[0]):
                return self._text(_GENERIC_OPEN + body + _GENERIC_CLOSE)
            # A generic-JSON envelope keeps envelope purity (the unary parse
            # accepts it before the Qwen rules even run).
            return self._commit(call[0], call[1])
        if self._pure_so_far:
            function_match = _QWEN_FUNCTION_PATTERN.fullmatch(body)
            if function_match is not None:
                name = function_match.group(1).strip()
                schema = (
                    _tool_parameters_schema(self._tools, name) if name else None
                )
                if schema is not None:
                    try:
                        arguments = _qwen_arguments(
                            function_match.group(2), schema
                        )
                    except (TypeError, ValueError, RecursionError):
                        arguments = None
                    if arguments is not None:
                        if not self._allowed(name):
                            return self._text(
                                _GENERIC_OPEN + body + _GENERIC_CLOSE
                            )
                        self._qwen_committed = True
                        return self._commit(name, arguments)
        return self._text(_GENERIC_OPEN + body + _GENERIC_CLOSE)

    def _dangling(self, buffered: str) -> list[ToolStreamEvent]:
        if self._qwen_committed:
            return self._invalidate(
                "unterminated Qwen tool call voids the committed calls"
            )
        return super()._dangling(buffered)


class DsmlToolStreamScanner(ToolStreamScanner):
    """DEEPSEEK_V4: one ``<｜DSML｜tool_calls>`` block must be the entire
    stripped output; anything else falls back to the generic text mode."""

    protocol = ToolCallProtocol.DEEPSEEK_V4

    _LEADING = "leading"
    _IN_BLOCK = "in_block"
    _IN_INVOKE = "in_invoke"
    _AFTER_BLOCK = "after_block"
    _GENERIC = "generic"

    def __init__(self, tools, tool_choice) -> None:
        super().__init__(tools, tool_choice)
        self._state = self._LEADING
        self._invoke_header = ""

    def feed(self, delta: str, *, final: bool = False) -> list[ToolStreamEvent]:
        if self.invalidated:
            return []
        if self._state == self._GENERIC:
            return super().feed(delta, final=final)
        if delta:
            self._raw_parts.append(delta)
        self._buffer += delta
        events: list[ToolStreamEvent] = []
        while not self.invalidated:
            if self._state == self._LEADING:
                stripped = self._buffer.lstrip()
                if not stripped:
                    break
                if _DSML_BLOCK_OPEN.startswith(stripped) and not final:
                    break  # could still become the block header
                if stripped.startswith(_DSML_BLOCK_OPEN):
                    self._buffer = stripped[len(_DSML_BLOCK_OPEN) :]
                    self._state = self._IN_BLOCK
                    continue
                events.extend(self._fallback_to_generic(final))
                break
            if self._state == self._IN_BLOCK:
                stripped = self._buffer.lstrip()
                if not stripped:
                    if final:
                        events.extend(self._structure_failure(final))
                    break
                if (
                    _DSML_BLOCK_CLOSE.startswith(stripped)
                    or self._invoke_open_prefix(stripped)
                ) and not final:
                    break
                if stripped.startswith(_DSML_BLOCK_CLOSE):
                    self._buffer = stripped[len(_DSML_BLOCK_CLOSE) :]
                    self._state = self._AFTER_BLOCK
                    continue
                invoke_open = self._match_invoke_open(stripped)
                if invoke_open is not None:
                    self._invoke_header, self._buffer = invoke_open
                    self._state = self._IN_INVOKE
                    continue
                if not final and self._invoke_open_could_grow(stripped):
                    break
                events.extend(self._structure_failure(final))
                break
            if self._state == self._IN_INVOKE:
                close_at = self._buffer.find(_DSML_INVOKE_CLOSE)
                if close_at < 0:
                    if final:
                        events.extend(self._structure_failure(final))
                    break
                invoke_text = (
                    self._invoke_header
                    + self._buffer[: close_at + len(_DSML_INVOKE_CLOSE)]
                )
                self._buffer = self._buffer[close_at + len(_DSML_INVOKE_CLOSE) :]
                committed = self._commit_invoke(invoke_text)
                if committed is None:
                    events.extend(self._structure_failure(final))
                    break
                events.extend(committed)
                self._state = self._IN_BLOCK
                continue
            if self._state == self._AFTER_BLOCK:
                if self._buffer.strip():
                    events.extend(self._structure_failure(final))
                break
            break
        return events

    @staticmethod
    def _invoke_open_prefix(stripped: str) -> bool:
        return '<｜DSML｜invoke name="'.startswith(stripped)

    @staticmethod
    def _invoke_open_could_grow(stripped: str) -> bool:
        # An invoke opener is `<｜DSML｜invoke name="NAME">`; while the buffer
        # is a prefix of that shape (name still streaming) keep waiting.
        prefix = '<｜DSML｜invoke name="'
        if stripped.startswith(prefix):
            remainder = stripped[len(prefix) :]
            return '"' not in remainder or ">" not in remainder.split('"', 1)[1]
        return prefix.startswith(stripped)

    @staticmethod
    def _match_invoke_open(stripped: str) -> tuple[str, str] | None:
        prefix = '<｜DSML｜invoke name="'
        if not stripped.startswith(prefix):
            return None
        rest = stripped[len(prefix) :]
        quote_at = rest.find('"')
        if quote_at < 0 or "\n" in rest[:quote_at]:
            return None
        if not rest[quote_at:].startswith('">'):
            return None
        header = stripped[: len(prefix) + quote_at + 2]
        return header, rest[quote_at + 2 :]

    def _commit_invoke(self, invoke_text: str) -> list[ToolStreamEvent] | None:
        match = _DSML_INVOKE_PATTERN.fullmatch(invoke_text)
        if match is None:
            return None
        name = match.group(1).strip()
        schema = _tool_parameters_schema(self._tools, name) if name else None
        if schema is None:
            return None
        arguments: dict[str, object] = {}
        cursor = 0
        body = match.group(2)
        for parameter in _DSML_PARAMETER_PATTERN.finditer(body):
            if body[cursor : parameter.start()].strip():
                return None
            key, is_string, value = parameter.groups()
            if key in arguments:
                return None
            if is_string == "true":
                arguments[key] = value
            else:
                try:
                    arguments[key] = _strict_json_loads(value)
                except (TypeError, ValueError, RecursionError):
                    return None
            cursor = parameter.end()
        if body[cursor:].strip():
            return None
        call = self._validated_call({"name": name, "arguments": arguments})
        if call is None or not self._allowed(call[0]):
            return None
        return self._commit(call[0], call[1])

    def _structure_failure(self, final: bool) -> list[ToolStreamEvent]:
        if self.committed_calls:
            return self._invalidate(
                "malformed DSML tool-call block voids the committed calls"
            )
        return self._fallback_to_generic(final)

    def _fallback_to_generic(self, final: bool) -> list[ToolStreamEvent]:
        """Replay the (never-emitted) raw prefix through the generic mode."""

        raw = self.raw_text
        self._buffer = ""
        self._state = self._GENERIC
        # Reset generic state and replay without double-recording raw parts.
        self._in_call = False
        replay = super().feed("", final=False)
        self._buffer = raw
        replay += super().feed("", final=final)
        return replay


class LlamaToolStreamScanner(ToolStreamScanner):
    """LLAMA: the whole output (after prefix/suffix strip) is one JSON call.

    A candidate call defers all output until ``final`` (matching today's
    buffered behavior for exactly this shape); anything else replays through
    the generic text mode.
    """

    protocol = ToolCallProtocol.LLAMA

    _START = "start"
    _HOLD = "hold"
    _GENERIC = "generic"

    def __init__(self, tools, tool_choice) -> None:
        super().__init__(tools, tool_choice)
        self._state = self._START

    def feed(self, delta: str, *, final: bool = False) -> list[ToolStreamEvent]:
        if self.invalidated:
            return []
        if self._state == self._GENERIC:
            return super().feed(delta, final=final)
        if delta:
            self._raw_parts.append(delta)
        self._buffer += delta
        stripped = self._buffer.lstrip()
        if self._state == self._START and stripped:
            candidate = (
                stripped.startswith("{")
                or stripped.startswith(_LLAMA_TOOL_PREFIX)
                or (_LLAMA_TOOL_PREFIX.startswith(stripped) and not final)
            )
            if candidate:
                self._state = self._HOLD
            else:
                return self._fallback_to_generic(final)
        if not final:
            return []
        return self._finalize()

    def _finalize(self) -> list[ToolStreamEvent]:
        candidate = self._buffer.strip()
        if candidate.startswith(_LLAMA_TOOL_PREFIX):
            candidate = candidate[len(_LLAMA_TOOL_PREFIX) :].lstrip()
        for suffix in _LLAMA_TOOL_SUFFIXES:
            if candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)].rstrip()
                break
        try:
            payload = _strict_json_loads(candidate)
        except (TypeError, ValueError, RecursionError):
            payload = None
        call = self._validated_call(payload)
        if call is not None and self._allowed(call[0]):
            self._buffer = ""
            return self._commit(call[0], call[1])
        return self._fallback_to_generic(True)

    def _fallback_to_generic(self, final: bool) -> list[ToolStreamEvent]:
        raw = self.raw_text
        self._state = self._GENERIC
        self._in_call = False
        self._buffer = raw
        return super().feed("", final=final)


_SCANNERS = {
    ToolCallProtocol.GENERIC: ToolStreamScanner,
    ToolCallProtocol.QWEN: QwenToolStreamScanner,
    ToolCallProtocol.DEEPSEEK_V4: DsmlToolStreamScanner,
    ToolCallProtocol.LLAMA: LlamaToolStreamScanner,
}


def tool_stream_scanner_for(
    protocol: ToolCallProtocol,
    tools,
    tool_choice: NormalizedToolChoice,
) -> ToolStreamScanner:
    return _SCANNERS[protocol](tools, tool_choice)


@dataclass
class FoldedToolStream:
    blocks: list[dict]
    invalidated: bool
    committed_calls: int


def fold_tool_stream(
    events: list[ToolStreamEvent], *, raw_text: str
) -> FoldedToolStream:
    """Fold scanner events into Anthropic content blocks (the unary body).

    ``StreamInvalid`` reproduces the unary rules' silent downgrade: every
    committed call is voided and the raw transcript becomes one text block.
    Whitespace-only text runs adjacent to tool blocks are dropped.
    """

    blocks: list[dict] = []
    text_parts: list[str] = []
    committed = 0
    pending_tool: dict | None = None
    pending_json: list[str] = []

    def flush_text() -> None:
        text = "".join(text_parts)
        text_parts.clear()
        if text.strip():
            blocks.append({"type": "text", "text": text})

    for event in events:
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, ToolStart):
            flush_text()
            pending_tool = {"type": "tool_use", "id": event.id, "name": event.name}
            pending_json = []
        elif isinstance(event, ToolArgsDelta):
            pending_json.append(event.partial_json)
        elif isinstance(event, ToolStop):
            assert pending_tool is not None
            pending_tool["input"] = json.loads("".join(pending_json) or "{}")
            blocks.append(pending_tool)
            pending_tool = None
            committed += 1
        elif isinstance(event, StreamInvalid):
            return FoldedToolStream(
                blocks=[{"type": "text", "text": raw_text}],
                invalidated=True,
                committed_calls=0,
            )
    flush_text()
    if not blocks:
        blocks.append({"type": "text", "text": "".join(text_parts) or raw_text})
    return FoldedToolStream(
        blocks=blocks, invalidated=False, committed_calls=committed
    )
