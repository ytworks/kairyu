"""Responses SSE events: one emitter per response and the output assembler.

``ResponseEmitter`` is the only place Responses SSE events are built, so every
stream is canonical by construction: ``response.created`` first, gapless
sequence numbers, ``output_index`` 0..n-1 in item order, unique item ids,
``content_part.added`` before any delta, and done events in text -> part ->
item order. Its snapshot is exactly what a client accumulated from the events
written so far (openai-node replaces its snapshot on every lifecycle event),
which is what the ``response.in_progress`` data heartbeats carry. In unary mode
the same calls build the final envelope and the frames are discarded.

``OutputAssembler`` maps parsed model output onto items: reasoning deltas, then
content through the protocol's tool-stream scanner (prose and calls), with the
Messages adapter's gates (``parallel_tool_calls=false``, required/named
``tool_choice``, invalid call markup).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from kairyu.engine.prompt import prompt_text
from kairyu.entrypoints.server.chat_service import NormalizedToolChoice, ReasoningDeltaParser
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesRequest,
    _BufferedFailure,
    _response_envelope,
    _usage_payload_from_wire,
)
from kairyu.entrypoints.server.sse_encode import ResponsesTextDeltaSSEEncoder
from kairyu.entrypoints.server.tool_stream import (
    StreamInvalid,
    TextDelta,
    ToolArgsDelta,
    ToolStart,
    ToolStop,
)
from kairyu.sse import escape_json_line_separators

# ResponseErrorCode values of the pinned spec, plus context_length_exceeded:
# Codex auto-compacts only on that code inside response.failed (OpenAI sends it
# there too, outside the published enum). Every other failure keeps its
# specific code on the ``error`` event and reports server_error in the response.
_RESPONSE_ERROR_CODES = frozenset(
    {
        "server_error",
        "rate_limit_exceeded",
        "invalid_prompt",
        "data_residency_mismatch",
        "bio_policy",
        "misalignment_policy_violation",
        "vector_store_timeout",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "image_content_policy_violation",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
        "image_file_not_found",
        "context_length_exceeded",
    }
)
# Codex aborts a stream after 300 s without a *data* event (SSE comments do not
# count), so a response.in_progress heartbeat repeats the current snapshot after
# this much data silence; streams check it every _HEARTBEAT_SECONDS / 3.
_HEARTBEAT_SECONDS = 15.0


def heartbeat_tick() -> float:
    return _HEARTBEAT_SECONDS / 3


def _sse(event_type: str, sequence_number: int, **payload) -> str:
    event = {"type": event_type, "sequence_number": sequence_number, **payload}
    serialized = escape_json_line_separators(
        json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    )
    return (
        f"event: {event_type}\n"
        f"data: {serialized}\n\n"
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _output_text(text: str) -> dict:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


@dataclass
class _Item:
    kind: str
    id: str
    status: str = "in_progress"
    parts: list[str] = field(default_factory=list)
    call_id: str | None = None
    name: str | None = None
    namespace: str | None = None
    encrypted_content: str | None = None
    fixed: dict | None = None

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def wire(self) -> dict:
        if self.fixed is not None:
            return dict(self.fixed)
        if self.kind == "message":
            return {
                "type": "message",
                "id": self.id,
                "role": "assistant",
                "status": self.status,
                "content": [_output_text(self.text)],
            }
        if self.kind == "reasoning":
            item = {
                "type": "reasoning",
                "id": self.id,
                "summary": [],
                "content": [{"type": "reasoning_text", "text": self.text}],
                "status": self.status,
            }
            if self.encrypted_content is not None:
                item["encrypted_content"] = self.encrypted_content
            return item
        item = {
            "type": "function_call",
            "id": self.id,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.text,
            "status": self.status,
        }
        if self.namespace is not None:
            item["namespace"] = self.namespace
        return item


class ResponseEmitter:
    """Typed SSE events and the matching snapshot for one response."""

    def __init__(
        self,
        request: ResponsesRequest,
        *,
        response_id: str,
        created_at: int,
        seal_reasoning: Callable[[str], str] | None = None,
    ) -> None:
        self.request = request
        self.response_id = response_id
        self.created_at = created_at
        self._seal_reasoning = seal_reasoning
        self._sequence = 0
        self._items: list[_Item] = []
        self._text_encoder: ResponsesTextDeltaSSEEncoder | None = None
        self.last_data_at = time.monotonic()

    def _event(self, event_type: str, **payload) -> str:
        frame = _sse(event_type, self._sequence, **payload)
        self._sequence += 1
        self.last_data_at = time.monotonic()
        return frame

    def output(self) -> list[dict]:
        return [item.wire() for item in self._items]

    def envelope(
        self,
        status: str,
        *,
        usage: dict | None = None,
        error: dict | None = None,
        incomplete_details: dict | None = None,
    ) -> dict:
        return _response_envelope(
            self.request,
            response_id=self.response_id,
            created_at=self.created_at,
            status=status,
            output=self.output(),
            usage=usage,
            error=error,
            incomplete_details=incomplete_details,
        )

    @property
    def open_kind(self) -> str | None:
        if self._items and self._items[-1].status == "in_progress":
            return self._items[-1].kind
        return None

    @property
    def has_answer(self) -> bool:
        return any(item.kind in ("message", "function_call") for item in self._items)

    def reset(self) -> None:
        """Discard every item; only for unary turns, where nothing was sent."""

        self._items.clear()
        self._text_encoder = None

    def start(self) -> list[str]:
        snapshot = self.envelope("in_progress")
        return [
            self._event("response.created", response=snapshot),
            self._event("response.in_progress", response=snapshot),
        ]

    def heartbeat_due(self) -> bool:
        return time.monotonic() - self.last_data_at >= _HEARTBEAT_SECONDS

    def heartbeat(self) -> str:
        return self._event("response.in_progress", response=self.envelope("in_progress"))

    def _open(self, item: _Item) -> int:
        self._items.append(item)
        return len(self._items) - 1

    def begin_message(self) -> list[str]:
        item = _Item("message", _new_id("msg"))
        index = self._open(item)
        self._text_encoder = ResponsesTextDeltaSSEEncoder(item.id, output_index=index)
        return [
            self._event(
                "response.output_item.added",
                output_index=index,
                item={**item.wire(), "content": []},
            ),
            self._event(
                "response.content_part.added",
                item_id=item.id,
                output_index=index,
                content_index=0,
                part=_output_text(""),
            ),
        ]

    def text_delta(self, text: str) -> list[str | bytes]:
        self._items[-1].parts.append(text)
        assert self._text_encoder is not None
        frame = self._text_encoder.encode(self._sequence, text)
        self._sequence += 1
        self.last_data_at = time.monotonic()
        return [frame]

    def begin_reasoning(self) -> list[str]:
        item = _Item("reasoning", _new_id("rs"))
        index = self._open(item)
        return [
            self._event(
                "response.output_item.added",
                output_index=index,
                item={**item.wire(), "content": []},
            ),
            self._event(
                "response.content_part.added",
                item_id=item.id,
                output_index=index,
                content_index=0,
                part={"type": "reasoning_text", "text": ""},
            ),
        ]

    def reasoning_delta(self, text: str) -> list[str]:
        item = self._items[-1]
        item.parts.append(text)
        return [
            self._event(
                "response.reasoning_text.delta",
                item_id=item.id,
                output_index=len(self._items) - 1,
                content_index=0,
                delta=text,
            )
        ]

    def begin_call(self, call_id: str, name: str, namespace: str | None) -> list[str]:
        item = _Item(
            "function_call", _new_id("fc"), call_id=call_id, name=name, namespace=namespace
        )
        index = self._open(item)
        return [self._event("response.output_item.added", output_index=index, item=item.wire())]

    def arguments_delta(self, delta: str) -> list[str]:
        item = self._items[-1]
        item.parts.append(delta)
        return [
            self._event(
                "response.function_call_arguments.delta",
                item_id=item.id,
                output_index=len(self._items) - 1,
                delta=delta,
            )
        ]

    def close(self, status: str = "completed") -> list[str]:
        """Finish the open item (if any) with its done events."""

        if self.open_kind is None:
            return []
        item = self._items[-1]
        index = len(self._items) - 1
        item.status = status
        frames: list[str] = []
        if item.kind == "message":
            frames.append(
                self._event(
                    "response.output_text.done",
                    item_id=item.id,
                    output_index=index,
                    content_index=0,
                    text=item.text,
                    logprobs=[],
                )
            )
            frames.append(
                self._event(
                    "response.content_part.done",
                    item_id=item.id,
                    output_index=index,
                    content_index=0,
                    part=_output_text(item.text),
                )
            )
        elif item.kind == "reasoning":
            if self._seal_reasoning is not None:
                item.encrypted_content = self._seal_reasoning(item.text)
            frames.append(
                self._event(
                    "response.reasoning_text.done",
                    item_id=item.id,
                    output_index=index,
                    content_index=0,
                    text=item.text,
                )
            )
            frames.append(
                self._event(
                    "response.content_part.done",
                    item_id=item.id,
                    output_index=index,
                    content_index=0,
                    part={"type": "reasoning_text", "text": item.text},
                )
            )
        else:
            frames.append(
                self._event(
                    "response.function_call_arguments.done",
                    item_id=item.id,
                    output_index=index,
                    arguments=item.text,
                    name=item.name,
                )
            )
        frames.append(
            self._event("response.output_item.done", output_index=index, item=item.wire())
        )
        return frames

    def add_item(self, wire_item: dict) -> list[str]:
        """Emit a complete item (e.g. ``compaction``) as added + done."""

        item = _Item(wire_item["type"], wire_item["id"], status="completed", fixed=dict(wire_item))
        index = self._open(item)
        return [
            self._event(
                "response.output_item.added",
                output_index=index,
                item={**wire_item, "status": "in_progress"},
            ),
            self._event("response.output_item.done", output_index=index, item=wire_item),
        ]

    def complete(
        self, status: str, usage: dict, incomplete_details: dict | None = None
    ) -> tuple[dict, list[str]]:
        frames = self.close("incomplete" if status == "incomplete" else "completed")
        envelope = self.envelope(status, usage=usage, incomplete_details=incomplete_details)
        terminal = "response.completed" if status == "completed" else "response.incomplete"
        frames.append(self._event(terminal, response=envelope))
        return envelope, frames

    def fail(self, payload: dict, usage: dict) -> tuple[dict, list[str]]:
        """``error`` + ``response.failed``; a half-written item stays incomplete."""

        if self.open_kind is not None:
            self._items[-1].status = "incomplete"
        code = payload.get("code") or "server_error"
        message = payload.get("message") or "generation failed"
        frames = [self._event("error", code=code, message=message, param=payload.get("param"))]
        response_code = code if code in _RESPONSE_ERROR_CODES else "server_error"
        envelope = self.envelope(
            "failed", usage=usage, error={"code": response_code, "message": message}
        )
        frames.append(self._event("response.failed", response=envelope))
        return envelope, frames


def gate_failure(code: str, message: str) -> dict:
    return {"message": message, "type": "upstream_error", "param": None, "code": code}


class OutputAssembler:
    """Maps reasoning, prose, and tool calls onto one emitter's items."""

    def __init__(
        self,
        emitter: ResponseEmitter,
        *,
        scanner=None,
        tool_choice: NormalizedToolChoice | None = None,
        parallel: bool | None = None,
        namespaces: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._emitter = emitter
        self._scanner = scanner
        self._choice = tool_choice
        self._parallel = parallel
        self._namespaces = namespaces or {}
        self._pending_ws = ""
        self._late_reasoning: list[str] = []
        self.calls = 0
        self.failure: dict | None = None

    def reasoning(self, text: str) -> list[str]:
        if not text or self.failure is not None:
            return []
        if self._emitter.open_kind in ("message", "function_call"):
            # Reasoning interleaved into an answer item waits for that item to
            # close: closing it early would split a message or cut a call's
            # arguments.
            self._late_reasoning.append(text)
            return []
        frames: list[str] = []
        if self._emitter.open_kind != "reasoning":
            frames += self._emitter.begin_reasoning()
        return frames + self._emitter.reasoning_delta(text)

    def _close(self, status: str = "completed", *, keep_reasoning_open: bool = False) -> list[str]:
        """Close the open item, then place reasoning that waited for it."""

        frames = self._emitter.close(status)
        if self._late_reasoning:
            text = "".join(self._late_reasoning)
            self._late_reasoning.clear()
            frames += self._emitter.begin_reasoning() + self._emitter.reasoning_delta(text)
            if not keep_reasoning_open:
                frames += self._emitter.close(status)
        return frames

    def content(self, delta: str, *, final: bool = False) -> list[str | bytes]:
        if self.failure is not None:
            return []
        if self._scanner is None:
            return self._apply([TextDelta(delta)] if delta else [])
        if not delta and not final:
            return []
        return self._apply(self._scanner.feed(delta, final=final))

    def parsed_calls(self, tool_calls: list[dict]) -> list[str | bytes]:
        """Calls the chat handler already parsed and gated (AUTO unary)."""

        frames: list[str | bytes] = []
        for call in tool_calls:
            function = call.get("function") or {}
            frames += self._apply(
                [
                    ToolStart(
                        id=call.get("id") or _new_id("call"), name=function.get("name") or ""
                    ),
                    ToolArgsDelta(function.get("arguments") or ""),
                    ToolStop(),
                ]
            )
        return frames

    def _apply(self, events) -> list[str | bytes]:
        frames: list[str | bytes] = []
        for event in events:
            if self.failure is not None:
                break
            if isinstance(event, StreamInvalid):
                self.failure = gate_failure("invalid_tool_call", event.message)
            elif isinstance(event, TextDelta):
                if self._emitter.open_kind != "message":
                    if not event.text.strip():
                        self._pending_ws += event.text
                        continue
                    frames += self._close()
                    frames += self._emitter.begin_message()
                text, self._pending_ws = self._pending_ws + event.text, ""
                frames += self._emitter.text_delta(text)
            elif isinstance(event, ToolStart):
                if self._parallel is False and self.calls >= 1:
                    self.failure = gate_failure(
                        "parallel_tool_calls_not_satisfied",
                        "upstream model emitted multiple calls while parallel_tool_calls=false",
                    )
                    break
                frames += self._close()
                self._pending_ws = ""
                namespace, name = self._namespaces.get(event.name, (None, event.name))
                frames += self._emitter.begin_call(event.id, name, namespace)
                self.calls += 1
            elif isinstance(event, ToolArgsDelta):
                frames += self._emitter.arguments_delta(event.partial_json)
            elif isinstance(event, ToolStop):
                frames += self._close(keep_reasoning_open=True)
        return frames

    def finish(self, *, incomplete: bool) -> list[str | bytes]:
        """Close the output, applying the end-of-stream tool_choice gate."""

        if self.failure is not None:
            return []
        if (
            self._choice is not None
            and self._choice.mode in {"required", "named"}
            and self.calls == 0
        ):
            self.failure = gate_failure(
                "tool_choice_not_satisfied", "upstream model did not satisfy tool_choice"
            )
            return []
        status = "incomplete" if incomplete else "completed"
        frames: list[str | bytes] = list(self._close(status))
        if not self._emitter.has_answer:
            frames += self._emitter.begin_message()
            if self._pending_ws:
                frames += self._emitter.text_delta(self._pending_ws)
            frames += self._emitter.close(status)
        return frames


class ReasoningSplit:
    """Separates reasoning from content for one completion stream.

    Backend-separated reasoning (``reasoning_delta_after``) wins; otherwise an
    inline ``<think>`` parser runs whether or not ``reasoning.effort`` was set,
    starting inside the block when the rendered prompt already opened it.
    """

    def __init__(self, *, opened: bool = False) -> None:
        self._parser = ReasoningDeltaParser()
        if opened:
            self._parser.reasoning = True
        self._reasoning_offset = 0
        self._remote = False

    def feed(self, completion, text: str, *, final: bool) -> tuple[str, str]:
        remote = ""
        if completion is not None:
            remote, self._reasoning_offset = completion.reasoning_delta_after(
                self._reasoning_offset
            )
        if remote:
            self._remote = True
        if self._remote:
            return remote, text
        return self._parser.feed(text, final=final)


def prompt_opens_reasoning(prompt) -> bool:
    """True when the rendered generation prompt ends inside ``<think>``."""

    text = prompt_text(prompt)
    return text is not None and text.rstrip().endswith("<think>")


async def buffered_stream(emitter: ResponseEmitter, produce, saver) -> AsyncIterator[str]:
    """Stream a turn whose output exists only at the end (e.g. compaction).

    The opening events flush immediately and data heartbeats cover the wait;
    ``produce`` returns ``(output, usage, status, incomplete_details)`` or
    raises ``_BufferedFailure``.
    """

    for frame in emitter.start():
        yield frame
    task = asyncio.ensure_future(produce())
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=heartbeat_tick())
            if done:
                break
            if emitter.heartbeat_due():
                yield emitter.heartbeat()
    except BaseException:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise
    try:
        output, usage, status, incomplete_details = task.result()
    except _BufferedFailure as failure:
        _envelope, frames = emitter.fail(failure.payload, _usage_payload_from_wire(None))
        for frame in frames:
            yield frame
        return
    frames: list[str] = []
    for item in output:
        frames += emitter.add_item(item)
    envelope, terminal = emitter.complete(status, usage, incomplete_details)
    for frame in frames + terminal:
        yield frame
    if output:
        # An incomplete compaction produced no replacement context; storing an
        # empty continuation entry would silently blank a thread.
        saver.commit(envelope)


async def failed_stream(emitter: ResponseEmitter, payload: dict) -> AsyncIterator[str]:
    """An in-band failure for a stream that never dispatched (context overflow)."""

    for frame in emitter.start():
        yield frame
    _envelope, frames = emitter.fail(payload, _usage_payload_from_wire(None))
    for frame in frames:
        yield frame
