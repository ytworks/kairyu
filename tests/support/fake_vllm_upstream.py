"""FakeVLLMUpstream: a vLLM OpenAI-compatible server double (M20 WP-02b).

It serves what ``OpenAICompatBackend`` (``kairyu/engine/openai_backend.py``)
sends and parses: ``POST /v1/chat/completions`` and ``/v1/completions``
(unary and SSE, with the ``stream_options.include_usage`` chunk), root
``POST /tokenize``, vLLM error bodies (including the context-length 400 and
in-band stream error frames), and usage carrying ``prompt_tokens_details``
and ``completion_tokens_details``.

Use it as ``transport()`` (an ``httpx.MockTransport``) or as an ASGI app: the
instance itself, e.g. ``uvicorn.run(fake)`` or ``httpx.ASGITransport(fake)``.
Chat turns stream like vLLM's tool and reasoning parsers: a ``ToolCall``
becomes ``tool_calls`` deltas (id/type/name first, then argument fragments)
and typed ``Reasoning`` uses the configured ``reasoning_field``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from tests.support.scenario_script import (
    Scenario,
    Turn,
    UnscriptedCallError,
    chunk_delay_s,
    chunk_pieces,
    limit_pieces,
    pending_user_text,
    toy_tokens,
    unary_delay_s,
)
from tests.support.vllm_wire import (
    CHAT_PATH,
    COMPLETIONS_PATH,
    ErrorStyle,
    Generation,
    ReasoningField,
    chat_prompt_text,
    pending_user_message,
    render_stream,
    render_unary,
    turn_pieces,
    vllm_context_length_message,
    vllm_context_overflow_message,
    vllm_error_body,
    vllm_stream_error_frame,
)

__all__ = [
    "FakeVLLMUpstream",
    "UpstreamCall",
    "vllm_context_length_message",
    "vllm_error_body",
    "vllm_stream_error_frame",
]

TOKENIZE_PATH = "/tokenize"
_ROUTES = frozenset({CHAT_PATH, COMPLETIONS_PATH, TOKENIZE_PATH})
_JSON = "application/json"
_SSE = "text/event-stream"
_DONE = b"data: [DONE]\n\n"


@dataclass(frozen=True)
class UpstreamCall:
    """One received HTTP request; ``content`` is the raw body."""

    method: str
    path: str
    content: bytes

    def json(self) -> Any:
        return json.loads(self.content)


@dataclass(frozen=True)
class _Reply:
    """A response as frames, each sent after its delay (unary: one frame)."""

    status: int
    content_type: str
    frames: tuple[bytes, ...]
    delays: tuple[float, ...]

    async def iterate(self) -> AsyncIterator[bytes]:
        for delay, frame in zip(self.delays, self.frames, strict=True):
            if delay > 0:
                await asyncio.sleep(delay)
            yield frame


class _ReplyStream(httpx.AsyncByteStream):
    def __init__(self, reply: _Reply) -> None:
        self._reply = reply

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for frame in self._reply.iterate():
            yield frame


def _json_reply(status: int, payload: object, delay: float = 0.0) -> _Reply:
    return _Reply(status, _JSON, (json.dumps(payload).encode(),), (delay,))


def _sse(payload: object) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _paced_stream(
    turn: Turn,
    head: tuple[object, ...],
    chunks: tuple[object, ...],
    tail: tuple[object, ...],
    failure: object,
) -> _Reply:
    """SSE reply paced like the turn.

    vLLM sends ``head`` (the role chunk) with the first token, so it shares
    the first frame. An injected failure replaces the tail after
    ``fail_after_chunks`` chunks.
    """

    if turn.fail_after_chunks is not None:
        chunks, tail = chunks[: turn.fail_after_chunks], (failure,)
    lead = b"".join(_sse(payload) for payload in head)
    encoded = tuple(_sse(payload) for payload in chunks)
    timed = (lead + encoded[0], *encoded[1:]) if encoded else ((lead,) if lead else ())
    rest = (*(_sse(payload) for payload in tail), _DONE)
    delays = tuple(chunk_delay_s(turn, position) for position in range(len(timed)))
    return _Reply(200, _SSE, (*timed, *rest), (*delays, *(0.0 for _ in rest)))


class FakeVLLMUpstream:
    """Scripted vLLM server; ``calls`` is its request log (single owner)."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        model: str = "fake-vllm",
        max_model_len: int | None = None,
        reasoning_field: ReasoningField = "reasoning",
        error_style: ErrorStyle = "nested",
        completion_reasoning_end_tag: str = "</think>",
        report_token_details: bool = True,
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError("FakeVLLMUpstream requires a Scenario")
        if max_model_len is not None and (type(max_model_len) is not int or max_model_len < 1):
            raise ValueError("max_model_len must be a positive integer or None")
        if reasoning_field not in {"reasoning", "reasoning_content"}:
            raise ValueError("reasoning_field must be reasoning or reasoning_content")
        if error_style not in {"nested", "flat"}:
            raise ValueError("error_style must be nested or flat")
        self._scenario = scenario
        self._model = model
        self._max_model_len = max_model_len
        self._reasoning_field: ReasoningField = reasoning_field
        self._error_style: ErrorStyle = error_style
        self._end_tag = completion_reasoning_end_tag
        self._token_details = report_token_details
        self._calls: list[UpstreamCall] = []
        self._generations = 0

    @property
    def calls(self) -> tuple[UpstreamCall, ...]:
        return tuple(self._calls)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        reply = self._route(request.method, request.url.path, await request.aread())
        return httpx.Response(
            reply.status,
            headers={"content-type": reply.content_type},
            stream=_ReplyStream(reply),
        )

    async def __call__(
        self,
        scope: Mapping[str, Any],
        receive: Callable[[], Awaitable[Mapping[str, Any]]],
        send: Callable[[Mapping[str, Any]], Awaitable[None]],
    ) -> None:
        """ASGI entry point (HTTP and lifespan scopes)."""

        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        if scope["type"] != "http":
            return
        parts: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            parts.append(message.get("body", b""))
            more_body = bool(message.get("more_body", False))
        reply = self._route(scope["method"], scope["path"], b"".join(parts))
        headers = [(b"content-type", reply.content_type.encode())]
        await send({"type": "http.response.start", "status": reply.status, "headers": headers})
        async for frame in reply.iterate():
            await send({"type": "http.response.body", "body": frame, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    def _error(self, status: int, message: str, err_type: str = "BadRequestError") -> _Reply:
        body = vllm_error_body(message, err_type=err_type, code=status, style=self._error_style)
        return _json_reply(status, body)

    def _route(self, method: str, path: str, content: bytes) -> _Reply:
        self._calls.append(UpstreamCall(method, path, content))
        if path not in _ROUTES:
            return _json_reply(404, {"detail": "Not Found"})
        if method != "POST":
            return _json_reply(405, {"detail": "Method Not Allowed"})
        try:
            body = json.loads(content)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return self._error(400, "request body must be a JSON object")
        if path == TOKENIZE_PATH:
            return self._tokenize(body)
        return self._generate(path == CHAT_PATH, body)

    def _tokenize(self, body: Mapping[str, object]) -> _Reply:
        text = chat_prompt_text(body) if "messages" in body else str(body.get("prompt", ""))
        count = len(toy_tokens(text))
        payload = {
            "count": count,
            "max_model_len": self._max_model_len,
            "tokens": list(range(count)),
            "token_strs": None,
        }
        return _json_reply(200, payload)

    def _overflow(self, prompt_tokens: int, max_tokens: int | None) -> _Reply | None:
        """vLLM's pre-generation context check, with its 400 error body."""

        if self._max_model_len is None:
            return None
        message = vllm_context_overflow_message(
            max_model_len=self._max_model_len,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            style=self._error_style,
        )
        return None if message is None else self._error(400, message)

    def _generate(self, chat: bool, body: Mapping[str, object]) -> _Reply:
        prompt = chat_prompt_text(body) if chat else str(body.get("prompt", ""))
        prompt_tokens = len(toy_tokens(prompt))
        requested = body.get("max_completion_tokens", body.get("max_tokens"))
        max_tokens = requested if type(requested) is int else None
        overflow = self._overflow(prompt_tokens, max_tokens)
        if overflow is not None:
            return overflow
        index, self._generations = self._generations, self._generations + 1
        try:
            user_text = pending_user_message(body) if chat else pending_user_text(prompt)
            turn = self._scenario.select(index, user_text)
        except UnscriptedCallError as error:
            return self._error(500, str(error), "InternalServerError")
        if max_tokens is None and self._max_model_len is not None:
            max_tokens = self._max_model_len - prompt_tokens
        pieces, cut = limit_pieces(turn_pieces(turn, chat=chat, end_tag=self._end_tag), max_tokens)
        generation = Generation(
            chat=chat,
            index=index,
            model=self._model,
            reasoning_field=self._reasoning_field,
            turn=turn,
            pieces=pieces,
            cut=cut,
            prompt_tokens=prompt_tokens,
            token_details=self._token_details,
        )
        if body.get("stream") is True:
            options = body.get("stream_options")
            with_usage = isinstance(options, Mapping) and options.get("include_usage") is True
            head, chunks, tail = render_stream(generation, include_usage=with_usage)
            # vLLM reports a failure inside the stream, as BadRequestError.
            failure = vllm_stream_error_frame("scripted stream failure", style=self._error_style)
            return _paced_stream(turn, head, chunks, tail, failure)
        if turn.fail_after_chunks is not None:
            return self._error(500, "scripted generation failure", "InternalServerError")
        delay = unary_delay_s(turn, len(chunk_pieces(pieces, turn.chunk_tokens)))
        return _json_reply(200, render_unary(generation), delay)
