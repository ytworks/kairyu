"""vLLM OpenAI-server wire shapes rendered by ``FakeVLLMUpstream`` (M20 WP-02b).

Pure functions from a scripted ``Turn`` to vLLM's JSON payloads: chat and
text-completion bodies and SSE chunks, usage with token details, and the
error bodies vLLM returns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from tests.support.scenario_script import (
    Part,
    Piece,
    Reasoning,
    ToolCall,
    Turn,
    channel_text,
    chunk_pieces,
    toy_tokens,
)

ErrorStyle = Literal["nested", "flat"]
ReasoningField = Literal["reasoning", "reasoning_content"]
CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
_CREATED = 1_700_000_000


def vllm_error_body(
    message: str,
    *,
    err_type: str = "BadRequestError",
    code: int = 400,
    style: ErrorStyle = "nested",
) -> dict[str, object]:
    """vLLM's ``ErrorResponse``: nested under ``error`` (current releases) or
    flat with ``"object": "error"`` (older releases)."""

    fields = {"message": message, "type": err_type, "param": None, "code": code}
    return {"error": fields} if style == "nested" else {"object": "error", **fields}


def vllm_context_length_message(
    *,
    max_model_len: int,
    prompt_tokens: int,
    max_tokens: int | None,
    style: ErrorStyle = "nested",
) -> str:
    """vLLM's prompt-validation overflow text for each body style's releases.

    ``max_tokens=None`` is the prompt-alone overflow. Reconstructed from
    vLLM's ``_validate_input``; SP-1 replaces these with bodies captured from
    the pinned vLLM image.
    """

    head = f"This model's maximum context length is {max_model_len} tokens."
    if style == "flat":
        if max_tokens is None:
            return (
                f"{head} However, you requested {prompt_tokens} tokens in the "
                "messages, Please reduce the length of the messages."
            )
        return (
            f"{head} However, you requested {prompt_tokens + max_tokens} tokens "
            f"({prompt_tokens} in the messages, {max_tokens} in the completion). "
            "Please reduce the length of the messages or completion."
        )
    if max_tokens is None:
        return (
            f"{head} However, your request has {prompt_tokens} input tokens. "
            "Please reduce the length of the input messages."
        )
    return (
        f"'max_tokens' or 'max_completion_tokens' is too large: {max_tokens}. "
        f"This model's maximum context length is {max_model_len} tokens and your "
        f"request has {prompt_tokens} input tokens "
        f"({max_tokens} > {max_model_len} - {prompt_tokens})."
    )


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, Mapping)
        )
    return ""


def _messages(body: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ()
    return tuple(message for message in messages if isinstance(message, Mapping))


def chat_prompt_text(body: Mapping[str, object]) -> str:
    """A role-prefixed rendering whose toy tokens stand in for the template's."""

    lines = []
    for message in _messages(body):
        calls = message.get("tool_calls")
        rendered_calls = json.dumps(calls, sort_keys=True) if isinstance(calls, list) else ""
        lines.append(f"{message.get('role')}: {_text_of(message.get('content'))}{rendered_calls}")
    return "\n".join(lines)


def last_user_message(body: Mapping[str, object]) -> str:
    users = tuple(message for message in _messages(body) if message.get("role") == "user")
    return _text_of(users[-1].get("content")) if users else ""


def _chat_part_pieces(index: int, part: Part) -> tuple[Piece, ...]:
    if isinstance(part, ToolCall):
        # Models emit spaced JSON, so argument fragments split at spaces.
        spaced = json.dumps(part.argument_object(), ensure_ascii=False)
        arguments = tuple(Piece(index, "call_args", text) for text in toy_tokens(spaced))
        return (Piece(index, "call_name", part.name), *arguments)
    if isinstance(part, Reasoning) and part.typed:
        return tuple(Piece(index, "reasoning", text) for text in toy_tokens(part.text))
    if isinstance(part, Reasoning):
        texts = ("<think>", *toy_tokens(part.text), "</think>")
        return tuple(Piece(index, "content", text) for text in texts)
    return tuple(Piece(index, "content", text) for text in toy_tokens(part))


def _completion_part_pieces(index: int, part: Part, end_tag: str) -> tuple[Piece, ...]:
    if isinstance(part, ToolCall):
        texts = toy_tokens(part.generic_envelope())
    elif isinstance(part, Reasoning) and part.typed:
        # A template that opens the span: the model emits only its terminator.
        texts = (*toy_tokens(part.text), end_tag)
    elif isinstance(part, Reasoning):
        texts = ("<think>", *toy_tokens(part.text), "</think>")
    else:
        texts = toy_tokens(part)
    return tuple(Piece(index, "text", text) for text in texts)


def turn_pieces(turn: Turn, *, chat: bool, end_tag: str) -> tuple[Piece, ...]:
    """The tokens vLLM would stream for ``turn`` on the chosen endpoint."""

    return tuple(
        piece
        for index, part in enumerate(turn.parts)
        for piece in (
            _chat_part_pieces(index, part)
            if chat
            else _completion_part_pieces(index, part, end_tag)
        )
    )


def usage_payload(
    turn: Turn,
    prompt_tokens: int,
    pieces: Sequence[Piece],
    *,
    details: bool,
) -> dict[str, object]:
    """vLLM ``UsageInfo``; ``turn.usage`` replaces the computed counts."""

    exact = turn.usage
    prompt = exact.prompt_tokens if exact is not None else prompt_tokens
    completion = exact.completion_tokens if exact is not None else len(pieces)
    usage: dict[str, object] = {
        "prompt_tokens": prompt,
        "total_tokens": prompt + completion,
        "completion_tokens": completion,
        "prompt_tokens_details": None,
    }
    if not details:
        return usage
    reasoning_tokens = sum(
        1
        for piece in pieces
        if isinstance(part := turn.parts[piece.part], Reasoning) and part.typed
    )
    return {
        **usage,
        "prompt_tokens_details": {"cached_tokens": exact.cached_tokens if exact else 0},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


@dataclass(frozen=True)
class Generation:
    """One accepted generation request, ready to render."""

    chat: bool
    index: int
    model: str
    reasoning_field: ReasoningField
    turn: Turn
    pieces: tuple[Piece, ...]
    cut: bool
    prompt_tokens: int
    token_details: bool

    @property
    def finish_reason(self) -> str:
        if self.cut:
            return "length"
        if self.chat and self.turn.has_tool_calls:
            return "tool_calls"
        return self.turn.finish_reason

    def usage(self) -> dict[str, object]:
        return usage_payload(self.turn, self.prompt_tokens, self.pieces, details=self.token_details)

    def envelope(self, choices: Sequence[object], *, chunk: bool, **extra: object) -> dict:
        if self.chat:
            prefix = "chatcmpl-fake"
            kind = "chat.completion.chunk" if chunk else "chat.completion"
        else:
            prefix, kind = "cmpl-fake", "text_completion"
        return {
            "id": f"{prefix}{self.index}",
            "object": kind,
            "created": _CREATED,
            "model": self.model,
            "choices": list(choices),
            **extra,
        }

    def call_ordinal(self, part_index: int) -> int:
        return sum(isinstance(part, ToolCall) for part in self.turn.parts[:part_index])

    def call_id(self, part_index: int) -> str:
        return f"chatcmpl-tool-{self.index}-{self.call_ordinal(part_index)}"


def _chat_delta(generation: Generation, chunk: tuple[Piece, ...]) -> dict[str, object]:
    head = chunk[0]
    part = generation.turn.parts[head.part]
    if isinstance(part, ToolCall):
        arguments = channel_text(chunk, "call_args")
        call: dict[str, object] = {
            "index": generation.call_ordinal(head.part),
            "function": {"arguments": arguments},
        }
        if head.channel == "call_name":
            call = {
                "id": generation.call_id(head.part),
                "type": "function",
                **call,
                "function": {"name": part.name, "arguments": arguments},
            }
        return {"tool_calls": [call]}
    if head.channel == "reasoning":
        return {generation.reasoning_field: channel_text(chunk, "reasoning")}
    return {"content": channel_text(chunk, "content")}


def render_stream(
    generation: Generation,
    *,
    include_usage: bool,
) -> tuple[tuple[dict, ...], tuple[dict, ...], tuple[dict, ...]]:
    """Return (head, per-chunk, tail) SSE payloads; ``[DONE]`` is the caller's."""

    chunks = chunk_pieces(generation.pieces, generation.turn.chunk_tokens)
    finish = generation.finish_reason
    if generation.chat:

        def choice(delta: dict, finish_reason: str | None) -> dict:
            return {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}

        role = choice({"role": "assistant", "content": ""}, None)
        head = (generation.envelope([role], chunk=True),)
        body = tuple(
            generation.envelope([choice(_chat_delta(generation, chunk), None)], chunk=True)
            for chunk in chunks
        )
        final = generation.envelope([{**choice({}, finish), "stop_reason": None}], chunk=True)
    else:

        def text_choice(text: str, finish_reason: str | None) -> dict:
            return {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
                "stop_reason": None,
            }

        head = ()
        body = tuple(
            generation.envelope([text_choice(channel_text(chunk, "text"), None)], chunk=True)
            for chunk in chunks
        )
        final = generation.envelope([text_choice("", finish)], chunk=True)
    if not include_usage:
        return head, body, (final,)
    return head, body, (final, generation.envelope([], chunk=True, usage=generation.usage()))


def render_unary(generation: Generation) -> dict:
    pieces = generation.pieces
    finish = generation.finish_reason
    if not generation.chat:
        choice = {
            "index": 0,
            "text": channel_text(pieces, "text"),
            "logprobs": None,
            "finish_reason": finish,
            "stop_reason": None,
            "prompt_logprobs": None,
        }
        return generation.envelope([choice], chunk=False, usage=generation.usage())
    calls = [
        {
            "id": generation.call_id(index),
            "type": "function",
            "function": {
                "name": part.name,
                "arguments": channel_text(
                    tuple(piece for piece in pieces if piece.part == index), "call_args"
                ),
            },
        }
        for index, part in enumerate(generation.turn.parts)
        if isinstance(part, ToolCall) and any(piece.part == index for piece in pieces)
    ]
    content = channel_text(pieces, "content")
    message = {
        "role": "assistant",
        "content": content or (None if calls else ""),
        generation.reasoning_field: channel_text(pieces, "reasoning") or None,
        "tool_calls": calls,
    }
    choice = {
        "index": 0,
        "message": message,
        "logprobs": None,
        "finish_reason": finish,
        "stop_reason": None,
    }
    envelope = generation.envelope([choice], chunk=False, usage=generation.usage())
    return {**envelope, "prompt_logprobs": None}
