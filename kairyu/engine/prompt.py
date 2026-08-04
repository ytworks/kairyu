"""Typed prompt values and their lossless transport representation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, TypeAlias

PromptKind: TypeAlias = Literal["text", "tokens", "multimodal"]
MultimodalEncoding: TypeAlias = Literal["uri", "base64", "bytes", "json"]

_MAX_TOKEN_ID = (1 << 64) - 1
_TOKEN_FINGERPRINT_DOMAIN = b"kairyu:token-prompt:v1\x00"
_MULTIMODAL_ENCODINGS = frozenset({"uri", "base64", "bytes", "json"})


@dataclass(frozen=True)
class TextPrompt:
    """An explicitly typed text prompt.

    Plain strings remain valid ``PromptInput`` values for backward
    compatibility. This wrapper is useful at typed transport boundaries.
    """

    prompt: str

    def __post_init__(self) -> None:
        if type(self.prompt) is not str:
            raise TypeError("TextPrompt.prompt must be a string")


class TemplatedPrompt(str):
    """Text already rendered by a tokenizer-owned chat template.

    Unlike an ordinary completion prompt, the template already owns the
    model's BOS/EOS and control-token policy, including deliberate omission.
    A backend that tokenizes it must therefore disable automatic special-token
    insertion. Keeping that ownership in the prompt value prevents chat and
    completion paths from silently sharing incompatible tokenizer defaults.
    """

    def __new__(cls, prompt: str) -> TemplatedPrompt:
        if type(prompt) is not str:
            raise TypeError("TemplatedPrompt.prompt must be a string")
        return super().__new__(cls, prompt)

    @property
    def prompt(self) -> str:
        return str(self)


@dataclass(frozen=True)
class TokensPrompt:
    """Caller-tokenized input whose IDs are the source of accounting truth.

    ``prompt`` is optional display text only. It is deliberately excluded from
    cache identity because the caller, rather than a backend tokenizer, owns
    the token sequence.
    """

    prompt_token_ids: tuple[int, ...]
    prompt: str | None = None

    def __post_init__(self) -> None:
        token_ids = self.prompt_token_ids
        if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(
            token_ids, Sequence
        ):
            raise TypeError("TokensPrompt.prompt_token_ids must be a sequence of integers")
        copied = tuple(token_ids)
        if not copied:
            raise ValueError("TokensPrompt.prompt_token_ids must not be empty")
        for token_id in copied:
            if type(token_id) is not int:
                raise TypeError(
                    "TokensPrompt.prompt_token_ids must contain integers, not "
                    f"{type(token_id).__name__}"
                )
            if not 0 <= token_id <= _MAX_TOKEN_ID:
                raise ValueError(
                    "TokensPrompt.prompt_token_ids values must be in "
                    f"the unsigned 64-bit range 0..{_MAX_TOKEN_ID}"
                )
        if self.prompt is not None and type(self.prompt) is not str:
            raise TypeError("TokensPrompt.prompt must be a string or None")
        object.__setattr__(self, "prompt_token_ids", copied)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _render_canonical_json(value: object) -> str:
    """Render parsed strict JSON without binary-float numeric conversion."""

    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return str(value)
    if type(value) is str:
        return json.dumps(value, ensure_ascii=False)
    if type(value) is list:
        return "[" + ",".join(_render_canonical_json(item) for item in value) + "]"
    if type(value) is dict:
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:"
                f"{_render_canonical_json(value[key])}"
                for key in sorted(value)
            )
            + "}"
        )
    raise TypeError(f"unsupported parsed JSON value: {type(value).__name__}")


def _canonical_json(value: str) -> str:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
            parse_float=Decimal,
            parse_int=Decimal,
        )
        return _render_canonical_json(parsed)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("json multimodal data must be valid strict JSON") from exc


@dataclass(frozen=True)
class MultimodalItem:
    """One opaque multimodal input with an explicit wire encoding."""

    modality: str
    encoding: MultimodalEncoding
    data: str | bytes

    def __post_init__(self) -> None:
        if type(self.modality) is not str or not self.modality.strip():
            raise ValueError("MultimodalItem.modality must be a non-empty string")
        if type(self.encoding) is not str or self.encoding not in _MULTIMODAL_ENCODINGS:
            allowed = ", ".join(sorted(_MULTIMODAL_ENCODINGS))
            raise ValueError(f"MultimodalItem.encoding must be one of: {allowed}")

        if self.encoding == "bytes":
            if type(self.data) is not bytes:
                raise TypeError("bytes multimodal data must be bytes")
            return

        if type(self.data) is not str:
            raise TypeError(f"{self.encoding} multimodal data must be a string")
        if self.encoding == "uri":
            if not self.data:
                raise ValueError("uri multimodal data must not be empty")
            return
        if self.encoding == "base64":
            try:
                base64.b64decode(self.data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("base64 multimodal data is not valid base64") from exc
            return

        object.__setattr__(self, "data", _canonical_json(self.data))


@dataclass(frozen=True)
class MultimodalMessagePart:
    """One ordered chat part: literal text or a reference into ``items``.

    A reference is deliberately an integer index rather than a copied media
    payload.  This keeps the chat layout small while preserving the exact
    message/part position of every image across process and replica
    boundaries.
    """

    type: Literal["text", "item"]
    text: str | None = None
    item_index: int | None = None
    detail: Literal["auto", "low", "high"] | None = None

    def __post_init__(self) -> None:
        if self.type == "text":
            if (
                type(self.text) is not str
                or self.item_index is not None
                or self.detail is not None
            ):
                raise ValueError(
                    "text multimodal message parts require only a string text payload"
                )
            return
        if self.type != "item":
            raise ValueError("multimodal message part type must be 'text' or 'item'")
        if (
            type(self.item_index) is not int
            or self.item_index < 0
            or self.text is not None
            or self.detail not in (None, "auto", "low", "high")
        ):
            raise ValueError(
                "item multimodal message parts require only a non-negative item_index"
            )


@dataclass(frozen=True)
class MultimodalMessage:
    """One role-preserving message consumed by a capable chat backend."""

    role: str
    content: tuple[MultimodalMessagePart, ...]

    def __post_init__(self) -> None:
        if type(self.role) is not str or not self.role.strip():
            raise ValueError("MultimodalMessage.role must be a non-empty string")
        content = self.content
        if isinstance(content, (str, bytes, bytearray)) or not isinstance(
            content, Sequence
        ):
            raise TypeError("MultimodalMessage.content must be a sequence")
        copied = tuple(content)
        if not copied:
            raise ValueError("MultimodalMessage.content must not be empty")
        if any(not isinstance(part, MultimodalMessagePart) for part in copied):
            raise TypeError(
                "MultimodalMessage.content must contain MultimodalMessagePart values"
            )
        object.__setattr__(self, "content", copied)


@dataclass(frozen=True)
class MultimodalPrompt:
    """A text or token base plus non-text items owned by a capable backend."""

    base: str | TextPrompt | TokensPrompt
    items: tuple[MultimodalItem, ...]
    messages: tuple[MultimodalMessage, ...] = ()

    def __post_init__(self) -> None:
        if type(self.base) is not str and not isinstance(
            self.base, (TextPrompt, TokensPrompt)
        ):
            raise TypeError(
                "MultimodalPrompt.base must be a string, TextPrompt, or TokensPrompt"
            )
        items = self.items
        if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
            raise TypeError("MultimodalPrompt.items must be a sequence")
        copied = tuple(items)
        if not copied:
            raise ValueError("MultimodalPrompt.items must not be empty")
        if any(not isinstance(item, MultimodalItem) for item in copied):
            raise TypeError("MultimodalPrompt.items must contain MultimodalItem values")
        object.__setattr__(self, "items", copied)

        messages = self.messages
        if isinstance(messages, (str, bytes, bytearray)) or not isinstance(
            messages, Sequence
        ):
            raise TypeError("MultimodalPrompt.messages must be a sequence")
        copied_messages = tuple(messages)
        if any(not isinstance(message, MultimodalMessage) for message in copied_messages):
            raise TypeError(
                "MultimodalPrompt.messages must contain MultimodalMessage values"
            )
        if copied_messages:
            references = [
                part.item_index
                for message in copied_messages
                for part in message.content
                if part.type == "item"
            ]
            expected = list(range(len(copied)))
            if sorted(references) != expected:
                raise ValueError(
                    "multimodal chat layout must reference every item exactly once"
                )
        object.__setattr__(self, "messages", copied_messages)


PromptInput: TypeAlias = (
    str | TextPrompt | TemplatedPrompt | TokensPrompt | MultimodalPrompt
)
PromptWire: TypeAlias = str | dict[str, object]


def prompt_kind(prompt: PromptInput) -> PromptKind:
    """Return the nominal prompt variant without flattening its contents."""

    if type(prompt) is str or isinstance(prompt, (TextPrompt, TemplatedPrompt)):
        return "text"
    if isinstance(prompt, TokensPrompt):
        return "tokens"
    if isinstance(prompt, MultimodalPrompt):
        return "multimodal"
    raise TypeError(f"unsupported prompt type: {type(prompt).__name__}")


def prompt_text(prompt: PromptInput) -> str | None:
    """Return directly usable text, never a lossy multimodal projection."""

    if type(prompt) is str:
        return prompt
    if isinstance(prompt, TextPrompt):
        return prompt.prompt
    if isinstance(prompt, TemplatedPrompt):
        return prompt.prompt
    if isinstance(prompt, TokensPrompt):
        return prompt.prompt
    if isinstance(prompt, MultimodalPrompt):
        return None
    raise TypeError(f"unsupported prompt type: {type(prompt).__name__}")


def supplied_prompt_token_ids(prompt: PromptInput) -> tuple[int, ...] | None:
    """Return caller-owned IDs, including a token-based multimodal prompt."""

    if isinstance(prompt, TokensPrompt):
        return prompt.prompt_token_ids
    if isinstance(prompt, MultimodalPrompt) and isinstance(prompt.base, TokensPrompt):
        return prompt.base.prompt_token_ids
    if type(prompt) is str or isinstance(
        prompt,
        (TextPrompt, TemplatedPrompt, MultimodalPrompt),
    ):
        return None
    raise TypeError(f"unsupported prompt type: {type(prompt).__name__}")


def token_prompt_fingerprint(prompt: TokensPrompt) -> str:
    """Hash only exact caller-supplied IDs using an unambiguous v1 format."""

    if not isinstance(prompt, TokensPrompt):
        raise TypeError("token_prompt_fingerprint requires a TokensPrompt")
    digest = hashlib.sha256()
    digest.update(_TOKEN_FINGERPRINT_DOMAIN)
    for token_id in prompt.prompt_token_ids:
        digest.update(token_id.to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def prompt_to_wire(prompt: PromptInput) -> PromptWire:
    """Serialize a prompt without changing legacy string representation."""

    if type(prompt) is str:
        return prompt
    if isinstance(prompt, TextPrompt):
        return {"kind": "text", "prompt": prompt.prompt}
    if isinstance(prompt, TemplatedPrompt):
        return {"kind": "templated_text", "prompt": prompt.prompt}
    if isinstance(prompt, TokensPrompt):
        return {
            "kind": "tokens",
            "prompt_token_ids": list(prompt.prompt_token_ids),
            "prompt": prompt.prompt,
        }
    if isinstance(prompt, MultimodalPrompt):
        wire: dict[str, object] = {
            "kind": "multimodal",
            "base": prompt_to_wire(prompt.base),
            "items": [
                {
                    "modality": item.modality,
                    "encoding": item.encoding,
                    "data": item.data,
                }
                for item in prompt.items
            ],
        }
        if prompt.messages:
            wire["messages"] = [
                {
                    "role": message.role,
                    "content": [
                        (
                            {"type": "text", "text": part.text}
                            if part.type == "text"
                            else {
                                "type": "item",
                                "item_index": part.item_index,
                                **(
                                    {"detail": part.detail}
                                    if part.detail is not None
                                    else {}
                                ),
                            }
                        )
                        for part in message.content
                    ],
                }
                for message in prompt.messages
            ]
        return wire
    raise TypeError(f"unsupported prompt type: {type(prompt).__name__}")


def _require_exact_fields(
    value: dict[object, object],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    keys = set(value)
    missing = expected - keys
    extra = keys - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected fields: {sorted(extra, key=repr)!r}")
        raise ValueError(f"invalid {context}: {', '.join(details)}")


def prompt_from_wire(value: object) -> PromptInput:
    """Decode the strict, tagged prompt wire representation."""

    if type(value) is str:
        return value
    if type(value) is not dict:
        raise TypeError("wire prompt must be a string or tagged dictionary")

    kind = value.get("kind")
    if kind == "text":
        _require_exact_fields(value, frozenset({"kind", "prompt"}), context="text prompt")
        return TextPrompt(value["prompt"])
    if kind == "templated_text":
        _require_exact_fields(
            value,
            frozenset({"kind", "prompt"}),
            context="templated text prompt",
        )
        return TemplatedPrompt(value["prompt"])
    if kind == "tokens":
        _require_exact_fields(
            value,
            frozenset({"kind", "prompt_token_ids", "prompt"}),
            context="tokens prompt",
        )
        token_ids = value["prompt_token_ids"]
        if type(token_ids) is not list:
            raise TypeError("tokens prompt_token_ids wire field must be a list")
        return TokensPrompt(token_ids, value["prompt"])
    if kind == "multimodal":
        expected = {"kind", "base", "items"}
        if "messages" in value:
            expected.add("messages")
        _require_exact_fields(
            value,
            frozenset(expected),
            context="multimodal prompt",
        )
        item_values = value["items"]
        if type(item_values) is not list:
            raise TypeError("multimodal items wire field must be a list")
        items: list[MultimodalItem] = []
        for index, item_value in enumerate(item_values):
            if type(item_value) is not dict:
                raise TypeError(f"multimodal item {index} must be a dictionary")
            _require_exact_fields(
                item_value,
                frozenset({"modality", "encoding", "data"}),
                context=f"multimodal item {index}",
            )
            items.append(
                MultimodalItem(
                    modality=item_value["modality"],
                    encoding=item_value["encoding"],
                    data=item_value["data"],
                )
            )
        messages: list[MultimodalMessage] = []
        message_values = value.get("messages", [])
        if type(message_values) is not list:
            raise TypeError("multimodal messages wire field must be a list")
        for message_index, message_value in enumerate(message_values):
            if type(message_value) is not dict:
                raise TypeError(
                    f"multimodal message {message_index} must be a dictionary"
                )
            _require_exact_fields(
                message_value,
                frozenset({"role", "content"}),
                context=f"multimodal message {message_index}",
            )
            content_values = message_value["content"]
            if type(content_values) is not list:
                raise TypeError(
                    f"multimodal message {message_index} content must be a list"
                )
            parts: list[MultimodalMessagePart] = []
            for part_index, part_value in enumerate(content_values):
                if type(part_value) is not dict:
                    raise TypeError(
                        "multimodal message "
                        f"{message_index} part {part_index} must be a dictionary"
                    )
                part_type = part_value.get("type")
                if part_type == "text":
                    _require_exact_fields(
                        part_value,
                        frozenset({"type", "text"}),
                        context=(
                            f"multimodal message {message_index} "
                            f"text part {part_index}"
                        ),
                    )
                    parts.append(
                        MultimodalMessagePart(type="text", text=part_value["text"])
                    )
                elif part_type == "item":
                    expected_item_fields = {"type", "item_index"}
                    if "detail" in part_value:
                        expected_item_fields.add("detail")
                    _require_exact_fields(
                        part_value,
                        frozenset(expected_item_fields),
                        context=(
                            f"multimodal message {message_index} "
                            f"item part {part_index}"
                        ),
                    )
                    parts.append(
                        MultimodalMessagePart(
                            type="item",
                            item_index=part_value["item_index"],
                            detail=part_value.get("detail"),
                        )
                    )
                else:
                    raise ValueError(
                        "multimodal message "
                        f"{message_index} part {part_index} has unknown type "
                        f"{part_type!r}"
                    )
            messages.append(
                MultimodalMessage(role=message_value["role"], content=parts)
            )
        return MultimodalPrompt(
            base=prompt_from_wire(value["base"]),
            items=items,
            messages=messages,
        )

    if "kind" not in value:
        raise ValueError("tagged wire prompt is missing required field: 'kind'")
    raise ValueError(f"unknown wire prompt kind: {kind!r}")


__all__ = [
    "MultimodalEncoding",
    "MultimodalItem",
    "MultimodalMessage",
    "MultimodalMessagePart",
    "MultimodalPrompt",
    "PromptInput",
    "PromptKind",
    "PromptWire",
    "TemplatedPrompt",
    "TextPrompt",
    "TokensPrompt",
    "prompt_from_wire",
    "prompt_kind",
    "prompt_text",
    "prompt_to_wire",
    "supplied_prompt_token_ids",
    "token_prompt_fingerprint",
]
