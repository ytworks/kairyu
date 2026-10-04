"""Request-local chat message snapshot shared by every prompt renderer.

``_prepare_chat_messages`` walks each message's raw content exactly once,
validating it and recording the text, image and display views that the
template, legacy and multimodal renderers in ``chat_render`` consume.
"""

from __future__ import annotations

from dataclasses import dataclass

from kairyu.entrypoints.chat_template import _iter_validated_content_parts
from kairyu.entrypoints.server.chat_errors import ChatRequestError
from kairyu.entrypoints.server.protocol import ChatCompletionRequest, ChatMessage

_SINGLE_TOOL_CONSTRAINT = "Call at most one function in this response."


@dataclass(frozen=True)
class _PreparedContentPart:
    """One validated content part retained independently of Pydantic."""

    type: str
    text: str | None = None
    image_url: str | None = None
    detail: object | None = None
    item_index: int | None = None


@dataclass(frozen=True)
class _PreparedMessage:
    """One request-local message snapshot shared by every prompt renderer."""

    text_message: dict[str, object]
    role: str
    content_kind: str
    content_parts: tuple[_PreparedContentPart, ...]
    display_content: str
    has_images: bool
    contains_single_tool_constraint_part: bool
    has_tool_transcript: bool


@dataclass(frozen=True)
class _PreparedChatMessages:
    messages: tuple[_PreparedMessage, ...]
    image_urls: tuple[str, ...]

    @property
    def has_images(self) -> bool:
        return bool(self.image_urls)


def _message_wire_shape(message: ChatMessage) -> dict[str, object]:
    """Copy the fields present on the wire without recursively dumping models."""

    fields_set = message.model_fields_set
    wire = {
        name: getattr(message, name)
        for name in type(message).model_fields
        if name in fields_set
    }
    wire.update(
        {
            name: value
            for name, value in (message.model_extra or {}).items()
            if not _is_ignored_message_extra(message, name, value)
        }
    )
    return wire


def _is_ignored_message_extra(
    message: ChatMessage,
    name: str,
    value: object,
) -> bool:
    if name == "provider_specific_fields":
        return value is None or (
            message.role == "assistant" and isinstance(value, dict)
        )
    return name == "function_call" and message.role == "assistant" and value is None


def _prepare_message_content(
    content: object,
    *,
    image_urls: list[str],
) -> tuple[
    str,
    str,
    tuple[_PreparedContentPart, ...],
    str,
    bool,
    bool,
]:
    """Validate and snapshot content with exactly one walk over its raw parts."""

    if content is None:
        return "", "none", (), "", False, False
    if isinstance(content, str):
        part = _PreparedContentPart("text", text=content)
        return content, "string", (part,), content, False, False

    texts: list[str] = []
    parts: list[_PreparedContentPart] = []
    display: list[str] = []
    has_images = False
    contains_constraint = False
    for kind, text, image_url in _iter_validated_content_parts(content):
        if kind == "text":
            assert text is not None
            texts.append(text)
            display.append(text)
            parts.append(_PreparedContentPart("text", text=text))
            contains_constraint = (
                contains_constraint or _SINGLE_TOOL_CONSTRAINT in text
            )
            continue
        assert image_url is not None
        item_index = len(image_urls)
        image_urls.append(image_url["url"])
        parts.append(
            _PreparedContentPart(
                "image_url",
                image_url=image_url["url"],
                detail=image_url.get("detail"),
                item_index=item_index,
            )
        )
        display.append(f"<image:{item_index}>")
        has_images = True

    return (
        "".join(texts),
        "list" if isinstance(content, list) else "other",
        tuple(parts),
        "".join(display),
        has_images,
        contains_constraint,
    )


def _prepare_chat_messages(
    request: ChatCompletionRequest,
    *,
    validate_message_fields: bool,
) -> _PreparedChatMessages:
    """Take the sole raw-content snapshot used by text and VLM rendering."""

    image_urls: list[str] = []
    prepared: list[_PreparedMessage] = []
    for index, message in enumerate(request.messages):
        if validate_message_fields:
            if not message.role.strip():
                raise ChatRequestError(
                    f"messages[{index}].role must be a non-empty string"
                )
            unsupported_fields = {
                name
                for name, value in (message.model_extra or {}).items()
                if not _is_ignored_message_extra(message, name, value)
            }
            if unsupported_fields:
                raise ChatRequestError(
                    f"messages[{index}] has unsupported fields: "
                    + ", ".join(sorted(unsupported_fields))
                )

        wire = _message_wire_shape(message)
        (
            flattened_content,
            content_kind,
            content_parts,
            display_content,
            has_images,
            contains_constraint,
        ) = _prepare_message_content(message.content, image_urls=image_urls)
        if "content" in wire and content_kind not in {"none", "string"}:
            # ChatTemplate and the legacy renderer now receive the validated
            # flattened value, so neither needs to revisit the raw part list.
            wire["content"] = flattened_content
        prepared.append(
            _PreparedMessage(
                text_message=wire,
                role=message.role,
                content_kind=content_kind,
                content_parts=content_parts,
                display_content=display_content,
                has_images=has_images,
                contains_single_tool_constraint_part=contains_constraint,
                has_tool_transcript=(
                    message.name is not None
                    or message.tool_call_id is not None
                    or message.tool_calls is not None
                ),
            )
        )
    return _PreparedChatMessages(tuple(prepared), tuple(image_urls))
