"""Chat rendering policy and the template, legacy and multimodal prompt renderers.

Every renderer consumes the one prepared snapshot from ``chat_prepare``.
``render_prompt`` renders identically for HTTP and batch transports, and
``chat_service`` composes the renderers into chat request validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalMessage,
    MultimodalMessagePart,
    MultimodalPrompt,
    TemplatedPrompt,
    TextPrompt,
)
from kairyu.entrypoints.chat_template import (
    ChatTemplate,
    render_chat,
    validate_upstream_chat_template_kwargs,
)
from kairyu.entrypoints.server.chat_errors import ChatRequestError
from kairyu.entrypoints.server.chat_prepare import (
    _SINGLE_TOOL_CONSTRAINT,
    _prepare_chat_messages,
    _PreparedChatMessages,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest
from kairyu.sampling_params import PROMPT_CARRIER_EXTRA_ARGS, resolve_parallel_tool_calls


def validate_chat_policy(
    chat_templates: Mapping[str, ChatTemplate] | None,
    legacy_chat_models: AbstractSet[str] | None = None,
) -> None:
    """Reject ambiguous or unverifiable chat rendering configuration."""

    templates = chat_templates or {}
    overlap = set(templates) & set(legacy_chat_models or ())
    if overlap:
        raise ValueError(
            "models cannot be configured in both chat_templates and "
            f"legacy_chat_models: {sorted(overlap)}"
        )

    invalid = {
        model: sorted(template.unverified_special_token_variables)
        for model, template in templates.items()
        if template.unverified_special_token_variables
    }
    if invalid:
        detail = "; ".join(
            f"{model}={variables}" for model, variables in sorted(invalid.items())
        )
        raise ValueError(
            "chat_templates reference tokenizer-owned special-token variables "
            f"without verified values: {detail}; supply special_tokens when "
            "constructing each explicit ChatTemplate or use checkpoint-owned "
            "tokenizer metadata"
        )


def _add_single_tool_constraint(
    prepared: _PreparedChatMessages,
    *,
    insert_if_missing: bool,
) -> tuple[list[dict[str, object]], bool]:
    """Copy prepared text messages and preserve shape-sensitive hint merging."""

    messages = [dict(message.text_message) for message in prepared.messages]
    for index, message in enumerate(prepared.messages):
        if message.role != "system":
            continue
        content = messages[index].get("content")
        if message.content_kind == "string":
            assert isinstance(content, str)
            if _SINGLE_TOOL_CONSTRAINT not in content:
                separator = "\n\n" if content else ""
                messages[index]["content"] = (
                    content + separator + _SINGLE_TOOL_CONSTRAINT
                )
            return messages, False
        if message.content_kind == "none":
            messages[index]["content"] = _SINGLE_TOOL_CONSTRAINT
            return messages, False
        if message.content_kind == "list":
            if not message.contains_single_tool_constraint_part:
                assert isinstance(content, str)
                messages[index]["content"] = content + _SINGLE_TOOL_CONSTRAINT
            return messages, False
        raise ChatRequestError("system message content has an unsupported shape")
    if insert_if_missing:
        messages.insert(
            0,
            {"role": "system", "content": _SINGLE_TOOL_CONSTRAINT},
        )
        return messages, True
    return messages, False


def _reject_prepared_images_in_text_renderer(
    prepared: _PreparedChatMessages,
    messages: Sequence[Mapping[str, object]],
    *,
    inserted_system_message: bool,
    renderer: str,
) -> None:
    """Keep image/carrier failure ordering after raw lists are flattened."""

    if not prepared.has_images:
        return
    offset = 1 if inserted_system_message else 0
    for original_index, message in enumerate(prepared.messages):
        rendered_index = original_index + offset
        carriers = PROMPT_CARRIER_EXTRA_ARGS.intersection(
            messages[rendered_index]
        )
        if carriers:
            raise ValueError(
                f"message {rendered_index} contains alternate prompt carriers "
                f"{sorted(carriers)}; pass input through PromptInput"
            )
        if message.has_images:
            raise ValueError(
                f"message {rendered_index} contains image input that a text chat "
                f"{renderer} cannot execute; use MultimodalPrompt"
            )


def _resolved_parallel_tool_calls(
    request: ChatCompletionRequest,
) -> bool | None:
    try:
        return resolve_parallel_tool_calls(
            request.parallel_tool_calls,
            request.extra_args or {},
        )
    except ValueError as error:
        raise ChatRequestError(str(error)) from error


def render_prompt(
    request: ChatCompletionRequest,
    chat_templates: Mapping[str, ChatTemplate] | None,
    *,
    legacy_chat_models: AbstractSet[str] | None = None,
) -> str | TemplatedPrompt:
    """Render one prompt identically for HTTP and batch transports."""
    try:
        validate_chat_policy(chat_templates, legacy_chat_models)
    except ValueError as error:
        raise ChatRequestError(str(error)) from error
    template = (chat_templates or {}).get(request.model)
    try:
        prepared = _prepare_chat_messages(
            request,
            validate_message_fields=False,
        )
    except Exception as error:
        # ChatTemplate.render is a request boundary and historically wrapped
        # malformed content there. The legacy renderer continues to expose its
        # ValueError directly to its existing callers.
        if template is not None:
            raise ChatRequestError(str(error)) from error
        raise
    return _render_prepared_prompt(
        request,
        chat_templates,
        prepared,
        legacy_chat_models=legacy_chat_models,
    )


def _render_prepared_prompt(
    request: ChatCompletionRequest,
    chat_templates: Mapping[str, ChatTemplate] | None,
    prepared: _PreparedChatMessages,
    *,
    legacy_chat_models: AbstractSet[str] | None,
) -> str | TemplatedPrompt:
    """Render text from a snapshot whose raw content was already consumed."""

    template = (chat_templates or {}).get(request.model)
    messages = [dict(message.text_message) for message in prepared.messages]
    inserted_system_message = False
    if (
        _resolved_parallel_tool_calls(request) is False
        and request.tools
        and request.tool_choice != "none"
    ):
        messages, inserted_system_message = _add_single_tool_constraint(
            prepared,
            insert_if_missing=template is None,
        )
    tools = None if request.tool_choice == "none" else request.tools
    template_kwargs = dict(request.chat_template_kwargs or {})
    if request.reasoning_effort is not None:
        template_kwargs.setdefault("reasoning_effort", request.reasoning_effort)
        template_kwargs.setdefault("thinking_mode", "thinking")
    if template is None:
        if request.model not in (legacy_chat_models or ()):
            raise ChatRequestError(
                f"model {request.model!r} has no Kairyu chat template; "
                "configure chat_templates or explicitly opt in through "
                "legacy_chat_models"
            )
        if request.chat_template_kwargs:
            raise ChatRequestError(
                f"model {request.model!r} has no Kairyu chat template; "
                "chat_template_kwargs cannot be applied"
            )
        _reject_prepared_images_in_text_renderer(
            prepared,
            messages,
            inserted_system_message=inserted_system_message,
            renderer="renderer",
        )
        return render_chat(messages)
    try:
        _reject_prepared_images_in_text_renderer(
            prepared,
            messages,
            inserted_system_message=inserted_system_message,
            renderer="template",
        )
        return TemplatedPrompt(
            template.render(
                messages,
                tools=tools,
                template_kwargs=template_kwargs or None,
            )
        )
    # Jinja preserves Python exceptions raised by expressions (for example a
    # TypeError from request-dependent concatenation) instead of wrapping every
    # failure in TemplateError. Rendering is a request-validation boundary: no
    # template/input mismatch should escape as an HTTP or batch 500.
    except Exception as error:
        raise ChatRequestError(str(error)) from error


def _render_multimodal_prompt(
    request: ChatCompletionRequest,
    chat_templates: Mapping[str, ChatTemplate] | None,
    prepared: _PreparedChatMessages,
) -> MultimodalPrompt:
    """Preserve roles and content-part order for a chat-capable VLM backend.

    The remote VLM owns its Hugging Face processor/chat template. Rendering
    this request through Kairyu's text-only Jinja path first would collapse the
    structured image parts and apply the model template twice.
    """

    try:
        validate_upstream_chat_template_kwargs(request.chat_template_kwargs)
    except ValueError as error:
        raise ChatRequestError(str(error)) from error
    if (chat_templates or {}).get(request.model) is not None:
        raise ChatRequestError(
            f"model {request.model!r} cannot combine a Kairyu text chat template "
            "with image input; the VLM backend owns multimodal templating"
        )

    items = [
        MultimodalItem(
            modality="image",
            encoding="uri",
            data=image_url,
        )
        for image_url in prepared.image_urls
    ]
    messages: list[MultimodalMessage] = []
    display_lines: list[str] = []
    for message_index, message in enumerate(prepared.messages):
        if message.has_tool_transcript:
            raise ChatRequestError(
                f"messages[{message_index}] tool transcript fields are not "
                "supported together with image input"
            )
        parts: list[MultimodalMessagePart] = []
        if message.content_kind == "string":
            part = message.content_parts[0]
            assert part.text is not None
            parts.append(MultimodalMessagePart("text", text=part.text))
        elif message.content_kind == "list":
            if not message.content_parts:
                raise ChatRequestError(
                    f"messages[{message_index}].content must not be an empty part list"
                )
            for part in message.content_parts:
                if part.type == "text":
                    assert part.text is not None
                    parts.append(MultimodalMessagePart("text", text=part.text))
                    continue
                assert part.item_index is not None
                parts.append(
                    MultimodalMessagePart(
                        "item",
                        item_index=part.item_index,
                        detail=part.detail,
                    )
                )
        else:
            raise ChatRequestError(
                f"messages[{message_index}].content must contain text or image parts"
            )
        messages.append(MultimodalMessage(message.role, parts))
        display_lines.append(f"{message.role}: {message.display_content}")

    if not items:  # Defensive: the caller detects images before entering.
        raise ChatRequestError("multimodal chat input must contain at least one image")
    return MultimodalPrompt(
        base=TextPrompt("\n".join(display_lines)),
        items=items,
        messages=messages,
    )
