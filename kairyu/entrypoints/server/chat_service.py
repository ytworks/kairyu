"""Transport-neutral validation and buffered execution for chat requests."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    UpstreamClientError,
    validate_backend_request,
)
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalMessage,
    MultimodalMessagePart,
    MultimodalPrompt,
    PromptInput,
    TemplatedPrompt,
    TextPrompt,
)
from kairyu.entrypoints.chat_template import ChatTemplate, flatten_content, render_chat
from kairyu.entrypoints.server.metering import resolve_usage_counts
from kairyu.entrypoints.server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceLogprobs,
    FunctionCall,
    LogprobEntry,
    PromptTokensDetails,
    ResponseMessage,
    ToolCall,
    TopLogprobEntry,
    Usage,
)
from kairyu.outputs import CompletionOutput, TokenLogprob
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    SamplingParams,
)

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN_FUNCTION_PATTERN = re.compile(
    r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL
)
_QWEN_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL
)
_LLAMA_TOOL_PREFIX = "<|python_tag|>"
logger = logging.getLogger(__name__)


class ChatRequestError(Exception):
    """A controlled request-boundary failure safe to return to a tenant."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "invalid_request",
        error_type: str = "invalid_request_error",
        execution: ExecutedChat | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.error_type = error_type
        self.execution = execution

    def payload(self) -> dict:
        return {
            "message": str(self),
            "type": self.error_type,
            "code": self.code,
        }


@dataclass(frozen=True)
class NormalizedToolChoice:
    mode: str
    allowed_names: frozenset[str]
    named: str | None = None


@dataclass(frozen=True)
class ValidatedChatInput:
    request: ChatCompletionRequest
    prompt: PromptInput
    normalized_tool_choice: NormalizedToolChoice
    tools_in_prompt: bool
    include_usage: bool


@dataclass(frozen=True)
class ValidatedChatRequest:
    input: ValidatedChatInput
    engine: EngineBackend
    generation_request: GenerationRequest


@dataclass(frozen=True)
class ExecutedChat:
    response: ChatCompletionResponse
    result: GenerationResult


def sampling_params_from(request: ChatCompletionRequest) -> SamplingParams:
    extra_args = dict(request.extra_args or {})
    if "response_format" in extra_args:
        raise ValueError(
            "extra_args.response_format is reserved; use the top-level response_format field"
        )
    if request.response_format:
        extra_args["response_format"] = request.response_format
    logprobs = None
    if request.logprobs:
        logprobs = request.top_logprobs or 0
    max_tokens = (
        request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
    )
    # Tenant compute admission needs a finite bound before dispatch.  Sixteen
    # is already SamplingParams/Kairyu's historical default; materialize it so
    # remote OpenAI-compatible backends cannot substitute an unbounded default.
    if max_tokens is None:
        max_tokens = 16
    params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        n=request.n,
        best_of=request.best_of,
        max_tokens=max_tokens,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty,
        stop=request.stop,
        stop_token_ids=request.stop_token_ids,
        min_tokens=request.min_tokens,
        ignore_eos=request.ignore_eos,
        seed=request.seed,
        logprobs=logprobs,
        prompt_logprobs=request.prompt_logprobs,
        skip_special_tokens=request.skip_special_tokens,
        extra_args=extra_args,
    )
    omitted = {
        name
        for name in GENERATION_CONFIG_SAMPLING_FIELDS
        if name not in request.model_fields_set
    }
    return params.with_generation_config_omitted(omitted)


def _normalize_tool_choice(request: ChatCompletionRequest) -> NormalizedToolChoice:
    allowed_names: set[str] = set()
    for index, tool in enumerate(request.tools or []):
        if tool.get("type") != "function":
            raise ChatRequestError(f"tools[{index}].type must be 'function'")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ChatRequestError(f"tools[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ChatRequestError(f"tools[{index}].function.name must be a non-empty string")
        if name in allowed_names:
            raise ChatRequestError(f"tools[{index}].function.name {name!r} is duplicated")
        allowed_names.add(name)

    choice = request.tool_choice
    if choice is None:
        return NormalizedToolChoice("auto", frozenset(allowed_names))
    if isinstance(choice, str):
        if choice not in {"auto", "none", "required"}:
            raise ChatRequestError("tool_choice must be 'auto', 'none', 'required', or a function")
        if choice == "required" and not allowed_names:
            raise ChatRequestError("tool_choice 'required' requires at least one tool")
        return NormalizedToolChoice(choice, frozenset(allowed_names))
    if choice.get("type") != "function":
        raise ChatRequestError("named tool_choice.type must be 'function'")
    function = choice.get("function")
    if not isinstance(function, dict):
        raise ChatRequestError("named tool_choice.function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ChatRequestError("named tool_choice.function.name must be a non-empty string")
    if name not in allowed_names:
        raise ChatRequestError(f"named tool_choice function {name!r} is not declared in tools")
    return NormalizedToolChoice("named", frozenset(allowed_names), named=name)


def _validate_response_format(response_format: dict | None) -> None:
    if response_format is None:
        return
    kind = response_format.get("type")
    if kind not in ("text", "json_object", "json_schema"):
        raise ChatRequestError(
            f"response_format.type must be text, json_object or json_schema, got {kind!r}"
        )
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise ChatRequestError(
                "response_format.json_schema.schema must be a JSON schema object"
            )


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
    messages = []
    for message in request.messages:
        # Preserve the OpenAI wire shape for key-sensitive HF templates:
        # omitted optional fields stay undefined, while an explicit null stays
        # present. Pydantic extras remain included as sent.
        messages.append(message.model_dump(exclude_unset=True))
    if (
        request.parallel_tool_calls is False
        and request.tools
        and request.tool_choice != "none"
    ):
        messages.insert(
            0,
            {
                "role": "system",
                "content": "Call at most one function in this response.",
            },
        )
    tools = None if request.tool_choice == "none" else request.tools
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
        return render_chat(messages)
    try:
        return TemplatedPrompt(
            template.render(
                messages,
                tools=tools,
                template_kwargs=request.chat_template_kwargs,
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
) -> MultimodalPrompt:
    """Preserve roles and content-part order for a chat-capable VLM backend.

    The remote VLM owns its Hugging Face processor/chat template. Rendering
    this request through Kairyu's text-only Jinja path first would collapse the
    structured image parts and apply the model template twice.
    """

    if request.chat_template_kwargs:
        raise ChatRequestError(
            "chat_template_kwargs cannot be applied to an upstream-owned "
            "multimodal chat template"
        )
    if (chat_templates or {}).get(request.model) is not None:
        raise ChatRequestError(
            f"model {request.model!r} cannot combine a Kairyu text chat template "
            "with image input; the VLM backend owns multimodal templating"
        )

    items: list[MultimodalItem] = []
    messages: list[MultimodalMessage] = []
    display_lines: list[str] = []
    for message_index, message in enumerate(request.messages):
        if (
            message.name is not None
            or message.tool_call_id is not None
            or message.tool_calls is not None
        ):
            raise ChatRequestError(
                f"messages[{message_index}] tool transcript fields are not "
                "supported together with image input"
            )
        content = message.content
        parts: list[MultimodalMessagePart] = []
        display: list[str] = []
        if isinstance(content, str):
            parts.append(MultimodalMessagePart("text", text=content))
            display.append(content)
        elif isinstance(content, list):
            if not content:
                raise ChatRequestError(
                    f"messages[{message_index}].content must not be an empty part list"
                )
            for part in content:
                if part.type == "text":
                    assert part.text is not None
                    parts.append(MultimodalMessagePart("text", text=part.text))
                    display.append(part.text)
                    continue
                assert part.image_url is not None
                item_index = len(items)
                items.append(
                    MultimodalItem(
                        modality="image",
                        encoding="uri",
                        data=part.image_url.url,
                    )
                )
                parts.append(
                    MultimodalMessagePart(
                        "item",
                        item_index=item_index,
                        detail=part.image_url.detail,
                    )
                )
                display.append(f"<image:{item_index}>")
        else:
            raise ChatRequestError(
                f"messages[{message_index}].content must contain text or image parts"
            )
        messages.append(MultimodalMessage(message.role, parts))
        display_lines.append(f"{message.role}: {''.join(display)}")

    if not items:  # Defensive: the caller detects images before entering.
        raise ChatRequestError("multimodal chat input must contain at least one image")
    return MultimodalPrompt(
        base=TextPrompt("\n".join(display_lines)),
        items=items,
        messages=messages,
    )


def validate_chat_input(
    request: ChatCompletionRequest,
    chat_templates: Mapping[str, ChatTemplate] | None,
    *,
    allow_multimodal: bool = False,
    legacy_chat_models: AbstractSet[str] | None = None,
) -> ValidatedChatInput:
    if request.model_extra:
        raise ChatRequestError(
            "unsupported request fields: " + ", ".join(sorted(request.model_extra))
        )
    normalized_tool_choice = _normalize_tool_choice(request)
    if request.stream_options is not None and not request.stream:
        raise ChatRequestError("stream_options is only allowed when stream is true")
    if request.top_logprobs is not None and not request.logprobs:
        raise ChatRequestError("top_logprobs requires logprobs to be true")
    if request.top_logprobs is not None and not 0 <= request.top_logprobs <= 20:
        raise ChatRequestError("top_logprobs must be between 0 and 20")
    _validate_response_format(request.response_format)
    has_images = False
    for index, message in enumerate(request.messages):
        if not message.role.strip():
            raise ChatRequestError(
                f"messages[{index}].role must be a non-empty string"
            )
        if message.model_extra:
            raise ChatRequestError(
                f"messages[{index}] has unsupported fields: "
                + ", ".join(sorted(message.model_extra))
            )
        _, message_has_images = flatten_content(message.content)
        has_images = has_images or message_has_images
    if has_images and not allow_multimodal:
        raise ChatRequestError(f"model {request.model!r} does not support image inputs")
    prompt: PromptInput = (
        _render_multimodal_prompt(request, chat_templates)
        if has_images
        else render_prompt(
            request,
            chat_templates,
            legacy_chat_models=legacy_chat_models,
        )
    )
    return ValidatedChatInput(
        request=request,
        prompt=prompt,
        normalized_tool_choice=normalized_tool_choice,
        tools_in_prompt=bool(
            (chat_templates or {}).get(request.model)
            and request.tools
            and request.tool_choice != "none"
        ),
        include_usage=bool(request.stream_options and request.stream_options.include_usage),
    )


def validate_chat_request(
    request: ChatCompletionRequest,
    engines: Mapping[str, EngineBackend],
    chat_templates: Mapping[str, ChatTemplate] | None,
    *,
    request_id: str,
    cache_hint: CacheHint | None = None,
    priority: int | None = None,
    scheduling_class: str = "interactive",
    placement_started_ns: int | None = None,
    trace_requested: bool = False,
    legacy_chat_models: AbstractSet[str] | None = None,
) -> ValidatedChatRequest:
    engine = engines.get(request.model)
    if engine is None:
        raise ChatRequestError(
            f"model {request.model!r} not found",
            status_code=404,
            code="model_not_found",
        )
    validated_input = validate_chat_input(
        request,
        chat_templates,
        allow_multimodal=True,
        legacy_chat_models=legacy_chat_models,
    )
    if request.n > 1 and getattr(engine, "supports_n", True) is False:
        raise ChatRequestError(f"model {request.model!r} does not support n > 1")
    try:
        sampling = sampling_params_from(request)
    except ValueError as error:
        raise ChatRequestError(str(error)) from error
    generation_request = GenerationRequest(
        request_id=request_id,
        prompt=validated_input.prompt,
        sampling_params=sampling,
        placement_started_ns=placement_started_ns,
        trace_requested=trace_requested,
        priority=request.priority if priority is None else priority,
        scheduling_class=scheduling_class,
        cache_hint=cache_hint,
        tools=tuple(request.tools or ()),
        tool_choice=request.tool_choice,
        tools_in_prompt=validated_input.tools_in_prompt,
    )
    try:
        validate_backend_request(engine, generation_request)
    except ValueError as error:
        raise ChatRequestError(
            str(error),
            code=getattr(error, "code", "invalid_request"),
        ) from error
    return ValidatedChatRequest(
        input=validated_input,
        engine=engine,
        generation_request=generation_request,
    )


async def execute_chat(validated: ValidatedChatRequest) -> ExecutedChat:
    try:
        result = await validated.engine.generate(validated.generation_request)
    except UpstreamClientError as error:
        raise chat_error_from_upstream_client_error(error) from error
    response = completion_response(
        validated.input.request,
        validated.input.prompt,
        result.completions,
        result.usage,
        normalized_tool_choice=validated.input.normalized_tool_choice,
    )
    execution = ExecutedChat(response=response, result=result)
    if not tool_choice_is_satisfied(response.choices, validated.input.normalized_tool_choice):
        raise ChatRequestError(
            "upstream model did not satisfy tool_choice",
            status_code=502,
            code="tool_choice_not_satisfied",
            error_type="upstream_error",
            execution=execution,
        )
    if not parallel_tool_calls_is_satisfied(
        response.choices,
        validated.input.request.parallel_tool_calls,
    ):
        raise ChatRequestError(
            "upstream model emitted multiple calls while parallel_tool_calls=false",
            status_code=502,
            code="parallel_tool_calls_not_satisfied",
            error_type="upstream_error",
            execution=execution,
        )
    return execution


def chat_error_from_upstream_client_error(
    error: UpstreamClientError,
) -> ChatRequestError:
    """Translate a backend 4xx without exposing arbitrary upstream text."""

    if error.public_message is None:
        logger.warning(
            "OpenAI-compatible upstream rejected a request",
            exc_info=error,
        )
        return ChatRequestError(
            "upstream backend rejected the request",
            status_code=502,
            code="backend_error",
            error_type="upstream_error",
        )
    return ChatRequestError(
        error.public_message,
        status_code=error.status_code,
        code=getattr(error, "code", "invalid_request"),
    )


def _tool_call_from_payload(payload: object) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = payload.get("arguments", payload.get("parameters", {}))
    if not isinstance(arguments, (dict, str)):
        return None
    if isinstance(arguments, dict):
        try:
            serialized_arguments = json.dumps(arguments)
        except (ValueError, RecursionError):
            return None
    else:
        serialized_arguments = arguments
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:12]}",
        function=FunctionCall(name=name, arguments=serialized_arguments),
    )


def _tool_parameter_schemas(
    tools: Sequence[Mapping[str, object]],
    function_name: str,
) -> Mapping[str, object]:
    for tool in tools:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, Mapping):
            continue
        if function.get("name") != function_name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            return {}
        properties = parameters.get("properties")
        return properties if isinstance(properties, Mapping) else {}
    return {}


def _qwen_parameter_value(value: str, schema: object) -> object:
    # Qwen3-Coder puts one formatting newline on each side of parameter values.
    # Remove only those protocol newlines so intentional spaces/newlines survive.
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    if value.lower() == "null":
        return None
    if not isinstance(schema, Mapping):
        return value
    kind = schema.get("type")
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean":
        normalized = value.lower()
        if normalized not in {"true", "false"}:
            raise ValueError("invalid boolean tool parameter")
        return normalized == "true"
    if kind in {"array", "object"}:
        parsed = json.loads(value)
        if kind == "array" and not isinstance(parsed, list):
            raise ValueError("tool parameter does not match array schema")
        if kind == "object" and not isinstance(parsed, dict):
            raise ValueError("tool parameter does not match object schema")
        return parsed
    return value


def _qwen_tool_calls(
    payload: str,
    tools: Sequence[Mapping[str, object]],
) -> list[ToolCall]:
    calls = []
    for function_match in _QWEN_FUNCTION_PATTERN.finditer(payload):
        name = function_match.group(1).strip()
        schemas = _tool_parameter_schemas(tools, name)
        arguments: dict[str, object] = {}
        try:
            for parameter_match in _QWEN_PARAMETER_PATTERN.finditer(
                function_match.group(2)
            ):
                parameter_name = parameter_match.group(1).strip()
                if not parameter_name:
                    raise ValueError("empty tool parameter name")
                arguments[parameter_name] = _qwen_parameter_value(
                    parameter_match.group(2), schemas.get(parameter_name)
                )
        except (TypeError, ValueError, RecursionError):
            continue
        call = _tool_call_from_payload({"name": name, "arguments": arguments})
        if call is not None:
            calls.append(call)
    return calls


def _parse_tool_calls(
    text: str,
    tools: Sequence[Mapping[str, object]] = (),
) -> list[ToolCall]:
    calls = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except (ValueError, RecursionError):
            calls.extend(_qwen_tool_calls(match.group(1), tools))
            continue
        call = _tool_call_from_payload(payload)
        if call is not None:
            calls.append(call)
    if calls:
        return calls

    # Llama 3.1's tokenizer template asks for one bare
    # {"name": ..., "parameters": ...} object. Depending on tokenizer decode
    # settings, the leading python-tag special token may still be present.
    candidate = text.strip()
    if candidate.startswith(_LLAMA_TOOL_PREFIX):
        candidate = candidate[len(_LLAMA_TOOL_PREFIX) :].lstrip()
    try:
        payload = json.loads(candidate)
    except (ValueError, RecursionError):
        return []
    call = _tool_call_from_payload(payload)
    return [call] if call is not None else []


def _logprob_entries(content: tuple[TokenLogprob, ...]) -> list[LogprobEntry]:
    return [
        LogprobEntry(
            token=entry.token,
            logprob=entry.logprob,
            bytes=list(entry.bytes_) if entry.bytes_ is not None else None,
            top_logprobs=[
                TopLogprobEntry(
                    token=top.token,
                    logprob=top.logprob,
                    bytes=list(top.bytes_) if top.bytes_ is not None else None,
                )
                for top in entry.top
            ],
        )
        for entry in content
    ]


def _choice_logprobs(completion: CompletionOutput) -> ChoiceLogprobs | None:
    if completion.logprob_content is None:
        return None
    return ChoiceLogprobs(content=_logprob_entries(completion.logprob_content))


def _build_choice(
    index: int,
    text: str,
    tool_choice: NormalizedToolChoice,
    finish_reason: str | None,
    logprobs: ChoiceLogprobs | None = None,
    tools: Sequence[Mapping[str, object]] = (),
) -> Choice:
    tool_calls = []
    if tool_choice.mode != "none":
        tool_calls = [
            call
            for call in _parse_tool_calls(text, tools)
            if call.function.name in tool_choice.allowed_names
            and (tool_choice.named is None or call.function.name == tool_choice.named)
        ]
    if tool_calls:
        return Choice(
            index=index,
            message=ResponseMessage(content=None, tool_calls=tool_calls),
            finish_reason="tool_calls",
            logprobs=logprobs,
        )
    return Choice(
        index=index,
        message=ResponseMessage(content=text),
        finish_reason="stop" if finish_reason == "tool_calls" else finish_reason or "stop",
        logprobs=logprobs,
    )


def _wire_usage(
    prompt: PromptInput,
    completions: Sequence[CompletionOutput],
    usage: GenerationUsage | Usage | None,
) -> Usage:
    prompt_tokens, completion_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    if isinstance(usage, GenerationUsage):
        details = (
            PromptTokensDetails(cached_tokens=usage.cached_tokens) if usage.cached_tokens else None
        )
    elif isinstance(usage, Usage):
        details = usage.prompt_tokens_details
    else:
        details = None
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=details,
    )


def completion_response(
    request: ChatCompletionRequest,
    prompt: PromptInput,
    completions: Sequence[CompletionOutput],
    usage: GenerationUsage | None = None,
    normalized_tool_choice: NormalizedToolChoice | None = None,
) -> ChatCompletionResponse:
    if normalized_tool_choice is None:
        normalized_tool_choice = _normalize_tool_choice(request)
    choices = [
        _build_choice(
            completion.index,
            completion.text,
            normalized_tool_choice,
            completion.finish_reason,
            _choice_logprobs(completion),
            request.tools or (),
        )
        for completion in completions
    ]
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
        created=int(time.time()),
        model=request.model,
        choices=choices,
        usage=_wire_usage(prompt, completions, usage),
    )


def tool_choice_is_satisfied(choices: Sequence[Choice], tool_choice: NormalizedToolChoice) -> bool:
    if tool_choice.mode not in {"required", "named"}:
        return True
    return bool(choices) and all(choice.message.tool_calls for choice in choices)


def parallel_tool_calls_is_satisfied(
    choices: Sequence[Choice],
    parallel_tool_calls: bool | None,
) -> bool:
    """Enforce the limit per alternative; ``n`` choices are not parallel calls."""

    if parallel_tool_calls is not False:
        return True
    return all(len(choice.message.tool_calls or ()) <= 1 for choice in choices)
