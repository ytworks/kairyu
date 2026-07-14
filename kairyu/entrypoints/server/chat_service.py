"""Transport-neutral validation and buffered execution for chat requests."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kairyu.engine.backend import (
    CacheHint,
    EngineBackend,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
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
from kairyu.sampling_params import SamplingParams

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


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
    prompt: str
    normalized_tool_choice: NormalizedToolChoice
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
    extra_args = (
        {"response_format": request.response_format} if request.response_format else {}
    )
    logprobs = None
    if request.logprobs:
        logprobs = request.top_logprobs or 0
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else request.max_completion_tokens
    )
    return SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        max_tokens=max_tokens,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        stop=request.stop,
        seed=request.seed,
        logprobs=logprobs,
        extra_args=extra_args,
    )


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
            raise ChatRequestError(
                f"tools[{index}].function.name must be a non-empty string"
            )
        if name in allowed_names:
            raise ChatRequestError(
                f"tools[{index}].function.name {name!r} is duplicated"
            )
        allowed_names.add(name)

    choice = request.tool_choice
    if choice is None:
        return NormalizedToolChoice("auto", frozenset(allowed_names))
    if isinstance(choice, str):
        if choice not in {"auto", "none", "required"}:
            raise ChatRequestError(
                "tool_choice must be 'auto', 'none', 'required', or a function"
            )
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
        raise ChatRequestError(
            "named tool_choice.function.name must be a non-empty string"
        )
    if name not in allowed_names:
        raise ChatRequestError(
            f"named tool_choice function {name!r} is not declared in tools"
        )
    return NormalizedToolChoice("named", frozenset(allowed_names), named=name)


def _validate_response_format(response_format: dict | None) -> None:
    if response_format is None:
        return
    kind = response_format.get("type")
    if kind not in ("text", "json_object", "json_schema"):
        raise ChatRequestError(
            "response_format.type must be text, json_object or json_schema, "
            f"got {kind!r}"
        )
    if kind == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise ChatRequestError(
                "response_format.json_schema.schema must be a JSON schema object"
            )


def render_prompt(
    request: ChatCompletionRequest,
    chat_templates: Mapping[str, ChatTemplate] | None,
) -> str:
    """Render one prompt identically for HTTP and batch transports."""
    template = (chat_templates or {}).get(request.model)
    messages = [message.model_dump() for message in request.messages]
    if template is None:
        return render_chat(messages)
    return template.render(messages, tools=request.tools)


def validate_chat_input(
    request: ChatCompletionRequest,
    chat_templates: Mapping[str, ChatTemplate] | None,
) -> ValidatedChatInput:
    normalized_tool_choice = _normalize_tool_choice(request)
    if request.stream_options is not None and not request.stream:
        raise ChatRequestError("stream_options is only allowed when stream is true")
    if request.top_logprobs is not None and not request.logprobs:
        raise ChatRequestError("top_logprobs requires logprobs to be true")
    if request.top_logprobs is not None and not 0 <= request.top_logprobs <= 20:
        raise ChatRequestError("top_logprobs must be between 0 and 20")
    _validate_response_format(request.response_format)
    for message in request.messages:
        _, has_images = flatten_content(message.content)
        if has_images:
            raise ChatRequestError(
                f"model {request.model!r} does not support image inputs"
            )
    return ValidatedChatInput(
        request=request,
        prompt=render_prompt(request, chat_templates),
        normalized_tool_choice=normalized_tool_choice,
        include_usage=bool(
            request.stream_options and request.stream_options.include_usage
        ),
    )


def validate_chat_request(
    request: ChatCompletionRequest,
    engines: Mapping[str, EngineBackend],
    chat_templates: Mapping[str, ChatTemplate] | None,
    *,
    request_id: str,
    cache_hint: CacheHint | None = None,
) -> ValidatedChatRequest:
    validated_input = validate_chat_input(request, chat_templates)
    engine = engines.get(request.model)
    if engine is None:
        raise ChatRequestError(
            f"model {request.model!r} not found",
            status_code=404,
            code="model_not_found",
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
        cache_hint=cache_hint,
    )
    validate = getattr(engine, "validate_request", None)
    if validate is not None:
        try:
            validate(generation_request)
        except ValueError as error:
            raise ChatRequestError(str(error)) from error
    return ValidatedChatRequest(
        input=validated_input,
        engine=engine,
        generation_request=generation_request,
    )


async def execute_chat(validated: ValidatedChatRequest) -> ExecutedChat:
    result = await validated.engine.generate(validated.generation_request)
    response = completion_response(
        validated.input.request,
        validated.input.prompt,
        result.completions,
        result.usage,
        normalized_tool_choice=validated.input.normalized_tool_choice,
    )
    execution = ExecutedChat(response=response, result=result)
    if not _tool_choice_is_satisfied(
        response.choices, validated.input.normalized_tool_choice
    ):
        raise ChatRequestError(
            "upstream model did not satisfy tool_choice",
            status_code=502,
            code="tool_choice_not_satisfied",
            error_type="upstream_error",
            execution=execution,
        )
    return execution


def _parse_tool_calls(text: str) -> list[ToolCall]:
    calls = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except (ValueError, RecursionError):
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, (dict, str)):
            continue
        if isinstance(arguments, dict):
            try:
                serialized_arguments = json.dumps(arguments)
            except (ValueError, RecursionError):
                continue
        else:
            serialized_arguments = arguments
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                function=FunctionCall(name=name, arguments=serialized_arguments),
            )
        )
    return calls


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
) -> Choice:
    tool_calls = []
    if tool_choice.mode != "none":
        tool_calls = [
            call
            for call in _parse_tool_calls(text)
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
    prompt: str,
    completions: Sequence[CompletionOutput],
    usage: GenerationUsage | Usage | None,
) -> Usage:
    prompt_tokens, completion_tokens = resolve_usage_counts(
        usage, prompt=prompt, completions=completions
    )
    if isinstance(usage, GenerationUsage):
        details = (
            PromptTokensDetails(cached_tokens=usage.cached_tokens)
            if usage.cached_tokens
            else None
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
    prompt: str,
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


def _tool_choice_is_satisfied(
    choices: Sequence[Choice], tool_choice: NormalizedToolChoice
) -> bool:
    if tool_choice.mode not in {"required", "named"}:
        return True
    return any(choice.message.tool_calls for choice in choices)
