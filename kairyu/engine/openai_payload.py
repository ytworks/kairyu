"""Request-payload construction for OpenAI-compatible upstreams.

Capability validation maps a ``GenerationRequest`` onto the sampling, tool and
chat-template fields the configured upstream contract can execute, failing
closed with ``OpenAIRequestValidationError``. ``_upstream_payload`` then
assembles the chat or vLLM completions request body around those fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from kairyu.engine.backend import GenerationRequest
from kairyu.engine.openai_capabilities import OpenAIRequestCapabilities
from kairyu.engine.prompt import (
    MultimodalPrompt,
    TemplatedPrompt,
    prompt_kind,
    prompt_text,
)
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    SamplingParams,
    resolve_parallel_tool_calls,
)


class OpenAIRequestValidationError(ValueError):
    """A request cannot be represented by the configured upstream contract."""


def _client_error(upstream: str, message: str) -> OpenAIRequestValidationError:
    return OpenAIRequestValidationError(f"OpenAI-compatible upstream {upstream!r} {message}")


def _active_sampling_fields(params: SamplingParams) -> set[str]:
    """Return non-neutral request intent that an upstream must execute."""

    active: set[str] = set()
    values_and_neutral = {
        "best_of": (params.best_of, None),
        "frequency_penalty": (params.frequency_penalty, 0.0),
        "forced_token_ids": (params.forced_token_ids, None),
        "ignore_eos": (params.ignore_eos, False),
        "logprobs": (params.logprobs, None),
        "max_tokens": (params.max_tokens, None),
        "min_p": (params.min_p, 0.0),
        "min_tokens": (params.min_tokens, 0),
        "n": (params.n, 1),
        "presence_penalty": (params.presence_penalty, 0.0),
        "prompt_logprobs": (params.prompt_logprobs, None),
        "repetition_penalty": (params.repetition_penalty, 1.0),
        "response_format": (params.extra_args.get("response_format"), None)
        if isinstance(params.extra_args, Mapping)
        else (None, None),
        "seed": (params.seed, None),
        "skip_special_tokens": (params.skip_special_tokens, True),
        "stop": (params.stop, ()),
        "stop_token_ids": (params.stop_token_ids, ()),
        "temperature": (params.temperature, 1.0),
        "top_k": (params.top_k, -1),
        "top_p": (params.top_p, 1.0),
    }
    for field, (value, neutral) in values_and_neutral.items():
        if field in GENERATION_CONFIG_SAMPLING_FIELDS:
            if field not in params.generation_config_omitted:
                active.add(field)
        elif value != neutral:
            active.add(field)
    return active


def _validate_sampling_values(
    params: SamplingParams,
    capabilities: OpenAIRequestCapabilities,
) -> None:
    invalid: list[str] = []
    if params.logprobs is not None and params.logprobs < 0:
        invalid.append("logprobs must be non-negative")
    if params.prompt_logprobs is not None and params.prompt_logprobs < 0:
        invalid.append("prompt_logprobs must be non-negative")
    if params.best_of is not None and params.best_of < params.n:
        invalid.append("best_of must be greater than or equal to n")
    if any(token_id < 0 for token_id in params.stop_token_ids):
        invalid.append("stop_token_ids must be non-negative")
    if capabilities.max_n is not None and params.n > capabilities.max_n:
        invalid.append(f"n must be <= {capabilities.max_n}")
    if (
        capabilities.max_temperature is not None
        and params.temperature > capabilities.max_temperature
    ):
        invalid.append(f"temperature must be <= {capabilities.max_temperature}")
    if invalid:
        raise _client_error(capabilities.upstream, "rejects request: " + "; ".join(invalid))


def _sampling_payload(
    params: SamplingParams,
    capabilities: OpenAIRequestCapabilities,
    *,
    validate_json: bool = True,
) -> dict[str, object]:
    extra_args = params.extra_args
    if not isinstance(extra_args, Mapping):
        raise _client_error(capabilities.upstream, "requires extra_args to be a mapping")
    if any(not isinstance(key, str) for key in extra_args):
        raise _client_error(capabilities.upstream, "requires string extra_args keys")

    _validate_sampling_values(params, capabilities)
    active = _active_sampling_fields(params)
    unsupported = active - capabilities.sampling_fields
    if unsupported:
        raise _client_error(
            capabilities.upstream,
            "does not support request fields: " + ", ".join(sorted(unsupported)),
        )

    payload: dict[str, object] = {}
    omitted = params.generation_config_omitted
    always = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "n": params.n,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
    }
    payload.update(
        (field, value)
        for field, value in always.items()
        if field not in omitted
        and field in capabilities.sampling_fields
        and (
            field in active
            or field in capabilities.forward_neutral_fields
            or field in GENERATION_CONFIG_SAMPLING_FIELDS
        )
    )

    optional = {
        "best_of": params.best_of,
        "ignore_eos": params.ignore_eos if params.ignore_eos else None,
        "min_p": params.min_p,
        "min_tokens": params.min_tokens if params.min_tokens != 0 else None,
        "prompt_logprobs": params.prompt_logprobs,
        "repetition_penalty": params.repetition_penalty,
        "seed": params.seed,
        "skip_special_tokens": False if not params.skip_special_tokens else None,
        "stop": list(params.stop) if params.stop else None,
        "stop_token_ids": list(params.stop_token_ids) if params.stop_token_ids else None,
        "top_k": params.top_k,
    }
    payload.update(
        (field, value)
        for field, value in optional.items()
        if field not in omitted
        and value is not None
        and field in capabilities.sampling_fields
    )
    if params.max_tokens is not None and "max_tokens" in capabilities.sampling_fields:
        payload[capabilities.max_tokens_wire_name] = params.max_tokens
    if params.logprobs is not None and "logprobs" in capabilities.sampling_fields:
        payload["logprobs"] = True
        payload["top_logprobs"] = params.logprobs
    response_format = extra_args.get("response_format")
    if response_format is not None and "response_format" in capabilities.sampling_fields:
        payload["response_format"] = response_format

    vendor_args = {key: value for key, value in extra_args.items() if key != "response_format"}
    unapproved = set(vendor_args) - capabilities.extra_args
    if unapproved:
        fields = ", ".join(f"extra_args.{key}" for key in sorted(unapproved))
        raise _client_error(
            capabilities.upstream,
            f"does not allow vendor extension fields: {fields}",
        )
    payload.update(vendor_args)
    if validate_json:
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise _client_error(
                capabilities.upstream,
                f"requires JSON-serializable request fields: {error}",
            ) from error
    return payload


def _validate_tools(
    request: GenerationRequest,
    capabilities: OpenAIRequestCapabilities,
) -> None:
    if capabilities.strict_tools:
        return
    for index, tool in enumerate(request.tools):
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("strict") is True:
            raise _client_error(
                capabilities.upstream,
                f"does not support request field tools[{index}].function.strict",
            )


def _validated_request_payload(
    request: GenerationRequest,
    capabilities: OpenAIRequestCapabilities,
    *,
    validate_json: bool = True,
) -> dict[str, object]:
    kind = prompt_kind(request.prompt)
    if kind not in capabilities.prompt_kinds:
        raise _client_error(
            capabilities.upstream,
            f"does not support prompt kind: {kind}",
        )
    _validate_tools(request, capabilities)
    if request.priority != 0 and not capabilities.priority:
        raise _client_error(
            capabilities.upstream,
            "does not support request field: priority",
        )
    payload = _sampling_payload(
        request.sampling_params,
        capabilities,
        validate_json=validate_json,
    )
    parallel_tool_calls = resolve_parallel_tool_calls(
        request.parallel_tool_calls,
        request.sampling_params.extra_args,
    )
    if (
        request.parallel_tool_calls is not None
        and parallel_tool_calls is not None
        and capabilities.parallel_tool_calls
    ):
        payload["parallel_tool_calls"] = parallel_tool_calls
    if capabilities.priority:
        payload["priority"] = request.priority
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.chat_template_kwargs is not None:
        unsupported = (
            set(request.chat_template_kwargs) - capabilities.chat_template_kwargs
        )
        if unsupported:
            raise _client_error(
                capabilities.upstream,
                "does not support chat_template_kwargs: "
                + ", ".join(sorted(unsupported)),
            )
        payload["chat_template_kwargs"] = dict(request.chat_template_kwargs)
    return payload


def _upstream_payload(
    request: GenerationRequest,
    *,
    model: str,
    use_completions: bool,
    validated: dict[str, object],
    image_urls: tuple[str, ...] | None,
) -> dict:
    """Build the upstream chat or vLLM completions body for one request."""

    prompt = request.prompt
    if isinstance(prompt, MultimodalPrompt):
        if image_urls is None:
            raise RuntimeError(
                "multimodal payload requires fully validated image data"
            )
        messages: list[dict[str, object]] = []
        for message in prompt.messages:
            content: list[dict[str, object]] = []
            for part in message.content:
                if part.type == "text":
                    content.append({"type": "text", "text": part.text})
                    continue
                assert part.item_index is not None
                image_url: dict[str, object] = {
                    "url": image_urls[part.item_index],
                }
                if part.detail is not None:
                    image_url["detail"] = part.detail
                content.append(
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    }
                )
            messages.append({"role": message.role, "content": content})
    else:
        text = prompt_text(prompt)
        assert text is not None
        if use_completions:
            completion_args = dict(validated)
            # These chat-only fields are already represented in Kairyu's
            # rendered DeepSeek prompt and are invalid on /completions.
            completion_args.pop("parallel_tool_calls", None)
            completion_args.pop("reasoning_effort", None)
            if completion_args.get("logprobs") is True:
                completion_args["logprobs"] = completion_args.pop(
                    "top_logprobs", 0
                )
            else:
                completion_args.pop("top_logprobs", None)
            return {
                "model": model,
                "prompt": text,
                **completion_args,
            }
        messages = [{"role": "user", "content": text}]
    if request.assistant_prefill is not None:
        # The upstream template renders this final assistant turn and
        # vLLM continues it instead of opening a new generation prompt.
        messages.append({"role": "assistant", "content": request.assistant_prefill})
    payload: dict = {
        "model": model,
        "messages": messages,
        **validated,
    }
    if request.assistant_prefill is not None:
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
    # A passthrough vLLM replica uses an identity chat template. Kairyu has
    # already rendered tools and reasoning controls into the model-owned
    # template, so duplicating the structured tool fields would create two
    # competing prompt owners.
    if request.tools and not isinstance(prompt, TemplatedPrompt):
        payload["tools"] = list(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
    return payload
