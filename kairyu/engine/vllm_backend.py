"""vLLM adapter behind the EngineBackend seam (design doc D1).

The module always imports; vLLM itself is imported lazily at instantiation so
Kairyu works on machines where vLLM cannot even be installed. Prefix caching is
enabled by default so orchestration steps sharing a prompt prefix already get
KV hits on the vLLM backend (design doc D5).

The adapter fails closed (m9 D6, M20 WP-30): a request intent it has no vLLM
mapping for is a ``ValueError`` before dispatch, never a silent drop, and every
result carries the engine-reported token usage.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterator, Mapping
from typing import Any

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    prompt_with_tool_intent,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    PromptInput,
    TemplatedPrompt,
    TextPrompt,
    TokensPrompt,
    prompt_text,
    supplied_prompt_token_ids,
)
from kairyu.engine.registry import register_backend
from kairyu.models.generation import GenerationDefaults
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


def _unhonored_sampling_fields(params: SamplingParams) -> tuple[str, ...]:
    """Sampling intents without a vLLM mapping in ``to_vllm_sampling_kwargs``.

    Logprobs would need a ``TokenLogprob`` conversion and ``extra_args`` (the
    ``response_format`` carrier included) a structured-output translation;
    neither exists here, so both are rejected rather than dropped.
    """

    unmapped = tuple(
        name
        for name, value in (
            ("best_of", params.best_of),
            ("logprobs", params.logprobs),
            ("prompt_logprobs", params.prompt_logprobs),
            ("forced_token_ids", params.forced_token_ids),
        )
        if value is not None
    )
    extra_args = params.extra_args
    if not isinstance(extra_args, Mapping):
        return (*unmapped, "extra_args")
    return (*unmapped, *(f"extra_args.{key}" for key in extra_args))


def _unhonored_request_fields(request: GenerationRequest) -> tuple[str, ...]:
    """Every request intent this adapter would otherwise silently drop."""

    unsupported = tuple(
        name
        for name, value in (
            ("chat_template_kwargs", request.chat_template_kwargs),
            ("assistant_prefill", request.assistant_prefill),
        )
        if value is not None
    )
    # Strict tools are a grammar intent; the native engine compiles one and
    # OpenAI-compatible upstreams without strict support reject it.
    strict_tools = tuple(
        f"tools[{index}].function.strict"
        for index, tool in enumerate(request.tools)
        if isinstance(function := tool.get("function"), Mapping)
        and function.get("strict") is True
    )
    return (
        *_unhonored_sampling_fields(request.sampling_params),
        *unsupported,
        *strict_tools,
    )


def _reject_unhonored(fields: tuple[str, ...]) -> None:
    if fields:
        raise ValueError(
            "vLLM backend does not support request fields: " + ", ".join(fields)
        )


def to_vllm_sampling_kwargs(params: SamplingParams) -> dict:
    """Map kairyu SamplingParams to vllm.SamplingParams constructor kwargs.

    Raises ``ValueError`` for a set field this mapping cannot carry.
    """
    _reject_unhonored(_unhonored_sampling_fields(params))
    return {
        "n": params.n,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "seed": params.seed,
        "stop": list(params.stop),
        "stop_token_ids": list(params.stop_token_ids),
        "max_tokens": params.max_tokens,
        "min_tokens": params.min_tokens,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
        "repetition_penalty": params.repetition_penalty,
        "ignore_eos": params.ignore_eos,
        "skip_special_tokens": params.skip_special_tokens,
    }


def _engine_usage(
    output: Any,
    completions: tuple[CompletionOutput, ...],
) -> GenerationUsage | None:
    """Exact usage from one cumulative ``vllm.RequestOutput`` (m9 D1).

    The prompt is counted once and completion tokens are summed across the
    ``n`` completions, as vLLM's own OpenAI server does.
    """

    prompt_token_ids = output.prompt_token_ids
    if prompt_token_ids is None:
        # vLLM omits IDs only for prompt-embedding inputs, never sent here.
        return None
    return GenerationUsage(
        prompt_tokens=len(prompt_token_ids),
        completion_tokens=sum(len(completion.token_ids) for completion in completions),
        cached_tokens=output.num_cached_tokens or 0,
    )


def _import_vllm():
    try:
        return importlib.import_module("vllm")
    except ImportError as error:
        raise RuntimeError(
            "the 'vllm' backend requires vLLM (pip install vllm); "
            "on unsupported platforms use backend='mock' or an 'openai' worker"
        ) from error


def _generation_defaults_from_model_config(model_config: object) -> GenerationDefaults:
    get_sampling = getattr(model_config, "get_diff_sampling_param", None)
    if not callable(get_sampling):
        raise RuntimeError("vLLM backend cannot resolve model generation defaults")
    raw = get_sampling()
    if not isinstance(raw, Mapping):
        raise RuntimeError("vLLM model generation defaults must be a mapping")
    values: dict[str, float | int] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    for name in values:
        if raw.get(name) is not None:
            values[name] = raw[name]
    if values["top_k"] == 0:
        values["top_k"] = -1
    try:
        return GenerationDefaults(
            **values,
            source="vllm-model-config",
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise RuntimeError("vLLM model generation defaults are invalid") from error


class VLLMBackend:
    def __init__(
        self,
        model: str,
        enable_prefix_caching: bool | None = None,
        scheduling_policy: str = "priority",
        **engine_args: object,
    ) -> None:
        if scheduling_policy != "priority":
            raise ValueError(
                "Kairyu's vLLM backend requires scheduling_policy='priority' "
                "so request priority is never silently ignored"
            )
        vllm = _import_vllm()
        args = vllm.AsyncEngineArgs(
            model=model,
            enable_prefix_caching=True if enable_prefix_caching is None else enable_prefix_caching,
            scheduling_policy=scheduling_policy,
            **engine_args,
        )
        configured_sequence_budget = engine_args.get("max_num_seqs")
        self._sequence_budget = (
            configured_sequence_budget
            if type(configured_sequence_budget) is int
            and configured_sequence_budget > 0
            else None
        )
        self._vllm = vllm
        self._engine = vllm.AsyncLLMEngine.from_engine_args(args)
        model_config = getattr(self._engine, "model_config", None)
        self._generation_defaults: GenerationDefaults | None = (
            _generation_defaults_from_model_config(model_config)
            if model_config is not None
            else None
        )

    @property
    def sequence_budget(self) -> int | None:
        """Explicit vLLM active sequence capacity, when configured."""

        return self._sequence_budget

    @property
    def generation_defaults(self) -> GenerationDefaults | None:
        """Resolved vLLM model policy, available for `/backends` auditing."""

        return self._generation_defaults

    async def _resolve_generation_defaults(
        self,
        params: SamplingParams,
    ) -> SamplingParams:
        """Mirror vLLM ``LLM.generate(None)`` over the raw async engine seam."""

        if not params.generation_config_omitted:
            return params
        defaults = self._generation_defaults
        if defaults is None:
            model_config = getattr(self._engine, "model_config", None)
            if model_config is None:
                get_model_config = getattr(self._engine, "get_model_config", None)
                if callable(get_model_config):
                    model_config = get_model_config()
                    if inspect.isawaitable(model_config):
                        model_config = await model_config
            defaults = _generation_defaults_from_model_config(model_config)
            self._generation_defaults = defaults
        return defaults.apply(params)

    @staticmethod
    def _validated_prompt(request: GenerationRequest) -> PromptInput:
        _reject_unhonored(_unhonored_request_fields(request))
        prompt = request.prompt
        if isinstance(prompt, MultimodalPrompt):
            raise ValueError(
                "vLLM backend does not support multimodal prompts through "
                "Kairyu's typed prompt adapter"
            )
        if not isinstance(prompt, (str, TextPrompt, TemplatedPrompt, TokensPrompt)):
            raise ValueError(
                "vLLM backend requires a text or token-ID prompt, "
                f"got {type(prompt).__name__}"
            )
        # Besides rendering text tool intent, this rejects token-ID prompts
        # whose caller did not declare that the tool intent was already rendered.
        return prompt_with_tool_intent(request)

    def validate_request(self, request: GenerationRequest) -> None:
        """Reject every intent and prompt variant this adapter cannot honor.

        As on the native engine, ``tool_choice`` and ``parallel_tool_calls``
        are enforced by the public-boundary tool gate, and ``reasoning_effort``
        reaches the model only through a Kairyu chat template.
        """

        self._validated_prompt(request)

    def _to_result(self, request: GenerationRequest, output) -> GenerationResult:
        completions = tuple(
            CompletionOutput(
                index=completion.index,
                text=completion.text,
                token_ids=tuple(completion.token_ids),
                cumulative_logprob=completion.cumulative_logprob,
                finish_reason=completion.finish_reason,
                stop_reason=completion.stop_reason,
            )
            for completion in output.outputs
        )
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=completions,
            finished=output.finished,
            usage=_engine_usage(output, completions),
            prompt_token_ids=supplied_prompt_token_ids(request.prompt) or (),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        final = None
        async for result in self.stream(request):
            final = result
        if final is None:
            raise RuntimeError(f"vLLM produced no output for request {request.request_id}")
        return final

    async def stream(self, request: GenerationRequest) -> AsyncIterator[GenerationResult]:
        prompt = self._validated_prompt(request)
        if isinstance(prompt, TokensPrompt):
            vllm_prompt: str | dict[str, list[int]] = {
                "prompt_token_ids": list(prompt.prompt_token_ids)
            }
        else:
            text = prompt_text(prompt)
            if text is None:  # Defensive: multimodal prompts were rejected above.
                raise ValueError("vLLM backend requires a text or token-ID prompt")
            vllm_prompt = text
        resolved_params = await self._resolve_generation_defaults(
            request.sampling_params
        )
        vllm_params = self._vllm.SamplingParams(
            **to_vllm_sampling_kwargs(resolved_params)
        )
        generate_kwargs: dict[str, object] = {"priority": request.priority}
        if isinstance(prompt, TemplatedPrompt):
            # vLLM completion tokenization defaults to adding special tokens.
            # HF chat templates already own BOS/EOS/control-token insertion, so
            # applying that completion default here would duplicate tokens.
            generate_kwargs["tokenization_kwargs"] = {
                "add_special_tokens": False,
            }
        async for output in self._engine.generate(
            vllm_prompt,
            vllm_params,
            request.request_id,
            **generate_kwargs,
        ):
            yield self._to_result(request, output)

    async def shutdown(self) -> None:
        shutdown = getattr(self._engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


register_backend("vllm", VLLMBackend)
