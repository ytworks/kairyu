"""vLLM adapter behind the EngineBackend seam (design doc D1).

The module always imports; vLLM itself is imported lazily at instantiation so
Kairyu works on machines where vLLM cannot even be installed. Prefix caching is
enabled by default so orchestration steps sharing a prompt prefix already get
KV hits on the vLLM backend (design doc D5).
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator

from kairyu.engine.backend import (
    GenerationRequest,
    GenerationResult,
    prompt_with_tool_intent,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    PromptInput,
    TextPrompt,
    TokensPrompt,
    prompt_text,
    supplied_prompt_token_ids,
)
from kairyu.engine.registry import register_backend
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


def to_vllm_sampling_kwargs(params: SamplingParams) -> dict:
    """Map kairyu SamplingParams to vllm.SamplingParams constructor kwargs."""
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


def _import_vllm():
    try:
        return importlib.import_module("vllm")
    except ImportError as error:
        raise RuntimeError(
            "the 'vllm' backend requires vLLM (pip install vllm); "
            "on unsupported platforms use backend='mock' or an 'openai' worker"
        ) from error


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
        self._vllm = vllm
        self._engine = vllm.AsyncLLMEngine.from_engine_args(args)

    @staticmethod
    def _validated_prompt(request: GenerationRequest) -> PromptInput:
        prompt = request.prompt
        if isinstance(prompt, MultimodalPrompt):
            raise ValueError(
                "vLLM backend does not support multimodal prompts through "
                "Kairyu's typed prompt adapter"
            )
        if not isinstance(prompt, (str, TextPrompt, TokensPrompt)):
            raise ValueError(
                "vLLM backend requires a text or token-ID prompt, "
                f"got {type(prompt).__name__}"
            )
        # Besides rendering text tool intent, this rejects token-ID prompts
        # whose caller did not declare that the tool intent was already rendered.
        return prompt_with_tool_intent(request)

    def validate_request(self, request: GenerationRequest) -> None:
        """Reject prompt variants that this adapter cannot preserve exactly."""

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
        vllm_params = self._vllm.SamplingParams(**to_vllm_sampling_kwargs(request.sampling_params))
        async for output in self._engine.generate(
            vllm_prompt,
            vllm_params,
            request.request_id,
            priority=request.priority,
        ):
            yield self._to_result(request, output)

    async def shutdown(self) -> None:
        shutdown = getattr(self._engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


register_backend("vllm", VLLMBackend)
