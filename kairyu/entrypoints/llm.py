"""vLLM-signature-compatible offline ``LLM`` entrypoint (design doc D2).

vLLM's offline examples run with an import rewrite only. Unknown engine kwargs
are stored, never fatal, so newer vLLM example flags don't break construction.
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from collections.abc import Mapping, Sequence

from kairyu.engine.backend import (
    EngineBackend,
    GenerationRequest,
    validate_backend_request,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    PromptInput,
    TextPrompt,
    TokensPrompt,
    prompt_kind,
    prompt_text,
    supplied_prompt_token_ids,
)
from kairyu.entrypoints.chat_template import render_chat
from kairyu.outputs import RequestOutput
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    SamplingParams,
)

_DEFAULT_PARAMS = SamplingParams().with_generation_config_omitted(
    GENERATION_CONFIG_SAMPLING_FIELDS
)


def _default_backend(
    model: str, enable_prefix_caching: bool | None, tensor_parallel_size: int = 1
) -> EngineBackend:
    if importlib.util.find_spec("vllm") is not None:
        from kairyu.engine.vllm_backend import VLLMBackend

        return VLLMBackend(
            model=model,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
        )
    from kairyu.engine.mock import MockBackend

    return MockBackend(tensor_parallel_size=tensor_parallel_size)


class LLM:
    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        seed: int | None = 0,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool | None = None,
        trust_remote_code: bool = False,
        backend: EngineBackend | None = None,
        **engine_kwargs: object,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.trust_remote_code = trust_remote_code
        self.engine_kwargs = dict(engine_kwargs)
        self.backend = backend or _default_backend(
            model, enable_prefix_caching, tensor_parallel_size
        )

    def _normalize(
        self,
        prompts: PromptInput | Sequence[PromptInput],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None,
    ) -> tuple[tuple[PromptInput, ...], tuple[SamplingParams, ...]]:
        if isinstance(prompts, (str, TextPrompt, TokensPrompt, MultimodalPrompt)):
            prompt_list = (prompts,)
        elif isinstance(prompts, Mapping) or not isinstance(prompts, Sequence):
            raise TypeError(
                "prompts must be a PromptInput or a sequence of PromptInput values"
            )
        else:
            prompt_list = tuple(prompts)
        for index, prompt in enumerate(prompt_list):
            try:
                prompt_kind(prompt)
            except TypeError as error:
                raise TypeError(
                    f"prompts[{index}] is not a supported PromptInput: {error}"
                ) from error
        if sampling_params is None:
            params_list: tuple[SamplingParams, ...] = (_DEFAULT_PARAMS,) * len(prompt_list)
        elif isinstance(sampling_params, SamplingParams):
            params_list = (sampling_params,) * len(prompt_list)
        else:
            params_list = tuple(sampling_params)
            if len(params_list) != len(prompt_list):
                raise ValueError(
                    f"sampling_params length {len(params_list)} does not match "
                    f"prompts length {len(prompt_list)}"
                )
        return prompt_list, params_list

    async def _generate_async(
        self,
        prompt_list: tuple[PromptInput, ...],
        params_list: tuple[SamplingParams, ...],
        priorities: tuple[int, ...],
    ) -> list[RequestOutput]:
        batch = uuid.uuid4().hex[:12]
        requests = [
            GenerationRequest(
                request_id=f"{batch}-{i}",
                prompt=prompt,
                sampling_params=params,
                priority=priority,
            )
            for i, (prompt, params, priority) in enumerate(
                zip(prompt_list, params_list, priorities, strict=True)
            )
        ]
        for request in requests:
            validate_backend_request(self.backend, request)
        results = await asyncio.gather(*(self.backend.generate(r) for r in requests))
        return [
            RequestOutput(
                request_id=result.request_id,
                prompt=prompt_text(result.prompt),
                prompt_token_ids=(
                    result.prompt_token_ids
                    or supplied_prompt_token_ids(result.prompt)
                    or ()
                ),
                outputs=result.completions,
            )
            for result in results
        ]

    def _run(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "LLM.generate() cannot be called from a running event loop; "
            "use the EngineBackend API (await backend.generate(...)) instead"
        )

    def generate(
        self,
        prompts: PromptInput | Sequence[PromptInput],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool = True,
        priority: Sequence[int] | None = None,
    ) -> list[RequestOutput]:
        prompt_list, params_list = self._normalize(prompts, sampling_params)
        priorities = tuple(priority) if priority is not None else (0,) * len(prompt_list)
        if len(priorities) != len(prompt_list):
            raise ValueError(
                f"priority length {len(priorities)} does not match prompts length "
                f"{len(prompt_list)}"
            )
        return self._run(self._generate_async(prompt_list, params_list, priorities))

    def chat(
        self,
        messages: Sequence[dict] | Sequence[Sequence[dict]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool = True,
        priority: Sequence[int] | None = None,
    ) -> list[RequestOutput]:
        conversations: Sequence[Sequence[dict]]
        if messages and isinstance(messages[0], dict):
            conversations = [messages]  # type: ignore[list-item]
        else:
            conversations = messages  # type: ignore[assignment]
        prompts = [render_chat(conversation) for conversation in conversations]
        return self.generate(
            prompts,
            sampling_params,
            use_tqdm=use_tqdm,
            priority=priority,
        )
