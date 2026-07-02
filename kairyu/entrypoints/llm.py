"""vLLM-signature-compatible offline ``LLM`` entrypoint (design doc D2).

vLLM's offline examples run with an import rewrite only. Unknown engine kwargs
are stored, never fatal, so newer vLLM example flags don't break construction.
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from collections.abc import Sequence

from kairyu.engine.backend import EngineBackend, GenerationRequest
from kairyu.outputs import RequestOutput
from kairyu.sampling_params import SamplingParams

_DEFAULT_PARAMS = SamplingParams()


def _default_backend(model: str, enable_prefix_caching: bool | None) -> EngineBackend:
    if importlib.util.find_spec("vllm") is not None:
        from kairyu.engine.vllm_backend import VLLMBackend

        return VLLMBackend(model=model, enable_prefix_caching=enable_prefix_caching)
    from kairyu.engine.mock import MockBackend

    return MockBackend()


def _render_chat(messages: Sequence[dict]) -> str:
    lines = [f"{message['role']}: {message['content']}" for message in messages]
    return "\n".join(lines) + "\nassistant:"


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
        self.backend = backend or _default_backend(model, enable_prefix_caching)

    def _normalize(
        self,
        prompts: str | Sequence[str],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None,
    ) -> tuple[tuple[str, ...], tuple[SamplingParams, ...]]:
        prompt_list = (prompts,) if isinstance(prompts, str) else tuple(prompts)
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
        self, prompt_list: tuple[str, ...], params_list: tuple[SamplingParams, ...]
    ) -> list[RequestOutput]:
        batch = uuid.uuid4().hex[:12]
        requests = [
            GenerationRequest(
                request_id=f"{batch}-{i}", prompt=prompt, sampling_params=params
            )
            for i, (prompt, params) in enumerate(zip(prompt_list, params_list))
        ]
        results = await asyncio.gather(*(self.backend.generate(r) for r in requests))
        return [
            RequestOutput(
                request_id=result.request_id,
                prompt=result.prompt,
                prompt_token_ids=(),
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
        prompts: str | Sequence[str],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        prompt_list, params_list = self._normalize(prompts, sampling_params)
        return self._run(self._generate_async(prompt_list, params_list))

    def chat(
        self,
        messages: Sequence[dict] | Sequence[Sequence[dict]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        conversations: Sequence[Sequence[dict]]
        if messages and isinstance(messages[0], dict):
            conversations = [messages]  # type: ignore[list-item]
        else:
            conversations = messages  # type: ignore[assignment]
        prompts = [_render_chat(conversation) for conversation in conversations]
        return self.generate(prompts, sampling_params, use_tqdm=use_tqdm)
