"""vLLM-signature-compatible offline ``LLM`` entrypoint (design doc D2).

vLLM's offline examples run with an import rewrite only. Unknown engine kwargs
are stored, never fatal, so newer vLLM example flags don't break construction.
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

from kairyu.engine.backend import (
    EngineBackend,
    GenerationRequest,
    prepare_backend_request,
    validate_backend_request_before_prepare,
)
from kairyu.engine.prompt import (
    MultimodalPrompt,
    PromptInput,
    TemplatedPrompt,
    TextPrompt,
    TokensPrompt,
    prompt_kind,
    prompt_text,
    supplied_prompt_token_ids,
)
from kairyu.engine.tokenizer import (
    TokenizerChatMetadata,
    load_tokenizer_chat_metadata,
)
from kairyu.entrypoints.chat_template import ChatTemplate, render_chat
from kairyu.outputs import RequestOutput
from kairyu.sampling_params import (
    GENERATION_CONFIG_SAMPLING_FIELDS,
    SamplingParams,
)

_DEFAULT_PARAMS = SamplingParams().with_generation_config_omitted(
    GENERATION_CONFIG_SAMPLING_FIELDS
)


def _default_backend(
    model: str,
    enable_prefix_caching: bool | None,
    tensor_parallel_size: int = 1,
    *,
    tokenizer: str | None = None,
) -> EngineBackend:
    if importlib.util.find_spec("vllm") is not None:
        from kairyu.engine.vllm_backend import VLLMBackend

        tokenizer_kwargs = {} if tokenizer is None else {"tokenizer": tokenizer}
        return VLLMBackend(
            model=model,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
            **tokenizer_kwargs,
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
        chat_template: str | Mapping[str, str] | ChatTemplate | None = None,
        legacy_chat: bool = False,
        **engine_kwargs: object,
    ) -> None:
        if type(legacy_chat) is not bool:
            raise TypeError("legacy_chat must be a boolean")
        if legacy_chat and chat_template is not None:
            raise ValueError("chat_template and legacy_chat cannot both be configured")
        self.model = model
        self.tokenizer = tokenizer
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.trust_remote_code = trust_remote_code
        self.engine_kwargs = dict(engine_kwargs)
        self.chat_template = chat_template
        self.legacy_chat = legacy_chat
        self._chat_metadata_cache: dict[
            bool,
            TokenizerChatMetadata | Exception,
        ] = {}
        # Resolve lazily even for a precompiled ChatTemplate so the same
        # unresolved-special-token guard applies to constructor and per-call
        # overrides before the first dispatch.
        self._configured_chat_template: ChatTemplate | None = None
        if legacy_chat:
            warnings.warn(
                "legacy_chat=True enables role-concatenation compatibility; "
                "use a tokenizer-owned HF chat template for real models",
                RuntimeWarning,
                stacklevel=2,
            )
        self.backend = backend or _default_backend(
            model,
            enable_prefix_caching,
            tensor_parallel_size,
            tokenizer=tokenizer,
        )

    def _local_tokenizer_source(self) -> Path | None:
        source = self.tokenizer if self.tokenizer is not None else self.model
        path = Path(source)
        return path if path.exists() else None

    def _chat_metadata(
        self,
        *,
        include_templates: bool,
    ) -> TokenizerChatMetadata | None:
        source = self._local_tokenizer_source()
        if source is None:
            return None
        cached = self._chat_metadata_cache.get(include_templates)
        if isinstance(cached, Exception):
            raise ValueError(
                f"could not load tokenizer chat metadata from {source}: {cached}"
            ) from cached
        if cached is not None:
            return cached
        try:
            metadata = load_tokenizer_chat_metadata(
                source,
                include_templates=include_templates,
            )
        except (OSError, ValueError) as error:
            self._chat_metadata_cache[include_templates] = error
            raise ValueError(
                f"could not load tokenizer chat metadata from {source}: {error}"
            ) from error
        self._chat_metadata_cache[include_templates] = metadata
        return metadata

    def _load_template(
        self,
        source: str | Mapping[str, str] | ChatTemplate,
    ) -> ChatTemplate:
        if isinstance(source, ChatTemplate):
            if source.unresolved_special_token_variables:
                raise ValueError(
                    "ChatTemplate references tokenizer-owned special-token "
                    f"variables {sorted(source.unresolved_special_token_variables)} "
                    "without supplying their values; construct it with "
                    "special_tokens= or pass a string/mapping so LLM can load the "
                    "effective local tokenizer metadata"
                )
            return source
        metadata = self._chat_metadata(include_templates=False)
        special_tokens = None if metadata is None else metadata.special_tokens
        template = ChatTemplate.load(source, special_tokens)
        if template.unresolved_special_token_variables:
            if metadata is None:
                detail = "the effective tokenizer is not materialized locally"
            else:
                detail = (
                    "the effective local tokenizer metadata does not define "
                    "their values"
                )
            raise ValueError(
                "explicit chat template references tokenizer-owned special-token "
                f"variables {sorted(template.unresolved_special_token_variables)}, "
                f"but {detail}; use a complete local tokenizer snapshot, make the "
                "template self-contained, or explicitly opt in with legacy_chat=True"
            )
        return template

    def _resolve_chat_template(
        self,
        override: str | Mapping[str, str] | ChatTemplate | None,
    ) -> ChatTemplate | None:
        if override is not None:
            return self._load_template(override)
        if self.chat_template is not None:
            if self._configured_chat_template is None:
                self._configured_chat_template = self._load_template(self.chat_template)
            return self._configured_chat_template
        if self.legacy_chat:
            return None

        metadata = self._chat_metadata(include_templates=True)
        if metadata is not None and metadata.templates:
            if self._configured_chat_template is None:
                self._configured_chat_template = ChatTemplate.from_tokenizer_metadata(
                    metadata.templates,
                    metadata.special_tokens,
                )
            return self._configured_chat_template
        source = self.tokenizer if self.tokenizer is not None else self.model
        raise ValueError(
            f"model {self.model!r} has no resolvable chat template for tokenizer "
            f"{source!r}; use a local tokenizer snapshot with chat metadata, pass "
            "chat_template=, or explicitly opt in with legacy_chat=True"
        )

    def _normalize(
        self,
        prompts: PromptInput | Sequence[PromptInput],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None,
    ) -> tuple[tuple[PromptInput, ...], tuple[SamplingParams, ...]]:
        if isinstance(
            prompts,
            (str, TextPrompt, TemplatedPrompt, TokensPrompt, MultimodalPrompt),
        ):
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
            validate_backend_request_before_prepare(self.backend, request)
        for request in requests:
            await prepare_backend_request(self.backend, request)
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
        *,
        chat_template: str | Mapping[str, str] | ChatTemplate | None = None,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, object]] | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> list[RequestOutput]:
        conversations: Sequence[Sequence[dict]]
        if messages and isinstance(messages[0], dict):
            conversations = [messages]  # type: ignore[list-item]
        else:
            conversations = messages  # type: ignore[assignment]
        template = self._resolve_chat_template(chat_template)
        if template is None:
            unsupported: list[str] = []
            if not add_generation_prompt:
                unsupported.append("add_generation_prompt=false")
            if tools is not None:
                unsupported.append("tools")
            if chat_template_kwargs:
                unsupported.append("chat_template_kwargs")
            if unsupported:
                raise ValueError(
                    "legacy_chat cannot apply: " + ", ".join(unsupported)
                )
            prompts: list[PromptInput] = [
                render_chat(conversation) for conversation in conversations
            ]
        else:
            prompts = [
                TemplatedPrompt(
                    template.render(
                        conversation,
                        tools=tools,
                        add_generation_prompt=add_generation_prompt,
                        template_kwargs=chat_template_kwargs,
                    )
                )
                for conversation in conversations
            ]
        return self.generate(
            prompts,
            sampling_params,
            use_tqdm=use_tqdm,
            priority=priority,
        )
