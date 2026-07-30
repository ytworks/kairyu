"""vLLM-signature-compatible sampling parameters.

Field names and defaults mirror ``vllm.SamplingParams`` so vLLM examples run
with an import rewrite only (design doc D2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


class _FrozenDict(dict):
    """JSON/msgpack-compatible dictionary with no public mutation methods."""

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("SamplingParams.extra_args is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

PROMPT_CARRIER_EXTRA_ARGS = frozenset(
    {
        # Prompt ownership is represented only by GenerationRequest.prompt.
        # These common vLLM/OpenAI carrier names must never enter through the
        # sampling-extension bag, where some backends would silently drop them.
        "prompt",
        "messages",
        "input",
        "inputs",
        "prompt_token_ids",
        "prompt_embeds",
        "prompt_is_token_ids",
        "prompt_token_offsets",
        "input_ids",
        "input_embeds",
        "inputs_embeds",
        "token_type_ids",
        "decoder_input_ids",
        "decoder_inputs_embeds",
        "encoder_prompt",
        "encoder_prompt_token_ids",
        "decoder_prompt",
        "multi_modal_data",
        "multi_modal_uuids",
        "mm_processor_kwargs",
        # OpenAI/IO/model-runner content carriers are equally non-sampling
        # input, even though they are not members of vLLM's PromptType.
        "input_text",
        "input_image",
        "input_file",
        "image_data",
        "audio_data",
        "video_data",
        "input_audio",
        "image_url",
        "audio_url",
        "video_url",
        "pixel_values",
        "input_features",
    }
)
PROMPT_IDENTITY_EXTRA_ARGS = frozenset(
    {
        # vLLM PromptType owns this prefix-cache identity control. Treating it
        # as a sampling extension would let adapters silently discard or
        # reinterpret cache authority.
        "cache_salt",
    }
)
PROMPT_OWNED_EXTRA_ARGS = PROMPT_CARRIER_EXTRA_ARGS | PROMPT_IDENTITY_EXTRA_ARGS


def validate_prompt_owned_extra_args(extra_args: object) -> None:
    if not isinstance(extra_args, Mapping):
        return
    prompt_owned = PROMPT_OWNED_EXTRA_ARGS.intersection(extra_args)
    if prompt_owned:
        raise ValueError(
            "extra_args may not contain prompt-owned fields "
            f"{sorted(prompt_owned)}; pass input through PromptInput and "
            "cache identity through CacheHint"
        )


def _normalize_stop(stop: str | Sequence[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


@dataclass(frozen=True)
class SamplingParams:
    n: int = 1
    best_of: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: int | None = None
    stop: str | Sequence[str] | None = None
    stop_token_ids: Sequence[int] | None = None
    max_tokens: int | None = 16
    min_tokens: int = 0
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    extra_args: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop", _normalize_stop(self.stop))
        object.__setattr__(
            self, "stop_token_ids", tuple(self.stop_token_ids) if self.stop_token_ids else ()
        )
        if isinstance(self.extra_args, Mapping):
            # A frozen SamplingParams must not expose a mutable top-level bag
            # that can acquire a prompt carrier after construction.
            object.__setattr__(
                self,
                "extra_args",
                _FrozenDict(self.extra_args),
            )
        self._validate()

    def _validate(self) -> None:
        validate_prompt_owned_extra_args(self.extra_args)
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 or >= 1, got {self.top_k}")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(f"presence_penalty must be in [-2, 2], got {self.presence_penalty}")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be >= 0, got {self.min_tokens}")

    def clone(self, **overrides: object) -> SamplingParams:
        """Return a new SamplingParams with the given fields replaced."""
        return dataclasses.replace(self, **overrides)  # type: ignore[arg-type]
