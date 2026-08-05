"""vLLM-signature-compatible sampling parameters.

Field names and defaults mirror ``vllm.SamplingParams`` so vLLM examples run
with an import rewrite only (design doc D2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

GENERATION_CONFIG_SAMPLING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
    }
)

# ``response_format`` is the only sampling-extension key the native engine
# translates into structured-output/grammar state.  Keep that ownership shared
# by SamplingParams validation, native request validation, and engine mapping so
# a new grammar carrier cannot silently bypass forced-continuation safeguards.
RESPONSE_FORMAT_EXTRA_ARG = "response_format"
STRUCTURED_OUTPUT_EXTRA_ARGS = frozenset({RESPONSE_FORMAT_EXTRA_ARG})


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
    # Native-engine continuation scoring: emit these exact token IDs in order
    # while retaining their raw model log probabilities.  Appending the field
    # after the existing public parameters preserves their positional order.
    forced_token_ids: Sequence[int] | None = None
    # HTTP/Responses requests must preserve which model-owned sampling values
    # were omitted until the final backend is known.  Concrete Python
    # SamplingParams remain fully explicit by default.  The field is private
    # transport intent, excluded from equality/repr, and consumed before a
    # native engine samples.
    _generation_config_omitted: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        omitted = frozenset(self._generation_config_omitted)
        unknown = omitted - GENERATION_CONFIG_SAMPLING_FIELDS
        if unknown:
            raise ValueError(
                "unknown generation-config sampling fields: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "_generation_config_omitted", omitted)
        object.__setattr__(self, "stop", _normalize_stop(self.stop))
        object.__setattr__(
            self, "stop_token_ids", tuple(self.stop_token_ids) if self.stop_token_ids else ()
        )
        if self.forced_token_ids is not None:
            try:
                forced_token_ids = tuple(self.forced_token_ids)
            except TypeError as error:
                raise ValueError(
                    "forced_token_ids must be a non-empty sequence of "
                    "non-negative integers"
                ) from error
            object.__setattr__(self, "forced_token_ids", forced_token_ids)
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
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                "min_tokens must be <= max_tokens, "
                f"got min_tokens={self.min_tokens}, max_tokens={self.max_tokens}"
            )
        if self.forced_token_ids is not None:
            if not self.forced_token_ids:
                raise ValueError("forced_token_ids must not be empty when set")
            if any(
                type(token_id) is not int or token_id < 0
                for token_id in self.forced_token_ids
            ):
                raise ValueError(
                    "forced_token_ids must contain only non-negative integers"
                )
            if self.max_tokens != len(self.forced_token_ids):
                raise ValueError(
                    "max_tokens must equal the forced_token_ids length, "
                    f"got max_tokens={self.max_tokens}, "
                    f"forced_token_ids length={len(self.forced_token_ids)}"
                )
            if not self.ignore_eos:
                raise ValueError(
                    "forced_token_ids requires ignore_eos=True so an EOS token "
                    "can be scored without terminating the continuation"
                )
            if self.stop or self.stop_token_ids:
                raise ValueError(
                    "forced_token_ids cannot be combined with stop or "
                    "stop_token_ids"
                )
            if self.min_tokens != 0:
                raise ValueError(
                    "forced_token_ids cannot be combined with min_tokens"
                )
            structured_output = (
                STRUCTURED_OUTPUT_EXTRA_ARGS.intersection(self.extra_args)
                if isinstance(self.extra_args, Mapping)
                else set()
            )
            if structured_output:
                raise ValueError(
                    "forced_token_ids cannot be combined with structured-output "
                    "extra args: " + ", ".join(sorted(structured_output))
                )

    def clone(self, **overrides: object) -> SamplingParams:
        """Return a new SamplingParams with the given fields replaced."""
        if "_generation_config_omitted" not in overrides:
            explicitly_overridden = GENERATION_CONFIG_SAMPLING_FIELDS.intersection(
                overrides
            )
            overrides["_generation_config_omitted"] = (
                self._generation_config_omitted - explicitly_overridden
            )
        return dataclasses.replace(self, **overrides)  # type: ignore[arg-type]

    @property
    def generation_config_omitted(self) -> frozenset[str]:
        """Model-owned sampling fields omitted at the public request boundary."""

        return self._generation_config_omitted

    def with_generation_config_omitted(
        self,
        fields: Sequence[str],
    ) -> SamplingParams:
        """Retain request omission until a concrete backend resolves defaults."""

        return dataclasses.replace(
            self,
            _generation_config_omitted=frozenset(fields),
        )
