"""Explicit request-capability policy for OpenAI-compatible upstreams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from kairyu.sampling_params import PROMPT_OWNED_EXTRA_ARGS

SAMPLING_FIELD_NAMES = frozenset(
    {
        "best_of",
        "frequency_penalty",
        "ignore_eos",
        "logprobs",
        "max_tokens",
        "min_p",
        "min_tokens",
        "n",
        "presence_penalty",
        "repetition_penalty",
        "response_format",
        "seed",
        "skip_special_tokens",
        "stop",
        "stop_token_ids",
        "temperature",
        "top_k",
        "top_p",
    }
)
PROMPT_KINDS = frozenset({"text", "tokens", "multimodal"})

_OPENAI_CORE = frozenset(
    {
        "frequency_penalty",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "top_p",
    }
)
_VLLM_EXTENSIONS = frozenset(
    {
        "ignore_eos",
        "min_p",
        "min_tokens",
        "repetition_penalty",
        "skip_special_tokens",
        "stop_token_ids",
        "top_k",
    }
)

# ``extra_args`` is merged into this request body. These keys are owned by the
# backend and may never be allowlisted as vendor extensions.
RESERVED_EXTRA_ARGS = frozenset(
    {
        "model",
        "stream",
        "stream_options",
        "tool_choice",
        "tools",
        "top_logprobs",
        "max_completion_tokens",
        "prompt_logprobs",
        "priority",
        *SAMPLING_FIELD_NAMES,
        *PROMPT_OWNED_EXTRA_ARGS,
    }
)


@dataclass(frozen=True)
class OpenAIRequestCapabilities:
    """One resolved upstream contract.

    ``sampling_fields`` names Kairyu ``SamplingParams`` fields, not arbitrary
    JSON keys. ``extra_args`` is a separately reviewed allowlist for vendor
    extensions. Constraints prevent a compatibility layer from accepting a
    request and then silently coercing or ignoring its intent.
    """

    upstream: str
    sampling_fields: frozenset[str]
    extra_args: frozenset[str] = frozenset()
    max_n: int | None = None
    max_temperature: float | None = None
    max_tokens_wire_name: str = "max_tokens"
    forward_neutral_fields: frozenset[str] = frozenset()
    strict_tools: bool = False
    priority: bool = False
    # The current adapter targets Chat Completions, whose portable request
    # contract is text-only. Keeping prompt kinds in the immutable validation
    # key makes future modality/token support an explicit capability change.
    prompt_kinds: frozenset[str] = frozenset({"text"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling_fields", frozenset(self.sampling_fields))
        object.__setattr__(self, "extra_args", frozenset(self.extra_args))
        object.__setattr__(
            self,
            "forward_neutral_fields",
            frozenset(self.forward_neutral_fields),
        )
        object.__setattr__(self, "prompt_kinds", frozenset(self.prompt_kinds))
        if not isinstance(self.upstream, str) or not self.upstream:
            raise ValueError("upstream must be a non-empty string")
        unknown = self.sampling_fields - SAMPLING_FIELD_NAMES
        if unknown:
            raise ValueError(f"unknown OpenAI sampling capability fields: {sorted(unknown)}")
        unknown_prompt_kinds = self.prompt_kinds - PROMPT_KINDS
        if unknown_prompt_kinds:
            raise ValueError(
                "unknown OpenAI prompt capability kinds: "
                f"{sorted(unknown_prompt_kinds)}"
            )
        invalid_neutral = self.forward_neutral_fields - self.sampling_fields
        if invalid_neutral:
            raise ValueError(
                "neutral forwarding fields must be sampling capabilities: "
                f"{sorted(invalid_neutral)}"
            )
        unsafe = self.extra_args & RESERVED_EXTRA_ARGS
        if unsafe:
            raise ValueError(
                "vendor extra_args may not override backend-owned fields: "
                + ", ".join(sorted(unsafe))
            )
        if self.max_n is not None and self.max_n < 1:
            raise ValueError("max_n must be positive")
        if self.max_temperature is not None and self.max_temperature < 0:
            raise ValueError("max_temperature must be non-negative")
        if self.max_tokens_wire_name not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("max_tokens_wire_name must be 'max_tokens' or 'max_completion_tokens'")


_PROFILES = {
    # Backwards-compatible pre-#209 behavior: OpenAI core plus the two
    # historically forwarded local-server extensions.
    "generic": OpenAIRequestCapabilities(
        upstream="generic",
        sampling_fields=_OPENAI_CORE | {"top_k", "min_p"},
        forward_neutral_fields={
            "frequency_penalty",
            "n",
            "presence_penalty",
            "temperature",
            "top_p",
        },
        strict_tools=True,
    ),
    "openai": OpenAIRequestCapabilities(
        upstream="openai",
        sampling_fields=_OPENAI_CORE,
        extra_args={"parallel_tool_calls", "reasoning_effort", "service_tier"},
        max_tokens_wire_name="max_completion_tokens",
        strict_tools=True,
    ),
    "vllm": OpenAIRequestCapabilities(
        upstream="vllm",
        sampling_fields=_OPENAI_CORE | _VLLM_EXTENSIONS,
        priority=True,
    ),
    # Kairyu's typed HTTP boundary exposes the vLLM-style sampling controls
    # that its native engines execute. ``best_of``, ``prompt_logprobs``, and
    # ``skip_special_tokens`` remain excluded until their complete semantics
    # are represented by the API response/native decode path.
    "kairyu": OpenAIRequestCapabilities(
        upstream="kairyu",
        sampling_fields=_OPENAI_CORE
        | {
            "ignore_eos",
            "min_p",
            "min_tokens",
            "repetition_penalty",
            "stop_token_ids",
            "top_k",
        },
        max_tokens_wire_name="max_completion_tokens",
        priority=True,
    ),
    # Anthropic documents that these are the compatible controls it executes;
    # n>1 and temperature>1 are not truthful passthrough.
    "anthropic": OpenAIRequestCapabilities(
        upstream="anthropic",
        sampling_fields={"max_tokens", "n", "stop", "temperature", "top_p"},
        max_n=1,
        max_temperature=1.0,
        max_tokens_wire_name="max_completion_tokens",
    ),
    # Gemini's compatibility contract and current model families vary in their
    # sampling controls. Keep the provider-wide preset to documented portable
    # output controls; a deployment may use ``generic`` plus an explicitly
    # verified custom contract for a pinned model.
    "gemini": OpenAIRequestCapabilities(
        upstream="gemini",
        sampling_fields={"max_tokens", "response_format"},
        extra_args={"extra_body", "reasoning_effort"},
        max_tokens_wire_name="max_completion_tokens",
    ),
}

_OVERRIDE_KEYS = frozenset(
    {
        "allow_extra_args",
        "allow_prompt_kinds",
        "allow_sampling_fields",
        "deny_sampling_fields",
        "strict_tools",
    }
)


def _string_set(value: object, *, field: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"capabilities.{field} must be a sequence of strings")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"capabilities.{field} must contain non-empty strings")
    if len(items) != len(set(items)):
        raise ValueError(f"capabilities.{field} contains duplicates")
    return frozenset(items)


def resolve_openai_capabilities(
    upstream: str,
    overrides: Mapping[str, object] | OpenAIRequestCapabilities | None = None,
) -> OpenAIRequestCapabilities:
    """Resolve a named preset plus safe additive/subtractive configuration."""

    if isinstance(overrides, OpenAIRequestCapabilities):
        return overrides
    if not isinstance(upstream, str):
        raise ValueError("upstream must be a string")
    try:
        base = _PROFILES[upstream]
    except KeyError as error:
        raise ValueError(
            f"unknown OpenAI-compatible upstream {upstream!r}; "
            f"known profiles: {', '.join(sorted(_PROFILES))}"
        ) from error
    if overrides is None:
        return base
    if not isinstance(overrides, Mapping):
        raise ValueError("capabilities must be a mapping or OpenAIRequestCapabilities")
    if any(not isinstance(key, str) for key in overrides):
        raise ValueError("capability override keys must be strings")
    unknown_keys = set(overrides) - _OVERRIDE_KEYS
    if unknown_keys:
        raise ValueError(f"unknown capability override keys: {sorted(unknown_keys)}")

    allow_sampling = _string_set(
        overrides.get("allow_sampling_fields"),
        field="allow_sampling_fields",
    )
    deny_sampling = _string_set(
        overrides.get("deny_sampling_fields"),
        field="deny_sampling_fields",
    )
    unknown_fields = (allow_sampling | deny_sampling) - SAMPLING_FIELD_NAMES
    if unknown_fields:
        raise ValueError(f"unknown OpenAI sampling capability fields: {sorted(unknown_fields)}")
    overlap = allow_sampling & deny_sampling
    if overlap:
        raise ValueError(
            f"sampling capability fields cannot be both allowed and denied: {sorted(overlap)}"
        )
    if allow_sampling and upstream != "generic":
        raise ValueError(
            "additive sampling capabilities require upstream='generic'; "
            "named provider presets may only be narrowed"
        )

    allow_extra = _string_set(
        overrides.get("allow_extra_args"),
        field="allow_extra_args",
    )
    allow_prompt_kinds = _string_set(
        overrides.get("allow_prompt_kinds"),
        field="allow_prompt_kinds",
    )
    unknown_prompt_kinds = allow_prompt_kinds - PROMPT_KINDS
    if unknown_prompt_kinds:
        raise ValueError(
            "unknown OpenAI prompt capability kinds: "
            f"{sorted(unknown_prompt_kinds)}"
        )
    strict_tools = overrides.get("strict_tools", base.strict_tools)
    if not isinstance(strict_tools, bool):
        raise ValueError("capabilities.strict_tools must be a boolean")
    if strict_tools and not base.strict_tools and upstream != "generic":
        raise ValueError(
            "enabling strict tool semantics requires upstream='generic'; "
            "named provider presets may only be narrowed"
        )
    return replace(
        base,
        sampling_fields=(base.sampling_fields | allow_sampling) - deny_sampling,
        extra_args=base.extra_args | allow_extra,
        forward_neutral_fields=base.forward_neutral_fields - deny_sampling,
        strict_tools=strict_tools,
        prompt_kinds=base.prompt_kinds | allow_prompt_kinds,
    )


def known_openai_upstreams() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))
