"""Admission upper bounds and native tool-intent rendering for engine backends.

Shared-engine work is reserved before dispatch from a transport-neutral,
no-I/O upper bound or a backend's explicit override, and the native
tool-intent suffix is rendered exactly once so billing, admission and
``/v1/messages/count_tokens`` hash the same bytes. ``kairyu.engine.backend``
re-exports every public name here; it remains the established import path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kairyu.engine.prompt import (
    PromptInput,
    TemplatedPrompt,
    TextPrompt,
    prompt_kind,
    prompt_text,
    supplied_prompt_token_ids,
)

if TYPE_CHECKING:
    from kairyu.engine.backend import GenerationRequest

# Reservation-only decode ceiling for OpenAI-style limitless requests
# (max_tokens omitted) against a backend whose max_model_len is unknown.
# Generation itself stays bounded by the serving model's context upstream;
# admission merely needs a finite reservation, which exact usage settles.
UNLIMITED_OUTPUT_ADMISSION_TOKENS = 8192


@dataclass(frozen=True)
class AdmissionUpperBound:
    """Worst-case shared-engine work reserved before dispatch."""

    tokens: int
    # Standard usage counts prompt tokens once even when n/best_of duplicates
    # prefill work.  Only single-candidate requests may refund to wire usage.
    refundable_on_exact_usage: bool


def render_tool_intent(
    prompt: PromptInput,
    *,
    tools,
    tool_choice,
    tools_in_prompt: bool,
) -> PromptInput:
    """Render native-engine tool intent exactly once when no HF template did.

    A pre-tokenized prompt is caller-owned: adding a text suffix would silently
    mix two tokenizer owners. Multimodal prompts likewise cannot be flattened
    into text without dropping modality data. Pure function of its arguments:
    billing, admission, and /v1/messages/count_tokens all hash the same bytes.
    """

    if not tools or tools_in_prompt or tool_choice == "none":
        return prompt
    if isinstance(prompt, TemplatedPrompt):
        raise ValueError(
            "templated prompts cannot receive an implicit tool-instruction suffix; "
            "render tools inside the chat template and set tools_in_prompt=true"
        )
    kind = prompt_kind(prompt)
    if kind != "text":
        raise ValueError(
            f"{kind} prompts cannot receive an implicit tool-instruction suffix; "
            "render tools before tokenization and set tools_in_prompt=true"
        )
    text = prompt_text(prompt)
    assert text is not None
    choice = tool_choice
    if isinstance(choice, Mapping):
        named = (choice.get("function") or {}).get("name")
        policy = f"You must call the function {named!r}."
    elif choice == "required":
        policy = "You must call one of the available functions."
    else:
        policy = "Call a function when it is useful."
    schemas = json.dumps(
        list(tools),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rendered = (
        f"{text}\n\nAvailable functions:\n{schemas}\n"
        f"{policy} Emit each call exactly as "
        '<tool_call>{"name":"function_name","arguments":{}}</tool_call>.'
    )
    return TextPrompt(rendered) if isinstance(prompt, TextPrompt) else rendered


def prompt_with_tool_intent(request: GenerationRequest) -> PromptInput:
    """Render the request's tool intent (see ``render_tool_intent``)."""

    return render_tool_intent(
        request.prompt,
        tools=request.tools,
        tool_choice=request.tool_choice,
        tools_in_prompt=request.tools_in_prompt,
    )


async def backend_count_prompt_tokens_async(
    backend: object, prompt: str
) -> int | None:
    """Probe-count prompt tokens for ``/v1/messages/count_tokens``.

    ``None`` is a first-class "declined" answer for backends that cannot
    provide an authoritative count.
    """

    counter = getattr(backend, "count_prompt_tokens_async", None)
    if not callable(counter):
        return None
    count = await counter(prompt)
    if count is None:
        return None
    if type(count) is not int or count < 0:
        raise TypeError(
            "backend count_prompt_tokens_async must return a non-negative int or None"
        )
    return count


def admission_upper_bound(
    request: GenerationRequest,
    *,
    fallback_output_tokens: int | None = None,
) -> AdmissionUpperBound:
    """Transport-neutral, no-I/O upper bound for one generation.

    The gateway intentionally does not own model tokenizers.  We count the
    complete native prompt/tool intent, response-format metadata, and a fixed
    chat-template envelope in UTF-8 work units, then multiply both prefill and
    decode by the actual candidate fan-out.  This stays O(request bytes) and
    avoids placing tenant bookkeeping in the scheduler token hot path.

    ``max_tokens=None`` follows the OpenAI chat contract (generation bounded
    by the model's remaining context), so the decode reservation falls back
    to ``fallback_output_tokens`` — the backend's ``max_model_len`` when
    known — or ``UNLIMITED_OUTPUT_ADMISSION_TOKENS``. The fallback sizes the
    reservation only; exact usage settles the difference afterwards.
    """

    params = request.sampling_params
    max_tokens = params.max_tokens
    if max_tokens is None:
        max_tokens = fallback_output_tokens or UNLIMITED_OUTPUT_ADMISSION_TOKENS
    candidates = max(params.n, params.best_of or params.n)
    prompt = prompt_with_tool_intent(request)
    metadata = json.dumps(
        params.extra_args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    kind = prompt_kind(prompt)
    if kind == "multimodal":
        # A media processor can expand one item into a model-specific number of
        # placeholder tokens. Guessing from bytes would under-reserve some
        # models and overstate usage for others.
        raise ValueError(
            "multimodal prompt admission requires a backend-reported processed "
            "token count"
        )
    token_ids = supplied_prompt_token_ids(prompt)
    if token_ids is not None:
        # Pretokenized inputs are complete: the caller owns BOS, templates, and
        # tool rendering. Their exact sequence length is therefore the truthful
        # prefill reservation; optional display text is never counted.
        prompt_upper = len(token_ids)
    else:
        text = prompt_text(prompt)
        assert text is not None
        # Covers the fixed role/control tokens inserted by common HF/OpenAI chat
        # templates. Supplied prompt, tool schemas, and response schema are
        # counted explicitly above.
        fixed_template_envelope = 256
        prompt_upper = max(
            1,
            len(text.encode("utf-8"))
            + len(metadata.encode("utf-8"))
            + fixed_template_envelope,
        )
    return AdmissionUpperBound(
        tokens=candidates * (prompt_upper + max_tokens),
        refundable_on_exact_usage=candidates == 1,
    )


def backend_admission_upper_bound(
    backend: object,
    request: GenerationRequest,
) -> AdmissionUpperBound:
    """Use an explicit backend processor ceiling when the generic bound cannot.

    Text/token behavior remains byte-for-byte on the historical helper. A
    multimodal backend must opt in with a synchronous, I/O-free bound derived
    from its configured processor limits; media byte length is never treated as
    a token estimate.
    """

    resolve = getattr(backend, "admission_upper_bound", None)
    if callable(resolve):
        bound = resolve(request)
    else:
        bound = admission_upper_bound(
            request,
            fallback_output_tokens=getattr(backend, "max_model_len", None),
        )
    return _validated_admission_upper_bound(backend, bound)


def _validated_admission_upper_bound(
    backend: object,
    bound: object,
) -> AdmissionUpperBound:
    """Validate one optional backend admission result at every call boundary."""

    if not isinstance(bound, AdmissionUpperBound):
        raise TypeError(
            f"{type(backend).__name__}.admission_upper_bound must return "
            "AdmissionUpperBound"
        )
    if type(bound.tokens) is not int or bound.tokens < 1:
        raise ValueError("backend admission bound must contain a positive token count")
    return bound


async def backend_admission_upper_bound_async(
    backend: object,
    request: GenerationRequest,
) -> AdmissionUpperBound:
    """Resolve admission while keeping generic prompt serialization off-loop.

    Stateful composite backends can expose ``admission_upper_bound_async`` and
    snapshot their routing state on the event loop.  Synchronous overrides are
    already required to be I/O-free configured-policy calculations, so their
    request-sized serialization shares the bounded prompt lane with the pure
    transport-neutral fallback.
    """

    resolve_async = getattr(backend, "admission_upper_bound_async", None)
    if callable(resolve_async):
        bound = await resolve_async(request)
        return _validated_admission_upper_bound(backend, bound)
    resolve = getattr(backend, "admission_upper_bound", None)
    from kairyu.async_thread import run_prompt_work

    if callable(resolve):
        calculate = resolve
    else:

        def calculate(request: GenerationRequest) -> AdmissionUpperBound:
            return admission_upper_bound(
                request,
                fallback_output_tokens=getattr(backend, "max_model_len", None),
            )

    bound = await run_prompt_work(calculate, request)
    return _validated_admission_upper_bound(backend, bound)


_GENERIC_ADMISSION_CONTRACT = object()


def backend_admission_upper_bound_key(backend: object) -> object | None:
    """Return an explicit immutable key for equivalent admission semantics.

    The generic fallback depends only on ``GenerationRequest`` plus the
    backend's ``max_model_len`` (the limitless-request reservation ceiling),
    so backends sharing that ceiling share one contract. A backend override
    must opt in with its own immutable key; unknown or unhashable
    declarations are never deduped.
    """

    resolve_async = getattr(backend, "admission_upper_bound_async", None)
    resolve = getattr(backend, "admission_upper_bound", None)
    if not callable(resolve_async) and not callable(resolve):
        return (_GENERIC_ADMISSION_CONTRACT, getattr(backend, "max_model_len", None))
    key = getattr(backend, "admission_upper_bound_key", None)
    if key is None:
        return None
    typed_key = (type(backend), key)
    try:
        hash(typed_key)
    except TypeError:
        return None
    return typed_key


async def prepare_backend_request(
    backend: object,
    request: GenerationRequest,
) -> None:
    """Run optional bounded async preparation before admission/HTTP streaming."""

    prepare = getattr(backend, "prepare_request", None)
    if callable(prepare):
        await prepare(request)
