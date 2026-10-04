"""Typed request-limit errors shared by every engine (M20 WP-04).

A prompt that cannot fit the served context window is a client-correctable
condition with its own public code (``context_length_exceeded``): clients such
as Codex compact their conversation only when they see that code. Engines raise
the typed error with the exact counts; the server renders it per surface.
"""

from __future__ import annotations

CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
# vLLM's SamplingParams default, used only when neither the request nor a
# configured max_model_len bounds the output.
UNBOUNDED_DEFAULT_MAX_TOKENS = 16


class ContextLengthExceededError(ValueError):
    """The prompt plus its output budget does not fit ``max_model_len``.

    A ``ValueError`` so existing request-validation handlers keep rejecting it
    as a client error. ``max_tokens`` is ``None`` when the request omitted it
    and the prompt alone already fills the context window.
    """

    code = CONTEXT_LENGTH_EXCEEDED

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int,
        max_tokens: int | None,
        max_model_len: int,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len


def resolve_output_budget(
    prompt_tokens: int,
    max_tokens: int | None,
    max_model_len: int | None,
) -> int:
    """Return the output-token budget of one request, rejecting overflow.

    An omitted ``max_tokens`` follows the OpenAI chat contract: generate up to
    the remaining context (#496). The engine and every parent-side preflight
    call this one function so they can never disagree about what fits.
    """

    if max_tokens is None:
        if max_model_len is None:
            return UNBOUNDED_DEFAULT_MAX_TOKENS
        remaining = max_model_len - prompt_tokens
        if remaining < 1:
            raise ContextLengthExceededError(
                f"prompt tokens ({prompt_tokens}) already fill "
                f"max_model_len ({max_model_len})",
                prompt_tokens=prompt_tokens,
                max_tokens=None,
                max_model_len=max_model_len,
            )
        return remaining
    if max_model_len is not None and prompt_tokens + max_tokens > max_model_len:
        raise ContextLengthExceededError(
            f"prompt tokens ({prompt_tokens}) plus max_tokens ({max_tokens}) "
            f"exceed max_model_len ({max_model_len})",
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
        )
    return max_tokens
