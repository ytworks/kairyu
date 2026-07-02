"""N-gram draft speculative decoding policy (design doc m3 §2).

Pure policy: drafting is a prompt lookup (no model), verification accepts the
longest draft prefix matching the target model's greedy tokens plus one bonus
token. The invariant — output identical to plain greedy decoding — is pinned
by tests; the GPU runner only changes how target_tokens are produced.
"""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_MAX_DRAFT = 4
_DEFAULT_MAX_NGRAM = 3
_DEFAULT_MIN_NGRAM = 1


def propose_ngram(
    context: tuple[int, ...],
    max_draft: int = _DEFAULT_MAX_DRAFT,
    max_ngram: int = _DEFAULT_MAX_NGRAM,
    min_ngram: int = _DEFAULT_MIN_NGRAM,
) -> tuple[int, ...]:
    """Propose draft tokens by matching the context suffix to an earlier occurrence."""
    for n in range(min(max_ngram, len(context) - 1), min_ngram - 1, -1):
        suffix = context[-n:]
        # latest earlier occurrence that still has a continuation
        for i in range(len(context) - n - 1, -1, -1):
            if context[i : i + n] == suffix:
                return context[i + n : i + n + max_draft]
    return ()


@dataclass(frozen=True)
class VerificationResult:
    accepted: int
    tokens: tuple[int, ...]  # accepted draft tokens + the target's bonus/correction token


def verify_greedy(draft: tuple[int, ...], target_tokens: tuple[int, ...]) -> VerificationResult:
    """Greedy acceptance: longest matching prefix, then the target's own next token."""
    if len(target_tokens) != len(draft) + 1:
        raise ValueError(
            f"target_tokens must score each draft position plus a bonus token: "
            f"expected {len(draft) + 1} tokens, got {len(target_tokens)}"
        )
    accepted = 0
    for draft_token, target_token in zip(draft, target_tokens, strict=False):
        if draft_token != target_token:
            break
        accepted += 1
    return VerificationResult(accepted=accepted, tokens=target_tokens[: accepted + 1])
