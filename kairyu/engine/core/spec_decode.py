"""Deterministic-draft speculative decoding policy (design doc m3 §2).

Drafting is deterministic (n-gram lookup or greedy learned-head rollout).
Greedy verification accepts the longest matching prefix plus one bonus token.
For stochastic target sampling, the same token comparison implements standard
rejection sampling for the resulting point-mass draft distribution.
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


def verify_rejection(
    draft: tuple[int, ...], target_tokens: tuple[int, ...]
) -> VerificationResult:
    """Standard rejection sampling for a deterministic draft distribution.

    Every current ``DraftSource`` deterministically proposes one token ``t``,
    hence q(t)=1 and q(x)=0 otherwise. One draw X~p_target implements the
    rejection rule exactly: X=t occurs with acceptance probability p_target(t);
    conditional on X!=t, X has the normalized residual distribution
    p_target(x)/(1-p_target(t)). The first mismatch is therefore already the
    correction token, while matching rows are accepted draft tokens.
    """
    return verify_greedy(draft, target_tokens)
