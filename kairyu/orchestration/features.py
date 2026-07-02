"""Pure query feature extraction for routing.

Kept model-free so the routing hot path stays well under the 10ms budget
(design doc D3) and features double as the M4 training corpus schema.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_CODE_FENCE = "```"
_MATH_SYMBOLS = frozenset("=+-*/^<>∑∫√±≤≥")
_REASONING_KEYWORDS = (
    "prove",
    "derive",
    "explain",
    "why",
    "reason",
    "step by step",
    "theorem",
    "analyze",
    "optimize",
    "trade-off",
)
_MULTI_STEP_MARKERS = (
    re.compile(r"\bfirst\b", re.IGNORECASE),
    re.compile(r"\bthen\b", re.IGNORECASE),
    re.compile(r"\bafter that\b", re.IGNORECASE),
    re.compile(r"\bfinally\b", re.IGNORECASE),
    re.compile(r"\bstep \d", re.IGNORECASE),
)


@dataclass(frozen=True)
class QueryFeatures:
    char_len: int
    word_count: int
    has_code_fence: bool
    math_symbol_count: int
    reasoning_keyword_count: int
    multi_step_marker_count: int
    question_count: int

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def extract_features(query: str) -> QueryFeatures:
    lowered = query.lower()
    return QueryFeatures(
        char_len=len(query),
        word_count=len(query.split()),
        has_code_fence=_CODE_FENCE in query,
        math_symbol_count=sum(1 for ch in query if ch in _MATH_SYMBOLS),
        reasoning_keyword_count=sum(1 for kw in _REASONING_KEYWORDS if kw in lowered),
        multi_step_marker_count=sum(len(m.findall(query)) for m in _MULTI_STEP_MARKERS),
        question_count=query.count("?"),
    )
