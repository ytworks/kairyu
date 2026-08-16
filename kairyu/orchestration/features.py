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


# Role-profile selection (issue #509) is a separate helper rather than a new
# QueryFeatures field: the dataclass doubles as the M4 training corpus schema
# and must not drift when serving-side signals are added.
_CODE_TASK_KEYWORDS = (
    "code",
    "coding",
    "program",
    "python",
    "function",
    "script",
    "implement",
    "algorithm",
    "compile",
    "debug",
    "refactor",
    "unit test",
    "コード",
    "プログラム",
    "実装",
    "関数",
    "スクリプト",
    "アルゴリズム",
    "デバッグ",
)

# Mirrors the L2 conversation rendering in
# kairyu/entrypoints/server/chat_service.py: the rendered prompt's fixed
# preamble mentions code/JSON/tool vocabulary, so profile keywords must be
# matched against the caller's own words, not the envelope.
_LATEST_USER_START = (
    "--- LATEST USER REQUEST (plain text view of the final user turn) ---\n"
)
_LATEST_USER_END = "\n--- END LATEST USER REQUEST ---"


def latest_user_view(query: str) -> str:
    """Return the final user turn when the L2 envelope markers are present."""

    start = query.rfind(_LATEST_USER_START)
    if start < 0:
        return query
    body = query[start + len(_LATEST_USER_START):]
    end = body.rfind(_LATEST_USER_END)
    return body if end < 0 else body[:end]


def code_task_signal(query: str) -> bool:
    """True when the text plainly asks for code authoring (profile hint)."""

    if _CODE_FENCE in query:
        return True
    lowered = query.lower()
    return any(keyword in lowered for keyword in _CODE_TASK_KEYWORDS)
