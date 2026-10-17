"""Responses effort overlay for DeepSeek-V4-Flash-0731 (vllm-effort.Dockerfile).

Kairyu's public effort levels are low/high/max. On the OpenAI Responses wire,
whose effort vocabulary ends at ``xhigh``, Kairyu sends L3 ``max`` as
``xhigh`` (m11 D4 amendment 2026-10-17). The pinned DeepSeek-V4 tokenizer
(``vllm/tokenizers/deepseek_v4.py`` at jasl/vllm@aa0d5130) folds every spelling
other than ``max``, ``none`` and ``low``/``minimal``/``medium`` onto ``high``,
so ``xhigh`` would render the high tier. The edit maps ``xhigh`` to ``max``.

The edit replaces exactly one known source anchor and fails the image build
when the anchor is missing or repeated, so an upstream change can never be
patched silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ANCHOR = '            elif reasoning_effort == "max":\n                reasoning_effort = "max"\n'
REPLACEMENT = (
    '            elif reasoning_effort in ("max", "xhigh"):\n'
    '                reasoning_effort = "max"\n'
)


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(
            f"expected exactly one source anchor, found {source.count(before)}: {before[:120]!r}"
        )
    if after in source:
        raise ValueError(f"source already contains the replacement: {after[:120]!r}")
    return source.replace(before, after, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("vllm is not installed")
    path = Path(spec.origin).parent / "tokenizers/deepseek_v4.py"
    path.write_text(replace_once(path.read_text(), ANCHOR, REPLACEMENT))
    print(f"patched {path}: Responses xhigh renders the max tier")


if __name__ == "__main__":
    main()
