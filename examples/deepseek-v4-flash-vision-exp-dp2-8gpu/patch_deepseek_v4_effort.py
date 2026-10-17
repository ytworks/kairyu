"""Render the Responses wire's ``xhigh`` effort as DeepSeek-V4's ``max``.

Kairyu forwards an L3 ``max`` request to vLLM's ``/v1/responses`` as
``reasoning.effort: "xhigh"``, the OpenAI vocabulary's top tier (m11 D4
amendment 2026-10-17). The pinned DeepSeek-V4 tokenizer folds every effort but
``low``/``minimal``/``medium``/``max`` onto ``high``, so without this edit an
L3 max request would render the high prompt. The edit replaces exactly one
known source anchor and fails the image build when it is missing or repeated,
so an upstream change can never be patched silently.
"""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

BEFORE = """            elif reasoning_effort == "max":
                reasoning_effort = "max"
"""
AFTER = """            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
"""


def patch(source: str) -> str:
    if source.count(BEFORE) != 1:
        raise ValueError(f"expected exactly one effort anchor, found {source.count(BEFORE)}")
    if AFTER in source:
        raise ValueError("the tokenizer already renders xhigh as max")
    return source.replace(BEFORE, AFTER, 1)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("vllm is not installed")
    path = Path(spec.origin).parent / "tokenizers/deepseek_v4.py"
    path.write_text(patch(path.read_text()))
    py_compile.compile(str(path), doraise=True)
    print(f"patched {path}: xhigh renders as max")


if __name__ == "__main__":
    main()
