"""DeepSeek-V4-Flash-0731 tokenizer adaptations for the Qwen + DeepSeek ensemble.

Every edit replaces exactly one known source anchor of the pinned vLLM
revision (example.json ``vllm.deepseek.source_revision``) and fails the image
build when the anchor is missing or repeated, so an upstream change can never
be patched silently.

1. Effort alias: Kairyu sends the L3 effort ``max`` as the OpenAI Responses
   value ``xhigh`` (m11 D4 amendment 2026-10-17). The pinned deepseek_v4
   tokenizer maps ``max`` to the encoder's max preamble but every other value
   outside low/minimal/medium to high, so ``xhigh`` would silently render as
   high. It now renders as max, like ``max`` itself.
2. Final assistant prefill: the encoder ignores ``continue_final_message``,
   so a final assistant turn is closed with end-of-sentence and its content is
   rendered after an empty ``<think></think>``. Kairyu's public-output floor
   (DTO-D9/D15) continues a thinking span that consumed its cap by sending
   ``<think>\\n{reasoning}\\n</think>\\n\\n`` as that turn. The encoder now
   renders it as the open turn ``<｜Assistant｜><think>\\n{reasoning}\\n</think>\\n\\n``:
   the span becomes the turn's reasoning and end-of-sentence is omitted.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(
            f"expected exactly one source anchor, found {source.count(before)}: {before[:120]!r}"
        )
    if after in source:
        raise ValueError(f"source already contains the replacement: {after[:120]!r}")
    return source.replace(before, after, 1)


# --- 1. effort alias -----------------------------------------------------------

EFFORT_ANCHOR = '            elif reasoning_effort == "max":\n'
EFFORT_REPLACEMENT = (
    "            # Kairyu: the Responses wire carries the L3 max as xhigh.\n"
    '            elif reasoning_effort in ("max", "xhigh"):\n'
)


def xhigh_is_max(source: str) -> str:
    return replace_once(source, EFFORT_ANCHOR, EFFORT_REPLACEMENT)


# --- 2. final assistant prefill -------------------------------------------------

CONTINUATION_ANCHOR = (
    '            conversation = kwargs.get("conversation", messages)\n'
    "            messages = conversation.copy()\n"
)
CONTINUATION_BLOCK = """\
            # Kairyu: continue a final assistant prefill instead of closing it.
            if (
                kwargs.get("continue_final_message")
                and messages
                and messages[-1].get("role") == "assistant"
            ):
                final = dict(messages[-1])
                text = final.get("content") or ""
                if (
                    isinstance(text, str)
                    and text.startswith("<think>")
                    and "</think>" in text
                ):
                    reasoning, _, rest = text[len("<think>") :].partition("</think>")
                    final["reasoning"] = reasoning
                    final["content"] = rest
                final["wo_eos"] = True
                messages[-1] = final
"""
CONTINUATION_REPLACEMENT = CONTINUATION_ANCHOR + CONTINUATION_BLOCK


def final_assistant_prefill(source: str) -> str:
    return replace_once(source, CONTINUATION_ANCHOR, CONTINUATION_REPLACEMENT)


EDITS = [("vllm", "tokenizers/deepseek_v4.py", (xhigh_is_max, final_assistant_prefill))]


def _package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"package {name} is not installed")
    return Path(spec.origin).parent


def main() -> None:
    # Transform everything first; write only when every anchor matched.
    updates: list[tuple[Path, str]] = []
    for package, relative, edits in EDITS:
        path = _package_root(package) / relative
        source = path.read_text()
        for edit in edits:
            source = edit(source)
        updates.append((path, source))
    for path, source in updates:
        path.write_text(source)
        print(f"patched {path}")


if __name__ == "__main__":
    main()
