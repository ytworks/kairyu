"""Minimal chat template shared by the LLM entrypoint and the server.

M1 renders a plain role-tagged transcript; model-specific templates arrive with
the M2 tokenizer integration.
"""

from __future__ import annotations

from collections.abc import Sequence


def render_chat(messages: Sequence[dict]) -> str:
    lines = [f"{message['role']}: {message.get('content') or ''}" for message in messages]
    return "\n".join(lines) + "\nassistant:"
