"""The context window a local model is served with (M20 WP-04).

Moved from ``kairyu_backend.py`` so the ``kairyu-proc`` parent preflight can
derive the same ``max_model_len`` its child engine builds with, without
importing the engine (and torch) into the HTTP-facing parent process.
"""

from __future__ import annotations

import json
from pathlib import Path


def resolve_model_max_model_len(
    model_path: str | None,
    max_model_len: int | None,
    *,
    raw_model_config: dict | None = None,
) -> int | None:
    """Bind native admission to the resident precomputed RoPE table.

    Without a real model (the toy runner) the configured limit stands.
    """

    if model_path is None:
        return max_model_len
    raw = raw_model_config
    if raw is None:
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            return max_model_len
        raw = json.loads(config_path.read_text())
    text_config = raw.get("text_config")
    position_source = text_config if isinstance(text_config, dict) else raw
    position_limit = position_source.get("max_position_embeddings", 4096)
    if type(position_limit) is not int or position_limit < 1:
        raise ValueError("max_position_embeddings must be an integer >= 1")
    if max_model_len is None:
        return position_limit
    if max_model_len > position_limit:
        raise ValueError(
            f"max_model_len={max_model_len} exceeds the model's "
            f"max_position_embeddings={position_limit}; the precomputed RoPE "
            "table cannot serve positions beyond the model limit"
        )
    return max_model_len
