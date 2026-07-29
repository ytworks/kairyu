"""Pure parsing for Hugging Face generation defaults."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationDefaults:
    """EOS defaults used by the engine after model metadata validation."""

    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()


def parse_generation_defaults(
    directory: Path,
    config: Mapping[str, object],
) -> GenerationDefaults:
    """Parse config plus an optional sibling ``generation_config.json``."""

    eos: object = config.get("eos_token_id")
    generation_file = directory / "generation_config.json"
    if generation_file.is_file():
        loaded = json.loads(generation_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("generation_config.json must contain a JSON object")
        eos = loaded.get("eos_token_id", eos)
    if eos is None:
        return GenerationDefaults()
    if isinstance(eos, list):
        if not eos:
            raise ValueError("eos_token_id list must not be empty")
        ids = [int(token) for token in eos]
        return GenerationDefaults(
            eos_token_id=ids[0],
            stop_token_ids=tuple(ids[1:]),
        )
    return GenerationDefaults(eos_token_id=int(eos))
