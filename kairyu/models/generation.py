"""Parse, validate, and resolve Hugging Face generation defaults."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kairyu.sampling_params import SamplingParams

GenerationConfigMode = Literal["auto", "vllm", "none"]
GENERATION_CONFIG_MODES = frozenset({"auto", "vllm", "none"})


@dataclass(frozen=True)
class GenerationDefaults:
    """Validated model-owned defaults used when a request omits a field."""

    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    mode: GenerationConfigMode = "auto"
    source: str = "builtin"

    def __post_init__(self) -> None:
        if self.eos_token_id is not None and type(self.eos_token_id) is not int:
            raise ValueError("generation eos_token_id must be an integer or null")
        stop_token_ids = tuple(self.stop_token_ids)
        if any(type(token_id) is not int for token_id in stop_token_ids):
            raise ValueError("generation stop_token_ids must contain integers")
        object.__setattr__(self, "stop_token_ids", stop_token_ids)
        for name in ("temperature", "top_p", "min_p", "repetition_penalty"):
            raw = getattr(self, name)
            try:
                finite = math.isfinite(raw)
            except (OverflowError, TypeError):
                finite = False
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not finite:
                raise ValueError(f"generation {name} must be a finite number")
        if type(self.top_k) is not int:
            raise ValueError("generation top_k must be an integer")
        if not isinstance(self.mode, str) or self.mode not in GENERATION_CONFIG_MODES:
            raise ValueError("generation mode is invalid")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("generation source must be a non-empty string")
        SamplingParams(**self.sampling_defaults())

    def sampling_defaults(self) -> dict[str, float | int]:
        """Return the complete effective defaults for public introspection."""

        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }

    def apply(self, params: SamplingParams) -> SamplingParams:
        """Resolve only fields omitted at a public request boundary."""

        omitted = params.generation_config_omitted
        if not omitted:
            return params
        values = self.sampling_defaults()
        return params.clone(
            **{name: values[name] for name in omitted},
            _generation_config_omitted=frozenset(),
        )


def validate_generation_config_mode(mode: object) -> GenerationConfigMode:
    if not isinstance(mode, str) or mode not in GENERATION_CONFIG_MODES:
        choices = ", ".join(sorted(GENERATION_CONFIG_MODES))
        raise ValueError(f"generation_config must be one of {choices}")
    return mode  # type: ignore[return-value]


def _finite_number(raw: object, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"generation_config.json {name} must be numeric")
    try:
        value = float(raw)
    except OverflowError as error:
        raise ValueError(
            f"generation_config.json {name} must be finite"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"generation_config.json {name} must be finite")
    return value


def _sampling_defaults(
    loaded: Mapping[str, object],
) -> dict[str, float | int]:
    neutral: dict[str, float | int] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    values = dict(neutral)
    for name in ("temperature", "top_p", "min_p", "repetition_penalty"):
        raw = loaded.get(name)
        if raw is not None:
            values[name] = _finite_number(raw, name)
    raw_top_k = loaded.get("top_k")
    if raw_top_k is not None:
        if isinstance(raw_top_k, bool) or not isinstance(raw_top_k, int):
            raise ValueError("generation_config.json top_k must be an integer")
        # Hugging Face GenerationConfig uses 0 to disable top-k, while the
        # vLLM/Kairyu SamplingParams spelling for the same state is -1.
        values["top_k"] = -1 if raw_top_k == 0 else raw_top_k
    # Reuse the public sampling contract as the single range validator.
    SamplingParams(**values)
    return values


def parse_generation_defaults(
    directory: Path,
    config: Mapping[str, object],
    generation_config: GenerationConfigMode = "auto",
) -> GenerationDefaults:
    """Parse config plus an optional sibling ``generation_config.json``."""

    mode = validate_generation_config_mode(generation_config)
    eos: object = config.get("eos_token_id")
    loaded: dict[str, object] = {}
    generation_file = directory / "generation_config.json"
    # ``none`` is the explicit no-file mode.  ``vllm`` retains model stop-token
    # metadata but uses vLLM's neutral sampling defaults.  This preserves
    # Kairyu's established multi-EOS termination contract independently from
    # the sampling overlay.
    if mode != "none" and generation_file.is_file():
        raw_loaded = json.loads(generation_file.read_text(encoding="utf-8"))
        if not isinstance(raw_loaded, dict):
            raise ValueError("generation_config.json must contain a JSON object")
        loaded = raw_loaded
        eos = loaded.get("eos_token_id", eos)
    sampling = _sampling_defaults(loaded if mode == "auto" else {})
    source = (
        "generation_config.json"
        if mode == "auto" and generation_file.is_file()
        else "vllm"
        if mode == "vllm"
        else "disabled"
        if mode == "none"
        else "builtin"
    )
    common = {
        **sampling,
        "mode": mode,
        "source": source,
    }
    if eos is None:
        return GenerationDefaults(**common)
    if isinstance(eos, list):
        if not eos:
            raise ValueError("eos_token_id list must not be empty")
        ids = [int(token) for token in eos]
        return GenerationDefaults(
            eos_token_id=ids[0],
            stop_token_ids=tuple(ids[1:]),
            **common,
        )
    return GenerationDefaults(eos_token_id=int(eos), **common)
