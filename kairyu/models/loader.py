"""Checkpoint dir → (DenseDecoder, ModelConfig, generation metadata) (m12 D5).

Uses m8's ``CheckpointReader`` (index.json / sharded / single-file safetensors).
Tied embeddings are mandatory to handle — ``save_pretrained`` genuinely omits
``lm_head.weight`` from the file. Quantized checkpoints fail fast ("arrives in
M14"); the ``linear_factory`` hook is where M14's ``QuantizedLinear`` slots in
without touching this body.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from kairyu.engine.core.quant_config import QuantMethod, detect_quantization
from kairyu.engine.core.weights import CheckpointReader
from kairyu.models.config import ModelConfig, parse_model_config
from kairyu.models.llama import DenseDecoder

_SUPPORTED_BUILDERS = ("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM")


@dataclass(frozen=True)
class GenerationDefaults:
    """From generation_config.json: HF eos may be a LIST (Llama-3 Instruct) —
    the first entry becomes eos_token_id, the rest stop_token_ids (m12 D5)."""

    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()


def _generation_defaults(directory: Path, config: dict) -> GenerationDefaults:
    eos: object = config.get("eos_token_id")
    generation_file = directory / "generation_config.json"
    if generation_file.is_file():
        eos = json.loads(generation_file.read_text()).get("eos_token_id", eos)
    if eos is None:
        return GenerationDefaults()
    if isinstance(eos, list):
        ids = [int(token) for token in eos]
        return GenerationDefaults(eos_token_id=ids[0], stop_token_ids=tuple(ids[1:]))
    return GenerationDefaults(eos_token_id=int(eos))


def build_model(config: ModelConfig, attention_backend=None) -> DenseDecoder:
    """Registry: architecture -> module (one builder covers the dense family)."""
    if config.architecture not in _SUPPORTED_BUILDERS:
        raise ValueError(f"no builder for architecture {config.architecture!r}")
    return DenseDecoder(config, attention_backend=attention_backend)


def load_model(
    path: str | Path,
    dtype: torch.dtype = torch.float32,
    attention_backend=None,
) -> tuple[DenseDecoder, ModelConfig, GenerationDefaults]:
    directory = Path(path)
    config_file = directory / "config.json"
    if not config_file.is_file():
        raise ValueError(f"no config.json at {path}")
    raw_config = json.loads(config_file.read_text())
    quant = detect_quantization(raw_config)
    if quant.method is not QuantMethod.NONE:
        raise ValueError(
            f"quantized checkpoint ({quant.method.value}) loading arrives in M14"
        )
    config = parse_model_config(raw_config)
    model = build_model(config, attention_backend=attention_backend)
    reader = CheckpointReader(directory)
    state: dict[str, torch.Tensor] = {}
    for name, _ in model.named_parameters():
        if name == "lm_head.weight" and config.tie_word_embeddings:
            continue  # tied: the file omits it; DenseDecoder already shares the weight
        if name not in reader:
            raise KeyError(f"checkpoint at {path} is missing tensor {name!r}")
        state[name] = reader.tensor(name).to(dtype)
    model.load_state_dict(state, strict=False)
    model.to(dtype).eval()
    return model, config, _generation_defaults(directory, raw_config)
