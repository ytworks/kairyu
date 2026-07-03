"""ModelConfig: pure HF config.json parsing for the dense family (m12 D1).

Accepts BOTH config generations (reviewed, CRITICAL): transformers-5
``save_pretrained`` writes ``rope_parameters`` (with ``rope_theta`` nested)
and ``dtype``; hub checkpoints carry top-level ``rope_theta``,
``rope_scaling`` and ``torch_dtype``. Qwen2 has no ``attention_bias`` field —
its qkv bias (q/k/v True, o_proj False) is derived from the architecture;
Qwen3's ``attention_bias`` gates all four projections.
"""

from __future__ import annotations

from dataclasses import dataclass

_SUPPORTED_ARCHITECTURES = (
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
)


@dataclass(frozen=True)
class RopeScaling:
    """llama3-type rope scaling (HF ``_compute_llama3_parameters``)."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: RopeScaling | None
    tie_word_embeddings: bool
    dtype: str

    @property
    def qkv_bias(self) -> bool:
        """q/k/v projection bias (o_proj handled separately)."""
        if self.architecture == "Qwen2ForCausalLM":
            return True  # hardcoded in HF Qwen2Attention; no config field exists
        return self._attention_bias

    @property
    def o_bias(self) -> bool:
        if self.architecture == "Qwen2ForCausalLM":
            return False
        return self._attention_bias

    @property
    def qk_norm(self) -> bool:
        return self.architecture == "Qwen3ForCausalLM"

    # set via object.__setattr__ in from_dict (frozen dataclass)
    _attention_bias: bool = False


def _rope_fields(config: dict) -> tuple[float, RopeScaling | None]:
    """Both generations: rope_parameters (nested theta) or rope_scaling + theta."""
    parameters = config.get("rope_parameters") or config.get("rope_scaling") or {}
    theta = config.get("rope_theta", parameters.get("rope_theta", 10000.0))
    scaling = None
    if parameters.get("rope_type", parameters.get("type")) == "llama3":
        scaling = RopeScaling(
            factor=parameters["factor"],
            low_freq_factor=parameters["low_freq_factor"],
            high_freq_factor=parameters["high_freq_factor"],
            original_max_position_embeddings=parameters[
                "original_max_position_embeddings"
            ],
        )
    return float(theta), scaling


def parse_model_config(config: dict) -> ModelConfig:
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else ""
    if architecture not in _SUPPORTED_ARCHITECTURES:
        supported = ", ".join(_SUPPORTED_ARCHITECTURES)
        raise ValueError(
            f"unsupported architecture {architecture!r}; supported: {supported}"
        )
    if config.get("sliding_window") and config.get("use_sliding_window", True):
        raise ValueError("sliding-window attention is not supported (m12 §3)")
    hidden_size = config["hidden_size"]
    heads = config["num_attention_heads"]
    rope_theta, rope_scaling = _rope_fields(config)
    model_config = ModelConfig(
        architecture=architecture,
        hidden_size=hidden_size,
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=config.get("num_key_value_heads", heads),
        head_dim=config.get("head_dim") or hidden_size // heads,
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
        dtype=config.get("dtype") or config.get("torch_dtype") or "float32",
    )
    object.__setattr__(model_config, "_attention_bias", bool(config.get("attention_bias")))
    return model_config
