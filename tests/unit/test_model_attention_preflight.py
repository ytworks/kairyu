"""Model-construction coverage for optional attention-backend preflight."""

import pytest

from kairyu.models.config import ModelConfig
from kairyu.models.llama import DenseDecoder


def _config() -> ModelConfig:
    return ModelConfig(
        architecture="Qwen3ForCausalLM",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        dtype="float32",
    )


def test_attention_backend_model_preflight_runs_once_before_layer_construction():
    class Backend:
        def __init__(self):
            self.configs = []

        def validate_model_config(self, config):
            self.configs.append(config)
            raise ValueError("unsupported kernel shape")

    backend = Backend()
    with pytest.raises(ValueError, match="unsupported kernel shape"):
        DenseDecoder(_config(), attention_backend=backend)
    assert backend.configs == [_config()]


def test_backend_without_model_preflight_retains_existing_construction_path():
    backend = object()
    model = DenseDecoder(_config(), attention_backend=backend)
    assert len(model.model.layers) == 2
