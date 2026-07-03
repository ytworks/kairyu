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
    "Qwen3MoeForCausalLM",
    "DeepseekV3ForCausalLM",
)

_MOE_ARCHITECTURES = ("Qwen3MoeForCausalLM", "DeepseekV3ForCausalLM")


@dataclass(frozen=True)
class RopeScaling:
    """llama3 or yarn rope scaling (HF parameter conventions, m15 A5/A7)."""

    kind: str  # "llama3" | "yarn"
    factor: float
    # llama3 fields
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192
    # yarn fields (DeepSeek-V3): mscale keys feed the MLA softmax scale
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float | None = None
    mscale_all_dim: float | None = None


@dataclass(frozen=True)
class MoeConfig:
    """Sparse-MLP fields (m15 D1; aliases per A7: num_experts vs
    num_local_experts vs n_routed_experts)."""

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    # DeepSeek-only routing/topology fields
    n_group: int | None = None
    topk_group: int | None = None
    routed_scaling_factor: float = 1.0
    n_shared_experts: int = 0
    first_k_dense_replace: int = 0

    def is_sparse_layer(self, layer_index: int) -> bool:
        if layer_index < self.first_k_dense_replace:
            return False
        if layer_index in self.mlp_only_layers:
            return False
        return self.num_experts > 0 and (layer_index + 1) % self.decoder_sparse_step == 0


@dataclass(frozen=True)
class MlaConfig:
    """DeepSeek MLA fields (m15 D2/A4)."""

    kv_lora_rank: int
    q_lora_rank: int | None
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


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
    moe: MoeConfig | None = None
    mla: MlaConfig | None = None

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
        # A8: Qwen3-MoE attention is source-identical to Qwen3
        return self.architecture in ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM")

    @property
    def is_mla(self) -> bool:
        return self.mla is not None

    @property
    def kv_cache_num_heads(self) -> int:
        """What PagedKVPool.for_cache allocates (m15 A7)."""
        return 1 if self.is_mla else self.num_key_value_heads

    @property
    def kv_cache_head_dim(self) -> int:
        if self.mla is not None:
            return self.mla.kv_lora_rank + self.mla.qk_rope_head_dim
        return self.head_dim

    @property
    def kv_cache_v_head_dim(self) -> int:
        """MLA caches only the latent in k; the v tensor is unused (width 0)."""
        return 0 if self.is_mla else self.head_dim

    @property
    def rope_dim(self) -> int:
        return self.mla.qk_rope_head_dim if self.mla is not None else self.head_dim

    # set via object.__setattr__ in from_dict (frozen dataclass)
    _attention_bias: bool = False


def _rope_fields(config: dict) -> tuple[float, RopeScaling | None]:
    """Both generations: rope_parameters (nested theta) or rope_scaling + theta."""
    parameters = config.get("rope_parameters") or config.get("rope_scaling") or {}
    theta = config.get("rope_theta", parameters.get("rope_theta", 10000.0))
    scaling = None
    kind = parameters.get("rope_type", parameters.get("type"))
    if kind == "llama3":
        scaling = RopeScaling(
            kind="llama3",
            factor=parameters["factor"],
            low_freq_factor=parameters["low_freq_factor"],
            high_freq_factor=parameters["high_freq_factor"],
            original_max_position_embeddings=parameters[
                "original_max_position_embeddings"
            ],
        )
    elif kind == "yarn":
        scaling = RopeScaling(
            kind="yarn",
            factor=parameters["factor"],
            beta_fast=parameters.get("beta_fast", 32.0),
            beta_slow=parameters.get("beta_slow", 1.0),
            mscale=parameters.get("mscale"),
            mscale_all_dim=parameters.get("mscale_all_dim"),
            original_max_position_embeddings=parameters.get(
                "original_max_position_embeddings",
                config.get("max_position_embeddings", 4096),
            ),
        )
    return float(theta), scaling


def _moe_fields(config: dict, architecture: str) -> MoeConfig | None:
    if architecture not in _MOE_ARCHITECTURES:
        return None
    # A7: hub Qwen3-MoE writes num_experts, save_pretrained writes
    # num_local_experts; DeepSeek uses n_routed_experts
    num_experts = (
        config.get("num_experts")
        or config.get("num_local_experts")
        or config.get("n_routed_experts")
    )
    return MoeConfig(
        num_experts=int(num_experts),
        num_experts_per_tok=int(config["num_experts_per_tok"]),
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        norm_topk_prob=bool(config.get("norm_topk_prob", False)),
        decoder_sparse_step=int(config.get("decoder_sparse_step", 1)),
        mlp_only_layers=tuple(config.get("mlp_only_layers") or ()),
        n_group=config.get("n_group"),
        topk_group=config.get("topk_group"),
        routed_scaling_factor=float(config.get("routed_scaling_factor", 1.0)),
        n_shared_experts=int(config.get("n_shared_experts") or 0),
        first_k_dense_replace=int(config.get("first_k_dense_replace", 0)),
    )


def _mla_fields(config: dict, architecture: str) -> MlaConfig | None:
    if architecture != "DeepseekV3ForCausalLM":
        return None
    return MlaConfig(
        kv_lora_rank=int(config["kv_lora_rank"]),
        q_lora_rank=config.get("q_lora_rank"),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        v_head_dim=int(config["v_head_dim"]),
    )


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
    mla = _mla_fields(config, architecture)
    # A7: DeepSeek saved configs carry head_dim == qk_rope_head_dim; hub
    # originals omit it — for MLA the GQA head_dim is never used, so pin it
    # to the MLA qk head dim rather than deriving hidden//heads
    head_dim = config.get("head_dim") or hidden_size // heads
    if mla is not None:
        head_dim = mla.qk_head_dim
    model_config = ModelConfig(
        architecture=architecture,
        hidden_size=hidden_size,
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=config.get("num_key_value_heads", heads),
        head_dim=head_dim,
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
        dtype=config.get("dtype") or config.get("torch_dtype") or "float32",
        moe=_moe_fields(config, architecture),
        mla=mla,
    )
    object.__setattr__(model_config, "_attention_bias", bool(config.get("attention_bias")))
    return model_config
