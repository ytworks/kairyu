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
    "DeepseekV4ForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "KimiLinearForCausalLM",
    "KimiK3ForConditionalGeneration",
)

_MOE_ARCHITECTURES = (
    "Qwen3MoeForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "KimiLinearForCausalLM",
    "KimiK3ForConditionalGeneration",
)

_REFERENCE_ARCHITECTURES = frozenset(
    {
        "DeepseekV4ForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "KimiLinearForCausalLM",
        "KimiK3ForConditionalGeneration",
    }
)


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
    max_position_embeddings: int = 4096
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
        return self.architecture in (
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
        )

    @property
    def requires_full_recompute(self) -> bool:
        """Whether this architecture uses the correctness-first sequence runner."""
        return self.architecture in _REFERENCE_ARCHITECTURES

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


def validate_tensor_parallel_config(config: ModelConfig, tp: int) -> None:
    """Validate the pure model-shape constraints used by TP shard construction."""

    if tp < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tp}")
    if config.num_key_value_heads < 1:
        raise ValueError(
            f"num_key_value_heads must be >= 1, got {config.num_key_value_heads}"
        )
    if config.num_key_value_heads % tp != 0:
        raise ValueError(
            f"num_key_value_heads={config.num_key_value_heads} "
            f"not divisible by tp={tp}"
        )
    if config.num_attention_heads % tp != 0:
        raise ValueError(
            f"num_attention_heads={config.num_attention_heads} "
            f"not divisible by tp={tp}"
        )
    if config.intermediate_size % tp != 0:
        raise ValueError(
            f"intermediate_size={config.intermediate_size} not divisible by tp={tp}"
        )
    if config.vocab_size % tp != 0:
        raise ValueError(f"vocab_size={config.vocab_size} not divisible by tp={tp}")
    if tp == 1:
        return
    if config.is_mla:
        raise ValueError("TP for MLA models is not supported (attention-DP, m16 §3)")
    if config.moe is not None:
        raise ValueError(
            "TP for MoE models is not supported; use expert parallelism (m16 EP)"
        )


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
    elif kind not in (None, "default"):
        # unsupported kinds (linear/dynamic/longrope) must fail fast, not be
        # silently dropped to None — that would load fine and then generate
        # confidently wrong tokens vs hf.generate (M3), a silent parity break
        raise ValueError(
            f"unsupported rope_scaling type {kind!r}: only 'llama3' and 'yarn' "
            "are implemented"
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
    if num_experts is None:
        raise ValueError(
            "MoE config missing expert count "
            "(num_experts / num_local_experts / n_routed_experts)"
        )
    # A trimmed/distilled config that omits these keys must fall back to the
    # SAME defaults as the reference HF config for the architecture (M2), or the
    # block routes unnormalized/unscaled where hf.generate normalizes and scales.
    is_deepseek = architecture in (
        "DeepseekV3ForCausalLM",
        "DeepseekV4ForCausalLM",
    )
    experts_per_token = config.get("num_experts_per_tok")
    if experts_per_token is None:
        experts_per_token = config.get("num_experts_per_token")
    if experts_per_token is None:
        raise ValueError("MoE config missing num_experts_per_tok")
    return MoeConfig(
        num_experts=int(num_experts),
        num_experts_per_tok=int(experts_per_token),
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        norm_topk_prob=bool(config.get("norm_topk_prob", is_deepseek)),
        decoder_sparse_step=int(config.get("decoder_sparse_step", 1)),
        mlp_only_layers=tuple(config.get("mlp_only_layers") or ()),
        n_group=config.get("n_group", 8 if is_deepseek else None),
        topk_group=config.get("topk_group", 4 if is_deepseek else None),
        routed_scaling_factor=float(
            config.get("routed_scaling_factor", 2.5 if is_deepseek else 1.0)
        ),
        n_shared_experts=int(
            config.get("n_shared_experts") or config.get("num_shared_experts") or 0
        ),
        first_k_dense_replace=int(config.get("first_k_dense_replace", 3 if is_deepseek else 0)),
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


def _required_int(
    config: dict,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    value = config[field]
    if type(value) is not int or (
        minimum is not None and value < minimum
    ):
        constraint = (
            "an integer"
            if minimum is None
            else f"an integer >= {minimum}"
        )
        raise ValueError(f"{field} must be {constraint}")
    return value


def parse_model_config(config: dict) -> ModelConfig:
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else ""
    if architecture not in _SUPPORTED_ARCHITECTURES:
        supported = ", ".join(_SUPPORTED_ARCHITECTURES)
        raise ValueError(
            f"unsupported architecture {architecture!r}; supported: {supported}"
        )
    outer_config = config
    if architecture in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
    ):
        text_config = config.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError(f"{architecture} requires a text_config object")
        config = text_config
    if (
        architecture != "DeepseekV4ForCausalLM"
        and config.get("sliding_window")
        and config.get("use_sliding_window", True)
    ):
        raise ValueError("sliding-window attention is not supported (m12 §3)")
    hidden_size = _required_int(config, "hidden_size", minimum=1)
    heads = _required_int(config, "num_attention_heads", minimum=1)
    num_hidden_layers = _required_int(config, "num_hidden_layers")
    num_key_value_heads = config.get("num_key_value_heads", heads)
    if type(num_key_value_heads) is not int or num_key_value_heads < 1:
        raise ValueError("num_key_value_heads must be an integer >= 1")
    intermediate_field = (
        "intermediate_size"
        if "intermediate_size" in config
        else "moe_intermediate_size"
    )
    intermediate_size = _required_int(config, intermediate_field, minimum=1)
    vocab_size = _required_int(config, "vocab_size", minimum=1)
    max_position_embeddings = config.get("max_position_embeddings", 4096)
    if type(max_position_embeddings) is not int or max_position_embeddings < 1:
        raise ValueError("max_position_embeddings must be an integer >= 1")
    rope_theta, rope_scaling = _rope_fields(config)
    mla = _mla_fields(config, architecture)
    # A7: DeepSeek saved configs carry head_dim == qk_rope_head_dim; hub
    # originals omit it — for MLA the GQA head_dim is never used, so pin it
    # to the MLA qk head dim rather than deriving hidden//heads
    head_dim = config.get("head_dim") or hidden_size // heads
    if mla is not None:
        head_dim = mla.qk_head_dim
    if type(head_dim) is not int or head_dim < 1:
        raise ValueError("head_dim must be an integer >= 1")
    model_config = ModelConfig(
        architecture=architecture,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        tie_word_embeddings=outer_config.get(
            "tie_word_embeddings", config.get("tie_word_embeddings", False)
        ),
        dtype=(
            outer_config.get("dtype")
            or outer_config.get("torch_dtype")
            or config.get("dtype")
            or config.get("torch_dtype")
            or "float32"
        ),
        max_position_embeddings=max_position_embeddings,
        moe=_moe_fields(config, architecture),
        mla=mla,
    )
    object.__setattr__(model_config, "_attention_bias", bool(config.get("attention_bias")))
    return model_config
