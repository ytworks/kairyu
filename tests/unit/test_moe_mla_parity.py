"""m15 flagship gates: Qwen3-MoE and DeepSeek-V3 tiny parity vs transformers.

Weight transfer goes save_pretrained → load_model (m15 A1: transformers keeps
experts FUSED in memory; the hub/per-expert names only exist on disk).
"""


import pytest
import torch

from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.engine.core.model_runner import PagedModelRunner
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.models.loader import load_model

transformers = pytest.importorskip("transformers")

PAGE = 4

QWEN3_MOE = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    max_position_embeddings=512,
)

DSV3_BASE = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=16,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,  # V3 convention (n_rep = 1)
    kv_lora_rank=16,
    qk_nope_head_dim=8,
    qk_rope_head_dim=4,
    v_head_dim=12,  # != qk dims: catches transposes (m15 A9)
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
    routed_scaling_factor=2.5,
    norm_topk_prob=True,
    n_shared_experts=2,
    first_k_dense_replace=1,
    rms_norm_eps=1e-6,  # HF hardcodes 1e-6 in the MLA norms (m15 A4)
    max_position_embeddings=512,
)

YARN = {
    "type": "yarn",
    "factor": 4.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 64,
}


def _decisive_routing(model) -> None:
    """Random tiny gates produce near-tied expert scores; fp32 noise (~1e-7)
    between our forward and HF's can then flip top-k selection and amplify to
    O(1) diffs (verified: the MoE block matches to 1e-9 on identical inputs).
    Scaling the gate weights widens routing margins well past the noise floor
    so exact-parity gates stay meaningful."""
    for module in model.modules():
        if module.__class__.__name__ in ("Qwen3MoeSparseMoeBlock", "DeepseekV3MoE"):
            gate = module.gate
            with torch.no_grad():
                gate.weight.mul_(6.0)


def _build(kind: str):
    torch.manual_seed(51)
    if kind == "qwen3moe":
        model = transformers.Qwen3MoeForCausalLM(transformers.Qwen3MoeConfig(**QWEN3_MOE))
    elif kind == "dsv3":
        model = transformers.DeepseekV3ForCausalLM(
            transformers.DeepseekV3Config(**DSV3_BASE, q_lora_rank=24)
        )
    elif kind == "dsv3-noqlora":
        model = transformers.DeepseekV3ForCausalLM(
            transformers.DeepseekV3Config(**DSV3_BASE, q_lora_rank=None)
        )
    else:  # dsv3-yarn
        model = transformers.DeepseekV3ForCausalLM(
            transformers.DeepseekV3Config(**DSV3_BASE, q_lora_rank=24, rope_scaling=YARN)
        )
    model = model.to(torch.float32).eval()
    _decisive_routing(model)
    return model


@pytest.fixture(
    scope="module", params=["qwen3moe", "dsv3", "dsv3-noqlora", "dsv3-yarn"]
)
def arch(request, tmp_path_factory):
    hf_model = _build(request.param)
    path = tmp_path_factory.mktemp(f"moe-{request.param}")
    hf_model.save_pretrained(path, safe_serialization=True)
    ours, config, _ = load_model(path)
    return request.param, hf_model, ours, config


def _our_logits(ours, config, prompt: list[int]) -> torch.Tensor:
    cache = RadixKVCache(num_pages=64, page_size=PAGE)
    pool = PagedKVPool.for_cache(cache, config)
    length = len(prompt)
    page_table = list(range(-(-length // PAGE)))
    hidden = ours.forward_tokens(
        torch.tensor(prompt), torch.arange(length), pool, page_table, seq_len=length
    )
    return ours.logits(hidden)


def test_full_sequence_logits_match_hf(arch):
    name, hf_model, ours, config = arch
    torch.manual_seed(53)
    prompt = torch.randint(0, config.vocab_size, (21,)).tolist()
    theirs = hf_model(torch.tensor([prompt])).logits[0]
    mine = _our_logits(ours, config, prompt)
    diff = (mine - theirs).abs().max().item()
    assert diff < 1e-4, f"{name}: max abs logits diff {diff}"


def test_full_engine_greedy_matches_hf_generate(arch):
    name, hf_model, ours, config = arch
    torch.manual_seed(57)
    prompt = torch.randint(0, config.vocab_size, (13,)).tolist()
    reference = hf_model.generate(
        torch.tensor([prompt]),
        max_new_tokens=16,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=0,
    )[0, len(prompt):].tolist()

    cache = RadixKVCache(num_pages=128, page_size=PAGE)
    scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=PAGE)  # chunked
    pool = PagedKVPool.for_cache(cache, config)
    engine = EngineCore(
        scheduler, PagedModelRunner(ours, pool, sampler=Sampler(), cache=cache)
    )
    engine.add_request(
        EngineRequest(
            "a", tuple(prompt), max_new_tokens=16, sampling=EngineSampling()
        )
    )
    outputs = engine.run_to_completion()["a"]
    assert list(outputs) == reference, f"{name} diverged from hf.generate"


def test_dsv3_yarn_softmax_scale_is_amplified():
    from kairyu.models.layers import mla_softmax_scale

    plain = mla_softmax_scale(12, None)
    from kairyu.models.config import RopeScaling

    yarn = RopeScaling(
        kind="yarn", factor=4.0, mscale=1.0, mscale_all_dim=1.0,
        original_max_position_embeddings=64,
    )
    amplified = mla_softmax_scale(12, yarn)
    import math

    expected = 12**-0.5 * (0.1 * 1.0 * math.log(4.0) + 1.0) ** 2
    assert amplified == pytest.approx(expected)
    assert amplified > plain


def test_dense_layers_and_shared_experts_present(arch):
    name, _, ours, config = arch
    if not name.startswith("dsv3"):
        pytest.skip("deepseek only")
    from kairyu.models.layers import SwiGluMlp
    from kairyu.models.moe import DeepseekV3MoeBlock

    layers = ours.model.layers
    assert isinstance(layers[0].mlp, SwiGluMlp)  # first_k_dense_replace=1
    assert isinstance(layers[1].mlp, DeepseekV3MoeBlock)
    assert layers[1].mlp.shared_experts is not None


def test_mla_pool_shape(arch):
    name, _, _, config = arch
    if not name.startswith("dsv3"):
        pytest.skip("deepseek only")
    cache = RadixKVCache(num_pages=8, page_size=PAGE)
    pool = PagedKVPool.for_cache(cache, config)
    assert pool.num_kv_heads == 1
    assert pool.head_dim == 16 + 4  # kv_lora_rank + qk_rope_head_dim
    assert pool.v_head_dim == 0  # latent-only cache; M18 serde relies on this
