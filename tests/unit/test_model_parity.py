"""m12 parity gates: DenseDecoder vs transformers on tiny configs.

Session-scoped fixtures build + save each arch ONCE (reviewed budget rule);
the HF oracle and our loaded decoder are read-only across tests.
"""

import pytest
import torch

from kairyu.engine.core.kv_pool import PagedKVPool
from kairyu.models.config import parse_model_config
from kairyu.models.llama import DenseDecoder

transformers = pytest.importorskip("transformers")

PAGE = 4
TINY = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=512,
)


def _build(arch: str, **overrides):
    torch.manual_seed(7)
    if arch == "llama":
        config = transformers.LlamaConfig(**TINY, **overrides)
        model = transformers.LlamaForCausalLM(config)
    elif arch == "qwen2":
        config = transformers.Qwen2Config(**TINY, **overrides)
        model = transformers.Qwen2ForCausalLM(config)
    else:
        config = transformers.Qwen3Config(**TINY, head_dim=32, **overrides)
        model = transformers.Qwen3ForCausalLM(config)
    model = model.to(torch.float32).eval()
    return model


@pytest.fixture(scope="module", params=["llama", "qwen2", "qwen3"])
def arch_pair(request, tmp_path_factory):
    hf_model = _build(request.param)
    path = tmp_path_factory.mktemp(f"m-{request.param}")
    hf_model.save_pretrained(path, safe_serialization=True)
    import json

    config = parse_model_config(json.loads((path / "config.json").read_text()))
    ours = DenseDecoder(config).eval()
    missing, unexpected = ours.load_state_dict(hf_model.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all("rotary" in name or "lm_head" in name for name in missing), missing
    return request.param, hf_model, ours, config, path


def _our_full_logits(ours: DenseDecoder, config, prompt: list[int]) -> torch.Tensor:
    pool = PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_pages=64,
        page_size=PAGE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    length = len(prompt)
    page_table = list(range(-(-length // PAGE)))
    hidden = ours.forward_tokens(
        torch.tensor(prompt),
        torch.arange(length),
        pool,
        page_table,
        seq_len=length,
    )
    return ours.logits(hidden)


def test_full_sequence_logits_match_hf(arch_pair):
    arch, hf_model, ours, config, _ = arch_pair
    torch.manual_seed(3)
    prompt = torch.randint(0, config.vocab_size, (23,)).tolist()
    theirs = hf_model(torch.tensor([prompt])).logits[0]
    mine = _our_full_logits(ours, config, prompt)
    diff = (mine - theirs).abs().max().item()
    assert diff < 1e-4, f"{arch}: max abs logits diff {diff}"


def test_chunked_prefill_matches_full_forward(arch_pair):
    arch, _, ours, config, _ = arch_pair
    torch.manual_seed(5)
    prompt = torch.randint(0, config.vocab_size, (17,)).tolist()
    full = _our_full_logits(ours, config, prompt)

    pool = PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_pages=64,
        page_size=PAGE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    page_table = list(range(-(-len(prompt) // PAGE)))
    chunks = [(0, 7), (7, 14), (14, 17)]
    last_hidden = None
    for start, end in chunks:
        last_hidden = ours.forward_tokens(
            torch.tensor(prompt[start:end]),
            torch.arange(start, end),
            pool,
            page_table,
            seq_len=end,
        )
    chunked_logits = ours.logits(last_hidden)
    diff = (chunked_logits[-1] - full[-1]).abs().max().item()
    assert diff < 1e-5, f"{arch}: chunked vs full diff {diff}"


def test_llama3_rope_scaling_parses_and_matches():
    scaling = {
        "rope_type": "llama3",
        "factor": 2.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 128,
    }
    hf_model = _build("llama", rope_parameters={"rope_theta": 50000.0, **scaling})
    import json

    config_dict = json.loads(hf_model.config.to_json_string())
    # architectures is stamped by save_pretrained, not to_json_string
    config_dict["architectures"] = ["LlamaForCausalLM"]
    config = parse_model_config(config_dict)
    assert config.rope_theta == 50000.0  # nested rope_theta parsed (CRITICAL amendment)
    assert config.rope_scaling is not None
    ours = DenseDecoder(config).eval()
    ours.load_state_dict(hf_model.state_dict(), strict=False)
    torch.manual_seed(2)
    prompt = torch.randint(0, config.vocab_size, (140,)).tolist()  # beyond original max
    theirs = hf_model(torch.tensor([prompt])).logits[0]
    mine = _our_full_logits(ours, config, prompt)
    assert (mine - theirs).abs().max().item() < 1e-4


def test_write_skip_leaves_cached_kv_untouched(arch_pair):
    arch, _, ours, config, _ = arch_pair
    torch.manual_seed(9)
    prompt = torch.randint(0, config.vocab_size, (12,)).tolist()
    pool = PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_pages=64,
        page_size=PAGE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    page_table = [0, 1, 2]
    ours.forward_tokens(
        torch.tensor(prompt), torch.arange(12), pool, page_table, seq_len=12
    )
    snapshot = pool.k.clone()
    # recompute positions 8..12 with write_from=12: no pool mutation allowed
    ours.forward_tokens(
        torch.tensor(prompt[8:]),
        torch.arange(8, 12),
        pool,
        page_table,
        seq_len=12,
        write_from=12,
    )
    assert torch.equal(pool.k, snapshot)
