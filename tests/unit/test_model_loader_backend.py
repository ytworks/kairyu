"""m12 D5: loader round-trip + KairyuBackend(model_path=) end to end."""

import json

import pytest
import torch

from kairyu import SamplingParams
from kairyu.engine.backend import GenerationRequest
from kairyu.engine.kairyu_backend import KairyuBackend, build_engine_loop
from kairyu.models.loader import load_model

transformers = pytest.importorskip("transformers")

TINY = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=512,
    tie_word_embeddings=True,  # exercises the omitted-lm_head path
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(31)
    hf_model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    hf_model = hf_model.to(torch.float32).eval()
    path = tmp_path_factory.mktemp("ckpt")
    hf_model.save_pretrained(path, safe_serialization=True)
    (path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [7, 9]})  # Llama-3-style eos LIST
    )
    return path, hf_model


class _SmallVocabTokenizer:
    """256-vocab test tokenizer for the real model."""

    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple((3 + 7 * i + ord(ch)) % 250 for i, ch in enumerate(text)) or (1,)

    def decode(self, token_ids) -> str:
        return " ".join(f"t{t}" for t in token_ids)

    def vocab(self) -> list[str]:
        return [f"t{i}" for i in range(256)]


def test_loader_roundtrip_matches_source_model(checkpoint):
    path, hf_model = checkpoint
    model, config, generation = load_model(path)
    assert config.tie_word_embeddings is True
    assert generation.eos_token_id == 7
    assert generation.stop_token_ids == (9,)
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 9))
    theirs = hf_model(ids).logits[0]
    from kairyu.engine.core.kv_pool import PagedKVPool

    pool = PagedKVPool(2, 16, 4, 2, 16)
    hidden = model.forward_tokens(
        ids[0], torch.arange(9), pool, [0, 1, 2], seq_len=9
    )
    mine = model.logits(hidden)
    assert (mine - theirs).abs().max().item() < 1e-4


def test_quantized_config_with_unquantized_tensors_fails_loudly(checkpoint, tmp_path):
    # m14: quantized configs now LOAD — but a checkpoint whose tensors don't
    # match the declared scheme (missing weight_scale) must fail fast
    path, _ = checkpoint
    import shutil

    quantized = tmp_path / "q"
    shutil.copytree(path, quantized)
    config = json.loads((quantized / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "fp8"}
    (quantized / "config.json").write_text(json.dumps(config))
    with pytest.raises(KeyError, match="weight_scale"):
        load_model(quantized)


async def test_backend_model_path_generates(checkpoint):
    path, _ = checkpoint
    backend = KairyuBackend(
        num_pages=256, page_size=4, model_path=str(path), tokenizer=_SmallVocabTokenizer()
    )
    result = await backend.generate(
        GenerationRequest(
            request_id="r1",
            prompt="hello real model",
            sampling_params=SamplingParams(max_tokens=6, temperature=0.0),
        )
    )
    completion = result.completions[0]
    assert len(completion.token_ids) <= 6  # may stop early on the config eos list
    assert result.usage is not None and result.usage.prompt_tokens > 0
    # deterministic: same request reproduces exactly
    again = await backend.generate(
        GenerationRequest(
            request_id="r2",
            prompt="hello real model",
            sampling_params=SamplingParams(max_tokens=6, temperature=0.0),
        )
    )
    assert again.completions[0].token_ids == completion.token_ids


def test_model_path_mutual_exclusions(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_engine_loop(model_path=str(path), runner=object())
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        build_engine_loop(model_path=str(path), tensor_parallel_size=2)


def test_oversized_tokenizer_vocab_fails_fast(checkpoint):
    path, _ = checkpoint

    class _Huge(_SmallVocabTokenizer):
        def vocab(self) -> list[str]:
            return [f"t{i}" for i in range(50_000)]

    with pytest.raises(ValueError, match="vocab"):
        build_engine_loop(model_path=str(path), tokenizer=_Huge())
