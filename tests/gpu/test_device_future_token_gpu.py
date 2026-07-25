"""The decode input slot is patched device-side (m2 §2.2).

`torch.tensor(list_of_ints, device=...)` copies host values the device produced
one step earlier. Carrying the sampled token as a tensor removes that round trip
for every step after the first.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
PROMPTS = [list(range(9)), list(range(4, 15)), list(range(2, 11))]


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device future-token gate needs CUDA")


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("device-future-token")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _generate(model_dir, core_cls, max_new=8):
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir, dtype=torch.bfloat16)
    cache = RadixKVCache(num_pages=128, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    runner = PagedModelRunner(model.to("cuda:0"), pool, sampler=Sampler())
    core = core_cls(scheduler, runner)
    for index, prompt in enumerate(PROMPTS):
        core.add_request(
            EngineRequest(
                f"r{index}", tuple(prompt), max_new_tokens=max_new,
                sampling=EngineSampling(),
            )
        )
    return core.run_to_completion(), runner


def test_the_sampler_hands_back_a_device_tensor():
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling

    _require_cuda()
    logits = torch.tensor([0.1, 0.9, 0.3], device="cuda:0")
    token = Sampler().sample("a", EngineSampling(), 0, logits)

    assert token.device_token is not None
    assert token.device_token.device.type == "cuda"
    assert int(token.device_token.item()) == token.token_id


def test_device_sourced_inputs_produce_the_same_tokens(llama_dir):
    """The whole point is that this changes nothing observable."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    _require_cuda()
    eager, runner = _generate(llama_dir, EngineCore)
    overlapped, _ = _generate(llama_dir, OverlapEngineCore)

    assert overlapped == eager
    assert all(len(tokens) == 8 for tokens in eager.values())
    # and the device tokens really were retained
    assert runner._device_tokens


def test_the_decode_input_comes_from_the_device_when_it_can(llama_dir):
    """Directly: with in-flight device tokens for every row, the input tensor is
    stacked from them rather than copied from the host."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import SampledToken
    from kairyu.engine.core.scheduler import ScheduledChunk
    from kairyu.models.loader import load_model

    _require_cuda()
    model, config, _ = load_model(llama_dir, dtype=torch.bfloat16)
    cache = RadixKVCache(num_pages=32, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    runner = PagedModelRunner(model.to("cuda:0"), pool)

    class _State:
        outputs = ()

    for request_id, value in (("a", 11), ("b", 22)):
        runner._remember(
            request_id, 0,
            SampledToken(value, device_token=torch.tensor(value, device="cuda:0")),
        )

    chunks = [
        ScheduledChunk(request_id="a", num_tokens=1, is_prefill=False, position=1),
        ScheduledChunk(request_id="b", num_tokens=1, is_prefill=False, position=1),
    ]
    states = {"a": _State(), "b": _State()}
    built = runner._decode_token_tensor(chunks, states, [999, 999])

    assert built.device.type == "cuda"
    # the fallback values were deliberately wrong: if they appear, the host copy
    # was taken instead of the device stack
    assert built.tolist() == [11, 22]
