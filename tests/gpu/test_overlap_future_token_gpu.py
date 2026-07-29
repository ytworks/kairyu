"""Overlap ON against a real model on real GPUs (m2 §2.2, G2 A1's second half).

The CPU gate covers the logic; this covers the path an A1 run actually takes —
bf16 weights, the flashinfer kernel, and the KV pool on device.
"""

import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    path = tmp_path_factory.mktemp("gpu-overlap-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _generate(
    model_dir: str,
    core_cls,
    prompts,
    max_new: int,
    sampling=None,
    request_kwargs=None,
):
    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(
        model_dir, dtype=torch.bfloat16, attention_backend=select_backend(probe())
    )
    model = model.to("cuda:0")
    cache = RadixKVCache(num_pages=128, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=16)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    runner = PagedModelRunner(model, pool, sampler=Sampler())
    core = core_cls(scheduler, runner)
    if sampling is None:
        sampling = EngineSampling()
    if request_kwargs is None:
        request_kwargs = {}
    for index, tokens in enumerate(prompts):
        core.add_request(
            EngineRequest(
                f"r{index}", tuple(tokens), max_new_tokens=max_new,
                sampling=sampling,
                **request_kwargs,
            )
        )
    return core.run_to_completion(), runner


def test_overlap_on_matches_overlap_off_on_gpu(llama_dir):
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("overlap-on-GPU gate needs CUDA")

    prompts = [list(range(11)), list(range(5, 20)), list(range(3, 12))]
    off, _ = _generate(llama_dir, EngineCore, prompts, 12)
    on, _ = _generate(llama_dir, OverlapEngineCore, prompts, 12)

    assert on == off, "overlap ON diverged from OFF on device"
    assert all(len(tokens) == 12 for tokens in off.values())


@pytest.mark.parametrize(
    "sampling",
    [
        pytest.param(
            {"temperature": 0.8, "top_k": 32, "top_p": 0.9, "min_p": 0.01, "seed": 9},
            id="seeded-filtered",
        ),
        pytest.param({"presence_penalty": 1.5}, id="presence"),
        pytest.param({"frequency_penalty": 0.8}, id="frequency"),
        pytest.param({"repetition_penalty": 1.4}, id="repetition"),
    ],
)
def test_device_sampling_overlap_matches_eager(llama_dir, sampling):
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore
    from kairyu.engine.core.sampling_types import EngineSampling

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device sampler gate needs CUDA")
    prompts = [list(range(11)), list(range(5, 20)), list(range(3, 12))]
    params = EngineSampling(**sampling)

    eager, _ = _generate(llama_dir, EngineCore, prompts, 10, sampling=params)
    overlap, _ = _generate(
        llama_dir, OverlapEngineCore, prompts, 10, sampling=params
    )

    assert overlap == eager


def test_device_feedback_preserves_eos_stop_and_min_tokens(llama_dir):
    """Speculatively submitted surplus is trimmed at the host commit boundary."""
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.overlap import OverlapEngineCore

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device stop-semantics gate needs CUDA")
    prompts = [list(range(11))]
    first, _ = _generate(llama_dir, EngineCore, prompts, 1)
    stop_id = first["r0"][0]

    for request_kwargs in (
        {"eos_token_id": stop_id},
        {"stop_token_ids": (stop_id,)},
    ):
        eager, _ = _generate(
            llama_dir,
            EngineCore,
            prompts,
            6,
            request_kwargs=request_kwargs,
        )
        overlap, _ = _generate(
            llama_dir,
            OverlapEngineCore,
            prompts,
            6,
            request_kwargs=request_kwargs,
        )
        assert eager["r0"] == (stop_id,)
        assert overlap == eager

    # The first stop-valued token must be retained rather than terminating
    # below min_tokens.  Whatever the model picks next, eager and overlap see
    # the same committed history and finish no earlier than the second token.
    eager, _ = _generate(
        llama_dir,
        EngineCore,
        prompts,
        4,
        request_kwargs={"eos_token_id": stop_id, "min_tokens": 2},
    )
    overlap, _ = _generate(
        llama_dir,
        OverlapEngineCore,
        prompts,
        4,
        request_kwargs={"eos_token_id": stop_id, "min_tokens": 2},
    )
    assert overlap == eager
    assert len(overlap["r0"]) >= 2


def test_seeded_device_sample_is_idempotent_and_reports_raw_logprobs():
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device sampler gate needs CUDA")
    logits = torch.tensor(
        [0.2, -0.1, 1.7, 0.6, -2.0, 0.9],
        dtype=torch.float32,
        device="cuda",
    )
    params = EngineSampling(
        temperature=0.8,
        top_k=5,
        top_p=0.95,
        seed=71,
        logprobs=3,
    )

    first = Sampler().sample_device("a", params, 4, logits)
    replay = Sampler().sample_device("b", params, 4, logits)

    assert first.token_id.device.type == "cuda"
    assert torch.equal(first.token_id, replay.token_id)
    assert first.logprob is not None
    assert first.top_indices is not None
    assert first.top_logprobs is not None
    assert first.top_indices.numel() == 3
    raw = torch.log_softmax(logits, dim=-1)
    assert torch.allclose(
        first.logprob,
        raw.gather(0, first.token_id.view(1)).squeeze(0),
    )
    assert math.isfinite(float(first.logprob.cpu()))


def test_device_penalties_cover_prompt_committed_and_pending_history():
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device penalty gate needs CUDA")
    logits = torch.zeros(16, device="cuda")
    logits[3] = 5.0
    logits[7] = 1.0
    repetition = Sampler().sample_device(
        "repetition",
        EngineSampling(repetition_penalty=100.0),
        0,
        logits,
        prompt=(3,),
    )
    assert int(repetition.token_id.cpu()) == 7

    logits[3] = 1.0
    logits[7] = 0.9
    presence = Sampler().sample_device(
        "presence",
        EngineSampling(presence_penalty=2.0),
        1,
        logits,
        outputs=(),
        pending_outputs=(torch.tensor(3, device="cuda"),),
    )
    assert int(presence.token_id.cpu()) == 7


def test_deferred_public_output_preserves_device_logprobs():
    from kairyu.engine.core.model_runner import (
        _DeferredStepOutput,
        _PendingDeviceToken,
    )
    from kairyu.engine.core.sampler import DeviceSample

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device logprob transfer gate needs CUDA")
    logits = torch.tensor([0.2, 1.7, 0.9, -0.1], device="cuda")
    raw = torch.log_softmax(logits, dim=-1)
    token_id = torch.argmax(logits)
    top_values, top_indices = torch.topk(raw, 3)
    resolved = []
    record = _PendingDeviceToken(
        sample=DeviceSample(
            token_id=token_id,
            logprob=raw.gather(0, token_id.view(1)).squeeze(0),
            top_indices=top_indices,
            top_logprobs=top_values,
        ),
        on_resolve=resolved.append,
    )
    output = _DeferredStepOutput(
        {"r": (record,)},
        copy_stream=torch.cuda.Stream(device=logits.device),
    )
    assert output._copy_sources
    # The TP sampling packet is allocated immediately after this constructor.
    # Stress the caching allocator with the packet's sentinel; the stacked D2H
    # source must remain owned by the deferred output until its event resolves.
    clobber = [torch.full((1,), -1, dtype=torch.int64, device="cuda") for _ in range(64)]

    token = output["r"][0]

    assert clobber[-1].item() == -1
    assert output._copy_sources == ()
    assert resolved == [token]
    assert token.token_id == 1
    assert token.logprob == pytest.approx(float(raw[1].cpu()))
    assert token.top_logprobs is not None
    assert [index for index, _value in token.top_logprobs] == top_indices.cpu().tolist()
    assert [value for _index, value in token.top_logprobs] == pytest.approx(
        top_values.cpu().tolist()
    )


class _FakeEnforcer:
    """CPU matcher stand-in: only even token ids are legal."""

    def mask_logits(self, logits):
        logits[1::2] = float("-inf")
        return logits

    def accept(self, token_id: int) -> bool:
        return token_id % 2 == 0

    def is_terminated(self) -> bool:
        return False


def test_structured_sampling_keeps_the_reviewed_cpu_matcher_path():
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device sampler gate needs CUDA")
    sampler = Sampler()
    params = EngineSampling(temperature=0.0)
    state = sampler._state_for("structured", params)
    state.enforcer = _FakeEnforcer()
    logits = torch.zeros(8, device="cuda")
    logits[3] = 10.0  # illegal winner before the grammar mask
    logits[4] = 9.0

    with pytest.raises(ValueError, match="CPU matcher"):
        sampler.sample_device("structured", params, 0, logits)

    token = sampler.sample("structured", params, 0, logits)
    assert token.token_id == 4


def test_steady_decode_feedback_profiler_has_no_host_scalar_sync(
    llama_dir, monkeypatch
):
    from torch.profiler import ProfilerActivity, profile

    from kairyu.engine.core.engine_core import token_ids
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("device sampler profiler gate needs CUDA")
    model, config, _ = load_model(llama_dir, dtype=torch.bfloat16)
    model = model.to("cuda:0")
    cache = RadixKVCache(num_pages=128, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=64, page_size=16)
    pool = PagedKVPool.for_cache(
        cache, config, dtype=torch.bfloat16, device="cuda:0"
    )
    runner = PagedModelRunner(model, pool, sampler=Sampler())
    scheduler.add_request(
        EngineRequest(
            "r",
            tuple(range(11)),
            max_new_tokens=4,
            sampling=EngineSampling(temperature=0.8, top_p=0.9, seed=17),
        )
    )

    # Resolve prefill outside the measured range.  Its device token remains in
    # the runner's future buffer and is the next decode input.
    prefill = scheduler.schedule()
    first = runner.execute(prefill.scheduled, scheduler.states)
    scheduler.update(token_ids(first))
    decode = scheduler.schedule()

    seen_inputs = []
    original = runner._decode_input_slots

    def record_sources(tokens, positions):
        seen_inputs.extend(tokens)
        return original(tokens, positions)

    runner._decode_input_slots = record_sources
    original_item = torch.Tensor.item

    def forbid_device_item(tensor, *args, **kwargs):
        if tensor.device.type == "cuda":
            raise AssertionError("device Tensor.item() in decode feedback range")
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", forbid_device_item)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        acc_events=True,
    ) as prof:
        deferred = runner.execute(decode.scheduled, scheduler.states)
    monkeypatch.undo()
    runner._decode_input_slots = original

    keys = {event.key for event in prof.key_averages()}
    scalar_events = [
        (
            event.name,
            None if event.cpu_parent is None else event.cpu_parent.name,
            event.stack,
        )
        for event in prof.events()
        if event.name == "aten::_local_scalar_dense"
    ]
    assert not scalar_events, scalar_events
    assert not any("cudaEventSynchronize" in key for key in keys)
    assert seen_inputs and all(isinstance(token, torch.Tensor) for token in seen_inputs)

    # Host EOS/streaming commit is intentionally one step late and outside the
    # feedback range measured above.
    scheduler.update(token_ids(deferred))
