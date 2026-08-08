"""m16 spawn gates: communicator contract, TP=2, EP=2, PP=2 parity (@dist)."""

import pytest
import torch

from tests.dist import dist_targets

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.dist

JSON_VOCAB = ["{", "}", "[", "]", '"', "a", "1", ":", ",", " ", "true", "null", "<eos>"]
TP_VOCAB = JSON_VOCAB + [f"unused-{i}" for i in range(256 - len(JSON_VOCAB))]

TINY_LLAMA = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
TINY_QWEN2 = dict(  # biases exercise the A4 row/column bias rules
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)
TINY_MOE = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY_LLAMA))
    path = tmp_path_factory.mktemp("dist-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="module")
def qwen2_dir(tmp_path_factory):
    torch.manual_seed(73)
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**TINY_QWEN2))
    path = tmp_path_factory.mktemp("dist-qwen2")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="module")
def moe_dir(tmp_path_factory):
    torch.manual_seed(79)
    model = transformers.Qwen3MoeForCausalLM(transformers.Qwen3MoeConfig(**TINY_MOE))
    with torch.no_grad():  # decisive routing margins (m15 fixture lesson)
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3MoeSparseMoeBlock":
                module.gate.weight.mul_(6.0)
    path = tmp_path_factory.mktemp("dist-moe")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _single_process_greedy(model_dir: str, prompt: list[int], max_new: int) -> list[int]:
    return _single_process_sample(model_dir, prompt, max_new, {})


def _single_process_sample(
    model_dir: str,
    prompt: list[int],
    max_new: int,
    sampling_kwargs: dict,
) -> list[int]:
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    model, config, _ = load_model(model_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=4)
    pool = PagedKVPool.for_cache(cache, config)
    engine = EngineCore(scheduler, PagedModelRunner(model, pool, sampler=Sampler()))
    engine.add_request(
        EngineRequest(
            "a",
            tuple(prompt),
            max_new_tokens=max_new,
            sampling=EngineSampling(**sampling_kwargs),
        )
    )
    return list(engine.run_to_completion()["a"])


def test_communicator_contract_on_gloo(spawn2):
    results = spawn2(dist_targets.comm_contract)
    for result in results:
        assert result["broadcast"] == {"step": 7}
        assert result["reduced"] == [3.0, 20.0]  # (1+2, 10+10)
        assert result["gathered"] == ["r0", "r1"]
        assert result["token_packet"] == [17, -1, 29]
    assert results[1]["received"] == {"hello": 0}
    # rank0 keeps its row 0 + gets rank1's first 2; rank1 gets rank0's rows 1..3 + own tail
    assert results[0]["a2a"] == [0.0, 100.0, 101.0]
    assert results[1]["a2a"] == [1.0, 2.0, 3.0, 102.0, 103.0]


@pytest.mark.parametrize("fixture_name", ["llama_dir", "qwen2_dir"])
def test_tp2_engine_greedy_matches_single_process(spawn2, fixture_name, request):
    model_dir = request.getfixturevalue(fixture_name)
    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    reference = _single_process_greedy(model_dir, prompt, max_new=12)
    results = spawn2(dist_targets.tp_engine_parity, model_dir, prompt, 12, TP_VOCAB)
    assert results[0]["outputs"] == reference
    assert results[1]["steps"] > 0  # the worker actually executed the steps


@pytest.mark.parametrize(
    "sampling_kwargs",
    [
        {"temperature": 0.8, "top_k": 32, "top_p": 0.9, "min_p": 0.01, "seed": 17},
        {
            "temperature": 0.7,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.1,
            "logprobs": 3,
        },
    ],
    ids=("explicit-seed-filtered", "request-seed-penalties-logprobs"),
)
def test_tp2_rank0_sampling_matrix_matches_single_process(
    spawn2, llama_dir, sampling_kwargs
):
    torch.manual_seed(97)
    prompt = torch.randint(0, 256, (13,)).tolist()
    reference = _single_process_sample(
        llama_dir, prompt, max_new=10, sampling_kwargs=sampling_kwargs
    )

    first = spawn2(
        dist_targets.tp_engine_parity,
        llama_dir,
        prompt,
        10,
        TP_VOCAB,
        sampling_kwargs,
    )
    replay = spawn2(
        dist_targets.tp_engine_parity,
        llama_dir,
        prompt,
        10,
        TP_VOCAB,
        sampling_kwargs,
    )

    assert first[0]["outputs"] == reference
    assert replay[0]["outputs"] == reference
    assert first[0]["outputs"] == replay[0]["outputs"]
    assert first[1]["steps"] > 0


def test_tp_structured_sampling_and_release_on_every_rank(spawn2, llama_dir):
    results = spawn2(dist_targets.tp_structured_release, llama_dir, TP_VOCAB)
    for rank, result in enumerate(results):
        assert "error" not in result, f"rank {rank} failed: {result.get('error')}"
    assert results[0]["structured_completed"] is True
    assert results[0]["sampling_owner"] is True
    assert results[1]["sampling_owner"] is False
    assert results[0]["sampler_states"] == 0
    assert results[1]["sampler_states"] == 0
    assert results[0]["released_requests"] == 34
    assert results[1]["released_requests"] == 34


def test_dist_tp_launcher_serve_path_matches_single_process(llama_dir):
    # Deploy wiring: DistTPLauncher (rank 0 here + spawned worker) is what
    # `kairyu serve --tp 2` uses. Driving EngineCore through it must reproduce the
    # single-process greedy output, and shutdown() must stop the worker cleanly.
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.worker import DistTPLauncher

    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    reference = _single_process_greedy(llama_dir, prompt, max_new=12)

    # force_cpu: the reference is a single-process fp32 HOST run, so on a GPU
    # box the launcher must not follow the probe onto bf16 devices
    launcher = DistTPLauncher(
        llama_dir, tp=2, num_pages=64, page_size=4, vocab=TP_VOCAB, force_cpu=True
    )
    try:
        scheduler = Scheduler(
            RadixKVCache(num_pages=64, page_size=4), max_num_batched_tokens=6, page_size=4
        )
        engine = EngineCore(scheduler, launcher.runner)
        engine.add_request(
            EngineRequest("a", tuple(prompt), max_new_tokens=12, sampling=EngineSampling())
        )
        outputs = list(engine.run_to_completion()["a"])
    finally:
        launcher.shutdown()
    assert outputs == reference


def test_ep2_moe_block_matches_single_process(spawn2, moe_dir):
    results = spawn2(dist_targets.ep_block_parity, moe_dir)
    for result in results:
        assert result["local_experts"] == 4  # 8 experts / 2 ranks
        assert result["maxdiff"] < 1e-5  # accumulation-order tolerance (A7)


def test_ep2_bf16_combine_is_exactly_degree_invariant(spawn2):
    results = spawn2(dist_targets.ep_block_bf16_invariance)
    for result in results:
        assert result == {
            "exact": True,
            "actual": 0.455078125,
            "reference": 0.455078125,
            "dtype": "torch.bfloat16",
        }


def test_pp2_greedy_matches_single_process(spawn2, llama_dir):
    torch.manual_seed(89)
    prompt = torch.randint(0, 256, (9,)).tolist()
    reference = _single_process_greedy(llama_dir, prompt, max_new=10)
    results = spawn2(dist_targets.pp_greedy_parity, llama_dir, prompt, 10)
    assert results[0]["outputs"] == reference
    assert results[1]["outputs"] == reference


def test_serving_groups_separate_idle_control_and_fail_fast_model_timeouts(tmp_path):
    """Review [P1] on #124: `init_process_group(timeout=)` bounds EVERY operation
    on that group, so the cold-load allowance must not become the operational one.

    Single process on purpose — this is a property of how the groups are built,
    and a spawn adds teardown-ordering noise without testing anything more."""
    import torch.distributed as dist

    from kairyu.engine.core import worker as worker_module

    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'rdv-timeout'}",
        rank=0,
        world_size=1,
        timeout=__import__("datetime").timedelta(
            seconds=worker_module._STARTUP_TIMEOUT_S
        ),
    )
    try:
        groups = worker_module.serving_groups("gloo")
        host = torch.device("cpu")
        control = (
            groups.control._get_backend(host).options._timeout.total_seconds()
        )
        readiness = (
            groups.startup_control
            ._get_backend(host)
            .options._timeout.total_seconds()
        )
        model = groups.model._get_backend(host).options._timeout.total_seconds()
        startup = (
            dist.distributed_c10d._get_default_group()
            ._get_backend(host)
            .options._timeout.total_seconds()
        )
        assert startup == worker_module._STARTUP_TIMEOUT_S == 1800.0
        assert model == worker_module._SERVE_OP_TIMEOUT_S == 120.0
        assert readiness == worker_module._SERVE_OP_TIMEOUT_S
        assert control == worker_module._CONTROL_IDLE_TIMEOUT_S
        assert len({groups.control, groups.startup_control, groups.model}) == 3
        assert model == readiness < startup < control
    finally:
        dist.destroy_process_group()


def test_idle_control_receive_outlives_the_model_timeout(spawn2):
    results = spawn2(dist_targets.control_idle_wait)
    for result in results:
        assert result["payload"] == {"after_idle": True}
        assert result["idle_s"] > result["model_timeout_s"]



def test_reduce_scatter_matches_all_reduce_sliced(spawn2):
    # the primitive m16 D1 recorded as missing (gloo has none). Under gloo it is
    # all_reduce + a local slice; this pins the contract the NCCL path honours.
    results = spawn2(dist_targets.reduce_scatter_equivalence)
    for rank, result in enumerate(results):
        assert result["shape"] == [4, 3], rank
        assert result["max_error"] == 0.0, rank
        assert result["regathered_matches"] is True, rank


def test_reduce_scatter_rejects_an_indivisible_first_dimension(spawn2):
    results = spawn2(dist_targets.reduce_scatter_rejects_indivisible)
    for result in results:
        assert result["raised"] is True
        assert "divisible" in result["message"]


def test_sequence_parallel_matches_plain_tp(spawn2, llama_dir):
    # Megatron TP+SP: the residual stream between blocks is sharded along TOKENS
    # and the norms run on the shard. Norms are per-token, so this is the SAME
    # arithmetic — the gain is activation memory, not latency (rs+ag moves what
    # one all_reduce does; bench/reduce_scatter_bench.py measures it at ~0.96x).
    results = spawn2(dist_targets.sequence_parallel_parity, llama_dir)
    for rank, result in enumerate(results):
        assert result["max_error"] < 1e-4, f"rank {rank}: {result['max_error']}"
        assert result["argmax_equal"] is True, rank


def test_sequence_parallel_handles_a_ragged_token_count(spawn2, llama_dir):
    """11 tokens across 2 ranks: padded to shard, trimmed on the way out."""
    results = spawn2(dist_targets.sequence_parallel_ragged, llama_dir)
    for rank, result in enumerate(results):
        assert result["rows"] == result["expected_rows"], rank
        assert result["max_error"] < 1e-4, f"rank {rank}: {result['max_error']}"


class _AlwaysWrongDraft:
    """Proposes a token the model will never pick, so every draft is REJECTED.

    The n-gram source proposes nothing on a random-token fixture (verified: 0
    proposals), so a parity test using it never enters the rollback path — which
    is exactly where the delta protocol was broken.
    """

    def __init__(self, token: int = 200) -> None:
        self.token = token

    def propose(self, context, max_draft: int) -> list[int]:
        return [self.token] * min(1, max_draft)


class _EchoDraft:
    """Proposes what the reference actually produced, so drafts are ACCEPTED."""

    def __init__(self, reference: list[int]) -> None:
        self._reference = list(reference)

    def propose(self, context, max_draft: int) -> list[int]:
        # context is prompt + committed outputs; the next reference token is the
        # one at the current output length
        produced = len(context) - self._prompt_len
        remaining = self._reference[produced : produced + max_draft]
        return list(remaining)

    def bind_prompt(self, prompt_len: int) -> "_EchoDraft":
        self._prompt_len = prompt_len
        return self


def _tp_speculative_outputs(llama_dir, prompt, max_new, draft_source):
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.spec_runner import SpeculativeRunner
    from kairyu.engine.core.worker import DistTPLauncher

    launcher = DistTPLauncher(
        llama_dir, tp=2, num_pages=64, page_size=4, vocab=TP_VOCAB, force_cpu=True
    )
    try:
        scheduler = Scheduler(
            RadixKVCache(num_pages=64, page_size=4),
            max_num_batched_tokens=6,
            page_size=4,
            speculative_tokens=4,
        )
        runner = SpeculativeRunner(launcher.runner, draft_source=draft_source)
        engine = EngineCore(scheduler, runner)
        engine.add_request(
            EngineRequest(
                "a", tuple(prompt), max_new_tokens=max_new, sampling=EngineSampling()
            )
        )
        outputs = list(engine.run_to_completion()["a"])
        verification_stats = runner.verification_execution_stats()
    finally:
        launcher.shutdown()
    return outputs, runner, verification_stats


def test_tp_speculative_rejected_draft_does_not_corrupt_rank_state(llama_dir):
    """The rollback path: a rejected draft replaces already-broadcast tokens with
    DIFFERENT ones at the SAME length. StateSync only re-sent a full snapshot when
    the output list got SHORTER, so the delta came out empty and every worker kept
    the rejected token."""
    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    reference = _single_process_greedy(llama_dir, prompt, max_new=12)

    outputs, runner, verification_stats = _tp_speculative_outputs(
        llama_dir, prompt, 12, _AlwaysWrongDraft()
    )
    assert runner.draft_proposed > 0, "the draft path was never entered"
    assert runner.draft_accepted == 0, "the fixture must force rejections"
    assert outputs == reference
    local = [row["stats"] for row in verification_stats]
    assert all(row == local[0] for row in local)
    assert local[0]["model_calls"] == local[0]["batched_groups"] > 0
    assert local[0]["positions"] > local[0]["model_calls"]
    assert local[0]["sequential_positions"] == 0


def test_tp_speculative_accepted_draft_matches_greedy(llama_dir):
    torch.manual_seed(83)
    prompt = torch.randint(0, 256, (11,)).tolist()
    reference = _single_process_greedy(llama_dir, prompt, max_new=12)

    outputs, runner, verification_stats = _tp_speculative_outputs(
        llama_dir, prompt, 12, _EchoDraft(reference).bind_prompt(len(prompt))
    )
    assert runner.draft_proposed > 0, "the draft path was never entered"
    assert runner.draft_accepted > 0, "the fixture must produce acceptances"
    assert outputs == reference
    local = [row["stats"] for row in verification_stats]
    assert all(row == local[0] for row in local)
    assert local[0]["model_calls"] == local[0]["batched_groups"] > 0
    assert local[0]["positions"] > local[0]["model_calls"]
    assert local[0]["sequential_positions"] == 0


class _TinyTokenizer:
    """256 ids, matching the fixture's vocab_size (the toy tokenizer's 50k would
    trip the vocab-agreement guard before the wiring is reached)."""

    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(ch) % 256 for ch in text)

    def decode(self, token_ids) -> str:
        return "".join(chr(int(t) % 128) for t in token_ids)

    def vocab(self) -> list[str]:
        return [f"t{i}" for i in range(256)]


def test_build_engine_loop_accepts_speculative_with_real_model_tp(llama_dir, monkeypatch):
    from kairyu.engine.core.spec_runner import SpeculativeRunner
    from kairyu.engine.kairyu_backend import build_engine_loop

    # rank 0 AND the spawned workers must agree on the backend; the env var is
    # the only seam both see, so this runs on a GPU box without a second device
    monkeypatch.setenv("KAIRYU_TP_FORCE_CPU", "1")

    loop, _cache, scheduler = build_engine_loop(
        model_path=llama_dir,
        tensor_parallel_size=2,
        num_pages=64,
        page_size=4,
        max_num_batched_tokens=6,
        tokenizer=_TinyTokenizer(),
        speculative="ngram",
        speculative_tokens=4,
    )
    try:
        assert isinstance(loop._runner, SpeculativeRunner)
        # the scheduler must reserve for the draft, or every speculative chunk
        # is silently downgraded to a single token
        assert scheduler._spec_k == 4
    finally:
        loop.tp_launcher.shutdown()


@pytest.mark.asyncio
async def test_process_backend_owns_real_model_tp2_service_tree(
    llama_dir,
    monkeypatch,
    tmp_path,
):
    """The process split must serve through and clean up real TP workers."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from kairyu.engine.backend import GenerationRequest
    from kairyu.engine.zmq_backend import (
        ZmqEngineBackend,
        _live_process_group_members,
        _process_group_member_states,
    )
    from kairyu.sampling_params import SamplingParams

    monkeypatch.setenv("KAIRYU_TP_FORCE_CPU", "1")
    tokenizer_path = tmp_path / "tokenizer.json"
    Tokenizer(
        WordLevel(
            {token: index for index, token in enumerate(TP_VOCAB)},
            unk_token=TP_VOCAB[-1],
        )
    ).save(str(tokenizer_path))
    backend = ZmqEngineBackend(
        model_path=llama_dir,
        tokenizer=str(tokenizer_path),
        tensor_parallel_size=2,
        num_pages=64,
        page_size=4,
        max_num_batched_tokens=6,
        max_num_seqs=4,
        generation_config="none",
        death_timeout_s=30.0,
    )
    process = None
    try:
        await backend.startup()
        process = backend._process
        assert process is not None and process.is_alive()
        assert process.daemon is False
        assert backend.tensor_parallel_size == 2
        members = _live_process_group_members(process)
        assert process.pid in members
        assert len(members) >= 2, "real TP follower is missing from the owned group"

        updates = [
            update
            async for update in backend.stream(
                GenerationRequest(
                    request_id="real-tp2-process-stream",
                    prompt="{",
                    sampling_params=SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                    ),
                )
            )
        ]
        assert len(updates) == 1
        result = updates[0]
        assert result.finished is True
        assert result.usage.completion_tokens == 1
        assert len(result.completions) == 1
        assert len(result.completions[0].token_ids) == 1
        assert result.completions[0].finish_reason == "length"
    finally:
        await backend.shutdown()

    assert process is not None and process.exitcode == 0
    assert _live_process_group_members(process) == ()
    assert _process_group_member_states(process) == ()


def test_in_process_tp_still_rejects_speculative():
    # the toy path shares ONE runner across ranks: drafting there would score
    # against itself, so the guard stays for that case
    from kairyu.engine.kairyu_backend import build_engine_loop

    with pytest.raises(ValueError, match="requires a real model"):
        build_engine_loop(tensor_parallel_size=2, speculative="ngram")


def test_failed_launcher_start_leaves_no_group_or_workers(llama_dir):
    # A constructor that raises after mp.spawn used to leave the workers running
    # and the process group initialized: the caller got the real error, then a
    # "destroy_process_group() was not called" warning, and the NEXT launcher in
    # the same process could not rendezvous at all.
    import torch.distributed as dist

    from kairyu.engine.core.worker import DistTPLauncher

    assert not dist.is_initialized()
    # TINY_LLAMA has 2 kv heads, so tp=4 fails inside build_tp_runner -> tp_view,
    # i.e. after the spawn and after the group is up
    with pytest.raises(ValueError, match=r"num_(?:kv|key_value)_heads"):
        DistTPLauncher(
            llama_dir, tp=4, num_pages=64, page_size=4, vocab=TP_VOCAB, force_cpu=True
        )
    assert not dist.is_initialized(), "process group survived a failed start"

    # and the process is still usable: a good launcher can rendezvous afterwards
    launcher = DistTPLauncher(
        llama_dir, tp=2, num_pages=64, page_size=4, vocab=TP_VOCAB, force_cpu=True
    )
    launcher.shutdown()
    assert not dist.is_initialized()
