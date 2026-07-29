"""Blocking real-NCCL gate for rank-0-authoritative TP sampling (#225)."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

WORLD_SIZE = 2
SPAWN_TIMEOUT_S = 240
NUM_PAGES = 96
PAGE_SIZE = 8
VOCAB_SIZE = 256
VOCAB = [f"token-{index}" for index in range(VOCAB_SIZE)]
TINY = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


def _require_two_gpus() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU host
        pytest.skip("TP sampling ownership needs CUDA")
    if torch.cuda.device_count() < WORLD_SIZE:  # pragma: no cover - small host
        pytest.skip(
            f"TP sampling ownership needs {WORLD_SIZE} GPUs; "
            f"found {torch.cuda.device_count()}"
        )


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory) -> str:
    torch.manual_seed(225)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    path = tmp_path_factory.mktemp("tp-sampling-owner")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _request_specs():
    from kairyu.engine.core.sampling_types import EngineSampling

    # Different prompt/output lengths force mixed prefill and completion states.
    # Two filtered RNG streams plus greedy cover ownership of both deterministic
    # and stateful sampling parameters.
    return (
        (
            "greedy-short",
            (1, 2, 3, 4, 5),
            2,
            EngineSampling(),
        ),
        (
            "seeded-filtered",
            tuple(range(11, 22)),
            7,
            EngineSampling(
                temperature=0.8,
                top_k=32,
                top_p=0.9,
                min_p=0.01,
                seed=9,
            ),
        ),
        (
            "seeded-penalties",
            tuple(range(31, 48)),
            5,
            EngineSampling(
                temperature=0.65,
                top_k=48,
                top_p=0.95,
                presence_penalty=0.3,
                frequency_penalty=0.2,
                repetition_penalty=1.1,
                seed=71,
                logprobs=2,
            ),
        ),
    )


def _drive(runner, cache) -> dict[str, tuple[int, ...]]:
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        page_size=PAGE_SIZE,
    )
    engine = EngineCore(scheduler, runner)
    for request_id, prompt, max_new, sampling in _request_specs():
        engine.add_request(
            EngineRequest(
                request_id,
                prompt,
                max_new_tokens=max_new,
                ignore_eos=True,
                sampling=sampling,
            )
        )
    return engine.run_to_completion()


def _tp1(model_dir: str, mode: str):
    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.models.loader import load_model

    torch.cuda.set_device(0)
    model, config, _ = load_model(
        model_dir,
        dtype=torch.bfloat16,
        attention_backend=select_backend(probe()),
        target_device="cuda:0",
    )
    model = model.to("cuda:0")
    cache = RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE)
    pool = PagedKVPool.for_cache(
        cache, config, dtype=torch.bfloat16, device="cuda:0"
    )
    graph_options = {}
    if mode == "cuda_graph":
        graph_options = {
            "graph_backend": CudaGraphBackend(warmup_iters=1),
            "graph_max_batch": 8,
            "graph_max_pages": 8,
        }
    runner = PagedModelRunner(
        model,
        pool,
        sampler=Sampler(),
        cache=cache,
        **graph_options,
    )
    try:
        outputs = _drive(runner, cache)
        stats = None
        if runner._graph is not None:
            backend = runner._graph._backend
            stats = {"captures": backend.captures, "replays": backend.replays}
        return outputs, stats
    finally:
        runner.invalidate_graphs()
        torch.cuda.synchronize(0)
        del runner, pool, model
        gc.collect()
        torch.cuda.empty_cache()


def _tp2(model_dir: str, mode: str):
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.worker import DistTPLauncher

    cache = RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE)
    scratch_page = cache.reserve_scratch_page() if mode == "cuda_graph" else None
    launcher = DistTPLauncher(
        model_dir,
        tp=WORLD_SIZE,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        vocab=VOCAB,
        graph_scratch_page=scratch_page,
        graph_max_batch=8,
        graph_max_pages=8,
        graph_warmup_iters=1,
    )
    try:
        local = launcher.runner._local
        assert local.sampling_owner is True
        assert local.sampler is not None
        outputs = _drive(launcher.runner, cache)
        assert launcher.runner.sampling_ownership_metadata() == (
            {
                "rank": 0,
                "control_world_size": WORLD_SIZE,
                "control_backend": "gloo",
                "model_world_size": WORLD_SIZE,
                "model_backend": "nccl",
                "sampling_owner": True,
                "sampler_present": True,
                "device": "cuda:0",
            },
            {
                "rank": 1,
                "control_world_size": WORLD_SIZE,
                "control_backend": "gloo",
                "model_world_size": WORLD_SIZE,
                "model_backend": "nccl",
                "sampling_owner": False,
                "sampler_present": False,
                "device": "cuda:1",
            },
        )
        prefill_rows = launcher.runner.prefill_execution_stats()
        assert [row["rank"] for row in prefill_rows] == [0, 1]
        assert [row["device"] for row in prefill_rows] == [
            "cuda:0",
            "cuda:1",
        ]
        for row in prefill_rows:
            structural = row["stats"]
            assert structural["rows"] > 0
            assert structural["model_calls"] == structural["rows"]
            assert structural["sequential_rows"] == structural["rows"]
            assert structural["batched_groups"] == 0
            assert "TorchAttentionBackend" in structural["capability_gap"]
        stats = None
        if local._graph is not None:
            backend = local._graph._backend
            stats = {"captures": backend.captures, "replays": backend.replays}
        return outputs, stats
    finally:
        launcher.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("mode", ["eager", "cuda_graph"])
def test_tp2_rank0_sampling_matches_tp1_for_seeded_mixed_requests(
    llama_dir: str, monkeypatch, mode: str
) -> None:
    _require_two_gpus()
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")

    reference, tp1_stats = _tp1(llama_dir, mode)
    candidate, tp2_stats = _tp2(llama_dir, mode)

    assert candidate == reference, {
        request_id: {
            "tp1": list(reference[request_id]),
            "tp2": list(candidate[request_id]),
        }
        for request_id in reference
        if reference[request_id] != candidate[request_id]
    }
    assert {key: len(value) for key, value in candidate.items()} == {
        "greedy-short": 2,
        "seeded-filtered": 7,
        "seeded-penalties": 5,
    }
    if mode == "cuda_graph":
        assert tp1_stats is not None and tp2_stats is not None
        assert tp1_stats["captures"] > 0
        assert tp1_stats["replays"] > tp1_stats["captures"]
        assert tp2_stats["captures"] > 0
        assert tp2_stats["replays"] > tp2_stats["captures"]
    else:
        assert tp1_stats is None and tp2_stats is None


class _InjectedFollower:
    """Inject a divergent future token immediately before every adoption."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.injected = 0
        self.overwritten = 0
        self.next_decode_checks = 0
        self.authoritative: dict[tuple[str, int], int] = {}

    def execute_passive(self, scheduled, states):
        # This is the exact dependency the real execute_passive call is about to
        # consume. It must still be rank 0's packet from the preceding step.
        for chunk in scheduled:
            if chunk.is_prefill or chunk.position == 0:
                continue
            key = (chunk.request_id, chunk.position - 1)
            if key not in self.authoritative:
                raise AssertionError(f"no adopted token available for next decode {key}")
            observed = self.runner._previous_token(
                states[chunk.request_id], chunk.position
            )
            observed_id = int(observed.detach().cpu()) if isinstance(
                observed, torch.Tensor
            ) else int(observed)
            expected_id = self.authoritative[key]
            if observed_id != expected_id:
                raise AssertionError(
                    f"rank advanced with divergent token for {key}: "
                    f"{observed_id} != authoritative {expected_id}"
                )
            self.next_decode_checks += 1

        result = self.runner.execute_passive(scheduled, states)
        # Simulate exactly the old failure mode after model/KV work and before
        # the authoritative tensor collective. Adoption must replace every one.
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not self.runner._chunk_emits_token(chunk, state):
                continue
            wrong = torch.tensor(
                VOCAB_SIZE + 10_000 + self.injected,
                dtype=torch.int64,
                device=self.runner._device,
            )
            self.runner._remember_device(chunk.request_id, chunk.position, wrong)
            self.injected += 1
        return result

    def make_sampling_token_packet(self, scheduled, states, sampled=None):
        return self.runner.make_sampling_token_packet(
            scheduled, states, sampled=sampled
        )

    def adopt_sampling_token_packet(self, scheduled, states, packet) -> None:
        before: dict[tuple[str, int], int] = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not self.runner._chunk_emits_token(chunk, state):
                continue
            value = self.runner._future_device_tokens[chunk.request_id][
                chunk.position
            ]
            before[(chunk.request_id, chunk.position)] = int(value.detach().cpu())

        self.runner.adopt_sampling_token_packet(scheduled, states, packet)
        for index, chunk in enumerate(scheduled):
            state = states[chunk.request_id]
            if not self.runner._chunk_emits_token(chunk, state):
                continue
            key = (chunk.request_id, chunk.position)
            adopted = self.runner._future_device_tokens[chunk.request_id][
                chunk.position
            ]
            adopted_id = int(adopted.detach().cpu())
            packet_id = int(packet[index].detach().cpu())
            if before[key] == packet_id or adopted_id != packet_id:
                raise AssertionError(
                    f"authoritative adoption failed for {key}: "
                    f"before={before[key]}, packet={packet_id}, after={adopted_id}"
                )
            self.authoritative[key] = packet_id
            self.overwritten += 1

    def release(self, request_id: str) -> None:
        self.runner.release(request_id)

    def invalidate_graphs(self) -> None:
        self.runner.invalidate_graphs()


class _FailAfterRank0Model:
    """Raise after rank 0 has finished model/sampling, before packet broadcast."""

    def __init__(self, runner) -> None:
        self.runner = runner

    def execute(self, scheduled, states):
        self.runner.execute(scheduled, states)
        raise RuntimeError("injected post-model rank-0 sampling-owner failure")

    def __getattr__(self, name: str):
        return getattr(self.runner, name)


def _probe_requests(runner) -> dict[str, tuple[int, ...]]:
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    scheduler = Scheduler(
        RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE),
        max_num_batched_tokens=12,
        max_num_seqs=4,
        page_size=PAGE_SIZE,
    )
    engine = EngineCore(scheduler, runner)
    engine.add_request(
        EngineRequest(
            "probe-short",
            tuple(range(1, 8)),
            max_new_tokens=4,
            ignore_eos=True,
            sampling=EngineSampling(),
        )
    )
    engine.add_request(
        EngineRequest(
            "probe-seeded",
            tuple(range(21, 34)),
            max_new_tokens=6,
            ignore_eos=True,
            sampling=EngineSampling(
                temperature=0.8, top_k=32, top_p=0.9, seed=225
            ),
        )
    )
    return engine.run_to_completion()


def _sampling_owner_probe_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
    model_dir: str,
) -> None:
    import os

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
    from kairyu.engine.core.worker import (
        DistTPModelRunner,
        build_tp_runner,
        serving_groups,
        tp_placement,
        worker_step_loop,
    )

    os.environ["KAIRYU_ATTENTION_BACKEND"] = "torch"
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    init_distributed(
        rank,
        world_size,
        f"file://{init_file}",
        backend="nccl",
        timeout_s=240,
    )
    groups = serving_groups("nccl")
    assert dist.get_backend(groups.control) == "gloo"
    assert dist.get_backend(groups.model) == "nccl"
    control = TorchDistCommunicator(group=groups.control)
    model_comm = TorchDistCommunicator(
        group=groups.model, device=f"cuda:{rank}"
    )
    runner, _ = build_tp_runner(
        model_dir,
        world_size,
        rank,
        model_comm,
        NUM_PAGES,
        PAGE_SIZE,
        VOCAB,
        tp_placement(world_size, rank),
    )
    payload: dict[str, object]
    try:
        if rank == 0:
            distributed = DistTPModelRunner(control, runner, model_comm)
            outputs = _probe_requests(distributed)
            distributed.shutdown()
            payload = {
                "rank": rank,
                "sampling_owner": runner.sampling_owner,
                "sampler_present": runner.sampler is not None,
                "outputs": {
                    request_id: list(tokens)
                    for request_id, tokens in outputs.items()
                },
            }
        else:
            follower = _InjectedFollower(runner)
            steps = worker_step_loop(control, follower, model_comm)
            payload = {
                "rank": rank,
                "sampling_owner": runner.sampling_owner,
                "sampler_present": runner.sampler is not None,
                "steps": steps,
                "injected": follower.injected,
                "overwritten": follower.overwritten,
                "next_decode_checks": follower.next_decode_checks,
            }
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        runner.invalidate_graphs()
        torch.cuda.synchronize(rank)
        model_comm.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group(groups.model)
            dist.destroy_process_group(groups.control)
            dist.destroy_process_group()


def _run_probe(
    model_dir: str, root: Path, launch_index: int
) -> list[dict[str, object]]:
    launch_dir = root / f"launch-{launch_index}"
    launch_dir.mkdir()
    result_dir = launch_dir / "results"
    result_dir.mkdir()
    context = mp.start_processes(
        _sampling_owner_probe_worker,
        args=(
            WORLD_SIZE,
            str(launch_dir / "rdv"),
            str(result_dir),
            model_dir,
        ),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:  # pragma: no cover - hang guard
            for process in context.processes:
                process.terminate()
            pytest.fail(
                "rank-0 sampling ownership probe timed out after "
                f"{SPAWN_TIMEOUT_S}s on launch {launch_index}"
            )
    return [
        json.loads((result_dir / f"rank{rank}.json").read_text())
        for rank in range(WORLD_SIZE)
    ]


def test_follower_divergence_is_overwritten_before_decode_and_groups_relaunch(
    llama_dir: str, tmp_path: Path
) -> None:
    _require_two_gpus()

    first = _run_probe(llama_dir, tmp_path, 0)
    second = _run_probe(llama_dir, tmp_path, 1)

    # An identical second launch proves every NCCL/gloo group and CUDA resource
    # from the first run was torn down rather than poisoning the next process.
    assert first[0]["outputs"] == second[0]["outputs"]
    for launch in (first, second):
        owner, follower = launch
        assert owner["sampling_owner"] is True
        assert owner["sampler_present"] is True
        assert follower["sampling_owner"] is False
        assert follower["sampler_present"] is False
        assert follower["steps"] > 0
        assert follower["injected"] == follower["overwritten"] == 10
        # First token for each request has no preceding decode dependency.
        assert follower["next_decode_checks"] == follower["injected"] - 2


def _post_model_failure_relaunch_worker(
    _rank: int, model_dir: str, result_path: str
) -> None:
    import os

    from kairyu.engine.core.worker import DistTPLauncher

    os.environ["KAIRYU_ATTENTION_BACKEND"] = "torch"
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    first = DistTPLauncher(
        model_dir,
        tp=WORLD_SIZE,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        vocab=VOCAB,
    )
    first.runner._local = _FailAfterRank0Model(first.runner._local)
    error = None
    try:
        _probe_requests(first.runner)
    except RuntimeError as caught:
        error = str(caught)
    shutdown_started = time.monotonic()
    first.shutdown()
    shutdown_s = time.monotonic() - shutdown_started
    del first
    gc.collect()
    torch.cuda.empty_cache()

    second = DistTPLauncher(
        model_dir,
        tp=WORLD_SIZE,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        vocab=VOCAB,
    )
    try:
        outputs = _probe_requests(second.runner)
    finally:
        second.shutdown()
    Path(result_path).write_text(
        json.dumps(
            {
                "error": error,
                "shutdown_s": shutdown_s,
                "outputs": {
                    request_id: list(tokens)
                    for request_id, tokens in outputs.items()
                },
            }
        ),
        encoding="utf-8",
    )


def test_post_model_rank0_failure_aborts_model_group_and_relaunches(
    llama_dir: str, tmp_path: Path
) -> None:
    _require_two_gpus()
    result_path = tmp_path / "post-model-failure.json"
    context = mp.start_processes(
        _post_model_failure_relaunch_worker,
        args=(llama_dir, str(result_path)),
        nprocs=1,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:  # pragma: no cover - hang guard
            for process in context.processes:
                process.terminate()
            pytest.fail(
                "post-model rank-0 failure teardown/relaunch exceeded "
                f"{SPAWN_TIMEOUT_S}s"
            )
    result = json.loads(result_path.read_text())
    assert result["error"] == "injected post-model rank-0 sampling-owner failure"
    assert result["shutdown_s"] < 30
    assert {key: len(value) for key, value in result["outputs"].items()} == {
        "probe-short": 4,
        "probe-seeded": 6,
    }
